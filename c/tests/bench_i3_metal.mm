// One batched MoE block on the PRODUCTION path, per-row int4 (fmt=2) against int3-g64 (fmt=5),
// on the CPU expert loop and on the Metal kernels, at GLM expert dims. This is the bytes-vs-speed
// question: fmt=5 moves 0.75x the weight bytes of fmt=2, so it only pays if the decode keeps up.
//
// The two formats are timed INTERLEAVED, round by round, and reported as medians. Timing them one
// after the other is not safe on this hardware: the heavier arm heats the GPU and whatever runs
// next measures under lower clocks. Observed while building this, the same fmt=2 kernel reads
// 0.84 ms when it runs first and 1.26 ms when the heavy int3 arm runs before it, which is larger
// than the effect being measured. Interleaving puts both formats in the same thermal window.
//
// The per-call coli_metal_matmul path is not measured on purpose: docs/metal.md says per-matmul
// dispatch is a loss, everything real goes through moe_submit.
#include "../backend_metal.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
extern "C" void ref_i3(float*,const float*,const unsigned char*,const float*,int,int,int);
extern "C" void ref_i4(float*,const float*,const unsigned char*,const float*,int,int,int);
extern "C" double coli_metal_moe_kernel_time(void);
static double now(){ using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count(); }
static size_t roundpg(size_t n){ size_t p=16384; return ((n+p-1)/p)*p; }
static double median(std::vector<double> v){ std::sort(v.begin(),v.end());
  size_t n=v.size(); return n ? (n%2 ? v[n/2] : 0.5*(v[n/2-1]+v[n/2])) : 0.0; }

// bytes per weight row and per scale row, for the two formats this bench compares
static size_t rowbytes(int fmt,int I){ return fmt==5 ? (size_t)((I+63)/64)*24 : (size_t)((I+1)/2); }
static size_t scalerow(int fmt,int I){ return fmt==5 ? (size_t)((I+63)/64) : 1; }

struct Arm {
  int fmt,nb,D,I,R;
  std::vector<void*> slab,fslab;
  std::vector<const void*> g,u,d; std::vector<const float*> gs,us,ds;
  std::vector<int> nr,xoff,rows;
  std::vector<float> xg,rw,gout,cpuout,gg,uu,hh;
  size_t wbytes;

  void setup(int fmt_,int nb_,int D_,int I_){
    fmt=fmt_; nb=nb_; D=D_; I=I_;
    size_t rbG=rowbytes(fmt,D), rbD=rowbytes(fmt,I);   // gate/up: I rows of D; down: D rows of I
    size_t sgG=scalerow(fmt,D), sgD=scalerow(fmt,I);
    nr.assign(nb,1); xoff.resize(nb); R=0;
    for(int e=0;e<nb;e++){ xoff[e]=R; R+=nr[e]; }
    srand(11);
    slab.resize(nb); fslab.resize(nb); g.resize(nb); u.resize(nb); d.resize(nb);
    gs.resize(nb); us.resize(nb); ds.resize(nb);
    size_t wraw=(size_t)I*rbG*2 + (size_t)D*rbD, wlen=roundpg(wraw);
    size_t sraw=((size_t)I*sgG*2 + (size_t)D*sgD)*sizeof(float), flen=roundpg(sraw);
    wbytes=wraw;
    const char* fe=getenv("BENCH_FILL"); int fill=fe?atoi(fe):0;   // 0 random, 1 zeros, 2 ones
    for(int e=0;e<nb;e++){
      posix_memalign(&slab[e],16384,wlen); posix_memalign(&fslab[e],16384,flen);
      uint8_t* sp=(uint8_t*)slab[e];
      for(size_t i=0;i<wraw;i++) sp[i]=(uint8_t)(fill==1?0x00:fill==2?0xFF:(rand()&0xFF));
      float* fp=(float*)fslab[e];
      for(size_t i=0;i<sraw/sizeof(float);i++) fp[i]=0.01f+(rand()%50)/50000.f;
      g[e]=sp; u[e]=sp+(size_t)I*rbG; d[e]=sp+(size_t)I*rbG*2;
      gs[e]=fp; us[e]=fp+(size_t)I*sgG; ds[e]=fp+(size_t)I*sgG*2;
      coli_metal_register(slab[e],wlen); coli_metal_register(fslab[e],flen);
    }
    xg.resize((size_t)R*D); for(auto&v:xg) v=((rand()%2000)-1000)/1000.f;
    rows.assign(R,0); rw.resize(R); for(auto&v:rw) v=0.1f+(rand()%100)/100.f;
    gout.assign((size_t)D,0.f); cpuout.assign((size_t)D,0.f);
    gg.resize(I); uu.resize(I); hh.resize(D);
  }

  // the engine's own per-expert loop: gate/up matmul, silu, down matmul, weighted add
  void cpu_block(){
    auto mm=[&](float* y,const float* x,const void* w,const float* s,int Iin,int Oout){
      if(fmt==5) ref_i3(y,x,(const unsigned char*)w,s,1,Iin,Oout);
      else       ref_i4(y,x,(const unsigned char*)w,s,1,Iin,Oout); };
    std::fill(cpuout.begin(),cpuout.end(),0.f);
    for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++){
      const float* xr=&xg[(size_t)(xoff[e]+r)*D];
      mm(gg.data(),xr,g[e],gs[e],D,I);
      mm(uu.data(),xr,u[e],us[e],D,I);
      for(int o=0;o<I;o++){ float v=gg[o]; gg[o]=(v/(1.f+expf(-v)))*uu[o]; }
      mm(hh.data(),gg.data(),d[e],ds[e],I,D);
      for(int o=0;o<D;o++) cpuout[o]+=rw[xoff[e]+r]*hh[o];
    }
  }
  int gpu_block(){
    std::fill(gout.begin(),gout.end(),0.f);
    return coli_metal_moe_block(nb,D,I,fmt,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                                xg.data(),xoff.data(),nr.data(),rows.data(),rw.data(),gout.data(),1);
  }
  double agreement(){
    double mx=0,ym=0;
    for(int i=0;i<D;i++){ double e=fabs(gout[i]-cpuout[i]); if(e>mx)mx=e; if(fabs(cpuout[i])>ym)ym=fabs(cpuout[i]); }
    return mx/(ym+1e-9);
  }
  void release(){ for(int e=0;e<nb;e++){ free(slab[e]); free(fslab[e]); } }
};

int main(int argc, char** argv){
  if(!coli_metal_init()){ printf("no metal\n"); return 1; }
  int nb = argc>1 ? atoi(argv[1]) : 8;                 // experts in the block (top-8 decode)
  int D  = argc>2 ? atoi(argv[2]) : 6144;              // GLM hidden; pass a non-multiple of 8 to
  int I  = argc>3 ? atoi(argv[3]) : 2048;              // knock fmt=2 off its 8-wide vector path
  const char* re=getenv("BENCH_REP");    int REP    = re?atoi(re):50;   // GPU calls per round
  const char* ro=getenv("BENCH_ROUNDS"); int ROUNDS = ro?atoi(ro):9;    // interleaved rounds

  Arm a4, a3; a4.setup(2,nb,D,I); a3.setup(5,nb,D,I);
  a4.cpu_block(); a3.cpu_block();
  if(!a4.gpu_block() || !a3.gpu_block()){ printf("moe_block refused a format\n"); return 1; }
  double n4=a4.agreement(), n3=a3.agreement();

  std::vector<double> k4,k3,c4,c3;
  for(int r=0;r<ROUNDS;r++){
    double t=now(); a4.cpu_block(); c4.push_back((now()-t)*1e3);
    t=now();        a3.cpu_block(); c3.push_back((now()-t)*1e3);
    double b=coli_metal_moe_kernel_time();
    for(int i=0;i<REP;i++) a4.gpu_block();
    double m=coli_metal_moe_kernel_time(); k4.push_back((m-b)/REP*1e3);
    for(int i=0;i<REP;i++) a3.gpu_block();
    k3.push_back((coli_metal_moe_kernel_time()-m)/REP*1e3);
  }
  double K4=median(k4), K3=median(k3), C4=median(c4), C3=median(c3);
  printf("MoE block, %d experts, D=%d I=%d, decode row, %d rounds x %d reps, medians\n",
         nb, D, I, ROUNDS, REP);
  printf("  fmt=2 per-row int4   CPU %7.3f ms   GPU kernel %7.3f ms   nerr %.1e   %.1f MB/expert\n",
         C4, K4, n4, a4.wbytes/1048576.0);
  printf("  fmt=5 int3-g64       CPU %7.3f ms   GPU kernel %7.3f ms   nerr %.1e   %.1f MB/expert\n",
         C3, K3, n3, a3.wbytes/1048576.0);
  printf("  int3 vs int4         CPU %.2fx        GPU kernel %.2fx        bytes %.4fx\n",
         C4/C3, K4/K3, (double)a3.wbytes/(double)a4.wbytes);
  printf("  kernel spread        int4 %.3f..%.3f   int3 %.3f..%.3f\n",
         *std::min_element(k4.begin(),k4.end()), *std::max_element(k4.begin(),k4.end()),
         *std::min_element(k3.begin(),k3.end()), *std::max_element(k3.begin(),k3.end()));
  a4.release(); a3.release();
  return 0;
}
