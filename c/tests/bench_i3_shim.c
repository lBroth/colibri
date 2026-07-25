/* Exposes the engine's own quantized matmuls to the Metal bench without duplicating them:
 * same trick tests/test_int3.c uses to pull colibri.c into a test binary. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

void ref_i3(float *y, const float *x, const unsigned char *q, const float *s,
            int S, int I, int O)
{
    matmul_i3(y, x, q, s, S, I, O);
}

void ref_i4(float *y, const float *x, const unsigned char *q, const float *s,
            int S, int I, int O)
{
    matmul_i4(y, x, q, s, S, I, O);
}
