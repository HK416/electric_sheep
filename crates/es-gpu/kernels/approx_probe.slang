// Spec 28.7 gate 3: the same transcendental on the CPU and on the GPU, bit for bit.
//
// `approx.slang` is the mirror of `es_math::approx` (spec 3.2); this kernel only evaluates
// it, so any difference is the compiler, the driver or a missing execution mode — not a
// different formula.
//
// ES_N inputs at binding 1; 2*ES_N outputs at binding 0: sin in the first half, exp in the
// second.

#include "approx.slang"

#ifndef ES_N
#define ES_N 4096
#endif

[[vk::binding(0, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(1, 0)]] StructuredBuffer<float> src;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    float x = src[i];
    dst[i] = es_sin(x);
    dst[i + ES_N] = es_exp(x);
}
