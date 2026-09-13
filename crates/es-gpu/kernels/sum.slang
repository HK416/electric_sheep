// One level of a fixed-order pairwise tree reduction (spec 3.4: deterministic reduction,
// deterministic scheduling, fixed algorithm).
//
// dst[i] = src[2i] + src[2i+1]. The summation order is a property of the index arithmetic
// alone: no shared memory, no subgroup operations, no atomics, no dependence on the
// workgroup count or the device's subgroup size. Two runs on the same device with the same
// input must give the same bits, and a CPU mirror doing the same tree in f32 must give those
// same bits too.
//
// The dispatch count is chosen so that every invocation is in range; the kernel has no
// bounds check because the tree stops while the element count is still a multiple of 128.

[[vk::binding(0, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(1, 0)]] StructuredBuffer<float> src;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    dst[i] = src[2 * i] + src[2 * i + 1];
}
