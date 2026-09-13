// The a-trous half of SVGF (spec 28.6). One dispatch per iteration, stride 1 << i, the
// standard 5-tap B-spline wavelet with depth and normal edge stopping.
//
// This is NOT SVGF: there is no temporal accumulation, no per-pixel variance estimate, no
// variance-guided luminance weight, no variance prefilter, no disocclusion handling and no
// history-driven kernel widening. It is named for spec 28.6's deliverable; what it actually
// is, is an edge-stopping a-trous filter. See docs/design/renderer.md.

#include "common.slang"

static const float ES_SVGF_SIGMA_Z = 0.1;

[[vk::binding(2, 0)]] StructuredBuffer<float> color_in;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> color_out;
[[vk::binding(4, 0)]] StructuredBuffer<float> g_depth;
[[vk::binding(5, 0)]] StructuredBuffer<float> g_normal;
[[vk::binding(6, 0)]] StructuredBuffer<float> iter_params; // [0] = a-trous stride

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return; }
    EsPixel p = es_pixel(tid);
    if (!p.valid) {
        color_out[p.index * 3 + 0] = 0.0;
        color_out[p.index * 3 + 1] = 0.0;
        color_out[p.index * 3 + 2] = 0.0;
        return;
    }

    float h[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
    int stride = int(asuint(iter_params[0]));

    float3 sum = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    for (int ky = 0; ky < 5; ++ky) {
        for (int kx = 0; kx < 5; ++kx) {
            int nx = int(p.local.x) + (kx - 2) * stride;
            int ny = int(p.local.y) + (ky - 2) * stride;
            if (nx < 0 || ny < 0 || nx >= int(es_tile_w()) || ny >= int(es_tile_h())) { continue; }
            uint ax = tid.x - p.local.x + uint(nx);
            uint ay = tid.y - p.local.y + uint(ny);
            uint j = ay * es_atlas_w() + ax;

            float wh = h[kx] * h[ky];
            float wz = es_exp(-abs(g_depth[p.index] - g_depth[j]) / ES_SVGF_SIGMA_Z);
            float wn = max(g_normal[p.index * 3 + 0] * g_normal[j * 3 + 0]
                         + g_normal[p.index * 3 + 1] * g_normal[j * 3 + 1]
                         + g_normal[p.index * 3 + 2] * g_normal[j * 3 + 2], 0.0);
            // n^32 by five squarings: no transcendental, the same ops as the Rust mirror.
            for (uint s = 0; s < 5; ++s) { wn *= wn; }

            float weight = wh * wz * wn;
            sum += float3(color_in[j * 3 + 0], color_in[j * 3 + 1], color_in[j * 3 + 2]) * weight;
            wsum += weight;
        }
    }
    float inv = wsum > 0.0 ? 1.0 / wsum : 0.0;
    color_out[p.index * 3 + 0] = sum.x * inv;
    color_out[p.index * 3 + 1] = sum.y * inv;
    color_out[p.index * 3 + 2] = sum.z * inv;
}
