// The variance half of SVGF (packet M7/R4, spec 28.6; Schied et al. 2017 section 4.2).
//
// One dispatch after the tracer, before the a-trous filter, because the short-history
// fallback reads a 7x7 neighbourhood and every pixel's accumulated colour has to be written
// first. The accumulation itself is `pt.slang`'s: it is per-pixel and belongs in the loop
// that produces the samples.
//
// With `n >= ES_SVGF_MIN_HISTORY` frames the variance comes from the moments the tracer
// keeps, `E[l^2] - E[l]^2`, divided by `n` once more because what the filter needs is the
// variance of the *mean* of those frames. Below that the moments are too few and the paper's
// depth/normal-weighted spatial estimate stands in, divided by `n` for the same reason.
//
// Mirror of `cpu::variance_estimate`. No atomics, no shared memory, a fixed 7x7 window.

#include "common.slang"

[[vk::binding(2, 0)]] RWStructuredBuffer<float> history;    // ES_HIST_STRIDE per pixel
[[vk::binding(3, 0)]] StructuredBuffer<float> color;        // the accumulated radiance
[[vk::binding(4, 0)]] StructuredBuffer<float> g_depth;
[[vk::binding(5, 0)]] StructuredBuffer<float> g_normal;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> out_variance;

float3 es_color_at(uint j) {
    return float3(color[j * 3 + 0], color[j * 3 + 1], color[j * 3 + 2]);
}

float3 es_normal_at(uint j) {
    return float3(g_normal[j * 3 + 0], g_normal[j * 3 + 1], g_normal[j * 3 + 2]);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return; }
    EsPixel p = es_pixel(tid);
    out_variance[p.index] = 0.0;
    if (!p.valid) { return; }

    uint hb = p.index * ES_HIST_STRIDE;
    uint n = asuint(history[hb + 5]);
    if (n == 0) { return; }
    float inv_n = 1.0 / float(n);

    float var;
    if (n >= ES_SVGF_MIN_HISTORY) {
        float m1 = history[hb + 3] * inv_n;
        float m2 = history[hb + 4] * inv_n;
        var = max(m2 - m1 * m1, 0.0) * inv_n;
    } else {
        float sw = 0.0;
        float swl = 0.0;
        float swl2 = 0.0;
        float3 n_p = es_normal_at(p.index);
        for (int dy = -ES_SVGF_SPATIAL_RADIUS; dy <= ES_SVGF_SPATIAL_RADIUS; ++dy) {
            for (int dx = -ES_SVGF_SPATIAL_RADIUS; dx <= ES_SVGF_SPATIAL_RADIUS; ++dx) {
                int nx = int(p.local.x) + dx;
                int ny = int(p.local.y) + dy;
                if (nx < 0 || ny < 0 || nx >= int(es_tile_w()) || ny >= int(es_tile_h())) {
                    continue;
                }
                uint ax = tid.x - p.local.x + uint(nx);
                uint ay = tid.y - p.local.y + uint(ny);
                uint j = ay * es_atlas_w() + ax;
                float wz;
                float wn;
                es_edge_weights(g_depth[p.index], g_depth[j], n_p, es_normal_at(j), wz, wn);
                float weight = wz * wn;
                float l = es_luminance(es_color_at(j));
                sw += weight;
                swl += weight * l;
                swl2 += weight * l * l;
            }
        }
        float inv = sw > 0.0 ? 1.0 / sw : 0.0;
        float mean = swl * inv;
        var = max(swl2 * inv - mean * mean, 0.0) * inv_n;
    }
    history[hb + 11] = var;
    out_variance[p.index] = var;
}
