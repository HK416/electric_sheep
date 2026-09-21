// The a-trous half of SVGF (spec 28.6). One dispatch per iteration, stride 1 << i, the
// standard 5-tap B-spline wavelet with depth, normal and -- since packet M7/R4, when a
// temporal history exists -- luminance edge stopping.
//
// The luminance weight is `exp(-|l_p - l_q| / (sigma_l * sqrt(var_p) + eps))` over the
// variance `accumulate.slang` estimated, and the variance is filtered alongside the colour
// with the squared weights. With `es_max_history() == 0` the weight is exactly 1.0 and the
// output is byte for byte what M4's filter produced.
//
// What is still skipped: history-length-driven kernel widening. See
// docs/design/renderer.md sections 4.3 and 11.

#include "common.slang"

[[vk::binding(2, 0)]] StructuredBuffer<float> color_in;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> color_out;
[[vk::binding(4, 0)]] StructuredBuffer<float> g_depth;
[[vk::binding(5, 0)]] StructuredBuffer<float> g_normal;
[[vk::binding(6, 0)]] StructuredBuffer<float> iter_params; // [0] = a-trous stride
[[vk::binding(7, 0)]] StructuredBuffer<float> var_in;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> var_out;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return; }
    EsPixel p = es_pixel(tid);
    if (!p.valid) {
        color_out[p.index * 3 + 0] = 0.0;
        color_out[p.index * 3 + 1] = 0.0;
        color_out[p.index * 3 + 2] = 0.0;
        var_out[p.index] = 0.0;
        return;
    }

    float h[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
    int stride = int(asuint(iter_params[0]));
    bool guided = es_max_history() != 0;

    float3 c_p = float3(color_in[p.index * 3 + 0],
                        color_in[p.index * 3 + 1],
                        color_in[p.index * 3 + 2]);
    float l_p = es_luminance(c_p);
    float sigma_l = guided ? ES_SVGF_SIGMA_L * es_sqrt(var_in[p.index]) + ES_SVGF_VAR_EPS : 0.0;
    float3 n_p = float3(g_normal[p.index * 3 + 0],
                        g_normal[p.index * 3 + 1],
                        g_normal[p.index * 3 + 2]);

    float3 sum = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    float vsum = 0.0;
    for (int ky = 0; ky < 5; ++ky) {
        for (int kx = 0; kx < 5; ++kx) {
            int nx = int(p.local.x) + (kx - 2) * stride;
            int ny = int(p.local.y) + (ky - 2) * stride;
            if (nx < 0 || ny < 0 || nx >= int(es_tile_w()) || ny >= int(es_tile_h())) { continue; }
            uint ax = tid.x - p.local.x + uint(nx);
            uint ay = tid.y - p.local.y + uint(ny);
            uint j = ay * es_atlas_w() + ax;

            float wh = h[kx] * h[ky];
            float wz;
            float wn;
            float3 n_q = float3(g_normal[j * 3 + 0], g_normal[j * 3 + 1], g_normal[j * 3 + 2]);
            es_edge_weights(g_depth[p.index], g_depth[j], n_p, n_q, wz, wn);
            float3 c_q = float3(color_in[j * 3 + 0], color_in[j * 3 + 1], color_in[j * 3 + 2]);
            float wl = guided ? es_exp(-abs(l_p - es_luminance(c_q)) / sigma_l) : 1.0;

            float weight = wh * wz * wn * wl;
            sum += c_q * weight;
            wsum += weight;
            if (guided) { vsum += weight * weight * var_in[j]; }
        }
    }
    float inv = wsum > 0.0 ? 1.0 / wsum : 0.0;
    color_out[p.index * 3 + 0] = sum.x * inv;
    color_out[p.index * 3 + 1] = sum.y * inv;
    color_out[p.index * 3 + 2] = sum.z * inv;
    var_out[p.index] = guided ? vsum * inv * inv : 0.0;
}
