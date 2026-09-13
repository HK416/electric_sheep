// PT — the compute path tracer (spec 15.3, spec 28.6). Spec 1.9 item 2: this is the second
// thing cut under scope pressure, so it is the minimal honest version.
//
// Diffuse only, cosine-weighted sampling, fixed bounce count, no Russian roulette (it would
// make the work per pixel data-dependent for no gain at these bounce counts), sequential
// sample accumulation in ascending order. Depth, segmentation and normal come from the
// primary hit of sample 0, before any RNG draw, so they equal the RS outputs bit for bit.

#include "common.slang"

[[vk::binding(2, 0)]] RWStructuredBuffer<float> out_radiance; // linear, 3 per pixel
[[vk::binding(3, 0)]] RWStructuredBuffer<float> out_depth;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> out_seg;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> out_normal;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> out_tri;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return; }
    EsPixel p = es_pixel(tid);

    out_seg[p.index] = 0;
    out_tri[p.index] = 0;
    out_normal[p.index * 3 + 0] = 0.0;
    out_normal[p.index * 3 + 1] = 0.0;
    out_normal[p.index * 3 + 2] = 0.0;
    out_radiance[p.index * 3 + 0] = 0.0;
    out_radiance[p.index * 3 + 1] = 0.0;
    out_radiance[p.index * 3 + 2] = 0.0;
    if (!p.valid) {
        out_depth[p.index] = 0.0;
        return;
    }

    EsView vw = es_view(p.view);
    out_depth[p.index] = vw.far;

    float3 primary = es_primary_dir(vw, p.local);
    float t0;
    uint hit0;
    if (es_nearest(vw.pos, primary, vw.near, vw.far, t0, hit0)) {
        EsTri tr0 = es_tri(hit0);
        float3 n0 = es_qrot_inv(vw.quat, es_face_forward(tr0.n, primary));
        out_depth[p.index] = t0;
        out_seg[p.index] = tr0.seg;
        out_tri[p.index] = hit0 + 1;
        out_normal[p.index * 3 + 0] = n0.x;
        out_normal[p.index * 3 + 1] = n0.y;
        out_normal[p.index * 3 + 2] = n0.z;
    }

    uint spp = max(es_spp(), 1u);
    uint bounces = max(es_bounces(), 1u);
    float3 acc = float3(0.0, 0.0, 0.0);
    for (uint s = 0; s < spp; ++s) {
        float3 throughput = float3(1.0, 1.0, 1.0);
        float3 o = vw.pos;
        float3 d = primary;
        float near = vw.near;
        for (uint b = 0; b < bounces; ++b) {
            float t;
            uint hit;
            if (!es_nearest(o, d, near, vw.far, t, hit)) {
                acc += throughput * es_sky();
                break;
            }
            EsTri tr = es_tri(hit);
            acc += throughput * tr.emission;
            throughput *= tr.albedo;
            float3 n = es_face_forward(tr.n, d);
            o = o + d * t + n * ES_RAY_EPS;
            d = es_cosine_hemisphere(
                n, es_rng_key(es_seed(), p.view, p.local.x, p.local.y, s, b, 0));
            near = 0.0;
        }
    }
    acc = acc * (1.0 / float(spp));
    out_radiance[p.index * 3 + 0] = acc.x;
    out_radiance[p.index * 3 + 1] = acc.y;
    out_radiance[p.index * 3 + 2] = acc.z;
}
