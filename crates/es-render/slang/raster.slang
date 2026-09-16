// RS — the compute rasterizer (spec 15.3), one thread per atlas pixel.
//
// Coverage and depth come from the same ray-triangle intersection the path tracer uses, not
// from screen-space edge functions and interpolated 1/z. That is deliberate: it is what makes
// spec 15.3's "depth, seg and normal are bit-identical between RS and PT" hold by
// construction instead of to a tolerance. "Rasterizer" here is a statement about the loop
// structure — one thread per pixel, scan the primitives, depth test — not about fixed-
// function triangle setup. See docs/design/renderer.md.

#include "common.slang"

[[vk::binding(2, 0)]] RWStructuredBuffer<uint> out_rgb;     // packed RGBA8, one word per pixel
[[vk::binding(3, 0)]] RWStructuredBuffer<float> out_depth;  // metres, camera space
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> out_seg;     // 1-based geom id, 0 = background
[[vk::binding(5, 0)]] RWStructuredBuffer<float> out_normal; // camera space, 3 per pixel
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> out_tri;     // hit triangle + 1, 0 = miss

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return; }
    EsPixel p = es_pixel(tid);

    out_rgb[p.index] = 0xff000000u;
    out_seg[p.index] = 0;
    out_tri[p.index] = 0;
    out_normal[p.index * 3 + 0] = 0.0;
    out_normal[p.index * 3 + 1] = 0.0;
    out_normal[p.index * 3 + 2] = 0.0;
    if (!p.valid) {
        out_depth[p.index] = 0.0;
        return;
    }

    EsView vw = es_view(p.view);
    out_depth[p.index] = vw.far;

    // The geometry channels are the **centre** ray's, whatever the shading is: that is what
    // keeps them bit-identical to Lambert's, and to the Pt path's (spec 15.3).
    float3 d = es_primary_dir(vw, p.local);
    float t;
    uint hit;
    bool got = es_nearest(vw.pos, d, vw.near, vw.far, t, hit);
    if (got) {
        EsTri tr = es_tri(hit);
        float3 n_cam = es_qrot_inv(vw.quat, es_face_forward(tr.n, d));
        out_depth[p.index] = t;
        out_seg[p.index] = tr.seg;
        out_tri[p.index] = hit + 1;
        out_normal[p.index * 3 + 0] = n_cam.x;
        out_normal[p.index * 3 + 1] = n_cam.y;
        out_normal[p.index * 3 + 2] = n_cam.z;
    }

    if (es_shading() == 0) {
        if (got) {
            EsTri tr = es_tri(hit);
            out_rgb[p.index] = es_pack_rgba8(es_shade_lambert(tr, es_face_forward(tr.n, d)));
        }
        return;
    }

    // Shading::Full (packet M7/R2), the mirror of `cpu::rasterize`'s Full branch. Own rays: a
    // sub-sample the centre ray missed can still cover geometry. The ssaa x ssaa block is
    // summed in row-major order (sy outer, sx inner, one sequential +=), multiplied once by
    // 1 / (ssaa * ssaa) and only then encoded (spec 3.4: a fixed accumulation order). A
    // sub-sample that hits nothing contributes linear zero — Lambert's background.
    uint ssaa = es_ssaa();
    float inv_ssaa = 1.0 / float(ssaa);
    float3 acc = float3(0.0, 0.0, 0.0);
    for (uint sy = 0; sy < ssaa; ++sy) {
        for (uint sx = 0; sx < ssaa; ++sx) {
            float3 ds = es_primary_dir_sub(vw, p.local, sx, sy, inv_ssaa);
            float ts;
            uint hs;
            if (!es_nearest(vw.pos, ds, vw.near, vw.far, ts, hs)) { continue; }
            EsTri trs = es_tri(hs);
            float3 ns = es_face_forward(trs.n, ds);
            acc += es_shade_full(trs, ns, vw.pos + ds * ts, ds);
        }
    }
    out_rgb[p.index] = es_pack_rgba8(acc * (1.0 / float(ssaa * ssaa)));
}
