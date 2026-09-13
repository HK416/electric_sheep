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

    float3 d = es_primary_dir(vw, p.local);
    float t;
    uint hit;
    if (!es_nearest(vw.pos, d, vw.near, vw.far, t, hit)) { return; }

    EsTri tr = es_tri(hit);
    float3 n = es_face_forward(tr.n, d);
    float3 n_cam = es_qrot_inv(vw.quat, n);

    out_depth[p.index] = t;
    out_seg[p.index] = tr.seg;
    out_tri[p.index] = hit + 1;
    out_normal[p.index * 3 + 0] = n_cam.x;
    out_normal[p.index * 3 + 1] = n_cam.y;
    out_normal[p.index * 3 + 2] = n_cam.z;
    out_rgb[p.index] = es_pack_rgba8(es_shade_lambert(tr, n));
}
