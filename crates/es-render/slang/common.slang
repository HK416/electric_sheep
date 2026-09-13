// Shared geometry, camera and shading code for both render paths.
//
// Mirror of `crates/es-render/src/cpu.rs`: same intersection routine, same traversal order,
// same `approx` calls, same accumulation order. Where the two texts diverge this file is
// wrong, not the Rust.
//
// Bindings 0 and 1 are the same in every kernel of this crate:
//   0  params  — globals then one 16-float record per view
//   1  tris    — 20 floats per triangle

#ifndef ES_RENDER_COMMON_SLANG
#define ES_RENDER_COMMON_SLANG

#include "approx.slang"
#include "rng.slang"

static const float ES_DET_EPS = 1e-8;
static const float ES_RAY_EPS = 1e-4;
static const float ES_PI = 3.14159265358979323846;
static const float ES_INV_PI = 0.31830988618379067154;
static const float ES_INV_GAMMA = 0.41666666;

static const uint ES_TRI_STRIDE = 20;
static const uint ES_VIEW_STRIDE = 16;
static const uint ES_PARAM_VIEW_BASE = 20;

[[vk::binding(0, 0)]] StructuredBuffer<float> es_params;
[[vk::binding(1, 0)]] StructuredBuffer<float> es_tris;

uint es_param_u(uint i) { return asuint(es_params[i]); }

uint es_n_tri()        { return es_param_u(0); }
uint es_n_views()      { return es_param_u(1); }
uint es_tile_w()       { return es_param_u(2); }
uint es_tile_h()       { return es_param_u(3); }
uint es_tiles_per_row(){ return es_param_u(4); }
uint es_rows()         { return es_param_u(5); }
float3 es_light_dir()  { return float3(es_params[6], es_params[7], es_params[8]); }
float es_ambient()     { return es_params[9]; }
float3 es_sky()        { return float3(es_params[10], es_params[11], es_params[12]); }
uint es_seed()         { return es_param_u(13); }
uint es_spp()          { return es_param_u(14); }
uint es_bounces()      { return es_param_u(15); }
uint es_n_lights()     { return es_param_u(16); }
uint es_atlas_w()      { return es_tiles_per_row() * es_tile_w(); }
uint es_atlas_h()      { return es_rows() * es_tile_h(); }

struct EsTri {
    float3 v0;
    float3 v1;
    float3 v2;
    float3 n;
    float3 albedo;
    float3 emission;
    uint seg;
};

EsTri es_tri(uint i) {
    uint b = i * ES_TRI_STRIDE;
    EsTri t;
    t.v0 = float3(es_tris[b + 0], es_tris[b + 1], es_tris[b + 2]);
    t.v1 = float3(es_tris[b + 3], es_tris[b + 4], es_tris[b + 5]);
    t.v2 = float3(es_tris[b + 6], es_tris[b + 7], es_tris[b + 8]);
    t.n = float3(es_tris[b + 9], es_tris[b + 10], es_tris[b + 11]);
    t.albedo = float3(es_tris[b + 12], es_tris[b + 13], es_tris[b + 14]);
    t.emission = float3(es_tris[b + 15], es_tris[b + 16], es_tris[b + 17]);
    t.seg = asuint(es_tris[b + 18]);
    return t;
}

// Emissive triangle `k` of the light list. The list is the ascending run of emissive
// triangles, so it is recomputed here by scanning rather than uploaded a second time.
uint es_light_tri(uint k) {
    uint seen = 0;
    uint n = es_n_tri();
    for (uint i = 0; i < n; ++i) {
        uint b = i * ES_TRI_STRIDE;
        if (es_tris[b + 15] > 0.0 || es_tris[b + 16] > 0.0 || es_tris[b + 17] > 0.0) {
            if (seen == k) { return i; }
            seen += 1;
        }
    }
    return 0;
}

struct EsView {
    float3 pos;
    float4 quat;
    float fx;
    float fy;
    float cx;
    float cy;
    float near;
    float far;
};

EsView es_view(uint v) {
    uint b = ES_PARAM_VIEW_BASE + v * ES_VIEW_STRIDE;
    EsView o;
    o.pos = float3(es_params[b + 0], es_params[b + 1], es_params[b + 2]);
    o.quat = float4(es_params[b + 3], es_params[b + 4], es_params[b + 5], es_params[b + 6]);
    o.fx = es_params[b + 7];
    o.fy = es_params[b + 8];
    o.cx = es_params[b + 9];
    o.cy = es_params[b + 10];
    o.near = es_params[b + 11];
    o.far = es_params[b + 12];
    return o;
}

float3 es_qrot(float4 q, float3 v) {
    float3 u = float3(q.x, q.y, q.z);
    float3 t = 2.0 * cross(u, v);
    return v + q.w * t + cross(u, t);
}

float3 es_qrot_inv(float4 q, float3 v) {
    return es_qrot(float4(-q.x, -q.y, -q.z, q.w), v);
}

// Which tile, which view, which atlas slot. `valid` is false for a padding tile or a thread
// off the end of the atlas.
struct EsPixel {
    uint2 local;   // pixel inside the tile
    uint view;
    uint index;    // atlas-linear index
    bool valid;
};

EsPixel es_pixel(uint3 tid) {
    EsPixel p;
    p.local = uint2(0, 0);
    p.view = 0;
    p.index = 0;
    p.valid = false;
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return p; }
    p.index = tid.y * es_atlas_w() + tid.x;
    uint tile = (tid.y / es_tile_h()) * es_tiles_per_row() + (tid.x / es_tile_w());
    p.local = uint2(tid.x % es_tile_w(), tid.y % es_tile_h());
    p.view = tile;
    p.valid = tile < es_n_views();
    return p;
}

// Moller-Trumbore. `t` is unfiltered by near/far.
bool es_intersect(EsTri tr, float3 o, float3 d, out float t) {
    t = 0.0;
    float3 e1 = tr.v1 - tr.v0;
    float3 e2 = tr.v2 - tr.v0;
    float3 p = cross(d, e2);
    float det = dot(e1, p);
    if (abs(det) < ES_DET_EPS) { return false; }
    float inv = 1.0 / det;
    float3 tv = o - tr.v0;          // camera-relative already (spec 3.3): `o` is the camera
    float u = dot(tv, p) * inv;
    if (u < 0.0 || u > 1.0) { return false; }
    float3 q = cross(tv, e1);
    float v = dot(d, q) * inv;
    if (v < 0.0 || u + v > 1.0) { return false; }
    t = dot(e2, q) * inv;
    return true;
}

// Scan every triangle in ascending index order; nearest strictly inside (near, far) wins,
// ties to the lower index. No atomics, no shared memory, no dependence on scheduling.
// ponytail: O(pixels x triangles) — spec 15.4's TLAS is the upgrade path and needs a Vulkan
// extension es-gpu does not expose.
bool es_nearest(float3 o, float3 d, float near, float far, out float best_t, out uint best_i) {
    best_t = far;
    best_i = 0;
    bool any = false;
    uint n = es_n_tri();
    for (uint i = 0; i < n; ++i) {
        float t;
        if (es_intersect(es_tri(i), o, d, t)) {
            if (t > near && t < best_t) { best_t = t; best_i = i; any = true; }
        }
    }
    return any;
}

bool es_any_hit(float3 o, float3 d, float near, float far) {
    uint n = es_n_tri();
    for (uint i = 0; i < n; ++i) {
        float t;
        if (es_intersect(es_tri(i), o, d, t)) {
            if (t > near && t < far) { return true; }
        }
    }
    return false;
}

float3 es_face_forward(float3 n, float3 d) {
    return dot(n, d) > 0.0 ? -n : n;
}

// Camera-space primary ray, NOT normalised: z == 1, so the ray parameter is the camera-space
// depth in metres and Depth32's unit_m is 1 (spec 3.1, spec 7.2).
float3 es_primary_dir(EsView vw, uint2 px) {
    float3 d_cam = float3((float(px.x) + 0.5 - vw.cx) / vw.fx,
                          (float(px.y) + 0.5 - vw.cy) / vw.fy,
                          1.0);
    return es_qrot(vw.quat, d_cam);
}

// Flat + Lambert, one directional light, no shadow ray (the Pt path models occlusion).
float3 es_shade_lambert(EsTri tr, float3 n) {
    float ndl = max(dot(n, es_light_dir()), 0.0);
    float lambert = es_ambient() + ndl * (1.0 - es_ambient());
    return tr.albedo * lambert + tr.emission;
}

// Exact piecewise sRGB transfer (spec 3.1); the power goes through `approx.slang`, not
// GLSL.std.450 (spec 3.2 DET-010).
float es_srgb(float c) {
    c = clamp(c, 0.0, 1.0);
    if (c <= 0.0031308) { return 12.92 * c; }
    return 1.055 * es_exp(es_ln(c) * ES_INV_GAMMA) - 0.055;
}

uint es_u8(float c) {
    return uint(clamp(floor(255.0 * es_srgb(c) + 0.5), 0.0, 255.0));
}

uint es_pack_rgba8(float3 lin) {
    return es_u8(lin.x) | (es_u8(lin.y) << 8) | (es_u8(lin.z) << 16) | 0xff000000u;
}

float3 es_normalize(float3 a) {
    float n = es_sqrt(dot(a, a));
    return n > 0.0 ? a * (1.0 / n) : float3(0.0, 0.0, 0.0);
}

float es_luminance(float3 c) {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

// Duff et al., branchless orthonormal basis.
void es_onb(float3 n, out float3 t, out float3 b) {
    float sgn = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sgn + n.z);
    float bb = n.x * n.y * a;
    t = float3(1.0 + sgn * n.x * n.x * a, sgn * bb, -sgn * n.x);
    b = float3(bb, sgn + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere sample: the weighting cancels the cos term and the 1/pi of a
// Lambertian BRDF, so throughput is a plain multiply by albedo — no pdf division, no 0/0.
float3 es_cosine_hemisphere(float3 n, uint key) {
    float u1 = es_rng_uniform(key, 0);
    float u2 = es_rng_uniform(key, 1);
    float r = es_sqrt(u1);
    float phi = 2.0 * ES_PI * u2;
    float3 t;
    float3 b;
    es_onb(n, t, b);
    float z = es_sqrt(max(1.0 - u1, 0.0));
    return es_normalize(t * (r * es_cos(phi)) + b * (r * es_sin(phi)) + n * z);
}

// Uniform point on a triangle from two canonical randoms.
float3 es_tri_point(EsTri tr, float u, float v) {
    float su = es_sqrt(u);
    float b0 = 1.0 - su;
    float b1 = v * su;
    return tr.v0 * (1.0 - b0 - b1) + tr.v1 * b0 + tr.v2 * b1;
}

float es_tri_area(EsTri tr) {
    float3 c = cross(tr.v1 - tr.v0, tr.v2 - tr.v0);
    return 0.5 * es_sqrt(dot(c, c));
}

#endif
