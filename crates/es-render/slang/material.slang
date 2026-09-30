// Materials and textures (plan H, packet HT1). Mirror of `crates/es-render/src/material.rs`,
// expression for expression; where the two texts diverge this file is wrong, not the Rust.
//
// The region sits in `es_tris` at `es_mat_base()` (global slot 41), after the scene:
//   LUT (512 floats: i/255, then the sRGB decode of i/255)
//   materials, ES_MAT_STRIDE floats each (rgb tex+1, metallic, roughness, metal tex+1,
//     metal channel, rough tex+1, rough channel, pbr, normal tex+1, normal scale,
//     emissive tex+1, pad — the last four plan H's HT2)
//   texture headers, 4 words each (texel offset from the region start, width, face height,
//     flags: bit 0 cube, bit 1 sRGB, bits 2-3 / 4-5 the s / t wrap: 0 repeat, 1 clamp, 2 mirror)
//   texels, one packed r | g << 8 | b << 16 word each, cube faces stacked
// A triangle with `mat == 0` never reads any of it.
//
// Every dot product and cross product below is written out (`es_dot3`, `es_cross`): SPIR-V's
// `OpDot` does not pin its summation order (renderer.md 9.3), and the texture coordinate of a
// hit has to be the CPU's.

#ifndef ES_RENDER_MATERIAL_SLANG
#define ES_RENDER_MATERIAL_SLANG

static const uint ES_MAT_STRIDE = 12;
static const uint ES_LUT_LEN = 512;
static const float ES_MIN_ROUGHNESS = 0.05;

uint es_mat_base() { return es_env_u(41); }
uint es_n_mat()    { return es_env_u(42); }

struct EsSurface {
    float3 base;
    float metallic;
    float roughness;
    bool pbr;
    // Plan H, HT2: the emission at the hit, and the normal-mapped shading normal (facing the
    // ray) when `bumped`.
    float3 emission;
    bool bumped;
    float3 ns;
};

EsSurface es_surface_flat(float3 albedo) {
    EsSurface s;
    s.base = albedo;
    s.metallic = 0.0;
    s.roughness = 1.0;
    s.pbr = false;
    s.emission = float3(0.0, 0.0, 0.0);
    s.bumped = false;
    s.ns = float3(0.0, 0.0, 0.0);
    return s;
}

// The untextured triangle: its flat albedo and its emission.
EsSurface es_surface_of(EsTri tr) {
    EsSurface s = es_surface_flat(tr.albedo);
    s.emission = tr.emission;
    return s;
}

float es_dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

float3 es_norm3(float3 a) {
    float n = es_sqrt(es_dot3(a, a));
    return n > 0.0 ? a * (1.0 / n) : float3(0.0, 0.0, 0.0);
}

// Moller-Trumbore's barycentrics at the hit, and the interpolated texture coordinate.
float3 es_tex_coord(EsTri tr, float3 o, float3 d) {
    float3 e1 = tr.v1 - tr.v0;
    float3 e2 = tr.v2 - tr.v0;
    float3 p = es_cross(d, e2);
    float inv = 1.0 / es_dot3(e1, p);
    float3 tv = o - tr.v0;
    float u = es_dot3(tv, p) * inv;
    float3 q = es_cross(tv, e1);
    float v = es_dot3(d, q) * inv;
    float w = 1.0 - u - v;
    return tr.tc0 * w + tr.tc1 * u + tr.tc2 * v;
}

// OpenGL's cube-map face selection and the face's (s, t).
void es_cube_face(float3 r, out uint face, out float s, out float t) {
    float ax = abs(r.x);
    float ay = abs(r.y);
    float az = abs(r.z);
    float ma;
    float sc;
    float tc;
    if (ax >= ay && ax >= az) {
        if (r.x >= 0.0) { face = 0; ma = ax; sc = -r.z; tc = -r.y; }
        else            { face = 1; ma = ax; sc = r.z;  tc = -r.y; }
    } else if (ay >= az) {
        if (r.y >= 0.0) { face = 2; ma = ay; sc = r.x; tc = r.z; }
        else            { face = 3; ma = ay; sc = r.x; tc = -r.z; }
    } else if (r.z >= 0.0) {
        face = 4; ma = az; sc = r.x; tc = -r.y;
    } else {
        face = 5; ma = az; sc = -r.x; tc = -r.y;
    }
    if (ma <= 0.0) { face = 0; s = 0.5; t = 0.5; return; }
    s = (sc / ma + 1.0) * 0.5;
    t = (tc / ma + 1.0) * 0.5;
}

float3 es_texel(uint hdr, uint face, uint row, uint col) {
    uint rb = es_mat_base();
    uint off = asuint(es_tris[hdr + 0]);
    uint w = asuint(es_tris[hdr + 1]);
    uint h = asuint(es_tris[hdr + 2]);
    uint flags = asuint(es_tris[hdr + 3]);
    uint word = asuint(es_tris[rb + off + (face * h + row) * w + col]);
    uint lb = rb + (((flags & 2u) != 0u) ? 256u : 0u);
    return float3(es_tris[lb + (word & 0xffu)],
                  es_tris[lb + ((word >> 8) & 0xffu)],
                  es_tris[lb + ((word >> 16) & 0xffu)]);
}

// Folds texel index i into 0..n by wrap mode (0 repeat, 1 clamp, 2 mirror).
uint es_wrap(int i, int n, uint mode) {
    if (mode == 1u) { return uint(clamp(i, 0, n - 1)); }
    if (mode == 2u) {
        int p = 2 * n;
        int m = ((i % p) + p) % p;
        return uint(m < n ? m : p - 1 - m);
    }
    return uint(((i % n) + n) % n);
}

// Bilinear, texel centres at +0.5: each axis of a 2D texture by its wrap mode, clamp to the
// edge of a cube face.
float3 es_sample(uint tex, float3 tc) {
    uint hdr = es_mat_base() + ES_LUT_LEN + es_n_mat() * ES_MAT_STRIDE + tex * 4u;
    int w = int(asuint(es_tris[hdr + 1]));
    int h = int(asuint(es_tris[hdr + 2]));
    uint flags = asuint(es_tris[hdr + 3]);
    bool cube = (flags & 1u) != 0u;
    uint face = 0;
    float s = tc.x;
    float t = tc.y;
    if (cube) { es_cube_face(tc, face, s, t); }
    float x = s * float(w) - 0.5;
    float y = t * float(h) - 0.5;
    float x0 = floor(x);
    float y0 = floor(y);
    float fx = x - x0;
    float fy = y - y0;
    int ix = int(x0);
    int iy = int(y0);
    uint ws = cube ? 1u : ((flags >> 2) & 3u);
    uint wt = cube ? 1u : ((flags >> 4) & 3u);
    uint cx0 = es_wrap(ix, w, ws);
    uint cx1 = es_wrap(ix + 1, w, ws);
    uint cy0 = es_wrap(iy, h, wt);
    uint cy1 = es_wrap(iy + 1, h, wt);
    float3 c00 = es_texel(hdr, face, cy0, cx0);
    float3 c10 = es_texel(hdr, face, cy0, cx1);
    float3 c01 = es_texel(hdr, face, cy1, cx0);
    float3 c11 = es_texel(hdr, face, cy1, cx1);
    float3 top = c00 * (1.0 - fx) + c10 * fx;
    float3 bottom = c01 * (1.0 - fx) + c11 * fx;
    return top * (1.0 - fy) + bottom * fy;
}

float es_channel(float3 c, uint ch) {
    return ch == 0u ? c.x : (ch == 1u ? c.y : c.z);
}

// The normal-mapped shading normal of texel `c` on `tr`, facing the ray `d`; false when the
// triangle's UVs have no area. Mirror of `material::bump`: the per-face MikkTSpace frame of
// glTF, the bitangent pointing up the image (towards decreasing t).
bool es_bump(EsTri tr, float3 d, float3 c, float k, out float3 ns) {
    ns = float3(0.0, 0.0, 0.0);
    float3 e1 = tr.v1 - tr.v0;
    float3 e2 = tr.v2 - tr.v0;
    float du1 = tr.tc1.x - tr.tc0.x;
    float dv1 = tr.tc1.y - tr.tc0.y;
    float du2 = tr.tc2.x - tr.tc0.x;
    float dv2 = tr.tc2.y - tr.tc0.y;
    float det = du1 * dv2 - du2 * dv1;
    if (!(det > 0.0 || det < 0.0)) { return false; }
    float r = 1.0 / det;
    float3 dpds = (e1 * dv2 - e2 * dv1) * r;
    float3 dpdt = (e2 * du1 - e1 * du2) * r;
    float3 n = tr.n;
    float3 t = es_norm3(dpds - n * es_dot3(n, dpds));
    if (es_dot3(t, t) <= 0.0) { return false; }
    float3 nt = es_cross(n, t);
    float w = es_dot3(nt, dpdt) > 0.0 ? -1.0 : 1.0;
    float x = (c.x * 2.0 - 1.0) * k;
    float y = (c.y * 2.0 - 1.0) * k;
    float z = c.z * 2.0 - 1.0;
    ns = es_norm3(float3(t.x * x + nt.x * w * y + n.x * z,
                         t.y * x + nt.y * w * y + n.y * z,
                         t.z * x + nt.z * w * y + n.z * z));
    if (es_dot3(n, d) > 0.0) { ns = -ns; }
    return true;
}

// The surface the ray (o, d) sees on `tr`. `mat == 0` is the flat albedo and emission and
// nothing else.
EsSurface es_surface(EsTri tr, float3 o, float3 d) {
    if (tr.mat == 0u) { return es_surface_of(tr); }
    uint mb = es_mat_base() + ES_LUT_LEN + (tr.mat - 1u) * ES_MAT_STRIDE;
    uint rgb = asuint(es_tris[mb + 0]);
    float metallic = es_tris[mb + 1];
    float roughness = es_tris[mb + 2];
    uint metal_tex = asuint(es_tris[mb + 3]);
    uint metal_ch = asuint(es_tris[mb + 4]);
    uint rough_tex = asuint(es_tris[mb + 5]);
    uint rough_ch = asuint(es_tris[mb + 6]);
    uint normal_tex = asuint(es_tris[mb + 8]);
    float normal_scale = es_tris[mb + 9];
    uint emissive_tex = asuint(es_tris[mb + 10]);
    EsSurface s;
    s.base = tr.albedo;
    s.emission = tr.emission;
    s.bumped = false;
    s.ns = float3(0.0, 0.0, 0.0);
    if (rgb != 0u || metal_tex != 0u || rough_tex != 0u || emissive_tex != 0u || normal_tex != 0u) {
        float3 tc = es_tex_coord(tr, o, d);
        if (rgb != 0u) { s.base = s.base * es_sample(rgb - 1u, tc); }
        if (metal_tex != 0u) { metallic = metallic * es_channel(es_sample(metal_tex - 1u, tc), metal_ch); }
        if (rough_tex != 0u) { roughness = roughness * es_channel(es_sample(rough_tex - 1u, tc), rough_ch); }
        if (emissive_tex != 0u) { s.emission = s.emission * es_sample(emissive_tex - 1u, tc); }
        if (normal_tex != 0u) {
            s.bumped = es_bump(tr, d, es_sample(normal_tex - 1u, tc), normal_scale, s.ns);
        }
    }
    s.metallic = metallic;
    s.roughness = clamp(roughness, ES_MIN_ROUGHNESS, 1.0);
    s.pbr = asuint(es_tris[mb + 7]) != 0u;
    return s;
}

// The emitted radiance at barycentrics (b0, b1) of `tr`, where es_tri_point puts a light
// sample: the emission, times the emissive texel under a map (plan H, HT2). Mirror of
// `Materials::emission_at`.
float3 es_emission_at(EsTri tr, float b0, float b1) {
    if (tr.mat == 0u) { return tr.emission; }
    uint mb = es_mat_base() + ES_LUT_LEN + (tr.mat - 1u) * ES_MAT_STRIDE;
    uint emissive_tex = asuint(es_tris[mb + 10]);
    if (emissive_tex == 0u) { return tr.emission; }
    float3 tc = tr.tc0 * (1.0 - b0 - b1) + tr.tc1 * b0 + tr.tc2 * b1;
    return tr.emission * es_sample(emissive_tex - 1u, tc);
}

// --- glTF 2.0 metallic-roughness (Appendix B) -------------------------------------------------

void es_lobes(EsSurface s, out float3 c_diff, out float3 f0, out float a) {
    float k = 1.0 - s.metallic;
    c_diff = s.base * k;
    f0 = float3(0.04 * k + s.base.x * s.metallic,
                0.04 * k + s.base.y * s.metallic,
                0.04 * k + s.base.z * s.metallic);
    a = s.roughness * s.roughness;
}

// f(l, v), no cosine: (1 - F) c_diff / pi + F D V.
float3 es_brdf(EsSurface s, float3 n, float3 v, float3 l) {
    float nl = es_dot3(n, l);
    float nv = es_dot3(n, v);
    if (nl <= 0.0 || nv <= 0.0) { return float3(0.0, 0.0, 0.0); }
    float3 c_diff;
    float3 f0;
    float a;
    es_lobes(s, c_diff, f0, a);
    float3 h = es_norm3(float3(v.x + l.x, v.y + l.y, v.z + l.z));
    float nh = max(es_dot3(n, h), 0.0);
    float vh = max(es_dot3(v, h), 0.0);
    float a2 = a * a;
    float dd = nh * nh * (a2 - 1.0) + 1.0;
    float dist = a2 / (ES_PI * dd * dd);
    float gv = nl * es_sqrt(nv * nv * (1.0 - a2) + a2);
    float gl = nv * es_sqrt(nl * nl * (1.0 - a2) + a2);
    float vis = 0.5 / (gv + gl);
    float m = 1.0 - vh;
    float m5 = m * m * m * m * m;
    float3 f = float3(f0.x + (1.0 - f0.x) * m5, f0.y + (1.0 - f0.y) * m5, f0.z + (1.0 - f0.z) * m5);
    // The Lambert lobe keeps (1 - F(n.l)) (1 - F(n.v)): reciprocal, and <= 1 in the furnace.
    float ml = 1.0 - nl;
    float mv = 1.0 - nv;
    float ml5 = ml * ml * ml * ml * ml;
    float mv5 = mv * mv * mv * mv * mv;
    float3 fl = float3(f0.x + (1.0 - f0.x) * ml5, f0.y + (1.0 - f0.y) * ml5, f0.z + (1.0 - f0.z) * ml5);
    float3 fv = float3(f0.x + (1.0 - f0.x) * mv5, f0.y + (1.0 - f0.y) * mv5, f0.z + (1.0 - f0.z) * mv5);
    return float3((1.0 - fl.x) * (1.0 - fv.x) * c_diff.x * ES_INV_PI + f.x * dist * vis,
                  (1.0 - fl.y) * (1.0 - fv.y) * c_diff.y * ES_INV_PI + f.y * dist * vis,
                  (1.0 - fl.z) * (1.0 - fv.z) * c_diff.z * ES_INV_PI + f.z * dist * vis);
}

float es_lum3(float3 c) { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

float es_p_spec(EsSurface s, float nv) {
    float3 c_diff;
    float3 f0;
    float a;
    es_lobes(s, c_diff, f0, a);
    float m = 1.0 - max(nv, 0.0);
    float m5 = m * m * m * m * m;
    float3 f = float3(f0.x + (1.0 - f0.x) * m5, f0.y + (1.0 - f0.y) * m5, f0.z + (1.0 - f0.z) * m5);
    float ws = es_lum3(f);
    float wd = es_lum3(c_diff);
    return (ws + wd > 0.0) ? ws / (ws + wd) : 0.5;
}

// Solid-angle pdf of es_bsdf_sample producing l: the lobe mixture.
float es_bsdf_pdf(EsSurface s, float3 n, float3 v, float3 l) {
    float nl = es_dot3(n, l);
    float nv = es_dot3(n, v);
    if (nl <= 0.0 || nv <= 0.0) { return 0.0; }
    float ps = es_p_spec(s, nv);
    float a = s.roughness * s.roughness;
    float a2 = a * a;
    float3 h = es_norm3(float3(v.x + l.x, v.y + l.y, v.z + l.z));
    float nh = max(es_dot3(n, h), 0.0);
    float dd = nh * nh * (a2 - 1.0) + 1.0;
    float dist = a2 / (ES_PI * dd * dd);
    float g1 = 2.0 * nv / (nv + es_sqrt(a2 + (1.0 - a2) * nv * nv));
    float spec = g1 * dist / (4.0 * nv);
    return ps * spec + (1.0 - ps) * nl * ES_INV_PI;
}

// A direction from the lobe mixture (Heitz 2018 for the GGX visible normals); the diffuse
// lobe takes `cosine`, the caller's stream-0 cosine-weighted direction.
float3 es_bsdf_sample(EsSurface s, float3 n, float3 v, float3 cosine, float3 u, float3 t, float3 b) {
    float nv = es_dot3(n, v);
    if (u.x >= es_p_spec(s, nv)) { return cosine; }
    float a = s.roughness * s.roughness;
    float3 vl = float3(es_dot3(v, t), es_dot3(v, b), nv);
    float3 ve = es_norm3(float3(a * vl.x, a * vl.y, vl.z));
    float lensq = ve.x * ve.x + ve.y * ve.y;
    float3 t1 = lensq > 0.0 ? float3(-ve.y, ve.x, 0.0) * es_rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 t2 = es_cross(ve, t1);
    float r = es_sqrt(u.y);
    float phi = 2.0 * ES_PI * u.z;
    float p1 = r * es_cos(phi);
    float sm = 0.5 * (1.0 + ve.z);
    float p2 = (1.0 - sm) * es_sqrt(max(1.0 - p1 * p1, 0.0)) + sm * (r * es_sin(phi));
    float pz = es_sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
    float3 nh = float3(p1 * t1.x + p2 * t2.x + pz * ve.x,
                       p1 * t1.y + p2 * t2.y + pz * ve.y,
                       p1 * t1.z + p2 * t2.z + pz * ve.z);
    float3 ne = es_norm3(float3(a * nh.x, a * nh.y, max(nh.z, 0.0)));
    float3 h = float3(t.x * ne.x + b.x * ne.y + n.x * ne.z,
                      t.y * ne.x + b.y * ne.y + n.y * ne.z,
                      t.z * ne.x + b.z * ne.y + n.z * ne.z);
    float vh = es_dot3(v, h);
    return float3(2.0 * vh * h.x - v.x, 2.0 * vh * h.y - v.y, 2.0 * vh * h.z - v.z);
}

#endif
