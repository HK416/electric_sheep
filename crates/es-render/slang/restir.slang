// ReSTIR DI (spec 28.6), minimal and textbook. Three entry points, three dispatches, one
// reservoir buffer written per pass — no in-place update, so no read/write hazard and no
// dependence on dispatch order within a pass.
//
// Skipped, and this is the list a real spec 28.6 packet fills in: MIS weights (the reuse is
// the biased 1/M combination, not GRIS pairwise MIS), the bias-correction visibility re-test
// on reuse, ReSTIR GI entirely (this is direct lighting only), light types other than
// emissive triangles, and reservoir ageing beyond the M clamp. Temporal reuse reads the
// previous frame at the SAME pixel — no motion-vector reprojection — so it is correct only
// for a static camera, and on the first frame the previous buffer is empty and the pass is a
// no-op. See docs/design/renderer.md.
//
// Reservoir stride: 8 floats — tri (asuint), u, v, w_sum, M, W, pad, pad.

#include "common.slang"

static const uint ES_RES_STRIDE = 8;
static const uint ES_RESTIR_CANDIDATES = 8;
static const float ES_RESTIR_M_CLAMP = 20.0;
static const float ES_RESTIR_M_MAX = 100.0;   // M_CLAMP * (1 + neighbours)

[[vk::binding(2, 0)]] RWStructuredBuffer<float> res_out;
[[vk::binding(3, 0)]] StructuredBuffer<float> res_a;
[[vk::binding(4, 0)]] StructuredBuffer<float> res_prev;
[[vk::binding(5, 0)]] StructuredBuffer<float> g_depth;
[[vk::binding(6, 0)]] StructuredBuffer<float> g_normal;
[[vk::binding(7, 0)]] StructuredBuffer<uint> g_tri;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> out_radiance;

struct EsRes {
    uint tri;
    float u;
    float v;
    float w_sum;
    float m;
    float w;
};

EsRes es_res_zero() {
    EsRes r;
    r.tri = 0; r.u = 0.0; r.v = 0.0; r.w_sum = 0.0; r.m = 0.0; r.w = 0.0;
    return r;
}

EsRes es_res_load(StructuredBuffer<float> buf, uint i) {
    uint b = i * ES_RES_STRIDE;
    EsRes r;
    r.tri = asuint(buf[b + 0]);
    r.u = buf[b + 1];
    r.v = buf[b + 2];
    r.w_sum = buf[b + 3];
    r.m = buf[b + 4];
    r.w = buf[b + 5];
    return r;
}

void es_res_store(uint i, EsRes r) {
    uint b = i * ES_RES_STRIDE;
    res_out[b + 0] = asfloat(r.tri);
    res_out[b + 1] = r.u;
    res_out[b + 2] = r.v;
    res_out[b + 3] = r.w_sum;
    res_out[b + 4] = r.m;
    res_out[b + 5] = r.w;
    res_out[b + 6] = 0.0;
    res_out[b + 7] = 0.0;
}

void es_res_update(inout EsRes r, uint tri, float u, float v, float weight, float rand) {
    r.w_sum += weight;
    r.m += 1.0;
    if (r.w_sum > 0.0 && rand * r.w_sum < weight) {
        r.tri = tri;
        r.u = u;
        r.v = v;
    }
}

// Unshadowed contribution and its target function (the luminance of that contribution).
float3 es_di(float3 shade_p, float3 n, float3 albedo, EsRes r, out float p_hat) {
    EsTri light = es_tri(r.tri);
    float3 lp = es_tri_point(light, r.u, r.v);
    float3 to_light = lp - shade_p;
    float dist2 = max(dot(to_light, to_light), 1e-8);
    float3 dir = to_light * (1.0 / es_sqrt(dist2));
    float cos_s = max(dot(n, dir), 0.0);
    float cos_l = abs(dot(light.n, -dir));
    float g = cos_s * cos_l / dist2 * es_tri_area(light) * float(es_n_lights());
    float3 radiance = (albedo * ES_INV_PI) * light.emission * g;
    p_hat = es_luminance(radiance);
    return radiance;
}

// Primary hit reconstructed from the g-buffer. Returns false for a background pixel.
bool es_hit_of(EsPixel p, EsView vw, out float3 shade_p, out float3 n, out float3 albedo) {
    shade_p = float3(0.0, 0.0, 0.0);
    n = float3(0.0, 0.0, 1.0);
    albedo = float3(0.0, 0.0, 0.0);
    if (g_tri[p.index] == 0) { return false; }
    EsTri tr = es_tri(g_tri[p.index] - 1);
    float3 d = es_primary_dir(vw, p.local);
    n = es_face_forward(tr.n, d);
    shade_p = vw.pos + d * g_depth[p.index] + n * ES_RAY_EPS;
    albedo = tr.albedo;
    return true;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void initial(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return; }
    EsPixel p = es_pixel(tid);
    es_res_store(p.index, es_res_zero());
    if (!p.valid || es_n_lights() == 0) { return; }

    EsView vw = es_view(p.view);
    float3 shade_p;
    float3 n;
    float3 albedo;
    if (!es_hit_of(p, vw, shade_p, n, albedo)) { return; }

    EsRes r = es_res_zero();
    uint key = es_rng_key(es_seed(), p.view, p.local.x, p.local.y, 0, 0, 1);
    for (uint c = 0; c < ES_RESTIR_CANDIDATES; ++c) {
        float pick = es_rng_uniform(key, c * 3);
        uint idx = min(uint(pick * float(es_n_lights())), es_n_lights() - 1);
        EsRes cand = es_res_zero();
        cand.tri = es_light_tri(idx);
        cand.u = es_rng_uniform(key, c * 3 + 1);
        cand.v = es_rng_uniform(key, c * 3 + 2);
        float p_hat;
        es_di(shade_p, n, albedo, cand, p_hat);
        es_res_update(r, cand.tri, cand.u, cand.v, p_hat, es_rng_uniform(key, 64 + c));
    }
    float p_hat;
    es_di(shade_p, n, albedo, r, p_hat);
    r.w = (p_hat > 0.0 && r.m > 0.0) ? r.w_sum / (r.m * p_hat) : 0.0;
    if (r.w > 0.0) {
        EsTri light = es_tri(r.tri);
        float3 seg = es_tri_point(light, r.u, r.v) - shade_p;
        float dist = es_sqrt(dot(seg, seg));
        if (es_any_hit(shade_p, seg * (1.0 / dist), 0.0, dist * (1.0 - 1e-3))) { r.w = 0.0; }
    }
    es_res_store(p.index, r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void temporal(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return; }
    EsPixel p = es_pixel(tid);
    EsRes cur = es_res_load(res_a, p.index);
    if (!p.valid) { es_res_store(p.index, cur); return; }

    EsRes old = es_res_load(res_prev, p.index);
    if (old.m <= 0.0) { es_res_store(p.index, cur); return; }

    EsView vw = es_view(p.view);
    float3 shade_p;
    float3 n;
    float3 albedo;
    if (!es_hit_of(p, vw, shade_p, n, albedo)) { es_res_store(p.index, cur); return; }

    uint key = es_rng_key(es_seed(), p.view, p.local.x, p.local.y, 0, 0, 3);
    EsRes merged = es_res_zero();
    float p_cur;
    es_di(shade_p, n, albedo, cur, p_cur);
    es_res_update(merged, cur.tri, cur.u, cur.v, p_cur * cur.w * cur.m, es_rng_uniform(key, 0));
    float m_sum = cur.m;

    float m_old = min(old.m, ES_RESTIR_M_CLAMP);
    float p_old;
    es_di(shade_p, n, albedo, old, p_old);
    es_res_update(merged, old.tri, old.u, old.v, p_old * old.w * m_old, es_rng_uniform(key, 1));
    m_sum += m_old;

    merged.m = m_sum;
    float p_hat;
    es_di(shade_p, n, albedo, merged, p_hat);
    merged.w = (p_hat > 0.0 && merged.m > 0.0) ? merged.w_sum / (merged.m * p_hat) : 0.0;
    es_res_store(p.index, merged);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void spatial(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= es_atlas_w() || tid.y >= es_atlas_h()) { return; }
    EsPixel p = es_pixel(tid);
    es_res_store(p.index, es_res_zero());
    out_radiance[p.index * 3 + 0] = 0.0;
    out_radiance[p.index * 3 + 1] = 0.0;
    out_radiance[p.index * 3 + 2] = 0.0;
    if (!p.valid) { return; }

    EsView vw = es_view(p.view);
    float3 shade_p;
    float3 n;
    float3 albedo;
    if (!es_hit_of(p, vw, shade_p, n, albedo)) { return; }

    uint key = es_rng_key(es_seed(), p.view, p.local.x, p.local.y, 0, 0, 2);
    EsRes combined = es_res_zero();
    float m_sum = 0.0;

    // Centre first, then the four fixed neighbours — fixed offsets, not RNG-jittered: jitter
    // buys less correlated noise and costs the ability to say the result is a function of the
    // pixel grid alone. Neighbours are clamped to the tile, never across into another camera.
    EsRes centre = es_res_load(res_a, p.index);
    if (centre.m > 0.0) {
        float p_hat;
        es_di(shade_p, n, albedo, centre, p_hat);
        es_res_update(combined, centre.tri, centre.u, centre.v,
                      p_hat * centre.w * centre.m, es_rng_uniform(key, 0));
        m_sum += centre.m;
    }

    int2 offsets[4] = { int2(3, 0), int2(-3, 0), int2(0, 3), int2(0, -3) };
    for (uint k = 0; k < 4; ++k) {
        int nx = int(p.local.x) + offsets[k].x;
        int ny = int(p.local.y) + offsets[k].y;
        if (nx < 0 || ny < 0 || nx >= int(es_tile_w()) || ny >= int(es_tile_h())) { continue; }
        uint2 atlas_xy = uint2(tid.x - p.local.x + uint(nx), tid.y - p.local.y + uint(ny));
        uint j = atlas_xy.y * es_atlas_w() + atlas_xy.x;
        // Standard geometric similarity: depth within 10%, normal within 25 degrees.
        float dz = abs(g_depth[p.index] - g_depth[j]);
        float nn = g_normal[p.index * 3 + 0] * g_normal[j * 3 + 0]
                 + g_normal[p.index * 3 + 1] * g_normal[j * 3 + 1]
                 + g_normal[p.index * 3 + 2] * g_normal[j * 3 + 2];
        if (dz > 0.1 * abs(g_depth[p.index]) || nn < 0.906) { continue; }
        EsRes nb = es_res_load(res_a, j);
        if (nb.m <= 0.0) { continue; }
        float p_hat;
        es_di(shade_p, n, albedo, nb, p_hat);
        es_res_update(combined, nb.tri, nb.u, nb.v, p_hat * nb.w * nb.m,
                      es_rng_uniform(key, k + 1));
        m_sum += nb.m;
    }

    combined.m = min(m_sum, ES_RESTIR_M_MAX);
    float p_hat;
    float3 radiance = es_di(shade_p, n, albedo, combined, p_hat);
    combined.w = (p_hat > 0.0 && combined.m > 0.0) ? combined.w_sum / (combined.m * p_hat) : 0.0;
    es_res_store(p.index, combined);

    float3 shaded = radiance * combined.w + es_tri(g_tri[p.index] - 1).emission;
    out_radiance[p.index * 3 + 0] = shaded.x;
    out_radiance[p.index * 3 + 1] = shaded.y;
    out_radiance[p.index * 3 + 2] = shaded.z;
}
