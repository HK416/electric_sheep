// ReSTIR DI (spec 28.6), minimal and textbook. Three entry points, three dispatches, one
// reservoir buffer written per pass — no in-place update, so no read/write hazard and no
// dependence on dispatch order within a pass.
//
// Since packet M7/R3 the spatial reuse is UNBIASED: pairwise MIS (Wyman et al. 2023 section
// 5) replaces the biased 1/M combination, and the visibility test moved from the survivor of
// the initial pass to the finally selected sample at the destination. Mirror of
// `cpu::restir_di`; where the two texts diverge this file is wrong.
//
// Skipped, still: ReSTIR GI entirely (this is direct lighting only), light types other than
// emissive triangles, and reservoir ageing beyond the M clamp. Temporal reuse reads the
// previous frame at the SAME pixel — no motion-vector reprojection — so its two candidates
// share one domain and 1/M is already the right weight there; it is correct only for a static
// camera, and on the first frame the previous buffer is empty and the pass is a no-op.
// See docs/design/renderer.md.
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

// Primary hit reconstructed from the g-buffer, at any atlas index with its own tile-local
// pixel: the spatial pass needs a *neighbour's* shading point too, because pairwise MIS
// evaluates that neighbour's own target function.
bool es_hit_at(uint index, uint2 local, EsView vw, out float3 shade_p, out float3 n,
               out float3 albedo) {
    shade_p = float3(0.0, 0.0, 0.0);
    n = float3(0.0, 0.0, 1.0);
    albedo = float3(0.0, 0.0, 0.0);
    if (g_tri[index] == 0) { return false; }
    EsTri tr = es_tri(g_tri[index] - 1);
    float3 d = es_primary_dir(vw, local);
    n = es_face_forward(tr.n, d);
    shade_p = vw.pos + d * g_depth[index] + n * ES_RAY_EPS;
    albedo = tr.albedo;
    return true;
}

bool es_hit_of(EsPixel p, EsView vw, out float3 shade_p, out float3 n, out float3 albedo) {
    return es_hit_at(p.index, p.local, vw, shade_p, n, albedo);
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
    // No shadow ray here: visibility is tested once, in `spatial`, on the sample that pass
    // actually selects — which is what makes the reuse unbiased (packet M7/R3).
    r.w = (p_hat > 0.0 && r.m > 0.0) ? r.w_sum / (r.m * p_hat) : 0.0;
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

    // The neighbours that pass the geometric similarity test, each with its own shading
    // point: pairwise MIS evaluates the neighbour's own target function, which the reservoir
    // alone does not carry. Fixed offsets, not RNG-jittered, and clamped to the tile so a
    // neighbour is never another camera's pixel.
    uint nb_index[4];
    float3 nb_p[4];
    float3 nb_n[4];
    float3 nb_a[4];
    uint nb_slot[4];
    uint n_nbr_i = 0;
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
        if (es_res_load(res_a, j).m <= 0.0) { continue; }
        float3 pj;
        float3 nj;
        float3 aj;
        if (!es_hit_at(j, uint2(uint(nx), uint(ny)), vw, pj, nj, aj)) { continue; }
        nb_index[n_nbr_i] = j;
        nb_p[n_nbr_i] = pj;
        nb_n[n_nbr_i] = nj;
        nb_a[n_nbr_i] = aj;
        nb_slot[n_nbr_i] = k + 1;
        n_nbr_i += 1;
    }
    float n_nbr = float(n_nbr_i);

    // Pairwise MIS (Wyman et al. 2023 section 5), with the centre as the canonical technique:
    //   m_i(X) = (1/N) * (M_i p_i(X)) / (M_i p_i(X) + M_c p_c(X))
    //   m_c(X) = (1/N) * sum_i (M_c p_c(X)) / (M_i p_i(X) + M_c p_c(X))
    // which sum to one term by term. The resampling weight is `m * p_destination(X) * W`.
    EsRes centre = es_res_load(res_a, p.index);
    EsRes combined = es_res_zero();
    float m_sum = 0.0;
    if (centre.m > 0.0) {
        float p_c;
        es_di(shade_p, n, albedo, centre, p_c);
        float m_c = 0.0;
        for (uint i = 0; i < n_nbr_i; ++i) {
            float p_j;
            es_di(nb_p[i], nb_n[i], nb_a[i], centre, p_j);
            float a = centre.m * p_c;
            float b = es_res_load(res_a, nb_index[i]).m * p_j;
            if (a + b > 0.0) { m_c += a / (a + b); }
        }
        // No neighbour to share with: the canonical technique is the only one, weight 1.
        m_c = n_nbr > 0.0 ? m_c / n_nbr : 1.0;
        es_res_update(combined, centre.tri, centre.u, centre.v, m_c * p_c * centre.w,
                      es_rng_uniform(key, 0));
        m_sum += centre.m;
    }
    for (uint i = 0; i < n_nbr_i; ++i) {
        EsRes src = es_res_load(res_a, nb_index[i]);
        float p_at_src;
        es_di(nb_p[i], nb_n[i], nb_a[i], src, p_at_src);
        float p_at_dst;
        es_di(shade_p, n, albedo, src, p_at_dst);
        float a = src.m * p_at_src;
        float b = centre.m * p_at_dst;
        float m_i = (a + b > 0.0) ? a / (a + b) / n_nbr : 0.0;
        es_res_update(combined, src.tri, src.u, src.v, m_i * p_at_dst * src.w,
                      es_rng_uniform(key, nb_slot[i]));
        m_sum += src.m;
    }

    float p_hat;
    float3 radiance = es_di(shade_p, n, albedo, combined, p_hat);
    combined.w = (p_hat > 0.0) ? combined.w_sum / p_hat : 0.0;
    combined.m = min(m_sum, ES_RESTIR_M_MAX);
    // The reservoir keeps the UNSHADOWED W, so temporal reuse next frame still reuses the
    // quantity the target function is defined over. Only this pixel's output is shadowed, by
    // one ray towards the sample finally selected — at this pixel, not at whichever one
    // generated it.
    es_res_store(p.index, combined);

    float vis = combined.w;
    if (vis > 0.0) {
        EsTri light = es_tri(combined.tri);
        float3 seg = es_tri_point(light, combined.u, combined.v) - shade_p;
        float dist = es_sqrt(dot(seg, seg));
        if (es_any_hit(shade_p, seg * (1.0 / dist), 0.0, dist * (1.0 - 1e-3))) { vis = 0.0; }
    }
    float3 shaded = radiance * vis + es_tri(g_tri[p.index] - 1).emission;
    out_radiance[p.index * 3 + 0] = shaded.x;
    out_radiance[p.index * 3 + 1] = shaded.y;
    out_radiance[p.index * 3 + 2] = shaded.z;
}
