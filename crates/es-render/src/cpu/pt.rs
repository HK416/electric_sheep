//! The `Pt` path (spec 15.3): the path tracer with next-event estimation and MIS, one BSDF
//! bounce per vertex, and the per-pixel temporal accumulation of packet M7/R4. `pt.slang` and
//! `accumulate.slang` are the line-for-line mirrors.

use std::collections::BTreeMap;

use es_math::approx;
use es_sensor::Channel;

use super::geometry::{any_hit, face_forward, nearest_hit, primary_dir, tri_area, tri_point};
use super::restir::restir_di;
use super::shade::tonemap_to_u8;
use super::svgf::{atrous, variance_estimate};
use super::{
    add, dot, g_buffer, geometry_channels, luminance, mul, normalize, scale, sub, view_bits, Frame,
    History, RAY_EPS, SHADOW_FAR,
};
use crate::atlas::{Tile, TileData};
use crate::bvh::Bvh;
use crate::material::Surface;
use crate::rng;
use crate::scene::{Tri, TriScene};
use crate::view::{CameraView, RenderConfig, RenderPath, ViewParams};

/// Power heuristic with `beta = 2` (PBR 4e 13.10), over two solid-angle pdfs. `0` when both
/// are zero, so a strategy that cannot have produced the sample contributes nothing.
fn power_heuristic(a: f32, b: f32) -> f32 {
    let (a2, b2) = (a * a, b * b);
    let d = a2 + b2;
    if d > 0.0 {
        a2 / d
    } else {
        0.0
    }
}

/// Orthonormal basis around `n` (Duff et al., branchless).
fn onb(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let sign = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n[2]);
    let b = n[0] * n[1] * a;
    (
        [1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]],
        [b, sign + n[1] * n[1] * a, -n[1]],
    )
}

/// Cosine-weighted hemisphere sample around `n`. Cosine weighting cancels the `cos` term and
/// the `1/pi` of a Lambertian BRDF, so the throughput update is a plain multiply by albedo —
/// no pdf division and no 0/0.
pub(super) fn cosine_hemisphere(n: [f32; 3], key: u32) -> [f32; 3] {
    let u1 = rng::uniform(key, 0);
    let u2 = rng::uniform(key, 1);
    let r = approx::sqrt(u1);
    let phi = 2.0 * std::f32::consts::PI * u2;
    let (t, b) = onb(n);
    let z = approx::sqrt((1.0 - u1).max(0.0));
    normalize(add(
        add(
            scale(t, r * approx::cos(phi)),
            scale(b, r * approx::sin(phi)),
        ),
        scale(n, z),
    ))
}

/// Solid-angle pdf of the light-sampling strategy for `light`, seen from a shading point
/// `dist` away along `dir` (packet M7/R3): uniform over the `n_lights` emissive triangles,
/// then uniform over the chosen triangle's area, converted to solid angle by
/// `d^2 / |cos_l|`. `0` when the strategy cannot produce that direction at all.
fn light_pdf(light: &Tri, dir: [f32; 3], dist2: f32, n_lights: f32) -> f32 {
    let cos_l = dot(light.n, scale(dir, -1.0)).abs();
    let area = tri_area(light);
    if cos_l > 0.0 && area > 0.0 && n_lights > 0.0 {
        dist2 / (cos_l * area * n_lights)
    } else {
        0.0
    }
}

/// Next-event estimation at one diffuse hit, MIS-weighted against the BSDF strategy
/// (packet M7/R3; PBR 4e 13.10). Returns the radiance to add *before* the throughput is
/// multiplied by the albedo — the `f = albedo / pi` here is this surface's BRDF.
///
/// Three light kinds, each a fixed amount of work for every pixel of a given render (spec
/// 3.4 forbids a data-dependent loop bound; the three `if`s below are on *config*, uniform
/// across the dispatch, not on the pixel):
///
/// 1. **emissive triangles** — one pick uniform over the light list, then one uniform point
///    on that triangle, one shadow ray, power-heuristic MIS against the BSDF strategy;
/// 2. **the directional light** — a delta distribution, so no MIS is possible or needed:
///    one shadow ray and the full contribution;
/// 3. **the sky** — a cosine-weighted hemisphere direction and a shadow ray that has to
///    *escape*. Its pdf is the BSDF's, so the power heuristic gives exactly 1/2 to each and
///    the pair is a two-sample estimate of the same integral.
///
/// `keys` are the three RNG streams of `rng::key`'s table: 4 (light pick), 5 (area sample),
/// 6 (sky direction).
///
/// `last` is the bounce budget's final vertex, where **the MIS weights are dropped**: no
/// BSDF continuation is traced from there, so the complementary `w_bsdf` share would be lost
/// rather than estimated elsewhere, and NEE has to carry the whole term. With it, next-event
/// estimation at `B` bounces is an estimator of exactly what the BSDF-only path tracer
/// estimates at `B + 1` — which is what `nee_converges_to_the_same_image` compares, and
/// without it the two differ by ~4% of the mean radiance on Cornell.
#[allow(clippy::too_many_arguments)]
fn nee_direct(
    scene: &TriScene,
    bvh: &Bvh,
    cfg: &RenderConfig,
    p: [f32; 3],
    n: [f32; 3],
    surf: &Surface,
    v: [f32; 3],
    far: f32,
    last: bool,
    keys: [u32; 3],
) -> [f32; 3] {
    // The BRDF towards `l` and the BSDF strategy's pdf of `l` (the MIS partner). A Lambertian
    // surface is today's `albedo / pi` and `cos / pi`, the same expressions in the same order,
    // so every committed `Pt` frame is where it was; a PBR one is the glTF lobe mixture
    // (plan H, HT1).
    let lambert = scale(surf.base, std::f32::consts::FRAC_1_PI);
    let f = |l: [f32; 3]| {
        if surf.pbr {
            surf.brdf(n, v, l)
        } else {
            lambert
        }
    };
    let bsdf_pdf = |l: [f32; 3], cos_s: f32| {
        if surf.pbr {
            surf.pdf(n, v, l)
        } else {
            cos_s * std::f32::consts::FRAC_1_PI
        }
    };
    let mut out = [0.0f32; 3];

    if !scene.lights.is_empty() {
        let n_lights = scene.lights.len() as f32;
        let pick = rng::uniform(keys[0], 0);
        let idx = ((pick * n_lights) as usize).min(scene.lights.len() - 1);
        let light = &scene.tris[scene.lights[idx] as usize];
        let (lp, b0, b1) = tri_point(light, rng::uniform(keys[1], 0), rng::uniform(keys[1], 1));
        let seg = sub(lp, p);
        let dist2 = dot(seg, seg);
        let dist = approx::sqrt(dist2);
        if dist > 0.0 {
            let dir = scale(seg, 1.0 / dist);
            let cos_s = dot(n, dir);
            let p_light = light_pdf(light, dir, dist2, n_lights);
            if cos_s > 0.0 && p_light > 0.0 {
                let w = if last {
                    1.0
                } else {
                    power_heuristic(p_light, bsdf_pdf(dir, cos_s))
                };
                // `dist * (1 - 1e-3)` so the shadow ray stops short of the light itself.
                if !any_hit(&scene.tris, bvh, p, dir, 0.0, dist * (1.0 - 1e-3)) {
                    let le = scene.materials.emission_at(light, b0, b1);
                    out = add(out, scale(mul(f(dir), le), cos_s / p_light * w));
                }
            }
        }
    }

    if cfg.light_rgb.iter().any(|c| *c > 0.0) {
        let l = [
            cfg.light_dir.x as f32,
            cfg.light_dir.y as f32,
            cfg.light_dir.z as f32,
        ];
        let cos_s = dot(n, l);
        if cos_s > 0.0 && !any_hit(&scene.tris, bvh, p, l, 0.0, SHADOW_FAR) {
            out = add(out, scale(mul(f(l), cfg.light_rgb), cos_s));
        }
    }

    if cfg.sky.iter().any(|c| *c > 0.0) {
        let d = cosine_hemisphere(n, keys[2]);
        let cos_s = dot(n, d);
        let pdf = cos_s * std::f32::consts::FRAC_1_PI;
        if pdf > 0.0 && !any_hit(&scene.tris, bvh, p, d, 0.0, far) {
            let w = if last {
                1.0
            } else {
                power_heuristic(pdf, bsdf_pdf(d, cos_s))
            };
            out = add(out, scale(mul(f(d), cfg.sky), cos_s / pdf * w));
        }
    }
    out
}

/// One BSDF bounce at a hit (plan H, HT1): the new direction, the throughput factor, the
/// solid-angle pdf of the direction (for the MIS weight of whatever the ray hits next) and
/// the sky strategy's pdf of it (`cos / pi`, NEE's sky sample).
///
/// A Lambertian surface is today's bounce exactly: a cosine-weighted direction from stream
/// 0, a throughput factor of `base` (the cosine and the `1/pi` cancel), and both pdfs the one
/// `cos / pi` expression. A PBR surface picks a lobe with stream 7 index 0, samples it
/// (stream 0 for the diffuse lobe, stream 7 indices 1 and 2 for GGX's visible normals) and
/// weights by `f * cos / pdf` over the lobe mixture; a direction below the surface ends the
/// path's throughput at zero rather than its loop (spec 3.4: the bounce count stays fixed).
///
/// `n` is the shading normal; `ng` the geometric one. A normal-mapped surface's direction below
/// the geometric surface is a zero-throughput one (plan H, HT2), which on any other surface
/// never happens, since there the two normals are the same.
fn bounce_dir(
    surf: &Surface,
    n: [f32; 3],
    ng: [f32; 3],
    v: [f32; 3],
    key_dir: u32,
    key_lobe: u32,
) -> ([f32; 3], [f32; 3], f32, f32) {
    let (l, throughput, pdf, sky) = bounce_lobe(surf, n, v, key_dir, key_lobe);
    if surf.normal.is_some() && dot(ng, l) <= 0.0 {
        return (l, [0.0; 3], pdf, sky);
    }
    (l, throughput, pdf, sky)
}

fn bounce_lobe(
    surf: &Surface,
    n: [f32; 3],
    v: [f32; 3],
    key_dir: u32,
    key_lobe: u32,
) -> ([f32; 3], [f32; 3], f32, f32) {
    let cosine = cosine_hemisphere(n, key_dir);
    if !surf.pbr {
        let pdf = dot(n, cosine) * std::f32::consts::FRAC_1_PI;
        return (cosine, surf.base, pdf, pdf);
    }
    let u = [0, 1, 2].map(|i| rng::uniform(key_lobe, i));
    let l = normalize(surf.sample(n, v, cosine, u, onb(n)));
    let nl = dot(n, l);
    let pdf = surf.pdf(n, v, l);
    let throughput = if pdf > 0.0 && nl > 0.0 {
        scale(surf.brdf(n, v, l), nl / pdf)
    } else {
        [0.0; 3]
    };
    (
        l,
        throughput,
        pdf,
        nl.max(0.0) * std::f32::consts::FRAC_1_PI,
    )
}

/// Path-trace one view (spec 15.3 `PT`, spec 1.9 item 2), with no history: one frame
/// standing alone, which is what every golden but `cornell_pt_accum8_rgb8` pins.
pub fn path_trace(
    scene: &TriScene,
    view: &CameraView,
    cfg: &RenderConfig,
    view_index: u32,
) -> Frame {
    path_trace_accum(scene, view, cfg, view_index, &mut History::default())
}

/// [`path_trace`] keeping [`crate::Temporal`]'s per-pixel history in `history` (packet
/// M7/R4). Call it once per frame with the same `history` and the same camera and the
/// samples accumulate; `cfg.temporal` at `None` makes every frame start over, which is
/// [`path_trace`] exactly.
///
/// The mirror of `pt.slang`'s `main` plus `accumulate.slang`: the accumulator **starts** at
/// the history sum and the frame's samples are added onto it in the loop's own order, so
/// `N` frames of `spp` samples are bit for bit one frame of `N * spp` — see
/// `accumulation_of_n_frames_is_n_spp`.
pub fn path_trace_accum(
    scene: &TriScene,
    view: &CameraView,
    cfg: &RenderConfig,
    view_index: u32,
    history: &mut History,
) -> Frame {
    let vp = ViewParams::new(view);
    let (w, h) = (view.spec.width, view.spec.height);
    let n_px = (w as usize) * (h as usize);
    let bvh = Bvh::build(&scene.tris);
    let g = g_buffer(scene, &bvh, &vp, w, h);
    let (spp, bounces) = (cfg.spp().max(1), cfg.bounces().max(1));
    let (restir, svgf) = match cfg.path {
        RenderPath::Pt { restir, svgf, .. } => (restir, svgf),
        RenderPath::Rs => (false, false),
    };
    let nee = cfg.nee();
    let n_lights = scene.lights.len() as f32;

    // The slot is kept only for a camera that is bitwise the one the history was built with
    // (no reprojection: a moved camera invalidates every pixel at once).
    let max_history = cfg.max_history();
    let bits = view_bits(&vp);
    if max_history.is_none() || history.view != Some(bits) || history.n.len() != n_px {
        history.reset(n_px);
    }
    history.view = Some(bits);
    let max_h = max_history.unwrap_or(1).max(1);
    // The sample base of this frame. It is the slot's frame counter, which *is* the pixel's
    // history length everywhere the history was never dropped — the case the bitwise oracle
    // pins. Where it was dropped the pixel restarts its average but keeps drawing forward, so
    // a pixel at the `max_history` clamp never redraws the samples it already holds.
    let base = history.frame;

    let mut radiance = vec![0.0f32; n_px * 3];
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) as usize;
            // Disocclusion: the history survives only where this pixel's primary hit is
            // bitwise the previous frame's.
            let n_prev = if history.n[i] > 0 && history.same_geometry(i, &g) {
                history.n[i]
            } else {
                0
            };
            // At the cap the oldest frame's share is scaled out of the sum rather than the
            // whole history being thrown away: an exponential moving average with
            // `alpha = 1 / max_history`, and the exact sum below the cap.
            let keep = n_prev.min(max_h - 1);
            let k = if keep == n_prev {
                1.0
            } else {
                keep as f32 / n_prev as f32
            };
            let mut acc = if keep == 0 {
                [0.0f32; 3]
            } else {
                scale(
                    [
                        history.sum[i * 3],
                        history.sum[i * 3 + 1],
                        history.sum[i * 3 + 2],
                    ],
                    k,
                )
            };
            let acc0 = acc;
            for s in 0..spp {
                // The sample index of packet M7/R4: frame `base` draws `base * spp + s`, so
                // `N` frames of `spp` draw what one frame of `N * spp` draws, in order.
                let s = base.wrapping_mul(spp).wrapping_add(s);
                let mut throughput = [1.0f32; 3];
                let mut o = vp.pos;
                let mut d = primary_dir(&vp, px, py);
                let mut near = vp.near;
                // Solid-angle pdf of the BSDF sample that produced the current ray. The
                // camera ray has none — no light-sampling strategy could have generated it,
                // so its MIS weight is 1 and `cornell_pt1spp` does not move.
                let mut prev_pdf = 0.0f32;
                // The sky strategy's pdf of the current ray; `prev_pdf` itself on a Lambertian
                // vertex, where the two strategies draw from the same cosine pdf.
                let mut prev_sky = 0.0f32;
                for bounce in 0..bounces {
                    let Some(hit) = nearest_hit(&scene.tris, &bvh, o, d, near, vp.far) else {
                        // The sky through the BSDF strategy. Its NEE counterpart draws from
                        // the cosine pdf, so on a Lambertian vertex the power heuristic splits
                        // it in half.
                        let w = if nee && bounce > 0 {
                            power_heuristic(prev_pdf, prev_sky)
                        } else {
                            1.0
                        };
                        acc = add(acc, scale(mul(throughput, cfg.sky), w));
                        break;
                    };
                    let tri = &scene.tris[hit.tri as usize];
                    // The surface this ray sees: its emission is the triangle's, times the
                    // emissive texel under a map (plan H, HT2).
                    let surf = scene.materials.surface(tri, o, d);
                    // An emissive hit reached by a BSDF bounce: the light-sampling strategy
                    // could have produced it too, so it is MIS-weighted against that pdf.
                    let w_em = if nee && bounce > 0 {
                        let p_light = light_pdf(tri, d, hit.t * hit.t, n_lights);
                        power_heuristic(prev_pdf, p_light)
                    } else {
                        1.0
                    };
                    acc = add(acc, scale(mul(throughput, surf.emission), w_em));
                    let ng = face_forward(tri, d);
                    let p = add(add(o, scale(d, hit.t)), scale(ng, RAY_EPS));
                    let n = surf.normal.unwrap_or(ng);
                    // The view direction a PBR lobe needs; a Lambertian one never reads it.
                    let v = if surf.pbr {
                        normalize(scale(d, -1.0))
                    } else {
                        [0.0; 3]
                    };
                    let key = |stream| rng::key(cfg.seed, view_index, px, py, s, bounce, stream);
                    if nee {
                        let direct = nee_direct(
                            scene,
                            &bvh,
                            cfg,
                            p,
                            n,
                            &surf,
                            v,
                            vp.far,
                            bounce + 1 == bounces,
                            [key(4), key(5), key(6)],
                        );
                        acc = add(acc, mul(throughput, direct));
                    }
                    let (next, weight, pdf, sky_pdf) = bounce_dir(&surf, n, ng, v, key(0), key(7));
                    throughput = mul(throughput, weight);
                    o = p;
                    d = next;
                    prev_pdf = pdf;
                    prev_sky = sky_pdf;
                    near = 0.0;
                }
            }
            // Sequential accumulation in ascending sample order, and one divide at the end:
            // the order is fixed by the loop, so the cheap sum is also the reproducible one
            // (spec 18.4 is for reductions whose order is not). `n` is 1 without a history,
            // which is `1 / spp` — today's bytes.
            let n = keep + 1;
            // This frame's own contribution, for the luminance moments. Taken as a difference
            // rather than a second accumulator so the sum above stays one unbroken chain; it
            // is exact to ~n ULP, and it feeds a variance estimate, not an image.
            let l = luminance(scale(sub(acc, acc0), 1.0 / spp as f32));
            let (m1, m2) = if keep == 0 {
                (0.0, 0.0)
            } else {
                (history.moments[i * 2] * k, history.moments[i * 2 + 1] * k)
            };
            history.moments[i * 2] = m1 + l;
            history.moments[i * 2 + 1] = m2 + l * l;
            history.sum[i * 3..i * 3 + 3].copy_from_slice(&acc);
            history.n[i] = n;
            history.depth[i] = g.depth[i];
            history.tri[i] = g.tri[i];
            history.normal[i * 3..i * 3 + 3].copy_from_slice(&g.normal[i * 3..i * 3 + 3]);
            radiance[i * 3..i * 3 + 3].copy_from_slice(&scale(acc, 1.0 / (n * spp) as f32));
        }
    }
    history.frame = base.wrapping_add(1);
    // The variance of the accumulated estimate, which the a-trous pass reads. Computed
    // whenever a history is kept, so `variance_falls_with_history` can read it with the
    // filter off.
    if max_history.is_some() {
        history.variance = variance_estimate(&radiance, history, &g, w, h);
    }

    if restir {
        radiance = restir_di(scene, &bvh, &vp, cfg, &g, w, h, view_index);
    }
    if svgf {
        let variance = max_history.map(|_| history.variance.as_slice());
        let (color, var) = atrous(
            &radiance,
            variance,
            &g.depth,
            &g.normal,
            w,
            h,
            cfg.svgf_iterations,
        );
        radiance = color;
        if max_history.is_some() {
            history.variance = var;
        }
    }

    let mut frame = Frame {
        width: w,
        height: h,
        channels: BTreeMap::new(),
    };
    // The tone-mapped `Rgb8` (packet M7/R3), before `radiance` is moved into the tile.
    if cfg.channels.contains(&Channel::Rgb8) {
        let mut rgb = vec![0u8; (w as usize) * (h as usize) * 3];
        for i in 0..(w as usize) * (h as usize) {
            let lin = [radiance[i * 3], radiance[i * 3 + 1], radiance[i * 3 + 2]];
            rgb[i * 3..i * 3 + 3].copy_from_slice(&tonemap_to_u8(lin, cfg.exposure, cfg.tonemap));
        }
        frame.channels.insert(
            Channel::Rgb8,
            Tile {
                shape: [h as usize, w as usize, 3],
                data: TileData::U8(rgb),
            },
        );
    }
    if cfg.channels.contains(&Channel::PtRadiance) {
        frame.channels.insert(
            Channel::PtRadiance,
            Tile {
                shape: [h as usize, w as usize, 3],
                data: TileData::F32(radiance),
            },
        );
    }
    // The history length after this frame (packet M7/R4): 1 everywhere without a history,
    // which is the truth — the estimate rests on this frame and nothing else.
    if cfg.channels.contains(&Channel::History) {
        frame.channels.insert(
            Channel::History,
            Tile {
                shape: [h as usize, w as usize, 1],
                data: TileData::U32(history.n.clone()),
            },
        );
    }
    geometry_channels(&g, cfg, w, h, &mut frame);
    frame
}
