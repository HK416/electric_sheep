//! `ReSTIR` DI (packet M7/R3): RIS over the emissive triangles, spatial reuse combined with
//! pairwise MIS, and one shadow ray towards the sample finally selected. `restir.slang` is the
//! line-for-line mirror.

use es_math::approx;

use super::geometry::{any_hit, face_forward, primary_dir, tri_area, tri_point};
use super::{add, dot, luminance, mul, normalize, scale, sub, GBuffer, RAY_EPS};
use crate::bvh::Bvh;
use crate::material::Surface;
use crate::rng;
use crate::scene::TriScene;
use crate::view::{RenderConfig, ViewParams};

/// Direct-light candidates per pixel in the `ReSTIR` initial pass.
const RESTIR_CANDIDATES: u32 = 8;
/// `M` clamp on temporal reuse.
const RESTIR_M_CLAMP: f32 = 20.0;
/// Spatial-reuse neighbour offsets. Fixed, not RNG-jittered: jitter buys less correlated
/// noise and costs the ability to say the result is a function of the pixel grid alone.
const RESTIR_NEIGHBOURS: [(i32, i32); 4] = [(3, 0), (-3, 0), (0, 3), (0, -3)];

// --- `ReSTIR` DI ------------------------------------------------------------------------------

/// One direct-lighting reservoir. Mirrors the 8-float stride of `restir.slang`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reservoir {
    pub tri: u32,
    pub u: f32,
    pub v: f32,
    pub w_sum: f32,
    pub m: f32,
    pub w: f32,
}

impl Reservoir {
    fn update(&mut self, tri: u32, u: f32, v: f32, weight: f32, rand: f32) {
        self.w_sum += weight;
        self.m += 1.0;
        if self.w_sum > 0.0 && rand * self.w_sum < weight {
            self.tri = tri;
            self.u = u;
            self.v = v;
        }
    }
}

/// A spatial-reuse neighbour that passed the geometric similarity test: where its reservoir
/// is, its own shading point (pairwise MIS evaluates *its* target function, not only the
/// destination's) and which RNG slot its resampling draw takes.
/// A `ReSTIR` shading point: position, normal, and the surface with its view direction.
type ShadePoint = ([f32; 3], [f32; 3], (Surface, [f32; 3]));

struct Neighbour {
    index: usize,
    p: [f32; 3],
    n: [f32; 3],
    albedo: (Surface, [f32; 3]),
    slot: u32,
}

/// Unshadowed target function and the radiance it stands for.
fn di_contribution(
    scene: &TriScene,
    shade_p: [f32; 3],
    n: [f32; 3],
    albedo: (Surface, [f32; 3]),
    r: &Reservoir,
) -> ([f32; 3], f32) {
    let Some(light) = scene.tris.get(r.tri as usize) else {
        return ([0.0; 3], 0.0);
    };
    let (lp, b0, b1) = tri_point(light, r.u, r.v);
    let to_light = sub(lp, shade_p);
    let dist2 = dot(to_light, to_light).max(1e-8);
    let dir = scale(to_light, 1.0 / approx::sqrt(dist2));
    let cos_s = dot(n, dir).max(0.0);
    let cos_l = dot(light.n, scale(dir, -1.0)).abs();
    let g = cos_s * cos_l / dist2 * tri_area(light) * scene.lights.len() as f32;
    // The surface and the view direction at the shading point: a Lambertian surface is today's
    // `albedo / pi`, a PBR one the glTF BRDF towards the light sample (plan H, HT1).
    let (surf, v) = albedo;
    let f = if surf.pbr {
        surf.brdf(n, v, dir)
    } else {
        scale(surf.base, std::f32::consts::FRAC_1_PI)
    };
    let radiance = scale(mul(f, scene.materials.emission_at(light, b0, b1)), g);
    (radiance, luminance(radiance))
}

/// `ReSTIR` DI: initial candidates, temporal reuse, spatial reuse, then shade.
///
/// **The spatial reuse is unbiased** since packet M7/R3: the biased `1/M` combination is
/// replaced by pairwise MIS (Wyman et al. 2023 section 5), so a neighbour whose target
/// function is zero at the destination's sample — a light below its horizon, say — takes
/// zero weight instead of inflating the divisor. The visibility test moved with it, from the
/// survivor of the initial pass to the *finally selected* sample at the destination: still
/// one shadow ray per pixel, but now the estimator is `f_shadowed(y) * W(y)` with `W` built
/// from the unshadowed target, which is unbiased (`docs/design/renderer.md` section 10).
///
/// Skipped, still: `ReSTIR` GI entirely (this is direct lighting only), light types other
/// than emissive triangles, and reservoir ageing beyond the `M` clamp. Temporal reuse reads
/// the previous frame at the *same* pixel — no motion-vector reprojection — so its two
/// candidates share one domain and `1/M` is already the correct weight there; it is correct
/// only for a static camera, and on the first frame the previous buffer is empty and the
/// pass is a no-op.
#[allow(clippy::too_many_arguments)] // one more than seven: the BVH beside the triangles
pub(super) fn restir_di(
    scene: &TriScene,
    bvh: &Bvh,
    vp: &ViewParams,
    cfg: &RenderConfig,
    g: &GBuffer,
    w: u32,
    h: u32,
    view_index: u32,
) -> Vec<f32> {
    let n_px = (w as usize) * (h as usize);
    let mut initial = vec![Reservoir::default(); n_px];
    let mut spatial = vec![Reservoir::default(); n_px];
    let mut out = vec![0.0f32; n_px * 3];
    if scene.lights.is_empty() {
        return out;
    }

    let hit_of = |i: usize, px: u32, py: u32| -> Option<ShadePoint> {
        if g.tri[i] == 0 {
            return None;
        }
        let tri = &scene.tris[(g.tri[i] - 1) as usize];
        let d = primary_dir(vp, px, py);
        let ng = face_forward(tri, d);
        let p = add(add(vp.pos, scale(d, g.depth[i])), scale(ng, RAY_EPS));
        let surf = scene.materials.surface(tri, vp.pos, d);
        let n = surf.normal.unwrap_or(ng);
        let v = if surf.pbr {
            normalize(scale(d, -1.0))
        } else {
            [0.0; 3]
        };
        Some((p, n, (surf, v)))
    };

    // Pass 1: RIS over `RESTIR_CANDIDATES` candidates. No shadow ray: visibility is tested
    // once, at the end, on the sample the spatial pass actually selects.
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) as usize;
            let Some((p, n, albedo)) = hit_of(i, px, py) else {
                continue;
            };
            let mut r = Reservoir::default();
            let key = rng::key(cfg.seed, view_index, px, py, 0, 0, 1);
            for c in 0..RESTIR_CANDIDATES {
                let pick = rng::uniform(key, c * 3);
                let idx = ((pick * scene.lights.len() as f32) as usize).min(scene.lights.len() - 1);
                let cand = Reservoir {
                    tri: scene.lights[idx],
                    u: rng::uniform(key, c * 3 + 1),
                    v: rng::uniform(key, c * 3 + 2),
                    ..Reservoir::default()
                };
                let (_, p_hat) = di_contribution(scene, p, n, albedo, &cand);
                r.update(cand.tri, cand.u, cand.v, p_hat, rng::uniform(key, 64 + c));
            }
            let (_, p_hat) = di_contribution(scene, p, n, albedo, &r);
            r.w = if p_hat > 0.0 && r.m > 0.0 {
                r.w_sum / (r.m * p_hat)
            } else {
                0.0
            };
            initial[i] = r;
        }
    }

    // Pass 2 (temporal) is a no-op in the CPU reference: it has no previous frame to read.
    // The GPU renderer keeps one, and `Renderer::render` documents the same caveat.

    // Pass 3: spatial reuse over the four fixed neighbours, combined with **pairwise MIS**.
    //
    // For techniques {canonical c} u {neighbours 1..N} and any sample X, the weights
    //
    //   m_i(X) = (1/N) * (M_i p_i(X)) / (M_i p_i(X) + M_c p_c(X))
    //   m_c(X) = (1/N) * sum_i (M_c p_c(X)) / (M_i p_i(X) + M_c p_c(X))
    //
    // sum to one term by term, so they are valid MIS weights; `p_i` is neighbour `i`'s own
    // target function, evaluated at *its* shading point. The resampling weight of a
    // candidate is then `m * p_destination(X) * W`, and the combined contribution weight is
    // `w_sum / p_destination(Y)` with no `1/M` and no `1/Z` left to divide by.
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) as usize;
            let Some((p, n, albedo)) = hit_of(i, px, py) else {
                continue;
            };
            let key = rng::key(cfg.seed, view_index, px, py, 0, 0, 2);

            // The neighbours that pass the geometric similarity test, each with its own
            // shading point: pairwise MIS needs their target functions, not just their
            // reservoirs. Fixed offsets, ascending, so the set is a function of the grid.
            let mut nbrs: Vec<Neighbour> = Vec::new();
            for (k, (dx, dy)) in RESTIR_NEIGHBOURS.iter().enumerate() {
                let (nx, ny) = (px as i32 + dx, py as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let j = (ny as u32 * w + nx as u32) as usize;
                // Standard geometric similarity test: depth within 10%, normal within 25deg.
                let dz = (g.depth[i] - g.depth[j]).abs();
                let nn = g.normal[i * 3] * g.normal[j * 3]
                    + g.normal[i * 3 + 1] * g.normal[j * 3 + 1]
                    + g.normal[i * 3 + 2] * g.normal[j * 3 + 2];
                if dz > 0.1 * g.depth[i].abs() || nn < 0.906 || initial[j].m <= 0.0 {
                    continue;
                }
                let Some((pj, nj, aj)) = hit_of(j, nx as u32, ny as u32) else {
                    continue;
                };
                nbrs.push(Neighbour {
                    index: j,
                    p: pj,
                    n: nj,
                    albedo: aj,
                    slot: k as u32 + 1,
                });
            }
            let n_nbr = nbrs.len() as f32;
            let centre = initial[i];
            let mut combined = Reservoir::default();
            let mut m_sum = 0.0f32;

            if centre.m > 0.0 {
                let (_, p_c) = di_contribution(scene, p, n, albedo, &centre);
                let mut m_c = 0.0f32;
                for nb in &nbrs {
                    let (_, p_j) = di_contribution(scene, nb.p, nb.n, nb.albedo, &centre);
                    let (a, b) = (centre.m * p_c, initial[nb.index].m * p_j);
                    if a + b > 0.0 {
                        m_c += a / (a + b);
                    }
                }
                // No neighbour to share with: the canonical technique is the only one, and
                // its weight is 1 (which reproduces plain RIS at this pixel).
                m_c = if n_nbr > 0.0 { m_c / n_nbr } else { 1.0 };
                combined.update(
                    centre.tri,
                    centre.u,
                    centre.v,
                    m_c * p_c * centre.w,
                    rng::uniform(key, 0),
                );
                m_sum += centre.m;
            }
            for nb in &nbrs {
                let src = initial[nb.index];
                let (_, p_at_src) = di_contribution(scene, nb.p, nb.n, nb.albedo, &src);
                let (_, p_at_dst) = di_contribution(scene, p, n, albedo, &src);
                let (a, b) = (src.m * p_at_src, centre.m * p_at_dst);
                let m_i = if a + b > 0.0 {
                    a / (a + b) / n_nbr
                } else {
                    0.0
                };
                combined.update(
                    src.tri,
                    src.u,
                    src.v,
                    m_i * p_at_dst * src.w,
                    rng::uniform(key, nb.slot),
                );
                m_sum += src.m;
            }

            let (radiance, p_hat) = di_contribution(scene, p, n, albedo, &combined);
            combined.w = if p_hat > 0.0 {
                combined.w_sum / p_hat
            } else {
                0.0
            };
            combined.m = m_sum.min(RESTIR_M_CLAMP * (1.0 + RESTIR_NEIGHBOURS.len() as f32));
            // The reservoir keeps the *unshadowed* `W`, so temporal reuse next frame is
            // still reusing the quantity the target function is defined over. Only this
            // pixel's output is shadowed, by one ray towards the sample finally selected.
            spatial[i] = combined;
            let mut vis = combined.w;
            if vis > 0.0 {
                let light = &scene.tris[combined.tri as usize];
                let (lp, _, _) = tri_point(light, combined.u, combined.v);
                let seg = sub(lp, p);
                let dist = approx::sqrt(dot(seg, seg));
                if any_hit(
                    &scene.tris,
                    bvh,
                    p,
                    scale(seg, 1.0 / dist),
                    0.0,
                    dist * (1.0 - 1e-3),
                ) {
                    vis = 0.0;
                }
            }
            let shaded = scale(radiance, vis);
            out[i * 3..i * 3 + 3].copy_from_slice(&add(shaded, albedo.0.emission));
        }
    }
    let _ = spatial;
    out
}
