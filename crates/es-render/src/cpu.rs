//! The CPU reference (spec 1.4). Pure Rust, no GPU, no `unsafe`.
//!
//! This is the oracle: it generates every golden in `tests/golden/render/` and the Slang
//! kernels are line-for-line mirrors of it — same intersection routine, same traversal
//! order, same `es_math::approx` calls, same accumulation order. Where the two texts diverge
//! the Slang is wrong, not this file.
//!
//! Goldens are generated here and never by the GPU: a golden produced on one driver would
//! bake that driver's arithmetic into the repository and the next device would "fail" a test
//! that is really a device difference.
//!
//! `o`, `d`, `n`, `u`, `v`, `t` are the standard names for ray origin, direction, normal,
//! barycentrics and ray parameter, and the Slang mirror uses the same letters; renaming them
//! here would make the two texts harder to diff, so the lint is off for this module.
#![allow(clippy::many_single_char_names, clippy::cast_possible_wrap)]

use std::collections::BTreeMap;

use es_math::approx;
use es_sensor::Channel;

use crate::atlas::{Tile, TileData};
use crate::bvh::Bvh;
use crate::scene::TriScene;
use crate::view::{CameraView, RenderConfig, RenderPath, Shading, ViewParams, VIEW_STRIDE};

mod geometry;
mod pt;
mod restir;
mod shade;
mod svgf;

pub use geometry::{
    any_hit, any_hit_flat, face_forward, nearest_hit, nearest_hit_flat, primary_dir,
    primary_dir_sub, Hit,
};
pub use pt::{path_trace, path_trace_accum};
pub use restir::Reservoir;
pub(crate) use shade::shade_lambert;
pub use shade::{shade_full, shade_full_surface, srgb_encode, to_u8, tonemap, tonemap_to_u8};
pub use svgf::atrous;

use shade::shade_lambert_base;

/// Secondary rays start this far along the normal, to not re-hit the surface they left.
const RAY_EPS: f32 = 1e-4;
/// How far a [`Shading::Full`] shadow ray looks. The light is *directional* — infinitely far —
/// so the occluder may sit outside the camera's far plane, and the camera's `far` is the wrong
/// bound. Large and finite, because `rcp_safe` keeps the slab test's products finite.
const SHADOW_FAR: f32 = 1e30;

// --- small vector helpers (mirrored by Slang's builtin float3 ops) -------------------------

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}
fn scale(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let n = approx::sqrt(dot(a, a));
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        [0.0; 3]
    }
}
fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Rotate `v` by the unit quaternion `q` (xyzw), spec 3.1.
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let t = scale(cross(u, v), 2.0);
    add(add(v, scale(t, q[3])), cross(u, t))
}

/// Rotate `v` by the inverse of `q`.
pub fn quat_rotate_inv(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    quat_rotate([-q[0], -q[1], -q[2], q[3]], v)
}

// --- frames ---------------------------------------------------------------------------------

/// One camera's channels. The GPU path returns the same shapes through `Atlas::read_tile`.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub channels: BTreeMap<Channel, Tile>,
}

impl Frame {
    pub fn tile(&self, channel: Channel) -> Option<&Tile> {
        self.channels.get(&channel)
    }
}

/// Primary-hit geometry buffers, shared by both paths and by the `ReSTIR` and `SVGF` passes.
struct GBuffer {
    depth: Vec<f32>,
    seg: Vec<u32>,
    /// Camera-space unit normal, 3 per pixel.
    normal: Vec<f32>,
    /// Hit triangle index + 1 per pixel; `0` is a miss.
    tri: Vec<u32>,
}

fn g_buffer(scene: &TriScene, bvh: &Bvh, vp: &ViewParams, w: u32, h: u32) -> GBuffer {
    let n = (w as usize) * (h as usize);
    let mut g = GBuffer {
        depth: vec![vp.far; n],
        seg: vec![0; n],
        normal: vec![0.0; n * 3],
        tri: vec![0; n],
    };
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) as usize;
            let d = primary_dir(vp, px, py);
            let Some(hit) = nearest_hit(&scene.tris, bvh, vp.pos, d, vp.near, vp.far) else {
                continue;
            };
            let tri = &scene.tris[hit.tri as usize];
            let n_cam = quat_rotate_inv(vp.quat, face_forward(tri, d));
            g.depth[i] = hit.t;
            g.seg[i] = tri.seg;
            g.normal[i * 3..i * 3 + 3].copy_from_slice(&n_cam);
            g.tri[i] = hit.tri + 1;
        }
    }
    g
}

fn geometry_channels(g: &GBuffer, cfg: &RenderConfig, w: u32, h: u32, out: &mut Frame) {
    let shape1 = [h as usize, w as usize, 1];
    let shape3 = [h as usize, w as usize, 3];
    for channel in &cfg.channels {
        match channel {
            Channel::Depth32 { .. } => {
                out.channels.insert(
                    *channel,
                    Tile {
                        shape: shape1,
                        data: TileData::F32(g.depth.clone()),
                    },
                );
            }
            Channel::SegmentationId => {
                out.channels.insert(
                    *channel,
                    Tile {
                        shape: shape1,
                        data: TileData::U32(g.seg.clone()),
                    },
                );
            }
            Channel::Normal => {
                out.channels.insert(
                    *channel,
                    Tile {
                        shape: shape3,
                        data: TileData::F32(g.normal.clone()),
                    },
                );
            }
            _ => {}
        }
    }
}

/// Rasterize one view (spec 15.3 `RS`).
pub fn rasterize(
    scene: &TriScene,
    view: &CameraView,
    cfg: &RenderConfig,
    view_index: u32,
) -> Frame {
    let _ = view_index; // the `Rs` path draws no random numbers
    let vp = ViewParams::new(view);
    let (w, h) = (view.spec.width, view.spec.height);
    let bvh = Bvh::build(&scene.tris);
    let g = g_buffer(scene, &bvh, &vp, w, h);

    let mut rgb = vec![0u8; (w as usize) * (h as usize) * 3];
    match cfg.shading {
        // Untouched since M4: the hit is the g-buffer's, one sample per pixel, and every
        // golden pins the result (spec 28.10 rule 1).
        Shading::Lambert => {
            for py in 0..h {
                for px in 0..w {
                    let i = (py * w + px) as usize;
                    if g.tri[i] == 0 {
                        continue;
                    }
                    let d = primary_dir(&vp, px, py);
                    let tri = &scene.tris[(g.tri[i] - 1) as usize];
                    let surf = scene.materials.surface(tri, vp.pos, d);
                    let n = surf.normal.unwrap_or_else(|| face_forward(tri, d));
                    let lin = shade_lambert_base(surf.base, surf.emission, n, cfg);
                    for c in 0..3 {
                        rgb[i * 3 + c] = to_u8(lin[c]);
                    }
                }
            }
        }
        // The `Full` look traces its own rays: a sub-sample that the centre ray missed can
        // still cover geometry, so the loop is over every pixel and not over the g-buffer.
        // The `ssaa * ssaa` block is summed in **row-major order** (`sy` outer, `sx` inner,
        // one sequential `+=`), then multiplied once by `1 / (ssaa * ssaa)` and only then
        // encoded — the same order in `raster.slang` (spec 3.4: a fixed accumulation order).
        // A sub-sample that hits nothing contributes linear zero, which is the background
        // `Lambert` leaves too.
        Shading::Full { .. } => {
            let ssaa = cfg.shading.ssaa();
            let inv_ssaa = 1.0 / ssaa as f32;
            let norm = 1.0 / (ssaa * ssaa) as f32;
            for py in 0..h {
                for px in 0..w {
                    let i = (py * w + px) as usize;
                    let mut acc = [0.0f32; 3];
                    for sy in 0..ssaa {
                        for sx in 0..ssaa {
                            let d = primary_dir_sub(&vp, px, py, sx, sy, inv_ssaa);
                            let Some(hit) =
                                nearest_hit(&scene.tris, &bvh, vp.pos, d, vp.near, vp.far)
                            else {
                                continue;
                            };
                            let tri = &scene.tris[hit.tri as usize];
                            let n = face_forward(tri, d);
                            let p = add(vp.pos, scale(d, hit.t));
                            let surf = scene.materials.surface(tri, vp.pos, d);
                            acc = add(
                                acc,
                                shade_full_surface(&scene.tris, &bvh, tri, &surf, n, p, d, cfg),
                            );
                        }
                    }
                    let lin = scale(acc, norm);
                    for c in 0..3 {
                        rgb[i * 3 + c] = to_u8(lin[c]);
                    }
                }
            }
        }
    }

    let mut frame = Frame {
        width: w,
        height: h,
        channels: BTreeMap::new(),
    };
    if cfg.channels.contains(&Channel::Rgb8) {
        frame.channels.insert(
            Channel::Rgb8,
            Tile {
                shape: [h as usize, w as usize, 3],
                data: TileData::U8(rgb),
            },
        );
    }
    geometry_channels(&g, cfg, w, h, &mut frame);
    frame
}

/// The reference for [`crate::Renderer::render_batch`] (packet M11/X3b): tile `k` is env `k`
/// rendered alone as view 0 — its own BVH over its own triangles, its own seed and light, view
/// 0's sample keys — which is the claim the batched dispatch makes about each of its tiles.
pub fn render_batch(envs: &[(TriScene, CameraView, RenderConfig)]) -> Vec<Frame> {
    envs.iter()
        .map(|(tri, view, cfg)| match cfg.path {
            RenderPath::Rs => rasterize(tri, view, cfg, 0),
            RenderPath::Pt { .. } => path_trace(tri, view, cfg, 0),
        })
        .collect()
}

// --- the temporal history (packet M7/R4) -----------------------------------------------------

/// One camera slot's accumulated frames: what `path_trace_accum` carries from frame to frame,
/// and the mirror of the 12-float-per-pixel buffer `renderer.rs` keeps on the device.
///
/// Nothing in here is a *setting* — [`crate::Temporal`] is. This is the state, and it is
/// owned by the caller so that the reference stays a pure function of its inputs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct History {
    /// The `ViewParams` floats, bitwise, of the camera the history was built with.
    view: Option<[u32; VIEW_STRIDE]>,
    /// Frames the slot has rendered since the last whole-slot reset: the sample base.
    frame: u32,
    /// Running radiance sum, 3 per pixel, in the kernel's own accumulation order.
    sum: Vec<f32>,
    /// First and second raw moments of the per-frame luminance, 2 per pixel.
    moments: Vec<f32>,
    /// History length per pixel — the `n` of [`Channel::History`].
    n: Vec<u32>,
    /// The previous frame's primary hit: what the disocclusion test compares, bitwise.
    depth: Vec<f32>,
    normal: Vec<f32>,
    tri: Vec<u32>,
    /// Variance of the accumulated estimate, 1 per pixel: what the a-trous pass reads.
    variance: Vec<f32>,
}

impl History {
    /// History length per pixel after the last frame.
    #[must_use]
    pub fn n(&self) -> &[u32] {
        &self.n
    }

    /// Variance of the accumulated estimate per pixel after the last frame.
    #[must_use]
    pub fn variance(&self) -> &[f32] {
        &self.variance
    }

    /// Frames accumulated since the last whole-slot reset.
    #[must_use]
    pub fn frames(&self) -> u32 {
        self.frame
    }

    fn reset(&mut self, n_px: usize) {
        self.view = None;
        self.frame = 0;
        self.sum = vec![0.0; n_px * 3];
        self.moments = vec![0.0; n_px * 2];
        self.n = vec![0; n_px];
        self.depth = vec![0.0; n_px];
        self.normal = vec![0.0; n_px * 3];
        self.tri = vec![0; n_px];
        self.variance = vec![0.0; n_px];
    }

    /// Bitwise, never `==`: a depth that moved by one ULP is a different surface as far as
    /// this test is concerned, and saying so costs nothing (spec 3.4).
    fn same_geometry(&self, i: usize, g: &GBuffer) -> bool {
        self.depth[i].to_bits() == g.depth[i].to_bits()
            && self.tri[i] == g.tri[i]
            && (0..3).all(|c| self.normal[i * 3 + c].to_bits() == g.normal[i * 3 + c].to_bits())
    }
}

fn view_bits(vp: &ViewParams) -> [u32; VIEW_STRIDE] {
    let mut out = [0u32; VIEW_STRIDE];
    for (slot, f) in out.iter_mut().zip(vp.to_floats()) {
        *slot = f.to_bits();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::geometry::intersect;
    use super::pt::cosine_hemisphere;
    use super::*;
    use crate::cornell::{cornell_box, cornell_camera};
    use crate::rng;
    use crate::view::TileAtlasCfg;

    fn view(w: u32, h: u32) -> CameraView {
        cornell_camera(w, h)
    }

    #[test]
    fn srgb_matches_the_reference_curve_at_the_knee() {
        assert!((srgb_encode(0.0) - 0.0).abs() < 1e-7);
        assert!((srgb_encode(1.0) - 1.0).abs() < 1e-4);
        // Continuity at the piecewise boundary.
        let lo = srgb_encode(0.003_130_7);
        let hi = srgb_encode(0.003_131);
        assert!((hi - lo).abs() < 1e-4, "{lo} vs {hi}");
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(1.0), 255);
    }

    #[test]
    fn a_ray_down_the_axis_hits_the_nearest_triangle_first() {
        let scene = crate::scene::TriScene::from_scene(&cornell_box()).unwrap();
        let vp = ViewParams::new(&view(8, 8));
        let d = primary_dir(&vp, 4, 4);
        let bvh = Bvh::build(&scene.tris);
        let hit = nearest_hit(&scene.tris, &bvh, vp.pos, d, vp.near, vp.far).unwrap();
        assert!(hit.t > 0.0 && hit.t < vp.far);
        // No other triangle is closer.
        for tri in &scene.tris {
            if let Some(t) = intersect(tri, vp.pos, d) {
                if t > vp.near {
                    assert!(t >= hit.t - 1e-5);
                }
            }
        }
    }

    #[test]
    fn rasterizer_is_deterministic_and_covers_the_tile() {
        let scene = crate::scene::TriScene::from_scene(&cornell_box()).unwrap();
        let cfg = RenderConfig::rs(TileAtlasCfg::row(16, 16, 1));
        let v = view(16, 16);
        let a = rasterize(&scene, &v, &cfg, 0);
        let b = rasterize(&scene, &v, &cfg, 0);
        assert_eq!(a, b);
        let seg = a.tile(Channel::SegmentationId).unwrap().as_u32().unwrap();
        assert!(seg.iter().all(|s| *s > 0), "the box should fill the tile");
    }

    /// Spec 15.3: depth, segmentation and normal are bit-identical between the render paths.
    #[test]
    fn pt_and_rs_agree_on_geometry() {
        let scene = crate::scene::TriScene::from_scene(&cornell_box()).unwrap();
        let atlas = TileAtlasCfg::row(16, 16, 1);
        let v = view(16, 16);
        let rs = rasterize(&scene, &v, &RenderConfig::rs(atlas), 0);
        let pt = path_trace(&scene, &v, &RenderConfig::pt(atlas, 1, 2), 0);
        for channel in [
            Channel::Depth32 { unit_m: 1.0 },
            Channel::SegmentationId,
            Channel::Normal,
        ] {
            assert_eq!(
                rs.tile(channel).unwrap().to_bytes(),
                pt.tile(channel).unwrap().to_bytes(),
                "{channel:?} differs between RS and PT"
            );
        }
    }

    #[test]
    fn path_tracer_is_deterministic_and_finite() {
        let scene = crate::scene::TriScene::from_scene(&cornell_box()).unwrap();
        let cfg = RenderConfig::pt(TileAtlasCfg::row(12, 12, 1), 4, 3);
        let v = view(12, 12);
        let a = path_trace(&scene, &v, &cfg, 0);
        assert_eq!(a, path_trace(&scene, &v, &cfg, 0));
        let r = a.tile(Channel::PtRadiance).unwrap().as_f32().unwrap();
        assert!(r.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!(r.iter().any(|x| *x > 0.0), "the light should be visible");
    }

    #[test]
    fn restir_and_svgf_produce_finite_images() {
        let scene = crate::scene::TriScene::from_scene(&cornell_box()).unwrap();
        let mut cfg = RenderConfig::pt(TileAtlasCfg::row(12, 12, 1), 1, 2);
        cfg.path = RenderPath::Pt {
            spp: 1,
            bounces: 2,
            nee: false,
            restir: true,
            svgf: true,
        };
        let f = path_trace(&scene, &view(12, 12), &cfg, 0);
        let r = f.tile(Channel::PtRadiance).unwrap().as_f32().unwrap();
        assert!(r.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!(r.iter().any(|x| *x > 0.0));
    }

    #[test]
    fn cosine_samples_stay_in_the_hemisphere() {
        let n = normalize([0.3, -0.5, 0.8]);
        for i in 0..2000 {
            let d = cosine_hemisphere(n, rng::key(1, 0, i % 50, i / 50, 0, 0, 0));
            assert!(dot(d, n) >= -1e-6, "{d:?}");
            assert!((dot(d, d) - 1.0).abs() < 1e-3);
        }
    }
}
