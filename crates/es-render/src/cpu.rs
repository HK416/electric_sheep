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
use crate::bvh::{self, Bvh};
use crate::rng;
use crate::scene::{Tri, TriScene};
use crate::view::{
    CameraView, RenderConfig, RenderPath, Shading, Tonemap, ViewParams, VIEW_STRIDE,
};

/// Below this determinant a triangle is edge-on to the ray and is skipped.
const DET_EPS: f32 = 1e-8;
/// Secondary rays start this far along the normal, to not re-hit the surface they left.
const RAY_EPS: f32 = 1e-4;
/// How far a [`Shading::Full`] shadow ray looks. The light is *directional* — infinitely far —
/// so the occluder may sit outside the camera's far plane, and the camera's `far` is the wrong
/// bound. Large and finite, because `rcp_safe` keeps the slab test's products finite.
const SHADOW_FAR: f32 = 1e30;
/// Direct-light candidates per pixel in the `ReSTIR` initial pass.
const RESTIR_CANDIDATES: u32 = 8;
/// `M` clamp on temporal reuse.
const RESTIR_M_CLAMP: f32 = 20.0;
/// Spatial-reuse neighbour offsets. Fixed, not RNG-jittered: jitter buys less correlated
/// noise and costs the ability to say the result is a function of the pixel grid alone.
const RESTIR_NEIGHBOURS: [(i32, i32); 4] = [(3, 0), (-3, 0), (0, 3), (0, -3)];
/// Depth edge-stopping scale of the a-trous filter, m.
const SVGF_SIGMA_Z: f32 = 0.1;
/// 5-tap B-spline wavelet row.
const SVGF_H: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
/// Luminance edge-stopping scale (packet M7/R4; Schied et al. 2017 section 4.3).
const SVGF_SIGMA_L: f32 = 4.0;
/// Keeps the luminance weight's denominator off zero where the variance is zero.
const SVGF_VAR_EPS: f32 = 1e-10;
/// Frames of history below which the moments are too few to trust and the 7x7 spatial
/// estimate stands in (Schied et al. 2017 section 4.2).
const SVGF_MIN_HISTORY: u32 = 4;
/// Radius of that spatial estimate: 7x7.
const SVGF_SPATIAL_RADIUS: i32 = 3;

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

// --- geometry ------------------------------------------------------------------------------

/// A primary or secondary hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub t: f32,
    pub tri: u32,
}

/// Moller-Trumbore. Returns the ray parameter, unfiltered by near/far.
fn intersect(tri: &Tri, o: [f32; 3], d: [f32; 3]) -> Option<f32> {
    let e1 = sub(tri.v[1], tri.v[0]);
    let e2 = sub(tri.v[2], tri.v[0]);
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < DET_EPS {
        return None;
    }
    let inv = 1.0 / det;
    // `o - v0` is the only place a world coordinate enters: the intersection is already
    // camera-relative (spec 3.3) because `o` is the camera origin.
    let tv = sub(o, tri.v[0]);
    let u = dot(tv, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(tv, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(dot(e2, q) * inv)
}

/// Scan every triangle in ascending index order and keep the nearest hit strictly inside
/// `(near, far)`. Ties go to the lower index, so the winner never depends on scheduling.
///
/// Retained as [`nearest_hit`]'s oracle, not as a render path: `bvh_traversal_is_the_flat_scan`
/// asserts the two agree on `Option<Hit>` bitwise over both scenes (packet M7/R1 oracle 3).
pub fn nearest_hit_flat(
    tris: &[Tri],
    o: [f32; 3],
    d: [f32; 3],
    near: f32,
    far: f32,
) -> Option<Hit> {
    let mut best = far;
    let mut which = None;
    for (i, tri) in tris.iter().enumerate() {
        if let Some(t) = intersect(tri, o, d) {
            if t > near && t < best {
                best = t;
                which = Some(i as u32);
            }
        }
    }
    which.map(|tri| Hit { t: best, tri })
}

/// [`nearest_hit_flat`] without the flat scan: `1 / d` once, then a stack-based descent of
/// `bvh` in a fixed order — left child first, always, never ordered by the ray's sign.
///
/// **The hit rule is the flat scan's**, which is what makes the two agree bit for bit: nearest
/// `t` with a strict `<`, ties broken by the lower *global* triangle index. A tie-break on the
/// index rather than on the visit order is what buys the freedom to visit nodes in any order
/// at all. The slab test is conservative (`bvh::LO`/`HI`, and every box padded outward), so
/// the visited set is a superset of the triangles that can win; it may even differ by an ULP
/// between the CPU and the GPU without changing the answer.
#[allow(clippy::float_cmp)]
pub fn nearest_hit(
    tris: &[Tri],
    bvh: &Bvh,
    o: [f32; 3],
    d: [f32; 3],
    near: f32,
    far: f32,
) -> Option<Hit> {
    let mut best = far;
    let mut which: Option<u32> = None;
    if bvh.nodes.is_empty() {
        return None;
    }
    let inv = [rcp_safe(d[0]), rcp_safe(d[1]), rcp_safe(d[2])];
    let mut stack = [0u32; bvh::STACK as usize];
    let mut sp = 1usize;
    while sp > 0 {
        sp -= 1;
        let ni = stack[sp] as usize;
        let node = &bvh.nodes[ni];
        if !slab(node, o, inv, near, best) {
            continue;
        }
        if node.count == 0 {
            stack[sp] = node.a;
            stack[sp + 1] = ni as u32 + 1; // left child, popped first
            sp += 2;
            continue;
        }
        for k in 0..node.count as usize {
            let i = bvh.prim[node.a as usize + k];
            if let Some(t) = intersect(&tris[i as usize], o, d) {
                if t > near && (t < best || matches!(which, Some(w) if t == best && i < w)) {
                    best = t;
                    which = Some(i);
                }
            }
        }
    }
    which.map(|tri| Hit { t: best, tri })
}

/// [`any_hit`]'s oracle: the retained flat shadow scan.
pub fn any_hit_flat(tris: &[Tri], o: [f32; 3], d: [f32; 3], near: f32, far: f32) -> bool {
    tris.iter()
        .any(|tri| intersect(tri, o, d).is_some_and(|t| t > near && t < far))
}

/// Is anything in `(near, far)`? A boolean does not depend on the visit order at all, so this
/// is the same descent with an early return.
pub fn any_hit(tris: &[Tri], bvh: &Bvh, o: [f32; 3], d: [f32; 3], near: f32, far: f32) -> bool {
    if bvh.nodes.is_empty() {
        return false;
    }
    let inv = [rcp_safe(d[0]), rcp_safe(d[1]), rcp_safe(d[2])];
    let mut stack = [0u32; bvh::STACK as usize];
    let mut sp = 1usize;
    while sp > 0 {
        sp -= 1;
        let ni = stack[sp] as usize;
        let node = &bvh.nodes[ni];
        if !slab(node, o, inv, near, far) {
            continue;
        }
        if node.count == 0 {
            stack[sp] = node.a;
            stack[sp + 1] = ni as u32 + 1;
            sp += 2;
            continue;
        }
        for k in 0..node.count as usize {
            let i = bvh.prim[node.a as usize + k] as usize;
            if intersect(&tris[i], o, d).is_some_and(|t| t > near && t < far) {
                return true;
            }
        }
    }
    false
}

/// `1 / x`, with the infinity clamped away. A finite reciprocal is what keeps `0 * inf` — the
/// only NaN the slab test can produce, for a ray exactly parallel to a slab whose origin is
/// exactly on it — out of both texts, so neither has to define a NaN-ordering convention that
/// Rust's `min` and Slang's `min` disagree about.
fn rcp_safe(x: f32) -> f32 {
    if x.abs() > 1e-30 {
        1.0 / x
    } else if x < 0.0 {
        -1e30
    } else {
        1e30
    }
}

/// Ray-box overlap on `(near, far)`. `min`/`max` are exact, so only the two products round,
/// and `bvh::LO`/`HI` cover that rounding in the conservative direction.
fn slab(node: &bvh::Node, o: [f32; 3], inv: [f32; 3], near: f32, far: f32) -> bool {
    let t0 = [
        (node.min[0] - o[0]) * inv[0],
        (node.min[1] - o[1]) * inv[1],
        (node.min[2] - o[2]) * inv[2],
    ];
    let t1 = [
        (node.max[0] - o[0]) * inv[0],
        (node.max[1] - o[1]) * inv[1],
        (node.max[2] - o[2]) * inv[2],
    ];
    let lo = near
        .max(t0[0].min(t1[0]))
        .max(t0[1].min(t1[1]))
        .max(t0[2].min(t1[2]));
    let hi = far
        .min(t0[0].max(t1[0]))
        .min(t0[1].max(t1[1]))
        .min(t0[2].max(t1[2]));
    lo * bvh::LO <= hi * bvh::HI
}

/// Normal of the hit triangle, flipped to face the incoming ray.
pub fn face_forward(tri: &Tri, d: [f32; 3]) -> [f32; 3] {
    if dot(tri.n, d) > 0.0 {
        scale(tri.n, -1.0)
    } else {
        tri.n
    }
}

/// Camera-space primary ray direction, **not** normalised: `z == 1`, so the ray parameter is
/// the camera-space depth in metres and `Depth32`'s `unit_m` is 1 (spec 3.1, spec 7.2).
///
/// Through the pixel centre: [`primary_dir_sub`] with one sub-sample puts the offset at
/// exactly `0.5`, so this is the same float it always was.
pub fn primary_dir(vp: &ViewParams, px: u32, py: u32) -> [f32; 3] {
    primary_dir_sub(vp, px, py, 0, 0, 1.0)
}

/// [`primary_dir`] through sub-sample `(sx, sy)` of an `ssaa * ssaa` grid, `inv_ssaa` being
/// `1 / ssaa` (packet M7/R2). The sample sits at `px + (sx + 0.5) / ssaa`, so a single
/// sub-sample is the pixel centre and the geometry channels do not move.
pub fn primary_dir_sub(
    vp: &ViewParams,
    px: u32,
    py: u32,
    sx: u32,
    sy: u32,
    inv_ssaa: f32,
) -> [f32; 3] {
    let d_cam = [
        (px as f32 + (sx as f32 + 0.5) * inv_ssaa - vp.cx) / vp.fx,
        (py as f32 + (sy as f32 + 0.5) * inv_ssaa - vp.cy) / vp.fy,
        1.0,
    ];
    quat_rotate(vp.quat, d_cam)
}

// --- shading -------------------------------------------------------------------------------

/// Flat + Lambert with one directional light. No shadow ray: that is a second scan of the
/// whole array per pixel, and the `Pt` path models occlusion properly.
///
/// `ponytail:` no shadows in `Rs`; add a shadow scan when a golden shows the missing contact
/// shadow costs a policy something.
fn shade_lambert(tri: &Tri, n: [f32; 3], cfg: &RenderConfig) -> [f32; 3] {
    let light = [
        cfg.light_dir.x as f32,
        cfg.light_dir.y as f32,
        cfg.light_dir.z as f32,
    ];
    let ndl = dot(n, light).max(0.0);
    let lambert = cfg.ambient + ndl * (1.0 - cfg.ambient);
    add(scale(tri.albedo, lambert), tri.emission)
}

/// [`Shading::Full`]: one shadow ray, a hemisphere ambient, a Blinn-Phong highlight (packet
/// M7/R2). `es_shade_full` in `common.slang` is the line-for-line mirror; the order below is
/// the order there.
///
/// `n` is the world-space normal already face-forwarded, `p` the hit point, `d` the (not
/// normalised) view ray. The shadow ray runs on `(0, SHADOW_FAR)` from `p + n * RAY_EPS`
/// towards the light, so the surface cannot shadow itself. `pow` is `exp(ln(x) * k)` guarded at
/// `x <= 0` — no `std`, no `GLSL.std.450` (spec 3.2 `DET-010`).
///
/// A [`Shading::Lambert`] config never reaches here; it is answered as itself so this is a
/// total function on `cfg`.
pub fn shade_full(
    tris: &[Tri],
    bvh: &Bvh,
    tri: &Tri,
    n: [f32; 3],
    p: [f32; 3],
    d: [f32; 3],
    cfg: &RenderConfig,
) -> [f32; 3] {
    let Shading::Full {
        shadows,
        specular,
        shininess,
        sky_rgb,
        ground_rgb,
        ..
    } = cfg.shading
    else {
        return shade_lambert(tri, n, cfg);
    };
    let light = [
        cfg.light_dir.x as f32,
        cfg.light_dir.y as f32,
        cfg.light_dir.z as f32,
    ];
    let vis = if shadows && any_hit(tris, bvh, add(p, scale(n, RAY_EPS)), light, 0.0, SHADOW_FAR) {
        0.0
    } else {
        1.0
    };
    let diffuse = dot(n, light).max(0.0) * vis;
    let h = normalize(sub(light, d));
    let ndh = dot(n, h).max(0.0);
    let spec = if ndh > 0.0 {
        specular * approx::exp(approx::ln(ndh) * shininess) * vis
    } else {
        0.0
    };
    // Written out rather than a `lerp`: HLSL's `lerp` and GLSL's `mix` are not the same
    // expression, and the two texts have to round the same way.
    // Not `f32::midpoint`: `common.slang` computes `(n.z + 1.0) * 0.5` and the two texts have
    // to be the same two operations, not two functions that usually agree.
    #[allow(clippy::manual_midpoint)]
    let t = (n[2] + 1.0) * 0.5;
    let hemi = add(ground_rgb, scale(sub(sky_rgb, ground_rgb), t));
    // Energy-conserving, the same mix `shade_lambert` uses for its constant ambient (amended
    // at review): `hemi + diffuse * (1 - hemi)` per channel, so a fully lit surface returns
    // its albedo and a shadowed one `albedo * hemi`, and nothing clips to white before the
    // sRGB transfer. `spec` stays additive: a highlight is allowed to blow out.
    let lit = [
        hemi[0] + diffuse * (1.0 - hemi[0]),
        hemi[1] + diffuse * (1.0 - hemi[1]),
        hemi[2] + diffuse * (1.0 - hemi[2]),
    ];
    add(add(mul(tri.albedo, lit), [spec; 3]), tri.emission)
}

/// Exact piecewise sRGB transfer (spec 3.1). `c^(1/2.4)` goes through `es_math::approx`, not
/// `std`, so the CPU and the GPU agree bit for bit (spec 3.2 `DET-010`, spec 28.7 gate 3).
pub fn srgb_encode(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * approx::exp(approx::ln(c) * (1.0 / 2.4)) - 0.055
    }
}

pub fn to_u8(c: f32) -> u8 {
    (255.0 * srgb_encode(c) + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// Linear radiance -> display value, one channel (packet M7/R3).
///
/// Additions, multiplies and one division: **no transcendental at all**, which is why the
/// CPU and the GPU are asserted bit-equal here rather than to a ULP budget.
/// `es_tonemap` in `common.slang` is the line-for-line mirror.
pub fn tonemap(c: f32, exposure: f32, map: Tonemap) -> f32 {
    let x = c * exposure;
    match map {
        // Reinhard et al. 2002. Monotone on [0, inf), never reaches 1, so it never clips.
        Tonemap::Reinhard => x / (1.0 + x),
        // Narkowicz 2015's rational ACES fit. Clamped: the fit dips below 0 for x < 0 and
        // creeps past 1 at the top of its range.
        Tonemap::Aces => {
            let num = x * (2.51 * x + 0.03);
            let den = x * (2.43 * x + 0.59) + 0.14;
            (num / den).clamp(0.0, 1.0)
        }
    }
}

/// One linear RGB triple through [`tonemap`] and then the exact sRGB transfer of
/// [`to_u8`] — the `Pt` path's [`Channel::Rgb8`].
pub fn tonemap_to_u8(lin: [f32; 3], exposure: f32, map: Tonemap) -> [u8; 3] {
    [0, 1, 2].map(|c| to_u8(tonemap(lin[c], exposure, map)))
}

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
                    let lin = shade_lambert(tri, face_forward(tri, d), cfg);
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
                            acc = add(acc, shade_full(&scene.tris, &bvh, tri, n, p, d, cfg));
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
fn cosine_hemisphere(n: [f32; 3], key: u32) -> [f32; 3] {
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
    albedo: [f32; 3],
    far: f32,
    last: bool,
    keys: [u32; 3],
) -> [f32; 3] {
    let f = scale(albedo, std::f32::consts::FRAC_1_PI);
    let mut out = [0.0f32; 3];

    if !scene.lights.is_empty() {
        let n_lights = scene.lights.len() as f32;
        let pick = rng::uniform(keys[0], 0);
        let idx = ((pick * n_lights) as usize).min(scene.lights.len() - 1);
        let light = &scene.tris[scene.lights[idx] as usize];
        let (lp, _, _) = tri_point(light, rng::uniform(keys[1], 0), rng::uniform(keys[1], 1));
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
                    power_heuristic(p_light, cos_s * std::f32::consts::FRAC_1_PI)
                };
                // `dist * (1 - 1e-3)` so the shadow ray stops short of the light itself.
                if !any_hit(&scene.tris, bvh, p, dir, 0.0, dist * (1.0 - 1e-3)) {
                    out = add(out, scale(mul(f, light.emission), cos_s / p_light * w));
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
            out = add(out, scale(mul(f, cfg.light_rgb), cos_s));
        }
    }

    if cfg.sky.iter().any(|c| *c > 0.0) {
        let d = cosine_hemisphere(n, keys[2]);
        let cos_s = dot(n, d);
        let pdf = cos_s * std::f32::consts::FRAC_1_PI;
        if pdf > 0.0 && !any_hit(&scene.tris, bvh, p, d, 0.0, far) {
            let w = if last { 1.0 } else { power_heuristic(pdf, pdf) };
            out = add(out, scale(mul(f, cfg.sky), cos_s / pdf * w));
        }
    }
    out
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
                for bounce in 0..bounces {
                    let Some(hit) = nearest_hit(&scene.tris, &bvh, o, d, near, vp.far) else {
                        // The sky through the BSDF strategy. Its NEE counterpart draws from
                        // the same cosine pdf, so the power heuristic splits it in half.
                        let w = if nee && bounce > 0 {
                            power_heuristic(prev_pdf, prev_pdf)
                        } else {
                            1.0
                        };
                        acc = add(acc, scale(mul(throughput, cfg.sky), w));
                        break;
                    };
                    let tri = &scene.tris[hit.tri as usize];
                    // An emissive hit reached by a BSDF bounce: the light-sampling strategy
                    // could have produced it too, so it is MIS-weighted against that pdf.
                    let w_em = if nee && bounce > 0 {
                        let p_light = light_pdf(tri, d, hit.t * hit.t, n_lights);
                        power_heuristic(prev_pdf, p_light)
                    } else {
                        1.0
                    };
                    acc = add(acc, scale(mul(throughput, tri.emission), w_em));
                    let n = face_forward(tri, d);
                    let p = add(add(o, scale(d, hit.t)), scale(n, RAY_EPS));
                    if nee {
                        let key =
                            |stream| rng::key(cfg.seed, view_index, px, py, s, bounce, stream);
                        let direct = nee_direct(
                            scene,
                            &bvh,
                            cfg,
                            p,
                            n,
                            tri.albedo,
                            vp.far,
                            bounce + 1 == bounces,
                            [key(4), key(5), key(6)],
                        );
                        acc = add(acc, mul(throughput, direct));
                    }
                    throughput = mul(throughput, tri.albedo);
                    o = p;
                    d = cosine_hemisphere(n, rng::key(cfg.seed, view_index, px, py, s, bounce, 0));
                    prev_pdf = dot(n, d) * std::f32::consts::FRAC_1_PI;
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

/// Variance of the accumulated estimate, per pixel (packet M7/R4; Schied et al. 2017 section
/// 4.2). Mirror of `accumulate.slang`.
///
/// With `n >= 4` frames it comes from the moments: `var(l) = E[l^2] - E[l]^2` over the
/// per-frame luminances, divided by `n` once more because what the filter needs is the
/// variance of the *mean* of those `n` frames, which is what the pixel holds. Below 4 the
/// moments are too few to be worth anything and the paper's 7x7 depth/normal-weighted spatial
/// estimate stands in, divided by `n` for the same reason — so the quantity is continuous
/// across the switch and always means "how uncertain is this pixel".
fn variance_estimate(radiance: &[f32], history: &History, g: &GBuffer, w: u32, h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (w as usize) * (h as usize)];
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) as usize;
            let n = history.n[i];
            if n == 0 {
                continue;
            }
            let inv_n = 1.0 / n as f32;
            out[i] = if n >= SVGF_MIN_HISTORY {
                let m1 = history.moments[i * 2] * inv_n;
                let m2 = history.moments[i * 2 + 1] * inv_n;
                (m2 - m1 * m1).max(0.0) * inv_n
            } else {
                let (mut sw, mut swl, mut swl2) = (0.0f32, 0.0f32, 0.0f32);
                for dy in -SVGF_SPATIAL_RADIUS..=SVGF_SPATIAL_RADIUS {
                    for dx in -SVGF_SPATIAL_RADIUS..=SVGF_SPATIAL_RADIUS {
                        let (nx, ny) = (px as i32 + dx, py as i32 + dy);
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let j = (ny as u32 * w + nx as u32) as usize;
                        let (wz, wn) = edge_weights(&g.depth, &g.normal, i, j);
                        let weight = wz * wn;
                        let l =
                            luminance([radiance[j * 3], radiance[j * 3 + 1], radiance[j * 3 + 2]]);
                        sw += weight;
                        swl += weight * l;
                        swl2 += weight * l * l;
                    }
                }
                let inv = if sw > 0.0 { 1.0 / sw } else { 0.0 };
                let mean = swl * inv;
                (swl2 * inv - mean * mean).max(0.0) * inv_n
            };
        }
    }
    out
}

/// The depth and normal edge-stopping weights, returned separately so each caller multiplies
/// them in its own order — the a-trous filter's `wh * wz * wn` is the order its output bits
/// were measured in and must not move.
fn edge_weights(depth: &[f32], normal: &[f32], i: usize, j: usize) -> (f32, f32) {
    let wz = approx::exp(-(depth[i] - depth[j]).abs() / SVGF_SIGMA_Z);
    let mut wn = (normal[i * 3] * normal[j * 3]
        + normal[i * 3 + 1] * normal[j * 3 + 1]
        + normal[i * 3 + 2] * normal[j * 3 + 2])
        .max(0.0);
    // n^32 by five squarings: no transcendental, exact same ops in Slang.
    for _ in 0..5 {
        wn *= wn;
    }
    (wz, wn)
}

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
struct Neighbour {
    index: usize,
    p: [f32; 3],
    n: [f32; 3],
    albedo: [f32; 3],
    slot: u32,
}

/// Uniform point on a triangle from two canonical randoms.
fn tri_point(tri: &Tri, u: f32, v: f32) -> ([f32; 3], f32, f32) {
    let su = approx::sqrt(u);
    let (b0, b1) = (1.0 - su, v * su);
    let p = add(
        add(scale(tri.v[0], 1.0 - b0 - b1), scale(tri.v[1], b0)),
        scale(tri.v[2], b1),
    );
    (p, b0, b1)
}

fn tri_area(tri: &Tri) -> f32 {
    let c = cross(sub(tri.v[1], tri.v[0]), sub(tri.v[2], tri.v[0]));
    0.5 * approx::sqrt(dot(c, c))
}

/// Unshadowed target function and the radiance it stands for.
fn di_contribution(
    scene: &TriScene,
    shade_p: [f32; 3],
    n: [f32; 3],
    albedo: [f32; 3],
    r: &Reservoir,
) -> ([f32; 3], f32) {
    let Some(light) = scene.tris.get(r.tri as usize) else {
        return ([0.0; 3], 0.0);
    };
    let (lp, _, _) = tri_point(light, r.u, r.v);
    let to_light = sub(lp, shade_p);
    let dist2 = dot(to_light, to_light).max(1e-8);
    let dir = scale(to_light, 1.0 / approx::sqrt(dist2));
    let cos_s = dot(n, dir).max(0.0);
    let cos_l = dot(light.n, scale(dir, -1.0)).abs();
    let g = cos_s * cos_l / dist2 * tri_area(light) * scene.lights.len() as f32;
    let f = scale(albedo, std::f32::consts::FRAC_1_PI);
    let radiance = scale(mul(f, light.emission), g);
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
fn restir_di(
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

    let hit_of = |i: usize, px: u32, py: u32| -> Option<([f32; 3], [f32; 3], [f32; 3])> {
        if g.tri[i] == 0 {
            return None;
        }
        let tri = &scene.tris[(g.tri[i] - 1) as usize];
        let d = primary_dir(vp, px, py);
        let n = face_forward(tri, d);
        let p = add(add(vp.pos, scale(d, g.depth[i])), scale(n, RAY_EPS));
        Some((p, n, tri.albedo))
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
            out[i * 3..i * 3 + 3]
                .copy_from_slice(&add(shaded, scene.tris[(g.tri[i] - 1) as usize].emission));
        }
    }
    let _ = spatial;
    out
}

// --- SVGF (the a-trous half) -----------------------------------------------------------------

/// Edge-aware a-trous wavelet filter, `iterations` passes at stride `1 << i`, edge-stopping
/// on depth, normal and — when `variance` is `Some` (packet M7/R4) — luminance.
///
/// `variance` is the per-pixel variance of the accumulated estimate ([`History::variance`]).
/// With it the filter is variance-guided the way Schied et al. 2017 section 4.3 is: the
/// luminance weight `exp(-|l_p - l_q| / (sigma_l * sqrt(var_p) + eps))` narrows the kernel
/// wherever the estimate has converged, and the variance is filtered alongside the colour
/// with the squared weights so the next iteration sees the variance of what it is reading.
/// Without it — `temporal: None` — the weight is exactly `1.0` and the output is byte for
/// byte what M4's filter produced.
///
/// Still **not** the whole of SVGF: no history-length-driven kernel widening, and the
/// temporal half is `path_trace_accum`'s, not this function's. See
/// `docs/design/renderer.md` sections 4.3 and 11.
pub fn atrous(
    color: &[f32],
    variance: Option<&[f32]>,
    depth: &[f32],
    normal: &[f32],
    w: u32,
    h: u32,
    iterations: u32,
) -> (Vec<f32>, Vec<f32>) {
    let mut src = color.to_vec();
    let mut dst = src.clone();
    let mut vsrc = variance.map(<[f32]>::to_vec).unwrap_or_default();
    let mut vdst = vsrc.clone();
    let guided = !vsrc.is_empty();
    for it in 0..iterations {
        let stride = 1i32 << it;
        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) as usize;
                let mut sum = [0.0f32; 3];
                let (mut wsum, mut vsum) = (0.0f32, 0.0f32);
                let l_p = luminance([src[i * 3], src[i * 3 + 1], src[i * 3 + 2]]);
                let sigma_l = if guided {
                    SVGF_SIGMA_L * approx::sqrt(vsrc[i]) + SVGF_VAR_EPS
                } else {
                    0.0
                };
                for ky in 0..5i32 {
                    for kx in 0..5i32 {
                        let nx = px as i32 + (kx - 2) * stride;
                        let ny = py as i32 + (ky - 2) * stride;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let j = (ny as u32 * w + nx as u32) as usize;
                        let wh = SVGF_H[kx as usize] * SVGF_H[ky as usize];
                        let (wz, wn) = edge_weights(depth, normal, i, j);
                        let wl = if guided {
                            let l_q = luminance([src[j * 3], src[j * 3 + 1], src[j * 3 + 2]]);
                            approx::exp(-(l_p - l_q).abs() / sigma_l)
                        } else {
                            1.0
                        };
                        let weight = wh * wz * wn * wl;
                        sum = add(
                            sum,
                            scale([src[j * 3], src[j * 3 + 1], src[j * 3 + 2]], weight),
                        );
                        wsum += weight;
                        if guided {
                            vsum += weight * weight * vsrc[j];
                        }
                    }
                }
                let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
                dst[i * 3..i * 3 + 3].copy_from_slice(&scale(sum, inv));
                if guided {
                    vdst[i] = vsum * inv * inv;
                }
            }
        }
        std::mem::swap(&mut src, &mut dst);
        if guided {
            std::mem::swap(&mut vsrc, &mut vdst);
        }
    }
    (src, vsrc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cornell::{cornell_box, cornell_camera};
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
