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
use crate::view::{CameraView, RenderConfig, RenderPath, ViewParams};

/// Below this determinant a triangle is edge-on to the ray and is skipped.
const DET_EPS: f32 = 1e-8;
/// Secondary rays start this far along the normal, to not re-hit the surface they left.
const RAY_EPS: f32 = 1e-4;
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
fn face_forward(tri: &Tri, d: [f32; 3]) -> [f32; 3] {
    if dot(tri.n, d) > 0.0 {
        scale(tri.n, -1.0)
    } else {
        tri.n
    }
}

/// Camera-space primary ray direction, **not** normalised: `z == 1`, so the ray parameter is
/// the camera-space depth in metres and `Depth32`'s `unit_m` is 1 (spec 3.1, spec 7.2).
pub fn primary_dir(vp: &ViewParams, px: u32, py: u32) -> [f32; 3] {
    let d_cam = [
        (px as f32 + 0.5 - vp.cx) / vp.fx,
        (py as f32 + 0.5 - vp.cy) / vp.fy,
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

fn to_u8(c: f32) -> u8 {
    (255.0 * srgb_encode(c) + 0.5).floor().clamp(0.0, 255.0) as u8
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

/// Path-trace one view (spec 15.3 `PT`, spec 1.9 item 2).
pub fn path_trace(
    scene: &TriScene,
    view: &CameraView,
    cfg: &RenderConfig,
    view_index: u32,
) -> Frame {
    let vp = ViewParams::new(view);
    let (w, h) = (view.spec.width, view.spec.height);
    let bvh = Bvh::build(&scene.tris);
    let g = g_buffer(scene, &bvh, &vp, w, h);
    let (spp, bounces) = (cfg.spp().max(1), cfg.bounces().max(1));
    let (restir, svgf) = match cfg.path {
        RenderPath::Pt { restir, svgf, .. } => (restir, svgf),
        RenderPath::Rs => (false, false),
    };

    let mut radiance = vec![0.0f32; (w as usize) * (h as usize) * 3];
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) as usize;
            let mut acc = [0.0f32; 3];
            for s in 0..spp {
                let mut throughput = [1.0f32; 3];
                let mut o = vp.pos;
                let mut d = primary_dir(&vp, px, py);
                let mut near = vp.near;
                for bounce in 0..bounces {
                    let Some(hit) = nearest_hit(&scene.tris, &bvh, o, d, near, vp.far) else {
                        acc = add(acc, mul(throughput, cfg.sky));
                        break;
                    };
                    let tri = &scene.tris[hit.tri as usize];
                    acc = add(acc, mul(throughput, tri.emission));
                    throughput = mul(throughput, tri.albedo);
                    let n = face_forward(tri, d);
                    o = add(add(o, scale(d, hit.t)), scale(n, RAY_EPS));
                    d = cosine_hemisphere(n, rng::key(cfg.seed, view_index, px, py, s, bounce, 0));
                    near = 0.0;
                }
            }
            // Sequential accumulation in ascending sample order: the order is fixed by the
            // loop, so the cheap sum is also the reproducible one (spec 18.4 is for
            // reductions whose order is not).
            radiance[i * 3..i * 3 + 3].copy_from_slice(&scale(acc, 1.0 / spp as f32));
        }
    }

    if restir {
        radiance = restir_di(scene, &bvh, &vp, cfg, &g, w, h, view_index);
    }
    if svgf {
        radiance = atrous(&radiance, &g, w, h, cfg.svgf_iterations);
    }

    let mut frame = Frame {
        width: w,
        height: h,
        channels: BTreeMap::new(),
    };
    if cfg.channels.contains(&Channel::PtRadiance) {
        frame.channels.insert(
            Channel::PtRadiance,
            Tile {
                shape: [h as usize, w as usize, 3],
                data: TileData::F32(radiance),
            },
        );
    }
    geometry_channels(&g, cfg, w, h, &mut frame);
    frame
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

/// Minimal textbook `ReSTIR` DI: initial candidates, temporal reuse, spatial reuse, then shade.
///
/// Skipped, and this is the list spec 28.6 fills in: MIS weights (the reuse is the biased
/// `1/M` combination, not GRIS pairwise MIS), the bias-correction visibility re-test on
/// reuse, `ReSTIR` GI entirely, light types other than emissive triangles, and reservoir ageing
/// beyond the `M` clamp. Temporal reuse reads the previous frame at the *same* pixel — no
/// motion-vector reprojection — so it is correct only for a static camera, and on the first
/// frame the previous buffer is empty and the pass is a no-op.
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

    // Pass 1: candidates + one shadow ray on the survivor.
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
            if r.w > 0.0 {
                let light = &scene.tris[r.tri as usize];
                let (lp, _, _) = tri_point(light, r.u, r.v);
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
                    r.w = 0.0;
                }
            }
            initial[i] = r;
        }
    }

    // Pass 2 (temporal) is a no-op in the CPU reference: it has no previous frame to read.
    // The GPU renderer keeps one, and `Renderer::render` documents the same caveat.

    // Pass 3: spatial reuse over fixed neighbours.
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) as usize;
            let Some((p, n, albedo)) = hit_of(i, px, py) else {
                continue;
            };
            let key = rng::key(cfg.seed, view_index, px, py, 0, 0, 2);
            let mut combined = Reservoir::default();
            let mut m_sum = 0.0;
            let fold = |src: &Reservoir, slot: u32, combined: &mut Reservoir, m: &mut f32| {
                if src.m <= 0.0 {
                    return;
                }
                let (_, p_hat) = di_contribution(scene, p, n, albedo, src);
                combined.update(
                    src.tri,
                    src.u,
                    src.v,
                    p_hat * src.w * src.m,
                    rng::uniform(key, slot),
                );
                *m += src.m;
            };
            fold(&initial[i], 0, &mut combined, &mut m_sum);
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
                if dz > 0.1 * g.depth[i].abs() || nn < 0.906 {
                    continue;
                }
                fold(&initial[j], k as u32 + 1, &mut combined, &mut m_sum);
            }
            combined.m = m_sum.min(RESTIR_M_CLAMP * (1.0 + RESTIR_NEIGHBOURS.len() as f32));
            let (radiance, p_hat) = di_contribution(scene, p, n, albedo, &combined);
            combined.w = if p_hat > 0.0 && combined.m > 0.0 {
                combined.w_sum / (combined.m * p_hat)
            } else {
                0.0
            };
            spatial[i] = combined;
            let shaded = scale(radiance, combined.w);
            out[i * 3..i * 3 + 3]
                .copy_from_slice(&add(shaded, scene.tris[(g.tri[i] - 1) as usize].emission));
        }
    }
    let _ = spatial;
    out
}

// --- SVGF (the a-trous half) -----------------------------------------------------------------

/// Edge-aware a-trous wavelet filter, `iterations` passes at stride `1 << i`.
///
/// This is **not** SVGF: there is no temporal accumulation, no per-pixel variance estimate,
/// no variance-guided luminance weight, no variance prefilter, no disocclusion handling and
/// no history-driven kernel widening. It is named for spec 28.6's deliverable; what it
/// actually is, is an edge-stopping a-trous filter.
fn atrous(color: &[f32], g: &GBuffer, w: u32, h: u32, iterations: u32) -> Vec<f32> {
    let mut src = color.to_vec();
    let mut dst = src.clone();
    for it in 0..iterations {
        let stride = 1i32 << it;
        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) as usize;
                let mut sum = [0.0f32; 3];
                let mut wsum = 0.0f32;
                for ky in 0..5i32 {
                    for kx in 0..5i32 {
                        let nx = px as i32 + (kx - 2) * stride;
                        let ny = py as i32 + (ky - 2) * stride;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let j = (ny as u32 * w + nx as u32) as usize;
                        let wh = SVGF_H[kx as usize] * SVGF_H[ky as usize];
                        let wz = approx::exp(-(g.depth[i] - g.depth[j]).abs() / SVGF_SIGMA_Z);
                        let mut wn = (g.normal[i * 3] * g.normal[j * 3]
                            + g.normal[i * 3 + 1] * g.normal[j * 3 + 1]
                            + g.normal[i * 3 + 2] * g.normal[j * 3 + 2])
                            .max(0.0);
                        // n^32 by five squarings: no transcendental, exact same ops in Slang.
                        for _ in 0..5 {
                            wn *= wn;
                        }
                        let weight = wh * wz * wn;
                        sum = add(
                            sum,
                            scale([src[j * 3], src[j * 3 + 1], src[j * 3 + 2]], weight),
                        );
                        wsum += weight;
                    }
                }
                let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
                dst[i * 3..i * 3 + 3].copy_from_slice(&scale(sum, inv));
            }
        }
        std::mem::swap(&mut src, &mut dst);
    }
    src
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
