//! Rays against triangles: the camera's primary ray, Moller-Trumbore, the flat scans and the
//! BVH descent that returns exactly what they return, and the uniform triangle sample the light
//! strategies share. `common.slang` is the line-for-line mirror.

use es_math::approx;

use super::{add, cross, dot, quat_rotate, scale, sub};
use crate::bvh::{self, Bvh};
use crate::scene::Tri;
use crate::view::ViewParams;

/// Below this determinant a triangle is edge-on to the ray and is skipped.
const DET_EPS: f32 = 1e-8;

// --- geometry ------------------------------------------------------------------------------

/// A primary or secondary hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub t: f32,
    pub tri: u32,
}

/// Moller-Trumbore. Returns the ray parameter, unfiltered by near/far.
pub(super) fn intersect(tri: &Tri, o: [f32; 3], d: [f32; 3]) -> Option<f32> {
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
    any_hit_but_panels(tris, bvh, o, d, near, far, false)
}

/// [`any_hit`], passing through the `_light` panels ([`Tri::light`]) when `skip_panels`: the
/// [`Shading::Full`] shadow ray's (packet M17/R6). A panel is the path tracer's emitter, the
/// light the directional one stands for, and an emitter does not occlude its own light. The
/// path tracer's rays keep the panel: there it is a surface, and it emits.
pub(super) fn any_hit_but_panels(
    tris: &[Tri],
    bvh: &Bvh,
    o: [f32; 3],
    d: [f32; 3],
    near: f32,
    far: f32,
    skip_panels: bool,
) -> bool {
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
            let tri = &tris[bvh.prim[node.a as usize + k] as usize];
            if skip_panels && tri.light {
                continue;
            }
            if intersect(tri, o, d).is_some_and(|t| t > near && t < far) {
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

/// Uniform point on a triangle from two canonical randoms.
pub(super) fn tri_point(tri: &Tri, u: f32, v: f32) -> ([f32; 3], f32, f32) {
    let su = approx::sqrt(u);
    let (b0, b1) = (1.0 - su, v * su);
    let p = add(
        add(scale(tri.v[0], 1.0 - b0 - b1), scale(tri.v[1], b0)),
        scale(tri.v[2], b1),
    );
    (p, b0, b1)
}

pub(super) fn tri_area(tri: &Tri) -> f32 {
    let c = cross(sub(tri.v[1], tri.v[0]), sub(tri.v[2], tri.v[0]));
    0.5 * approx::sqrt(dot(c, c))
}
