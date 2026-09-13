//! Linear blend skinning of Gaussians onto rigid bodies (spec 16.2).
//!
//! Spec 16.2's first design decision: **splats do not replace physics.** Each Gaussian rides
//! one to four rigid bodies; the physics backend of spec 17 says where those bodies are, and
//! the visual follows. This is the CPU path — the same arithmetic belongs in a compute shader
//! next to the rasteriser, which does not exist yet.

use std::collections::BTreeMap;

use es_assets::scene::{Geom, SceneDesc, Shape};
use es_core::StableId;
use es_math::{Pose, Quat, Vec3};

use crate::SplatScene;

/// Bodies at most this far from a Gaussian's surface own it outright: weight 1.0, one body.
///
/// Both the degenerate-case guard for `1 / d` and the common case — a splat reconstructed
/// from photos of a link *is* on that link.
pub const WELD_EPS: f64 = 1e-9;

/// The maximum number of bodies one Gaussian may be bound to.
pub const MAX_INFLUENCES: usize = 4;

/// Which bodies drive each Gaussian, and how strongly.
#[derive(Clone, Debug)]
pub struct Binding {
    /// Index space for [`Binding::indices`].
    pub bodies: Vec<StableId>,
    /// Each body's world pose when [`Binding::bind`] ran. Skinning is relative to this.
    pub rest: Vec<Pose>,
    /// Per Gaussian, up to [`MAX_INFLUENCES`] body indices; unused slots repeat slot 0.
    pub indices: Vec<[u16; MAX_INFLUENCES]>,
    /// Per Gaussian, weights summing to 1; unused slots are 0.
    pub weights: Vec<[f32; MAX_INFLUENCES]>,
}

/// Skinned Gaussian centres and orientations, same layout as the matching [`SplatScene`]
/// fields. Scales and colours are unchanged by skinning, so they are not copied.
#[derive(Clone, Debug)]
pub struct SkinnedSplats {
    /// `3 * count`, metres.
    pub positions: Vec<f32>,
    /// `4 * count`, xyzw, unit norm, `w >= 0`.
    pub rotations: Vec<f32>,
}

impl Binding {
    /// Binds every Gaussian to its `k` nearest bodies, by distance to their nearest geom
    /// surface, with inverse-distance weights normalised to 1. `k` is clamped to
    /// `1..=MAX_INFLUENCES`.
    ///
    /// A body with no geoms is measured from its origin, so no body is unbindable. Surface
    /// distance is exact for `Sphere`, `Box` and `Capsule` and falls back to the geom origin
    /// otherwise — see [`geom_distance`].
    pub fn bind(scene: &SplatScene, desc: &SceneDesc, k: usize) -> Self {
        let k = k.clamp(1, MAX_INFLUENCES);
        let world = world_poses(desc);
        let bodies: Vec<StableId> = desc.bodies.iter().map(|b| b.id).collect();

        let mut out = Self {
            bodies,
            rest: world.clone(),
            indices: Vec::with_capacity(scene.len()),
            weights: Vec::with_capacity(scene.len()),
        };
        if out.bodies.is_empty() {
            // Nothing to ride. Every Gaussian keeps its rest position under `skin`.
            out.indices.resize(scene.len(), [0; MAX_INFLUENCES]);
            out.weights.resize(scene.len(), [0.0; MAX_INFLUENCES]);
            return out;
        }

        let mut ranked: Vec<(f64, usize)> = Vec::with_capacity(desc.bodies.len());
        for g in 0..scene.len() {
            let p = scene.position(g);
            ranked.clear();
            for (bi, body) in desc.bodies.iter().enumerate() {
                let pose = world[bi];
                let d = if body.geoms.is_empty() {
                    (p - pose.position).norm()
                } else {
                    body.geoms
                        .iter()
                        .map(|geom| {
                            geom_distance(
                                geom,
                                pose.compose(geom.pose).inverse().transform_point(p),
                            )
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                ranked.push((d.max(0.0), bi));
            }
            // Ties break on body index, so the binding does not depend on sort stability.
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

            let mut idx = [0u16; MAX_INFLUENCES];
            let mut w = [0.0f32; MAX_INFLUENCES];
            let take = k.min(ranked.len());
            for slot in 0..take {
                idx[slot] = ranked[slot].1 as u16;
            }
            for slot in take..MAX_INFLUENCES {
                idx[slot] = idx[0];
            }
            if ranked[0].0 <= WELD_EPS {
                w[0] = 1.0;
            } else {
                let total: f64 = ranked[..take].iter().map(|(d, _)| 1.0 / d).sum();
                for slot in 0..take {
                    w[slot] = (1.0 / ranked[slot].0 / total) as f32;
                }
            }
            out.indices.push(idx);
            out.weights.push(w);
        }
        out
    }
}

/// Applies `body_poses` to a bound capture by linear blend skinning.
///
/// Per body, `delta = pose_now * rest^-1`; per Gaussian, positions blend linearly and
/// orientations by a weighted quaternion sum, sign-aligned to the highest-weight body and
/// renormalised. That quaternion blend is the standard LBS approximation — not a proper
/// rotation average, and it shrinks under large relative rotations, which for a splat means a
/// slightly wrong ellipsoid orientation between two bodies rotating apart. Dual quaternion
/// skinning is the fix if it ever shows up in a `domain_gap` number (spec 10); it is not worth
/// the code before then.
///
/// A body missing from `body_poses` contributes its rest pose, so a partial pose map leaves
/// those Gaussians where they were instead of collapsing them onto the origin.
pub fn skin(
    binding: &Binding,
    scene: &SplatScene,
    body_poses: &BTreeMap<StableId, Pose>,
) -> SkinnedSplats {
    let deltas: Vec<Pose> = binding
        .bodies
        .iter()
        .zip(&binding.rest)
        .map(|(id, rest)| {
            body_poses
                .get(id)
                .map_or(Pose::IDENTITY, |now| now.compose(rest.inverse()))
        })
        .collect();

    let count = scene.len();
    let mut out = SkinnedSplats {
        positions: Vec::with_capacity(3 * count),
        rotations: Vec::with_capacity(4 * count),
    };
    for gi in 0..count {
        let rest_pos = scene.position(gi);
        let rest_rot = Quat::from_xyzw(
            f64::from(scene.rotations[4 * gi]),
            f64::from(scene.rotations[4 * gi + 1]),
            f64::from(scene.rotations[4 * gi + 2]),
            f64::from(scene.rotations[4 * gi + 3]),
        );
        let (idx, weights) = (binding.indices[gi], binding.weights[gi]);

        let mut acc_p = Vec3::ZERO;
        let mut acc_q = [0.0f64; 4];
        let mut lead: Option<Quat> = None;
        let mut total = 0.0f64;
        for slot in 0..MAX_INFLUENCES {
            let weight = f64::from(weights[slot]);
            if weight == 0.0 {
                continue;
            }
            let delta = deltas
                .get(idx[slot] as usize)
                .copied()
                .unwrap_or(Pose::IDENTITY);
            acc_p = acc_p + delta.transform_point(rest_pos).scale(weight);
            let mut rotated = delta.orientation * rest_rot;
            // Align to the dominant influence before summing: `r` and `-r` are the same
            // rotation but cancel in a linear blend.
            match lead {
                None => lead = Some(rotated),
                Some(first) => {
                    let dot = first.x * rotated.x
                        + first.y * rotated.y
                        + first.z * rotated.z
                        + first.w * rotated.w;
                    if dot < 0.0 {
                        rotated = Quat::from_xyzw(-rotated.x, -rotated.y, -rotated.z, -rotated.w);
                    }
                }
            }
            acc_q[0] += weight * rotated.x;
            acc_q[1] += weight * rotated.y;
            acc_q[2] += weight * rotated.z;
            acc_q[3] += weight * rotated.w;
            total += weight;
        }
        if total <= 0.0 {
            acc_p = rest_pos;
            acc_q = [rest_rot.x, rest_rot.y, rest_rot.z, rest_rot.w];
        }
        let blended = Quat::from_xyzw(acc_q[0], acc_q[1], acc_q[2], acc_q[3]).normalize();
        out.positions
            .extend_from_slice(&[acc_p.x as f32, acc_p.y as f32, acc_p.z as f32]);
        out.rotations.extend_from_slice(&[
            blended.x as f32,
            blended.y as f32,
            blended.z as f32,
            blended.w as f32,
        ]);
    }
    out
}

/// World pose per body, in `desc.bodies` order.
///
/// A fixed-point pass rather than a recursion: importers emit parents before children, but
/// nothing in [`SceneDesc`] promises it, and a cycle must terminate rather than overflow the
/// stack. A body whose parent never resolves keeps its local pose.
fn world_poses(desc: &SceneDesc) -> Vec<Pose> {
    let index: BTreeMap<StableId, usize> = desc
        .bodies
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();
    let mut world: Vec<Pose> = desc.bodies.iter().map(|b| b.pose).collect();
    let mut done: Vec<bool> = desc.bodies.iter().map(|b| b.parent.is_none()).collect();
    for _ in 0..desc.bodies.len() {
        let mut progressed = false;
        for (i, body) in desc.bodies.iter().enumerate() {
            if done[i] {
                continue;
            }
            let Some(parent) = body.parent.and_then(|p| index.get(&p).copied()) else {
                done[i] = true;
                progressed = true;
                continue;
            };
            if done[parent] {
                world[i] = world[parent].compose(body.pose);
                done[i] = true;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    world
}

/// Distance from `p` (in the geom's own frame) to the geom's surface: negative inside.
///
/// Exact for `Sphere`, `Box` and `Capsule`. Every other shape falls back to the distance from
/// the geom origin — spec 16.2's answer for object geometry is a convex-decomposition proxy,
/// which belongs to the packet that builds one. Approximating a mesh with something that
/// looks exact and is not would be worse than a documented fallback.
fn geom_distance(geom: &Geom, p: Vec3) -> f64 {
    match geom.shape {
        Shape::Sphere { radius } => p.norm() - radius,
        Shape::Box { half_extents } => {
            let q = Vec3::new(
                p.x.abs() - half_extents.x,
                p.y.abs() - half_extents.y,
                p.z.abs() - half_extents.z,
            );
            let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
            outside + q.x.max(q.y).max(q.z).min(0.0)
        }
        Shape::Capsule {
            radius,
            half_length,
        } => {
            // Axis along local z, following the MJCF/URDF convention `es-assets` imports.
            let axis = Vec3::new(0.0, 0.0, p.z.clamp(-half_length, half_length));
            (p - axis).norm() - radius
        }
        _ => p.norm(),
    }
}
