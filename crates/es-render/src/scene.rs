//! `SceneDesc` -> triangles.
//!
//! There is no acceleration structure. Spec 15.4 wants one TLAS over the scene with a shared
//! BLAS per repeated robot, but `es-gpu` exposes compute pipelines only — no ray-tracing
//! extension, no acceleration-structure build — so both render paths scan a flat array in
//! index order. That is the honest ceiling of this packet: a few hundred triangles.

use es_assets::scene::{Body, Geom, SceneDesc, Shape};
use es_core::StableId;
use es_math::{Pose, Vec3};
use std::collections::BTreeMap;

use crate::error::RenderError;

/// Floats per triangle in the flat upload buffer. `v0 v1 v2 n albedo emission seg pad`.
pub const TRI_STRIDE: usize = 20;

/// Tessellation counts. Constants, not quality settings: changing one changes every golden,
/// so it must be a deliberate edit rather than a knob somebody turns.
const SPHERE_SEGMENTS: u32 = 16;
const SPHERE_RINGS: u32 = 8;
const CAP_RINGS: u32 = 4;
/// Half-extent used for a `Plane` that declares itself infinite (`half == 0`).
const INFINITE_PLANE_HALF: f64 = 100.0;

/// A geom whose name ends in this emits light (see `docs/design/renderer.md`). `SceneDesc`
/// has no emissive material field, and inventing one would mean editing `es-assets`, which
/// this packet does not own.
const LIGHT_SUFFIX: &str = "_light";

/// One world-space triangle with everything both shaders need.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tri {
    pub v: [[f32; 3]; 3],
    /// Geometric normal from the winding, unit length.
    pub n: [f32; 3],
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    /// 1-based geom id; `0` is reserved for "no hit" in the segmentation channel.
    pub seg: u32,
}

/// A tessellated scene, ready to upload.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriScene {
    pub tris: Vec<Tri>,
    /// Indices into `tris` of the emissive triangles, ascending (the `ReSTIR` light list).
    pub lights: Vec<u32>,
    /// `seg id -> geom name`, so a segmentation map can be read back to names.
    pub names: BTreeMap<u32, String>,
}

impl TriScene {
    /// Tessellate every geom of every body, in `SceneDesc` body order then geom order.
    pub fn from_scene(scene: &SceneDesc) -> Result<Self, RenderError> {
        let world = world_poses(scene);
        let mut out = Self::default();
        for body in &scene.bodies {
            let body_pose = world.get(&body.id).copied().unwrap_or(Pose::IDENTITY);
            for geom in &body.geoms {
                out.push_geom(geom, body_pose.compose(geom.pose))?;
            }
        }
        Ok(out)
    }

    fn push_geom(&mut self, geom: &Geom, pose: Pose) -> Result<(), RenderError> {
        let seg = u32::try_from(self.names.len() + 1).unwrap_or(u32::MAX);
        let albedo = [
            geom.rgba[0] as f32,
            geom.rgba[1] as f32,
            geom.rgba[2] as f32,
        ];
        let emission = if geom.name.ends_with(LIGHT_SUFFIX) {
            albedo
        } else {
            [0.0; 3]
        };
        let local = tessellate(geom)?;
        self.names.insert(seg, geom.name.clone());
        for [a, b, c] in local {
            let v = [
                to_f32(pose.transform_point(a)),
                to_f32(pose.transform_point(b)),
                to_f32(pose.transform_point(c)),
            ];
            let Some(n) = face_normal(&v) else {
                continue; // degenerate after transform; a zero-area triangle shades nothing
            };
            if emission.iter().any(|e| *e > 0.0) {
                self.lights
                    .push(u32::try_from(self.tris.len()).unwrap_or(u32::MAX));
            }
            self.tris.push(Tri {
                v,
                n,
                albedo,
                emission,
                seg,
            });
        }
        Ok(())
    }

    /// Flat upload buffer, [`TRI_STRIDE`] floats per triangle, segmentation id bitcast into
    /// the second-to-last slot. One buffer, one stride, the same layout on both sides.
    pub fn to_floats(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.tris.len() * TRI_STRIDE);
        for t in &self.tris {
            out.extend_from_slice(&t.v[0]);
            out.extend_from_slice(&t.v[1]);
            out.extend_from_slice(&t.v[2]);
            out.extend_from_slice(&t.n);
            out.extend_from_slice(&t.albedo);
            out.extend_from_slice(&t.emission);
            out.push(f32::from_bits(t.seg));
            out.push(0.0);
        }
        out
    }
}

fn to_f32(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

/// Unit normal from the winding, or `None` for a zero-area triangle.
fn face_normal(v: &[[f32; 3]; 3]) -> Option<[f32; 3]> {
    let e1 = sub(v[1], v[0]);
    let e2 = sub(v[2], v[0]);
    let c = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = es_math::approx::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    (len > 0.0).then(|| [c[0] / len, c[1] / len, c[2] / len])
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// World pose of every body, resolved through the parent chain. `SceneDesc` guarantees no
/// cycles (an importer builds a tree), but a missing parent is treated as world rather than
/// as a panic.
fn world_poses(scene: &SceneDesc) -> BTreeMap<StableId, Pose> {
    let by_id: BTreeMap<StableId, &Body> = scene.bodies.iter().map(|b| (b.id, b)).collect();
    let mut world = BTreeMap::new();
    for body in &scene.bodies {
        let mut chain = Vec::new();
        let mut cursor = Some(body);
        // Bounded by the body count: a malformed cycle stops instead of hanging.
        for _ in 0..=scene.bodies.len() {
            let Some(b) = cursor else { break };
            chain.push(b.pose);
            cursor = b.parent.and_then(|p| by_id.get(&p).copied());
        }
        let pose = chain
            .iter()
            .rev()
            .fold(Pose::IDENTITY, |acc, p| acc.compose(*p));
        world.insert(body.id, pose);
    }
    world
}

/// Local-frame triangles for a primitive.
fn tessellate(geom: &Geom) -> Result<Vec<[Vec3; 3]>, RenderError> {
    let unsupported = |shape| {
        Err(RenderError::UnsupportedShape {
            geom: geom.name.clone(),
            shape,
        })
    };
    match geom.shape {
        Shape::Box { half_extents } => Ok(box_tris(half_extents)),
        Shape::Plane { half_x, half_y, .. } => Ok(plane_tris(
            if half_x > 0.0 {
                half_x
            } else {
                INFINITE_PLANE_HALF
            },
            if half_y > 0.0 {
                half_y
            } else {
                INFINITE_PLANE_HALF
            },
        )),
        Shape::Sphere { radius } => Ok(ellipsoid_tris(Vec3::new(radius, radius, radius))),
        Shape::Ellipsoid { radii } => Ok(ellipsoid_tris(radii)),
        Shape::Capsule {
            radius,
            half_length,
        } => Ok(capsule_tris(radius, half_length, true)),
        Shape::Cylinder {
            radius,
            half_length,
        } => Ok(capsule_tris(radius, half_length, false)),
        Shape::Mesh { .. } => unsupported("Mesh (needs an asset resolver, see the design doc)"),
        Shape::HeightField { .. } => unsupported("HeightField"),
    }
}

/// Two triangles in the local XY plane, normal +Z.
fn plane_tris(hx: f64, hy: f64) -> Vec<[Vec3; 3]> {
    let p = |x: f64, y: f64| Vec3::new(x, y, 0.0);
    vec![
        [p(-hx, -hy), p(hx, -hy), p(hx, hy)],
        [p(-hx, -hy), p(hx, hy), p(-hx, hy)],
    ]
}

/// 12 triangles, all wound counter-clockwise seen from outside.
fn box_tris(h: Vec3) -> Vec<[Vec3; 3]> {
    let c = |sx: f64, sy: f64, sz: f64| Vec3::new(sx * h.x, sy * h.y, sz * h.z);
    // Each face as (a, b, c, d) counter-clockwise from outside.
    let faces = [
        [
            c(1., -1., -1.),
            c(1., 1., -1.),
            c(1., 1., 1.),
            c(1., -1., 1.),
        ], // +X
        [
            c(-1., 1., -1.),
            c(-1., -1., -1.),
            c(-1., -1., 1.),
            c(-1., 1., 1.),
        ], // -X
        [
            c(1., 1., -1.),
            c(-1., 1., -1.),
            c(-1., 1., 1.),
            c(1., 1., 1.),
        ], // +Y
        [
            c(-1., -1., -1.),
            c(1., -1., -1.),
            c(1., -1., 1.),
            c(-1., -1., 1.),
        ], // -Y
        [
            c(-1., -1., 1.),
            c(1., -1., 1.),
            c(1., 1., 1.),
            c(-1., 1., 1.),
        ], // +Z
        [
            c(-1., 1., -1.),
            c(1., 1., -1.),
            c(1., -1., -1.),
            c(-1., -1., -1.),
        ], // -Z
    ];
    faces
        .iter()
        .flat_map(|f| [[f[0], f[1], f[2]], [f[0], f[2], f[3]]])
        .collect()
}

/// UV sphere scaled per axis. `SPHERE_RINGS` latitude bands, `SPHERE_SEGMENTS` longitude.
#[allow(clippy::many_single_char_names)]
fn ellipsoid_tris(r: Vec3) -> Vec<[Vec3; 3]> {
    let point = |ring: u32, seg: u32| {
        let theta = std::f64::consts::PI * f64::from(ring) / f64::from(SPHERE_RINGS);
        let phi = 2.0 * std::f64::consts::PI * f64::from(seg) / f64::from(SPHERE_SEGMENTS);
        Vec3::new(
            r.x * theta.sin() * phi.cos(),
            r.y * theta.sin() * phi.sin(),
            r.z * theta.cos(),
        )
    };
    let mut out = Vec::new();
    for ring in 0..SPHERE_RINGS {
        for seg in 0..SPHERE_SEGMENTS {
            let (a, b) = (point(ring, seg), point(ring, seg + 1));
            let (c, d) = (point(ring + 1, seg + 1), point(ring + 1, seg));
            if ring > 0 {
                out.push([a, b, c]);
            }
            if ring + 1 < SPHERE_RINGS {
                out.push([a, c, d]);
            }
        }
    }
    out
}

/// Cylinder side along local Z, optionally capped with hemispheres (capsule) instead of
/// flat discs (cylinder).
fn capsule_tris(radius: f64, half_length: f64, round_caps: bool) -> Vec<[Vec3; 3]> {
    let ring = |seg: u32, z: f64, r: f64| {
        let phi = 2.0 * std::f64::consts::PI * f64::from(seg) / f64::from(SPHERE_SEGMENTS);
        Vec3::new(r * phi.cos(), r * phi.sin(), z)
    };
    let mut out = Vec::new();
    for seg in 0..SPHERE_SEGMENTS {
        let (a, b) = (
            ring(seg, -half_length, radius),
            ring(seg + 1, -half_length, radius),
        );
        let (c, d) = (
            ring(seg + 1, half_length, radius),
            ring(seg, half_length, radius),
        );
        out.push([a, b, c]);
        out.push([a, c, d]);
    }
    for (sign, z0) in [(1.0_f64, half_length), (-1.0, -half_length)] {
        if round_caps {
            for band in 0..CAP_RINGS {
                for seg in 0..SPHERE_SEGMENTS {
                    let cap = |band: u32, seg: u32| {
                        let t =
                            std::f64::consts::FRAC_PI_2 * f64::from(band) / f64::from(CAP_RINGS);
                        ring(seg, z0 + sign * radius * t.sin(), radius * t.cos())
                    };
                    let (a, b) = (cap(band, seg), cap(band, seg + 1));
                    let (c, d) = (cap(band + 1, seg + 1), cap(band + 1, seg));
                    out.push([a, b, c]);
                    if band + 1 < CAP_RINGS {
                        out.push([a, c, d]);
                    }
                }
            }
        } else {
            for seg in 0..SPHERE_SEGMENTS {
                out.push([
                    Vec3::new(0.0, 0.0, z0),
                    ring(seg, z0, radius),
                    ring(seg + 1, z0, radius),
                ]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_assets::scene::scene_id;

    fn geom(name: &str, shape: Shape, rgba: [f64; 4]) -> Geom {
        Geom {
            id: scene_id("geom", name),
            name: name.to_owned(),
            shape,
            pose: Pose::IDENTITY,
            friction: [1.0, 0.005, 0.0001],
            contype: 1,
            conaffinity: 1,
            condim: 3,
            density: 1000.0,
            mass: None,
            margin: 0.0,
            gap: 0.0,
            solref: [0.02, 1.0],
            solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
            material: None,
            rgba,
            visual_only: false,
        }
    }

    fn body(name: &str, pose: Pose, geoms: Vec<Geom>) -> Body {
        Body {
            id: scene_id("body", name),
            name: name.to_owned(),
            parent: None,
            pose,
            inertial: None,
            geoms,
            sites: Vec::new(),
        }
    }

    fn scene(bodies: Vec<Body>) -> SceneDesc {
        SceneDesc {
            name: "t".to_owned(),
            bodies,
            joints: Vec::new(),
            actuators: Vec::new(),
            sensors: Vec::new(),
            tendons: Vec::new(),
            cameras: Vec::new(),
            assets: Vec::new(),
            options: es_assets::scene::PhysicsOptions::default(),
        }
    }

    #[test]
    fn a_box_is_twelve_outward_facing_triangles() {
        let s = scene(vec![body(
            "b",
            Pose::IDENTITY,
            vec![geom(
                "g",
                Shape::Box {
                    half_extents: Vec3::new(1.0, 1.0, 1.0),
                },
                [0.5, 0.5, 0.5, 1.0],
            )],
        )]);
        let tri = TriScene::from_scene(&s).unwrap();
        assert_eq!(tri.tris.len(), 12);
        // Every face normal points away from the centre.
        for t in &tri.tris {
            let centroid = [
                (t.v[0][0] + t.v[1][0] + t.v[2][0]) / 3.0,
                (t.v[0][1] + t.v[1][1] + t.v[2][1]) / 3.0,
                (t.v[0][2] + t.v[1][2] + t.v[2][2]) / 3.0,
            ];
            let dot = centroid[0] * t.n[0] + centroid[1] * t.n[1] + centroid[2] * t.n[2];
            assert!(dot > 0.0, "inward-facing triangle: {t:?}");
        }
        assert_eq!(tri.names.get(&1).map(String::as_str), Some("g"));
        assert!(tri.lights.is_empty());
    }

    #[test]
    fn segmentation_ids_are_dense_and_one_based() {
        let s = scene(vec![body(
            "b",
            Pose::IDENTITY,
            vec![
                geom("a", Shape::Sphere { radius: 1.0 }, [1.0, 0.0, 0.0, 1.0]),
                geom("b", Shape::Sphere { radius: 1.0 }, [0.0, 1.0, 0.0, 1.0]),
            ],
        )]);
        let tri = TriScene::from_scene(&s).unwrap();
        let ids: std::collections::BTreeSet<u32> = tri.tris.iter().map(|t| t.seg).collect();
        assert_eq!(ids, [1, 2].into_iter().collect());
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn light_suffix_makes_a_geom_emissive() {
        let s = scene(vec![body(
            "b",
            Pose::IDENTITY,
            vec![geom(
                "ceiling_light",
                Shape::Plane {
                    half_x: 1.0,
                    half_y: 1.0,
                    grid: 0.0,
                },
                [1.0, 0.9, 0.8, 1.0],
            )],
        )]);
        let tri = TriScene::from_scene(&s).unwrap();
        assert_eq!(tri.lights, vec![0, 1]);
        assert_eq!(tri.tris[0].emission, [1.0, 0.9, 0.8]);
    }

    #[test]
    fn mesh_and_height_field_are_refused_by_name() {
        let s = scene(vec![body(
            "b",
            Pose::IDENTITY,
            vec![geom(
                "m",
                Shape::Mesh {
                    asset: scene_id("asset", "mesh/x"),
                },
                [1.0; 4],
            )],
        )]);
        let err = TriScene::from_scene(&s).unwrap_err();
        assert!(matches!(err, RenderError::UnsupportedShape { ref geom, .. } if geom == "m"));
    }

    #[test]
    fn flat_buffer_stride_matches_the_shader() {
        let s = scene(vec![body(
            "b",
            Pose::IDENTITY,
            vec![geom(
                "g",
                Shape::Box {
                    half_extents: Vec3::new(1.0, 1.0, 1.0),
                },
                [0.25, 0.5, 0.75, 1.0],
            )],
        )]);
        let tri = TriScene::from_scene(&s).unwrap();
        let floats = tri.to_floats();
        assert_eq!(floats.len(), tri.tris.len() * TRI_STRIDE);
        assert_eq!(floats[18].to_bits(), 1);
    }

    #[test]
    fn nested_bodies_compose_world_poses() {
        let mut child = body(
            "child",
            Pose::new(Vec3::new(1.0, 0.0, 0.0), es_math::Quat::IDENTITY),
            vec![geom("g", Shape::Sphere { radius: 0.1 }, [1.0; 4])],
        );
        let parent = body(
            "parent",
            Pose::new(Vec3::new(0.0, 2.0, 0.0), es_math::Quat::IDENTITY),
            Vec::new(),
        );
        child.parent = Some(parent.id);
        let tri = TriScene::from_scene(&scene(vec![parent, child])).unwrap();
        let any = tri.tris[0].v[0];
        assert!((any[0] - 1.0).abs() < 0.2 && (any[1] - 2.0).abs() < 0.2);
    }
}
