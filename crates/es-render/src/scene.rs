//! `SceneDesc` -> triangles.
//!
//! The acceleration structure is a software one: spec 15.4 wants one TLAS over the scene with
//! a shared BLAS per repeated robot, but `es-gpu` exposes compute pipelines only — no
//! ray-tracing extension, no acceleration-structure build — so both render paths traverse
//! [`crate::bvh::Bvh`], built here on the CPU per frame (packet M7/R1), and its answer is the
//! flat index-order scan's answer bit for bit.

use es_assets::scene::{Body, Geom, SceneDesc, Shape};
use es_core::StableId;
use es_math::approx::{self, coeffs::PI};
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
        Self::from_scene_with_poses(scene, &BTreeMap::new())
    }

    /// [`Self::from_scene`] with the body poses of a *running* env: a body listed in `world`
    /// is placed at that world pose, a body absent from it keeps the scene's static pose. The
    /// tessellation, the geom order and the segmentation ids are the same either way, so a
    /// posed frame and a static frame are comparable pixel for pixel.
    ///
    /// A map rather than a `StateView`: this crate is layer 5 and the physics state is layer 3
    /// (spec 4.2), so the caller does the lookup and `es-render` gains no dependency.
    ///
    /// One call, one tessellation of every geom. A caller that renders frame after frame of
    /// the *same* scene should keep a [`SceneCache`] and call [`SceneCache::tri_scene`]
    /// instead, which reuses the local tessellation and produces the same bits.
    pub fn from_scene_with_poses(
        scene: &SceneDesc,
        world: &BTreeMap<StableId, Pose>,
    ) -> Result<Self, RenderError> {
        SceneCache::default().tri_scene(scene, world)
    }

    fn push_geom(&mut self, geom: &Geom, pose: Pose, local: &[[Vec3; 3]]) {
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
        self.names.insert(seg, geom.name.clone());
        for &[a, b, c] in local {
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

/// The per-geom **local** tessellation, kept across frames (packet M7/R1 step 1).
///
/// [`tessellate`] is a pure function of `geom.shape`, and a replay re-poses the same geoms
/// tick after tick, so recomputing it every frame recomputes the same `approx::{sin, cos}`
/// vertices. The cache holds the local triangles; the pose is still applied per frame, in
/// `f64`, by the same [`TriScene::push_geom`] — so a cached frame is **bit-identical** to an
/// uncached one (`bvh_and_cache_tests::cached_tessellation_is_bit_identical`).
///
/// A struct, not a trait: `INV-17` allows seven extension points and this is none of them.
/// `BTreeMap`, never `HashMap` (spec 3.4). The stored [`Shape`] is checked on every hit: geom
/// ids come from names (`es_assets::scene::scene_id`), so two scenes can share one, and a
/// stale entry would be a silently wrong mesh rather than a miss.
#[derive(Clone, Debug, Default)]
pub struct SceneCache {
    local: BTreeMap<StableId, (Shape, Vec<[Vec3; 3]>)>,
}

impl SceneCache {
    /// [`TriScene::from_scene_with_poses`], reusing whatever this cache already holds.
    pub fn tri_scene(
        &mut self,
        scene: &SceneDesc,
        world: &BTreeMap<StableId, Pose>,
    ) -> Result<TriScene, RenderError> {
        let statics = world_poses(scene);
        let mut out = TriScene::default();
        for body in &scene.bodies {
            let body_pose = world
                .get(&body.id)
                .or_else(|| statics.get(&body.id))
                .copied()
                .unwrap_or(Pose::IDENTITY);
            for geom in &body.geoms {
                if !matches!(self.local.get(&geom.id), Some((s, _)) if *s == geom.shape) {
                    let tris = tessellate(geom)?;
                    self.local.insert(geom.id, (geom.shape, tris));
                }
                let local = &self.local[&geom.id].1;
                out.push_geom(geom, body_pose.compose(geom.pose), local);
            }
        }
        Ok(out)
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
    let len = approx::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
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

/// Longitude angle of segment `seg`, in `f32`. `seg == SPHERE_SEGMENTS` is the wrap-around
/// copy of segment 0 and is folded onto it, so the seam closes exactly instead of on a
/// `sin(2*pi)` residue.
fn phi_of(seg: u32) -> f32 {
    2.0 * PI * (seg % SPHERE_SEGMENTS) as f32 / SPHERE_SEGMENTS as f32
}

/// UV sphere scaled per axis. `SPHERE_RINGS` latitude bands, `SPHERE_SEGMENTS` longitude.
///
/// The unit direction is computed entirely in `f32` through [`approx`] — never the host
/// `libm` (spec 3.2, 3.4) — then widened exactly and scaled by the `f64` radii. These
/// vertices are what `TriScene::to_floats` uploads to the GPU *and* what the CPU reference
/// traverses, so a host-dependent `sin` here would desynchronize the two paths.
#[allow(clippy::many_single_char_names)]
fn ellipsoid_tris(r: Vec3) -> Vec<[Vec3; 3]> {
    let point = |ring: u32, seg: u32| {
        let theta = PI * ring as f32 / SPHERE_RINGS as f32;
        let phi = phi_of(seg);
        let (st, ct) = (approx::sin(theta), approx::cos(theta));
        let (sp, cp) = (approx::sin(phi), approx::cos(phi));
        Vec3::new(
            r.x * f64::from(st * cp),
            r.y * f64::from(st * sp),
            r.z * f64::from(ct),
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
        let phi = phi_of(seg);
        Vec3::new(
            r * f64::from(approx::cos(phi)),
            r * f64::from(approx::sin(phi)),
            z,
        )
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
                        let t = 0.5 * PI * band as f32 / CAP_RINGS as f32;
                        ring(
                            seg,
                            z0 + sign * radius * f64::from(approx::sin(t)),
                            radius * f64::from(approx::cos(t)),
                        )
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
            priority: 0,
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
            meshes: BTreeMap::new(),
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

    /// A mesh asset in `scene.meshes`, and the `AssetRef` naming it with `hash`.
    fn with_mesh(
        mut s: SceneDesc,
        name: &str,
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        hash: [u8; 32],
    ) -> SceneDesc {
        let id = scene_id("asset", name);
        s.assets.push(es_assets::scene::AssetRef {
            id,
            name: name.to_owned(),
            kind: es_assets::scene::AssetKind::Mesh,
            path: format!("{name}.stl"),
            hash,
        });
        s.meshes.insert(
            id,
            es_assets::gltf::MeshData {
                id,
                name: name.to_owned(),
                positions,
                normals: None,
                uvs: None,
                indices,
                material: None,
            },
        );
        s
    }

    /// A tetrahedron with legs of `scale`, every face wound counter-clockwise from outside.
    fn tetra(scale: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
        (
            vec![
                [0.0, 0.0, 0.0],
                [scale, 0.0, 0.0],
                [0.0, scale, 0.0],
                [0.0, 0.0, scale],
            ],
            vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
        )
    }

    fn mesh_geom(name: &str, asset: &str) -> Geom {
        geom(
            name,
            Shape::Mesh {
                asset: scene_id("asset", asset),
            },
            [0.5, 0.5, 0.5, 1.0],
        )
    }

    /// Packet M10/W2b oracle 1: the `Mesh` arm reads `scene.meshes` — one triangle per index
    /// triple, the same dense segmentation ids as a primitive, normals from the winding.
    #[test]
    fn mesh_tessellates_from_the_scene_store() {
        let (positions, indices) = tetra(1.0);
        let s = with_mesh(
            scene(vec![body(
                "b",
                Pose::new(Vec3::new(0.0, 0.0, 1.0), es_math::Quat::IDENTITY),
                vec![
                    mesh_geom("m", "tet"),
                    geom("s", Shape::Sphere { radius: 0.1 }, [1.0; 4]),
                ],
            )]),
            "tet",
            positions,
            indices,
            [7; 32],
        );
        let tri = TriScene::from_scene(&s).unwrap();
        let mesh: Vec<&Tri> = tri.tris.iter().filter(|t| t.seg == 1).collect();
        assert_eq!(mesh.len(), 4);
        let ids: std::collections::BTreeSet<u32> = tri.tris.iter().map(|t| t.seg).collect();
        assert_eq!(ids, [1, 2].into_iter().collect());
        assert_eq!(tri.names.get(&1).map(String::as_str), Some("m"));
        // The body pose lifted every vertex by 1; each normal points away from the centroid.
        let c = [0.25_f32, 0.25, 1.25];
        for t in mesh {
            let f = [
                (t.v[0][0] + t.v[1][0] + t.v[2][0]) / 3.0 - c[0],
                (t.v[0][1] + t.v[1][1] + t.v[2][1]) / 3.0 - c[1],
                (t.v[0][2] + t.v[1][2] + t.v[2][2]) / 3.0 - c[2],
            ];
            assert!(
                f[0] * t.n[0] + f[1] * t.n[1] + f[2] * t.n[2] > 0.0,
                "{t:?}"
            );
        }
    }

    #[test]
    fn mesh_without_store_is_refused_by_name() {
        let s = scene(vec![body("b", Pose::IDENTITY, vec![mesh_geom("m", "x")])]);
        let err = TriScene::from_scene(&s).unwrap_err();
        assert!(
            matches!(err, RenderError::UnsupportedShape { ref geom, shape }
                if geom == "m" && shape.contains("mesh::load")),
            "{err}"
        );
        let hf = scene(vec![body(
            "b",
            Pose::IDENTITY,
            vec![geom(
                "h",
                Shape::HeightField {
                    asset: scene_id("asset", "hf"),
                },
                [1.0; 4],
            )],
        )]);
        let err = TriScene::from_scene(&hf).unwrap_err();
        assert!(matches!(err, RenderError::UnsupportedShape { ref geom, .. } if geom == "h"));
    }

    /// Geom ids come from names, so two scenes can share one: a mesh geom whose asset content
    /// differs must not be served the other scene's cached triangles.
    #[test]
    fn scene_cache_keys_on_mesh_content() {
        let make = |scale: f32, hash: u8| {
            let (p, i) = tetra(scale);
            with_mesh(
                scene(vec![body("b", Pose::IDENTITY, vec![mesh_geom("m", "tet")])]),
                "tet",
                p,
                i,
                [hash; 32],
            )
        };
        let (small, big) = (make(1.0, 1), make(2.0, 2));
        let mut cache = SceneCache::default();
        let none = BTreeMap::new();
        let a = cache.tri_scene(&small, &none).unwrap();
        let b = cache.tri_scene(&big, &none).unwrap();
        assert_eq!(a, TriScene::from_scene(&small).unwrap());
        assert_eq!(b, TriScene::from_scene(&big).unwrap());
        assert_ne!(a, b, "the cache served the first scene's mesh to the second");
    }

    /// The tessellation feeds both paths: `to_floats` is what `Renderer::upload_tris`
    /// hands the GPU, `tris` is what `cpu::rasterize` traverses. Asserting the upload
    /// buffer carries the vertices bit for bit is what makes "same vertices on both sides"
    /// a test rather than a comment — and the repeat pass pins that the `f32`
    /// `approx::{sin, cos}` tessellation is reproducible (spec 3.4).
    #[test]
    fn curved_shapes_upload_the_vertices_the_cpu_reference_traverses() {
        let s = scene(vec![body(
            "b",
            Pose::IDENTITY,
            vec![
                geom("s", Shape::Sphere { radius: 0.37 }, [1.0; 4]),
                geom(
                    "c",
                    Shape::Capsule {
                        radius: 0.21,
                        half_length: 0.6,
                    },
                    [1.0; 4],
                ),
                geom(
                    "y",
                    Shape::Cylinder {
                        radius: 0.21,
                        half_length: 0.6,
                    },
                    [1.0; 4],
                ),
                mesh_geom("m", "tet"),
            ],
        )]);
        let (positions, indices) = tetra(0.3);
        let s = with_mesh(s, "tet", positions.clone(), indices.clone(), [3; 32]);
        let tri = TriScene::from_scene(&s).unwrap();
        // Packet M10/W2b: at the identity pose a mesh vertex is the file's `f32`, bit for bit
        // (`f64::from(f32)` is exact and the round trip back is too).
        let mesh: Vec<&Tri> = tri.tris.iter().filter(|t| t.seg == 4).collect();
        assert_eq!(mesh.len(), indices.len() / 3);
        for (t, face) in mesh.iter().zip(indices.chunks(3)) {
            for (v, &i) in t.v.iter().zip(face) {
                let want = positions[i as usize].map(f32::to_bits);
                assert_eq!(v.map(f32::to_bits), want, "mesh vertex {i}");
            }
        }
        let floats = tri.to_floats();
        assert!(tri.tris.len() > 200, "{} triangles", tri.tris.len());
        for (i, t) in tri.tris.iter().enumerate() {
            for (j, v) in t.v.iter().flatten().enumerate() {
                let got = floats[i * TRI_STRIDE + j];
                assert_eq!(
                    got.to_bits(),
                    v.to_bits(),
                    "triangle {i} component {j}: upload buffer differs from the CPU vertex"
                );
                assert!(v.is_finite(), "triangle {i} component {j} is not finite");
            }
        }
        assert!(
            TriScene::from_scene(&s).unwrap() == tri,
            "tessellation is not reproducible"
        );
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
