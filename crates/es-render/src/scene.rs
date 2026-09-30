//! `SceneDesc` -> triangles.
//!
//! The acceleration structure is a software one: spec 15.4 wants one TLAS over the scene with
//! a shared BLAS per repeated robot, but `es-gpu` exposes compute pipelines only — no
//! ray-tracing extension, no acceleration-structure build — so both render paths traverse
//! [`crate::bvh::Bvh`], built here on the CPU per frame (packet M7/R1), and its answer is the
//! flat index-order scan's answer bit for bit.
//!
//! Since plan H's HT1 every triangle also carries three texture coordinates and a material
//! slot (0 = none). A geom without a drawn material gets zeros there and renders exactly as
//! before: the slot is read, never computed with.

// `MuJoCo`'s texture coordinates are transcribed as its source writes them (`(x + 1) * 0.5`,
// not `midpoint`), with its one-letter names (`a`, `b`, `c` corners, `s` sign, `t` angle).
#![allow(clippy::manual_midpoint, clippy::many_single_char_names)]

use es_assets::scene::{Body, Geom, SceneDesc, Shape};
use es_core::StableId;
use es_math::approx::{self, coeffs::PI};
use es_math::{Pose, Vec3};
use std::collections::BTreeMap;
use std::sync::Arc;

/// One tessellated vertex: local position, `MuJoCo`'s UV, the unit-object coordinate.
type Vert = (Vec3, [f32; 2], [f32; 3]);
/// A box corner and its signs.
type Corner = (Vec3, [f64; 3]);

use crate::error::RenderError;
use crate::material::{Look, Materials};

/// Floats per triangle in the flat upload buffer:
/// `v0 v1 v2 n albedo emission seg mat tc0 tc1 tc2 pad` (packet HT1 grew it from 20).
pub const TRI_STRIDE: usize = 32;

/// Tessellation counts. Constants, not quality settings: changing one changes every golden,
/// so it must be a deliberate edit rather than a knob somebody turns.
const SPHERE_SEGMENTS: u32 = 16;
const SPHERE_RINGS: u32 = 8;
const CAP_RINGS: u32 = 4;
/// Half-extent used for a `Plane` that declares itself infinite (`half == 0`).
const INFINITE_PLANE_HALF: f64 = 100.0;
/// `MuJoCo`'s default geom `rgba`: a geom that writes anything else overrides its material's.
const DEFAULT_RGBA: [f64; 4] = [0.5, 0.5, 0.5, 1.0];

/// A geom whose name ends in this emits light (see `docs/design/renderer.md`). A drawn
/// material's `emission` is the other way (plan H, HT1).
const LIGHT_SUFFIX: &str = "_light";

/// One world-space triangle with everything both shaders need.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tri {
    pub v: [[f32; 3]; 3],
    /// Geometric normal from the winding, unit length.
    pub n: [f32; 3],
    /// The flat colour, and a drawn material's base colour factor.
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    /// 1-based geom id; `0` is reserved for "no hit" in the segmentation channel.
    pub seg: u32,
    /// 1 + the index of the triangle's material in [`TriScene::materials`]; 0 for none.
    pub mat: u32,
    /// Per-vertex texture coordinates: `(s, t, 0)` for a 2D texture, the cube-map direction
    /// `(s, t, r)` for a cube one (packet HT1).
    pub tc: [[f32; 3]; 3],
}

/// A tessellated scene, ready to upload.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriScene {
    pub tris: Vec<Tri>,
    /// Indices into `tris` of the emissive triangles, ascending (the `ReSTIR` light list).
    pub lights: Vec<u32>,
    /// `seg id -> geom name`, so a segmentation map can be read back to names.
    pub names: BTreeMap<u32, String>,
    /// The materials and textures `Tri::mat` points into, shared across frames. Empty for a
    /// scene without drawn materials, and then nothing is uploaded for it.
    pub materials: Arc<Materials>,
}

/// One local-frame triangle of a geom's tessellation, with what texturing needs: `MuJoCo`'s
/// own texture coordinate of each vertex before `texrepeat` (`uv`), and the vertex in the
/// unit object `MuJoCo` draws and scales (`unit`), which its cube mapping reads.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LocalTri {
    p: [Vec3; 3],
    uv: [[f32; 2]; 3],
    unit: [[f32; 3]; 3],
}

/// How one geom is drawn, resolved once per frame from its material.
struct Wear {
    rgba: [f64; 4],
    emission: f64,
    /// A drawn material's emitted colour, replacing `emission x rgba` (plan H, HT2).
    emissive: Option<[f64; 3]>,
    look: Option<Look>,
    /// `MuJoCo`'s `mjvGeom` size: what `texuniform` multiplies by.
    size: [f32; 3],
    /// Plane of infinite extent: its texture matrix is shifted by `-0.5`.
    infinite: bool,
    /// A mesh without UVs: texture coordinates are generated from the position.
    texgen: bool,
}

impl Wear {
    /// The texture coordinate of one local vertex.
    fn tc(&self, uv: [f32; 2], unit: [f32; 3]) -> [f32; 3] {
        let Some(look) = self.look else {
            return [0.0; 3];
        };
        if look.cube {
            let k = if look.texuniform { self.size } else { [1.0; 3] };
            return [unit[0] * k[0], unit[1] * k[1], unit[2] * k[2]];
        }
        let mut scl = [look.texrepeat[0] as f32, look.texrepeat[1] as f32];
        if self.texgen {
            // `settexture`'s 2D path for a mesh without UVs: repeat per object size, then the
            // object-linear planes (0.5 s, 0, 0, -0.5) and (0, -0.5 t, 0, -0.5).
            for (s, size) in scl.iter_mut().zip(self.size) {
                if size > 0.0 {
                    *s /= size;
                }
            }
            if look.texuniform {
                for (s, size) in scl.iter_mut().zip(self.size) {
                    if size > 0.0 {
                        *s *= size;
                    }
                }
            }
            return [
                0.5 * scl[0] * unit[0] - 0.5,
                -0.5 * scl[1] * unit[1] - 0.5,
                0.0,
            ];
        }
        for s in &mut scl {
            if *s <= 0.0 {
                *s = 1.0;
            }
        }
        if look.texuniform {
            for (s, size) in scl.iter_mut().zip(self.size) {
                if size > 0.0 {
                    *s *= size;
                }
            }
        }
        let off = if self.infinite { -0.5 } else { 0.0 };
        [uv[0] * scl[0] + off, uv[1] * scl[1] + off, 0.0]
    }
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

    fn push_geom(&mut self, geom: &Geom, wear: &Wear, pose: Pose, local: &[LocalTri]) {
        let seg = u32::try_from(self.names.len() + 1).unwrap_or(u32::MAX);
        let albedo = [
            wear.rgba[0] as f32,
            wear.rgba[1] as f32,
            wear.rgba[2] as f32,
        ];
        let emission = if geom.name.ends_with(LIGHT_SUFFIX) {
            albedo
        } else if let Some(e) = wear.emissive {
            e.map(|c| c as f32)
        } else if wear.emission > 0.0 {
            [0, 1, 2].map(|c| (wear.emission * wear.rgba[c]) as f32)
        } else {
            [0.0; 3]
        };
        let mat = wear.look.map_or(0, |l| l.mat);
        self.names.insert(seg, geom.name.clone());
        for lt in local {
            let [a, b, c] = lt.p;
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
            let tc = [0, 1, 2].map(|k| wear.tc(lt.uv[k], lt.unit[k]));
            self.tris.push(Tri {
                v,
                n,
                albedo,
                emission,
                seg,
                mat,
                tc,
            });
        }
    }

    /// Flat upload buffer, [`TRI_STRIDE`] floats per triangle, segmentation id and material
    /// slot bitcast into slots 18 and 19. One buffer, one stride, the same layout on both
    /// sides.
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
            out.push(f32::from_bits(t.mat));
            for tc in &t.tc {
                out.extend_from_slice(tc);
            }
            out.extend_from_slice(&[0.0; 3]);
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
/// `BTreeMap`, never `HashMap` (spec 3.4). The stored [`Shape`] — and for a `Mesh` the
/// asset's content hash (packet M10/W2b) — is checked on every hit: geom ids come from names
/// (`es_assets::scene::scene_id`), so two scenes can share one, and a stale entry would be a
/// silently wrong mesh rather than a miss.
///
/// It also keeps the scene's material table (packet HT1), rebuilt only when the drawn
/// materials or a texture's content digest change, so a replay uploads one shared table.
#[derive(Clone, Debug, Default)]
pub struct SceneCache {
    local: BTreeMap<StableId, (CacheKey, Vec<LocalTri>)>,
    materials: Option<(MaterialKey, Arc<Materials>, BTreeMap<StableId, Look>)>,
}

/// The geom's shape and, for a `Mesh`, its asset's content hash (zeros otherwise).
type CacheKey = (Shape, [u8; 32]);
/// The drawn materials and every texture asset's digest.
type MaterialKey = (
    BTreeMap<StableId, es_assets::scene::Material>,
    Vec<[u8; 32]>,
);

impl SceneCache {
    /// [`TriScene::from_scene_with_poses`], reusing whatever this cache already holds.
    pub fn tri_scene(
        &mut self,
        scene: &SceneDesc,
        world: &BTreeMap<StableId, Pose>,
    ) -> Result<TriScene, RenderError> {
        let statics = world_poses(scene);
        let (table, looks) = self.materials(scene)?;
        let mut out = TriScene {
            materials: table,
            ..TriScene::default()
        };
        for body in &scene.bodies {
            let body_pose = world
                .get(&body.id)
                .or_else(|| statics.get(&body.id))
                .copied()
                .unwrap_or(Pose::IDENTITY);
            for geom in &body.geoms {
                let wear = wear(geom, scene, &looks);
                if wear.rgba[3] == 0.0 {
                    // Alpha 0 is not drawn (collision-only geoms, an invisible floor). It
                    // keeps its segmentation id, which is the geom's traversal index
                    // (`renderer.md` section 2.1), and contributes no triangle.
                    out.push_geom(geom, &wear, Pose::IDENTITY, &[]);
                    continue;
                }
                let key = (geom.shape, content_hash(geom, scene));
                if !matches!(self.local.get(&geom.id), Some((k, _)) if *k == key) {
                    let tris = tessellate(geom, scene)?;
                    self.local.insert(geom.id, (key, tris));
                }
                let local = &self.local[&geom.id].1;
                out.push_geom(geom, &wear, body_pose.compose(geom.pose), local);
            }
        }
        Ok(out)
    }

    fn materials(
        &mut self,
        scene: &SceneDesc,
    ) -> Result<(Arc<Materials>, BTreeMap<StableId, Look>), RenderError> {
        if scene.materials.is_empty() {
            return Ok((Arc::default(), BTreeMap::new()));
        }
        let digests = scene
            .assets
            .iter()
            .filter(|a| scene.textures.contains_key(&a.id))
            .map(|a| a.hash)
            .collect();
        let key = (scene.materials.clone(), digests);
        if let Some((k, table, looks)) = &self.materials {
            if *k == key {
                return Ok((Arc::clone(table), looks.clone()));
            }
        }
        let (table, looks) = Materials::build(scene)?;
        let table = Arc::new(table);
        self.materials = Some((key, Arc::clone(&table), looks.clone()));
        Ok((table, looks))
    }
}

/// What `geom` wears: `MuJoCo`'s `setMaterial` — the material's `rgba` unless the geom writes
/// its own — for a drawn material, the geom's `rgba` otherwise.
fn wear(geom: &Geom, scene: &SceneDesc, looks: &BTreeMap<StableId, Look>) -> Wear {
    let drawn = geom
        .material
        .and_then(|id| scene.materials.get(&id).map(|m| (id, m)));
    let Some((id, m)) = drawn else {
        return Wear {
            rgba: geom.rgba,
            emission: 0.0,
            emissive: None,
            look: None,
            size: [0.0; 3],
            infinite: false,
            texgen: false,
        };
    };
    let rgba = if geom.rgba == DEFAULT_RGBA {
        m.rgba
    } else {
        geom.rgba
    };
    let f = |x: f64| x as f32;
    let (size, infinite, texgen) = match geom.shape {
        Shape::Box { half_extents: h } => ([f(h.x), f(h.y), f(h.z)], false, false),
        Shape::Sphere { radius: r } => ([f(r); 3], false, false),
        Shape::Ellipsoid { radii: r } => ([f(r.x), f(r.y), f(r.z)], false, false),
        Shape::Capsule {
            radius,
            half_length,
        }
        | Shape::Cylinder {
            radius,
            half_length,
        } => ([f(radius), f(radius), f(half_length)], false, false),
        Shape::Plane { half_x, half_y, .. } => (
            [f(half_x), f(half_y), 0.0],
            half_x <= 0.0 || half_y <= 0.0,
            false,
        ),
        Shape::Mesh { asset } => {
            let mesh = scene.meshes.get(&asset);
            let size = mesh.map_or([0.0; 3], |m| mesh_half_size(&m.positions));
            (size, false, mesh.is_none_or(|m| m.uvs.is_none()))
        }
        Shape::HeightField { .. } => ([0.0; 3], false, false),
    };
    Wear {
        rgba,
        emission: m.emission,
        emissive: m.emissive,
        look: looks.get(&id).copied(),
        size,
        infinite,
        texgen,
    }
}

/// Half the axis-aligned extent of a mesh: `MuJoCo`'s `geom_size` for a mesh geom.
fn mesh_half_size(positions: &[[f32; 3]]) -> [f32; 3] {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in positions {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    [0, 1, 2].map(|c| {
        if hi[c] >= lo[c] {
            (hi[c] - lo[c]) * 0.5
        } else {
            0.0
        }
    })
}

/// The `AssetRef::hash` of a `Mesh` geom's asset (its content, once `es_assets::mesh::load`
/// ran), zeros for every other shape.
fn content_hash(geom: &Geom, scene: &SceneDesc) -> [u8; 32] {
    match geom.shape {
        Shape::Mesh { asset } => scene
            .assets
            .iter()
            .find(|a| a.id == asset)
            .map_or([0; 32], |a| a.hash),
        _ => [0; 32],
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

/// Local-frame triangles for a primitive, or for a mesh `scene.meshes` carries.
fn tessellate(geom: &Geom, scene: &SceneDesc) -> Result<Vec<LocalTri>, RenderError> {
    let unsupported = |shape| {
        Err(RenderError::UnsupportedShape {
            geom: geom.name.clone(),
            shape,
        })
    };
    match geom.shape {
        Shape::Box { half_extents } => Ok(box_tris(half_extents)),
        Shape::Plane { half_x, half_y, .. } => Ok(plane_tris(half_x, half_y)),
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
        // The file's `f32` positions widen exactly; the pose is applied per frame in `f64` by
        // `push_geom`, exactly as for a primitive (packet M10/W2b). A mesh's own UVs are its
        // texture coordinates; without them the position feeds `MuJoCo`'s texgen (packet HT1).
        Shape::Mesh { asset } => {
            let Some(mesh) = scene.meshes.get(&asset) else {
                return unsupported("Mesh (asset not loaded: es_assets::mesh::load)");
            };
            let vertex = |i: u32| -> Option<Vert> {
                let p = *mesh.positions.get(i as usize)?;
                let uv = match &mesh.uvs {
                    Some(uvs) => *uvs.get(i as usize)?,
                    None => [p[0], p[1]],
                };
                let v = Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
                Some((v, uv, p))
            };
            mesh.indices
                .chunks_exact(3)
                .map(|f| {
                    let (a, b, c) = (vertex(f[0])?, vertex(f[1])?, vertex(f[2])?);
                    Some(LocalTri {
                        p: [a.0, b.0, c.0],
                        uv: [a.1, b.1, c.1],
                        unit: [a.2, b.2, c.2],
                    })
                })
                .collect::<Option<Vec<_>>>()
                .map_or_else(|| unsupported("Mesh (index out of range)"), Ok)
        }
        Shape::HeightField { .. } => unsupported("HeightField"),
    }
}

/// Two triangles in the local XY plane, normal +Z. An infinite side (`half == 0`) is drawn to
/// a fixed half-extent; `MuJoCo`'s texture coordinates follow `makePlane`: across the extent
/// for a finite side, `0.5 x` / `-0.5 y` for an infinite one.
fn plane_tris(half_x: f64, half_y: f64) -> Vec<LocalTri> {
    let hx = if half_x > 0.0 {
        half_x
    } else {
        INFINITE_PLANE_HALF
    };
    let hy = if half_y > 0.0 {
        half_y
    } else {
        INFINITE_PLANE_HALF
    };
    let vert = |sx: f64, sy: f64| {
        let (x, y) = (sx * hx, sy * hy);
        let u = if half_x > 0.0 {
            (sx + 1.0) * 0.5
        } else {
            0.5 * x
        };
        let v = if half_y > 0.0 {
            1.0 - (sy + 1.0) * 0.5
        } else {
            -0.5 * y
        };
        (
            Vec3::new(x, y, 0.0),
            [u as f32, v as f32],
            [x as f32, y as f32, 0.0],
        )
    };
    let tri = |a: Vert, b: Vert, c: Vert| LocalTri {
        p: [a.0, b.0, c.0],
        uv: [a.1, b.1, c.1],
        unit: [a.2, b.2, c.2],
    };
    let (a, b, c, d) = (vert(-1., -1.), vert(1., -1.), vert(1., 1.), vert(-1., 1.));
    vec![tri(a, b, c), tri(a, c, d)]
}

/// 12 triangles, all wound counter-clockwise seen from outside. `MuJoCo`'s box UVs: `x` / `y`
/// across a `z` face, `y` / `z` across an `x` face, `x` / `z` across a `y` face, `v` running
/// down the second axis.
fn box_tris(h: Vec3) -> Vec<LocalTri> {
    let c = |sx: f64, sy: f64, sz: f64| (Vec3::new(sx * h.x, sy * h.y, sz * h.z), [sx, sy, sz]);
    // Each face as (a, b, c, d) counter-clockwise from outside, with the unit axes its UV
    // reads.
    let faces: [([Corner; 4], [usize; 2]); 6] = [
        (
            [
                c(1., -1., -1.),
                c(1., 1., -1.),
                c(1., 1., 1.),
                c(1., -1., 1.),
            ],
            [1, 2],
        ), // +X
        (
            [
                c(-1., 1., -1.),
                c(-1., -1., -1.),
                c(-1., -1., 1.),
                c(-1., 1., 1.),
            ],
            [1, 2],
        ), // -X
        (
            [
                c(1., 1., -1.),
                c(-1., 1., -1.),
                c(-1., 1., 1.),
                c(1., 1., 1.),
            ],
            [0, 2],
        ), // +Y
        (
            [
                c(-1., -1., -1.),
                c(1., -1., -1.),
                c(1., -1., 1.),
                c(-1., -1., 1.),
            ],
            [0, 2],
        ), // -Y
        (
            [
                c(-1., -1., 1.),
                c(1., -1., 1.),
                c(1., 1., 1.),
                c(-1., 1., 1.),
            ],
            [0, 1],
        ), // +Z
        (
            [
                c(-1., 1., -1.),
                c(1., 1., -1.),
                c(1., -1., -1.),
                c(-1., -1., -1.),
            ],
            [0, 1],
        ), // -Z
    ];
    let mut out = Vec::new();
    for (f, [ua, va]) in faces {
        let uv = |k: usize| {
            let s = f[k].1;
            [((s[ua] + 1.0) * 0.5) as f32, ((1.0 - s[va]) * 0.5) as f32]
        };
        let unit = |k: usize| f[k].1.map(|x| x as f32);
        for [i, j, k] in [[0, 1, 2], [0, 2, 3]] {
            out.push(LocalTri {
                p: [f[i].0, f[j].0, f[k].0],
                uv: [uv(i), uv(j), uv(k)],
                unit: [unit(i), unit(j), unit(k)],
            });
        }
    }
    out
}

/// Longitude angle of segment `seg`, in `f32`. `seg == SPHERE_SEGMENTS` is the wrap-around
/// copy of segment 0 and is folded onto it, so the seam closes exactly instead of on a
/// `sin(2*pi)` residue.
fn phi_of(seg: u32) -> f32 {
    2.0 * PI * (seg % SPHERE_SEGMENTS) as f32 / SPHERE_SEGMENTS as f32
}

/// `u` of segment `seg`: `seg / SPHERE_SEGMENTS`, **not** folded, so the seam's two copies
/// sit at 0 and 1 as `MuJoCo`'s do.
fn u_of(seg: u32) -> f32 {
    seg as f32 / SPHERE_SEGMENTS as f32
}

/// UV sphere scaled per axis. `SPHERE_RINGS` latitude bands, `SPHERE_SEGMENTS` longitude.
///
/// The unit direction is computed entirely in `f32` through [`approx`] — never the host
/// `libm` (spec 3.2, 3.4) — then widened exactly and scaled by the `f64` radii. These
/// vertices are what `TriScene::to_floats` uploads to the GPU *and* what the CPU reference
/// traverses, so a host-dependent `sin` here would desynchronize the two paths.
///
/// UVs are `MuJoCo`'s `sphere()`: `u = az / 2 pi`, `v = 0.5 - el / pi` (0 at the +Z pole), and
/// a pole vertex takes its triangle's middle `u`.
#[allow(clippy::many_single_char_names)]
fn ellipsoid_tris(r: Vec3) -> Vec<LocalTri> {
    let unit = |ring: u32, seg: u32| {
        let theta = PI * ring as f32 / SPHERE_RINGS as f32;
        let phi = phi_of(seg);
        let (st, ct) = (approx::sin(theta), approx::cos(theta));
        let (sp, cp) = (approx::sin(phi), approx::cos(phi));
        [st * cp, st * sp, ct]
    };
    // `seg_uv` is the triangle's segment, for the middle `u` of a pole vertex.
    let point = |ring: u32, seg: u32, seg_uv: u32| {
        let d = unit(ring, seg);
        let p = Vec3::new(
            r.x * f64::from(d[0]),
            r.y * f64::from(d[1]),
            r.z * f64::from(d[2]),
        );
        let u = if ring == 0 || ring == SPHERE_RINGS {
            (seg_uv as f32 + 0.5) / SPHERE_SEGMENTS as f32
        } else {
            u_of(seg)
        };
        (p, [u, ring as f32 / SPHERE_RINGS as f32], d)
    };
    let mut out = Vec::new();
    for ring in 0..SPHERE_RINGS {
        for seg in 0..SPHERE_SEGMENTS {
            let (a, b) = (point(ring, seg, seg), point(ring, seg + 1, seg));
            let (c, d) = (point(ring + 1, seg + 1, seg), point(ring + 1, seg, seg));
            if ring > 0 {
                out.push(local(a, b, c));
            }
            if ring + 1 < SPHERE_RINGS {
                out.push(local(a, c, d));
            }
        }
    }
    out
}

fn local(a: Vert, b: Vert, c: Vert) -> LocalTri {
    LocalTri {
        p: [a.0, b.0, c.0],
        uv: [a.1, b.1, c.1],
        unit: [a.2, b.2, c.2],
    }
}

/// Cylinder side along local Z, optionally capped with hemispheres (capsule) instead of
/// flat discs (cylinder). UVs are `MuJoCo`'s `cylinder()` (`v = (1 - z / hl) / 2`),
/// `halfSphere()` (`v` from 1 at the equator to 0 at the top pole, 0 to 1 below) and `disk()`
/// (`0.5 + 0.5 (x, y) / r`); `unit` is the vertex in the unit part `MuJoCo` draws.
fn capsule_tris(radius: f64, half_length: f64, round_caps: bool) -> Vec<LocalTri> {
    let ring = |seg: u32, z: f64, r: f64| {
        let phi = phi_of(seg);
        Vec3::new(
            r * f64::from(approx::cos(phi)),
            r * f64::from(approx::sin(phi)),
            z,
        )
    };
    let side = |seg: u32, top: bool| {
        let z = if top { half_length } else { -half_length };
        let phi = phi_of(seg);
        let h = if top { 1.0f32 } else { -1.0 };
        (
            ring(seg, z, radius),
            [u_of(seg), (1.0 - h) * 0.5],
            [approx::cos(phi), approx::sin(phi), h],
        )
    };
    let mut out = Vec::new();
    for seg in 0..SPHERE_SEGMENTS {
        let (a, b) = (side(seg, false), side(seg + 1, false));
        let (c, d) = (side(seg + 1, true), side(seg, true));
        out.push(local(a, b, c));
        out.push(local(a, c, d));
    }
    for (sign, z0) in [(1.0_f64, half_length), (-1.0, -half_length)] {
        if round_caps {
            let s = sign as f32;
            for band in 0..CAP_RINGS {
                for seg in 0..SPHERE_SEGMENTS {
                    let cap = |band: u32, seg: u32| {
                        let t = 0.5 * PI * band as f32 / CAP_RINGS as f32;
                        let p = ring(
                            seg,
                            z0 + sign * radius * f64::from(approx::sin(t)),
                            radius * f64::from(approx::cos(t)),
                        );
                        let k = band as f32 / CAP_RINGS as f32;
                        let u = if band == CAP_RINGS {
                            (seg as f32 - 0.5) / SPHERE_SEGMENTS as f32
                        } else {
                            u_of(seg)
                        };
                        let v = if sign > 0.0 { 1.0 - k } else { k };
                        let phi = phi_of(seg);
                        let (st, ct) = (approx::sin(t), approx::cos(t));
                        (
                            p,
                            [u, v],
                            [ct * approx::cos(phi), ct * approx::sin(phi), s * (1.0 + st)],
                        )
                    };
                    let (a, b) = (cap(band, seg), cap(band, seg + 1));
                    let (c, d) = (cap(band + 1, seg + 1), cap(band + 1, seg));
                    out.push(local(a, b, c));
                    if band + 1 < CAP_RINGS {
                        out.push(local(a, c, d));
                    }
                }
            }
        } else {
            let s = sign as f32;
            let rim = |seg: u32| {
                let phi = phi_of(seg);
                let (c, sn) = (approx::cos(phi), approx::sin(phi));
                (
                    ring(seg, z0, radius),
                    [0.5 + 0.5 * c, 0.5 + 0.5 * sn],
                    [c, sn, s],
                )
            };
            for seg in 0..SPHERE_SEGMENTS {
                out.push(local(
                    (Vec3::new(0.0, 0.0, z0), [0.5, 0.5], [0.0, 0.0, s]),
                    rim(seg),
                    rim(seg + 1),
                ));
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
            ..SceneDesc::default()
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

    /// Plan H, H1: alpha 0 draws nothing but keeps its segmentation id, so the geoms after it
    /// keep theirs (`es_env::render` reads `seg - 1` as the geom's traversal index).
    #[test]
    fn alpha_zero_is_not_drawn_and_keeps_its_segmentation_id() {
        let sphere = Shape::Sphere { radius: 1.0 };
        let s = scene(vec![body(
            "b",
            Pose::IDENTITY,
            vec![
                geom("hidden", sphere, [0.4, 0.5, 0.6, 0.0]),
                geom("shown", sphere, [0.0, 1.0, 0.0, 1.0]),
                geom("faint", sphere, [0.0, 0.0, 1.0, 0.1]),
            ],
        )]);
        let tri = TriScene::from_scene(&s).unwrap();
        let ids: std::collections::BTreeSet<u32> = tri.tris.iter().map(|t| t.seg).collect();
        assert_eq!(ids, [2, 3].into_iter().collect());
        assert_eq!(tri.names[&1], "hidden");
        assert_eq!(tri.names[&2], "shown");
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
            assert!(f[0] * t.n[0] + f[1] * t.n[1] + f[2] * t.n[2] > 0.0, "{t:?}");
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
        assert_ne!(
            a, b,
            "the cache served the first scene's mesh to the second"
        );
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
