//! The scene document `*.esscene` (plan G, packet G1; spec 14.3): the person's scene, a TOML
//! authoring format read into the same [`SceneDesc`] every other reader produces, so
//! `scene_hash` and the asset hashes are the `SceneDesc`'s whatever the file format (spec 5.3).
//! Not an IR. Design: `docs/design/scene-authoring.md` section 3.
//!
//! [`EsScene`] is the document as written — every optional field an `Option`, so writing back
//! what was read changes nothing (the editor round-trips the document, not its expansion) — and
//! [`expand`] turns it into a [`SceneDesc`]: the `[[include]]`s through the existing readers
//! first, then the document's own entities in document order. Orientations are quaternions,
//! `[x, y, z, w]` (spec 3.1); angles a person reads (a camera's `fovy`) are degrees, every other
//! quantity is `SceneDesc`'s SI unit. Unknown keys are refused by name.

mod expand;
mod include;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use expand::expand;

use crate::mesh::MeshError;
use crate::scene::SceneError;

/// Why a document did not read or expand. A refusal of the document itself names the field.
#[derive(Debug, Error)]
pub enum EsSceneError {
    /// Not TOML, or not this schema: an unknown key, a missing one, a wrong type.
    #[error("{0}")]
    Toml(#[from] toml::de::Error),
    #[error("{0}")]
    Write(#[from] toml::ser::Error),
    #[error("`{field}`: {reason}")]
    Field { field: String, reason: String },
    /// The expansion is not a valid scene (two elements sharing a name path, for one).
    #[error("{0}")]
    Scene(#[from] SceneError),
    #[error("{0}")]
    Mesh(#[from] MeshError),
}

pub(crate) fn refuse<T>(
    field: impl Into<String>,
    reason: impl Into<String>,
) -> Result<T, EsSceneError> {
    Err(EsSceneError::Field {
        field: field.into(),
        reason: reason.into(),
    })
}

/// A scene document. Arrays of tables keep document order, which is the expansion's order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EsScene {
    /// Always `"scene"`.
    pub kind: String,
    /// Always 1.
    pub schema: u32,
    /// `SceneDesc::name`, hashed; absent is `"scene"`.
    pub name: Option<String>,
    pub physics: Option<Physics>,
    #[serde(default, rename = "include", skip_serializing_if = "Vec::is_empty")]
    pub includes: Vec<Include>,
    /// Static scenery: geoms on the world body.
    #[serde(default, rename = "geom", skip_serializing_if = "Vec::is_empty")]
    pub geoms: Vec<GeomDoc>,
    #[serde(default, rename = "body", skip_serializing_if = "Vec::is_empty")]
    pub bodies: Vec<BodyDoc>,
    #[serde(default, rename = "texture", skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<TextureDoc>,
    #[serde(default, rename = "material", skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<MaterialDoc>,
    #[serde(default, rename = "camera", skip_serializing_if = "Vec::is_empty")]
    pub cameras: Vec<CameraDoc>,
    #[serde(default, rename = "light", skip_serializing_if = "Vec::is_empty")]
    pub lights: Vec<LightDoc>,
    #[serde(default, rename = "region", skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<RegionDoc>,
}

/// `SceneDesc::options`; a field left out is `MuJoCo`'s default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Physics {
    pub timestep: Option<f64>,
    pub gravity: Option<[f64; 3]>,
    pub integrator: Option<IntegratorDoc>,
    pub cone: Option<ConeDoc>,
    pub jacobian: Option<JacobianDoc>,
    pub solver: Option<SolverDoc>,
    pub iterations: Option<u32>,
    pub ls_iterations: Option<u32>,
    pub eulerdamp: Option<bool>,
    pub impratio: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntegratorDoc {
    Euler,
    Rk4,
    Implicit,
    ImplicitFast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConeDoc {
    Pyramidal,
    Elliptic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JacobianDoc {
    Dense,
    Sparse,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SolverDoc {
    Pgs,
    Cg,
    Newton,
}

/// A robot or any multi-body asset by reference: MJCF (`.xml`), URDF, glTF (`.gltf`/`.glb`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Include {
    /// The include's handle in this document (a task names its robot by it).
    pub name: String,
    /// Relative to the document's directory.
    pub source: String,
    /// Prepended to every name the file declares; absent keeps the file's own names and ids.
    pub prefix: Option<String>,
    /// Placement of the file's root bodies and world-level elements.
    pub pos: Option<[f64; 3]>,
    pub quat: Option<[f64; 4]>,
    /// Per-instance overrides, addressed by the file's own names.
    pub set: Option<IncludeSet>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IncludeSet {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub joint: BTreeMap<String, JointSet>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub actuator: BTreeMap<String, ActuatorSet>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub geom: BTreeMap<String, GeomSet>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSet {
    pub range: Option<[f64; 2]>,
    pub damping: Option<f64>,
    pub armature: Option<f64>,
    pub stiffness: Option<f64>,
    pub frictionloss: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActuatorSet {
    /// Position gain (`<position>` only).
    pub kp: Option<f64>,
    /// Velocity gain (`<position>` and `<velocity>`).
    pub kv: Option<f64>,
    pub ctrlrange: Option<[f64; 2]>,
    pub forcerange: Option<[f64; 2]>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeomSet {
    pub rgba: Option<[f64; 4]>,
    /// A material of the included file or of this document.
    pub material: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyDoc {
    pub name: String,
    /// A body defined before this one (an included one too); absent is the world.
    pub parent: Option<String>,
    pub pos: Option<[f64; 3]>,
    pub quat: Option<[f64; 4]>,
    pub gravcomp: Option<f64>,
    /// Absent is welded to the parent.
    pub joint: Option<JointDoc>,
    /// Absent derives the mass properties from the geoms.
    pub inertial: Option<InertialDoc>,
    #[serde(default, rename = "geom", skip_serializing_if = "Vec::is_empty")]
    pub geoms: Vec<GeomDoc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JointKindDoc {
    /// No joint: the body is welded to its parent.
    Fixed,
    Free,
    Ball,
    Hinge,
    Slide,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointDoc {
    pub kind: JointKindDoc,
    /// Absent is the body's name.
    pub name: Option<String>,
    pub axis: Option<[f64; 3]>,
    pub pos: Option<[f64; 3]>,
    /// rad (hinge, ball) or m (slide); absent is unlimited.
    pub range: Option<[f64; 2]>,
    pub damping: Option<f64>,
    pub armature: Option<f64>,
    pub stiffness: Option<f64>,
    pub frictionloss: Option<f64>,
    pub springref: Option<f64>,
}

/// Exactly one of `diaginertia` / `fullinertia` (`ixx iyy izz ixy ixz iyz`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InertialDoc {
    pub mass: f64,
    pub pos: Option<[f64; 3]>,
    pub quat: Option<[f64; 4]>,
    pub diaginertia: Option<[f64; 3]>,
    pub fullinertia: Option<[f64; 6]>,
}

/// Sizes are `SceneDesc::Shape`'s: half-extents, radii, half-lengths, m. Written as a table of
/// one shape key, `{ box = [0.03, 0.03, 0.03] }`, and beside `mesh` an optional `scale`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ShapeTable", into = "ShapeTable")]
pub enum ShapeDoc {
    /// Half x, half y (0 = infinite), grid spacing.
    Plane([f64; 3]),
    Sphere(f64),
    /// Radius, half length.
    Capsule([f64; 2]),
    Cylinder([f64; 2]),
    Box([f64; 3]),
    Ellipsoid([f64; 3]),
    /// An `.stl` / `.obj` file relative to the document, its vertices times `scale` per axis
    /// (absent is 1; packet M17/R3). The asset is named by the file's stem.
    Mesh {
        file: String,
        scale: Option<[f64; 3]>,
    },
}

/// [`ShapeDoc`] as written: exactly one shape key, and `scale` only beside `mesh`.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShapeTable {
    #[serde(skip_serializing_if = "Option::is_none")]
    plane: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sphere: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capsule: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cylinder: Option<[f64; 2]>,
    #[serde(rename = "box", skip_serializing_if = "Option::is_none")]
    cuboid: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ellipsoid: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scale: Option<[f64; 3]>,
}

impl TryFrom<ShapeTable> for ShapeDoc {
    type Error = String;

    fn try_from(t: ShapeTable) -> Result<Self, String> {
        let scale = t.scale;
        let mut given = [
            t.plane.map(Self::Plane),
            t.sphere.map(Self::Sphere),
            t.capsule.map(Self::Capsule),
            t.cylinder.map(Self::Cylinder),
            t.cuboid.map(Self::Box),
            t.ellipsoid.map(Self::Ellipsoid),
            t.mesh.map(|file| Self::Mesh { file, scale }),
        ]
        .into_iter()
        .flatten();
        match (given.next(), given.next()) {
            (Some(s @ Self::Mesh { .. }), None) => Ok(s),
            (Some(s), None) if scale.is_none() => Ok(s),
            (Some(_), None) => Err("`scale` is a mesh's".to_owned()),
            _ => Err(
                "a shape is one of plane, sphere, capsule, cylinder, box, ellipsoid, mesh"
                    .to_owned(),
            ),
        }
    }
}

impl From<ShapeDoc> for ShapeTable {
    fn from(s: ShapeDoc) -> Self {
        let mut t = Self::default();
        match s {
            ShapeDoc::Plane(v) => t.plane = Some(v),
            ShapeDoc::Sphere(r) => t.sphere = Some(r),
            ShapeDoc::Capsule(v) => t.capsule = Some(v),
            ShapeDoc::Cylinder(v) => t.cylinder = Some(v),
            ShapeDoc::Box(v) => t.cuboid = Some(v),
            ShapeDoc::Ellipsoid(v) => t.ellipsoid = Some(v),
            ShapeDoc::Mesh { file, scale } => (t.mesh, t.scale) = (Some(file), scale),
        }
        t
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeomDoc {
    /// Absent is `geom<n>`, counted per body as MJCF counts unnamed geoms.
    pub name: Option<String>,
    pub shape: ShapeDoc,
    pub pos: Option<[f64; 3]>,
    pub quat: Option<[f64; 4]>,
    /// Absent derives the mass from `density`.
    pub mass: Option<f64>,
    pub density: Option<f64>,
    pub friction: Option<[f64; 3]>,
    pub condim: Option<u32>,
    pub contype: Option<u32>,
    pub conaffinity: Option<u32>,
    pub priority: Option<i32>,
    pub margin: Option<f64>,
    pub gap: Option<f64>,
    pub solref: Option<[f64; 2]>,
    pub solimp: Option<[f64; 5]>,
    pub rgba: Option<[f64; 4]>,
    pub material: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TexKindDoc {
    #[serde(rename = "2d")]
    TwoD,
    Cube,
    Skybox,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColorSpaceDoc {
    Auto,
    Srgb,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BuiltinDoc {
    None,
    Gradient,
    Checker,
    Flat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MarkDoc {
    None,
    Edge,
    Cross,
    Random,
}

/// MJCF `<texture>`'s fields (plan H, HT1); a field left out is `MuJoCo`'s default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureDoc {
    pub name: String,
    pub kind: Option<TexKindDoc>,
    pub colorspace: Option<ColorSpaceDoc>,
    /// A PNG relative to the document.
    pub file: Option<String>,
    pub gridsize: Option<[u32; 2]>,
    pub gridlayout: Option<String>,
    /// `right left up down front back`.
    pub cubefiles: Option<[String; 6]>,
    pub builtin: Option<BuiltinDoc>,
    pub rgb1: Option<[f64; 3]>,
    pub rgb2: Option<[f64; 3]>,
    pub mark: Option<MarkDoc>,
    pub markrgb: Option<[f64; 3]>,
    pub random: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// MJCF `<material>`'s fields (plan H, HT1/HT2); texture slots name a `[[texture]]`. As in
/// MJCF, a material that writes only `rgba` is a name and nothing more: it is drawn as the
/// geom's own colour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialDoc {
    pub name: String,
    pub rgba: Option<[f64; 4]>,
    pub emission: Option<f64>,
    pub specular: Option<f64>,
    pub shininess: Option<f64>,
    pub metallic: Option<f64>,
    pub roughness: Option<f64>,
    pub texrepeat: Option<[f64; 2]>,
    pub texuniform: Option<bool>,
    /// The base colour texture (`texture=` in MJCF).
    pub texture: Option<String>,
    pub orm: Option<String>,
    pub metallic_map: Option<String>,
    pub roughness_map: Option<String>,
    pub normal_map: Option<String>,
    pub normal_scale: Option<f64>,
    /// Emitted colour map, scaled by `emission` (1 when absent).
    pub emissive_map: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraDoc {
    pub name: String,
    /// Absent is fixed to the world.
    pub parent: Option<String>,
    pub pos: Option<[f64; 3]>,
    /// Identity looks along -Z with image-up +Y (MJCF's camera frame).
    pub quat: Option<[f64; 4]>,
    /// Vertical field of view, degrees; absent is 45.
    pub fovy: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LightKind {
    /// A thin emissive box on the world, `<name>_light` (the renderer's emitter convention).
    Area,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightDoc {
    pub name: String,
    pub kind: Option<LightKind>,
    pub pos: Option<[f64; 3]>,
    pub quat: Option<[f64; 4]>,
    /// Half-extents in x and y, m.
    pub size: [f64; 2],
    /// Emitted colour; absent is white.
    pub rgb: Option<[f64; 3]>,
    /// Scales `rgb`; absent is 1.
    pub intensity: Option<f64>,
}

/// A named zone (an MJCF site): no collision, no mass; sentences name it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionDoc {
    pub name: String,
    /// Absent is the world.
    pub parent: Option<String>,
    pub pos: Option<[f64; 3]>,
    pub quat: Option<[f64; 4]>,
    /// Half-extents, m; absent is MJCF's site default 0.005.
    pub size: Option<[f64; 3]>,
}

impl EsScene {
    /// Reads a document; refuses an unknown key, a wrong `kind` or `schema`.
    pub fn from_toml(text: &str) -> Result<Self, EsSceneError> {
        let doc: Self = toml::from_str(text)?;
        if doc.kind != "scene" {
            return refuse("kind", format!("expected `scene`, found `{}`", doc.kind));
        }
        if doc.schema != 1 {
            return refuse("schema", format!("expected 1, found {}", doc.schema));
        }
        Ok(doc)
    }

    /// Writes the document back: what was read, field for field; floats in round-trip form.
    pub fn to_toml(&self) -> Result<String, EsSceneError> {
        Ok(toml::to_string(self)?)
    }
}
