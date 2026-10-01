//! The elements a `SceneDesc` is made of: bodies with their geoms and sites, joints, cameras,
//! actuators, sensors, tendons, contact pairs, drawn materials and the asset references with
//! their `asset_hash`.

use es_core::StableId;
use es_math::{Inertia, Pose, Quat, Vec3};
use serde::{Deserialize, Serialize};

use super::scene_id;

/// Domain separator for a single [`AssetRef`].
const ASSET_TAG: &str = "es.asset.v1";

/// A drawn material (plan H, HT1): the file's values, unset ones `None`. The renderer's glTF
/// metallic-roughness parameters come from [`Material::metallic`] and
/// [`Material::roughness`], the one place the classic pair is converted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Base colour factor; a geom's own `rgba` wins when it is not `MuJoCo`'s default.
    pub rgba: [f64; 4],
    /// Emitted radiance as a multiple of the base colour (`MuJoCo`'s scalar `emission`).
    pub emission: f64,
    pub specular: Option<f64>,
    pub shininess: Option<f64>,
    pub metallic: Option<f64>,
    pub roughness: Option<f64>,
    pub texrepeat: [f64; 2],
    pub texuniform: bool,
    /// Texture of role `rgb` (the `texture` attribute or `<layer role="rgb">`).
    pub rgb: Option<StableId>,
    /// `<layer role="orm">`: occlusion, roughness, metallic in R, G, B (glTF's packing).
    pub orm: Option<StableId>,
    /// `<layer role="metallic">` and `role="roughness">`: read from the red channel.
    pub metallic_map: Option<StableId>,
    pub roughness_map: Option<StableId>,
    /// Tangent-space normal map (plan H, HT2): `<layer role="normal">`, glTF `normalTexture`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_map: Option<StableId>,
    /// glTF `normalTexture.scale`: the texel's X and Y are multiplied by it. `None` is 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_scale: Option<f64>,
    /// Emissive map: `<layer role="emissive">`, glTF `emissiveTexture`. The emitted radiance
    /// is [`Material::emissive`] times the texel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive_map: Option<StableId>,
    /// Emitted radiance as a colour, replacing `emission x rgba` when set: glTF's
    /// `emissiveFactor x emissive_strength`, or an MJCF emissive layer's `emission` (1 when the
    /// attribute is not written) on all three channels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive: Option<[f64; 3]>,
}

impl Default for Material {
    /// `mjs_defaultMaterial`, with nothing written explicitly.
    fn default() -> Self {
        Self {
            rgba: [1.0; 4],
            emission: 0.0,
            specular: None,
            shininess: None,
            metallic: None,
            roughness: None,
            texrepeat: [1.0, 1.0],
            texuniform: false,
            rgb: None,
            orm: None,
            metallic_map: None,
            roughness_map: None,
            normal_map: None,
            normal_scale: None,
            emissive_map: None,
            emissive: None,
        }
    }
}

impl Material {
    /// glTF metallic factor: the explicit value, else 1 under a metallic map (glTF's default
    /// factor), else 0 — a dielectric.
    #[must_use]
    pub fn metallic(&self) -> f64 {
        self.metallic
            .unwrap_or(if self.metallic_map.is_some() || self.orm.is_some() {
                1.0
            } else {
                0.0
            })
    }

    /// glTF roughness factor: the explicit value, else 1 under a roughness map, else the
    /// classic `shininess` converted by the one formula this crate writes down —
    ///
    /// ```text
    /// n = 128 * shininess                  (MuJoCo's Blinn-Phong exponent, OpenGL's 0..128)
    /// roughness = (2 / (n + 2))^(1/4)      (Walter et al. 2007: alpha^2 = 2 / (n + 2), alpha = r^2)
    /// ```
    ///
    /// at the explicit `shininess`, or at `MuJoCo`'s default 0.5 when only `specular` is
    /// written. `specular` itself maps to nothing: `F0` is glTF's `mix(0.04, base, metallic)`.
    #[must_use]
    pub fn roughness(&self) -> f64 {
        if let Some(r) = self.roughness {
            return r;
        }
        if self.roughness_map.is_some() || self.orm.is_some() {
            return 1.0;
        }
        let n = 128.0 * self.shininess.unwrap_or(0.5).clamp(0.0, 1.0);
        // The fourth root by two square roots: IEEE-exact, no host `powf` (spec 3.4).
        (2.0 / (n + 2.0)).sqrt().sqrt()
    }
}

/// One `<contact><pair>`. `None` leaves the parameter to `MuJoCo`, which then mixes it from
/// the two geoms as for any other contact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactPair {
    pub geom1: StableId,
    pub geom2: StableId,
    pub condim: Option<u32>,
    pub friction: Option<[f64; 5]>,
    pub solref: Option<[f64; 2]>,
    pub solimp: Option<[f64; 5]>,
    pub margin: Option<f64>,
    pub gap: Option<f64>,
}

/// A rigid body. `pose` is relative to `parent` (or to the world when `parent` is `None`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Body {
    pub id: StableId,
    pub name: String,
    pub parent: Option<StableId>,
    pub pose: Pose,
    /// Explicit mass properties. `None` means "derive from the geoms" — the importer records
    /// the intent, the backend (or a later packet) does the arithmetic.
    pub inertial: Option<BodyInertial>,
    pub geoms: Vec<Geom>,
    pub sites: Vec<Site>,
}

/// Mass properties in the body frame.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BodyInertial {
    pub mass: f64,
    /// Centre of mass in the body frame, m.
    pub com: Vec3,
    /// Inertia about the centre of mass, kg*m^2.
    pub inertia: Inertia,
    /// Frame the inertia tensor is expressed in, relative to the body frame.
    pub frame: Quat,
}

/// Degree of freedom connecting a body to its parent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Joint {
    pub id: StableId,
    pub name: String,
    /// The body this joint moves (the child of the pair).
    pub body: StableId,
    pub kind: JointKind,
    /// Rotation or slide axis in the child body frame; unit norm for hinge and slide.
    pub axis: Vec3,
    /// Anchor point in the child body frame, m.
    pub anchor: Vec3,
    /// Limits in rad (hinge, ball) or m (slide). `None` means unlimited.
    pub range: Option<(f64, f64)>,
    pub damping: f64,
    pub armature: f64,
    pub stiffness: f64,
    pub friction_loss: f64,
    /// Position the spring pulls towards, in the joint's own units.
    pub spring_ref: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointKind {
    /// Six degrees of freedom; the body floats. `axis` and `range` are meaningless.
    Free,
    /// Three rotational degrees of freedom about `anchor`.
    Ball,
    /// One rotational degree of freedom about `axis`.
    Hinge,
    /// One translational degree of freedom along `axis`.
    Slide,
    /// Welded to the parent: no degrees of freedom. MJCF spells this as a body with no joint
    /// and never emits it; a URDF fixed joint (P34) maps here.
    Fixed,
}

/// Collision / visual shape attached to a body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Geom {
    pub id: StableId,
    pub name: String,
    pub shape: Shape,
    /// Pose relative to the owning body.
    pub pose: Pose,
    /// Sliding, torsional and rolling friction.
    pub friction: [f64; 3],
    pub contype: u32,
    pub conaffinity: u32,
    /// Contact dimensionality: 1, 3, 4 or 6.
    pub condim: u32,
    /// Contact parameter precedence. When two geoms meet, the higher `priority` decides
    /// friction, `condim`, `solref` and `solimp` outright; equal priorities mix them
    /// (`MuJoCo` computation docs). Default 0.
    pub priority: i32,
    /// kg/m^3, used when `mass` is `None`.
    pub density: f64,
    pub mass: Option<f64>,
    pub margin: f64,
    pub gap: f64,
    /// Contact solver reference (time constant, damping ratio).
    pub solref: [f64; 2],
    /// Contact solver impedance.
    pub solimp: [f64; 5],
    /// Referenced material asset, if any.
    pub material: Option<StableId>,
    pub rgba: [f64; 4],
    /// Excluded from collision (visual only).
    pub visual_only: bool,
}

/// Geometric primitive. Sizes are half-extents / radii in m, following spec 3.1 units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    /// Half-extents in x and y (0 = infinite) plus the rendering grid spacing.
    Plane {
        half_x: f64,
        half_y: f64,
        grid: f64,
    },
    Sphere {
        radius: f64,
    },
    Capsule {
        radius: f64,
        half_length: f64,
    },
    Cylinder {
        radius: f64,
        half_length: f64,
    },
    Box {
        half_extents: Vec3,
    },
    Ellipsoid {
        radii: Vec3,
    },
    /// Reference to a mesh [`AssetRef`]; the file is loaded by P31 / P34, not here.
    Mesh {
        asset: StableId,
    },
    /// Reference to a height field [`AssetRef`].
    HeightField {
        asset: StableId,
    },
}

/// Named frame on a body: attachment point for sensors, tendons and actuators.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Site {
    pub id: StableId,
    pub name: String,
    pub pose: Pose,
    pub size: Vec3,
}

/// Camera frame. `body` is `None` for a camera fixed to the world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Camera {
    pub id: StableId,
    pub name: String,
    pub body: Option<StableId>,
    pub pose: Pose,
    /// Vertical field of view, rad.
    pub fovy: f64,
}

/// Actuator (spec 18.2). Transduction detail beyond gain / bias is the backend's mapping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Actuator {
    pub id: StableId,
    pub name: String,
    pub kind: ActuatorKind,
    pub target: ActuatorTarget,
    /// Length-to-transmission scaling; up to 6 components (the tail is 0 for scalar targets).
    pub gear: [f64; 6],
    pub ctrl_range: Option<(f64, f64)>,
    pub force_range: Option<(f64, f64)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActuatorKind {
    /// Control is force / torque directly.
    Motor,
    /// Position servo: `force = kp * (ctrl - length) - kv * velocity`.
    Position { kp: f64, kv: f64 },
    /// Velocity servo: `force = kv * (ctrl - velocity)`.
    Velocity { kv: f64 },
    /// Affine gain / bias form; `force = (gain . [1, length, velocity]) * ctrl + bias . ...`.
    General { gain: [f64; 3], bias: [f64; 3] },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActuatorTarget {
    Joint(StableId),
    Tendon(StableId),
    /// Cartesian actuation at a site.
    Site(StableId),
}

/// Sensor (spec 18.3). Noise is the standard deviation of the sensor's own units.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sensor {
    pub id: StableId,
    pub name: String,
    pub kind: SensorKind,
    pub target: SensorTarget,
    pub noise: f64,
    /// Absolute clamp on the output; 0 means no clamp.
    pub cutoff: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorKind {
    JointPos,
    JointVel,
    ActuatorFrc,
    FramePos,
    FrameQuat,
    Accelerometer,
    Gyro,
    Force,
    Torque,
    Touch,
    RangeFinder,
    /// Projection of a point into a camera.
    Camera,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorTarget {
    Body(StableId),
    Joint(StableId),
    Geom(StableId),
    Site(StableId),
    Camera(StableId),
    Actuator(StableId),
}

/// Tendon: a length coupling between joints (fixed) or through sites (spatial).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tendon {
    pub id: StableId,
    pub name: String,
    pub kind: TendonKind,
    pub range: Option<(f64, f64)>,
    pub stiffness: f64,
    pub damping: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TendonKind {
    /// Length is `sum(coef * joint position)`.
    Fixed { joints: Vec<(StableId, f64)> },
    /// Length is the polyline through these sites.
    Spatial { sites: Vec<StableId> },
}

/// External file a scene refers to. The file itself is *not* read here: `hash` is the digest
/// of the path string, and P31 / P35 replace it with the content digest once files are loaded.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssetRef {
    pub id: StableId,
    pub name: String,
    pub kind: AssetKind,
    pub path: String,
    pub hash: [u8; 32],
}

impl AssetRef {
    /// Builds a reference whose `hash` is `blake3(path)` — a placeholder for the content hash.
    pub fn from_path(kind: AssetKind, name: &str, path: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSET_TAG.as_bytes());
        hasher.update(kind.tag().as_bytes());
        hasher.update(path.as_bytes());
        Self {
            id: scene_id("asset", &format!("{}/{name}", kind.tag())),
            name: name.to_owned(),
            kind,
            path: path.to_owned(),
            hash: *hasher.finalize().as_bytes(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetKind {
    Mesh,
    Texture,
    Material,
    HeightField,
}

impl AssetKind {
    /// Stable string form; part of both the id path and the asset hash.
    pub fn tag(self) -> &'static str {
        match self {
            AssetKind::Mesh => "mesh",
            AssetKind::Texture => "texture",
            AssetKind::Material => "material",
            AssetKind::HeightField => "hfield",
        }
    }
}
