//! `SceneDesc` — the backend-neutral scene description (P30).
//!
//! Every importer (MJCF, URDF, glTF) produces a `SceneDesc`; every physics backend consumes
//! one. Nothing here computes physics: the mapping from these fields onto a solver is the
//! backend's job (spec 17.2), and unmapped fields are that backend's report, not a parse error.
//!
//! Conventions are the spec 3.1 ones, carried by `es-math`: right-handed Z-up, metres,
//! radians, kilograms, seconds, quaternions xyzw with `w >= 0`.
//!
//! # Stable ids (spec 5.3)
//!
//! Every element carries a [`StableId`] derived from its *name path* with [`scene_id`]:
//!
//! ```text
//! body       body/<body path>                 e.g. body/world/arm/link1
//! joint      joint/<owning body path>/<name>  e.g. joint/world/arm/link1/elbow
//! geom       geom/<owning body path>/<name>
//! site       site/<owning body path>/<name>
//! camera     camera/<owning body path>/<name>
//! actuator   actuator/<name>
//! sensor     sensor/<name>
//! tendon     tendon/<name>
//! asset      asset/<kind>/<name>
//! ```
//!
//! A body path is the `/`-joined chain of body names from the root body down to the body
//! itself — the MJCF importer names the root body `world`, so every element has an owner —
//! so an id depends on names only — not on element order, file layout, or insertion index.
//! Elements the source file leaves unnamed are given the local name `<kind><n>` with `n`
//! counted per owner in document order; such an id *is* order-dependent, which is the honest
//! consequence of the source not naming the element.
//!
//! # Hashing
//!
//! [`SceneDesc::scene_hash`] is blake3 over a canonical encoding: entities sorted by id, `f64`
//! as IEEE bits with `-0.0` normalised to `0.0` and `NaN` to one fixed pattern. Reordering
//! elements in the source file therefore cannot change the hash. Each [`AssetRef`] contributes
//! its own `asset_hash`, so `asset_hash -> scene_hash` of spec 5.3 holds by construction.

use std::collections::BTreeSet;

use es_core::StableId;
use es_math::{Inertia, Pose, Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Domain separator: changing the canonical encoding must change every stored hash.
const SCENE_TAG: &str = "es.scene.v1";
/// Domain separator for a single [`AssetRef`].
const ASSET_TAG: &str = "es.asset.v1";

/// Derives the id of `path` within `kind`, e.g. `scene_id("joint", "arm/link1/elbow")`.
///
/// The only way a scene element gets an id; see the module docs for the path scheme.
pub fn scene_id(kind: &str, path: &str) -> StableId {
    StableId::from_path(&format!("{kind}/{path}"))
}

/// A complete scene: bodies and their tree, joints, actuation, sensing and solver options.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SceneDesc {
    pub name: String,
    pub bodies: Vec<Body>,
    pub joints: Vec<Joint>,
    pub actuators: Vec<Actuator>,
    pub sensors: Vec<Sensor>,
    pub tendons: Vec<Tendon>,
    pub cameras: Vec<Camera>,
    pub assets: Vec<AssetRef>,
    pub options: PhysicsOptions,
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

/// Solver-level options. Backends map what they support and report the rest (spec 17.2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicsOptions {
    /// Physics step, s.
    pub timestep: f64,
    /// m/s^2, Z-up so the default is `-Z`.
    pub gravity: Vec3,
    pub integrator: Integrator,
    pub cone: FrictionCone,
    pub jacobian: Jacobian,
    pub solver: Solver,
    pub iterations: u32,
    /// Solver line-search iterations. Carried because a policy trained under
    /// `ls_iterations = 5` (`MuJoCo` Playground's locomotion setting) resolves contact
    /// differently from one stepped at `MuJoCo`'s default 50, and a backend that dropped it
    /// would change the physics silently (spec 17.2; packet M6/B1).
    #[serde(default = "default_ls_iterations")]
    pub ls_iterations: u32,
    /// `<option><flag eulerdamp>`: `MuJoCo`'s default integrates joint damping implicitly under
    /// the Euler integrator. MJX-trained models disable it, and the difference is visible in
    /// `qpos` within a few steps, so it is a carried option rather than an ignored flag.
    #[serde(default = "default_eulerdamp")]
    pub eulerdamp: bool,
    /// Frictional-to-normal constraint impedance ratio.
    pub impratio: f64,
}

fn default_ls_iterations() -> u32 {
    50
}

fn default_eulerdamp() -> bool {
    true
}

impl Default for PhysicsOptions {
    fn default() -> Self {
        Self {
            timestep: 0.002,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            integrator: Integrator::Euler,
            cone: FrictionCone::Pyramidal,
            jacobian: Jacobian::Auto,
            solver: Solver::Newton,
            iterations: 100,
            ls_iterations: default_ls_iterations(),
            eulerdamp: default_eulerdamp(),
            impratio: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Integrator {
    Euler,
    Rk4,
    Implicit,
    ImplicitFast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrictionCone {
    Pyramidal,
    Elliptic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Jacobian {
    Dense,
    Sparse,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Solver {
    Pgs,
    Cg,
    Newton,
}

/// Why a [`SceneDesc`] is not usable. Structural only — nothing here is a physics judgement.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SceneError {
    #[error("duplicate stable id {id} on `{name}`: two elements share a name path")]
    DuplicateId { id: StableId, name: String },
    #[error("body `{body}` has a parent that is not in the scene")]
    MissingParent { body: String },
    #[error("body `{body}` is its own ancestor")]
    BodyCycle { body: String },
    #[error("{kind} `{name}` references an element that is not in the scene")]
    DanglingRef { kind: &'static str, name: String },
}

impl SceneDesc {
    /// Structural validation: unique ids, resolvable references, acyclic body tree.
    pub fn validate(&self) -> Result<(), SceneError> {
        let mut ids = BTreeSet::new();
        let mut bodies = BTreeSet::new();
        let mut named = |id: StableId, name: &str| -> Result<(), SceneError> {
            if ids.insert(id) {
                Ok(())
            } else {
                Err(SceneError::DuplicateId {
                    id,
                    name: name.to_owned(),
                })
            }
        };
        for body in &self.bodies {
            named(body.id, &body.name)?;
            bodies.insert(body.id);
            for geom in &body.geoms {
                named(geom.id, &geom.name)?;
            }
            for site in &body.sites {
                named(site.id, &site.name)?;
            }
        }
        for joint in &self.joints {
            named(joint.id, &joint.name)?;
        }
        for camera in &self.cameras {
            named(camera.id, &camera.name)?;
        }
        for tendon in &self.tendons {
            named(tendon.id, &tendon.name)?;
        }
        for actuator in &self.actuators {
            named(actuator.id, &actuator.name)?;
        }
        for sensor in &self.sensors {
            named(sensor.id, &sensor.name)?;
        }
        for asset in &self.assets {
            named(asset.id, &asset.name)?;
        }

        for body in &self.bodies {
            if body.parent.is_some_and(|p| !bodies.contains(&p)) {
                return Err(SceneError::MissingParent {
                    body: body.name.clone(),
                });
            }
        }
        self.check_acyclic()?;

        for joint in &self.joints {
            if !bodies.contains(&joint.body) {
                return Err(SceneError::DanglingRef {
                    kind: "joint",
                    name: joint.name.clone(),
                });
            }
        }
        for tendon in &self.tendons {
            let ok = match &tendon.kind {
                TendonKind::Fixed { joints } => joints.iter().all(|(j, _)| ids.contains(j)),
                TendonKind::Spatial { sites } => sites.iter().all(|s| ids.contains(s)),
            };
            if !ok {
                return Err(SceneError::DanglingRef {
                    kind: "tendon",
                    name: tendon.name.clone(),
                });
            }
        }
        for actuator in &self.actuators {
            let (ActuatorTarget::Joint(target)
            | ActuatorTarget::Tendon(target)
            | ActuatorTarget::Site(target)) = actuator.target;
            if !ids.contains(&target) {
                return Err(SceneError::DanglingRef {
                    kind: "actuator",
                    name: actuator.name.clone(),
                });
            }
        }
        for sensor in &self.sensors {
            let target = match sensor.target {
                SensorTarget::Body(id)
                | SensorTarget::Joint(id)
                | SensorTarget::Geom(id)
                | SensorTarget::Site(id)
                | SensorTarget::Camera(id)
                | SensorTarget::Actuator(id) => id,
            };
            if !ids.contains(&target) {
                return Err(SceneError::DanglingRef {
                    kind: "sensor",
                    name: sensor.name.clone(),
                });
            }
        }
        Ok(())
    }

    // ponytail: parent-chain walk per body, O(n * depth). A scene has hundreds of bodies and a
    // depth of ten; swap in a colour DFS if that ever stops being true.
    fn check_acyclic(&self) -> Result<(), SceneError> {
        for body in &self.bodies {
            let mut cursor = body.parent;
            for _ in 0..self.bodies.len() {
                let Some(id) = cursor else {
                    break;
                };
                if id == body.id {
                    return Err(SceneError::BodyCycle {
                        body: body.name.clone(),
                    });
                }
                cursor = self
                    .bodies
                    .iter()
                    .find(|b| b.id == id)
                    .and_then(|b| b.parent);
            }
            if cursor.is_some() {
                return Err(SceneError::BodyCycle {
                    body: body.name.clone(),
                });
            }
        }
        Ok(())
    }

    /// Content identity of the scene (spec 5.3): blake3 of the canonical encoding.
    ///
    /// Independent of element order in the source file, sensitive to every field that can
    /// change simulation behaviour.
    pub fn scene_hash(&self) -> [u8; 32] {
        let mut c = Canon::default();
        c.str(SCENE_TAG);
        c.str(&self.name);
        self.encode_options(&mut c);

        let mut bodies: Vec<&Body> = self.bodies.iter().collect();
        bodies.sort_by_key(|b| b.id);
        c.seq(bodies.len());
        for body in bodies {
            encode_body(&mut c, body);
        }
        encode_sorted(&mut c, &self.joints, |j| j.id, encode_joint);
        encode_sorted(
            &mut c,
            &self.cameras,
            |x| x.id,
            |c, cam| {
                c.id(cam.id);
                c.str(&cam.name);
                c.opt_id(cam.body);
                c.pose(cam.pose);
                c.f64(cam.fovy);
            },
        );
        encode_sorted(&mut c, &self.tendons, |t| t.id, encode_tendon);
        encode_sorted(&mut c, &self.actuators, |a| a.id, encode_actuator);
        encode_sorted(&mut c, &self.sensors, |s| s.id, encode_sensor);
        encode_sorted(
            &mut c,
            &self.assets,
            |a| a.id,
            |c, asset| {
                c.id(asset.id);
                c.str(&asset.name);
                c.str(asset.kind.tag());
                c.str(&asset.path);
                c.raw(&asset.hash);
            },
        );
        c.finish()
    }

    fn encode_options(&self, c: &mut Canon) {
        let o = &self.options;
        c.f64(o.timestep);
        c.vec3(o.gravity);
        c.u32(o.integrator as u32);
        c.u32(o.cone as u32);
        c.u32(o.jacobian as u32);
        c.u32(o.solver as u32);
        c.u32(o.iterations);
        c.f64(o.impratio);
    }
}

fn encode_sorted<T>(
    c: &mut Canon,
    items: &[T],
    key: impl Fn(&T) -> StableId,
    encode: impl Fn(&mut Canon, &T),
) {
    let mut sorted: Vec<&T> = items.iter().collect();
    sorted.sort_by_key(|item| key(item));
    c.seq(sorted.len());
    for item in sorted {
        encode(c, item);
    }
}

fn encode_body(c: &mut Canon, body: &Body) {
    c.id(body.id);
    c.str(&body.name);
    c.opt_id(body.parent);
    c.pose(body.pose);
    match &body.inertial {
        None => c.u8(0),
        Some(i) => {
            c.u8(1);
            c.f64(i.mass);
            c.vec3(i.com);
            for row in i.inertia.matrix() {
                for v in row {
                    c.f64(v);
                }
            }
            c.quat(i.frame);
        }
    }
    encode_sorted(c, &body.geoms, |g| g.id, encode_geom);
    encode_sorted(
        c,
        &body.sites,
        |s| s.id,
        |c, site| {
            c.id(site.id);
            c.str(&site.name);
            c.pose(site.pose);
            c.vec3(site.size);
        },
    );
}

fn encode_geom(c: &mut Canon, g: &Geom) {
    c.id(g.id);
    c.str(&g.name);
    match g.shape {
        Shape::Plane {
            half_x,
            half_y,
            grid,
        } => {
            c.u8(0);
            c.f64(half_x);
            c.f64(half_y);
            c.f64(grid);
        }
        Shape::Sphere { radius } => {
            c.u8(1);
            c.f64(radius);
        }
        Shape::Capsule {
            radius,
            half_length,
        } => {
            c.u8(2);
            c.f64(radius);
            c.f64(half_length);
        }
        Shape::Cylinder {
            radius,
            half_length,
        } => {
            c.u8(3);
            c.f64(radius);
            c.f64(half_length);
        }
        Shape::Box { half_extents } => {
            c.u8(4);
            c.vec3(half_extents);
        }
        Shape::Ellipsoid { radii } => {
            c.u8(5);
            c.vec3(radii);
        }
        Shape::Mesh { asset } => {
            c.u8(6);
            c.id(asset);
        }
        Shape::HeightField { asset } => {
            c.u8(7);
            c.id(asset);
        }
    }
    c.pose(g.pose);
    for v in g.friction {
        c.f64(v);
    }
    c.u32(g.contype);
    c.u32(g.conaffinity);
    c.u32(g.condim);
    c.i32(g.priority);
    c.f64(g.density);
    match g.mass {
        None => c.u8(0),
        Some(m) => {
            c.u8(1);
            c.f64(m);
        }
    }
    c.f64(g.margin);
    c.f64(g.gap);
    for v in g.solref {
        c.f64(v);
    }
    for v in g.solimp {
        c.f64(v);
    }
    c.opt_id(g.material);
    for v in g.rgba {
        c.f64(v);
    }
    c.u8(u8::from(g.visual_only));
}

fn encode_joint(c: &mut Canon, j: &Joint) {
    c.id(j.id);
    c.str(&j.name);
    c.id(j.body);
    c.u8(j.kind as u8);
    c.vec3(j.axis);
    c.vec3(j.anchor);
    c.opt_range(j.range);
    c.f64(j.damping);
    c.f64(j.armature);
    c.f64(j.stiffness);
    c.f64(j.friction_loss);
    c.f64(j.spring_ref);
}

fn encode_tendon(c: &mut Canon, t: &Tendon) {
    c.id(t.id);
    c.str(&t.name);
    match &t.kind {
        TendonKind::Fixed { joints } => {
            c.u8(0);
            c.seq(joints.len());
            for (id, coef) in joints {
                c.id(*id);
                c.f64(*coef);
            }
        }
        TendonKind::Spatial { sites } => {
            c.u8(1);
            c.seq(sites.len());
            for id in sites {
                c.id(*id);
            }
        }
    }
    c.opt_range(t.range);
    c.f64(t.stiffness);
    c.f64(t.damping);
}

fn encode_actuator(c: &mut Canon, a: &Actuator) {
    c.id(a.id);
    c.str(&a.name);
    match a.kind {
        ActuatorKind::Motor => c.u8(0),
        ActuatorKind::Position { kp, kv } => {
            c.u8(1);
            c.f64(kp);
            c.f64(kv);
        }
        ActuatorKind::Velocity { kv } => {
            c.u8(2);
            c.f64(kv);
        }
        ActuatorKind::General { gain, bias } => {
            c.u8(3);
            for v in gain.into_iter().chain(bias) {
                c.f64(v);
            }
        }
    }
    match a.target {
        ActuatorTarget::Joint(id) => {
            c.u8(0);
            c.id(id);
        }
        ActuatorTarget::Tendon(id) => {
            c.u8(1);
            c.id(id);
        }
        ActuatorTarget::Site(id) => {
            c.u8(2);
            c.id(id);
        }
    }
    for v in a.gear {
        c.f64(v);
    }
    c.opt_range(a.ctrl_range);
    c.opt_range(a.force_range);
}

fn encode_sensor(c: &mut Canon, s: &Sensor) {
    c.id(s.id);
    c.str(&s.name);
    c.u8(s.kind as u8);
    match s.target {
        SensorTarget::Body(id) => {
            c.u8(0);
            c.id(id);
        }
        SensorTarget::Joint(id) => {
            c.u8(1);
            c.id(id);
        }
        SensorTarget::Geom(id) => {
            c.u8(2);
            c.id(id);
        }
        SensorTarget::Site(id) => {
            c.u8(3);
            c.id(id);
        }
        SensorTarget::Camera(id) => {
            c.u8(4);
            c.id(id);
        }
        SensorTarget::Actuator(id) => {
            c.u8(5);
            c.id(id);
        }
    }
    c.f64(s.noise);
    c.f64(s.cutoff);
}

/// Canonical byte encoder: little-endian, length-prefixed, `f64` as normalised IEEE bits.
///
/// Deliberately tiny and local — `es-ir` has its own (spec 5.3 hashes a different layer), and
/// `es-assets` (layer 2) cannot depend on `es-ir` (layer 6).
#[derive(Debug, Default)]
struct Canon {
    buf: Vec<u8>,
}

impl Canon {
    fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn seq(&mut self, len: usize) {
        self.u32(u32::try_from(len).unwrap_or(u32::MAX));
    }

    fn str(&mut self, s: &str) {
        self.seq(s.len());
        self.raw(s.as_bytes());
    }

    /// `-0.0` normalises to `0.0` (they compare equal, so they must hash equal); every `NaN`
    /// normalises to one pattern (no `NaN` is distinguishable from another by comparison).
    fn f64(&mut self, v: f64) {
        let bits = if v == 0.0 {
            0f64.to_bits()
        } else if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        };
        self.buf.extend_from_slice(&bits.to_le_bytes());
    }

    fn id(&mut self, id: StableId) {
        self.raw(id.as_bytes());
    }

    fn opt_id(&mut self, id: Option<StableId>) {
        match id {
            None => self.u8(0),
            Some(id) => {
                self.u8(1);
                self.id(id);
            }
        }
    }

    fn opt_range(&mut self, range: Option<(f64, f64)>) {
        match range {
            None => self.u8(0),
            Some((lo, hi)) => {
                self.u8(1);
                self.f64(lo);
                self.f64(hi);
            }
        }
    }

    fn vec3(&mut self, v: Vec3) {
        self.f64(v.x);
        self.f64(v.y);
        self.f64(v.z);
    }

    fn quat(&mut self, q: Quat) {
        self.f64(q.x);
        self.f64(q.y);
        self.f64(q.z);
        self.f64(q.w);
    }

    fn pose(&mut self, p: Pose) {
        self.vec3(p.position);
        self.quat(p.orientation);
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.buf).as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(name: &str, parent: Option<StableId>) -> Body {
        Body {
            id: scene_id("body", name),
            name: name.to_owned(),
            parent,
            pose: Pose::IDENTITY,
            inertial: None,
            geoms: Vec::new(),
            sites: Vec::new(),
        }
    }

    fn scene() -> SceneDesc {
        let a = body("a", None);
        let b = body("b", Some(a.id));
        SceneDesc {
            name: "t".to_owned(),
            joints: vec![Joint {
                id: scene_id("joint", "a/b/j"),
                name: "j".to_owned(),
                body: b.id,
                kind: JointKind::Hinge,
                axis: Vec3::new(0.0, 0.0, 1.0),
                anchor: Vec3::ZERO,
                range: Some((-1.0, 1.0)),
                damping: 0.1,
                armature: 0.0,
                stiffness: 0.0,
                friction_loss: 0.0,
                spring_ref: 0.0,
            }],
            bodies: vec![a, b],
            ..SceneDesc::default()
        }
    }

    #[test]
    fn hash_ignores_element_order() {
        let mut reordered = scene();
        reordered.bodies.reverse();
        assert_eq!(scene().scene_hash(), reordered.scene_hash());
    }

    #[test]
    fn hash_tracks_values() {
        let mut changed = scene();
        changed.joints[0].damping = 0.2;
        assert_ne!(scene().scene_hash(), changed.scene_hash());
    }

    #[test]
    fn negative_zero_hashes_as_zero() {
        let mut a = scene();
        let mut b = scene();
        a.options.timestep = 0.0;
        b.options.timestep = -0.0;
        assert_eq!(a.scene_hash(), b.scene_hash());
    }

    #[test]
    fn validate_accepts_a_tree_and_rejects_dangling_refs() {
        assert!(scene().validate().is_ok());

        let mut orphan = scene();
        orphan.bodies[1].parent = Some(scene_id("body", "ghost"));
        assert!(matches!(
            orphan.validate(),
            Err(SceneError::MissingParent { .. })
        ));

        let mut dangling = scene();
        dangling.joints[0].body = scene_id("body", "ghost");
        assert!(matches!(
            dangling.validate(),
            Err(SceneError::DanglingRef { kind: "joint", .. })
        ));
    }

    #[test]
    fn validate_rejects_duplicate_ids_and_cycles() {
        let mut dup = scene();
        dup.bodies.push(body("a", None));
        assert!(matches!(
            dup.validate(),
            Err(SceneError::DuplicateId { .. })
        ));

        let mut cycle = scene();
        let (a, b) = (cycle.bodies[0].id, cycle.bodies[1].id);
        cycle.bodies[0].parent = Some(b);
        cycle.bodies[1].parent = Some(a);
        assert!(matches!(
            cycle.validate(),
            Err(SceneError::BodyCycle { .. })
        ));
    }

    #[test]
    fn asset_hash_follows_the_path() {
        let a = AssetRef::from_path(AssetKind::Mesh, "arm", "meshes/arm.stl");
        assert_eq!(
            a.hash,
            AssetRef::from_path(AssetKind::Mesh, "arm", "meshes/arm.stl").hash
        );
        assert_ne!(
            a.hash,
            AssetRef::from_path(AssetKind::Mesh, "arm", "meshes/arm2.stl").hash
        );
        // The asset feeds the scene hash (spec 5.3).
        let mut with_asset = scene();
        with_asset.assets.push(a);
        assert_ne!(scene().scene_hash(), with_asset.scene_hash());
    }
}
