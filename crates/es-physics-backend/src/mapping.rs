//! Backend semantic mapping (spec 17.2) and the cross-backend comparison it feeds.
//!
//! The same Task IR must not behave differently from one backend to the next. Since v1.0 does
//! not build a solver (spec 4.3), *the mapping and its verification are the work*: every
//! feature a scene uses is looked up in a table that says, for each backend, whether the
//! mapping is native, an approximation (with the approximation named), or missing.
//!
//! Severity is what makes the table operational: an unmapped row with `severity: error` blocks
//! execution (spec 14.4). A row that is merely unverified is a `Warning` carrying
//! `TODO(api-notes)` — a mapping is never *guessed* native.
//!
//! [`compare_backends`] runs one scene on two backends and reports spec 3.5 tier 3 metrics
//! next to both mapping reports; it is the body of `es backend compare` (spec 17.2), whose CLI
//! wiring lives in another packet.

use std::collections::BTreeSet;
use std::fmt;

use es_assets::scene::{ActuatorKind, SceneDesc, SensorKind};
use es_physics_core::{
    DeterminismTier, Feature, LoadConfig, PhysicsBackend, PhysicsError, Requirements,
};

/// A backend named in the spec 17.2 table. The spelling is the `--backends` one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BackendKind {
    MuJoCoCpu,
    MjWarp,
    Newton,
    PhysX,
}

impl BackendKind {
    pub const ALL: [Self; 4] = [Self::MuJoCoCpu, Self::MjWarp, Self::Newton, Self::PhysX];

    pub const fn name(self) -> &'static str {
        match self {
            Self::MuJoCoCpu => "mujoco-cpu",
            Self::MjWarp => "mjwarp",
            Self::Newton => "newton",
            Self::PhysX => "physx",
        }
    }

    /// The backend a [`Capabilities::name`](es_physics_core::Capabilities::name) stands for, or
    /// `None` for a name outside the spec 17.2 table (a test double, say).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A row of the spec 17.2 table, spelled as the spec spells it.
///
/// These are Task IR concerns that do not line up one-to-one with a
/// [`Feature`]: `actuator.pd` is a pair of gains rather than an actuator kind, and
/// `sensor.contact_force` is one question asked of two sensor kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Spec17Row {
    ActuatorPd,
    ContactFrictionCone,
    ContactSoftParams,
    JointArmature,
    SensorContactForce,
}

impl Spec17Row {
    pub const ALL: [Self; 5] = [
        Self::ActuatorPd,
        Self::ContactFrictionCone,
        Self::ContactSoftParams,
        Self::JointArmature,
        Self::SensorContactForce,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::ActuatorPd => "actuator.pd",
            Self::ContactFrictionCone => "contact.friction_cone",
            Self::ContactSoftParams => "contact.soft_params",
            Self::JointArmature => "joint.armature",
            Self::SensorContactForce => "sensor.contact_force",
        }
    }
}

/// Every [`Feature`] `es-physics-core` knows. The table covers all of them, so a scene can
/// never use something the report is silent about.
const ALL_FEATURES: [Feature; 35] = [
    Feature::JointFree,
    Feature::JointBall,
    Feature::JointHinge,
    Feature::JointSlide,
    Feature::JointFixed,
    Feature::JointLimit,
    Feature::JointArmature,
    Feature::JointSpring,
    Feature::JointFrictionLoss,
    Feature::ActuatorMotor,
    Feature::ActuatorPosition,
    Feature::ActuatorVelocity,
    Feature::ActuatorGeneral,
    Feature::ActuatorOnJoint,
    Feature::ActuatorOnTendon,
    Feature::ActuatorOnSite,
    Feature::SensorJointPos,
    Feature::SensorJointVel,
    Feature::SensorActuatorFrc,
    Feature::SensorFramePos,
    Feature::SensorFrameQuat,
    Feature::SensorAccelerometer,
    Feature::SensorGyro,
    Feature::SensorForce,
    Feature::SensorTorque,
    Feature::SensorTouch,
    Feature::SensorRangeFinder,
    Feature::SensorCamera,
    Feature::ContactPyramidal,
    Feature::ContactElliptic,
    Feature::ContactSoftParams,
    Feature::ContactCondim6,
    Feature::ContactMesh,
    Feature::ContactHeightField,
    Feature::Tendon,
];

/// One thing a scene can ask a backend for: either a spec 17.2 table row or a capability
/// [`Feature`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskFeature {
    Spec17(Spec17Row),
    Capability(Feature),
}

impl TaskFeature {
    /// Every feature the table has a row for: the five spec 17.2 rows plus every [`Feature`].
    pub fn all() -> Vec<Self> {
        Spec17Row::ALL
            .into_iter()
            .map(Self::Spec17)
            .chain(ALL_FEATURES.into_iter().map(Self::Capability))
            .collect()
    }
}

impl fmt::Display for TaskFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spec17(row) => f.write_str(row.name()),
            Self::Capability(feature) => write!(f, "{feature}"),
        }
    }
}

/// How a backend realises a feature. Every variant carries the note that goes in the report:
/// an unexplained `Unsupported` is what spec 17.2 exists to prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The backend has the same concept, with the same meaning.
    Native(&'static str),
    /// The backend has something close; the note says what the difference is.
    Approximated(&'static str),
    /// No mapping. The note says whether that is known or merely unverified.
    Unsupported(&'static str),
}

impl Status {
    pub const fn note(self) -> &'static str {
        match self {
            Self::Native(note) | Self::Approximated(note) | Self::Unsupported(note) => note,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Native(_) => "native",
            Self::Approximated(_) => "approximated",
            Self::Unsupported(_) => "unsupported",
        }
    }
}

/// What an imperfect mapping costs. `Error` on an unmapped row blocks execution (spec 14.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl Severity {
    const fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// One cell of the spec 17.2 table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping {
    pub status: Status,
    pub severity: Severity,
}

impl Mapping {
    const fn native(note: &'static str) -> Self {
        Self {
            status: Status::Native(note),
            severity: Severity::Info,
        }
    }

    /// A mapping that is close but not the same thing: the run keeps going, loudly.
    const fn approximated(note: &'static str) -> Self {
        Self {
            status: Status::Approximated(note),
            severity: Severity::Warning,
        }
    }

    /// A mapping that is known to be missing: the scene cannot run here (spec 14.4).
    const fn blocked(note: &'static str) -> Self {
        Self {
            status: Status::Unsupported(note),
            severity: Severity::Error,
        }
    }

    /// Not checked against the engine yet. Never a guessed `Native` (spec 1.7: a plausible
    /// API name is the cheapest thing an agent can invent).
    const fn unverified() -> Self {
        Self {
            status: Status::Unsupported("TODO(api-notes): unverified against the engine"),
            severity: Severity::Warning,
        }
    }

    /// Whether this row blocks execution (spec 14.4).
    pub const fn blocks(self) -> bool {
        matches!(self.status, Status::Unsupported(_)) && matches!(self.severity, Severity::Error)
    }
}

/// The spec 17.2 table: `(TaskFeature, BackendKind) -> Mapping`, total by construction.
///
/// Held as a function rather than a map because it is a constant: [`SemanticMapping`]
/// materialises it when something needs to iterate.
pub fn lookup(feature: TaskFeature, backend: BackendKind) -> Mapping {
    match backend {
        BackendKind::MuJoCoCpu => mujoco_cpu(feature),
        BackendKind::MjWarp => mjwarp(feature),
        BackendKind::Newton => newton(feature),
        BackendKind::PhysX => physx(feature),
    }
}

/// `mujoco-cpu` is `MuJoCo`, so its column is whatever the adapter declares — derived from
/// [`crate::mujoco::capabilities`] rather than restated, so the two cannot drift apart.
fn mujoco_cpu(feature: TaskFeature) -> Mapping {
    match feature {
        TaskFeature::Spec17(Spec17Row::ActuatorPd) => Mapping::native("position actuator kp / kv"),
        TaskFeature::Spec17(Spec17Row::ContactFrictionCone) => {
            Mapping::native("pyramidal or elliptic, as the scene asks")
        }
        TaskFeature::Spec17(Spec17Row::ContactSoftParams) => {
            Mapping::native("solref / solimp impedance, the reference semantics")
        }
        TaskFeature::Spec17(Spec17Row::JointArmature) => Mapping::native("armature"),
        TaskFeature::Spec17(Spec17Row::SensorContactForce) => {
            Mapping::blocked("MuJoCo has force / touch sensors but the MJCF emitter writes none")
        }
        TaskFeature::Capability(capability) => {
            if crate::mujoco::capabilities().has(capability) {
                Mapping::native("MuJoCo's own semantics")
            } else {
                Mapping::blocked("not in the declared capability set of `mujoco-cpu`")
            }
        }
    }
}

/// `MJWarp` is `MuJoCo`'s semantics on the GPU and is fed by the same MJCF emitter, so its
/// column follows `mujoco-cpu` except where spec 17.2 pins something narrower.
fn mjwarp(feature: TaskFeature) -> Mapping {
    const EMITTER: &str = "the shared MJCF emitter cannot write this";
    match feature {
        TaskFeature::Spec17(Spec17Row::ActuatorPd) => {
            Mapping::native("position gain (spec 17.2), kp / kv on a position actuator")
        }
        TaskFeature::Spec17(Spec17Row::ContactFrictionCone)
        | TaskFeature::Capability(Feature::ContactPyramidal) => Mapping::native("pyramidal"),
        TaskFeature::Capability(Feature::ContactElliptic) => Mapping::blocked(
            "spec 17.2 maps MJWarp's friction cone to pyramidal; running an elliptic scene here \
             would change contact dynamics silently",
        ),
        TaskFeature::Spec17(Spec17Row::ContactSoftParams)
        | TaskFeature::Capability(Feature::ContactSoftParams) => {
            Mapping::native("impedance (spec 17.2), solref / solimp")
        }
        TaskFeature::Spec17(Spec17Row::JointArmature)
        | TaskFeature::Capability(Feature::JointArmature) => Mapping::native("armature"),
        TaskFeature::Spec17(Spec17Row::SensorContactForce) => Mapping::blocked(EMITTER),
        TaskFeature::Capability(Feature::ContactCondim6) => Mapping::unverified(),
        TaskFeature::Capability(capability) => {
            // Everything else the emitter can write has MuJoCo meaning on this backend too.
            if crate::mujoco::capabilities().has(capability) {
                Mapping::native("MuJoCo semantics, batched on the GPU")
            } else {
                Mapping::blocked(EMITTER)
            }
        }
    }
}

/// Newton has no adapter yet, so only the spec 17.2 rows are filled: everything else is
/// `TODO(api-notes)`, never a guess.
fn newton(feature: TaskFeature) -> Mapping {
    // The Spec17 rows are the spec 17.2 table's statement about the *engine*; the Capability
    // rows are what `NewtonBackend` was verified to deliver through `add_mjcf` in newton 1.6.0.
    // Where the two differ, the Capability row is the one that gates execution (spec 14.4).
    const CONTROLLER: &str = "a joint controller (spec 17.2), not a MuJoCo position gain";
    const SOLVER: &str = "solver-dependent (spec 17.2), not MuJoCo solref / solimp";
    const CONTACT: &str = "read from the contact buffer (spec 17.2), not a sensor";
    const NO_ACTUATOR: &str =
        "newton 1.6 add_mjcf imports no <actuator> (Model.actuators is empty): unactuated";
    const NO_SENSOR: &str =
        "newton 1.6 add_mjcf imports no <sensor>: sensordata would silently read empty";
    const NO_CONTACT: &str =
        "this adapter steps with contacts = None: the scene would run without contacts";
    match feature {
        TaskFeature::Spec17(Spec17Row::ActuatorPd) => Mapping::approximated(CONTROLLER),
        TaskFeature::Spec17(Spec17Row::ContactFrictionCone) => {
            Mapping::native("selectable pyramidal or elliptic (spec 17.2), `SolverMuJoCo(cone=)`")
        }
        TaskFeature::Spec17(Spec17Row::ContactSoftParams) => Mapping::approximated(SOLVER),
        TaskFeature::Spec17(Spec17Row::JointArmature)
        | TaskFeature::Capability(Feature::JointArmature) => {
            // Verified: `Model.joint_armature` carries the MJCF value at the joint's dof.
            Mapping::native("armature")
        }
        TaskFeature::Spec17(Spec17Row::SensorContactForce) => Mapping::approximated(CONTACT),
        // Verified: `Model.joint_type` carries all four MJCF joint kinds, and
        // `Model.joint_limit_lower` the scene's limits.
        TaskFeature::Capability(
            Feature::JointFree
            | Feature::JointBall
            | Feature::JointHinge
            | Feature::JointSlide
            | Feature::JointFixed
            | Feature::JointLimit,
        ) => Mapping::native("Newton's own joint, imported from the MJCF"),
        TaskFeature::Capability(
            Feature::ActuatorMotor
            | Feature::ActuatorPosition
            | Feature::ActuatorVelocity
            | Feature::ActuatorGeneral
            | Feature::ActuatorOnJoint
            | Feature::ActuatorOnTendon
            | Feature::ActuatorOnSite
            | Feature::Tendon,
        ) => Mapping::blocked(NO_ACTUATOR),
        TaskFeature::Capability(
            Feature::SensorJointPos
            | Feature::SensorJointVel
            | Feature::SensorActuatorFrc
            | Feature::SensorFramePos
            | Feature::SensorFrameQuat
            | Feature::SensorAccelerometer
            | Feature::SensorGyro
            | Feature::SensorForce
            | Feature::SensorTorque
            | Feature::SensorTouch
            | Feature::SensorRangeFinder
            | Feature::SensorCamera,
        ) => Mapping::blocked(NO_SENSOR),
        // Every scene carries the MJCF default cone, so blocking it would block everything.
        // It is warned about instead: the run keeps going, loudly (spec 14.4 severity rules).
        TaskFeature::Capability(Feature::ContactPyramidal) => Mapping::approximated(NO_CONTACT),
        // These are contact semantics a scene explicitly asked for, and this adapter has none.
        TaskFeature::Capability(
            Feature::ContactElliptic
            | Feature::ContactSoftParams
            | Feature::ContactCondim6
            | Feature::ContactMesh
            | Feature::ContactHeightField,
        ) => Mapping::blocked(NO_CONTACT),
        // Joint springs and friction loss were not checked against the importer.
        TaskFeature::Capability(_) => Mapping::unverified(),
    }
}

/// `PhysX`, likewise: only what spec 17.2 states, including the one row the spec marks as
/// unsupported outright.
fn physx(feature: TaskFeature) -> Mapping {
    const DRIVE: &str = "drive stiffness / damping (spec 17.2), not a MuJoCo position gain";
    const OFFSET: &str = "contact offset (spec 17.2), not MuJoCo impedance";
    const NO_ARMATURE: &str = "PhysX has no joint armature (spec 17.2: unsupported, warn)";
    match feature {
        TaskFeature::Spec17(Spec17Row::ActuatorPd)
        | TaskFeature::Capability(Feature::ActuatorPosition) => Mapping::approximated(DRIVE),
        TaskFeature::Spec17(Spec17Row::ContactFrictionCone)
        | TaskFeature::Capability(Feature::ContactPyramidal) => Mapping::native("pyramidal"),
        TaskFeature::Capability(Feature::ContactElliptic) => Mapping {
            status: Status::Unsupported("PhysX friction is pyramidal (spec 17.2)"),
            severity: Severity::Warning,
        },
        TaskFeature::Spec17(Spec17Row::ContactSoftParams)
        | TaskFeature::Capability(Feature::ContactSoftParams) => Mapping::approximated(OFFSET),
        TaskFeature::Spec17(Spec17Row::JointArmature)
        | TaskFeature::Capability(Feature::JointArmature) => Mapping {
            status: Status::Unsupported(NO_ARMATURE),
            severity: Severity::Warning,
        },
        TaskFeature::Spec17(Spec17Row::SensorContactForce)
        | TaskFeature::Capability(Feature::SensorForce | Feature::SensorTouch) => {
            Mapping::approximated("contact report (spec 17.2)")
        }
        TaskFeature::Capability(_) => Mapping::unverified(),
    }
}

/// The whole spec 17.2 table, materialised so it can be iterated and checked for gaps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticMapping {
    rows: std::collections::BTreeMap<(TaskFeature, BackendKind), Mapping>,
}

impl Default for SemanticMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticMapping {
    /// Builds the full cross product of features and backends.
    pub fn new() -> Self {
        let mut rows = std::collections::BTreeMap::new();
        for feature in TaskFeature::all() {
            for backend in BackendKind::ALL {
                rows.insert((feature, backend), lookup(feature, backend));
            }
        }
        Self { rows }
    }

    pub fn get(&self, feature: TaskFeature, backend: BackendKind) -> Option<Mapping> {
        self.rows.get(&(feature, backend)).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = ((TaskFeature, BackendKind), Mapping)> + '_ {
        self.rows.iter().map(|(key, mapping)| (*key, *mapping))
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// One line of a [`MappingReport`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappingRow {
    pub feature: TaskFeature,
    pub mapping: Mapping,
}

/// What one backend makes of one scene (spec 17.2). Printed in the header of
/// `es backend compare`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappingReport {
    pub backend: BackendKind,
    /// One row per feature the scene actually uses, in a stable order (spec 3.4).
    pub rows: Vec<MappingRow>,
    /// Any unmapped row with `severity: error` (spec 14.4).
    pub blocked: bool,
}

impl MappingReport {
    /// The rows that block, for an error message that names them.
    pub fn blocking(&self) -> impl Iterator<Item = &MappingRow> {
        self.rows.iter().filter(|row| row.mapping.blocks())
    }
}

/// `MuJoCo`'s defaults; a geom that keeps them is not asking for soft contact parameters.
const DEFAULT_SOLREF: [f64; 2] = [0.02, 1.0];
const DEFAULT_SOLIMP: [f64; 5] = [0.9, 0.95, 0.001, 0.5, 2.0];

/// Scans `scene` for the features it uses and reports how `backend` maps each (spec 17.2).
///
/// Feature rows come from [`Requirements::from_scene`], so the report and the compile-time
/// capability check (spec 11.6) see exactly the same scene. The five spec 17.2 rows are added
/// on top, each when the scene actually asks for it.
pub fn mapping_report(scene: &SceneDesc, backend: BackendKind) -> MappingReport {
    let mut used: BTreeSet<TaskFeature> = Requirements::from_scene(scene)
        .features
        .into_iter()
        .map(TaskFeature::Capability)
        .collect();

    if scene
        .actuators
        .iter()
        .any(|a| matches!(a.kind, ActuatorKind::Position { .. }))
    {
        used.insert(TaskFeature::Spec17(Spec17Row::ActuatorPd));
    }
    // Every scene has a friction cone, so this row is always asked.
    used.insert(TaskFeature::Spec17(Spec17Row::ContactFrictionCone));
    if scene
        .bodies
        .iter()
        .flat_map(|b| &b.geoms)
        .any(|g| g.solref != DEFAULT_SOLREF || g.solimp != DEFAULT_SOLIMP)
    {
        used.insert(TaskFeature::Spec17(Spec17Row::ContactSoftParams));
    }
    if scene.joints.iter().any(|j| j.armature != 0.0) {
        used.insert(TaskFeature::Spec17(Spec17Row::JointArmature));
    }
    if scene
        .sensors
        .iter()
        .any(|s| matches!(s.kind, SensorKind::Force | SensorKind::Touch))
    {
        used.insert(TaskFeature::Spec17(Spec17Row::SensorContactForce));
    }

    let rows: Vec<MappingRow> = used
        .into_iter()
        .map(|feature| MappingRow {
            feature,
            mapping: lookup(feature, backend),
        })
        .collect();
    let blocked = rows.iter().any(|row| row.mapping.blocks());
    MappingReport {
        backend,
        rows,
        blocked,
    }
}

const FEATURE_WIDTH: usize = 24;
const STATUS_WIDTH: usize = 12;
const SEVERITY_WIDTH: usize = 8;

impl fmt::Display for MappingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "semantic mapping report - backend `{}` (spec 17.2)",
            self.backend
        )?;
        writeln!(
            f,
            "{:<FEATURE_WIDTH$}  {:<STATUS_WIDTH$}  {:<SEVERITY_WIDTH$}  note",
            "feature", "status", "severity"
        )?;
        writeln!(
            f,
            "{}  {}  {}  {}",
            "-".repeat(FEATURE_WIDTH),
            "-".repeat(STATUS_WIDTH),
            "-".repeat(SEVERITY_WIDTH),
            "-".repeat(40)
        )?;
        for row in &self.rows {
            writeln!(
                f,
                "{:<FEATURE_WIDTH$}  {:<STATUS_WIDTH$}  {:<SEVERITY_WIDTH$}  {}",
                row.feature.to_string(),
                row.mapping.status.label(),
                row.mapping.severity.label(),
                row.mapping.status.note()
            )?;
        }
        if self.blocked {
            let names: Vec<String> = self.blocking().map(|r| r.feature.to_string()).collect();
            write!(
                f,
                "blocked: yes - unmapped with severity error (spec 14.4): {}",
                names.join(", ")
            )
        } else {
            write!(f, "blocked: no")
        }
    }
}

/// The `|dqpos|` above which two backends are called diverged.
pub const DIVERGENCE_TOL: f64 = 1e-6;

/// One `es backend compare` run (spec 17.2), scored with spec 3.5 tier 3 metrics.
#[derive(Clone, Debug)]
pub struct CompareReport {
    pub backend_a: String,
    pub backend_b: String,
    /// What each backend *declares* (spec 17.3: only `mujoco-cpu` may claim bitwise).
    pub tier_a: DeterminismTier,
    pub tier_b: DeterminismTier,
    pub n_ticks: u32,
    /// `max |qpos_a - qpos_b|` over every tick and every element.
    pub max_dqpos: f64,
    /// `max |qvel_a - qvel_b|` over every tick and every element.
    pub max_dqvel: f64,
    /// Energy drift **proxy**: `sum 1/2 qvel^2` at the last tick. It is not the system's
    /// energy — no mass matrix, no potential term — but it is monotone in the kinetic part and
    /// needs nothing from the backend beyond `qvel`, so it is comparable across backends.
    /// Replace it with a real Hamiltonian when `ModelInfo` carries inertia.
    pub energy_proxy_a: f64,
    pub energy_proxy_b: f64,
    /// `max |energy_proxy_a - energy_proxy_b|` over every tick.
    pub max_energy_delta: f64,
    /// First tick whose `max |dqpos|` exceeded [`DIVERGENCE_TOL`].
    pub divergence_tick: Option<u32>,
    /// `None` when the backend's name is not one of the spec 17.2 table's.
    pub mapping_a: Option<MappingReport>,
    pub mapping_b: Option<MappingReport>,
}

fn energy_proxy(qvel: &[f64]) -> f64 {
    0.5 * qvel.iter().map(|v| v * v).sum::<f64>()
}

fn max_abs_diff(what: &'static str, a: &[f64], b: &[f64]) -> Result<f64, PhysicsError> {
    if a.len() != b.len() {
        return Err(PhysicsError::ShapeMismatch {
            what,
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max))
}

/// Runs `scene` on two backends from the same reset state with the same control sequence, and
/// scores the difference with spec 3.5 tier 3 metrics. Backs `es backend compare` (spec 17.2).
///
/// `ctrl_seq` is cycled when it is shorter than `n_ticks`, and may be empty for a scene with no
/// actuators. A backend that is not loaded yet is loaded with [`LoadConfig::default`].
///
/// Returns `Err` rather than a report with a hole in it: a load or step failure is not a
/// comparison result (deviates from the packet brief's infallible signature on purpose).
pub fn compare_backends(
    a: &mut dyn PhysicsBackend,
    b: &mut dyn PhysicsBackend,
    scene: &SceneDesc,
    ctrl_seq: &[Vec<f64>],
    n_ticks: u32,
) -> Result<CompareReport, PhysicsError> {
    let cfg = LoadConfig::default();
    for backend in [&mut *a as &mut dyn PhysicsBackend, &mut *b] {
        if backend.model_info().is_none() {
            backend.load(scene, &cfg)?;
        }
        backend.reset(None, None)?;
    }

    let mut max_dqpos = 0.0_f64;
    let mut max_dqvel = 0.0_f64;
    let mut max_energy_delta = 0.0_f64;
    let mut divergence_tick = None;
    let (mut energy_proxy_a, mut energy_proxy_b) = (0.0, 0.0);

    for tick in 0..n_ticks {
        if !ctrl_seq.is_empty() {
            let ctrl = &ctrl_seq[tick as usize % ctrl_seq.len()];
            a.set_ctrl(ctrl)?;
            b.set_ctrl(ctrl)?;
        }
        a.step(1)?;
        b.step(1)?;

        let (state_a, state_b) = (a.state(), b.state());
        let dqpos = max_abs_diff("qpos", state_a.qpos, state_b.qpos)?;
        max_dqpos = max_dqpos.max(dqpos);
        max_dqvel = max_dqvel.max(max_abs_diff("qvel", state_a.qvel, state_b.qvel)?);
        energy_proxy_a = energy_proxy(state_a.qvel);
        energy_proxy_b = energy_proxy(state_b.qvel);
        max_energy_delta = max_energy_delta.max((energy_proxy_a - energy_proxy_b).abs());
        if divergence_tick.is_none() && dqpos > DIVERGENCE_TOL {
            divergence_tick = Some(tick);
        }
    }

    let mapping_of = |backend: &dyn PhysicsBackend| {
        BackendKind::from_name(&backend.capabilities().name).map(|kind| mapping_report(scene, kind))
    };
    Ok(CompareReport {
        backend_a: a.capabilities().name.clone(),
        backend_b: b.capabilities().name.clone(),
        tier_a: a.capabilities().determinism,
        tier_b: b.capabilities().determinism,
        n_ticks,
        max_dqpos,
        max_dqvel,
        energy_proxy_a,
        energy_proxy_b,
        max_energy_delta,
        divergence_tick,
        mapping_a: mapping_of(a),
        mapping_b: mapping_of(b),
    })
}

impl fmt::Display for CompareReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for report in [&self.mapping_a, &self.mapping_b].into_iter().flatten() {
            writeln!(f, "{report}\n")?;
        }
        writeln!(
            f,
            "backend compare: `{}` vs `{}`, {} ticks (spec 3.5 tier 3)",
            self.backend_a, self.backend_b, self.n_ticks
        )?;
        writeln!(f, "{:<26}  {:>16}  {:>16}", "metric", "value", "")?;
        writeln!(
            f,
            "{}  {}  {}",
            "-".repeat(26),
            "-".repeat(16),
            "-".repeat(16)
        )?;
        writeln!(
            f,
            "{:<26}  {:>16}  {:>16}",
            "determinism declared",
            format!("{:?}", self.tier_a),
            format!("{:?}", self.tier_b)
        )?;
        writeln!(
            f,
            "{:<26}  {:>16.6e}  {:>16}",
            "max |dqpos|", self.max_dqpos, ""
        )?;
        writeln!(
            f,
            "{:<26}  {:>16.6e}  {:>16}",
            "max |dqvel|", self.max_dqvel, ""
        )?;
        writeln!(
            f,
            "{:<26}  {:>16.6e}  {:>16.6e}",
            "energy proxy (sum qv^2/2)", self.energy_proxy_a, self.energy_proxy_b
        )?;
        writeln!(
            f,
            "{:<26}  {:>16.6e}  {:>16}",
            "max energy proxy delta", self.max_energy_delta, ""
        )?;
        match self.divergence_tick {
            Some(tick) => write!(
                f,
                "{:<26}  {:>16}  (tolerance {DIVERGENCE_TOL:e})",
                "divergence tick", tick
            ),
            None => write!(
                f,
                "{:<26}  {:>16}  (tolerance {DIVERGENCE_TOL:e})",
                "divergence tick", "none"
            ),
        }
    }
}

#[cfg(test)]
// Exact values are the property under test: two identical integrators agree exactly.
#[allow(clippy::float_cmp)]
mod tests {
    use es_core::{FailureKind, PhysTick, TickRate};
    use es_physics_core::caps::{BatchSupport, FloatPrecision};
    use es_physics_core::{Capabilities, ModelInfo, StateView, StepReport};

    use super::*;
    use crate::tests_support::fixture;

    fn scene(name: &str) -> SceneDesc {
        es_assets::parse_mjcf(&fixture(name)).unwrap().scene
    }

    #[test]
    fn every_feature_and_backend_pair_has_a_row() {
        let table = SemanticMapping::new();
        assert_eq!(table.len(), (5 + 35) * 4);
        for feature in TaskFeature::all() {
            for backend in BackendKind::ALL {
                let mapping = table
                    .get(feature, backend)
                    .unwrap_or_else(|| panic!("no row for {feature} on {backend}"));
                assert_eq!(mapping, lookup(feature, backend));
                // A row always says something: an unexplained cell is what spec 17.2 forbids.
                assert!(!mapping.status.note().is_empty());
            }
        }
        // `ALL_FEATURES` must stay in step with `es-physics-core`; bump both when a
        // `Feature` variant is added.
        let unique: BTreeSet<Feature> = ALL_FEATURES.into_iter().collect();
        assert_eq!(unique.len(), ALL_FEATURES.len());
    }

    /// The five rows the spec 17.2 table spells out, asserted cell by cell.
    #[test]
    fn the_spec_table_rows_are_what_the_spec_says() {
        use BackendKind::{MjWarp, Newton, PhysX};
        use Spec17Row::{
            ActuatorPd, ContactFrictionCone, ContactSoftParams, JointArmature, SensorContactForce,
        };
        let cell = |row, backend| lookup(TaskFeature::Spec17(row), backend);
        let is_native = |m: Mapping| matches!(m.status, Status::Native(_));
        let is_approx = |m: Mapping| matches!(m.status, Status::Approximated(_));

        // MJWarp: position gain, pyramidal, impedance, armature, sensor.
        assert!(is_native(cell(ActuatorPd, MjWarp)));
        assert!(is_native(cell(ContactFrictionCone, MjWarp)));
        assert!(is_native(cell(ContactSoftParams, MjWarp)));
        assert!(is_native(cell(JointArmature, MjWarp)));
        // Newton: controller, selectable, solver-dependent, armature, contact.
        assert!(is_approx(cell(ActuatorPd, Newton)));
        assert!(is_native(cell(ContactFrictionCone, Newton)));
        assert!(is_approx(cell(ContactSoftParams, Newton)));
        assert!(is_native(cell(JointArmature, Newton)));
        assert!(is_approx(cell(SensorContactForce, Newton)));
        // PhysX: drive stiffness, pyramidal, contact offset, unsupported armature, report.
        assert!(is_approx(cell(ActuatorPd, PhysX)));
        assert!(is_native(cell(ContactFrictionCone, PhysX)));
        assert!(is_approx(cell(ContactSoftParams, PhysX)));
        assert_eq!(
            cell(JointArmature, PhysX).severity,
            Severity::Warning,
            "spec 17.2 marks PhysX armature unsupported with a warning, not a block"
        );
        assert!(matches!(
            cell(JointArmature, PhysX).status,
            Status::Unsupported(_)
        ));
        assert!(is_approx(cell(SensorContactForce, PhysX)));
    }

    /// Never a guessed native mapping: what has not been checked says so (spec 1.7).
    #[test]
    fn unchecked_rows_are_unsupported_warnings_with_a_todo() {
        // Joint springs were never checked against Newton's MJCF importer; sensors and
        // actuators were, and are blocked rather than merely unverified.
        let mapping = lookup(
            TaskFeature::Capability(Feature::JointSpring),
            BackendKind::Newton,
        );
        assert_eq!(mapping.severity, Severity::Warning);
        assert!(mapping.status.note().contains("TODO(api-notes)"));
        assert!(!mapping.blocks());
    }

    #[test]
    fn the_pendulum_is_blocked_on_mjwarp_and_not_on_mujoco_cpu() {
        let pendulum = scene("pendulum.xml");
        let cpu = mapping_report(&pendulum, BackendKind::MuJoCoCpu);
        assert!(!cpu.blocked, "{cpu}");
        // pendulum.xml asks for an elliptic cone and a non-zero armature.
        let features: Vec<String> = cpu.rows.iter().map(|r| r.feature.to_string()).collect();
        assert!(
            features.contains(&"joint.armature".to_owned()),
            "{features:?}"
        );
        assert!(
            features.contains(&"contact.friction_cone".to_owned()),
            "{features:?}"
        );
        assert!(
            features.contains(&"ContactElliptic".to_owned()),
            "{features:?}"
        );

        let warp = mapping_report(&pendulum, BackendKind::MjWarp);
        assert!(warp.blocked, "{warp}");
        let blocking: Vec<String> = warp.blocking().map(|r| r.feature.to_string()).collect();
        assert_eq!(blocking, vec!["ContactElliptic".to_owned()]);
        // The rendered table is what `es backend compare` prints.
        let text = warp.to_string();
        assert!(text.contains("blocked: yes"));
        assert!(text.contains("unsupported"));
        assert!(text.lines().count() > 4);
    }

    #[test]
    fn an_actuated_scene_reports_its_pd_tendon_and_contact_rows() {
        let actuated = scene("actuated.xml");
        let report = mapping_report(&actuated, BackendKind::MjWarp);
        let features: Vec<String> = report.rows.iter().map(|r| r.feature.to_string()).collect();
        for expected in ["actuator.pd", "sensor.contact_force", "Tendon"] {
            assert!(features.contains(&expected.to_owned()), "{features:?}");
        }
        assert!(report.blocked, "{report}");
        // Newton blocks it too, for its own reason: `add_mjcf` imports no `<actuator>`, so
        // running this scene there would quietly drop the actuation (spec 14.4).
        let newton = mapping_report(&actuated, BackendKind::Newton);
        assert!(newton.blocked, "{newton}");
        assert!(newton.to_string().contains("add_mjcf"), "{newton}");
    }

    /// A scene every backend can map is not blocked anywhere, so `blocked` is not just "true".
    #[test]
    fn a_plain_scene_blocks_nowhere() {
        let plain = es_assets::parse_mjcf(
            r#"<mujoco><worldbody><body name="b">
                 <joint name="j" type="hinge"/><geom name="g" type="sphere" size="0.1"/>
               </body></worldbody></mujoco>"#,
        )
        .unwrap()
        .scene;
        for backend in BackendKind::ALL {
            let report = mapping_report(&plain, backend);
            assert!(!report.blocked, "{report}");
            assert!(report.to_string().contains("blocked: no"));
        }
    }

    /// A two-body integrator: `qvel += (accel + ctrl) * 1`, `qpos += qvel`. One tick is one
    /// unit of time, which is all `compare_backends` needs to see a divergence grow.
    #[derive(Debug)]
    struct FakeBackend {
        caps: Capabilities,
        accel: f64,
        model: Option<ModelInfo>,
        qpos: Vec<f64>,
        qvel: Vec<f64>,
        ctrl: Vec<f64>,
        tick: PhysTick,
    }

    impl FakeBackend {
        fn new(name: &str, accel: f64) -> Self {
            Self {
                caps: Capabilities {
                    name: name.to_owned(),
                    determinism: DeterminismTier::SemanticEqual,
                    batch: BatchSupport {
                        max_envs: 1,
                        gpu_resident: false,
                    },
                    joints: BTreeSet::new(),
                    actuators: BTreeSet::new(),
                    sensors: BTreeSet::new(),
                    contact: BTreeSet::new(),
                    float: FloatPrecision::F64,
                    supports_reset_subset: false,
                    supports_state_get_set: true,
                    quirks: Vec::new(),
                },
                accel,
                model: None,
                qpos: Vec::new(),
                qvel: Vec::new(),
                ctrl: Vec::new(),
                tick: PhysTick::ZERO,
            }
        }
    }

    impl PhysicsBackend for FakeBackend {
        fn capabilities(&self) -> &Capabilities {
            &self.caps
        }

        fn load(&mut self, scene: &SceneDesc, cfg: &LoadConfig) -> Result<ModelInfo, PhysicsError> {
            let nq = scene.joints.len() as u32;
            let info = ModelInfo {
                nq,
                nv: nq,
                n_envs: cfg.n_envs,
                rate: TickRate::hz(1000),
                ..ModelInfo::default()
            };
            self.qpos = vec![0.0; (nq * cfg.n_envs) as usize];
            self.qvel = self.qpos.clone();
            self.ctrl = Vec::new();
            self.model = Some(info.clone());
            Ok(info)
        }

        fn model_info(&self) -> Option<&ModelInfo> {
            self.model.as_ref()
        }

        fn reset(
            &mut self,
            _envs: Option<&[u32]>,
            _state: Option<&StateView<'_>>,
        ) -> Result<(), PhysicsError> {
            self.qpos.fill(0.0);
            self.qvel.fill(0.0);
            self.tick = PhysTick::ZERO;
            Ok(())
        }

        fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), PhysicsError> {
            self.ctrl = ctrl.to_vec();
            Ok(())
        }

        fn step(&mut self, n_substeps: u32) -> Result<StepReport, PhysicsError> {
            self.model.as_ref().ok_or(PhysicsError::NotLoaded)?;
            for _ in 0..n_substeps {
                for (i, (pos, vel)) in self.qpos.iter_mut().zip(&mut self.qvel).enumerate() {
                    *vel += self.accel + self.ctrl.get(i).copied().unwrap_or(0.0);
                    *pos += *vel;
                }
            }
            self.tick = self.tick.add_ticks(u64::from(n_substeps));
            let failures: Vec<(u32, FailureKind)> = self
                .qpos
                .iter()
                .enumerate()
                .filter(|(_, v)| !v.is_finite())
                .map(|(env, _)| (env as u32, FailureKind::NanDetected))
                .collect();
            Ok(StepReport {
                tick: self.tick,
                failures,
            })
        }

        fn state(&self) -> StateView<'_> {
            StateView {
                n_envs: self.model.as_ref().map_or(0, |m| m.n_envs),
                tick: self.tick,
                qpos: &self.qpos,
                qvel: &self.qvel,
                ..StateView::default()
            }
        }

        fn set_state(&mut self, state: &StateView<'_>) -> Result<(), PhysicsError> {
            if state.qpos.len() != self.qpos.len() {
                return Err(PhysicsError::ShapeMismatch {
                    what: "qpos",
                    expected: self.qpos.len(),
                    got: state.qpos.len(),
                });
            }
            self.qpos.copy_from_slice(state.qpos);
            Ok(())
        }
    }

    fn one_joint() -> SceneDesc {
        es_assets::parse_mjcf(
            r#"<mujoco><worldbody><body name="b">
                 <joint name="j" type="hinge"/><geom name="g" type="sphere" size="0.1"/>
               </body></worldbody></mujoco>"#,
        )
        .unwrap()
        .scene
    }

    #[test]
    fn identical_backends_never_diverge() {
        let scene = one_joint();
        let mut a = FakeBackend::new("fake-a", -9.81);
        let mut b = FakeBackend::new("fake-b", -9.81);
        let report = compare_backends(&mut a, &mut b, &scene, &[], 100).unwrap();
        assert_eq!(report.divergence_tick, None);
        assert!(report.max_dqpos == 0.0 && report.max_dqvel == 0.0);
        assert!(report.energy_proxy_a > 0.0);
        assert_eq!(report.max_energy_delta, 0.0);
        // A name outside the spec 17.2 table has no mapping report to print.
        assert!(report.mapping_a.is_none() && report.mapping_b.is_none());
        assert!(report.to_string().contains("divergence tick"));
    }

    #[test]
    fn a_1e_9_perturbation_is_found_and_dated() {
        let scene = one_joint();
        let mut a = FakeBackend::new("fake-a", -9.81);
        let mut b = FakeBackend::new("fake-b", -9.81 + 1e-9);
        let report = compare_backends(&mut a, &mut b, &scene, &[], 100).unwrap();
        // dqpos after n ticks is 1e-9 * n(n+1)/2, so it crosses 1e-6 around tick 44.
        let tick = report.divergence_tick.expect("divergence not found");
        assert!((40..50).contains(&tick), "diverged at {tick}");
        assert!(report.max_dqpos > DIVERGENCE_TOL);
        assert!(report.max_dqvel > 0.0);
        assert!(report.max_energy_delta > 0.0);
        assert!(report.to_string().contains(&tick.to_string()));
    }

    /// The control sequence is applied to both, and cycled when shorter than the run.
    #[test]
    fn the_same_control_sequence_reaches_both_backends() {
        let scene = one_joint();
        let mut a = FakeBackend::new("mujoco-cpu", 0.0);
        let mut b = FakeBackend::new("mjwarp", 0.0);
        let ctrl = vec![vec![1.0], vec![-1.0]];
        let report = compare_backends(&mut a, &mut b, &scene, &ctrl, 10).unwrap();
        assert_eq!(report.divergence_tick, None);
        // Known names do get a mapping report, and it is printed above the metrics.
        assert!(report.mapping_a.is_some() && report.mapping_b.is_some());
        let text = report.to_string();
        assert!(text.contains("mujoco-cpu") && text.contains("mjwarp"));
        assert!(text.contains("semantic mapping report"));
    }
}
