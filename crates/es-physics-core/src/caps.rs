//! Backend capabilities, declared quirks, and the compile-time requirement check
//! (spec 4.3, spec 11.6, spec 17.2).
//!
//! A backend declares what it can do; the compiler checks a scene's requirements against that
//! declaration *before* a run starts (spec 11.6: the failure must appear at compile time, not
//! after a 4,096-env batch has been running for a while). Anything a backend maps
//! approximately is declared as a [`BackendQuirk`] rather than hidden (spec 17.2).

use std::collections::BTreeSet;
use std::fmt;

use es_assets::scene::{ActuatorKind, ActuatorTarget, JointKind, SceneDesc, SensorKind, Shape};
use serde::{Deserialize, Serialize};

/// Determinism tier a backend declares (spec 3.5). Tiers are kinds of guarantee, not a ladder:
/// a requirement matches a declaration exactly or not at all.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismTier {
    /// 0: same `*_hash`, so the definition is the same. No numerical guarantee.
    #[default]
    SemanticEqual,
    /// 1: same `execution_hash` on the same device and driver gives bit-identical results.
    /// **An external backend never declares this** (spec 4.3).
    Bitwise,
    /// 2: CPU vs GPU, or one backend swapped for another, within a defined tolerance.
    CrossBackend,
    /// 3: physics-metric tolerance against `MuJoCo` or against measurement.
    PhysicsMeaning,
    /// 4: the same IR run in `PyTorch` vs the native runtime, action-tensor tolerance (spec 8.9).
    PolicyEquivalent,
}

/// Float width the backend's state is computed in (spec 3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatPrecision {
    F32,
    F64,
}

/// One thing a backend may or may not be able to do. The four capability sets on
/// [`Capabilities`] are drawn from this one enum so that a requirement and a declaration are
/// always comparable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    // Joints (spec 18.1).
    JointFree,
    JointBall,
    JointHinge,
    JointSlide,
    JointFixed,
    JointLimit,
    JointArmature,
    JointSpring,
    JointFrictionLoss,
    // Actuators (spec 18.2).
    ActuatorMotor,
    ActuatorPosition,
    ActuatorVelocity,
    ActuatorGeneral,
    ActuatorOnJoint,
    ActuatorOnTendon,
    ActuatorOnSite,
    // Sensors (spec 18.3).
    SensorJointPos,
    SensorJointVel,
    SensorActuatorFrc,
    SensorFramePos,
    SensorFrameQuat,
    SensorAccelerometer,
    SensorGyro,
    SensorForce,
    SensorTorque,
    SensorTouch,
    SensorRangeFinder,
    SensorCamera,
    // Contact and collision geometry (spec 17.2).
    ContactPyramidal,
    ContactElliptic,
    ContactSoftParams,
    ContactCondim6,
    ContactMesh,
    ContactHeightField,
    // Other (spec 18.2): tendons are a transmission, not an actuator kind.
    Tendon,
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Feature {
    pub fn of_joint(kind: JointKind) -> Self {
        match kind {
            JointKind::Free => Self::JointFree,
            JointKind::Ball => Self::JointBall,
            JointKind::Hinge => Self::JointHinge,
            JointKind::Slide => Self::JointSlide,
            JointKind::Fixed => Self::JointFixed,
        }
    }

    pub fn of_actuator(kind: &ActuatorKind) -> Self {
        match kind {
            ActuatorKind::Motor => Self::ActuatorMotor,
            ActuatorKind::Position { .. } => Self::ActuatorPosition,
            ActuatorKind::Velocity { .. } => Self::ActuatorVelocity,
            ActuatorKind::General { .. } => Self::ActuatorGeneral,
        }
    }

    pub fn of_actuator_target(target: &ActuatorTarget) -> Self {
        match target {
            ActuatorTarget::Joint(_) => Self::ActuatorOnJoint,
            ActuatorTarget::Tendon(_) => Self::ActuatorOnTendon,
            ActuatorTarget::Site(_) => Self::ActuatorOnSite,
        }
    }

    pub fn of_sensor(kind: SensorKind) -> Self {
        match kind {
            SensorKind::JointPos => Self::SensorJointPos,
            SensorKind::JointVel => Self::SensorJointVel,
            SensorKind::ActuatorFrc => Self::SensorActuatorFrc,
            SensorKind::FramePos => Self::SensorFramePos,
            SensorKind::FrameQuat => Self::SensorFrameQuat,
            SensorKind::Accelerometer => Self::SensorAccelerometer,
            SensorKind::Gyro => Self::SensorGyro,
            SensorKind::Force => Self::SensorForce,
            SensorKind::Torque => Self::SensorTorque,
            SensorKind::Touch => Self::SensorTouch,
            SensorKind::RangeFinder => Self::SensorRangeFinder,
            SensorKind::Camera => Self::SensorCamera,
        }
    }

    /// The collision feature a shape needs, for the shapes that are not primitives every
    /// backend has.
    pub fn of_shape(shape: &Shape) -> Option<Self> {
        match shape {
            Shape::Mesh { .. } => Some(Self::ContactMesh),
            Shape::HeightField { .. } => Some(Self::ContactHeightField),
            _ => None,
        }
    }
}

/// How many environments a backend steps at once, and where their state lives (spec 12.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSupport {
    /// Largest `n_envs` the backend declares native support for.
    pub max_envs: u32,
    /// Whether env state stays on the GPU between steps (spec 12.1, spec 12.2).
    pub gpu_resident: bool,
}

/// A mapping a backend performs that is *not* exactly what the scene asked for (spec 17.2).
///
/// Quirks are declared, never hidden: `es backend compare` reports them, and a mismatch that
/// matters is an error at the call site, not a surprise in the trajectory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendQuirk {
    pub feature: Feature,
    pub description: String,
}

impl BackendQuirk {
    pub fn new(feature: Feature, description: impl Into<String>) -> Self {
        Self {
            feature,
            description: description.into(),
        }
    }
}

/// Everything a backend declares about itself (spec 4.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Stable backend name, e.g. `mujoco-cpu`; the `--backends` spelling of spec 17.2.
    pub name: String,
    pub determinism: DeterminismTier,
    pub batch: BatchSupport,
    pub joints: BTreeSet<Feature>,
    pub actuators: BTreeSet<Feature>,
    pub sensors: BTreeSet<Feature>,
    pub contact: BTreeSet<Feature>,
    pub float: FloatPrecision,
    /// Whether [`reset`](crate::PhysicsBackend::reset) can reset a subset of envs.
    pub supports_reset_subset: bool,
    /// Whether state can be read back and written whole (evidence bundles, spec 27.1).
    pub supports_state_get_set: bool,
    /// Approximate mappings, declared (spec 17.2).
    pub quirks: Vec<BackendQuirk>,
}

impl Capabilities {
    /// Whether `feature` appears in any of the four sets.
    pub fn has(&self, feature: Feature) -> bool {
        self.joints.contains(&feature)
            || self.actuators.contains(&feature)
            || self.sensors.contains(&feature)
            || self.contact.contains(&feature)
    }
}

/// What a scene (plus the run configuration) needs from a backend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirements {
    pub n_envs: u32,
    /// State must stay GPU resident between steps.
    pub gpu_resident: bool,
    /// Exact determinism tier the workflow needs, if any (spec 3.5, spec 17.3).
    pub determinism: Option<DeterminismTier>,
    pub features: BTreeSet<Feature>,
    pub reset_subset: bool,
    pub state_get_set: bool,
    pub float: Option<FloatPrecision>,
}

impl Requirements {
    /// The features a scene uses. Run-level requirements (`n_envs`, determinism, residency)
    /// come from the compile request, not from the scene, and stay at their defaults here.
    pub fn from_scene(scene: &SceneDesc) -> Self {
        let mut features = BTreeSet::new();
        for joint in &scene.joints {
            features.insert(Feature::of_joint(joint.kind));
            if joint.range.is_some() {
                features.insert(Feature::JointLimit);
            }
            if joint.armature != 0.0 {
                features.insert(Feature::JointArmature);
            }
            if joint.stiffness != 0.0 {
                features.insert(Feature::JointSpring);
            }
            if joint.friction_loss != 0.0 {
                features.insert(Feature::JointFrictionLoss);
            }
        }
        for actuator in &scene.actuators {
            features.insert(Feature::of_actuator(&actuator.kind));
            features.insert(Feature::of_actuator_target(&actuator.target));
        }
        for sensor in &scene.sensors {
            features.insert(Feature::of_sensor(sensor.kind));
        }
        for body in &scene.bodies {
            for geom in &body.geoms {
                if let Some(feature) = Feature::of_shape(&geom.shape) {
                    features.insert(feature);
                }
                if geom.condim == 6 {
                    features.insert(Feature::ContactCondim6);
                }
            }
        }
        if !scene.tendons.is_empty() {
            features.insert(Feature::Tendon);
        }
        features.insert(match scene.options.cone {
            es_assets::scene::FrictionCone::Pyramidal => Feature::ContactPyramidal,
            es_assets::scene::FrictionCone::Elliptic => Feature::ContactElliptic,
        });
        Self {
            n_envs: 1,
            features,
            ..Self::default()
        }
    }
}

/// One requirement a backend cannot meet (spec 11.6's `DEP-114` payload).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Unsupported {
    Feature(Feature),
    BatchSize {
        requested: u32,
        max_envs: u32,
    },
    GpuResidency,
    Determinism {
        required: DeterminismTier,
        declared: DeterminismTier,
    },
    ResetSubset,
    StateGetSet,
    Float {
        required: FloatPrecision,
        declared: FloatPrecision,
    },
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Feature(feature) => write!(f, "feature {feature}"),
            Self::BatchSize {
                requested,
                max_envs,
            } => {
                write!(f, "{requested} envs (backend maximum is {max_envs})")
            }
            Self::GpuResidency => write!(f, "GPU-resident state"),
            Self::Determinism { required, declared } => {
                write!(
                    f,
                    "determinism {required:?} (backend declares {declared:?})"
                )
            }
            Self::ResetSubset => write!(f, "resetting a subset of envs"),
            Self::StateGetSet => write!(f, "reading and writing whole state"),
            Self::Float { required, declared } => {
                write!(f, "{required:?} state (backend computes in {declared:?})")
            }
        }
    }
}

/// Checks a scene's requirements against a backend's declaration (spec 11.6).
///
/// The compiler calls this and refuses to compile on a non-empty result. Order is stable: the
/// sets are `BTreeSet`s and the run-level checks run in a fixed order (spec 3.4).
pub fn check_requirements(req: &Requirements, caps: &Capabilities) -> Vec<Unsupported> {
    let mut out = Vec::new();
    if req.n_envs > caps.batch.max_envs {
        out.push(Unsupported::BatchSize {
            requested: req.n_envs,
            max_envs: caps.batch.max_envs,
        });
    }
    if req.gpu_resident && !caps.batch.gpu_resident {
        out.push(Unsupported::GpuResidency);
    }
    if let Some(required) = req.determinism {
        if required != caps.determinism {
            out.push(Unsupported::Determinism {
                required,
                declared: caps.determinism,
            });
        }
    }
    if let Some(required) = req.float {
        if required != caps.float {
            out.push(Unsupported::Float {
                required,
                declared: caps.float,
            });
        }
    }
    if req.reset_subset && !caps.supports_reset_subset {
        out.push(Unsupported::ResetSubset);
    }
    if req.state_get_set && !caps.supports_state_get_set {
        out.push(Unsupported::StateGetSet);
    }
    out.extend(
        req.features
            .iter()
            .filter(|f| !caps.has(**f))
            .map(|f| Unsupported::Feature(*f)),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> Capabilities {
        Capabilities {
            name: "test".to_owned(),
            determinism: DeterminismTier::PhysicsMeaning,
            batch: BatchSupport {
                max_envs: 4,
                gpu_resident: false,
            },
            joints: [
                Feature::JointHinge,
                Feature::JointSlide,
                Feature::JointLimit,
                Feature::JointArmature,
            ]
            .into(),
            actuators: [Feature::ActuatorMotor, Feature::ActuatorOnJoint].into(),
            sensors: [Feature::SensorJointPos].into(),
            contact: [Feature::ContactPyramidal].into(),
            float: FloatPrecision::F64,
            supports_reset_subset: true,
            supports_state_get_set: true,
            quirks: Vec::new(),
        }
    }

    #[test]
    fn a_met_requirement_reports_nothing() {
        let req = Requirements {
            n_envs: 4,
            determinism: Some(DeterminismTier::PhysicsMeaning),
            features: [Feature::JointHinge, Feature::ContactPyramidal].into(),
            reset_subset: true,
            state_get_set: true,
            float: Some(FloatPrecision::F64),
            gpu_resident: false,
        };
        assert_eq!(check_requirements(&req, &caps()), Vec::new());
    }

    #[test]
    fn every_unmet_requirement_is_reported_once() {
        let mut bare = caps();
        bare.supports_reset_subset = false;
        bare.supports_state_get_set = false;
        let req = Requirements {
            n_envs: 64,
            gpu_resident: true,
            determinism: Some(DeterminismTier::Bitwise),
            features: [
                Feature::JointBall,
                Feature::ContactMesh,
                Feature::JointHinge,
            ]
            .into(),
            reset_subset: true,
            state_get_set: true,
            float: Some(FloatPrecision::F32),
        };
        let got = check_requirements(&req, &bare);
        assert_eq!(
            got,
            vec![
                Unsupported::BatchSize {
                    requested: 64,
                    max_envs: 4
                },
                Unsupported::GpuResidency,
                Unsupported::Determinism {
                    required: DeterminismTier::Bitwise,
                    declared: DeterminismTier::PhysicsMeaning,
                },
                Unsupported::Float {
                    required: FloatPrecision::F32,
                    declared: FloatPrecision::F64,
                },
                Unsupported::ResetSubset,
                Unsupported::StateGetSet,
                Unsupported::Feature(Feature::JointBall),
                Unsupported::Feature(Feature::ContactMesh),
            ]
        );
        // Reported, not swallowed: every entry renders (spec 11.6 DEP-114).
        assert!(got.iter().all(|u| !u.to_string().is_empty()));
    }

    #[test]
    fn requirements_come_from_what_the_scene_actually_uses() {
        let import = es_assets::parse_mjcf(
            r#"<mujoco>
                 <worldbody>
                   <body name="b">
                     <joint name="j" type="hinge" armature="0.01" range="-1 1"/>
                     <geom name="g" type="sphere" size="0.1"/>
                   </body>
                 </worldbody>
                 <actuator><motor name="m" joint="j"/></actuator>
                 <sensor><jointpos joint="j"/></sensor>
               </mujoco>"#,
        )
        .unwrap();
        let req = Requirements::from_scene(&import.scene);
        for expected in [
            Feature::JointHinge,
            Feature::JointLimit,
            Feature::JointArmature,
            Feature::ActuatorMotor,
            Feature::ActuatorOnJoint,
            Feature::SensorJointPos,
            Feature::ContactPyramidal,
        ] {
            assert!(req.features.contains(&expected), "missing {expected}");
        }
        assert!(!req.features.contains(&Feature::JointFree));
        assert!(!req.features.contains(&Feature::Tendon));
        assert_eq!(check_requirements(&req, &caps()), Vec::new());
    }

    #[test]
    fn serde_round_trip() {
        let caps = caps();
        let json = serde_json::to_string(&caps).unwrap();
        assert_eq!(serde_json::from_str::<Capabilities>(&json).unwrap(), caps);
    }
}
