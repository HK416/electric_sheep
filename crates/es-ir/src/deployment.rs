//! Deployment IR: safety limits, execution mode, deadlines and fallback (spec 9).
//!
//! Schema and validation only. The runtime `SafetyPlane` lives in `es-safety` (layer 8), which
//! consumes this IR and never depends on `es-policy` (spec 4.2 rule 8). Nothing here refers to
//! a policy, a runtime or a tensor, so that dependency stays impossible to write.
//!
//! **INV-12: there is no way to turn the envelope off.** No `enabled` flag, no `Option` around
//! [`SafetyEnvelope`], no "disabled" execution mode, and no watchdog variant for `NaN`/`Inf` —
//! non-finite action rejection is unconditional in [`SafetyEnvelope::contains`], so it cannot
//! be left out of a configuration. A test that needs room widens the limits.
//!
//! Sizes: every per-joint vector is `n_joints` long ([`SafetyEnvelope::n_joints`]). The
//! const-generic `SafetyEnvelope<NJ>` of Appendix B.4 is the pre-allocated `es-safety` runtime
//! mirror of this type, not this type.

use std::collections::BTreeSet;

use es_core::time::TickRate;
use serde::{Deserialize, Serialize};

use crate::diag::Diagnostic;
use crate::hash::CanonWriter;

pub const SCHEMA_VERSION: u32 = 1;

// --- Diagnostic codes ---------------------------------------------------------------------
// `codes::DEP_031` (envelope exceeds robot capability) needs the robot description, which this
// IR only references by name; that check belongs to the cross-IR pass (P26), not here.
// TODO(P27-merge): move these rows into `crate::codes::CODES` so they gain titles.
pub const DEP_001: &str = "DEP-001";
pub const DEP_010: &str = "DEP-010";
pub const DEP_011: &str = "DEP-011";
pub const DEP_012: &str = "DEP-012";
pub const DEP_013: &str = "DEP-013";
pub const DEP_014: &str = "DEP-014";
pub const DEP_015: &str = "DEP-015";
pub const DEP_020: &str = "DEP-020";
pub const DEP_021: &str = "DEP-021";
pub const DEP_022: &str = "DEP-022";
pub const DEP_023: &str = "DEP-023";
pub const DEP_024: &str = "DEP-024";
pub const DEP_030: &str = "DEP-030";
pub const DEP_040: &str = "DEP-040";

// --- Small shared types -------------------------------------------------------------------

/// A duration in whole microseconds. Integer by construction: `f64` time is forbidden
/// (spec 3.4).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Micros(pub u64);

/// A closed interval. `lower < upper` is a validated invariant, not an assumption.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Limit {
    pub lower: f64,
    pub upper: f64,
}

impl Limit {
    pub fn symmetric(half: f64) -> Self {
        Self {
            lower: -half,
            upper: half,
        }
    }

    pub fn contains(self, v: f64) -> bool {
        v.is_finite() && v >= self.lower && v <= self.upper
    }
}

// --- Robot --------------------------------------------------------------------------------

/// Which robot this deployment targets (spec 9.2): a simulated scene or a physical drive.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobotTarget {
    /// Scene asset path; the Safety Plane runs identically here (spec 9.5).
    Simulated { scene: String },
    /// Driver / bus identifier of the real robot.
    Physical { driver: String },
}

/// Reference to the robot the envelope is written against (spec 9.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RobotRef {
    pub name: String,
    pub target: RobotTarget,
    pub n_joints: usize,
}

// --- Action contract ----------------------------------------------------------------------

/// Command space of the action the Safety Plane validates (spec 8.5).
///
/// TODO(P26-merge): Learning IR owns the authoring-side `ActionSpec` (spec 8.5); this is the
/// deployment-side contract, reduced to what the Safety Plane consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSpace {
    JointPosition,
    JointVelocity,
    JointTorque,
    EePose,
    EeDelta,
    Gripper,
    Composite,
}

/// Shape of the action chunk arriving at the Safety Plane (spec 8.5, spec 8.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionContract {
    pub space: ActionSpace,
    pub dim: usize,
    /// Prediction length H.
    pub horizon: usize,
    /// Executed length K, `1 <= K <= H`.
    pub execute_chunk: usize,
}

/// How a chunk is consumed (spec 8.5 `ActionExecutionMode`, spec 9.2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Execute the whole chunk, then replan.
    OpenLoopChunk,
    /// Execute K, then replan (default).
    RecedingHorizon,
    /// ACT: exponentially weighted average of overlapping predictions.
    TemporalEnsemble { decay: f64 },
    /// Compute the next chunk while the current one runs (spec 8.6).
    RealTimeChunking,
}

/// Controller and replanning rates (spec 9.2 `RateSpec`). Rational, so a period is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateSpec {
    pub control: TickRate,
    pub inference: TickRate,
}

impl RateSpec {
    /// Control period in whole microseconds, rounded down.
    pub fn control_period(self) -> Micros {
        Micros(1_000_000 * self.control.den() / self.control.num())
    }
}

// --- Deadlines ----------------------------------------------------------------------------

/// The three budgets of one control cycle (spec 9.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deadlines {
    /// How stale an observation may be when inference starts.
    pub observation_age: Micros,
    /// Inference budget. May exceed the control period: that is the normal case (spec 8.6).
    pub inference_budget: Micros,
    /// Time from a validated action to the actuator write. Must fit one control period.
    pub actuation_budget: Micros,
}

// --- Workspace ----------------------------------------------------------------------------

/// Half-space `n . x <= d`, the face of a convex polyhedron.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HalfSpace {
    pub normal: [f64; 3],
    pub offset: f64,
}

/// End-effector workspace (spec 9.3): box, cylinder or convex polyhedron.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Workspace {
    Box {
        min: [f64; 3],
        max: [f64; 3],
    },
    Cylinder {
        center: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        half_height: f64,
    },
    ConvexHull {
        faces: Vec<HalfSpace>,
    },
}

// --- Safety envelope ----------------------------------------------------------------------

/// Bounds on the action's first and second difference per control tick (spec 9.3
/// `rate_limit`). Per joint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RateLimit {
    /// Max `|a[t] - a[t-1]|` per tick.
    pub first_diff_max: Vec<f64>,
    /// Max `|a[t] - 2a[t-1] + a[t-2]|` per tick.
    pub second_diff_max: Vec<f64>,
}

/// The static and dynamic constraints applied to every policy output (spec 9.3).
///
/// **All of it is mandatory.** There is no field that switches a constraint off; `jerk_max` is
/// the single optional entry the spec marks optional, and absent means "not rate limited on
/// jerk", never "position and torque limits are off".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    /// Hard joint limits.
    pub position: Vec<Limit>,
    /// Soft margin inside each hard limit; `2 * margin < upper - lower`.
    pub position_soft_margin: Vec<f64>,
    pub velocity_max: Vec<f64>,
    pub acceleration_max: Vec<f64>,
    pub torque_max: Vec<f64>,
    /// Optional per spec 9.3.
    pub jerk_max: Option<Vec<f64>>,
    pub action_rate: RateLimit,
    pub workspace: Workspace,
    /// End-effector linear speed cap.
    pub ee_velocity_max: f64,
    /// Minimum self-collision distance.
    pub min_self_distance: f64,
    /// Minimum distance to the environment.
    pub min_env_distance: f64,
    /// Contact force cap.
    pub contact_force_max: f64,
}

impl SafetyEnvelope {
    pub fn n_joints(&self) -> usize {
        self.position.len()
    }

    /// Whether a joint state is inside the envelope. Non-finite input is never inside: `NaN`
    /// cannot be clamped, so it is a fallback trigger (spec 9.3 `action_validity`, INV-12).
    pub fn contains(&self, q: &[f64], qd: &[f64], tau: &[f64]) -> bool {
        let n = self.n_joints();
        if q.len() != n || qd.len() != n || tau.len() != n {
            return false;
        }
        (0..n).all(|i| {
            self.position[i].contains(q[i])
                && Limit::symmetric(self.velocity_max[i]).contains(qd[i])
                && Limit::symmetric(self.torque_max[i]).contains(tau[i])
        })
    }
}

// --- Watchdogs and fallback ---------------------------------------------------------------

/// A condition the Safety Plane arms (spec 9.4).
///
/// There is deliberately no `NanInf` variant: non-finite rejection is unconditional, so it
/// cannot be omitted from a `WatchdogSet` (INV-12).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Watchdog {
    InferenceDeadline { budget: Micros },
    ChunkUnderrun,
    StaleObservation { max_age: Micros },
    EnvelopeViolationRate { window: u32, max_frac: f64 },
    ControllerHeartbeat { timeout: Micros },
    SensorDropout { sensor: String, max_gap: Micros },
}

impl Watchdog {
    /// Identity for duplicate detection: one per kind, except `SensorDropout`, which is one
    /// per sensor.
    fn key(&self) -> String {
        match self {
            Self::InferenceDeadline { .. } => "inference_deadline".into(),
            Self::ChunkUnderrun => "chunk_underrun".into(),
            Self::StaleObservation { .. } => "stale_observation".into(),
            Self::EnvelopeViolationRate { .. } => "envelope_violation_rate".into(),
            Self::ControllerHeartbeat { .. } => "controller_heartbeat".into(),
            Self::SensorDropout { sensor, .. } => format!("sensor_dropout:{sensor}"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WatchdogSet(pub Vec<Watchdog>);

/// What runs when a watchdog fires (spec 9.4). Deterministic and policy-free by construction:
/// every variant is data this crate can describe without a network.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackPolicy {
    /// Hold the current pose.
    HoldPosition,
    /// Decelerate to a stop.
    ZeroVelocity,
    /// Replay a precomputed safe trajectory; each waypoint is one joint position vector.
    RetractToHome { trajectory: Vec<Vec<f64>> },
    /// Hand over to a classical controller.
    HandoffController { id: String },
    /// Stop now and latch.
    EmergencyStop,
}

// --- The IR -------------------------------------------------------------------------------

/// Deployment IR (spec 9.2): the constraints an action is executed under.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeploymentIr {
    pub schema_version: u32,
    pub robot: RobotRef,
    pub action: ActionContract,
    pub safety: SafetyEnvelope,
    pub execution: ExecutionMode,
    pub deadlines: Deadlines,
    pub watchdogs: WatchdogSet,
    pub fallback: FallbackPolicy,
    pub rate: RateSpec,
}

impl DeploymentIr {
    /// Every check of spec 9.2 / 9.3 / 9.4 this IR can make on its own. Empty means valid.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        if self.schema_version != SCHEMA_VERSION {
            out.push(Diagnostic::new(
                DEP_001,
                format!(
                    "schema_version {} is not supported (expected {SCHEMA_VERSION})",
                    self.schema_version
                ),
            ));
        }
        let n = self.robot.n_joints;
        if n == 0 {
            out.push(Diagnostic::new(DEP_010, "robot.n_joints is 0"));
        }
        self.check_envelope(n, &mut out);
        self.check_timing(&mut out);
        self.check_watchdogs(&mut out);
        self.check_fallback(n, &mut out);
        self.check_action(n, &mut out);
        out
    }

    fn check_envelope(&self, n: usize, out: &mut Vec<Diagnostic>) {
        let s = &self.safety;
        if s.n_joints() != n {
            out.push(Diagnostic::new(
                DEP_010,
                format!(
                    "safety.position has {} entries but robot.n_joints is {n}",
                    s.n_joints()
                ),
            ));
        }
        for (name, v) in [
            ("safety.velocity_max", &s.velocity_max),
            ("safety.acceleration_max", &s.acceleration_max),
            ("safety.torque_max", &s.torque_max),
            (
                "safety.action_rate.first_diff_max",
                &s.action_rate.first_diff_max,
            ),
            (
                "safety.action_rate.second_diff_max",
                &s.action_rate.second_diff_max,
            ),
        ] {
            check_positive(name, v, n, out);
        }
        if let Some(jerk) = &s.jerk_max {
            check_positive("safety.jerk_max", jerk, n, out);
        }
        for (name, v) in [
            ("safety.ee_velocity_max", s.ee_velocity_max),
            ("safety.min_self_distance", s.min_self_distance),
            ("safety.min_env_distance", s.min_env_distance),
            ("safety.contact_force_max", s.contact_force_max),
        ] {
            check_scalar(name, v, out);
        }
        for (i, limit) in s.position.iter().enumerate() {
            if !limit.lower.is_finite() || !limit.upper.is_finite() {
                out.push(Diagnostic::new(
                    DEP_011,
                    format!("safety.position[{i}] is not finite"),
                ));
            } else if limit.lower >= limit.upper {
                out.push(Diagnostic::new(
                    DEP_012,
                    format!(
                        "safety.position[{i}]: lower {} is not below upper {}",
                        limit.lower, limit.upper
                    ),
                ));
            }
        }
        if s.position_soft_margin.len() != s.position.len() {
            out.push(Diagnostic::new(
                DEP_010,
                format!(
                    "safety.position_soft_margin has {} entries but safety.position has {}",
                    s.position_soft_margin.len(),
                    s.position.len()
                ),
            ));
        }
        for (i, (margin, limit)) in s.position_soft_margin.iter().zip(&s.position).enumerate() {
            if !margin.is_finite() || *margin < 0.0 {
                out.push(Diagnostic::new(
                    DEP_011,
                    format!("safety.position_soft_margin[{i}] is not a non-negative number"),
                ));
            } else if 2.0 * margin >= limit.upper - limit.lower {
                out.push(Diagnostic::new(
                    DEP_014,
                    format!("safety.position_soft_margin[{i}] leaves no room inside the limit"),
                ));
            }
        }
        check_workspace(&s.workspace, out);
    }

    fn check_timing(&self, out: &mut Vec<Diagnostic>) {
        let d = self.deadlines;
        for (name, v) in [
            ("deadlines.observation_age", d.observation_age),
            ("deadlines.inference_budget", d.inference_budget),
            ("deadlines.actuation_budget", d.actuation_budget),
        ] {
            if v.0 == 0 {
                out.push(Diagnostic::new(DEP_020, format!("{name} is 0")));
            }
        }
        let period = self.rate.control_period();
        if d.actuation_budget > period {
            out.push(Diagnostic::new(
                DEP_021,
                format!(
                    "deadlines.actuation_budget {} us exceeds the control period {} us",
                    d.actuation_budget.0, period.0
                ),
            ));
        }
        if d.observation_age < d.inference_budget {
            out.push(Diagnostic::new(
                DEP_021,
                format!(
                    "deadlines.observation_age {} us is below inference_budget {} us, so the \
                     staleness bound can never be met",
                    d.observation_age.0, d.inference_budget.0
                ),
            ));
        }
        let (c, i) = (self.rate.control, self.rate.inference);
        if u128::from(i.num()) * u128::from(c.den()) > u128::from(c.num()) * u128::from(i.den()) {
            out.push(Diagnostic::new(
                DEP_021,
                "rate.inference is faster than rate.control",
            ));
        }
    }

    fn check_watchdogs(&self, out: &mut Vec<Diagnostic>) {
        let d = self.deadlines;
        let mut seen = BTreeSet::new();
        for w in &self.watchdogs.0 {
            if !seen.insert(w.key()) {
                out.push(Diagnostic::new(
                    DEP_024,
                    format!("duplicate watchdog {}", w.key()),
                ));
            }
            match w {
                Watchdog::InferenceDeadline { budget } => {
                    check_within(
                        "watchdog inference_deadline",
                        *budget,
                        d.inference_budget,
                        out,
                    );
                }
                Watchdog::StaleObservation { max_age } => {
                    check_within(
                        "watchdog stale_observation",
                        *max_age,
                        d.observation_age,
                        out,
                    );
                }
                Watchdog::ControllerHeartbeat { timeout } => {
                    check_within(
                        "watchdog controller_heartbeat",
                        *timeout,
                        d.observation_age,
                        out,
                    );
                }
                Watchdog::SensorDropout { sensor, max_gap } => {
                    if sensor.is_empty() {
                        out.push(Diagnostic::new(
                            DEP_023,
                            "watchdog sensor_dropout has no sensor",
                        ));
                    }
                    check_within("watchdog sensor_dropout", *max_gap, d.observation_age, out);
                }
                Watchdog::EnvelopeViolationRate { window, max_frac } => {
                    if *window == 0 || !(0.0..=1.0).contains(max_frac) || *max_frac <= 0.0 {
                        out.push(Diagnostic::new(
                            DEP_023,
                            format!(
                                "watchdog envelope_violation_rate: window {window}, max_frac \
                                 {max_frac} must be a positive window and a fraction in (0, 1]"
                            ),
                        ));
                    }
                }
                Watchdog::ChunkUnderrun => {}
            }
        }
    }

    fn check_fallback(&self, n: usize, out: &mut Vec<Diagnostic>) {
        match &self.fallback {
            FallbackPolicy::RetractToHome { trajectory } => {
                if trajectory.is_empty() {
                    out.push(Diagnostic::new(
                        DEP_030,
                        "fallback retract_to_home has an empty trajectory",
                    ));
                }
                for (i, waypoint) in trajectory.iter().enumerate() {
                    if waypoint.len() != n {
                        out.push(Diagnostic::new(
                            DEP_010,
                            format!(
                                "fallback waypoint {i} has {} entries but robot.n_joints is {n}",
                                waypoint.len()
                            ),
                        ));
                    } else if !self
                        .safety
                        .position
                        .iter()
                        .zip(waypoint)
                        .all(|(l, q)| l.contains(*q))
                    {
                        out.push(Diagnostic::new(
                            DEP_030,
                            format!("fallback waypoint {i} is outside the position limits"),
                        ));
                    }
                }
            }
            FallbackPolicy::HandoffController { id } if id.is_empty() => {
                out.push(Diagnostic::new(
                    DEP_030,
                    "fallback handoff_controller has no controller id",
                ));
            }
            _ => {}
        }
    }

    fn check_action(&self, n: usize, out: &mut Vec<Diagnostic>) {
        let a = self.action;
        if a.horizon == 0 || a.execute_chunk == 0 || a.execute_chunk > a.horizon {
            out.push(Diagnostic::new(
                DEP_040,
                format!(
                    "action: execute_chunk {} must be in 1..={} (horizon)",
                    a.execute_chunk, a.horizon
                ),
            ));
        }
        let expected = match a.space {
            ActionSpace::JointPosition | ActionSpace::JointVelocity | ActionSpace::JointTorque => {
                Some(n)
            }
            ActionSpace::EePose | ActionSpace::EeDelta => Some(6),
            ActionSpace::Gripper | ActionSpace::Composite => None,
        };
        match expected {
            Some(dim) if a.dim != dim => out.push(Diagnostic::new(
                DEP_010,
                format!(
                    "action.dim {} does not match {dim} for {:?}",
                    a.dim, a.space
                ),
            )),
            _ if a.dim == 0 => out.push(Diagnostic::new(DEP_040, "action.dim is 0")),
            _ => {}
        }
        if let ExecutionMode::TemporalEnsemble { decay } = self.execution {
            if !decay.is_finite() || decay <= 0.0 {
                out.push(Diagnostic::new(
                    DEP_040,
                    format!("execution temporal_ensemble decay {decay} must be positive"),
                ));
            }
        }
    }

    /// `deployment_hash` of the chain (spec 5.3, spec 9.5): equal hash means equal safety
    /// behaviour.
    pub fn deployment_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let mut w = CanonWriter::new();
        w.str("es.ir.deployment.v1");
        w.u32(self.schema_version);
        w.str(&self.robot.name);
        match &self.robot.target {
            RobotTarget::Simulated { scene } => {
                w.u8(0);
                w.str(scene);
            }
            RobotTarget::Physical { driver } => {
                w.u8(1);
                w.str(driver);
            }
        }
        w.seq(self.robot.n_joints);
        w.u8(self.action.space as u8);
        w.seq(self.action.dim);
        w.seq(self.action.horizon);
        w.seq(self.action.execute_chunk);
        self.safety.canonical(&mut w);
        match self.execution {
            ExecutionMode::OpenLoopChunk => w.u8(0),
            ExecutionMode::RecedingHorizon => w.u8(1),
            ExecutionMode::TemporalEnsemble { decay } => {
                w.u8(2);
                w.f64(decay);
            }
            ExecutionMode::RealTimeChunking => w.u8(3),
        }
        for m in [
            self.deadlines.observation_age,
            self.deadlines.inference_budget,
            self.deadlines.actuation_budget,
        ] {
            w.u64(m.0);
        }
        w.seq(self.watchdogs.0.len());
        for watchdog in &self.watchdogs.0 {
            match watchdog {
                Watchdog::InferenceDeadline { budget } => {
                    w.u8(0);
                    w.u64(budget.0);
                }
                Watchdog::ChunkUnderrun => w.u8(1),
                Watchdog::StaleObservation { max_age } => {
                    w.u8(2);
                    w.u64(max_age.0);
                }
                Watchdog::EnvelopeViolationRate { window, max_frac } => {
                    w.u8(3);
                    w.u32(*window);
                    w.f64(*max_frac);
                }
                Watchdog::ControllerHeartbeat { timeout } => {
                    w.u8(4);
                    w.u64(timeout.0);
                }
                Watchdog::SensorDropout { sensor, max_gap } => {
                    w.u8(5);
                    w.str(sensor);
                    w.u64(max_gap.0);
                }
            }
        }
        match &self.fallback {
            FallbackPolicy::HoldPosition => w.u8(0),
            FallbackPolicy::ZeroVelocity => w.u8(1),
            FallbackPolicy::RetractToHome { trajectory } => {
                w.u8(2);
                w.seq(trajectory.len());
                for waypoint in trajectory {
                    canon_f64s(&mut w, waypoint);
                }
            }
            FallbackPolicy::HandoffController { id } => {
                w.u8(3);
                w.str(id);
            }
            FallbackPolicy::EmergencyStop => w.u8(4),
        }
        for rate in [self.rate.control, self.rate.inference] {
            w.u64(rate.num());
            w.u64(rate.den());
        }
        w.hash()
    }
}

impl SafetyEnvelope {
    fn canonical(&self, w: &mut CanonWriter) {
        w.seq(self.position.len());
        for limit in &self.position {
            w.f64(limit.lower);
            w.f64(limit.upper);
        }
        for v in [
            &self.position_soft_margin,
            &self.velocity_max,
            &self.acceleration_max,
            &self.torque_max,
            &self.action_rate.first_diff_max,
            &self.action_rate.second_diff_max,
        ] {
            canon_f64s(w, v);
        }
        match &self.jerk_max {
            Some(jerk) => {
                w.bool(true);
                canon_f64s(w, jerk);
            }
            None => w.bool(false),
        }
        match &self.workspace {
            Workspace::Box { min, max } => {
                w.u8(0);
                canon_f64s(w, min);
                canon_f64s(w, max);
            }
            Workspace::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => {
                w.u8(1);
                canon_f64s(w, center);
                canon_f64s(w, axis);
                w.f64(*radius);
                w.f64(*half_height);
            }
            Workspace::ConvexHull { faces } => {
                w.u8(2);
                w.seq(faces.len());
                for face in faces {
                    canon_f64s(w, &face.normal);
                    w.f64(face.offset);
                }
            }
        }
        for v in [
            self.ee_velocity_max,
            self.min_self_distance,
            self.min_env_distance,
            self.contact_force_max,
        ] {
            w.f64(v);
        }
    }
}

// --- Check helpers ------------------------------------------------------------------------

fn canon_f64s(w: &mut CanonWriter, v: &[f64]) {
    w.seq(v.len());
    for x in v {
        w.f64(*x);
    }
}

fn check_scalar(name: &str, v: f64, out: &mut Vec<Diagnostic>) {
    if !v.is_finite() {
        out.push(Diagnostic::new(DEP_011, format!("{name} is not finite")));
    } else if v <= 0.0 {
        out.push(Diagnostic::new(
            DEP_013,
            format!("{name} must be positive, got {v}"),
        ));
    }
}

fn check_positive(name: &str, v: &[f64], n: usize, out: &mut Vec<Diagnostic>) {
    if v.len() != n {
        out.push(Diagnostic::new(
            DEP_010,
            format!("{name} has {} entries but robot.n_joints is {n}", v.len()),
        ));
    }
    for (i, x) in v.iter().enumerate() {
        check_scalar(&format!("{name}[{i}]"), *x, out);
    }
}

fn check_within(name: &str, v: Micros, budget: Micros, out: &mut Vec<Diagnostic>) {
    if v.0 == 0 {
        out.push(Diagnostic::new(DEP_023, format!("{name} timeout is 0")));
    } else if v > budget {
        out.push(Diagnostic::new(
            DEP_022,
            format!(
                "{name} timeout {} us exceeds its deadline {} us",
                v.0, budget.0
            ),
        ));
    }
}

fn check_workspace(ws: &Workspace, out: &mut Vec<Diagnostic>) {
    let bad = |msg: String| Diagnostic::new(DEP_015, msg);
    match ws {
        Workspace::Box { min, max } => {
            for i in 0..3 {
                if !min[i].is_finite() || !max[i].is_finite() || min[i] >= max[i] {
                    out.push(bad(format!(
                        "workspace box axis {i} is empty or not finite"
                    )));
                }
            }
        }
        Workspace::Cylinder {
            center,
            axis,
            radius,
            half_height,
        } => {
            if !center.iter().all(|v| v.is_finite()) {
                out.push(bad("workspace cylinder center is not finite".into()));
            }
            let norm2: f64 = axis.iter().map(|v| v * v).sum();
            if !norm2.is_finite() || norm2 <= 0.0 {
                out.push(bad("workspace cylinder axis is zero or not finite".into()));
            }
            check_scalar("workspace cylinder radius", *radius, out);
            check_scalar("workspace cylinder half_height", *half_height, out);
        }
        Workspace::ConvexHull { faces } => {
            if faces.is_empty() {
                out.push(bad("workspace convex hull has no faces".into()));
            }
            for (i, face) in faces.iter().enumerate() {
                let norm2: f64 = face.normal.iter().map(|v| v * v).sum();
                if !norm2.is_finite() || norm2 <= 0.0 || !face.offset.is_finite() {
                    out.push(bad(format!("workspace hull face {i} is degenerate")));
                }
            }
        }
    }
}

// --- Generators ---------------------------------------------------------------------------

// `proptest` is a dev-dependency of this crate, so the generator is `cfg(test)` rather than
// behind a `testing` feature (adding the feature means editing `Cargo.toml`, which P23 does
// not own).
#[cfg(test)]
pub mod testing {
    use super::*;
    use proptest::prelude::*;

    const RATES_HZ: [u64; 5] = [50, 100, 200, 500, 1000];

    /// Deployment IRs that always [`DeploymentIr::validate`] clean.
    pub fn arbitrary_deployment_ir() -> impl Strategy<Value = DeploymentIr> {
        (1usize..=12, 0usize..5, 1usize..=32, 0usize..5)
            .prop_flat_map(|(n, rate_idx, horizon, variant)| {
                (
                    Just((n, rate_idx, horizon, variant)),
                    1usize..=horizon,
                    proptest::collection::vec((-10.0f64..-0.1, 0.1f64..10.0), n),
                    proptest::collection::vec(0.1f64..100.0, n * 5),
                    1u32..1000,
                )
            })
            .prop_map(
                |((n, rate_idx, horizon, variant), chunk, pos, pool, window)| {
                    let position: Vec<Limit> = pos
                        .into_iter()
                        .map(|(lower, upper)| Limit { lower, upper })
                        .collect();
                    let take = |k: usize| pool[k * n..(k + 1) * n].to_vec();
                    let hz = RATES_HZ[rate_idx];
                    let rate = RateSpec {
                        control: TickRate::hz(hz),
                        inference: TickRate::hz(hz / 10 + 1),
                    };
                    let period = rate.control_period();
                    let deadlines = Deadlines {
                        observation_age: Micros(period.0 * 4),
                        inference_budget: Micros(period.0 * 2),
                        actuation_budget: Micros(period.0 / 2 + 1),
                    };
                    DeploymentIr {
                        schema_version: SCHEMA_VERSION,
                        robot: RobotRef {
                            name: format!("robot{n}"),
                            target: if variant % 2 == 0 {
                                RobotTarget::Simulated {
                                    scene: "scene.usd".into(),
                                }
                            } else {
                                RobotTarget::Physical {
                                    driver: "can0".into(),
                                }
                            },
                            n_joints: n,
                        },
                        action: ActionContract {
                            space: ActionSpace::JointPosition,
                            dim: n,
                            horizon,
                            execute_chunk: chunk,
                        },
                        safety: SafetyEnvelope {
                            position_soft_margin: vec![0.01; n],
                            position,
                            velocity_max: take(0),
                            acceleration_max: take(1),
                            torque_max: take(2),
                            jerk_max: (variant % 2 == 0).then(|| take(3)),
                            action_rate: RateLimit {
                                first_diff_max: take(3),
                                second_diff_max: take(4),
                            },
                            workspace: workspace_for(variant),
                            ee_velocity_max: 1.5,
                            min_self_distance: 0.01,
                            min_env_distance: 0.02,
                            contact_force_max: 40.0,
                        },
                        execution: match variant % 4 {
                            0 => ExecutionMode::OpenLoopChunk,
                            1 => ExecutionMode::RecedingHorizon,
                            2 => ExecutionMode::TemporalEnsemble { decay: 0.01 },
                            _ => ExecutionMode::RealTimeChunking,
                        },
                        deadlines,
                        watchdogs: WatchdogSet(vec![
                            Watchdog::InferenceDeadline {
                                budget: deadlines.inference_budget,
                            },
                            Watchdog::ChunkUnderrun,
                            Watchdog::StaleObservation {
                                max_age: deadlines.observation_age,
                            },
                            Watchdog::EnvelopeViolationRate {
                                window,
                                max_frac: 0.05,
                            },
                        ]),
                        fallback: match variant {
                            0 => FallbackPolicy::HoldPosition,
                            1 => FallbackPolicy::ZeroVelocity,
                            2 => FallbackPolicy::RetractToHome {
                                trajectory: vec![vec![0.0; n]],
                            },
                            3 => FallbackPolicy::HandoffController { id: "pid".into() },
                            _ => FallbackPolicy::EmergencyStop,
                        },
                        rate,
                    }
                },
            )
    }

    fn workspace_for(variant: usize) -> Workspace {
        match variant % 3 {
            0 => Workspace::Box {
                min: [-1.0, -1.0, 0.0],
                max: [1.0, 1.0, 2.0],
            },
            1 => Workspace::Cylinder {
                center: [0.0, 0.0, 0.5],
                axis: [0.0, 0.0, 1.0],
                radius: 0.8,
                half_height: 0.5,
            },
            _ => Workspace::ConvexHull {
                faces: vec![HalfSpace {
                    normal: [0.0, 0.0, 1.0],
                    offset: 1.0,
                }],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::arbitrary_deployment_ir;
    use super::*;
    use proptest::prelude::*;

    /// A plausible 7-DoF arm.
    fn seven_dof() -> DeploymentIr {
        let n = 7;
        DeploymentIr {
            schema_version: SCHEMA_VERSION,
            robot: RobotRef {
                name: "panda".into(),
                target: RobotTarget::Physical {
                    driver: "fci".into(),
                },
                n_joints: n,
            },
            action: ActionContract {
                space: ActionSpace::JointPosition,
                dim: n,
                horizon: 16,
                execute_chunk: 8,
            },
            safety: SafetyEnvelope {
                position: vec![Limit::symmetric(2.8); n],
                position_soft_margin: vec![0.05; n],
                velocity_max: vec![2.0; n],
                acceleration_max: vec![10.0; n],
                torque_max: vec![87.0; n],
                jerk_max: Some(vec![1000.0; n]),
                action_rate: RateLimit {
                    first_diff_max: vec![0.02; n],
                    second_diff_max: vec![0.004; n],
                },
                workspace: Workspace::Box {
                    min: [-0.8, -0.8, 0.0],
                    max: [0.8, 0.8, 1.2],
                },
                ee_velocity_max: 1.7,
                min_self_distance: 0.01,
                min_env_distance: 0.02,
                contact_force_max: 40.0,
            },
            execution: ExecutionMode::RecedingHorizon,
            deadlines: Deadlines {
                observation_age: Micros(20_000),
                inference_budget: Micros(15_000),
                actuation_budget: Micros(800),
            },
            watchdogs: WatchdogSet(vec![
                Watchdog::InferenceDeadline {
                    budget: Micros(15_000),
                },
                Watchdog::ChunkUnderrun,
                Watchdog::StaleObservation {
                    max_age: Micros(20_000),
                },
                Watchdog::SensorDropout {
                    sensor: "wrist_cam".into(),
                    max_gap: Micros(10_000),
                },
            ]),
            fallback: FallbackPolicy::RetractToHome {
                trajectory: vec![vec![0.0; n], vec![0.1; n]],
            },
            rate: RateSpec {
                control: TickRate::hz(1000),
                inference: TickRate::hz(50),
            },
        }
    }

    /// Owned so that `codes(&ir.validate())` does not borrow a temporary.
    fn codes(diags: &[Diagnostic]) -> Vec<String> {
        diags.iter().map(|d| d.code.to_string()).collect()
    }

    fn has(diags: &[Diagnostic], code: &str) -> bool {
        diags.iter().any(|d| d.code.as_str() == code)
    }

    #[test]
    fn a_seven_dof_fixture_validates_clean() {
        assert_eq!(codes(&seven_dof().validate()), Vec::<String>::new());
    }

    #[test]
    fn serde_round_trip() {
        let ir = seven_dof();
        let json = serde_json::to_string(&ir).unwrap();
        assert_eq!(serde_json::from_str::<DeploymentIr>(&json).unwrap(), ir);
    }

    #[test]
    fn an_inverted_position_limit_is_reported() {
        let mut ir = seven_dof();
        ir.safety.position[3] = Limit {
            lower: 1.0,
            upper: -1.0,
        };
        assert!(has(&ir.validate(), DEP_012));
    }

    #[test]
    fn a_non_positive_or_non_finite_limit_is_reported() {
        let mut ir = seven_dof();
        ir.safety.velocity_max[0] = 0.0;
        ir.safety.torque_max[1] = f64::NAN;
        let diags = ir.validate();
        assert!(
            has(&diags, DEP_013) && has(&diags, DEP_011),
            "{:?}",
            codes(&diags)
        );
    }

    #[test]
    fn mismatched_joint_counts_are_reported() {
        let mut ir = seven_dof();
        ir.safety.velocity_max.pop();
        ir.robot.n_joints = 6;
        let found = codes(&ir.validate());
        assert!(
            found.iter().filter(|c| *c == DEP_010).count() >= 2,
            "{found:?}"
        );
    }

    #[test]
    fn unsatisfiable_deadlines_and_out_of_budget_watchdogs_are_reported() {
        let mut ir = seven_dof();
        ir.deadlines.actuation_budget = Micros(5_000); // > 1 ms control period
        ir.deadlines.observation_age = Micros(1_000); // < inference budget
        let found = codes(&ir.validate());
        assert!(
            found.iter().filter(|c| *c == DEP_021).count() == 2,
            "{found:?}"
        );
        // The stale-observation watchdog now sits outside its own deadline.
        assert!(found.iter().any(|c| c == DEP_022), "{found:?}");
    }

    #[test]
    fn an_unreachable_fallback_is_reported() {
        let mut ir = seven_dof();
        ir.fallback = FallbackPolicy::RetractToHome {
            trajectory: vec![vec![99.0; 7]],
        };
        assert!(has(&ir.validate(), DEP_030));
        ir.fallback = FallbackPolicy::HandoffController { id: String::new() };
        assert!(has(&ir.validate(), DEP_030));
    }

    #[test]
    fn duplicate_watchdogs_are_reported() {
        let mut ir = seven_dof();
        ir.watchdogs.0.push(Watchdog::ChunkUnderrun);
        assert!(has(&ir.validate(), DEP_024));
    }

    #[test]
    fn the_hash_changes_when_any_limit_changes() {
        let ir = seven_dof();
        let base = ir.deployment_hash().unwrap();
        assert_eq!(ir.deployment_hash().unwrap(), base);

        let mut widened = ir.clone();
        widened.safety.torque_max[6] = 87.5;
        assert_ne!(widened.deployment_hash().unwrap(), base);

        let mut other = ir.clone();
        other.safety.workspace = Workspace::Box {
            min: [-0.9, -0.8, 0.0],
            max: [0.8, 0.8, 1.2],
        };
        assert_ne!(other.deployment_hash().unwrap(), base);

        let mut mode = ir;
        mode.execution = ExecutionMode::TemporalEnsemble { decay: 0.01 };
        assert_ne!(mode.deployment_hash().unwrap(), base);
    }

    #[test]
    fn contains_rejects_out_of_range_and_non_finite_states() {
        let ir = seven_dof();
        let s = &ir.safety;
        assert_eq!(s.n_joints(), 7);
        let ok = vec![0.0; 7];
        assert!(s.contains(&ok, &ok, &ok));
        let mut nan = ok.clone();
        nan[2] = f64::NAN;
        assert!(!s.contains(&nan, &ok, &ok));
        assert!(!s.contains(&ok, &ok, &nan));
        let mut far = ok.clone();
        far[0] = 3.0;
        assert!(!s.contains(&far, &ok, &ok));
        // Widening the envelope is the only way to admit it (INV-12).
        let mut wide = ir.clone();
        wide.safety.position[0] = Limit::symmetric(4.0);
        assert!(wide.safety.contains(&far, &ok, &ok));
        assert!(!s.contains(&ok[..6], &ok, &ok));
    }

    proptest! {
        #[test]
        fn arbitrary_deployment_irs_validate_clean(ir in arbitrary_deployment_ir()) {
            prop_assert_eq!(codes(&ir.validate()), Vec::<String>::new());
            prop_assert!(ir.deployment_hash().is_ok());
        }
    }
}
