//! Turning the Deployment IR into the runtime's pre-allocated form (spec 9.2, spec 9.3).
//!
//! The IR is `Vec`-shaped and serde-shaped; the runtime is array-shaped. This module is the
//! one place the two meet, and it is the only way to build a [`crate::SafetyPlane`]: there is
//! no `new`, no `Default`, and no public field, so an envelope-less plane cannot be written
//! (INV-12).

use es_ir::deployment::{
    ActionSpace, DeploymentIr, FallbackPolicy, HalfSpace, Limit, Micros, Watchdog, Workspace,
};

use crate::counters::WINDOW_CAP;
use crate::types::FallbackKind;

/// Why a `DeploymentIr` cannot become a `SafetyPlane<NJ, H>`.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SafetyConfigError {
    #[error("deployment IR is invalid: {0}")]
    InvalidIr(String),
    #[error("robot.n_joints {n_joints} and action.dim {dim} must both equal NJ = {expected}")]
    JointCount {
        n_joints: usize,
        dim: usize,
        expected: usize,
    },
    #[error("action.horizon {horizon} must equal H = {expected}")]
    Horizon { horizon: usize, expected: usize },
    #[error("{field} is not finite")]
    NonFinite { field: &'static str },
    #[error("envelope_violation_rate window {window} exceeds the pre-allocated cap {WINDOW_CAP}")]
    WindowTooLarge { window: u32 },
    #[error("duplicate watchdog: {0}")]
    DuplicateWatchdog(&'static str),
}

/// The spec 9.3 constraints in fixed-size form.
///
/// Every field is mandatory. There is no switch, no `Option` other than `jerk_max` (which the
/// spec itself marks optional and which means "no jerk bound", never "no bounds").
#[derive(Clone, Debug)]
pub struct Envelope<const NJ: usize> {
    /// Hard joint limits (spec 9.3 `position_limit`).
    pub hard: [Limit; NJ],
    /// Hard limits shrunk by the soft margin: what the clamp actually targets.
    pub soft: [Limit; NJ],
    pub vel_max: [f64; NJ],
    pub acc_max: [f64; NJ],
    pub tau_max: [f64; NJ],
    pub jerk_max: Option<[f64; NJ]>,
    /// Action first/second difference bounds per tick (spec 9.3 `rate_limit`).
    pub d1_max: [f64; NJ],
    pub d2_max: [f64; NJ],
    /// End-effector workspace (spec 9.3). Sized once; only indexed afterwards.
    pub workspace: Workspace,
    /// Carried for `deployment_hash` (spec 9.5); enforced where contact state exists.
    pub ee_velocity_max: f64,
    pub min_self_distance: f64,
    pub min_env_distance: f64,
    pub contact_force_max: f64,
    /// Which space the action lives in: decides whether the torque and workspace stages apply.
    pub space: ActionSpace,
    /// Control period in seconds. A constant derived once from the rational tick rate; never
    /// accumulated (spec 3.4).
    pub dt_s: f64,
    /// Control period in whole microseconds, for integer watchdog arithmetic.
    pub period_us: u64,
    /// Executed chunk length K (spec 8.5).
    pub execute_chunk: usize,
}

/// The armed watchdogs (spec 9.4). `ChunkUnderrun` and non-finite rejection are not here:
/// they are unconditional in the plane and cannot be left out of a configuration (INV-12).
#[derive(Clone, Debug, Default)]
pub struct Watchdogs {
    pub inference_budget: Option<Micros>,
    pub max_obs_age: Option<Micros>,
    pub heartbeat_timeout: Option<Micros>,
    pub violation_rate: Option<(usize, f64)>,
    /// One entry per configured sensor. A `Vec` sized in `from_ir` and scanned linearly:
    /// no `HashMap`, so iteration order is construction order (spec 3.4).
    pub sensors: Vec<SensorWatch>,
}

#[derive(Clone, Debug)]
pub struct SensorWatch {
    pub name: String,
    pub max_gap: Micros,
    pub last_seen: es_core::PhysTick,
}

/// The fallback, with its trajectory already in fixed-width rows (spec 9.4).
#[derive(Clone, Debug)]
pub struct Fallback<const NJ: usize> {
    pub kind: FallbackKind,
    /// Non-empty only for `RetractToHome`. Sized once in `from_ir`.
    pub trajectory: Vec<[f64; NJ]>,
}

fn finite(v: f64, field: &'static str) -> Result<f64, SafetyConfigError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(SafetyConfigError::NonFinite { field })
    }
}

fn row<const NJ: usize>(src: &[f64], field: &'static str) -> Result<[f64; NJ], SafetyConfigError> {
    let mut out = [0.0; NJ];
    if src.len() != NJ {
        return Err(SafetyConfigError::InvalidIr(format!(
            "{field} has {} entries, expected {NJ}",
            src.len()
        )));
    }
    for (o, v) in out.iter_mut().zip(src) {
        *o = finite(*v, field)?;
    }
    Ok(out)
}

impl<const NJ: usize> Envelope<NJ> {
    pub(crate) fn from_ir(ir: &DeploymentIr) -> Result<Self, SafetyConfigError> {
        let s = &ir.safety;
        if s.position.len() != NJ || s.position_soft_margin.len() != NJ {
            return Err(SafetyConfigError::InvalidIr(format!(
                "safety.position has {} entries, expected {NJ}",
                s.position.len()
            )));
        }
        let mut hard = [Limit {
            lower: 0.0,
            upper: 0.0,
        }; NJ];
        let mut soft = hard;
        for i in 0..NJ {
            let l = s.position[i];
            let m = finite(s.position_soft_margin[i], "safety.position_soft_margin")?;
            hard[i] = Limit {
                lower: finite(l.lower, "safety.position.lower")?,
                upper: finite(l.upper, "safety.position.upper")?,
            };
            if hard[i].lower >= hard[i].upper {
                return Err(SafetyConfigError::InvalidIr(format!(
                    "safety.position[{i}] is empty"
                )));
            }
            soft[i] = Limit {
                lower: hard[i].lower + m,
                upper: hard[i].upper - m,
            };
            if soft[i].lower >= soft[i].upper {
                return Err(SafetyConfigError::InvalidIr(format!(
                    "safety.position_soft_margin[{i}] leaves no room"
                )));
            }
        }
        let jerk_max = match &s.jerk_max {
            Some(j) => Some(row::<NJ>(j, "safety.jerk_max")?),
            None => None,
        };
        // A rational rate, so the period is exact and the float below is a constant, not an
        // accumulator (spec 3.4).
        let period_us = ir.rate.control_period().0.max(1);
        Ok(Self {
            hard,
            soft,
            vel_max: row(&s.velocity_max, "safety.velocity_max")?,
            acc_max: row(&s.acceleration_max, "safety.acceleration_max")?,
            tau_max: row(&s.torque_max, "safety.torque_max")?,
            jerk_max,
            d1_max: row(&s.action_rate.first_diff_max, "safety.action_rate.first")?,
            d2_max: row(&s.action_rate.second_diff_max, "safety.action_rate.second")?,
            workspace: s.workspace.clone(),
            ee_velocity_max: finite(s.ee_velocity_max, "safety.ee_velocity_max")?,
            min_self_distance: finite(s.min_self_distance, "safety.min_self_distance")?,
            min_env_distance: finite(s.min_env_distance, "safety.min_env_distance")?,
            contact_force_max: finite(s.contact_force_max, "safety.contact_force_max")?,
            space: ir.action.space,
            dt_s: ir.rate.control.period_secs_f64(),
            period_us,
            execute_chunk: ir.action.execute_chunk.max(1),
        })
    }

    /// Whether the workspace stage applies: only end-effector spaces, where components
    /// `[0..3]` are a Cartesian position. There is no forward kinematics in this crate, so a
    /// joint-space action is not projected (see docs/design/safety-plane.md).
    pub(crate) fn workspace_applies(&self) -> bool {
        NJ >= 3 && matches!(self.space, ActionSpace::EePose | ActionSpace::EeDelta)
    }

    /// Projects `p` into the workspace, returning whether it moved.
    pub(crate) fn project_workspace(&self, p: &mut [f64; 3]) -> bool {
        let before = *p;
        match &self.workspace {
            Workspace::Box { min, max } => {
                for i in 0..3 {
                    p[i] = p[i].clamp(min[i], max[i]);
                }
            }
            Workspace::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => project_cylinder(p, center, axis, *radius, *half_height),
            Workspace::ConvexHull { faces } => project_hull(p, faces),
        }
        p.iter()
            .zip(before)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    }
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn project_cylinder(p: &mut [f64; 3], center: &[f64; 3], axis: &[f64; 3], radius: f64, hh: f64) {
    let norm = norm3(*axis);
    if norm.is_nan() || norm <= 0.0 || !radius.is_finite() || !hh.is_finite() {
        return;
    }
    let unit = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let rel = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    let along = (rel[0] * unit[0] + rel[1] * unit[1] + rel[2] * unit[2]).clamp(-hh, hh);
    let mut radial = [
        rel[0] - along * unit[0],
        rel[1] - along * unit[1],
        rel[2] - along * unit[2],
    ];
    let rn = norm3(radial);
    if rn > radius && rn > 0.0 {
        let scale = radius / rn;
        radial = [radial[0] * scale, radial[1] * scale, radial[2] * scale];
    }
    for (i, out) in p.iter_mut().enumerate() {
        *out = center[i] + along * unit[i] + radial[i];
    }
}

/// Successive projection onto the violated half-spaces. Exact for a single violated face and
/// convergent for several; the sweep count is fixed so the result is deterministic and the
/// runtime is bounded (spec 3.4).
fn project_hull(p: &mut [f64; 3], faces: &[HalfSpace]) {
    const SWEEPS: usize = 8;
    for _ in 0..SWEEPS {
        let mut moved = false;
        for f in faces {
            let n2 =
                f.normal[0] * f.normal[0] + f.normal[1] * f.normal[1] + f.normal[2] * f.normal[2];
            if n2.is_nan() || n2 <= 0.0 {
                continue;
            }
            let depth = f.normal[0] * p[0] + f.normal[1] * p[1] + f.normal[2] * p[2] - f.offset;
            if depth > 0.0 {
                let k = depth / n2;
                for (i, out) in p.iter_mut().enumerate() {
                    *out -= k * f.normal[i];
                }
                moved = true;
            }
        }
        if !moved {
            return;
        }
    }
}

impl Watchdogs {
    pub(crate) fn from_ir(ir: &DeploymentIr) -> Result<Self, SafetyConfigError> {
        let mut w = Self::default();
        for entry in &ir.watchdogs.0 {
            match entry {
                Watchdog::InferenceDeadline { budget } => {
                    if w.inference_budget.replace(*budget).is_some() {
                        return Err(SafetyConfigError::DuplicateWatchdog("inference_deadline"));
                    }
                }
                Watchdog::StaleObservation { max_age } => {
                    if w.max_obs_age.replace(*max_age).is_some() {
                        return Err(SafetyConfigError::DuplicateWatchdog("stale_observation"));
                    }
                }
                Watchdog::ControllerHeartbeat { timeout } => {
                    if w.heartbeat_timeout.replace(*timeout).is_some() {
                        return Err(SafetyConfigError::DuplicateWatchdog("controller_heartbeat"));
                    }
                }
                Watchdog::EnvelopeViolationRate { window, max_frac } => {
                    if *window as usize > WINDOW_CAP {
                        return Err(SafetyConfigError::WindowTooLarge { window: *window });
                    }
                    let frac = finite(*max_frac, "watchdog.max_frac")?;
                    if w.violation_rate.replace((*window as usize, frac)).is_some() {
                        return Err(SafetyConfigError::DuplicateWatchdog(
                            "envelope_violation_rate",
                        ));
                    }
                }
                Watchdog::SensorDropout { sensor, max_gap } => {
                    if w.sensors.iter().any(|s| s.name == *sensor) {
                        return Err(SafetyConfigError::DuplicateWatchdog("sensor_dropout"));
                    }
                    w.sensors.push(SensorWatch {
                        name: sensor.clone(),
                        max_gap: *max_gap,
                        last_seen: es_core::PhysTick::ZERO,
                    });
                }
                // Unconditional in the plane; listing it changes nothing (INV-12).
                Watchdog::ChunkUnderrun => {}
            }
        }
        Ok(w)
    }

    /// Ring length for the counters: the configured window, or a default sampling window.
    pub(crate) fn window_len(&self) -> usize {
        self.violation_rate.map_or(64, |(w, _)| w)
    }
}

impl<const NJ: usize> Fallback<NJ> {
    pub(crate) fn from_ir(ir: &DeploymentIr) -> Result<Self, SafetyConfigError> {
        let (kind, trajectory) = match &ir.fallback {
            FallbackPolicy::HoldPosition => (FallbackKind::HoldPosition, Vec::new()),
            FallbackPolicy::ZeroVelocity => (FallbackKind::ZeroVelocity, Vec::new()),
            FallbackPolicy::HandoffController { .. } => {
                (FallbackKind::HandoffController, Vec::new())
            }
            FallbackPolicy::EmergencyStop => (FallbackKind::EmergencyStop, Vec::new()),
            FallbackPolicy::RetractToHome { trajectory } => {
                if trajectory.is_empty() {
                    return Err(SafetyConfigError::InvalidIr(
                        "fallback retract_to_home has an empty trajectory".into(),
                    ));
                }
                let mut rows = Vec::with_capacity(trajectory.len());
                for wp in trajectory {
                    rows.push(row::<NJ>(wp, "fallback.trajectory")?);
                }
                (FallbackKind::RetractToHome, rows)
            }
        };
        Ok(Self { kind, trajectory })
    }
}
