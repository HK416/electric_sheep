//! The plane's configuration in pre-allocated, fixed-size form (spec 9.2, spec 9.3).
//!
//! Two halves, split by the `std` feature:
//!
//! - **always compiled** — [`SafetyConfig`] and its parts ([`Envelope`], [`Watchdogs`],
//!   [`Fallback`]). Plain `Copy` data: arrays and integers, no `Vec`, no `String`, no heap.
//!   This is what the embedded deployment target builds a plane from (spec 9.6).
//! - **`std` only** — the `from_ir` conversions. The IR is `Vec`-shaped and serde-shaped, and
//!   turning it into the form above is the one place the two meet.
//!
//! There is still exactly one way to build a [`crate::SafetyPlane`]: `from_ir` produces a
//! `SafetyConfig` and hands it to `SafetyPlane::from_config`. Neither has a `new`, a
//! `Default` or an `enabled` flag, so an envelope-less plane cannot be written (INV-12).

use es_core::PhysTick;

use crate::ir_types::{ActionSpace, HalfSpace, Limit, Micros};
use crate::types::FallbackKind;

#[cfg(feature = "std")]
use crate::counters::WINDOW_CAP;
#[cfg(feature = "std")]
use es_ir::deployment::{DeploymentIr, FallbackPolicy, Watchdog, Workspace};

/// Faces a convex-hull workspace may have. Inline, so the hull costs no allocation (spec 9.6).
pub const MAX_HULL_FACES: usize = 16;
/// Waypoints a `RetractToHome` trajectory may have (spec 9.4).
pub const MAX_RETRACT_WAYPOINTS: usize = 32;
/// Sensors the dropout watchdog may watch (spec 9.4).
pub const MAX_SENSORS: usize = 8;
/// Bytes a watched sensor's name may take.
pub const SENSOR_NAME_CAP: usize = 32;

/// Why a `DeploymentIr` cannot become a `SafetyPlane<NJ, H>`.
#[cfg(feature = "std")]
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
    #[error("{what} has {found} entries, more than the pre-allocated {cap}")]
    TooMany {
        what: &'static str,
        found: usize,
        cap: usize,
    },
    #[error("sensor name {0:?} is longer than the pre-allocated {SENSOR_NAME_CAP} bytes")]
    SensorNameTooLong(String),
}

/// End-effector workspace (spec 9.3) with its hull faces inline.
///
/// The IR's `Workspace::ConvexHull` owns a `Vec`; this is the same shape with the faces in a
/// fixed array, because the plane must not touch the heap (spec 9.6, Appendix B.4).
// The hull variant is deliberately the large one: inlining the faces is the point.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorkspaceSpec {
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
    /// Only `faces[..n_faces]` is meaningful; the rest is padding.
    ConvexHull {
        faces: [HalfSpace; MAX_HULL_FACES],
        n_faces: usize,
    },
}

impl WorkspaceSpec {
    /// The active faces of a hull, or an empty slice for the other shapes.
    pub fn faces(&self) -> &[HalfSpace] {
        match self {
            Self::ConvexHull { faces, n_faces } => &faces[..(*n_faces).min(MAX_HULL_FACES)],
            _ => &[],
        }
    }
}

/// The spec 9.3 constraints in fixed-size form.
///
/// Every field is mandatory. There is no switch, no `Option` other than `jerk_max` (which the
/// spec itself marks optional and which means "no jerk bound", never "no bounds").
#[derive(Clone, Copy, Debug)]
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
    pub workspace: WorkspaceSpec,
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

/// One watched sensor (spec 9.4 `SensorDropout`). The name is inline bytes, not a `String`.
#[derive(Clone, Copy, Debug)]
pub struct SensorWatch {
    name: [u8; SENSOR_NAME_CAP],
    name_len: usize,
    pub max_gap: Micros,
    pub last_seen: PhysTick,
}

impl SensorWatch {
    /// An unused slot: a zero-length name matches nothing.
    pub const EMPTY: Self = Self {
        name: [0; SENSOR_NAME_CAP],
        name_len: 0,
        max_gap: Micros(0),
        last_seen: PhysTick::ZERO,
    };

    /// `None` if the name does not fit [`SENSOR_NAME_CAP`]. Never truncates: two sensors whose
    /// names share a prefix must not collapse into one watchdog.
    pub fn new(name: &str, max_gap: Micros) -> Option<Self> {
        let bytes = name.as_bytes();
        if bytes.is_empty() || bytes.len() > SENSOR_NAME_CAP {
            return None;
        }
        let mut buf = [0u8; SENSOR_NAME_CAP];
        buf[..bytes.len()].copy_from_slice(bytes);
        Some(Self {
            name: buf,
            name_len: bytes.len(),
            max_gap,
            last_seen: PhysTick::ZERO,
        })
    }

    pub fn name(&self) -> &str {
        core::str::from_utf8(&self.name[..self.name_len]).unwrap_or("")
    }

    /// Byte-exact name match. An empty slot matches nothing.
    pub fn matches(&self, name: &str) -> bool {
        self.name_len != 0 && self.name[..self.name_len] == *name.as_bytes()
    }
}

/// The armed watchdogs (spec 9.4). `ChunkUnderrun` and non-finite rejection are not here:
/// they are unconditional in the plane and cannot be left out of a configuration (INV-12).
#[derive(Clone, Copy, Debug)]
pub struct Watchdogs {
    pub inference_budget: Option<Micros>,
    pub max_obs_age: Option<Micros>,
    pub heartbeat_timeout: Option<Micros>,
    pub violation_rate: Option<(usize, f64)>,
    /// Only `sensors[..n_sensors]` is armed. Scanned linearly: no `HashMap`, so iteration
    /// order is construction order (spec 3.4).
    sensors: [SensorWatch; MAX_SENSORS],
    n_sensors: usize,
}

impl Default for Watchdogs {
    fn default() -> Self {
        Self {
            inference_budget: None,
            max_obs_age: None,
            heartbeat_timeout: None,
            violation_rate: None,
            sensors: [SensorWatch::EMPTY; MAX_SENSORS],
            n_sensors: 0,
        }
    }
}

impl Watchdogs {
    pub fn sensors(&self) -> &[SensorWatch] {
        &self.sensors[..self.n_sensors.min(MAX_SENSORS)]
    }

    pub(crate) fn sensors_mut(&mut self) -> &mut [SensorWatch] {
        let n = self.n_sensors.min(MAX_SENSORS);
        &mut self.sensors[..n]
    }

    /// Arms one more sensor. `false` if the name is already watched, does not fit, or the
    /// pre-allocated table is full.
    pub fn arm_sensor(&mut self, name: &str, max_gap: Micros) -> bool {
        if self.n_sensors >= MAX_SENSORS || self.sensors().iter().any(|s| s.matches(name)) {
            return false;
        }
        match SensorWatch::new(name, max_gap) {
            Some(w) => {
                self.sensors[self.n_sensors] = w;
                self.n_sensors += 1;
                true
            }
            None => false,
        }
    }

    /// Ring length for the counters: the configured window, or a default sampling window.
    pub(crate) fn window_len(&self) -> usize {
        self.violation_rate.map_or(64, |(w, _)| w)
    }
}

/// The fallback, with its trajectory already in fixed-width rows (spec 9.4).
#[derive(Clone, Copy, Debug)]
pub struct Fallback<const NJ: usize> {
    pub kind: FallbackKind,
    trajectory: [[f64; NJ]; MAX_RETRACT_WAYPOINTS],
    n_waypoints: usize,
}

impl<const NJ: usize> Fallback<NJ> {
    /// A fallback that holds, zeroes velocity, hands off or latches: no trajectory.
    ///
    /// `FallbackKind::RetractToHome` is not constructible this way — it would have nowhere to
    /// retract to — and degrades to `HoldPosition`, which is safe, not disabled (INV-12).
    pub fn stationary(kind: FallbackKind) -> Self {
        let kind = if kind == FallbackKind::RetractToHome {
            FallbackKind::HoldPosition
        } else {
            kind
        };
        Self {
            kind,
            trajectory: [[0.0; NJ]; MAX_RETRACT_WAYPOINTS],
            n_waypoints: 0,
        }
    }

    /// `RetractToHome` along `waypoints`. `None` if it is empty or longer than
    /// [`MAX_RETRACT_WAYPOINTS`].
    pub fn retract(waypoints: &[[f64; NJ]]) -> Option<Self> {
        if waypoints.is_empty() || waypoints.len() > MAX_RETRACT_WAYPOINTS {
            return None;
        }
        let mut trajectory = [[0.0; NJ]; MAX_RETRACT_WAYPOINTS];
        trajectory[..waypoints.len()].copy_from_slice(waypoints);
        Some(Self {
            kind: FallbackKind::RetractToHome,
            trajectory,
            n_waypoints: waypoints.len(),
        })
    }

    pub fn trajectory(&self) -> &[[f64; NJ]] {
        &self.trajectory[..self.n_waypoints.min(MAX_RETRACT_WAYPOINTS)]
    }
}

/// Everything a [`crate::SafetyPlane`] is built from: plain `Copy` data, no heap, no `es-ir`.
///
/// This is the `no_std` construction path of spec 9.6 and the *only* construction path there
/// is — `SafetyPlane::from_ir` builds one of these and delegates, so a bundle-loaded plane and
/// a hand-written embedded plane run identical code (spec 9.5).
#[derive(Clone, Copy, Debug)]
pub struct SafetyConfig<const NJ: usize> {
    pub envelope: Envelope<NJ>,
    pub watchdogs: Watchdogs,
    pub fallback: Fallback<NJ>,
}

impl<const NJ: usize> Envelope<NJ> {
    /// Repairs a hand-written envelope so the clamp stages cannot be skipped: a soft limit
    /// that is not a finite sub-interval of its hard limit becomes the hard limit, and a
    /// non-finite bound becomes zero (the tightest value, never the loosest).
    ///
    /// This narrows, never widens — the spec 9.1 way to make room is a wider *hard* limit
    /// (INV-12).
    pub(crate) fn sanitize(&mut self) {
        for i in 0..NJ {
            let h = self.hard[i];
            if !h.lower.is_finite() || !h.upper.is_finite() || h.lower >= h.upper {
                self.hard[i] = Limit {
                    lower: 0.0,
                    upper: 0.0,
                };
            }
            let h = self.hard[i];
            let s = self.soft[i];
            if !s.lower.is_finite()
                || !s.upper.is_finite()
                || s.lower < h.lower
                || s.upper > h.upper
            {
                self.soft[i] = h;
            }
            if self.soft[i].lower > self.soft[i].upper {
                self.soft[i] = h;
            }
            for v in [
                &mut self.vel_max[i],
                &mut self.acc_max[i],
                &mut self.tau_max[i],
                &mut self.d1_max[i],
                &mut self.d2_max[i],
            ] {
                if !v.is_finite() || *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        if !self.dt_s.is_finite() || self.dt_s <= 0.0 {
            self.dt_s = 0.001;
        }
        self.period_us = self.period_us.max(1);
        self.execute_chunk = self.execute_chunk.max(1);
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
            WorkspaceSpec::Box { min, max } => {
                for i in 0..3 {
                    p[i] = p[i].clamp(min[i], max[i]);
                }
            }
            WorkspaceSpec::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => project_cylinder(p, center, axis, *radius, *half_height),
            WorkspaceSpec::ConvexHull { .. } => project_hull(p, self.workspace.faces()),
        }
        p.iter()
            .zip(before)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    }
}

/// Correctly-rounded square root, from `libm` on every target.
///
/// `f64::sqrt` lives in `std`, so it is unavailable on the embedded target; taking `libm`'s on
/// *both* sides keeps one code path rather than two that happen to agree (spec 9.5). IEEE-754
/// square root is correctly rounded, so this is bit-identical to the hardware instruction
/// `std` would have used (spec 3.5 tier 1).
fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

fn norm3(v: [f64; 3]) -> f64 {
    sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
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

// --- Deployment IR -> SafetyConfig (std only) ----------------------------------------------

#[cfg(feature = "std")]
fn finite(v: f64, field: &'static str) -> Result<f64, SafetyConfigError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(SafetyConfigError::NonFinite { field })
    }
}

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
impl WorkspaceSpec {
    pub(crate) fn from_ir(ws: &Workspace) -> Result<Self, SafetyConfigError> {
        Ok(match ws {
            Workspace::Box { min, max } => Self::Box {
                min: *min,
                max: *max,
            },
            Workspace::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => Self::Cylinder {
                center: *center,
                axis: *axis,
                radius: *radius,
                half_height: *half_height,
            },
            Workspace::ConvexHull { faces: src } => {
                if src.len() > MAX_HULL_FACES {
                    return Err(SafetyConfigError::TooMany {
                        what: "safety.workspace.convex_hull.faces",
                        found: src.len(),
                        cap: MAX_HULL_FACES,
                    });
                }
                let mut faces = [HalfSpace {
                    normal: [0.0; 3],
                    offset: 0.0,
                }; MAX_HULL_FACES];
                faces[..src.len()].copy_from_slice(src);
                Self::ConvexHull {
                    faces,
                    n_faces: src.len(),
                }
            }
        })
    }
}

#[cfg(feature = "std")]
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
            workspace: WorkspaceSpec::from_ir(&s.workspace)?,
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
}

#[cfg(feature = "std")]
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
                    if w.sensors().iter().any(|s| s.matches(sensor)) {
                        return Err(SafetyConfigError::DuplicateWatchdog("sensor_dropout"));
                    }
                    if sensor.len() > SENSOR_NAME_CAP || sensor.is_empty() {
                        return Err(SafetyConfigError::SensorNameTooLong(sensor.clone()));
                    }
                    if !w.arm_sensor(sensor, *max_gap) {
                        return Err(SafetyConfigError::TooMany {
                            what: "watchdogs.sensor_dropout",
                            found: w.sensors().len() + 1,
                            cap: MAX_SENSORS,
                        });
                    }
                }
                // Unconditional in the plane; listing it changes nothing (INV-12).
                Watchdog::ChunkUnderrun => {}
            }
        }
        Ok(w)
    }
}

#[cfg(feature = "std")]
impl<const NJ: usize> Fallback<NJ> {
    pub(crate) fn from_ir(ir: &DeploymentIr) -> Result<Self, SafetyConfigError> {
        Ok(match &ir.fallback {
            FallbackPolicy::HoldPosition => Self::stationary(FallbackKind::HoldPosition),
            FallbackPolicy::ZeroVelocity => Self::stationary(FallbackKind::ZeroVelocity),
            FallbackPolicy::HandoffController { .. } => {
                Self::stationary(FallbackKind::HandoffController)
            }
            FallbackPolicy::EmergencyStop => Self::stationary(FallbackKind::EmergencyStop),
            FallbackPolicy::RetractToHome { trajectory } => {
                if trajectory.is_empty() {
                    return Err(SafetyConfigError::InvalidIr(
                        "fallback retract_to_home has an empty trajectory".into(),
                    ));
                }
                if trajectory.len() > MAX_RETRACT_WAYPOINTS {
                    return Err(SafetyConfigError::TooMany {
                        what: "fallback.retract_to_home.trajectory",
                        found: trajectory.len(),
                        cap: MAX_RETRACT_WAYPOINTS,
                    });
                }
                let mut rows = [[0.0; NJ]; MAX_RETRACT_WAYPOINTS];
                for (out, wp) in rows.iter_mut().zip(trajectory) {
                    *out = row::<NJ>(wp, "fallback.trajectory")?;
                }
                Self::retract(&rows[..trajectory.len()])
                    .expect("length checked against MAX_RETRACT_WAYPOINTS above")
            }
        })
    }
}

// `std` because the test harness itself needs it; the code under test does not.
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn sensor_names_are_exact_not_truncated() {
        let long = "s".repeat(SENSOR_NAME_CAP + 1);
        assert!(SensorWatch::new(&long, Micros(1)).is_none());
        assert!(SensorWatch::new("", Micros(1)).is_none());
        let w = SensorWatch::new("wrist_cam", Micros(5)).unwrap();
        assert_eq!(w.name(), "wrist_cam");
        assert!(w.matches("wrist_cam"));
        assert!(!w.matches("wrist_ca"));
    }

    #[test]
    fn sensor_table_is_bounded_and_rejects_duplicates() {
        let mut w = Watchdogs::default();
        for i in 0..MAX_SENSORS {
            assert!(w.arm_sensor(&format!("s{i}"), Micros(1)));
        }
        assert!(!w.arm_sensor("s0", Micros(1)), "duplicate");
        assert!(!w.arm_sensor("overflow", Micros(1)), "table full");
        assert_eq!(w.sensors().len(), MAX_SENSORS);
    }

    #[test]
    fn retract_needs_waypoints_and_stationary_cannot_claim_to_retract() {
        assert!(Fallback::<3>::retract(&[]).is_none());
        assert!(Fallback::<3>::retract(&[[0.0; 3]; MAX_RETRACT_WAYPOINTS + 1]).is_none());
        let f = Fallback::<3>::retract(&[[1.0; 3]]).unwrap();
        assert_eq!(f.kind, FallbackKind::RetractToHome);
        assert_eq!(f.trajectory(), &[[1.0; 3]]);
        // A trajectory-less retract would index an empty slice; it degrades to hold instead.
        assert_eq!(
            Fallback::<3>::stationary(FallbackKind::RetractToHome).kind,
            FallbackKind::HoldPosition
        );
    }
}
