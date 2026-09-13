//! Actuator transduction (spec 18.2): the force law from `(ctrl, q, qd)` to generalized force,
//! range clamping, a fixed transport delay and a small backlash model. Pure `f64` math, no
//! transcendentals, so nothing here needs `es_math::approx`.

use std::collections::VecDeque;

use thiserror::Error;

/// Force law from control input `ctrl` and joint state `(q, qd)` (position, velocity) to
/// generalized force. Field names mirror `crates/es-assets::scene::ActuatorKind` (mirrored
/// rather than imported — see the crate-level docs) but `gear` is folded into each variant
/// here instead of living on a wrapping `Actuator` struct, since this crate has no such
/// wrapper.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ActuatorModel {
    /// Direct force/torque: `force = gear * ctrl`.
    Motor { gear: f64 },
    /// Position servo (`MuJoCo` PD convention): `force = gear * (kp * (ctrl - q) - kd * qd)`.
    Position { kp: f64, kd: f64, gear: f64 },
    /// Velocity servo: `force = gear * kv * (ctrl - qd)`.
    Velocity { kv: f64, gear: f64 },
    /// `MuJoCo` `gaintype=affine, biastype=affine` general actuator (unverified against
    /// upstream `MuJoCo` source; deduced from the `es-assets` doc comment on
    /// `ActuatorKind::General`): `force = gear * ((gain . [1, q, qd]) * ctrl + (bias . [1, q,
    /// qd]))`, i.e. `gain`/`bias` are dotted with `[1, q, qd]` before combining with `ctrl`.
    General {
        gain: [f64; 3],
        bias: [f64; 3],
        gear: f64,
    },
}

impl ActuatorModel {
    /// Generalized force for control `ctrl` and joint state `(q, qd)`.
    #[must_use]
    pub fn force(&self, ctrl: f64, q: f64, qd: f64) -> f64 {
        match *self {
            Self::Motor { gear } => gear * ctrl,
            Self::Position { kp, kd, gear } => gear * (kp * (ctrl - q) - kd * qd),
            Self::Velocity { kv, gear } => gear * kv * (ctrl - qd),
            Self::General { gain, bias, gear } => {
                let g = gain[0] + gain[1] * q + gain[2] * qd;
                let b = bias[0] + bias[1] * q + bias[2] * qd;
                gear * (g * ctrl + b)
            }
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ActuatorError {
    #[error("range lo ({lo}) must be <= hi ({hi})")]
    InvalidRange { lo: String, hi: String },
    #[error("deadband must be >= 0 (got {0})")]
    NegativeDeadband(String),
}

/// A clamp to `[lo, hi]`, used for both `ctrl_range` and `force_range` (spec 18.2). `None`
/// means unbounded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Saturation {
    pub range: Option<(f64, f64)>,
}

impl Saturation {
    /// # Errors
    /// [`ActuatorError::InvalidRange`] if `lo > hi`.
    pub fn new(range: Option<(f64, f64)>) -> Result<Self, ActuatorError> {
        if let Some((lo, hi)) = range {
            if lo > hi {
                return Err(ActuatorError::InvalidRange {
                    lo: lo.to_string(),
                    hi: hi.to_string(),
                });
            }
        }
        Ok(Self { range })
    }

    #[must_use]
    pub fn unbounded() -> Self {
        Self { range: None }
    }

    #[must_use]
    pub fn apply(&self, x: f64) -> f64 {
        self.range.map_or(x, |(lo, hi)| x.clamp(lo, hi))
    }
}

/// Fixed transport delay of `ticks` physics ticks over a scalar signal (ctrl or force), via a
/// pre-allocated ring. Pre-fill (before the ring has seen `ticks` samples) reads back as zero.
#[derive(Clone, Debug)]
pub struct ActuatorDelay {
    ticks: u32,
    ring: VecDeque<f64>,
}

impl ActuatorDelay {
    #[must_use]
    pub fn new(ticks: u32) -> Self {
        Self {
            ticks,
            ring: VecDeque::new(),
        }
    }

    /// Pushes `input`, returns the value from `ticks` steps ago (or `0.0` during pre-fill).
    pub fn step(&mut self, input: f64) -> f64 {
        if self.ticks == 0 {
            return input;
        }
        if self.ring.is_empty() {
            self.ring
                .extend(std::iter::repeat_n(0.0, self.ticks as usize));
        }
        self.ring.push_back(input);
        self.ring
            .pop_front()
            .expect("ring pre-filled to `ticks` above")
    }
}

/// Dead-zone backlash (spec 18.2, gear and backlash realism): output tracks
/// input only once input moves more than `deadband / 2` away from the current output,
/// otherwise it holds. This is the minimal mechanical-play model (a single dead zone, no
/// separate loaded/unloaded stiffness); upgrade to a two-mass model if that realism gap
/// matters later (spec 18.2 also lists joint elasticity (two-mass) as a further-out item).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Backlash {
    pub deadband: f64,
    output: f64,
}

impl Backlash {
    /// # Errors
    /// [`ActuatorError::NegativeDeadband`] if `deadband < 0.0`.
    pub fn new(deadband: f64) -> Result<Self, ActuatorError> {
        if deadband < 0.0 {
            return Err(ActuatorError::NegativeDeadband(deadband.to_string()));
        }
        Ok(Self {
            deadband,
            output: 0.0,
        })
    }

    #[must_use]
    pub fn output(&self) -> f64 {
        self.output
    }

    pub fn step(&mut self, input: f64) -> f64 {
        let half = self.deadband * 0.5;
        if input > self.output + half {
            self.output = input - half;
        } else if input < self.output - half {
            self.output = input + half;
        }
        self.output
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn motor_force_is_gear_times_ctrl() {
        let m = ActuatorModel::Motor { gear: 2.0 };
        assert_eq!(m.force(3.0, 999.0, 999.0), 6.0);
    }

    #[test]
    fn position_force_hand_computed() {
        let m = ActuatorModel::Position {
            kp: 10.0,
            kd: 0.5,
            gear: 1.0,
        };
        // force = 1 * (10 * (1.0 - 0.2) - 0.5 * 3.0) = 8.0 - 1.5 = 6.5
        assert_eq!(m.force(1.0, 0.2, 3.0), 6.5);
    }

    #[test]
    fn velocity_force_hand_computed() {
        let m = ActuatorModel::Velocity { kv: 4.0, gear: 2.0 };
        // force = 2 * 4 * (1.5 - 0.5) = 8.0
        assert_eq!(m.force(1.5, 0.0, 0.5), 8.0);
    }

    #[test]
    fn general_force_hand_computed() {
        let m = ActuatorModel::General {
            gain: [2.0, 0.0, 0.0],
            bias: [1.0, 0.5, -0.25],
            gear: 1.0,
        };
        // g = 2.0, b = 1.0 + 0.5*q - 0.25*qd = 1.0 + 0.5*2.0 - 0.25*4.0 = 1.0
        // force = 1 * (2.0 * ctrl + 1.0) = 2*3.0 + 1.0 = 7.0
        assert_eq!(m.force(3.0, 2.0, 4.0), 7.0);
    }

    #[test]
    fn saturation_clamps_and_rejects_invalid_range() {
        let s = Saturation::new(Some((-1.0, 1.0))).unwrap();
        assert_eq!(s.apply(5.0), 1.0);
        assert_eq!(s.apply(-5.0), -1.0);
        assert_eq!(s.apply(0.3), 0.3);
        assert_eq!(Saturation::unbounded().apply(1e9), 1e9);
        assert!(Saturation::new(Some((1.0, -1.0))).is_err());
    }

    #[test]
    fn backlash_rejects_negative_deadband() {
        assert!(Backlash::new(-0.1).is_err());
        assert!(Backlash::new(0.0).is_ok());
    }

    #[test]
    fn backlash_holds_within_deadband_then_tracks() {
        let mut b = Backlash::new(1.0).unwrap();
        assert_eq!(b.step(0.0), 0.0);
        // Within +-0.5 of output (0.0): holds.
        assert_eq!(b.step(0.4), 0.0);
        assert_eq!(b.step(-0.4), 0.0);
        // Past the dead zone: tracks input minus the half-band.
        assert_eq!(b.step(1.0), 0.5);
        assert_eq!(b.output(), 0.5);
    }

    #[test]
    fn delay_line_shifts_by_exactly_n_ticks_with_zero_prefill() {
        let mut d = ActuatorDelay::new(3);
        let outputs: Vec<f64> = (0..8).map(|t| d.step(f64::from(t))).collect();
        assert_eq!(outputs, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn delay_line_zero_ticks_is_passthrough() {
        let mut d = ActuatorDelay::new(0);
        assert_eq!(d.step(1.0), 1.0);
        assert_eq!(d.step(2.0), 2.0);
    }

    #[test]
    fn same_inputs_give_bitwise_same_outputs() {
        let m = ActuatorModel::General {
            gain: [1.3, 0.2, -0.7],
            bias: [0.1, -0.4, 0.9],
            gear: 1.7,
        };
        let a = m.force(0.37, -1.2, 2.5);
        let b = m.force(0.37, -1.2, 2.5);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    fn finite_f64() -> impl Strategy<Value = f64> {
        -1.0e6..1.0e6
    }

    proptest! {
        #[test]
        fn force_finite_for_finite_input(
            ctrl in finite_f64(), q in finite_f64(), qd in finite_f64(),
            kp in -10.0..10.0, kd in -10.0..10.0, kv in -10.0..10.0, gear in -10.0..10.0,
            gain in prop::array::uniform3(-10.0..10.0_f64),
            bias in prop::array::uniform3(-10.0..10.0_f64),
        ) {
            let models = [
                ActuatorModel::Motor { gear },
                ActuatorModel::Position { kp, kd, gear },
                ActuatorModel::Velocity { kv, gear },
                ActuatorModel::General { gain, bias, gear },
            ];
            for m in models {
                prop_assert!(m.force(ctrl, q, qd).is_finite());
            }
        }

        #[test]
        fn delay_and_backlash_finite_for_finite_input(
            inputs in prop::collection::vec(finite_f64(), 1..20),
            ticks in 0u32..6,
            deadband in 0.0..10.0,
        ) {
            let mut delay = ActuatorDelay::new(ticks);
            let mut backlash = Backlash::new(deadband).unwrap();
            for x in inputs {
                prop_assert!(delay.step(x).is_finite());
                prop_assert!(backlash.step(x).is_finite());
            }
        }
    }
}
