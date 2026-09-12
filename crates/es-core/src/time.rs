//! Integer time model (§18.1, P15).
//!
//! Every step is an integer tick. Floating-point time accumulation is forbidden (§3.4) and is
//! prevented here by construction: no type in this module stores an `f64`, and the only way to
//! obtain seconds is [`SimTime::as_secs_f64`], at the edge of the system.

use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::Error;

/// A physics tick number. Every sensor sample carries one (§18.1).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct PhysTick(pub u64);

impl PhysTick {
    pub const ZERO: Self = Self(0);

    /// Advances by `n` ticks.
    ///
    /// # Panics
    /// If the tick counter would overflow `u64` (about 5.8e8 years at 1 kHz).
    #[must_use]
    pub fn add_ticks(self, n: u64) -> Self {
        Self(self.0.checked_add(n).expect("physics tick overflow"))
    }

    pub fn checked_sub(self, n: u64) -> Option<Self> {
        self.0.checked_sub(n).map(Self)
    }

    /// Ticks elapsed since `earlier`, or `None` if `earlier` is in the future.
    pub fn ticks_since(self, earlier: Self) -> Option<u64> {
        self.0.checked_sub(earlier.0)
    }
}

/// An exact rational tick rate in hertz (`num / den`).
///
/// Rational rather than `f64` so that 30 Hz, 240/7 Hz and 1 kHz are all exact and a tick count
/// converts to seconds without accumulated error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TickRate {
    num: NonZeroU64,
    den: NonZeroU64,
}

impl TickRate {
    /// `hz` ticks per second.
    ///
    /// # Panics
    /// If `hz` is zero.
    pub fn hz(hz: u64) -> Self {
        Self::rational(hz, 1).expect("tick rate must be non-zero")
    }

    /// `num / den` ticks per second, e.g. `rational(30000, 1001)` for 29.97 Hz.
    pub fn rational(num: u64, den: u64) -> Result<Self, Error> {
        match (NonZeroU64::new(num), NonZeroU64::new(den)) {
            (Some(num), Some(den)) => Ok(Self { num, den }),
            _ => Err(Error::InvalidTickRate { num, den }),
        }
    }

    pub const fn num(self) -> u64 {
        self.num.get()
    }

    pub const fn den(self) -> u64 {
        self.den.get()
    }

    /// Rate in hertz. Edge conversion only.
    pub fn as_hz_f64(self) -> f64 {
        self.num() as f64 / self.den() as f64
    }

    /// Tick period in seconds. Edge conversion only; never accumulate this.
    pub fn period_secs_f64(self) -> f64 {
        self.den() as f64 / self.num() as f64
    }
}

/// Simulation time: a tick count plus the rate it is counted at (§18.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimTime {
    tick: PhysTick,
    rate: TickRate,
}

impl SimTime {
    pub const fn new(tick: PhysTick, rate: TickRate) -> Self {
        Self { tick, rate }
    }

    pub const fn zero(rate: TickRate) -> Self {
        Self {
            tick: PhysTick::ZERO,
            rate,
        }
    }

    pub const fn tick(self) -> PhysTick {
        self.tick
    }

    pub const fn rate(self) -> TickRate {
        self.rate
    }

    /// Advances by `n` ticks. The rate is unchanged, so nothing is ever accumulated in floats.
    ///
    /// # Panics
    /// If the tick counter would overflow `u64`.
    #[must_use]
    pub fn add_ticks(self, n: u64) -> Self {
        Self {
            tick: self.tick.add_ticks(n),
            rate: self.rate,
        }
    }

    /// Elapsed time since `earlier`, or `None` if the rates differ or `earlier` is later.
    pub fn elapsed_since(self, earlier: Self) -> Option<Self> {
        if self.rate != earlier.rate {
            return None;
        }
        Some(Self {
            tick: PhysTick(self.tick.ticks_since(earlier.tick)?),
            rate: self.rate,
        })
    }

    /// Whole nanoseconds, truncated. Exact whenever the tick period is a whole number of
    /// nanoseconds (every rate that divides 1e9, which includes all the defaults of §18.1).
    pub fn as_nanos(self) -> u128 {
        u128::from(self.tick.0) * u128::from(self.rate.den()) * 1_000_000_000
            / u128::from(self.rate.num())
    }

    /// Seconds as `f64`. The only float in the time model; split into an exact integer part
    /// plus a remainder so that exactly representable times stay exact.
    pub fn as_secs_f64(self) -> f64 {
        let ticks = u128::from(self.tick.0) * u128::from(self.rate.den());
        let num = u128::from(self.rate.num());
        (ticks / num) as f64 + (ticks % num) as f64 / num as f64
    }
}

#[cfg(test)]
// Exactness is the property under test: these comparisons are deliberate.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn one_million_ticks_at_1khz_is_exactly_1000_seconds() {
        let t = SimTime::new(PhysTick(1_000_000), TickRate::hz(1000));
        assert_eq!(t.as_secs_f64(), 1000.0);
        assert_eq!(t.as_nanos(), 1_000_000_000_000);
    }

    #[test]
    fn stepping_a_tick_at_a_time_never_drifts() {
        let rate = TickRate::hz(1000);
        let mut t = SimTime::zero(rate);
        for _ in 0..1_000_000 {
            t = t.add_ticks(1);
        }
        assert_eq!(t.tick(), PhysTick(1_000_000));
        assert_eq!(t.as_secs_f64(), 1000.0);
    }

    #[test]
    fn rational_rates_are_exact() {
        // 29.97 Hz: 1001 ticks take exactly 1001/29.97 s = 30000 ms / 29.97... check via nanos.
        let rate = TickRate::rational(30_000, 1001).unwrap();
        let t = SimTime::new(PhysTick(30_000), rate);
        assert_eq!(t.as_secs_f64(), 1001.0);
        assert_eq!(t.as_nanos(), 1_001_000_000_000);
        assert_eq!(
            SimTime::new(PhysTick(3), TickRate::hz(1)).as_secs_f64(),
            3.0
        );
    }

    #[test]
    fn zero_rate_is_an_error_not_a_panic() {
        assert!(TickRate::rational(1000, 0).is_err());
        assert!(TickRate::rational(0, 1).is_err());
        assert!(TickRate::rational(1000, 1).is_ok());
    }

    #[test]
    fn tick_arithmetic() {
        let a = PhysTick(10);
        assert_eq!(a.add_ticks(5), PhysTick(15));
        assert_eq!(a.checked_sub(11), None);
        assert_eq!(PhysTick(15).ticks_since(a), Some(5));
        assert_eq!(a.ticks_since(PhysTick(15)), None);

        let rate = TickRate::hz(500);
        let t0 = SimTime::new(PhysTick(10), rate);
        let t1 = t0.add_ticks(40);
        assert_eq!(t1.elapsed_since(t0).unwrap().as_secs_f64(), 0.08);
        assert_eq!(
            t1.elapsed_since(SimTime::new(PhysTick(10), TickRate::hz(1000))),
            None
        );
    }

    #[test]
    fn serde_round_trip() {
        let t = SimTime::new(PhysTick(7), TickRate::rational(30_000, 1001).unwrap());
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<SimTime>(&json).unwrap(), t);
        assert_eq!(serde_json::to_string(&PhysTick(7)).unwrap(), "7");
        // A zero denominator cannot be deserialized back in.
        assert!(serde_json::from_str::<TickRate>("{\"num\":1,\"den\":0}").is_err());
    }
}
