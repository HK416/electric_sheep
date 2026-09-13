//! Deterministic sensor realism models (spec 18.3): Gaussian noise, bias/drift, quantization,
//! dropout, a fixed-delay latency line and rolling-shutter row timing, composed into a fixed
//! op-order [`SensorModel`] pipeline. All pure over `f64` slices; the only transcendentals
//! (Box-Muller's `sqrt`/`ln`/`cos`) go through `es_math::approx` (spec 3.2 `DET-010`, spec 3.4:
//! no `std` transcendentals in a deterministic-mode kernel).

use std::collections::VecDeque;

use es_core::PhysTick;

/// `splitmix64`'s finalizer, adapted from `crates/es-env/src/rng.rs` (`EnvRng`). Copied rather
/// than depended on: `es-sensor` is layer 3 and `es-env` is layer 9 (spec 4.2 forbids the
/// reverse dependency), so this crate carries its own tiny RNG core instead.
const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Folds a sensor's `StableId` into a base seed so every sensor draws an independent stream
/// even under the same base seed (spec 3.4 "no global RNG", `DET-001` stream addressing).
#[must_use]
pub fn sensor_seed(seed: u64, id: es_core::StableId) -> u64 {
    let bytes = id.as_bytes();
    let (lo, hi) = bytes.split_at(8);
    let to_u64 = |s: &[u8]| u64::from_le_bytes(s.try_into().expect("8 bytes"));
    splitmix64(splitmix64(seed) ^ to_u64(lo)) ^ splitmix64(to_u64(hi))
}

/// One RNG stream, keyed by `(seed, env, tick)`. `seed` is expected to already be per-sensor
/// (see [`sensor_seed`]); counter-based so the nth draw depends only on the key and `n`, never
/// on call order from other stages (spec 3.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoiseRng {
    key: u64,
    counter: u64,
}

impl NoiseRng {
    #[must_use]
    pub fn new(seed: u64, env: u32, tick: PhysTick) -> Self {
        let mut key = splitmix64(seed);
        for part in [u64::from(env), tick.0] {
            key = splitmix64(key ^ part);
        }
        Self { key, counter: 0 }
    }

    pub fn next_u64(&mut self) -> u64 {
        let n = self.counter;
        self.counter += 1;
        splitmix64(self.key ^ n.wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Standard normal (mean 0, variance 1) via Box-Muller, fixed operand order, second
    /// variate discarded (mirrors `EnvRng::sample`'s `Distribution::Normal` case).
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let r = f64::from(es_math::approx::sqrt((-2.0 * ln(u1)) as f32));
        let theta = f64::from(es_math::approx::cos((std::f64::consts::TAU * u2) as f32));
        r * theta
    }
}

/// `ln` via `es_math::approx`, promoted to `f64` at the call site (spec 3.2 `DET-010`: no
/// `std` transcendentals in a deterministic kernel).
fn ln(x: f64) -> f64 {
    f64::from(es_math::approx::ln(x as f32))
}

// Per-stage salts so stages sharing the same `(seed, env, tick)` draw independent streams
// instead of replaying each other's random bits.
const SALT_GAUSSIAN: u64 = 0x5153_5f47_4155_5353;
const SALT_DROPOUT: u64 = 0x5153_5f44_524f_5030;

/// One stage of a [`SensorModel`] pipeline (spec 18.3). Order in `SensorModel::stages` is the
/// fixed application order.
#[derive(Clone, Debug)]
pub enum Stage {
    /// Additive white Gaussian noise, standard deviation `sigma` in the sensor's own units.
    Gaussian { sigma: f64 },
    /// Constant offset plus linear drift in absolute tick count (stateless: a pure function of
    /// `tick`, not an accumulator, so it needs no internal state to stay deterministic under
    /// resets).
    Bias { offset: f64, drift_per_tick: f64 },
    /// Rounds to the nearest multiple of `step` (`step <= 0` is a no-op).
    Quantize { step: f64 },
    /// Drops the whole sample with probability `p_drop`; `hold_last` repeats the previous
    /// sample instead of zeroing it. `last` is scratch state, sized on first use.
    Dropout {
        p_drop: f64,
        hold_last: bool,
        last: Vec<f64>,
    },
    /// Fixed delay of `ticks` physics ticks, via a pre-allocated ring. Pre-fill (before the
    /// ring has seen `ticks` samples) reads back as zero.
    Latency {
        ticks: u32,
        ring: VecDeque<Vec<f64>>,
    },
}

impl Stage {
    #[must_use]
    pub fn gaussian(sigma: f64) -> Self {
        Self::Gaussian { sigma }
    }

    #[must_use]
    pub fn bias(offset: f64, drift_per_tick: f64) -> Self {
        Self::Bias {
            offset,
            drift_per_tick,
        }
    }

    #[must_use]
    pub fn quantize(step: f64) -> Self {
        Self::Quantize { step }
    }

    #[must_use]
    pub fn dropout(p_drop: f64, hold_last: bool) -> Self {
        Self::Dropout {
            p_drop,
            hold_last,
            last: Vec::new(),
        }
    }

    #[must_use]
    pub fn latency(ticks: u32) -> Self {
        Self::Latency {
            ticks,
            ring: VecDeque::new(),
        }
    }

    fn apply(&mut self, env: u32, tick: PhysTick, values: &mut [f64], rng_seed: u64) {
        match self {
            Self::Gaussian { sigma } => {
                let mut rng = NoiseRng::new(rng_seed ^ SALT_GAUSSIAN, env, tick);
                for v in values.iter_mut() {
                    *v += *sigma * rng.gaussian();
                }
            }
            Self::Bias {
                offset,
                drift_per_tick,
            } => {
                let b = *offset + *drift_per_tick * tick.0 as f64;
                for v in values.iter_mut() {
                    *v += b;
                }
            }
            Self::Quantize { step } => {
                if *step > 0.0 {
                    for v in values.iter_mut() {
                        *v = (*v / *step).round() * *step;
                    }
                }
            }
            Self::Dropout {
                p_drop,
                hold_last,
                last,
            } => {
                if last.len() != values.len() {
                    *last = vec![0.0; values.len()];
                }
                let mut rng = NoiseRng::new(rng_seed ^ SALT_DROPOUT, env, tick);
                if rng.next_f64() < *p_drop {
                    if *hold_last {
                        values.copy_from_slice(last);
                    } else {
                        for v in values.iter_mut() {
                            *v = 0.0;
                        }
                    }
                } else {
                    last.copy_from_slice(values);
                }
            }
            Self::Latency { ticks, ring } => {
                if *ticks == 0 {
                    return;
                }
                if ring.is_empty() {
                    for _ in 0..*ticks {
                        ring.push_back(vec![0.0; values.len()]);
                    }
                }
                ring.push_back(values.to_vec());
                let delayed = ring.pop_front().expect("ring pre-filled to `ticks` above");
                values.copy_from_slice(&delayed);
            }
        }
    }
}

/// A fixed-order pipeline of [`Stage`]s applied to one sensor's flat sample buffer per tick.
#[derive(Clone, Debug, Default)]
pub struct SensorModel {
    pub stages: Vec<Stage>,
}

impl SensorModel {
    /// Runs every stage in `self.stages`, in order, over `values`. `rng_seed` should already
    /// be per-sensor (see [`sensor_seed`]); this call folds in `env` and `tick`.
    pub fn apply(&mut self, env: u32, tick: PhysTick, values: &mut [f64], rng_seed: u64) {
        for stage in &mut self.stages {
            stage.apply(env, tick, values, rng_seed);
        }
    }
}

/// Rolling-shutter row timing (spec 18.3). Math only: it hands back the tick offset for one
/// image row so a capture stage can time-sample per row; it is not a [`Stage`] because it
/// needs image height, not a flat value buffer, and produces a table rather than filtering
/// one.
#[derive(Clone, Copy, Debug)]
pub struct RollingShutterSkew {
    pub readout_ticks: u32,
}

impl RollingShutterSkew {
    /// Tick offset for `row` of `height` rows: `0` at the top row, `readout_ticks` at the
    /// bottom row, monotonic non-decreasing in between. Top-to-bottom readout only
    /// (unverified: real rolling-shutter sensors may read bottom-to-top; add a direction flag
    /// if that turns out to matter).
    #[must_use]
    pub fn row_offset_ticks(&self, row: u32, height: u32) -> u32 {
        if height <= 1 {
            return 0;
        }
        let row = row.min(height - 1);
        (u64::from(row) * u64::from(self.readout_ticks) / u64::from(height - 1)) as u32
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use es_core::StableId;
    use proptest::prelude::*;

    #[test]
    fn gaussian_same_seed_same_output_different_sensor_differs() {
        let seed_a = sensor_seed(1, StableId::from_path("sensor/a"));
        let seed_b = sensor_seed(1, StableId::from_path("sensor/b"));
        let mut m1 = SensorModel {
            stages: vec![Stage::gaussian(1.0)],
        };
        let mut m2 = SensorModel {
            stages: vec![Stage::gaussian(1.0)],
        };
        let mut v1 = [0.0_f64; 4];
        let mut v2 = [0.0_f64; 4];
        m1.apply(0, PhysTick(10), &mut v1, seed_a);
        m2.apply(0, PhysTick(10), &mut v2, seed_a);
        assert_eq!(v1, v2, "same seed must reproduce bit-for-bit");

        let mut v3 = [0.0_f64; 4];
        let mut m3 = SensorModel {
            stages: vec![Stage::gaussian(1.0)],
        };
        m3.apply(0, PhysTick(10), &mut v3, seed_b);
        assert_ne!(v1, v3, "different sensor id must draw a different stream");
    }

    #[test]
    fn gaussian_mean_and_variance_within_tolerance() {
        let mut rng = NoiseRng::new(42, 0, PhysTick(0));
        let n = 10_000_u32;
        let samples: Vec<f64> = (0..n).map(|_| rng.gaussian()).collect();
        let mean: f64 = samples.iter().sum::<f64>() / f64::from(n);
        let var: f64 = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / f64::from(n);
        assert!(mean.abs() < 0.05, "mean {mean} too far from 0");
        assert!((var - 1.0).abs() < 0.1, "variance {var} too far from 1");
    }

    #[test]
    fn latency_delays_by_exactly_n_ticks_with_zero_prefill() {
        let mut model = SensorModel {
            stages: vec![Stage::latency(3)],
        };
        let seed = 7;
        let mut outputs = Vec::new();
        for t in 0..8u64 {
            let mut v = [t as f64];
            model.apply(0, PhysTick(t), &mut v, seed);
            outputs.push(v[0]);
        }
        // First 3 ticks read back the zero pre-fill; tick t >= 3 reads input from t - 3.
        assert_eq!(outputs, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dropout_hold_last_repeats_previous_sample() {
        // p_drop = 1.0 always drops, deterministically, regardless of the draw.
        let mut model = SensorModel {
            stages: vec![Stage::dropout(1.0, true)],
        };
        let mut v = [5.0, 6.0];
        model.apply(0, PhysTick(0), &mut v, 1);
        // First call: `last` starts at zero, so a drop reads back zero.
        assert_eq!(v, [0.0, 0.0]);
        let mut v2 = [9.0, 9.0];
        model.apply(0, PhysTick(1), &mut v2, 1);
        // Still dropping, so it holds the last *output*, which was [0, 0].
        assert_eq!(v2, [0.0, 0.0]);

        let mut model = SensorModel {
            stages: vec![Stage::dropout(0.0, true)],
        };
        let mut v3 = [3.0, 4.0];
        model.apply(0, PhysTick(0), &mut v3, 1);
        assert_eq!(v3, [3.0, 4.0], "p_drop = 0 never drops");
    }

    #[test]
    fn rolling_shutter_offsets_monotonic_and_bounded() {
        let skew = RollingShutterSkew { readout_ticks: 20 };
        let height = 480;
        let mut prev = 0;
        for row in 0..height {
            let off = skew.row_offset_ticks(row, height);
            assert!(off <= skew.readout_ticks);
            assert!(off >= prev);
            prev = off;
        }
        assert_eq!(skew.row_offset_ticks(0, height), 0);
        assert_eq!(
            skew.row_offset_ticks(height - 1, height),
            skew.readout_ticks
        );
    }

    proptest! {
        #[test]
        fn pipeline_output_finite_for_finite_input(
            sigma in 0.0..10.0,
            offset in -100.0..100.0,
            drift in -1.0..1.0,
            step in 0.0..5.0,
            p_drop in 0.0..1.0,
            ticks in 0u32..5,
            input in prop::collection::vec(-1000.0..1000.0_f64, 1..8),
            tick in 0u64..1000,
            seed in any::<u64>(),
        ) {
            let mut model = SensorModel {
                stages: vec![
                    Stage::gaussian(sigma),
                    Stage::bias(offset, drift),
                    Stage::quantize(step),
                    Stage::dropout(p_drop, true),
                    Stage::latency(ticks),
                ],
            };
            let mut values = input.clone();
            model.apply(0, PhysTick(tick), &mut values, seed);
            for v in values {
                prop_assert!(v.is_finite());
            }
        }
    }
}
