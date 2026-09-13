//! Counter-based per-env RNG (§3.4 "no global RNG", §6.6 `DET-001` "stream is mandatory").
//!
//! A stream is addressed, not stepped: the nth draw of `(seed, env, episode, stream)` is a pure
//! function of those four plus `n`. Nothing is shared between envs, so thread count, env order
//! and subset resets cannot change a draw. See `docs/design/batch-domains.md` §4.

use es_core::StableId;
use es_ir::task::Distribution;

/// `splitmix64`'s finalizer — the whole mixing primitive, inline so there is no `rand`
/// dependency and no hidden global state.
const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// One RNG stream, keyed by `(seed, env, episode, stream_id)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvRng {
    key: u64,
    counter: u64,
}

impl EnvRng {
    /// Derives the stream key. `stream` is `StableId::from_path(stream_name)` of the node's
    /// declared stream, so renaming a stream changes its draws and nothing else does.
    pub fn new(seed: u64, env: u32, episode: u64, stream: StableId) -> Self {
        let b = stream.as_bytes();
        let (lo, hi) = b.split_at(8);
        let to_u64 = |s: &[u8]| u64::from_le_bytes(s.try_into().expect("8 bytes"));
        let mut key = splitmix64(seed);
        for part in [u64::from(env), episode, to_u64(lo), to_u64(hi)] {
            key = splitmix64(key ^ part);
        }
        Self { key, counter: 0 }
    }

    /// The next 64 raw bits. Counter-based: the nth call depends only on the key and `n`.
    pub fn next_u64(&mut self) -> u64 {
        let n = self.counter;
        self.counter += 1;
        splitmix64(self.key ^ n.wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    /// Uniform in `[0, 1)`, from the top 53 bits — exact, no rounding to 1.0.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Draws from a Task IR distribution (§6.3). Every variant is covered; adding one to
    /// `es_ir` without deciding its sampler will not compile.
    ///
    /// Transcendentals go through `es_math::approx`, not `std` (§3.2 `DET-010`): a draw does
    /// not need `f64` precision, but it does need to be bit-identical on every target.
    pub fn sample(&mut self, dist: &Distribution) -> f64 {
        match dist {
            Distribution::Constant(v) => *v,
            Distribution::Uniform { lo, hi } => lo + (hi - lo) * self.next_f64(),
            Distribution::LogUniform { lo, hi } => {
                // Non-positive bounds have no logarithm; the plan rejects them at compile time,
                // and a hand-built distribution degrades to the linear form rather than NaN.
                if *lo <= 0.0 || *hi <= 0.0 {
                    return lo + (hi - lo) * self.next_f64();
                }
                let (a, b) = (ln(*lo), ln(*hi));
                f64::from(es_math::approx::exp((a + (b - a) * self.next_f64()) as f32))
            }
            Distribution::Normal { mean, std } => {
                // Box-Muller, fixed operand order, second variate discarded so that a draw
                // costs exactly two words whatever the call sequence.
                let u1 = self.next_f64().max(f64::MIN_POSITIVE);
                let u2 = self.next_f64();
                let r = f64::from(es_math::approx::sqrt((-2.0 * ln(u1)) as f32));
                let theta = f64::from(es_math::approx::cos((std::f64::consts::TAU * u2) as f32));
                mean + std * r * theta
            }
            Distribution::Choice(vs) => match vs.len() {
                0 => 0.0,
                n => vs[(self.next_u64() % n as u64) as usize],
            },
        }
    }
}

fn ln(x: f64) -> f64 {
    f64::from(es_math::approx::ln(x as f32))
}

// Bitwise reproducibility is the property under test: these comparisons are deliberate.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn stream(name: &str) -> StableId {
        StableId::from_path(name)
    }

    fn draws(seed: u64, env: u32, episode: u64, name: &str, n: usize) -> Vec<u64> {
        let mut rng = EnvRng::new(seed, env, episode, stream(name));
        (0..n).map(|_| rng.next_u64()).collect()
    }

    proptest! {
        #[test]
        fn the_same_key_gives_the_same_sequence(
            seed: u64, env: u32, episode: u64, name in "[a-z]{1,8}",
        ) {
            prop_assert_eq!(
                draws(seed, env, episode, &name, 16),
                draws(seed, env, episode, &name, 16)
            );
        }

        #[test]
        fn changing_any_key_component_changes_the_sequence(
            seed: u64, env in 0u32..1024, episode in 0u64..1024,
        ) {
            let base = draws(seed, env, episode, "reset", 8);
            prop_assert_ne!(&base, &draws(seed ^ 1, env, episode, "reset", 8));
            prop_assert_ne!(&base, &draws(seed, env + 1, episode, "reset", 8));
            prop_assert_ne!(&base, &draws(seed, env, episode + 1, "reset", 8));
            prop_assert_ne!(&base, &draws(seed, env, episode, "mass", 8));
        }

        #[test]
        fn uniform_stays_in_range(lo in -1e3f64..1e3, width in 0.0f64..1e3, seed: u64) {
            let dist = Distribution::Uniform { lo, hi: lo + width };
            let mut rng = EnvRng::new(seed, 0, 0, stream("s"));
            for _ in 0..32 {
                let v = rng.sample(&dist);
                prop_assert!((lo..=lo + width).contains(&v), "{v}");
            }
        }
    }

    #[test]
    fn a_draw_is_addressable_not_sequential() {
        // Env 7 episode 3 sees the same stream whether or not env 6 drew anything first.
        let mut a = EnvRng::new(99, 7, 3, stream("reset"));
        let mut noise = EnvRng::new(99, 6, 3, stream("reset"));
        for _ in 0..100 {
            noise.next_u64();
        }
        let mut b = EnvRng::new(99, 7, 3, stream("reset"));
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn every_distribution_variant_is_sampled() {
        let mut rng = EnvRng::new(1, 0, 0, stream("s"));
        assert_eq!(rng.sample(&Distribution::Constant(4.25)), 4.25);

        let u = rng.sample(&Distribution::Uniform { lo: -2.0, hi: 2.0 });
        assert!((-2.0..=2.0).contains(&u), "{u}");

        let l = rng.sample(&Distribution::LogUniform { lo: 0.5, hi: 8.0 });
        assert!((0.49..=8.1).contains(&l), "{l}");
        // Degenerate bounds give a finite number, never a NaN.
        assert!(rng
            .sample(&Distribution::LogUniform { lo: -1.0, hi: 1.0 })
            .is_finite());

        let mut sum = 0.0;
        for _ in 0..512 {
            let n = rng.sample(&Distribution::Normal {
                mean: 3.0,
                std: 1.0,
            });
            assert!(n.is_finite());
            sum += n;
        }
        assert!(
            (sum / 512.0 - 3.0).abs() < 0.25,
            "mean drifted: {}",
            sum / 512.0
        );

        let choices = vec![10.0, 20.0, 30.0];
        let mut hit = [false; 3];
        for _ in 0..64 {
            let c = rng.sample(&Distribution::Choice(choices.clone()));
            let i = choices
                .iter()
                .position(|v| *v == c)
                .expect("a listed value");
            hit[i] = true;
        }
        assert_eq!(hit, [true; 3]);
        assert_eq!(rng.sample(&Distribution::Choice(Vec::new())), 0.0);
    }

    #[test]
    fn next_f64_is_half_open() {
        let mut rng = EnvRng::new(0xdead_beef, 0, 0, stream("s"));
        for _ in 0..4096 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "{v}");
        }
    }
}
