//! Domain-gap diagnostics (§24.3): metrics comparing a sim dataset against a real one, plus
//! the report format read by `es gap` and gate 15 (§28.7).
//!
//! Full design (why this takes a [`GapInput`] rather than a `LeRobotDataset`, the pairing
//! rule, the suspects heuristic): `docs/design/domain-gap.md`.
//!
//! Same "nothing is invented" rule as [`crate::metrics`]: an episode-level number this
//! dataset does not carry a column for is [`MetricValue::Unavailable`], never a fabricated
//! `0.0`. `BTreeMap` only, no `HashMap` (§18.4) — every report is walked in sorted key order
//! so two runs over the same inputs produce byte-identical JSON.

use std::collections::BTreeMap;
use std::fmt;

use es_ir::evaluation::MetricValue;
use serde::{Deserialize, Serialize};

use crate::metrics::{mean, std};

/// §18.3 knob a flagged channel points at (design doc §6). A lookup table, not inference.
const SUSPECT_TABLE: &[(&[&str], &str)] = &[
    (
        &["image", "cam", "rgb", "depth"],
        "exposure, white balance, lens distortion, rolling shutter (spec 18.3)",
    ),
    (
        &["vel", "velocity"],
        "IMU/joint-encoder noise model, latency (spec 18.3)",
    ),
    (
        &["force", "torque", ".ft", "wrench"],
        "F/T sensor Gaussian noise, temperature drift (spec 18.3)",
    ),
    (
        &["state", "joint", "position", "qpos"],
        "joint encoder quantization, delay, offset (spec 18.3)",
    ),
    (
        &["action"],
        "actuator/controller latency, action chunk delay (spec 18.3)",
    ),
    (
        &["latency."],
        "deadline/jitter budget (spec 24.2), not a sensor knob",
    ),
];

const DEFAULT_KNOB: &str = "generic sensor noise model (no specific spec 18.3 knob)";

fn suspect_knob(channel: &str) -> &'static str {
    let lower = channel.to_ascii_lowercase();
    for (needles, knob) in SUSPECT_TABLE {
        if needles.iter().any(|n| lower.contains(n)) {
            return knob;
        }
    }
    DEFAULT_KNOB
}

/// Options for [`DomainGap::compute`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GapOptions {
    /// Reservoir cap the *caller* subsamples channels to before calling `compute` (design
    /// doc §5); `compute` itself does no subsampling, it only records `sim_n`/`real_n`.
    pub max_samples_per_feature: usize,
    /// A channel is "flagged" (suspects list, exit code 1 on the CLI) when its KS `D`
    /// exceeds this.
    pub threshold: f64,
}

impl Default for GapOptions {
    fn default() -> Self {
        Self {
            max_samples_per_feature: 4096,
            threshold: 0.3,
        }
    }
}

/// One channel's worth of already-collected numeric samples: flat, row-major, `dims` wide.
///
/// A scalar feature is `dims == 1`. Values are assumed already subsampled by the caller
/// (design doc §5) — this type does not know how many frames it came from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureSamples {
    pub dims: usize,
    pub values: Vec<f64>,
    /// Frames the caller refused to put in `values` because at least one of their dims was
    /// not finite (design doc §5). Reported, never silently absorbed: `serde_json` has no
    /// encoding for NaN/Inf, and a distribution statistic over them has no meaning either.
    pub nonfinite_dropped: usize,
}

impl FeatureSamples {
    pub fn new(dims: usize, values: Vec<f64>) -> Self {
        Self {
            dims,
            values,
            nonfinite_dropped: 0,
        }
    }

    /// The `d`-th dim's samples across all frames.
    fn column(&self, d: usize) -> Vec<f64> {
        self.values
            .iter()
            .skip(d)
            .step_by(self.dims.max(1))
            .copied()
            .collect()
    }
}

/// One episode's outcome, for the episode-level block. `None` means the dataset carries no
/// column this could be read from (design doc §1's "nothing is invented" rule).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EpisodeSummary {
    pub success: Option<bool>,
    pub length: u64,
    pub envelope_violation: Option<f64>,
}

/// Everything [`DomainGap::compute`] needs from one side (sim or real).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GapInput {
    /// Channel name (already namespaced, e.g. `observation.state`, `action`,
    /// `latency.obs_age_ms`) -> its samples. Sorted key order for determinism.
    pub channels: BTreeMap<String, FeatureSamples>,
    pub episodes: Vec<EpisodeSummary>,
}

/// A gap computed over one numeric channel dim (design doc §2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelGap {
    pub name: String,
    pub sim_n: usize,
    pub real_n: usize,
    pub sim_mean: f64,
    pub real_mean: f64,
    pub sim_std: f64,
    pub real_std: f64,
    pub mean_diff: f64,
    pub ks_d: f64,
    pub wasserstein1: f64,
    pub sim_quantiles: [f64; 3],
    pub real_quantiles: [f64; 3],
    /// Frames dropped for a non-finite value on each side, carried through from
    /// [`FeatureSamples::nonfinite_dropped`] so the row says how much of the channel it saw.
    #[serde(default)]
    pub sim_nonfinite_dropped: usize,
    #[serde(default)]
    pub real_nonfinite_dropped: usize,
    pub flagged: bool,
}

/// A channel both sides carry under the same name but with a different width. Listed, not
/// scored: dim `d` of a 6-wide sim feature and dim `d` of a 7-wide real one are two different
/// physical quantities, and comparing them is the "fabricated number" the design doc forbids.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimsMismatch {
    pub channel: String,
    pub sim_dims: usize,
    pub real_dims: usize,
}

/// Episode-level gap (§10.3 `success_rate`, `episode_length`, `envelope_violation_rate`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpisodeGap {
    pub sim_success_rate: MetricValue,
    pub real_success_rate: MetricValue,
    pub sim_length_mean: MetricValue,
    pub real_length_mean: MetricValue,
    pub sim_envelope_violation_rate: MetricValue,
    pub real_envelope_violation_rate: MetricValue,
}

/// A flagged channel mapped to a §18.3 knob to go check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suspect {
    pub channel: String,
    pub ks_d: f64,
    pub knob: String,
}

/// `gap_report.json` (design doc §7).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GapReport {
    pub threshold: f64,
    pub channels: Vec<ChannelGap>,
    pub unmatched_sim: Vec<String>,
    pub unmatched_real: Vec<String>,
    /// Channels paired by name whose widths disagree (see [`DimsMismatch`]).
    #[serde(default)]
    pub dims_mismatch: Vec<DimsMismatch>,
    pub episodes: EpisodeGap,
    pub suspects: Vec<Suspect>,
}

impl GapReport {
    /// `Result`, not an `.expect`: `serde_json` refuses NaN/Inf, and a report the CLI has
    /// already printed must not take the process down on the way to disk.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// True when at least one channel exceeded `threshold` — the CLI's exit-code-1 condition.
    pub fn has_flagged(&self) -> bool {
        !self.suspects.is_empty()
    }
}

impl fmt::Display for GapReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "domain gap report (threshold D > {:.3})", self.threshold)?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<32} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "channel",
            "n_sim",
            "n_real",
            "mean_sim",
            "mean_real",
            "std_sim",
            "std_real",
            "ks_d",
            "w1"
        )?;
        for c in &self.channels {
            writeln!(
                f,
                "{:<32} {:>6} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>8.4} {:>8.4}{}",
                c.name,
                c.sim_n,
                c.real_n,
                c.sim_mean,
                c.real_mean,
                c.sim_std,
                c.real_std,
                c.ks_d,
                c.wasserstein1,
                if c.flagged { "  *" } else { "" }
            )?;
        }
        if !self.unmatched_sim.is_empty() {
            writeln!(f)?;
            writeln!(f, "unmatched (sim only): {}", self.unmatched_sim.join(", "))?;
        }
        if !self.unmatched_real.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "unmatched (real only): {}",
                self.unmatched_real.join(", ")
            )?;
        }
        if !self.dims_mismatch.is_empty() {
            writeln!(f)?;
            writeln!(f, "dims mismatch (not scored):")?;
            for m in &self.dims_mismatch {
                writeln!(f, "  {} sim={} real={}", m.channel, m.sim_dims, m.real_dims)?;
            }
        }
        let dropped: usize = self
            .channels
            .iter()
            .map(|c| c.sim_nonfinite_dropped + c.real_nonfinite_dropped)
            .sum();
        if dropped > 0 {
            writeln!(f)?;
            writeln!(f, "dropped {dropped} non-finite frame(s) before comparing")?;
        }
        writeln!(f)?;
        writeln!(f, "episodes:")?;
        writeln!(
            f,
            "  success_rate           sim={:?} real={:?}",
            self.episodes.sim_success_rate, self.episodes.real_success_rate
        )?;
        writeln!(
            f,
            "  length_mean            sim={:?} real={:?}",
            self.episodes.sim_length_mean, self.episodes.real_length_mean
        )?;
        writeln!(
            f,
            "  envelope_violation_rate sim={:?} real={:?}",
            self.episodes.sim_envelope_violation_rate, self.episodes.real_envelope_violation_rate
        )?;
        if !self.suspects.is_empty() {
            writeln!(f)?;
            writeln!(f, "suspects:")?;
            for s in &self.suspects {
                writeln!(f, "  {} (D={:.4}) -> {}", s.channel, s.ks_d, s.knob)?;
            }
        }
        Ok(())
    }
}

/// Everything [`GapOptions`] can reject before any sample is touched.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GapError {
    #[error("--threshold must be in [0, 1], got {0}")]
    InvalidThreshold(f64),
    #[error("--max-samples must be nonzero")]
    ZeroMaxSamples,
}

/// The domain-gap diagnostic (§24.3). No state: [`Self::compute`] is the entire API.
#[derive(Debug)]
pub struct DomainGap;

impl DomainGap {
    /// Computes the gap between `sim` and `real` (design doc). Channels are compared by
    /// exact name match, in sorted order; unmatched names are listed, not scored.
    pub fn compute(
        sim: &GapInput,
        real: &GapInput,
        opts: &GapOptions,
    ) -> Result<GapReport, GapError> {
        if !(0.0..=1.0).contains(&opts.threshold) {
            return Err(GapError::InvalidThreshold(opts.threshold));
        }
        if opts.max_samples_per_feature == 0 {
            return Err(GapError::ZeroMaxSamples);
        }

        let mut channels = Vec::new();
        let mut unmatched_sim = Vec::new();
        let mut unmatched_real = Vec::new();
        let mut dims_mismatch = Vec::new();

        // BTreeMap keys are already sorted; a merge-join gives deterministic paired /
        // unmatched partitioning in one pass with no HashMap involved.
        let mut si = sim.channels.iter().peekable();
        let mut ri = real.channels.iter().peekable();
        loop {
            match (si.peek(), ri.peek()) {
                (Some((sk, sv)), Some((rk, rv))) => match sk.cmp(rk) {
                    std::cmp::Ordering::Equal => {
                        if sv.dims == rv.dims {
                            channels.extend(compare_channel(sk, sv, rv, opts.threshold));
                        } else {
                            dims_mismatch.push(DimsMismatch {
                                channel: (*sk).clone(),
                                sim_dims: sv.dims,
                                real_dims: rv.dims,
                            });
                        }
                        si.next();
                        ri.next();
                    }
                    std::cmp::Ordering::Less => {
                        unmatched_sim.push((*sk).clone());
                        si.next();
                    }
                    std::cmp::Ordering::Greater => {
                        unmatched_real.push((*rk).clone());
                        ri.next();
                    }
                },
                (Some((sk, _)), None) => {
                    unmatched_sim.push((*sk).clone());
                    si.next();
                }
                (None, Some((rk, _))) => {
                    unmatched_real.push((*rk).clone());
                    ri.next();
                }
                (None, None) => break,
            }
        }

        let suspects = channels
            .iter()
            .filter(|c| c.flagged)
            .map(|c| Suspect {
                channel: c.name.clone(),
                ks_d: c.ks_d,
                knob: suspect_knob(&c.name).to_owned(),
            })
            .collect();

        let episodes = episode_gap(&sim.episodes, &real.episodes);

        Ok(GapReport {
            threshold: opts.threshold,
            channels,
            unmatched_sim,
            unmatched_real,
            dims_mismatch,
            episodes,
            suspects,
        })
    }
}

fn compare_channel(
    name: &str,
    sim: &FeatureSamples,
    real: &FeatureSamples,
    threshold: f64,
) -> Vec<ChannelGap> {
    // `compute` only pairs channels of equal width; a mismatch is listed, never clamped onto
    // an unrelated column.
    debug_assert_eq!(sim.dims, real.dims);
    let dims = sim.dims.max(1);
    (0..dims)
        .map(|d| {
            let a = sim.column(d);
            let b = real.column(d);
            let dim_name = if dims == 1 {
                name.to_owned()
            } else {
                format!("{name}[{d}]")
            };
            let ks_d = ks_statistic(&a, &b);
            ChannelGap {
                name: dim_name,
                sim_n: a.len(),
                real_n: b.len(),
                sim_mean: mean(&a),
                real_mean: mean(&b),
                sim_std: std(&a),
                real_std: std(&b),
                mean_diff: mean(&b) - mean(&a),
                ks_d,
                wasserstein1: wasserstein1(&a, &b),
                sim_quantiles: quantiles(&a),
                real_quantiles: quantiles(&b),
                sim_nonfinite_dropped: sim.nonfinite_dropped,
                real_nonfinite_dropped: real.nonfinite_dropped,
                flagged: ks_d > threshold,
            }
        })
        .collect()
}

fn episode_gap(sim: &[EpisodeSummary], real: &[EpisodeSummary]) -> EpisodeGap {
    EpisodeGap {
        sim_success_rate: success_rate(sim),
        real_success_rate: success_rate(real),
        sim_length_mean: length_mean(sim),
        real_length_mean: length_mean(real),
        sim_envelope_violation_rate: envelope_violation_rate(sim),
        real_envelope_violation_rate: envelope_violation_rate(real),
    }
}

fn unavailable(reason: &str) -> MetricValue {
    MetricValue::Unavailable {
        reason: reason.to_owned(),
    }
}

fn success_rate(episodes: &[EpisodeSummary]) -> MetricValue {
    let known: Vec<f64> = episodes
        .iter()
        .filter_map(|e| e.success.map(|s| f64::from(u8::from(s))))
        .collect();
    if known.is_empty() {
        unavailable("no success/reward column in this dataset")
    } else {
        MetricValue::Scalar(mean(&known))
    }
}

fn length_mean(episodes: &[EpisodeSummary]) -> MetricValue {
    if episodes.is_empty() {
        return unavailable("no episodes in this dataset");
    }
    let lengths: Vec<f64> = episodes.iter().map(|e| e.length as f64).collect();
    MetricValue::Scalar(mean(&lengths))
}

fn envelope_violation_rate(episodes: &[EpisodeSummary]) -> MetricValue {
    let known: Vec<f64> = episodes
        .iter()
        .filter_map(|e| e.envelope_violation)
        .collect();
    if known.is_empty() {
        unavailable("no action_source column in this dataset")
    } else {
        MetricValue::Scalar(mean(&known))
    }
}

// --- Statistics (design doc §2): sorted-merge KS D and Wasserstein-1, no stats crate ------

/// Two-sample Kolmogorov-Smirnov statistic `D = sup_x |F_a(x) - F_b(x)|`, via a sorted merge
/// of both samples (`O((n+m) log(n+m))` for the sort, `O(n+m)` for the merge).
///
/// The empirical CDFs are right-continuous step functions, so the supremum is only ever
/// attained *after* a value's whole run of repeats: each step of the merge picks the smaller
/// of the two cursors' values and advances **both** cursors past every sample equal to it
/// before evaluating the gap (standard `ks_2samp` semantics). Advancing one repeat at a time
/// instead reports a gap at a point that is not on either CDF, which on robot data — binary
/// flags, quantised encoder counts, unequal sample counts — is a large fabricated `D`.
pub fn ks_statistic(sim: &[f64], real: &[f64]) -> f64 {
    if sim.is_empty() || real.is_empty() {
        return 0.0;
    }
    let mut sim = sim.to_vec();
    let mut real = real.to_vec();
    sim.sort_by(f64::total_cmp);
    real.sort_by(f64::total_cmp);
    let (n_sim, n_real) = (sim.len(), real.len());
    let (mut i_sim, mut i_real) = (0usize, 0usize);
    let mut max_gap = 0.0f64;
    while i_sim < n_sim && i_real < n_real {
        // `total_cmp`, the same order the sort used, so the two loops below cannot disagree
        // with it on -0.0/NaN and the merge always advances.
        let (sv, rv) = (sim[i_sim], real[i_real]);
        let x = if sv.total_cmp(&rv).is_le() { sv } else { rv };
        while i_sim < n_sim && sim[i_sim].total_cmp(&x).is_le() {
            i_sim += 1;
        }
        while i_real < n_real && real[i_real].total_cmp(&x).is_le() {
            i_real += 1;
        }
        let f_sim = i_sim as f64 / n_sim as f64;
        let f_real = i_real as f64 / n_real as f64;
        max_gap = max_gap.max((f_sim - f_real).abs());
    }
    max_gap
}

/// Wasserstein-1 distance between two 1-D empirical distributions: the integral of
/// `|F_a(x) - F_b(x)|` over the combined support, evaluated on the merged sorted breakpoints
/// so no interpolation or binning is needed.
pub fn wasserstein1(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (na, nb) = (a.len() as f64, b.len() as f64);

    let mut xs: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    xs.sort_by(f64::total_cmp);
    // Exact dedup of merged breakpoints (not a tolerance comparison): a bit-pattern check
    // sidesteps `clippy::float_cmp` without changing the semantics.
    xs.dedup_by(|x, y| x.to_bits() == y.to_bits());

    let cdf = |sorted: &[f64], x: f64| sorted.partition_point(|&v| v <= x) as f64;

    let mut area = 0.0;
    for w in xs.windows(2) {
        let (x0, x1) = (w[0], w[1]);
        let fa = cdf(&a, x0) / na;
        let fb = cdf(&b, x0) / nb;
        area += (fa - fb).abs() * (x1 - x0);
    }
    area
}

/// p10/p50/p90, nearest-rank (no interpolation — same convention as
/// `es_eval::metrics::aggregate`'s `Aggregation::P95`), so it is exactly reproducible.
fn quantiles(values: &[f64]) -> [f64; 3] {
    if values.is_empty() {
        return [0.0; 3];
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    [0.1, 0.5, 0.9].map(|q| {
        let rank = ((sorted.len() as f64) * q).ceil() as usize;
        sorted[rank.clamp(1, sorted.len()) - 1]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- KS / W1 on hand-computed samples --------------------------------------------

    #[test]
    fn ks_statistic_is_one_for_fully_separated_samples() {
        // a entirely below b: the empirical CDFs never overlap, D = 1.
        assert!((ks_statistic(&[0.0, 1.0], &[2.0, 3.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ks_statistic_is_zero_for_identical_samples() {
        let v = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert!(ks_statistic(&v, &v).abs() < 1e-12);
    }

    #[test]
    fn ks_statistic_half_overlap_hand_computed() {
        // a = {0,1,2,3}, b = {2,3,4,5}. At x=1: Fa=2/4=0.5, Fb=0. At x=3: Fa=1, Fb=2/4=0.5.
        // Max |Fa-Fb| = 0.5.
        let d = ks_statistic(&[0.0, 1.0, 2.0, 3.0], &[2.0, 3.0, 4.0, 5.0]);
        assert!((d - 0.5).abs() < 1e-12, "{d}");
    }

    // --- KS past ties (docs/packets/M3/P-M3-R1.md) -------------------------------------

    #[test]
    fn ks_statistic_is_zero_for_one_repeated_value_at_unequal_n() {
        // Both CDFs are the single step 0 -> 1 at x = 1. Advancing one repeat at a time
        // evaluated the gap at points that are on neither CDF and reported D = 2/3.
        assert!(ks_statistic(&[1.0, 1.0, 1.0], &[1.0]).abs() < 1e-12);
        assert!(ks_statistic(&[1.0], &[1.0, 1.0, 1.0]).abs() < 1e-12);
    }

    #[test]
    fn ks_statistic_is_zero_for_a_binary_channel_at_unequal_n() {
        // A gripper flag / an `action_source` bit: 40% zeros on both sides, 300 vs 100
        // samples. Identical distributions, so D must be exactly 0.
        let sim: Vec<f64> = (0..300).map(|i| f64::from(u8::from(i % 5 >= 2))).collect();
        let real: Vec<f64> = (0..100).map(|i| f64::from(u8::from(i % 5 >= 2))).collect();
        let d = ks_statistic(&sim, &real);
        assert!(d.abs() < 1e-12, "{d}");
    }

    #[test]
    fn ks_statistic_is_zero_for_a_quantised_channel_at_unequal_n() {
        // A 10-level quantised encoder channel, same levels and same proportions on both
        // sides at 500 vs 100 samples.
        let sim: Vec<f64> = (0..500).map(|i| f64::from(i % 10)).collect();
        let real: Vec<f64> = (0..100).map(|i| f64::from(i % 10)).collect();
        let d = ks_statistic(&sim, &real);
        assert!(d.abs() < 1e-12, "{d}");
    }

    #[test]
    fn ks_statistic_still_sees_a_shifted_tie_run() {
        // Ties are not an excuse to report 0: {0,0,0,0} vs {1,1} share no support.
        assert!((ks_statistic(&[0.0; 4], &[1.0, 1.0]) - 1.0).abs() < 1e-12);
        // Half the sim mass sits on a value the real side never takes.
        let d = ks_statistic(&[0.0, 0.0, 1.0, 1.0], &[1.0]);
        assert!((d - 0.5).abs() < 1e-12, "{d}");
    }

    #[test]
    fn wasserstein1_is_zero_for_identical_samples() {
        let v = [0.0, 1.0, 2.0, 3.0];
        assert!(wasserstein1(&v, &v).abs() < 1e-12);
    }

    #[test]
    fn wasserstein1_matches_mean_shift_hand_computed() {
        // a = {0,1}, b = {2,3}: optimal pairing (0,2),(1,3), mean |diff| = 2.
        let w = wasserstein1(&[0.0, 1.0], &[2.0, 3.0]);
        assert!((w - 2.0).abs() < 1e-12, "{w}");
    }

    #[test]
    fn wasserstein1_single_point_distance() {
        assert!((wasserstein1(&[0.0], &[1.0]) - 1.0).abs() < 1e-12);
    }

    // --- DomainGap::compute --------------------------------------------------------

    fn input(channels: &[(&str, usize, &[f64])], episodes: Vec<EpisodeSummary>) -> GapInput {
        let mut map = BTreeMap::new();
        for (name, dims, values) in channels {
            map.insert(
                (*name).to_owned(),
                FeatureSamples::new(*dims, (*values).to_vec()),
            );
        }
        GapInput {
            channels: map,
            episodes,
        }
    }

    #[test]
    fn identical_datasets_have_all_channels_near_zero_gap() {
        let v: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.1).collect();
        let a = input(&[("observation.state", 1, &v)], vec![]);
        let b = a.clone();
        let report = DomainGap::compute(&a, &b, &GapOptions::default()).unwrap();
        assert_eq!(report.channels.len(), 1);
        assert!(report.channels[0].ks_d < 1e-9, "{:?}", report.channels[0]);
        assert!(!report.channels[0].flagged);
        assert!(report.unmatched_sim.is_empty());
        assert!(report.unmatched_real.is_empty());
        assert!(!report.has_flagged());
    }

    #[test]
    fn shifted_feature_is_flagged_others_are_not() {
        let stable: Vec<f64> = (0..200).map(|i| f64::from(i % 7)).collect();
        let sim_shift: Vec<f64> = (0..200).map(|i| f64::from(i % 5)).collect();
        let real_shift: Vec<f64> = (0..200).map(|i| f64::from(i % 5) + 10.0).collect();

        let sim = input(
            &[("observation.state", 1, &stable), ("action", 1, &sim_shift)],
            vec![],
        );
        let real = input(
            &[
                ("observation.state", 1, &stable),
                ("action", 1, &real_shift),
            ],
            vec![],
        );
        let report = DomainGap::compute(&sim, &real, &GapOptions::default()).unwrap();
        let by_name = |n: &str| report.channels.iter().find(|c| c.name == n).unwrap();
        assert!(!by_name("observation.state").flagged, "{report}");
        assert!(by_name("action").flagged, "{report}");
        assert_eq!(report.suspects.len(), 1);
        assert_eq!(report.suspects[0].channel, "action");
    }

    #[test]
    fn unmatched_feature_is_listed_not_scored() {
        let v: Vec<f64> = vec![1.0, 2.0, 3.0];
        let sim = input(
            &[
                ("observation.state", 1, &v),
                ("observation.only_sim", 1, &v),
            ],
            vec![],
        );
        let real = input(
            &[
                ("observation.state", 1, &v),
                ("observation.only_real", 1, &v),
            ],
            vec![],
        );
        let report = DomainGap::compute(&sim, &real, &GapOptions::default()).unwrap();
        assert_eq!(report.channels.len(), 1);
        assert_eq!(
            report.unmatched_sim,
            vec!["observation.only_sim".to_owned()]
        );
        assert_eq!(
            report.unmatched_real,
            vec!["observation.only_real".to_owned()]
        );
    }

    #[test]
    fn multi_dim_channel_expands_per_dim_rows() {
        // 2 dims, 3 frames: row-major [f0d0,f0d1, f1d0,f1d1, f2d0,f2d1].
        let v = vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0];
        let a = input(&[("observation.pose", 2, &v)], vec![]);
        let b = a.clone();
        let report = DomainGap::compute(&a, &b, &GapOptions::default()).unwrap();
        assert_eq!(report.channels.len(), 2);
        assert_eq!(report.channels[0].name, "observation.pose[0]");
        assert_eq!(report.channels[1].name, "observation.pose[1]");
    }

    #[test]
    fn episode_stats_are_unavailable_when_no_column_and_scalar_otherwise() {
        let empty = input(
            &[],
            vec![EpisodeSummary::default(), EpisodeSummary::default()],
        );
        let report = DomainGap::compute(&empty, &empty, &GapOptions::default()).unwrap();
        assert!(matches!(
            report.episodes.sim_success_rate,
            MetricValue::Unavailable { .. }
        ));
        assert!(matches!(
            report.episodes.sim_envelope_violation_rate,
            MetricValue::Unavailable { .. }
        ));
        assert_eq!(report.episodes.sim_length_mean, MetricValue::Scalar(0.0));

        let with_data = input(
            &[],
            vec![
                EpisodeSummary {
                    success: Some(true),
                    length: 10,
                    envelope_violation: Some(0.1),
                },
                EpisodeSummary {
                    success: Some(false),
                    length: 20,
                    envelope_violation: Some(0.3),
                },
            ],
        );
        let report = DomainGap::compute(&with_data, &with_data, &GapOptions::default()).unwrap();
        assert_eq!(report.episodes.sim_success_rate, MetricValue::Scalar(0.5));
        assert_eq!(report.episodes.sim_length_mean, MetricValue::Scalar(15.0));
        assert_eq!(
            report.episodes.sim_envelope_violation_rate,
            MetricValue::Scalar(0.2)
        );
    }

    #[test]
    fn invalid_options_are_rejected() {
        let empty = GapInput::default();
        assert_eq!(
            DomainGap::compute(
                &empty,
                &empty,
                &GapOptions {
                    threshold: 1.5,
                    ..GapOptions::default()
                }
            ),
            Err(GapError::InvalidThreshold(1.5))
        );
        assert_eq!(
            DomainGap::compute(
                &empty,
                &empty,
                &GapOptions {
                    max_samples_per_feature: 0,
                    ..GapOptions::default()
                }
            ),
            Err(GapError::ZeroMaxSamples)
        );
    }

    #[test]
    fn report_json_round_trips() {
        let v: Vec<f64> = vec![1.0, 2.0, 3.0];
        let a = input(&[("action", 1, &v)], vec![]);
        let report = DomainGap::compute(&a, &a, &GapOptions::default()).unwrap();
        let json = report.to_json().expect("serializes");
        let back: GapReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }

    #[test]
    fn a_channel_whose_width_disagrees_is_listed_not_clamped() {
        let sim = input(&[("observation.state", 2, &[0.0, 1.0, 2.0, 3.0])], vec![]);
        let real = input(&[("observation.state", 3, &[0.0, 1.0, 2.0])], vec![]);
        let report = DomainGap::compute(&sim, &real, &GapOptions::default()).unwrap();
        assert!(report.channels.is_empty(), "{report}");
        assert_eq!(
            report.dims_mismatch,
            vec![DimsMismatch {
                channel: "observation.state".to_owned(),
                sim_dims: 2,
                real_dims: 3,
            }]
        );
        assert!(!report.has_flagged());
    }

    #[test]
    fn suspect_knob_maps_known_prefixes() {
        assert!(suspect_knob("observation.images.top").contains("exposure"));
        assert!(suspect_knob("observation.joint_velocity").contains("IMU"));
        assert!(suspect_knob("observation.wrench").contains("F/T"));
        assert!(suspect_knob("observation.state").contains("encoder"));
        assert!(suspect_knob("action").contains("actuator"));
        assert!(suspect_knob("latency.obs_age_ms").contains("jitter"));
        assert_eq!(suspect_knob("mystery_channel"), DEFAULT_KNOB);
    }
}
