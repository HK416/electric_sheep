//! Evaluation IR (spec 10): perturbation suite x metric x acceptance criterion.
//!
//! Data only. Sampling a perturbation, running an episode and computing a metric belong to
//! `es-eval` (layer 10); this module is the declaration those stages read, plus the checks
//! that can be made without running anything.
//!
//! Two rules of spec 10.4 are structural here rather than run-time checks:
//!
//! * Fairness — every perturbation is drawn from
//!   `TaskRng(seed_base, suite_id, episode_idx, stream)`, so a cell's perturbation sequence is
//!   fixed by [`EpisodeBatch`] and [`Perturbation::stream`] alone and cannot depend on which
//!   policy is under evaluation.
//! * INV-15 — augmentation is off during evaluation. [`AugmentationPolicy`] has no `Enabled`
//!   variant; the only way past `Disabled` is a named allow-list carrying a justification.
//!
//! One transport caveat: `evaluation_hash` survives a JSON round trip only if `serde_json` is
//! built with its `float_roundtrip` feature. Without it the parser lands within one ULP of the
//! written value, which is enough to move the hash of a perturbation range. The same document
//! read through a non-lossy transport hashes identically.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::codes;
use crate::diag::Diagnostic;
use crate::hash::CanonWriter;

/// Domain separator for [`EvaluationIr::evaluation_hash`].
const EVAL_TAG: &str = "es.ir.evaluation.v1";

// --- Ranges ------------------------------------------------------------------------------

/// A closed real interval, as written `range: [lo, hi]` in spec 10.2.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Range {
    pub lo: f64,
    pub hi: f64,
}

impl Range {
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    fn canonical(&self, w: &mut CanonWriter) {
        w.f64(self.lo);
        w.f64(self.hi);
    }
}

/// A closed integer interval, as written `count: [1, 3]` in spec 10.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountRange {
    pub lo: u32,
    pub hi: u32,
}

impl CountRange {
    pub fn new(lo: u32, hi: u32) -> Self {
        Self { lo, hi }
    }

    fn canonical(self, w: &mut CanonWriter) {
        w.u32(self.lo);
        w.u32(self.hi);
    }
}

/// How a range is sampled (spec 10.2 `dist`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distribution {
    #[default]
    Uniform,
    #[serde(rename = "loguniform")]
    LogUniform,
    Normal,
}

impl Distribution {
    pub fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::LogUniform => "loguniform",
            Self::Normal => "normal",
        }
    }
}

// --- Perturbations -----------------------------------------------------------------------

/// The perturbation kinds of spec 10.2, with their parameters.
///
/// Externally tagged, and never `#[serde(flatten)]`ed: serde's flatten buffers numbers through
/// an untyped `Content` and comes back a few ULPs off, which would move `evaluation_hash` on a
/// round trip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerturbationKind {
    LightIntensity {
        range: Range,
        #[serde(default)]
        dist: Distribution,
    },
    LightDirection {
        range_deg: f64,
    },
    ColorTemperature {
        range_k: Range,
    },
    CameraExtrinsic {
        pos_sigma_m: f64,
        rot_sigma_deg: f64,
    },
    CameraIntrinsic {
        focal_rel_sigma: f64,
    },
    ObjectPose {
        target: String,
        pos_sigma_m: f64,
        yaw_deg: f64,
    },
    Occluder {
        count: CountRange,
        size_m: Range,
    },
    ObservationDelay {
        ms: Vec<u32>,
    },
    ActionDelay {
        ms: Vec<u32>,
    },
    FrameDrop {
        prob: f64,
        burst: CountRange,
    },
    TorqueNoise {
        rel_sigma: f64,
    },
    Backlash {
        rad: Range,
    },
}

impl PerturbationKind {
    /// The spec 10.2 `kind` spelling. Also the hash input, so renaming a variant without
    /// renaming this string would silently keep old hashes valid.
    pub fn name(&self) -> &'static str {
        match self {
            Self::LightIntensity { .. } => "light_intensity",
            Self::LightDirection { .. } => "light_direction",
            Self::ColorTemperature { .. } => "color_temperature",
            Self::CameraExtrinsic { .. } => "camera_extrinsic",
            Self::CameraIntrinsic { .. } => "camera_intrinsic",
            Self::ObjectPose { .. } => "object_pose",
            Self::Occluder { .. } => "occluder",
            Self::ObservationDelay { .. } => "observation_delay",
            Self::ActionDelay { .. } => "action_delay",
            Self::FrameDrop { .. } => "frame_drop",
            Self::TorqueNoise { .. } => "torque_noise",
            Self::Backlash { .. } => "backlash",
        }
    }

    fn canonical(&self, w: &mut CanonWriter) {
        w.str(self.name());
        match self {
            Self::LightIntensity { range, dist } => {
                range.canonical(w);
                w.str(dist.name());
            }
            Self::LightDirection { range_deg } => w.f64(*range_deg),
            Self::ColorTemperature { range_k } => range_k.canonical(w),
            Self::CameraExtrinsic {
                pos_sigma_m,
                rot_sigma_deg,
            } => {
                w.f64(*pos_sigma_m);
                w.f64(*rot_sigma_deg);
            }
            Self::CameraIntrinsic { focal_rel_sigma } => w.f64(*focal_rel_sigma),
            Self::ObjectPose {
                target,
                pos_sigma_m,
                yaw_deg,
            } => {
                w.str(target);
                w.f64(*pos_sigma_m);
                w.f64(*yaw_deg);
            }
            Self::Occluder { count, size_m } => {
                count.canonical(w);
                size_m.canonical(w);
            }
            Self::ObservationDelay { ms } | Self::ActionDelay { ms } => {
                w.seq(ms.len());
                for v in ms {
                    w.u32(*v);
                }
            }
            Self::FrameDrop { prob, burst } => {
                w.f64(*prob);
                burst.canonical(w);
            }
            Self::TorqueNoise { rel_sigma } => w.f64(*rel_sigma),
            Self::Backlash { rad } => rad.canonical(w),
        }
    }

    fn check(&self, at: &str, out: &mut Vec<Diagnostic>) {
        // Every real interval, integer interval and bare scalar this kind carries.
        let (mut r, mut c, mut s): (Vec<Range>, Vec<CountRange>, Vec<f64>) =
            (Vec::new(), Vec::new(), Vec::new());
        match self {
            Self::LightIntensity { range, .. } => r.push(*range),
            Self::ColorTemperature { range_k } => r.push(*range_k),
            Self::Backlash { rad } => r.push(*rad),
            Self::LightDirection { range_deg } => s.push(*range_deg),
            Self::CameraExtrinsic {
                pos_sigma_m,
                rot_sigma_deg,
            } => s.extend([*pos_sigma_m, *rot_sigma_deg]),
            Self::CameraIntrinsic { focal_rel_sigma } => s.push(*focal_rel_sigma),
            Self::ObjectPose {
                pos_sigma_m,
                yaw_deg,
                ..
            } => s.extend([*pos_sigma_m, *yaw_deg]),
            Self::Occluder { count, size_m } => {
                r.push(*size_m);
                c.push(*count);
            }
            Self::FrameDrop { prob, burst } => {
                s.push(*prob);
                c.push(*burst);
            }
            Self::TorqueNoise { rel_sigma } => s.push(*rel_sigma),
            Self::ObservationDelay { .. } | Self::ActionDelay { .. } => {}
        }
        for r in r {
            if !r.lo.is_finite() || !r.hi.is_finite() {
                out.push(Diagnostic::new(
                    codes::EVAL_002,
                    format!("{at}: {} range is not finite", self.name()),
                ));
            } else if r.lo > r.hi {
                out.push(
                    Diagnostic::new(
                        codes::EVAL_005,
                        format!("{at}: {} range [{}, {}]", self.name(), r.lo, r.hi),
                    )
                    .with_hint("a range is written [lo, hi] with lo <= hi"),
                );
            }
        }
        for c in c {
            if c.lo > c.hi {
                out.push(Diagnostic::new(
                    codes::EVAL_005,
                    format!("{at}: {} count [{}, {}]", self.name(), c.lo, c.hi),
                ));
            }
        }
        for v in s {
            if !v.is_finite() {
                out.push(Diagnostic::new(
                    codes::EVAL_002,
                    format!("{at}: {} has a non-finite parameter", self.name()),
                ));
            }
        }
    }
}

/// One perturbation and the random stream it owns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Perturbation {
    pub kind: PerturbationKind,
    /// The `stream` index of `TaskRng(seed_base, suite_id, episode_idx, stream)` (spec 10.4).
    /// Two perturbations in one suite must not share a stream, or they draw the same numbers.
    pub stream: u32,
}

impl Perturbation {
    pub fn new(kind: PerturbationKind, stream: u32) -> Self {
        Self { kind, stream }
    }

    fn canonical(&self, w: &mut CanonWriter) {
        self.kind.canonical(w);
        w.u32(self.stream);
    }
}

/// A named row of the spec 10.1 table: the perturbations applied to every episode of the cell.
/// An empty list is the `nominal` suite.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerturbationSuite {
    pub name: String,
    #[serde(default)]
    pub perturbations: Vec<Perturbation>,
}

impl PerturbationSuite {
    fn canonical(&self, w: &mut CanonWriter) {
        w.str(&self.name);
        w.seq(self.perturbations.len());
        for p in &self.perturbations {
            p.canonical(w);
        }
    }
}

// --- Metrics -----------------------------------------------------------------------------

/// The metrics of spec 10.3 plus the nine-metric performance set of spec 12.4.
///
/// `chunk_underrun_rate` and the end-to-end latency quantiles appear in both tables and are
/// one variant each here. There is no single `step/s` metric on purpose (spec 12.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricSpec {
    // spec 10.3 outcome and safety metrics.
    SuccessRate,
    InterventionRate,
    CollisionRate,
    /// Fraction of steps the Safety Plane clamped or projected (spec 9.3).
    EnvelopeViolationRate,
    ActionSmoothness,
    EpisodeLength,
    FailureModeHistogram,
    /// Observation distribution distance on real-log replay (spec 24.3).
    DomainGap,
    // spec 12.4 performance set (`chunk_underrun_rate` and the latencies are shared).
    ChunkUnderrunRate,
    EndToEndLatencyP50,
    EndToEndLatencyP95,
    PhysicsStepsPerSec,
    CameraFramesPerSec,
    PixelsPerSec,
    ObservationGbPerSec,
    PolicyInferencesPerSec,
    ActionsPerSec,
    GpuMemoryPeak,
}

impl MetricSpec {
    /// The nine-metric performance set of spec 12.4; the latency row is two variants.
    pub const PERFORMANCE_SET: [Self; 10] = [
        Self::PhysicsStepsPerSec,
        Self::CameraFramesPerSec,
        Self::PixelsPerSec,
        Self::ObservationGbPerSec,
        Self::PolicyInferencesPerSec,
        Self::ActionsPerSec,
        Self::EndToEndLatencyP50,
        Self::EndToEndLatencyP95,
        Self::GpuMemoryPeak,
        Self::ChunkUnderrunRate,
    ];

    /// Every metric, for report layout and exhaustive iteration in tests.
    pub const ALL: [Self; 18] = [
        Self::SuccessRate,
        Self::InterventionRate,
        Self::CollisionRate,
        Self::EnvelopeViolationRate,
        Self::ActionSmoothness,
        Self::EpisodeLength,
        Self::FailureModeHistogram,
        Self::DomainGap,
        Self::ChunkUnderrunRate,
        Self::EndToEndLatencyP50,
        Self::EndToEndLatencyP95,
        Self::PhysicsStepsPerSec,
        Self::CameraFramesPerSec,
        Self::PixelsPerSec,
        Self::ObservationGbPerSec,
        Self::PolicyInferencesPerSec,
        Self::ActionsPerSec,
        Self::GpuMemoryPeak,
    ];

    /// The spec spelling, matching the serde name. Also the hash input.
    pub fn name(self) -> &'static str {
        match self {
            Self::SuccessRate => "success_rate",
            Self::InterventionRate => "intervention_rate",
            Self::CollisionRate => "collision_rate",
            Self::EnvelopeViolationRate => "envelope_violation_rate",
            Self::ActionSmoothness => "action_smoothness",
            Self::EpisodeLength => "episode_length",
            Self::FailureModeHistogram => "failure_mode_histogram",
            Self::DomainGap => "domain_gap",
            Self::ChunkUnderrunRate => "chunk_underrun_rate",
            Self::EndToEndLatencyP50 => "end_to_end_latency_p50",
            Self::EndToEndLatencyP95 => "end_to_end_latency_p95",
            Self::PhysicsStepsPerSec => "physics_steps_per_sec",
            Self::CameraFramesPerSec => "camera_frames_per_sec",
            Self::PixelsPerSec => "pixels_per_sec",
            Self::ObservationGbPerSec => "observation_gb_per_sec",
            Self::PolicyInferencesPerSec => "policy_inferences_per_sec",
            Self::ActionsPerSec => "actions_per_sec",
            Self::GpuMemoryPeak => "gpu_memory_peak",
        }
    }
}

// --- Acceptance --------------------------------------------------------------------------

/// The comparison of a spec 10.2 acceptance string such as `">= 0.85"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparator {
    Ge,
    Gt,
    Le,
    Lt,
}

impl Comparator {
    pub fn name(self) -> &'static str {
        match self {
            Self::Ge => ">=",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Lt => "<",
        }
    }

    pub fn holds(self, observed: f64, threshold: f64) -> bool {
        match self {
            Self::Ge => observed >= threshold,
            Self::Gt => observed > threshold,
            Self::Le => observed <= threshold,
            Self::Lt => observed < threshold,
        }
    }
}

/// How a metric is reduced over the episodes of a cell before the comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregation {
    #[default]
    Mean,
    Min,
    Max,
    P95,
}

impl Aggregation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Min => "min",
            Self::Max => "max",
            Self::P95 => "p95",
        }
    }
}

/// One line of the spec 10.2 `acceptance` block. `suite: None` is the spec's bare
/// `envelope_violation_rate: "<= 0.01"` form: the criterion applies to every suite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceCriterion {
    #[serde(default)]
    pub suite: Option<String>,
    pub metric: MetricSpec,
    pub comparator: Comparator,
    pub threshold: f64,
    #[serde(default)]
    pub aggregation: Aggregation,
}

impl AcceptanceCriterion {
    fn canonical(&self, w: &mut CanonWriter) {
        w.str(self.suite.as_deref().unwrap_or(""));
        w.bool(self.suite.is_some());
        w.str(self.metric.name());
        w.str(self.comparator.name());
        w.f64(self.threshold);
        w.str(self.aggregation.name());
    }
}

// --- Episodes and fairness ---------------------------------------------------------------

/// How the per-episode seeds are chosen (spec 10.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedPlan {
    /// `seed_base` of spec 10.2: episode `i` uses `TaskRng(base, suite_id, i, stream)`.
    Base(u64),
    /// One seed per episode, written out. Must be unique, or two episodes are the same run.
    Explicit(Vec<u64>),
}

/// The episode batch domain of spec 5.1: how many episodes per cell, and their seeds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeBatch {
    /// `episodes_per_cell` of spec 10.2.
    pub n_episodes: u32,
    pub seeds: SeedPlan,
}

impl EpisodeBatch {
    fn canonical(&self, w: &mut CanonWriter) {
        w.u32(self.n_episodes);
        match &self.seeds {
            SeedPlan::Base(b) => {
                w.str("base");
                w.u64(*b);
            }
            SeedPlan::Explicit(list) => {
                w.str("explicit");
                w.seq(list.len());
                for s in list {
                    w.u64(*s);
                }
            }
        }
    }

    fn check(&self, out: &mut Vec<Diagnostic>) {
        if self.n_episodes == 0 {
            out.push(Diagnostic::new(
                codes::EVAL_003,
                "n_episodes is 0; a cell with no episodes has no metric",
            ));
        }
        let SeedPlan::Explicit(list) = &self.seeds else {
            return;
        };
        if list.len() as u64 != u64::from(self.n_episodes) {
            out.push(Diagnostic::new(
                codes::EVAL_003,
                format!(
                    "{} explicit seeds for {} episodes",
                    list.len(),
                    self.n_episodes
                ),
            ));
        }
        let mut seen = BTreeSet::new();
        for s in list {
            if !seen.insert(*s) {
                out.push(
                    Diagnostic::new(codes::EVAL_004, format!("seed {s} appears twice"))
                        .with_hint("duplicate seeds replay the same episode and skew the metric"),
                );
            }
        }
    }
}

/// INV-15: augmentation nodes (spec 7.3) are disabled during evaluation.
///
/// There is deliberately no `Enabled` variant. Keeping a specific node on costs the author an
/// explicit list and a written reason, which is what shows up in the report.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AugmentationPolicy {
    #[default]
    Disabled,
    AllowList {
        nodes: BTreeSet<String>,
        justification: String,
    },
}

impl AugmentationPolicy {
    fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::Disabled => w.str("disabled"),
            Self::AllowList {
                nodes,
                justification,
            } => {
                w.str("allow_list");
                w.seq(nodes.len());
                for n in nodes {
                    w.str(n);
                }
                w.str(justification);
            }
        }
    }
}

/// Which episodes are kept for replay (spec 10.4, spec 10.5 `episodes/`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPolicy {
    #[default]
    FailuresFirst,
    All,
    None,
}

impl ReplayPolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::FailuresFirst => "failures_first",
            Self::All => "all",
            Self::None => "none",
        }
    }
}

// --- The IR ------------------------------------------------------------------------------

/// The spec 10.2 document: what is perturbed, what is measured, and what counts as passing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationIr {
    pub schema_version: u32,
    /// Reference to the Task IR under evaluation (spec 10.2 `task`).
    pub task: String,
    /// Reference to the Observation IR under evaluation (spec 10.2 `observation`).
    pub observation: String,
    pub episodes: EpisodeBatch,
    /// The rows of the spec 10.1 table. Order is semantic: it is the report's row order.
    pub suites: Vec<PerturbationSuite>,
    /// The columns of the spec 10.1 table.
    pub metrics: Vec<MetricSpec>,
    pub acceptance: Vec<AcceptanceCriterion>,
    /// INV-15. Defaults to [`AugmentationPolicy::Disabled`].
    #[serde(default)]
    pub augmentation: AugmentationPolicy,
    #[serde(default)]
    pub replay: ReplayPolicy,
}

impl EvaluationIr {
    /// Static checks. An empty result means the document is well formed; running it is
    /// `es-eval`'s problem.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        self.episodes.check(&mut out);

        let mut suite_names = BTreeSet::new();
        for suite in &self.suites {
            if !suite_names.insert(suite.name.as_str()) {
                out.push(Diagnostic::new(
                    codes::EVAL_004,
                    format!("suite \"{}\" is declared twice", suite.name),
                ));
            }
            let mut streams = BTreeSet::new();
            for p in &suite.perturbations {
                if !streams.insert(p.stream) {
                    out.push(
                        Diagnostic::new(
                            codes::EVAL_004,
                            format!(
                                "suite \"{}\": stream {} is used twice",
                                suite.name, p.stream
                            ),
                        )
                        .with_hint("perturbations sharing a stream draw the same numbers"),
                    );
                }
                p.kind.check(&format!("suite \"{}\"", suite.name), &mut out);
            }
        }

        let declared: BTreeSet<MetricSpec> = self.metrics.iter().copied().collect();
        for c in &self.acceptance {
            if !declared.contains(&c.metric) {
                out.push(
                    Diagnostic::new(
                        codes::EVAL_001,
                        format!(
                            "acceptance names metric {}, which is not in `metrics`",
                            c.metric.name()
                        ),
                    )
                    .with_hint("add the metric to `metrics` or drop the criterion"),
                );
            }
            if let Some(s) = &c.suite {
                if !suite_names.contains(s.as_str()) {
                    out.push(Diagnostic::new(
                        codes::EVAL_001,
                        format!("acceptance names suite \"{s}\", which is not declared"),
                    ));
                }
            }
            if !c.threshold.is_finite() {
                out.push(Diagnostic::new(
                    codes::EVAL_002,
                    format!("threshold for {} is not finite", c.metric.name()),
                ));
            }
        }

        if let AugmentationPolicy::AllowList { justification, .. } = &self.augmentation {
            if justification.trim().is_empty() {
                out.push(
                    Diagnostic::new(
                        codes::EVAL_006,
                        "augmentation allow-list has no justification",
                    )
                    .with_hint("INV-15: say why these nodes stay on during evaluation"),
                );
            }
        }
        out
    }

    /// `evaluation_hash` of spec 10.4: equal hash, equal evaluation conditions. Included in
    /// the report and in [`crate::HashChain`].
    pub fn evaluation_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let mut w = CanonWriter::new();
        w.str(EVAL_TAG);
        w.u32(self.schema_version);
        w.str(&self.task);
        w.str(&self.observation);
        self.episodes.canonical(&mut w);
        w.seq(self.suites.len());
        for s in &self.suites {
            s.canonical(&mut w);
        }
        w.seq(self.metrics.len());
        for m in &self.metrics {
            w.str(m.name());
        }
        w.seq(self.acceptance.len());
        for a in &self.acceptance {
            a.canonical(&mut w);
        }
        self.augmentation.canonical(&mut w);
        w.str(self.replay.name());
        w.hash()
    }
}

// --- Artifacts (spec 10.5) ---------------------------------------------------------------

/// One cell of the spec 10.1 table: a metric measured under one suite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellResult {
    pub suite: String,
    pub metric: MetricSpec,
    pub value: MetricValue,
    /// Episodes that actually contributed; quarantined envs drop out (spec 18.5).
    pub n_episodes: u32,
}

/// A metric aggregates either to a number or, for `failure_mode_histogram`, to counts per
/// failure cause (spec 10.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricValue {
    Scalar(f64),
    Histogram(BTreeMap<String, u64>),
}

/// One acceptance line's verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceResult {
    pub criterion: AcceptanceCriterion,
    pub observed: f64,
    pub passed: bool,
}

/// `report.json` of spec 10.5, and the content of `evaluation.lock`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub schema_version: u32,
    /// The conditions this report was produced under (spec 10.4).
    pub evaluation_hash: [u8; 32],
    /// The run that was evaluated (spec 5.3).
    pub execution_hash: [u8; 32],
    pub cells: Vec<CellResult>,
    pub acceptance: Vec<AcceptanceResult>,
    /// True when every entry of `acceptance` passed.
    pub passed: bool,
    /// Replay paths kept under `episodes/`, failures first (spec 10.5).
    #[serde(default)]
    pub episodes: Vec<String>,
}

#[cfg(any(test, feature = "testing"))]
pub mod testing {
    //! Proptest strategy for a well-formed [`EvaluationIr`] (Appendix B.7, P25).

    // `use super::*` is how a test-support module reads; clippy only exempts `#[cfg(test)]`
    // ones automatically, and this one is also reachable through `feature = "testing"`.
    #![allow(clippy::wildcard_imports)]

    use super::*;
    use proptest::prelude::*;

    fn arb_range() -> impl Strategy<Value = Range> {
        (-100.0f64..100.0, 0.0f64..100.0).prop_map(|(lo, span)| Range::new(lo, lo + span))
    }

    fn arb_count() -> impl Strategy<Value = CountRange> {
        (0u32..10, 0u32..10).prop_map(|(lo, span)| CountRange::new(lo, lo + span))
    }

    fn arb_kind() -> impl Strategy<Value = PerturbationKind> {
        prop_oneof![
            arb_range().prop_map(|range| PerturbationKind::LightIntensity {
                range,
                dist: Distribution::LogUniform
            }),
            (-180.0f64..180.0).prop_map(|range_deg| PerturbationKind::LightDirection { range_deg }),
            arb_range().prop_map(|range_k| PerturbationKind::ColorTemperature { range_k }),
            (0.0f64..1.0, 0.0f64..10.0).prop_map(|(pos_sigma_m, rot_sigma_deg)| {
                PerturbationKind::CameraExtrinsic {
                    pos_sigma_m,
                    rot_sigma_deg,
                }
            }),
            ("[a-z]{1,6}", 0.0f64..1.0, -180.0f64..180.0).prop_map(
                |(target, pos_sigma_m, yaw_deg)| PerturbationKind::ObjectPose {
                    target,
                    pos_sigma_m,
                    yaw_deg
                }
            ),
            (arb_count(), arb_range())
                .prop_map(|(count, size_m)| PerturbationKind::Occluder { count, size_m }),
            prop::collection::vec(0u32..100, 1..4)
                .prop_map(|ms| PerturbationKind::ObservationDelay { ms }),
            (0.0f64..1.0, arb_count())
                .prop_map(|(prob, burst)| PerturbationKind::FrameDrop { prob, burst }),
            arb_range().prop_map(|rad| PerturbationKind::Backlash { rad }),
        ]
    }

    fn arb_suite(idx: usize) -> impl Strategy<Value = PerturbationSuite> {
        prop::collection::vec(arb_kind(), 0..4).prop_map(move |kinds| PerturbationSuite {
            name: format!("suite_{idx}"),
            // Streams are handed out by position, so they are unique by construction.
            perturbations: kinds
                .into_iter()
                .enumerate()
                .map(|(i, k)| Perturbation::new(k, i as u32))
                .collect(),
        })
    }

    /// An [`EvaluationIr`] that always passes [`EvaluationIr::validate`].
    pub fn arbitrary_evaluation_ir() -> impl Strategy<Value = EvaluationIr> {
        (
            1usize..4,
            1u32..64,
            prop::collection::vec(0usize..MetricSpec::ALL.len(), 1..6),
            prop::collection::vec((any::<prop::sample::Index>(), -1.0f64..1.0), 0..4),
            any::<bool>(),
        )
            .prop_flat_map(
                |(n_suites, n_episodes, metric_idx, acceptance, explicit_seeds)| {
                    let suites: Vec<_> = (0..n_suites).map(arb_suite).collect();
                    let metrics: Vec<MetricSpec> = {
                        let mut m: Vec<_> =
                            metric_idx.into_iter().map(|i| MetricSpec::ALL[i]).collect();
                        m.sort_unstable();
                        m.dedup();
                        m
                    };
                    suites.prop_map(move |suites| {
                        let acceptance = acceptance
                            .iter()
                            .map(|(idx, threshold)| AcceptanceCriterion {
                                suite: Some(suites[idx.index(suites.len())].name.clone()),
                                metric: metrics[idx.index(metrics.len())],
                                comparator: Comparator::Ge,
                                threshold: *threshold,
                                aggregation: Aggregation::Mean,
                            })
                            .collect();
                        EvaluationIr {
                            schema_version: 1,
                            task: "tasks/pick_cube.toml".into(),
                            observation: "obs/two_view_224.toml".into(),
                            episodes: EpisodeBatch {
                                n_episodes,
                                seeds: if explicit_seeds {
                                    SeedPlan::Explicit((0..u64::from(n_episodes)).collect())
                                } else {
                                    SeedPlan::Base(20_260_912)
                                },
                            },
                            suites: suites.clone(),
                            metrics: metrics.clone(),
                            acceptance,
                            augmentation: AugmentationPolicy::Disabled,
                            replay: ReplayPolicy::FailuresFirst,
                        }
                    })
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use super::testing::arbitrary_evaluation_ir;
    use super::*;
    use proptest::prelude::*;

    /// The spec 10.2 example, trimmed to two suites.
    fn fixture() -> EvaluationIr {
        EvaluationIr {
            schema_version: 1,
            task: "tasks/pick_cube.toml".into(),
            observation: "obs/two_view_224.toml".into(),
            episodes: EpisodeBatch {
                n_episodes: 100,
                seeds: SeedPlan::Base(20_260_912),
            },
            suites: vec![
                PerturbationSuite {
                    name: "nominal".into(),
                    perturbations: vec![],
                },
                PerturbationSuite {
                    name: "lighting_shift".into(),
                    perturbations: vec![
                        Perturbation::new(
                            PerturbationKind::LightIntensity {
                                range: Range::new(0.3, 2.5),
                                dist: Distribution::LogUniform,
                            },
                            0,
                        ),
                        Perturbation::new(
                            PerturbationKind::ColorTemperature {
                                range_k: Range::new(2700.0, 7500.0),
                            },
                            1,
                        ),
                    ],
                },
            ],
            metrics: vec![
                MetricSpec::SuccessRate,
                MetricSpec::EnvelopeViolationRate,
                MetricSpec::EndToEndLatencyP95,
            ],
            acceptance: vec![
                AcceptanceCriterion {
                    suite: Some("nominal".into()),
                    metric: MetricSpec::SuccessRate,
                    comparator: Comparator::Ge,
                    threshold: 0.85,
                    aggregation: Aggregation::Mean,
                },
                AcceptanceCriterion {
                    suite: None,
                    metric: MetricSpec::EnvelopeViolationRate,
                    comparator: Comparator::Le,
                    threshold: 0.01,
                    aggregation: Aggregation::Mean,
                },
            ],
            augmentation: AugmentationPolicy::Disabled,
            replay: ReplayPolicy::FailuresFirst,
        }
    }

    fn codes(d: &[Diagnostic]) -> Vec<&str> {
        d.iter().map(|d| d.code.as_str()).collect()
    }

    #[test]
    fn serde_round_trip() {
        let ir = fixture();
        let json = serde_json::to_string(&ir).unwrap();
        assert_eq!(serde_json::from_str::<EvaluationIr>(&json).unwrap(), ir);
    }

    #[test]
    fn valid_fixture_is_clean() {
        assert_eq!(fixture().validate(), vec![]);
    }

    #[test]
    fn augmentation_defaults_to_disabled() {
        // INV-15: a document that says nothing about augmentation still has it off, and the
        // enum offers no way to turn it all back on.
        assert_eq!(AugmentationPolicy::default(), AugmentationPolicy::Disabled);
    }

    #[test]
    fn allow_list_needs_a_justification() {
        let mut ir = fixture();
        ir.augmentation = AugmentationPolicy::AllowList {
            nodes: ["color_jitter".to_owned()].into_iter().collect(),
            justification: "  ".into(),
        };
        assert_eq!(codes(&ir.validate()), [codes::EVAL_006]);
    }

    #[test]
    fn dangling_metric_and_suite_references_are_reported() {
        let mut ir = fixture();
        ir.acceptance[0].metric = MetricSpec::DomainGap; // not in `metrics`
        ir.acceptance[1].suite = Some("no_such_suite".into());
        assert_eq!(codes(&ir.validate()), [codes::EVAL_001, codes::EVAL_001]);
    }

    #[test]
    fn duplicate_seeds_are_reported() {
        let mut ir = fixture();
        ir.episodes = EpisodeBatch {
            n_episodes: 3,
            seeds: SeedPlan::Explicit(vec![7, 7, 9]),
        };
        assert_eq!(codes(&ir.validate()), [codes::EVAL_004]);
    }

    #[test]
    fn empty_batch_and_seed_count_mismatch_are_reported() {
        let mut ir = fixture();
        ir.episodes = EpisodeBatch {
            n_episodes: 0,
            seeds: SeedPlan::Explicit(vec![1, 2]),
        };
        assert_eq!(codes(&ir.validate()), [codes::EVAL_003, codes::EVAL_003]);
    }

    #[test]
    fn backwards_ranges_and_non_finite_values_are_reported() {
        let mut ir = fixture();
        ir.suites[1].perturbations[0].kind = PerturbationKind::LightIntensity {
            range: Range::new(2.5, 0.3),
            dist: Distribution::Uniform,
        };
        ir.suites[1].perturbations[1].kind = PerturbationKind::Occluder {
            count: CountRange::new(3, 1),
            size_m: Range::new(f64::INFINITY, 0.1),
        };
        ir.acceptance[0].threshold = f64::NAN;
        assert_eq!(
            codes(&ir.validate()),
            [
                codes::EVAL_005,
                codes::EVAL_002,
                codes::EVAL_005,
                codes::EVAL_002
            ]
        );
    }

    #[test]
    fn duplicate_streams_in_one_suite_are_reported() {
        let mut ir = fixture();
        ir.suites[1].perturbations[1].stream = 0;
        assert_eq!(codes(&ir.validate()), [codes::EVAL_004]);
    }

    #[test]
    fn hash_tracks_every_condition() {
        type Mutation = (&'static str, fn(&mut EvaluationIr));
        let base = fixture().evaluation_hash().unwrap();
        let mutations: Vec<Mutation> = vec![
            ("threshold", |ir| ir.acceptance[0].threshold = 0.86),
            ("comparator", |ir| {
                ir.acceptance[0].comparator = Comparator::Gt;
            }),
            ("aggregation", |ir| {
                ir.acceptance[0].aggregation = Aggregation::Min;
            }),
            ("suite scope", |ir| {
                ir.acceptance[1].suite = Some("nominal".into());
            }),
            ("perturbation range", |ir| {
                ir.suites[1].perturbations[0].kind = PerturbationKind::LightIntensity {
                    range: Range::new(0.3, 2.6),
                    dist: Distribution::LogUniform,
                };
            }),
            ("stream", |ir| ir.suites[1].perturbations[0].stream = 9),
            ("metrics", |ir| ir.metrics.push(MetricSpec::DomainGap)),
            ("episodes", |ir| ir.episodes.n_episodes = 101),
            ("seed base", |ir| ir.episodes.seeds = SeedPlan::Base(1)),
            ("augmentation", |ir| {
                ir.augmentation = AugmentationPolicy::AllowList {
                    nodes: ["color_jitter".to_owned()].into_iter().collect(),
                    justification: "sensor model under test".into(),
                };
            }),
            ("replay", |ir| ir.replay = ReplayPolicy::All),
            ("schema", |ir| ir.schema_version = 2),
        ];
        for (what, mutate) in mutations {
            let mut ir = fixture();
            mutate(&mut ir);
            assert_ne!(
                base,
                ir.evaluation_hash().unwrap(),
                "{what} did not reach evaluation_hash"
            );
        }
        assert_eq!(base, fixture().evaluation_hash().unwrap());
    }

    #[test]
    fn nan_threshold_is_refused_by_the_hash() {
        let mut ir = fixture();
        ir.acceptance[0].threshold = f64::NAN;
        assert_eq!(
            ir.evaluation_hash().unwrap_err().code.as_str(),
            crate::codes::HASH_001
        );
    }

    #[test]
    fn performance_set_is_the_nine_metrics_of_spec_12_4() {
        // The spec's table has nine rows; the latency row carries two quantiles.
        assert_eq!(MetricSpec::PERFORMANCE_SET.len(), 10);
        for m in MetricSpec::PERFORMANCE_SET {
            assert!(MetricSpec::ALL.contains(&m));
        }
        let mut names: Vec<_> = MetricSpec::ALL.iter().map(|m| m.name()).collect();
        names.sort_unstable();
        let unique = names.len();
        names.dedup();
        assert_eq!(names.len(), unique, "metric names must be unique");
    }

    #[test]
    fn report_round_trips() {
        let report = EvaluationReport {
            schema_version: 1,
            evaluation_hash: fixture().evaluation_hash().unwrap(),
            execution_hash: [7u8; 32],
            cells: vec![
                CellResult {
                    suite: "nominal".into(),
                    metric: MetricSpec::SuccessRate,
                    value: MetricValue::Scalar(0.92),
                    n_episodes: 100,
                },
                CellResult {
                    suite: "nominal".into(),
                    metric: MetricSpec::FailureModeHistogram,
                    value: MetricValue::Histogram(
                        [("timeout".to_owned(), 3), ("collision".to_owned(), 5)]
                            .into_iter()
                            .collect(),
                    ),
                    n_episodes: 100,
                },
            ],
            acceptance: vec![AcceptanceResult {
                criterion: fixture().acceptance[0].clone(),
                observed: 0.92,
                passed: true,
            }],
            passed: true,
            episodes: vec!["episodes/nominal_0007.esr".into()],
        };
        let json = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<EvaluationReport>(&json).unwrap(),
            report
        );
    }

    proptest! {
        #[test]
        fn arbitrary_ir_validates_clean(ir in arbitrary_evaluation_ir()) {
            prop_assert_eq!(ir.validate(), vec![]);
            prop_assert!(ir.evaluation_hash().is_ok());
        }
    }
}
