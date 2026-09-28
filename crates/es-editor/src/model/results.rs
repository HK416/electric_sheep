//! The Results step: what a finished evaluation says, in the shape the results screen draws it
//! (packet M12/Y8, design note `editor-redesign.md` 6.5).
//!
//! Nothing here measures anything. The verdict is `report.passed`, the per-suite numbers are
//! the report's own `success_rate` cells or the `episodes.json` rows `es-eval` wrote beside
//! it, and a comparison is a difference of those - and only between two reports of the same
//! `evaluation_hash`, because a number across different conditions reads as progress that
//! is not there (spec 13.3).

use std::collections::BTreeSet;

use es_core::FailureKind;
use es_eval::episodes::EpisodeRow;
use es_eval::metrics::{failure_name, violation_name};
use es_ir::evaluation::{
    EvaluationIr, EvaluationReport, MetricSpec, MetricValue, PerturbationKind,
};
use es_safety::ViolationKind;

/// Why an episode failed, in the words a person reads - one per group of histogram buckets.
/// The order is the tie-break of [`causes`] and a tile's pick: how the episode ended first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cause {
    Timeout,
    FailureCondition,
    Unfinished,
    SafetyLimit,
    SafetyFallback,
    MotionGaps,
    TooLate,
    Unstable,
    SensorDrop,
    ActuatorFault,
    BackendUnsupported,
}

impl Cause {
    pub const ALL: [Cause; 11] = [
        Cause::Timeout,
        Cause::FailureCondition,
        Cause::Unfinished,
        Cause::SafetyLimit,
        Cause::SafetyFallback,
        Cause::MotionGaps,
        Cause::TooLate,
        Cause::Unstable,
        Cause::SensorDrop,
        Cause::ActuatorFault,
        Cause::BackendUnsupported,
    ];
}

/// A histogram bucket's cause. `success` is `None`, and so is a name this build does not know,
/// which the screen shows raw.
///
/// Total over every name `es-eval` writes (`metrics::failure_histogram`): the four
/// terminations and `fallback` are spelt here, because `es-eval` spells them inline; every
/// [`FailureKind`] and [`ViolationKind`] is found through `es-eval`'s own name for it and
/// mapped by a match with one arm per variant.
pub fn cause_of(bucket: &str) -> Option<Cause> {
    match bucket {
        "success" => None,
        "failure" => Some(Cause::FailureCondition),
        "timeout" => Some(Cause::Timeout),
        "unfinished" => Some(Cause::Unfinished),
        "fallback" => Some(Cause::SafetyFallback),
        _ => FAILURE_KINDS
            .into_iter()
            .find(|&k| failure_name(k) == bucket)
            .map(failure_cause)
            .or_else(|| {
                ViolationKind::ALL
                    .into_iter()
                    .find(|&k| violation_name(k) == bucket)
                    .map(violation_cause)
            }),
    }
}

fn failure_cause(kind: FailureKind) -> Cause {
    match kind {
        FailureKind::NanDetected | FailureKind::Diverged => Cause::Unstable,
        FailureKind::DeadlineMiss => Cause::TooLate,
        FailureKind::SensorDrop => Cause::SensorDrop,
        FailureKind::ActuatorFault => Cause::ActuatorFault,
        FailureKind::ChunkUnderrun => Cause::MotionGaps,
        FailureKind::SafetyViolation => Cause::SafetyLimit,
        FailureKind::BackendUnsupported => Cause::BackendUnsupported,
    }
}

fn violation_cause(kind: ViolationKind) -> Cause {
    match kind {
        // A NaN or infinite command cannot be clamped: the numbers themselves broke.
        ViolationKind::NonFinite => Cause::Unstable,
        ViolationKind::Position
        | ViolationKind::Velocity
        | ViolationKind::Acceleration
        | ViolationKind::Torque
        | ViolationKind::Workspace
        | ViolationKind::RateLimit
        | ViolationKind::ViolationRate => Cause::SafetyLimit,
        ViolationKind::StaleObservation | ViolationKind::SensorDropout => Cause::SensorDrop,
        ViolationKind::InferenceDeadline | ViolationKind::HeartbeatLoss => Cause::TooLate,
        ViolationKind::ChunkUnderrun => Cause::MotionGaps,
        ViolationKind::EstopLatched => Cause::SafetyFallback,
    }
}

/// Every [`FailureKind`]. `es-core` has no `ALL`; a variant added there breaks
/// [`failure_cause`]'s build, and belongs in this list too.
const FAILURE_KINDS: [FailureKind; 8] = [
    FailureKind::NanDetected,
    FailureKind::Diverged,
    FailureKind::DeadlineMiss,
    FailureKind::SensorDrop,
    FailureKind::ActuatorFault,
    FailureKind::ChunkUnderrun,
    FailureKind::SafetyViolation,
    FailureKind::BackendUnsupported,
];

const SUCCESS: &str = "success";

/// One episode's causes, each once however many of its buckets name it.
fn row_causes(row: &EpisodeRow) -> BTreeSet<Cause> {
    row.histogram.keys().filter_map(|b| cause_of(b)).collect()
}

/// The headline: pass or fail, and successes over all episodes ("9 of 16").
#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    pub passed: bool,
    pub successes: u32,
    pub episodes: u32,
}

/// The verdict is `report.passed`; the counts are the sum of [`situations`].
pub fn card(report: &EvaluationReport, rows: Option<&[EpisodeRow]>) -> Card {
    let (successes, episodes) = situations(report, rows, None)
        .iter()
        .fold((0, 0), |(s, e), x| (s + x.successes, e + x.episodes));
    Card {
        passed: report.passed,
        successes,
        episodes,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Comparison {
    NoPrevious,
    /// Different `evaluation_hash`es - or a report with no measured episode, which has no
    /// success fraction to subtract.
    NotComparable,
    Delta {
        success_points: f64,
    },
}

/// Success fraction difference in percentage points, only for an equal `evaluation_hash`.
pub fn compare(current: &EvaluationReport, previous: Option<&EvaluationReport>) -> Comparison {
    let Some(previous) = previous else {
        return Comparison::NoPrevious;
    };
    if current.evaluation_hash != previous.evaluation_hash {
        return Comparison::NotComparable;
    }
    let fraction = |report| {
        let c = card(report, None);
        (c.episodes > 0).then(|| f64::from(c.successes) / f64::from(c.episodes))
    };
    match (fraction(current), fraction(previous)) {
        (Some(now), Some(before)) => Comparison::Delta {
            success_points: (now - before) * 100.0,
        },
        _ => Comparison::NotComparable,
    }
}

/// Failed episodes per cause, most first, then `Cause` order; an episode with two causes
/// counts once under each.
pub fn causes(rows: &[EpisodeRow]) -> Vec<(Cause, u32)> {
    let mut counts = [0u32; Cause::ALL.len()];
    for row in rows.iter().filter(|r| r.termination != SUCCESS) {
        for cause in row_causes(row) {
            counts[cause as usize] += 1;
        }
    }
    let mut out: Vec<(Cause, u32)> = Cause::ALL
        .into_iter()
        .zip(counts)
        .filter(|&(_, n)| n > 0)
        .collect();
    // Stable: equal counts keep `Cause::ALL`'s order.
    out.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    out
}

/// One bar of "per situation".
#[derive(Clone, Debug, PartialEq)]
pub struct Situation {
    pub suite: String,
    /// The suite's perturbation kinds, each once, in the document's order.
    pub kinds: Vec<PerturbationKind>,
    pub successes: u32,
    pub episodes: u32,
}

/// Per suite in the evaluation's own order. With rows: counted from them. Without (an old
/// run): from the report's `success_rate` value x `n_episodes`, rounded - and a suite whose
/// success rate was not measured is 0 of 0, never a made-up count. `kinds` empty = nominal;
/// `ir == None` leaves `kinds` empty and the UI shows the raw suite name.
pub fn situations(
    report: &EvaluationReport,
    rows: Option<&[EpisodeRow]>,
    ir: Option<&EvaluationIr>,
) -> Vec<Situation> {
    // The evaluation's order, then any suite only the report or the rows name.
    let mut suites: Vec<&str> = Vec::new();
    let names = ir
        .into_iter()
        .flat_map(|ir| ir.suites.iter().map(|s| s.name.as_str()))
        .chain(report.cells.iter().map(|c| c.suite.as_str()))
        .chain(rows.into_iter().flatten().map(|r| r.suite.as_str()));
    for name in names {
        if !suites.contains(&name) {
            suites.push(name);
        }
    }

    suites
        .into_iter()
        .map(|suite| {
            let mut kinds: Vec<PerturbationKind> = Vec::new();
            let declared = ir.and_then(|ir| ir.suites.iter().find(|s| s.name == suite));
            for p in declared.into_iter().flat_map(|s| &s.perturbations) {
                if !kinds.iter().any(|k| k.name() == p.kind.name()) {
                    kinds.push(p.kind.clone());
                }
            }
            let (successes, episodes) = match rows {
                Some(rows) => rows
                    .iter()
                    .filter(|r| r.suite == suite)
                    .fold((0, 0), |(s, e), r| {
                        (s + u32::from(r.termination == SUCCESS), e + 1)
                    }),
                None => report
                    .cells
                    .iter()
                    .find(|c| c.suite == suite && c.metric == MetricSpec::SuccessRate)
                    .and_then(|c| match c.value {
                        MetricValue::Scalar(rate) => Some((
                            (rate * f64::from(c.n_episodes)).round() as u32,
                            c.n_episodes,
                        )),
                        MetricValue::Histogram(_) | MetricValue::Unavailable { .. } => None,
                    })
                    .unwrap_or((0, 0)),
            };
            Situation {
                suite: suite.to_owned(),
                kinds,
                successes,
                episodes,
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFilter {
    All,
    Successes,
    Failures,
}

/// One episode on the tile wall.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    pub cell: String,
    pub suite: String,
    pub success: bool,
    /// A failure's first cause in `Cause` order - how it ended; `None` for a success.
    pub cause: Option<Cause>,
}

pub fn tiles(rows: &[EpisodeRow], filter: TileFilter) -> Vec<Tile> {
    rows.iter()
        .filter_map(|row| {
            let success = row.termination == SUCCESS;
            let shown = match filter {
                TileFilter::All => true,
                TileFilter::Successes => success,
                TileFilter::Failures => !success,
            };
            shown.then(|| Tile {
                cell: row.cell.clone(),
                suite: row.suite.clone(),
                success,
                cause: if success {
                    None
                } else {
                    row_causes(row).first().copied()
                },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use es_eval::metrics::violation_name as violation_bucket;
    use es_ir::evaluation::{
        AugmentationPolicy, CellResult, Distribution, EpisodeBatch, Perturbation,
        PerturbationSuite, Range, ReplayPolicy, SeedPlan,
    };

    fn row(suite: &str, ep: u64, termination: &str, extra: &[(&str, u64)]) -> EpisodeRow {
        let mut histogram: BTreeMap<String, u64> = [(termination.to_owned(), 1)].into();
        for (k, n) in extra {
            histogram.insert((*k).to_owned(), *n);
        }
        EpisodeRow {
            suite: suite.into(),
            cell: format!("{suite}-{ep:02}"),
            episode: ep,
            seed: 100 + ep,
            termination: termination.into(),
            steps: 10,
            changed_steps: 0,
            histogram,
        }
    }

    fn hash(hex: &str) -> [u8; 32] {
        [u8::from_str_radix(hex, 16).unwrap(); 32]
    }

    /// One suite `nominal`, four episodes, a `success_rate` cell, passed.
    fn report_with(success_rate: f64, hash_hex: &str) -> EvaluationReport {
        EvaluationReport {
            schema_version: 1,
            evaluation_hash: hash(hash_hex),
            execution_hash: [0; 32],
            cells: vec![CellResult {
                suite: "nominal".into(),
                metric: MetricSpec::SuccessRate,
                value: MetricValue::Scalar(success_rate),
                n_episodes: 4,
            }],
            acceptance: Vec::new(),
            passed: true,
            episodes: Vec::new(),
        }
    }

    #[test]
    fn every_bucket_es_eval_writes_has_a_cause_or_is_success() {
        assert_eq!(cause_of("success"), None);
        for t in ["failure", "timeout", "unfinished", "fallback"] {
            assert!(cause_of(t).is_some(), "{t}");
        }
        for k in es_safety::ViolationKind::ALL {
            assert!(cause_of(violation_bucket(k)).is_some(), "{k:?}");
        }
        for f in FAILURE_KINDS.map(failure_name) {
            assert!(cause_of(f).is_some(), "{f}");
        }
        assert_eq!(cause_of("invented_by_a_future_run"), None, "shown raw");
    }

    #[test]
    fn causes_count_failed_episodes_most_first() {
        let rows = [
            row("nominal", 0, "timeout", &[]),
            row("nominal", 1, "failure", &[("fallback", 2)]),
            row("nominal", 2, "timeout", &[]),
            row("nominal", 3, "success", &[]),
        ];
        assert_eq!(
            causes(&rows),
            [
                (Cause::Timeout, 2),
                (Cause::FailureCondition, 1),
                (Cause::SafetyFallback, 1)
            ]
        );
        // Two buckets of one cause count the episode once; a success's clamps are not a cause.
        let rows = [
            row(
                "nominal",
                0,
                "failure",
                &[("safety_violation", 3), ("violation.position", 3)],
            ),
            row("nominal", 1, "success", &[("violation.torque", 1)]),
        ];
        assert_eq!(
            causes(&rows),
            [(Cause::FailureCondition, 1), (Cause::SafetyLimit, 1)]
        );
    }

    #[test]
    fn different_evaluation_hash_is_not_comparable() {
        let (a, mut b) = (report_with(0.5, "aa"), report_with(0.25, "aa"));
        assert_eq!(
            compare(&a, Some(&b)),
            Comparison::Delta {
                success_points: 25.0
            }
        );
        b.evaluation_hash = hash("bb");
        assert_eq!(compare(&a, Some(&b)), Comparison::NotComparable);
        assert_eq!(compare(&a, None), Comparison::NoPrevious);
    }

    #[test]
    fn old_run_falls_back_to_suite_metrics() {
        let report = report_with(0.75, "aa");
        let s = situations(&report, None, None);
        assert_eq!((s[0].successes, s[0].episodes), (3, 4));
        assert!(s[0].kinds.is_empty());
        assert!(tiles(&[], TileFilter::All).is_empty());
        assert_eq!(
            card(&report, None),
            Card {
                passed: report.passed,
                successes: 3,
                episodes: 4
            }
        );
        // An unmeasured success rate is no bar, never a made-up zero out of four.
        let mut unmeasured = report;
        unmeasured.cells[0].value = MetricValue::Unavailable {
            reason: "no episode finished in this cell".into(),
        };
        let s = situations(&unmeasured, None, None);
        assert_eq!((s[0].successes, s[0].episodes), (0, 0));
    }

    #[test]
    fn situations_follow_the_evaluation_and_count_rows() {
        let light = Perturbation::new(
            PerturbationKind::LightIntensity {
                range: Range::new(0.5, 1.5),
                dist: Distribution::Uniform,
            },
            0,
        );
        let ir = EvaluationIr {
            schema_version: 1,
            task: "task.toml".into(),
            observation: "observation.toml".into(),
            episodes: EpisodeBatch {
                n_episodes: 2,
                seeds: SeedPlan::Base(0),
            },
            suites: vec![
                PerturbationSuite {
                    name: "dim".into(),
                    perturbations: vec![light.clone(), light],
                },
                PerturbationSuite {
                    name: "nominal".into(),
                    perturbations: Vec::new(),
                },
            ],
            metrics: vec![MetricSpec::SuccessRate],
            acceptance: Vec::new(),
            augmentation: AugmentationPolicy::Disabled,
            replay: ReplayPolicy::default(),
        };
        let rows = [
            row("nominal", 0, "success", &[]),
            row("nominal", 1, "timeout", &[]),
            row("dim", 0, "failure", &[]),
            row("dim", 1, "failure", &[]),
        ];
        let s = situations(&report_with(0.5, "aa"), Some(&rows), Some(&ir));
        let got: Vec<_> = s
            .iter()
            .map(|s| (s.suite.as_str(), s.kinds.len(), s.successes, s.episodes))
            .collect();
        // The evaluation's order; one kind named once however many perturbations share it.
        assert_eq!(got, [("dim", 1, 0, 2), ("nominal", 0, 1, 2)]);
        assert_eq!(
            card(&report_with(0.5, "aa"), Some(&rows)),
            Card {
                passed: true,
                successes: 1,
                episodes: 4
            }
        );
    }

    #[test]
    fn tiles_filter_and_name_their_cause() {
        let rows = [
            row("nominal", 0, "success", &[]),
            row("nominal", 1, "timeout", &[]),
        ];
        assert_eq!(tiles(&rows, TileFilter::Failures).len(), 1);
        assert_eq!(
            tiles(&rows, TileFilter::Failures)[0].cause,
            Some(Cause::Timeout)
        );
        assert_eq!(tiles(&rows, TileFilter::Successes)[0].cell, "nominal-00");
        assert_eq!(tiles(&rows, TileFilter::Successes)[0].cause, None);
        assert_eq!(tiles(&rows, TileFilter::All).len(), 2);
    }
}
