//! The Results step: what a finished evaluation says, in the shape the results screen draws it
//! (packet M12/Y8, design note `editor-redesign.md` 6.5).
//!
//! Nothing here measures anything. The verdict is `report.passed`, the per-suite numbers are
//! the report's own `success_rate` cells or the `episodes.json` rows `es-eval` wrote beside
//! it, and a comparison is a difference of those - and only between two reports of the same
//! `evaluation_hash`, because a number across different conditions reads as progress that
//! is not there (spec 13.3).
//!
//! The screen's own choices are here too (packet M12/Y13): which run is shown and which it is
//! compared with, each acceptance line in plain words, a suite's plain name, the views an
//! attempt can be played in and the one it opens on, and which policy file Export copies.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use es_core::FailureKind;
use es_data::collect::{read_loop_steps, LoopKind, CHECKPOINT};
use es_data::training::{collect_root, lerobot_checkpoint, Cycle, Route};
use es_eval::episodes::{read_episodes, EpisodeRow};
use es_eval::metrics::{failure_name, violation_name};
use es_eval::perturb::unseen_age;
use es_eval::run_dir::{CellRow, RunDir};
use es_ir::deployment::DeploymentIr;
use es_ir::evaluation::{
    AcceptanceResult, Comparator, EvaluationIr, EvaluationReport, MetricSpec, MetricValue,
    PerturbationKind,
};
use es_ir::serial::{deployment_from_toml, evaluation_from_toml};
use es_safety::ViolationKind;

use crate::model::i18n::{fill, t, Lang, Strings};
use crate::model::labels::{cause_key, metric_label, perturbation_key};
use crate::model::outcome::{self, Outcome};
use crate::model::project::{Project, RunFolder, StartSettings, RUN_RECIPE};
use crate::model::teacher::{self, Score};
use crate::model::template::{load, Length, OutcomeKind, OutcomeSpec, Template};

/// Why an episode failed, in the words a person reads - one per group of histogram buckets.
/// The order is the tie-break of [`causes`] and a tile's pick: how the episode ended first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cause {
    /// A timeout, told apart by where the object went (packet M13/Z4): it takes `Timeout`'s
    /// place whenever the attempt's trajectory says it.
    Outcome(Outcome),
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
    pub const ALL: [Cause; 15] = [
        Cause::Outcome(Outcome::NeverLifted),
        Cause::Outcome(Outcome::LeftOutside),
        Cause::Outcome(Outcome::NotReleased),
        Cause::Outcome(Outcome::InsideTooLate),
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

/// The share of an attempt's steps, in percent, that one of the Safety Plane's step counters
/// (`fallback` and every `violation.*`: steps, not endings) must cover before it is a cause of
/// that attempt (review of plan Z, R3).
///
/// With a declared latency of one control period no chunk exists at tick 0, so every attempt of
/// the hint card's first real run -- its successes too -- counted one `fallback`, one
/// `violation.chunk_underrun` and a few `violation.acceleration` steps while the first chunk
/// arrived: the start, not why it failed. 1 % of that run's 1,800-step horizon is 18 steps:
/// above a start-up's handful, far below a policy that fights the envelope (hundreds). On its
/// 90 failed attempts it leaves "not done in time" on all 90, a safety limit on 43 and the
/// fallback on the 4 that fell back for 25 to 153 steps, where counting every bucket read 90
/// fallbacks, 90 motion gaps and 87 safety limits. How an attempt ended (the termination
/// buckets) and a failure the backend reported (a `FailureKind`) always count.
const STEP_SHARE_PERCENT: u64 = 1;

/// A Safety Plane step counter, as `es_eval::metrics` names them: counted per step, not per
/// ending.
fn per_step(bucket: &str) -> bool {
    bucket == "fallback" || bucket.starts_with("violation.")
}

/// One episode's causes, each once however many of its buckets name it; its outcome class, when
/// there is one, in place of `Timeout`. A failure something else ended keeps that cause. A step
/// counter below [`STEP_SHARE_PERCENT`] of the attempt's steps is no cause, so an attempt whose
/// only other buckets are that small keeps how it ended.
fn row_causes(row: &EpisodeRow, outcome: Option<Outcome>) -> BTreeSet<Cause> {
    let covers = |n: u64| n.saturating_mul(100) >= row.steps.saturating_mul(STEP_SHARE_PERCENT);
    let mut causes: BTreeSet<Cause> = (row.histogram.iter())
        .filter(|(bucket, n)| !per_step(bucket) || covers(**n))
        .filter_map(|(bucket, _)| cause_of(bucket))
        .collect();
    if let Some(o) = outcome {
        if causes.remove(&Cause::Timeout) {
            causes.insert(Cause::Outcome(o));
        }
    }
    causes
}

/// The outcome classes of a run's attempts, by cell ([`outcome::outcomes`]).
pub type Outcomes = BTreeMap<String, Outcome>;

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
/// counts once under each, and one with an outcome class under the class.
pub fn causes(rows: &[EpisodeRow], outcomes: &Outcomes) -> Vec<(Cause, u32)> {
    let mut counts: BTreeMap<Cause, u32> = BTreeMap::new();
    for row in rows.iter().filter(|r| r.termination != SUCCESS) {
        for cause in row_causes(row, outcomes.get(&row.cell).copied()) {
            *counts.entry(cause).or_default() += 1;
        }
    }
    let mut out: Vec<(Cause, u32)> = counts.into_iter().collect();
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TileFilter {
    #[default]
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
    /// A failure's first cause in `Cause` order - how it ended, its outcome class first;
    /// `None` for a success.
    pub cause: Option<Cause>,
}

pub fn tiles(rows: &[EpisodeRow], filter: TileFilter, outcomes: &Outcomes) -> Vec<Tile> {
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
                    row_causes(row, outcomes.get(&row.cell).copied())
                        .first()
                        .copied()
                },
            })
        })
        .collect()
}

// --- the screen (packet M12/Y13) --------------------------------------------------------------

impl TileFilter {
    pub const ALL: [TileFilter; 3] = [TileFilter::All, TileFilter::Successes, TileFilter::Failures];

    pub fn key(self) -> &'static str {
        match self {
            TileFilter::All => "results.filter.all",
            TileFilter::Successes => "results.filter.successes",
            TileFilter::Failures => "results.filter.failures",
        }
    }
}

impl Card {
    pub fn verdict_key(&self) -> &'static str {
        if self.passed {
            "results.card.passed"
        } else {
            "results.card.failed"
        }
    }
}

/// How the player shows an attempt: the recorded motion re-posed on the scene and seen from a
/// camera the person turns, the pictures the policy was given, or both beside each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Outside,
    Eye,
    SideBySide,
}

impl View {
    pub fn key(self) -> &'static str {
        match self {
            View::Outside => "results.view.outside",
            View::Eye => "results.view.eye",
            View::SideBySide => "results.view.both",
        }
    }
}

/// The views an attempt can be played in, the first being the one it opens in: outside needs
/// its motion (a trajectory and a scene that loaded), the policy's eye its pictures, side by
/// side both. An attempt that kept neither has none.
pub fn views(motion: bool, frames: usize) -> Vec<View> {
    match (motion, frames > 0) {
        (true, true) => vec![View::Outside, View::Eye, View::SideBySide],
        (true, false) => vec![View::Outside],
        (false, true) => vec![View::Eye],
        (false, false) => Vec::new(),
    }
}

/// The playback speeds offered, as multiples of the run's own rate.
pub const SPEEDS: [f64; 3] = [0.5, 1.0, 2.0];

/// The recorded picture shown at playback index `tick`. `es eval run` pushes the trajectory
/// where it captures the frame, so the two share an index; past the last picture, the last.
pub fn frame_at(tick: usize, frames: usize) -> Option<usize> {
    frames.checked_sub(1).map(|last| tick.min(last))
}

/// The attempt the player opens on: the first failure (spec 10.5 replays failures first), else
/// the first attempt. An old run has no rows: its first cell that kept its motion, else its
/// first cell.
pub fn first_to_play(rows: Option<&[EpisodeRow]>, cells: &[CellRow]) -> Option<String> {
    match rows {
        Some(rows) => rows
            .iter()
            .find(|r| r.termination != SUCCESS)
            .or(rows.first())
            .map(|r| r.cell.clone()),
        None => cells
            .iter()
            .find(|c| c.has_traj)
            .or(cells.first())
            .map(|c| c.name.clone()),
    }
}

/// A suite's plain name: its perturbation kinds' names, each once, in the document's order;
/// `results.nominal` for a suite that perturbs nothing. A suite the Evaluation IR does not
/// declare - or every suite, when the IR could not be read - keeps its raw name.
pub fn suite_label(lang: Lang, suite: &str, ir: Option<&EvaluationIr>) -> String {
    let Some(declared) = ir.and_then(|ir| ir.suites.iter().find(|s| s.name == suite)) else {
        return suite.to_owned();
    };
    let mut words: Vec<&str> = Vec::new();
    for p in &declared.perturbations {
        let word = t(lang, perturbation_key(&p.kind));
        if !words.contains(&word) {
            words.push(word);
        }
    }
    if words.is_empty() {
        t(lang, "results.nominal").to_owned()
    } else {
        words.join(", ")
    }
}

/// One acceptance line: passed, failed or not measured (`None`), the plain sentence, and the
/// criterion as the Evaluation IR spells it, for the hover.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub passed: Option<bool>,
    pub text: String,
    pub raw: String,
}

fn comparator_key(comparator: Comparator) -> &'static str {
    match comparator {
        Comparator::Ge => "results.ge",
        Comparator::Gt => "results.gt",
        Comparator::Le => "results.le",
        Comparator::Lt => "results.lt",
    }
}

/// At most three decimals, and none that are only zeros: `0.5`, `0.688`, `12`.
fn number(x: f64) -> String {
    let s = format!("{x:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// "Every situation: Success rate at least 0.5 (measured 0.688)" - the report's own verdict
/// and number, never recomputed.
pub fn acceptance_line(lang: Lang, line: &AcceptanceResult, ir: Option<&EvaluationIr>) -> Line {
    match line {
        AcceptanceResult::Determined {
            criterion: c,
            observed,
            passed,
        } => {
            let place = c.suite.as_deref().map_or_else(
                || t(lang, "results.everywhere").to_owned(),
                |suite| suite_label(lang, suite, ir),
            );
            let condition = fill(
                lang,
                comparator_key(c.comparator),
                &[metric_label(lang, c.metric), &number(c.threshold)],
            );
            Line {
                passed: Some(*passed),
                text: fill(
                    lang,
                    "results.criterion",
                    &[&place, &condition, &number(*observed)],
                ),
                raw: format!(
                    "{} {} {} ({}, {})",
                    c.metric.name(),
                    c.comparator.name(),
                    c.threshold,
                    c.aggregation.name(),
                    c.suite.as_deref().unwrap_or("*")
                ),
            }
        }
        AcceptanceResult::Unavailable { metric, reason } => Line {
            passed: None,
            text: format!(
                "{}: {} ({reason})",
                metric_label(lang, *metric),
                t(lang, "results.not_measured")
            ),
            raw: metric.name().to_owned(),
        },
    }
}

/// The change from run `previous`, in percentage points with its sign; "not comparable" for
/// other conditions; nothing when there is no earlier run.
pub fn change_text(lang: Lang, comparison: &Comparison, previous: u32) -> Option<String> {
    match comparison {
        Comparison::NoPrevious => None,
        Comparison::NotComparable => Some(t(lang, "results.not_comparable").to_owned()),
        Comparison::Delta { success_points } => Some(fill(
            lang,
            "results.change",
            &[&format!("{success_points:+.1}"), &format!("{previous:03}")],
        )),
    }
}

/// The runs ⑤ can show: those with an `eval/report.json`, ascending.
pub fn finished_runs(project: &Project) -> Vec<RunFolder> {
    project
        .runs()
        .into_iter()
        .filter(|r| r.report_path().is_file())
        .collect()
}

/// The run on screen: the one picked from the list while it is there, else the newest.
pub fn shown(runs: &[RunFolder], chosen: Option<u32>) -> Option<&RunFolder> {
    runs.iter()
        .find(|r| Some(r.number) == chosen)
        .or(runs.last())
}

/// The policy file to export: the checkpoint the evaluation judged - the recipe's
/// `[eval] checkpoint` when it names a mark that is on disk - else the newest
/// `train/checkpoints/<mark>.esb` (`"last"`, or the recipe is gone). `None` before training
/// wrote one.
pub fn export_bundle(run: &RunFolder, cycle: Option<&Cycle>) -> Option<PathBuf> {
    let dir = run.path.join("train").join("checkpoints");
    let evaluated = cycle
        .and_then(|c| c.eval.checkpoint.parse::<u32>().ok())
        .map(|mark| dir.join(format!("{mark}.esb")))
        .filter(|p| p.is_file());
    evaluated.or_else(|| {
        std::fs::read_dir(&dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "esb"))
            .filter_map(|p| Some((p.file_stem()?.to_str()?.parse::<u32>().ok()?, p)))
            .max_by_key(|(mark, _)| *mark)
            .map(|(_, p)| p)
    })
}

/// The file name the save dialog suggests: the project folder, the run and the mark, so two
/// exports never suggest the same name.
pub fn export_name(project: &Project, run: &RunFolder, bundle: &Path) -> String {
    let stem = |p: &Path| p.file_stem().map(|s| s.to_string_lossy().into_owned());
    let folder = project
        .root
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    format!(
        "{folder}-{:03}-{}.esb",
        run.number,
        stem(bundle).unwrap_or_default()
    )
}

/// What Run again starts ③ with: the run's `[collect] episodes` and the preset whose marks its
/// inline `[train] run` holds. `None` for a recipe the editor did not write.
pub fn start_settings(cycle: &Cycle, template: &Template) -> Option<StartSettings> {
    let demonstrations = cycle.collect.as_ref()?.episodes;
    let marks = &cycle.train.run.as_ref()?.checkpoint_at;
    let length = [Length::Short, Length::Medium, Length::Long]
        .into_iter()
        .find(|l| template.marks(*l) == marks.as_slice())?;
    Some(StartSettings {
        demonstrations,
        length,
    })
}

/// What the outside camera re-poses a run's motion on: the recipe's scene, else the
/// template's, under the repository root. ⑤'s player and ③'s preview player both use it.
pub fn scene(
    cycle: Option<&Cycle>,
    template: Option<&Template>,
    repo_root: Option<&Path>,
) -> Option<PathBuf> {
    cycle
        .map(|c| &c.scene)
        .or(template.map(|t| &t.scene))
        .zip(repo_root)
        .map(|(scene, root)| root.join(scene))
}

/// What "train again on what failed" adds to the next run's recipe (packet M13/Z4b,
/// [`crate::model::project::write_run_again`]); the seed is chosen when the recipe is written,
/// against every run on disk then.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Again {
    /// The previous run's `[eval] config`, verbatim (a repository-relative path, which `es`
    /// resolves from the repository root it runs in): collected under, and judging the new run.
    pub config: String,
    /// [`weakest`].
    pub suites: Vec<String>,
    /// Every dataset root the previous run trained on: its own `collect/ds` first, then its own
    /// `merge` list -- never its `collect/merged`, so no attempt is merged twice. Absolute.
    pub merge: Vec<String>,
    /// The checkpoint the previous run's evaluation judged: its `.esb` on the IR route, the
    /// `lerobot` `pretrained_model` directory it was imported from on the `LeRobot` route.
    pub init: String,
}

/// Whether ⑤'s "train again on what failed" is offered for `run`, and what it adds; `Err` is the
/// i18n key of why not, shown on the button. `refused` names the suites `es loop collect` would
/// not collect under ([`uncollectable`]); the suites practised are [`weakest`] of the others.
pub fn again(
    run: &RunFolder,
    cycle: Option<&Cycle>,
    situations: &[Situation],
    refused: &[String],
) -> Result<Again, &'static str> {
    let cycle = cycle.ok_or("results.again.no_recipe")?;
    let collect = cycle.collect.as_ref().ok_or("results.again.no_collect")?;
    let suites = weakest(situations, refused)?;
    let no_checkpoint = "results.again.no_checkpoint";
    let recipe = cycle.training(None, &run.path).map_err(|_| no_checkpoint)?;
    let mark = match evaluated_mark(&run.path) {
        Some(mark) => mark,
        None => cycle.mark(&recipe).map_err(|_| no_checkpoint)?,
    };
    let train = run.path.join("train");
    // What `Cycle::check_inputs` asks of `[train] init` before the next run collects anything.
    let (init, complete) = if matches!(recipe.route(), Ok(Route::External)) {
        let dir = PathBuf::from(lerobot_checkpoint(&train, mark));
        let ok = dir.join("model.safetensors").is_file() && dir.join("config.json").is_file();
        (dir, ok)
    } else {
        let file = train.join("checkpoints").join(format!("{mark}.esb"));
        let ok = file.is_file();
        (file, ok)
    };
    if !complete {
        return Err(no_checkpoint);
    }
    Ok(Again {
        config: cycle.eval.config.clone(),
        suites,
        merge: std::iter::once(collect_root(&run.path))
            .chain(collect.merge.iter().cloned())
            .collect(),
        init: init.display().to_string(),
    })
}

/// The mark the run's own ledger says its evaluation judged: the newest evaluate step's
/// `policy_hash`, among the train steps' `checkpoint.<mark>` outputs. The expert gate's step
/// names no checkpoint, so it matches none.
fn evaluated_mark(run: &Path) -> Option<u32> {
    let ledger = read_loop_steps(run).ok()?;
    let marks: Vec<(&String, &String)> = ledger
        .iter()
        .filter(|s| s.kind == LoopKind::Train)
        .flat_map(|s| &s.outputs)
        .collect();
    ledger
        .iter()
        .rev()
        .filter(|s| s.kind == LoopKind::Evaluate)
        .filter_map(|s| s.inputs.get("policy_hash"))
        .find_map(|hash| {
            let (key, _) = marks.iter().find(|(_, v)| *v == hash)?;
            key.strip_prefix(CHECKPOINT)?.parse().ok()
        })
}

/// The suites with the lowest success rate among those that ran an attempt and are not
/// `refused`, every one of them on a tie, in the evaluation's order. Refused when nothing was
/// measured, when every attempt was a success (plan Z review focus 4), and when every suite with
/// a failure is `refused` (review of plan Z, R2): a suite the collector cannot realise is left
/// out as if it had not run, so the next-lowest takes its place, and none that failed is
/// practised only when none can be.
pub fn weakest(situations: &[Situation], refused: &[String]) -> Result<Vec<String>, &'static str> {
    // `a` below `b`: compared as fractions, exactly.
    let rate = |a: &Situation, b: &Situation| {
        (u64::from(a.successes) * u64::from(b.episodes))
            .cmp(&(u64::from(b.successes) * u64::from(a.episodes)))
    };
    let ran: Vec<&Situation> = situations.iter().filter(|s| s.episodes > 0).collect();
    if ran.is_empty() {
        return Err("results.again.no_evaluation");
    }
    if ran.iter().all(|s| s.successes == s.episodes) {
        return Err("results.again.no_failures");
    }
    let open: Vec<&Situation> = ran
        .into_iter()
        .filter(|s| !refused.contains(&s.suite))
        .collect();
    let low = (open.iter().copied())
        .min_by(|a, b| rate(a, b))
        .filter(|low| low.successes < low.episodes)
        .ok_or("results.again.cannot_collect")?;
    Ok(open
        .iter()
        .filter(|s| rate(s, low).is_eq())
        .map(|s| s.suite.clone())
        .collect())
}

/// The suites of `ir` that `es loop collect --perturb` refuses under `deploy`, by name: the
/// same [`es_eval::perturb::unseen_age`] it asks, at the deployment's control period.
pub fn uncollectable(ir: &EvaluationIr, deploy: &DeploymentIr) -> Vec<String> {
    let all: Vec<usize> = (0..ir.suites.len()).collect();
    let control_us = deploy.rate.control_period().0;
    (unseen_age(ir, &all, deploy, control_us).into_iter())
        .map(|r| r.suite)
        .collect()
}

/// One run, read for ⑤: its `eval/` folder, `episodes.json` when it has one, and what its
/// `cycle.toml` names.
#[derive(Debug)]
pub struct RunResults {
    pub run: RunFolder,
    pub dir: RunDir,
    /// `None` is a run from before `episodes.json`: no tiles, bars from the report.
    pub rows: Option<Vec<EpisodeRow>>,
    /// Why an `episodes.json` that is there could not be read.
    pub rows_error: Option<String>,
    pub cycle: Option<Cycle>,
    /// The Evaluation IR `[eval] config` names; `None` shows the raw suite names.
    pub ir: Option<EvaluationIr>,
    /// What the outside camera re-poses the motion on: the recipe's scene, else the template's.
    pub scene: Option<PathBuf>,
    /// The project's previous run with a result, and its report.
    pub previous: Option<(u32, EvaluationReport)>,
    /// What Run again starts ③ with ([`start_settings`]).
    pub settings: Option<StartSettings>,
    /// The policy file Export copies ([`export_bundle`]).
    pub export: Option<PathBuf>,
    /// The template's `[outcome]`, and the class of each timed-out attempt whose trajectory
    /// says one (packet M13/Z4).
    pub outcome: Option<OutcomeSpec>,
    pub outcomes: Outcomes,
    /// A reorientation's timed-out attempts: how far each ended off the goal, degrees, by cell
    /// (packet M16/H7, [`outcome::final_angles`]).
    pub angles: BTreeMap<String, f64>,
    /// The project's chosen teacher's own score, the student's reference (packet M16/H7).
    pub teacher: Option<Score>,
    /// "Train again on what failed", or the i18n key of why not ([`again`]).
    pub again: Result<Again, &'static str>,
}

impl RunResults {
    /// Only `eval/report.json` is required. `repo_root` resolves the recipe's
    /// repository-relative paths and finds the project's template; without it the labels
    /// are raw and the outside camera has no scene.
    pub fn read(
        project: &Project,
        run: &RunFolder,
        repo_root: Option<&Path>,
    ) -> Result<Self, String> {
        let dir = RunDir::open(&run.eval_dir()).map_err(|e| e.to_string())?;
        let (rows, rows_error) = match read_episodes(&run.eval_dir()) {
            Ok(rows) => (rows, None),
            Err(e) => (None, Some(e.to_string())),
        };
        let cycle = std::fs::read_to_string(run.path.join(RUN_RECIPE))
            .ok()
            .and_then(|text| Cycle::parse(&text).ok());
        let template = repo_root.and_then(|root| {
            load(root)
                .0
                .into_iter()
                .find(|t| t.id == project.file.template)
        });
        let ir = cycle.as_ref().zip(repo_root).and_then(|(c, root)| {
            let text = std::fs::read_to_string(root.join(&c.eval.config)).ok()?;
            evaluation_from_toml(&text).ok()
        });
        let scene = scene(cycle.as_ref(), template.as_ref(), repo_root);
        let previous = finished_runs(project)
            .into_iter()
            .rfind(|r| r.number < run.number)
            .and_then(|r| {
                let text = std::fs::read_to_string(r.report_path()).ok()?;
                Some((r.number, serde_json::from_str(&text).ok()?))
            });
        let settings = cycle
            .as_ref()
            .zip(template.as_ref())
            .and_then(|(c, t)| start_settings(c, t));
        // The plane `es loop collect` would collect under: the template's Deployment IR, which
        // the project's collect bundle was built from. Without it or the Evaluation IR nothing
        // is known to be refused, and the collection itself still refuses by name.
        let deploy = template.as_ref().zip(repo_root).and_then(|(t, root)| {
            let text = std::fs::read_to_string(root.join(&t.bundle.deployment)).ok()?;
            deployment_from_toml(&text).ok()
        });
        let refused = (ir.as_ref().zip(deploy.as_ref()))
            .map_or_else(Vec::new, |(ir, deploy)| uncollectable(ir, deploy));
        let outcome = template.and_then(|t| t.outcome);
        let (mut outcomes, mut angles) = (Outcomes::new(), BTreeMap::new());
        if let (Some(rows), Some(spec), Some(scene)) = (&rows, &outcome, &scene) {
            match spec.kind {
                OutcomeKind::Place => outcomes = outcome::outcomes(scene, spec, rows, &dir),
                OutcomeKind::Reorient => angles = outcome::final_angles(scene, spec, rows, &dir),
            }
        }
        let bars = situations(&dir.report, rows.as_deref(), ir.as_ref());
        Ok(Self {
            again: again(run, cycle.as_ref(), &bars, &refused),
            run: run.clone(),
            export: export_bundle(run, cycle.as_ref()),
            dir,
            rows,
            rows_error,
            cycle,
            ir,
            scene,
            previous,
            settings,
            outcome,
            outcomes,
            angles,
            teacher: teacher::chosen_score(project),
        })
    }

    /// The template's `[outcome]` when it is a reorientation.
    fn reorient(&self) -> Option<&OutcomeSpec> {
        (self.outcome.as_ref()).filter(|o| o.kind == OutcomeKind::Reorient)
    }

    /// A tile's word: its cause, and for a reorientation's timeout how far it ended off the goal.
    pub fn tile_label(&self, lang: Lang, cell: &str, cause: Cause) -> String {
        match (cause, self.angles.get(cell)) {
            (Cause::Timeout, Some(deg)) => {
                fill(lang, "outcome.not_aligned_by", &[&format!("{deg:.0}")])
            }
            _ => self.cause_label(lang, cause),
        }
    }

    /// ⑤'s lines for a reorientation (packet M16/H7): the teacher's score beside the student's,
    /// the share that succeeds by chance, and how far the timed-out attempts ended off the goal.
    pub fn reorient_lines(&self, lang: Lang) -> Vec<String> {
        let Some(o) = self.reorient() else {
            return Vec::new();
        };
        let success = format!("{:.1}", o.angle_rad.to_degrees());
        let mut lines = Vec::new();
        let student = card(&self.dir.report, self.rows.as_deref());
        if let (Some(t), true) = (self.teacher, student.episodes > 0) {
            let s = f64::from(student.successes) / f64::from(student.episodes);
            let [t, s] = [t.rate(), s].map(|r| format!("{r:.2}"));
            lines.push(fill(lang, "outcome.teacher", &[&t, &s]));
        }
        if let Some(chance) = o.chance {
            let chance = format!("{:.0}", chance * 100.0);
            lines.push(fill(lang, "outcome.chance", &[&chance, &success]));
        }
        let mut a: Vec<f64> = self.angles.values().copied().collect();
        a.sort_by(f64::total_cmp);
        if let (Some(lo), Some(hi)) = (a.first(), a.last()) {
            let [median, lo, hi] = [a[a.len() / 2], *lo, *hi].map(|d| format!("{d:.0}"));
            lines.push(fill(lang, "outcome.angles", &[&median, &lo, &hi, &success]));
        }
        lines
    }

    pub fn comparison(&self) -> Comparison {
        compare(&self.dir.report, self.previous.as_ref().map(|(_, r)| r))
    }

    /// A cause's plain name; an outcome class's names the template's object and target.
    pub fn cause_label(&self, lang: Lang, cause: Cause) -> String {
        let table = Strings::get(lang);
        if let Some(o) = self.reorient() {
            match cause {
                Cause::FailureCondition => {
                    let cm = format!("{:.0}", o.drop_m * 100.0);
                    return fill(lang, "outcome.dropped", &[table.t(&o.object_name), &cm]);
                }
                Cause::Timeout => {
                    return fill(lang, "outcome.not_aligned", &[table.t(&o.target_name)]);
                }
                _ => {}
            }
        }
        let names: Vec<&str> = (self.outcome.iter())
            .flat_map(|o| [table.t(&o.object_name), table.t(&o.target_name)])
            .collect();
        fill(lang, cause_key(cause), &names)
    }
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

    fn none() -> Outcomes {
        Outcomes::new()
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
            causes(&rows, &none()),
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
            causes(&rows, &none()),
            [(Cause::FailureCondition, 1), (Cause::SafetyLimit, 1)]
        );
    }

    /// Review R3, in the shape of the hint card's first run: nominal-00's start-up buckets over
    /// 1,800 steps are no cause -- it keeps how it ended, or its outcome class -- while a
    /// torque-noise attempt clamped on 800 of its 1,800 steps hit a safety limit.
    #[test]
    fn a_step_counter_under_one_percent_of_the_steps_is_no_cause() {
        fn long(suite: &str, ep: u64, termination: &str, extra: &[(&str, u64)]) -> EpisodeRow {
            EpisodeRow {
                steps: 1800,
                ..row(suite, ep, termination, extra)
            }
        }
        let start = [
            ("fallback", 1),
            ("violation.acceleration", 2),
            ("violation.chunk_underrun", 1),
        ];
        let noisy = [
            ("fallback", 1),
            ("violation.chunk_underrun", 1),
            ("violation.velocity", 800),
        ];
        let rows = [
            long("nominal", 0, "timeout", &start),
            long("torque_noise", 1, "timeout", &noisy),
            long("nominal", 2, "success", &start),
        ];
        assert_eq!(
            causes(&rows, &none()),
            [(Cause::Timeout, 2), (Cause::SafetyLimit, 1)]
        );
        let left = Cause::Outcome(Outcome::LeftOutside);
        let classes = Outcomes::from([("nominal-00".to_owned(), Outcome::LeftOutside)]);
        assert_eq!(causes(&rows[..1], &classes), [(left, 1)]);
        let picked: Vec<_> = tiles(&rows, TileFilter::Failures, &classes)
            .iter()
            .map(|t| t.cause)
            .collect();
        assert_eq!(picked, [Some(left), Some(Cause::Timeout)]);
        // 18 of 1,800 steps is a cause and 17 is not; a failure the backend reported always is.
        let at = |n| {
            let extra = [("violation.position", n), ("chunk_underrun", 1)];
            causes(&[long("nominal", 0, "failure", &extra)], &none())
        };
        assert_eq!(
            at(18),
            [
                (Cause::FailureCondition, 1),
                (Cause::SafetyLimit, 1),
                (Cause::MotionGaps, 1)
            ]
        );
        assert_eq!(
            at(17),
            [(Cause::FailureCondition, 1), (Cause::MotionGaps, 1)]
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
        assert!(tiles(&[], TileFilter::All, &none()).is_empty());
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

    /// Two suites: `dim` (one kind, twice) and `nominal` (none).
    fn two_suite_ir() -> EvaluationIr {
        let light = Perturbation::new(
            PerturbationKind::LightIntensity {
                range: Range::new(0.5, 1.5),
                dist: Distribution::Uniform,
            },
            0,
        );
        EvaluationIr {
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
        }
    }

    #[test]
    fn situations_follow_the_evaluation_and_count_rows() {
        let ir = two_suite_ir();
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
        assert_eq!(tiles(&rows, TileFilter::Failures, &none()).len(), 1);
        assert_eq!(
            tiles(&rows, TileFilter::Failures, &none())[0].cause,
            Some(Cause::Timeout)
        );
        assert_eq!(
            tiles(&rows, TileFilter::Successes, &none())[0].cell,
            "nominal-00"
        );
        assert_eq!(tiles(&rows, TileFilter::Successes, &none())[0].cause, None);
        assert_eq!(tiles(&rows, TileFilter::All, &none()).len(), 2);
    }

    // --- the screen (packet M12/Y13) ----------------------------------------------------------

    use crate::model::project::tests::{cube, repo};
    use crate::model::project::{write_run, write_run_again, ProjectFile};
    use es_data::training::PerturbRef;
    use es_eval::episodes::write_episodes;
    use es_ir::evaluation::{AcceptanceCriterion, Aggregation};

    /// A project folder holding only its `project.toml`: nothing here builds a bundle.
    fn scratch(name: &str) -> Project {
        let root = std::env::temp_dir().join(format!("es-y13-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let file = ProjectFile {
            kind: "project".into(),
            name: name.into(),
            template: "cube-into-bin".into(),
        };
        std::fs::write(root.join("project.toml"), toml::to_string(&file).unwrap()).unwrap();
        Project::open(&root).unwrap()
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                copy_dir(&path, &to.join(entry.file_name()));
            } else {
                std::fs::copy(&path, to.join(entry.file_name())).unwrap();
            }
        }
    }

    /// The committed E5 fixture run, copied into `runs/<n>/eval/`. The fixture is only read.
    fn copy_fixture(project: &Project, n: u32) -> RunFolder {
        let run = RunFolder {
            number: n,
            path: project.root.join("runs").join(format!("{n:03}")),
        };
        copy_dir(
            &repo().join("tests/fixtures/visible-learning/run"),
            &run.eval_dir(),
        );
        run
    }

    /// Review focus 5, on disk: the fixture has no `episodes.json`, so it is the old-run path
    /// (no rows, bars and headline from the report); rows written beside it that agree with its
    /// report give tiles and causes, and a broken file is named rather than guessed around.
    #[test]
    fn a_copied_fixture_run_is_an_old_run_until_episodes_json_is_beside_it() {
        let p = scratch("old");
        let run = copy_fixture(&p, 1);
        let runs = finished_runs(&p);
        assert_eq!(runs, std::slice::from_ref(&run));
        assert_eq!(shown(&runs, None), Some(&run));

        let r = RunResults::read(&p, &run, Some(&repo())).unwrap();
        assert!(r.rows.is_none() && r.rows_error.is_none());
        assert!(r.cycle.is_none() && r.ir.is_none() && r.settings.is_none());
        assert_eq!(r.again, Err("results.again.no_recipe"));
        assert_eq!(
            r.scene,
            Some(repo().join("tests/fixtures/mjcf/so101_pick_place.xml")),
            "no recipe: the template's scene"
        );
        assert_eq!(r.comparison(), Comparison::NoPrevious);
        let old = card(&r.dir.report, None);
        assert_eq!((old.passed, old.successes, old.episodes), (false, 3, 4));
        assert_eq!(
            first_to_play(None, r.dir.cells()).as_deref(),
            Some("nominal-00"),
            "the one cell that kept its motion"
        );
        assert_eq!(suite_label(Lang::En, "light", r.ir.as_ref()), "light");

        let rows = [
            row("nominal", 0, "success", &[]),
            row("nominal", 1, "success", &[("violation.velocity", 2)]),
            row("light", 0, "success", &[]),
            row("light", 1, "timeout", &[("fallback", 1)]),
        ];
        write_episodes(&rows, &run.eval_dir()).unwrap();
        let r = RunResults::read(&p, &run, Some(&repo())).unwrap();
        let read = r.rows.as_deref().expect("rows");
        assert_eq!(
            card(&r.dir.report, Some(read)),
            old,
            "rows and report agree"
        );
        assert_eq!(
            causes(read, &none()),
            [(Cause::Timeout, 1), (Cause::SafetyFallback, 1)]
        );
        assert_eq!(
            first_to_play(Some(read), r.dir.cells()).as_deref(),
            Some("light-01"),
            "failures first"
        );

        std::fs::write(run.eval_dir().join("episodes.json"), "{").unwrap();
        let r = RunResults::read(&p, &run, None).unwrap();
        assert!(r.rows.is_none() && r.rows_error.is_some(), "named, not old");
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Packet M13/Z4, review focus 3: a timed-out attempt with a trajectory is named by where
    /// the cube went, in the template's words; one without keeps "not done in time", and a
    /// failure something else ended keeps its own cause.
    #[test]
    fn a_reorientation_names_drops_and_misses_and_shows_the_teacher() {
        use crate::model::teacher::tests::{fake_run, hand_project};
        let p = hand_project("results");
        let teacher = fake_run(&p, &[250]);
        crate::model::teacher::choose(&teacher::tests::hand(), &repo(), &p, &teacher, 250).unwrap();
        let run = copy_fixture(&p, 1);
        let rows = [
            row("nominal", 0, "timeout", &[]),
            row("nominal", 1, "failure", &[]),
            row("nominal", 2, "success", &[]),
            row("nominal", 3, "timeout", &[]),
        ];
        write_episodes(&rows, &run.eval_dir()).unwrap();
        let mut r = RunResults::read(&p, &run, Some(&repo())).unwrap();
        assert!(r.outcomes.is_empty(), "no place classes");
        assert_eq!(
            r.teacher,
            Some(Score {
                successes: 31,
                episodes: 64
            })
        );
        // The fixture's trajectory is the arm's: no cube body, so no angle from it.
        assert!(r.angles.is_empty());
        r.angles = BTreeMap::from([("nominal-00".into(), 37.4), ("nominal-03".into(), 120.0)]);
        for lang in Lang::ALL {
            let dropped = r.cause_label(lang, Cause::FailureCondition);
            assert!(dropped.contains("24") && dropped.contains(t(lang, "outcome.cube")));
            let late = r.cause_label(lang, Cause::Timeout);
            assert!(late.contains(t(lang, "outcome.goal")) && !late.contains("{}"));
            assert!(r
                .tile_label(lang, "nominal-00", Cause::Timeout)
                .contains("37"));
            assert_eq!(r.tile_label(lang, "nominal-01", Cause::Timeout), late);
            let lines = r.reorient_lines(lang);
            assert_eq!(lines.len(), 3, "{lines:?}");
            assert!(
                lines[0].contains("0.48") && lines[0].contains("0.25"),
                "{lines:?}"
            );
            assert!(
                lines[1].contains('6') && lines[1].contains("5.7"),
                "{lines:?}"
            );
            assert!(
                lines[2].contains("120") && lines[2].contains("37"),
                "{lines:?}"
            );
            assert!(lines.iter().all(|l| !l.contains("{}")));
        }
        std::fs::remove_dir_all(&p.root).ok();
    }

    #[test]
    fn a_timeout_with_a_trajectory_is_named_by_its_outcome() {
        let p = scratch("outcome");
        let run = copy_fixture(&p, 1);
        let rows = [
            // The fixture's one trajectory: the arm sweeps, the cube never moves.
            row("nominal", 0, "timeout", &[("violation.velocity", 2)]),
            row("nominal", 1, "failure", &[]),
            row("light", 0, "success", &[]),
            row("light", 1, "timeout", &[]),
        ];
        write_episodes(&rows, &run.eval_dir()).unwrap();
        let r = RunResults::read(&p, &run, Some(&repo())).unwrap();
        let never = Cause::Outcome(Outcome::NeverLifted);
        assert_eq!(
            r.outcomes,
            Outcomes::from([("nominal-00".to_owned(), Outcome::NeverLifted)]),
            "no trajectory for light-01, and nominal-01 did not time out"
        );
        let read = r.rows.as_deref().expect("rows");
        assert_eq!(
            causes(read, &r.outcomes),
            [
                (never, 1),
                (Cause::Timeout, 1),
                (Cause::FailureCondition, 1),
                (Cause::SafetyLimit, 1)
            ]
        );
        let failed = tiles(read, TileFilter::Failures, &r.outcomes);
        let picked: Vec<_> = failed.iter().map(|t| t.cause).collect();
        assert_eq!(
            picked,
            [
                Some(never),
                Some(Cause::FailureCondition),
                Some(Cause::Timeout)
            ]
        );
        for lang in Lang::ALL {
            let label = r.cause_label(lang, never);
            assert!(
                label.contains(t(lang, "outcome.cube")) && !label.contains("{}"),
                "{label}"
            );
            for o in Outcome::ALL {
                let label = r.cause_label(lang, Cause::Outcome(o));
                assert!(!label.contains("{}"), "{o:?} {lang:?}: {label}");
            }
            assert_eq!(
                r.cause_label(lang, Cause::Timeout),
                t(lang, "cause.timeout")
            );
        }
        // Without a checkout there is no template, so no class: the recorded causes stand.
        let bare = RunResults::read(&p, &run, None).unwrap();
        assert!(bare.outcomes.is_empty() && bare.outcome.is_none());
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Review focus 5, the other half: a previous run under the same evaluation gives a
    /// number, under another none at all; a run folder without a result is not listed.
    #[test]
    fn the_previous_run_is_compared_only_under_the_same_evaluation() {
        let p = scratch("compare");
        copy_fixture(&p, 1);
        let two = copy_fixture(&p, 2);
        let r = RunResults::read(&p, &two, None).unwrap();
        assert_eq!(r.previous.as_ref().map(|(n, _)| *n), Some(1));
        assert_eq!(
            r.comparison(),
            Comparison::Delta {
                success_points: 0.0
            }
        );
        assert_eq!(
            change_text(Lang::En, &r.comparison(), 1),
            Some(fill(Lang::En, "results.change", &["+0.0", "001"]))
        );

        let mut report = r.dir.report.clone();
        report.evaluation_hash = [7; 32];
        std::fs::write(two.report_path(), serde_json::to_string(&report).unwrap()).unwrap();
        let r = RunResults::read(&p, &two, None).unwrap();
        assert_eq!(r.comparison(), Comparison::NotComparable);
        assert_eq!(
            change_text(Lang::En, &r.comparison(), 1).as_deref(),
            Some(t(Lang::En, "results.not_comparable"))
        );
        assert_eq!(change_text(Lang::En, &Comparison::NoPrevious, 1), None);

        std::fs::create_dir_all(p.root.join("runs/003")).unwrap();
        let runs = finished_runs(&p);
        assert_eq!(runs.iter().map(|r| r.number).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(shown(&runs, Some(1)).map(|r| r.number), Some(1));
        assert_eq!(
            shown(&runs, Some(3)).map(|r| r.number),
            Some(2),
            "gone: newest"
        );
        assert_eq!(shown(&[], None), None);
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Export takes the checkpoint the evaluation judged, else the newest one on disk; Run
    /// again reads back the two settings the editor wrote.
    #[test]
    fn export_and_run_again_read_what_the_editor_wrote() {
        let p = scratch("export");
        let settings = StartSettings {
            demonstrations: 50,
            length: Length::Short,
        };
        write_run(
            &cube(),
            &repo(),
            &p,
            settings,
            &p.next_run_dir(),
            "127.0.0.1:7010",
        )
        .unwrap();
        let run = p.latest_run().unwrap();
        let cycle =
            Cycle::parse(&std::fs::read_to_string(run.path.join(RUN_RECIPE)).unwrap()).unwrap();
        assert_eq!(start_settings(&cycle, &cube()), Some(settings));
        let mut hand = cycle.clone();
        hand.train.run.as_mut().unwrap().checkpoint_at = vec![123];
        assert_eq!(start_settings(&hand, &cube()), None, "no preset");

        assert_eq!(export_bundle(&run, Some(&cycle)), None, "nothing trained");
        let dir = run.path.join("train").join("checkpoints");
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["2500.esb", "5000.esb", "notes.txt"] {
            std::fs::write(dir.join(name), "").unwrap();
        }
        let newest = Some(dir.join("5000.esb"));
        assert_eq!(export_bundle(&run, Some(&cycle)), newest, "last");
        let mut early = cycle.clone();
        early.eval.checkpoint = "2500".into();
        assert_eq!(
            export_bundle(&run, Some(&early)),
            Some(dir.join("2500.esb"))
        );
        early.eval.checkpoint = "1000".into();
        assert_eq!(export_bundle(&run, Some(&early)), newest, "not on disk");
        assert_eq!(export_bundle(&run, None), newest);
        assert_eq!(
            export_name(&p, &run, &dir.join("5000.esb")),
            format!("es-y13-{}-export-001-5000.esb", std::process::id())
        );
        std::fs::remove_dir_all(&p.root).ok();
    }

    #[test]
    fn the_player_offers_what_was_recorded() {
        assert_eq!(views(true, 3), [View::Outside, View::Eye, View::SideBySide]);
        assert_eq!(views(true, 0), [View::Outside]);
        assert_eq!(views(false, 1), [View::Eye]);
        assert!(views(false, 0).is_empty());
        assert_eq!(frame_at(0, 0), None);
        assert_eq!(frame_at(5, 10), Some(5));
        assert_eq!(frame_at(47, 1), Some(0), "past the last picture, the last");
        for lang in Lang::ALL {
            let words: BTreeSet<&str> = [View::Outside, View::Eye, View::SideBySide]
                .map(|v| t(lang, v.key()))
                .into_iter()
                .chain(TileFilter::ALL.map(|f| t(lang, f.key())))
                .collect();
            assert_eq!(words.len(), 6, "{lang:?}");
            assert!(words.iter().all(|w| !w.contains('.')), "{words:?}");
        }
    }

    #[test]
    fn acceptance_lines_and_suites_read_in_plain_words() {
        let ir = two_suite_ir();
        let en = Lang::En;
        assert_eq!(
            suite_label(en, "dim", Some(&ir)),
            t(en, "perturb.light_intensity"),
            "one kind, named once"
        );
        assert_eq!(
            suite_label(en, "nominal", Some(&ir)),
            t(en, "results.nominal")
        );
        assert_eq!(suite_label(en, "elsewhere", Some(&ir)), "elsewhere");
        assert_eq!(suite_label(en, "dim", None), "dim");

        let line = AcceptanceResult::Determined {
            criterion: AcceptanceCriterion {
                suite: Some("dim".into()),
                metric: MetricSpec::SuccessRate,
                comparator: Comparator::Ge,
                threshold: 0.5,
                aggregation: Aggregation::Mean,
            },
            observed: 0.6875,
            passed: true,
        };
        let l = acceptance_line(en, &line, Some(&ir));
        assert_eq!(l.passed, Some(true));
        let condition = fill(
            en,
            "results.ge",
            &[metric_label(en, MetricSpec::SuccessRate), "0.5"],
        );
        assert_eq!(
            l.text,
            fill(
                en,
                "results.criterion",
                &[t(en, "perturb.light_intensity"), &condition, "0.688"]
            )
        );
        assert_eq!(l.raw, "success_rate >= 0.5 (mean, dim)");
        let gone = AcceptanceResult::Unavailable {
            metric: MetricSpec::SuccessRate,
            reason: "no episode finished".into(),
        };
        let l = acceptance_line(en, &gone, None);
        assert!(l.passed.is_none() && l.text.contains("no episode finished"));
        for lang in Lang::ALL {
            let words: BTreeSet<&str> = [
                Comparator::Ge,
                Comparator::Gt,
                Comparator::Le,
                Comparator::Lt,
            ]
            .map(|c| t(lang, comparator_key(c)))
            .into_iter()
            .collect();
            assert_eq!(words.len(), 4, "{lang:?}");
            assert!(
                words.iter().all(|w| w.matches("{}").count() == 2),
                "{words:?}"
            );
        }
        assert_eq!([number(0.5), number(12.0), number(0.0)], ["0.5", "12", "0"]);
    }

    fn bar(suite: &str, successes: u32, episodes: u32) -> Situation {
        Situation {
            suite: suite.into(),
            kinds: Vec::new(),
            successes,
            episodes,
        }
    }

    /// Packet M13/Z4b: the lowest success rate, every suite tied at it, in the evaluation's
    /// order; a suite nobody ran is not the weakest; no failure anywhere refuses (review focus 4).
    #[test]
    fn the_weakest_suites_are_every_one_tied_at_the_lowest_rate() {
        let bars = [
            bar("nominal", 2, 2),
            bar("light_intensity", 1, 2),
            bar("backlash", 0, 0),
            bar("torque_noise", 2, 4),
        ];
        assert_eq!(
            weakest(&bars, &[]),
            Ok(vec!["light_intensity".into(), "torque_noise".into()])
        );
        assert_eq!(
            weakest(&[bar("nominal", 3, 4), bar("light", 0, 1)], &[]),
            Ok(vec!["light".into()])
        );
        assert_eq!(
            weakest(&[bar("nominal", 2, 2), bar("light", 4, 4)], &[]),
            Err("results.again.no_failures")
        );
        assert_eq!(
            weakest(&[bar("nominal", 0, 0)], &[]),
            Err("results.again.no_evaluation")
        );
        assert_eq!(weakest(&[], &[]), Err("results.again.no_evaluation"));
    }

    /// Review R2: a suite the collector refuses is left out as if it had not run -- the lowest
    /// of the others is practised, a tie keeps only its realisable half, and when only refused
    /// suites failed nothing is.
    #[test]
    fn the_weakest_suites_are_among_those_the_collector_can_realise() {
        let drop = vec!["frame_drop".to_owned()];
        let bars = [
            bar("nominal", 2, 2),
            bar("frame_drop", 0, 2),
            bar("torque_noise", 1, 2),
            bar("backlash", 1, 2),
        ];
        assert_eq!(weakest(&bars, &[]), Ok(drop.clone()));
        assert_eq!(
            weakest(&bars, &drop),
            Ok(vec!["torque_noise".into(), "backlash".into()])
        );
        let tied = [bar("frame_drop", 0, 2), bar("torque_noise", 0, 2)];
        assert_eq!(weakest(&tied, &drop), Ok(vec!["torque_noise".into()]));
        let only = [bar("nominal", 2, 2), bar("frame_drop", 1, 2)];
        assert_eq!(weakest(&only, &drop), Err("results.again.cannot_collect"));
        assert_eq!(
            weakest(&[bar("frame_drop", 0, 2)], &drop),
            Err("results.again.cannot_collect")
        );
    }

    /// A run folder as `es loop cycle` leaves it for ⑤: the fixture's `eval/` with `rows` beside
    /// it. `failing` suites lose their second attempt.
    fn evaluated(project: &Project, n: u32, failing: &[&str]) -> RunFolder {
        let run = copy_fixture(project, n);
        let rows: Vec<EpisodeRow> = ["nominal", "light_intensity", "torque_noise"]
            .into_iter()
            .flat_map(|suite| {
                let second = if failing.contains(&suite) {
                    "timeout"
                } else {
                    "success"
                };
                [row(suite, 0, "success", &[]), row(suite, 1, second, &[])]
            })
            .collect();
        write_episodes(&rows, &run.eval_dir()).unwrap();
        run
    }

    fn lerobot_checkpoint_on_disk(run: &RunFolder, mark: u32) -> PathBuf {
        let dir = PathBuf::from(lerobot_checkpoint(&run.path.join("train"), mark));
        std::fs::create_dir_all(&dir).unwrap();
        for file in ["model.safetensors", "config.json"] {
            std::fs::write(dir.join(file), "").unwrap();
        }
        dir
    }

    fn written(run: &Path) -> Cycle {
        Cycle::parse(&std::fs::read_to_string(run.join(RUN_RECIPE)).unwrap()).unwrap()
    }

    /// Packet M13/Z4b on the camera-only card (the `LeRobot` route): a run whose evaluation had
    /// no failure, or whose checkpoint is gone, is refused by name; then two "train again" runs
    /// in a row. Each writes a Z3 cycle `es loop cycle` parses and resolves -- perturbed under
    /// the weakest suites of the same Evaluation IR, seeds after every earlier collection and
    /// clear of the evaluation's 101-116, every earlier `collect/ds` merged and none of the
    /// `collect/merged` roots, started from the checkpoint the last one judged.
    #[test]
    fn train_again_writes_a_cycle_that_collects_fresh_seeds_merges_and_continues() {
        let p = scratch("again");
        let settings = StartSettings {
            demonstrations: 100,
            length: Length::Short,
        };
        write_run(
            &cube(),
            &repo(),
            &p,
            settings,
            &p.next_run_dir(),
            "127.0.0.1:7020",
        )
        .unwrap();
        let first = evaluated(&p, 1, &[]);
        let read = |run: &RunFolder| RunResults::read(&p, run, Some(&repo())).unwrap();
        assert_eq!(read(&first).again, Err("results.again.no_failures"));
        let first = evaluated(&p, 1, &["light_intensity", "torque_noise"]);
        assert_eq!(read(&first).again, Err("results.again.no_checkpoint"));
        // `[eval] checkpoint = "last"` is the Short preset's 5000.
        let init = lerobot_checkpoint_on_disk(&first, 5000);
        let again = read(&first).again.unwrap();
        let config = written(&first.path).eval.config;
        assert_eq!(
            again,
            Again {
                config: config.clone(),
                suites: vec!["light_intensity".into(), "torque_noise".into()],
                merge: vec![collect_root(&first.path)],
                init: init.display().to_string(),
            }
        );

        let judged: Vec<u64> = (101..=116).collect();
        let ir =
            evaluation_from_toml(&std::fs::read_to_string(repo().join(&config)).unwrap()).unwrap();
        assert_eq!(crate::model::project::evaluation_seeds(&ir), judged);
        let second = p.next_run_dir();
        write_run_again(
            &cube(),
            &repo(),
            &p,
            &again,
            settings,
            &second,
            "127.0.0.1:7021",
        )
        .unwrap();
        let cycle = written(&second);
        let c = cycle.collect.as_ref().unwrap();
        assert_eq!(
            c.perturb,
            Some(PerturbRef {
                config: config.clone(),
                suites: again.suites.clone(),
            })
        );
        assert_eq!(cycle.eval, written(&first.path).eval, "the same judge");
        assert!(cycle.eval.preview.is_some());
        assert_eq!(c.merge, again.merge);
        assert_eq!(cycle.train.init, Some(again.init.clone()));
        // 1-100 was the first run's, and 101-200 would meet the evaluation's 101-116.
        assert_eq!((c.seed, c.episodes), (117, 100));
        let range = c.seed..c.seed + u64::from(c.episodes);
        assert!(judged.iter().all(|s| !range.contains(s)) && range.start >= 101);
        let recipe = cycle.training(None, &second).unwrap();
        let lerobot = recipe.policy.lerobot.as_ref().unwrap();
        assert_eq!(lerobot.path, Some(again.init.clone()), "--policy.path");
        assert_eq!(
            recipe.dataset.unwrap().root,
            cycle.dataset_root(&second),
            "the merged root trains"
        );

        // Again after the again-run: both earlier roots, never a `collect/merged`.
        let second = evaluated(&p, 2, &["nominal"]);
        let init = lerobot_checkpoint_on_disk(&second, 5000);
        let again = read(&second).again.unwrap();
        assert_eq!(again.suites, ["nominal"]);
        assert_eq!(
            again.merge,
            [collect_root(&second.path), collect_root(&first.path)]
        );
        assert!(again.merge.iter().all(|m| !m.contains("merged")));
        assert_eq!(again.init, init.display().to_string());
        let third = p.next_run_dir();
        write_run_again(
            &cube(),
            &repo(),
            &p,
            &again,
            settings,
            &third,
            "127.0.0.1:7022",
        )
        .unwrap();
        let c = written(&third).collect.unwrap();
        assert_eq!(
            (c.seed, c.merge),
            (217, again.merge.clone()),
            "after 117-216"
        );
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// The IR route (the cube-pose card) starts from the evaluated `.esb`, and the mark is the
    /// one the run's own ledger says its evaluation judged, whatever `[eval] checkpoint` says.
    #[test]
    fn train_again_on_the_ir_route_starts_from_the_bundle_the_ledger_judged() {
        use es_data::collect::{append_loop_step, LoopStep};

        let p = scratch("again-ir");
        let settings = StartSettings {
            demonstrations: 10,
            length: Length::Short,
        };
        let hint = crate::model::project::tests::hint();
        write_run(
            &hint,
            &repo(),
            &p,
            settings,
            &p.next_run_dir(),
            "127.0.0.1:7023",
        )
        .unwrap();
        let run = evaluated(&p, 1, &["torque_noise"]);
        let dir = run.path.join("train").join("checkpoints");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("5000.esb"), "").unwrap();
        let again = RunResults::read(&p, &run, Some(&repo())).unwrap().again;
        assert_eq!(
            again.map(|a| a.init),
            Ok(dir.join("5000.esb").display().to_string()),
            "no ledger: `last`, the preset's 5000"
        );

        let train = LoopStep::new(LoopKind::Train)
            .output("checkpoint.1000", &"aa")
            .output("checkpoint.5000", &"bb");
        let judged = LoopStep::new(LoopKind::Evaluate).input("policy_hash", &"aa");
        let expert = LoopStep::new(LoopKind::Evaluate).input("policy_hash", &"cc");
        for step in [train, judged, expert] {
            append_loop_step(&run.path, &step).unwrap();
        }
        assert_eq!(
            evaluated_mark(&run.path),
            Some(1000),
            "the expert gate matches none"
        );
        assert_eq!(
            RunResults::read(&p, &run, Some(&repo())).unwrap().again,
            Err("results.again.no_checkpoint"),
            "1000.esb is not on disk"
        );
        std::fs::write(dir.join("1000.esb"), "").unwrap();
        let again = RunResults::read(&p, &run, Some(&repo()))
            .unwrap()
            .again
            .unwrap();
        assert_eq!(again.init, dir.join("1000.esb").display().to_string());
        let next = p.next_run_dir();
        write_run_again(
            &hint,
            &repo(),
            &p,
            &again,
            settings,
            &next,
            "127.0.0.1:7024",
        )
        .unwrap();
        let cycle = written(&next);
        let recipe = cycle.training(None, &next).unwrap();
        assert_eq!(
            recipe.init.map(|i| i.policy),
            Some(again.init),
            "[init] policy"
        );
        assert_eq!(cycle.collect.unwrap().seed, 11, "after 1-10, below 101");
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Review R2 on disk: the run was judged by an Evaluation IR whose `frame_drop` suite failed
    /// most, which `es loop collect` refuses under the template's deployment (its
    /// `stale_observation` watchdog), so "train again" practises the next-lowest suite -- and
    /// is refused by name once that suite is the only one that failed.
    #[test]
    fn train_again_leaves_out_a_suite_the_collector_cannot_realise() {
        use es_ir::evaluation::{CountRange, Perturbation, PerturbationSuite};

        let p = scratch("again-drop");
        let settings = StartSettings {
            demonstrations: 10,
            length: Length::Short,
        };
        write_run(
            &cube(),
            &repo(),
            &p,
            settings,
            &p.next_run_dir(),
            "127.0.0.1:7025",
        )
        .unwrap();
        let run = copy_fixture(&p, 1);
        lerobot_checkpoint_on_disk(&run, 5000);
        let mut cycle = written(&run.path);
        let judge = std::fs::read_to_string(repo().join(&cycle.eval.config)).unwrap();
        let mut ir = evaluation_from_toml(&judge).unwrap();
        ir.suites.push(PerturbationSuite {
            name: "dropped_frames".into(),
            perturbations: vec![Perturbation::new(
                PerturbationKind::FrameDrop {
                    prob: 0.05,
                    burst: CountRange::new(1, 3),
                },
                9,
            )],
        });
        let config = run.path.join("evaluation.toml");
        let text = es_ir::serial::evaluation_to_toml(&ir).unwrap();
        std::fs::write(&config, text).unwrap();
        cycle.eval.config = config.display().to_string();
        std::fs::write(run.path.join(RUN_RECIPE), toml::to_string(&cycle).unwrap()).unwrap();

        let judged = |torque: &str| {
            let rows = [
                row("nominal", 0, "success", &[]),
                row("torque_noise", 0, "success", &[]),
                row("torque_noise", 1, torque, &[]),
                row("dropped_frames", 0, "timeout", &[]),
                row("dropped_frames", 1, "timeout", &[]),
            ];
            write_episodes(&rows, &run.eval_dir()).unwrap();
            RunResults::read(&p, &run, Some(&repo())).unwrap().again
        };
        assert_eq!(
            judged("timeout").map(|a| a.suites),
            Ok(vec!["torque_noise".into()])
        );
        assert_eq!(judged("success"), Err("results.again.cannot_collect"));
        std::fs::remove_dir_all(&p.root).ok();
    }
}
