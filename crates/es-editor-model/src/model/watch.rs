//! ③ Train and ④ Evaluate while a run goes (`docs/design/editor-redesign.md` section 6.4,
//! packet M12/Y12): starting a project's run, watching it, stopping it and picking it up again.
//!
//! The editor still hosts nothing (spec 23.1). Start writes the run's recipe
//! ([`project::write_run`]) and queues `es loop cycle` for [`LaunchModel::start_in`], with the
//! repository root as its working directory; what the run does arrives as telemetry and what it
//! left arrives on disk ([`RunFacts`]). [`Watch`] is what sits between the two - which run,
//! whether the child is this editor's, when watching began, when something was last heard, the
//! re-open dial, the done latch - and [`Watch::view`] is every choice `ui/train.rs` draws.

use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use es_data::collect::{read_loop_steps, LoopKind, LoopStep, CHECKPOINT};
use es_data::training::Cycle;
use es_ir::evaluation::EvaluationIr;
use es_ir::serial::evaluation_from_toml;
use es_telemetry::transport::Client;

use crate::model::health::{self, Input, Light, Point, Verdict, THRESHOLDS};
use crate::model::i18n::Lang;
use crate::model::launch::{self, LaunchModel, State};
use crate::model::layout;
use crate::model::live_run::StageRow;
use crate::model::preview::{self, Preview};
use crate::model::project::{
    self, Project, ProjectError, RunFolder, StartSettings, RUN_RECIPE, TELEMETRY_FILE,
};
use crate::model::results::{self, Again};
use crate::model::telemetry_view::{self, Closed, Event, SeriesKey, Source, TelemetryModel};
use crate::model::template::{self, Length, Template};
use crate::model::train_view::STREAM_TRAIN;
use crate::model::workflow::{
    self, Child, LiveFacts, Phase, PhaseState, RunFacts, EVALUATE_STAGES, TRAIN_STAGES,
};

/// How often a run started here sends a picture (`--telemetry-image-every`): the observation
/// every N control ticks while collecting and evaluating, the training sample every N optimizer
/// steps. Without it the centre pane has nothing to show.
///
/// ponytail: one number for every stage, a first guess (a picture a second at the demo's 50 Hz
/// control rate); what publishing costs a collection or a training run is unmeasured
/// (`docs/design/editor-shell.md` section 16, gate 9), so Y-V times it.
pub const IMAGE_EVERY: &str = "50";

/// What a fresh connection hears before anything the run published: the handshake's
/// `HelloAck`, which [`telemetry_view::source_of`] hands over first. It is not news of the run,
/// and counting it would turn the light green before any data.
const HANDSHAKE: u64 = 1;

/// How long nothing attached waits before its unfinished run's `telemetry.txt` is dialled again
/// (packet M12/R3): counted from the last dial's refusal, or from the attachment closing.
pub const REDIAL_S: u64 = 10;

/// `Termination::Success` as `episode.end` and `cell.end` both spell it (`{:?}`).
const SUCCESS: &str = "Success";

/// The three lengths, in the order the panel offers them.
pub const LENGTHS: [Length; 3] = [Length::Short, Length::Medium, Length::Long];

/// What the demonstrations field accepts: at least one, and at most fifty times the template's
/// 200 - a bound on a typo, not a recommendation.
pub const DEMONSTRATIONS: std::ops::RangeInclusive<u32> = 1..=10_000;

pub fn length_key(length: Length) -> &'static str {
    match length {
        Length::Short => "watch.length.short",
        Length::Medium => "watch.length.medium",
        Length::Long => "watch.length.long",
    }
}

// --- what the wire says ------------------------------------------------------------------------

/// Collect's `episode.end` events: demonstrations made, and how many of them succeeded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Episodes {
    pub made: u32,
    pub succeeded: u32,
}

fn outcome(e: &Event) -> Option<&str> {
    e.fields.get("outcome").map(String::as_str)
}

/// Counted from the event log, which keeps the last `telemetry_view::DEFAULT_CAP` events - some
/// ten times what a 200-demonstration cycle sends.
pub fn episodes(events: &[Event]) -> Episodes {
    let mut out = Episodes::default();
    for e in events.iter().filter(|e| e.kind == "episode.end") {
        out.made += 1;
        out.succeeded += u32::from(outcome(e) == Some(SUCCESS));
    }
    out
}

/// One finished test attempt of ④'s strip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tile {
    pub cell: String,
    pub success: bool,
}

/// The evaluation's `cell.end` events, in the order they finished. The expert gate's cells test
/// the demonstrator, not the policy, and stay out.
pub fn tiles(events: &[Event]) -> Vec<Tile> {
    events
        .iter()
        .filter(|e| e.kind == "cell.end" && e.fields.get("stage").is_some_and(|s| s == "eval"))
        .map(|e| Tile {
            cell: e.fields.get("cell").cloned().unwrap_or_default(),
            success: outcome(e) == Some(SUCCESS),
        })
        .collect()
}

/// How far the stage in progress is: demonstrations made of those asked for while collecting,
/// the optimizer step of the run's total while training. Nothing is invented for the others.
pub fn progress(
    stage: &str,
    made: u32,
    demonstrations: u32,
    step: Option<u32>,
    total: Option<u32>,
) -> Option<f32> {
    let part = |done: u32, of: u32| (of > 0).then(|| (done as f32 / of as f32).clamp(0.0, 1.0));
    match stage {
        "collect" => part(made, demonstrations),
        "train" => part(step?, total?),
        _ => None,
    }
}

fn series(telemetry: &TelemetryModel, index: usize) -> &[(u64, f64)] {
    let key = SeriesKey {
        stream: STREAM_TRAIN,
        index,
    };
    telemetry.series.get(&key).map_or(&[], Vec::as_slice)
}

/// The training curve as the light reads it: stream 5's `[step, loss, lr, samples_per_s]`, one
/// point per progress line, the loss in the `f64` it was sent as (a NaN stays a NaN).
pub fn points(telemetry: &TelemetryModel) -> Vec<Point> {
    series(telemetry, 0)
        .iter()
        .zip(series(telemetry, 1))
        .zip(series(telemetry, 3))
        .map(|(((_, step), (_, loss)), (_, rate))| Point {
            step: *step as u64,
            loss: *loss,
            samples_per_s: *rate,
        })
        .collect()
}

// --- the stage cards -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CardState {
    Waiting,
    Running,
    /// With the seconds `stage.end` reported, when it did.
    Done(Option<f64>),
    Failed,
    /// Ended part-way without failing: stopped by the person, or cut off.
    Stopped,
}

impl CardState {
    pub fn key(self) -> &'static str {
        match self {
            CardState::Waiting => "watch.waiting",
            CardState::Running => "run.running",
            CardState::Done(_) => "run.done",
            CardState::Failed => "run.failed",
            CardState::Stopped => "watch.stopped",
        }
    }
}

fn stages_of(phase: Phase) -> &'static [&'static str] {
    match phase {
        Phase::Evaluate => EVALUATE_STAGES,
        _ => TRAIN_STAGES,
    }
}

/// The stage `state` says runs.
fn running(state: &PhaseState) -> Option<&str> {
    match state {
        PhaseState::Running { stage, .. } => Some(stage.as_str()),
        _ => None,
    }
}

/// One card per stage of `phase`, from the live `stage.begin` / `stage.end` rows and the phase's
/// state. A stage with no row is done when a later stage has one (a `--from` run skipped it) or
/// the phase says a later one runs (attached mid-run: what began before was never heard), or
/// when the phase is done; running when the phase says the child is about to begin it (nothing
/// heard yet), and otherwise still to come. A phase that is done - a failed acceptance included
/// ([`settle`]) - shows every stage that ended as done.
pub fn cards(
    phase: Phase,
    rows: &[StageRow],
    state: &PhaseState,
) -> Vec<(&'static str, CardState)> {
    let order = |name: &str| {
        TRAIN_STAGES
            .iter()
            .chain(EVALUATE_STAGES)
            .position(|s| *s == name)
    };
    let later = |stage: &str| {
        (rows.iter().map(|r| r.name.as_str()))
            .chain(running(state))
            .any(|s| order(s) > order(stage))
    };
    let done = *state == PhaseState::Done;
    stages_of(phase)
        .iter()
        .map(|&stage| {
            let row = rows.iter().rev().find(|r| r.name == stage);
            let card = match (row, state) {
                (_, PhaseState::Failed { stage: s, .. }) if s == stage => CardState::Failed,
                (Some(r), _) if done || r.code == Some(0) => CardState::Done(r.seconds),
                (Some(r), _) if r.code.is_some() => CardState::Failed,
                (Some(_), PhaseState::Running { .. }) => CardState::Running,
                (Some(_), _) => CardState::Stopped,
                (None, _) if done || later(stage) => CardState::Done(None),
                (None, PhaseState::Running { stage: s, .. }) if s == stage => CardState::Running,
                (None, _) => CardState::Waiting,
            };
            (stage, card)
        })
        .collect()
}

// --- how a run ended, and the light --------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    Passed,
    DidNotPass,
    Stopped,
}

/// How a cycle that exited on its own reads (`crates/es/src/cmd/cycle.rs`'s exit codes).
///
/// 0 is a cycle whose evaluation passed. 1 is both a crash and an acceptance that failed; the
/// second is the eval stage ending 1 with its report written - a crashed eval leaves none, since
/// the stage moves the previous report aside before it starts - and that is a result, not a
/// stop. Showcase repeats eval's code on its own `stage.end`, but eval's row comes first, so a
/// failed showcase is a stop. A 0 without a report is a stop too: a result that is not on disk
/// is never a pass. Everything else (2 usage, 3 skipped, a crash) is a stop.
pub fn ended(code: i32, stage: Option<&str>, report: bool) -> Ended {
    match (code, stage, report) {
        (0, _, true) => Ended::Passed,
        (1, Some("eval"), true) => Ended::DidNotPass,
        _ => Ended::Stopped,
    }
}

/// How the run has ended so far: by its exit, once a child this editor started has exited; and,
/// a failed acceptance being final the moment eval says so, by eval's own code while the run
/// still goes on to showcase, or when it is a run this editor only attached to and whose exit it
/// cannot see. `failed` is the stage ③ or ④ failed in and its code.
pub fn ending(
    child: Option<Child>,
    failed: Option<&(String, Option<i32>)>,
    report: bool,
) -> Option<Ended> {
    let stage = failed.map(|(stage, _)| stage.as_str());
    if let Child::Exited(code) = child? {
        return Some(ended(code, stage, report));
    }
    let code = failed?.1?;
    (ended(code, stage, report) == Ended::DidNotPass).then_some(Ended::DidNotPass)
}

/// A run that did not pass is still one whose evaluation ran to the end: ④ is done, and ⑤ holds
/// the verdict.
pub fn settle(phases: &mut [PhaseState; 5], ended: Option<Ended>) {
    if ended == Some(Ended::DidNotPass) {
        phases[3] = PhaseState::Done;
    }
}

/// The stage ③ or ④ failed in, and its code.
fn failed(phases: &[PhaseState; 5]) -> Option<(String, Option<i32>)> {
    phases[2..4].iter().find_map(|p| match p {
        PhaseState::Failed { stage, code } => Some((stage.clone(), *code)),
        _ => None,
    })
}

/// What the right-hand pane says about the run.
#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    pub light: Light,
    /// i18n keys: the light's name, and what the person can do about it.
    pub name: &'static str,
    pub advice: &'static str,
    /// The stage a stopped run stopped in, as the wire names it.
    pub stage: Option<String>,
}

/// The light (`health::judge`) with what only the run's ending knows: a finished run says
/// whether it passed, and a stop says in which stage, with that stage's own advice where there
/// is one - a skipped stage (exit 3: this machine lacks what it needs) and the expert gate.
pub fn signal(
    input: &Input<'_>,
    ended: Option<Ended>,
    failed: Option<&(String, Option<i32>)>,
) -> Signal {
    let finished = |light, name, advice| Signal {
        light,
        name,
        advice,
        stage: None,
    };
    let verdict = match ended {
        Some(Ended::Passed) => {
            return finished(Light::Green, "watch.passed", "watch.passed.advice");
        }
        // Grey like a stop the person asked for: not an alarm, and ⑤ has the causes.
        Some(Ended::DidNotPass) => {
            return finished(
                Light::Grey,
                "watch.did_not_pass",
                "watch.did_not_pass.advice",
            );
        }
        Some(Ended::Stopped) => Verdict::Stopped,
        None => health::judge(input, &THRESHOLDS),
    };
    let stopped = verdict == Verdict::Stopped;
    let advice = match failed.filter(|_| stopped) {
        Some((_, Some(3))) => "watch.advice.skipped",
        Some((stage, _)) if stage == "expert-gate" => "watch.advice.expert_gate",
        _ => verdict.advice_key(),
    };
    Signal {
        light: verdict.light(),
        name: verdict.key(),
        advice,
        stage: failed.filter(|_| stopped).map(|(stage, _)| stage.clone()),
    }
}

/// The light's colour; the step bar's own palette (`layout::colour`).
pub fn light_colour(light: Light) -> [u8; 3] {
    match light {
        Light::Grey => [160, 160, 160],
        Light::Green => [70, 170, 90],
        Light::Amber => [230, 165, 40],
        Light::Red => [220, 80, 70],
    }
}

// --- when a run may start, and when it is done ---------------------------------------------------

/// Whether Start may make a new run now: never while a child runs, while a start is already
/// queued, or while an attached run is still going - each would leave two runs, or a numbered
/// folder nothing ever ran in (`write_run` creates the folder).
pub fn may_start(pid: Option<u32>, queued: bool, phases: &[PhaseState; 5]) -> bool {
    pid.is_none()
        && !queued
        && !phases
            .iter()
            .any(|p| matches!(p, PhaseState::Running { .. }))
}

/// The address a re-opened project dials, once: its latest run's `telemetry.txt`, when disk says
/// that run did not finish. A finished run has nobody left to answer.
pub fn should_dial(run: Option<&RunFolder>, disk: &[PhaseState; 5]) -> Option<String> {
    run.filter(|_| unfinished(disk))?.telemetry_addr()
}

fn unfinished(disk: &[PhaseState; 5]) -> bool {
    disk[2..4]
        .iter()
        .any(|p| matches!(p, PhaseState::Interrupted { .. }))
}

/// The run's `telemetry.txt`, read and dialled on a thread of its own - never the UI's.
fn redial(run: RunFolder) -> Receiver<Result<Client, String>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let answer = run
            .telemetry_addr()
            .ok_or_else(String::new)
            .and_then(|addr| launch::dial(&addr, "", 1, launch::ATTACH_DELAY));
        let _ = tx.send(answer);
    });
    rx
}

/// The frame ④ becomes done: `Some(the step to show)` once - ⑤ when the person is on ③ or ④ and
/// ⑤ opens, else where they are - and `None` on every other frame.
pub fn just_done(before: &PhaseState, now: &[PhaseState; 5], at: Phase) -> Option<Phase> {
    if *before == PhaseState::Done || now[3] != PhaseState::Done {
        return None;
    }
    let moving = matches!(at, Phase::Train | Phase::Evaluate) && layout::can_open(&now[4]);
    Some(if moving { Phase::Results } else { at })
}

// --- files ---------------------------------------------------------------------------------------

fn fail(path: &Path, e: impl Display) -> ProjectError {
    ProjectError(format!("{}: {e}", path.display()))
}

fn address(port: u16) -> String {
    format!("127.0.0.1:{port}")
}

fn with_pictures(mut argv: Vec<String>) -> Vec<String> {
    argv.extend(["--telemetry-image-every".to_owned(), IMAGE_EVERY.to_owned()]);
    argv
}

/// `--from <from>` on `run`, at a new address - which is written into the run's `telemetry.txt`
/// first, so a later re-open dials the process that is actually there.
pub fn resume_command(run: &RunFolder, from: &str, port: u16) -> Result<Vec<String>, ProjectError> {
    let addr = address(port);
    let path = run.path.join(TELEMETRY_FILE);
    std::fs::write(&path, &addr).map_err(|e| fail(&path, e))?;
    Ok(with_pictures(project::resume_argv(run, from, &addr)))
}

/// The newest checkpoint mark the ledger's train rows (`checkpoint.<mark>`) or the live
/// `checkpoint` events report; `None` before anything was saved.
pub fn newest_mark(ledger: &[LoopStep], events: &[(u32, String)]) -> Option<u32> {
    ledger
        .iter()
        .filter(|s| s.kind == LoopKind::Train)
        .flat_map(|s| s.outputs.keys())
        .filter_map(|k| k.strip_prefix(CHECKPOINT)?.parse().ok())
        .chain(events.iter().map(|(step, _)| *step))
        .max()
}

/// `[eval] checkpoint = "<mark>"` in the run's own recipe, everything else as it was.
pub fn set_eval_checkpoint(run: &RunFolder, mark: u32) -> Result<(), ProjectError> {
    let path = run.path.join(RUN_RECIPE);
    let text = std::fs::read_to_string(&path).map_err(|e| fail(&path, e))?;
    let mut cycle = Cycle::parse(&text).map_err(|e| fail(&path, e))?;
    cycle.eval.checkpoint = mark.to_string();
    let text = toml::to_string(&cycle).map_err(|e| fail(&path, e))?;
    std::fs::write(&path, text).map_err(|e| fail(&path, e))
}

/// The run's own `[collect] episodes`: what its collect progress is a fraction of.
fn demonstrations_of(run: &RunFolder) -> Option<u32> {
    let text = std::fs::read_to_string(run.path.join(RUN_RECIPE)).ok()?;
    Some(Cycle::parse(&text).ok()?.collect?.episodes)
}

// --- the watch -----------------------------------------------------------------------------------

enum Dial {
    Idle,
    /// The re-open dial: ③ and ④ say *checking* until it answers.
    Dialling(Receiver<Result<Client, String>>),
    /// A later dial of `telemetry.txt` (packet M12/R3), which says nothing until it attaches.
    Redialling(Receiver<Result<Client, String>>),
    /// Until the connection says it closed.
    Attached(Closed),
}

/// What one frame asks of the window.
#[derive(Default)]
pub struct Tick {
    /// A run still going that a dial reached: the telemetry pane's new source.
    pub attached: Option<Source>,
    /// A child started this frame: what was heard before belongs to another run.
    pub started: bool,
    /// ④ has just become done: notify once, and show this step.
    pub done: Option<Phase>,
}

impl std::fmt::Debug for Tick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tick")
            .field("attached", &self.attached.is_some())
            .field("started", &self.started)
            .field("done", &self.done)
            .finish()
    }
}

/// Everything the three panes of ③ or ④ draw.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub state: PhaseState,
    pub cards: Vec<(&'static str, CardState)>,
    /// Of the stage in progress; and the time left, which only training has a rate for.
    pub fraction: Option<f32>,
    pub eta: Option<Duration>,
    /// `None` while no child runs and nothing is attached.
    pub signal: Option<Signal>,
    pub centre: Centre,
    /// ④'s strip; `None` on ③.
    pub tiles: Option<Vec<Tile>>,
    /// The Start button's key (`watch.start`, `watch.start_over`, or `watch.again.start` while
    /// a plan from ⑤ is pending), when it is offered.
    pub start: Option<&'static str>,
    /// ⑤'s pending "train again on what failed", shown above Start while Start is offered.
    pub again: Option<AgainPlan>,
    /// The stage `--from` resumes at, when Resume is offered.
    pub resume: Option<String>,
    pub stop: bool,
    pub evaluate_now: bool,
    /// The re-open dial has not answered yet.
    pub checking: bool,
    pub interrupted: bool,
    /// Why no run can start here (an i18n key), when none can.
    pub cannot_start: Option<&'static str>,
    /// The checkpoints' short tests, newest first (packet M13/Z4): from disk, and from stream 1
    /// while the run is watched. ③'s only (packet M13/Z5a): ④ shows its own attempts.
    pub previews: Vec<Preview>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Centre {
    /// No child runs and nothing is attached: a run that ended has its word on the light, and
    /// its pictures in ⑤.
    Idle,
    /// Collecting: the latest picture and the demonstrations so far.
    Demonstrations { made: u32, of: u32, succeeded: u32 },
    /// Training: the sample the network is fitting and the loss curve.
    Learning,
    /// Anything else: the latest picture.
    Picture,
}

/// What the centre shows; `alive` is a child that runs, or a run attached to.
pub fn centre(alive: bool, stage: Option<&str>, episodes: Episodes, of: u32) -> Centre {
    match (alive, stage) {
        (false, _) => Centre::Idle,
        (true, Some("collect")) => Centre::Demonstrations {
            made: episodes.made,
            of,
            succeeded: episodes.succeeded,
        },
        (true, Some("train")) => Centre::Learning,
        (true, _) => Centre::Picture,
    }
}

/// ⑤'s "train again on what failed", handed to ③ and kept until Start or Cancel (review of plan
/// Z, R1): what ③'s start panel says the next run will do before anything starts. How many new
/// demonstrations and how long are ③'s two settings, which stay editable.
#[derive(Clone, Debug, PartialEq)]
pub struct AgainPlan {
    /// What Start hands [`project::write_run_again`].
    pub again: Again,
    /// The Evaluation IR `again.config` names, read when the plan was handed over, for the
    /// situations' plain names; `None` (unreadable) keeps the raw suite names.
    ir: Option<EvaluationIr>,
}

impl AgainPlan {
    /// The situations the next run practises, in plain words, in the evaluation's order.
    pub fn practise(&self, lang: Lang) -> Vec<String> {
        (self.again.suites.iter())
            .map(|suite| results::suite_label(lang, suite, self.ir.as_ref()))
            .collect()
    }
}

/// ③ and ④ of one open project.
pub struct Watch {
    /// The run ③ and ④ follow: the project's latest, or the one started here.
    pub run: Option<RunFolder>,
    /// ③'s two settings.
    pub settings: StartSettings,
    /// ⑤'s plan, until Start starts it or Cancel drops it.
    again: Option<AgainPlan>,
    /// The template and the repository root `es` runs in, or the i18n key of why there are none.
    source: Result<(Template, PathBuf), &'static str>,
    /// The run's `[collect] episodes`.
    demonstrations: u32,
    /// Whether the launch model's child is this run: started or resumed here.
    ours: bool,
    /// An argv to start once no child runs: Start, Resume, and the evaluation queued behind a kill.
    queued: Option<Vec<String>>,
    dial: Dial,
    /// When the last dial was refused or the attachment closed: [`REDIAL_S`] counts from here.
    quiet_since: Option<Instant>,
    /// When watching began, and the count and time of the last message heard.
    since: Option<Instant>,
    heard: (u64, Option<Instant>),
    /// What disk says, re-read when a stage ends or the child changes.
    facts: Option<RunFacts>,
    facts_key: Option<(usize, usize, Option<Child>)>,
    failed: Option<(String, Option<i32>)>,
    ended: Option<Ended>,
    /// ④ on the previous frame: the done latch.
    evaluate: PhaseState,
    /// The run's previews: re-read with the facts, and whenever what stream 1 says of them
    /// changes.
    previews: Vec<Preview>,
    heard_previews: Vec<Preview>,
}

impl std::fmt::Debug for Watch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Watch")
            .field("run", &self.run)
            .field("ours", &self.ours)
            .field("queued", &self.queued)
            .field("ended", &self.ended)
            .finish_non_exhaustive()
    }
}

impl Watch {
    /// The project's latest run as disk has it. A run that did not finish is dialled once at its
    /// `telemetry.txt` address, on a thread of its own: an answer attaches, silence leaves it
    /// *interrupted* - and [`Self::tick`] dials again every [`REDIAL_S`] while it stays so.
    /// `repo_root` is `template::templates_root()`'s answer.
    pub fn new(project: &Project, repo_root: Option<PathBuf>) -> Self {
        let source = source_of(project, repo_root);
        let run = project.latest_run();
        let facts = run.as_ref().map(RunFacts::read);
        let disk = workflow::phases(facts.as_ref(), None);
        let dial = match should_dial(run.as_ref(), &disk) {
            Some(addr) => Dial::Dialling(launch::dial_in_background(
                addr,
                String::new(),
                1,
                launch::ATTACH_DELAY,
            )),
            None => Dial::Idle,
        };
        let previews = (run.as_ref()).map_or_else(Vec::new, |r| preview::previews(&r.path, &[]));
        Self {
            settings: StartSettings {
                demonstrations: source.as_ref().map_or(1, |(t, _)| t.demonstrations),
                length: Length::Medium,
            },
            again: None,
            demonstrations: run.as_ref().and_then(demonstrations_of).unwrap_or(0),
            source,
            run,
            ours: false,
            queued: None,
            dial,
            quiet_since: None,
            since: None,
            heard: (0, None),
            facts,
            facts_key: None,
            failed: None,
            ended: None,
            evaluate: disk[3].clone(),
            previews,
            heard_previews: Vec::new(),
        }
    }

    /// Once a frame, after the launch model and the telemetry were polled: the dial's answer,
    /// the queued start, what was heard, what disk says - and the five steps' states.
    pub fn tick(
        &mut self,
        launch: &mut LaunchModel,
        telemetry: &TelemetryModel,
        at: Phase,
        now: Instant,
    ) -> (Tick, [PhaseState; 5]) {
        let mut tick = Tick::default();
        if telemetry.received > HANDSHAKE && telemetry.received != self.heard.0 {
            self.heard = (telemetry.received, Some(now));
        }
        let answer = match &self.dial {
            Dial::Dialling(rx) | Dial::Redialling(rx) => match rx.try_recv() {
                Ok(answer) => Some(answer),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Err(String::new())),
            },
            // A closed attachment is none (packet M12/R3): the phases go back to disk.
            Dial::Attached(closed) if closed.get() => Some(Err(String::new())),
            _ => None,
        };
        match answer {
            Some(Ok(client)) => {
                let (source, closed) = telemetry_view::source_and_closed(client);
                self.dial = Dial::Attached(closed);
                self.ours = false;
                self.begin(now);
                tick.attached = Some(source);
            }
            Some(Err(_)) => {
                self.dial = Dial::Idle;
                self.quiet_since = Some(now);
            }
            None => {}
        }
        if launch.pid().is_none() {
            if let (Some(argv), Ok((_, root))) = (self.queued.take(), &self.source) {
                launch.start_in(&argv, root);
                self.ours = true;
                self.dial = Dial::Idle;
                self.begin(now);
                tick.started = true;
            }
        }
        // What the telemetry holds this frame belongs to the run before a start or an attach;
        // the window clears it once this returns.
        let fresh = tick.started || tick.attached.is_some();
        let rows: &[StageRow] = if fresh { &[] } else { telemetry.live.stages() };
        let child = self.child(launch);
        let key = (
            rows.len(),
            rows.iter().filter(|r| !r.running()).count(),
            child,
        );
        let reread = self.facts_key != Some(key);
        if reread {
            self.facts_key = Some(key);
            self.facts = self.run.as_ref().map(RunFacts::read);
        }
        let heard = preview::heard(if fresh { &[] } else { &telemetry.events });
        if reread || heard != self.heard_previews {
            self.previews =
                (self.run.as_ref()).map_or_else(Vec::new, |r| preview::previews(&r.path, &heard));
            self.heard_previews = heard;
        }
        // Nothing runs, nothing is attached and disk says the run did not finish: it may have
        // been resumed elsewhere, at a new address (packet M12/R3).
        let due = self
            .quiet_since
            .is_none_or(|t| now.saturating_duration_since(t) >= Duration::from_secs(REDIAL_S));
        let idle = matches!(self.dial, Dial::Idle) && launch.pid().is_none();
        if let Some(run) = self
            .run
            .as_ref()
            .filter(|_| idle && due && self.queued.is_none())
        {
            if unfinished(&workflow::phases(self.facts.as_ref(), None)) {
                self.dial = Dial::Redialling(redial(run.clone()));
            }
        }
        let live = child.map(|c| LiveFacts::new(rows, self.fraction(rows, telemetry), c));
        let mut phases = workflow::phases(self.facts.as_ref(), live.as_ref());
        self.failed = failed(&phases);
        let report = self.facts.as_ref().is_some_and(|f| f.report);
        self.ended = ending(child, self.failed.as_ref(), report);
        settle(&mut phases, self.ended);
        tick.done = just_done(&self.evaluate, &phases, at);
        self.evaluate = phases[3].clone();
        (tick, phases)
    }

    fn begin(&mut self, now: Instant) {
        self.since = Some(now);
        self.heard = (0, None);
        self.facts_key = None;
    }

    fn child(&self, launch: &LaunchModel) -> Option<Child> {
        if self.ours {
            return match launch.state() {
                State::Idle => None,
                State::Running { .. } => Some(Child::Running),
                State::Exited { .. } if launch.was_killed() => Some(Child::Killed),
                State::Exited { code, .. } => Some(Child::Exited(*code)),
                // The binary could not be started: nothing ran, and the status line says why.
                State::Failed(_) => Some(Child::Exited(-1)),
            };
        }
        matches!(self.dial, Dial::Attached(_)).then_some(Child::NotOurs)
    }

    fn fraction(&self, rows: &[StageRow], telemetry: &TelemetryModel) -> Option<f32> {
        let stage = rows.iter().rev().find(|r| r.running())?;
        let train = &telemetry.train;
        let made = episodes(&telemetry.events).made;
        progress(
            &stage.name,
            made,
            self.demonstrations,
            train.step(),
            train.total(),
        )
    }

    fn signal_of(
        &self,
        child: Child,
        rows: &[StageRow],
        telemetry: &TelemetryModel,
        now: Instant,
    ) -> Signal {
        let codes: Vec<Option<i32>> = rows.iter().map(|r| r.code.map(i32::from)).collect();
        let points = points(telemetry);
        let since = |t: Instant| now.saturating_duration_since(t).as_secs_f64();
        let input = Input {
            since_start_s: self.since.map_or(0.0, since),
            since_last_message_s: self.heard.1.map(since),
            child_alive: matches!(child, Child::Running | Child::NotOurs),
            killed: child == Child::Killed,
            exit_code: match child {
                Child::Exited(code) => Some(code),
                _ => None,
            },
            stage_codes: &codes,
            total_steps: telemetry.train.total().map(u64::from),
            points: &points,
        };
        signal(&input, self.ended, self.failed.as_ref())
    }

    /// What `phase`'s three panes draw, given the states [`Self::tick`] returned.
    pub fn view(
        &self,
        phase: Phase,
        launch: &LaunchModel,
        telemetry: &TelemetryModel,
        phases: &[PhaseState; 5],
        now: Instant,
    ) -> View {
        let child = self.child(launch);
        let rows = telemetry.live.stages();
        let state = phases[Phase::ALL.iter().position(|p| *p == phase).unwrap_or(0)].clone();
        // Attached mid-stage, its `stage.begin` was never heard: the phase's state names it.
        let stage = (rows.iter().rev().find(|r| r.running()))
            .map(|r| r.name.as_str())
            .or(running(&state));
        let train = &telemetry.train;
        let running = launch.pid().is_some();
        let checking = matches!(self.dial, Dial::Dialling(_));
        let free = self.source.is_ok()
            && !checking
            && may_start(launch.pid(), self.queued.is_some(), phases);
        let resume = phases[2..4].iter().find_map(|p| match p {
            PhaseState::Interrupted { resume_from } | PhaseState::StoppedByYou { resume_from } => {
                resume_from.clone()
            }
            _ => None,
        });
        View {
            cards: cards(phase, rows, &state),
            fraction: match &state {
                PhaseState::Running { fraction, .. } => *fraction,
                _ => None,
            },
            eta: stage
                .filter(|s| *s == "train")
                .and_then(|_| train.eta(train.total()?)),
            signal: child.map(|c| self.signal_of(c, rows, telemetry, now)),
            centre: centre(
                matches!(child, Some(Child::Running | Child::NotOurs)),
                stage,
                episodes(&telemetry.events),
                self.demonstrations,
            ),
            tiles: (phase == Phase::Evaluate).then(|| tiles(&telemetry.events)),
            start: free.then_some(if self.again.is_some() {
                "watch.again.start"
            } else if self.run.is_some() {
                "watch.start_over"
            } else {
                "watch.start"
            }),
            again: self.again.clone().filter(|_| free),
            resume: resume.filter(|_| free),
            stop: self.ours && running,
            evaluate_now: self.ours
                && running
                && stage == Some("train")
                && !train.checkpoints().is_empty(),
            checking,
            interrupted: !checking
                && child.is_none()
                && phases[2..4]
                    .iter()
                    .any(|p| matches!(p, PhaseState::Interrupted { .. })),
            cannot_start: self.source.as_ref().err().copied(),
            previews: if phase == Phase::Train {
                self.previews.clone()
            } else {
                Vec::new()
            },
            state,
        }
    }

    /// What ③'s preview player re-poses a motion on: the scene ⑤ uses for the same run
    /// ([`results::scene`]). Read from disk, so asked for when a player opens, not every frame.
    pub fn scene(&self) -> Option<PathBuf> {
        let cycle = self.run.as_ref().and_then(|r| {
            let text = std::fs::read_to_string(r.path.join(RUN_RECIPE)).ok()?;
            Cycle::parse(&text).ok()
        });
        let source = self.source.as_ref().ok();
        results::scene(
            cycle.as_ref(),
            source.map(|(t, _)| t),
            source.map(|(_, root)| root.as_path()),
        )
    }

    /// What ⑤ hands ③ (packets M12/Y13, M13/Z4b; review of plan Z, R1), which starts nothing.
    /// Run again: its settings, and any pending plan dropped. "Train again on what failed": its
    /// settings and its plan, which ③'s panel shows until Start starts it or Cancel drops it.
    /// No settings (a recipe the editor did not write) keeps the ones ③ has.
    pub fn prepare(&mut self, settings: Option<StartSettings>, again: Option<Again>) {
        if let Some(settings) = settings {
            self.settings = settings;
        }
        let root = self.source.as_ref().ok().map(|(_, root)| root);
        self.again = again.map(|again| AgainPlan {
            ir: root.and_then(|root| {
                let text = std::fs::read_to_string(root.join(&again.config)).ok()?;
                evaluation_from_toml(&text).ok()
            }),
            again,
        });
    }

    /// Cancel on the plan: Start is an ordinary start again.
    pub fn cancel_again(&mut self) {
        self.again = None;
    }

    /// Start: a new run folder, its recipe and its `telemetry.txt`, and `es loop cycle` queued
    /// for the next frame - the pending plan's ([`Self::start_again`]) when there is one.
    /// `Ok(false)`, with nothing written, whenever [`may_start`] says no.
    pub fn start(
        &mut self,
        project: &Project,
        pid: Option<u32>,
        phases: &[PhaseState; 5],
    ) -> Result<bool, ProjectError> {
        match self.again.as_ref().map(|plan| plan.again.clone()) {
            Some(again) => self.start_again(project, &again, pid, phases),
            None => self.start_with(project, None, pid, phases),
        }
    }

    /// ⑤'s "train again on what failed" (packet M13/Z4b): [`Self::start`], under the same
    /// [`may_start`] gate, with the recipe [`project::write_run_again`] writes. Once started, no
    /// plan is pending.
    pub fn start_again(
        &mut self,
        project: &Project,
        again: &Again,
        pid: Option<u32>,
        phases: &[PhaseState; 5],
    ) -> Result<bool, ProjectError> {
        let started = self.start_with(project, Some(again), pid, phases)?;
        if started {
            self.again = None;
        }
        Ok(started)
    }

    fn start_with(
        &mut self,
        project: &Project,
        again: Option<&Again>,
        pid: Option<u32>,
        phases: &[PhaseState; 5],
    ) -> Result<bool, ProjectError> {
        let Ok((template, root)) = &self.source else {
            return Ok(false);
        };
        if !may_start(pid, self.queued.is_some(), phases) {
            return Ok(false);
        }
        let path = project.next_run_dir();
        let addr = address(launch::free_local_port());
        let (s, p) = (self.settings, project);
        let argv = match again {
            None => project::write_run(template, root, p, s, &path, &addr)?,
            Some(a) => project::write_run_again(template, root, p, a, s, &path, &addr)?,
        };
        self.run = project.latest_run();
        self.demonstrations = self.settings.demonstrations;
        self.queued = Some(with_pictures(argv));
        Ok(true)
    }

    /// Resume the same run `--from` a stage, at a new address.
    pub fn resume(
        &mut self,
        from: &str,
        pid: Option<u32>,
        phases: &[PhaseState; 5],
    ) -> Result<bool, ProjectError> {
        let Some(run) = &self.run else {
            return Ok(false);
        };
        if self.source.is_err() || !may_start(pid, self.queued.is_some(), phases) {
            return Ok(false);
        }
        self.queued = Some(resume_command(run, from, launch::free_local_port())?);
        Ok(true)
    }

    /// Evaluate what it has learned so far: `[eval] checkpoint` set to the newest mark the ledger
    /// or a `checkpoint` event reports, the child killed, and `--from eval` queued for the frame
    /// it is gone. Whether `--from eval` accepts a run whose training stopped part-way is Y-V's
    /// item 3; until then a refusal shows on the light like any other stop.
    pub fn evaluate_now(
        &mut self,
        launch: &mut LaunchModel,
        checkpoints: &[(u32, String)],
    ) -> Result<bool, ProjectError> {
        let Some(run) = &self.run else {
            return Ok(false);
        };
        let ledger = read_loop_steps(&run.path).unwrap_or_default();
        let Some(mark) = newest_mark(&ledger, checkpoints) else {
            return Ok(false);
        };
        set_eval_checkpoint(run, mark)?;
        self.queued = Some(resume_command(run, "eval", launch::free_local_port())?);
        launch.kill();
        Ok(true)
    }
}

pub(crate) fn source_of(
    project: &Project,
    repo_root: Option<PathBuf>,
) -> Result<(Template, PathBuf), &'static str> {
    let root = repo_root.ok_or("watch.no_checkout")?;
    let (templates, _) = template::load(&root);
    let found = templates
        .into_iter()
        .find(|t| t.id == project.file.template)
        .ok_or("watch.no_template")?;
    Ok((found, root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::i18n::{t, Lang};
    use crate::model::project::tests::{cube, repo, scratch_project};
    use crate::model::telemetry_view::tests::hanging_up_peer;
    use es_core::PhysTick;
    use es_telemetry::protocol::{Frame, Message, Payload};
    use PhaseState::{Done, Failed, Interrupted, Locked, NotStarted, Running, StoppedByYou};

    fn fresh() -> [PhaseState; 5] {
        [Done, Done, NotStarted, Locked, Locked]
    }

    fn settings() -> StartSettings {
        StartSettings {
            demonstrations: 7,
            length: Length::Short,
        }
    }

    fn event(kind: &str, fields: &[(&str, &str)]) -> Event {
        Event {
            tick: 0,
            kind: kind.to_owned(),
            fields: fields
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        }
    }

    fn row(name: &str, code: Option<u8>) -> StageRow {
        StageRow {
            name: name.to_owned(),
            seconds: code.map(|_| 2.0),
            code,
        }
    }

    fn sleeper() -> (PathBuf, Vec<String>) {
        if cfg!(windows) {
            (
                "cmd".into(),
                vec!["/C".into(), "ping -n 60 127.0.0.1 > nul".into()],
            )
        } else {
            ("sh".into(), vec!["-c".into(), "sleep 60".into()])
        }
    }

    fn wait_for_exit(launch: &mut LaunchModel) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while launch.pid().is_some() {
            assert!(Instant::now() < deadline, "{:?}", launch.state());
            std::thread::sleep(Duration::from_millis(10));
            launch.poll();
        }
    }

    fn after<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
        argv.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].as_str())
    }

    /// Review focus 2: Start pressed while a child runs, or pressed twice before the child is
    /// up, makes no run folder - one child, one run directory.
    #[test]
    fn start_while_running_makes_no_run_folder() {
        assert!(may_start(None, false, &fresh()));
        assert!(!may_start(Some(7), false, &fresh()), "a child runs");
        assert!(!may_start(None, true, &fresh()), "a start is queued");
        let attached = [
            Done,
            Done,
            Running {
                stage: "train".into(),
                fraction: None,
            },
            Locked,
            Locked,
        ];
        assert!(
            !may_start(None, false, &attached),
            "an attached run goes on"
        );

        let p = scratch_project("y12-start");
        let mut watch = Watch::new(&p, Some(repo()));
        watch.settings = settings();
        let mut launch = LaunchModel::default();
        let (shell, args) = sleeper();
        launch.start_program(&shell, &args);
        assert_eq!(watch.start(&p, launch.pid(), &fresh()), Ok(false));
        assert!(
            p.runs().is_empty(),
            "a folder while a child runs: {:?}",
            p.runs()
        );
        launch.kill();
        wait_for_exit(&mut launch);

        assert_eq!(watch.start(&p, launch.pid(), &fresh()), Ok(true));
        assert_eq!(
            watch.start(&p, launch.pid(), &fresh()),
            Ok(false),
            "pressed twice"
        );
        assert_eq!(p.runs().len(), 1);
        let run = watch.run.clone().expect("the new run");
        assert_eq!(run, p.runs()[0]);
        let argv = watch.queued.clone().expect("queued for the next frame");
        assert_eq!(
            after(&argv, "--out"),
            Some(run.path.to_str().expect("utf-8"))
        );
        assert_eq!(after(&argv, "--telemetry-image-every"), Some(IMAGE_EVERY));
        assert_eq!(
            after(&argv, "--telemetry"),
            run.telemetry_addr().as_deref(),
            "telemetry.txt is the address the child publishes on"
        );

        // ⑤'s "train again" (packet M13/Z4b): the same gate, then the next run's Z3 recipe.
        let again = Again {
            config: "tests/fixtures/visible-learning/evaluation-v8.toml".into(),
            suites: vec!["nominal".into()],
            merge: vec![es_data::training::collect_root(&run.path)],
            init: run
                .path
                .join("train/checkpoints/5000.esb")
                .display()
                .to_string(),
        };
        let start_again = |w: &mut Watch| w.start_again(&p, &again, None, &fresh());
        assert_eq!(start_again(&mut watch), Ok(false), "a start is queued");
        watch.queued = None;
        assert_eq!(start_again(&mut watch), Ok(true));
        assert_eq!(p.runs().len(), 2);
        let next = watch.run.clone().expect("the again run");
        let text = std::fs::read_to_string(next.path.join(RUN_RECIPE)).unwrap();
        let cycle = Cycle::parse(&text).unwrap();
        assert_eq!(cycle.train.init, Some(again.init.clone()));
        assert_eq!(
            cycle.collect.map(|c| c.seed),
            Some(8),
            "after run 001's 1-7"
        );
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Review of plan Z, R1: ⑤'s "train again on what failed" prepares ③ and starts nothing.
    /// The panel shows the plan in plain words and Start starts it, with the settings as edited
    /// there; Cancel, or Run again, leaves an ordinary start.
    #[test]
    fn train_again_is_prepared_on_the_panel_and_started_by_start() {
        let p = scratch_project("r1-again");
        let mut watch = Watch::new(&p, Some(repo()));
        let (launch, telemetry, now) = (
            LaunchModel::default(),
            TelemetryModel::default(),
            Instant::now(),
        );
        let view = |w: &Watch| w.view(Phase::Train, &launch, &telemetry, &fresh(), now);
        assert_eq!(view(&watch).start, Some("watch.start"));
        assert_eq!(view(&watch).again, None);

        let again = Again {
            config: "tests/fixtures/visible-learning/evaluation.toml".into(),
            suites: vec!["light_intensity".into(), "torque_noise".into()],
            merge: vec![p.root.join("runs/000/collect/ds").display().to_string()],
            init: p
                .root
                .join("runs/000/train/checkpoints/5000.esb")
                .display()
                .to_string(),
        };
        let long = StartSettings {
            demonstrations: 200,
            length: Length::Long,
        };
        watch.prepare(Some(long), Some(again.clone()));
        assert!(
            p.runs().is_empty() && watch.queued.is_none(),
            "nothing starts"
        );
        assert_eq!(watch.settings, long);
        let shown = view(&watch);
        assert_eq!(shown.start, Some("watch.again.start"));
        let plan = shown.again.expect("the plan on the panel");
        assert_eq!(plan.again, again);
        for lang in Lang::ALL {
            let words = [
                t(lang, "perturb.light_intensity"),
                t(lang, "perturb.torque_noise"),
            ];
            assert_eq!(plan.practise(lang), words, "{lang:?}");
        }

        watch.cancel_again();
        assert_eq!(view(&watch).start, Some("watch.start"));
        assert_eq!(view(&watch).again, None);
        watch.prepare(Some(long), Some(again.clone()));
        watch.prepare(None, None);
        assert_eq!(view(&watch).again, None, "Run again drops the plan");
        assert_eq!(watch.settings, long, "no settings keeps ③'s");

        // Start: the plan's recipe under the settings as edited on the panel.
        watch.prepare(Some(long), Some(again.clone()));
        watch.settings = settings();
        assert_eq!(watch.start(&p, None, &fresh()), Ok(true));
        assert!(watch.again.is_none(), "started, so no longer pending");
        let run = watch.run.clone().expect("the again run");
        let text = std::fs::read_to_string(run.path.join(RUN_RECIPE)).unwrap();
        let cycle = Cycle::parse(&text).unwrap();
        assert_eq!(cycle.train.init, Some(again.init.clone()));
        let collect = cycle.collect.expect("[collect]");
        assert_eq!(collect.perturb.map(|x| x.suites), Some(again.suites));
        assert_eq!((collect.episodes, collect.merge), (7, again.merge));
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// `es loop cycle` exits 1 for a failed acceptance as for a crash (`cycle.rs`); only the
    /// first is the eval stage ending 1 beside its report, and it reads as finished, never as
    /// the red *stopped*.
    #[test]
    fn a_failed_acceptance_is_finished_not_stopped() {
        assert_eq!(ended(1, Some("eval"), true), Ended::DidNotPass);
        assert_eq!(
            ended(1, Some("eval"), false),
            Ended::Stopped,
            "a crash left no report"
        );
        assert_eq!(
            ended(1, Some("showcase"), true),
            Ended::Stopped,
            "the video failed"
        );
        assert_eq!(ended(1, Some("expert-gate"), false), Ended::Stopped);
        assert_eq!(ended(3, Some("collect"), false), Ended::Stopped);
        assert_eq!(ended(0, None, true), Ended::Passed);
        assert_eq!(
            ended(0, None, false),
            Ended::Stopped,
            "no result file is no pass"
        );

        // Final the moment eval says so: while showcase still runs, and for an attached run
        // whose exit this editor cannot see - never the red stop in between.
        let eval = ("eval".to_owned(), Some(1));
        for child in [
            Child::Running,
            Child::NotOurs,
            Child::Killed,
            Child::Exited(1),
        ] {
            assert_eq!(
                ending(Some(child), Some(&eval), true),
                Some(Ended::DidNotPass),
                "{child:?}"
            );
        }
        assert_eq!(ending(Some(Child::Running), Some(&eval), false), None);
        let gate = ("expert-gate".to_owned(), Some(1));
        assert_eq!(ending(Some(Child::Running), Some(&gate), false), None);
        assert_eq!(
            ending(Some(Child::Exited(1)), Some(&gate), false),
            Some(Ended::Stopped)
        );
        assert_eq!(
            ending(Some(Child::Killed), None, false),
            None,
            "the light says"
        );
        assert_eq!(ending(None, Some(&eval), true), None, "nothing watched");

        let codes = [Some(0), Some(0), Some(0), Some(1), Some(1)];
        let input = Input {
            since_start_s: 600.0,
            since_last_message_s: Some(1.0),
            child_alive: false,
            killed: false,
            exit_code: Some(1),
            stage_codes: &codes,
            total_steps: None,
            points: &[],
        };
        assert_eq!(
            health::judge(&input, &THRESHOLDS),
            Verdict::Stopped,
            "the codes alone"
        );
        let s = signal(&input, Some(Ended::DidNotPass), Some(&eval));
        assert_ne!(s.light, Light::Red);
        assert_eq!((s.name, s.stage), ("watch.did_not_pass", None));
        assert_eq!(
            signal(&input, Some(Ended::Passed), None).light,
            Light::Green
        );

        let mut phases = [
            Done,
            Done,
            Done,
            Failed {
                stage: "eval".into(),
                code: Some(1),
            },
            Done,
        ];
        let crashed = phases.clone();
        settle(&mut phases, Some(Ended::DidNotPass));
        assert_eq!(phases[3], Done, "④ ran to the end");
        let mut still = crashed.clone();
        settle(&mut still, Some(Ended::Stopped));
        assert_eq!(still, crashed);
        let rows = [row("eval", Some(1)), row("showcase", Some(1))];
        assert_eq!(
            cards(Phase::Evaluate, &rows, &phases[3]),
            [
                ("eval", CardState::Done(Some(2.0))),
                ("showcase", CardState::Done(Some(2.0)))
            ]
        );

        // A stop names its stage; a skipped stage and the expert gate have their own advice.
        let skipped = ("collect".to_owned(), Some(3));
        let s = signal(&input, Some(Ended::Stopped), Some(&skipped));
        assert_eq!(
            (s.light, s.name, s.advice, s.stage.as_deref()),
            (
                Light::Red,
                "health.stopped",
                "watch.advice.skipped",
                Some("collect")
            )
        );
        let s = signal(&input, Some(Ended::Stopped), Some(&gate));
        assert_eq!(s.advice, "watch.advice.expert_gate");
        let train = ("train".to_owned(), Some(1));
        let s = signal(&input, Some(Ended::Stopped), Some(&train));
        assert_eq!(s.advice, Verdict::Stopped.advice_key());

        // A kill is stopped by you, whatever the exit code says.
        let killed = Input {
            killed: true,
            ..input.clone()
        };
        let s = signal(&killed, None, None);
        assert_eq!(s.name, Verdict::StoppedByYou.key());
        assert_ne!(s.light, Light::Red);

        for key in [
            "watch.passed",
            "watch.passed.advice",
            "watch.did_not_pass",
            "watch.did_not_pass.advice",
            "watch.advice.skipped",
            "watch.advice.expert_gate",
        ] {
            for lang in Lang::ALL {
                assert_ne!(t(lang, key), key, "{lang:?}");
            }
        }
    }

    /// Resuming writes the new address into `telemetry.txt` before anything starts; evaluating
    /// now points `[eval] checkpoint` at the newest mark, which `es loop cycle` resolves.
    #[test]
    fn resuming_rewrites_the_address_and_evaluating_now_names_the_newest_mark() {
        let p = scratch_project("y12-resume");
        let path = p.next_run_dir();
        project::write_run(&cube(), &repo(), &p, settings(), &path, "127.0.0.1:7001").unwrap();
        let run = p.latest_run().expect("the run");
        let argv = resume_command(&run, "eval", 7123).unwrap();
        assert_eq!(run.telemetry_addr().as_deref(), Some("127.0.0.1:7123"));
        assert_eq!(after(&argv, "--telemetry"), Some("127.0.0.1:7123"));
        assert_eq!(after(&argv, "--from"), Some("eval"));
        assert_eq!(after(&argv, "--telemetry-image-every"), Some(IMAGE_EVERY));

        let ledger = [LoopStep::new(LoopKind::Train).output(&format!("{CHECKPOINT}2500"), &"ab")];
        assert_eq!(newest_mark(&ledger, &[]), Some(2500));
        assert_eq!(newest_mark(&ledger, &[(5000, String::new())]), Some(5000));
        assert_eq!(
            newest_mark(&[], &[]),
            None,
            "nothing saved: nothing to evaluate"
        );

        set_eval_checkpoint(&run, 2500).unwrap();
        let text = std::fs::read_to_string(run.path.join(RUN_RECIPE)).unwrap();
        let cycle = Cycle::parse(&text).unwrap();
        assert_eq!(cycle.eval.checkpoint, "2500");
        let recipe = cycle.training(None, &run.path).unwrap();
        assert_eq!(cycle.mark(&recipe).unwrap(), 2500);
        assert_eq!(demonstrations_of(&run), Some(7));
        std::fs::remove_dir_all(&p.root).ok();
    }

    fn train_frame(v: Vec<f64>) -> Message {
        Message::Frame(Frame {
            tick: PhysTick(0),
            wall_ns: 0,
            stream: STREAM_TRAIN,
            payload: Payload::Scalars(v),
        })
    }

    /// Progress is demonstrations made while collecting and steps while training; the tiles are
    /// the evaluation's cells, not the expert gate's; the curve the light reads keeps a NaN.
    #[test]
    fn progress_counts_demonstrations_steps_and_tiles() {
        let events = [
            event(
                "episode.end",
                &[("outcome", "Success"), ("stage", "collect")],
            ),
            event(
                "episode.end",
                &[("outcome", "Timeout"), ("stage", "collect")],
            ),
            event(
                "cell.end",
                &[
                    ("cell", "nominal-00"),
                    ("outcome", "Success"),
                    ("stage", "expert-gate"),
                ],
            ),
            event(
                "cell.end",
                &[
                    ("cell", "nominal-00"),
                    ("outcome", "Success"),
                    ("stage", "eval"),
                ],
            ),
            event(
                "cell.end",
                &[
                    ("cell", "nominal-01"),
                    ("outcome", "Timeout"),
                    ("stage", "eval"),
                ],
            ),
        ];
        let e = episodes(&events);
        assert_eq!((e.made, e.succeeded), (2, 1));
        assert_eq!(progress("collect", e.made, 8, None, None), Some(0.25));
        assert_eq!(progress("collect", 3, 0, None, None), None, "of nothing");
        assert_eq!(progress("train", 0, 8, Some(500), Some(1000)), Some(0.5));
        assert_eq!(
            progress("train", 0, 8, Some(500), None),
            None,
            "no total yet"
        );
        assert_eq!(progress("eval", 9, 8, Some(1), Some(2)), None);
        assert_eq!(
            tiles(&events),
            [
                Tile {
                    cell: "nominal-00".into(),
                    success: true
                },
                Tile {
                    cell: "nominal-01".into(),
                    success: false
                }
            ]
        );
        assert_eq!(
            centre(true, Some("collect"), e, 8),
            Centre::Demonstrations {
                made: 2,
                of: 8,
                succeeded: 1
            }
        );
        assert_eq!(centre(true, Some("train"), e, 8), Centre::Learning);
        assert_eq!(centre(true, None, e, 8), Centre::Picture);
        assert_eq!(centre(false, Some("train"), e, 8), Centre::Idle);

        let mut telemetry = TelemetryModel::default();
        telemetry.ingest(&train_frame(vec![10.0, 1.5, 1e-4, 40.0]));
        telemetry.ingest(&train_frame(vec![20.0, f64::NAN, 1e-4, 38.0]));
        let p = points(&telemetry);
        assert_eq!(p.len(), 2);
        assert_eq!((p[0].step, p[0].loss, p[0].samples_per_s), (10, 1.5, 40.0));
        assert!(
            p[1].loss.is_nan(),
            "a NaN is what makes the light say broken"
        );
    }

    /// The cards follow the stage rows; a stage a `--from` run skipped reads as done, and a stage
    /// the run died in reads as failed even when its `stage.end` never came.
    #[test]
    fn cards_follow_the_stage_rows() {
        let running = Running {
            stage: "expert-gate".into(),
            fraction: None,
        };
        let rows = [row("collect", Some(0)), row("expert-gate", None)];
        assert_eq!(
            cards(Phase::Train, &rows, &running),
            [
                ("collect", CardState::Done(Some(2.0))),
                ("expert-gate", CardState::Running),
                ("train", CardState::Waiting)
            ]
        );
        let died = Failed {
            stage: "expert-gate".into(),
            code: Some(1),
        };
        assert_eq!(cards(Phase::Train, &rows, &died)[1].1, CardState::Failed);
        let stopped = StoppedByYou { resume_from: None };
        assert_eq!(
            cards(Phase::Train, &rows, &stopped)[1].1,
            CardState::Stopped
        );
        let resumed = [row("train", None)];
        let train = Running {
            stage: "train".into(),
            fraction: Some(0.5),
        };
        assert_eq!(
            cards(Phase::Train, &resumed, &train),
            [
                ("collect", CardState::Done(None)),
                ("expert-gate", CardState::Done(None)),
                ("train", CardState::Running)
            ]
        );
        assert_eq!(
            cards(Phase::Train, &[], &train),
            [
                ("collect", CardState::Done(None)),
                ("expert-gate", CardState::Done(None)),
                ("train", CardState::Running)
            ],
            "attached mid-train: what began before was never heard"
        );
        let starting = Running {
            stage: "collect".into(),
            fraction: None,
        };
        assert_eq!(
            cards(Phase::Train, &[], &starting),
            [
                ("collect", CardState::Running),
                ("expert-gate", CardState::Waiting),
                ("train", CardState::Waiting)
            ],
            "nothing heard yet: the stage the child is about to begin"
        );
        assert_eq!(
            cards(Phase::Evaluate, &[], &Done),
            [
                ("eval", CardState::Done(None)),
                ("showcase", CardState::Done(None))
            ]
        );
        for card in [
            CardState::Waiting,
            CardState::Running,
            CardState::Done(None),
            CardState::Failed,
            CardState::Stopped,
        ] {
            for lang in Lang::ALL {
                assert_ne!(t(lang, card.key()), card.key(), "{card:?} {lang:?}");
            }
        }
        for length in LENGTHS {
            assert_ne!(t(Lang::Ko, length_key(length)), length_key(length));
        }
    }

    /// Attached while training runs, its `stage.begin` never heard: the step bar's running stage
    /// is what ③'s centre shows - the loss curve - and ④'s stays the picture.
    #[test]
    fn attached_mid_train_shows_the_training() {
        let p = scratch_project("attach-mid-train");
        let mut watch = Watch::new(&p, Some(repo()));
        watch.dial = Dial::Attached(Closed::default());
        let launch = LaunchModel::default();
        let telemetry = TelemetryModel::default();
        let now = Instant::now();
        let running = |stage: &str| Running {
            stage: stage.into(),
            fraction: None,
        };
        let phases = [Done, Done, running("train"), Locked, Locked];
        let view = watch.view(Phase::Train, &launch, &telemetry, &phases, now);
        assert_eq!(view.centre, Centre::Learning);
        assert_eq!(view.cards[0].1, CardState::Done(None));
        let phases = [Done, Done, Done, running("eval"), Locked];
        let view = watch.view(Phase::Evaluate, &launch, &telemetry, &phases, now);
        assert_eq!(view.centre, Centre::Picture);
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Review focus 3: a re-opened project dials its unfinished run's address once, and a
    /// finished one not at all. ④ becoming done notifies once and moves ③ or ④ on to ⑤.
    #[test]
    fn a_reopened_run_is_dialled_once_and_done_is_announced_once() {
        let p = scratch_project("y12-dial");
        let run = RunFolder {
            number: 1,
            path: p.next_run_dir(),
        };
        std::fs::create_dir_all(&run.path).unwrap();
        std::fs::write(run.path.join(TELEMETRY_FILE), "127.0.0.1:7555").unwrap();
        let interrupted = [
            Done,
            Done,
            Interrupted { resume_from: None },
            Locked,
            Locked,
        ];
        assert_eq!(
            should_dial(Some(&run), &interrupted).as_deref(),
            Some("127.0.0.1:7555")
        );
        assert_eq!(
            should_dial(Some(&run), &[Done, Done, Done, Done, Done]),
            None
        );
        assert_eq!(should_dial(None, &interrupted), None);
        std::fs::remove_dir_all(&p.root).ok();

        let evaluating = Running {
            stage: "eval".into(),
            fraction: None,
        };
        let all = [Done, Done, Done, Done, Done];
        assert_eq!(
            just_done(&evaluating, &all, Phase::Evaluate),
            Some(Phase::Results)
        );
        assert_eq!(
            just_done(&evaluating, &all, Phase::Train),
            Some(Phase::Results)
        );
        assert_eq!(
            just_done(&evaluating, &all, Phase::Scene),
            Some(Phase::Scene),
            "stays"
        );
        assert_eq!(just_done(&Done, &all, Phase::Evaluate), None, "once");
        let no_report = [Done, Done, Done, Done, Locked];
        assert_eq!(
            just_done(&evaluating, &no_report, Phase::Evaluate),
            Some(Phase::Evaluate),
            "⑤ does not open without a report"
        );
        assert_eq!(just_done(&evaluating, &fresh(), Phase::Train), None);
    }

    /// A kill reads as stopped by you on the step bar and on the light, through the launch
    /// model's own flag (packet M12/Y12's `was_killed`).
    #[test]
    fn a_kill_through_the_watch_is_stopped_by_you() {
        let p = scratch_project("y12-kill");
        let mut watch = Watch::new(&p, Some(repo()));
        watch.ours = true;
        watch.run = Some(RunFolder {
            number: 1,
            path: p.next_run_dir(),
        });
        let mut launch = LaunchModel::default();
        let (shell, args) = sleeper();
        launch.start_program(&shell, &args);
        let telemetry = TelemetryModel::default();
        let now = Instant::now();
        let (_, phases) = watch.tick(&mut launch, &telemetry, Phase::Train, now);
        assert!(matches!(&phases[2], Running { .. }), "{phases:?}");
        let view = watch.view(Phase::Train, &launch, &telemetry, &phases, now);
        assert!(view.stop && view.start.is_none());
        assert_eq!(
            view.signal.map(|s| s.name),
            Some(Verdict::Starting.key()),
            "nothing heard yet is not green"
        );

        launch.kill();
        wait_for_exit(&mut launch);
        let (_, phases) = watch.tick(&mut launch, &telemetry, Phase::Train, now);
        assert_eq!(phases[2], StoppedByYou { resume_from: None });
        let view = watch.view(Phase::Train, &launch, &telemetry, &phases, now);
        assert_eq!(
            view.signal.map(|s| s.name),
            Some(Verdict::StoppedByYou.key())
        );
        assert!(!view.stop);
        assert_eq!(view.start, Some("watch.start_over"));
        assert_eq!(
            view.centre,
            Centre::Idle,
            "a stopped run shows nothing live"
        );
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Packet M13/Z4: a re-opened run shows the previews its disk holds; watched, what stream 1
    /// says of them joins in, newest first.
    #[test]
    fn previews_come_from_disk_then_from_the_stream() {
        use crate::model::preview::tests::{begin, end, index_row};
        let p = scratch_project("z4-previews");
        let run = p.next_run_dir();
        std::fs::create_dir_all(run.join("preview")).unwrap();
        std::fs::write(run.join(preview::INDEX), index_row(1000, 1, 0) + "\n").unwrap();
        let mut watch = Watch::new(&p, Some(repo()));
        let mut launch = LaunchModel::default();
        let mut telemetry = TelemetryModel::default();
        let now = Instant::now();
        let mut steps = |telemetry: &TelemetryModel| {
            let (_, phases) = watch.tick(&mut launch, telemetry, Phase::Train, now);
            let view = watch.view(Phase::Train, &launch, telemetry, &phases, now);
            view.previews
                .iter()
                .map(|p| (p.step, p.code))
                .collect::<Vec<_>>()
        };
        assert_eq!(steps(&telemetry), [(1000, Some(0))], "from disk alone");
        telemetry
            .events
            .extend([end("5000", "3", "0"), begin("20000")]);
        assert_eq!(
            steps(&telemetry),
            [(20000, None), (5000, Some(0)), (1000, Some(0))]
        );
        let (_, phases) = watch.tick(&mut launch, &telemetry, Phase::Evaluate, now);
        let view = watch.view(Phase::Evaluate, &launch, &telemetry, &phases, now);
        assert!(view.previews.is_empty(), "④ shows its own attempts");
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Packet M13/Z5a: ③'s preview player poses a motion on the scene ⑤ uses for the run - its
    /// recipe's, else the template's.
    #[test]
    fn the_preview_player_poses_on_the_scene_results_uses() {
        let p = scratch_project("z5-scene");
        let template = repo().join(&cube().scene);
        assert_eq!(
            Watch::new(&p, Some(repo())).scene(),
            Some(template),
            "no run yet"
        );
        let run = p.next_run_dir();
        std::fs::create_dir_all(&run).unwrap();
        let fixture = repo().join("tests/fixtures/visible-learning/cycle.toml");
        let recipe = std::fs::read_to_string(fixture).unwrap();
        let elsewhere = "tests/fixtures/mjcf/elsewhere.xml";
        let moved = recipe.replace(&cube().scene, elsewhere);
        assert_ne!(moved, recipe, "the fixture names the template's scene");
        std::fs::write(run.join(RUN_RECIPE), moved).unwrap();
        assert_eq!(
            Watch::new(&p, Some(repo())).scene(),
            Some(repo().join(elsewhere)),
            "the recipe's"
        );
        assert_eq!(Watch::new(&p, None).scene(), None, "no checkout, no scene");
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Ticks at `now` until a dial attaches, and hands back the source it made.
    fn attach(watch: &mut Watch, launch: &mut LaunchModel, now: Instant) -> Source {
        let telemetry = TelemetryModel::default();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(Instant::now() < deadline, "nothing attached");
            if let Some(source) = watch.tick(launch, &telemetry, Phase::Train, now).0.attached {
                return source;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Packet M12/R3: an attachment whose connection closes is no attachment - no light, never
    /// *not responding*, and the phases disk's - and the run's `telemetry.txt` is dialled again
    /// `REDIAL_S` later and not before, so a run resumed elsewhere is picked up.
    #[test]
    fn a_closed_attachment_falls_back_to_disk_and_is_dialled_again() {
        let p = scratch_project("r3-closed");
        let run = RunFolder {
            number: 1,
            path: p.next_run_dir(),
        };
        std::fs::create_dir_all(&run.path).unwrap();
        let (first, hang_up) = hanging_up_peer();
        std::fs::write(run.path.join(TELEMETRY_FILE), &first).unwrap();
        let mut watch = Watch::new(&p, Some(repo()));
        let mut launch = LaunchModel::default();
        let t0 = Instant::now();
        let mut source = attach(&mut watch, &mut launch, t0);
        let mut telemetry = TelemetryModel::default();
        telemetry.pump(&mut source, 8);

        let late = t0 + Duration::from_secs(3600);
        let (_, phases) = watch.tick(&mut launch, &telemetry, Phase::Train, late);
        let view = watch.view(Phase::Train, &launch, &telemetry, &phases, late);
        assert_eq!(
            view.signal.map(|s| s.name),
            Some(Verdict::NotResponding.key()),
            "attached and silent"
        );

        drop(hang_up);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !matches!(&watch.dial, Dial::Attached(closed) if closed.get()) {
            assert!(Instant::now() < deadline, "the hang-up was never noticed");
            telemetry.pump(&mut source, 8);
            std::thread::sleep(Duration::from_millis(10));
        }
        let (tick, phases) = watch.tick(&mut launch, &telemetry, Phase::Train, late);
        assert!(tick.attached.is_none());
        let view = watch.view(Phase::Train, &launch, &telemetry, &phases, late);
        assert_eq!(
            view.signal, None,
            "no light at all, so never not responding"
        );
        assert_eq!(phases, workflow::phases(Some(&RunFacts::read(&run)), None));
        assert!(view.interrupted && view.resume.is_none() && view.start.is_some());

        // The run resumed elsewhere, publishing on a new address.
        let server =
            es_telemetry::transport::Server::bind("127.0.0.1:0".parse().unwrap(), None).unwrap();
        std::fs::write(
            run.path.join(TELEMETRY_FILE),
            server.local_addr().to_string(),
        )
        .unwrap();
        let redial = late + Duration::from_secs(REDIAL_S);
        let early = late + Duration::from_millis(REDIAL_S * 1000 - 1);
        watch.tick(&mut launch, &telemetry, Phase::Train, early);
        assert!(matches!(watch.dial, Dial::Idle), "not before REDIAL_S");
        let (_, phases) = watch.tick(&mut launch, &telemetry, Phase::Train, redial);
        assert!(matches!(watch.dial, Dial::Redialling(_)), "dialled again");
        assert!(
            !watch
                .view(Phase::Train, &launch, &telemetry, &phases, redial)
                .checking
        );
        let mut source = attach(&mut watch, &mut launch, redial);
        assert!(matches!(source(), Some(Message::HelloAck(_))));
        assert_eq!(watch.child(&launch), Some(Child::NotOurs));
        std::fs::remove_dir_all(&p.root).ok();
    }
}
