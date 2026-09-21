//! Starting a run from the editor (spec 23.1, spec 13.1; packet M7/E5).
//!
//! The editor **hosts nothing** (spec 23.1). It builds a command line out of what it already
//! knows, starts it as an ordinary child process whose semantics it does not own, shows the
//! child's exit code and its last lines, and then attaches to the telemetry it asked that
//! child to publish — as a client, exactly as if a person had typed the address into the
//! Telemetry tab. Nothing here evaluates or trains anything.
//!
//! Every decision lives in this file and none in `app.rs` (spec 28.10 rule 3): which flags a
//! kind renders, what the rendered command line is, which `es` binary is meant, what an exit
//! code means, and how long to wait for a producer's socket. `app.rs` draws a combo, some
//! text fields and two buttons over it.
//!
//! Three things are deliberately not here (spec 23.3): pause, step and reset — the run speaks
//! no control protocol, so [`LaunchModel::kill`] is the only control there can honestly be.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use crate::model::telemetry_view::{self, Source};

/// Lines of the child's stdout and stderr kept for the panel. The last 200: enough for a
/// usage error, a `SKIPPED (...)` line and the tail of a run, and bounded so a chatty child
/// cannot grow the editor's memory.
pub const RING: usize = 200;

/// Attach attempts after the child reports `Running`, and the wait between them.
///
/// `es eval run --telemetry` binds its server *before it opens anything* (`eval.rs`'s help),
/// so the socket is there early — but "early" is after process creation, linking and argument
/// parsing, which on a cold page cache is not instant. Two seconds of 100 ms polls is longer
/// than that and shorter than someone's patience.
pub const ATTACH_TRIES: usize = 20;
pub const ATTACH_DELAY: Duration = Duration::from_millis(100);

/// The default publish address (`eval.rs`'s help uses it as its example), used when the
/// Telemetry tab has nothing typed in it yet.
pub const DEFAULT_TELEMETRY: &str = "127.0.0.1:7777";

/// Which command the panel renders. One per stage of spec 13.1's loop that the editor can
/// start; the stages it cannot (`es loop collect`, `es video showcase`) are not here because
/// nothing in the session pre-fills them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    #[default]
    Eval,
    Train,
    Cycle,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Eval, Kind::Train, Kind::Cycle];

    /// What the combo shows: the command itself, not a noun for it.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Eval => "es eval run",
            Kind::Train => "es train",
            Kind::Cycle => "es loop cycle",
        }
    }

    /// The subcommand words, before any flag.
    fn words(self) -> &'static [&'static str] {
        match self {
            Kind::Eval => &["eval", "run"],
            Kind::Train => &["train"],
            Kind::Cycle => &["loop", "cycle"],
        }
    }
}

/// A flag that takes a value. The variants are the flags of the three commands and nothing
/// else — a field the CLI does not have cannot be typed into this panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchField {
    Config,
    Policy,
    Scene,
    Out,
    Frames,
    Jobs,
    Telemetry,
    TelemetryToken,
    TelemetryImageEvery,
    Recipe,
    From,
}

/// A flag that takes no value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchFlag {
    DryRun,
    AllowNewEvaluation,
    SkipExpertGate,
}

impl LaunchField {
    /// Every field, whichever kind renders it. What the panel *shows* is
    /// [`crate::model::labels::launch_label`], which is judged total over this
    /// (packet M7/E6).
    pub const ALL: [LaunchField; 11] = [
        Self::Config,
        Self::Policy,
        Self::Scene,
        Self::Out,
        Self::Frames,
        Self::Jobs,
        Self::Telemetry,
        Self::TelemetryToken,
        Self::TelemetryImageEvery,
        Self::Recipe,
        Self::From,
    ];

    /// The flag as the CLI spells it. Once the field's label, now its hover: the panel says
    /// what the field means and keeps the spelling it is going to type one hover away
    /// (packet M7/E6).
    pub fn flag(self) -> &'static str {
        match self {
            Self::Config => "--config",
            Self::Policy => "--policy",
            Self::Scene => "--scene",
            Self::Out => "--out",
            Self::Frames => "--frames",
            Self::Jobs => "--jobs",
            Self::Telemetry => "--telemetry",
            Self::TelemetryToken => "--telemetry-token",
            Self::TelemetryImageEvery => "--telemetry-image-every",
            Self::Recipe => "--recipe",
            Self::From => "--from",
        }
    }

    /// The greyed-out example in the empty field.
    pub fn hint(self) -> &'static str {
        match self {
            Self::Config => "eval.toml",
            Self::Policy => "policy.esb",
            Self::Scene => "scene.xml",
            Self::Out => "eval-out",
            Self::Frames => "eval-out/frames (optional)",
            Self::Jobs => "1 (optional)",
            Self::Telemetry => DEFAULT_TELEMETRY,
            Self::TelemetryToken => "token (optional)",
            Self::TelemetryImageEvery => "N ticks (optional)",
            Self::Recipe => "training.toml",
            Self::From => "collect | train | eval | showcase (optional)",
        }
    }
}

impl LaunchFlag {
    pub const ALL: [LaunchFlag; 3] =
        [Self::DryRun, Self::AllowNewEvaluation, Self::SkipExpertGate];

    pub fn flag(self) -> &'static str {
        match self {
            Self::DryRun => "--dry-run",
            Self::AllowNewEvaluation => "--allow-new-evaluation",
            Self::SkipExpertGate => "--skip-expert-gate",
        }
    }
}

use LaunchField as F;
use LaunchFlag as L;

/// The fields each kind renders, in the order the CLI's own help lists them.
const EVAL_FIELDS: &[F] = &[
    F::Config,
    F::Policy,
    F::Scene,
    F::Out,
    F::Frames,
    F::Jobs,
    F::Telemetry,
    F::TelemetryToken,
    F::TelemetryImageEvery,
];
const TRAIN_FIELDS: &[F] = &[F::Recipe, F::Out];
const CYCLE_FIELDS: &[F] = &[F::Recipe, F::Out, F::From];
const EVAL_FLAGS: &[L] = &[];
const TRAIN_FLAGS: &[L] = &[L::DryRun];
const CYCLE_FLAGS: &[L] = &[L::DryRun, L::AllowNewEvaluation, L::SkipExpertGate];

/// Where the `es` executable came from. Reported beside the command line so the person can
/// see *which* `es` is about to run before they press Start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EsBinary {
    pub path: PathBuf,
    /// One of `ES_BIN`, `beside the editor`, `on PATH` — the rule that found it.
    pub reason: String,
}

/// What the child is doing. `Exited` carries the ring as it was at exit, so the last lines
/// survive the next Start.
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Idle,
    Running {
        pid: u32,
        since: Instant,
    },
    Exited {
        code: i32,
        lines: Vec<String>,
    },
    /// The child could not be started at all — a missing binary, a denied execution. Not an
    /// exit code, because nothing ran.
    Failed(String),
}

/// **The exit codes the three commands document**, in one place (`eval.rs`, `train.rs` and
/// `cycle.rs`'s help agree on all four). The panel prints what this says; it decides nothing.
///
/// 3 is not a failure and must never be shown as one: it means this machine does not have the
/// backend or the runtime, so nothing ran (spec 1.4 — an evaluation this machine cannot run is
/// refused, never faked).
pub fn exit_meaning(code: i32) -> &'static str {
    match code {
        0 => "passed",
        1 => "failed, or a runtime error",
        2 => "usage error",
        3 => "skipped: a backend or runtime this machine does not have",
        _ => "ended without one of the documented codes (killed, or a crash)",
    }
}

/// The launch panel's whole state: the fields, the resolved binary, the child and its lines.
pub struct LaunchModel {
    pub kind: Kind,
    config: String,
    policy: String,
    scene: String,
    out: String,
    frames: String,
    jobs: String,
    telemetry: String,
    telemetry_token: String,
    telemetry_image_every: String,
    recipe: String,
    from: String,
    dry_run: bool,
    allow_new_evaluation: bool,
    skip_expert_gate: bool,

    /// Resolved once, at construction: the rule is one rule and the answer does not change
    /// while the editor is open.
    binary: EsBinary,
    state: State,
    child: Option<Child>,
    lines: Receiver<String>,
    ring: VecDeque<String>,
    /// Whether [`Self::kill`] ended this child. `TerminateProcess` exits 1 on Windows and a
    /// signal leaves no code at all on Unix, so the exit code alone cannot tell a killed run
    /// from a failed one — and reporting "failed" for something the person themselves ended
    /// would be the panel lying about the run.
    killed: bool,
}

impl std::fmt::Debug for LaunchModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaunchModel")
            .field("kind", &self.kind)
            .field("state", &self.state)
            .field("binary", &self.binary)
            .finish_non_exhaustive()
    }
}

impl Default for LaunchModel {
    fn default() -> Self {
        // A receiver with no sender: `try_recv` is `Disconnected` forever, which is exactly
        // "there is no child", so `poll` needs no `Option` around it.
        let (_, lines) = mpsc::channel();
        Self {
            kind: Kind::default(),
            config: String::new(),
            policy: String::new(),
            scene: String::new(),
            out: String::new(),
            frames: String::new(),
            jobs: String::new(),
            telemetry: DEFAULT_TELEMETRY.to_owned(),
            telemetry_token: String::new(),
            telemetry_image_every: String::new(),
            recipe: String::new(),
            from: String::new(),
            dry_run: false,
            allow_new_evaluation: false,
            skip_expert_gate: false,
            binary: es_binary(),
            state: State::Idle,
            child: None,
            lines,
            ring: VecDeque::new(),
            killed: false,
        }
    }
}

impl LaunchModel {
    // --- fields -------------------------------------------------------------------------

    /// The value-taking flags of the current kind, in order. `app.rs` draws one text box per
    /// entry and knows nothing about which flags a command has.
    pub fn fields(&self) -> &'static [LaunchField] {
        match self.kind {
            Kind::Eval => EVAL_FIELDS,
            Kind::Train => TRAIN_FIELDS,
            Kind::Cycle => CYCLE_FIELDS,
        }
    }

    /// The value-less flags of the current kind, in order.
    pub fn flags(&self) -> &'static [LaunchFlag] {
        match self.kind {
            Kind::Eval => EVAL_FLAGS,
            Kind::Train => TRAIN_FLAGS,
            Kind::Cycle => CYCLE_FLAGS,
        }
    }

    pub fn field(&self, f: LaunchField) -> &str {
        match f {
            F::Config => &self.config,
            F::Policy => &self.policy,
            F::Scene => &self.scene,
            F::Out => &self.out,
            F::Frames => &self.frames,
            F::Jobs => &self.jobs,
            F::Telemetry => &self.telemetry,
            F::TelemetryToken => &self.telemetry_token,
            F::TelemetryImageEvery => &self.telemetry_image_every,
            F::Recipe => &self.recipe,
            F::From => &self.from,
        }
    }

    /// What a text box edits.
    pub fn field_mut(&mut self, f: LaunchField) -> &mut String {
        match f {
            F::Config => &mut self.config,
            F::Policy => &mut self.policy,
            F::Scene => &mut self.scene,
            F::Out => &mut self.out,
            F::Frames => &mut self.frames,
            F::Jobs => &mut self.jobs,
            F::Telemetry => &mut self.telemetry,
            F::TelemetryToken => &mut self.telemetry_token,
            F::TelemetryImageEvery => &mut self.telemetry_image_every,
            F::Recipe => &mut self.recipe,
            F::From => &mut self.from,
        }
    }

    pub fn flag(&self, l: LaunchFlag) -> bool {
        match l {
            L::DryRun => self.dry_run,
            L::AllowNewEvaluation => self.allow_new_evaluation,
            L::SkipExpertGate => self.skip_expert_gate,
        }
    }

    /// What a checkbox toggles.
    pub fn flag_mut(&mut self, l: LaunchFlag) -> &mut bool {
        match l {
            L::DryRun => &mut self.dry_run,
            L::AllowNewEvaluation => &mut self.allow_new_evaluation,
            L::SkipExpertGate => &mut self.skip_expert_gate,
        }
    }

    /// Fills the empty fields from the session — never the typed ones, so opening a second
    /// bundle does not throw away a half-filled form.
    ///
    /// `--policy` is the open bundle's own path, `--out` the *parent* of an open run
    /// directory (a sibling of the run someone is looking at is where the next one goes),
    /// `--scene` whatever the Replay panel is already posing, and `--telemetry` the Telemetry
    /// tab's attach field.
    pub fn prefill(
        &mut self,
        bundle: Option<&Path>,
        run_dir: Option<&Path>,
        scene: &str,
        attach_addr: &str,
    ) {
        let fill = |slot: &mut String, value: &str| {
            if slot.trim().is_empty() && !value.trim().is_empty() {
                value.clone_into(slot);
            }
        };
        if let Some(bundle) = bundle {
            fill(&mut self.policy, &bundle.display().to_string());
        }
        if let Some(parent) = run_dir.and_then(Path::parent) {
            fill(&mut self.out, &parent.display().to_string());
        }
        fill(&mut self.scene, scene);
        fill(&mut self.telemetry, attach_addr);
    }

    // --- the command line ---------------------------------------------------------------

    /// The arguments, one per element, as a **pure function of the fields**: no environment,
    /// no filesystem, no normalisation the CLI does not do — a path is passed through exactly
    /// as it was typed, because the CLI is what resolves it.
    ///
    /// `argv[0]` is not here. The program is [`Self::binary`], which the environment decides,
    /// and keeping it out is what lets the rendering be a golden file.
    ///
    /// One rule for every flag: **an empty value is not rendered**. So `--frames` and
    /// `--jobs` simply disappear when nobody typed them, and a missing required flag reaches
    /// the CLI as a missing flag — which it refuses by name with exit 2, rather than as an
    /// empty string it would have to invent an error for.
    pub fn argv(&self) -> Vec<String> {
        let mut argv: Vec<String> = self.kind.words().iter().map(|w| (*w).to_owned()).collect();
        for f in self.fields() {
            let value = self.field(*f).trim();
            if !value.is_empty() {
                argv.push(f.flag().to_owned());
                argv.push(value.to_owned());
            }
        }
        for l in self.flags() {
            if self.flag(*l) {
                argv.push(l.flag().to_owned());
            }
        }
        argv
    }

    /// The binary this panel would run, and why it is that one.
    pub fn binary(&self) -> &EsBinary {
        &self.binary
    }

    /// The whole command line as one read-only line, for the panel.
    pub fn command_line(&self) -> String {
        let mut line = self.binary.path.display().to_string();
        for arg in self.argv() {
            line.push(' ');
            // Only what a shell would need quoting for, and only for reading: this string is
            // never parsed back, the child is started from the `Vec<String>` itself.
            if arg.contains(' ') {
                line.push('"');
                line.push_str(&arg);
                line.push('"');
            } else {
                line.push_str(&arg);
            }
        }
        line
    }

    // --- the child ----------------------------------------------------------------------

    /// Starts [`Self::binary`] with [`Self::argv`].
    pub fn start(&mut self) {
        let (program, args) = (self.binary.path.clone(), self.argv());
        self.start_program(&program, &args);
    }

    /// The whole of [`Self::start`] with the program as a parameter, so the oracle can start
    /// the platform shell down the same path a real run takes (spec 1.4: the harness must
    /// exercise the code, not a copy of it).
    pub fn start_program(&mut self, program: &Path, args: &[String]) {
        if matches!(self.state, State::Running { .. }) {
            return;
        }
        self.ring.clear();
        self.killed = false;
        let child = Command::new(program)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(e) => {
                self.state = State::Failed(format!("{}: {e}", program.display()));
                return;
            }
        };
        // Both pipes feed one channel from their own thread. The UI thread never reads a
        // pipe: a child that writes more than a pipe buffer would block the editor, and a
        // child that writes nothing would block it forever.
        let (tx, rx) = mpsc::channel();
        if let Some(out) = child.stdout.take() {
            reader(out, tx.clone());
        }
        if let Some(err) = child.stderr.take() {
            reader(err, tx);
        }
        self.state = State::Running {
            pid: child.id(),
            since: Instant::now(),
        };
        self.lines = rx;
        self.child = Some(child);
    }

    /// Drains whatever the reader threads have queued and asks the child whether it is still
    /// there. Called once a frame; never blocks while the child runs.
    pub fn poll(&mut self) {
        self.drain();
        let Some(child) = self.child.as_mut() else {
            return;
        };
        let status = match child.try_wait() {
            Ok(Some(status)) => status,
            Ok(None) => return,
            Err(e) => {
                self.state = State::Failed(format!("waiting on the child: {e}"));
                self.child = None;
                return;
            }
        };
        self.child = None;
        // The pipes are at EOF now, so the reader threads are finishing; block until they
        // drop their senders so the child's last lines — the usage error, the `SKIPPED`
        // reason — are not lost to whichever frame the exit landed in.
        while let Ok(line) = self.lines.recv() {
            self.push(line);
        }
        self.state = State::Exited {
            // A child killed by a signal has no code (Unix); `-1` is not one of the four
            // documented codes, and `exit_meaning` says so.
            code: status.code().unwrap_or(-1),
            lines: self.ring.iter().cloned().collect(),
        };
    }

    fn drain(&mut self) {
        // `Disconnected` is `Empty` here: the reader threads are gone, so there is nothing
        // more to come and nothing to report about it.
        while let Ok(line) = self.lines.try_recv() {
            self.push(line);
        }
    }

    fn push(&mut self, line: String) {
        if self.ring.len() == RING {
            self.ring.pop_front();
        }
        self.ring.push_back(line);
    }

    /// Ends the child. The **only** control there is: the run speaks no protocol, so pause,
    /// step and reset (spec 23.3) cannot be offered honestly and are not (packet M7/E5).
    pub fn kill(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            self.killed = true;
        }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// The last lines, oldest first, at most [`RING`].
    pub fn lines(&self) -> impl Iterator<Item = &str> {
        self.ring.iter().map(String::as_str)
    }

    /// The one line the panel shows under the buttons. Includes the exit code *and* what the
    /// CLI documents it to mean, so nobody has to remember that 3 is not a failure.
    pub fn status_line(&self) -> String {
        match &self.state {
            State::Idle => format!(
                "idle - {} ({})",
                self.binary.path.display(),
                self.binary.reason
            ),
            State::Running { pid, since } => {
                format!("running: pid {pid}, {:.1}s", since.elapsed().as_secs_f32())
            }
            State::Exited { code, .. } if self.killed => {
                format!("exit {code}: killed from here")
            }
            State::Exited { code, .. } => format!("exit {code}: {}", exit_meaning(*code)),
            State::Failed(e) => format!("could not start: {e}"),
        }
    }

    // --- attach -------------------------------------------------------------------------

    /// The address to attach to, once there is something to attach to.
    ///
    /// `None` until the child is `Running` — the editor is a client (spec 23.1) and a client
    /// dials a producer that exists — and `None` for a command that carries no `--telemetry`,
    /// which is every `es train` and `es loop cycle` there is today.
    pub fn attach(&self) -> Option<String> {
        if !matches!(self.state, State::Running { .. }) {
            return None;
        }
        let argv = self.argv();
        let i = argv.iter().position(|a| a == F::Telemetry.flag())?;
        argv.get(i + 1).cloned()
    }

    /// [`Self::attach`]'s address, dialled. `None` when there is nothing to attach to.
    ///
    /// The retry is here and not in `app.rs` (spec 28.10 rule 3): a producer binds before it
    /// opens anything, but "before" is still some milliseconds after the process started, and
    /// the panel must not decide how many.
    pub fn attach_source(&self) -> Option<Result<Source, String>> {
        let addr = self.attach()?;
        Some(dial(
            &addr,
            self.field(F::TelemetryToken),
            ATTACH_TRIES,
            ATTACH_DELAY,
        ))
    }
}

/// One pipe, read line by line into `tx` until it ends. A line that is not UTF-8 ends the
/// reading: this is a log pane, not a decoder.
fn reader(pipe: impl Read + Send + 'static, tx: mpsc::Sender<String>) {
    std::thread::spawn(move || {
        for line in BufReader::new(pipe).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
}

/// Attaches to `addr`, retrying a producer that has not opened its socket yet.
///
/// A malformed address is not retried — waiting `tries * delay` for a typo to fix itself is
/// time nobody has. Every other failure is, and the last one is what the caller is told.
pub fn dial(addr: &str, token: &str, tries: usize, delay: Duration) -> Result<Source, String> {
    addr.trim()
        .parse::<SocketAddr>()
        .map_err(|e| format!("{addr:?} is not a host:port address: {e}"))?;
    let tries = tries.max(1);
    let mut last = String::new();
    for i in 0..tries {
        if i > 0 {
            std::thread::sleep(delay);
        }
        match telemetry_view::attach(addr, token) {
            Ok(source) => return Ok(source),
            Err(e) => last = e,
        }
    }
    Err(format!(
        "nothing answered at {addr} in {tries} tries over {:?}: {last}",
        delay * tries as u32
    ))
}

/// **One rule** for which `es` this is: `ES_BIN` if it is set, else the `es` beside the
/// editor's own executable, else plain `es` for `PATH` to resolve.
///
/// The middle rule is the one that matters in practice: `cargo run -p es-editor` and
/// `cargo build -p es` put both binaries in the same `target/<profile>` directory, so the
/// editor starts the `es` it was built with rather than whatever is installed.
pub fn es_binary() -> EsBinary {
    resolve(
        std::env::var("ES_BIN").ok().as_deref(),
        std::env::current_exe()
            .ok()
            .as_deref()
            .and_then(Path::parent),
    )
}

/// The file name of the `es` binary on this platform (`es.exe` on Windows).
fn es_exe() -> String {
    format!("es{}", std::env::consts::EXE_SUFFIX)
}

/// [`es_binary`] with the environment as arguments, so the rule can be judged without setting
/// process-wide state in a test.
fn resolve(es_bin: Option<&str>, exe_dir: Option<&Path>) -> EsBinary {
    if let Some(path) = es_bin.map(str::trim).filter(|p| !p.is_empty()) {
        return EsBinary {
            path: PathBuf::from(path),
            reason: "ES_BIN".to_owned(),
        };
    }
    if let Some(sibling) = exe_dir.map(|d| d.join(es_exe())).filter(|p| p.is_file()) {
        return EsBinary {
            path: sibling,
            reason: "beside the editor".to_owned(),
        };
    }
    EsBinary {
        path: PathBuf::from(es_exe()),
        reason: "on PATH".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    fn root() -> PathBuf {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).to_path_buf()
    }

    fn golden_path(name: &str) -> PathBuf {
        root().join(format!("tests/golden/editor/launch-{name}.txt"))
    }

    /// The fixture form: every field of every kind filled, so the goldens show each flag
    /// rendered rather than each flag's absence. Paths are repository-relative and are passed
    /// through verbatim.
    fn fixture(kind: Kind) -> LaunchModel {
        let mut m = LaunchModel {
            kind,
            ..LaunchModel::default()
        };
        for (f, v) in [
            (F::Config, "tests/fixtures/visible-learning/evaluation.toml"),
            (F::Policy, "eval-out/policy.esb"),
            (F::Scene, "tests/fixtures/mjcf/so101_pick_place.xml"),
            (F::Out, "eval-out"),
            (F::Frames, "eval-out/frames"),
            (F::Jobs, "1"),
            (F::Telemetry, DEFAULT_TELEMETRY),
            (F::TelemetryToken, "s3cret"),
            (F::TelemetryImageEvery, "10"),
            (F::Recipe, "tests/fixtures/visible-learning/training.toml"),
            (F::From, "train"),
        ] {
            v.clone_into(m.field_mut(f));
        }
        for l in [L::DryRun, L::AllowNewEvaluation, L::SkipExpertGate] {
            *m.flag_mut(l) = true;
        }
        m
    }

    fn name(kind: Kind) -> &'static str {
        match kind {
            Kind::Eval => "eval",
            Kind::Train => "train",
            Kind::Cycle => "cycle",
        }
    }

    /// Writes `tests/golden/editor/launch-{eval,train,cycle}.txt`. Run **once**, explicitly,
    /// with `ES_GENERATE_GOLDENS=1` set; the files are read-only afterwards (spec 1.4).
    ///
    /// The environment variable is not belt and braces: `cargo test -- --include-ignored`
    /// sweeps up every `#[ignore]`d test in the workspace, and a generator that rewrites its
    /// own goldens under that sweep turns "the goldens still match" into a tautology (an M7
    /// review item). Without it this writes nothing and says so — a refusal and not a
    /// failure, so the sweep itself still passes. It is not a `SKIP`: nothing is missing from
    /// this machine, and `cargo xtask ci`'s oracle scan must not count it as an oracle.
    #[test]
    #[ignore = "golden generator; run explicitly with ES_GENERATE_GOLDENS=1"]
    fn generate_launch_goldens() {
        if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
            println!(
                "generate_launch_goldens: refusing to rewrite read-only goldens; set \
                 ES_GENERATE_GOLDENS=1 to mean it"
            );
            return;
        }
        fs::create_dir_all(golden_path("eval").parent().expect("a golden directory"))
            .expect("golden directory");
        for kind in Kind::ALL {
            let text = format!("{}\n", fixture(kind).argv().join("\n"));
            fs::write(golden_path(name(kind)), &text).expect("write the golden");
            println!("wrote {}", golden_path(name(kind)).display());
        }
    }

    /// Oracle 1: the rendering is the golden, byte for byte.
    #[test]
    fn launch_argv_is_the_golden() {
        for kind in Kind::ALL {
            let path = golden_path(name(kind));
            let expected = fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!(
                    "{}: {e} (run ES_GENERATE_GOLDENS=1 cargo test -p es-editor -- --ignored \
                     generate_launch_goldens)",
                    path.display()
                )
            });
            let rendered = format!("{}\n", fixture(kind).argv().join("\n"));
            assert_eq!(rendered, expected, "{} drifted", path.display());
        }

        // Every path reaches the CLI exactly as typed: no canonicalisation, no separator
        // rewriting, no quoting - the CLI is what resolves a path.
        let mut m = fixture(Kind::Eval);
        "C:\\tmp\\es m7\\eval.toml".clone_into(m.field_mut(F::Config));
        assert!(m.argv().contains(&"C:\\tmp\\es m7\\eval.toml".to_owned()));

        // An unset optional flag is absent, not empty.
        for f in [F::Frames, F::Jobs, F::TelemetryImageEvery] {
            m.field_mut(f).clear();
        }
        let argv = m.argv();
        assert!(!argv.iter().any(|a| a == "--frames"), "{argv:?}");
        assert!(!argv.iter().any(|a| a == "--jobs"), "{argv:?}");
        assert!(
            !argv.iter().any(|a| a == "--telemetry-image-every"),
            "{argv:?}"
        );
        assert!(!argv.iter().any(String::is_empty), "{argv:?}");

        // So is an unset boolean, on the kind that has three of them.
        let mut cycle = fixture(Kind::Cycle);
        for l in [L::DryRun, L::AllowNewEvaluation, L::SkipExpertGate] {
            *cycle.flag_mut(l) = false;
        }
        assert_eq!(
            cycle.argv(),
            vec![
                "loop",
                "cycle",
                "--recipe",
                "tests/fixtures/visible-learning/training.toml",
                "--out",
                "eval-out",
                "--from",
                "train",
            ]
        );

        // A kind renders its own flags and no other kind's.
        assert!(!fixture(Kind::Train).argv().iter().any(|a| a == "--config"));
        assert!(!fixture(Kind::Eval).argv().iter().any(|a| a == "--recipe"));
    }

    /// The platform's "run this one command line" shell, as oracle 2 names it.
    fn shell(script: &str) -> (PathBuf, Vec<String>) {
        if cfg!(windows) {
            (
                PathBuf::from("cmd"),
                vec!["/C".to_owned(), script.to_owned()],
            )
        } else {
            (
                PathBuf::from("sh"),
                vec!["-c".to_owned(), script.to_owned()],
            )
        }
    }

    fn sleeper() -> (PathBuf, Vec<String>) {
        shell(if cfg!(windows) {
            "ping -n 60 127.0.0.1 > nul"
        } else {
            "sleep 60"
        })
    }

    /// Polls until the child is no longer running, or gives up. Bounded: a `poll()` that
    /// never reaches `Exited` is the failure this asserts against.
    fn poll_until_exit(m: &mut LaunchModel) -> i32 {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            m.poll();
            if let State::Exited { code, .. } = m.state() {
                return *code;
            }
            assert!(Instant::now() < deadline, "still {:?} after 30s", m.state());
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Oracle 2: an exit code, the lines in order, and `kill` ending a sleeping child - all
    /// through the same `start_program` a real run goes through.
    #[test]
    fn launch_reports_the_exit_code() {
        let mut m = LaunchModel::default();
        assert_eq!(*m.state(), State::Idle);

        let (program, args) = shell("echo one && echo two && exit 3");
        m.start_program(&program, &args);
        let State::Running { pid, .. } = m.state() else {
            panic!("not running: {:?}", m.state())
        };
        assert!(*pid > 0);

        assert_eq!(poll_until_exit(&mut m), 3);
        let State::Exited { lines, .. } = m.state() else {
            unreachable!("poll_until_exit returned")
        };
        let stdout: Vec<&str> = lines
            .iter()
            .map(|l| l.trim())
            .filter(|l| *l == "one" || *l == "two")
            .collect();
        assert_eq!(stdout, vec!["one", "two"], "stdout in order: {lines:?}");
        assert_eq!(
            exit_meaning(3),
            "skipped: a backend or runtime this machine does not have"
        );

        // stderr reaches the same ring.
        let (program, args) = shell("echo boom 1>&2 && exit 2");
        m.start_program(&program, &args);
        assert_eq!(poll_until_exit(&mut m), 2);
        assert!(
            m.lines().any(|l| l.trim() == "boom"),
            "stderr is in the ring: {:?}",
            m.lines().collect::<Vec<_>>()
        );

        // Kill ends a child that would otherwise outlast the test.
        let (program, args) = sleeper();
        m.start_program(&program, &args);
        assert!(matches!(m.state(), State::Running { .. }));
        m.kill();
        assert_ne!(poll_until_exit(&mut m), 0, "a killed child did not succeed");
        // ... and is not reported as a failure, which its exit code alone would say.
        assert!(
            m.status_line().contains("killed from here"),
            "{}",
            m.status_line()
        );

        // Starting something that is not a program is `Failed`, not an exit code: nothing ran.
        m.start_program(Path::new("no-such-binary-es-editor-test"), &[]);
        assert!(matches!(m.state(), State::Failed(_)), "{:?}", m.state());
    }

    /// Oracle 2b: the ring keeps the *last* [`RING`] lines and nothing more.
    #[test]
    fn launch_keeps_the_last_two_hundred_lines() {
        let mut m = LaunchModel::default();
        let (program, args) = shell(if cfg!(windows) {
            "for /L %i in (1,1,250) do @echo %i"
        } else {
            "seq 1 250"
        });
        m.start_program(&program, &args);
        assert_eq!(poll_until_exit(&mut m), 0);
        let lines: Vec<String> = m.lines().map(str::to_owned).collect();
        assert_eq!(lines.len(), RING);
        assert_eq!(lines.first().map(|l| l.trim()), Some("51"));
        assert_eq!(lines.last().map(|l| l.trim()), Some("250"));
    }

    /// Oracle 3: attaching follows launching, and only for a command that publishes.
    #[test]
    fn launch_attaches_only_after_running() {
        let mut m = fixture(Kind::Eval);
        assert_eq!(m.attach(), None, "nothing to attach to while Idle");
        assert!(m.attach_source().is_none());

        let (program, args) = sleeper();
        m.start_program(&program, &args);
        assert_eq!(m.attach().as_deref(), Some(DEFAULT_TELEMETRY));

        // A command with no `--telemetry` has no address, running or not.
        m.field_mut(F::Telemetry).clear();
        assert_eq!(m.attach(), None);
        m.kill();
        poll_until_exit(&mut m);
        assert_eq!(m.attach(), None, "and none once it has exited");

        // `es train` and `es loop cycle` do not publish at all (design note section 13,
        // "Not here"), so neither kind can produce an address.
        for kind in [Kind::Train, Kind::Cycle] {
            let mut m = fixture(kind);
            m.start_program(&program, &args);
            assert_eq!(m.attach(), None, "{kind:?} carries no --telemetry");
            m.kill();
            poll_until_exit(&mut m);
        }
    }

    /// A free ephemeral port, reserved and released. `es_telemetry`'s own `Server` cannot be
    /// used for this - its accept thread is fire-and-forget and keeps the listener bound for
    /// the process's life (`transport.rs`'s own note) - so the reservation is a plain
    /// `TcpListener`, which does close on drop.
    fn free_port() -> SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a free port");
        let addr = listener.local_addr().expect("its number");
        drop(listener);
        addr
    }

    /// Oracle 3b: the retry gives a producer time to bind, and gives up with one error.
    #[test]
    fn dial_retries_a_producer_that_is_not_up_yet() {
        let addr = free_port().to_string();
        let started = Instant::now();
        let Err(e) = dial(&addr, "", 3, Duration::from_millis(20)) else {
            panic!("nothing is listening on {addr}")
        };
        assert!(e.contains(&addr) && e.contains("3 tries"), "{e}");
        assert!(started.elapsed() >= Duration::from_millis(40), "it retried");

        // A typo is not retried: it cannot become an address by waiting.
        let started = Instant::now();
        let Err(e) = dial("not-an-address", "", 50, Duration::from_millis(100)) else {
            panic!("that is not an address")
        };
        assert!(e.contains("host:port"), "{e}");
        assert!(started.elapsed() < Duration::from_secs(1), "and not slowly");

        // A producer that only binds after the first attempts have failed is still found -
        // which is the whole point of the loop, since `es eval run` binds a little after the
        // process starts.
        let late: SocketAddr = free_port();
        let binder = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(250));
            es_telemetry::transport::Server::bind(late, None)
        });
        let dialled = dial(&late.to_string(), "", ATTACH_TRIES, ATTACH_DELAY);
        match binder.join().expect("the binder thread") {
            // If the reserved port was taken between the reservation and the bind, there is
            // nothing to assert about: the producer never came up.
            Ok(_server) => assert!(dialled.is_ok(), "{late} bound late and was missed"),
            Err(e) => println!("skipping the late-bind case: {late} was taken ({e})"),
        }
    }

    /// Oracle 4: one rule, in order, and the reason names which part of it answered.
    #[test]
    fn es_binary_resolution_is_one_rule() {
        let dir = std::env::temp_dir().join("es-editor-launch-binary");
        fs::create_dir_all(&dir).expect("a temporary directory");
        let sibling = dir.join(es_exe());
        fs::write(&sibling, b"not really a binary").expect("write");

        // 1. ES_BIN wins, even when a sibling is right there.
        let found = resolve(Some("D:\\tools\\es.exe"), Some(&dir));
        assert_eq!(found.path, PathBuf::from("D:\\tools\\es.exe"));
        assert_eq!(found.reason, "ES_BIN");

        // 2. Then the `es` beside the editor's own executable.
        let found = resolve(None, Some(&dir));
        assert_eq!(found.path, sibling);
        assert_eq!(found.reason, "beside the editor");

        // 3. Then the bare name, for PATH to resolve.
        let found = resolve(None, Some(&dir.join("empty")));
        assert_eq!(found.path, PathBuf::from(es_exe()));
        assert_eq!(found.reason, "on PATH");
        assert_eq!(resolve(None, None).reason, "on PATH");

        // An empty or blank ES_BIN is not a choice; it falls through.
        assert_eq!(resolve(Some(""), Some(&dir)).reason, "beside the editor");
        assert_eq!(resolve(Some("   "), None).reason, "on PATH");

        // The real call answers with one of exactly those three reasons.
        let real = es_binary();
        assert!(
            ["ES_BIN", "beside the editor", "on PATH"].contains(&real.reason.as_str()),
            "{real:?}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// The panel's own strings: which fields a kind has, what the command line reads as, and
    /// what the status line says. All decided here, none in `app.rs` (spec 28.10 rule 3).
    #[test]
    fn the_panel_is_told_everything_it_draws() {
        let m = fixture(Kind::Eval);
        assert_eq!(m.fields().len(), 9, "the nine flags of `es eval run`");
        assert!(
            m.flags().is_empty(),
            "`es eval run` has no boolean flag here"
        );
        assert_eq!(m.fields()[0].flag(), "--config");
        assert!(m
            .command_line()
            .starts_with(&m.binary().path.display().to_string()));
        assert!(m.command_line().contains("eval run --config"));
        assert!(m.status_line().starts_with("idle - "));

        let train = fixture(Kind::Train);
        assert_eq!(train.fields(), TRAIN_FIELDS);
        assert_eq!(train.flags(), TRAIN_FLAGS);

        // A value with a space reads as one argument, and is still one element of `argv`.
        let mut m = fixture(Kind::Train);
        "C:\\es m7\\training.toml".clone_into(m.field_mut(F::Recipe));
        assert!(m.command_line().contains("\"C:\\es m7\\training.toml\""));
        assert!(m.argv().contains(&"C:\\es m7\\training.toml".to_owned()));

        // Every documented exit code has its own sentence, and 3 is not a failure.
        assert_eq!(exit_meaning(0), "passed");
        assert_eq!(exit_meaning(1), "failed, or a runtime error");
        assert_eq!(exit_meaning(2), "usage error");
        assert!(exit_meaning(3).starts_with("skipped"));
        assert!(exit_meaning(-1).contains("killed"));
    }

    /// Pre-filling takes from the session and never overwrites what someone typed.
    #[test]
    fn prefill_fills_only_the_empty_fields() {
        let mut m = LaunchModel::default();
        let run = root().join("tests/fixtures/visible-learning/run");
        m.prefill(
            Some(Path::new("out/policy.esb")),
            Some(&run),
            "scene.xml",
            "10.0.0.2:9000",
        );
        assert_eq!(m.field(F::Policy), "out/policy.esb");
        assert_eq!(
            m.field(F::Out),
            run.parent().expect("a parent").display().to_string()
        );
        assert_eq!(m.field(F::Scene), "scene.xml");
        // The telemetry field starts at the default, so the tab's address does not win.
        assert_eq!(m.field(F::Telemetry), DEFAULT_TELEMETRY);

        m.prefill(Some(Path::new("other.esb")), None, "other.xml", "");
        assert_eq!(m.field(F::Policy), "out/policy.esb", "typed values stand");
        assert_eq!(m.field(F::Scene), "scene.xml");

        // An empty telemetry field does take the tab's address.
        let mut m = LaunchModel::default();
        m.field_mut(F::Telemetry).clear();
        m.prefill(None, None, "", "10.0.0.2:9000");
        assert_eq!(m.field(F::Telemetry), "10.0.0.2:9000");
    }
}
