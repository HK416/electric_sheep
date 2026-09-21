//! The learning loop's collect and distill steps, and the `loop.jsonl` ledger that links
//! them (spec 13.1, spec 13.3).
//!
//! Read `docs/design/learning-loop.md` first. Two of its rules are the whole module:
//!
//! - **The Safety Plane is on the only actuator path** (`INV-12`). [`Collector::run`] drives
//!   [`es_env::Env::step_with_policy`], and an injected human action is wired in as a
//!   `PolicyRuntime` wrapper — so it becomes *just another chunk*, bounded by the same
//!   envelope and counted by the same counters as a policy chunk. There is no collect-mode
//!   branch around `validate`, tests included.
//! - **The training run is not here.** [`distill`] produces the *input identity* of a
//!   training run — merged dataset, deterministic split, `training_identity.json` — and the
//!   slots this side of the boundary cannot know are all-zero digests, never fabricated
//!   (spec 19.3, spec 2.3: training is on the Python path).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use es_assets::scene::SceneDesc;
use es_compile::{PolicyBundle, Tensor};
use es_core::{PhysTick, TickRate};
use es_env::scheduler::BatchDomains;
use es_env::traj::Trajectory;
use es_env::{DomainRunner, Env, Termination};
use es_ir::learning::{ChunkBlendPolicy, LearningGraph, LearningNode};
use es_ir::types::ElemType;
use es_physics_core::backend::{ModelInfo, PhysicsBackend, StateView};
use es_policy::{PolicyError, PolicyInfo, PolicyRuntime, WeightsSource};
use es_safety::{EventSet, SafetyPlane, ViolationKind};
use serde::{Deserialize, Serialize};

use crate::identity::{BaseModel, DatasetIdentity, Split, TrainingIdentity};
use crate::intervention::{
    identity_of, segments_of, ActionSourceCode, InterventionSegment, InterventionSource,
    ACTION_SOURCE, INTERVENTION,
};
use crate::lerobot::meta::{Dtype, FeatureSpec, Info};
use crate::lerobot::{Column, Episode, LeRobotDataset, LeRobotWriter};
use crate::{write_file, DataError};

/// Where the loop ledger lives, relative to a dataset root.
pub const LOOP_FILE: &str = "loop.jsonl";
/// Where [`distill`] writes the spec 19.3 identity.
pub const TRAINING_IDENTITY_FILE: &str = "training_identity.json";
/// The raw pre-plane command, beside `action` (spec 13.2).
pub const ACTION_COMMANDED: &str = "action_commanded";

pub(crate) fn hex(d: &[u8; 32]) -> String {
    d.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// --- the loop ledger (spec 13.3) ------------------------------------------------------------

/// Which step of spec 13.1's loop a [`LoopStep`] records.
///
/// [`Train`](Self::Train) and [`Evaluate`](Self::Evaluate) are `es loop cycle`'s (packet
/// M7/T2): the ledger stopped at `distill` and so never recorded that a dataset trained a
/// policy or that a policy was judged. Adding variants leaves every line an older `es` wrote
/// readable — the tag is the only thing serde matches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopKind {
    Collect,
    Intervene,
    Distill,
    Train,
    Evaluate,
}

/// The key prefix a [`LoopKind::Train`] step's checkpoint outputs carry: `checkpoint.<step>`
/// -> the spec 19.3 `policy_hash` of that mark. One key per checkpoint rather than one packed
/// value, because membership is what [`check_chain`] asks of it.
pub const CHECKPOINT: &str = "checkpoint.";

/// One line of `loop.jsonl`: what a loop step consumed and what it produced (spec 13.3).
///
/// `created` is the only non-reproducible value in the file, for the same reason
/// `evaluation.lock`'s is (spec 10.4): provenance is not identity. Nothing here feeds a hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopStep {
    pub kind: LoopKind,
    pub inputs: BTreeMap<String, String>,
    pub outputs: BTreeMap<String, String>,
    pub created: u64,
}

impl LoopStep {
    pub fn new(kind: LoopKind) -> Self {
        Self {
            kind,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            created: now_unix(),
        }
    }

    #[must_use]
    pub fn input(mut self, key: &str, value: &impl ToString) -> Self {
        self.inputs.insert(key.to_owned(), value.to_string());
        self
    }

    #[must_use]
    pub fn output(mut self, key: &str, value: &impl ToString) -> Self {
        self.outputs.insert(key.to_owned(), value.to_string());
        self
    }
}

/// Appends one step to `<root>/loop.jsonl`. Append-only: the ledger is never rewritten, so a
/// chain that does not link up stays visible (spec 13.3).
pub fn append_loop_step(root: &Path, step: &LoopStep) -> Result<(), DataError> {
    let path = root.join(LOOP_FILE);
    let mut text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(DataError::io(&path, e)),
    };
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(
        &serde_json::to_string(step).map_err(|source| DataError::Json {
            path: path.clone(),
            source,
        })?,
    );
    text.push('\n');
    write_file(&path, text.as_bytes())
}

/// Reads `<root>/loop.jsonl`; an absent file is an empty ledger.
pub fn read_loop_steps(root: &Path) -> Result<Vec<LoopStep>, DataError> {
    let path = root.join(LOOP_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(DataError::io(&path, e)),
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).map_err(|source| DataError::Json {
                path: path.clone(),
                source,
            })
        })
        .collect()
}

/// Spec 13.3's chain, checked over one ledger: a `train` consumed the dataset the `collect`
/// before it produced, and an `evaluate` judged a checkpoint the `train` before it wrote.
///
/// Both halves are conditional on there being an earlier step to chain to — a ledger that
/// only holds an `evaluate` (a bare `es eval run` against a bundle from elsewhere) claims no
/// chain and breaks none. The expert gate of §28.9 rule 1 is an `evaluate` step carrying an
/// `expert` input: it judges the scripted demonstrator rather than a checkpoint, so it is
/// skipped by name rather than by position.
pub fn check_chain(steps: &[LoopStep]) -> Result<(), DataError> {
    let mut content: Option<&str> = None;
    let mut checkpoints: Vec<(&str, &str)> = Vec::new();
    for step in steps {
        match step.kind {
            LoopKind::Collect | LoopKind::Intervene | LoopKind::Distill => {
                if let Some(c) = step.outputs.get("content") {
                    content = Some(c);
                }
            }
            LoopKind::Train => {
                let read = step.inputs.get("content").map_or("-", String::as_str);
                if let Some(want) = content {
                    if read != want {
                        return Err(refuse(format!(
                            "the ledger does not chain (spec 13.3): the train step reads dataset \
                             content {read} and the collect step before it wrote {want}"
                        )));
                    }
                }
                for (key, hash) in &step.outputs {
                    if let Some(mark) = key.strip_prefix(CHECKPOINT) {
                        checkpoints.push((mark, hash));
                    }
                }
            }
            LoopKind::Evaluate => {
                if step.inputs.contains_key("expert") {
                    continue;
                }
                let judged = step.inputs.get("policy_hash").map_or("-", String::as_str);
                if !checkpoints.is_empty() && !checkpoints.iter().any(|(_, h)| *h == judged) {
                    let known: Vec<String> = checkpoints
                        .iter()
                        .map(|(m, h)| format!("{m}={h}"))
                        .collect();
                    return Err(refuse(format!(
                        "the ledger does not chain (spec 13.3): the evaluate step judged \
                         policy_hash {judged}, which is not a checkpoint of any train step \
                         before it ({})",
                        known.join(" ")
                    )));
                }
            }
        }
    }
    Ok(())
}

/// The `evaluation_hash` of the last `evaluate` step in a ledger, if there is one.
///
/// Spec 13.3's discipline — hold the evaluation conditions fixed while data and policy move —
/// is a comparison between this and the next iteration's; `es loop cycle` refuses rather than
/// warns when the two differ.
pub fn last_evaluation_hash(steps: &[LoopStep]) -> Option<&str> {
    steps
        .iter()
        .rev()
        .filter(|s| s.kind == LoopKind::Evaluate)
        .find_map(|s| s.inputs.get("evaluation_hash").map(String::as_str))
}

fn refuse(msg: impl Into<String>) -> DataError {
    DataError::Loop(msg.into())
}

// --- collection -----------------------------------------------------------------------------

/// What a scripted intervener does with one control tick.
///
/// [`Abort`](Self::Abort) exists because a scripted driver can run out of answers — a waypoint
/// outside the robot's workspace, say — and the honest record of that is a demonstration that
/// ended in failure, not one clamped to something reachable (spec 17.2).
#[derive(Clone, Debug, PartialEq)]
pub enum Intervention<const NJ: usize> {
    /// Leave this tick to the policy.
    Policy,
    /// Drive this tick with this action, held for the whole chunk horizon. It still travels
    /// chunk buffer -> `SafetyPlane` -> `ctrl` like any other (`INV-12`).
    Action([f64; NJ]),
    /// Drive the next ticks with these actions, one row per control tick. A row short of the
    /// horizon repeats the last one; rows past it are dropped. A scripted driver that paces
    /// itself to the Safety Plane's envelope emits a chunk rather than a step, because a step
    /// is what the envelope has to clamp (design note section 5.3).
    Chunk(Vec<[f64; NJ]>),
    /// Stop here: the episode is recorded as a failed demonstration.
    Abort,
}

/// The scripted-intervention hook: `Fn(episode, frame, &model, &obs) -> Intervention<NJ>`.
///
/// `obs` is the observation the policy was handed for that control tick (the `qpos ‖ qvel` row
/// of spec 12.2's raw path) and `model` says where each joint sits in it, so a scripted
/// intervener is a pure function of the state and the run stays bit-reproducible for a seed —
/// which is what makes it usable as an oracle.
pub type Intervener<'a, const NJ: usize> =
    &'a mut dyn FnMut(u32, u32, &ModelInfo, &[f64]) -> Intervention<NJ>;

/// Called once per control step with the state that step is entered with -- the same
/// instant the `observation.state` row and the `.estraj` pose of that step carry (M5/V12).
///
/// A closure, not a renderer: `es-data` is layer 10 and `es-render` layer 5, and this is the
/// same trade `es_eval::runner::FrameSource` makes — the caller owns
/// `es_env::render::EnvRenderer` (feature `render`) and hands its `frame` in through this, so
/// nothing here links Vulkan. With a sink, `info.json`'s video feature stops being a dangling
/// reference.
pub type FrameSink<'a> = &'a mut dyn FnMut(&ModelInfo, &StateView<'_>) -> Result<(), String>;

/// One moment of a running collection, for a [`CollectSink`] (packet M7/E7).
///
/// The shape [`es_eval::runner::RunEvent`] has for an evaluation, for the same reason: a
/// collection that publishes what it is doing must publish what it *already had*, so a viewer
/// sees the plane's own verdict and not a second derivation of it. `es-data` links no
/// transport (layer 10 beside `es-telemetry`, spec 4.2 forbids the dependency); the caller
/// turns these into wire frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectEvent {
    /// One episode starts, with the seed the whole run was drawn from.
    EpisodeBegin { episode: u32, seed: u64 },
    /// One control tick, as the dataset's own `action_source` column records it and with the
    /// `es_safety::EventSet` bits of that step — the two counters' delta, which is exactly
    /// the set the plane raised (one env, one `validate` per step).
    Tick {
        episode: u32,
        frame: u32,
        tick: PhysTick,
        source: ActionSourceCode,
        events: u32,
    },
    /// The episode is over and its rows are written.
    EpisodeEnd {
        episode: u32,
        outcome: Termination,
        steps: usize,
    },
}

/// The collection sink: a closure, not an eighth extension point (`INV-17`). `None` is the
/// run of before — nothing is computed for a sink that is not there.
pub type CollectSink<'a> = &'a mut dyn FnMut(CollectEvent);

/// What [`Collector::run`] is given that the bundle does not say.
#[derive(Debug)]
pub struct CollectSpec<'a> {
    pub bundle: &'a PolicyBundle,
    pub scene: &'a SceneDesc,
    pub n_episodes: u32,
    pub seed: u64,
    /// Episode step budget; `0` uses the task's own `max_episode_steps`.
    pub max_steps: u32,
    pub out_root: &'a Path,
    /// Where to write each episode's `.estraj` state trajectory (`es_env::traj`, packet
    /// M5/V9). `None` writes none. Not inside the `LeRobot` dataset's own files: nothing in
    /// `dataset_content_hash` or `dataset_schema_hash` moves because of it.
    pub traj_dir: Option<PathBuf>,
}

/// What one collect run produced.
#[derive(Clone, Debug)]
pub struct CollectReport {
    pub root: PathBuf,
    pub episodes: u32,
    pub frames: u64,
    pub intervention_frames: u64,
    pub segments: Vec<InterventionSegment>,
    /// How each episode ended, in episode order — the demonstration's own verdict, which the
    /// `LeRobot` columns have no place for.
    pub terminations: Vec<Termination>,
    /// Frames handed to the [`FrameSink`], if there was one.
    pub rendered: u64,
    /// What the Safety Plane did across the run: spec 10.3's `failure_mode_histogram`, plus
    /// the clamp and fallback counts `action_source` is classified from.
    pub safety: es_safety::SafetyCounters,
    pub content: [u8; 32],
    pub schema: [u8; 32],
    /// Honest gaps, one line each — currently the image channels for which `info.json`
    /// declares a video feature but no mp4 was written (design note section 3.1).
    pub warnings: Vec<String>,
}

/// A `PolicyRuntime` that lets a scripted intervener take over a control tick.
///
/// The injected action is emitted as the policy's action chunk, so it travels the ordinary
/// path: chunk buffer -> `SafetyPlane::validate` -> `ctrl`. Nothing is bypassed (`INV-12`).
struct Intervened<'a, const NJ: usize, const H: usize> {
    inner: &'a mut dyn PolicyRuntime,
    intervener: Intervener<'a, NJ>,
    action_port: String,
    /// The loaded model, so the intervener can find a joint inside the observation row.
    model: ModelInfo,
    episode: u32,
    frame: u32,
    /// Set by the last `infer` call: did the intervener take this tick?
    injected: bool,
    /// Set by the last `infer` call: did the intervener give up?
    aborted: bool,
    /// The last chunk the intervener asked for, repeated on an abort.
    last: Vec<[f64; NJ]>,
}

impl<const NJ: usize, const H: usize> PolicyRuntime for Intervened<'_, NJ, H> {
    fn load(
        &mut self,
        graph: &LearningGraph,
        weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError> {
        self.inner.load(graph, weights)
    }

    fn infer(
        &mut self,
        inputs: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>, PolicyError> {
        let obs = inputs.values().next().map_or_else(Vec::new, tensor_to_f64);
        self.injected = false;
        let taken = (self.intervener)(self.episode, self.frame, &self.model, &obs);
        self.aborted = taken == Intervention::Abort;
        // A driver that gave up repeats what it last asked for -- through the plane, like every
        // other chunk (`INV-12`) -- and the collector ends the episode on the flag above. The
        // alternative, an empty chunk, is a watchdog event the demonstration did not have.
        let rows = match taken {
            Intervention::Action(a) => Some(vec![a]),
            Intervention::Chunk(rows) if !rows.is_empty() => Some(rows),
            // An empty chunk is nothing to execute, which is the policy's tick.
            Intervention::Policy | Intervention::Chunk(_) => None,
            Intervention::Abort => Some(self.last.clone()),
        };
        if let Some(rows) = &rows {
            if !self.aborted {
                self.injected = true;
                self.last.clone_from(rows);
            }
        }
        if let Some(rows) = rows {
            let last = *rows.last().unwrap_or(&[0.0; NJ]);
            let mut data = Vec::with_capacity(H * NJ * 4);
            for k in 0..H {
                for v in rows.get(k).unwrap_or(&last) {
                    data.extend_from_slice(&(*v as f32).to_le_bytes());
                }
            }
            return Ok(BTreeMap::from([(
                self.action_port.clone(),
                Tensor {
                    dtype: ElemType::F32,
                    shape: vec![H as u64, NJ as u64],
                    data,
                },
            )]));
        }
        self.inner.infer(inputs)
    }

    fn info(&self) -> Option<&PolicyInfo> {
        self.inner.info()
    }

    fn runtime_hash(&self) -> [u8; 32] {
        self.inner.runtime_hash()
    }
}

fn tensor_to_f64(t: &Tensor) -> Vec<f64> {
    match t.dtype {
        ElemType::F32 => t
            .data
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        ElemType::F64 => t
            .data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        _ => Vec::new(),
    }
}

/// The blend the Learning IR's `ActionChunker` declares (spec 8.5). A graph without one
/// switches hard, which is the behaviour of a policy that replans every chunk.
fn chunk_blend(learning: &LearningGraph) -> ChunkBlendPolicy {
    learning
        .nodes
        .nodes
        .values()
        .find_map(|n| match n {
            LearningNode::ActionChunker { blend, .. } => Some(*blend),
            _ => None,
        })
        .unwrap_or(ChunkBlendPolicy::HardSwitch)
}

/// Namespace for [`Collector::run`]. Not a trait and not state: `INV-17` allows seven
/// extension points and this is none of them.
#[derive(Debug)]
pub struct Collector;

impl Collector {
    /// Rolls out `n_episodes` episodes and writes them as a `LeRobot` dataset (spec 13.1).
    ///
    /// `NJ` / `H` are the deployment's joint count and chunk horizon; `SafetyPlane::from_ir`
    /// and `DomainRunner::new` both reject a mismatch, so they cannot silently disagree with
    /// the bundle.
    pub fn run<B, F, const NJ: usize, const H: usize>(
        spec: &CollectSpec<'_>,
        policy: &mut dyn PolicyRuntime,
        new_backend: F,
        intervener: Intervener<'_, NJ>,
        frame_sink: Option<FrameSink<'_>>,
    ) -> Result<CollectReport, DataError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
        Self::run_with_sink::<B, F, NJ, H>(spec, policy, new_backend, intervener, frame_sink, None)
    }

    /// [`Self::run`] with a [`CollectSink`]: the same run, saying what it does as it does it
    /// (packet M7/E7).
    ///
    /// With `None` this is the run of before, byte for byte — nothing is computed for a sink
    /// that is not there, and the two per-step numbers a sink is given (the source code and
    /// the violation bits) come out of counters this loop already snapshots.
    pub fn run_with_sink<B, F, const NJ: usize, const H: usize>(
        spec: &CollectSpec<'_>,
        policy: &mut dyn PolicyRuntime,
        mut new_backend: F,
        intervener: Intervener<'_, NJ>,
        mut frame_sink: Option<FrameSink<'_>>,
        mut sink: Option<CollectSink<'_>>,
    ) -> Result<CollectReport, DataError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
        let bundle = spec.bundle;
        let deploy = &bundle.deployment;
        let contract = &bundle.learning.policy.contract;
        let bad = |e: &dyn std::fmt::Display| DataError::Loop(e.to_string());

        // One control step is one control period (packet M5/V11): the schedule is derived from
        // the scene's own timestep and the Deployment IR's `rate.control`, so a recorded row is
        // 1 / `rate.control` seconds of simulated time and the dataset's `fps` below is true.
        let domains = BatchDomains::single_env_at(
            TickRate::from_period_secs(spec.scene.options.timestep).map_err(|e| bad(&e))?,
            deploy.rate.control,
        )
        .map_err(|e| bad(&e))?;
        let mut env: Env<B> =
            Env::new(&bundle.task, spec.scene, new_backend(), &domains, spec.seed)
                .map_err(|e| bad(&e))?;
        let (nq, nv, nu) = {
            let m = env.model();
            (m.nq as usize, m.nv as usize, m.nu as usize)
        };
        if nu != NJ {
            return Err(DataError::Loop(format!(
                "the deployment declares {NJ} joints and the loaded model has {nu} actuators"
            )));
        }
        let control = deploy.rate.control;
        // `DomainRunner` reads both of the Deployment IR's rates: it replans every
        // `rate.control / rate.inference` control ticks and the chunk drives the ticks in
        // between (packet M5/V17). A rate that does not divide is refused here, by name.
        let mut runner = DomainRunner::<NJ, H>::new(
            env.schedule(),
            contract,
            chunk_blend(&bundle.learning),
            deploy.rate,
        )
        .map_err(|e| bad(&e))?
        // The deployment's action space decides whether a chunk row is a target or an
        // increment the runner integrates before the plane sees it (packet M9/T1, spec 8.5);
        // absent, `JointPosition`, is the runner's own default.
        .with_action_space(deploy.action.space);
        let latency = runner.inference().latency_ticks() as usize;
        let execute = (contract.execute_chunk as usize).clamp(1, H.max(1));
        let mut planes = vec![SafetyPlane::<NJ, H>::from_ir(deploy).map_err(|e| bad(&e))?];

        let max_steps = if spec.max_steps == 0 {
            bundle.task.config.max_episode_steps
        } else {
            spec.max_steps
        };
        let fps = control.num() as f64 / control.den() as f64;
        let (features, warnings) = collect_features(bundle, nq + nv, nu, frame_sink.is_some());
        let task_name = format!(
            "es:task:{}",
            hex(&bundle.task.task_hash().map_err(DataError::Canon)?)
        );

        let mut wrapper = Intervened::<NJ, H> {
            inner: policy,
            intervener,
            action_port: "action".to_owned(),
            model: env.model().clone(),
            episode: 0,
            frame: 0,
            injected: false,
            aborted: false,
            last: vec![[0.0; NJ]],
        };

        let mut writer = LeRobotWriter::create(spec.out_root, Info::new(fps, features))?;
        let mut segments = Vec::new();
        let mut frames = 0u64;
        let mut intervention_frames = 0u64;
        let mut terminations = Vec::with_capacity(spec.n_episodes as usize);
        let mut rendered = 0u64;

        for index in 0..spec.n_episodes {
            if let Some(sink) = sink.as_deref_mut() {
                sink(CollectEvent::EpisodeBegin {
                    episode: index,
                    seed: spec.seed,
                });
            }
            let mut traj = spec.traj_dir.as_ref().map(|_| Trajectory::new(env.model()));
            let mut sources: Vec<i64> = Vec::with_capacity(max_steps as usize);
            let mut commanded: Vec<f64> = Vec::with_capacity(max_steps as usize * NJ);
            let mut human = vec![false; max_steps as usize + latency + execute + 1];
            let mut closed = None;
            let mut aborted = false;
            for frame in 0..max_steps {
                wrapper.episode = index;
                wrapper.frame = frame;
                // Every consumer of the plane observes before every `validate` and none of
                // them decides what that means: `SafetyPlane::observe_state` seeds the
                // command chain on the first call after `begin_episode` and ignores the rest,
                // so collection, evaluation, HIL and the embedded runtime all measure the
                // envelope against the same thing -- the plane's own commands (packet M5/V6,
                // design note section 7.12). Before V6 this was `if frame == 0` here and
                // unconditional in `es_eval::runner`, and the two paths disagreed about what
                // `velocity_max` bounds.
                {
                    let state = env.backend().state();
                    let (mut q, mut qd) = ([0.0; NJ], [0.0; NJ]);
                    q.copy_from_slice(&state.qpos_of(0)[..NJ]);
                    qd.copy_from_slice(&state.qvel_of(0)[..NJ]);
                    planes[0].observe_state(&q, &qd);
                    // Image, `.estraj` pose and `observation.state` row are one instant: the
                    // state this control step is entered with, which is the state the expert
                    // is about to compute its action from and the state the policy will be
                    // handed at inference (packet M5/V12, design note section 7.20).
                    // `Env::step` records the same instant, so trajectory index, frame index
                    // and row index still agree -- and the terminal step no longer renders
                    // the *next* episode's reset state, which is what the post-step read did
                    // once `Env::step` auto-reset a done env.
                    if let Some(t) = traj.as_mut() {
                        t.push(env.model(), &state, 0).map_err(|e| bad(&e))?;
                    }
                    if let Some(sink) = frame_sink.as_deref_mut() {
                        sink(env.model(), &state).map_err(DataError::Loop)?;
                        rendered += 1;
                    }
                }
                let before = counters_of(&planes[0]);
                // Only when someone is watching: the per-kind array is 14 words and the
                // stream-2 sample is the only reader of it (packet M7/E7).
                let kinds_before = sink.as_ref().map(|_| planes[0].counters().violations);
                let tick = env.tick();
                let outcome = env
                    .step_with_policy(&mut runner, &mut wrapper, &mut planes, &mut [])
                    .map_err(|e| bad(&e))?;
                // Provenance for the row `env.step` just recorded: what the plane was asked
                // for, beside what it allowed (spec 13.2).
                commanded.extend_from_slice(&runner.commanded()[..NJ]);
                // An injection decided at tick `t` reaches the actuator at `t + latency` and
                // drives `execute_chunk` ticks (design note section 5.1) -- so mark first,
                // then classify this frame against the marks.
                if wrapper.injected {
                    let start = frame as usize + latency;
                    for f in start..start + execute {
                        if let Some(slot) = human.get_mut(f) {
                            *slot = true;
                        }
                    }
                }
                let after = counters_of(&planes[0]);
                let source = classify(before, after, human[frame as usize]);
                sources.push(source.as_i64());
                if let (Some(sink), Some(kinds_before)) = (sink.as_deref_mut(), kinds_before) {
                    sink(CollectEvent::Tick {
                        episode: index,
                        frame,
                        tick,
                        source,
                        events: raised(&kinds_before, &planes[0].counters().violations).bits(),
                    });
                }
                if let Some(ep) = outcome.episodes.into_iter().next() {
                    closed = Some(ep);
                    break;
                }
                if wrapper.aborted {
                    aborted = true;
                    break;
                }
            }
            // The budget ran out before a terminal condition: close the open episode. Exactly
            // one reset happens per episode either way, so the task's own randomization draws
            // stay a pure function of `seed` and the episode index.
            let mut episode = match closed {
                Some(ep) => ep,
                None => env
                    .reset(None)
                    .map_err(|e| bad(&e))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        DataError::Loop(format!("episode {index} closed no episode on reset"))
                    })?,
            };
            // A scripted driver that gave up is a failed demonstration, and the episode is
            // still written: dropping it would hide what the expert cannot do.
            if aborted {
                episode.termination = Termination::Failure;
            }
            terminations.push(episode.termination);
            if let (Some(dir), Some(t)) = (spec.traj_dir.as_ref(), &traj) {
                t.write(&dir.join(format!("ep-{index:03}.estraj")))
                    .map_err(|e| bad(&e))?;
            }
            runner.reset_env(0);
            // Clears the latch *and* re-arms the seed, so the next episode's first
            // `observe_state` puts the command chain back on the arm (packet M5/V6).
            planes[0].begin_episode();

            let n = episode.steps();
            if sources.len() < n {
                return Err(DataError::Loop(format!(
                    "episode {index}: {n} recorded steps but only {} action sources",
                    sources.len()
                )));
            }
            sources.truncate(n);
            commanded.truncate(n * NJ);
            human.truncate(n);
            human.resize(n, false);
            intervention_frames += human.iter().filter(|h| **h).count() as u64;
            frames += n as u64;
            if let Some(sink) = sink.as_deref_mut() {
                sink(CollectEvent::EpisodeEnd {
                    episode: index,
                    outcome: episode.termination,
                    steps: n,
                });
            }
            segments.extend(segments_of(index, &human, InterventionSource::Scripted));
            writer.write_episode(&to_lerobot(
                index,
                &episode,
                (nq, nv, nu),
                &Recorded {
                    sources: &sources,
                    human: &human,
                    commanded: &commanded,
                },
                fps,
                &task_name,
            ))?;
        }
        writer.finish()?;
        crate::intervention::write_segments(spec.out_root, &segments)?;

        let dataset = LeRobotDataset::open(spec.out_root)?;
        let (content, schema) = identity_of(&dataset)?;
        let h = &bundle.manifest.hashes;
        let slot = |d: Option<[u8; 32]>| d.as_ref().map_or_else(|| "-".to_owned(), hex);
        append_loop_step(
            spec.out_root,
            &LoopStep::new(LoopKind::Collect)
                .input("task", &slot(h.task))
                .input("observation", &slot(h.observation))
                .input("learning", &slot(h.learning))
                .input("deployment", &slot(h.deployment))
                .input("seed", &spec.seed)
                .input("episodes", &spec.n_episodes)
                .output("content", &hex(&content))
                .output("schema", &hex(&schema)),
        )?;

        Ok(CollectReport {
            root: spec.out_root.to_path_buf(),
            episodes: spec.n_episodes,
            frames,
            intervention_frames,
            segments,
            terminations,
            rendered,
            safety: *planes[0].counters(),
            content,
            schema,
            warnings,
        })
    }
}

/// The `EventSet` one step raised, as the delta of the plane's own per-kind counters.
///
/// One env means exactly one `validate` per step, so a kind whose count moved is a kind that
/// step raised — the same bitset `SafeAction::events` carries, read from the side the
/// collector can see (`es_env::DomainRunner::emit_actions` keeps the `SafeAction` itself).
fn raised(before: &[u64; ViolationKind::COUNT], after: &[u64; ViolationKind::COUNT]) -> EventSet {
    let mut set = EventSet::EMPTY;
    for kind in ViolationKind::ALL {
        if after[kind.index()] > before[kind.index()] {
            set.insert(kind);
        }
    }
    set
}

/// `(fallback_activations, clamped_steps)` — the two counters that say what the plane did.
fn counters_of<const NJ: usize, const H: usize>(plane: &SafetyPlane<NJ, H>) -> (u64, u64) {
    let c = plane.counters();
    (c.fallback_activations, c.clamped_steps)
}

/// One frame's `action_source`, from the plane's counter deltas across the step.
///
/// One env per collect run, so exactly one `validate` happens per step and the delta is
/// unambiguous. A clamped human action reports `Clamped`, not `Human`: the frame is still an
/// intervention (the `intervention` column says so) but the actuator did not see what the
/// human asked for, and hiding that would discard the evidence spec 9.3 exists to keep.
fn classify(before: (u64, u64), after: (u64, u64), human: bool) -> ActionSourceCode {
    if after.0 > before.0 {
        ActionSourceCode::Fallback
    } else if after.1 > before.1 {
        ActionSourceCode::Clamped
    } else if human {
        ActionSourceCode::Human
    } else {
        ActionSourceCode::Policy
    }
}

/// The feature set of design note section 3, plus one `video` feature per image channel the
/// task declares — and, with no frame sink, one warning for each, because then nothing at all
/// was written for that channel.
fn collect_features(
    bundle: &PolicyBundle,
    state: usize,
    nu: usize,
    rendering: bool,
) -> (BTreeMap<String, FeatureSpec>, Vec<String>) {
    let mut f = BTreeMap::new();
    f.insert(
        "observation.state".to_owned(),
        FeatureSpec::new(Dtype::Float32, [state as u64]),
    );
    f.insert(
        "action".to_owned(),
        FeatureSpec::new(Dtype::Float32, [nu as u64]),
    );
    f.insert(
        ACTION_COMMANDED.to_owned(),
        FeatureSpec::new(Dtype::Float32, [nu as u64]),
    );
    f.insert("reward".to_owned(), FeatureSpec::new(Dtype::Float64, [1]));
    f.insert(INTERVENTION.to_owned(), FeatureSpec::new(Dtype::Int64, [1]));
    f.insert(
        ACTION_SOURCE.to_owned(),
        FeatureSpec::new(Dtype::Int64, [1]),
    );

    let mut warnings = Vec::new();
    for (name, channel) in &bundle.task.observation_spec.channels {
        if channel.ty.image.is_none() {
            continue;
        }
        f.insert(
            format!("observation.images.{name}"),
            FeatureSpec::new(Dtype::Video, channel.ty.shape.dims().to_vec()),
        );
        if !rendering {
            warnings.push(format!(
                "camera {name:?}: info.json declares a video feature and read_episode will \
                 produce VideoRef placeholders, but no mp4 was written (no renderer in this \
                 build)"
            ));
        }
    }
    (f, warnings)
}

/// The per-frame columns [`Collector::run`] accumulates itself, beside the ones the episode
/// recorder already holds.
struct Recorded<'a> {
    sources: &'a [i64],
    human: &'a [bool],
    /// The pre-plane command of each frame, `n * nu` (spec 13.2).
    commanded: &'a [f64],
}

/// One `es_env::Episode` as `LeRobot` columns (design note section 3).
fn to_lerobot(
    index: u32,
    ep: &es_env::Episode,
    shape: (usize, usize, usize),
    frames: &Recorded<'_>,
    fps: f64,
    task: &str,
) -> Episode {
    let (sources, human, commanded) = (frames.sources, frames.human, frames.commanded);
    let (nq, nv, nu) = shape;
    let n = ep.steps();
    let mut state = Vec::with_capacity(n * (nq + nv));
    for i in 0..n {
        state.extend(ep.qpos[i * nq..(i + 1) * nq].iter().map(|v| *v as f32));
        state.extend(ep.qvel[i * nv..(i + 1) * nv].iter().map(|v| *v as f32));
    }
    let columns = BTreeMap::from([
        ("observation.state".to_owned(), Column::F32(state)),
        (
            // `ep.ctrl` is the `SafeAction` the plane returned for that step, copied into
            // `ctrl` by `DomainRunner::emit_actions` and into the episode by `Env::step`:
            // what is executed is what is recorded (spec 13.2, `INV-12`).
            "action".to_owned(),
            Column::F32(ep.ctrl[..n * nu].iter().map(|v| *v as f32).collect()),
        ),
        (
            ACTION_COMMANDED.to_owned(),
            Column::F32(commanded[..n * nu].iter().map(|v| *v as f32).collect()),
        ),
        ("reward".to_owned(), Column::F64(ep.reward[..n].to_vec())),
        (
            INTERVENTION.to_owned(),
            Column::I64(human.iter().map(|h| i64::from(*h)).collect()),
        ),
        (ACTION_SOURCE.to_owned(), Column::I64(sources.to_vec())),
    ]);
    Episode {
        index,
        tasks: vec![task.to_owned()],
        timestamps: (0..n).map(|i| i as f64 / fps).collect(),
        task_index: vec![0; n],
        columns,
        // Regenerated by the reader from `video_path`; no mp4 is written here.
        video: BTreeMap::new(),
    }
}

// --- distillation ---------------------------------------------------------------------------

/// The spec 19.2 split, as the two things that decide it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitSpec {
    /// `[train, val, test]`. `test` takes the remainder, so the three are always an exact
    /// partition.
    pub ratios: [f64; 3],
    pub seed: u64,
}

impl Default for SplitSpec {
    fn default() -> Self {
        Self {
            ratios: [0.8, 0.1, 0.1],
            seed: 0,
        }
    }
}

/// Merges datasets, splits deterministically, and writes the spec 19.3 identity.
///
/// Episodes are re-indexed `0..N` in input order then original index — a total order, so two
/// runs of the same inputs produce byte-identical parquet and the same `TrainingIdentity`.
/// `meta/interventions.jsonl` of each input is merged with its episode indices remapped, so
/// labels survive the merge.
///
/// The `Distill` loop step is appended to the output root **and** every input root: a
/// dataset's own ledger should record that it was consumed (spec 13.3).
pub fn distill<P: AsRef<Path>>(
    inputs: &[P],
    split: &SplitSpec,
    out_root: &Path,
) -> Result<TrainingIdentity, DataError> {
    if inputs.is_empty() {
        return Err(DataError::Loop(
            "distill needs at least one input dataset".to_owned(),
        ));
    }
    if split.ratios.iter().sum::<f64>() > 1.0 + 1e-9 {
        return Err(DataError::Loop(format!(
            "split ratios {:?} sum above 1.0; that is silent truncation of the test set, \
             not a clamp",
            split.ratios
        )));
    }
    let datasets: Vec<LeRobotDataset> = inputs
        .iter()
        .map(|p| LeRobotDataset::open(p.as_ref()))
        .collect::<Result<_, _>>()?;
    let first = datasets[0].info();
    for (i, ds) in datasets.iter().enumerate().skip(1) {
        if ds.info().features != first.features {
            return Err(DataError::Inconsistent(format!(
                "input {i} ({}) has a different feature set than input 0; merging them would \
                 produce a schema hash that describes neither",
                ds.root().display()
            )));
        }
        if ds.info().fps.to_bits() != first.fps.to_bits() {
            return Err(DataError::Inconsistent(format!(
                "input {i} ({}) runs at {} fps, input 0 at {}",
                ds.root().display(),
                ds.info().fps,
                first.fps
            )));
        }
    }

    let mut info = first.clone();
    crate::intervention::ensure_columns(&mut info);
    let mut writer = LeRobotWriter::create(out_root, info)?;
    let mut segments = Vec::new();
    let mut next = 0u32;
    for ds in &datasets {
        let mut remap = BTreeMap::new();
        for meta in ds.episodes() {
            let mut ep = ds.read_episode(meta.episode_index)?;
            remap.insert(meta.episode_index, next);
            ep.index = next;
            next += 1;
            writer.write_episode(&ep)?;
        }
        for mut s in crate::intervention::read_segments(ds.root())? {
            if let Some(index) = remap.get(&s.episode) {
                s.episode = *index;
                segments.push(s);
            }
        }
    }
    writer.finish()?;
    crate::intervention::write_segments(out_root, &segments)?;

    let merged = LeRobotDataset::open(out_root)?;
    let lists = Split::deterministic(next, split.ratios, split.seed);
    let identity = DatasetIdentity::compute(&merged, &lists)?;
    let training = TrainingIdentity {
        // Everything the training run owns is unknown on this side of spec 2.3's boundary. A
        // zero digest reads as "unset"; a fabricated one would make `training_hash` a lie.
        config: [0; 32],
        optimizer: [0; 32],
        scheduler: [0; 32],
        seed: [0; 32],
        dataset: identity.to_dataset_hash(),
        base_model: BaseModel::default(),
        augmentation: [0; 32],
        precision: [0; 32],
        topology: [0; 32],
        checkpoint_manifest: [0; 32],
        metrics: [0; 32],
        hardware: [0; 32],
    };
    write_json(&out_root.join(TRAINING_IDENTITY_FILE), &training)?;
    // The split lists themselves, so the `dataset_split_hash` above is reproducible by hand
    // and not just by re-running this function (spec 19.2).
    write_json(&out_root.join("split.json"), &lists)?;

    let step = LoopStep::new(LoopKind::Distill)
        .input("seed", &split.seed)
        .input("ratios", &format!("{:?}", split.ratios))
        .output("content", &hex(&identity.content))
        .output("schema", &hex(&identity.schema))
        .output("split", &hex(&identity.split))
        .output("training_hash", &hex(&training.training_hash()?));
    let mut step = step;
    for (i, ds) in datasets.iter().enumerate() {
        let (content, schema) = identity_of(ds)?;
        step = step
            .input(&format!("input.{i}.content"), &hex(&content))
            .input(&format!("input.{i}.schema"), &hex(&schema));
    }
    append_loop_step(out_root, &step)?;
    for ds in &datasets {
        if ds.root() != out_root {
            append_loop_step(ds.root(), &step)?;
        }
    }
    Ok(training)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), DataError> {
    let mut text = serde_json::to_string_pretty(value).map_err(|source| DataError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    text.push('\n');
    write_file(path, text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clamp_is_not_an_intervention_and_an_intervention_is_not_a_failure() {
        // Fallback wins over everything: the actuator saw the watchdog's action.
        assert_eq!(classify((0, 0), (1, 1), true), ActionSourceCode::Fallback);
        // A human action the envelope changed is `Clamped`, not `Human` (design note 2.2).
        assert_eq!(classify((0, 0), (0, 1), true), ActionSourceCode::Clamped);
        assert_eq!(classify((0, 0), (0, 0), true), ActionSourceCode::Human);
        assert_eq!(classify((0, 0), (0, 0), false), ActionSourceCode::Policy);
        // A clamp with nobody intervening is still just a clamp (spec 18.5).
        assert_eq!(classify((3, 7), (3, 8), false), ActionSourceCode::Clamped);
    }

    #[test]
    fn the_ledger_is_append_only() {
        let dir = std::env::temp_dir().join(format!("es-data-loop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        append_loop_step(
            &dir,
            &LoopStep::new(LoopKind::Collect).output("content", &"aa"),
        )
        .unwrap();
        append_loop_step(
            &dir,
            &LoopStep::new(LoopKind::Intervene).input("content", &"aa"),
        )
        .unwrap();
        let steps = read_loop_steps(&dir).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].outputs["content"], steps[1].inputs["content"]);
    }

    /// Packet M7/T2 oracle 2: `collect -> train -> evaluate` chains, and each half fails by
    /// name when it does not (spec 13.3).
    #[test]
    fn loop_train_and_evaluate_steps_chain() {
        let collect = LoopStep::new(LoopKind::Collect)
            .output("content", &"aa")
            .output("schema", &"bb");
        let train = LoopStep::new(LoopKind::Train)
            .input("content", &"aa")
            .input("identity_hash", &"11")
            .output("training_hash", &"22")
            .output("checkpoint.20000", &"cc");
        let evaluate = LoopStep::new(LoopKind::Evaluate)
            .input("policy_hash", &"cc")
            .input("evaluation_hash", &"ee")
            .output("passed", &true);
        let chain = [collect.clone(), train.clone(), evaluate.clone()];
        check_chain(&chain).expect("collect -> train -> evaluate chains");
        assert_eq!(last_evaluation_hash(&chain), Some("ee"));

        // A train step that read another directory's dataset.
        let moved = LoopStep::new(LoopKind::Train).input("content", &"ff");
        let e = check_chain(&[collect.clone(), moved]).expect_err("refused");
        assert!(
            e.to_string().contains("ff") && e.to_string().contains("aa"),
            "{e}"
        );

        // An evaluate step judging a policy no train step in this ledger wrote.
        let stranger = LoopStep::new(LoopKind::Evaluate).input("policy_hash", &"dd");
        let e = check_chain(&[collect.clone(), train.clone(), stranger]).expect_err("refused");
        assert!(
            e.to_string().contains("dd") && e.to_string().contains("20000=cc"),
            "{e}"
        );

        // The expert gate judges the demonstrator, not a checkpoint, and chains to nothing.
        let gate = LoopStep::new(LoopKind::Evaluate)
            .input("expert", &"so101-pick-place")
            .input("policy_hash", &"00")
            .input("evaluation_hash", &"ee");
        check_chain(&[collect, gate, train, evaluate]).expect("the gate is skipped by name");
    }
}
