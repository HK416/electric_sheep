//! The cell loop, the hash chain, and the two artifacts of §10.5.
//!
//! One `Env`, one `SafetyPlane` and one `CpuPlan` per cell — see
//! `docs/design/evaluation-execution.md` section 2 for why the `Env` is rebuilt per cell
//! (fairness, §10.4) rather than shared.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use es_assets::scene::SceneDesc;
use es_compile::{CpuPlan, Home, PlanMode, Tensor, TensorRef};
use es_core::{PhysTick, StableId};
use es_env::scheduler::BatchDomains;
use es_env::{plane_chunk, ChunkBuffer, Env, EnvMetrics, Episode, PlaneFeed};
use es_ir::deployment::{DeploymentIr, ExecutionMode, Micros};
use es_ir::evaluation::{
    AcceptanceResult, CellResult, EvaluationIr, EvaluationReport, MetricSpec, MetricValue, SeedPlan,
};
use es_ir::hash::{canonical_hash, DatasetHash, HardwareCapability, HashChain};
use es_ir::observation::{ObservationIr, ObservationNode};
use es_ir::task::TaskIr;
use es_ir::types::ElemType;
use es_physics_core::backend::{ModelInfo, PhysicsBackend, StateView};
use es_policy::PolicyRuntime;
use es_safety::{ActionChunk, ActionSource, SafetyPlane};
use serde::{Deserialize, Serialize};

use crate::metrics;
use crate::perturb::{LightOverride, PerturbationPlan, ResetOverrides, StepState};
use crate::{hex32, EvalError};

/// Schema version of `report.json` and `evaluation.lock`.
pub const SCHEMA_VERSION: u32 = 1;

/// Knobs that are not part of the evaluation document: what the caller knows and the IR
/// cannot (the dataset it trained on, the machine it ran on, when).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunConfig {
    /// Episode step budget. `None` uses the task's `max_episode_steps`.
    pub max_steps: Option<u32>,
    /// Which policy output is the action chunk. `None` requires the policy to have exactly
    /// one output.
    pub action_output: Option<String>,
    /// Unix seconds written to `evaluation.lock`. The only non-reproducible value in either
    /// artifact, and deliberately absent from `report.json` (§10.4).
    pub created: u64,
    pub dataset: DatasetHash,
    pub hardware: HardwareCapability,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_steps: None,
            action_output: None,
            created: 0,
            dataset: DatasetHash {
                content: [0; 32],
                schema: [0; 32],
                split: [0; 32],
            },
            hardware: HardwareCapability([0; 32]),
        }
    }
}

/// Where an image observation input's pixels come from (§7.2, §10.1).
///
/// Given this episode's lighting, the loaded model and the current state, one camera frame in
/// exactly the dtype, count and layout the plan declared, or the reason there is none. A
/// closure rather than the renderer itself: `es-eval` is layer 10 and `es-render` layer 5, and
/// linking Vulkan here just to *refuse* an image would be the wrong trade — the caller owns
/// `es_env::render::EnvRenderer` (feature `render`) and hands its `frame` in through this.
///
/// The [`LightOverride`] is this cell and episode's draw (§10.2). It is constant for a whole
/// episode, so a caller rebuilds its renderer only when it changes.
///
/// Nothing on this path resamples, converts or reorders: a frame that is not what the plan
/// declared is an error, and every conversion is an Observation IR node (§7.2, `INV-14`).
pub type FrameSource<'a> =
    dyn FnMut(&LightOverride, &ModelInfo, &StateView<'_>) -> Result<Vec<u8>, String> + 'a;

/// Where the emitted action came from, as `es video mosaic` spells it.
///
/// The same four outcomes `es-data` writes into a dataset's `action_source` column
/// (`es_data::ActionSourceCode`, `crates/es-data/src/collect.rs:552`), duplicated rather than
/// shared because `es-data` is layer 10 like this crate and §4.2 forbids a same-layer
/// dependency. `Human` cannot occur here — an evaluation has no teleop — but it is one of the
/// four the overlay reads, so the variant stays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventSource {
    Policy,
    Clamped,
    Fallback,
    Human,
}

impl From<ActionSource> for EventSource {
    /// Read off the plane's own per-step output rather than re-derived from counter deltas:
    /// `SafeAction` already says which of the four this step was (§9.3, §9.4).
    fn from(s: ActionSource) -> Self {
        match s {
            ActionSource::Policy => Self::Policy,
            ActionSource::Clamped => Self::Clamped,
            ActionSource::Fallback(_) => Self::Fallback,
        }
    }
}

/// One record per rendered frame, written as `events.json` beside `frames/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepEvent {
    /// Index of the frame this describes inside its cell, dense and ascending from 0.
    pub frame: u64,
    pub tick: PhysTick,
    pub source: EventSource,
    /// `es_safety::EventSet::bits()` for this step: the `ViolationKind` bitset, `0` when the
    /// step was clean.
    pub events: u32,
}

/// Where a run puts its frames and events; `None` is today's behaviour exactly.
///
/// One subdirectory per cell — a cell being one episode of one suite, because
/// `BatchDomains::single_env()` makes every episode its own run — holding `<NNNNNN>.bin` plus
/// one `layout.json`, which is the directory shape `es video mosaic` tiles.
#[derive(Clone, Debug, Default)]
pub struct FrameSink {
    pub dir: PathBuf,
    /// Cell name -> its frame-ordered records, in the shape `events.json` is written in.
    pub events: BTreeMap<String, Vec<StepEvent>>,
}

impl FrameSink {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            events: BTreeMap::new(),
        }
    }

    /// Writes `events.json`: `{ "<cell>": [ StepEvent, ... ] }`.
    pub fn write_events(&self, path: &Path) -> Result<(), EvalError> {
        let mut text = serde_json::to_string_pretty(&self.events)
            .map_err(|e| EvalError::Plan(e.to_string()))?;
        text.push('\n');
        fs::write(path, text).map_err(|source| EvalError::Io {
            path: path.display().to_string(),
            source,
        })
    }
}

/// One cell's frame directory: the raw `.bin` sequence plus the `layout.json` that pins their
/// shape, written as the frames are captured.
struct CellFrames {
    dir: PathBuf,
    n: u64,
}

impl CellFrames {
    /// Appends one frame and returns its index. The first one writes `layout.json`, so a cell
    /// that rendered nothing leaves no half-described directory behind.
    fn write(&mut self, dtype: ElemType, shape: &[u64], data: &[u8]) -> Result<u64, EvalError> {
        let io = |path: &Path| {
            let p = path.display().to_string();
            move |source| EvalError::Io {
                path: p.clone(),
                source,
            }
        };
        if self.n == 0 {
            fs::create_dir_all(&self.dir).map_err(io(&self.dir))?;
            let layout = self.dir.join("layout.json");
            let text = format!(
                "{{\"dtype\":\"{}\",\"shape\":{:?}}}\n",
                dtype_name(dtype),
                shape
            );
            fs::write(&layout, text).map_err(io(&layout))?;
        }
        let path = self.dir.join(format!("{:06}.bin", self.n));
        fs::write(&path, data).map_err(io(&path))?;
        self.n += 1;
        Ok(self.n - 1)
    }
}

/// The `layout.json` spelling of an element type — the one `es video mosaic` and the render
/// goldens read.
fn dtype_name(e: ElemType) -> &'static str {
    match e {
        ElemType::U8 => "u8",
        ElemType::Bool => "bool",
        ElemType::F16 => "f16",
        ElemType::Bf16 => "bf16",
        ElemType::F32 => "f32",
        ElemType::F64 => "f64",
        ElemType::I32 => "i32",
    }
}

/// The backend half of `evaluation.lock`: what a re-run would have to match (§17.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCaps {
    pub name: String,
    pub determinism: String,
    pub float: String,
    pub max_envs: u32,
    pub gpu_resident: bool,
    pub supports_reset_subset: bool,
    pub supports_state_get_set: bool,
    pub quirks: Vec<String>,
}

/// `evaluation.lock` (§10.5): the conditions, in hex, next to the seeds that produced them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationLock {
    pub schema_version: u32,
    pub evaluation_hash: String,
    pub execution_hash: String,
    pub seeds: Vec<u64>,
    pub backend: BackendCaps,
    pub created: u64,
}

/// One cell's contribution to the §10.1 table, exactly as `record_cell` produced it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardCell {
    /// Index into `EvaluationIr::suites`. This is what puts a merged report back into the
    /// order the sequential run would have written it, whichever worker produced the cell and
    /// whichever finished first (§10.4).
    pub cell: u32,
    /// One entry per declared metric, in `MetricSpec::ALL` order.
    pub results: Vec<CellResult>,
    /// The per-episode samples of the metrics that have one ([`metrics::per_episode`]), which
    /// is what an `Aggregation` other than `Mean` needs and `CellResult` does not carry.
    pub samples: Vec<(MetricSpec, Vec<f64>)>,
}

/// What one worker of a `--jobs N` run hands back (design note `docs/design/visible-learning.md`
/// section 7.11).
///
/// A shard is a *partition of the cells*, never of the episodes: one cell owns one `Env`, one
/// `SafetyPlane` and one monotonic `seq` for all of its episodes, and `Env::reset` keys the
/// task's own randomization by an episode counter that cannot be seeked (§10.4). Splitting
/// finer would change the run; splitting here does not.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Shard {
    pub cells: Vec<ShardCell>,
    /// The backend this worker actually opened; `None` when the shard owned no cell.
    pub backend: Option<BackendCaps>,
    /// This worker's slice of `events.json`, keyed by cell name exactly as [`FrameSink`] keys
    /// it. Cell names are globally unique, so the workers' maps are disjoint.
    pub events: BTreeMap<String, Vec<StepEvent>>,
}

/// Namespace for [`Evaluation::run`]. Not a trait and not state: INV-17 allows seven
/// extension points and this is none of them.
#[derive(Debug)]
pub struct Evaluation;

impl Evaluation {
    /// Runs every suite x episode of `ir` and produces the two §10.5 artifacts.
    ///
    /// `new_backend` rather than one backend: `Env::new` consumes its backend and each cell
    /// needs a fresh `Env` so that every cell's episode counter — which keys the task's own
    /// `RandomizationPlan` draws — starts at 0 and the table rows stay comparable (§10.4).
    ///
    /// `NJ` / `H` are the deployment's joint count and chunk horizon; `SafetyPlane::from_ir`
    /// rejects a mismatch, so they cannot silently disagree with `deploy`.
    #[allow(clippy::too_many_arguments)]
    pub fn run<B, F, const NJ: usize, const H: usize>(
        ir: &EvaluationIr,
        task: &TaskIr,
        scene: &SceneDesc,
        obs: &ObservationIr,
        policy: &mut dyn PolicyRuntime,
        deploy: &DeploymentIr,
        new_backend: F,
        cfg: &RunConfig,
    ) -> Result<(EvaluationReport, EvaluationLock), EvalError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
        Self::run_with_frames::<B, F, NJ, H>(
            ir,
            task,
            scene,
            obs,
            policy,
            deploy,
            new_backend,
            cfg,
            None,
            None,
        )
    }

    /// [`Self::run`] with a source for image observation inputs (§7.2) and, optionally, a
    /// place to put the frames and the per-step records that describe them.
    ///
    /// With `frames: None` — what [`Self::run`] passes — an image input is refused exactly as
    /// before: this narrows the §10.1 refusal, it does not remove it, and nothing is ever
    /// zero-filled. With `sink: None` the frames are rendered and consumed, exactly as V0b
    /// left it; with one, every captured frame also lands on disk and gains a [`StepEvent`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_frames<B, F, const NJ: usize, const H: usize>(
        ir: &EvaluationIr,
        task: &TaskIr,
        scene: &SceneDesc,
        obs: &ObservationIr,
        policy: &mut dyn PolicyRuntime,
        deploy: &DeploymentIr,
        new_backend: F,
        cfg: &RunConfig,
        frames: Option<&mut FrameSource<'_>>,
        sink: Option<&mut FrameSink>,
    ) -> Result<(EvaluationReport, EvaluationLock), EvalError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
        // The sequential path *is* the sharded path with one shard: byte-identity between
        // `--jobs 1` and this is by construction, not by a second implementation (§3.5).
        let dir = sink.as_deref().map(|s| s.dir.clone());
        let mut shard = Self::run_shard::<B, F, NJ, H>(
            ir,
            task,
            scene,
            obs,
            policy,
            deploy,
            new_backend,
            cfg,
            frames,
            dir.as_deref(),
            (0, 1),
        )?;
        if let Some(s) = sink {
            s.events = std::mem::take(&mut shard.events);
        }
        Self::merge(ir, task, obs, deploy, policy, cfg, &[shard])
    }

    /// The cells this shard owns, run and left unjudged for [`Self::merge`].
    ///
    /// `shard` is `(index, count)` and the partition is round-robin on the cell index: cell
    /// `c` belongs to shard `c % count`. Fixed by the cell index alone, so it does not depend
    /// on how long a suite takes, on how many workers actually started, or on any iteration
    /// order (§3.4). `(0, 1)` is every cell.
    ///
    /// `frames_dir` is [`FrameSink::dir`]: every cell writes into its own `<dir>/<cell>/`, and
    /// cell names are globally unique, so two shards writing into one directory cannot collide
    /// and no frame merge step exists.
    #[allow(clippy::too_many_arguments)]
    pub fn run_shard<B, F, const NJ: usize, const H: usize>(
        ir: &EvaluationIr,
        task: &TaskIr,
        scene: &SceneDesc,
        obs: &ObservationIr,
        policy: &mut dyn PolicyRuntime,
        deploy: &DeploymentIr,
        mut new_backend: F,
        cfg: &RunConfig,
        mut frames: Option<&mut FrameSource<'_>>,
        frames_dir: Option<&Path>,
        shard: (u32, u32),
    ) -> Result<Shard, EvalError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
        let (index, count) = shard;
        if count == 0 || index >= count {
            return Err(EvalError::Shard(format!(
                "{index}/{count} is not a partition: the count must be at least 1 and the \
                 index below it"
            )));
        }
        if let Some(d) = ir.validate().first() {
            return Err(EvalError::InvalidIr(format!("{d}")));
        }
        refuse_augmentation(ir, obs)?;

        let mut plan = CpuPlan::compile(obs, PlanMode::Release)
            .map_err(|d| EvalError::Plan(d.iter().map(ToString::to_string).collect()))?;
        let seeds = resolve_seeds(ir);
        let max_steps = cfg.max_steps.unwrap_or(task.config.max_episode_steps);
        let domains = BatchDomains::single_env();
        // One control step is `inference.period` simulation ticks (§12.1); the deployment
        // states what that step is worth in wall time.
        let control_us = deploy.rate.control_period().0;
        let ms_to_steps = |ms: u32| (u64::from(ms) * 1000 / control_us.max(1)) as usize;

        // The Deployment IR decides, never a flag: `execution` carries both the mode the plane
        // reads and the blend the chunk buffer applies (spec 8.5, spec 9.2).
        let blend = blend_of(deploy.execution);
        let mut perturbations: Option<PerturbationPlan> = None;
        let mut sources: Option<BTreeMap<String, Capture>> = None;
        let mut out = Shard::default();

        for (cell, suite) in ir.suites.iter().enumerate() {
            if cell as u32 % count != index {
                continue;
            }
            let mut env: Env<B> = Env::new(task, scene, new_backend(), &domains, seeds[0])?;
            if perturbations.is_none() {
                perturbations = Some(PerturbationPlan::compile(
                    ir,
                    scene,
                    env.model(),
                    frames.is_some(),
                )?);
                sources = Some(input_sources(&plan, obs, task, Some(env.model()))?);
                // Every cell opens the same backend on the same scene, so which cell this
                // shard happened to reach first does not change the answer.
                out.backend = Some(backend_caps(&env));
            }
            let perturbations = perturbations.as_ref().expect("just compiled");
            let mut safety = SafetyPlane::<NJ, H>::from_ir(deploy)
                .map_err(|e| EvalError::Safety(e.to_string()))?;

            let mut episodes = Vec::with_capacity(seeds.len());
            // The chunk buffer and the plane's feed live exactly as long as the plane does,
            // and for the same reason: the `seq` the plane judges freshness by is monotonic
            // per cell (spec 8.6). `PlaneFeed::end_episode` clears both between episodes.
            //
            // This is `es loop collect`'s own path -- `es_env::plane_chunk`, the one place a
            // buffered chunk becomes an actuator command. Before V6b this loop handed the
            // plane each raw inference result under a fresh `seq`, so only row 0 of every
            // chunk executed and the Deployment IR's `action.execute_chunk` and `execution`
            // were dead here (packet M5/V6b, design note section 7.13).
            let mut buffer = ChunkBuffer::<NJ, H>::new(deploy.action.execute_chunk, blend);
            let mut feed = PlaneFeed::default();
            for (idx, seed) in seeds.iter().enumerate() {
                // One cell of the mosaic is one episode of one suite: `single_env()` makes
                // them independent runs, so the grid is `suites x episodes` directories.
                let name = format!("{}-{idx:02}", suite.name);
                let mut cell_frames = frames_dir.map(|d| CellFrames {
                    dir: d.join(&name),
                    n: 0,
                });
                let mut events = Vec::new();
                let episode = run_episode::<B, NJ, H>(
                    &mut env,
                    &mut plan,
                    sources.as_ref().expect("just resolved"),
                    policy,
                    &mut safety,
                    perturbations,
                    cell,
                    *seed,
                    idx as u64,
                    max_steps,
                    control_us,
                    &ms_to_steps,
                    cfg.action_output.as_deref(),
                    deploy.execution,
                    &mut buffer,
                    &mut feed,
                    frames.as_deref_mut(),
                    cell_frames.as_mut(),
                    &mut events,
                )?;
                if frames_dir.is_some() {
                    out.events.insert(name, events);
                }
                episodes.push(episode);
            }

            let env_metrics = env.metrics();
            out.cells.push(record_cell(
                ir,
                cell as u32,
                &suite.name,
                &episodes,
                safety.counters(),
                &env_metrics,
            ));
        }
        Ok(out)
    }

    /// The §10.5 artifacts, from every worker's cells put back in canonical order.
    ///
    /// The judgement, the hash chain and both artifacts are computed **here and only here**,
    /// from the same `judge` over the same `measured`/`samples` maps the sequential path
    /// builds — so `--jobs N` cannot produce a report that `--jobs 1` would not have (§10.4).
    ///
    /// A cell set that is not exactly `0..suites.len()`, each cell once, is refused. A report
    /// missing a suite because a worker was lost would otherwise carry a correct
    /// `evaluation_hash` over numbers nobody measured.
    pub fn merge(
        ir: &EvaluationIr,
        task: &TaskIr,
        obs: &ObservationIr,
        deploy: &DeploymentIr,
        policy: &dyn PolicyRuntime,
        cfg: &RunConfig,
        shards: &[Shard],
    ) -> Result<(EvaluationReport, EvaluationLock), EvalError> {
        let plan = CpuPlan::compile(obs, PlanMode::Release)
            .map_err(|d| EvalError::Plan(d.iter().map(ToString::to_string).collect()))?;

        // Stable, so the metric order inside a cell is untouched and only the cells move.
        let mut merged: Vec<&ShardCell> = shards.iter().flat_map(|s| &s.cells).collect();
        merged.sort_by_key(|c| c.cell);
        let covered: Vec<u32> = merged.iter().map(|c| c.cell).collect();
        if covered.iter().copied().ne(0..ir.suites.len() as u32) {
            return Err(EvalError::Shard(format!(
                "the merged workers cover cells {covered:?}; the evaluation has {} suites and \
                 every cell must appear exactly once",
                ir.suites.len()
            )));
        }

        let mut cells: Vec<CellResult> = Vec::new();
        let mut measured: BTreeMap<(String, MetricSpec), MetricValue> = BTreeMap::new();
        let mut samples: BTreeMap<(String, MetricSpec), Vec<f64>> = BTreeMap::new();
        for c in merged {
            let suite = &ir.suites[c.cell as usize].name;
            for r in &c.results {
                measured.insert((suite.clone(), r.metric), r.value.clone());
                cells.push(r.clone());
            }
            for (metric, v) in &c.samples {
                samples.insert((suite.clone(), *metric), v.clone());
            }
        }

        let acceptance = judge(ir, &measured, &samples);
        let passed = acceptance
            .iter()
            .all(|a| matches!(a, AcceptanceResult::Determined { passed: true, .. }));

        let evaluation_hash = ir
            .evaluation_hash()
            .map_err(|d| EvalError::InvalidIr(format!("{d}")))?;
        let chain = hash_chain(task, obs, deploy, evaluation_hash, &plan, policy, cfg)?;
        let execution_hash = chain.execution_hash();

        let report = EvaluationReport {
            schema_version: SCHEMA_VERSION,
            evaluation_hash,
            execution_hash,
            cells,
            acceptance,
            passed,
            // `report.html` and `episodes/` replay are a later packet; an empty list is
            // honest, a list of paths to files nobody wrote is not.
            episodes: Vec::new(),
        };
        let lock = EvaluationLock {
            schema_version: SCHEMA_VERSION,
            evaluation_hash: hex32(&evaluation_hash),
            execution_hash: hex32(&execution_hash),
            seeds: resolve_seeds(ir),
            // Shard 0 owns cell 0, and the workers are collected in shard order, so this is
            // the same backend the sequential run would have reported.
            backend: shards
                .iter()
                .find_map(|s| s.backend.clone())
                .unwrap_or_else(|| BackendCaps {
                    name: "none".to_owned(),
                    determinism: "unknown".to_owned(),
                    float: "unknown".to_owned(),
                    max_envs: 0,
                    gpu_resident: false,
                    supports_reset_subset: false,
                    supports_state_get_set: false,
                    quirks: Vec::new(),
                }),
            created: cfg.created,
        };
        Ok((report, lock))
    }
}

/// INV-15. A `training_only` `Augment` node is *auto-disabled* (§10.4): `CpuPlan::compile`
/// lowers it to an identity pass-through, so it cannot run here and does not need the
/// allow-list. Any other `Augment` node outside the allow-list refuses the run; the graph is
/// never rewritten, because stripping a node would make `observation_hash` describe a graph
/// the caller never declared.
fn refuse_augmentation(ir: &EvaluationIr, obs: &ObservationIr) -> Result<(), EvalError> {
    use es_ir::evaluation::AugmentationPolicy;
    for (id, node) in &obs.graph.nodes {
        if !matches!(
            node,
            ObservationNode::Augment {
                training_only: false,
                ..
            }
        ) {
            continue;
        }
        let key = id.0.to_string();
        let allowed = match &ir.augmentation {
            AugmentationPolicy::Disabled => false,
            AugmentationPolicy::AllowList { nodes, .. } => nodes.contains(&key),
        };
        if !allowed {
            return Err(EvalError::AugmentationEnabled { node: key });
        }
    }
    Ok(())
}

/// §10.2 `seed_base` / explicit seeds, resolved to one seed per episode.
fn resolve_seeds(ir: &EvaluationIr) -> Vec<u64> {
    match &ir.episodes.seeds {
        SeedPlan::Base(base) => (0..u64::from(ir.episodes.n_episodes))
            .map(|i| base.wrapping_add(i))
            .collect(),
        SeedPlan::Explicit(list) => list.clone(),
    }
}

fn backend_caps<B: PhysicsBackend>(env: &Env<B>) -> BackendCaps {
    let c = env.backend().capabilities();
    BackendCaps {
        name: c.name.clone(),
        determinism: format!("{:?}", c.determinism),
        float: format!("{:?}", c.float),
        max_envs: c.batch.max_envs,
        gpu_resident: c.batch.gpu_resident,
        supports_reset_subset: c.supports_reset_subset,
        supports_state_get_set: c.supports_state_get_set,
        quirks: c.quirks.iter().map(|q| q.description.clone()).collect(),
    }
}

/// One episode: reset, then `observe -> plan -> infer -> validate -> step` until done.
#[allow(clippy::too_many_arguments)]
fn run_episode<B: PhysicsBackend, const NJ: usize, const H: usize>(
    env: &mut Env<B>,
    plan: &mut CpuPlan,
    sources: &BTreeMap<String, Capture>,
    policy: &mut dyn PolicyRuntime,
    safety: &mut SafetyPlane<NJ, H>,
    perturbations: &PerturbationPlan,
    cell: usize,
    seed: u64,
    episode: u64,
    max_steps: u32,
    control_us: u64,
    ms_to_steps: &dyn Fn(u32) -> usize,
    action_output: Option<&str>,
    mode: ExecutionMode,
    buffer: &mut ChunkBuffer<NJ, H>,
    feed: &mut PlaneFeed,
    mut frames: Option<&mut FrameSource<'_>>,
    mut cell_frames: Option<&mut CellFrames>,
    events: &mut Vec<StepEvent>,
) -> Result<Episode, EvalError> {
    let (nu, nq, nv) = {
        let m = env.model();
        (m.nu as usize, m.nq as usize, m.nv as usize)
    };
    // The deployment's NJ *is* the actuator count; a model that disagrees would otherwise be
    // driven by a broadcast copy of joint NJ-1 and observed through zero-padded state.
    if nu != NJ || nq < NJ || nv < NJ {
        return Err(EvalError::JointMismatch { nu, nq, nv, nj: NJ });
    }

    // **No `env.reset` here** (packet M5/V6b). `Env::new` already reset once, and every
    // episode below ends with a reset — `Env::step`'s own on a terminal condition, or the
    // explicit one at the bottom when the step budget runs out — so the env is always freshly
    // drawn when this function is entered. Resetting again made evaluation's episode `i` see
    // randomization draw `2i + 1` while `es loop collect`'s episode `i` sees draw `i`, so
    // `--seed S` named a different scene on the two paths (§10.4, §6.3).
    //
    // An episode is where an observation stream ends (spec 7.5 layer 1): without this, the
    // first frames of this episode would see the tail of the previous one, and cell 2 would
    // see cell 1 — making the §10.1 table depend on suite order (§10.4).
    plan.reset();
    // The same boundary for the chunk buffer: a chunk predicted for the previous episode has
    // no meaning in this one, and the plane must drop the one it still holds (spec 13.1).
    // `DomainRunner::reset_env` is the collector's call to the identical function.
    feed.end_episode(buffer);
    // A latch left over from the previous episode would poison the rest of the cell, and so
    // would a command chain still anchored on where the previous episode's last command left
    // the arm. `begin_episode` clears both; it is not disabling the plane (INV-12), since the
    // envelope, watchdogs and counters are untouched and the next violation latches again.
    safety.begin_episode();

    let mut overrides = ResetOverrides::default();
    perturbations.apply_at_reset(cell, seed, episode, &mut overrides);
    let hold = vec![0.0; nu];
    let mut step_state = StepState::new(
        &overrides,
        ms_to_steps(overrides.action_delay_ms),
        &hold,
        seed,
        cell,
        episode,
    );

    let obs_delay = ms_to_steps(overrides.observation_delay_ms);
    let mut ring: Vec<BTreeMap<String, Tensor>> = Vec::new();
    let mut ring_cursor = 0usize;
    let mut extra_age = 0u64;
    let mut ctrl = vec![0.0; nu];

    let mut frame_idx: Option<u64> = None;
    for step in 0..max_steps {
        let dropped = step_state.drop_observation();
        if dropped && !ring.is_empty() {
            extra_age += 1;
        } else {
            let (names, bytes, rendered) = capture(
                plan,
                sources,
                env.model(),
                &env.backend().state(),
                frames.as_deref_mut(),
                &overrides.light,
                cell_frames.as_deref_mut(),
            )?;
            frame_idx = rendered;
            let inputs: BTreeMap<String, TensorRef<'_>> = names
                .iter()
                .zip(&bytes)
                .map(|((name, dtype, shape), data)| {
                    (
                        name.clone(),
                        TensorRef::new(*dtype, shape.clone(), data.as_slice()),
                    )
                })
                .collect();
            let out = plan
                .run(&inputs)
                .map_err(|e| EvalError::Plan(e.to_string()))?;
            extra_age = 0;
            if ring.len() < obs_delay + 1 {
                ring.push(out);
            } else {
                ring[ring_cursor] = out;
                ring_cursor = (ring_cursor + 1) % ring.len();
            }
        }
        // The oldest frame in the ring is the delayed observation (§10.2 `observation_delay`).
        let observed = &ring[if ring.len() > obs_delay {
            ring_cursor
        } else {
            0
        }];
        // One policy invocation per control tick, which is what `BatchDomains::single_env()`
        // declares (inference period 1) and therefore what produced the demonstrations. The
        // *execution* cadence is the buffer's: a chunk drives `action.execute_chunk` ticks,
        // or the whole overlap under `TemporalEnsemble` (packet M5/V6b).
        let chunk = infer_chunk::<NJ, H>(policy, observed, action_output, mode)?;
        buffer.push(&chunk, u64::from(step));
        let (fed, _commanded) = plane_chunk(buffer, feed, u64::from(step), mode);

        // Before every `validate`, exactly like `es_data::Collector`, `es_ros2::hil` and
        // `es_runtime_embedded`: the plane decides that only the first call of an episode
        // seeds the command chain, so the envelope bounds the plane's own commands and not
        // the servo's following error (packet M5/V6, design note section 7.12).
        let (q, qd) = joint_state::<NJ>(&env.backend().state());
        safety.observe_state(&q, &qd);
        safety.heartbeat(env.tick());
        let age = Micros(((ring.len().saturating_sub(1) as u64) + extra_age) * control_us);
        let safe = safety.validate(&fed, age, env.tick());
        // One record per frame that reached disk, carrying the plane's own verdict on the step
        // that frame was captured for (design note section 8). A step whose observation was
        // dropped rendered nothing, so it adds no record and the two stay the same length.
        if let Some(frame) = frame_idx.take() {
            events.push(StepEvent {
                frame,
                tick: env.tick(),
                source: safe.source.into(),
                events: safe.events.bits(),
            });
        }

        // `nu == NJ` was checked at run start, so this is a copy, not a broadcast.
        ctrl.copy_from_slice(&safe.q);
        step_state.apply_per_step(&mut ctrl);
        let out = env.step(&ctrl)?;
        if let Some(ep) = out.episodes.into_iter().next() {
            return Ok(ep);
        }
    }
    // The step budget ran out before a terminal condition: close the open episode.
    env.reset(None)?
        .into_iter()
        .next()
        .ok_or_else(|| EvalError::Plan("the env closed no episode on reset".to_owned()))
}

/// The plan's input buffers, filled from the physics state.
///
/// Returns the descriptors and the owned bytes separately so the caller can build the
/// borrowed `TensorRef`s over them, plus the index of the frame this step wrote, if any.
type Captured = (Vec<(String, ElemType, Vec<u64>)>, Vec<Vec<u8>>, Option<u64>);

/// Where one plan input's values come from (§7.4, §10.1).
///
/// Resolved once, against the *documents* rather than guessed per step: an input is an image
/// because the Observation IR says `ImageInput`, not because nothing else matched it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Capture {
    /// One joint's `qpos` range in the loaded model.
    Qpos(es_physics_core::backend::IndexRange),
    Sensor(es_physics_core::backend::IndexRange),
    /// A Task IR `ObsSource::JointState { body, dof }` channel: the leading `dof` joint
    /// positions of env 0. The same convention [`joint_state`] uses for the Safety Plane —
    /// which is why `run_episode` refuses a model carrying fewer than `NJ` of them — and the
    /// only reading available, because that channel names a body and a count, not joints.
    Joints(usize),
    /// An `ObservationNode::ImageInput`: the frame source's bytes, unconverted.
    Image,
}

/// Resolves every plan input **before the first episode**, so an observation this build
/// cannot capture is one error at the start of the run rather than a surprise mid-table.
///
/// `model` is `None` when the frames are *recorded* rather than simulated (`crate::bake`): a
/// dataset carries `observation.state` and tiles, so the two arms that index a loaded model
/// have no reading there and the input is refused by name instead of guessed at.
pub(crate) fn input_sources(
    plan: &CpuPlan,
    obs: &ObservationIr,
    task: &TaskIr,
    model: Option<&ModelInfo>,
) -> Result<BTreeMap<String, Capture>, EvalError> {
    use es_ir::task::ObsSource;

    let mut out = BTreeMap::new();
    for (name, id) in &plan.inputs {
        if !matches!(plan.buffers[id.0].home, Home::Input(_)) {
            continue;
        }
        let source = StableId::from_hex(name).map_err(|e| EvalError::Plan(e.to_string()))?;
        let is_image =
            obs.graph.nodes.values().any(
                |n| matches!(n, ObservationNode::ImageInput { sensor, .. } if *sensor == source),
            );
        let joints = task
            .observation_spec
            .channels
            .values()
            .find_map(|c| match c.source {
                ObsSource::JointState { body, dof } if body == source => Some(dof as usize),
                _ => None,
            });
        let how = if is_image {
            Capture::Image
        } else if let Some(r) = model.and_then(|m| m.qpos.get(&source)) {
            Capture::Qpos(*r)
        } else if let Some(r) = model.and_then(|m| m.sensor.get(&source)) {
            Capture::Sensor(*r)
        } else if let Some(dof) = joints {
            Capture::Joints(dof)
        } else {
            let model = if model.is_some() {
                "a joint or sensor of the loaded model, "
            } else {
                // No model here means recorded frames, and saying "the loaded model" would
                // send the reader looking for a scene that this path never opens.
                "(there is no loaded model here: the frames are recorded) "
            };
            return Err(EvalError::Plan(format!(
                "observation input \"{name}\" is none of: {model}an ImageInput of the \
                 Observation IR, or a JointState channel of the Task IR's ObservationSpec"
            )));
        };
        out.insert(name.clone(), how);
    }
    Ok(out)
}

fn capture(
    plan: &CpuPlan,
    sources: &BTreeMap<String, Capture>,
    model: &ModelInfo,
    state: &StateView<'_>,
    mut frames: Option<&mut FrameSource<'_>>,
    light: &LightOverride,
    mut cell_frames: Option<&mut CellFrames>,
) -> Result<Captured, EvalError> {
    let mut descs = Vec::new();
    let mut bytes = Vec::new();
    let mut rendered = None;
    for (name, id) in &plan.inputs {
        let desc = &plan.buffers[id.0];
        let Home::Input(_) = &desc.home else {
            continue;
        };
        let how = sources.get(name).copied().ok_or_else(|| {
            EvalError::Plan(format!("observation input \"{name}\" is unresolved"))
        })?;
        let values: Vec<f64> = match how {
            Capture::Qpos(r) => state.qpos_of(0)[r.as_range()].to_vec(),
            Capture::Sensor(r) => state.sensordata[r.as_range()].to_vec(),
            Capture::Joints(dof) => {
                let q = state.qpos_of(0);
                if q.len() < dof {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" wants {dof} joint positions; the model \
                         carries {}",
                        q.len()
                    )));
                }
                q[..dof].to_vec()
            }
            Capture::Image => {
                // Without a frame source there is no renderer in this build (§4.3, es-render is
                // layer 5): feeding it zeros would produce a number, and a wrong number in the
                // §10.1 table is worse than no table.
                let Some(frame) = frames.as_deref_mut() else {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\" is an image; image inputs need a \
                         renderer, which this build has none of"
                    )));
                };
                // Exactly what the plan declared, or nothing: the frame is not resized, converted
                // or reordered here — every such step is an Observation IR node (§7.2, INV-14).
                let data = frame(light, model, state).map_err(EvalError::Plan)?;
                let want = desc.elems * elem_bytes(desc.dtype);
                if data.len() != want {
                    return Err(EvalError::Plan(format!(
                        "observation input \"{name}\": the plan wants {} {:?} elements \
                         ({want} bytes), the frame supplies {} bytes",
                        desc.elems,
                        desc.dtype,
                        data.len()
                    )));
                }
                // The frame the policy sees is the frame on disk: written here, from the same
                // bytes, before anything downstream can touch them.
                if let Some(cell) = cell_frames.as_deref_mut() {
                    rendered = Some(cell.write(desc.dtype, &desc.shape, &data)?);
                }
                descs.push((name.clone(), desc.dtype, desc.shape.clone()));
                bytes.push(data);
                continue;
            }
        };
        descs.push((name.clone(), desc.dtype, desc.shape.clone()));
        bytes.push(encode_state(name, desc.dtype, desc.elems, &values)?);
    }
    Ok((descs, bytes, rendered))
}

/// State values into one plan input buffer.
///
/// The one place this conversion happens. `crate::bake` calls it with a recorded
/// `observation.state` row where [`capture`] calls it with a live `qpos` slice, so a training
/// input and an inference input cannot be two different roundings of the same number
/// (design note `docs/design/visible-learning.md` section 7.9).
pub(crate) fn encode_state(
    name: &str,
    dtype: ElemType,
    elems: usize,
    values: &[f64],
) -> Result<Vec<u8>, EvalError> {
    if values.len() != elems {
        return Err(EvalError::Plan(format!(
            "observation input \"{name}\": the plan wants {elems} elements, the source supplies \
             {}",
            values.len()
        )));
    }
    Ok(match dtype {
        ElemType::F32 => values
            .iter()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect(),
        ElemType::F64 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        other => {
            return Err(EvalError::Plan(format!(
                "observation input \"{name}\" is {other:?}; state capture produces floats"
            )))
        }
    })
}

/// Bytes one element of `e` occupies in a plan buffer.
pub(crate) fn elem_bytes(e: ElemType) -> usize {
    match e {
        ElemType::F32 | ElemType::I32 => 4,
        ElemType::F16 | ElemType::Bf16 => 2,
        ElemType::F64 => 8,
        ElemType::U8 | ElemType::Bool => 1,
    }
}

/// The first `NJ` joint positions and velocities of env 0. Nothing is padded: `run_episode`
/// refused the run unless the model carries at least `NJ` of each.
fn joint_state<const NJ: usize>(state: &StateView<'_>) -> ([f64; NJ], [f64; NJ]) {
    let (mut q, mut qd) = ([0.0; NJ], [0.0; NJ]);
    q.copy_from_slice(&state.qpos_of(0)[..NJ]);
    qd.copy_from_slice(&state.qvel_of(0)[..NJ]);
    (q, qd)
}

/// One `infer` call, reshaped into the chunk the Safety Plane validates.
fn infer_chunk<const NJ: usize, const H: usize>(
    policy: &mut dyn PolicyRuntime,
    inputs: &BTreeMap<String, Tensor>,
    action_output: Option<&str>,
    mode: ExecutionMode,
) -> Result<ActionChunk<NJ, H>, EvalError> {
    let outputs = policy
        .infer(inputs)
        .map_err(|e| EvalError::Policy(e.to_string()))?;
    let tensor = match action_output {
        Some(name) => outputs
            .get(name)
            .ok_or_else(|| EvalError::Policy(format!("no policy output named \"{name}\"")))?,
        None if outputs.len() == 1 => outputs.values().next().expect("len == 1"),
        None => {
            return Err(EvalError::Policy(format!(
                "the policy has {} outputs; name the action chunk in RunConfig::action_output",
                outputs.len()
            )))
        }
    };
    let flat = to_f64(tensor)?;
    let rows = (flat.len() / NJ.max(1)).min(H);
    if rows == 0 {
        return Err(EvalError::Policy(format!(
            "the action tensor holds {} values, which is less than one row of {NJ} joints",
            flat.len()
        )));
    }
    let mut actions = [[0.0; NJ]; H];
    for (r, row) in actions.iter_mut().enumerate().take(rows) {
        for (j, v) in row.iter_mut().enumerate() {
            *v = flat[r * NJ + j];
        }
    }
    // No `seq` here: `es_env::plane_chunk` stamps it, once per result the buffer accepted,
    // which is what `SafetyPlane::accept` judges freshness by (spec 8.6).
    Ok(ActionChunk::new(actions, rows, mode))
}

/// The chunk-blend policy the Deployment IR's execution mode implies (spec 8.5, spec 9.2).
///
/// `es loop collect` reads the same decay out of the Learning IR's `ActionChunker`; the two
/// documents carry the same number for the demo, and the Deployment IR is what `es eval run`
/// is given. Everything that is not an ensemble is a hard switch, which is what the buffer's
/// `span` already assumes.
fn blend_of(mode: ExecutionMode) -> es_ir::learning::ChunkBlendPolicy {
    match mode {
        ExecutionMode::TemporalEnsemble { decay } => {
            es_ir::learning::ChunkBlendPolicy::TemporalEnsemble {
                weight_decay: decay as f32,
            }
        }
        _ => es_ir::learning::ChunkBlendPolicy::HardSwitch,
    }
}

fn to_f64(t: &Tensor) -> Result<Vec<f64>, EvalError> {
    match t.dtype {
        ElemType::F32 => Ok(t
            .data
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        ElemType::F64 => Ok(t
            .data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()),
        other => Err(EvalError::Policy(format!(
            "the action chunk is {other:?}; the Safety Plane takes floats"
        ))),
    }
}

/// Computes every declared metric for one cell. Every declared metric gets exactly one
/// `CellResult`, measured or `MetricValue::Unavailable` — never a missing row.
fn record_cell(
    ir: &EvaluationIr,
    cell: u32,
    suite: &str,
    episodes: &[Episode],
    counters: &es_safety::SafetyCounters,
    env_metrics: &EnvMetrics,
) -> ShardCell {
    let mut out = ShardCell {
        cell,
        results: Vec::new(),
        samples: Vec::new(),
    };
    // `MetricSpec::ALL` order, not the document's, so the report is byte-stable whatever
    // order the author listed the metrics in.
    for metric in MetricSpec::ALL {
        if !ir.metrics.contains(&metric) {
            continue;
        }
        out.results.push(CellResult {
            suite: suite.to_owned(),
            metric,
            value: metrics::compute(&metric, episodes, counters, env_metrics),
            n_episodes: episodes.len() as u32,
        });
        if let Some(v) = metrics::per_episode(&metric, episodes) {
            out.samples.push((metric, v));
        }
    }
    out
}

/// §10.2 acceptance. A criterion with no `suite` applies to every suite; a criterion whose
/// metric was not measured is `AcceptanceResult::Unavailable`, which is not a pass.
fn judge(
    ir: &EvaluationIr,
    measured: &BTreeMap<(String, MetricSpec), MetricValue>,
    samples: &BTreeMap<(String, MetricSpec), Vec<f64>>,
) -> Vec<AcceptanceResult> {
    let mut out = Vec::new();
    for c in &ir.acceptance {
        for suite in &ir.suites {
            if c.suite.as_ref().is_some_and(|s| *s != suite.name) {
                continue;
            }
            let key = (suite.name.clone(), c.metric);
            let unavailable = |reason: String| AcceptanceResult::Unavailable {
                metric: c.metric,
                reason,
            };
            let result = match measured.get(&key) {
                None => unavailable("the suite did not declare this metric".to_owned()),
                Some(MetricValue::Unavailable { reason }) => unavailable(reason.clone()),
                Some(MetricValue::Histogram(_)) => unavailable(
                    "a histogram has no scalar to compare against a threshold".to_owned(),
                ),
                Some(MetricValue::Scalar(cell)) => {
                    // A metric with a per-episode sample honours the criterion's aggregation;
                    // one that only exists at cell level has the same value under every
                    // aggregation (design note section 4).
                    let observed = samples
                        .get(&key)
                        .and_then(|v| metrics::aggregate(v, c.aggregation))
                        .unwrap_or(*cell);
                    AcceptanceResult::Determined {
                        criterion: c.clone(),
                        observed,
                        passed: c.comparator.holds(observed, c.threshold),
                    }
                }
            };
            out.push(result);
        }
    }
    out
}

/// §5.3. `learning` and `policy` come from the loaded `PolicyInfo`: `run` is not given the
/// `LearningGraph`, and these two digests are what the runtime can attest to.
fn hash_chain(
    task: &TaskIr,
    obs: &ObservationIr,
    deploy: &DeploymentIr,
    evaluation: [u8; 32],
    plan: &CpuPlan,
    policy: &dyn PolicyRuntime,
    cfg: &RunConfig,
) -> Result<HashChain, EvalError> {
    let bad = |d: es_ir::diag::Diagnostic| EvalError::InvalidIr(format!("{d}"));
    let info = policy.info();
    Ok(HashChain {
        asset: vec![task.scene.asset_hash],
        scene: task.scene.scene_hash,
        task_graph: canonical_hash(&task.graph).map_err(bad)?,
        task: task.task_hash().map_err(bad)?,
        observation: obs.observation_hash().map_err(bad)?,
        learning: info.map_or([0; 32], |i| i.lowering_hash),
        policy: info.map_or([0; 32], |i| i.weights_hash),
        dataset: cfg.dataset,
        deployment: deploy.deployment_hash().map_err(bad)?,
        evaluation: Some(evaluation),
        compiler: plan.compiler_hash(),
        runtime: policy.runtime_hash(),
        hardware: cfg.hardware,
    })
}

/// §10.5. `report.html` and `episodes/` are a later packet, and this writes neither.
pub fn write_artifacts(
    report: &EvaluationReport,
    lock: &EvaluationLock,
    dir: &Path,
) -> Result<(), EvalError> {
    let io = |path: &Path| {
        let p = path.display().to_string();
        move |source| EvalError::Io {
            path: p.clone(),
            source,
        }
    };
    fs::create_dir_all(dir).map_err(io(dir))?;
    for (name, json) in [
        ("report.json", serde_json::to_string_pretty(report)),
        ("evaluation.lock", serde_json::to_string_pretty(lock)),
    ] {
        let path = dir.join(name);
        let mut text = json.map_err(|e| EvalError::Plan(e.to_string()))?;
        text.push('\n');
        fs::write(&path, text).map_err(io(&path))?;
    }
    Ok(())
}
