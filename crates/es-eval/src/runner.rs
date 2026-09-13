//! The cell loop, the hash chain, and the two artifacts of §10.5.
//!
//! One `Env`, one `SafetyPlane` and one `CpuPlan` per cell — see
//! `docs/design/evaluation-execution.md` section 2 for why the `Env` is rebuilt per cell
//! (fairness, §10.4) rather than shared.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use es_assets::scene::SceneDesc;
use es_compile::{CpuPlan, Home, PlanMode, Tensor, TensorRef};
use es_core::StableId;
use es_env::scheduler::BatchDomains;
use es_env::{Env, EnvMetrics, Episode};
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
use es_safety::{ActionChunk, SafetyPlane};
use serde::{Deserialize, Serialize};

use crate::metrics;
use crate::perturb::{PerturbationPlan, ResetOverrides, StepState};
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
        mut new_backend: F,
        cfg: &RunConfig,
    ) -> Result<(EvaluationReport, EvaluationLock), EvalError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
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

        let mut perturbations: Option<PerturbationPlan> = None;
        let mut caps: Option<BackendCaps> = None;
        let mut cells: Vec<CellResult> = Vec::new();
        let mut measured: BTreeMap<(String, MetricSpec), MetricValue> = BTreeMap::new();
        let mut samples: BTreeMap<(String, MetricSpec), Vec<f64>> = BTreeMap::new();

        for (cell, suite) in ir.suites.iter().enumerate() {
            let mut env: Env<B> = Env::new(task, scene, new_backend(), &domains, seeds[0])?;
            if perturbations.is_none() {
                perturbations = Some(PerturbationPlan::compile(ir, scene, env.model())?);
                caps = Some(backend_caps(&env));
            }
            let perturbations = perturbations.as_ref().expect("just compiled");
            let mut safety = SafetyPlane::<NJ, H>::from_ir(deploy)
                .map_err(|e| EvalError::Safety(e.to_string()))?;

            let mut episodes = Vec::with_capacity(seeds.len());
            // Monotonic per policy invocation and per cell: the plane treats a chunk as new
            // iff its `seq` grew (spec 8.6), and the plane lives as long as the cell.
            let mut seq = 0u64;
            for (idx, seed) in seeds.iter().enumerate() {
                episodes.push(run_episode::<B, NJ, H>(
                    &mut env,
                    &mut plan,
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
                    &mut seq,
                )?);
            }

            let env_metrics = env.metrics();
            record_cell(
                ir,
                &suite.name,
                &episodes,
                safety.counters(),
                &env_metrics,
                &mut cells,
                &mut measured,
                &mut samples,
            );
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
            seeds,
            backend: caps.unwrap_or_else(|| BackendCaps {
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

/// INV-15. An `Augment` node outside the allow-list refuses the run; the graph is never
/// rewritten, because stripping a node would make `observation_hash` describe a graph the
/// caller never declared.
fn refuse_augmentation(ir: &EvaluationIr, obs: &ObservationIr) -> Result<(), EvalError> {
    use es_ir::evaluation::AugmentationPolicy;
    for (id, node) in &obs.graph.nodes {
        if !matches!(node, ObservationNode::Augment { .. }) {
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
    seq: &mut u64,
) -> Result<Episode, EvalError> {
    env.reset(None)?;
    // A latch left over from the previous episode would poison the rest of the cell. Clearing
    // it is not disabling the plane (INV-12): the envelope, watchdogs and counters are
    // untouched and the next violation latches again.
    safety.reset_latch();

    let nu = env.model().nu as usize;
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

    for _ in 0..max_steps {
        let dropped = step_state.drop_observation();
        if dropped && !ring.is_empty() {
            extra_age += 1;
        } else {
            let (names, bytes) = capture(plan, env.model(), &env.backend().state())?;
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
        *seq += 1;
        let chunk = infer_chunk::<NJ, H>(policy, observed, action_output, mode, *seq)?;

        let (q, qd) = joint_state::<NJ>(&env.backend().state());
        safety.observe_state(&q, &qd);
        safety.heartbeat(env.tick());
        let age = Micros(((ring.len().saturating_sub(1) as u64) + extra_age) * control_us);
        let safe = safety.validate(&chunk, age, env.tick());

        for (i, v) in ctrl.iter_mut().enumerate() {
            *v = safe.q[i.min(NJ - 1)];
        }
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
/// borrowed `TensorRef`s over them.
type Captured = (Vec<(String, ElemType, Vec<u64>)>, Vec<Vec<u8>>);

fn capture(
    plan: &CpuPlan,
    model: &ModelInfo,
    state: &StateView<'_>,
) -> Result<Captured, EvalError> {
    let mut descs = Vec::new();
    let mut bytes = Vec::new();
    for (name, id) in &plan.inputs {
        let desc = &plan.buffers[id.0];
        let Home::Input(_) = &desc.home else {
            continue;
        };
        let source = StableId::from_hex(name).map_err(|e| EvalError::Plan(e.to_string()))?;
        let values: Vec<f64> = if let Some(r) = model.qpos.get(&source) {
            state.qpos_of(0)[r.as_range()].to_vec()
        } else if let Some(r) = model.sensor.get(&source) {
            state.sensordata[r.as_range()].to_vec()
        } else {
            // An image input: there is no renderer in this build (§4.3, es-render is layer 5).
            // Feeding it zeros would produce a number, and a wrong number in the §10.1 table
            // is worse than no table.
            return Err(EvalError::Plan(format!(
                "observation input \"{name}\" is not a joint or sensor of the loaded model; \
                 image inputs need a renderer, which this build has none of"
            )));
        };
        if values.len() != desc.elems {
            return Err(EvalError::Plan(format!(
                "observation input \"{name}\": the plan wants {} elements, the model supplies {}",
                desc.elems,
                values.len()
            )));
        }
        let data = match desc.dtype {
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
        };
        descs.push((name.clone(), desc.dtype, desc.shape.clone()));
        bytes.push(data);
    }
    Ok((descs, bytes))
}

/// The first `NJ` joint positions and velocities of env 0.
fn joint_state<const NJ: usize>(state: &StateView<'_>) -> ([f64; NJ], [f64; NJ]) {
    let (mut q, mut qd) = ([0.0; NJ], [0.0; NJ]);
    let (qpos, qvel) = (state.qpos_of(0), state.qvel_of(0));
    for i in 0..NJ {
        q[i] = qpos.get(i).copied().unwrap_or(0.0);
        qd[i] = qvel.get(i).copied().unwrap_or(0.0);
    }
    (q, qd)
}

/// One `infer` call, reshaped into the chunk the Safety Plane validates.
fn infer_chunk<const NJ: usize, const H: usize>(
    policy: &mut dyn PolicyRuntime,
    inputs: &BTreeMap<String, Tensor>,
    action_output: Option<&str>,
    mode: ExecutionMode,
    seq: u64,
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
    Ok(ActionChunk::new(actions, rows, mode).with_seq(seq))
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
#[allow(clippy::too_many_arguments)]
fn record_cell(
    ir: &EvaluationIr,
    suite: &str,
    episodes: &[Episode],
    counters: &es_safety::SafetyCounters,
    env_metrics: &EnvMetrics,
    cells: &mut Vec<CellResult>,
    measured: &mut BTreeMap<(String, MetricSpec), MetricValue>,
    samples: &mut BTreeMap<(String, MetricSpec), Vec<f64>>,
) {
    // `MetricSpec::ALL` order, not the document's, so the report is byte-stable whatever
    // order the author listed the metrics in.
    for metric in MetricSpec::ALL {
        if !ir.metrics.contains(&metric) {
            continue;
        }
        let value = metrics::compute(&metric, episodes, counters, env_metrics);
        cells.push(CellResult {
            suite: suite.to_owned(),
            metric,
            value: value.clone(),
            n_episodes: episodes.len() as u32,
        });
        if let Some(v) = metrics::per_episode(&metric, episodes) {
            samples.insert((suite.to_owned(), metric), v);
        }
        measured.insert((suite.to_owned(), metric), value);
    }
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
