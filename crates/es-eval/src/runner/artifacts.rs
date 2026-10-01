//! The two artifacts of §10.5: [`Evaluation::merge`] puts every worker's units back in
//! order, judges acceptance and computes the hash chain; [`write_artifacts`] writes
//! `report.json` and `evaluation.lock`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use es_compile::{CpuPlan, PlanMode};
use es_env::Env;
use es_ir::deployment::DeploymentIr;
use es_ir::evaluation::{
    AcceptanceResult, CellResult, EvaluationIr, EvaluationReport, MetricSpec, MetricValue,
};
use es_ir::hash::{canonical_hash, HashChain};
use es_ir::observation::ObservationIr;
use es_ir::task::TaskIr;
use es_physics_core::backend::PhysicsBackend;
use es_policy::PolicyRuntime;
use serde::{Deserialize, Serialize};

use super::{resolve_seeds, Evaluation, RunConfig, Shard, ShardCell, SCHEMA_VERSION};
use crate::metrics;
use crate::{hex32, EvalError};

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
    /// The engine the backend's load reply named (packet M11/X1), in plain text beside the
    /// `execution_hash` whose `hardware_capability` slot covers it on every backend but
    /// `mujoco-cpu`. Filled in by the caller that opened the backend -- `PhysicsBackend` has
    /// no such accessor (INV-17) -- and absent from the bytes when it did not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_version: Option<String>,
    /// The blake3 of the adapter script the backend spawned, in hex (packet M11/R1): covered
    /// by `hardware_capability` beside the engine version. Absent on `mujoco-cpu`, whose slot
    /// does not hash it, so no reference lock moves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script_blake3: Option<String>,
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
    /// The policy runtime's intra-op thread count, in plain text beside the `execution_hash`
    /// that covers it (§5.3, packet M10/W0a): Torch's CPU inference is not bitwise across
    /// counts, so two reports taken at two counts are two conditions. `None` for a runtime
    /// with no pool, and absent from the bytes then — `merge` writes `None` and the caller
    /// that built the runtime fills it in (`PolicyRuntime` has no such accessor: INV-17).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_threads: Option<u32>,
}

impl Evaluation {
    /// The §10.5 artifacts, from every worker's units put back in canonical order.
    ///
    /// **Every metric is computed here and only here** (packet M7/R1): a worker hands back one
    /// [`metrics::CellSummary`] per episode, this adds a cell's together in ascending episode
    /// order and computes the §10.1 row, the judgement, the hash chain and both artifacts from
    /// the result — so a cell is the same row whichever worker ran which of its episodes, and
    /// `--jobs N` cannot produce a report that `--jobs 1` would not have (§10.4).
    ///
    /// A unit set that is not exactly every `(cell, episode)` pair, each once, is refused. A
    /// report missing an episode because a worker was lost would otherwise carry a correct
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

        // Stable, so units of one cell keep the order their worker ran them in before the key
        // below puts the whole list into `(cell, episode)` order.
        let mut merged: Vec<&ShardCell> = shards.iter().flat_map(|s| &s.cells).collect();
        merged.sort_by_key(|c| (c.cell, c.episode));
        let n_episodes = resolve_seeds(ir).len() as u64;
        let covered: Vec<(u32, u64)> = merged.iter().map(|c| (c.cell, c.episode)).collect();
        let wanted = (0..ir.suites.len() as u32).flat_map(|c| (0..n_episodes).map(move |e| (c, e)));
        if covered.iter().copied().ne(wanted) {
            return Err(EvalError::Shard(format!(
                "the merged workers cover the (cell, episode) units {covered:?}; the evaluation \
                 has {} suite(s) x {n_episodes} episode(s) and every unit must appear exactly \
                 once",
                ir.suites.len()
            )));
        }

        let mut cells: Vec<CellResult> = Vec::new();
        let mut measured: BTreeMap<(String, MetricSpec), MetricValue> = BTreeMap::new();
        let mut samples: BTreeMap<(String, MetricSpec), Vec<f64>> = BTreeMap::new();
        for (cell, suite) in ir.suites.iter().enumerate() {
            let mut summary = metrics::CellSummary::default();
            for unit in merged.iter().filter(|u| u.cell as usize == cell) {
                summary.add(&unit.summary);
            }
            for r in cell_results(ir, &suite.name, &summary) {
                measured.insert((suite.name.clone(), r.metric), r.value.clone());
                cells.push(r);
            }
            for (metric, v) in &summary.samples {
                samples.insert((suite.name.clone(), *metric), v.clone());
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
                    engine_version: None,
                    script_blake3: None,
                }),
            created: cfg.created,
            runtime_threads: None,
        };
        Ok((report, lock))
    }
}

pub(super) fn backend_caps<B: PhysicsBackend>(env: &Env<B>) -> BackendCaps {
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
        engine_version: None,
        script_blake3: None,
    }
}

/// Computes every declared metric for one cell from its summed episodes. Every declared metric
/// gets exactly one `CellResult`, measured or `MetricValue::Unavailable` — never a missing row.
pub(super) fn cell_results(
    ir: &EvaluationIr,
    suite: &str,
    summary: &metrics::CellSummary,
) -> Vec<CellResult> {
    // `MetricSpec::ALL` order, not the document's, so the report is byte-stable whatever
    // order the author listed the metrics in.
    MetricSpec::ALL
        .into_iter()
        .filter(|m| ir.metrics.contains(m))
        .map(|metric| CellResult {
            suite: suite.to_owned(),
            metric,
            value: metrics::compute(&metric, summary),
            n_episodes: summary.n_episodes,
        })
        .collect()
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
