//! The five MCP tool bodies (spec 14.5): `validate` / `compile` / `estimate_cost` / `eval` /
//! `eval_run` / `hash_chain`. Each takes the `arguments` object of an MCP `tools/call` request
//! and returns either a JSON result or a [`ToolError`].
//!
//! No new abstractions here beyond what the CLI already does in `crates/es/src/cmd/{ir,task,
//! bench,eval}.rs` -- this module calls the same library functions those commands call
//! (`es_ir::serial`, `es_compile::CpuPlan`, `es_compile::budget::MemoryBudget`), it just returns
//! JSON instead of printing a table, so a tool this crate cannot shell out to `es` reuses.

use std::fmt::Write as _;

use es_compile::budget::{BudgetDomains, BudgetInputs, BudgetItem, MemoryBudget, Precision};
use es_compile::{CpuPlan, PlanMode};
use es_ir::deployment::DeploymentIr;
use es_ir::evaluation::{CellResult, EvaluationIr, EvaluationReport, MetricValue};
use es_ir::learning::LearningGraph;
use es_ir::observation::ObservationIr;
use es_ir::serial::{
    deployment_from_toml, evaluation_from_toml, learning_from_toml, observation_from_toml,
    task_from_toml, IrKind, SerialError,
};
use es_ir::task::TaskIr;
use es_ir::Diagnostic;
use serde_json::{json, Value};

/// An MCP tool call failed. `BadParams` is a protocol-level error (JSON-RPC `-32602`,
/// spec 25.1 "malformed input"): the caller's `arguments` object itself is wrong shape.
/// `Failed` is a tool-execution error (spec MCP `isError: true`): the arguments parsed, but
/// what they named (a TOML file, a report) did not work -- never a crash either way.
#[derive(Debug)]
pub enum ToolError {
    BadParams(String),
    Failed(String),
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn require_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, ToolError> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::BadParams(format!("missing or non-string '{key}'")))
}

fn require_u32(args: &Value, key: &str) -> Result<u32, ToolError> {
    args.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| ToolError::BadParams(format!("missing or non-integer '{key}'")))
}

// --- `validate` -------------------------------------------------------------------------------

/// One of the five typed IRs, loaded from its own envelope kind (mirrors `crate::cmd::ir::AnyIr`
/// in `crates/es/src/cmd/ir.rs`, which this crate cannot depend on -- `es` is layer 12's binary,
/// `es-script` is layer 11).
enum AnyIr {
    Task(TaskIr),
    Observation(ObservationIr),
    Learning(LearningGraph),
    Deployment(DeploymentIr),
    Evaluation(EvaluationIr),
}

impl AnyIr {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Task(_) => "task",
            Self::Observation(_) => "observation",
            Self::Learning(_) => "learning",
            Self::Deployment(_) => "deployment",
            Self::Evaluation(_) => "evaluation",
        }
    }

    fn validate(&self) -> Vec<Diagnostic> {
        match self {
            Self::Task(ir) => ir.validate(),
            Self::Observation(ir) => ir.validate(),
            Self::Learning(ir) => ir.validate(),
            Self::Deployment(ir) => ir.validate(),
            Self::Evaluation(ir) => ir.validate(),
        }
    }

    fn hash(&self) -> Result<[u8; 32], Diagnostic> {
        match self {
            Self::Task(ir) => ir.task_hash(),
            Self::Observation(ir) => ir.observation_hash(),
            Self::Learning(ir) => ir.learning_hash(),
            Self::Deployment(ir) => ir.deployment_hash(),
            Self::Evaluation(ir) => ir.evaluation_hash(),
        }
    }
}

fn parse_kind_str(k: &str) -> Result<IrKind, ToolError> {
    match k {
        "task" => Ok(IrKind::Task),
        "observation" => Ok(IrKind::Observation),
        "learning" => Ok(IrKind::Learning),
        "deployment" => Ok(IrKind::Deployment),
        "evaluation" => Ok(IrKind::Evaluation),
        other => Err(ToolError::BadParams(format!("unknown IR kind '{other}'"))),
    }
}

/// Sniffs which IR kind the TOML envelope (spec 14.1) declares, with no extra `toml` crate
/// dependency: `task_from_toml` already parses the envelope and reports `KindMismatch { found,
/// .. }` when it is some other kind, so a first parse attempt doubles as the sniff.
fn sniff_kind(raw: &str) -> Result<IrKind, ToolError> {
    match task_from_toml(raw) {
        Ok(_) => Ok(IrKind::Task),
        Err(SerialError::KindMismatch { found, .. }) => Ok(found),
        Err(e) => Err(ToolError::Failed(e.to_string())),
    }
}

fn load_any(kind: Option<&str>, raw: &str) -> Result<AnyIr, ToolError> {
    let kind = match kind {
        Some(k) => parse_kind_str(k)?,
        None => sniff_kind(raw)?,
    };
    let err = |e: SerialError| ToolError::Failed(e.to_string());
    Ok(match kind {
        IrKind::Task => AnyIr::Task(task_from_toml(raw).map_err(err)?),
        IrKind::Observation => AnyIr::Observation(observation_from_toml(raw).map_err(err)?),
        IrKind::Learning => AnyIr::Learning(learning_from_toml(raw).map_err(err)?),
        IrKind::Deployment => AnyIr::Deployment(deployment_from_toml(raw).map_err(err)?),
        IrKind::Evaluation => AnyIr::Evaluation(evaluation_from_toml(raw).map_err(err)?),
    })
}

/// `validate { kind?, toml }` -> diagnostics JSON (spec 11.1: each IR validates itself).
pub fn validate(args: &Value) -> Result<Value, ToolError> {
    let toml_src = require_str(args, "toml")?;
    let kind = args.get("kind").and_then(Value::as_str);
    let ir = load_any(kind, toml_src)?;

    let diags = ir.validate();
    let has_error = diags.iter().any(Diagnostic::is_error);
    let (hash, hash_error) = match ir.hash() {
        Ok(h) => (Some(hex(&h)), None),
        Err(d) => (None, Some(serde_json::to_value(&d).unwrap_or(Value::Null))),
    };
    Ok(json!({
        "kind": ir.kind_name(),
        "diagnostics": diags,
        "has_error": has_error,
        "hash": hash,
        "hash_error": hash_error,
    }))
}

// --- `compile` --------------------------------------------------------------------------------

/// `compile { observation_toml, mode }` -> plan summary + `compiler_hash` (spec 11.1 Lower).
pub fn compile(args: &Value) -> Result<Value, ToolError> {
    let obs_toml = require_str(args, "observation_toml")?;
    let mode = match require_str(args, "mode")? {
        "debug" => PlanMode::Debug,
        "release" => PlanMode::Release,
        other => {
            return Err(ToolError::BadParams(format!(
                "mode must be 'debug' or 'release', got '{other}'"
            )))
        }
    };
    let obs = observation_from_toml(obs_toml).map_err(|e| ToolError::Failed(e.to_string()))?;

    match CpuPlan::compile(&obs, mode) {
        Ok(plan) => Ok(json!({
            "steps": plan.steps.len(),
            "buffers": plan.buffers.len(),
            "arena_elems": plan.arena_elems,
            "inputs": plan.inputs.keys().collect::<Vec<_>>(),
            "outputs": plan.outputs.keys().collect::<Vec<_>>(),
            "warnings": plan.warnings,
            "compiler_hash": hex(&plan.compiler_hash()),
        })),
        Err(diags) => Err(ToolError::Failed(
            diags.iter().map(ToString::to_string).collect::<String>(),
        )),
    }
}

// --- `estimate_cost` --------------------------------------------------------------------------

fn item_json(i: &BudgetItem) -> Value {
    json!({"name": i.name, "bytes": i.bytes, "formula": i.formula})
}

/// `estimate_cost { observation_toml, learning_toml?, sim_envs, obs_envs, views,
/// inference_batch, precision }` -> `MemoryReport` JSON + spec 20.3 violations. No `ModelSizes`
/// / tile-atlas / device budget here (an MCP caller has no scene or GPU to size those from);
/// `physics_state` and `render_tile_atlas` (without a camera-shaped observation IR) report
/// `unavailable`, same as `es bench --memory-report` without `--scene`/`--tile-*`.
pub fn estimate_cost(args: &Value) -> Result<Value, ToolError> {
    let obs_toml = require_str(args, "observation_toml")?;
    let learning_toml = args.get("learning_toml").and_then(Value::as_str);
    let domains = BudgetDomains {
        n_sim_envs: require_u32(args, "sim_envs")?,
        n_obs_envs: require_u32(args, "obs_envs")?,
        n_views: require_u32(args, "views")?,
        inference_batch: require_u32(args, "inference_batch")?,
    };
    let precision = match require_str(args, "precision")? {
        "f32" => Precision::F32,
        "f16" => Precision::F16,
        other => {
            return Err(ToolError::BadParams(format!(
                "precision must be 'f32' or 'f16', got '{other}'"
            )))
        }
    };

    let obs = observation_from_toml(obs_toml).map_err(|e| ToolError::Failed(e.to_string()))?;
    let learning = learning_toml
        .map(|s| learning_from_toml(s).map_err(|e| ToolError::Failed(e.to_string())))
        .transpose()?;

    let inputs = BudgetInputs {
        domains,
        obs: &obs,
        learning: learning.as_ref(),
        model: None,
        precision,
        tile_atlas: None,
        device_bytes: None,
    };
    let report = MemoryBudget::estimate(&inputs);
    let violations = report.violations(&inputs);

    Ok(json!({
        "items": report.items.iter().map(item_json).collect::<Vec<_>>(),
        "total_bytes": report.total_bytes,
        "per_domain": report.per_domain,
        "bandwidth_per_tick": report.bandwidth_per_tick.as_ref().map(item_json),
        "violations": violations.iter().map(|v| json!({"rule": v.rule, "message": v.message})).collect::<Vec<_>>(),
    }))
}

// --- `eval` (compare two reports) -----------------------------------------------------------

fn scalar(v: &MetricValue) -> Option<f64> {
    match v {
        MetricValue::Scalar(x) => Some(*x),
        MetricValue::Histogram(_) | MetricValue::Unavailable { .. } => None,
    }
}

fn compare_row(a: &CellResult, b: Option<&CellResult>) -> Value {
    let a_val = serde_json::to_value(&a.value).unwrap_or(Value::Null);
    match b {
        None => {
            json!({"suite": a.suite, "metric": a.metric.name(), "a": a_val, "b": null, "delta": null})
        }
        Some(b) => {
            let delta = match (scalar(&a.value), scalar(&b.value)) {
                (Some(av), Some(bv)) => json!(bv - av),
                _ => Value::Null,
            };
            let b_val = serde_json::to_value(&b.value).unwrap_or(Value::Null);
            json!({"suite": a.suite, "metric": a.metric.name(), "a": a_val, "b": b_val, "delta": delta})
        }
    }
}

/// `eval { report_a_json, report_b_json }` -> the compare table JSON (spec 10.5 `es eval
/// compare`, minus the Welch-t-test significance column that CLI command adds -- a per-cell
/// scalar delta is what spec 14.5 asks an external caller for; the CLI keeps the fuller table
/// for a human).
pub fn eval_compare(args: &Value) -> Result<Value, ToolError> {
    let a_raw = require_str(args, "report_a_json")?;
    let b_raw = require_str(args, "report_b_json")?;
    let a: EvaluationReport = serde_json::from_str(a_raw)
        .map_err(|e| ToolError::Failed(format!("report_a_json: {e}")))?;
    let b: EvaluationReport = serde_json::from_str(b_raw)
        .map_err(|e| ToolError::Failed(format!("report_b_json: {e}")))?;

    let mut rows: Vec<Value> = a
        .cells
        .iter()
        .map(|ca| {
            let cb = b
                .cells
                .iter()
                .find(|cb| cb.suite == ca.suite && cb.metric == ca.metric);
            compare_row(ca, cb)
        })
        .collect();
    for cb in &b.cells {
        if !a
            .cells
            .iter()
            .any(|ca| ca.suite == cb.suite && ca.metric == cb.metric)
        {
            rows.push(json!({
                "suite": cb.suite, "metric": cb.metric.name(),
                "a": null, "b": serde_json::to_value(&cb.value).unwrap_or(Value::Null), "delta": null,
            }));
        }
    }

    Ok(json!({"rows": rows, "a_passed": a.passed, "b_passed": b.passed}))
}

/// `eval_run { ... }`: the actual evaluation run needs a `PhysicsBackend` and a `PolicyRuntime`
/// (spec 9.6, spec 17.1), neither of which `es-script` links -- mirrors `es eval run` reporting
/// `SKIPPED` when a backend is unavailable (spec 1.4: never fake a result this machine cannot
/// produce). Use the `es eval run` CLI for an actual run.
pub fn eval_run(_args: &Value) -> Result<Value, ToolError> {
    Ok(json!({
        "status": "SKIPPED",
        "reason": "es-script (layer 11) has no PhysicsBackend/PolicyRuntime dependency; \
                   run `es eval run --config ... --policy ... --scene ...` for an actual evaluation",
    }))
}

// --- `hash_chain` -----------------------------------------------------------------------------

fn slot(h: Result<[u8; 32], Diagnostic>) -> Value {
    match h {
        Ok(h) => json!(hex(&h)),
        Err(d) => json!({"error": d.to_string()}),
    }
}

/// `hash_chain { task?, observation?, learning?, deployment? }` -> the spec 5.3 hash chain
/// slots knowable from IR TOML alone at authoring time (mirrors `es ir check`'s hash-chain
/// footer). `dataset` / `runtime` / `hardware` are only known at run time and always print
/// `"unset"`, same as the CLI.
pub fn hash_chain(args: &Value) -> Result<Value, ToolError> {
    let mut out = serde_json::Map::new();

    out.insert(
        "task".to_owned(),
        match args.get("task").and_then(Value::as_str) {
            Some(t) => slot(
                task_from_toml(t)
                    .map_err(|e| ToolError::Failed(format!("task: {e}")))?
                    .task_hash(),
            ),
            None => json!("unset"),
        },
    );
    out.insert(
        "observation".to_owned(),
        match args.get("observation").and_then(Value::as_str) {
            Some(o) => slot(
                observation_from_toml(o)
                    .map_err(|e| ToolError::Failed(format!("observation: {e}")))?
                    .observation_hash(),
            ),
            None => json!("unset"),
        },
    );
    if let Some(l) = args.get("learning").and_then(Value::as_str) {
        let learning =
            learning_from_toml(l).map_err(|e| ToolError::Failed(format!("learning: {e}")))?;
        out.insert("learning".to_owned(), slot(learning.learning_hash()));
        out.insert("policy".to_owned(), slot(learning.policy_hash()));
    } else {
        out.insert("learning".to_owned(), json!("unset"));
        out.insert("policy".to_owned(), json!("unset"));
    }
    out.insert(
        "deployment".to_owned(),
        match args.get("deployment").and_then(Value::as_str) {
            Some(d) => slot(
                deployment_from_toml(d)
                    .map_err(|e| ToolError::Failed(format!("deployment: {e}")))?
                    .deployment_hash(),
            ),
            None => json!("unset"),
        },
    );
    out.insert("dataset".to_owned(), json!("unset"));
    out.insert("runtime".to_owned(), json!("unset"));
    out.insert("hardware".to_owned(), json!("unset"));

    Ok(Value::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_missing_toml_as_bad_params() {
        let err = validate(&json!({})).expect_err("bad params");
        assert!(matches!(err, ToolError::BadParams(_)));
    }

    #[test]
    fn validate_sniffs_kind_and_reports_no_error_for_an_empty_task() {
        let toml = es_ir::serial::task_to_toml(&TaskIr {
            schema_version: 1,
            scene: es_ir::task::SceneRef {
                path: "scenes/fixture.usda".to_owned(),
                scene_hash: [0; 32],
                asset_hash: [0; 32],
            },
            graph: es_ir::graph::Graph::new(1),
            observation_spec: es_ir::task::ObservationSpec {
                channels: std::collections::BTreeMap::default(),
            },
            control: None,
            config: es_ir::task::TaskConfig {
                max_episode_steps: 10,
                control_rate_hz: 50.0,
                deterministic: true,
                rng_streams: std::collections::BTreeSet::default(),
            },
        })
        .expect("toml");
        let result = validate(&json!({"toml": toml})).expect("validates");
        assert_eq!(result["kind"], "task");
        assert_eq!(result["has_error"], false);
        assert!(result["hash"].is_string());
    }

    #[test]
    fn estimate_cost_rejects_bad_precision() {
        let toml = es_ir::serial::observation_to_toml(&ObservationIr {
            schema_version: 1,
            task_ref: [0; 32],
            graph: es_ir::graph::Graph::new(1),
            temporal: es_ir::observation::TemporalModel::default(),
            outputs: std::collections::BTreeMap::default(),
        })
        .expect("toml");
        let err = estimate_cost(&json!({
            "observation_toml": toml,
            "sim_envs": 1, "obs_envs": 1, "views": 1, "inference_batch": 1,
            "precision": "fp8",
        }))
        .expect_err("bad params");
        assert!(matches!(err, ToolError::BadParams(_)));
    }

    #[test]
    fn eval_run_always_reports_skipped() {
        let v = eval_run(&json!({})).expect("skipped, not failed");
        assert_eq!(v["status"], "SKIPPED");
    }

    #[test]
    fn hash_chain_with_nothing_given_is_all_unset() {
        let v = hash_chain(&json!({})).expect("no ir given is fine");
        for slot in [
            "task",
            "observation",
            "learning",
            "policy",
            "deployment",
            "dataset",
            "runtime",
            "hardware",
        ] {
            assert_eq!(v[slot], "unset", "{slot}");
        }
    }
}
