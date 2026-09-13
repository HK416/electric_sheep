//! Oracle for the M4 W6 `RoboVerse` / `MetaSim` task conversion
//! (`docs/packets/M4/W6-roboverse-import.md`).
//!
//! Each converted Task IR / Observation IR pair must validate on its own and must agree on
//! the Task <-> Observation boundary of `es_ir::cross::check` (spec §7.4). `cross::check`
//! takes a full five-IR `IrBundle`, so this file closes the loop with a deliberately trivial
//! Learning IR / Deployment IR (mirrors `crates/es-data/tests/lerobot_config.rs`'s
//! `task_and_deployment` helper) — only the Task <-> Observation boundary (`XIR-001`,
//! `XIR-002`) is asserted clean; the other boundaries are free to disagree with these
//! placeholders.

use std::collections::BTreeMap;

use es_core::time::TickRate;
use es_data::roboverse::{convert, RoboVerseTask, Severity};
use es_ir::cross::{self, IrBundle};
use es_ir::deployment::{
    ActionContract, ActionSpace as DepSpace, Deadlines, DeploymentIr, ExecutionMode,
    FallbackPolicy, Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope,
    WatchdogSet, Workspace,
};
use es_ir::graph::Graph;
use es_ir::learning::{
    ActionExecutionMode, ArchKind, LearningGraph, PolicyContract, PolicyHandle, RuntimeHints,
    WeightsRef,
};
use es_ir::types::ElemType;
use es_ir::Diagnostic;

const PICK_AND_PLACE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/roboverse/pick_and_place.json"
));
const REACH_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/roboverse/reach.json"
));
const UNKNOWN_CHECKER_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/roboverse/unknown_checker.json"
));

fn errors(diags: &[Diagnostic]) -> Vec<String> {
    diags
        .iter()
        .filter(|d| d.is_error())
        .map(|d| format!("{} {}", d.code, d.message))
        .collect()
}

/// A Learning IR / Deployment IR that type-checks but agrees with nothing (empty contract,
/// zero joints): enough to complete an `IrBundle` for `cross::check` without claiming
/// anything about the boundaries this test does not exercise.
fn placeholder_learning_and_deployment() -> (LearningGraph, DeploymentIr) {
    let learning = LearningGraph {
        schema_version: 1,
        inputs: vec![],
        nodes: Graph::new(1),
        outputs: vec![],
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0u8; 32],
            },
            contract: PolicyContract {
                inputs: BTreeMap::new(),
                observation_window: 1,
                action_dim: 0,
                horizon: 1,
                execute_chunk: 1,
                replanning_hz: 1.0,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 0.0,
                    deadline_ms: 1000.0,
                },
            },
        },
    };
    let deployment = DeploymentIr {
        schema_version: 1,
        robot: RobotRef {
            name: "placeholder".to_owned(),
            target: RobotTarget::Simulated {
                scene: "placeholder".to_owned(),
            },
            n_joints: 0,
        },
        action: ActionContract {
            space: DepSpace::JointPosition,
            dim: 0,
            horizon: 1,
            execute_chunk: 1,
        },
        safety: SafetyEnvelope {
            position: vec![],
            position_soft_margin: vec![],
            velocity_max: vec![],
            acceleration_max: vec![],
            torque_max: vec![],
            jerk_max: None,
            action_rate: RateLimit {
                first_diff_max: vec![],
                second_diff_max: vec![],
            },
            workspace: Workspace::Box {
                min: [0.0; 3],
                max: [1.0; 3],
            },
            ee_velocity_max: 1.0,
            min_self_distance: 0.01,
            min_env_distance: 0.01,
            contact_force_max: 10.0,
        },
        execution: ExecutionMode::RecedingHorizon,
        deadlines: Deadlines {
            observation_age: Micros(0),
            inference_budget: Micros(1_000_000),
            actuation_budget: Micros(1_000),
        },
        watchdogs: WatchdogSet(vec![]),
        fallback: FallbackPolicy::HoldPosition,
        rate: RateSpec {
            control: TickRate::hz(100),
            inference: TickRate::hz(10),
        },
    };
    (learning, deployment)
}

/// `TaskIr::validate`, `ObservationIr::validate`, and only the `XIR-001`/`XIR-002` (Task <->
/// Observation) diagnostics out of a full `cross::check` — the boundary spec §7.4 and this
/// packet's acceptance criteria name.
fn assert_clean(task: &es_ir::task::TaskIr, observation: &es_ir::observation::ObservationIr) {
    assert!(
        errors(&task.validate()).is_empty(),
        "{:#?}",
        errors(&task.validate())
    );
    assert!(
        errors(&observation.validate()).is_empty(),
        "{:#?}",
        errors(&observation.validate())
    );
    let (learning, deployment) = placeholder_learning_and_deployment();
    let diags = cross::check(&IrBundle {
        task,
        observation,
        learning: &learning,
        deployment: &deployment,
        evaluation: None,
    });
    let task_obs_errors: Vec<String> = errors(&diags)
        .into_iter()
        .filter(|m| m.starts_with("XIR-001") || m.starts_with("XIR-002"))
        .collect();
    assert!(task_obs_errors.is_empty(), "{task_obs_errors:#?}");
}

#[test]
fn pick_and_place_converts_and_validates_clean() {
    let task = RoboVerseTask::parse(PICK_AND_PLACE_JSON).unwrap();
    let conv = convert(&task).unwrap();
    assert!(conv.unmapped.is_empty(), "{:#?}", conv.unmapped);
    assert_clean(&conv.task, &conv.observation);
    assert_eq!(conv.provenance.license.as_deref(), Some("Apache-2.0"));
    assert_eq!(conv.scene_refs.len(), 2, "one robot + one object");
}

#[test]
fn reach_converts_and_validates_clean_with_randomization() {
    let task = RoboVerseTask::parse(REACH_JSON).unwrap();
    let conv = convert(&task).unwrap();
    assert!(conv.unmapped.is_empty(), "{:#?}", conv.unmapped);
    assert_clean(&conv.task, &conv.observation);
    assert!(
        conv.task
            .config
            .rng_streams
            .iter()
            .any(|s| s.starts_with("rand_")),
        "{:?}",
        conv.task.config.rng_streams
    );
}

#[test]
fn unknown_checker_is_an_unmapped_error() {
    let task = RoboVerseTask::parse(UNKNOWN_CHECKER_JSON).unwrap();
    let conv = convert(&task).unwrap();
    assert_eq!(conv.unmapped.len(), 1, "{:#?}", conv.unmapped);
    assert_eq!(conv.unmapped[0].severity, Severity::Error);
    assert!(conv.unmapped[0].item.contains("SomeFutureChecker"));
}

#[test]
fn task_hash_and_observation_hash_are_stable_across_runs() {
    let task = RoboVerseTask::parse(PICK_AND_PLACE_JSON).unwrap();
    let a = convert(&task).unwrap();
    let b = convert(&task).unwrap();
    assert_eq!(a.task.task_hash().unwrap(), b.task.task_hash().unwrap());
    assert_eq!(
        a.observation.observation_hash().unwrap(),
        b.observation.observation_hash().unwrap()
    );
}
