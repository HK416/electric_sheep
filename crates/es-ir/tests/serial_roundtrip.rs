//! P28 oracle: TOML round-trip for the five typed IRs, plus the `.esgraph`/`.eslayout` split.
//!
//! Fixtures are built by hand (no `testing` helpers: those are owned by other in-flight
//! packets) and are deliberately tiny — just enough to exercise every field shape `serial.rs`
//! has to handle, in particular the `NodeId`-keyed node map that TOML cannot serialize as-is.

use std::collections::{BTreeMap, BTreeSet};

use es_core::time::TickRate;
use es_core::StableId;

use es_ir::deployment::{
    ActionContract, ActionSpace as DepActionSpace, Deadlines, DeploymentIr, FallbackPolicy, Limit,
    Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope, Watchdog, WatchdogSet,
    Workspace,
};
use es_ir::evaluation::{AugmentationPolicy, EpisodeBatch, EvaluationIr, ReplayPolicy, SeedPlan};
use es_ir::learning::{
    ActionExecutionMode, ArchKind, LearningGraph, LearningNode, PolicyContract, PolicyHandle,
    RuntimeHints, StateEncoderKind, WeightsRef,
};
use es_ir::observation::{Io, ObservationIr, ObservationNode};
use es_ir::serial::{self, AnyIr, IrKind, Layout, SerialError};
use es_ir::task::{ObservationSpec, SceneRef, TaskConfig, TaskGraph, TaskIr, TaskNode};
use es_ir::{ElemType, Frame, Graph, NodeId, PortType, Shape, TimeRef, Unit};

fn scalar_port() -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([1u64]),
        unit: Unit::Dimensionless,
        frame: Frame::World,
        time: TimeRef::Tick,
        image: None,
    }
}

fn task_ir() -> TaskIr {
    let mut graph: TaskGraph = Graph::new(es_ir::task::SCHEMA_VERSION);
    graph.insert(NodeId(0), TaskNode::GetTime { since_reset: false });
    TaskIr {
        schema_version: es_ir::task::SCHEMA_VERSION,
        scene: SceneRef {
            path: "scenes/table.usda".into(),
            scene_hash: [1u8; 32],
            asset_hash: [2u8; 32],
        },
        graph,
        observation_spec: ObservationSpec::default(),
        config: TaskConfig {
            max_episode_steps: 200,
            control_rate_hz: 30.0,
            deterministic: true,
            rng_streams: BTreeSet::from(["reset".to_string()]),
        },
    }
}

fn observation_ir() -> ObservationIr {
    let mut ir = ObservationIr::new(1, [3u8; 32]);
    ir.graph.insert(
        NodeId(0),
        ObservationNode::StateInput {
            source: StableId::from_path("joint_state"),
            io: Io::source(scalar_port()),
        },
    );
    ir
}

fn learning_ir() -> LearningGraph {
    let mut nodes = Graph::new(1);
    nodes.insert(
        NodeId(0),
        LearningNode::StateEncoder {
            inputs: vec![],
            kind: StateEncoderKind::Identity,
            out_dim: 4,
        },
    );
    LearningGraph {
        schema_version: 1,
        inputs: vec![],
        nodes,
        outputs: vec![],
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".into(),
                hash: [4u8; 32],
            },
            contract: PolicyContract {
                inputs: BTreeMap::new(),
                observation_window: 1,
                action_dim: 4,
                horizon: 8,
                execute_chunk: 4,
                replanning_hz: 10.0,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 5.0,
                    deadline_ms: 10.0,
                },
            },
        },
    }
}

fn deployment_ir() -> DeploymentIr {
    DeploymentIr {
        schema_version: es_ir::deployment::SCHEMA_VERSION,
        robot: RobotRef {
            name: "arm".into(),
            target: RobotTarget::Simulated {
                scene: "scenes/table.usda".into(),
            },
            n_joints: 1,
        },
        action: ActionContract {
            space: DepActionSpace::JointPosition,
            dim: 1,
            horizon: 8,
            execute_chunk: 4,
        },
        safety: SafetyEnvelope {
            position: vec![Limit {
                lower: -1.0,
                upper: 1.0,
            }],
            position_soft_margin: vec![0.1],
            velocity_max: vec![1.0],
            acceleration_max: vec![1.0],
            torque_max: vec![1.0],
            jerk_max: None,
            action_rate: RateLimit {
                first_diff_max: vec![0.1],
                second_diff_max: vec![0.1],
            },
            workspace: Workspace::Box {
                min: [-1.0, -1.0, -1.0],
                max: [1.0, 1.0, 1.0],
            },
            ee_velocity_max: 1.0,
            min_self_distance: 0.01,
            min_env_distance: 0.01,
            contact_force_max: 50.0,
        },
        execution: es_ir::deployment::ExecutionMode::RecedingHorizon,
        deadlines: Deadlines {
            observation_age: Micros(1_000),
            inference_budget: Micros(5_000),
            actuation_budget: Micros(1_000),
        },
        watchdogs: WatchdogSet(vec![Watchdog::ChunkUnderrun]),
        fallback: FallbackPolicy::HoldPosition,
        rate: RateSpec {
            control: TickRate::hz(60),
            inference: TickRate::hz(30),
        },
    }
}

fn evaluation_ir() -> EvaluationIr {
    EvaluationIr {
        schema_version: 1,
        task: "task".into(),
        observation: "obs".into(),
        episodes: EpisodeBatch {
            n_episodes: 1,
            seeds: SeedPlan::Base(0),
        },
        suites: vec![],
        metrics: vec![],
        acceptance: vec![],
        augmentation: AugmentationPolicy::default(),
        replay: ReplayPolicy::default(),
    }
}

// --- Round-trip, per IR: `parse(write(ir)) == ir` and the hash is unchanged -------------------

#[test]
fn serial_task_round_trips_and_keeps_hash() {
    let ir = task_ir();
    let toml = serial::task_to_toml(&ir).expect("write");
    let back = serial::task_from_toml(&toml).expect("parse");
    assert_eq!(back, ir);
    assert_eq!(back.task_hash().unwrap(), ir.task_hash().unwrap());
}

#[test]
fn serial_observation_round_trips_and_keeps_hash() {
    let ir = observation_ir();
    let toml = serial::observation_to_toml(&ir).expect("write");
    let back = serial::observation_from_toml(&toml).expect("parse");
    assert_eq!(back, ir);
    assert_eq!(
        back.observation_hash().unwrap(),
        ir.observation_hash().unwrap()
    );
}

#[test]
fn serial_learning_round_trips_and_keeps_hash() {
    let ir = learning_ir();
    let toml = serial::learning_to_toml(&ir).expect("write");
    let back = serial::learning_from_toml(&toml).expect("parse");
    assert_eq!(back, ir);
    assert_eq!(back.learning_hash().unwrap(), ir.learning_hash().unwrap());
}

#[test]
fn serial_deployment_round_trips_and_keeps_hash() {
    let ir = deployment_ir();
    let toml = serial::deployment_to_toml(&ir).expect("write");
    let back = serial::deployment_from_toml(&toml).expect("parse");
    assert_eq!(back, ir);
    assert_eq!(
        back.deployment_hash().unwrap(),
        ir.deployment_hash().unwrap()
    );
}

#[test]
fn serial_evaluation_round_trips_and_keeps_hash() {
    let ir = evaluation_ir();
    let toml = serial::evaluation_to_toml(&ir).expect("write");
    let back = serial::evaluation_from_toml(&toml).expect("parse");
    assert_eq!(back, ir);
    assert_eq!(
        back.evaluation_hash().unwrap(),
        ir.evaluation_hash().unwrap()
    );
}

// --- `.eslayout` never influences the IR ------------------------------------------------------

#[test]
fn serial_layout_never_changes_the_ir_or_its_hash() {
    let ir = task_ir();
    let expected_hash = ir.task_hash().unwrap();

    let mut layout = Layout::default();
    layout.positions.insert(NodeId(0), [123.5, -42.25]);
    layout.positions.insert(NodeId(7), [0.0, 0.0]);
    layout.collapsed.insert(NodeId(7));
    layout.notes.insert(NodeId(0), "arbitrary note".into());

    let (graph_no_layout, sidecar_none) =
        serial::write_esgraph(&AnyIr::Task(ir.clone()), None).unwrap();
    assert!(sidecar_none.is_none());

    let (graph_with_layout, sidecar_some) =
        serial::write_esgraph(&AnyIr::Task(ir.clone()), Some(&layout)).unwrap();
    let sidecar_some = sidecar_some.expect("layout was given");

    // The `.esgraph` body itself does not change when a layout is attached.
    assert_eq!(graph_no_layout, graph_with_layout);

    let (any_a, layout_a) = serial::parse_esgraph(&graph_no_layout, None).unwrap();
    let (any_b, layout_b) = serial::parse_esgraph(&graph_with_layout, Some(&sidecar_some)).unwrap();
    assert!(layout_a.is_none());
    assert_eq!(layout_b.as_ref(), Some(&layout));

    let (AnyIr::Task(ir_a), AnyIr::Task(ir_b)) = (any_a, any_b) else {
        panic!("expected Task IR back");
    };
    assert_eq!(ir_a, ir);
    assert_eq!(ir_b, ir);
    assert_eq!(ir_a.task_hash().unwrap(), expected_hash);
    assert_eq!(ir_b.task_hash().unwrap(), expected_hash);
}

// --- A file whose `kind` does not match the requested IR errors --------------------------------

#[test]
fn serial_mismatched_kind_is_an_error() {
    let dep_toml = serial::deployment_to_toml(&deployment_ir()).unwrap();
    match serial::task_from_toml(&dep_toml) {
        Err(SerialError::KindMismatch { expected, found }) => {
            assert_eq!(expected, IrKind::Task);
            assert_eq!(found, IrKind::Deployment);
        }
        other => panic!("expected KindMismatch, got {other:?}"),
    }
}

// --- Non-finite floats are rejected on write ---------------------------------------------------

#[test]
fn serial_non_finite_float_is_rejected_on_write() {
    let mut ir = deployment_ir();
    ir.safety.contact_force_max = f64::NAN;
    match serial::deployment_to_toml(&ir) {
        Err(SerialError::NonFiniteFloat) => {}
        other => panic!("expected NonFiniteFloat, got {other:?}"),
    }

    let mut ir = deployment_ir();
    ir.safety.ee_velocity_max = f64::INFINITY;
    match serial::deployment_to_toml(&ir) {
        Err(SerialError::NonFiniteFloat) => {}
        other => panic!("expected NonFiniteFloat, got {other:?}"),
    }
}

// --- Generic helpers work directly on the layout type too ---------------------------------------

#[test]
fn serial_layout_round_trips_through_generic_helpers() {
    let mut layout = Layout::default();
    layout.positions.insert(NodeId(1), [1.0, 2.0]);
    layout.notes.insert(NodeId(1), "hello".into());
    layout.collapsed.insert(NodeId(1));

    let toml = serial::write_toml(&layout).unwrap();
    let back: Layout = serial::parse_toml(&toml).unwrap();
    assert_eq!(back, layout);
}
