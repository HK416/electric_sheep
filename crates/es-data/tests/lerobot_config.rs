//! Oracle for the M2 W6 `LeRobot` config conversion (`docs/packets/M2/W6-lerobot-config.md`).
//!
//! Each converted `(ObservationIr, LearningGraph)` pair must validate on its own and must
//! agree across the Observation<->Learning boundary of `es_ir::cross::check`. `cross::check`
//! takes a full five-IR `IrBundle`, so this file builds the smallest Task IR / Deployment IR
//! that closes the loop around a `Converted` result — generic over it, not hand-tuned per
//! fixture, so it also serves as the "does this pair fit inside a real bundle at all" check.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use es_compile::{CpuPlan, PlanMode};
use es_core::time::TickRate;
use es_data::lerobot::Info;
use es_data::lerobot_config::LeRobotPolicyConfig::Act;
use es_data::lerobot_config::{ConfigError, Converted, LeRobotPolicyConfig, Stats};
use es_ir::cross::{self, IrBundle};
use es_ir::deployment::{
    ActionContract, ActionSpace as DepSpace, Deadlines, DeploymentIr, ExecutionMode,
    FallbackPolicy, Limit, Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope,
    WatchdogSet, Workspace,
};
use es_ir::graph::{Graph, NodeId};
use es_ir::image::ChannelFormat;
use es_ir::learning::ActionExecutionMode;
use es_ir::observation::ObservationNode;
use es_ir::task::{
    ActionSpace as TaskSpace, ObsChannel, ObsSource, ObservationSpec, SceneRef, TaskConfig, TaskIr,
    TaskNode,
};
use es_ir::Diagnostic;

const ACT_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/lerobot_config/act_config.json"
));
const DIFFUSION_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/lerobot_config/diffusion_config.json"
));
const STATS_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/lerobot_config/stats.json"
));
const UNKNOWN_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/lerobot_config/unknown_type.json"
));

fn stats() -> Stats {
    serde_json::from_str(STATS_JSON).unwrap()
}

fn errors(diags: &[Diagnostic]) -> Vec<String> {
    diags
        .iter()
        .filter(|d| d.is_error())
        .map(|d| format!("{} {}", d.code, d.message))
        .collect()
}

fn deployed_mode(m: ActionExecutionMode) -> ExecutionMode {
    match m {
        ActionExecutionMode::OpenLoopChunk => ExecutionMode::OpenLoopChunk,
        ActionExecutionMode::RecedingHorizon => ExecutionMode::RecedingHorizon,
        ActionExecutionMode::TemporalEnsemble => ExecutionMode::TemporalEnsemble { decay: 0.01 },
        ActionExecutionMode::RealTimeChunking => ExecutionMode::RealTimeChunking,
    }
}

/// The smallest Task IR / Deployment IR that makes `conv` a valid five-IR bundle (minus
/// Evaluation, which `cross::check` allows to be absent): one `ObservationSpec` channel per
/// Observation IR source node, one `ActionSpec` matching the learning contract, and a
/// Deployment IR whose envelope width and rate agree with it. Generic over `conv` rather than
/// tuned per fixture.
fn task_and_deployment(conv: &Converted) -> (TaskIr, DeploymentIr) {
    let c = &conv.learning.policy.contract;
    let mut channels = BTreeMap::new();
    let mut state_dim = 0u32;
    for node in conv.observation.graph.nodes.values() {
        match node {
            ObservationNode::ImageInput { sensor, io } => {
                channels.entry(sensor.to_string()).or_insert(ObsChannel {
                    source: ObsSource::Sensor {
                        id: *sensor,
                        format: ChannelFormat::Rgb,
                    },
                    ty: io.output.clone(),
                });
            }
            ObservationNode::StateInput { source, io } => {
                let dof = io.output.shape.dims()[0] as u32;
                state_dim = dof;
                channels.entry(source.to_string()).or_insert(ObsChannel {
                    source: ObsSource::JointState { body: *source, dof },
                    ty: io.output.clone(),
                });
            }
            _ => {}
        }
    }

    let control_hz = 100u64;
    let mut graph: Graph<TaskNode> = Graph::new(1);
    graph.insert(
        NodeId(0),
        TaskNode::ActionSpec {
            space: TaskSpace::JointPosition,
            dim: c.action_dim,
            control_rate_hz: control_hz as f32,
        },
    );
    let task = TaskIr {
        schema_version: 1,
        scene: SceneRef {
            path: "scenes/lerobot.usd".to_owned(),
            scene_hash: [1u8; 32],
            asset_hash: [2u8; 32],
        },
        graph,
        observation_spec: ObservationSpec { channels },
        control: None,
        config: TaskConfig {
            max_episode_steps: 400,
            control_rate_hz: control_hz as f32,
            deterministic: true,
            rng_streams: BTreeSet::new(),
        },
    };

    let n = state_dim as usize;
    let rate = RateSpec {
        control: TickRate::hz(control_hz),
        inference: TickRate::hz(10),
    };
    let period = rate.control_period();
    let deployment = DeploymentIr {
        schema_version: 1,
        robot: RobotRef {
            name: "robot".to_owned(),
            target: RobotTarget::Physical {
                driver: "can0".to_owned(),
            },
            n_joints: n,
        },
        action: ActionContract {
            space: DepSpace::JointPosition,
            dim: c.action_dim as usize,
            horizon: c.horizon as usize,
            execute_chunk: c.execute_chunk as usize,
        },
        safety: SafetyEnvelope {
            position: vec![Limit::symmetric(2.8); n],
            position_soft_margin: vec![0.05; n],
            velocity_max: vec![2.0; n],
            acceleration_max: vec![8.0; n],
            torque_max: vec![80.0; n],
            jerk_max: None,
            action_rate: RateLimit {
                first_diff_max: vec![0.1; n],
                second_diff_max: vec![0.05; n],
            },
            workspace: Workspace::Box {
                min: [-0.8, -0.8, 0.0],
                max: [0.8, 0.8, 1.2],
            },
            ee_velocity_max: 1.5,
            min_self_distance: 0.01,
            min_env_distance: 0.02,
            contact_force_max: 40.0,
        },
        execution: deployed_mode(c.execution_mode),
        deadlines: Deadlines {
            observation_age: Micros(period.0 * 4),
            inference_budget: Micros(200_000),
            actuation_budget: Micros(period.0 / 2),
        },
        watchdogs: WatchdogSet(vec![]),
        fallback: FallbackPolicy::HoldPosition,
        rate,
    };
    (task, deployment)
}

fn assert_clean(conv: &Converted) {
    assert!(
        errors(&conv.observation.validate()).is_empty(),
        "{:#?}",
        errors(&conv.observation.validate())
    );
    assert!(
        errors(&conv.learning.validate()).is_empty(),
        "{:#?}",
        errors(&conv.learning.validate())
    );
    let (task, deployment) = task_and_deployment(conv);
    let mut observation = conv.observation.clone();
    observation.task_ref = task.task_hash().unwrap();
    let diags = cross::check(&IrBundle {
        task: &task,
        observation: &observation,
        learning: &conv.learning,
        deployment: &deployment,
        evaluation: None,
    });
    assert!(errors(&diags).is_empty(), "{:#?}", errors(&diags));
}

#[test]
fn act_config_converts_and_validates_clean() {
    let cfg = LeRobotPolicyConfig::parse(ACT_JSON).unwrap();
    let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
    assert_clean(&conv);
}

#[test]
fn diffusion_config_converts_and_validates_clean() {
    let cfg = LeRobotPolicyConfig::parse(DIFFUSION_JSON).unwrap();
    let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
    assert_clean(&conv);
}

/// P-M2-R6: the scheduler configuration has to survive the conversion, or `es-policy` builds
/// the schedule over the wrong grid. `num_inference_steps` (10) and `num_train_timesteps` (100)
/// are different numbers in the fixture precisely so a collapse of the two would show.
#[test]
fn the_diffusion_scheduler_config_passes_through() {
    let cfg = LeRobotPolicyConfig::parse(DIFFUSION_JSON).unwrap();
    let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
    let kind = conv
        .learning
        .nodes
        .nodes
        .values()
        .find_map(|n| match n {
            es_ir::learning::LearningNode::PolicyHead { kind, .. } => Some(*kind),
            _ => None,
        })
        .expect("the converted graph has a head");
    assert_eq!(
        kind,
        es_ir::learning::HeadKind::Diffusion {
            n_steps: 10,
            scheduler: es_ir::learning::DiffusionScheduler::Ddpm,
            num_train_timesteps: 100,
            beta_schedule: es_ir::learning::BetaSchedule::SquaredcosCapV2,
            variance_type: es_ir::learning::VarianceType::FixedSmall,
            prediction_type: es_ir::learning::PredictionType::Epsilon,
            // Absent from the fixture, so these are the `LeRobot` defaults.
            clip_sample: true,
            clip_sample_range: 1.0,
        }
    );
}

#[test]
fn n_obs_steps_above_one_yields_history_and_window() {
    let Act(mut c) = LeRobotPolicyConfig::parse(ACT_JSON).unwrap() else {
        panic!("act_config.json is an ACT config");
    };
    c.n_obs_steps = 2;
    let cfg = Act(c);
    let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
    assert_clean(&conv);
    let window = conv.observation.temporal.window.expect("window is set");
    assert_eq!(window.n_steps, 2);
    assert!(
        !conv.observation.temporal.history.is_empty(),
        "n_obs_steps=2 should register a History depth per stream"
    );
    assert!(conv
        .observation
        .temporal
        .history
        .values()
        .all(|h| h.depth == 2));
}

#[test]
fn diffusion_crop_yields_a_crop_node_with_rescaled_intrinsics() {
    let cfg = LeRobotPolicyConfig::parse(DIFFUSION_JSON).unwrap();
    let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
    let crops: Vec<_> = conv
        .observation
        .graph
        .nodes
        .values()
        .filter_map(|n| match n {
            ObservationNode::Crop {
                mode,
                rescale_intrinsics,
                io,
            } => Some((*mode, *rescale_intrinsics, io.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(crops.len(), 1, "{crops:?}");
    let (_, rescale, io) = &crops[0];
    assert!(*rescale, "INV-14: Crop must rescale the intrinsics");
    let (before, after) = (
        io.inputs[0].image.expect("Crop input carries an ImageSpec"),
        io.output.image.expect("Crop output carries an ImageSpec"),
    );
    assert_ne!((before.width, before.height), (after.width, after.height));
    assert!(
        (before.intrinsics.cx - after.intrinsics.cx).abs() > 1.0,
        "INV-14: Crop must move the principal point"
    );

    // `crop_is_random` is an offset jitter of that centred window: a training_only Augment
    // node wired after the Crop (spec §7.3), which evaluation auto-disables (INV-15).
    let aug = conv
        .observation
        .graph
        .nodes
        .iter()
        .find_map(|(id, n)| match n {
            ObservationNode::Augment { training_only, .. } => Some((*id, *training_only)),
            _ => None,
        })
        .expect("crop_is_random yields an Augment node");
    assert!(aug.1, "INV-15: the node must be training_only");
    assert!(
        conv.observation
            .graph
            .edges
            .iter()
            .any(|e| e.to.node == aug.0),
        "the Augment node must be wired into the chain, not left dangling"
    );
    assert!(
        conv.observation
            .graph
            .edges
            .iter()
            .any(|e| e.from.node == aug.0),
        "the Augment node must feed the rest of the chain"
    );
}

/// P-M2-R2. The whole point of a conversion is that the result runs: every fixture's
/// `Converted::observation` must lower to a `CpuPlan` with no diagnostics — including the
/// `crop_shape` one, whose `Augment` node used to make the graph unplannable.
#[test]
fn every_converted_observation_compiles_to_a_plan() {
    for json in [ACT_JSON, DIFFUSION_JSON] {
        let cfg = LeRobotPolicyConfig::parse(json).unwrap();
        let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
        for mode in [PlanMode::Debug, PlanMode::Release] {
            let plan = CpuPlan::compile(&conv.observation, mode)
                .unwrap_or_else(|d| panic!("{:#?}", errors(&d)));
            assert!(plan.warnings.is_empty(), "{:#?}", plan.warnings);
            assert!(!plan.steps.is_empty());
        }
    }
}

#[test]
fn an_unbounded_state_dim_is_rejected() {
    let json = ACT_JSON.replace(r#""shape": [8] }"#, r#""shape": [70000] }"#);
    let cfg = LeRobotPolicyConfig::parse(&json).unwrap();
    let err = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap_err();
    assert!(matches!(err, ConfigError::OutOfRange(_)), "{err}");
}

#[test]
fn unknown_policy_type_is_an_error() {
    let err = LeRobotPolicyConfig::parse(UNKNOWN_JSON).unwrap_err();
    assert!(
        matches!(&err, ConfigError::Unsupported(t) if t == "smolvla"),
        "{err}"
    );
}

#[test]
fn unknown_field_is_a_warning_not_an_error() {
    let cfg = LeRobotPolicyConfig::parse(ACT_JSON).unwrap();
    let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
    assert!(
        conv.warnings.iter().any(|w| w.contains("optimizer_lr")),
        "{:#?}",
        conv.warnings
    );
}

#[test]
fn conversion_is_deterministic() {
    for json in [ACT_JSON, DIFFUSION_JSON] {
        let cfg = LeRobotPolicyConfig::parse(json).unwrap();
        let a = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
        let b = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
        assert_eq!(
            a.observation.observation_hash().unwrap(),
            b.observation.observation_hash().unwrap()
        );
        assert_eq!(
            a.learning.learning_hash().unwrap(),
            b.learning.learning_hash().unwrap()
        );
    }
}

#[test]
fn missing_dataset_info_warns_about_replanning_hz() {
    let cfg = LeRobotPolicyConfig::parse(ACT_JSON).unwrap();
    let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), None).unwrap();
    assert!(conv.warnings.iter().any(|w| w.contains("replanning_hz")));
    assert!((conv.learning.policy.contract.replanning_hz - 10.0).abs() < 1e-6);
}

#[test]
fn dataset_info_supplies_replanning_hz() {
    let cfg = LeRobotPolicyConfig::parse(ACT_JSON).unwrap();
    let info = Info::new(30.0, BTreeMap::new());
    let conv = es_data::lerobot_config::convert(&cfg, Some(&stats()), Some(&info)).unwrap();
    assert!((conv.learning.policy.contract.replanning_hz - 30.0).abs() < 1e-6);
}
