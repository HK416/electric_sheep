//! Integration tests for the `es` CLI binary (spec 2.5, 10.5, 11.1).
//!
//! The fixture-building functions below (through `hex`) are copied verbatim from
//! `crates/es-ir/tests/cross_fixture.rs` (per the M1 CLI-es work packet: reuse that
//! crate's own cross-IR fixture rather than inventing a second idea of a consistent
//! bundle) with two now-unused imports (`AugmentKind`, `codes`) dropped and the
//! violation-scenario tests left out -- this file only needs `Fixture::new()`.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use es_core::id::StableId;
use es_core::time::TickRate;
use es_ir::cross::{self, IrBundle};
use es_ir::deployment::{
    ActionContract, ActionSpace as DepSpace, Deadlines, DeploymentIr, ExecutionMode,
    FallbackPolicy, Limit, Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope,
    Watchdog, WatchdogSet, Workspace,
};
use es_ir::evaluation::{
    AcceptanceCriterion, Aggregation, AugmentationPolicy, Comparator, Distribution, EpisodeBatch,
    EvaluationIr, MetricSpec, Perturbation, PerturbationKind, PerturbationSuite, Range,
    ReplayPolicy, SeedPlan,
};
use es_ir::graph::{Graph, NodeId, Port, PortRef};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    ShutterModel,
};
use es_ir::learning::{
    ActionExecutionMode, Activation, ArchKind, ChunkBlendPolicy, FusionKind, HeadKind,
    LearningGraph, LearningNode, PolicyContract, PolicyHandle, RuntimeHints, Squash,
    StateEncoderKind, TemporalKind, VisionBackbone, WeightsRef,
};
use es_ir::observation::{
    Io, NormalizeStats, ObservationIr, ObservationNode, ObservationOutput, ResizeFilter,
    TemporalWindow,
};
use es_ir::task::{
    ActionSpace as TaskSpace, JointQuantity, ObsChannel, ObsSource, ObservationSpec, SceneRef,
    TaskConfig, TaskIr, TaskNode,
};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_ir::Diagnostic;

const CAM: &str = "cam_front";
const ROBOT: &str = "franka";
const DOF: u32 = 8;
const CONTROL_HZ: u64 = 100;
const HORIZON: u32 = 50;
const CHUNK: u32 = 20;

fn cam_id() -> StableId {
    StableId::from_path(CAM)
}

fn robot_id() -> StableId {
    StableId::from_path(ROBOT)
}

/// The 640x480 sRGB pinhole camera of the fixture.
fn camera() -> ImageSpec {
    ImageSpec {
        width: 640,
        height: 480,
        channels: ChannelFormat::Rgb,
        dtype: ImageDType::U8,
        color_space: ColorSpace::SRgb,
        camera_model: CameraModel::Pinhole,
        intrinsics: Intrinsics::new(600.0, 600.0, 320.0, 240.0),
        extrinsics: es_math::conventions::Pose::IDENTITY,
        distortion: DistortionModel::None,
        shutter: ShutterModel::Global,
        exposure: Duration::from_micros(500),
        rate_hz: 30.0,
        depth_scale: None,
    }
}

fn image_ty(spec: ImageSpec, unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([3, u64::from(spec.height), u64::from(spec.width)]),
        unit,
        frame: Frame::Camera(cam_id()),
        time: TimeRef::Sensor {
            id: cam_id(),
            align: Align::Hold,
        },
        image: Some(spec),
    }
}

fn state_ty(unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([u64::from(DOF)]),
        unit,
        frame: Frame::Joint(robot_id()),
        time: TimeRef::Tick,
        image: None,
    }
}

/// A tensor port as the policy sees it: inside the network, on the control tick.
fn policy_ty(shape: Shape, unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape,
        unit,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    }
}

fn rgb_unit() -> Unit {
    Unit::Normalized { lo: 0.0, hi: 1.0 }
}

fn action_unit() -> Unit {
    Unit::Normalized { lo: -1.0, hi: 1.0 }
}

fn resized() -> ImageSpec {
    camera().resized(224, 224, true)
}

fn normalized_spec() -> ImageSpec {
    ImageSpec {
        dtype: ImageDType::F32,
        ..resized()
    }
}

// --- Task IR --------------------------------------------------------------------------------

fn task_ir() -> TaskIr {
    let raw = image_ty(camera(), Unit::Pixel);
    let joints = state_ty(JointQuantity::Position.into_unit());
    let mut graph: Graph<TaskNode> = Graph::new(1);
    graph.insert(
        NodeId(0),
        TaskNode::GetSensor {
            sensor: cam_id(),
            ty: raw.clone(),
        },
    );
    graph.insert(
        NodeId(1),
        TaskNode::ObservationSpec {
            channel: "rgb_front".to_owned(),
            ty: raw.clone(),
        },
    );
    graph.insert(
        NodeId(2),
        TaskNode::GetJointState {
            body: robot_id(),
            joints: (0..DOF).map(|i| format!("joint{i}")).collect(),
            quantity: JointQuantity::Position,
        },
    );
    graph.insert(
        NodeId(3),
        TaskNode::ObservationSpec {
            channel: "joint_state".to_owned(),
            ty: joints.clone(),
        },
    );
    graph.insert(
        NodeId(4),
        TaskNode::ActionSpec {
            space: TaskSpace::JointPosition,
            dim: DOF,
            control_rate_hz: CONTROL_HZ as f32,
        },
    );
    graph.connect(NodeId(0), "value", NodeId(1), "value");
    graph.connect(NodeId(2), "value", NodeId(3), "value");

    let channels = BTreeMap::from([
        (
            "rgb_front".to_owned(),
            ObsChannel {
                source: ObsSource::Sensor {
                    id: cam_id(),
                    format: ChannelFormat::Rgb,
                    render: es_ir::task::SensorRender::default(),
                },
                ty: raw,
            },
        ),
        (
            "joint_state".to_owned(),
            ObsChannel {
                source: ObsSource::JointState {
                    body: robot_id(),
                    dof: DOF,
                },
                ty: joints,
            },
        ),
    ]);
    TaskIr {
        schema_version: 1,
        scene: SceneRef {
            path: "scenes/pick_cube.usd".to_owned(),
            scene_hash: [1u8; 32],
            asset_hash: [2u8; 32],
        },
        graph,
        observation_spec: ObservationSpec { channels },
        control: None,
        config: TaskConfig {
            max_episode_steps: 400,
            control_rate_hz: CONTROL_HZ as f32,
            deterministic: true,
            rng_streams: BTreeSet::new(),
        },
    }
}

/// `JointQuantity::unit` is private to `task`; the fixture states the same mapping explicitly.
trait QuantityUnit {
    fn into_unit(self) -> Unit;
}

impl QuantityUnit for JointQuantity {
    fn into_unit(self) -> Unit {
        match self {
            Self::Position => Unit::Angle,
            Self::Velocity => Unit::AngularVelocity,
            Self::Torque => Unit::Torque,
        }
    }
}

// --- Observation IR -------------------------------------------------------------------------

/// `ImageInput -> Resize(224) -> Normalize` and `StateInput -> Normalize`, one frame.
fn observation_ir(task_ref: [u8; 32]) -> ObservationIr {
    let raw = image_ty(camera(), Unit::Pixel);
    let small = image_ty(resized(), Unit::Pixel);
    let rgb = PortType {
        unit: rgb_unit(),
        image: Some(normalized_spec()),
        ..small.clone()
    };
    let joints = state_ty(JointQuantity::Position.into_unit());
    let joints_norm = PortType {
        unit: action_unit(),
        ..joints.clone()
    };

    let mut ir = ObservationIr::new(1, task_ref);
    ir.graph.insert(
        NodeId(0),
        ObservationNode::ImageInput {
            sensor: cam_id(),
            io: Io::source(raw.clone()),
        },
    );
    ir.graph.insert(
        NodeId(1),
        ObservationNode::Resize {
            width: 224,
            height: 224,
            filter: ResizeFilter::Bilinear,
            rescale_intrinsics: true,
            io: Io::unary(raw, small.clone()),
        },
    );
    ir.graph.insert(
        NodeId(2),
        ObservationNode::Normalize {
            stats: NormalizeStats::Range { lo: 0.0, hi: 1.0 },
            io: Io::unary(small, rgb.clone()),
        },
    );
    ir.graph.insert(
        NodeId(3),
        ObservationNode::StateInput {
            source: robot_id(),
            io: Io::source(joints.clone()),
        },
    );
    ir.graph.insert(
        NodeId(4),
        ObservationNode::Normalize {
            stats: NormalizeStats::Range { lo: -1.0, hi: 1.0 },
            io: Io::unary(joints, joints_norm.clone()),
        },
    );
    ir.graph.connect(NodeId(0), "out", NodeId(1), "in0");
    ir.graph.connect(NodeId(1), "out", NodeId(2), "in0");
    ir.graph.connect(NodeId(3), "out", NodeId(4), "in0");
    ir.temporal.window = Some(TemporalWindow {
        n_steps: 1,
        stride: 1,
        align: Align::Hold,
    });
    ir.outputs = BTreeMap::from([
        (
            "rgb_front".to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(2), "out"),
                ty: rgb,
            },
        ),
        (
            "joint_state".to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(4), "out"),
                ty: joints_norm,
            },
        ),
    ]);
    ir
}

// --- Learning IR ----------------------------------------------------------------------------

fn feature(name: &str, dim: u64) -> Port {
    Port::new(name, policy_ty(Shape::new([dim]), Unit::Dimensionless))
}

fn chunk_port(name: &str) -> Port {
    Port::new(
        name,
        policy_ty(
            Shape::new([u64::from(CHUNK), u64::from(DOF)]),
            action_unit(),
        ),
    )
}

/// ACT shaped: two encoders, fusion, temporal encoder, regression head, chunker.
fn learning_graph() -> LearningGraph {
    let rgb = Port::new(
        "rgb_front",
        policy_ty(Shape::new([3, 224, 224]), rgb_unit()),
    );
    let state = Port::new(
        "joint_state",
        policy_ty(Shape::new([u64::from(DOF)]), action_unit()),
    );
    let feat = 512;
    let mut g: Graph<LearningNode> = Graph::new(1);
    g.insert(
        NodeId(0),
        LearningNode::VisionEncoder {
            inputs: vec![rgb.clone()],
            backbone: VisionBackbone::ResNet18,
            // From scratch: `lower_to_torch` refuses `true` rather than ignoring it, and this
            // fixture is lowered (packet M5/V2b).
            pretrained: false,
            frozen: false,
            out_dim: feat,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(1),
        LearningNode::StateEncoder {
            inputs: vec![state.clone()],
            kind: StateEncoderKind::Mlp {
                hidden: vec![256],
                activation: Activation::Relu,
                activate_output: false,
            },
            out_dim: feat,
        },
    );
    g.insert(
        NodeId(2),
        LearningNode::Fusion {
            inputs: vec![
                feature("image", u64::from(feat)),
                feature("state", u64::from(feat)),
            ],
            kind: FusionKind::Concat,
            out_dim: feat,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(3),
        LearningNode::TemporalEncoder {
            inputs: vec![feature("seq", u64::from(feat))],
            kind: TemporalKind::Transformer,
            n_frames: 1,
            out_dim: feat,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(4),
        LearningNode::PolicyHead {
            inputs: vec![feature("feat", u64::from(feat))],
            kind: HeadKind::Regression,
            action_dim: DOF,
            horizon: HORIZON,
            squash: Squash::None,
        },
    );
    g.insert(
        NodeId(5),
        LearningNode::ActionChunker {
            inputs: vec![Port::new(
                "chunk",
                policy_ty(
                    Shape::new([u64::from(HORIZON), u64::from(DOF)]),
                    action_unit(),
                ),
            )],
            horizon: HORIZON,
            execute_chunk: CHUNK,
            replan_hz: 10.0,
            mode: ActionExecutionMode::TemporalEnsemble,
            blend: ChunkBlendPolicy::TemporalEnsemble { weight_decay: 0.01 },
            buffer_chunks: 2,
        },
    );
    g.connect(NodeId(0), "out", NodeId(2), "image");
    g.connect(NodeId(1), "out", NodeId(2), "state");
    g.connect(NodeId(2), "out", NodeId(3), "seq");
    g.connect(NodeId(3), "out", NodeId(4), "feat");
    g.connect(NodeId(4), "chunk", NodeId(5), "chunk");
    g.inputs = vec![
        PortRef::new(NodeId(0), "rgb_front"),
        PortRef::new(NodeId(1), "joint_state"),
    ];
    g.outputs = vec![PortRef::new(NodeId(5), "actions")];

    LearningGraph {
        schema_version: 1,
        inputs: vec![rgb.clone(), state.clone()],
        outputs: vec![chunk_port("actions")],
        nodes: g,
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [7u8; 32],
            },
            contract: PolicyContract {
                inputs: BTreeMap::from([(rgb.name.clone(), rgb), (state.name.clone(), state)]),
                observation_window: 1,
                action_dim: DOF,
                horizon: HORIZON,
                execute_chunk: CHUNK,
                replanning_hz: 10.0,
                execution_mode: ActionExecutionMode::TemporalEnsemble,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 5.0,
                    deadline_ms: 15.0,
                },
            },
        },
    }
}

// --- Deployment IR --------------------------------------------------------------------------

fn deployment_ir() -> DeploymentIr {
    let n = DOF as usize;
    let rate = RateSpec {
        control: TickRate::hz(CONTROL_HZ),
        inference: TickRate::hz(10),
    };
    let period = rate.control_period();
    let deadlines = Deadlines {
        observation_age: Micros(period.0 * 4),
        inference_budget: Micros(period.0 * 2),
        actuation_budget: Micros(period.0 / 2),
    };
    DeploymentIr {
        schema_version: 1,
        robot: RobotRef {
            name: ROBOT.to_owned(),
            target: RobotTarget::Physical {
                driver: "can0".to_owned(),
            },
            n_joints: n,
        },
        action: ActionContract {
            space: DepSpace::JointPosition,
            dim: n,
            horizon: HORIZON as usize,
            execute_chunk: CHUNK as usize,
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
        execution: ExecutionMode::TemporalEnsemble { decay: 0.01 },
        deadlines,
        watchdogs: WatchdogSet(vec![
            Watchdog::InferenceDeadline {
                budget: deadlines.inference_budget,
            },
            Watchdog::ChunkUnderrun,
            Watchdog::StaleObservation {
                max_age: deadlines.observation_age,
            },
            Watchdog::EnvelopeViolationRate {
                window: 200,
                max_frac: 0.05,
            },
        ]),
        fallback: FallbackPolicy::HoldPosition,
        rate,
    }
}

// --- Evaluation IR --------------------------------------------------------------------------

fn evaluation_ir() -> EvaluationIr {
    EvaluationIr {
        schema_version: 1,
        task: String::new(),
        observation: String::new(),
        episodes: EpisodeBatch {
            n_episodes: 50,
            seeds: SeedPlan::Base(7),
        },
        suites: vec![PerturbationSuite {
            name: "lighting".to_owned(),
            perturbations: vec![Perturbation::new(
                PerturbationKind::LightIntensity {
                    range: Range::new(0.5, 1.5),
                    dist: Distribution::Uniform,
                },
                0,
            )],
        }],
        metrics: vec![MetricSpec::SuccessRate, MetricSpec::EnvelopeViolationRate],
        acceptance: vec![AcceptanceCriterion {
            suite: Some("lighting".to_owned()),
            metric: MetricSpec::SuccessRate,
            comparator: Comparator::Ge,
            threshold: 0.8,
            aggregation: Aggregation::Mean,
        }],
        augmentation: AugmentationPolicy::Disabled,
        replay: ReplayPolicy::FailuresFirst,
    }
}

// --- The bundle -----------------------------------------------------------------------------

struct Fixture {
    task: TaskIr,
    observation: ObservationIr,
    learning: LearningGraph,
    deployment: DeploymentIr,
    evaluation: EvaluationIr,
}

impl Fixture {
    /// The five IRs of the valid bundle, already cross-referenced.
    fn new() -> Self {
        let task = task_ir();
        let task_ref = task.task_hash().expect("task hashes");
        let mut f = Self {
            observation: observation_ir(task_ref),
            task,
            learning: learning_graph(),
            deployment: deployment_ir(),
            evaluation: evaluation_ir(),
        };
        f.seal();
        f
    }

    /// Recomputes every reference a mutation may have invalidated, so that a violation test
    /// sees only the diagnostic it is about.
    fn seal(&mut self) {
        self.observation.task_ref = self.task.task_hash().expect("task hashes");
        self.evaluation.task = hex(&self.observation.task_ref);
        self.evaluation.observation =
            hex(&self.observation.observation_hash().expect("obs hashes"));
    }

    fn diags(&self) -> Vec<Diagnostic> {
        cross::check(&IrBundle {
            task: &self.task,
            observation: &self.observation,
            learning: &self.learning,
            deployment: &self.deployment,
            evaluation: Some(&self.evaluation),
        })
    }
}

fn hex(d: &[u8; 32]) -> String {
    use std::fmt::Write;
    d.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

// --- `es` CLI integration tests --------------------------------------------------------------

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_es"))
}

/// A fresh scratch directory for one test, so parallel `cargo test` runs never collide.
fn scratch_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("es-cli-test-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn write(path: &Path, contents: &str) {
    std::fs::write(path, contents).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Writes the fixture's five IRs as the TOML files `ir validate`/`ir check`/`task compile`
/// expect, and returns their paths in `task, observation, learning, deployment, evaluation`
/// order.
fn write_fixture_toml(dir: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf, PathBuf) {
    let f = Fixture::new();
    let task = dir.join("task.toml");
    let observation = dir.join("observation.toml");
    let learning = dir.join("learning.toml");
    let deployment = dir.join("deployment.toml");
    let evaluation = dir.join("evaluation.toml");
    write(
        &task,
        &es_ir::serial::task_to_toml(&f.task).expect("task toml"),
    );
    write(
        &observation,
        &es_ir::serial::observation_to_toml(&f.observation).expect("observation toml"),
    );
    write(
        &learning,
        &es_ir::serial::learning_to_toml(&f.learning).expect("learning toml"),
    );
    write(
        &deployment,
        &es_ir::serial::deployment_to_toml(&f.deployment).expect("deployment toml"),
    );
    write(
        &evaluation,
        &es_ir::serial::evaluation_to_toml(&f.evaluation).expect("evaluation toml"),
    );
    (task, observation, learning, deployment, evaluation)
}

#[test]
fn ir_validate_reports_hash_and_no_errors() {
    let dir = scratch_dir("ir-validate");
    let (task, ..) = write_fixture_toml(&dir);

    let out = bin()
        .args(["ir", "validate", task.to_str().unwrap()])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        out.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("task_hash:"), "{text}");
    assert!(!text.contains("ERROR"), "{text}");
}

#[test]
fn ir_check_reports_full_hash_chain() {
    let dir = scratch_dir("ir-check");
    let (task, observation, learning, deployment, evaluation) = write_fixture_toml(&dir);

    let out = bin()
        .args(["ir", "check"])
        .args([&task, &observation, &learning, &deployment, &evaluation])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        out.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The fixture bundle is already cross-referenced (`Fixture::seal`), so `ir check` must
    // find no violations -- every slot but the run-time three should be a real hash.
    assert!(!text.contains("ERROR"), "{text}");
    for slot in [
        "task",
        "observation",
        "learning",
        "policy",
        "deployment",
        "evaluation",
        "compiler",
    ] {
        assert!(
            text.contains(&format!("{slot}: ")),
            "missing slot '{slot}' in:\n{text}"
        );
    }
    assert!(text.contains("dataset: unset"), "{text}");
    assert!(text.contains("runtime: unset"), "{text}");
    assert!(text.contains("hardware: unset"), "{text}");
}

#[test]
fn task_compile_reports_compiler_hash() {
    let dir = scratch_dir("task-compile");
    let (task, observation, ..) = write_fixture_toml(&dir);

    let out = bin()
        .args(["task", "compile"])
        .args([&task, &observation])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        out.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("compiler_hash:"), "{text}");
    assert!(text.contains("buffers:"), "{text}");
    assert!(text.contains("total:"), "{text}");
}

/// A 32-byte all-zero hash, spelled as the JSON array `serde` expects for `[u8; 32]`.
fn zero_hash_json() -> String {
    format!("[{}]", vec!["0"; 32].join(","))
}

#[test]
fn eval_compare_prints_table_and_flags_significance() {
    let dir = scratch_dir("eval-compare");
    let zero = zero_hash_json();

    // Hand-written EvaluationReport JSON (spec 10.5): one metric both reports carry with
    // per-episode `samples` (the non-standard extension `eval compare` looks for), one only
    // in A, one only in B, to exercise every branch of the comparison table.
    let a_path = dir.join("a.json");
    write(
        &a_path,
        &format!(
            r#"{{
  "schema_version": 1,
  "evaluation_hash": {zero},
  "execution_hash": {zero},
  "cells": [
    {{"suite": "lighting", "metric": "success_rate", "value": {{"scalar": 0.80}}, "n_episodes": 5,
      "samples": [0.75, 0.78, 0.80, 0.82, 0.85]}},
    {{"suite": "lighting", "metric": "collision_rate", "value": {{"scalar": 0.10}}, "n_episodes": 5}}
  ],
  "acceptance": [],
  "passed": true,
  "episodes": []
}}"#
        ),
    );
    let b_path = dir.join("b.json");
    write(
        &b_path,
        &format!(
            r#"{{
  "schema_version": 1,
  "evaluation_hash": {zero},
  "execution_hash": {zero},
  "cells": [
    {{"suite": "lighting", "metric": "success_rate", "value": {{"scalar": 0.95}}, "n_episodes": 5,
      "samples": [0.93, 0.94, 0.95, 0.96, 0.97]}},
    {{"suite": "lighting", "metric": "action_smoothness", "value": {{"scalar": 0.5}}, "n_episodes": 5}}
  ],
  "acceptance": [],
  "passed": true,
  "episodes": []
}}"#
        ),
    );

    let out = bin()
        .args(["eval", "compare"])
        .args([&a_path, &b_path])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        out.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("success_rate"), "{text}");
    assert!(text.contains("collision_rate"), "{text}");
    assert!(text.contains("action_smoothness"), "{text}");
    // Both sides carry `samples` for `success_rate`, so the Welch t-test path must run.
    assert!(text.contains("p="), "{text}");
    // `collision_rate`/`action_smoothness` are each in only one report.
    assert!(text.contains("n/a"), "{text}");
}

#[test]
fn check_deps_always_exits_zero() {
    let out = bin().arg("--check-deps").output().expect("run es");
    assert!(out.status.success());
    assert!(stdout(&out).contains("es --check-deps"));
}

#[test]
fn version_and_help_exit_zero() {
    let out = bin().arg("--version").output().expect("run es");
    assert!(out.status.success());
    assert!(stdout(&out).contains(env!("CARGO_PKG_VERSION")));

    let out = bin().arg("--help").output().expect("run es");
    assert!(out.status.success());
    assert!(stdout(&out).contains("USAGE"));
}

#[test]
fn usage_error_exits_2() {
    // `ir validate` with no files is a usage error, not a runtime one.
    let out = bin().args(["ir", "validate"]).output().expect("run es");
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn missing_file_exits_1() {
    let out = bin()
        .args(["ir", "validate", "does-not-exist.toml"])
        .output()
        .expect("run es");
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn video_usage_errors_exit_2() {
    // No flags at all is a usage error, not a runtime one (see `crates/es/tests/video.rs` for
    // the mosaic's own behavior tests).
    let out = bin().args(["video", "mosaic"]).output().expect("run es");
    assert_eq!(out.status.code(), Some(2), "{}", stdout(&out));
}

#[test]
fn video_mosaic_missing_events_is_exit_1() {
    let dir = scratch_dir("video-missing-events");
    let frames = dir.join("frames");
    let cell = frames.join("00");
    std::fs::create_dir_all(&cell).expect("create cell dir");
    write(
        &cell.join("layout.json"),
        r#"{"shape":[2,2,3],"dtype":"u8"}"#,
    );
    std::fs::write(cell.join("000000.bin"), [0u8; 12]).expect("write frame");
    let report = dir.join("report.json");
    write(
        &report,
        r#"{"cells":[{"metric":"success_rate","value":{"scalar":1.0},"n_episodes":1}]}"#,
    );
    let out_dir = dir.join("out");

    // Every required flag is present and well-formed; only the file --events names is
    // missing, so this is a runtime failure (exit 1), not a usage error (exit 2).
    let out = bin()
        .args(["video", "mosaic"])
        .arg("--frames")
        .arg(&frames)
        .arg("--events")
        .arg(dir.join("does-not-exist.json"))
        .arg("--report")
        .arg(&report)
        .args(["--grid", "1x1"])
        .arg("--out")
        .arg(&out_dir)
        .output()
        .expect("run es");
    assert_eq!(out.status.code(), Some(1), "{}", stdout(&out));
}

/// The shared MJCF fixture (`option cone="elliptic"`), also used by `es-physics-backend`'s own
/// mapping tests: it blocks `mjwarp` (spec 17.2 maps `MJWarp`'s cone to pyramidal only) but not
/// `mujoco-cpu`.
fn pendulum_fixture() -> PathBuf {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/mjcf/pendulum.xml"
    ))
    .to_path_buf()
}

#[test]
fn backend_compare_blocked_backend_exits_1() {
    let scene = pendulum_fixture();
    let out = bin()
        .args(["backend", "compare", "--scene"])
        .arg(&scene)
        .args(["--backends", "mujoco-cpu,mjwarp"])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        text.contains("semantic mapping report - backend `mujoco-cpu`"),
        "{text}"
    );
    assert!(
        text.contains("semantic mapping report - backend `mjwarp`"),
        "{text}"
    );
    // `mjwarp` maps the scene's elliptic friction cone to `blocked` (spec 17.2/14.4)
    // regardless of whether a Python `mujoco_warp` is installed, so it is always skipped.
    assert!(text.contains("backend `mjwarp`: SKIPPED"), "{text}");
    assert_eq!(
        out.status.code(),
        Some(1),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn backend_compare_unblocked_but_unavailable_exits_0() {
    let scene = pendulum_fixture();
    let out = bin()
        .env("ES_PYTHON", "es-no-such-python")
        .args(["backend", "compare", "--scene"])
        .arg(&scene)
        .args(["--backends", "mujoco-cpu"])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        text.contains("semantic mapping report - backend `mujoco-cpu`"),
        "{text}"
    );
    // CI's Python has no `mujoco` package, so the only requested backend is skipped for an
    // environment reason (spec: environment-reason skips still exit 0).
    assert!(text.contains("backend `mujoco-cpu`: SKIPPED"), "{text}");
    assert_eq!(
        out.status.code(),
        Some(0),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn backend_compare_newton_is_skipped_or_runs() {
    // arm2.xml has no explicit cone (default pyramidal) and no <actuator>/<sensor>, so it is
    // never blocked by Newton's mapping report (spec 14.4) -- unlike `pendulum_fixture`, which
    // is. That isolates the availability check this test targets.
    let scene = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/mjcf/arm2.xml"
    ));
    let out = bin()
        .env("ES_PYTHON", "es-no-such-python")
        .args(["backend", "compare", "--scene"])
        .arg(scene)
        .args(["--backends", "newton"])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        text.contains("semantic mapping report - backend `newton`"),
        "{text}"
    );
    assert!(text.contains("backend `newton`: SKIPPED"), "{text}");
}

#[test]
fn backend_compare_unknown_backend_exits_2() {
    let scene = pendulum_fixture();
    let out = bin()
        .args(["backend", "compare", "--scene"])
        .arg(&scene)
        .args(["--backends", "not-a-real-backend"])
        .output()
        .expect("run es");
    assert_eq!(out.status.code(), Some(2));
}

// --- `es bench` (spec 12.4, 20.2, 20.3) -------------------------------------------------------

#[test]
fn bench_without_memory_report_prints_nine_metric_stub() {
    let out = bin().args(["bench"]).output().expect("run es");
    let text = stdout(&out);
    assert!(out.status.success(), "stdout:\n{text}");
    for metric in [
        "physics_steps_per_sec",
        "camera_frames_per_sec",
        "pixels_per_sec",
        "observation_gb_per_sec",
        "policy_inferences_per_sec",
        "actions_per_sec",
        "end_to_end_latency",
        "gpu_memory_peak",
        "chunk_underrun_rate",
    ] {
        assert!(
            text.contains(metric),
            "missing metric '{metric}' in:\n{text}"
        );
    }
    assert!(text.contains("unmeasured"), "{text}");
    assert!(text.contains("Target / Status: unverified"), "{text}");
}

#[test]
fn bench_memory_report_prints_table_and_exits_zero_when_balanced() {
    let dir = scratch_dir("bench-memory-report");
    let (_, observation, learning, ..) = write_fixture_toml(&dir);

    let out = bin()
        .args(["bench", "--memory-report", "--obs"])
        .arg(&observation)
        .arg("--learning")
        .arg(&learning)
        .args([
            "--sim-envs",
            "8",
            "--obs-envs",
            "4",
            "--views",
            "1",
            "--inference-batch",
            "16",
        ])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        out.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    for item in [
        "physics_state",
        "render_tile_atlas",
        "observation_intermediates",
        "history_buffers",
        "policy_weights",
        "inference_activations",
        "chunk_buffers",
        "total",
        "per domain:",
    ] {
        assert!(text.contains(item), "missing '{item}' in:\n{text}");
    }
    assert!(text.contains("GiB"), "{text}");
    assert!(text.contains("no spec 20.3 rule violations"), "{text}");
}

#[test]
fn bench_memory_report_obs_over_sim_envs_exits_1() {
    let dir = scratch_dir("bench-memory-report-violation");
    let (_, observation, learning, ..) = write_fixture_toml(&dir);

    let out = bin()
        .args(["bench", "--memory-report", "--obs"])
        .arg(&observation)
        .arg("--learning")
        .arg(&learning)
        .args([
            "--sim-envs",
            "2",
            "--obs-envs",
            "4",
            "--views",
            "1",
            "--inference-batch",
            "4",
        ])
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert_eq!(
        out.status.code(),
        Some(1),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("obs_batch_le_sim_batch"), "{text}");
}

#[test]
fn bench_memory_report_missing_obs_is_usage_error() {
    let out = bin()
        .args([
            "bench",
            "--memory-report",
            "--sim-envs",
            "1",
            "--obs-envs",
            "1",
            "--views",
            "1",
            "--inference-batch",
            "1",
        ])
        .output()
        .expect("run es");
    assert_eq!(out.status.code(), Some(2));
}

// --- `es eval run` / `es import lerobot-config` (M2 packet CLI-eval-run-import) --------------

/// Bytes that stand in for a checkpoint, exactly as
/// `crates/es-runtime-embedded/tests/embedded.rs` does it: nothing parses them, the bundle
/// only has to carry them intact and hash-match.
const WEIGHTS: &[u8] = b"not really safetensors, but hashed like it";

/// The fixture's inference budget is two control periods, which only admits a replan every
/// other tick; widening it (INV-12: widen the envelope, never disable a watchdog) lets the
/// 10 Hz replan cadence the Learning IR declares actually fit -- copied from
/// `crates/es-runtime-embedded/tests/embedded.rs`.
fn widen_inference_budget(dep: &mut DeploymentIr) {
    let period = dep.rate.control_period().0;
    dep.deadlines.inference_budget = Micros(period * 12);
    dep.deadlines.observation_age = Micros(period * 16);
    for w in &mut dep.watchdogs.0 {
        match w {
            Watchdog::InferenceDeadline { budget } => *budget = dep.deadlines.inference_budget,
            Watchdog::StaleObservation { max_age } => *max_age = dep.deadlines.observation_age,
            _ => {}
        }
    }
}

/// The fixture, with the weights reference pointing at [`WEIGHTS`] and a deployment whose
/// timing admits the declared replan cadence.
fn deployable_fixture() -> Fixture {
    let mut f = Fixture::new();
    f.learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(WEIGHTS).as_bytes(),
    };
    widen_inference_budget(&mut f.deployment);
    assert!(f.diags().is_empty(), "fixture must stay consistent");
    f
}

fn build_policy_bundle(f: &Fixture) -> Vec<u8> {
    es_compile::PolicyBundle::build(&f.task, &f.observation, &f.learning, &f.deployment, WEIGHTS)
        .expect("the fixture builds a policy bundle")
}

/// CI's PR job (spec 1.4) has neither a `mujoco` Python nor a `torch` Python, so `eval run`
/// must refuse to fake a run and exit with the distinct SKIPPED code instead of 0 or 1.
#[test]
fn eval_run_skips_when_backend_or_runtime_unavailable() {
    let dir = scratch_dir("eval-run-skip");
    let f = deployable_fixture();

    let policy_path = dir.join("policy.esb");
    std::fs::write(&policy_path, build_policy_bundle(&f)).expect("write policy.esb");
    let config_path = dir.join("eval.toml");
    write(
        &config_path,
        &es_ir::serial::evaluation_to_toml(&f.evaluation).expect("evaluation toml"),
    );

    let out = bin()
        .env("ES_PYTHON", "es-no-such-python")
        .args(["eval", "run", "--config"])
        .arg(&config_path)
        .arg("--policy")
        .arg(&policy_path)
        // The backend/runtime availability check happens before the scene is ever read
        // (never fake a run, spec 1.4), so a scene that does not exist is fine here.
        .arg("--scene")
        .arg("does-not-exist.xml")
        .arg("--out")
        .arg(dir.join("out"))
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert_eq!(
        out.status.code(),
        Some(3),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("SKIPPED"), "{text}");
}

/// Packet M5/V5. `--jobs 0` is "run no cell and report on it": a usage error (exit 2) raised
/// while parsing, before a bundle, a scene or a Python interpreter is touched -- which is why
/// this runs in the PR tier where neither `mujoco` nor `torch` exists. The same check covers
/// the worker-mode flags, which must never write a report over a subset of the suites.
#[test]
fn eval_run_jobs_zero_and_the_worker_flags_are_usage_errors() {
    let dir = scratch_dir("eval-run-jobs");
    for (flags, wanted) in [
        (vec!["--jobs", "0"], "--jobs 0 runs no cell"),
        (vec!["--jobs", "x"], "is not a number"),
        (vec!["--shard", "0/4"], "go together"),
        (
            vec!["--shard", "4/4", "--shard-out", "s.json"],
            "index below it",
        ),
        (
            vec!["--shard", "0/4", "--shard-out", "s.json", "--jobs", "2"],
            "never a parent",
        ),
    ] {
        let out = bin()
            .env("ES_PYTHON", "es-no-such-python")
            .args(["eval", "run", "--config"])
            .arg(dir.join("nothing.toml"))
            .arg("--policy")
            .arg(dir.join("nothing.esb"))
            .arg("--scene")
            .arg("does-not-exist.xml")
            .arg("--out")
            .arg(dir.join("out"))
            .args(&flags)
            .output()
            .expect("run es eval run");
        let text = format!("{}{}", stdout(&out), String::from_utf8_lossy(&out.stderr));
        assert_eq!(out.status.code(), Some(2), "{flags:?}: {text}");
        assert!(text.contains(wanted), "{flags:?}: {text}");
        // Refused before anything was opened, let alone written.
        assert!(!dir.join("out").exists(), "{flags:?}: {text}");
    }
    println!("RAN eval_run_jobs_zero_and_the_worker_flags_are_usage_errors");
}

/// `es import lerobot-config` on the shared M2 W6 fixture (spec 14.4), round-tripping the
/// two IRs it writes through `es ir validate`.
#[test]
fn import_lerobot_config_round_trips_through_ir_validate() {
    let dir = scratch_dir("import-lerobot");
    let config = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/lerobot_config/act_config.json"
    );
    let out_dir = dir.join("out");

    let out = bin()
        .args(["import", "lerobot-config", "--config", config, "--out"])
        .arg(&out_dir)
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        out.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("observation_hash:"), "{text}");
    assert!(text.contains("learning_hash:"), "{text}");

    let observation = out_dir.join("observation.toml");
    let learning = out_dir.join("learning.toml");
    assert!(observation.is_file());
    assert!(learning.is_file());

    let validate = bin()
        .args(["ir", "validate"])
        .args([&observation, &learning])
        .output()
        .expect("run es");
    let vtext = stdout(&validate);
    assert!(
        validate.status.success(),
        "stdout:\n{vtext}\nstderr:\n{}",
        String::from_utf8_lossy(&validate.stderr)
    );
    assert!(!vtext.contains("ERROR"), "{vtext}");
}

// --- `es evidence` (spec 27.1, spec 28.7 gate 16) --------------------------------------------

use ed25519_dalek::SigningKey;
use es_compile::bundle::{
    self, BundleManifest, EVALUATION_LOCK, POLICY_BUNDLE, REPORT_JSON, SAFETY_CASE,
};
use es_compile::PolicyBundle;
use es_eval::evidence::{
    report_entries, Claim, Evidence, EvidenceBundle, EvidenceKind, Requirement, SafetyCase,
    Severity, SignatureStatus, EVID_ENTRY_HASH, EVID_NOT_CANONICAL,
};
use es_eval::{BackendCaps, EvaluationLock};
use es_ir::evaluation::{
    AcceptanceResult, CellResult, EvaluationReport, MetricValue as EvalMetricValue,
};
use es_ir::hash::{canonical_hash, ChangedComponent, DatasetHash, HardwareCapability, HashChain};

/// [`deployable_fixture`] with one envelope knob the `--against` test can tighten, which is
/// the smallest real change that moves `deployment_hash` (spec 5.3).
fn evidence_fixture(contact_force_max: f64) -> Fixture {
    let mut f = deployable_fixture();
    f.deployment.safety.contact_force_max = contact_force_max;
    f
}

/// Everything one `evidence.esb` is built from, kept so a test can rebuild it after tampering.
struct EvidenceCase {
    policy: Vec<u8>,
    chain: HashChain,
    runs: Vec<(EvaluationReport, EvaluationLock)>,
    case: SafetyCase,
    bytes: Vec<u8>,
}

/// One evaluation run's artifacts (spec 10.5), of the execution `chain` describes.
fn run_artifacts(f: &Fixture, chain: &HashChain) -> (EvaluationReport, EvaluationLock) {
    let evaluation_hash = f.evaluation.evaluation_hash().expect("evaluation hashes");
    let execution_hash = chain.execution_hash();
    let report = EvaluationReport {
        schema_version: 1,
        evaluation_hash,
        execution_hash,
        cells: vec![CellResult {
            suite: "lighting".to_owned(),
            metric: MetricSpec::SuccessRate,
            value: EvalMetricValue::Scalar(0.91),
            n_episodes: 50,
        }],
        acceptance: vec![AcceptanceResult::Determined {
            criterion: f.evaluation.acceptance[0].clone(),
            observed: 0.91,
            passed: true,
        }],
        passed: true,
        episodes: Vec::new(),
    };
    let lock = EvaluationLock {
        schema_version: 1,
        evaluation_hash: hex(&evaluation_hash),
        execution_hash: hex(&execution_hash),
        seeds: vec![7],
        backend: BackendCaps {
            name: "mujoco-cpu".to_owned(),
            determinism: "bitwise".to_owned(),
            float: "f64".to_owned(),
            max_envs: 1,
            gpu_resident: false,
            supports_reset_subset: false,
            supports_state_get_set: true,
            quirks: Vec::new(),
        },
        created: 0,
    };
    (report, lock)
}

/// A complete Safety Case over `runs`: one requirement evidenced by the report, one by the
/// lock, plus the claim a human would write for the first.
fn safety_case(
    runs: &[(EvaluationReport, EvaluationLock)],
    execution_hash: [u8; 32],
) -> SafetyCase {
    let entries = report_entries(runs).expect("report entries");
    let evidence = |id: &str, kind: EvidenceKind, entry: String| Evidence {
        id: id.to_owned(),
        kind,
        hash: *blake3::hash(&entries[&entry]).as_bytes(),
        entry,
        execution_hash,
    };
    let requirement = |id: &str, text: &str| Requirement {
        id: id.to_owned(),
        text: text.to_owned(),
        source: "(EU) 2023/1230 Annex III".to_owned(),
        severity: Severity::Critical,
    };
    SafetyCase {
        schema_version: 1,
        requirements: vec![
            requirement(
                "REQ-07",
                "the end effector stays inside the declared envelope",
            ),
            requirement(
                "REQ-11",
                "the evaluated conditions are recorded and reproducible",
            ),
        ],
        claims: vec![Claim {
            id: "CLM-01".to_owned(),
            text: "the envelope bounds it and the lighting suite measured it".to_owned(),
            requirement: "REQ-07".to_owned(),
        }],
        evidence: vec![
            evidence(
                "EV-report",
                EvidenceKind::EvalReport,
                bundle::report_entry(0, REPORT_JSON),
            ),
            evidence(
                "EV-lock",
                EvidenceKind::Lock,
                bundle::report_entry(0, EVALUATION_LOCK),
            ),
        ],
        traceability: BTreeMap::from([
            ("REQ-07".to_owned(), vec!["EV-report".to_owned()]),
            ("REQ-11".to_owned(), vec!["EV-lock".to_owned()]),
        ]),
    }
}

fn build_evidence(contact_force_max: f64) -> EvidenceCase {
    let f = evidence_fixture(contact_force_max);
    let policy = build_policy_bundle(&f);
    let m = PolicyBundle::open(&policy)
        .expect("policy bundle opens")
        .manifest
        .hashes;
    let slot = |v: Option<[u8; 32]>| v.expect("a policy bundle fills this slot");
    let chain = HashChain {
        asset: vec![f.task.scene.asset_hash],
        scene: f.task.scene.scene_hash,
        task_graph: canonical_hash(&f.task.graph).expect("task graph hashes"),
        task: slot(m.task),
        observation: slot(m.observation),
        learning: slot(m.learning),
        policy: slot(m.policy),
        dataset: DatasetHash {
            content: [1; 32],
            schema: [2; 32],
            split: [3; 32],
        },
        deployment: slot(m.deployment),
        evaluation: Some(f.evaluation.evaluation_hash().expect("evaluation hashes")),
        compiler: slot(m.compiler),
        runtime: [9; 32],
        hardware: HardwareCapability([8; 32]),
    };
    let runs = vec![run_artifacts(&f, &chain)];
    let case = safety_case(&runs, chain.execution_hash());
    let bytes = EvidenceBundle::build(&policy, &chain, &runs, &case, &BTreeMap::new())
        .expect("evidence bundle builds");
    EvidenceCase {
        policy,
        chain,
        runs,
        case,
        bytes,
    }
}

/// Rewrite one entry of a sealed bundle and re-seal the container, so that the container's own
/// blake3 check passes and only the Safety Case's recorded hash can catch the change.
fn rewrite_entry(bytes: &[u8], entry: &str, payload: Vec<u8>) -> Vec<u8> {
    let raw = bundle::read(bytes).expect("reads");
    let mut entries = raw.entries;
    entries.insert(entry.to_owned(), payload);
    bundle::write(&raw.manifest, &entries).expect("writes")
}

#[test]
fn evidence_bundle_round_trips() {
    let built = build_evidence(40.0);
    let report = EvidenceBundle::verify(&built.bytes, None, &[]).expect("verifies");
    assert!(
        report.ok(),
        "diagnostics: {:?}\ncoverage: {:?}",
        report.diagnostics,
        report.coverage
    );
    assert_eq!(report.execution_hash, built.chain.execution_hash());
    assert!(report.coverage.iter().all(|c| c.covered));
    assert!(report.coverage.iter().all(|c| c.stale.is_empty()));
    assert!(report.revalidation.is_empty());
    // The policy bundle is embedded whole and still opens on its own (spec 9.6).
    let opened = EvidenceBundle::open(&built.bytes).expect("opens");
    assert_eq!(opened.entries[POLICY_BUNDLE], built.policy);
    assert_eq!(opened.case, built.case);
    assert_eq!(opened.chain, built.chain);
    // Deterministic: same inputs, same bytes.
    let again = EvidenceBundle::build(
        &built.policy,
        &built.chain,
        &built.runs,
        &built.case,
        &BTreeMap::new(),
    )
    .expect("builds");
    assert_eq!(again, built.bytes);
}

#[test]
fn evidence_verify_catches_a_tampered_report() {
    let built = build_evidence(40.0);
    let entry = bundle::report_entry(0, REPORT_JSON);
    // Still valid JSON and still the same report -- only the bytes differ, which is exactly
    // what the recorded blake3 exists to catch.
    let mut payload = bundle::read(&built.bytes).expect("reads").entries[&entry].clone();
    payload.push(b' ');
    let tampered = rewrite_entry(&built.bytes, &entry, payload);

    let report = EvidenceBundle::verify(&tampered, None, &[]).expect("verifies");
    assert!(!report.ok(), "a rewritten report entry must not verify");
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code.as_str() == EVID_ENTRY_HASH),
        "{:?}",
        report.diagnostics
    );
    let covered = report.coverage.iter().filter(|c| c.covered).count();
    assert_eq!(covered, 1, "only the lock's requirement stays covered");
}

#[test]
fn evidence_verify_fails_an_uncovered_requirement() {
    let built = build_evidence(40.0);
    // `build` refuses a case with a missing link, so the link is cut in the sealed bundle --
    // the same thing an editor downstream of the build would do.
    let mut case = built.case.clone();
    case.traceability.remove("REQ-07");
    let mut text = serde_json::to_string_pretty(&case).expect("serializes");
    text.push('\n');
    let cut = rewrite_entry(&built.bytes, SAFETY_CASE, text.into_bytes());

    let report = EvidenceBundle::verify(&cut, None, &[]).expect("verifies");
    assert!(!report.ok());
    let req = report
        .coverage
        .iter()
        .find(|c| c.requirement == "REQ-07")
        .expect("REQ-07 is still a requirement");
    assert!(!req.covered);
    assert!(req.evidence.is_empty());

    let dir = scratch_dir("evidence-uncovered");
    let path = dir.join("evidence.esb");
    std::fs::write(&path, &cut).expect("write bundle");
    let out = bin()
        .args(["evidence", "verify", path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert_eq!(out.status.code(), Some(1), "{}", stdout(&out));
    assert!(stdout(&out).contains("UNCOVERED"), "{}", stdout(&out));
}

#[test]
fn evidence_verify_against_lists_revalidation() {
    let a = build_evidence(40.0);
    let b = build_evidence(35.0); // a tighter envelope: `deployment_hash` moves
    assert_ne!(a.chain.deployment, b.chain.deployment);

    let report = EvidenceBundle::verify(&a.bytes, Some(&b.bytes), &[]).expect("verifies");
    // `--against` is advisory: a differing predecessor is not a defect of this bundle.
    assert!(report.ok(), "{:?}", report.diagnostics);
    let deployment: Vec<_> = report
        .revalidation
        .iter()
        .filter(|(c, _)| *c == ChangedComponent::Deployment)
        .collect();
    assert_eq!(deployment.len(), 1, "{:?}", report.revalidation);
    assert_eq!(
        deployment[0].1,
        vec!["EV-report".to_owned(), "EV-lock".to_owned()],
        "a changed envelope invalidates both run-produced kinds (spec 27.1)"
    );
    // Nothing else moved: the two bundles differ in the envelope alone.
    assert!(
        report
            .revalidation
            .iter()
            .all(|(c, _)| *c == ChangedComponent::Deployment),
        "{:?}",
        report.revalidation
    );
}

#[test]
fn evidence_build_and_verify_cli_round_trip() {
    let dir = scratch_dir("evidence-cli");
    let built = build_evidence(40.0);

    let policy = dir.join("policy.esb");
    std::fs::write(&policy, &built.policy).expect("write policy");
    let chain = dir.join("chain.json");
    write(
        &chain,
        &serde_json::to_string_pretty(&built.chain).expect("chain json"),
    );
    let case = dir.join("case.json");
    write(
        &case,
        &serde_json::to_string_pretty(&built.case).expect("case json"),
    );
    let run_dir = dir.join("run0");
    es_eval::write_artifacts(&built.runs[0].0, &built.runs[0].1, &run_dir).expect("artifacts");
    let out_path = dir.join("evidence.esb");

    let out = bin()
        .args(["evidence", "build"])
        .args(["--policy", policy.to_str().unwrap()])
        .args(["--chain", chain.to_str().unwrap()])
        .args(["--report", run_dir.to_str().unwrap()])
        .args(["--case", case.to_str().unwrap()])
        .args(["--out", out_path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        stdout(&out),
        String::from_utf8_lossy(&out.stderr)
    );
    // The CLI re-serializes the two artifacts it read from disk; they must be the same bytes
    // the library sealed, or the case's recorded hashes would not match.
    assert_eq!(std::fs::read(&out_path).expect("read bundle"), built.bytes);

    let verify = bin()
        .args(["evidence", "verify", out_path.to_str().unwrap()])
        .output()
        .expect("run es");
    let text = stdout(&verify);
    assert!(verify.status.success(), "{text}");
    assert!(text.contains("REQ-07"), "{text}");
    assert!(text.contains("signature:      absent"), "{text}");
    assert!(!text.contains("UNCOVERED"), "{text}");

    let json = bin()
        .args(["evidence", "verify", out_path.to_str().unwrap(), "--json"])
        .output()
        .expect("run es");
    assert!(json.status.success(), "{}", stdout(&json));
    let parsed: serde_json::Value = serde_json::from_str(&stdout(&json)).expect("json");
    assert_eq!(parsed["coverage"].as_array().expect("coverage").len(), 2);
}

// --- `es evidence sign`/`verify --trust` (spec 25.1) ---------------------------------------

#[test]
fn evidence_sign_then_verify_is_valid_against_the_trusted_key() {
    let built = build_evidence(40.0);
    let signing_key = SigningKey::from_bytes(&[7u8; 32]);
    let signed = EvidenceBundle::sign(&built.bytes, &signing_key).expect("signs");

    let trusted = [signing_key.verifying_key()];
    let report = EvidenceBundle::verify(&signed, None, &trusted).expect("verifies");
    assert!(report.ok(), "{:?}", report.diagnostics);
    assert_eq!(
        report.signature,
        SignatureStatus::Valid(signing_key.verifying_key().to_bytes())
    );

    // Untrusted (empty trust list): the same signature checks out cryptographically but is
    // not `Valid` until the caller says so.
    let untrusted_report = EvidenceBundle::verify(&signed, None, &[]).expect("verifies");
    assert_eq!(
        untrusted_report.signature,
        SignatureStatus::UntrustedKey(signing_key.verifying_key().to_bytes())
    );
    assert!(
        untrusted_report.ok(),
        "an untrusted signature is not a gate-16 defect"
    );
}

#[test]
fn evidence_verify_flags_a_tampered_signed_bundle_invalid() {
    let built = build_evidence(40.0);
    let signing_key = SigningKey::from_bytes(&[7u8; 32]);
    let signed = EvidenceBundle::sign(&built.bytes, &signing_key).expect("signs");

    let entry = bundle::report_entry(0, REPORT_JSON);
    let mut payload = bundle::read(&signed).expect("reads").entries[&entry].clone();
    payload.push(b' ');
    let tampered = rewrite_entry(&signed, &entry, payload);

    let report =
        EvidenceBundle::verify(&tampered, None, &[signing_key.verifying_key()]).expect("verifies");
    assert_eq!(report.signature, SignatureStatus::Invalid);
}

#[test]
fn evidence_verify_reports_wrong_trusted_key_as_untrusted() {
    let built = build_evidence(40.0);
    let signing_key = SigningKey::from_bytes(&[7u8; 32]);
    let other_key = SigningKey::from_bytes(&[9u8; 32]);
    let signed = EvidenceBundle::sign(&built.bytes, &signing_key).expect("signs");

    let report =
        EvidenceBundle::verify(&signed, None, &[other_key.verifying_key()]).expect("verifies");
    assert_eq!(
        report.signature,
        SignatureStatus::UntrustedKey(signing_key.verifying_key().to_bytes())
    );
}

#[test]
fn evidence_verify_a_schema_v1_bundle_reports_absent_signature() {
    let built = build_evidence(40.0);
    // `build` always writes the current schema version; force it back to 1 (what every
    // evidence.esb written before this packet looks like -- no signature fields at all) and
    // confirm it still opens and verifies (spec 25.3: old versions stay readable).
    let raw = bundle::read(&built.bytes).expect("reads");
    let v1 = BundleManifest {
        schema_version: 1,
        ..raw.manifest
    };
    let v1_bytes = bundle::write(&v1, &raw.entries).expect("writes");

    let report = EvidenceBundle::verify(&v1_bytes, None, &[]).expect("verifies");
    assert!(report.ok(), "{:?}", report.diagnostics);
    assert_eq!(report.signature, SignatureStatus::Absent);
}

#[test]
fn evidence_verify_catches_a_reindented_report() {
    let built = build_evidence(40.0);
    let entry = bundle::report_entry(0, REPORT_JSON);
    let raw = bundle::read(&built.bytes).expect("reads");
    // Same JSON value, compact instead of `build`'s canonical sorted-key pretty-print: proves
    // the canonical-form check is about exact bytes, not just parseability.
    let value: serde_json::Value = serde_json::from_slice(&raw.entries[&entry]).expect("parses");
    let reindented = serde_json::to_string(&value)
        .expect("compact json")
        .into_bytes();
    let bad = rewrite_entry(&built.bytes, &entry, reindented);

    let report = EvidenceBundle::verify(&bad, None, &[]).expect("verifies");
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code.as_str() == EVID_NOT_CANONICAL),
        "{:?}",
        report.diagnostics
    );
}

#[test]
fn evidence_keygen_sign_and_verify_trust_cli_round_trip() {
    let dir = scratch_dir("evidence-sign-cli");
    let built = build_evidence(40.0);
    let bundle_path = dir.join("evidence.esb");
    std::fs::write(&bundle_path, &built.bytes).expect("write bundle");

    let key_path = dir.join("key.eskey");
    let keygen_out = bin()
        .args(["evidence", "keygen", "--out", key_path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert!(keygen_out.status.success(), "{}", stdout(&keygen_out));
    assert!(
        stdout(&keygen_out).contains("public key:"),
        "{}",
        stdout(&keygen_out)
    );
    assert_eq!(
        std::fs::metadata(&key_path).expect("key file").len(),
        32,
        "a signing seed is exactly 32 bytes"
    );

    let signed_path = dir.join("signed.esb");
    let sign_out = bin()
        .args(["evidence", "sign"])
        .args(["--key", key_path.to_str().unwrap()])
        .args(["--in", bundle_path.to_str().unwrap()])
        .args(["--out", signed_path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert!(sign_out.status.success(), "{}", stdout(&sign_out));
    let sign_text = stdout(&sign_out);
    let pub_hex = sign_text
        .lines()
        .find_map(|l| l.strip_prefix("signer public key: "))
        .expect("sign prints the signer's public key")
        .to_owned();
    let pub_path = dir.join("pub.hex");
    write(&pub_path, &pub_hex);

    // No `--trust`: the signature checks out but nothing says to trust that key.
    let untrusted = bin()
        .args(["evidence", "verify", signed_path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert!(untrusted.status.success(), "{}", stdout(&untrusted));
    assert!(
        stdout(&untrusted).contains("untrusted key"),
        "{}",
        stdout(&untrusted)
    );

    // `--trust pub.hex`: now it is valid.
    let trusted = bin()
        .args(["evidence", "verify", signed_path.to_str().unwrap()])
        .args(["--trust", pub_path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert!(trusted.status.success(), "{}", stdout(&trusted));
    assert!(
        stdout(&trusted).contains("signature:      valid"),
        "{}",
        stdout(&trusted)
    );

    // `--require-signature` with no trusted key fails even though gate 16 passes.
    let required_untrusted = bin()
        .args([
            "evidence",
            "verify",
            signed_path.to_str().unwrap(),
            "--require-signature",
        ])
        .output()
        .expect("run es");
    assert_eq!(
        required_untrusted.status.code(),
        Some(1),
        "{}",
        stdout(&required_untrusted)
    );

    // `--require-signature` with the right `--trust` succeeds.
    let required_trusted = bin()
        .args(["evidence", "verify", signed_path.to_str().unwrap()])
        .args(["--trust", pub_path.to_str().unwrap(), "--require-signature"])
        .output()
        .expect("run es");
    assert!(
        required_trusted.status.success(),
        "{}",
        stdout(&required_trusted)
    );
}

#[test]
fn evidence_replay_dry_run_prints_the_plan_and_is_skipped_otherwise() {
    let dir = scratch_dir("evidence-replay");
    let built = build_evidence(40.0);
    let path = dir.join("evidence.esb");
    std::fs::write(&path, &built.bytes).expect("write bundle");

    let skipped = bin()
        .args(["evidence", "replay", path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert_eq!(skipped.status.code(), Some(3), "{}", stdout(&skipped));
    assert!(stdout(&skipped).contains("SKIPPED"), "{}", stdout(&skipped));

    let dry = bin()
        .args(["evidence", "replay", "--dry-run", path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert!(dry.status.success(), "{}", stdout(&dry));
    let text = stdout(&dry);
    assert!(text.contains("REPORT"), "{text}");
    assert!(text.contains("1 plan(s) printed"), "{text}");
}

// --- `es gap` (spec 24.3, spec 28.7 gate 15) -----------------------------------------------

use es_data::{Column, Dtype, Episode, FeatureSpec, Info, LeRobotWriter};

/// A minimal `LeRobot` dataset: one episode, one scalar `observation.state` feature, one
/// `action` feature, plus `success` and `action_source` bookkeeping columns (the design doc's
/// episode-level convention). `shift` is added to every `action` value so the sim/real
/// fixtures can disagree on exactly one channel.
fn write_gap_fixture(root: &Path, shift: f64) {
    let mut features = BTreeMap::new();
    features.insert(
        "observation.state".to_owned(),
        FeatureSpec::new(Dtype::Float32, [1u64]),
    );
    features.insert(
        "action".to_owned(),
        FeatureSpec::new(Dtype::Float32, [1u64]),
    );
    features.insert("success".to_owned(), FeatureSpec::new(Dtype::Int64, [1u64]));
    features.insert(
        "action_source".to_owned(),
        FeatureSpec::new(Dtype::Int64, [1u64]),
    );

    let mut writer = LeRobotWriter::create(root, Info::new(30.0, features)).expect("create");
    for ep_idx in 0..2u32 {
        let n = 40usize;
        let mut columns = BTreeMap::new();
        columns.insert(
            "observation.state".to_owned(),
            Column::F32((0..n).map(|i| (i % 9) as f32).collect()),
        );
        columns.insert(
            "action".to_owned(),
            Column::F32((0..n).map(|i| (i % 5) as f32 + shift as f32).collect()),
        );
        columns.insert(
            "success".to_owned(),
            Column::I64(vec![i64::from(ep_idx % 2 == 0); n]),
        );
        columns.insert(
            "action_source".to_owned(),
            // Every 10th frame is "clamped" (nonzero) -- a fixed, checkable rate.
            Column::I64((0..n).map(|i| i64::from(i % 10 == 0)).collect()),
        );
        let episode = Episode {
            index: ep_idx,
            tasks: vec!["task".to_owned()],
            timestamps: (0..n).map(|i| i as f64 / 30.0).collect(),
            task_index: vec![0; n],
            columns,
            video: BTreeMap::new(),
        };
        writer.write_episode(&episode).expect("write episode");
    }
    writer.finish().expect("finish");
}

#[test]
fn gap_identical_datasets_exit_zero_and_write_report() {
    let dir = scratch_dir("gap-identical");
    let sim = dir.join("sim");
    let real = dir.join("real");
    write_gap_fixture(&sim, 0.0);
    write_gap_fixture(&real, 0.0);
    let out = dir.join("gap_report.json");

    let result = bin()
        .args([
            "gap",
            "--sim",
            sim.to_str().unwrap(),
            "--real",
            real.to_str().unwrap(),
        ])
        .args(["--out", out.to_str().unwrap()])
        .output()
        .expect("run es");
    let text = stdout(&result);
    assert!(
        result.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(text.contains("observation.state"), "{text}");
    assert!(text.contains("action"), "{text}");
    assert!(out.is_file());

    let report: es_eval::domain_gap::GapReport =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("read report"))
            .expect("report json");
    assert!(!report.has_flagged(), "{report:?}");
}

#[test]
fn gap_shifted_action_channel_is_flagged_and_exits_one() {
    let dir = scratch_dir("gap-shifted");
    let sim = dir.join("sim");
    let real = dir.join("real");
    write_gap_fixture(&sim, 0.0);
    write_gap_fixture(&real, 50.0);
    let out = dir.join("gap_report.json");

    let result = bin()
        .args([
            "gap",
            "--sim",
            sim.to_str().unwrap(),
            "--real",
            real.to_str().unwrap(),
        ])
        .args(["--out", out.to_str().unwrap()])
        .output()
        .expect("run es");
    assert_eq!(result.status.code(), Some(1), "{}", stdout(&result));

    let report: es_eval::domain_gap::GapReport =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("read report"))
            .expect("report json");
    assert!(report.has_flagged(), "{report:?}");
    assert!(
        report.suspects.iter().any(|s| s.channel == "action"),
        "{report:?}"
    );
    let state = report
        .channels
        .iter()
        .find(|c| c.name == "observation.state")
        .expect("state channel present");
    assert!(!state.flagged, "{report:?}");
}

#[test]
fn gap_missing_required_flags_is_usage_error() {
    let out = bin()
        .args(["gap", "--sim", "/nonexistent"])
        .output()
        .expect("run es");
    assert_eq!(out.status.code(), Some(2));
}

// --- `es loop` (spec 13.1, 13.2, 13.3; docs/packets/M3/W7-learning-loop.md) -----------------

/// A minimal collected-looking dataset: `episodes` episodes of 8 frames with the two loop
/// columns already in the schema, so `es loop intervene` moves `content` and not `schema`.
fn write_loop_fixture(root: &Path, episodes: u32) {
    let mut features = BTreeMap::new();
    features.insert(
        "observation.state".to_owned(),
        FeatureSpec::new(Dtype::Float32, [2u64]),
    );
    features.insert(
        "action".to_owned(),
        FeatureSpec::new(Dtype::Float32, [1u64]),
    );
    features.insert(
        "intervention".to_owned(),
        FeatureSpec::new(Dtype::Int64, [1u64]),
    );
    features.insert(
        "action_source".to_owned(),
        FeatureSpec::new(Dtype::Int64, [1u64]),
    );
    let mut writer = LeRobotWriter::create(root, Info::new(100.0, features)).expect("create");
    for index in 0..episodes {
        let n = 8usize;
        let mut columns = BTreeMap::new();
        columns.insert(
            "observation.state".to_owned(),
            Column::F32((0..n * 2).map(|i| i as f32 + index as f32).collect()),
        );
        columns.insert(
            "action".to_owned(),
            Column::F32((0..n).map(|i| i as f32 * 0.25).collect()),
        );
        columns.insert("intervention".to_owned(), Column::I64(vec![0; n]));
        columns.insert("action_source".to_owned(), Column::I64(vec![0; n]));
        writer
            .write_episode(&Episode {
                index,
                tasks: vec!["es:task:fixture".to_owned()],
                timestamps: (0..n).map(|i| i as f64 / 100.0).collect(),
                task_index: vec![0; n],
                columns,
                video: BTreeMap::new(),
            })
            .expect("write episode");
    }
    writer.finish().expect("finish");
}

/// `es loop intervene` labels a dataset in place: `content` moves, `schema` does not, and the
/// ledger records the step (spec 13.2, 13.3, 19.2).
#[test]
fn loop_intervene_labels_a_dataset_and_moves_only_the_content_hash() {
    let dir = scratch_dir("loop-intervene");
    let root = dir.join("data");
    write_loop_fixture(&root, 2);

    let segments = dir.join("segments.json");
    write(
        &segments,
        r#"[{"episode": 1, "start_frame": 2, "end_frame": 4, "source": "teleop",
             "operator_id": "op-1", "note": "corrected the grasp"}]"#,
    );

    let out = bin()
        .args(["loop", "intervene", "--dataset"])
        .arg(&root)
        .arg("--segments")
        .arg(&segments)
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        out.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("labelled 3 frames across 1 episodes"),
        "{text}"
    );
    assert!(text.contains("unchanged"), "schema must not move: {text}");
    let before = text
        .lines()
        .find_map(|l| l.strip_prefix("content before: "))
        .expect("content before");
    let after = text
        .lines()
        .find_map(|l| l.strip_prefix("content after:  "))
        .expect("content after");
    assert_ne!(before, after, "the parquet bytes changed: {text}");

    // Provenance on disk, and one ledger line (spec 13.2, 13.3).
    let jsonl = std::fs::read_to_string(root.join("meta/interventions.jsonl")).expect("segments");
    assert!(jsonl.contains("\"operator_id\":\"op-1\""), "{jsonl}");
    let ledger = std::fs::read_to_string(root.join("loop.jsonl")).expect("ledger");
    assert_eq!(ledger.lines().count(), 1, "{ledger}");
    assert!(ledger.contains("\"intervene\""), "{ledger}");
}

/// `es loop distill` merges two datasets, writes the spec 19.3 identity, and appends the step
/// to every input ledger as well as the output's (spec 13.3).
#[test]
fn loop_distill_merges_two_datasets_and_writes_training_identity() {
    let dir = scratch_dir("loop-distill");
    let (a, b, out) = (dir.join("a"), dir.join("b"), dir.join("merged"));
    write_loop_fixture(&a, 3);
    write_loop_fixture(&b, 2);

    let run = bin()
        .args(["loop", "distill", "--in"])
        .arg(&a)
        .arg("--in")
        .arg(&b)
        .args([
            "--train", "0.6", "--val", "0.2", "--test", "0.2", "--seed", "7",
        ])
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run es");
    let text = stdout(&run);
    assert!(
        run.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(text.contains("training_hash:"), "{text}");
    assert!(text.contains("all-zero digest"), "{text}");

    let identity: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("training_identity.json")).expect("identity"),
    )
    .expect("identity json");
    assert!(identity.get("dataset").is_some(), "{identity}");

    // `es dataset info` sees five episodes: the merge summed them.
    let info = bin()
        .args(["dataset", "info"])
        .arg(&out)
        .output()
        .expect("run es");
    let itext = stdout(&info);
    assert!(info.status.success(), "{itext}");
    assert!(itext.contains("episodes: 5"), "{itext}");

    for root in [&a, &b, &out] {
        let ledger = std::fs::read_to_string(root.join("loop.jsonl")).expect("ledger");
        assert!(
            ledger.contains("\"distill\""),
            "{}: {ledger}",
            root.display()
        );
    }
}

/// `es loop collect` refuses rather than fakes when the backend or the runtime is missing
/// (spec 1.4). Exit 3 is "nothing ran", distinct from a failed run (1) or bad usage (2).
#[test]
fn loop_collect_skips_with_exit_three_when_the_backend_is_unavailable() {
    let dir = scratch_dir("loop-collect-skip");
    let policy = dir.join("policy.esb");
    std::fs::write(&policy, build_policy_bundle(&deployable_fixture())).expect("write policy.esb");

    let out = bin()
        .env("ES_PYTHON", "es-no-such-python")
        .args(["loop", "collect", "--policy"])
        .arg(&policy)
        // Availability is checked before the scene is touched, so a missing file is fine.
        .args([
            "--scene",
            "does-not-exist.xml",
            "--episodes",
            "1",
            "--seed",
            "1",
            "--out",
        ])
        .arg(dir.join("out"))
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert_eq!(
        out.status.code(),
        Some(3),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("SKIPPED"), "{text}");
}

/// `es dataset export --lerobot-v3` turns a v2.1 dataset into the layout `lerobot` 0.6.1
/// reads, leaves the source alone, and records where it came from (spec 19.1, 19.2; packet
/// `docs/packets/M5/V1b-lerobot-v3-export.md`). What the real package makes of the output is
/// `crates/es-data/tests/lerobot_v3.rs`; this only checks the command.
#[test]
fn dataset_export_writes_a_v3_dataset() {
    let dir = scratch_dir("dataset-export");
    let root = dir.join("v21");
    let out = dir.join("v30");
    write_loop_fixture(&root, 2);

    let res = bin()
        .args(["dataset", "export", "--lerobot-v3"])
        .arg(&root)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run es");
    let text = stdout(&res);
    assert!(res.status.success(), "{text}");
    assert!(text.contains("episodes: 2"), "{text}");
    assert!(text.contains("frames: 16"), "{text}");

    let info: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("meta/info.json")).expect("exported info.json"),
    )
    .expect("info json");
    assert_eq!(info["codebase_version"], "v3.0", "{info}");
    assert!(out.join("data/chunk-000/file-000.parquet").is_file());
    assert!(out
        .join("meta/episodes/chunk-000/file-000.parquet")
        .is_file());
    assert!(out.join("meta/tasks.parquet").is_file());
    assert!(out.join("meta/es_provenance.json").is_file());

    // The source keeps writing and reading v2.1 (packet `forbidden`).
    let source: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("meta/info.json")).expect("source info.json"),
    )
    .expect("info json");
    assert_eq!(source["codebase_version"], "v2.1", "{source}");

    // A missing --out is usage (2); a root that is not a dataset is a runtime failure (1).
    let usage = bin()
        .args(["dataset", "export", "--lerobot-v3"])
        .arg(&root)
        .output()
        .expect("run es");
    assert_eq!(usage.status.code(), Some(2), "{}", stdout(&usage));
    let missing = bin()
        .args([
            "dataset",
            "export",
            "--lerobot-v3",
            "does-not-exist",
            "--out",
        ])
        .arg(dir.join("nope"))
        .output()
        .expect("run es");
    assert_eq!(missing.status.code(), Some(1), "{}", stdout(&missing));
}

/// An unknown subcommand and a missing required flag are usage errors, not runtime ones.
#[test]
fn loop_usage_errors_exit_two() {
    for args in [
        vec!["loop", "nope"],
        vec!["loop", "distill", "--out", "x"],
        vec!["loop", "intervene", "--dataset", "x"],
    ] {
        let out = bin().args(&args).output().expect("run es");
        assert_eq!(out.status.code(), Some(2), "{args:?}");
    }
}

// --- P-M3-R2 / P-M3-R5 follow-ups (docs/reviews/M3.md) --------------------------------------

/// A hostile `LeRobot` dataset: `write_gap_fixture`'s schema with `observation.state` values
/// replaced by `NaN`, and -- after the writer has had its say -- `meta/info.json` patched to
/// declare `action` two elements wide when the parquet column holds one per frame. Both are
/// things a foreign dataset can be and `LeRobotWriter` cannot produce, so they are made here
/// rather than fixed up in `es-data`.
fn write_hostile_gap_fixture(root: &Path, nan: bool, widen_action: bool) {
    let mut features = BTreeMap::new();
    features.insert(
        "observation.state".to_owned(),
        FeatureSpec::new(Dtype::Float32, [1u64]),
    );
    features.insert(
        "action".to_owned(),
        FeatureSpec::new(Dtype::Float32, [1u64]),
    );

    let mut writer = LeRobotWriter::create(root, Info::new(30.0, features)).expect("create");
    let n = 40usize;
    let mut columns = BTreeMap::new();
    columns.insert(
        "observation.state".to_owned(),
        Column::F32(
            (0..n)
                .map(|i| {
                    if nan && i % 7 == 0 {
                        f32::NAN
                    } else {
                        (i % 9) as f32
                    }
                })
                .collect(),
        ),
    );
    columns.insert(
        "action".to_owned(),
        Column::F32((0..n).map(|i| (i % 5) as f32).collect()),
    );
    writer
        .write_episode(&Episode {
            index: 0,
            tasks: vec!["task".to_owned()],
            timestamps: (0..n).map(|i| i as f64 / 30.0).collect(),
            task_index: vec![0; n],
            columns,
            video: BTreeMap::new(),
        })
        .expect("write episode");
    writer.finish().expect("finish");

    if widen_action {
        let path = root.join("meta").join("info.json");
        let text = std::fs::read_to_string(&path).expect("read info.json");
        // The `action` feature is the only `[1]` shape before `observation.state`'s, and
        // serde_json preserves key order, so a targeted replace is enough here.
        let patched = text.replacen(
            "\"shape\": [\n        1\n      ]",
            "\"shape\": [\n        2\n      ]",
            1,
        );
        assert_ne!(patched, text, "info.json shape not found:\n{text}");
        write(&path, &patched);
    }
}

/// Both hostile shapes at once: `es gap` must exit with a message, never panic (P-M3-R2).
#[test]
fn gap_hostile_dataset_errors_instead_of_panicking() {
    let dir = scratch_dir("gap-hostile");
    let sim = dir.join("sim");
    let real = dir.join("real");
    write_hostile_gap_fixture(&sim, true, true);
    write_gap_fixture(&real, 0.0);

    let out = bin()
        .args([
            "gap",
            "--sim",
            sim.to_str().unwrap(),
            "--real",
            real.to_str().unwrap(),
        ])
        .args(["--out", dir.join("gap_report.json").to_str().unwrap()])
        .output()
        .expect("run es");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}", stdout(&out));
    assert!(err.contains("error:"), "{err}");
    assert!(err.contains("inconsistent"), "{err}");
    assert!(!err.contains("panicked"), "{err}");
}

/// A `NaN` on its own is dropped and counted, not a panic on the way to `gap_report.json`
/// (the `.expect` in `GapReport::to_json` that P-M3-R2 removed).
#[test]
fn gap_nonfinite_samples_are_dropped_and_counted() {
    let dir = scratch_dir("gap-nonfinite");
    let sim = dir.join("sim");
    let real = dir.join("real");
    write_hostile_gap_fixture(&sim, true, false);
    write_hostile_gap_fixture(&real, false, false);
    let out_path = dir.join("gap_report.json");

    let out = bin()
        .args([
            "gap",
            "--sim",
            sim.to_str().unwrap(),
            "--real",
            real.to_str().unwrap(),
        ])
        .args(["--out", out_path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert!(
        out.status.code() == Some(0) || out.status.code() == Some(1),
        "stdout:\n{}\nstderr:\n{}",
        stdout(&out),
        String::from_utf8_lossy(&out.stderr)
    );
    let report: es_eval::domain_gap::GapReport =
        serde_json::from_str(&std::fs::read_to_string(&out_path).expect("read report"))
            .expect("report json");
    let state = report
        .channels
        .iter()
        .find(|c| c.name == "observation.state")
        .expect("state channel present");
    assert_eq!(state.sim_nonfinite_dropped, 6, "{report}");
    assert_eq!(state.real_nonfinite_dropped, 0, "{report}");
    assert!(
        state.sim_mean.is_finite() && state.sim_std.is_finite(),
        "{report}"
    );
}

/// A `reports/0/evaluation.lock` lifted from another run, with the Safety Case rewritten to
/// record *its* blake3, so the entry-hash check passes and only the lock's own
/// `execution_hash` can catch the swap (P-M3-R5).
#[test]
fn evidence_verify_treats_a_foreign_lock_as_stale() {
    let a = build_evidence(40.0);
    let b = build_evidence(35.0); // a tighter envelope: `execution_hash` moves
    assert_ne!(a.chain.execution_hash(), b.chain.execution_hash());

    let entry = bundle::report_entry(0, EVALUATION_LOCK);
    let foreign = bundle::read(&b.bytes).expect("reads").entries[&entry].clone();
    let mut case = a.case.clone();
    for e in &mut case.evidence {
        if e.entry == entry {
            e.hash = *blake3::hash(&foreign).as_bytes();
        }
    }
    let mut text = serde_json::to_string_pretty(&case).expect("serializes");
    text.push('\n');
    let swapped = rewrite_entry(&a.bytes, &entry, foreign);
    let swapped = rewrite_entry(&swapped, SAFETY_CASE, text.into_bytes());

    let report = EvidenceBundle::verify(&swapped, None, &[]).expect("verifies");
    assert!(!report.ok(), "a lock of another run is not coverage");
    let req = report
        .coverage
        .iter()
        .find(|c| c.requirement == "REQ-11")
        .expect("REQ-11 is the lock's requirement");
    assert!(!req.covered, "{req:?}");
    assert_eq!(req.stale, vec!["EV-lock".to_owned()]);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code.as_str() == es_eval::evidence::EVID_EXECUTION_HASH),
        "{:?}",
        report.diagnostics
    );
}

/// A case with no requirements is vacuously "every requirement covered"; verify must still
/// fail it (P-M3-R5).
#[test]
fn evidence_verify_fails_a_case_with_no_requirements() {
    let built = build_evidence(40.0);
    let mut case = built.case.clone();
    case.requirements.clear();
    case.claims.clear();
    case.traceability.clear();
    let mut text = serde_json::to_string_pretty(&case).expect("serializes");
    text.push('\n');
    let empty = rewrite_entry(&built.bytes, SAFETY_CASE, text.into_bytes());

    let report = EvidenceBundle::verify(&empty, None, &[]).expect("verifies");
    assert!(report.coverage.is_empty());
    assert!(!report.ok(), "an empty case must not verify");
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code.as_str() == es_eval::evidence::EVID_EMPTY_CASE),
        "{:?}",
        report.diagnostics
    );

    let dir = scratch_dir("evidence-empty-case");
    let path = dir.join("evidence.esb");
    std::fs::write(&path, &empty).expect("write bundle");
    let out = bin()
        .args(["evidence", "verify", path.to_str().unwrap()])
        .output()
        .expect("run es");
    assert_eq!(out.status.code(), Some(1), "{}", stdout(&out));
}

// --- `es mcp` (spec 14.5 MCP interface) ------------------------------------------------------

/// Spawns `es mcp` with piped stdin/stdout, writes newline-delimited JSON-RPC requests, closes
/// stdin, and returns the newline-delimited responses parsed as JSON -- the CLI-process
/// counterpart of `crates/es-script/tests/mcp.rs`'s in-process harness.
fn run_mcp(requests: &[String]) -> Vec<serde_json::Value> {
    use std::io::Write as _;
    use std::process::Stdio;

    let mut child = bin()
        .args(["mcp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn es mcp");
    {
        let stdin = child.stdin.as_mut().expect("piped stdin");
        for req in requests {
            writeln!(stdin, "{req}").expect("write request");
        }
    } // drop stdin: EOF, so the server loop ends
    let out = child.wait_with_output().expect("es mcp exits");
    assert!(
        out.status.success(),
        "es mcp exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout(&out)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON: {l}: {e}")))
        .collect()
}

#[test]
fn mcp_initialize_tools_list_and_validate_over_piped_stdin() {
    let dir = scratch_dir("mcp");
    let (task, ..) = write_fixture_toml(&dir);
    let task_toml = std::fs::read_to_string(&task).expect("read fixture task toml");

    let requests = vec![
        serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            .to_string(),
        serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).to_string(),
        serde_json::json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "validate", "arguments": {"kind": "task", "toml": task_toml}}
        })
        .to_string(),
    ];
    let responses = run_mcp(&requests);
    assert_eq!(responses.len(), 3);
    assert!(responses[0]["result"]["protocolVersion"].is_string());
    let tools = responses[1]["result"]["tools"]
        .as_array()
        .expect("tools array");
    assert!(tools.iter().any(|t| t["name"] == "validate"));
    assert_eq!(responses[2]["result"]["isError"], false);
    let body: serde_json::Value = serde_json::from_str(
        responses[2]["result"]["content"][0]["text"]
            .as_str()
            .unwrap(),
    )
    .expect("json content");
    assert_eq!(body["kind"], "task");
    assert_eq!(body["has_error"], false);
}

// --- `es import roboverse` (spec 14.4 external conversion, M4 W6) ---------------------------

/// `es import roboverse` on the M4 W6 pick-and-place fixture, round-tripping the Task IR /
/// Observation IR pair it writes through `es ir validate`.
#[test]
fn import_roboverse_round_trips_through_ir_validate() {
    let dir = scratch_dir("import-roboverse");
    let task_json = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/roboverse/pick_and_place.json"
    );
    let out_dir = dir.join("out");

    let out = bin()
        .args(["import", "roboverse", task_json, "--out"])
        .arg(&out_dir)
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert!(
        out.status.success(),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("task_hash:"), "{text}");
    assert!(text.contains("observation_hash:"), "{text}");

    let task = out_dir.join("task.toml");
    let observation = out_dir.join("observation.toml");
    let provenance = out_dir.join("provenance.json");
    assert!(task.is_file());
    assert!(observation.is_file());
    assert!(provenance.is_file());

    let provenance_body: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&provenance).expect("read provenance.json"))
            .expect("provenance.json is valid JSON");
    assert_eq!(provenance_body["license"], "Apache-2.0");

    let validate = bin()
        .args(["ir", "validate"])
        .args([&task, &observation])
        .output()
        .expect("run es");
    let vtext = stdout(&validate);
    assert!(
        validate.status.success(),
        "stdout:\n{vtext}\nstderr:\n{}",
        String::from_utf8_lossy(&validate.stderr)
    );
    assert!(!vtext.contains("ERROR"), "{vtext}");
}

/// spec 14.4: an unmapped item with `severity: error` blocks execution -- an unrecognized
/// checker kind must exit 1, not 0.
#[test]
fn import_roboverse_exits_1_on_an_unmapped_checker() {
    let dir = scratch_dir("import-roboverse-unmapped");
    let task_json = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/roboverse/unknown_checker.json"
    );

    let out = bin()
        .args(["import", "roboverse", task_json, "--out"])
        .arg(dir.join("out"))
        .output()
        .expect("run es");
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{text}");
    assert!(text.contains("unmapped"), "{text}");
    assert!(text.contains("SomeFutureChecker"), "{text}");
}

// --- `es task generate` (spec 14.5) --------------------------------------------------------

/// `task_ir()` (above) has no `Reward` node; `es-script::generate`'s completeness check
/// (`GEN-001`) wants one, so this appends a disconnected but well-typed one.
fn task_ir_with_reward() -> TaskIr {
    let mut task = task_ir();
    task.graph.insert(
        NodeId(99),
        TaskNode::Reward {
            name: "progress".to_owned(),
            weight: 1.0,
            aggregation: es_ir::task::Aggregation::Sum,
            ty: PortType {
                elem: ElemType::F32,
                shape: Shape::new([1]),
                unit: Unit::Normalized { lo: -1.0, hi: 1.0 },
                frame: Frame::World,
                time: TimeRef::Tick,
                image: None,
            },
        },
    );
    task
}

#[test]
fn task_generate_stdin_provider_accepts_a_valid_reply() {
    use std::io::Write as _;
    use std::process::Stdio;

    let task = task_ir_with_reward();
    let reply = format!(
        "```toml\n{}\n```",
        es_ir::serial::task_to_toml(&task).expect("task toml")
    );

    let dir = scratch_dir("task-generate-stdin");
    let mut child = bin()
        .args(["task", "generate", "--prompt", "reach the target"])
        .args(["--provider", "stdin", "--out"])
        .arg(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn es task generate");
    child
        .stdin
        .as_mut()
        .expect("piped stdin")
        .write_all(reply.as_bytes())
        .expect("write reply");
    let out = child.wait_with_output().expect("es task generate exits");
    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        stdout(&out),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(dir.join("task.toml").exists(), "{}", stdout(&out));
}

#[test]
fn task_generate_reports_failure_when_no_round_validates() {
    let dir = scratch_dir("task-generate-stdin-empty");
    let out = bin()
        .args(["task", "generate", "--prompt", "reach the target"])
        .args(["--provider", "stdin", "--rounds", "1", "--out"])
        .arg(&dir)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run es task generate");
    assert_eq!(out.status.code(), Some(1), "{}", stdout(&out));
    assert!(!dir.join("task.toml").exists());
}

/// Workspace-root path of a hand-written `.usda` fixture (M4 W3).
fn usd_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/usd")
        .join(name)
}

#[test]
fn import_usd_writes_a_scene_that_round_trips() {
    let dir = scratch_dir("import-usd");
    let out = dir.join("scene.json");
    let run = bin()
        .args(["import", "usd"])
        .arg(usd_fixture("pendulum.usda"))
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run es import usd");
    assert_eq!(run.status.code(), Some(0), "{}", stdout(&run));
    let printed = stdout(&run);
    assert!(printed.contains("bodies: 2"), "{printed}");
    assert!(printed.contains("joints: 1"), "{printed}");
    assert!(printed.contains("scene_hash: "), "{printed}");

    let json = std::fs::read_to_string(&out).expect("scene.json");
    let scene: es_assets::scene::SceneDesc = serde_json::from_str(&json).expect("scene json");
    assert!(scene.validate().is_ok());
    // The hash the CLI printed is the hash of what it wrote.
    let hash = hex(&scene.scene_hash());
    assert!(printed.contains(&hash), "{printed}");
}

#[test]
fn import_usd_refuses_a_referenced_layer_by_prim_path() {
    let dir = scratch_dir("import-usd-refused");
    let out = dir.join("scene.json");
    let run = bin()
        .args(["import", "usd"])
        .arg(usd_fixture("referenced.usda"))
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run es import usd");
    assert_eq!(run.status.code(), Some(1), "{}", stdout(&run));
    let err = String::from_utf8_lossy(&run.stderr);
    assert!(err.contains("/World/robot"), "{err}");
    assert!(!out.exists(), "nothing is written on a refusal");
}

/// S-2 (docs/reviews/M4.md): the CLI must refuse a file over `es_usd::MAX_USDA_BYTES` by its
/// metadata length, the same cap `parse_usda` enforces, rather than reading it fully into
/// memory first.
#[test]
fn import_usd_refuses_an_oversized_file() {
    let dir = scratch_dir("import-usd-oversized");
    let huge = dir.join("huge.usda");
    std::fs::write(&huge, vec![b'a'; es_usd::MAX_USDA_BYTES + 1]).expect("write huge fixture");
    let out = dir.join("scene.json");
    let run = bin()
        .args(["import", "usd"])
        .arg(&huge)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run es import usd");
    assert_eq!(run.status.code(), Some(1), "{}", stdout(&run));
    let err = String::from_utf8_lossy(&run.stderr);
    assert!(err.contains("cap"), "{err}");
    assert!(!out.exists(), "nothing is written on a refusal");
    let _ = std::fs::remove_file(&huge);
}

// --- signed-manifest tamper (P-M4-R2, review B-2) ------------------------------------------

/// The signature must cover the manifest, not just the entries: rewriting the declared spec
/// 5.3 chain on a signed bundle -- same entries, same key, same signature bytes -- must not
/// still report `Valid`. It must also be caught unsigned, because `verify` now cross-checks
/// the bundle's own `manifest.hashes` against its `chain.json`.
#[test]
fn evidence_verify_flags_a_rewritten_manifest_on_a_signed_bundle() {
    let built = build_evidence(40.0);
    let signing_key = SigningKey::from_bytes(&[7u8; 32]);
    let signed = EvidenceBundle::sign(&built.bytes, &signing_key).expect("signs");
    let raw = bundle::read(&signed).expect("reads");

    let manifest = BundleManifest {
        hashes: es_compile::bundle::BundleHashes {
            task: Some([0xAB; 32]),
            ..raw.manifest.hashes
        },
        ..raw.manifest
    };
    let rewritten = bundle::write(&manifest, &raw.entries).expect("writes");

    let report =
        EvidenceBundle::verify(&rewritten, None, &[signing_key.verifying_key()]).expect("verifies");
    assert_eq!(report.signature, SignatureStatus::Invalid);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code.as_str() == es_eval::evidence::EVID_CHAIN_SLOT),
        "{:?}",
        report.diagnostics
    );
    assert!(
        !report.ok(),
        "a manifest that contradicts chain.json is a defect"
    );
}

/// S-9 (P-M4-S9-S10): a blocked `RoboVerse` conversion must leave no output behind -- the
/// `severity: error` check now runs before the first `std::fs::write`, matching `import usd`.
#[test]
fn import_roboverse_writes_nothing_when_an_unmapped_item_blocks_it() {
    let dir = scratch_dir("import-roboverse-nothing-written");
    let task_json = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/roboverse/unknown_checker.json"
    );
    let out_dir = dir.join("out");

    let out = bin()
        .args(["import", "roboverse", task_json, "--out"])
        .arg(&out_dir)
        .output()
        .expect("run es");
    assert_eq!(out.status.code(), Some(1), "{}", stdout(&out));
    assert!(!out_dir.exists(), "nothing is written on a refusal");
}

// --- plan V: the SO-101 cube-into-bin demo documents (packet M5/V0) -------------------------

/// Workspace-root path of one of the four plan V fixture documents.
fn vl_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning")
        .join(name)
}

/// The four documents validate, cross-check and compile, and every hash they print is stable
/// across two runs -- the hash chain (spec 5.3) is a function of the documents alone.
#[test]
fn visible_learning_documents_compile() {
    let task = vl_fixture("task.toml");
    let observation = vl_fixture("observation.toml");
    let learning = vl_fixture("learning.toml");
    let deployment = vl_fixture("deployment.toml");

    let validate = || {
        let out = bin()
            .args(["ir", "validate"])
            .args([&task, &observation, &learning, &deployment])
            .output()
            .expect("run es ir validate");
        assert!(
            out.status.success(),
            "stdout:\n{}\nstderr:\n{}",
            stdout(&out),
            String::from_utf8_lossy(&out.stderr)
        );
        stdout(&out)
    };
    let first = validate();
    for kind in ["task", "observation", "learning", "deployment"] {
        assert!(first.contains(&format!("{kind}_hash: ")), "{first}");
    }
    assert!(!first.contains("ERROR"), "{first}");
    assert_eq!(first, validate(), "the four hashes are not stable");

    // The cross-IR check (spec 11.1) over the same four documents.
    let cross = bin()
        .args(["ir", "check"])
        .args([&task, &observation, &learning, &deployment])
        .output()
        .expect("run es ir check");
    let text = stdout(&cross);
    assert!(cross.status.success(), "{text}");
    assert!(!text.contains("ERROR"), "{text}");

    // And the Observation IR lowers to a CPU plan.
    let compile = || {
        let out = bin()
            .args(["task", "compile"])
            .args([&task, &observation])
            .output()
            .expect("run es task compile");
        assert!(out.status.success(), "{}", stdout(&out));
        stdout(&out)
    };
    let plan = compile();
    assert!(plan.contains("compiler_hash: "), "{plan}");
    assert!(plan.contains("shape=[3, 96, 96]"), "{plan}");
    assert_eq!(plan, compile(), "the compiler hash is not stable");
}

/// Packet M5/V0 puts this test here rather than in `es-physics-backend`'s own
/// `tests/so101_scene.rs`: `RandomizationPlan` lives in `es-env` (layer 9), and a dev
/// dependency from layer 4 on layer 9 is a `cargo xtask layering` rule-1 violation --
/// `cargo metadata` does not distinguish a dev dependency. The `es` binary crate is not in
/// the spec 4.2 layer table and already depends on both.
#[test]
fn the_cube_free_joint_is_randomizable() {
    use es_env::randomize::{RandomizationPlan, ResetBuffer};
    use es_physics_backend::MuJoCoCpuBackend;
    use es_physics_core::{LoadConfig, PhysicsBackend};

    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP the_cube_free_joint_is_randomizable: {reason}");
        return;
    }
    let xml = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/mjcf/so101_pick_place.xml"),
    )
    .expect("the demo scene");
    let scene = es_assets::parse_mjcf(&xml)
        .expect("the demo scene parses")
        .scene;
    let mut backend = MuJoCoCpuBackend::new();
    let model = backend
        .load(
            &scene,
            &LoadConfig {
                n_envs: 1,
                rate: Some(TickRate::hz(200)),
                seed: 1,
            },
        )
        .expect("the demo scene loads");

    let raw = std::fs::read_to_string(vl_fixture("task.toml")).expect("task.toml");
    let task = es_ir::serial::task_from_toml(&raw).expect("task.toml parses");
    // Every declared target resolves; an unresolvable one is `EnvError::Unsupported` by name.
    let plan = RandomizationPlan::compile(&task, &scene, &model).expect("every target resolves");
    assert!(!plan.is_empty());

    let draw = |seed: u64, episode: u64| {
        let mut qpos = vec![0.0; model.nq as usize];
        let mut qvel = vec![0.0; model.nv as usize];
        let mut scales = es_env::randomize::ParamScales::new();
        plan.apply(
            seed,
            0,
            episode,
            &mut ResetBuffer {
                qpos: &mut qpos,
                qvel: &mut qvel,
                scales: &mut scales,
            },
        );
        // No mass/friction/gain target is declared, so nothing is recorded but not applied.
        assert!(scales.is_empty(), "{scales:?}");
        qpos
    };

    let a = draw(7, 0);
    assert_eq!(
        a,
        draw(7, 0),
        "the same (seed, episode) is not reproducible"
    );
    let b = draw(7, 1);
    assert_ne!(a[6..9], b[6..9], "two episodes give the same cube pose");
    assert_ne!(
        a[6..9],
        draw(8, 0)[6..9],
        "two seeds give the same cube pose"
    );
    // The arm's reset is deterministic; only the cube moves.
    assert_eq!(a[..6], b[..6]);
    println!("RAN the_cube_free_joint_is_randomizable");
}

// --- plan V, packet M5/V1: the fixture regenerator and the expert oracles ---------------------

/// The header comment each regenerated document keeps: `task_to_toml` writes values, not prose,
/// and the prose is what says *why* the values are what they are.
const TASK_HEADER: &str = "\
# Task IR (spec 6) for the SO-101 cube-into-bin demo -- plan V, packets M5/V0 and M5/V1.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`
# from tests/fixtures/mjcf/so101_pick_place.xml, so `scene_hash`, `asset_hash` and every
# hash downstream of them are derived, never typed in.
#
# Parameters, fixed here so V2 and V3 cite them: NJ = 6, control rate 50 Hz,
# max_episode_steps = 1800 (36 s, against a scripted demonstration that takes about 7),
# image 96x96 Rgb8 from the one fixed `overhead` camera.
#
# `sim_cube_pose` IS SIMULATOR-PRIVILEGED. A real SO-101 has no sensor that reports where the
# cube is; this channel is the simulator handing the policy an answer, so that stage 1 of the
# demo can ask whether the policy can do the task *given* the cube's pose before asking whether
# it can find a 25 mm cube in a 96x96 frame (packet M5/V7a, design note section 7.14). Nothing
# validates that: `ObsChannel` is { source, ty } and spec 7.4 says nothing else belongs there,
# so the `sim_` prefix is the whole of the mark, and it is carried unchanged by the Observation
# IR output port, the Learning IR input and the policy contract. A deployment aimed at hardware
# must drop this channel; the vision packet is the one that does.
#
# This document *declares* the ObservationSpec and owns no preprocessing and no neural net
# (spec 5.1): the image chain is observation.toml, the policy is learning.toml. The channel
# is declared as the renderer delivers it -- U8, HWC [96, 96, 3] -- and observation.toml's
# `Dequantize` node is what turns it into the CHW F32 the policy reads (V0b section 7.4).
#
# Randomization (spec 6.3) is the cube free joint's `qpos` only. Mass, friction and
# actuator-gain targets are deliberately absent: `es-env` records such a draw but never
# pushes it into the backend (crates/es-env/src/randomize.rs:24-26), so declaring one would
# put a number in the episode record that never reached the physics.
#
# Two ceilings of the IR-D cone, recorded rather than papered over (design note section 5.4):
#  * `GetJointState` binds one scalar -- a joint's *first* qpos/qvel index
#    (crates/es-env/src/plan.rs). For the cube's free joint that is `x`, so the success
#    predicate is the bin's x span plus a settling bound, not a 3-axis AABB. V1's expert
#    oracle checks y and z directly against the backend state, so the measured success rate
#    is honest even though this predicate is narrower.
#  * Task IR-D has no constant leaf, and `Arith::Mul`'s right operand is dimensionless under
#    the spec 5.4 unit algebra, so neither `x - c` nor `v * v` is expressible; the shaped
#    reward is a `Normalize` of the cube's x and `|v| < b` is two `Compare`s and an `And`.
# `Normalize` and `Logic` are `TaskNode` variants `es-env`'s cone lowering handles as of V1
# (crates/es-env/src/plan.rs); neither is an es-ir change.
";

const OBSERVATION_HEADER: &str = "\
# Observation IR (spec 7) for the SO-101 cube-into-bin demo -- plan V, packets M5/V0, M5/V1.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`;
# `task_ref` is task.toml's `task_hash`. One Task IR may carry several Observation IRs.
#
# The ImageSpec is declared once, here, at the size the renderer will produce: 96x96 Rgb8
# from the `overhead` camera, pinhole, fx = fy = (96/2)/tan(45 deg / 2), principal point at
# the centre. There is no `Resize` and no `Crop` node, so no intrinsics transform is owed
# (spec 7.2, INV-14). There is no `MultiViewPack`: plan lowering rejects it
# (crates/es-compile/src/plan.rs), which is why the demo has exactly one camera.
#
# The chain is `ImageInput` (U8, HWC, exactly the renderer's tile) -> `Dequantize` (CHW F32,
# /255) -> `Normalize`. V0 declared the input as F32 CHW while its own ImageSpec said U8, so
# the renderer's tile did not fit the buffer and `capture` refused it (V0b section 7.4); V1
# fixes that here, which moves `observation_hash`.
#
# Time model (spec 7.5): `TemporalWindow { n_steps = 1 }`, matching the policy contract's
# `observation_window` (XIR-011); the `TemporalEncoder` lives in learning.toml.
#
# `StateInput` names the robot's `base` body, which is what the Task IR channel declares
# (spec 7.4 / XIR-002) and what DEP-031 measures the Safety Plane envelope against.
#
# The second state branch is `sim_cube_pose`, and IT IS SIMULATOR-PRIVILEGED (packet M5/V7a,
# design note section 7.14): no robot reports where the cube is. Its `StateInput` names the
# cube's **free joint**, not the cube body, which is what makes the reading exact --
# `input_sources` resolves a source id against `ModelInfo.qpos` before it falls back to the
# Task IR channel's leading-`dof` reading, so the port is served the joint's own qpos[6..13] at
# inference and the same range of the recorded `observation.state` row (`qpos || qvel`) when
# `es dataset bake` reads it. Its `Normalize{Range}` is +-0.3 m, the arm's reach, which holds
# both the cube's draw and the bin's interior; the quaternion's four values leave [0, 1] under
# that range, which is harmless (a Normalize is affine, not a clamp) and deliberate.
";

/// Adds `gripper is open` to the Task IR's success predicate, idempotently.
///
/// Without it the predicate fires while the cube is still *in the jaws*: a cone leaf is one
/// scalar and the cube's free joint gives `x` alone (design note section 5.4), so a cube
/// carried across the bin at rest looks exactly like a cube lying in it. The gripper's own
/// joint is a scalar the cone can read, and "in the bin's x span, at rest, and not being
/// held" is as close to the truth as IR-D gets. It is still narrower than the truth, which is
/// why `expert_solves_the_pinned_seeds` checks y and z itself.
fn add_gripper_open_term(task: &mut es_ir::task::TaskIr) {
    /// A jaw holding the 30 mm cube stalls at about 0.09 however hard it is told to close, so
    /// "the gripper is open" has to mean wider than that; the demonstration commands 0.9.
    ///
    /// **Why 0.85 and not 0.6** (packet M5/V15, design note section 7.23). The episode ends the
    /// instant this term goes true, so the threshold decides how much of the release survives
    /// into the recorded demonstration. At 0.6 the expert's opening ramp -- paced to the
    /// Deployment IR's envelope like every other joint -- was cut 14 control steps in, with the
    /// *command* still at 0.616 of the 0.9 it was aiming at: every one of V14's 200
    /// demonstrations contained exactly **one** frame commanded above 0.6 and none that reached
    /// `grip_open`, so the policy's only wide-open jaw was the 53-frame approach at the other
    /// end of the episode. 0.85 is reached only by a jaw that is both commanded to `grip_open`
    /// and empty -- it cannot close on a 30 mm cube and read this -- so it ends the episode on a
    /// finished release rather than on the first millimetre of one.
    const GRIPPER_OPEN: f64 = 0.85;

    use es_ir::graph::{NodeId, PortRef};
    use es_ir::task::{CmpOp, JointQuantity, LogicOp, TaskNode, TerminationKind};

    let success = *task
        .graph
        .nodes
        .iter()
        .find(|(_, n)| {
            matches!(
                n,
                TaskNode::Terminate {
                    kind: TerminationKind::Success
                }
            )
        })
        .expect("a Success node")
        .0;
    let feeding = task
        .graph
        .edges
        .iter()
        .find(|e| e.to.node == success)
        .expect("the Success node has an input")
        .from
        .clone();
    // The joint-state leaf type the cone already uses, borrowed from the cube's own reader so
    // nothing here is invented.
    let (robot, leaf_ty) = task
        .graph
        .nodes
        .values()
        .find_map(|n| match n {
            TaskNode::GetJointState {
                body,
                joints,
                quantity: JointQuantity::Position,
            } if joints.len() > 1 => Some((*body, joints.clone())),
            _ => None,
        })
        .expect("the arm's joint reader");
    assert!(leaf_ty.contains(&"gripper".to_owned()), "{leaf_ty:?}");
    let mut scalar = task
        .graph
        .nodes
        .values()
        .find_map(|n| match n {
            TaskNode::Compare { ty, .. } => Some(ty.clone()),
            _ => None,
        })
        .expect("a Compare type to copy");
    scalar.frame = es_ir::types::Frame::Joint(robot);
    scalar.unit = es_ir::types::Unit::Angle;

    // Already rewritten by an earlier run? Then only the threshold can have moved.
    let existing = task.graph.nodes.iter().find_map(|(id, n)| {
        matches!(n, TaskNode::GetJointState { joints, .. } if joints == &["gripper".to_owned()])
            .then_some(*id)
    });
    if let Some(leaf) = existing {
        let cmp = task
            .graph
            .edges
            .iter()
            .find(|e| e.from.node == leaf)
            .expect("the gripper leaf feeds a Compare")
            .to
            .node;
        match task.graph.nodes.get_mut(&cmp).expect("the Compare") {
            TaskNode::Compare { rhs, .. } => *rhs = Some(GRIPPER_OPEN),
            other => panic!("{other:?}"),
        }
        return;
    }
    let next = task.graph.nodes.keys().map(|n| n.0).max().expect("nodes") + 1;
    let (leaf, cmp, and) = (NodeId(next), NodeId(next + 1), NodeId(next + 2));
    task.graph.insert(
        leaf,
        TaskNode::GetJointState {
            body: robot,
            joints: vec!["gripper".to_owned()],
            quantity: JointQuantity::Position,
        },
    );
    task.graph.insert(
        cmp,
        TaskNode::Compare {
            op: CmpOp::Gt,
            rhs: Some(GRIPPER_OPEN),
            ty: scalar,
        },
    );
    task.graph.insert(
        and,
        TaskNode::Logic {
            op: LogicOp::And,
            shape: es_ir::types::Shape::new([1u64]),
        },
    );
    task.graph.edges.retain(|e| e.to.node != success);
    task.graph.connect(leaf, "value", cmp, "a");
    task.graph.edges.push(es_ir::graph::Edge {
        from: feeding,
        to: PortRef::new(and, "a"),
    });
    task.graph.connect(cmp, "value", and, "b");
    task.graph.connect(and, "value", success, "value");
}

/// Packet M5/V7a: declares the cube's pose as a second `ObservationSpec` channel.
///
/// **`sim_cube_pose` is simulator-privileged.** A real SO-101 has no sensor that reports where
/// the cube is; this channel exists so stage 1 of the demo can ask whether the policy can do
/// the task *given* the cube's pose, before asking whether it can find a 25 mm cube in a 96x96
/// frame. `ObsChannel` carries `source` and `ty` and nothing else (spec 7.4 forbids the rest),
/// so there is no field to tag — the `sim_` prefix is the mark, and it is carried unchanged by
/// the Observation IR output port, the Learning IR input and the policy contract.
///
/// Two ids, deliberately different. The graph *shows* the value as `GetBodyPose -> Concat`,
/// because Task IR-D's `GetJointState` binds one scalar per joint name (design note section
/// 5.4) and cannot emit a free joint's seven-wide `qpos`. The **channel** names the free joint
/// instead, because that is what decides the reading: `es_eval::runner::input_sources` resolves
/// a source id against `ModelInfo.qpos` first, so the channel is served as the joint's exact
/// `qpos[6..13]` rather than as a body pose derived from it.
fn add_privileged_cube_pose(
    task: &mut es_ir::task::TaskIr,
    scene: &es_assets::scene::SceneDesc,
) -> (es_core::StableId, es_ir::types::PortType) {
    use es_ir::graph::{Edge, NodeId, PortRef};
    use es_ir::task::{ObsChannel, ObsSource, TaskNode};
    use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

    let joint = scene
        .joints
        .iter()
        .find(|j| j.name == "cube_free")
        .expect("the cube's free joint")
        .id;
    let body = scene
        .bodies
        .iter()
        .find(|b| b.name == "cube")
        .expect("the cube body")
        .id;
    let vec_of = |n: u64, unit: Unit| PortType {
        elem: ElemType::F32,
        shape: Shape::new([n]),
        unit,
        frame: Frame::World,
        time: TimeRef::Tick,
        image: None,
    };
    // `Unit::Length` for all seven: three of them are metres and the quaternion's four are
    // dimensionless, and spec 5.4's algebra has no mixed unit. The position is what the policy
    // is being given; the tail is named in the fixture header rather than mistyped.
    let (pos, quat, pose) = (
        vec_of(3, Unit::Length),
        vec_of(4, Unit::Quaternion),
        vec_of(7, Unit::Length),
    );
    let (get, cat, decl) = (NodeId(34), NodeId(35), NodeId(36));
    task.graph.insert(
        get,
        TaskNode::GetBodyPose {
            body,
            relative_to: Frame::World,
        },
    );
    task.graph.insert(
        cat,
        TaskNode::Concat {
            parts: vec![pos, quat],
            axis: 0,
        },
    );
    task.graph.insert(
        decl,
        TaskNode::ObservationSpec {
            channel: CUBE_POSE.to_owned(),
            ty: pose.clone(),
        },
    );
    // Idempotent: a second run replaces this packet's own edges rather than doubling them.
    task.graph
        .edges
        .retain(|e| e.to.node != cat && e.to.node != decl);
    task.graph.edges.extend([
        Edge {
            from: PortRef::new(get, "pos"),
            to: PortRef::new(cat, "in0"),
        },
        Edge {
            from: PortRef::new(get, "quat"),
            to: PortRef::new(cat, "in1"),
        },
    ]);
    task.graph.connect(cat, "value", decl, "value");
    task.observation_spec.channels.insert(
        CUBE_POSE.to_owned(),
        ObsChannel {
            source: ObsSource::JointState {
                body: joint,
                dof: 7,
            },
            ty: pose.clone(),
        },
    );
    (joint, pose)
}

/// The privileged channel's name, shared by the Task IR channel, the Observation IR output
/// port, the Learning IR input and the policy contract (packet M5/V7a).
const CUBE_POSE: &str = "sim_cube_pose";

/// The half-width, in metres, of the `Normalize{Range}` on `sim_cube_pose` -- the arm's reach,
/// which contains the cube's draw and the bin's interior (packet M5/V7a).
const CUBE_POSE_RANGE: f64 = 0.3;

/// Regenerates `tests/fixtures/visible-learning/{task,observation}.toml` from the scene and
/// from each other, so no hash in them is ever typed in by hand. Run explicitly:
///
///     cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents
///
/// What it decides, rather than copies, is written out below: the bin's x span (which is the
/// success predicate), the shaped reward's range, and the image channel's dtype and layout.
#[test]
#[ignore = "fixture generator; run explicitly"]
fn regenerate_visible_learning_documents() {
    use es_ir::image::ImageDType;
    use es_ir::observation::{Io, ObservationNode};
    use es_ir::task::{CmpOp, TaskNode};
    use es_ir::types::{ElemType, Shape};
    // Spec 1.4: goldens and fixtures are CI read-only, and `cargo test -- --include-ignored`
    // runs every ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!(
            "SKIP regenerate_visible_learning_documents: set ES_GENERATE_GOLDENS=1 to regenerate"
        );
        return;
    }

    // The bin's interior in x, which is what the success predicate can see, and the cube's
    // draw. The draw stops 30 mm short of the bin's near wall in y: closer than that, the
    // gripper's own pad fouls the wall on the way down and the demonstration fails
    // (measured, packet M5/V1).
    let (bin_lo, bin_hi) = (0.09, 0.19);
    let (cube_y_lo, cube_y_hi) = (-0.03, 0.05);
    // The episode budget: what the scripted demonstration needs at the 50 Hz control rate,
    // with room for the settling the success predicate waits on (packet M5/V1), doubled by
    // packet M5/V14 because V13's three carrying episodes were still holding the cube over
    // the bin when 900 steps ran out, so the budget was hiding whether the policy ever
    // releases (design note section 7.22).
    let (max_steps, timeout_s) = (1800u32, 36.0);

    let xml_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mjcf/so101_pick_place.xml");
    let xml = std::fs::read(&xml_path).expect("the demo scene");
    let scene = es_assets::parse_mjcf(&String::from_utf8(xml.clone()).expect("utf-8"))
        .expect("the demo scene parses")
        .scene;

    // --- task.toml ---------------------------------------------------------------------
    let mut task =
        es_ir::serial::task_from_toml(&std::fs::read_to_string(vl_fixture("task.toml")).unwrap())
            .expect("task.toml parses");
    task.config.max_episode_steps = max_steps;
    task.scene.scene_hash = scene.scene_hash();
    task.scene.asset_hash = *blake3::hash(&xml).as_bytes();

    // The image channel as the renderer delivers it: U8, HWC.
    let hwc = |ty: &mut es_ir::types::PortType| {
        if let Some(image) = &ty.image {
            assert_eq!(image.dtype, ImageDType::U8, "the renderer produces Rgb8");
            ty.elem = ElemType::U8;
            ty.shape = Shape::new([
                u64::from(image.height),
                u64::from(image.width),
                match image.channels {
                    es_ir::image::ChannelFormat::Rgba => 4,
                    es_ir::image::ChannelFormat::Rgb | es_ir::image::ChannelFormat::Normal => 3,
                    es_ir::image::ChannelFormat::Flow => 2,
                    _ => 1,
                },
            ]);
        }
    };
    for node in task.graph.nodes.values_mut() {
        match node {
            TaskNode::GetSensor { ty, .. } | TaskNode::ObservationSpec { ty, .. } => hwc(ty),
            // The success predicate: the cube's x inside the bin's span.
            TaskNode::Compare {
                op: CmpOp::Ge,
                rhs: Some(v),
                ..
            } => *v = timeout_s,
            TaskNode::Compare {
                op: CmpOp::Gt,
                rhs: Some(v),
                ..
            } if *v > 0.0 && *v < 0.2 => *v = bin_lo,
            TaskNode::Compare {
                op: CmpOp::Lt,
                rhs: Some(v),
                ..
            } if *v > 0.1 && *v < 0.2 => *v = bin_hi,
            // The shaped reward: 0 at the bin, 1 out at the far edge of the draw.
            TaskNode::Normalize { lo, hi, .. } => {
                *lo = vec![bin_lo];
                *hi = vec![0.30];
            }
            TaskNode::Randomization { stream, dist, .. } if stream == "cube.y" => {
                *dist = es_ir::task::Distribution::Uniform {
                    lo: cube_y_lo,
                    hi: cube_y_hi,
                };
            }
            _ => {}
        }
    }
    for channel in task.observation_spec.channels.values_mut() {
        hwc(&mut channel.ty);
    }
    add_gripper_open_term(&mut task);
    let (cube_joint, cube_pose_ty) = add_privileged_cube_pose(&mut task, &scene);
    let diags = task.validate();
    assert!(diags.is_empty(), "{diags:?}");
    write(
        &vl_fixture("task.toml"),
        &format!(
            "{TASK_HEADER}\n{}",
            es_ir::serial::task_to_toml(&task).expect("task toml")
        ),
    );

    // --- observation.toml --------------------------------------------------------------
    let mut obs = es_ir::serial::observation_from_toml(
        &std::fs::read_to_string(vl_fixture("observation.toml")).unwrap(),
    )
    .expect("observation.toml parses");
    obs.task_ref = task.task_hash().expect("the task hashes");

    let image_node = *obs
        .graph
        .nodes
        .iter()
        .find(|(_, n)| matches!(n, ObservationNode::ImageInput { .. }))
        .expect("an ImageInput")
        .0;
    let mut raw = obs.graph.nodes[&image_node].io().output.clone();
    hwc(&mut raw);
    let mut dequantized = raw.clone();
    dequantized.elem = ElemType::F32;
    dequantized.shape = Shape::new([
        raw.shape.dims()[2],
        raw.shape.dims()[0],
        raw.shape.dims()[1],
    ]);
    if let Some(image) = &mut dequantized.image {
        image.dtype = ImageDType::F32;
    }
    // The consumer of the raw image becomes the consumer of the dequantized one.
    let sink = obs
        .graph
        .edges
        .iter()
        .find(|e| e.from.node == image_node)
        .expect("the image input feeds something")
        .to
        .clone();
    match obs.graph.nodes.get_mut(&image_node).expect("just found") {
        ObservationNode::ImageInput { io, .. } => io.output = raw.clone(),
        other => panic!("{other:?}"),
    }
    // Idempotent: a second run finds the `Dequantize` this one inserted and only re-types it.
    if let Some(ObservationNode::Dequantize { io }) = obs.graph.nodes.get_mut(&sink.node) {
        *io = Io::unary(raw, dequantized);
    } else {
        let deq =
            es_ir::graph::NodeId(obs.graph.nodes.keys().map(|n| n.0).max().expect("nodes") + 1);
        match obs.graph.nodes.get_mut(&sink.node).expect("the consumer") {
            ObservationNode::Normalize { io, .. } => io.inputs = vec![dequantized.clone()],
            other => panic!("the image input must feed a Normalize, not {other:?}"),
        }
        obs.graph.insert(
            deq,
            ObservationNode::Dequantize {
                io: Io::unary(raw, dequantized),
            },
        );
        obs.graph.edges.retain(|e| e.from.node != image_node);
        obs.graph.connect(image_node, "out", deq, "in0");
        obs.graph.connect(deq, "out", sink.node, &sink.port);
    }

    // --- the privileged state branch (packet M5/V7a) -----------------------------------
    //
    // `StateInput` names the cube's **free joint**, which is what makes this exact: with a
    // model loaded, `input_sources` resolves a source id against `ModelInfo.qpos` before it
    // falls back to the Task IR channel's leading-`dof` reading, so the port is served the
    // joint's own `qpos[6..13]` at inference and the same range of the recorded
    // `observation.state` row (`qpos || qvel`) when `es dataset bake` reads it.
    //
    // One `Normalize{Range}` covers all seven values, because `NormalizeStats::Range` is one
    // (lo, hi) per port. +-0.3 m is the arm's reach: the cube's draw (x in [0.21, 0.27]) and
    // the bin's interior (x in [0.09, 0.19], y in [-0.15, -0.05]) both sit inside it, and it
    // is the bound that keeps the *position* signal wide -- the 0.06 m draw spans 0.10 of the
    // output range here against 0.03 under the state branch's own +-1. The quaternion's four
    // values leave [0, 1] under it (w = 1 maps to 2.17); that is harmless and deliberate, a
    // `Normalize` is affine and not a clamp, and a box resting flat carries no signal there.
    let cube_raw = cube_pose_ty;
    let cube_norm = es_ir::types::PortType {
        unit: es_ir::types::Unit::Normalized {
            lo: -CUBE_POSE_RANGE,
            hi: CUBE_POSE_RANGE,
        },
        ..cube_raw.clone()
    };
    let (state_in, normalize) = (es_ir::graph::NodeId(5), es_ir::graph::NodeId(6));
    obs.graph.insert(
        state_in,
        ObservationNode::StateInput {
            source: cube_joint,
            io: Io::source(cube_raw.clone()),
        },
    );
    obs.graph.insert(
        normalize,
        ObservationNode::Normalize {
            stats: es_ir::observation::NormalizeStats::Range {
                lo: -CUBE_POSE_RANGE,
                hi: CUBE_POSE_RANGE,
            },
            io: Io::unary(cube_raw, cube_norm.clone()),
        },
    );
    // Idempotent: a second run replaces this packet's own edge rather than doubling it.
    obs.graph.edges.retain(|e| e.to.node != normalize);
    obs.graph.connect(state_in, "out", normalize, "in0");
    obs.outputs.insert(
        CUBE_POSE.to_owned(),
        es_ir::observation::ObservationOutput {
            port: es_ir::graph::PortRef::new(normalize, "out"),
            ty: cube_norm,
        },
    );

    let diags = obs.validate();
    assert!(diags.is_empty(), "{diags:?}");
    write(
        &vl_fixture("observation.toml"),
        &format!(
            "{OBSERVATION_HEADER}\n{}",
            es_ir::serial::observation_to_toml(&obs).expect("observation toml")
        ),
    );
    // --- observation-v8.toml, evaluation-v8.toml (packet M5/V8) ------------------------
    //
    // A second Observation IR on the same Task IR (spec 7: same `task_hash`, different
    // `observation_hash`), for a policy that was designed and trained outside this project.
    // Three differences, and each is forced by what LeRobot's ACT *is*:
    //
    //  * the privileged branch is gone. `lerobot.utils.feature_utils` gives a policy exactly
    //    one `observation.state` feature, and a second state port has nowhere to land.
    //  * the state `Normalize` is the **identity** — `Range{0..1}`, i.e. `(q - 0) / (1 - 0)`.
    //    ACT carries its own statistics in the checkpoint
    //    (`normalize_inputs.buffer_observation_state.{mean,std}`, fitted to the values it was
    //    trained on) and applies them as the first operation of its forward pass, so the
    //    conversion is already owned. Applying a *second*, unrelated affine map here would
    //    only change what those statistics had to absorb — and it would have to be applied to
    //    the exported dataset too, which means implementing `Op::Normalize` a second time,
    //    which is the defect section 7.9 removed. The node stays because spec 5.4 (`XIR`'s
    //    `TYPE-011`) requires a policy input to be `Normalized`, and the honest thing to say
    //    is that the Observation IR's normalization decision here *is* the identity.
    //  * the ports are named the way the checkpoint's `config.json` names its features, with
    //    dots replaced by underscores, because the name is what `XIR-010` matches and what
    //    reaches the lowered module as a `forward(**inputs)` keyword.
    //
    // The image branch is untouched, byte for byte: `Dequantize` then `Normalize{0..1}` is
    // already exactly the [0, 1] float CHW tensor LeRobot's own loader produces from a PNG —
    // and it, too, is the identity on that branch, which is why nothing had to change.
    let mut v8 = obs.clone();
    v8.outputs.remove(CUBE_POSE);
    for node in [state_in, normalize] {
        v8.graph.nodes.remove(&node);
    }
    v8.graph.edges.retain(|e| {
        ![state_in, normalize].contains(&e.from.node) && ![state_in, normalize].contains(&e.to.node)
    });
    let mut joint = v8.outputs.remove("joint_state").expect("the state output");
    let identity = es_ir::types::Unit::Normalized { lo: 0.0, hi: 1.0 };
    let state_norm = joint.port.node;
    match v8.graph.nodes.get_mut(&state_norm).expect("the state node") {
        ObservationNode::Normalize { stats, io } => {
            *stats = es_ir::observation::NormalizeStats::Range { lo: 0.0, hi: 1.0 };
            io.output.unit = identity.clone();
        }
        other => panic!("the state output must come from a Normalize, not {other:?}"),
    }
    joint.ty.unit = identity;
    v8.outputs.insert("observation_state".to_owned(), joint);
    let image = v8.outputs.remove("rgb_overhead").expect("the image output");
    v8.outputs
        .insert("observation_images_rgb_overhead".to_owned(), image);
    let diags = v8.validate();
    assert!(diags.is_empty(), "{diags:?}");
    write(
        &vl_fixture("observation-v8.toml"),
        &format!(
            "{OBSERVATION_V8_HEADER}\n{}",
            es_ir::serial::observation_to_toml(&v8).expect("observation-v8 toml")
        ),
    );
    let evaluation_v8 = demo_evaluation_ir(
        hex(&task.task_hash().expect("task hash")),
        hex(&v8.observation_hash().expect("observation-v8 hash")),
    );
    let diags = evaluation_v8.validate();
    assert!(diags.is_empty(), "{diags:?}");
    write(
        &vl_fixture("evaluation-v8.toml"),
        &format!(
            "{EVALUATION_V8_HEADER}\n{}",
            es_ir::serial::evaluation_to_toml(&evaluation_v8).expect("evaluation-v8 toml")
        ),
    );

    // --- evaluation.toml ----------------------------------------------------------------
    let evaluation = demo_evaluation_ir(
        hex(&task.task_hash().expect("task hash")),
        hex(&obs.observation_hash().expect("observation hash")),
    );
    let diags = evaluation.validate();
    assert!(diags.is_empty(), "{diags:?}");
    write(
        &vl_fixture("evaluation.toml"),
        &format!(
            "{EVALUATION_HEADER}\n{}",
            es_ir::serial::evaluation_to_toml(&evaluation).expect("evaluation toml")
        ),
    );

    println!(
        "task_hash {}\nobservation_hash {}\nevaluation_hash {}\nobservation_v8_hash \
         {}\nevaluation_v8_hash {}",
        hex(&task.task_hash().expect("task hash")),
        hex(&obs.observation_hash().expect("observation hash")),
        hex(&evaluation.evaluation_hash().expect("evaluation hash")),
        hex(&v8.observation_hash().expect("observation-v8 hash")),
        hex(&evaluation_v8.evaluation_hash().expect("evaluation-v8 hash"))
    );
}

const OBSERVATION_V8_HEADER: &str = "\
# Observation IR (spec 7) for the SO-101 cube-into-bin demo, for an **external** policy --
# packet M5/V8.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`
# from observation.toml, so no hash here is typed in. Same `task_ref`, different
# `observation_hash`: spec 7 says one Task IR may carry several Observation IRs, and this is
# what that is for.
#
# The policy this feeds is LeRobot's own ACT, trained by `lerobot-train` and imported with
# `es policy import-lerobot` -- not an IR-owned graph. Three things follow, and each is forced
# by what that checkpoint *is*:
#
#  1. **The state `Normalize` is the identity**: `Range{0..1}`, which is `(q - 0) / (1 - 0)`,
#     so the policy is served the joint angles in radians as the scene reports them. ACT
#     stores its own statistics in the checkpoint
#     (`normalize_inputs.buffer_observation_state.{mean,std}`, fitted to the values it was
#     trained on) and applies them as the first operation of its forward pass -- the
#     conversion is already owned. A second, unrelated affine map here would only change what
#     those statistics had to absorb, and it would have to be applied to the exported dataset
#     as well, which means implementing `Op::Normalize` a second time. That is the defect
#     section 7.9 removed. The node stays because spec 5.4 requires a policy input to be
#     `Normalized`, `Dimensionless` or `Token` (`TYPE-011`, checked against the contract by
#     the cross-IR pass), and what this port declares is exactly true: the Observation IR's
#     normalization decision for an externally-normalizing policy is the identity, stated in
#     the one node that decides it rather than left to a reader's inference.
#  2. **There is one state port.** `lerobot.utils.feature_utils.dataset_to_policy_features`
#     gives a policy exactly one `observation.state` feature, so V7a's simulator-privileged
#     `sim_cube_pose` has nowhere to land and is dropped. V8 is the *vision* question, which
#     is the one V7a set aside.
#  3. **The port names are the checkpoint's feature names**, with dots replaced by
#     underscores (a Python keyword argument may not contain one). `XIR-010` matches the
#     Observation IR's output names against `PolicyContract::inputs` verbatim, and those come
#     from `config.json`.
#
# The image branch is observation.toml's, byte for byte: `ImageInput` (U8 HWC, the renderer's
# own tile) -> `Dequantize` (CHW F32, /255) -> `Normalize{0..1}`. That is already exactly the
# tensor LeRobot's loader hands its policy from a PNG, so nothing had to be added -- and no
# `Resize` or `Crop` was added either, so no intrinsics transform is owed (spec 7.2, INV-14).
";

const EVALUATION_V8_HEADER: &str = "\
# Evaluation IR (spec 10) for the external-ACT demo -- packet M5/V8.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`.
# Identical to evaluation.toml in every suite, perturbation, metric and acceptance threshold --
# `success_rate >= 0.5` on seeds 101-116 and not one number lowered. The only thing that moves
# is `observation`, which names observation-v8.toml because that is the Observation IR the
# external policy is fed through (XIR-040).
";

const EVALUATION_HEADER: &str = "\
# Evaluation IR (spec 10) for the SO-101 cube-into-bin demo -- plan V, packet M5/V3.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`;
# `task` and `observation` are task.toml's and observation.toml's own hashes.
#
# 16 episodes on fixed seeds 101-116, deliberately outside the 1-50 the demonstrations were
# collected on (packet M5/V2), so the table is a held-out measurement and not a memory test.
# Sixteen because the demo video is a 4x4 grid and `Evaluation::run` gives every episode its
# own env (`BatchDomains::single_env()`), so a cell of that grid is an episode.
#
# The cube's initial pose is *not* a perturbation here: `PerturbationKind::ObjectPose` is
# `Unsupported` (`Env::reset` takes no state override) and does not need to be -- task.toml's
# own `Randomization` node moves the cube's free joint at every reset, in every suite, which
# is what spec 6.3 is for (design note section 2.7).
#
# The five perturbation suites are the kinds this build has kernels for and whose effect on a
# vision policy is worth a row:
#  * `light_intensity` / `light_direction` -- V3's two new kernels. The gain scales the scene's
#    colours before the `TriScene` upload, which is exactly a light gain for the `Rs` path's
#    Lambert term; the yaw turns `RenderConfig::light_dir`. Both need `--frames`; without a
#    renderer they are refused by name, never quietly skipped (spec 17.2).
#  * `observation_delay` -- one and two control steps at 50 Hz.
#  * `torque_noise`, `backlash` -- the actuator is not ideal.
# `color_temperature`, `occluder`, `camera_extrinsic`, `camera_intrinsic` and `object_pose`
# stay `Unsupported`; `docs/design/evaluation-execution.md` section 3 says why for each.
#
# Every suite draws from its own `stream`, so adding one of these rows cannot move another's
# numbers (spec 10.4). The acceptance criterion is on the nominal suite alone: a perturbed
# success rate is a measurement to report, not a gate to pass.
";

/// The demo's Evaluation IR: the §10.1 table plan V's video is a picture of.
fn demo_evaluation_ir(task: String, observation: String) -> es_ir::evaluation::EvaluationIr {
    use es_ir::evaluation::{
        AcceptanceCriterion, Comparator, EpisodeBatch, EvaluationIr, Perturbation,
        PerturbationKind, PerturbationSuite, Range, SeedPlan,
    };

    let suite = |name: &str, perturbations: Vec<Perturbation>| PerturbationSuite {
        name: name.to_owned(),
        perturbations,
    };
    EvaluationIr {
        schema_version: 1,
        task,
        observation,
        episodes: EpisodeBatch {
            n_episodes: 16,
            seeds: SeedPlan::Explicit((101..117).collect()),
        },
        suites: vec![
            suite("nominal", Vec::new()),
            suite(
                "light_intensity",
                vec![Perturbation::new(
                    PerturbationKind::LightIntensity {
                        range: Range::new(0.5, 1.5),
                        dist: es_ir::evaluation::Distribution::Uniform,
                    },
                    0,
                )],
            ),
            suite(
                "light_direction",
                vec![Perturbation::new(
                    PerturbationKind::LightDirection { range_deg: 45.0 },
                    1,
                )],
            ),
            suite(
                "observation_delay",
                vec![Perturbation::new(
                    PerturbationKind::ObservationDelay { ms: vec![20, 40] },
                    2,
                )],
            ),
            suite(
                "torque_noise",
                vec![Perturbation::new(
                    PerturbationKind::TorqueNoise { rel_sigma: 0.05 },
                    3,
                )],
            ),
            suite(
                "backlash",
                vec![Perturbation::new(
                    PerturbationKind::Backlash {
                        rad: Range::new(0.0, 0.01),
                    },
                    4,
                )],
            ),
        ],
        metrics: vec![
            es_ir::evaluation::MetricSpec::SuccessRate,
            es_ir::evaluation::MetricSpec::EpisodeLength,
            es_ir::evaluation::MetricSpec::EnvelopeViolationRate,
            es_ir::evaluation::MetricSpec::FailureModeHistogram,
        ],
        acceptance: vec![AcceptanceCriterion {
            suite: Some("nominal".to_owned()),
            metric: es_ir::evaluation::MetricSpec::SuccessRate,
            comparator: Comparator::Ge,
            threshold: 0.5,
            aggregation: es_ir::evaluation::Aggregation::Mean,
        }],
        augmentation: es_ir::evaluation::AugmentationPolicy::Disabled,
        replay: es_ir::evaluation::ReplayPolicy::default(),
    }
}

/// The demo's `policy.esb`: V0's four documents, packed. `--expert` never loads the weights,
/// so the byte the bundle carries for them is a placeholder, not a policy.
fn write_demo_bundle(dir: &Path) -> PathBuf {
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let mut learning =
        es_ir::serial::learning_from_toml(&read("learning.toml")).expect("learning.toml");
    let weights = b"es-v1-expert-placeholder".to_vec();
    learning.policy.weights = es_ir::learning::WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = es_compile::PolicyBundle::build(
        &es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the four demo documents pack into a bundle");
    let path = dir.join("policy.esb");
    std::fs::write(&path, bytes).expect("write policy.esb");
    path
}

fn demo_scene_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/so101_pick_place.xml")
}

/// The pinned seed set both expert oracles run on: `expert_solves_the_pinned_seeds` through
/// `es loop collect`, `expert_passes_the_evaluation_harness` through `es_eval::Evaluation`.
/// Same seeds and same threshold on both paths is the point of packet M5/V6 -- a harness the
/// expert fails is a harness no policy can pass (design note section 7.12).
const SEEDS: [u64; 8] = [1, 2, 3, 5, 8, 13, 21, 34];

/// Rows of each chunk the demo executes before the policy is asked again: the Deployment IR's
/// `rate.control / rate.inference`, capped at `action.execute_chunk` (packet M5/V17). The one
/// definition `es loop collect` and `es_eval::runner` both drive the expert with.
fn demo_replan(deploy: &es_ir::deployment::DeploymentIr) -> u32 {
    es_env::replan_interval(deploy.rate)
        .expect("the demo's inference rate divides its control rate")
        .min(deploy.action.execute_chunk as u64) as u32
}

/// Fraction of the pinned seeds that must end in `Success`. A property of the expert, not a
/// tuning knob: lowering it to make a change pass is the same as editing a golden.
const THRESHOLD: f64 = 0.875;

/// How many frames of every demonstration must command the gripper open past the stall value
/// (packet M5/V15). The same kind of pin as `THRESHOLD`: at the old 0.6 predicate threshold
/// this was 1, which is what "the policy never opens the hand" looks like from the data's side.
const RELEASE_FRAMES: usize = 5;

/// Packet M5/V1 oracle 1 -- the expert's success rate over a pinned seed set.
///
/// The threshold is a property of the expert, not a tuning knob: lowering it to make a change
/// pass is the same as editing a golden. Measured on the oracle server, 2026-09-15: 8 of 8
/// seeds succeed, and every one of them puts the cube inside the bin's **three-dimensional**
/// interior -- which the Task IR's own predicate cannot check, because a cone leaf is one
/// scalar (design note section 5.4). The y and z here come straight from the recorded
/// `observation.state`, so the rate reported below is honest even though the IR predicate is
/// narrower than it.
#[test]
fn expert_solves_the_pinned_seeds() {
    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP expert_success: {reason}");
        return;
    }
    let dir = scratch_dir("expert-success");
    let policy = write_demo_bundle(&dir);

    let mut succeeded = 0usize;
    let mut in_the_bin = 0usize;
    for seed in SEEDS {
        let out = dir.join(format!("ds{seed}"));
        let run = bin()
            .args([
                "loop",
                "collect",
                "--expert",
                "so101-pick-place",
                "--policy",
            ])
            .arg(&policy)
            .arg("--scene")
            .arg(demo_scene_path())
            .args(["--episodes", "1", "--seed", &seed.to_string()])
            .arg("--out")
            .arg(&out)
            .output()
            .expect("run es loop collect --expert");
        let text = stdout(&run);
        assert_eq!(
            run.status.code(),
            Some(0),
            "seed {seed}\nstdout:\n{text}\nstderr:\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let line = text
            .lines()
            .find(|l| l.starts_with("terminations:"))
            .unwrap_or_else(|| panic!("no termination summary in\n{text}"));
        if line.contains("success 1") {
            succeeded += 1;
        }

        // The honest check: where the cube actually ended, read back out of the dataset.
        let dataset = es_data::LeRobotDataset::open(&out).expect("the demonstration opens");
        let ep = dataset.read_episode(0).expect("episode 0");
        let state = match ep.columns.get("observation.state") {
            Some(es_data::Column::F32(v)) => v.clone(),
            other => panic!("observation.state: {other:?}"),
        };
        // `observation.state` is the `qpos ‖ qvel` row; the cube's free joint starts at
        // qpos[6] and its first three values are its world position.
        let width = state.len() / ep.len();
        let last = &state[(ep.len() - 1) * width..];
        let (x, y, z) = (f64::from(last[6]), f64::from(last[7]), f64::from(last[8]));
        let inside = (0.09..0.19).contains(&x) && (-0.15..-0.05).contains(&y) && z < 0.09;
        if inside {
            in_the_bin += 1;
        }

        // Packet M5/V15: **the release has to be in the data.** The episode ends the instant
        // the success predicate goes true, so a threshold set too low truncates the expert's
        // opening ramp and the demonstration teaches a policy to hold. Counted on the recorded
        // *commands*, over the frames that follow the last one commanding the jaw shut.
        let action = match ep.columns.get("action") {
            Some(es_data::Column::F32(v)) => v.clone(),
            other => panic!("action: {other:?}"),
        };
        let nj = action.len() / ep.len();
        let grip: Vec<f64> = (0..ep.len())
            .map(|f| f64::from(action[f * nj + nj - 1]))
            .collect();
        let tail = grip.iter().rposition(|g| *g < 0.0).map_or(0, |k| k + 1);
        let opening = grip[tail..].iter().filter(|g| **g > 0.6).count();
        println!(
            "seed {seed}: {} steps, {line}   cube ({x:.3}, {y:.3}, {z:.3}) inside={inside}   \
             release: {} frames after the last closed command, {opening} of them above 0.6, \
             peak {:.3}",
            ep.len(),
            grip.len() - tail,
            grip[tail..].iter().fold(0.0f64, |a, g| a.max(*g)),
        );
        assert!(
            opening >= RELEASE_FRAMES,
            "seed {seed}'s demonstration ends {opening} frames into the release, below the \
             {RELEASE_FRAMES} this test pins: the recorded actions do not contain a hand that \
             opens, so nothing trained on them can learn to let go (packet M5/V15)"
        );
    }

    let rate = succeeded as f64 / SEEDS.len() as f64;
    println!(
        "RAN expert_success: {succeeded}/{} seeds ended in Success, {in_the_bin}/{} put the \
         cube inside the bin's 3-D interior",
        SEEDS.len(),
        SEEDS.len()
    );
    assert!(
        rate >= THRESHOLD,
        "the expert solved {succeeded}/{} pinned seeds ({rate:.3}), below the {THRESHOLD} this \
         test pins. Lowering the threshold is editing a golden.",
        SEEDS.len()
    );
    assert_eq!(
        in_the_bin, succeeded,
        "the Task IR called an episode a success whose cube is not in the bin: the predicate \
         reads x alone, and this is what it misses"
    );
}

/// Packet M5/V6 oracle -- **the harness passes the expert**, on the same pinned seeds and the
/// same threshold as `expert_solves_the_pinned_seeds` above.
///
/// This is the test that was missing. Until V6, `es_eval::runner` re-seeded the Safety Plane
/// from the measured joints before every `validate`, so `velocity_max` / `acceleration_max` /
/// the action-rate limits bounded the servo's *following error* rather than the plane's own
/// commands; the bound collapsed to `acceleration_max * dt^2 = 0.008` rad and the scripted
/// expert -- which paces itself to exactly what the Deployment IR allows and passes 50/50
/// through `es loop collect` -- scored **0 of 16** through `es eval run`. No policy can pass a
/// harness the expert fails, so every evaluation number the demo has ever reported was taken
/// against a ceiling of zero (design note `docs/design/visible-learning.md` section 7.12).
///
/// **The server oracle.** Only `MuJoCoCpuBackend` can drive the SO-101 scene, so this is named
/// here and run on the oracle server; without `mujoco` it prints a reason and skips, exactly
/// like the collection oracle it mirrors.
///
/// Two deliberate narrowings, both of which leave the measurement honest:
///
/// * **One `Evaluation::run` per seed, one episode each.** `Evaluation::run` seeds one `Env`
///   from `seeds[0]` and keys the Task IR's own randomization by an episode counter, so a
///   single 8-episode cell would be eight draws of *one* seed. Eight single-episode runs are
///   eight draws of the eight pinned seeds, which is what the collection oracle measures.
/// * **A constant frame.** The demo's Observation IR carries a 96x96 `Rgb8` input, and the
///   runner refuses an image input it cannot serve rather than zero-filling one. The expert
///   reads joints, never pixels (`ScriptedExpert::chunk` takes a `StateView`), so the frame
///   cannot change its decisions -- and serving a constant one is what lets this oracle need
///   `mujoco` and not also a Vulkan device. `visible_learning_demo_run` is the run that
///   renders for real.
#[test]
fn expert_passes_the_evaluation_harness() {
    use std::cell::RefCell;
    use std::rc::Rc;

    const NJ: usize = 6;
    const H: usize = 16;

    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP expert_passes_the_evaluation_harness: {reason}");
        return;
    }
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml");
    let obs =
        es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml");
    let deploy =
        es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml");
    // The document's declared inference latency, which since packet M7/T7 the evaluator honours
    // exactly as `es loop collect` always has: 15 ms is one control tick at 50 Hz, so tick 0 of
    // every episode is a chunk underrun the plane answers for and the expert drives from tick 1.
    let latency_ms = es_ir::serial::learning_from_toml(&read("learning.toml"))
        .expect("learning.toml")
        .policy
        .contract
        .runtime
        .expected_latency_ms;
    let scene = es_assets::parse_mjcf(
        &std::fs::read_to_string(demo_scene_path()).expect("the demo scene is in the repo"),
    )
    .expect("the demo scene parses")
    .scene;
    let cube = scene
        .joints
        .iter()
        .find(|j| j.kind == es_assets::scene::JointKind::Free)
        .expect("the scene has one free-joint body to pick up")
        .id;
    // The image the plan declares, once: 96x96 Rgb8, HWC.
    let blank = vec![0u8; 96 * 96 * 3];

    let mut succeeded = 0usize;
    let mut worst_violation = 0.0f64;
    for seed in SEEDS {
        let mut cfg = es_env::expert::demo_cfg(cube);
        // `es loop collect`'s own pacing, which since packet M5/V17 is also the evaluation
        // runner's: both replan every `rate.control / rate.inference` control ticks and
        // execute the chunk's rows in between.
        cfg.pace_to(&deploy, demo_replan(&deploy));
        let expert = es_env::expert::ScriptedExpert::new(&scene, cfg).expect("the expert builds");
        let seen: SeenState = Rc::new(RefCell::new(None));

        let mut ir = demo_evaluation_ir(
            hex(&task.task_hash().expect("task hash")),
            hex(&obs.observation_hash().expect("observation hash")),
        );
        ir.episodes = es_ir::evaluation::EpisodeBatch {
            n_episodes: 1,
            seeds: es_ir::evaluation::SeedPlan::Explicit(vec![seed]),
        };
        ir.suites.truncate(1);
        assert_eq!(ir.suites[0].name, "nominal");

        let mut policy = ExpertPolicy::<NJ, H> {
            expert,
            seen: Rc::clone(&seen),
            calls: std::rc::Rc::default(),
        };
        let taken = Rc::clone(&seen);
        let blank_for_source = blank.clone();
        let mut frames =
            move |_light: &es_eval::LightOverride,
                  model: &es_physics_core::backend::ModelInfo,
                  state: &es_physics_core::backend::StateView<'_>| {
                let mut row = state.qpos_of(0).to_vec();
                row.extend_from_slice(state.qvel_of(0));
                *taken.borrow_mut() = Some((model.clone(), row));
                Ok::<Vec<u8>, String>(blank_for_source.clone())
            };
        let (report, _lock) =
            es_eval::Evaluation::run_with_frames::<es_physics_backend::MuJoCoCpuBackend, _, NJ, H>(
                &ir,
                &task,
                &scene,
                &obs,
                &mut policy,
                &deploy,
                es_physics_backend::MuJoCoCpuBackend::new,
                &es_eval::RunConfig {
                    expected_latency_ms: latency_ms,
                    ..es_eval::RunConfig::default()
                },
                Some(&mut frames),
                None,
            )
            .expect("the expert runs through the evaluation harness");

        let metric = |m: es_ir::evaluation::MetricSpec| {
            report
                .cells
                .iter()
                .find(|c| c.suite == "nominal" && c.metric == m)
                .and_then(|c| match c.value {
                    es_ir::evaluation::MetricValue::Scalar(v) => Some(v),
                    _ => None,
                })
                .unwrap_or(f64::NAN)
        };
        let rate = metric(es_ir::evaluation::MetricSpec::SuccessRate);
        let violation = metric(es_ir::evaluation::MetricSpec::EnvelopeViolationRate);
        let length = metric(es_ir::evaluation::MetricSpec::EpisodeLength);
        worst_violation = worst_violation.max(violation);
        if rate > 0.0 {
            succeeded += 1;
        }
        println!(
            "seed {seed}: success_rate {rate:.4}  envelope_violation_rate {violation:.4}  \
             episode_length {length:.1}"
        );
    }

    let rate = succeeded as f64 / SEEDS.len() as f64;
    println!(
        "RAN expert_passes_the_evaluation_harness: {succeeded}/{} pinned seeds ended in \
         Success through es-eval, worst envelope_violation_rate {worst_violation:.4}",
        SEEDS.len()
    );
    assert!(
        rate >= THRESHOLD,
        "the evaluation harness passed the expert on {succeeded}/{} pinned seeds ({rate:.3}), \
         below the {THRESHOLD} `expert_solves_the_pinned_seeds` measures on the collection \
         path. The two paths run the same expert through the same Deployment IR and must \
         agree; lowering this threshold is editing a golden.",
        SEEDS.len()
    );
    // **Re-pinned by the phase-2 measurement (design note section 7.13).** Phase 1 asserted
    // `< 0.02` here, on the assumption that the expert's paced ramp reaches the plane as the
    // expert emitted it. It does not, and must not: V6b routed this path through the
    // collector's `ChunkBuffer`, so what the plane judges is the **temporal ensemble** of the
    // sixteen overlapping chunks (`deployment.toml`'s `execution.temporal_ensemble`), whose
    // tick-to-tick delta moves by more than `acceleration_max * dt^2` even when every chunk
    // inside it is paced. Measured: 0.48-0.55 through evaluation, against 0.63 of the frames
    // in V1c's own collected set -- the same phenomenon, on both paths, which is what V6 was
    // for. `deployment.toml` has said so since V1: "a demonstration that asks for a pose the
    // arm has not reached yet is clamped on most ticks by construction, not by anomaly".
    //
    // So the bound that means something is the Deployment IR's own: the run must stay under
    // the `EnvelopeViolationRate` watchdog's `max_frac`, because above it the plane latches
    // the fallback and the expert stops driving (spec 9.4). That is read out of the document
    // rather than typed here.
    let max_frac = deploy
        .watchdogs
        .0
        .iter()
        .find_map(|w| match w {
            es_ir::deployment::Watchdog::EnvelopeViolationRate { max_frac, .. } => Some(*max_frac),
            _ => None,
        })
        .expect("the demo declares an envelope-violation watchdog");
    assert!(
        worst_violation < max_frac,
        "the plane corrected {worst_violation:.4} of the expert's steps, at or past the \
         {max_frac} the Deployment IR's own `EnvelopeViolationRate` watchdog latches on -- \
         the expert would have been driving the fallback, not the task (spec 9.4)"
    );
}

/// Packet M5/V6b (b) -- **`--seed S` names the same scene on both paths**, bit for bit.
///
/// The Task IR's `Randomization` node draws from `(seed, env, episode)` (§6.3), so episode 0 of
/// seed `S` is one specific cube pose. `es loop collect` sees draw 0 of it. `es eval run` used
/// to reset a second time at the top of `run_episode` and see draw 1, so every A/B between a
/// collected demonstration and an evaluated episode compared two different scenes -- silently,
/// because both are valid poses.
///
/// This runs both paths on the demo scene with the same seed and compares the `qpos` each one
/// first hands its policy, as raw `f64` bits. The collector's policy is handed the `qpos ‖ qvel`
/// row directly (`es_env::domains::state_row`, the plan-free path the demonstrations are
/// collected through); the evaluation runner's frame source is handed the same `StateView`
/// immediately before its policy is called. Two different hooks onto the same number.
///
/// **The server oracle** -- the SO-101 scene needs `MuJoCoCpuBackend`, like the two expert
/// oracles above; without `mujoco` it prints a reason and skips. `es-eval`'s
/// `episode_zero_runs_on_the_first_randomization_draw` and `es-data`'s
/// `the_collector_resets_once_per_episode_and_never_before_the_first_observation` are the local
/// halves that run in every CI tier.
#[test]
fn collection_and_evaluation_draw_the_same_scene_for_a_seed() {
    use std::cell::RefCell;
    use std::rc::Rc;

    const NJ: usize = 6;
    const H: usize = 16;
    const SEED: u64 = 1;

    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP collection_and_evaluation_draw_the_same_scene_for_a_seed: {reason}");
        return;
    }
    let dir = scratch_dir("seed-parity");
    let policy_path = write_demo_bundle(&dir);
    let bundle_bytes = std::fs::read(&policy_path).expect("policy.esb");
    let bundle = es_compile::PolicyBundle::open(&bundle_bytes).expect("the demo bundle opens");
    let scene = es_assets::parse_mjcf(
        &std::fs::read_to_string(demo_scene_path()).expect("the demo scene is in the repo"),
    )
    .expect("the demo scene parses")
    .scene;

    // --- the collection path: the first observation its policy is handed -------------------
    let collected: Rc<RefCell<Option<Vec<f64>>>> = Rc::new(RefCell::new(None));
    let mut recorder = FirstObservation {
        seen: Rc::clone(&collected),
    };
    let out = dir.join("ds");
    es_data::Collector::run::<es_physics_backend::MuJoCoCpuBackend, _, NJ, H>(
        &es_data::CollectSpec {
            bundle: &bundle,
            scene: &scene,
            out_root: &out,
            traj_dir: None,
            n_episodes: 1,
            seed: SEED,
            max_steps: 4,
        },
        &mut recorder,
        es_physics_backend::MuJoCoCpuBackend::new,
        &mut |_, _, _, _| es_data::Intervention::Policy,
        None,
    )
    .expect("the demo collects one short episode");

    // --- the evaluation path: the state its frame source is handed first -------------------
    let evaluated: Rc<RefCell<Option<Vec<f64>>>> = Rc::new(RefCell::new(None));
    let taken = Rc::clone(&evaluated);
    let blank = vec![0u8; 96 * 96 * 3];
    let mut frames = move |_light: &es_eval::LightOverride,
                           _model: &es_physics_core::backend::ModelInfo,
                           state: &es_physics_core::backend::StateView<'_>| {
        let mut slot = taken.borrow_mut();
        if slot.is_none() {
            let mut row = state.qpos_of(0).to_vec();
            row.extend_from_slice(state.qvel_of(0));
            *slot = Some(row);
        }
        Ok::<Vec<u8>, String>(blank.clone())
    };
    let mut ir = demo_evaluation_ir(
        hex(&bundle.task.task_hash().expect("task hash")),
        hex(&bundle.observation.observation_hash().expect("obs hash")),
    );
    ir.episodes = es_ir::evaluation::EpisodeBatch {
        n_episodes: 1,
        seeds: es_ir::evaluation::SeedPlan::Explicit(vec![SEED]),
    };
    ir.suites.truncate(1);
    let mut policy = FirstObservation {
        seen: Rc::new(RefCell::new(None)),
    };
    es_eval::Evaluation::run_with_frames::<es_physics_backend::MuJoCoCpuBackend, _, NJ, H>(
        &ir,
        &bundle.task,
        &scene,
        &bundle.observation,
        &mut policy,
        &bundle.deployment,
        es_physics_backend::MuJoCoCpuBackend::new,
        &es_eval::RunConfig {
            max_steps: Some(4),
            ..es_eval::RunConfig::default()
        },
        Some(&mut frames),
        None,
    )
    .expect("the demo evaluates one short episode");

    let a = collected.borrow().clone().expect("the collector observed");
    let b = evaluated.borrow().clone().expect("the runner observed");
    assert_eq!(a.len(), b.len(), "the two rows are different widths");
    // At `f32`, because that is the width the collector's own observation has: the plan-free
    // path hands the policy an `f32` tensor (`es_env::domains::state_row`) and the dataset
    // stores `observation.state` as `f32`, while the runner is handed the backend's `f64`
    // state directly. Comparing raw `f64` bits compares two encodings of one number --
    // measured, phase 2: `0.2550719976425171` against `0.255071989355131`, the same draw
    // rounded twice. `f32` is the precision at which the two paths are the same object, and
    // it is the precision every demonstration is recorded at.
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let (x, y) = (*x as f32, *y as f32);
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "element {i} of episode 0's first state differs between `es loop collect` ({x}) \
             and `es eval run` ({y}) for seed {SEED}: the two paths are not drawing the same \
             scene (packet M5/V6b)"
        );
    }
    println!(
        "RAN collection_and_evaluation_draw_the_same_scene_for_a_seed: {} values identical, \
         cube at ({:.4}, {:.4}, {:.4})",
        a.len(),
        a[6],
        a[7],
        a[8]
    );
}

/// Packet M5/V17 oracle (b) -- **the two paths ask the policy at the same cadence**, and the
/// one thing that still separates them is named.
///
/// V6b made `es_eval::runner` feed the Safety Plane through the collector's own
/// `es_env::plane_chunk`; what it left behind is that both loops asked for a fresh chunk on
/// every control tick, ignoring the Deployment IR's `rate.inference`. V17 gives both the one
/// rule (`es_env::replan_interval`), and this is the test that it is *the same* rule: the same
/// scripted expert, the same seed, the same documents, and the policy invoked
/// `ceil(steps / replan)` times on each path.
///
/// **What it also measures.** The per-tick `qpos ‖ qvel` each path hands its own state hook --
/// the collector's `FrameSink` and the runner's frame source, both called at the top of a
/// control step with the backend's `f64` state (packet M5/V12). Under V17 these were *not*
/// bit-identical and the first divergent tick was only printed: `es loop collect` applied the
/// Learning IR's `RuntimeHints::expected_latency_ms` (15 ms, one control tick at 50 Hz) through
/// `AsyncInference`, so its first chunk reached the plane at tick 1 and tick 0 was a recorded
/// underrun, while `es_eval::runner` had no latency model and executed row 0 at tick 0 (design
/// note section 7.25, open question 24). Packet M7/T7 gave the evaluator that same model, so
/// the traces now agree to the last tick and this asserts it.
///
/// The sibling `collection_and_evaluation_draw_the_same_scene_for_a_seed` pins the first state.
/// **The server oracle** -- it needs `mujoco`.
#[test]
fn collection_and_evaluation_ask_the_policy_at_the_same_cadence() {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    const NJ: usize = 6;
    const H: usize = 16;
    const SEED: u64 = 1;
    /// Long enough to cover the expert's approach, the descent and the grasp, and a whole
    /// number of re-plan periods so the expected call count has no rounding in it.
    const STEPS: u32 = 120;

    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP collection_and_evaluation_ask_the_policy_at_the_same_cadence: {reason}");
        return;
    }
    let dir = scratch_dir("chunk-parity");
    let bundle_bytes = std::fs::read(write_demo_bundle(&dir)).expect("policy.esb");
    let bundle = es_compile::PolicyBundle::open(&bundle_bytes).expect("the demo bundle opens");
    let scene = es_assets::parse_mjcf(
        &std::fs::read_to_string(demo_scene_path()).expect("the demo scene is in the repo"),
    )
    .expect("the demo scene parses")
    .scene;
    let cube = scene
        .joints
        .iter()
        .find(|j| j.kind == es_assets::scene::JointKind::Free)
        .expect("the scene has one free-joint body to pick up")
        .id;
    let deploy = &bundle.deployment;
    let replan = demo_replan(deploy);
    let expert = || {
        let mut cfg = es_env::expert::demo_cfg(cube);
        cfg.pace_to(deploy, replan);
        es_env::expert::ScriptedExpert::new(&scene, cfg).expect("the expert builds")
    };
    let row_of = |state: &es_physics_core::backend::StateView<'_>| {
        let mut row = state.qpos_of(0).to_vec();
        row.extend_from_slice(state.qvel_of(0));
        row
    };
    let expected = u64::from(STEPS).div_ceil(u64::from(replan));

    // --- the collection path -----------------------------------------------------------------
    let collected: Rc<RefCell<Vec<Vec<f64>>>> = Rc::new(RefCell::new(Vec::new()));
    let seen: SeenState = Rc::new(RefCell::new(None));
    let collect_calls: Rc<Cell<u64>> = Rc::default();
    let mut policy = ExpertPolicy::<NJ, H> {
        expert: expert(),
        seen: Rc::clone(&seen),
        calls: Rc::clone(&collect_calls),
    };
    {
        let (trace, seen) = (Rc::clone(&collected), Rc::clone(&seen));
        let mut sink = |model: &es_physics_core::backend::ModelInfo,
                        state: &es_physics_core::backend::StateView<'_>| {
            let row = row_of(state);
            *seen.borrow_mut() = Some((model.clone(), row.clone()));
            trace.borrow_mut().push(row);
            Ok::<(), String>(())
        };
        es_data::Collector::run::<es_physics_backend::MuJoCoCpuBackend, _, NJ, H>(
            &es_data::CollectSpec {
                bundle: &bundle,
                scene: &scene,
                out_root: &dir.join("ds"),
                traj_dir: None,
                n_episodes: 1,
                seed: SEED,
                max_steps: STEPS,
            },
            &mut policy,
            es_physics_backend::MuJoCoCpuBackend::new,
            &mut |_, _, _, _| es_data::Intervention::Policy,
            Some(&mut sink),
        )
        .expect("the demo collects one episode");
    }

    // --- the evaluation path -------------------------------------------------------------------
    let evaluated: Rc<RefCell<Vec<Vec<f64>>>> = Rc::new(RefCell::new(Vec::new()));
    let seen: SeenState = Rc::new(RefCell::new(None));
    let eval_calls: Rc<Cell<u64>> = Rc::default();
    let mut policy = ExpertPolicy::<NJ, H> {
        expert: expert(),
        seen: Rc::clone(&seen),
        calls: Rc::clone(&eval_calls),
    };
    let mut ir = demo_evaluation_ir(
        hex(&bundle.task.task_hash().expect("task hash")),
        hex(&bundle.observation.observation_hash().expect("obs hash")),
    );
    ir.episodes = es_ir::evaluation::EpisodeBatch {
        n_episodes: 1,
        seeds: es_ir::evaluation::SeedPlan::Explicit(vec![SEED]),
    };
    ir.suites.truncate(1);
    assert_eq!(ir.suites[0].name, "nominal");
    {
        let (trace, seen) = (Rc::clone(&evaluated), Rc::clone(&seen));
        let blank = vec![0u8; 96 * 96 * 3];
        let mut frames =
            move |_light: &es_eval::LightOverride,
                  model: &es_physics_core::backend::ModelInfo,
                  state: &es_physics_core::backend::StateView<'_>| {
                let row = row_of(state);
                *seen.borrow_mut() = Some((model.clone(), row.clone()));
                trace.borrow_mut().push(row);
                Ok::<Vec<u8>, String>(blank.clone())
            };
        es_eval::Evaluation::run_with_frames::<es_physics_backend::MuJoCoCpuBackend, _, NJ, H>(
            &ir,
            &bundle.task,
            &scene,
            &bundle.observation,
            &mut policy,
            deploy,
            es_physics_backend::MuJoCoCpuBackend::new,
            &es_eval::RunConfig {
                max_steps: Some(STEPS),
                expected_latency_ms: demo_latency_ms(&bundle),
                ..es_eval::RunConfig::default()
            },
            Some(&mut frames),
            None,
        )
        .expect("the demo evaluates one episode");
    }

    let (a, b) = (collected.borrow(), evaluated.borrow());
    assert_eq!(
        a.len(),
        STEPS as usize,
        "the collection ran the whole budget"
    );
    assert_eq!(
        b.len(),
        STEPS as usize,
        "the evaluation ran the whole budget"
    );
    assert_eq!(
        (collect_calls.get(), eval_calls.get()),
        (expected, expected),
        "{STEPS} control ticks at one re-plan every {replan} is {expected} policy calls on each \
         path; `es loop collect` made {} and `es eval run` {}. The two paths must execute chunks \
         identically (packet M5/V6b, M5/V17)",
        collect_calls.get(),
        eval_calls.get()
    );
    let diverged = a
        .iter()
        .zip(b.iter())
        .position(|(x, y)| x.iter().zip(y).any(|(x, y)| x.to_bits() != y.to_bits()));
    println!(
        "RAN collection_and_evaluation_ask_the_policy_at_the_same_cadence: {expected} policy \
         calls on each path over {STEPS} control ticks (one every {replan}); the state traces \
         first differ at tick {diverged:?} -- `None` since packet M7/T7 gave the evaluator the \
         collector's own latency model (design note sections 7.25 and 7.30)"
    );
    assert_eq!(
        diverged, None,
        "the two state traces must agree to the last tick; the sibling \
         `collection_and_evaluation_draw_the_same_trajectory` compares the `.estraj` files and \
         names the values (packet M7/T7)"
    );
}

/// The bundle's declared inference latency, which `es eval run` reads out of the same field
/// (`crates/es/src/cmd/eval.rs`) and `DomainRunner::new` reads straight off the contract.
///
/// `Evaluation::run` is handed the four IRs it judges and never the `LearningGraph`, so the
/// number crosses on `RunConfig`; there is still one latency *model*, `es_env::latency_ticks`,
/// and both paths call it with this value and `rate.control` (packet M7/T7).
fn demo_latency_ms(bundle: &es_compile::PolicyBundle) -> f32 {
    bundle.learning.policy.contract.runtime.expected_latency_ms
}

/// Packet M7/T7 oracle 2 -- **the two paths draw the same trajectory**, not merely the same
/// scene and the same cadence: the `.estraj` each one writes for one seed is equal tick by
/// tick, to the last tick, as raw `f64` bits.
///
/// This is the oracle V6b and V17 were missing. V6b made both paths feed the plane through
/// `es_env::plane_chunk` and V17 made both honour `rate.inference`, and the sibling
/// `collection_and_evaluation_ask_the_policy_at_the_same_cadence` pinned the *schedule*; what
/// was left was that `es loop collect` executed a chunk at `submit_tick + latency` while
/// `es_eval::runner` executed row 0 in the tick it inferred. The `qpos ‖ qvel` traces of one
/// seed therefore agreed at tick 0 and diverged at tick 1, which is what open question 24
/// named and packet M7/T7 fixed. **Measured before the fix** -- SO-101 demo scene, seed 1, the
/// scripted expert, `expected_latency_ms = 15` (one control tick at 50 Hz): see design note
/// section 7.30.
///
/// Both files are written by `es_env::traj::Trajectory` from the state at the top of a control
/// step, before `Env::step` (packet M5/V12), so the comparison is of the same instant on both
/// paths and not of two conventions.
///
/// **The server oracle** -- the SO-101 scene needs `MuJoCoCpuBackend`; without `mujoco` it
/// prints a reason and skips. Ignored by default because it runs two full episodes of physics:
///
/// ```text
/// cargo test -p es --test cli collection_and_evaluation_draw_the_same_trajectory -- --ignored
/// ```
#[test]
#[ignore = "runs two full demo episodes through MuJoCo; the T7 server oracle"]
fn collection_and_evaluation_draw_the_same_trajectory() {
    use std::cell::RefCell;
    use std::rc::Rc;

    const NJ: usize = 6;
    const H: usize = 16;
    const SEED: u64 = 1;
    /// A whole number of re-plan periods, long enough to cover approach, descent and grasp.
    const STEPS: u32 = 120;

    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP collection_and_evaluation_draw_the_same_trajectory: {reason}");
        return;
    }
    let dir = scratch_dir("traj-parity");
    let bundle_bytes = std::fs::read(write_demo_bundle(&dir)).expect("policy.esb");
    let bundle = es_compile::PolicyBundle::open(&bundle_bytes).expect("the demo bundle opens");
    let scene = es_assets::parse_mjcf(
        &std::fs::read_to_string(demo_scene_path()).expect("the demo scene is in the repo"),
    )
    .expect("the demo scene parses")
    .scene;
    let cube = scene
        .joints
        .iter()
        .find(|j| j.kind == es_assets::scene::JointKind::Free)
        .expect("the scene has one free-joint body to pick up")
        .id;
    let deploy = &bundle.deployment;
    let replan = demo_replan(deploy);
    let expert = || {
        let mut cfg = es_env::expert::demo_cfg(cube);
        cfg.pace_to(deploy, replan);
        es_env::expert::ScriptedExpert::new(&scene, cfg).expect("the expert builds")
    };
    let seen: SeenState = Rc::new(RefCell::new(None));
    let row_of = |state: &es_physics_core::backend::StateView<'_>| {
        let mut row = state.qpos_of(0).to_vec();
        row.extend_from_slice(state.qvel_of(0));
        row
    };

    // --- the collection path ---------------------------------------------------------------
    let collect_traj = dir.join("collect-traj");
    {
        let mut policy = ExpertPolicy::<NJ, H> {
            expert: expert(),
            seen: Rc::clone(&seen),
            calls: Rc::default(),
        };
        let seen = Rc::clone(&seen);
        let mut sink = |model: &es_physics_core::backend::ModelInfo,
                        state: &es_physics_core::backend::StateView<'_>| {
            *seen.borrow_mut() = Some((model.clone(), row_of(state)));
            Ok::<(), String>(())
        };
        es_data::Collector::run::<es_physics_backend::MuJoCoCpuBackend, _, NJ, H>(
            &es_data::CollectSpec {
                bundle: &bundle,
                scene: &scene,
                out_root: &dir.join("ds"),
                traj_dir: Some(collect_traj.clone()),
                n_episodes: 1,
                seed: SEED,
                max_steps: STEPS,
            },
            &mut policy,
            es_physics_backend::MuJoCoCpuBackend::new,
            &mut |_, _, _, _| es_data::Intervention::Policy,
            Some(&mut sink),
        )
        .expect("the demo collects one episode");
    }

    // --- the evaluation path -----------------------------------------------------------------
    let eval_traj = dir.join("eval-traj");
    {
        let mut policy = ExpertPolicy::<NJ, H> {
            expert: expert(),
            seen: Rc::clone(&seen),
            calls: Rc::default(),
        };
        let mut ir = demo_evaluation_ir(
            hex(&bundle.task.task_hash().expect("task hash")),
            hex(&bundle.observation.observation_hash().expect("obs hash")),
        );
        ir.episodes = es_ir::evaluation::EpisodeBatch {
            n_episodes: 1,
            seeds: es_ir::evaluation::SeedPlan::Explicit(vec![SEED]),
        };
        ir.suites.truncate(1);
        assert_eq!(ir.suites[0].name, "nominal");
        let seen = Rc::clone(&seen);
        let blank = vec![0u8; 96 * 96 * 3];
        let mut frames =
            move |_light: &es_eval::LightOverride,
                  model: &es_physics_core::backend::ModelInfo,
                  state: &es_physics_core::backend::StateView<'_>| {
                *seen.borrow_mut() = Some((model.clone(), row_of(state)));
                Ok::<Vec<u8>, String>(blank.clone())
            };
        es_eval::Evaluation::run_with_frames::<es_physics_backend::MuJoCoCpuBackend, _, NJ, H>(
            &ir,
            &bundle.task,
            &scene,
            &bundle.observation,
            &mut policy,
            deploy,
            es_physics_backend::MuJoCoCpuBackend::new,
            &es_eval::RunConfig {
                max_steps: Some(STEPS),
                traj_dir: Some(eval_traj.clone()),
                // The document's own number, the one `es loop collect` just ran under.
                expected_latency_ms: demo_latency_ms(&bundle),
                ..es_eval::RunConfig::default()
            },
            Some(&mut frames),
            None,
        )
        .expect("the demo evaluates one episode");
    }

    // --- the comparison ------------------------------------------------------------------------
    let a = es_env::Trajectory::read(&collect_traj.join("ep-000.estraj"))
        .expect("the collector wrote an .estraj");
    let b = es_env::Trajectory::read(&eval_traj.join("nominal-00.estraj"))
        .expect("the evaluation wrote an .estraj");
    assert_eq!(
        (a.ticks(), b.ticks()),
        (STEPS as usize, STEPS as usize),
        "both paths ran the whole budget: collection {} ticks, evaluation {} ticks",
        a.ticks(),
        b.ticks()
    );
    for tick in 0..a.ticks() {
        let (qa, qb) = (a.qpos(tick), b.qpos(tick));
        let (va, vb) = (a.qvel(tick), b.qvel(tick));
        for (i, (x, y)) in qa.iter().zip(qb).chain(va.iter().zip(vb)).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "tick {tick}, element {i} of `qpos ‖ qvel`: `es loop collect` has {x} and \
                 `es eval run` {y} for seed {SEED}. The two paths must execute chunks \
                 identically -- not only at the same cadence (packet M7/T7, open question 24)"
            );
        }
    }
    // The whole file, so the body poses `es video showcase` replays are pinned too.
    assert_eq!(
        blake3::hash(&a.to_bytes()),
        blake3::hash(&b.to_bytes()),
        "the two `.estraj` files differ outside `qpos ‖ qvel`"
    );
    println!(
        "RAN collection_and_evaluation_draw_the_same_trajectory: {} ticks identical bitwise, \
         expected_latency_ms = {} (one control tick at {} Hz)",
        a.ticks(),
        demo_latency_ms(&bundle),
        deploy.rate.control.as_hz_f64()
    );
}

/// A `PolicyRuntime` that records the first observation it is handed and commands nothing.
///
/// `infer` returning a zero chunk is not a bypass: the chunk still goes through the plane, and
/// the run is four steps long because only the *first* state is under test.
struct FirstObservation {
    seen: std::rc::Rc<std::cell::RefCell<Option<Vec<f64>>>>,
}

impl es_policy::PolicyRuntime for FirstObservation {
    fn load(
        &mut self,
        _graph: &es_ir::learning::LearningGraph,
        _weights: &es_policy::WeightsSource,
    ) -> Result<es_policy::PolicyInfo, es_policy::PolicyError> {
        Err(es_policy::PolicyError::NotLoaded)
    }

    fn infer(
        &mut self,
        inputs: &std::collections::BTreeMap<String, es_compile::Tensor>,
    ) -> Result<std::collections::BTreeMap<String, es_compile::Tensor>, es_policy::PolicyError>
    {
        let mut slot = self.seen.borrow_mut();
        if slot.is_none() {
            if let Some(t) = inputs.values().next() {
                *slot = Some(
                    t.data
                        .chunks_exact(4)
                        .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                        .collect(),
                );
            }
        }
        // `[batch, horizon, joints]`: `DomainRunner` checks the policy contract's declared
        // shape, and a `[1, 6]` step is not a chunk.
        Ok(std::collections::BTreeMap::from([(
            "action".to_owned(),
            es_compile::Tensor {
                dtype: es_ir::types::ElemType::F64,
                shape: vec![1, 16, 6],
                data: vec![0u8; 16 * 6 * 8],
            },
        )]))
    }

    fn info(&self) -> Option<&es_policy::PolicyInfo> {
        None
    }

    fn runtime_hash(&self) -> [u8; 32] {
        *blake3::hash(b"es::tests::FirstObservation").as_bytes()
    }
}

/// [`es_env::expert::ScriptedExpert`] wearing the `PolicyRuntime` the evaluation runner drives.
///
/// The expert needs the cube's free-joint `qpos`, which the demo's Observation IR does not
/// carry (it emits six normalized joint angles and an image). The runner's frame source is
/// handed the full `StateView` immediately before the policy is called, so the frame closure
/// leaves the raw `qpos || qvel` row here on its way past. That is a test scaffold, not a new
/// extension point: `PolicyRuntime` is one of the seven `INV-17` allows and this is an impl
/// of it, nothing more.
struct ExpertPolicy<const NJ: usize, const H: usize> {
    expert: es_env::expert::ScriptedExpert,
    seen: SeenState,
    /// Calls to [`es_policy::PolicyRuntime::infer`] -- the re-plan cadence, counted (V17).
    calls: std::rc::Rc<std::cell::Cell<u64>>,
}

/// The loaded model and the raw `qpos || qvel` row the frame source last saw.
type SeenState =
    std::rc::Rc<std::cell::RefCell<Option<(es_physics_core::backend::ModelInfo, Vec<f64>)>>>;

impl<const NJ: usize, const H: usize> es_policy::PolicyRuntime for ExpertPolicy<NJ, H> {
    fn load(
        &mut self,
        _graph: &es_ir::learning::LearningGraph,
        _weights: &es_policy::WeightsSource,
    ) -> Result<es_policy::PolicyInfo, es_policy::PolicyError> {
        Err(es_policy::PolicyError::NotLoaded)
    }

    fn infer(
        &mut self,
        _inputs: &std::collections::BTreeMap<String, es_compile::Tensor>,
    ) -> Result<std::collections::BTreeMap<String, es_compile::Tensor>, es_policy::PolicyError>
    {
        self.calls.set(self.calls.get() + 1);
        let seen = self.seen.borrow();
        let (model, row) = seen
            .as_ref()
            .ok_or_else(|| es_policy::PolicyError::Backend("no state captured yet".to_owned()))?;
        let state = es_env::expert::state_of_row(model, row);
        // Out of reach ends the demonstration in `es loop collect`; here it holds, so the
        // episode runs out its budget and is scored a failure rather than an error.
        let rows = self
            .expert
            .chunk(model, &state, 0)
            .unwrap_or_else(|| vec![state.qpos_of(0)[..NJ].to_vec(); H]);
        let mut data = Vec::with_capacity(H * NJ * 8);
        for r in rows.iter().take(H) {
            for j in 0..NJ {
                data.extend_from_slice(&r.get(j).copied().unwrap_or(0.0).to_le_bytes());
            }
        }
        Ok(std::collections::BTreeMap::from([(
            "action".to_owned(),
            es_compile::Tensor {
                dtype: es_ir::types::ElemType::F64,
                shape: vec![H as u64, NJ as u64],
                data,
            },
        )]))
    }

    fn info(&self) -> Option<&es_policy::PolicyInfo> {
        None
    }

    fn runtime_hash(&self) -> [u8; 32] {
        *blake3::hash(b"es-env::ScriptedExpert").as_bytes()
    }
}

/// Packet M5/V1: `--expert` drives every tick itself, so a machine with `mujoco` but no
/// `torch` can collect demonstrations. Without `mujoco` it still refuses to fake a run.
#[test]
fn loop_collect_expert_needs_no_torch() {
    let dir = scratch_dir("expert-no-torch");
    let policy = write_demo_bundle(&dir);
    let run = |out: &str| {
        bin()
            .args([
                "loop",
                "collect",
                "--expert",
                "so101-pick-place",
                "--policy",
            ])
            .arg(&policy)
            .arg("--scene")
            .arg(demo_scene_path())
            .args([
                "--episodes",
                "1",
                "--seed",
                "4",
                "--max-steps",
                "40",
                "--out",
            ])
            .arg(dir.join(out))
            .output()
            .expect("run es loop collect --expert")
    };

    if es_physics_backend::MuJoCoCpuBackend::is_available().is_err() {
        // No backend: the SKIPPED path, still exit 3, still nothing written.
        let out = run("skipped");
        let text = stdout(&out);
        assert_eq!(out.status.code(), Some(3), "{text}");
        assert!(text.contains("SKIPPED"), "{text}");
        println!("SKIP loop_collect_expert_needs_no_torch: no mujoco backend");
        return;
    }
    // `torch` is not checked at all under `--expert`, so this exits 0 on a machine that has
    // `mujoco` and no `torch` -- which is what the oracle tier is.
    let out = run("collected");
    let text = stdout(&out);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("terminations:"), "{text}");
    println!("RAN loop_collect_expert_needs_no_torch");
}

// --- plan V: `es policy lower` / `es policy pack` (packet M5/V2) ------------------------------

/// A checkpoint that fits a lowered module exactly: every declared shape, plus one tensor
/// standing in for each opaque sub-module a `prefix.*` claim covers.
fn conforming_checkpoint(module: &es_policy::TorchModule) -> es_policy::weights::Checkpoint {
    let mut file: es_policy::weights::Checkpoint = module
        .weight_shapes
        .iter()
        .map(|(k, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            (k.clone(), (shape.clone(), vec![0.01f32; n]))
        })
        .collect();
    for claim in module.weight_keys.iter().filter(|k| k.ends_with(".*")) {
        file.insert(
            format!("{}conv1.weight", claim.trim_end_matches('*')),
            (vec![2], vec![1.0, 2.0]),
        );
    }
    file
}

/// `es policy lower` writes the module verbatim and a contract that describes it (spec 8.7).
#[test]
fn policy_lower_writes_the_module_and_contract() {
    let dir = scratch_dir("policy-lower");
    let policy = write_demo_bundle(&dir);
    let build = dir.join("build");

    let out = bin()
        .args(["policy", "lower", "--policy"])
        .arg(&policy)
        .arg("--out")
        .arg(&build)
        .output()
        .expect("run es policy lower");
    let text = stdout(&out);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stdout:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The module is the IR's, byte for byte -- the packet's whole reason for existing.
    let bytes = std::fs::read(&policy).expect("read policy.esb");
    let bundle = es_compile::PolicyBundle::open(&bytes).expect("open policy.esb");
    let module = es_policy::lower_to_torch(&bundle.learning).expect("the demo graph lowers");
    let written = std::fs::read_to_string(build.join("es_policy.py")).expect("es_policy.py");
    assert_eq!(written, module.source);

    let contract: es_policy::lower::Contract = serde_json::from_str(
        &std::fs::read_to_string(build.join("contract.json")).expect("contract.json"),
    )
    .expect("contract.json parses");
    assert_eq!(
        contract,
        es_policy::lower::Contract::new(&module, &bundle.learning)
    );
    assert!(text.contains("lowering_hash: "), "{text}");
    // Printed rather than pinned: no golden fixes the lowering, and a packet that moves the
    // demo's Learning IR has to record where `lowering_hash` landed (design note section 7.6).
    println!(
        "RAN policy_lower: lowering_hash {} over inputs {:?}",
        hex(&module.lowering_hash),
        contract.inputs.keys().collect::<Vec<_>>()
    );
}

/// `INV-16`: the only weight format is safetensors, and a file that is not one is refused by
/// the header reader before anything can execute. There is no pickle path to reach.
#[test]
fn policy_pack_rejects_a_non_safetensors_file() {
    let dir = scratch_dir("policy-pack-pickle");
    let policy = write_demo_bundle(&dir);
    // A real pickle prologue. Nothing here unpickles it; it fails as a malformed header.
    let pickled = dir.join("model.pkl");
    std::fs::write(&pickled, b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00").expect("write");

    let out = bin()
        .args(["policy", "pack", "--policy"])
        .arg(&policy)
        .arg("--weights")
        .arg(&pickled)
        .arg("--out")
        .arg(dir.join("trained.esb"))
        .output()
        .expect("run es policy pack");
    let text = format!("{}{}", stdout(&out), String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("safetensors"), "{text}");
    assert!(
        !dir.join("trained.esb").exists(),
        "a refusal wrote a bundle"
    );
}

/// Usage errors are exit 2, not 1 -- the distinction `main.rs` makes for every subcommand.
#[test]
fn policy_usage_errors_exit_2() {
    for args in [
        vec!["policy", "lower"],
        vec!["policy", "lower", "--policy", "a.esb"],
        vec!["policy", "pack", "--policy", "a.esb", "--out", "b.esb"],
        vec!["policy", "lower", "--nonsense", "x"],
        vec!["policy", "distill"],
    ] {
        let out = bin().args(&args).output().expect("run es policy");
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", stdout(&out));
    }
    let help = bin()
        .args(["policy", "--help"])
        .output()
        .expect("run es policy --help");
    assert_eq!(help.status.code(), Some(0));
    assert!(
        stdout(&help).contains("es policy lower"),
        "{}",
        stdout(&help)
    );
}

/// The packet's headline: a bundle `es policy pack` produced is loaded by `es eval run
/// --policy` with **no change to the eval path** (`crates/es/src/cmd/eval.rs:387-393`). That is
/// the assertion a `lower_act` bundle would fail, and it is why plan V does not use it
/// (`docs/design/visible-learning.md` section 2.5).
///
/// Without `mujoco` and `torch` this is the SKIPPED path (exit 3), which still proves the
/// bundle opened and its four IRs re-validated -- both happen before the availability check.
#[test]
fn policy_pack_output_is_accepted_by_eval_run() {
    let dir = scratch_dir("policy-eval-run");
    let f = deployable_fixture();
    let untrained = dir.join("untrained.esb");
    std::fs::write(&untrained, build_policy_bundle(&f)).expect("write untrained.esb");
    let config = dir.join("eval.toml");
    write(
        &config,
        &es_ir::serial::evaluation_to_toml(&f.evaluation).expect("evaluation toml"),
    );

    let module = es_policy::lower_to_torch(&f.learning).expect("the fixture graph lowers");
    let weights = dir.join("model.safetensors");
    std::fs::write(
        &weights,
        es_policy::weights::write_safetensors(&conforming_checkpoint(&module)),
    )
    .expect("write the checkpoint");

    let trained = dir.join("trained.esb");
    let packed = bin()
        .args(["policy", "pack", "--policy"])
        .arg(&untrained)
        .arg("--weights")
        .arg(&weights)
        .arg("--out")
        .arg(&trained)
        .output()
        .expect("run es policy pack");
    assert_eq!(
        packed.status.code(),
        Some(0),
        "stdout:\n{}\nstderr:\n{}",
        stdout(&packed),
        String::from_utf8_lossy(&packed.stderr)
    );

    let run = bin()
        .args(["eval", "run", "--config"])
        .arg(&config)
        .arg("--policy")
        .arg(&trained)
        .arg("--scene")
        .arg(demo_scene_path())
        .arg("--out")
        .arg(dir.join("out"))
        .output()
        .expect("run es eval run");
    let text = format!("{}{}", stdout(&run), String::from_utf8_lossy(&run.stderr));
    if run.status.code() == Some(3) {
        assert!(text.contains("SKIPPED"), "{text}");
        println!(
            "SKIP policy_pack_output_is_accepted_by_eval_run: {}",
            text.trim()
        );
        return;
    }
    // Anything else: it must not have been the *policy load* that failed. What the run does
    // after that is the Evaluation IR's business, not this packet's.
    for refusal in [
        "weights hash mismatch",
        "does not match the lowered graph",
        "malformed safetensors",
        "no PyTorch lowering",
    ] {
        assert!(
            !text.contains(refusal),
            "eval run refused the packed bundle: {text}"
        );
    }
    println!(
        "RAN policy_pack_output_is_accepted_by_eval_run (exit {:?})",
        run.status.code()
    );
}

// --- plan V, packet M5/V3: `es eval run --frames` and the demo run ----------------------------

/// A build with no renderer refuses `--frames` as a usage error (exit 2), before it loads
/// anything -- never a run that quietly writes no frames.
#[cfg(not(feature = "render"))]
#[test]
fn eval_run_frames_without_a_renderer_is_a_usage_error() {
    let dir = scratch_dir("eval-frames-refused");
    let out = bin()
        .args(["eval", "run", "--config"])
        .arg(vl_fixture("evaluation.toml"))
        .arg("--policy")
        .arg(dir.join("nothing.esb"))
        .arg("--scene")
        .arg(demo_scene_path())
        .arg("--out")
        .arg(dir.join("out"))
        .arg("--frames")
        .arg(dir.join("frames"))
        .output()
        .expect("run es eval run --frames");
    let text = format!("{}{}", stdout(&out), String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.code(), Some(2), "{text}");
    assert!(text.contains("needs the `render` feature"), "{text}");
    // It refused before opening the bundle, which does not exist.
    assert!(!dir.join("out").exists(), "{text}");
    println!("RAN eval_run_frames_without_a_renderer_is_a_usage_error");
}

/// V3 writes what V4 reads. The `events.json` shape, the `Policy | Clamped | Fallback | Human`
/// spelling, the `<cell>/NNNNNN.bin` + `layout.json` directories and `report.json`'s
/// `success_rate` cell are one contract between two packets, and this is the only place both
/// ends of it exist at once. No GPU, no physics: the frames are written by hand in exactly the
/// shape `es_eval::FrameSink` writes them.
#[test]
fn eval_run_output_is_what_es_video_mosaic_reads() {
    use es_ir::evaluation::{CellResult, EvaluationReport, MetricSpec, MetricValue};

    let dir = scratch_dir("eval-mosaic");
    let frames = dir.join("frames");
    let (h, w) = (3_u64, 2_u64);
    let mut events: std::collections::BTreeMap<String, Vec<es_eval::StepEvent>> =
        std::collections::BTreeMap::new();
    for cell in 0..2u64 {
        let name = format!("nominal-{cell:02}");
        let cell_dir = frames.join(&name);
        std::fs::create_dir_all(&cell_dir).expect("cell dir");
        write(
            &cell_dir.join("layout.json"),
            &format!("{{\"dtype\":\"u8\",\"shape\":[{h}, {w}, 3]}}\n"),
        );
        let mut records = Vec::new();
        for frame in 0..2u64 {
            std::fs::write(
                cell_dir.join(format!("{frame:06}.bin")),
                vec![(cell * 16 + frame) as u8; (h * w * 3) as usize],
            )
            .expect("frame");
            records.push(es_eval::StepEvent {
                frame,
                tick: es_core::PhysTick(frame),
                // One clamped step, so the overlay branch is exercised too.
                source: if cell == 1 && frame == 0 {
                    es_eval::EventSource::Clamped
                } else {
                    es_eval::EventSource::Policy
                },
                events: u32::from(cell == 1 && frame == 0),
            });
        }
        events.insert(name, records);
    }
    let sink = es_eval::FrameSink {
        dir: frames.clone(),
        events,
    };
    let events_path = dir.join("events.json");
    sink.write_events(&events_path).expect("events.json");

    let report = EvaluationReport {
        schema_version: es_eval::runner::SCHEMA_VERSION,
        evaluation_hash: [0; 32],
        execution_hash: [0; 32],
        cells: vec![CellResult {
            suite: "nominal".to_owned(),
            metric: MetricSpec::SuccessRate,
            value: MetricValue::Scalar(0.5),
            n_episodes: 2,
        }],
        acceptance: Vec::new(),
        passed: true,
        episodes: Vec::new(),
    };
    let report_path = dir.join("report.json");
    write(
        &report_path,
        &serde_json::to_string_pretty(&report).expect("report json"),
    );

    let out = bin()
        .args(["video", "mosaic", "--frames"])
        .arg(&frames)
        .arg("--events")
        .arg(&events_path)
        .arg("--report")
        .arg(&report_path)
        .args(["--grid", "1x2", "--out"])
        .arg(dir.join("mosaic"))
        .output()
        .expect("run es video mosaic");
    let text = format!("{}{}", stdout(&out), String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("2 cell(s)"), "{text}");
    assert!(dir.join("mosaic/000000.bin").is_file(), "{text}");
    println!("RAN eval_run_output_is_what_es_video_mosaic_reads");
}

/// The demo's own Evaluation IR is a document this build validates, hashes and can plan --
/// no backend, no policy, no device. The measurement itself is `visible_learning_demo_run`.
#[test]
fn the_demo_evaluation_document_validates() {
    let text = std::fs::read_to_string(vl_fixture("evaluation.toml")).expect("evaluation.toml");
    let ir = es_ir::serial::evaluation_from_toml(&text).expect("evaluation.toml parses");
    assert!(ir.validate().is_empty(), "{:?}", ir.validate());
    assert_eq!(ir.episodes.n_episodes, 16, "the video is a 4x4 grid");
    assert_eq!(ir.suites.len(), 6);
    assert_eq!(ir.suites[0].name, "nominal");
    // It names the documents it was generated against (spec 10.2).
    let task = es_ir::serial::task_from_toml(
        &std::fs::read_to_string(vl_fixture("task.toml")).expect("task.toml"),
    )
    .expect("task.toml parses");
    let obs = es_ir::serial::observation_from_toml(
        &std::fs::read_to_string(vl_fixture("observation.toml")).expect("observation.toml"),
    )
    .expect("observation.toml parses");
    assert_eq!(ir.task, hex(&task.task_hash().expect("task hash")));
    assert_eq!(
        ir.observation,
        hex(&obs.observation_hash().expect("observation hash"))
    );
    println!(
        "RAN the_demo_evaluation_document_validates (evaluation_hash {})",
        hex(&ir.evaluation_hash().expect("evaluation hash"))
    );
}

/// Packet M5/V3's oracle: the demo run itself, and the gate is **non-vacuous**.
///
/// It passes only if the run contains at least one `Success` episode *and* at least one step
/// the Safety Plane classified `Clamped`. A demo whose plane never fires demonstrates nothing,
/// and a demo with no success demonstrates nothing either. Neither comes from a flag:
/// successes come from the trained policy and clamps from the envelope the Deployment IR
/// already declares (INV-12 -- nothing here disables or bypasses the plane).
///
/// `#[ignore]`: it needs `mujoco`, `torch`, a Vulkan device and a trained bundle, which is 61
/// MB and not in the repository. `ES_TRAINED_BUNDLE` points at it (packet M5/V2 leaves it in
/// `~/artifacts/plan-v/` on the oracle server).
#[test]
#[ignore = "needs mujoco, torch, a GPU and a trained bundle (ES_TRAINED_BUNDLE)"]
fn visible_learning_demo_run() {
    if cfg!(not(feature = "render")) {
        println!("SKIP visible_learning_demo_run: built without the `render` feature");
        return;
    }
    let Ok(bundle) = std::env::var("ES_TRAINED_BUNDLE") else {
        println!("SKIP visible_learning_demo_run: ES_TRAINED_BUNDLE is not set");
        return;
    };
    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP visible_learning_demo_run: {reason}");
        return;
    }
    if let Err(reason) = es_policy::torch_runtime::is_available() {
        println!("SKIP visible_learning_demo_run: {reason}");
        return;
    }

    let dir = scratch_dir("visible-learning-demo");
    let out = dir.join("out");
    // Printed because the frames under it are what `es video mosaic` is pointed at afterwards.
    println!("visible_learning_demo_run out: {}", out.display());
    let run = bin()
        .args(["eval", "run", "--config"])
        .arg(vl_fixture("evaluation.toml"))
        .arg("--policy")
        .arg(&bundle)
        .arg("--scene")
        .arg(demo_scene_path())
        .arg("--out")
        .arg(&out)
        .arg("--frames")
        .arg(out.join("frames"))
        .output()
        .expect("run es eval run --frames");
    let text = format!("{}{}", stdout(&run), String::from_utf8_lossy(&run.stderr));
    if run.status.code() == Some(3) {
        println!("SKIP visible_learning_demo_run: {}", text.trim());
        return;
    }
    // Exit 1 is an acceptance criterion that did not hold, which is a measurement and not a
    // failure of this oracle; anything else is a real error.
    assert!(
        matches!(run.status.code(), Some(0 | 1)),
        "exit {:?}\n{text}",
        run.status.code()
    );

    // Every suite's success rate, printed as the table this packet exists to produce.
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("report.json")).expect("report"))
            .expect("report.json parses");
    let mut any_success = false;
    println!("SUITE                 SUCCESS_RATE  EPISODES");
    for cell in report["cells"].as_array().expect("cells") {
        if cell["metric"] != "success_rate" {
            continue;
        }
        let rate = cell["value"]["scalar"].as_f64().unwrap_or(f64::NAN);
        any_success |= rate > 0.0;
        println!(
            "{:<20} {rate:>13.4}  {}",
            cell["suite"].as_str().unwrap_or("?"),
            cell["n_episodes"]
        );
    }

    // One cell directory per suite x episode, and one event record per frame in it.
    let events: std::collections::BTreeMap<String, Vec<serde_json::Value>> =
        serde_json::from_str(&std::fs::read_to_string(out.join("events.json")).expect("events"))
            .expect("events.json parses");
    let ir = es_ir::serial::evaluation_from_toml(
        &std::fs::read_to_string(vl_fixture("evaluation.toml")).expect("evaluation.toml"),
    )
    .expect("evaluation.toml parses");
    assert_eq!(
        events.len(),
        ir.suites.len() * ir.episodes.n_episodes as usize
    );
    let mut clamped = 0usize;
    let mut fallback = 0usize;
    let mut frames = 0usize;
    for (cell, records) in &events {
        let on_disk = std::fs::read_dir(out.join("frames").join(cell))
            .expect("the cell's frames")
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
            .count();
        assert_eq!(records.len(), on_disk, "{cell}");
        frames += on_disk;
        for r in records {
            match r["source"].as_str() {
                Some("Clamped") => clamped += 1,
                Some("Fallback") => fallback += 1,
                _ => {}
            }
        }
    }
    println!("frames {frames}   clamped {clamped}   fallback {fallback}");

    assert!(
        any_success,
        "non-vacuity: no suite produced a single Success episode"
    );
    assert!(
        clamped > 0,
        "non-vacuity: the Safety Plane never clamped, so the overlay shows nothing"
    );
    println!("RAN visible_learning_demo_run");
}

// --- packet M5/V2b: `es dataset bake` ------------------------------------------------------

/// The demo's own `action` width, and a tile the demo `ImageSpec` fits.
const BAKE_DOF: usize = 6;

/// The width of the demo scene's `observation.state` row: `qpos || qvel`, `nq = 13` (six arm
/// hinges and the cube's seven-wide free joint) and `nv = 12`, exactly as
/// `es_data::collect::to_lerobot` writes it. The old fixture wrote six values, which no run of
/// `es loop collect` has ever produced -- it only passed because one channel read the leading
/// six (packet M5/V7a).
const BAKE_STATE: usize = 13 + 12;

const BAKE_TILE: usize = 96 * 96 * 3;

/// The demo's Observation IR has two `JointState` channels since packet M5/V7a, and only one
/// of them can be the leading values of the recorded row, so `es dataset bake` resolves the
/// other against the scene's `qpos` ranges -- which come from the backend that loads it. The
/// executor itself is covered locally and backend-free by
/// `a_baked_frame_is_bit_identical_to_what_capture_serves` (`es-eval`); what needs the backend
/// here is only the CLI's plumbing around it.
fn skip_without_bake_model(test: &str) -> bool {
    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP {test}: {reason}");
        return true;
    }
    false
}

/// Two short episodes with the columns the demo reads, plus the flat `<NNNNNN>.bin` tiles
/// `es loop collect --frames` writes beside them.
fn write_bake_fixture(root: &Path, tiles: &Path, episodes: u32, frames: usize) {
    let mut features = BTreeMap::new();
    features.insert(
        "observation.state".to_owned(),
        FeatureSpec::new(Dtype::Float32, [BAKE_STATE as u64]),
    );
    features.insert(
        "action".to_owned(),
        FeatureSpec::new(Dtype::Float32, [BAKE_DOF as u64]),
    );
    let mut writer = LeRobotWriter::create(root, Info::new(50.0, features)).expect("create");
    std::fs::create_dir_all(tiles).expect("tiles dir");
    let mut global = 0usize;
    for index in 0..episodes {
        let mut columns = BTreeMap::new();
        // A ramp inside the joint range, so `Normalize{Range -1..1}` has something to move.
        columns.insert(
            "observation.state".to_owned(),
            Column::F32(
                (0..frames * BAKE_STATE)
                    .map(|i| (i as f32 % 7.0) / 7.0 - 0.5)
                    .collect(),
            ),
        );
        columns.insert(
            "action".to_owned(),
            Column::F32(
                (0..frames * BAKE_DOF)
                    .map(|i| (i % 5) as f32 * 0.1)
                    .collect(),
            ),
        );
        writer
            .write_episode(&Episode {
                index,
                tasks: vec!["bake".to_owned()],
                timestamps: (0..frames).map(|i| i as f64 / 50.0).collect(),
                task_index: vec![0; frames],
                columns,
                video: BTreeMap::new(),
            })
            .expect("write episode");
        for _ in 0..frames {
            let tile: Vec<u8> = (0..BAKE_TILE)
                .map(|i| ((i + global * 13) % 256) as u8)
                .collect();
            std::fs::write(tiles.join(format!("{global:06}.bin")), &tile).expect("tile");
            global += 1;
        }
    }
    writer.finish().expect("finish");
}

/// `es dataset bake` writes one safetensors per episode under the Observation IR's own output
/// names, plus a manifest that says which documents produced them (packet M5/V2b, spec 19.2).
#[test]
fn dataset_bake_writes_safetensors_and_a_manifest() {
    if skip_without_bake_model("dataset_bake_writes_safetensors_and_a_manifest") {
        return;
    }
    let dir = scratch_dir("dataset-bake");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles, out) = (dir.join("ds"), dir.join("tiles"), dir.join("baked"));
    let (episodes, frames) = (2u32, 3usize);
    write_bake_fixture(&root, &tiles, episodes, frames);

    // The demo's second JointState channel sends the bake to the scene (packet M5/V7a),
    // and the Task IR's `scene.path` is repository-relative: name the file the way
    // `es eval run --scene` does, since this process does not run from the root.
    let scene = demo_scene_path();
    let result = bin()
        .args([
            "dataset",
            "bake",
            "--policy",
            bundle.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--frames",
            tiles.to_str().unwrap(),
            "--scene",
            scene.to_str().unwrap(),
            root.to_str().unwrap(),
        ])
        .output()
        .expect("run es dataset bake");
    let printed = format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.status.code(), Some(0), "{printed}");

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("manifest.json")).expect("manifest.json"),
    )
    .expect("manifest.json parses");
    assert_eq!(manifest["frames"].as_u64(), Some(6));
    assert_eq!(manifest["episodes"].as_array().map(Vec::len), Some(2));
    // The manifest names the Observation IR that produced the tensors, taken from the bundle
    // rather than re-derived -- so a baked set and the bundle that evaluates it cannot drift.
    let opened = es_compile::PolicyBundle::open(&std::fs::read(&bundle).expect("read bundle"))
        .expect("the bundle opens");
    let expected = opened
        .manifest
        .hashes
        .observation
        .expect("observation_hash");
    assert_eq!(
        manifest["observation_hash"].as_str(),
        Some(hex(&expected).as_str()),
        "{manifest}"
    );

    let header = es_policy::weights::parse_header(
        &std::fs::read(out.join("episode_000000.safetensors")).expect("episode 0"),
    )
    .expect("the baked file is safetensors");
    let shape = |name: &str| header.get(name).map(|e| e.shape.clone());
    assert_eq!(shape("joint_state"), Some(vec![frames as u64, 6]));
    assert_eq!(
        shape("rgb_overhead"),
        Some(vec![frames as u64, 3, 96, 96]),
        "the image output is CHW f32, which is `Dequantize`'s output and not the raw tile"
    );
    assert_eq!(shape("action"), Some(vec![frames as u64, 6]));
    assert!(out.join("episode_000001.safetensors").is_file());
}

/// An image input with no pixels is refused, naming the port. Baking zeros there would produce
/// a file that looks complete and trains a policy on a blank camera (design note section 7.6).
#[test]
fn dataset_bake_without_frames_refuses_an_image_observation() {
    if skip_without_bake_model("dataset_bake_without_frames_refuses_an_image_observation") {
        return;
    }
    let dir = scratch_dir("dataset-bake-nopix");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles, out) = (dir.join("ds"), dir.join("tiles"), dir.join("baked"));
    write_bake_fixture(&root, &tiles, 1, 2);

    let scene = demo_scene_path();
    let result = bin()
        .args([
            "dataset",
            "bake",
            "--policy",
            bundle.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--scene",
            scene.to_str().unwrap(),
            root.to_str().unwrap(),
        ])
        .output()
        .expect("run es dataset bake");
    let printed = format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.status.code(), Some(1), "{printed}");
    assert!(printed.contains("--frames was not given"), "{printed}");
    assert!(
        !out.join("episode_000000.safetensors").is_file(),
        "a refused bake left a file behind"
    );
}

/// The demo's two `JointState` channels cannot both be the leading values of the recorded row,
/// so `es dataset bake` resolves model-free first and goes to the scene when that is refused
/// (packet M5/V7a). Where the backend is missing -- PR CI, and this is the only coverage that
/// branch has there -- both halves of the failure must be named: the refusal and why it could
/// not be answered. Where the backend is present the two tests above cover the success, and
/// this one steps aside rather than asserting an error that did not happen.
#[test]
fn dataset_bake_names_both_refusals_when_the_scene_cannot_be_loaded() {
    let dir = scratch_dir("dataset-bake-nomodel");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles, out) = (dir.join("ds"), dir.join("tiles"), dir.join("baked"));
    write_bake_fixture(&root, &tiles, 1, 2);

    let result = bin()
        .args([
            "dataset",
            "bake",
            "--policy",
            bundle.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            root.to_str().unwrap(),
        ])
        .output()
        .expect("run es dataset bake");
    let printed = format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    if es_physics_backend::MuJoCoCpuBackend::is_available().is_ok() {
        println!("SKIP dataset_bake_names_both_refusals: the backend resolved the channel");
        return;
    }
    assert_eq!(result.status.code(), Some(1), "{printed}");
    assert!(
        printed.contains("only one channel can be the leading"),
        "the model-free refusal is not in the message: {printed}"
    );
    assert!(
        printed.contains("could not be loaded to resolve it"),
        "why the refusal could not be answered is not in the message: {printed}"
    );
    println!("RAN dataset_bake_names_both_refusals");
}

// --- packet M6/B1: the quadruped track's four documents ---------------------------------------

/// `tests/fixtures/quadruped/<name>`.
fn quad_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/quadruped")
        .join(name)
}

fn go1_scene_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/go1_primitives.xml")
}

/// The twelve actuated joints in the XML's declaration order, which is also `qpos[7..19]`,
/// `ctrl[0..12]` and every twelve-wide vector in the four documents.
const GO1_JOINTS: [&str; 12] = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
];

/// Playground's `Go1JoystickFlatTerrain` observation width
/// (`docs/api-notes/mujoco-playground-quadruped.md` section 1).
const GO1_OBS_DIM: u64 = 48;
const GO1_ACTION_DIM: u32 = 12;
/// `ctrl_dt = 0.02`.
const GO1_CONTROL_HZ: u64 = 50;
/// `action_scale`: `motor_targets = default_pose + action * 0.5`.
const GO1_ACTION_SCALE: f64 = 0.5;

fn go1_scene() -> (es_assets::scene::SceneDesc, Vec<u8>) {
    let xml = std::fs::read(go1_scene_path()).expect("the Go1 scene");
    let scene = es_assets::parse_mjcf(&String::from_utf8(xml.clone()).expect("utf-8"))
        .expect("the Go1 scene parses")
        .scene;
    (scene, xml)
}

/// The `home` keyframe's `qpos[7..19]` -- `default_pose`, the action's zero-point -- read out
/// of the scene rather than transcribed.
fn go1_default_pose() -> Vec<f64> {
    let xml = std::fs::read_to_string(go1_scene_path()).expect("the Go1 scene");
    let after = xml
        .split_once("<key name=\"home\" qpos=")
        .expect("the `home` keyframe")
        .1;
    let qpos: Vec<f64> = after
        .split_once('"')
        .expect("an opening quote")
        .1
        .split_once('"')
        .expect("a closing quote")
        .0
        .split_whitespace()
        .map(|t| t.parse().expect("a number"))
        .collect();
    assert_eq!(qpos.len(), 19, "free base + twelve hinges");
    qpos[7..].to_vec()
}

/// The policy's observation tensor: one 48-wide row, `Dimensionless` because it is a *mixed*
/// vector (m/s, rad/s, a unit gravity vector, rad, rad/s, a dimensionless action and a
/// command) and spec 5.4's unit algebra has no mixed unit. The per-block units are named in
/// the fixture header, which is where a reader looks for them.
fn go1_state_ty(frame: Frame) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([GO1_OBS_DIM]),
        unit: Unit::Dimensionless,
        frame,
        time: TimeRef::Tick,
        image: None,
    }
}

fn go1_scalar(unit: Unit, frame: Frame) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([1]),
        unit,
        frame,
        time: TimeRef::Tick,
        image: None,
    }
}

/// Builds the Go1 Task IR from the parsed scene: ids, the reset pose and the joint names all
/// come out of `go1_primitives.xml`, never out of this file.
fn go1_task(scene: &es_assets::scene::SceneDesc, xml: &[u8]) -> TaskIr {
    use es_ir::task::Distribution as TaskDist;

    let trunk = scene
        .bodies
        .iter()
        .find(|b| b.name == "trunk")
        .expect("the scene has a `trunk` body")
        .id;
    let base_joint = scene
        .joints
        .iter()
        .find(|j| j.kind == es_assets::scene::JointKind::Free)
        .expect("the trunk carries a free joint")
        .name
        .clone();
    let default_pose = go1_default_pose();

    let mut graph: Graph<TaskNode> = Graph::new(2);

    // Episode timeout: 1000 control ticks at 50 Hz = 20 s, upstream's `episode_length`.
    graph.insert(NodeId(0), TaskNode::GetTime { since_reset: true });
    graph.insert(
        NodeId(1),
        TaskNode::Compare {
            op: es_ir::task::CmpOp::Ge,
            rhs: Some(20.0),
            ty: go1_scalar(Unit::Time, Frame::World),
        },
    );
    graph.insert(
        NodeId(2),
        TaskNode::Terminate {
            kind: es_ir::task::TerminationKind::Timeout,
        },
    );
    graph.connect(NodeId(0), "value", NodeId(1), "a");
    graph.connect(NodeId(1), "value", NodeId(2), "value");

    // Out of the arena. This is *not* upstream's termination, which is "the upright vector's
    // z-component is negative" -- see the fixture header: `es-env`'s reward/termination cone
    // binds a joint's FIRST qpos index, and a free joint's first index is x, so neither the
    // base height nor the upright vector is reachable from IR-D today.
    graph.insert(
        NodeId(3),
        TaskNode::GetJointState {
            body: trunk,
            joints: vec![base_joint.clone()],
            quantity: JointQuantity::Position,
        },
    );
    graph.insert(
        NodeId(4),
        TaskNode::Compare {
            op: es_ir::task::CmpOp::Gt,
            rhs: Some(3.0),
            ty: go1_scalar(Unit::Angle, Frame::Joint(trunk)),
        },
    );
    graph.insert(
        NodeId(5),
        TaskNode::Terminate {
            kind: es_ir::task::TerminationKind::Failure,
        },
    );
    graph.connect(NodeId(3), "value", NodeId(4), "a");
    graph.connect(NodeId(4), "value", NodeId(5), "value");

    // The reward our cone can express: forward speed, normalized to [0, 1] over the command's
    // own amplitude bound (1.5 m/s). Upstream's is `exp(-err^2 / 0.25)` against a commanded
    // velocity, and `MathFn` is not in the cone -- but nothing here trains anything: the
    // policy is trained by `mujoco_playground` and imported (Track B), and this term exists
    // for our own evaluation harness.
    graph.insert(
        NodeId(6),
        TaskNode::GetJointState {
            body: trunk,
            joints: vec![base_joint],
            quantity: JointQuantity::Velocity,
        },
    );
    graph.insert(
        NodeId(7),
        TaskNode::Normalize {
            lo: vec![0.0],
            hi: vec![1.5],
            out_lo: 0.0,
            out_hi: 1.0,
            ty: go1_scalar(Unit::AngularVelocity, Frame::Joint(trunk)),
        },
    );
    graph.insert(
        NodeId(8),
        TaskNode::Reward {
            name: "forward_velocity".to_owned(),
            weight: 1.0,
            aggregation: es_ir::task::Aggregation::Sum,
            ty: PortType {
                unit: Unit::Normalized { lo: 0.0, hi: 1.0 },
                ..go1_scalar(Unit::AngularVelocity, Frame::Joint(trunk))
            },
        },
    );
    graph.connect(NodeId(6), "value", NodeId(7), "value");
    graph.connect(NodeId(7), "value", NodeId(8), "value");

    graph.insert(
        NodeId(9),
        TaskNode::ActionSpec {
            space: TaskSpace::JointPosition,
            dim: GO1_ACTION_DIM,
            control_rate_hz: GO1_CONTROL_HZ as f32,
        },
    );
    // Declared, and deliberately unfed: no IR-D source produces base linear velocity in the
    // body frame, projected gravity, the previous action or the joystick command, so there is
    // no graph value to bind here. The channel is the *contract* the trainer's env already
    // satisfies; Observation IR implements it (spec 5.1, spec 7.4).
    graph.insert(
        NodeId(10),
        TaskNode::ObservationSpec {
            channel: "state".to_owned(),
            ty: go1_state_ty(Frame::World),
        },
    );

    // The joystick command, drawn once per episode. Upstream resamples it mid-episode on an
    // Ornstein-Uhlenbeck-like schedule; a per-episode draw is what IR-D can say.
    let mut rng_streams = BTreeSet::new();
    let mut next = 11u32;
    for (axis, bound) in [("lin_x", 1.5), ("lin_y", 0.8), ("ang_z", 1.2)] {
        let stream = format!("command.{axis}");
        rng_streams.insert(stream.clone());
        graph.insert(
            NodeId(next),
            TaskNode::Randomization {
                target: stream.clone(),
                dist: TaskDist::Uniform {
                    lo: -bound,
                    hi: bound,
                },
                stream,
            },
        );
        next += 1;
    }

    // Reset to `home`. This is load-bearing rather than decorative: `SceneDesc` carries no
    // keyframe, so without these nodes the runtime would reset to the XML's `qpos0` -- every
    // hinge at 0 and the trunk at z = 0.445 -- and `default_pose`, the action's zero-point,
    // would never reach the simulator at all.
    rng_streams.insert("reset.base_z".to_owned());
    graph.insert(
        NodeId(next),
        TaskNode::ResetState {
            target: "qpos[2]".to_owned(),
            dist: TaskDist::Constant(0.278),
            stream: "reset.base_z".to_owned(),
        },
    );
    next += 1;
    for (i, joint) in GO1_JOINTS.iter().enumerate() {
        let stream = format!("reset.{joint}");
        rng_streams.insert(stream.clone());
        graph.insert(
            NodeId(next),
            TaskNode::ResetState {
                target: format!("joint.{joint}.qpos"),
                dist: TaskDist::Constant(default_pose[i]),
                stream,
            },
        );
        next += 1;
    }

    let mut channels = BTreeMap::new();
    channels.insert(
        "state".to_owned(),
        ObsChannel {
            source: ObsSource::JointState {
                body: trunk,
                dof: GO1_OBS_DIM as u32,
            },
            ty: go1_state_ty(Frame::World),
        },
    );

    TaskIr {
        schema_version: 2,
        scene: SceneRef {
            path: "tests/fixtures/mjcf/go1_primitives.xml".to_owned(),
            scene_hash: scene.scene_hash(),
            asset_hash: *blake3::hash(xml).as_bytes(),
        },
        graph,
        observation_spec: ObservationSpec { channels },
        config: TaskConfig {
            max_episode_steps: 1000,
            control_rate_hz: GO1_CONTROL_HZ as f32,
            deterministic: true,
            rng_streams,
        },
        control: None,
    }
}

fn go1_observation(task: &TaskIr, scene: &es_assets::scene::SceneDesc) -> ObservationIr {
    let trunk = scene
        .bodies
        .iter()
        .find(|b| b.name == "trunk")
        .expect("the scene has a `trunk` body")
        .id;
    let mut obs = ObservationIr::new(1, task.task_hash().expect("the task hashes"));
    obs.graph.insert(
        NodeId(0),
        ObservationNode::StateInput {
            source: trunk,
            io: Io::source(go1_state_ty(Frame::World)),
        },
    );
    // `history_len = 1`: Go1 stacks no frames, and layer 2 of spec 7.5's time model says so
    // explicitly rather than by omission. The same field carries `history_len = 3` if a later
    // Playground env (Spot, H1) joins the track.
    obs.temporal.window = Some(TemporalWindow {
        n_steps: 1,
        stride: 1,
        align: Align::Hold,
    });
    obs.outputs.insert(
        "state".to_owned(),
        ObservationOutput {
            port: PortRef::new(NodeId(0), "out"),
            ty: go1_state_ty(Frame::World),
        },
    );
    obs
}

fn go1_learning() -> LearningGraph {
    use es_ir::learning::{NormalizeDir, StatsSource};

    let default_pose = go1_default_pose();
    let input = Port::new("state", go1_state_ty(Frame::Policy));
    let feature = |dim: u64| PortType {
        elem: ElemType::F32,
        shape: Shape::new([dim]),
        unit: Unit::Dimensionless,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    };
    let chunk = |rows: u64| PortType {
        elem: ElemType::F32,
        shape: Shape::new([rows, u64::from(GO1_ACTION_DIM)]),
        unit: Unit::Normalized { lo: -1.0, hi: 1.0 },
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    };

    let mut nodes: Graph<LearningNode> = Graph::new(1);
    // brax's `running_statistics` observation normalizer, as an IR node. **The values are
    // placeholders** -- mean 0, std 1, i.e. the identity -- because they are a product of
    // training: `normalizer_params.mean["state"]` / `.std["state"]` come out of the
    // checkpoint, and the import packet rewrites this node (and with it `learning_hash`).
    nodes.insert(
        NodeId(0),
        LearningNode::Normalizer {
            inputs: vec![input.clone()],
            direction: NormalizeDir::Forward,
            stats: StatsSource::MeanStd {
                mean: vec![0.0; GO1_OBS_DIM as usize],
                std: vec![1.0; GO1_OBS_DIM as usize],
            },
            out_unit: Unit::Dimensionless,
        },
    );
    // The policy MLP. Playground's is four `Dense` layers -- 512, 256, 128, then 2 x 12 --
    // with `swish` between them; ours is `hidden = [512, 256]` with `out_dim = 128` (the same
    // three hidden widths the api-note names) plus the head's own `Linear(128, 12)`, which is
    // that fourth `Dense` restricted to the mean half of the Gaussian. Two gaps, both open
    // items in docs/design/quadruped-track.md section 3: `lower_to_torch` emits `nn.ReLU`, not
    // swish, and it puts no activation between `out_dim` and the head.
    nodes.insert(
        NodeId(1),
        LearningNode::StateEncoder {
            inputs: vec![Port::new("state", feature(GO1_OBS_DIM))],
            kind: StateEncoderKind::Mlp {
                hidden: vec![512, 256],
                activation: Activation::Relu,
                activate_output: false,
            },
            out_dim: 128,
        },
    );
    nodes.insert(
        NodeId(2),
        LearningNode::PolicyHead {
            inputs: vec![Port::new("feat", feature(128))],
            kind: HeadKind::Regression,
            action_dim: GO1_ACTION_DIM,
            horizon: 1,
            squash: Squash::None,
        },
    );
    nodes.insert(
        NodeId(3),
        LearningNode::ActionChunker {
            inputs: vec![Port::new("chunk", chunk(1))],
            horizon: 1,
            execute_chunk: 1,
            replan_hz: GO1_CONTROL_HZ as f32,
            mode: ActionExecutionMode::RecedingHorizon,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 2,
        },
    );
    // `motor_targets = default_pose + action * action_scale`, spelled as spec 8.3's
    // `ActionUnnormalizer`: `MeanStd` inverse is `x * std + mean`, which is exactly that with
    // `mean = default_pose` and `std = 0.5`. These numbers are NOT placeholders -- they come
    // from the scene's `home` keyframe and the api-note, and training does not move them.
    nodes.insert(
        NodeId(4),
        LearningNode::Normalizer {
            inputs: vec![Port::new("actions", chunk(1))],
            direction: NormalizeDir::Inverse,
            stats: StatsSource::MeanStd {
                mean: default_pose,
                std: vec![GO1_ACTION_SCALE; GO1_ACTION_DIM as usize],
            },
            out_unit: Unit::Angle,
        },
    );
    nodes.connect(NodeId(0), "out", NodeId(1), "state");
    nodes.connect(NodeId(1), "out", NodeId(2), "feat");
    nodes.connect(NodeId(2), "chunk", NodeId(3), "chunk");
    nodes.connect(NodeId(3), "actions", NodeId(4), "actions");
    nodes.inputs.push(PortRef::new(NodeId(0), "state"));
    nodes.outputs.push(PortRef::new(NodeId(4), "out"));

    let mut contract_inputs = BTreeMap::new();
    contract_inputs.insert("state".to_owned(), input.clone());
    LearningGraph {
        schema_version: 1,
        inputs: vec![input],
        nodes,
        outputs: vec![Port::new(
            "actions",
            PortType {
                unit: Unit::Angle,
                ..chunk(1)
            },
        )],
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            // The untrained placeholder. The import packet packs the brax checkpoint's
            // `hidden_0..3` kernels (transposed to [out, in]) and recomputes this hash; no
            // pickle path exists anywhere (INV-16).
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0; 32],
            },
            contract: PolicyContract {
                inputs: contract_inputs,
                observation_window: 1,
                action_dim: GO1_ACTION_DIM,
                horizon: 1,
                execute_chunk: 1,
                replanning_hz: GO1_CONTROL_HZ as f32,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 2.0,
                    deadline_ms: 20.0,
                },
            },
        },
    }
}

fn go1_deployment(scene: &es_assets::scene::SceneDesc) -> DeploymentIr {
    // Hard limits, read out of the scene: `safety.position` is each hinge's own `range` and
    // `torque_max` is its actuator's `forcerange` (+/-23.7 N m for hip and thigh, +/-35.55 for
    // the knee). Transcribing either would be a second source of truth.
    let limit_of = |name: &str| -> Limit {
        let j = scene
            .joints
            .iter()
            .find(|j| j.name == name)
            .unwrap_or_else(|| panic!("no joint `{name}`"));
        let (lower, upper) = j.range.unwrap_or_else(|| panic!("`{name}` has no range"));
        Limit { lower, upper }
    };
    let torque_of = |joint: &str| -> f64 {
        let id = scene
            .joints
            .iter()
            .find(|j| j.name == joint)
            .expect("the joint")
            .id;
        let a = scene
            .actuators
            .iter()
            .find(|a| a.target == es_assets::scene::ActuatorTarget::Joint(id))
            .unwrap_or_else(|| panic!("no actuator drives `{joint}`"));
        a.force_range.expect("the actuator has a forcerange").1
    };
    let position: Vec<Limit> = GO1_JOINTS.iter().map(|n| limit_of(n)).collect();
    let torque_max: Vec<f64> = GO1_JOINTS.iter().map(|n| torque_of(n)).collect();
    let n = GO1_ACTION_DIM as usize;

    DeploymentIr {
        schema_version: 1,
        robot: RobotRef {
            name: "go1".to_owned(),
            target: RobotTarget::Simulated {
                scene: "tests/fixtures/mjcf/go1_primitives.xml".to_owned(),
            },
            n_joints: n,
        },
        action: ActionContract {
            space: DepSpace::JointPosition,
            dim: n,
            horizon: 1,
            execute_chunk: 1,
        },
        safety: SafetyEnvelope {
            position,
            // 0.02 rad (1.1 deg) inside each hard limit. The knee's range is the tightest at
            // 1.93 rad, so this leaves room in every row.
            position_soft_margin: vec![0.02; n],
            // Go1's joints are geared A1-class motors; ~21 rad/s is about where they run under
            // load. Upstream bounds no velocity at all, so this is the envelope *widening*
            // the deployment adds, never a limit the policy trained against (INV-12).
            velocity_max: vec![21.0; n],
            // 21 rad/s reached in 0.042 s, two control ticks. A swing leg is the fastest thing
            // on this robot and it has to fit.
            acceleration_max: vec![500.0; n],
            torque_max,
            jerk_max: None,
            // The action is a joint position target and `tanh` bounds it, not its rate:
            // `default_pose +/- 0.5` rad is reachable in one tick by construction, so the
            // first difference is bounded at 1.0 rad and the second at twice that. Tighter
            // than this and the plane clamps every tick of a normal gait, which is the defect
            // packet M5/V6 found on the arm.
            action_rate: RateLimit {
                first_diff_max: vec![1.0; n],
                second_diff_max: vec![2.0; n],
            },
            // Declared, and **not enforced by `es-safety`** (docs/design/safety-plane.md):
            // there is no forward kinematics and no contact query in the plane. A quadruped
            // roams, so the box is the arena, not a reach envelope.
            workspace: Workspace::Box {
                min: [-50.0, -50.0, -0.01],
                max: [50.0, 50.0, 1.5],
            },
            ee_velocity_max: 5.0,
            min_self_distance: 0.005,
            min_env_distance: 0.005,
            // ~12.7 kg landing on one or two feet.
            contact_force_max: 500.0,
        },
        execution: ExecutionMode::RecedingHorizon,
        // 50 Hz control, 50 Hz inference: Go1 is reactive, chunk 1, one inference per tick
        // (api-note section 1; the real-robot deployment in the paper runs at 50 Hz too).
        deadlines: Deadlines {
            observation_age: Micros(40_000),
            inference_budget: Micros(20_000),
            actuation_budget: Micros(10_000),
        },
        watchdogs: WatchdogSet(vec![
            Watchdog::InferenceDeadline {
                budget: Micros(20_000),
            },
            Watchdog::ChunkUnderrun,
            Watchdog::StaleObservation {
                max_age: Micros(40_000),
            },
            Watchdog::EnvelopeViolationRate {
                window: 200,
                max_frac: 0.9,
            },
        ]),
        // A standing quadruped that freezes its PD targets stays standing; zeroing velocity
        // would be a fall. Nothing here can switch the plane off (INV-12).
        fallback: FallbackPolicy::HoldPosition,
        rate: RateSpec {
            control: TickRate::hz(GO1_CONTROL_HZ),
            inference: TickRate::hz(GO1_CONTROL_HZ),
        },
    }
}

/// Regenerates `tests/fixtures/quadruped/{task,observation,learning,deployment}.toml`. Every
/// number in them is read out of `tests/fixtures/mjcf/go1_primitives.xml` or out of
/// `docs/api-notes/mujoco-playground-quadruped.md`, and every hash is derived, so no value in
/// the four files is ever typed in by hand. Run explicitly:
///
///     cargo test -p es --test cli -- --ignored regenerate_quadruped_documents
#[test]
#[ignore = "fixture generator; run explicitly"]
fn regenerate_quadruped_documents() {
    // Spec 1.4: goldens and fixtures are CI read-only, and `cargo test -- --include-ignored`
    // runs every ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP regenerate_quadruped_documents: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    let (scene, xml) = go1_scene();
    let task = go1_task(&scene, &xml);
    let observation = go1_observation(&task, &scene);
    let learning = go1_learning();
    let deployment = go1_deployment(&scene);

    for (name, header, text) in [
        (
            "task.toml",
            QUAD_TASK_HEADER,
            es_ir::serial::task_to_toml(&task).expect("task toml"),
        ),
        (
            "observation.toml",
            QUAD_OBSERVATION_HEADER,
            es_ir::serial::observation_to_toml(&observation).expect("observation toml"),
        ),
        (
            "learning.toml",
            QUAD_LEARNING_HEADER,
            es_ir::serial::learning_to_toml(&learning).expect("learning toml"),
        ),
        (
            "deployment.toml",
            QUAD_DEPLOYMENT_HEADER,
            es_ir::serial::deployment_to_toml(&deployment).expect("deployment toml"),
        ),
    ] {
        write(&quad_fixture(name), &format!("{header}\n{text}"));
        println!("wrote {}", quad_fixture(name).display());
    }
}

const QUAD_TASK_HEADER: &str = "\
# Task IR (spec 6) for the MuJoCo Playground Go1 joystick task -- packet M6/B1.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_quadruped_documents` from
# tests/fixtures/mjcf/go1_primitives.xml, so `scene_hash`, `asset_hash`, the reset pose and
# the twelve joint names are derived, never typed in. Upstream's parameters are pinned in
# docs/api-notes/mujoco-playground-quadruped.md; the design note is
# docs/design/quadruped-track.md.
#
# Track B, and this is the whole framing: the policy is trained *externally* by
# `mujoco_playground` (JAX/MJX PPO, 200 M steps) and imported as weights (spec 8, spec 1.9 --
# we do not build a second training stack). So this document is not what trains anything. It
# is (a) the scene, reset and termination contract our runtime executes the imported policy
# under, and (b) the declaration of the observation channel the Observation IR implements.
#
# Parameters: NJ = 12, control 50 Hz (`ctrl_dt = 0.02`), max_episode_steps = 1000 (20 s,
# upstream's `episode_length`), action = 12 joint position targets.
#
# `state`, the one declared channel, is Playground's 48-wide policy input in its exact order:
#
#     local_linvel(3, m/s)  gyro(3, rad/s)  gravity(3, unit vector)
#     joint_pos(12, rad, minus default_pose)  joint_vel(12, rad/s)
#     last_action(12, dimensionless)  command(3, [m/s, m/s, rad/s])
#
# The port's unit is `Dimensionless` because that vector is mixed and spec 5.4's algebra has
# no mixed unit -- the block units are the list above, and this comment is where they live.
# Upstream's per-channel additive uniform observation noise is applied inside the trainer's
# env; it is not represented here, and `AugmentKind` has no uniform variant to represent it
# with (INV-15 would keep it off during evaluation anyway).
#
# THREE CEILINGS, recorded rather than papered over (design note section 3):
#
#  * The `ObservationSpec` node is deliberately UNFED. No IR-D source produces base linear
#    velocity in the body frame, projected gravity, the previous action or the joystick
#    command, so there is no graph value to bind to it. Declaring the channel is still
#    correct -- spec 7.4 says Task IR declares and Observation IR implements -- but nothing in
#    IR-D computes this vector, and `es-eval`'s capture path cannot serve it either: it reads
#    `qpos` slices and MJCF sensors, and this scene has neither the velocities nor (because
#    `mjcf_out` emits only jointpos/jointvel) the sensors. That is the track's first blocking
#    item, and `es eval run` names it rather than faking a run.
#  * Termination is timeout plus \"the base left the arena\" (|x| > 3 m). Upstream terminates
#    on the upright vector's z-component going negative, and the usual second guard is base
#    height -- neither is expressible: `es-env`'s reward/termination cone binds a joint's
#    FIRST qpos index (crates/es-env/src/plan.rs `joint_leaf`), and a free joint's first index
#    is x, not z and not a quaternion component. (Its unit here is `Angle`, not `Length`:
#    `JointQuantity::Position.unit()` is per-quantity, not per-joint-kind, and a free joint's
#    translational coordinates come out typed as if they were hinge angles.)
#  * The reward is forward speed normalized over the command bound, not upstream's
#    `exp(-err^2 / tracking_sigma)`: `MathFn` is not in the cone and IR-D has no constant leaf
#    to subtract a commanded velocity with. Upstream's fifteen shaped terms train the policy
#    in `mujoco_playground`; this one term is for *our* evaluation harness.
#
# The thirteen `ResetState` nodes are load-bearing, not decorative: `SceneDesc` carries no
# keyframe, so without them the runtime resets to the XML's `qpos0` -- every hinge at 0, the
# trunk at z = 0.445 -- and `default_pose`, the zero-point of every action, never reaches the
# simulator.";

const QUAD_OBSERVATION_HEADER: &str = "\
# Observation IR (spec 7) for the Go1 joystick task -- packet M6/B1.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_quadruped_documents`;
# `task_ref` is task.toml's own `task_hash`.
#
# It is one node long, and that is faithful rather than lazy: Playground applies no
# preprocessing to the state vector. The noise is added inside the training env, and the
# running-statistics normalization is part of the *policy* (brax keeps it in
# `normalizer_params`), so it is a Learning IR `Normalizer` node in learning.toml, not a
# `Normalize` here.
#
# `temporal.window = { n_steps = 1, stride = 1, align = \"Hold\" }` is layer 2 of spec 7.5's
# time model and is upstream's `history_len = 1` said out loud: Go1 stacks no frames. The same
# field carries `history_len = 3` if a later Playground env (Spot, H1) joins the track.
# XIR-011 checks it against the policy contract's `observation_window`.
#
# The 48-wide layout, its block order and its units are in task.toml's header -- this document
# implements that declaration and does not restate it.";

const QUAD_LEARNING_HEADER: &str = "\
# Learning IR (spec 8) for the Go1 joystick policy -- packet M6/B1.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_quadruped_documents`.
#
#   Normalizer{Forward, MeanStd}  ->  StateEncoder{Mlp [512, 256] -> 128}
#       ->  PolicyHead{Regression, 12 x 1}  ->  ActionChunker  ->  Normalizer{Inverse, MeanStd}
#
# horizon H = 1, execute_chunk = 1, replanning_hz = 50 (= the control rate; Go1 is reactive
# and upstream chunks nothing), action_dim = 12.
#
# THE WEIGHTS ARE A PLACEHOLDER and so is the observation normalizer. `policy.weights` is a
# `safetensors` reference with an all-zero hash, and node 0's `mean = 0` / `std = 1` is the
# identity. Both are filled by the import packet from a trained brax checkpoint:
# `params = (normalizer_params, policy_params)`, `normalizer_params.mean[\"state\"]` and
# `.std[\"state\"]` are the running statistics, and `policy_params['params']['hidden_0..3']`
# are the kernels (shape [in, out], transposed to [out, in] for our convention). No pickle
# path exists anywhere (INV-16).
#
# WHAT IS NOT YET EXACT, and is an open item in docs/design/quadruped-track.md section 3:
#
#  * Activation. brax's `MLP` uses `linen.swish`; `lower_to_torch` emits `nn.ReLU`. Different
#    function, so imported weights would not reproduce the trained policy until one of the two
#    moves. This is the single largest import risk and it is a Learning IR lowering question,
#    not a document question.
#  * Layer count matches, activation placement does not. Playground is Dense(512) swish,
#    Dense(256) swish, Dense(128) swish, Dense(24); ours is Linear(48,512) ReLU Linear(512,256)
#    ReLU Linear(256,128) then the head's Linear(128,12) -- four linear layers, as upstream,
#    but no activation before the head.
#  * `tanh`. Deterministic inference upstream is `tanh(location)` on the first half of the
#    24-wide output (`NormalTanhDistribution.mode`); the second half is the log-scale and is
#    unused. Our node set has no activation node, so `tanh` is not represented. The action
#    port's `Normalized { lo = -1, hi = 1 }` unit says where the value must land and the
#    Safety Plane clamps to it, which is the range but not the shape of `tanh`.
#
# The last node is NOT a placeholder: `motor_targets = default_pose + action * action_scale`
# is `Normalizer{Inverse, MeanStd}` exactly -- inverse MeanStd is `x * std + mean`, with
# `mean = default_pose` (the scene's `home` keyframe) and `std = 0.5` (`action_scale`).
#
# `architecture = \"Act\"` is this repo's name for a plain regression / behaviour-cloning head
# (`HeadKind::Regression`'s own doc comment), not a claim that this is ACT: there is no VAE,
# no transformer and no chunking here.";

const QUAD_DEPLOYMENT_HEADER: &str = "\
# Deployment IR + Safety Plane (spec 9) for the Go1 joystick policy -- packet M6/B1.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_quadruped_documents`:
# `safety.position` is each hinge's own `range` and `torque_max` is its actuator's
# `forcerange`, both read out of tests/fixtures/mjcf/go1_primitives.xml.
#
# The envelope is a *document*, never a switch (INV-12): nothing here can disable the Safety
# Plane, and every limit below is at least as wide as what the policy trained under.
#
# What every number is, physically:
#
#  * `position` -- upstream's own joint ranges: abduction +/-0.863 rad, hip -0.686..4.501,
#    knee -2.818..-0.888. `position_soft_margin = 0.02` rad (1.1 deg) inside each; the knee's
#    1.93 rad range is the tightest and leaves room.
#  * `torque_max` -- the `<position>` actuators' `forcerange`: 23.7 N m for hip and thigh,
#    35.55 N m for the knee. The PD loop is MuJoCo's own (`kp = 35`, `dof_damping = 0.5`
#    written in by `Go1Env.__init__`), so torque is what that loop makes of a position error.
#  * `velocity_max = 21.0` rad/s. Go1's geared A1-class joints run near this under load.
#    **Upstream bounds no joint velocity at all**, so this row is the deployment widening the
#    envelope, never a limit the policy was trained against.
#  * `acceleration_max = 500.0` rad/s^2: 21 rad/s reached in 0.042 s, two control ticks. A
#    swing leg is the fastest thing on this robot and it has to fit inside the bound.
#  * `action_rate.first_diff_max = 1.0` rad, `second_diff_max = 2.0`. The action is a joint
#    position target: `default_pose +/- 0.5` rad is reachable in one tick by construction
#    (`tanh` bounds the action, not its rate), so a full swing of the command is 1.0 rad.
#    Anything tighter clamps every tick of a normal gait, which is the defect packet M5/V6
#    found on the arm.
#  * `ee_velocity_max`, `workspace`, `contact_force_max`, `min_self_distance` and
#    `min_env_distance` are declared and **not enforced by `es-safety`** -- there is no forward
#    kinematics and no contact query in the plane (docs/design/safety-plane.md). For a robot
#    that walks away, `workspace` is the arena, not a reach envelope.
#  * `fallback = \"hold_position\"`: a standing quadruped that freezes its PD targets stays
#    standing. `zero_velocity` would be a fall.
#
# Rates are integer `TickRate`s (spec 18.1): control 50 Hz and inference 50 Hz, because Go1 is
# reactive -- one inference per control tick, `horizon = execute_chunk = 1`, no chunking
# upstream. The deadlines are whole multiples of the 20 ms control period: observation_age 2x,
# inference_budget 1x, actuation_budget 1/2x.
#
# KNOWN DEFECT on the simulated path, and it is this document's headline open item: the
# Safety Plane is fed `qpos[0..12]` / `qvel[0..12]` of the loaded model
# (crates/es-eval/src/runner.rs `joint_state`), a convention written for a FIXED-BASE arm. Go1
# floats: `qpos[0..7]` and `qvel[0..6]` are the trunk's free joint, so the plane would judge
# the base pose against joint limits and the first six hinges against nothing. The envelope
# below is correct for the robot; feeding it the right twelve rows is an `es-eval` change and
# the quadruped track's second blocking item (design note section 3). It is not worked around
# here, and it is certainly not worked around by disabling anything (INV-12).
#
# `target` is the simulated scene; a real Go1 swaps it for `Physical { driver }` without
# touching the envelope.";

/// The four documents, read back from disk exactly as `es ir check` reads them.
fn quadruped_documents() -> (TaskIr, ObservationIr, LearningGraph, DeploymentIr) {
    let read = |name: &str| std::fs::read_to_string(quad_fixture(name)).expect(name);
    (
        es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml parses"),
        es_ir::serial::observation_from_toml(&read("observation.toml"))
            .expect("observation.toml parses"),
        es_ir::serial::learning_from_toml(&read("learning.toml")).expect("learning.toml parses"),
        es_ir::serial::deployment_from_toml(&read("deployment.toml"))
            .expect("deployment.toml parses"),
    )
}

/// Each of the four validates on its own, the Cross-IR Check (spec 11.1) is clean, and the
/// `es ir check` binary agrees -- the packet's first oracle, and the one that needs nothing
/// but Rust.
#[test]
fn quadruped_documents_validate_and_cross_check() {
    let (task, observation, learning, deployment) = quadruped_documents();
    for (name, diags) in [
        ("task", task.validate()),
        ("observation", observation.validate()),
        ("learning", learning.validate()),
        ("deployment", deployment.validate()),
    ] {
        assert!(diags.is_empty(), "{name}.toml: {diags:#?}");
    }
    let diags = cross::check(&IrBundle {
        task: &task,
        observation: &observation,
        learning: &learning,
        deployment: &deployment,
        evaluation: None,
    });
    assert!(diags.is_empty(), "cross-IR: {diags:#?}");

    // The documents say what the api-note says (`docs/api-notes/mujoco-playground-quadruped.md`
    // section 1): 48 in, 12 out, one frame of history, 50 Hz, chunk 1.
    let contract = &learning.policy.contract;
    assert_eq!(contract.action_dim, GO1_ACTION_DIM);
    assert_eq!((contract.horizon, contract.execute_chunk), (1, 1));
    assert_eq!(contract.observation_window, 1);
    // An exact comparison is the point: `replanning_hz` is the control rate, not near it.
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(contract.replanning_hz, GO1_CONTROL_HZ as f32);
    }
    assert_eq!(
        observation.outputs["state"].ty.shape.dims(),
        [GO1_OBS_DIM],
        "Playground's policy input is 48-wide"
    );
    assert_eq!(
        observation.temporal.window,
        Some(TemporalWindow {
            n_steps: 1,
            stride: 1,
            align: Align::Hold
        }),
        "history_len = 1"
    );
    assert_eq!(deployment.rate.control, TickRate::hz(GO1_CONTROL_HZ));
    assert_eq!(deployment.robot.n_joints, GO1_ACTION_DIM as usize);

    let out = bin()
        .args(["ir", "check"])
        .arg(quad_fixture("task.toml"))
        .arg(quad_fixture("observation.toml"))
        .arg(quad_fixture("learning.toml"))
        .arg(quad_fixture("deployment.toml"))
        .output()
        .expect("run es ir check");
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("hash chain"), "{text}");
    println!("RAN quadruped_documents_validate_and_cross_check");
}

/// The committed files are exactly what the generator writes, so a hand edit to any of the
/// four is a failing test rather than a number nobody can trace back to the scene.
#[test]
fn quadruped_documents_are_what_the_generator_produces() {
    let (scene, xml) = go1_scene();
    let task = go1_task(&scene, &xml);
    let observation = go1_observation(&task, &scene);
    for (name, built) in [
        ("task.toml", es_ir::serial::task_to_toml(&task).unwrap()),
        (
            "observation.toml",
            es_ir::serial::observation_to_toml(&observation).unwrap(),
        ),
        (
            "learning.toml",
            es_ir::serial::learning_to_toml(&go1_learning()).unwrap(),
        ),
        (
            "deployment.toml",
            es_ir::serial::deployment_to_toml(&go1_deployment(&scene)).unwrap(),
        ),
    ] {
        let on_disk = std::fs::read_to_string(quad_fixture(name)).expect(name);
        let body = on_disk
            .split_once("\nes_schema")
            .map(|(_, rest)| format!("es_schema{rest}"))
            .unwrap_or(on_disk);
        assert_eq!(
            body, built,
            "{name} is not what the generator writes; rerun \
             `cargo test -p es --test cli -- --ignored regenerate_quadruped_documents`"
        );
    }
    // And the scene the documents name is the scene they were generated from.
    assert_eq!(task.scene.scene_hash, scene.scene_hash());
    println!("RAN quadruped_documents_are_what_the_generator_produces");
}

/// A 32-bit xorshift, seeded from the documents themselves: an untrained policy's chunk is
/// arbitrary inside `tanh`'s range, and what the Safety Plane does with an arbitrary chunk is
/// the property under test. Deterministic, because a flaky Safety Plane test is worthless.
fn xorshift(state: &mut u32) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    f64::from(*state % 2001) / 1000.0 - 1.0
}

/// The packet's pipeline oracle: the four documents pack into a `policy.esb`, and 100 control
/// ticks of an untrained policy's actions go through the real `SafetyPlane` built from
/// `deployment.toml` without a panic and without ever leaving the envelope.
///
/// This is the Safety Plane path in-process rather than through `es eval run`, and
/// deliberately: `es eval run` needs a `mujoco` and a `torch` interpreter, and it also cannot
/// serve this observation yet (see [`quadruped_eval_run_names_the_observation_gap`]). What it
/// can prove today is that the envelope in `deployment.toml` is one the plane accepts, applies
/// and never has to be disabled for (INV-12) -- at 12 joints, which no fixture in this repo
/// had before.
#[test]
fn quadruped_bundle_runs_100_ticks_through_the_safety_plane() {
    use es_safety::{ActionChunk, ActionSource, SafetyPlane};
    const NJ: usize = 12;

    let (task, observation, learning, deployment) = quadruped_documents();
    let weights = b"es-m6-b1-untrained-placeholder".to_vec();
    let mut learning = learning;
    learning.policy.weights = es_ir::learning::WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes =
        es_compile::PolicyBundle::build(&task, &observation, &learning, &deployment, &weights)
            .expect("the four quadruped documents pack into a bundle");
    let bundle = es_compile::PolicyBundle::open(&bytes).expect("the bundle re-opens");
    assert_eq!(bundle.deployment.robot.n_joints, NJ);

    let default_pose = go1_default_pose();
    let mut seed = u32::from_le_bytes(
        learning.learning_hash().expect("the learning IR hashes")[..4]
            .try_into()
            .expect("4 bytes"),
    ) | 1;

    // 100 control ticks at `scale` rad of action amplitude, returning the `ActionSource`
    // histogram. Every tick is checked, so a panic or an escape is a failure wherever it
    // happens, not only at the end.
    let mut run = |scale: f64| -> BTreeMap<String, u32> {
        let mut plane = SafetyPlane::<NJ, 1>::from_ir(&deployment).expect("the envelope is valid");
        let mut q = [0.0; NJ];
        q.copy_from_slice(&default_pose);
        let mut sources = BTreeMap::new();
        for tick in 0..100u64 {
            plane.observe_state(&q, &[0.0; NJ]);
            // `default_pose + action * action_scale`, the unnormalizer's own formula.
            let mut row = [0.0; NJ];
            for (j, v) in row.iter_mut().enumerate() {
                *v = default_pose[j] + xorshift(&mut seed) * scale;
            }
            let chunk = ActionChunk::<NJ, 1>::new([row], 1, ExecutionMode::RecedingHorizon)
                .with_seq(tick + 1);
            let safe = plane.validate(&chunk, Micros(0), es_core::PhysTick::ZERO.add_ticks(tick));
            *sources.entry(format!("{:?}", safe.source)).or_insert(0u32) += 1;
            for (j, v) in safe.q.iter().enumerate() {
                assert!(v.is_finite(), "tick {tick} joint {j}: {v}");
                let limit = deployment.safety.position[j];
                assert!(
                    *v >= limit.lower && *v <= limit.upper,
                    "tick {tick} joint {j}: {v} is outside {limit:?}"
                );
            }
            // The plane's own output becomes the next measured state: a perfect servo, which
            // is the harshest case for the rate limiter.
            q = safe.q;
        }
        sources
    };

    // An untrained network: `tanh` bounds it to +/-1, so the command jumps up to a full
    // `+/-action_scale` rad per tick. That is ~50 rad/s at 50 Hz against a 21 rad/s joint, so
    // the plane SHOULD clamp -- what matters is that it clamps rather than latching a
    // fallback, and that nothing ever leaves the position limits.
    let untrained = run(GO1_ACTION_SCALE);
    println!("quadruped safety plane, 100 untrained ticks: {untrained:?}");
    assert_eq!(untrained.values().sum::<u32>(), 100);
    assert!(
        !untrained.keys().any(|k| k.starts_with("Fallback")),
        "a watchdog latched over 100 ticks of an untrained policy: {untrained:?}"
    );

    // And the envelope is a bound, not a wall: a command the robot can physically follow
    // (0.02 rad per tick, 1 rad/s) passes through untouched. Without this half, an envelope
    // clamped to a constant would also "pass" the half above -- the defect packet M5/V6 found
    // on the arm.
    let followable = run(0.02);
    println!("quadruped safety plane, 100 followable ticks: {followable:?}");
    assert!(
        followable.contains_key(&format!("{:?}", ActionSource::Policy)),
        "not one followable command reached the actuator unchanged, so the envelope is too \
         tight to be a bound on anything: {followable:?}"
    );
    println!("RAN quadruped_bundle_runs_100_ticks_through_the_safety_plane");
}

/// One suite, one seed: the cheapest Evaluation IR that reaches the runtime.
fn quadruped_evaluation(task: &TaskIr, observation: &ObservationIr) -> EvaluationIr {
    EvaluationIr {
        schema_version: 1,
        task: hex(&task.task_hash().expect("the task hashes")),
        observation: hex(&observation
            .observation_hash()
            .expect("the observation hashes")),
        episodes: EpisodeBatch {
            n_episodes: 1,
            seeds: SeedPlan::Explicit(vec![1]),
        },
        suites: vec![PerturbationSuite {
            name: "nominal".to_owned(),
            perturbations: Vec::new(),
        }],
        metrics: vec![MetricSpec::SuccessRate],
        acceptance: vec![AcceptanceCriterion {
            suite: Some("nominal".to_owned()),
            metric: MetricSpec::SuccessRate,
            comparator: Comparator::Ge,
            threshold: 0.0,
            aggregation: Aggregation::Mean,
        }],
        augmentation: AugmentationPolicy::Disabled,
        replay: ReplayPolicy::default(),
    }
}

/// `es eval run` on the four documents either runs or says why, and never fakes a run
/// (spec 1.4).
///
/// Today it cannot run: the capture path (`crates/es-eval/src/runner.rs input_sources`) serves
/// an observation input from a `qpos` slice, an MJCF sensor or an image, and Playground's
/// 48-wide vector is none of those -- it wants base linear velocity in the body frame, a gyro,
/// projected gravity, joint velocities, the previous action and the joystick command. This
/// test pins the refusal *by its message* so that the day the capture path grows base state,
/// this test fails and is updated rather than quietly staying green over a gap.
#[test]
fn quadruped_eval_run_names_the_observation_gap() {
    let dir = scratch_dir("quadruped-eval-run");
    let (task, observation, learning, deployment) = quadruped_documents();
    let weights = b"es-m6-b1-untrained-placeholder".to_vec();
    let mut learning = learning;
    learning.policy.weights = es_ir::learning::WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let policy = dir.join("policy.esb");
    std::fs::write(
        &policy,
        es_compile::PolicyBundle::build(&task, &observation, &learning, &deployment, &weights)
            .expect("the bundle builds"),
    )
    .expect("write policy.esb");
    let config = dir.join("eval.toml");
    write(
        &config,
        &es_ir::serial::evaluation_to_toml(&quadruped_evaluation(&task, &observation))
            .expect("evaluation toml"),
    );

    let out = bin()
        .args(["eval", "run", "--config"])
        .arg(&config)
        .arg("--policy")
        .arg(&policy)
        .arg("--scene")
        .arg(go1_scene_path())
        .arg("--out")
        .arg(dir.join("out"))
        .output()
        .expect("run es eval run");
    let text = format!("{}{}", stdout(&out), String::from_utf8_lossy(&out.stderr));
    match out.status.code() {
        // No `mujoco` and/or no `torch`: the documented SKIPPED exit (spec 1.4).
        Some(3) => {
            assert!(text.contains("SKIPPED"), "{text}");
            println!(
                "SKIP quadruped_eval_run_names_the_observation_gap: {}",
                text.trim()
            );
        }
        Some(1) => {
            assert!(
                text.contains("joint positions") || text.contains("is none of"),
                "eval run failed for a reason that is not the known observation gap:\n{text}"
            );
            assert!(
                !dir.join("out").join("report.json").exists(),
                "a report was written over a run that never happened"
            );
            println!("RAN quadruped_eval_run_names_the_observation_gap (refused by name)");
        }
        // The gap closed: `es eval run` served the observation and produced a report.
        Some(0) => {
            println!("RAN quadruped_eval_run_names_the_observation_gap (the run completed)");
        }
        other => panic!("es eval run exited with {other:?}:\n{text}"),
    }
}

// --- packet M5/V9: `es video showcase` -----------------------------------------------------

/// `es video showcase` needs a camera and a run, and says so instead of writing an empty
/// directory. Runs everywhere: no GPU, no backend, no fixture (spec 26.1).
#[test]
fn video_showcase_usage_errors_exit_2() {
    for extra in [
        vec!["showcase"],
        vec!["showcase", "--run", "nowhere"],
        vec!["showcase", "--run", "r", "--scene", "s", "--out", "o"],
        vec![
            "showcase", "--run", "r", "--scene", "s", "--out", "o", "--camera", "overhead",
            "--eye", "0,0,1",
        ],
        vec!["showcase", "--run", "r", "--eye", "not,a,number"],
    ] {
        let out = bin().arg("video").args(&extra).output().expect("es video");
        assert_eq!(
            out.status.code(),
            Some(2),
            "{extra:?}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    println!("RAN video_showcase_usage_errors_exit_2");
}

/// Packet M5/V9's oracle -- **a replay is the run**, byte for byte.
///
/// The expert is driven through `es_eval::Evaluation` exactly as
/// `expert_passes_the_evaluation_harness` drives it, but the frame source is the CPU reference
/// rasterizer (`es_render::cpu`, the same function that generates every render golden) at the
/// demo's own 96x96 observation `ImageSpec`, and the run records its `.estraj` trajectories.
/// Then every tick of the trajectory is re-rendered from the *file* and compared to the frame
/// the run wrote. If the recorded states were not the states the policy saw, or if
/// `Trajectory::poses` were a re-derivation rather than the pose map the renderer was handed,
/// the two would differ -- there is no tolerance here.
///
/// The CPU rasterizer and not a Vulkan device, deliberately: the claim is about the *states*,
/// and the renderer is a pure function of them either way (`docs/design/renderer.md` section
/// 5). That keeps this oracle needing only `mujoco` -- it is a **server oracle**, and prints a
/// reason and skips without it, like the two expert oracles it sits beside.
/// `showcase_replay_of_a_real_run_is_bit_identical` is the same claim through the GPU path and
/// through the CLI, on a real run.
#[test]
#[cfg(feature = "render")]
fn a_showcase_replay_reproduces_the_frames_the_policy_saw() {
    use std::cell::RefCell;
    use std::rc::Rc;

    const NJ: usize = 6;
    const H: usize = 16;
    /// Two and a half seconds of episode: enough for the arm to move, short enough that
    /// 2 x N software rasterizations is a test and not a job.
    const TICKS: u32 = 120;

    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP a_showcase_replay_reproduces_the_frames_the_policy_saw: {reason}");
        return;
    }
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml");
    let obs =
        es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml");
    let deploy =
        es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml");
    let scene = es_assets::parse_mjcf(
        &std::fs::read_to_string(demo_scene_path()).expect("the demo scene is in the repo"),
    )
    .expect("the demo scene parses")
    .scene;
    let cube = scene
        .joints
        .iter()
        .find(|j| j.kind == es_assets::scene::JointKind::Free)
        .expect("the scene has one free-joint body to pick up")
        .id;
    // The camera and the size the Observation IR declares: the run's own `ImageSpec`.
    let camera = scene.cameras.first().expect("the demo scene has a camera");
    let rcfg = es_env::EnvRendererCfg::rgb(camera.id, 96, 96);
    let rc = es_env::render::config(96, 96, es_render::Channel::Rgb8, es_render::RenderPath::Rs);
    let cpu_frame = |poses: &std::collections::BTreeMap<es_core::StableId, es_math::Pose>| {
        let tri = es_render::TriScene::from_scene_with_poses(&scene, poses)
            .expect("the demo scene tessellates");
        let view = es_env::render::camera_view(&scene, &rcfg, poses).expect("the camera resolves");
        es_render::cpu::rasterize(&tri, &view, &rc, 0)
            .tile(es_render::Channel::Rgb8)
            .expect("an Rgb8 tile")
            .to_bytes()
    };

    let dir = scratch_dir("showcase-replay");
    let frames_dir = dir.join("frames");
    let traj_dir = dir.join("traj");

    let mut cfg = es_env::expert::demo_cfg(cube);
    cfg.pace_to(&deploy, demo_replan(&deploy));
    let expert = es_env::expert::ScriptedExpert::new(&scene, cfg).expect("the expert builds");
    let seen: SeenState = Rc::new(RefCell::new(None));
    let mut policy = ExpertPolicy::<NJ, H> {
        expert,
        seen: Rc::clone(&seen),
        calls: std::rc::Rc::default(),
    };
    let taken = Rc::clone(&seen);
    let mut source = |_light: &es_eval::LightOverride,
                      model: &es_physics_core::backend::ModelInfo,
                      state: &es_physics_core::backend::StateView<'_>| {
        let mut row = state.qpos_of(0).to_vec();
        row.extend_from_slice(state.qvel_of(0));
        *taken.borrow_mut() = Some((model.clone(), row));
        Ok::<Vec<u8>, String>(cpu_frame(&es_env::render::body_poses(model, state, 0)))
    };

    let mut ir = demo_evaluation_ir(
        hex(&task.task_hash().expect("task hash")),
        hex(&obs.observation_hash().expect("observation hash")),
    );
    ir.episodes = es_ir::evaluation::EpisodeBatch {
        n_episodes: 1,
        seeds: es_ir::evaluation::SeedPlan::Explicit(vec![SEEDS[0]]),
    };
    ir.suites.truncate(1);
    assert_eq!(ir.suites[0].name, "nominal");
    let mut sink = es_eval::FrameSink::new(&frames_dir);
    es_eval::Evaluation::run_with_frames::<es_physics_backend::MuJoCoCpuBackend, _, NJ, H>(
        &ir,
        &task,
        &scene,
        &obs,
        &mut policy,
        &deploy,
        es_physics_backend::MuJoCoCpuBackend::new,
        &es_eval::RunConfig {
            max_steps: Some(TICKS),
            traj_dir: Some(traj_dir.clone()),
            ..es_eval::RunConfig::default()
        },
        Some(&mut source),
        Some(&mut sink),
    )
    .expect("the expert runs through the evaluation harness");

    // The replay: the trajectory file alone, no backend, no policy, no `StateView`.
    let traj = es_env::traj::Trajectory::read(&traj_dir.join("nominal-00.estraj"))
        .expect("the run wrote a trajectory");
    let cell = frames_dir.join("nominal-00");
    assert_eq!(
        traj.ticks(),
        sink.events["nominal-00"].len(),
        "one trajectory record per frame the run wrote"
    );
    assert!(traj.ticks() > 0, "the run recorded nothing");
    for tick in 0..traj.ticks() {
        let recorded = std::fs::read(cell.join(format!("{tick:06}.bin")))
            .unwrap_or_else(|e| panic!("frame {tick}: {e}"));
        assert_eq!(
            cpu_frame(&traj.poses(tick)),
            recorded,
            "frame {tick} differs: the replayed state is not the state the policy saw"
        );
    }
    println!(
        "RAN a_showcase_replay_reproduces_the_frames_the_policy_saw: {} frame(s) identical",
        traj.ticks()
    );
}

/// The same claim on a real run and through the real command: `es video showcase --camera
/// <the run's own camera> --width/--height <the run's own ImageSpec>` reproduces the frames
/// `es eval run --frames` wrote, byte for byte, on the GPU path.
///
/// `#[ignore]`: it needs a finished run with both `traj/` and `frames/` in it, and a Vulkan
/// device. `ES_SHOWCASE_RUN` points at the run directory and `ES_SHOWCASE_CELL` at one of its
/// cells (default `nominal-00`).
#[test]
#[ignore = "needs a finished run with traj/ and frames/ (ES_SHOWCASE_RUN) and a GPU"]
fn showcase_replay_of_a_real_run_is_bit_identical() {
    let Ok(run) = std::env::var("ES_SHOWCASE_RUN") else {
        println!("SKIP showcase_replay_of_a_real_run_is_bit_identical: ES_SHOWCASE_RUN unset");
        return;
    };
    let run = PathBuf::from(run);
    let cell = std::env::var("ES_SHOWCASE_CELL").unwrap_or_else(|_| "nominal-00".to_owned());
    let recorded = run.join("frames").join(&cell);
    let out = scratch_dir("showcase-real").join("replay");
    let got = bin()
        .args(["video", "showcase", "--run"])
        .arg(&run)
        .arg("--scene")
        .arg(demo_scene_path())
        .args(["--camera", "overhead", "--width", "96", "--height", "96"])
        .args(["--cell", &cell])
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run es video showcase");
    let text = format!("{}{}", stdout(&got), String::from_utf8_lossy(&got.stderr));
    assert_eq!(got.status.code(), Some(0), "{text}");

    let mut n = 0usize;
    loop {
        let name = format!("{n:06}.bin");
        let (Ok(a), Ok(b)) = (
            std::fs::read(recorded.join(&name)),
            std::fs::read(out.join(&name)),
        ) else {
            break;
        };
        assert_eq!(a, b, "frame {n} of {cell} differs");
        n += 1;
    }
    assert!(
        n > 0,
        "no frames compared: {} vs {}",
        recorded.display(),
        out.display()
    );
    // The replay must not stop early either.
    let on_disk = std::fs::read_dir(&recorded)
        .expect("the recorded cell")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
        .count();
    assert_eq!(
        n, on_disk,
        "the replay rendered {n} of {on_disk} recorded frames"
    );
    println!("RAN showcase_replay_of_a_real_run_is_bit_identical: {n} frame(s) identical");
}

// --- packet M5/V10: the scene diagnosis -------------------------------------------------------

/// The bin's three-dimensional interior, which the Task IR's own predicate cannot check
/// because a cone leaf is one scalar (design note section 5.4). Same numbers as
/// `expert_solves_the_pinned_seeds`.
fn cube_in_the_bin(x: f64, y: f64, z: f64) -> bool {
    (0.09..0.19).contains(&x) && (-0.15..-0.05).contains(&y) && z < 0.09
}

/// The schedule the demo loops run on: one control step is one control period (packet M5/V11).
///
/// The same derivation `Collector::run` and `es_eval::runner` apply, so a test that drives
/// `Env` directly steps the scene exactly as the collector and the evaluator do.
fn demo_domains(
    scene: &es_assets::scene::SceneDesc,
    deploy: &es_ir::deployment::DeploymentIr,
) -> es_env::scheduler::BatchDomains {
    es_env::scheduler::BatchDomains::single_env_at(
        es_core::TickRate::from_period_secs(scene.options.timestep).expect("the scene's timestep"),
        deploy.rate.control,
    )
    .expect("the scene's timestep divides the control period")
}

/// The seed the dataset under diagnosis was collected with; V1c's `ds-train` is `--seed 1`.
fn v10_seed() -> u64 {
    std::env::var("ES_V10_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

/// Packet M5/V10 measurement 1 -- **do the recorded actions reproduce success?**
///
/// Every demonstration's recorded action sequence is replayed open-loop from the same reset,
/// through the same physics and the same Safety Plane, and scored by the demonstration's own
/// predicate. `action` is the executed `SafeAction` and `action_commanded` the raw pre-plane
/// command (design note section 7.10), so replaying both separates "the data is sound" from
/// "the executed-vs-commanded distinction loses the task".
///
/// A plain replay loop over `es_env::Env`, not a second `PolicyRuntime`: `INV-17` allows seven
/// extension points and a recorded-action player is none of them. The plane still sees every
/// row -- one row per control tick under a fresh `seq`, which is the cursor discipline
/// `plane_chunk` produces for a one-row chunk -- so nothing is bypassed (`INV-12`).
///
/// **The server oracle.** It needs `MuJoCoCpuBackend` *and* a collected dataset, which is an
/// input rather than a fixture: point `ES_V10_DATASET` at one (V1c's is
/// `~/artifacts/plan-v/v1c/ds-train`). Without either it prints a reason and skips.
#[test]
fn recorded_actions_replay_to_the_same_outcome() {
    use es_physics_core::PhysicsBackend as _;
    use std::fmt::Write as _;

    const NJ: usize = 6;
    const H: usize = 16;

    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP recorded_actions_replay_to_the_same_outcome: {reason}");
        return;
    }
    let Ok(root) = std::env::var("ES_V10_DATASET") else {
        println!(
            "SKIP recorded_actions_replay_to_the_same_outcome: set ES_V10_DATASET to a \
             demonstration dataset collected from this scene"
        );
        return;
    };
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml");
    let deploy =
        es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml");
    let scene = es_assets::parse_mjcf(
        &std::fs::read_to_string(demo_scene_path()).expect("the demo scene is in the repo"),
    )
    .expect("the demo scene parses")
    .scene;

    let dataset = es_data::LeRobotDataset::open(&root).expect("the demonstration dataset opens");
    let n = dataset.episodes().len() as u32;
    let demos: Vec<es_data::Episode> = (0..n)
        .map(|i| dataset.read_episode(i).expect("episode reads back"))
        .collect();
    let f32col = |ep: &es_data::Episode, name: &str| -> Vec<f32> {
        match ep.columns.get(name) {
            Some(es_data::Column::F32(v)) => v.clone(),
            other => panic!("{name}: {other:?}"),
        }
    };

    // What the demonstrations themselves recorded: since packet M5/V12 a row is the state its
    // action was computed *from*, so the last `observation.state` row is the `qpos ‖ qvel` of
    // one control step before the episode ended, and the cube's free joint starts at qpos[6].
    // The state after the last action is in no row -- an episode's terminal step resets the env
    // inside `Env::step`, so it is not readable from the env afterwards either. Both sides of
    // this comparison read the same index of their own recording, so the shift is common to
    // both; and the task's success predicate requires the cube to have *settled* in the bin, so
    // a cube that is in the bin at the end was already in it one step earlier.
    let mut recorded_in_bin = 0usize;
    for ep in &demos {
        let state = f32col(ep, "observation.state");
        let width = state.len() / ep.len();
        let last = &state[(ep.len() - 1) * width..];
        if cube_in_the_bin(f64::from(last[6]), f64::from(last[7]), f64::from(last[8])) {
            recorded_in_bin += 1;
        }
    }

    let seed = v10_seed();
    let dump_dir = std::env::var_os("ES_V10_DUMP").map(PathBuf::from);
    if let Some(dir) = dump_dir.as_ref() {
        std::fs::create_dir_all(dir).expect("create the dump directory");
    }
    let mut results = Vec::new();
    for column in ["action", "action_commanded"] {
        let mut env = es_env::Env::new(
            &task,
            &scene,
            es_physics_backend::MuJoCoCpuBackend::new(),
            &demo_domains(&scene, &deploy),
            seed,
        )
        .expect("the demo scene loads");
        let (nq, nu) = (env.model().nq as usize, env.model().nu as usize);
        assert_eq!(nu, NJ, "the demo scene has {nu} actuators");
        let mut plane =
            es_safety::SafetyPlane::<NJ, H>::from_ir(&deploy).expect("the envelope builds");
        let (mut ok, mut in_bin, mut corrected) = (0usize, 0usize, 0u64);
        let mut worst_correction = 0.0f64;
        // `SafetyPlane::accept` takes a chunk only for a `seq` strictly greater than the last
        // it saw, and `begin_episode` deliberately does not reset that (spec 8.6). One counter
        // for the whole run, not one per episode -- restarting it makes every episode after
        // the first a permanent chunk underrun, which is `hold_position` forever.
        let mut seq = 0u64;
        for (index, demo) in demos.iter().enumerate() {
            let rows = f32col(demo, column);
            let mut closed = None;
            // Measurement 2's input: only this side knows the Task IR's randomization draw, so
            // the mujoco probe is handed the reset state and the executed rows rather than
            // guessing either. One file per episode, `ES_V10_DUMP` or nothing.
            let mut dump = (column == "action" && dump_dir.is_some()).then(|| {
                let state = env.backend().state();
                let mut text = String::new();
                for v in state.qpos_of(0).iter().chain(state.qvel_of(0)) {
                    let _ = write!(text, "{v:.17e} ");
                }
                text.push('\n');
                text
            });
            for t in 0..demo.len() {
                let mut want = [0.0f64; NJ];
                for (j, v) in want.iter_mut().enumerate() {
                    *v = f64::from(rows[t * nu + j]);
                }
                {
                    let state = env.backend().state();
                    let (mut q, mut qd) = ([0.0; NJ], [0.0; NJ]);
                    q.copy_from_slice(&state.qpos_of(0)[..NJ]);
                    qd.copy_from_slice(&state.qvel_of(0)[..NJ]);
                    plane.observe_state(&q, &qd);
                }
                let mut actions = [[0.0; NJ]; H];
                actions[0] = want;
                seq += 1;
                let chunk = es_safety::ActionChunk::new(actions, 1, ExecutionMode::RecedingHorizon)
                    .with_seq(seq);
                let safe = plane.validate(&chunk, Micros(0), env.tick());
                let delta = (0..NJ)
                    .map(|j| (safe.q[j] - want[j]).abs())
                    .fold(0.0f64, f64::max);
                if delta > 1e-9 {
                    corrected += 1;
                    worst_correction = worst_correction.max(delta);
                }
                let outcome = env.step(&safe.q).expect("the replay steps");
                if let Some(text) = dump.as_mut() {
                    let state = env.backend().state();
                    let cube = &state.qpos_of(0)[6..9];
                    for v in safe.q.iter().chain(cube) {
                        let _ = write!(text, "{v:.17e} ");
                    }
                    text.push('\n');
                }
                if let Some(ep) = outcome.episodes.into_iter().next() {
                    closed = Some(ep);
                    break;
                }
            }
            if let (Some(dir), Some(text)) = (dump_dir.as_ref(), dump) {
                write(&dir.join(format!("ep-{index:03}.txt")), &text);
            }
            let episode = match closed {
                Some(ep) => ep,
                None => env
                    .reset(None)
                    .expect("the budget-exhausted episode closes")
                    .into_iter()
                    .next()
                    .expect("one env"),
            };
            plane.begin_episode();
            if episode.termination == es_env::Termination::Success {
                ok += 1;
            }
            // The same index of the replay's own recording as `recorded_in_bin` reads of the
            // demonstration's (M5/V12).
            let at = (episode.steps() - 1) * nq;
            let (x, y, z) = (
                episode.qpos[at + 6],
                episode.qpos[at + 7],
                episode.qpos[at + 8],
            );
            let inside = cube_in_the_bin(x, y, z);
            if inside {
                in_bin += 1;
            }
            println!(
                "{column} ep {index:>2}: {:>4}/{:<4} steps  {:?}  cube ({x:.3}, {y:.3}, {z:.3}) \
                 inside={inside}",
                episode.steps(),
                demo.len(),
                episode.termination
            );
        }
        println!(
            "{column}: success {ok}/{n}, cube in the bin {in_bin}/{n}, plane corrected \
             {corrected} ticks by at most {worst_correction:.3e} rad"
        );
        results.push((ok, in_bin));
    }

    let (executed_ok, executed_in_bin) = results[0];
    let (commanded_ok, commanded_in_bin) = results[1];
    println!(
        "RAN recorded_actions_replay_to_the_same_outcome: {n} demonstrations recorded \
         {recorded_in_bin} cubes in the bin; replaying `action` reproduces {executed_in_bin} \
         ({executed_ok} Success), replaying `action_commanded` reproduces {commanded_in_bin} \
         ({commanded_ok} Success)"
    );
    assert_eq!(
        executed_in_bin, recorded_in_bin,
        "replaying the executed action open-loop does not reproduce the demonstration it was \
         recorded from: the physics, the seeding or the executed-vs-commanded distinction moved \
         between collection and replay (packet M5/V10 measurement 1)"
    );
}

/// Packet M5/V10 measurement 3 -- **does the temporal ensemble survive the grasp window?**
///
/// One demonstration is driven through `es_env::plane_chunk` exactly as `es loop collect` and
/// `es eval run` do (design note section 7.13): the expert's chunk is pushed every control tick
/// and the Deployment IR's `TemporalEnsemble { decay }` blends the overlapping ones. A second
/// buffer, fed the identical chunks under `HardSwitch`, is the *raw* command -- the newest
/// chunk's own row for the tick -- so the difference between the two is the ensemble and
/// nothing else.
///
/// The gripper is the joint this can break: the expert commands `grip_closed` the moment it
/// enters `Stage::Close`, while up to `es_env::CHUNK_SLOTS` older chunks still ramp toward
/// `grip_open`. If the blend cannot reach the closure the jaws need, every demonstration is a
/// push rather than a grasp.
///
/// **The server oracle**, like the two expert ones: the SO-101 scene has no driver but
/// `MuJoCoCpuBackend`.
#[test]
fn the_temporal_ensemble_survives_the_grasp_window() {
    use es_physics_core::PhysicsBackend as _;

    const NJ: usize = 6;
    const H: usize = 16;

    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP the_temporal_ensemble_survives_the_grasp_window: {reason}");
        return;
    }
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml");
    let learning =
        es_ir::serial::learning_from_toml(&read("learning.toml")).expect("learning.toml");
    let deploy =
        es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml");
    let scene = es_assets::parse_mjcf(
        &std::fs::read_to_string(demo_scene_path()).expect("the demo scene is in the repo"),
    )
    .expect("the demo scene parses")
    .scene;
    let cube = scene
        .joints
        .iter()
        .find(|j| j.kind == es_assets::scene::JointKind::Free)
        .expect("the scene has one free-joint body to pick up")
        .id;

    let blend = match deploy.execution {
        ExecutionMode::TemporalEnsemble { decay } => {
            es_ir::learning::ChunkBlendPolicy::TemporalEnsemble {
                weight_decay: decay as f32,
            }
        }
        other => panic!("the demo's deployment declares {other:?}, not a temporal ensemble"),
    };
    let latency = es_env::latency_ticks(
        learning.policy.contract.runtime.expected_latency_ms,
        deploy.rate.control,
    );

    let mut cfg = es_env::expert::demo_cfg(cube);
    // `es loop collect`'s own pacing (`crates/es/src/cmd/loop.rs`): the expert integrates its
    // command over the rows one chunk executes per replan.
    cfg.pace_to(&deploy, demo_replan(&deploy));
    let (grip_open, grip_closed) = (cfg.grip_open, cfg.grip_closed);
    let mut expert = es_env::ScriptedExpert::new(&scene, cfg).expect("the expert builds");
    let mut env = es_env::Env::new(
        &task,
        &scene,
        es_physics_backend::MuJoCoCpuBackend::new(),
        &demo_domains(&scene, &deploy),
        v10_seed(),
    )
    .expect("the demo scene loads");
    let model = env.model().clone();
    let nq = model.nq as usize;
    let mut plane = es_safety::SafetyPlane::<NJ, H>::from_ir(&deploy).expect("the envelope");
    let mut ensemble = es_env::ChunkBuffer::<NJ, H>::new(deploy.action.execute_chunk, blend);
    let mut newest = es_env::ChunkBuffer::<NJ, H>::new(
        deploy.action.execute_chunk,
        es_ir::learning::ChunkBlendPolicy::HardSwitch,
    );
    let (mut feed, mut raw_feed) = (es_env::PlaneFeed::default(), es_env::PlaneFeed::default());

    // Per tick: the stage, the raw newest-chunk row, the blended row, and the measured gripper
    // joint.
    let mut log: Vec<(es_env::Stage, [f64; NJ], [f64; NJ], f64)> = Vec::new();
    let mut termination = es_env::Termination::Timeout;
    for t in 0..u64::from(task.config.max_episode_steps) {
        let (row, q, qd) = {
            let state = env.backend().state();
            let mut row = state.qpos_of(0).to_vec();
            row.extend_from_slice(state.qvel_of(0));
            let (mut q, mut qd) = ([0.0; NJ], [0.0; NJ]);
            q.copy_from_slice(&row[..NJ]);
            qd.copy_from_slice(&row[nq..nq + NJ]);
            (row, q, qd)
        };
        let view = es_env::expert::state_of_row(&model, &row);
        let stage = expert.stage();
        let rows = expert
            .chunk(&model, &view, 0)
            .unwrap_or_else(|| panic!("the expert ran out of reach at tick {t}"));
        // `es-data`'s intervener hook carries the chunk as an `f32` tensor, so the rows the
        // buffer sees are `f32`-rounded on the collection path too.
        let mut actions = [[0.0; NJ]; H];
        let last = rows.last().cloned().unwrap_or_default();
        for (k, out) in actions.iter_mut().enumerate() {
            let r = rows.get(k).unwrap_or(&last);
            for (j, v) in out.iter_mut().enumerate() {
                *v = f64::from(r.get(j).copied().unwrap_or(0.0) as f32);
            }
        }
        let chunk = es_safety::ActionChunk::new(actions, H, deploy.execution);
        ensemble.push(&chunk, t + latency);
        newest.push(&chunk, t + latency);
        let (fed, blended) = es_env::plane_chunk(&mut ensemble, &mut feed, t, deploy.execution);
        let (_, raw) = es_env::plane_chunk(
            &mut newest,
            &mut raw_feed,
            t,
            ExecutionMode::RecedingHorizon,
        );
        plane.observe_state(&q, &qd);
        let safe = plane.validate(&fed, Micros(0), env.tick());
        if let (Some(b), Some(r)) = (blended, raw) {
            log.push((stage, r, b, q[NJ - 1]));
        }
        let outcome = env.step(&safe.q).expect("the demonstration steps");
        if let Some(ep) = outcome.episodes.into_iter().next() {
            termination = ep.termination;
            break;
        }
    }

    // The grasp window: the first tick in `Close` to the last tick that is not yet `Done`.
    let first = log
        .iter()
        .position(|(s, ..)| *s == es_env::Stage::Close)
        .expect("the expert reaches Stage::Close");
    let end = log
        .iter()
        .rposition(|(s, ..)| *s != es_env::Stage::Done)
        .unwrap_or(log.len() - 1);
    let window = &log[first..=end];
    println!(
        "grasp window: ticks {first}..={end} of {}, termination {termination:?}",
        log.len()
    );
    for stage in [
        es_env::Stage::Approach,
        es_env::Stage::Descend,
        es_env::Stage::Close,
        es_env::Stage::Lift,
        es_env::Stage::Transport,
        es_env::Stage::Lower,
        es_env::Stage::Release,
    ] {
        if let Some(at) = log.iter().position(|(s, ..)| *s == stage) {
            println!("  {stage:?} opens at tick {at}");
        }
    }
    println!("joint   max|blend-raw|      raw min     blend min       raw max     blend max");
    let mut worst = [0.0f64; NJ];
    let mut reaches = [true; NJ];
    for (j, w) in worst.iter_mut().enumerate() {
        let dev = window
            .iter()
            .map(|(_, r, b, _)| (b[j] - r[j]).abs())
            .fold(0.0f64, f64::max);
        *w = dev;
        let rmin = window
            .iter()
            .map(|(_, r, ..)| r[j])
            .fold(f64::MAX, f64::min);
        let bmin = window
            .iter()
            .map(|(_, _, b, _)| b[j])
            .fold(f64::MAX, f64::min);
        let rmax = window
            .iter()
            .map(|(_, r, ..)| r[j])
            .fold(f64::MIN, f64::max);
        let bmax = window
            .iter()
            .map(|(_, _, b, _)| b[j])
            .fold(f64::MIN, f64::max);
        reaches[j] = bmin <= rmin && bmax >= rmax;
        println!("{j:>5} {dev:>15.5} {rmin:>13.5} {bmin:>13.5} {rmax:>13.5} {bmax:>13.5}");
    }
    let measured = window.iter().map(|(.., m)| *m).fold(f64::MAX, f64::min);
    println!(
        "gripper: open {grip_open}, closed {grip_closed}; the blend's closure lags the raw \
         command by {:.5} rad at worst, and the jaw joint measured {measured:.5} rad at its \
         tightest",
        worst[NJ - 1]
    );
    println!(
        "RAN the_temporal_ensemble_survives_the_grasp_window: {termination:?}, worst per-joint \
         blend deviation {:.5} rad (gripper {:.5})",
        worst.iter().fold(0.0f64, |a, b| a.max(*b)),
        worst[NJ - 1]
    );
    assert_eq!(
        termination,
        es_env::Termination::Success,
        "the expert driven through the temporal ensemble did not solve the task"
    );
    // The property, exactly: the ensemble is a *lag*, not a loss of range. Every joint's
    // blended command still reaches both ends of what the newest chunk asked for -- including
    // the gripper, whose closure is what decides whether the demonstrations grasp or push. No
    // tolerance: the extremes are held long enough for every overlapping chunk to agree, so
    // this is an equality and a regression in `decay` or `CHUNK_SLOTS` breaks it.
    assert!(
        reaches[NJ - 1],
        "the temporal ensemble never lets the gripper reach the closure the expert commands: \
         the blend opens the jaws and every demonstration is a push (packet M5/V10 \
         measurement 3)"
    );
    assert!(
        reaches.iter().all(|r| *r),
        "a joint's blended command does not span what the newest chunk asked for: {reaches:?}"
    );
}

// --- packet M7/T1: `es train` -----------------------------------------------------------

/// `es train` is run from the repository root: the IR route's trainer is
/// `python/es/train_act.py` and a Task IR's `scene.path` is repository-relative.
fn train_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn train_golden(name: &str) -> PathBuf {
    train_root().join("tests/golden/train").join(name)
}

/// TOML basic strings take `\` as an escape, and `es` opens either separator on Windows.
fn train_toml_path(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// The three committed recipes, by the path `es train --recipe` is given from the root.
const TRAIN_RECIPES: [(&str, &str); 3] = [
    (
        "tests/fixtures/visible-learning/training.toml",
        "plan-ir.txt",
    ),
    (
        "tests/fixtures/visible-learning/training-lerobot.toml",
        "plan-lerobot.txt",
    ),
    // Packet M8/S1: `[init] policy` and `steps = 0`. Its plan is one trainer flag and one
    // pack line away from `plan-ir.txt`, which is the point of pinning it beside it.
    (
        "tests/fixtures/visible-learning/training-init.toml",
        "plan-init.txt",
    ),
];

/// `ES_PYTHON` is removed on purpose: it is the one machine-dependent word in the plan, so a
/// golden that is a property of the recipe alone has to be taken without it.
fn run_train(recipe: &str, out: &Path, extra: &[&str]) -> Output {
    bin()
        .current_dir(train_root())
        .env_remove("ES_PYTHON")
        .args(["train", "--recipe", recipe, "--out"])
        .arg(out)
        .args(extra)
        .output()
        .expect("run es train")
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Regenerates `tests/golden/train/plan-*.txt`. Run once, explicitly; they are then read-only
/// (spec 1.4), exactly like `generate_fixture_and_golden` in `tests/video.rs`.
#[test]
#[ignore = "golden generator; run explicitly"]
fn generate_train_goldens() {
    // Spec 1.4: goldens and fixtures are CI read-only, and `cargo test -- --include-ignored`
    // runs every ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_train_goldens: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    let dir = scratch_dir("train-goldens");
    std::fs::create_dir_all(train_root().join("tests/golden/train")).expect("golden dir");
    for (recipe, golden) in TRAIN_RECIPES {
        let out = run_train(recipe, &dir.join(golden), &["--dry-run"]);
        assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
        write(&train_golden(golden), &stdout(&out));
    }
}

/// Oracle 1. The plan is a property of the recipe: same bytes on any machine, in any output
/// directory, on either path separator -- which is what "paths relative to `<out>`" buys.
#[test]
fn train_dry_run_plan_is_the_golden() {
    for (recipe, golden) in TRAIN_RECIPES {
        let dir = scratch_dir("train-dry");
        let out = run_train(recipe, &dir, &["--dry-run"]);
        assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
        let want = std::fs::read_to_string(train_golden(golden))
            .unwrap_or_else(|e| panic!("{}: {e}", train_golden(golden).display()));
        assert_eq!(stdout(&out), want, "{recipe}: stdout is not the golden");
        let written = std::fs::read_to_string(dir.join("training").join("plan.txt"))
            .expect("--dry-run writes training/plan.txt");
        assert_eq!(written, want, "{recipe}: plan.txt is not the golden");
        // "writes nothing but plan.txt": no identity is claimed for a run that did not happen.
        assert!(!dir.join("training.lock").exists());
        assert!(!dir.join("training").join("config.json").exists());
    }
}

/// A recipe for the IR route against a fixture dataset, with an interpreter that cannot
/// exist -- so the run always stops at the trainer and the assertions are about the identity
/// the run wrote *before* spending anything.
fn train_fixture_recipe(bundle: &Path, root: &Path, tiles: &Path, seed: u64, lr: &str) -> String {
    format!(
        "kind = \"training\"\n\
         [dataset]\n\
         root = \"{}\"\n\
         frames = \"{}\"\n\
         [policy]\n\
         bundle = \"{}\"\n\
         [run]\n\
         steps = 40\n\
         batch = 2\n\
         lr = {lr}\n\
         seed = {seed}\n\
         checkpoint_at = [40]\n\
         device = \"cpu\"\n\
         interpreter = \"es-no-such-interpreter\"\n",
        train_toml_path(root),
        train_toml_path(tiles),
        train_toml_path(bundle),
    )
}

fn train_lock(out: &Path) -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(out.join("training.lock"))
            .unwrap_or_else(|e| panic!("{}: {e}", out.join("training.lock").display())),
    )
    .expect("training.lock is JSON")
}

/// Oracle 2. One recipe, one `identity_hash`, wherever it is run; `seed` and `lr` move it;
/// every slot is the digest of a file that exists and parses, and none of them is zero.
#[test]
fn train_identity_is_a_function_of_the_recipe() {
    let dir = scratch_dir("train-identity");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 2, 12);

    let recipe = dir.join("training.toml");
    write(
        &recipe,
        &train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4"),
    );
    let path = train_toml_path(&recipe);
    let (a, b) = (dir.join("out-a"), dir.join("out-b"));
    let run_a = run_train(&path, &a, &[]);
    // Neither run can finish: the recipe's interpreter does not exist, and the identity is
    // written before it is ever asked to (the pre-run/post-run split, design note section 3).
    assert!(!run_a.status.success(), "{}", stdout(&run_a));
    run_train(&path, &b, &[]);

    let (la, lb) = (train_lock(&a), train_lock(&b));
    assert_eq!(
        la["identity_hash"], lb["identity_hash"],
        "two output directories, two identities"
    );
    assert_eq!(
        la["training_hash"],
        serde_json::json!({"unset": true}),
        "a run that did not train claimed a training_hash"
    );

    for name in es_data::training::FILES {
        let text = std::fs::read_to_string(a.join("training").join(name))
            .unwrap_or_else(|e| panic!("training/{name}: {e}"));
        serde_json::from_str::<serde_json::Value>(&text)
            .unwrap_or_else(|e| panic!("training/{name} is not canonical JSON: {e}"));
        let digest = la["files"][name].as_str().expect(name);
        assert_eq!(
            digest,
            hex(blake3::hash(text.as_bytes()).as_bytes()),
            "training.lock's digest of {name} is not the digest of {name}"
        );
        assert_ne!(digest, hex(&[0u8; 32]), "{name} is an all-zero digest");
    }

    for (seed, lr) in [(1u64, "1e-4"), (0, "2e-4")] {
        let other = dir.join(format!("recipe-{seed}-{lr}.toml"));
        write(
            &other,
            &train_fixture_recipe(&bundle, &root, &tiles, seed, lr),
        );
        let out = dir.join(format!("out-{seed}-{lr}"));
        run_train(&train_toml_path(&other), &out, &[]);
        assert_ne!(
            la["identity_hash"],
            train_lock(&out)["identity_hash"],
            "seed {seed} lr {lr} did not move identity_hash"
        );
    }
}

// --- packet M7/T2: `es loop cycle` ------------------------------------------------------

const CYCLE_RECIPE: &str = "tests/fixtures/visible-learning/cycle.toml";

/// `es loop cycle`, from the repository root, with `ES_PYTHON` removed for the same reason
/// `run_train` removes it: the interpreter is the one machine-dependent word in the plan.
fn run_cycle(recipe: &str, out: &Path, extra: &[&str]) -> Output {
    bin()
        .current_dir(train_root())
        .env_remove("ES_PYTHON")
        .args(["loop", "cycle", "--recipe", recipe, "--out"])
        .arg(out)
        .args(extra)
        .output()
        .expect("run es loop cycle")
}

/// Regenerates `tests/golden/train/plan-cycle.txt`. Run once, explicitly; it is then
/// read-only (spec 1.4), exactly like `generate_train_goldens` beside it.
#[test]
#[ignore = "golden generator; run explicitly"]
fn generate_cycle_golden() {
    // Spec 1.4: goldens and fixtures are CI read-only, and `cargo test -- --include-ignored`
    // runs every ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_cycle_golden: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    let dir = scratch_dir("cycle-golden");
    let out = run_cycle(CYCLE_RECIPE, &dir, &["--dry-run"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
    write(&train_golden("plan-cycle.txt"), &stdout(&out));
}

/// Oracle 1. The stage plan is a property of the document: the same bytes on any machine, in
/// any output directory, on either path separator -- and T1's plan nested under `train`, with
/// the cycle's own collect output where the recipe's `[dataset]` used to be.
#[test]
fn cycle_dry_run_plan_is_the_golden() {
    let dir = scratch_dir("cycle-dry");
    let out = run_cycle(CYCLE_RECIPE, &dir, &["--dry-run"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
    let golden = train_golden("plan-cycle.txt");
    let want =
        std::fs::read_to_string(&golden).unwrap_or_else(|e| panic!("{}: {e}", golden.display()));
    assert_eq!(stdout(&out), want, "the stage plan is not the golden");
    // A run that did not happen writes nothing, not even a directory (the same rule
    // `es train --dry-run` follows for `training.lock`).
    assert!(!dir.join("loop.jsonl").exists());
    assert!(!dir.join("train").exists());
}

/// Oracle 3. Spec 13.3's "if `evaluation_hash` changes, the comparison is invalid" as a
/// refusal by name: both hashes are printed, and `--allow-new-evaluation` is the deliberate
/// act that proceeds.
#[test]
fn cycle_refuses_a_moved_evaluation_hash() {
    let dir = scratch_dir("cycle-eval-hash");
    // A ledger from an earlier iteration, judged under other conditions.
    let stale = "a".repeat(64);
    write(
        &dir.join("loop.jsonl"),
        &format!(
            "{{\"kind\":\"evaluate\",\"inputs\":{{\"evaluation_hash\":\"{stale}\"}},\
             \"outputs\":{{\"passed\":\"true\"}},\"created\":0}}\n"
        ),
    );
    let out = run_cycle(CYCLE_RECIPE, &dir, &["--dry-run"]);
    assert_eq!(out.status.code(), Some(1), "{}", stdout(&out));
    let said = stderr_of(&out);
    let now = hex(&es_ir::serial::evaluation_from_toml(
        &std::fs::read_to_string(vl_fixture("evaluation.toml")).expect("evaluation.toml"),
    )
    .expect("the demo Evaluation IR parses")
    .evaluation_hash()
    .expect("evaluation_hash"));
    assert!(said.contains(&stale), "the old hash is not named: {said}");
    assert!(said.contains(&now), "the new hash is not named: {said}");

    // Named deliberately, the same document proceeds.
    let out = run_cycle(CYCLE_RECIPE, &dir, &["--dry-run", "--allow-new-evaluation"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
}

/// Oracle 4. One real cycle on the demo fixtures: a 2-episode expert collect, the harness on
/// the expert *before* the 40-step IR-route training, then the trained checkpoint through the
/// same harness -- with `loop.jsonl` holding `collect`, `evaluate` (the gate), `train`,
/// `evaluate`, in that order, chained.
///
/// `#[ignore]`d because it needs `ES_PYTHON` (torch, mujoco) and a Vulkan device; without
/// either it prints why and stops rather than pretending (spec 1.4).
#[test]
#[ignore = "needs ES_PYTHON with torch and mujoco, and a render build"]
fn cycle_runs_the_expert_through_the_harness_first() {
    let name = "cycle_runs_the_expert_through_the_harness_first";
    let Ok(python) = std::env::var("ES_PYTHON") else {
        println!("SKIP {name}: ES_PYTHON is not set");
        return;
    };
    if !cfg!(feature = "render") {
        println!("SKIP {name}: built without the `render` feature, so --frames writes nothing");
        return;
    }
    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP {name}: {reason}");
        return;
    }
    let dir = scratch_dir("cycle-expert-first");
    let bundle = write_demo_bundle(&dir);
    let p = |path: &Path| train_toml_path(path);

    // The demo's own Evaluation IR, cut to one suite and two of the pinned seeds the expert
    // solves -- the same document the gate and the policy are both judged against.
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml");
    let obs =
        es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml");
    let mut ir = demo_evaluation_ir(
        hex(&task.task_hash().expect("task hash")),
        hex(&obs.observation_hash().expect("observation hash")),
    );
    ir.episodes = es_ir::evaluation::EpisodeBatch {
        n_episodes: 2,
        seeds: es_ir::evaluation::SeedPlan::Explicit(vec![SEEDS[0], SEEDS[1]]),
    };
    ir.suites.truncate(1);
    let eval_config = dir.join("evaluation.toml");
    write(
        &eval_config,
        &es_ir::serial::evaluation_to_toml(&ir).expect("the Evaluation IR serialises"),
    );

    let recipe = dir.join("training.toml");
    write(
        &recipe,
        &format!(
            "kind = \"training\"\n\
             [dataset]\nroot = \"unused\"\n\
             [policy]\nbundle = \"{}\"\n\
             [run]\nsteps = 40\nbatch = 2\nlr = 1e-4\nseed = 0\ncheckpoint_at = [40]\n\
             device = \"cpu\"\ninterpreter = \"{}\"\n",
            p(&bundle),
            p(Path::new(&python)),
        ),
    );
    let document = dir.join("cycle.toml");
    write(
        &document,
        &format!(
            "kind = \"cycle\"\nscene = \"{}\"\n\
             [collect]\npolicy = \"{}\"\nexpert = \"so101-pick-place\"\nepisodes = 2\n\
             seed = 1\nframes = true\n\
             [train]\nrecipe = \"{}\"\n\
             [eval]\nconfig = \"{}\"\njobs = 1\nframes = true\n",
            p(&demo_scene_path()),
            p(&bundle),
            p(&recipe),
            p(&eval_config),
        ),
    );

    let out = dir.join("run");
    let run = bin()
        .current_dir(train_root())
        .args(["loop", "cycle", "--recipe", &p(&document), "--out"])
        .arg(&out)
        .output()
        .expect("run es loop cycle");
    let (said, err) = (stdout(&run), stderr_of(&run));
    // 0 or 1: the acceptance of a 40-step policy is a measurement, not this oracle's subject.
    // Anything else means a stage failed, and the gate's refusal is exit 1 with its own line.
    assert!(
        matches!(run.status.code(), Some(0 | 1)),
        "stdout:\n{said}\nstderr:\n{err}"
    );
    assert!(
        !err.contains("did not pass the evaluation harness"),
        "the expert failed the harness, so nothing was trained (spec 28.9 rule 1):\n{err}"
    );

    let ledger = es_data::read_loop_steps(&out).expect("the cycle's ledger reads back");
    let kinds: Vec<es_data::LoopKind> = ledger.iter().map(|s| s.kind).collect();
    assert_eq!(
        kinds,
        vec![
            es_data::LoopKind::Collect,
            es_data::LoopKind::Evaluate,
            es_data::LoopKind::Train,
            es_data::LoopKind::Evaluate
        ],
        "stdout:\n{said}"
    );
    es_data::check_chain(&ledger).expect("the ledger chains");
    // The gate is the evaluate step that names an expert, and it came before the train step.
    assert_eq!(ledger[1].inputs["expert"], "so101-pick-place");
    assert_eq!(ledger[1].outputs["passed"], "true");
    assert_eq!(ledger[2].inputs["expert_gate"], "passed");
    assert_eq!(ledger[0].outputs["content"], ledger[2].inputs["content"]);
    let judged = &ledger[3].inputs["policy_hash"];
    assert_eq!(ledger[2].outputs["checkpoint.40"], *judged);
    assert!(out.join("eval").join("report.json").exists());
    assert!(out.join("eval-expert").join("report.json").exists());
    println!(
        "RAN {name}: expert gate success_rate {}, policy success_rate {} at policy_hash {judged}",
        ledger[1].outputs["success_rate"], ledger[3].outputs["success_rate"]
    );
}

/// Oracle 4. Every refusal names the field that caused it (spec 17.2).
#[test]
fn train_refuses_by_name() {
    let dir = scratch_dir("train-refuse");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 1, 12);
    let base = train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4");

    let refuse = |name: &str, body: &str, extra: &[&str]| -> String {
        let recipe = dir.join(format!("{name}.toml"));
        write(&recipe, body);
        let out = run_train(&train_toml_path(&recipe), &dir.join(name), extra);
        assert!(
            !out.status.success(),
            "{name} was accepted:\n{}",
            stdout(&out)
        );
        stderr_of(&out)
    };

    // Both routes, and neither.
    let both = base.replace(
        "bundle = ",
        "lerobot = { type = \"act\", chunk_size = 16, n_action_steps = 16 }\nbundle = ",
    );
    assert!(refuse("both", &both, &[]).contains("both"), "both");
    let neither = base
        .lines()
        .filter(|l| !l.starts_with("bundle ="))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(refuse("neither", &neither, &[]).contains("neither"));

    // An image input with no pixels behind it.
    let no_frames = base
        .lines()
        .filter(|l| !l.starts_with("frames ="))
        .collect::<Vec<_>>()
        .join("\n");
    let said = refuse("no-frames", &no_frames, &[]);
    assert!(said.contains("frames"), "{said}");

    // A dataset recorded under another Task IR (M5 review S-3/R4). `meta/tasks.jsonl` is the
    // one place collection writes it, and `LeRobotDataset::open` reads it back verbatim.
    let retired = "0".repeat(64);
    write(
        &root.join("meta").join("tasks.jsonl"),
        &format!("{{\"task_index\":0,\"task\":\"es:task:{retired}\"}}\n"),
    );
    let said = refuse("retired", &base, &[]);
    assert!(said.contains(&retired), "{said}");
    assert!(said.contains("--allow-retired-task"), "{said}");
    // Named, it is accepted -- and the run then stops at the interpreter instead, which is
    // the other refusal the packet lists.
    let said = refuse("allowed", &base, &["--allow-retired-task", &retired]);
    assert!(!said.contains("M5 review S-3"), "{said}");
    assert!(
        said.contains("es-no-such-interpreter"),
        "the refusal does not name the interpreter: {said}"
    );
}

/// Oracle 3. 40 steps on the demo bundle and the bake fixture, then the bundle `es train`
/// packed is opened by `TorchRuntime` and asked for a chunk.
///
/// `#[ignore]`d because it is the one `es train` oracle that needs torch; without `ES_PYTHON`
/// it prints why and stops rather than pretending (spec 1.4).
#[test]
#[ignore = "needs ES_PYTHON with torch"]
fn train_ir_path_packs_a_bundle_torch_opens() {
    let Ok(python) = std::env::var("ES_PYTHON") else {
        println!("SKIP train_ir_path_packs_a_bundle_torch_opens: ES_PYTHON is not set");
        return;
    };
    if skip_without_bake_model("train_ir_path_packs_a_bundle_torch_opens") {
        return;
    }
    let dir = scratch_dir("train-ir");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 4, 16);
    let recipe = dir.join("training.toml");
    write(
        &recipe,
        &train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4").replace(
            "es-no-such-interpreter",
            &train_toml_path(Path::new(&python)),
        ),
    );
    let out = dir.join("out");
    let run = bin()
        .current_dir(train_root())
        .args(["train", "--recipe", &train_toml_path(&recipe), "--out"])
        .arg(&out)
        .output()
        .expect("run es train");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stdout:\n{}\nstderr:\n{}",
        stdout(&run),
        stderr_of(&run)
    );

    // The bundle the run packed, opened the way `es eval run` opens one.
    let packed = out.join("checkpoints").join("40.esb");
    let bytes = std::fs::read(&packed).expect("checkpoints/40.esb");
    let trained = es_compile::PolicyBundle::open(&bytes).expect("the packed bundle opens");
    let mut runtime = es_policy::TorchRuntime::new();
    let info = es_policy::PolicyRuntime::load(
        &mut runtime,
        &trained.learning,
        &es_policy::WeightsSource::InMemory(trained.weights.clone()),
    )
    .expect("TorchRuntime::load accepts what es train packed");
    let inputs: BTreeMap<String, es_compile::Tensor> = trained
        .learning
        .policy
        .contract
        .inputs
        .iter()
        .map(|(port, ty)| {
            let dims = ty.ty.shape.dims().to_vec();
            let n: u64 = dims.iter().product();
            (
                port.clone(),
                es_compile::Tensor {
                    dtype: ElemType::F32,
                    shape: dims,
                    data: (0..n)
                        .flat_map(|i| ((i % 23) as f32 / 23.0).to_le_bytes())
                        .collect(),
                },
            )
        })
        .collect();
    let outputs = es_policy::PolicyRuntime::infer(&mut runtime, &inputs).expect("infer a chunk");
    let chunk = outputs.values().next().expect("one output");
    let values: Vec<f32> = chunk
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "{values:?}"
    );
    assert!(info.action_dim > 0);

    // `checkpoint.manifest` names that file, with the digest of its bytes.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("training").join("checkpoint.manifest"))
            .expect("checkpoint.manifest"),
    )
    .expect("checkpoint.manifest is JSON");
    let row = &manifest["checkpoints"][0];
    assert_eq!(row["step"].as_u64(), Some(40), "{manifest}");
    assert_eq!(
        row["bundle"].as_str(),
        Some("checkpoints/40.esb"),
        "{manifest}"
    );
    assert_eq!(
        row["weights_blake3"].as_str(),
        Some(hex(blake3::hash(&bytes).as_bytes()).as_str()),
        "{manifest}"
    );

    let lock = train_lock(&out);
    let training_hash = lock["training_hash"]
        .as_str()
        .expect("training_hash is set");
    assert_ne!(
        training_hash,
        lock["identity_hash"].as_str().expect("identity_hash"),
        "the post-run slots did not move training_hash"
    );
    assert_eq!(lock["checkpoints"][0]["step"].as_u64(), Some(40));
    println!("RAN train_ir_path_packs_a_bundle_torch_opens: training_hash {training_hash}");
}

// --- packet M7/E4: `es eval run --telemetry` publishes, a client attaches --------------------

use es_telemetry::{Client, Message, Payload, StreamId};

/// The demo documents with `max_episode_steps` cut to 60, one `es train` step to get a
/// checkpoint `TorchRuntime` will actually load, and a one-suite, three-episode Evaluation IR:
/// the smallest run that goes through the whole `es eval run` path -- `mujoco`, `torch`, one
/// rendered 96x96 frame per control tick. `None`, with the reason printed, when this machine
/// cannot run it.
///
/// The training step is not about training: the demo graph's `VisionEncoder` and
/// `TemporalEncoder` lower to opaque torch sub-modules (`nodes.N.*` prefix claims), and
/// `load_state_dict(strict=True)` refuses a synthetic checkpoint over them -- so the only
/// checkpoint this path can open is one the trainer itself wrote. One step at `batch = 2` on
/// the bake fixture is the cheapest such bundle, and what the policy learned in it does not
/// matter to a telemetry oracle.
///
/// Three episodes rather than one because [`eval_telemetry_never_blocks_the_run`] has to
/// overflow the loopback socket buffers with image frames (about 100 kB of JSON each): 180
/// ticks is some 18 MB, well past what a loopback pair autotunes to.
fn telemetry_run_inputs(test: &str, dir: &Path) -> Option<(PathBuf, PathBuf)> {
    if cfg!(not(feature = "render")) {
        println!("SKIP {test}: built without the `render` feature");
        return None;
    }
    let Ok(python) = std::env::var("ES_PYTHON") else {
        println!("SKIP {test}: ES_PYTHON is not set, so nothing can train a checkpoint");
        return None;
    };
    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP {test}: {reason}");
        return None;
    }
    if let Err(reason) = es_policy::torch_runtime::is_available() {
        println!("SKIP {test}: {reason}");
        return None;
    }
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let mut task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml");
    task.config.max_episode_steps = 60;
    let task_hash = task.task_hash().expect("task hashes");
    let mut obs =
        es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml");
    obs.task_ref = task_hash;
    let obs_hash = obs.observation_hash().expect("observation hashes");
    let mut learning =
        es_ir::serial::learning_from_toml(&read("learning.toml")).expect("learning.toml");
    let weights = b"es-e4-untrained-placeholder".to_vec();
    learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let deploy =
        es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml");
    let bundle = es_compile::PolicyBundle::build(&task, &obs, &learning, &deploy, &weights)
        .expect("the shortened demo documents pack");
    let untrained = dir.join("untrained.esb");
    std::fs::write(&untrained, bundle).expect("write untrained.esb");

    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 2, 12);
    let recipe = dir.join("training.toml");
    write(
        &recipe,
        &train_fixture_recipe(&untrained, &root, &tiles, 0, "1e-4")
            .replace("steps = 40", "steps = 1")
            .replace("checkpoint_at = [40]", "checkpoint_at = [1]")
            .replace(
                "es-no-such-interpreter",
                &train_toml_path(Path::new(&python)),
            ),
    );
    let trained = bin()
        .current_dir(train_root())
        .args(["train", "--recipe", &train_toml_path(&recipe), "--out"])
        .arg(dir.join("train"))
        .output()
        .expect("run es train");
    assert_eq!(
        trained.status.code(),
        Some(0),
        "es train:\n{}\n{}",
        stdout(&trained),
        stderr_of(&trained)
    );
    let policy = dir.join("train").join("checkpoints").join("1.esb");
    assert!(policy.is_file(), "{}", policy.display());

    let mut ir = demo_evaluation_ir(hex(&task_hash), hex(&obs_hash));
    ir.suites.truncate(1);
    assert_eq!(ir.suites[0].name, "nominal");
    ir.episodes = EpisodeBatch {
        n_episodes: 3,
        seeds: SeedPlan::Explicit(vec![101, 102, 103]),
    };
    let config = dir.join("eval.toml");
    write(
        &config,
        &es_ir::serial::evaluation_to_toml(&ir).expect("evaluation toml"),
    );
    Some((config, policy))
}

/// `es eval run --frames` on the inputs above, into `out`, plus whatever `extra` flags.
fn telemetry_eval_run(config: &Path, policy: &Path, out: &Path, extra: &[&str]) -> Output {
    let run = bin()
        .args(["eval", "run", "--config"])
        .arg(config)
        .arg("--policy")
        .arg(policy)
        .arg("--scene")
        .arg(demo_scene_path())
        .arg("--out")
        .arg(out)
        .arg("--frames")
        .arg(out.join("frames"))
        .args(extra)
        .output()
        .expect("run es eval run");
    let text = format!("{}{}", stdout(&run), String::from_utf8_lossy(&run.stderr));
    // Exit 1 is an acceptance criterion that did not hold, which an untrained policy earns;
    // anything else is a real error. A run that exited 1 for any other reason wrote no
    // report, and saying so here beats a missing-file panic further down.
    assert!(
        matches!(run.status.code(), Some(0 | 1)) && out.join("report.json").is_file(),
        "exit {:?}, report.json {}\n{text}",
        run.status.code(),
        out.join("report.json").is_file()
    );
    run
}

/// A loopback port nobody is listening on, chosen here rather than left to
/// `--telemetry 127.0.0.1:0`, whose port only exists once the run prints it: the client has
/// to be connected and subscribed before the first cell, and retrying a known address until
/// the run binds it has no window at all. The run still prints the address it bound -- which
/// is what makes `:0` usable by hand -- and the assertion below reads that line.
fn free_loopback_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

/// Connects once the run has bound its server. The run binds before it opens the bundle,
/// the scene or either Python interpreter, so this succeeds seconds before the first cell.
fn connect_when_bound(addr: std::net::SocketAddr) -> Client {
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    loop {
        match Client::connect(addr, None, "cli-test") {
            Ok(client) => return client,
            Err(_) if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => panic!("the run never bound {addr}: {e}"),
        }
    }
}

fn read_bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Oracle 1. A client subscribed to streams 1-3 sees `cell.begin`/`cell.end` for every cell,
/// one stream-2 frame per control tick with strictly increasing ticks inside each cell -- the
/// same ticks `events.json` recorded -- and one `Metrics` frame per cell; and the run's
/// `report.json` and `events.json` are byte-identical to a run without the flag.
#[test]
fn eval_telemetry_publishes_every_tick_in_order() {
    const TEST: &str = "eval_telemetry_publishes_every_tick_in_order";
    let dir = scratch_dir("eval-telemetry-order");
    let Some((config, policy)) = telemetry_run_inputs(TEST, &dir) else {
        return;
    };
    telemetry_eval_run(&config, &policy, &dir.join("plain"), &[]);

    let port = free_loopback_port();
    let addr = format!("127.0.0.1:{port}");
    let socket: std::net::SocketAddr = addr.parse().expect("socket addr");
    let received: std::sync::Arc<std::sync::Mutex<Vec<Message>>> = std::sync::Arc::default();
    let reader = {
        let received = std::sync::Arc::clone(&received);
        std::thread::spawn(move || {
            let mut client = connect_when_bound(socket);
            client
                .subscribe(vec![StreamId(1), StreamId(2), StreamId(3)])
                .expect("subscribe");
            // Drains until the run exits and the connection closes.
            while let Ok(msg) = client.recv() {
                received.lock().expect("received").push(msg);
            }
        })
    };
    let live = telemetry_eval_run(&config, &policy, &dir.join("live"), &["--telemetry", &addr]);
    reader.join().expect("reader thread");
    let text = stdout(&live);
    assert!(text.contains(&format!("telemetry: {addr}")), "{text}");

    for name in ["report.json", "events.json"] {
        assert_eq!(
            read_bytes(&dir.join("plain").join(name)),
            read_bytes(&dir.join("live").join(name)),
            "{name} differs with --telemetry"
        );
    }
    let events: BTreeMap<String, Vec<es_eval::StepEvent>> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("live/events.json")).expect("events"),
    )
    .expect("events.json parses");
    let cells: Vec<&String> = events.keys().collect();
    assert_eq!(cells, ["nominal-00", "nominal-01", "nominal-02"]);

    let received = received.lock().expect("received");
    let frames: Vec<&es_telemetry::Frame> = received
        .iter()
        .filter_map(|m| match m {
            Message::Frame(f) => Some(f),
            _ => None,
        })
        .collect();
    assert!(!frames.is_empty(), "no frame arrived: {received:?}");

    // Walk the stream in arrival order: a cell opens, its ticks follow, it closes with a
    // metrics frame, and nothing is attributed to a cell that is not open.
    let mut begun: Vec<String> = Vec::new();
    let mut ended: Vec<String> = Vec::new();
    let mut suites_ended = 0;
    let mut open: Option<String> = None;
    let mut ticks: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
    let mut metrics_after_end = 0;
    for f in &frames {
        match (&f.stream, &f.payload) {
            (StreamId(1), Payload::Event { kind, fields }) => match kind.as_str() {
                "cell.begin" => {
                    assert!(open.is_none(), "cell.begin while {open:?} is open");
                    assert_eq!(fields["suite"], "nominal", "{fields:?}");
                    assert!(fields.contains_key("seed"), "{fields:?}");
                    open = Some(fields["cell"].clone());
                    begun.push(fields["cell"].clone());
                }
                "cell.end" => {
                    assert_eq!(open.as_deref(), Some(fields["cell"].as_str()), "{fields:?}");
                    assert!(fields.contains_key("outcome"), "{fields:?}");
                    ended.push(fields["cell"].clone());
                    open = None;
                }
                "suite.end" => {
                    assert_eq!(fields["suite"], "nominal", "{fields:?}");
                    assert!(fields.contains_key("metric.success_rate"), "{fields:?}");
                    suites_ended += 1;
                }
                other => panic!("unexpected stream-1 event {other}"),
            },
            (StreamId(2), Payload::Scalars(v)) => {
                let cell = open.clone().expect("a tick outside any cell");
                assert_eq!(v.len(), 4, "[frame, tick, source, events]: {v:?}");
                let (frame, tick) = (v[0] as u64, v[1] as u64);
                let seen = ticks.entry(cell).or_default();
                assert!(
                    seen.last().is_none_or(|(_, last)| *last < tick),
                    "ticks not strictly increasing: {seen:?} then {tick}"
                );
                seen.push((frame, tick));
            }
            (StreamId(3), Payload::Metrics(m)) => {
                // Right after a `cell.end`, with the fields the run can fill and never a zero
                // standing in for a number nobody measured (spec 12.4).
                assert!(open.is_none(), "metrics inside an open cell");
                assert!(m.actions_per_sec.is_some(), "{m:?}");
                assert!(m.chunk_underrun_rate.is_some(), "{m:?}");
                assert!(m.p50_end_to_end_latency.is_none(), "{m:?}");
                assert!(m.gpu_memory_peak.is_none(), "{m:?}");
                metrics_after_end += 1;
            }
            (stream, payload) => panic!("unexpected frame on stream {stream:?}: {payload:?}"),
        }
    }
    assert_eq!(
        begun,
        cells.iter().map(|s| (*s).clone()).collect::<Vec<_>>()
    );
    assert_eq!(ended, begun);
    assert_eq!(suites_ended, 1);
    assert_eq!(metrics_after_end, cells.len());
    // One stream-2 frame per control tick, carrying the very record `events.json` got for
    // that step -- the nominal suite drops no observation, so the two sequences are the same
    // and the live viewer's rows are the finished run's rows.
    for (cell, records) in &events {
        let want: Vec<(u64, u64)> = records.iter().map(|r| (r.frame, r.tick.0)).collect();
        assert_eq!(ticks[cell], want, "cell {cell}");
    }
    println!(
        "RAN {TEST}: {} frame(s) over {} cell(s)",
        frames.len(),
        cells.len()
    );
}

/// Oracle 2. A client that subscribes to the image stream and then never reads: the run
/// finishes, its report is unchanged, and the server counted dropped frames -- the producer
/// never waited for the socket.
#[test]
fn eval_telemetry_never_blocks_the_run() {
    const TEST: &str = "eval_telemetry_never_blocks_the_run";
    let dir = scratch_dir("eval-telemetry-slow");
    let Some((config, policy)) = telemetry_run_inputs(TEST, &dir) else {
        return;
    };
    telemetry_eval_run(&config, &policy, &dir.join("plain"), &[]);

    let port = free_loopback_port();
    let addr = format!("127.0.0.1:{port}");
    let socket: std::net::SocketAddr = addr.parse().expect("socket addr");
    // Subscribes and returns the client without ever calling `recv`; the join below keeps it
    // alive -- and its socket open, unread -- until the run is over.
    let stalled = std::thread::spawn(move || {
        let mut client = connect_when_bound(socket);
        client
            .subscribe(vec![StreamId(2), StreamId(4)])
            .expect("subscribe");
        client
    });
    let started = std::time::Instant::now();
    let live = telemetry_eval_run(
        &config,
        &policy,
        &dir.join("live"),
        &["--telemetry", &addr, "--telemetry-image-every", "1"],
    );
    let elapsed = started.elapsed();
    let _client = stalled.join().expect("stalled client thread");

    assert_eq!(
        read_bytes(&dir.join("plain/report.json")),
        read_bytes(&dir.join("live/report.json")),
        "report.json differs with a stalled client"
    );
    let text = stdout(&live);
    let stats = text
        .lines()
        .find(|l| l.starts_with("telemetry:") && l.contains("dropped"))
        .unwrap_or_else(|| panic!("no telemetry stats line:\n{text}"));
    let words: Vec<&str> = stats.split_whitespace().collect();
    let at = words
        .iter()
        .position(|w| *w == "dropped")
        .expect("the word `dropped`");
    let dropped: u64 = words[at - 1].parse().unwrap_or_else(|_| panic!("{stats}"));
    assert!(
        dropped > 0,
        "a client that never reads must lose frames, not stall the run: {stats}"
    );
    println!("RAN {TEST}: {stats} ({elapsed:?})");
}

// --- packet M7/T4: the learning-rate schedule -------------------------------------------

/// Oracle 3. The schedule is a document value, so it moves `identity_hash`; `scheduler.json`
/// and `optimizer.json` carry what the trainer was told, and `config.json`'s plan carries the
/// flags it would have been told with.
///
/// Like `train_identity_is_a_function_of_the_recipe`, the recipe names an interpreter that
/// cannot exist: every assertion here is about the identity written *before* the run, which
/// is the half of the split that needs no Python.
#[test]
fn train_identity_moves_with_the_schedule() {
    let dir = scratch_dir("train-schedule");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 2, 12);
    let base = train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4");

    let go = |name: &str, body: &str| -> (serde_json::Value, PathBuf) {
        let recipe = dir.join(format!("{name}.toml"));
        write(&recipe, body);
        let out = dir.join(name);
        run_train(&train_toml_path(&recipe), &out, &[]);
        (train_lock(&out), out)
    };
    let scheduled = |warmup: u32| {
        base.replace(
            "device = ",
            &format!(
                "schedule = {{ kind = \"warmup_cosine\", warmup = {warmup}, lr_min = 1e-6 }}\n\
                 grad_clip = 1.0\ndevice = "
            ),
        )
    };

    let (plain, plain_out) = go("plain", &base);
    let (ten, ten_out) = go("warmup-10", &scheduled(10));
    let (twenty, _) = go("warmup-20", &scheduled(20));
    assert_ne!(
        plain["identity_hash"], ten["identity_hash"],
        "a schedule did not move identity_hash"
    );
    assert_ne!(
        ten["identity_hash"], twenty["identity_hash"],
        "`warmup` did not move identity_hash"
    );

    let slot = |out: &Path, name: &str| -> String {
        let path = out.join("training").join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    };
    let json = |out: &Path, name: &str| -> serde_json::Value {
        serde_json::from_str(&slot(out, name)).expect("a training slot is JSON")
    };
    let scheduler = json(&ten_out, "scheduler.json");
    assert_eq!(scheduler["kind"], "warmup_cosine", "{scheduler}");
    assert_eq!(scheduler["warmup"], 10, "{scheduler}");
    assert_eq!(scheduler["lr_min"], 1e-6, "{scheduler}");
    // The cosine's period is the length of the run, so `steps` is part of the schedule.
    assert_eq!(scheduler["total_steps"], 40, "{scheduler}");
    let optimizer = json(&ten_out, "optimizer.json");
    assert_eq!(optimizer["grad_clip"], 1.0, "{optimizer}");
    assert_eq!(optimizer["weight_decay"], 0.01, "{optimizer}");

    // ...and the flags are on the line the trainer would have been run with.
    let plan = json(&ten_out, "config.json")["plan"].to_string();
    assert!(
        plan.contains("--schedule warmup_cosine --warmup-steps 10 --lr-min 0.000001 --grad-clip 1"),
        "{plan}"
    );

    // The recipe that names no schedule is the run of before, hash included: the same nine
    // pre-run slots it had before this packet existed (design note `training-recipe.md` 10).
    assert_eq!(
        slot(&plain_out, "scheduler.json"),
        "{\"kind\":\"constant\",\"lr\":0.0001}\n"
    );
    assert!(
        !slot(&plain_out, "optimizer.json").contains("grad_clip"),
        "a run with no clip declared one"
    );
}

// --- packet M7/T5: the pretrained backbone ----------------------------------------------

/// `write_demo_bundle`'s graph with `pretrained = true` on its `VisionEncoder` — the shape
/// `tests/fixtures/visible-learning/learning-pretrained.toml` has, built here so the test
/// does not also depend on the file's weights placeholder.
fn write_pretrained_bundle(dir: &Path) -> PathBuf {
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let mut learning = es_ir::serial::learning_from_toml(&read("learning-pretrained.toml"))
        .expect("learning-pretrained.toml");
    let weights = b"es-t5-pretrained-placeholder".to_vec();
    learning.policy.weights = es_ir::learning::WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = es_compile::PolicyBundle::build(
        &es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the pretrained demo documents pack into a bundle");
    let path = dir.join("pretrained.esb");
    std::fs::write(&path, bytes).expect("write pretrained.esb");
    path
}

/// A `base_model` pair on disk: the safetensors `bytes` and the lock file beside it, with
/// `edit` applied to the lock before it is written.
fn write_base_model(
    dir: &Path,
    name: &str,
    bytes: &str,
    edit: impl FnOnce(&mut serde_json::Value),
) -> PathBuf {
    let weights = dir.join(format!("{name}.safetensors"));
    write(&weights, bytes);
    let mut lock = serde_json::json!({
        "source": es_data::training::BASE_MODEL_SOURCE,
        "torchvision": "0.26.0+cu129",
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "sha256_upstream":
            "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        "blake3": hex(blake3::hash(bytes.as_bytes()).as_bytes()),
        "dropped": "*.num_batches_tracked",
        "license": "BSD-3-Clause",
        "license_url": "https://github.com/pytorch/vision/blob/main/LICENSE",
    });
    edit(&mut lock);
    write(
        &dir.join(format!("{name}.lock.json")),
        &(lock.to_string() + "\n"),
    );
    weights
}

/// Oracle 4 of packet M7/T5, first half. Every way a `base_model` can be wrong is refused by
/// the name of what is wrong, and nothing is trained.
///
/// The recipe names an interpreter that cannot exist, exactly as the other `es train` tests
/// do, so nothing here needs Python: every refusal below happens before the trainer is
/// reached, and the one accepted case stops at the interpreter instead.
#[test]
fn train_refuses_a_mismatched_base_model() {
    let dir = scratch_dir("train-base-model");
    let bundle = write_pretrained_bundle(&dir);
    let plain = write_demo_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 1, 12);

    // The pin is what the repository claims the real artifact hashes to; nothing here has
    // 45 MB of ImageNet, so the *pinned* case is the oracle-3 artifact's job and what this
    // test judges is every way the three claims can disagree.
    let wrong = write_base_model(&dir, "wrong", "not the pinned backbone", |_| {});
    let lying = write_base_model(&dir, "lying", "not the pinned backbone", |lock| {
        lock["blake3"] = serde_json::json!("0".repeat(64));
    });
    let unlicensed = write_base_model(&dir, "unlicensed", "not the pinned backbone", |lock| {
        lock["license"] = serde_json::json!("");
    });

    let recipe_with = |name: &str, bundle: &Path, base: Option<&Path>| -> String {
        let mut body = train_fixture_recipe(bundle, &root, &tiles, 0, "1e-4");
        if let Some(base) = base {
            body = body.replace(
                "[run]",
                &format!("base_model = \"{}\"\n[run]", train_toml_path(base)),
            );
        }
        let recipe = dir.join(format!("{name}.toml"));
        write(&recipe, &body);
        train_toml_path(&recipe)
    };
    let refuse = |name: &str, bundle: &Path, base: Option<&Path>| -> String {
        let out = run_train(&recipe_with(name, bundle, base), &dir.join(name), &[]);
        assert!(
            !out.status.success(),
            "{name} was accepted:\n{}",
            stdout(&out)
        );
        stderr_of(&out)
    };

    // 1. The lock file disagrees with the bytes beside it.
    let said = refuse("lying", &bundle, Some(&lying));
    assert!(said.contains("does not hash to what"), "{said}");
    assert!(said.contains("lying.lock.json"), "{said}");

    // 2. The bytes disagree with the pin, which is the claim the repository makes.
    let said = refuse("wrong", &bundle, Some(&wrong));
    assert!(said.contains("not the pinned backbone"), "{said}");
    assert!(
        said.contains(es_data::training::RESNET18_IMAGENET1K_V1_BLAKE3),
        "the refusal does not name the pin: {said}"
    );

    // 3. A provenance record with an empty licence tracks nothing (spec 19.3).
    let said = refuse("unlicensed", &bundle, Some(&unlicensed));
    assert!(said.contains("license") && said.contains("19.3"), "{said}");

    // 4. A pretrained bundle with no `base_model` at all: the tensors would come from
    //    nowhere, and the only other way to get them is the network (spec 2.5).
    let said = refuse("no-base", &bundle, None);
    assert!(said.contains("pretrained = true"), "{said}");
    assert!(said.contains("base_model"), "{said}");

    // 5. ...and the reverse: weights nothing in the graph would ever read.
    let said = refuse("no-encoder", &plain, Some(&wrong));
    assert!(said.contains("base_model"), "{said}");
    assert!(said.contains("pretrained = true"), "{said}");

    // 6. The missing lock file is named, not the weights.
    let orphan = dir.join("orphan.safetensors");
    write(&orphan, "");
    let said = refuse("orphan", &bundle, Some(&orphan));
    assert!(said.contains("orphan.lock.json"), "{said}");
}

/// Oracle 4 of packet M7/T5, second half. A `base_model` that passes writes spec 19.3's
/// `base_model.lock` from the lock file — real values, not `{"source": "none"}` — and puts
/// `--init-backbone` on the trainer's line.
///
/// The pin cannot be met without the 45 MB artifact, which is not committed and is oracle 3's
/// subject. So the two halves are judged where each is observable with no artifact on disk:
/// the flag through `--dry-run`, which reads no file, and the slot through the same headless
/// `Training::pre_run` the shell calls, given the lock the shell would have verified. Nothing
/// is faked — what is not exercised here is `Backbone::verify`'s success path, and that is
/// exactly what oracle 3 and the server run of oracle 5 cover.
#[test]
fn train_writes_base_model_lock_from_the_lock_file() {
    let dir = scratch_dir("train-base-lock");
    let bundle = write_pretrained_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 1, 12);
    let base = dir.join("resnet18-imagenet1k-v1.safetensors");

    let recipe = dir.join("training.toml");
    write(
        &recipe,
        &train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4").replace(
            "[run]",
            &format!("base_model = \"{}\"\n[run]", train_toml_path(&base)),
        ),
    );
    // `--dry-run` reads no file, so the plan is observable here with no artifact on disk.
    let out = run_train(&train_toml_path(&recipe), &dir.join("dry"), &["--dry-run"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
    let plan = stdout(&out);
    assert!(
        plan.contains(&format!("--init-backbone {}", train_toml_path(&base))),
        "{plan}"
    );
    // `frozen` is the IR's and never reaches the command line (design note 5.3).
    assert!(
        !plan.contains("--frozen") && !plan.contains("--freeze"),
        "{plan}"
    );

    // And the slot the verified lock produces, through the same function the shell calls.
    let parsed =
        es_data::training::Recipe::parse(&std::fs::read_to_string(&recipe).expect("the recipe"))
            .expect("the recipe parses");
    let built = es_data::training::Plan::build(
        &parsed,
        &dir.join("dry"),
        "python",
        &["python".to_owned()],
        None,
        false,
    )
    .expect("the plan builds");
    let lock = es_data::training::Backbone {
        source: es_data::training::BASE_MODEL_SOURCE.to_owned(),
        url: "https://download.pytorch.org/models/resnet18-f37072fd.pth".to_owned(),
        sha256_upstream: "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
            .to_owned(),
        blake3: es_data::training::RESNET18_IMAGENET1K_V1_BLAKE3.to_owned(),
        dropped: "*.num_batches_tracked".to_owned(),
        license: "BSD-3-Clause".to_owned(),
        license_url: "https://github.com/pytorch/vision/blob/main/LICENSE".to_owned(),
    };
    let facts = es_data::training::DatasetFacts {
        hashes: es_ir::DatasetHash {
            content: [1; 32],
            schema: [2; 32],
            split: [3; 32],
        },
        episodes: 1,
        frames: 12,
        recorded_task: None,
        split_source: "all-train",
    };
    let without = es_data::training::Training::pre_run(
        &parsed,
        &built,
        &dir.join("dry"),
        "python",
        &facts,
        None,
        None,
    )
    .expect("pre_run");
    let with = es_data::training::Training::pre_run(
        &parsed,
        &built,
        &dir.join("dry"),
        "python",
        &facts,
        Some(&lock),
        None,
    )
    .expect("pre_run");
    let slot: serde_json::Value =
        serde_json::from_str(with.file("base_model.lock")).expect("base_model.lock is JSON");
    assert_eq!(
        slot["blake3"],
        es_data::training::RESNET18_IMAGENET1K_V1_BLAKE3
    );
    assert_eq!(slot["license"], "BSD-3-Clause");
    assert_eq!(slot["source"], es_data::training::BASE_MODEL_SOURCE);
    // Spec 19.3: the licence is part of the run's identity, so the slot is not decoration.
    assert_eq!(with.identity().base_model.license, "BSD-3-Clause");
    assert_ne!(
        without.hash().expect("hash"),
        with.hash().expect("hash"),
        "a real base_model did not move training_hash"
    );
}

// --- packet M7/T6: training augmentation -------------------------------------------------

const AUGMENTED_HEADER: &str = "\
# Observation IR (spec 7) for the SO-101 cube-into-bin demo, with the training augmentation
# chain -- plan U, packet M7/T6.
#
# Generated by `cargo test -p es --test cli -- --ignored generate_augmented_observation_fixture`
# from observation.toml, which is unchanged: this file is that document plus three nodes on
# the image path, so `task_ref` and every state branch are copies.
#
#   node 1 Normalize -> node 7 Pad{4,4,4,4} -> node 8 Crop{Random 96x96}
#                    -> node 9 Augment{ColorJitter brightness 0.2 contrast 0.2, training_only}
#
# The crop is `Crop { CropMode::Random }` and not `Augment { RandomCrop }` because only the
# first propagates geometry: an `Augment` port keeps its incoming ImageSpec, so the declared
# output would disagree with propagation and XIR's TYPE-020 refuses the bundle. Both spell
# the same thing -- a per-sample offset whose nominal rectangle is the centred one, which is
# how es-ir documents CropMode::Random -- and the trainer treats them as one kind. See the
# design note docs/design/training-recipe.md section 13.
#
# `Pad` grows the 96x96 canvas to 104x104 and moves the principal point with the old origin
# (cx 48 -> 52); the `RandomCrop` takes it back to 96x96. **In training** the crop offset is
# drawn per sample and per step by `python/es/augment.py` -- DrQ's random shift -- from the
# tensor `es dataset bake --for-training` writes at the chain boundary, which is the 104x104
# one. **In evaluation** INV-15 disables augmentation, and the plan lowers the pair to the
# deterministic centre crop (offset 4,4) through `Crop`'s op and `Crop`'s intrinsics
# transform (INV-14): cx 52 - 4 = 48, so the output ImageSpec is the un-augmented document's
# and so are the pixels, bit for bit.
#
# The output port `rgb_overhead` therefore carries exactly the PortType observation.toml's
# does -- 3x96x96, Normalized{0,1}, the same ImageSpec -- so learning.toml is unchanged and
# `learning_hash` does not move. `observation_hash` does: the graph has three more nodes.
# An Evaluation IR names `observation`, so evaluating a policy trained on this document is a
# new `evaluation_hash` (spec 13.3) -- see docs/design/training-recipe.md section 13.
";

/// Regenerates `tests/fixtures/visible-learning/observation-augmented.toml` from the
/// committed `observation.toml`, and prints the `observation_hash` pair the design note
/// records. Run explicitly:
///
///     cargo test -p es --test cli -- --ignored generate_augmented_observation_fixture
#[test]
#[ignore = "fixture generator; run explicitly"]
fn generate_augmented_observation_fixture() {
    use es_ir::graph::{NodeId, PortRef};
    use es_ir::image::{ImageSpec, Intrinsics};
    use es_ir::observation::{AugmentKind, CropMode, Io, ObservationNode, ObservationOutput, OUT};
    use es_ir::types::Shape;
    // Spec 1.4: goldens and fixtures are CI read-only, and `cargo test -- --include-ignored`
    // runs every ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!(
            "SKIP generate_augmented_observation_fixture: set ES_GENERATE_GOLDENS=1 to regenerate"
        );
        return;
    }

    let committed = std::fs::read_to_string(vl_fixture("observation.toml")).expect("read");
    let base = es_ir::serial::observation_from_toml(&committed).expect("observation.toml");
    let mut obs = base.clone();

    // The image output's producer is the node the chain is appended to.
    let out = obs.outputs["rgb_overhead"].clone();
    let normalized = out.ty.clone();
    let spec = normalized
        .image
        .expect("the image port carries an ImageSpec");
    let pad = 4u32;

    let padded_spec = ImageSpec {
        width: spec.width + 2 * pad,
        height: spec.height + 2 * pad,
        intrinsics: Intrinsics {
            cx: spec.intrinsics.cx + f64::from(pad),
            cy: spec.intrinsics.cy + f64::from(pad),
            ..spec.intrinsics
        },
        ..spec
    };
    let mut padded = normalized.clone();
    padded.shape = Shape::new([
        3,
        u64::from(padded_spec.height),
        u64::from(padded_spec.width),
    ]);
    padded.image = Some(padded_spec);

    let (p, crop, jitter) = (NodeId(7), NodeId(8), NodeId(9));
    obs.graph.insert(
        p,
        ObservationNode::Pad {
            left: pad,
            top: pad,
            right: pad,
            bottom: pad,
            io: Io::unary(normalized.clone(), padded.clone()),
        },
    );
    // `Crop { CropMode::Random }`, not `Augment { RandomCrop }`: the IR propagates geometry
    // for the first and not for the second, and a document whose declared output disagrees
    // with propagation is refused by `XIR`'s `TYPE-020` before a bundle can be built. Both
    // mean "per-sample offset, centred nominally"; see the design note's section 13.
    obs.graph.insert(
        crop,
        ObservationNode::Crop {
            mode: CropMode::Random {
                width: spec.width,
                height: spec.height,
            },
            rescale_intrinsics: true,
            io: Io::unary(padded, normalized.clone()),
        },
    );
    obs.graph.insert(
        jitter,
        ObservationNode::Augment {
            kind: AugmentKind::ColorJitter {
                brightness: 0.2,
                contrast: 0.2,
                saturation: 0.0,
                hue: 0.0,
            },
            training_only: true,
            io: Io::unary(normalized.clone(), normalized.clone()),
        },
    );
    obs.graph.connect(out.port.node, OUT, p, "in0");
    obs.graph.connect(p, OUT, crop, "in0");
    obs.graph.connect(crop, OUT, jitter, "in0");
    obs.outputs.insert(
        "rgb_overhead".to_owned(),
        ObservationOutput {
            port: PortRef::new(jitter, OUT),
            ty: normalized,
        },
    );

    let errors: Vec<_> = obs
        .validate()
        .into_iter()
        .filter(es_ir::diag::Diagnostic::is_error)
        .collect();
    assert!(
        errors.is_empty(),
        "the augmented document does not validate: {errors:?}"
    );
    write(
        &vl_fixture("observation-augmented.toml"),
        &format!(
            "{AUGMENTED_HEADER}\n{}",
            es_ir::serial::observation_to_toml(&obs).expect("observation toml")
        ),
    );
    println!(
        "RAN generate_augmented_observation_fixture:\n  observation.toml           {}\n  \
         observation-augmented.toml {}",
        hex(&base.observation_hash().expect("committed hash")),
        hex(&obs.observation_hash().expect("augmented hash")),
    );
}

/// `write_demo_bundle` with `observation-augmented.toml` in place of `observation.toml`.
fn write_augmented_bundle(dir: &Path) -> PathBuf {
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let mut learning =
        es_ir::serial::learning_from_toml(&read("learning.toml")).expect("learning.toml");
    let weights = b"es-t6-augmented-placeholder".to_vec();
    learning.policy.weights = es_ir::learning::WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = es_compile::PolicyBundle::build(
        &es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&read("observation-augmented.toml"))
            .expect("observation-augmented.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the augmented demo documents pack into a bundle");
    let path = dir.join("augmented.esb");
    std::fs::write(&path, bytes).expect("write augmented.esb");
    path
}

/// Oracle 4 of packet M7/T6. A bundle whose Observation IR declares a `training_only` chain
/// writes a real `augmentation.json` and a real `seed.json.augmentation`, and both reach
/// `identity_hash`; the un-augmented bundle writes the `{"kind": "none"}` it always wrote, so
/// every measured run's identity is where it was.
///
/// Like the other `es train` tests here the recipe names an interpreter that cannot exist:
/// every assertion is about the identity written *before* the run, which needs no Python.
#[test]
fn train_identity_moves_with_augmentation() {
    let dir = scratch_dir("train-augment");
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 2, 12);
    let plain_bundle = write_demo_bundle(&dir);
    let bundle = write_augmented_bundle(&dir);

    let go = |name: &str, body: &str| -> (serde_json::Value, PathBuf) {
        let recipe = dir.join(format!("{name}.toml"));
        write(&recipe, body);
        let out = dir.join(name);
        run_train(&train_toml_path(&recipe), &out, &[]);
        (train_lock(&out), out)
    };
    let base = train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4");
    let (augmented, augmented_out) = go("augmented", &base);
    let (reseeded, _) = go(
        "reseeded",
        &base.replace("seed = 0", "seed = 0\naugmentation_seed = 7"),
    );
    let (plain, plain_out) = go(
        "plain",
        &train_fixture_recipe(&plain_bundle, &root, &tiles, 0, "1e-4"),
    );

    let slot = |out: &Path, name: &str| -> serde_json::Value {
        let path = out.join("training").join(name);
        serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        )
        .expect("a training slot is JSON")
    };
    let chain = slot(&augmented_out, "augmentation.json");
    assert_eq!(chain["kind"], "observation-ir", "{chain}");
    assert_eq!(chain["seed"], 0, "{chain}");
    let nodes = chain["chains"]["rgb_overhead"]
        .as_array()
        .unwrap_or_else(|| panic!("{chain}"));
    assert_eq!(nodes.len(), 2, "{chain}");
    assert_eq!(nodes[0]["kind"], "RandomCrop");
    assert_eq!(nodes[0]["width"], 96);
    assert_eq!(nodes[1]["kind"], "ColorJitter");
    assert_eq!(nodes[1]["brightness"], 0.2);
    // The node id is the RNG's `node_index` coordinate, so it is part of the file.
    assert!(nodes.iter().all(|n| n["node"].is_number()), "{chain}");
    assert_eq!(slot(&augmented_out, "seed.json")["augmentation"], 0);

    // ...and the plan the run recorded is the one that carries the two flags.
    let plan = slot(&augmented_out, "config.json")["plan"].to_string();
    assert!(plan.contains("--for-training"), "{plan}");
    assert!(
        plan.contains("--augmentation training/augmentation.json"),
        "{plan}"
    );

    // The un-augmented bundle is the run of before, slot for slot.
    assert_eq!(
        slot(&plain_out, "augmentation.json"),
        serde_json::json!({"kind": "none"})
    );
    assert_eq!(
        slot(&plain_out, "seed.json"),
        serde_json::json!({"global": 0, "dataloader": 0, "augmentation": {"unset": true}})
    );
    let plain_plan = slot(&plain_out, "config.json")["plan"].to_string();
    assert!(!plain_plan.contains("--for-training"), "{plain_plan}");
    assert!(!plain_plan.contains("--augmentation"), "{plain_plan}");

    assert_ne!(
        augmented["identity_hash"], plain["identity_hash"],
        "the augmented document did not move identity_hash"
    );
    assert_ne!(
        augmented["identity_hash"], reseeded["identity_hash"],
        "augmentation_seed did not move identity_hash"
    );
    assert_ne!(
        augmented["files"]["augmentation.json"], plain["files"]["augmentation.json"],
        "one augmentation.json digest for two documents"
    );
}

/// Oracle 3 of packet M7/T6, the half that needs the real command: `--for-training` writes
/// the image port at `104x104` and the chain into `manifest.json`, and the same bake without
/// the flag writes the `96x96` the un-augmented document writes.
///
/// The shapes are the claim. `augmentation.json` tells the trainer what to apply; this is the
/// tensor it applies it *to*, and if the two disagreed the reshape in `train_act.py` would be
/// the only thing standing between a silent mis-crop and a training run.
#[test]
fn dataset_bake_for_training_writes_the_chain() {
    if skip_without_bake_model("dataset_bake_for_training_writes_the_chain") {
        return;
    }
    let dir = scratch_dir("bake-for-training");
    let bundle = write_augmented_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 1, 3);
    let scene = demo_scene_path();

    let bake = |out: &Path, flag: &[&str]| -> serde_json::Value {
        let result = bin()
            .args(["dataset", "bake", "--policy", bundle.to_str().unwrap()])
            .args(["--out", out.to_str().unwrap()])
            .args(["--frames", tiles.to_str().unwrap()])
            .args(["--scene", scene.to_str().unwrap()])
            .args(flag)
            .arg(root.to_str().unwrap())
            .output()
            .expect("run es dataset bake");
        assert_eq!(
            result.status.code(),
            Some(0),
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        serde_json::from_str(
            &std::fs::read_to_string(out.join("manifest.json")).expect("manifest.json"),
        )
        .expect("manifest.json parses")
    };

    let training = bake(&dir.join("training"), &["--for-training"]);
    assert_eq!(
        training["tensors"]["rgb_overhead"]["shape"],
        serde_json::json!([3, 104, 104]),
        "{training}"
    );
    assert_eq!(
        training["augmentation"]["rgb_overhead"][0]["kind"], "RandomCrop",
        "{training}"
    );
    assert_eq!(
        training["augmentation"]["rgb_overhead"][1]["kind"], "ColorJitter",
        "{training}"
    );

    let evaluation = bake(&dir.join("evaluation"), &[]);
    assert_eq!(
        evaluation["tensors"]["rgb_overhead"]["shape"],
        serde_json::json!([3, 96, 96]),
        "{evaluation}"
    );
    assert!(
        evaluation.get("augmentation").is_none(),
        "a bake without the flag recorded a chain: {evaluation}"
    );
    // ...and those are the bytes the un-augmented document bakes: `Pad(4)` then the centre
    // crop is the image that went in (INV-14, INV-15).
    let plain = write_demo_bundle(&dir);
    let unaugmented = dir.join("unaugmented");
    let result = bin()
        .args(["dataset", "bake", "--policy", plain.to_str().unwrap()])
        .args(["--out", unaugmented.to_str().unwrap()])
        .args(["--frames", tiles.to_str().unwrap()])
        .args(["--scene", scene.to_str().unwrap()])
        .arg(root.to_str().unwrap())
        .output()
        .expect("run es dataset bake");
    assert_eq!(result.status.code(), Some(0));
    let episode = "episode_000000.safetensors";
    assert_eq!(
        std::fs::read(dir.join("evaluation").join(episode)).expect("the augmented bake"),
        std::fs::read(unaugmented.join(episode)).expect("the un-augmented bake"),
        "the augmented document's evaluation bake is not the un-augmented document's"
    );
}

/// Packet M7/T8 oracle 4 — **`--jobs` still splits cells, not episodes, and the design note
/// says why.**
///
/// The packet's question was whether `Env::seek_episode` (which ships, and is bitwise: see
/// `cargo test -p es-env --test seek`) lets a shard be one episode rather than one whole
/// suite. Measured on the committed demo documents by running the pre-T8 and the T8 build over
/// the nominal suite: no. Two things carry from episode `k-1` into episode `k` and show in the
/// artifacts -- the Safety Plane's `ViolationRate` window and `StepEvent::tick` -- and both are
/// decisions for the M7 review, not fixes for this packet. So no flag was added, and this is
/// what the shipped state looks like from the CLI's side.
#[test]
fn eval_jobs_splits_episodes() {
    // 1. The finding is written down where the review will look for it, with its evidence.
    let note = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/evaluation-execution.md"),
    )
    .expect("the evaluation-execution design note");
    for wanted in [
        "2.7 Sharding",
        "ViolationRate` window carries across the episode boundary",
        "tick 24",
    ] {
        assert!(
            note.contains(wanted),
            "the design note is missing {wanted:?}"
        );
    }

    // 2. The worker protocol's unit is still the cell: a shard is a list of cells, each
    //    carrying the results `record_cell` computed inside the worker.
    let shard = es_eval::Shard {
        cells: vec![es_eval::ShardCell {
            cell: 0,
            results: Vec::new(),
            samples: Vec::new(),
        }],
        backend: None,
        events: BTreeMap::new(),
    };
    let text = serde_json::to_string(&shard).expect("a shard serializes");
    assert!(text.contains("\"results\""), "{text}");
    assert!(!text.contains("\"episodes\""), "{text}");

    // 3. And the CLI says so, including the clamp a one-suite evaluation runs into.
    let help = bin()
        .args(["eval", "run", "--help"])
        .output()
        .expect("run es eval run --help");
    let help = stdout(&help);
    assert!(
        help.contains("The split is by suite and not by episode"),
        "{help}"
    );
    assert!(help.contains("N is clamped to the suite count"), "{help}");

    // 4. The behaviour itself, where a real run is possible: the demo's nominal suite is one
    //    cell, so `--jobs 4` clamps to one and spawns no worker at all -- no `shards/`.
    let ran = eval_jobs_one_suite_runs_one_wide();
    println!("RAN eval_jobs_splits_episodes{ran}");
}

/// `--jobs 4` over a one-suite evaluation, when this machine can run one. Returns what to
/// append to the `RAN` line; a machine without the pieces prints its own `SKIP` and returns "".
///
/// Driven by `--expert` (packet M7/T2), so this needs `MuJoCo` and a renderer but no Torch and
/// no trained bundle: what is under test is the partition, not the policy.
fn eval_jobs_one_suite_runs_one_wide() -> String {
    if cfg!(not(feature = "render")) {
        println!("SKIP eval_jobs_splits_episodes (the live run): built without `render`");
        return String::new();
    }
    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP eval_jobs_splits_episodes (the live run): {reason}");
        return String::new();
    }

    let dir = scratch_dir("eval-jobs-episodes");
    let bundle = write_demo_bundle(&dir);
    // The committed demo evaluation, cut to its first suite and two episodes: one cell, which
    // is what `--jobs` has nothing to split.
    let mut ir = es_ir::serial::evaluation_from_toml(
        &std::fs::read_to_string(vl_fixture("evaluation.toml")).expect("evaluation.toml"),
    )
    .expect("evaluation.toml parses");
    ir.suites.truncate(1);
    ir.episodes.n_episodes = 2;
    ir.episodes.seeds = es_ir::evaluation::SeedPlan::Explicit(vec![101, 102]);
    ir.acceptance.clear();
    let config = dir.join("nominal.toml");
    write(
        &config,
        &es_ir::serial::evaluation_to_toml(&ir).expect("evaluation toml"),
    );

    let out = dir.join("out");
    let run = bin()
        .args(["eval", "run", "--config"])
        .arg(&config)
        .arg("--policy")
        .arg(&bundle)
        .arg("--scene")
        .arg(demo_scene_path())
        .arg("--out")
        .arg(&out)
        .arg("--frames")
        .arg(out.join("frames"))
        .args(["--expert", "so101-pick-place", "--jobs", "4"])
        .output()
        .expect("run es eval run --jobs 4");
    let text = format!("{}{}", stdout(&run), String::from_utf8_lossy(&run.stderr));
    if run.status.code() == Some(3) {
        println!(
            "SKIP eval_jobs_splits_episodes (the live run): {}",
            text.trim()
        );
        return String::new();
    }
    assert!(
        matches!(run.status.code(), Some(0 | 1)),
        "exit {:?}\n{text}",
        run.status.code()
    );
    // One cell, so the clamp took --jobs 4 down to 1: no worker was spawned and nothing
    // announced a partition.
    assert!(
        !out.join("shards").exists(),
        "a one-suite evaluation spawned workers:\n{text}"
    );
    assert!(!text.contains("worker(s)"), "{text}");
    assert!(out.join("report.json").is_file(), "{text}");
    " (with the live one-suite run)".to_owned()
}

// --- packet M7/R5: the path-traced observation documents -----------------------------------

/// The exposure `task-pt.toml` declares. Not the neutral 1.0: on the `Pt` path the demo cell
/// is lit by one 0.24 m emissive panel and nothing else, so at exposure 1.0 the 96x96
/// observation has a mean byte of 34 against the rasterized observation's 188 -- a picture of
/// the right scene that a policy would have to learn in the dark. `64` is the rung of the
/// exposure sweep in `pt_observation_cost_and_ssim` whose mean byte (184) is closest to that
/// 188, so the two documents differ in how the light got there and not in how bright the
/// picture is, and a `success_rate` measured under one can be read beside the other
/// (`docs/design/renderer.md` section 12).
const PT_EXPOSURE: f32 = 64.0;

const PT_TASK_HEADER: &str = "\
# Task IR (spec 6) for the SO-101 cube-into-bin demo, rendered by the path tracer -- plan U,
# packet M7/R5.
#
# Generated by `cargo test -p es --test cli -- --ignored generate_pt_fixtures` from task.toml,
# which is unchanged: this file is that document with `render` set on the one sensor channel.
# Every other byte, every other hash input and the whole graph are copies.
#
#   [body.observation_spec.channels.rgb_overhead.source.Sensor.render]
#   path = \"pt\", spp = 64, bounces = 3, exposure = 64.0, tonemap = \"reinhard\"
#
# `render` declares how the *simulation* produces the sensor, which is the Task IR's business;
# what the sensor is -- 96x96, Rgb, sRGB, these intrinsics -- stays `ImageSpec`'s and is
# untouched (INV-14), so observation.toml and observation-augmented.toml both serve this
# document unchanged and neither `observation_hash` moves.
#
# `task_hash` does move, because `render` is not the default one, and that is the point: a
# path-traced observation is a different comparison (spec 13.3), so it gets its own Evaluation
# IR in evaluation-pt.toml rather than being quietly judged by the committed one.
";

const PT_OBSERVATION_HEADER: &str = "\
# Observation IR (spec 7) for the path-traced SO-101 demo -- plan U, packet M7/R5.
#
# Generated by `cargo test -p es --test cli -- --ignored generate_pt_fixtures`; the committed
# observation.toml with `task_ref` pointed at task-pt.toml. **The graph is byte-identical** --
# every node, every port, every ImageSpec and every intrinsic is a copy, because how the
# simulation produces the sensor is the Task IR's business and the preprocessing does not
# change when the renderer does.
#
# `observation_hash` still moves, and there is no way for it not to: `task_ref` is hash input
# (crates/es-ir/src/observation.rs, spec 5.3), and XIR_001 requires it to equal the task's own
# hash. So spec 7.4's `several Observation IRs share one task_hash` has no mirror image -- one
# Observation IR cannot serve two Task IRs -- and a new Task IR always drags a new Observation
# IR document behind it, even when the pipeline it describes is the same bytes.
";

const PT_EVALUATION_HEADER: &str = "\
# Evaluation IR (spec 10) for the path-traced SO-101 demo -- plan U, packet M7/R5.
#
# Generated by `cargo test -p es --test cli -- --ignored generate_pt_fixtures`; the committed
# evaluation.toml with `task` and `observation` replaced by task-pt.toml's and
# observation-pt.toml's own hashes. Same 16 held-out seeds, same six suites, same perturbation
# parameters, same acceptance -- a diff with the two fields put back is empty.
#
# It is a separate document and not a widened one because spec 13.3 says so: the Task IR it
# names is a different Task IR, so a report produced under it is not chained to a report
# produced under evaluation.toml. A reader may compare the two by judgement; the hash chain
# does not.
";

/// An untrained bundle from four documents named by path, with the placeholder weights
/// `--expert` never loads (packet M7/R5).
fn pack_untrained(task: &Path, observation: &Path, learning: &Path, deployment: &Path) -> Vec<u8> {
    let read =
        |p: &Path| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut learning_ir =
        es_ir::serial::learning_from_toml(&read(learning)).expect("the Learning IR parses");
    let weights = b"es-v1-expert-placeholder".to_vec();
    learning_ir.policy.weights = es_ir::learning::WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    es_compile::PolicyBundle::build(
        &es_ir::serial::task_from_toml(&read(task)).expect("the Task IR parses"),
        &es_ir::serial::observation_from_toml(&read(observation))
            .expect("the Observation IR parses"),
        &learning_ir,
        &es_ir::serial::deployment_from_toml(&read(deployment)).expect("the Deployment IR parses"),
        &weights,
    )
    .unwrap_or_else(|e| panic!("the four documents pack into a bundle: {e}"))
}

/// The demo's bundle built from a named Task IR + Observation IR pair, so the `Rs` and the
/// `Pt` documents can be packed by the same three lines (packet M7/R5).
fn write_demo_bundle_from(dir: &Path, task: &str, observation: &str, name: &str) -> PathBuf {
    let bytes = pack_untrained(
        &vl_fixture(task),
        &vl_fixture(observation),
        &vl_fixture("learning.toml"),
        &vl_fixture("deployment.toml"),
    );
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write bundle");
    path
}

/// Packs an untrained bundle from four documents this repository does not have to own, so a
/// measurement can build the bundle it is about without a scratch tree of swapped fixtures
/// (which is how M5/V18 and M7/U1 had to do it). `es policy pack` takes a bundle and weights,
/// not four documents, and this packet is not the one that gives it a second mode.
///
///     ES_PACK_TASK=... ES_PACK_OBSERVATION=... ES_PACK_LEARNING=... ES_PACK_DEPLOYMENT=... \
///     ES_PACK_OUT=out.esb cargo test -p es --test cli -- --ignored pack_untrained_bundle
#[test]
#[ignore = "bundle builder for a measurement; run explicitly"]
fn pack_untrained_bundle() {
    let var = |k: &str| {
        PathBuf::from(std::env::var(k).unwrap_or_else(|e| {
            panic!("{k}: {e}; pack_untrained_bundle needs all five ES_PACK_* variables")
        }))
    };
    let Ok(out) = std::env::var("ES_PACK_OUT") else {
        println!("SKIP pack_untrained_bundle: ES_PACK_OUT unset");
        return;
    };
    let bytes = pack_untrained(
        &var("ES_PACK_TASK"),
        &var("ES_PACK_OBSERVATION"),
        &var("ES_PACK_LEARNING"),
        &var("ES_PACK_DEPLOYMENT"),
    );
    std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("{out}: {e}"));
    let opened = es_compile::PolicyBundle::open(&bytes).expect("the bundle opens");
    let h = &opened.manifest.hashes;
    println!(
        "wrote {out}\ntask_hash {}\nobservation_hash {}\nlearning_hash {}\ndeployment_hash {}",
        hex(&h.task.expect("task_hash")),
        hex(&h.observation.expect("observation_hash")),
        hex(&h.learning.expect("learning_hash")),
        hex(&h.deployment.expect("deployment_hash")),
    );
}

/// Packet M7/R5 oracle 3: `es loop collect --frames` renders the sensor the way the **Task
/// IR** says, and a `Pt` sensor really is path-traced.
///
/// Same seed, same expert, same everything but the render declaration, so the two runs walk
/// the identical trajectory and tick `k` of one is tick `k` of the other -- which is what
/// makes "the frames differ" a statement about the renderer and not about the physics. Two
/// `Pt` runs are then compared to each other: the path tracer is a counter-based RNG at a
/// fixed seed (spec 3.4), so an observation frame is a pure function of the pose, and R4's
/// accumulation is deliberately not on this path.
#[test]
#[cfg(feature = "render")]
fn collect_renders_the_sensor_with_the_path_tracer() {
    /// Long enough for the arm to move off its rest pose, short enough to be a test.
    const TICKS: usize = 24;

    let test = "collect_renders_the_sensor_with_the_path_tracer";
    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP {test}: {reason}");
        return;
    }
    if let Err(e) = es_gpu::SlangCompiler::new() {
        println!("SKIP {test}: no slangc ({e})");
        return;
    }
    let device = match es_gpu::Gpu::open(es_gpu::GpuOptions::default()) {
        Ok(gpu) => gpu.capabilities().device_name.clone(),
        Err(e) => {
            println!("SKIP {test}: no Vulkan device ({e})");
            return;
        }
    };

    let dir = scratch_dir("collect-pt");
    let collect = |bundle: &Path, tag: &str| -> PathBuf {
        let frames = dir.join(format!("frames-{tag}"));
        let out = bin()
            .args([
                "loop",
                "collect",
                "--expert",
                "so101-pick-place",
                "--policy",
            ])
            .arg(bundle)
            .arg("--scene")
            .arg(demo_scene_path())
            .args([
                "--episodes",
                "1",
                "--seed",
                "1",
                "--max-steps",
                &TICKS.to_string(),
            ])
            .arg("--frames")
            .arg(&frames)
            .arg("--out")
            .arg(dir.join(format!("ds-{tag}")))
            .output()
            .expect("run es loop collect --frames");
        assert_eq!(
            out.status.code(),
            Some(0),
            "{tag}\nstdout:\n{}\nstderr:\n{}",
            stdout(&out),
            String::from_utf8_lossy(&out.stderr)
        );
        frames
    };

    let rs = collect(
        &write_demo_bundle_from(&dir, "task.toml", "observation.toml", "rs.esb"),
        "rs",
    );
    let pt_bundle = write_demo_bundle_from(&dir, "task-pt.toml", "observation-pt.toml", "pt.esb");
    let pt = collect(&pt_bundle, "pt");
    let pt_again = collect(&pt_bundle, "pt2");

    let frame = |dir: &Path, tick: usize| {
        std::fs::read(dir.join(format!("{tick:06}.bin")))
            .unwrap_or_else(|e| panic!("{}/{tick:06}.bin: {e}", dir.display()))
    };
    // `Rgb8`, at the declared size: the Task IR moved the renderer, never the `ImageSpec`.
    let layout: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(pt.join("000000.json")).expect("sidecar"))
            .expect("the frame sidecar parses");
    assert_eq!(layout["dtype"], "u8", "{layout}");
    assert_eq!(layout["shape"], serde_json::json!([96, 96, 3]), "{layout}");

    let mut differing = 0usize;
    for tick in 0..TICKS {
        let (a, b) = (frame(&rs, tick), frame(&pt, tick));
        assert_eq!(a.len(), 96 * 96 * 3, "tick {tick}: not a 96x96 Rgb8 tile");
        assert_eq!(b.len(), a.len());
        assert!(
            b.iter().any(|x| *x != 0),
            "tick {tick}: the path-traced observation is black -- the scene's emitter did not \
             reach the camera"
        );
        if a != b {
            differing += 1;
        }
        assert!(
            frame(&pt_again, tick) == b,
            "tick {tick}: two path-traced runs of the same documents differ"
        );
    }
    assert_eq!(
        differing,
        TICKS,
        "the path tracer produced the rasterizer's pixels on {} of {TICKS} ticks",
        TICKS - differing
    );
    let mean = |v: &[u8]| v.iter().map(|b| u32::from(*b)).sum::<u32>() / v.len() as u32;
    println!(
        "RAN {test} on {device}: {TICKS} ticks, mean byte Rs {} / Pt {}",
        mean(&frame(&rs, 0)),
        mean(&frame(&pt, 0))
    );
}

/// Packet M7/R5 oracle 4: a path-traced bundle judged by the rasterizer's Evaluation IR is
/// refused by name, which is spec 13.3 doing its job -- the numbers would otherwise carry a
/// correct `evaluation_hash` and describe documents nobody evaluated.
///
/// No backend and no GPU: the refusal happens on the documents, before anything opens.
#[test]
fn eval_refuses_a_pt_policy_on_the_rs_document() {
    let dir = scratch_dir("eval-pt-mismatch");
    let bundle = write_demo_bundle_from(&dir, "task-pt.toml", "observation-pt.toml", "pt.esb");
    let out = bin()
        .args(["eval", "run", "--config"])
        .arg(vl_fixture("evaluation.toml"))
        .arg("--policy")
        .arg(&bundle)
        .arg("--scene")
        .arg(demo_scene_path())
        .arg("--out")
        .arg(dir.join("run"))
        .output()
        .expect("run es eval run");
    let text = format!("{}{}", stdout(&out), String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("evaluation task reference"), "{text}");
    // Named by hash, both sides, so a person can tell which document is which.
    assert!(text.contains("eb6efefa"), "{text}");
    assert!(text.contains("d546b808"), "{text}");
    assert!(
        !dir.join("run").join("report.json").is_file(),
        "a refused evaluation wrote a report"
    );

    // ... and its own document is accepted: the refusal is about the mismatch, not about `Pt`.
    let ok = bin()
        .args(["eval", "run", "--config"])
        .arg(vl_fixture("evaluation-pt.toml"))
        .arg("--policy")
        .arg(&bundle)
        .arg("--scene")
        .arg(demo_scene_path())
        .arg("--out")
        .arg(dir.join("run-pt"))
        .output()
        .expect("run es eval run");
    let text = format!("{}{}", stdout(&ok), String::from_utf8_lossy(&ok.stderr));
    assert!(
        !text.contains("does not judge"),
        "evaluation-pt.toml was refused for its own bundle:\n{text}"
    );
    println!("RAN eval_refuses_a_pt_policy_on_the_rs_document");
}

/// Regenerates `tests/fixtures/visible-learning/{task-pt,observation-pt,evaluation-pt}.toml`
/// from the
/// committed `task.toml` / `observation.toml`, and prints the hashes the design notes record.
/// Run explicitly:
///
///     cargo test -p es --test cli -- --ignored generate_pt_fixtures
#[test]
#[ignore = "fixture generator; run explicitly"]
fn generate_pt_fixtures() {
    // Spec 1.4: goldens and fixtures are CI read-only, and `cargo test -- --include-ignored`
    // runs every ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_pt_fixtures: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let mut task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml parses");
    let committed = hex(&task.task_hash().expect("task hash"));

    let channel = task
        .observation_spec
        .channels
        .get_mut("rgb_overhead")
        .expect("the demo declares one image channel");
    match &mut channel.source {
        es_ir::task::ObsSource::Sensor { render, .. } => {
            *render = es_ir::task::SensorRender {
                path: es_ir::task::SensorPath::Pt {
                    spp: 64,
                    bounces: 3,
                },
                exposure: PT_EXPOSURE,
                tonemap: es_ir::task::Tonemap::Reinhard,
            };
        }
        other => panic!("rgb_overhead is not a sensor channel: {other:?}"),
    }
    let diags = task.validate();
    assert!(diags.is_empty(), "{diags:?}");
    let task_pt = hex(&task.task_hash().expect("task hash"));
    assert_ne!(task_pt, committed, "a Pt sensor must move task_hash");
    write(
        &vl_fixture("task-pt.toml"),
        &format!(
            "{PT_TASK_HEADER}\n{}",
            es_ir::serial::task_to_toml(&task).expect("task-pt toml")
        ),
    );

    // `task_ref` is hash input and `XIR_001` requires it to be the task's own hash, so the
    // path-traced task needs its own Observation IR document -- with a byte-identical graph.
    let mut obs = es_ir::serial::observation_from_toml(&read("observation.toml"))
        .expect("observation.toml parses");
    obs.task_ref = task.task_hash().expect("task hash");
    let diags = obs.validate();
    assert!(diags.is_empty(), "{diags:?}");
    write(
        &vl_fixture("observation-pt.toml"),
        &format!(
            "{PT_OBSERVATION_HEADER}\n{}",
            es_ir::serial::observation_to_toml(&obs).expect("observation-pt toml")
        ),
    );

    let evaluation = demo_evaluation_ir(
        task_pt.clone(),
        hex(&obs.observation_hash().expect("observation hash")),
    );
    let diags = evaluation.validate();
    assert!(diags.is_empty(), "{diags:?}");
    write(
        &vl_fixture("evaluation-pt.toml"),
        &format!(
            "{PT_EVALUATION_HEADER}\n{}",
            es_ir::serial::evaluation_to_toml(&evaluation).expect("evaluation-pt toml")
        ),
    );

    println!(
        "task_hash {committed} (committed, unmoved)\ntask_hash {task_pt} (task-pt)\n\
         observation_hash {} (observation-pt)\nevaluation_hash {} (evaluation-pt)",
        hex(&obs.observation_hash().expect("observation hash")),
        hex(&evaluation.evaluation_hash().expect("evaluation hash"))
    );
}

// --- packet M7/E7: collection and training publish, one address for a cycle ------------------

/// Every regular file under `root`, relative path -> bytes, for a byte-identity assertion over
/// a whole directory. `loop.jsonl` is dropped: its `created` is a wall clock, and spec 13.3
/// says provenance is not identity.
fn tree_of(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(dir: &Path, base: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, base, out);
            } else if let Ok(bytes) = std::fs::read(&path) {
                let name = path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(name, bytes);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out.remove("loop.jsonl");
    out
}

/// Subscribes on `addr` once the producer has bound it and drains every stream until the run
/// closes the connection; the handle gives the messages back.
fn drain_from(addr: std::net::SocketAddr) -> std::thread::JoinHandle<Vec<Message>> {
    std::thread::spawn(move || {
        let mut client = connect_when_bound(addr);
        client
            .subscribe(vec![
                StreamId(1),
                StreamId(2),
                StreamId(3),
                StreamId(4),
                StreamId(5),
            ])
            .expect("subscribe");
        let mut received = Vec::new();
        while let Ok(msg) = client.recv() {
            received.push(msg);
        }
        received
    })
}

fn frames_of(messages: &[Message]) -> Vec<&es_telemetry::Frame> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::Frame(f) => Some(f),
            _ => None,
        })
        .collect()
}

/// Every stream-1 event of `kind`, as its field map.
fn events_of<'a>(
    frames: &'a [&es_telemetry::Frame],
    kind: &str,
) -> Vec<&'a BTreeMap<String, String>> {
    frames
        .iter()
        .filter_map(|f| match (&f.stream, &f.payload) {
            (StreamId(1), Payload::Event { kind: k, fields }) if k == kind => Some(fields),
            _ => None,
        })
        .collect()
}

/// Every `Scalars` payload on one stream, in arrival order.
fn scalars_on<'a>(frames: &'a [&es_telemetry::Frame], stream: u32) -> Vec<&'a Vec<f64>> {
    frames
        .iter()
        .filter_map(|f| match (&f.stream, &f.payload) {
            (s, Payload::Scalars(v)) if s.0 == stream => Some(v),
            _ => None,
        })
        .collect()
}

/// Oracle 1 (packet M7/E7). `es train --telemetry --progress-every 10` on the bake fixture:
/// four stream-5 frames with strictly increasing steps and one `checkpoint` event at the mark
/// -- and `training.lock`, the checkpoint bundle and `metrics/loss.json` byte-identical to the
/// same run without the flag.
#[test]
fn train_telemetry_streams_the_curve() {
    const TEST: &str = "train_telemetry_streams_the_curve";
    let Ok(python) = std::env::var("ES_PYTHON") else {
        println!("SKIP {TEST}: ES_PYTHON is not set, so nothing can train");
        return;
    };
    if skip_without_bake_model(TEST) {
        return;
    }
    let dir = scratch_dir("train-telemetry");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 4, 16);
    let recipe = dir.join("training.toml");
    write(
        &recipe,
        &train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4").replace(
            "es-no-such-interpreter",
            &train_toml_path(Path::new(&python)),
        ),
    );
    let train = |out: &Path, extra: &[&str]| {
        let run = bin()
            .current_dir(train_root())
            .args(["train", "--recipe", &train_toml_path(&recipe), "--out"])
            .arg(out)
            .args(extra)
            .output()
            .expect("run es train");
        assert_eq!(
            run.status.code(),
            Some(0),
            "stdout:\n{}\nstderr:\n{}",
            stdout(&run),
            stderr_of(&run)
        );
        run
    };
    let plain = dir.join("plain");
    train(&plain, &[]);

    let port = free_loopback_port();
    let addr = format!("127.0.0.1:{port}");
    let reader = drain_from(addr.parse().expect("socket addr"));
    let live = dir.join("live");
    let run = train(
        &live,
        &[
            "--telemetry",
            &addr,
            "--progress-every",
            "10",
            "--sample-every",
            "20",
        ],
    );
    let received = reader.join().expect("reader thread");
    let text = stdout(&run);
    assert!(text.contains(&format!("telemetry: {addr}")), "{text}");

    // What the run wrote, unmoved by having been watched -- `config.json` included, because
    // it carries the plan and the plan must not gain the two trainer flags.
    for name in [
        "training.lock",
        "metrics/loss.json",
        "checkpoints/40.esb",
        "training/config.json",
    ] {
        assert_eq!(
            read_bytes(&plain.join(name)),
            read_bytes(&live.join(name)),
            "{name} differs with --telemetry"
        );
    }

    let frames = frames_of(&received);
    let curve = scalars_on(&frames, 5);
    let steps: Vec<f64> = curve.iter().map(|v| v[0]).collect();
    assert_eq!(steps, vec![10.0, 20.0, 30.0, 40.0], "{curve:?}");
    for v in &curve {
        assert_eq!(v.len(), 4, "[step, loss, lr, samples_per_s]: {v:?}");
        assert!(v[1].is_finite() && v[1] >= 0.0, "loss {v:?}");
        assert!((v[2] - 1e-4).abs() < 1e-9, "lr {v:?}");
        assert!(v[3] > 0.0, "samples_per_s {v:?}");
    }
    // What the network is fitting, after augmentation, as `Rgb8` -- two of them at 40 steps
    // every 20, and the tensor's own shape, not the renderer's.
    let samples: Vec<(u32, u32)> = frames
        .iter()
        .filter_map(|f| match (&f.stream, &f.payload) {
            (
                StreamId(4),
                Payload::Image {
                    w,
                    h,
                    format,
                    bytes,
                },
            ) => {
                assert_eq!(format, "rgb8", "the sample is Rgb8");
                assert_eq!(bytes.len() as u32, w * h * 3, "{w}x{h}");
                Some((*w, *h))
            }
            _ => None,
        })
        .collect();
    assert_eq!(samples.len(), 2, "40 steps every 20: {samples:?}");
    // The file the trainer wrote is beside the curve and says what it holds.
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(live.join("metrics/sample-40.json")).expect("sample-40.json"),
    )
    .expect("the sidecar is JSON");
    assert_eq!(meta["shape"][2].as_u64(), Some(3), "{meta}");
    assert_eq!(meta["step"].as_u64(), Some(40), "{meta}");
    assert!(meta["mapping"].is_string(), "{meta}");
    // ...and `--sample-every` is not part of what the run computed: the plain run wrote none.
    assert!(!plain.join("metrics/sample-40.bin").exists());

    let marks = events_of(&frames, "checkpoint");
    assert_eq!(marks.len(), 1, "one mark at 40: {marks:?}");
    assert_eq!(marks[0]["step"], "40", "{marks:?}");
    assert_eq!(marks[0]["stage"], "train", "{marks:?}");
    assert_eq!(
        marks[0]["policy_hash"].len(),
        64,
        "the bundle's own policy_hash: {marks:?}"
    );
    println!("RAN {TEST}: {} curve point(s), 1 checkpoint", curve.len());
}

/// Oracle 2 (packet M7/E7). `es loop collect --frames --telemetry`: an `episode.begin` and an
/// `episode.end` per episode, one stream-2 frame per control tick, at least one image -- and
/// the dataset directory byte-identical to a run without the flag.
#[test]
fn collect_telemetry_publishes_every_episode() {
    const TEST: &str = "collect_telemetry_publishes_every_episode";
    if cfg!(not(feature = "render")) {
        println!("SKIP {TEST}: built without the `render` feature");
        return;
    }
    if skip_without_bake_model(TEST) {
        return;
    }
    let dir = scratch_dir("collect-telemetry");
    let policy = write_demo_bundle(&dir);
    let collect = |out: &Path, extra: &[&str]| {
        bin()
            .args([
                "loop",
                "collect",
                "--expert",
                "so101-pick-place",
                "--policy",
            ])
            .arg(&policy)
            .arg("--scene")
            .arg(demo_scene_path())
            .args([
                "--episodes",
                "2",
                "--seed",
                "4",
                "--max-steps",
                "40",
                "--out",
            ])
            .arg(out)
            .arg("--frames")
            .arg(out.join("frames"))
            .args(extra)
            .output()
            .expect("run es loop collect")
    };
    let plain = dir.join("plain");
    let first = collect(&plain, &[]);
    if first.status.code() == Some(3) {
        println!("SKIP {TEST}: {}", stdout(&first).trim());
        return;
    }
    assert_eq!(
        first.status.code(),
        Some(0),
        "stdout:\n{}\nstderr:\n{}",
        stdout(&first),
        String::from_utf8_lossy(&first.stderr)
    );

    let port = free_loopback_port();
    let addr = format!("127.0.0.1:{port}");
    let reader = drain_from(addr.parse().expect("socket addr"));
    let live = dir.join("live");
    let run = collect(
        &live,
        &["--telemetry", &addr, "--telemetry-image-every", "5"],
    );
    let received = reader.join().expect("reader thread");
    assert_eq!(run.status.code(), Some(0), "{}", stdout(&run));

    // Every file the dataset is made of, byte for byte -- `frames/` is under `--out` too, so
    // the rendered tiles are compared with it.
    let (before, after) = (tree_of(&plain), tree_of(&live));
    assert_eq!(
        before.keys().collect::<Vec<_>>(),
        after.keys().collect::<Vec<_>>(),
        "the two runs wrote different files"
    );
    for (name, bytes) in &before {
        assert_eq!(bytes, &after[name], "{name} differs with --telemetry");
    }

    let frames = frames_of(&received);
    let begun = events_of(&frames, "episode.begin");
    let ended = events_of(&frames, "episode.end");
    assert_eq!(begun.len(), 2, "{begun:?}");
    assert_eq!(ended.len(), 2, "{ended:?}");
    for (i, fields) in begun.iter().enumerate() {
        assert_eq!(fields["episode"], i.to_string(), "{fields:?}");
        assert_eq!(fields["seed"], "4", "{fields:?}");
        assert_eq!(fields["stage"], "collect", "{fields:?}");
    }
    for fields in &ended {
        assert!(fields.contains_key("outcome"), "{fields:?}");
        assert!(
            fields["steps"].parse::<u64>().expect("steps") > 0,
            "{fields:?}"
        );
    }
    // One sample per control tick, with the four numbers `es eval run` sends.
    let ticks = scalars_on(&frames, 2);
    let steps: u64 = ended
        .iter()
        .map(|f| f["steps"].parse::<u64>().expect("steps"))
        .sum();
    assert_eq!(ticks.len() as u64, steps, "one sample per control tick");
    for v in &ticks {
        assert_eq!(v.len(), 4, "[frame, tick, source, events]: {v:?}");
        assert!((0.0..=3.0).contains(&v[2]), "source code {v:?}");
    }
    let images = frames
        .iter()
        .filter(|f| {
            matches!(
                (&f.stream, &f.payload),
                (StreamId(4), Payload::Image { .. })
            )
        })
        .count();
    assert!(images > 0, "no observation image on stream 4");
    println!(
        "RAN {TEST}: {} tick(s), {images} image(s) over {} episode(s)",
        ticks.len(),
        begun.len()
    );
}

/// Oracle 3 (packet M7/E7). The address is the *cycle's*, not a stage's: `--dry-run` prints
/// the same plan with and without it, and no stage line carries `--telemetry`.
#[test]
fn cycle_telemetry_is_one_address() {
    const TEST: &str = "cycle_telemetry_is_one_address";
    let dir = scratch_dir("cycle-telemetry-plan");
    let plain = run_cycle(CYCLE_RECIPE, &dir, &["--dry-run"]);
    assert_eq!(plain.status.code(), Some(0), "{}", stderr_of(&plain));
    let watched = run_cycle(
        CYCLE_RECIPE,
        &dir,
        &[
            "--dry-run",
            "--telemetry",
            "127.0.0.1:7777",
            "--progress-every",
            "10",
        ],
    );
    assert_eq!(watched.status.code(), Some(0), "{}", stderr_of(&watched));
    let text = stdout(&plain);
    assert_eq!(
        text,
        stdout(&watched),
        "--telemetry moved the plan; the address is the cycle's, not a stage's"
    );
    assert!(!text.contains("--telemetry"), "{text}");
    println!("RAN {TEST}: the plan is unmoved by --telemetry");
}

/// The cycle document `cycle_telemetry_delivers_every_stage` runs: the demo's own Evaluation
/// IR cut to one suite and two pinned seeds, a 40-step IR-route training and a 2-episode
/// expert collect -- the shape `cycle_runs_the_expert_through_the_harness_first` established,
/// written here so this oracle owns its own fixture.
fn cycle_live_document(dir: &Path, python: &str) -> PathBuf {
    let bundle = write_demo_bundle(dir);
    let p = |path: &Path| train_toml_path(path);
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml");
    let obs =
        es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml");
    let mut ir = demo_evaluation_ir(
        hex(&task.task_hash().expect("task hash")),
        hex(&obs.observation_hash().expect("observation hash")),
    );
    ir.episodes = es_ir::evaluation::EpisodeBatch {
        n_episodes: 2,
        seeds: es_ir::evaluation::SeedPlan::Explicit(vec![SEEDS[0], SEEDS[1]]),
    };
    ir.suites.truncate(1);
    let eval_config = dir.join("evaluation.toml");
    write(
        &eval_config,
        &es_ir::serial::evaluation_to_toml(&ir).expect("the Evaluation IR serialises"),
    );
    let recipe = dir.join("training.toml");
    write(
        &recipe,
        &format!(
            "kind = \"training\"\n\
             [dataset]\nroot = \"unused\"\n\
             [policy]\nbundle = \"{}\"\n\
             [run]\nsteps = 40\nbatch = 2\nlr = 1e-4\nseed = 0\ncheckpoint_at = [40]\n\
             device = \"cpu\"\ninterpreter = \"{}\"\n",
            p(&bundle),
            p(Path::new(python)),
        ),
    );
    let document = dir.join("cycle.toml");
    write(
        &document,
        &format!(
            "kind = \"cycle\"\nscene = \"{}\"\n\
             [collect]\npolicy = \"{}\"\nexpert = \"so101-pick-place\"\nepisodes = 2\n\
             seed = 1\nframes = true\n\
             [train]\nrecipe = \"{}\"\n\
             [eval]\nconfig = \"{}\"\njobs = 1\nframes = true\n",
            p(&demo_scene_path()),
            p(&bundle),
            p(&recipe),
            p(&eval_config),
        ),
    );
    document
}

/// Oracle 3's live half (packet M7/E7): the four stages of one cycle, in order, on one socket.
///
/// `#[ignore]`d for the same reason `cycle_runs_the_expert_through_the_harness_first` is -- it
/// needs `ES_PYTHON` and a render build -- and it is that oracle with a subscriber attached.
#[test]
#[ignore = "needs ES_PYTHON with torch and mujoco, and a render build"]
fn cycle_telemetry_delivers_every_stage() {
    const TEST: &str = "cycle_telemetry_delivers_every_stage";
    let Ok(python) = std::env::var("ES_PYTHON") else {
        println!("SKIP {TEST}: ES_PYTHON is not set");
        return;
    };
    if cfg!(not(feature = "render")) {
        println!("SKIP {TEST}: built without the `render` feature");
        return;
    }
    if skip_without_bake_model(TEST) {
        return;
    }
    let dir = scratch_dir("cycle-telemetry-live");
    let document = cycle_live_document(&dir, &python);
    let port = free_loopback_port();
    let addr = format!("127.0.0.1:{port}");
    let reader = drain_from(addr.parse().expect("socket addr"));
    let out = bin()
        .current_dir(train_root())
        .args(["loop", "cycle", "--recipe"])
        .arg(&document)
        .arg("--out")
        .arg(dir.join("run"))
        .args(["--telemetry", &addr, "--progress-every", "10"])
        .output()
        .expect("run es loop cycle --telemetry");
    let received = reader.join().expect("reader thread");
    let text = format!("{}{}", stdout(&out), stderr_of(&out));
    if out.status.code() == Some(3) {
        println!("SKIP {TEST}: {}", text.trim());
        return;
    }
    let frames = frames_of(&received);
    let stages: Vec<String> = events_of(&frames, "stage.begin")
        .iter()
        .map(|f| f["name"].clone())
        .collect();
    assert_eq!(
        stages,
        ["collect", "expert-gate", "train", "eval"],
        "one socket, four stages, in order\n{text}"
    );
    assert!(
        !events_of(&frames, "episode.begin").is_empty(),
        "collect published no episode\n{text}"
    );
    assert!(
        !scalars_on(&frames, 5).is_empty(),
        "training published no curve\n{text}"
    );
    println!("RAN {TEST}: {stages:?} on one address");
}

// --- packet M8/S1: `[init] policy` ---------------------------------------------------------

/// A bundle carrying `learning-pretrained.toml`'s graph with `edit` applied, and a real
/// safetensors checkpoint generated from *that graph's own* lowering: every exact key the
/// module declares, filled with a value derived from the key so no two tensors are equal.
///
/// The prefix claims (`nodes.0.*`, `nodes.3.*`) are deliberately absent. They belong to
/// torchvision and `torch.nn`, this test has no torch, and a checkpoint that does not carry
/// them is exactly the bundle that makes `initialised` non-empty — which is the half of
/// `init.lock` a full copy never exercises.
fn write_init_source(
    dir: &Path,
    name: &str,
    rename: impl Fn(&str) -> String,
    edit: impl FnOnce(&mut es_ir::learning::LearningGraph),
) -> PathBuf {
    let read = |n: &str| std::fs::read_to_string(vl_fixture(n)).expect(n);
    let mut learning = es_ir::serial::learning_from_toml(&read("learning-pretrained.toml"))
        .expect("learning-pretrained.toml");
    edit(&mut learning);
    let module = es_policy::lower_to_torch(&learning).expect("the edited graph lowers");
    let tensors: es_policy::weights::Checkpoint = module
        .weight_shapes
        .iter()
        .map(|(key, shape)| {
            let n: u64 = shape.iter().product();
            let seed = f32::from(u8::try_from(key.len()).unwrap_or(7));
            let values = (0..n).map(|i| seed + i as f32 * 0.5).collect();
            (rename(key), (shape.clone(), values))
        })
        .collect();
    let weights = es_policy::weights::write_safetensors(&tensors);
    learning.policy.weights = es_ir::learning::WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = es_compile::PolicyBundle::build(
        &es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the edited documents pack into a bundle");
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write the init source bundle");
    path
}

/// The recipe of [`train_fixture_recipe`] with `[init] policy` in front of `[run]`.
fn with_init(recipe: &str, source: &Path) -> String {
    recipe.replace(
        "[run]\n",
        &format!("[init]\npolicy = \"{}\"\n[run]\n", train_toml_path(source)),
    )
}

fn init_lock(out: &Path) -> serde_json::Value {
    let path = out.join("training").join("init.lock");
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .expect("init.lock is JSON")
}

fn names(lock: &serde_json::Value, field: &str) -> Vec<String> {
    lock[field]
        .as_array()
        .unwrap_or_else(|| panic!("init.lock has no {field}"))
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_owned())
        .collect()
}

/// Oracle 3. A bundle that shares part of the module warm-starts that part and says so; one
/// that shares nothing is refused; `[init]` on the lerobot route is refused.
///
/// No Python: the comparison is a property of two documents, and `es train` does it among the
/// refusals that need the documents — before the interpreter is probed, before the bake, and
/// before a single GPU-second — because `init.lock` enters `identity_hash` and the identity
/// of a run exists before the run does.
#[test]
fn train_init_partial_and_refused() {
    let dir = scratch_dir("train-init-partial");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 1, 12);
    let base = train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4");

    let run = |name: &str, body: &str| -> (bool, String, PathBuf) {
        let recipe = dir.join(format!("{name}.toml"));
        write(&recipe, body);
        let out = dir.join(name);
        let done = run_train(&train_toml_path(&recipe), &out, &[]);
        (
            done.status.success(),
            format!("{}{}", stdout(&done), stderr_of(&done)),
            out,
        )
    };

    // The intersection. The source is `learning-pretrained.toml`'s graph -- a different
    // `learning_hash` -- with one `StateEncoder`'s hidden width moved from 256 to 128, so its
    // three tensors disagree about a shape while everything else still fits.
    let partial = write_init_source(&dir, "partial.esb", str::to_owned, |graph| {
        let node = graph
            .nodes
            .nodes
            .get_mut(&es_ir::NodeId(1))
            .expect("node 1 is the joint StateEncoder");
        let es_ir::learning::LearningNode::StateEncoder { kind, .. } = node else {
            panic!("node 1 is a StateEncoder");
        };
        *kind = es_ir::learning::StateEncoderKind::Mlp {
            hidden: vec![128],
            activation: es_ir::learning::Activation::Relu,
            activate_output: false,
        };
    });
    let (ok, said, out) = run("partial", &with_init(&base, &partial));
    // It stops at the interpreter, which is the refusal every other `es train` test stops at.
    assert!(!ok, "the fixture interpreter exists: {said}");
    assert!(said.contains("es-no-such-interpreter"), "{said}");
    let lock = init_lock(&out);
    assert_eq!(lock["source"], train_toml_path(&partial));
    assert!(lock["policy_hash"].is_string(), "{lock}");
    assert!(lock["learning_hash"].is_string(), "{lock}");

    let copied = names(&lock, "copied");
    let initialised = names(&lock, "initialised");
    let mismatch: Vec<String> = lock["shape_mismatch"]
        .as_array()
        .expect("shape_mismatch")
        .iter()
        .map(|m| m["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert!(copied.contains(&"nodes.4.weight".to_owned()), "{lock}");
    assert!(copied.contains(&"nodes.2.weight".to_owned()), "{lock}");
    // The opaque sub-modules: the source carries no tensor under either claim, so the module
    // draws them, and the lock names the claim rather than pretending to enumerate it.
    assert_eq!(initialised, ["nodes.0.*", "nodes.3.*"], "{lock}");
    assert_eq!(
        mismatch,
        ["nodes.1.0.bias", "nodes.1.0.weight", "nodes.1.2.weight"],
        "a 128-wide hidden layer was copied into a 256-wide one\n{lock}"
    );
    assert!(
        !copied.iter().any(|k| mismatch.contains(k)),
        "a tensor is both copied and a shape mismatch\n{lock}"
    );
    // The lock is a slot of the identity: `training.lock` carries its digest beside the
    // twelve, and it is the digest of the file on disk.
    let text = std::fs::read_to_string(out.join("training").join("init.lock")).expect("init.lock");
    assert_eq!(
        train_lock(&out)["files"]["init.lock"]
            .as_str()
            .expect("slot"),
        hex(blake3::hash(text.as_bytes()).as_bytes())
    );
    // ... and it moves the identity. The same recipe without `[init]` is a different run.
    let (_, _, plain) = run("plain", &base);
    assert_ne!(
        train_lock(&out)["identity_hash"],
        train_lock(&plain)["identity_hash"],
        "[init] did not move identity_hash"
    );
    assert!(!plain.join("training").join("init.lock").exists());
    assert!(train_lock(&plain)["files"]["init.lock"].is_null());

    // Nothing in common: a warm start that shares nothing is a mistake, not a warm start.
    // Every node id is shifted by ten, which is what a genuinely differently-laid-out graph
    // produces -- no key is one the module declares and none falls under a prefix claim.
    let alien = write_init_source(
        &dir,
        "alien.esb",
        |key| key.replacen("nodes.", "nodes.1", 1),
        |_| {},
    );
    let (ok, said, _) = run("alien", &with_init(&base, &alien));
    assert!(!ok, "a bundle sharing no tensor was accepted: {said}");
    assert!(said.contains("shares no tensor"), "{said}");
    assert!(said.contains("warm start"), "{said}");

    // The external route has its own `--policy.path`, and mixing the two fabricates a
    // provenance: `init.lock` would describe tensors `lerobot-train` never loaded.
    let external = with_init(
        &std::fs::read_to_string(train_root().join(TRAIN_RECIPES[1].0))
            .expect("the lerobot recipe"),
        &partial,
    );
    let (ok, said, _) = run("external", &external);
    assert!(!ok, "[init] was accepted on the lerobot route: {said}");
    assert!(said.contains("[init]"), "{said}");
    assert!(said.contains("IR route"), "{said}");
}

/// Oracle 2. Forty steps produce a bundle; a second recipe starting from that bundle with
/// `steps = 0` produces a checkpoint whose every tensor is bitwise the first one's.
///
/// `#[ignore]`d for the same reason `train_ir_path_packs_a_bundle_torch_opens` is: it is the
/// arm of this packet that needs torch, and without `ES_PYTHON` it says so rather than
/// pretending (spec 1.4).
#[test]
#[ignore = "needs ES_PYTHON with torch"]
fn train_init_from_bundle_zero_steps() {
    const NAME: &str = "train_init_from_bundle_zero_steps";
    let Ok(python) = std::env::var("ES_PYTHON") else {
        println!("SKIP {NAME}: ES_PYTHON is not set");
        return;
    };
    if skip_without_bake_model(NAME) {
        return;
    }
    let dir = scratch_dir("train-init-zero");
    let bundle = write_demo_bundle(&dir);
    let (root, tiles) = (dir.join("ds"), dir.join("tiles"));
    write_bake_fixture(&root, &tiles, 4, 16);
    let real = |body: &str| -> String {
        body.replace(
            "es-no-such-interpreter",
            &train_toml_path(Path::new(&python)),
        )
    };
    let go = |name: &str, body: &str| -> PathBuf {
        let recipe = dir.join(format!("{name}.toml"));
        write(&recipe, body);
        let out = dir.join(name);
        let done = bin()
            .current_dir(train_root())
            .args(["train", "--recipe", &train_toml_path(&recipe), "--out"])
            .arg(&out)
            .output()
            .expect("run es train");
        assert_eq!(
            done.status.code(),
            Some(0),
            "{name}\nstdout:\n{}\nstderr:\n{}",
            stdout(&done),
            stderr_of(&done)
        );
        out
    };

    // A. Forty real steps.
    let base = real(&train_fixture_recipe(&bundle, &root, &tiles, 0, "1e-4"));
    let trained = go("trained", &base).join("checkpoints").join("40.esb");

    // B. Zero steps from A: "checkpoint immediately".
    let zero = base
        .replace("steps = 40", "steps = 0")
        .replace("checkpoint_at = [40]\n", "");
    let out = go("zero", &with_init(&zero, &trained));

    let opened = |p: &Path| {
        let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        es_compile::PolicyBundle::open(&bytes).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    };
    let a = opened(&trained);
    let b = opened(&out.join("checkpoints").join("0.esb"));
    assert_eq!(
        es_policy::weights::parse_header(&a.weights).expect("A's header"),
        es_policy::weights::parse_header(&b.weights).expect("B's header"),
        "the re-packed checkpoint has another shape"
    );
    assert_eq!(
        blake3::hash(&a.weights),
        blake3::hash(&b.weights),
        "zero steps from a checkpoint did not reproduce it bitwise"
    );

    // Every tensor of A reached the module, and nothing was drawn.
    let lock = init_lock(&out);
    let copied = names(&lock, "copied");
    let header = es_policy::weights::parse_header(&a.weights).expect("A's header");
    assert_eq!(
        copied,
        header.keys().cloned().collect::<Vec<_>>(),
        "init.lock does not list every tensor of the bundle it started from"
    );
    assert!(names(&lock, "initialised").is_empty(), "{lock}");
    assert!(
        lock["shape_mismatch"].as_array().expect("array").is_empty(),
        "{lock}"
    );
    println!(
        "RAN {NAME}: {} tensor(s) copied, weights {} reproduced bitwise at policy_hash {}",
        copied.len(),
        hex(blake3::hash(&a.weights).as_bytes()),
        train_lock(&out)["checkpoints"][0]["policy_hash"]
    );
}

// --- packet M8/S4b: the `[rl]` route and `python/es/train_ppo.py` --------------------------

/// Workspace-root path of one of the RL fixture documents.
fn rl_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rl")
        .join(name)
}

/// The demo scene's six `ctrlrange`s, read out of the parsed MJCF rather than transcribed --
/// the same rule `deployment.toml`'s `safety.position` follows.
fn demo_ctrlrange() -> Vec<(f64, f64)> {
    let xml = std::fs::read_to_string(demo_scene_path()).expect("the demo scene");
    let scene = es_assets::parse_mjcf(&xml)
        .expect("the demo scene parses")
        .scene;
    scene
        .actuators
        .iter()
        .map(|a| {
            a.ctrl_range
                .expect("every sts3215 actuator declares a ctrlrange")
        })
        .collect()
}

/// The Learning IR an RL run optimizes: the state-only Observation IR's two ports, an MLP
/// each, and a horizon-1 regression head whose output is unnormalized into actuator units.
///
/// It is `learning.toml`'s graph with the vision branch dropped and three things changed, and
/// every one of them is forced by `rl-continuation.md` rather than chosen here:
///
///  * `horizon = 1` / `execute_chunk = 1` -- PPO acts on every control tick (section 3), so
///    the deployment, the chunker and the head all carry the same 1.
///  * a `Normalizer { Inverse, MeanStd }` at the end, with `mean` the centre of each
///    actuator's `ctrlrange` and `std` its half-range (section 5). That makes the module's
///    output **actuator units**, which is what `Rollout::act` takes and therefore what the
///    trainer's Gaussian is defined around (section 2). Training does not move these numbers:
///    they are the scene's, not the data's.
///  * `hidden = [64, 64]` with the S2a defaults (`Relu`, no output activation). A PPO actor
///    is small on purpose; the 256-wide demo encoder is sized for imitating from pixels.
fn rl_learning() -> LearningGraph {
    use es_ir::learning::{NormalizeDir, StatsSource};

    let feature = |dim: u64| PortType {
        elem: ElemType::F32,
        shape: Shape::new([dim]),
        unit: Unit::Dimensionless,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    };
    let chunk = |unit: Unit| PortType {
        elem: ElemType::F32,
        shape: Shape::new([1, 6]),
        unit,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    };
    let normalized = |lo: f64, hi: f64, dim: u64| PortType {
        elem: ElemType::F32,
        shape: Shape::new([dim]),
        unit: Unit::Normalized { lo, hi },
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    };
    // The two ports `tests/fixtures/rl/observation-state.toml` produces, at the types that
    // document declares -- the demo's own, because it is the demo's document with the camera
    // branch removed.
    let joint = Port::new("joint_state", normalized(-1.0, 1.0, 6));
    let cube = Port::new("sim_cube_pose", normalized(-0.3, 0.3, 7));

    let encoder = |input: Port| LearningNode::StateEncoder {
        inputs: vec![input],
        kind: StateEncoderKind::Mlp {
            hidden: vec![64, 64],
            activation: Activation::Relu,
            activate_output: false,
        },
        out_dim: 64,
    };
    let mut nodes: Graph<LearningNode> = Graph::new(1);
    nodes.insert(NodeId(0), encoder(joint.clone()));
    nodes.insert(NodeId(1), encoder(cube.clone()));
    nodes.insert(
        NodeId(2),
        LearningNode::Fusion {
            inputs: vec![
                Port::new("state", feature(64)),
                Port::new("cube", feature(64)),
            ],
            kind: FusionKind::Concat,
            out_dim: 128,
            token_count: 0,
        },
    );
    nodes.insert(
        NodeId(3),
        LearningNode::PolicyHead {
            inputs: vec![Port::new("feat", feature(128))],
            kind: HeadKind::Regression,
            action_dim: 6,
            horizon: 1,
            // The head's output is squashed into [-1, 1] before the unnormalizer, so the
            // Gaussian's mean is always inside the actuator's own range and the plane is
            // clamping the *sample*, not a mean that left the envelope (packet M8/S2a).
            squash: Squash::Tanh,
        },
    );
    nodes.insert(
        NodeId(4),
        LearningNode::ActionChunker {
            inputs: vec![Port::new(
                "chunk",
                chunk(Unit::Normalized { lo: -1.0, hi: 1.0 }),
            )],
            horizon: 1,
            execute_chunk: 1,
            replan_hz: 50.0,
            mode: ActionExecutionMode::RecedingHorizon,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 2,
        },
    );
    let range = demo_ctrlrange();
    nodes.insert(
        NodeId(5),
        LearningNode::Normalizer {
            inputs: vec![Port::new(
                "actions",
                chunk(Unit::Normalized { lo: -1.0, hi: 1.0 }),
            )],
            direction: NormalizeDir::Inverse,
            stats: StatsSource::MeanStd {
                mean: range.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect(),
                std: range.iter().map(|(lo, hi)| (hi - lo) / 2.0).collect(),
            },
            out_unit: Unit::Angle,
        },
    );
    nodes.connect(NodeId(0), "out", NodeId(2), "state");
    nodes.connect(NodeId(1), "out", NodeId(2), "cube");
    nodes.connect(NodeId(2), "out", NodeId(3), "feat");
    nodes.connect(NodeId(3), "chunk", NodeId(4), "chunk");
    nodes.connect(NodeId(4), "actions", NodeId(5), "actions");
    nodes.inputs.push(PortRef::new(NodeId(0), "joint_state"));
    nodes.inputs.push(PortRef::new(NodeId(1), "sim_cube_pose"));
    nodes.outputs.push(PortRef::new(NodeId(5), "out"));

    let mut contract_inputs = BTreeMap::new();
    contract_inputs.insert("joint_state".to_owned(), joint.clone());
    contract_inputs.insert("sim_cube_pose".to_owned(), cube.clone());
    LearningGraph {
        schema_version: 1,
        inputs: vec![joint, cube],
        nodes,
        outputs: vec![Port::new("actions", chunk(Unit::Angle))],
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0; 32],
            },
            contract: PolicyContract {
                inputs: contract_inputs,
                action_dim: 6,
                horizon: 1,
                execute_chunk: 1,
                observation_window: 1,
                replanning_hz: 50.0,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    // Horizon 1 and synchronous: the rollout declares no latency, and the
                    // evaluation of the same policy is what runs under one (section 3).
                    expected_latency_ms: 0.0,
                    deadline_ms: 20.0,
                    dtype: ElemType::F32,
                },
            },
        },
    }
}

/// Regenerates `tests/fixtures/rl/learning-state.toml`. Run once, explicitly; it is then
/// read-only (spec 1.4), like every other generated fixture in this repository.
#[test]
#[ignore = "fixture generator; run explicitly"]
fn generate_rl_learning_document() {
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_rl_learning_document: set ES_GENERATE_GOLDENS=1");
        return;
    }
    let header = "\
# Learning IR (spec 8) for the demo task under PPO -- packet M8/S4b.
#
# Generated by `ES_GENERATE_GOLDENS=1 cargo test -p es --test cli -- --ignored
# generate_rl_learning_document` from tests/fixtures/mjcf/so101_pick_place.xml, so the
# unnormalizer's mean and std are the scene's own `ctrlrange`s and are never typed in.
#
#   StateEncoder{Mlp [64, 64]}  -.
#                                Fusion{Concat} -> PolicyHead{Regression, tanh, 1 x 6}
#   StateEncoder{Mlp [64, 64]}  -'                  -> ActionChunker -> Normalizer{Inverse}
#
# It reads tests/fixtures/rl/observation-state.toml's two ports -- the demo's Observation IR
# with the camera branch removed -- and its output is in **actuator units**, which is what
# `es_native.Rollout::act` takes and therefore what `train_ppo.py`'s Gaussian is defined
# around (docs/design/rl-continuation.md section 2).
#
# The value network and the Gaussian's `log_std` are NOT here and never will be: PPO is a
# trainer, not an IR (rule 1). They live in `training/value.safetensors` and move
# `training_hash`, never `learning_hash`.
";
    let text = es_ir::serial::learning_to_toml(&rl_learning()).expect("the RL graph serialises");
    write(
        &rl_fixture("learning-state.toml"),
        &format!("{header}{text}"),
    );
}

/// The bundle an `[rl]` recipe names: the demo Task IR, the state-only Observation IR, the
/// horizon-1 Learning IR and the RL Deployment IR, packed with placeholder weights.
fn write_rl_bundle(dir: &Path, name: &str) -> PathBuf {
    let bytes = pack_untrained(
        &vl_fixture("task.toml"),
        &rl_fixture("observation-state.toml"),
        &rl_fixture("learning-state.toml"),
        &rl_fixture("deployment-rl.toml"),
    );
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write the rl bundle");
    path
}

/// `name -> the tensor's raw bytes`, for comparing two safetensors files whose headers were
/// written by two different writers.
fn tensor_bytes(blob: &[u8]) -> BTreeMap<String, Vec<u8>> {
    let header = es_policy::weights::parse_header(blob).expect("safetensors");
    let base = 8 + u64::from_le_bytes(blob[..8].try_into().expect("the length prefix")) as usize;
    header
        .iter()
        .map(|(name, entry)| {
            let (a, b) = (entry.offsets.0 as usize, entry.offsets.1 as usize);
            (name.clone(), blob[base + a..base + b].to_vec())
        })
        .collect()
}

/// The committed `[rl]` recipe with `[policy] bundle` pointed at a real one and the budget
/// cut to whatever the caller can afford to run twice.
fn rl_recipe(bundle: &Path, iterations: u32, envs: u32, horizon: u32, marks: &str) -> String {
    let text = std::fs::read_to_string(rl_fixture("training-rl-demo.toml")).expect("the recipe");
    let body = text.split_once("kind = ").expect("the recipe has a body").1;
    format!("kind = {body}")
        .replace(
            "bundle = \"runs/rl-001/untrained.esb\"",
            &format!("bundle = \"{}\"", train_toml_path(bundle)),
        )
        .replace("envs        = 8", &format!("envs        = {envs}"))
        .replace("horizon     = 64", &format!("horizon     = {horizon}"))
        .replace(
            "steps         = 200",
            &format!("steps         = {iterations}"),
        )
        .replace(
            "checkpoint_at = [50]",
            &format!("checkpoint_at = [{marks}]"),
        )
        .replace(
            "interpreter   = \"python\"",
            "interpreter   = \"es-no-such-interpreter\"",
        )
}

/// Oracle 1. The `[rl]` plan is a property of the recipe -- same bytes on any machine, in any
/// output directory -- and the two fields the route does not have are refused **by name**.
///
/// No bundle and no Python: `--dry-run` on this route opens neither, which is what lets the
/// golden be judged on a machine that has only the repository.
#[test]
fn train_rl_dry_run_plan() {
    const RECIPE: &str = "tests/fixtures/rl/training-rl-demo.toml";
    let dir = scratch_dir("train-rl-dry");
    let out = run_train(RECIPE, &dir, &["--dry-run"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
    let golden = train_golden("plan-rl.txt");
    let want =
        std::fs::read_to_string(&golden).unwrap_or_else(|e| panic!("{}: {e}", golden.display()));
    assert_eq!(stdout(&out), want, "{RECIPE}: stdout is not the golden");
    let written = std::fs::read_to_string(dir.join("training").join("plan.txt"))
        .expect("--dry-run writes training/plan.txt");
    assert_eq!(written, want, "{RECIPE}: plan.txt is not the golden");
    // The route is `rl`, there is no bake, and the trainer is the PPO one.
    assert!(want.starts_with("# route: rl\n"), "{want}");
    assert!(!want.contains("es dataset bake"), "{want}");
    assert!(want.contains("python/es/train_ppo.py"), "{want}");
    assert!(want.contains("--rollout-docs docs"), "{want}");
    // No identity is claimed for a run that did not happen.
    assert!(!dir.join("training.lock").exists());

    let text = std::fs::read_to_string(rl_fixture("training-rl-demo.toml")).expect("the recipe");
    let refused = |body: &str, wanted: &str| {
        let recipe = dir.join("refused.toml");
        write(&recipe, body);
        let done = run_train(&train_toml_path(&recipe), &dir.join("refused"), &[]);
        let said = format!("{}{}", stdout(&done), stderr_of(&done));
        assert!(!done.status.success(), "accepted:\n{body}");
        assert!(said.contains(wanted), "wanted {wanted:?}, said:\n{said}");
    };
    // `[run] batch` beside `[rl]`: refused by name, with the derived number in the message,
    // rather than silently ignored into `precision.json`.
    refused(
        &text.replace(
            "steps         = 200",
            "steps         = 200\nbatch         = 8",
        ),
        "`batch` is 8 and this recipe has an `[rl]` table",
    );
    // ... and the mirror of it: the two demonstration routes still owe a batch.
    refused(
        &std::fs::read_to_string(vl_fixture("training.toml"))
            .expect("training.toml")
            .replace("batch         = 8\n", ""),
        "[run] `batch` is required",
    );
    // `[dataset]` is *accepted* absent here, which the golden above already proves, and a
    // `[dataset]`-less recipe on the IR route is still the old refusal.
    refused(
        &std::fs::read_to_string(vl_fixture("training.toml"))
            .expect("training.toml")
            .replace("[dataset]", "[unused]"),
        "unknown field",
    );
    // A second algorithm is a value, not a table: it is refused until it is implemented.
    refused(
        &text.replace("algo        = \"ppo\"", "algo        = \"sac\""),
        "[rl] `algo` is \"sac\"",
    );
    // A minibatch count that does not divide `envs * horizon` would weight the last rows of
    // every epoch differently, so it is named rather than rounded.
    refused(
        &text.replace("minibatches = 4", "minibatches = 7"),
        "it has to divide them",
    );
}

/// Regenerates `tests/golden/train/plan-rl.txt`.
#[test]
#[ignore = "golden generator; run explicitly"]
fn generate_rl_plan_golden() {
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_rl_plan_golden: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    let dir = scratch_dir("train-rl-golden");
    let out = run_train(
        "tests/fixtures/rl/training-rl-demo.toml",
        &dir,
        &["--dry-run"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", stderr_of(&out));
    write(&train_golden("plan-rl.txt"), &stdout(&out));
}

/// Runs one real `[rl]` recipe end to end under `ES_PYTHON`, or says why it did not.
///
/// `ES_PYTHON` has to name an interpreter with **torch, mujoco and the `es_native`
/// extension**: the trainer steps `es_native.Rollout`, which runs the `MuJoCo` reference
/// backend out of process. `python/es/README.md` has the one `maturin develop` line that
/// builds it.
fn run_rl_train(recipe: &str, dir: &Path, name: &str) -> Option<(PathBuf, String)> {
    let Ok(python) = std::env::var("ES_PYTHON") else {
        println!("SKIP {name}: ES_PYTHON is not set");
        return None;
    };
    let path = dir.join(format!("{name}.toml"));
    write(&path, recipe);
    let out = dir.join(name);
    let done = bin()
        .current_dir(train_root())
        .args(["train", "--recipe", &train_toml_path(&path), "--out"])
        .arg(&out)
        .output()
        .expect("run es train");
    let said = format!("{}{}", stdout(&done), stderr_of(&done));
    if !done.status.success() {
        // A missing package is a skip with the interpreter's own words, never a green test
        // (spec 1.4): every other Python oracle in this repository reports it the same way.
        if said.contains("cannot import") || said.contains("es_native is not importable") {
            println!("SKIP {name}: {python} cannot import what the rl route needs\n{said}");
            return None;
        }
        panic!("es train failed:\n{said}");
    }
    Some((out, said))
}

/// Oracle 2. Two runs of one `[rl]` recipe on the CPU backend are bitwise equal -- every
/// checkpoint, the `training_hash` and the loss curve -- and the run records what it does not
/// know rather than inventing it.
///
/// This is spec 3.5 tier 1 for a *trainer*: the env's RNG, the action noise and the minibatch
/// order are all seeded and local, so the only thing that could move the bits is a
/// non-deterministic kernel, and `torch.use_deterministic_algorithms(True)` turns that into
/// an error rather than a different number.
#[test]
fn train_rl_two_runs_are_bitwise() {
    const TEST: &str = "train_rl_two_runs_are_bitwise";
    let dir = scratch_dir("train-rl-bitwise");
    let bundle = write_rl_bundle(&dir, "untrained.esb");
    let recipe = rl_recipe(&bundle, 3, 4, 16, "0,3").replace(
        "interpreter   = \"es-no-such-interpreter\"",
        "interpreter   = \"python\"",
    );
    let Some((a, _)) = run_rl_train(&recipe, &dir, "run-a") else {
        return;
    };
    let (b, _) = run_rl_train(&recipe, &dir, "run-b").expect("the first run resolved ES_PYTHON");

    for mark in ["0", "3"] {
        let name = format!("checkpoints/{mark}.esb");
        let (x, y) = (
            std::fs::read(a.join(&name)).expect(&name),
            std::fs::read(b.join(&name)).expect(&name),
        );
        assert_eq!(
            hex(blake3::hash(&x).as_bytes()),
            hex(blake3::hash(&y).as_bytes()),
            "{name} is not bitwise between two runs of one recipe"
        );
    }
    assert_eq!(
        train_lock(&a)["training_hash"],
        train_lock(&b)["training_hash"],
        "two identical runs have two training_hashes"
    );
    // `dataset.lock` reads `unset` -- the file, hashed as such, not a zero digest.
    let lock = std::fs::read_to_string(a.join("training").join("dataset.lock")).expect("lock");
    assert_eq!(lock, "{\"unset\":true}\n", "dataset.lock is {lock}");
    // The value network is written for resumption and is *not* in any bundle: the lowered
    // module declares no such tensor, so `es policy pack` could not have taken it.
    assert!(a.join("training").join("value.safetensors").is_file());

    let curve: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(a.join("metrics").join("loss-curve.json")).expect("curve"),
    )
    .expect("the curve is JSON");
    let rows = curve.as_array().expect("the curve is an array");
    assert_eq!(rows.len(), 3, "one row per iteration: {curve}");
    for field in [
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "return",
        "episode_len",
        "envelope_violation_rate",
        "executed_ne_sampled_rate",
        "samples_per_sec",
    ] {
        assert!(
            rows[0][field].is_number(),
            "loss-curve.json has no {field}: {curve}"
        );
    }
    println!("RAN {TEST}: 3 iterations twice, bitwise");
}

/// Oracle 3. A run that starts from `[init] policy` *is* that policy at iteration 0.
///
/// `[run] checkpoint_at = [0]` writes the module's state before the first update, so
/// `checkpoints/0.esb` has to carry the tensors the source bundle carried, bit for bit -- the
/// S1 identity, now through the PPO trainer. The source here is a bundle this test packs
/// itself from the same documents with drawn weights; swapping it for S2b's synthetic
/// playground import is the fixture change that packet owns.
#[test]
fn train_rl_init_from_import() {
    const TEST: &str = "train_rl_init_from_import";
    let dir = scratch_dir("train-rl-init");
    let bundle = write_rl_bundle(&dir, "untrained.esb");

    // A real checkpoint for *this* graph's own lowering: every key the module declares,
    // filled with a value derived from the key so no two tensors are equal.
    let learning = es_ir::serial::learning_from_toml(
        &std::fs::read_to_string(rl_fixture("learning-state.toml")).expect("learning-state"),
    )
    .expect("learning-state.toml");
    let module = es_policy::lower_to_torch(&learning).expect("the RL graph lowers");
    let tensors: es_policy::weights::Checkpoint = module
        .weight_shapes
        .iter()
        .map(|(key, shape)| {
            let n: u64 = shape.iter().product();
            let seed = f32::from(u8::try_from(key.len()).unwrap_or(7));
            let values = (0..n).map(|i| seed + i as f32 * 0.25).collect();
            (key.clone(), (shape.clone(), values))
        })
        .collect();
    let weights = es_policy::weights::write_safetensors(&tensors);
    let source = dir.join("source.esb");
    std::fs::write(
        &source,
        es_compile::PolicyBundle::build(
            &es_ir::serial::task_from_toml(
                &std::fs::read_to_string(vl_fixture("task.toml")).expect("task"),
            )
            .expect("task.toml"),
            &es_ir::serial::observation_from_toml(
                &std::fs::read_to_string(rl_fixture("observation-state.toml")).expect("obs"),
            )
            .expect("observation-state.toml"),
            &{
                let mut g = learning.clone();
                g.policy.weights = es_ir::learning::WeightsRef::Safetensors {
                    path: "policy.safetensors".to_owned(),
                    hash: *blake3::hash(&weights).as_bytes(),
                };
                g
            },
            &es_ir::serial::deployment_from_toml(
                &std::fs::read_to_string(rl_fixture("deployment-rl.toml")).expect("deployment"),
            )
            .expect("deployment-rl.toml"),
            &weights,
        )
        .expect("the source documents pack"),
    )
    .expect("write source.esb");

    // `steps = 1` and a mark at 0: one iteration so the run is real, and the mark that is
    // written before it so the comparison is against the state the run started from.
    let recipe = rl_recipe(&bundle, 1, 4, 16, "0")
        .replace(
            "interpreter   = \"es-no-such-interpreter\"",
            "interpreter   = \"python\"",
        )
        .replace(
            "[run]\n",
            &format!("[init]\npolicy = \"{}\"\n[run]\n", train_toml_path(&source)),
        );
    let Some((out, said)) = run_rl_train(&recipe, &dir, "init") else {
        return;
    };

    // Every tensor the module declares was copied, so iteration 0 is the source exactly.
    let lock = init_lock(&out);
    assert!(
        names(&lock, "copied").len() == module.weight_shapes.len(),
        "the source shares every tensor with the module\n{lock}"
    );
    let packed = std::fs::read(out.join("checkpoints").join("0.esb")).expect("0.esb");
    let opened = es_compile::PolicyBundle::open(&packed).expect("0.esb opens");
    // Tensor by tensor, not file byte by file byte: the source was written by
    // `es_policy::weights::write_safetensors` and this one by `train_ppo.py`, and the two
    // spell one header's keys in two orders (`data_offsets` first vs `dtype` first). What
    // "bitwise" means for a checkpoint is the *tensors*, and those are compared raw here --
    // no tolerance, no epsilon.
    assert_eq!(
        tensor_bytes(&opened.weights),
        tensor_bytes(&weights),
        "iteration 0 is not the policy this run started from, bit for bit\n{said}"
    );
    println!("RAN {TEST}: iteration 0 == [init] policy, bitwise");
}

// --- packet M8/S4d: the reach task's four documents -------------------------------------------

/// `tests/fixtures/rl/<name>`.
fn rl_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rl")
        .join(name)
}

/// The six actuated joints in the XML's declaration order, which is also `qpos[0..6]`,
/// `qvel[0..6]` and `ctrl[0..6]`.
const REACH_JOINTS: [&str; 6] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
];
/// `joint_pos[6] || joint_vel[6] || cube_pose[7] || gripper_pose[7]`
/// (`docs/design/rl-continuation.md` section 5).
const REACH_OBS_DIM: u64 = 26;
/// Success: the gripper body's origin within 30 mm of the cube body's.
const REACH_SUCCESS_M: f64 = 0.03;
/// 200 control steps at 50 Hz -- four seconds.
const REACH_STEPS: u32 = 200;
const REACH_CONTROL_HZ: u64 = 50;
/// The span the distance is normalized over. **One metre on purpose:** the affine map is then
/// the identity everywhere the arm can reach, so the reward term is exactly
/// `-||cube - gripper||` in metres, bit for bit. The `Normalize` is there because a `Reward`
/// must carry a policy-input unit (`TYPE-011`), not to rescale anything.
const REACH_SPAN_M: f64 = 1.0;

fn so101_scene() -> (es_assets::scene::SceneDesc, Vec<u8>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mjcf/so101_pick_place.xml");
    let xml = std::fs::read(&path).expect("the demo scene");
    let scene = es_assets::parse_mjcf(&String::from_utf8(xml.clone()).expect("utf-8"))
        .expect("the demo scene parses")
        .scene;
    (scene, xml)
}

fn reach_vec(n: u64, unit: Unit, frame: Frame) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([n]),
        unit,
        frame,
        time: TimeRef::Tick,
        image: None,
    }
}

/// The reach task: the demo's scene, reset draws and action space, with one reward cone --
/// `-||cube - gripper||`, plus a bonus on success -- and two `Terminate` predicates.
///
/// The `ResetState` and `Randomization` nodes are **the demo's own**, read out of its
/// committed document rather than restated: the cube's per-episode draw is what makes 16
/// held-out seeds 16 different problems, and a reach task has no reason to draw it otherwise.
fn reach_task() -> TaskIr {
    use es_ir::task::{Aggregation, ArithOp, CmpOp, NormKind, TerminationKind};

    let (scene, xml) = so101_scene();
    let body = |name: &str| {
        scene
            .bodies
            .iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("the scene has a body named {name}"))
            .id
    };
    let joint = |name: &str| {
        scene
            .joints
            .iter()
            .find(|j| j.name == name)
            .unwrap_or_else(|| panic!("the scene has a joint named {name}"))
            .id
    };

    let mut task = es_ir::serial::task_from_toml(
        &std::fs::read_to_string(vl_fixture("task.toml")).expect("the demo task"),
    )
    .expect("the demo task parses");
    let resets: Vec<TaskNode> = task
        .graph
        .nodes
        .values()
        .filter(|n| {
            matches!(
                n,
                TaskNode::ResetState { .. } | TaskNode::Randomization { .. }
            )
        })
        .cloned()
        .collect();
    task.graph.nodes.clear();
    task.graph.edges.clear();
    task.observation_spec.channels.clear();
    task.config.max_episode_steps = REACH_STEPS;
    task.scene.scene_hash = scene.scene_hash();
    task.scene.asset_hash = *blake3::hash(&xml).as_bytes();

    let (base, cube, gripper) = (body("base"), body("cube"), body("gripper"));
    let joints6 = |unit: Unit| reach_vec(6, unit, Frame::Joint(base));
    let pos3 = reach_vec(3, Unit::Length, Frame::World);
    let quat4 = reach_vec(4, Unit::Quaternion, Frame::World);
    let pose7 = reach_vec(7, Unit::Length, Frame::World);
    let metres = reach_vec(1, Unit::Length, Frame::World);
    let names = REACH_JOINTS.map(str::to_owned).to_vec();

    let nodes = [
        TaskNode::GetJointState {
            body: base,
            joints: names.clone(),
            quantity: JointQuantity::Position,
        },
        TaskNode::ObservationSpec {
            channel: "joint_pos".to_owned(),
            ty: joints6(Unit::Angle),
        },
        TaskNode::GetJointState {
            body: base,
            joints: names,
            quantity: JointQuantity::Velocity,
        },
        TaskNode::ObservationSpec {
            channel: "joint_vel".to_owned(),
            ty: joints6(Unit::AngularVelocity),
        },
        TaskNode::GetBodyPose {
            body: cube,
            relative_to: Frame::World,
        },
        TaskNode::Concat {
            parts: vec![pos3.clone(), quat4.clone()],
            axis: 0,
        },
        TaskNode::ObservationSpec {
            channel: "cube_pose".to_owned(),
            ty: pose7.clone(),
        },
        TaskNode::GetBodyPose {
            body: gripper,
            relative_to: Frame::World,
        },
        TaskNode::Concat {
            parts: vec![pos3.clone(), quat4],
            axis: 0,
        },
        TaskNode::ObservationSpec {
            channel: "gripper_pose".to_owned(),
            ty: pose7.clone(),
        },
        TaskNode::ActionSpec {
            space: TaskSpace::JointPosition,
            dim: 6,
            control_rate_hz: REACH_CONTROL_HZ as f32,
        },
        // The cone spec 6.5 draws: two body positions, a lane-wise difference, an L2 norm.
        TaskNode::Arith {
            op: ArithOp::Sub,
            ty: pos3,
        },
        TaskNode::Norm {
            kind: NormKind::L2,
            ty: reach_vec(3, Unit::Length, Frame::World),
        },
        TaskNode::Normalize {
            lo: vec![0.0],
            hi: vec![REACH_SPAN_M],
            out_lo: 0.0,
            out_hi: 1.0,
            ty: metres.clone(),
        },
        TaskNode::Reward {
            name: "reach_distance".to_owned(),
            weight: -1.0,
            aggregation: Aggregation::Sum,
            ty: PortType {
                unit: Unit::Normalized { lo: 0.0, hi: 1.0 },
                ..metres.clone()
            },
        },
        TaskNode::Compare {
            op: CmpOp::Lt,
            rhs: Some(REACH_SUCCESS_M),
            ty: metres,
        },
        // The flag is `Bool`, which is what `boolish` makes of the `Compare`'s own type;
        // the reward that scores it declares the same port.
        TaskNode::Reward {
            name: "reach_success".to_owned(),
            weight: 1.0,
            aggregation: Aggregation::Sum,
            ty: PortType {
                elem: ElemType::Bool,
                ..reach_vec(1, Unit::Dimensionless, Frame::World)
            },
        },
        TaskNode::Terminate {
            kind: TerminationKind::Success,
        },
        TaskNode::GetTime { since_reset: true },
        TaskNode::Compare {
            op: CmpOp::Ge,
            rhs: Some(f64::from(REACH_STEPS) / REACH_CONTROL_HZ as f64),
            ty: reach_vec(1, Unit::Time, Frame::World),
        },
        TaskNode::Terminate {
            kind: TerminationKind::Timeout,
        },
    ];
    for (i, node) in nodes.into_iter().enumerate() {
        task.graph.insert(NodeId(i as u32), node);
    }
    for (i, node) in resets.into_iter().enumerate() {
        task.graph.insert(NodeId(21 + i as u32), node);
    }
    let n = NodeId;
    for (from, from_port, to, to_port) in [
        (0, "value", 1, "value"),
        (2, "value", 3, "value"),
        (4, "pos", 5, "in0"),
        (4, "quat", 5, "in1"),
        (5, "value", 6, "value"),
        (7, "pos", 8, "in0"),
        (7, "quat", 8, "in1"),
        (8, "value", 9, "value"),
        // The same two `pos` ports feed the reward cone: one read of the world, two readers.
        (4, "pos", 11, "a"),
        (7, "pos", 11, "b"),
        (11, "value", 12, "value"),
        (12, "value", 13, "value"),
        (13, "value", 14, "value"),
        (12, "value", 15, "a"),
        (15, "value", 16, "value"),
        (15, "value", 17, "value"),
        (18, "value", 19, "a"),
        (19, "value", 20, "value"),
    ] {
        task.graph.connect(n(from), from_port, n(to), to_port);
    }

    task.observation_spec.channels.extend([
        (
            "joint_pos".to_owned(),
            ObsChannel {
                source: ObsSource::JointState { body: base, dof: 6 },
                ty: joints6(Unit::Angle),
            },
        ),
        (
            "joint_vel".to_owned(),
            ObsChannel {
                source: ObsSource::JointState {
                    body: joint("shoulder_pan"),
                    dof: 6,
                },
                ty: joints6(Unit::AngularVelocity),
            },
        ),
        (
            "cube_pose".to_owned(),
            ObsChannel {
                source: ObsSource::JointState {
                    body: joint("cube_free"),
                    dof: 7,
                },
                ty: pose7.clone(),
            },
        ),
        (
            "gripper_pose".to_owned(),
            ObsChannel {
                source: ObsSource::BodyPose(gripper),
                ty: pose7,
            },
        ),
    ]);
    task
}

/// Four `StateInput`s in the layout order, concatenated to 26 and normalized once.
fn reach_observation(task: &TaskIr) -> ObservationIr {
    let channel = |name: &str| task.observation_spec.channels[name].clone();
    let source_of = |name: &str| match channel(name).source {
        ObsSource::JointState { body, .. } | ObsSource::BodyPose(body) => body,
        other => panic!("{other:?} is not a state source"),
    };
    let order = ["joint_pos", "joint_vel", "cube_pose", "gripper_pose"];

    let mut ir = ObservationIr::new(1, task.task_hash().expect("the reach task hashes"));
    let mut parts = Vec::new();
    for (i, name) in order.iter().enumerate() {
        let ty = channel(name).ty;
        ir.graph.insert(
            NodeId(i as u32),
            ObservationNode::StateInput {
                source: source_of(name),
                io: Io::source(ty.clone()),
            },
        );
        parts.push(ty);
    }
    let wide = reach_vec(REACH_OBS_DIM, Unit::Dimensionless, Frame::World);
    let normalized = PortType {
        unit: action_unit(),
        ..wide.clone()
    };
    let (cat, norm) = (NodeId(4), NodeId(5));
    ir.graph.insert(
        cat,
        ObservationNode::Concat {
            axis: 0,
            time_align: None,
            io: Io::new(parts, wide.clone()),
        },
    );
    ir.graph.insert(
        norm,
        ObservationNode::Normalize {
            stats: NormalizeStats::Range { lo: -1.0, hi: 1.0 },
            io: Io::unary(wide, normalized.clone()),
        },
    );
    for i in 0..order.len() {
        ir.graph
            .connect(NodeId(i as u32), "out", cat, &format!("in{i}"));
    }
    ir.graph.connect(cat, "out", norm, "in0");
    ir.graph.outputs.push(PortRef::new(norm, "out"));
    ir.temporal.window = Some(TemporalWindow {
        n_steps: 1,
        stride: 1,
        align: Align::Hold,
    });
    ir.outputs = BTreeMap::from([(
        "state".to_owned(),
        ObservationOutput {
            port: PortRef::new(norm, "out"),
            ty: normalized,
        },
    )]);
    ir
}

/// The demo's envelope, executed one action at a time.
fn reach_deployment() -> DeploymentIr {
    let mut dep = es_ir::serial::deployment_from_toml(
        &std::fs::read_to_string(vl_fixture("deployment.toml")).expect("the demo deployment"),
    )
    .expect("the demo deployment parses");
    dep.action.horizon = 1;
    dep.action.execute_chunk = 1;
    dep.execution = ExecutionMode::RecedingHorizon;
    dep.rate.inference = dep.rate.control;
    for w in &mut dep.watchdogs.0 {
        if let Watchdog::EnvelopeViolationRate { max_frac, .. } = w {
            // INV-12: the watchdog stays on and every clamp is still counted; what moves is
            // the number it is measured against. An untrained policy commands poses the arm
            // is nowhere near, so the plane clamps nearly every tick by construction -- at
            // 0.9 the fallback latches a second into training and freezes the arm.
            *max_frac = 1.0;
        }
    }
    dep
}

/// The demo's evaluation, on held-out seeds, with the suites a state policy can feel.
fn reach_evaluation(task: &TaskIr, observation: &ObservationIr) -> EvaluationIr {
    let mut ev = es_ir::serial::evaluation_from_toml(
        &std::fs::read_to_string(vl_fixture("evaluation.toml")).expect("the demo evaluation"),
    )
    .expect("the demo evaluation parses");
    ev.task = hex(&task.task_hash().expect("the reach task hashes"));
    ev.observation = hex(&observation
        .observation_hash()
        .expect("the reach observation hashes"));
    // The two light suites turn a renderer knob, and this policy has no camera.
    ev.suites
        .retain(|s| !s.name.starts_with("light_") && !s.name.starts_with("color_"));
    ev.episodes = EpisodeBatch {
        n_episodes: 16,
        seeds: SeedPlan::Explicit((201..=216).collect()),
    };
    ev.acceptance = vec![AcceptanceCriterion {
        suite: Some("nominal".to_owned()),
        metric: MetricSpec::SuccessRate,
        comparator: Comparator::Ge,
        threshold: 0.8,
        aggregation: Aggregation::Mean,
    }];
    ev
}

/// A Learning IR the reach documents do **not** ship: `cross::check` needs all four sides, and
/// the policy is the packet after this one (S4, the PPO trainer). Nothing here is written to
/// disk -- it exists so the three boundaries that touch the contract are checked against the
/// shapes the four committed documents declare.
fn reach_learning_stand_in() -> LearningGraph {
    let state = Port::new(
        "state",
        PortType {
            frame: Frame::Policy,
            ..reach_vec(REACH_OBS_DIM, action_unit(), Frame::Policy)
        },
    );
    let feature = |dim: u64| reach_vec(dim, Unit::Dimensionless, Frame::Policy);
    let chunk = PortType {
        shape: Shape::new([1, 6]),
        ..reach_vec(1, action_unit(), Frame::Policy)
    };

    let mut nodes: Graph<LearningNode> = Graph::new(1);
    nodes.insert(
        NodeId(0),
        LearningNode::StateEncoder {
            inputs: vec![state.clone()],
            kind: StateEncoderKind::Mlp {
                hidden: vec![64, 64],
            },
            out_dim: 64,
        },
    );
    nodes.insert(
        NodeId(1),
        LearningNode::PolicyHead {
            inputs: vec![Port::new("feat", feature(64))],
            kind: HeadKind::Regression,
            action_dim: 6,
            horizon: 1,
        },
    );
    nodes.insert(
        NodeId(2),
        LearningNode::ActionChunker {
            inputs: vec![Port::new("chunk", chunk.clone())],
            horizon: 1,
            execute_chunk: 1,
            replan_hz: REACH_CONTROL_HZ as f32,
            mode: ActionExecutionMode::RecedingHorizon,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 2,
        },
    );
    nodes.connect(NodeId(0), "out", NodeId(1), "feat");
    nodes.connect(NodeId(1), "chunk", NodeId(2), "chunk");
    nodes.inputs.push(PortRef::new(NodeId(0), "state"));
    nodes.outputs.push(PortRef::new(NodeId(2), "out"));
    LearningGraph {
        schema_version: 1,
        inputs: vec![state.clone()],
        nodes,
        outputs: vec![Port::new("actions", chunk)],
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0; 32],
            },
            contract: PolicyContract {
                inputs: BTreeMap::from([("state".to_owned(), state)]),
                observation_window: 1,
                action_dim: 6,
                horizon: 1,
                execute_chunk: 1,
                replanning_hz: REACH_CONTROL_HZ as f32,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 2.0,
                    deadline_ms: 20.0,
                },
            },
        },
    }
}

const REACH_TASK_HEADER: &str = "\
# Task IR (spec 6) for the SO-101 reach task -- packet M8/S4d, the first RL-continuation task.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_reach_documents` from
# tests/fixtures/mjcf/so101_pick_place.xml and from tests/fixtures/visible-learning/task.toml,
# so no hash and no reset distribution in it is typed in by hand.
#
# What it asks for (docs/design/rl-continuation.md section 5): bring the gripper to the cube.
# The reward is `-||cube_pos - gripper_pos||` in metres at every control step, plus `1` on the
# step the distance first falls under 0.03 m; the episode ends there, or after 200 control
# steps (4 s at 50 Hz). Nothing is picked up: this is the smallest task that needs a *body
# position* in its reward, which is what packet M8/S4d makes executable.
#
# THE REWARD IS A NEGATED NORM, exactly. `Norm { kind: L2 }` lowers to `Sqrt` of the squares
# summed in lane order (crates/es-env/src/plan.rs), `Sqrt` is the IEEE basic operation and not
# a `DET-010` transcendental (spec 6.6), and the `Normalize` between the norm and the reward is
# a *unit* conversion, not a rescaling: `[0, 1] -> [0, 1]` is the identity map, so the term is
# the distance itself and the `Reward`'s own `weight = -1` is the minus sign. The clamp the
# `Normalize` carries bites only past one metre, which is outside the arm's reach. Without it
# the reward would carry `Unit::Length` and `TYPE-011` would refuse it, rightly: a reward is a
# policy-facing number.
#
# The success term is a second `Reward` reading the same `Compare` the `Terminate` reads, so
# the bonus and the episode end cannot disagree about what success is.
#
# The reset and randomization nodes ARE THE DEMO'S OWN, copied from task.toml rather than
# restated: the arm's reset pose and the cube's per-episode draw (`cube.x`, `cube.y`) are what
# make 16 held-out seeds 16 different problems.
#
# THE FOUR OBSERVATION CHANNELS and what an `ObsSource` can say about them (spec 7.4):
#  * `joint_pos[6]`  -- `JointState { body = base, dof = 6 }`, the leading six of the state
#    row, which is `qpos[0..6]`: the six joints in the XML's declaration order. This is the
#    binding the demo's own `joint_state` channel uses.
#  * `joint_vel[6]`  -- `qvel[0..6]`, the same six joints. `ObsSource` HAS NO VELOCITY SOURCE
#    AND NO OFFSET, and the Cross-IR check keys a channel by its source id alone (`XIR-002`),
#    so two joint channels cannot name the same id; this one names the block's first joint,
#    `shoulder_pan`, and means `dof` dofs from there. `es-eval`'s capture path
#    (crates/es-eval/src/runner.rs, `input_sources`) has no `qvel` reading at all, so this
#    channel is served by the trainer reading the env, not by that path.
#  * `cube_pose[7]`  -- the cube's free joint, `qpos[6..13]`: position and quaternion, exactly
#    the demo's `sim_cube_pose` binding, and exact rather than a leading-`dof` guess.
#  * `gripper_pose[7]` -- `BodyPose(gripper)`. The gripper has no joint of its own, so its
#    pose is only in `xpos`/`xquat`; `input_sources` has no `BodyPose` reading either, and
#    refuses it by name rather than serving something else.
# All seven values of each pose carry `Unit::Length` for the reason the demo's header gives:
# three of them are metres, the quaternion's four are dimensionless, and spec 5.4's algebra
# has no mixed unit.
#
# `cube_pose` IS SIMULATOR-PRIVILEGED, like the demo's `sim_cube_pose`: no real SO-101 knows
# where the cube is. The name has no `sim_` prefix here because this task is a training
# environment for the RL continuation, not a hardware deployment -- a deployment aimed at
# hardware must replace both pose channels with something a camera can produce.
";

const REACH_OBSERVATION_HEADER: &str = "\
# Observation IR (spec 7) for the SO-101 reach task -- packet M8/S4d.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_reach_documents`;
# `task_ref` is task-reach.toml's own `task_hash`.
#
# Four `StateInput`s -> `Concat` -> one `Normalize`, in the layout order
# `joint_pos[6] || joint_vel[6] || cube_pose[7] || gripper_pose[7]` = 26
# (docs/design/rl-continuation.md section 5). The concatenated vector is `Dimensionless`
# because it is a *mixed* one -- rad, rad/s, metres and a quaternion -- and spec 5.4's unit
# algebra has no mixed unit; the per-block units are in task-reach.toml's header, which is
# where a reader looks for them.
#
# The `Normalize { Range { -1, 1 } }` is what makes the tensor admissible as a policy input
# (`OBS-040`, spec 5.4): the policy reads `Unit::Normalized { -1, 1 }`. Its range is the
# identity on that interval and is not a per-channel standardization -- the running statistics
# a PPO run accumulates belong to a Learning IR `Normalizer` node (the Go1 track's argument in
# tests/fixtures/quadruped/observation.toml), not here.
#
# `temporal.window = { n_steps = 1, stride = 1, align = \"Hold\" }` is layer 2 of spec 7.5:
# one frame, no stacking. `XIR-011` checks it against the policy contract's
# `observation_window`.
#
# There is no image chain and no `ImageSpec`: this is a state policy, so there is no `Resize`,
# no `Crop`, and nothing owes an intrinsics transform (spec 7.2, INV-14).
";

const REACH_DEPLOYMENT_HEADER: &str = "\
# Deployment IR + Safety Plane (spec 9) for the SO-101 reach task -- packet M8/S4d.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_reach_documents` from
# tests/fixtures/visible-learning/deployment.toml: the envelope IS THE DEMO'S, joint for
# joint, because it is the same arm in the same scene. That document's header is where every
# number's physical argument lives -- the STS3215's 2.94 N m stall torque, 3.0 rad/s, the
# 80 rad/s^2 that packet M5/V18 measured, the workspace box -- and none of them moved here.
#
# Three things did move, and all three are about *how a chunk is executed*, not about what is
# safe:
#  * `action.horizon = 1`, `execute_chunk = 1`, `execution = receding_horizon`. An RL policy
#    emits one action per control step; there is no chunk to buffer and nothing to ensemble.
#  * `rate.inference = rate.control = 50 Hz`. The policy runs every control tick, so the gap
#    the `inference_deadline` watchdog measures is one control period, not the ten the demo's
#    5 Hz replanning left.
#  * `envelope_violation_rate.max_frac = 1.0`. INV-12 IS WHY THIS IS THE SHAPE OF THE CHANGE:
#    the watchdog stays on, every clamp is still counted and still recorded as
#    `action_source = Clamped`; what moves is the rate it latches the fallback at. An
#    untrained policy -- and PPO starts untrained -- commands poses the arm is nowhere near,
#    so the plane clamps nearly every tick by construction. At the demo's 0.9 the fallback
#    latches a second into the first episode and freezes the arm, and a frozen arm teaches a
#    policy nothing. Widening the number is the sanctioned move; disabling the plane is not.
#
# `target` is the simulated scene. A real SO-101 swaps it for `Physical { driver }` without
# touching the envelope -- but not without replacing the two privileged pose channels first
# (task-reach.toml's header).
";

const REACH_EVALUATION_HEADER: &str = "\
# Evaluation IR (spec 10) for the SO-101 reach task -- packet M8/S4d.
#
# Generated by `cargo test -p es --test cli -- --ignored regenerate_reach_documents`; `task`
# and `observation` are task-reach.toml's and observation-reach.toml's own hashes.
#
# 16 episodes on fixed seeds 201-216, outside both the 1-50 the demo's demonstrations were
# collected on and the 101-116 its own evaluation uses, so a reach number is never a memory of
# either run. Sixteen because `Evaluation::run` gives every episode its own env.
#
# The cube's initial pose is not a perturbation: task-reach.toml's `Randomization` node moves
# the cube's free joint at every reset, in every suite, which is what spec 6.3 is for.
#
# The suites are the demo's minus the two light ones: `light_intensity` and `light_direction`
# turn a renderer knob, and a policy that reads joint angles and two poses cannot feel it --
# a row that is identical to `nominal` by construction is not a measurement. What is left
# touches the actuator and the clock, which this policy does feel:
#  * `observation_delay` -- one and two control steps at 50 Hz.
#  * `torque_noise`, `backlash` -- the actuator is not ideal.
#
# `success_rate >= 0.8` on `nominal` is the acceptance criterion, and it is deliberately
# harder than the demo's 0.5: reaching is a far easier task than picking and placing, and a
# continuation that cannot reach four times out of five has not learned it. The perturbed
# rows are measurements to report, not gates to pass (spec 10.4).
";

/// Regenerates `tests/fixtures/rl/{task,observation,deployment,evaluation}-reach.toml`. Run
/// explicitly:
///
///     ES_GENERATE_GOLDENS=1 cargo test -p es --test cli -- --ignored regenerate_reach_documents
#[test]
#[ignore = "fixture generator; run explicitly"]
fn regenerate_reach_documents() {
    // Spec 1.4: goldens and fixtures are CI read-only, and `cargo test -- --include-ignored`
    // runs every ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP regenerate_reach_documents: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    let task = reach_task();
    let observation = reach_observation(&task);
    let deployment = reach_deployment();
    let evaluation = reach_evaluation(&task, &observation);
    std::fs::create_dir_all(rl_fixture(".")).expect("tests/fixtures/rl");
    for (name, header, text) in [
        (
            "task-reach.toml",
            REACH_TASK_HEADER,
            es_ir::serial::task_to_toml(&task).expect("task toml"),
        ),
        (
            "observation-reach.toml",
            REACH_OBSERVATION_HEADER,
            es_ir::serial::observation_to_toml(&observation).expect("observation toml"),
        ),
        (
            "deployment-reach.toml",
            REACH_DEPLOYMENT_HEADER,
            es_ir::serial::deployment_to_toml(&deployment).expect("deployment toml"),
        ),
        (
            "evaluation-reach.toml",
            REACH_EVALUATION_HEADER,
            es_ir::serial::evaluation_to_toml(&evaluation).expect("evaluation toml"),
        ),
    ] {
        write(&rl_fixture(name), &format!("{header}\n{text}"));
        println!("wrote {}", rl_fixture(name).display());
    }
}

/// Packet M8/S4d oracle 4: the four committed documents parse, validate, agree across every
/// boundary the Cross-IR check can see, compile, and carry the 26-wide port the design note
/// declares -- and they are exactly what the generator writes.
#[test]
fn reach_documents_validate() {
    let read = |name: &str| std::fs::read_to_string(rl_fixture(name)).expect(name);
    let task = es_ir::serial::task_from_toml(&read("task-reach.toml")).expect("the task parses");
    let observation = es_ir::serial::observation_from_toml(&read("observation-reach.toml"))
        .expect("the observation parses");
    let deployment = es_ir::serial::deployment_from_toml(&read("deployment-reach.toml"))
        .expect("the deployment parses");
    let evaluation = es_ir::serial::evaluation_from_toml(&read("evaluation-reach.toml"))
        .expect("the evaluation parses");

    for (name, diags) in [
        ("task", task.validate()),
        ("observation", observation.validate()),
        ("deployment", deployment.validate()),
        ("evaluation", evaluation.validate()),
    ] {
        assert!(diags.is_empty(), "{name}-reach.toml: {diags:#?}");
    }

    // The Learning IR is the next packet's; the stand-in carries the contract the four
    // documents imply, so every `XIR_*` boundary is checked rather than assumed.
    let learning = reach_learning_stand_in();
    let diags = cross::check(&IrBundle {
        task: &task,
        observation: &observation,
        learning: &learning,
        deployment: &deployment,
        evaluation: Some(&evaluation),
    });
    assert!(diags.is_empty(), "cross-IR: {diags:#?}");

    // The 26 of `docs/design/rl-continuation.md` section 5, and the blocks that make it up.
    assert_eq!(
        observation.outputs["state"].ty.shape.dims(),
        [REACH_OBS_DIM],
        "the policy's input is 26 wide"
    );
    let widths: Vec<u64> = ["joint_pos", "joint_vel", "cube_pose", "gripper_pose"]
        .iter()
        .map(|n| task.observation_spec.channels[*n].ty.shape.dims()[0])
        .collect();
    assert_eq!(widths, [6, 6, 7, 7]);
    assert_eq!(widths.iter().sum::<u64>(), REACH_OBS_DIM);
    assert_eq!(task.config.max_episode_steps, REACH_STEPS);

    // The documents are what the generator writes: a hand edit to any of the four is a
    // failing test, not a number nobody can trace back to the scene.
    //
    // ONE FIELD IS EXEMPT, and not because this packet wanted it that way. `scene_hash` is
    // **not the same number on every OS** for this scene: `crates/es-assets/src/mjcf/orient.rs`
    // turns an `euler=` attribute into a quaternion with `f64::sin_cos`, the platform's libm,
    // and the SO-101 scene has two of them -- so Windows and Linux hash the same XML bytes to
    // different digests (measured, packet M8/S4d; regenerating the *demo's* task.toml on Linux
    // moves its `scene_hash` the same way). That is a spec 3.4 / `DET-010` defect on the asset
    // path and it predates this packet; `asset_hash`, which is blake3 of the file, is checked
    // instead, and the scene ref the committed document carries is what the rest is rebuilt
    // against so the comparison measures this packet's documents and not that bug.
    let (_, xml) = so101_scene();
    assert_eq!(
        task.scene.asset_hash,
        *blake3::hash(&xml).as_bytes(),
        "the document names a different file than the one it was generated from"
    );
    let mut built = reach_task();
    built.scene = task.scene.clone();
    let built_obs = reach_observation(&built);
    for (name, text) in [
        ("task-reach.toml", es_ir::serial::task_to_toml(&built)),
        (
            "observation-reach.toml",
            es_ir::serial::observation_to_toml(&built_obs),
        ),
        (
            "deployment-reach.toml",
            es_ir::serial::deployment_to_toml(&reach_deployment()),
        ),
        (
            "evaluation-reach.toml",
            es_ir::serial::evaluation_to_toml(&reach_evaluation(&built, &built_obs)),
        ),
    ] {
        let on_disk = read(name);
        let body = on_disk
            .split_once("\nes_schema")
            .map(|(_, rest)| format!("es_schema{rest}"))
            .unwrap_or(on_disk);
        assert_eq!(
            body,
            text.expect("the document serializes"),
            "{name} is not what the generator writes; rerun \
             `ES_GENERATE_GOLDENS=1 cargo test -p es --test cli -- --ignored \
             regenerate_reach_documents`"
        );
    }

    // And the CLI agrees: `es task compile` accepts the pair.
    let out = bin()
        .args(["task", "compile"])
        .arg(rl_fixture("task-reach.toml"))
        .arg(rl_fixture("observation-reach.toml"))
        .output()
        .expect("run es task compile");
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("compiler_hash: "), "{text}");
    assert!(text.contains("shape=[26]"), "{text}");
    println!(
        "RAN reach_documents_validate: task {} observation {}",
        hex(&task.task_hash().expect("task hash")),
        hex(&observation.observation_hash().expect("observation hash"))
    );
}
