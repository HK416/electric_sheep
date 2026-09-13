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
    ActionExecutionMode, ArchKind, ChunkBlendPolicy, FusionKind, HeadKind, LearningGraph,
    LearningNode, PolicyContract, PolicyHandle, RuntimeHints, StateEncoderKind, TemporalKind,
    VisionBackbone, WeightsRef,
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
            pretrained: true,
            frozen: false,
            out_dim: feat,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(1),
        LearningNode::StateEncoder {
            inputs: vec![state.clone()],
            kind: StateEncoderKind::Mlp { hidden: vec![256] },
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
