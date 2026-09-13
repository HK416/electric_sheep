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

/// S-9 (P-M4-S9-S10): a blocked RoboVerse conversion must leave no output behind -- the
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
