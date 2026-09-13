//! `EmbeddedRuntime` and `PolicyBundle` over the cross-IR fixture (spec 9.6, spec 10.5).
//!
//! The fixture-building functions below (through `hex`) are copied verbatim from
//! `crates/es-ir/tests/cross_fixture.rs`, exactly as `crates/es/tests/cli.rs` does and for
//! the same reason (the M1 packets say: reuse that crate's own cross-IR fixture rather than
//! inventing a second idea of a consistent bundle). Only `Fixture::new()` is needed here, so
//! the violation-scenario tests are left out; the deployment timing is widened in
//! `widen_inference_budget` so a 10 Hz replan cadence fits the watchdogs (INV-12: widen the
//! envelope, never disable it).
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

// --- W9: policy.esb and the embedded runtime --------------------------------------------------

use es_compile::bundle::{self, BundleError, BundleKind, PolicyBundle};
use es_compile::{CpuPlan, PlanMode, Tensor};
use es_core::PhysTick;
use es_policy::runtime::{runtime_hash_of, InferenceBackend, PolicyInfo};
use es_policy::{PolicyError, PolicyRuntime, WeightsSource};
use es_runtime_embedded::{EmbeddedRuntime, RuntimeError};
use es_safety::{ActionSource, FallbackKind};

const NJ: usize = DOF as usize;
const H: usize = HORIZON as usize;

/// Bytes that stand in for a checkpoint. Nothing parses them: `FakeRuntime` is the whole
/// policy, and the container only has to carry them intact.
const WEIGHTS: &[u8] = b"not really safetensors, but hashed like it";

/// A `PolicyRuntime` that emits a constant chunk, so the test can judge the runtime around it
/// (spec 1.4: the harness must not depend on `torch` being installed).
#[derive(Debug)]
struct FakeRuntime {
    /// Every action component of every emitted row.
    value: f64,
    /// Added to `value` on each call, so two successive chunks differ. The Safety Plane keeps
    /// its cursor when a chunk is byte-identical to the stored one, which a constant policy
    /// would run out of (see `docs/design/safety-plane.md`).
    drift: f64,
    calls: u32,
    info: Option<PolicyInfo>,
}

impl FakeRuntime {
    fn new(value: f64, drift: f64) -> Self {
        Self {
            value,
            drift,
            calls: 0,
            info: None,
        }
    }
}

impl PolicyRuntime for FakeRuntime {
    fn load(
        &mut self,
        graph: &LearningGraph,
        _weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError> {
        let c = &graph.policy.contract;
        let info = PolicyInfo {
            backend: InferenceBackend::Onnx,
            lowering_hash: [0u8; 32],
            weights_hash: *graph.policy.weights.hash(),
            action_dim: c.action_dim,
            horizon: c.horizon,
            version: "fake-0".to_owned(),
        };
        self.info = Some(info.clone());
        Ok(info)
    }

    fn infer(
        &mut self,
        _inputs: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>, PolicyError> {
        let v = (self.value + self.drift * f64::from(self.calls)) as f32;
        self.calls += 1;
        let rows = CHUNK as usize;
        let mut data = Vec::with_capacity(rows * NJ * 4);
        for _ in 0..rows * NJ {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Ok(BTreeMap::from([(
            "actions".to_owned(),
            Tensor {
                dtype: ElemType::F32,
                shape: vec![u64::from(CHUNK), u64::from(DOF)],
                data,
            },
        )]))
    }

    fn info(&self) -> Option<&PolicyInfo> {
        self.info.as_ref()
    }

    fn runtime_hash(&self) -> [u8; 32] {
        runtime_hash_of(InferenceBackend::Onnx, 0, "fake-0")
    }
}

/// The fixture's inference budget is two control periods, which only admits a replan every
/// other tick. Widening it (INV-12: widen the envelope, never switch a watchdog off) lets the
/// 10 Hz replan cadence the Learning IR declares actually run.
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
fn deployable() -> Fixture {
    let mut f = Fixture::new();
    f.learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(WEIGHTS).as_bytes(),
    };
    widen_inference_budget(&mut f.deployment);
    assert!(f.diags().is_empty(), "fixture must stay consistent");
    f
}

fn build() -> Vec<u8> {
    let f = deployable();
    PolicyBundle::build(&f.task, &f.observation, &f.learning, &f.deployment, WEIGHTS)
        .expect("the fixture builds a policy bundle")
}

/// Sensor tensors keyed the way `CpuPlan::inputs` names them (the `StableId` of the source).
fn sensors() -> BTreeMap<String, Tensor> {
    let image = Tensor {
        dtype: ElemType::F32,
        shape: vec![3, 480, 640],
        data: vec![0u8; 3 * 480 * 640 * 4],
    };
    let joints = Tensor {
        dtype: ElemType::F32,
        shape: vec![u64::from(DOF)],
        data: vec![0u8; NJ * 4],
    };
    BTreeMap::from([
        (cam_id().to_string(), image),
        (robot_id().to_string(), joints),
    ])
}

fn runtime_with(policy: FakeRuntime) -> EmbeddedRuntime<NJ, H> {
    EmbeddedRuntime::from_bundle(&build(), Box::new(policy)).expect("the bundle loads")
}

fn replanned(rt: &EmbeddedRuntime<NJ, H>) -> bool {
    rt.telemetry()
        .iter_newest()
        .next()
        .expect("a record")
        .replanned
}

// --- bundle -----------------------------------------------------------------------------------

#[test]
fn policy_bundle_round_trips_and_is_byte_identical() {
    let bytes = build();
    assert_eq!(bytes, build(), "identical inputs give identical bytes");

    let open = PolicyBundle::open(&bytes).expect("opens");
    let f = deployable();
    assert_eq!(open.task, f.task);
    assert_eq!(open.observation, f.observation);
    assert_eq!(open.learning, f.learning);
    assert_eq!(open.deployment, f.deployment);
    assert_eq!(open.weights, WEIGHTS);
    assert_eq!(open.manifest.kind, BundleKind::Policy);
    assert_eq!(
        open.manifest.hashes.task,
        Some(f.task.task_hash().expect("hashes"))
    );
    assert_eq!(
        open.manifest.hashes.compiler,
        Some(
            CpuPlan::compile(&f.observation, PlanMode::Release)
                .expect("compiles")
                .compiler_hash()
        )
    );
    // The plan is rebuilt from the IR, not stored: `CpuPlan` is not `Serialize`.
    assert!(open.compile_plan().is_ok());
}

#[test]
fn a_flipped_payload_byte_is_caught() {
    let mut bytes = build();
    let at = bytes.len() / 2;
    bytes[at] ^= 0x01;
    assert!(
        matches!(
            PolicyBundle::open(&bytes),
            Err(BundleError::EntryHash { .. })
        ),
        "a single flipped byte must not open"
    );
}

#[test]
fn a_swapped_manifest_hash_names_its_slot() {
    let raw = bundle::read(&build()).expect("reads");
    let mut manifest = raw.manifest.clone();
    manifest.hashes.learning = Some([0x5a; 32]);
    let forged = bundle::write(&manifest, &raw.entries).expect("writes");
    assert_eq!(
        PolicyBundle::open(&forged),
        Err(BundleError::HashMismatch { slot: "learning" })
    );
}

#[test]
fn weights_that_do_not_match_the_ir_are_refused() {
    let f = deployable();
    let err = PolicyBundle::build(
        &f.task,
        &f.observation,
        &f.learning,
        &f.deployment,
        b"other bytes",
    )
    .expect_err("the declared weights hash is checked");
    assert_eq!(err, BundleError::HashMismatch { slot: "weights" });
}

// --- runtime ----------------------------------------------------------------------------------

#[test]
fn a_benign_chunk_reaches_the_actuator_as_policy() {
    let mut rt = runtime_with(FakeRuntime::new(0.0, 1e-6));
    let action = rt.tick(&sensors(), PhysTick(0), Micros(0));
    assert_eq!(action.source, ActionSource::Policy, "{action:?}");
    assert!(action.is_clean());
    assert_eq!(rt.telemetry().len(), 1);
    assert!(replanned(&rt));
}

#[test]
fn a_nan_chunk_falls_back() {
    let mut rt = runtime_with(FakeRuntime::new(f64::NAN, 0.0));
    let action = rt.tick(&sensors(), PhysTick(0), Micros(0));
    assert_eq!(
        action.source,
        ActionSource::Fallback(FallbackKind::HoldPosition),
        "{action:?}"
    );
    assert!(action.q.iter().all(|v| v.is_finite()), "INV-13: {action:?}");
}

#[test]
fn a_wrong_joint_count_cannot_load() {
    let err = EmbeddedRuntime::<3, H>::from_bundle(&build(), Box::new(FakeRuntime::new(0.0, 0.0)))
        .expect_err("NJ must match the deployment");
    assert!(matches!(err, RuntimeError::Safety(_)), "{err:?}");
}

#[test]
fn the_replan_cadence_is_honored() {
    // 100 Hz control / 10 Hz inference = one inference per 10 control ticks, capped by
    // execute_chunk = 20 (spec 8.6).
    let mut rt = runtime_with(FakeRuntime::new(0.0, 1e-6));
    assert_eq!(rt.replan_interval(), 10);

    let s = sensors();
    let mut replans = 0;
    for t in 0..35u64 {
        let action = rt.tick(&s, PhysTick(t), Micros(0));
        assert!(action.is_clean(), "tick {t}: {action:?}");
        if replanned(&rt) {
            replans += 1;
        }
    }
    assert_eq!(replans, 4, "ticks 0, 10, 20 and 30");
    assert_eq!(rt.telemetry().len(), 35);
}

#[test]
fn a_chunk_reuse_tick_does_not_allocate() {
    if !es_core::alloc_count::counting_enabled() {
        return;
    }
    let mut rt = runtime_with(FakeRuntime::new(0.0, 1e-6));
    let s = sensors();
    // Tick 0 replans (plan arena + PolicyRuntime: the two documented boundaries). Ticks 1..9
    // reuse the buffered chunk, and that path is the one spec 9.6 requires to be allocation
    // free.
    let _ = rt.tick(&s, PhysTick(0), Micros(0));
    for t in 1..10u64 {
        let action = es_core::alloc_count::assert_no_alloc(|| rt.tick(&s, PhysTick(t), Micros(0)));
        assert!(action.is_clean(), "tick {t}: {action:?}");
    }
}

#[test]
fn execution_hash_covers_the_runtime_and_the_hardware() {
    let rt = runtime_with(FakeRuntime::new(0.0, 0.0));
    let h = rt.execution_hash();
    assert_ne!(h, [0u8; 32]);
    assert_eq!(h, rt.execution_hash(), "it is a pure function");
    assert_eq!(rt.hardware(), es_runtime_embedded::hardware_capability());
    // Two bundles of the same IRs on the same machine with the same backend agree.
    let other = runtime_with(FakeRuntime::new(0.0, 0.0));
    assert_eq!(h, other.execution_hash());
}
