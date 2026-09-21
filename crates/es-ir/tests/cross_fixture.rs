//! Cross-IR fixtures (P26, CI gate 4 of spec 28.7): one bundle of five IRs that agree, and one
//! violation scenario per rule of `es_ir::cross`.
//!
//! Everything is built by hand here rather than through an IR module's `testing` helpers: the
//! point of the gate is that the five IRs are consistent *as written*, so the fixture must not
//! borrow a neighbour's idea of consistency.

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
    AugmentKind, Io, NormalizeStats, ObservationIr, ObservationNode, ObservationOutput,
    ResizeFilter, TemporalWindow,
};
use es_ir::task::{
    ActionSpace as TaskSpace, JointQuantity, ObsChannel, ObsSource, ObservationSpec, SceneRef,
    TaskConfig, TaskIr, TaskNode,
};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_ir::{codes, Diagnostic};

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

#[track_caller]
fn assert_code(f: &Fixture, code: &str) {
    let diags = f.diags();
    assert!(
        diags.iter().any(|d| d.code.as_str() == code),
        "expected {code}, got {diags:#?}"
    );
}

// --- The valid bundle -------------------------------------------------------------------------

#[test]
fn cross_valid_bundle_has_no_diagnostics() {
    let f = Fixture::new();
    assert!(f.diags().is_empty(), "{:#?}", f.diags());
}

#[test]
fn cross_valid_bundle_validates_per_ir() {
    let f = Fixture::new();
    assert!(f.task.validate().is_empty(), "{:#?}", f.task.validate());
    assert!(
        f.observation.validate().is_empty(),
        "{:#?}",
        f.observation.validate()
    );
    assert!(
        f.learning.validate().is_empty(),
        "{:#?}",
        f.learning.validate()
    );
    assert!(
        f.deployment.validate().is_empty(),
        "{:#?}",
        f.deployment.validate()
    );
    assert!(
        f.evaluation.validate().is_empty(),
        "{:#?}",
        f.evaluation.validate()
    );
}

#[test]
fn cross_evaluation_is_optional() {
    let f = Fixture::new();
    assert!(cross::check(&IrBundle {
        task: &f.task,
        observation: &f.observation,
        learning: &f.learning,
        deployment: &f.deployment,
        evaluation: None,
    })
    .is_empty());
}

// --- Rule 1: Task.ObservationSpec <-> Observation inputs --------------------------------------

#[test]
fn cross_observation_of_another_task() {
    let mut f = Fixture::new();
    f.observation.task_ref = [0u8; 32];
    assert_code(&f, codes::XIR_001);
}

#[test]
fn cross_source_node_reads_an_undeclared_channel() {
    let mut f = Fixture::new();
    let ObservationNode::ImageInput { sensor, .. } =
        f.observation.graph.nodes.get_mut(&NodeId(0)).unwrap()
    else {
        panic!("node 0 is the ImageInput");
    };
    *sensor = StableId::from_path("cam_wrist");
    assert_code(&f, codes::XIR_002);
}

#[test]
fn cross_source_node_type_disagrees_with_the_declaration() {
    let mut f = Fixture::new();
    let node = f.observation.graph.nodes.get_mut(&NodeId(3)).unwrap();
    let ObservationNode::StateInput { io, .. } = node else {
        panic!("node 3 is the StateInput");
    };
    *io = Io::source(PortType {
        shape: Shape::new([6]),
        ..state_ty(Unit::Angle)
    });
    assert_code(&f, codes::TYPE_002);
}

// --- Rule 2: Observation.outputs <-> Learning.inputs ------------------------------------------

#[test]
fn cross_policy_input_is_not_produced() {
    let mut f = Fixture::new();
    let out = f.observation.outputs.remove("rgb_front").unwrap();
    f.observation.outputs.insert("rgb_wrist".to_owned(), out);
    f.seal();
    assert_code(&f, codes::XIR_010);
}

#[test]
fn cross_policy_input_shape_disagrees() {
    let mut f = Fixture::new();
    let port = f
        .learning
        .policy
        .contract
        .inputs
        .get_mut("rgb_front")
        .unwrap();
    port.ty.shape = Shape::new([3, 256, 256]);
    assert_code(&f, codes::TYPE_002);
}

#[test]
fn cross_policy_input_is_not_normalized() {
    let mut f = Fixture::new();
    let port = f
        .learning
        .policy
        .contract
        .inputs
        .get_mut("joint_state")
        .unwrap();
    port.ty.unit = Unit::Angle;
    assert_code(&f, codes::TYPE_011);
}

#[test]
fn cross_image_output_advertises_stale_geometry() {
    let mut f = Fixture::new();
    let out = f.observation.outputs.get_mut("rgb_front").unwrap();
    // The graph resizes to 224x224; the tensor still claims the sensor's own geometry.
    out.ty.image = Some(camera());
    f.seal();
    assert_code(&f, codes::TYPE_020);
}

#[test]
fn cross_temporal_window_disagrees_with_observation_window() {
    let mut f = Fixture::new();
    f.observation.temporal.window = Some(TemporalWindow {
        n_steps: 2,
        stride: 1,
        align: Align::Hold,
    });
    f.seal();
    assert_code(&f, codes::XIR_011);
}

// --- Rules 3 and 4: Learning <-> Deployment ---------------------------------------------------

#[test]
fn cross_action_dim_disagrees_with_the_deployment() {
    let mut f = Fixture::new();
    f.learning.policy.contract.action_dim = 7;
    assert_code(&f, codes::XIR_020);
}

#[test]
fn cross_execution_mode_disagrees() {
    let mut f = Fixture::new();
    f.deployment.execution = ExecutionMode::RecedingHorizon;
    assert_code(&f, codes::XIR_021);
}

#[test]
fn cross_chunk_shape_disagrees() {
    let mut f = Fixture::new();
    f.learning.policy.contract.horizon = 40;
    assert_code(&f, codes::XIR_022);
}

#[test]
fn cross_replanning_rate_is_not_an_integer_divisor() {
    let mut f = Fixture::new();
    f.learning.policy.contract.replanning_hz = 30.0;
    assert_code(&f, codes::XIR_023);
}

#[test]
fn cross_inference_latency_exceeds_the_budget() {
    let mut f = Fixture::new();
    // Diffusion Policy on an RTX 4090 (spec 8.4): 369.8 ms against a 20 ms budget.
    f.learning.policy.contract.runtime.expected_latency_ms = 369.8;
    assert_code(&f, codes::LRN_052);
}

#[test]
fn cross_policy_deadline_exceeds_the_watchdog_budget() {
    let mut f = Fixture::new();
    f.learning.policy.contract.runtime.deadline_ms = 50.0;
    assert_code(&f, codes::XIR_024);
}

// --- Rule 5: Task.ActionSpec <-> Learning / Deployment ----------------------------------------

#[test]
fn cross_task_action_dim_disagrees() {
    let mut f = Fixture::new();
    let node = f.task.graph.nodes.get_mut(&NodeId(4)).unwrap();
    let TaskNode::ActionSpec { dim, .. } = node else {
        panic!("node 4 is the ActionSpec");
    };
    *dim = 6;
    f.seal();
    assert_code(&f, codes::XIR_030);
}

#[test]
fn cross_task_action_space_disagrees() {
    let mut f = Fixture::new();
    f.deployment.action.space = DepSpace::JointTorque;
    assert_code(&f, codes::XIR_031);
}

#[test]
fn cross_task_control_rate_disagrees() {
    let mut f = Fixture::new();
    let node = f.task.graph.nodes.get_mut(&NodeId(4)).unwrap();
    let TaskNode::ActionSpec {
        control_rate_hz, ..
    } = node
    else {
        panic!("node 4 is the ActionSpec");
    };
    *control_rate_hz = 50.0;
    f.seal();
    assert_code(&f, codes::XIR_032);
}

// --- Rule 6: Deployment.envelope <-> robot capability -----------------------------------------

#[test]
fn cross_envelope_is_wider_than_the_robot() {
    let mut f = Fixture::new();
    let channel = f
        .task
        .observation_spec
        .channels
        .get_mut("joint_state")
        .unwrap();
    channel.source = ObsSource::JointState {
        body: robot_id(),
        dof: 6,
    };
    f.seal();
    assert_code(&f, codes::DEP_031);
}

// --- Rule 7: Evaluation <-> Task / Observation ------------------------------------------------

#[test]
fn cross_evaluation_references_another_task() {
    let mut f = Fixture::new();
    f.evaluation.task = hex(&[0u8; 32]);
    assert_code(&f, codes::XIR_040);
}

#[test]
fn cross_evaluation_reference_by_path_is_not_checked() {
    let mut f = Fixture::new();
    f.evaluation.task = "tasks/pick_cube.toml".to_owned();
    f.evaluation.observation = "obs/act_224.toml".to_owned();
    assert!(f.diags().is_empty(), "{:#?}", f.diags());
}

#[test]
fn cross_augmentation_would_run_during_evaluation() {
    let mut f = Fixture::new();
    let rgb = f.observation.outputs["rgb_front"].ty.clone();
    f.observation.graph.insert(
        NodeId(9),
        ObservationNode::Augment {
            kind: AugmentKind::ColorJitter {
                brightness: 0.2,
                contrast: 0.2,
                saturation: 0.2,
                hue: 0.05,
            },
            training_only: false,
            io: Io::unary(rgb.clone(), rgb),
        },
    );
    f.seal();
    assert_code(&f, codes::XIR_050);
}

#[test]
fn cross_allow_list_names_an_unknown_node() {
    let mut f = Fixture::new();
    f.evaluation.augmentation = AugmentationPolicy::AllowList {
        nodes: BTreeSet::from(["99".to_owned()]),
        justification: "domain gap study (spec 24.3)".to_owned(),
    };
    f.seal();
    assert_code(&f, codes::XIR_051);
}
