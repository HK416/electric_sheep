//! Oracle for M2 W1 (spec 28.4): the Evaluation IR runs, the report is reproducible, and
//! every refusal this crate promises actually fires.
//!
//! The fixtures are built by hand, the way `es-ir/tests/cross_fixture.rs` does it: borrowing a
//! neighbour's idea of a consistent bundle would make the gate test the neighbour.

use std::collections::{BTreeMap, BTreeSet};

use es_assets::scene::SceneDesc;
use es_compile::Tensor;
use es_core::time::{PhysTick, TickRate};
use es_core::{FailureKind, StableId};
use es_eval::{EvalError, Evaluation, EventSource, FrameSink, LightOverride, RunConfig};
use es_ir::deployment::{
    ActionContract, ActionSpace as DepSpace, Deadlines, DeploymentIr, ExecutionMode,
    FallbackPolicy, Limit, Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope,
    Watchdog, WatchdogSet, Workspace,
};
use es_ir::evaluation::{
    AcceptanceCriterion, AcceptanceResult, Aggregation, AugmentationPolicy, Comparator,
    EpisodeBatch, EvaluationIr, EvaluationReport, MetricSpec, Perturbation, PerturbationKind,
    PerturbationSuite, Range, ReplayPolicy, SeedPlan,
};
use es_ir::graph::{NodeId, PortRef};
use es_ir::learning::LearningGraph;
use es_ir::observation::{
    AugmentKind, Io, NormalizeStats, ObservationIr, ObservationNode, ObservationOutput,
};
use es_ir::task::{
    ActionSpace as TaskSpace, CmpOp, Distribution, JointQuantity, ObsChannel, ObsSource,
    ObservationSpec, SceneRef, TaskConfig, TaskGraph, TaskIr, TaskNode, TerminationKind,
};
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_physics_core::backend::{
    IndexRange, LoadConfig, ModelInfo, PhysicsBackend, PhysicsError, StateView, StepReport,
};
use es_physics_core::caps::{BatchSupport, Capabilities, DeterminismTier, FloatPrecision};
use es_policy::{PolicyError, PolicyInfo, PolicyRuntime, WeightsSource};
use es_safety::ViolationKind;

const NJ: usize = 2;
const H: usize = 2;
const CONTROL_HZ: u64 = 100;
/// Side of the square test frame, in pixels.
const IMG: u32 = 8;
const N_EPISODES: u32 = 6;

// --- Scene, model, backend ----------------------------------------------------------------

const MJCF: &str = r#"<mujoco>
    <worldbody><body name="link">
        <joint name="j0" type="hinge" axis="0 0 1"/>
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <geom name="ball" type="sphere" size="0.1"/>
    </body></worldbody>
    <actuator>
        <motor name="m0" joint="j0"/>
        <motor name="m1" joint="j1"/>
    </actuator>
    <sensor><jointpos name="angle" joint="j0"/></sensor>
</mujoco>"#;

fn scene() -> SceneDesc {
    es_assets::parse_mjcf(MJCF)
        .expect("the fixture parses")
        .scene
}

fn joint_id(name: &str) -> StableId {
    scene()
        .joints
        .iter()
        .find(|j| j.name == name)
        .expect("fixture joint")
        .id
}

fn model() -> ModelInfo {
    let s = scene();
    ModelInfo {
        nq: 2,
        nv: 2,
        nu: 2,
        nsensordata: 1,
        nbody: 1,
        n_envs: 1,
        qpos: [
            (joint_id("j0"), IndexRange::new(0, 1)),
            (joint_id("j1"), IndexRange::new(1, 1)),
        ]
        .into(),
        dof: [
            (joint_id("j0"), IndexRange::new(0, 1)),
            (joint_id("j1"), IndexRange::new(1, 1)),
        ]
        .into(),
        actuator: [
            (s.actuators[0].id, IndexRange::new(0, 1)),
            (s.actuators[1].id, IndexRange::new(1, 1)),
        ]
        .into(),
        sensor: [(s.sensors[0].id, IndexRange::new(0, 1))].into(),
        // A visible time step: the default rate integrates so slowly that nothing moves in a
        // dozen control steps, and every episode would look the same whatever the seed.
        rate: TickRate::hz(CONTROL_HZ),
        ..ModelInfo::default()
    }
}

/// A spring-damper integrator: deterministic, transcendental-free (§3.4).
#[derive(Debug)]
struct FakeBackend {
    /// Actuators beyond the scene's two, so a `nu != NJ` model can be loaded (P-M2-R7).
    nu_extra: u32,
    caps: Capabilities,
    model: Option<ModelInfo>,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    sensordata: Vec<f64>,
    ctrl: Vec<f64>,
    tick: PhysTick,
}

impl FakeBackend {
    /// The same integrator with one actuator more than the deployment declares joints.
    fn wide() -> Self {
        Self {
            nu_extra: 1,
            ..Self::new()
        }
    }

    fn new() -> Self {
        Self {
            nu_extra: 0,
            caps: Capabilities {
                name: "fake".to_owned(),
                determinism: DeterminismTier::Bitwise,
                batch: BatchSupport {
                    max_envs: 16,
                    gpu_resident: false,
                },
                joints: BTreeSet::new(),
                actuators: BTreeSet::new(),
                sensors: BTreeSet::new(),
                contact: BTreeSet::new(),
                float: FloatPrecision::F64,
                supports_reset_subset: true,
                supports_state_get_set: true,
                quirks: Vec::new(),
            },
            model: None,
            qpos: Vec::new(),
            qvel: Vec::new(),
            sensordata: Vec::new(),
            ctrl: Vec::new(),
            tick: PhysTick::ZERO,
        }
    }
}

impl PhysicsBackend for FakeBackend {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn load(&mut self, _scene: &SceneDesc, cfg: &LoadConfig) -> Result<ModelInfo, PhysicsError> {
        let info = ModelInfo {
            n_envs: cfg.n_envs,
            nu: model().nu + self.nu_extra,
            ..model()
        };
        let n = cfg.n_envs as usize;
        self.qpos = vec![0.0; n * info.nq as usize];
        self.qvel = vec![0.0; n * info.nv as usize];
        self.sensordata = vec![0.0; n * info.nsensordata as usize];
        self.ctrl = vec![0.0; n * info.nu as usize];
        self.model = Some(info.clone());
        Ok(info)
    }

    fn model_info(&self) -> Option<&ModelInfo> {
        self.model.as_ref()
    }

    fn reset(
        &mut self,
        envs: Option<&[u32]>,
        state: Option<&StateView<'_>>,
    ) -> Result<(), PhysicsError> {
        let info = self.model.clone().ok_or(PhysicsError::NotLoaded)?;
        let all: Vec<u32> = (0..info.n_envs).collect();
        let list = envs.unwrap_or(&all);
        for (i, env) in list.iter().enumerate() {
            let (nq, nv) = (info.nq as usize, info.nv as usize);
            let (q, v) = (*env as usize * nq, *env as usize * nv);
            self.qpos[q..q + nq].fill(0.0);
            self.qvel[v..v + nv].fill(0.0);
            if let Some(s) = state {
                self.qpos[q..q + nq].copy_from_slice(&s.qpos[i * nq..(i + 1) * nq]);
                self.qvel[v..v + nv].copy_from_slice(&s.qvel[i * nv..(i + 1) * nv]);
            }
            self.sensordata[*env as usize] = self.qpos[q];
        }
        Ok(())
    }

    fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), PhysicsError> {
        if ctrl.len() != self.ctrl.len() {
            return Err(PhysicsError::ShapeMismatch {
                what: "ctrl",
                expected: self.ctrl.len(),
                got: ctrl.len(),
            });
        }
        self.ctrl.copy_from_slice(ctrl);
        Ok(())
    }

    fn step(&mut self, n_substeps: u32) -> Result<StepReport, PhysicsError> {
        let info = self.model.clone().ok_or(PhysicsError::NotLoaded)?;
        let dt = info.rate.period_secs_f64();
        for _ in 0..n_substeps {
            for env in 0..info.n_envs as usize {
                let (q, v) = (env * info.nq as usize, env * info.nv as usize);
                for j in 0..info.nq as usize {
                    let torque = self.ctrl[env * info.nu as usize + j];
                    let accel = torque - 9.81 * self.qpos[q + j] - 0.1 * self.qvel[v + j];
                    self.qvel[v + j] += dt * accel;
                    self.qpos[q + j] += dt * self.qvel[v + j];
                }
                self.sensordata[env] = self.qpos[q];
            }
        }
        self.tick = self.tick.add_ticks(u64::from(n_substeps));
        Ok(StepReport {
            tick: self.tick,
            failures: Vec::<(u32, FailureKind)>::new(),
        })
    }

    fn state(&self) -> StateView<'_> {
        StateView {
            n_envs: self.model.as_ref().map_or(0, |m| m.n_envs),
            tick: self.tick,
            qpos: &self.qpos,
            qvel: &self.qvel,
            sensordata: &self.sensordata,
            ..StateView::default()
        }
    }

    fn set_state(&mut self, state: &StateView<'_>) -> Result<(), PhysicsError> {
        if state.qpos.len() != self.qpos.len() {
            return Err(PhysicsError::ShapeMismatch {
                what: "qpos",
                expected: self.qpos.len(),
                got: state.qpos.len(),
            });
        }
        self.qpos.copy_from_slice(state.qpos);
        Ok(())
    }
}

// --- Policy -------------------------------------------------------------------------------

/// A proportional controller: `target - 0.5 * observation`, repeated over the chunk.
///
/// Proportional rather than constant on purpose. An identical chunk is *the same chunk* to
/// `SafetyPlane::accept`, which keeps consuming the old one and reports a chunk underrun from
/// the third step onward; a real policy emits a different chunk each call.
///
/// `target` outside the envelope is how the envelope-violation metric is exercised without
/// disabling anything (INV-12).
#[derive(Debug)]
struct FakePolicy {
    target: f64,
}

impl PolicyRuntime for FakePolicy {
    fn load(
        &mut self,
        _graph: &LearningGraph,
        _weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError> {
        Err(PolicyError::NotLoaded)
    }

    fn infer(
        &mut self,
        inputs: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>, PolicyError> {
        let observed = inputs
            .values()
            .next()
            .and_then(|t| t.data.get(..4))
            .map_or(0.0, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let action = (self.target as f32) - 0.5 * observed;
        let data: Vec<u8> = (0..H * NJ).flat_map(|_| action.to_le_bytes()).collect();
        Ok(BTreeMap::from([(
            "action".to_owned(),
            Tensor {
                dtype: ElemType::F32,
                shape: vec![H as u64, NJ as u64],
                data,
            },
        )]))
    }

    fn info(&self) -> Option<&PolicyInfo> {
        None
    }

    fn runtime_hash(&self) -> [u8; 32] {
        [7; 32]
    }
}

// --- IR fixtures --------------------------------------------------------------------------

fn scalar_ty(unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([1]),
        unit,
        frame: Frame::World,
        time: TimeRef::Tick,
        image: None,
    }
}

fn joint_ty(unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([1]),
        unit,
        frame: Frame::Joint(joint_id("j0")),
        time: TimeRef::Tick,
        image: None,
    }
}

/// `reward = -angle`, success when `angle > 0.5`, and a randomized initial angle so that two
/// seeds are two different runs.
fn task_ir() -> TaskIr {
    let angle_ty = scalar_ty(Unit::Angle);
    let mut graph = TaskGraph::new(1);
    graph.insert(
        NodeId(0),
        TaskNode::GetJointState {
            body: scene().bodies[0].id,
            joints: vec!["j0".to_owned()],
            quantity: JointQuantity::Position,
        },
    );
    graph.insert(
        NodeId(1),
        TaskNode::Reward {
            name: "upright".to_owned(),
            weight: -1.0,
            aggregation: es_ir::task::Aggregation::Sum,
            ty: angle_ty.clone(),
        },
    );
    graph.insert(
        NodeId(2),
        TaskNode::Compare {
            op: CmpOp::Gt,
            rhs: Some(0.5),
            ty: angle_ty.clone(),
        },
    );
    graph.insert(
        NodeId(3),
        TaskNode::Terminate {
            kind: TerminationKind::Success,
        },
    );
    graph.insert(
        NodeId(4),
        TaskNode::ResetState {
            target: "qpos[0]".to_owned(),
            dist: Distribution::Uniform { lo: -1.0, hi: 1.0 },
            stream: "reset.j0".to_owned(),
        },
    );
    graph.insert(
        NodeId(5),
        TaskNode::ActionSpec {
            space: TaskSpace::JointPosition,
            dim: NJ as u32,
            control_rate_hz: CONTROL_HZ as f32,
        },
    );
    graph.connect(NodeId(0), "value", NodeId(1), "value");
    graph.connect(NodeId(0), "value", NodeId(2), "a");
    graph.connect(NodeId(2), "value", NodeId(3), "value");

    TaskIr {
        schema_version: 1,
        scene: SceneRef {
            path: "fixture.xml".to_owned(),
            scene_hash: [1; 32],
            asset_hash: [2; 32],
        },
        graph,
        observation_spec: ObservationSpec {
            channels: BTreeMap::from([(
                "joint_state".to_owned(),
                ObsChannel {
                    source: ObsSource::JointState {
                        body: scene().bodies[0].id,
                        dof: 1,
                        quantity: JointQuantity::Position,
                    },
                    ty: joint_ty(Unit::Angle),
                },
            )]),
        },
        control: None,
        config: TaskConfig {
            max_episode_steps: 12,
            control_rate_hz: CONTROL_HZ as f32,
            deterministic: true,
            rng_streams: BTreeSet::new(),
        },
    }
}

/// `StateInput(j0) -> Normalize`. No image input: there is no renderer in this build.
/// `augment` appends an `Augment` node with that `training_only` flag.
fn observation_ir(task_ref: [u8; 32], augment: Option<bool>) -> ObservationIr {
    let raw = joint_ty(Unit::Angle);
    let norm = PortType {
        unit: Unit::Normalized { lo: -1.0, hi: 1.0 },
        ..raw.clone()
    };
    let mut ir = ObservationIr::new(1, task_ref);
    ir.graph.insert(
        NodeId(0),
        ObservationNode::StateInput {
            source: joint_id("j0"),
            io: Io::source(raw.clone()),
        },
    );
    ir.graph.insert(
        NodeId(1),
        ObservationNode::Normalize {
            stats: NormalizeStats::Range { lo: -1.0, hi: 1.0 },
            io: Io::unary(raw, norm.clone()),
        },
    );
    ir.graph.connect(NodeId(0), "out", NodeId(1), "in0");
    let mut last = NodeId(1);
    if let Some(training_only) = augment {
        ir.graph.insert(
            NodeId(2),
            ObservationNode::Augment {
                kind: AugmentKind::GaussianNoise { sigma: 0.01 },
                training_only,
                io: Io::unary(norm.clone(), norm.clone()),
            },
        );
        ir.graph.connect(NodeId(1), "out", NodeId(2), "in0");
        last = NodeId(2);
    }
    ir.outputs = BTreeMap::from([(
        "joint_state".to_owned(),
        ObservationOutput {
            port: PortRef::new(last, "out"),
            ty: norm,
        },
    )]);
    ir
}

/// One `ImageInput`, 8x8 `Rgb8`: the port §10.1 refuses without a frame source and serves
/// with one (packet `docs/packets/M5/V0b-render-in-the-loop.md`).
#[allow(clippy::default_trait_access)] // `es-eval` does not depend on `es-math` for `Pose`.
fn image_observation_ir(task_ref: [u8; 32]) -> ObservationIr {
    let camera = StableId::from_path("camera/overhead");
    let ty = PortType {
        elem: ElemType::U8,
        shape: Shape::new([u64::from(IMG), u64::from(IMG), 3]),
        unit: Unit::Pixel,
        frame: Frame::Camera(camera),
        time: TimeRef::Sensor {
            id: camera,
            align: es_ir::types::Align::Hold,
        },
        image: Some(es_ir::image::ImageSpec {
            width: IMG,
            height: IMG,
            channels: es_ir::image::ChannelFormat::Rgb,
            dtype: es_ir::image::ImageDType::U8,
            color_space: es_ir::image::ColorSpace::SRgb,
            camera_model: es_ir::image::CameraModel::Pinhole,
            intrinsics: es_ir::image::Intrinsics::new(4.0, 4.0, 4.0, 4.0),
            extrinsics: Default::default(),
            distortion: es_ir::image::DistortionModel::None,
            shutter: es_ir::image::ShutterModel::Global,
            exposure: std::time::Duration::ZERO,
            rate_hz: CONTROL_HZ as f32,
            depth_scale: None,
        }),
    };
    let mut ir = ObservationIr::new(1, task_ref);
    ir.graph.insert(
        NodeId(0),
        ObservationNode::ImageInput {
            sensor: camera,
            io: Io::source(ty.clone()),
        },
    );
    ir.outputs = BTreeMap::from([(
        "rgb".to_owned(),
        ObservationOutput {
            port: PortRef::new(NodeId(0), "out"),
            ty,
        },
    )]);
    ir
}

/// `StateInput(j0) -> Normalize -> TemporalWindow(n = 2)`. The policy reads the first element
/// of the window, which is the *oldest* frame, so anything left in the ring by a previous
/// episode or cell changes this cell's numbers — the leak P-M2-R1 closes.
fn windowed_observation_ir(task_ref: [u8; 32]) -> ObservationIr {
    let mut ir = observation_ir(task_ref, None);
    let norm = ir.outputs["joint_state"].ty.clone();
    let windowed = PortType {
        shape: Shape::new([2, 1]),
        time: TimeRef::Window {
            base: Box::new(norm.time.clone()),
            n: 2,
            stride: 1,
        },
        ..norm.clone()
    };
    // Layer 1 of spec 7.5: the ring the window reaches back over.
    ir.temporal
        .history
        .insert(joint_id("j0"), es_ir::observation::History { depth: 2 });
    ir.graph.insert(
        NodeId(2),
        ObservationNode::TemporalWindowNode {
            window: es_ir::observation::TemporalWindow {
                n_steps: 2,
                stride: 1,
                align: es_ir::types::Align::Hold,
            },
            io: Io::unary(norm, windowed.clone()),
        },
    );
    ir.graph.connect(NodeId(1), "out", NodeId(2), "in0");
    ir.outputs.insert(
        "joint_state".to_owned(),
        ObservationOutput {
            port: PortRef::new(NodeId(2), "out"),
            ty: windowed,
        },
    );
    ir
}

fn deployment_ir() -> DeploymentIr {
    let rate = RateSpec {
        control: TickRate::hz(CONTROL_HZ),
        inference: TickRate::hz(CONTROL_HZ),
    };
    let period = rate.control_period();
    DeploymentIr {
        schema_version: 1,
        robot: RobotRef {
            name: "fixture".to_owned(),
            target: RobotTarget::Simulated {
                scene: "fixture.xml".to_owned(),
            },
            n_joints: NJ,
        },
        action: ActionContract {
            space: DepSpace::JointPosition,
            dim: NJ,
            horizon: H,
            execute_chunk: H,
        },
        safety: SafetyEnvelope {
            position: vec![Limit::symmetric(2.8); NJ],
            position_soft_margin: vec![0.05; NJ],
            velocity_max: vec![200.0; NJ],
            acceleration_max: vec![8000.0; NJ],
            torque_max: vec![80.0; NJ],
            jerk_max: None,
            action_rate: RateLimit {
                first_diff_max: vec![10.0; NJ],
                second_diff_max: vec![10.0; NJ],
            },
            workspace: Workspace::Box {
                min: [-100.0, -100.0, -100.0],
                max: [100.0, 100.0, 100.0],
            },
            ee_velocity_max: 100.0,
            min_self_distance: 0.001,
            min_env_distance: 0.001,
            contact_force_max: 400.0,
        },
        execution: ExecutionMode::RecedingHorizon,
        deadlines: Deadlines {
            observation_age: Micros(period.0 * 8),
            inference_budget: Micros(period.0 * 8),
            actuation_budget: Micros(period.0 / 2),
        },
        watchdogs: WatchdogSet(vec![Watchdog::ChunkUnderrun]),
        fallback: FallbackPolicy::HoldPosition,
        rate,
    }
}

fn evaluation_ir(
    seed: u64,
    metrics: Vec<MetricSpec>,
    acceptance: Vec<AcceptanceCriterion>,
) -> EvaluationIr {
    EvaluationIr {
        schema_version: 1,
        task: "fixture.task".to_owned(),
        observation: "fixture.obs".to_owned(),
        episodes: EpisodeBatch {
            n_episodes: N_EPISODES,
            seeds: SeedPlan::Base(seed),
        },
        suites: vec![
            PerturbationSuite {
                name: "nominal".to_owned(),
                perturbations: Vec::new(),
            },
            PerturbationSuite {
                name: "actuator_noise".to_owned(),
                perturbations: vec![
                    Perturbation::new(PerturbationKind::TorqueNoise { rel_sigma: 0.05 }, 0),
                    Perturbation::new(
                        PerturbationKind::Backlash {
                            rad: Range::new(0.0, 0.01),
                        },
                        1,
                    ),
                    Perturbation::new(
                        PerturbationKind::FrameDrop {
                            prob: 0.2,
                            burst: es_ir::evaluation::CountRange::new(1, 2),
                        },
                        2,
                    ),
                    Perturbation::new(PerturbationKind::ActionDelay { ms: vec![0, 20] }, 3),
                ],
            },
        ],
        metrics,
        acceptance,
        augmentation: AugmentationPolicy::Disabled,
        replay: ReplayPolicy::None,
    }
}

fn basic_metrics() -> Vec<MetricSpec> {
    vec![
        MetricSpec::SuccessRate,
        MetricSpec::EpisodeLength,
        MetricSpec::EnvelopeViolationRate,
        // Sensitive to the control trace itself, which is what the actuator perturbations and
        // the randomized initial state actually move.
        MetricSpec::ActionSmoothness,
    ]
}

// --- Driver -------------------------------------------------------------------------------

fn run_with(
    ir: &EvaluationIr,
    obs_augment: Option<bool>,
    target: f64,
) -> Result<EvaluationReport, EvalError> {
    let task = task_ir();
    let obs = observation_ir(task.task_hash().expect("task hashes"), obs_augment);
    run_obs(ir, &obs, target)
}

/// The same run over a `TemporalWindow` observation: the plan then carries ring state across
/// `run` calls, which is what P-M2-R1 is about.
fn run_windowed(ir: &EvaluationIr) -> EvaluationReport {
    let task = task_ir();
    let obs = windowed_observation_ir(task.task_hash().expect("task hashes"));
    run_obs(ir, &obs, 0.2).expect("the windowed fixture evaluation runs")
}

fn run_obs(
    ir: &EvaluationIr,
    obs: &ObservationIr,
    target: f64,
) -> Result<EvaluationReport, EvalError> {
    run_obs_frames(ir, obs, target, None, None)
}

fn run_obs_frames(
    ir: &EvaluationIr,
    obs: &ObservationIr,
    target: f64,
    frames: Option<&mut es_eval::runner::FrameSource<'_>>,
    sink: Option<&mut es_eval::FrameSink>,
) -> Result<EvaluationReport, EvalError> {
    run_deploy(ir, obs, target, &deployment_ir(), frames, sink)
}

fn run_deploy(
    ir: &EvaluationIr,
    obs: &ObservationIr,
    target: f64,
    deploy: &DeploymentIr,
    frames: Option<&mut es_eval::runner::FrameSource<'_>>,
    sink: Option<&mut es_eval::FrameSink>,
) -> Result<EvaluationReport, EvalError> {
    let task = task_ir();
    let mut policy = FakePolicy { target };
    Evaluation::run_with_frames::<FakeBackend, _, NJ, H>(
        ir,
        &task,
        &scene(),
        obs,
        &mut policy,
        deploy,
        FakeBackend::new,
        &RunConfig::default(),
        frames,
        sink,
    )
    .map(|(report, _lock)| report)
}

fn run_ok(ir: &EvaluationIr) -> EvaluationReport {
    run_with(ir, None, 0.2).expect("the fixture evaluation runs")
}

// --- Tests --------------------------------------------------------------------------------

/// A `PolicyRuntime` that counts the calls and commands nothing.
///
/// Zero rows still go through the Safety Plane -- counting is not a bypass (`INV-12`).
struct CountingPolicy {
    calls: u64,
}

impl PolicyRuntime for CountingPolicy {
    fn load(
        &mut self,
        _graph: &LearningGraph,
        _weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError> {
        Err(PolicyError::NotLoaded)
    }

    fn infer(
        &mut self,
        _inputs: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>, PolicyError> {
        self.calls += 1;
        Ok(BTreeMap::from([(
            "action".to_owned(),
            Tensor {
                dtype: ElemType::F32,
                shape: vec![H as u64, NJ as u64],
                data: vec![0u8; H * NJ * 4],
            },
        )]))
    }

    fn info(&self) -> Option<&PolicyInfo> {
        None
    }

    fn runtime_hash(&self) -> [u8; 32] {
        [11; 32]
    }
}

/// One `n_episodes = 1` nominal run of `steps` control steps at a declared inference rate,
/// returning the policy calls it took and the `episode_length` the report measured.
fn replan_run(inference: TickRate, steps: u32) -> Result<(u64, f64), EvalError> {
    let mut ir = evaluation_ir(7, basic_metrics(), Vec::new());
    ir.episodes = EpisodeBatch {
        n_episodes: 1,
        seeds: SeedPlan::Explicit(vec![7]),
    };
    ir.suites.truncate(1);
    let deploy = DeploymentIr {
        rate: RateSpec {
            control: TickRate::hz(CONTROL_HZ),
            inference,
        },
        ..deployment_ir()
    };
    // The fixture task stops at 12 steps; the cadence is what is under test, so the budget is
    // raised to hold `steps` and `RunConfig::max_steps` closes the episode. Its
    // `Terminate { Success }` fires on `j0 > 0.5` and its reset draws `j0` from `U(-1, 1)`, so
    // the reset is pinned to zero as well -- the spring-damper decays toward zero under the
    // zero command, and what is counted is then the cadence and not the draw.
    let mut task = task_ir();
    task.config.max_episode_steps = steps;
    task.graph.insert(
        NodeId(4),
        TaskNode::ResetState {
            target: "qpos[0]".to_owned(),
            dist: Distribution::Constant(0.0),
            stream: "reset.j0".to_owned(),
        },
    );
    let obs = observation_ir(task.task_hash().expect("task hashes"), None);
    let mut policy = CountingPolicy { calls: 0 };
    let report = Evaluation::run_with_frames::<FakeBackend, _, NJ, H>(
        &ir,
        &task,
        &scene(),
        &obs,
        &mut policy,
        &deploy,
        FakeBackend::new,
        &RunConfig {
            max_steps: Some(steps),
            ..RunConfig::default()
        },
        None,
        None,
    )
    .map(|(report, _lock)| report)?;
    let length = report
        .cells
        .iter()
        .find(|c| c.metric == MetricSpec::EpisodeLength)
        .and_then(|c| match c.value {
            es_ir::evaluation::MetricValue::Scalar(v) => Some(v),
            _ => None,
        })
        .expect("episode_length is a reported metric");
    Ok((policy.calls, length))
}

/// Packet M5/V17 oracle (a) -- **the evaluator infers at the rate the Deployment IR declares.**
///
/// `rate.control / rate.inference` control ticks pass between two policy calls, and the chunk
/// drives the ticks in between through the `ChunkBuffer`. Until V17 `es_eval::runner` called
/// `infer_chunk` on every control tick whatever the document said, so rows 1.. of every chunk
/// were dead on that path and a policy trained for a 5 Hz re-plan was evaluated at 50 Hz
/// (design note `docs/design/visible-learning.md` section 7.25).
///
/// The fixture's control rate is 100 Hz, so `rate.inference = 10 Hz` is the demo's own 10:1
/// ratio (50 Hz control against 5 Hz inference) and `100 Hz` is one call per control tick.
#[test]
fn the_runner_infers_once_per_declared_replan_period() {
    for (hz, expect) in [(10u64, 10u64), (CONTROL_HZ, 100)] {
        let (calls, length) = replan_run(TickRate::hz(hz), 100).expect("the fixture evaluates");
        assert_eq!(
            length as u32, 100,
            "the episode ran the whole budget at {hz} Hz"
        );
        assert_eq!(
            calls, expect,
            "100 control steps at rate.inference = {hz} Hz against a {CONTROL_HZ} Hz control \
             rate is {expect} policy calls, not {calls}"
        );
    }
}

/// Packet M5/V17 oracle (a), second half -- a re-plan period that is not a whole number of
/// control ticks is **refused by name**, not rounded.
#[test]
fn an_inference_rate_that_does_not_divide_the_control_rate_is_refused() {
    let err = replan_run(TickRate::hz(30), 8).expect_err("100 / 30 is not a cadence");
    let message = err.to_string();
    assert!(matches!(err, EvalError::Env(_)), "{message}");
    for part in ["rate.control", "rate.inference", "3.33"] {
        assert!(message.contains(part), "{part} is not named in: {message}");
    }
}

#[test]
fn the_report_has_one_cell_per_suite_and_metric() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let report = run_ok(&ir);
    assert_eq!(report.cells.len(), 2 * basic_metrics().len());
    for suite in ["nominal", "actuator_noise"] {
        for metric in basic_metrics() {
            assert!(
                report
                    .cells
                    .iter()
                    .any(|c| c.suite == suite && c.metric == metric),
                "missing cell {suite}.{}",
                metric.name()
            );
        }
    }
    assert!(
        report.cells.iter().all(|c| c.n_episodes == N_EPISODES),
        "every episode of the cell must contribute"
    );
}

#[test]
fn two_runs_produce_byte_identical_reports() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let a = serde_json::to_string_pretty(&run_ok(&ir)).expect("serializes");
    let b = serde_json::to_string_pretty(&run_ok(&ir)).expect("serializes");
    assert_eq!(a, b, "spec 10.4: same conditions, same report");
}

#[test]
fn changing_the_seed_changes_at_least_one_metric() {
    let a = run_ok(&evaluation_ir(1, basic_metrics(), Vec::new()));
    let b = run_ok(&evaluation_ir(999, basic_metrics(), Vec::new()));
    assert_ne!(a.cells, b.cells);
}

#[test]
fn artifacts_land_where_the_spec_says() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let task = task_ir();
    let obs = observation_ir(task.task_hash().expect("task hashes"), None);
    let mut policy = FakePolicy { target: 0.2 };
    let (report, lock) = Evaluation::run::<FakeBackend, _, NJ, H>(
        &ir,
        &task,
        &scene(),
        &obs,
        &mut policy,
        &deployment_ir(),
        FakeBackend::new,
        &RunConfig::default(),
    )
    .expect("runs");

    let dir = std::env::temp_dir().join("es-eval-artifacts");
    let _ = std::fs::remove_dir_all(&dir);
    es_eval::write_artifacts(&report, &lock, &dir).expect("writes");
    assert!(dir.join("report.json").is_file());
    assert!(dir.join("evaluation.lock").is_file());
    assert_eq!(lock.evaluation_hash.len(), 64);
    assert_eq!(lock.seeds.len(), N_EPISODES as usize);
    assert_eq!(lock.backend.name, "fake");
    assert!(
        report.episodes.is_empty(),
        "episodes/ replay is a later packet and must not be claimed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unavailable_metric_never_passes_acceptance() {
    let mut metrics = basic_metrics();
    metrics.push(MetricSpec::CollisionRate);
    let acceptance = vec![AcceptanceCriterion {
        suite: Some("nominal".to_owned()),
        metric: MetricSpec::CollisionRate,
        comparator: Comparator::Le,
        threshold: 0.1,
        aggregation: Aggregation::Mean,
    }];
    let report = run_ok(&evaluation_ir(20_260_912, metrics, acceptance));

    let verdict = report
        .acceptance
        .iter()
        .find(|a| match a {
            AcceptanceResult::Unavailable { metric, .. } => *metric == MetricSpec::CollisionRate,
            AcceptanceResult::Determined { criterion, .. } => {
                criterion.metric == MetricSpec::CollisionRate
            }
        })
        .expect("the criterion is judged");
    // `AcceptanceResult::Unavailable` has no `observed` field to fabricate a measurement in.
    assert!(
        matches!(verdict, AcceptanceResult::Unavailable { .. }),
        "an unmeasured metric is Unavailable, not a pass: {verdict:?}"
    );
    assert!(!report.passed, "Unavailable must not pass the run");
    assert!(
        report
            .cells
            .iter()
            .any(|c| c.metric == MetricSpec::CollisionRate
                && matches!(c.value, es_ir::evaluation::MetricValue::Unavailable { .. })),
        "the reason must be recorded on the cell"
    );
}

#[test]
fn an_augment_node_outside_the_allow_list_is_refused() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let err = run_with(&ir, Some(false), 0.2).expect_err("INV-15 refuses the run");
    assert!(
        matches!(err, EvalError::AugmentationEnabled { .. }),
        "expected an INV-15 refusal, got {err}"
    );
}

/// The allow-list is honoured — the INV-15 refusal does not fire for a node named in it —
/// and the run is then refused by the compiler, which has no augmentation kernel. Asserting
/// the exact outcome, because `Ok(_) | Err(Plan(_))` would pass without the allow-list ever
/// being read.
#[test]
fn an_allow_listed_augment_node_gets_past_inv_15_and_dies_in_the_compiler() {
    let mut ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    ir.augmentation = AugmentationPolicy::AllowList {
        nodes: BTreeSet::from(["2".to_owned()]),
        justification: "the fixture measures that the allow-list is honoured".to_owned(),
    };
    match run_with(&ir, Some(false), 0.2) {
        Err(EvalError::Plan(msg)) => assert!(
            msg.contains("augmentation") || msg.contains("Augment"),
            "the compiler must name the node it cannot lower: {msg}"
        ),
        other => panic!("expected a compiler refusal, got {other:?}"),
    }
}

/// INV-15 as §10.4 words it: a `training_only` node is *auto-disabled*, not refused. The plan
/// lowers it to an identity pass-through, so the report must equal the one from the same
/// graph without the node.
#[test]
fn a_training_only_augment_node_is_disabled_not_refused() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let with = run_with(&ir, Some(true), 0.2).expect("a training_only node must not refuse");
    let without = run_with(&ir, None, 0.2).expect("runs");
    assert_eq!(
        with.cells, without.cells,
        "a disabled Augment node must change nothing it touches"
    );
}

/// P-M2-R7. `nu != NJ` is an error at run start, never a broadcast of joint `NJ - 1`.
#[test]
fn a_model_with_more_actuators_than_joints_is_refused() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let task = task_ir();
    let obs = observation_ir(task.task_hash().expect("task hashes"), None);
    let mut policy = FakePolicy { target: 0.2 };
    let err = Evaluation::run::<FakeBackend, _, NJ, H>(
        &ir,
        &task,
        &scene(),
        &obs,
        &mut policy,
        &deployment_ir(),
        FakeBackend::wide,
        &RunConfig::default(),
    )
    .expect_err("a 3-actuator model against NJ = 2 must refuse");
    match err {
        EvalError::JointMismatch { nu, nj, .. } => assert_eq!((nu, nj), (3, NJ)),
        other => panic!("expected JointMismatch, got {other}"),
    }
}

/// P-M2-R1. The plan's `TemporalWindow` rings are episode state: with them cleared per
/// episode, a cell's numbers cannot depend on which suite ran before it.
#[test]
fn reversing_the_suite_order_leaves_every_cell_unchanged() {
    let mut ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    // Two perturbation-free suites: `Perturbation` draws are keyed by the suite's *position*
    // (`EnvRng::new(seed, cell, episode, stream)`), so a perturbed suite is order-dependent by
    // construction and would test that derivation rather than the ring state.
    ir.suites = vec![
        PerturbationSuite {
            name: "nominal_a".to_owned(),
            perturbations: Vec::new(),
        },
        PerturbationSuite {
            name: "nominal_b".to_owned(),
            perturbations: Vec::new(),
        },
    ];
    let forward = run_windowed(&ir);
    for metric in basic_metrics() {
        let of = |suite: &str| {
            forward
                .cells
                .iter()
                .find(|c| c.suite == suite && c.metric == metric)
                .map(|c| c.value.clone())
                .expect("the cell is measured")
        };
        assert_eq!(
            of("nominal_a"),
            of("nominal_b"),
            "two identical suites must measure the same: {}",
            metric.name()
        );
    }

    ir.suites.reverse();
    let reversed = run_windowed(&ir);
    for cell in &forward.cells {
        let other = reversed
            .cells
            .iter()
            .find(|c| c.suite == cell.suite && c.metric == cell.metric)
            .expect("the same cells, in the other order");
        assert_eq!(
            cell,
            other,
            "§10.4: {}.{} must not depend on suite order",
            cell.suite,
            cell.metric.name()
        );
    }
}

#[test]
fn an_unsupported_perturbation_kind_is_named_not_skipped() {
    let mut ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    ir.suites.push(PerturbationSuite {
        name: "lighting_shift".to_owned(),
        perturbations: vec![Perturbation::new(
            PerturbationKind::LightIntensity {
                range: Range::new(0.3, 2.5),
                dist: es_ir::evaluation::Distribution::LogUniform,
            },
            0,
        )],
    });
    let err = run_with(&ir, None, 0.2).expect_err("an unrealisable kind refuses the run");
    match err {
        EvalError::Unsupported { kind, .. } => assert_eq!(kind, "light_intensity"),
        other => panic!("expected Unsupported, got {other}"),
    }
}

#[test]
fn the_envelope_violation_rate_rises_when_the_policy_leaves_the_envelope() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let clean = rate_of(&run_with(&ir, None, 0.2).expect("runs"));
    let wild = rate_of(&run_with(&ir, None, 100.0).expect("runs"));
    assert!(
        wild > 0.0,
        "commanding 100 rad against a 2.8 rad limit must show up as a violation"
    );
    assert!(
        wild > clean,
        "a policy outside the envelope must score worse than one inside: {wild} vs {clean}"
    );
}

fn rate_of(report: &EvaluationReport) -> f64 {
    report
        .cells
        .iter()
        .find(|c| c.suite == "nominal" && c.metric == MetricSpec::EnvelopeViolationRate)
        .and_then(|c| match c.value {
            es_ir::evaluation::MetricValue::Scalar(v) => Some(v),
            es_ir::evaluation::MetricValue::Histogram(_)
            | es_ir::evaluation::MetricValue::Unavailable { .. } => None,
        })
        .expect("the metric is measured")
}

/// The perturbed suite must not be a copy of `nominal`: a kernel that compiled but did
/// nothing would otherwise pass every other test in this file.
#[test]
fn a_perturbed_suite_differs_from_nominal() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let report = run_ok(&ir);
    let of = |suite: &str| -> Vec<es_ir::evaluation::MetricValue> {
        report
            .cells
            .iter()
            .filter(|c| c.suite == suite)
            .map(|c| c.value.clone())
            .collect()
    };
    assert_ne!(of("nominal"), of("actuator_noise"));
}

// --- the image observation port (packet M5/V0b) ------------------------------------------

/// §10.1: an image input the runner cannot serve is refused, with the same message as before
/// V0b. Nothing is ever zero-filled — a wrong number in the §10.1 table is worse than no row.
#[test]
fn without_a_renderer_an_image_input_is_still_refused() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let task = task_ir();
    let obs = image_observation_ir(task.task_hash().expect("task hashes"));
    let err = run_obs(&ir, &obs, 0.2).expect_err("an image input with no frame source");
    let EvalError::Plan(message) = &err else {
        panic!("expected EvalError::Plan, got {err}");
    };
    assert!(
        message.contains("image inputs need a renderer, which this build has none of"),
        "{message}"
    );
}

/// With a frame source (`es_env::render::EnvRenderer::frame` in a real run) the same input is
/// served, and the bytes reach the plan unchanged.
#[test]
fn a_frame_source_serves_the_image_input() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let task = task_ir();
    let obs = image_observation_ir(task.task_hash().expect("task hashes"));
    let mut calls = 0u32;
    let mut frames = |_: &LightOverride, _: &ModelInfo, _: &StateView<'_>| {
        calls += 1;
        Ok(vec![0x5a_u8; IMG as usize * IMG as usize * 3])
    };
    let report = run_obs_frames(&ir, &obs, 0.2, Some(&mut frames), None).expect("the image run");
    assert!(!report.cells.is_empty());
    assert!(calls > 0, "the frame source was never asked for a frame");
}

/// A frame that is not exactly what the plan declared is an error naming both sizes: `es-eval`
/// does not resample, pad or convert to make one fit (§7.2, `INV-14`).
#[test]
fn a_frame_of_the_wrong_size_is_refused_not_resized() {
    let ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    let task = task_ir();
    let obs = image_observation_ir(task.task_hash().expect("task hashes"));
    let mut frames = |_: &LightOverride, _: &ModelInfo, _: &StateView<'_>| Ok(vec![0_u8; 4]);
    let err = run_obs_frames(&ir, &obs, 0.2, Some(&mut frames), None).expect_err("a short frame");
    let EvalError::Plan(message) = &err else {
        panic!("expected EvalError::Plan, got {err}");
    };
    assert!(message.contains("the frame supplies 4 bytes"), "{message}");

    // A frame source that cannot render says so, and the reason survives.
    let mut broken = |_: &LightOverride, _: &ModelInfo, _: &StateView<'_>| {
        Err("no camera in the scene".to_owned())
    };
    let err = run_obs_frames(&ir, &obs, 0.2, Some(&mut broken), None).expect_err("a broken source");
    assert!(format!("{err}").contains("no camera in the scene"), "{err}");
}

// --- packet M5/V3: the frame sink, the event stream and the two lighting kernels -------------

/// A scratch directory of this test binary's own, emptied first so a rerun starts clean.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("es-eval-v3").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A frame source whose bytes are a pure function of the state and the lighting, so two runs
/// of the same conditions agree.
fn state_frames(
) -> impl FnMut(&LightOverride, &ModelInfo, &StateView<'_>) -> Result<Vec<u8>, String> {
    |light: &LightOverride, _: &ModelInfo, state: &StateView<'_>| {
        let q = state.qpos_of(0)[0] * light.intensity;
        let mut out = vec![0_u8; IMG as usize * IMG as usize * 3];
        // `FakePolicy::infer` reads the first four bytes of its first input as an `f32`, so
        // they carry the joint angle; the rest is a flat fill derived from the same number.
        out[..4].copy_from_slice(&(q as f32).to_le_bytes());
        out[4..].fill(((q.abs() * 100.0) as u32 % 251) as u8);
        Ok(out)
    }
}

/// An evaluation with one suite, so `suites x episodes` is just the episode count.
fn one_suite(perturbations: Vec<Perturbation>) -> EvaluationIr {
    let mut ir = evaluation_ir(20_260_912, basic_metrics(), Vec::new());
    ir.suites = vec![PerturbationSuite {
        name: "nominal".to_owned(),
        perturbations,
    }];
    ir
}

fn image_ir() -> (EvaluationIr, ObservationIr) {
    let task = task_ir();
    (
        one_suite(Vec::new()),
        image_observation_ir(task.task_hash().expect("task hashes")),
    )
}

/// The grid is one cell per **episode**, not per suite: `Evaluation::run` hardcodes
/// `BatchDomains::single_env()`, so sixteen demo episodes are sixteen independent runs and
/// sixteen directories for `es video mosaic` to tile.
#[test]
fn every_episode_of_every_suite_gets_its_own_frame_dir() {
    let (ir, obs) = image_ir();
    let dir = scratch("cells");
    let mut sink = FrameSink::new(&dir);
    let mut frames = state_frames();
    run_obs_frames(&ir, &obs, 0.2, Some(&mut frames), Some(&mut sink)).expect("the image run");

    let cells = ir.suites.len() * N_EPISODES as usize;
    assert_eq!(sink.events.len(), cells);
    let mut layouts = BTreeSet::new();
    for name in sink.events.keys() {
        let cell = dir.join(name);
        let layout = std::fs::read_to_string(cell.join("layout.json")).expect("layout.json");
        assert_eq!(
            layout.trim(),
            format!("{{\"dtype\":\"u8\",\"shape\":[{IMG}, {IMG}, 3]}}"),
            "{name}"
        );
        layouts.insert(layout);
        assert!(cell.join("000000.bin").is_file(), "{name} has no frame 0");
    }
    assert_eq!(layouts.len(), 1, "the cells must be mosaic-able together");
}

/// `events.json` describes the frames on disk and nothing else: one record per frame, dense
/// and ascending, each carrying the `PhysTick` of the step it was captured for.
#[test]
fn events_json_has_one_record_per_frame() {
    let (ir, obs) = image_ir();
    let dir = scratch("events");
    let mut sink = FrameSink::new(&dir);
    let mut frames = state_frames();
    run_obs_frames(&ir, &obs, 0.2, Some(&mut frames), Some(&mut sink)).expect("the image run");

    for (name, records) in &sink.events {
        let on_disk = std::fs::read_dir(dir.join(name))
            .expect("the cell directory")
            .filter(|e| {
                e.as_ref()
                    .expect("entry")
                    .path()
                    .extension()
                    .is_some_and(|x| x == "bin")
            })
            .count();
        assert_eq!(records.len(), on_disk, "{name}");
        assert!(!records.is_empty(), "{name} rendered nothing");
        for (i, r) in records.iter().enumerate() {
            assert_eq!(r.frame, i as u64, "{name}");
        }
        // The tick is the step's, so it advances with the episode and never repeats.
        let ticks: Vec<u64> = records.iter().map(|r| r.tick.0).collect();
        assert!(ticks.windows(2).all(|w| w[0] < w[1]), "{name}: {ticks:?}");
    }

    let path = dir.join("events.json");
    sink.write_events(&path).expect("events.json");
    let text = std::fs::read_to_string(&path).expect("read back");
    // The spelling `es video mosaic` reads (`crates/es/src/cmd/video.rs`).
    assert!(text.contains("\"source\": \"Policy\""), "{text:.400}");
    assert!(text.contains("\"frame\": 0"), "{text:.400}");
}

/// Packet M7/T7 oracle 1 -- **tick 0 of every episode is a chunk underrun under a declared
/// inference latency**, because the chunk computed from tick 0's observation is not executable
/// until tick `latency`.
///
/// `es loop collect` has modelled this since M2: `DomainRunner` submits at tick `t` and the
/// `ChunkBuffer` releases at `t + latency_ticks(expected_latency_ms, rate.control)`, so the
/// plane sees an empty buffer on tick 0 and answers with its own fallback. `es_eval::runner`
/// had no latency model and executed row 0 at tick 0, which is what made the two paths'
/// trajectories for one seed diverge at tick 1 (design note `docs/design/visible-learning.md`
/// open question 24, and `es`'s `collection_and_evaluation_draw_the_same_trajectory`).
///
/// Nothing here widens or bypasses the plane (INV-12): the underrun is the plane's own event,
/// counted and recorded exactly as it is in collection. The fixture's `WatchdogSet` already
/// arms `chunk_underrun`, and no envelope and no fixture's `expected_latency_ms` moves -- the
/// run states the document's number at the call.
#[test]
fn tick_zero_is_a_chunk_underrun_under_a_declared_latency() {
    /// One control period in milliseconds: `latency_ticks` rounds it to exactly one tick.
    const ONE_PERIOD_MS: f32 = 1000.0 / CONTROL_HZ as f32;
    const UNDERRUN: u32 = 1 << (ViolationKind::ChunkUnderrun as u32);

    let (ir, obs) = image_ir();
    for (ms, under) in [(0.0_f32, false), (ONE_PERIOD_MS, true)] {
        let dir = scratch(if under { "underrun-1" } else { "underrun-0" });
        let mut sink = FrameSink::new(&dir);
        let mut frames = state_frames();
        let mut policy = FakePolicy { target: 0.2 };
        Evaluation::run_with_frames::<FakeBackend, _, NJ, H>(
            &ir,
            &task_ir(),
            &scene(),
            &obs,
            &mut policy,
            &deployment_ir(),
            FakeBackend::new,
            &RunConfig {
                expected_latency_ms: ms,
                ..RunConfig::default()
            },
            Some(&mut frames),
            Some(&mut sink),
        )
        .expect("the fixture evaluates under a declared latency");

        assert_eq!(
            sink.events.len(),
            N_EPISODES as usize,
            "one cell per episode"
        );
        for (name, records) in &sink.events {
            // Frame 0 of the cell, which is step 0 of the episode. Not `tick == 0`: the cell's
            // `Env` is not rebuilt between episodes, so its `PhysTick` keeps counting.
            let first = records.first().expect("every cell rendered frame 0");
            assert_eq!(
                first.frame, 0,
                "{name}: the first record is the first frame"
            );
            assert_eq!(
                first.events & UNDERRUN != 0,
                under,
                "{name}: tick 0 at expected_latency_ms = {ms} -- events {:#x}, source {:?}",
                first.events,
                first.source
            );
            if under {
                assert_ne!(
                    first.source,
                    EventSource::Policy,
                    "{name}: an underrun tick is driven by the plane's fallback, not the policy"
                );
                // And the very next tick is the chunk landing: this is one tick of latency,
                // not a policy that never produced. (An episode of the fixture can terminate
                // on its first step, in which case there is no next tick to check.)
                if let Some(second) = records.get(1) {
                    assert_eq!(second.events & UNDERRUN, 0, "{name}: tick 1 is served");
                }
            } else {
                // `Policy` or `Clamped` -- either way a row was served and judged. Only
                // `Fallback` means the plane had nothing, which is the case under test.
                assert_ne!(
                    first.source,
                    EventSource::Fallback,
                    "{name}: with no declared latency tick 0 executes row 0"
                );
            }
        }
    }
}

/// The overlay reads the Safety Plane, not the policy: a tightened envelope clamps the same
/// trajectory a wide one passes through untouched. Nothing here disables the plane (INV-12) --
/// both runs validate every step, and only the limits differ.
#[test]
fn a_tightened_envelope_clamps_and_a_widened_one_does_not() {
    let (ir, obs) = image_ir();

    let mut wide = deployment_ir();
    // Widening, never disabling (INV-12): every step is still validated and still counted.
    wide.safety.position = vec![Limit::symmetric(1e6); NJ];
    wide.safety.velocity_max = vec![1e6; NJ];
    wide.safety.acceleration_max = vec![1e9; NJ];
    wide.safety.torque_max = vec![1e6; NJ];
    wide.safety.action_rate = RateLimit {
        first_diff_max: vec![1e6; NJ],
        second_diff_max: vec![1e6; NJ],
    };
    let dir = scratch("wide");
    let mut sink = FrameSink::new(&dir);
    let mut frames = state_frames();
    run_deploy(&ir, &obs, 0.2, &wide, Some(&mut frames), Some(&mut sink)).expect("the wide run");
    let dirty: usize = sink
        .events
        .values()
        .flatten()
        .filter(|e| e.source != EventSource::Policy)
        .count();
    assert_eq!(dirty, 0, "a wide envelope must not clamp");
    assert!(sink.events.values().flatten().all(|e| e.events == 0));

    let mut tight = deployment_ir();
    tight.safety.action_rate.first_diff_max = vec![1e-4; NJ];
    let tight_dir = scratch("tight");
    let mut tight_sink = FrameSink::new(&tight_dir);
    let mut frames = state_frames();
    run_deploy(
        &ir,
        &obs,
        0.2,
        &tight,
        Some(&mut frames),
        Some(&mut tight_sink),
    )
    .expect("the tight run");
    let clamped: Vec<_> = tight_sink
        .events
        .values()
        .flatten()
        .filter(|e| e.source == EventSource::Clamped)
        .collect();
    assert!(!clamped.is_empty(), "a tightened envelope must clamp");
    assert!(
        clamped.iter().all(|e| e.events != 0),
        "a clamped step records which limit it hit"
    );
}

/// INV-12: no path in this crate skips `SafetyPlane::validate`, and there is no flag, `cfg` or
/// test hook that would. A source scan, because the property is about code that does not exist.
#[test]
fn the_plane_is_never_disabled() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut validates = 0usize;
    for entry in std::fs::read_dir(&src).expect("src/") {
        let path = entry.expect("entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).expect("source");
            validates += text.matches("safety.validate(").count();
            for banned in [
                "skip_safety",
                "disable_safety",
                "no_safety",
                "unchecked_action",
            ] {
                assert!(!text.contains(banned), "{}: {banned}", path.display());
            }
        }
    }
    assert_eq!(validates, 1, "exactly one call site, in the step loop");
}

/// Packet M5/V6b (b): `--seed S` must name the same randomization draw here as it does in
/// `es loop collect`, and the two paths only agree if neither resets before its first
/// observation.
///
/// `Env::new` already resets once — that is randomization draw 0 of `(seed, env, episode)`
/// (§6.3) — and every episode ends with a reset, `Env::step`'s own on a terminal condition or
/// the explicit one when the step budget runs out. `run_episode` used to reset *again* at the
/// top, so evaluation's episode `i` ran on draw `2i + 1` while collection's ran on draw `i`:
/// the same `--seed 1` put the cube somewhere else on the two paths, and every A/B between a
/// collected demonstration and an evaluated episode was comparing two different scenes.
///
/// The ground truth is built here, from `Env` directly, rather than transcribed.
#[test]
fn episode_zero_runs_on_the_first_randomization_draw() {
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Any seed; the point is that both paths resolve it to the same draw.
    const SEED: u64 = 20_260_915;

    let task = task_ir();
    let env = es_env::Env::new(
        &task,
        &scene(),
        FakeBackend::new(),
        &es_env::scheduler::BatchDomains::single_env(),
        SEED,
    )
    .expect("the fixture env builds");
    let draw0 = env.backend().state().qpos_of(0)[0];
    drop(env);

    let seen: Rc<RefCell<Vec<f64>>> = Rc::new(RefCell::new(Vec::new()));
    let taken = Rc::clone(&seen);
    let mut frames = move |light: &LightOverride,
                           model: &ModelInfo,
                           state: &StateView<'_>|
          -> Result<Vec<u8>, String> {
        taken.borrow_mut().push(state.qpos_of(0)[0]);
        state_frames()(light, model, state)
    };

    let (mut ir, obs) = image_ir();
    ir.episodes.n_episodes = 1;
    ir.episodes.seeds = SeedPlan::Explicit(vec![SEED]);
    run_obs_frames(&ir, &obs, 0.2, Some(&mut frames), None).expect("the fixture evaluation runs");

    let seen = seen.borrow();
    let first = *seen
        .first()
        .expect("the frame source served at least one step");
    assert_eq!(
        first.to_bits(),
        draw0.to_bits(),
        "episode 0 ran on a different randomization draw than `Env::new` produced: {first} vs          {draw0}. A second reset here is what made `--seed S` name one scene in `es loop          collect` and another in `es eval run` (packet M5/V6b)."
    );
    println!(
        "RAN episode_zero_runs_on_the_first_randomization_draw: draw 0 = {draw0}, {} steps          observed",
        seen.len()
    );
}

/// Packet M5/V6b (a): the chunk a policy returns drives `action.execute_chunk` control ticks
/// (or the whole overlap under `TemporalEnsemble`), because this crate feeds the plane through
/// `es_env::plane_chunk` — the same function `DomainRunner::emit_actions` calls, which is what
/// `es loop collect` drives.
///
/// Before V6b the loop handed the plane each raw inference result under a fresh `seq`, so the
/// plane's cursor reset every tick, **only row 0 of every chunk ever executed**, and the
/// Deployment IR's `action.execute_chunk` and `execution` were dead here. A source scan, in
/// the style of the two beside it: the property is about which call site exists.
#[test]
fn the_runner_feeds_the_plane_through_the_collectors_chunk_buffer() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runner.rs");
    let text = std::fs::read_to_string(&path).expect("src/runner.rs");
    assert_eq!(
        text.matches("plane_chunk(buffer, feed,").count(),
        1,
        "one feed, in the step loop, and it is `es_env`'s"
    );
    assert_eq!(
        text.matches("buffer.push(&chunk,").count(),
        1,
        "every inference result goes into the buffer, never straight to the plane"
    );
    assert_eq!(
        text.matches("feed.end_episode(buffer)").count(),
        1,
        "one episode boundary, the same call `DomainRunner::reset_env` makes"
    );
    assert!(
        !text.contains("env.reset(None)?;\n    // An episode is where"),
        "the episode-start reset is what made evaluation draw ahead of collection"
    );
    assert_eq!(
        text.matches("env.reset(None)").count(),
        1,
        "one reset, at the bottom, closing an episode whose step budget ran out"
    );
}

/// Packet M5/V6: this crate's half of the one envelope semantics. The runner observes before
/// **every** `validate` and opens every episode with `begin_episode`, exactly as
/// `es_data::Collector` does -- what those two calls *mean* is `SafetyPlane`'s to decide
/// (`crates/es-safety/tests/envelope_reference.rs`), and neither consumer may decide it
/// locally again. A source scan, because the property is about call discipline rather than
/// about a value this crate can read back.
#[test]
fn the_runner_seeds_the_plane_once_per_episode_and_observes_every_step() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runner.rs");
    let text = std::fs::read_to_string(&path).expect("src/runner.rs");
    assert_eq!(
        text.matches("safety.observe_state(").count(),
        1,
        "one observation, in the step loop, before the one `safety.validate(`"
    );
    assert_eq!(
        text.matches("safety.begin_episode()").count(),
        1,
        "one episode boundary, at the top of `run_episode`"
    );
    assert!(
        !text.contains("safety.reset_latch()"),
        "`begin_episode` is the episode boundary: clearing only the latch leaves the command          chain anchored on the previous episode's last command (packet M5/V1c, M5/V6)"
    );
    let observe = text.find("safety.observe_state(").expect("the observation");
    let validate = text.find("safety.validate(").expect("the validation");
    assert!(observe < validate, "the observation comes first");
}

/// The two lighting kernels are arithmetic on the scene and on the light direction, so they
/// are checkable without a device: the gain multiplies every colour, the yaw turns the
/// direction about `+Z`, and both are the identity when nothing was drawn.
#[test]
fn the_light_kernels_scale_the_scene_and_turn_the_light() {
    let base = scene();
    let identity = LightOverride::default();
    assert!(identity.is_identity());
    assert_eq!(
        identity.rotate_dir([0.3, 0.4, 0.8]).map(f64::to_bits),
        [0.3_f64, 0.4, 0.8].map(f64::to_bits),
    );
    let rgba = |s: &SceneDesc| -> Vec<[f64; 4]> {
        s.bodies
            .iter()
            .flat_map(|b| &b.geoms)
            .map(|g| g.rgba)
            .collect()
    };
    assert!(!rgba(&base).is_empty(), "the fixture scene has a geom");
    assert_eq!(rgba(&identity.scene(&base)), rgba(&base));

    let dim = LightOverride {
        intensity: 0.25,
        yaw_deg: 0.0,
    };
    let dimmed = dim.scene(&base);
    for (a, b) in dimmed.bodies.iter().zip(&base.bodies) {
        for (g, h) in a.geoms.iter().zip(&b.geoms) {
            for c in 0..3 {
                assert!((g.rgba[c] - h.rgba[c] * 0.25).abs() < 1e-12);
            }
            // Alpha is not radiance.
            assert_eq!(g.rgba[3].to_bits(), h.rgba[3].to_bits());
        }
    }

    let turned = LightOverride {
        intensity: 1.0,
        yaw_deg: 90.0,
    }
    .rotate_dir([1.0, 0.0, 0.5]);
    assert!(turned[0].abs() < 1e-6, "{turned:?}");
    assert!((turned[1] - 1.0).abs() < 1e-6, "{turned:?}");
    assert_eq!(
        turned[2].to_bits(),
        0.5_f64.to_bits(),
        "a yaw does not change the elevation"
    );
}

/// With a frame source the two lighting kinds run and reach the frames; without one they are
/// refused by name, never drawn and dropped (§17.2).
#[test]
fn the_light_kinds_need_a_frame_source() {
    let light = vec![
        Perturbation::new(
            PerturbationKind::LightIntensity {
                range: Range::new(0.2, 0.4),
                dist: es_ir::evaluation::Distribution::Uniform,
            },
            0,
        ),
        Perturbation::new(PerturbationKind::LightDirection { range_deg: 30.0 }, 1),
    ];
    let ir = one_suite(light);
    let task = task_ir();
    let obs = image_observation_ir(task.task_hash().expect("task hashes"));

    let err = run_obs_frames(&ir, &obs, 0.2, None, None).expect_err("no frame source");
    let EvalError::Unsupported { kind, reason } = &err else {
        panic!("expected EvalError::Unsupported, got {err}");
    };
    assert_eq!(*kind, "light_intensity");
    assert!(reason.contains("no frame source"), "{reason}");

    // With one, the draw reaches the frame source and is not the identity.
    let mut seen: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut frames = |light: &LightOverride, _: &ModelInfo, _: &StateView<'_>| {
        seen.insert((light.intensity.to_bits(), light.yaw_deg.to_bits()));
        Ok(vec![0x5a_u8; IMG as usize * IMG as usize * 3])
    };
    run_obs_frames(&ir, &obs, 0.2, Some(&mut frames), None).expect("the lit run");
    assert_eq!(
        seen.len(),
        N_EPISODES as usize,
        "each episode draws its own lighting"
    );
    assert!(
        !seen.contains(&(1.0_f64.to_bits(), 0.0_f64.to_bits())),
        "a declared light perturbation must not leave the scene as authored"
    );
}

/// The kinds this build still cannot realise are refused by name, each with a reason that is
/// true *after* the two lighting kernels landed.
#[test]
fn the_remaining_kinds_are_still_unsupported_by_name() {
    let cases = [
        (
            PerturbationKind::ColorTemperature {
                range_k: Range::new(3000.0, 6500.0),
            },
            "color_temperature",
            "light colour",
        ),
        (
            PerturbationKind::Occluder {
                count: es_ir::evaluation::CountRange::new(1, 2),
                size_m: Range::new(0.01, 0.05),
            },
            "occluder",
            "not in the scene",
        ),
        (
            PerturbationKind::CameraExtrinsic {
                pos_sigma_m: 0.01,
                rot_sigma_deg: 1.0,
            },
            "camera_extrinsic",
            "INV-14",
        ),
        (
            PerturbationKind::CameraIntrinsic {
                focal_rel_sigma: 0.01,
            },
            "camera_intrinsic",
            "INV-14",
        ),
        (
            PerturbationKind::ObjectPose {
                target: "cube".to_owned(),
                pos_sigma_m: 0.01,
                yaw_deg: 10.0,
            },
            "object_pose",
            "Env::reset takes no state override",
        ),
    ];
    let task = task_ir();
    let obs = image_observation_ir(task.task_hash().expect("task hashes"));
    for (kind, name, because) in cases {
        let ir = one_suite(vec![Perturbation::new(kind, 0)]);
        // Even *with* a frame source: a renderer is not what these are blocked on.
        let mut frames = state_frames();
        let err =
            run_obs_frames(&ir, &obs, 0.2, Some(&mut frames), None).expect_err("still unsupported");
        let EvalError::Unsupported { kind, reason } = &err else {
            panic!("expected EvalError::Unsupported for {name}, got {err}");
        };
        assert_eq!(*kind, name);
        assert!(reason.contains(because), "{name}: {reason}");
    }
}

/// §10.4 fairness: the lighting kinds draw from their own streams, so adding one to a suite
/// cannot move any other stream's cursor. If this fails, every report ever written moves.
#[test]
fn enabling_the_light_kinds_does_not_move_other_streams() {
    let noisy = vec![
        Perturbation::new(PerturbationKind::TorqueNoise { rel_sigma: 0.05 }, 0),
        Perturbation::new(
            PerturbationKind::Backlash {
                rad: Range::new(0.0, 0.01),
            },
            1,
        ),
    ];
    let task = task_ir();
    let obs = image_observation_ir(task.task_hash().expect("task hashes"));
    let mut frames = state_frames();
    let without = run_obs_frames(
        &one_suite(noisy.clone()),
        &obs,
        0.2,
        Some(&mut frames),
        None,
    )
    .expect("the run without lighting");

    let mut lit = noisy;
    lit.push(Perturbation::new(
        PerturbationKind::LightDirection { range_deg: 30.0 },
        2,
    ));
    let mut frames = state_frames();
    let with = run_obs_frames(&one_suite(lit), &obs, 0.2, Some(&mut frames), None)
        .expect("the run with lighting");

    assert_eq!(
        serde_json::to_string(&without.cells).expect("cells"),
        serde_json::to_string(&with.cells).expect("cells"),
        "a lighting draw moved another stream"
    );
}

/// Design note section 9: same conditions, byte-identical frames on the CPU path. The frame
/// source here is a pure function of the state, so this is the runner's half of that claim --
/// the ordering, the drop handling and the file names.
#[test]
fn the_frames_are_byte_identical_across_runs() {
    let (ir, obs) = image_ir();
    let mut runs = Vec::new();
    for run in 0..2 {
        let dir = scratch(&format!("repeat{run}"));
        let mut sink = FrameSink::new(&dir);
        let mut frames = state_frames();
        run_obs_frames(&ir, &obs, 0.2, Some(&mut frames), Some(&mut sink)).expect("the run");
        let mut all = Vec::new();
        for name in sink.events.keys() {
            for entry in std::fs::read_dir(dir.join(name)).expect("cell") {
                let path = entry.expect("entry").path();
                all.push((
                    path.file_name()
                        .expect("name")
                        .to_string_lossy()
                        .into_owned(),
                    std::fs::read(&path).expect("frame"),
                ));
            }
        }
        all.sort();
        runs.push((all, sink.events));
    }
    assert_eq!(runs[0].0, runs[1].0, "the frames moved between runs");
    assert_eq!(runs[0].1, runs[1].1, "the events moved between runs");
}

// --- packet M5/V2b: the training set goes through the same executor -------------------------

/// The privileged channel's name in this fixture (packet M5/V7a).
const SIM_J1: &str = "sim_j1";

/// [`task_ir`] plus the privileged channel the branch above implements. Separate rather than
/// folded into `task_ir` so every other test in this file keeps the `task_hash` it had.
///
/// The channel names the **joint** `j1`, not a body: `input_sources` resolves a source id
/// against `ModelInfo::qpos` before it falls back to the channel's leading-`dof` reading, and
/// that is what serves the port `qpos[1..2]` on both the inference and the bake path.
fn privileged_task_ir() -> TaskIr {
    let mut task = task_ir();
    task.observation_spec.channels.insert(
        SIM_J1.to_owned(),
        es_ir::task::ObsChannel {
            source: ObsSource::JointState {
                body: joint_id("j1"),
                dof: 1,
                quantity: JointQuantity::Position,
            },
            ty: PortType {
                frame: Frame::Joint(joint_id("j1")),
                ..joint_ty(Unit::Angle)
            },
        },
    );
    task
}

/// A demo-shaped Observation IR: `StateInput -> Normalize{−1..1}` **and**
/// `ImageInput -> Dequantize -> Normalize{0..1}`, which is the shape of
/// `tests/fixtures/visible-learning/observation.toml`. The state input names the **body** the
/// Task IR's `ObservationSpec` declares, not a joint, so it resolves through
/// `ObsSource::JointState { body, dof }` — the one state reading a recorded dataset can feed.
fn baked_observation_ir(task_ref: [u8; 32]) -> ObservationIr {
    let body = scene().bodies[0].id;
    let raw = PortType {
        frame: Frame::Joint(body),
        ..joint_ty(Unit::Angle)
    };
    let norm = PortType {
        unit: Unit::Normalized { lo: -1.0, hi: 1.0 },
        ..raw.clone()
    };
    let mut ir = image_observation_ir(task_ref);
    let u8_image = ir.graph.nodes[&NodeId(0)].io().output.clone();
    let f32_image = PortType {
        elem: ElemType::F32,
        shape: Shape::new([3, u64::from(IMG), u64::from(IMG)]),
        image: u8_image.image.map(|mut i| {
            i.dtype = es_ir::image::ImageDType::F32;
            i
        }),
        ..u8_image.clone()
    };
    let scaled = PortType {
        unit: Unit::Normalized { lo: 0.0, hi: 1.0 },
        ..f32_image.clone()
    };
    ir.graph.insert(
        NodeId(1),
        ObservationNode::Dequantize {
            io: Io::unary(u8_image, f32_image.clone()),
        },
    );
    ir.graph.insert(
        NodeId(2),
        ObservationNode::Normalize {
            stats: NormalizeStats::Range { lo: 0.0, hi: 1.0 },
            io: Io::unary(f32_image, scaled.clone()),
        },
    );
    ir.graph.insert(
        NodeId(3),
        ObservationNode::StateInput {
            source: body,
            io: Io::source(raw.clone()),
        },
    );
    ir.graph.insert(
        NodeId(4),
        ObservationNode::Normalize {
            stats: NormalizeStats::Range { lo: -1.0, hi: 1.0 },
            io: Io::unary(raw, norm.clone()),
        },
    );
    // The privileged branch (packet M5/V7a), and the reason it is `j1` and not `j0`: `j1`'s
    // `qpos` range starts at 1, so a bake that read "the leading `dof` of the recorded row"
    // would serve `q[0]` here. The demo's own privileged channel is the cube's free joint at
    // `qpos[6..13]`; this is the same shape of reading at the smallest size that has it.
    let raw_j1 = PortType {
        frame: Frame::Joint(joint_id("j1")),
        ..joint_ty(Unit::Angle)
    };
    let norm_j1 = PortType {
        unit: Unit::Normalized { lo: 0.0, hi: 2.0 },
        ..raw_j1.clone()
    };
    ir.graph.insert(
        NodeId(5),
        ObservationNode::StateInput {
            source: joint_id("j1"),
            io: Io::source(raw_j1.clone()),
        },
    );
    ir.graph.insert(
        NodeId(6),
        ObservationNode::Normalize {
            stats: NormalizeStats::Range { lo: 0.0, hi: 2.0 },
            io: Io::unary(raw_j1, norm_j1.clone()),
        },
    );
    ir.graph.connect(NodeId(0), "out", NodeId(1), "in0");
    ir.graph.connect(NodeId(1), "out", NodeId(2), "in0");
    ir.graph.connect(NodeId(3), "out", NodeId(4), "in0");
    ir.graph.connect(NodeId(5), "out", NodeId(6), "in0");
    ir.outputs = BTreeMap::from([
        (
            "rgb".to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(2), "out"),
                ty: scaled,
            },
        ),
        (
            "joint_state".to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(4), "out"),
                ty: norm,
            },
        ),
        (
            SIM_J1.to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(6), "out"),
                ty: norm_j1,
            },
        ),
    ]);
    ir
}

/// `baked_observation_ir` with the demo's augmentation chain on the image path (packet
/// M7/T6): `Pad(p) -> Augment{RandomCrop IMG x IMG, training_only}`, the pair
/// `tests/fixtures/visible-learning/observation-augmented.toml` carries at 96x96.
fn augmented_observation_ir(task_ref: [u8; 32], pad: u32) -> ObservationIr {
    let mut ir = baked_observation_ir(task_ref);
    let scaled = ir.outputs["rgb"].ty.clone();
    let spec = scaled.image.expect("the image port carries a spec");
    let padded_spec = es_ir::image::ImageSpec {
        width: spec.width + 2 * pad,
        height: spec.height + 2 * pad,
        intrinsics: es_ir::image::Intrinsics {
            cx: spec.intrinsics.cx + f64::from(pad),
            cy: spec.intrinsics.cy + f64::from(pad),
            ..spec.intrinsics
        },
        ..spec
    };
    let padded = PortType {
        shape: Shape::new([
            3,
            u64::from(padded_spec.height),
            u64::from(padded_spec.width),
        ]),
        image: Some(padded_spec),
        ..scaled.clone()
    };
    ir.graph.insert(
        NodeId(7),
        ObservationNode::Pad {
            left: pad,
            top: pad,
            right: pad,
            bottom: pad,
            io: Io::unary(scaled.clone(), padded.clone()),
        },
    );
    ir.graph.insert(
        NodeId(8),
        ObservationNode::Augment {
            kind: es_ir::observation::AugmentKind::RandomCrop {
                width: IMG,
                height: IMG,
            },
            training_only: true,
            io: Io::unary(padded, scaled.clone()),
        },
    );
    ir.graph.connect(NodeId(2), "out", NodeId(7), "in0");
    ir.graph.connect(NodeId(7), "out", NodeId(8), "in0");
    ir.outputs.insert(
        "rgb".to_owned(),
        ObservationOutput {
            port: PortRef::new(NodeId(8), "out"),
            ty: scaled,
        },
    );
    ir
}

/// Oracle 3 of packet M7/T6. `--for-training` writes the tensor *entering* the augmentation
/// chain — the padded canvas — and the chain beside it; without it the same document bakes
/// the network's input, at the network's size, byte-identical to the un-augmented document's
/// bake. The boundary is the whole claim: everything before the chain still runs here.
#[test]
fn bake_for_training_writes_the_boundary() {
    let task = privileged_task_ir();
    let hash = task.task_hash().expect("task hashes");
    let plain = baked_observation_ir(hash);
    let obs = augmented_observation_ir(hash, 4);

    let shape_of = |bake: &es_eval::ObservationBake| -> Vec<u64> {
        bake.outputs()
            .find(|(n, _, _)| *n == "rgb")
            .map(|(_, _, s)| s.to_vec())
            .expect("an rgb port")
    };
    let state: Vec<f64> = (0..8).map(|i| f64::from(i) * 0.05).collect();
    let mut tile = |_: &str| {
        Ok((0..IMG as usize * IMG as usize * 3)
            .map(|i| (i * 7 % 251) as u8)
            .collect::<Vec<u8>>())
    };

    let mut evaluation =
        es_eval::ObservationBake::new(&obs, &task, Some(&model())).expect("the bake compiles");
    assert_eq!(
        shape_of(&evaluation),
        vec![3, u64::from(IMG), u64::from(IMG)]
    );

    let mut training =
        es_eval::ObservationBake::new(&obs, &task, Some(&model())).expect("the bake compiles");
    let chains = training.for_training(&obs).expect("one chain");
    assert_eq!(chains.len(), 1, "{chains:?}");
    assert_eq!(chains["rgb"].len(), 1);
    assert_eq!(
        shape_of(&training),
        vec![3, u64::from(IMG) + 8, u64::from(IMG) + 8],
        "--for-training did not write the chain's input"
    );

    // Without the flag: the un-augmented document's bytes, exactly. The centre crop of a
    // symmetric pad is the image that went in (INV-14, INV-15).
    let mut unaugmented =
        es_eval::ObservationBake::new(&plain, &task, Some(&model())).expect("the bake compiles");
    assert_eq!(
        evaluation.frame(&state, &mut tile).expect("a frame")["rgb"],
        unaugmented.frame(&state, &mut tile).expect("a frame")["rgb"],
        "the augmented document's evaluation bake moved a pixel"
    );
    // ...and the training bake is the same pixels in a bigger canvas: the interior matches.
    let padded = training.frame(&state, &mut tile).expect("a frame")["rgb"].clone();
    assert_eq!(padded.data.len(), 3 * (IMG as usize + 8).pow(2) * 4);
    println!(
        "RAN bake_for_training_writes_the_boundary: {:?} at the boundary, {:?} at the network",
        shape_of(&training),
        shape_of(&evaluation)
    );

    // A document with no `training_only` node is untouched by the flag -- which is what makes
    // `--for-training` safe to pass for every recipe.
    let mut same =
        es_eval::ObservationBake::new(&plain, &task, Some(&model())).expect("the bake compiles");
    assert!(same.for_training(&plain).expect("no chain").is_empty());
    assert_eq!(shape_of(&same), vec![3, u64::from(IMG), u64::from(IMG)]);
}

/// A policy that keeps every observation map it was handed, so the test can compare what the
/// *inference* path computed against what the bake computes from the same raw values.
#[derive(Debug, Default)]
struct RecordingPolicy {
    seen: Vec<BTreeMap<String, Tensor>>,
}

impl PolicyRuntime for RecordingPolicy {
    fn load(
        &mut self,
        _graph: &LearningGraph,
        _weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError> {
        Err(PolicyError::NotLoaded)
    }

    fn infer(
        &mut self,
        inputs: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>, PolicyError> {
        self.seen.push(inputs.clone());
        let data: Vec<u8> = (0..H * NJ).flat_map(|_| 0.1f32.to_le_bytes()).collect();
        Ok(BTreeMap::from([(
            "action".to_owned(),
            Tensor {
                dtype: ElemType::F32,
                shape: vec![H as u64, NJ as u64],
                data,
            },
        )]))
    }

    fn info(&self) -> Option<&PolicyInfo> {
        None
    }

    fn runtime_hash(&self) -> [u8; 32] {
        [11; 32]
    }
}

/// **The oracle of packet M5/V2b.** What `es dataset bake` writes for a frame is what
/// `capture` serves the policy for that same frame — every port, every byte, for a whole run.
///
/// This is the test that would have failed before the packet: training fed the raw
/// `observation.state` row where inference feeds `(q + 1) / 2`, and re-implemented
/// `Op::Dequantize` in Python (design note section 7.6 finding 3, open question 11). There is
/// no tolerance here on purpose — the two paths run the same `CpuPlan` over the same bytes, so
/// "close" would mean one of them had grown a conversion of its own.
#[test]
fn a_baked_frame_is_bit_identical_to_what_capture_serves() {
    let ir = one_suite(Vec::new());
    let task = privileged_task_ir();
    let obs = baked_observation_ir(task.task_hash().expect("task hashes"));

    // The frame source is also the recorder: it is handed the very `StateView` `capture` reads,
    // so the rows below are the rows the plan saw, not a re-simulation of them.
    let mut recorded: Vec<(Vec<f64>, Vec<u8>)> = Vec::new();
    let mut frames = |_: &LightOverride, _: &ModelInfo, state: &StateView<'_>| {
        let q = state.qpos_of(0)[0];
        let mut tile = vec![0_u8; IMG as usize * IMG as usize * 3];
        for (i, byte) in tile.iter_mut().enumerate() {
            *byte = (((q.abs() * 1000.0) as usize + i * 7) % 256) as u8;
        }
        recorded.push((state.qpos_of(0).to_vec(), tile.clone()));
        Ok(tile)
    };
    let mut policy = RecordingPolicy::default();
    Evaluation::run_with_frames::<FakeBackend, _, NJ, H>(
        &ir,
        &task,
        &scene(),
        &obs,
        &mut policy,
        &deployment_ir(),
        FakeBackend::new,
        &RunConfig::default(),
        Some(&mut frames),
        None,
    )
    .expect("the demo-shaped observation runs");

    assert_eq!(
        recorded.len(),
        policy.seen.len(),
        "one captured frame per inference"
    );
    assert!(recorded.len() >= N_EPISODES as usize, "{}", recorded.len());

    // The model is the one the run resolved against, and it is what carries `j1`'s `qpos`
    // range. `observation.state` is `qpos ‖ qvel`, so that range indexes the recorded row
    // exactly as `capture` indexes `StateView::qpos_of(0)`.
    let mut bake =
        es_eval::ObservationBake::new(&obs, &task, Some(&model())).expect("the bake compiles");
    let ports: Vec<String> = bake.outputs().map(|(n, _, _)| n.to_owned()).collect();
    assert_eq!(
        ports,
        vec![
            "joint_state".to_owned(),
            "rgb".to_owned(),
            SIM_J1.to_owned()
        ]
    );
    for (frame, ((state, tile), served)) in recorded.iter().zip(&policy.seen).enumerate() {
        let baked = bake
            .frame(state, &mut |_| Ok(tile.clone()))
            .unwrap_or_else(|e| panic!("bake frame {frame}: {e}"));
        assert_eq!(
            baked, *served,
            "frame {frame}: the bake and `capture` disagree. Training and inference would see \
             different observations, which is exactly the defect packet M5/V2b closes."
        );
    }
    // Non-vacuity: both nodes the old training path got wrong must actually fire here, or the
    // assertion above would be comparing two copies of the raw input and would have passed
    // before this packet too. `Normalize{−1..1}` maps `q` to `(q + 1) / 2`, which is the affine
    // shift open question 11 names; `Dequantize` maps an HWC `u8` tile to CHW `f32 / 255`.
    let last = policy.seen.last().expect("at least one inference");
    let (raw_q, raw_tile) = recorded.last().expect("at least one frame");
    let state_out = f32::from_le_bytes(last["joint_state"].data[..4].try_into().expect("4 bytes"));
    assert!(
        (f64::from(state_out) - f64::midpoint(raw_q[0], 1.0)).abs() < 1e-6,
        "the state Normalize did not fire: {state_out} for q = {}",
        raw_q[0]
    );
    assert_ne!(
        last["rgb"].data, *raw_tile,
        "the image Dequantize did not fire"
    );
    // Non-vacuity for the privileged port (packet M5/V7a): it must be `q[1] / 2`, and `q[1]`
    // must differ from `q[0]`, or the byte comparison above would pass on a bake that read the
    // leading value of the row for both state ports -- which is what one did before this
    // packet, and what it still does for any channel that names a body rather than a joint.
    let privileged = f32::from_le_bytes(last[SIM_J1].data[..4].try_into().expect("4 bytes"));
    assert!(
        (f64::from(privileged) - raw_q[1] / 2.0).abs() < 1e-6,
        "the privileged Normalize did not fire: {privileged} for q[1] = {}",
        raw_q[1]
    );
    assert!(
        (raw_q[1] - raw_q[0]).abs() > 1e-3,
        "q[0] and q[1] coincide ({}, {}), so the offset is untested",
        raw_q[0],
        raw_q[1]
    );
    println!(
        "RAN observation_bake_bit_identity: {} frames x {} ports, byte-equal; state Normalize, \
         image Dequantize and the privileged qpos[1..2] port all fired",
        recorded.len(),
        ports.len()
    );
}

/// An input that is read out of a loaded model has no reading in a recorded dataset. The bake
/// says so at construction, naming the port, rather than guessing an offset into
/// `observation.state` halfway through an episode.
#[test]
fn a_bake_refuses_an_input_the_dataset_cannot_feed() {
    let task = task_ir();
    // `observation_ir`'s `StateInput` names joint `j0`, which resolves through `ModelInfo::qpos`
    // at inference and through nothing at all here.
    let obs = observation_ir(task.task_hash().expect("task hashes"), None);
    let err = es_eval::ObservationBake::new(&obs, &task, None).expect_err("a qpos input");
    let EvalError::Plan(message) = &err else {
        panic!("expected EvalError::Plan, got {err}");
    };
    assert!(
        message.contains("there is no loaded model here: the frames are recorded"),
        "{message}"
    );
}

/// Two `JointState` channels and no model is the one case the leading-`dof` convention cannot
/// answer, and the bake says so at construction rather than serving one of them the other's
/// values (packet M5/V7a). It is the refusal `es dataset bake` catches to go and load the
/// scene's `qpos` ranges.
#[test]
fn a_bake_refuses_two_state_channels_without_a_model() {
    let task = privileged_task_ir();
    let obs = baked_observation_ir(task.task_hash().expect("task hashes"));
    let err = es_eval::ObservationBake::new(&obs, &task, None).expect_err("two state channels");
    let EvalError::Plan(message) = &err else {
        panic!("expected EvalError::Plan, got {err}");
    };
    assert!(
        message.contains("only one channel can be the leading"),
        "{message}"
    );
    // And the same pair resolves once the model that ran is handed over.
    es_eval::ObservationBake::new(&obs, &task, Some(&model())).expect("with the model");
}

// --- packet M5/V5: `--jobs N` is a partition of the cells ------------------------------------

/// Four suites, so `--jobs 4` is one cell per worker and `--jobs 1` is all four in a row. The
/// perturbations are the supported kinds on four separate streams, which is what makes the
/// four cells four different runs rather than four copies of one.
fn four_suite_image_ir() -> (EvaluationIr, ObservationIr) {
    let (mut ir, obs) = image_ir();
    ir.suites = vec![
        PerturbationSuite {
            name: "nominal".to_owned(),
            perturbations: Vec::new(),
        },
        PerturbationSuite {
            name: "torque_noise".to_owned(),
            perturbations: vec![Perturbation::new(
                PerturbationKind::TorqueNoise { rel_sigma: 0.05 },
                0,
            )],
        },
        PerturbationSuite {
            name: "backlash".to_owned(),
            perturbations: vec![Perturbation::new(
                PerturbationKind::Backlash {
                    rad: Range::new(0.0, 0.01),
                },
                1,
            )],
        },
        PerturbationSuite {
            name: "light_intensity".to_owned(),
            perturbations: vec![Perturbation::new(
                PerturbationKind::LightIntensity {
                    range: Range::new(0.5, 1.5),
                    dist: es_ir::evaluation::Distribution::default(),
                },
                2,
            )],
        },
    ];
    (ir, obs)
}

/// The three artifacts as bytes, plus every frame on disk: what `--jobs N` is allowed to move,
/// which is nothing.
type Artifacts = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<(String, Vec<u8>)>);

fn read_artifacts(
    report: &EvaluationReport,
    lock: &es_eval::EvaluationLock,
    events: &FrameSink,
    dir: &std::path::Path,
) -> Artifacts {
    let out = dir.join("artifacts");
    es_eval::write_artifacts(report, lock, &out).expect("the two spec 10.5 artifacts");
    let events_path = out.join("events.json");
    events.write_events(&events_path).expect("events.json");

    let mut frames = Vec::new();
    for name in events.events.keys() {
        for entry in std::fs::read_dir(dir.join(name)).expect("the cell directory") {
            let path = entry.expect("entry").path();
            let file = path
                .file_name()
                .expect("name")
                .to_string_lossy()
                .into_owned();
            frames.push((
                format!("{name}/{file}"),
                std::fs::read(&path).expect("frame"),
            ));
        }
    }
    frames.sort();
    (
        std::fs::read(out.join("report.json")).expect("report.json"),
        std::fs::read(out.join("evaluation.lock")).expect("evaluation.lock"),
        std::fs::read(&events_path).expect("events.json"),
        frames,
    )
}

/// `jobs` workers over the cells, merged the way `es eval run --jobs N` merges them.
///
/// A **fresh `FakePolicy` per shard**, which is what a separate process gives: if a cell read
/// anything the previous cell left behind in the policy, this would not agree with the
/// sequential run.
fn sharded(ir: &EvaluationIr, obs: &ObservationIr, jobs: u32, dir: &std::path::Path) -> Artifacts {
    let task = task_ir();
    let (deploy, cfg) = (deployment_ir(), RunConfig::default());
    let mut shards = Vec::new();
    let mut events = FrameSink::new(dir);
    for i in 0..jobs {
        let mut policy = FakePolicy { target: 0.2 };
        let mut frames = state_frames();
        let mut shard = Evaluation::run_shard::<FakeBackend, _, NJ, H>(
            ir,
            &task,
            &scene(),
            obs,
            &mut policy,
            &deploy,
            FakeBackend::new,
            &cfg,
            Some(&mut frames),
            Some(dir),
            (i, jobs),
        )
        .unwrap_or_else(|e| panic!("shard {i}/{jobs}: {e}"));
        events.events.append(&mut shard.events);
        shards.push(shard);
    }
    let policy = FakePolicy { target: 0.2 };
    let (report, lock) = Evaluation::merge(ir, &task, obs, &deploy, &policy, &cfg, &shards)
        .expect("the workers merge");
    read_artifacts(&report, &lock, &events, dir)
}

/// The headline of the packet: `--jobs 4` is a scheduling choice, not a different evaluation.
///
/// Three runs of one four-suite image evaluation — the sequential entry point, one worker, and
/// four workers — and all three must produce byte-identical `report.json`, `evaluation.lock`
/// and `events.json`, and byte-identical frames in every cell directory (spec 10.4, spec 3.5
/// tier 1).
#[test]
fn sharding_the_cells_produces_a_byte_identical_report() {
    let (ir, obs) = four_suite_image_ir();

    let sequential = {
        let dir = scratch("jobs-seq");
        let task = task_ir();
        let mut policy = FakePolicy { target: 0.2 };
        let mut frames = state_frames();
        let mut sink = FrameSink::new(&dir);
        let (report, lock) = Evaluation::run_with_frames::<FakeBackend, _, NJ, H>(
            &ir,
            &task,
            &scene(),
            &obs,
            &mut policy,
            &deployment_ir(),
            FakeBackend::new,
            &RunConfig::default(),
            Some(&mut frames),
            Some(&mut sink),
        )
        .expect("the sequential run");
        read_artifacts(&report, &lock, &sink, &dir)
    };

    let one = sharded(&ir, &obs, 1, &scratch("jobs-1"));
    let four = sharded(&ir, &obs, 4, &scratch("jobs-4"));

    for (jobs, got) in [(1, &one), (4, &four)] {
        assert_eq!(
            String::from_utf8_lossy(&got.0),
            String::from_utf8_lossy(&sequential.0),
            "--jobs {jobs} moved report.json"
        );
        assert_eq!(got.1, sequential.1, "--jobs {jobs} moved evaluation.lock");
        assert_eq!(
            String::from_utf8_lossy(&got.2),
            String::from_utf8_lossy(&sequential.2),
            "--jobs {jobs} moved events.json"
        );
        assert_eq!(
            got.3.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            sequential.3.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            "--jobs {jobs} moved which frames exist"
        );
        assert_eq!(got.3, sequential.3, "--jobs {jobs} moved the frame bytes");
    }
    // The comparison has to be over something: four suites x N_EPISODES cells, each holding a
    // `layout.json` and at least one frame.
    let cells = ir.suites.len() * N_EPISODES as usize;
    assert!(
        sequential.3.len() > cells * 2,
        "the fixture rendered {} files over {cells} cells",
        sequential.3.len()
    );
}

/// A worker that died must not become a report over the suites that survived: a merge that
/// does not cover every cell exactly once is refused, and the refusal names what it got
/// (spec 10.4).
#[test]
fn a_merge_missing_a_cell_is_refused() {
    let (ir, obs) = four_suite_image_ir();
    let task = task_ir();
    let (deploy, cfg) = (deployment_ir(), RunConfig::default());
    let dir = scratch("jobs-lost");

    let mut shards = Vec::new();
    for i in 0..4 {
        let mut policy = FakePolicy { target: 0.2 };
        let mut frames = state_frames();
        shards.push(
            Evaluation::run_shard::<FakeBackend, _, NJ, H>(
                &ir,
                &task,
                &scene(),
                &obs,
                &mut policy,
                &deploy,
                FakeBackend::new,
                &cfg,
                Some(&mut frames),
                Some(&dir),
                (i, 4),
            )
            .expect("the shard runs"),
        );
    }
    let policy = FakePolicy { target: 0.2 };
    shards.remove(2);
    let err = Evaluation::merge(&ir, &task, &obs, &deploy, &policy, &cfg, &shards)
        .expect_err("a merge missing a cell is not a report");
    let message = err.to_string();
    assert!(matches!(err, EvalError::Shard(_)), "{message}");
    assert!(message.contains("[0, 1, 3]"), "{message}");

    // The same four shards, all present, do produce one.
    assert!(
        Evaluation::run_shard::<FakeBackend, _, NJ, H>(
            &ir,
            &task,
            &scene(),
            &obs,
            &mut FakePolicy { target: 0.2 },
            &deploy,
            FakeBackend::new,
            &cfg,
            Some(&mut state_frames()),
            Some(&dir),
            (0, 0),
        )
        .is_err(),
        "a zero-count partition is not a partition"
    );
}

/// Packet M7/T8 oracle 3 — **the `(cell, episode)` partition does not reproduce the
/// sequential run, and this is the state that stops it.**
///
/// `Env::seek_episode` makes an env's episode `k` reachable without replaying `0..k`
/// (`cargo test -p es-env --test seek`), so the question the packet asked was whether a shard
/// could be one episode rather than one whole suite. Measured on the committed demo documents
/// by running the pre-T8 and the T8 build of `es eval run` over the nominal suite (oracle
/// server, 2026-09-21, `docs/design/evaluation-execution.md` section 2.7): no. The artifacts
/// moved, in two independent places, and this test pins both so that a change to either is
/// loud rather than silent.
///
/// It is a tripwire and not a gate: when `begin_episode` starts clearing the window, or when
/// `StepEvent::tick` becomes episode-relative, the matching assertion here fails and sends the
/// reader back to section 2.7 — where the partition is written down as a small change on top
/// of what shipped.
#[test]
#[ignore = "oracle tier: the M7/T8 parity finding, pinned"]
fn episode_shards_reproduce_the_sequential_run() {
    // (1) The plane's spec 9.4 `ViolationRate` ring survives `begin_episode`, so episode `k`
    //     is judged partly on episode `k-1`'s steps. A per-episode plane starts empty, which
    //     reads 0.0 until the window fills -- 227 fewer `violation.rate` events over the
    //     demo's four nominal episodes, and the trajectory parts company at tick 24.
    const WINDOW: u32 = 8;
    let mut deploy = deployment_ir();
    deploy.watchdogs = WatchdogSet(vec![
        Watchdog::ChunkUnderrun,
        Watchdog::EnvelopeViolationRate {
            window: WINDOW,
            max_frac: 0.5,
        },
    ]);
    let mut plane = es_safety::SafetyPlane::<NJ, H>::from_ir(&deploy).expect("the plane builds");
    // Every step past the soft position limit, so every step is dirty and the ring fills.
    let far = [100.0; NJ];
    let chunk = es_safety::ActionChunk::new([far; H], H, ExecutionMode::RecedingHorizon);
    for tick in 0..u64::from(WINDOW) {
        plane.observe_state(&[0.0; NJ], &[0.0; NJ]);
        plane.heartbeat(PhysTick(tick));
        plane.validate(&chunk.with_seq(tick + 1), Micros(0), PhysTick(tick));
    }
    let filled = plane.counters().envelope_violation_rate();
    assert!(
        filled > 0.5,
        "the fixture never filled the window: {filled}"
    );
    plane.begin_episode();
    let after = plane.counters().envelope_violation_rate();
    assert_eq!(
        filled.to_bits(),
        after.to_bits(),
        "begin_episode now clears the ViolationRate window -- re-read \
         docs/design/evaluation-execution.md section 2.7: the partition may be shippable"
    );

    // (2) `StepEvent::tick` is the *env's* cumulative physics clock, not the episode's, so a
    //     per-episode env writes a different `events.json` whatever the plane does.
    let (ir, obs) = image_ir();
    let dir = scratch("t8-tick");
    let mut sink = FrameSink::new(&dir);
    run_deploy(
        &ir,
        &obs,
        0.2,
        &deployment_ir(),
        Some(&mut state_frames()),
        Some(&mut sink),
    )
    .expect("the one-suite image fixture runs");
    let second = sink
        .events
        .get("nominal-01")
        .expect("the fixture runs more than one episode");
    assert!(
        second[0].tick.0 > 0,
        "StepEvent::tick is episode-relative now ({:?}) -- re-read \
         docs/design/evaluation-execution.md section 2.7",
        second[0].tick
    );

    println!(
        "RAN episode_shards_reproduce_the_sequential_run: not identical. Measured on the demo \
         documents -- first difference at nominal-01/tick 0 (events.json tick 7200 against 0) \
         and nominal-01/tick 24 (.estraj; violation.rate 2318 against 2091). Pinned here: the \
         window survives begin_episode ({filled}), and this fixture's nominal-01 opens at env \
         tick {}. The partition stays cell-level.",
        second[0].tick.0
    );
}

// --- packet M8/S4e: joint velocities and a body's pose ----------------------------------------

/// One `StateInput` per channel, in the order the channels are named.
///
/// Deliberately not `baked_observation_ir`'s shape: this fixture is about *capture*, so the
/// graph is the thinnest thing that gives every channel its own plan input buffer.
fn state_only_observation_ir(task: &TaskIr, order: &[&str]) -> ObservationIr {
    let mut ir = ObservationIr::new(1, task.task_hash().expect("the task hashes"));
    for (i, name) in order.iter().enumerate() {
        let ch = &task.observation_spec.channels[*name];
        let source = match ch.source {
            ObsSource::JointState { body, .. } | ObsSource::BodyPose(body) => body,
            ref other => panic!("{other:?} is not a state source"),
        };
        ir.graph.insert(
            NodeId(i as u32),
            ObservationNode::StateInput {
                source,
                io: es_ir::observation::Io::source(ch.ty.clone()),
            },
        );
        ir.graph.outputs.push(PortRef::new(NodeId(i as u32), "out"));
        ir.outputs.insert(
            (*name).to_owned(),
            es_ir::observation::ObservationOutput {
                port: PortRef::new(NodeId(i as u32), "out"),
                ty: ch.ty.clone(),
            },
        );
    }
    ir
}

/// Packet M8/S4e oracle 2: `Capture::Qvel`, `Capture::JointsVel` and `Capture::BodyPose` land
/// the right numbers in the right ports, the quaternion is xyzw, the model-free path refuses
/// both new kinds by name -- and the committed demo's resolution does not move.
#[test]
fn capture_reads_qvel_and_body_pose() {
    use es_eval::runner::{capture, input_sources, Capture};
    use es_physics_core::backend::IndexRange;

    let s = scene();
    let body = s.bodies[0].id;
    let vec_ty = |n: u64, unit: Unit, frame: Frame| PortType {
        elem: ElemType::F32,
        shape: Shape::new([n]),
        unit,
        frame,
        time: TimeRef::Tick,
        image: None,
    };

    // A two-joint arm on one body, read three ways. `j1` is the *second* joint, so a velocity
    // channel naming it must start at `qvel[1]` and not at `qvel[0]` -- which is the whole
    // difference between the `Qvel` arm and the `JointsVel` one.
    let mut task = task_ir();
    task.observation_spec.channels.clear();
    for (name, source, ty) in [
        (
            "joint_pos",
            ObsSource::JointState {
                body,
                dof: 2,
                quantity: JointQuantity::Position,
            },
            vec_ty(2, Unit::Angle, Frame::Joint(body)),
        ),
        (
            "joint_vel",
            ObsSource::JointState {
                body: joint_id("j1"),
                dof: 1,
                quantity: JointQuantity::Velocity,
            },
            vec_ty(1, Unit::AngularVelocity, Frame::Joint(joint_id("j1"))),
        ),
        (
            "link_pose",
            ObsSource::BodyPose(s.bodies[1].id),
            vec_ty(7, Unit::Length, Frame::World),
        ),
    ] {
        task.observation_spec
            .channels
            .insert(name.to_owned(), es_ir::task::ObsChannel { source, ty });
    }
    let order = ["joint_pos", "joint_vel", "link_pose"];
    let obs = state_only_observation_ir(&task, &order);
    assert!(obs.validate().is_empty(), "{:#?}", obs.validate());
    let plan = es_compile::plan::CpuPlan::compile(&obs, es_compile::plan::PlanMode::Debug)
        .expect("the state graph lowers");

    // Body 1 is row 1 of `xpos`/`xquat`; body 0 has no pose channel here.
    let mut model = model();
    model.nbody = 2;
    model.body = [
        (s.bodies[0].id, IndexRange::new(0, 1)),
        (s.bodies[1].id, IndexRange::new(1, 1)),
    ]
    .into();

    let sources = input_sources(&plan, &obs, &task, Some(&model)).expect("every input resolves");
    let key_of = |name: &str| match task.observation_spec.channels[name].source {
        ObsSource::JointState { body, .. } | ObsSource::BodyPose(body) => body.to_string(),
        ref other => panic!("{other:?}"),
    };
    let of = |name: &str| sources[&key_of(name)];
    assert!(matches!(of("joint_pos"), Capture::Joints(2)), "joint_pos");
    assert!(
        matches!(of("joint_vel"), Capture::Qvel(r) if r == IndexRange::new(1, 1)),
        "joint_vel is {:?}",
        of("joint_vel")
    );
    assert!(matches!(of("link_pose"), Capture::BodyPose(1)), "link_pose");

    // Distinct values everywhere, so a wrong slice cannot look right by accident.
    let qpos = [1.0, 2.0];
    let qvel = [3.0, 4.0];
    let xpos = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
    // Body 1's orientation, in `StateView`'s own order: x, y, z, w (spec 3.1).
    // Exactly representable in f32, which the port is: the comparison is bitwise.
    let xquat = [0.0, 0.0, 0.0, 1.0, 0.125, 0.25, 0.375, 0.5];
    let state = StateView {
        n_envs: 1,
        tick: PhysTick(0),
        qpos: &qpos,
        qvel: &qvel,
        act: &[],
        sensordata: &[0.0],
        xpos: &xpos,
        xquat: &xquat,
    };
    let (descs, bytes, _) = capture(
        &plan,
        &sources,
        &model,
        &state,
        None,
        &LightOverride::default(),
        None,
    )
    .expect("the capture reads every port");
    let read = |name: &str| -> Vec<f64> {
        let key = key_of(name);
        let i = descs
            .iter()
            .position(|(n, _, _)| *n == key)
            .unwrap_or_else(|| panic!("{name} is not in the capture"));
        bytes[i]
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()
    };
    assert_eq!(read("joint_pos"), vec![1.0, 2.0], "qpos[0..2]");
    assert_eq!(read("joint_vel"), vec![4.0], "qvel[1..2], j1's own dof");
    assert_eq!(
        read("link_pose"),
        vec![20.0, 21.0, 22.0, 0.125, 0.25, 0.375, 0.5],
        "xpos[3..6] || xquat[4..8], the quaternion xyzw"
    );

    // A velocity channel naming a **body** takes the leading `dof` of the row instead -- the
    // mirror of `Joints`. On its own task because this scene has only two body ids.
    let mut leading = task.clone();
    leading.observation_spec.channels.clear();
    leading.observation_spec.channels.insert(
        "body_vel".to_owned(),
        es_ir::task::ObsChannel {
            source: ObsSource::JointState {
                body: s.bodies[1].id,
                dof: 2,
                quantity: JointQuantity::Velocity,
            },
            ty: vec_ty(2, Unit::AngularVelocity, Frame::World),
        },
    );
    let lead_obs = state_only_observation_ir(&leading, &["body_vel"]);
    let lead_plan =
        es_compile::plan::CpuPlan::compile(&lead_obs, es_compile::plan::PlanMode::Debug)
            .expect("it lowers");
    let lead = input_sources(&lead_plan, &lead_obs, &leading, Some(&model)).expect("it resolves");
    assert!(
        matches!(lead[&s.bodies[1].id.to_string()], Capture::JointsVel(2)),
        "body_vel is {:?}",
        lead[&s.bodies[1].id.to_string()]
    );
    let (_, lead_bytes, _) = capture(
        &lead_plan,
        &lead,
        &model,
        &state,
        None,
        &LightOverride::default(),
        None,
    )
    .expect("the leading-dof velocity arm reads");
    assert_eq!(
        lead_bytes[0]
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect::<Vec<f64>>(),
        vec![3.0, 4.0],
        "qvel[0..2]"
    );

    // Two channels on one id: the observation plan allocates one buffer per source id, so the
    // pair is refused by name rather than served the same numbers twice (packet M8/S4e).
    let mut both = task.clone();
    both.observation_spec
        .channels
        .get_mut("joint_vel")
        .expect("joint_vel")
        .source = ObsSource::JointState {
        body,
        dof: 2,
        quantity: JointQuantity::Velocity,
    };
    let two = state_only_observation_ir(&both, &["joint_pos"]);
    let two_plan = es_compile::plan::CpuPlan::compile(&two, es_compile::plan::PlanMode::Debug)
        .expect("it lowers");
    let said = input_sources(&two_plan, &two, &both, Some(&model))
        .expect_err("one buffer cannot serve two channels")
        .to_string();
    assert!(said.contains("is named by 2 channels"), "{said}");

    // Model-free (recorded frames): both new kinds are refused **by name**, never guessed.
    for (name, want) in [
        ("joint_vel", "joint **velocity** channel"),
        ("link_pose", "BodyPose channel"),
    ] {
        let mut one = task.clone();
        one.observation_spec.channels.retain(|k, _| k == name);
        let obs = state_only_observation_ir(&one, &[name]);
        let plan = es_compile::plan::CpuPlan::compile(&obs, es_compile::plan::PlanMode::Debug)
            .expect("it lowers");
        let said = input_sources(&plan, &obs, &one, None)
            .expect_err("a recorded row carries neither")
            .to_string();
        assert!(said.contains(want), "{name}: {said}");
    }

    // And the committed demo's resolution is exactly what it was: an image, the arm's leading
    // six positions, and the cube's own `qpos` range. Pinned here, against the documents.
    let vl = |n: &str| {
        std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/visible-learning")
                .join(n),
        )
        .unwrap_or_else(|e| panic!("{n}: {e}"))
    };
    let demo_task = es_ir::serial::task_from_toml(&vl("task.toml")).expect("task.toml");
    let demo_obs =
        es_ir::serial::observation_from_toml(&vl("observation.toml")).expect("observation.toml");
    let demo_plan =
        es_compile::plan::CpuPlan::compile(&demo_obs, es_compile::plan::PlanMode::Debug)
            .expect("the demo observation lowers");
    let id_of = |n: &str| match demo_task.observation_spec.channels[n].source {
        ObsSource::JointState { body, .. } | ObsSource::BodyPose(body) => body,
        ObsSource::Sensor { id, .. } => id,
        ObsSource::Language => panic!("{n} is a language channel"),
    };
    let (cube, arm, camera) = (
        id_of("sim_cube_pose"),
        id_of("joint_state"),
        id_of("rgb_overhead"),
    );
    let demo_model = ModelInfo {
        nq: 13,
        nv: 12,
        qpos: [(cube, IndexRange::new(6, 7))].into(),
        ..ModelInfo::default()
    };
    let demo = input_sources(&demo_plan, &demo_obs, &demo_task, Some(&demo_model))
        .expect("the demo documents resolve");
    assert!(matches!(demo[&camera.to_string()], Capture::Image));
    assert!(matches!(demo[&arm.to_string()], Capture::Joints(6)));
    assert!(
        matches!(demo[&cube.to_string()], Capture::Qpos(r) if r == IndexRange::new(6, 7)),
        "the demo's cube channel is {:?}",
        demo[&cube.to_string()]
    );
    assert_eq!(
        demo.len(),
        3,
        "the demo declares three plan inputs: {demo:?}"
    );
    println!("RAN capture_reads_qvel_and_body_pose: demo {demo:?}");
}
