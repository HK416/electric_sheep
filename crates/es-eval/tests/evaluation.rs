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
