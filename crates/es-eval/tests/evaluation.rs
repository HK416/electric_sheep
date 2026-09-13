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
use es_eval::{EvalError, Evaluation, RunConfig};
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
    let task = task_ir();
    let deploy = deployment_ir();
    let mut policy = FakePolicy { target };
    Evaluation::run::<FakeBackend, _, NJ, H>(
        ir,
        &task,
        &scene(),
        obs,
        &mut policy,
        &deploy,
        FakeBackend::new,
        &RunConfig::default(),
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
