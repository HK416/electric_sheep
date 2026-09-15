//! Oracle for `docs/packets/M3/W7-learning-loop.md` (spec 13.1, 13.2, 13.3, 19.2, 19.3).
//!
//! `FakeBackend` and `FakePolicy` are **copied** from `crates/es-eval/tests/evaluation.rs`
//! rather than imported: borrowing a neighbour's idea of a consistent fixture would make this
//! gate test the neighbour. The task graph is trimmed to have no `Terminate` node, so every
//! episode runs the full step budget and the frame indices an intervener injects on are known
//! exactly.

use std::collections::{BTreeMap, BTreeSet};

use es_assets::scene::SceneDesc;
use es_compile::{BundleKind, BundleManifest, PolicyBundle, Tensor};
use es_core::time::{PhysTick, TickRate};
use es_core::{FailureKind, StableId};
use es_data::collect::{
    read_loop_steps, CollectSpec, Collector, Intervention, LoopKind, SplitSpec,
};
use es_data::intervention::{ActionSourceCode, InterventionSegment, InterventionSource};
use es_data::{Column, LeRobotDataset};
use es_data::{ACTION_COMMANDED, ACTION_SOURCE, INTERVENTION};
use es_env::Termination;
use es_ir::deployment::{
    ActionContract, ActionSpace as DepSpace, Deadlines, DeploymentIr, ExecutionMode,
    FallbackPolicy, Limit, Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope,
    Watchdog, WatchdogSet, Workspace,
};
use es_ir::graph::{Graph, NodeId, PortRef};
use es_ir::learning::{
    ActionExecutionMode, ArchKind, LearningGraph, PolicyContract, PolicyHandle, RuntimeHints,
    WeightsRef,
};
use es_ir::observation::{Io, NormalizeStats, ObservationIr, ObservationNode, ObservationOutput};
use es_ir::task::{
    ActionSpace as TaskSpace, Distribution, JointQuantity, ObsChannel, ObsSource, ObservationSpec,
    SceneRef, TaskConfig, TaskGraph, TaskIr, TaskNode,
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
const STEPS: u32 = 12;
/// The frames the scripted intervener takes over, inclusive.
const INJECT: std::ops::RangeInclusive<u32> = 2..=4;

// --- Scene, model, backend ------------------------------------------------------------------

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
        rate: TickRate::hz(CONTROL_HZ),
        ..ModelInfo::default()
    }
}

/// A spring-damper integrator: deterministic, transcendental-free (spec 3.4).
#[derive(Debug)]
struct FakeBackend {
    caps: Capabilities,
    model: Option<ModelInfo>,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    sensordata: Vec<f64>,
    ctrl: Vec<f64>,
    tick: PhysTick,
}

impl FakeBackend {
    fn new() -> Self {
        Self {
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

// --- Policy ---------------------------------------------------------------------------------

/// A proportional controller: `target - 0.5 * observation`, repeated over the chunk.
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

// --- IR fixtures ----------------------------------------------------------------------------

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

/// `reward = -angle` with a randomized initial angle and **no** `Terminate` node, so every
/// episode runs the whole budget and the intervened frame indices are known exactly.
fn task_ir() -> TaskIr {
    let angle_ty = joint_ty(Unit::Angle);
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
        TaskNode::ResetState {
            target: "qpos[0]".to_owned(),
            dist: Distribution::Uniform { lo: -1.0, hi: 1.0 },
            stream: "reset.j0".to_owned(),
        },
    );
    graph.insert(
        NodeId(3),
        TaskNode::ActionSpec {
            space: TaskSpace::JointPosition,
            dim: NJ as u32,
            control_rate_hz: CONTROL_HZ as f32,
        },
    );
    graph.connect(NodeId(0), "value", NodeId(1), "value");

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
            max_episode_steps: STEPS,
            control_rate_hz: CONTROL_HZ as f32,
            deterministic: true,
            rng_streams: BTreeSet::new(),
        },
    }
}

fn observation_ir(task_ref: [u8; 32]) -> ObservationIr {
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
    ir.outputs = BTreeMap::from([(
        "joint_state".to_owned(),
        ObservationOutput {
            port: PortRef::new(NodeId(1), "out"),
            ty: norm,
        },
    )]);
    ir
}

/// `execute_chunk = 1` and zero expected latency: one policy result per control tick, so an
/// injection at frame `f` drives exactly frame `f` (design note section 5.1).
fn learning_graph() -> LearningGraph {
    LearningGraph {
        schema_version: 1,
        inputs: Vec::new(),
        nodes: Graph::new(1),
        outputs: Vec::new(),
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "weights.safetensors".to_owned(),
                hash: [9; 32],
            },
            contract: PolicyContract {
                inputs: BTreeMap::new(),
                observation_window: 1,
                action_dim: NJ as u32,
                horizon: H as u32,
                execute_chunk: 1,
                replanning_hz: CONTROL_HZ as f32,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 0.0,
                    deadline_ms: 100.0,
                },
            },
        },
    }
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
            execute_chunk: 1,
        },
        safety: SafetyEnvelope {
            position: vec![Limit::symmetric(2.8); NJ],
            position_soft_margin: vec![0.05; NJ],
            // Wide enough that the step into an intervention (a jump of up to ~1.2 rad in
            // one 10 ms period, so ~1.2e4 rad/s^2) is not a clamp: this test asserts that
            // action_source is exactly {policy, human}, and a clamped human action is
            // legitimately clamped. Widening the envelope is the sanctioned move; disabling
            // the plane to make a test pass is not (INV-12).
            velocity_max: vec![1.0e5; NJ],
            acceleration_max: vec![1.0e8; NJ],
            torque_max: vec![80.0; NJ],
            jerk_max: None,
            action_rate: RateLimit {
                first_diff_max: vec![1000.0; NJ],
                second_diff_max: vec![1000.0; NJ],
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

/// The fields `Collector::run` reads. Assembled directly rather than through
/// `PolicyBundle::build`: this packet is about the loop, not about the container.
fn bundle() -> PolicyBundle {
    let task = task_ir();
    let observation = observation_ir(task.task_hash().expect("task hashes"));
    PolicyBundle {
        manifest: BundleManifest::new(BundleKind::Policy, es_compile::BundleHashes::default()),
        task,
        observation,
        learning: learning_graph(),
        deployment: deployment_ir(),
        weights: Vec::new(),
        evaluation: None,
    }
}

// --- Helpers --------------------------------------------------------------------------------

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("es-data-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Injects a constant action on [`INJECT`] of every episode; a pure function of the frame, so
/// the run stays reproducible.
fn scripted(_episode: u32, frame: u32, _model: &ModelInfo, _obs: &[f64]) -> Intervention<NJ> {
    if INJECT.contains(&frame) {
        Intervention::Action([0.7; NJ])
    } else {
        Intervention::Policy
    }
}

fn collect_into(root: &std::path::Path, episodes: u32, seed: u64) -> es_data::CollectReport {
    let b = bundle();
    let s = scene();
    let mut policy = FakePolicy { target: 0.2 };
    let mut hook = scripted;
    Collector::run::<FakeBackend, _, NJ, H>(
        &CollectSpec {
            bundle: &b,
            scene: &s,
            n_episodes: episodes,
            seed,
            max_steps: STEPS,
            out_root: root,
        },
        &mut policy,
        FakeBackend::new,
        &mut hook,
        None,
    )
    .expect("the fixture collect run succeeds")
}

/// The same bundle with one image channel declared, so `collect_features` emits the `video`
/// feature whose dangling-reference warning V1 is about. Only the Task IR's `ObservationSpec`
/// matters here: nothing on the collect path reads the pixels.
fn bundle_with_camera() -> PolicyBundle {
    use es_ir::image::{
        CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
        ShutterModel,
    };
    let sensor = StableId::from_path("sensor:overhead");
    let spec = ImageSpec {
        width: 8,
        height: 6,
        channels: ChannelFormat::Rgb,
        dtype: ImageDType::U8,
        color_space: ColorSpace::SRgb,
        camera_model: CameraModel::Pinhole,
        intrinsics: Intrinsics::new(8.0, 8.0, 4.0, 3.0),
        extrinsics: es_math::Pose::IDENTITY,
        distortion: DistortionModel::None,
        shutter: ShutterModel::Global,
        exposure: std::time::Duration::from_micros(500),
        rate_hz: CONTROL_HZ as f32,
        depth_scale: None,
    };
    let mut b = bundle();
    b.task.observation_spec.channels.insert(
        "overhead".to_owned(),
        ObsChannel {
            source: ObsSource::Sensor {
                id: sensor,
                format: ChannelFormat::Rgb,
            },
            ty: PortType {
                elem: ElemType::U8,
                shape: Shape::new([6, 8, 3]),
                unit: Unit::Pixel,
                frame: Frame::Camera(sensor),
                time: TimeRef::Tick,
                image: Some(spec),
            },
        },
    );
    b
}

/// One collect run with a bundle, an intervener and an optional frame sink, so the tests below
/// differ in exactly the thing they are about.
fn collect_with(
    root: &std::path::Path,
    b: &PolicyBundle,
    hook: es_data::Intervener<'_, NJ>,
    sink: Option<es_data::FrameSink<'_>>,
) -> es_data::CollectReport {
    let s = scene();
    let mut policy = FakePolicy { target: 0.2 };
    Collector::run::<FakeBackend, _, NJ, H>(
        &CollectSpec {
            bundle: b,
            scene: &s,
            n_episodes: 2,
            seed: 7,
            max_steps: STEPS,
            out_root: root,
        },
        &mut policy,
        FakeBackend::new,
        hook,
        sink,
    )
    .expect("the fixture collect run succeeds")
}

fn i64_column(ep: &es_data::Episode, name: &str) -> Vec<i64> {
    match ep.columns.get(name) {
        Some(Column::I64(v)) => v.clone(),
        other => panic!("{name}: expected an int64 column, got {other:?}"),
    }
}

fn f32_column(ep: &es_data::Episode, name: &str) -> Vec<f32> {
    match ep.columns.get(name) {
        Some(Column::F32(v)) => v.clone(),
        other => panic!("{name}: expected a float32 column, got {other:?}"),
    }
}

// --- Tests ----------------------------------------------------------------------------------

/// Spec 13.2: a collected dataset carries per-frame provenance, and the segments come back
/// exactly as the intervener injected them.
/// Packet M5/V6b (b): the collector's half of "`--seed S` names one scene".
///
/// `Env::new` resets once — randomization draw 0 of `(seed, env, episode)` (spec 6.3) — and
/// every episode ends with exactly one reset, `Env::step`'s own on a terminal condition or the
/// explicit one when the step budget runs out. So episode `i` runs on draw `i`. `es_eval`'s
/// `run_episode` used to reset *again* at the top and ran episode `i` on draw `2i + 1`; V6b
/// removed that reset there, and this pins the invariant on this side so the fix cannot be
/// undone by moving the extra reset over here instead.
///
/// A source scan, in the style of `es_eval`'s own: the property is about which call sites
/// exist, and a behavioural test here would only re-measure what `es_eval`'s
/// `episode_zero_runs_on_the_first_randomization_draw` already measures.
#[test]
fn the_collector_resets_once_per_episode_and_never_before_the_first_observation() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/collect.rs");
    let text = std::fs::read_to_string(&path).expect("src/collect.rs");
    assert_eq!(
        text.matches("Env::new(").count(),
        1,
        "one env per run, and its construction is the first draw"
    );
    assert_eq!(
        text.matches(".reset(None)").count(),
        1,
        "one reset, closing an episode whose step budget ran out -- `Env::step` resets the \
         terminal case itself, so a second call here would draw ahead of `es eval run` \
         (packet M5/V6b)"
    );
    // And it is inside the `None =>` arm that closes an open episode, not before the loop.
    let construction = text.find("Env::new(").expect("the construction");
    let reset = text.find(".reset(None)").expect("the reset");
    assert!(
        reset > construction,
        "nothing resets between `Env::new` and the episode loop"
    );
}

#[test]
fn collect_records_action_source_and_matching_intervention_segments() {
    let root = scratch("loop-collect");
    let report = collect_into(&root, 2, 20_260_913);

    assert_eq!(report.episodes, 2);
    assert_eq!(report.frames, u64::from(STEPS) * 2);
    let per_episode = INJECT.clone().count() as u64;
    assert_eq!(report.intervention_frames, per_episode * 2);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);

    // The segments the report claims are the segments on disk, one per episode.
    let on_disk = es_data::intervention::read_segments(&root).expect("segments read back");
    assert_eq!(on_disk, report.segments);
    assert_eq!(
        on_disk,
        (0..2)
            .map(|e| InterventionSegment::new(
                e,
                *INJECT.start(),
                *INJECT.end(),
                InterventionSource::Scripted
            ))
            .collect::<Vec<_>>()
    );

    let dataset = LeRobotDataset::open(&root).expect("the collected dataset opens");
    assert_eq!(dataset.episodes().len(), 2);
    for index in 0..2 {
        let ep = dataset.read_episode(index).expect("episode reads back");
        assert_eq!(ep.len(), STEPS as usize);
        let sources = i64_column(&ep, ACTION_SOURCE);
        let intervention = i64_column(&ep, INTERVENTION);

        // Exactly {policy, human}: nothing was clamped and no watchdog fired, so the run is
        // evidence that an injected action travels the ordinary path rather than a wider one.
        let mut seen: Vec<i64> = sources.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen,
            vec![
                ActionSourceCode::Policy.as_i64(),
                ActionSourceCode::Human.as_i64()
            ],
            "episode {index} action_source values: {sources:?}"
        );
        for frame in 0..STEPS {
            let human = INJECT.contains(&frame);
            assert_eq!(
                intervention[frame as usize],
                i64::from(human),
                "episode {index} frame {frame}"
            );
            assert_eq!(
                ActionSourceCode::from_i64(sources[frame as usize]),
                Some(if human {
                    ActionSourceCode::Human
                } else {
                    ActionSourceCode::Policy
                }),
                "episode {index} frame {frame}"
            );
        }
    }
}

/// Spec 19.2: labelling moves `dataset_content_hash` and leaves `dataset_schema_hash` alone,
/// because the schema already carries both columns. `action_source` is not the labeller's.
#[test]
fn label_changes_content_and_preserves_schema_and_action_source() {
    let root = scratch("loop-label");
    let before = collect_into(&root, 2, 7);
    let sources_before = i64_column(
        &LeRobotDataset::open(&root)
            .unwrap()
            .read_episode(1)
            .unwrap(),
        ACTION_SOURCE,
    );

    let mut segment = InterventionSegment::new(1, 7, 9, InterventionSource::Teleop);
    segment.operator_id = Some("op-1".to_owned());
    segment.note = Some("corrected the grasp".to_owned());
    let report = es_data::label(&root, &[segment.clone()]).expect("labelling succeeds");

    assert!(
        !report.schema_changed,
        "the schema already had both columns"
    );
    assert_eq!(report.schema, before.schema, "schema_hash must not move");
    assert_ne!(
        report.content_after, report.content_before,
        "content_hash must move: the parquet bytes changed"
    );
    assert_eq!(report.content_before, before.content);
    // Episode 0 keeps its collect-time labels, episode 1 gains three frames.
    assert_eq!(report.episodes, vec![0, 1]);
    assert_eq!(report.frames, INJECT.clone().count() as u64 * 2 + 3);

    let after = LeRobotDataset::open(&root).expect("the labelled dataset opens");
    let ep = after.read_episode(1).expect("episode 1 reads back");
    let intervention = i64_column(&ep, INTERVENTION);
    for frame in 0..STEPS {
        let want = INJECT.contains(&frame) || (7..=9).contains(&frame);
        assert_eq!(
            intervention[frame as usize],
            i64::from(want),
            "frame {frame}"
        );
    }
    assert_eq!(
        i64_column(&ep, ACTION_SOURCE),
        sources_before,
        "label() must not rewrite collection-time provenance"
    );
    // Provenance survives, not merged away.
    let segments = es_data::intervention::read_segments(&root).unwrap();
    assert!(segments.contains(&segment), "{segments:?}");
    assert_eq!(segments.len(), 3);
}

/// Spec 19.2 / 19.3: distilling two datasets sums their episodes, the split is an exact
/// disjoint partition, and the `TrainingIdentity` is the same on a second run.
#[test]
fn distill_merges_splits_disjointly_and_is_stable() {
    let dir = scratch("loop-distill");
    let (a, b) = (dir.join("a"), dir.join("b"));
    collect_into(&a, 3, 11);
    collect_into(&b, 2, 12);

    let split = SplitSpec {
        ratios: [0.6, 0.2, 0.2],
        seed: 20_260_913,
    };
    let out = dir.join("merged");
    let first = es_data::distill(&[a.clone(), b.clone()], &split, &out).expect("distill runs");

    let merged = LeRobotDataset::open(&out).expect("the merged dataset opens");
    assert_eq!(merged.episodes().len(), 5, "episode counts sum");
    assert_eq!(merged.info().total_frames, u64::from(STEPS) * 5);
    let indices: Vec<u32> = merged.episodes().iter().map(|e| e.episode_index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3, 4], "episodes are re-indexed");

    // The segments of both inputs survive, remapped onto the merged indices.
    let segments = es_data::intervention::read_segments(&out).unwrap();
    assert_eq!(segments.len(), 5);
    assert_eq!(
        segments.iter().map(|s| s.episode).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );

    let lists: es_data::Split =
        serde_json::from_str(&std::fs::read_to_string(out.join("split.json")).unwrap()).unwrap();
    let mut all: Vec<u32> = lists
        .train
        .iter()
        .chain(&lists.val)
        .chain(&lists.test)
        .copied()
        .collect();
    let total = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), total, "an episode landed in two splits");
    assert_eq!(all, vec![0, 1, 2, 3, 4], "the split is not a partition");

    // Stable across runs: the merged parquet is byte-identical, so every hash is.
    let again = dir.join("merged2");
    let second = es_data::distill(&[a, b], &split, &again).expect("distill runs again");
    assert_eq!(first, second);
    assert_eq!(
        first.training_hash().unwrap(),
        second.training_hash().unwrap()
    );
    // Only the dataset slot is real; the training run is PyTorch-side (spec 19.3, spec 2.3).
    assert_eq!(first.config, [0; 32]);
    assert_eq!(first.base_model, es_data::BaseModel::default());

    let identity: es_data::TrainingIdentity =
        serde_json::from_str(&std::fs::read_to_string(out.join("training_identity.json")).unwrap())
            .unwrap();
    assert_eq!(identity, first);
}

/// Spec 13.3: the ledger of one dataset root records all three steps, and each step's input
/// hashes are the previous step's output hashes.
#[test]
fn loop_jsonl_chains_collect_intervene_distill() {
    let dir = scratch("loop-ledger");
    let root = dir.join("run");
    let collected = collect_into(&root, 2, 5);
    es_data::label(
        &root,
        &[InterventionSegment::new(
            0,
            6,
            6,
            InterventionSource::Corrective,
        )],
    )
    .expect("labelling succeeds");
    es_data::distill(
        std::slice::from_ref(&root),
        &SplitSpec::default(),
        &dir.join("distilled"),
    )
    .expect("distill runs");

    let steps = read_loop_steps(&root).expect("the ledger reads back");
    assert_eq!(
        steps.iter().map(|s| s.kind).collect::<Vec<_>>(),
        vec![LoopKind::Collect, LoopKind::Intervene, LoopKind::Distill]
    );
    let hex = |d: &[u8; 32]| {
        use std::fmt::Write as _;
        d.iter().fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
    };
    assert_eq!(steps[0].outputs["content"], hex(&collected.content));
    assert_eq!(
        steps[1].inputs["content"], steps[0].outputs["content"],
        "intervene must chain onto collect"
    );
    assert_eq!(
        steps[2].inputs["input.0.content"], steps[1].outputs["content"],
        "distill must chain onto intervene"
    );
    assert!(steps[2].outputs.contains_key("training_hash"));

    // The output root's own ledger records only the step that made it.
    let out_steps = read_loop_steps(&dir.join("distilled")).unwrap();
    assert_eq!(out_steps.len(), 1);
    assert_eq!(out_steps[0].kind, LoopKind::Distill);
}

/// Review `docs/reviews/M3.md` Should-fix, packet `docs/packets/M3/P-M3-R7.md`: a dataset
/// whose `meta/episodes.jsonl` disagrees with its parquet is a reported error, not a panic,
/// on the mask-indexing path in `intervention::label`.
#[test]
fn label_on_mismatched_metadata_is_an_error_not_a_panic() {
    let root = scratch("loop-mismatch");
    collect_into(&root, 1, 11);

    // `meta/episodes.jsonl` now claims episode 0 is shorter than the `STEPS`-row parquet file
    // `collect_into` actually wrote.
    let path = root.join("meta/episodes.jsonl");
    let text = std::fs::read_to_string(&path).unwrap();
    let shorter = format!("\"length\":{}", STEPS - 3);
    let full = format!("\"length\":{STEPS}");
    assert!(text.contains(&full), "fixture episode length: {text}");
    std::fs::write(&path, text.replace(&full, &shorter)).unwrap();

    let err = es_data::label(&root, &[]).expect_err("mismatched metadata must not panic");
    assert!(matches!(err, es_data::DataError::Inconsistent(_)), "{err}");
}

// --- V1: frames, aborted demonstrations, and a clamped scripted action -----------------------

fn no_intervention(_: u32, _: u32, _: &ModelInfo, _: &[f64]) -> Intervention<NJ> {
    Intervention::Policy
}

/// Packet M5/V1: with a frame sink, the collector calls it exactly once per recorded control
/// step, and `info.json`'s video feature stops carrying the "no mp4 was written" warning.
#[test]
fn frames_are_written_once_per_control_step() {
    let root = scratch("loop-frames");
    let b = bundle_with_camera();
    let mut seen: Vec<(u32, f64)> = Vec::new();
    let mut sink = |model: &ModelInfo, state: &StateView<'_>| {
        seen.push((model.nq, state.qpos_of(0)[0]));
        Ok(())
    };
    let report = collect_with(&root, &b, &mut no_intervention, Some(&mut sink));

    assert_eq!(report.rendered, report.frames);
    assert_eq!(seen.len() as u64, report.frames);
    assert!(
        report.warnings.is_empty(),
        "a rendered run has no dangling video reference: {:?}",
        report.warnings
    );
    // The frame is taken from the state the step ended in, not from a constant.
    let moved = seen
        .iter()
        .filter(|(_, q)| (q - seen[0].1).abs() > 1e-12)
        .count();
    assert!(moved > 0, "every frame saw the same state");
}

/// The other half: no sink, and the run is exactly what it was before V1 -- the warning, the
/// declared video feature and the `VideoRef` placeholders all unchanged.
#[test]
fn without_a_renderer_collect_behaves_as_before() {
    let root = scratch("loop-no-frames");
    let b = bundle_with_camera();
    let report = collect_with(&root, &b, &mut no_intervention, None);

    assert_eq!(report.rendered, 0);
    assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    assert!(
        report.warnings[0].contains("no mp4 was written"),
        "{:?}",
        report.warnings
    );
    let dataset = LeRobotDataset::open(&root).expect("the collected dataset opens");
    let ep = dataset.read_episode(0).expect("episode reads back");
    assert!(
        ep.video.contains_key("observation.images.overhead"),
        "the dangling VideoRef is still produced: {:?}",
        ep.video.keys().collect::<Vec<_>>()
    );
}

/// Packet M5/V1: a scripted driver that gives up ends that episode as a failed demonstration,
/// and the episode is still written -- dropping it would hide what the expert cannot do.
#[test]
fn an_unreachable_waypoint_fails_the_episode() {
    let root = scratch("loop-abort");
    let b = bundle();
    let mut hook = |_: u32, frame: u32, _: &ModelInfo, _: &[f64]| {
        if frame >= 3 {
            Intervention::Abort
        } else {
            Intervention::Action([0.1; NJ])
        }
    };
    let report = collect_with(&root, &b, &mut hook, None);

    assert_eq!(
        report.terminations,
        vec![Termination::Failure; 2],
        "an aborted demonstration is a failure, not a timeout"
    );
    let dataset = LeRobotDataset::open(&root).expect("the collected dataset opens");
    assert_eq!(dataset.episodes().len(), 2, "aborted episodes are written");
    let ep = dataset.read_episode(0).expect("episode reads back");
    assert!(ep.len() >= 3 && ep.len() < STEPS as usize, "{}", ep.len());
}

/// Spec 8.6 / 9.3 and `INV-12`: the demonstration is recorded as the actuator saw it. With an
/// envelope too tight for the injected action, the recorded `action` is the clamped value and
/// `action_source` says `Clamped`, not `Human`.
#[test]
fn clamped_expert_actions_are_recorded_clamped() {
    let root = scratch("loop-clamped");
    let mut b = bundle();
    // Tighten the envelope -- never disable the plane (INV-12).
    b.deployment.safety.position = vec![Limit::symmetric(0.3); NJ];
    b.deployment.safety.position_soft_margin = vec![0.0; NJ];
    let report = collect_with(&root, &b, &mut scripted, None);

    let dataset = LeRobotDataset::open(&root).expect("the collected dataset opens");
    let ep = dataset.read_episode(0).expect("episode reads back");
    let sources = i64_column(&ep, ACTION_SOURCE);
    let intervention = i64_column(&ep, INTERVENTION);
    let action = match ep.columns.get("action") {
        Some(Column::F32(v)) => v.clone(),
        other => panic!("action: expected a float32 column, got {other:?}"),
    };
    // The frames the expert drove: none of them may read `Human`, because none of them
    // reached the actuator as the expert asked.
    let injected: Vec<usize> = (0..ep.len()).filter(|f| intervention[*f] == 1).collect();
    assert!(!injected.is_empty(), "nothing was injected");
    for frame in injected {
        assert_eq!(
            sources[frame],
            ActionSourceCode::Clamped.as_i64(),
            "frame {frame}: a clamped demonstration must not be recorded as Human ({sources:?})"
        );
        for (j, v) in action[frame * NJ..(frame + 1) * NJ].iter().enumerate() {
            assert!(
                *v <= 0.3 + 1e-6,
                "frame {frame} joint {j}: recorded {v}, outside the envelope it passed through"
            );
        }
    }
    assert!(report.intervention_frames > 0);
}

/// Spec 13.2 and `INV-12`, packet M5/V1c: **what is executed is what is recorded**. `action`
/// is the `SafeAction` the plane handed toward the actuator; `action_commanded` keeps the raw
/// command it was asked for. With an intervener the envelope cannot follow, the two differ —
/// and only the first one is inside the envelope.
#[test]
fn the_dataset_records_the_executed_action_beside_the_raw_command() {
    const LIMIT: f32 = 0.3;
    let root = scratch("loop-executed");
    let mut b = bundle();
    // Tighten the envelope -- never disable the plane (INV-12).
    b.deployment.safety.position = vec![Limit::symmetric(f64::from(LIMIT)); NJ];
    b.deployment.safety.position_soft_margin = vec![0.0; NJ];
    // Deliberately over-fast: every tick asks for a pose the envelope refuses.
    let mut over_fast =
        |_: u32, _: u32, _: &ModelInfo, _: &[f64]| Intervention::Action([0.9_f64; NJ]);
    collect_with(&root, &b, &mut over_fast, None);

    let dataset = LeRobotDataset::open(&root).expect("the collected dataset opens");
    let ep = dataset.read_episode(0).expect("episode reads back");
    let action = f32_column(&ep, "action");
    let commanded = f32_column(&ep, ACTION_COMMANDED);
    assert_eq!(action.len(), commanded.len(), "one row each, same width");
    assert_eq!(action.len(), ep.len() * NJ);

    let differs = action
        .iter()
        .zip(&commanded)
        .filter(|(a, c)| a.to_bits() != c.to_bits())
        .count();
    assert!(
        differs > 0,
        "the plane corrected nothing, so the two columns cannot be told apart: {action:?}"
    );
    for (i, v) in action.iter().enumerate() {
        assert!(
            v.abs() <= LIMIT + 1e-6,
            "action[{i}] = {v} was recorded outside the envelope it passed through"
        );
    }
    assert!(
        commanded.iter().any(|v| v.abs() > LIMIT + 1e-6),
        "the raw command was recorded already clamped, which loses the provenance"
    );
}

/// The intervener hook runs inside `PolicyRuntime::infer`, which runs when a submitted
/// observation is *released* — `expected_latency_ms` ticks later (§12.3). So `frame == 0` is
/// **not** a hook an intervener may reset itself on: with any declared latency the first call of
/// every episode is frame 1 or later, and a scripted driver keyed on frame 0 never resets at all.
/// It looks correct only because a freshly constructed one starts reset, which is why
/// `es loop collect --episodes N` solved episode 0 and nothing after it (design note section 7.6
/// finding 5). `es loop collect --expert` keys on the episode index instead.
#[test]
fn frame_zero_is_not_a_hook_an_intervener_may_reset_on() {
    let root = scratch("loop-latency-phase");
    let mut b = bundle();
    // One control tick of inference latency, which is what the demo's `learning.toml` declares.
    b.learning.policy.contract.runtime.expected_latency_ms = 1000.0 / CONTROL_HZ as f32;
    let mut seen: Vec<(u32, u32)> = Vec::new();
    let mut record = |episode: u32, frame: u32, _: &ModelInfo, _: &[f64]| {
        seen.push((episode, frame));
        Intervention::Policy
    };
    collect_with(&root, &b, &mut record, None);

    assert!(!seen.is_empty(), "the intervener was never called");
    assert!(
        seen.iter().all(|(_, frame)| *frame != 0),
        "a declared latency must move the first call off frame 0: {seen:?}"
    );
    for episode in 0..2 {
        assert!(
            seen.iter().any(|(e, _)| *e == episode),
            "episode {episode} never reached the intervener: {seen:?}"
        );
    }
}

/// The episode boundary carries nothing (spec 13.1). The fixture task resets to a constant, so
/// with no randomization the second episode of one run is the first, row for row.
///
/// This is the regression for "`es loop collect --episodes N` only solves episode 0" (design
/// note section 7.6 finding 5): what carried across was the Safety Plane's hold target, velocity
/// and rate history, which the collector now re-seeds from the measured state every step.
#[test]
fn a_second_episode_repeats_the_first_exactly() {
    let root = scratch("loop-episode-reset");
    let mut b = bundle();
    // Tight enough that where the plane thinks the robot is decides what it lets through --
    // which is how carried-over plane state becomes a visible difference (INV-12: tightened,
    // never disabled).
    b.deployment.safety.velocity_max = vec![0.5; NJ];
    // The fixture task draws `qpos[0]` uniformly per episode, which is the one thing that may
    // legitimately differ between two episodes. Pin it, and then nothing may.
    b.task.graph.insert(
        NodeId(2),
        TaskNode::ResetState {
            target: "qpos[0]".to_owned(),
            dist: Distribution::Constant(0.25),
            stream: "reset.j0".to_owned(),
        },
    );
    collect_with(&root, &b, &mut no_intervention, None);
    let dataset = LeRobotDataset::open(&root).expect("the collected dataset opens");
    let first = dataset.read_episode(0).expect("episode 0 reads back");
    let second = dataset.read_episode(1).expect("episode 1 reads back");
    assert_eq!(first.len(), second.len(), "same step budget");
    for name in ["action", ACTION_COMMANDED, "observation.state"] {
        assert_eq!(
            f32_column(&first, name),
            f32_column(&second, name),
            "{name}: episode 1 differs from episode 0, so something survived the reset"
        );
    }
    assert_eq!(
        i64_column(&first, ACTION_SOURCE),
        i64_column(&second, ACTION_SOURCE),
        "the plane treated the two identical episodes differently"
    );
}
