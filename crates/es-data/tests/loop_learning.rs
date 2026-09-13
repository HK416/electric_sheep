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
use es_data::collect::{read_loop_steps, CollectSpec, Collector, LoopKind, SplitSpec};
use es_data::intervention::{ActionSourceCode, InterventionSegment, InterventionSource};
use es_data::{Column, LeRobotDataset};
use es_data::{ACTION_SOURCE, INTERVENTION};
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
fn scripted(_episode: u32, frame: u32, _obs: &[f64]) -> Option<[f64; NJ]> {
    INJECT.contains(&frame).then_some([0.7; NJ])
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
    )
    .expect("the fixture collect run succeeds")
}

fn i64_column(ep: &es_data::Episode, name: &str) -> Vec<i64> {
    match ep.columns.get(name) {
        Some(Column::I64(v)) => v.clone(),
        other => panic!("{name}: expected an int64 column, got {other:?}"),
    }
}

// --- Tests ----------------------------------------------------------------------------------

/// Spec 13.2: a collected dataset carries per-frame provenance, and the segments come back
/// exactly as the intervener injected them.
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
