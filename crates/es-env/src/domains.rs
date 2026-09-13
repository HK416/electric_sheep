//! Driving the four batch domains over one [`Env`](crate::Env) (§12.1–12.3, App. B.5).
//!
//! [`Schedule`] says *when* each domain fires and which envs the camera round-robin selects
//! (§12.2); this module is what actually runs on those ticks:
//!
//! ```text
//! observation tick  ──▶  Schedule::observation_envs(t)  ──▶  CpuPlan::run for those envs
//! inference tick    ──▶  submit(env, control_tick)      ──▶  AsyncInference queue
//! every control tick ─▶  poll  ──▶  policy.infer(batch) ──▶  ChunkBuffer::push
//!                    ─▶  ChunkBuffer::next_action ──▶ SafetyPlane::validate ──▶ set_ctrl
//! ```
//!
//! The last line is the only path from a chunk to an actuator. There is no branch that skips
//! the plane, including the underrun branch — an underrun hands the plane an *empty* chunk and
//! the plane answers with the fallback (§8.6, §9.4, `INV-12`).
//!
//! Everything is counted in ticks. The runner's own clock is the **control tick**: one
//! `inference.period` window of simulation ticks, which is exactly what one
//! [`Env::step`](crate::Env::step) advances. Inference latency is a whole number of those
//! ticks (§12.3) — never wall-clock, so a fast host and a slow host replay identically.

use std::collections::BTreeMap;

use es_compile::{CpuPlan, Tensor};
use es_core::{PhysTick, TickRate};
use es_ir::deployment::{ExecutionMode, Micros};
use es_ir::learning::{ActionExecutionMode, ChunkBlendPolicy, PolicyContract};
use es_ir::types::ElemType;
use es_physics_core::backend::{ModelInfo, StateView};
use es_policy::PolicyRuntime;
use es_safety::{ActionChunk, SafetyPlane};

use crate::chunk_buffer::ChunkBuffer;
use crate::inference::{latency_ticks, AsyncInference, Submission};
use crate::scheduler::Schedule;
use crate::EnvError;

/// The latest observation of one env, and whether inference has seen it yet.
#[derive(Clone, Debug)]
struct Latest {
    /// Control tick the observation was produced on — the `computed_from` of App. B.5.
    tick: u64,
    inputs: BTreeMap<String, Tensor>,
    submitted: bool,
}

/// Runs a [`Schedule`] over an [`Env`](crate::Env): camera selection, asynchronous inference,
/// chunk buffers (§12).
///
/// `NJ` is the action dimension and `H` the prediction horizon; both are checked against the
/// `PolicyContract` at construction, so a mismatch is a build error rather than a truncated
/// chunk at run time.
#[derive(Clone, Debug)]
pub struct DomainRunner<const NJ: usize, const H: usize> {
    schedule: Schedule,
    inference: AsyncInference,
    buffers: Vec<ChunkBuffer<NJ, H>>,
    latest: Vec<Option<Latest>>,
    obs_port: String,
    action_port: String,
    mode: ExecutionMode,
    control: TickRate,
    control_tick: u64,
    observations: u64,
    inference_calls: u64,
}

impl<const NJ: usize, const H: usize> DomainRunner<NJ, H> {
    /// `control` is the control rate — the simulation rate divided by `inference.period` — and
    /// it is what turns `RuntimeHints::expected_latency_ms` into whole ticks (§12.3).
    pub fn new(
        schedule: &Schedule,
        contract: &PolicyContract,
        blend: ChunkBlendPolicy,
        control: TickRate,
    ) -> Result<Self, EnvError> {
        if contract.action_dim as usize != NJ {
            return Err(EnvError::shape(
                "action dim",
                NJ,
                contract.action_dim as usize,
            ));
        }
        if contract.horizon as usize != H {
            return Err(EnvError::shape("horizon", H, contract.horizon as usize));
        }
        let domains = *schedule.domains();
        let envs = domains.simulation.batch as usize;
        let k = contract.execute_chunk as usize;
        Ok(Self {
            inference: AsyncInference::new(
                latency_ticks(contract.runtime.expected_latency_ms, control),
                domains.inference.batch,
            ),
            buffers: vec![ChunkBuffer::new(k, blend); envs],
            latest: vec![None; envs],
            obs_port: "state".to_owned(),
            action_port: "action".to_owned(),
            mode: execution_mode(contract.execution_mode, blend),
            schedule: schedule.clone(),
            control,
            control_tick: 0,
            observations: 0,
            inference_calls: 0,
        })
    }

    /// Renames the observation input the runner feeds (default `"state"`) and the policy
    /// output it reads the chunk from (default `"action"`).
    #[must_use]
    pub fn with_ports(mut self, observation: &str, action: &str) -> Self {
        observation.clone_into(&mut self.obs_port);
        action.clone_into(&mut self.action_port);
        self
    }

    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn control_tick(&self) -> u64 {
        self.control_tick
    }

    /// Observation-plan runs so far, summed over envs (`camera_frames_per_sec`'s numerator).
    pub fn observations(&self) -> u64 {
        self.observations
    }

    /// Calls to [`PolicyRuntime::infer`] — batches, not envs (§12.4).
    pub fn inference_calls(&self) -> u64 {
        self.inference_calls
    }

    pub fn inference(&self) -> &AsyncInference {
        &self.inference
    }

    pub fn buffer(&self, env: u32) -> Option<&ChunkBuffer<NJ, H>> {
        self.buffers.get(env as usize)
    }

    /// `chunk_underrun_rate` of §12.4, over every env. `None` before the first control tick.
    pub fn chunk_underrun_rate(&self) -> Option<f64> {
        let under: u64 = self.buffers.iter().map(ChunkBuffer::underruns).sum();
        let served: u64 = self.buffers.iter().map(ChunkBuffer::served).sum();
        (under + served > 0).then(|| under as f64 / (under + served) as f64)
    }

    /// Control period in whole microseconds — the unit `SafetyPlane::validate` wants for the
    /// observation age (§9.4). Integer, never an accumulated float (§18.1).
    fn control_period_us(&self) -> u64 {
        (1_000_000 * self.control.den()).div_ceil(self.control.num())
    }

    /// Observation phase: for each observation tick inside this control window, run the plan
    /// for the envs the round-robin selected (§12.2).
    ///
    /// `plans` is either empty — the raw path, where the observation is the env's `qpos ‖ qvel`
    /// row — or one [`CpuPlan`] per simulation env. Per env, because a plan's `TemporalWindow`
    /// rings are per-env state (§7.5): sharing one plan across a batch would interleave their
    /// histories.
    pub fn observe_window(
        &mut self,
        sim_tick: u64,
        model: &ModelInfo,
        state: &StateView<'_>,
        plans: &mut [CpuPlan],
    ) -> Result<(), EnvError> {
        if !plans.is_empty() && plans.len() != self.buffers.len() {
            return Err(EnvError::shape(
                "observation plans",
                self.buffers.len(),
                plans.len(),
            ));
        }
        let period = u64::from(self.schedule.domains().inference.period);
        for t in sim_tick..sim_tick + period {
            for env in self.schedule.observation_envs(t) {
                let raw = state_row(state, model, env);
                let inputs = match plans.get_mut(env as usize) {
                    None => [(self.obs_port.clone(), raw)].into_iter().collect(),
                    Some(plan) => {
                        let one = [(self.obs_port.clone(), raw.as_ref())]
                            .into_iter()
                            .collect();
                        plan.run(&one)
                            .map_err(|e| EnvError::Unsupported(format!("observation plan: {e}")))?
                    }
                };
                self.latest[env as usize] = Some(Latest {
                    tick: self.control_tick,
                    inputs,
                    submitted: false,
                });
                self.observations += 1;
            }
        }
        Ok(())
    }

    /// Inference phase: submit on inference ticks, then poll and run the released batch.
    ///
    /// Submission order is ascending env id (§12.3) and the queue is FIFO, so which
    /// observations end up in one `infer` call is a pure function of the schedule.
    pub fn infer_window(
        &mut self,
        sim_tick: u64,
        policy: &mut dyn PolicyRuntime,
    ) -> Result<(), EnvError> {
        let period = u64::from(self.schedule.domains().inference.period);
        if (sim_tick..sim_tick + period).any(|t| self.schedule.at(t).inference) {
            for env in 0..self.latest.len() {
                if let Some(obs) = &mut self.latest[env] {
                    if !obs.submitted {
                        obs.submitted = true;
                        self.inference
                            .submit(env as u32, self.control_tick, obs.inputs.clone());
                    }
                }
            }
        }
        let ready = self.inference.poll(self.control_tick);
        if ready.is_empty() {
            return Ok(());
        }
        let batched = stack(&ready)?;
        let outputs = policy
            .infer(&batched)
            .map_err(|e| EnvError::Task(format!("policy inference: {e}")))?;
        self.inference_calls += 1;
        let chunks = split::<NJ, H>(&outputs, &self.action_port, ready.len())?;
        let latency = self.inference.latency_ticks();
        for (sub, actions) in ready.iter().zip(chunks) {
            // App. B.5: `apply_at = computed_from + deterministic latency`, not the arrival
            // tick. A batch that backed up delivers late; rows already in the past are simply
            // never served, which is visible as an underrun rather than as a time shift.
            let apply_at = sub.submit_tick.saturating_add(latency);
            self.buffers[sub.env as usize].push(&ActionChunk::new(actions, H, self.mode), apply_at);
        }
        Ok(())
    }

    /// Action phase: chunk buffer → Safety Plane → `ctrl`. The only actuator path (`INV-12`).
    ///
    /// One plane per env: the plane's hold target, rate-limit history and latch are per-robot
    /// state (§9.3), so a shared plane would blend one env's history into another's limits.
    pub fn emit_actions(
        &mut self,
        now: PhysTick,
        planes: &mut [SafetyPlane<NJ, H>],
        ctrl: &mut [f64],
    ) -> Result<(), EnvError> {
        let envs = self.buffers.len();
        if planes.len() != envs {
            return Err(EnvError::shape("safety planes", envs, planes.len()));
        }
        if ctrl.len() != envs * NJ {
            return Err(EnvError::shape("ctrl", envs * NJ, ctrl.len()));
        }
        let period_us = self.control_period_us();
        for env in 0..envs {
            let chunk = match self.buffers[env].next_action(self.control_tick) {
                Some(row) => {
                    let mut actions = [[0.0; NJ]; H];
                    actions[0] = row;
                    ActionChunk::new(actions, 1, self.mode)
                }
                // Underrun: an empty chunk, so the plane produces the fallback (§8.6, §9.4).
                // Never a fabricated action.
                None => ActionChunk::empty(self.mode),
            };
            let age = self.latest[env]
                .as_ref()
                .map_or(self.control_tick + 1, |o| self.control_tick - o.tick);
            let safe = planes[env].validate(&chunk, Micros(age.saturating_mul(period_us)), now);
            ctrl[env * NJ..(env + 1) * NJ].copy_from_slice(&safe.q);
        }
        Ok(())
    }

    /// The episode of `env` ended: its chunks and queued inference describe a state that no
    /// longer exists (§13.1).
    pub fn reset_env(&mut self, env: u32) {
        if let Some(b) = self.buffers.get_mut(env as usize) {
            b.clear();
        }
        if let Some(l) = self.latest.get_mut(env as usize) {
            *l = None;
        }
        self.inference.drop_env(env);
    }

    /// Closes the control tick.
    pub fn advance(&mut self) {
        self.control_tick += 1;
    }
}

/// `ActionChunk`'s execution mode, from the Learning IR's mode plus the blend's decay.
fn execution_mode(mode: ActionExecutionMode, blend: ChunkBlendPolicy) -> ExecutionMode {
    match mode {
        ActionExecutionMode::OpenLoopChunk => ExecutionMode::OpenLoopChunk,
        ActionExecutionMode::RecedingHorizon => ExecutionMode::RecedingHorizon,
        ActionExecutionMode::RealTimeChunking => ExecutionMode::RealTimeChunking,
        ActionExecutionMode::TemporalEnsemble => ExecutionMode::TemporalEnsemble {
            decay: match blend {
                ChunkBlendPolicy::TemporalEnsemble { weight_decay } => f64::from(weight_decay),
                _ => 0.0,
            },
        },
    }
}

/// One env's `qpos ‖ qvel` row as an `f32` tensor — the raw observation of the plan-free path.
///
/// A real camera observation is a sensor-domain concern (§7, M2 W3); this wave owns *when*
/// observations happen and for which envs, not what a pixel is.
fn state_row(state: &StateView<'_>, model: &ModelInfo, env: u32) -> Tensor {
    let slice = |v: &[f64], width: u32| -> Vec<f64> {
        let start = env as usize * width as usize;
        v.get(start..start + width as usize).unwrap_or(&[]).to_vec()
    };
    let values: Vec<f64> = slice(state.qpos, model.nq)
        .into_iter()
        .chain(slice(state.qvel, model.nv))
        .collect();
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in &values {
        data.extend_from_slice(&(*v as f32).to_le_bytes());
    }
    Tensor {
        dtype: ElemType::F32,
        shape: vec![values.len() as u64],
        data,
    }
}

/// Stacks a released batch along a new leading axis, in release order (§12.1 inference batch).
fn stack(ready: &[Submission]) -> Result<BTreeMap<String, Tensor>, EnvError> {
    let first = &ready[0].inputs;
    let mut out = BTreeMap::new();
    for (name, t0) in first {
        let mut data = Vec::with_capacity(t0.data.len() * ready.len());
        for sub in ready {
            let t = sub.inputs.get(name).ok_or_else(|| {
                EnvError::Task(format!("env {} has no input \"{name}\"", sub.env))
            })?;
            if t.dtype != t0.dtype || t.shape != t0.shape {
                return Err(EnvError::Task(format!(
                    "input \"{name}\" is not batchable: env {} disagrees on dtype or shape",
                    sub.env
                )));
            }
            data.extend_from_slice(&t.data);
        }
        let mut shape = vec![ready.len() as u64];
        shape.extend_from_slice(&t0.shape);
        out.insert(
            name.clone(),
            Tensor {
                dtype: t0.dtype,
                shape,
                data,
            },
        );
    }
    Ok(out)
}

/// Splits the policy's `[B, H, NJ]` output (or `[H, NJ]` when `B == 1`) into per-env chunks.
fn split<const NJ: usize, const H: usize>(
    outputs: &BTreeMap<String, Tensor>,
    port: &str,
    batch: usize,
) -> Result<Vec<[[f64; NJ]; H]>, EnvError> {
    let t = outputs
        .get(port)
        .ok_or_else(|| EnvError::Task(format!("the policy has no output named \"{port}\"")))?;
    let want = [batch as u64, H as u64, NJ as u64];
    let ok = t.shape == want || (batch == 1 && t.shape == want[1..]);
    if !ok {
        return Err(EnvError::Task(format!(
            "policy output \"{port}\": expected {want:?}, got {:?}",
            t.shape
        )));
    }
    let width = match t.dtype {
        ElemType::F32 => 4,
        ElemType::F64 => 8,
        other => {
            return Err(EnvError::Unsupported(format!(
                "policy output \"{port}\" of dtype {other:?}"
            )))
        }
    };
    let n = batch * H * NJ;
    if t.data.len() != n * width {
        return Err(EnvError::shape(
            "policy output bytes",
            n * width,
            t.data.len(),
        ));
    }
    let value = |i: usize| -> f64 {
        let b = &t.data[i * width..(i + 1) * width];
        if width == 4 {
            f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        } else {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        }
    };
    let mut out = Vec::with_capacity(batch);
    for env in 0..batch {
        let mut rows = [[0.0; NJ]; H];
        for (h, row) in rows.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = value((env * H + h) * NJ + j);
            }
        }
        out.push(rows);
    }
    Ok(out)
}

// --- Sizing -------------------------------------------------------------------------------

/// The §12.4 sizing arithmetic for a batch-domain configuration.
///
/// Integer-only and allocation-free, so the M2 W2 gate configuration (4,096 sim env × 512 obs
/// env) can be *costed* on any machine even where it cannot be run. Every number it produces
/// is a budget, not a measurement — report them as `Target / Status: unverified` (§12.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainSizing {
    pub sim_envs: u64,
    /// Envs whose cameras are on per observation tick (`round_robin` k of §12.2).
    pub obs_envs: u64,
    pub views: u64,
    pub width: u64,
    pub height: u64,
    pub camera_hz: u64,
    pub action_dim: u64,
    pub horizon: u64,
    /// Chunks kept per env — [`crate::chunk_buffer::CHUNK_SLOTS`].
    pub chunk_slots: u64,
}

impl DomainSizing {
    /// The §12.4 reference configuration: 4,096 env × 2 cameras × 224×224 RGB × 30 Hz, with
    /// the round-robin cut to 512 observation envs.
    pub const GATE: Self = Self {
        sim_envs: 4096,
        obs_envs: 512,
        views: 2,
        width: 224,
        height: 224,
        camera_hz: 30,
        action_dim: 7,
        horizon: 20,
        chunk_slots: crate::chunk_buffer::CHUNK_SLOTS as u64,
    };

    /// Round-robin cycle length: how many observation ticks it takes to visit every env
    /// (§12.2).
    pub fn round_robin_cycle(self) -> u64 {
        self.sim_envs.div_ceil(self.obs_envs.max(1))
    }

    /// Rendered frames per second, `env × view × hz` (§12.4 `camera_frames_per_sec`).
    pub fn camera_frames_per_sec(self) -> u64 {
        self.obs_envs * self.views * self.camera_hz
    }

    pub fn pixels_per_sec(self) -> u64 {
        self.camera_frames_per_sec() * self.width * self.height
    }

    /// Render output only, RGB8.
    pub fn render_bytes_per_sec(self) -> u64 {
        self.pixels_per_sec() * 3
    }

    /// After the `f32` normalization the Observation IR inserts — the intermediate tensor, and
    /// the number that actually sizes the bus (§12.4).
    pub fn observation_bytes_per_sec(self) -> u64 {
        self.render_bytes_per_sec() * 4
    }

    /// Chunk buffers for the whole simulation batch: `f64[H][NJ]` per slot, per env.
    pub fn chunk_buffer_bytes(self) -> u64 {
        self.sim_envs * self.chunk_slots * self.horizon * self.action_dim * 8
    }

    /// What the same configuration would cost with every camera on (`EnvSelection::All`) —
    /// the number §12.2 exists to avoid.
    #[must_use]
    pub fn without_round_robin(self) -> Self {
        Self {
            obs_envs: self.sim_envs,
            ..self
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::env::tests::{fake_scene, task_with, FakeBackend};
    use crate::scheduler::{BatchDomains, DomainCfg};
    use crate::{Env, EnvError};
    use es_ir::deployment::{
        ActionContract, ActionSpace, Deadlines, DeploymentIr, FallbackPolicy, Limit, RateLimit,
        RateSpec, RobotRef, RobotTarget, SafetyEnvelope, Watchdog, WatchdogSet, Workspace,
    };
    use es_ir::learning::{LearningGraph, RuntimeHints};
    use es_ir::task::{Distribution, JointQuantity, TaskNode};
    use es_policy::{PolicyError, PolicyInfo, WeightsSource};

    const NJ: usize = 1;
    const H: usize = 4;

    /// A policy that ignores its input and returns the same chunk every time, so any
    /// difference between two runs comes from the runtime, never from the model.
    #[derive(Debug, Default)]
    struct FakePolicy {
        calls: u64,
    }

    impl PolicyRuntime for FakePolicy {
        fn load(
            &mut self,
            _graph: &LearningGraph,
            _weights: &WeightsSource,
        ) -> Result<PolicyInfo, PolicyError> {
            Err(PolicyError::Unavailable("fake policy loads nothing".into()))
        }

        fn infer(
            &mut self,
            inputs: &BTreeMap<String, Tensor>,
        ) -> Result<BTreeMap<String, Tensor>, PolicyError> {
            self.calls += 1;
            let batch = inputs
                .values()
                .next()
                .and_then(|t| t.shape.first().copied())
                .unwrap_or(1) as usize;
            let mut data = Vec::new();
            for _ in 0..batch {
                for h in 0..H {
                    for _ in 0..NJ {
                        data.extend_from_slice(&(0.02 * (h + 1) as f32).to_le_bytes());
                    }
                }
            }
            Ok([(
                "action".to_owned(),
                Tensor {
                    dtype: ElemType::F32,
                    shape: vec![batch as u64, H as u64, NJ as u64],
                    data,
                },
            )]
            .into_iter()
            .collect())
        }

        fn info(&self) -> Option<&PolicyInfo> {
            None
        }

        fn runtime_hash(&self) -> [u8; 32] {
            [7; 32]
        }
    }

    fn contract() -> PolicyContract {
        PolicyContract {
            inputs: BTreeMap::new(),
            observation_window: 1,
            action_dim: NJ as u32,
            horizon: H as u32,
            execute_chunk: H as u32,
            replanning_hz: 62.5,
            execution_mode: ActionExecutionMode::RecedingHorizon,
            runtime: RuntimeHints {
                dtype: ElemType::F32,
                expected_latency_ms: 8.0,
                deadline_ms: 100.0,
            },
        }
    }

    /// A 1-joint arm with an envelope wide enough that the policy's action passes through
    /// unchanged. Widening the envelope is the sanctioned way to keep a test's actions legal;
    /// disabling the plane is not (`INV-12`).
    fn deployment_ir() -> DeploymentIr {
        DeploymentIr {
            schema_version: es_ir::deployment::SCHEMA_VERSION,
            robot: RobotRef {
                name: "fixture".into(),
                target: RobotTarget::Simulated {
                    scene: "fixture.xml".into(),
                },
                n_joints: NJ,
            },
            action: ActionContract {
                space: ActionSpace::JointPosition,
                dim: NJ,
                horizon: H,
                execute_chunk: H,
            },
            safety: SafetyEnvelope {
                position: vec![Limit::symmetric(1.0e3); NJ],
                position_soft_margin: vec![1.0; NJ],
                velocity_max: vec![1.0e6; NJ],
                acceleration_max: vec![1.0e9; NJ],
                torque_max: vec![1.0e6; NJ],
                jerk_max: None,
                action_rate: RateLimit {
                    first_diff_max: vec![1.0e3; NJ],
                    second_diff_max: vec![1.0e3; NJ],
                },
                workspace: Workspace::Box {
                    min: [-100.0; 3],
                    max: [100.0; 3],
                },
                ee_velocity_max: 1.0e3,
                min_self_distance: 0.001,
                min_env_distance: 0.001,
                contact_force_max: 1.0e6,
            },
            execution: ExecutionMode::RecedingHorizon,
            deadlines: Deadlines {
                observation_age: Micros(10_000_000),
                inference_budget: Micros(3_000),
                actuation_budget: Micros(500),
            },
            watchdogs: WatchdogSet(vec![Watchdog::ChunkUnderrun]),
            fallback: FallbackPolicy::HoldPosition,
            rate: RateSpec {
                control: TickRate::hz(250),
                inference: TickRate::hz(250),
            },
        }
    }

    fn task() -> es_ir::task::TaskIr {
        use es_ir::graph::NodeId;
        let mut t = task_with(&[
            TaskNode::GetJointState {
                body: fake_scene().bodies[0].id,
                joints: vec!["hinge".to_owned()],
                quantity: JointQuantity::Position,
            },
            TaskNode::Reward {
                name: "angle".to_owned(),
                weight: 1.0,
                aggregation: es_ir::task::Aggregation::Sum,
                ty: es_ir::types::PortType {
                    elem: ElemType::F32,
                    shape: es_ir::types::Shape::new([1]),
                    unit: es_ir::types::Unit::Angle,
                    frame: es_ir::types::Frame::World,
                    time: es_ir::types::TimeRef::Tick,
                    image: None,
                },
            },
            // A constant reset, so every env is physically identical and the only thing that
            // can separate two envs is when the round-robin observed them.
            TaskNode::ResetState {
                target: "qpos[0]".to_owned(),
                dist: Distribution::Constant(0.1),
                stream: "reset.hinge".to_owned(),
            },
        ]);
        t.graph.connect(NodeId(0), "value", NodeId(1), "value");
        t
    }

    /// `sim = 16`, `observation.batch = obs_batch`, everything on a 4-tick control period, so
    /// every configuration advances the same physics per control step and only the camera
    /// round-robin differs.
    fn domains(obs_batch: u32) -> BatchDomains {
        BatchDomains {
            simulation: DomainCfg::new(16, 1),
            observation: DomainCfg::new(obs_batch, 4),
            inference: DomainCfg::new(obs_batch, 4),
            training: None,
        }
    }

    /// Runs `steps` control steps and returns each env's reward trace.
    fn run(obs_batch: u32, steps: usize) -> Vec<Vec<f64>> {
        let task = task();
        let mut env = Env::new(
            &task,
            &fake_scene(),
            FakeBackend::new(),
            &domains(obs_batch),
            5,
        )
        .expect("the fixture compiles");
        let mut runner = DomainRunner::<NJ, H>::new(
            env.schedule(),
            &contract(),
            ChunkBlendPolicy::TemporalEnsemble { weight_decay: 0.01 },
            TickRate::hz(250),
        )
        .expect("the contract matches NJ and H");
        let plane = SafetyPlane::<NJ, H>::from_ir(&deployment_ir()).expect("a valid envelope");
        let mut planes = vec![plane; 16];
        let mut policy = FakePolicy::default();
        let mut traces: Vec<Vec<f64>> = (0..16).map(|_| Vec::with_capacity(steps)).collect();
        for _ in 0..steps {
            let out = env
                .step_with_policy(&mut runner, &mut policy, &mut planes, &mut [])
                .expect("a control step");
            for (i, r) in out.rewards.iter().enumerate() {
                traces[i].push(*r);
            }
        }
        assert!(runner.inference_calls() > 0, "the policy was actually run");
        traces
    }

    /// Control steps on which the round-robin observes `env`, per the schedule alone.
    fn observation_steps(obs_batch: u32, steps: usize) -> Vec<Vec<u64>> {
        let s = crate::Schedule::build(&domains(obs_batch)).unwrap();
        let period = u64::from(s.domains().inference.period);
        (0..16)
            .map(|env| {
                (0..steps as u64)
                    .filter(|c| {
                        (c * period..(c + 1) * period)
                            .any(|t| s.observation_envs(t).contains(&(env as u32)))
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_run_is_bitwise_identical_to_itself_at_every_batch_size() {
        for obs_batch in [4, 8, 16] {
            assert_eq!(
                run(obs_batch, 200),
                run(obs_batch, 200),
                "observation.batch {obs_batch} does not replay"
            );
        }
    }

    /// The precise batch-domain invariant (§12.2, §12.3).
    ///
    /// Round-robin changes **which** envs observe **when**, and a policy chunk only reaches an
    /// env that was observed, so per-env trajectories are *not* independent of
    /// `observation.batch` — an env observed half as often underruns twice as often and gets
    /// the Safety Plane's fallback in between (§8.6). What *is* invariant: two envs whose
    /// observation ticks coincide are indistinguishable. That is asserted here by grouping the
    /// envs of one configuration by their schedule-derived observation ticks.
    #[test]
    fn envs_with_the_same_observation_ticks_have_the_same_trajectory() {
        for obs_batch in [4, 8, 16] {
            let traces = run(obs_batch, 60);
            let ticks = observation_steps(obs_batch, 60);
            let mut groups: BTreeMap<Vec<u64>, Vec<usize>> = BTreeMap::new();
            for (env, t) in ticks.iter().enumerate() {
                groups.entry(t.clone()).or_default().push(env);
            }
            assert_eq!(
                groups.len(),
                (16 / obs_batch) as usize,
                "one group per round-robin slot, batch {obs_batch}"
            );
            for envs in groups.values() {
                for e in envs {
                    assert_eq!(traces[*e], traces[envs[0]], "env {e}, batch {obs_batch}");
                }
            }
            // ... and the groups are genuinely different runs, not a degenerate all-equal.
            if obs_batch < 16 {
                let first: Vec<&Vec<usize>> = groups.values().collect();
                assert_ne!(
                    traces[first[0][0]], traces[first[1][0]],
                    "different observation ticks must give different trajectories"
                );
            }
        }
    }

    /// Full batch-independence does not hold, and this pins why: a smaller observation batch
    /// observes each env less often, so it underruns more.
    #[test]
    fn a_smaller_observation_batch_underruns_more() {
        let rate = |obs_batch: u32| {
            let task = task();
            let mut env = Env::new(
                &task,
                &fake_scene(),
                FakeBackend::new(),
                &domains(obs_batch),
                5,
            )
            .unwrap();
            let mut runner = DomainRunner::<NJ, H>::new(
                env.schedule(),
                &contract(),
                ChunkBlendPolicy::HardSwitch,
                TickRate::hz(250),
            )
            .unwrap();
            let mut planes = vec![SafetyPlane::<NJ, H>::from_ir(&deployment_ir()).unwrap(); 16];
            let mut policy = FakePolicy::default();
            for _ in 0..60 {
                env.step_with_policy(&mut runner, &mut policy, &mut planes, &mut [])
                    .unwrap();
            }
            runner.chunk_underrun_rate().expect("ticks were served")
        };
        let (r4, r8, r16) = (rate(4), rate(8), rate(16));
        assert!(r4 > r8 && r8 > r16, "{r4} {r8} {r16}");
        assert!(
            r16 < 0.2,
            "a fully observed batch mostly has a chunk: {r16}"
        );
        assert_ne!(run(4, 60), run(16, 60), "trajectories are not batch-free");
    }

    #[test]
    fn the_contract_must_match_the_const_generics() {
        let s = crate::Schedule::build(&domains(4)).unwrap();
        let blend = ChunkBlendPolicy::HardSwitch;
        let err = DomainRunner::<3, H>::new(&s, &contract(), blend, TickRate::hz(250)).unwrap_err();
        assert!(err.to_string().contains("action dim"), "{err}");
        let err =
            DomainRunner::<NJ, 9>::new(&s, &contract(), blend, TickRate::hz(250)).unwrap_err();
        assert!(err.to_string().contains("horizon"), "{err}");
    }

    #[test]
    fn a_plane_array_of_the_wrong_length_is_refused() {
        let task = task();
        let mut env = Env::new(&task, &fake_scene(), FakeBackend::new(), &domains(4), 5).unwrap();
        let mut runner = DomainRunner::<NJ, H>::new(
            env.schedule(),
            &contract(),
            ChunkBlendPolicy::HardSwitch,
            TickRate::hz(250),
        )
        .unwrap();
        let mut planes = vec![SafetyPlane::<NJ, H>::from_ir(&deployment_ir()).unwrap(); 2];
        let mut policy = FakePolicy::default();
        let err = env
            .step_with_policy(&mut runner, &mut policy, &mut planes, &mut [])
            .unwrap_err();
        assert!(matches!(err, EnvError::ShapeMismatch { .. }), "{err}");
    }

    #[test]
    fn a_policy_output_of_the_wrong_shape_is_named_not_guessed() {
        let out: BTreeMap<String, Tensor> = [(
            "action".to_owned(),
            Tensor {
                dtype: ElemType::F32,
                shape: vec![2, 3, 1],
                data: vec![0; 24],
            },
        )]
        .into_iter()
        .collect();
        let err = split::<NJ, H>(&out, "action", 2).unwrap_err();
        assert!(err.to_string().contains("expected [2, 4, 1]"), "{err}");
        let err = split::<NJ, H>(&out, "missing", 2).unwrap_err();
        assert!(err.to_string().contains("no output named"), "{err}");
    }

    /// The M2 W2 gate configuration, costed rather than run (§28.4, §12.4).
    ///
    /// 4,096 sim env × 512 obs env does not fit on a development machine, so the gate number
    /// this crate can honestly produce is the budget. `Target / Status: unverified`.
    #[test]
    fn the_gate_configuration_is_sized_without_being_allocated() {
        let g = DomainSizing::GATE;
        assert_eq!(g.round_robin_cycle(), 8, "512 envs, 8 slots, 4096 covered");
        assert_eq!(g.camera_frames_per_sec(), 30_720);
        assert_eq!(g.pixels_per_sec(), 1_541_406_720);
        // 4.6 GB/s of RGB8, 18.5 GB/s once normalized to f32 — the §12.4 figure.
        assert_eq!(g.render_bytes_per_sec(), 4_624_220_160);
        assert_eq!(g.observation_bytes_per_sec(), 18_496_880_640);
        // Chunk buffers for the whole simulation batch: 4096 × 8 × 20 × 7 × 8 B = 36.7 MB.
        assert_eq!(g.chunk_buffer_bytes(), 36_700_160);

        // Every camera on is the §12.4 worked example: 245,760 frames/s, 12.3 Gpixel/s,
        // 37 GB/s of render output and 148 GB/s of preprocessed tensor — which is why
        // round_robin exists (§12.2).
        let all = g.without_round_robin();
        assert_eq!(all.round_robin_cycle(), 1);
        assert_eq!(all.camera_frames_per_sec(), 245_760);
        assert_eq!(all.pixels_per_sec(), 12_331_253_760);
        assert_eq!(all.render_bytes_per_sec(), 36_993_761_280);
        assert_eq!(all.observation_bytes_per_sec(), 147_975_045_120);
        assert_eq!(
            all.observation_bytes_per_sec() / g.observation_bytes_per_sec(),
            8
        );
    }
}
