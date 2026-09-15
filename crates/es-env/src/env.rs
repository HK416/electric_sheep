//! The environment runtime: reset, step, reward, termination, recording (§6.4, §12, §18.5).
//!
//! One `Env::step` advances the batch by one **control** step — `inference.period` simulation
//! ticks (§12.1) — and runs §6.4's phase order: physics, then reward, then termination, then
//! record. Envs that terminated are reset at the end of the same call, so an episode boundary
//! is always a tick boundary. The recorded row's *state* is the one the step was entered with
//! — the state its `ctrl` was computed from — while its reward, termination and failure are
//! the transition's (packet M5/V12).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use es_assets::scene::SceneDesc;
use es_compile::CpuPlan;
use es_core::{EnvHealth, FailureAction, FailureKind, FailurePolicy, PhysTick, SimTime};
use es_ir::task::TaskIr;
use es_physics_core::backend::{LoadConfig, ModelInfo, PhysicsBackend, StateView};
use es_policy::PolicyRuntime;
use es_safety::SafetyPlane;

use crate::control::ControlExecutor;
use crate::domains::DomainRunner;
use crate::episode::{self, Episode, EpisodeRecorder, EpisodeShape, StepRow, Termination};
use crate::plan::{ScalarPlan, Source};
use crate::randomize::{ParamScales, RandomizationPlan, ResetBuffer};
use crate::scheduler::{BatchDomains, Schedule};
use crate::EnvError;

/// The §12.4 metric set. Every rate is `Option` — a metric nobody measured is `None`, never a
/// fabricated zero — and there is deliberately **no `step/s` field**: simulation throughput is
/// not learning throughput (§12.4).
///
/// `es-env` is layer 9 and `es-telemetry` is layer 10, so this is a plain struct the telemetry
/// crate converts into its `PerfMetrics`, not a dependency on it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnvMetrics {
    pub physics_steps_per_sec: Option<f64>,
    pub camera_frames_per_sec: Option<f64>,
    pub pixels_per_sec: Option<f64>,
    pub observation_gb_per_sec: Option<f64>,
    pub policy_inferences_per_sec: Option<f64>,
    pub actions_per_sec: Option<f64>,
    pub p50_end_to_end_latency: Option<f64>,
    pub p95_end_to_end_latency: Option<f64>,
    pub gpu_memory_peak: Option<f64>,
    pub chunk_underrun_rate: Option<f64>,
    /// Raw counters behind the rates: env-ticks of physics, control steps, and the wall time
    /// spent inside the backend. Integer-only, so nothing accumulates in a float (§18.1).
    pub physics_env_steps: u64,
    pub action_env_steps: u64,
    pub simulation_wall: Duration,
}

/// What one control step produced, per env.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepOutcome {
    pub rewards: Vec<f64>,
    pub dones: Vec<bool>,
    pub failures: Vec<Option<FailureKind>>,
    /// Episodes closed by this step, in ascending env order (§13.1).
    pub episodes: Vec<Episode>,
}

/// A batch of environments driven by one physics backend.
pub struct Env<B: PhysicsBackend> {
    backend: B,
    schedule: Schedule,
    model: ModelInfo,
    scalar: ScalarPlan,
    /// IR-C (spec 6.2). `None` is the IR-D-only task, and then nothing below changes.
    control: Option<ControlExecutor>,
    randomization: RandomizationPlan,
    recorder: EpisodeRecorder,
    failure_policy: FailurePolicy,
    seed: u64,
    max_episode_steps: u32,
    n_envs: u32,
    tick: PhysTick,
    episode: Vec<u64>,
    steps: Vec<u32>,
    health: Vec<EnvHealth>,
    /// Scratch, allocated once: reset state rows, the last control vector, and the port map.
    reset_qpos: Vec<f64>,
    reset_qvel: Vec<f64>,
    last_ctrl: Vec<f64>,
    /// Scratch, allocated once: the state one control step is entered with, which is the state
    /// the recorded row carries (§13.2, packet M5/V12).
    pre: PreStep,
    ports: BTreeMap<String, f64>,
    metrics: EnvMetrics,
}

impl<B: PhysicsBackend> std::fmt::Debug for Env<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env")
            .field("n_envs", &self.n_envs)
            .field("tick", &self.tick)
            .field("episode", &self.episode)
            .field("health", &self.health)
            .finish_non_exhaustive()
    }
}

impl<B: PhysicsBackend> Env<B> {
    /// Loads `scene` into `backend`, compiles the task's plans against the loaded model, and
    /// resets every env.
    pub fn new(
        task: &TaskIr,
        scene: &SceneDesc,
        mut backend: B,
        domains: &BatchDomains,
        seed: u64,
    ) -> Result<Self, EnvError> {
        let schedule = Schedule::build(domains)?;
        let n_envs = domains.simulation.batch;
        let model = backend.load(
            scene,
            &LoadConfig {
                n_envs,
                rate: None,
                seed,
            },
        )?;
        if model.n_envs != n_envs {
            return Err(EnvError::shape(
                "backend env count",
                n_envs as usize,
                model.n_envs as usize,
            ));
        }
        let shape = EpisodeShape {
            nq: model.nq as usize,
            nv: model.nv as usize,
            nu: model.nu as usize,
            nsensordata: model.nsensordata as usize,
        };
        let envs = n_envs as usize;
        let mut env = Self {
            scalar: ScalarPlan::compile(task, scene, &model)?,
            control: ControlExecutor::new(task, n_envs),
            randomization: RandomizationPlan::compile(task, scene, &model)?,
            recorder: EpisodeRecorder::new(n_envs, shape, task.config.max_episode_steps),
            failure_policy: FailurePolicy::default(),
            seed,
            max_episode_steps: task.config.max_episode_steps,
            n_envs,
            tick: PhysTick::ZERO,
            episode: vec![0; envs],
            steps: vec![0; envs],
            health: vec![EnvHealth::Ok; envs],
            reset_qpos: vec![0.0; envs * shape.nq],
            reset_qvel: vec![0.0; envs * shape.nv],
            last_ctrl: vec![0.0; envs * shape.nu],
            pre: PreStep::default(),
            ports: BTreeMap::new(),
            metrics: EnvMetrics::default(),
            model,
            schedule,
            backend,
        };
        env.reset(None)?;
        Ok(env)
    }

    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn model(&self) -> &ModelInfo {
        &self.model
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn tick(&self) -> PhysTick {
        self.tick
    }

    pub fn health(&self) -> &[EnvHealth] {
        &self.health
    }

    /// Replaces the default failure policy (§18.5).
    pub fn set_failure_policy(&mut self, policy: FailurePolicy) {
        self.failure_policy = policy;
    }

    /// Resets `envs` (all of them when `None`): bumps the episode counter, draws the reset and
    /// randomization distributions, and pushes the new state into the backend.
    ///
    /// Episodes that were open on the reset envs are closed and returned.
    pub fn reset(&mut self, envs: Option<&[u32]>) -> Result<Vec<Episode>, EnvError> {
        let list: Vec<u32> = match envs {
            Some(e) => {
                if let Some(bad) = e.iter().find(|i| **i >= self.n_envs) {
                    return Err(EnvError::Task(format!("env {bad} is out of range")));
                }
                e.to_vec()
            }
            None => (0..self.n_envs).collect(),
        };
        let (nq, nv) = (self.model.nq as usize, self.model.nv as usize);
        let mut closed = Vec::with_capacity(list.len());
        let mut rows_qpos = Vec::with_capacity(list.len() * nq);
        let mut rows_qvel = Vec::with_capacity(list.len() * nv);

        for env in &list {
            let i = *env as usize;
            self.steps[i] = 0;
            self.health[i] = EnvHealth::Ok;
            let (qpos, qvel) = (
                &mut self.reset_qpos[i * nq..(i + 1) * nq],
                &mut self.reset_qvel[i * nv..(i + 1) * nv],
            );
            qpos.fill(0.0);
            qvel.fill(0.0);
            let mut scales = ParamScales::new();
            self.randomization.apply(
                self.seed,
                *env,
                self.episode[i],
                &mut ResetBuffer {
                    qpos,
                    qvel,
                    scales: &mut scales,
                },
            );
            self.episode[i] += 1;
            if let Some(control) = self.control.as_mut() {
                control.reset_env(*env);
            }
            rows_qpos.extend_from_slice(qpos);
            rows_qvel.extend_from_slice(qvel);
            if self.recorder.open(*env).steps() > 0 {
                closed.push(self.recorder.finish(*env));
            }
            self.recorder.set_param_scales(*env, scales);
            self.last_ctrl[i * self.model.nu as usize..(i + 1) * self.model.nu as usize].fill(0.0);
        }

        let state = StateView {
            n_envs: list.len() as u32,
            tick: self.tick,
            qpos: &rows_qpos,
            qvel: &rows_qvel,
            ..StateView::default()
        };
        self.backend.reset(envs, Some(&state))?;
        Ok(closed)
    }

    /// Advances every env by one control step: `inference.period` simulation ticks.
    ///
    /// `ctrl` is `n_envs * nu`, row-major by env.
    pub fn step(&mut self, ctrl: &[f64]) -> Result<StepOutcome, EnvError> {
        let nu = self.model.nu as usize;
        let expected = self.n_envs as usize * nu;
        if ctrl.len() != expected {
            return Err(EnvError::shape("ctrl", expected, ctrl.len()));
        }
        self.last_ctrl.copy_from_slice(ctrl);
        self.backend.set_ctrl(ctrl)?;
        // The row this step records is the state the step is *entered* with, because that is
        // the state `ctrl` was computed from -- by `step_with_policy`'s observe-infer-act
        // order here, by `es_eval::runner` at evaluation, and by LeRobot's own convention
        // (`observation[t]` is what `action[t]` was chosen from). Recording the post-step
        // state instead taught every policy `(s_t+1) -> a_t` while inference asks `(s_t)`,
        // one control period of lag in every input (packet M5/V12, design note section 7.20).
        self.pre.snapshot(&self.backend.state(), self.tick);

        let substeps = self.schedule.domains().inference.period;
        let started = Instant::now();
        let report = self.backend.step(substeps)?;
        self.metrics.simulation_wall += started.elapsed();
        self.tick = report.tick;
        self.metrics.physics_env_steps += u64::from(substeps) * u64::from(self.n_envs);
        self.metrics.action_env_steps += u64::from(self.n_envs);

        let mut outcome = StepOutcome {
            rewards: vec![0.0; self.n_envs as usize],
            dones: vec![false; self.n_envs as usize],
            failures: vec![None; self.n_envs as usize],
            episodes: Vec::new(),
        };
        let mut to_reset = Vec::new();

        for env in 0..self.n_envs {
            let i = env as usize;
            self.steps[i] += 1;
            let failure = report
                .failures
                .iter()
                .find(|(e, _)| *e == env)
                .map(|(_, kind)| *kind);
            let mut termination = self.evaluate_env(env);
            if let Some(kind) = failure {
                match self.failure_policy.action_for(kind) {
                    FailureAction::Quarantine => {
                        self.health[i] = EnvHealth::Quarantined;
                        termination = Termination::Failure;
                    }
                    FailureAction::AbortEpisode => termination = Termination::Failure,
                    // Record and Fallback leave the env healthy (§18.5); AbortRun is the
                    // caller's decision, and it sees the failure in the outcome.
                    _ => {}
                }
            }
            let reward = self.reward_of(env);
            outcome.rewards[i] = reward;
            outcome.failures[i] = failure;
            outcome.dones[i] = termination.is_done();

            // State: what the step was entered with. Reward, termination and failure: what the
            // transition produced, so `done` still marks the last frame of the episode.
            let row = StepRow {
                tick: self.pre.tick,
                qpos: row_of(&self.pre.qpos, env, self.model.nq),
                qvel: row_of(&self.pre.qvel, env, self.model.nv),
                ctrl: row_of(&self.last_ctrl, env, self.model.nu),
                sensordata: row_of(&self.pre.sensordata, env, self.model.nsensordata),
                reward,
                termination,
                failure,
            };
            // A quarantined env is out of the dataset as well as out of reductions (§18.5).
            if self.health[i] != EnvHealth::Quarantined || termination.is_done() {
                self.recorder.push(env, &row);
            }
            if termination.is_done() {
                to_reset.push(env);
            }
        }

        if !to_reset.is_empty() {
            outcome.episodes = self.reset(Some(&to_reset))?;
        }
        Ok(outcome)
    }

    /// One control step driven by a policy instead of by a caller-supplied `ctrl` (§12).
    ///
    /// The phase order is §12.1's: observation (camera round-robin over the simulation ticks
    /// of this control window), then inference (submit, poll, run the released batch into the
    /// chunk buffers), then action. The action phase is the only actuator path, and it runs
    /// **through** `planes` — an underrun hands the plane an empty chunk and the plane answers
    /// with the fallback; no branch reaches [`Env::step`] around it (`INV-12`).
    ///
    /// `planes` is one [`SafetyPlane`] per env, because its hold target, rate-limit history and
    /// latch are per-robot state (§9.3). `plans` is either empty — the raw `qpos ‖ qvel`
    /// observation — or one [`CpuPlan`] per env, whose `TemporalWindow` rings are likewise
    /// per-env (§7.5).
    pub fn step_with_policy<const NJ: usize, const H: usize>(
        &mut self,
        runner: &mut DomainRunner<NJ, H>,
        policy: &mut dyn PolicyRuntime,
        planes: &mut [SafetyPlane<NJ, H>],
        plans: &mut [CpuPlan],
    ) -> Result<StepOutcome, EnvError> {
        let nu = self.model.nu as usize;
        if nu != NJ {
            return Err(EnvError::shape("action dim", nu, NJ));
        }
        let sim_tick = self.tick.0;
        {
            let state = self.backend.state();
            runner.observe_window(sim_tick, &self.model, &state, plans)?;
        }
        runner.infer_window(sim_tick, policy)?;
        let mut ctrl = vec![0.0; self.n_envs as usize * nu];
        runner.emit_actions(self.tick, planes, &mut ctrl)?;
        let outcome = self.step(&ctrl)?;
        for (env, done) in outcome.dones.iter().enumerate() {
            if *done {
                runner.reset_env(env as u32);
            }
        }
        runner.advance();
        Ok(outcome)
    }

    /// The ports one env's reward and termination cones read, refilled in place.
    ///
    /// Every binding below is overwritten for this env, but the `stage.*` ports the control
    /// executor writes are not in that set, so they are dropped first: a leftover would make
    /// the next env's `Terminate` cone and first-entry `Branch` a function of the previous env
    /// (`docs/reviews/M4.md` B-1).
    fn bind_ports(&mut self, env: u32) {
        crate::control::clear_stage_ports(&mut self.ports);
        let state = self.backend.state();
        let rate = self.model.rate;
        let episode_ticks = u64::from(self.steps[env as usize])
            * u64::from(self.schedule.domains().inference.period);
        for (name, source) in &self.scalar.bindings {
            let value = match *source {
                Source::Qpos(i) => at(state.qpos, env, self.model.nq, i),
                Source::Qvel(i) => at(state.qvel, env, self.model.nv, i),
                Source::Sensor(i) => at(state.sensordata, env, self.model.nsensordata, i),
                // Seconds are derived from a tick count at the edge, never accumulated (§18.1).
                Source::Time { since_reset } => {
                    let ticks = if since_reset {
                        PhysTick(episode_ticks)
                    } else {
                        self.tick
                    };
                    SimTime::new(ticks, rate).as_secs_f64()
                }
            };
            self.ports.insert(name.clone(), value);
        }
    }

    /// The task's own `Terminate` predicates and the episode budget, then — when the task has
    /// a control graph — the stage machine (spec 6.2). A task-level predicate is a
    /// whole-episode statement and wins over a stage transition.
    fn evaluate_env(&mut self, env: u32) -> Termination {
        self.bind_ports(env);
        let task_level = episode::evaluate(
            &self.scalar,
            &self.ports,
            self.steps[env as usize],
            self.max_episode_steps,
        );
        let Some(control) = self.control.as_mut() else {
            return task_level;
        };
        let outcome = control.step(env, &mut self.ports);
        match (task_level, outcome.done, outcome.failed) {
            (Termination::Running, true, true) => Termination::Failure,
            (Termination::Running, true, false) => Termination::Success,
            (other, _, _) => other,
        }
    }

    /// The active stage's name, when the task has a control graph (spec 6.2).
    pub fn stage_name(&self, env: u32) -> Option<&str> {
        self.control.as_ref()?.stage_name(env)
    }

    /// `sum(weight * term)` over the task's `Reward` nodes, in ascending node id. A term that
    /// cannot be evaluated contributes nothing rather than poisoning the sum with a `NaN`.
    ///
    /// With a control graph the sum runs over the **active stage's** terms only, scaled by the
    /// stage weight (`docs/design/control-graph.md` §3.2).
    fn reward_of(&self, env: u32) -> f64 {
        if !self.health[env as usize].participates_in_reduction() {
            return 0.0;
        }
        let (stage, weight) = match self.control.as_ref().map(|c| c.scored_stage(env)) {
            None => (None, 1.0),
            // The graph finished or timed out: no stage is scoring this step.
            Some(None) => return 0.0,
            Some(Some((names, weight))) => (Some(names), weight),
        };
        weight
            * self
                .scalar
                .rewards
                .iter()
                .filter(|t| stage.is_none_or(|names| names.contains(&t.name)))
                .filter_map(|t| t.expr.eval(&self.ports).map(|v| t.weight * v))
                .sum::<f64>()
    }

    /// Episodes closed so far are handed out by `step` and `reset`; this is the live one.
    pub fn open_episode(&self, env: u32) -> &Episode {
        self.recorder.open(env)
    }

    /// The §12.4 metrics measured so far. Domains this packet does not run (render, inference,
    /// VRAM) stay `None`.
    pub fn metrics(&self) -> EnvMetrics {
        let secs = self.metrics.simulation_wall.as_secs_f64();
        let rate = |count: u64| (secs > 0.0).then(|| count as f64 / secs);
        EnvMetrics {
            physics_steps_per_sec: rate(self.metrics.physics_env_steps),
            actions_per_sec: rate(self.metrics.action_env_steps),
            ..self.metrics
        }
    }
}

/// The state a control step was entered with, copied out of the backend before it advances.
#[derive(Clone, Debug, Default)]
struct PreStep {
    tick: PhysTick,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    sensordata: Vec<f64>,
}

impl PreStep {
    fn snapshot(&mut self, state: &StateView<'_>, tick: PhysTick) {
        self.tick = tick;
        for (dst, src) in [
            (&mut self.qpos, state.qpos),
            (&mut self.qvel, state.qvel),
            (&mut self.sensordata, state.sensordata),
        ] {
            dst.clear();
            dst.extend_from_slice(src);
        }
    }
}

fn row_of(values: &[f64], env: u32, width: u32) -> &[f64] {
    let start = env as usize * width as usize;
    values.get(start..start + width as usize).unwrap_or(&[])
}

fn at(values: &[f64], env: u32, width: u32, index: u32) -> f64 {
    row_of(values, env, width)
        .get(index as usize)
        .copied()
        .unwrap_or(0.0)
}

// Bitwise reproducibility is the property under test: these comparisons are deliberate.
#[cfg(test)]
#[allow(clippy::float_cmp)]
pub(crate) mod tests {
    use super::*;
    use es_ir::graph::NodeId;
    use es_ir::task::{
        ArithOp, CmpOp, Distribution, Expr, JointQuantity, TaskConfig, TaskGraph, TaskNode,
        TerminationKind,
    };
    use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
    use es_physics_core::backend::{IndexRange, PhysicsError, StepReport};
    use es_physics_core::caps::{BatchSupport, Capabilities, DeterminismTier, FloatPrecision};
    use std::collections::BTreeSet;

    const MJCF: &str = r#"<mujoco>
        <worldbody><body name="link">
            <joint name="hinge" type="hinge" axis="0 0 1"/>
            <joint name="slide" type="slide" axis="1 0 0"/>
            <geom name="ball" type="sphere" size="0.1"/>
        </body></worldbody>
        <actuator><motor name="motor" joint="hinge"/></actuator>
        <sensor><jointpos name="angle" joint="hinge"/></sensor>
    </mujoco>"#;

    pub(crate) fn fake_scene() -> SceneDesc {
        es_assets::parse_mjcf(MJCF)
            .expect("the fixture parses")
            .scene
    }

    /// A `ModelInfo` matching [`fake_scene`]: two 1-dof joints, one actuator, one sensor.
    pub(crate) fn fake_model() -> ModelInfo {
        let scene = fake_scene();
        let id = |name: &str| {
            scene
                .joints
                .iter()
                .find(|j| j.name == name)
                .expect("fixture joint")
                .id
        };
        ModelInfo {
            nq: 2,
            nv: 2,
            nu: 1,
            nsensordata: 1,
            nbody: 1,
            n_envs: 1,
            qpos: [
                (id("hinge"), IndexRange::new(0, 1)),
                (id("slide"), IndexRange::new(1, 1)),
            ]
            .into(),
            dof: [
                (id("hinge"), IndexRange::new(0, 1)),
                (id("slide"), IndexRange::new(1, 1)),
            ]
            .into(),
            actuator: [(scene.actuators[0].id, IndexRange::new(0, 1))].into(),
            sensor: [(scene.sensors[0].id, IndexRange::new(0, 1))].into(),
            ..ModelInfo::default()
        }
    }

    pub(crate) fn task_with(nodes: &[TaskNode]) -> TaskIr {
        let mut graph = TaskGraph::new(1);
        for (i, node) in nodes.iter().enumerate() {
            graph.insert(NodeId(i as u32), node.clone());
        }
        TaskIr {
            schema_version: 1,
            scene: es_ir::task::SceneRef {
                path: "fixture.xml".to_owned(),
                scene_hash: [0; 32],
                asset_hash: [0; 32],
            },
            graph,
            observation_spec: es_ir::task::ObservationSpec::default(),
            control: None,
            config: TaskConfig {
                max_episode_steps: 0,
                control_rate_hz: 50.0,
                deterministic: true,
                rng_streams: BTreeSet::new(),
            },
        }
    }

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

    /// A hinge-angle reward, a `|angle| > 1.5` failure predicate, and a reset distribution.
    ///
    /// `reward = -1 * clamp(qpos[hinge], -10, 10)` — small, but it exercises the whole cone
    /// lowering (source -> Clamp -> Reward) and the `Compare`-with-literal form.
    fn pendulum_task(max_steps: u32, reset: Option<Distribution>) -> TaskIr {
        let angle = TaskNode::GetJointState {
            body: fake_scene().bodies[0].id,
            joints: vec!["hinge".to_owned()],
            quantity: JointQuantity::Position,
        };
        let mut task = task_with(&[
            angle,
            TaskNode::Clamp {
                lo: vec![-10.0],
                hi: vec![10.0],
                ty: scalar_ty(Unit::Angle),
            },
            TaskNode::Reward {
                name: "upright".to_owned(),
                weight: -1.0,
                aggregation: es_ir::task::Aggregation::Sum,
                ty: scalar_ty(Unit::Angle),
            },
            TaskNode::Compare {
                op: CmpOp::Gt,
                rhs: Some(1.5),
                ty: scalar_ty(Unit::Angle),
            },
            TaskNode::Terminate {
                kind: TerminationKind::Failure,
            },
        ]);
        task.graph.connect(NodeId(0), "value", NodeId(1), "value");
        task.graph.connect(NodeId(1), "value", NodeId(2), "value");
        task.graph.connect(NodeId(0), "value", NodeId(3), "a");
        task.graph.connect(NodeId(3), "value", NodeId(4), "value");
        if let Some(dist) = reset {
            task.graph.insert(
                NodeId(5),
                TaskNode::ResetState {
                    target: "qpos[0]".to_owned(),
                    dist,
                    stream: "reset.hinge".to_owned(),
                },
            );
        }
        task.config.max_episode_steps = max_steps;
        task
    }

    /// A test double: a spring-damper integrator, deterministic and transcendental-free.
    ///
    /// `es-physics-core`'s own `NullBackend` is `#[cfg(test)]` and does not integrate anything,
    /// so it cannot be reused here.
    pub(crate) struct FakeBackend {
        caps: Capabilities,
        model: Option<ModelInfo>,
        qpos: Vec<f64>,
        qvel: Vec<f64>,
        sensordata: Vec<f64>,
        ctrl: Vec<f64>,
        tick: PhysTick,
        /// Envs the next step reports as diverged.
        pub diverge: Vec<u32>,
    }

    impl FakeBackend {
        pub(crate) fn new() -> Self {
            Self {
                caps: Capabilities {
                    name: "fake".to_owned(),
                    determinism: DeterminismTier::Bitwise,
                    batch: BatchSupport {
                        max_envs: 1024,
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
                diverge: Vec::new(),
            }
        }
    }

    impl PhysicsBackend for FakeBackend {
        fn capabilities(&self) -> &Capabilities {
            &self.caps
        }

        fn load(
            &mut self,
            _scene: &SceneDesc,
            cfg: &LoadConfig,
        ) -> Result<ModelInfo, PhysicsError> {
            let info = ModelInfo {
                n_envs: cfg.n_envs,
                ..fake_model()
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
                    let torque = self.ctrl[env * info.nu as usize];
                    // Linear spring-damper: no transcendentals, so the double is exactly
                    // reproducible on every target (§3.4).
                    let accel = torque - 9.81 * self.qpos[q] - 0.1 * self.qvel[v];
                    self.qvel[v] += dt * accel;
                    self.qpos[q] += dt * self.qvel[v];
                    self.sensordata[env] = self.qpos[q];
                }
            }
            self.tick = self.tick.add_ticks(u64::from(n_substeps));
            Ok(StepReport {
                tick: self.tick,
                failures: std::mem::take(&mut self.diverge)
                    .into_iter()
                    .map(|e| (e, FailureKind::Diverged))
                    .collect(),
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

    fn domains(n_envs: u32) -> BatchDomains {
        BatchDomains {
            simulation: crate::scheduler::DomainCfg::new(n_envs, 1),
            observation: crate::scheduler::DomainCfg::new(n_envs, 10),
            inference: crate::scheduler::DomainCfg::new(n_envs, 20),
            training: None,
        }
    }

    fn env_of(task: &TaskIr, n_envs: u32, seed: u64) -> Env<FakeBackend> {
        Env::new(
            task,
            &fake_scene(),
            FakeBackend::new(),
            &domains(n_envs),
            seed,
        )
        .expect("the fixture task compiles")
    }

    /// Runs `ticks` control steps with a constant torque and returns each env's reward trace.
    fn run(env: &mut Env<FakeBackend>, n_envs: u32, ticks: usize) -> Vec<Vec<f64>> {
        let ctrl = vec![0.05; n_envs as usize];
        let mut traces = vec![Vec::with_capacity(ticks); n_envs as usize];
        for _ in 0..ticks {
            let out = env.step(&ctrl).expect("step");
            for (i, r) in out.rewards.iter().enumerate() {
                traces[i].push(*r);
            }
        }
        traces
    }

    #[test]
    fn two_envs_with_the_same_reset_draw_follow_the_same_trajectory() {
        let task = pendulum_task(0, Some(Distribution::Constant(0.3)));
        let mut env = env_of(&task, 2, 1234);
        let traces = run(&mut env, 2, 100);
        assert_eq!(
            traces[0], traces[1],
            "a constant reset makes envs identical"
        );
        assert_eq!(env.tick(), PhysTick(100 * 20));
        assert_eq!(traces[0].len(), 100);
        assert!(traces[0].iter().all(|r| r.is_finite()));
    }

    #[test]
    fn randomized_envs_diverge_but_a_seed_replays_exactly() {
        let task = pendulum_task(0, Some(Distribution::Uniform { lo: -0.5, hi: 0.5 }));
        let a = run(&mut env_of(&task, 2, 7), 2, 100);
        let b = run(&mut env_of(&task, 2, 7), 2, 100);
        let c = run(&mut env_of(&task, 2, 8), 2, 100);
        assert_eq!(a, b, "same seed, same trajectories");
        assert_ne!(a[0], a[1], "different envs draw different resets");
        assert_ne!(a, c, "a different seed gives a different run");
    }

    #[test]
    fn termination_auto_resets_and_hands_back_the_episode() {
        // Start past the |angle| > 1.5 predicate: the first step terminates.
        let task = pendulum_task(0, Some(Distribution::Constant(2.0)));
        let mut env = env_of(&task, 2, 0);
        let out = env.step(&[0.0, 0.0]).unwrap();
        assert_eq!(out.dones, [true, true]);
        assert_eq!(out.episodes.len(), 2);
        assert_eq!(out.episodes[0].termination, Termination::Failure);
        assert_eq!(out.episodes[0].steps(), 1);
        assert_eq!(out.episodes[0].episode, 0, "the run's first episode");
        assert_eq!(env.open_episode(0).episode, 1);
        assert_eq!(env.open_episode(0).steps(), 0, "a fresh episode is open");
        // Reset re-drew the constant, so the env is back at 2.0 rad.
        assert_eq!(env.backend().state().qpos_of(0)[0], 2.0);
    }

    #[test]
    fn the_episode_budget_terminates_with_timeout() {
        let task = pendulum_task(3, Some(Distribution::Constant(0.0)));
        let mut env = env_of(&task, 1, 0);
        for _ in 0..2 {
            assert_eq!(env.step(&[0.0]).unwrap().dones, [false]);
        }
        let out = env.step(&[0.0]).unwrap();
        assert_eq!(out.dones, [true]);
        assert_eq!(out.episodes[0].termination, Termination::Timeout);
        assert_eq!(out.episodes[0].steps(), 3);
    }

    #[test]
    fn a_diverged_env_is_quarantined_and_leaves_the_reduction() {
        let task = pendulum_task(0, Some(Distribution::Constant(0.3)));
        let mut env = Env::new(&task, &fake_scene(), FakeBackend::new(), &domains(2), 0).unwrap();
        env.step(&[0.0, 0.0]).unwrap();
        // The backend reports env 1 diverged on the next step.
        env.backend.diverge = vec![1];
        let out = env.step(&[0.0, 0.0]).unwrap();
        assert_eq!(out.failures, [None, Some(FailureKind::Diverged)]);
        assert_eq!(out.dones, [false, true]);
        assert_eq!(out.rewards[1], 0.0, "quarantined envs do not contribute");
        // Reset clears the quarantine (§18.5: until reset).
        assert_eq!(env.health(), [EnvHealth::Ok, EnvHealth::Ok]);
    }

    #[test]
    fn ctrl_of_the_wrong_length_is_rejected() {
        let task = pendulum_task(0, None);
        let mut env = env_of(&task, 2, 0);
        let err = env.step(&[0.0]).unwrap_err();
        assert_eq!(err.to_string(), "ctrl: expected 2 values, got 1");
        assert!(env.reset(Some(&[5])).is_err(), "env 5 does not exist");
    }

    #[test]
    fn metrics_report_the_nine_names_and_never_a_single_step_per_sec() {
        let task = pendulum_task(0, None);
        let mut env = env_of(&task, 4, 0);
        run(&mut env, 4, 10);
        let m = env.metrics();
        assert_eq!(m.physics_env_steps, 10 * 20 * 4);
        assert_eq!(m.action_env_steps, 40);
        assert!(m.physics_steps_per_sec.is_some_and(|v| v > 0.0));
        assert!(m.actions_per_sec.is_some());
        // Domains this packet does not run stay None rather than reporting a fabricated zero.
        assert_eq!(m.camera_frames_per_sec, None);
        assert_eq!(m.policy_inferences_per_sec, None);
        assert_eq!(m.gpu_memory_peak, None);
        assert_eq!(m.chunk_underrun_rate, None);
    }

    #[test]
    fn an_unsupported_node_in_a_reward_cone_is_named() {
        let mut task = pendulum_task(0, None);
        task.graph.insert(
            NodeId(0),
            TaskNode::GetLanguage {
                key: "instruction".to_owned(),
                max_tokens: 8,
            },
        );
        let err = Env::new(&task, &fake_scene(), FakeBackend::new(), &domains(1), 0).unwrap_err();
        assert!(err.to_string().contains("GetLanguage"), "{err}");
    }

    #[test]
    fn the_recorded_episode_carries_every_column_and_the_param_scales() {
        let mut task = pendulum_task(2, Some(Distribution::Constant(0.1)));
        task.graph.insert(
            NodeId(6),
            TaskNode::Randomization {
                target: "body.link.mass".to_owned(),
                dist: Distribution::Uniform { lo: 0.9, hi: 1.1 },
                stream: "mass".to_owned(),
            },
        );
        let mut env = env_of(&task, 1, 3);
        let out = {
            env.step(&[0.1]).unwrap();
            env.step(&[0.2]).unwrap()
        };
        let ep = &out.episodes[0];
        assert_eq!(ep.steps(), 2);
        assert_eq!(ep.ctrl, vec![0.1, 0.2]);
        assert_eq!(ep.qpos.len(), 4);
        assert_eq!(ep.sensordata.len(), 2);
        assert_eq!(ep.param_scales.len(), 1, "the mass scale was recorded");
        assert!(ep.param_scales.values().all(|v| (0.9..=1.1).contains(v)));
        // The row is the state the step was entered with, so its tick is the step's first
        // tick, not its last (packet M5/V12).
        assert_eq!(ep.ticks, [PhysTick(0), PhysTick(20)]);
    }

    /// Packet M5/V12 oracle 1: `observation.state[t]` is the state `action[t]` was computed
    /// from, which is the state the control step is entered with -- not the one it produced.
    #[test]
    fn the_recorded_row_is_the_state_the_action_was_computed_from() {
        let task = pendulum_task(0, Some(Distribution::Constant(0.1)));
        let mut env = env_of(&task, 1, 1);
        let before = env.backend().state().qpos.to_vec();
        env.step(&[0.3]).unwrap();
        let after = env.backend().state().qpos.to_vec();
        assert_ne!(
            before, after,
            "the fixture must move for this to mean anything"
        );
        let ep = env.recorder.open(0);
        assert_eq!(ep.qpos, before, "the row is the pre-step state");
        assert_ne!(ep.qpos, after, "the row is not the post-step state");
        assert_eq!(ep.ctrl, vec![0.3], "with the ctrl that was applied to it");
    }

    #[test]
    fn a_time_source_reads_ticks_not_an_accumulated_float() {
        let mut task = pendulum_task(0, None);
        task.graph
            .insert(NodeId(7), TaskNode::GetTime { since_reset: true });
        task.graph.insert(
            NodeId(8),
            TaskNode::Compare {
                op: CmpOp::Ge,
                rhs: Some(0.05),
                ty: scalar_ty(Unit::Time),
            },
        );
        task.graph.insert(
            NodeId(9),
            TaskNode::Terminate {
                kind: TerminationKind::Success,
            },
        );
        task.graph.connect(NodeId(7), "value", NodeId(8), "a");
        task.graph.connect(NodeId(8), "value", NodeId(9), "value");
        let mut env = env_of(&task, 1, 0);
        // 20 ticks at 1 kHz = 20 ms per control step; success on the third.
        assert_eq!(env.step(&[0.0]).unwrap().dones, [false]);
        assert_eq!(env.step(&[0.0]).unwrap().dones, [false]);
        let out = env.step(&[0.0]).unwrap();
        assert_eq!(out.episodes[0].termination, Termination::Success);
    }

    #[test]
    fn an_arith_cone_lowers_and_evaluates() {
        // reward = qpos[hinge] + qvel[hinge], through Arith rather than Clamp.
        let body = fake_scene().bodies[0].id;
        let mut task = task_with(&[
            TaskNode::GetJointState {
                body,
                joints: vec!["hinge".to_owned()],
                quantity: JointQuantity::Position,
            },
            TaskNode::GetJointState {
                body,
                joints: vec!["hinge".to_owned()],
                quantity: JointQuantity::Velocity,
            },
            TaskNode::Arith {
                op: ArithOp::Add,
                ty: scalar_ty(Unit::Angle),
            },
            TaskNode::Reward {
                name: "sum".to_owned(),
                weight: 2.0,
                aggregation: es_ir::task::Aggregation::Sum,
                ty: scalar_ty(Unit::Angle),
            },
            TaskNode::ResetState {
                target: "joint.hinge.qvel".to_owned(),
                dist: Distribution::Constant(1.0),
                stream: "v".to_owned(),
            },
        ]);
        task.graph.connect(NodeId(0), "value", NodeId(2), "a");
        task.graph.connect(NodeId(1), "value", NodeId(2), "b");
        task.graph.connect(NodeId(2), "value", NodeId(3), "value");
        let mut env = env_of(&task, 1, 0);
        let out = env.step(&[0.0]).unwrap();
        let state = env.backend().state();
        let expected = 2.0 * (state.qpos_of(0)[0] + state.qvel_of(0)[0]);
        assert!(
            (out.rewards[0] - expected).abs() < 1e-12,
            "{:?}",
            out.rewards
        );
    }

    #[test]
    fn a_dangling_reward_cone_is_a_task_error_not_a_panic() {
        let task = task_with(&[TaskNode::Reward {
            name: "r".to_owned(),
            weight: 1.0,
            aggregation: es_ir::task::Aggregation::Sum,
            ty: scalar_ty(Unit::Angle),
        }]);
        let err = Env::new(&task, &fake_scene(), FakeBackend::new(), &domains(1), 0).unwrap_err();
        assert!(err.to_string().contains("no incoming edge"), "{err}");
        assert!(Expr::Port("missing".to_owned())
            .eval(&BTreeMap::new())
            .is_none());
    }
}
