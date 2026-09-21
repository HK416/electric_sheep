//! `Rollout` — a trainer steps our `Env` and the Safety Plane (packet M8/S4a, §13.4).
//!
//! This is the Rust half: a plain struct, `rlib`-visible, with no `pyo3` in sight, so
//! `cargo test -p es-py` exercises it without a Python build. [`crate::pybind`] wraps it for
//! `train_ppo.py`.
//!
//! The rule that makes it worth having (`docs/design/rl-continuation.md` section 1, rule 2):
//! **the trainer never calls a simulator of its own, and every sampled action goes through
//! `SafetyPlane::validate` before the actuator** (`INV-12`). So one control step here is the
//! per-step order `es_eval::runner::run_episode` runs, and nothing else:
//!
//! ```text
//! observe()  ──▶  capture(plan inputs from the state)  ──▶  CpuPlan::run
//! act(a)     ──▶  observe_state ──▶ heartbeat ──▶ validate ──▶ Env::step
//! ```
//!
//! and the capture half is *literally* `es_eval::runner`'s (`capture`, `input_sources`,
//! `joint_state`, made `pub` by this packet): a second implementation is how a trainer and an
//! evaluation quietly start reading two different observations off one state.
//!
//! Horizon 1, synchronous: PPO acts on every control tick, so the chunk handed to the plane
//! carries exactly one row under a fresh `seq` and there is no chunk buffer and no declared
//! latency here (`rl-continuation.md` section 3). The **evaluation** of the same policy runs
//! under the Deployment IR's latency through `es eval run`, which is what keeps that number
//! honest rather than the trainer's own.

use std::collections::BTreeMap;

use es_assets::scene::{Actuator, SceneDesc};
use es_compile::{CpuPlan, PlanMode, Tensor, TensorRef};
use es_core::{PhysTick, TickRate};
use es_env::scheduler::BatchDomains;
use es_env::{Env, EnvMetrics};
use es_eval::runner::{capture, input_sources, joint_state, Capture};
use es_eval::LightOverride;
use es_ir::deployment::{ActionSpace, ExecutionMode, Micros};
use es_ir::types::ElemType;
use es_physics_backend::MuJoCoCpuBackend;
use es_physics_core::backend::{PhysicsBackend, StateView};
use es_safety::{ActionChunk, SafetyPlane};

/// Everything that stops a rollout from being built or stepped.
#[derive(Debug, thiserror::Error)]
pub enum RolloutError {
    #[error("{what}: {source}")]
    Document {
        what: &'static str,
        #[source]
        source: es_ir::serial::SerialError,
    },
    #[error("scene: {0}")]
    Scene(#[from] es_assets::MjcfError),
    #[error("observation plan: {0}")]
    Plan(String),
    #[error("safety plane: {0}")]
    Safety(String),
    #[error(transparent)]
    Env(#[from] es_env::EnvError),
    #[error(transparent)]
    Eval(#[from] es_eval::EvalError),
    /// The deployment's joint count is the actuator count, and the plane reads the leading
    /// `NJ` joints of the state: a model that disagrees would be driven by a broadcast copy.
    /// Refused by name, exactly as `es_eval::runner::run_episode` refuses it (§10.1).
    #[error(
        "the model has {nu} actuators, {nq} qpos and {nv} qvel entries; the deployment \
         declares {nj} joints"
    )]
    JointMismatch {
        nu: usize,
        nq: usize,
        nv: usize,
        nj: usize,
    },
    #[error("{what}: expected {expected} values, got {got}")]
    Shape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

/// What one [`Rollout::act`] produced, per env.
///
/// `executed` is what reached the actuator — the plane's output, not the sample — and
/// `events` is `EventSet::bits()` for that step, so a trainer can measure how often the two
/// differ (`rl-continuation.md` section 6, open question 2).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Act {
    /// `n_envs * NJ`, row-major by env.
    pub executed: Vec<f64>,
    pub events: Vec<u32>,
    pub rewards: Vec<f64>,
    pub dones: Vec<bool>,
}

/// One env batch, one Safety Plane and one observation plan per env (§13.4).
///
/// `NJ` is the deployment's joint count and `H` its declared chunk horizon: both come from
/// the document, because `SafetyPlane::from_ir` refuses an envelope whose width is not its
/// own. Only row 0 of the chunk is ever filled here — see the module docs.
pub struct Rollout<const NJ: usize, const H: usize> {
    env: Env<MuJoCoCpuBackend>,
    /// Per robot, because the plane's hold target, rate-limit history and latch are per-robot
    /// state (§9.3) — the same reason `Env::step_with_policy` takes one per env.
    planes: Vec<SafetyPlane<NJ, H>>,
    /// Per env, because a plan's `TemporalWindow` rings are per-env state (§7.5).
    plans: Vec<CpuPlan>,
    /// Resolved once, against the documents (§7.4, §10.1).
    sources: BTreeMap<String, Capture>,
    /// Kept for the names and `ctrlrange` [`Rollout::model`] reports: `ModelInfo` carries
    /// `StableId`s and index ranges, and a trainer needs the names a human wrote.
    scene: SceneDesc,
    mode: ExecutionMode,
    /// What a sampled row means (spec 8.5). `JointPosition` is a target;
    /// [`ActionSpace::JointDelta`] is an increment `absolute_target` adds to the plane's last
    /// executed command (packet M9/T1).
    space: ActionSpace,
    n_envs: usize,
    /// The **control** tick, which is the plane's clock (packet M5/V17) and what `now` is
    /// stamped with. `Env::tick` counts simulation ticks, `inference.period` per control step.
    step: u64,
    /// Monotonic per policy invocation (§8.6). One invocation per control tick here, so the
    /// plane accepts a fresh chunk every step instead of consuming a stale one.
    seq: u64,
    /// Scratch, allocated once: the post-plane command handed to `Env::step`.
    ctrl: Vec<f64>,
}

impl<const NJ: usize, const H: usize> std::fmt::Debug for Rollout<NJ, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rollout")
            .field("n_envs", &self.n_envs)
            .field("step", &self.step)
            .finish_non_exhaustive()
    }
}

impl<const NJ: usize, const H: usize> Rollout<NJ, H> {
    /// Builds the env, the planes and the plans from the four documents.
    ///
    /// `n_envs` sizes the simulation batch; the observation and inference domains follow it,
    /// and the control period is the scene's timestep over the deployment's `rate.control` —
    /// `BatchDomains::single_env_at`'s own derivation, which is what `es eval run` and
    /// `es loop collect` step at (packet M5/V11).
    pub fn new(
        task_toml: &str,
        observation_toml: &str,
        deployment_toml: &str,
        scene_xml: &str,
        seed: u64,
        n_envs: u32,
    ) -> Result<Self, RolloutError> {
        let doc = |what: &'static str| move |source| RolloutError::Document { what, source };
        let task = es_ir::serial::task_from_toml(task_toml).map_err(doc("task.toml"))?;
        let obs =
            es_ir::serial::observation_from_toml(observation_toml).map_err(doc("observation"))?;
        let deploy =
            es_ir::serial::deployment_from_toml(deployment_toml).map_err(doc("deployment"))?;
        let scene = es_assets::parse_mjcf(scene_xml)?.scene;

        let mut domains = BatchDomains::single_env_at(
            TickRate::from_period_secs(scene.options.timestep)
                .map_err(|e| es_env::EnvError::Schedule(e.to_string()))?,
            deploy.rate.control,
        )?;
        for d in [
            &mut domains.simulation,
            &mut domains.observation,
            &mut domains.inference,
        ] {
            d.batch = n_envs;
        }
        let env: Env<MuJoCoCpuBackend> =
            Env::new(&task, &scene, MuJoCoCpuBackend::new(), &domains, seed)?;

        let (nu, nq, nv) = {
            let m = env.model();
            (m.nu as usize, m.nq as usize, m.nv as usize)
        };
        if nu != NJ || nq < NJ || nv < NJ {
            return Err(RolloutError::JointMismatch { nu, nq, nv, nj: NJ });
        }

        let mut plans = Vec::with_capacity(n_envs as usize);
        for _ in 0..n_envs {
            plans
                .push(CpuPlan::compile(&obs, PlanMode::Release).map_err(|d| {
                    RolloutError::Plan(d.iter().map(ToString::to_string).collect())
                })?);
        }
        let sources = input_sources(&plans[0], &obs, &task, Some(env.model()))?;
        let mut planes = Vec::with_capacity(n_envs as usize);
        for _ in 0..n_envs {
            planes.push(
                SafetyPlane::<NJ, H>::from_ir(&deploy)
                    .map_err(|e| RolloutError::Safety(e.to_string()))?,
            );
        }

        Ok(Self {
            env,
            planes,
            plans,
            sources,
            scene,
            mode: deploy.execution,
            space: deploy.action.space,
            n_envs: n_envs as usize,
            step: 0,
            seq: 0,
            ctrl: vec![0.0; n_envs as usize * nu],
        })
    }

    /// Resets `envs` (all of them when `None`).
    ///
    /// An episode is where an observation stream ends (§7.5 layer 1) and where a command
    /// chain ends (packet P-M7-R1): the plan's rings are dropped and the plane's latch and
    /// hold target are re-seeded by `begin_episode`. Neither is disabling the plane
    /// (`INV-12`) — the envelope, the watchdogs and the counters are untouched.
    pub fn reset(&mut self, envs: Option<&[u32]>) -> Result<(), RolloutError> {
        self.env.reset(envs)?;
        let all: Vec<u32> = (0..self.n_envs as u32).collect();
        for env in envs.unwrap_or(&all) {
            self.begin_episode(*env as usize);
        }
        Ok(())
    }

    /// The Observation IR's output ports for every env: port name -> `[n_envs, dim]`,
    /// row-major.
    ///
    /// The inputs are captured by `es_eval::runner::capture` off this env's own row of the
    /// state, and run through this env's own `CpuPlan` — the same two calls `es eval run`
    /// makes. An image input is refused there by name: images need a renderer, and this
    /// binding links none (§4.3).
    pub fn observe(&mut self) -> Result<BTreeMap<String, Vec<f64>>, RolloutError> {
        let light = LightOverride::default();
        let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let model = self.env.model();
        let state = self.env.backend().state();
        for i in 0..self.n_envs {
            let view = env_view(&state, i);
            let (names, bytes, _) = capture(
                &self.plans[i],
                &self.sources,
                model,
                &view,
                None,
                &light,
                None,
            )?;
            let inputs: BTreeMap<String, TensorRef<'_>> = names
                .iter()
                .zip(&bytes)
                .map(|((name, dtype, shape), data)| {
                    (
                        name.clone(),
                        TensorRef::new(*dtype, shape.clone(), data.as_slice()),
                    )
                })
                .collect();
            let ports = self.plans[i]
                .run(&inputs)
                .map_err(|e| RolloutError::Plan(e.to_string()))?;
            for (name, tensor) in &ports {
                out.entry(name.clone())
                    .or_default()
                    .extend(port_values(name, tensor)?);
            }
        }
        Ok(out)
    }

    /// One control step: the plane validates every env's sampled action, `Env::step` executes
    /// what the plane returned.
    ///
    /// `actions` is `[n_envs * NJ]` in the unit the deployment's `action.space` names --
    /// absolute targets under `JointPosition`, increments on the current target under
    /// `JointDelta`, which `es_env::chunk_buffer::absolute_target` integrates against the
    /// plane's last executed command (spec 8.5, packet M9/T1). The per-env order is
    /// `run_episode`'s — `observe_state`, `heartbeat`, `validate` — and `Env::step` is handed
    /// `SafeAction::q` and nothing else: there is no path around the plane (`INV-12`).
    pub fn act(&mut self, actions: &[f64]) -> Result<Act, RolloutError> {
        if actions.len() != self.n_envs * NJ {
            return Err(RolloutError::Shape {
                what: "actions",
                expected: self.n_envs * NJ,
                got: actions.len(),
            });
        }
        let now = PhysTick(self.step);
        self.seq += 1;
        let mut events = Vec::with_capacity(self.n_envs);
        {
            let state = self.env.backend().state();
            for i in 0..self.n_envs {
                let view = env_view(&state, i);
                let (q, qd) = joint_state::<NJ>(&view);
                let plane = &mut self.planes[i];
                plane.observe_state(&q, &qd);
                plane.heartbeat(now);
                let mut rows = [[0.0; NJ]; H];
                rows[0].copy_from_slice(&actions[i * NJ..(i + 1) * NJ]);
                // Read after `observe_state`, so the first tick of an episode integrates from
                // the measured pose the plane just seeded and every later tick from what the
                // arm was actually told to do -- never from the raw sample, which is what
                // keeps a clamped increment from accumulating (spec 8.5).
                rows[0] = es_env::chunk_buffer::absolute_target(
                    self.space,
                    &plane.last_safe_action(),
                    &rows[0],
                );
                // One fresh row per control tick under its own `seq`: the plane accepts a
                // chunk it has not seen and its cursor starts at row 0 (§8.6). The
                // observation is this tick's, so its age is zero.
                let chunk = ActionChunk::new(rows, 1, self.mode).with_seq(self.seq);
                let safe = plane.validate(&chunk, Micros(0), now);
                // `nu == NJ` was checked in `new`, so this is a copy, not a broadcast.
                self.ctrl[i * NJ..(i + 1) * NJ].copy_from_slice(&safe.q);
                events.push(safe.events.bits());
            }
        }
        let out = self.env.step(&self.ctrl)?;
        self.step += 1;
        // `Env::step` already reset every env it closed an episode on; the plane and the plan
        // it left behind are what still hold the old episode (packet P-M7-R1).
        for i in 0..self.n_envs {
            if out.dones[i] {
                self.begin_episode(i);
            }
        }
        Ok(Act {
            executed: self.ctrl.clone(),
            events,
            rewards: out.rewards,
            dones: out.dones,
        })
    }

    /// Control ticks since construction — the plane's own clock, not `Env::tick`'s
    /// simulation tick.
    pub fn tick(&self) -> u64 {
        self.step
    }

    /// The §12.4 metric set as the env measured it. A domain this path never runs (render,
    /// inference, VRAM) stays `None` rather than a fabricated zero.
    pub fn metrics(&self) -> EnvMetrics {
        self.env.metrics()
    }

    pub fn n_envs(&self) -> usize {
        self.n_envs
    }

    /// `(nq, nv, nu)` of the loaded model.
    pub fn dims(&self) -> (usize, usize, usize) {
        let m = self.env.model();
        (m.nq as usize, m.nv as usize, m.nu as usize)
    }

    /// Actuator names in `ctrl` order, with each one's `ctrlrange` as the scene declares it
    /// (`None` where the MJCF left it unlimited).
    pub fn actuators(&self) -> Vec<(String, Option<(f64, f64)>)> {
        let index = &self.env.model().actuator;
        let mut rows: Vec<(u32, &Actuator)> = self
            .scene
            .actuators
            .iter()
            .filter_map(|a| index.get(&a.id).map(|r| (r.start, a)))
            .collect();
        rows.sort_by_key(|(i, _)| *i);
        rows.into_iter()
            .map(|(_, a)| (a.name.clone(), a.ctrl_range))
            .collect()
    }

    /// Joint names in `qpos` order.
    pub fn joints(&self) -> Vec<String> {
        let index = &self.env.model().qpos;
        let mut rows: Vec<(usize, String)> = self
            .scene
            .joints
            .iter()
            .filter_map(|j| index.get(&j.id).map(|r| (r.start as usize, j.name.clone())))
            .collect();
        rows.sort_by_key(|(i, _)| *i);
        rows.into_iter().map(|(_, n)| n).collect()
    }

    /// `qpos` of one env — what the golden vector pins, and what a trainer's own
    /// bookkeeping (a value function over privileged state) reads.
    pub fn qpos(&self, env: usize) -> Vec<f64> {
        let state = self.env.backend().state();
        env_view(&state, env).qpos.to_vec()
    }

    fn begin_episode(&mut self, env: usize) {
        self.planes[env].begin_episode();
        self.plans[env].reset();
    }
}

/// One env's row of the batch state, as a one-env [`StateView`].
///
/// `es_eval::runner`'s capture reads env 0 — it was written for the single-env evaluation
/// path — so this is how a batch of `n` reaches it without a second capture implementation:
/// the view *is* env `i`'s rows, and every stride is read off the array the same way
/// `StateView::qpos_of` reads it.
fn env_view<'a>(state: &StateView<'a>, env: usize) -> StateView<'a> {
    let row = |values: &'a [f64]| {
        let stride = values.len() / state.n_envs.max(1) as usize;
        values
            .get(env * stride..(env + 1) * stride)
            .unwrap_or_default()
    };
    StateView {
        n_envs: 1,
        tick: state.tick,
        qpos: row(state.qpos),
        qvel: row(state.qvel),
        act: row(state.act),
        sensordata: row(state.sensordata),
        xpos: row(state.xpos),
        xquat: row(state.xquat),
    }
}

/// One output port's values as Python floats. The plan's own dtype, widened, never
/// reinterpreted.
fn port_values(name: &str, t: &Tensor) -> Result<Vec<f64>, RolloutError> {
    match t.dtype {
        ElemType::F32 => Ok(t
            .data
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        ElemType::F64 => Ok(t
            .data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()),
        other => Err(RolloutError::Plan(format!(
            "observation port \"{name}\" is {other:?}; a rollout hands the trainer floats"
        ))),
    }
}
