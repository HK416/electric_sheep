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
//! **Images** (packet M11/X3). Built with the `render` feature, `Rollout` owns one
//! `es_env::EnvRenderer` per env for the Observation IR's one `ImageInput`, configured by
//! `es_env::render::sensor_cfg` from the Task IR channel that declares that sensor -- the
//! function `es loop collect --frames` and `es eval run --frames` build theirs with -- and hands
//! its `frame` to the same `capture` as a frame source. Each env's reset calls that env's
//! `EnvRenderer::begin_episode`, so a `seed = "tick"` sensor keys its samples on the tick of
//! *that env's* episode, and a `Rollout` frame at `(episode, tick)` is the collector's frame at
//! the same `(episode, tick)`, bit for bit (`crates/es-py/tests/vision_reach.rs`). One frame
//! per env per [`Rollout::observe`]: the tick is the render index, as it is for the collector.
//! Without the feature an image input is refused by name at `observe`, as it always was.
//!
//! Horizon 1, synchronous: PPO acts on every control tick, so the chunk handed to the plane
//! carries exactly one row under a fresh `seq` and there is no chunk buffer and no declared
//! latency here (`rl-continuation.md` section 3). The **evaluation** of the same policy runs
//! under the Deployment IR's latency through `es eval run`, which is what keeps that number
//! honest rather than the trainer's own.

use std::collections::BTreeMap;
use std::time::Duration;

use es_assets::scene::{Actuator, SceneDesc};
use es_compile::{CpuPlan, PlanMode, Tensor, TensorRef};
use es_core::{PhysTick, TickRate};
use es_env::scheduler::BatchDomains;
use es_env::{Env, EnvMetrics, StepOutcome};
use es_eval::runner::{capture_at, input_sources, joint_state, previous_action_initial, Capture};
use es_eval::LightOverride;
use es_ir::deployment::{ActionSpace, ExecutionMode, Micros};
use es_ir::types::ElemType;
use es_physics_backend::{BackendKind, MjWarpBackend, MuJoCoCpuBackend, PhysXBackend};
use es_physics_core::backend::{ModelInfo, PhysicsBackend, StateView};
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
    /// A backend `Rollout` has no closed-loop path for: Newton's own `load` refuses with its
    /// mapping report (packet M11/X1).
    #[error("{0}")]
    Backend(String),
    /// The renderer could not be built for the observation's image input (packet M11/X3).
    #[error("render: {0}")]
    Render(String),
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
    env: Sim,
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
    /// A `PreviousAction` channel's value per env (packet M11/X2): the row last handed to
    /// [`Rollout::act`], before the plane, and `initial` after a reset. Empty when the task
    /// declares no such channel.
    previous: Vec<Vec<f64>>,
    initial: Vec<f64>,
    /// One renderer per env for the image input, empty when the observation has none (packet
    /// M11/X3). Per env because the `Tick` seed clock is per-episode state.
    #[cfg(feature = "render")]
    cameras: Vec<es_env::EnvRenderer<'static>>,
    /// The image bytes each env's last [`Rollout::observe`] captured, empty without one: the
    /// plan's own input buffer, moved here after the plan ran rather than copied.
    frames: Vec<Vec<u8>>,
    /// Whole-frame wall clock (re-pose, upload, trace, readback) and the frames it covers --
    /// the render row of spec 12.4.
    render_wall: Duration,
    rendered: u64,
    /// Width x height of one frame, for `pixels_per_sec`.
    frame_pixels: u64,
}

/// The env on the backend `Rollout` was built for (packets M11/X1, M11/R1): a closed enum over
/// the three engines with a closed-loop path, each arm the generic `Env<B>` monomorphized -- not a trait
/// object and not a new trait (INV-17).
enum Sim {
    Cpu(Env<MuJoCoCpuBackend>),
    Warp(Env<MjWarpBackend>),
    /// Packet M11/R1.
    PhysX(Env<PhysXBackend>),
}

macro_rules! on_env {
    ($sim:expr, $e:ident => $body:expr) => {
        match $sim {
            Sim::Cpu($e) => $body,
            Sim::Warp($e) => $body,
            Sim::PhysX($e) => $body,
        }
    };
}

impl Sim {
    fn model(&self) -> &ModelInfo {
        on_env!(self, e => e.model())
    }
    fn state(&self) -> StateView<'_> {
        on_env!(self, e => e.backend().state())
    }
    fn reset(&mut self, envs: Option<&[u32]>) -> Result<(), es_env::EnvError> {
        on_env!(self, e => e.reset(envs).map(|_| ()))
    }
    fn step(&mut self, ctrl: &[f64]) -> Result<StepOutcome, es_env::EnvError> {
        on_env!(self, e => e.step(ctrl))
    }
    fn metrics(&self) -> EnvMetrics {
        on_env!(self, e => e.metrics())
    }
    /// This env's episode render draws (packet M11/X5), for its renderer.
    #[cfg(feature = "render")]
    fn render_overrides(&self, env: u32) -> &es_env::randomize::RenderOverrides {
        on_env!(self, e => e.render_overrides(env))
    }
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
        Self::with_backend(
            BackendKind::MuJoCoCpu,
            task_toml,
            observation_toml,
            deployment_toml,
            scene_xml,
            seed,
            n_envs,
        )
    }

    /// [`Rollout::new`] on the named backend (packets M11/X1, M11/R1): `mujoco-cpu`, `mjwarp` or
    /// `physx`. Newton is refused by its own `load` -- the mapping report, before a process
    /// spawns.
    pub fn with_backend(
        backend: BackendKind,
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
        let env = match backend {
            BackendKind::MuJoCoCpu => Sim::Cpu(Env::new(
                &task,
                &scene,
                MuJoCoCpuBackend::new(),
                &domains,
                seed,
            )?),
            BackendKind::MjWarp => Sim::Warp(Env::new(
                &task,
                &scene,
                MjWarpBackend::new(),
                &domains,
                seed,
            )?),
            BackendKind::Newton => {
                // Its own load is the gate, and it refuses before spawning (spec 14.4).
                let refused = es_physics_backend::NewtonBackend::new()
                    .load(&scene, &es_physics_core::backend::LoadConfig::default())
                    .err()
                    .map_or_else(|| "loaded".to_owned(), |e| e.to_string());
                return Err(RolloutError::Backend(format!(
                    "backend `newton` has no closed-loop path: {refused}"
                )));
            }
            BackendKind::PhysX => Sim::PhysX(Env::new(
                &task,
                &scene,
                PhysXBackend::new(),
                &domains,
                seed,
            )?),
        };

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
        let initial = previous_action_initial(&task).unwrap_or_default();
        if !initial.is_empty() && initial.len() != NJ {
            return Err(RolloutError::Shape {
                what: "PreviousAction channel",
                expected: NJ,
                got: initial.len(),
            });
        }
        #[cfg(feature = "render")]
        let cameras = cameras(&sources, &task, &scene, n_envs)?;
        #[cfg(feature = "render")]
        let frame_pixels = cameras
            .first()
            .map_or(0, |c| u64::from(c.cfg().width) * u64::from(c.cfg().height));
        #[cfg(not(feature = "render"))]
        let frame_pixels = 0;
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
            previous: vec![initial.clone(); n_envs as usize],
            initial,
            #[cfg(feature = "render")]
            cameras,
            frames: vec![Vec::new(); n_envs as usize],
            render_wall: Duration::ZERO,
            rendered: 0,
            frame_pixels,
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
    /// makes. An image input is served by this env's renderer under the `render` feature
    /// (packet M11/X3) -- one frame per call, which is the tick the `Tick` seed stream counts --
    /// and refused there by name without it: images need a renderer (§4.3).
    pub fn observe(&mut self) -> Result<BTreeMap<String, Vec<f64>>, RolloutError> {
        let light = LightOverride::default();
        let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let model = self.env.model();
        let state = self.env.state();
        for i in 0..self.n_envs {
            let view = env_view(&state, i);
            // This env's renderer as `capture`'s frame source, timed whole-frame.
            // Under this env's episode draws (packet M11/X5): the identity for a task with none.
            #[cfg(feature = "render")]
            let drawn = self.env.render_overrides(i as u32);
            #[cfg(feature = "render")]
            let mut source = self.cameras.get_mut(i).map(|camera| {
                let (wall, rendered) = (&mut self.render_wall, &mut self.rendered);
                move |_: &LightOverride, model: &ModelInfo, state: &StateView<'_>| {
                    let started = std::time::Instant::now();
                    let tile = camera
                        .frame_with(model, state, 0, drawn)
                        .map_err(|e| e.to_string())?;
                    *wall += started.elapsed();
                    *rendered += 1;
                    Ok(tile.to_bytes())
                }
            });
            #[cfg(feature = "render")]
            let frames = source
                .as_mut()
                .map(|f| f as &mut es_eval::runner::FrameSource<'_>);
            #[cfg(not(feature = "render"))]
            let frames = None;
            let (names, mut bytes, _) = capture_at(
                &self.plans[i],
                &self.sources,
                model,
                &view,
                frames,
                &light,
                None,
                &self.previous[i],
            )?;
            {
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
            if let Some(k) = names
                .iter()
                .position(|(n, ..)| matches!(self.sources.get(n), Some(Capture::Image)))
            {
                self.frames[i] = std::mem::take(&mut bytes[k]);
            }
        }
        Ok(out)
    }

    /// The image bytes env `env`'s last [`Self::observe`] captured -- exactly what the plan's
    /// `ImageInput` was handed, in its declared dtype and layout -- or `None` when the
    /// observation has no image input or nothing was observed yet.
    pub fn frame(&self, env: usize) -> Option<&[u8]> {
        self.frames
            .get(env)
            .filter(|f| !f.is_empty())
            .map(Vec::as_slice)
    }

    /// Mean whole-frame render time in milliseconds over every frame so far (re-pose, upload,
    /// dispatch, readback), `None` when nothing was rendered.
    pub fn render_ms_per_frame(&self) -> Option<f64> {
        (self.rendered > 0).then(|| self.render_wall.as_secs_f64() * 1e3 / self.rendered as f64)
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
            let state = self.env.state();
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
        // The sampled rows, before the plane: what the next `observe` reads as the previous
        // action. Set before the resets below, which put `initial` back.
        if !self.initial.is_empty() {
            for (i, prev) in self.previous.iter_mut().enumerate() {
                prev.copy_from_slice(&actions[i * NJ..(i + 1) * NJ]);
            }
        }
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

    /// The §12.4 metric set as the env measured it, with the render row filled from this
    /// rollout's own renderers when it has any (packet M11/X3). A domain this path never runs
    /// (inference, VRAM, and render without an image input) stays `None` rather than a
    /// fabricated zero.
    pub fn metrics(&self) -> EnvMetrics {
        let mut m = self.env.metrics();
        let secs = self.render_wall.as_secs_f64();
        if self.rendered > 0 && secs > 0.0 {
            let frames = self.rendered as f64;
            m.camera_frames_per_sec = Some(frames / secs);
            m.pixels_per_sec = Some(frames * self.frame_pixels as f64 / secs);
        }
        m
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
        let state = self.env.state();
        env_view(&state, env).qpos.to_vec()
    }

    fn begin_episode(&mut self, env: usize) {
        self.planes[env].begin_episode();
        self.plans[env].reset();
        self.previous[env].clone_from(&self.initial);
        // The `Tick` seed clock restarts at *this* env's reset (packet M11/X3).
        #[cfg(feature = "render")]
        if let Some(camera) = self.cameras.get_mut(env) {
            camera.begin_episode();
        }
    }
}

/// One renderer per env for the observation's image input, or none when it has no image
/// (packet M11/X3).
///
/// The Task IR channel that declares the input's sensor says which camera, at which
/// `ImageSpec`, on which render path: `es_env::render::sensor_cfg` turns it into the config --
/// the one function the collector and the evaluator build theirs with -- and `check` refuses a
/// camera that does not produce the declared image rather than resampling it (`INV-14`). One
/// image input: `capture` hands its frame source no input name, and the collector and the
/// evaluator render one image channel too.
#[cfg(feature = "render")]
fn cameras(
    sources: &BTreeMap<String, Capture>,
    task: &es_ir::task::TaskIr,
    scene: &SceneDesc,
    n_envs: u32,
) -> Result<Vec<es_env::EnvRenderer<'static>>, RolloutError> {
    use es_ir::task::ObsSource;
    use es_ir::types::Frame;

    let images: Vec<&String> = sources
        .iter()
        .filter(|(_, c)| matches!(c, Capture::Image))
        .map(|(n, _)| n)
        .collect();
    let name = match images.as_slice() {
        [] => return Ok(Vec::new()),
        [one] => *one,
        more => {
            return Err(RolloutError::Render(format!(
                "the observation has {} image inputs; a rollout renders one camera per env",
                more.len()
            )))
        }
    };
    let sensor = es_core::StableId::from_hex(name)
        .map_err(|e| RolloutError::Render(format!("image input \"{name}\": {e}")))?;
    let (channel, declared) = task
        .observation_spec
        .channels
        .iter()
        .find(|(_, c)| matches!(c.source, ObsSource::Sensor { id, .. } if id == sensor))
        .ok_or_else(|| {
            RolloutError::Render(format!(
                "image input \"{name}\" is declared by no Sensor channel of the Task IR"
            ))
        })?;
    let (ObsSource::Sensor { render, .. }, Some(spec), Frame::Camera(camera)) =
        (&declared.source, declared.ty.image, declared.ty.frame)
    else {
        return Err(RolloutError::Render(format!(
            "image channel \"{channel}\" declares no ImageSpec or no camera frame, so there is \
             no camera to render it from"
        )));
    };
    let cfg = es_env::render::sensor_cfg(camera, &spec, render, None);
    let gpu = gpu()?;
    (0..n_envs)
        .map(|_| {
            let camera = es_env::EnvRenderer::new(gpu, scene, cfg.clone())?;
            camera.check(&spec)?;
            Ok(camera)
        })
        .collect()
}

/// The Vulkan device this thread's rollouts render on: opened once per thread and leaked, so
/// a renderer can borrow it for `'static` and `Rollout` stays a plain owned struct.
///
/// ponytail: one leaked device per thread that builds a rendering rollout (a trainer builds
/// one); give it an owner if a process ever builds rendering rollouts on many threads.
#[cfg(feature = "render")]
fn gpu() -> Result<&'static es_gpu::Gpu, RolloutError> {
    thread_local! {
        static GPU: std::cell::OnceCell<&'static es_gpu::Gpu> =
            const { std::cell::OnceCell::new() };
    }
    GPU.with(|cell| {
        if let Some(gpu) = cell.get() {
            return Ok(*gpu);
        }
        let gpu = es_gpu::Gpu::open(es_gpu::GpuOptions::default())
            .map_err(|e| RolloutError::Render(format!("no Vulkan device: {e}")))?;
        let gpu: &'static es_gpu::Gpu = Box::leak(Box::new(gpu));
        let _ = cell.set(gpu);
        Ok(gpu)
    })
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
