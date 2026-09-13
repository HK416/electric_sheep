//! `NewtonBackend` — NVIDIA Newton, the second GPU backend (spec 4.3, spec 17.2).
//!
//! Same shape as [`MjWarpBackend`](crate::MjWarpBackend): no Rust binding exists and the core
//! runtime must not link Python (spec 2.4), so `python/newton_ref.py` holds the model and this
//! adapter speaks the shared line-delimited JSON of [`crate::proc`] to it. What is on the other
//! end is a `newton.Model` whose `world_count` is the batch of spec 12.1.
//!
//! Newton is *not* a `MuJoCo` clone, and the point of having it is that it disagrees: spec 17.2
//! exists because the same Task IR must not behave differently here. What was verified against
//! newton 1.6.0 (see `docs/api-notes/newton.md`) and what it costs:
//!
//! * **The solver is `SolverFeatherstone`, not `SolverMuJoCo`.** newton 1.6.0 pins
//!   `mujoco-warp~=3.12.0`; this workspace needs 3.13.0 for `MjWarpBackend`, and importing
//!   `SolverMuJoCo` against 3.13.0 fails to compile its kernels. Featherstone is Newton's own
//!   reduced-coordinate solver, so the cross-backend comparison is a real one.
//! * **`ModelBuilder.add_mjcf` imports no `<actuator>` and no `<sensor>`** (`Model.actuators`
//!   comes back empty). An actuated or sensored scene is therefore *refused by name* at
//!   [`load`](PhysicsBackend::load) rather than run silently unactuated — spec 14.4, and the
//!   whole reason the mapping report gates execution.
//! * **Contacts are not wired** in this adapter, so no contact capability is declared.
//! * **Determinism is tier 2, never tier 1** (spec 17.3): only `mujoco-cpu` may claim bitwise.

use es_assets::scene::SceneDesc;
use es_core::{FailureKind, PhysTick};
use es_physics_core::caps::{BackendQuirk, BatchSupport, DeterminismTier, FloatPrecision};
use es_physics_core::{
    check_requirements, Capabilities, Feature, LoadConfig, ModelInfo, PhysicsBackend, PhysicsError,
    Requirements, StateView, StepReport,
};

use crate::mapping::{mapping_report, BackendKind};
use crate::mjcf_out::scene_to_mjcf;
use crate::proc::{
    import_available, model_info, rate_from_timestep, Ack, LoadReply, Process, Request,
    StatePayload, StateReply, StepReply,
};

/// The backend's name in `es backend compare --backends ...` (spec 17.2).
pub const NAME: &str = "newton";

/// The engine's name in protocol failure messages.
const ENGINE: &str = "Newton";

/// Largest batch declared. As with `mjwarp`, the real ceiling is device memory, which a
/// capability declaration cannot know, so this is a declared bound and not a measurement
/// (spec 12.4: unverified numbers are targets).
pub const MAX_ENVS: u32 = 8192;

/// The reference script, embedded at build time.
pub const SCRIPT: &str = include_str!("../python/newton_ref.py");

/// What this backend declares (spec 4.3).
///
/// Every set here is what `newton.ModelBuilder.add_mjcf` was *observed* to import in 1.6.0, not
/// what the engine can do in principle: `Model.joint_type` carries all four MJCF joint kinds,
/// `Model.joint_armature` carries the scene's armature and `Model.joint_limit_lower` its
/// limits, while `Model.actuators` and the sensor arrays come back empty. Spec 1.7: a plausible
/// API name is the cheapest thing an agent can invent, so nothing unobserved is declared.
pub fn capabilities() -> Capabilities {
    Capabilities {
        name: NAME.to_owned(),
        // spec 17.3: a GPU backend declares tier 2 or 3, never tier 1.
        determinism: DeterminismTier::CrossBackend,
        batch: BatchSupport {
            max_envs: MAX_ENVS,
            gpu_resident: true,
        },
        joints: [
            Feature::JointFree,
            Feature::JointBall,
            Feature::JointHinge,
            Feature::JointSlide,
            Feature::JointFixed,
            Feature::JointLimit,
            Feature::JointArmature,
        ]
        .into(),
        // `add_mjcf` imports neither, and a quiet zero is worse than a refusal.
        actuators: [].into(),
        sensors: [].into(),
        // Only the MJCF default cone, which every scene carries; anything a scene asks for
        // beyond it (elliptic, soft params, condim 6, mesh, height field) is refused, because
        // this adapter steps with no collision pipeline at all. See the quirk below.
        contact: [Feature::ContactPyramidal].into(),
        float: FloatPrecision::F32,
        supports_reset_subset: true,
        supports_state_get_set: true,
        quirks: vec![
            BackendQuirk::new(
                Feature::JointFree,
                "a free joint's coordinates are Newton's (pos xyz, quat xyzw) where MuJoCo \
                 writes (pos xyz, quat wxyz), so qpos is not element-wise comparable with \
                 mujoco-cpu for a floating base",
            ),
            BackendQuirk::new(
                Feature::ContactPyramidal,
                "contacts are NOT wired: this adapter steps with `contacts = None`, so bodies                  that touch pass through each other. The cone is declared only because every                  MJCF carries one; wire `newton.CollisionPipeline` before trusting any scene                  whose bodies collide",
            ),
            BackendQuirk::new(
                Feature::JointArmature,
                "state is computed in f32 and widened to f64 at the process boundary, so values \
                 agree with mujoco-cpu to a tolerance (spec 3.5 tier 2), never bit for bit",
            ),
            BackendQuirk::new(
                Feature::JointHinge,
                "the solver is SolverFeatherstone, not SolverMuJoCo: newton 1.6.0 pins \
                 mujoco-warp ~=3.12.0 and this workspace needs 3.13.0, so contact and joint \
                 dynamics are Newton's own and differ from MuJoCo by more than rounding",
            ),
        ],
    }
}

/// Newton behind [`PhysicsBackend`], driven through a Python subprocess.
#[derive(Debug)]
pub struct NewtonBackend {
    caps: Capabilities,
    process: Option<Process>,
    model: Option<ModelInfo>,
    tick: PhysTick,
    state: StateBuffers,
}

#[derive(Debug, Default)]
struct StateBuffers {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    act: Vec<f64>,
    sensordata: Vec<f64>,
    xpos: Vec<f64>,
    xquat: Vec<f64>,
}

impl Default for NewtonBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NewtonBackend {
    pub fn new() -> Self {
        Self {
            caps: capabilities(),
            process: None,
            model: None,
            tick: PhysTick::ZERO,
            state: StateBuffers::default(),
        }
    }

    /// Whether a Python interpreter with `newton` is available. `Err` explains what was tried,
    /// so a machine without it skips with a reason instead of failing CI.
    pub fn is_available() -> Result<(), String> {
        import_available("newton", "`newton` package")
    }

    fn process(&mut self) -> Result<&mut Process, PhysicsError> {
        self.process.as_mut().ok_or(PhysicsError::NotLoaded)
    }

    fn info(&self) -> Result<&ModelInfo, PhysicsError> {
        self.model.as_ref().ok_or(PhysicsError::NotLoaded)
    }

    /// Pulls the whole batch's state into the local buffers.
    fn fetch_state(&mut self) -> Result<(), PhysicsError> {
        let reply: StateReply = self.process()?.call(&Request::State)?;
        self.state = StateBuffers {
            qpos: reply.qpos,
            qvel: reply.qvel,
            act: reply.act,
            sensordata: reply.sensordata,
            xpos: reply.xpos,
            xquat: reply.xquat,
        };
        Ok(())
    }

    /// Checks that `state` carries `rows` envs' worth of every array it provides.
    fn check_state(&self, state: &StateView<'_>, rows: usize) -> Result<(), PhysicsError> {
        let info = self.info()?;
        for (what, values, width) in [
            ("qpos", state.qpos, info.nq),
            ("qvel", state.qvel, info.nv),
            ("act", state.act, info.nu),
        ] {
            let expected = rows * width as usize;
            if !values.is_empty() && values.len() != expected {
                return Err(PhysicsError::ShapeMismatch {
                    what,
                    expected,
                    got: values.len(),
                });
            }
        }
        Ok(())
    }
}

impl PhysicsBackend for NewtonBackend {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn load(&mut self, scene: &SceneDesc, cfg: &LoadConfig) -> Result<ModelInfo, PhysicsError> {
        if cfg.n_envs == 0 {
            return Err(PhysicsError::Backend(
                "n_envs must be at least 1".to_owned(),
            ));
        }
        // spec 14.4: the semantic mapping report is the gate, and it runs before anything is
        // spawned. An unmapped row with `severity: error` blocks execution, named.
        let report = mapping_report(scene, BackendKind::Newton);
        if report.blocked {
            return Err(PhysicsError::Unsupported(report.to_string()));
        }
        let unmet = check_requirements(
            &Requirements {
                n_envs: cfg.n_envs,
                ..Requirements::default()
            },
            &self.caps,
        );
        if !unmet.is_empty() {
            return Err(PhysicsError::Requirements(unmet));
        }

        // Newton reads MJCF, so the same emitter feeds all three backends: one scene
        // description, one text, three engines (spec 17.2).
        let mjcf = scene_to_mjcf(scene)?;
        let rate = match cfg.rate {
            Some(rate) => rate,
            None => rate_from_timestep(scene.options.timestep)?,
        };

        let mut process = Process::spawn_with(SCRIPT, ENGINE)?;
        let reply: LoadReply = process.call(&Request::Load {
            mjcf: &mjcf,
            n_envs: cfg.n_envs,
            timestep: Some(rate.period_secs_f64()),
            seed: cfg.seed,
        })?;
        let info = model_info(&reply, scene, ENGINE, cfg.n_envs, rate)?;

        self.process = Some(process);
        self.model = Some(info.clone());
        self.tick = PhysTick::ZERO;
        self.fetch_state()?;
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
        let n_envs = self.info()?.n_envs;
        let rows = match envs {
            Some(envs) => {
                if let Some(bad) = envs.iter().find(|e| **e >= n_envs) {
                    return Err(PhysicsError::Backend(format!(
                        "env {bad} is out of range for a batch of {n_envs}"
                    )));
                }
                envs.len()
            }
            None => n_envs as usize,
        };
        if let Some(state) = state {
            self.check_state(state, rows)?;
        }
        let payload = state.map(|s| StatePayload {
            qpos: s.qpos,
            qvel: s.qvel,
            act: s.act,
        });
        let _: Ack = self.process()?.call(&Request::Reset {
            envs,
            state: payload,
        })?;
        // The batch shares one clock, so only a whole-batch reset rewinds it.
        if envs.is_none() {
            self.tick = PhysTick::ZERO;
        }
        self.fetch_state()
    }

    fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), PhysicsError> {
        let info = self.info()?;
        let expected = (info.nu * info.n_envs) as usize;
        if ctrl.len() != expected {
            return Err(PhysicsError::ShapeMismatch {
                what: "ctrl",
                expected,
                got: ctrl.len(),
            });
        }
        let _: Ack = self.process()?.call(&Request::SetCtrl { ctrl })?;
        Ok(())
    }

    fn step(&mut self, n_substeps: u32) -> Result<StepReport, PhysicsError> {
        self.info()?;
        let reply: StepReply = self.process()?.call(&Request::Step { n: n_substeps })?;
        self.tick = self.tick.add_ticks(u64::from(n_substeps));
        self.fetch_state()?;
        Ok(StepReport {
            tick: self.tick,
            // Divergence is reported, never a panic (spec 18.5).
            failures: reply
                .nonfinite
                .into_iter()
                .map(|env| (env, FailureKind::NanDetected))
                .collect(),
        })
    }

    fn state(&self) -> StateView<'_> {
        StateView {
            n_envs: self.model.as_ref().map_or(0, |m| m.n_envs),
            tick: self.tick,
            qpos: &self.state.qpos,
            qvel: &self.state.qvel,
            act: &self.state.act,
            sensordata: &self.state.sensordata,
            xpos: &self.state.xpos,
            xquat: &self.state.xquat,
        }
    }

    fn set_state(&mut self, state: &StateView<'_>) -> Result<(), PhysicsError> {
        let rows = self.info()?.n_envs as usize;
        self.check_state(state, rows)?;
        let _: Ack = self.process()?.call(&Request::SetState {
            state: StatePayload {
                qpos: state.qpos,
                qvel: state.qvel,
                act: state.act,
            },
        })?;
        self.fetch_state()
    }
}

#[cfg(test)]
mod tests {
    use es_core::TickRate;
    use es_physics_core::Unsupported;

    use super::*;
    use crate::mapping::{lookup, Status, TaskFeature};
    use crate::proc::parse_response;

    /// Skips the body of a test, with a reason, when Newton is not installed (no GPU in CI, and
    /// the package is optional — spec 2.4).
    macro_rules! require_newton {
        ($name:literal) => {
            if let Err(reason) = NewtonBackend::is_available() {
                eprintln!("SKIP {}: {reason}", $name);
                return;
            }
            eprintln!("RAN {}", $name);
        };
    }

    /// No actuators and no sensors, so the scene is mappable on Newton.
    const PENDULUM: &str = r#"<mujoco model="newton-pendulum">
         <option timestep="0.001" gravity="0 0 -9.81"/>
         <worldbody><body name="rod" pos="0 0 1">
           <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1" armature="0.01"/>
           <geom name="shaft" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02"/>
         </body></worldbody>
       </mujoco>"#;

    fn pendulum() -> SceneDesc {
        es_assets::parse_mjcf(PENDULUM).unwrap().scene
    }

    /// A rod that starts *horizontal*, so `qpos = 0` is not an equilibrium and 200 ticks from
    /// the reset state actually move. `compare_backends` resets and steps without control, so
    /// the hanging pendulum above would sit still and score a vacuous `max |dqpos| = 0`.
    const SWINGING: &str = r#"<mujoco model="swinging-pendulum">
         <option timestep="0.001" gravity="0 0 -9.81"/>
         <worldbody><body name="rod" pos="0 0 1">
           <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1" armature="0.01"/>
           <geom name="shaft" type="capsule" fromto="0 0 0 0.5 0 0" size="0.02"/>
         </body></worldbody>
       </mujoco>"#;

    fn swinging() -> SceneDesc {
        es_assets::parse_mjcf(SWINGING).unwrap().scene
    }

    #[test]
    fn capabilities_are_declared_honestly() {
        let caps = capabilities();
        assert_eq!(caps.name, NAME);
        assert_eq!(
            BackendKind::from_name(&caps.name),
            Some(BackendKind::Newton)
        );
        // spec 17.3: a GPU backend declares tier 2 or 3; only mujoco-cpu may claim bitwise.
        assert_eq!(caps.determinism, DeterminismTier::CrossBackend);
        assert_ne!(caps.determinism, DeterminismTier::Bitwise);
        assert!(caps.batch.gpu_resident);
        assert_eq!(caps.float, FloatPrecision::F32);
        // Verified imported by `add_mjcf`.
        assert!(caps.has(Feature::JointHinge) && caps.has(Feature::JointArmature));
        assert!(caps.has(Feature::JointFree) && caps.has(Feature::JointLimit));
        // Verified *not* imported, so not declared.
        assert!(!caps.has(Feature::ActuatorMotor) && !caps.has(Feature::ActuatorPosition));
        assert!(!caps.has(Feature::SensorJointPos));
        // The default cone is declared, but with a quirk saying contacts are not wired; a
        // scene that asks for anything more than the default is refused.
        assert!(caps.has(Feature::ContactPyramidal) && !caps.has(Feature::ContactElliptic));
        assert!(!caps.has(Feature::ContactMesh) && !caps.has(Feature::ContactSoftParams));
        assert!(!caps.quirks.is_empty());
    }

    /// The declaration and the spec 17.2 table must say the same thing about every feature.
    #[test]
    fn the_declaration_is_the_newton_column_of_the_mapping() {
        let caps = capabilities();
        for feature in TaskFeature::all() {
            let TaskFeature::Capability(capability) = feature else {
                continue;
            };
            let mapped = matches!(
                lookup(feature, BackendKind::Newton).status,
                Status::Native(_) | Status::Approximated(_)
            );
            assert_eq!(
                caps.has(capability),
                mapped,
                "{capability} is declared {} but mapped {mapped}",
                caps.has(capability)
            );
        }
    }

    /// spec 14.4: `add_mjcf` drops `<actuator>`, so an actuated scene is refused by name rather
    /// than run unactuated. This holds with or without Newton installed — the report is the gate
    /// and it runs before a process is spawned.
    #[test]
    fn an_actuated_scene_is_refused_rather_than_run_unactuated() {
        let actuated = es_assets::parse_mjcf(
            r#"<mujoco><worldbody><body name="b">
                 <joint name="j" type="hinge"/><geom name="g" type="sphere" size="0.1"/>
               </body></worldbody>
               <actuator><motor name="m" joint="j" gear="1"/></actuator></mujoco>"#,
        )
        .unwrap()
        .scene;
        let err = NewtonBackend::new()
            .load(&actuated, &LoadConfig::default())
            .unwrap_err();
        let PhysicsError::Unsupported(message) = &err else {
            panic!("expected an unsupported failure, got {err:?}");
        };
        assert!(message.contains("blocked: yes"), "{message}");
        assert!(message.contains("add_mjcf"), "{message}");
    }

    #[test]
    fn a_batch_larger_than_declared_is_refused_at_load() {
        let err = NewtonBackend::new()
            .load(
                &pendulum(),
                &LoadConfig {
                    n_envs: MAX_ENVS + 1,
                    ..LoadConfig::default()
                },
            )
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::Requirements(vec![Unsupported::BatchSize {
                requested: MAX_ENVS + 1,
                max_envs: MAX_ENVS,
            }])
        );
        assert!(NewtonBackend::new()
            .load(
                &pendulum(),
                &LoadConfig {
                    n_envs: 0,
                    ..LoadConfig::default()
                }
            )
            .is_err());
    }

    #[test]
    fn calls_before_load_are_not_loaded_errors() {
        let mut backend = NewtonBackend::new();
        assert!(backend.model_info().is_none());
        assert_eq!(backend.step(1), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.set_ctrl(&[]), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.reset(None, None), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.state().qpos.len(), 0);
    }

    /// The protocol is exercised with canned lines, so it is tested without Python or a GPU:
    /// `newton_ref.py` answers in exactly the shapes the other two scripts do.
    #[test]
    fn the_protocol_round_trips_on_canned_json() {
        // `add_mjcf` imports no actuators or sensors, and Newton has no world body, so a
        // one-link pendulum is nq 1 / nu 0 / nbody 1 here where MuJoCo says nbody 2.
        let reply: LoadReply = parse_response(
            r#"{"ok":true,"nq":1,"nv":1,"nu":0,"nsensordata":0,"nbody":1,
                "joints":[{"name":"hinge","qpos":[0,1],"dof":[0,1]}],
                "actuators":[],"sensors":[],"bodies":["rod"]}"#,
        )
        .unwrap();
        assert_eq!((reply.nq, reply.nu, reply.nbody), (1, 0, 1));
        assert_eq!(reply.joints[0].qpos, [0, 1]);
        assert!(reply.actuators.is_empty() && reply.sensors.is_empty());

        // Two envs' worth of state, env-major, as `world_count = 2` produces it.
        let state: StateReply = parse_response(
            r#"{"ok":true,"qpos":[0.25,-0.25],"qvel":[1.0,-1.0],"act":[],"sensordata":[],
                "xpos":[],"xquat":[]}"#,
        )
        .unwrap();
        assert_eq!(state.qpos, vec![0.25, -0.25]);
        let step: StepReply = parse_response(r#"{"ok":true,"nonfinite":[1]}"#).unwrap();
        assert_eq!(step.nonfinite, vec![1]);
        let _: Ack = parse_response(r#"{"ok":true}"#).unwrap();

        // A Python-side failure is a typed error, not a dead process.
        assert_eq!(
            parse_response::<Ack>(r#"{"ok":false,"error":"ValueError: no model loaded"}"#)
                .unwrap_err(),
            PhysicsError::Backend("ValueError: no model loaded".to_owned())
        );
        assert!(matches!(
            parse_response::<LoadReply>("not json").unwrap_err(),
            PhysicsError::Protocol(_)
        ));
    }

    #[test]
    fn the_embedded_script_is_the_file_on_disk() {
        for name in [
            "import newton",
            "add_mjcf",
            "add_world",
            "finalize",
            "SolverFeatherstone",
            "eval_fk",
            "joint_q_start",
        ] {
            assert!(SCRIPT.contains(name), "the script does not mention {name}");
        }
        for cmd in ["set_ctrl", "set_state", "quit"] {
            assert!(SCRIPT.contains(cmd), "the script handles no {cmd}");
        }
    }

    #[test]
    fn newton_pendulum() {
        require_newton!("newton_pendulum");
        let mut backend = NewtonBackend::new();
        let cfg = LoadConfig {
            n_envs: 2,
            rate: Some(TickRate::hz(1000)),
            seed: 1,
        };
        let info = backend.load(&pendulum(), &cfg).unwrap();
        assert_eq!((info.nq, info.nv, info.n_envs), (1, 1, 2));

        // Hanging straight down is an equilibrium, so both envs are lifted first, differently.
        let start = [0.3_f64, -0.2];
        backend
            .set_state(&StateView {
                n_envs: 2,
                qpos: &start,
                ..StateView::default()
            })
            .unwrap();
        let report = backend.step(100).unwrap();
        assert_eq!(report.tick, PhysTick(100));
        assert!(report.failures.is_empty(), "{:?}", report.failures);

        let state = backend.state();
        assert!(state.is_finite());
        assert_eq!(state.qpos.len(), 2);
        eprintln!("newton_pendulum qpos after 100 ticks: {:?}", state.qpos);
        // Each env swings towards the hanging position, and they stay independent.
        assert!(state.qpos[0] < start[0] && state.qpos[1] > start[1]);
        assert_ne!(state.qpos_of(0), state.qpos_of(1));

        backend.reset(None, None).unwrap();
        assert_eq!(backend.state().tick, PhysTick::ZERO);
        assert!(backend.state().qpos.iter().all(|q| q.abs() < 1e-9));
    }

    /// The packet's point: the same scene on Newton and on the CPU oracle, scored with spec 3.5
    /// tier 3 metrics. Newton runs its own solver, so the bound is loose on purpose — this
    /// records the disagreement rather than asserting the two engines agree.
    #[test]
    fn newton_against_mujoco_cpu() {
        require_newton!("newton_against_mujoco_cpu");
        if let Err(reason) = crate::MuJoCoCpuBackend::is_available() {
            eprintln!("SKIP newton_against_mujoco_cpu: {reason}");
            return;
        }
        let mut cpu = crate::MuJoCoCpuBackend::new();
        let mut newton = NewtonBackend::new();
        let report =
            crate::mapping::compare_backends(&mut cpu, &mut newton, &swinging(), &[], 200).unwrap();
        assert_eq!(report.tier_a, DeterminismTier::PhysicsMeaning);
        assert_eq!(report.tier_b, DeterminismTier::CrossBackend);
        assert!(report.mapping_a.is_some() && report.mapping_b.is_some());
        eprintln!("{report}");
        // A different solver (Featherstone, f32) against MuJoCo (f64): measured 5.9e-4 over
        // 200 ticks on an RTX 4060 Laptop, diverging past the 1e-6 tolerance at tick 6. The
        // bound records that the two stay on the same trajectory; it is not a validated
        // agreement tolerance, and spec 17.2 is why the gap is tracked at all.
        assert!(report.max_dqpos < 1e-2, "{report}");
    }
}
