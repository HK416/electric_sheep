//! `MjWarpBackend` — `MuJoCo` Warp, the batched GPU backend (spec 4.3, spec 17.1).
//!
//! Same shape as [`MuJoCoCpuBackend`](crate::MuJoCoCpuBackend): no Rust binding exists and the
//! core runtime must not link Python (spec 2.4), so `python/mjwarp_ref.py` holds the model and
//! this adapter speaks the same line-delimited JSON to it. The difference is what is on the
//! other end — one `Data` with `nworld = n_envs` on a GPU instead of a list of `MjData`, which
//! is the simulation batch of spec 12.1.
//!
//! Two things are declared honestly rather than flatteringly:
//!
//! * **Determinism is tier 2, never tier 1** (spec 17.3): a GPU backend declares cross-backend
//!   tolerance, and only `mujoco-cpu` may claim bitwise. Evidence bundles run on the CPU.
//! * **`mujoco_warp` is young.** Every API name used here is listed as *unverified* in
//!   `docs/api-notes/mujoco-warp.md`; CI has no GPU, so the live test skips (spec 1.7 —
//!   inventing a plausible API name is the cheapest mistake an agent can make).
//!
//! Anything in the scene `MJWarp` cannot map is refused at [`load`](PhysicsBackend::load) by the
//! spec 17.2 mapping report, before a process is spawned (spec 14.4).

use es_assets::scene::SceneDesc;
use std::collections::BTreeMap;

use es_core::{FailureKind, PhysTick, StableId};
use es_physics_core::backend::Param;
use es_physics_core::caps::{BackendQuirk, BatchSupport, DeterminismTier, FloatPrecision};
use es_physics_core::{
    check_requirements, Capabilities, Feature, LoadConfig, ModelInfo, PhysicsBackend, PhysicsError,
    Requirements, StateView, StepReport,
};

use crate::mapping::{mapping_report, BackendKind};
use crate::mjcf_out::scene_to_mjcf;
use crate::proc::{
    applied_values, check_envs, import_available, model_info, param_index, param_wire,
    rate_from_timestep, Ack, AppliedKey, LoadReply, Process, Request, SetParamsReply, StatePayload,
    StateReply, StepReply,
};

/// The backend's name in `es backend compare --backends ...` (spec 17.2).
pub const NAME: &str = "mjwarp";

/// Largest batch declared. Spec 12.1 sizes the simulation domain at 4,096 envs; the ceiling
/// here is device memory, which the capability declaration cannot know, so this is a declared
/// bound and not a measurement (spec 12.4: unverified numbers are targets).
pub const MAX_ENVS: u32 = 8192;

/// The reference script, embedded at build time.
pub const SCRIPT: &str = include_str!("../python/mjwarp_ref.py");

/// What this backend declares (spec 4.3).
///
/// The feature sets *are* the `MJWarp` column of the spec 17.2 table: `MuJoCo` semantics fed by
/// the same MJCF emitter, minus what the table pins narrower. Deriving them from
/// [`crate::mujoco::capabilities`] rather than restating them is what keeps the declaration and
/// the mapping report from drifting apart; a test asserts the two agree feature by feature.
pub fn capabilities() -> Capabilities {
    let cpu = crate::mujoco::capabilities();
    let mut contact = cpu.contact;
    // Unverified against the engine, so not declared (TODO(api-notes)).
    contact.remove(&Feature::ContactCondim6);
    // The inline `<asset><mesh>` mujoco-cpu emits was never run through MJWarp's own convex
    // hull path, so this column does not claim it (packet M10/W2a, a named follow-up).
    contact.remove(&Feature::ContactMesh);
    let mut quirks = cpu.quirks;
    quirks.push(BackendQuirk::new(
        Feature::ContactElliptic,
        "the friction cone is the scene's, pyramidal or elliptic (spec 17.2 footnote); an \
         elliptic scene is a tier 2 row whose contacts differ from mujoco-cpu -- measured on \
         SO-101, the cube's free joint by 0.103 over 1,000 steps while the arm joints agree to \
         1.1e-5 (packet M11/X1)",
    ));
    quirks.push(BackendQuirk::new(
        Feature::JointArmature,
        "state is computed in f32 and widened to f64 at the process boundary, so values agree \
         with mujoco-cpu to a tolerance (spec 3.5 tier 2), never bit for bit",
    ));
    Capabilities {
        name: NAME.to_owned(),
        // spec 17.3: a GPU backend declares tier 2 or 3, never tier 1.
        determinism: DeterminismTier::CrossBackend,
        batch: BatchSupport {
            max_envs: MAX_ENVS,
            gpu_resident: true,
        },
        joints: cpu.joints,
        actuators: cpu.actuators,
        sensors: cpu.sensors,
        contact,
        float: FloatPrecision::F32,
        supports_reset_subset: true,
        supports_state_get_set: true,
        quirks,
    }
}

/// `MuJoCo` Warp behind [`PhysicsBackend`], driven through a Python subprocess.
#[derive(Debug)]
pub struct MjWarpBackend {
    caps: Capabilities,
    process: Option<Process>,
    model: Option<ModelInfo>,
    tick: PhysTick,
    state: StateBuffers,
    /// Where each parameter target lives in the model, built at load (packet M11/X4).
    params: BTreeMap<(Param, StableId), (u32, u32)>,
    /// `[nominal, applied]` of every parameter `set_params` wrote, the applied value read
    /// back out of the world's `f32` model array.
    applied: BTreeMap<AppliedKey, [f64; 2]>,
    /// The load reply's engine version (packet M11/X1); `None` until loaded.
    engine_version: Option<String>,
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

impl Default for MjWarpBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MjWarpBackend {
    pub fn new() -> Self {
        Self {
            caps: capabilities(),
            process: None,
            model: None,
            tick: PhysTick::ZERO,
            state: StateBuffers::default(),
            params: BTreeMap::new(),
            applied: BTreeMap::new(),
            engine_version: None,
        }
    }

    /// `(env, param, id) -> [nominal, applied]` for every parameter [`PhysicsBackend::set_params`]
    /// wrote since the last load, as read back out of that world's model arrays.
    pub fn applied_params(&self) -> &BTreeMap<AppliedKey, [f64; 2]> {
        &self.applied
    }

    /// The engine the loaded process runs, as its load reply named it (packet M11/X1).
    pub fn engine_version(&self) -> Option<&str> {
        self.engine_version.as_deref()
    }

    /// Whether a Python interpreter with `mujoco_warp` and `warp` is available. `Err` explains
    /// what was tried, so a machine without a GPU skips with a reason instead of failing CI.
    pub fn is_available() -> Result<(), String> {
        import_available("mujoco_warp, warp", "`mujoco_warp` and `warp` packages")
    }

    fn process(&mut self) -> Result<&mut Process, PhysicsError> {
        self.process.as_mut().ok_or(PhysicsError::NotLoaded)
    }

    fn info(&self) -> Result<&ModelInfo, PhysicsError> {
        self.model.as_ref().ok_or(PhysicsError::NotLoaded)
    }

    /// Pulls the whole batch's state into the local buffers.
    fn fetch_state(&mut self) -> Result<(), PhysicsError> {
        let reply: StateReply = self.process()?.call_state()?;
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

impl PhysicsBackend for MjWarpBackend {
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
        let report = mapping_report(scene, BackendKind::MjWarp);
        if report.blocked {
            return Err(PhysicsError::Unsupported(report.to_string()));
        }
        // Features are the report's business; what is left for the capability check is the
        // run-level shape of the request (spec 11.6).
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

        let mjcf = scene_to_mjcf(scene)?;
        let rate = match cfg.rate {
            Some(rate) => rate,
            None => rate_from_timestep(scene.options.timestep)?,
        };

        let load = Request::Load {
            mjcf: &mjcf,
            n_envs: cfg.n_envs,
            timestep: Some(rate.period_secs_f64()),
            seed: cfg.seed,
        };
        let (process, reply): (_, LoadReply) = Process::start(SCRIPT, "MuJoCo Warp", &load)?;

        let info = model_info(&reply, scene, "MuJoCo Warp", cfg.n_envs, rate)?;

        self.engine_version = Some(reply.checked_engine_version()?.to_owned());
        self.process = Some(process);
        self.params = param_index(scene, &info);
        self.applied.clear();
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
        match state {
            // As a frame (packet M16/H2c): the same values, not printed and re-parsed.
            Some(s) => self.process()?.reset_frame(
                envs,
                &StatePayload {
                    qpos: s.qpos,
                    qvel: s.qvel,
                    act: s.act,
                },
            )?,
            None => {
                let _: Ack = self
                    .process()?
                    .call(&Request::Reset { envs, state: None })?;
            }
        }
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
        self.process()?.set_ctrl_frame(ctrl)?;
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

    /// `mjwarp_ref.py` edits a per-env CPU `MjModel` exactly as `mujoco-cpu` does (including
    /// `mj_setConst` after a mass edit) and uploads every field that edit moved as that world's
    /// row of a per-world `mujoco_warp` model array (`mujoco_warp` 3.13 indexes model fields
    /// by `worldid % shape[0]`). Until the first call every array keeps its single shared row.
    fn set_params(
        &mut self,
        envs: &[u32],
        params: &[(Param, StableId, f64)],
    ) -> Result<(), PhysicsError> {
        check_envs(envs, self.info()?.n_envs)?;
        let wire = param_wire(&self.params, params)?;
        let reply: SetParamsReply = self.process()?.call(&Request::SetParams {
            envs,
            params: &wire,
        })?;
        self.applied.extend(applied_values(envs, params, &reply)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use es_core::TickRate;
    use es_physics_core::Unsupported;

    use super::*;
    use crate::mapping::{lookup, Status, TaskFeature};
    use crate::proc::parse_response;

    /// Skips the body of a test, with a reason, when `MuJoCo` Warp is not installed (no GPU in
    /// CI, and the package is optional — spec 2.4).
    macro_rules! require_mjwarp {
        ($name:literal) => {
            if let Err(reason) = MjWarpBackend::is_available() {
                eprintln!("SKIP {}: {reason}", $name);
                return;
            }
            eprintln!("RAN {}", $name);
        };
    }

    /// Pyramidal cone (the MJCF default); `pendulum.xml` asks for an elliptic one, which maps
    /// too since the spec 17.2 footnote (packet M11/X1).
    const PENDULUM: &str = r#"<mujoco model="warp-pendulum">
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
        // spec 17.3: a GPU backend declares tier 2 or 3; only mujoco-cpu may claim bitwise.
        assert_eq!(caps.determinism, DeterminismTier::CrossBackend);
        assert_ne!(caps.determinism, DeterminismTier::Bitwise);
        assert!(caps.batch.gpu_resident);
        assert_eq!(caps.batch.max_envs, MAX_ENVS);
        assert_eq!(caps.float, FloatPrecision::F32);
        assert!(caps.has(Feature::JointHinge) && caps.has(Feature::ContactPyramidal));
        // spec 17.2's footnote: the cone is the scene's, elliptic included (tier 2).
        assert!(caps.has(Feature::ContactElliptic));
        assert!(!caps.quirks.is_empty());
    }

    /// The declaration and the spec 17.2 table must say the same thing about every feature.
    #[test]
    fn the_declaration_is_the_mjwarp_column_of_the_mapping() {
        let caps = capabilities();
        for feature in TaskFeature::all() {
            let TaskFeature::Capability(capability) = feature else {
                continue;
            };
            let mapped = matches!(
                lookup(feature, BackendKind::MjWarp).status,
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

    /// spec 14.4: an unmapped row with `severity: error` blocks execution, and it does so
    /// before a process is spawned, so this holds with or without `MuJoCo` Warp installed.
    #[test]
    fn a_blocked_scene_is_refused_with_the_report_as_the_message() {
        // A `general` actuator: the shared emitter cannot write one, so the row blocks.
        let general = es_assets::parse_mjcf(
            r#"<mujoco><worldbody><body name="b">
                 <joint name="j" type="hinge"/><geom name="g" type="sphere" size="0.1"/>
               </body></worldbody>
               <actuator><general name="a" joint="j"/></actuator></mujoco>"#,
        )
        .unwrap()
        .scene;
        let err = MjWarpBackend::new()
            .load(&general, &LoadConfig::default())
            .unwrap_err();
        let PhysicsError::Unsupported(message) = &err else {
            panic!("expected an unsupported failure, got {err:?}");
        };
        assert!(message.contains("ActuatorGeneral"), "{message}");
        assert!(message.contains("blocked: yes"), "{message}");
        assert!(message.contains("spec 17.2"), "{message}");
    }

    #[test]
    fn a_batch_larger_than_declared_is_refused_at_load() {
        let err = MjWarpBackend::new()
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
        assert!(MjWarpBackend::new()
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
        let mut backend = MjWarpBackend::new();
        assert!(backend.model_info().is_none());
        assert_eq!(backend.step(1), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.set_ctrl(&[]), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.reset(None, None), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.state().qpos.len(), 0);
    }

    #[test]
    fn rate_and_timestep_agree() {
        assert_eq!(rate_from_timestep(0.001).unwrap(), TickRate::hz(1000));
        assert!(rate_from_timestep(0.0).is_err());
        assert!(rate_from_timestep(f64::NAN).is_err());
    }

    /// The protocol is exercised with canned lines, so it is tested without Python or a GPU:
    /// `mjwarp_ref.py` answers in exactly the shapes `mujoco_ref.py` does.
    #[test]
    fn the_protocol_round_trips_on_canned_json() {
        let reply: LoadReply = parse_response(
            r#"{"ok":true,"nq":1,"nv":1,"nu":1,"nsensordata":0,"nbody":2,
                "joints":[{"name":"hinge","qpos":[0,1],"dof":[0,1]}],
                "actuators":["m"],"sensors":[],"bodies":["world","rod"],
                "engine_version":"mujoco_warp 3.3.2"}"#,
        )
        .unwrap();
        assert_eq!((reply.nq, reply.nbody), (1, 2));
        assert_eq!(reply.joints[0].dof, [0, 1]);

        // Two envs' worth of state, env-major, as `nworld = 2` produces it.
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
        let err = parse_response::<Ack>(r#"{"ok":false,"error":"ValueError: no model loaded"}"#)
            .unwrap_err();
        assert_eq!(
            err,
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
            "mujoco_warp",
            "put_model",
            "put_data",
            "nworld",
            "mjw.step",
            "mjw.forward",
            "mj_resetData",
        ] {
            assert!(SCRIPT.contains(name), "the script does not mention {name}");
        }
        for cmd in ["set_ctrl", "set_state", "quit"] {
            assert!(SCRIPT.contains(cmd), "the script handles no {cmd}");
        }
    }

    #[test]
    fn mjwarp_pendulum() {
        require_mjwarp!("mjwarp_pendulum");
        let mut backend = MjWarpBackend::new();
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
        eprintln!("mjwarp_pendulum qpos after 100 ticks: {:?}", state.qpos);
        // Each env swings towards the hanging position, and they stay independent.
        assert!(state.qpos[0] < start[0] && state.qpos[1] > start[1]);
        assert_ne!(state.qpos_of(0), state.qpos_of(1));

        backend.reset(None, None).unwrap();
        assert_eq!(backend.state().tick, PhysTick::ZERO);
        assert!(backend.state().qpos.iter().all(|q| q.abs() < 1e-9));
    }

    /// The whole point of the packet: the same scene on both backends, scored with spec 3.5
    /// tier 3 metrics. Needs both engines, so it skips without them.
    /// spec 17.3: `mjwarp` declares tier 2, so tier 2 is what is asserted. Whether the runs
    /// came out bit for bit is *recorded* rather than relied on — a GPU backend that happens to
    /// be reproducible today must not become a test that fails on the next driver.
    #[test]
    fn mjwarp_runs_agree_to_the_declared_tier() {
        require_mjwarp!("mjwarp_runs_agree_to_the_declared_tier");
        let run = || {
            let mut backend = MjWarpBackend::new();
            backend
                .load(
                    &pendulum(),
                    &LoadConfig {
                        n_envs: 2,
                        rate: Some(TickRate::hz(1000)),
                        seed: 1,
                    },
                )
                .unwrap();
            backend
                .set_state(&StateView {
                    n_envs: 2,
                    qpos: &[0.3_f64, -0.2],
                    ..StateView::default()
                })
                .unwrap();
            backend.step(200).unwrap();
            let state = backend.state();
            (state.qpos.to_vec(), state.qvel.to_vec())
        };
        let (qpos_a, qvel_a) = run();
        let (qpos_b, qvel_b) = run();

        let bitwise = qpos_a
            .iter()
            .zip(&qpos_b)
            .all(|(x, y)| x.to_bits() == y.to_bits())
            && qvel_a
                .iter()
                .zip(&qvel_b)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        let delta = qpos_a
            .iter()
            .chain(&qvel_a)
            .zip(qpos_b.iter().chain(&qvel_b))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "mjwarp determinism over 200 ticks: bitwise={bitwise}, max |delta|={delta:e}              (declared tier {:?})",
            capabilities().determinism
        );
        // The declared contract, and all that may be asserted of a GPU backend.
        assert!(delta < 1e-9, "run-to-run delta {delta:e} exceeds tier 2");
    }

    #[test]
    fn mjwarp_against_mujoco_cpu() {
        require_mjwarp!("mjwarp_against_mujoco_cpu");
        if let Err(reason) = crate::MuJoCoCpuBackend::is_available() {
            eprintln!("SKIP mjwarp_against_mujoco_cpu: {reason}");
            return;
        }
        let mut cpu = crate::MuJoCoCpuBackend::new();
        let mut warp = MjWarpBackend::new();
        let report =
            crate::mapping::compare_backends(&mut cpu, &mut warp, &swinging(), &[], 200).unwrap();
        assert_eq!(report.tier_a, DeterminismTier::PhysicsMeaning);
        assert_eq!(report.tier_b, DeterminismTier::CrossBackend);
        assert!(report.mapping_a.is_some() && report.mapping_b.is_some());
        eprintln!("{report}");
        // f32 on the GPU against f64 on the CPU: agreement is a tolerance, not bit equality
        // (spec 17.3). Measured 7.4e-8 on an RTX 4060 Laptop with mujoco-warp 3.13.0; the
        // bound keeps two decades of headroom for other hardware and is a regression guard,
        // not a validated accuracy claim (spec 12.4).
        assert!(report.max_dqpos < 1e-5, "{report}");
    }
}
