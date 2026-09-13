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

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use es_assets::scene::SceneDesc;
use es_core::{FailureKind, PhysTick, StableId, TickRate};
use es_physics_core::caps::{BackendQuirk, BatchSupport, DeterminismTier, FloatPrecision};
use es_physics_core::{
    check_requirements, Capabilities, Feature, IndexRange, LoadConfig, ModelInfo, PhysicsBackend,
    PhysicsError, Requirements, StateView, StepReport,
};
use serde::de::DeserializeOwned;

use crate::mapping::{mapping_report, BackendKind};
use crate::mjcf_out::scene_to_mjcf;
use crate::proc::{parse_response, Ack, LoadReply, Request, StatePayload, StateReply, StepReply};

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
    // spec 17.2 maps MJWarp's friction cone to pyramidal.
    contact.remove(&Feature::ContactElliptic);
    // Unverified against the engine, so not declared (TODO(api-notes)).
    contact.remove(&Feature::ContactCondim6);
    let mut quirks = cpu.quirks;
    quirks.push(BackendQuirk::new(
        Feature::ContactPyramidal,
        "spec 17.2 pins MJWarp's friction cone to pyramidal; an elliptic scene is refused by \
         name rather than silently re-coned",
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
    process: Option<WarpProcess>,
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
        }
    }

    /// Whether a Python interpreter with `mujoco_warp` and `warp` is available. `Err` explains
    /// what was tried, so a machine without a GPU skips with a reason instead of failing CI.
    pub fn is_available() -> Result<(), String> {
        let mut tried = Vec::new();
        for python in python_candidates() {
            match Command::new(&python)
                .args(["-c", "import mujoco_warp, warp"])
                .output()
            {
                Ok(out) if out.status.success() => return Ok(()),
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    tried.push(format!(
                        "`{python}`: {}",
                        stderr.lines().last().unwrap_or("import failed").trim()
                    ));
                }
                Err(e) => tried.push(format!("`{python}`: {e}")),
            }
        }
        Err(format!(
            "no Python interpreter with the `mujoco_warp` and `warp` packages (set ES_PYTHON to \
             choose one): {}",
            tried.join("; ")
        ))
    }

    fn process(&mut self) -> Result<&mut WarpProcess, PhysicsError> {
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

// ponytail: `proc.rs` hardcodes `mujoco_ref.py`, and this packet may not edit it, so the
// spawn / call / drop trio is duplicated here. Fold both into one `Process::spawn_with(SCRIPT)`
// when a third out-of-process backend appears.

/// Interpreters to try, in order. `ES_PYTHON` overrides the search entirely.
fn python_candidates() -> Vec<String> {
    match std::env::var("ES_PYTHON") {
        Ok(path) if !path.trim().is_empty() => vec![path],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    }
}

/// A running `mjwarp_ref.py`.
#[derive(Debug)]
struct WarpProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl WarpProcess {
    fn spawn() -> Result<Self, PhysicsError> {
        let mut tried = Vec::new();
        for python in python_candidates() {
            let spawned = Command::new(&python)
                .args(["-c", SCRIPT])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                // Every Python-side failure is reported on stdout as JSON.
                .stderr(Stdio::null())
                .spawn();
            match spawned {
                Ok(mut child) => {
                    let stdin = child.stdin.take().expect("stdin was piped");
                    let stdout = child.stdout.take().expect("stdout was piped");
                    return Ok(Self {
                        child,
                        stdin,
                        stdout: BufReader::new(stdout),
                    });
                }
                Err(e) => tried.push(format!("`{python}`: {e}")),
            }
        }
        Err(PhysicsError::Backend(format!(
            "cannot start the MuJoCo Warp reference process: {}",
            tried.join("; ")
        )))
    }

    fn call<T: DeserializeOwned>(&mut self, request: &Request<'_>) -> Result<T, PhysicsError> {
        let line = serde_json::to_string(request)
            .map_err(|e| PhysicsError::Protocol(format!("cannot encode request: {e}")))?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .map_err(|e| PhysicsError::ProcessDied(e.to_string()))?;

        let mut reply = String::new();
        match self.stdout.read_line(&mut reply) {
            Ok(0) => Err(PhysicsError::ProcessDied(
                "the process closed its output without answering".to_owned(),
            )),
            Ok(_) => parse_response(&reply),
            Err(e) => Err(PhysicsError::ProcessDied(e.to_string())),
        }
    }
}

impl Drop for WarpProcess {
    fn drop(&mut self) {
        let _ = self
            .stdin
            .write_all(b"{\"cmd\":\"quit\"}\n")
            .and_then(|()| self.stdin.flush());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The tick rate a timestep in seconds stands for (spec 18.1: integer ticks are the model).
fn rate_from_timestep(timestep: f64) -> Result<TickRate, PhysicsError> {
    let nanos = (timestep * 1e9).round();
    if !(nanos.is_finite() && nanos >= 1.0) {
        return Err(PhysicsError::Backend(format!(
            "timestep {timestep} is not a positive number of nanoseconds"
        )));
    }
    let divisor = gcd(1_000_000_000, nanos as u64);
    TickRate::rational(1_000_000_000 / divisor, nanos as u64 / divisor)
        .map_err(|e| PhysicsError::Backend(e.to_string()))
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn index_by_name<'a, T: 'a>(
    items: impl IntoIterator<Item = &'a T>,
    field: impl Fn(&'a T) -> (&'a str, StableId),
) -> BTreeMap<&'a str, StableId> {
    items.into_iter().map(field).collect()
}

fn id_of(map: &BTreeMap<&str, StableId>, kind: &str, name: &str) -> Result<StableId, PhysicsError> {
    map.get(name).copied().ok_or_else(|| {
        PhysicsError::Protocol(format!(
            "MuJoCo Warp reported a {kind} `{name}` the scene does not have"
        ))
    })
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

        let mut process = WarpProcess::spawn()?;
        let reply: LoadReply = process.call(&Request::Load {
            mjcf: &mjcf,
            n_envs: cfg.n_envs,
            timestep: Some(rate.period_secs_f64()),
            seed: cfg.seed,
        })?;

        let joints = index_by_name(&scene.joints, |j| (j.name.as_str(), j.id));
        let actuators = index_by_name(&scene.actuators, |a| (a.name.as_str(), a.id));
        let sensors = index_by_name(&scene.sensors, |s| (s.name.as_str(), s.id));
        let bodies = index_by_name(&scene.bodies, |b| (b.name.as_str(), b.id));

        let mut info = ModelInfo {
            nq: reply.nq,
            nv: reply.nv,
            nu: reply.nu,
            nsensordata: reply.nsensordata,
            nbody: reply.nbody,
            n_envs: cfg.n_envs,
            rate,
            ..ModelInfo::default()
        };
        for joint in &reply.joints {
            let id = id_of(&joints, "joint", &joint.name)?;
            info.qpos
                .insert(id, IndexRange::new(joint.qpos[0], joint.qpos[1]));
            info.dof
                .insert(id, IndexRange::new(joint.dof[0], joint.dof[1]));
        }
        for (index, name) in reply.actuators.iter().enumerate() {
            let id = id_of(&actuators, "actuator", name)?;
            info.actuator.insert(id, IndexRange::new(index as u32, 1));
        }
        for sensor in &reply.sensors {
            let id = id_of(&sensors, "sensor", &sensor.name)?;
            info.sensor
                .insert(id, IndexRange::new(sensor.adr, sensor.dim));
        }
        for (index, name) in reply.bodies.iter().enumerate() {
            let id = id_of(&bodies, "body", name)?;
            info.body.insert(id, IndexRange::new(index as u32, 1));
        }

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
    use es_physics_core::Unsupported;

    use super::*;
    use crate::mapping::{lookup, Status, TaskFeature};

    /// Skips the body of a test, with a reason, when `MuJoCo` Warp is not installed (no GPU in
    /// CI, and the package is optional — spec 2.4).
    macro_rules! require_mjwarp {
        ($name:literal) => {
            if let Err(reason) = MjWarpBackend::is_available() {
                eprintln!("SKIP {}: {reason}", $name);
                return;
            }
        };
    }

    /// Pyramidal cone (the MJCF default), so the scene is mappable on `MJWarp`; `pendulum.xml`
    /// asks for an elliptic one and is refused by design.
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
        // spec 17.2 pins the cone to pyramidal.
        assert!(!caps.has(Feature::ContactElliptic));
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
        let elliptic = es_assets::parse_mjcf(
            r#"<mujoco><option cone="elliptic"/><worldbody><body name="b">
                 <joint name="j" type="hinge"/><geom name="g" type="sphere" size="0.1"/>
               </body></worldbody></mujoco>"#,
        )
        .unwrap()
        .scene;
        let err = MjWarpBackend::new()
            .load(&elliptic, &LoadConfig::default())
            .unwrap_err();
        let PhysicsError::Unsupported(message) = &err else {
            panic!("expected an unsupported failure, got {err:?}");
        };
        assert!(message.contains("ContactElliptic"), "{message}");
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
                "actuators":["m"],"sensors":[],"bodies":["world","rod"]}"#,
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
        // Each env swings towards the hanging position, and they stay independent.
        assert!(state.qpos[0] < start[0] && state.qpos[1] > start[1]);
        assert_ne!(state.qpos_of(0), state.qpos_of(1));

        backend.reset(None, None).unwrap();
        assert_eq!(backend.state().tick, PhysTick::ZERO);
        assert!(backend.state().qpos.iter().all(|q| q.abs() < 1e-9));
    }

    /// The whole point of the packet: the same scene on both backends, scored with spec 3.5
    /// tier 3 metrics. Needs both engines, so it skips without them.
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
            crate::mapping::compare_backends(&mut cpu, &mut warp, &pendulum(), &[], 200).unwrap();
        assert_eq!(report.tier_a, DeterminismTier::PhysicsMeaning);
        assert_eq!(report.tier_b, DeterminismTier::CrossBackend);
        assert!(report.mapping_a.is_some() && report.mapping_b.is_some());
        // f32 on the GPU against f64 on the CPU: agreement is a tolerance, not bit equality
        // (spec 17.3). The number is a smoke bound, not a validated one.
        assert!(report.max_dqpos < 1e-2, "{report}");
    }
}
