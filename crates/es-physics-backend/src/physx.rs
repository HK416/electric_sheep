//! `PhysXBackend` — NVIDIA `PhysX` through Isaac Sim, headless (packet M11/I1, spec 28.14 rule 6,
//! spec 17.1-17.2).
//!
//! Isaac Sim is a Python application, and no Rust crate may import it: the stage lives in
//! `python/physx_ref.py`, which speaks the same line-delimited JSON as the other
//! out-of-process backends. The scene reaches it as the MJCF [`scene_to_mjcf`] emits, imported
//! by Isaac Sim's own MJCF importer; `physx_ref.py` repairs what the importer gets wrong (every
//! repair is a [`BackendQuirk`] below) and the mapping report names what it cannot repair
//! (`crate::mapping`, the `PhysX` column), both measured in `docs/api-notes/isaac-sim.md`.
//!
//! * `ES_ISAAC_PYTHON` names the Isaac Sim interpreter (not `ES_PYTHON`'s). Unset or unusable
//!   is [`PhysXBackend::is_available`]'s `Err`, the callers' documented SKIPPED path.
//! * The process gets `OMNI_KIT_ACCEPT_EULA=YES` (the owner accepted the EULA on 2026-09-23)
//!   and, where the host needs them, the compat libraries of api-note 7.2 on
//!   `LD_LIBRARY_PATH`: `ES_ISAAC_COMPAT_LIBS`, else `$HOME/opt/isaac-compat/root/usr/lib/
//!   x86_64-linux-gnu` when it exists. The script picks the physics-only experience itself.
//! * `ES_PHYSX_DEVICE` picks the pipeline: `cpu` (the default) or `cuda:N`. It is part of the
//!   engine version, so a GPU run hashes apart from a CPU one (spec 28.14 rule 2).
//! * `ES_PHYSX_STDERR`, when set, is a file that receives Kit's log (otherwise discarded).

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use es_assets::scene::SceneDesc;
use es_core::{FailureKind, PhysTick};
use es_physics_core::caps::{BackendQuirk, BatchSupport, DeterminismTier, FloatPrecision};
use es_physics_core::{
    check_requirements, Capabilities, Feature, LoadConfig, ModelInfo, PhysicsBackend, PhysicsError,
    Requirements, StateView, StepReport,
};

use crate::mapping::{lookup, mapping_report, BackendKind, Status, TaskFeature};
use crate::mjcf_out::scene_to_mjcf;
use crate::proc::{
    model_info, rate_from_timestep, Ack, LoadReply, Process, Request, StatePayload, StateReply,
    StepReply,
};

/// The backend's name in `--backend` / `--backends` (spec 17.2).
pub const NAME: &str = "physx";

/// Largest batch declared: a bound, not a measurement (spec 12.4). Every step crosses the
/// process boundary as JSON, so this path is a reference, not a throughput one.
pub const MAX_ENVS: u32 = 1024;

/// The script, embedded at build time and written to a file at spawn (Kit cannot run from
/// `python -c`).
pub const SCRIPT: &str = include_str!("../python/physx_ref.py");

/// The CPU pipeline's declaration, the default.
pub fn capabilities() -> Capabilities {
    capabilities_on(false)
}

/// What this backend declares on the CPU or the GPU pipeline (spec 4.3).
///
/// The feature sets are the `PhysX` column of the mapping (`Native` or `Approximated`), derived
/// from it rather than restated, so the two cannot drift apart.
pub fn capabilities_on(gpu: bool) -> Capabilities {
    let mut caps = Capabilities {
        name: NAME.to_owned(),
        // spec 17.3: only mujoco-cpu claims bitwise. PhysX CPU reruns were bitwise (I0), CPU
        // against GPU is not (api-note 7.4).
        determinism: DeterminismTier::CrossBackend,
        batch: BatchSupport {
            max_envs: MAX_ENVS,
            gpu_resident: gpu,
        },
        joints: BTreeSet::default(),
        actuators: BTreeSet::default(),
        sensors: BTreeSet::default(),
        contact: BTreeSet::default(),
        float: FloatPrecision::F32,
        supports_reset_subset: true,
        supports_state_get_set: true,
        quirks: quirks(),
    };
    for feature in TaskFeature::all() {
        let TaskFeature::Capability(capability) = feature else {
            continue;
        };
        if matches!(
            lookup(feature, BackendKind::PhysX).status,
            Status::Native(_) | Status::Approximated(_)
        ) {
            // `Capabilities::has` reads all four sets; the split only groups them by name.
            let name = capability.to_string();
            let set = if name.starts_with("Joint") {
                &mut caps.joints
            } else if name.starts_with("Actuator") {
                &mut caps.actuators
            } else if name.starts_with("Sensor") {
                &mut caps.sensors
            } else {
                &mut caps.contact
            };
            set.insert(capability);
        }
    }
    caps
}

/// What `physx_ref.py` repairs so that `MuJoCo` semantics hold, each on the feature it bites
/// (api-note section 8 has the measurement behind every line).
fn quirks() -> Vec<BackendQuirk> {
    vec![
        BackendQuirk::new(
            Feature::ActuatorPosition,
            "drives authored by the adapter through the PhysX tensor API in SI units: stiffness \
             = kp, damping = kv, max force = forcerange (the importer writes kp unconverted into \
             USD's per-degree gain and drops kv); ctrl is clamped to ctrlrange by the adapter \
             (the importer drops ctrlrange)",
        ),
        BackendQuirk::new(
            Feature::JointHinge,
            "World.reset() runs two hidden physics steps; the adapter writes qpos0 and zero \
             velocity after it, so tick 0 is MuJoCo's tick 0",
        ),
        BackendQuirk::new(
            Feature::JointFree,
            "a free body is imported as a plain rigid body (the importer's fix_base weld and \
             articulation root removed); qpos quaternion w-first, linear velocity converted from \
             PhysX's centre of mass to the body origin, angular velocity to the body frame",
        ),
        BackendQuirk::new(
            Feature::ContactMesh,
            "each inline mesh is written to an OBJ file in a temp dir for the importer, which \
             cannot read MJCF vertex= / face= and exits 0 on it",
        ),
        BackendQuirk::new(
            Feature::ContactPyramidal,
            "the importer's collider prototypes under /collisions are live static colliders at \
             the world origin and are deactivated; its collision groups are removed; PhysX's \
             extra angular damping (0.05) and sleeping are turned off on every body; envs are \
             isolated by Isaac Sim's cloner collision groups",
        ),
        BackendQuirk::new(
            Feature::JointArmature,
            "state is computed in f32 and widened at the process boundary; the CPU and GPU \
             pipelines differ (up to 0.27 rad measured in I0), so the pipeline is named in the \
             engine version",
        ),
    ]
}

/// `PhysX` behind [`PhysicsBackend`], driven through an Isaac Sim subprocess.
#[derive(Debug)]
pub struct PhysXBackend {
    caps: Capabilities,
    device: String,
    process: Option<Process>,
    model: Option<ModelInfo>,
    tick: PhysTick,
    state: StateReply,
    engine_version: Option<String>,
}

impl Default for PhysXBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysXBackend {
    /// The pipeline `ES_PHYSX_DEVICE` names (`cpu` when unset).
    pub fn new() -> Self {
        let device = std::env::var("ES_PHYSX_DEVICE")
            .ok()
            .filter(|d| !d.trim().is_empty())
            .unwrap_or_else(|| "cpu".to_owned());
        Self::with_device(&device)
    }

    /// The pipeline `device` names: `cpu` or `cuda:N`.
    pub fn with_device(device: &str) -> Self {
        let device = device.to_owned();
        Self {
            caps: capabilities_on(device.starts_with("cuda")),
            device,
            process: None,
            model: None,
            tick: PhysTick::ZERO,
            state: StateReply::default(),
            engine_version: None,
        }
    }

    /// The engine the loaded process runs, pipeline included (packet M11/X1).
    pub fn engine_version(&self) -> Option<&str> {
        self.engine_version.as_deref()
    }

    /// Whether `ES_ISAAC_PYTHON` names an interpreter that imports `isaacsim`.
    pub fn is_available() -> Result<(), String> {
        let python = isaac_python()?;
        match isaac_command(&python)
            .args(["-c", "import isaacsim"])
            .output()
        {
            Ok(out) if out.status.success() => Ok(()),
            Ok(out) => Err(format!(
                "ES_ISAAC_PYTHON=`{python}` cannot import isaacsim: {}",
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .last()
                    .unwrap_or("import failed")
                    .trim()
            )),
            Err(e) => Err(format!("ES_ISAAC_PYTHON=`{python}`: {e}")),
        }
    }

    fn spawn(&self) -> Result<Process, PhysicsError> {
        let python = isaac_python().map_err(PhysicsError::Backend)?;
        let script = script_file().map_err(|e| {
            PhysicsError::Backend(format!("cannot write the PhysX script to a temp file: {e}"))
        })?;
        let mut cmd = isaac_command(&python);
        cmd.arg(&script).env("ES_PHYSX_DEVICE", &self.device);
        match std::env::var_os("ES_PHYSX_STDERR") {
            Some(path) => {
                let file = std::fs::File::create(&path)
                    .map_err(|e| PhysicsError::Backend(format!("ES_PHYSX_STDERR={path:?}: {e}")))?;
                cmd.stderr(file);
            }
            None => {
                cmd.stderr(Stdio::null());
            }
        }
        Process::spawn_command(cmd).map_err(|e| {
            PhysicsError::Backend(format!("cannot start the PhysX process `{python}`: {e}"))
        })
    }

    fn process(&mut self) -> Result<&mut Process, PhysicsError> {
        self.process.as_mut().ok_or(PhysicsError::NotLoaded)
    }

    fn info(&self) -> Result<&ModelInfo, PhysicsError> {
        self.model.as_ref().ok_or(PhysicsError::NotLoaded)
    }

    fn fetch_state(&mut self) -> Result<(), PhysicsError> {
        self.state = self.process()?.call(&Request::State)?;
        Ok(())
    }

    fn check_state(&self, state: &StateView<'_>, rows: usize) -> Result<(), PhysicsError> {
        let info = self.info()?;
        for (what, values, width) in [("qpos", state.qpos, info.nq), ("qvel", state.qvel, info.nv)]
        {
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

fn isaac_python() -> Result<String, String> {
    match std::env::var("ES_ISAAC_PYTHON") {
        Ok(p) if !p.trim().is_empty() => Ok(p),
        _ => Err("ES_ISAAC_PYTHON is not set (the Isaac Sim interpreter, \
                  docs/api-notes/isaac-sim.md)"
            .to_owned()),
    }
}

/// `python` with the environment every Isaac Sim process needs.
fn isaac_command(python: &str) -> Command {
    let mut cmd = Command::new(python);
    cmd.env("OMNI_KIT_ACCEPT_EULA", "YES");
    let compat = std::env::var_os("ES_ISAAC_COMPAT_LIBS")
        .map(PathBuf::from)
        .or_else(|| {
            let dir = PathBuf::from(std::env::var_os("HOME")?)
                .join("opt/isaac-compat/root/usr/lib/x86_64-linux-gnu");
            dir.is_dir().then_some(dir)
        });
    if let Some(dir) = compat {
        let mut paths = vec![dir];
        if let Some(old) = std::env::var_os("LD_LIBRARY_PATH") {
            paths.extend(std::env::split_paths(&old));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            cmd.env("LD_LIBRARY_PATH", joined);
        }
    }
    cmd
}

/// [`SCRIPT`] in the temp dir, named by its content hash, written once and renamed into place
/// so concurrent processes never read a half-written file.
fn script_file() -> std::io::Result<PathBuf> {
    let hash = blake3::hash(SCRIPT.as_bytes()).to_hex();
    let path = std::env::temp_dir().join(format!("es-physx-ref-{}.py", &hash[..16]));
    if !path.is_file() {
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, SCRIPT)?;
        std::fs::rename(&tmp, &path)?;
    }
    Ok(path)
}

impl PhysicsBackend for PhysXBackend {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn load(&mut self, scene: &SceneDesc, cfg: &LoadConfig) -> Result<ModelInfo, PhysicsError> {
        if cfg.n_envs == 0 {
            return Err(PhysicsError::Backend(
                "n_envs must be at least 1".to_owned(),
            ));
        }
        // spec 14.4: the mapping report gates, before anything spawns.
        let report = mapping_report(scene, BackendKind::PhysX);
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
        let mjcf = scene_to_mjcf(scene)?;
        let rate = match cfg.rate {
            Some(rate) => rate,
            None => rate_from_timestep(scene.options.timestep)?,
        };

        // One stage per process: a reload is a new process.
        self.process = None;
        let mut process = self.spawn()?;
        let reply: LoadReply = process.call(&Request::Load {
            mjcf: &mjcf,
            n_envs: cfg.n_envs,
            timestep: Some(rate.period_secs_f64()),
            seed: cfg.seed,
        })?;
        let info = model_info(&reply, scene, "PhysX", cfg.n_envs, rate)?;
        self.engine_version = Some(reply.checked_engine_version()?.to_owned());
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
                crate::proc::check_envs(envs, n_envs)?;
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
    use super::*;

    #[test]
    fn calls_before_load_are_not_loaded_errors() {
        let mut backend = PhysXBackend::new();
        assert!(backend.model_info().is_none());
        assert_eq!(backend.step(1), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.set_ctrl(&[]), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.reset(None, None), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.state().qpos.len(), 0);
    }

    #[test]
    fn the_embedded_script_is_the_file_on_disk() {
        for name in [
            "MJCFCreateAsset",
            "isaaclab.python.headless.kit",
            "set_dof_stiffnesses",
            "externalize_meshes",
            "SetActive(False)",
            "world.reset()",
        ] {
            assert!(SCRIPT.contains(name), "the script does not mention {name}");
        }
        for cmd in ["set_ctrl", "set_state", "quit"] {
            assert!(SCRIPT.contains(cmd), "the script handles no {cmd}");
        }
    }

    #[test]
    fn the_script_file_is_written_once_by_content() {
        let a = script_file().unwrap();
        let b = script_file().unwrap();
        assert_eq!(a, b);
        assert_eq!(std::fs::read_to_string(a).unwrap(), SCRIPT);
    }
}
