//! `MuJoCoCpuBackend` — `MuJoCo` (CPU) as the reference backend and CI oracle (spec 17.1).
//!
//! `MuJoCo` has no Rust binding, and the core runtime must not link Python (spec 2.4). So the
//! engine runs out of process: `python/mujoco_ref.py` holds the model and one `MjData` per
//! env, and this adapter speaks line-delimited JSON to it. That is slow and deliberately so —
//! this is the oracle every other backend is judged against (spec 1.4), not a throughput path.
//!
//! State is cached on this side after every step and reset, so [`PhysicsBackend::state`] is a
//! borrow of local buffers rather than a round trip.

use std::collections::BTreeMap;

use es_assets::scene::SceneDesc;
use es_core::{FailureKind, PhysTick, StableId, TickRate};
use es_physics_core::caps::{BackendQuirk, BatchSupport, DeterminismTier, FloatPrecision};
use es_physics_core::{
    check_requirements, Capabilities, Feature, IndexRange, LoadConfig, ModelInfo, PhysicsBackend,
    PhysicsError, Requirements, StateView, StepReport,
};

use crate::mjcf_out::scene_to_mjcf;
use crate::proc::{Ack, LoadReply, Process, Request, StatePayload, StateReply, StepReply};

/// The backend's name in `es backend compare --backends ...` (spec 17.2).
pub const NAME: &str = "mujoco-cpu";

/// What this backend declares (spec 4.3).
///
/// **Determinism is tier 3, not tier 1.** Spec 17.3 reserves tier 1 (bitwise) for
/// `MuJoCoCpuBackend`, but tier 1 is defined (spec 3.5) as an `execution_hash` guarantee on a
/// pinned device and driver, and nothing in this adapter's chain — a Python interpreter, a
/// wheel-built `MuJoCo`, JSON on a pipe — is pinned by that hash. What it does guarantee is
/// physics-meaning agreement with `MuJoCo`, which is tier 3 and is trivially true because it
/// *is* `MuJoCo`. Spec 4.3's rule ("an external backend never declares tier 1") is the one
/// followed here; a native binding that pins the `MuJoCo` build can revisit it.
pub fn capabilities() -> Capabilities {
    Capabilities {
        name: NAME.to_owned(),
        determinism: DeterminismTier::PhysicsMeaning,
        batch: BatchSupport {
            // No native batching. `n_envs > 1` works — the script holds one `MjData` per env
            // and loops — but it is emulation, not a declared batch capability, so a task that
            // *requires* a batch is told to pick another backend (spec 11.6).
            max_envs: 1,
            gpu_resident: false,
        },
        joints: [
            Feature::JointFree,
            Feature::JointBall,
            Feature::JointHinge,
            Feature::JointSlide,
            Feature::JointFixed,
            Feature::JointLimit,
            Feature::JointArmature,
            Feature::JointSpring,
            Feature::JointFrictionLoss,
        ]
        .into(),
        actuators: [
            Feature::ActuatorMotor,
            Feature::ActuatorPosition,
            Feature::ActuatorVelocity,
            Feature::ActuatorOnJoint,
        ]
        .into(),
        sensors: [Feature::SensorJointPos, Feature::SensorJointVel].into(),
        contact: [
            Feature::ContactPyramidal,
            Feature::ContactElliptic,
            Feature::ContactSoftParams,
            Feature::ContactCondim6,
        ]
        .into(),
        float: FloatPrecision::F64,
        supports_reset_subset: true,
        supports_state_get_set: true,
        quirks: vec![
            BackendQuirk::new(
                Feature::JointFixed,
                "a fixed joint is emitted as a body with no joint, MJCF's own spelling for a \
                 weld",
            ),
            BackendQuirk::new(
                Feature::SensorJointPos,
                "sensor `noise` and `cutoff` are written into the MJCF but MuJoCo applies them \
                 only with the sensornoise enable flag, which this adapter does not set: \
                 sensor readings are noiseless",
            ),
        ],
    }
}

/// `MuJoCo` (CPU) behind [`PhysicsBackend`], driven through a Python subprocess.
#[derive(Debug)]
pub struct MuJoCoCpuBackend {
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

impl Default for MuJoCoCpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MuJoCoCpuBackend {
    pub fn new() -> Self {
        Self {
            caps: capabilities(),
            process: None,
            model: None,
            tick: PhysTick::ZERO,
            state: StateBuffers::default(),
        }
    }

    /// Whether a Python interpreter with the `mujoco` package is available. `Err` explains what
    /// was tried, so a test can skip with a reason instead of failing CI (spec 1.4).
    pub fn is_available() -> Result<(), String> {
        crate::proc::is_available()
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

/// The tick rate a timestep in seconds stands for. Integer ticks are the model (spec 18.1);
/// this is the one conversion, at the edge, where a backend's own `dt` is set.
fn rate_from_timestep(timestep: f64) -> Result<TickRate, PhysicsError> {
    let nanos = (timestep * 1e9).round();
    if !(nanos.is_finite() && nanos >= 1.0) {
        return Err(PhysicsError::Backend(format!(
            "timestep {timestep} is not a positive number of nanoseconds"
        )));
    }
    // Reduced, so that two ways of spelling the same rate compare equal.
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

/// `name -> id` for one kind of scene element.
fn index_by_name<'a, T: 'a>(
    items: impl IntoIterator<Item = &'a T>,
    field: impl Fn(&'a T) -> (&'a str, StableId),
) -> BTreeMap<&'a str, StableId> {
    items.into_iter().map(field).collect()
}

fn id_of(map: &BTreeMap<&str, StableId>, kind: &str, name: &str) -> Result<StableId, PhysicsError> {
    map.get(name).copied().ok_or_else(|| {
        PhysicsError::Protocol(format!(
            "MuJoCo reported a {kind} `{name}` the scene does not have"
        ))
    })
}

impl PhysicsBackend for MuJoCoCpuBackend {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn load(&mut self, scene: &SceneDesc, cfg: &LoadConfig) -> Result<ModelInfo, PhysicsError> {
        if cfg.n_envs == 0 {
            return Err(PhysicsError::Backend(
                "n_envs must be at least 1".to_owned(),
            ));
        }
        // The scene's features are checked against the declaration before anything is spawned
        // (spec 11.6). `n_envs` stays at 1: envs are emulated by looping, see `capabilities`.
        let requirements = Requirements {
            n_envs: 1,
            ..Requirements::from_scene(scene)
        };
        let unmet = check_requirements(&requirements, &self.caps);
        if !unmet.is_empty() {
            return Err(PhysicsError::Requirements(unmet));
        }
        let mjcf = scene_to_mjcf(scene)?;
        let rate = match cfg.rate {
            Some(rate) => rate,
            None => rate_from_timestep(scene.options.timestep)?,
        };

        let mut process = Process::spawn()?;
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
// Exact values are the property under test: a rate that round-trips, a state that is copied
// back unchanged, a run that repeats bit for bit.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::tests_support::fixture;

    /// Skips the body of a test, with a reason, when `MuJoCo` is not installed. A missing
    /// optional Python package must not fail CI (spec 2.4: the core path is Python-free).
    macro_rules! require_mujoco {
        ($name:literal) => {
            if let Err(reason) = MuJoCoCpuBackend::is_available() {
                eprintln!("SKIP {}: {reason}", $name);
                return;
            }
        };
    }

    fn pendulum() -> SceneDesc {
        es_assets::parse_mjcf(&fixture("pendulum.xml"))
            .unwrap()
            .scene
    }

    fn load_pendulum() -> Result<(MuJoCoCpuBackend, ModelInfo), PhysicsError> {
        let mut backend = MuJoCoCpuBackend::new();
        let cfg = LoadConfig {
            n_envs: 1,
            rate: Some(TickRate::hz(1000)),
            seed: 1,
        };
        let info = backend.load(&pendulum(), &cfg)?;
        Ok((backend, info))
    }

    #[test]
    fn capabilities_are_declared_honestly() {
        let caps = capabilities();
        assert_eq!(caps.name, NAME);
        // spec 4.3: an external backend never claims tier 1.
        assert_ne!(caps.determinism, DeterminismTier::Bitwise);
        assert_eq!(caps.determinism, DeterminismTier::PhysicsMeaning);
        assert!(!caps.batch.gpu_resident);
        assert!(caps.has(Feature::JointHinge) && caps.has(Feature::ContactElliptic));
        // What the MJCF emitter cannot write is not declared.
        assert!(!caps.has(Feature::ActuatorGeneral));
        assert!(!caps.has(Feature::ActuatorOnTendon));
        assert!(!caps.has(Feature::SensorTouch));
        assert!(!caps.has(Feature::ContactMesh));
        assert!(!caps.quirks.is_empty());
    }

    /// The capability check runs before the process is spawned (spec 11.6), so these fail the
    /// same way with or without `MuJoCo` installed.
    #[test]
    fn a_scene_the_backend_cannot_map_is_refused_before_anything_is_spawned() {
        for mjcf in [
            // A site-mounted sensor: not in the declared sensor set.
            r#"<mujoco><worldbody><body name="b"><site name="s"/>
                 <joint name="j" type="hinge"/><geom name="g" type="sphere" size="0.1"/>
               </body></worldbody>
               <sensor><accelerometer name="a" site="s"/></sensor></mujoco>"#,
            // A mesh geom: not in the declared contact set.
            r#"<mujoco><asset><mesh name="m" file="m.obj"/></asset>
               <worldbody><body name="b"><joint name="j" type="hinge"/>
                 <geom name="g" type="mesh" mesh="m"/></body></worldbody></mujoco>"#,
        ] {
            let scene = es_assets::parse_mjcf(mjcf).unwrap().scene;
            let err = MuJoCoCpuBackend::new()
                .load(&scene, &LoadConfig::default())
                .unwrap_err();
            let PhysicsError::Requirements(unmet) = &err else {
                panic!("expected a requirements failure, got {err:?}");
            };
            assert_eq!(unmet.len(), 1, "{unmet:?}");
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn calls_before_load_are_not_loaded_errors() {
        let mut backend = MuJoCoCpuBackend::new();
        assert!(backend.model_info().is_none());
        assert_eq!(backend.step(1), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.set_ctrl(&[]), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.reset(None, None), Err(PhysicsError::NotLoaded));
        assert_eq!(backend.state().qpos.len(), 0);
    }

    #[test]
    fn rate_and_timestep_agree() {
        assert_eq!(rate_from_timestep(0.001).unwrap(), TickRate::hz(1000));
        assert_eq!(rate_from_timestep(0.002).unwrap().as_hz_f64(), 500.0);
        assert!(rate_from_timestep(0.0).is_err());
        assert!(rate_from_timestep(f64::NAN).is_err());
    }

    #[test]
    fn mujoco_pendulum_falls() {
        require_mujoco!("mujoco_pendulum_falls");
        let (mut backend, info) = load_pendulum().unwrap();
        assert_eq!((info.nq, info.nv, info.n_envs), (1, 1, 1));
        assert_eq!(info.rate, TickRate::hz(1000));
        assert_eq!(info.qpos.len(), 1);
        assert_eq!(info.nbody, 2);

        // Hanging straight down is an equilibrium, so the pendulum is lifted first.
        let start = [0.3_f64];
        backend
            .set_state(&StateView {
                n_envs: 1,
                qpos: &start,
                ..StateView::default()
            })
            .unwrap();
        assert_eq!(backend.state().qpos, &start);

        let report = backend.step(100).unwrap();
        assert_eq!(report.tick, PhysTick(100));
        assert!(report.failures.is_empty(), "{:?}", report.failures);

        let state = backend.state();
        assert!(state.is_finite());
        assert!(
            (state.qpos[0] - start[0]).abs() > 1e-6,
            "the pendulum did not move: {:?}",
            state.qpos
        );
        // It swings towards the hanging position, so it moves the way gravity points.
        assert!(state.qpos[0] < start[0]);
        assert!(state.qvel[0].abs() > 1e-6);
        // Body 1 (the rod) is placed by forward kinematics, not left at the origin.
        assert_eq!(state.xpos.len(), 6);
        assert!((state.xpos[5] - 1.0).abs() < 1e-9);

        // A whole-batch reset rewinds the clock and the state.
        backend.reset(None, None).unwrap();
        assert_eq!(backend.state().tick, PhysTick::ZERO);
        assert!(backend.state().qpos[0].abs() < 1e-12);
    }

    #[test]
    fn mujoco_runs_are_bit_identical_from_the_same_state() {
        require_mujoco!("mujoco_runs_are_bit_identical_from_the_same_state");
        let run = || {
            let (mut backend, _) = load_pendulum().unwrap();
            let start = [0.3_f64];
            backend
                .set_state(&StateView {
                    n_envs: 1,
                    qpos: &start,
                    ..StateView::default()
                })
                .unwrap();
            backend.step(100).unwrap();
            let state = backend.state();
            (state.qpos.to_vec(), state.qvel.to_vec())
        };
        let (qpos_a, qvel_a) = run();
        let (qpos_b, qvel_b) = run();
        for (a, b) in qpos_a.iter().zip(&qpos_b).chain(qvel_a.iter().zip(&qvel_b)) {
            assert_eq!(a.to_bits(), b.to_bits(), "{a} != {b}");
        }
    }

    /// `n_envs > 1` is emulated by looping over independent `MjData` (see `capabilities`), and
    /// the two-link arm is the other shape the emitter claims to cover.
    #[test]
    fn mujoco_runs_a_batch_and_a_two_link_arm() {
        require_mujoco!("mujoco_runs_a_batch_and_a_two_link_arm");
        let mut backend = MuJoCoCpuBackend::new();
        let cfg = LoadConfig {
            n_envs: 2,
            rate: Some(TickRate::hz(1000)),
            seed: 0,
        };
        let info = backend.load(&pendulum(), &cfg).unwrap();
        assert_eq!(info.n_envs, 2);
        backend
            .set_state(&StateView {
                n_envs: 2,
                qpos: &[0.2, -0.4],
                ..StateView::default()
            })
            .unwrap();
        backend.step(50).unwrap();
        let state = backend.state();
        assert_eq!(state.qpos.len(), 2);
        assert_ne!(state.qpos_of(0), state.qpos_of(1));
        // Resetting one env leaves the other alone, and the batch clock keeps running.
        backend.reset(Some(&[0]), None).unwrap();
        assert!(backend.state().qpos[0].abs() < 1e-12);
        assert!(backend.state().qpos[1].abs() > 1e-6);
        assert_eq!(backend.state().tick, PhysTick(50));

        let arm = es_assets::parse_mjcf(&fixture("arm2.xml")).unwrap().scene;
        let info = MuJoCoCpuBackend::new()
            .load(&arm, &LoadConfig::default())
            .unwrap();
        assert_eq!((info.nq, info.nv, info.nbody), (2, 2, 3));
        assert_eq!(info.qpos.len(), 2);
    }

    #[test]
    fn mujoco_reports_shape_mismatches_and_control() {
        require_mujoco!("mujoco_reports_shape_mismatches_and_control");
        let scene = es_assets::parse_mjcf(&fixture("actuated.xml"));
        // `actuated.xml` uses tendons and site sensors, so it must be refused, by name.
        if let Ok(import) = scene {
            let err = MuJoCoCpuBackend::new()
                .load(&import.scene, &LoadConfig::default())
                .unwrap_err();
            assert!(matches!(
                err,
                PhysicsError::Requirements(_) | PhysicsError::Unsupported(_)
            ));
        }

        let (mut backend, _) = load_pendulum().unwrap();
        // The pendulum has no actuator, so the only valid control vector is the empty one.
        assert_eq!(backend.set_ctrl(&[]), Ok(()));
        assert_eq!(
            backend.set_ctrl(&[1.0]),
            Err(PhysicsError::ShapeMismatch {
                what: "ctrl",
                expected: 0,
                got: 1
            })
        );
        assert!(matches!(
            backend.set_state(&StateView {
                n_envs: 1,
                qpos: &[0.0, 0.0],
                ..StateView::default()
            }),
            Err(PhysicsError::ShapeMismatch { what: "qpos", .. })
        ));
        assert!(backend.reset(Some(&[7]), None).is_err());
    }
}
