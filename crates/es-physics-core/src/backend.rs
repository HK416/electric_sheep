//! The [`PhysicsBackend`] extension point (spec 4.3, spec 17).
//!
//! One of the seven single-implementation traits allowed by INV-17. External backends are the
//! default path in v1.0 (spec 17.1): `MuJoCoCpuBackend` (the CI oracle), `MuJoCoWarpBackend`,
//! `NewtonBackend`, `PhysXBackend`, and one day a native solver, all behind this trait.
//!
//! Two properties shape it:
//!
//! * **Time is integer ticks** (spec 18.1). No method takes or returns a duration in seconds;
//!   the step size is a [`TickRate`] fixed at load time and a step count is a `u32`.
//! * **Divergence is reported, not panicked** (spec 18.5). A backend detects non-finite or
//!   out-of-envelope state itself and returns it in [`StepReport::failures`]; the caller
//!   decides what to do with it through `FailurePolicy`.

use std::collections::BTreeMap;

use es_assets::scene::SceneDesc;
use es_core::{FailureKind, PhysTick, StableId, TickRate};
use serde::{Deserialize, Serialize};

use crate::caps::{Capabilities, Unsupported};

/// Why a backend call failed. A failure *of an env* is not one of these: it is a
/// [`StepReport::failures`] entry (spec 18.5).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PhysicsError {
    /// The scene asks for something this backend does not implement; the item is named, never
    /// approximated silently (spec 17.2).
    #[error("unsupported by this backend: {0}")]
    Unsupported(String),
    /// Requirements checked against the declaration and found wanting (spec 11.6).
    #[error("backend cannot meet the scene's requirements: {}", format_unsupported(.0))]
    Requirements(Vec<Unsupported>),
    /// The backend rejected an otherwise well-formed call.
    #[error("backend error: {0}")]
    Backend(String),
    /// An out-of-process backend stopped answering.
    #[error("backend process died: {0}")]
    ProcessDied(String),
    /// An out-of-process backend answered something this adapter cannot read.
    #[error("backend protocol mismatch: {0}")]
    Protocol(String),
    /// A method was called before [`PhysicsBackend::load`].
    #[error("no model is loaded")]
    NotLoaded,
    /// A caller-supplied buffer is the wrong length.
    #[error("{what}: expected {expected} values, got {got}")]
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

fn format_unsupported(items: &[Unsupported]) -> String {
    items
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

/// How to instantiate a scene (spec 12.1: the simulation batch size is independent of every
/// other batch domain).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadConfig {
    /// Simulation batch size, `>= 1`.
    pub n_envs: u32,
    /// Physics tick rate. `None` keeps the scene's `PhysicsOptions::timestep`; `Some` overrides
    /// it. Integer ticks, never accumulated seconds (spec 18.1).
    pub rate: Option<TickRate>,
    /// Seed for whatever the backend randomises. Not a global RNG (spec 3.4).
    pub seed: u64,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            n_envs: 1,
            rate: None,
            seed: 0,
        }
    }
}

/// A half-open `[start, start + len)` slice of one of the state arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexRange {
    pub start: u32,
    pub len: u32,
}

impl IndexRange {
    pub const fn new(start: u32, len: u32) -> Self {
        Self { start, len }
    }

    /// As a `Range<usize>` for slicing one env's row.
    pub fn as_range(self) -> std::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

/// The shape of a loaded model, and where each scene element lives in the state arrays.
///
/// Maps are keyed by `StableId` (spec 5.3) so an index never has to be looked up by name or
/// by position, and `BTreeMap` so iteration order is deterministic (spec 3.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub nq: u32,
    pub nv: u32,
    pub nu: u32,
    pub nsensordata: u32,
    pub nbody: u32,
    pub n_envs: u32,
    /// The rate ticks are counted at; `dt_phys` of spec 18.1.
    pub rate: TickRate,
    /// Joint id to its slice of `qpos`.
    pub qpos: BTreeMap<StableId, IndexRange>,
    /// Joint id to its slice of `qvel` (the dof range).
    pub dof: BTreeMap<StableId, IndexRange>,
    /// Actuator id to its slice of `ctrl` / `act`.
    pub actuator: BTreeMap<StableId, IndexRange>,
    /// Sensor id to its slice of `sensordata`.
    pub sensor: BTreeMap<StableId, IndexRange>,
    /// Body id to its row in `xpos` / `xquat`.
    pub body: BTreeMap<StableId, IndexRange>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            nq: 0,
            nv: 0,
            nu: 0,
            nsensordata: 0,
            nbody: 0,
            n_envs: 1,
            // spec 18.1: `dt_phys` defaults to 1 ms.
            rate: TickRate::hz(1000),
            qpos: BTreeMap::new(),
            dof: BTreeMap::new(),
            actuator: BTreeMap::new(),
            sensor: BTreeMap::new(),
            body: BTreeMap::new(),
        }
    }
}

/// A borrowed view of the whole batch's state. Every array is env-major: env `e` occupies
/// `[e * stride, (e + 1) * stride)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StateView<'a> {
    pub n_envs: u32,
    /// Tick the state was sampled at (spec 18.1: every sample carries a tick).
    pub tick: PhysTick,
    /// `n_envs * nq`.
    pub qpos: &'a [f64],
    /// `n_envs * nv`.
    pub qvel: &'a [f64],
    /// `n_envs * nu` actuator activation state.
    pub act: &'a [f64],
    /// `n_envs * nsensordata`.
    pub sensordata: &'a [f64],
    /// `n_envs * nbody * 3` body positions in the world frame.
    pub xpos: &'a [f64],
    /// `n_envs * nbody * 4` body orientations, xyzw (spec 3.1).
    pub xquat: &'a [f64],
}

impl StateView<'_> {
    /// `values`' row for env `e`, given a per-env stride.
    fn row(values: &[f64], env: u32, stride: usize) -> &[f64] {
        let start = env as usize * stride;
        &values[start..start + stride]
    }

    /// `qpos` for one env.
    pub fn qpos_of(&self, env: u32) -> &[f64] {
        Self::row(
            self.qpos,
            env,
            self.qpos.len() / self.n_envs.max(1) as usize,
        )
    }

    /// `qvel` for one env.
    pub fn qvel_of(&self, env: u32) -> &[f64] {
        Self::row(
            self.qvel,
            env,
            self.qvel.len() / self.n_envs.max(1) as usize,
        )
    }

    /// Whether every value in `qpos`, `qvel` and `sensordata` is finite.
    pub fn is_finite(&self) -> bool {
        self.qpos
            .iter()
            .chain(self.qvel)
            .chain(self.sensordata)
            .all(|v| v.is_finite())
    }
}

/// What one [`PhysicsBackend::step`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepReport {
    /// Tick after the step.
    pub tick: PhysTick,
    /// Per-env failures detected by the backend (spec 18.5). Sorted by env.
    pub failures: Vec<(u32, FailureKind)>,
}

/// A physics engine behind the one backend-neutral interface (spec 4.3, INV-17).
///
/// Object safe on purpose: a run holds a `Box<dyn PhysicsBackend>` chosen at runtime by
/// `es backend compare` (spec 17.2).
pub trait PhysicsBackend {
    /// What this backend declares it can do (spec 4.3). Checked against a scene's
    /// [`Requirements`](crate::caps::Requirements) by the compiler before a run (spec 11.6).
    fn capabilities(&self) -> &Capabilities;

    /// Instantiates `scene` for `cfg.n_envs` environments.
    ///
    /// Anything in the scene the backend cannot map is [`PhysicsError::Unsupported`] naming the
    /// item; it is never approximated silently (spec 17.2).
    fn load(&mut self, scene: &SceneDesc, cfg: &LoadConfig) -> Result<ModelInfo, PhysicsError>;

    /// The loaded model, or `None` before the first successful [`load`](Self::load).
    fn model_info(&self) -> Option<&ModelInfo>;

    /// Resets `envs` (all of them when `None`) to `state` (the scene's initial state when
    /// `None`).
    ///
    /// When both are given, row `i` of `state` is env `envs[i]`. Subset reset requires
    /// `capabilities().supports_reset_subset`.
    fn reset(
        &mut self,
        envs: Option<&[u32]>,
        state: Option<&StateView<'_>>,
    ) -> Result<(), PhysicsError>;

    /// Sets the whole batch's control vector: `n_envs * nu`, row-major by env.
    fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), PhysicsError>;

    /// Advances every env by `n_substeps` physics ticks.
    fn step(&mut self, n_substeps: u32) -> Result<StepReport, PhysicsError>;

    /// The current state of the whole batch, as of the last step or reset.
    fn state(&self) -> StateView<'_>;

    /// Overwrites the whole batch's state.
    fn set_state(&mut self, state: &StateView<'_>) -> Result<(), PhysicsError>;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::caps::{BatchSupport, DeterminismTier, FloatPrecision};

    /// A backend that does nothing, to pin the trait's object safety and its contracts.
    struct NullBackend {
        caps: Capabilities,
        model: Option<ModelInfo>,
        qpos: Vec<f64>,
        tick: PhysTick,
    }

    impl NullBackend {
        fn new() -> Self {
            Self {
                caps: Capabilities {
                    name: "null".to_owned(),
                    determinism: DeterminismTier::SemanticEqual,
                    batch: BatchSupport {
                        max_envs: 1,
                        gpu_resident: false,
                    },
                    joints: BTreeSet::new(),
                    actuators: BTreeSet::new(),
                    sensors: BTreeSet::new(),
                    contact: BTreeSet::new(),
                    float: FloatPrecision::F64,
                    supports_reset_subset: false,
                    supports_state_get_set: true,
                    quirks: Vec::new(),
                },
                model: None,
                qpos: Vec::new(),
                tick: PhysTick::ZERO,
            }
        }
    }

    impl PhysicsBackend for NullBackend {
        fn capabilities(&self) -> &Capabilities {
            &self.caps
        }

        fn load(&mut self, scene: &SceneDesc, cfg: &LoadConfig) -> Result<ModelInfo, PhysicsError> {
            let nq = scene.joints.len() as u32;
            let info = ModelInfo {
                nq,
                n_envs: cfg.n_envs,
                rate: cfg.rate.unwrap_or_else(|| TickRate::hz(1000)),
                ..ModelInfo::default()
            };
            self.qpos = vec![0.0; (nq * cfg.n_envs) as usize];
            self.model = Some(info.clone());
            Ok(info)
        }

        fn model_info(&self) -> Option<&ModelInfo> {
            self.model.as_ref()
        }

        fn reset(
            &mut self,
            envs: Option<&[u32]>,
            _state: Option<&StateView<'_>>,
        ) -> Result<(), PhysicsError> {
            if envs.is_some() && !self.caps.supports_reset_subset {
                return Err(PhysicsError::Unsupported(
                    "reset of an env subset".to_owned(),
                ));
            }
            self.qpos.fill(0.0);
            self.tick = PhysTick::ZERO;
            Ok(())
        }

        fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), PhysicsError> {
            if !ctrl.is_empty() {
                return Err(PhysicsError::ShapeMismatch {
                    what: "ctrl",
                    expected: 0,
                    got: ctrl.len(),
                });
            }
            Ok(())
        }

        fn step(&mut self, n_substeps: u32) -> Result<StepReport, PhysicsError> {
            if self.model.is_none() {
                return Err(PhysicsError::NotLoaded);
            }
            self.tick = self.tick.add_ticks(u64::from(n_substeps));
            Ok(StepReport {
                tick: self.tick,
                failures: Vec::new(),
            })
        }

        fn state(&self) -> StateView<'_> {
            StateView {
                n_envs: self.model.as_ref().map_or(0, |m| m.n_envs),
                tick: self.tick,
                qpos: &self.qpos,
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

    #[test]
    fn the_trait_is_object_safe() {
        let mut backend: Box<dyn PhysicsBackend> = Box::new(NullBackend::new());
        assert!(backend.model_info().is_none());
        assert_eq!(backend.step(1), Err(PhysicsError::NotLoaded));

        let import = es_assets::parse_mjcf(
            r#"<mujoco><worldbody><body name="b">
                 <joint name="j" type="hinge"/><geom name="g" type="sphere" size="0.1"/>
               </body></worldbody></mujoco>"#,
        )
        .unwrap();
        let cfg = LoadConfig {
            n_envs: 2,
            ..LoadConfig::default()
        };
        let info = backend.load(&import.scene, &cfg).unwrap();
        assert_eq!((info.nq, info.n_envs), (1, 2));
        assert_eq!(backend.model_info(), Some(&info));

        let report = backend.step(10).unwrap();
        assert_eq!(report.tick, PhysTick(10));
        assert!(report.failures.is_empty());

        // Time is ticks: stepping twice adds, it does not accumulate seconds.
        assert_eq!(backend.step(10).unwrap().tick, PhysTick(20));

        let state = backend.state();
        assert_eq!(state.qpos.len(), 2);
        assert_eq!(state.qpos_of(1), &[0.0]);
        assert!(state.is_finite());

        let owned: Vec<f64> = vec![1.0, 2.0];
        let set = StateView {
            n_envs: 2,
            qpos: &owned,
            ..StateView::default()
        };
        backend.set_state(&set).unwrap();
        assert_eq!(backend.state().qpos, &[1.0, 2.0]);
        backend.reset(None, None).unwrap();
        assert_eq!(backend.state().qpos, &[0.0, 0.0]);
    }

    #[test]
    fn unsupported_and_shape_errors_name_the_thing() {
        let mut backend = NullBackend::new();
        let err = backend.reset(Some(&[0]), None).unwrap_err();
        assert!(err.to_string().contains("subset"));
        let err = backend.set_ctrl(&[0.0]).unwrap_err();
        assert!(err.to_string().contains("expected 0 values, got 1"));
        let err = PhysicsError::Requirements(vec![Unsupported::GpuResidency]);
        assert!(err.to_string().contains("GPU-resident"));
    }

    #[test]
    fn index_range_slices_a_row() {
        let range = IndexRange::new(2, 3);
        assert_eq!(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0][range.as_range()],
            &[2.0, 3.0, 4.0]
        );
    }
}
