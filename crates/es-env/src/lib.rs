//! `es-env` (layer 9): the environment runtime — batch-domain scheduling, reset and domain
//! randomization, reward/termination evaluation, and episode recording.
//!
//! See `docs/design/batch-domains.md` and `docs/ARCHITECTURE.ko.md` §12 (four batch domains),
//! §6.4 (execution semantics), §18.1 (integer time), §18.5 (failure semantics), and
//! `docs/design/control-graph.md` for IR-C execution ([`control`]).
//!
//! Layer rule (§4.2): may depend on layers 0..=8 only. In particular **not** on `es-telemetry`
//! (layer 10) — [`EnvMetrics`] is a plain struct that `es-telemetry` converts.

pub mod chunk_buffer;
pub mod control;
pub mod domains;
pub mod env;
pub mod episode;
/// The scripted pick-and-place expert (`docs/design/visible-learning.md` section 5).
pub mod expert;
pub mod inference;
pub mod plan;
pub mod randomize;
/// The renderer in the env loop. Feature `render` (off by default): the only part of this
/// crate that links Vulkan (§15, `docs/design/visible-learning.md` section 7).
#[cfg(feature = "render")]
pub mod render;
pub mod rng;
pub mod scheduler;

pub use chunk_buffer::{plane_chunk, ChunkBuffer, PlaneFeed, CHUNK_SLOTS};
pub use control::{ControlExecutor, StageOutcome, StageState};
pub use domains::{DomainRunner, DomainSizing};
pub use env::{Env, EnvMetrics, StepOutcome};
pub use episode::{Episode, EpisodeRecorder, Termination};
pub use expert::{so101_ik, ExpertCfg, Links, ScriptedExpert, Stage};
pub use inference::{default_max_pending, latency_ticks, AsyncInference, Submission};
pub use randomize::RandomizationPlan;
#[cfg(feature = "render")]
pub use render::{EnvRenderer, EnvRendererCfg};
pub use rng::EnvRng;
pub use scheduler::{BatchDomains, Device, DomainCfg, Schedule, TickPlan};

/// Everything that can go wrong in the env runtime.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EnvError {
    /// A batch-domain configuration that [`Schedule::build`] refuses (§12.1, App. B.5).
    #[error("invalid batch domain schedule: {0}")]
    Schedule(String),
    /// The task asks for something this runtime does not implement. The item is always named;
    /// nothing is silently skipped (§17.2 applied to the env runtime).
    #[error("unsupported by the env runtime: {0}")]
    Unsupported(String),
    /// The Task IR is well formed but cannot drive an env (a dangling sink, say).
    #[error("task: {0}")]
    Task(String),
    /// The Observation IR's declared `ImageSpec` and what the renderer produces disagree
    /// (§7.2, §26.1, `INV-14`). Never repaired: resampling or converting to fit would make
    /// `observation_hash` describe a pipeline nobody declared, and `Resize` / `Crop` /
    /// `ColorTransform` are Observation IR nodes that carry the intrinsics with them.
    #[error("image spec: the observation declares {field} = {declared}, the renderer produces {produced}")]
    ImageSpec {
        field: &'static str,
        declared: String,
        produced: String,
    },
    /// The backend refused a call.
    #[error(transparent)]
    Physics(#[from] es_physics_core::backend::PhysicsError),
    /// A caller-supplied buffer is the wrong length.
    #[error("{what}: expected {expected} values, got {got}")]
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

impl EnvError {
    pub(crate) fn shape(what: &'static str, expected: usize, got: usize) -> Self {
        Self::ShapeMismatch {
            what,
            expected,
            got,
        }
    }
}
