//! `es-runtime-embedded` (layer 9): the minimal runtime that runs a `policy.esb` on real
//! hardware (spec 9.6).
//!
//! Spec 9.6 pins the contents exactly:
//!
//! ```text
//! contains: Observation IR evaluator + Learning IR pre/post-processing
//!           + PolicyRuntime (ONNX or Vulkan or NPU)
//!           + the whole Safety Plane
//!           + a telemetry ring buffer
//!
//! excludes: physics engine, renderer, editor, Python, training
//! ```
//!
//! and spec 9.5 gives the reason it is worth having: the Safety Plane, the chunker and the
//! preprocessing must be *the same code* in simulation and on the robot, so that
//! `deployment_hash` being equal means the safe behaviour is equal. That is why this crate
//! composes `es-compile`, `es-safety` and `es-policy` rather than reimplementing any of them,
//! and why there is no constructor that omits the plane (`INV-12`).
//!
//! Layer rule (spec 4.2): layer 9, alongside `es-env`. It may use `es-safety` / `es-policy`
//! (8) and `es-compile` (7); only `es-ros2` and `es-py` (11) may depend on it. Notably it may
//! **not** use `es-telemetry` (10) — its telemetry ring is `es_core::ring::RingBuffer`
//! (layer 1) instead, the same type `es-telemetry` re-exports, so no layering violation is
//! needed to share it.
//!
//! No Python, no physics, no rendering, no filesystem: [`EmbeddedRuntime::from_bundle`] takes
//! bytes.
#![forbid(unsafe_code)]

mod hardware;
mod runtime;

pub use hardware::hardware_capability;
pub use runtime::{EmbeddedRuntime, RuntimeError, TickRecord};
