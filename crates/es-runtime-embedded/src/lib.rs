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
//!
//! # Two layers (spec 9.6: no-std capable, zero heap)
//!
//! - [`core_rt`] — [`EmbeddedCore`], the control loop itself: Safety Plane + chunk cursor +
//!   telemetry ring, built with `--no-default-features` on a bare-metal target. The
//!   Observation IR evaluation and the inference step are function pointers
//!   ([`ObserveFn`], [`InferFn`]), not traits, so nothing allocates and INV-17 is untouched.
//! - the default `std` layer — [`EmbeddedRuntime`], which plugs `es_compile::CpuPlan` and a
//!   `dyn es_policy::PolicyRuntime` (ONNX / Vulkan / NPU) into that same loop and adds the
//!   bundle reader and the hash chain.
//!
//! See `docs/design/embedded-runtime.md`.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

pub mod core_rt;
#[cfg(feature = "std")]
mod hardware;
#[cfg(feature = "std")]
mod runtime;

pub use core_rt::{EmbeddedCore, InferFn, ObserveFn, TickRecord, TELEMETRY_TICKS};
#[cfg(feature = "std")]
pub use hardware::hardware_capability;
#[cfg(feature = "std")]
pub use runtime::{EmbeddedRuntime, RuntimeError};
