//! es-safety (layer 8): the Safety Plane runtime (spec 9, Appendix B.4).
//!
//! > **The policy is not trusted.** Safety is enforced outside the policy, independently of
//! > it, deterministically (spec 9.1).
//!
//! `es-ir::deployment` is the *configuration*; this crate is the *runtime* that executes it
//! once per control tick. The same code runs in simulation and on hardware (spec 9.5), so
//! there is no simulation mode and no way to switch the plane off.
//!
//! ```text
//! policy -> ActionChunk -> SafetyPlane::validate -> SafeAction -> controller -> actuator
//! ```
//!
//! Invariants this crate exists to hold:
//!
//! - **INV-11** — no dependency on `es-policy`, enforced by `cargo xtask layering` (spec 4.2
//!   rule 8). Nothing here names a policy, a network or a tensor.
//! - **INV-12** — no code path disables the plane. [`SafetyPlane::from_config`] is the only
//!   constructor (`from_ir` converts the IR and delegates to it), every field is private,
//!   there is no `enabled` flag and no `#[cfg(test)]` shortcut. A test that needs room widens
//!   its envelope.
//! - **INV-13** — [`SafetyPlane::validate`] returns no `Result`. A safe action always exists,
//!   so propagating an error would only invite the caller to ignore it.
//!
//! Determinism (spec 3.4): `validate` is a pure function of (configuration, state, inputs).
//! Time is integer ticks and whole microseconds; the only float derived from time is the
//! constant control period, computed once from the rational tick rate and never accumulated.
//! No `HashMap`, no RNG, no global state, no `unsafe`.
//!
//! Allocation: nothing in this crate allocates — not the hot path, not construction. The
//! retract trajectory, the convex-hull faces and the sensor-dropout table are fixed-size
//! arrays inside [`SafetyConfig`].
//!
//! # `no_std` (spec 9.6, Appendix B.4)
//!
//! `--no-default-features` drops `std`, `es-ir`, `thiserror` and `serde/std`, and what is left
//! is the whole runtime: [`SafetyPlane`], [`SafetyConfig`], [`ActionChunk`], [`SafeAction`],
//! [`SafetyCounters`] and the envelope / watchdog / fallback config structs. What goes is only
//! the Deployment IR front door — [`SafetyPlane::from_ir`] and [`SafetyConfigError`] — because
//! `es-ir` is a `std` crate. `validate` is identical either way (INV-13), which is the point:
//! spec 9.5 wants the *same code* in simulation and on the robot.
//!
//! See `docs/design/embedded-runtime.md` for the split and `docs/design/safety-plane.md` for
//! the algorithm.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod config;
mod counters;
mod ir_types;
mod plane;
mod types;

#[cfg(feature = "std")]
pub use config::SafetyConfigError;
pub use config::{
    Envelope, Fallback, SafetyConfig, SensorWatch, Watchdogs, WorkspaceSpec, MAX_HULL_FACES,
    MAX_RETRACT_WAYPOINTS, MAX_SENSORS, SENSOR_NAME_CAP,
};
pub use counters::{SafetyCounters, WINDOW_CAP};
pub use ir_types::{ActionSpace, ExecutionMode, HalfSpace, Limit, Micros};
pub use plane::{SafetyPlane, SafetyState};
pub use types::{ActionChunk, ActionSource, EventSet, FallbackKind, SafeAction, ViolationKind};
