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
//! - **INV-12** — no code path disables the plane. [`SafetyPlane::from_ir`] is the only
//!   constructor, every field is private, there is no `enabled` flag and no `#[cfg(test)]`
//!   shortcut. A test that needs room widens its envelope.
//! - **INV-13** — [`SafetyPlane::validate`] returns no `Result`. A safe action always exists,
//!   so propagating an error would only invite the caller to ignore it.
//!
//! Determinism (spec 3.4): `validate` is a pure function of (configuration, state, inputs).
//! Time is integer ticks and whole microseconds; the only float derived from time is the
//! constant control period, computed once from the rational tick rate and never accumulated.
//! No `HashMap`, no RNG, no global state, no `unsafe`.
//!
//! Allocation: the hot path allocates nothing. The retract trajectory and the sensor-dropout
//! table are the only heap objects and both are sized in `from_ir`. See
//! `docs/design/safety-plane.md` for why the crate is `std` rather than `no_std` in M1.

#![forbid(unsafe_code)]

mod config;
mod counters;
mod plane;
mod types;

pub use config::{Envelope, Fallback, SafetyConfigError, SensorWatch, Watchdogs};
pub use counters::{SafetyCounters, WINDOW_CAP};
pub use plane::{SafetyPlane, SafetyState};
pub use types::{ActionChunk, ActionSource, EventSet, FallbackKind, SafeAction, ViolationKind};
