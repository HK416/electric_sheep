//! `es-physics-core` (layer 3): the `PhysicsBackend` extension point and the capability
//! vocabulary every backend declares itself in. See `docs/ARCHITECTURE.ko.md` spec 4.3
//! (backends are the default path), spec 11.6 (capability check at compile time), spec 17
//! (backend layer, semantic mapping, determinism) and `docs/packets/M0/P36.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-core`, `es-math`, `es-assets`, `es-usd`
//! and external crates only. It contains no engine: every concrete backend lives in
//! `es-physics-backend` (layer 4).
#![forbid(unsafe_code)]

pub mod backend;
pub mod caps;
pub mod usd;

pub use backend::{
    IndexRange, LoadConfig, ModelInfo, PhysicsBackend, PhysicsError, StateView, StepReport,
};
pub use caps::{
    check_requirements, BackendQuirk, BatchSupport, Capabilities, DeterminismTier, Feature,
    FloatPrecision, Requirements, Unsupported,
};
