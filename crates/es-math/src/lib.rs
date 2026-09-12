//! `es-math` — layer 0 (spec §4.2): scalar representations, deterministic transcendentals,
//! convention types, reproducible reduction, SIMD primitives.
//!
//! Nothing here depends on another `es-*` crate. Everything here is a determinism contract:
//! physics, observation and reward kernels must use `approx` instead of `std`
//! transcendentals (spec §3.2, `DET-010`) and `reduce` instead of a plain sum
//! (spec §18.4, `DET-011`).
#![forbid(unsafe_code)]

pub mod approx;
pub mod conventions;
pub mod reduce;
pub mod scalar;
pub mod simd;

pub use conventions::{axis, units, Inertia, Pose, Quat, Vec3};
pub use reduce::{BinnedAcc, DeterministicAcc};
pub use scalar::{DoubleF32, Scalar};
