//! `es-py` (layer 11): the Python authoring builder (spec 14.2, M2 W6).
//!
//! [`builder`] is the language-neutral core — four builders (Task, Observation, Learning,
//! Deployment) over the existing `es-ir` factories and graph skeleton, with zero Python
//! involvement and zero per-node-kind code. It compiles and is unit-tested in every
//! `cargo test --workspace` run.
//!
//! [`rollout`] is the second half (packet M8/S4a, spec 13.4): `Rollout`, a plain Rust struct
//! that owns an `Env`, one `SafetyPlane` and one `CpuPlan` per env, so a Python trainer steps
//! *our* runtime — through the Safety Plane, never around it — instead of a simulator of its
//! own. It too is Python-free and unit-tested as ordinary Rust.
//!
//! `#[cfg(feature = "python")]` adds [`pybind`], a pyo3 `es_native` extension module wrapping
//! those same four builders with the Pythonic names spec 14.2 uses. The feature is off by
//! default (see `Cargo.toml`) so the default workspace build never links libpython; `maturin`
//! turns it on.

pub mod builder;
pub mod rollout;

// `#[pymodule] fn es_native` inside generates the `PyInit_es_native` entry point maturin's
// build (crate-type `cdylib`, `[lib] name = "es_native"`) loads by symbol name — nothing here
// needs to re-export it.
#[cfg(feature = "python")]
mod pybind;
