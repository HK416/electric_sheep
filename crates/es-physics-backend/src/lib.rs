//! `es-physics-backend` (layer 4): concrete [`PhysicsBackend`](es_physics_core::PhysicsBackend)
//! adapters. See `docs/ARCHITECTURE.ko.md` spec 4.3, spec 17 and `docs/packets/M0/P36.md`.
//!
//! `MuJoCoCpuBackend` is the reference backend and CI oracle (spec 17.1). It needs a Python
//! interpreter with the `mujoco` package; [`MuJoCoCpuBackend::is_available`] says whether one
//! is there, so a build without it skips the oracle instead of failing. Nothing on the runtime
//! path depends on Python (spec 2.4) — this adapter is a test and evidence tool.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-core`, `es-math`, `es-assets`,
//! `es-physics-core` and external crates only.
#![forbid(unsafe_code)]

pub mod mjcf_out;
pub mod mujoco;
pub mod proc;

pub use mjcf_out::scene_to_mjcf;
pub use mujoco::MuJoCoCpuBackend;

#[cfg(test)]
mod tests_support {
    /// Reads a shared MJCF fixture from the workspace `tests/fixtures/mjcf` directory.
    pub fn fixture(name: &str) -> String {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/mjcf/");
        std::fs::read_to_string(format!("{path}{name}"))
            .unwrap_or_else(|e| panic!("cannot read fixture {name}: {e}"))
    }
}
