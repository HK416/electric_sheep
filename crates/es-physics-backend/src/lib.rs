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

pub mod mapping;
pub mod mjcf_out;
pub mod mjwarp;
pub mod mujoco;
pub mod newton;
pub mod proc;

pub use mapping::{
    compare_backends, lookup, mapping_report, BackendKind, CompareReport, Mapping, MappingReport,
    MappingRow, SemanticMapping, Severity, Spec17Row, Status, TaskFeature,
};
pub use mjcf_out::scene_to_mjcf;
pub use mjwarp::MjWarpBackend;
pub use mujoco::MuJoCoCpuBackend;
pub use newton::NewtonBackend;

use es_assets::scene::SceneDesc;
use es_physics_core::{Capabilities, LoadConfig, PhysicsBackend, PhysicsError};

/// What `--backend physx` says until packet M11/I1 lands.
pub const PHYSX_NOT_IMPLEMENTED: &str = "backend `physx`: not implemented (M11/I1)";

/// `H("es.backend.v1", name, engine version, float, determinism tier)` (spec 28.14 rule 2).
///
/// Every field is length-prefixed, so no two tuples hash the same bytes. The float and tier are
/// spelled as `evaluation.lock`'s `backend` block spells them.
pub fn backend_identity(caps: &Capabilities, engine_version: &str) -> [u8; 32] {
    let float = format!("{:?}", caps.float);
    let tier = format!("{:?}", caps.determinism);
    let mut h = blake3::Hasher::new();
    for field in [
        "es.backend.v1",
        caps.name.as_str(),
        engine_version,
        float.as_str(),
        tier.as_str(),
    ] {
        h.update(&(field.len() as u64).to_le_bytes());
        h.update(field.as_bytes());
    }
    *h.finalize().as_bytes()
}

/// The `hardware_capability` slot of `execution_hash` (spec 5.3) for a run on this backend:
/// [`backend_identity`] everywhere except `mujoco-cpu`, the reference (spec 17.1), whose slot
/// stays the all-zero value every committed `evaluation.lock` was hashed with.
pub fn hardware_capability(caps: &Capabilities, engine_version: &str) -> [u8; 32] {
    if BackendKind::from_name(&caps.name) == Some(BackendKind::MuJoCoCpu) {
        [0; 32]
    } else {
        backend_identity(caps, engine_version)
    }
}

/// A backend opened once on a scene: what a run on it writes into the hash chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendIdentity {
    pub kind: BackendKind,
    /// The load reply's engine version.
    pub engine_version: String,
    /// [`hardware_capability`] of the declaration and that version.
    pub hardware: [u8; 32],
}

/// Loads `scene` once on `kind` (one env) and reads the engine version off the load reply.
///
/// This is also the gate: a scene the backend cannot map is refused by its own `load`, with
/// the mapping report, before a process spawns (spec 14.4) -- which is where Newton stops on
/// any scene with actuators. `PhysX` is [`PHYSX_NOT_IMPLEMENTED`].
pub fn identify(kind: BackendKind, scene: &SceneDesc) -> Result<BackendIdentity, PhysicsError> {
    fn open<B: PhysicsBackend>(
        kind: BackendKind,
        mut backend: B,
        scene: &SceneDesc,
        version: fn(&B) -> Option<&str>,
    ) -> Result<BackendIdentity, PhysicsError> {
        backend.load(scene, &LoadConfig::default())?;
        let engine_version = version(&backend)
            .ok_or_else(|| PhysicsError::Protocol("loaded without an engine version".to_owned()))?
            .to_owned();
        Ok(BackendIdentity {
            kind,
            hardware: hardware_capability(backend.capabilities(), &engine_version),
            engine_version,
        })
    }
    match kind {
        BackendKind::MuJoCoCpu => open(
            kind,
            MuJoCoCpuBackend::new(),
            scene,
            MuJoCoCpuBackend::engine_version,
        ),
        BackendKind::MjWarp => open(
            kind,
            MjWarpBackend::new(),
            scene,
            MjWarpBackend::engine_version,
        ),
        BackendKind::Newton => open(
            kind,
            NewtonBackend::new(),
            scene,
            NewtonBackend::engine_version,
        ),
        BackendKind::PhysX => Err(PhysicsError::Unsupported(PHYSX_NOT_IMPLEMENTED.to_owned())),
    }
}

/// Whether `kind` can run here: the backend's own availability probe, or the `PhysX` refusal.
pub fn is_available(kind: BackendKind) -> Result<(), String> {
    match kind {
        BackendKind::MuJoCoCpu => MuJoCoCpuBackend::is_available(),
        BackendKind::MjWarp => MjWarpBackend::is_available(),
        BackendKind::Newton => NewtonBackend::is_available(),
        BackendKind::PhysX => Err(PHYSX_NOT_IMPLEMENTED.to_owned()),
    }
}

#[cfg(test)]
mod tests_support {
    /// Reads a shared MJCF fixture from the workspace `tests/fixtures/mjcf` directory.
    pub fn fixture(name: &str) -> String {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/mjcf/");
        std::fs::read_to_string(format!("{path}{name}"))
            .unwrap_or_else(|e| panic!("cannot read fixture {name}: {e}"))
    }
}
