//! `es-assets` (layer 2): the backend-neutral scene description and the importers that
//! produce it. See `docs/ARCHITECTURE.ko.md` spec 5.3 (`asset_hash` -> `scene_hash`), spec 3.1
//! (conventions), spec 17.2 (backend semantic mapping) and the work packets
//! `docs/packets/M0/P30.md`, `P32.md`, `P33.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-core`, `es-math` and external crates
//! only. Nothing here computes physics, and nothing here reads a file from disk.
#![forbid(unsafe_code)]

pub mod mjcf;
pub mod scene;

pub use mjcf::{parse_str as parse_mjcf, Import, MjcfError, Warning};
pub use scene::{scene_id, SceneDesc, SceneError};
