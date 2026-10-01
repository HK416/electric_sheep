//! `es-assets` (layer 2): the backend-neutral scene description and the importers that
//! produce it. See `docs/ARCHITECTURE.ko.md` spec 5.3 (`asset_hash` -> `scene_hash`), spec 3.1
//! (conventions), spec 17.2 (backend semantic mapping) and the work packets
//! `docs/packets/M0/P30.md`, `P32.md`, `P33.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-core`, `es-math` and external crates
//! only. Nothing here computes physics, and the one thing that reads a file from disk is
//! [`mesh::load`], the mesh resolver of packet M10/W2a, which since plan H's HT1 also decodes
//! the scene's textures through [`texture::load`] -- an importer is still handed bytes.
#![forbid(unsafe_code)]

pub mod esscene;
pub mod fuzz;
pub mod gltf;
pub mod mesh;
pub mod mjcf;
pub mod obj;
pub mod scene;
pub mod stl;
pub mod texture;
pub mod urdf;

pub use mesh::MeshError;
pub use mjcf::{parse_str as parse_mjcf, Import, MjcfError, Warning};
pub use scene::{scene_id, SceneDesc, SceneError};
