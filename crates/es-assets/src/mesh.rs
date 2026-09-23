//! The mesh resolver: files named by a scene's `AssetKind::Mesh` entries become geometry on
//! the scene itself, and each `AssetRef` stops hashing its path and starts hashing its content
//! (packet M10/W2a, spec 5.3).
//!
//! This is the one place in `es-assets` that reads a file from disk. It is a free function
//! rather than a trait: `PhysicsBackend::load` and `EnvRenderer::new` take a `&SceneDesc` and
//! nothing else, so geometry a backend needs has to ride on the scene (INV-17, spec 28.13
//! rule 2), and resolving it is a step a caller takes between the importer and the backend.
//!
//! `AssetRef::path` is never rewritten — it is hash input (spec 5.3) — so a path is resolved
//! against `base_dir` on every call and nothing about the machine enters the scene.

use std::path::Path;

use thiserror::Error;

use crate::gltf::{mesh_content_hash, MeshData};
use crate::scene::{AssetKind, SceneDesc};

/// Why a mesh file did not become geometry. Always names the file.
#[derive(Debug, Error)]
pub enum MeshError {
    #[error("mesh `{name}`: cannot read `{path}`: {reason}")]
    Read {
        name: String,
        path: String,
        reason: String,
    },
    #[error("mesh `{name}`: `{path}` has no reader (`.stl` and `.obj` only)")]
    Unsupported { name: String, path: String },
    #[error("mesh `{name}`: `{path}`: {reason}")]
    Malformed {
        name: String,
        path: String,
        reason: String,
    },
}

/// Loads every mesh asset `scene` names but does not yet carry, relative to `base_dir`.
///
/// Idempotent: an asset already in `scene.meshes` is left alone, so calling this twice hashes
/// the same and costs one directory walk. A scene with no mesh asset is untouched — which is
/// why the committed primitives-only scenes' `scene_hash` does not move (spec 28.13 rule 2).
pub fn load(scene: &mut SceneDesc, base_dir: &Path) -> Result<(), MeshError> {
    let SceneDesc { assets, meshes, .. } = scene;
    for asset in assets.iter_mut() {
        if asset.kind != AssetKind::Mesh || meshes.contains_key(&asset.id) {
            continue;
        }
        let path = base_dir.join(&asset.path);
        let fail = |reason: String| MeshError::Read {
            name: asset.name.clone(),
            path: asset.path.clone(),
            reason,
        };
        // The extension decides before anything is read, so a `.dae` reports "no reader"
        // rather than whatever the file system says about it.
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if extension != "stl" && extension != "obj" {
            return Err(MeshError::Unsupported {
                name: asset.name.clone(),
                path: asset.path.clone(),
            });
        }
        let bytes = std::fs::read(&path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
        let decoded = if extension == "stl" {
            crate::stl::parse(&bytes)
        } else {
            std::str::from_utf8(&bytes)
                .map_err(|e| format!("OBJ is not UTF-8: {e}"))
                .and_then(crate::obj::parse)
        };
        let (positions, indices) = decoded.map_err(|reason| MeshError::Malformed {
            name: asset.name.clone(),
            path: asset.path.clone(),
            reason,
        })?;
        asset.hash = mesh_content_hash(&positions, None, None, &indices);
        meshes.insert(
            asset.id,
            MeshData {
                id: asset.id,
                name: asset.name.clone(),
                positions,
                normals: None,
                uvs: None,
                indices,
                material: None,
            },
        );
    }
    Ok(())
}
