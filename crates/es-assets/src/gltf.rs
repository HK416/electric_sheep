//! glTF 2.0 importer (P31): `.gltf` (JSON + embedded/external buffers) and `.glb` to
//! [`SceneDesc`] plus the mesh and material payloads a physics backend or renderer needs.
//!
//! glTF has no joints, bodies or physics: every node becomes a static [`Body`] with no
//! [`crate::scene::Joint`], welded to its parent by construction (there is nothing to weld —
//! `Body::parent` already means "fixed to parent" when no joint targets the body). A node's
//! mesh becomes one [`Geom`] per primitive, `contype`/`conaffinity` `0` (visual only): glTF
//! carries no collision intent, so nothing here invents one.
//!
//! # Axis conversion (spec 3.1)
//!
//! glTF is right-handed Y-up (+X right, +Y up, +Z toward the viewer). Spec 3.1 is right-handed
//! Z-up, X-forward. [`AXIS_FIX`] is the one fixed rotation between them: glTF `+Y` (up) becomes
//! our `+Z`, glTF `-Z` becomes our `+X`, glTF `-X` becomes our `+Y`. Every position is rotated
//! by it ([`convert_point`]); every orientation is conjugated by it ([`convert_rotation`]):
//! `q' = FIX * q * FIX^-1`. Conjugation is a homomorphism and rotation distributes over
//! [`Pose::compose`], so converting each node independently and then composing the hierarchy
//! gives exactly the same result as composing in glTF space and converting once — a per-node
//! conversion is correct without walking the whole tree twice.
//!
//! Mesh vertex positions and normals are authored in the node's own local frame, which is a
//! glTF-axis frame like any other in the hierarchy, so they get the same rotation (no
//! translation) before [`MeshData`] is built — a caller never has to axis-convert geometry
//! itself to place it under the [`Geom`] whose `pose` is identity relative to its body.
//!
//! Scale has no place in a rigid [`Pose`]. A uniform node scale is baked into that node's mesh
//! vertex positions (harmless: it does not change normal directions). A non-uniform scale would
//! need an inverse-transpose correction on normals to stay correct, which this importer does not
//! implement; instead it records a [`Warning`] and leaves the geometry unscaled, honestly rather
//! than silently wrong.
//!
//! # Stable ids (spec 5.3), following the scheme documented in [`crate::mjcf`]
//!
//! ```text
//! body   body/<node path>                node path = names from a root node down, "/"-joined
//! geom   geom/<node path>/prim<n>        one geom per primitive in the node's mesh
//! asset  asset/mesh/<mesh name>/prim<n>  mesh name from the glTF mesh, not the node
//! asset  asset/material/<material name>
//! ```
//!
//! A node or mesh the source file leaves unnamed is named `node_<index>` / `mesh_<index>`
//! (the glTF document index), so the id is stable across re-imports of the same file but, like
//! the MJCF importer's unnamed elements, depends on the file's own node order.
//!
//! Node traversal is document order: [`gltf::Document::nodes`] visits nodes index-first: `0,
//! 1, 2, ...`; a node reachable only as a child is visited when its parent recurses into it, so
//! the effective order is a preorder walk seeded by the nodes nobody claims as a child.
//!
//! # Content hash (spec 5.3)
//!
//! A mesh [`AssetRef::hash`] is `blake3` of the *decoded* vertex/index arrays, not of the
//! source bytes: the same triangle mesh stored as a `.gltf` data URI and as a `.glb` binary
//! chunk decodes to the same `f32`/`u32` arrays and therefore hashes identically (tested below).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use es_core::StableId;
use es_math::{Pose, Quat, Vec3};
use gltf::mesh::Mode;
use gltf::{Document, Gltf};
use thiserror::Error;

use crate::scene::{scene_id, AssetKind, AssetRef, Body, Geom, SceneDesc, Shape};

/// The fixed rotation from glTF (Y-up) to spec 3.1 (Z-up, X-forward); see the module docs.
/// Order 3 (`AXIS_FIX^3 == IDENTITY`): it is a 120 degree rotation about `(-1, 1, 1)`.
const AXIS_FIX: Quat = Quat::from_xyzw(0.5, -0.5, -0.5, 0.5);

/// Domain separator for [`mesh_content_hash`].
const MESH_TAG: &str = "es.gltf.mesh.v1";

/// Decoded positions/normals/UVs/indices for one primitive.
type PrimitiveGeometry = (
    Vec<[f32; 3]>,
    Option<Vec<[f32; 3]>>,
    Option<Vec<[f32; 2]>>,
    Vec<u32>,
);
/// Per-primitive `(mesh asset id, material id, fallback rgba)`, cached by glTF mesh index.
type MeshPrimEntry = (StableId, Option<StableId>, [f64; 4]);
type MeshCache = BTreeMap<usize, Vec<MeshPrimEntry>>;

/// A parsed glTF asset: the scene it describes plus the geometry and materials a backend or
/// renderer needs to actually draw or collide with it.
#[derive(Debug)]
pub struct GltfImport {
    pub scene: SceneDesc,
    pub meshes: Vec<MeshData>,
    pub materials: Vec<MaterialData>,
    pub warnings: Vec<Warning>,
}

/// Decoded geometry for one glTF primitive. `id` matches the [`AssetRef`] of the same mesh in
/// [`GltfImport::scene`], so a caller can join the two.
#[derive(Clone, Debug)]
pub struct MeshData {
    pub id: StableId,
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub indices: Vec<u32>,
    pub material: Option<StableId>,
}

/// A glTF material's PBR metallic-roughness parameters. `id` matches the [`AssetRef`] of the
/// same material in [`GltfImport::scene`].
#[derive(Clone, Debug)]
pub struct MaterialData {
    pub id: StableId,
    pub name: String,
    pub base_color: [f64; 4],
    pub metallic: f64,
    pub roughness: f64,
    pub emissive: [f64; 3],
    /// Keyed by slot: `base_color`, `metallic_roughness`, `normal`, `emissive`, `occlusion`.
    pub textures: BTreeMap<String, AssetRef>,
}

/// Something the source file says that the scene description cannot carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    pub message: String,
}

/// Why a glTF/GLB buffer could not be turned into a [`GltfImport`]. Never a panic: malformed
/// input (a truncated buffer, a dangling accessor reference) is one of these, not a crash.
#[derive(Debug, Error)]
pub enum GltfError {
    #[error("glTF: {0}")]
    Parse(#[from] gltf::Error),
    #[error("buffer {index}: the file has no BIN chunk (not a .glb, or the chunk is missing)")]
    MissingBlob { index: usize },
    #[error("buffer {index}: data URI has no `base64` marker or no `,` payload separator")]
    BadDataUri { index: usize },
    #[error("buffer {index}: declared length {expected} bytes, decoded {actual} bytes")]
    BufferLength {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("buffer {index}: external file `{path}` could not be read: {reason}")]
    ExternalBuffer {
        index: usize,
        path: String,
        reason: String,
    },
    #[error("mesh `{mesh}` primitive {primitive}: no POSITION attribute")]
    MissingPositions { mesh: String, primitive: usize },
    #[error("node `{node}` is its own ancestor (cycle in the node hierarchy)")]
    NodeCycle { node: String },
}

/// Parses a `.gltf` (JSON, with embedded data-URI or external buffers) or `.glb` asset.
///
/// `base_dir` resolves buffers and images given as relative file URIs; pass `None` when every
/// buffer is embedded (a data URI, or the `.glb` binary chunk) — an external reference without
/// a `base_dir` is a [`GltfError::ExternalBuffer`], not a panic.
pub fn import_gltf(bytes: &[u8], base_dir: Option<&Path>) -> Result<GltfImport, GltfError> {
    let gltf = Gltf::from_slice(bytes)?;
    let document = &gltf.document;
    let mut warnings = Vec::new();
    let buffers = load_buffers(document, gltf.blob.as_deref(), base_dir)?;

    let mut assets = Vec::new();
    let mut material_ids = BTreeMap::new();
    let mut materials = Vec::new();
    for material in document.materials() {
        let index = material.index().unwrap_or_default();
        let name = material
            .name()
            .map_or_else(|| format!("material_{index}"), str::to_owned);
        let asset = AssetRef::from_path(AssetKind::Material, &name, &format!("material/{name}"));
        material_ids.insert(index, asset.id);
        materials.push(material_data(&material, asset.id, name));
        assets.push(asset);
    }

    // A node reachable only as someone's child is visited when that parent recurses into it;
    // a node nobody claims is a traversal root (spec: "document order" — see module docs).
    let is_child: BTreeSet<usize> = document
        .nodes()
        .flat_map(|n| n.children().map(|c| c.index()))
        .collect();

    let mut bodies = Vec::new();
    let mut meshes = Vec::new();
    let mut mesh_cache = BTreeMap::new();
    for node in document.nodes() {
        if is_child.contains(&node.index()) {
            continue;
        }
        let mut ancestors = Vec::new();
        visit_node(
            &node,
            None,
            "",
            &buffers,
            &material_ids,
            &mut mesh_cache,
            &mut meshes,
            &mut assets,
            &mut bodies,
            &mut warnings,
            &mut ancestors,
        )?;
    }

    let scene = SceneDesc {
        name: "gltf".to_owned(),
        bodies,
        assets,
        ..SceneDesc::default()
    };
    Ok(GltfImport {
        scene,
        meshes,
        materials,
        warnings,
    })
}

/// Visits one node and its subtree, appending to `bodies`/`meshes`/`assets`/`warnings` in
/// document order. `ancestors` is the current root-to-node path, used only to reject a cycle.
#[allow(clippy::too_many_arguments)]
fn visit_node(
    node: &gltf::Node<'_>,
    parent_id: Option<StableId>,
    parent_path: &str,
    buffers: &[Vec<u8>],
    material_ids: &BTreeMap<usize, StableId>,
    mesh_cache: &mut MeshCache,
    meshes: &mut Vec<MeshData>,
    assets: &mut Vec<AssetRef>,
    bodies: &mut Vec<Body>,
    warnings: &mut Vec<Warning>,
    ancestors: &mut Vec<usize>,
) -> Result<(), GltfError> {
    if ancestors.contains(&node.index()) {
        return Err(GltfError::NodeCycle {
            node: node.name().unwrap_or("<unnamed>").to_owned(),
        });
    }
    let local_name = node
        .name()
        .map_or_else(|| format!("node_{}", node.index()), str::to_owned);
    let path = if parent_path.is_empty() {
        local_name.clone()
    } else {
        format!("{parent_path}/{local_name}")
    };
    let id = scene_id("body", &path);

    let (t, r, s) = node.transform().decomposed();
    let pose = Pose::new(convert_point(t), convert_rotation(r));

    let mut geoms = Vec::new();
    if let Some(mesh) = node.mesh() {
        let scale = if let Some(scale) = uniform_scale(s) {
            scale
        } else {
            let mesh_name = mesh.name().unwrap_or("<unnamed>");
            warnings.push(Warning {
                message: format!(
                    "node `{path}`: non-uniform scale {s:?} on mesh `{mesh_name}` \
                     is not applied to geometry (would need an inverse-transpose \
                     correction on normals)"
                ),
            });
            1.0
        };
        let prims = mesh_geoms(
            &mesh,
            buffers,
            material_ids,
            scale,
            mesh_cache,
            meshes,
            assets,
            warnings,
        )?;
        for (i, (asset_id, material_id, rgba)) in prims.into_iter().enumerate() {
            geoms.push(Geom {
                id: scene_id("geom", &format!("{path}/prim{i}")),
                name: format!("prim{i}"),
                shape: Shape::Mesh { asset: asset_id },
                pose: Pose::IDENTITY,
                friction: [1.0, 0.005, 0.0001],
                contype: 0,
                conaffinity: 0,
                condim: 3,
                priority: 0,
                density: 1000.0,
                mass: None,
                margin: 0.0,
                gap: 0.0,
                solref: [0.02, 1.0],
                solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
                material: material_id,
                rgba,
                visual_only: true,
            });
        }
    }

    bodies.push(Body {
        id,
        name: local_name,
        parent: parent_id,
        pose,
        inertial: None,
        geoms,
        sites: Vec::new(),
    });

    ancestors.push(node.index());
    for child in node.children() {
        visit_node(
            &child,
            Some(id),
            &path,
            buffers,
            material_ids,
            mesh_cache,
            meshes,
            assets,
            bodies,
            warnings,
            ancestors,
        )?;
    }
    ancestors.pop();
    Ok(())
}

/// Builds (or returns the cached) `(asset id, material id, fallback rgba)` per primitive of
/// `mesh`. Caching is keyed by the glTF mesh index: a primitive's material binding is intrinsic
/// to the mesh, not the node, so it is safe to share across every node that instances this mesh.
// ponytail: `scale` is only applied the first time a mesh is built, from whichever node
// references it first. Two nodes instancing the same mesh with different uniform scales would
// share one (wrongly) pre-scaled asset; add per-instance mesh variants if that case shows up.
#[allow(clippy::too_many_arguments)]
fn mesh_geoms(
    mesh: &gltf::Mesh<'_>,
    buffers: &[Vec<u8>],
    material_ids: &BTreeMap<usize, StableId>,
    scale: f64,
    cache: &mut MeshCache,
    meshes: &mut Vec<MeshData>,
    assets: &mut Vec<AssetRef>,
    warnings: &mut Vec<Warning>,
) -> Result<Vec<MeshPrimEntry>, GltfError> {
    if let Some(cached) = cache.get(&mesh.index()) {
        return Ok(cached.clone());
    }
    let mesh_name = mesh
        .name()
        .map_or_else(|| format!("mesh_{}", mesh.index()), str::to_owned);
    let mut out = Vec::new();
    for primitive in mesh.primitives() {
        if primitive.mode() != Mode::Triangles {
            warnings.push(Warning {
                message: format!(
                    "mesh `{mesh_name}` primitive {}: mode {:?} is not Triangles; \
                     indices kept as-is",
                    primitive.index(),
                    primitive.mode(),
                ),
            });
        }
        let (positions, normals, uvs, indices) =
            primitive_geometry(&primitive, buffers, &mesh_name)?;
        // Vertex data is authored in the node's local frame, which is a glTF-axis frame like
        // any other in the hierarchy: it needs the same AXIS_FIX rotation as a pose (spec 3.1
        // module docs). Normals are directions, not positions, but a rotation (no translation)
        // treats both alike.
        let mut positions: Vec<[f32; 3]> = positions.into_iter().map(convert_vec).collect();
        let normals = normals.map(|ns| ns.into_iter().map(convert_vec).collect());
        // Multiplying by the identity scale (1.0) is harmless, so this always runs rather
        // than branching on a float comparison.
        for p in &mut positions {
            for c in p {
                *c = (f64::from(*c) * scale) as f32;
            }
        }
        let hash = mesh_content_hash(&positions, normals.as_deref(), uvs.as_deref(), &indices);
        let asset_name = format!("{mesh_name}/prim{}", primitive.index());
        let mut asset =
            AssetRef::from_path(AssetKind::Mesh, &asset_name, &format!("mesh/{asset_name}"));
        asset.hash = hash;

        let material = primitive.material();
        let material_id = material.index().and_then(|i| material_ids.get(&i).copied());
        let rgba = material
            .pbr_metallic_roughness()
            .base_color_factor()
            .map(f64::from);

        meshes.push(MeshData {
            id: asset.id,
            name: asset_name,
            positions,
            normals,
            uvs,
            indices,
            material: material_id,
        });
        out.push((asset.id, material_id, rgba));
        assets.push(asset);
    }
    cache.insert(mesh.index(), out.clone());
    Ok(out)
}

/// Reads one primitive's vertex/index data out of the already-decoded `buffers`. Missing
/// indices means non-indexed geometry: `0..vertex_count` in order, per the glTF spec.
fn primitive_geometry<'a>(
    primitive: &'a gltf::Primitive<'a>,
    buffers: &[Vec<u8>],
    mesh_name: &str,
) -> Result<PrimitiveGeometry, GltfError> {
    let reader = primitive.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| GltfError::MissingPositions {
            mesh: mesh_name.to_owned(),
            primitive: primitive.index(),
        })?
        .collect();
    let normals = reader.read_normals().map(Iterator::collect);
    let uvs = reader.read_tex_coords(0).map(|it| it.into_f32().collect());
    let indices = match reader.read_indices() {
        Some(idx) => idx.into_u32().collect(),
        None => (0..u32::try_from(positions.len()).unwrap_or(u32::MAX)).collect(),
    };
    Ok((positions, normals, uvs, indices))
}

/// `blake3` of the decoded vertex/index arrays, length-prefixed so presence/absence of
/// normals and UVs cannot be confused with different array contents.
fn mesh_content_hash(
    positions: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    uvs: Option<&[[f32; 2]]>,
    indices: &[u32],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(MESH_TAG.as_bytes());
    h.update(&(positions.len() as u32).to_le_bytes());
    for p in positions {
        for c in p {
            h.update(&c.to_le_bytes());
        }
    }
    match normals {
        None => {
            h.update(&[0u8]);
        }
        Some(ns) => {
            h.update(&[1u8]);
            h.update(&(ns.len() as u32).to_le_bytes());
            for n in ns {
                for c in n {
                    h.update(&c.to_le_bytes());
                }
            }
        }
    }
    match uvs {
        None => {
            h.update(&[0u8]);
        }
        Some(uv) => {
            h.update(&[1u8]);
            h.update(&(uv.len() as u32).to_le_bytes());
            for p in uv {
                for c in p {
                    h.update(&c.to_le_bytes());
                }
            }
        }
    }
    h.update(&(indices.len() as u32).to_le_bytes());
    for i in indices {
        h.update(&i.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

fn material_data(material: &gltf::Material<'_>, id: StableId, name: String) -> MaterialData {
    let pbr = material.pbr_metallic_roughness();
    let mut textures = BTreeMap::new();
    if let Some(info) = pbr.base_color_texture() {
        textures.insert("base_color".to_owned(), texture_asset(&info.texture()));
    }
    if let Some(info) = pbr.metallic_roughness_texture() {
        textures.insert(
            "metallic_roughness".to_owned(),
            texture_asset(&info.texture()),
        );
    }
    if let Some(info) = material.normal_texture() {
        textures.insert("normal".to_owned(), texture_asset(&info.texture()));
    }
    if let Some(info) = material.emissive_texture() {
        textures.insert("emissive".to_owned(), texture_asset(&info.texture()));
    }
    if let Some(info) = material.occlusion_texture() {
        textures.insert("occlusion".to_owned(), texture_asset(&info.texture()));
    }
    MaterialData {
        id,
        name,
        base_color: pbr.base_color_factor().map(f64::from),
        metallic: f64::from(pbr.metallic_factor()),
        roughness: f64::from(pbr.roughness_factor()),
        emissive: material.emissive_factor().map(f64::from),
        textures,
    }
}

/// A texture's identity is path-based (spec 5.3 leaves that to a later packet once image bytes
/// are actually decoded): the URI for an external/data-URI image, else a synthetic path naming
/// the embedded buffer view, so two textures pointing at the same bytes hash the same.
fn texture_asset(texture: &gltf::Texture<'_>) -> AssetRef {
    let image = texture.source();
    let path = match image.source() {
        gltf::image::Source::Uri { uri, .. } => uri.to_owned(),
        gltf::image::Source::View { view, mime_type } => {
            format!("bufferview:{}:{mime_type}", view.index())
        }
    };
    let name = image
        .name()
        .map_or_else(|| format!("image_{}", image.index()), str::to_owned);
    AssetRef::from_path(AssetKind::Texture, &name, &path)
}

/// `Some(scale)` when `x`/`y`/`z` agree to within a relative tolerance; `None` otherwise.
fn uniform_scale(scale: [f32; 3]) -> Option<f64> {
    let [sx, sy, sz] = scale.map(f64::from);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0);
    (close(sx, sy) && close(sy, sz)).then_some((sx + sy + sz) / 3.0)
}

/// Rotates a glTF-space point into spec 3.1 space; see the module docs for [`AXIS_FIX`].
fn convert_point(p: [f32; 3]) -> Vec3 {
    AXIS_FIX.rotate(Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])))
}

/// [`convert_point`], rounded back to `f32` for vertex data (positions and normals alike: a
/// rotation with no translation treats a direction the same as a point).
fn convert_vec(v: [f32; 3]) -> [f32; 3] {
    let c = convert_point(v);
    [c.x as f32, c.y as f32, c.z as f32]
}

/// Conjugates a glTF-space orientation into spec 3.1 space; see the module docs for
/// [`AXIS_FIX`].
fn convert_rotation(q: [f32; 4]) -> Quat {
    let q = Quat::from_xyzw(
        f64::from(q[0]),
        f64::from(q[1]),
        f64::from(q[2]),
        f64::from(q[3]),
    )
    .normalize();
    (AXIS_FIX * q * AXIS_FIX.conjugate()).normalize()
}

/// Decodes every buffer in `document` up front so accessor readers can borrow plain slices.
/// `blob` is the `.glb` BIN chunk, if any; `base_dir` resolves relative file URIs.
fn load_buffers(
    document: &Document,
    blob: Option<&[u8]>,
    base_dir: Option<&Path>,
) -> Result<Vec<Vec<u8>>, GltfError> {
    let mut out = Vec::with_capacity(document.buffers().count());
    for buffer in document.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => {
                let blob = blob.ok_or(GltfError::MissingBlob {
                    index: buffer.index(),
                })?;
                blob.get(..buffer.length())
                    .ok_or(GltfError::BufferLength {
                        index: buffer.index(),
                        expected: buffer.length(),
                        actual: blob.len(),
                    })?
                    .to_vec()
            }
            gltf::buffer::Source::Uri(uri) => {
                if let Some(rest) = uri.strip_prefix("data:") {
                    let (header, payload) = rest.split_once(',').ok_or(GltfError::BadDataUri {
                        index: buffer.index(),
                    })?;
                    if !header.contains("base64") {
                        return Err(GltfError::BadDataUri {
                            index: buffer.index(),
                        });
                    }
                    base64_decode(payload).ok_or(GltfError::BadDataUri {
                        index: buffer.index(),
                    })?
                } else {
                    let dir = base_dir.ok_or_else(|| GltfError::ExternalBuffer {
                        index: buffer.index(),
                        path: uri.to_owned(),
                        reason: "no base directory given for an external buffer".to_owned(),
                    })?;
                    let path = dir.join(percent_decode(uri));
                    std::fs::read(&path).map_err(|e| GltfError::ExternalBuffer {
                        index: buffer.index(),
                        path: path.display().to_string(),
                        reason: e.to_string(),
                    })?
                }
            }
        };
        if data.len() != buffer.length() {
            return Err(GltfError::BufferLength {
                index: buffer.index(),
                expected: buffer.length(),
                actual: data.len(),
            });
        }
        out.push(data);
    }
    Ok(out)
}

/// Standard base64 (RFC 4648) decoder for data-URI buffers. Not a crate dependency: `gltf`'s
/// own base64 support is behind its `import` feature, which this crate does not enable (it
/// pulls in `image` and reads from disk, neither of which belongs in an in-memory importer).
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn value(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes: Vec<u8> = s.bytes().filter(u8::is_ascii_graphic).collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let b0 = value(chunk[0])?;
        let b1 = value(*chunk.get(1)?)?;
        out.push((b0 << 2) | (b1 >> 4));
        match chunk.get(2) {
            Some(&b'=') | None => {}
            Some(&c) => {
                let b2 = value(c)?;
                out.push((b1 << 4) | (b2 >> 2));
                match chunk.get(3) {
                    Some(&b'=') | None => {}
                    Some(&c3) => out.push((b2 << 6) | value(c3)?),
                }
            }
        }
    }
    Some(out)
}

/// Percent-decodes a URI path component (RFC 3986 `%XX`); anything that is not a valid escape
/// passes through unchanged rather than failing the whole import over a cosmetic filename.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_fix_maps_gltf_up_and_forward() {
        // glTF +Y (up) -> our +Z; glTF -Z (forward) -> our +X.
        assert_eq!(convert_point([0.0, 1.0, 0.0]), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(convert_point([0.0, 0.0, -1.0]), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(convert_point([1.0, 0.0, 0.0]), Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn axis_fix_has_order_three() {
        // A 120 degree rotation applied three times is the identity.
        let twice = AXIS_FIX * AXIS_FIX;
        let thrice = (AXIS_FIX * twice).normalize();
        assert!((thrice.w.abs() - 1.0).abs() < 1e-9, "{thrice:?}");
    }

    #[test]
    fn convert_rotation_preserves_a_up_axis_spin() {
        // 90 degrees about glTF +Y (up) must become 90 degrees about our +Z (up).
        let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let q = convert_rotation([0.0, half_sqrt2, 0.0, half_sqrt2]);
        let expected = f64::from(half_sqrt2);
        assert!((q.x).abs() < 1e-6, "{q:?}");
        assert!((q.y).abs() < 1e-6, "{q:?}");
        assert!((q.z - expected).abs() < 1e-6, "{q:?}");
        assert!((q.w - expected).abs() < 1e-6, "{q:?}");
    }

    #[test]
    fn uniform_scale_accepts_equal_components_only() {
        assert_eq!(uniform_scale([2.0, 2.0, 2.0]), Some(2.0));
        assert_eq!(uniform_scale([1.0, 1.0, 1.0]), Some(1.0));
        assert_eq!(uniform_scale([2.0, 1.0, 1.0]), None);
    }

    #[test]
    fn mesh_hash_is_sensitive_to_every_array() {
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let idx = [0u32, 1, 2];
        let base = mesh_content_hash(&p, None, None, &idx);
        assert_eq!(base, mesh_content_hash(&p, None, None, &idx));
        assert_ne!(base, mesh_content_hash(&p, Some(&p), None, &idx));
        assert_ne!(base, mesh_content_hash(&p, None, None, &[0, 2, 1]));
    }

    #[test]
    fn base64_round_trips() {
        assert_eq!(base64_decode("AAAA").unwrap(), vec![0, 0, 0]);
        assert_eq!(base64_decode("/w==").unwrap(), vec![0xff]);
        assert_eq!(base64_decode("//8=").unwrap(), vec![0xff, 0xff]);
        assert!(base64_decode("not valid!!").is_none());
    }

    #[test]
    fn percent_decode_handles_escapes_and_leaves_the_rest() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("plain.bin"), "plain.bin");
    }
}
