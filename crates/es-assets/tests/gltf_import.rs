//! P31 oracle: hand-written glTF fixtures, asserted value by value, plus a `.glb` built at test
//! time from the same geometry so content hashes can be compared across containers.
//!
//! Fixtures live in `tests/fixtures/gltf/` at the workspace root (inputs, not golden outputs,
//! so they are not under `tests/golden/`), same convention as `tests/fixtures/mjcf/` and
//! `tests/fixtures/urdf/`.
#![allow(clippy::float_cmp)]

use es_assets::gltf::{import_gltf, GltfError};
use es_assets::scene::{scene_id, Shape};
use es_math::{Quat, Vec3};

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/../../tests/fixtures/gltf/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The triangle shared by every fixture: positions as authored in the `.gltf`/`.glb` (glTF
/// Y-up local space), packed little-endian, followed by its `u16` indices.
fn triangle_bin() -> Vec<u8> {
    let mut buf = Vec::new();
    for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in [0u16, 1, 2] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// A `.glb` with the same node/mesh/accessor JSON as `tests/fixtures/gltf/hierarchy.gltf`, but
/// with the buffer's `uri` dropped (so it reads from the BIN chunk instead) and its bytes
/// supplied directly, rather than as a data URI. This is the "build a GLB from the JSON" step
/// the packet asks for: same document, same triangle, different container.
fn hierarchy_glb() -> Vec<u8> {
    let json = r#"{
      "asset": { "version": "2.0" },
      "scene": 0,
      "scenes": [{ "nodes": [0] }],
      "nodes": [
        { "name": "root", "translation": [10.0, 0.0, 0.0], "children": [1] },
        {
          "mesh": 0,
          "translation": [0.0, 0.0, 5.0],
          "rotation": [0.0, 0.70710678, 0.0, 0.70710678]
        }
      ],
      "meshes": [
        {
          "name": "tri",
          "primitives": [
            { "attributes": { "POSITION": 0 }, "indices": 1, "material": 0 }
          ]
        }
      ],
      "materials": [
        {
          "name": "red",
          "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 0.0, 0.0, 1.0],
            "metallicFactor": 0.2,
            "roughnessFactor": 0.8
          }
        }
      ],
      "accessors": [
        {
          "bufferView": 0,
          "byteOffset": 0,
          "componentType": 5126,
          "count": 3,
          "type": "VEC3",
          "min": [0.0, 0.0, 0.0],
          "max": [1.0, 1.0, 0.0]
        },
        {
          "bufferView": 1,
          "byteOffset": 0,
          "componentType": 5123,
          "count": 3,
          "type": "SCALAR"
        }
      ],
      "bufferViews": [
        { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
        { "buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963 }
      ],
      "buffers": [{ "byteLength": 42 }]
    }"#;
    build_glb(json.as_bytes(), &triangle_bin())
}

/// Packs `json` and `bin` into a minimal `.glb`: a 12-byte header, then a `JSON` chunk (padded
/// with spaces) and a `BIN\0` chunk (padded with zeros) — glTF requires every chunk length to
/// be a multiple of 4 bytes.
fn build_glb(json: &[u8], bin: &[u8]) -> Vec<u8> {
    fn padded(data: &[u8], pad_with: u8) -> Vec<u8> {
        let mut out = data.to_vec();
        while out.len() % 4 != 0 {
            out.push(pad_with);
        }
        out
    }
    let json = padded(json, b' ');
    let bin = padded(bin, 0);
    let total = 12 + 8 + json.len() + 8 + bin.len();

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(total).unwrap().to_le_bytes());
    out.extend_from_slice(&u32::try_from(json.len()).unwrap().to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&u32::try_from(bin.len()).unwrap().to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}

#[test]
fn hierarchy_pose_and_axis_conversion() {
    let import = import_gltf(&fixture("hierarchy.gltf"), None).unwrap();
    import.scene.validate().unwrap();
    assert!(import.warnings.is_empty(), "{:?}", import.warnings);
    assert_eq!(import.scene.bodies.len(), 2);

    let root = import
        .scene
        .bodies
        .iter()
        .find(|b| b.name == "root")
        .expect("root body");
    assert_eq!(root.id, scene_id("body", "root"));
    assert_eq!(root.parent, None);
    // glTF translation [10,0,0] -> our (-z,-x,y) = (0,-10,0).
    assert_eq!(root.pose.position, Vec3::new(0.0, -10.0, 0.0));
    assert_eq!(root.pose.orientation, Quat::IDENTITY);

    // The child node has no "name" in the fixture: node_<index> (spec: unnamed -> node_<index>).
    let child = import
        .scene
        .bodies
        .iter()
        .find(|b| b.name == "node_1")
        .expect("unnamed child body");
    assert_eq!(child.id, scene_id("body", "root/node_1"));
    assert_eq!(child.parent, Some(root.id));
    // glTF translation [0,0,5] -> (-5,0,0).
    let p = child.pose.position;
    assert!(
        (p.x + 5.0).abs() < 1e-6 && p.y.abs() < 1e-6 && p.z.abs() < 1e-6,
        "{p:?}"
    );
    // glTF: 90 degrees about +Y (up) -> our 90 degrees about +Z (up).
    let q = child.pose.orientation;
    let half_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert!(q.x.abs() < 1e-6 && q.y.abs() < 1e-6, "{q:?}");
    assert!(
        (q.z - half_sqrt2).abs() < 1e-6 && (q.w - half_sqrt2).abs() < 1e-6,
        "{q:?}"
    );

    assert_eq!(child.geoms.len(), 1);
    let Shape::Mesh { asset } = child.geoms[0].shape else {
        panic!("expected a mesh geom");
    };
    assert_eq!(import.meshes.len(), 1);
    assert_eq!(import.meshes[0].id, asset);
    // Vertex data is authored in the node's local (glTF) axes and gets the same rotation as a
    // pose: (0,0,0)->(0,0,0), (1,0,0)->(0,-1,0), (0,1,0)->(0,0,1).
    assert_eq!(
        import.meshes[0].positions,
        vec![[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    );
    assert_eq!(import.meshes[0].indices, vec![0, 1, 2]);

    assert_eq!(import.materials.len(), 1);
    assert_eq!(import.materials[0].name, "red");
    assert_eq!(import.materials[0].base_color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(import.meshes[0].material, Some(import.materials[0].id));
}

#[test]
fn mesh_hash_matches_across_gltf_and_glb() {
    let from_gltf = import_gltf(&fixture("hierarchy.gltf"), None).unwrap();
    let from_glb = import_gltf(&hierarchy_glb(), None).unwrap();

    let hash_of = |import: &es_assets::gltf::GltfImport| {
        import
            .scene
            .assets
            .iter()
            .find(|a| a.kind == es_assets::scene::AssetKind::Mesh)
            .expect("mesh asset")
            .hash
    };
    assert_eq!(hash_of(&from_gltf), hash_of(&from_glb));
    // Sanity: the hash is not just a constant — it actually reflects the geometry.
    assert_eq!(from_gltf.meshes[0].positions, from_glb.meshes[0].positions);
    assert_eq!(from_gltf.meshes[0].indices, from_glb.meshes[0].indices);
}

#[test]
fn non_uniform_scale_warns_and_leaves_geometry_unscaled() {
    let import = import_gltf(&fixture("nonuniform_scale.gltf"), None).unwrap();
    assert_eq!(import.warnings.len(), 1);
    assert!(
        import.warnings[0].message.contains("non-uniform"),
        "{:?}",
        import.warnings[0]
    );
    // Unscaled (but still axis-converted): (1,0,0) -> (0,-1,0), (0,1,0) -> (0,0,1).
    assert_eq!(
        import.meshes[0].positions,
        vec![[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    );
}

#[test]
fn malformed_truncated_buffer_is_a_typed_error_not_a_panic() {
    let err = import_gltf(&fixture("malformed_truncated.gltf"), None).unwrap_err();
    assert!(
        matches!(
            err,
            GltfError::BufferLength {
                expected: 42,
                actual: 36,
                ..
            }
        ),
        "{err:?}"
    );
}
