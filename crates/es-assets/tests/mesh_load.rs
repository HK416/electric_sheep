//! Packet M10/W2a oracle 1: a mesh geom's file becomes vertices, and the hash chain follows
//! the *content* rather than the path (spec 5.3).
//!
//! The readers are hand-written (`es_assets::stl`, `es_assets::obj`) and take no host `libm`:
//! `str::parse::<f32>` and `f32::from_le_bytes` only (spec 3.4 / 3.2), so a mesh hashes to one
//! number on every platform. `committed_scenes_are_unmoved_by_load` is the other half of spec
//! 28.13 rule 2: a mesh is an addition, so a scene that has none must hash exactly as before.

use std::fmt::Write as _;
use std::path::PathBuf;

use es_assets::scene::{AssetKind, SceneDesc};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf")
}

fn read(name: &str) -> String {
    let path = fixtures().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn loaded(name: &str) -> SceneDesc {
    let mut scene = es_assets::parse_mjcf(&read(name))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .scene;
    es_assets::mesh::load(&mut scene, &fixtures()).unwrap_or_else(|e| panic!("{name}: {e}"));
    scene
}

// --- STL --------------------------------------------------------------------------------------

/// Two facets sharing an edge, written as the 84 + 50*n binary layout.
fn binary_stl(facets: &[([f32; 3], [[f32; 3]; 3])]) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    out.extend_from_slice(&(facets.len() as u32).to_le_bytes());
    for (normal, verts) in facets {
        for c in normal {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for v in verts {
            for c in v {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

const A: [f32; 3] = [0.0, 0.0, 0.0];
const B: [f32; 3] = [1.0, 0.0, 0.0];
const C: [f32; 3] = [1.0, 1.0, 0.0];
const D: [f32; 3] = [0.0, 1.0, 0.0];

#[test]
fn binary_stl_dedups_shared_vertices() {
    let up = [0.0, 0.0, 1.0];
    let bytes = binary_stl(&[(up, [A, B, C]), (up, [A, C, D])]);
    assert_eq!(bytes.len(), 84 + 50 * 2);
    let (positions, indices) = es_assets::stl::parse(&bytes).expect("two facets");
    // A and C are shared: four positions, not six.
    assert_eq!(positions, vec![A, B, C, D]);
    assert_eq!(indices, vec![0, 1, 2, 0, 2, 3]);
}

/// The triangle the indices actually describe, which is what the renderer and the convex hull
/// see — independent of the order deduplication happened to assign the positions.
fn triangles(decoded: &(Vec<[f32; 3]>, Vec<u32>)) -> Vec<[f32; 3]> {
    decoded
        .1
        .iter()
        .map(|i| decoded.0[*i as usize])
        .collect::<Vec<_>>()
}

#[test]
fn binary_stl_flips_a_facet_against_its_normal() {
    // The winding of (A, B, C) points at +Z; the stored normal says -Z, so `b` and `c` swap.
    let bytes = binary_stl(&[([0.0, 0.0, -1.0], [A, B, C])]);
    let decoded = es_assets::stl::parse(&bytes).expect("one facet");
    assert_eq!(triangles(&decoded), vec![A, C, B]);
    // A zero normal carries no opinion, so the winding stands as written.
    let bytes = binary_stl(&[([0.0, 0.0, 0.0], [A, B, C])]);
    let decoded = es_assets::stl::parse(&bytes).expect("one facet");
    assert_eq!(triangles(&decoded), vec![A, B, C]);
    // As does a normal that agrees with it.
    let bytes = binary_stl(&[([0.0, 0.0, 1.0], [A, B, C])]);
    assert_eq!(
        triangles(&es_assets::stl::parse(&bytes).unwrap()),
        [A, B, C]
    );
}

#[test]
fn ascii_stl_decodes() {
    let text = "solid two\n\
        facet normal 0 0 1\n outer loop\n\
          vertex 0 0 0\n vertex 1 0 0\n vertex 1 1 0\n\
        endloop\n endfacet\n\
        facet normal 0 0 1\n outer loop\n\
          vertex 0 0 0\n vertex 1 1 0\n vertex 0 1 0\n\
        endloop\n endfacet\n\
        endsolid two\n";
    let (positions, indices) = es_assets::stl::parse(text.as_bytes()).expect("ascii");
    assert_eq!(positions, vec![A, B, C, D]);
    assert_eq!(indices, vec![0, 1, 2, 0, 2, 3]);
}

// --- OBJ --------------------------------------------------------------------------------------

#[test]
fn obj_negative_indices_and_quad_fans() {
    let text = "# a quad\n\
        v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
        f -4 -3 -2 -1\n";
    let (positions, indices) = es_assets::obj::parse(text).expect("quad");
    assert_eq!(positions, vec![A, B, C, D]);
    // Fan-triangulated around the first vertex.
    assert_eq!(indices, vec![0, 1, 2, 0, 2, 3]);
}

#[test]
fn obj_face_forms_slash_and_double_slash() {
    let text = "mtllib x.mtl\nusemtl m\no thing\ng group\ns off\n\
        v 0 0 0\nv 1 0 0\nv 1 1 0\n\
        vt 0 0\nvt 1 0\nvt 1 1\nvn 0 0 1\n\
        f 1 2 3\nf 1/1 2/2 3/3\nf 1//1 2//1 3//1\nf 1/1/1 2/2/1 3/3/1\n";
    let (positions, indices) = es_assets::obj::parse(text).expect("four forms");
    assert_eq!(positions.len(), 3);
    // All four faces are the same triangle; `vt` / `vn` / names are ignored, not an error.
    assert_eq!(indices, vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
}

// --- the fixture ------------------------------------------------------------------------------

/// The one number `tests/fixtures/mjcf/meshes/box.stl` hashes to, typed in on purpose: the
/// same constant is asserted on Windows and on the Linux oracle server.
const BOX_CONTENT_HASH: &str = "725969e3d81e21dc1750244fc03202a187a52984240cbf53529db6af01f64c27";

#[test]
fn box_stl_content_hash_is_pinned() {
    let scene = loaded("mesh_box.xml");
    let asset = scene
        .assets
        .iter()
        .find(|a| a.name == "box")
        .expect("the box mesh asset");
    assert_eq!(asset.kind, AssetKind::Mesh);
    let mesh = &scene.meshes[&asset.id];
    // The ±0.05 box: eight corners, twelve triangles.
    assert_eq!(mesh.positions.len(), 8);
    assert_eq!(mesh.indices.len(), 36);
    assert!(mesh.normals.is_none() && mesh.uvs.is_none() && mesh.material.is_none());
    assert_eq!(hex(&asset.hash), BOX_CONTENT_HASH);
}

#[test]
fn mesh_box_load_rewrites_asset_hash_and_scene_hash() {
    let before = es_assets::parse_mjcf(&read("mesh_box.xml")).unwrap().scene;
    let after = loaded("mesh_box.xml");
    let asset = |s: &SceneDesc| s.assets.iter().find(|a| a.name == "box").unwrap().hash;
    // Before the load the digest is the path's (spec 5.3 placeholder); after it is the
    // content's, and `scene_hash` follows because the `AssetRef` is hashed into it.
    assert_ne!(hex(&asset(&before)), hex(&asset(&after)));
    assert_ne!(hex(&before.scene_hash()), hex(&after.scene_hash()));
    assert_eq!(hex(&asset(&after)), BOX_CONTENT_HASH);
    assert_eq!(
        hex(&after.scene_hash()),
        "729c4a481fc8fe644e3cdd030379c2d19554d1452da3c16fc90029c6fa6af021"
    );
    // Loading twice is loading once: the meshes are already in the scene.
    let mut twice = after.clone();
    es_assets::mesh::load(&mut twice, &fixtures()).unwrap();
    assert_eq!(hex(&twice.scene_hash()), hex(&after.scene_hash()));
}

/// Spec 28.13 rule 2: a mesh is an addition. The committed primitives-only scenes have no
/// `AssetKind::Mesh` at all, so `load` is a no-op on them and their pins (packet W0b,
/// `scene_hash_pins.rs`) do not move.
#[test]
fn committed_scenes_are_unmoved_by_load() {
    for (name, pinned) in [
        (
            "so101_pick_place.xml",
            "882e7d0b67d32fe8df9260d753d5e985a2bd641de557882bc4e654b5a03b5ba4",
        ),
        (
            "go1_primitives.xml",
            "3c9348ea4be1fd36223cda47bac7b447c6dfccc831aaa7d26550e2f189c49a99",
        ),
    ] {
        let mut scene = es_assets::parse_mjcf(&read(name)).unwrap().scene;
        let before = hex(&scene.scene_hash());
        assert_eq!(before, pinned, "{name} moved before the load even ran");
        es_assets::mesh::load(&mut scene, &fixtures()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(scene.meshes.is_empty(), "{name} grew a mesh");
        assert_eq!(hex(&scene.scene_hash()), pinned, "{name} moved under load");
    }
}

// --- refusals ---------------------------------------------------------------------------------

/// `<mesh scale>` / `refpos` / `refquat` transform the vertices, and this packet bakes none of
/// them. A warning would be a silent geometry error once the mesh is drawn and collided with,
/// so the parse fails by name instead (spec 17.2's rule applied to the importer).
#[test]
fn scale_refpos_refquat_are_refused_by_name() {
    for (attr, value) in [
        ("scale", "0.001 0.001 0.001"),
        ("refpos", "0 0 0.1"),
        ("refquat", "0 1 0 0"),
    ] {
        let xml = format!(
            r#"<mujoco><asset><mesh name="m" file="m.stl" {attr}="{value}"/></asset>
               <worldbody><geom name="g" type="mesh" mesh="m"/></worldbody></mujoco>"#
        );
        let err = es_assets::parse_mjcf(&xml).expect_err("{attr} must be refused");
        let text = err.to_string();
        assert!(text.contains(attr), "{attr}: {text}");
        assert!(text.contains("not supported"), "{attr}: {text}");
    }
    // The defaults are not a refusal.
    let xml = r#"<mujoco><asset><mesh name="m" file="m.stl" scale="1 1 1" refpos="0 0 0"
           refquat="1 0 0 0"/></asset>
           <worldbody><geom name="g" type="mesh" mesh="m"/></worldbody></mujoco>"#;
    assert!(es_assets::parse_mjcf(xml).is_ok());
}

#[test]
fn missing_file_names_the_path() {
    let xml = r#"<mujoco><compiler meshdir="meshes"/>
           <asset><mesh name="nope" file="nope.stl"/></asset>
           <worldbody><geom name="g" type="mesh" mesh="nope"/></worldbody></mujoco>"#;
    let mut scene = es_assets::parse_mjcf(xml).unwrap().scene;
    let err = es_assets::mesh::load(&mut scene, &fixtures()).expect_err("no such file");
    let text = err.to_string();
    assert!(text.contains("nope.stl"), "{text}");
    assert!(text.contains("nope"), "{text}");

    // An extension with no reader is named too, rather than guessed at.
    let xml = r#"<mujoco><asset><mesh name="m" file="m.dae"/></asset>
           <worldbody><geom name="g" type="mesh" mesh="m"/></worldbody></mujoco>"#;
    let mut scene = es_assets::parse_mjcf(xml).unwrap().scene;
    let text = es_assets::mesh::load(&mut scene, &fixtures())
        .expect_err("no reader")
        .to_string();
    assert!(text.contains("m.dae"), "{text}");
}

#[test]
fn bad_index_is_an_error() {
    // OBJ: 1-based index past the end of `v`.
    let text = es_assets::obj::parse("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 4\n")
        .expect_err("index 4 of 3");
    assert!(text.contains('4'), "{text}");
    // OBJ: index 0 does not exist in a 1-based scheme.
    assert!(es_assets::obj::parse("v 0 0 0\nf 0 0 0\n").is_err());
    // STL: a buffer that is neither `84 + 50*n` nor parsable ASCII.
    assert!(es_assets::stl::parse(&[1, 2, 3]).is_err());
    assert!(es_assets::stl::parse(b"solid x\nfacet normal 0 0\nendsolid\n").is_err());
}

// --- the generator ----------------------------------------------------------------------------

/// The fixture box's half-extent, m.
const HALF: f32 = 0.05;

/// Writes `tests/fixtures/mjcf/meshes/box.stl`: the ±0.05 m box as 12 binary-STL facets,
/// 684 bytes (80 header + 4 count + 12 * 50). Committed, so this runs only on demand:
///
///     ES_GENERATE_GOLDENS=1 cargo test -p es-assets --test mesh_load -- --ignored
#[test]
#[ignore = "writes a committed fixture; set ES_GENERATE_GOLDENS=1"]
fn generate_mesh_box_stl() {
    assert_eq!(
        std::env::var("ES_GENERATE_GOLDENS").as_deref(),
        Ok("1"),
        "set ES_GENERATE_GOLDENS=1 to rewrite the fixture"
    );
    let corner = |i: usize| {
        [
            if i & 1 == 0 { -HALF } else { HALF },
            if i & 2 == 0 { -HALF } else { HALF },
            if i & 4 == 0 { -HALF } else { HALF },
        ]
    };
    // Per face: the outward normal and the four corner indices, counter-clockwise seen from
    // outside. Two triangles each, fanned from the first corner.
    let faces: [([f32; 3], [usize; 4]); 6] = [
        ([-1.0, 0.0, 0.0], [0, 2, 6, 4]),
        ([1.0, 0.0, 0.0], [1, 5, 7, 3]),
        ([0.0, -1.0, 0.0], [0, 4, 5, 1]),
        ([0.0, 1.0, 0.0], [2, 3, 7, 6]),
        ([0.0, 0.0, -1.0], [0, 1, 3, 2]),
        ([0.0, 0.0, 1.0], [4, 6, 7, 5]),
    ];
    let mut facets = Vec::new();
    for (normal, quad) in faces {
        facets.push((normal, [corner(quad[0]), corner(quad[1]), corner(quad[2])]));
        facets.push((normal, [corner(quad[0]), corner(quad[2]), corner(quad[3])]));
    }
    let mut bytes = binary_stl(&facets);
    // A binary header that does not begin with `solid`, so no reader mistakes it for ASCII.
    let header = b"es-assets mesh_box fixture: the 0.05 m half-extent box, 12 facets";
    bytes[..header.len()].copy_from_slice(header);
    assert_eq!(bytes.len(), 684);

    let path = fixtures().join("meshes").join("box.stl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    println!("wrote {} ({} B)", path.display(), bytes.len());
    // And print what the pins above must say.
    let mut scene = es_assets::parse_mjcf(&read("mesh_box.xml")).unwrap().scene;
    es_assets::mesh::load(&mut scene, &fixtures()).unwrap();
    let asset = scene.assets.iter().find(|a| a.name == "box").unwrap();
    println!("BOX_CONTENT_HASH {}", hex(&asset.hash));
    println!("mesh_box scene_hash {}", hex(&scene.scene_hash()));
}
