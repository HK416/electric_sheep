//! Plan H, packet HT1: `<texture>` and `<material>` become texels and drawn materials.
//!
//! * the Shadow Hand bundle's `block.png`, declared as the bundle declares it, decodes into
//!   `MuJoCo`'s six faces, and its asset hash is the texels' digest — not the path's;
//! * a material becomes a drawn one only by an explicit texture or PBR attribute, and only a
//!   drawn one enters `scene_hash` — so the committed scenes (the go1 fixture carries an
//!   `rgba`-only material) keep their digests under parse *and* under load;
//! * the classic `specular` / `shininess` pair maps to roughness by the one written formula.

// The defaults under test are exact (0, 1): `==` is the claim, not a shortcut.
#![allow(clippy::float_cmp, clippy::format_collect)]

use std::path::PathBuf;

use es_assets::scene::AssetKind;
use es_assets::texture::TexKind;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf")
}

fn hex(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn textured() -> es_assets::SceneDesc {
    let dir = fixtures().join("textured");
    let xml = std::fs::read_to_string(dir.join("textured.xml")).expect("the fixture");
    let mut scene = es_assets::parse_mjcf(&xml).expect("parses").scene;
    es_assets::mesh::load(&mut scene, &dir).expect("the textures decode");
    scene
}

/// The face order is `MuJoCo`'s (+X, -X, +Y, -Y, +Z, -Z = R L U D F B), each face cut out of
/// the grid cell `gridlayout` names: `.U..LFRB.D..` puts U at row 0 column 1, L F R B across
/// row 1 and D at row 2 column 1. The centre texel of each face is its letter's backdrop
/// colour, read off the PNG at that cell's corner.
#[test]
fn block_png_decodes_into_mujocos_six_faces() {
    let scene = textured();
    let (id, asset) = scene
        .assets
        .iter()
        .find(|a| a.name == "block")
        .map(|a| (a.id, a))
        .expect("the block texture");
    let data = scene.textures[&id].data.as_ref().expect("decoded");
    assert_eq!(
        (data.kind, data.width, data.height),
        (TexKind::Cube, 250, 250)
    );
    assert!(!data.srgb, "declared linear");
    assert_eq!(data.rgb.len(), 250 * 250 * 3 * 6);

    // The PNG itself, cell by cell, at a backdrop texel (5, 5) inside each cell.
    let png = std::fs::read(fixtures().join("textured/textures/block.png")).unwrap();
    let decoder = png::Decoder::new(std::io::Cursor::new(png));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let stride = info.line_size;
    let bpp = stride / info.width as usize;
    let at = |row: usize, col: usize| {
        let i = (row * 250 + 5) * stride + (col * 250 + 5) * bpp;
        [buf[i], buf[i + 1], buf[i + 2]]
    };
    for (face, (row, col)) in [(1, 2), (1, 0), (0, 1), (2, 1), (1, 1), (1, 3)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(data.texel(face as u32, 5, 5), at(row, col), "face {face}");
    }

    // Content, not path: the digest is the texels', and it is the digest `load` wrote.
    assert_eq!(asset.hash, data.content_hash());
    let path_digest =
        es_assets::scene::AssetRef::from_path(AssetKind::Texture, "block", &asset.path);
    assert_ne!(asset.hash, path_digest.hash);
    println!("block.png content hash {}", hex(&asset.hash));
}

/// The same texels under another name in another directory hash the same; loading twice is
/// loading once.
#[test]
fn a_texture_hashes_by_content() {
    let tmp = std::env::temp_dir().join(format!("es-ht1-{}", std::process::id()));
    std::fs::create_dir_all(tmp.join("elsewhere")).unwrap();
    std::fs::copy(
        fixtures().join("textured/textures/block.png"),
        tmp.join("elsewhere/renamed.png"),
    )
    .unwrap();
    let xml = r#"<mujoco><compiler texturedir="elsewhere"/><asset>
        <texture name="block" type="cube" colorspace="linear" file="renamed.png"
                 gridsize="3 4" gridlayout=".U..LFRB.D.."/>
        <material name="m" texture="block"/></asset>
        <worldbody><geom name="g" type="box" size="1 1 1" material="m"/></worldbody></mujoco>"#;
    let mut moved = es_assets::parse_mjcf(xml).unwrap().scene;
    es_assets::mesh::load(&mut moved, &tmp).unwrap();
    let digest = |s: &es_assets::SceneDesc| {
        s.assets
            .iter()
            .find(|a| a.kind == AssetKind::Texture)
            .unwrap()
            .hash
    };
    assert_eq!(digest(&moved), digest(&textured()));
    let before = moved.scene_hash();
    es_assets::mesh::load(&mut moved, &tmp).unwrap();
    assert_eq!(moved.scene_hash(), before);
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Spec 28.13 rule 2 applied to appearance: a material that only carries `rgba` (the go1
/// fixture's `dark`) is not a drawn material and moves nothing; the pins are
/// `scene_hash_pins.rs`'s, re-derived here under `load` too.
#[test]
fn an_rgba_only_material_is_not_drawn_and_moves_nothing() {
    for (name, pinned) in [
        (
            "go1_primitives.xml",
            "3c9348ea4be1fd36223cda47bac7b447c6dfccc831aaa7d26550e2f189c49a99",
        ),
        (
            "so101_pick_place.xml",
            "882e7d0b67d32fe8df9260d753d5e985a2bd641de557882bc4e654b5a03b5ba4",
        ),
    ] {
        let xml = std::fs::read_to_string(fixtures().join(name)).unwrap();
        let mut scene = es_assets::parse_mjcf(&xml).unwrap().scene;
        assert!(scene.materials.is_empty(), "{name} grew a drawn material");
        es_assets::mesh::load(&mut scene, &fixtures()).unwrap();
        assert_eq!(hex(&scene.scene_hash()), pinned, "{name}");
    }
    // A drawn material is in the hash: the same file with `roughness` written differs.
    let plain = r#"<mujoco><asset><material name="m" rgba="1 0 0 1"/></asset>
        <worldbody><geom name="g" type="box" size="1 1 1" material="m"/></worldbody></mujoco>"#;
    let drawn = plain.replace(r#"rgba="1 0 0 1""#, r#"rgba="1 0 0 1" roughness="0.4""#);
    let a = es_assets::parse_mjcf(plain).unwrap();
    let b = es_assets::parse_mjcf(&drawn).unwrap();
    assert!(a.scene.materials.is_empty());
    assert_eq!(b.scene.materials.len(), 1);
    assert_ne!(a.scene.scene_hash(), b.scene.scene_hash());
    // And the `rgba`-only material keeps its "not represented" warning, as before.
    assert!(a.warnings.iter().any(|w| w.message.contains("`rgba`")));
}

/// `roughness = (2 / (128 shininess + 2))^(1/4)`, only when written; `MuJoCo`'s -1 is unset.
#[test]
fn classic_pair_maps_to_roughness_by_one_formula() {
    let scene = textured();
    let mat = |name: &str| {
        let id = scene
            .assets
            .iter()
            .find(|a| a.name == name && a.kind == AssetKind::Material)
            .unwrap()
            .id;
        scene.materials[&id].clone()
    };
    let capsule = mat("capsule");
    let want = (2.0f64 / (128.0 * 0.3 + 2.0)).powf(0.25);
    assert!(
        (capsule.roughness() - want).abs() < 1e-12,
        "{}",
        capsule.roughness()
    );
    assert_eq!(capsule.metallic(), 0.0);
    let ball = mat("metal_ball");
    assert!((ball.roughness() - 0.3).abs() < 1e-12 && (ball.metallic() - 0.8).abs() < 1e-12);
    // A texture alone is drawn but writes no BRDF attribute.
    let block = mat("block");
    assert!(block.rgb.is_some() && block.specular.is_none() && block.shininess.is_none());
    // Only `specular` written: the default shininess 0.5 through the same formula.
    let xml = r#"<mujoco><asset><material name="m" specular="0" metallic="-1"/></asset>
        <worldbody><geom name="g" type="box" size="1 1 1" material="m"/></worldbody></mujoco>"#;
    let s = es_assets::parse_mjcf(xml).unwrap().scene;
    let m = s.materials.values().next().unwrap();
    assert!(m.metallic.is_none());
    assert!((m.roughness() - (2.0f64 / 66.0).powf(0.25)).abs() < 1e-12);
}

#[test]
fn a_missing_texture_or_a_bad_layer_is_named() {
    let xml = r#"<mujoco><asset><material name="m" texture="nope"/></asset>
        <worldbody><geom name="g" type="box" size="1 1 1" material="m"/></worldbody></mujoco>"#;
    let text = es_assets::parse_mjcf(xml).unwrap_err().to_string();
    assert!(text.contains("nope") && text.contains("texture"), "{text}");
    let xml = r#"<mujoco><asset>
        <texture name="t" type="2d" builtin="flat" width="2" height="2"/>
        <material name="m"><layer role="orm" texture="t"/><layer role="rgb" texture="t"/></material>
        </asset><worldbody><geom name="g" type="box" size="1 1 1" material="m"/></worldbody></mujoco>"#;
    let s = es_assets::parse_mjcf(xml).unwrap().scene;
    let m = s.materials.values().next().unwrap();
    assert!(m.orm.is_some() && m.rgb.is_some());
    assert!(
        m.metallic() == 1.0 && m.roughness() == 1.0,
        "a map's default factor is 1"
    );
    let xml = r#"<mujoco><asset>
        <texture name="t" type="2d" builtin="flat" width="2" height="2"/>
        <material name="m"><layer role="normal" texture="t"/></material>
        </asset><worldbody/></mujoco>"#;
    let text = es_assets::parse_mjcf(xml).unwrap_err().to_string();
    assert!(text.contains("normal"), "{text}");
    // A missing file is named at load, not at parse.
    let xml = r#"<mujoco><asset><texture name="t" type="2d" file="gone.png"/>
        <material name="m" texture="t"/></asset><worldbody/></mujoco>"#;
    let mut s = es_assets::parse_mjcf(xml).unwrap().scene;
    let text = es_assets::mesh::load(&mut s, &fixtures())
        .unwrap_err()
        .to_string();
    assert!(text.contains("gone.png"), "{text}");
}
