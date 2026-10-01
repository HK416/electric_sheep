//! `es scene export` (plan G, packet G2): a scene document exports to an MJCF file whose
//! re-read scene hashes as the original, with its meshes and textures beside it.

use std::path::PathBuf;
use std::process::Command;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn scratch() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("es-scene-export-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_scene_document_exports_to_mjcf_that_reads_back_with_its_hash() {
    let dir = scratch();
    let out = dir.join("hand.xml");
    let doc = fixtures().join("mjcf/shadow_hand/shadow_hand_repose.esscene");
    let run = Command::new(env!("CARGO_BIN_EXE_es"))
        .args(["scene", "export"])
        .arg(&doc)
        .arg("--mjcf")
        .arg(&out)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&run.stderr)
    );
    // G1's measured digest of the Shadow Hand scene, MJCF and document alike.
    let hash = "a900841d2a2bfb8f7faed123bd4f8f4bc7bce9a61d82f06f4e4a54126004cc87";
    assert!(stdout.contains(&format!("scene_hash: {hash}")), "{stdout}");
    assert!(dir.join("meshes/palm.stl").is_file() && dir.join("textures/block.png").is_file());

    let xml = std::fs::read_to_string(&out).unwrap();
    let mut again = es_assets::parse_mjcf(&xml).unwrap().scene;
    es_assets::mesh::load(&mut again, &dir).unwrap();
    let hex: String = again
        .scene_hash()
        .iter()
        .fold(String::new(), |s, b| s + &format!("{b:02x}"));
    assert_eq!(hex, hash);
}

#[test]
fn export_without_an_output_is_a_usage_error() {
    let run = Command::new(env!("CARGO_BIN_EXE_es"))
        .args(["scene", "export"])
        .arg(fixtures().join("mjcf/pendulum.xml"))
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&run.stderr).contains("--mjcf <out.xml>"));
}
