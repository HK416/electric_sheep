//! Packet M17/G9 (`docs/packets/M17/plan-g.md`): an editable project's documents copied whole
//! ("save as template" and a project made from one), and whether a saved scene maps onto
//! `MuJoCo` Warp, which ②'s teacher trains on. The end-to-end oracles are
//! `crates/es-editor/tests/authored.rs`.

use std::path::{Path, PathBuf};

use es_editor_scene::check::maps_onto;
use es_editor_scene::copy::documents;
use es_editor_scene::{make_editable, BackendKind, SCENE_FILE, SPEC_FILE};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-g9-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// `assets/` goes whole — a glTF's buffer is in no asset list — then the specification, then
/// the scene document; a folder that already holds a scene is refused, and nothing is written.
#[test]
fn documents_are_copied_whole_and_never_over_a_scene() {
    let from = scratch("from");
    make_editable(
        &from,
        &repo().join("tests/fixtures/esscene/so101_pick_place.esscene"),
        None,
    )
    .unwrap();
    std::fs::write(from.join(SPEC_FILE), "kind = \"task-spec\"\n").unwrap();
    let deep = from.join("assets").join("abc").join("meshes");
    std::fs::create_dir_all(&deep).unwrap();
    std::fs::write(deep.join("arm.bin"), [1_u8, 2, 3]).unwrap();

    let to = scratch("to").join("nested");
    documents(&from, &to).unwrap();
    for rel in [SCENE_FILE, SPEC_FILE, "assets/abc/meshes/arm.bin"] {
        assert_eq!(
            std::fs::read(from.join(rel)).unwrap(),
            std::fs::read(to.join(rel)).unwrap(),
            "{rel}"
        );
    }
    let before = std::fs::read(to.join(SPEC_FILE)).unwrap();
    std::fs::write(from.join(SPEC_FILE), "changed").unwrap();
    assert!(documents(&from, &to)
        .unwrap_err()
        .contains("already editable"));
    assert_eq!(std::fs::read(to.join(SPEC_FILE)).unwrap(), before);
    for d in [from, to] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// The SO-101 copy maps onto `MuJoCo` Warp (its meshes are a warning there, not a block); a
/// project with no readable scene is refused in the reader's words.
#[test]
fn the_teacher_card_asks_mjwarp() {
    let root = scratch("warp");
    make_editable(
        &root,
        &repo().join("tests/fixtures/esscene/so101_pick_place.esscene"),
        None,
    )
    .unwrap();
    assert_eq!(maps_onto(&root, BackendKind::MjWarp), Ok(()));
    std::fs::remove_file(root.join(SCENE_FILE)).unwrap();
    let r = maps_onto(&root, BackendKind::MjWarp).unwrap_err();
    assert_eq!((r.field.as_str(), r.key), ("scene", "author.refused.other"));
    let _ = std::fs::remove_dir_all(root);
}
