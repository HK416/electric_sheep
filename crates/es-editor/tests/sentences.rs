//! Packet G8 (`docs/packets/M17/plan-g.md`), oracle 1's words: the two committed task
//! specifications read as ①'s sentences, in English and in Korean, pinned in `sentences.txt`
//! beside this file (Korean may live only in documentation and string tables). The headless half
//! of the oracle (the sentences written back give the same specification) is
//! `crates/es-editor-scene/tests/sentences.rs`.

use std::path::{Path, PathBuf};

use es_assets::scene::SceneDesc;
use es_editor::model::i18n::Lang;
use es_editor::ui::sentence::words;
use es_editor_scene::sentence::{self as s, TaskSpec};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scene(rel: &str) -> SceneDesc {
    let path = repo().join(rel);
    let text = std::fs::read_to_string(&path).unwrap();
    let dir = path.parent().unwrap();
    if rel.ends_with(".esscene") {
        let doc = es_assets::esscene::EsScene::from_toml(&text).unwrap();
        es_assets::esscene::expand(&doc, dir).unwrap()
    } else {
        let mut d = es_assets::parse_mjcf(&text).unwrap().scene;
        es_assets::mesh::load(&mut d, dir).unwrap();
        d
    }
}

/// Every sentence of `spec`: success and failure, each under its header (packet M18/K7: an
/// absent hold reads 0 s), the time limit, the start.
fn sentences(lang: Lang, scene: &SceneDesc, spec: &TaskSpec) -> Vec<String> {
    let mut out = Vec::new();
    let success: Vec<_> = spec.success.clauses.iter().collect();
    let failure: Vec<_> = spec.failure.iter().flat_map(|f| &f.clauses).collect();
    for (header, clauses) in [(false, success), (true, failure)] {
        out.push(words(lang, &s::section(spec, header)));
        out.extend(clauses.iter().map(|c| words(lang, &s::clause(scene, c))));
    }
    out.push(words(lang, &s::timeout(spec)));
    let items = spec.start.iter().flat_map(|st| &st.items);
    out.extend(items.map(|i| words(lang, &s::start_item(scene, i))));
    out
}

#[test]
fn the_committed_specifications_read_as_the_pinned_sentences() {
    let scenes = [
        (
            "tests/fixtures/estask/shadow_hand_repose.estask",
            "tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.esscene",
        ),
        (
            "tests/fixtures/estask/so101_views.estask",
            "tests/fixtures/mjcf/so101_pick_place_views.xml",
        ),
    ];
    let pinned =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/sentences.txt"))
            .unwrap()
            .replace('\r', "");
    let mut blocks = 0;
    for block in pinned.split("\n[").skip(1) {
        let (head, body) = block.split_once("]\n").expect("a block's header");
        let (spec_rel, code) = head
            .rsplit_once(' ')
            .expect("a specification and a language");
        let lang = Lang::from_code(code);
        assert_eq!(lang.code(), code, "{head}");
        let scene_rel = scenes
            .iter()
            .find(|(s, _)| *s == spec_rel)
            .expect(spec_rel)
            .1;
        let text = std::fs::read_to_string(repo().join(spec_rel)).unwrap();
        let spec = TaskSpec::from_toml(&text).unwrap();
        let want: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(sentences(lang, &scene(scene_rel), &spec), want, "{head}");
        blocks += 1;
    }
    assert_eq!(blocks, 4, "two specifications, two languages");
}
