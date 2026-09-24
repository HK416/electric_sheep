//! Oracle 1 of packet `docs/packets/M11/P-M11-R4-scene-hash-check.md`: **a run refuses a scene
//! its Task IR does not pin**.
//!
//! `scene_hash` is a hash-chain input (spec 5.3), so `Env::new` compares the loaded scene's
//! content hash with the Task IR's declared one before it loads anything. The refusal half
//! needs no backend; the half that builds the committed task needs `MuJoCo` and prints
//! `SKIP <test>: <why>` without it (spec 1.4).

use es_assets::scene::SceneDesc;
use es_env::scheduler::BatchDomains;
use es_env::Env;
use es_physics_backend::MuJoCoCpuBackend;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn scene_of(xml: &str) -> SceneDesc {
    es_assets::parse_mjcf(xml).expect("the scene parses").scene
}

fn short(d: &[u8; 32]) -> String {
    format!("{:02x}{:02x}{:02x}{:02x}", d[0], d[1], d[2], d[3])
}

#[test]
fn scene_hash_mismatch_is_refused() {
    let task = es_ir::serial::task_from_toml(&read("tests/fixtures/visible-learning/task.toml"))
        .expect("the demo task document parses");
    let xml = read("tests/fixtures/mjcf/so101_pick_place.xml");
    let committed = scene_of(&xml);
    assert_eq!(committed.scene_hash(), task.scene.scene_hash);

    // One damping value changed: a different physical condition, a different content hash.
    let edited_xml = xml.replacen("damping=\"0.60\"", "damping=\"0.61\"", 1);
    assert_ne!(edited_xml, xml, "the fixture carries damping=\"0.60\"");
    let edited = scene_of(&edited_xml);
    let domains = BatchDomains::single_env();
    let err = Env::new(&task, &edited, MuJoCoCpuBackend::new(), &domains, 0)
        .expect_err("an edited scene is refused")
        .to_string();
    for needle in [
        short(&task.scene.scene_hash),
        short(&edited.scene_hash()),
        task.scene.path.clone(),
    ] {
        assert!(err.contains(&needle), "`{err}` does not name {needle}");
    }

    if let Err(why) = MuJoCoCpuBackend::is_available() {
        println!("SKIP scene_hash_mismatch_is_refused (committed half): {why}");
        return;
    }
    Env::new(&task, &committed, MuJoCoCpuBackend::new(), &domains, 0)
        .expect("the committed scene builds");
}

/// Every committed Task IR under `tests/fixtures` pins the scene it names.
#[test]
fn every_committed_task_pins_its_scene() {
    let mut stack = vec![repo_root().join("tests/fixtures")];
    let mut checked = 0;
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("readable fixture dir") {
            let path = entry.expect("entry").path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if !(name.starts_with("task") && path.extension().is_some_and(|e| e == "toml")) {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("readable");
            let Ok(task) = es_ir::serial::task_from_toml(&text) else {
                continue;
            };
            let scene = scene_of(&read(&task.scene.path));
            assert!(
                scene.scene_hash() == task.scene.scene_hash,
                "{} declares scene_hash {}…, its scene {} hashes to {}…",
                path.display(),
                short(&task.scene.scene_hash),
                task.scene.path,
                short(&scene.scene_hash())
            );
            checked += 1;
        }
    }
    assert!(checked >= 14, "only {checked} committed Task IRs found");
}
