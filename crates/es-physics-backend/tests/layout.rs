//! Packet M17/R8 (design note `scene-authoring.md` section 4.8): `layout(scene)`, the `qpos` /
//! `qvel` addresses and body rows `MuJoCo` gives a scene, computed without loading it, equals
//! what `mujoco-cpu` reports on every committed scene that loads there, and on GV's authored
//! scene when the owner's copy is on this machine (read only, never written).
//!
//! Without a Python interpreter carrying `mujoco` it prints `SKIP layout: <why>`.
//!
//!     ES_PYTHON=.venv/Scripts/python.exe cargo test -p es-physics-backend --test layout

use std::path::{Path, PathBuf};

use es_assets::scene::SceneDesc;
use es_core::TickRate;
use es_physics_backend::{layout, MuJoCoCpuBackend};
use es_physics_core::{LoadConfig, PhysicsBackend};

/// GV's project (review M17), the owner's; skipped where it is not.
const GV: &str = "C:/Users/User/Documents/Electric Sheep/push-box/scene.esscene";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every scene file in the repository: `git ls-files` would need git, so the folders that hold
/// them, walked.
fn committed() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p
                .extension()
                .is_some_and(|e| e == "xml" || e == "esscene" || e == "urdf")
            {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    walk(&repo().join("tests/fixtures"), &mut out);
    walk(&repo().join("python/es/rl_source"), &mut out);
    out.sort();
    out
}

/// The scene by extension, as `es_tools::backend::load_scene` reads it; `None` for a file that is
/// no scene (a parser's refusal fixture).
fn load(path: &Path) -> Option<SceneDesc> {
    let text = std::fs::read_to_string(path).ok()?;
    let dir = path.parent()?;
    match path.extension()?.to_str()? {
        "esscene" => {
            let doc = es_assets::esscene::EsScene::from_toml(&text).ok()?;
            es_assets::esscene::expand(&doc, dir).ok()
        }
        "urdf" => {
            let resolver = es_assets::urdf::PackageResolver::from_env();
            Some(es_assets::urdf::parse_urdf(&text, &resolver).ok()?.scene)
        }
        _ => {
            let mut scene = es_assets::parse_mjcf(&text).ok()?.scene;
            es_assets::mesh::load(&mut scene, dir).ok()?;
            Some(scene)
        }
    }
}

#[test]
fn the_layout_is_mujocos_on_every_committed_scene() {
    if let Err(reason) = MuJoCoCpuBackend::is_available() {
        println!("SKIP layout: {reason}");
        return;
    }
    let mut paths = committed();
    if Path::new(GV).is_file() {
        paths.push(PathBuf::from(GV));
    } else {
        println!("skipped: {GV} is not on this machine");
    }
    let mut compared = 0;
    for path in &paths {
        let name = path.display();
        let Some(scene) = load(path) else {
            println!("not a scene: {name}");
            continue;
        };
        let cfg = LoadConfig {
            n_envs: 1,
            rate: Some(TickRate::hz(1000)),
            seed: 1,
        };
        let info = match MuJoCoCpuBackend::new().load(&scene, &cfg) {
            Ok(info) => info,
            Err(e) => {
                println!("does not load on mujoco-cpu: {name}: {e}");
                continue;
            }
        };
        let ours = layout(&scene);
        assert_eq!(
            (ours.nq, ours.nv, ours.nbody),
            (info.nq, info.nv, info.nbody),
            "{name}"
        );
        assert_eq!(ours.qpos, info.qpos, "{name}: qpos");
        assert_eq!(ours.dof, info.dof, "{name}: dof");
        assert_eq!(ours.body, info.body, "{name}: body rows");
        println!(
            "equal: {name} (nq {}, nv {}, nbody {})",
            info.nq, info.nv, info.nbody
        );
        compared += 1;
    }
    assert!(compared >= 10, "only {compared} scenes compared");
    println!("RAN layout: {compared} scenes");
}
