//! Packet M10/W2a oracle 5: the real SO-101, meshes and all, through the whole pipeline.
//!
//! `mesh_box` proves one hand-made box. This proves the robot: upstream
//! `robotstudio_so101/so101.xml` and its 19 binary STL files at the commit
//! `tests/fixtures/mjcf/so101_pick_place.PROVENANCE.json` pins, parsed by `es-assets`, resolved
//! by `es_assets::mesh::load`, emitted inline by `scene_to_mjcf`, and compiled by `MuJoCo`
//! beside a direct `from_xml_path` of the upstream file. Every body's mass must agree: the
//! bodies with an explicit `<inertial>` trivially, and `camera_mount`, which has none, only if
//! the vertices that crossed as `<mesh vertex face>` really are the ones on disk.
//!
//! `#[ignore]`: 17 MB over the network and an 11 MB inline MJCF. Nothing is vendored -- the
//! files land in `target/menagerie/<commit>/`.
//!
//!     ES_PYTHON=$HOME/venvs/es/bin/python cargo test -p es-physics-backend \
//!       --test menagerie_meshes -- --ignored --nocapture

// A mass of exactly zero is a structural fact -- MuJoCo's world body -- not a measurement;
// every mass that is a measurement is compared through a relative error below.
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use es_assets::scene::AssetKind;
use es_physics_backend::{scene_to_mjcf, MuJoCoCpuBackend};

const MANIFEST: &str = "so101_pick_place.PROVENANCE.json";
const DIR: &str = "robotstudio_so101";
/// The camera mount is the body whose mass `MuJoCo` derives from a mesh: it has no
/// `<inertial>`, so its 12 g is the density times the hull volume of the vertices we sent.
const CAMERA_MOUNT_KG: f64 = 0.012;

const SCRIPT: &str = r#"
import sys
import mujoco

ours = mujoco.MjModel.from_xml_path(sys.argv[1])
theirs = mujoco.MjModel.from_xml_path(sys.argv[2])
print("nbody %d %d" % (ours.nbody, theirs.nbody))
print("nmesh %d %d" % (ours.nmesh, theirs.nmesh))
for i in range(ours.nbody):
    name = mujoco.mj_id2name(ours, mujoco.mjtObj.mjOBJ_BODY, i)
    j = mujoco.mj_name2id(theirs, mujoco.mjtObj.mjOBJ_BODY, name)
    if j < 0:
        raise SystemExit("upstream has no body %r" % name)
    print("mass %s %r %r" % (name, float(ours.body_mass[i]), float(theirs.body_mass[j])))
"#;

fn manifest_field(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mjcf")
        .join(MANIFEST);
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let after = json
        .split_once(&format!("\"{name}\""))
        .unwrap_or_else(|| panic!("{MANIFEST}: no field \"{name}\""))
        .1;
    let after = after.split_once(':').expect("a field has a value").1;
    let start = after.find('"').expect("a string value");
    let rest = &after[start + 1..];
    rest[..rest.find('"').expect("a terminated string")].to_owned()
}

fn menagerie(commit: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/menagerie")
        .join(commit)
}

/// Upstream bytes of `repo_path`, cached in the real directory layout so `meshdir` resolves.
fn fetch(commit: &str, repo_path: &str) -> Result<Vec<u8>, String> {
    let name = repo_path.rsplit('/').next().unwrap_or(repo_path);
    if let Ok(root) = std::env::var("ES_MENAGERIE_CACHE") {
        let cached = PathBuf::from(root).join(commit).join(repo_path);
        if cached.is_file() {
            return std::fs::read(&cached).map_err(|e| format!("{}: {e}", cached.display()));
        }
    }
    let out = menagerie(commit).join(repo_path);
    let out_dir = out.parent().expect("a file has a parent").to_owned();
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    if out.is_file() {
        return std::fs::read(&out).map_err(|e| format!("{}: {e}", out.display()));
    }
    let url = format!(
        "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{repo_path}"
    );
    let tmp = out_dir.join(format!("{name}.{:?}.part", std::thread::current().id()));
    let status = Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(&tmp)
        .arg(&url)
        .status()
        .map_err(|e| format!("no cache (set ES_MENAGERIE_CACHE) and `curl` failed: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("`curl {url}` exited with {status}"));
    }
    let bytes = std::fs::read(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let _ = std::fs::rename(&tmp, &out);
    let _ = std::fs::remove_file(&tmp);
    Ok(bytes)
}

#[test]
#[ignore = "fetches so101.xml and 19 STL files (17 MB) from GitHub and needs MuJoCo"]
fn so101_upstream_loads_in_mujoco_with_meshes() {
    if let Err(reason) = MuJoCoCpuBackend::is_available() {
        println!("SKIP menagerie_meshes: {reason}");
        return;
    }
    let commit = manifest_field("commit");
    let model = format!("{DIR}/so101.xml");
    let bytes = match fetch(&commit, &model) {
        Ok(bytes) => bytes,
        Err(why) => {
            println!("SKIP menagerie_meshes: {why}");
            return;
        }
    };
    let xml = String::from_utf8(bytes).expect("UTF-8");
    let mut scene = es_assets::parse_mjcf(&xml)
        .expect("upstream so101.xml parses")
        .scene;

    // The meshes the model itself names, fetched into the layout `meshdir` expects.
    let meshes: Vec<String> = scene
        .assets
        .iter()
        .filter(|a| a.kind == AssetKind::Mesh)
        .map(|a| a.path.clone())
        .collect();
    assert_eq!(meshes.len(), 18, "so101.xml declares 18 meshes");
    for path in &meshes {
        if let Err(why) = fetch(&commit, &format!("{DIR}/{path}")) {
            println!("SKIP menagerie_meshes: {why}");
            return;
        }
    }
    let base = menagerie(&commit).join(DIR);
    es_assets::mesh::load(&mut scene, &base).expect("every upstream STL decodes");
    assert_eq!(scene.meshes.len(), 18);
    // The whole bundle's identity in one number: 18 mesh content digests inside the
    // `AssetRef`s inside `scene_hash` (spec 5.3). `so101_provenance` owns the per-file blake3
    // pins against the manifest -- this one says the 18 arrived here intact and, being the
    // W0b platform-stable encoding, says the Linux run decoded them identically.
    let digest = scene.scene_hash().iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    });
    assert_eq!(
        digest, "5ac94cfd352b37a63706f0eaaf5ce9b3fdc35d7043fe13048821f865d125faa8",
        "the resolved upstream SO-101 does not hash to the pinned number"
    );

    let mjcf = scene_to_mjcf(&scene).expect("the upstream scene emits");
    let emitted = menagerie(&commit).join("so101_emitted.xml");
    std::fs::write(&emitted, &mjcf).expect("the emitted MJCF is written");
    println!(
        "emitted {} ({} MB)",
        emitted.display(),
        mjcf.len() / 1_000_000
    );

    let python = match std::env::var("ES_PYTHON") {
        Ok(path) if !path.trim().is_empty() => path,
        _ => "python".to_owned(),
    };
    let out = Command::new(&python)
        .args(["-c", SCRIPT])
        .arg(&emitted)
        .arg(base.join("so101.xml"))
        .output()
        .unwrap_or_else(|e| panic!("`{python}`: {e}"));
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_owned();
    assert!(out.status.success(), "`{python}`: {stderr}");
    let text = String::from_utf8(out.stdout).expect("UTF-8");
    println!("RAN menagerie_meshes\n{text}");

    let mut counts = BTreeMap::new();
    let mut worst: (String, f64) = (String::new(), 0.0);
    let mut camera_mount = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            [key @ ("nbody" | "nmesh"), ours, theirs] => {
                counts.insert((*key).to_owned(), (ours.to_owned(), theirs.to_owned()));
            }
            ["mass", name, ours, theirs] => {
                let ours: f64 = ours.parse().expect("a number");
                let theirs: f64 = theirs.parse().expect("a number");
                if *name == "camera_mount" {
                    camera_mount = Some(ours);
                }
                // A mass of exactly zero is structural, not a measurement: MuJoCo's world body
                // has one and nothing else may. Every other body is compared relatively.
                assert_eq!(
                    ours == 0.0,
                    theirs == 0.0,
                    "{name}: one side is massless and the other is not ({ours} vs {theirs})"
                );
                let error = if theirs == 0.0 {
                    0.0
                } else {
                    (ours - theirs).abs() / theirs.abs()
                };
                if error > worst.1 {
                    worst = ((*name).to_owned(), error);
                }
            }
            _ => {}
        }
    }
    assert_eq!(counts["nbody"].0, counts["nbody"].1, "body counts differ");
    assert_eq!(counts["nmesh"].0, counts["nmesh"].1, "mesh counts differ");
    assert!(
        worst.1 < 1e-9,
        "body `{}` is the worst mass disagreement at {:e} relative",
        worst.0,
        worst.1
    );
    let mount = camera_mount.expect("the model has a `camera_mount` body");
    assert!(
        (mount - CAMERA_MOUNT_KG).abs() < 5e-5,
        "the mesh-derived camera_mount mass is {mount} kg, not {CAMERA_MOUNT_KG} kg"
    );
    println!("worst mass disagreement: `{}` at {:e}", worst.0, worst.1);
}
