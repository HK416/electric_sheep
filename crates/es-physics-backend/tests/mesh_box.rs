//! Packet M10/W2a oracle 4: the mesh box and the primitive box are the same box.
//!
//! `tests/fixtures/mjcf/mesh_box.xml` holds two free bodies of identical geometry — one a
//! `type="mesh"` geom whose vertices come from `meshes/box.stl` through
//! `es_assets::mesh::load`, one a `type="box"` primitive — at the same density. `MuJoCo` is the
//! reference (spec 1.4, spec 17.1): if the inline `<asset><mesh vertex face>` the emitter
//! writes is the box we think it is, then `MuJoCo`'s own compiler must derive the same mass and
//! inertia for both and drop them to the same rest height.
//!
//! Needs a Python interpreter with `mujoco`; without one this prints `SKIP mesh_box: <why>`.
//!
//!     ES_PYTHON=$HOME/venvs/es/bin/python cargo test -p es-physics-backend \
//!       --test mesh_box -- --nocapture

// Exactness is the property under test where `==` appears: MuJoCo's counts are integers that
// cross as `f64`, and `legacy` against `exact` has to agree bit for bit or not at all. Every
// comparison that is a measurement rather than a count goes through `rel` and a tolerance.
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use es_assets::scene::SceneDesc;
use es_physics_backend::{scene_to_mjcf, MuJoCoCpuBackend};
use es_physics_core::{LoadConfig, PhysicsBackend};

const FIXTURE: &str = "mesh_box.xml";
/// Both bodies start here and must land on the plane.
const REST_Z: f64 = 0.05;

/// What the Python side reports about the compiled model and about 2,000 steps of it. Flat
/// `key value` lines rather than JSON: this crate's test target needs no parser for that.
const SCRIPT: &str = r#"
import sys
import mujoco

xml = sys.stdin.read()
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
# The same file with the mesh's inertia algorithm switched off its `legacy` default, to say
# whether `legacy` and `exact` agree on a convex closed box (they are the only difference).
exact = mujoco.MjModel.from_xml_string(xml.replace("<mesh ", '<mesh inertia="exact" '))
out = []
out.append(("nbody", model.nbody))
out.append(("nmesh", model.nmesh))
out.append(("vertnum", model.mesh_vertnum[0]))
out.append(("facenum", model.mesh_facenum[0]))
bodies = ("mesh_box", "prim_box")
ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in bodies}
for n in bodies:
    out.append(("mass.%s" % n, model.body_mass[ids[n]]))
    for k in range(3):
        out.append(("inertia.%s.%d" % (n, k), model.body_inertia[ids[n]][k]))
out.append(("mass.exact", exact.body_mass[ids["mesh_box"]]))
for k in range(3):
    out.append(("inertia.exact.%d" % k, exact.body_inertia[ids["mesh_box"]][k]))
for _ in range(2000):
    mujoco.mj_step(model, data)
for n in bodies:
    out.append(("z.%s" % n, data.xpos[ids[n]][2]))
out.append(("warnings", sum(int(w.number) for w in data.warning)))
sys.stdout.write("".join("%s %r\n" % (k, float(v)) for k, v in out))
"#;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf")
}

/// The fixture with its STL resolved: the emitter has no path channel, so the vertices must be
/// on the scene before `scene_to_mjcf` sees it (INV-17).
fn scene() -> SceneDesc {
    let path = fixtures_dir().join(FIXTURE);
    let xml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let import = es_assets::parse_mjcf(&xml).unwrap_or_else(|e| panic!("{FIXTURE}: {e}"));
    let mut scene = import.scene;
    es_assets::mesh::load(&mut scene, &fixtures_dir()).expect("meshes/box.stl loads");
    scene
}

/// `ES_PYTHON`, else the usual two names — the search `es_physics_backend::proc` runs, repeated
/// here because this test drives `mujoco` directly rather than through the adapter protocol.
fn pythons() -> Vec<String> {
    match std::env::var("ES_PYTHON") {
        Ok(path) if !path.trim().is_empty() => vec![path],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    }
}

/// Runs [`SCRIPT`] over the emitted MJCF, or the reason it could not.
fn facts() -> Option<BTreeMap<String, f64>> {
    if let Err(reason) = MuJoCoCpuBackend::is_available() {
        println!("SKIP mesh_box: {reason}");
        return None;
    }
    let mjcf = scene_to_mjcf(&scene()).expect("the loaded mesh scene emits");
    let python = pythons().remove(0);
    let mut child = Command::new(&python)
        .args(["-c", SCRIPT])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("`{python}`: {e}"));
    child
        .stdin
        .take()
        .expect("piped")
        .write_all(mjcf.as_bytes())
        .expect("the MJCF reaches the interpreter");
    let out = child.wait_with_output().expect("the interpreter exits");
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_owned();
    assert!(
        out.status.success(),
        "`{python}` exited with {}: {stderr}",
        out.status
    );
    // A `MuJoCo` compile warning is printed here and nowhere else, so an empty stderr is the
    // assertion that the emitted file compiles cleanly.
    assert!(stderr.is_empty(), "MuJoCo wrote to stderr: {stderr}");
    let text = String::from_utf8(out.stdout).expect("UTF-8");
    println!("RAN mesh_box\n{text}");
    Some(
        text.lines()
            .filter_map(|line| line.split_once(' '))
            .map(|(k, v)| (k.to_owned(), v.parse::<f64>().expect("a number")))
            .collect(),
    )
}

/// `|a - b| / |b|`, the relative error the tolerances below are stated in.
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn mesh_box_loads_in_mujoco() {
    let Some(f) = facts() else { return };
    // world + the two free bodies.
    assert_eq!(f["nbody"], 3.0);
    assert_eq!(f["nmesh"], 1.0);
    // The eight corners and twelve triangles the STL carries, unchanged by the round trip
    // through `<mesh vertex face>` -- MuJoCo did not have to invent a hull.
    assert_eq!(f["vertnum"], 8.0);
    assert_eq!(f["facenum"], 12.0);

    // And the adapter's own load agrees about the body count.
    let mut backend = MuJoCoCpuBackend::new();
    let info = backend
        .load(&scene(), &LoadConfig::default())
        .expect("the mesh scene loads through the adapter");
    assert_eq!(info.nbody, 3);
    // Two free joints: seven coordinates and six dofs each.
    assert_eq!((info.nq, info.nv), (14, 12));
}

#[test]
fn mesh_box_rests_like_the_primitive_box() {
    let Some(f) = facts() else { return };
    let (mesh, prim) = (f["z.mesh_box"], f["z.prim_box"]);
    assert!(mesh.is_finite() && prim.is_finite(), "{mesh} {prim}");
    assert!(
        (mesh - prim).abs() < 1e-3,
        "the convex hull of the mesh rests at {mesh} m, the primitive at {prim} m"
    );
    for (what, z) in [("mesh", mesh), ("prim", prim)] {
        assert!(
            (z - REST_Z).abs() < 2e-3,
            "the {what} box rests at {z} m, not {REST_Z} m"
        );
    }
    assert_eq!(f["warnings"], 0.0, "MuJoCo raised a runtime warning");
}

/// MuJoCo derives mass and inertia from `density` and the geom's volume, so the two bodies
/// agree exactly as far as their volumes do -- and their volumes differ by one thing only.
///
/// `MESH_F32_GAP` is that thing: the primitive's half-extent is the `f64` 0.05 the MJCF says,
/// while the mesh's is the `f32` nearest it, `0.05000000074505806`. A volume is three of those,
/// so the mesh is `3 * 1.49e-8 = 4.47e-8` heavier in relative terms, and its diagonal inertia
/// (mass times length squared) is off by the same order. That is the floor a `f32` vertex
/// buffer puts under this comparison, not slack: the packet's 1e-9 would have demanded `f64`
/// vertices, which is neither what STL carries nor what `MeshData` stores.
const MESH_F32_GAP: f64 = 1e-7;

#[test]
fn mesh_box_mass_and_inertia_match_the_primitive() {
    let Some(f) = facts() else { return };
    let mass = rel(f["mass.mesh_box"], f["mass.prim_box"]);
    assert!(
        mass < MESH_F32_GAP,
        "mass {} vs {} (relative {mass:e})",
        f["mass.mesh_box"],
        f["mass.prim_box"]
    );
    for k in 0..3 {
        let a = f[&format!("inertia.mesh_box.{k}")];
        let b = f[&format!("inertia.prim_box.{k}")];
        let error = rel(a, b);
        assert!(error < 1e-6, "inertia[{k}] {a} vs {b} (relative {error:e})");
    }
}

/// The packet left this unverified: does MuJoCo 3.13's default `<mesh inertia="legacy">` agree
/// with `"exact"` on a convex closed box? Measured here rather than assumed -- `legacy`
/// overcounts the volume of a *concave* mesh, and the box gives it nothing to overcount.
#[test]
fn legacy_and_exact_inertia_agree_on_a_convex_closed_box() {
    let Some(f) = facts() else { return };
    assert_eq!(
        f["mass.exact"], f["mass.mesh_box"],
        "legacy and exact disagree about the box's mass"
    );
    for k in 0..3 {
        assert_eq!(
            f[&format!("inertia.exact.{k}")],
            f[&format!("inertia.mesh_box.{k}")],
            "legacy and exact disagree about inertia[{k}]"
        );
    }
}
