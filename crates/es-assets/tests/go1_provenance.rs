//! The Go1 locomotion scene is judged against upstream, not against review (spec 1.4).
//!
//! `tests/fixtures/mjcf/go1_primitives.xml` is a single-file, primitives-only derivative of
//! **two** files of `google-deepmind/mujoco_playground` at the pinned commit: the robot
//! (`go1/xmls/go1_mjx_feetonly.xml`) and the flat-terrain scene that includes it
//! (`go1/xmls/scene_mjx_feetonly_flat_terrain.xml`), which is where the floor geom and the
//! `home` keyframe live. This test obtains both at the pin, verifies their blake3 **before
//! parsing them** (spec 25.1: bytes off the network are untrusted), and proves that the
//! derivative changed nothing but the enumerated substitutions.
//!
//! Upstream comes from `$ES_PLAYGROUND_CACHE/<commit>/<file>` when that exists, else from
//! `curl -fsSL` into `target/`. No cache and no network prints `SKIP go1_provenance: <why>`;
//! having run, it prints `RAN go1_provenance`. A blake3 **mismatch is a failure, never a
//! skip**: it means upstream moved under the pin.
//!
//!     cargo test -p es-assets --test go1_provenance -- --nocapture
//!
//! What is *not* checked here: that `MuJoCo` steps the two files the same way. That is
//! `crates/es-physics-backend/tests/go1_step.rs`, which needs a Python with `mujoco`.

// Equality with upstream is the property under test; a tolerance here would be the bug.
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use es_assets::scene::{Body, Geom, Joint, SceneDesc, Shape};
use es_assets::Import;

const FIXTURE: &str = "go1_primitives.xml";
const MANIFEST: &str = "go1_primitives.PROVENANCE.json";

/// The robot file, and the scene file the floor and the `home` keyframe come from.
const ROBOT: &str = "go1_mjx_feetonly.xml";
const SCENE: &str = "scene_mjx_feetonly_flat_terrain.xml";

/// The body tree, root first: trunk plus hip/thigh/calf per leg.
const LEGS: [&str; 4] = ["FR", "FL", "RR", "RL"];

/// The twelve actuated joints, in upstream declaration order.
const JOINTS: [&str; 12] = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
];

/// `home`, the keyframe `Go1Env` resets to and whose `qpos[7..]` is the action zero-point
/// (`docs/api-notes/mujoco-playground-quadruped.md` section 1). Transcribed here *and*
/// checked against upstream's own bytes by [`derivative_carries_the_home_keyframe`].
const HOME_QPOS: [f64; 19] = [
    0.0, 0.0, 0.278, 1.0, 0.0, 0.0, 0.0, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1,
    0.9, -1.8,
];

fn bodies() -> Vec<String> {
    let mut out = vec!["trunk".to_owned()];
    for leg in LEGS {
        for part in ["hip", "thigh", "calf"] {
            out.push(format!("{leg}_{part}"));
        }
    }
    out
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf")
}

fn read_fixture(name: &str) -> String {
    let path = fixtures_dir().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn parse(xml: &str, what: &str) -> Import {
    es_assets::parse_mjcf(xml).unwrap_or_else(|e| panic!("{what}: {e}"))
}

fn derivative() -> Import {
    parse(&read_fixture(FIXTURE), FIXTURE)
}

// --- the manifest -----------------------------------------------------------------------------

/// The one string field `name` holds, read without a JSON dependency: this crate has none and
/// the packet adds none (same reader as `so101_provenance.rs`).
fn manifest_field(json: &str, name: &str) -> String {
    let key = format!("\"{name}\"");
    let after = json
        .split_once(&key)
        .unwrap_or_else(|| panic!("{MANIFEST}: no field \"{name}\""))
        .1;
    let after = after
        .split_once(':')
        .unwrap_or_else(|| panic!("{MANIFEST}: field \"{name}\" has no value"))
        .1;
    let start = after
        .find('"')
        .unwrap_or_else(|| panic!("{MANIFEST}: field \"{name}\" is not a string"));
    let rest = &after[start + 1..];
    let end = rest
        .find('"')
        .unwrap_or_else(|| panic!("{MANIFEST}: field \"{name}\" is unterminated"));
    rest[..end].to_owned()
}

/// `(commit, blake3 of the robot xml, blake3 of the scene xml)`.
fn pin() -> (String, String, String) {
    let json = read_fixture(MANIFEST);
    assert_eq!(manifest_field(&json, "license"), "BSD-3-Clause");
    assert_eq!(manifest_field(&json, "menagerie_commit").len(), 40);
    (
        manifest_field(&json, "commit"),
        manifest_field(&json, "blake3_go1_mjx_feetonly_xml"),
        manifest_field(&json, "blake3_scene_flat_terrain_xml"),
    )
}

// --- upstream ---------------------------------------------------------------------------------

/// Upstream bytes for one of the two files, or the reason there are none. Never writes into
/// the tree: a download lands under `target/`.
fn fetch_upstream(commit: &str, file: &str) -> Result<Vec<u8>, String> {
    if let Ok(root) = std::env::var("ES_PLAYGROUND_CACHE") {
        let cached = PathBuf::from(root).join(commit).join(file);
        if cached.is_file() {
            return std::fs::read(&cached).map_err(|e| format!("{}: {e}", cached.display()));
        }
    }
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/playground")
        .join(commit);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let out = out_dir.join(file);
    if !out.is_file() {
        let url = format!(
            "https://raw.githubusercontent.com/google-deepmind/mujoco_playground/{commit}\
             /mujoco_playground/_src/locomotion/go1/xmls/{file}"
        );
        // Every test in this file fetches, and `cargo test` runs them in parallel: curl into a
        // per-thread temporary and rename, or one thread reads what another is still writing.
        let tmp = out_dir.join(format!("{file}.{:?}.part", std::thread::current().id()));
        let status = Command::new("curl")
            .args(["-fsSL", "-o"])
            .arg(&tmp)
            .arg(&url)
            .status()
            .map_err(|e| format!("no cache (set ES_PLAYGROUND_CACHE) and `curl` failed: {e}"))?;
        if !status.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!(
                "no cache (set ES_PLAYGROUND_CACHE) and `curl {url}` exited with {status}"
            ));
        }
        let bytes = std::fs::read(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let _ = std::fs::rename(&tmp, &out);
        let _ = std::fs::remove_file(&tmp);
        return Ok(bytes);
    }
    std::fs::read(&out).map_err(|e| format!("{}: {e}", out.display()))
}

/// Verified upstream text for one pinned file, or `None` after printing the SKIP line.
fn upstream_text(file: &str) -> Option<String> {
    let (commit, robot_hash, scene_hash) = pin();
    let expected = match file {
        ROBOT => robot_hash,
        SCENE => scene_hash,
        other => panic!("{other} is not a pinned file"),
    };
    let bytes = match fetch_upstream(&commit, file) {
        Ok(bytes) => bytes,
        Err(why) => {
            println!("SKIP go1_provenance: {why}");
            return None;
        }
    };
    // Before parsing: untrusted bytes (spec 25.1). A mismatch is a failure, not a skip.
    let got = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(
        got, expected,
        "upstream {file} at {commit} hashes to {got}, the manifest pins {expected}: upstream \
         moved under the pin, or the download is corrupt"
    );
    println!("RAN go1_provenance ({file})");
    Some(String::from_utf8(bytes).expect("upstream MJCF is UTF-8"))
}

/// The two edits upstream's text needs before `parse_mjcf` will read it at all. Both are
/// importer limitations, not content, and neither touches anything this file asserts on:
///
///  * `<mesh class="go1" ...>` -> `<mesh ...>`: `crates/es-assets` reads `<asset>` before
///    `<default>` (`ROOT_ORDER`), so an asset naming a class is `unknown default class` --
///    and the `go1` class carries no `<mesh>` attribute for it to have inherited anyway.
///  * `<site>` lines are removed: the importer wants a 3-vector `size` and upstream writes
///    MJCF's scalar shorthand (`size="0.02"`). The derivative drops the same five sites,
///    which existed only for the `<sensor>` block that cannot survive `mjcf_out` either
///    (`go1_primitives.PROVENANCE.json`); a site carries no dynamics and `mjcf_out` emits none.
///
/// Upstream's mesh *geoms* are kept: the importer records a mesh asset by path without
/// opening the STL, so they parse, and they are exactly what the derivative replaced.
fn importable(text: &str) -> String {
    let text = text.replace("<mesh class=\"go1\" ", "<mesh ");
    let mut out = Vec::new();
    let mut in_sensor = false;
    for line in text.lines() {
        let t = line.trim_start();
        in_sensor |= t.starts_with("<sensor>");
        let drop = in_sensor || t.starts_with("<site ");
        in_sensor &= !t.starts_with("</sensor>");
        if !drop {
            out.push(line);
        }
    }
    out.join("\n")
}

/// Runs `body` with the parsed upstream **robot**, or returns after the SKIP line.
macro_rules! with_upstream {
    ($scene:ident, $body:block) => {
        let Some(text) = upstream_text(ROBOT) else {
            return;
        };
        let $scene = parse(&importable(&text), ROBOT).scene;
        $body
    };
}

// --- helpers ----------------------------------------------------------------------------------

fn joints_by_name(scene: &SceneDesc) -> BTreeMap<&str, &Joint> {
    scene.joints.iter().map(|j| (j.name.as_str(), j)).collect()
}

fn bodies_by_name(scene: &SceneDesc) -> BTreeMap<&str, &Body> {
    scene.bodies.iter().map(|b| (b.name.as_str(), b)).collect()
}

/// What makes two geoms the same primitive, independent of the name the file gave it:
/// upstream leaves every collision geom unnamed but the feet, so the importer's generated
/// `geom<n>` names differ between the two files by construction.
type GeomKey = (Shape, [u64; 3], [u64; 4]);

fn bits(v: f64) -> u64 {
    v.to_bits()
}

fn geom_key(g: &Geom) -> GeomKey {
    let p = g.pose.position;
    let q = g.pose.orientation;
    (
        g.shape,
        [bits(p.x), bits(p.y), bits(p.z)],
        [bits(q.x), bits(q.y), bits(q.z), bits(q.w)],
    )
}

fn is_mesh(g: &Geom) -> bool {
    matches!(g.shape, Shape::Mesh { .. } | Shape::HeightField { .. })
}

// --- the cross-checks ---------------------------------------------------------------------------

#[test]
fn derivative_has_the_upstream_joints_and_actuators() {
    with_upstream!(up, {
        let ours = derivative().scene;
        let mine = joints_by_name(&ours);
        let theirs = joints_by_name(&up);

        // Same twelve hinges in the same order, after the free joint both files start with.
        let ordered: Vec<&str> = ours
            .joints
            .iter()
            .map(|j| j.name.as_str())
            .filter(|n| JOINTS.contains(n))
            .collect();
        assert_eq!(ordered, JOINTS, "the derivative reorders the legs");
        let ordered_up: Vec<&str> = up
            .joints
            .iter()
            .map(|j| j.name.as_str())
            .filter(|n| JOINTS.contains(n))
            .collect();
        assert_eq!(
            ordered_up, JOINTS,
            "upstream is not the pinned twelve joints"
        );

        for name in JOINTS {
            let a = mine[name];
            let b = theirs[name];
            assert_eq!(a.kind, b.kind, "{name}: joint kind");
            assert_eq!(
                (bits(a.axis.x), bits(a.axis.y), bits(a.axis.z)),
                (bits(b.axis.x), bits(b.axis.y), bits(b.axis.z)),
                "{name}: axis"
            );
            assert_eq!(a.range, b.range, "{name}: range");
            assert_eq!(a.damping, b.damping, "{name}: damping");
            assert_eq!(a.armature, b.armature, "{name}: armature");
            assert_eq!(a.friction_loss, b.friction_loss, "{name}: frictionloss");
        }

        // The twelve position actuators: same gains, same forcerange, same driven joint.
        assert_eq!(ours.actuators.len(), up.actuators.len());
        for (a, b) in ours.actuators.iter().zip(&up.actuators) {
            assert_eq!(a.name, b.name, "actuator order");
            assert_eq!(a.kind, b.kind, "{}: gains", a.name);
            assert_eq!(a.ctrl_range, b.ctrl_range, "{}: ctrlrange", a.name);
            assert_eq!(a.force_range, b.force_range, "{}: forcerange", a.name);
            assert_eq!(a.gear, b.gear, "{}: gear", a.name);
            assert_eq!(
                mine.values()
                    .find(|j| a.target == es_assets::scene::ActuatorTarget::Joint(j.id))
                    .map(|j| &j.name),
                theirs
                    .values()
                    .find(|j| b.target == es_assets::scene::ActuatorTarget::Joint(j.id))
                    .map(|j| &j.name),
                "{}: driven joint",
                a.name
            );
        }
    });
}

#[test]
fn derivative_has_the_upstream_body_frames_and_inertials() {
    with_upstream!(up, {
        let ours = derivative().scene;
        let mine = bodies_by_name(&ours);
        let theirs = bodies_by_name(&up);
        for name in bodies() {
            let name = name.as_str();
            let a = mine
                .get(name)
                .unwrap_or_else(|| panic!("the derivative dropped body `{name}`"));
            let b = theirs
                .get(name)
                .unwrap_or_else(|| panic!("upstream has no body `{name}`"));
            let (pa, pb) = (a.pose.position, b.pose.position);
            assert_eq!(
                (bits(pa.x), bits(pa.y), bits(pa.z)),
                (bits(pb.x), bits(pb.y), bits(pb.z)),
                "{name}: pos"
            );
            // Both sides went through the same `mjcf::orient` normalization, so the
            // comparison is exact.
            let (qa, qb) = (a.pose.orientation, b.pose.orientation);
            assert_eq!(
                (bits(qa.x), bits(qa.y), bits(qa.z), bits(qa.w)),
                (bits(qb.x), bits(qb.y), bits(qb.z), bits(qb.w)),
                "{name}: orientation"
            );
            // The parent link is part of the frame: a re-parented body has the same pos.
            let parent = |scene: &SceneDesc, body: &Body| {
                body.parent
                    .and_then(|id| scene.bodies.iter().find(|p| p.id == id))
                    .map(|p| p.name.clone())
            };
            assert_eq!(parent(&ours, a), parent(&up, b), "{name}: parent");

            match (a.inertial, b.inertial) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.mass, b.mass, "{name}: mass");
                    let (ma, mb) = (a.inertia.matrix(), b.inertia.matrix());
                    for i in 0..3 {
                        assert_eq!(ma[i][i], mb[i][i], "{name}: inertia diagonal {i}");
                    }
                }
                (a, b) => panic!(
                    "{name}: inertial {} vs upstream {}",
                    a.is_some(),
                    b.is_some()
                ),
            }
        }
    });
}

#[test]
fn derivative_substitutes_only_the_visual_mesh_geoms() {
    with_upstream!(up, {
        let ours = derivative().scene;
        let mine = bodies_by_name(&ours);
        let theirs = bodies_by_name(&up);
        let mut substituted = 0usize;
        for name in bodies() {
            let name = name.as_str();
            let kept: Vec<GeomKey> = mine[name].geoms.iter().map(geom_key).collect();
            // Every upstream *collision* primitive survives, pose for pose.
            for g in &theirs[name].geoms {
                if is_mesh(g) {
                    substituted += 1;
                    continue;
                }
                assert!(
                    kept.contains(&geom_key(g)),
                    "{name}: upstream primitive geom at {:?} {:?} is missing from the derivative",
                    g.pose.position,
                    g.shape
                );
            }
            // And nothing upstream has that the derivative lacks is anything but a mesh.
            for g in &theirs[name].geoms {
                if !kept.contains(&geom_key(g)) {
                    assert!(
                        is_mesh(g),
                        "{name}: the derivative drops non-mesh geom `{}` ({:?})",
                        g.name,
                        g.shape
                    );
                }
            }
            // One box per dropped mesh, in the same body: a substitution, not a deletion.
            let upstream_meshes = theirs[name].geoms.iter().filter(|g| is_mesh(g)).count();
            let extra = mine[name].geoms.len() - (theirs[name].geoms.len() - upstream_meshes);
            assert_eq!(
                extra, upstream_meshes,
                "{name}: {upstream_meshes} mesh geom(s) upstream, {extra} added primitive(s) here"
            );
        }
        // 13 visual geoms across the tree, drawn from 5 distinct STL files.
        assert_eq!(
            substituted, 13,
            "upstream is not the pinned 13 visual geoms"
        );

        // The derivative itself carries no mesh anywhere -- that is why it exists.
        for body in &ours.bodies {
            for g in &body.geoms {
                assert!(!is_mesh(g), "the derivative has mesh geom `{}`", g.name);
            }
        }
        assert!(
            ours.assets.iter().all(|a| a.name == "dark"),
            "the derivative references mesh assets: {:?}",
            ours.assets.iter().map(|a| &a.name).collect::<Vec<_>>()
        );
    });
}

/// The solver block the policy trains under (api-note section 1), which is the whole physics
/// parity argument: `iterations=1 ls_iterations=5 timestep=0.004 integrator=Euler`.
#[test]
fn derivative_has_the_upstream_option_block() {
    with_upstream!(up, {
        let ours = derivative().scene;
        assert_eq!(
            ours.options, up.options,
            "<option> moved under the derivative"
        );
        assert_eq!(ours.options.timestep, 0.004);
        assert_eq!(ours.options.iterations, 1);
        assert_eq!(ours.options.ls_iterations, 5);
        assert!(!ours.options.eulerdamp, "<flag eulerdamp=\"disable\"/>");
        assert_eq!(ours.options.integrator, es_assets::scene::Integrator::Euler);
        assert_eq!(ours.options.cone, es_assets::scene::FrictionCone::Pyramidal);
    });
}

/// `home` is the reset pose *and* the action zero-point, so it is compared against the
/// upstream scene file's bytes rather than trusted (api-note section 1).
#[test]
fn derivative_carries_the_home_keyframe() {
    let ours = read_fixture(FIXTURE);
    let numbers = |text: &str, key: &str| -> Vec<f64> {
        let after = text.split_once(key).expect("the key is present").1;
        let value = after
            .split_once('"')
            .expect("an opening quote")
            .1
            .split_once('"')
            .expect("a closing quote")
            .0;
        value
            .split_whitespace()
            .map(|t| t.parse().expect("a number"))
            .collect()
    };
    let qpos = numbers(&ours, "<key name=\"home\" qpos=");
    assert_eq!(qpos, HOME_QPOS, "the derivative's `home` qpos");

    let Some(scene) = upstream_text(SCENE) else {
        return;
    };
    assert_eq!(
        numbers(&scene, "<key name=\"home\" qpos="),
        HOME_QPOS,
        "upstream's `home` qpos moved under the pin"
    );
    // The floor, the other thing inlined from the scene file.
    assert!(
        scene.contains("friction=\"0.6\" condim=\"3\"") && ours.contains("friction=\"0.6\""),
        "the inlined floor is not upstream's"
    );
}

/// Only the deviations the manifest enumerates reach the importer's warning list. `<keyframe>`
/// has no `SceneDesc` representation and geom `priority` / `group` are not carried, so a
/// clean parse is impossible here (unlike `so101_pick_place.xml`) -- what must hold is that
/// the set never grows silently.
#[test]
fn derivative_parses_with_only_the_enumerated_warnings() {
    let import = derivative();
    let expected = [
        "<keyframe>",
        "`priority`",
        "`group`",
        "`inheritrange`",
        "`rgba`",
        "`mode`",
    ];
    let unexpected: Vec<&str> = import
        .warnings
        .iter()
        .map(|w| w.message.as_str())
        .filter(|m| !expected.iter().any(|e| m.contains(e)))
        .collect();
    assert!(
        unexpected.is_empty(),
        "importer warnings the manifest does not enumerate: {unexpected:#?}"
    );
    println!(
        "go1_primitives.xml: {} enumerated importer warning(s)",
        import.warnings.len()
    );
    assert!(
        !read_fixture(FIXTURE).contains("<include"),
        "both root and body <include> are MjcfError::Include"
    );
    for forbidden in ["meshdir", "type=\"mesh\"", "<sensor", ".stl"] {
        assert!(
            !read_fixture(FIXTURE).contains(forbidden),
            "the fixture contains {forbidden}"
        );
    }
}

/// The numbers the four IR documents are built from, derived from the parse rather than
/// transcribed (`tests/fixtures/quadruped/*.toml`): the twelve joint ranges the Deployment
/// IR's `safety.position` copies, and the twelve `forcerange`s `torque_max` copies.
#[test]
fn joint_limits_are_derived_not_transcribed() {
    let scene = derivative().scene;
    let by_name = joints_by_name(&scene);
    for name in JOINTS {
        let j = by_name[name];
        let (lo, hi) = j.range.unwrap_or_else(|| panic!("{name} has no range"));
        assert!(
            lo < hi && lo.is_finite() && hi.is_finite(),
            "{name}: {lo}..{hi}"
        );
    }
    for a in &scene.actuators {
        let (lo, hi) = a
            .force_range
            .unwrap_or_else(|| panic!("{}: no forcerange", a.name));
        assert!(lo < 0.0 && hi > 0.0, "{}: {lo}..{hi}", a.name);
    }
    // Free base plus twelve hinges is what makes this a floating-base robot, which is the one
    // structural difference from every fixture before it (design note section 3).
    assert_eq!(
        scene
            .joints
            .iter()
            .filter(|j| j.kind == es_assets::scene::JointKind::Free)
            .count(),
        1,
        "the trunk carries exactly one free joint"
    );
    assert_eq!(scene.joints.len(), 13);
}
