//! The SO-101 demo scene is judged against upstream, not against review (spec 1.4).
//!
//! `tests/fixtures/mjcf/so101_pick_place.xml` is a hand-derived, primitives-only,
//! single-file derivative of `robotstudio_so101/so101.xml` in
//! `google-deepmind/mujoco_menagerie`. This test obtains the upstream file at the pinned
//! commit, verifies its blake3 **before parsing it** (spec 25.1: bytes off the network are
//! untrusted), and proves that the derivative carries the same kinematics.
//!
//! Upstream comes from `$ES_MENAGERIE_CACHE/<commit>/so101.xml` when that exists, else from
//! `curl -fsSL` into `target/`. No cache and no network prints `SKIP so101_provenance: <why>`;
//! having run, it prints `RAN so101_provenance`. A blake3 **mismatch is a failure, never a
//! skip**: it means upstream moved under the pin.
//!
//!     ES_MENAGERIE_CACHE=$HOME/cache/menagerie \
//!       cargo test -p es-assets --test so101_provenance -- --nocapture

// Equality with upstream is the property under test; a tolerance here would be the bug.
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use es_assets::scene::{Body, Geom, Joint, SceneDesc, Shape};
use es_assets::Import;

const FIXTURE: &str = "so101_pick_place.xml";
const MANIFEST: &str = "so101_pick_place.PROVENANCE.json";

/// The kinematic chain, root first. `camera_mount` hangs off `gripper` with no joint.
const CHAIN: [&str; 8] = [
    "base",
    "shoulder",
    "upper_arm",
    "lower_arm",
    "wrist",
    "gripper",
    "camera_mount",
    "moving_jaw_so101_v1",
];

/// The six actuated joints, in upstream declaration order.
const JOINTS: [&str; 6] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
];

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
/// the packet adds none. The manifest is repo-owned and machine-written, so a flat
/// `"name": "value"` scan is enough -- and a malformed manifest fails the test, loudly.
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

/// `(commit, expected blake3 hex)`.
fn pin() -> (String, String) {
    let json = read_fixture(MANIFEST);
    assert_eq!(manifest_field(&json, "license"), "Apache-2.0");
    assert_eq!(manifest_field(&json, "path"), "robotstudio_so101/so101.xml");
    (
        manifest_field(&json, "commit"),
        manifest_field(&json, "blake3_so101_xml"),
    )
}

// --- upstream ---------------------------------------------------------------------------------

/// Upstream `so101.xml` bytes, or the reason there are none. Never writes into the tree:
/// a download lands under `target/`.
fn fetch_upstream(commit: &str) -> Result<Vec<u8>, String> {
    if let Ok(root) = std::env::var("ES_MENAGERIE_CACHE") {
        let cached = PathBuf::from(root).join(commit).join("so101.xml");
        if cached.is_file() {
            return std::fs::read(&cached).map_err(|e| format!("{}: {e}", cached.display()));
        }
    }
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/menagerie")
        .join(commit);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let out = out_dir.join("so101.xml");
    if !out.is_file() {
        let url = format!(
            "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}\
             /robotstudio_so101/so101.xml"
        );
        // Every test in this file fetches, and `cargo test` runs them in parallel: curl into a
        // per-thread temporary and rename, or one thread reads what another is still writing.
        let tmp = out_dir.join(format!("so101.xml.{:?}.part", std::thread::current().id()));
        let status = Command::new("curl")
            .args(["-fsSL", "-o"])
            .arg(&tmp)
            .arg(&url)
            .status()
            .map_err(|e| format!("no cache (set ES_MENAGERIE_CACHE) and `curl` failed: {e}"))?;
        if !status.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!(
                "no cache (set ES_MENAGERIE_CACHE) and `curl {url}` exited with {status}"
            ));
        }
        let bytes = std::fs::read(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let _ = std::fs::rename(&tmp, &out);
        let _ = std::fs::remove_file(&tmp);
        return Ok(bytes);
    }
    std::fs::read(&out).map_err(|e| format!("{}: {e}", out.display()))
}

/// The parsed upstream scene, or `None` after printing the SKIP line.
fn upstream() -> Option<SceneDesc> {
    let (commit, expected) = pin();
    let bytes = match fetch_upstream(&commit) {
        Ok(bytes) => bytes,
        Err(why) => {
            println!("SKIP so101_provenance: {why}");
            return None;
        }
    };
    // Before parsing: untrusted bytes (spec 25.1). A mismatch is a failure, not a skip.
    let got = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(
        got, expected,
        "upstream so101.xml at {commit} hashes to {got}, the manifest pins {expected}: \
         upstream moved under the pin, or the download is corrupt"
    );
    let xml = String::from_utf8(bytes).expect("upstream so101.xml is UTF-8");
    println!("RAN so101_provenance");
    Some(parse(&xml, "upstream so101.xml").scene)
}

/// Runs `body` with the upstream scene, or returns after the SKIP line.
macro_rules! with_upstream {
    ($scene:ident, $body:block) => {
        let Some($scene) = upstream() else { return };
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

/// What makes two geoms the same primitive, independent of the name the file gave it: upstream
/// leaves most collision geoms unnamed, so the importer's generated `geom<n>` names differ
/// between the two files by construction.
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
fn derivative_has_the_upstream_kinematics() {
    with_upstream!(up, {
        let ours = derivative().scene;
        let mine = joints_by_name(&ours);
        let theirs = joints_by_name(&up);

        // Same six joints, same order in the file.
        let ordered: Vec<&str> = ours
            .joints
            .iter()
            .map(|j| j.name.as_str())
            .filter(|n| JOINTS.contains(n))
            .collect();
        assert_eq!(ordered, JOINTS, "the derivative reorders the chain");
        let ordered_up: Vec<&str> = up.joints.iter().map(|j| j.name.as_str()).collect();
        assert_eq!(ordered_up, JOINTS, "upstream is not the pinned six joints");

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

        // The six position actuators: same gains, same ctrlrange, same joint.
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
fn derivative_has_the_upstream_body_frames() {
    with_upstream!(up, {
        let ours = derivative().scene;
        let mine = bodies_by_name(&ours);
        let theirs = bodies_by_name(&up);
        for name in CHAIN {
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
        }
    });
}

#[test]
fn derivative_has_the_upstream_inertials() {
    with_upstream!(up, {
        let ours = derivative().scene;
        let mine = bodies_by_name(&ours);
        let theirs = bodies_by_name(&up);
        for name in CHAIN {
            let a = mine[name].inertial;
            let b = theirs[name].inertial;
            match (a, b) {
                (None, None) => {}
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
fn derivative_drops_only_visual_mesh_geoms() {
    with_upstream!(up, {
        let ours = derivative().scene;
        let mine = bodies_by_name(&ours);
        let theirs = bodies_by_name(&up);
        for name in CHAIN {
            let kept: Vec<GeomKey> = mine[name].geoms.iter().map(geom_key).collect();
            for g in &theirs[name].geoms {
                if is_mesh(g) {
                    continue;
                }
                assert!(
                    kept.contains(&geom_key(g)),
                    "{name}: upstream primitive geom at {:?} {:?} is missing from the derivative",
                    g.pose.position,
                    g.shape
                );
            }
            // Anything upstream has that the derivative does not is a mesh, and nothing else.
            let ours_keys: Vec<GeomKey> = kept.clone();
            for g in &theirs[name].geoms {
                if !ours_keys.contains(&geom_key(g)) {
                    assert!(
                        is_mesh(g),
                        "{name}: the derivative drops non-mesh geom `{}` ({:?})",
                        g.name,
                        g.shape
                    );
                }
            }
        }
        // And the derivative itself carries no mesh anywhere -- that is why it exists.
        for body in &ours.bodies {
            for g in &body.geoms {
                assert!(!is_mesh(g), "the derivative has mesh geom `{}`", g.name);
            }
        }
        assert!(
            ours.assets.is_empty(),
            "the derivative references assets: {:?}",
            ours.assets.iter().map(|a| &a.name).collect::<Vec<_>>()
        );
    });
}

#[test]
fn derivative_parses_with_no_warnings() {
    let import = derivative();
    assert!(
        import.warnings.is_empty(),
        "the demo scene must contain no element SceneDesc drops:\n{:#?}",
        import.warnings
    );
}

#[test]
fn derivative_has_no_include() {
    let text = read_fixture(FIXTURE);
    assert!(
        !text.contains("<include"),
        "both root and body <include> are MjcfError::Include"
    );
    for forbidden in ["<keyframe", "<equality", "meshdir", "type=\"mesh\""] {
        assert!(
            !text.contains(forbidden),
            "the fixture contains {forbidden}"
        );
    }
}

/// The link lengths V1's closed-form IK consumes, derived from the parse rather than
/// transcribed: `so101_ik` must never hold a hand-copied number (design note section 4.2).
fn links(scene: &SceneDesc) -> Vec<(String, f64)> {
    let by_name = bodies_by_name(scene);
    let mut out = Vec::new();
    for name in ["upper_arm", "lower_arm", "wrist", "gripper"] {
        let p = by_name[name].pose.position;
        out.push((name.to_owned(), p.norm()));
    }
    let gripper = by_name["gripper"];
    let tip = gripper
        .sites
        .iter()
        .find(|s| s.name == "gripperframe")
        .expect("the gripper carries the `gripperframe` site");
    out.push(("gripperframe".to_owned(), tip.pose.position.norm()));
    out
}

#[test]
fn link_lengths_are_derived_not_transcribed() {
    let scene = derivative().scene;
    let derived = links(&scene);
    // Pure function of the parse: a second parse of the same bytes gives the same numbers.
    assert_eq!(derived, links(&derivative().scene));
    assert_eq!(derived.len(), 5);
    assert!(
        derived.iter().all(|(_, v)| v.is_finite() && *v > 0.0),
        "{derived:?}"
    );

    let json = format!(
        "{{\n{}\n}}\n",
        derived
            .iter()
            .map(|(name, v)| format!("  \"{name}\": {v:?}"))
            .collect::<Vec<_>>()
            .join(",\n")
    );
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/so101_links.json");
    std::fs::write(&out, &json).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
    println!("links -> {}\n{json}", out.display());

    with_upstream!(up, {
        assert_eq!(
            derived,
            links(&up),
            "the derivative's link lengths differ from upstream"
        );
    });
}
