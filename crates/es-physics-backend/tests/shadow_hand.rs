//! Plan H, task H1: the Shadow Hand scene (`tests/fixtures/mjcf/shadow_hand/`) in this runtime.
//!
//! The fixture is the SSR bundle's repose-cube MJCF, derived (`PROVENANCE.json`). What this
//! file pins:
//!
//! * it parses with **zero warnings** -- everything it says reaches `SceneDesc` -- and names
//!   the 24 hand joints, the two free joints and the 20 servos in `MuJoCo`'s order;
//! * the emitter writes what the hand's physics needs and `SceneDesc` now carries: the four
//!   J1/J0 coupling tendons, the 17 fingertip `<contact><pair>`s and `gravcomp`;
//! * `mujoco-cpu` and `mjwarp` map every feature it uses (spec 17.2, 14.4);
//! * the oracle (spec 1.4): 240 ticks through `MuJoCoCpuBackend` (parse -> `SceneDesc` ->
//!   emitted MJCF) agree to 1e-9 with `MuJoCo` loading the fixture file directly, under the
//!   same controls. Needs an interpreter with `mujoco` (`ES_PYTHON`); prints `SKIP` without.
//!
//! ```text
//! ES_PYTHON=.venv/Scripts/python.exe cargo test -p es-physics-backend --test shadow_hand -- --nocapture
//! ```

// The fixture's numbers are compared exactly: they are read, not computed.
#![allow(clippy::float_cmp)]

use std::path::PathBuf;
use std::process::Command;

use es_assets::scene::{ActuatorKind, JointKind, SceneDesc, TendonKind};
use es_core::TickRate;
use es_physics_backend::{
    mapping_report, scene_to_mjcf, BackendKind, MjcfRow, MuJoCoCpuBackend, Status, TaskFeature,
};
use es_physics_core::{Feature, LoadConfig, PhysicsBackend};

const JOINTS: [&str; 24] = [
    "robot0:WRJ1",
    "robot0:WRJ0",
    "robot0:FFJ3",
    "robot0:FFJ2",
    "robot0:FFJ1",
    "robot0:FFJ0",
    "robot0:MFJ3",
    "robot0:MFJ2",
    "robot0:MFJ1",
    "robot0:MFJ0",
    "robot0:RFJ3",
    "robot0:RFJ2",
    "robot0:RFJ1",
    "robot0:RFJ0",
    "robot0:LFJ4",
    "robot0:LFJ3",
    "robot0:LFJ2",
    "robot0:LFJ1",
    "robot0:LFJ0",
    "robot0:THJ4",
    "robot0:THJ3",
    "robot0:THJ2",
    "robot0:THJ1",
    "robot0:THJ0",
];

/// The J0 joints of the four fingers have no servo: the coupling tendon drives them.
const ACTUATORS: [&str; 20] = [
    "robot0:A_WRJ1",
    "robot0:A_WRJ0",
    "robot0:A_FFJ3",
    "robot0:A_FFJ2",
    "robot0:A_FFJ1",
    "robot0:A_MFJ3",
    "robot0:A_MFJ2",
    "robot0:A_MFJ1",
    "robot0:A_RFJ3",
    "robot0:A_RFJ2",
    "robot0:A_RFJ1",
    "robot0:A_LFJ4",
    "robot0:A_LFJ3",
    "robot0:A_LFJ2",
    "robot0:A_LFJ1",
    "robot0:A_THJ4",
    "robot0:A_THJ3",
    "robot0:A_THJ2",
    "robot0:A_THJ1",
    "robot0:A_THJ0",
];

/// 1 / 120 s, the bundle's physics step.
const RATE_HZ: u64 = 120;
/// Two seconds of physics: 120 ticks with the servos at zero (the cube lands on the palm),
/// then 120 with every servo at 40% of its range (the fingers close on it and on each other).
const TICKS: u32 = 240;
const CLOSE_FRACTION: f64 = 0.4;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/shadow_hand")
}

fn fixture() -> PathBuf {
    dir().join("shadow_hand_repose.xml")
}

fn scene() -> SceneDesc {
    let xml = std::fs::read_to_string(fixture()).expect("the Shadow Hand fixture");
    let import = es_assets::parse_mjcf(&xml).expect("the Shadow Hand fixture parses");
    assert!(
        import.warnings.is_empty(),
        "the fixture parses with warnings: {:#?}",
        import.warnings
    );
    let mut scene = import.scene;
    es_assets::mesh::load(&mut scene, &dir()).expect("the twelve STLs load");
    scene
}

#[test]
fn the_hand_parses_and_names_its_dofs_in_mujoco_order() {
    let scene = scene();
    let names: Vec<&str> = scene.joints.iter().map(|j| j.name.as_str()).collect();
    assert_eq!(&names[..24], &JOINTS[..]);
    assert_eq!(&names[24..], ["object:joint", "target:joint"]);
    assert!(scene.joints[..24]
        .iter()
        .all(|j| j.kind == JointKind::Hinge && j.range.is_some()));
    assert!(scene.joints[24..].iter().all(|j| j.kind == JointKind::Free));

    let names: Vec<&str> = scene.actuators.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ACTUATORS);
    // The bundle's `<general gainprm=kp biasprm="0 -kp -kv">`, spelled as the servo it is.
    for a in &scene.actuators {
        let ActuatorKind::Position { kp, kv } = a.kind else {
            panic!("{} is not a position servo", a.name)
        };
        let (want_kp, want_kv) = if a.name.starts_with("robot0:A_WR") {
            (5.0, 0.5)
        } else {
            (1.0, 0.100_000_001_490_116_12)
        };
        assert_eq!((kp, kv), (want_kp, want_kv), "{}", a.name);
        assert!(
            a.ctrl_range.is_some() && a.force_range.is_some(),
            "{}",
            a.name
        );
    }

    assert_eq!(scene.tendons.len(), 4);
    assert!(scene
        .tendons
        .iter()
        .all(|t| matches!(&t.kind, TendonKind::Fixed { joints } if joints.len() == 2)));
    assert_eq!(scene.contact_pairs.len(), 17);
    assert!(scene.contact_pairs.iter().all(|p| p.condim == Some(1)));
    let body = |name: &str| scene.bodies.iter().find(|b| b.name == name).unwrap();
    // Every hand body, and the floating goal cube; never the cube the hand holds.
    assert_eq!(scene.gravcomp.get(&body("target").id), Some(&1.0));
    assert_eq!(scene.gravcomp.get(&body("robot0:palm").id), Some(&1.0));
    assert!(!scene.gravcomp.contains_key(&body("object").id));

    let cameras: Vec<&str> = scene.cameras.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(cameras, ["top", "front", "side"]);

    // The face slabs change nothing about either cube's mass: the box geom carries it all.
    for cube in ["object", "target"] {
        let b = body(cube);
        assert!(b.inertial.is_none(), "{cube} takes its mass from its geoms");
        let slabs: Vec<_> = b.geoms.iter().filter(|g| g.name != cube).collect();
        assert_eq!(slabs.len(), 6, "{cube}");
        assert!(slabs
            .iter()
            .all(|g| g.mass == Some(0.0) && g.visual_only && g.rgba[3] == 1.0));
        let own = b.geoms.iter().find(|g| g.name == cube).unwrap();
        assert_eq!(own.rgba[3], 0.0, "{cube}'s own box is not drawn");
    }
}

#[test]
fn the_emitted_mjcf_carries_tendons_pairs_and_gravcomp() {
    let scene = scene();
    let mjcf = scene_to_mjcf(&scene).expect("the Shadow Hand scene emits");
    for wanted in [
        "<fixed name=\"robot0:T_FFJ1c\" limited=\"true\" range=\"-0.0010000000474974513 \
         0.0010000000474974513\" damping=\"0.10000000149011612\">",
        "<joint joint=\"robot0:FFJ1\" coef=\"-0.008050000295042992\"/>",
        "<pair geom1=\"robot0:C_ffproximal\" geom2=\"robot0:C_mfproximal\" condim=\"1\"/>",
        "<body name=\"target\" pos=\"",
        "gravcomp=\"1.0\"",
    ] {
        assert!(mjcf.contains(wanted), "the emitted MJCF lacks {wanted}");
    }
    let again = es_assets::parse_mjcf(&mjcf)
        .unwrap_or_else(|e| panic!("re-parsing the emitted MJCF: {e}"))
        .scene;
    assert_eq!(again.tendons.len(), 4);
    assert_eq!(again.contact_pairs, scene.contact_pairs);
    assert_eq!(again.gravcomp, scene.gravcomp);
}

#[test]
fn mujoco_cpu_and_mjwarp_map_every_feature_the_hand_uses() {
    let scene = scene();
    for backend in [BackendKind::MuJoCoCpu, BackendKind::MjWarp] {
        let report = mapping_report(&scene, backend);
        println!("{report}");
        assert!(!report.blocked, "{backend} blocks the hand:\n{report}");
        for feature in [
            TaskFeature::Capability(Feature::Tendon),
            TaskFeature::Mjcf(MjcfRow::ContactPair),
            TaskFeature::Mjcf(MjcfRow::BodyGravComp),
        ] {
            let row = report
                .rows
                .iter()
                .find(|r| r.feature == feature)
                .unwrap_or_else(|| panic!("{backend}: no `{feature}` row"));
            assert!(
                matches!(row.mapping.status, Status::Native(_)),
                "{backend}: `{feature}` is {:?}",
                row.mapping
            );
        }
    }
}

/// `mujoco` stepping the fixture straight off disk under the controls the backend gets.
const DIRECT: &str = r#"
import json, sys
import mujoco
import numpy as np

path, ticks, close = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)  # what the adapter does after load
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
contacts = 0
for t in range(ticks):
    d.ctrl[:] = np.clip(0.0, lo, hi) if t < ticks // 2 else lo + close * (hi - lo)
    mujoco.mj_step(m, d)
    contacts = max(contacts, int(d.ncon))
print(json.dumps({
    "qpos": d.qpos.tolist(),
    "contacts": contacts,
    "bad": int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number),
}))
"#;

fn run_direct() -> Result<String, String> {
    let python = std::env::var("ES_PYTHON")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| "python".to_owned());
    let out = Command::new(&python)
        .args(["-c", DIRECT])
        .arg(fixture())
        .arg(TICKS.to_string())
        .arg(CLOSE_FRACTION.to_string())
        .output()
        .map_err(|e| format!("`{python}`: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// One JSON array or number field, without a JSON dependency in this crate.
fn field(json: &str, name: &str) -> Vec<f64> {
    let after = json
        .split_once(&format!("\"{name}\":"))
        .unwrap_or_else(|| panic!("no \"{name}\" in {json}"))
        .1
        .trim_start();
    let end = if after.starts_with('[') {
        after.find(']').unwrap() + 1
    } else {
        after.find([',', '}']).unwrap()
    };
    after[..end]
        .trim_matches(['[', ']'].as_slice())
        .split(',')
        .map(|t| t.trim().parse().expect("a number"))
        .collect()
}

#[test]
fn the_backend_steps_the_hand_as_mujoco_loading_the_file_does() {
    if let Err(reason) = MuJoCoCpuBackend::is_available() {
        println!("SKIP shadow_hand: {reason}");
        return;
    }
    let scene = scene();
    let mut backend = MuJoCoCpuBackend::new();
    let info = backend
        .load(
            &scene,
            &LoadConfig {
                n_envs: 1,
                rate: Some(TickRate::hz(RATE_HZ)),
                seed: 1,
            },
        )
        .expect("the Shadow Hand loads in MuJoCo");
    // 24 hinges, then the object's free joint at qpos 24..31 and the target's at 31..38.
    assert_eq!((info.nq, info.nv, info.nu), (38, 36, 20));

    let ranges: Vec<(f64, f64)> = scene
        .actuators
        .iter()
        .map(|a| a.ctrl_range.expect("every servo is ctrl-limited"))
        .collect();
    let open: Vec<f64> = ranges
        .iter()
        .map(|(lo, hi)| 0.0f64.clamp(*lo, *hi))
        .collect();
    let close: Vec<f64> = ranges
        .iter()
        .map(|(lo, hi)| lo + CLOSE_FRACTION * (hi - lo))
        .collect();
    for ctrl in [open, close] {
        backend.set_ctrl(&ctrl).expect("20 controls");
        let report = backend.step(TICKS / 2).expect("a step");
        assert!(report.failures.is_empty(), "{:?}", report.failures);
    }
    let ours = backend.state().qpos_of(0).to_vec();

    let json = match run_direct() {
        Ok(json) => json,
        Err(why) => {
            println!("SKIP shadow_hand (direct mujoco): {why}");
            return;
        }
    };
    assert_eq!(
        field(&json, "bad"),
        [0.0],
        "the reference run went unstable"
    );
    let contacts = field(&json, "contacts")[0];
    assert!(contacts > 0.0, "the run never touched anything");
    let theirs = field(&json, "qpos");
    assert_eq!(theirs.len(), ours.len());
    let worst = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "shadow_hand: {TICKS} ticks, up to {contacts} contacts, worst |dqpos| {worst:.3e}; \
         cube at {:?}",
        &ours[24..27]
    );
    assert!(
        worst <= 1e-9,
        "the backend and MuJoCo on the file disagree by {worst:e}:\nours   {ours:?}\ntheirs {theirs:?}"
    );
}
