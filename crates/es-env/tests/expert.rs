//! Oracles for the scripted expert (packet `docs/packets/M5/V1-expert-dataset.md`).
//!
//! The closed-form IK is judged by **`MuJoCo`'s own forward kinematics**, not by restating the
//! algebra in the test (spec 1.4): the solution is written into `qpos`, the backend runs
//! `mj_forward`, and the tool site it reports is compared with the target that was asked for.
//! No `mujoco` on this machine prints `SKIP <test>: <why>` and returns; having run, it prints
//! `RAN <test>`.
//!
//!     ES_PYTHON=$HOME/venvs/es/bin/python \
//!       cargo test -p es-env --test expert -- --ignored --nocapture
//!
//! The expert's **success rate** is measured where the demonstrations are actually written,
//! through `es loop collect --expert` (`crates/es/tests/cli.rs`, `expert_success`): a number
//! taken from anything but the recorded dataset would be a number about a different run.

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_env::expert::{demo_cfg, so101_ik, state_of_row, Links, ScriptedExpert};
use es_env::Env;
use es_math::{Pose, Quat, Vec3};
use es_physics_backend::MuJoCoCpuBackend;
use es_physics_core::backend::{LoadConfig, ModelInfo, PhysicsBackend, StateView};

const DOWN: f64 = -std::f64::consts::FRAC_PI_2;
/// The tool site's position, in metres, must land this close to what the IK was asked for.
/// `es_math::approx` is `f32`, so the floor is about a micrometre; this is 20 of them.
const FK_TOL: f64 = 2e-5;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scene() -> SceneDesc {
    let path = repo_root().join("tests/fixtures/mjcf/so101_pick_place.xml");
    let xml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    es_assets::parse_mjcf(&xml)
        .expect("the V0 fixture parses")
        .scene
}

fn joint_id(scene: &SceneDesc, name: &str) -> StableId {
    scene
        .joints
        .iter()
        .find(|j| j.name == name)
        .unwrap_or_else(|| panic!("the fixture has a joint named {name}"))
        .id
}

/// `Some(backend)` with the demo scene loaded, or a printed SKIP.
fn loaded(test: &str) -> Option<(MuJoCoCpuBackend, ModelInfo, SceneDesc)> {
    if let Err(why) = MuJoCoCpuBackend::is_available() {
        println!("SKIP {test}: {why}");
        return None;
    }
    let scene = scene();
    let mut backend = MuJoCoCpuBackend::new();
    let model = backend
        .load(&scene, &LoadConfig::default())
        .expect("the demo scene loads into MuJoCo");
    println!("RAN {test}");
    Some((backend, model, scene))
}

/// The tool site's world position after `MuJoCo`'s own forward kinematics for `q`.
fn fk_tool(
    backend: &mut MuJoCoCpuBackend,
    model: &ModelInfo,
    scene: &SceneDesc,
    q: &[f64; 4],
) -> Vec3 {
    let mut qpos = vec![0.0; model.nq as usize];
    for (i, name) in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
        .into_iter()
        .enumerate()
    {
        let at = model.qpos[&joint_id(scene, name)].start as usize;
        qpos[at] = q[i];
    }
    let qvel = vec![0.0; model.nv as usize];
    let state = StateView {
        n_envs: 1,
        qpos: &qpos,
        qvel: &qvel,
        ..StateView::default()
    };
    // `reset` runs `mj_forward`, so `xpos`/`xquat` describe exactly this configuration.
    backend.reset(None, Some(&state)).expect("reset");
    let after = backend.state();

    let body = scene
        .bodies
        .iter()
        .find(|b| b.sites.iter().any(|s| s.name == "gripperframe"))
        .expect("the tool body");
    let site = body
        .sites
        .iter()
        .find(|s| s.name == "gripperframe")
        .expect("the tool site");
    let row = model.body[&body.id].start as usize;
    let p = &after.xpos[row * 3..row * 3 + 3];
    let r = &after.xquat[row * 4..row * 4 + 4];
    Pose::new(
        Vec3::new(p[0], p[1], p[2]),
        Quat::from_xyzw(r[0], r[1], r[2], r[3]),
    )
    .transform_point(site.pose.position)
}

/// Spec 1.4: the oracle for the closed form is `MuJoCo`'s forward kinematics, not our own
/// algebra restated.
#[test]
#[ignore = "needs a Python with mujoco (ES_PYTHON); oracle tier"]
fn ik_round_trips_through_forward_kinematics() {
    let Some((mut backend, model, scene)) = loaded("ik_round_trips_through_forward_kinematics")
    else {
        return;
    };
    let links = Links::from_scene(&scene).expect("the fixture is a yaw plus a planar 3R");

    let mut checked = 0;
    let mut worst = 0.0f64;
    for x in [0.16, 0.20, 0.24, 0.28] {
        for y in [-0.08, -0.03, 0.0, 0.05] {
            for z in [0.015, 0.05, 0.09] {
                for pitch in [DOWN, -1.3, -1.0] {
                    let target = Vec3::new(x, y, z);
                    let Some(q) = so101_ik(&links, target, pitch) else {
                        continue;
                    };
                    let got = fk_tool(&mut backend, &model, &scene, &q);
                    let err = (got - target).norm();
                    worst = worst.max(err);
                    assert!(
                        err <= FK_TOL,
                        "IK for {target:?} at pitch {pitch} solved {q:?}, but MuJoCo puts the \
                         tool at {got:?} ({err} m away)"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 40, "only {checked} targets were reachable");
    println!("{checked} targets round-tripped, worst error {worst:.3e} m");
}

/// Spec 3.4 / spec 6.3: the demonstration is a pure function of `(task, seed, episode)`. Two
/// runs of the same seed produce byte-identical `ctrl` rows -- which is what makes the
/// dataset's content hash mean anything.
#[test]
#[ignore = "needs a Python with mujoco (ES_PYTHON); oracle tier"]
fn the_same_seed_gives_the_same_demonstration() {
    if MuJoCoCpuBackend::is_available().is_err() {
        println!(
            "SKIP the_same_seed_gives_the_same_demonstration: {}",
            MuJoCoCpuBackend::is_available().unwrap_err()
        );
        return;
    }
    println!("RAN the_same_seed_gives_the_same_demonstration");
    let a = rollout(31, 60);
    let b = rollout(31, 60);
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty(), "the rollout recorded nothing");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "ctrl value {i} differs between two runs of seed 31: {x} vs {y}"
        );
    }
    // A different seed draws a different cube pose, so the demonstration must differ.
    let c = rollout(32, 60);
    assert!(a != c, "two seeds produced the same demonstration");
}

/// `steps` control steps of one expert-driven episode, as the flat `ctrl` rows it commanded.
fn rollout(seed: u64, steps: u32) -> Vec<f64> {
    let scene = scene();
    let task = task_ir();
    let domains = es_env::scheduler::BatchDomains::single_env();
    let mut env: Env<MuJoCoCpuBackend> =
        Env::new(&task, &scene, MuJoCoCpuBackend::new(), &domains, seed)
            .expect("the demo task builds an env");
    let mut expert = ScriptedExpert::new(&scene, demo_cfg(joint_id(&scene, "cube_free")))
        .expect("the fixture builds an expert");
    let model = env.model().clone();

    let mut out = Vec::new();
    for _ in 0..steps {
        let row: Vec<f64> = {
            let state = env.backend().state();
            let mut row = state.qpos_of(0).to_vec();
            row.extend_from_slice(state.qvel_of(0));
            row
        };
        let Some(ctrl) = expert.action(&model, &state_of_row(&model, &row), 0) else {
            break;
        };
        out.extend_from_slice(&ctrl);
        let outcome = env.step(&ctrl).expect("step");
        if !outcome.episodes.is_empty() {
            break;
        }
    }
    out
}

/// V0's Task IR, which owns the cube's randomization and the success predicate.
fn task_ir() -> es_ir::task::TaskIr {
    let path = repo_root().join("tests/fixtures/visible-learning/task.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    es_ir::serial::task_from_toml(&text).expect("V0's task document parses")
}
