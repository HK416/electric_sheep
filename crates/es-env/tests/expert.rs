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

// --- the chunk golden (packet M14/Q1) ----------------------------------------------------------

const CHUNKS_GOLDEN: &str = "tests/golden/expert/so101-pick-place-chunks.json";
/// Three cube positions inside the Task IR's draw: its two corners and a point between them.
/// Pinned rather than drawn, so the golden does not move with the draw's RNG; the check below
/// holds them to the Task IR's `Randomization` range.
const CUBES: [(f64, f64); 3] = [(0.21, -0.03), (0.24, 0.01), (0.27, 0.05)];
/// Chunks recorded once the demonstration is over, holding its last command.
const AFTER_DONE: usize = 3;
/// The golden's names for the built-in program's blocks: the stages it was recorded under,
/// before packet M14/Q1 made them the blocks of `templates/teach/so101-pick-place.toml`.
const STAGES: [&str; 8] = [
    "Approach",
    "Descend",
    "Close",
    "Lift",
    "Transport",
    "Lower",
    "Release",
    "Done",
];

/// `lo..=hi` of the Task IR's `Uniform` draw on `target`.
fn drawn_range(task: &es_ir::task::TaskIr, target: &str) -> (f64, f64) {
    use es_ir::task::{Distribution, TaskNode};
    task.graph
        .nodes
        .values()
        .find_map(|node| match node {
            TaskNode::Randomization {
                target: t,
                dist: Distribution::Uniform { lo, hi },
                ..
            } if t == target => Some((*lo, *hi)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the Task IR draws {target} from a Uniform"))
}

/// A `ModelInfo` laid out as `MuJoCo` lays out this scene -- joints and actuators in document
/// order -- so the golden needs no simulator.
fn model_of(scene: &SceneDesc) -> ModelInfo {
    use es_assets::scene::JointKind;
    use es_physics_core::backend::IndexRange;
    let mut model = ModelInfo::default();
    for joint in &scene.joints {
        let (nq, nv) = match joint.kind {
            JointKind::Free => (7, 6),
            JointKind::Ball => (4, 3),
            JointKind::Hinge | JointKind::Slide => (1, 1),
            JointKind::Fixed => (0, 0),
        };
        model.qpos.insert(joint.id, IndexRange::new(model.nq, nq));
        model.dof.insert(joint.id, IndexRange::new(model.nv, nv));
        model.nq += nq;
        model.nv += nv;
    }
    for actuator in &scene.actuators {
        model
            .actuator
            .insert(actuator.id, IndexRange::new(model.nu, 1));
        model.nu += 1;
    }
    model
}

fn hex(v: f64) -> String {
    format!("\"{:016x}\"", v.to_bits())
}

/// The golden's text: every chunk `es loop collect --expert so101-pick-place` builds for three
/// cubes, on a **perfect follower** -- each arm and gripper joint lands exactly on the last
/// executed row of the chunk, at rest, and the cube stays where it was put. No simulator, so
/// the sequence is a function of the expert alone.
fn chunks_golden_text() -> String {
    use es_assets::scene::{ActuatorTarget, JointKind};
    use std::fmt::Write as _;
    let scene = scene();
    let task = task_ir();
    let (x, y) = (drawn_range(&task, "qpos[6]"), drawn_range(&task, "qpos[7]"));
    for (cx, cy) in CUBES {
        assert!(
            (x.0..=x.1).contains(&cx) && (y.0..=y.1).contains(&cy),
            "cube ({cx}, {cy}) is outside the Task IR's draw {x:?} x {y:?}"
        );
    }
    let path = repo_root().join("tests/fixtures/visible-learning/deployment.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let deploy = es_ir::serial::deployment_from_toml(&text).expect("the demo deployment parses");
    let model = model_of(&scene);
    let nq = model.nq as usize;
    // `build_expert` in `crates/es/src/cmd/loop.rs`, step for step.
    let free: Vec<_> = scene
        .joints
        .iter()
        .filter(|j| j.kind == JointKind::Free)
        .collect();
    let [cube] = free.as_slice() else {
        panic!("the demo scene has one free joint")
    };
    let replan = es_env::replan_interval(deploy.rate)
        .expect("the demo's rates")
        .min(deploy.action.execute_chunk as u64);
    // Where each actuator's command lands in `qpos`.
    let follow: Vec<(usize, usize)> = scene
        .actuators
        .iter()
        .map(|a| {
            let ActuatorTarget::Joint(j) = a.target else {
                panic!("the demo's actuators drive joints")
            };
            (
                model.actuator[&a.id].start as usize,
                model.qpos[&j].start as usize,
            )
        })
        .collect();
    let cube_at = model.qpos[&cube.id].start as usize;
    let cube_z = scene
        .bodies
        .iter()
        .find(|b| b.id == cube.body)
        .expect("the cube's body")
        .pose
        .position
        .z;

    let mut out = format!(
        "{{\n  \"expert\": \"so101-pick-place\",\n  \"scene\": \"tests/fixtures/mjcf/so101_pick_place.xml\",\n  \"deployment\": \"tests/fixtures/visible-learning/deployment.toml\",\n  \"replan_every\": {replan},\n  \"follower\": \"perfect: joints at the last executed row, at rest; the cube stays put\",\n  \"episodes\": ["
    );
    for (e, (cx, cy)) in CUBES.into_iter().enumerate() {
        let mut cfg = demo_cfg(cube.id);
        cfg.pace_to(&deploy, replan as u32);
        let executed = cfg.execute as usize;
        let mut expert = ScriptedExpert::new(&scene, cfg).expect("the fixture builds an expert");
        let mut row = vec![0.0; nq + model.nv as usize];
        row[cube_at..cube_at + 4].copy_from_slice(&[cx, cy, cube_z, 1.0]);
        let sep = if e == 0 { "" } else { "," };
        write!(
            out,
            "{sep}\n    {{\n      \"cube\": [{}, {}, {}],\n      \"chunks\": [",
            hex(cx),
            hex(cy),
            hex(cube_z)
        )
        .expect("a String takes any text");
        // The episode's budget, in re-plans: `max_episode_steps` control ticks.
        let cap = task.config.max_episode_steps as usize / replan as usize;
        let (mut n, mut after_done) = (0, 0);
        while after_done < AFTER_DONE {
            assert!(n < cap, "cube ({cx}, {cy}): not done in {cap} re-plans");
            let stage = STAGES[expert.block()];
            let rows = expert
                .chunk(&model, &state_of_row(&model, &row), 0)
                .unwrap_or_else(|| panic!("cube ({cx}, {cy}): out of reach in {stage}"));
            if stage == "Done" {
                after_done += 1;
            }
            let sep = if n == 0 { "" } else { "," };
            write!(out, "{sep}\n        {{\"stage\": \"{stage}\", \"rows\": [")
                .expect("a String takes any text");
            for (k, r) in rows.iter().enumerate() {
                let sep = if k == 0 { "" } else { "," };
                let values: Vec<String> = r.iter().map(|v| hex(*v)).collect();
                write!(out, "{sep}\n          [{}]", values.join(", "))
                    .expect("a String takes any text");
            }
            out.push_str("\n        ]}");
            for &(slot, q) in &follow {
                row[q] = rows[executed - 1][slot];
            }
            n += 1;
        }
        out.push_str("\n      ]\n    }");
    }
    out.push_str("\n  ]\n}\n");
    out
}

/// Regenerates [`CHUNKS_GOLDEN`]. Run once, explicitly; it is then read-only (spec 1.4).
#[test]
#[ignore = "golden generator; run explicitly with ES_GENERATE_GOLDENS=1"]
fn generate_expert_chunks_golden() {
    // Spec 1.4: goldens are CI read-only, and `cargo test -- --include-ignored` runs every
    // ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_expert_chunks_golden: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    let path = repo_root().join(CHUNKS_GOLDEN);
    std::fs::create_dir_all(path.parent().expect("a parent")).expect("golden dir");
    std::fs::write(&path, chunks_golden_text()).expect("write the golden");
}

/// Packet M14/Q1: the expert's chunk sequence is the golden, bit for bit -- the oracle that
/// the demonstration program reproduces today's demonstrator exactly.
#[test]
fn expert_chunks_are_the_golden() {
    let path = repo_root().join(CHUNKS_GOLDEN);
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let got = chunks_golden_text();
    if let Some((i, (g, w))) = got
        .lines()
        .zip(want.lines())
        .enumerate()
        .find(|(_, (g, w))| g != w)
    {
        panic!("line {}: the expert gives\n{g}\nthe golden has\n{w}", i + 1);
    }
    assert_eq!(
        got.len(),
        want.len(),
        "the golden and the expert differ in length"
    );
}
