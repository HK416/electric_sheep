//! Saying the task (packet M17/G9): a new specification for a scene, with its robot, one
//! clause a fresh scene can satisfy, every camera observed and the learners.

use std::path::Path;

use es_assets::esscene::EsScene;
use es_assets::scene::{JointKind, SceneDesc};
use es_script::spec::{Clauses, CycleDoc, RewardDoc, Student, Teacher};

use super::edit::free_bodies;
use super::{add_clause, add_source, set_camera, TaskSpec, BONUS};
use crate::SCENE_FILE;

/// The control rate when nothing says one: SO-101's committed documents' (`task.toml`).
pub const CONTROL_HZ: f64 = 50.0;
/// "Say the task"'s time limit, seconds.
pub const TIMEOUT_S: f64 = 8.0;

/// The control rate of the Task IR at `task` (a template's), for "say the task" on its copy.
pub fn control_hz(task: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(task).ok()?;
    let task = es_ir::serial::task_from_toml(&text).ok()?;
    Some(f64::from(task.config.control_rate_hz))
}

/// What can be the robot: the scene document's includes, then the bodies hanging from the world
/// with a hinge or slide joint at or below them.
pub fn robots(scene: &SceneDesc, doc: &EsScene) -> Vec<String> {
    let mut out: Vec<String> = doc.includes.iter().map(|i| i.name.clone()).collect();
    let world = scene
        .bodies
        .iter()
        .find(|b| b.name == "world")
        .map(|b| b.id);
    for root in (scene.bodies.iter()).filter(|b| b.parent.is_none() || b.parent == world) {
        if Some(root.id) == world {
            continue;
        }
        // Bodies come parent first.
        let mut tree = std::collections::BTreeSet::from([root.id]);
        for b in &scene.bodies {
            if b.parent.is_some_and(|p| tree.contains(&p)) {
                tree.insert(b.id);
            }
        }
        let moves = |j: &&es_assets::scene::Joint| {
            tree.contains(&j.body) && matches!(j.kind, JointKind::Hinge | JointKind::Slide)
        };
        if scene.joints.iter().any(|j| moves(&j)) {
            out.push(root.name.clone());
        }
    }
    out
}

/// The robot of a new task: the scene document's first include, else the root of the body that
/// holds the first hinge or slide joint.
fn robot(scene: &SceneDesc, doc: &EsScene) -> String {
    if let Some(i) = doc.includes.first() {
        return i.name.clone();
    }
    let jointed =
        (scene.joints.iter()).find(|j| matches!(j.kind, JointKind::Hinge | JointKind::Slide));
    let mut at = jointed.and_then(|j| scene.bodies.iter().find(|b| b.id == j.body));
    // Bounded by the body count: an expanded scene has no cycle.
    for _ in 0..scene.bodies.len() {
        let up = (at.and_then(|b| b.parent)).and_then(|p| scene.bodies.iter().find(|b| b.id == p));
        match up {
            Some(p) if p.name != "world" => at = Some(p),
            _ => break,
        }
    }
    at.map(|b| b.name.clone()).unwrap_or_default()
}

/// "Say the task"'s learners (packet M17/G9; design note section 5.4, the owner's to change).
/// `[reward] scale` is plan H's (`rl_games`' `scale_value`), which the weight levels were read
/// against; the bonus is the medium level.
pub const REWARD_SCALE: f64 = 0.01;
/// The student's chunk: plan H's and plan N's 16 rows.
pub const HORIZON: u32 = 16;
/// The cycle's demonstrations (the cube cards' 200) and first collect seed — plan H's 1001,
/// clear of the evaluation's default seeds (101 on).
pub const EPISODES: u32 = 200;
pub const SEED: u64 = 1001;

/// The rows a chunk executes before the next is asked for: the largest of 1 to 10 that divides
/// `control_hz` (`control_hz / execute` must be whole, XIR-023) — 10 at 50 or 60 Hz, as the
/// committed students' 5 and 10 Hz replanning.
fn execute(control_hz: f64) -> u32 {
    (1..=10_u32)
        .rev()
        .find(|&d| (control_hz / f64::from(d)).fract() == 0.0)
        .unwrap_or(1)
}

/// What "say the task" adds so that `generate` writes the whole set (packet M17/G9): the
/// robot's joints, velocities and last action observed, the first free body's pose and velocity
/// for the teacher alone; a success bonus; the PPO teacher reading every channel (G3b's
/// defaults: plan H's preset); a camera student over every observed camera reading the joint
/// positions (`h3`, the `tanh` head); and a cycle the trained teacher demonstrates, its
/// successes kept.
fn learners(spec: &mut TaskSpec, scene: &SceneDesc) {
    for source in [
        "robot.joint_pos",
        "robot.joint_vel",
        "robot.previous_action",
    ] {
        add_source(spec, source, false);
    }
    if let Some(body) = free_bodies(scene).into_iter().next() {
        add_source(spec, &format!("{body}.pose"), true);
        add_source(spec, &format!("{body}.vel"), true);
    }
    spec.reward = Some(RewardDoc {
        scale: Some(REWARD_SCALE),
        success: Some(BONUS[1]),
        failure: None,
    });
    spec.teacher = Some(Teacher::default());
    spec.student = Some(Student {
        name: None,
        views: None,
        state: Some(vec!["joint_pos".to_owned()]),
        family: None,
        preset: None,
        horizon: HORIZON,
        execute: execute(spec.control_hz),
        training: None,
    });
    spec.cycle = Some(CycleDoc {
        runs: None,
        expert: None,
        episodes: EPISODES,
        seed: SEED,
        success_only: Some(true),
        nominal_only: None,
        jobs: None,
        preview: None,
        showcase: None,
    });
}

/// "Say the task": the robot, `control_hz`, 8 s, every camera observed at 96 px, and one clause
/// a fresh scene can satisfy — the first free body still (under 5 cm/s) — and the learners
/// ([`learners`]). The person edits from there.
pub fn new_spec(scene: &SceneDesc, doc: &EsScene, control_hz: f64) -> TaskSpec {
    let mut spec = TaskSpec {
        kind: "task-spec".to_owned(),
        schema: 1,
        scene: SCENE_FILE.to_owned(),
        robot: robot(scene, doc),
        control_hz,
        timeout_s: TIMEOUT_S,
        success: Clauses::default(),
        failure: None,
        start: None,
        observe: None,
        reward: None,
        teacher: None,
        student: None,
        deploy: None,
        evaluate: None,
        cycle: None,
    };
    add_clause(&mut spec, scene, false);
    for c in &scene.cameras {
        set_camera(&mut spec, &c.name, true);
    }
    learners(&mut spec, scene);
    spec
}
