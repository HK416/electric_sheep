//! Packet M17/R8 (design note `scene-authoring.md` section 4.8), on GV's real evaluation: 16
//! attempts, 12 successes and 4 timeouts. Each timeout is explained by a success clause that is
//! false on its end row, and on the successes every success clause holds there.
//!
//! The owner's project is read, never written; elsewhere the test says so and passes. Its
//! trajectories predate the end row (review M17 F-7), so they are read on their last recorded row
//! and the explanation says so. `ES_GV_RERUN` names a copy of the project whose `runs/001/eval`
//! was evaluated again since, with end rows: there every success holds every success clause.

use std::path::Path;

use es_editor_scene::missing::explain;
use es_env::plan::eval_on_row;
use es_env::traj::Trajectory;
use es_eval::episodes::read_episodes;
use es_eval::run_dir::RunDir;
use es_ir::task::TerminationKind;
use es_script::spec::{compile_clauses, load_scene, TaskSpec};

/// GV's project (review M17), the owner's.
const GV: &str = "C:/Users/User/Documents/Electric Sheep/push-box";

fn check(root: &Path, end_rows: bool) {
    let eval = root.join("runs/001/eval");
    let run = RunDir::open(&eval).expect("the evaluation");
    let rows = read_episodes(&eval).expect("episodes.json").expect("rows");
    let ended = |how: &'static str| rows.iter().filter(move |r| r.termination == how);
    assert_eq!(rows.len(), 16);
    assert_eq!(ended("success").count(), 12, "{rows:?}");
    assert_eq!(ended("timeout").count(), 4, "{rows:?}");

    let e = explain(root, &run, &rows).expect("the run's own task explains it");
    assert_eq!((e.failed, e.before_end), (4, !end_rows));
    for (cell, why) in &e.cells {
        println!("{cell}: {why:?}");
    }
    for line in &e.lines {
        println!(
            "{:?} {} of {}: {:?}",
            line.at, line.attempts, e.failed, line.sentence
        );
    }
    for r in ended("timeout") {
        let why = &e.cells[&r.cell];
        assert!(why.iter().any(|a| !a.failure), "{}: {why:?}", r.cell);
    }

    // The successes, on the same end rows: every success clause holds there.
    let text = std::fs::read_to_string(root.join("task.estask")).expect("task.estask");
    let spec = TaskSpec::from_toml(&text).expect("reads");
    let (task, clauses) = compile_clauses(&spec, root).expect("compiles");
    let (scene, _) = load_scene(&root.join(&spec.scene)).expect("the scene");
    let model = es_physics_backend::layout(&scene);
    let nodes: Vec<_> = (clauses.iter())
        .filter(|c| c.0 == TerminationKind::Success)
        .map(|c| c.2)
        .collect();
    let mut short = Vec::new();
    for r in ended("success") {
        let traj = Trajectory::read(&run.traj_path(&r.cell)).expect("the trajectory");
        let last = traj.ticks() - 1;
        let truth = eval_on_row(&task, &scene, &model, &nodes, &traj, last).expect("evaluates");
        println!(
            "{} ({} rows, {} steps): {truth:?}",
            r.cell,
            traj.ticks(),
            r.steps
        );
        // A predicate is exactly 1.0 or 0.0.
        if truth.contains(&0.0) {
            short.push(r.cell.clone());
        }
    }
    if end_rows {
        assert!(short.is_empty(), "successes short of a clause: {short:?}");
    } else {
        // The last recorded row is the state before the step that succeeded.
        println!("on the row before the end, short of a success clause: {short:?}");
    }
}

#[test]
fn gvs_timeouts_are_explained_by_a_missing_success_clause() {
    let root = Path::new(GV);
    if !root.join("runs/001/eval/episodes.json").is_file() {
        println!("skipped: {GV} is not on this machine");
        return;
    }
    check(root, false);
}

#[test]
fn gvs_evaluation_again_ends_on_its_end_rows() {
    let Some(root) = std::env::var_os("ES_GV_RERUN") else {
        println!("skipped: ES_GV_RERUN names no re-evaluated copy of GV's project");
        return;
    };
    check(Path::new(&root), true);
}
