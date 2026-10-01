//! Packet M17/R8 (design note `scene-authoring.md` section 4.8), on GV's real evaluation: 16
//! attempts, 12 successes and 4 timeouts. Each timeout is explained by a success clause that is
//! false on its end row, and on the successes every success clause holds there.
//!
//! The owner's project is read, never written; elsewhere the test says so and passes. Its
//! trajectories predate the end row (review M17 F-7), so they are read on their last recorded row
//! and the explanation says so. `ES_GV_RERUN` names a copy of the project whose `runs/001/eval`
//! was evaluated again since, with end rows: there every success holds every success clause.
//!
//! Packet M18/K7 (section 4.9) on a scripted run: a success section held for 1 s explains an
//! attempt that ended with every success clause true by how long they had held.

use std::collections::BTreeMap;
use std::path::Path;

use es_assets::scene::JointKind;
use es_editor_scene::missing::explain;
use es_editor_scene::sentence::{self as s, At};
use es_editor_scene::{make_editable, BackendKind, Command, SceneModel};
use es_env::plan::eval_on_row;
use es_env::traj::Trajectory;
use es_eval::episodes::read_episodes;
use es_eval::episodes::EpisodeRow;
use es_eval::run_dir::RunDir;
use es_ir::evaluation::EvaluationReport;
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

/// Packet M18/K7 oracle 4 (design note `scene-authoring.md` section 4.9): on the SO-101 copy, "say
/// the task" ("[cube] is still") held for 1 s at 50 Hz, and a run of three timeouts written by
/// hand, one row per control tick plus the end row. An attempt whose end row holds the clause is
/// explained by how long it had held, counted back from the end: 20 rows is 0.4 s, and a cube
/// still from the reset on held 30 ticks, 0.6 s, because row 0 (the reset) is no tick the env
/// evaluated. An attempt moving at the end is explained by the clause, as before.
#[test]
fn a_too_short_hold_is_explained_by_how_long_it_held() {
    let root = std::env::temp_dir().join(format!("es-k7-{}-held", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let scene_file = repo.join("tests/fixtures/esscene/so101_pick_place.esscene");
    make_editable(&root, &scene_file, None).unwrap();
    let mut m = SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).unwrap();
    let mut spec = s::new_spec(m.scene(), m.doc(), 50.0);
    s::set_hold(&mut spec, false, Some(1.0));
    m.apply(&Command::Spec(Some(Box::new(spec.clone()))))
        .unwrap();
    m.save().unwrap();

    // The run evaluated the generated teacher evaluation, so it is the saved task's.
    let generated = root.join("generated/evaluation-teacher.toml");
    let text = std::fs::read_to_string(generated).unwrap();
    let evaluation = es_ir::serial::evaluation_from_toml(&text).unwrap();
    let dir = root.join("runs/001/eval");
    std::fs::create_dir_all(dir.join("traj")).unwrap();
    let report = EvaluationReport {
        schema_version: 1,
        evaluation_hash: evaluation.evaluation_hash().unwrap(),
        execution_hash: [0; 32],
        cells: Vec::new(),
        acceptance: Vec::new(),
        passed: false,
        episodes: Vec::new(),
    };
    std::fs::write(
        dir.join("report.json"),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();

    // `.estraj` rows as `Trajectory::to_bytes` writes them: `qpos`, `qvel`, then each body's
    // position and `x y z w` quaternion. Only the cube's linear velocity is read.
    let (scene, _) = load_scene(&root.join(&spec.scene)).unwrap();
    let model = es_physics_backend::layout(&scene);
    let free = (scene.joints.iter())
        .find(|j| j.kind == JointKind::Free)
        .unwrap();
    let lane = model.dof[&free.id].start as usize;
    let write = |cell: &str, moving: &dyn Fn(usize) -> bool| {
        let mut bytes = b"ESTRAJ01".to_vec();
        let nbody = model.body.len() as u32;
        for n in [model.nq, model.nv, nbody, 31] {
            bytes.extend(n.to_le_bytes());
        }
        for id in model.body.keys() {
            bytes.extend(id.as_bytes());
        }
        for r in 0..31 {
            let mut qvel = vec![0.0; model.nv as usize];
            qvel[lane] = if moving(r) { 0.1 } else { 0.0 };
            let poses = (0..nbody).flat_map(|_| [0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
            let row = (vec![0.0; model.nq as usize].into_iter())
                .chain(qvel)
                .chain(poses);
            for x in row {
                bytes.extend(x.to_le_bytes());
            }
        }
        let path = dir.join("traj").join(format!("{cell}.estraj"));
        std::fs::write(path, bytes).unwrap();
    };
    write("nominal-00", &|r| r <= 10);
    write("nominal-01", &|_| false);
    write("nominal-02", &|r| r == 30);
    let row = |cell: &str| EpisodeRow {
        suite: "nominal".to_owned(),
        cell: cell.to_owned(),
        episode: 0,
        seed: 0,
        termination: "timeout".to_owned(),
        steps: 30,
        changed_steps: 0,
        histogram: BTreeMap::new(),
    };
    let rows = [row("nominal-00"), row("nominal-01"), row("nominal-02")];
    let run = RunDir::open(&dir).unwrap();

    let e = explain(&root, &run, &rows).expect("the run's own task explains it");
    assert_eq!((e.failed, e.before_end, e.hold_s), (3, false, Some(1.0)));
    let held = BTreeMap::from([
        ("nominal-00".to_owned(), 0.4),
        ("nominal-01".to_owned(), 0.6),
    ]);
    assert_eq!(e.held, held);
    let clause = At {
        failure: false,
        index: 0,
    };
    assert_eq!(e.cells["nominal-02"], [clause]);
    assert!(e.cells["nominal-00"].is_empty() && e.cells["nominal-01"].is_empty());
    assert_eq!(e.lines.len(), 1);
    assert_eq!((e.lines[0].at, e.lines[0].attempts), (clause, 1));
    let _ = std::fs::remove_dir_all(&root);
}
