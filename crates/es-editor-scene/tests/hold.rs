//! Packet M18/K7 oracle 3 (design note `scene-authoring.md` section 4.9), on `mujoco-cpu`: GV's
//! scene as ① builds it (the empty project, the library's SO-101 where the view of the origin
//! looks, a 5 cm box) and "say the task"'s "[box] is still", the box dropped turned at random from
//! 15 cm. Held for no time, success fires the first control tick the box is slow; held for 1 s
//! (50 ticks at 50 Hz) it fires only once the box has been slow on each of the last 50 ticks.
//! Measured (seeds 7 and 9): the box is slow for one tick on landing, at once succeeds there, and
//! the box then bounces back up past the bound; held, it succeeds 53 and 51 ticks later.
//! Needs `ES_PYTHON` (the backend's `MuJoCo`); without it the test says so and passes.

use std::path::{Path, PathBuf};

use es_assets::esscene::ShapeDoc;
use es_core::time::TickRate;
use es_editor_scene::sentence as s;
use es_editor_scene::{
    add, make_editable, new_body, BackendKind, Camera, Command, Item, Record, SceneModel,
};
use es_env::{BatchDomains, Env, Termination};
use es_ir::task::TaskIr;
use es_physics_backend::MuJoCoCpuBackend;
use es_script::spec::{compile_task, load_scene, Draw, Start, StartItem, TaskSpec};

/// `still`'s bound: "say the task"'s, SO-101's settling bound (m/s).
const SPEED: f64 = 0.05;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// GV's scene, saved, and "say the task" on it with the box dropped as `start` says.
fn project(root: &Path) -> (SceneModel, TaskSpec) {
    let empty = repo().join("tests/fixtures/esscene/empty.esscene");
    make_editable(root, &empty, None).unwrap();
    let mut m = SceneModel::open(root, vec![BackendKind::MuJoCoCpu]).unwrap();
    let lib = add::library(&repo()).unwrap();
    let so101 = (lib.into_iter().find(|r| r.id == "so101")).expect("the library's SO-101");
    let origin = Camera {
        eye: [0.6, -0.6, 0.8],
        look_at: [0.0; 3],
        fov_y: std::f64::consts::FRAC_PI_4,
        width: 640,
        height: 400,
    };
    m.add(&Item::Robot(so101), &origin, true).unwrap();
    let mut cube = new_body("box", ShapeDoc::Box([0.025; 3]));
    cube.pos = Some([0.22, 0.0, 0.025]);
    m.apply(&Command::Add(Record::Body(cube))).unwrap();
    let mut spec = s::new_spec(m.scene(), m.doc(), 50.0);
    assert_eq!(spec.success.clauses.len(), 1, "the box is still");
    // Out of the arm's way, 15 cm up, turned at random: it lands on an edge or a corner.
    let at = |what: &str, value| StartItem {
        what: what.to_owned(),
        value: Some(value),
        noise: None,
        range: None,
        draw: None,
        tilt: None,
        tilt_max_deg: None,
        coupled: None,
        dice: None,
        stream: None,
    };
    let turned = StartItem {
        draw: Some(Draw::Any),
        value: None,
        ..at("box.orientation", 0.0)
    };
    spec.start = Some(Start {
        strength: None,
        zero_unset: None,
        items: vec![
            at("box.x", 0.4),
            at("box.y", 0.4),
            at("box.z", 0.15),
            turned,
        ],
    });
    m.apply(&Command::Spec(Some(Box::new(spec.clone()))))
        .unwrap();
    m.save().unwrap();
    (m, spec)
}

/// One episode on `mujoco-cpu` with the arm told to stay at zero: how it ended, at which step,
/// and the box's speed after every step (the state the step's termination was decided on).
fn episode(root: &Path, spec: &TaskSpec, seed: u64) -> (Termination, usize, Vec<f64>) {
    let task: TaskIr = compile_task(spec, root).unwrap();
    let (scene, _) = load_scene(&root.join(&spec.scene)).unwrap();
    let joint = (scene.joints.iter())
        .find(|j| j.name.starts_with("box"))
        .unwrap();
    let rate = TickRate::from_period_secs(scene.options.timestep).unwrap();
    let domains = BatchDomains::single_env_at(rate, TickRate::hz(50)).unwrap();
    let mut env = Env::new(&task, &scene, MuJoCoCpuBackend::new(), &domains, seed).unwrap();
    let lane = env.model().dof[&joint.id].start as usize;
    let ctrl = vec![0.0; env.model().nu as usize];
    let speed = |qvel: &[f64]| (qvel[lane..lane + 3].iter().map(|x| x * x).sum::<f64>()).sqrt();
    loop {
        let out = env.step(&ctrl).unwrap();
        let Some(ep) = out.episodes.first() else {
            continue;
        };
        // Row `i` is the state step `i` was entered with: after step `i` is row `i + 1`, and
        // after the last step the terminal state.
        let nv = ep.shape.nv;
        let mut speeds: Vec<f64> = (1..ep.steps())
            .map(|i| speed(&ep.qvel[i * nv..(i + 1) * nv]))
            .collect();
        speeds.push(speed(env.terminal_state().qvel_of(0)));
        return (ep.termination, ep.steps(), speeds);
    }
}

#[test]
fn a_dropped_box_succeeds_at_once_when_slow_and_held_only_once_it_rests() {
    if std::env::var_os("ES_PYTHON").is_none() {
        println!("SKIP a_dropped_box_succeeds_at_once_when_slow_and_held_only_once_it_rests: ES_PYTHON is not set");
        return;
    }
    let root = std::env::temp_dir().join(format!("es-k7-{}-drop", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let (_m, spec) = project(&root);
    let mut held = spec.clone();
    s::set_hold(&mut held, false, Some(1.0));

    let mut bounced = 0;
    for seed in [7, 8, 9] {
        let (now, at, trace) = episode(&root, &spec, seed);
        let (later, at_held, speeds) = episode(&root, &held, seed);
        let shown: Vec<String> = speeds.iter().map(|v| format!("{v:.3}")).collect();
        println!(
            "seed {seed}: at once {now:?} at step {at}, held {later:?} at step {at_held}\n  {}",
            shown.join(" ")
        );
        // The hold changes when the episode ends, not the physics.
        assert_eq!(trace[..], speeds[..at], "seed {seed}");
        assert_eq!((now, later), (Termination::Success, Termination::Success));
        assert!(trace[at - 1] < SPEED && trace[..at - 1].iter().all(|v| *v >= SPEED));
        // Held: slow on each of the last 50 steps, not on the one before them.
        assert!(at_held > 50, "seed {seed}");
        assert!(
            speeds[at_held - 50..].iter().all(|v| *v < SPEED),
            "seed {seed}"
        );
        assert!(speeds[at_held - 51] >= SPEED, "seed {seed}");
        let moved = speeds[at..at_held].iter().any(|v| *v >= SPEED);
        println!("  moved again after the instantaneous success: {moved}");
        bounced += usize::from(moved);
    }
    // Measured: seeds 7 and 9 are slow for a tick at a bounce and fast again after it.
    assert!(
        bounced > 0,
        "no drop bounced after its instantaneous success"
    );
    let _ = std::fs::remove_dir_all(&root);
}
