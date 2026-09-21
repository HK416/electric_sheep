//! Oracle 3 of packet `docs/packets/M8/S4d-reward-cone.md`: **the reach task runs**.
//!
//! The committed `tests/fixtures/rl/task-reach.toml` is driven on the demo scene through
//! `MuJoCo`, and its reward is judged against the backend's own `StateView`: the distance is
//! recomputed here from `xpos` and compared **bitwise** with what `Env::step` returned. Then
//! the cube is placed at the gripper and the next step must end the episode with
//! `Termination::Success` — the same `Compare` scores the bonus and ends the episode, so one
//! step proves both.
//!
//!     ES_PYTHON=$HOME/venvs/es/bin/python \
//!       cargo test -p es-env --test reach -- --ignored --nocapture
//!
//! No `mujoco` on this machine prints `SKIP <test>: <why>` and returns (spec 1.4).

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_env::scheduler::BatchDomains;
use es_env::{Env, Termination};
use es_ir::task::TaskIr;
use es_physics_backend::MuJoCoCpuBackend;
use es_physics_core::backend::{PhysicsBackend, StateView};

const SEED: u64 = 201;
/// What `task-reach.toml`'s success `Compare` reads.
const SUCCESS_M: f64 = 0.03;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    let path = repo_root().join(path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn scene() -> SceneDesc {
    es_assets::parse_mjcf(&read("tests/fixtures/mjcf/so101_pick_place.xml"))
        .expect("the demo scene parses")
        .scene
}

fn task_ir() -> TaskIr {
    es_ir::serial::task_from_toml(&read("tests/fixtures/rl/task-reach.toml"))
        .expect("the reach task document parses")
}

fn body_id(scene: &SceneDesc, name: &str) -> StableId {
    scene
        .bodies
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("the scene has a body named {name}"))
        .id
}

/// One body's world position out of the backend's own state, for env 0.
fn xpos_of(state: &StateView<'_>, row: u32) -> [f64; 3] {
    let at = row as usize * 3;
    [state.xpos[at], state.xpos[at + 1], state.xpos[at + 2]]
}

/// `||a - b||`, associated exactly the way `Norm { kind: L2 }` lowers it: the squares summed
/// in lane order, then one IEEE `sqrt` (spec 6.6, `DET-020`).
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    ((dx * dx + dy * dy) + dz * dz).sqrt()
}

// The bitwise claim is the property under test; these comparisons are deliberate.
#[allow(clippy::float_cmp)]
#[test]
#[ignore = "needs a Python with mujoco (ES_PYTHON); oracle tier"]
fn reach_task_executes() {
    if let Err(why) = MuJoCoCpuBackend::is_available() {
        println!("SKIP reach_task_executes: {why}");
        return;
    }
    let (scene, task) = (scene(), task_ir());
    let mut env = Env::new(
        &task,
        &scene,
        MuJoCoCpuBackend::new(),
        &BatchDomains::single_env(),
        SEED,
    )
    .expect("the reach task builds an env");
    let caps = env.backend().capabilities();
    assert!(
        caps.supports_state_get_set,
        "this oracle writes the cube's pose: {}",
        caps.name
    );

    let (cube, gripper) = (
        env.model().body[&body_id(&scene, "cube")].start,
        env.model().body[&body_id(&scene, "gripper")].start,
    );
    let nq = env.model().nq as usize;

    // Hold the arm where the reset put it: the cone, not the controller, is under test.
    let hold: Vec<f64> = env.backend().state().qpos_of(0)[..env.model().nu as usize].to_vec();

    let out = env.step(&hold).expect("one control step");
    let state = env.backend().state();
    let expected = -distance(xpos_of(&state, cube), xpos_of(&state, gripper));
    assert!(
        expected < -SUCCESS_M,
        "the cube starts on the table, not in the jaws: {expected}"
    );
    assert_eq!(
        out.rewards[0], expected,
        "the reward is not -||cube - gripper|| computed from the backend's own xpos"
    );
    assert!(!out.dones[0], "nothing has happened yet");

    // Put the cube where the gripper is. Its free joint's `qpos` *is* its pose, so this needs
    // no IK: seven values, position then quaternion.
    let at = xpos_of(&env.backend().state(), gripper);
    let mut qpos = env.backend().state().qpos_of(0).to_vec();
    let qvel = vec![0.0; env.model().nv as usize];
    qpos[6..9].copy_from_slice(&at);
    qpos[9..13].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    let placed = StateView {
        n_envs: 1,
        tick: env.tick(),
        qpos: &qpos,
        qvel: &qvel,
        act: &[],
        sensordata: &[],
        xpos: &[],
        xquat: &[],
    };
    env.backend_mut()
        .set_state(&placed)
        .expect("the cube's free joint is writable");
    assert_eq!(
        env.backend().state().qpos_of(0).len(),
        nq,
        "the write did not change the model"
    );

    let out = env.step(&hold).expect("the step after the cube was placed");
    let episode = out
        .episodes
        .first()
        .expect("a cube at the gripper ends the episode");
    assert_eq!(
        episode.termination,
        Termination::Success,
        "reward {:?}",
        out.rewards
    );
    // The bonus is on the same `Compare` the `Terminate` reads: the successful step scores
    // `1 - d`, which is positive for any `d` under 30 mm.
    assert!(
        out.rewards[0] > 0.0,
        "the success bonus did not score: {:?}",
        out.rewards
    );
    println!(
        "RAN reach_task_executes: reward {expected} then {:?}",
        out.rewards[0]
    );
}
