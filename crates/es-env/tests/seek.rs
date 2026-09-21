//! Oracle 1 of packet `docs/packets/M7/T8-episode-seek.md`: **a seek is the replay**.
//!
//! An `Env` told to be at episode `k` must produce, from there, the bytes the sequential run
//! produces — `Env::reset` keys the task's own `RandomizationPlan` by
//! `(seed, env, episode, stream)` (spec 6.3), so the claim is that the episode counter is the
//! *whole* of what a reset carries forward. What makes it true on this backend is that
//! `MuJoCoCpuBackend::reset` runs `mj_resetData` before it writes the state
//! (`python/mujoco_ref.py`); this file is what proves it rather than assuming it.
//!
//! Judged on the demo scene and the committed Task IR, for `k in {1, 3, 7}`:
//! the `StateView` after the reset (`qpos`, `qvel`), the episode's `ParamScales`, and the
//! whole `.estraj` of an episode driven by `ScriptedExpert` — all bitwise.
//!
//!     ES_PYTHON=$HOME/venvs/es/bin/python \
//!       cargo test -p es-env --test seek -- --ignored --nocapture
//!
//! No `mujoco` on this machine prints `SKIP <test>: <why>` and returns (spec 1.4).

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_env::expert::{demo_cfg, state_of_row, ScriptedExpert};
use es_env::scheduler::BatchDomains;
use es_env::traj::Trajectory;
use es_env::Env;
use es_physics_backend::MuJoCoCpuBackend;
use es_physics_core::backend::PhysicsBackend;

/// The episodes this oracle seeks to. `0` is excluded on purpose: it is what `Env::new`
/// already draws, so it would pass without a seek having done anything.
const EPISODES: [u64; 3] = [1, 3, 7];

/// Control steps of the expert-driven episode compared as `.estraj`. Long enough to reach
/// the approach and the grasp, short enough to keep the oracle a few seconds per `k`.
const STEPS: u32 = 60;

const SEED: u64 = 101;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scene() -> SceneDesc {
    let path = repo_root().join("tests/fixtures/mjcf/so101_pick_place.xml");
    let xml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    es_assets::parse_mjcf(&xml)
        .expect("the demo scene parses")
        .scene
}

/// The demo's committed Task IR, which owns the cube's randomization and the reset draws.
fn task_ir() -> es_ir::task::TaskIr {
    let path = repo_root().join("tests/fixtures/visible-learning/task.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    es_ir::serial::task_from_toml(&text).expect("the demo task document parses")
}

fn joint_id(scene: &SceneDesc, name: &str) -> StableId {
    scene
        .joints
        .iter()
        .find(|j| j.name == name)
        .unwrap_or_else(|| panic!("the demo scene has a joint named {name}"))
        .id
}

/// What one episode of one env is, as bytes: the drawn state, the scales it was drawn with,
/// and the trajectory the expert produced from it.
struct Drawn {
    qpos: Vec<u64>,
    qvel: Vec<u64>,
    scales: Vec<(String, u64)>,
    traj: Vec<u8>,
}

/// Draws episode `episode` of `seed` and rolls it out.
///
/// `seek` chooses *how* the env gets there: `true` seeks a fresh env straight to `episode`,
/// `false` replays `episode` resets from a fresh env. Both then run the same expert.
fn drawn(scene: &SceneDesc, task: &es_ir::task::TaskIr, episode: u64, seek: bool) -> Drawn {
    let domains = BatchDomains::single_env();
    let mut env: Env<MuJoCoCpuBackend> =
        Env::new(task, scene, MuJoCoCpuBackend::new(), &domains, SEED)
            .expect("the demo task builds an env");
    // `Env::new` drew episode 0 already, so the replay owes `episode` more resets and the
    // seek owes exactly one.
    if seek {
        env.seek_episode(0, episode).expect("nothing is open yet");
        env.reset(Some(&[0])).expect("reset");
    } else {
        for _ in 0..episode {
            env.reset(Some(&[0])).expect("reset");
        }
    }

    let state = env.backend().state();
    let qpos = state.qpos_of(0).iter().map(|v| v.to_bits()).collect();
    let qvel = state.qvel_of(0).iter().map(|v| v.to_bits()).collect();
    let scales = env
        .open_episode(0)
        .param_scales
        .iter()
        .map(|((param, id), v)| (format!("{param:?}/{id}"), v.to_bits()))
        .collect();

    let model = env.model().clone();
    let mut expert = ScriptedExpert::new(scene, demo_cfg(joint_id(scene, "cube_free")))
        .expect("the demo scene builds an expert");
    let mut traj = Trajectory::new(&model);
    for _ in 0..STEPS {
        let row: Vec<f64> = {
            let state = env.backend().state();
            let mut row = state.qpos_of(0).to_vec();
            row.extend_from_slice(state.qvel_of(0));
            row
        };
        traj.push(&model, &env.backend().state(), 0).expect("push");
        let Some(ctrl) = expert.action(&model, &state_of_row(&model, &row), 0) else {
            break;
        };
        if !env.step(&ctrl).expect("step").episodes.is_empty() {
            break;
        }
    }
    Drawn {
        qpos,
        qvel,
        scales,
        traj: traj.to_bytes(),
    }
}

#[test]
#[ignore = "needs a Python with mujoco (ES_PYTHON); oracle tier"]
fn seek_is_the_replay() {
    if let Err(why) = MuJoCoCpuBackend::is_available() {
        println!("SKIP seek_is_the_replay: {why}");
        return;
    }
    let (scene, task) = (scene(), task_ir());
    for k in EPISODES {
        let replayed = drawn(&scene, &task, k, false);
        let seeked = drawn(&scene, &task, k, true);
        assert_eq!(replayed.qpos, seeked.qpos, "episode {k}: qpos after reset");
        assert_eq!(replayed.qvel, seeked.qvel, "episode {k}: qvel after reset");
        assert_eq!(replayed.scales, seeked.scales, "episode {k}: param scales");
        assert!(
            !replayed.traj.is_empty(),
            "episode {k}: the rollout recorded nothing"
        );
        assert_eq!(
            replayed.traj, seeked.traj,
            "episode {k}: the .estraj of the expert-driven episode"
        );
        // Non-vacuity: a different episode is a different draw, so the oracle is comparing
        // something that actually moves.
        let other = drawn(&scene, &task, k + 1, true);
        assert_ne!(
            replayed.qpos,
            other.qpos,
            "episode {k} and {} drew the same state",
            k + 1
        );
        println!(
            "seek_is_the_replay: episode {k} identical ({} traj bytes)",
            replayed.traj.len()
        );
    }
    println!("RAN seek_is_the_replay");
}
