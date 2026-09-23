//! Packet M11/X4 oracle 3: **a mass draw reaches each env's model at reset**.
//!
//! The committed reach task gains one `Randomization` node, `body.cube.mass Uniform(0.8, 1.2)`,
//! and runs over four envs on `MuJoCo`. After the reset `Env::new` performs, each env's cube
//! mass — read back out of that env's `MjModel` by the backend — must equal the loaded mass
//! times the scale the episode records, bit for bit; the envs must differ; and a second `Env`
//! on the same seed must draw and apply exactly the same masses.
//!
//! No `mujoco` on this machine prints `SKIP <test>: <why>` and returns (spec 1.4).

// Bitwise reproducibility is the property under test.
#![allow(clippy::float_cmp)]

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_env::scheduler::{BatchDomains, DomainCfg};
use es_env::Env;
use es_ir::graph::NodeId;
use es_ir::task::{Distribution, TaskIr, TaskNode};
use es_physics_backend::MuJoCoCpuBackend;
use es_physics_core::backend::Param;

const SEED: u64 = 7;
const N_ENVS: u32 = 4;

fn read(path: &str) -> String {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn scene() -> SceneDesc {
    es_assets::parse_mjcf(&read("tests/fixtures/mjcf/so101_pick_place.xml"))
        .expect("the demo scene parses")
        .scene
}

fn task() -> TaskIr {
    let mut task = es_ir::serial::task_from_toml(&read("tests/fixtures/rl/task-reach.toml"))
        .expect("the reach task document parses");
    task.graph.insert(
        NodeId(1000),
        TaskNode::Randomization {
            target: "body.cube.mass".to_owned(),
            dist: Distribution::Uniform { lo: 0.8, hi: 1.2 },
            stream: "cube.mass".to_owned(),
        },
    );
    task
}

fn domains() -> BatchDomains {
    BatchDomains {
        simulation: DomainCfg::new(N_ENVS, 1),
        observation: DomainCfg::new(N_ENVS, 1),
        inference: DomainCfg::new(N_ENVS, 1),
        training: None,
    }
}

/// Each env's `[nominal, applied]` cube mass after the reset `Env::new` performs, and the
/// scale each env's first episode recorded.
fn run(scene: &SceneDesc, cube: StableId) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut env = Env::new(&task(), scene, MuJoCoCpuBackend::new(), &domains(), SEED)
        .expect("the reach task with a mass draw builds an env");
    let applied = (0..N_ENVS)
        .map(|e| env.backend().applied_params()[&(e, Param::BodyMass, cube)])
        .collect();
    // The draw is recorded in the episode; one step and a reset close it and hand it back.
    let nu = env.model().nu as usize;
    env.step(&vec![0.0; N_ENVS as usize * nu]).unwrap();
    let episodes = env.reset(None).unwrap();
    assert_eq!(episodes.len(), N_ENVS as usize);
    let recorded = episodes
        .iter()
        .map(|ep| ep.param_scales[&(Param::BodyMass, cube)])
        .collect();
    (applied, recorded)
}

#[test]
fn randomized_params_reach_the_backend() {
    if let Err(why) = MuJoCoCpuBackend::is_available() {
        println!("SKIP randomized_params_reach_the_backend: {why}");
        return;
    }
    let scene = scene();
    let cube = scene.bodies.iter().find(|b| b.name == "cube").unwrap().id;
    let (applied, recorded) = run(&scene, cube);
    for (env, ([nominal, mass], scale)) in applied.iter().zip(&recorded).enumerate() {
        assert!((0.8..=1.2).contains(scale), "env {env}: scale {scale}");
        assert_eq!(
            *mass,
            nominal * scale,
            "env {env}: the model holds the draw"
        );
    }
    assert!(
        applied.windows(2).all(|w| w[0][1] != w[1][1]),
        "every env draws its own mass: {applied:?}"
    );

    // `seed` replays exactly: the same draws, applied to the same bits.
    let (again, recorded_again) = run(&scene, cube);
    assert_eq!(recorded, recorded_again);
    for (a, b) in applied.iter().zip(&again) {
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
    }
}
