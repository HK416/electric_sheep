//! `es scene simulate` (plan G, packet G4): a scene alone, dropped for some seconds with its
//! servos held, written as the `.estraj` the replay reads.
//!
//! The oracles: the CLI's file is, byte for byte, what stepping the same scene through
//! `MuJoCoCpuBackend` in process records; a cube dropped above a plane comes to rest on it; the
//! Shadow Hand held for three seconds stays finite with the cube on its palm (packet M16/H2b's
//! stability). The `MuJoCo` ones need `ES_PYTHON` with `mujoco` and print `SKIP` without.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use es_assets::scene::{ActuatorKind, ActuatorTarget};
use es_env::Trajectory;
use es_physics_backend::MuJoCoCpuBackend;
use es_physics_core::{LoadConfig, PhysicsBackend};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-scene-simulate-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn mujoco() -> bool {
    match MuJoCoCpuBackend::is_available() {
        Ok(()) => true,
        Err(reason) => {
            eprintln!("SKIP: {reason}");
            false
        }
    }
}

fn simulate(scene: &Path, args: &[&str], out: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_es"))
        .args(["scene", "simulate"])
        .arg(scene)
        .args(args)
        .arg("--out")
        .arg(out)
        .output()
        .unwrap()
}

fn ok(run: &Output) -> String {
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(
        run.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&run.stderr)
    );
    stdout
}

/// Oracle 1: the CLI writes what stepping the scene through `MuJoCoCpuBackend` in process
/// records -- the initial state, then one row per physics tick, every position servo held at
/// its joint's initial position -- to the byte.
#[test]
fn the_cli_trajectory_is_the_backend_stepped_in_process() {
    if !mujoco() {
        return;
    }
    let dir = scratch("oracle");
    let path = fixtures().join("mjcf/so101_pick_place.xml");
    let out = dir.join("arm.estraj");
    let stdout = ok(&simulate(&path, &["--seconds", "0.5"], &out));
    assert!(stdout.contains("101 tick(s)"), "{stdout}");

    let scene = es_tools::backend::load_scene(path.to_str().unwrap()).unwrap();
    let mut backend = MuJoCoCpuBackend::new();
    let model = backend.load(&scene, &LoadConfig::default()).unwrap();
    backend.reset(None, None).unwrap();
    let qpos0 = backend.state().qpos.to_vec();
    let mut ctrl = vec![0.0; model.nu as usize];
    for a in &scene.actuators {
        if let (ActuatorKind::Position { .. }, ActuatorTarget::Joint(j)) = (a.kind, a.target) {
            let at = model.qpos[&j].start as usize;
            ctrl[model.actuator[&a.id].start as usize] = a.gear[0] * qpos0[at];
        }
    }
    backend.set_ctrl(&ctrl).unwrap();
    let mut traj = Trajectory::new(&model);
    traj.push(&model, &backend.state(), 0).unwrap();
    // 0.5 s at the scene's 5 ms step.
    for _ in 0..100 {
        assert!(backend.step(1).unwrap().failures.is_empty());
        traj.push(&model, &backend.state(), 0).unwrap();
    }
    assert_eq!(std::fs::read(&out).unwrap(), traj.to_bytes());
}

/// Oracle 2: a cube let go 30 cm above a plane lands and rests on it.
#[test]
fn a_dropped_cube_comes_to_rest_on_the_plane() {
    if !mujoco() {
        return;
    }
    let dir = scratch("drop");
    let scene = dir.join("drop.xml");
    std::fs::write(
        &scene,
        r#"<mujoco model="drop">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="cube" pos="0 0 0.3">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="0.025 0.025 0.025" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>"#,
    )
    .unwrap();
    let out = dir.join("drop.estraj");
    ok(&simulate(
        &scene,
        &["--seconds", "2", "--ctrl", "zero"],
        &out,
    ));
    let traj = Trajectory::read(&out).unwrap();
    assert_eq!(traj.ticks(), 1001);
    let start = traj.qpos(0)[2];
    let (z, speed) = {
        let last = traj.ticks() - 1;
        let v = traj.qvel(last);
        (
            traj.qpos(last)[2],
            v.iter().map(|x| x * x).sum::<f64>().sqrt(),
        )
    };
    assert!((start - 0.3).abs() < 1e-12, "starts at {start}");
    assert!((z - 0.025).abs() < 2e-3, "rests at z = {z}");
    assert!(speed < 1e-2, "still moving at {speed}");
}

/// Oracle 3: the Shadow Hand held at its initial servo targets for three seconds stays finite,
/// and the cube stays on the palm, where the task's reward says it rests.
#[test]
fn the_held_shadow_hand_keeps_the_cube_on_its_palm() {
    if !mujoco() {
        return;
    }
    let dir = scratch("hand");
    let path = fixtures().join("mjcf/shadow_hand/shadow_hand_repose.xml");
    let out = dir.join("hand.estraj");
    ok(&simulate(&path, &["--seconds", "3"], &out));
    let traj = Trajectory::read(&out).unwrap();
    // 3 s at 1/120 s, and the initial state.
    assert_eq!(traj.ticks(), 361);
    for k in 0..traj.ticks() {
        assert!(traj
            .qpos(k)
            .iter()
            .chain(traj.qvel(k))
            .all(|v| v.is_finite()));
    }
    let scene = es_tools::backend::load_scene(path.to_str().unwrap()).unwrap();
    let cube = scene.bodies.iter().find(|b| b.name == "object").unwrap().id;
    let p = traj.poses(traj.ticks() - 1)[&cube].position;
    // `task-repose.toml`'s p_ref, where the cube rests on the palm (measured); the task fails
    // an episode at 0.24 m from it.
    let off = ((p.x - 1.0).powi(2) + (p.y - 0.867).powi(2) + (p.z - 0.1772).powi(2)).sqrt();
    assert!(
        off < 0.03,
        "the cube is {off} m from its rest on the palm: {p:?}"
    );
}

/// A spring so stiff (1e12 N/m on 1 kg, pulling 1 m) that the first step's `qacc` is past
/// `mjMAXVAL`: `MuJoCo` resets the state to its start by itself, which the backend reports as
/// divergence (packet M16/H2b). The run says when, keeps the tick before it, and fails.
#[test]
fn a_divergence_is_reported_with_its_time() {
    if !mujoco() {
        return;
    }
    let dir = scratch("diverge");
    let scene = dir.join("spring.xml");
    std::fs::write(
        &scene,
        r#"<mujoco model="spring">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="slider">
      <joint name="slide" type="slide" axis="1 0 0" stiffness="1e12" springref="1"/>
      <geom name="ball" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>"#,
    )
    .unwrap();
    let out = dir.join("spring.estraj");
    let run = simulate(&scene, &["--seconds", "1"], &out);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert_eq!(run.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("error: diverged at 0.002 s (tick 1): the backend reported NanDetected"),
        "{stderr}"
    );
    assert_eq!(Trajectory::read(&out).unwrap().ticks(), 1);
}

/// A scene the backend's mapping report blocks is refused before any process starts, naming
/// what blocks it; no Python is needed to say so.
#[test]
fn a_blocked_scene_is_refused_by_name() {
    let dir = scratch("blocked");
    let path = fixtures().join("mjcf/so101_pick_place.xml");
    let out = dir.join("x.estraj");
    let run = simulate(&path, &["--seconds", "1", "--backend", "newton"], &out);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert_eq!(run.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("newton cannot simulate this scene: "),
        "{stderr}"
    );
    assert!(!out.exists());
}

#[test]
fn bad_arguments_are_usage_errors() {
    let path = fixtures().join("mjcf/pendulum.xml");
    let out = scratch("usage").join("x.estraj");
    for args in [
        &["--ctrl", "hold"][..],
        &["--seconds", "-1"],
        &["--seconds", "1", "--ctrl", "loose"],
        &["--seconds", "1", "--backend", "bullet"],
    ] {
        let run = simulate(&path, args, &out);
        assert_eq!(run.status.code(), Some(2), "{args:?}");
    }
}
