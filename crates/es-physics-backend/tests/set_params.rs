//! Packet M11/X4 oracle 2: a drawn scale reaches one env's model and nothing else.
//!
//! `tests/fixtures/mjcf/set_params.xml` is a block that drops onto a floor and is pushed along it
//! by a position servo, loaded as a batch of two. Env 1 gets its body mass ×1.5, its geom
//! friction ×0.5 and the servo's gain ×1.2 through `PhysicsBackend::set_params`; env 0 gets
//! nothing. `MuJoCo` itself is the judge (spec 1.4): a Python script edits a fresh `MjModel`
//! the documented way and steps it, and 500 steps of each env must equal that run **bit for
//! bit** — env 1 the edited run, env 0 the unedited one. The same scales applied a second time
//! must give the same trajectory again: scales are relative to the loaded model, never to the
//! last edit.
//!
//! Needs a Python interpreter with `mujoco`; without one this prints `SKIP <test>: <why>`.
//!
//!     ES_PYTHON=$PWD/.venv/Scripts/python.exe cargo test -p es-physics-backend \
//!       --test set_params -- --nocapture

// Bitwise equality is the property under test.
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_physics_backend::{scene_to_mjcf, MjWarpBackend, MuJoCoCpuBackend, NewtonBackend};
use es_physics_core::backend::Param;
use es_physics_core::{Feature, LoadConfig, PhysicsBackend, PhysicsError};

const STEPS: u32 = 500;
const CTRL: f64 = 0.3;
const MASS: f64 = 1.5;
const FRICTION: f64 = 0.5;
const GAIN: f64 = 1.2;

/// The reference: the same MJCF, edited directly on an `MjModel` and stepped with `mj_step`.
/// Reads `timestep` and `edit` (0 or 1) on the first line and the MJCF after it, prints
/// `qpos` then `qvel` as `repr` floats, one per line.
const SCRIPT: &str = r#"
import sys
import mujoco

head = sys.stdin.readline().split()
timestep, edit = float(head[0]), int(head[1])
model = mujoco.MjModel.from_xml_string(sys.stdin.read())
model.opt.timestep = timestep
if edit:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "push")
    model.body_mass[body] *= 1.5
    model.geom_friction[geom, 0:3] *= 0.5
    model.actuator_gainprm[act, 0] *= 1.2
    model.actuator_biasprm[act, 1] *= 1.2
    # Mass feeds body_subtreemass, the invweights, actuator_acc0 and meaninertia.
    mujoco.mj_setConst(model, mujoco.MjData(model))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
data.ctrl[:] = 0.3
for _ in range(500):
    mujoco.mj_step(model, data)
sys.stdout.write("".join("%r\n" % float(v) for v in list(data.qpos) + list(data.qvel)))
"#;

fn scene() -> SceneDesc {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/set_params.xml");
    let xml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    es_assets::parse_mjcf(&xml)
        .unwrap_or_else(|e| panic!("set_params.xml: {e}"))
        .scene
}

fn body(scene: &SceneDesc, name: &str) -> StableId {
    scene.bodies.iter().find(|b| b.name == name).unwrap().id
}

fn geom(scene: &SceneDesc, name: &str) -> StableId {
    scene
        .bodies
        .iter()
        .flat_map(|b| &b.geoms)
        .find(|g| g.name == name)
        .unwrap()
        .id
}

fn actuator(scene: &SceneDesc, name: &str) -> StableId {
    scene.actuators.iter().find(|a| a.name == name).unwrap().id
}

fn scales(scene: &SceneDesc) -> Vec<(Param, StableId, f64)> {
    vec![
        (Param::BodyMass, body(scene, "block"), MASS),
        (Param::GeomFriction, geom(scene, "block_geom"), FRICTION),
        (Param::ActuatorGain, actuator(scene, "push"), GAIN),
    ]
}

/// `qpos ++ qvel` of the reference run, edited or not.
fn reference(mjcf: &str, timestep: f64, edit: bool) -> Vec<f64> {
    let python = std::env::var("ES_PYTHON").unwrap_or_else(|_| "python".to_owned());
    let mut child = Command::new(python)
        .args(["-c", SCRIPT])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("the reference interpreter starts");
    let mut stdin = child.stdin.take().unwrap();
    write!(stdin, "{timestep:?} {}\n{mjcf}", u8::from(edit)).unwrap();
    drop(stdin);
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "the reference script failed");
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect()
}

/// One env's `qpos ++ qvel` after `set_params` (when `params` is given), a whole-batch reset,
/// a constant servo target and `STEPS` steps.
fn run(
    backend: &mut dyn PhysicsBackend,
    params: Option<&[(Param, StableId, f64)]>,
) -> [Vec<f64>; 2] {
    if let Some(params) = params {
        backend.set_params(&[1], params).unwrap();
    }
    backend.reset(None, None).unwrap();
    backend.set_ctrl(&[CTRL, CTRL]).unwrap();
    backend.step(STEPS).unwrap();
    let state = backend.state();
    [0, 1].map(|env| [state.qpos_of(env), state.qvel_of(env)].concat())
}

fn assert_bits(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: {g} != {w}");
    }
}

#[test]
fn set_params_reproduces_a_direct_mujoco_edit_bitwise() {
    if let Err(why) = MuJoCoCpuBackend::is_available() {
        println!("SKIP set_params_reproduces_a_direct_mujoco_edit_bitwise: {why}");
        return;
    }
    let scene = scene();
    let mjcf = scene_to_mjcf(&scene).unwrap();
    let mut backend = MuJoCoCpuBackend::new();
    assert!(backend.capabilities().has(Feature::ModelParams));
    let info = backend
        .load(
            &scene,
            &LoadConfig {
                n_envs: 2,
                ..LoadConfig::default()
            },
        )
        .unwrap();
    let timestep = info.rate.period_secs_f64();
    let unedited = reference(&mjcf, timestep, false);
    let edited = reference(&mjcf, timestep, true);
    assert_ne!(unedited, edited, "the scales must change the trajectory");

    // Before any `set_params`, both envs are the unedited model.
    let [env0, env1] = run(&mut backend, None);
    assert_bits("no params, env 0", &env0, &unedited);
    assert_bits("no params, env 1", &env1, &unedited);

    let params = scales(&scene);
    let [env0, env1] = run(&mut backend, Some(&params));
    assert_bits("env 0 (untouched)", &env0, &unedited);
    assert_bits("env 1 (edited)", &env1, &edited);
    // The bits themselves, so two platforms' runs can be compared line for line.
    let hex = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{:016x}", x.to_bits()))
            .collect::<Vec<_>>()
    };
    println!(
        "RAN set_params: env 0 {:?} env 1 {:?}",
        hex(&env0),
        hex(&env1)
    );

    // What the model now holds, read back out of it: nominal × scale, for env 1 only.
    let applied: &BTreeMap<(u32, Param, StableId), [f64; 2]> = backend.applied_params();
    assert_eq!(applied.len(), params.len());
    for (param, id, scale) in &params {
        let [nominal, value] = applied[&(1, *param, *id)];
        assert_eq!(value, nominal * scale, "{param:?}");
    }
    assert_eq!(
        applied[&(1, Param::BodyMass, body(&scene, "block"))][0],
        1.0
    );

    // The same scales again (a second reset of the same draw): no compounding.
    let [env0, env1] = run(&mut backend, Some(&params));
    assert_bits("env 0, second episode", &env0, &unedited);
    assert_bits("env 1, second episode", &env1, &edited);
}

#[test]
fn set_params_names_what_it_cannot_resolve() {
    if let Err(why) = MuJoCoCpuBackend::is_available() {
        println!("SKIP set_params_names_what_it_cannot_resolve: {why}");
        return;
    }
    let scene = scene();
    let mut backend = MuJoCoCpuBackend::new();
    backend
        .load(
            &scene,
            &LoadConfig {
                n_envs: 2,
                ..LoadConfig::default()
            },
        )
        .unwrap();
    let stranger = StableId::from_path("no.such.body");
    let err = backend
        .set_params(&[0], &[(Param::BodyMass, stranger, 1.1)])
        .unwrap_err();
    assert!(matches!(err, PhysicsError::Unsupported(_)), "{err:?}");
    let err = backend.set_params(&[2], &scales(&scene)).unwrap_err();
    assert!(err.to_string().contains("env 2"), "{err}");
    // A rejected call leaves nothing half-applied.
    assert!(backend.applied_params().is_empty());
}

/// Newton declares no model parameters, and says so by name rather than dropping the draw.
#[test]
fn newton_refuses_set_params_by_name() {
    let mut newton = NewtonBackend::new();
    assert!(!newton.capabilities().has(Feature::ModelParams));
    assert_eq!(
        newton.set_params(&[0], &scales(&scene())),
        Err(PhysicsError::Unsupported("set_params".to_owned()))
    );
}

fn max_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// `mjwarp` holds per-world model arrays (`mujoco_warp` 3.13 indexes them by
/// `worldid % shape[0]`), so it declares `ModelParams` too -- judged against `mujoco-cpu` at
/// its declared tier (spec 17.3: tolerance, never bitwise), and against itself for the envs a
/// draw did not touch. Needs a GPU with `mujoco_warp`; skips otherwise.
#[test]
fn mjwarp_set_params_tracks_mujoco_cpu() {
    for check in [
        MuJoCoCpuBackend::is_available(),
        MjWarpBackend::is_available(),
    ] {
        if let Err(why) = check {
            println!("SKIP mjwarp_set_params_tracks_mujoco_cpu: {why}");
            return;
        }
    }
    let scene = scene();
    let cfg = LoadConfig {
        n_envs: 2,
        ..LoadConfig::default()
    };
    let params = scales(&scene);

    let mut cpu = MuJoCoCpuBackend::new();
    cpu.load(&scene, &cfg).unwrap();
    let [cpu_plain, _] = run(&mut cpu, None);
    let [_, cpu_edited] = run(&mut cpu, Some(&params));

    let mut warp = MjWarpBackend::new();
    assert!(warp.capabilities().has(Feature::ModelParams));
    warp.load(&scene, &cfg).unwrap();
    let [warp_plain, _] = run(&mut warp, None);
    let [warp_env0, warp_env1] = run(&mut warp, Some(&params));
    let [again_env0, again_env1] = run(&mut warp, Some(&params));

    for (param, id, scale) in &params {
        let [nominal, value] = warp.applied_params()[&(1, *param, *id)];
        // The world holds `f32(nominal * scale)`.
        assert!(
            (value - nominal * scale).abs() <= 1e-6 * (nominal * scale).abs(),
            "{param:?}: {value} vs {nominal} x {scale}"
        );
    }
    let untouched = max_delta(&warp_env0, &warp_plain);
    let repeat = max_delta(&again_env1, &warp_env1).max(max_delta(&again_env0, &warp_env0));
    let to_edited = max_delta(&warp_env1, &cpu_edited);
    let to_plain = max_delta(&warp_env1, &cpu_plain);
    println!(
        "RAN mjwarp_set_params_tracks_mujoco_cpu: env 0 vs unedited {untouched:e}, second application {repeat:e}, env 1 vs cpu edited {to_edited:e}, vs cpu unedited {to_plain:e}"
    );
    assert!(untouched <= 1e-6, "env 0 moved: {untouched:e}");
    assert!(repeat <= 1e-6, "scales compounded: {repeat:e}");
    assert!(
        to_edited <= 1e-3,
        "env 1 is not the edited model: {to_edited:e}"
    );
    assert!(
        to_edited * 10.0 < to_plain,
        "the draw did not reach physics"
    );
}
