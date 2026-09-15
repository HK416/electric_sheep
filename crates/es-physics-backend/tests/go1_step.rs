//! Our `MuJoCo` CPU backend steps the Go1 scene the way `MuJoCo` Playground's training physics
//! does (packet M6/B1, spec 17.1, spec 1.4).
//!
//! The risk this test exists to pin (`docs/design/quadruped-track.md` section 3): a policy
//! trained in MJX under `iterations=1 ls_iterations=5 timestep=0.004 integrator=Euler` with
//! `eulerdamp` disabled is deployed by us through `parse_mjcf` -> `scene_to_mjcf` -> `MuJoCo`.
//! Everything that round trip drops is physics the policy never saw.
//!
//! Two tests, deliberately split by what they need:
//!
//! * [`the_emitted_mjcf_keeps_the_playground_option_block`] needs no Python and runs in PR
//!   CI. It is the `<option>` survival oracle: `ls_iterations` and `<flag eulerdamp>` were
//!   both dropped by `mjcf_out` before this packet.
//! * [`go1_stands_from_home_and_agrees_with_mujoco_directly`] is `#[ignore]` and needs an
//!   interpreter with `mujoco`. It steps `home` with a zero action (PD targets = `home`)
//!   through our backend, and steps the same XML through `mujoco` directly, and compares.
//!
//! ```text
//! ES_PYTHON=$HOME/venvs/es/bin/python cargo test -p es-physics-backend \
//!   --test go1_step -- --ignored --nocapture
//! ```

use std::io::Write as _;
use std::process::Command;

use es_assets::scene::SceneDesc;
use es_core::TickRate;
use es_physics_backend::{scene_to_mjcf, MuJoCoCpuBackend};
use es_physics_core::{LoadConfig, PhysicsBackend, StateView};

const FIXTURE: &str = "go1_primitives.xml";

/// `ctrl_dt = 0.02` over `sim_dt = 0.004`: five physics steps per control tick
/// (`docs/api-notes/mujoco-playground-quadruped.md` section 1).
const SUBSTEPS: u32 = 5;

/// Control ticks to run: 5 s at 50 Hz, 1250 physics steps. Long enough that a scene that does
/// not actually stand has fallen, short enough to stay a unit test.
const CONTROL_TICKS: u32 = 250;

/// Trunk height, m, under which the robot is not standing. `home` starts at 0.278; the PD
/// loop sags a little under gravity, and anything below this is a collapse, not a sag.
const STANDING_Z: f64 = 0.20;

fn fixture_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mjcf")
        .join(FIXTURE)
}

fn fixture_text() -> String {
    let path = fixture_path();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn scene() -> SceneDesc {
    es_assets::parse_mjcf(&fixture_text())
        .unwrap_or_else(|e| panic!("{FIXTURE}: {e}"))
        .scene
}

/// The `home` keyframe, read out of the fixture rather than transcribed: `qpos` is the reset
/// state and `ctrl` is the action's zero-point, so a copy of either here would be a second
/// source of truth (`crates/es-assets/tests/go1_provenance.rs` checks the fixture's own
/// against upstream).
fn home() -> (Vec<f64>, Vec<f64>) {
    let text = fixture_text();
    let values = |key: &str| -> Vec<f64> {
        let after = text.split_once(key).expect("the `home` key is present").1;
        after
            .split_once('"')
            .expect("an opening quote")
            .1
            .split_once('"')
            .expect("a closing quote")
            .0
            .split_whitespace()
            .map(|t| t.parse().expect("a number"))
            .collect()
    };
    let qpos = values("<key name=\"home\" qpos=");
    let ctrl = values(" ctrl=");
    assert_eq!((qpos.len(), ctrl.len()), (19, 12), "`home` is 7 + 12 / 12");
    (qpos, ctrl)
}

// --- the no-Python half -------------------------------------------------------------------

/// The solver block the policy trains under has to survive `SceneDesc` and come back out the
/// other side. Before packet M6/B1, `ls_iterations` had no field to survive in and
/// `<flag eulerdamp="disable"/>` was warned about and dropped -- `MuJoCo` would then have used
/// its own defaults (50 line-search iterations, implicit damping), which is not the physics
/// the policy trained in.
#[test]
fn the_emitted_mjcf_keeps_the_playground_option_block() {
    let mjcf = scene_to_mjcf(&scene()).expect("no geom, actuator or sensor is Unsupported");
    for wanted in [
        "timestep=\"0.004\"",
        "integrator=\"Euler\"",
        "iterations=\"1\"",
        "ls_iterations=\"5\"",
        "cone=\"pyramidal\"",
        "<flag eulerdamp=\"disable\"/>",
    ] {
        assert!(
            mjcf.contains(wanted),
            "the re-emitted MJCF lost {wanted}:\n{}",
            mjcf.lines().take(6).collect::<Vec<_>>().join("\n")
        );
    }
    // The primitives-only point of the fixture, restated on the far side of the round trip.
    assert!(!mjcf.contains("mesh"), "the emitted MJCF references a mesh");
    assert!(!mjcf.contains("<include"));
    for name in ["trunk", "FR_calf_joint", "RL_hip", "floor"] {
        assert!(mjcf.contains(name), "the emitted MJCF lost `{name}`");
    }
    println!("RAN the_emitted_mjcf_keeps_the_playground_option_block");
}

// --- the MuJoCo half ----------------------------------------------------------------------

/// `mujoco` stepping `go1_primitives.xml` straight off disk: the reference half of the
/// comparison, with none of our `SceneDesc` round trip in it.
const DIRECT: &str = r#"
import json, sys, time
import mujoco

path, steps, = sys.argv[1], int(sys.argv[2])
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, key)
d.ctrl[:] = m.key_ctrl[key]
t0 = time.perf_counter()
for _ in range(steps):
    mujoco.mj_step(m, d)
seconds = time.perf_counter() - t0
print(json.dumps({
    "qpos": d.qpos.tolist(),
    "qvel": d.qvel.tolist(),
    "seconds": seconds,
    "timestep": m.opt.timestep,
    "iterations": int(m.opt.iterations),
    "ls_iterations": int(m.opt.ls_iterations),
    # mjENBL/mjDSBL_EULERDAMP is bit 1<<2 of disableflags.
    "eulerdamp": not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
}))
"#;

/// One field of the JSON [`DIRECT`] prints, without a JSON dependency in this crate.
fn field<'a>(json: &'a str, name: &str) -> &'a str {
    let after = json
        .split_once(&format!("\"{name}\":"))
        .unwrap_or_else(|| panic!("the reference printed no \"{name}\": {json}"))
        .1
        .trim_start();
    // An array runs to its `]`; a scalar to the next separator. Cutting an array at its
    // first `,` would hand back one element and call it the whole `qpos`.
    let end = if after.starts_with('[') {
        after.find(']').map_or(after.len(), |i| i + 1)
    } else {
        after.find([',', '}']).unwrap_or(after.len() - 1)
    };
    after[..end].trim().trim_matches(['[', ']'].as_slice())
}

fn numbers(json: &str, name: &str) -> Vec<f64> {
    field(json, name)
        .split(',')
        .map(|t| t.trim().parse().expect("a number"))
        .collect()
}

/// `mujoco` run directly, or the reason it could not be.
fn run_direct(steps: u32) -> Result<String, String> {
    let dir = std::env::temp_dir().join("es-go1-step");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let script = dir.join("direct.py");
    let mut f = std::fs::File::create(&script).map_err(|e| e.to_string())?;
    f.write_all(DIRECT.as_bytes()).map_err(|e| e.to_string())?;
    drop(f);

    let candidates = match std::env::var("ES_PYTHON") {
        Ok(p) if !p.trim().is_empty() => vec![p],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    };
    let mut tried = Vec::new();
    for python in candidates {
        match Command::new(&python)
            .arg(&script)
            .arg(fixture_path())
            .arg(steps.to_string())
            .output()
        {
            Ok(out) if out.status.success() => {
                return Ok(String::from_utf8_lossy(&out.stdout).into_owned())
            }
            Ok(out) => tried.push(format!(
                "`{python}`: {}",
                String::from_utf8_lossy(&out.stderr).trim().to_owned()
            )),
            Err(e) => tried.push(format!("`{python}`: {e}")),
        }
    }
    Err(tried.join("; "))
}

/// The packet's headline oracle. Two claims, in order:
///
///  1. The scene **stands**: 1250 physics steps from `home` with the action at zero (the PD
///     targets are `home`'s own `ctrl`) leave no NaN and the trunk above [`STANDING_Z`]. A
///     model whose feet fall through the floor or whose gains are wrong fails here.
///  2. Our backend and `mujoco` **agree**. The tolerance is the backend's declared tier, not
///     bitwise: `MuJoCoCpuBackend` declares `DeterminismTier::PhysicsMeaning` (spec 4.3
///     forbids an external backend declaring tier 1), and the two paths are not the same
///     bytes -- ours goes through `SceneDesc`, which carries no geom `priority` and no
///     `<position inheritrange>`. The measured deviation is printed either way, because it,
///     not the threshold, is the number the track has to watch.
#[test]
#[ignore = "needs a Python with `mujoco`; run explicitly"]
fn go1_stands_from_home_and_agrees_with_mujoco_directly() {
    if let Err(reason) = MuJoCoCpuBackend::is_available() {
        println!("SKIP go1_step: {reason}");
        return;
    }
    let scene = scene();
    let (qpos, ctrl) = home();
    let steps = CONTROL_TICKS * SUBSTEPS;

    let mut backend = MuJoCoCpuBackend::new();
    let info = backend
        .load(
            &scene,
            &LoadConfig {
                n_envs: 1,
                // 1 / 0.004 s. Passing it explicitly is the point: a rate that disagreed with
                // `<option timestep>` would be a silent 5x in the control period.
                rate: Some(TickRate::hz(250)),
                seed: 1,
            },
        )
        .expect("the Go1 scene loads in MuJoCo");
    assert_eq!((info.nq, info.nv, info.nu), (19, 18, 12), "free base + 12");

    let qvel = vec![0.0; info.nv as usize];
    backend
        .reset(
            None,
            Some(&StateView {
                n_envs: 1,
                tick: es_core::PhysTick::ZERO,
                qpos: &qpos,
                qvel: &qvel,
                act: &[],
                sensordata: &[],
                xpos: &[],
                xquat: &[],
            }),
        )
        .expect("reset to `home`");
    backend.set_ctrl(&ctrl).expect("PD targets = `home`");

    let t0 = std::time::Instant::now();
    for tick in 0..CONTROL_TICKS {
        let report = backend.step(SUBSTEPS).expect("a step");
        assert!(
            report.failures.is_empty(),
            "control tick {tick}: {:?}",
            report.failures
        );
    }
    let ours_secs = t0.elapsed().as_secs_f64();

    let state = backend.state();
    assert!(state.is_finite(), "the state went non-finite");
    let ours: Vec<f64> = state.qpos_of(0).to_vec();
    assert!(
        ours[2] > STANDING_Z,
        "after {CONTROL_TICKS} control ticks the trunk is at z = {:.4} m, below {STANDING_Z}: \
         the Go1 does not stand under a zero action",
        ours[2]
    );

    // Spec 12.4: a wall-clock observation, not a `step/s` claim. This path is one JSON round
    // trip per step over a pipe and is the oracle, never a throughput number.
    println!(
        "go1_step: {steps} physics steps; MuJoCoCpuBackend {:.1} ms / 1000 steps",
        ours_secs * 1000.0 * 1000.0 / f64::from(steps)
    );

    let json = match run_direct(steps) {
        Ok(json) => json,
        Err(why) => {
            println!("SKIP go1_step (direct mujoco): {why}");
            return;
        }
    };
    // MuJoCo read the training solver settings out of our fixture, not its own defaults.
    assert_eq!(field(&json, "iterations"), "1");
    assert_eq!(field(&json, "ls_iterations"), "5");
    assert_eq!(field(&json, "eulerdamp"), "false");
    assert_eq!(field(&json, "timestep"), "0.004");

    let theirs = numbers(&json, "qpos");
    assert_eq!(theirs.len(), ours.len());
    assert!(
        theirs[2] > STANDING_Z,
        "the reference run does not stand either: z = {:.4}",
        theirs[2]
    );
    let worst = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "go1_step: direct mujoco {:.1} ms / 1000 steps; worst |dqpos| after {steps} steps = \
         {worst:.3e}",
        numbers(&json, "seconds")[0] * 1000.0 * 1000.0 / f64::from(steps)
    );
    // `PhysicsMeaning`: the same gait, the same pose, not the same bits. A millimetre over
    // 5 s of standing is well inside "the policy would not notice" and far outside the
    // divergence a dropped solver option produces (which shows up as centimetres, or a fall).
    assert!(
        worst < 1e-3,
        "our backend and `mujoco` disagree by {worst:.3e} on qpos after {steps} steps, which \
         is more than DeterminismTier::PhysicsMeaning allows:\nours   {ours:?}\ntheirs {theirs:?}"
    );
    println!("RAN go1_stands_from_home_and_agrees_with_mujoco_directly");
}
