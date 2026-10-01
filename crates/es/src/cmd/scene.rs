//! `es scene export` (plan G, packet G2; `docs/design/scene-authoring.md` section 3.3) and
//! `es scene simulate` (packet G4, section 5: ①'s physics preview).

use std::path::Path;

use es_assets::scene::{ActuatorKind, ActuatorTarget, SceneDesc, Tendon, TendonKind};
use es_core::StableId;
use es_env::Trajectory;
use es_physics_backend::{
    BackendKind, MjWarpBackend, MuJoCoCpuBackend, NewtonBackend, PhysXBackend,
};
use es_physics_core::{LoadConfig, ModelInfo, PhysicsBackend, StateView};

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es scene export <scene.xml|urdf|esscene> --mjcf <out.xml>

Writes the scene as one MJCF file a person opens in MuJoCo's viewer: bodies, joints, geoms,
sites, cameras, lights, materials and textures, meshes, tendons, actuators, sensors, contact
pairs and excludes, gravcomp and <option> -- everything the scene description carries, so that
reading the export back gives the same scene and the same scene_hash. Mesh and texture files
are written beside the XML at the scene's own relative paths (an absolute or `..` path goes
under mesh/ or texture/ instead); a file already there with other bytes is not overwritten.
What MJCF cannot say (a height field, a glTF-only material channel, a joint on the world) is
an error naming it.

es scene simulate <scene.xml|urdf|esscene> --seconds S --out <traj.estraj>
                  [--ctrl hold|zero] [--backend mujoco-cpu]

Drops the scene for S seconds and records it; no Task IR is needed. The scene starts from its
initial pose (every hinge and slide at 0, every free body where the scene puts it) with every
actuator held at its initial target (`hold`, the default: a position servo's target is its
joint's or fixed tendon's initial length, so it holds what it starts at; any other actuator 0)
or at 0 (`zero`), and is stepped round(S / timestep) times at its own timestep. One row per
physics tick, tick 0 being the initial state, goes into the .estraj (qpos, qvel and every
body's world pose) the editor's replay and `es render --traj` read; it plays at 1 / timestep.
A scene the backend's mapping report blocks is refused naming the blocking features. A
divergence (MuJoCo resetting a non-finite or runaway state to its start, which the backend
reports as NanDetected) ends the run: the ticks before it are written, and the exit code is 1.

Exit code: 0 written; 1 refused or diverged; 2 usage error; 3 the backend is not available on
this machine (nothing ran).
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("export") => export(&args[1..]),
        Some("simulate") => simulate(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es scene: unknown subcommand '{other}'\n\n{HELP}"
        ))),
    }
}

fn export(args: &[String]) -> Result<u8, CliError> {
    let (mut scene, mut out) = (None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(0);
            }
            "--mjcf" => out = it.next().cloned(),
            other if scene.is_none() && !other.starts_with("--") => scene = Some(other.to_owned()),
            other => {
                return Err(CliError::Usage(format!(
                    "es scene export: unexpected '{other}'\n\n{HELP}"
                )))
            }
        }
    }
    let (Some(scene_path), Some(out)) = (scene, out) else {
        return Err(CliError::Usage(HELP.to_owned()));
    };
    let scene = crate::cmd::backend::load_scene(&scene_path)?;
    let out = Path::new(&out);
    let dir = out
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let fail = |e: &dyn std::fmt::Display| CliError::Runtime(format!("{}: {e}", out.display()));
    std::fs::create_dir_all(dir).map_err(|e| fail(&e))?;
    let xml = es_assets::mjcf::write_mjcf(&scene, dir).map_err(|e| fail(&e))?;
    std::fs::write(out, xml).map_err(|e| fail(&e))?;
    println!(
        "wrote {} ({} bodies, {} joints, {} actuators, {} assets)",
        out.display(),
        scene.bodies.len(),
        scene.joints.len(),
        scene.actuators.len(),
        scene.assets.len()
    );
    println!("scene_hash: {}", hex(&scene.scene_hash()));
    Ok(0)
}

fn simulate(args: &[String]) -> Result<u8, CliError> {
    let usage = |m: &str| CliError::Usage(format!("es scene simulate: {m}\n\n{HELP}"));
    let (mut scene, mut seconds, mut out) = (None, None, None);
    let (mut hold, mut backend) = (true, BackendKind::MuJoCoCpu);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(0);
            }
            "--seconds" => seconds = it.next().and_then(|s| s.parse::<f64>().ok()),
            "--out" => out = it.next().cloned(),
            "--ctrl" => match it.next().map(String::as_str) {
                Some("hold") => hold = true,
                Some("zero") => hold = false,
                _ => return Err(usage("--ctrl is hold or zero")),
            },
            "--backend" => {
                let name = it.next().map_or("", String::as_str);
                backend = crate::cmd::eval::parse_backend(name, HELP)?;
            }
            other if scene.is_none() && !other.starts_with("--") => scene = Some(other.to_owned()),
            other => return Err(usage(&format!("unexpected '{other}'"))),
        }
    }
    let (Some(scene_path), Some(seconds), Some(out)) = (scene, seconds, out) else {
        return Err(usage("a scene, --seconds and --out are required"));
    };
    if !(seconds.is_finite() && seconds > 0.0) {
        return Err(usage("--seconds is a positive number"));
    }
    let scene = crate::cmd::backend::load_scene(&scene_path)?;
    let report = es_physics_backend::mapping_report(&scene, backend);
    if report.blocked {
        let names: Vec<String> = report.blocking().map(|r| r.feature.to_string()).collect();
        return Err(CliError::Runtime(format!(
            "{backend} cannot simulate this scene: {}",
            names.join(", ")
        )));
    }
    if let Err(reason) = es_physics_backend::is_available(backend) {
        eprintln!("SKIPPED: {backend} is not available on this machine: {reason}");
        return Ok(3);
    }
    let mut physics: Box<dyn PhysicsBackend> = match backend {
        BackendKind::MuJoCoCpu => Box::new(MuJoCoCpuBackend::new()),
        BackendKind::MjWarp => Box::new(MjWarpBackend::new()),
        BackendKind::Newton => Box::new(NewtonBackend::new()),
        BackendKind::PhysX => Box::new(PhysXBackend::new()),
    };
    let fail = |e: &dyn std::fmt::Display| CliError::Runtime(format!("{backend}: {e}"));
    let model = (physics.load(&scene, &LoadConfig::default())).map_err(|e| fail(&e))?;
    physics.reset(None, None).map_err(|e| fail(&e))?;
    let ctrl = if hold {
        held(&scene, &model, &physics.state())
    } else {
        vec![0.0; model.nu as usize]
    };
    physics.set_ctrl(&ctrl).map_err(|e| fail(&e))?;

    let hz = model.rate.as_hz_f64();
    let steps = (seconds * hz).round() as u64;
    let mut traj = Trajectory::new(&model);
    let mut diverged = None;
    for tick in 0..=steps {
        if tick > 0 {
            let report = physics.step(1).map_err(|e| fail(&e))?;
            if let Some((_, kind)) = report.failures.first() {
                diverged = Some((tick, *kind));
                break;
            }
        }
        (traj.push(&model, &physics.state(), 0)).map_err(|e| fail(&e))?;
    }
    let out = Path::new(&out);
    traj.write(out).map_err(|e| fail(&e))?;
    println!(
        "wrote {} ({} tick(s) of {:.3} ms, {} bodies, ctrl {})",
        out.display(),
        traj.ticks(),
        model.rate.period_secs_f64() * 1e3,
        traj.bodies().len(),
        if hold { "hold" } else { "zero" }
    );
    match diverged {
        // The time first, in seconds: the editor's preview line reads it after "diverged at ".
        Some((tick, kind)) => Err(CliError::Runtime(format!(
            "diverged at {:.3} s (tick {tick}): the backend reported {kind:?} (MuJoCo resets a \
             non-finite or runaway state to its start); the {} tick(s) before it are written",
            tick as f64 / hz,
            traj.ticks()
        ))),
        None => Ok(0),
    }
}

/// `--ctrl hold`: every position servo's target at its transmission's length in `state` (a
/// joint's position, or a fixed tendon's sum, times the gear), so it holds what it starts at;
/// every other actuator 0. In the backend's actuator order.
fn held(scene: &SceneDesc, model: &ModelInfo, state: &StateView<'_>) -> Vec<f64> {
    let qpos = state.qpos_of(0);
    let at = |joint: &StableId| {
        model
            .qpos
            .get(joint)
            .map_or(0.0, |r| qpos[r.start as usize])
    };
    let mut ctrl = vec![0.0; model.nu as usize];
    for a in &scene.actuators {
        let (ActuatorKind::Position { .. }, Some(slot)) = (a.kind, model.actuator.get(&a.id))
        else {
            continue;
        };
        let length = match a.target {
            ActuatorTarget::Joint(j) => at(&j),
            ActuatorTarget::Tendon(t) => match scene.tendons.iter().find(|x| x.id == t) {
                Some(Tendon {
                    kind: TendonKind::Fixed { joints },
                    ..
                }) => joints.iter().map(|(j, coef)| coef * at(j)).sum(),
                _ => 0.0,
            },
            ActuatorTarget::Site(_) => 0.0,
        };
        ctrl[slot.start as usize] = a.gear[0] * length;
    }
    ctrl
}
