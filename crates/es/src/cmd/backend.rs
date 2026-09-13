//! `es backend compare` (spec 17.2, 17.3, 14.4).

use es_assets::scene::SceneDesc;
use es_physics_backend::{
    compare_backends, mapping_report, BackendKind, MjWarpBackend, MuJoCoCpuBackend, NewtonBackend,
};
use es_physics_core::{LoadConfig, PhysicsBackend};

use crate::error::CliError;

const HELP: &str = "\
es backend compare --scene <file.xml|urdf> --backends mujoco-cpu,mjwarp[,newton,physx] [OPTIONS]
es backend compare --task <task.toml> --backends ...

Parses a scene (or a Task IR's SceneRef) and, for every requested backend, prints the
semantic-mapping report (spec 17.2) -- always, even when the backend is unavailable or not
yet implemented. Then, for every pair of requested backends that are both available and not
blocked by their mapping report, runs the same scene on both from a shared reset state and
prints a spec 3.5 tier 3 comparison (spec 14.4: a mapping blocked by severity error never
runs). `physx` has no adapter yet and prints a `not implemented (M2/M3)` row.

    --scene <path>     MJCF (.xml) or URDF (.urdf) scene file
    --task <path>      Task IR TOML; reads its SceneRef path (must not be hash-only)
    --backends <csv>   backend names, comma separated (mujoco-cpu, mjwarp, newton, physx)
    --ticks N          physics ticks to compare (default 100)
    --envs N           LoadConfig::n_envs (default 1)
    --seed S           LoadConfig::seed (default 0)
    --tol T            max |dqpos| above which the comparison fails (default 1e-6)
    --ctrl-random      seeded pseudo-random control sequence instead of all-zero

Exit code: 0 once every requested backend ran or was skipped for an environment reason
(unavailable / not implemented); 1 when a requested backend's mapping report is blocked
(spec 14.4) or a comparison exceeds --tol; 2 on a usage error (bad flags, unknown backend).
";

struct Args {
    scene: Option<String>,
    task: Option<String>,
    backends: Vec<BackendKind>,
    ticks: u32,
    envs: u32,
    seed: u64,
    tol: f64,
    ctrl_random: bool,
}

fn parse_args(args: &[String]) -> Result<Args, CliError> {
    let mut scene = None;
    let mut task = None;
    let mut backend_names: Option<String> = None;
    let mut ticks = 100u32;
    let mut envs = 1u32;
    let mut seed = 0u64;
    let mut tol = 1e-6f64;
    let mut ctrl_random = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{HELP}")))
        };
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(HELP.to_owned())),
            "--scene" => scene = Some(val()?.clone()),
            "--task" => task = Some(val()?.clone()),
            "--backends" => backend_names = Some(val()?.clone()),
            "--ticks" => {
                ticks = val()?
                    .parse()
                    .map_err(|e| CliError::Usage(format!("--ticks: {e}")))?;
            }
            "--envs" => {
                envs = val()?
                    .parse()
                    .map_err(|e| CliError::Usage(format!("--envs: {e}")))?;
            }
            "--seed" => {
                seed = val()?
                    .parse()
                    .map_err(|e| CliError::Usage(format!("--seed: {e}")))?;
            }
            "--tol" => {
                tol = val()?
                    .parse()
                    .map_err(|e| CliError::Usage(format!("--tol: {e}")))?;
            }
            "--ctrl-random" => ctrl_random = true,
            other => return Err(CliError::Usage(format!("unknown flag '{other}'\n\n{HELP}"))),
        }
    }

    let Some(names) = backend_names else {
        return Err(CliError::Usage(format!("--backends is required\n\n{HELP}")));
    };
    let mut backends = Vec::new();
    for name in names.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let kind = BackendKind::from_name(name)
            .ok_or_else(|| CliError::Usage(format!("unknown backend '{name}'\n\n{HELP}")))?;
        if !backends.contains(&kind) {
            backends.push(kind);
        }
    }
    if backends.is_empty() {
        return Err(CliError::Usage(format!("--backends is empty\n\n{HELP}")));
    }
    if scene.is_none() && task.is_none() {
        return Err(CliError::Usage(format!(
            "one of --scene or --task is required\n\n{HELP}"
        )));
    }

    Ok(Args {
        scene,
        task,
        backends,
        ticks,
        envs,
        seed,
        tol,
        ctrl_random,
    })
}

pub(crate) fn load_scene(path: &str) -> Result<SceneDesc, CliError> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    let is_urdf = std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("urdf"));
    if is_urdf {
        let resolver = es_assets::urdf::PackageResolver::from_env();
        let import = es_assets::urdf::parse_urdf(&raw, &resolver)
            .map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
        Ok(import.scene)
    } else {
        let import =
            es_assets::parse_mjcf(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
        Ok(import.scene)
    }
}

/// Resolves `--scene`/`--task` to a scene file path.
fn scene_path(a: &Args) -> Result<String, CliError> {
    if let Some(scene) = &a.scene {
        return Ok(scene.clone());
    }
    let task_path = a.task.as_ref().expect("checked in parse_args");
    let raw = std::fs::read_to_string(task_path)
        .map_err(|e| CliError::Runtime(format!("{task_path}: {e}")))?;
    let task = es_ir::serial::task_from_toml(&raw)
        .map_err(|e| CliError::Runtime(format!("{task_path}: {e}")))?;
    if task.scene.path.is_empty() {
        return Err(CliError::Runtime(format!(
            "{task_path}: SceneRef carries only a content hash, no path -- pass --scene <file.xml|urdf> instead"
        )));
    }
    Ok(task.scene.path)
}

/// A tiny seeded PRNG (splitmix64) for `--ctrl-random` -- deterministic, no `rand` dependency.
fn splitmix64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // map to [-1, 1]
    (z as f64 / u64::MAX as f64) * 2.0 - 1.0
}

fn make_backend(kind: BackendKind) -> Box<dyn PhysicsBackend> {
    match kind {
        BackendKind::MuJoCoCpu => Box::new(MuJoCoCpuBackend::new()),
        BackendKind::MjWarp => Box::new(MjWarpBackend::new()),
        BackendKind::Newton => Box::new(NewtonBackend::new()),
        BackendKind::PhysX => {
            unreachable!("physx is never in the `available` list")
        }
    }
}

/// `Ok(())` when the backend is implemented and its Python side is reachable; `Err(reason)`
/// otherwise, `physx` included (spec 17.1: no adapter until M3).
fn availability(kind: BackendKind) -> Result<(), String> {
    match kind {
        BackendKind::MuJoCoCpu => MuJoCoCpuBackend::is_available(),
        BackendKind::MjWarp => MjWarpBackend::is_available(),
        BackendKind::Newton => NewtonBackend::is_available(),
        BackendKind::PhysX => Err("not implemented (M2/M3)".to_owned()),
    }
}

fn ctrl_seq(nu: u32, n_envs: u32, n_ticks: u32, random: bool, seed: u64) -> Vec<Vec<f64>> {
    let len = (nu * n_envs) as usize;
    if len == 0 {
        return Vec::new();
    }
    if !random {
        return vec![vec![0.0; len]];
    }
    let mut state = seed ^ 0x2545_F491_4F6C_DD1D;
    (0..n_ticks)
        .map(|_| (0..len).map(|_| splitmix64(&mut state)).collect())
        .collect()
}

fn compare(args: &[String]) -> Result<u8, CliError> {
    let a = parse_args(args)?;
    let path = scene_path(&a)?;
    let scene = load_scene(&path)?;

    let reports: Vec<_> = a
        .backends
        .iter()
        .map(|&kind| mapping_report(&scene, kind))
        .collect();
    for report in &reports {
        println!("{report}\n");
    }
    let blocked_any = reports.iter().any(|r| r.blocked);

    let mut available = Vec::new();
    for (&kind, report) in a.backends.iter().zip(&reports) {
        if report.blocked {
            let names: Vec<String> = report.blocking().map(|r| r.feature.to_string()).collect();
            println!(
                "backend `{kind}`: SKIPPED (blocked by mapping report, spec 14.4: {})",
                names.join(", ")
            );
            continue;
        }
        match availability(kind) {
            Ok(()) => {
                println!("backend `{kind}`: available");
                available.push(kind);
            }
            Err(reason) => println!("backend `{kind}`: SKIPPED ({reason})"),
        }
    }

    let mut tol_exceeded = false;
    let cfg = LoadConfig {
        n_envs: a.envs,
        rate: None,
        seed: a.seed,
    };
    for i in 0..available.len() {
        for j in (i + 1)..available.len() {
            let (ka, kb) = (available[i], available[j]);
            let mut ba = make_backend(ka);
            let mut bb = make_backend(kb);
            let info = ba
                .load(&scene, &cfg)
                .map_err(|e| CliError::Runtime(format!("{ka}: {e}")))?;
            bb.load(&scene, &cfg)
                .map_err(|e| CliError::Runtime(format!("{kb}: {e}")))?;
            let ctrl = ctrl_seq(info.nu, a.envs, a.ticks, a.ctrl_random, a.seed);

            let result = compare_backends(&mut *ba, &mut *bb, &scene, &ctrl, a.ticks)
                .map_err(|e| CliError::Runtime(format!("{ka} vs {kb}: {e}")))?;
            println!("{result}");
            if result.max_dqpos > a.tol {
                tol_exceeded = true;
            }
        }
    }

    Ok(u8::from(blocked_any || tol_exceeded))
}

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("compare") => compare(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es backend: unknown subcommand '{other}'"
        ))),
    }
}
