//! `es bench` (spec 12.4 performance metrics, spec 20.2/20.3 memory budget).
//!
//! Two independent things live here because they share a command name in the spec, not
//! because they share code: `--memory-report` runs the static [`es_compile::budget`] model
//! (spec 20.2) against a compiled Observation/Learning IR pair, no GPU and no running backend
//! required. Without it, `es bench` prints the spec 12.4 nine-metric table stub -- every field
//! `unmeasured`, because the measurement loop itself is a later packet.

use es_compile::budget::{
    BudgetDomains, BudgetInputs, MemoryBudget, ModelSizes, Precision, TileAtlasCfg,
};
use es_physics_core::{LoadConfig, PhysicsBackend};
use es_telemetry::PerfMetrics;

use crate::error::CliError;

const HELP: &str = "\
es bench [OPTIONS]

Without --memory-report, prints the spec 12.4 nine-metric table with every field
`unmeasured` (the measurement loop is a later packet).

    --memory-report              run the spec 20.2 static memory budget model instead
    --obs <observation.toml>     Observation IR (required with --memory-report)
    --learning <learning.toml>   Learning IR (optional -- without it, inference activations,
                                  policy weights and chunk buffers are unavailable)
    --scene <file.xml|urdf>      scene to size the physics-state item from (optional; loads
                                  MuJoCo CPU to read nq/nv/nu/nsensordata, skipped if
                                  unavailable)
    --sim-envs N                 simulation batch size (required with --memory-report)
    --obs-envs N                 observation batch size (required with --memory-report)
    --views N                    cameras per observation env (required with --memory-report)
    --inference-batch N          inference batch size (required with --memory-report)
    --precision f16|f32          inference activation precision (default f32)
    --device-gib G                device memory, GiB -- enables the spec 20.3 headroom rule
    --tile-w / --tile-h / --tiles-per-row N   explicit tile-atlas layout (spec 15.2); all
                                  three or none

Exit code (with --memory-report): 0 with no spec 20.3 rule violations, 1 otherwise.
Exit code (without it): always 0 -- there is nothing yet to fail on.
";

struct MemArgs {
    obs: String,
    learning: Option<String>,
    scene: Option<String>,
    sim_envs: u32,
    obs_envs: u32,
    views: u32,
    inference_batch: u32,
    precision: Precision,
    device_gib: Option<f64>,
    tile_w: Option<u32>,
    tile_h: Option<u32>,
    tiles_per_row: Option<u32>,
}

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    let mut memory_report = false;
    let mut obs = None;
    let mut learning = None;
    let mut scene = None;
    let mut sim_envs = None;
    let mut obs_envs = None;
    let mut views = None;
    let mut inference_batch = None;
    let mut precision = Precision::F32;
    let mut device_gib = None;
    let mut tile_w = None;
    let mut tile_h = None;
    let mut tiles_per_row = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{HELP}")))
        };
        match a.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(0);
            }
            "--memory-report" => memory_report = true,
            "--obs" => obs = Some(val()?.clone()),
            "--learning" => learning = Some(val()?.clone()),
            "--scene" => scene = Some(val()?.clone()),
            "--sim-envs" => sim_envs = Some(parse(&a.clone(), val()?)?),
            "--obs-envs" => obs_envs = Some(parse(&a.clone(), val()?)?),
            "--views" => views = Some(parse(&a.clone(), val()?)?),
            "--inference-batch" => inference_batch = Some(parse(&a.clone(), val()?)?),
            "--precision" => {
                precision = match val()?.as_str() {
                    "f32" => Precision::F32,
                    "f16" => Precision::F16,
                    other => {
                        return Err(CliError::Usage(format!(
                            "--precision: unknown '{other}', expected f16 or f32\n\n{HELP}"
                        )))
                    }
                };
            }
            "--device-gib" => device_gib = Some(parse(&a.clone(), val()?)?),
            "--tile-w" => tile_w = Some(parse(&a.clone(), val()?)?),
            "--tile-h" => tile_h = Some(parse(&a.clone(), val()?)?),
            "--tiles-per-row" => tiles_per_row = Some(parse(&a.clone(), val()?)?),
            other => return Err(CliError::Usage(format!("unknown flag '{other}'\n\n{HELP}"))),
        }
    }

    if !memory_report {
        print_metric_stub();
        return Ok(0);
    }

    let Some(obs) = obs else {
        return Err(CliError::Usage(format!(
            "--memory-report requires --obs\n\n{HELP}"
        )));
    };
    let (Some(sim_envs), Some(obs_envs), Some(views), Some(inference_batch)) =
        (sim_envs, obs_envs, views, inference_batch)
    else {
        return Err(CliError::Usage(format!(
            "--memory-report requires --sim-envs, --obs-envs, --views and --inference-batch\n\n{HELP}"
        )));
    };
    if !matches!(
        (tile_w, tile_h, tiles_per_row),
        (Some(_), Some(_), Some(_)) | (None, None, None)
    ) {
        return Err(CliError::Usage(format!(
            "--tile-w/--tile-h/--tiles-per-row must be given together\n\n{HELP}"
        )));
    }

    memory_report_cmd(&MemArgs {
        obs,
        learning,
        scene,
        sim_envs,
        obs_envs,
        views,
        inference_batch,
        precision,
        device_gib,
        tile_w,
        tile_h,
        tiles_per_row,
    })
}

fn parse<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, CliError>
where
    T::Err: std::fmt::Display,
{
    raw.parse()
        .map_err(|e| CliError::Usage(format!("{flag}: {e}")))
}

/// The spec 12.4 nine-metric table, every field `unmeasured` -- the measurement loop that
/// would fill this in is a later packet (this one is offline: budget model only).
fn print_metric_stub() {
    let m = PerfMetrics::default();
    let opt = |v: Option<f64>| v.map_or_else(|| "unmeasured".to_owned(), |v| v.to_string());
    println!("performance metrics (spec 12.4 -- never a single step/s figure):");
    println!(
        "  physics_steps_per_sec:    {}",
        opt(m.physics_steps_per_sec)
    );
    println!(
        "  camera_frames_per_sec:    {}",
        opt(m.camera_frames_per_sec)
    );
    println!("  pixels_per_sec:           {}", opt(m.pixels_per_sec));
    println!(
        "  observation_gb_per_sec:   {}",
        opt(m.observation_gb_per_sec)
    );
    println!(
        "  policy_inferences_per_sec:{}",
        opt(m.policy_inferences_per_sec)
    );
    println!("  actions_per_sec:          {}", opt(m.actions_per_sec));
    println!(
        "  p50/p95 end_to_end_latency: {} / {}",
        opt(m.p50_end_to_end_latency),
        opt(m.p95_end_to_end_latency)
    );
    println!("  gpu_memory_peak:          {}", opt(m.gpu_memory_peak));
    println!("  chunk_underrun_rate:      {}", opt(m.chunk_underrun_rate));
    println!();
    println!(
        "Target / Status: unverified -- no measurement loop wired up yet (M2 W5 offline packet)."
    );
}

/// Loads a scene and reads `nq`/`nv`/`nu`/`nsensordata` off the `MuJoCo` CPU reference backend.
/// Best-effort: an unavailable backend or a parse failure is reported and treated as "no
/// physics-state data", never a hard failure of the memory report itself.
fn model_sizes_from_scene(path: &str) -> Option<ModelSizes> {
    let raw = match std::fs::read_to_string(path) {
        Ok(r) => r,
        Err(e) => {
            println!("note: --scene {path}: {e} -- physics_state left unavailable");
            return None;
        }
    };
    let is_urdf = std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("urdf"));
    let scene = if is_urdf {
        let resolver = es_assets::urdf::PackageResolver::from_env();
        es_assets::urdf::parse_urdf(&raw, &resolver)
            .map(|i| i.scene)
            .map_err(|e| e.to_string())
    } else {
        es_assets::parse_mjcf(&raw)
            .map(|i| i.scene)
            .map_err(|e| e.to_string())
    };
    let scene = match scene {
        Ok(s) => s,
        Err(e) => {
            println!("note: --scene {path}: {e} -- physics_state left unavailable");
            return None;
        }
    };
    let mut backend = es_physics_backend::MuJoCoCpuBackend::new();
    match backend.load(&scene, &LoadConfig::default()) {
        Ok(info) => Some(ModelSizes {
            nq: info.nq,
            nv: info.nv,
            nu: info.nu,
            nsensordata: info.nsensordata,
        }),
        Err(e) => {
            println!(
                "note: MuJoCo CPU backend unavailable ({e}) -- physics_state left unavailable"
            );
            None
        }
    }
}

fn memory_report_cmd(a: &MemArgs) -> Result<u8, CliError> {
    let obs_raw = std::fs::read_to_string(&a.obs)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.obs)))?;
    let obs = es_ir::serial::observation_from_toml(&obs_raw)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.obs)))?;

    let learning = a
        .learning
        .as_ref()
        .map(|path| {
            let raw = std::fs::read_to_string(path)
                .map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
            es_ir::serial::learning_from_toml(&raw)
                .map_err(|e| CliError::Runtime(format!("{path}: {e}")))
        })
        .transpose()?;

    let model = a.scene.as_deref().and_then(model_sizes_from_scene);

    let inputs = BudgetInputs {
        domains: BudgetDomains {
            n_sim_envs: a.sim_envs,
            n_obs_envs: a.obs_envs,
            n_views: a.views,
            inference_batch: a.inference_batch,
        },
        obs: &obs,
        learning: learning.as_ref(),
        model: model.as_ref(),
        precision: a.precision,
        tile_atlas: match (a.tile_w, a.tile_h, a.tiles_per_row) {
            (Some(tile_w), Some(tile_h), Some(tiles_per_row)) => Some(TileAtlasCfg {
                tile_w,
                tile_h,
                tiles_per_row,
            }),
            _ => None,
        },
        device_bytes: a.device_gib.map(|g| (g * 1024.0 * 1024.0 * 1024.0) as u64),
    };

    let report = MemoryBudget::estimate(&inputs);
    println!("{report}");
    let violations = report.violations(&inputs);
    if violations.is_empty() {
        println!("no spec 20.3 rule violations");
        Ok(0)
    } else {
        println!("spec 20.3 violations:");
        for v in &violations {
            println!("  [{}] {}", v.rule, v.message);
        }
        Ok(1)
    }
}
