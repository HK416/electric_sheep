//! `es import lerobot-config` (spec 14.4 external conversion), `es import roboverse` and
//! `es import usd` (spec 28.6, the native `.usda` subset reader).

use std::path::PathBuf;

use es_data::lerobot::LeRobotDataset;
use es_data::lerobot_config::{convert, LeRobotPolicyConfig, Stats};
use es_data::roboverse::{self, RoboVerseTask};
use es_ir::serial;
use serde::Serialize;

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es import lerobot-config --config <config.json> [--stats <stats.json>] [--dataset <root>] --out <dir>

Converts a LeRobot policy config.json (spec 14.4 external conversion) into an
(Observation IR, Learning IR) pair via `es_data::lerobot_config::convert`, and writes,
via `es_ir::serial`:
  <out>/observation.toml
  <out>/learning.toml

Prints every non-fatal warning the conversion made (no lerobot version is pinned in this
workspace, spec 1.7, so an unrecognized config field is a warning, not an error) and both
IR content hashes.

    --stats <stats.json>   meta/stats.json, for normalization moments (else identity + warning)
    --dataset <root>       a LeRobot dataset root; its meta/info.json supplies fps/features

Exit code: 0 on success, 1 on a ConfigError (malformed JSON, unsupported policy type,
missing feature) or an I/O error, 2 on a usage error.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("lerobot-config") => lerobot_config(&args[1..]),
        Some("roboverse") => roboverse_import(&args[1..]),
        Some("usd") => usd_import(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es import: unknown subcommand '{other}'"
        ))),
    }
}

struct Args {
    config: String,
    stats: Option<String>,
    dataset: Option<String>,
    out: PathBuf,
}

fn parse_args(args: &[String]) -> Result<Args, CliError> {
    let (mut config, mut stats, mut dataset, mut out) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{HELP}")))
        };
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(HELP.to_owned())),
            "--config" => config = Some(val()?.clone()),
            "--stats" => stats = Some(val()?.clone()),
            "--dataset" => dataset = Some(val()?.clone()),
            "--out" => out = Some(PathBuf::from(val()?)),
            other => return Err(CliError::Usage(format!("unknown flag '{other}'\n\n{HELP}"))),
        }
    }
    Ok(Args {
        config: config.ok_or_else(|| CliError::Usage(format!("--config is required\n\n{HELP}")))?,
        stats,
        dataset,
        out: out.ok_or_else(|| CliError::Usage(format!("--out is required\n\n{HELP}")))?,
    })
}

fn lerobot_config(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let a = parse_args(args)?;

    let raw = std::fs::read_to_string(&a.config)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.config)))?;
    let cfg = match LeRobotPolicyConfig::parse(&raw) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("error: {}: {e}", a.config);
            return Ok(1);
        }
    };

    let stats = match &a.stats {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
            match serde_json::from_str::<Stats>(&raw) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("error: {path}: {e}");
                    return Ok(1);
                }
            }
        }
        None => None,
    };

    let dataset = match &a.dataset {
        Some(root) => Some(
            LeRobotDataset::open(root.as_str())
                .map_err(|e| CliError::Runtime(format!("{root}: {e}")))?,
        ),
        None => None,
    };

    let converted = match convert(
        &cfg,
        stats.as_ref(),
        dataset.as_ref().map(LeRobotDataset::info),
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return Ok(1);
        }
    };

    let obs_hash = converted
        .observation
        .observation_hash()
        .map_err(|d| CliError::Runtime(d.to_string()))?;
    let learning_hash = converted
        .learning
        .learning_hash()
        .map_err(|d| CliError::Runtime(d.to_string()))?;
    let obs_toml = serial::observation_to_toml(&converted.observation)
        .map_err(|e| CliError::Runtime(e.to_string()))?;
    let learning_toml = serial::learning_to_toml(&converted.learning)
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    std::fs::create_dir_all(&a.out)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.out.display())))?;
    std::fs::write(a.out.join("observation.toml"), obs_toml)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.out.display())))?;
    std::fs::write(a.out.join("learning.toml"), learning_toml)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.out.display())))?;

    for w in &converted.warnings {
        println!("warning: {w}");
    }
    println!("observation_hash: {}", hex(&obs_hash));
    println!("learning_hash: {}", hex(&learning_hash));
    Ok(0)
}

// --- `es import roboverse` (spec 14.4 external conversion) ---------------------------------

const ROBOVERSE_HELP: &str = "\
es import roboverse <task.json> --out <dir>

Converts a RoboVerse / MetaSim task config (spec 14.4 external conversion) into a
(Task IR, Observation IR) pair via `es_data::roboverse::convert`, and writes:
  <out>/task.toml           (via es_ir::serial)
  <out>/observation.toml    (via es_ir::serial)
  <out>/provenance.json     (source name, version, license, and every external asset path)

Prints every non-fatal warning, every unmapped item (spec 14.4: an unrecognized checker or
asset shape is reported, not silently dropped), and both IR content hashes.

Exit code: 0 on success, 1 on a ConvertError, an I/O error, or an unmapped item with
severity = error (spec 14.4: unmapped items block execution), 2 on a usage error. On a
refusal nothing is written -- the check runs before the first file.
";

#[derive(Serialize)]
struct ProvenanceFile {
    name: String,
    version: Option<String>,
    license: Option<String>,
    scene_refs: Vec<roboverse::SceneAssetRef>,
}

fn roboverse_import(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{ROBOVERSE_HELP}");
        return Ok(0);
    }
    let (task_path, out) = match args {
        [task, flag, out] if flag == "--out" => (task.clone(), PathBuf::from(out)),
        _ => {
            return Err(CliError::Usage(format!(
                "usage: es import roboverse <task.json> --out <dir>\n\n{ROBOVERSE_HELP}"
            )))
        }
    };

    let raw = std::fs::read_to_string(&task_path)
        .map_err(|e| CliError::Runtime(format!("{task_path}: {e}")))?;
    let task = match RoboVerseTask::parse(&raw) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {task_path}: {e}");
            return Ok(1);
        }
    };
    let converted = match roboverse::convert(&task) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return Ok(1);
        }
    };

    for w in &converted.warnings {
        println!("warning: {w}");
    }
    for u in &converted.unmapped {
        println!("unmapped ({:?}): {}", u.severity, u.item);
    }

    // spec 14.4: an `Error`-severity unmapped item blocks the import. Decided *before* any
    // artifact is written, so a refusal leaves no complete-looking output directory behind
    // (`usd_import` below has the same contract).
    if converted
        .unmapped
        .iter()
        .any(|u| u.severity == roboverse::Severity::Error)
    {
        eprintln!("error: unmapped items block this conversion; nothing was written");
        return Ok(1);
    }

    let task_hash = converted
        .task
        .task_hash()
        .map_err(|d| CliError::Runtime(d.to_string()))?;
    let obs_hash = converted
        .observation
        .observation_hash()
        .map_err(|d| CliError::Runtime(d.to_string()))?;
    let task_toml =
        serial::task_to_toml(&converted.task).map_err(|e| CliError::Runtime(e.to_string()))?;
    let obs_toml = serial::observation_to_toml(&converted.observation)
        .map_err(|e| CliError::Runtime(e.to_string()))?;
    let provenance_json = serde_json::to_string_pretty(&ProvenanceFile {
        name: converted.provenance.name,
        version: converted.provenance.version,
        license: converted.provenance.license,
        scene_refs: converted.scene_refs,
    })
    .map_err(|e| CliError::Runtime(e.to_string()))?;

    std::fs::create_dir_all(&out)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;
    std::fs::write(out.join("task.toml"), task_toml)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;
    std::fs::write(out.join("observation.toml"), obs_toml)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;
    std::fs::write(out.join("provenance.json"), provenance_json)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;

    println!("task_hash: {}", hex(&task_hash));
    println!("observation_hash: {}", hex(&obs_hash));
    Ok(0)
}

const USD_HELP: &str = "\
es import usd <file.usda> --out <scene.json>

Reads a `.usda` text layer with the native subset reader (spec 28.6; spec 1.9 item 6 keeps
this reader minimal, with USD Bake as the supported fallback) and writes the resulting
`SceneDesc` as JSON.

Applies the spec 3.1 conventions once: `upAxis = \"Y\"` is rotated onto Z-up, every length is
scaled by `metersPerUnit`, and revolute limits and angular drive targets are converted from
the schema's degrees to radians. Prints every warning, then the scene content hash.

Composition arcs (`references`, `payload`, `variantSet`), `.usdc` and `.usdz` are refused by
prim path rather than silently ignored: flatten the layer with USD Bake first.

Exit code: 0 on success, 1 on a parse or mapping error, 2 on a usage error.
";

fn usd_import(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USD_HELP}");
        return Ok(0);
    }
    let (path, out) = match args {
        [file, flag, out] if flag == "--out" => (file.clone(), PathBuf::from(out)),
        _ => {
            return Err(CliError::Usage(format!(
                "usage: es import usd <file.usda> --out <scene.json>\n\n{USD_HELP}"
            )))
        }
    };

    // S-2 (docs/reviews/M4.md): cap the read at the same size `es_usd::parse_usda` enforces,
    // so a huge file is refused by its metadata length rather than fully read into memory
    // first only to be rejected afterward.
    let len = std::fs::metadata(&path)
        .map_err(|e| CliError::Runtime(format!("{path}: {e}")))?
        .len();
    if len > es_usd::MAX_USDA_BYTES as u64 {
        eprintln!(
            "error: {path}: file is {len} bytes, over the {}-byte cap",
            es_usd::MAX_USDA_BYTES
        );
        return Ok(1);
    }
    let text =
        std::fs::read_to_string(&path).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    let (scene, warnings) = match es_physics_core::usd::import_usda(&text) {
        Ok(imported) => imported,
        Err(e) => {
            eprintln!("error: {path}: {e}");
            return Ok(1);
        }
    };
    for w in &warnings {
        println!("warning: {w}");
    }

    let json =
        serde_json::to_string_pretty(&scene).map_err(|e| CliError::Runtime(e.to_string()))?;
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", dir.display())))?;
    }
    std::fs::write(&out, json).map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;

    let geoms: usize = scene.bodies.iter().map(|b| b.geoms.len()).sum();
    println!(
        "bodies: {} joints: {} geoms: {} assets: {}",
        scene.bodies.len(),
        scene.joints.len(),
        geoms,
        scene.assets.len()
    );
    println!("scene_hash: {}", hex(&scene.scene_hash()));
    Ok(0)
}
