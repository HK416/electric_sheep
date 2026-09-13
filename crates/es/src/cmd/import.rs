//! `es import lerobot-config` (spec 14.4 external conversion).

use std::path::PathBuf;

use es_data::lerobot::LeRobotDataset;
use es_data::lerobot_config::{convert, LeRobotPolicyConfig, Stats};
use es_ir::serial;

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
