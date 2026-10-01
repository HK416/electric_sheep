//! `es` -- the Electric Sheep runtime/CLI (spec 2.5, 10.5, 11.1, 17.2, 26.2).
//!
//! A hand-rolled dispatcher, no `clap`: exit code 0 on success, 1 when a run produced
//! error diagnostics or a runtime failure, 2 on a usage error.

mod cmd;

use std::process::ExitCode;

// `CliError` and `hex` moved to `es-tools` with the verbs that use them most (packet
// `docs/packets/M10/W3b-es-tools-split.md`); re-homed here under their old paths so every
// `crate::error::` / `crate::util::` in `cmd/*` still resolves.
pub use es_tools::{error, util};

use error::CliError;

const TOP_HELP: &str = "\
es -- Electric Sheep runtime/CLI

USAGE:
    es --check-deps [--json]
    es --version
    es --help
    es ir validate <file.toml>...
    es ir check <task.toml> <obs.toml> <learning.toml> <deploy.toml> [eval.toml]
    es task compile <task.toml> <obs.toml> [--release]
    es eval compare <A.json> <B.json>
    es eval run --config <eval.toml> --policy <policy.esb> --scene <file.xml|urdf> [OPTIONS]
    es evidence verify <bundle.esb> [--against <other.esb>] [--json]
    es gap --sim <root> --real <root> [--out gap_report.json] [--max-samples N] [--threshold D]
    es loop collect|intervene|distill|cycle ...   (see `es loop --help`)
    es import lerobot-config --config <config.json> [--stats ...] [--dataset ...] --out <dir>
    es dataset info <root>
    es train --recipe <training.toml> [--out <dir>] [--dry-run]
    es policy lower --policy <in.esb> --out <dir>
    es policy pack --policy <in.esb> --weights <model.safetensors> --out <out.esb>
    es backend compare --scene <file.xml|urdf> --backends mujoco-cpu,mjwarp[,newton,physx]
    es bench [--memory-report --obs <obs.toml> ...]
    es render --scene <file.xml|urdf> --eye X,Y,Z --look-at X,Y,Z --out <file.ppm|-> [OPTIONS]
    es scene export <file.xml|urdf|esscene> --mjcf <out.xml>
    es project generate --spec <task.estask> --out <dir> [--scene <scene>]
    es video mosaic --frames <dir> --events <events.json> --report <report.json>
                    --grid RxC --out <dir> [--label-height N]

Run `es <subcommand> --help` for details on one subcommand.
";

fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        None | Some("--help" | "-h") => {
            println!("{TOP_HELP}");
            Ok(0)
        }
        Some("--version") => {
            println!("es {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        Some("--check-deps") => Ok(cmd::check_deps::run(args.iter().any(|a| a == "--json"))),
        Some("ir") => cmd::ir::dispatch(&args[1..]),
        Some("task") if args.get(1).map(String::as_str) == Some("generate") => {
            cmd::generate::dispatch(&args[2..])
        }
        Some("task") => cmd::task::dispatch(&args[1..]),
        Some("eval") => cmd::eval::dispatch(&args[1..]),
        Some("evidence") => cmd::evidence::dispatch(&args[1..]),
        Some("gap") => cmd::gap::dispatch(&args[1..]),
        Some("loop") => cmd::r#loop::dispatch(&args[1..]),
        Some("mcp") => cmd::mcp::dispatch(&args[1..]),
        Some("import") => cmd::import::dispatch(&args[1..]),
        Some("dataset") => cmd::dataset::dispatch(&args[1..]),
        Some("train") => cmd::train::dispatch(&args[1..]),
        Some("policy") => Ok(cmd::policy::dispatch(&args[1..]) as u8),
        Some("project") => cmd::project::dispatch(&args[1..]),
        Some("backend") => cmd::backend::dispatch(&args[1..]),
        Some("bench") => cmd::bench::dispatch(&args[1..]),
        Some("scene") => cmd::scene::dispatch(&args[1..]),
        Some("video") => Ok(cmd::video::dispatch(&args[1..]) as u8),
        #[cfg(feature = "render")]
        Some("render") => es_tools::render::run(&args[1..]),
        #[cfg(not(feature = "render"))]
        Some("render") => Err(CliError::Usage(
            "es render needs the `render` feature; this build links no renderer".to_owned(),
        )),
        Some(other) => Err(CliError::Usage(format!(
            "unknown command '{other}'\n\n{TOP_HELP}"
        ))),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(code) => ExitCode::from(code),
        Err(CliError::Usage(msg)) => {
            eprintln!("{msg}");
            ExitCode::from(2)
        }
        Err(CliError::Runtime(msg)) => {
            eprintln!("error: {msg}");
            ExitCode::from(1)
        }
    }
}
