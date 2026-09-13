//! `es` -- the Electric Sheep runtime/CLI (spec 2.5, 10.5, 11.1, 17.2, 26.2).
//!
//! A hand-rolled dispatcher, no `clap`: exit code 0 on success, 1 when a run produced
//! error diagnostics or a runtime failure, 2 on a usage error.

mod cmd;
mod error;
mod util;

use std::process::ExitCode;

use error::CliError;

const TOP_HELP: &str = "\
es -- Electric Sheep runtime/CLI

USAGE:
    es --check-deps
    es --version
    es --help
    es ir validate <file.toml>...
    es ir check <task.toml> <obs.toml> <learning.toml> <deploy.toml> [eval.toml]
    es task compile <task.toml> <obs.toml> [--release]
    es eval compare <A.json> <B.json>
    es dataset info <root>

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
        Some("--check-deps") => Ok(cmd::check_deps::run()),
        Some("ir") => cmd::ir::dispatch(&args[1..]),
        Some("task") => cmd::task::dispatch(&args[1..]),
        Some("eval") => cmd::eval::dispatch(&args[1..]),
        Some("dataset") => cmd::dataset::dispatch(&args[1..]),
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
