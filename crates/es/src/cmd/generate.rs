//! `es task generate` (spec 14.5): the `es-script` generate -> validate -> repair loop, wired
//! to a provider closure -- `stdin` (any external LLM, no key needed) or `anthropic` (feature
//! `llm` on `es-script`).

use std::io::Read as _;
use std::path::Path;

use es_script::generate::{self, GenerateError, GenerationReport, Palette, Prompt, TaskSpecPrompt};

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es task generate --prompt \"...\" [--scene scene.xml] [--rounds N] [--provider anthropic|stdin] --out <dir>

Builds a prompt from the node palette (es_ir::factory, spec 14.5) and the description,
asks the provider for a Task IR as TOML, validates the candidate with the same validator
`es ir validate` uses, and feeds diagnostics back verbatim until one candidate passes or
--rounds is exhausted (default 3). On success, writes <out>/task.toml.

    --provider stdin       reads one model reply from stdin per round (default; works with
                            any external LLM, no API key needed)
    --provider anthropic   calls the Anthropic Messages API (needs ANTHROPIC_API_KEY; this
                            binary must be built with `--features es-script-llm`)
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    let mut description = None;
    let mut scene_path = None;
    let mut rounds = 3u32;
    let mut provider_name = "stdin".to_owned();
    let mut out_dir = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(0);
            }
            "--prompt" => {
                i += 1;
                description = Some(next_arg(args, i)?);
            }
            "--scene" => {
                i += 1;
                scene_path = Some(next_arg(args, i)?);
            }
            "--rounds" => {
                i += 1;
                rounds = next_arg(args, i)?
                    .parse()
                    .map_err(|_| CliError::Usage(HELP.to_owned()))?;
            }
            "--provider" => {
                i += 1;
                provider_name = next_arg(args, i)?;
            }
            "--out" => {
                i += 1;
                out_dir = Some(next_arg(args, i)?);
            }
            other => {
                return Err(CliError::Usage(format!(
                    "es task generate: unknown argument '{other}'\n\n{HELP}"
                )))
            }
        }
        i += 1;
    }

    let description = description.ok_or_else(|| CliError::Usage(HELP.to_owned()))?;
    let out_dir = out_dir.ok_or_else(|| CliError::Usage(HELP.to_owned()))?;
    let scene = scene_path.map(|p| scene_ref(&p)).transpose()?;

    let req = TaskSpecPrompt {
        description,
        scene,
        constraints: std::collections::BTreeMap::new(),
    };
    let palette = Palette::from_builtins();
    let prompt = generate::build_prompt(&palette, &req);

    let mut provider = make_provider(&provider_name)?;
    let report = generate::repair_loop(&prompt, &mut *provider, rounds);
    print_report(&report);

    if let Some(task) = report.result {
        write_task(Path::new(&out_dir), &task)?;
        Ok(0)
    } else {
        eprintln!("no candidate validated within {rounds} round(s)");
        Ok(1)
    }
}

fn next_arg(args: &[String], i: usize) -> Result<String, CliError> {
    args.get(i)
        .cloned()
        .ok_or_else(|| CliError::Usage(HELP.to_owned()))
}

fn scene_ref(path: &str) -> Result<es_ir::task::SceneRef, CliError> {
    let bytes = std::fs::read(path).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    let hash = *blake3::hash(&bytes).as_bytes();
    Ok(es_ir::task::SceneRef {
        path: path.to_owned(),
        scene_hash: hash,
        asset_hash: hash,
    })
}

type Provider = Box<dyn FnMut(&Prompt) -> Result<String, GenerateError>>;

fn make_provider(name: &str) -> Result<Provider, CliError> {
    match name {
        "stdin" => Ok(Box::new(stdin_provider)),
        "anthropic" => anthropic_provider(),
        other => Err(CliError::Usage(format!(
            "es task generate: unknown --provider '{other}'\n\n{HELP}"
        ))),
    }
}

/// Reads one full reply from stdin. A pipe only supplies one reply, so a round past the first
/// one (past a repair) reads EOF (`""`), fails to parse, and the loop ends at `--rounds`
/// without hanging -- fine for the common one-shot case; an interactive multi-round session
/// wants a provider that reopens or re-prompts stdin itself.
fn stdin_provider(_prompt: &Prompt) -> Result<String, GenerateError> {
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| GenerateError::Provider(e.to_string()))?;
    Ok(buf)
}

#[cfg(feature = "es-script-llm")]
fn anthropic_provider() -> Result<Provider, CliError> {
    let mut provider =
        generate::AnthropicProvider::from_env().map_err(|e| CliError::Runtime(e.to_string()))?;
    Ok(Box::new(move |p: &Prompt| provider.call(p)))
}

#[cfg(not(feature = "es-script-llm"))]
fn anthropic_provider() -> Result<Provider, CliError> {
    Err(CliError::Usage(
        "--provider anthropic needs `es` built with `--features es-script-llm`".to_owned(),
    ))
}

fn print_report(report: &GenerationReport) {
    for (i, round) in report.rounds.iter().enumerate() {
        println!("round {}: candidate {}", i + 1, hex(&round.candidate_hash));
        for d in &round.diagnostics {
            print!("{d}");
        }
    }
}

fn write_task(dir: &Path, task: &es_ir::task::TaskIr) -> Result<(), CliError> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", dir.display())))?;
    let toml = es_ir::serial::task_to_toml(task).map_err(|e| CliError::Runtime(e.to_string()))?;
    let path = dir.join("task.toml");
    std::fs::write(&path, toml)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))?;
    let hash = task
        .task_hash()
        .map_err(|d| CliError::Runtime(d.to_string()))?;
    println!("wrote {}", path.display());
    println!("task_hash: {}", hex(&hash));
    Ok(())
}
