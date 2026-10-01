//! `es project generate` (plan G, packet G3b; `docs/design/scene-authoring.md` section 4.2): a
//! task specification and its scene to every document of the project. The work is
//! `es_script::spec::generate`'s; this verb reads, writes and reports.

use std::path::Path;

use es_script::spec::{generate, TaskSpec};

use crate::error::CliError;

const HELP: &str = "\
es project generate --spec <task.estask> --out <dir> [--scene <scene>]

Compiles the task specification with its scene into the project's documents and writes them
into <dir>: task.toml (the Task IR); with [teacher] observation-, learning-, deployment-,
evaluation- and training-teacher.toml; with [student] (arm <name>, default `student`) the same
five for the arm plus evaluation-<name>-nominal.toml; with [cycle], cycle-<name>.toml. Paths
are relative to the current directory, which is the project root: the spec's `scene`, --scene
(in place of the spec's) and --out, as the recipes and the cycle will name the documents.
Each file says which spec it was generated from; the same spec and scene give the same bytes.

Each recipe's header has the `es policy init` line that builds its bundle from the documents.

Exit code: 0 written; 1 the spec or scene was refused (the message names the section and the
field); 2 usage error.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("generate") => run(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es project: unknown subcommand '{other}'\n\n{HELP}"
        ))),
    }
}

fn run(args: &[String]) -> Result<u8, CliError> {
    let (mut spec_path, mut out, mut scene) = (None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(0);
            }
            "--spec" => spec_path = it.next().cloned(),
            "--out" => out = it.next().cloned(),
            "--scene" => scene = it.next().cloned(),
            other => {
                return Err(CliError::Usage(format!(
                    "es project generate: unexpected '{other}'\n\n{HELP}"
                )))
            }
        }
    }
    let (Some(spec_path), Some(out)) = (spec_path, out) else {
        return Err(CliError::Usage(HELP.to_owned()));
    };
    let fail = |e: &dyn std::fmt::Display| CliError::Runtime(format!("{spec_path}: {e}"));
    let text = std::fs::read_to_string(&spec_path).map_err(|e| fail(&e))?;
    let mut spec = TaskSpec::from_toml(&text).map_err(|e| fail(&e))?;
    if let Some(scene) = scene {
        spec.scene = scene;
    }
    let source = spec_path.replace('\\', "/");
    let docs = generate(&spec, Path::new("."), &out, &source).map_err(|e| fail(&e))?;
    let dir = Path::new(&out);
    std::fs::create_dir_all(dir).map_err(|e| fail(&e))?;
    for d in &docs {
        let path = dir.join(&d.file);
        std::fs::write(&path, &d.text).map_err(|e| fail(&e))?;
        println!("wrote {}", path.display());
    }
    Ok(0)
}
