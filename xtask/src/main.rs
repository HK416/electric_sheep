//! `cargo xtask` — the single entry point for all verification (spec 1.6, 26.2).

mod context_budget;
mod goldens;
mod layering;
mod scope;
mod spec_refs;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

fn workspace_root() -> PathBuf {
    // xtask's own manifest dir is `<root>/xtask`; the workspace root is its parent.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask always lives directly under the workspace root")
        .to_path_buf()
}

fn run_cargo(root: &Path, args: &[&str]) -> bool {
    println!("$ cargo {}", args.join(" "));
    Command::new("cargo")
        .args(args)
        .current_dir(root)
        .status()
        .is_ok_and(|s| s.success())
}

fn cmd_ci(root: &Path) -> bool {
    run_cargo(root, &["fmt", "--check"])
        && run_cargo(
            root,
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        )
        && run_cargo(root, &["test", "--workspace"])
        && context_budget::run(&root.join("crates"))
        && layering::run(root)
        && spec_refs::run(root)
        && goldens::run(root)
}

fn main() -> ExitCode {
    let root = workspace_root();
    let mut args = std::env::args().skip(1);
    let cmd = args.next();

    let ok = match cmd.as_deref() {
        Some("context-budget") => context_budget::run(&root.join("crates")),
        Some("layering") => layering::run(&root),
        Some("check-spec-refs") => spec_refs::run(&root),
        Some("verify-goldens") => goldens::run(&root),
        Some("check-scope") => {
            if let Some(packet) = args.next() {
                scope::run(&root, Path::new(&packet))
            } else {
                eprintln!("usage: cargo xtask check-scope <packet.md>");
                false
            }
        }
        Some("ci") => cmd_ci(&root),
        Some(other) => {
            eprintln!("unknown xtask command: {other}");
            eprintln!(
                "available: context-budget, layering, check-spec-refs, verify-goldens, check-scope <packet.md>, ci"
            );
            false
        }
        None => {
            eprintln!(
                "usage: cargo xtask <context-budget|layering|check-spec-refs|verify-goldens|check-scope|ci>"
            );
            false
        }
    };

    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
