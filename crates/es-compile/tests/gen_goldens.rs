//! Provenance harness for `tests/golden/observation/**` (spec 1.4).
//!
//! The goldens are produced by `crates/es-compile/python/gen_observation_goldens.py`, which
//! calls `PyTorch` and torchvision and never touches `es-compile`. This test re-runs that
//! script into a temp directory and fails if a single byte differs from the checked-in files,
//! so "these goldens came from the reference oracle" is a CI-checkable claim rather than a
//! sentence in a design note.
//!
//! It **SKIPs, loudly, when torch/torchvision is absent** — spec 1.4 wants the harness to
//! exist whether or not the oracle is installed on a given machine, and a missing wheel must
//! not be reported as a passing provenance check. Point `ES_PYTHON` at an interpreter that
//! has both to run it for real:
//!
//! ```text
//! ES_PYTHON=<venv>/Scripts/python cargo test -p es-compile --test gen_goldens
//! ```
//!
//! To *replace* the goldens after a deliberate spec change, run the script at the golden
//! directory directly and gate the commit with `GOLDEN_UPDATE=1 cargo xtask verify-goldens`
//! (design note §13). A golden is never edited to make a test pass.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Interpreters to try, in order. `ES_PYTHON` overrides the search entirely — the same knob
/// the `MuJoCo` and torch oracles use, so one venv serves all three.
fn python_candidates() -> Vec<String> {
    match std::env::var("ES_PYTHON") {
        Ok(p) if !p.is_empty() => vec![p],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    }
}

/// The first interpreter that can import both dependencies, or why none could.
fn oracle_python() -> Result<String, String> {
    let mut tried = Vec::new();
    for python in python_candidates() {
        match Command::new(&python)
            .args(["-c", "import torch, torchvision"])
            .output()
        {
            Ok(out) if out.status.success() => return Ok(python),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tried.push(format!(
                    "`{python}`: {}",
                    stderr.lines().last().unwrap_or("import failed").trim()
                ));
            }
            Err(e) => tried.push(format!("`{python}`: {e}")),
        }
    }
    Err(format!(
        "no Python interpreter with torch + torchvision (set ES_PYTHON to choose one): {}",
        tried.join("; ")
    ))
}

fn run_script(python: &str, out_dir: &Path) {
    let script = manifest("python/gen_observation_goldens.py");
    let out = Command::new(python)
        .arg(&script)
        .arg(out_dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn {python} {}: {e}", script.display()));
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn the_goldens_are_what_the_torch_oracle_produces() {
    let python = match oracle_python() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIPPED the_goldens_are_what_the_torch_oracle_produces: {why}");
            return;
        }
    };

    let fresh = std::env::temp_dir().join("es-compile-observation-goldens");
    let _ = std::fs::remove_dir_all(&fresh);
    run_script(&python, &fresh);

    let checked_in = manifest("../../tests/golden/observation");
    let mut names: Vec<_> = std::fs::read_dir(&checked_in)
        .expect("tests/golden/observation")
        .map(|e| e.expect("dir entry").file_name())
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no goldens to check");

    for name in names {
        let want = std::fs::read(checked_in.join(&name)).expect("read checked-in golden");
        let got = std::fs::read(fresh.join(&name)).unwrap_or_else(|e| {
            panic!(
                "the oracle did not produce `{}`: {e} — a golden with no generator is a golden \
                 with no provenance (spec 1.4)",
                name.to_string_lossy()
            )
        });
        assert!(
            want == got,
            "`{}` differs from what the oracle just produced ({} vs {} bytes). Either the \
             kernel semantics moved or torch/torchvision were upgraded past the versions \
             pinned in docs/api-notes/torchvision.md.",
            name.to_string_lossy(),
            want.len(),
            got.len()
        );
    }
}
