//! Oracle for the Python authoring builder (spec 14.2, M2 W6 acceptance): the `pick_place.py`
//! example must produce TOML that the real IR validators accept with zero errors.
//!
//! This is a Rust integration test, not a Python one, on purpose: `es-py`'s own `cargo test`
//! must not require a Python environment (CLAUDE.md, spec 2.1), so this test SKIPs unless
//! `ES_PYTHON` names a Python 3.12 interpreter with `es` installed (`maturin develop --release
//! --features python` from `python/es`, or an editable install pointing at it). There is no
//! `es` CLI yet to shell out to for `es ir validate`/`es ir check` (`CLAUDE.md`: "es is the
//! runtime/CLI, not yet implemented") — this test calls the same validators that CLI would
//! (`TaskIr::validate`, `ObservationIr::validate`, `LearningGraph::validate`,
//! `DeploymentIr::validate`) directly instead.

use std::path::Path;
use std::process::Command;

use es_ir::serial::{
    deployment_from_toml, learning_from_toml, observation_from_toml, task_from_toml,
};
use es_ir::Diagnostic;

fn assert_no_errors(what: &str, diags: &[Diagnostic]) {
    let errors: Vec<&Diagnostic> = diags.iter().filter(|d| d.is_error()).collect();
    assert!(
        errors.is_empty(),
        "{what} has validation errors:\n{}",
        errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn pick_place_example_produces_valid_ir() {
    let Ok(python) = std::env::var("ES_PYTHON") else {
        eprintln!(
            "SKIP: pick_place_example_produces_valid_ir (set ES_PYTHON=<python with `es` \
             installed, e.g. via `maturin develop --release --features python` from python/es>)"
        );
        return;
    };

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crates/es-py is two levels under the repo root");
    let example = repo_root.join("python/es/examples/pick_place.py");
    let out_dir = std::env::temp_dir().join(format!("es-py-pick-place-{}", std::process::id()));

    let status = Command::new(&python)
        .arg(&example)
        .arg(&out_dir)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {python} {}: {e}", example.display()));
    assert!(status.success(), "pick_place.py exited with {status}");

    let read = |name: &str| std::fs::read_to_string(out_dir.join(name)).unwrap();

    let task = task_from_toml(&read("task.toml")).expect("task.toml parses");
    assert_no_errors("task.toml", &task.validate());

    let observation =
        observation_from_toml(&read("observation.toml")).expect("observation.toml parses");
    assert_no_errors("observation.toml", &observation.validate());

    let learning = learning_from_toml(&read("learning.toml")).expect("learning.toml parses");
    assert_no_errors("learning.toml", &learning.validate());

    let deployment =
        deployment_from_toml(&read("deployment.toml")).expect("deployment.toml parses");
    assert_no_errors("deployment.toml", &deployment.validate());

    let _ = std::fs::remove_dir_all(&out_dir);
}
