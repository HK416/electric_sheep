//! Packet M15/N7: `es train --dry-run` refuses `[run] single_view` by name before it prints a
//! plan -- for the committed demo bundle, whose one camera meets the state in a `Concat`, and
//! for a weight that is not above 0. Python-free; the planning and the checks themselves are
//! `crates/es-data/tests/training_single_view.rs`'s.

use std::path::{Path, PathBuf};
use std::process::Command;

use es_compile::PolicyBundle;
use es_ir::learning::WeightsRef;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn demo_bundle(dir: &Path) -> PathBuf {
    let mut learning =
        es_ir::serial::learning_from_toml(&fixture("learning.toml")).expect("learning.toml");
    let weights = b"es-n7-placeholder".to_vec();
    learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = PolicyBundle::build(
        &es_ir::serial::task_from_toml(&fixture("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&fixture("observation.toml"))
            .expect("observation.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&fixture("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the demo documents pack");
    let path = dir.join("untrained.esb");
    std::fs::write(&path, bytes).expect("write the bundle");
    path
}

fn dry_run(dir: &Path, weight: &str, bundle: &Path) -> (Option<i32>, String) {
    let recipe = dir.join(format!("training-{weight}.toml"));
    std::fs::write(
        &recipe,
        format!(
            "kind = \"training\"\n[dataset]\nroot = \"ds\"\nframes = \"frames\"\n[policy]\n\
             bundle = \"{}\"\n[run]\nsteps = 10\nbatch = 2\nlr = 1e-4\nseed = 0\n\
             device = \"cpu\"\nsingle_view = {{ weight = {weight} }}\n",
            bundle.to_string_lossy().replace('\\', "/")
        ),
    )
    .expect("write the recipe");
    let out = Command::new(env!("CARGO_BIN_EXE_es"))
        .args(["train", "--recipe", &recipe.to_string_lossy()])
        .args(["--out", &dir.join("out").to_string_lossy(), "--dry-run"])
        .output()
        .expect("run es");
    (
        out.status.code(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

#[test]
fn train_dry_run_refuses_single_view_by_name() {
    let dir = std::env::temp_dir().join(format!("es-train-single-view-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    let bundle = demo_bundle(&dir);

    let (code, text) = dry_run(&dir, "0.5", &bundle);
    assert_eq!(code, Some(1), "{text}");
    assert!(
        text.contains("`single_view`") && text.contains("no `Sum`"),
        "{text}"
    );
    let (code, text) = dry_run(&dir, "0.0", &bundle);
    assert_eq!(code, Some(1), "{text}");
    assert!(text.contains("`single_view.weight` is 0"), "{text}");
    assert!(
        !dir.join("out/training/plan.txt").exists(),
        "a refused recipe printed a plan"
    );
    println!("RAN train_dry_run_refuses_single_view_by_name");
    let _ = std::fs::remove_dir_all(&dir);
}
