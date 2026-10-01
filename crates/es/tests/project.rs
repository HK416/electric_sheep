//! Packet G3b (`docs/packets/M17/plan-g.md`), oracle (3): `es project generate` writes each
//! reference specification's project; `es ir check` accepts each arm's five documents, `es
//! policy init` builds each arm's bundle from them as the recipe headers say, and the recipes
//! and the cycle dry-run. That the documents are the committed ones, hash for hash, is
//! `crates/es-script/tests/estask_generate.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn es(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_es"))
        .current_dir(repo())
        .args(args)
        .output()
        .expect("run es");
    (
        out.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

fn scratch(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("es-project-{tag}-{nanos}"))
}

/// `es project generate` on `spec` into a fresh directory, returned as the CLI names it.
fn generate(spec: &str, tag: &str) -> String {
    let dir = scratch(tag).to_string_lossy().replace('\\', "/");
    let (ok, text) = es(&["project", "generate", "--spec", spec, "--out", &dir]);
    assert!(ok, "{text}");
    println!("{text}");
    dir
}

/// `es ir check` on the arm's five documents, then `es policy init` on its four.
fn check_and_build(dir: &str, arm: &str) {
    let doc = |kind: &str| format!("{dir}/{kind}-{arm}.toml");
    let task = format!("{dir}/task.toml");
    let (ok, text) = es(&[
        "ir",
        "check",
        &task,
        &doc("observation"),
        &doc("learning"),
        &doc("deployment"),
        &doc("evaluation"),
    ]);
    assert!(ok && !text.contains("ERROR"), "es ir check ({arm})\n{text}");
    let bundle = format!("{dir}/{arm}.esb");
    let (ok, text) = es(&[
        "policy",
        "init",
        "--task",
        &task,
        "--observation",
        &doc("observation"),
        "--learning",
        &doc("learning"),
        "--deployment",
        &doc("deployment"),
        "--out",
        &bundle,
    ]);
    assert!(ok, "es policy init ({arm})\n{text}");
    es_compile::PolicyBundle::open(&std::fs::read(&bundle).expect("written")).expect("opens");
    println!("RAN es ir check + es policy init ({arm})");
}

fn dry_run(args: &[&str], want: &[&str]) {
    let out = scratch("dry-run").to_string_lossy().into_owned();
    let mut args = args.to_vec();
    args.extend(["--out", &out, "--dry-run"]);
    let (ok, text) = es(&args);
    assert!(ok, "{args:?}\n{text}");
    for w in want {
        assert!(text.contains(w), "{w:?} not in\n{text}");
    }
}

#[test]
fn the_shadow_hand_project_generates_checks_and_builds() {
    let dir = generate("tests/fixtures/estask/shadow_hand_repose.estask", "hand");
    for arm in ["teacher", "student"] {
        check_and_build(&dir, arm);
    }
    let file = |f: &str| format!("{dir}/{f}");
    dry_run(
        &["train", "--recipe", &file("training-teacher.toml")],
        &["train_ppo", "mjwarp"],
    );
    dry_run(
        &["train", "--recipe", &file("training-student.toml")],
        &["train_act"],
    );
    dry_run(
        &["loop", "cycle", "--recipe", &file("cycle-student.toml")],
        &[
            "--policy runs/shadow-hand/teacher.esb",
            "--success-only",
            "evaluation-student-nominal.toml",
        ],
    );
}

#[test]
fn the_so101_views_project_generates_checks_and_builds() {
    let dir = generate("tests/fixtures/estask/so101_views.estask", "views");
    check_and_build(&dir, "views");
    let file = |f: &str| format!("{dir}/{f}");
    dry_run(
        &["train", "--recipe", &file("training-views.toml")],
        &["train_act"],
    );
    dry_run(
        &["loop", "cycle", "--recipe", &file("cycle-views.toml")],
        &["--expert so101-pick-place", "evaluation-views.toml"],
    );
}

#[test]
fn a_bad_spec_is_refused_by_name() {
    let dir = scratch("refused");
    std::fs::create_dir_all(&dir).expect("scratch");
    let spec = dir.join("bad.estask");
    let text = std::fs::read_to_string(repo().join("tests/fixtures/estask/so101_views.estask"))
        .expect("the spec")
        .replace("execute = 10", "execute = 3");
    std::fs::write(&spec, text).expect("write");
    let (ok, text) = es(&[
        "project",
        "generate",
        "--spec",
        &spec.to_string_lossy(),
        "--out",
        &dir.join("out").to_string_lossy(),
    ]);
    assert!(
        !ok && text.contains("student") && text.contains("execute"),
        "{text}"
    );
}
