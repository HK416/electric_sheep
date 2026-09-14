//! `cargo xtask` — the single entry point for all verification (spec 1.6, 26.2).

mod context_budget;
mod goldens;
mod layering;
mod nostd;
mod scope;
mod spec_refs;

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

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

/// Words that mark a `SKIP` line as one of the GPU-dependent oracles (`es-gpu`, `es-render`,
/// `es-compile`'s GPU lowering). Matched case-insensitively over the rest of the line.
const GPU_SKIP_WORDS: [&str; 5] = ["gpu", "vulkan", "render", "slangc", "device"];

/// Whether a captured line is a GPU oracle reporting that it did not run.
///
/// Pure, so xtask's own tests cover it without a GPU (review `docs/reviews/M4.md` S-7).
fn is_gpu_skip(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("SKIP") else {
        return false;
    };
    let rest = rest.to_ascii_lowercase();
    GPU_SKIP_WORDS.iter().any(|w| rest.contains(w))
}

/// `cargo test --workspace` with `--nocapture`, forwarding the output and scanning it.
///
/// `--nocapture` is what makes the `SKIP` lines visible at all: without it a GPU-less
/// machine prints nothing and reports `ok`. With `ES_REQUIRE_GPU=1` — a machine that claims
/// a GPU — a GPU SKIP is a failure, the same way `nostd --require` turns a missing target
/// into one. Unset (the PR runner, which has no GPU) it is only reported.
///
/// `ES_REQUIRE_ORACLES=1` is the symmetric flag for every other reference oracle (`MuJoCo`,
/// `PyTorch`, the `es-ros2` golden provenance harnesses, …): any `SKIP` line `is_gpu_skip`
/// does **not** claim is one of those, and with the flag set it fails the run the same way a
/// GPU skip does under `ES_REQUIRE_GPU=1`. Unset, it is only a `NOTE` (review `docs/reviews/
/// M3-W1.md` S-3, packet `docs/packets/M3/P-M3-W1-R4.md`).
fn run_tests(root: &Path) -> bool {
    // `es-ros2/zenoh` (docs/packets/M3/W1b-ros2-zenoh-session.md): off by default (spec 4.2 —
    // `es` and any embedded consumer must not link zenoh), but the PR tier still builds, lints
    // and runs the in-process loopback session tests against it. `es-env/render`
    // (docs/packets/M5/V0b-render-in-the-loop.md): same precedent, so the render-in-the-loop
    // golden and the GPU==CPU frame check are gated here rather than never.
    let args = [
        "test",
        "--workspace",
        "--features",
        "es-ir/testing,es-ros2/zenoh,es-env/render,es/render",
        "--",
        "--nocapture",
    ];
    println!("$ cargo {}", args.join(" "));
    let mut child = match Command::new("cargo")
        .args(args)
        .current_dir(root)
        .stdout(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            eprintln!("cannot start `cargo test`: {e}");
            return false;
        }
    };
    let stdout = child.stdout.take().expect("stdout is piped");
    let mut gpu_skipped: Vec<String> = Vec::new();
    let mut other_skipped: Vec<String> = Vec::new();
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        println!("{line}");
        if line.starts_with("SKIP") {
            if is_gpu_skip(&line) {
                gpu_skipped.push(line);
            } else {
                other_skipped.push(line);
            }
        }
    }
    if !child.wait().is_ok_and(|s| s.success()) {
        return false;
    }

    let require_gpu = std::env::var("ES_REQUIRE_GPU").as_deref() == Ok("1");
    let mut ok = true;
    for line in &gpu_skipped {
        if require_gpu {
            eprintln!("FAIL ES_REQUIRE_GPU=1 but a GPU oracle did not run: {line}");
            ok = false;
        } else {
            println!("NOTE GPU oracle skipped (ES_REQUIRE_GPU unset): {line}");
        }
    }

    let require_oracles = std::env::var("ES_REQUIRE_ORACLES").as_deref() == Ok("1");
    for line in &other_skipped {
        if require_oracles {
            eprintln!("FAIL ES_REQUIRE_ORACLES=1 but a reference oracle did not run: {line}");
            ok = false;
        } else {
            println!("NOTE oracle skipped (ES_REQUIRE_ORACLES unset): {line}");
        }
    }
    ok
}

fn cmd_ci(root: &Path) -> bool {
    run_cargo(root, &["fmt", "--check"])
        && run_cargo(
            root,
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--features",
                "es-ros2/zenoh,es-env/render,es/render",
                "--",
                "-D",
                "warnings",
            ],
        )
        && run_tests(root)
        && context_budget::run(&root.join("crates"))
        && layering::run(root)
        && nostd::run(root, false)
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
        Some("nostd") => nostd::run(&root, args.next().as_deref() == Some("--require")),
        Some("ci") => cmd_ci(&root),
        Some(other) => {
            eprintln!("unknown xtask command: {other}");
            eprintln!(
                "available: context-budget, layering, check-spec-refs, verify-goldens, check-scope <packet.md>, nostd [--require], ci"
            );
            false
        }
        None => {
            eprintln!(
                "usage: cargo xtask <context-budget|layering|check-spec-refs|verify-goldens|check-scope|nostd|ci>"
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

#[cfg(test)]
mod tests {
    use super::is_gpu_skip;

    #[test]
    fn spots_the_gpu_oracles_reporting_that_they_did_not_run() {
        for line in [
            "SKIP tree_reduction: no Vulkan device (no Vulkan loader: ...)",
            "SKIP execution_modes_are_present_in_the_spirv: no slangc (slangc not found)",
            "SKIP path_traced_cornell_box_matches_the_golden: no GPU",
            "SKIP render: no device",
        ] {
            assert!(is_gpu_skip(line), "missed: {line}");
        }
    }

    #[test]
    fn leaves_other_skips_and_other_lines_alone() {
        for line in [
            "SKIP nostd: target thumbv7em-none-eabihf not installed",
            "SKIP: ES_ACT_CHECKPOINT is unset",
            "SKIP: no Python with `torch`",
            "    SKIP tree_reduction: no Vulkan device", // not at the start of the line
            "test gpu_thing ... ok",
            "RAN device: NVIDIA GeForce RTX 4060",
        ] {
            assert!(!is_gpu_skip(line), "false positive: {line}");
        }
    }

    /// The es-ros2 golden-provenance SKIP lines carry no GPU word by design (design note
    /// `docs/design/ros2-boundary.md` section 8), so `ES_REQUIRE_ORACLES=1` — not
    /// `ES_REQUIRE_GPU=1` — is what catches them (review `docs/reviews/M3-W1.md` S-3).
    #[test]
    fn a_non_gpu_skip_is_not_a_gpu_skip() {
        let line =
            "SKIP gen_camera_goldens: no Python interpreter with rosbags + cv2 (set ES_PYTHON to choose one)";
        assert!(!is_gpu_skip(line), "false positive: {line}");
    }
}
