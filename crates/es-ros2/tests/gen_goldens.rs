//! Provenance harness for `tests/golden/ros2/**` (spec 1.4): the goldens are produced by
//! `crates/es-ros2/python/gen_ros2_goldens.py`, which calls rosbags and `PyPI` xxhash and never
//! touches `es-ros2`. This test re-runs that script into a temp directory and fails if a
//! single byte differs from the checked-in files (`goldens_are_what_rosbags_and_xxhash_produce`,
//! the `crates/es-compile/tests/gen_goldens.rs` pattern), and separately cross-checks the same
//! goldens against a live `RoboStack` ROS 2 Kilted install
//! (`robostack_hashes_and_rclpy_bytes_agree_with_the_goldens`), which is the live-capture oracle
//! `docs/api-notes/ros2-cdr.md`'s "Live ROS 2 byte capture" row and
//! `docs/api-notes/rmw-zenoh.md`'s "Open items" table point at.
//!
//! Both **SKIP, loudly, when their oracle is absent** — a missing interpreter or environment
//! must not be reported as a passing provenance check.
//!
//! ```text
//! ES_PYTHON=<venv>/bin/python cargo test -p es-ros2 --test gen_goldens -- --nocapture
//! ES_ROS2_ENV=<prefix> ES_PYTHON=<venv>/bin/python \
//!     cargo test -p es-ros2 --test gen_goldens -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn python_candidates() -> Vec<String> {
    match std::env::var("ES_PYTHON") {
        Ok(p) if !p.is_empty() => vec![p],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    }
}

/// The first interpreter that can import both `rosbags` and `xxhash`, or why none could.
fn oracle_python() -> Result<String, String> {
    let mut tried = Vec::new();
    for python in python_candidates() {
        match Command::new(&python)
            .args(["-c", "import rosbags, xxhash"])
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
        "no Python interpreter with rosbags + xxhash (set ES_PYTHON to choose one): {}",
        tried.join("; ")
    ))
}

fn run_generate(python: &str, out_dir: &Path) {
    let script = manifest("python/gen_ros2_goldens.py");
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

/// Every regular file under `dir`, as paths relative to `dir` (POSIX-style separators, so the
/// comparison is stable across Windows and Linux).
fn list_files_relative(dir: &Path) -> Vec<String> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(base, &path, out);
            } else {
                let rel = path.strip_prefix(base).unwrap();
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

#[test]
fn goldens_are_what_rosbags_and_xxhash_produce() {
    let python = match oracle_python() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP goldens_are_what_rosbags_and_xxhash_produce: {why}");
            return;
        }
    };

    let fresh = std::env::temp_dir().join("es-ros2-cdr-goldens");
    let _ = std::fs::remove_dir_all(&fresh);
    run_generate(&python, &fresh);
    println!("RAN gen_goldens");

    let checked_in = manifest("../../tests/golden/ros2");
    let names = list_files_relative(&checked_in);
    assert!(!names.is_empty(), "no goldens to check");
    assert_eq!(
        names,
        list_files_relative(&fresh),
        "the oracle produced a different file set than what is checked in"
    );

    for name in names {
        let want = std::fs::read(checked_in.join(&name)).expect("read checked-in golden");
        let got = std::fs::read(fresh.join(&name)).expect("read freshly generated golden");
        assert!(
            want == got,
            "`{name}` differs from what the oracle just produced ({} vs {} bytes)",
            want.len(),
            got.len()
        );
    }
}

#[test]
fn robostack_hashes_and_rclpy_bytes_agree_with_the_goldens() {
    let Ok(ros2_env) = std::env::var("ES_ROS2_ENV") else {
        println!(
            "SKIP robostack_hashes_and_rclpy_bytes_agree_with_the_goldens: ES_ROS2_ENV is unset"
        );
        return;
    };

    let script = manifest("scripts/ros2-env.sh");
    let golden_dir = manifest("../../tests/golden/ros2");
    let out = Command::new("sh")
        .arg(&script)
        .arg(&ros2_env)
        .arg("python")
        .arg(manifest("python/gen_ros2_goldens.py"))
        .arg("--check-ros")
        .arg(&golden_dir)
        .env("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")
        .output()
        .unwrap_or_else(|e| panic!("spawn {}: {e}", script.display()));

    println!("RAN robostack_hashes_and_rclpy_bytes_agree_with_the_goldens");
    println!("{}", String::from_utf8_lossy(&out.stdout));
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stderr.is_empty() {
        println!("stderr:\n{stderr}");
    }
    assert!(
        out.status.success(),
        "live RoboStack cross-check reported a difference; see the report above and record the \
         outcome in docs/api-notes/ros2-cdr.md's \"Live ROS 2 byte capture\" row"
    );
}
