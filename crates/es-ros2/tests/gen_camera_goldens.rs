//! Provenance harness for `tests/golden/ros2/camera/**` (spec 1.4): the camera goldens are
//! produced by `crates/es-ros2/python/gen_camera_goldens.py`, which calls rosbags, `OpenCV`,
//! `image_geometry` and `cv_bridge` and never touches `es-ros2`. This test re-runs both of the
//! script's modes into a temp directory and fails if a single byte differs from the checked-in
//! files (the `tests/gen_goldens.rs` pattern).
//!
//! Each part **SKIPs, loudly, when its oracle is absent** — a missing interpreter or ROS 2
//! environment must not be reported as a passing provenance check — and `RAN
//! gen_camera_goldens` is printed only when both parts ran and every file matched.
//!
//! ```text
//! ES_PYTHON=<venv>/bin/python cargo test -p es-ros2 --test gen_camera_goldens -- --nocapture
//! ES_ROS2_ENV=<prefix> ES_PYTHON=<venv>/bin/python \
//!     cargo test -p es-ros2 --test gen_camera_goldens -- --nocapture
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

/// The first interpreter that can import both `rosbags` and `cv2`, or why none could.
fn oracle_python() -> Result<String, String> {
    let mut tried = Vec::new();
    for python in python_candidates() {
        match Command::new(&python)
            .args(["-c", "import rosbags, cv2"])
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
        "no Python interpreter with rosbags + cv2 (set ES_PYTHON to choose one): {}",
        tried.join("; ")
    ))
}

fn assert_ok(what: &str, out: &std::process::Output) {
    println!("{}", String::from_utf8_lossy(&out.stdout));
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stderr.is_empty() {
        println!("stderr:\n{stderr}");
    }
    assert!(out.status.success(), "{what} failed; see the report above");
}

/// Every regular file under `dir` whose path starts with `prefix`, relative to `dir`
/// (POSIX-style separators, so the comparison is stable across Windows and Linux).
fn list_files(dir: &Path, prefix: &str) -> Vec<String> {
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
    out.retain(|n| n.starts_with(prefix));
    out.sort();
    out
}

/// The freshly generated files under `prefix` must be exactly the checked-in ones, byte for
/// byte.
fn compare(fresh: &Path, checked_in: &Path, prefix: &str) {
    let names = list_files(fresh, prefix);
    assert!(!names.is_empty(), "the oracle produced no `{prefix}` files");
    assert_eq!(
        names,
        list_files(checked_in, prefix),
        "the oracle produced a different `{prefix}` file set than what is checked in"
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
fn camera_goldens_are_what_the_reference_oracles_produce() {
    let fresh = std::env::temp_dir().join("es-ros2-camera-goldens");
    let _ = std::fs::remove_dir_all(&fresh);
    std::fs::create_dir_all(&fresh).expect("create the temp golden directory");
    let checked_in = manifest("../../tests/golden/ros2/camera");

    let python = match oracle_python() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP gen_camera_goldens: {why}");
            return;
        }
    };
    let script = manifest("python/gen_camera_goldens.py");
    let out = Command::new(&python)
        .arg(&script)
        .arg(&fresh)
        .output()
        .unwrap_or_else(|e| panic!("spawn {python} {}: {e}", script.display()));
    assert_ok("gen_camera_goldens.py (rosbags + OpenCV)", &out);
    compare(&fresh, &checked_in, "cdr/");
    compare(&fresh, &checked_in, "yuv/");

    // The `--ros` part reads `yuv/*.rgb` back to cross-check `cv_bridge` against `cv2`, so it
    // only runs after the part above wrote them into the same directory.
    let Ok(ros2_env) = std::env::var("ES_ROS2_ENV") else {
        println!("SKIP gen_camera_goldens: ES_ROS2_ENV is unset (image_geometry / cv_bridge)");
        return;
    };
    let env_script = manifest("scripts/ros2-env.sh");
    let out = Command::new("bash")
        .arg(&env_script)
        .arg(&ros2_env)
        .arg("python")
        .arg(&script)
        .arg("--ros")
        .arg(&fresh)
        .output()
        .unwrap_or_else(|e| panic!("spawn {}: {e}", env_script.display()));
    assert_ok(
        "gen_camera_goldens.py --ros (image_geometry + cv_bridge)",
        &out,
    );
    compare(&fresh, &checked_in, "intrinsics.json");
    compare(&fresh, &checked_in, "cv_bridge.json");

    assert_eq!(
        list_files(&checked_in, ""),
        list_files(&fresh, ""),
        "the checked-in camera goldens are not exactly what the two oracles produce"
    );
    println!("RAN gen_camera_goldens");
}
