//! Oracle 3 of packet M7/T5: the `ImageNet` backbone is a *pinned* artifact.
//!
//! `python/es/fetch_backbone.py` is the only place in this project where a pickle is read, and
//! torchvision is what reads it, inside its own downloader, on the learning path (spec 2.3,
//! INV-16). What comes out is a safetensors file and a lock file, and this test is what says
//! the two are what the repository claims they are:
//!
//! 1. the script writes both files, and the safetensors' blake3 equals the pin in
//!    `crates/es-data/src/training.rs`;
//! 2. the lock file names the licence spec 19.3 makes it the basis for tracking;
//! 3. a run against a *tampered* pin, without `--repin`, refuses by name and writes nothing.
//!
//! `#[ignore]`d because it needs torchvision and, the first time, the network. A machine
//! without either prints `SKIP` and its reason rather than pretending (spec 1.4). On the
//! oracle server the artifact lives at `~/artifacts/plan-v/m7-t5/`; the 45 MB file is never
//! committed, which is why the pin exists at all.
//!
//! ```text
//! ES_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python \
//!   cargo test -p es-policy --test backbone_provenance -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// The pinned blake3, read out of `crates/es-data/src/training.rs` as text.
///
/// One copy of the number exists in this repository and this is not it: `es-data` is layer 10
/// and this crate is layer 8 (spec 4.2), so the constant cannot be imported, and a second
/// literal would be a second thing to forget when it moves.
fn pinned_blake3() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../es-data/src/training.rs");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let after = text
        .split_once("pub const RESNET18_IMAGENET1K_V1_BLAKE3: &str =")
        .unwrap_or_else(|| panic!("{}: the pin is gone", path.display()))
        .1;
    after
        .split('"')
        .nth(1)
        .expect("the pin is a string literal")
        .to_owned()
}

fn fetch_backbone_py() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/es/fetch_backbone.py")
}

fn scratch_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("es-backbone-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// The same probe every Python oracle in this repository uses: "no torchvision here" is a skip
/// and everything after it is an error (`docs/reviews/M4.md:67`, S-7).
fn python_with_torchvision() -> Result<String, String> {
    let python = std::env::var("ES_PYTHON").map_err(|_| "ES_PYTHON is not set".to_owned())?;
    let out = Command::new(&python)
        .args(["-c", "import torchvision, blake3"])
        .output()
        .map_err(|e| format!("{python}: {e}"))?;
    if out.status.success() {
        Ok(python)
    } else {
        Err(format!(
            "{python}: {}",
            String::from_utf8_lossy(&out.stderr).trim_end()
        ))
    }
}

/// Runs the script; returns (success, stdout, stderr).
fn fetch(python: &str, out: &Path, expect: &str, repin: bool) -> (bool, String, String) {
    let mut cmd = Command::new(python);
    cmd.arg(fetch_backbone_py())
        .args(["--arch", "resnet18", "--out"])
        .arg(out)
        .args(["--expect", expect]);
    if repin {
        cmd.arg("--repin");
    }
    let done = cmd.output().expect("run fetch_backbone.py");
    (
        done.status.success(),
        String::from_utf8_lossy(&done.stdout).into_owned(),
        String::from_utf8_lossy(&done.stderr).into_owned(),
    )
}

#[test]
#[ignore = "needs torchvision, blake3 and (once) the network"]
fn the_backbone_artifact_matches_the_pin() {
    let python = match python_with_torchvision() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP the_backbone_artifact_matches_the_pin: {why}");
            return;
        }
    };
    let pin = pinned_blake3();
    let dir = scratch_dir("provenance");

    // 1. The script writes the artifact, and it hashes to the pin. Passing `--expect` is what
    //    makes that a refusal rather than a comparison this test does afterwards: the file is
    //    never left on disk under its real name unless it agrees.
    let (ok, stdout, stderr) = fetch(&python, &dir, &pin, false);
    assert!(ok, "fetch_backbone.py failed:\n{stdout}{stderr}");
    let weights = dir.join("resnet18-imagenet1k-v1.safetensors");
    let lock_path = dir.join("resnet18-imagenet1k-v1.lock.json");
    assert!(weights.is_file(), "no safetensors at {}", weights.display());
    assert!(
        !dir.join("resnet18-imagenet1k-v1.safetensors.partial")
            .exists(),
        "the scratch file survived"
    );
    let bytes = std::fs::read(&weights).expect("read the artifact");
    let mut digest = String::new();
    for byte in blake3::hash(&bytes).as_bytes() {
        use std::fmt::Write as _;
        let _ = write!(digest, "{byte:02x}");
    }
    assert_eq!(digest, pin, "the artifact does not hash to the pin");

    // 2. The lock file is the provenance spec 19.3 asks for, and its licence slot is filled.
    let lock: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&lock_path).expect("the lock file"))
            .expect("the lock file is JSON");
    assert_eq!(lock["blake3"], pin);
    assert_eq!(lock["license"], "BSD-3-Clause");
    assert_eq!(
        lock["source"], "torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
        "the source is the licence decision (spec 29 row, owner 2026-09-15)"
    );
    for named in ["url", "sha256_upstream", "license_url", "torchvision"] {
        assert!(
            lock[named].as_str().is_some_and(|s| !s.is_empty()),
            "the lock file's `{named}` is empty: {lock}"
        );
    }

    // 3. A tampered pin is refused by name, and nothing is written under the real name.
    let tampered = scratch_dir("provenance-tampered");
    let wrong = "0".repeat(64);
    let (ok, stdout, stderr) = fetch(&python, &tampered, &wrong, false);
    assert!(!ok, "a mismatching pin was accepted:\n{stdout}");
    let said = format!("{stdout}{stderr}");
    assert!(
        said.contains("refusing to write") && said.contains(&wrong) && said.contains(&pin),
        "the refusal names neither hash:\n{said}"
    );
    assert!(
        said.contains("RESNET18_IMAGENET1K_V1_BLAKE3"),
        "the refusal does not name the pin to move:\n{said}"
    );
    assert!(
        !tampered.join("resnet18-imagenet1k-v1.safetensors").exists(),
        "the refusal still left the file behind"
    );

    // ... and `--repin` is the deliberate way through, which says so on stderr.
    let (ok, _, stderr) = fetch(&python, &tampered, &wrong, true);
    assert!(ok, "--repin was refused: {stderr}");
    assert!(stderr.contains("--repin"), "{stderr}");
    assert!(tampered
        .join("resnet18-imagenet1k-v1.safetensors")
        .is_file());

    println!("RAN the_backbone_artifact_matches_the_pin: {pin}\n  {lock}");
}
