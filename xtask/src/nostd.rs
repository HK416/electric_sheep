//! `cargo xtask nostd` (spec W2; review follow-up `docs/packets/M3/P-M3-R4.md`): builds the
//! `no_std` embedded target so the split does not rot silently. SKIPs (exit success) when the
//! target isn't installed locally — `--require` turns that SKIP into a failure, which is what
//! CI passes once its toolchain step installs the target.

use std::path::Path;
use std::process::Command;

const TARGET: &str = "thumbv7em-none-eabihf";

/// Whether `rustup target list --installed`'s output lists `target`. A pure string check so
/// the parser is testable without a real `rustup` invocation.
fn target_installed(installed: &str, target: &str) -> bool {
    installed.lines().any(|line| line.trim() == target)
}

fn installed_targets() -> String {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

pub fn run(root: &Path, require: bool) -> bool {
    if !target_installed(&installed_targets(), TARGET) {
        let msg = format!("target {TARGET} not installed (rustup target add {TARGET})");
        if require {
            eprintln!("FAIL nostd: {msg}");
            return false;
        }
        println!("SKIP nostd: {msg}");
        return true;
    }

    println!(
        "$ cargo build -p es-core -p es-safety -p es-runtime-embedded --no-default-features --target {TARGET}"
    );
    Command::new("cargo")
        .args([
            "build",
            "-p",
            "es-core",
            "-p",
            "es-safety",
            "-p",
            "es-runtime-embedded",
            "--no-default-features",
            "--target",
            TARGET,
        ])
        .current_dir(root)
        .status()
        .is_ok_and(|s| s.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_installed_target() {
        let installed = "thumbv7em-none-eabihf\nx86_64-pc-windows-msvc\n";
        assert!(target_installed(installed, TARGET));
    }

    #[test]
    fn detects_missing_target() {
        let installed = "x86_64-pc-windows-msvc\nwasm32-unknown-unknown\n";
        assert!(!target_installed(installed, TARGET));
    }

    #[test]
    fn tolerates_trailing_whitespace_and_empty_output() {
        assert!(target_installed("thumbv7em-none-eabihf\r\n", TARGET));
        assert!(!target_installed("", TARGET));
    }
}
