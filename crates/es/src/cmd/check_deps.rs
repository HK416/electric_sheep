//! `es --check-deps` (spec 2.5): report per-environment capability. Never fails -- a missing
//! tool is reported, not an error.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long a probed `python -c "import ..."` gets before it is killed and counted as absent.
const PY_TIMEOUT: Duration = Duration::from_secs(10);

/// `ES_PYTHON` overrides the search entirely, exactly like the mujoco-cpu backend (spec 2.4).
fn python_candidates() -> Vec<String> {
    match std::env::var("ES_PYTHON") {
        Ok(p) if !p.trim().is_empty() => vec![p],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    }
}

fn find_python() -> Option<String> {
    python_candidates().into_iter().find(|p| {
        Command::new(p)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// `python -c "import <module>"`, with a timeout so a wedged interpreter cannot hang the CLI.
fn module_importable(python: &str, module: &str) -> bool {
    let Ok(mut child) = Command::new(python)
        .args(["-c", &format!("import {module}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) if start.elapsed() > PY_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(_) => return false,
        }
    }
}

/// File-existence heuristic only -- not a driver probe (spec: "heuristic").
fn vulkan_loader_present() -> bool {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        candidates.push(PathBuf::from(r"C:\Windows\System32\vulkan-1.dll"));
        if let Ok(path) = std::env::var("PATH") {
            candidates.extend(std::env::split_paths(&path).map(|p| p.join("vulkan-1.dll")));
        }
    } else {
        for dir in [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
            "/usr/local/lib",
        ] {
            candidates.push(PathBuf::from(dir).join("libvulkan.so.1"));
        }
    }
    candidates.iter().any(|p| p.exists())
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// Always returns `0`: this command reports, it never fails (spec 2.5).
pub fn run() -> u8 {
    let python = find_python();
    // `MuJoCoCpuBackend::is_available` does its own interpreter search (honoring `ES_PYTHON`
    // too), so it is not gated on `python` being found by this module's own search.
    let mujoco = es_physics_backend::mujoco::MuJoCoCpuBackend::is_available().is_ok();
    let torch = python
        .as_deref()
        .is_some_and(|p| module_importable(p, "torch"));
    let lerobot = python
        .as_deref()
        .is_some_and(|p| module_importable(p, "lerobot"));
    let vulkan = vulkan_loader_present();

    println!("es --check-deps (spec 2.5)");
    println!();
    match &python {
        Some(p) => println!("Python interpreter:    found ({p})"),
        None => println!("Python interpreter:    not found (tried python, python3; set ES_PYTHON)"),
    }
    println!("  mujoco importable:    {}", yes_no(mujoco));
    println!("  torch importable:     {}", yes_no(torch));
    println!("  lerobot importable:   {}", yes_no(lerobot));
    println!(
        "Vulkan loader:          {} (heuristic: file existence only, not a driver probe)",
        yes_no(vulkan)
    );
    println!();
    println!("es capabilities (spec 2.5):");
    println!(
        "  learning path (Python + PyTorch)        {}",
        yes_no(torch)
    );
    println!(
        "  MuJoCo (CPU) reference oracle            {}",
        yes_no(mujoco)
    );
    println!(
        "  LeRobot dataset export (needs Python)    {}",
        yes_no(lerobot)
    );
    println!("  LeRobot dataset read (Rust-native)       yes  (no Python needed)");
    println!(
        "  observation/preprocessing GPU lowering   {}  (unneeded on a Slang cache hit)",
        yes_no(vulkan)
    );
    0
}
