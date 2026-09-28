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

/// The facts both outputs report, gathered once.
struct Deps {
    python: Option<String>,
    mujoco: bool,
    torch: bool,
    lerobot: bool,
    vulkan: bool,
    /// `cfg!(feature = "render")` of this binary: `es eval run --frames` needs it.
    render: bool,
    backends: Vec<(&'static str, Result<(), String>)>,
}

fn gather() -> Deps {
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
    // `--backend` of `es eval run` / `es loop collect` / `[rl] backend` (packet M11/X1).
    let backends = es_physics_backend::BackendKind::ALL
        .into_iter()
        .map(|kind| (kind.name(), es_physics_backend::is_available(kind)))
        .collect();
    Deps {
        python,
        mujoco,
        torch,
        lerobot,
        vulkan: vulkan_loader_present(),
        render: cfg!(feature = "render"),
        backends,
    }
}

/// Schema 1, one line: what the editor's start screen reads (design note editor-redesign 6.6).
/// `python.path` is present only when an interpreter was found; `reason` only when a backend
/// is unavailable.
fn to_json(d: &Deps) -> String {
    let mut python = serde_json::json!({ "found": d.python.is_some() });
    if let Some(p) = &d.python {
        python["path"] = p.as_str().into();
    }
    let backends: Vec<serde_json::Value> = d
        .backends
        .iter()
        .map(|(name, status)| match status {
            Ok(()) => serde_json::json!({ "name": name, "available": true }),
            Err(why) => serde_json::json!({ "name": name, "available": false, "reason": why }),
        })
        .collect();
    serde_json::json!({
        "schema": 1,
        "python": python,
        "modules": { "mujoco": d.mujoco, "torch": d.torch, "lerobot": d.lerobot },
        "vulkan_loader": d.vulkan,
        "render": d.render,
        "backends": backends,
    })
    .to_string()
}

/// Always returns `0`: this command reports, it never fails (spec 2.5). `json` prints the same
/// facts as one JSON object instead of the text.
pub fn run(json: bool) -> u8 {
    let deps = gather();
    if json {
        println!("{}", to_json(&deps));
        return 0;
    }
    let Deps {
        python,
        mujoco,
        torch,
        lerobot,
        vulkan,
        backends,
        ..
    } = deps;

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
    println!("physics backends (--backend, spec 17.2):");
    for (name, status) in backends {
        let status = match status {
            Ok(()) => "available".to_owned(),
            Err(why) => format!("unavailable ({why})"),
        };
        println!("  {name:<11} {status}");
    }
    println!(
        "  observation/preprocessing GPU lowering   {}  (unneeded on a Slang cache hit)",
        yes_no(vulkan)
    );
    0
}
