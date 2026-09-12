//! `cargo xtask layering` — spec 4.2 crate layering rules, table from Appendix B.8.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

/// Layer table, spec 4.2 / Appendix B.8. Layer 12 (`es-editor`) is included
/// only so rule 4 ("nothing depends on es-editor") can name it; nothing may
/// depend on it regardless of layer.
const LAYERS: &[(&str, u8)] = &[
    ("es-math", 0),
    ("es-core", 1),
    ("es-gpu", 2),
    ("es-assets", 2),
    ("es-usd", 2),
    ("es-actuator", 3),
    ("es-sensor", 3),
    ("es-physics-core", 3),
    ("es-physics-backend", 4),
    ("es-physics-cpu", 4),
    ("es-physics-gpu", 4),
    ("es-render", 5),
    ("es-splat", 5),
    ("es-ir", 6),
    ("es-compile", 7),
    ("es-policy", 8),
    ("es-safety", 8),
    ("es-env", 9),
    ("es-data", 10),
    ("es-telemetry", 10),
    ("es-eval", 10),
    ("es-ros2", 11),
    ("es-py", 11),
    ("es-script", 11),
    ("es-transport", 11),
    ("es-editor", 12),
];

fn layer_of(name: &str) -> Option<u8> {
    LAYERS.iter().find(|(n, _)| *n == name).map(|(_, l)| *l)
}

#[derive(Debug, Clone)]
pub struct Pkg {
    pub name: String,
    pub deps: Vec<String>,
}

/// Pure rule evaluation over a fake-able dependency map (spec 4.2 rules 1-8,
/// Appendix B.8 "additional checks"). Returns one message per violation.
pub fn check_layering(pkgs: &[Pkg]) -> Vec<String> {
    let mut errors = Vec::new();
    let workspace_names: BTreeMap<&str, &Pkg> = pkgs.iter().map(|p| (p.name.as_str(), p)).collect();

    for pkg in pkgs {
        let is_es = pkg.name.starts_with("es-");
        let pkg_layer = layer_of(&pkg.name);
        if is_es && pkg_layer.is_none() {
            errors.push(format!(
                "{}: unknown es-* crate not in the spec 4.2 LAYERS table",
                pkg.name
            ));
            continue;
        }

        for dep in &pkg.deps {
            // Rule 4: nothing depends on es-editor.
            if dep == "es-editor" {
                errors.push(format!(
                    "{}: must not depend on es-editor (rule 4)",
                    pkg.name
                ));
            }

            // Rule 5: only es-transport may depend on CUDA/HIP crates.
            if pkg.name != "es-transport"
                && (dep == "cust" || dep == "cudarc" || dep.starts_with("hip"))
            {
                errors.push(format!(
                    "{}: only es-transport may depend on CUDA/HIP crates, found `{}` (rule 5)",
                    pkg.name, dep
                ));
            }

            // Rule 3: layers <= 2, except es-gpu, must not depend on Vulkan crates.
            if let Some(l) = pkg_layer {
                if l <= 2 && pkg.name != "es-gpu" && (dep == "ash" || dep == "gpu-allocator") {
                    errors.push(format!(
                        "{}: layer <=2 must not depend on `{}` (rule 3)",
                        pkg.name, dep
                    ));
                }
            }

            // Rule 6: es-ir knows neither compiler, backends, nor torch.
            if pkg.name == "es-ir"
                && (dep == "es-compile"
                    || dep.starts_with("es-physics-")
                    || dep == "tch"
                    || dep.starts_with("torch")
                    || dep == "ort")
            {
                errors.push(format!("es-ir: must not depend on `{dep}` (rule 6)"));
            }

            // Rule 7: es-ir knows no UI types.
            if pkg.name == "es-ir" && (dep.starts_with("egui") || dep == "winit") {
                errors.push(format!("es-ir: must not depend on `{dep}` (rule 7)"));
            }

            // Rule 8: es-safety never depends on es-policy (INV-11).
            if pkg.name == "es-safety" && dep == "es-policy" {
                errors.push("es-safety: must not depend on es-policy (rule 8, INV-11)".into());
            }

            // Rule 2: es-physics-cpu and es-physics-gpu never depend on each other.
            if (pkg.name == "es-physics-cpu" && dep == "es-physics-gpu")
                || (pkg.name == "es-physics-gpu" && dep == "es-physics-cpu")
            {
                errors.push(format!("{}: must not depend on {} (rule 2)", pkg.name, dep));
            }

            // Rule 1: upper depends on strictly lower layers only; no same-layer deps.
            if let (Some(pl), Some(dep_pkg)) = (pkg_layer, workspace_names.get(dep.as_str())) {
                if dep_pkg.name.starts_with("es-") {
                    match layer_of(&dep_pkg.name) {
                        Some(dl) if dl < pl => {}
                        Some(dl) if dl == pl => errors.push(format!(
                            "{}: same-layer dependency on {} (layer {}) (rule 1)",
                            pkg.name, dep, dl
                        )),
                        Some(dl) => errors.push(format!(
                            "{}: (layer {}) depends on higher layer {} (layer {}) (rule 1)",
                            pkg.name, pl, dep, dl
                        )),
                        None => errors.push(format!(
                            "{dep}: unknown es-* crate not in the spec 4.2 LAYERS table"
                        )),
                    }
                }
            }
        }
    }
    errors.sort();
    errors.dedup();
    errors
}

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<MetaPkg>,
}

#[derive(Debug, Deserialize)]
struct MetaPkg {
    name: String,
    dependencies: Vec<MetaDep>,
}

#[derive(Debug, Deserialize)]
struct MetaDep {
    name: String,
}

fn workspace_packages(manifest_dir: &Path) -> Result<Vec<Pkg>, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(manifest_dir)
        .output()
        .map_err(|e| format!("failed to run `cargo metadata`: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo metadata` failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let meta: Metadata = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("failed to parse `cargo metadata` output: {e}"))?;
    Ok(meta
        .packages
        .into_iter()
        .map(|p| Pkg {
            name: p.name,
            deps: p.dependencies.into_iter().map(|d| d.name).collect(),
        })
        .collect())
}

pub fn run(workspace_root: &Path) -> bool {
    let pkgs = match workspace_packages(workspace_root) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return false;
        }
    };
    let errors = check_layering(&pkgs);
    if errors.is_empty() {
        println!("layering: {} crates checked, no violations", pkgs.len());
        true
    } else {
        for e in &errors {
            println!("layering violation: {e}");
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pkg(name: &str, deps: &[&str]) -> Pkg {
        Pkg {
            name: name.into(),
            deps: deps.iter().map(ToString::to_string).collect(),
        }
    }

    #[test]
    fn allows_strictly_lower_layer_deps() {
        let pkgs = vec![pkg("es-math", &[]), pkg("es-core", &["es-math"])];
        assert!(check_layering(&pkgs).is_empty());
    }

    #[test]
    fn rejects_same_layer_dep() {
        let pkgs = vec![pkg("es-assets", &[]), pkg("es-gpu", &["es-assets"])];
        let errors = check_layering(&pkgs);
        assert!(
            errors.iter().any(|e| e.contains("same-layer")),
            "{errors:?}"
        );
    }

    #[test]
    fn rejects_upward_dep() {
        let pkgs = vec![pkg("es-math", &["es-core"]), pkg("es-core", &[])];
        let errors = check_layering(&pkgs);
        assert!(
            errors.iter().any(|e| e.contains("higher layer")),
            "{errors:?}"
        );
    }

    #[test]
    fn rejects_physics_cpu_gpu_cross_dep() {
        let pkgs = vec![
            pkg("es-physics-cpu", &["es-physics-gpu"]),
            pkg("es-physics-gpu", &[]),
        ];
        let errors = check_layering(&pkgs);
        assert!(errors.iter().any(|e| e.contains("rule 2")), "{errors:?}");
    }

    #[test]
    fn rejects_low_layer_vulkan_dep() {
        let pkgs = vec![pkg("es-core", &["ash"])];
        let errors = check_layering(&pkgs);
        assert!(errors.iter().any(|e| e.contains("rule 3")), "{errors:?}");
    }

    #[test]
    fn allows_es_gpu_vulkan_dep() {
        let pkgs = vec![pkg("es-gpu", &["ash", "gpu-allocator"])];
        assert!(check_layering(&pkgs).is_empty());
    }

    #[test]
    fn rejects_dep_on_editor() {
        let pkgs = vec![pkg("es-env", &["es-editor"]), pkg("es-editor", &[])];
        let errors = check_layering(&pkgs);
        assert!(errors.iter().any(|e| e.contains("rule 4")), "{errors:?}");
    }

    #[test]
    fn rejects_cuda_outside_transport() {
        let pkgs = vec![pkg("es-policy", &["cust"])];
        let errors = check_layering(&pkgs);
        assert!(errors.iter().any(|e| e.contains("rule 5")), "{errors:?}");
    }

    #[test]
    fn allows_cuda_in_transport() {
        let pkgs = vec![pkg("es-transport", &["cust", "cudarc", "hip-sys"])];
        assert!(check_layering(&pkgs).is_empty());
    }

    #[test]
    fn rejects_ir_on_compiler_and_torch() {
        let pkgs = vec![pkg(
            "es-ir",
            &["es-compile", "es-physics-cpu", "tch", "torch-sys", "ort"],
        )];
        let errors = check_layering(&pkgs);
        assert!(
            errors.iter().any(|e| e.contains("es-compile")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.contains("es-physics-cpu") && e.contains("rule 6")),
            "{errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.contains('`') && e.contains("tch")),
            "{errors:?}"
        );
    }

    #[test]
    fn rejects_ir_on_ui() {
        let pkgs = vec![pkg("es-ir", &["egui", "winit"])];
        let errors = check_layering(&pkgs);
        assert_eq!(errors.len(), 2);
        assert!(errors.iter().all(|e| e.contains("rule 7")));
    }

    #[test]
    fn rejects_safety_on_policy() {
        let pkgs = vec![pkg("es-safety", &["es-policy"]), pkg("es-policy", &[])];
        let errors = check_layering(&pkgs);
        assert!(errors.iter().any(|e| e.contains("INV-11")), "{errors:?}");
    }

    #[test]
    fn rejects_unknown_es_crate() {
        let pkgs = vec![pkg("es-mystery", &[])];
        let errors = check_layering(&pkgs);
        assert!(
            errors.iter().any(|e| e.contains("unknown es-* crate")),
            "{errors:?}"
        );
    }

    #[test]
    fn ignores_non_es_dependencies() {
        let pkgs = vec![pkg("es-core", &["serde", "blake3"])];
        assert!(check_layering(&pkgs).is_empty());
    }
}
