//! `--jobs N`: the worker processes, their thread caps, and how a failed one is reported.

use super::args::RunArgs;
use crate::error::CliError;

// --- `--jobs N`: one worker process per shard (packet M5/V5) --------------------------------

/// Spawns one `es eval run --shard i/N --shard-out <file>` per shard, all at once, and
/// collects their cells in shard order.
///
/// Processes and not threads, because the physics backend is itself a Python subprocess with
/// one env in it and `TorchRuntime` holds another: threads here would queue behind the same
/// two interpreters (design note section 7.11). The same binary and the same documents, so a
/// worker is this code with a different shard, never a second implementation.
///
/// The children write their frames straight into the shared `--frames` directory: cell names
/// are globally unique and shards own disjoint cells, so there is nothing to merge and nothing
/// to collide.
/// The env vars that size a CPU math-library thread pool to every core on the box by default.
/// `TORCH_NUM_THREADS` is not an official `PyTorch` var but is harmless to set; `PyTorch`'s own
/// intra-op pool falls back to `OMP_NUM_THREADS`/`MKL_NUM_THREADS` when neither
/// `torch.set_num_threads` nor a build-time default has run yet, which is true at process start.
const THREAD_ENV_VARS: [&str; 4] = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TORCH_NUM_THREADS",
];

/// One shard's fair share of the box's cores, floored at 1. Measured on the 16-core oracle
/// server (design note section 7.11): six shards left uncapped each sized their own subprocess's
/// thread pool to all 16 cores, and `--jobs 6` ran *slower* than `--jobs 1` for it.
fn shard_thread_cap(jobs: u32, cores: usize) -> usize {
    (cores / jobs.max(1) as usize).max(1)
}

/// Which of [`THREAD_ENV_VARS`] this shard should set, and to what -- skipping any the caller
/// already chose a value for, since a value the user exported wins over every shard's guess
/// (`already_set` reads the real environment in production; the test below stubs it).
pub(super) fn shard_thread_env(
    jobs: u32,
    cores: usize,
    already_set: impl Fn(&str) -> bool,
) -> Vec<(&'static str, String)> {
    let cap = shard_thread_cap(jobs, cores).to_string();
    THREAD_ENV_VARS
        .into_iter()
        .filter(|name| !already_set(name))
        .map(|name| (name, cap.clone()))
        .collect()
}

/// `thread_env` is [`shard_thread_env`]'s answer, exported to every worker — and, by the
/// caller, to the parent's own torch runtime, so the lock quotes the workers' count.
pub(super) fn spawn_shards(
    a: &RunArgs,
    jobs: u32,
    thread_env: &[(&'static str, String)],
) -> Result<Vec<es_eval::Shard>, CliError> {
    let exe = std::env::current_exe()
        .map_err(|e| CliError::Runtime(format!("cannot find this executable to re-run it: {e}")))?;
    let dir = a.out.join("shards");
    std::fs::create_dir_all(&dir)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", dir.display())))?;

    let mut running = Vec::new();
    for i in 0..jobs {
        let path = dir.join(format!("{i}.json"));
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["eval", "run", "--config"])
            .arg(&a.config)
            .arg("--policy")
            .arg(&a.policy)
            .arg("--scene")
            .arg(&a.scene)
            .arg("--backend")
            .arg(&a.backend)
            .arg("--runtime")
            .arg(&a.runtime)
            .arg("--shard")
            .arg(format!("{i}/{jobs}"))
            .arg("--shard-out")
            .arg(&path)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if let Some(f) = &a.frames {
            cmd.arg("--frames").arg(f);
        }
        if let Some(e) = &a.expert {
            cmd.arg("--expert").arg(e);
        }
        // Cell names are globally unique and shards own disjoint cells, so the trajectories
        // share one directory exactly as the frames do, with nothing to merge.
        cmd.arg("--traj")
            .arg(a.traj.clone().unwrap_or_else(|| a.out.join("traj")));
        for (name, value) in thread_env {
            cmd.env(name, value);
        }
        let child = cmd
            .spawn()
            .map_err(|e| CliError::Runtime(format!("spawning {}: {e}", exe.display())))?;
        running.push((i, path, child));
    }

    // Every child is waited on before anything is returned, so a failure does not leave the
    // rest of them orphaned behind a `?`.
    let mut shards = Vec::with_capacity(running.len());
    let mut failed: Option<CliError> = None;
    for (i, path, child) in running {
        let out = match child.wait_with_output() {
            Ok(out) => out,
            Err(e) => {
                failed.get_or_insert(CliError::Runtime(format!(
                    "waiting for shard {i}/{jobs}: {e}"
                )));
                continue;
            }
        };
        if !out.status.success() {
            failed.get_or_insert(shard_failed(
                i,
                jobs,
                out.status.code(),
                &String::from_utf8_lossy(&out.stderr),
            ));
            continue;
        }
        match std::fs::read(&path)
            .map_err(|e| format!("{}: {e}", path.display()))
            .and_then(|b| {
                serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", path.display()))
            }) {
            Ok(s) => shards.push(s),
            Err(e) => {
                failed.get_or_insert(CliError::Runtime(format!(
                    "shard {i}/{jobs} exited 0 but its cells are unreadable: {e}"
                )));
            }
        }
    }
    if let Some(e) = failed {
        return Err(e);
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(shards)
}

/// The one error a failed worker becomes: which shard, what it exited with, and the last thing
/// it said. Never a partial report -- a merge missing a suite would wear a correct
/// `evaluation_hash` over numbers nobody measured (spec 10.4).
fn shard_failed(index: u32, count: u32, code: Option<i32>, stderr: &str) -> CliError {
    let last = stderr
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(no output)");
    CliError::Runtime(format!(
        "es eval run --shard {index}/{count} failed ({}): {last}",
        code.map_or_else(|| "killed by a signal".to_owned(), |c| format!("exit {c}")),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lost worker is one named error carrying enough to act on: which shard, how it died,
    /// and the last thing it printed.
    #[test]
    fn a_failed_shard_names_itself() {
        let CliError::Runtime(text) =
            shard_failed(2, 6, Some(3), "loading\nSKIPPED (no torch)\n\n")
        else {
            panic!("a failed shard must be a runtime error");
        };
        assert!(text.contains("--shard 2/6"), "{text}");
        assert!(text.contains("exit 3"), "{text}");
        assert!(text.contains("SKIPPED (no torch)"), "{text}");

        let CliError::Runtime(text) = shard_failed(0, 2, None, "") else {
            panic!("a killed shard must be a runtime error");
        };
        assert!(text.contains("killed by a signal"), "{text}");
        assert!(text.contains("(no output)"), "{text}");
    }

    /// Six shards on a 16-core box get 2 cores each, one shard gets the whole box, and a shard
    /// count above the core count still gets at least one (never a zero-thread pool).
    #[test]
    fn shard_thread_cap_divides_the_box_and_floors_at_one() {
        assert_eq!(shard_thread_cap(6, 16), 2);
        assert_eq!(shard_thread_cap(1, 16), 16);
        assert_eq!(shard_thread_cap(32, 16), 1);
    }

    /// The fix caps a pool the user left unset; it does not override one they chose. Regression
    /// for the oracle server measurement (design note section 7.11): uncapped, `--jobs 6` each
    /// sized their own torch subprocess to all 16 cores and the run was slower than `--jobs 1`.
    #[test]
    fn shard_thread_env_skips_a_var_the_user_already_set() {
        let env = shard_thread_env(6, 16, |name| name == "OMP_NUM_THREADS");
        assert!(!env.iter().any(|(name, _)| *name == "OMP_NUM_THREADS"));
        assert_eq!(env.len(), THREAD_ENV_VARS.len() - 1);
        assert!(env
            .iter()
            .any(|(name, value)| *name == "MKL_NUM_THREADS" && value == "2"));
    }

    /// Nothing set by the caller: every var is capped to the same fair share.
    #[test]
    fn shard_thread_env_caps_every_var_by_default() {
        let env = shard_thread_env(4, 16, |_| false);
        assert_eq!(env.len(), THREAD_ENV_VARS.len());
        assert!(env.iter().all(|(_, value)| value == "4"));
    }
}
