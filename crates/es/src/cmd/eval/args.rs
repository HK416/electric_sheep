//! `es eval run`'s command line, checked before anything opens.

use std::path::PathBuf;

use super::RUN_HELP;
use crate::error::CliError;

// --- `es eval run` ---------------------------------------------------------------------------

#[derive(Debug)]
pub(super) struct RunArgs {
    pub(super) config: String,
    pub(super) policy: String,
    pub(super) scene: String,
    pub(super) out: PathBuf,
    pub(super) frames: Option<PathBuf>,
    /// The scripted demonstrator, driving instead of the bundle's weights (spec 28.9 rule 1).
    pub(super) expert: Option<String>,
    /// Where the per-episode `.estraj` state trajectories go; `<out>/traj` unless named.
    pub(super) traj: Option<PathBuf>,
    pub(super) backend: String,
    pub(super) runtime: String,
    /// Worker processes to partition the cells over. 1 is the sequential path, which is the
    /// same code with one shard.
    pub(super) jobs: u32,
    /// `(index, count)` when this process *is* a worker.
    pub(super) shard: Option<(u32, u32)>,
    pub(super) shard_out: Option<PathBuf>,
    /// Where to publish the run live (packet M7/E4). `None` publishes nothing and binds
    /// nothing.
    pub(super) telemetry: Option<std::net::SocketAddr>,
    pub(super) telemetry_token: Option<String>,
    /// Control ticks between two observation images on stream 4; `0` publishes none.
    pub(super) telemetry_image_every: u64,
}

pub(super) fn parse_run_args(args: &[String]) -> Result<RunArgs, CliError> {
    let (mut config, mut policy, mut scene, mut out) = (None, None, None, None);
    let (mut backend, mut runtime) = ("mujoco-cpu".to_owned(), "torch".to_owned());
    let (mut frames, mut jobs, mut shard, mut shard_out) = (None, 1u32, None, None);
    let (mut traj, mut expert) = (None, None);
    let (mut telemetry, mut telemetry_token, mut telemetry_image_every) = (None, None, 0u64);

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{RUN_HELP}")))
        };
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(RUN_HELP.to_owned())),
            "--config" => config = Some(val()?.clone()),
            "--policy" => policy = Some(val()?.clone()),
            "--scene" => scene = Some(val()?.clone()),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--frames" => frames = Some(PathBuf::from(val()?)),
            "--expert" => expert = Some(val()?.clone()),
            "--traj" => traj = Some(PathBuf::from(val()?)),
            "--backend" => backend.clone_from(val()?),
            "--runtime" => runtime.clone_from(val()?),
            "--jobs" => {
                let v = val()?;
                jobs = v.parse().map_err(|_| {
                    CliError::Usage(format!("--jobs {v:?} is not a number\n\n{RUN_HELP}"))
                })?;
            }
            "--shard" => shard = Some(parse_shard(val()?)?),
            "--shard-out" => shard_out = Some(PathBuf::from(val()?)),
            "--telemetry" => {
                let v = val()?;
                telemetry = Some(v.parse().map_err(|e| {
                    CliError::Usage(format!(
                        "--telemetry {v:?} is not an address: {e}

{RUN_HELP}"
                    ))
                })?);
            }
            "--telemetry-token" => telemetry_token = Some(val()?.clone()),
            "--telemetry-image-every" => {
                let v = val()?;
                telemetry_image_every = v.parse().map_err(|_| {
                    CliError::Usage(format!(
                        "--telemetry-image-every {v:?} is not a number

{RUN_HELP}"
                    ))
                })?;
            }
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag '{other}'\n\n{RUN_HELP}"
                )))
            }
        }
    }
    // `--jobs 0` would be "run nothing and report on it", which is a lie with an exit code.
    if jobs == 0 {
        return Err(CliError::Usage(format!(
            "--jobs 0 runs no cell; the sequential run is --jobs 1\n\n{RUN_HELP}"
        )));
    }
    if shard.is_some() && jobs > 1 {
        return Err(CliError::Usage(format!(
            "--shard is worker mode and --jobs spawns workers; a worker is never a parent\n\n\
             {RUN_HELP}"
        )));
    }
    // A worker that wrote a report would write one covering a subset of the suites, under a
    // correct `evaluation_hash` (spec 10.4). The two flags travel together or not at all.
    if shard.is_some() != shard_out.is_some() {
        return Err(CliError::Usage(format!(
            "--shard and --shard-out go together: a worker's cells are not a report\n\n\
             {RUN_HELP}"
        )));
    }
    // One process, one listener: the cells of a `--jobs N` run happen in N child processes,
    // and a worker that inherited the address would fight the parent for the port.
    if telemetry.is_some() && (jobs > 1 || shard.is_some()) {
        return Err(CliError::Usage(format!(
            "--telemetry publishes one process's run; use --jobs 1 (a --jobs N run's cells              happen in worker processes, which cannot share the address)

{RUN_HELP}"
        )));
    }
    let req = |v: Option<String>, name: &str| {
        v.ok_or_else(|| CliError::Usage(format!("{name} is required\n\n{RUN_HELP}")))
    };
    Ok(RunArgs {
        config: req(config, "--config")?,
        policy: req(policy, "--policy")?,
        scene: req(scene, "--scene")?,
        out: out.unwrap_or_else(|| PathBuf::from("eval-out")),
        frames,
        expert,
        traj,
        backend,
        runtime,
        jobs,
        shard,
        shard_out,
        telemetry,
        telemetry_token,
        telemetry_image_every,
    })
}

/// `i/N`, with the range checked here so a worker cannot be asked for a shard that is not part
/// of the partition.
fn parse_shard(s: &str) -> Result<(u32, u32), CliError> {
    let bad = || CliError::Usage(format!("--shard {s:?} is not `i/N`\n\n{RUN_HELP}"));
    let (i, n) = s.split_once('/').ok_or_else(bad)?;
    let (i, n): (u32, u32) = (i.parse().map_err(|_| bad())?, n.parse().map_err(|_| bad())?);
    if n == 0 || i >= n {
        return Err(CliError::Usage(format!(
            "--shard {i}/{n}: the count must be at least 1 and the index below it\n\n{RUN_HELP}"
        )));
    }
    Ok((i, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Vec<String> {
        let mut v = vec![
            "--config".to_owned(),
            "e.toml".to_owned(),
            "--policy".to_owned(),
            "p.esb".to_owned(),
            "--scene".to_owned(),
            "s.xml".to_owned(),
        ];
        v.extend(extra.iter().map(|s| (*s).to_owned()));
        v
    }

    fn usage(extra: &[&str]) -> String {
        match parse_run_args(&args(extra)) {
            Err(CliError::Usage(text)) => text,
            other => panic!("{extra:?} was not refused: {other:?}"),
        }
    }

    /// The default is the sequential run, and it is the same partition the sharded path uses.
    #[test]
    fn jobs_defaults_to_one_and_names_no_shard() {
        let a = parse_run_args(&args(&[])).expect("the bare form parses");
        assert_eq!(a.jobs, 1);
        assert_eq!(a.shard, None);
        assert_eq!(a.shard_out, None);
    }

    /// `--jobs 0` is "run no cell and report on it". Refused before anything is opened.
    #[test]
    fn jobs_zero_is_refused() {
        assert!(usage(&["--jobs", "0"]).contains("--jobs 0 runs no cell"));
    }

    /// A worker's cells are not a report (spec 10.4): the two flags travel together, and a
    /// worker is never also a parent.
    #[test]
    fn a_shard_needs_a_shard_out_and_refuses_to_be_a_parent() {
        assert!(usage(&["--shard", "0/4"]).contains("go together"));
        assert!(usage(&["--shard-out", "s.json"]).contains("go together"));
        assert!(
            usage(&["--shard", "0/4", "--shard-out", "s.json", "--jobs", "2"])
                .contains("never a parent")
        );
    }

    /// The partition is checked where it is parsed, so no worker is ever asked for a shard
    /// outside it.
    #[test]
    fn a_shard_outside_the_partition_is_refused() {
        assert!(usage(&["--shard", "4/4", "--shard-out", "s.json"]).contains("index below it"));
        assert!(usage(&["--shard", "0/0", "--shard-out", "s.json"]).contains("index below it"));
        assert!(usage(&["--shard", "0-4", "--shard-out", "s.json"]).contains("is not `i/N`"));
    }
}
