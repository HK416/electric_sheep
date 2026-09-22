//! `es gap` -- domain-gap diagnostics between a sim and a real `LeRobot` dataset (spec 24.3).
//!
//! This is the glue `docs/design/domain-gap.md` §1 describes: `es_eval::domain_gap` cannot
//! depend on `es_data` (both are spec-4.2 layer 10), so this command is the only place that
//! turns a `LeRobotDataset` into the crate-agnostic `GapInput` the compute function takes.

use std::collections::BTreeMap;

use es_data::{Column, LeRobotDataset};
use es_eval::domain_gap::{DomainGap, EpisodeSummary, FeatureSamples, GapInput, GapOptions};

use crate::error::CliError;

const HELP: &str = "\
es gap --sim <root> --real <root> [OPTIONS]

Opens two LeRobot datasets (spec 19.1) -- one from sim rollouts, one from a real robot --
and reports the distribution gap between them per observation/action channel (KS statistic,
Wasserstein-1, mean/std, quantiles), plus the episode-level success/length/envelope-
violation gap (spec 10.3, spec 24.3). Design: docs/design/domain-gap.md.

    --out <file>          where to write gap_report.json (default: ./gap_report.json)
    --max-samples N       per-channel subsampling cap (default: 4096, deterministic stride)
    --threshold D         KS D above which a channel is flagged (default: 0.3)

Exit code: 0 when no channel is flagged; 1 when at least one is (also printed as
`suspects`); 2 on a usage error.
";

fn parse<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, CliError> {
    raw.parse()
        .map_err(|_| CliError::Usage(format!("{flag}: invalid value '{raw}'\n\n{HELP}")))
}

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }

    let mut sim = None;
    let mut real = None;
    let mut out = "gap_report.json".to_owned();
    let mut max_samples = 4096usize;
    let mut threshold = 0.3f64;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{HELP}")))
        };
        match a.as_str() {
            "--sim" => sim = Some(val()?.clone()),
            "--real" => real = Some(val()?.clone()),
            "--out" => out.clone_from(val()?),
            "--max-samples" => max_samples = parse("--max-samples", val()?)?,
            "--threshold" => threshold = parse("--threshold", val()?)?,
            other => return Err(CliError::Usage(format!("unknown flag '{other}'\n\n{HELP}"))),
        }
    }
    let (Some(sim), Some(real)) = (sim, real) else {
        return Err(CliError::Usage(format!(
            "--sim and --real are both required\n\n{HELP}"
        )));
    };

    let sim_ds =
        LeRobotDataset::open(&sim).map_err(|e| CliError::Runtime(format!("{sim}: {e}")))?;
    let real_ds =
        LeRobotDataset::open(&real).map_err(|e| CliError::Runtime(format!("{real}: {e}")))?;
    let sim_input =
        build_input(&sim_ds, max_samples).map_err(|e| CliError::Runtime(e.to_string()))?;
    let real_input =
        build_input(&real_ds, max_samples).map_err(|e| CliError::Runtime(e.to_string()))?;

    let opts = GapOptions {
        max_samples_per_feature: max_samples,
        threshold,
    };
    let report = DomainGap::compute(&sim_input, &real_input, &opts)
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    print!("{report}");
    let json = report
        .to_json()
        .map_err(|e| CliError::Runtime(format!("{out}: {e}")))?;
    std::fs::write(&out, json).map_err(|e| CliError::Runtime(format!("{out}: {e}")))?;
    println!("\nwrote {out}");

    Ok(u8::from(report.has_flagged()))
}

/// Episode-level column names that feed [`EpisodeSummary`] rather than the per-channel
/// comparison (design doc §1's episode-level block).
const SUCCESS_COLUMN: &str = "success";
const ACTION_SOURCE_COLUMN: &str = "action_source";

fn column_f64(col: &Column) -> Vec<f64> {
    match col {
        Column::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        Column::F64(v) => v.clone(),
        Column::I64(v) => v.iter().map(|&x| x as f64).collect(),
        Column::Bool(v) => v.iter().map(|&b| f64::from(u8::from(b))).collect(),
    }
}

/// Reads every episode of `dataset` and builds the [`GapInput`] `DomainGap::compute` takes
/// (design doc §5): every numeric, non-reserved feature becomes a channel, deterministically
/// subsampled by a fixed stride computed from the dataset's total frame count; a `success`
/// column (episode's last frame, truthy) and an `action_source` column (fraction of an
/// episode's frames with a nonzero value) feed the episode-level block instead.
fn build_input(
    dataset: &LeRobotDataset,
    max_samples: usize,
) -> Result<GapInput, es_data::DataError> {
    let total_frames: u64 = dataset.episodes().iter().map(|e| e.length).sum();
    let stride = if total_frames == 0 {
        1
    } else {
        ((total_frames as f64) / (max_samples.max(1) as f64))
            .ceil()
            .max(1.0) as u64
    };

    let mut channels: BTreeMap<String, FeatureSamples> = BTreeMap::new();
    for (name, feat) in dataset.info().columnar() {
        if name == SUCCESS_COLUMN || name == ACTION_SOURCE_COLUMN {
            continue;
        }
        channels.insert(
            name.clone(),
            FeatureSamples::new(feat.elem_count() as usize, Vec::new()),
        );
    }

    let mut episodes = Vec::with_capacity(dataset.episodes().len());
    let mut offset: u64 = 0;
    for meta in dataset.episodes() {
        let episode = dataset.read_episode(meta.episode_index)?;
        let n = episode.len();

        let success = episode
            .columns
            .get(SUCCESS_COLUMN)
            .map(|c| column_f64(c).last().copied().unwrap_or(0.0) != 0.0);
        let envelope_violation = episode.columns.get(ACTION_SOURCE_COLUMN).map(|c| {
            let v = column_f64(c);
            if v.is_empty() {
                0.0
            } else {
                v.iter().filter(|&&x| x != 0.0).count() as f64 / v.len() as f64
            }
        });
        episodes.push(EpisodeSummary {
            success,
            length: n as u64,
            envelope_violation,
        });

        for (name, samples) in &mut channels {
            let Some(col) = episode.columns.get(name) else {
                continue;
            };
            let flat = column_f64(col);
            let dims = samples.dims.max(1);
            for local in 0..n {
                if (offset + local as u64) % stride != 0 {
                    continue;
                }
                // A foreign dataset whose column is shorter than `n * dims` is inconsistent,
                // not a panic: the declared `elem_count` is what `dims` came from.
                let row = flat.get(local * dims..(local + 1) * dims).ok_or_else(|| {
                    es_data::DataError::Inconsistent(format!(
                        "episode {}: column \"{name}\" holds {} values, {} short of the {} \
                         frames x {dims} dims its feature declares",
                        meta.episode_index,
                        flat.len(),
                        n * dims - flat.len().min(n * dims),
                        n
                    ))
                })?;
                // `serde_json` has no encoding for NaN/Inf and neither KS nor Wasserstein-1
                // has a meaning over them, so the frame is dropped and counted (design §5).
                if row.iter().all(|v| v.is_finite()) {
                    samples.values.extend_from_slice(row);
                } else {
                    samples.nonfinite_dropped += 1;
                }
            }
        }
        offset += n as u64;
    }

    Ok(GapInput { channels, episodes })
}
