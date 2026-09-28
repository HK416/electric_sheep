//! `episodes.json`: one row per evaluated episode, beside `report.json` (design note
//! `docs/design/editor-redesign.md` 6.6).
//!
//! A separate file and not a `report.json` field, so that no committed report moves. Nothing
//! here computes a metric: a row is one [`ShardCell`]'s own numbers, read back out of the units
//! [`crate::Evaluation::merge`] already accepted.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use es_ir::evaluation::EvaluationIr;
use serde::{Deserialize, Serialize};

use crate::runner::{cell_name, resolve_seeds, Shard, ShardCell};
use crate::EvalError;

pub const EPISODES_FILE: &str = "episodes.json";

/// The histogram buckets that say how an episode ended; every episode has exactly one
/// (`metrics::failure_histogram`).
const TERMINATIONS: [&str; 4] = ["success", "failure", "timeout", "unfinished"];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeRow {
    /// `EvaluationIr::suites[cell].name`.
    pub suite: String,
    /// `<suite>-<NN>`: the key of `frames/<cell>/`, `traj/<cell>.estraj`, `events.json`.
    pub cell: String,
    /// Index into the resolved seed list (§10.2).
    pub episode: u64,
    pub seed: u64,
    /// `success` | `failure` | `timeout` | `unfinished` — the one termination bucket.
    pub termination: String,
    /// Steps the Safety Plane validated, and how many of them it changed.
    pub steps: u64,
    pub changed_steps: u64,
    /// This episode's own §10.3 `failure_mode_histogram` buckets.
    pub histogram: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeIndex {
    pub schema_version: u32,
    pub episodes: Vec<EpisodeRow>,
}

/// Rows in `(cell, episode)` order, whatever order the shards arrive in.
///
/// `shards` are units of `ir` — the ones [`crate::Evaluation::merge`] accepted, which refuses a
/// unit outside `suites x seeds`.
pub fn episode_rows(ir: &EvaluationIr, shards: &[Shard]) -> Vec<EpisodeRow> {
    let seeds = resolve_seeds(ir);
    let mut units: Vec<&ShardCell> = shards.iter().flat_map(|s| &s.cells).collect();
    units.sort_by_key(|u| (u.cell, u.episode));
    units
        .into_iter()
        .map(|u| {
            let suite = &ir.suites[u.cell as usize].name;
            let h = &u.summary.histogram;
            EpisodeRow {
                suite: suite.clone(),
                cell: cell_name(suite, u.episode),
                episode: u.episode,
                seed: seeds[u.episode as usize],
                // Empty only for a summary `CellSummary::episode` did not build; never guessed.
                termination: TERMINATIONS
                    .into_iter()
                    .find(|t| h.contains_key(*t))
                    .unwrap_or_default()
                    .to_owned(),
                steps: u.summary.steps,
                changed_steps: u.summary.dirty_steps,
                histogram: h.clone(),
            }
        })
        .collect()
}

fn io_err(path: &Path) -> impl Fn(std::io::Error) -> EvalError + '_ {
    move |source| EvalError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// `<dir>/episodes.json`, pretty JSON with a trailing newline like `write_artifacts`.
pub fn write_episodes(rows: &[EpisodeRow], dir: &Path) -> Result<(), EvalError> {
    let path = dir.join(EPISODES_FILE);
    let index = EpisodeIndex {
        schema_version: 1,
        episodes: rows.to_vec(),
    };
    let mut text = serde_json::to_string_pretty(&index).map_err(|e| io_err(&path)(e.into()))?;
    text.push('\n');
    fs::write(&path, text).map_err(io_err(&path))
}

/// `Ok(None)` when the file is absent (a run from before this packet).
pub fn read_episodes(dir: &Path) -> Result<Option<Vec<EpisodeRow>>, EvalError> {
    let path = dir.join(EPISODES_FILE);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_err(&path)(e)),
    };
    let index: EpisodeIndex = serde_json::from_str(&text).map_err(|e| io_err(&path)(e.into()))?;
    Ok(Some(index.episodes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics;
    use es_ir::evaluation::{
        AugmentationPolicy, EpisodeBatch, PerturbationSuite, ReplayPolicy, SeedPlan,
    };

    fn two_suite_ir() -> EvaluationIr {
        let suite = |name: &str| PerturbationSuite {
            name: name.to_owned(),
            perturbations: Vec::new(),
        };
        EvaluationIr {
            schema_version: 1,
            task: "fixture.task".to_owned(),
            observation: "fixture.obs".to_owned(),
            episodes: EpisodeBatch {
                n_episodes: 2,
                seeds: SeedPlan::Base(101),
            },
            suites: vec![suite("nominal"), suite("light_intensity")],
            metrics: Vec::new(),
            acceptance: Vec::new(),
            augmentation: AugmentationPolicy::Disabled,
            replay: ReplayPolicy::default(),
        }
    }

    fn summary(termination: &str, steps: u64, changed: u64) -> metrics::CellSummary {
        let mut s = metrics::CellSummary::default();
        s.histogram.insert(termination.to_owned(), 1);
        s.steps = steps;
        s.dirty_steps = changed;
        s.n_episodes = 1;
        s
    }

    fn unit(cell: u32, episode: u64, termination: &str) -> ShardCell {
        ShardCell {
            cell,
            episode,
            summary: summary(termination, 100, 3),
        }
    }

    #[test]
    fn rows_come_out_in_cell_episode_order_with_names_and_seeds() {
        let ir = two_suite_ir(); // suites "nominal", "light_intensity"; seeds base 101, 2 episodes
        let a = Shard {
            cells: vec![unit(1, 0, "failure"), unit(0, 1, "success")],
            ..Default::default()
        };
        let b = Shard {
            cells: vec![unit(0, 0, "timeout"), unit(1, 1, "success")],
            ..Default::default()
        };
        let rows = episode_rows(&ir, &[a, b]);
        let key: Vec<_> = rows
            .iter()
            .map(|r| (r.cell.as_str(), r.seed, r.termination.as_str()))
            .collect();
        assert_eq!(
            key,
            [
                ("nominal-00", 101, "timeout"),
                ("nominal-01", 102, "success"),
                ("light_intensity-00", 101, "failure"),
                ("light_intensity-01", 102, "success"),
            ]
        );
        assert_eq!(rows[0].changed_steps, 3);
    }

    #[test]
    fn the_file_round_trips_and_an_absent_file_is_none() {
        let dir = std::env::temp_dir().join(format!("es-episodes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(read_episodes(&dir).unwrap(), None);
        let rows = episode_rows(
            &two_suite_ir(),
            &[Shard {
                cells: vec![unit(0, 0, "success")],
                ..Default::default()
            }],
        );
        write_episodes(&rows, &dir).unwrap();
        assert_eq!(read_episodes(&dir).unwrap(), Some(rows));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
