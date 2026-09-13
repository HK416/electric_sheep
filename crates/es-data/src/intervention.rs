//! Intervention labels (spec 13.2) — the segment record, the two per-frame columns, and
//! [`label`], which writes both onto a dataset that is already on disk.
//!
//! `docs/design/learning-loop.md` section 2 is the schema this implements and section 2.3 is
//! the rule that shapes it: `label` owns `intervention`, never `action_source`. The latter is
//! collection-time provenance — what the Safety Plane actually emitted while the robot was
//! moving — and a label applied afterwards cannot know it.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::collect::hex;
use crate::identity::{DatasetIdentity, Split};
use crate::lerobot::meta::{Dtype, FeatureSpec, Info};
use crate::lerobot::{Column, LeRobotDataset, LeRobotWriter};
use crate::{read_file, write_file, DataError};

/// Per-frame column: `1` when the frame lies inside an intervention segment.
pub const INTERVENTION: &str = "intervention";
/// Per-frame column: an [`ActionSourceCode`].
pub const ACTION_SOURCE: &str = "action_source";
/// Where the segment records live, relative to the dataset root.
pub const SEGMENTS_FILE: &str = "meta/interventions.jsonl";

/// Who produced the action (spec 13.2 `source`).
///
/// `Scripted` is a machine intervener — a classical controller, a replayed demonstration, a
/// test hook; `Teleop` is a human at a device; `Corrective` is a human correcting a running
/// policy, the HIL-SERL case spec 13.2 names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionSource {
    Teleop,
    Scripted,
    Corrective,
}

/// One contiguous run of intervened frames within one episode (spec 13.2).
///
/// `end_frame` is inclusive. Segments are never merged or normalized: two overlapping teleop
/// segments from two operators stay two records, because collapsing them would destroy the
/// attribution that is the reason the file exists.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct InterventionSegment {
    pub episode: u32,
    pub start_frame: u32,
    /// Inclusive.
    pub end_frame: u32,
    pub source: InterventionSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator_id: Option<String>,
    /// Spec 13.2's `reason`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl InterventionSegment {
    pub fn new(episode: u32, start_frame: u32, end_frame: u32, source: InterventionSource) -> Self {
        Self {
            episode,
            start_frame,
            end_frame,
            source,
            operator_id: None,
            note: None,
        }
    }

    pub fn frames(&self) -> u64 {
        u64::from(self.end_frame.saturating_sub(self.start_frame)) + 1
    }
}

/// The `action_source` column's values.
///
/// A clamp or a fallback is normal operation, not a failure and not a human (spec 18.5), so
/// these are four distinct outcomes rather than a boolean. A human action the plane then
/// clamped is `Clamped` here *and* `1` in the `intervention` column: the frame was an
/// intervention, and the action on the actuator was not the one the human asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSourceCode {
    Policy,
    Clamped,
    Fallback,
    Human,
}

impl ActionSourceCode {
    pub const ALL: [Self; 4] = [Self::Policy, Self::Clamped, Self::Fallback, Self::Human];

    /// The stored value. `int64` rather than `u8` because the four column dtypes this crate
    /// writes are `float32 / float64 / int64 / bool`; see `docs/api-notes/lerobot-dataset.md`.
    pub fn as_i64(self) -> i64 {
        match self {
            Self::Policy => 0,
            Self::Clamped => 1,
            Self::Fallback => 2,
            Self::Human => 3,
        }
    }

    pub fn from_i64(v: i64) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.as_i64() == v)
    }
}

/// What [`label`] did, and what it did to the dataset's identity (spec 19.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelReport {
    /// Episodes that carry at least one labelled frame afterwards, ascending.
    pub episodes: Vec<u32>,
    /// Frames labelled `1` afterwards.
    pub frames: u64,
    /// True when a column had to be added to `meta/info.json`, which is the only case in
    /// which `dataset_schema_hash` moves.
    pub schema_changed: bool,
    pub content_before: [u8; 32],
    pub content_after: [u8; 32],
    pub schema: [u8; 32],
}

/// The dataset's content hash, over a placeholder split (the split is not this module's
/// business — spec 19.2 keeps the three components independent).
pub(crate) fn content_of(dataset: &LeRobotDataset) -> Result<[u8; 32], DataError> {
    Ok(DatasetIdentity::compute(dataset, &Split::default())?.content)
}

pub(crate) fn identity_of(dataset: &LeRobotDataset) -> Result<([u8; 32], [u8; 32]), DataError> {
    let id = DatasetIdentity::compute(dataset, &Split::default())?;
    Ok((id.content, id.schema))
}

/// Reads `meta/interventions.jsonl`; an absent file is an empty list, not an error.
pub fn read_segments(root: &Path) -> Result<Vec<InterventionSegment>, DataError> {
    let path = root.join(SEGMENTS_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = read_file(&path)?;
    let text = String::from_utf8(bytes).map_err(|e| DataError::Io {
        path: path.clone(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
    })?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).map_err(|source| DataError::Json {
                path: path.clone(),
                source,
            })
        })
        .collect()
}

/// Writes `meta/interventions.jsonl`, sorted and deduplicated so the file is a function of
/// the segment set rather than of the order they were supplied in.
pub fn write_segments(root: &Path, segments: &[InterventionSegment]) -> Result<(), DataError> {
    let path = root.join(SEGMENTS_FILE);
    let mut sorted = segments.to_vec();
    sorted.sort();
    sorted.dedup();
    let mut out = String::new();
    for s in &sorted {
        out.push_str(&serde_json::to_string(s).map_err(|source| DataError::Json {
            path: path.clone(),
            source,
        })?);
        out.push('\n');
    }
    write_file(&path, out.as_bytes())
}

/// Adds the two loop columns to `info.features` if they are missing; `true` when it changed
/// anything, which is the only case in which `dataset_schema_hash` moves.
pub(crate) fn ensure_columns(info: &mut Info) -> bool {
    let mut changed = false;
    for name in [INTERVENTION, ACTION_SOURCE] {
        if !info.features.contains_key(name) {
            info.features
                .insert(name.to_owned(), FeatureSpec::new(Dtype::Int64, [1]));
            changed = true;
        }
    }
    changed
}

/// Contiguous runs of `true` in `mask`, as segments of `episode`.
pub(crate) fn segments_of(
    episode: u32,
    mask: &[bool],
    source: InterventionSource,
) -> Vec<InterventionSegment> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for i in 0..=mask.len() {
        let on = i < mask.len() && mask[i];
        match (start, on) {
            (None, true) => start = Some(i),
            (Some(s), false) => {
                out.push(InterventionSegment::new(
                    episode,
                    s as u32,
                    (i - 1) as u32,
                    source,
                ));
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// Labels the intervened frames of a dataset in place (spec 13.2).
///
/// `segments` is *merged* with whatever `meta/interventions.jsonl` already holds, and the
/// per-frame `intervention` column is then rebuilt from the merged set — so a second call
/// adds labels rather than replacing them, and an episode nobody has ever labelled ends up
/// with an explicit column of zeros rather than a missing one.
///
/// Episodes are rewritten one at a time (read, mutate the one column, write) so a dataset
/// larger than memory still labels. `action_source` is carried through untouched, and only
/// defaulted to `Policy` for a dataset that predates the column (design note section 2.3).
pub fn label(root: &Path, segments: &[InterventionSegment]) -> Result<LabelReport, DataError> {
    let dataset = LeRobotDataset::open(root)?;
    let content_before = content_of(&dataset)?;

    let mut all = read_segments(root)?;
    all.extend_from_slice(segments);
    all.sort();
    all.dedup();

    let mut masks: BTreeMap<u32, Vec<bool>> = BTreeMap::new();
    for s in &all {
        let meta = dataset
            .episodes()
            .iter()
            .find(|e| e.episode_index == s.episode)
            .ok_or_else(|| {
                DataError::Inconsistent(format!(
                    "intervention segment names episode {}, which this dataset does not have",
                    s.episode
                ))
            })?;
        let n = meta.length as usize;
        if s.start_frame > s.end_frame {
            return Err(DataError::Inconsistent(format!(
                "episode {}: intervention segment {}..={} runs backwards",
                s.episode, s.start_frame, s.end_frame
            )));
        }
        if s.end_frame as usize >= n {
            return Err(DataError::Inconsistent(format!(
                "episode {}: intervention segment ends at frame {}, the episode has {n}",
                s.episode, s.end_frame
            )));
        }
        let mask = masks
            .entry(s.episode)
            .or_insert_with(|| vec![false; n.max(1)]);
        for f in s.start_frame..=s.end_frame {
            mask[f as usize] = true;
        }
    }

    let mut info = dataset.info().clone();
    let schema_changed = ensure_columns(&mut info);

    let mut writer = LeRobotWriter::create(root, info)?;
    let mut frames = 0u64;
    let mut labelled = Vec::new();
    for meta in dataset.episodes() {
        let index = meta.episode_index;
        let mut ep = dataset.read_episode(index)?;
        let n = ep.len();
        let meta_len = meta.length as usize;
        if meta_len != n {
            return Err(DataError::Inconsistent(format!(
                "episode {index}: meta.length {meta_len} disagrees with parquet length {n}"
            )));
        }
        let values: Vec<i64> = (0..n)
            .map(|i| i64::from(masks.get(&index).is_some_and(|m| m[i])))
            .collect();
        let on = values.iter().filter(|v| **v != 0).count() as u64;
        if on > 0 {
            labelled.push(index);
            frames += on;
        }
        ep.columns
            .insert(INTERVENTION.to_owned(), Column::I64(values));
        ep.columns
            .entry(ACTION_SOURCE.to_owned())
            .or_insert_with(|| Column::I64(vec![ActionSourceCode::Policy.as_i64(); n]));
        writer.write_episode(&ep)?;
    }
    writer.finish()?;
    write_segments(root, &all)?;

    let after = LeRobotDataset::open(root)?;
    let (content_after, schema) = identity_of(&after)?;
    // The loop ledger (spec 13.3): this step's input content is the previous step's output.
    crate::collect::append_loop_step(
        root,
        &crate::collect::LoopStep::new(crate::collect::LoopKind::Intervene)
            .input("content", &hex(&content_before))
            .input("segments", &all.len())
            .output("content", &hex(&content_after))
            .output("schema", &hex(&schema)),
    )?;
    Ok(LabelReport {
        episodes: labelled,
        frames,
        schema_changed,
        content_before,
        content_after,
        schema,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_runs_become_segments() {
        let mask = [false, true, true, false, true];
        let got = segments_of(3, &mask, InterventionSource::Scripted);
        assert_eq!(
            got,
            vec![
                InterventionSegment::new(3, 1, 2, InterventionSource::Scripted),
                InterventionSegment::new(3, 4, 4, InterventionSource::Scripted),
            ]
        );
        assert_eq!(got[0].frames(), 2);
        assert!(segments_of(0, &[false, false], InterventionSource::Teleop).is_empty());
    }

    #[test]
    fn action_source_codes_round_trip() {
        for c in ActionSourceCode::ALL {
            assert_eq!(ActionSourceCode::from_i64(c.as_i64()), Some(c));
        }
        assert_eq!(ActionSourceCode::from_i64(4), None);
    }
}
