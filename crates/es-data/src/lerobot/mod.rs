//! `LeRobot` dataset reading and writing (spec 19.1). **Rust-native, no Python** (spec 2.5).
//!
//! The format itself — and how much of it is guessed rather than verified — is documented in
//! `docs/api-notes/lerobot-dataset.md`. Nothing here decodes video: an image feature surfaces
//! as a [`VideoRef`] naming the mp4 and the frame within it.

mod columns;
pub mod meta;
pub mod v3;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use columns::Column;
pub use meta::{Dtype, EpisodeMeta, FeatureSpec, Info, Task};
pub use v3::{export_v3, ExportReport};

use crate::{read_file, write_file, DataError};

/// Where one frame of an image feature lives. Decoding is a later packet.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VideoRef {
    /// Relative to the dataset root.
    pub path: PathBuf,
    /// Position within the episode.
    pub frame_index: u32,
}

/// One episode, columnar.
///
/// `frame_index`, `episode_index` and the dataset-global `index` are pure functions of
/// position and so are not stored: the writer derives them and the reader drops them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Episode {
    pub index: u32,
    /// Task strings, from `meta/episodes.jsonl`.
    pub tasks: Vec<String>,
    pub timestamps: Vec<f64>,
    /// Per-frame task label (spec 13.2 allows it to vary within an episode).
    pub task_index: Vec<i64>,
    /// Feature name -> flat row-major values, `len() * elem_count` long.
    pub columns: BTreeMap<String, Column>,
    /// Camera key -> one ref per frame.
    pub video: BTreeMap<String, Vec<VideoRef>>,
}

impl Episode {
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

fn meta_dir(root: &Path) -> PathBuf {
    root.join("meta")
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, DataError> {
    let bytes = read_file(path)?;
    serde_json::from_slice(&bytes).map_err(|source| DataError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>, DataError> {
    let bytes = read_file(path)?;
    let text = String::from_utf8(bytes).map_err(|e| DataError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
    })?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).map_err(|source| DataError::Json {
                path: path.to_path_buf(),
                source,
            })
        })
        .collect()
}

fn write_jsonl<T: serde::Serialize>(path: &Path, rows: &[T]) -> Result<(), DataError> {
    let mut out = String::new();
    for row in rows {
        out.push_str(
            &serde_json::to_string(row).map_err(|source| DataError::Json {
                path: path.to_path_buf(),
                source,
            })?,
        );
        out.push('\n');
    }
    write_file(path, out.as_bytes())
}

/// A dataset on disk, opened read-only.
#[derive(Clone, Debug)]
pub struct LeRobotDataset {
    root: PathBuf,
    info: Info,
    episodes: Vec<EpisodeMeta>,
    tasks: Vec<Task>,
}

impl LeRobotDataset {
    /// Parses `meta/info.json`, `meta/episodes.jsonl` and `meta/tasks.jsonl`. No sample data
    /// is read; [`Self::read_episode`] does that one episode at a time.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, DataError> {
        let root = root.into();
        let meta = meta_dir(&root);
        let info: Info = read_json(&meta.join("info.json"))?;
        let mut episodes: Vec<EpisodeMeta> = read_jsonl(&meta.join("episodes.jsonl"))?;
        episodes.sort_by_key(|e| e.episode_index);
        let tasks_path = meta.join("tasks.jsonl");
        let tasks = if tasks_path.exists() {
            read_jsonl(&tasks_path)?
        } else {
            Vec::new()
        };
        Ok(Self {
            root,
            info,
            episodes,
            tasks,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn features(&self) -> &BTreeMap<String, FeatureSpec> {
        &self.info.features
    }

    /// Sorted by `episode_index`.
    pub fn episodes(&self) -> &[EpisodeMeta] {
        &self.episodes
    }

    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    fn meta_of(&self, index: u32) -> Result<&EpisodeMeta, DataError> {
        self.episodes
            .iter()
            .find(|e| e.episode_index == index)
            .ok_or_else(|| DataError::Inconsistent(format!("no episode {index}")))
    }

    /// Absolute path of an episode's parquet file.
    pub fn episode_path(&self, index: u32) -> Result<PathBuf, DataError> {
        Ok(self.root.join(self.info.data_path(index)?))
    }

    /// Reads one episode's parquet file. No video is decoded.
    pub fn read_episode(&self, index: u32) -> Result<Episode, DataError> {
        let meta = self.meta_of(index)?;
        let path = self.episode_path(index)?;
        let (cols, timestamps, task_index) =
            columns::read_episode(&path, &self.info, index, meta.length)?;
        Ok(Episode {
            index,
            tasks: meta.tasks.clone(),
            timestamps,
            task_index,
            columns: cols,
            video: columns::video_refs(&self.info, index, meta.length as usize)?,
        })
    }
}

/// Writes the layout [`LeRobotDataset::open`] reads back.
///
/// Totals in the `info` handed to [`Self::create`] are placeholders; [`Self::finish`]
/// recomputes them. No mp4 is written — video refs on an [`Episode`] are ignored and
/// regenerated from `video_path` on read.
#[derive(Debug)]
pub struct LeRobotWriter {
    root: PathBuf,
    info: Info,
    episodes: Vec<EpisodeMeta>,
    tasks: Vec<Task>,
    seen: BTreeMap<String, u32>,
    frames: u64,
}

impl LeRobotWriter {
    pub fn create(root: impl Into<PathBuf>, info: Info) -> Result<Self, DataError> {
        let root = root.into();
        let meta = meta_dir(&root);
        std::fs::create_dir_all(&meta).map_err(|e| DataError::io(&meta, e))?;
        Ok(Self {
            root,
            info,
            episodes: Vec::new(),
            tasks: Vec::new(),
            seen: BTreeMap::new(),
            frames: 0,
        })
    }

    /// Writes one episode's parquet file and records its metadata.
    ///
    /// The caller owns the correspondence between `episode.task_index` and `episode.tasks`;
    /// task strings are assigned indices in first-seen order and the column is written
    /// verbatim.
    pub fn write_episode(&mut self, episode: &Episode) -> Result<(), DataError> {
        let n = episode.len();
        if episode.task_index.len() != n {
            return Err(DataError::Inconsistent(format!(
                "episode {}: {} task indices for {n} frames",
                episode.index,
                episode.task_index.len()
            )));
        }
        let path = self.root.join(self.info.data_path(episode.index)?);
        columns::write_episode(&path, &self.info, episode, self.frames)?;
        for task in &episode.tasks {
            let next = u32::try_from(self.seen.len()).unwrap_or(u32::MAX);
            if self.seen.insert(task.clone(), next).is_none() {
                self.tasks.push(Task {
                    task_index: next,
                    task: task.clone(),
                });
            }
        }
        self.episodes.push(EpisodeMeta {
            episode_index: episode.index,
            tasks: episode.tasks.clone(),
            length: n as u64,
        });
        self.frames += n as u64;
        Ok(())
    }

    /// Writes `meta/info.json`, `meta/episodes.jsonl` and `meta/tasks.jsonl` with totals
    /// recomputed from what was actually written.
    pub fn finish(mut self) -> Result<(), DataError> {
        self.episodes.sort_by_key(|e| e.episode_index);
        self.info.total_episodes = u32::try_from(self.episodes.len()).unwrap_or(u32::MAX);
        self.info.total_frames = self.frames;
        self.info.total_tasks = u32::try_from(self.tasks.len()).unwrap_or(u32::MAX);
        self.info.total_chunks = self
            .episodes
            .iter()
            .map(|e| self.info.chunk_of(e.episode_index) + 1)
            .max()
            .unwrap_or(0);
        let meta = meta_dir(&self.root);
        let path = meta.join("info.json");
        let json = serde_json::to_vec_pretty(&self.info).map_err(|source| DataError::Json {
            path: path.clone(),
            source,
        })?;
        write_file(&path, &json)?;
        write_jsonl(&meta.join("episodes.jsonl"), &self.episodes)?;
        write_jsonl(&meta.join("tasks.jsonl"), &self.tasks)
    }
}
