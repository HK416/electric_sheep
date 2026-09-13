//! `es-data` (layer 10): dataset I/O and dataset identity. See `docs/ARCHITECTURE.ko.md`
//! spec 19 (format, Dataset Identity, Training Identity), spec 5.3 (`dataset_hash`) and the
//! work packet `docs/packets/M1/W7-lerobot-dataset.md`.
//!
//! `LeRobot` compatibility is the first-class interop target (spec 19.1) and **reading is
//! Rust-native** (spec 2.5): nothing here runs Python, at build time or at run time. The
//! on-disk format this crate assumes is described — and its uncertainty marked — in
//! `docs/api-notes/lerobot-dataset.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-core`, `es-ir` and external crates
//! only.
#![forbid(unsafe_code)]

pub mod identity;
pub mod lerobot;
pub mod lerobot_config;

use std::path::{Path, PathBuf};

pub use identity::{BaseModel, DatasetIdentity, Split, TrainingIdentity};
pub use lerobot::{
    Column, Dtype, Episode, EpisodeMeta, FeatureSpec, Info, LeRobotDataset, LeRobotWriter, Task,
    VideoRef,
};

/// Everything that can go wrong reading or writing a dataset.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: malformed JSON: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}: {source}")]
    Parquet {
        path: PathBuf,
        #[source]
        source: parquet::errors::ParquetError,
    },
    /// The file parses but says something this crate does not implement (an unsupported
    /// `dtype`, a compression codec whose `parquet` feature is off, a path template with an
    /// unknown placeholder). See `docs/api-notes/lerobot-dataset.md`.
    #[error("unsupported LeRobot dataset: {0}")]
    Unsupported(String),
    /// The file parses but contradicts itself (a column length that disagrees with the
    /// episode length, a ragged list, an episode index with no metadata entry).
    #[error("inconsistent LeRobot dataset: {0}")]
    Inconsistent(String),
    /// Canonical encoding refused a value (spec 11.2: `NaN` has no canonical form).
    #[error("canonical encoding failed: {0}")]
    Canon(es_ir::Diagnostic),
}

impl DataError {
    fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// `std::fs::read` with the path in the error.
fn read_file(path: &Path) -> Result<Vec<u8>, DataError> {
    std::fs::read(path).map_err(|e| DataError::io(path, e))
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), DataError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| DataError::io(dir, e))?;
    }
    std::fs::write(path, bytes).map_err(|e| DataError::io(path, e))
}
