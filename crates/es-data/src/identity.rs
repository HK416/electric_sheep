//! Dataset Identity (spec 19.2) and Training Identity (spec 19.3).
//!
//! `dataset_hash = H(content, schema, split)` (spec 5.3). The split is in the identity
//! because the same bytes split differently are a different training run — so the split must
//! be a deterministic function, which [`Split::deterministic`] is.

use std::collections::BTreeMap;
use std::fs::File;

use es_ir::{CanonWriter, DatasetHash};
use serde::{Deserialize, Serialize};

use crate::lerobot::{FeatureSpec, LeRobotDataset};
use crate::DataError;

/// Domain separators: changing one invalidates every stored hash, on purpose.
const CONTENT_TAG: &str = "es.dataset.content.v1";
const SCHEMA_TAG: &str = "es.dataset.schema.v1";
const SPLIT_TAG: &str = "es.dataset.split.v1";
const TRAINING_TAG: &str = "es.training.v1";
const POLICY_TAG: &str = "es.policy_hash.v1";

fn canon(w: CanonWriter) -> Result<[u8; 32], DataError> {
    w.hash().map_err(DataError::Canon)
}

/// train / val / test as episode index lists (spec 19.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Split {
    pub train: Vec<u32>,
    pub val: Vec<u32>,
    pub test: Vec<u32>,
}

/// `splitmix64` — a two-line PRNG with no global state (spec 3.4 bans global RNG).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Split {
    /// A seeded Fisher-Yates shuffle of `0..n_episodes`, cut by `fractions`
    /// `[train, val, test]`. `train` and `val` take `floor(n * fraction)` episodes and `test`
    /// takes the remainder, so the three lists are always an exact partition. Each list is
    /// returned sorted.
    pub fn deterministic(n_episodes: u32, fractions: [f64; 3], seed: u64) -> Self {
        let n = n_episodes as usize;
        let mut order: Vec<u32> = (0..n_episodes).collect();
        let mut state = seed;
        for i in (1..n).rev() {
            let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let cut = |f: f64| -> usize {
            let k = (n as f64 * f.clamp(0.0, 1.0)).floor();
            (k as usize).min(n)
        };
        let train_n = cut(fractions[0]);
        let val_n = cut(fractions[1]).min(n - train_n);
        let mut split = Self {
            train: order[..train_n].to_vec(),
            val: order[train_n..train_n + val_n].to_vec(),
            test: order[train_n + val_n..].to_vec(),
        };
        for list in [&mut split.train, &mut split.val, &mut split.test] {
            list.sort_unstable();
        }
        split
    }

    pub fn hash(&self) -> Result<[u8; 32], DataError> {
        let mut w = CanonWriter::new();
        w.str(SPLIT_TAG);
        for list in [&self.train, &self.val, &self.test] {
            w.seq(list.len());
            for i in list {
                w.u32(*i);
            }
        }
        canon(w)
    }
}

/// `content + schema + split` (spec 19.2), the three components of
/// [`es_ir::DatasetHash`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatasetIdentity {
    pub content: [u8; 32],
    pub schema: [u8; 32],
    pub split: [u8; 32],
}

impl DatasetIdentity {
    pub fn compute(dataset: &LeRobotDataset, split: &Split) -> Result<Self, DataError> {
        Ok(Self {
            content: content_hash(dataset)?,
            schema: schema_hash(dataset)?,
            split: split.hash()?,
        })
    }

    pub fn to_dataset_hash(self) -> DatasetHash {
        DatasetHash {
            content: self.content,
            schema: self.schema,
            split: self.split,
        }
    }
}

/// blake3 over every episode's parquet bytes, in episode index order.
///
/// Streamed: one `update_reader` per file, so a dataset that does not fit in memory still
/// hashes. Each file is framed by its episode index and byte length, so no rearrangement of
/// bytes across files can collide.
fn content_hash(dataset: &LeRobotDataset) -> Result<[u8; 32], DataError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CONTENT_TAG.as_bytes());
    hasher.update(
        &u32::try_from(dataset.episodes().len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for meta in dataset.episodes() {
        let path = dataset.episode_path(meta.episode_index)?;
        let file = File::open(&path).map_err(|e| DataError::io(&path, e))?;
        let len = file.metadata().map_err(|e| DataError::io(&path, e))?.len();
        hasher.update(&meta.episode_index.to_le_bytes());
        hasher.update(&len.to_le_bytes());
        hasher
            .update_reader(file)
            .map_err(|e| DataError::io(&path, e))?;
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Field composition, action space and frame rate — everything that decides what a sample
/// *means*, and nothing that decides what a sample *is*.
fn schema_hash(dataset: &LeRobotDataset) -> Result<[u8; 32], DataError> {
    let info = dataset.info();
    let mut w = CanonWriter::new();
    w.str(SCHEMA_TAG);
    w.str(&info.codebase_version);
    w.f64(info.fps);
    encode_features(&mut w, &info.features);
    canon(w)
}

/// Sorted by name — `BTreeMap` is sorted by construction (spec 3.4).
fn encode_features(w: &mut CanonWriter, features: &BTreeMap<String, FeatureSpec>) {
    w.seq(features.len());
    for (name, feat) in features {
        w.str(name);
        w.str(&format!("{:?}", feat.dtype));
        w.seq(feat.shape.len());
        for d in &feat.shape {
            w.u64(*d);
        }
        match &feat.names {
            Some(names) => {
                w.seq(names.len());
                for n in names {
                    w.str(n);
                }
            }
            None => w.seq(0),
        }
    }
}

/// Provenance of a pre-trained backbone (spec 19.3 `base_model.lock`).
///
/// The spec singles this out: pi-0 was pre-trained on 10,000 hours of cross-embodiment data
/// and `SmolVLA` on 30,000 community `GPU`-hours, so the backbone's origin directly determines
/// the result — and is the basis for license tracking.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseModel {
    pub source: String,
    pub hash: [u8; 32],
    pub license: String,
}

/// The spec 19.3 `training/` bundle, as digests of its artifacts.
///
/// The artifacts themselves (`config.json`, `optimizer.json`, …) are produced on the learning
/// path (spec 8.8 `es learn export`); identity only needs their digests, which keeps this
/// crate free of any opinion about hyperparameter schemas.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingIdentity {
    pub config: [u8; 32],
    pub optimizer: [u8; 32],
    pub scheduler: [u8; 32],
    pub seed: [u8; 32],
    /// `dataset.lock` — the spec 19.2 three-way hash.
    pub dataset: DatasetHash,
    pub base_model: BaseModel,
    pub augmentation: [u8; 32],
    pub precision: [u8; 32],
    pub topology: [u8; 32],
    pub checkpoint_manifest: [u8; 32],
    pub metrics: [u8; 32],
    pub hardware: [u8; 32],
}

impl TrainingIdentity {
    /// `training_hash = H(all of the above)` (spec 19.3).
    pub fn training_hash(&self) -> Result<[u8; 32], DataError> {
        let mut w = CanonWriter::new();
        w.str(TRAINING_TAG);
        for d in [
            &self.config,
            &self.optimizer,
            &self.scheduler,
            &self.seed,
            &self.dataset.content,
            &self.dataset.schema,
            &self.dataset.split,
        ] {
            w.digest(d);
        }
        w.str(&self.base_model.source);
        w.digest(&self.base_model.hash);
        w.str(&self.base_model.license);
        for d in [
            &self.augmentation,
            &self.precision,
            &self.topology,
            &self.checkpoint_manifest,
            &self.metrics,
            &self.hardware,
        ] {
            w.digest(d);
        }
        canon(w)
    }

    /// `policy_hash = H(training_hash, checkpoint_hash)` (spec 19.3).
    pub fn policy_hash(&self, checkpoint: &[u8; 32]) -> Result<[u8; 32], DataError> {
        let mut w = CanonWriter::new();
        w.str(POLICY_TAG);
        w.digest(&self.training_hash()?);
        w.digest(checkpoint);
        canon(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Deterministic for a seed, and an exact partition of `0..n` (spec 19.2).
        #[test]
        fn split_is_stable_and_partitions(n in 0u32..64, seed in any::<u64>()) {
            let f = [0.8, 0.1, 0.1];
            let a = Split::deterministic(n, f, seed);
            prop_assert_eq!(&a, &Split::deterministic(n, f, seed));

            let mut all: Vec<u32> = a.train.iter().chain(&a.val).chain(&a.test).copied().collect();
            let total = all.len();
            all.sort_unstable();
            all.dedup();
            prop_assert_eq!(all.len(), total, "an episode is in two splits");
            prop_assert_eq!(all, (0..n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn split_hash_follows_the_split() {
        let a = Split::deterministic(20, [0.8, 0.1, 0.1], 20_260_912);
        let b = Split::deterministic(20, [0.8, 0.1, 0.1], 20_260_913);
        assert_ne!(a, b);
        assert_ne!(a.hash().unwrap(), b.hash().unwrap());
        assert_eq!(a.hash().unwrap(), a.clone().hash().unwrap());
    }

    #[test]
    fn training_hash_tracks_base_model_and_dataset() {
        let base = TrainingIdentity {
            config: [1; 32],
            optimizer: [2; 32],
            scheduler: [3; 32],
            seed: [4; 32],
            dataset: DatasetHash {
                content: [5; 32],
                schema: [6; 32],
                split: [7; 32],
            },
            base_model: BaseModel {
                source: "lerobot/smolvla_base".into(),
                hash: [8; 32],
                license: "apache-2.0".into(),
            },
            augmentation: [9; 32],
            precision: [10; 32],
            topology: [11; 32],
            checkpoint_manifest: [12; 32],
            metrics: [13; 32],
            hardware: [14; 32],
        };
        let h = base.training_hash().unwrap();

        let mut other = base.clone();
        other.base_model.license = "cc-by-nc-4.0".into();
        assert_ne!(h, other.training_hash().unwrap());

        let mut other = base.clone();
        other.dataset.split = [99; 32];
        assert_ne!(h, other.training_hash().unwrap());

        assert_ne!(
            base.policy_hash(&[1; 32]).unwrap(),
            base.policy_hash(&[2; 32]).unwrap()
        );
    }
}
