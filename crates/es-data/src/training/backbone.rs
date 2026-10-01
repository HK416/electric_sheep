//! The pretrained backbone (packets M7/T5, M12/R8): whether a Learning IR asks for one, how
//! `[policy] base_model_fetch` obtains it, and the lock that proves a file on disk is the
//! pinned one. The pin itself, [`RESNET18_IMAGENET1K_V1_BLAKE3`], stays in `training.rs`,
//! where the tests that read it as text look for it.

use std::path::Path;

use es_ir::learning::{LearningGraph, LearningNode};
use serde::{Deserialize, Serialize};

use super::{refuse, s, Recipe, BASE_MODEL_SOURCE, RESNET18_IMAGENET1K_V1_BLAKE3};
use crate::collect::hex;
use crate::DataError;

/// What writes a `base_model`, relative to the repository root (packets M7/T5, M12/R8).
pub const FETCH_BACKBONE: &str = "python/es/fetch_backbone.py";

impl Recipe {
    /// `[policy] base_model_fetch` validated, as [`FETCH_BACKBONE`]'s flags (packet M12/R8).
    ///
    /// Empty when the recipe declares none, which keeps every other plan and its golden
    /// byte-identical. `--out` is `base_model`'s directory and `--expect` the pin, so the file
    /// the script writes is the file this run reads, or the script refuses to write it.
    pub fn fetch_args(&self) -> Result<Vec<String>, DataError> {
        let Some(arch) = &self.policy.base_model_fetch else {
            return Ok(Vec::new());
        };
        let Some(path) = &self.policy.base_model else {
            return Err(refuse(
                "[policy] `base_model_fetch` says how to obtain `base_model`, and `base_model` \
                 names no file",
            ));
        };
        // One arch because one pin: a second is a licence decision and a measured hash
        // (`fetch_backbone.py`'s `ARCHS`), not a word in a recipe.
        if arch != "resnet18" {
            return Err(refuse(format!(
                "[policy] `base_model_fetch` is {arch:?}; the one backbone this repository has \
                 approved and pinned is \"resnet18\" ({BASE_MODEL_SOURCE})"
            )));
        }
        let want = format!("{arch}-imagenet1k-v1.safetensors");
        let path = Path::new(path);
        if path.file_name().and_then(|n| n.to_str()) != Some(want.as_str()) {
            return Err(refuse(format!(
                "[policy] `base_model` is {}, and `base_model_fetch` writes <dir>/{want}: the \
                 fetched file would not be the one this run reads",
                path.display()
            )));
        }
        let dir = path
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        Ok(vec![
            s("--arch"),
            arch.clone(),
            s("--out"),
            dir.to_string_lossy().into_owned(),
            s("--expect"),
            s(RESNET18_IMAGENET1K_V1_BLAKE3),
        ])
    }
}

/// Does the Learning IR declare a pretrained backbone? Then the recipe owes
/// `[policy] base_model` (packet M7/T5).
pub fn has_pretrained_backbone(learning: &LearningGraph) -> bool {
    learning
        .nodes
        .nodes
        .values()
        .any(|n| matches!(n, LearningNode::VisionEncoder { pretrained, .. } if *pretrained))
}

/// `<weights>.lock.json` as `python/es/fetch_backbone.py` writes it, reduced to the fields
/// that identify **the weights** (spec 19.3's `base_model.lock`).
///
/// The lock file beside the artifact also records the `torch` and `torchvision` that fetched
/// it; those are deliberately not here. Two machines fetching the same upstream file write
/// byte-identical tensors and different version strings, and carrying the strings into
/// `base_model.lock` would put the fetching machine into `identity_hash` — so one recipe
/// would have two identities depending on where its backbone was produced.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Backbone {
    /// `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`.
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub url: String,
    /// torchvision's own hash of the `.pth` it downloaded.
    #[serde(default)]
    pub sha256_upstream: String,
    /// blake3 of the safetensors file — the pin.
    #[serde(default)]
    pub blake3: String,
    /// The key the artifact does not carry, named rather than left to be noticed.
    #[serde(default)]
    pub dropped: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub license_url: String,
}

impl Backbone {
    /// Reads `weights` and the `.lock.json` beside it, and refuses by name unless the three
    /// claims agree: the file's blake3, the lock file's, and [`RESNET18_IMAGENET1K_V1_BLAKE3`].
    ///
    /// A licence field that is empty is a refusal of its own: spec 19.3 makes this file the
    /// basis for licence tracking, and a provenance record with nothing in that slot tracks
    /// nothing.
    pub fn verify(weights: &Path) -> Result<Self, DataError> {
        let bytes = std::fs::read(weights)
            .map_err(|e| refuse(format!("[policy] `base_model` {}: {e}", weights.display())))?;
        let digest = hex(blake3::hash(&bytes).as_bytes());
        let lock_path = weights.with_extension("lock.json");
        let text = std::fs::read_to_string(&lock_path).map_err(|e| {
            refuse(format!(
                "{}: {e}\nEvery `base_model` travels with the lock file \
                 `python/es/fetch_backbone.py` writes beside it; it is where the licence and \
                 the upstream URL come from (spec 19.3)",
                lock_path.display()
            ))
        })?;
        let lock: Self = serde_json::from_str(&text)
            .map_err(|e| refuse(format!("{}: {e}", lock_path.display())))?;

        if lock.blake3 != digest {
            return Err(refuse(format!(
                "{} does not hash to what {} claims:\n  file  {digest}\n  lock  {}",
                weights.display(),
                lock_path.display(),
                lock.blake3
            )));
        }
        if lock.source != BASE_MODEL_SOURCE {
            return Err(refuse(format!(
                "{} names the source {:?}; the one this repository has approved is {:?} \
                 (spec 29 licence row)",
                lock_path.display(),
                lock.source,
                BASE_MODEL_SOURCE
            )));
        }
        if lock.license.trim().is_empty() {
            return Err(refuse(format!(
                "{}: `license` is empty. Spec 19.3 makes base_model.lock the basis for \
                 licence tracking, and a run whose backbone carries no licence cannot be \
                 shipped anywhere",
                lock_path.display()
            )));
        }
        // Last, because the three above ask whether the lock file is a well-formed provenance
        // record for these bytes and this one asks whether these bytes are the artifact the
        // repository has actually measured. Both matter; they are different questions.
        if digest != RESNET18_IMAGENET1K_V1_BLAKE3 {
            return Err(refuse(format!(
                "[policy] `base_model` {} is not the pinned backbone:\n  file    \
                 {digest}\n  pinned  {RESNET18_IMAGENET1K_V1_BLAKE3}\nThe pin is \
                 `RESNET18_IMAGENET1K_V1_BLAKE3` in crates/es-data/src/training.rs. Re-fetch \
                 with `python/es/fetch_backbone.py --arch resnet18 --out <dir>`, or move the \
                 pin deliberately (`--repin`) together with the design note",
                weights.display()
            )));
        }
        Ok(lock)
    }
}
