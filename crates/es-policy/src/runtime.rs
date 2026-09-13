//! The `PolicyRuntime` extension point (spec 2.4, `INV-17`).
//!
//! One of the seven single-impl traits the project allows. Everything it does **not** do is
//! the point: spec 8.7 gives preprocessing to the Observation IR and post-processing
//! (chunking, unnormalizing) to the Learning IR, so a runtime only runs a forward pass over
//! named tensors. That is why [`PolicyRuntime::infer`] has no options, no dtype policy and no
//! action spec.

use std::collections::BTreeMap;
use std::path::PathBuf;

use es_compile::Tensor;
use es_ir::learning::LearningGraph;

use crate::lower::LowerError;

/// Domain separator for [`PolicyRuntime::runtime_hash`], the `runtime` slot of
/// `execution_hash` (spec 5.3).
pub const RUNTIME_TAG: &str = "es.runtime_hash.v1";

/// Where a checkpoint's bytes come from.
///
/// `INV-16`: there is no pickle variant and there never will be one. A weight format that
/// executes code on load has no business on a deployment path, and the absence of the variant
/// is what enforces it — adding one is an API break, which is the point. Mirrors
/// `es_ir::learning::WeightsRef`, which is the *declaration*; this is the *source*.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightsSource {
    Safetensors(PathBuf),
    /// The same bytes, already in memory — a bundle that was fetched rather than unpacked.
    InMemory(Vec<u8>),
}

/// Which inference backend a [`PolicyInfo`] came from (spec 2.4).
///
/// An **enum, not a trait**. Spec 2.4 names three backends and `INV-17` reserves
/// `InferenceBackend` as one of the seven extension points, but only `Torch` exists today and
/// a trait with one implementation is a speculative abstraction. The name stays reserved: when
/// `OnnxRuntime` (M2) and `VulkanRuntime` (M3) arrive they implement [`PolicyRuntime`], and if
/// a second axis of variation actually appears the trait slot is still there to take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum InferenceBackend {
    Torch,
    Onnx,
    Vulkan,
}

impl InferenceBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Torch => "torch",
            Self::Onnx => "onnx",
            Self::Vulkan => "vulkan",
        }
    }
}

/// What a successful [`PolicyRuntime::load`] learned. Every field is hash-chain input or a
/// shape the caller needs; nothing here is a tuning knob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyInfo {
    pub backend: InferenceBackend,
    /// blake3 of the lowered artifact (the generated module source, for `Torch`).
    pub lowering_hash: [u8; 32],
    /// blake3 of the checkpoint bytes, checked against `WeightsRef::hash` at load.
    pub weights_hash: [u8; 32],
    pub action_dim: u32,
    pub horizon: u32,
    /// The backend's own version, e.g. `"2.14.0+cpu"`. Part of `runtime_hash`.
    pub version: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PolicyError {
    #[error("no policy is loaded")]
    NotLoaded,
    #[error("lowering failed: {0}")]
    Lower(#[from] LowerError),
    #[error(
        "the checkpoint does not match the lowered graph: {} missing, {} unexpected, {} with the wrong shape",
        missing.len(), unexpected.len(), shape.len()
    )]
    WeightMismatch {
        /// Keys the lowering declared that the file does not have.
        missing: Vec<String>,
        /// Keys the file has that the lowering never declared.
        unexpected: Vec<String>,
        /// `"key: want [..], got [..]"`, one per disagreeing tensor.
        shape: Vec<String>,
    },
    #[error("weights hash mismatch: the IR declares {expected}, the file is {got}")]
    WeightsHash { expected: String, got: String },
    #[error("malformed safetensors: {0}")]
    Safetensors(String),
    #[error("{0}")]
    Io(String),
    #[error("backend unavailable: {0}")]
    Unavailable(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("the backend process died: {0}")]
    ProcessDied(String),
}

/// Running a lowered `LearningGraph`'s forward pass (spec 2.4).
///
/// Object-safe on purpose: a compiled plan holds one as `Box<dyn PolicyRuntime>` and the
/// backend is a deployment decision, not a type parameter.
///
/// The contract:
///
/// - `inputs` to [`infer`](PolicyRuntime::infer) are **already preprocessed** and named per
///   `PolicyContract::inputs` — the Observation IR produced them (spec 7.4).
/// - the outputs are the graph's declared output ports; for a policy graph that is the action
///   chunk `[H, action_dim]`, already unnormalized, because the unnormalizer is an IR node and
///   therefore inside the lowered graph rather than in the runtime (spec 8.7).
/// - [`load`](PolicyRuntime::load) must verify the checkpoint against the graph. A runtime
///   that loads whatever it is handed cannot support spec 5.3's hash chain.
pub trait PolicyRuntime {
    fn load(
        &mut self,
        graph: &LearningGraph,
        weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError>;

    fn infer(
        &mut self,
        inputs: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>, PolicyError>;

    fn info(&self) -> Option<&PolicyInfo>;

    /// The `runtime` slot of `execution_hash` (spec 5.3): what would make a bitwise-identical
    /// re-run impossible if it changed. Never the weights and never the graph — those are
    /// `policy_hash` and `learning_hash`.
    fn runtime_hash(&self) -> [u8; 32];
}

/// `blake3(RUNTIME_TAG || backend || protocol || version)` — the shape every backend's
/// `runtime_hash` takes, so two backends can never collide by accident.
pub fn runtime_hash_of(backend: InferenceBackend, protocol: u32, version: &str) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(RUNTIME_TAG.as_bytes());
    h.update(backend.name().as_bytes());
    h.update(&protocol.to_le_bytes());
    h.update(version.as_bytes());
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `INV-17` lets `PolicyRuntime` exist; object safety is what makes it worth existing.
    #[test]
    fn the_trait_is_object_safe() {
        fn _accepts(_: &dyn PolicyRuntime) {}
        fn _boxes(r: Box<dyn PolicyRuntime>) -> Box<dyn PolicyRuntime> {
            r
        }
    }

    /// `INV-16` is structural: these are the only two sources there are.
    #[test]
    fn weights_source_has_no_executable_format() {
        let sources = [
            WeightsSource::Safetensors(PathBuf::from("p.safetensors")),
            WeightsSource::InMemory(vec![0u8; 4]),
        ];
        for s in &sources {
            assert!(!format!("{s:?}").contains("Pickle"));
        }
        assert_eq!(sources.len(), 2);
    }

    #[test]
    fn runtime_hash_separates_backends_and_versions() {
        let a = runtime_hash_of(InferenceBackend::Torch, 1, "2.14.0+cpu");
        assert_ne!(a, runtime_hash_of(InferenceBackend::Onnx, 1, "2.14.0+cpu"));
        assert_ne!(a, runtime_hash_of(InferenceBackend::Torch, 2, "2.14.0+cpu"));
        assert_ne!(a, runtime_hash_of(InferenceBackend::Torch, 1, "2.13.0+cpu"));
        assert_eq!(a, runtime_hash_of(InferenceBackend::Torch, 1, "2.14.0+cpu"));
    }
}
