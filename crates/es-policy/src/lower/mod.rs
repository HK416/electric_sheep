//! Lowering a `LearningGraph` to an executable artifact (spec 8.7).
//!
//! One target so far: `PyTorch`, which is the reference oracle (spec 1.4). ONNX (M2) and
//! Vulkan (M3) are siblings of [`torch`], not layers on top of it.

pub mod torch;

use std::collections::BTreeMap;

use es_ir::learning::LearningGraph;
use serde::{Deserialize, Serialize};

pub use torch::{lower_to_torch, LowerError, TorchModule};

/// Everything a training run outside the Rust core needs to know about a lowered module, and
/// everything `es policy pack` checks the resulting checkpoint against (spec 25.1: weights
/// cross a trust boundary, so an unknown key is refused rather than ignored).
///
/// Written as `contract.json` by `es policy lower` next to `es_policy.py`, and read by
/// `python/es/train_act.py`. The optimizer is on the Python side of spec 2.3's split, and this
/// plus the module source is the whole of what crosses it — no hyperparameter here reaches a
/// hash slot (spec 8.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    /// [`TorchModule::lowering_hash`] as hex, so a checkpoint can be traced to the module it
    /// was trained against (spec 5.3).
    pub lowering_hash: String,
    pub weight_keys: Vec<String>,
    pub weight_shapes: BTreeMap<String, Vec<u64>>,
    pub action_dim: u32,
    pub horizon: u32,
    /// Graph input port -> the shape `forward(**inputs)` expects for it.
    pub inputs: BTreeMap<String, Vec<u64>>,
}

impl Contract {
    pub fn new(module: &TorchModule, graph: &LearningGraph) -> Self {
        Self {
            lowering_hash: crate::weights::hex(&module.lowering_hash),
            weight_keys: module.weight_keys.clone(),
            weight_shapes: module.weight_shapes.clone(),
            action_dim: graph.policy.contract.action_dim,
            horizon: graph.policy.contract.horizon,
            inputs: graph
                .inputs
                .iter()
                .map(|p| (p.name.clone(), p.ty.shape.dims().to_vec()))
                .collect(),
        }
    }
}
