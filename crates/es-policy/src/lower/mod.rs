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
    /// Graph input port -> the shape of **one sample** of it. Spec 5.2 keeps the batch out of
    /// every declared shape, so this is the IR's shape verbatim; `batch_axis` says what the
    /// module does with it.
    pub inputs: BTreeMap<String, Vec<u64>>,
    /// The module's `forward` takes every input with a leading batch axis and returns every
    /// output with one: `inputs[p]` is fed as `[N, ..inputs[p]]` (packet M7/T3).
    ///
    /// Always `true` for a module [`lower_to_torch`] produced. It is written down rather than
    /// assumed because a reader that does not know the key treats the module as single-sample
    /// and then fails on a shape, which is the failure one wants: silently feeding `[C, H, W]`
    /// to a module expecting `[N, C, H, W]` is how a trainer optimizes the wrong function.
    pub batch_axis: bool,
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
            batch_axis: true,
        }
    }
}

/// The ACT fixture with the backbone trained from scratch.
///
/// `es_ir::learning::testing::act_like` declares `pretrained = true` — which is what ACT
/// really is — and `lower_to_torch` now refuses that rather than silently ignoring it
/// (packet `docs/packets/M5/V2b-observation-bake.md`). The IR fixture keeps telling the
/// truth; the tests that put it through the lowering clear the one flag here, in one place.
#[cfg(test)]
pub(crate) fn act_from_scratch(
    state_dim: u32,
    feat: u32,
    heads: u32,
    horizon: u32,
    execute: u32,
    obs_window: u32,
) -> LearningGraph {
    let mut g =
        es_ir::learning::testing::act_like(state_dim, feat, heads, horizon, execute, obs_window);
    for node in g.nodes.nodes.values_mut() {
        if let es_ir::learning::LearningNode::VisionEncoder { pretrained, .. } = node {
            *pretrained = false;
        }
    }
    g
}
