//! Learning IR: tensor to action chunk — encoders, fusion, temporal model, policy head and
//! action decoding (spec 8). Batch semantics are free here (spec 5.2): no shape in this module
//! carries a batch axis.
//!
//! The rule of spec 8.1 is "the network is opaque, its interface is typed". Everything the
//! compiler can check lives in [`PolicyContract`] (spec 8.4); the weights themselves are a
//! reference ([`WeightsRef`]), never inline tensors, and the reference has no pickle variant,
//! so `INV-16` is structural rather than a review rule.
//!
//! Not here: `ActionSpec`, `ActionSpace` and `ActionLimits` of spec 8.5 — those are declared by
//! Task IR and consumed by the Safety Plane (spec 9); the Learning IR only needs the execution
//! mode and the chunk-transition data, which are [`ActionExecutionMode`] and
//! [`ChunkBlendPolicy`].
//!
//! Layout: the node set and its parameter types are `learning/node.rs` and
//! [`LearningGraph::validate`] is `learning/validate.rs`; the port helpers, the policy handle,
//! the graph, its two hashes and the test fixtures are here, and every type keeps its
//! `es_ir::learning::` path.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::diag::Diagnostic;
use crate::graph::{Graph, Port};
use crate::hash::{canonical_hash, CanonWriter};
use crate::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

mod node;
mod validate;

pub use node::{
    ActionExecutionMode, Activation, BetaSchedule, ChunkBlendPolicy, DiffusionScheduler,
    FusionKind, HeadKind, LearningNode, NormalizeDir, PredictionType, Squash, StateEncoderKind,
    StatsSource, TemporalKind, VarianceType, VisionBackbone,
};

/// Domain separators for the two hashes of spec 5.3.
const LEARNING_TAG: &str = "es.learning_hash.v1";
const POLICY_TAG: &str = "es.policy_hash.v1";

/// One named tensor port. Spec 8.2 calls this `TensorPort`; it is exactly [`Port`].
pub type TensorPort = Port;

// --- Port helpers -------------------------------------------------------------------------

fn policy_ty(shape: Shape, unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape,
        unit,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    }
}

/// A feature port: `[dim]` when `tokens == 0`, `[tokens, dim]` otherwise.
fn feature(name: &str, dim: u32, tokens: u32) -> TensorPort {
    let shape = if tokens == 0 {
        Shape::new([u64::from(dim)])
    } else {
        Shape::new([u64::from(tokens), u64::from(dim)])
    };
    TensorPort::new(name, policy_ty(shape, Unit::Dimensionless))
}

/// The space an action chunk lives in before unnormalization (spec 8.4).
pub fn action_unit() -> Unit {
    Unit::Normalized { lo: -1.0, hi: 1.0 }
}

fn chunk(name: &str, horizon: u32, action_dim: u32) -> TensorPort {
    TensorPort::new(
        name,
        policy_ty(
            Shape::new([u64::from(horizon), u64::from(action_dim)]),
            action_unit(),
        ),
    )
}

fn canon_ports(w: &mut CanonWriter, ports: &[TensorPort]) {
    w.seq(ports.len());
    for p in ports {
        w.str(&p.name);
        p.ty.canonical(w);
    }
}

/// `Unit` has no public canonical encoder; its `Debug` is derived, total and round-trips
/// floats, so it is a sound hash input.
fn canon_unit(w: &mut CanonWriter, unit: &Unit) {
    w.str(&format!("{unit:?}"));
}

// --- Policy handle (spec 8.4, Appendix B.3) -----------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchKind {
    Act,
    Diffusion,
    FlowMatching,
    Discrete,
    Bundle,
}

/// A pretrained checkpoint the weights were derived from, e.g. `"lerobot/pi05_base"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseModelRef {
    pub uri: String,
    pub hash: [u8; 32],
    pub license: String,
}

/// Where the weights live and in what format.
///
/// `INV-16`: there is no pickle variant and there never will be one — a format that executes
/// code on load has no business in a deployment artefact. Adding one is an API break, which is
/// the point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightsRef {
    Safetensors { path: String, hash: [u8; 32] },
    Onnx { path: String, hash: [u8; 32] },
    SpirV { path: String, hash: [u8; 32] },
}

impl WeightsRef {
    pub fn hash(&self) -> &[u8; 32] {
        match self {
            Self::Safetensors { hash, .. } | Self::Onnx { hash, .. } | Self::SpirV { hash, .. } => {
                hash
            }
        }
    }

    pub fn path(&self) -> &str {
        match self {
            Self::Safetensors { path, .. } | Self::Onnx { path, .. } | Self::SpirV { path, .. } => {
                path
            }
        }
    }

    fn format(&self) -> &'static str {
        match self {
            Self::Safetensors { .. } => "Safetensors",
            Self::Onnx { .. } => "Onnx",
            Self::SpirV { .. } => "SpirV",
        }
    }

    fn canonical(&self, w: &mut CanonWriter) {
        w.str(self.format());
        w.str(self.path());
        w.digest(self.hash());
    }
}

/// `dtype`, `expected_latency_ms`, `deadline_ms` of spec 8.4. Latency is a measurement on the
/// reference device of spec 0.3, not a promise.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeHints {
    pub dtype: ElemType,
    pub expected_latency_ms: f32,
    /// Exceeding this triggers the spec 9.4 fallback.
    pub deadline_ms: f32,
}

/// The metadata contract of spec 8.4 — what the compiler checks about an opaque policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolicyContract {
    pub inputs: BTreeMap<String, TensorPort>,
    pub observation_window: u32,
    pub action_dim: u32,
    /// Prediction length `H`.
    pub horizon: u32,
    /// Execution length `K ≤ H`.
    pub execute_chunk: u32,
    pub replanning_hz: f32,
    pub execution_mode: ActionExecutionMode,
    pub runtime: RuntimeHints,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolicyHandle {
    pub architecture: ArchKind,
    pub base_model: Option<BaseModelRef>,
    pub weights: WeightsRef,
    pub contract: PolicyContract,
}

// --- Graph --------------------------------------------------------------------------------

/// Appendix B.3. `inputs` and `outputs` are the contract with the Observation IR and with
/// `ActionSpec`; the graph's own boundary (`nodes.inputs` / `nodes.outputs`) says which node
/// port each one lands on, positionally.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearningGraph {
    pub schema_version: u32,
    pub inputs: Vec<TensorPort>,
    pub nodes: Graph<LearningNode>,
    pub outputs: Vec<TensorPort>,
    pub policy: PolicyHandle,
}

impl LearningGraph {
    /// `learning_hash` (spec 5.3): the architecture only. Independent of node ids and node
    /// order, and independent of the weights — changing a checkpoint must not change it.
    ///
    /// A `share` names a node by id, and an id is never hash input, so it is hashed the way
    /// IR-C hashes its child ids (`ControlGraph::as_graph`): as an edge `owner.share ->
    /// sharer.share`. No real edge can look like one — a `VisionEncoder`'s only output is
    /// `out` — and a graph without `share` gets no edge, so its hash is today's.
    pub fn learning_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let mut nodes = self.nodes.clone();
        for (id, node) in &self.nodes.nodes {
            if let LearningNode::VisionEncoder {
                share: Some(owner), ..
            } = node
            {
                nodes.connect(*owner, "share", *id, "share");
            }
        }
        let graph = canonical_hash(&nodes)?;
        let c = &self.policy.contract;
        let mut w = CanonWriter::new();
        w.str(LEARNING_TAG);
        w.u32(self.schema_version);
        w.digest(&graph);
        canon_ports(&mut w, &self.inputs);
        canon_ports(&mut w, &self.outputs);
        w.str(&format!("{:?}", self.policy.architecture));
        w.seq(c.inputs.len());
        for (name, port) in &c.inputs {
            w.str(name);
            w.str(&port.name);
            port.ty.canonical(&mut w);
        }
        for v in [
            c.observation_window,
            c.action_dim,
            c.horizon,
            c.execute_chunk,
        ] {
            w.u32(v);
        }
        w.f32(c.replanning_hz);
        w.str(&format!("{:?}", c.execution_mode));
        w.str(&format!("{:?}", c.runtime.dtype));
        w.f32(c.runtime.expected_latency_ms);
        w.f32(c.runtime.deadline_ms);
        w.hash()
    }

    /// `policy_hash = H(weights, learning_hash, base_model)` (spec 5.3).
    pub fn policy_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let learning = self.learning_hash()?;
        let mut w = CanonWriter::new();
        w.str(POLICY_TAG);
        self.policy.weights.canonical(&mut w);
        w.digest(&learning);
        match &self.policy.base_model {
            None => w.bool(false),
            Some(m) => {
                w.bool(true);
                w.str(&m.uri);
                w.digest(&m.hash);
                w.str(&m.license);
            }
        }
        w.hash()
    }
}

// --- Test support -------------------------------------------------------------------------

#[cfg(any(test, feature = "testing"))]
pub mod testing {
    //! Fixtures and the proptest generator for the Appendix B.7 properties (P25).

    // `use super::*` is how a test-support module reads; clippy only exempts `#[cfg(test)]`
    // ones automatically, and this one is also reachable through `feature = "testing"`.
    #![allow(clippy::wildcard_imports)]

    use super::*;
    use crate::graph::{NodeId, PortRef};
    use proptest::prelude::*;

    /// An ACT-shaped graph: image + state encoders, fusion, temporal encoder, regression head,
    /// chunker. Every generated graph validates clean, so it is usable as a "valid input"
    /// generator for downstream passes.
    pub fn act_like(
        state_dim: u32,
        feat: u32,
        action_dim: u32,
        horizon: u32,
        execute_chunk: u32,
        n_frames: u32,
    ) -> LearningGraph {
        let rgb = TensorPort::new(
            "rgb_front",
            policy_ty(
                Shape::new([3, 224, 224]),
                Unit::Normalized { lo: 0.0, hi: 1.0 },
            ),
        );
        let state = TensorPort::new(
            "joint_state",
            policy_ty(Shape::new([u64::from(state_dim)]), action_unit()),
        );

        let mut g = Graph::new(1);
        g.insert(
            NodeId(0),
            LearningNode::VisionEncoder {
                inputs: vec![rgb.clone()],
                backbone: VisionBackbone::ResNet18,
                pretrained: true,
                frozen: false,
                out_dim: feat,
                token_count: 0,
                share: None,
            },
        );
        g.insert(
            NodeId(1),
            LearningNode::StateEncoder {
                inputs: vec![state.clone()],
                kind: StateEncoderKind::Mlp {
                    hidden: vec![256],
                    activation: Activation::Relu,
                    activate_output: false,
                },
                out_dim: feat,
            },
        );
        g.insert(
            NodeId(2),
            LearningNode::Fusion {
                inputs: vec![feature("image", feat, 0), feature("state", feat, 0)],
                kind: FusionKind::Concat,
                out_dim: feat,
                token_count: 0,
            },
        );
        g.insert(
            NodeId(3),
            LearningNode::TemporalEncoder {
                inputs: vec![feature("seq", feat, 0)],
                kind: TemporalKind::Transformer,
                n_frames,
                out_dim: feat,
                token_count: 0,
            },
        );
        g.insert(
            NodeId(4),
            LearningNode::PolicyHead {
                inputs: vec![feature("feat", feat, 0)],
                kind: HeadKind::Regression,
                action_dim,
                horizon,
                squash: Squash::None,
            },
        );
        g.insert(
            NodeId(5),
            LearningNode::ActionChunker {
                inputs: vec![chunk("chunk", horizon, action_dim)],
                horizon,
                execute_chunk,
                replan_hz: 10.0,
                mode: ActionExecutionMode::TemporalEnsemble,
                blend: ChunkBlendPolicy::TemporalEnsemble { weight_decay: 0.01 },
                buffer_chunks: 2,
            },
        );
        g.connect(NodeId(0), "out", NodeId(2), "image");
        g.connect(NodeId(1), "out", NodeId(2), "state");
        g.connect(NodeId(2), "out", NodeId(3), "seq");
        g.connect(NodeId(3), "out", NodeId(4), "feat");
        g.connect(NodeId(4), "chunk", NodeId(5), "chunk");
        g.inputs = vec![
            PortRef::new(NodeId(0), "rgb_front"),
            PortRef::new(NodeId(1), "joint_state"),
        ];
        g.outputs = vec![PortRef::new(NodeId(5), "actions")];

        let contract = PolicyContract {
            inputs: [
                (rgb.name.clone(), rgb.clone()),
                (state.name.clone(), state.clone()),
            ]
            .into_iter()
            .collect(),
            observation_window: n_frames,
            action_dim,
            horizon,
            execute_chunk,
            replanning_hz: 10.0,
            execution_mode: ActionExecutionMode::TemporalEnsemble,
            runtime: RuntimeHints {
                dtype: ElemType::F32,
                expected_latency_ms: 5.0,
                deadline_ms: 50.0,
            },
        };
        LearningGraph {
            schema_version: 1,
            inputs: vec![rgb, state],
            outputs: vec![chunk("actions", execute_chunk, action_dim)],
            nodes: g,
            policy: PolicyHandle {
                architecture: ArchKind::Act,
                base_model: None,
                weights: WeightsRef::Safetensors {
                    path: "policy.safetensors".to_owned(),
                    hash: [7u8; 32],
                },
                contract,
            },
        }
    }

    pub fn arbitrary_learning_graph() -> impl Strategy<Value = LearningGraph> {
        (1u32..16, 1u32..512, 1u32..32, 1u32..64, 1u32..4).prop_flat_map(
            |(state_dim, feat, action_dim, horizon, n_frames)| {
                (1..=horizon).prop_map(move |execute_chunk| {
                    act_like(
                        state_dim,
                        feat,
                        action_dim,
                        horizon,
                        execute_chunk,
                        n_frames,
                    )
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{act_like, arbitrary_learning_graph};
    use super::*;
    use crate::codes;
    use crate::graph::{NodeId, PortRef};
    use proptest::prelude::*;

    fn act() -> LearningGraph {
        act_like(8, 512, 8, 50, 20, 1)
    }

    fn errors(g: &LearningGraph) -> Vec<String> {
        g.validate()
            .into_iter()
            .filter(Diagnostic::is_error)
            .map(|d| format!("{} {}", d.code, d.message))
            .collect()
    }

    fn has(g: &LearningGraph, code: &str) -> bool {
        g.validate().iter().any(|d| d.code.as_str() == code)
    }

    #[test]
    fn act_fixture_validates_clean() {
        let g = act();
        assert!(errors(&g).is_empty(), "{:#?}", errors(&g));
    }

    #[test]
    fn serde_round_trip() {
        let g = act();
        let json = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<LearningGraph>(&json).unwrap(), g);
    }

    #[test]
    fn raw_unit_into_the_policy_is_type_011() {
        let mut g = act();
        g.inputs[1].ty.unit = Unit::Length;
        assert!(has(&g, codes::TYPE_011), "{:#?}", errors(&g));
    }

    #[test]
    fn execute_chunk_above_horizon_is_lrn_020() {
        let mut g = act();
        g.policy.contract.execute_chunk = 80;
        assert!(has(&g, codes::LRN_020), "{:#?}", errors(&g));
    }

    #[test]
    fn contract_disagreeing_with_the_head_is_lrn_021() {
        let mut g = act();
        g.policy.contract.action_dim = 9;
        assert!(has(&g, codes::LRN_021), "{:#?}", errors(&g));
    }

    #[test]
    fn observation_window_must_match_the_temporal_node() {
        let mut g = act();
        g.policy.contract.observation_window = 2;
        assert!(has(&g, codes::LRN_022), "{:#?}", errors(&g));
    }

    #[test]
    fn slow_policy_inside_a_fast_loop_is_lrn_052() {
        let mut g = act();
        // Diffusion Policy on an RTX 4090 at 10 Hz control (spec 8.4).
        g.policy.contract.runtime.expected_latency_ms = 369.8;
        assert!(has(&g, codes::LRN_052), "{:#?}", errors(&g));
    }

    #[test]
    fn inverse_normalizer_must_leave_the_normalized_space() {
        let mut g = act();
        g.nodes.insert(
            NodeId(9),
            LearningNode::Normalizer {
                inputs: vec![chunk("x", 20, 8)],
                direction: NormalizeDir::Inverse,
                stats: StatsSource::Dataset {
                    dataset_hash: [1u8; 32],
                },
                out_unit: Unit::Dimensionless,
            },
        );
        assert!(has(&g, codes::LRN_030), "{:#?}", errors(&g));
    }

    #[test]
    fn wrong_input_count_is_lrn_002() {
        let mut g = act();
        let LearningNode::Fusion { inputs, .. } = g.nodes.nodes.get_mut(&NodeId(2)).unwrap() else {
            unreachable!()
        };
        inputs.truncate(1);
        assert!(has(&g, codes::LRN_002), "{:#?}", errors(&g));
    }

    #[test]
    fn learning_hash_ignores_node_labels() {
        let a = act();
        let mut b = a.clone();
        // Relabel every node by +100, keeping the same structure.
        let bump = |id: NodeId| NodeId(id.0 + 100);
        b.nodes.nodes = a
            .nodes
            .nodes
            .iter()
            .map(|(k, v)| (bump(*k), v.clone()))
            .collect();
        for e in &mut b.nodes.edges {
            e.from.node = bump(e.from.node);
            e.to.node = bump(e.to.node);
        }
        for side in [&mut b.nodes.inputs, &mut b.nodes.outputs] {
            for p in side.iter_mut() {
                *p = PortRef::new(bump(p.node), p.port.clone());
            }
        }
        assert_ne!(a.nodes.nodes, b.nodes.nodes);
        assert_eq!(a.learning_hash().unwrap(), b.learning_hash().unwrap());
        assert!(errors(&b).is_empty(), "{:#?}", errors(&b));
    }

    #[test]
    fn learning_hash_sees_architecture_changes() {
        let a = act();
        let b = act_like(8, 256, 8, 50, 20, 1);
        assert_ne!(a.learning_hash().unwrap(), b.learning_hash().unwrap());
    }

    #[test]
    fn new_weights_move_only_the_policy_hash() {
        let a = act();
        let mut b = a.clone();
        b.policy.weights = WeightsRef::Safetensors {
            path: "policy.safetensors".to_owned(),
            hash: [9u8; 32],
        };
        assert_eq!(a.learning_hash().unwrap(), b.learning_hash().unwrap());
        assert_ne!(a.policy_hash().unwrap(), b.policy_hash().unwrap());

        // A base model is part of the policy identity too (spec 5.3).
        let mut c = a.clone();
        c.policy.base_model = Some(BaseModelRef {
            uri: "lerobot/pi05_base".to_owned(),
            hash: [3u8; 32],
            license: "apache-2.0".to_owned(),
        });
        assert_eq!(a.learning_hash().unwrap(), c.learning_hash().unwrap());
        assert_ne!(a.policy_hash().unwrap(), c.policy_hash().unwrap());
    }

    #[test]
    fn weights_ref_has_no_executable_format() {
        // INV-16 is structural: these are the only three constructors there are.
        for w in [
            WeightsRef::Safetensors {
                path: "a".to_owned(),
                hash: [0u8; 32],
            },
            WeightsRef::Onnx {
                path: "a".to_owned(),
                hash: [0u8; 32],
            },
            WeightsRef::SpirV {
                path: "a".to_owned(),
                hash: [0u8; 32],
            },
        ] {
            assert_eq!(w.path(), "a");
            assert_ne!(w.format(), "Pickle");
        }
    }

    proptest! {
        #[test]
        fn arbitrary_graphs_validate_and_round_trip(g in arbitrary_learning_graph()) {
            let diags: Vec<_> = g.validate().into_iter().filter(Diagnostic::is_error).collect();
            prop_assert!(diags.is_empty(), "{diags:#?}");
            let json = serde_json::to_string(&g).unwrap();
            prop_assert_eq!(serde_json::from_str::<LearningGraph>(&json).unwrap(), g);
        }
    }
}
