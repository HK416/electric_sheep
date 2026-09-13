//! `LearningGraph` to a `PyTorch` module (spec 8.7), the reference oracle of spec 1.4.
//!
//! The node table, the weight-key scheme and what is deferred live in
//! `docs/design/learning-lowering.md`. This file implements that table and nothing else: one
//! `nn.Module` member per parameterised node, one forward line per node, in
//! `Graph::topo_order()` order. No fusion, no reordering, no cleverness — an oracle that is
//! hard to read against the spec is not an oracle.
//!
//! Determinism is a hard requirement, not a nicety: `lowering_hash` is a component of the
//! compiler identity, so the same graph must produce byte-identical source. Everything here
//! iterates a `BTreeMap` or a `Vec` built in topological order (spec 3.4).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use es_ir::graph::{IrNode, NodeId};
use es_ir::learning::{
    FusionKind, HeadKind, LearningGraph, LearningNode, NormalizeDir, StateEncoderKind, StatsSource,
    TemporalKind, VisionBackbone,
};
use es_ir::types::ElemType;
use es_ir::Diagnostic;

use crate::weights::WEIGHT_PREFIX;

/// Domain separator for [`TorchModule::lowering_hash`].
const LOWERING_TAG: &str = "es.lowering.torch.v1";

/// Attention heads in a lowered `TemporalEncoder { Transformer }`. Spec 8.3's node parameters
/// do not carry a head count, so it is a lowering constant; a checkpoint that disagrees fails
/// the shape check rather than being silently reinterpreted (design note section 3).
const TRANSFORMER_HEADS: u64 = 8;

/// A lowered graph: a complete `PyTorch` file plus the checkpoint contract it implies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorchModule {
    /// A self-contained Python file defining `class EsPolicy(nn.Module)` whose
    /// `forward(**inputs) -> dict` takes the graph's declared input names.
    pub source: String,
    /// Safetensors keys this module needs. An entry ending in `.*` is a *prefix claim* over an
    /// opaque sub-module (a torchvision backbone, a `torch.nn` transformer) whose internal
    /// parameter names are not ours to enumerate; every other entry is an exact key and has a
    /// shape in [`weight_shapes`](Self::weight_shapes).
    pub weight_keys: Vec<String>,
    pub weight_shapes: BTreeMap<String, Vec<u64>>,
    /// `blake3(LOWERING_TAG || source)`.
    pub lowering_hash: [u8; 32],
}

impl TorchModule {
    /// Whether `key` ends in `.*`, i.e. covers a sub-module rather than one tensor.
    pub fn is_prefix_claim(key: &str) -> bool {
        key.ends_with(".*")
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LowerError {
    /// A node kind with no `PyTorch` lowering in this packet. The payload is the node kind and
    /// the variant that was unsupported, e.g. `"PolicyHead{Diffusion}"`.
    #[error("no PyTorch lowering for {0} (see docs/design/learning-lowering.md section 7)")]
    Unsupported(String),
    #[error("the graph does not validate: {0}")]
    Invalid(String),
    #[error("node {node}: input port \"{port}\" is fed by neither an edge nor a graph input")]
    Unfed { node: u32, port: String },
    #[error("node {node}: {message}")]
    Shape { node: u32, message: String },
}

/// Lower `graph` to a `PyTorch` module.
///
/// The graph is validated first: an oracle built from an invalid IR proves nothing.
pub fn lower_to_torch(graph: &LearningGraph) -> Result<TorchModule, LowerError> {
    let errors: Vec<String> = graph
        .validate()
        .into_iter()
        .filter(Diagnostic::is_error)
        .map(|d| format!("{} {}", d.code, d.message))
        .collect();
    if !errors.is_empty() {
        return Err(LowerError::Invalid(errors.join("; ")));
    }
    if graph.policy.contract.runtime.dtype != ElemType::F32 {
        return Err(LowerError::Unsupported(format!(
            "dtype {:?}",
            graph.policy.contract.runtime.dtype
        )));
    }

    let mut lo = Lowering::default();
    let order = graph
        .nodes
        .topo_order()
        .map_err(|d| LowerError::Invalid(format!("{} {}", d.code, d.message)))?;

    for id in order {
        let node = &graph.nodes.nodes[&id];
        let args = inputs_of(graph, id, node)?;
        let expr = lo.node_expr(id, node, &args)?;
        let out = node
            .outputs()
            .first()
            .map(|p| p.name.clone())
            .ok_or_else(|| LowerError::Shape {
                node: id.0,
                message: "the node has no output port".to_owned(),
            })?;
        lo.body
            .push(format!("        {} = {expr}", value_of(id, &out)));
    }

    let mut returns = Vec::new();
    for (port, at) in graph.outputs.iter().zip(&graph.nodes.outputs) {
        returns.push(format!("{:?}: {}", port.name, value_of(at.node, &at.port)));
    }

    Ok(lo.finish(&returns))
}

/// The local holding output port `port` of node `id`.
fn value_of(id: NodeId, port: &str) -> String {
    format!("v{}_{}", id.0, port)
}

/// The Python expression feeding each declared input port of `node`, in declared order.
fn inputs_of(
    graph: &LearningGraph,
    id: NodeId,
    node: &LearningNode,
) -> Result<Vec<String>, LowerError> {
    let mut args = Vec::new();
    for port in node.inputs() {
        let edge = graph
            .nodes
            .edges
            .iter()
            .find(|e| e.to.node == id && e.to.port == port.name);
        if let Some(e) = edge {
            args.push(value_of(e.from.node, &e.from.port));
            continue;
        }
        let boundary = graph
            .nodes
            .inputs
            .iter()
            .position(|r| r.node == id && r.port == port.name)
            .and_then(|i| graph.inputs.get(i));
        match boundary {
            Some(declared) => args.push(format!("inputs[{:?}]", declared.name)),
            None => {
                return Err(LowerError::Unfed {
                    node: id.0,
                    port: port.name,
                })
            }
        }
    }
    Ok(args)
}

/// The last dimension of a node's `i`-th declared input port — the feature width every module
/// here is sized by.
fn in_dim(node: &LearningNode, i: usize, id: NodeId) -> Result<u64, LowerError> {
    node.inputs()
        .get(i)
        .and_then(|p| p.ty.shape.dims().last().copied())
        .filter(|d| *d > 0)
        .ok_or_else(|| LowerError::Shape {
            node: id.0,
            message: format!("input port {i} has no usable trailing dimension"),
        })
}

fn unsupported(kind: &str, variant: &impl std::fmt::Debug) -> LowerError {
    LowerError::Unsupported(format!("{kind}{{{variant:?}}}"))
}

#[derive(Default)]
struct Lowering {
    members: Vec<String>,
    body: Vec<String>,
    keys: Vec<String>,
    shapes: BTreeMap<String, Vec<u64>>,
    needs_torchvision: bool,
}

impl Lowering {
    fn member(&mut self, id: NodeId, ctor: &str) {
        self.members
            .push(format!("        self.n{} = {ctor}", id.0));
    }

    /// Declares an exact key. `sub` is the `state_dict` path below the node's member.
    fn exact(&mut self, id: NodeId, sub: &str, shape: Vec<u64>) {
        let key = format!("{WEIGHT_PREFIX}{}.{sub}", id.0);
        self.keys.push(key.clone());
        self.shapes.insert(key, shape);
    }

    /// Declares a prefix claim over an opaque sub-module.
    fn claim(&mut self, id: NodeId) {
        self.keys.push(format!("{WEIGHT_PREFIX}{}.*", id.0));
    }

    /// A `nn.Linear` member plus its two exact keys.
    fn linear(&mut self, id: NodeId, in_dim: u64, out_dim: u64) {
        self.member(id, &format!("nn.Linear({in_dim}, {out_dim})"));
        self.exact(id, "weight", vec![out_dim, in_dim]);
        self.exact(id, "bias", vec![out_dim]);
    }

    // One arm per node kind. Splitting it would hide the table it implements.
    fn node_expr(
        &mut self,
        id: NodeId,
        node: &LearningNode,
        args: &[String],
    ) -> Result<String, LowerError> {
        let k = id.0;
        match node {
            LearningNode::VisionEncoder {
                backbone, out_dim, ..
            } => {
                let name = match backbone {
                    VisionBackbone::ResNet18 => "resnet18",
                    VisionBackbone::ResNet34 => "resnet34",
                    other => return Err(unsupported("VisionEncoder", other)),
                };
                self.needs_torchvision = true;
                self.member(id, &format!("_backbone({name:?}, {out_dim})"));
                self.claim(id);
                Ok(format!("self.n{k}({})", args[0]))
            }

            LearningNode::StateEncoder { kind, out_dim, .. } => match kind {
                StateEncoderKind::Identity => Ok(args[0].clone()),
                StateEncoderKind::Mlp { hidden } => {
                    let mut dims = vec![in_dim(node, 0, id)?];
                    dims.extend(hidden.iter().map(|h| u64::from(*h)));
                    dims.push(u64::from(*out_dim));
                    let mut layers = Vec::new();
                    for (i, w) in dims.windows(2).enumerate() {
                        if i > 0 {
                            layers.push("nn.ReLU()".to_owned());
                        }
                        layers.push(format!("nn.Linear({}, {})", w[0], w[1]));
                        let at = layers.len() - 1;
                        self.exact(id, &format!("{at}.weight"), vec![w[1], w[0]]);
                        self.exact(id, &format!("{at}.bias"), vec![w[1]]);
                    }
                    self.member(id, &format!("nn.Sequential({})", layers.join(", ")));
                    Ok(format!("self.n{k}({})", args[0]))
                }
            },

            LearningNode::LanguageEncoder { .. } => {
                Err(LowerError::Unsupported("LanguageEncoder".to_owned()))
            }

            LearningNode::Fusion { kind, out_dim, .. } => {
                let out_dim = u64::from(*out_dim);
                let widths: Result<Vec<u64>, LowerError> =
                    (0..args.len()).map(|i| in_dim(node, i, id)).collect();
                let widths = widths?;
                match kind {
                    FusionKind::Concat => {
                        let cat = format!("torch.cat([{}], dim=-1)", args.join(", "));
                        let sum: u64 = widths.iter().sum();
                        if sum == out_dim {
                            Ok(cat)
                        } else {
                            self.linear(id, sum, out_dim);
                            Ok(format!("self.n{k}({cat})"))
                        }
                    }
                    FusionKind::TokenConcat => {
                        if widths.iter().any(|w| *w != out_dim) {
                            return Err(LowerError::Shape {
                                node: k,
                                message: format!(
                                    "TokenConcat needs every input at width {out_dim}, got {widths:?}"
                                ),
                            });
                        }
                        Ok(format!("torch.cat([{}], dim=-2)", args.join(", ")))
                    }
                    other => Err(unsupported("Fusion", other)),
                }
            }

            LearningNode::TemporalEncoder {
                kind,
                out_dim,
                token_count,
                ..
            } => match kind {
                TemporalKind::None => Ok(args[0].clone()),
                TemporalKind::Transformer => {
                    let d = u64::from(*out_dim);
                    let got = in_dim(node, 0, id)?;
                    if got != d {
                        return Err(LowerError::Shape {
                            node: k,
                            message: format!("a transformer keeps its width: in {got}, out {d}"),
                        });
                    }
                    if d % TRANSFORMER_HEADS != 0 {
                        return Err(LowerError::Shape {
                            node: k,
                            message: format!(
                                "d_model {d} is not a multiple of {TRANSFORMER_HEADS} heads"
                            ),
                        });
                    }
                    self.member(
                        id,
                        &format!(
                            "nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d}, nhead={TRANSFORMER_HEADS}, batch_first=True), num_layers=1)"
                        ),
                    );
                    self.claim(id);
                    // A pooled feature has no token axis; run it as a one-token sequence.
                    Ok(if *token_count == 0 {
                        format!("self.n{k}({}.unsqueeze(0)).squeeze(0)", args[0])
                    } else {
                        format!("self.n{k}({})", args[0])
                    })
                }
                other => Err(unsupported("TemporalEncoder", other)),
            },

            LearningNode::PolicyHead {
                kind,
                action_dim,
                horizon,
                ..
            } => match kind {
                HeadKind::Regression => {
                    let width = u64::from(*horizon) * u64::from(*action_dim);
                    self.linear(id, in_dim(node, 0, id)?, width);
                    Ok(format!(
                        "self.n{k}({}).reshape({horizon}, {action_dim})",
                        args[0]
                    ))
                }
                other => Err(unsupported("PolicyHead", other)),
            },

            LearningNode::PolicyBundle { .. } => {
                Err(LowerError::Unsupported("PolicyBundle".to_owned()))
            }

            // Only `execute_chunk` is a tensor operation; the rest of the node is runtime
            // scheduling (spec 8.6) and is not observable in one `infer`.
            LearningNode::ActionChunker { execute_chunk, .. } => {
                Ok(format!("{}[:{execute_chunk}]", args[0]))
            }

            LearningNode::Normalizer {
                direction, stats, ..
            } => {
                let (first, second) = match stats {
                    StatsSource::MeanStd { mean, std } => (mean, std),
                    StatsSource::MinMax { lo, hi } => (lo, hi),
                    StatsSource::Dataset { .. } => {
                        return Err(LowerError::Unsupported(
                            "Normalizer{Dataset} (resolving a dataset_hash is es-data's job, \
                             layer 10)"
                                .to_owned(),
                        ))
                    }
                };
                // Non-persistent, so the statistics never enter `state_dict` and a checkpoint
                // can never disagree with the IR about them.
                for (suffix, values) in [("a", first), ("b", second)] {
                    self.members.push(format!(
                        "        self.register_buffer(\"n{k}_{suffix}\", torch.tensor({}, dtype=torch.float32), persistent=False)",
                        json_floats(values)
                    ));
                }
                let x = &args[0];
                let (p, q) = (format!("self.n{k}_a"), format!("self.n{k}_b"));
                Ok(match (stats, direction) {
                    (StatsSource::MeanStd { .. }, NormalizeDir::Forward) => {
                        format!("({x} - {p}) / {q}")
                    }
                    (StatsSource::MeanStd { .. }, NormalizeDir::Inverse) => {
                        format!("{x} * {q} + {p}")
                    }
                    (_, NormalizeDir::Forward) => {
                        format!("2.0 * ({x} - {p}) / ({q} - {p}) - 1.0")
                    }
                    (_, NormalizeDir::Inverse) => {
                        format!("({x} + 1.0) * 0.5 * ({q} - {p}) + {p}")
                    }
                })
            }
        }
    }

    fn finish(self, returns: &[String]) -> TorchModule {
        let mut source = String::new();
        source.push_str(
            "# Generated by es-policy from a LearningGraph (spec 8.7). Do not edit.\n\
             # The contract this file implements: docs/design/learning-lowering.md\n\
             import torch\n\
             import torch.nn as nn\n",
        );
        if self.needs_torchvision {
            source.push_str(
                "\n\n\
                 def _backbone(name, out_dim):\n\
                 \x20   # Referenced, never re-implemented: torchvision is the provider.\n\
                 \x20   import torchvision\n\
                 \x20   m = getattr(torchvision.models, name)()\n\
                 \x20   m.fc = nn.Linear(m.fc.in_features, out_dim)\n\
                 \x20   return m\n",
            );
        }
        source.push_str(
            "\n\nclass EsPolicy(nn.Module):\n    def __init__(self):\n        super().__init__()\n",
        );
        for m in &self.members {
            let _ = writeln!(source, "{m}");
        }
        source.push_str("\n    def forward(self, **inputs):\n");
        for line in &self.body {
            let _ = writeln!(source, "{line}");
        }
        let _ = writeln!(source, "        return {{{}}}", returns.join(", "));

        let mut h = blake3::Hasher::new();
        h.update(LOWERING_TAG.as_bytes());
        h.update(source.as_bytes());
        TorchModule {
            lowering_hash: *h.finalize().as_bytes(),
            source,
            weight_keys: self.keys,
            weight_shapes: self.shapes,
        }
    }
}

/// A Python float list. `serde_json` renders f64 shortest-round-trip, which is both exact and
/// stable, so two lowerings of the same statistics are byte-identical.
fn json_floats(values: &[f64]) -> String {
    serde_json::to_string(values).unwrap_or_else(|_| "[]".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_ir::learning::testing::act_like;

    fn act() -> LearningGraph {
        act_like(8, 512, 8, 50, 20, 1)
    }

    #[test]
    fn lowering_is_byte_identical_across_runs() {
        let g = act();
        let a = lower_to_torch(&g).unwrap();
        let b = lower_to_torch(&g.clone()).unwrap();
        assert_eq!(a.source, b.source);
        assert_eq!(a.lowering_hash, b.lowering_hash);
        assert_eq!(a.weight_keys, b.weight_keys);
        assert_eq!(a.weight_shapes, b.weight_shapes);
    }

    #[test]
    fn a_different_architecture_moves_the_lowering_hash() {
        let a = lower_to_torch(&act()).unwrap();
        let b = lower_to_torch(&act_like(8, 256, 8, 50, 20, 1)).unwrap();
        assert_ne!(a.lowering_hash, b.lowering_hash);
    }

    #[test]
    fn the_act_fixture_lowers_one_forward_line_per_node() {
        let g = act();
        let m = lower_to_torch(&g).unwrap();

        let body: Vec<&str> = m
            .source
            .split("def forward(self, **inputs):\n")
            .nth(1)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("return"))
            .collect();
        assert_eq!(body.len(), g.nodes.nodes.len(), "{}", m.source);

        for expected in [
            "_backbone(\"resnet18\", 512)",
            "nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 512))",
            "nn.Linear(1024, 512)",
            "nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8",
            "nn.Linear(512, 400)",
        ] {
            assert!(m.source.contains(expected), "missing `{expected}`");
        }
        // Boundary inputs by name, the chunker's slice, and the declared output name.
        assert!(m.source.contains("inputs[\"rgb_front\"]"));
        assert!(m.source.contains("inputs[\"joint_state\"]"));
        assert!(m.source.contains("[:20]"));
        assert!(m.source.contains("return {\"actions\": v5_actions}"));
        assert!(m.source.contains("import torchvision"));
    }

    #[test]
    fn the_act_fixture_declares_the_documented_keys() {
        let m = lower_to_torch(&act()).unwrap();
        assert_eq!(
            m.weight_keys,
            vec![
                "nodes.0.*",
                "nodes.1.0.weight",
                "nodes.1.0.bias",
                "nodes.1.2.weight",
                "nodes.1.2.bias",
                "nodes.2.weight",
                "nodes.2.bias",
                "nodes.3.*",
                "nodes.4.weight",
                "nodes.4.bias",
            ]
        );
        assert_eq!(m.weight_shapes["nodes.1.0.weight"], vec![256, 8]);
        assert_eq!(m.weight_shapes["nodes.2.weight"], vec![512, 1024]);
        assert_eq!(m.weight_shapes["nodes.4.weight"], vec![400, 512]);
        assert_eq!(m.weight_shapes["nodes.4.bias"], vec![400]);
        // Prefix claims carry no shape.
        assert!(!m.weight_shapes.contains_key("nodes.0.*"));
    }

    #[test]
    fn deferred_heads_are_unsupported_not_silently_wrong() {
        use es_ir::learning::{ArchKind, DiffusionScheduler};
        let mut g = act();
        g.policy.architecture = ArchKind::Diffusion;
        let LearningNode::PolicyHead { kind, .. } = g.nodes.nodes.get_mut(&NodeId(4)).unwrap()
        else {
            unreachable!()
        };
        *kind = HeadKind::Diffusion {
            n_steps: 100,
            scheduler: DiffusionScheduler::Ddpm,
        };
        let err = lower_to_torch(&g).unwrap_err();
        assert!(
            matches!(&err, LowerError::Unsupported(k) if k.starts_with("PolicyHead{Diffusion")),
            "{err}"
        );
    }

    #[test]
    fn an_invalid_graph_never_lowers() {
        let mut g = act();
        g.policy.contract.execute_chunk = 80; // LRN-020: K > H
        assert!(matches!(
            lower_to_torch(&g).unwrap_err(),
            LowerError::Invalid(_)
        ));
    }
}
