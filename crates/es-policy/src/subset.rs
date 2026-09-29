//! `es policy subset` (packet M15/N8, `docs/design/multi-camera.md` section 3.4): a bundle that
//! reads fewer cameras than the one it came from, with **the same weights**.
//!
//! With a shared encoder and a `Sum` fusion (spec 8.3), dropping a camera is a graph edit that
//! keeps every weight: its `ImageInput` chain, its encoder and its term of the sum go, and the
//! remaining terms keep their meaning. The result is a new bundle with new hashes -- it *is* a
//! different deployable (spec 5.3) -- whose output is the parent's with the dropped terms left
//! out of the sum, i.e. the parent's `forward(keep_views=...)` (packet M15/N7).
//!
//! # Which graphs
//!
//! A *view* is a Learning IR input that feeds a `VisionEncoder`. Refused, by name:
//!
//! - a `--views` name that is not a view of the bundle (a joint-state input is not one);
//! - no view kept;
//! - a graph whose views do not meet in a `Sum` fusion ([`sum_views`] is empty): a `Concat`,
//!   `CrossAttention` or `FiLM` fusion was trained on every term, and a missing one changes what
//!   it computes;
//! - a dropped view whose encoder reaches the policy other than through **one** edge into a
//!   `Sum` -- a second consumer, a `Concat`, a graph output. Kept views are not constrained: a
//!   graph may sum two cameras and concatenate a third, and then only the summed two can go;
//! - a `Sum` left with none of its terms.
//!
//! Allowed whether or not the dropped view's encoder shares weights: an unshared encoder whose
//! only path is its `Sum` term goes with its tensors, which nothing else reads.
//!
//! # The rewrite
//!
//! - Observation IR: the dropped outputs, every node only they read (their `ImageInput` chains;
//!   a node a kept output also reads stays), and the history of a sensor no node reads any more.
//! - Learning IR: the dropped inputs, their contract inputs, their encoders and their `Sum`
//!   ports. A `Sum` left with two or more terms stays a `Sum` (`LRN-032` holds: the kept ports
//!   are unchanged). One left with a single term is removed and that term wired to its
//!   consumers: a `Fusion` takes at least two inputs (`LRN-002`), and the sum of one term is
//!   the term -- `torch.stack([t]).sum(0)` is `t`.
//! - `share`: when a group's owner is dropped and kept encoders shared it, the kept sharer with
//!   the lowest node id becomes the owner (its `share` removed, the others pointing at it) and
//!   the owner's tensors move to its `nodes.<id>.*` names. Every choice computes the same thing,
//!   one module applied to each view, so the rule is simply one a reader can check.
//! - Weights: each kept tensor's bytes are copied verbatim, never decoded, under its old name or
//!   the new owner's; a dropped node's tensors go. The parent's file must fit the parent's
//!   lowering and the result the subset's (`validate_keys`, as `TorchRuntime::load` asks).
//! - Provenance: the weights' safetensors `__metadata__` gains [`SUBSET_OF_KEY`] (the parent's
//!   `policy_hash`) and [`SUBSET_VIEWS_KEY`] (the kept views), beside where
//!   `lerobot::ACT_CONFIG_KEY` already travels: inside the bytes `WeightsRef::hash`, and so
//!   `policy_hash`, cover. The manifest is not in the chain, and no existing hash moves.
//!
//! The Task and Deployment IR are carried over unchanged -- an Observation IR may read a subset
//! of the Task IR's channels. An evaluation entry is not: it names the parent's observation.

use std::collections::{BTreeMap, BTreeSet};

use es_compile::PolicyBundle;
use es_core::StableId;
use es_ir::graph::{IrNode, NodeId, PortRef};
use es_ir::learning::{FusionKind, LearningGraph, LearningNode, WeightsRef};
use es_ir::observation::{ObservationIr, ObservationNode};

use crate::lower::{lower_to_torch, sum_views};
use crate::runtime::PolicyError;
use crate::weights::{header_of, hex, parse_header, validate_keys, weights_hash, WEIGHT_PREFIX};

/// `__metadata__` key: the parent bundle's `policy_hash`, hex.
pub const SUBSET_OF_KEY: &str = "es.subset.of";
/// `__metadata__` key: the views the subset keeps, comma-separated, in declared order.
pub const SUBSET_VIEWS_KEY: &str = "es.subset.views";

#[derive(Debug, thiserror::Error)]
pub enum SubsetError {
    #[error("the bundle has no camera view \"{view}\"; its views are {views:?}")]
    UnknownView { view: String, views: Vec<String> },
    #[error("no view kept: name at least one of the bundle's camera views")]
    NoViewKept,
    #[error(
        "the graph's camera views do not meet in a Sum fusion: dropping one would change what \
         its fusion was trained on (spec 8.3)"
    )]
    NoSum,
    #[error(
        "view \"{view}\" reaches the policy through {through}, not as one term of a Sum \
         fusion: dropping it would change what that node was trained on"
    )]
    NotASumTerm { view: String, through: String },
    #[error("{0}")]
    Unsupported(String),
    #[error(transparent)]
    Weights(#[from] PolicyError),
    #[error("the subset does not make a bundle: {0}")]
    Bundle(String),
}

/// A written subset and what it did.
#[derive(Debug)]
pub struct Subset {
    /// The new `policy.esb`.
    pub bytes: Vec<u8>,
    /// The parent's `policy_hash`, as the weights' [`SUBSET_OF_KEY`] records it.
    pub parent: [u8; 32],
    /// The kept views, in declared order.
    pub views: Vec<String>,
    /// Encoder node ids removed from the Learning IR.
    pub dropped: BTreeSet<u32>,
    /// Dropped owner -> the kept encoder that now owns its tensors.
    pub rehomed: BTreeMap<u32, u32>,
}

/// Every view of `g` once, in declared order: an input that feeds a `VisionEncoder`.
pub fn views(g: &LearningGraph) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (port, at) in g.inputs.iter().zip(&g.nodes.inputs) {
        let encoder = matches!(
            g.nodes.nodes.get(&at.node),
            Some(LearningNode::VisionEncoder { .. })
        );
        if encoder && !out.contains(&port.name) {
            out.push(port.name.clone());
        }
    }
    out
}

fn describe(id: NodeId, node: &LearningNode) -> String {
    match node {
        LearningNode::Fusion { kind, .. } => format!("node {} (Fusion {kind:?})", id.0),
        other => format!("node {} ({})", id.0, other.kind()),
    }
}

/// The bundle `bundle` becomes when it reads only the views in `keep` (see the module docs).
pub fn subset(bundle: &PolicyBundle, keep: &[String]) -> Result<Subset, SubsetError> {
    let g = &bundle.learning;
    let all = views(g);
    if let Some(view) = keep.iter().find(|v| !all.contains(v)) {
        return Err(SubsetError::UnknownView {
            view: view.clone(),
            views: all,
        });
    }
    if keep.is_empty() {
        return Err(SubsetError::NoViewKept);
    }
    if sum_views(g).is_empty() {
        return Err(SubsetError::NoSum);
    }
    let WeightsRef::Safetensors { path, .. } = &g.policy.weights else {
        return Err(SubsetError::Unsupported(format!(
            "the weights are {:?}; a subset rewrites safetensors only (INV-16)",
            g.policy.weights
        )));
    };
    // The parent fits its own lowering, so no sharer holds a tensor a re-homing could collide
    // with, and every tensor below is one the parent's module really loads.
    validate_keys(
        &lower_to_torch(g).map_err(PolicyError::from)?,
        &parse_header(&bundle.weights)?,
    )?;

    let dropped: BTreeSet<&str> = all
        .iter()
        .filter(|v| !keep.contains(v))
        .map(String::as_str)
        .collect();
    let mut encoders: BTreeSet<NodeId> = BTreeSet::new();
    let mut terms: Vec<PortRef> = Vec::new();
    for (port, at) in g.inputs.iter().zip(&g.nodes.inputs) {
        if !dropped.contains(port.name.as_str()) {
            continue;
        }
        let refuse = |through: String| SubsetError::NotASumTerm {
            view: port.name.clone(),
            through,
        };
        // Every node named here exists: the bundle opened, so the graph validated.
        let node = &g.nodes.nodes[&at.node];
        if !matches!(node, LearningNode::VisionEncoder { .. }) {
            return Err(refuse(describe(at.node, node)));
        }
        let out: Vec<_> = g
            .nodes
            .edges
            .iter()
            .filter(|e| e.from.node == at.node)
            .collect();
        let is_output = g.nodes.outputs.iter().any(|o| o.node == at.node);
        match out.as_slice() {
            [e] if !is_output
                && matches!(
                    g.nodes.nodes.get(&e.to.node),
                    Some(LearningNode::Fusion {
                        kind: FusionKind::Sum,
                        ..
                    })
                ) =>
            {
                encoders.insert(at.node);
                terms.push(e.to.clone());
            }
            _ => {
                let mut through: Vec<String> = out
                    .iter()
                    .map(|e| describe(e.to.node, &g.nodes.nodes[&e.to.node]))
                    .collect();
                if is_output {
                    through.push("a graph output".to_owned());
                }
                if through.is_empty() {
                    through.push("nothing".to_owned());
                }
                return Err(refuse(through.join(" and ")));
            }
        }
    }

    let mut learning = g.clone();
    // Re-home a dropped owner's group on its lowest-id kept sharer. A dropped sharer has no
    // sharers of its own (`share` does not chain, `LRN-033`), so it finds none here.
    let mut rehomed = BTreeMap::new();
    for owner in &encoders {
        let sharers: Vec<NodeId> = g
            .nodes
            .nodes
            .iter()
            .filter(|(id, n)| {
                !encoders.contains(id)
                    && matches!(n, LearningNode::VisionEncoder { share: Some(o), .. } if o == owner)
            })
            .map(|(id, _)| *id)
            .collect();
        let Some(&new) = sharers.first() else {
            continue;
        };
        rehomed.insert(owner.0, new.0);
        for id in sharers {
            if let Some(LearningNode::VisionEncoder { share, .. }) =
                learning.nodes.nodes.get_mut(&id)
            {
                *share = (id != new).then_some(new);
            }
        }
    }

    let (inputs, refs): (Vec<_>, Vec<_>) = g
        .inputs
        .iter()
        .cloned()
        .zip(g.nodes.inputs.iter().cloned())
        .filter(|(p, _)| !dropped.contains(p.name.as_str()))
        .unzip();
    learning.inputs = inputs;
    learning.nodes.inputs = refs;
    learning
        .policy
        .contract
        .inputs
        .retain(|name, _| !dropped.contains(name.as_str()));
    learning.nodes.nodes.retain(|id, _| !encoders.contains(id));
    learning
        .nodes
        .edges
        .retain(|e| !encoders.contains(&e.from.node) && !encoders.contains(&e.to.node));
    for term in &terms {
        if let Some(LearningNode::Fusion { inputs, .. }) = learning.nodes.nodes.get_mut(&term.node)
        {
            inputs.retain(|p| p.name != term.port);
        }
    }
    let sums: BTreeSet<NodeId> = terms.iter().map(|t| t.node).collect();
    for sum in sums {
        let left: Vec<String> = learning.nodes.nodes[&sum]
            .inputs()
            .into_iter()
            .map(|p| p.name)
            .collect();
        match left.as_slice() {
            [] => {
                return Err(SubsetError::Unsupported(format!(
                    "Sum fusion node {} would keep none of its terms: keep at least one of its \
                     views",
                    sum.0
                )))
            }
            [only] => collapse(&mut learning, sum, only)?,
            _ => {}
        }
    }

    let observation = subset_observation(&bundle.observation, &dropped);
    let parent = g
        .policy_hash()
        .map_err(|d| SubsetError::Bundle(d.to_string()))?;
    let kept = views(&learning);
    let weights = rewrite_weights(
        &bundle.weights,
        &encoders,
        &rehomed,
        [
            (SUBSET_OF_KEY, hex(&parent)),
            (SUBSET_VIEWS_KEY, kept.join(",")),
        ],
    )?;
    learning.policy.weights = WeightsRef::Safetensors {
        path: path.clone(),
        hash: weights_hash(&weights),
    };
    validate_keys(
        &lower_to_torch(&learning).map_err(PolicyError::from)?,
        &parse_header(&weights)?,
    )?;
    let bytes = PolicyBundle::build(
        &bundle.task,
        &observation,
        &learning,
        &bundle.deployment,
        &weights,
    )
    .map_err(|e| SubsetError::Bundle(e.to_string()))?;
    Ok(Subset {
        bytes,
        parent,
        views: kept,
        dropped: encoders.iter().map(|n| n.0).collect(),
        rehomed,
    })
}

/// A `Sum` left with one term is that term: its feeder takes over the sum's consumers.
fn collapse(g: &mut LearningGraph, sum: NodeId, port: &str) -> Result<(), SubsetError> {
    let Some(at) = g
        .nodes
        .edges
        .iter()
        .position(|e| e.to.node == sum && e.to.port == port)
    else {
        return Err(SubsetError::Unsupported(format!(
            "Sum fusion node {}'s one remaining term \"{port}\" is a graph input; a subset \
             rewires only a term another node computes",
            sum.0
        )));
    };
    let from = g.nodes.edges.remove(at).from;
    for e in &mut g.nodes.edges {
        if e.from.node == sum {
            e.from = from.clone();
        }
    }
    for o in &mut g.nodes.outputs {
        if o.node == sum {
            *o = from.clone();
        }
    }
    g.nodes.nodes.remove(&sum);
    Ok(())
}

/// The Observation IR without the `dropped` outputs and the nodes only they read.
fn subset_observation(o: &ObservationIr, dropped: &BTreeSet<&str>) -> ObservationIr {
    let upstream = |of_dropped: bool| {
        let mut seen = BTreeSet::new();
        let mut stack: Vec<NodeId> = o
            .outputs
            .iter()
            .filter(|(name, _)| dropped.contains(name.as_str()) == of_dropped)
            .map(|(_, out)| out.port.node)
            .collect();
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                stack.extend(
                    o.graph
                        .edges
                        .iter()
                        .filter(|e| e.to.node == id)
                        .map(|e| e.from.node),
                );
            }
        }
        seen
    };
    let kept = upstream(false);
    let gone: BTreeSet<NodeId> = upstream(true).difference(&kept).copied().collect();
    let sensor = |n: &ObservationNode| match n {
        ObservationNode::ImageInput { sensor, .. } => Some(*sensor),
        _ => None,
    };
    let unread: BTreeSet<StableId> = gone
        .iter()
        .filter_map(|id| o.graph.nodes.get(id).and_then(sensor))
        .collect();

    let mut out = o.clone();
    out.outputs
        .retain(|name, _| !dropped.contains(name.as_str()));
    out.graph.nodes.retain(|id, _| !gone.contains(id));
    out.graph
        .edges
        .retain(|e| !gone.contains(&e.from.node) && !gone.contains(&e.to.node));
    out.graph.inputs.retain(|r| !gone.contains(&r.node));
    out.graph.outputs.retain(|r| !gone.contains(&r.node));
    let read: BTreeSet<StableId> = out.graph.nodes.values().filter_map(sensor).collect();
    out.temporal
        .history
        .retain(|id, _| !unread.contains(id) || read.contains(id));
    out
}

/// The checkpoint without the dropped encoders' tensors, a re-homed owner's under its new
/// owner's id, and `provenance` added to `__metadata__`. Bytes are copied, never decoded.
fn rewrite_weights(
    bytes: &[u8],
    dropped: &BTreeSet<NodeId>,
    rehomed: &BTreeMap<u32, u32>,
    provenance: [(&str, String); 2],
) -> Result<Vec<u8>, PolicyError> {
    let (raw, data_len) = header_of(bytes)?;
    let base = bytes.len() - data_len as usize;
    let mut kept = BTreeMap::new();
    for (name, entry) in parse_header(bytes)? {
        let node = name
            .strip_prefix(WEIGHT_PREFIX)
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(id, rest)| Some((id.parse::<u32>().ok()?, rest)));
        let renamed = match node {
            Some((id, rest)) if dropped.contains(&NodeId(id)) => match rehomed.get(&id) {
                Some(to) => Some(format!("{WEIGHT_PREFIX}{to}.{rest}")),
                None => continue,
            },
            _ => None,
        };
        let (a, b) = entry.offsets;
        // In range: `parse_header` checked every offset against the data segment.
        let slice = &bytes[base + a as usize..base + b as usize];
        kept.insert(renamed.unwrap_or(name), (entry, slice));
    }

    let mut meta = raw
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (key, value) in provenance {
        meta.insert(key.to_owned(), value.into());
    }
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::Value::Object(meta));
    let mut data = Vec::with_capacity(bytes.len());
    for (name, (entry, slice)) in kept {
        let start = data.len();
        data.extend_from_slice(slice);
        header.insert(
            name,
            serde_json::json!({
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| PolicyError::Safetensors(e.to_string()))?;
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    Ok(out)
}
