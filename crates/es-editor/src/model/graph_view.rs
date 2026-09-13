//! The read-only layered graph view of spec 23.2, as a headless view-model.
//!
//! Spec 23.2 stacks the IRs on one screen — Task above Observation above Learning above the
//! action/safety plane — and draws the cross-IR joins between them. Everything that decides
//! *what* is drawn lives here, with no egui in sight, so it is testable in CI without a
//! display; `crate::app` only turns positions into rectangles.
//!
//! Two rules shape this module:
//!
//! * spec 4.2 rule 7 / `INV`: layout is **not** IR. Positions arrive from an `.eslayout`
//!   sidecar ([`es_ir::serial::Layout`]) or from [`LayeredGraph::auto_layout`]; nothing here
//!   ever writes a position back into an IR.
//! * the cross-IR edges mirror [`es_ir::cross`] — the same joins its five boundary checks
//!   make, drawn instead of diagnosed. The rules live there; this is the picture of them.

use std::collections::BTreeMap;

use es_core::id::StableId;
use es_ir::cross::{self, IrBundle};
use es_ir::deployment::DeploymentIr;
use es_ir::graph::{Edge, Graph, IrNode, NodeId, PortRef};
use es_ir::learning::LearningGraph;
use es_ir::observation::{ObservationIr, ObservationNode};
use es_ir::serial::{IrKind, Layout};
use es_ir::task::{ObsSource, TaskIr, TaskNode};
use es_ir::Diagnostic;

/// Layer index of each IR in [`LayeredGraph::layers`], in the spec 23.2 stacking order.
pub const TASK: usize = 0;
pub const OBSERVATION: usize = 1;
pub const LEARNING: usize = 2;
pub const DEPLOYMENT: usize = 3;

/// Horizontal distance between two topological ranks.
const DX: f32 = 220.0;
/// Vertical distance between two nodes of the same rank.
const DY: f32 = 70.0;
/// Blank space between two layer bands.
const BAND_GAP: f32 = 60.0;

/// The ports of one node, by name. Types stay in the IR: the view shows the wiring.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ports {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeView {
    pub id: NodeId,
    /// The IR's own stable kind tag (`IrNode::kind`), or a synthetic one for the
    /// Deployment IR, which has no graph.
    pub kind: &'static str,
    pub label: String,
    pub ports: Ports,
    /// `None` until [`LayeredGraph::auto_layout`] or [`LayeredGraph::apply_layout`] runs.
    pub layout: Option<[f32; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerView {
    pub kind: IrKind,
    pub nodes: Vec<NodeView>,
    pub edges: Vec<Edge>,
}

impl LayerView {
    fn node(&self, id: NodeId) -> Option<&NodeView> {
        self.nodes.iter().find(|n| n.id == id)
    }

    fn index_of(&self, id: NodeId) -> Option<usize> {
        self.nodes.iter().position(|n| n.id == id)
    }
}

/// One endpoint of a cross-IR edge: which layer, which node in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrossEnd {
    pub layer: usize,
    pub node: NodeId,
}

/// An edge between two IRs — the joins `es_ir::cross` checks, drawn (spec 23.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossEdge {
    pub from: CrossEnd,
    pub to: CrossEnd,
    /// The channel or tensor name the two sides agree on.
    pub label: String,
}

/// The whole stack: four layers, the edges between them, and everything the validators said.
#[derive(Clone, Debug, PartialEq)]
pub struct LayeredGraph {
    pub layers: Vec<LayerView>,
    pub cross_edges: Vec<CrossEdge>,
    pub diagnostics: Vec<Diagnostic>,
}

impl LayeredGraph {
    /// Build the view from four IRs. Never fails: a bundle that does not validate is exactly
    /// the one a user opens the editor to look at, so every complaint becomes a
    /// [`Diagnostic`] in `diagnostics` instead of an error return.
    pub fn from_bundle(
        task: &TaskIr,
        observation: &ObservationIr,
        learning: &LearningGraph,
        deployment: &DeploymentIr,
    ) -> Self {
        let layers = vec![
            layer_of(IrKind::Task, &task.graph),
            layer_of(IrKind::Observation, &observation.graph),
            layer_of(IrKind::Learning, &learning.nodes),
            deployment_layer(deployment),
        ];
        let mut diagnostics = task.validate();
        diagnostics.extend(observation.validate());
        diagnostics.extend(learning.validate());
        diagnostics.extend(deployment.validate());
        diagnostics.extend(cross::check(&IrBundle {
            task,
            observation,
            learning,
            deployment,
            evaluation: None,
        }));
        let cross_edges = cross_edges(task, observation, learning, &layers);
        Self {
            layers,
            cross_edges,
            diagnostics,
        }
    }

    /// Deterministic layered layout (Sugiyama without the crossing-reduction pass): a node's
    /// rank is its longest path from a source, rank fixes `x`, and position within the rank —
    /// taken in the graph's own canonical `NodeId` order — fixes `y`. Each layer gets its own
    /// horizontal band, so two IRs never overlap.
    ///
    /// Crossing reduction is the part that would make this *pretty*; it is also the part that
    /// needs a stable tie-break to stay deterministic. Read-only debugging does not need it
    /// (spec 23.4), so it is not here.
    pub fn auto_layout(&mut self) {
        let mut band_top = 0.0f32;
        for layer in &mut self.layers {
            let ranks = ranks_of(&layer.nodes, &layer.edges);
            // Nodes are already in canonical `NodeId` order, so `row` counts them in it.
            let mut row: BTreeMap<usize, usize> = BTreeMap::new();
            let mut height = 0.0f32;
            for node in &mut layer.nodes {
                let rank = ranks.get(&node.id).copied().unwrap_or(0);
                let r = row.entry(rank).or_default();
                let y = band_top + *r as f32 * DY;
                node.layout = Some([rank as f32 * DX, y]);
                *r += 1;
                height = height.max(y - band_top + DY);
            }
            band_top += height + BAND_GAP;
        }
    }

    /// Override positions from an `.eslayout` sidecar (spec 14.3). Nodes the sidecar does not
    /// mention keep what they have, so this composes with [`auto_layout`](Self::auto_layout).
    ///
    /// A sidecar is keyed by [`NodeId`] alone, which the four IRs do not share a namespace
    /// for; a position therefore applies to that id in *every* layer. Per-IR sidecars are a
    /// question for the editable graph of spec 23.4 stage 2, which is where saving layout
    /// starts to matter.
    pub fn apply_layout(&mut self, layout: &Layout) {
        for layer in &mut self.layers {
            for node in &mut layer.nodes {
                if let Some(pos) = layout.positions.get(&node.id) {
                    node.layout = Some(*pos);
                }
            }
        }
    }

    /// The position of one endpoint, once a layout has run.
    pub fn position(&self, end: CrossEnd) -> Option<[f32; 2]> {
        self.layers.get(end.layer)?.node(end.node)?.layout
    }
}

fn layer_of<N: IrNode>(kind: IrKind, graph: &Graph<N>) -> LayerView {
    LayerView {
        kind,
        nodes: graph
            .nodes
            .iter()
            .map(|(id, node)| NodeView {
                id: *id,
                kind: node.kind(),
                label: format!("{} #{}", node.kind(), id.0),
                ports: Ports {
                    inputs: node.inputs().into_iter().map(|p| p.name).collect(),
                    outputs: node.outputs().into_iter().map(|p| p.name).collect(),
                },
                layout: None,
            })
            .collect(),
        edges: graph.edges.clone(),
    }
}

/// Deployment IR is a record, not a graph (spec 9.2), but spec 23.2 draws it as the action and
/// safety bands of the same picture. These four nodes are that picture, in execution order.
fn deployment_layer(ir: &DeploymentIr) -> LayerView {
    let node =
        |n: u32, kind: &'static str, label: String, inputs: &[&str], outputs: &[&str]| NodeView {
            id: NodeId(n),
            kind,
            label,
            ports: Ports {
                inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
                outputs: outputs.iter().map(|s| (*s).to_owned()).collect(),
            },
            layout: None,
        };
    let nodes = vec![
        node(
            0,
            "ActionContract",
            format!(
                "{:?} dim {} chunk {}/{}",
                ir.action.space, ir.action.dim, ir.action.execute_chunk, ir.action.horizon
            ),
            &["actions"],
            &["action"],
        ),
        node(
            1,
            "SafetyEnvelope",
            format!("Envelope {} joints", ir.safety.n_joints()),
            &["action"],
            &["safe_action"],
        ),
        node(
            2,
            "Watchdogs",
            format!("{} watchdogs", ir.watchdogs.0.len()),
            &["safe_action"],
            &["safe_action"],
        ),
        node(
            3,
            "Fallback",
            format!("{:?}", ir.fallback),
            &["safe_action"],
            &[],
        ),
    ];
    let edges = vec![
        edge(NodeId(0), "action", NodeId(1), "action"),
        edge(NodeId(1), "safe_action", NodeId(2), "safe_action"),
        edge(NodeId(2), "safe_action", NodeId(3), "safe_action"),
    ];
    LayerView {
        kind: IrKind::Deployment,
        nodes,
        edges,
    }
}

fn edge(from: NodeId, from_port: &str, to: NodeId, to_port: &str) -> Edge {
    Edge {
        from: PortRef::new(from, from_port),
        to: PortRef::new(to, to_port),
    }
}

// --- cross-IR edges: the joins of `es_ir::cross`, drawn ---------------------------------------

/// What makes a declared channel and a source node the same channel — the pairing
/// `es_ir::cross::task_observation` checks (spec 7.4). Mirrored, not re-derived: the rules
/// stay there, this only needs to know which two nodes they put on the same wire.
type SourceKey = (&'static str, Option<StableId>);

fn channel_key(src: &ObsSource) -> SourceKey {
    match src {
        ObsSource::Sensor { id, .. } => ("sensor", Some(*id)),
        ObsSource::JointState { body, .. } | ObsSource::BodyPose(body) => ("state", Some(*body)),
        ObsSource::Language => ("language", None),
    }
}

fn node_key(node: &ObservationNode) -> Option<SourceKey> {
    match node {
        ObservationNode::ImageInput { sensor, .. } => Some(("sensor", Some(*sensor))),
        ObservationNode::StateInput { source, .. } => Some(("state", Some(*source))),
        ObservationNode::LanguageInput { .. } => Some(("language", None)),
        _ => None,
    }
}

fn cross_edges(
    task: &TaskIr,
    observation: &ObservationIr,
    learning: &LearningGraph,
    layers: &[LayerView],
) -> Vec<CrossEdge> {
    let mut out = Vec::new();

    // 1. Task `ObservationSpec` channel -> the Observation source node implementing it.
    for (obs_id, node) in &observation.graph.nodes {
        let Some(key) = node_key(node) else { continue };
        let Some((name, _)) = task
            .observation_spec
            .channels
            .iter()
            .find(|(_, ch)| channel_key(&ch.source) == key)
        else {
            continue;
        };
        let declaring = task.graph.nodes.iter().find(
            |(_, n)| matches!(n, TaskNode::ObservationSpec { channel, .. } if channel == name),
        );
        if let Some((task_id, _)) = declaring {
            out.push(CrossEdge {
                from: CrossEnd {
                    layer: TASK,
                    node: *task_id,
                },
                to: CrossEnd {
                    layer: OBSERVATION,
                    node: *obs_id,
                },
                label: name.clone(),
            });
        }
    }

    // 2. Observation outputs -> the policy contract inputs of the same name (spec 8.4).
    for name in learning.policy.contract.inputs.keys() {
        let Some(produced) = observation.outputs.get(name) else {
            continue;
        };
        let Some(consumer) = learning.nodes.inputs.iter().find(|p| &p.port == name) else {
            continue;
        };
        out.push(CrossEdge {
            from: CrossEnd {
                layer: OBSERVATION,
                node: produced.port.node,
            },
            to: CrossEnd {
                layer: LEARNING,
                node: consumer.node,
            },
            label: name.clone(),
        });
    }

    // 3. The Learning action output -> the deployed action contract (spec 8.5, spec 9.2).
    for port in &learning.nodes.outputs {
        out.push(CrossEdge {
            from: CrossEnd {
                layer: LEARNING,
                node: port.node,
            },
            to: CrossEnd {
                layer: DEPLOYMENT,
                node: NodeId(0),
            },
            label: port.port.clone(),
        });
    }

    out.retain(|e| {
        layers[e.from.layer].index_of(e.from.node).is_some()
            && layers[e.to.layer].index_of(e.to.node).is_some()
    });
    out
}

/// Longest-path rank per node: `rank(n) = 1 + max(rank(pred))`, `0` for a source. Computed on
/// the node order as given (already canonical), relaxed until it settles, so a cycle — which
/// an IR validator rejects but the editor still has to draw — terminates instead of looping.
fn ranks_of(nodes: &[NodeView], edges: &[Edge]) -> BTreeMap<NodeId, usize> {
    let mut rank: BTreeMap<NodeId, usize> = nodes.iter().map(|n| (n.id, 0)).collect();
    for _ in 0..nodes.len() {
        let mut changed = false;
        for e in edges {
            let Some(from) = rank.get(&e.from.node).copied() else {
                continue;
            };
            if let Some(to) = rank.get_mut(&e.to.node) {
                if *to < from + 1 {
                    *to = from + 1;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    rank
}
