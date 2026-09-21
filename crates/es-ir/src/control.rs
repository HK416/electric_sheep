//! IR-C: the Task IR control graph (spec 6.2, M4).
//!
//! IR-D ([`crate::task`]) is a pure dataflow DAG. IR-C is the sequencing layer on top of it:
//! `Sequence`, `Branch`, `SubTask`, `Repeat` and nothing else — spec 6.3 is explicit that
//! `Parallel` does not exist and that `Wait` / `Condition` are expressible with what is here.
//!
//! **Schema only.** The state machine that advances a control graph lives in `es-env`
//! (`es_env::control`), the same way reward-cone lowering does; this module owns the types,
//! the validation and the hash. See `docs/design/control-graph.md`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::codes;
use crate::diag::Diagnostic;
use crate::graph::{Graph, IrNode, NodeId, Port};
use crate::hash::{canonical_hash, CanonWriter};
use crate::task::{Expr, TaskIr};
use crate::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

/// A node of the control tree. The same id type IR-D uses, in its own namespace.
pub type ControlNodeId = NodeId;

/// When a [`ControlNode::Repeat`] stops re-entering its body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RepeatUntil {
    /// Exactly `n` iterations; `n > 0` (`CTRL-005`).
    Count(u32),
    /// Re-enter while the predicate is false. Bounded by the episode budget.
    Until(Expr),
}

/// One stage: a named slice of the parent IR-D graph.
///
/// Named, never by `NodeId`: `canon_task` relabels the IR-D graph, so an id stored here would
/// go stale and `task_hash` would stop being relabel-invariant (Appendix B.7 property 1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubTaskRef {
    /// Unique in the graph; the stage's identity.
    pub name: String,
    /// `TaskNode::Reward` names this stage scores.
    pub rewards: BTreeSet<String>,
    /// Observation channels this stage needs; a subset of `TaskIr::observation_spec`.
    pub observation: BTreeSet<String>,
    /// Stage completion predicate over IR-D ports. `None` completes on the first step.
    pub success: Option<Expr>,
    /// Redraw this env's reset distributions on stage entry (spec 6.3 `ResetState`).
    pub reset_on_entry: bool,
    /// This stage's factor in the episode reward sum.
    pub weight: f64,
}

/// The spec 6.2 IR-C node set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ControlNode {
    Sequence {
        children: Vec<ControlNodeId>,
    },
    Branch {
        /// Evaluated **once**, at entry (spec 6.6: the path may not drift with the trajectory).
        condition: Expr,
        then_: ControlNodeId,
        else_: ControlNodeId,
    },
    Repeat {
        body: ControlNodeId,
        until: RepeatUntil,
    },
    SubTask {
        task: SubTaskRef,
        /// Elapsing ends the **episode** with the failure flag; `> 0` (`CTRL-005`).
        timeout_ticks: u32,
    },
}

fn flag_ty() -> PortType {
    PortType {
        elem: ElemType::Bool,
        shape: Shape::new([1]),
        unit: Unit::Dimensionless,
        frame: Frame::World,
        time: TimeRef::Tick,
        image: None,
    }
}

/// Canonical encoding of an [`Expr`]: the shape of the tree plus every literal, no names
/// omitted. `w.f64` rejects a non-finite literal as `HASH-001`.
fn expr_canonical(e: &Expr, w: &mut CanonWriter) {
    match e {
        Expr::Port(name) => {
            w.str("Port");
            w.str(name);
        }
        Expr::Const(v) => {
            w.str("Const");
            w.f64(*v);
        }
        Expr::Arith { op, lhs, rhs } => {
            w.str("Arith");
            w.str(&format!("{op:?}"));
            expr_canonical(lhs, w);
            expr_canonical(rhs, w);
        }
        Expr::Compare { op, lhs, rhs } => {
            w.str("Compare");
            w.str(&format!("{op:?}"));
            expr_canonical(lhs, w);
            expr_canonical(rhs, w);
        }
        Expr::Clamp { value, lo, hi } => {
            w.str("Clamp");
            expr_canonical(value, w);
            w.f64(*lo);
            w.f64(*hi);
        }
        Expr::Sqrt(value) => {
            w.str("Sqrt");
            expr_canonical(value, w);
        }
    }
}

impl ControlNode {
    /// The children this node points at, in the order its output ports declare them.
    pub fn children(&self) -> Vec<ControlNodeId> {
        match self {
            Self::Sequence { children } => children.clone(),
            Self::Branch { then_, else_, .. } => vec![*then_, *else_],
            Self::Repeat { body, .. } => vec![*body],
            Self::SubTask { .. } => Vec::new(),
        }
    }
}

impl IrNode for ControlNode {
    fn kind(&self) -> &'static str {
        match self {
            Self::Sequence { .. } => "Sequence",
            Self::Branch { .. } => "Branch",
            Self::Repeat { .. } => "Repeat",
            Self::SubTask { .. } => "SubTask",
        }
    }

    fn inputs(&self) -> Vec<Port> {
        vec![Port::new("parent", flag_ty())]
    }

    /// Indexed port names, so a `Sequence` whose children are swapped is a different graph:
    /// [`canonical_hash`] reads structure off port names, never off node ids.
    fn outputs(&self) -> Vec<Port> {
        let names: Vec<String> = match self {
            Self::Sequence { children } => {
                (0..children.len()).map(|i| format!("child{i}")).collect()
            }
            Self::Branch { .. } => vec!["then".to_owned(), "else".to_owned()],
            Self::Repeat { .. } => vec!["body".to_owned()],
            Self::SubTask { .. } => Vec::new(),
        };
        names.into_iter().map(|n| Port::new(n, flag_ty())).collect()
    }

    /// The node's own parameters and **never** a child id — the edges carry the structure.
    fn params_canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::Sequence { children } => w.seq(children.len()),
            Self::Branch { condition, .. } => expr_canonical(condition, w),
            Self::Repeat { until, .. } => match until {
                RepeatUntil::Count(n) => {
                    w.str("Count");
                    w.u32(*n);
                }
                RepeatUntil::Until(e) => {
                    w.str("Until");
                    expr_canonical(e, w);
                }
            },
            Self::SubTask {
                task,
                timeout_ticks,
            } => {
                w.str(&task.name);
                w.seq(task.rewards.len());
                for r in &task.rewards {
                    w.str(r);
                }
                w.seq(task.observation.len());
                for c in &task.observation {
                    w.str(c);
                }
                match &task.success {
                    Some(e) => expr_canonical(e, w),
                    None => w.str("always"),
                }
                w.bool(task.reset_on_entry);
                w.f64(task.weight);
                w.u32(*timeout_ticks);
            }
        }
    }
}

/// A control tree: one root and the nodes reachable from it (spec 6.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlGraph {
    pub root: ControlNodeId,
    pub nodes: BTreeMap<ControlNodeId, ControlNode>,
}

impl ControlGraph {
    /// The tree as a `Graph<ControlNode>`, so [`canonical_hash`] and the rest of the IR-D
    /// machinery apply unchanged. Edges point parent -> child through the indexed port names
    /// [`ControlNode::outputs`] declares.
    pub fn as_graph(&self) -> Graph<ControlNode> {
        let mut g = Graph::new(crate::task::SCHEMA_VERSION);
        for (id, node) in &self.nodes {
            g.insert(*id, node.clone());
            for (port, child) in node.outputs().iter().zip(node.children()) {
                g.connect(*id, &port.name, child, "parent");
            }
        }
        g
    }

    /// Semantic identity of the control tree; part of `task_hash` (spec 11.2).
    pub fn control_hash(&self) -> Result<[u8; 32], Diagnostic> {
        canonical_hash(&self.as_graph())
    }

    /// Single existing root, tree shape, positive bounds, resolvable slice names, boolean
    /// conditions. `task` supplies the IR-D names a `SubTask` may slice.
    pub fn validate(&self, task: &TaskIr) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        let missing = |diags: &mut Vec<Diagnostic>, at: Option<NodeId>, id: NodeId| {
            let d = Diagnostic::new(
                codes::CTRL_001,
                format!("control node {} does not exist", id.0),
            );
            diags.push(match at {
                Some(a) => d.at(a),
                None => d.with_hint("the control graph root must be one of its nodes"),
            });
        };
        if !self.nodes.contains_key(&self.root) {
            missing(&mut diags, None, self.root);
            return diags;
        }

        // Depth-first walk from the root. A node reached twice is either a diamond (shared
        // child) or a cycle; the on-path set is what tells them apart.
        let mut seen: BTreeSet<NodeId> = BTreeSet::new();
        let mut path: BTreeSet<NodeId> = BTreeSet::new();
        let mut stack: Vec<(NodeId, bool)> = vec![(self.root, false)];
        while let Some((id, leaving)) = stack.pop() {
            if leaving {
                path.remove(&id);
                continue;
            }
            if path.contains(&id) {
                diags.push(
                    Diagnostic::new(codes::CTRL_004, "the control graph contains a cycle").at(id),
                );
                continue;
            }
            if !seen.insert(id) {
                diags.push(
                    Diagnostic::new(codes::CTRL_003, "control node has more than one parent")
                        .at(id)
                        .with_hint("IR-C is a tree; give the stage its own node"),
                );
                continue;
            }
            path.insert(id);
            stack.push((id, true));
            let Some(node) = self.nodes.get(&id) else {
                continue;
            };
            for child in node.children() {
                if self.nodes.contains_key(&child) {
                    stack.push((child, false));
                } else {
                    missing(&mut diags, Some(id), child);
                }
            }
        }
        for id in self.nodes.keys() {
            if !seen.contains(id) {
                diags.push(
                    Diagnostic::new(codes::CTRL_002, "control node is unreachable from the root")
                        .at(*id),
                );
            }
        }

        let rewards: BTreeSet<&str> = task
            .graph
            .nodes
            .values()
            .filter_map(|n| match n {
                crate::task::TaskNode::Reward { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let mut stages: BTreeSet<&str> = BTreeSet::new();
        for (id, node) in &self.nodes {
            let boolean = |diags: &mut Vec<Diagnostic>, e: &Expr, what: &str| {
                if !matches!(e, Expr::Compare { .. }) {
                    diags.push(
                        Diagnostic::new(
                            codes::TYPE_030,
                            format!("the {what} expression is not boolean"),
                        )
                        .at(*id)
                        .with_hint("a condition's root must be a Compare (spec 6.5)"),
                    );
                }
            };
            match node {
                ControlNode::Branch { condition, .. } => boolean(&mut diags, condition, "Branch"),
                ControlNode::Repeat { until, .. } => match until {
                    RepeatUntil::Count(0) => {
                        diags.push(Diagnostic::new(codes::CTRL_005, "Repeat count is 0").at(*id));
                    }
                    RepeatUntil::Count(_) => {}
                    RepeatUntil::Until(e) => boolean(&mut diags, e, "Repeat until"),
                },
                ControlNode::SubTask {
                    task: sub,
                    timeout_ticks,
                } => {
                    if *timeout_ticks == 0 {
                        diags.push(
                            Diagnostic::new(codes::CTRL_005, "SubTask timeout_ticks is 0").at(*id),
                        );
                    }
                    if let Some(e) = &sub.success {
                        boolean(&mut diags, e, "SubTask success");
                    }
                    if !stages.insert(sub.name.as_str()) {
                        diags.push(
                            Diagnostic::new(
                                codes::CTRL_006,
                                format!("stage name \"{}\" is used twice", sub.name),
                            )
                            .at(*id),
                        );
                    }
                    for name in &sub.rewards {
                        if !rewards.contains(name.as_str()) {
                            diags.push(
                                Diagnostic::new(
                                    codes::CTRL_006,
                                    format!("no Reward node named \"{name}\" in the task graph"),
                                )
                                .at(*id),
                            );
                        }
                    }
                    for channel in &sub.observation {
                        if !task.observation_spec.channels.contains_key(channel) {
                            diags.push(
                                Diagnostic::new(
                                    codes::CTRL_006,
                                    format!("channel \"{channel}\" is not in the ObservationSpec"),
                                )
                                .at(*id),
                            );
                        }
                    }
                }
                ControlNode::Sequence { .. } => {}
            }
        }
        diags
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{CmpOp, TaskNode};

    fn cmp(port: &str, rhs: f64) -> Expr {
        Expr::Compare {
            op: CmpOp::Gt,
            lhs: Box::new(Expr::Port(port.to_owned())),
            rhs: Box::new(Expr::Const(rhs)),
        }
    }

    fn stage(name: &str, rewards: &[&str]) -> ControlNode {
        ControlNode::SubTask {
            task: SubTaskRef {
                name: name.to_owned(),
                rewards: rewards.iter().map(|s| (*s).to_owned()).collect(),
                observation: BTreeSet::new(),
                success: Some(cmp("value", 0.5)),
                reset_on_entry: false,
                weight: 1.0,
            },
            timeout_ticks: 50,
        }
    }

    /// `Sequence(stage a, stage b)` over a task whose rewards are named `a` and `b`.
    fn fixture() -> (TaskIr, ControlGraph) {
        let terms = [("a".to_owned(), 1.0, 1.0), ("b".to_owned(), 1.0, 1.0)];
        let task = crate::task::testing::task_ir(&terms, 2);
        let graph = ControlGraph {
            root: NodeId(0),
            nodes: BTreeMap::from([
                (
                    NodeId(0),
                    ControlNode::Sequence {
                        children: vec![NodeId(1), NodeId(2)],
                    },
                ),
                (NodeId(1), stage("pick", &["a"])),
                (NodeId(2), stage("place", &["b"])),
            ]),
        };
        (task, graph)
    }

    fn codes_of(diags: &[Diagnostic]) -> Vec<&str> {
        diags.iter().map(|d| d.code.as_str()).collect()
    }

    #[test]
    fn a_well_formed_tree_validates_clean() {
        let (task, graph) = fixture();
        assert_eq!(codes_of(&graph.validate(&task)), Vec::<&str>::new());
        graph.control_hash().expect("a tree hashes");
    }

    #[test]
    fn a_shared_child_is_ctrl_003() {
        let (task, mut graph) = fixture();
        graph.nodes.insert(
            NodeId(0),
            ControlNode::Sequence {
                children: vec![NodeId(1), NodeId(1)],
            },
        );
        let diags = graph.validate(&task);
        assert!(codes_of(&diags).contains(&codes::CTRL_003), "{diags:?}");
    }

    #[test]
    fn a_cycle_is_ctrl_004() {
        let (task, mut graph) = fixture();
        graph.nodes.insert(
            NodeId(2),
            ControlNode::Repeat {
                body: NodeId(0),
                until: RepeatUntil::Count(2),
            },
        );
        let diags = graph.validate(&task);
        assert!(codes_of(&diags).contains(&codes::CTRL_004), "{diags:?}");
    }

    #[test]
    fn a_non_boolean_branch_condition_is_type_030() {
        let (task, mut graph) = fixture();
        graph.nodes.insert(
            NodeId(2),
            ControlNode::Branch {
                condition: Expr::Const(1.0),
                then_: NodeId(3),
                else_: NodeId(4),
            },
        );
        graph.nodes.insert(NodeId(3), stage("left", &[]));
        graph.nodes.insert(NodeId(4), stage("right", &[]));
        let diags = graph.validate(&task);
        assert!(codes_of(&diags).contains(&codes::TYPE_030), "{diags:?}");
    }

    #[test]
    fn zero_bounds_and_unknown_names_are_reported() {
        let (task, mut graph) = fixture();
        graph.nodes.insert(
            NodeId(2),
            ControlNode::SubTask {
                task: SubTaskRef {
                    name: "place".to_owned(),
                    rewards: ["nope".to_owned()].into(),
                    observation: ["nope".to_owned()].into(),
                    success: None,
                    reset_on_entry: true,
                    weight: 1.0,
                },
                timeout_ticks: 0,
            },
        );
        let diags = graph.validate(&task);
        let codes = codes_of(&diags);
        assert!(codes.contains(&codes::CTRL_005), "{diags:?}");
        assert_eq!(
            codes.iter().filter(|c| **c == codes::CTRL_006).count(),
            2,
            "{diags:?}"
        );
    }

    #[test]
    fn an_unreachable_node_and_a_missing_child_are_reported() {
        let (task, mut graph) = fixture();
        graph.nodes.insert(NodeId(9), stage("orphan", &[]));
        assert!(codes_of(&graph.validate(&task)).contains(&codes::CTRL_002));

        graph.nodes.insert(
            NodeId(0),
            ControlNode::Sequence {
                children: vec![NodeId(1), NodeId(42)],
            },
        );
        assert!(codes_of(&graph.validate(&task)).contains(&codes::CTRL_001));
    }

    #[test]
    fn sibling_order_is_semantic_and_node_ids_are_not() {
        let (_task, graph) = fixture();
        let base = graph.control_hash().unwrap();

        let mut swapped = graph.clone();
        swapped.nodes.insert(
            NodeId(0),
            ControlNode::Sequence {
                children: vec![NodeId(2), NodeId(1)],
            },
        );
        assert_ne!(swapped.control_hash().unwrap(), base, "order is semantic");

        // The same tree, renumbered: 0/1/2 -> 7/5/6.
        let relabelled = ControlGraph {
            root: NodeId(7),
            nodes: BTreeMap::from([
                (
                    NodeId(7),
                    ControlNode::Sequence {
                        children: vec![NodeId(5), NodeId(6)],
                    },
                ),
                (NodeId(5), stage("pick", &["a"])),
                (NodeId(6), stage("place", &["b"])),
            ]),
        };
        assert_eq!(relabelled.control_hash().unwrap(), base);
    }

    #[test]
    fn a_parameter_edit_moves_the_hash() {
        let (_task, graph) = fixture();
        let base = graph.control_hash().unwrap();
        let mut edited = graph.clone();
        let ControlNode::SubTask { task, .. } = edited.nodes.get_mut(&NodeId(1)).unwrap() else {
            unreachable!("node 1 is a SubTask")
        };
        task.weight += 1.0;
        assert_ne!(edited.control_hash().unwrap(), base);
    }

    #[test]
    fn serde_round_trip() {
        let (_task, graph) = fixture();
        let json = serde_json::to_string(&graph).unwrap();
        assert_eq!(serde_json::from_str::<ControlGraph>(&json).unwrap(), graph);
    }

    #[test]
    fn the_task_graph_is_untouched_by_slicing() {
        // A SubTask names rewards; it does not remove them from the IR-D graph.
        let (task, _graph) = fixture();
        let count = task
            .graph
            .nodes
            .values()
            .filter(|n| matches!(n, TaskNode::Reward { .. }))
            .count();
        assert_eq!(count, 2);
    }
}
