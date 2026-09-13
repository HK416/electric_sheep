//! The graph skeleton shared by the five IRs (spec 5.1, spec 11.1).
//!
//! `Graph<N>` is deliberately dumb: it holds nodes, edges and the boundary, and knows how to
//! order and type-check them. Everything that makes a graph a *Task* or *Learning* graph lives
//! in the node type `N` that each IR module supplies.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub use es_ir_types::NodeId;

use crate::codes;
use crate::diag::Diagnostic;
use crate::hash::CanonWriter;
use crate::types::PortType;

/// One side of an edge: a named port on a node.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PortRef {
    pub node: NodeId,
    pub port: String,
}

impl PortRef {
    pub fn new(node: NodeId, port: impl Into<String>) -> Self {
        Self {
            node,
            port: port.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Edge {
    /// An **output** port.
    pub from: PortRef,
    /// An **input** port. At most one edge may arrive at it.
    pub to: PortRef,
}

/// Which side of a node a port name is looked up on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    In,
    Out,
}

impl Dir {
    fn label(self) -> &'static str {
        match self {
            Self::In => "input",
            Self::Out => "output",
        }
    }
}

/// A declared port: name plus the type the node states for it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Port {
    pub name: String,
    pub ty: PortType,
}

impl Port {
    pub fn new(name: impl Into<String>, ty: PortType) -> Self {
        Self {
            name: name.into(),
            ty,
        }
    }
}

/// What every IR node kind provides. An internal helper trait with one impl per node kind —
/// not an extension point (INV-17).
pub trait IrNode {
    /// Stable kind tag; it is hash input, so renaming one is a schema change.
    fn kind(&self) -> &'static str;
    fn inputs(&self) -> Vec<Port>;
    fn outputs(&self) -> Vec<Port>;
    /// Writes everything about this node except its id and its edges.
    fn params_canonical(&self, w: &mut CanonWriter);
}

/// A typed dataflow graph. `nodes` is a `BTreeMap` so iteration is deterministic (spec 3.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph<N> {
    pub schema_version: u32,
    pub nodes: BTreeMap<NodeId, N>,
    pub edges: Vec<Edge>,
    /// Node input ports fed from outside the graph.
    pub inputs: Vec<PortRef>,
    /// Node output ports exposed to the outside.
    pub outputs: Vec<PortRef>,
}

impl<N> Graph<N> {
    pub fn new(schema_version: u32) -> Self {
        Self {
            schema_version,
            nodes: BTreeMap::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Inserts or replaces a node, returning the previous one.
    pub fn insert(&mut self, id: NodeId, node: N) -> Option<N> {
        self.nodes.insert(id, node)
    }

    pub fn connect(&mut self, from: NodeId, from_port: &str, to: NodeId, to_port: &str) {
        self.edges.push(Edge {
            from: PortRef::new(from, from_port),
            to: PortRef::new(to, to_port),
        });
    }

    /// Topological order, deterministic and independent of edge order.
    ///
    /// Errors: `GRAPH-002` if an edge names a node that is not in the graph, `GRAPH-001` if
    /// the graph has a cycle.
    pub fn topo_order(&self) -> Result<Vec<NodeId>, Diagnostic> {
        for e in &self.edges {
            for side in [&e.from, &e.to] {
                if !self.nodes.contains_key(&side.node) {
                    return Err(Diagnostic::new(
                        codes::GRAPH_002,
                        format!("edge endpoint node {} does not exist", side.node.0),
                    )
                    .at(side.node)
                    .on_port(side.port.clone()));
                }
            }
        }

        let mut indegree: BTreeMap<NodeId, usize> = self.nodes.keys().map(|id| (*id, 0)).collect();
        for e in &self.edges {
            *indegree.get_mut(&e.to.node).expect("checked above") += 1;
        }
        let mut ready: BTreeSet<NodeId> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);
            for e in self.edges.iter().filter(|e| e.from.node == id) {
                let d = indegree.get_mut(&e.to.node).expect("checked above");
                *d -= 1;
                if *d == 0 {
                    ready.insert(e.to.node);
                }
            }
        }

        if order.len() == self.nodes.len() {
            Ok(order)
        } else {
            let stuck: Vec<String> = indegree
                .iter()
                .filter(|(_, d)| **d > 0)
                .map(|(id, _)| id.0.to_string())
                .collect();
            Err(Diagnostic::new(
                codes::GRAPH_001,
                format!("nodes on the cycle: {}", stuck.join(", ")),
            )
            .with_hint("dataflow must be acyclic; feedback goes through a TemporalWindow"))
        }
    }

    /// Checks every edge and boundary port against the types `resolve` reports for them.
    ///
    /// `resolve` takes the node, the side of it, and the port name. Returning `None` means the
    /// node has no such port. A type-inferring front end can resolve differently from what the
    /// node declares; [`Graph::validate_declared_ports`] is the version that just asks the node.
    pub fn validate_ports<F>(&self, mut resolve: F) -> Vec<Diagnostic>
    where
        F: FnMut(&N, Dir, &str) -> Option<PortType>,
    {
        let mut diags = Vec::new();
        let ty_of = |this: &Self, side: &PortRef, dir: Dir, resolve: &mut F| match this
            .nodes
            .get(&side.node)
        {
            None => Err(Diagnostic::new(
                codes::GRAPH_002,
                format!("node {} does not exist", side.node.0),
            )
            .at(side.node)
            .on_port(side.port.clone())),
            Some(node) => resolve(node, dir, &side.port).ok_or_else(|| {
                Diagnostic::new(
                    codes::GRAPH_010,
                    format!("node has no {} port named \"{}\"", dir.label(), side.port),
                )
                .at(side.node)
                .on_port(side.port.clone())
            }),
        };

        let mut fed: BTreeSet<(NodeId, &str)> = BTreeSet::new();
        for e in &self.edges {
            if !fed.insert((e.to.node, e.to.port.as_str())) {
                diags.push(
                    Diagnostic::new(
                        codes::GRAPH_003,
                        format!("port \"{}\" already has an incoming edge", e.to.port),
                    )
                    .at(e.to.node)
                    .on_port(e.to.port.clone()),
                );
            }
            let from = ty_of(self, &e.from, Dir::Out, &mut resolve);
            let to = ty_of(self, &e.to, Dir::In, &mut resolve);
            match (from, to) {
                (Ok(from), Ok(to)) => {
                    if let Err(d) = from.compatible(&to) {
                        diags.push(d.at(e.to.node).on_port(e.to.port.clone()));
                    }
                }
                (a, b) => diags.extend([a, b].into_iter().filter_map(Result::err)),
            }
        }

        for (boundary, dir) in [(&self.inputs, Dir::In), (&self.outputs, Dir::Out)] {
            for side in boundary {
                if let Err(d) = ty_of(self, side, dir, &mut resolve) {
                    diags.push(d);
                }
            }
        }
        diags
    }
}

impl<N: IrNode> Graph<N> {
    /// [`Graph::validate_ports`] against the types the nodes themselves declare.
    pub fn validate_declared_ports(&self) -> Vec<Diagnostic> {
        self.validate_ports(|node, dir, name| {
            let ports = match dir {
                Dir::In => node.inputs(),
                Dir::Out => node.outputs(),
            };
            ports.into_iter().find(|p| p.name == name).map(|p| p.ty)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ElemType, Frame, Shape, TimeRef, Unit};

    fn ty(unit: Unit) -> PortType {
        PortType {
            elem: ElemType::F32,
            shape: Shape::new([3]),
            unit,
            frame: Frame::World,
            time: TimeRef::Tick,
            image: None,
        }
    }

    /// One input "in" and one output "out", both typed by the node's unit.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Node {
        unit: Unit,
    }

    impl IrNode for Node {
        fn kind(&self) -> &'static str {
            "Node"
        }
        fn inputs(&self) -> Vec<Port> {
            vec![Port::new("in", ty(self.unit.clone()))]
        }
        fn outputs(&self) -> Vec<Port> {
            vec![Port::new("out", ty(self.unit.clone()))]
        }
        fn params_canonical(&self, _w: &mut CanonWriter) {}
    }

    fn node(unit: Unit) -> Node {
        Node { unit }
    }

    fn chain_of(units: [Unit; 3]) -> Graph<Node> {
        let mut g = Graph::new(1);
        for (i, unit) in units.into_iter().enumerate() {
            g.insert(NodeId(i as u32), node(unit));
        }
        g.connect(NodeId(0), "out", NodeId(1), "in");
        g.connect(NodeId(1), "out", NodeId(2), "in");
        g
    }

    #[test]
    fn topo_order_is_a_valid_order() {
        let g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        assert_eq!(
            g.topo_order().unwrap(),
            vec![NodeId(0), NodeId(1), NodeId(2)]
        );
    }

    #[test]
    fn topo_order_ignores_edge_order() {
        let mut g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        g.edges.reverse();
        assert_eq!(
            g.topo_order().unwrap(),
            vec![NodeId(0), NodeId(1), NodeId(2)]
        );
    }

    #[test]
    fn cycle_is_graph_001() {
        let mut g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        g.connect(NodeId(2), "out", NodeId(0), "in");
        let err = g.topo_order().unwrap_err();
        assert_eq!(err.code.as_str(), codes::GRAPH_001);
        assert!(err.message.contains('0'), "{err}");
    }

    #[test]
    fn self_loop_is_a_cycle() {
        let mut g = Graph::new(1);
        g.insert(NodeId(0), node(Unit::Length));
        g.connect(NodeId(0), "out", NodeId(0), "in");
        assert_eq!(g.topo_order().unwrap_err().code.as_str(), codes::GRAPH_001);
    }

    #[test]
    fn missing_node_is_graph_002() {
        let mut g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        g.connect(NodeId(9), "out", NodeId(0), "in");
        assert_eq!(g.topo_order().unwrap_err().code.as_str(), codes::GRAPH_002);
        assert!(g
            .validate_declared_ports()
            .iter()
            .any(|d| d.code.as_str() == codes::GRAPH_002));
    }

    #[test]
    fn declared_ports_validate() {
        let g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        assert!(g.validate_declared_ports().is_empty());
    }

    #[test]
    fn unit_mismatch_across_an_edge_is_reported_on_the_consumer() {
        let g = chain_of([Unit::Length, Unit::Force, Unit::Length]);
        let diags = g.validate_declared_ports();
        assert_eq!(diags.len(), 2, "{diags:?}");
        assert!(diags.iter().all(|d| d.code.as_str() == codes::TYPE_003));
        assert_eq!(diags[0].node, Some(NodeId(1)));
        assert_eq!(diags[0].port.as_deref(), Some("in"));
    }

    #[test]
    fn unknown_port_is_graph_010() {
        let mut g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        g.connect(NodeId(0), "nope", NodeId(2), "in");
        let diags = g.validate_declared_ports();
        assert!(
            diags.iter().any(|d| d.code.as_str() == codes::GRAPH_010),
            "{diags:?}"
        );
    }

    #[test]
    fn second_incoming_edge_is_graph_003() {
        let mut g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        g.connect(NodeId(0), "out", NodeId(2), "in");
        let diags = g.validate_declared_ports();
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert_eq!(diags[0].code.as_str(), codes::GRAPH_003);
    }

    #[test]
    fn boundary_ports_are_checked() {
        let mut g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        g.inputs.push(PortRef::new(NodeId(0), "in"));
        g.outputs.push(PortRef::new(NodeId(2), "missing"));
        let diags = g.validate_declared_ports();
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert_eq!(diags[0].code.as_str(), codes::GRAPH_010);
    }

    #[test]
    fn serde_round_trip() {
        let g = chain_of([Unit::Length, Unit::Length, Unit::Length]);
        let json = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<Graph<Node>>(&json).unwrap(), g);
    }
}
