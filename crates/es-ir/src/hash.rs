//! Canonical encoding and the hash chain (spec 5.3, spec 11.2, Appendix B.6).
//!
//! [`CanonWriter`] is the only way a value becomes hash input: little-endian integers,
//! length-prefixed strings and byte strings, `f64` as IEEE bits with `-0.0` normalized and
//! `NaN` rejected. Maps must be iterated in sorted key order — use `BTreeMap`, which is sorted
//! by construction (`HashMap` is banned by spec 3.4 and by clippy).
//!
//! [`canonical_hash`] is the semantic identity of a graph (`*_hash` in spec 11.2): independent
//! of `NodeId` values and of node order, sensitive to every parameter.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::codes;
use crate::diag::Diagnostic;
use crate::graph::{Graph, IrNode, NodeId};

/// Domain separator: a change here invalidates every stored hash on purpose.
const CANON_TAG: &str = "es.ir.canon.v1";
const CHAIN_TAG: &str = "es.execution_hash.v1";

/// Canonical byte encoder. Writes never fail; an unencodable value latches an error that
/// [`CanonWriter::finish`] returns, so node encoders stay infallible.
#[derive(Debug, Default)]
pub struct CanonWriter {
    buf: Vec<u8>,
    err: Option<Diagnostic>,
}

impl CanonWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Length-prefixed UTF-8.
    pub fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    /// Length-prefixed bytes.
    pub fn bytes(&mut self, b: &[u8]) {
        self.seq(b.len());
        self.buf.extend_from_slice(b);
    }

    /// A raw 32-byte digest: fixed width, so no length prefix.
    pub fn digest(&mut self, d: &[u8; 32]) {
        self.buf.extend_from_slice(d);
    }

    /// Element count of a sequence or map. Every variable-length item is prefixed with one.
    pub fn seq(&mut self, len: usize) {
        self.u32(u32::try_from(len).unwrap_or(u32::MAX));
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }

    /// `f32` is widened: the two are then interchangeable in a hash, which is what a spec
    /// field switching precision should mean.
    pub fn f32(&mut self, v: f32) {
        self.f64(f64::from(v));
    }

    /// IEEE bits, with `-0.0` normalized to `0.0` (they compare equal, so they must hash
    /// equal) and `NaN` rejected (it compares unequal to itself, so no hash of it is sound).
    pub fn f64(&mut self, v: f64) {
        if v.is_nan() {
            self.fail(Diagnostic::new(
                codes::HASH_001,
                "NaN cannot be encoded canonically",
            ));
            return;
        }
        let v = if v == 0.0 { 0.0 } else { v };
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    /// Latches a diagnostic; the first one wins.
    pub fn fail(&mut self, diag: Diagnostic) {
        if self.err.is_none() {
            self.err = Some(diag);
        }
    }

    pub fn finish(self) -> Result<Vec<u8>, Diagnostic> {
        match self.err {
            Some(e) => Err(e),
            None => Ok(self.buf),
        }
    }

    pub fn hash(self) -> Result<[u8; 32], Diagnostic> {
        Ok(*blake3::hash(&self.finish()?).as_bytes())
    }
}

/// A node's *colour*: a digest of its kind, its parameters and — refined to a fixpoint — the
/// colours of everything it consumes and feeds, each through the port name that connects them.
/// A colour is what a node is, with no trace of what it is called.
type Colour = [u8; 32];

/// Colour every node, in topological order.
///
/// The first round is the node's own kind and parameters. Each further round mixes in the
/// sorted multiset of `(direction, near port, neighbour colour, far port)` over both incoming
/// and outgoing edges, which is one round of colour refinement; the loop stops as soon as a
/// round splits no further cell. So two nodes end up the same colour only when their whole
/// neighbourhoods, to any depth, agree — and node ids never enter.
fn refine<N: IrNode>(g: &Graph<N>) -> Result<(Vec<NodeId>, Vec<Colour>), Diagnostic> {
    let order = g.topo_order()?;
    let n = order.len();
    let pos: BTreeMap<NodeId, usize> = order.iter().enumerate().map(|(i, id)| (*id, i)).collect();

    let mut colour = Vec::with_capacity(n);
    for id in &order {
        let node = &g.nodes[id];
        let mut w = CanonWriter::new();
        w.str(node.kind());
        node.params_canonical(&mut w);
        colour.push(w.hash()?);
    }

    let cells = |c: &[Colour]| c.iter().copied().collect::<BTreeSet<Colour>>().len();
    let mut split = cells(&colour);
    // A refinement round splits at least one cell or it is at the fixpoint, so `n` rounds is
    // always enough.
    for _ in 0..n {
        let mut next = Vec::with_capacity(n);
        for (i, id) in order.iter().enumerate() {
            let mut neighbours: Vec<(u8, &str, Colour, &str)> = Vec::new();
            for e in &g.edges {
                if e.to.node == *id {
                    neighbours.push((
                        0,
                        e.to.port.as_str(),
                        colour[pos[&e.from.node]],
                        e.from.port.as_str(),
                    ));
                }
                if e.from.node == *id {
                    neighbours.push((
                        1,
                        e.from.port.as_str(),
                        colour[pos[&e.to.node]],
                        e.to.port.as_str(),
                    ));
                }
            }
            neighbours.sort_unstable();
            let mut w = CanonWriter::new();
            w.digest(&colour[i]);
            w.seq(neighbours.len());
            for (dir, near, digest, far) in neighbours {
                w.u8(dir);
                w.str(near);
                w.digest(&digest);
                w.str(far);
            }
            next.push(w.hash()?);
        }
        colour = next;
        let refined = cells(&colour);
        if refined == split {
            break;
        }
        split = refined;
    }
    Ok((order, colour))
}

/// The canonical node order: the sort by colour. [`crate::norm::canon_graph`] relabels by it.
///
/// Nodes that share a colour are interchangeable as far as [`canonical_hash`] is concerned, so
/// which of them comes first is left to the topological order and does not matter.
pub fn canonical_order<N: IrNode>(g: &Graph<N>) -> Result<Vec<NodeId>, Diagnostic> {
    let (order, colour) = refine(g)?;
    let mut ranked: Vec<usize> = (0..order.len()).collect();
    ranked.sort_by_key(|&i| colour[i]);
    Ok(ranked.into_iter().map(|i| order[i]).collect())
}

/// Semantic identity of a graph (spec 11.2 `*_hash`).
///
/// Nodes and edges are encoded by colour, as sorted multisets. A `NodeId` is therefore not
/// merely renumbered out of the hash, it is never consulted — which is what makes the
/// Appendix B.7 relabelling invariant hold for *every* graph, including one whose nodes tie
/// on colour, rather than for those where a tie-break happened to be stable.
pub fn canonical_hash<N: IrNode>(g: &Graph<N>) -> Result<[u8; 32], Diagnostic> {
    let (order, colour) = refine(g)?;
    let pos: BTreeMap<NodeId, usize> = order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let colour_of = |id: &NodeId| colour[pos[id]];

    let mut w = CanonWriter::new();
    w.str(CANON_TAG);
    w.u32(g.schema_version);
    let mut nodes = colour.clone();
    nodes.sort_unstable();
    w.seq(nodes.len());
    for c in &nodes {
        w.digest(c);
    }
    let mut edges: Vec<(Colour, &str, Colour, &str)> = g
        .edges
        .iter()
        .map(|e| {
            (
                colour_of(&e.from.node),
                e.from.port.as_str(),
                colour_of(&e.to.node),
                e.to.port.as_str(),
            )
        })
        .collect();
    edges.sort_unstable();
    w.seq(edges.len());
    for (from, from_port, to, to_port) in edges {
        w.digest(&from);
        w.str(from_port);
        w.digest(&to);
        w.str(to_port);
    }
    // Boundary order is semantic (it is the argument order), so it is not sorted.
    for boundary in [&g.inputs, &g.outputs] {
        w.seq(boundary.len());
        for p in boundary {
            w.digest(&colour_of(&p.node));
            w.str(&p.port);
        }
    }
    w.hash()
}

/// `content + schema + split` — the split is part of the identity because the same data with a
/// different train/val/test split is a different training run (spec 5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetHash {
    pub content: [u8; 32],
    pub schema: [u8; 32],
    pub split: [u8; 32],
}

/// Digest of the capability report of the machine that ran the execution. The report itself is
/// produced outside this crate (GPU limits, driver, CPU features); the IR only needs its
/// identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareCapability(pub [u8; 32]);

/// The hash chain of spec 5.3 / Appendix B.6.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashChain {
    pub asset: Vec<[u8; 32]>,
    pub scene: [u8; 32],
    /// Authoring identity: includes user-assigned node ids (spec 11.2).
    pub task_graph: [u8; 32],
    /// Semantic identity: after normalization and relabelling.
    pub task: [u8; 32],
    pub observation: [u8; 32],
    pub learning: [u8; 32],
    pub policy: [u8; 32],
    pub dataset: DatasetHash,
    pub deployment: [u8; 32],
    pub evaluation: Option<[u8; 32]>,
    pub compiler: [u8; 32],
    pub runtime: [u8; 32],
    pub hardware: HardwareCapability,
}

/// One component of the chain; the unit in which `diff` reports change (spec 27.1
/// `revalidation_trigger`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ChangedComponent {
    Asset,
    Scene,
    TaskGraph,
    Task,
    Observation,
    Learning,
    Policy,
    Dataset,
    Deployment,
    Evaluation,
    Compiler,
    Runtime,
    Hardware,
}

impl HashChain {
    /// `H(task, observation, learning, policy, dataset, deployment, compiler, runtime,
    /// hardware_capability)` — the tier-1 reproducibility condition (spec 5.3).
    ///
    /// `asset`, `scene` and `task_graph` are upstream of `task` and `evaluation` does not
    /// affect what is executed, so none of them are inputs.
    pub fn execution_hash(&self) -> [u8; 32] {
        let mut w = CanonWriter::new();
        w.str(CHAIN_TAG);
        for d in [
            &self.task,
            &self.observation,
            &self.learning,
            &self.policy,
            &self.dataset.content,
            &self.dataset.schema,
            &self.dataset.split,
            &self.deployment,
            &self.compiler,
            &self.runtime,
            &self.hardware.0,
        ] {
            w.digest(d);
        }
        w.hash()
            .expect("the hash chain contains no floats, so encoding cannot fail")
    }

    /// What changed between two chains, in declaration order.
    pub fn diff(&self, other: &Self) -> Vec<ChangedComponent> {
        use ChangedComponent as C;
        let mut out = Vec::new();
        let mut push = |changed: bool, c: C| {
            if changed {
                out.push(c);
            }
        };
        push(self.asset != other.asset, C::Asset);
        push(self.scene != other.scene, C::Scene);
        push(self.task_graph != other.task_graph, C::TaskGraph);
        push(self.task != other.task, C::Task);
        push(self.observation != other.observation, C::Observation);
        push(self.learning != other.learning, C::Learning);
        push(self.policy != other.policy, C::Policy);
        push(self.dataset != other.dataset, C::Dataset);
        push(self.deployment != other.deployment, C::Deployment);
        push(self.evaluation != other.evaluation, C::Evaluation);
        push(self.compiler != other.compiler, C::Compiler);
        push(self.runtime != other.runtime, C::Runtime);
        push(self.hardware != other.hardware, C::Hardware);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Port, PortRef};
    use crate::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
    use proptest::prelude::*;

    /// Test-only node type: enough shape to exercise params, arity and fan-in.
    #[derive(Clone, Debug, PartialEq)]
    enum ToyNode {
        Const { v: i64 },
        Scale { k: f64 },
        Add { bias: i64 },
    }

    /// An arity-preserving parameter change, for `param_change_changes_hash`.
    fn bump(node: &ToyNode) -> ToyNode {
        match node {
            ToyNode::Const { v } => ToyNode::Const { v: v + 1 },
            ToyNode::Scale { k } => ToyNode::Scale { k: k + 1.0 },
            ToyNode::Add { bias } => ToyNode::Add { bias: bias + 1 },
        }
    }

    fn scalar() -> PortType {
        PortType {
            elem: ElemType::F32,
            shape: Shape::new([1]),
            unit: Unit::Dimensionless,
            frame: Frame::World,
            time: TimeRef::Tick,
            image: None,
        }
    }

    impl IrNode for ToyNode {
        fn kind(&self) -> &'static str {
            match self {
                Self::Const { .. } => "Const",
                Self::Scale { .. } => "Scale",
                Self::Add { .. } => "Add",
            }
        }

        fn inputs(&self) -> Vec<Port> {
            match self {
                Self::Const { .. } => vec![],
                Self::Scale { .. } => vec![Port::new("x", scalar())],
                Self::Add { .. } => vec![Port::new("a", scalar()), Port::new("b", scalar())],
            }
        }

        fn outputs(&self) -> Vec<Port> {
            vec![Port::new("out", scalar())]
        }

        fn params_canonical(&self, w: &mut CanonWriter) {
            match self {
                Self::Const { v } => w.i64(*v),
                Self::Scale { k } => w.f64(*k),
                Self::Add { bias } => w.i64(*bias),
            }
        }
    }

    fn arb_node() -> impl Strategy<Value = ToyNode> {
        // A small value range on purpose: duplicate nodes are the hard case for relabelling.
        prop_oneof![
            (0i64..3).prop_map(|v| ToyNode::Const { v }),
            (-2.0f64..2.0).prop_map(|k| ToyNode::Scale { k }),
            (0i64..3).prop_map(|bias| ToyNode::Add { bias }),
        ]
    }

    fn build(nodes: Vec<ToyNode>, raw_edges: Vec<(u8, u8, bool)>) -> Graph<ToyNode> {
        let n = nodes.len();
        let ids: Vec<NodeId> = (0..n).map(|i| NodeId(i as u32 * 7 + 3)).collect();
        let mut g = Graph::new(1);
        for (id, node) in ids.iter().zip(nodes) {
            g.insert(*id, node);
        }
        let mut taken = std::collections::BTreeSet::new();
        for (a, b, p) in raw_edges {
            let (x, y) = (a as usize % n, b as usize % n);
            if x == y {
                continue;
            }
            // Edges always run low -> high index, so the graph is acyclic by construction.
            let (lo, hi) = (x.min(y), x.max(y));
            let ports = g.nodes[&ids[hi]].inputs();
            if ports.is_empty() {
                continue;
            }
            let port = ports[usize::from(p) % ports.len()].name.clone();
            if !taken.insert((ids[hi], port.clone())) {
                continue;
            }
            g.edges.push(Edge {
                from: PortRef::new(ids[lo], "out"),
                to: PortRef::new(ids[hi], port),
            });
        }
        g
    }

    fn arb_graph() -> impl Strategy<Value = Graph<ToyNode>> {
        (
            prop::collection::vec(arb_node(), 1..7),
            prop::collection::vec((any::<u8>(), any::<u8>(), any::<bool>()), 0..10),
        )
            .prop_map(|(nodes, edges)| build(nodes, edges))
    }

    /// Relabels every node, which also reverses the order `Graph::nodes` iterates in.
    fn relabel(g: &Graph<ToyNode>, f: impl Fn(u32) -> u32) -> Graph<ToyNode> {
        let mut out = Graph::new(g.schema_version);
        for (id, node) in &g.nodes {
            out.insert(NodeId(f(id.0)), node.clone());
        }
        out.edges = g
            .edges
            .iter()
            .map(|e| Edge {
                from: PortRef::new(NodeId(f(e.from.node.0)), e.from.port.clone()),
                to: PortRef::new(NodeId(f(e.to.node.0)), e.to.port.clone()),
            })
            .collect();
        out
    }

    proptest! {
        /// Appendix B.7: `hash(canon(g)) == hash(canon(shuffle_ids(g)))`.
        #[test]
        fn hash_independent_of_node_ids(g in arb_graph()) {
            let base = canonical_hash(&g).unwrap();
            // Order-reversing relabelling and an order-preserving one.
            prop_assert_eq!(canonical_hash(&relabel(&g, |i| 10_000 - i)).unwrap(), base);
            prop_assert_eq!(canonical_hash(&relabel(&g, |i| i * 13 + 1)).unwrap(), base);
        }

        /// Appendix B.7: `hash(canon(g)) != hash(canon(change_any_param(g)))`.
        #[test]
        fn param_change_changes_hash(g in arb_graph(), idx in any::<prop::sample::Index>()) {
            let ids: Vec<NodeId> = g.nodes.keys().copied().collect();
            let id = ids[idx.index(ids.len())];
            let changed = bump(&g.nodes[&id]);
            let mut g2 = g.clone();
            g2.insert(id, changed);
            prop_assert_ne!(canonical_hash(&g).unwrap(), canonical_hash(&g2).unwrap());
        }
    }

    #[test]
    fn node_order_does_not_reach_the_hash() {
        // Same graph, nodes inserted in the opposite order.
        let mut a = Graph::new(1);
        a.insert(NodeId(1), ToyNode::Const { v: 5 });
        a.insert(NodeId(2), ToyNode::Scale { k: 2.0 });
        a.connect(NodeId(1), "out", NodeId(2), "x");
        let mut b = Graph::new(1);
        b.insert(NodeId(9), ToyNode::Scale { k: 2.0 });
        b.insert(NodeId(4), ToyNode::Const { v: 5 });
        b.connect(NodeId(4), "out", NodeId(9), "x");
        assert_eq!(canonical_hash(&a).unwrap(), canonical_hash(&b).unwrap());
    }

    #[test]
    fn duplicate_nodes_with_different_consumers_are_distinguished() {
        // Two identical Const nodes, feeding two different Scale nodes. Relabelling must not
        // change the hash, and swapping which Const feeds which Scale must not either.
        let mut g = Graph::new(1);
        g.insert(NodeId(1), ToyNode::Const { v: 5 });
        g.insert(NodeId(2), ToyNode::Const { v: 5 });
        g.insert(NodeId(3), ToyNode::Scale { k: 1.0 });
        g.insert(NodeId(4), ToyNode::Scale { k: 2.0 });
        g.connect(NodeId(1), "out", NodeId(3), "x");
        g.connect(NodeId(2), "out", NodeId(4), "x");
        let mut swapped = Graph::new(1);
        swapped.insert(NodeId(1), ToyNode::Const { v: 5 });
        swapped.insert(NodeId(2), ToyNode::Const { v: 5 });
        swapped.insert(NodeId(3), ToyNode::Scale { k: 1.0 });
        swapped.insert(NodeId(4), ToyNode::Scale { k: 2.0 });
        swapped.connect(NodeId(2), "out", NodeId(3), "x");
        swapped.connect(NodeId(1), "out", NodeId(4), "x");
        assert_eq!(
            canonical_hash(&g).unwrap(),
            canonical_hash(&swapped).unwrap()
        );
    }

    /// Regression, found by `hash_independent_of_node_ids`: two `Add` nodes whose *parents*
    /// are indistinguishable from below tie on every fingerprint an earlier two-pass scheme
    /// computed, yet their parents are told apart from above — so ranking them by a tie-break
    /// made the hash depend on node ids. Encoding by colour removes the tie-break entirely.
    #[test]
    fn ties_whose_parents_differ_still_hash_stably() {
        let mut g = Graph::new(1);
        for (id, node) in [
            (10, ToyNode::Add { bias: 2 }),
            (17, ToyNode::Add { bias: 2 }),
            (24, ToyNode::Add { bias: 0 }),
            (31, ToyNode::Add { bias: 0 }),
            (38, ToyNode::Scale { k: 0.0 }),
        ] {
            g.insert(NodeId(id), node);
        }
        g.connect(NodeId(10), "out", NodeId(38), "x");
        g.connect(NodeId(10), "out", NodeId(24), "a");
        g.connect(NodeId(17), "out", NodeId(31), "a");
        assert_eq!(
            canonical_hash(&relabel(&g, |i| 10_000 - i)).unwrap(),
            canonical_hash(&g).unwrap()
        );
    }

    #[test]
    fn edge_change_changes_hash() {
        let mut a = Graph::new(1);
        a.insert(NodeId(1), ToyNode::Const { v: 5 });
        a.insert(NodeId(2), ToyNode::Const { v: 6 });
        a.insert(NodeId(3), ToyNode::Add { bias: 0 });
        a.connect(NodeId(1), "out", NodeId(3), "a");
        a.connect(NodeId(2), "out", NodeId(3), "b");
        let mut b = a.clone();
        b.edges[1].to.port = "a".into();
        b.edges[0].to.port = "b".into();
        assert_ne!(canonical_hash(&a).unwrap(), canonical_hash(&b).unwrap());
    }

    #[test]
    fn schema_version_reaches_the_hash() {
        let mut a = Graph::new(1);
        a.insert(NodeId(1), ToyNode::Add { bias: 0 });
        let mut b = a.clone();
        b.schema_version = 2;
        assert_ne!(canonical_hash(&a).unwrap(), canonical_hash(&b).unwrap());
    }

    #[test]
    fn cycle_is_reported() {
        let mut g = Graph::new(1);
        g.insert(NodeId(1), ToyNode::Scale { k: 1.0 });
        g.insert(NodeId(2), ToyNode::Scale { k: 1.0 });
        g.connect(NodeId(1), "out", NodeId(2), "x");
        g.connect(NodeId(2), "out", NodeId(1), "x");
        let err = canonical_hash(&g).unwrap_err();
        assert_eq!(err.code.as_str(), codes::GRAPH_001);
    }

    #[test]
    fn negative_zero_is_normalized_and_nan_is_rejected() {
        let mut a = CanonWriter::new();
        a.f64(-0.0);
        let mut b = CanonWriter::new();
        b.f64(0.0);
        assert_eq!(a.finish().unwrap(), b.finish().unwrap());

        let mut nan = CanonWriter::new();
        nan.f64(f64::NAN);
        nan.u32(1); // writes after a failure are still accepted, the error is latched
        assert_eq!(nan.finish().unwrap_err().code.as_str(), codes::HASH_001);

        let mut g = Graph::new(1);
        g.insert(NodeId(1), ToyNode::Scale { k: f64::NAN });
        assert_eq!(
            canonical_hash(&g).unwrap_err().code.as_str(),
            codes::HASH_001
        );
    }

    #[test]
    fn encoding_is_unambiguous() {
        // Length prefixes stop "ab" + "c" from colliding with "a" + "bc".
        let mut a = CanonWriter::new();
        a.str("ab");
        a.str("c");
        let mut b = CanonWriter::new();
        b.str("a");
        b.str("bc");
        assert_ne!(a.finish().unwrap(), b.finish().unwrap());
    }

    type Mutation = (ChangedComponent, fn(&mut HashChain));

    fn chain() -> HashChain {
        HashChain {
            asset: vec![[1u8; 32]],
            scene: [2u8; 32],
            task_graph: [3u8; 32],
            task: [4u8; 32],
            observation: [5u8; 32],
            learning: [6u8; 32],
            policy: [7u8; 32],
            dataset: DatasetHash {
                content: [8u8; 32],
                schema: [9u8; 32],
                split: [10u8; 32],
            },
            deployment: [11u8; 32],
            evaluation: Some([12u8; 32]),
            compiler: [13u8; 32],
            runtime: [14u8; 32],
            hardware: HardwareCapability([15u8; 32]),
        }
    }

    #[test]
    fn execution_hash_ignores_authoring_and_evaluation() {
        let a = chain();
        let mut b = a.clone();
        b.asset = vec![[99u8; 32]];
        b.scene = [99u8; 32];
        b.task_graph = [99u8; 32];
        b.evaluation = None;
        assert_eq!(a.execution_hash(), b.execution_hash());
        assert_eq!(
            a.diff(&b),
            vec![
                ChangedComponent::Asset,
                ChangedComponent::Scene,
                ChangedComponent::TaskGraph,
                ChangedComponent::Evaluation,
            ]
        );
    }

    #[test]
    fn execution_hash_tracks_every_input_component() {
        let base = chain();
        let mutations: Vec<Mutation> = vec![
            (ChangedComponent::Task, |c| c.task = [99u8; 32]),
            (ChangedComponent::Observation, |c| {
                c.observation = [99u8; 32];
            }),
            (ChangedComponent::Learning, |c| c.learning = [99u8; 32]),
            (ChangedComponent::Policy, |c| c.policy = [99u8; 32]),
            (ChangedComponent::Dataset, |c| c.dataset.split = [99u8; 32]),
            (ChangedComponent::Deployment, |c| c.deployment = [99u8; 32]),
            (ChangedComponent::Compiler, |c| c.compiler = [99u8; 32]),
            (ChangedComponent::Runtime, |c| c.runtime = [99u8; 32]),
            (ChangedComponent::Hardware, |c| {
                c.hardware = HardwareCapability([99u8; 32]);
            }),
        ];
        for (component, mutate) in mutations {
            let mut other = base.clone();
            mutate(&mut other);
            assert_ne!(
                base.execution_hash(),
                other.execution_hash(),
                "{component:?} did not reach execution_hash"
            );
            assert_eq!(base.diff(&other), vec![component]);
        }
        assert!(base.diff(&base.clone()).is_empty());
    }

    #[test]
    fn chain_serde_round_trip() {
        let c = chain();
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<HashChain>(&json).unwrap(), c);
    }
}
