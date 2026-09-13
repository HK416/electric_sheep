//! The editable graph of spec 23.4 stage 2, as a headless edit model.
//!
//! Everything an edit *decides* lives here and is judged by `cargo test -p es-editor`;
//! `crate::app` only turns a mouse gesture into an [`Edit`] and paints the result. CI has no
//! display, so a decision that leaks into the egui half is a decision nothing tests.
//!
//! Three rules shape this module:
//!
//! * **Layout is not IR** (spec 4.2 rule 7, spec 14.3). [`Edit::MoveNode`] writes
//!   [`Layout::positions`] and nothing else; the IR's `*_hash` cannot see it, which is a test
//!   ([`tests::move_node_does_not_move_the_hash`]).
//! * **Nodes are built by the Node SDK, never by this crate.** Every node comes from
//!   `TaskNodeRegistry::create` / `LearningNodeRegistry::create` (spec 6.3, spec 8.3,
//!   `INV-17`), so adding a node kind is an `es-ir` change and the editor learns about it for
//!   free. There is no per-kind code below this line. See `docs/design/node-sdk.md`.
//! * **Every edit is validated, and a structurally invalid one is refused.** An edit that
//!   introduces a new *error* in [`Graph::validate_declared_ports`] is reverted and reported;
//!   the full [`EditIr::validate`] output is advisory, because a half-authored graph is
//!   exactly the graph someone has the editor open for.
//!
//! Observation IR is editable in every way except adding a node and setting a parameter:
//! `INV-17` allows exactly two node factories, `TaskNodeFactory` and `LearningNodeFactory`,
//! and the editor is not allowed to invent a third. Those two edits report `FACTORY-001`.

use std::collections::BTreeMap;
use std::fmt;

use es_ir::graph::{Edge, NodeId, PortRef};
use es_ir::learning::LearningGraph;
use es_ir::observation::ObservationIr;
use es_ir::serial::{self, AnyIr, IrKind, Layout, SerialError};
use es_ir::task::TaskIr;
use es_ir::{codes, Diagnostic, IrNode};

use crate::model::palette::Registries;

/// Runs `$body` with `$g` bound to the `Graph<N>` inside `$ir`, whichever IR that is. The three
/// IRs spell their graph field differently and their node types differ, so this is a macro and
/// not a method — there is no trait here to add (`INV-17`).
macro_rules! with_graph {
    ($ir:expr, $g:ident => $body:expr) => {
        match $ir {
            EditIr::Task(ir) => {
                let $g = &ir.graph;
                $body
            }
            EditIr::Observation(ir) => {
                let $g = &ir.graph;
                $body
            }
            EditIr::Learning(ir) => {
                let $g = &ir.nodes;
                $body
            }
        }
    };
}

macro_rules! with_graph_mut {
    ($ir:expr, $g:ident => $body:expr) => {
        match $ir {
            EditIr::Task(ir) => {
                let $g = &mut ir.graph;
                $body
            }
            EditIr::Observation(ir) => {
                let $g = &mut ir.graph;
                $body
            }
            EditIr::Learning(ir) => {
                let $g = &mut ir.nodes;
                $body
            }
        }
    };
}

/// The three IRs that are node graphs a user authors. Deployment and Evaluation are records,
/// not graphs (spec 9.2, spec 10.2), so they are not editable here.
#[derive(Clone, Debug, PartialEq)]
pub enum EditIr {
    Task(TaskIr),
    Observation(ObservationIr),
    Learning(LearningGraph),
}

impl EditIr {
    pub fn kind(&self) -> IrKind {
        match self {
            Self::Task(_) => IrKind::Task,
            Self::Observation(_) => IrKind::Observation,
            Self::Learning(_) => IrKind::Learning,
        }
    }

    /// The IR's own full check (spec 6.6, spec 7.5, spec 8.4). Advisory: it complains about a
    /// graph that is merely unfinished, which is a normal state in an editor.
    pub fn validate(&self) -> Vec<Diagnostic> {
        match self {
            Self::Task(ir) => ir.validate(),
            Self::Observation(ir) => ir.validate(),
            Self::Learning(ir) => ir.validate(),
        }
    }

    /// The structural check an edit is judged by: port existence, arity, and port types along
    /// every edge (`GRAPH-002/003/010`, `TYPE-00x`).
    pub fn port_diagnostics(&self) -> Vec<Diagnostic> {
        with_graph!(self, g => g.validate_declared_ports())
    }

    /// The IR content hash of the spec 5.3 chain. Layout is not an input to it.
    pub fn hash(&self) -> Result<[u8; 32], Diagnostic> {
        match self {
            Self::Task(ir) => ir.task_hash(),
            Self::Observation(ir) => ir.observation_hash(),
            Self::Learning(ir) => ir.learning_hash(),
        }
    }

    pub fn edges(&self) -> &[Edge] {
        with_graph!(self, g => &g.edges)
    }

    /// `(id, kind)` for every node, in the graph's canonical `NodeId` order.
    pub fn nodes(&self) -> Vec<(NodeId, &'static str)> {
        with_graph!(self, g => g.nodes.iter().map(|(id, n)| (*id, n.kind())).collect())
    }

    pub fn contains(&self, id: NodeId) -> bool {
        with_graph!(self, g => g.nodes.contains_key(&id))
    }

    /// The port names one node actually declares — asked of the node, not of the palette's
    /// representative instance, because arity can depend on a parameter (`Concat.parts`,
    /// `Compare.rhs`). This is what the canvas hit-tests against.
    pub fn port_names(&self, id: NodeId, dir: es_ir::Dir) -> Vec<String> {
        with_graph!(self, g => g
            .nodes
            .get(&id)
            .map(|n| match dir {
                es_ir::Dir::In => n.inputs(),
                es_ir::Dir::Out => n.outputs(),
            })
            .unwrap_or_default()
            .into_iter()
            .map(|p| p.name)
            .collect())
    }

    /// The id a new node gets: one past the highest in use, so it is deterministic and never
    /// reuses an id an undone edit still refers to.
    pub fn next_id(&self) -> NodeId {
        with_graph!(self, g => NodeId(g.nodes.keys().next_back().map_or(0, |id| id.0 + 1)))
    }

    pub fn to_any(&self) -> AnyIr {
        match self {
            Self::Task(ir) => AnyIr::Task(ir.clone()),
            Self::Observation(ir) => AnyIr::Observation(ir.clone()),
            Self::Learning(ir) => AnyIr::Learning(ir.clone()),
        }
    }

    fn add_node(
        &mut self,
        reg: &Registries,
        kind: &str,
        params: &toml::Value,
    ) -> Result<NodeId, Diagnostic> {
        let id = self.next_id();
        match self {
            Self::Task(ir) => {
                ir.graph.insert(id, reg.task.create(kind, params)?);
            }
            Self::Learning(ir) => {
                ir.nodes.insert(id, reg.learning.create(kind, params)?);
            }
            Self::Observation(_) => return Err(no_factory(kind)),
        }
        Ok(id)
    }

    /// Replaces one parameter by re-building the node from the registry: the current node is
    /// serialized to its parameter table, the key is overwritten, and the factory builds it
    /// again. Nothing here knows what any kind's fields are.
    fn set_param(
        &mut self,
        reg: &Registries,
        id: NodeId,
        key: &str,
        value: toml::Value,
    ) -> Result<(), Diagnostic> {
        match self {
            Self::Task(ir) => {
                let node = ir.graph.nodes.get(&id).ok_or_else(|| unknown_node(id))?;
                let (kind, params) = params_with(node, node.kind(), key, value)?;
                let built = reg.task.create(&kind, &params)?;
                ir.graph.insert(id, built);
            }
            Self::Learning(ir) => {
                let node = ir.nodes.nodes.get(&id).ok_or_else(|| unknown_node(id))?;
                let (kind, params) = params_with(node, node.kind(), key, value)?;
                let built = reg.learning.create(&kind, &params)?;
                ir.nodes.insert(id, built);
            }
            Self::Observation(ir) => {
                let node = ir.graph.nodes.get(&id).ok_or_else(|| unknown_node(id))?;
                return Err(no_factory(node.kind()));
            }
        }
        Ok(())
    }

    /// Removes a node *and* every edge touching it, so a removal never leaves a dangling
    /// endpoint (which `GRAPH-002` would then refuse the edit for).
    fn remove_node(&mut self, id: NodeId) -> bool {
        with_graph_mut!(self, g => {
            let existed = g.nodes.remove(&id).is_some();
            g.edges.retain(|e| e.from.node != id && e.to.node != id);
            g.inputs.retain(|p| p.node != id);
            g.outputs.retain(|p| p.node != id);
            existed
        })
    }

    fn push_edge(&mut self, edge: Edge) {
        with_graph_mut!(self, g => g.edges.push(edge));
    }

    fn drop_edge(&mut self, edge: &Edge) -> bool {
        with_graph_mut!(self, g => {
            let before = g.edges.len();
            g.edges.retain(|e| e != edge);
            g.edges.len() != before
        })
    }
}

impl TryFrom<AnyIr> for EditIr {
    type Error = EditError;

    fn try_from(ir: AnyIr) -> Result<Self, EditError> {
        match ir {
            AnyIr::Task(ir) => Ok(Self::Task(ir)),
            AnyIr::Observation(ir) => Ok(Self::Observation(ir)),
            AnyIr::Learning(ir) => Ok(Self::Learning(ir)),
            other => Err(EditError::NotAGraph(other.kind())),
        }
    }
}

/// Serializes `node` to its externally tagged parameter table and overwrites one key.
/// Overwriting a key the node does not have is `FACTORY-003`: the editor has no reflection
/// beyond what the node itself serializes, so an unknown key is a typo, not a new field.
fn params_with<N: serde::Serialize>(
    node: &N,
    kind: &'static str,
    key: &str,
    value: toml::Value,
) -> Result<(String, toml::Value), Diagnostic> {
    let tagged = toml::Value::try_from(node).map_err(|e| {
        Diagnostic::new(
            codes::FACTORY_003,
            format!("cannot read the parameters of '{kind}': {e}"),
        )
    })?;
    let mut params = match tagged {
        toml::Value::Table(mut t) => match t.remove(kind) {
            Some(toml::Value::Table(params)) => params,
            _ => toml::Table::new(),
        },
        _ => toml::Table::new(),
    };
    if !params.contains_key(key) {
        return Err(Diagnostic::new(
            codes::FACTORY_003,
            format!("node kind '{kind}' has no parameter named '{key}'"),
        ));
    }
    params.insert(key.to_owned(), value);
    Ok((kind.to_owned(), toml::Value::Table(params)))
}

fn no_factory(kind: &str) -> Diagnostic {
    Diagnostic::new(
        codes::FACTORY_001,
        format!("no node factory can build '{kind}'"),
    )
    .with_hint(
        "INV-17 allows exactly two node factories, TaskNodeFactory and LearningNodeFactory; \
         Observation IR nodes are edited as text until one exists",
    )
}

fn unknown_node(id: NodeId) -> Diagnostic {
    Diagnostic::new(codes::GRAPH_002, format!("node {} does not exist", id.0)).at(id)
}

/// One user action. Everything the editor can do to a graph is one of these six, which is why
/// undo, redo, and the `.esgraph` on disk cannot drift apart.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// `params` is the node's own field table, keyed exactly as the node's serde field names;
    /// `Palette::defaults` produces a starting one.
    AddNode {
        kind: String,
        params: toml::Value,
        pos: [f32; 2],
    },
    RemoveNode {
        node: NodeId,
    },
    Connect {
        from: PortRef,
        to: PortRef,
    },
    Disconnect {
        from: PortRef,
        to: PortRef,
    },
    SetParam {
        node: NodeId,
        key: String,
        value: toml::Value,
    },
    /// **Layout only** (spec 4.2 rule 7): this cannot change any `*_hash`.
    MoveNode {
        node: NodeId,
        pos: [f32; 2],
    },
}

/// The state one [`Edit`] moves between. Whole-state snapshots rather than per-edit inverses:
/// an inverse of `RemoveNode` has to restore the node, its parameters, its layout entry and
/// every incident edge, which is four chances to be subtly wrong, and undo that is subtly
/// wrong is worse than undo that is fat.
//
// ponytail: one clone of the IR per edit; an authored graph is hundreds of nodes, not
// millions. If a graph ever gets big enough for this to show up, store edit inverses instead —
// `Edit` is already the log.
#[derive(Clone, Debug, PartialEq)]
struct Snapshot {
    graph: EditIr,
    layout: Layout,
}

/// One entry of the undo or redo stack: the edit, and the state to return to.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub edit: Edit,
    state: Snapshot,
}

/// An open graph: the IR, its `.eslayout` sidecar, the node factories, and the history.
#[derive(Debug)]
pub struct EditSession {
    pub graph: EditIr,
    pub layout: Layout,
    pub registries: Registries,
    diagnostics: Vec<Diagnostic>,
    undo: Vec<Step>,
    redo: Vec<Step>,
}

impl EditSession {
    pub fn new(graph: EditIr, layout: Layout) -> Self {
        let diagnostics = graph.validate();
        Self {
            graph,
            layout,
            registries: Registries::with_builtins(),
            diagnostics,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Opens a `.esgraph` and, if present, its `.eslayout` sidecar (spec 14.3). The IR is
    /// decoded from `graph_toml` alone; the sidecar never influences it.
    pub fn load(graph_toml: &str, layout_toml: Option<&str>) -> Result<Self, EditError> {
        let (ir, layout) = serial::parse_esgraph(graph_toml, layout_toml)?;
        Ok(Self::new(EditIr::try_from(ir)?, layout.unwrap_or_default()))
    }

    /// `(esgraph, eslayout)`. Two files, always: the layout is never mixed into the IR, so the
    /// `*_hash` of what is written back equals the `*_hash` of what is in memory.
    pub fn save(&self) -> Result<(String, String), EditError> {
        let (graph, layout) = serial::write_esgraph(&self.graph.to_any(), Some(&self.layout))?;
        Ok((graph, layout.expect("a layout was passed")))
    }

    /// Every complaint the IR has about its current state. Advisory — see the module docs.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn hash(&self) -> Result<[u8; 32], Diagnostic> {
        self.graph.hash()
    }

    pub fn history(&self) -> &[Step] {
        &self.undo
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Applies one edit, or refuses it.
    ///
    /// Refused (state untouched): the edit itself cannot be built — an unknown kind, a missing
    /// node, params the factory rejects — or it introduces a new **error** in
    /// [`EditIr::port_diagnostics`]. Errors that were already there stay there; an editor that
    /// refuses to work on a broken graph is an editor nobody can fix a graph with.
    pub fn apply(&mut self, edit: Edit) -> Result<(), Vec<Diagnostic>> {
        let before = self.snapshot();
        let before_ports = self.graph.port_diagnostics();
        if let Err(d) = self.mutate(&edit) {
            self.restore(before);
            return Err(vec![d]);
        }
        let new_errors: Vec<Diagnostic> = self
            .graph
            .port_diagnostics()
            .into_iter()
            .filter(|d| d.is_error() && !before_ports.contains(d))
            .collect();
        if !new_errors.is_empty() {
            self.restore(before);
            return Err(new_errors);
        }
        self.diagnostics = self.graph.validate();
        self.undo.push(Step {
            edit,
            state: before,
        });
        self.redo.clear();
        Ok(())
    }

    /// Reverts the last applied edit. `false` if there is none.
    pub fn undo(&mut self) -> bool {
        let Some(step) = self.undo.pop() else {
            return false;
        };
        let current = self.snapshot();
        self.restore(step.state);
        self.redo.push(Step {
            edit: step.edit,
            state: current,
        });
        true
    }

    /// Re-applies the last undone edit. `false` if there is none.
    pub fn redo(&mut self) -> bool {
        let Some(step) = self.redo.pop() else {
            return false;
        };
        let current = self.snapshot();
        self.restore(step.state);
        self.undo.push(Step {
            edit: step.edit,
            state: current,
        });
        true
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            graph: self.graph.clone(),
            layout: self.layout.clone(),
        }
    }

    fn restore(&mut self, state: Snapshot) {
        self.graph = state.graph;
        self.layout = state.layout;
        self.diagnostics = self.graph.validate();
    }

    fn mutate(&mut self, edit: &Edit) -> Result<(), Diagnostic> {
        match edit {
            Edit::AddNode { kind, params, pos } => {
                let id = self.graph.add_node(&self.registries, kind, params)?;
                self.layout.positions.insert(id, *pos);
            }
            Edit::RemoveNode { node } => {
                if !self.graph.remove_node(*node) {
                    return Err(unknown_node(*node));
                }
                self.layout.positions.remove(node);
                self.layout.collapsed.remove(node);
                self.layout.notes.remove(node);
            }
            Edit::Connect { from, to } => self.graph.push_edge(Edge {
                from: from.clone(),
                to: to.clone(),
            }),
            Edit::Disconnect { from, to } => {
                let edge = Edge {
                    from: from.clone(),
                    to: to.clone(),
                };
                if !self.graph.drop_edge(&edge) {
                    return Err(Diagnostic::new(
                        codes::GRAPH_002,
                        format!(
                            "no edge from node {} port \"{}\" to node {} port \"{}\"",
                            from.node.0, from.port, to.node.0, to.port
                        ),
                    )
                    .at(to.node));
                }
            }
            Edit::SetParam { node, key, value } => {
                self.graph
                    .set_param(&self.registries, *node, key, value.clone())?;
            }
            Edit::MoveNode { node, pos } => {
                if !self.graph.contains(*node) {
                    return Err(unknown_node(*node));
                }
                self.layout.positions.insert(*node, *pos);
            }
        }
        Ok(())
    }
}

/// Opening or saving a graph failed. Two cases only: the TOML half, and "this file is an IR
/// the editor cannot draw as a graph".
#[derive(Debug)]
pub enum EditError {
    Serial(SerialError),
    NotAGraph(IrKind),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serial(e) => write!(f, "{e}"),
            Self::NotAGraph(kind) => write!(
                f,
                "a {kind:?} IR is a record, not a node graph; it has no editable graph"
            ),
        }
    }
}

impl std::error::Error for EditError {}

impl From<SerialError> for EditError {
    fn from(e: SerialError) -> Self {
        Self::Serial(e)
    }
}

/// Positions for nodes the `.eslayout` does not mention, so a freshly imported graph is not a
/// pile at the origin. Deterministic: canonical node order, a fixed grid.
pub fn fill_missing_positions(graph: &EditIr, layout: &mut Layout, columns: usize) {
    let missing: Vec<NodeId> = graph
        .nodes()
        .into_iter()
        .map(|(id, _)| id)
        .filter(|id| !layout.positions.contains_key(id))
        .collect();
    for (i, id) in missing.into_iter().enumerate() {
        let col = i % columns.max(1);
        let row = i / columns.max(1);
        layout
            .positions
            .insert(id, [col as f32 * 220.0, row as f32 * 90.0]);
    }
}

/// `NodeId -> kind` for the whole graph, which is what the egui layer needs to look a node's
/// schema up in the palette.
pub fn kinds_by_id(graph: &EditIr) -> BTreeMap<NodeId, &'static str> {
    graph.nodes().into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::palette::Palette;

    use es_ir::factory::{NodeSchema, ParamSchema, ParamType, TaskNodeFactory};
    use es_ir::graph::{Graph, Port};
    use es_ir::task::{Aggregation, ObservationSpec, SceneRef, TaskConfig, TaskNode};
    use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
    use std::collections::BTreeSet;

    // --- an example third-party factory (docs/design/node-sdk.md) -----------------------------

    const GRASP_SCORE: &str = "GraspScore";

    fn scalar() -> PortType {
        PortType {
            elem: ElemType::F32,
            shape: Shape::new(vec![1]),
            unit: Unit::Dimensionless,
            frame: Frame::World,
            time: TimeRef::Tick,
            image: None,
        }
    }

    /// A third-party node kind: one authoring shortcut that expands to a builtin `Reward`.
    /// `TaskNode` is a closed enum and `BUILTIN_TASK_KINDS` is frozen, so a factory composes
    /// what exists rather than inventing a kind tag the hash has never seen.
    #[derive(Debug)]
    struct GraspScoreNodes;

    impl TaskNodeFactory for GraspScoreNodes {
        fn kinds(&self) -> &[&'static str] {
            &[GRASP_SCORE]
        }

        fn create(&self, kind: &str, params: &toml::Value) -> Result<TaskNode, Diagnostic> {
            if kind != GRASP_SCORE {
                return Err(Diagnostic::new(
                    codes::FACTORY_001,
                    format!("unknown node kind '{kind}'"),
                ));
            }
            Ok(TaskNode::Reward {
                name: params
                    .get("name")
                    .and_then(toml::Value::as_str)
                    .unwrap_or("grasp")
                    .to_owned(),
                weight: params
                    .get("weight")
                    .and_then(toml::Value::as_float)
                    .unwrap_or(1.0),
                aggregation: Aggregation::Sum,
                ty: scalar(),
            })
        }

        fn schema(&self, kind: &str) -> Option<NodeSchema> {
            (kind == GRASP_SCORE).then(|| NodeSchema {
                kind: GRASP_SCORE,
                inputs: vec![Port::new("value", scalar())],
                outputs: vec![],
                params: vec![
                    ParamSchema {
                        name: "name".to_owned(),
                        ty: ParamType::String,
                        required: false,
                        default: Some(toml::Value::String("grasp".to_owned())),
                    },
                    ParamSchema {
                        name: "weight".to_owned(),
                        ty: ParamType::Float,
                        required: false,
                        default: Some(toml::Value::Float(1.0)),
                    },
                ],
            })
        }
    }

    // --- fixtures ----------------------------------------------------------------------------

    fn empty_task() -> EditIr {
        EditIr::Task(TaskIr {
            schema_version: 1,
            scene: SceneRef {
                path: "scenes/table.xml".to_owned(),
                scene_hash: [0; 32],
                asset_hash: [0; 32],
            },
            graph: Graph::new(1),
            observation_spec: ObservationSpec::default(),
            config: TaskConfig {
                max_episode_steps: 200,
                control_rate_hz: 30.0,
                deterministic: true,
                rng_streams: BTreeSet::new(),
            },
        })
    }

    fn session() -> EditSession {
        EditSession::new(empty_task(), Layout::default())
    }

    fn palette(session: &EditSession) -> Palette {
        Palette::from_registries(&session.registries)
    }

    /// Adds `kind` with the palette's starting parameters and returns the new id.
    #[track_caller]
    fn add(session: &mut EditSession, kind: &str, pos: [f32; 2]) -> NodeId {
        let params = palette(session)
            .defaults(kind)
            .expect("kind is in the palette");
        let id = session.graph.next_id();
        session
            .apply(Edit::AddNode {
                kind: kind.to_owned(),
                params,
                pos,
            })
            .expect("the node is added");
        id
    }

    fn port(node: NodeId, name: &str) -> PortRef {
        PortRef::new(node, name)
    }

    // --- edits -------------------------------------------------------------------------------

    #[test]
    fn add_node_uses_the_registry_and_records_layout() {
        let mut s = session();
        let id = add(&mut s, "GetRandom", [10.0, 20.0]);
        assert_eq!(s.graph.nodes(), vec![(id, "GetRandom")]);
        assert_eq!(
            s.layout.positions.get(&id).map(|p| p[0].to_bits()),
            Some(10.0f32.to_bits())
        );
    }

    #[test]
    fn unknown_kind_is_factory_001() {
        let mut s = session();
        let err = s
            .apply(Edit::AddNode {
                kind: "NotANode".to_owned(),
                params: toml::Value::Table(toml::Table::new()),
                pos: [0.0, 0.0],
            })
            .unwrap_err();
        assert_eq!(err[0].code.as_str(), codes::FACTORY_001);
        assert!(s.graph.nodes().is_empty(), "the edit was not applied");
    }

    #[test]
    fn a_valid_connect_is_applied() {
        let mut s = session();
        let src = add(&mut s, "GetRandom", [0.0, 0.0]);
        let dst = add(&mut s, "Reward", [220.0, 0.0]);
        s.apply(Edit::Connect {
            from: port(src, "value"),
            to: port(dst, "value"),
        })
        .expect("f32[1] dimensionless feeds f32[1] dimensionless");
        assert_eq!(s.graph.edges().len(), 1);
    }

    #[test]
    fn a_type_mismatched_connect_is_refused() {
        let mut s = session();
        // GetTime emits `Unit::Time`; Reward's input is dimensionless.
        let src = add(&mut s, "GetTime", [0.0, 0.0]);
        let dst = add(&mut s, "Reward", [220.0, 0.0]);
        let before = s.hash().expect("hashes");
        let history = s.history().len();
        let err = s
            .apply(Edit::Connect {
                from: port(src, "value"),
                to: port(dst, "value"),
            })
            .unwrap_err();
        assert_eq!(err[0].code.as_str(), codes::TYPE_003, "{err:?}");
        assert_eq!(err[0].node, Some(dst));
        assert!(s.graph.edges().is_empty(), "the edge was not kept");
        assert_eq!(s.hash().expect("hashes"), before, "state is untouched");
        assert_eq!(s.history().len(), history, "a refused edit is not history");
    }

    #[test]
    fn an_unknown_port_is_refused() {
        let mut s = session();
        let src = add(&mut s, "GetRandom", [0.0, 0.0]);
        let dst = add(&mut s, "Reward", [220.0, 0.0]);
        let err = s
            .apply(Edit::Connect {
                from: port(src, "nope"),
                to: port(dst, "value"),
            })
            .unwrap_err();
        assert_eq!(err[0].code.as_str(), codes::GRAPH_010);
    }

    #[test]
    fn disconnect_removes_the_edge_and_a_missing_one_is_refused() {
        let mut s = session();
        let src = add(&mut s, "GetRandom", [0.0, 0.0]);
        let dst = add(&mut s, "Reward", [220.0, 0.0]);
        let (from, to) = (port(src, "value"), port(dst, "value"));
        s.apply(Edit::Connect {
            from: from.clone(),
            to: to.clone(),
        })
        .expect("connects");
        s.apply(Edit::Disconnect {
            from: from.clone(),
            to: to.clone(),
        })
        .expect("disconnects");
        assert!(s.graph.edges().is_empty());
        assert!(s.apply(Edit::Disconnect { from, to }).is_err());
    }

    #[test]
    fn remove_node_takes_its_edges_with_it() {
        let mut s = session();
        let src = add(&mut s, "GetRandom", [0.0, 0.0]);
        let dst = add(&mut s, "Reward", [220.0, 0.0]);
        s.apply(Edit::Connect {
            from: port(src, "value"),
            to: port(dst, "value"),
        })
        .expect("connects");
        s.apply(Edit::RemoveNode { node: src }).expect("removes");
        assert_eq!(s.graph.nodes(), vec![(dst, "Reward")]);
        assert!(s.graph.edges().is_empty(), "no dangling edge is left");
        assert!(!s.layout.positions.contains_key(&src));
    }

    #[test]
    fn set_param_rebuilds_the_node_through_the_registry() {
        let mut s = session();
        let id = add(&mut s, "Reward", [0.0, 0.0]);
        let before = s.hash().expect("hashes");
        s.apply(Edit::SetParam {
            node: id,
            key: "weight".to_owned(),
            value: toml::Value::Float(2.5),
        })
        .expect("weight is a Reward parameter");
        assert_ne!(s.hash().expect("hashes"), before, "a parameter is hashed");
        let EditIr::Task(ir) = &s.graph else {
            unreachable!()
        };
        let TaskNode::Reward { weight, .. } = &ir.graph.nodes[&id] else {
            panic!("still a Reward")
        };
        assert!((*weight - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn an_unknown_param_key_is_factory_003() {
        let mut s = session();
        let id = add(&mut s, "Reward", [0.0, 0.0]);
        let err = s
            .apply(Edit::SetParam {
                node: id,
                key: "nope".to_owned(),
                value: toml::Value::Float(1.0),
            })
            .unwrap_err();
        assert_eq!(err[0].code.as_str(), codes::FACTORY_003);
    }

    // --- the layout / hash boundary (spec 4.2 rule 7) -----------------------------------------

    #[test]
    fn move_node_does_not_move_the_hash() {
        let mut s = session();
        let id = add(&mut s, "GetRandom", [0.0, 0.0]);
        let before = s.hash().expect("hashes");
        let graph_before = s.graph.clone();
        for pos in [[100.0, 50.0], [-7.5, 3.25], [0.0, 0.0]] {
            s.apply(Edit::MoveNode { node: id, pos })
                .expect("the node exists");
            assert_eq!(s.hash().expect("hashes"), before, "layout is not IR");
        }
        assert_eq!(s.graph, graph_before, "the IR is untouched");
        assert_eq!(s.layout.positions[&id][0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn moving_a_missing_node_is_refused() {
        let mut s = session();
        assert!(s
            .apply(Edit::MoveNode {
                node: NodeId(9),
                pos: [0.0, 0.0],
            })
            .is_err());
    }

    // --- undo / redo --------------------------------------------------------------------------

    #[test]
    fn undo_and_redo_restore_the_exact_state_and_hash() {
        let mut s = session();
        let src = add(&mut s, "GetRandom", [0.0, 0.0]);
        let dst = add(&mut s, "Reward", [220.0, 0.0]);
        let before = (s.graph.clone(), s.layout.clone(), s.hash().expect("hashes"));

        s.apply(Edit::Connect {
            from: port(src, "value"),
            to: port(dst, "value"),
        })
        .expect("connects");
        s.apply(Edit::SetParam {
            node: dst,
            key: "weight".to_owned(),
            value: toml::Value::Float(3.0),
        })
        .expect("sets");
        s.apply(Edit::MoveNode {
            node: dst,
            pos: [1.0, 2.0],
        })
        .expect("moves");
        let after = (s.graph.clone(), s.layout.clone(), s.hash().expect("hashes"));

        for _ in 0..3 {
            assert!(s.undo());
        }
        assert_eq!((s.graph.clone(), s.layout.clone()), (before.0, before.1));
        assert_eq!(s.hash().expect("hashes"), before.2);

        for _ in 0..3 {
            assert!(s.redo());
        }
        assert!(!s.redo(), "the redo stack is empty");
        assert_eq!((s.graph.clone(), s.layout.clone()), (after.0, after.1));
        assert_eq!(s.hash().expect("hashes"), after.2);
    }

    #[test]
    fn a_new_edit_clears_the_redo_stack() {
        let mut s = session();
        add(&mut s, "GetRandom", [0.0, 0.0]);
        assert!(s.undo());
        assert!(s.can_redo());
        add(&mut s, "GetTime", [0.0, 0.0]);
        assert!(!s.can_redo());
    }

    // --- save / load ---------------------------------------------------------------------------

    #[test]
    fn save_then_load_round_trips_the_graph_and_the_layout() {
        let mut s = session();
        let src = add(&mut s, "GetRandom", [10.0, 20.0]);
        let dst = add(&mut s, "Reward", [230.0, 20.0]);
        s.apply(Edit::Connect {
            from: port(src, "value"),
            to: port(dst, "value"),
        })
        .expect("connects");
        s.apply(Edit::SetParam {
            node: dst,
            key: "weight".to_owned(),
            value: toml::Value::Float(0.5),
        })
        .expect("sets");

        let (graph_toml, layout_toml) = s.save().expect("writes");
        let loaded = EditSession::load(&graph_toml, Some(&layout_toml)).expect("reads");
        assert_eq!(loaded.graph, s.graph);
        assert_eq!(loaded.layout, s.layout);
        assert_eq!(loaded.hash().expect("hashes"), s.hash().expect("hashes"));
    }

    #[test]
    fn the_layout_sidecar_cannot_reach_the_ir_hash() {
        let mut s = session();
        let id = add(&mut s, "GetRandom", [10.0, 20.0]);
        let (graph_toml, _) = s.save().expect("writes");
        s.apply(Edit::MoveNode {
            node: id,
            pos: [999.0, -999.0],
        })
        .expect("moves");
        let (moved_toml, layout_toml) = s.save().expect("writes");
        assert_eq!(graph_toml, moved_toml, "the .esgraph half is identical");
        let without = EditSession::load(&graph_toml, None).expect("reads");
        let with = EditSession::load(&moved_toml, Some(&layout_toml)).expect("reads");
        assert_eq!(
            without.hash().expect("hashes"),
            with.hash().expect("hashes")
        );
        assert!(without.layout.positions.is_empty());
    }

    #[test]
    fn a_record_ir_is_not_an_editable_graph() {
        let err = EditError::NotAGraph(IrKind::Deployment);
        assert!(err.to_string().contains("record"), "{err}");
    }

    // --- the Node SDK --------------------------------------------------------------------------

    #[test]
    fn a_custom_factory_reaches_the_palette_and_builds_a_node() {
        let mut s = session();
        s.registries
            .register_task(Box::new(GraspScoreNodes))
            .expect("the kind is free");
        let p = palette(&s);
        let entry = p.get(GRASP_SCORE).expect("the custom kind is listed");
        assert_eq!(entry.category, "Task / custom");
        assert_eq!(entry.schema.inputs.len(), 1);

        let id = add(&mut s, GRASP_SCORE, [0.0, 0.0]);
        // The factory expands to a builtin node, so the *graph* records the frozen kind tag.
        assert_eq!(s.graph.nodes(), vec![(id, "Reward")]);
    }

    #[test]
    fn a_custom_factory_cannot_steal_a_builtin_kind() {
        #[derive(Debug)]
        struct Impostor;
        impl TaskNodeFactory for Impostor {
            fn kinds(&self) -> &[&'static str] {
                &["Reward"]
            }
            fn create(&self, _: &str, _: &toml::Value) -> Result<TaskNode, Diagnostic> {
                unreachable!("registration fails first")
            }
            fn schema(&self, _: &str) -> Option<NodeSchema> {
                None
            }
        }
        let mut s = session();
        let err = s.registries.register_task(Box::new(Impostor)).unwrap_err();
        assert_eq!(err.code.as_str(), codes::FACTORY_002);
    }

    #[test]
    fn observation_nodes_have_no_factory() {
        let obs = EditIr::Observation(ObservationIr::new(1, [0; 32]));
        let mut s = EditSession::new(obs, Layout::default());
        let err = s
            .apply(Edit::AddNode {
                kind: "ImageInput".to_owned(),
                params: toml::Value::Table(toml::Table::new()),
                pos: [0.0, 0.0],
            })
            .unwrap_err();
        assert_eq!(err[0].code.as_str(), codes::FACTORY_001);
    }

    // --- helpers -------------------------------------------------------------------------------

    #[test]
    fn port_names_come_from_the_node_not_the_palette() {
        let mut s = session();
        let id = add(&mut s, "Reward", [0.0, 0.0]);
        assert_eq!(s.graph.port_names(id, es_ir::Dir::In), vec!["value"]);
        assert!(s.graph.port_names(id, es_ir::Dir::Out).is_empty());
        assert!(s.graph.port_names(NodeId(99), es_ir::Dir::In).is_empty());
    }

    #[test]
    fn missing_positions_are_filled_deterministically() {
        let mut s = session();
        add(&mut s, "GetRandom", [0.0, 0.0]);
        s.layout.positions.clear();
        let mut a = Layout::default();
        let mut b = Layout::default();
        fill_missing_positions(&s.graph, &mut a, 4);
        fill_missing_positions(&s.graph, &mut b, 4);
        assert_eq!(a, b);
        assert_eq!(a.positions.len(), 1);
        assert_eq!(kinds_by_id(&s.graph).len(), 1);
    }
}
