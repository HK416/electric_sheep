//! Serialization (P28): TOML file format for the five typed IRs, `.esgraph` graph files, and
//! `.eslayout` sidecars (spec 14.3, spec 4.2 rule 7).
//!
//! **Layout lives here and only here.** [`Layout`] is never a field of any IR type; an
//! `.esgraph` parses to the identical IR, and its `*_hash` is identical, whether or not an
//! `.eslayout` sidecar is given alongside it.
//!
//! `Graph<N>::nodes` is a `BTreeMap<NodeId, N>`, but `NodeId`'s derived `Serialize` writes a
//! bare integer, and TOML tables require string keys. Rather than change `NodeId` or `Graph`
//! (owned by other packets), every node-keyed map is re-keyed through a decimal string for the
//! TOML mirror of that IR and parsed back on the way in (see `GraphToml`, `node_map_to_toml`,
//! `node_map_from_toml`, and `Layout`'s own hand-written `Serialize`/`Deserialize`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::deployment::DeploymentIr;
use crate::evaluation::EvaluationIr;
use crate::graph::{Edge, Graph, NodeId, PortRef};
use crate::learning::{LearningGraph, LearningNode, PolicyHandle, TensorPort};
use crate::observation::{ObservationIr, ObservationNode, ObservationOutput, TemporalModel};
use crate::task::{ObservationSpec, SceneRef, TaskConfig, TaskGraph, TaskIr, TaskNode};

/// Format version of the envelope itself (independent of each IR's own `schema_version`).
pub const ES_SCHEMA_VERSION: u32 = 1;

/// Which of the five typed IRs a file holds, so it self-identifies (spec 14.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrKind {
    Task,
    Observation,
    Learning,
    Deployment,
    Evaluation,
}

/// A self-identifying TOML document: the format version plus which IR `body` is.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IrFile<T> {
    pub es_schema: u32,
    pub kind: IrKind,
    pub body: T,
}

/// Errors from this module. `es-ir` has no `thiserror` dependency, so this is a plain enum
/// with a hand-written `Display`.
#[derive(Debug)]
pub enum SerialError {
    Ser(toml::ser::Error),
    De(toml::de::Error),
    /// TOML can represent `nan`/`inf` as bare literals, but this format forbids them.
    NonFiniteFloat,
    BadNodeId(String),
    KindMismatch {
        expected: IrKind,
        found: IrKind,
    },
}

impl fmt::Display for SerialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ser(e) => write!(f, "TOML serialization failed: {e}"),
            Self::De(e) => write!(f, "TOML parse failed: {e}"),
            Self::NonFiniteFloat => {
                write!(
                    f,
                    "value has a NaN or infinite float; this format requires finite floats"
                )
            }
            Self::BadNodeId(s) => write!(f, "'{s}' is not a valid node id"),
            Self::KindMismatch { expected, found } => {
                write!(f, "expected a {expected:?} IR file, found {found:?}")
            }
        }
    }
}

impl std::error::Error for SerialError {}

impl From<toml::ser::Error> for SerialError {
    fn from(e: toml::ser::Error) -> Self {
        Self::Ser(e)
    }
}

impl From<toml::de::Error> for SerialError {
    fn from(e: toml::de::Error) -> Self {
        Self::De(e)
    }
}

/// Serializes any value to a TOML string, rejecting non-finite floats anywhere in it.
pub fn write_toml<T: Serialize>(value: &T) -> Result<String, SerialError> {
    let value = toml::Value::try_from(value)?;
    if !floats_finite(&value) {
        return Err(SerialError::NonFiniteFloat);
    }
    Ok(toml::to_string_pretty(&value)?)
}

fn floats_finite(v: &toml::Value) -> bool {
    match v {
        toml::Value::Float(f) => f.is_finite(),
        toml::Value::Array(a) => a.iter().all(floats_finite),
        toml::Value::Table(t) => t.values().all(floats_finite),
        _ => true,
    }
}

/// Parses a TOML string into any `Deserialize` type.
pub fn parse_toml<T: DeserializeOwned>(s: &str) -> Result<T, SerialError> {
    Ok(toml::from_str(s)?)
}

fn write_envelope<T: Serialize>(kind: IrKind, body: &T) -> Result<String, SerialError> {
    write_toml(&IrFile {
        es_schema: ES_SCHEMA_VERSION,
        kind,
        body,
    })
}

/// Checks `kind` against the envelope before decoding `body`, so a mismatched file reports
/// `KindMismatch` instead of an unrelated shape error.
fn read_envelope<T: DeserializeOwned>(s: &str, expected: IrKind) -> Result<T, SerialError> {
    let probe: IrFile<toml::Value> = parse_toml(s)?;
    if probe.kind != expected {
        return Err(SerialError::KindMismatch {
            expected,
            found: probe.kind,
        });
    }
    Ok(parse_toml::<IrFile<T>>(s)?.body)
}

// --- NodeId-keyed maps: the TOML adapter ----------------------------------------------------

fn node_map_to_toml<V: Clone>(m: &BTreeMap<NodeId, V>) -> BTreeMap<String, V> {
    m.iter()
        .map(|(id, v)| (id.0.to_string(), v.clone()))
        .collect()
}

fn node_map_from_toml<V>(m: BTreeMap<String, V>) -> Result<BTreeMap<NodeId, V>, SerialError> {
    m.into_iter()
        .map(|(k, v)| {
            k.parse::<u32>()
                .map(|n| (NodeId(n), v))
                .map_err(|_| SerialError::BadNodeId(k.clone()))
        })
        .collect()
}

/// TOML-safe mirror of `Graph<N>`: `nodes` re-keyed from `NodeId` to its decimal string.
#[derive(Serialize, Deserialize)]
struct GraphToml<N> {
    schema_version: u32,
    nodes: BTreeMap<String, N>,
    edges: Vec<Edge>,
    inputs: Vec<PortRef>,
    outputs: Vec<PortRef>,
}

impl<N: Clone> From<&Graph<N>> for GraphToml<N> {
    fn from(g: &Graph<N>) -> Self {
        Self {
            schema_version: g.schema_version,
            nodes: node_map_to_toml(&g.nodes),
            edges: g.edges.clone(),
            inputs: g.inputs.clone(),
            outputs: g.outputs.clone(),
        }
    }
}

impl<N> TryFrom<GraphToml<N>> for Graph<N> {
    type Error = SerialError;

    fn try_from(g: GraphToml<N>) -> Result<Self, SerialError> {
        Ok(Graph {
            schema_version: g.schema_version,
            nodes: node_map_from_toml(g.nodes)?,
            edges: g.edges,
            inputs: g.inputs,
            outputs: g.outputs,
        })
    }
}

// --- Per-IR TOML mirrors ---------------------------------------------------------------------
// Task, Observation and Learning each carry a `Graph<N>` and need the mirror above. Deployment
// and Evaluation have no node-keyed map, so they serialize directly with no mirror at all.

#[derive(Serialize, Deserialize)]
struct TaskIrToml {
    schema_version: u32,
    scene: SceneRef,
    graph: GraphToml<TaskNode>,
    observation_spec: ObservationSpec,
    config: TaskConfig,
}

impl From<&TaskIr> for TaskIrToml {
    fn from(ir: &TaskIr) -> Self {
        Self {
            schema_version: ir.schema_version,
            scene: ir.scene.clone(),
            graph: GraphToml::from(&ir.graph),
            observation_spec: ir.observation_spec.clone(),
            config: ir.config.clone(),
        }
    }
}

impl TryFrom<TaskIrToml> for TaskIr {
    type Error = SerialError;

    fn try_from(m: TaskIrToml) -> Result<Self, SerialError> {
        Ok(TaskIr {
            schema_version: m.schema_version,
            scene: m.scene,
            graph: TaskGraph::try_from(m.graph)?,
            observation_spec: m.observation_spec,
            config: m.config,
        })
    }
}

/// Writes a Task IR to a self-identifying TOML string.
pub fn task_to_toml(ir: &TaskIr) -> Result<String, SerialError> {
    write_envelope(IrKind::Task, &TaskIrToml::from(ir))
}

/// Parses a Task IR TOML string. Errors with `KindMismatch` if the file is a different IR.
pub fn task_from_toml(s: &str) -> Result<TaskIr, SerialError> {
    TaskIr::try_from(read_envelope::<TaskIrToml>(s, IrKind::Task)?)
}

#[derive(Serialize, Deserialize)]
struct ObservationIrToml {
    schema_version: u32,
    task_ref: [u8; 32],
    graph: GraphToml<ObservationNode>,
    temporal: TemporalModel,
    outputs: BTreeMap<String, ObservationOutput>,
}

impl From<&ObservationIr> for ObservationIrToml {
    fn from(ir: &ObservationIr) -> Self {
        Self {
            schema_version: ir.schema_version,
            task_ref: ir.task_ref,
            graph: GraphToml::from(&ir.graph),
            temporal: ir.temporal.clone(),
            outputs: ir.outputs.clone(),
        }
    }
}

impl TryFrom<ObservationIrToml> for ObservationIr {
    type Error = SerialError;

    fn try_from(m: ObservationIrToml) -> Result<Self, SerialError> {
        Ok(ObservationIr {
            schema_version: m.schema_version,
            task_ref: m.task_ref,
            graph: Graph::try_from(m.graph)?,
            temporal: m.temporal,
            outputs: m.outputs,
        })
    }
}

/// Writes an Observation IR to a self-identifying TOML string.
pub fn observation_to_toml(ir: &ObservationIr) -> Result<String, SerialError> {
    write_envelope(IrKind::Observation, &ObservationIrToml::from(ir))
}

/// Parses an Observation IR TOML string. Errors with `KindMismatch` if the file is a different
/// IR.
pub fn observation_from_toml(s: &str) -> Result<ObservationIr, SerialError> {
    ObservationIr::try_from(read_envelope::<ObservationIrToml>(s, IrKind::Observation)?)
}

#[derive(Serialize, Deserialize)]
struct LearningGraphToml {
    schema_version: u32,
    inputs: Vec<TensorPort>,
    nodes: GraphToml<LearningNode>,
    outputs: Vec<TensorPort>,
    policy: PolicyHandle,
}

impl From<&LearningGraph> for LearningGraphToml {
    fn from(ir: &LearningGraph) -> Self {
        Self {
            schema_version: ir.schema_version,
            inputs: ir.inputs.clone(),
            nodes: GraphToml::from(&ir.nodes),
            outputs: ir.outputs.clone(),
            policy: ir.policy.clone(),
        }
    }
}

impl TryFrom<LearningGraphToml> for LearningGraph {
    type Error = SerialError;

    fn try_from(m: LearningGraphToml) -> Result<Self, SerialError> {
        Ok(LearningGraph {
            schema_version: m.schema_version,
            inputs: m.inputs,
            nodes: Graph::try_from(m.nodes)?,
            outputs: m.outputs,
            policy: m.policy,
        })
    }
}

/// Writes a Learning IR to a self-identifying TOML string.
pub fn learning_to_toml(ir: &LearningGraph) -> Result<String, SerialError> {
    write_envelope(IrKind::Learning, &LearningGraphToml::from(ir))
}

/// Parses a Learning IR TOML string. Errors with `KindMismatch` if the file is a different IR.
pub fn learning_from_toml(s: &str) -> Result<LearningGraph, SerialError> {
    LearningGraph::try_from(read_envelope::<LearningGraphToml>(s, IrKind::Learning)?)
}

/// Writes a Deployment IR to a self-identifying TOML string. No `NodeId` map, so no mirror.
pub fn deployment_to_toml(ir: &DeploymentIr) -> Result<String, SerialError> {
    write_envelope(IrKind::Deployment, ir)
}

/// Parses a Deployment IR TOML string. Errors with `KindMismatch` if the file is a different
/// IR.
pub fn deployment_from_toml(s: &str) -> Result<DeploymentIr, SerialError> {
    read_envelope(s, IrKind::Deployment)
}

/// Writes an Evaluation IR to a self-identifying TOML string. No `NodeId` map, so no mirror.
pub fn evaluation_to_toml(ir: &EvaluationIr) -> Result<String, SerialError> {
    write_envelope(IrKind::Evaluation, ir)
}

/// Parses an Evaluation IR TOML string. Errors with `KindMismatch` if the file is a different
/// IR.
pub fn evaluation_from_toml(s: &str) -> Result<EvaluationIr, SerialError> {
    read_envelope(s, IrKind::Evaluation)
}

// --- `.eslayout` sidecar ----------------------------------------------------------------------

/// Graph-editor metadata: node positions, collapsed state, free-text notes (spec 14.3, spec
/// 23.2). **The only place layout exists** (spec 4.2 rule 7): no IR type has a field like this,
/// and an IR's `*_hash` never depends on it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layout {
    pub positions: BTreeMap<NodeId, [f32; 2]>,
    pub collapsed: BTreeSet<NodeId>,
    pub notes: BTreeMap<NodeId, String>,
}

#[derive(Serialize, Deserialize)]
struct LayoutToml {
    positions: BTreeMap<String, [f32; 2]>,
    collapsed: BTreeSet<NodeId>,
    notes: BTreeMap<String, String>,
}

impl From<&Layout> for LayoutToml {
    fn from(l: &Layout) -> Self {
        Self {
            positions: node_map_to_toml(&l.positions),
            collapsed: l.collapsed.clone(),
            notes: node_map_to_toml(&l.notes),
        }
    }
}

impl TryFrom<LayoutToml> for Layout {
    type Error = SerialError;

    fn try_from(m: LayoutToml) -> Result<Self, SerialError> {
        Ok(Layout {
            positions: node_map_from_toml(m.positions)?,
            collapsed: m.collapsed,
            notes: node_map_from_toml(m.notes)?,
        })
    }
}

// `collapsed` (a `BTreeSet`, serialized as a plain array) has no map-key problem; only the two
// `BTreeMap`s need the string-keyed detour, so `Layout` gets hand-written impls instead of a
// derive.
impl Serialize for Layout {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        LayoutToml::from(self).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Layout {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mirror = LayoutToml::deserialize(d)?;
        Layout::try_from(mirror).map_err(serde::de::Error::custom)
    }
}

// --- `.esgraph` = IR + optional `.eslayout` ---------------------------------------------------

/// One of the five typed IRs, tagged by its own [`IrKind`] so a `.esgraph` file self-identifies.
#[derive(Clone, Debug, PartialEq)]
pub enum AnyIr {
    Task(TaskIr),
    Observation(ObservationIr),
    Learning(LearningGraph),
    Deployment(DeploymentIr),
    Evaluation(EvaluationIr),
}

impl AnyIr {
    pub fn kind(&self) -> IrKind {
        match self {
            Self::Task(_) => IrKind::Task,
            Self::Observation(_) => IrKind::Observation,
            Self::Learning(_) => IrKind::Learning,
            Self::Deployment(_) => IrKind::Deployment,
            Self::Evaluation(_) => IrKind::Evaluation,
        }
    }
}

/// Writes the `.esgraph` body (the IR) and, if given, the `.eslayout` sidecar. The layout is
/// never mixed into the IR string: [`parse_esgraph`] reconstructs the identical IR whether or
/// not one is supplied.
pub fn write_esgraph(
    ir: &AnyIr,
    layout: Option<&Layout>,
) -> Result<(String, Option<String>), SerialError> {
    let graph = match ir {
        AnyIr::Task(t) => task_to_toml(t),
        AnyIr::Observation(o) => observation_to_toml(o),
        AnyIr::Learning(l) => learning_to_toml(l),
        AnyIr::Deployment(d) => deployment_to_toml(d),
        AnyIr::Evaluation(e) => evaluation_to_toml(e),
    }?;
    let layout = layout.map(write_toml).transpose()?;
    Ok((graph, layout))
}

/// Parses a `.esgraph` body and, if given, its `.eslayout` sidecar. The IR half is decoded
/// purely from `graph_toml`; `layout_toml` never influences it.
pub fn parse_esgraph(
    graph_toml: &str,
    layout_toml: Option<&str>,
) -> Result<(AnyIr, Option<Layout>), SerialError> {
    let probe: IrFile<toml::Value> = parse_toml(graph_toml)?;
    let ir = match probe.kind {
        IrKind::Task => AnyIr::Task(task_from_toml(graph_toml)?),
        IrKind::Observation => AnyIr::Observation(observation_from_toml(graph_toml)?),
        IrKind::Learning => AnyIr::Learning(learning_from_toml(graph_toml)?),
        IrKind::Deployment => AnyIr::Deployment(deployment_from_toml(graph_toml)?),
        IrKind::Evaluation => AnyIr::Evaluation(evaluation_from_toml(graph_toml)?),
    };
    let layout = layout_toml.map(parse_toml).transpose()?;
    Ok((ir, layout))
}
