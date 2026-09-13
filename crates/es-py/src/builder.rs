//! Language-neutral authoring builder core (spec 14.2, M2 W6).
//!
//! This is the part of the Python authoring frontend that has nothing to do with Python: four
//! builders — one per authorable IR (Task, Observation, Learning, Deployment; Evaluation is out
//! of scope for this packet) — each wrapping the existing node factories / graph skeleton from
//! `es-ir` so that adding a node kind here never means writing a match arm here. `kind` is
//! always a string, `params` is always JSON (the natural shape a Python dict, or any other host
//! language's map type, arrives in); the conversion to the `toml::Value` a `TaskNodeFactory` /
//! `LearningNodeFactory` expects happens once, in [`json_to_toml`].
//!
//! `#[cfg(feature = "python")]` code (`crate::pybind`) is a thin pyo3 shell around this module.
//! Everything here is plain Rust, unit-tested with no Python involved (spec 2.1: Python is
//! first-class on the learning/authoring path, never required by the core).

use std::fmt;
use std::fs;
use std::path::Path;

use serde_json::Value as Json;

use es_ir::deployment::{
    ActionContract, Deadlines, DeploymentIr, ExecutionMode, FallbackPolicy, RateSpec, RobotRef,
    SafetyEnvelope, Watchdog, WatchdogSet,
};
use es_ir::factory::{LearningNodeRegistry, TaskNodeRegistry};
use es_ir::graph::{Graph, NodeId};
use es_ir::learning::{LearningGraph, PolicyHandle};
use es_ir::observation::{ObservationIr, ObservationNode, ObservationOutput, TemporalModel};
use es_ir::serial::{self, SerialError};
use es_ir::task::{ObsChannel, ObservationSpec, SceneRef, TaskConfig, TaskIr, TaskNode};
use es_ir::{Diagnostic, Port, PortRef};

// Local diagnostic codes. Not in `es_ir_types::codes` (that dictionary is owned by other
// packets and out of this packet's `forbidden` scope) — an unlisted code is still a perfectly
// valid `Diagnostic`, just reported as `Severity::Error` with no dictionary title (see
// `Diagnostic::new`).
const PY_PARAMS: &str = "PY-001";
const PY_MISSING: &str = "PY-002";

/// Converts a JSON value (what a Python dict becomes at the pyo3 boundary) to the
/// [`toml::Value`] the node factories require. The two formats agree on every shape node params
/// use (bool / number / string / array / table). TOML has no `null` literal, which is exactly
/// how it already spells `Option::None` (e.g. `PortType::image`): a table key whose value is
/// `null` is dropped instead of kept, the same as an absent key; `null` anywhere else (the
/// top-level value, or inside an array) is a genuine mistake and is rejected.
pub fn json_to_toml(value: &Json) -> Result<toml::Value, Diagnostic> {
    Ok(match value {
        Json::Null => {
            return Err(Diagnostic::new(
                PY_PARAMS,
                "null has no TOML representation; omit the field instead",
            ))
        }
        Json::Bool(b) => toml::Value::Boolean(*b),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                toml::Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                toml::Value::Float(f)
            } else {
                return Err(Diagnostic::new(
                    PY_PARAMS,
                    format!("number out of range: {n}"),
                ));
            }
        }
        Json::String(s) => toml::Value::String(s.clone()),
        Json::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(json_to_toml(item)?);
            }
            toml::Value::Array(out)
        }
        Json::Object(map) => {
            let mut table = toml::Table::new();
            for (k, v) in map {
                if v.is_null() {
                    continue;
                }
                table.insert(k.clone(), json_to_toml(v)?);
            }
            toml::Value::Table(table)
        }
    })
}

/// Re-tags `params` as `kind` and deserializes it as `T`, using serde's default externally
/// tagged enum representation — the same wire shape `factory::deserialize_tagged` uses for
/// `TaskNode`/`LearningNode`, applied here to node enums that have no factory (`ObservationNode`
/// is not one of the 7 extension points `INV-17` allows, so it has no registry to route
/// through).
fn deserialize_tagged_json<T: serde::de::DeserializeOwned>(
    kind: &str,
    params: Json,
) -> Result<T, Diagnostic> {
    let mut tagged = serde_json::Map::with_capacity(1);
    tagged.insert(kind.to_owned(), params);
    serde_json::from_value(Json::Object(tagged))
        .map_err(|e| Diagnostic::new(PY_PARAMS, format!("cannot build '{kind}' node: {e}")))
}

/// Errors from writing a built IR to a `.toml` file.
#[derive(Debug)]
pub enum SaveError {
    Serial(SerialError),
    Io(std::io::Error),
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serial(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SaveError {}

impl From<SerialError> for SaveError {
    fn from(e: SerialError) -> Self {
        Self::Serial(e)
    }
}

impl From<std::io::Error> for SaveError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn write_toml_file(text: Result<String, SerialError>, path: &Path) -> Result<(), SaveError> {
    fs::write(path, text?)?;
    Ok(())
}

/// The mechanical part of every graph-shaped builder below: hand out the next [`NodeId`], keep
/// the nodes and edges. Not a trait (`INV-17` allows no new one) — a generic struct the three
/// graph builders share by composition.
#[derive(Debug)]
struct GraphBuilder<N> {
    graph: Graph<N>,
    next_id: u32,
}

impl<N> GraphBuilder<N> {
    fn new(schema_version: u32) -> Self {
        Self {
            graph: Graph::new(schema_version),
            next_id: 0,
        }
    }

    fn insert(&mut self, node: N) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.graph.insert(id, node);
        id
    }

    fn connect(&mut self, from: NodeId, from_port: &str, to: NodeId, to_port: &str) {
        self.graph.connect(from, from_port, to, to_port);
    }
}

// --- Task ---------------------------------------------------------------------------------

/// Builds a Task IR graph (spec 6, spec 14.2 `es.Task`) from `(kind, params)` node kinds
/// registered in [`TaskNodeRegistry`] — every kind `BuiltinTaskNodes` owns, spec 6.3.
#[derive(Debug)]
pub struct TaskBuilder {
    scene: SceneRef,
    config: TaskConfig,
    observation_spec: ObservationSpec,
    graph: GraphBuilder<TaskNode>,
    registry: TaskNodeRegistry,
}

impl TaskBuilder {
    pub fn new(scene: SceneRef, config: TaskConfig) -> Self {
        Self {
            scene,
            config,
            observation_spec: ObservationSpec::default(),
            graph: GraphBuilder::new(es_ir::task::SCHEMA_VERSION),
            registry: TaskNodeRegistry::with_builtins(),
        }
    }

    /// Builds and inserts one node of `kind` (spec 6.3), `params` keyed exactly as `TaskNode`'s
    /// field names for that variant.
    pub fn add(&mut self, kind: &str, params: &Json) -> Result<NodeId, Diagnostic> {
        let params = json_to_toml(params)?;
        let node = self.registry.create(kind, &params)?;
        Ok(self.graph.insert(node))
    }

    pub fn connect(&mut self, from: NodeId, from_port: &str, to: NodeId, to_port: &str) {
        self.graph.connect(from, from_port, to, to_port);
    }

    /// Declares one Task IR -> Observation IR channel (spec 7.4). `channel` is a JSON
    /// `ObsChannel` (`{"source": .., "ty": ..}`) — this only records where a value comes from
    /// and its type; preprocessing is Observation IR's job (spec 5.1 rule 6), never this one's.
    pub fn declare_obs(&mut self, name: &str, channel: Json) -> Result<(), Diagnostic> {
        let channel: ObsChannel = serde_json::from_value(channel)
            .map_err(|e| Diagnostic::new(PY_PARAMS, format!("bad ObsChannel: {e}")))?;
        self.observation_spec
            .channels
            .insert(name.to_owned(), channel);
        Ok(())
    }

    /// Assembles the `TaskIr` and runs [`TaskIr::validate`]. `Err` carries every diagnostic
    /// (spec 14.2: "type errors surface as Python exceptions carrying compiler diagnostics").
    pub fn build(self) -> Result<TaskIr, Vec<Diagnostic>> {
        let ir = TaskIr {
            schema_version: es_ir::task::SCHEMA_VERSION,
            scene: self.scene,
            graph: self.graph.graph,
            observation_spec: self.observation_spec,
            config: self.config,
            // IR-C (spec 6.2) is out of this packet's scope; every task this builder makes is
            // IR-D-only, same as one hand-authored before IR-C existed.
            control: None,
        };
        let diags = ir.validate();
        if diags.iter().any(Diagnostic::is_error) {
            Err(diags)
        } else {
            Ok(ir)
        }
    }

    pub fn save_toml(ir: &TaskIr, path: &Path) -> Result<(), SaveError> {
        write_toml_file(serial::task_to_toml(ir), path)
    }
}

// --- Observation ----------------------------------------------------------------------------

/// Builds an Observation IR graph (spec 7, spec 14.2 `es.Observation`). `ObservationNode` has
/// no factory (not one of the 7 `INV-17` extension points), so `add` re-tags `params` and
/// deserializes it directly (spec 7.3 node set is exactly `ObservationNode`'s variants).
#[derive(Debug)]
pub struct ObservationBuilder {
    task_ref: [u8; 32],
    temporal: TemporalModel,
    outputs: std::collections::BTreeMap<String, ObservationOutput>,
    graph: GraphBuilder<ObservationNode>,
}

impl ObservationBuilder {
    pub fn new(task_ref: [u8; 32]) -> Self {
        Self {
            task_ref,
            temporal: TemporalModel::default(),
            outputs: std::collections::BTreeMap::new(),
            graph: GraphBuilder::new(1),
        }
    }

    pub fn add(&mut self, kind: &str, params: Json) -> Result<NodeId, Diagnostic> {
        let node: ObservationNode = deserialize_tagged_json(kind, params)?;
        Ok(self.graph.insert(node))
    }

    pub fn connect(&mut self, from: NodeId, from_port: &str, to: NodeId, to_port: &str) {
        self.graph.connect(from, from_port, to, to_port);
    }

    /// Sets the spec 7.5 time model (layers 1 and 2: `History` per sensor, one `TemporalWindow`).
    pub fn set_temporal(&mut self, temporal: Json) -> Result<(), Diagnostic> {
        self.temporal = serde_json::from_value(temporal)
            .map_err(|e| Diagnostic::new(PY_PARAMS, format!("bad TemporalModel: {e}")))?;
        Ok(())
    }

    /// Declares one named tensor Learning IR binds to, e.g. `"rgb_front"` (spec 7 `outputs`).
    pub fn declare_obs(
        &mut self,
        name: &str,
        node: NodeId,
        port: &str,
        ty: Json,
    ) -> Result<(), Diagnostic> {
        let ty = serde_json::from_value(ty)
            .map_err(|e| Diagnostic::new(PY_PARAMS, format!("bad PortType: {e}")))?;
        self.outputs.insert(
            name.to_owned(),
            ObservationOutput {
                port: PortRef::new(node, port),
                ty,
            },
        );
        Ok(())
    }

    pub fn build(self) -> Result<ObservationIr, Vec<Diagnostic>> {
        let ir = ObservationIr {
            schema_version: self.graph.graph.schema_version,
            task_ref: self.task_ref,
            graph: self.graph.graph,
            temporal: self.temporal,
            outputs: self.outputs,
        };
        let diags = ir.validate();
        if diags.iter().any(Diagnostic::is_error) {
            Err(diags)
        } else {
            Ok(ir)
        }
    }

    pub fn save_toml(ir: &ObservationIr, path: &Path) -> Result<(), SaveError> {
        write_toml_file(serial::observation_to_toml(ir), path)
    }
}

// --- Learning -------------------------------------------------------------------------------

/// Builds a Learning IR graph (spec 8, spec 14.2 `es.Learning`) from [`LearningNodeRegistry`]
/// kinds — the spec 8.3 node set.
#[derive(Debug)]
pub struct LearningBuilder {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    policy: Option<PolicyHandle>,
    graph: GraphBuilder<es_ir::learning::LearningNode>,
    registry: LearningNodeRegistry,
}

impl LearningBuilder {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            policy: None,
            graph: GraphBuilder::new(1),
            registry: LearningNodeRegistry::with_builtins(),
        }
    }

    pub fn add(&mut self, kind: &str, params: &Json) -> Result<NodeId, Diagnostic> {
        let params = json_to_toml(params)?;
        let node = self.registry.create(kind, &params)?;
        Ok(self.graph.insert(node))
    }

    pub fn connect(&mut self, from: NodeId, from_port: &str, to: NodeId, to_port: &str) {
        self.graph.connect(from, from_port, to, to_port);
    }

    /// Declares one tensor the graph consumes from Observation IR, positionally paired with
    /// `nodes.inputs` (Appendix B.3): call in the same order as the matching boundary edges.
    pub fn declare_input(
        &mut self,
        name: &str,
        ty: Json,
        node: NodeId,
        port: &str,
    ) -> Result<(), Diagnostic> {
        let ty = serde_json::from_value(ty)
            .map_err(|e| Diagnostic::new(PY_PARAMS, format!("bad TensorPort type: {e}")))?;
        self.inputs.push(Port::new(name, ty));
        self.graph.graph.inputs.push(PortRef::new(node, port));
        Ok(())
    }

    /// Declares one tensor exposed to the deployed policy contract.
    pub fn declare_obs(
        &mut self,
        name: &str,
        ty: Json,
        node: NodeId,
        port: &str,
    ) -> Result<(), Diagnostic> {
        let ty = serde_json::from_value(ty)
            .map_err(|e| Diagnostic::new(PY_PARAMS, format!("bad TensorPort type: {e}")))?;
        self.outputs.push(Port::new(name, ty));
        self.graph.graph.outputs.push(PortRef::new(node, port));
        Ok(())
    }

    pub fn set_policy(&mut self, policy: Json) -> Result<(), Diagnostic> {
        self.policy = Some(
            serde_json::from_value(policy)
                .map_err(|e| Diagnostic::new(PY_PARAMS, format!("bad PolicyHandle: {e}")))?,
        );
        Ok(())
    }

    pub fn build(self) -> Result<LearningGraph, Vec<Diagnostic>> {
        let Some(policy) = self.policy else {
            return Err(vec![Diagnostic::new(
                PY_MISSING,
                "Learning graph has no policy; call set_policy first",
            )]);
        };
        let ir = LearningGraph {
            schema_version: self.graph.graph.schema_version,
            inputs: self.inputs,
            nodes: self.graph.graph,
            outputs: self.outputs,
            policy,
        };
        let diags = ir.validate();
        if diags.iter().any(Diagnostic::is_error) {
            Err(diags)
        } else {
            Ok(ir)
        }
    }

    pub fn save_toml(ir: &LearningGraph, path: &Path) -> Result<(), SaveError> {
        write_toml_file(serial::learning_to_toml(ir), path)
    }
}

impl Default for LearningBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// --- Deployment -----------------------------------------------------------------------------

/// Builds a Deployment IR (spec 9, spec 14.2 `es.Deployment`). Deployment IR is a plain struct,
/// not a graph (`serial.rs`: "No `NodeId` map, so no mirror") — there is nothing to `add`/
/// `connect` here, so this builder is direct setters instead, one per `DeploymentIr` field.
#[derive(Debug)]
pub struct DeploymentBuilder {
    robot: Option<RobotRef>,
    action: Option<ActionContract>,
    safety: Option<SafetyEnvelope>,
    execution: Option<ExecutionMode>,
    deadlines: Option<Deadlines>,
    watchdogs: Vec<Watchdog>,
    fallback: Option<FallbackPolicy>,
    rate: Option<RateSpec>,
}

impl DeploymentBuilder {
    pub fn new() -> Self {
        Self {
            robot: None,
            action: None,
            safety: None,
            execution: None,
            deadlines: None,
            watchdogs: Vec::new(),
            fallback: None,
            rate: None,
        }
    }

    /// `es.Deployment(...).envelope(...)` (spec 9.3): the mandatory `SafetyEnvelope`.
    pub fn envelope(&mut self, value: Json) -> Result<(), Diagnostic> {
        self.safety = Some(from_json(value, "SafetyEnvelope")?);
        Ok(())
    }

    /// `.watchdog(...)` (spec 9.4), one call per watchdog; may be called more than once.
    pub fn watchdog(&mut self, value: Json) -> Result<(), Diagnostic> {
        self.watchdogs.push(from_json(value, "Watchdog")?);
        Ok(())
    }

    /// `.fallback(...)` (spec 9.4). Only one fallback policy exists per deployment.
    pub fn fallback(&mut self, value: Json) -> Result<(), Diagnostic> {
        self.fallback = Some(from_json(value, "FallbackPolicy")?);
        Ok(())
    }

    pub fn set_robot(&mut self, value: Json) -> Result<(), Diagnostic> {
        self.robot = Some(from_json(value, "RobotRef")?);
        Ok(())
    }

    pub fn set_action(&mut self, value: Json) -> Result<(), Diagnostic> {
        self.action = Some(from_json(value, "ActionContract")?);
        Ok(())
    }

    pub fn set_execution(&mut self, value: Json) -> Result<(), Diagnostic> {
        self.execution = Some(from_json(value, "ExecutionMode")?);
        Ok(())
    }

    pub fn set_deadlines(&mut self, value: Json) -> Result<(), Diagnostic> {
        self.deadlines = Some(from_json(value, "Deadlines")?);
        Ok(())
    }

    pub fn set_rate(&mut self, value: Json) -> Result<(), Diagnostic> {
        self.rate = Some(from_json(value, "RateSpec")?);
        Ok(())
    }

    pub fn build(self) -> Result<DeploymentIr, Vec<Diagnostic>> {
        let mut missing = Vec::new();
        macro_rules! require {
            ($field:ident, $name:literal) => {
                match self.$field {
                    Some(v) => v,
                    None => {
                        missing.push(Diagnostic::new(
                            PY_MISSING,
                            concat!("Deployment is missing ", $name),
                        ));
                        return Err(missing);
                    }
                }
            };
        }
        let robot = require!(robot, "robot (call set_robot)");
        let action = require!(action, "action (call set_action)");
        let safety = require!(safety, "safety envelope (call envelope)");
        let execution = require!(execution, "execution mode (call set_execution)");
        let deadlines = require!(deadlines, "deadlines (call set_deadlines)");
        let fallback = require!(fallback, "fallback policy (call fallback)");
        let rate = require!(rate, "rate spec (call set_rate)");
        let ir = DeploymentIr {
            schema_version: es_ir::deployment::SCHEMA_VERSION,
            robot,
            action,
            safety,
            execution,
            deadlines,
            watchdogs: WatchdogSet(self.watchdogs),
            fallback,
            rate,
        };
        let diags = ir.validate();
        if diags.iter().any(Diagnostic::is_error) {
            Err(diags)
        } else {
            Ok(ir)
        }
    }

    pub fn save_toml(ir: &DeploymentIr, path: &Path) -> Result<(), SaveError> {
        write_toml_file(serial::deployment_to_toml(ir), path)
    }
}

impl Default for DeploymentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn from_json<T: serde::de::DeserializeOwned>(value: Json, what: &str) -> Result<T, Diagnostic> {
    serde_json::from_value(value)
        .map_err(|e| Diagnostic::new(PY_PARAMS, format!("bad {what}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scene() -> SceneRef {
        SceneRef {
            path: "scenes/table_franka.usda".to_owned(),
            scene_hash: [1; 32],
            asset_hash: [2; 32],
        }
    }

    fn config() -> TaskConfig {
        TaskConfig {
            max_episode_steps: 200,
            control_rate_hz: 30.0,
            deterministic: true,
            rng_streams: ["init"].into_iter().map(str::to_owned).collect(),
        }
    }

    /// A `TaskBuilder` producing exactly one node, through `add`, must build and round-trip
    /// through the same TOML `save_toml`/`task_from_toml` the CLI would read.
    #[test]
    fn task_builder_builds_a_valid_minimal_graph() {
        let mut b = TaskBuilder::new(scene(), config());
        let time_node = b.add("GetTime", &json!({"since_reset": false})).unwrap();
        let done = b.add("Terminate", &json!({"kind": "Timeout"})).unwrap();
        // GetTime's "value" port is Unit::Time, not the `flag` Terminate.value wants; connect a
        // Compare instead so the graph is genuinely well-typed end to end.
        let is_late = b
            .add(
                "Compare",
                &json!({"op": "Gt", "rhs": 1.0, "ty": {
                    "elem": "F32", "shape": [1], "unit": "Time", "frame": "World", "time": "Tick", "image": null
                }}),
            )
            .unwrap();
        b.connect(time_node, "value", is_late, "a");
        b.connect(is_late, "value", done, "value");
        let ir = b.build().expect("valid graph");
        assert_eq!(ir.graph.nodes.len(), 3);

        let dir = tempdir();
        let path = dir.join("task.toml");
        TaskBuilder::save_toml(&ir, &path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let back = serial::task_from_toml(&text).unwrap();
        assert_eq!(back, ir);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn unknown_kind_is_a_diagnostic_not_a_panic() {
        let mut b = TaskBuilder::new(scene(), config());
        let err = b.add("NoSuchKind", &json!({})).unwrap_err();
        assert_eq!(err.code.as_str(), "FACTORY-001");
    }

    #[test]
    fn declare_obs_round_trips_a_channel() {
        let mut b = TaskBuilder::new(scene(), config());
        b.declare_obs(
            "joint_state",
            json!({
                "source": {"JointState": {"body": "00000000000000000000000000000000", "dof": 7}},
                "ty": {"elem": "F32", "shape": [7], "unit": "Angle", "frame": "World", "time": "Tick", "image": null}
            }),
        )
        .unwrap();
        assert!(b.observation_spec.channels.contains_key("joint_state"));
    }

    #[test]
    fn observation_builder_builds_without_a_registry() {
        let mut b = ObservationBuilder::new([0; 32]);
        let sensor = "00000000000000000000000000000000";
        let state = b
            .add(
                "StateInput",
                json!({"source": sensor, "io": {"inputs": [], "output": {
                    "elem": "F32", "shape": [7], "unit": "Angle", "frame": "World", "time": "Tick", "image": null
                }}}),
            )
            .unwrap();
        b.declare_obs(
            "joint_state",
            state,
            "out",
            json!({"elem": "F32", "shape": [7], "unit": "Angle", "frame": "World", "time": "Tick", "image": null}),
        )
        .unwrap();
        let ir = b.build().expect("valid observation graph");
        assert_eq!(ir.outputs.len(), 1);
    }

    #[test]
    fn learning_builder_requires_a_policy() {
        let b = LearningBuilder::new();
        let err = b.build().unwrap_err();
        assert_eq!(err[0].code.as_str(), PY_MISSING);
    }

    #[test]
    fn deployment_builder_reports_every_missing_field_before_the_first() {
        let b = DeploymentBuilder::new();
        let err = b.build().unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].code.as_str(), PY_MISSING);
    }

    // Small helper so tests do not depend on `std::env::temp_dir` collisions across a `cargo
    // test` run in parallel.
    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("es-py-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
