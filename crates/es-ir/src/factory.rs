//! Node factories (P29): `TaskNodeFactory` and `LearningNodeFactory` extension points.
//!
//! These are 2 of the 7 extension points the spec allows at all (`INV-17`); no other trait may
//! be added here. A factory turns `(kind, params)` into a node; [`NodeSchema`] tells an editor
//! or an LLM generator (spec 14.5) what kinds exist and what each one takes, without it having
//! to link the node enums at all. [`BuiltinTaskNodes`] / [`BuiltinLearningNodes`] cover the
//! spec 6.3 / spec 8.3 node sets; [`BUILTIN_TASK_KINDS`] / [`BUILTIN_LEARNING_KINDS`] pin their
//! kind strings so a rename is caught here rather than silently changing `*_hash` (spec 28.7
//! gate 10). [`BUILTIN_TASK_KINDS_HASH`] / [`BUILTIN_LEARNING_KINDS_HASH`] make that machine
//! checked (P-M1-R6): a test recomputes the hash from the live lists and fails the build if
//! either list moved.
//!
//! `params` is a `toml::Value` table of the node's own fields (no `kind` key); a factory adds
//! the tag itself. Both node enums derive `Serialize`/`Deserialize` with serde's default
//! externally tagged representation (`{"Kind": {..fields}}`), so building and reading one back
//! is a plain re-tag, not a hand-written mapping per kind.

use std::collections::BTreeMap;
use std::fmt;

use es_core::StableId;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::codes;
use crate::diag::Diagnostic;
use crate::graph::{IrNode, Port};
use crate::learning::{
    ActionExecutionMode, ChunkBlendPolicy, FusionKind, HeadKind, LearningNode, NormalizeDir,
    StateEncoderKind, StatsSource, TemporalKind, VisionBackbone, WeightsRef,
};
use crate::task::{
    ActionSpace, Aggregation, ArithOp, CmpOp, Distribution, JointQuantity, LogicOp, MathFunc,
    NormKind, ReduceOp, TaskNode, TerminationKind,
};
use crate::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

// The `FACTORY-00x` codes this module reports (severity and title in `codes.rs`):
// FACTORY-001 unknown node kind
// FACTORY-002 kind already owned by another registered factory
// FACTORY-003 params do not match the node's shape

/// One parameter an editor or generator needs to fill in.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamType {
    Bool,
    Int,
    Float,
    String,
    Enum(Vec<String>),
    Shape,
    PortType,
}

/// Name, type, and — since every builtin field is currently required — a value that is known
/// to type-check, offered as a starting point rather than a strict default.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamSchema {
    pub name: String,
    pub ty: ParamType,
    pub required: bool,
    pub default: Option<toml::Value>,
}

/// Everything an editor needs to draw one node kind and validate what a user puts into it,
/// without depending on the node enum itself.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeSchema {
    pub kind: &'static str,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub params: Vec<ParamSchema>,
}

/// Builds a Task IR node from its kind tag and parameters (spec 6.3, `INV-17`).
pub trait TaskNodeFactory {
    /// The kinds this factory can build. Two registered factories must never share one.
    fn kinds(&self) -> &[&'static str];
    /// `params` holds the node's own fields, keyed exactly as `TaskNode`'s serde field names.
    fn create(&self, kind: &str, params: &toml::Value) -> Result<TaskNode, Diagnostic>;
    /// `None` for a kind this factory does not own.
    fn schema(&self, kind: &str) -> Option<NodeSchema>;
}

/// Builds a Learning IR node from its kind tag and parameters (spec 8.3, `INV-17`).
pub trait LearningNodeFactory {
    fn kinds(&self) -> &[&'static str];
    fn create(&self, kind: &str, params: &toml::Value) -> Result<LearningNode, Diagnostic>;
    fn schema(&self, kind: &str) -> Option<NodeSchema>;
}

/// The spec 6.3 node set, frozen (spec 28.7 gate 10): renaming one changes `task_hash` for
/// every graph that uses it.
#[rustfmt::skip]
pub const BUILTIN_TASK_KINDS: &[&str] = &[
    "GetJointState", "GetBodyPose", "GetBodyVelocity", "GetContact", "GetSensor", "GetTime",
    "GetRandom", "GetLanguage",
    "Transform", "Normalize", "Clamp", "Arith", "MathFn", "Norm", "Dot", "Cross", "Compare",
    "Logic", "Select", "Concat", "Slice", "Reduce",
    "ObservationSpec", "ActionSpec", "Reward", "Terminate", "Randomization", "ResetState",
    "Record",
];

/// The spec 6.2 IR-C node set, frozen the same way (spec 28.7 gate 10).
///
/// Deliberately **not** part of [`BUILTIN_TASK_KINDS`]: that list is what a
/// [`TaskNodeFactory`] can build into a `TaskNode`, and a control node is not one. IR-C has no
/// factory trait of its own either — `INV-17` allows no eighth extension point, and editing a
/// Control Graph is a later M4 packet (spec 23.4).
pub const BUILTIN_CONTROL_KINDS: &[&str] = &["Sequence", "Branch", "Repeat", "SubTask"];

/// The spec 8.3 node set, frozen the same way for `learning_hash`.
pub const BUILTIN_LEARNING_KINDS: &[&str] = &[
    "VisionEncoder",
    "StateEncoder",
    "LanguageEncoder",
    "Fusion",
    "TemporalEncoder",
    "PolicyHead",
    "PolicyBundle",
    "ActionChunker",
    "Normalizer",
];

/// blake3 over a kind list, sorted and joined by `\n` so the hash depends on the kind set,
/// never on the order the array happens to be written in (spec 28.7 gate 10, P-M1-R6). Used
/// only by the freeze test below; nothing at runtime recomputes it.
#[cfg(test)]
fn kinds_hash(kinds: &[&str]) -> [u8; 32] {
    let mut sorted: Vec<&str> = kinds.to_vec();
    sorted.sort_unstable();
    *blake3::hash(sorted.join("\n").as_bytes()).as_bytes()
}

/// Frozen blake3 hash of [`BUILTIN_TASK_KINDS`] (spec 28.7 gate 10).
///
/// Changing this hash is a schema change: bump `schema_version` and add a migration note in
/// `docs/design/ir-types.md`.
#[rustfmt::skip]
pub const BUILTIN_TASK_KINDS_HASH: [u8; 32] = [
    0x07, 0x92, 0x83, 0x85, 0x2d, 0x9d, 0x3f, 0xaf,
    0x85, 0xc7, 0xda, 0x5d, 0x48, 0xfa, 0x0c, 0x88,
    0x21, 0x0a, 0xb5, 0xc3, 0x7c, 0x82, 0xef, 0xfc,
    0x3b, 0xb2, 0x36, 0xe2, 0x60, 0x3e, 0x14, 0x4c,
];

/// Frozen blake3 hash of [`BUILTIN_LEARNING_KINDS`] (spec 28.7 gate 10).
///
/// Changing this hash is a schema change: bump `schema_version` and add a migration note in
/// `docs/design/ir-types.md`.
#[rustfmt::skip]
pub const BUILTIN_LEARNING_KINDS_HASH: [u8; 32] = [
    0xa1, 0x7d, 0x06, 0x53, 0xf6, 0xc0, 0x16, 0x0c,
    0xfc, 0xe3, 0x52, 0xa4, 0x86, 0xca, 0x04, 0x0b,
    0xbd, 0xbd, 0xf4, 0xb5, 0x21, 0x39, 0xa1, 0x2f,
    0xfc, 0xf0, 0x92, 0x0e, 0x1c, 0xaf, 0xfe, 0x51,
];

/// Frozen blake3 hash of [`BUILTIN_CONTROL_KINDS`] (spec 28.7 gate 10).
///
/// Changing this hash is a schema change: bump `schema_version` and add a migration note in
/// `docs/design/ir-types.md`.
#[rustfmt::skip]
pub const BUILTIN_CONTROL_KINDS_HASH: [u8; 32] = [
    0x24, 0x38, 0x21, 0x8d, 0x1b, 0xb4, 0x98, 0xaa,
    0xe9, 0x8d, 0xe8, 0x95, 0x69, 0xfd, 0x62, 0xbb,
    0xe1, 0x82, 0x0e, 0xa1, 0xc8, 0x60, 0xda, 0x04,
    0x71, 0xa5, 0xe9, 0x82, 0x35, 0xa2, 0x11, 0x58,
];

fn placeholder_id() -> StableId {
    StableId::from_bytes([0; 16])
}

fn scalar_ty() -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new(vec![1]),
        unit: Unit::Dimensionless,
        frame: Frame::World,
        time: TimeRef::Tick,
        image: None,
    }
}

/// One representative instance per builtin Task IR kind: the smallest set of field values that
/// type-checks. Used to derive [`NodeSchema`] and, in tests, to prove the registry can build
/// every kind it claims to own.
fn example_task_node(kind: &str) -> Option<TaskNode> {
    let ty = scalar_ty();
    Some(match kind {
        "GetJointState" => TaskNode::GetJointState {
            body: placeholder_id(),
            joints: vec!["joint_0".to_owned()],
            quantity: JointQuantity::Position,
        },
        "GetBodyPose" => TaskNode::GetBodyPose {
            body: placeholder_id(),
            relative_to: Frame::World,
        },
        "GetBodyVelocity" => TaskNode::GetBodyVelocity {
            body: placeholder_id(),
            relative_to: Frame::World,
        },
        "GetContact" => TaskNode::GetContact {
            a: placeholder_id(),
            b: placeholder_id(),
        },
        "GetSensor" => TaskNode::GetSensor {
            sensor: placeholder_id(),
            ty: ty.clone(),
        },
        "GetTime" => TaskNode::GetTime { since_reset: false },
        "GetRandom" => TaskNode::GetRandom {
            stream: "stream_0".to_owned(),
            dist: Distribution::Constant(0.0),
            shape: Shape::new(vec![1]),
        },
        "GetLanguage" => TaskNode::GetLanguage {
            key: "instruction".to_owned(),
            max_tokens: 8,
        },
        "Transform" => TaskNode::Transform {
            to: Frame::World,
            ty: ty.clone(),
        },
        "Normalize" => TaskNode::Normalize {
            lo: vec![0.0],
            hi: vec![1.0],
            out_lo: 0.0,
            out_hi: 1.0,
            ty: ty.clone(),
        },
        "Clamp" => TaskNode::Clamp {
            lo: vec![0.0],
            hi: vec![1.0],
            ty: ty.clone(),
        },
        "Arith" => TaskNode::Arith {
            op: ArithOp::Add,
            ty: ty.clone(),
        },
        "MathFn" => TaskNode::MathFn {
            func: MathFunc::Abs,
            approx: false,
            ty: ty.clone(),
        },
        "Norm" => TaskNode::Norm {
            kind: NormKind::L2,
            ty: ty.clone(),
        },
        "Dot" => TaskNode::Dot { ty: ty.clone() },
        "Cross" => TaskNode::Cross { ty: ty.clone() },
        "Compare" => TaskNode::Compare {
            op: CmpOp::Lt,
            rhs: Some(0.0),
            ty: ty.clone(),
        },
        "Logic" => TaskNode::Logic {
            op: LogicOp::Not,
            shape: Shape::new(vec![1]),
        },
        "Select" => TaskNode::Select { ty: ty.clone() },
        "Concat" => TaskNode::Concat {
            parts: vec![ty.clone()],
            axis: 0,
        },
        "Slice" => TaskNode::Slice {
            ty: ty.clone(),
            axis: 0,
            start: 0,
            len: 1,
        },
        "Reduce" => TaskNode::Reduce {
            op: ReduceOp::Sum,
            axis: 0,
            unordered: false,
            ty: ty.clone(),
        },
        "ObservationSpec" => TaskNode::ObservationSpec {
            channel: "channel_0".to_owned(),
            ty: ty.clone(),
        },
        "ActionSpec" => TaskNode::ActionSpec {
            space: ActionSpace::JointPosition,
            dim: 1,
            control_rate_hz: 30.0,
        },
        "Reward" => TaskNode::Reward {
            name: "reward_0".to_owned(),
            weight: 1.0,
            aggregation: Aggregation::Sum,
            ty: ty.clone(),
        },
        "Terminate" => TaskNode::Terminate {
            kind: TerminationKind::Timeout,
        },
        "Randomization" => TaskNode::Randomization {
            target: "target_0".to_owned(),
            dist: Distribution::Constant(0.0),
            stream: "stream_0".to_owned(),
        },
        "ResetState" => TaskNode::ResetState {
            target: "target_0".to_owned(),
            dist: Distribution::Constant(0.0),
            stream: "stream_0".to_owned(),
        },
        "Record" => TaskNode::Record {
            key: "record_0".to_owned(),
            ty: ty.clone(),
        },
        _ => return None,
    })
}

/// One representative instance per builtin Learning IR kind, `inputs: vec![]` throughout: every
/// `outputs()`/`params_canonical()` implementation treats an empty input list as "unconnected
/// yet" rather than panicking (spec 8.3 node set is authored before wiring, same as an editor
/// would).
fn example_learning_node(kind: &str) -> Option<LearningNode> {
    Some(match kind {
        "VisionEncoder" => LearningNode::VisionEncoder {
            inputs: vec![],
            backbone: VisionBackbone::ResNet18,
            pretrained: true,
            frozen: false,
            out_dim: 8,
            token_count: 0,
        },
        "StateEncoder" => LearningNode::StateEncoder {
            inputs: vec![],
            kind: StateEncoderKind::Identity,
            out_dim: 8,
        },
        "LanguageEncoder" => LearningNode::LanguageEncoder {
            inputs: vec![],
            tokenizer: "tokenizer_0".to_owned(),
            model: "model_0".to_owned(),
            max_len: 16,
            out_dim: 8,
            token_count: 1,
        },
        "Fusion" => LearningNode::Fusion {
            inputs: vec![],
            kind: FusionKind::Concat,
            out_dim: 8,
            token_count: 0,
        },
        "TemporalEncoder" => LearningNode::TemporalEncoder {
            inputs: vec![],
            kind: TemporalKind::None,
            n_frames: 1,
            out_dim: 8,
            token_count: 0,
        },
        "PolicyHead" => LearningNode::PolicyHead {
            inputs: vec![],
            kind: HeadKind::Regression,
            action_dim: 4,
            horizon: 1,
        },
        "PolicyBundle" => LearningNode::PolicyBundle {
            inputs: vec![],
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0; 32],
            },
            action_dim: 4,
            horizon: 1,
        },
        "ActionChunker" => LearningNode::ActionChunker {
            inputs: vec![],
            horizon: 1,
            execute_chunk: 1,
            replan_hz: 1.0,
            mode: ActionExecutionMode::OpenLoopChunk,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 1,
        },
        "Normalizer" => LearningNode::Normalizer {
            inputs: vec![],
            direction: NormalizeDir::Forward,
            stats: StatsSource::MeanStd {
                mean: vec![0.0],
                std: vec![1.0],
            },
            out_unit: Unit::Dimensionless,
        },
        _ => return None,
    })
}

/// Infers the coarse shape an editor needs from one observed value. This is deliberately not a
/// full reflection of the node enums (Rust has none to read): a table with one key is read as
/// an externally tagged enum's current variant, and anything else falls back to `String` rather
/// than guessing wrong.
fn infer_param_type(value: &toml::Value) -> ParamType {
    match value {
        toml::Value::Boolean(_) => ParamType::Bool,
        toml::Value::Integer(_) => ParamType::Int,
        toml::Value::Float(_) => ParamType::Float,
        toml::Value::String(_) => ParamType::String,
        toml::Value::Array(items) if items.iter().all(|v| matches!(v, toml::Value::Integer(_))) => {
            ParamType::Shape
        }
        toml::Value::Table(t) if is_port_type_shaped(t) => ParamType::PortType,
        toml::Value::Table(t) if t.len() == 1 => ParamType::Enum(t.keys().cloned().collect()),
        toml::Value::Array(_) | toml::Value::Table(_) | toml::Value::Datetime(_) => {
            ParamType::String
        }
    }
}

fn is_port_type_shaped(table: &toml::Table) -> bool {
    ["elem", "shape", "unit", "frame", "time"]
        .iter()
        .all(|key| table.contains_key(*key))
}

fn param_schemas(table: &toml::Table) -> Vec<ParamSchema> {
    table
        .iter()
        .map(|(name, value)| ParamSchema {
            name: name.clone(),
            ty: infer_param_type(value),
            required: true,
            default: Some(value.clone()),
        })
        .collect()
}

/// Builds a [`NodeSchema`] from one example instance: ports come straight from [`IrNode`],
/// params from serializing the instance and reading its fields back as a table.
fn node_schema<N: IrNode + Serialize>(kind: &'static str, node: &N) -> NodeSchema {
    let params = toml::Value::try_from(node)
        .ok()
        .and_then(|v| match v {
            toml::Value::Table(mut t) => t.remove(kind),
            _ => None,
        })
        .map(|v| match v {
            toml::Value::Table(t) => param_schemas(&t),
            _ => Vec::new(),
        })
        .unwrap_or_default();
    NodeSchema {
        kind,
        inputs: node.inputs(),
        outputs: node.outputs(),
        params,
    }
}

/// Re-tags `params` as `kind` and deserializes it as `T`, using serde's default externally
/// tagged enum representation — the same one `TaskNode`/`LearningNode` derive.
fn deserialize_tagged<T: DeserializeOwned>(
    kind: &str,
    params: &toml::Value,
    valid_kinds: &[&'static str],
) -> Result<T, Diagnostic> {
    if !valid_kinds.contains(&kind) {
        return Err(Diagnostic::new(
            codes::FACTORY_001,
            format!("unknown node kind '{kind}'"),
        ));
    }
    let mut tagged = toml::Table::new();
    tagged.insert(kind.to_owned(), params.clone());
    toml::Value::Table(tagged)
        .try_into()
        .map_err(|e: toml::de::Error| {
            Diagnostic::new(
                codes::FACTORY_003,
                format!("cannot build '{kind}' node: {e}"),
            )
        })
}

/// The spec 6.3 node set (`INV-17`).
#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinTaskNodes;

impl TaskNodeFactory for BuiltinTaskNodes {
    fn kinds(&self) -> &[&'static str] {
        BUILTIN_TASK_KINDS
    }

    fn create(&self, kind: &str, params: &toml::Value) -> Result<TaskNode, Diagnostic> {
        deserialize_tagged(kind, params, BUILTIN_TASK_KINDS)
    }

    fn schema(&self, kind: &str) -> Option<NodeSchema> {
        let kind = *BUILTIN_TASK_KINDS.iter().find(|&&k| k == kind)?;
        Some(node_schema(kind, &example_task_node(kind)?))
    }
}

/// The spec 8.3 node set (`INV-17`).
#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinLearningNodes;

impl LearningNodeFactory for BuiltinLearningNodes {
    fn kinds(&self) -> &[&'static str] {
        BUILTIN_LEARNING_KINDS
    }

    fn create(&self, kind: &str, params: &toml::Value) -> Result<LearningNode, Diagnostic> {
        deserialize_tagged(kind, params, BUILTIN_LEARNING_KINDS)
    }

    fn schema(&self, kind: &str) -> Option<NodeSchema> {
        let kind = *BUILTIN_LEARNING_KINDS.iter().find(|&&k| k == kind)?;
        Some(node_schema(kind, &example_learning_node(kind)?))
    }
}

// Both IRs need the identical "resolve kind to the one factory that owns it, reject
// duplicates at registration" container; a macro keeps that logic in one place without adding
// a shared trait (`INV-17` allows no trait here beyond the two above).
macro_rules! node_registry {
    ($registry:ident, $factory_trait:ident, $node:ty, $builtin:ident) => {
        #[doc = concat!(
            "Resolves a kind to the registered `", stringify!($factory_trait), "` that owns it."
        )]
        #[derive(Default)]
        pub struct $registry {
            factories: Vec<Box<dyn $factory_trait>>,
            index: BTreeMap<&'static str, usize>,
        }

        impl fmt::Debug for $registry {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($registry))
                    .field("kinds", &self.index.keys().collect::<Vec<_>>())
                    .finish()
            }
        }

        impl $registry {
            pub fn new() -> Self {
                Self::default()
            }

            #[doc = concat!("A registry pre-loaded with [`", stringify!($builtin), "`].")]
            pub fn with_builtins() -> Self {
                let mut registry = Self::new();
                registry
                    .register(Box::new($builtin))
                    .expect("builtin kinds contain no duplicates");
                registry
            }

            /// Fails with `FACTORY-002` if any kind `factory` declares is already owned.
            pub fn register(
                &mut self,
                factory: Box<dyn $factory_trait>,
            ) -> Result<(), Diagnostic> {
                for &kind in factory.kinds() {
                    if self.index.contains_key(kind) {
                        return Err(Diagnostic::new(
                            codes::FACTORY_002,
                            format!("kind '{kind}' is already registered"),
                        ));
                    }
                }
                let slot = self.factories.len();
                for &kind in factory.kinds() {
                    self.index.insert(kind, slot);
                }
                self.factories.push(factory);
                Ok(())
            }

            pub fn create(&self, kind: &str, params: &toml::Value) -> Result<$node, Diagnostic> {
                let &slot = self.index.get(kind).ok_or_else(|| {
                    Diagnostic::new(codes::FACTORY_001, format!("unknown node kind '{kind}'"))
                })?;
                self.factories[slot].create(kind, params)
            }

            pub fn schema(&self, kind: &str) -> Option<NodeSchema> {
                let &slot = self.index.get(kind)?;
                self.factories[slot].schema(kind)
            }
        }
    };
}

node_registry!(
    TaskNodeRegistry,
    TaskNodeFactory,
    TaskNode,
    BuiltinTaskNodes
);
node_registry!(
    LearningNodeRegistry,
    LearningNodeFactory,
    LearningNode,
    BuiltinLearningNodes
);

#[cfg(test)]
mod frozen_kind_lists {
    use super::{
        kinds_hash, BUILTIN_CONTROL_KINDS, BUILTIN_CONTROL_KINDS_HASH, BUILTIN_LEARNING_KINDS,
        BUILTIN_LEARNING_KINDS_HASH, BUILTIN_TASK_KINDS, BUILTIN_TASK_KINDS_HASH,
    };

    /// Spec 28.7 gate 10: a rename, addition, or removal in either kind list must fail CI, not
    /// silently change `task_hash` / `learning_hash` for every graph that uses the kind.
    #[test]
    fn builtin_task_kinds_hash_is_frozen() {
        assert_eq!(
            kinds_hash(BUILTIN_TASK_KINDS),
            BUILTIN_TASK_KINDS_HASH,
            "BUILTIN_TASK_KINDS changed — this is a schema change (see the doc comment on \
             BUILTIN_TASK_KINDS_HASH)"
        );
    }

    /// The IR-C kind list is frozen for the same reason: it feeds `task_hash` through
    /// `ControlNode::kind`.
    #[test]
    fn builtin_control_kinds_hash_is_frozen() {
        assert_eq!(
            kinds_hash(BUILTIN_CONTROL_KINDS),
            BUILTIN_CONTROL_KINDS_HASH,
            "BUILTIN_CONTROL_KINDS changed — this is a schema change (see the doc comment on              BUILTIN_CONTROL_KINDS_HASH)"
        );
    }

    #[test]
    fn builtin_learning_kinds_hash_is_frozen() {
        assert_eq!(
            kinds_hash(BUILTIN_LEARNING_KINDS),
            BUILTIN_LEARNING_KINDS_HASH,
            "BUILTIN_LEARNING_KINDS changed — this is a schema change (see the doc comment on \
             BUILTIN_LEARNING_KINDS_HASH)"
        );
    }
}
