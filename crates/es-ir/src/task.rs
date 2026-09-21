//! Task IR: scene reference, goals, rewards, termination, randomization, reset, and the
//! `ObservationSpec` declaration (spec 6). Filled in by P20 — no neural nets here (rule 6).
//!
//! This module carries **IR-D** only: a pure dataflow DAG with no side effects (spec 6.2).
//! The IR-C control nodes (`Sequence`, `Branch`, `SubTask`, `Repeat`) are deliberately not
//! `TaskNode` variants; they live in [`crate::control`] and hang off [`TaskIr::control`].
//! Standard pick-and-place / reach / push tasks need IR-D alone.
//!
//! Three prohibitions of spec 6.1 are structural rather than checked: there is no node for a
//! neural network, none for preprocessing (Observation IR owns it) and none for wall-clock
//! access (`DET-002`) — [`TaskNode::GetTime`] reads simulation time only. `DET-020` is
//! likewise structural: every collection here is a `BTreeMap` / `BTreeSet`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use es_core::StableId;
use serde::{Deserialize, Serialize};

use crate::codes;
use crate::control::ControlGraph;
use crate::diag::Diagnostic;
use crate::graph::{Graph, IrNode, NodeId, Port};
use crate::hash::{canonical_hash, CanonWriter};
use crate::image::ChannelFormat;
use crate::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

/// Current Task IR schema version.
///
/// `2` since IR-C: [`TaskIr`] carries an optional [`ControlGraph`] and `task_hash` mixes it in
/// (migration note in `docs/design/ir-types.md`).
pub const SCHEMA_VERSION: u32 = 2;

const TASK_TAG: &str = "es.ir.task.v1";
const TASK_GRAPH_TAG: &str = "es.ir.task_graph.v1";

// --- parameter vocabulary and expressions ----------------------------------------------------
//
// Moved to `es-ir-types` for the spec 1.5 context budget (`docs/packets/M4/P-M4-S16.md`);
// nothing here knows the graph, and every path below stays `es_ir::task::..`.

pub use es_ir_types::expr::{
    ActionSpace, Aggregation, ArithOp, CmpOp, Distribution, Expr, JointQuantity, LogicOp, MathFunc,
    NormKind, ReduceOp, TerminationKind,
};

// --- port type helpers ---------------------------------------------------------------------

fn tensor(elem: ElemType, dims: Vec<u64>, unit: Unit, frame: Frame) -> PortType {
    PortType {
        elem,
        shape: Shape(dims),
        unit,
        frame,
        time: TimeRef::Tick,
        image: None,
    }
}

fn f32v(n: u64, unit: Unit, frame: Frame) -> PortType {
    tensor(ElemType::F32, vec![n], unit, frame)
}

/// `bool` with the shape and clock of `ty`; a predicate has no unit and no frame.
fn boolish(ty: &PortType) -> PortType {
    PortType {
        elem: ElemType::Bool,
        shape: ty.shape.clone(),
        unit: Unit::Dimensionless,
        frame: Frame::World,
        time: ty.time.clone(),
        image: None,
    }
}

fn flag() -> PortType {
    tensor(ElemType::Bool, vec![1], Unit::Dimensionless, Frame::World)
}

fn with_unit(ty: &PortType, unit: Unit) -> PortType {
    PortType { unit, ..ty.clone() }
}

fn with_dims(ty: &PortType, dims: Vec<u64>) -> PortType {
    PortType {
        shape: Shape(dims),
        ..ty.clone()
    }
}

/// The unit of a product; opaque units (spec 5.4) fall back to the operand's own unit so that
/// port declaration stays infallible — the unit algebra itself is checked in M1 lowering.
fn product_unit(ty: &PortType) -> Unit {
    ty.unit.mul(&ty.unit).unwrap_or_else(|_| ty.unit.clone())
}

fn p(name: &str, ty: PortType) -> Port {
    Port::new(name, ty)
}

// --- canonical encoding helpers --------------------------------------------------------------

fn wdbg(w: &mut CanonWriter, v: &impl fmt::Debug) {
    w.str(&format!("{v:?}"));
}

fn wid(w: &mut CanonWriter, id: &StableId) {
    w.bytes(id.as_bytes());
}

fn wstrs(w: &mut CanonWriter, xs: &[String]) {
    w.seq(xs.len());
    for x in xs {
        w.str(x);
    }
}

fn wf64s(w: &mut CanonWriter, xs: &[f64]) {
    w.seq(xs.len());
    for x in xs {
        w.f64(*x);
    }
}

fn wtys(w: &mut CanonWriter, tys: &[PortType]) {
    w.seq(tys.len());
    for t in tys {
        t.canonical(w);
    }
}

// --- nodes -----------------------------------------------------------------------------------

/// The spec 6.3 node set: sources (phase `Observation`), pure transforms, and sinks.
///
/// `Parallel` does not exist — parallelism is the compiler's decision. `Wait`, `Repeat` and
/// `Condition` are IR-C (M4) or are written as `Compare` + `Select`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TaskNode {
    // --- Source ---
    GetJointState {
        body: StableId,
        joints: Vec<String>,
        quantity: JointQuantity,
    },
    GetBodyPose {
        body: StableId,
        relative_to: Frame,
    },
    GetBodyVelocity {
        body: StableId,
        relative_to: Frame,
    },
    GetContact {
        a: StableId,
        b: StableId,
    },
    /// Generic sensor read; the sensor's own `PortType` carries its clock and `ImageSpec`.
    GetSensor {
        sensor: StableId,
        ty: PortType,
    },
    /// Simulation time. There is no wall-clock source at all (spec 6.6 `DET-002`).
    GetTime {
        since_reset: bool,
    },
    /// `TaskRng(seed, EnvId, PhysTick, stream)`; `stream` is mandatory (spec 6.6 `DET-001`).
    GetRandom {
        stream: String,
        dist: Distribution,
        shape: Shape,
    },
    /// Task instruction for VLA policies (spec 6.3, v1.0).
    GetLanguage {
        key: String,
        max_tokens: u32,
    },

    // --- Transform (pure) ---
    Transform {
        to: Frame,
        ty: PortType,
    },
    Normalize {
        lo: Vec<f64>,
        hi: Vec<f64>,
        out_lo: f64,
        out_hi: f64,
        ty: PortType,
    },
    Clamp {
        lo: Vec<f64>,
        hi: Vec<f64>,
        ty: PortType,
    },
    Arith {
        op: ArithOp,
        ty: PortType,
    },
    MathFn {
        func: MathFunc,
        /// `es-math::approx` implementation; `false` means the standard library one.
        approx: bool,
        ty: PortType,
    },
    Norm {
        kind: NormKind,
        ty: PortType,
    },
    Dot {
        ty: PortType,
    },
    Cross {
        ty: PortType,
    },
    /// `rhs = Some(c)` folds the literal of spec 6.5 (`Compare(<, 0.02)`) into the node.
    Compare {
        op: CmpOp,
        rhs: Option<f64>,
        ty: PortType,
    },
    Logic {
        op: LogicOp,
        shape: Shape,
    },
    Select {
        ty: PortType,
    },
    Concat {
        parts: Vec<PortType>,
        axis: u32,
    },
    Slice {
        ty: PortType,
        axis: u32,
        start: u64,
        len: u64,
    },
    Reduce {
        op: ReduceOp,
        axis: u32,
        /// Rejected in deterministic mode (spec 6.6 `DET-030`).
        unordered: bool,
        ty: PortType,
    },

    // --- Sink ---
    /// Binds a graph value to one channel of [`TaskIr::observation_spec`]. Declaration only —
    /// preprocessing belongs to Observation IR (spec 5.1, spec 7.4).
    ObservationSpec {
        channel: String,
        ty: PortType,
    },
    ActionSpec {
        space: ActionSpace,
        dim: u32,
        control_rate_hz: f32,
    },
    Reward {
        name: String,
        weight: f64,
        aggregation: Aggregation,
        ty: PortType,
    },
    Terminate {
        kind: TerminationKind,
    },
    Randomization {
        target: String,
        dist: Distribution,
        stream: String,
    },
    ResetState {
        target: String,
        dist: Distribution,
        stream: String,
    },
    Record {
        key: String,
        ty: PortType,
    },
}

impl IrNode for TaskNode {
    fn kind(&self) -> &'static str {
        match self {
            Self::GetJointState { .. } => "GetJointState",
            Self::GetBodyPose { .. } => "GetBodyPose",
            Self::GetBodyVelocity { .. } => "GetBodyVelocity",
            Self::GetContact { .. } => "GetContact",
            Self::GetSensor { .. } => "GetSensor",
            Self::GetTime { .. } => "GetTime",
            Self::GetRandom { .. } => "GetRandom",
            Self::GetLanguage { .. } => "GetLanguage",
            Self::Transform { .. } => "Transform",
            Self::Normalize { .. } => "Normalize",
            Self::Clamp { .. } => "Clamp",
            Self::Arith { .. } => "Arith",
            Self::MathFn { .. } => "MathFn",
            Self::Norm { .. } => "Norm",
            Self::Dot { .. } => "Dot",
            Self::Cross { .. } => "Cross",
            Self::Compare { .. } => "Compare",
            Self::Logic { .. } => "Logic",
            Self::Select { .. } => "Select",
            Self::Concat { .. } => "Concat",
            Self::Slice { .. } => "Slice",
            Self::Reduce { .. } => "Reduce",
            Self::ObservationSpec { .. } => "ObservationSpec",
            Self::ActionSpec { .. } => "ActionSpec",
            Self::Reward { .. } => "Reward",
            Self::Terminate { .. } => "Terminate",
            Self::Randomization { .. } => "Randomization",
            Self::ResetState { .. } => "ResetState",
            Self::Record { .. } => "Record",
        }
    }

    fn inputs(&self) -> Vec<Port> {
        match self {
            Self::GetJointState { .. }
            | Self::GetBodyPose { .. }
            | Self::GetBodyVelocity { .. }
            | Self::GetContact { .. }
            | Self::GetSensor { .. }
            | Self::GetTime { .. }
            | Self::GetRandom { .. }
            | Self::GetLanguage { .. }
            | Self::ActionSpec { .. }
            | Self::Randomization { .. }
            | Self::ResetState { .. } => vec![],
            Self::Transform { ty, .. }
            | Self::Normalize { ty, .. }
            | Self::Clamp { ty, .. }
            | Self::MathFn { ty, .. }
            | Self::Norm { ty, .. }
            | Self::Slice { ty, .. }
            | Self::Reduce { ty, .. }
            | Self::ObservationSpec { ty, .. }
            | Self::Reward { ty, .. }
            | Self::Record { ty, .. } => vec![p("value", ty.clone())],
            Self::Arith { op, ty } => {
                let rhs = match op {
                    ArithOp::Mul | ArithOp::Div => with_unit(ty, Unit::Dimensionless),
                    _ => ty.clone(),
                };
                vec![p("a", ty.clone()), p("b", rhs)]
            }
            Self::Dot { ty } | Self::Cross { ty } => {
                vec![p("a", ty.clone()), p("b", ty.clone())]
            }
            Self::Compare { rhs, ty, .. } => {
                let mut v = vec![p("a", ty.clone())];
                if rhs.is_none() {
                    v.push(p("b", ty.clone()));
                }
                v
            }
            Self::Logic { op, shape } => {
                let ty = tensor(
                    ElemType::Bool,
                    shape.dims().to_vec(),
                    Unit::Dimensionless,
                    Frame::World,
                );
                let mut v = vec![p("a", ty.clone())];
                if *op != LogicOp::Not {
                    v.push(p("b", ty));
                }
                v
            }
            Self::Select { ty } => vec![
                p("cond", boolish(ty)),
                p("a", ty.clone()),
                p("b", ty.clone()),
            ],
            Self::Concat { parts, .. } => parts
                .iter()
                .enumerate()
                .map(|(i, t)| p(&format!("in{i}"), t.clone()))
                .collect(),
            Self::Terminate { .. } => vec![p("value", flag())],
        }
    }

    fn outputs(&self) -> Vec<Port> {
        match self {
            Self::GetJointState {
                body,
                joints,
                quantity,
            } => vec![p(
                "value",
                f32v(joints.len() as u64, quantity.unit(), Frame::Joint(*body)),
            )],
            Self::GetBodyPose { relative_to, .. } => vec![
                p("pos", f32v(3, Unit::Length, *relative_to)),
                p("quat", f32v(4, Unit::Quaternion, *relative_to)),
            ],
            Self::GetBodyVelocity { relative_to, .. } => vec![
                p("linear", f32v(3, Unit::Velocity, *relative_to)),
                p("angular", f32v(3, Unit::AngularVelocity, *relative_to)),
            ],
            Self::GetContact { .. } => vec![
                p("force", f32v(3, Unit::Force, Frame::World)),
                p("touching", flag()),
            ],
            Self::GetSensor { ty, .. }
            | Self::Clamp { ty, .. }
            | Self::Arith { ty, .. }
            | Self::MathFn { ty, .. }
            | Self::Select { ty } => vec![p("value", ty.clone())],
            Self::GetTime { .. } => vec![p("value", f32v(1, Unit::Time, Frame::World))],
            Self::GetRandom { shape, .. } => vec![p(
                "value",
                tensor(
                    ElemType::F32,
                    shape.dims().to_vec(),
                    Unit::Dimensionless,
                    Frame::World,
                ),
            )],
            Self::GetLanguage { max_tokens, .. } => vec![p(
                "tokens",
                tensor(
                    ElemType::I32,
                    vec![u64::from(*max_tokens)],
                    Unit::Token,
                    Frame::Policy,
                ),
            )],
            Self::Transform { to, ty } => vec![p(
                "value",
                PortType {
                    frame: *to,
                    ..ty.clone()
                },
            )],
            Self::Normalize {
                out_lo, out_hi, ty, ..
            } => vec![p(
                "value",
                with_unit(
                    ty,
                    Unit::Normalized {
                        lo: *out_lo,
                        hi: *out_hi,
                    },
                ),
            )],
            Self::Norm { ty, .. } => vec![p("value", with_dims(ty, vec![1]))],
            Self::Dot { ty } => vec![p(
                "value",
                with_unit(&with_dims(ty, vec![1]), product_unit(ty)),
            )],
            Self::Cross { ty } => vec![p("value", with_unit(ty, product_unit(ty)))],
            Self::Compare { ty, .. } => vec![p("value", boolish(ty))],
            Self::Logic { shape, .. } => vec![p(
                "value",
                tensor(
                    ElemType::Bool,
                    shape.dims().to_vec(),
                    Unit::Dimensionless,
                    Frame::World,
                ),
            )],
            Self::Concat { parts, axis } => parts
                .first()
                .map(|head| {
                    let mut dims = head.shape.dims().to_vec();
                    if let Some(d) = dims.get_mut(*axis as usize) {
                        *d = parts
                            .iter()
                            .map(|t| t.shape.dims().get(*axis as usize).copied().unwrap_or(0))
                            .sum();
                    }
                    vec![p("value", with_dims(head, dims))]
                })
                .unwrap_or_default(),
            Self::Slice { ty, axis, len, .. } => {
                let mut dims = ty.shape.dims().to_vec();
                if let Some(d) = dims.get_mut(*axis as usize) {
                    *d = *len;
                }
                vec![p("value", with_dims(ty, dims))]
            }
            Self::Reduce { axis, ty, .. } => {
                let mut dims = ty.shape.dims().to_vec();
                if (*axis as usize) < dims.len() {
                    dims.remove(*axis as usize);
                }
                if dims.is_empty() {
                    dims.push(1);
                }
                vec![p("value", with_dims(ty, dims))]
            }
            Self::ObservationSpec { .. }
            | Self::ActionSpec { .. }
            | Self::Reward { .. }
            | Self::Terminate { .. }
            | Self::Randomization { .. }
            | Self::ResetState { .. }
            | Self::Record { .. } => vec![],
        }
    }

    fn params_canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::GetJointState {
                body,
                joints,
                quantity,
            } => {
                wid(w, body);
                wstrs(w, joints);
                wdbg(w, quantity);
            }
            Self::GetBodyPose { body, relative_to }
            | Self::GetBodyVelocity { body, relative_to } => {
                wid(w, body);
                wdbg(w, relative_to);
            }
            Self::GetContact { a, b } => {
                wid(w, a);
                wid(w, b);
            }
            Self::GetSensor { sensor, ty } => {
                wid(w, sensor);
                ty.canonical(w);
            }
            Self::GetTime { since_reset } => w.bool(*since_reset),
            Self::GetRandom {
                stream,
                dist,
                shape,
            } => {
                w.str(stream);
                dist.canonical(w);
                w.seq(shape.rank());
                for d in shape.dims() {
                    w.u64(*d);
                }
            }
            Self::GetLanguage { key, max_tokens } => {
                w.str(key);
                w.u32(*max_tokens);
            }
            Self::Transform { to, ty } => {
                wdbg(w, to);
                ty.canonical(w);
            }
            Self::Normalize {
                lo,
                hi,
                out_lo,
                out_hi,
                ty,
            } => {
                wf64s(w, lo);
                wf64s(w, hi);
                w.f64(*out_lo);
                w.f64(*out_hi);
                ty.canonical(w);
            }
            Self::Clamp { lo, hi, ty } => {
                wf64s(w, lo);
                wf64s(w, hi);
                ty.canonical(w);
            }
            Self::Arith { op, ty } => {
                wdbg(w, op);
                ty.canonical(w);
            }
            Self::MathFn { func, approx, ty } => {
                wdbg(w, func);
                w.bool(*approx);
                ty.canonical(w);
            }
            Self::Norm { kind, ty } => {
                wdbg(w, kind);
                ty.canonical(w);
            }
            Self::Dot { ty } | Self::Cross { ty } | Self::Select { ty } => ty.canonical(w),
            Self::Compare { op, rhs, ty } => {
                wdbg(w, op);
                match rhs {
                    None => w.bool(false),
                    Some(v) => {
                        w.bool(true);
                        w.f64(*v);
                    }
                }
                ty.canonical(w);
            }
            Self::Logic { op, shape } => {
                wdbg(w, op);
                w.seq(shape.rank());
                for d in shape.dims() {
                    w.u64(*d);
                }
            }
            Self::Concat { parts, axis } => {
                wtys(w, parts);
                w.u32(*axis);
            }
            Self::Slice {
                ty,
                axis,
                start,
                len,
            } => {
                ty.canonical(w);
                w.u32(*axis);
                w.u64(*start);
                w.u64(*len);
            }
            Self::Reduce {
                op,
                axis,
                unordered,
                ty,
            } => {
                wdbg(w, op);
                w.u32(*axis);
                w.bool(*unordered);
                ty.canonical(w);
            }
            Self::ObservationSpec { channel, ty } | Self::Record { key: channel, ty } => {
                w.str(channel);
                ty.canonical(w);
            }
            Self::ActionSpec {
                space,
                dim,
                control_rate_hz,
            } => {
                wdbg(w, space);
                w.u32(*dim);
                w.f32(*control_rate_hz);
            }
            Self::Reward {
                name,
                weight,
                aggregation,
                ty,
            } => {
                w.str(name);
                w.f64(*weight);
                wdbg(w, aggregation);
                ty.canonical(w);
            }
            Self::Terminate { kind } => wdbg(w, kind),
            Self::Randomization {
                target,
                dist,
                stream,
            }
            | Self::ResetState {
                target,
                dist,
                stream,
            } => {
                w.str(target);
                dist.canonical(w);
                w.str(stream);
            }
        }
    }
}

// --- the IR ------------------------------------------------------------------------------------

/// The scene the task runs in: the authored path plus the hashes that pin its content
/// (spec 5.3 `scene_hash` / `asset_hash`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneRef {
    pub path: String,
    pub scene_hash: [u8; 32],
    pub asset_hash: [u8; 32],
}

impl SceneRef {
    fn canonical(&self, w: &mut CanonWriter) {
        w.str(&self.path);
        w.digest(&self.scene_hash);
        w.digest(&self.asset_hash);
    }
}

/// Which renderer produces a [`ObsSource::Sensor`] channel (spec 15.3, packet M7/R5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "path", rename_all = "lowercase")]
pub enum SensorPath {
    /// The rasterizer: today's observation path, and the default.
    #[default]
    Rs,
    /// The path tracer, at this many samples and bounces per pixel.
    Pt { spp: u32, bounces: u32 },
}

/// How linear radiance becomes a display value, mirroring `es_render::Tonemap` (which is
/// layer 5 and out of reach here, spec 4.2). `Pt` only: the `Rs` path clamps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tonemap {
    #[default]
    Reinhard,
    Aces,
}

/// How the simulation *produces* one sensor channel (packet M7/R5).
///
/// Not an `ImageSpec` field and not an Observation IR one: what the sensor **is** —
/// resolution, colour space, intrinsics — is `ImageSpec`'s, and how the simulation **makes**
/// it is the Task IR's. One Observation IR therefore serves an `Rs` and a `Pt` task alike.
///
/// [`Self::default`] is the rasterizer at neutral exposure, and [`ObsSource::canonical`]
/// writes the block **only when it is not the default**, so an absent or default `render` is
/// byte for byte today's canonical form and no committed `task_hash` moves (spec 28.10
/// rule 1). A `Pt` sensor does move it, and is therefore a new document (spec 13.3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SensorRender {
    #[serde(flatten)]
    pub path: SensorPath,
    /// Linear multiplier applied before [`Self::tonemap`].
    pub exposure: f32,
    pub tonemap: Tonemap,
}

impl Default for SensorRender {
    fn default() -> Self {
        Self {
            path: SensorPath::Rs,
            exposure: 1.0,
            tonemap: Tonemap::Reinhard,
        }
    }
}

impl SensorRender {
    /// Whether this is the block an absent `render` means. The serde `skip_serializing_if`
    /// and the canonical-form rule are the same predicate, so a document that omits it and a
    /// document that writes it out cannot disagree.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    fn canonical(&self, w: &mut CanonWriter) {
        match self.path {
            SensorPath::Rs => w.str("rs"),
            SensorPath::Pt { spp, bounces } => {
                w.str("pt");
                w.u32(spp);
                w.u32(bounces);
            }
        }
        w.f32(self.exposure);
        wdbg(w, &self.tonemap);
    }
}

/// Where one declared observation channel comes from (spec 7.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObsSource {
    Sensor {
        id: StableId,
        format: ChannelFormat,
        /// Absent = [`SensorRender::default`] = today's canonical form (packet M7/R5).
        #[serde(default, skip_serializing_if = "SensorRender::is_default")]
        render: SensorRender,
    },
    JointState {
        body: StableId,
        dof: u32,
        /// Which joint quantity the channel carries, mirroring
        /// [`TaskNode::GetJointState`]'s own field (spec 6.3).
        ///
        /// Absent = [`JointQuantity::Position`] = today's canonical form (packet M8/S4e,
        /// the rule packet M7/R5 set for [`SensorRender`]), so no committed `task_hash`
        /// moves. A `Velocity` channel does move it, and is therefore a new document
        /// (spec 13.3).
        #[serde(default = "joint_position", skip_serializing_if = "is_joint_position")]
        quantity: JointQuantity,
    },
    BodyPose(StableId),
    Language,
}

fn joint_position() -> JointQuantity {
    JointQuantity::Position
}

/// Whether this is the quantity an absent `quantity` means. The serde `skip_serializing_if`
/// and the canonical-form rule are the same predicate, so a document that omits it and a
/// document that writes it out cannot disagree.
#[allow(clippy::trivially_copy_pass_by_ref)] // serde's `skip_serializing_if` wants `&T`
fn is_joint_position(q: &JointQuantity) -> bool {
    matches!(q, JointQuantity::Position)
}

impl ObsSource {
    fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::Sensor { id, format, render } => {
                w.str("Sensor");
                wid(w, id);
                wdbg(w, format);
                // Only when it is not the default: see [`SensorRender`].
                if !render.is_default() {
                    w.str("render");
                    render.canonical(w);
                }
            }
            Self::JointState {
                body,
                dof,
                quantity,
            } => {
                w.str("JointState");
                wid(w, body);
                w.u32(*dof);
                // Only when it is not the default: see the field's own note.
                if !is_joint_position(quantity) {
                    wdbg(w, quantity);
                }
            }
            Self::BodyPose(id) => {
                w.str("BodyPose");
                wid(w, id);
            }
            Self::Language => w.str("Language"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObsChannel {
    pub source: ObsSource,
    pub ty: PortType,
}

/// The input contract handed to Observation IR (spec 7.4): named channels with a source and a
/// type, and **nothing else**. Resize, color transform, normalization and the time window are
/// Observation IR's (spec 5.1, rule 6), so this type has no field for any of them — several
/// Observation IRs share one `task_hash` precisely because none of that is declared here.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObservationSpec {
    pub channels: BTreeMap<String, ObsChannel>,
}

impl ObservationSpec {
    fn canonical(&self, w: &mut CanonWriter) {
        w.seq(self.channels.len());
        for (name, ch) in &self.channels {
            w.str(name);
            ch.source.canonical(w);
            ch.ty.canonical(w);
        }
    }
}

/// What the episode needs that no single node owns.
///
/// Reset distributions, domain randomization and the termination predicates are *nodes*
/// (spec 6.3 `ResetState`, `Randomization`, `Terminate`); this carries the episode budget that
/// `Terminate(Timeout)` counts against, the control rate, the RNG streams the task declares
/// (spec 6.6 `DET-021`) and whether deterministic mode is required (`DET-030`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskConfig {
    pub max_episode_steps: u32,
    pub control_rate_hz: f32,
    pub deterministic: bool,
    pub rng_streams: BTreeSet<String>,
}

impl TaskConfig {
    fn canonical(&self, w: &mut CanonWriter) {
        w.u32(self.max_episode_steps);
        w.f32(self.control_rate_hz);
        w.bool(self.deterministic);
        w.seq(self.rng_streams.len());
        for s in &self.rng_streams {
            w.str(s);
        }
    }
}

pub type TaskGraph = Graph<TaskNode>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskIr {
    pub schema_version: u32,
    pub scene: SceneRef,
    pub graph: TaskGraph,
    pub observation_spec: ObservationSpec,
    pub config: TaskConfig,
    /// IR-C (spec 6.2). `None` is the IR-D-only task and the default, so every task authored
    /// before IR-C parses unchanged.
    #[serde(default)]
    pub control: Option<ControlGraph>,
}

impl TaskIr {
    /// Every check spec 6 mandates, in one pass: port and edge typing, acyclicity, the
    /// determinism rules of spec 6.6, and the spec 7.4 declaration link.
    ///
    /// "No neural nets in Task IR" needs no check — [`TaskNode`] has no such variant.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut diags = self.graph.validate_declared_ports();
        if let Err(d) = self.graph.topo_order() {
            diags.push(d);
        }
        // A file from a newer writer carries nodes and fields this build cannot see, and a `0`
        // is an unwritten field; either way `task_hash` would pin something never validated.
        // Older supported versions stay readable -- the version is hashed, so their hashes
        // survive (`docs/design/ir-types.md`).
        if self.schema_version == 0 || self.schema_version > SCHEMA_VERSION {
            diags.push(Diagnostic::new(
                codes::TASK_002,
                format!(
                    "schema_version {} is not supported (expected 1..={SCHEMA_VERSION})",
                    self.schema_version
                ),
            ));
        }

        let stream_check = |diags: &mut Vec<Diagnostic>, id: NodeId, stream: &String| {
            if stream.is_empty() {
                diags.push(
                    Diagnostic::new(codes::DET_001, "the RNG stream name is empty")
                        .at(id)
                        .with_hint("every draw names a TaskRng(seed, EnvId, PhysTick, stream)"),
                );
            } else if !self.config.rng_streams.contains(stream) {
                diags.push(
                    Diagnostic::new(
                        codes::DET_021,
                        format!("stream \"{stream}\" is not declared in the task config"),
                    )
                    .at(id),
                );
            }
        };

        for (id, node) in &self.graph.nodes {
            match node {
                TaskNode::GetRandom { stream, .. }
                | TaskNode::Randomization { stream, .. }
                | TaskNode::ResetState { stream, .. } => stream_check(&mut diags, *id, stream),
                TaskNode::MathFn { func, approx, .. } if func.is_transcendental() && !approx => {
                    diags.push(
                        Diagnostic::new(
                            codes::DET_010,
                            format!("{func:?} uses the standard library implementation"),
                        )
                        .at(*id)
                        .with_hint("set approx = true (es-math::approx)"),
                    );
                }
                TaskNode::Reduce { unordered, .. } if *unordered && self.config.deterministic => {
                    diags.push(
                        Diagnostic::new(codes::DET_030, "unordered Reduce in deterministic mode")
                            .at(*id),
                    );
                }
                TaskNode::Reward { ty, name, .. } => {
                    if let Err(d) = ty.check_policy_input() {
                        diags.push(d.at(*id).with_hint(format!(
                            "reward term \"{name}\" must be dimensionless; normalize it first"
                        )));
                    }
                }
                TaskNode::ObservationSpec { channel, ty } => {
                    match self.observation_spec.channels.get(channel) {
                        Some(ch) if ch.ty == *ty => {}
                        Some(_) => diags.push(
                            Diagnostic::new(
                                codes::TASK_001,
                                format!("channel \"{channel}\" is declared with a different type"),
                            )
                            .at(*id),
                        ),
                        None => diags.push(
                            Diagnostic::new(
                                codes::TASK_001,
                                format!("channel \"{channel}\" is not in the ObservationSpec"),
                            )
                            .at(*id),
                        ),
                    }
                }
                _ => {}
            }
        }

        let bound: BTreeSet<&str> = self
            .graph
            .nodes
            .values()
            .filter_map(|n| match n {
                TaskNode::ObservationSpec { channel, .. } => Some(channel.as_str()),
                _ => None,
            })
            .collect();
        for name in self.observation_spec.channels.keys() {
            if !bound.contains(name.as_str()) {
                diags.push(Diagnostic::new(
                    codes::TASK_001,
                    format!("declared channel \"{name}\" has no ObservationSpec node"),
                ));
            }
        }

        if let Some(control) = &self.control {
            diags.extend(control.validate(self));
        }

        if let Err(d) = self.task_hash() {
            diags.push(d);
        }
        diags
    }

    /// Semantic identity (spec 11.2 `task_hash`): independent of node ids and node order, so
    /// relabelling the graph leaves it unchanged.
    pub fn task_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let graph = canonical_hash(&self.graph)?;
        let mut w = CanonWriter::new();
        w.str(TASK_TAG);
        w.u32(self.schema_version);
        self.scene.canonical(&mut w);
        self.observation_spec.canonical(&mut w);
        self.config.canonical(&mut w);
        w.digest(&graph);
        match &self.control {
            Some(control) => {
                w.bool(true);
                w.digest(&control.control_hash()?);
            }
            None => w.bool(false),
        }
        w.hash()
    }

    /// Authoring identity (spec 11.2 `task_graph_hash`): the same inputs plus the user-assigned
    /// `NodeId`s, so an editor can tell "the same task, renumbered" from "unchanged".
    pub fn task_graph_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let mut w = CanonWriter::new();
        w.str(TASK_GRAPH_TAG);
        w.digest(&self.task_hash()?);
        w.seq(self.graph.nodes.len());
        for (id, node) in &self.graph.nodes {
            w.u32(id.0);
            w.str(node.kind());
        }
        let mut edges: Vec<(u32, &str, u32, &str)> = self
            .graph
            .edges
            .iter()
            .map(|e| {
                (
                    e.from.node.0,
                    e.from.port.as_str(),
                    e.to.node.0,
                    e.to.port.as_str(),
                )
            })
            .collect();
        edges.sort_unstable();
        w.seq(edges.len());
        for (from, from_port, to, to_port) in edges {
            w.u32(from);
            w.str(from_port);
            w.u32(to);
            w.str(to_port);
        }
        w.hash()
    }
}

/// Fixtures and the proptest generator for the Appendix B.7 properties (P25).
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use super::{
        ArithOp, CmpOp, Distribution, Expr, JointQuantity, NormKind, ObsChannel, ObsSource,
        ObservationSpec, SceneRef, TaskConfig, TaskGraph, TaskIr, TaskNode, TerminationKind, Unit,
        SCHEMA_VERSION,
    };
    use crate::control::{ControlGraph, ControlNode, RepeatUntil, SubTaskRef};
    use crate::graph::NodeId;
    use crate::types::{ElemType, Frame, PortType, Shape, TimeRef};
    use es_core::StableId;
    use proptest::prelude::*;
    use std::collections::{BTreeMap, BTreeSet};

    pub fn ty(elem: ElemType, n: u64, unit: Unit, frame: Frame) -> PortType {
        PortType {
            elem,
            shape: Shape::new([n]),
            unit,
            frame,
            time: TimeRef::Tick,
            image: None,
        }
    }

    fn vec3() -> PortType {
        ty(ElemType::F32, 3, Unit::Length, Frame::World)
    }

    fn scalar(unit: Unit) -> PortType {
        ty(ElemType::F32, 1, unit, Frame::World)
    }

    /// `k` reward terms (distance between two bodies), a timeout, one observation channel and
    /// a randomized / reset parameter — the shape of every pick-and-place-class task.
    pub fn task_ir(terms: &[(String, f64, f64)], dof: u32) -> TaskIr {
        let mut g = TaskGraph::new(SCHEMA_VERSION);
        let mut next = 0u32;
        let mut add = |g: &mut TaskGraph, node: TaskNode| {
            let id = NodeId(next);
            next += 1;
            g.insert(id, node);
            id
        };

        for (name, weight, hi) in terms {
            let a = add(
                &mut g,
                TaskNode::GetBodyPose {
                    body: StableId::from_path(&format!("{name}/a")),
                    relative_to: Frame::World,
                },
            );
            let b = add(
                &mut g,
                TaskNode::GetBodyPose {
                    body: StableId::from_path(&format!("{name}/b")),
                    relative_to: Frame::World,
                },
            );
            let sub = add(
                &mut g,
                TaskNode::Arith {
                    op: ArithOp::Sub,
                    ty: vec3(),
                },
            );
            let norm = add(
                &mut g,
                TaskNode::Norm {
                    kind: NormKind::L2,
                    ty: vec3(),
                },
            );
            let normalize = add(
                &mut g,
                TaskNode::Normalize {
                    lo: vec![0.0],
                    hi: vec![*hi],
                    out_lo: -1.0,
                    out_hi: 1.0,
                    ty: scalar(Unit::Length),
                },
            );
            let reward = add(
                &mut g,
                TaskNode::Reward {
                    name: name.clone(),
                    weight: *weight,
                    aggregation: super::Aggregation::Sum,
                    ty: scalar(Unit::Normalized { lo: -1.0, hi: 1.0 }),
                },
            );
            g.connect(a, "pos", sub, "a");
            g.connect(b, "pos", sub, "b");
            g.connect(sub, "value", norm, "value");
            g.connect(norm, "value", normalize, "value");
            g.connect(normalize, "value", reward, "value");
        }

        let time = add(&mut g, TaskNode::GetTime { since_reset: true });
        let over = add(
            &mut g,
            TaskNode::Compare {
                op: CmpOp::Gt,
                rhs: Some(10.0),
                ty: scalar(Unit::Time),
            },
        );
        let stop = add(
            &mut g,
            TaskNode::Terminate {
                kind: TerminationKind::Timeout,
            },
        );
        g.connect(time, "value", over, "a");
        g.connect(over, "value", stop, "value");

        let arm = StableId::from_path("arm");
        let joint_ty = ty(
            ElemType::F32,
            u64::from(dof),
            Unit::Angle,
            Frame::Joint(arm),
        );
        let joints = add(
            &mut g,
            TaskNode::GetJointState {
                body: arm,
                joints: (0..dof).map(|i| format!("j{i}")).collect(),
                quantity: JointQuantity::Position,
            },
        );
        let obs = add(
            &mut g,
            TaskNode::ObservationSpec {
                channel: "joint_state".to_owned(),
                ty: joint_ty.clone(),
            },
        );
        g.connect(joints, "value", obs, "value");

        add(
            &mut g,
            TaskNode::Randomization {
                target: "cube.mass".to_owned(),
                dist: Distribution::Uniform { lo: 0.1, hi: 0.5 },
                stream: "domain".to_owned(),
            },
        );
        add(
            &mut g,
            TaskNode::ResetState {
                target: "cube.pos".to_owned(),
                dist: Distribution::Normal {
                    mean: 0.0,
                    std: 0.05,
                },
                stream: "reset".to_owned(),
            },
        );
        add(
            &mut g,
            TaskNode::ActionSpec {
                space: super::ActionSpace::JointPosition,
                dim: dof,
                control_rate_hz: 50.0,
            },
        );

        let mut channels = BTreeMap::new();
        channels.insert(
            "joint_state".to_owned(),
            ObsChannel {
                source: ObsSource::JointState {
                    body: arm,
                    dof,
                    quantity: JointQuantity::Position,
                },
                ty: joint_ty,
            },
        );

        TaskIr {
            schema_version: SCHEMA_VERSION,
            scene: SceneRef {
                path: "scenes/pick.usda".to_owned(),
                scene_hash: [7u8; 32],
                asset_hash: [9u8; 32],
            },
            graph: g,
            observation_spec: ObservationSpec { channels },
            control: None,
            config: TaskConfig {
                max_episode_steps: 400,
                control_rate_hz: 50.0,
                deterministic: true,
                rng_streams: ["domain", "reset"]
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
            },
        }
    }

    /// A `Sequence` of one stage per reward term, wrapped in a `Repeat` and reached through a
    /// `Branch` — every IR-C kind of spec 6.2, over a task that actually declares those rewards.
    pub fn control_tree(terms: &[(String, f64, f64)]) -> ControlGraph {
        let done = |port: &str| Expr::Compare {
            op: CmpOp::Gt,
            lhs: Box::new(Expr::Port(port.to_owned())),
            rhs: Box::new(Expr::Const(0.5)),
        };
        let mut nodes = BTreeMap::new();
        let stage_ids: Vec<NodeId> = (0..terms.len())
            .map(|i| NodeId(u32::try_from(i).unwrap_or(0) + 4))
            .collect();
        for ((name, weight, _), id) in terms.iter().zip(&stage_ids) {
            nodes.insert(
                *id,
                ControlNode::SubTask {
                    task: SubTaskRef {
                        name: name.clone(),
                        rewards: [name.clone()].into(),
                        observation: ["joint_state".to_owned()].into(),
                        success: Some(done("stage.done")),
                        reset_on_entry: false,
                        weight: *weight,
                    },
                    timeout_ticks: 40,
                },
            );
        }
        nodes.insert(
            NodeId(3),
            ControlNode::Sequence {
                children: stage_ids,
            },
        );
        nodes.insert(
            NodeId(2),
            ControlNode::Repeat {
                body: NodeId(3),
                until: RepeatUntil::Count(2),
            },
        );
        nodes.insert(
            NodeId(1),
            ControlNode::SubTask {
                task: SubTaskRef {
                    name: "abort".to_owned(),
                    rewards: BTreeSet::new(),
                    observation: BTreeSet::new(),
                    success: Some(done("stage.ticks")),
                    reset_on_entry: true,
                    weight: 0.0,
                },
                timeout_ticks: 5,
            },
        );
        nodes.insert(
            NodeId(0),
            ControlNode::Branch {
                condition: done("time.episode"),
                then_: NodeId(2),
                else_: NodeId(1),
            },
        );
        ControlGraph {
            root: NodeId(0),
            nodes,
        }
    }

    /// Valid, connected small Task IRs (Appendix B.7), half of them carrying a control tree so
    /// the five properties cover IR-C as well as IR-D.
    pub fn arbitrary_task_ir() -> impl Strategy<Value = TaskIr> {
        (
            proptest::collection::vec(("term[a-c]{1,3}", 0.1f64..4.0, 0.01f64..2.0), 1..4),
            1u32..8,
            any::<bool>(),
        )
            .prop_map(|(mut terms, dof, control)| {
                terms.sort_by(|a, b| a.0.cmp(&b.0));
                terms.dedup_by(|a, b| a.0 == b.0);
                let mut ir = task_ir(&terms, dof);
                if control {
                    ir.control = Some(control_tree(&terms));
                }
                ir
            })
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{arbitrary_task_ir, task_ir, ty};
    use super::*;
    use crate::graph::{Edge, PortRef};
    use proptest::prelude::*;

    fn fixture() -> TaskIr {
        task_ir(&[("reach".to_owned(), 1.0, 0.5)], 7)
    }

    /// Shifts every `NodeId` by 100 and reverses the edge list: pure authoring churn.
    fn relabel(ir: &TaskIr) -> TaskIr {
        let mut out = ir.clone();
        out.graph.nodes = ir
            .graph
            .nodes
            .iter()
            .map(|(id, n)| (NodeId(id.0 + 100), n.clone()))
            .collect();
        out.graph.edges = ir
            .graph
            .edges
            .iter()
            .rev()
            .map(|e| Edge {
                from: PortRef::new(NodeId(e.from.node.0 + 100), e.from.port.clone()),
                to: PortRef::new(NodeId(e.to.node.0 + 100), e.to.port.clone()),
            })
            .collect();
        out
    }

    fn codes_of(diags: &[Diagnostic]) -> Vec<&str> {
        diags.iter().map(|d| d.code.as_str()).collect()
    }

    #[test]
    fn valid_fixture_has_no_diagnostics() {
        let diags = fixture().validate();
        assert!(diags.is_empty(), "{diags:?}");
    }

    #[test]
    fn serde_json_round_trip() {
        let ir = fixture();
        let json = serde_json::to_string(&ir).unwrap();
        assert_eq!(serde_json::from_str::<TaskIr>(&json).unwrap(), ir);
    }

    /// S-13: `0` and anything past [`SCHEMA_VERSION`] are rejected; every version this build
    /// still understands loads, from a file as much as from memory.
    #[test]
    fn unsupported_schema_versions_are_rejected() {
        let with = |v: u32| {
            let mut ir = fixture();
            ir.schema_version = v;
            ir
        };
        assert_eq!(codes_of(&with(0).validate()), [codes::TASK_002]);
        assert_eq!(
            codes_of(&with(SCHEMA_VERSION + 1).validate()),
            [codes::TASK_002]
        );
        for v in 1..=SCHEMA_VERSION {
            assert!(with(v).validate().is_empty(), "version {v} is supported");
        }
        // The same guard is what a loaded file meets: `task_from_toml` builds a `TaskIr`.
        let toml = crate::serial::task_to_toml(&with(SCHEMA_VERSION + 1)).unwrap();
        let loaded = crate::serial::task_from_toml(&toml).unwrap();
        assert_eq!(codes_of(&loaded.validate()), [codes::TASK_002]);
    }

    #[test]
    fn cycle_is_graph_001() {
        let mut ir = fixture();
        let ty3 = ty(ElemType::F32, 3, Unit::Length, Frame::World);
        for id in [900u32, 901] {
            ir.graph.insert(
                NodeId(id),
                TaskNode::Arith {
                    op: ArithOp::Add,
                    ty: ty3.clone(),
                },
            );
        }
        ir.graph.connect(NodeId(900), "value", NodeId(901), "a");
        ir.graph.connect(NodeId(901), "value", NodeId(900), "a");
        assert!(codes_of(&ir.validate()).contains(&codes::GRAPH_001));
    }

    #[test]
    fn unit_mismatch_is_type_003() {
        let mut ir = fixture();
        ir.graph.insert(
            NodeId(900),
            TaskNode::Norm {
                kind: NormKind::L2,
                ty: ty(ElemType::F32, 3, Unit::Force, Frame::World),
            },
        );
        ir.graph.insert(
            NodeId(901),
            TaskNode::GetBodyPose {
                body: StableId::from_path("cube"),
                relative_to: Frame::World,
            },
        );
        ir.graph.connect(NodeId(901), "pos", NodeId(900), "value");
        assert!(codes_of(&ir.validate()).contains(&codes::TYPE_003));
    }

    #[test]
    fn unaligned_time_ref_is_type_014() {
        let mut ir = fixture();
        let sensor = StableId::from_path("cam_front");
        let mut sensor_ty = ty(ElemType::F32, 3, Unit::Length, Frame::World);
        sensor_ty.time = TimeRef::Sensor {
            id: sensor,
            align: crate::types::Align::Reject,
        };
        ir.graph.insert(
            NodeId(900),
            TaskNode::GetSensor {
                sensor,
                ty: sensor_ty,
            },
        );
        ir.graph.insert(
            NodeId(901),
            TaskNode::Norm {
                kind: NormKind::L2,
                ty: ty(ElemType::F32, 3, Unit::Length, Frame::World),
            },
        );
        ir.graph.connect(NodeId(900), "value", NodeId(901), "value");
        assert!(codes_of(&ir.validate()).contains(&codes::TYPE_014));
    }

    #[test]
    fn determinism_rules_fire() {
        let mut ir = fixture();
        ir.graph.insert(
            NodeId(900),
            TaskNode::GetRandom {
                stream: String::new(),
                dist: Distribution::Constant(0.0),
                shape: Shape::new([1]),
            },
        );
        ir.graph.insert(
            NodeId(901),
            TaskNode::GetRandom {
                stream: "undeclared".to_owned(),
                dist: Distribution::Constant(0.0),
                shape: Shape::new([1]),
            },
        );
        ir.graph.insert(
            NodeId(902),
            TaskNode::MathFn {
                func: MathFunc::Sin,
                approx: false,
                ty: ty(ElemType::F32, 1, Unit::Angle, Frame::World),
            },
        );
        ir.graph.insert(
            NodeId(903),
            TaskNode::Reduce {
                op: ReduceOp::Sum,
                axis: 0,
                unordered: true,
                ty: ty(ElemType::F32, 3, Unit::Length, Frame::World),
            },
        );
        let diags = ir.validate();
        let got = codes_of(&diags);
        for want in [
            codes::DET_001,
            codes::DET_021,
            codes::DET_010,
            codes::DET_030,
        ] {
            assert!(got.contains(&want), "{want} missing from {got:?}");
        }
    }

    #[test]
    fn observation_channel_must_be_declared() {
        let mut ir = fixture();
        ir.observation_spec.channels.clear();
        assert_eq!(codes_of(&ir.validate()), vec![codes::TASK_001]);
    }

    #[test]
    fn nan_parameter_is_hash_001() {
        let mut ir = fixture();
        ir.graph.insert(
            NodeId(900),
            TaskNode::Reward {
                name: "bad".to_owned(),
                weight: f64::NAN,
                aggregation: Aggregation::Sum,
                ty: ty(ElemType::F32, 1, Unit::Dimensionless, Frame::World),
            },
        );
        assert!(codes_of(&ir.validate()).contains(&codes::HASH_001));
    }

    #[test]
    fn task_hash_ignores_node_ids_but_graph_hash_does_not() {
        let ir = fixture();
        let renamed = relabel(&ir);
        assert_eq!(ir.task_hash().unwrap(), renamed.task_hash().unwrap());
        assert_ne!(
            ir.task_graph_hash().unwrap(),
            renamed.task_graph_hash().unwrap()
        );
    }

    #[test]
    fn task_hash_follows_a_reward_weight() {
        let a = fixture();
        let b = task_ir(&[("reach".to_owned(), 2.0, 0.5)], 7);
        assert_ne!(a.task_hash().unwrap(), b.task_hash().unwrap());
    }

    #[test]
    fn task_hash_follows_the_scene_and_the_observation_spec() {
        let base = fixture();
        let mut scene = base.clone();
        scene.scene.scene_hash = [1u8; 32];
        assert_ne!(base.task_hash().unwrap(), scene.task_hash().unwrap());

        let other_dof = task_ir(&[("reach".to_owned(), 1.0, 0.5)], 6);
        assert_ne!(base.task_hash().unwrap(), other_dof.task_hash().unwrap());
    }

    #[test]
    fn expr_evaluates_deterministically() {
        let ports: BTreeMap<String, f64> = [("d".to_owned(), 0.01)].into_iter().collect();
        let near = Expr::Compare {
            op: CmpOp::Lt,
            lhs: Box::new(Expr::Port("d".to_owned())),
            rhs: Box::new(Expr::Const(0.02)),
        };
        assert_eq!(near.eval(&ports), Some(1.0));
        assert_eq!(
            Expr::Clamp {
                value: Box::new(Expr::Port("d".to_owned())),
                lo: 0.0,
                hi: 0.005,
            }
            .eval(&ports),
            Some(0.005)
        );
        // Missing port, division by zero and an inverted clamp are all `None`, never NaN.
        assert_eq!(Expr::Port("nope".to_owned()).eval(&ports), None);
        assert_eq!(
            Expr::Arith {
                op: ArithOp::Div,
                lhs: Box::new(Expr::Port("d".to_owned())),
                rhs: Box::new(Expr::Const(0.0)),
            }
            .eval(&ports),
            None
        );
        assert_eq!(
            Expr::Clamp {
                value: Box::new(Expr::Const(1.0)),
                lo: 1.0,
                hi: 0.0,
            }
            .eval(&ports),
            None
        );
    }

    proptest! {
        #[test]
        fn generated_tasks_are_valid(ir in arbitrary_task_ir()) {
            let diags = ir.validate();
            prop_assert!(diags.is_empty(), "{diags:?}");
        }

        #[test]
        fn generated_task_hash_survives_relabelling(ir in arbitrary_task_ir()) {
            prop_assert_eq!(ir.task_hash().unwrap(), relabel(&ir).task_hash().unwrap());
        }
    }
}
