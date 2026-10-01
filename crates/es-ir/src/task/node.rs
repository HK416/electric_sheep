//! The spec 6.3 node set: [`TaskNode`], the ports each node declares and its canonical
//! parameters (the `task_hash` input).

use es_core::StableId;
use serde::{Deserialize, Serialize};

use super::{
    wdbg, wid, ActionSpace, Aggregation, ArithOp, CmpOp, Distribution, JointQuantity, LogicOp,
    MathFunc, NormKind, ReduceOp, TerminationKind,
};
use crate::graph::{IrNode, Port};
use crate::hash::CanonWriter;
use crate::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

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
    /// Binds a graph value to one channel of
    /// [`TaskIr::observation_spec`](super::TaskIr::observation_spec). Declaration only —
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
