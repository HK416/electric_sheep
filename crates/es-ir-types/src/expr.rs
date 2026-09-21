//! The parameter vocabulary and the authoring expression form the IRs are written in
//! (spec 6.3 node parameters, spec 6.5 expressions).
//!
//! None of it knows what a graph is: these are the leaf values a `TaskNode`, a `SubTaskRef` or
//! a `Branch` condition is built out of, so they live below `es-ir` with the rest of the
//! vocabulary (spec 1.5 context budget; `docs/packets/M4/P-M4-S16.md`). `es-ir` re-exports
//! every item here from `es_ir::task`, its original path.

use serde::{Deserialize, Serialize};

use crate::canon::CanonWriter;
use crate::types::Unit;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointQuantity {
    Position,
    Velocity,
    Torque,
}

impl JointQuantity {
    pub fn unit(self) -> Unit {
        match self {
            Self::Position => Unit::Angle,
            Self::Velocity => Unit::AngularVelocity,
            Self::Torque => Unit::Torque,
        }
    }
}

/// Element-wise arithmetic. `Mul` and `Div` scale by a dimensionless operand; a product of two
/// different units is an explicit conversion and is not expressible as one node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicOp {
    And,
    Or,
    Xor,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    Sum,
    Mean,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormKind {
    L1,
    L2,
    Linf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathFunc {
    Abs,
    Sign,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Tanh,
}

impl MathFunc {
    /// Transcendentals need `es-math::approx` in task kernels (spec 6.6, `DET-010`).
    pub fn is_transcendental(self) -> bool {
        !matches!(self, Self::Abs | Self::Sign | Self::Sqrt)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Aggregation {
    Sum,
    Mean,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationKind {
    Success,
    Failure,
    Timeout,
}

/// Action space declaration only; the full `ActionSpec` of spec 9.2 lives in Deployment IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionSpace {
    JointPosition,
    JointVelocity,
    JointTorque,
    EePose,
    EeDelta,
    Gripper,
}

/// Sampling distribution for randomization and reset (spec 6.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Distribution {
    Constant(f64),
    Uniform { lo: f64, hi: f64 },
    LogUniform { lo: f64, hi: f64 },
    Normal { mean: f64, std: f64 },
    Choice(Vec<f64>),
}

impl Distribution {
    pub fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::Constant(v) => {
                w.str("Constant");
                w.f64(*v);
            }
            Self::Uniform { lo, hi } | Self::LogUniform { lo, hi } => {
                w.str(if matches!(self, Self::Uniform { .. }) {
                    "Uniform"
                } else {
                    "LogUniform"
                });
                w.f64(*lo);
                w.f64(*hi);
            }
            Self::Normal { mean, std } => {
                w.str("Normal");
                w.f64(*mean);
                w.f64(*std);
            }
            Self::Choice(vs) => {
                w.str("Choice");
                w.seq(vs.len());
                for v in vs {
                    w.f64(*v);
                }
            }
        }
    }
}

// --- expressions (spec 6.5) ------------------------------------------------------------------

/// The authoring-side expression form. The parser of spec 6.5 expands one of these into a
/// `TaskNode` subgraph in M1 lowering; no Rhai interpreter is ever called. Loops, assignment
/// and function definitions are parse errors, so they have no representation here — and there
/// is no transcendental variant (spec 6.6 `DET-010` makes those a `TaskNode::MathFn` with
/// `approx = true`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Reference to a named graph port.
    Port(String),
    Const(f64),
    Arith {
        op: ArithOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Compare {
        op: CmpOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Clamp {
        value: Box<Expr>,
        lo: f64,
        hi: f64,
    },
    /// Square root. **Not a `DET-010` transcendental** (spec 6.6): IEEE 754 requires `sqrt` to
    /// be correctly rounded, so every target computes the same bits from the same input -- it
    /// is one hardware instruction, like `Add` and `Mul`, and needs no `es-math::approx`
    /// polynomial. It is here because `Norm { kind: L2 }` lowers to it and nothing else does
    /// (`docs/design/batch-domains.md` section 6).
    Sqrt(Box<Expr>),
}

impl Expr {
    /// Evaluates against named port values.
    ///
    /// Deterministic by construction: every operation is an IEEE `f32`-representable `f64`
    /// arithmetic op in a fixed order, comparisons yield exactly `1.0` or `0.0`, and anything
    /// that would produce a non-finite value — a missing port, division by zero, an inverted
    /// clamp range, an overflow — is `None` rather than a `NaN` that no hash can encode.
    pub fn eval(&self, ports: &std::collections::BTreeMap<String, f64>) -> Option<f64> {
        let finite = |v: f64| v.is_finite().then_some(v);
        match self {
            Self::Port(name) => ports.get(name).copied().and_then(finite),
            Self::Const(v) => finite(*v),
            Self::Arith { op, lhs, rhs } => {
                let (a, b) = (lhs.eval(ports)?, rhs.eval(ports)?);
                finite(match op {
                    ArithOp::Add => a + b,
                    ArithOp::Sub => a - b,
                    ArithOp::Mul => a * b,
                    ArithOp::Div => a / b,
                    ArithOp::Min => a.min(b),
                    ArithOp::Max => a.max(b),
                })
            }
            Self::Compare { op, lhs, rhs } => {
                let (a, b) = (lhs.eval(ports)?, rhs.eval(ports)?);
                // Exact IEEE comparison is the rule, not an approximation of one: two runs
                // must agree bit for bit, and both operands are already finite here.
                #[allow(clippy::float_cmp)]
                let t = match op {
                    CmpOp::Lt => a < b,
                    CmpOp::Le => a <= b,
                    CmpOp::Gt => a > b,
                    CmpOp::Ge => a >= b,
                    CmpOp::Eq => a == b,
                    CmpOp::Ne => a != b,
                };
                Some(f64::from(u8::from(t)))
            }
            Self::Clamp { value, lo, hi } => {
                let v = value.eval(ports)?;
                (lo <= hi).then(|| v.clamp(*lo, *hi)).and_then(finite)
            }
            // A negative radicand is `None` for the same reason a division by zero is: the
            // alternative is a `NaN`, and no hash encodes one.
            Self::Sqrt(value) => {
                let v = value.eval(ports)?;
                (v >= 0.0).then(|| v.sqrt()).and_then(finite)
            }
        }
    }
}
