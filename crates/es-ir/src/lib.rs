//! `es-ir` (layer 6): the five typed IRs, their shared type system, graph skeleton,
//! diagnostics and hash chain. See `docs/ARCHITECTURE.ko.md` spec 5 to spec 10 and the work
//! packets `docs/packets/M0/P18.md` .. `P29.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-math`, `es-core` and external crates
//! only — never on the compiler, a physics backend, a tensor library or anything UI. Layout
//! belongs in `.eslayout` sidecars, not here (rule 7).
//!
//! Foundation (P18, P19, P27): [`types`], [`image`], [`diag`], [`codes`], [`graph`], [`hash`].
//! The first four live in `es-ir-types` since P-M0-R4 (spec 1.5 context budget) and are
//! re-exported here unchanged.
//! The five IRs (P20..P24) fill in [`task`], [`observation`], [`learning`], [`deployment`],
//! [`evaluation`], and the boundary rules (P26) fill in [`cross`]. IR-C, the Task IR control
//! graph (spec 6.2), is [`control`]; see `docs/design/control-graph.md`.

pub mod graph;
pub mod hash;
pub mod norm;

// Split out to `es-ir-types` for the spec 1.5 context budget (P-M0-R4), re-exported here so
// every `es_ir::codes::..` / `es_ir::types::..` path still resolves.
pub use es_ir_types::{codes, diag, image, types};

pub mod control;
pub mod cross;
pub mod deployment;
pub mod evaluation;
pub mod factory;
pub mod learning;
pub mod observation;
pub mod serial;
pub mod task;

/// Every generator the Appendix B.7 properties need, in one place. Enabled by `--features
/// testing`; `proptest` is optional, so a release build never pulls it in.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    pub use crate::deployment::testing::arbitrary_deployment_ir;
    pub use crate::evaluation::testing::arbitrary_evaluation_ir;
    pub use crate::learning::testing::arbitrary_learning_graph;
    pub use crate::norm::testing::*;
    pub use crate::observation::testing::arbitrary_observation_ir;
    pub use crate::task::testing::arbitrary_task_ir;
}

pub use control::{ControlGraph, ControlNode, ControlNodeId, RepeatUntil, SubTaskRef};
pub use diag::{DiagCode, Diagnostic, Severity};
pub use graph::{Dir, Edge, Graph, IrNode, NodeId, Port, PortRef};
pub use hash::{
    canonical_hash, canonical_order, CanonWriter, ChangedComponent, DatasetHash,
    HardwareCapability, HashChain,
};
pub use image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    Rect, ShutterModel,
};
pub use norm::{canon_graph, canon_learning, canon_observation, canon_task};
pub use types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit, UnitPowers};
