//! `es-ir` (layer 6): the five typed IRs, their shared type system, graph skeleton,
//! diagnostics and hash chain. See `docs/ARCHITECTURE.ko.md` spec 5 to spec 10 and the work
//! packets `docs/packets/M0/P18.md` .. `P29.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-math`, `es-core` and external crates
//! only — never on the compiler, a physics backend, a tensor library or anything UI. Layout
//! belongs in `.eslayout` sidecars, not here (rule 7).
//!
//! Foundation (P18, P19, P27): [`types`], [`image`], [`diag`], [`codes`], [`graph`], [`hash`].
//! The five IRs (P20..P24) fill in [`task`], [`observation`], [`learning`], [`deployment`],
//! [`evaluation`], and the boundary rules (P26) fill in [`cross`].

pub mod codes;
pub mod diag;
pub mod graph;
pub mod hash;
pub mod image;
pub mod norm;
pub mod types;

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
