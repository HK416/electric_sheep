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
pub mod types;

pub mod cross;
pub mod deployment;
pub mod evaluation;
pub mod factory;
pub mod learning;
pub mod observation;
pub mod serial;
pub mod task;

pub use diag::{DiagCode, Diagnostic, Severity};
pub use graph::{Dir, Edge, Graph, IrNode, NodeId, Port, PortRef};
pub use hash::{
    canonical_hash, CanonWriter, ChangedComponent, DatasetHash, HardwareCapability, HashChain,
};
pub use image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    Rect, ShutterModel,
};
pub use types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit, UnitPowers};
