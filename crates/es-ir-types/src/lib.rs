//! `es-ir-types` (layer 2): the vocabulary the five IRs of `es-ir` are written in — the common
//! type system, the image spec, diagnostics and the canonical byte encoder. See
//! `docs/ARCHITECTURE.ko.md` spec 5.3, spec 5.4, spec 7.2 and the packet
//! `docs/packets/M0/P-M0-R4.md`.
//!
//! Split out of `es-ir` purely to keep both crates inside the spec 1.5 context budget. Nothing
//! here knows what a graph *is*: no `Graph`, no `IrNode`, no IR node kinds. `es-ir` re-exports
//! every item of this crate at its original path, so `es_ir::types::PortType` and
//! `es_ir_types::types::PortType` name the same thing and no downstream crate had to change.
//!
//! Layer rule (spec 4.2): may depend on `es-math`, `es-core` and external crates only.

use serde::{Deserialize, Serialize};

pub mod canon;
pub mod codes;
pub mod diag;
pub mod image;
pub mod types;

/// Authoring identity of a node. **Not** part of the semantic hash (spec 11.2): it appears in
/// `*_graph_hash`, never in `*_hash`.
///
/// It lives here, below the graph itself, because [`diag::Diagnostic`] points at one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

pub use canon::CanonWriter;
pub use diag::{DiagCode, Diagnostic, Severity};
pub use image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    Rect, ShutterModel,
};
pub use types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit, UnitPowers};
