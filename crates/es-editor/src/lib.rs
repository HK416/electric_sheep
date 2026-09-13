//! `es-editor` (layer 12): the editor shell of spec 23 — tabs, the **read-only** layered
//! graph view, telemetry, and the before/after image pair.
//!
//! Layer rule (spec 4.2 rule 4): this crate may depend on anything, and **nothing may depend
//! on it**. `cargo xtask layering` enforces both halves.
//!
//! Spec 23.1: the editor does not host training. It is a client that attaches to a running
//! process. Everything it knows arrives as an IR bundle on disk or as telemetry messages.
//!
//! Spec 23.4 stage 1 only: read-only. Editing (stage 2, M3) needs layout persistence,
//! undo/redo, search and large-graph performance; read-only needs none of them and carries
//! most of the debugging value — and it is spec 1.9 cut 5, where the *editable* graph goes
//! and the read-only view stays.
//!
//! The split is deliberate: [`model`] is headless and fully tested, [`app`] is a thin egui
//! layer over it that CI only compiles. See `docs/design/editor-shell.md`.
#![forbid(unsafe_code)]

pub mod app;
pub mod model;

pub use app::EditorApp;
pub use model::graph_view::LayeredGraph;
pub use model::image_view::{BeforeAfter, ImagePair, Rgb8Image};
pub use model::telemetry_view::TelemetryModel;
