//! `es-editor` (layer 13): the editor shell of spec 23 — a step bar over a dock of panes
//! (packet M12/Y10), the layered graph view
//! (read-only and editable), telemetry, and the before/after image pair.
//!
//! Layer rule (spec 4.2 rule 4): this crate may depend on anything, and **nothing may depend
//! on it**; nothing but it may depend on `es-editor-model` (layer 12), whose [`model`] it
//! re-exports under the path it had before packet M12/R4 split the two. `cargo xtask
//! layering` enforces all of it.
//!
//! Spec 23.1: the editor does not host training. It is a client that attaches to a running
//! process. Everything it knows arrives as an IR bundle on disk or as telemetry messages.
//!
//! Spec 23.4 stages 1 and 2. Read-only came first and still stands on its own: it carries
//! most of the debugging value and is what survives spec 1.9 cut 5, which takes the *editable*
//! graph. Editing ([`model::edit`], [`model::palette`]) was added beside it — layout
//! persistence and undo/redo, which read-only needs none of.
//!
//! The split is deliberate: [`model`] is headless and fully tested; [`app`] (the
//! `eframe::App` glue and its state) and [`ui`] (the shell and the Advanced panes, packet
//! M12/Y10) are a thin egui layer over it that CI only compiles. See
//! `docs/design/editor-shell.md` and `docs/design/editor-redesign.md` section 6.1.
//!
//! It is also meant to be usable by someone who is not an engineer (packet M7/E6): every
//! visible string is a key in [`model::i18n`]'s two tables rather than a literal in
//! [`app`], every column and flag has a plain name in [`model::labels`] with the raw one a
//! hover away, the system's own CJK font is found by [`model::fonts`] so Korean renders, and
//! the Open dialog is wrapped in [`model::dialogs`] behind a target-specific feature.
//!
//! A run can be watched as it happens as well as read after the fact:
//! `es-editor --attach <addr>` is a client of `es eval run --telemetry <addr>`, and
//! [`model::live_run::LiveRun`] folds that stream into the same
//! [`es_eval::run_dir::CellRow`]/[`es_eval::run_dir::Timeline`] the
//! Run tab draws a finished run with (packet M7/E4).
#![forbid(unsafe_code)]

pub mod app;
pub mod gpu;
pub mod ui;

pub use es_editor_model::model;

pub use app::EditorApp;
pub use model::edit::{Edit, EditSession};
pub use model::graph_view::LayeredGraph;
pub use model::image_view::{BeforeAfter, ImagePair};
pub use model::inspector::{Field, Inspector, Widget};
pub use model::launch::{Kind as LaunchKind, LaunchModel};
pub use model::live_run::LiveRun;
pub use model::palette::{Palette, Registries};
pub use model::recent::{classify, Kind, Recent};
pub use model::replay_view::ReplayView;
pub use model::search::Search;
pub use model::telemetry_view::TelemetryModel;
