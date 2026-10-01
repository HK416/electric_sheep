//! `es-editor-graph` (layer 12): the Advanced graph's headless models (spec 23.2-23.4) — the
//! layered IR graph, its edits, the inspector, the node palette and search, and the
//! Observation IR's before/after images. All of it reads an opened IR bundle and nothing
//! else: no project, no run, no string table, so CI judges it with `cargo test`.
//!
//! Layer rule (spec 4.2 rule 4): **nothing depends on this crate except `es-editor`** (layer
//! 13), which re-exports these modules under its `model` path beside `es-editor-model`'s, so
//! a type is still `model::<module>::<Type>` there. It knows neither `es-editor-model` nor
//! `es-editor-scene`, its siblings on layer 12 (rule 1). It is a crate of its own because
//! `es-editor-model` reached 9,169 of its 10,000 lines (packet M17/R1); the modules moved
//! unchanged.
#![forbid(unsafe_code)]

pub mod edit;
pub mod graph_view;
pub mod image_view;
pub mod inspector;
pub mod palette;
pub mod search;
