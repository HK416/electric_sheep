//! `es-editor-model` (layer 12): the headless half of the editor of spec 23 — every
//! view-model the panes draw, and the two string tables (`i18n/en.toml`, `i18n/ko.toml`) every
//! visible word comes from. It opens no window, so CI judges all of it with `cargo test`.
//!
//! Layer rule (spec 4.2 rule 4): **nothing depends on this crate except `es-editor`** (layer
//! 13), the egui shell that draws it and re-exports [`model`] under its old path. `cargo xtask
//! layering` enforces it. The two were one crate until packet M12/R4 split them at the line
//! `docs/design/editor-shell.md` section 2 already drew; nothing was renamed, so a type is
//! still `model::<module>::<Type>`. Packet M17/R1 moved the Advanced graph's models (`edit`,
//! `graph_view`, `image_view`, `inspector`, `palette`, `search`) to `es-editor-graph`, a sibling
//! on layer 12 this crate does not know; `es-editor`'s `model` holds both under that path.
//!
//! Three egui-family names remain here, none of which needs a display: `model/fonts.rs` builds
//! an `egui::FontDefinitions`, and `model/layout.rs` holds an `egui_dock::DockState` and
//! persists it through `eframe::Storage`.
#![forbid(unsafe_code)]

pub mod model;
