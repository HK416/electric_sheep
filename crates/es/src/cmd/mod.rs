//! One module per top-level subcommand (spec 2.5, 10.5, 11.1).
//!
//! The verbs that only read a finished artifact live in `es-tools` (packet
//! `docs/packets/M10/W3b-es-tools-split.md`) and are re-exported here under their old paths,
//! so `crate::cmd::backend::load_scene` and `crate::cmd::showcase::run` still resolve.

/// `es video showcase` (packet M5/V9): only with the `render` feature, because it is the one
/// subcommand that opens a Vulkan device.
#[cfg(feature = "render")]
pub use es_tools::showcase;
pub use es_tools::{backend, bench, evidence, gap, video};

pub mod check_deps;
pub mod cycle;
pub mod dataset;
pub mod eval;
pub mod generate;
pub mod import;
pub mod ir;
pub mod r#loop;
pub mod mcp;
pub mod policy;
pub mod scene;
pub mod task;
pub mod telemetry;
pub mod train;
