//! One module per top-level subcommand (spec 2.5, 10.5, 11.1).

pub mod backend;
pub mod bench;
pub mod check_deps;
pub mod cycle;
pub mod dataset;
pub mod eval;
pub mod evidence;
pub mod gap;
pub mod generate;
pub mod import;
pub mod ir;
pub mod r#loop;
pub mod mcp;
pub mod policy;
/// `es video showcase` (packet M5/V9): only with the `render` feature, because it is the one
/// subcommand that opens a Vulkan device.
#[cfg(feature = "render")]
pub mod showcase;
pub mod task;
pub mod train;
pub mod video;
