//! The headless half of the editor (spec 23): everything the tabs display, with no egui and
//! no window, so CI judges it without a display.

pub mod edit;
pub mod graph_view;
pub mod image_view;
pub mod inspector;
pub mod launch;
pub mod live_run;
pub mod palette;
pub mod recent;
pub mod replay_view;
pub mod run_view;
pub mod search;
pub mod telemetry_view;
