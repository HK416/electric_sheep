//! The headless half of the editor (spec 23): everything the tabs display, with no egui and
//! no window, so CI judges it without a display.

pub mod edit;
pub mod graph_view;
pub mod image_view;
pub mod palette;
pub mod run_view;
pub mod telemetry_view;
