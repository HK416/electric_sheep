//! The headless half of the editor (spec 23): everything the tabs display, with no egui and
//! no window, so CI judges it without a display.

pub mod graph_view;
pub mod image_view;
pub mod telemetry_view;
