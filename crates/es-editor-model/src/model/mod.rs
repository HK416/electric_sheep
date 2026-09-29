//! The headless half of the editor (spec 23): everything the tabs display, with no window,
//! so CI judges it without a display.
//!
//! [`fonts`] is the one module that names an egui type - a `FontDefinitions` is the only way
//! to say "this face, last" - and it still needs no display, so it is judged here like the
//! rest rather than being left to `app.rs` (spec 28.10 rule 3).

pub mod dialogs;
pub mod edit;
pub mod fonts;
pub mod graph_view;
pub mod health;
pub mod home;
pub mod i18n;
pub mod image_view;
pub mod inspector;
pub mod labels;
pub mod launch;
pub mod layout;
pub mod live_run;
pub mod outcome;
pub mod palette;
pub mod preview;
pub mod project;
pub mod recent;
pub mod replay_view;
pub mod results;
pub mod run_view;
pub mod scene_view;
pub mod search;
pub mod teach;
pub mod telemetry_view;
pub mod template;
pub mod train_view;
pub mod watch;
pub mod workflow;
