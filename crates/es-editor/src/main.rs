//! `es-editor [bundle.esb]` — the editor shell of spec 23.
//!
//! The window is the only thing this file owns. Everything it shows is
//! [`es_editor::model`], which runs headless.

use es_editor::model::telemetry_view::replay;
use es_editor::EditorApp;

fn main() -> eframe::Result<()> {
    // Telemetry source: empty until `es_telemetry::transport` lands (spec 25.1 owns the
    // socket). The app only knows a `FnMut() -> Option<Message>`, so that is a one-line
    // change here and nowhere else.
    let mut app = EditorApp::new(replay(Vec::new()));
    if let Some(path) = std::env::args().nth(1) {
        app = app.with_bundle(&path);
    }
    eframe::run_native(
        "Electric Sheep editor",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    )
}
