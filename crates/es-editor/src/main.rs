//! `es-editor [bundle.esb|run-dir]` — the editor shell of spec 23.
//!
//! The window is the only thing this file owns. Everything it shows is
//! [`es_editor::model`], which runs headless.

use es_editor::model::telemetry_view::replay;
use es_editor::EditorApp;

fn main() -> eframe::Result<()> {
    let arg = std::env::args().nth(1);
    eframe::run_native(
        "Electric Sheep editor",
        eframe::NativeOptions::default(),
        Box::new(move |cc| {
            // Telemetry source: empty until `es_telemetry::transport` lands (spec 25.1 owns
            // the socket). The app only knows a `FnMut() -> Option<Message>`, so that is a
            // one-line change here and nowhere else.
            //
            // `cc.storage` is the previous session's recent list (packet M7/E3), read before
            // a path on the command line is opened so that path joins the list.
            let mut app = EditorApp::new(replay(Vec::new())).with_storage(cc.storage);
            if let Some(path) = &arg {
                app = app.with_path(path);
            }
            Ok(Box::new(app))
        }),
    )
}
