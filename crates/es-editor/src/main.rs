//! `es-editor [bundle.esb|run-dir] [--attach <addr> [--token <t>]]` — the editor shell of
//! spec 23.
//!
//! The window is the only thing this file owns. Everything it shows is
//! [`es_editor::model`], which runs headless.

use es_editor::model::telemetry_view::{attach, replay};
use es_editor::EditorApp;

fn main() -> eframe::Result<()> {
    let (mut path, mut addr, mut token) = (None, None, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--attach" => addr = args.next(),
            "--token" => token = args.next(),
            _ => path = Some(arg),
        }
    }
    // Spec 23.1: the editor is a client of a running process. `--attach` dials one at start,
    // which is the same call the Telemetry tab's Connect button makes — a refusal is reported
    // and the editor still opens, since a viewer that will not start is worse than an empty
    // one.
    let (source, status) = match &addr {
        Some(addr) => match attach(addr, token.as_deref().unwrap_or_default()) {
            Ok(source) => (source, Some(format!("attached to {addr}"))),
            Err(e) => {
                eprintln!("es-editor --attach: {e}");
                (replay(Vec::new()), Some(e))
            }
        },
        None => (replay(Vec::new()), None),
    };

    eframe::run_native(
        "Electric Sheep editor",
        eframe::NativeOptions::default(),
        Box::new(move |cc| {
            // `cc.storage` is the previous session's recent list (packet M7/E3), read before
            // a path on the command line is opened so that path joins the list.
            let mut app = EditorApp::new(source).with_storage(cc.storage);
            if let Some(path) = &path {
                app = app.with_path(path);
            }
            if let Some(status) = status {
                app = app.with_status(status);
            }
            Ok(Box::new(app))
        }),
    )
}
