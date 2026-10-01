//! `es-editor [project-dir|bundle.esb|run-dir] [--attach <addr> [--token <t>]] [--step <1-5>]`
//! — the editor shell of spec 23.
//!
//! `es-editor --import <project-dir> --template <id> ...` makes a project of runs that ran
//! outside the editor and opens it; the grammar is [`es_editor::model::import`]'s (packet
//! M16/H7).
//!
//! The window is the only thing this file owns. Everything it shows is
//! [`es_editor::model`], which runs headless.

use es_editor::model::import::Import;
use es_editor::model::telemetry_view::{attach, replay};
use es_editor::model::template;
use es_editor::EditorApp;

fn main() -> eframe::Result<()> {
    let (mut path, mut addr, mut token, mut step) = (None, None, None, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--attach" => addr = args.next(),
            "--token" => token = args.next(),
            "--step" => step = args.next().and_then(|s| s.parse::<usize>().ok()),
            "--import" => match import(&args.by_ref().collect::<Vec<_>>()) {
                Ok(project) => path = Some(project),
                Err(e) => {
                    eprintln!("es-editor --import: {e}");
                    std::process::exit(2);
                }
            },
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
            // `cc.storage` is the previous session's recent list (packet M7/E3) and dock
            // arrangements (M12/Y10), read before a path on the command line is opened so
            // that path joins the list.
            let mut app = EditorApp::new(source).with_storage(cc.storage);
            // The system CJK face and the persisted text size, before anything is drawn: a
            // Korean label or a Korean path in a field renders as boxes without it (packet
            // M7/E6). Which font, where in the fallback chain, and what a machine with none
            // is told are all `model/fonts.rs`'s and the string tables'.
            app.apply_style(&cc.egui_ctx);
            if let Some(path) = &path {
                app = app.with_path(path);
            }
            if let Some(step) = step {
                app = app.with_step(step);
            }
            if let Some(status) = status {
                app = app.with_status(status);
            }
            Ok(Box::new(app))
        }),
    )
}

/// `--import`: the project made, said on stdout, and its folder to open.
fn import(args: &[String]) -> Result<String, String> {
    let spec = Import::parse(args)?;
    let root = template::templates_root().ok_or("no checkout: templates/ was not found")?;
    let found = (template::load(&root).0.into_iter())
        .find(|t| t.id == spec.template)
        .ok_or_else(|| format!("no template {}", spec.template))?;
    let (project, choice) = spec.run(&found, &root).map_err(|e| e.to_string())?;
    println!("project: {}", project.root.display());
    match choice {
        Some(c) => println!("teacher: run {:03}, step {}", c.run, c.step),
        None => println!("teacher: none chosen"),
    }
    Ok(project.root.display().to_string())
}
