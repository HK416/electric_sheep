//! `es-editor [project-dir|bundle.esb|run-dir] [--attach <addr> [--token <t>]] [--step <1-5>]
//! [--viewport fast|material|pt] [--fps | --orbit-demo] [--physics-preview]
//! [--edit-demo copy|select|edit|undo|drag|corner|add-menu|add-box|add-robot]` — the editor
//! shell of spec 23.
//!
//! `--fps` prints each viewport's frames per second on stderr; `--orbit-demo` also turns every
//! viewport's camera a little each frame (packet M16/H9's measurement and captures). Both keep
//! the session's window and dock state in the temp directory, not the person's own.
//! `--physics-preview` has ① start its physics preview as soon as it shows the scene (packet
//! M17/G4's captures). `--edit-demo` has ① make the editable copy, select the cube, resize and
//! recolour it and undo that, up to the stage named (packet M17/G5's captures); `drag` holds the
//! cube's move handle mid-drag and `corner` lets it go and selects the front camera (packet
//! M17/G6's); `add-menu|add-box|add-robot` open ①'s Add menu, add a box and add the library's
//! first robot (packet M17/G7's); it too keeps the session's state in the temp directory.
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
use es_editor::model::viewport::Mode;
use es_editor::EditorApp;

fn main() -> eframe::Result<()> {
    let (mut path, mut addr, mut token, mut step) = (None, None, None, None);
    let mut look = None;
    let (mut demo, mut physics, mut edit) = (None, false, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--attach" => addr = args.next(),
            "--token" => token = args.next(),
            "--step" => step = args.next().and_then(|s| s.parse::<usize>().ok()),
            // The viewport's look at start (packet M16/H8); the selector changes it after.
            "--viewport" => look = args.next().as_deref().and_then(Mode::parse),
            // ① starts its physics preview at once (packet M17/G4's captures).
            "--physics-preview" => physics = true,
            // ①'s scripted edits (packet M17/G5's captures).
            "--edit-demo" => edit = args.next(),
            "--fps" => demo = Some(false),
            "--orbit-demo" => demo = Some(true),
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

    let mut options = eframe::NativeOptions::default();
    if demo.is_some() || edit.is_some() {
        options.persistence_path = Some(std::env::temp_dir().join("es-editor-demo"));
    }
    let result = eframe::run_native(
        "Electric Sheep editor",
        options,
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
            if let Some(look) = look {
                es_editor::ui::advanced::set_viewport_mode(&cc.egui_ctx, look);
            }
            if let Some(orbit) = demo {
                es_editor::ui::advanced::set_demo(&cc.egui_ctx, orbit);
            }
            if physics {
                es_editor::ui::scene::preview_at_start(&cc.egui_ctx);
            }
            if let Some(stage) = edit {
                es_editor::ui::scene::edit_demo(&cc.egui_ctx, stage);
            }
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
    );
    // The viewport's device, released before the process goes (packet M16/H9).
    es_editor::gpu::shutdown();
    result
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
