//! ① Scene and ② Teach of a template project (packet M12/Y15, `docs/design/editor-redesign.md`
//! sections 3 and 6.3): the template's scene through the CPU raster in the centre, what is in it
//! or how the robot is taught on the left, and the template's own words on the right. Read-only,
//! with "make an editable copy" when the template has a scene document; ① of an editable project
//! (packet M17/G5) is [`crate::ui::author`]'s. ①'s physics preview (packet M17/G4) plays
//! `es scene simulate`'s motion where the scene was — of the edited scene, when it is edited.
//!
//! Drawing only. The template, the scene at its initial pose, what is in it, the method's
//! words and the preview's argv and lines are [`crate::model::scene_view`]'s, under test.

use std::path::{Path, PathBuf};

use eframe::egui;
use egui::{Color32, RichText};
use es_editor_scene::{BackendKind, Regen, SCENE_FILE, SPEC_FILE};
use es_render::raster::Camera;

use crate::app::EditorApp;
use crate::model::home::{self, Mark};
use crate::model::i18n::{fill, t, Lang, Strings};
use crate::model::layout::Pane;
use crate::model::results::SPEEDS;
use crate::model::scene_view::{self, Physics, ScenePreview};
use crate::model::template::{self, templates_root, EditableDocs, Generated, Template};
use crate::model::workflow::Phase;
use crate::ui::advanced::{replay_canvas, scene_canvas, Canvas, Posed};
use crate::ui::author::Author;
use crate::ui::corner::Corner;

/// ① and ② between frames. It belongs to one project; opening another reads its scene afresh
/// and ends a preview still computing.
#[derive(Default)]
pub(crate) struct State {
    project: Option<PathBuf>,
    step: Option<(Template, Result<ScenePreview, String>)>,
    /// ① of an editable project (packet M17/G5): its scene document under edit, or why it
    /// could not be opened (or made by "make an editable copy").
    author: Option<Result<Author, String>>,
    /// Why the last "make an editable copy" failed.
    copy_error: Option<String>,
    camera: Option<Camera>,
    picture: Canvas,
    physics: Option<Physics>,
    /// The preview's pictures, apart from the static scene's: both start at tick 0.
    played: Canvas,
    /// The policy camera's view over a template's scene (packet M17/G6).
    corner: Corner,
}

/// A template's bundle Task IR and Observation IR: the cameras its policy sees.
fn bundle_of(template: &Template) -> Option<[PathBuf; 2]> {
    let root = templates_root()?;
    let b = template.bundle.as_ref()?;
    Some([root.join(&b.task), root.join(&b.observation)])
}

// --- an authored project (packet M17/G9) ------------------------------------------------------

/// What `es-editor-scene` found of an editable project's `generated/`, in `es-editor-model`'s
/// terms: `unsaved` when ① holds a task that is not saved.
pub fn generated(regen: &Regen, unsaved: bool) -> Generated {
    match regen {
        _ if unsaved => Generated::Unsaved,
        Regen::NoSpec => Generated::NoSpec,
        Regen::Written(files) => Generated::Fresh(files.clone()),
        Regen::Failed(why) => Generated::Failed(why.clone()),
        Regen::NotYet | Regen::Stale => Generated::Stale,
    }
}

/// Once a frame: the open project's `generated/` as ① has it now, and — when that changed (a
/// save, an edit, a failed generation) — what ② to ⑤ run, read again.
pub(crate) fn sync(app: &mut EditorApp) {
    let Some(open) = app.project.as_mut() else {
        return;
    };
    let author = (app.scene.project.as_ref() == Some(&open.project.root))
        .then_some(app.scene.author.as_ref())
        .flatten();
    let Some(Ok(author)) = author else {
        return;
    };
    let m = &author.model;
    let now = generated(m.generated(), m.dirty() && m.spec().is_some());
    if open.generated.as_ref() == Some(&now) {
        return;
    }
    let source = template::source(&open.project, templates_root(), Some(&now));
    open.generated = Some(now);
    open.watch.set_source(&open.project, source);
    app.teacher.forget();
    app.teach.forget();
    app.results = crate::ui::results::State::default();
}

/// A new project's own documents, before `Project::create` writes its `project.toml`: a saved
/// template's copied whole, a template without documents of its own (`empty.toml`) as ①'s
/// editable copy makes them; nothing for any other template.
pub fn new_documents(root: &Path, template: &Template, repo: &Path) -> Result<(), String> {
    let Some(e) = template
        .editable
        .as_ref()
        .filter(|_| template.bundle.is_none())
    else {
        return Ok(());
    };
    std::fs::create_dir_all(root).map_err(|e| format!("{}: {e}", root.display()))?;
    if template.base.is_some() {
        let dir = Path::new(&e.scene).parent().unwrap_or(Path::new("."));
        return es_editor_scene::copy::documents(dir, root);
    }
    let spec = e.spec.as_ref().map(|s| repo.join(s));
    es_editor_scene::make_editable(root, &repo.join(&e.scene), spec.as_deref())
}

/// The template `--edit-demo save-template` saves.
const DEMO: &str = "My push task";

/// "Save as template" of the project at `root`, made from `base`, as `name`: its saved documents
/// copied into a new folder under `documents`' templates, and its `template.toml` written last.
pub fn save_template(
    root: &Path,
    base: &Template,
    name: &str,
    documents: Option<&Path>,
) -> Result<PathBuf, String> {
    let dir = home::saved_folder(documents, name);
    es_editor_scene::copy::documents(root, &dir)?;
    let spec = root.join(SPEC_FILE).is_file().then(|| SPEC_FILE.to_owned());
    let editable = EditableDocs {
        scene: SCENE_FILE.to_owned(),
        spec,
    };
    template::write_saved(base, name, &dir, editable)?;
    Ok(dir)
}

/// The "save as template" dialog, while ①'s toolbar has it open: a name and Save. Saving saves
/// the edited documents first. `true` once a template was written.
fn save_as(ctx: &egui::Context, lang: Lang, author: &mut Author, base: &Template) -> bool {
    let Some(mut name) = author.save_as.take() else {
        return false;
    };
    let (mut save, mut cancel) = (false, false);
    let modal = egui::Modal::new(egui::Id::new("save-as-template")).show(ctx, |ui| {
        ui.heading(t(lang, "author.save_template"));
        ui.horizontal(|ui| {
            ui.label(t(lang, "author.save_template.name"));
            ui.text_edit_singleline(&mut name);
        });
        ui.horizontal(|ui| {
            let ok = !name.trim().is_empty();
            let button = egui::Button::new(t(lang, "author.save_template.save"));
            save = ui.add_enabled(ok, button).clicked();
            cancel = ui.button(t(lang, "home.cancel")).clicked();
        });
    });
    if save {
        let edited = if author.model.dirty() {
            author.model.save()
        } else {
            Ok(())
        };
        let saved = edited.and_then(|()| {
            let docs = home::documents_dir();
            save_template(author.model.root(), base, name.trim(), docs.as_deref())
        });
        let done = saved.is_ok();
        author.saved_as = Some(saved);
        return done;
    }
    if !(cancel || modal.should_close()) {
        author.save_as = Some(name);
    }
    false
}

fn at_start_id() -> egui::Id {
    egui::Id::new("physics-preview-at-start")
}

fn demo_id() -> egui::Id {
    egui::Id::new("scene-edit-demo")
}

/// `es-editor --physics-preview`: ① starts its physics preview as soon as it shows a scene,
/// for captures where a synthetic click does not reach the window (packet M17/G4).
pub fn preview_at_start(ctx: &egui::Context) {
    ctx.data_mut(|d| d.insert_temp(at_start_id(), true));
}

/// `es-editor --edit-demo copy|select|edit|undo|drag|corner`: ① makes the editable copy if the
/// project has none, then selects the cube, then resizes and recolours it, then undoes that —
/// each stage after the ones before it (packet M17/G5's captures). `drag` holds the cube's move
/// handle mid-drag; `corner` lets it go and selects the front camera (packet M17/G6's).
/// `add-menu`, `add-box` and `add-robot` open the Add menu, add a box and add the library's
/// first robot where the view looks (packet M17/G7's). `sentences`, `refuse` and `new-task` are
/// the task's (packet M17/G8's, `sentence::demo`). `authored` builds a task on the empty project
/// and saves it, and `save-template` saves the project as a template of the person's (packet
/// M17/G9's).
pub fn edit_demo(ctx: &egui::Context, stage: String) {
    ctx.data_mut(|d| d.insert_temp(demo_id(), stage));
}

/// The backends a project's commands are checked against: the reference one ①'s physics
/// preview runs, and `MuJoCo` Warp when the template trains there.
fn backends(template: Option<&Template>) -> Vec<BackendKind> {
    let warp = template.is_some_and(|t| t.needs.iter().any(|n| n == "mjwarp"));
    let mut out = vec![BackendKind::MuJoCoCpu];
    out.extend(warp.then_some(BackendKind::MjWarp));
    out
}

/// "Make an editable copy" of `template`'s scene into the project at `root`, opened.
fn copy(root: &Path, template: &Template) -> Result<Author, String> {
    let (Some(e), Some(repo)) = (&template.editable, templates_root()) else {
        return Err(template.id.clone());
    };
    let spec = e.spec.as_ref().map(|s| repo.join(s));
    es_editor_scene::make_editable(root, &repo.join(&e.scene), spec.as_deref())?;
    let mut author = Author::open(root, backends(Some(template)), bundle_of(template))?;
    author.model.regenerate();
    Ok(author)
}

/// ①'s and ②'s step panel, viewport and summary; `false` for every other pane and step.
pub(crate) fn draw(app: &mut EditorApp, ui: &mut egui::Ui, pane: Pane) -> bool {
    let Some(open) = app.project.as_ref() else {
        return false;
    };
    let phase = open.phase;
    let panes = matches!(pane, Pane::StepPanel | Pane::Viewport | Pane::Summary);
    if !panes || !matches!(phase, Phase::Scene | Phase::Teach) {
        return false;
    }
    let lang = app.settings.lang;
    let state = &mut app.scene;
    let root = &open.project.root;
    if state.project.as_ref() != Some(root) {
        let step = scene_view::open_step(&open.project, templates_root());
        let template = step.as_ref().map(|(t, _)| t);
        let author = (es_editor_scene::is_editable(root))
            .then(|| Author::open(root, backends(template), template.and_then(bundle_of)));
        *state = State {
            project: Some(root.clone()),
            step,
            author,
            ..State::default()
        };
    }
    let Some((template, preview)) = &state.step else {
        ui.label(fill(
            lang,
            "setup.no_template",
            &[&open.project.file.template],
        ));
        return true;
    };
    let demo = ui.ctx().data_mut(|d| d.remove_temp::<String>(demo_id()));
    if let Some(stage) = &demo {
        if state.author.is_none() && template.editable.is_some() {
            match copy(root, template) {
                Ok(author) => state.author = Some(Ok(author)),
                Err(why) => state.copy_error = Some(why),
            }
        }
        if let (Some(Ok(author)), false) = (&mut state.author, stage == "copy") {
            let camera = state
                .camera
                .get_or_insert_with(|| scene_view::camera_of(Some(template)));
            if stage == "save-template" {
                // Packet M17/G9's captures: the project saved as a template of the person's.
                let docs = home::documents_dir();
                let saved = save_template(author.model.root(), template, DEMO, docs.as_deref());
                author.saved_as = Some(saved);
                app.home.reload();
            } else {
                author.demo(stage, camera);
            }
        }
    }
    if phase == Phase::Scene {
        if let Some(author) = &mut state.author {
            let camera =
                (state.camera).get_or_insert_with(|| scene_view::camera_of(Some(template)));
            match author {
                Ok(author) => {
                    editable(
                        ui,
                        lang,
                        pane,
                        author,
                        camera,
                        &mut state.physics,
                        &mut state.picture,
                        &mut state.played,
                    );
                    // Once a frame (the viewport's turn): the "save as template" dialog.
                    if pane == Pane::Viewport && save_as(ui.ctx(), lang, author, template) {
                        app.home.reload();
                    }
                }
                Err(why) => {
                    ui.label(fill(lang, "author.open_failed", &[why]));
                }
            }
            return true;
        }
    }
    let word = |key: &str| Strings::get(lang).t(key).to_owned();
    match (pane, phase, preview) {
        (Pane::Viewport, _, Err(why)) => {
            ui.label(fill(lang, "setup.load_failed", &[why]));
        }
        (Pane::Viewport, _, Ok(preview)) => {
            if phase == Phase::Scene {
                physics_row(ui, lang, preview, &mut state.physics, &mut state.played);
            }
            ui.weak(t(lang, "setup.orbit_hint"));
            let camera =
                (state.camera).get_or_insert_with(|| scene_view::camera_of(Some(template)));
            let size = ui.available_size();
            let played = (state.physics.as_mut().filter(|_| phase == Phase::Scene))
                .and_then(|p| Some((p.rate_hz, p.replay()?)));
            if let Some((rate, view)) = played {
                playback(ui, lang, size, rate, view, camera, &mut state.played);
            } else {
                // The policy's camera in the corner (packet M17/G6), as the bundle declares it.
                let (scene, corner) = (preview.source().0, &mut state.corner);
                let mut overlay = |r: &egui::Response, p: &egui::Painter, _: &Camera| {
                    let cameras = || match bundle_of(template) {
                        Some([task, obs]) => es_editor_scene::policy::bundle(&task, &obs, &scene),
                        None => Ok(Vec::new()),
                    };
                    corner.paint(p, r.rect, lang, &scene, 0, None, cameras);
                    false
                };
                let posed = Posed::Scene(preview);
                let picture = &mut state.picture;
                scene_canvas(ui, lang, size, posed, camera, picture, Some(&mut overlay));
            }
        }
        (Pane::StepPanel, Phase::Scene, _) => {
            ui.heading(t(lang, "setup.contents"));
            let contents = preview
                .as_ref()
                .map(ScenePreview::contents)
                .unwrap_or_default();
            let robot = contents
                .robot
                .map(|root| format!("{} ({root})", template.robot));
            for (key, names) in [
                ("setup.robot", robot.into_iter().collect()),
                ("setup.objects", contents.objects),
                ("setup.cameras", contents.cameras),
            ] {
                ui.add_space(6.0);
                ui.strong(t(lang, key));
                if names.is_empty() {
                    ui.weak(t(lang, "setup.none"));
                }
                ui.indent(key, |ui| {
                    for name in &names {
                        ui.label(name);
                    }
                });
            }
        }
        (Pane::StepPanel, _, _) => {
            ui.heading(t(lang, "teach.heading"));
            let method = scene_view::method_key(template.method);
            ui.label(fill(lang, method, &[&template.robot]));
            let count = open.watch.settings.demonstrations.to_string();
            ui.label(fill(lang, "teach.demonstrations", &[&count]));
        }
        _ => {
            if phase == Phase::Scene {
                ui.heading(word(&template.name));
                ui.label(word(&template.summary));
            }
            if let Some(notice) = &template.notice {
                let [r, g, b] = Mark::Optional.colour();
                let notice = format!("\u{26a0} {}", word(notice));
                ui.label(RichText::new(notice).color(Color32::from_rgb(r, g, b)));
            }
            ui.weak(t(lang, "setup.read_only"));
            if phase == Phase::Scene && template.editable.is_some() {
                ui.add_space(6.0);
                let button = ui.button(t(lang, "author.copy"));
                if button.on_hover_text(t(lang, "author.copy.hint")).clicked() {
                    match copy(root, template) {
                        Ok(author) => state.author = Some(Ok(author)),
                        Err(why) => state.copy_error = Some(why),
                    }
                }
                if let Some(why) = &state.copy_error {
                    ui.label(fill(lang, "author.copy_failed", &[why]));
                }
            }
        }
    }
    true
}

/// ① of an editable project: the hierarchy left, the viewport with undo / redo / save and the
/// physics preview of the edited scene in the centre, the inspector right.
#[allow(clippy::too_many_arguments)] // the pane's share of `State`, borrowed apart
fn editable(
    ui: &mut egui::Ui,
    lang: Lang,
    pane: Pane,
    author: &mut Author,
    camera: &mut Camera,
    physics: &mut Option<Physics>,
    picture: &mut Canvas,
    played: &mut Canvas,
) {
    match pane {
        Pane::StepPanel => author.hierarchy(ui, lang, camera),
        Pane::Viewport => {
            author.toolbar(ui, lang, camera);
            let preview = match author.preview() {
                Ok(p) => p,
                Err(why) => {
                    ui.label(fill(lang, "setup.load_failed", &[&why]));
                    return;
                }
            };
            physics_row(ui, lang, &preview, physics, played);
            let size = ui.available_size();
            let replay = (physics.as_mut()).and_then(|p| Some((p.rate_hz, p.replay()?)));
            if let Some((rate, view)) = replay {
                playback(ui, lang, size, rate, view, camera, played);
            } else {
                // Picking, the handles and the corner (packet M17/G6).
                let mut overlay = |r: &egui::Response, p: &egui::Painter, c: &Camera| {
                    author.overlay(lang, r, p, c)
                };
                let posed = Posed::Scene(&preview);
                scene_canvas(ui, lang, size, posed, camera, picture, Some(&mut overlay));
            }
        }
        // The inspector, or the task as sentences (packet M17/G8).
        _ => crate::ui::sentence::summary(ui, lang, author),
    }
}

/// The preview button, what its child is doing, and once the motion plays: play / pause, the
/// speeds and the way back to the static scene.
fn physics_row(
    ui: &mut egui::Ui,
    lang: Lang,
    preview: &ScenePreview,
    physics: &mut Option<Physics>,
    played: &mut Canvas,
) {
    if let Some(p) = physics.as_mut() {
        p.poll();
        if p.running() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    let at_start = ui.ctx().data_mut(|d| d.remove_temp::<bool>(at_start_id()));
    ui.horizontal_wrapped(|ui| {
        let running = physics.as_ref().is_some_and(Physics::running);
        let button = ui.add_enabled(!running, egui::Button::new(t(lang, "setup.physics")));
        let start = button
            .on_hover_text(t(lang, "setup.physics.hint"))
            .clicked();
        if start || at_start == Some(true) {
            let es = crate::model::launch::es_binary().path;
            *physics = Some(Physics::start(preview, &es, scene_view::physics_out()));
            *played = Canvas::default();
        }
        let Some(p) = physics.as_mut() else {
            return;
        };
        if p.running() {
            ui.spinner();
        }
        if let Some((key, args)) = p.line() {
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            ui.label(fill(lang, key, &args));
        }
        if let Some(view) = p.replay() {
            let key = if view.playing {
                "replay.pause"
            } else {
                "replay.play"
            };
            if ui.button(t(lang, key)).clicked() {
                // Play at the end starts over.
                if !view.playing && view.tick + 1 >= view.ticks() {
                    view.tick = 0;
                }
                view.playing = !view.playing;
            }
            for speed in SPEEDS {
                ui.selectable_value(&mut view.speed, speed, format!("{speed}\u{d7}"));
            }
        }
        if ui.button(t(lang, "setup.physics.back")).clicked() {
            *physics = None;
        }
    });
}

/// The preview's motion in the viewport, under the look selector, and its timeline in seconds.
fn playback(
    ui: &mut egui::Ui,
    lang: Lang,
    size: egui::Vec2,
    rate: f64,
    view: &mut crate::model::replay_view::ReplayView,
    camera: &mut Camera,
    canvas: &mut Canvas,
) {
    view.advance(f64::from(ui.input(|i| i.stable_dt)), rate);
    let len = view.ticks();
    if view.playing && view.tick + 1 >= len {
        view.playing = false;
    }
    if view.playing {
        ui.ctx().request_repaint();
    }
    let slider = ui.spacing().interact_size.y + ui.spacing().item_spacing.y;
    let size = egui::Vec2::new(size.x, (size.y - slider).max(1.0));
    replay_canvas(ui, lang, size, view, camera, canvas);
    let mut at = view.tick;
    let secs = |tick: usize| tick as f64 / rate;
    ui.horizontal(|ui| {
        ui.label(format!(
            "{:.2} / {:.2} s",
            secs(at),
            secs(len.saturating_sub(1))
        ));
        ui.spacing_mut().slider_width = ui.available_width().max(80.0);
        let slider = egui::Slider::new(&mut at, 0..=len.saturating_sub(1)).show_value(false);
        if ui.add(slider).changed() {
            view.tick = at;
            view.playing = false;
        }
    });
}
