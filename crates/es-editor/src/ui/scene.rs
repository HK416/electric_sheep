//! ① Scene and ② Teach of a template project (packet M12/Y15, `docs/design/editor-redesign.md`
//! sections 3 and 6.3): the template's scene through the CPU raster in the centre, what is in it
//! or how the robot is taught on the left, and the template's own words on the right. Read-only.
//!
//! Drawing only. The template, the scene at its initial pose, what is in it and the method's
//! words are [`crate::model::scene_view`]'s, under test.

use std::path::PathBuf;

use eframe::egui;
use egui::{Color32, RichText};
use es_render::raster::Camera;

use crate::app::EditorApp;
use crate::model::home::Mark;
use crate::model::i18n::{fill, t, Strings};
use crate::model::layout::Pane;
use crate::model::scene_view::{self, ScenePreview};
use crate::model::template::{templates_root, Template};
use crate::model::workflow::Phase;
use crate::ui::advanced::{scene_canvas, Canvas, Posed};

/// ① and ② between frames. It belongs to one project; opening another reads its scene afresh.
#[derive(Default)]
pub(crate) struct State {
    project: Option<PathBuf>,
    step: Option<(Template, Result<ScenePreview, String>)>,
    camera: Option<Camera>,
    picture: Canvas,
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
    if state.project.as_ref() != Some(&open.project.root) {
        *state = State {
            project: Some(open.project.root.clone()),
            step: scene_view::open_step(&open.project, templates_root()),
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
    let word = |key: &str| Strings::get(lang).t(key).to_owned();
    match (pane, phase, preview) {
        (Pane::Viewport, _, Err(why)) => {
            ui.label(fill(lang, "setup.load_failed", &[why]));
        }
        (Pane::Viewport, _, Ok(preview)) => {
            ui.weak(t(lang, "setup.orbit_hint"));
            let camera =
                (state.camera).get_or_insert_with(|| scene_view::camera_of(Some(template)));
            let size = ui.available_size();
            let posed = Posed::Scene(preview);
            scene_canvas(ui, lang, size, posed, camera, &mut state.picture);
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
        }
    }
    true
}
