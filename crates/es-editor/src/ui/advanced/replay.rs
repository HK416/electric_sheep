//! The Results pane's Replay panel (packet M7/E2): the selected cell's `.estraj`, played in the
//! scene canvas of `viewport`.

use std::path::Path;

use eframe::egui;
use es_eval::run_dir::RunDir;

use super::viewport::{replay_canvas, Canvas};
use crate::app::EditorApp;
use crate::model::dialogs;
use crate::model::i18n;
use crate::model::labels::Browse;
use crate::model::replay_view::ReplayView;
use crate::model::run_view;

/// The control rate a recorded `.estraj` tick is worth. 50 Hz is the demo deployment's
/// `rate.control`; a run directory carries no Deployment IR to read it from, and playing at
/// the wrong rate only changes how fast the arm appears to move.
pub(crate) const REPLAY_RATE_HZ: f64 = 50.0;

impl EditorApp {
    /// The Replay panel (packet M7/E2): the selected cell's `.estraj`, posed and projected by
    /// [`ReplayView`] and painted as one mesh. The gestures map to the model's pure camera
    /// functions and to `advance`; nothing is decided here.
    pub(super) fn replay_panel(&mut self, ui: &mut egui::Ui) {
        let dt = f64::from(ui.input(|i| i.stable_dt));
        let lang = self.settings.lang;
        let scene_browse = dialogs::AVAILABLE.then(|| self.browse_label(Browse::Scene));
        let Self {
            run,
            replay,
            replay_cell,
            replay_texture,
            scene_path,
            frames_path,
            camera,
            status,
            ..
        } = self;
        let selected = run
            .as_ref()
            .and_then(RunDir::selected_cell)
            .filter(|c| c.has_traj)
            .map(|c| c.name.clone());

        ui.horizontal(|ui| {
            ui.label(i18n::t(lang, "replay.scene"));
            ui.add(
                egui::TextEdit::singleline(scene_path)
                    .hint_text(i18n::t(lang, "replay.scene.hint"))
                    .desired_width(220.0),
            );
            if let Some(label) = scene_browse {
                if ui.button(label).clicked() {
                    if let Some(path) = dialogs::pick(Browse::Scene) {
                        *scene_path = path.display().to_string();
                    }
                }
            }
            // `es eval run --frames <dir>` writes wherever it was told, which is usually a
            // sibling of the run directory; the model re-scans when this is applied.
            ui.label(i18n::t(lang, "replay.frames"));
            let field = ui.add(
                egui::TextEdit::singleline(frames_path)
                    .hint_text(i18n::t(lang, "replay.frames.hint"))
                    .desired_width(220.0),
            );
            if field.lost_focus() {
                if let Some(run) = run.as_mut() {
                    run.set_frames_root(frames_path.trim());
                    *status = format!("{}: {}", run.dir.display(), run_view::status(run));
                }
            }
            let replay_label = i18n::t(lang, "replay.replay");
            let label = selected.as_ref().map_or_else(
                || replay_label.to_owned(),
                |name| format!("{replay_label} {name}"),
            );
            if ui
                .add_enabled(selected.is_some(), egui::Button::new(label))
                .clicked()
            {
                let (Some(run), Some(cell)) = (run.as_ref(), selected.as_ref()) else {
                    return;
                };
                match ReplayView::open(Path::new(scene_path.trim()), &run.traj_path(cell)) {
                    Ok(view) => {
                        *status = format!("{cell}: {} tick(s) replayed", view.ticks());
                        cell.clone_into(replay_cell);
                        *replay = Some(view);
                        *replay_texture = Canvas::default();
                    }
                    Err(e) => *status = e.to_string(),
                }
            }
            if let Some(view) = replay.as_mut() {
                if ui
                    .button(i18n::t(
                        lang,
                        if view.playing {
                            "replay.pause"
                        } else {
                            "replay.play"
                        },
                    ))
                    .clicked()
                {
                    view.playing = !view.playing;
                }
                if ui.button("|<").clicked() {
                    view.step(-1);
                }
                if ui.button(">|").clicked() {
                    view.step(1);
                }
                ui.add(egui::Slider::new(&mut view.speed, 0.1..=4.0).text("x"));
            }
        });

        let Some(view) = replay.as_mut() else {
            ui.label(i18n::t(lang, "replay.empty"));
            return;
        };
        view.advance(dt, REPLAY_RATE_HZ);
        let last = view.ticks().saturating_sub(1);
        ui.horizontal(|ui| {
            ui.label(format!(
                "{replay_cell}  tick {}/{last}  {:.2} s",
                view.tick,
                view.tick as f64 / REPLAY_RATE_HZ
            ));
            ui.add(egui::Slider::new(&mut view.tick, 0..=last).text("tick"));
        });

        replay_canvas(ui, lang, ui.available_size(), view, camera, replay_texture);
        if view.playing {
            ui.ctx().request_repaint();
        }
    }
}
