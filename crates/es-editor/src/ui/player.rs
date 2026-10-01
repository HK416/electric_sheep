//! One attempt of an evaluation run, playing (packet M12/Y13): the recorded motion re-posed on
//! the scene and seen from a camera the person turns, the pictures the policy was given, or both
//! beside each other, over the attempt's timeline. ⑤ plays a finished run's attempts with it,
//! ③ a checkpoint preview's (packet M13/Z5a).
//!
//! Drawing only. Which views an attempt can be played in and which picture a tick shows are
//! [`crate::model::results`]'s, under test.

use std::path::Path;

use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Vec2};
use es_eval::run_dir::RunDir;
use es_render::raster::{Camera, BACKGROUND};

use crate::model::i18n::{t, Lang};
use crate::model::replay_view::ReplayView;
use crate::model::results::{self, View};
use crate::ui::advanced::{paint_timeline, replay_canvas, rgb_texture, Canvas, REPLAY_RATE_HZ};

/// One attempt, playing.
pub(crate) struct Player {
    pub(crate) cell: String,
    frames: usize,
    views: Vec<View>,
    view: View,
    replay: Option<ReplayView>,
    /// Why the motion could not be replayed, in the loader's own words.
    note: Option<String>,
    /// The playhead when there is no motion to carry it.
    index: usize,
    camera: Camera,
    picture: Canvas,
    eye: Option<(usize, Option<egui::TextureHandle>)>,
}

impl Player {
    /// The attempt `cell` of the run in `dir`, its motion re-posed on `scene` when it kept one.
    pub(crate) fn open(dir: &RunDir, scene: Option<&Path>, cell: String) -> Self {
        let row = dir.cells().iter().find(|c| c.name == cell);
        let frames = row.map_or(0, |r| r.frames);
        let (replay, note) = match (row.is_some_and(|r| r.has_traj), scene) {
            (true, Some(scene)) => match ReplayView::open(scene, &dir.traj_path(&cell)) {
                Ok(view) => (Some(view), None),
                Err(e) => (None, Some(e.to_string())),
            },
            _ => (None, None),
        };
        let views = results::views(replay.is_some(), frames);
        Self {
            view: views.first().copied().unwrap_or(View::Outside),
            views,
            cell,
            frames,
            replay,
            note,
            index: 0,
            camera: crate::model::scene_view::camera_for(scene),
            picture: Canvas::default(),
            eye: None,
        }
    }

    fn tick(&self) -> usize {
        self.replay.as_ref().map_or(self.index, |r| r.tick)
    }

    fn len(&self) -> usize {
        self.replay.as_ref().map_or(self.frames, ReplayView::ticks)
    }
}

/// The attempt picker, the views, play and speed, the picture and the timeline. An attempt
/// picked from the list lands in `pick`; the caller opens it.
pub(crate) fn play(
    lang: Lang,
    ui: &mut egui::Ui,
    dt: f64,
    dir: &RunDir,
    p: &mut Player,
    pick: &mut Option<String>,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(t(lang, "results.attempt"));
        egui::ComboBox::from_id_salt("results-attempt")
            .selected_text(&p.cell)
            .show_ui(ui, |ui| {
                for cell in dir.cells() {
                    if ui
                        .selectable_label(cell.name == p.cell, &cell.name)
                        .clicked()
                    {
                        *pick = Some(cell.name.clone());
                    }
                }
            });
        ui.separator();
        for view in p.views.clone() {
            ui.selectable_value(&mut p.view, view, t(lang, view.key()));
        }
        if let Some(replay) = p.replay.as_mut() {
            ui.separator();
            let key = if replay.playing {
                "replay.pause"
            } else {
                "replay.play"
            };
            if ui.button(t(lang, key)).clicked() {
                replay.playing = !replay.playing;
            }
            for speed in results::SPEEDS {
                ui.selectable_value(&mut replay.speed, speed, format!("{speed}\u{d7}"));
            }
        }
    });
    if p.views.is_empty() {
        ui.weak(t(lang, "results.nothing_recorded"));
    }
    if let Some(note) = &p.note {
        ui.weak(note);
    }
    if let Some(replay) = p.replay.as_mut() {
        replay.advance(dt, REPLAY_RATE_HZ);
        if replay.playing {
            ui.ctx().request_repaint();
        }
    }

    let size = Vec2::new(
        ui.available_width(),
        (ui.available_height() * 0.6).max(160.0),
    );
    let half = Vec2::new(size.x / 2.0 - 4.0, size.y);
    match (p.view, p.replay.as_ref()) {
        (View::Outside, Some(replay)) => {
            replay_canvas(ui, lang, size, replay, &mut p.camera, &mut p.picture);
        }
        (View::SideBySide, Some(replay)) => {
            ui.horizontal(|ui| {
                replay_canvas(ui, lang, half, replay, &mut p.camera, &mut p.picture);
                let tick = replay.tick;
                eye(ui, half, dir, &p.cell, p.frames, tick, &mut p.eye);
            });
        }
        (View::Eye, _) => {
            let tick = p.tick();
            eye(ui, size, dir, &p.cell, p.frames, tick, &mut p.eye);
        }
        (View::Outside | View::SideBySide, None) => {}
    }

    let len = p.len();
    if len > 0 {
        let n = (ui.available_width() / 4.0) as usize;
        paint_timeline(lang, ui, &dir.timeline(&p.cell).buckets(n));
        let mut at = p.tick();
        ui.spacing_mut().slider_width = (ui.available_width() - 60.0).max(80.0);
        if ui.add(egui::Slider::new(&mut at, 0..=len - 1)).changed() {
            match p.replay.as_mut() {
                Some(replay) => {
                    replay.tick = at;
                    replay.playing = false;
                }
                None => p.index = at,
            }
        }
    }
}

/// The policy's own picture at `tick`, fitted into `size` with its aspect kept.
fn eye(
    ui: &mut egui::Ui,
    size: Vec2,
    dir: &RunDir,
    cell: &str,
    frames: usize,
    tick: usize,
    cache: &mut Option<(usize, Option<egui::TextureHandle>)>,
) {
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter()
        .rect_filled(rect, 0.0, Color32::from_gray(BACKGROUND));
    let Some(index) = results::frame_at(tick, frames) else {
        return;
    };
    if cache.as_ref().is_none_or(|(i, _)| *i != index) {
        let texture = dir
            .frame(cell, index)
            .map(|img| rgb_texture(ui.ctx(), "results-eye", &img));
        *cache = Some((index, texture));
    }
    if let Some((_, Some(texture))) = cache {
        let px = texture.size_vec2();
        let scale = (size.x / px.x).min(size.y / px.y);
        ui.painter().image(
            texture.id(),
            Rect::from_center_size(rect.center(), px * scale),
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }
}
