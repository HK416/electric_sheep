//! The Live signals pane (spec 23.1, spec 23.3): attaching to a running process, its performance
//! table, streams and events, and the Training section's curves, checkpoint marks and sample.

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};

use super::rgb_texture;
use crate::app::EditorApp;
use crate::model::i18n;
use crate::model::labels;
use crate::model::telemetry_view;
use crate::model::train_view::{Plot, Series};

impl EditorApp {
    pub(crate) fn telemetry_tab(&mut self, ui: &mut egui::Ui) {
        // Spec 23.1: the editor attaches to a running process. The address and the token
        // are typed here and dialled by the model, which owns every error string.
        ui.horizontal(|ui| {
            ui.label(self.t("live.attach"))
                .on_hover_text(self.t("live.attach.hint"));
            ui.add(
                egui::TextEdit::singleline(&mut self.attach_addr)
                    .hint_text(crate::model::launch::DEFAULT_TELEMETRY)
                    .desired_width(160.0),
            );
            let token_hint = self.t("live.token");
            ui.add(
                egui::TextEdit::singleline(&mut self.attach_token)
                    .hint_text(token_hint)
                    .password(true)
                    .desired_width(160.0),
            );
            if ui.button(self.t("live.connect")).clicked() {
                match telemetry_view::attach(&self.attach_addr, &self.attach_token) {
                    Ok(source) => {
                        self.source = source;
                        self.status = format!("attached to {}", self.attach_addr.trim());
                    }
                    Err(e) => self.status = e,
                }
            }
        });
        ui.separator();
        // A training run's curves first: its speed table is all "not measured" (packet M16/H4).
        let learning_first = !self.telemetry.train.is_empty();
        if learning_first {
            self.training_section(ui);
            ui.separator();
        }
        ui.heading(self.t("live.performance"))
            .on_hover_text(self.t("live.performance.hint"));
        if self.telemetry.received == 0 {
            ui.label(self.t("live.empty"));
        }
        egui::Grid::new("metrics").striped(true).show(ui, |ui| {
            for (name, value) in self.telemetry.metric_rows() {
                // The row is named by the metric it is, and hovers the raw spelling the
                // producer sends (packet M7/E6).
                let plain = labels::metric_by_name(name)
                    .map_or(name, |m| labels::metric_label(self.settings.lang, m));
                ui.label(plain).on_hover_text(name);
                ui.label(value.map_or_else(
                    || self.t("value.not_measured").to_owned(),
                    |v| format!("{v:.3}"),
                ));
                ui.end_row();
            }
        });
        ui.separator();
        if !learning_first {
            self.training_section(ui);
            ui.separator();
        }
        ui.heading(self.t("live.streams"));
        egui::Grid::new("streams").striped(true).show(ui, |ui| {
            for (key, tick, value) in self.telemetry.latest() {
                ui.label(format!("stream {}[{}]", key.stream.0, key.index));
                ui.label(format!("tick {tick}"));
                ui.label(format!("{value:.4}"));
                ui.end_row();
            }
        });
        ui.separator();
        ui.heading(self.t("live.events"));
        for e in self.telemetry.events.iter().rev().take(200) {
            ui.label(format!("[{}] {} {:?}", e.tick, e.kind, e.fields));
        }
    }

    /// The Live tab's Training section (packet M7/E7): the learning curve, the learning rate,
    /// the checkpoint marks and the tensor the network is fitting.
    ///
    /// Wiring only. The curve's scale, its normalisation, the marks' positions and the ETA are
    /// [`crate::model::train_view::TrainView`]'s and are judged headlessly; this maps a
    /// rectangle and paints a polyline. **There is no plotting crate** — two `line_segment`
    /// runs over points already in the unit square is the whole of it (spec 28.10 rule 3).
    fn training_section(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.lang;
        let ctx = ui.ctx().clone();
        ui.heading(self.t("live.training"))
            .on_hover_text(self.t("live.training.hint"));
        if self.telemetry.train.is_empty() {
            ui.label(self.t("live.training.empty"));
            return;
        }
        let train = &self.telemetry.train;
        // Where the run has got to, how fast, and how much is left -- the model's numbers,
        // formatted by the tables (packet M7/E6).
        ui.horizontal_wrapped(|ui| {
            if let Some(step) = train.step() {
                ui.label(match train.total() {
                    Some(total) => {
                        i18n::fill(lang, "live.step", &[&step.to_string(), &total.to_string()])
                    }
                    None => i18n::fill(lang, "live.step_only", &[&step.to_string()]),
                });
            }
            if let Some(eta) = train.total().and_then(|t| train.eta(t)) {
                ui.label(i18n::fill(
                    lang,
                    "live.eta",
                    &[&format!("{:.0}s", eta.as_secs_f64())],
                ));
            }
            if let Some(rate) = train.throughput() {
                ui.label(i18n::fill(
                    lang,
                    "live.throughput",
                    &[&format!("{rate:.0}")],
                ));
            }
        });
        // The packed marks, one row each: step and `policy_hash` (packet M16/H4 -- a finished
        // run's folder is opened for these as much as for its curve).
        if !train.checkpoints().is_empty() {
            let title = i18n::fill(
                lang,
                "live.checkpoints",
                &[&train.checkpoints().len().to_string()],
            );
            egui::CollapsingHeader::new(title)
                .id_salt("train-marks")
                .show(ui, |ui| {
                    for (step, hash) in train.checkpoints() {
                        ui.monospace(format!("{step:>8}  {hash}"));
                    }
                });
        }
        let mut log = train.log_scale;
        ui.checkbox(&mut log, self.t("live.log_scale"));
        let loss = train.plot(Series::Loss, log);
        let lr = train.plot(Series::Lr, false);
        // An `[rl]` run's learning (packet M16/H4): drawn only for a run that has any.
        let rl: Vec<(Series, Option<Plot>)> = if train.rl().step.is_empty() {
            Vec::new()
        } else {
            Series::RL.map(|s| (s, train.plot(s, false))).to_vec()
        };
        let (sample, samples) = (train.sample().cloned(), train.samples());
        self.telemetry.train.log_scale = log;

        // An `[rl]` run is judged by its learning, not its loss, so that comes first; and a
        // rollout has no sample image to show, so its column is not offered.
        let picture = rl.is_empty() || sample.is_some();
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                if picture {
                    ui.set_max_width((ui.available_width() - 180.0).max(240.0));
                }
                if !rl.is_empty() {
                    ui.strong(i18n::t(lang, "live.rl"))
                        .on_hover_text(i18n::t(lang, "live.rl.hint"));
                }
                for (i, (series, plot)) in rl.into_iter().enumerate() {
                    ui.label(i18n::t(lang, series.key()));
                    match plot {
                        Some(plot) => {
                            paint_curve(ui, &plot, RL_COLOURS[i % RL_COLOURS.len()], 70.0);
                        }
                        None => {
                            ui.label(i18n::t(lang, "value.not_measured"));
                        }
                    }
                }
                for (key, plot, colour) in [
                    ("live.loss", loss, Color32::from_rgb(120, 200, 255)),
                    ("live.lr", lr, Color32::from_rgb(200, 160, 255)),
                ] {
                    ui.label(i18n::t(lang, key));
                    match plot {
                        Some(plot) => {
                            paint_curve(ui, &plot, colour, 90.0);
                        }
                        None => {
                            ui.label(i18n::t(lang, "results.no_events"));
                        }
                    }
                }
            });
            if !picture {
                return;
            }
            // What the network is looking at, beside the curve: the batch's own image input
            // after augmentation, uploaded once per sample rather than once per repaint.
            ui.vertical(|ui| {
                ui.label(i18n::t(lang, "live.sample"))
                    .on_hover_text(i18n::t(lang, "live.sample.hint"));
                let Some(image) = sample else {
                    ui.label(i18n::t(lang, "live.sample.empty"));
                    return;
                };
                let key = format!("sample#{samples}");
                if !self.run_frames.contains_key(&key) {
                    self.run_frames
                        .insert(key.clone(), rgb_texture(&ctx, &key, &image));
                }
                if let Some(texture) = self.run_frames.get(&key) {
                    let scale = (160.0 / texture.size_vec2().x).max(1.0);
                    ui.image(egui::load::SizedTexture::new(
                        texture.id(),
                        texture.size_vec2() * scale,
                    ));
                }
            });
        });
    }
}

/// One curve, painted (packet M7/E7). **No plotting crate**: the points arrive in the unit
/// square from [`crate::model::train_view::TrainView::plot`], and this maps them onto a
/// rectangle, joins them, and draws a vertical line where a checkpoint was packed. The two
/// numbers beside it are the range the model normalised against, in the series' own units.
/// `height` in points: a strip in the Live pane, the whole centre while ③ trains (M12/Y12).
/// Returns the rectangle the unit square was mapped onto, for ③'s preview marks (M13/Z5a).
/// One colour per `Series::RL` curve, in its order.
pub(crate) const RL_COLOURS: [Color32; 5] = [
    Color32::from_rgb(120, 220, 140),
    Color32::from_rgb(250, 210, 90),
    Color32::from_rgb(160, 200, 255),
    Color32::from_rgb(255, 150, 120),
    Color32::from_rgb(230, 120, 200),
];

pub(crate) fn paint_curve(ui: &mut egui::Ui, plot: &Plot, colour: Color32, height: f32) -> Rect {
    let (response, painter) =
        ui.allocate_painter(Vec2::new(ui.available_width(), height), Sense::hover());
    let rect = response.rect.shrink(4.0);
    let at = |p: &[f32; 2]| {
        Pos2::new(
            rect.left() + p[0] * rect.width(),
            rect.bottom() - p[1] * rect.height(),
        )
    };
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    for x in &plot.marks {
        let x = rect.left() + x * rect.width();
        painter.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(1.0_f32, Color32::from_rgb(240, 200, 80)),
        );
    }
    for pair in plot.points.windows(2) {
        painter.line_segment([at(&pair[0]), at(&pair[1])], Stroke::new(1.5_f32, colour));
    }
    let text = |pos: Pos2, anchor: Align2, value: f32| {
        painter.text(
            pos,
            anchor,
            format!("{value:.4}"),
            FontId::monospace(10.0),
            ui.visuals().weak_text_color(),
        );
    };
    text(rect.left_top(), Align2::LEFT_TOP, plot.max);
    text(rect.left_bottom(), Align2::LEFT_BOTTOM, plot.min);
    // The steps the two ends stand for, so a peak can be read off by where it falls.
    painter.text(
        rect.right_bottom(),
        Align2::RIGHT_BOTTOM,
        format!("{} … {}", plot.steps[0], plot.steps[1]),
        FontId::monospace(10.0),
        ui.visuals().weak_text_color(),
    );
    rect
}
