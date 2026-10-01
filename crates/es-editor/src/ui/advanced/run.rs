//! The Results pane (packet M7/E1): the stage strip, the cell table, the acceptance verdict, and
//! for the selected cell its Safety Plane timeline and filmstrip. The Launch section above the
//! table is `launch`'s and the Replay panel under it `replay`'s.

use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Vec2};
use es_eval::run_dir::{Bucket, Rgb8Image};

use super::rgb_texture;
use crate::app::EditorApp;
use crate::model::i18n::{self, Lang};
use crate::model::labels;
use crate::model::live_run::cell_key;
use crate::model::replay_view;
use crate::model::run_view;

impl EditorApp {
    /// The Run tab (packet M7/E1): the cell table, the acceptance verdict, and for the
    /// selected cell its Safety Plane timeline and a filmstrip. Every number, every order and
    /// every decoded byte is [`RunDir`]'s; this turns them into widgets.
    pub(crate) fn run_tab(&mut self, ui: &mut egui::Ui) {
        // The Launch section is above the table and there whether or not anything is open:
        // starting a run is how the tab gets something to show (packet M7/E5).
        egui::TopBottomPanel::top("launch")
            .resizable(true)
            .default_height(400.0)
            .show_inside(ui, |ui| {
                // A pane shorter than the form scrolls it rather than cutting its flags off.
                egui::ScrollArea::vertical()
                    .id_salt("launch-form")
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.launch_panel(ui));
            });
        if self.run.is_none() && self.telemetry.live.is_empty() {
            ui.label(self.t("results.empty"));
            return;
        }
        // The replay of the selected cell shares the tab (packet M7/E2): the table picks the
        // episode, the panel plays it. The model decides how tall it is: its control rows
        // until a replay is loaded, a canvas afterwards. Two panel ids, because egui
        // remembers a panel's dragged height per id and the two states want their own.
        match replay_view::panel_height(self.replay.as_ref(), ui.available_height()) {
            Some(height) => {
                egui::TopBottomPanel::bottom("replay-canvas")
                    .resizable(true)
                    .default_height(height)
                    .show_inside(ui, |ui| self.replay_panel(ui));
            }
            None => {
                egui::TopBottomPanel::bottom("replay-controls")
                    .resizable(false)
                    .show_inside(ui, |ui| self.replay_panel(ui));
            }
        }
        egui::CentralPanel::default().show_inside(ui, |ui| self.run_table(ui));
    }

    fn run_table(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        // Captured before the destructure below, which borrows the rest of `self`.
        let lang = self.settings.lang;
        let Self {
            run,
            run_frames,
            telemetry,
            ..
        } = self;
        // **One table, two ends** (design note section 13). A finished run's rows come off
        // disk and a live one's off the wire, but both are `CellRow`s and a `Timeline`, so
        // everything below this match is the same code for either -- and none of it decides
        // anything: the rows, the headings and the strip are the models' (spec 28.10 rule 3).
        let live = &telemetry.live;
        // Which end the rows came from, which is the only thing the "joined late" mark means
        // anything for: a run read off disk is whole by the time it is opened.
        let is_live = run.is_none();
        let (columns, rows, selected, heading, acceptance) = match run.as_ref() {
            Some(run) => (
                run_view::columns(run),
                run.cells().to_vec(),
                run.selected_cell().map(|c| c.name.clone()),
                i18n::t(
                    lang,
                    if run.report.passed {
                        "results.passed"
                    } else {
                        "results.failed"
                    },
                )
                .to_owned(),
                run.acceptance().to_vec(),
            ),
            // A live run has no verdict yet: `report.json` is written after the last suite.
            None => (
                live.columns(),
                live.cells(),
                live.selected_cell(),
                live.status(),
                Vec::new(),
            ),
        };
        let timeline = selected.as_ref().map(|cell| match run.as_ref() {
            Some(run) => run.timeline(cell),
            None => live.timeline(cell),
        });
        let mut sort = None;
        let mut select = None;
        // Everything below is one scroll area, so a short window clips nothing: the table, the
        // acceptance rows, the strip and the filmstrip scroll together.
        let stages: Vec<(String, Option<f64>, bool)> = live
            .stages()
            .iter()
            .map(|s| (s.name.clone(), s.seconds, s.running()))
            .collect();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // A cycle publishes every stage on one socket (packet M7/E7), so the strip
                // above the table says which one these rows came from. A run that is one
                // command publishes no stage and the strip is not drawn at all.
                if !stages.is_empty() {
                    ui.heading(i18n::t(lang, "results.stages"));
                    ui.horizontal_wrapped(|ui| {
                        for (name, seconds, running) in &stages {
                            let plain = labels::stage_label(lang, name);
                            let word = if plain.is_empty() { name } else { plain };
                            let text = match seconds {
                                Some(s) => format!("{word}  {s:.1}s"),
                                None => word.to_owned(),
                            };
                            let colour = if *running {
                                Color32::from_rgb(240, 200, 80)
                            } else {
                                ui.visuals().text_color()
                            };
                            ui.colored_label(colour, text).on_hover_text(name);
                        }
                    });
                    ui.separator();
                }
                egui::Grid::new("run-cells").striped(true).show(ui, |ui| {
                    // A cycle's rows come from several stages and two of them name their
                    // episodes alike (packet M7/R12), so the stage is a column of its own. A
                    // run that is one command has no stages and no column. It does not sort:
                    // the rows are in the order the cycle ran them.
                    if !stages.is_empty() {
                        ui.label(i18n::t(lang, "column.stage"));
                    }
                    // The header is the metric's plain name and the hover is the raw one the
                    // report carries; a column this build has no word for keeps its raw name,
                    // which is still better than an empty heading (packet M7/E6).
                    for (i, name) in columns.iter().enumerate() {
                        let plain = labels::column_label(lang, name);
                        let heading = if plain.is_empty() { name } else { plain };
                        if ui.button(heading).on_hover_text(name).clicked() {
                            sort = Some(i);
                        }
                    }
                    ui.label(i18n::t(lang, "column.traj"));
                    ui.label(i18n::t(lang, "column.frames"));
                    ui.end_row();
                    for row in &rows {
                        // The row is the pair, not the name (packet M7/R12): selecting and
                        // the strip below both go by the key the model builds.
                        let key = cell_key(&row.stage, &row.name);
                        if !stages.is_empty() {
                            let plain = labels::stage_label(lang, &row.stage);
                            let word = if plain.is_empty() { &row.stage } else { plain };
                            ui.label(word).on_hover_text(&row.stage);
                        }
                        let is_selected = selected.as_deref() == Some(key.as_str());
                        // A cell whose beginning this viewer missed says so beside its name:
                        // what it shows is the part it heard, not the whole episode.
                        let name = if is_live && live.joined_late(&key) {
                            format!("{}  {}", row.name, i18n::t(lang, "results.joined_late"))
                        } else {
                            row.name.clone()
                        };
                        if ui.selectable_label(is_selected, name).clicked() {
                            select = Some(key);
                        }
                        ui.label(&row.suite);
                        ui.label(row.seed.map_or_else(|| "--".to_owned(), |s| s.to_string()));
                        for column in columns.iter().skip(3) {
                            ui.label(
                                row.metrics
                                    .get(column)
                                    .map_or_else(|| dash(lang), |v| metric_text(lang, v)),
                            );
                        }
                        ui.label(i18n::t(
                            lang,
                            if row.has_traj {
                                "value.yes"
                            } else {
                                "value.none"
                            },
                        ));
                        ui.label(row.frames.to_string());
                        ui.end_row();
                    }
                });

                ui.separator();
                ui.heading(&heading)
                    .on_hover_text(i18n::t(lang, "results.acceptance.hint"));
                for line in &acceptance {
                    let (text, colour) = acceptance_row(lang, line);
                    ui.colored_label(colour, text);
                }

                let (Some(cell), Some(timeline)) = (selected.as_ref(), timeline.as_ref()) else {
                    ui.separator();
                    ui.label(i18n::t(lang, "results.select_cell"));
                    return;
                };
                ui.separator();
                ui.heading(run_view::timeline_heading(timeline, cell));
                // One column per ~4 px of the strip; the model folds the frames into them.
                let n = (ui.available_width() / 4.0) as usize;
                paint_timeline(lang, ui, &timeline.buckets(n));
                for kind in timeline.kind_rows() {
                    ui.label(run_view::kind_label(&kind));
                }

                ui.separator();
                ui.heading(i18n::t(lang, "results.frames"));
                // A live run has no filmstrip on disk: what it has is the observation frame
                // the producer is publishing right now (stream 4), uploaded once per image
                // rather than once per repaint.
                let Some(run) = run.as_ref() else {
                    if let Some(image) = live.image() {
                        let key = format!("live#{}", live.images());
                        if !run_frames.contains_key(&key) {
                            run_frames.clear();
                            run_frames.insert(key.clone(), rgb_texture(&ctx, &key, image));
                        }
                        if let Some(texture) = run_frames.get(&key) {
                            let scale = (160.0 / texture.size_vec2().x).max(1.0);
                            ui.image(egui::load::SizedTexture::new(
                                texture.id(),
                                texture.size_vec2() * scale,
                            ));
                        }
                    } else {
                        ui.label(i18n::t(lang, "results.no_image"));
                    }
                    return;
                };
                // Eight thumbnails are wider than a narrow window; scroll them sideways rather
                // than cutting the last ones off.
                egui::ScrollArea::horizontal()
                    .id_salt("filmstrip")
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for index in run.filmstrip(cell, FILMSTRIP) {
                                let key = format!("{cell}#{index}");
                                let texture = run_frames.entry(key.clone()).or_insert_with(|| {
                                    let image = run.frame(cell, index).unwrap_or(Rgb8Image {
                                        width: 1,
                                        height: 1,
                                        data: vec![0, 0, 0],
                                    });
                                    rgb_texture(&ctx, &key, &image)
                                });
                                ui.vertical(|ui| {
                                    ui.label(format!("{index}"));
                                    let scale = (160.0 / texture.size_vec2().x).max(1.0);
                                    ui.image(egui::load::SizedTexture::new(
                                        texture.id(),
                                        texture.size_vec2() * scale,
                                    ));
                                });
                            }
                        });
                    });
            });
        // Sorting is a finished run's: a live table is in cell-name order and its rows are
        // still arriving. Selecting works on either.
        if let (Some(column), Some(run)) = (sort, run.as_mut()) {
            run.sort_by(column);
        }
        if let Some(name) = select {
            match run.as_mut() {
                Some(run) => run.select(&name),
                None => telemetry.live.select(&name),
            }
        }
    }
}

/// Frames the filmstrip shows, sampled evenly over the cell by [`RunDir::filmstrip`].
const FILMSTRIP: usize = 8;

fn dash(lang: Lang) -> String {
    i18n::t(lang, "value.none").to_owned()
}

/// A metric cell. A histogram has no single number and an unmeasured metric has none at all
/// (spec 10.3): neither is rendered as `0`.
pub(crate) fn metric_text(lang: Lang, value: &es_ir::evaluation::MetricValue) -> String {
    match value {
        es_ir::evaluation::MetricValue::Scalar(v) => format!("{v:.4}"),
        es_ir::evaluation::MetricValue::Histogram(h) => {
            i18n::fill(lang, "value.histogram", &[&h.len().to_string()])
        }
        // The reason is the runner's own sentence and is shown verbatim: inventing a
        // translation for it would be the editor guessing at what `es` meant.
        es_ir::evaluation::MetricValue::Unavailable { reason } => {
            format!("{} ({reason})", i18n::t(lang, "value.not_measured"))
        }
    }
}

fn acceptance_row(lang: Lang, line: &es_ir::evaluation::AcceptanceResult) -> (String, Color32) {
    match line {
        es_ir::evaluation::AcceptanceResult::Determined {
            criterion,
            observed,
            passed,
        } => (
            format!(
                "{} {} {} ({}, {}): {observed:.4} -- {}",
                labels::metric_label(lang, criterion.metric),
                criterion.comparator.name(),
                criterion.threshold,
                criterion.aggregation.name(),
                criterion.suite.as_deref().unwrap_or("*"),
                i18n::t(
                    lang,
                    if *passed {
                        "results.pass"
                    } else {
                        "results.fail"
                    }
                )
            ),
            if *passed {
                Color32::from_rgb(120, 200, 120)
            } else {
                Color32::from_rgb(230, 120, 110)
            },
        ),
        es_ir::evaluation::AcceptanceResult::Unavailable { metric, reason } => (
            format!(
                "{}: {} ({reason})",
                labels::metric_label(lang, *metric),
                i18n::t(lang, "results.not_measured")
            ),
            Color32::from_gray(160),
        ),
    }
}

/// One colour per `EventSource`, violations as a tick beneath (spec 23.3).
pub(crate) fn paint_timeline(lang: Lang, ui: &mut egui::Ui, buckets: &[Bucket]) {
    if buckets.is_empty() {
        ui.label(i18n::t(lang, "results.no_events"));
        return;
    }
    let (response, painter) =
        ui.allocate_painter(Vec2::new(ui.available_width(), 30.0), Sense::hover());
    let rect = response.rect;
    let w = (rect.width() / buckets.len() as f32).max(1.0);
    for (i, bucket) in buckets.iter().enumerate() {
        let x = rect.left() + i as f32 * rect.width() / buckets.len() as f32;
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(w, 18.0)),
            0.0,
            source_colour(bucket.source),
        );
        if !bucket.counts.is_empty() {
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(x, rect.top() + 21.0), Vec2::new(w, 8.0)),
                0.0,
                Color32::from_rgb(240, 200, 80),
            );
        }
    }
}

fn source_colour(source: es_eval::runner::EventSource) -> Color32 {
    match source {
        es_eval::runner::EventSource::Policy => Color32::from_rgb(70, 130, 180),
        es_eval::runner::EventSource::Human => Color32::from_rgb(150, 150, 200),
        es_eval::runner::EventSource::Clamped => Color32::from_rgb(220, 170, 60),
        es_eval::runner::EventSource::Fallback => Color32::from_rgb(210, 90, 80),
    }
}
