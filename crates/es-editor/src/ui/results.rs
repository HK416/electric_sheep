//! ⑤ Results (packet M12/Y13, `docs/design/editor-redesign.md` section 6.5): the run list, the
//! verdict card and why attempts failed on the left; the player and every attempt as a tile in
//! the centre; success per situation, and folded the report's numbers and the run's hashes, on
//! the right.
//!
//! Drawing only. Which run is shown and what it is compared with, what each line says, which
//! views an attempt can be played in, which attempt opens first and which policy file Export
//! copies are [`crate::model::results`]'s, under test.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::SystemTime;

use eframe::egui;
use egui::load::SizedTexture;
use egui::{Color32, RichText, Vec2};

use crate::app::EditorApp;
use crate::model::dialogs;
use crate::model::i18n::{fill, t, Lang};
use crate::model::labels::{self, Browse};
use crate::model::layout::{self, Pane};
use crate::model::project::RunFolder;
use crate::model::results::{self, RunResults, TileFilter};
use crate::model::template;
use crate::model::workflow::{Phase, PhaseState};
use crate::ui::advanced::{metric_text, rgb_texture, short_hash};
use crate::ui::player::{play, Player};

const GREEN: Color32 = Color32::from_rgb(120, 200, 120);
const RED: Color32 = Color32::from_rgb(230, 120, 110);
/// A tile's thumbnail width, in points.
const TILE: f32 = 96.0;

/// Where a run was read from, and `report.json`'s time then: evaluating again into the same
/// folder is read again.
type Stamp = (PathBuf, Option<SystemTime>);

/// ⑤ between frames. It belongs to one project; opening another starts afresh.
#[derive(Default)]
pub struct State {
    project: Option<PathBuf>,
    /// The egui pass disk was last looked at in: three panes draw a frame, disk is read once.
    pass: Option<u64>,
    runs: Vec<RunFolder>,
    /// The run picked from the list; `None` follows the newest.
    chosen: Option<u32>,
    shown: Option<(Stamp, Result<RunResults, String>)>,
    filter: TileFilter,
    player: Option<Player>,
    /// Each tile's last picture, decoded once; `None` when it has none.
    thumbs: BTreeMap<String, Option<egui::TextureHandle>>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("project", &self.project)
            .field("chosen", &self.chosen)
            .finish_non_exhaustive()
    }
}

/// Draws `pane` when ⑤ is the open step and the pane is one of its three; `false` leaves it
/// to the shell.
pub fn draw(app: &mut EditorApp, ui: &mut egui::Ui, pane: Pane) -> bool {
    let ours = matches!(pane, Pane::StepPanel | Pane::Viewport | Pane::Summary);
    if !ours
        || app
            .project
            .as_ref()
            .is_none_or(|p| p.phase != Phase::Results)
    {
        return false;
    }
    refresh(app, ui.ctx().cumulative_pass_nr());
    match pane {
        Pane::StepPanel => step_panel(app, ui),
        Pane::Viewport => viewport(app, ui),
        _ => summary(app, ui),
    }
    true
}

/// Lists the project's finished runs and reads the one shown, when it is not what was read.
fn refresh(app: &mut EditorApp, pass: u64) {
    let Some(open) = &app.project else { return };
    let s = &mut app.results;
    if s.pass == Some(pass) {
        return;
    }
    if s.project.as_ref() != Some(&open.project.root) {
        *s = State {
            project: Some(open.project.root.clone()),
            ..State::default()
        };
    }
    s.pass = Some(pass);
    s.runs = results::finished_runs(&open.project);
    let Some(run) = results::shown(&s.runs, s.chosen) else {
        s.shown = None;
        s.player = None;
        return;
    };
    let modified = std::fs::metadata(run.report_path())
        .and_then(|m| m.modified())
        .ok();
    let stamp = (run.path.clone(), modified);
    if s.shown.as_ref().is_some_and(|(k, _)| *k == stamp) {
        return;
    }
    let read = RunResults::read(&open.project, run, template::templates_root().as_deref());
    s.thumbs.clear();
    s.player = read.as_ref().ok().and_then(|r| {
        let cell = results::first_to_play(r.rows.as_deref(), r.dir.cells())?;
        Some(Player::open(&r.dir, r.scene.as_deref(), cell))
    });
    s.shown = Some((stamp, read));
}

/// Left: the run list, the verdict, the acceptance lines, why attempts failed, the buttons.
fn step_panel(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    ui.heading(layout::step_text(lang, Phase::Results, &PhaseState::Done));
    let s = &mut app.results;
    let current = results::shown(&s.runs, s.chosen).map(|r| r.number);
    let mut picked = None;
    ui.horizontal_wrapped(|ui| {
        ui.label(t(lang, "results.run"));
        for run in &s.runs {
            let label = format!("{:03}", run.number);
            if ui
                .selectable_label(current == Some(run.number), label)
                .clicked()
            {
                picked = Some(run.number);
            }
        }
    });
    if picked.is_some() {
        s.chosen = picked;
        ui.ctx().request_repaint();
    }
    let shown = match &s.shown {
        None => {
            ui.label(t(lang, "results.none"));
            return;
        }
        Some((_, Err(e))) => {
            ui.colored_label(RED, e);
            return;
        }
        Some((_, Ok(shown))) => shown,
    };
    let mut run_again = None;
    let mut again = None;
    let mut status = None;
    verdict(lang, ui, shown);
    ui.separator();
    if let Some(rows) = shown.rows.as_deref() {
        ui.strong(t(lang, "results.why"));
        let causes = results::causes(rows, &shown.outcomes);
        if causes.is_empty() {
            ui.label(t(lang, "results.no_failures"));
        }
        for (cause, n) in causes {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(shown.cause_label(lang, cause)).strong());
                ui.weak(fill(lang, "results.times", &[&n.to_string()]));
            });
            ui.weak(t(lang, labels::cause_advice_key(cause)));
        }
        ui.separator();
    }
    let reason = if dialogs::AVAILABLE {
        "results.no_export"
    } else {
        "open.no_dialog.hint"
    };
    let export = ui
        .add_enabled(
            dialogs::AVAILABLE && shown.export.is_some(),
            egui::Button::new(t(lang, "results.export")),
        )
        .on_hover_text(t(lang, "results.export.hint"))
        .on_disabled_hover_text(t(lang, reason));
    if let (true, Some(bundle), Some(open)) = (export.clicked(), &shown.export, &app.project) {
        let name = results::export_name(&open.project, &shown.run, bundle);
        if let Some(dest) = dialogs::save_file(&name, Browse::Policy.filter()) {
            // ponytail: copied on the UI thread; a bundle is tens of MB, well under a
            // second. A background copy when bundles grow past that.
            status = Some(match std::fs::copy(bundle, &dest) {
                Ok(_) => fill(lang, "results.exported", &[&dest.display().to_string()]),
                Err(e) => format!("{}: {e}", dest.display()),
            });
        }
    }
    if ui
        .button(t(lang, "results.run_again"))
        .on_hover_text(t(lang, "results.run_again.hint"))
        .clicked()
    {
        run_again = Some(shown.settings);
    }
    // Packet M13/Z4b: the same settings and the plan, which ③ shows and starts only on
    // Start (review of plan Z, R1).
    let reason = shown.again.as_ref().err().copied().unwrap_or_default();
    if ui
        .add_enabled(
            shown.again.is_ok(),
            egui::Button::new(t(lang, "results.again")),
        )
        .on_disabled_hover_text(t(lang, reason))
        .clicked()
    {
        again = shown.again.clone().ok();
        run_again = Some(shown.settings);
    }
    ui.weak(t(lang, "results.again.about"));
    if let Some(status) = status {
        app.status = status;
    }
    if let Some(settings) = run_again {
        if let Some(open) = app.project.as_mut() {
            open.watch.prepare(settings, again);
            open.phase = Phase::Train;
        }
    }
}

/// The card: passed or not, "x of n", the change from the previous run, each acceptance line.
fn verdict(lang: Lang, ui: &mut egui::Ui, shown: &RunResults) {
    let card = results::card(&shown.dir.report, shown.rows.as_deref());
    let colour = if card.passed { GREEN } else { RED };
    ui.label(
        RichText::new(t(lang, card.verdict_key()))
            .heading()
            .color(colour),
    );
    ui.label(fill(
        lang,
        "results.headline",
        &[&card.successes.to_string(), &card.episodes.to_string()],
    ));
    let previous = shown.previous.as_ref().map_or(0, |(n, _)| *n);
    if let Some(change) = results::change_text(lang, &shown.comparison(), previous) {
        ui.label(change);
    }
    ui.add_space(6.0);
    ui.strong(t(lang, "results.to_pass"))
        .on_hover_text(t(lang, "results.acceptance.hint"));
    for line in &shown.dir.report.acceptance {
        let line = results::acceptance_line(lang, line, shown.ir.as_ref());
        let (mark, colour) = match line.passed {
            Some(true) => ("\u{2714}", GREEN),
            Some(false) => ("\u{2716}", RED),
            None => ("\u{2013}", Color32::GRAY),
        };
        ui.colored_label(colour, format!("{mark} {}", line.text))
            .on_hover_text(line.raw);
    }
}

/// Centre: the player, its timeline, and the tiles.
fn viewport(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    let dt = f64::from(ui.input(|i| i.stable_dt));
    let s = &mut app.results;
    let Some((_, Ok(shown))) = &s.shown else {
        ui.label(t(lang, "results.none"));
        return;
    };
    let mut pick = None;
    match s.player.as_mut() {
        Some(player) => play(lang, ui, dt, &shown.dir, player, &mut pick),
        None => {
            ui.weak(t(lang, "results.nothing_recorded"));
        }
    }
    ui.separator();
    let playing = s.player.as_ref().map(|p| p.cell.as_str());
    tiles(
        lang,
        ui,
        shown,
        &mut s.filter,
        &mut s.thumbs,
        playing,
        &mut pick,
    );
    if let Some(cell) = pick {
        s.player = Some(Player::open(&shown.dir, shown.scene.as_deref(), cell));
    }
}

/// Every attempt as a tile, its last picture as the thumbnail; an old run says why there are
/// none.
fn tiles(
    lang: Lang,
    ui: &mut egui::Ui,
    shown: &RunResults,
    filter: &mut TileFilter,
    thumbs: &mut BTreeMap<String, Option<egui::TextureHandle>>,
    playing: Option<&str>,
    pick: &mut Option<String>,
) {
    ui.horizontal(|ui| {
        ui.strong(t(lang, "results.tiles"));
        for f in TileFilter::ALL {
            ui.selectable_value(filter, f, t(lang, f.key()));
        }
    });
    let Some(rows) = shown.rows.as_deref() else {
        ui.weak(
            shown
                .rows_error
                .as_deref()
                .unwrap_or(t(lang, "results.old_run")),
        );
        return;
    };
    let ctx = ui.ctx().clone();
    egui::ScrollArea::vertical()
        .id_salt("results-tiles")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for tile in results::tiles(rows, *filter, &shown.outcomes) {
                    let frames = shown
                        .dir
                        .cells()
                        .iter()
                        .find(|c| c.name == tile.cell)
                        .map_or(0, |c| c.frames);
                    let thumb = thumbs.entry(tile.cell.clone()).or_insert_with(|| {
                        let last = results::frame_at(usize::MAX, frames)?;
                        let img = shown.dir.frame(&tile.cell, last)?;
                        Some(rgb_texture(&ctx, &tile.cell, &img))
                    });
                    ui.vertical(|ui| {
                        ui.set_width(TILE);
                        let mark = if tile.success { "\u{2714}" } else { "\u{2716}" };
                        let button = match thumb {
                            Some(texture) => {
                                let px = texture.size_vec2();
                                egui::Button::image(SizedTexture::new(
                                    texture.id(),
                                    px * (TILE / px.x),
                                ))
                            }
                            None => egui::Button::new(mark).min_size(Vec2::splat(TILE)),
                        };
                        let hover = results::suite_label(lang, &tile.suite, shown.ir.as_ref());
                        if ui
                            .add(button.selected(playing == Some(tile.cell.as_str())))
                            .on_hover_text(hover)
                            .clicked()
                        {
                            *pick = Some(tile.cell.clone());
                        }
                        let (colour, word) = match (tile.success, tile.cause) {
                            (true, _) => (GREEN, t(lang, "results.tile.success").to_owned()),
                            (false, Some(cause)) => (RED, shown.cause_label(lang, cause)),
                            (false, None) => (RED, t(lang, "results.tile.failure").to_owned()),
                        };
                        // One line each, the whole of it on hover: a tile is a thumbnail wide.
                        let name = RichText::new(format!("{mark} {}", tile.cell)).color(colour);
                        ui.add(egui::Label::new(name).truncate());
                        ui.add(egui::Label::new(RichText::new(word).small()).truncate());
                    });
                }
            });
        });
}

/// Right: success per situation; folded, the report's numbers and the run's settings and
/// hashes.
fn summary(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    // The left pane says why when there is nothing to show.
    let Some((_, Ok(shown))) = &app.results.shown else {
        return;
    };
    let report = &shown.dir.report;
    let ir = shown.ir.as_ref();
    // Sideways: the numbers table is wider than a narrow side pane. The dock scrolls it up and
    // down (`Pane::scrolls`).
    egui::ScrollArea::horizontal()
        .id_salt("results-right")
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.heading(t(lang, "results.situations"));
            for s in results::situations(report, shown.rows.as_deref(), ir) {
                ui.label(results::suite_label(lang, &s.suite, ir))
                    .on_hover_text(&s.suite);
                if s.episodes == 0 {
                    ui.weak(t(lang, "results.not_measured"));
                    continue;
                }
                let fraction = s.successes as f32 / s.episodes as f32;
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .text(format!("{} / {}", s.successes, s.episodes)),
                );
            }
            ui.separator();
            egui::CollapsingHeader::new(t(lang, "results.details"))
                .id_salt("results-details")
                .show(ui, |ui| {
                    egui::Grid::new("results-details-grid")
                        .striped(true)
                        .show(ui, |ui| {
                            for cell in &report.cells {
                                ui.label(results::suite_label(lang, &cell.suite, ir))
                                    .on_hover_text(&cell.suite);
                                ui.label(labels::metric_label(lang, cell.metric))
                                    .on_hover_text(cell.metric.name());
                                ui.label(metric_text(lang, &cell.value));
                                ui.end_row();
                            }
                        });
                });
            egui::CollapsingHeader::new(t(lang, "results.settings"))
                .id_salt("results-settings")
                .show(ui, |ui| {
                    let none = || t(lang, "value.none").to_owned();
                    let cycle = shown.cycle.as_ref();
                    let demonstrations = cycle
                        .and_then(|c| c.collect.as_ref())
                        .map_or_else(none, |c| c.episodes.to_string());
                    let steps = cycle
                        .and_then(|c| c.train.run.as_ref())
                        .map_or_else(none, |r| r.steps.to_string());
                    egui::Grid::new("results-settings-grid").show(ui, |ui| {
                        ui.label(t(lang, "results.demonstrations"));
                        ui.label(demonstrations);
                        ui.end_row();
                        ui.label(t(lang, "results.steps"));
                        ui.label(steps);
                        ui.end_row();
                        for (key, hint, hash) in [
                            (
                                "results.evaluation_hash",
                                "results.evaluation_hash.hint",
                                &report.evaluation_hash,
                            ),
                            (
                                "results.execution_hash",
                                "results.execution_hash.hint",
                                &report.execution_hash,
                            ),
                        ] {
                            ui.label(t(lang, key)).on_hover_text(t(lang, hint));
                            ui.monospace(short_hash(Ok(hash)))
                                .on_hover_text(es_compile::bundle::hex(hash));
                            ui.end_row();
                        }
                        ui.label(t(lang, "results.folder"));
                        ui.label(shown.run.path.display().to_string());
                        ui.end_row();
                    });
                });
        });
}
