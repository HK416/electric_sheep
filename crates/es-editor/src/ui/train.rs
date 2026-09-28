//! ③ Train and ④ Evaluate (packet M12/Y12, `docs/design/editor-redesign.md` section 6.4): the
//! step panel, the centre and the summary of a run being started, watched and stopped.
//!
//! Drawing and wiring only. Which run, each step's state, the light, the stage cards, which
//! buttons are offered and what they write to disk are [`crate::model::watch`]'s, under test;
//! this turns a [`View`] into widgets and a click into one call on the watch.

use std::time::Instant;

use eframe::egui;
use egui::{Color32, RichText};
use es_eval::run_dir::Rgb8Image;
use es_ir::evaluation::MetricSpec;

use crate::app::EditorApp;
use crate::model::health::Light;
use crate::model::i18n::{self, Lang};
use crate::model::labels;
use crate::model::layout::{self, Pane};
use crate::model::project::StartSettings;
use crate::model::telemetry_view::{replay, TelemetryModel};
use crate::model::train_view::Series;
use crate::model::watch::{self, CardState, Centre, View, DEMONSTRATIONS, LENGTHS};
use crate::model::workflow::Phase;
use crate::ui::advanced::{paint_curve, rgb_texture};

const LOSS: Color32 = Color32::from_rgb(120, 200, 255);

/// Once a frame, after the child and the telemetry were polled and before anything is drawn:
/// the watch's tick, and what it asks of the window.
pub(crate) fn tick(app: &mut EditorApp, ctx: &egui::Context) {
    let Some(open) = app.project.as_mut() else {
        return;
    };
    // ⑤'s "Run again" hands ③ the finished run's settings (packets M12/Y13, Y12).
    if let Some(settings) = app.results.run_again.take() {
        open.watch.settings = settings;
    }
    let (tick, phases) =
        open.watch
            .tick(&mut app.launch, &app.telemetry, open.phase, Instant::now());
    open.phases = phases;
    if let Some(phase) = tick.done {
        open.phase = phase;
    }
    // What was heard so far belongs to the run before this one.
    if tick.started || tick.attached.is_some() {
        app.telemetry = TelemetryModel::default();
        app.source = tick.attached.unwrap_or_else(|| replay(Vec::new()));
    }
    if tick.started {
        app.status = app.launch.status_line();
    }
    if tick.done.is_some() {
        ctx.send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(
            egui::UserAttentionType::Informational,
        ));
        app.status = app.t("watch.done").to_owned();
    }
}

/// ③'s and ④'s step panel, centre and summary; `false` for every other pane and step.
pub(crate) fn draw(app: &mut EditorApp, ui: &mut egui::Ui, pane: Pane) -> bool {
    if !matches!(pane, Pane::StepPanel | Pane::Viewport | Pane::Summary) {
        return false;
    }
    let Some(open) = app.project.as_ref() else {
        return false;
    };
    let phase = open.phase;
    if !matches!(phase, Phase::Train | Phase::Evaluate) {
        return false;
    }
    let view = open.watch.view(
        phase,
        &app.launch,
        &app.telemetry,
        &open.phases,
        Instant::now(),
    );
    match pane {
        Pane::StepPanel => {
            egui::ScrollArea::vertical()
                .id_salt("watch-panel")
                .show(ui, |ui| panel(app, ui, phase, &view));
        }
        Pane::Viewport => centre(app, ui, &view),
        _ => summary(app, ui, &view),
    }
    true
}

/// A click in the step panel.
pub(crate) enum Action {
    Start,
    Resume(String),
    Stop,
    EvaluateNow,
}

pub(crate) fn perform(app: &mut EditorApp, action: Action) {
    let Some(open) = app.project.as_mut() else {
        return;
    };
    let pid = app.launch.pid();
    let result = match action {
        Action::Start => open.watch.start(&open.project, pid, &open.phases),
        Action::Resume(from) => open.watch.resume(&from, pid, &open.phases),
        Action::Stop => {
            app.launch.kill();
            Ok(true)
        }
        Action::EvaluateNow => open
            .watch
            .evaluate_now(&mut app.launch, app.telemetry.train.checkpoints()),
    };
    if let Err(e) = result {
        app.status = e.to_string();
    }
}

fn panel(app: &mut EditorApp, ui: &mut egui::Ui, phase: Phase, view: &View) {
    let lang = app.settings.lang;
    let Some(open) = app.project.as_mut() else {
        return;
    };
    ui.heading(layout::step_text(lang, phase, &view.state));
    ui.label(i18n::t(lang, view.state.key()));
    if let Some(run) = &open.watch.run {
        ui.weak(i18n::fill(
            lang,
            "watch.run",
            &[&format!("{:03}", run.number)],
        ));
    }
    ui.separator();
    for (stage, card) in &view.cards {
        ui.horizontal(|ui| {
            ui.strong(labels::stage_label(lang, stage));
            ui.label(i18n::t(lang, card.key()));
            if let CardState::Done(Some(seconds)) = card {
                ui.weak(i18n::fill(
                    lang,
                    "watch.took",
                    &[&format!("{seconds:.0} s")],
                ));
            }
        });
        if *card == CardState::Running {
            if let Some(fraction) = view.fraction {
                ui.add(egui::ProgressBar::new(fraction).show_percentage());
            }
            if let Some(eta) = view.eta {
                let left = format!("{:.0}s", eta.as_secs_f64());
                ui.weak(i18n::fill(lang, "live.eta", &[&left]));
            }
        }
    }
    ui.separator();
    if view.checking {
        ui.label(i18n::t(lang, "watch.checking"));
    }
    if view.interrupted {
        ui.label(i18n::t(lang, "watch.interrupted"));
    }
    if let Some(key) = view.cannot_start {
        ui.label(i18n::fill(lang, key, &[&open.project.file.template]));
    }
    let mut action = None;
    if let Some(key) = view.start {
        settings(ui, lang, &mut open.watch.settings);
        if ui
            .button(i18n::t(lang, key))
            .on_hover_text(i18n::t(lang, "watch.start.hint"))
            .clicked()
        {
            action = Some(Action::Start);
        }
    }
    if let Some(from) = &view.resume {
        let text = i18n::fill(lang, "watch.resume", &[labels::stage_label(lang, from)]);
        if ui
            .button(text)
            .on_hover_text(i18n::t(lang, "watch.resume.hint"))
            .clicked()
        {
            action = Some(Action::Resume(from.clone()));
        }
    }
    ui.horizontal(|ui| {
        if view.stop && ui.button(i18n::t(lang, "watch.stop")).clicked() {
            action = Some(Action::Stop);
        }
        if view.evaluate_now
            && ui
                .button(i18n::t(lang, "watch.evaluate_now"))
                .on_hover_text(i18n::t(lang, "watch.evaluate_now.hint"))
                .clicked()
        {
            action = Some(Action::EvaluateNow);
        }
    });
    if let Some(action) = action {
        perform(app, action);
    }
}

/// ③'s two settings.
fn settings(ui: &mut egui::Ui, lang: Lang, settings: &mut StartSettings) {
    ui.label(RichText::new(i18n::t(lang, "watch.settings")).strong());
    ui.horizontal(|ui| {
        ui.label(i18n::t(lang, "watch.demonstrations"))
            .on_hover_text(i18n::t(lang, "watch.demonstrations.hint"));
        ui.add(egui::DragValue::new(&mut settings.demonstrations).range(DEMONSTRATIONS));
    });
    ui.horizontal(|ui| {
        ui.label(i18n::t(lang, "watch.length"))
            .on_hover_text(i18n::t(lang, "watch.length.hint"));
        for length in LENGTHS {
            let word = i18n::t(lang, watch::length_key(length));
            ui.selectable_value(&mut settings.length, length, word);
        }
    });
}

/// The centre: ④'s tiles, then what the stage in progress has to show.
fn centre(app: &mut EditorApp, ui: &mut egui::Ui, view: &View) {
    let lang = app.settings.lang;
    if let Some(tiles) = &view.tiles {
        ui.label(RichText::new(i18n::t(lang, "watch.tiles")).strong());
        if tiles.is_empty() {
            ui.weak(i18n::t(lang, "watch.tiles.empty"));
        }
        ui.horizontal_wrapped(|ui| {
            for tile in tiles {
                let light = if tile.success {
                    Light::Green
                } else {
                    Light::Red
                };
                let [r, g, b] = watch::light_colour(light);
                ui.label(
                    RichText::new(format!(" {} ", tile.cell))
                        .color(Color32::BLACK)
                        .background_color(Color32::from_rgb(r, g, b)),
                );
            }
        });
        ui.separator();
    }
    let live = &app.telemetry.live;
    let picture = (live.images(), live.image().cloned());
    match view.centre {
        Centre::Idle => {
            ui.weak(i18n::t(lang, "watch.idle"));
        }
        Centre::Demonstrations {
            made,
            of,
            succeeded,
        } => {
            let counts = [made.to_string(), of.to_string(), succeeded.to_string()];
            let counts: Vec<&str> = counts.iter().map(String::as_str).collect();
            ui.heading(i18n::fill(lang, "watch.demos", &counts));
            ui.label(i18n::t(lang, "watch.picture"));
            image(app, ui, "watch-picture", picture);
        }
        Centre::Learning => {
            let train = &app.telemetry.train;
            let plot = train.plot(Series::Loss, train.log_scale);
            let sample = (train.samples(), train.sample().cloned());
            ui.label(i18n::t(lang, "live.loss"));
            match plot {
                Some(plot) => {
                    let height = (ui.available_height() * 0.5).max(120.0);
                    paint_curve(ui, &plot, LOSS, height);
                }
                None => {
                    ui.weak(i18n::t(lang, "live.training.empty"));
                }
            }
            ui.label(i18n::t(lang, "live.sample"))
                .on_hover_text(i18n::t(lang, "live.sample.hint"));
            image(app, ui, "watch-sample", sample);
        }
        Centre::Picture => {
            ui.label(i18n::t(lang, "watch.picture"));
            image(app, ui, "watch-picture", picture);
        }
    }
}

/// One picture, fitted to what is left of the pane. One texture per slot: the newest replaces
/// the last, so a long run does not keep every frame it ever sent.
fn image(
    app: &mut EditorApp,
    ui: &mut egui::Ui,
    slot: &str,
    (seq, image): (u64, Option<Rgb8Image>),
) {
    let Some(image) = image else {
        ui.weak(app.t("watch.no_picture"));
        return;
    };
    let key = format!("{slot}#{seq}");
    let prefix = format!("{slot}#");
    app.run_frames
        .retain(|k, _| !k.starts_with(&prefix) || *k == key);
    let ctx = ui.ctx().clone();
    let texture = app
        .run_frames
        .entry(key.clone())
        .or_insert_with(|| rgb_texture(&ctx, &key, &image));
    let size = texture.size_vec2();
    let room = egui::vec2(ui.available_width(), ui.available_height().max(96.0));
    let scale = (room.x / size.x).min(room.y / size.y).max(0.1);
    ui.image(egui::load::SizedTexture::new(texture.id(), size * scale));
}

/// The summary, right: the light with its name and advice, the curve, the rate, and GPU memory
/// when a metric row carries it.
fn summary(app: &EditorApp, ui: &mut egui::Ui, view: &View) {
    let lang = app.settings.lang;
    let Some(open) = app.project.as_ref() else {
        return;
    };
    ui.heading(app.fill("shell.project", &[&open.project.file.name]));
    ui.weak(open.project.root.display().to_string());
    ui.separator();
    match &view.signal {
        Some(signal) => {
            let [r, g, b] = watch::light_colour(signal.light);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("\u{25cf}")
                        .size(24.0)
                        .color(Color32::from_rgb(r, g, b)),
                );
                ui.heading(i18n::t(lang, signal.name));
            });
            if let Some(stage) = &signal.stage {
                ui.label(i18n::fill(
                    lang,
                    "watch.during",
                    &[labels::stage_label(lang, stage)],
                ));
            }
            ui.label(i18n::t(lang, signal.advice));
        }
        None => {
            ui.label(i18n::t(lang, view.state.key()));
        }
    }
    ui.separator();
    let train = &app.telemetry.train;
    if let Some(plot) = train.plot(Series::Loss, train.log_scale) {
        ui.label(i18n::t(lang, "live.loss"));
        paint_curve(ui, &plot, LOSS, 90.0);
    }
    if let Some(step) = train.step() {
        ui.label(match train.total() {
            Some(total) => i18n::fill(lang, "live.step", &[&step.to_string(), &total.to_string()]),
            None => i18n::fill(lang, "live.step_only", &[&step.to_string()]),
        });
    }
    if let Some(rate) = train.throughput() {
        ui.label(i18n::fill(
            lang,
            "live.throughput",
            &[&format!("{rate:.0}")],
        ));
    }
    if let Some(peak) = app.telemetry.metrics.and_then(|m| m.gpu_memory_peak) {
        let name = labels::metric_label(lang, MetricSpec::GpuMemoryPeak);
        ui.label(format!("{name}: {peak:.0}"));
    }
}
