//! ② Choose a teacher (packet M16/H7): for a template whose demonstrator is a trained policy,
//! the project's teacher runs and the chosen teacher on the left, the selected run's learning
//! curves and its checkpoints with their test scores in the centre, the template's words on the
//! right.
//!
//! Drawing and wiring only. Which runs and checkpoints there are, their scores, the argv of
//! every job, what choosing writes and every word are [`crate::model::teacher`]'s, under test.
//! The jobs run as children of their own, never the launch model ③ and ④ follow; a training
//! job's telemetry is attached as the Live pane's source, so its curves draw as it learns.

use std::path::PathBuf;

use eframe::egui;
use egui::{Color32, RichText};
use es_editor_scene::check::maps_onto;
use es_editor_scene::{BackendKind, Refusal};

use crate::app::EditorApp;
use crate::model::home::Mark as Colour;
use crate::model::i18n::{fill, t, Lang, Strings};
use crate::model::launch::{self, State as LaunchState};
use crate::model::layout::Pane;
use crate::model::project::RunFolder;
use crate::model::teacher::{self, Choice, Chose, Jobs, Mark};
use crate::model::telemetry_view::TelemetryModel;
use crate::model::template::{Method, Template};
use crate::model::train_view::{Series, TrainView};
use crate::model::workflow::Phase;
use crate::ui::advanced::{paint_curve, RL_COLOURS};

/// How many of `es`'s last lines a job that did not finish shows.
const LAST_LINES: usize = 6;

/// ② between frames, for one project. Its jobs outlive a switch to another project's step.
#[derive(Default)]
pub(crate) struct State {
    project: Option<PathBuf>,
    /// The template and the folder `es` runs in (an authored project's own, packet M17/G9);
    /// `None` for a project taught any other way, or whose template is not here.
    source: Option<(Template, PathBuf)>,
    /// Why an authored project's saved scene does not map onto `MuJoCo` Warp, which the teacher
    /// trains on (packet M17/G9).
    warp: Option<Refusal>,
    jobs: Jobs,
    runs: Vec<RunFolder>,
    selected: Option<u32>,
    marks: Vec<Mark>,
    chosen: Option<Choice>,
    /// The selected run's curves as its folder has them.
    curves: Option<Result<TrainView, String>>,
    note: Option<String>,
}

/// Once a frame: the jobs polled, disk re-read when one ended, and a training job's telemetry
/// attached.
pub(crate) fn tick(app: &mut EditorApp) {
    let s = &mut app.teacher;
    let Some((_, root)) = &s.source else { return };
    if s.jobs.tick(root) {
        reload(s);
    }
    if let Some(attached) = s.jobs.launch.take_attached() {
        match attached {
            Ok(source) => {
                app.telemetry = TelemetryModel::default();
                app.source = source;
            }
            Err(e) => s.note = Some(e),
        }
    }
}

/// ②'s three panes of a `method = "teacher"` project; `false` for anything else.
pub(crate) fn draw(app: &mut EditorApp, ui: &mut egui::Ui, pane: Pane) -> bool {
    let ours = matches!(pane, Pane::StepPanel | Pane::Viewport | Pane::Summary);
    if !ours || app.project.as_ref().is_none_or(|p| p.phase != Phase::Teach) {
        return false;
    }
    open(app);
    if app.teacher.source.is_none() {
        return missing(app, ui, pane);
    }
    match pane {
        Pane::StepPanel => panel(app, ui),
        Pane::Viewport => centre(app, ui),
        _ => summary(app, ui),
    }
    true
}

/// ② of an editable project whose own documents are missing (packet M17/G9): what is missing,
/// in words; the scene is ①'s to show. `false` for any other project.
fn missing(app: &EditorApp, ui: &mut egui::Ui, pane: Pane) -> bool {
    let Some(Err((key, arg))) = app.project.as_ref().map(|p| p.watch.source().clone()) else {
        return false;
    };
    if !key.starts_with("watch.task.") {
        return false;
    }
    if pane != Pane::StepPanel {
        return true;
    }
    let lang = app.settings.lang;
    ui.heading(t(lang, "teach.heading"));
    ui.colored_label(colour(Colour::Optional), fill(lang, key, &[&arg]));
    true
}

impl State {
    /// Read again on the next frame: what ② to ⑤ run has changed (packet M17/G9).
    pub(crate) fn forget(&mut self) {
        self.project = None;
    }
}

/// Reads the open project's teacher runs, once per project.
fn open(app: &mut EditorApp) {
    let Some(open) = &app.project else { return };
    let s = &mut app.teacher;
    if s.project.as_ref() == Some(&open.project.root) {
        return;
    }
    let source = (open.watch.source().as_ref().ok())
        .filter(|(t, _)| t.method == Method::Teacher)
        .cloned();
    let warp = (source.as_ref().filter(|(t, _)| t.generated))
        .and_then(|_| maps_onto(&open.project.root, BackendKind::MjWarp).err());
    *s = State {
        project: Some(open.project.root.clone()),
        source,
        warp,
        jobs: std::mem::take(&mut s.jobs),
        ..State::default()
    };
    reload(s);
}

/// The runs, the selected run's checkpoints and curves, and the chosen teacher, from disk.
fn reload(s: &mut State) {
    let Some(project) = s
        .project
        .as_ref()
        .and_then(|p| crate::model::project::Project::open(p).ok())
    else {
        return;
    };
    s.runs = project.teacher_runs();
    s.chosen = teacher::chosen(&project);
    let numbers: Vec<u32> = s.runs.iter().map(|r| r.number).collect();
    if s.selected.is_none_or(|n| !numbers.contains(&n)) {
        s.selected = s.chosen.map(|c| c.run).or(numbers.last().copied());
    }
    let run = selected(s).cloned();
    s.marks = run.as_ref().map(teacher::marks).unwrap_or_default();
    s.curves = run.map(|r| TrainView::open_dir(&r.path));
}

fn selected(s: &State) -> Option<&RunFolder> {
    s.runs.iter().find(|r| Some(r.number) == s.selected)
}

fn colour(mark: Colour) -> Color32 {
    let [r, g, b] = mark.colour();
    Color32::from_rgb(r, g, b)
}

/// A click.
enum Action {
    Train,
    Stop,
    Select(u32),
    Test(Vec<u32>),
    Use(u32),
}

fn perform(app: &mut EditorApp, action: Action) {
    let Some(open) = &app.project else { return };
    let s = &mut app.teacher;
    let Some((template, root)) = &s.source else {
        return;
    };
    let result = match action {
        Action::Train => {
            let addr = format!("127.0.0.1:{}", launch::free_local_port());
            let dir = open.project.next_teacher_dir();
            teacher::train(template, root, &open.project, &dir, &addr).map(|argv| {
                s.jobs.push(argv);
                s.selected = None;
            })
        }
        Action::Stop => {
            s.jobs.stop();
            Ok(())
        }
        Action::Select(n) => {
            s.selected = Some(n);
            Ok(())
        }
        Action::Test(steps) => {
            let run = selected(s).cloned();
            (run.into_iter().flat_map(|run| {
                (steps.iter()).map(move |&step| teacher::evaluate(template, root, &run, step))
            }))
            .collect::<Result<Vec<_>, _>>()
            .map(|all| all.into_iter().for_each(|argv| s.jobs.push(argv)))
        }
        Action::Use(step) => match selected(s).cloned() {
            None => Ok(()),
            Some(run) => teacher::choose(template, root, &open.project, &run, step).map(|chose| {
                if let Chose::Pack(argv) = chose {
                    s.jobs.push(argv);
                }
            }),
        },
    };
    s.note = result.err().map(|e| e.to_string());
    reload(s);
}

/// Left: what this step is, the chosen teacher, the runs, Train and Stop, and what runs now.
fn panel(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    let s = &app.teacher;
    ui.heading(t(lang, "teach.teacher.heading"));
    let generated = s.source.as_ref().is_some_and(|(t, _)| t.generated);
    ui.label(t(
        lang,
        if generated {
            "teach.teacher.about_generated"
        } else {
            "teach.teacher.about"
        },
    ));
    // Packet M17/G9: a scene the GPU simulator cannot take is said here, not by a failed run.
    if let Some(r) = &s.warp {
        let args: Vec<&str> = r.args.iter().map(String::as_str).collect();
        let why = fill(lang, r.key, &args);
        ui.colored_label(
            colour(Colour::Missing),
            fill(lang, "teach.teacher.no_warp", &[&why]),
        );
    }
    ui.add_space(6.0);
    let chosen = s.chosen.map(|c| {
        let score = (s.runs.iter().find(|r| r.number == c.run))
            .and_then(|r| teacher::score(&teacher::eval_dir(r, c.step)))
            .map_or_else(
                || t(lang, "teach.teacher.untested").to_owned(),
                teacher::Score::text,
            );
        fill(
            lang,
            "teach.teacher.chosen",
            &[&format!("{:03}", c.run), &c.step.to_string(), &score],
        )
    });
    match chosen {
        Some(line) => ui.label(RichText::new(line).strong().color(colour(Colour::Have))),
        None => {
            ui.label(RichText::new(t(lang, "teach.teacher.none")).color(colour(Colour::Optional)))
        }
    };
    ui.separator();
    ui.strong(t(lang, "teach.teacher.runs"));
    let mut action = None;
    ui.horizontal_wrapped(|ui| {
        if s.runs.is_empty() {
            ui.weak(t(lang, "teach.teacher.no_runs"));
        }
        for run in &s.runs {
            let label = format!("{:03}", run.number);
            if ui
                .selectable_label(s.selected == Some(run.number), label)
                .clicked()
            {
                action = Some(Action::Select(run.number));
            }
        }
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let train = ui
            .add_enabled(
                !s.jobs.busy() && s.warp.is_none(),
                egui::Button::new(t(lang, "teach.teacher.train")),
            )
            .on_hover_text(t(lang, "teach.teacher.train.hint"));
        if train.clicked() {
            action = Some(Action::Train);
        }
        if s.jobs.busy() && ui.button(t(lang, "teach.teacher.stop")).clicked() {
            action = Some(Action::Stop);
        }
    });
    if let Some(argv) = s.jobs.current() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(teacher::job_text(lang, argv));
        });
    }
    if s.jobs.queued() > 0 {
        ui.weak(fill(
            lang,
            "teach.teacher.queued",
            &[&s.jobs.queued().to_string()],
        ));
    }
    let launch = &s.jobs.launch;
    if let (false, LaunchState::Exited { code, .. }) = (s.jobs.busy(), launch.state()) {
        if *code != 0 && !launch.was_killed() {
            ui.colored_label(
                colour(Colour::Missing),
                fill(lang, "teach.teacher.ended", &[&launch.status_line()]),
            );
            let lines: Vec<&str> = launch.lines().collect();
            for line in &lines[lines.len().saturating_sub(LAST_LINES)..] {
                ui.monospace(*line);
            }
        }
    }
    if let Some(note) = &s.note {
        ui.colored_label(colour(Colour::Missing), note);
    }
    if let Some(action) = action {
        perform(app, action);
    }
}

/// Centre: the selected run's learning - live while it trains - and its checkpoints.
fn centre(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    let Some(run) = selected(&app.teacher).cloned() else {
        ui.weak(t(lang, "teach.teacher.no_runs"));
        return;
    };
    let mut action = None;
    egui::ScrollArea::vertical()
        .id_salt("teacher-centre")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // The checkpoints first: choosing one is what this step is for.
            let s = &app.teacher;
            ui.horizontal(|ui| {
                ui.strong(t(lang, "teach.teacher.checkpoints"));
                let untested: Vec<u32> = (s.marks.iter())
                    .filter(|m| m.score.is_none())
                    .map(|m| m.step)
                    .collect();
                if ui
                    .add_enabled(
                        !untested.is_empty(),
                        egui::Button::new(t(lang, "teach.teacher.test_all")),
                    )
                    .clicked()
                {
                    action = Some(Action::Test(untested));
                }
            });
            marks(lang, ui, s, &run, &mut action);
            ui.separator();
            ui.heading(fill(
                lang,
                "teach.teacher.curves",
                &[&format!("{:03}", run.number)],
            ));
            let live = s.jobs.training().is_some_and(|p| p == run.path);
            let view = if live {
                Some(&app.telemetry.train)
            } else {
                s.curves.as_ref().and_then(|c| c.as_ref().ok())
            };
            curves(lang, ui, view);
        });
    if let Some(action) = action {
        perform(app, action);
    }
}

/// A PPO run's curves, as the Live pane draws them.
fn curves(lang: Lang, ui: &mut egui::Ui, view: Option<&TrainView>) {
    let plots: Vec<_> = (Series::RL.iter())
        .filter_map(|&s| Some((s, view?.plot(s, false)?)))
        .collect();
    if plots.is_empty() {
        ui.weak(t(lang, "teach.teacher.no_curves"));
    }
    for (i, (series, plot)) in plots.iter().enumerate() {
        ui.label(t(lang, series.key()));
        paint_curve(ui, plot, RL_COLOURS[i % RL_COLOURS.len()], 60.0);
    }
}

/// One row per checkpoint: its step, its test score, Test and Use.
fn marks(lang: Lang, ui: &mut egui::Ui, s: &State, run: &RunFolder, action: &mut Option<Action>) {
    let in_use = |step: u32| {
        s.chosen
            == Some(Choice {
                run: run.number,
                step,
            })
    };
    egui::Grid::new("teacher-marks")
        .striped(true)
        .show(ui, |ui| {
            ui.strong(t(lang, "teach.teacher.step"));
            ui.strong(t(lang, "teach.teacher.success"));
            ui.end_row();
            for mark in &s.marks {
                ui.label(mark.step.to_string());
                match mark.score {
                    Some(score) => ui.label(score.text()),
                    None => ui.weak(t(lang, "teach.teacher.untested")),
                };
                let test = ui
                    .button(t(lang, "teach.teacher.test"))
                    .on_hover_text(t(lang, "teach.teacher.test.hint"));
                if test.clicked() {
                    *action = Some(Action::Test(vec![mark.step]));
                }
                if in_use(mark.step) {
                    ui.label(
                        RichText::new(t(lang, "teach.teacher.in_use"))
                            .strong()
                            .color(colour(Colour::Have)),
                    );
                } else if ui.button(t(lang, "teach.teacher.use")).clicked() {
                    *action = Some(Action::Use(mark.step));
                }
                ui.end_row();
            }
        });
}

/// Right: the template's own words, and how long things take.
fn summary(app: &EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    let Some((template, _)) = &app.teacher.source else {
        return;
    };
    let word = |key: &str| Strings::get(lang).t(key).to_owned();
    ui.heading(word(&template.name));
    ui.label(word(&template.summary));
    if let Some(notice) = &template.notice {
        let line = format!("\u{26a0} {}", word(notice));
        ui.label(RichText::new(line).color(colour(Colour::Optional)));
    }
    ui.separator();
    ui.label(t(lang, "teach.teacher.time"));
}
