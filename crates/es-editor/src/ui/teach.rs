//! ② Teach (packet M14/Q4, `docs/design/editor-redesign.md` sections 3 and 11): the project's
//! demonstration program as a list of blocks in plain words on the left, the selected block's
//! fields on the right, and in the centre "try it once" and the shared player on the try.
//!
//! Drawing and wiring only. The program, its checks, every word, which buttons do something, the
//! try's argv and what it came to are [`crate::model::teach`]'s, under test. A try runs as a child
//! of its own, never the launch model ③ and ④ follow: the run the step bar reads never sees it.
//! A project whose program cannot be opened keeps the read-only ② of packet M12/Y15, with why.

use std::path::{Path, PathBuf};

use eframe::egui;
use egui::{AtomExt, Color32, RichText};
use es_eval::run_dir::RunDir;

use crate::app::EditorApp;
use crate::model::home::Mark;
use crate::model::i18n::{fill, t, Lang};
use crate::model::launch::{LaunchModel, State as LaunchState};
use crate::model::layout::Pane;
use crate::model::results::{self, Tile};
use crate::model::teach::{
    grip_key, Action, Field, FieldName, Grip, Notes, Target, TargetKind, Teach, TeachError, OBJECT,
};
use crate::model::workflow::Phase;
use crate::ui::player::{play, Player};

/// How many of `es`'s last lines a try that did not finish shows.
const LAST_LINES: usize = 8;

/// ② between frames. It belongs to one project; a try's child outlives a switch to another, so
/// no second try starts while it runs.
#[derive(Default)]
pub(crate) struct State {
    project: Option<PathBuf>,
    /// `None` when the project's template or the checkout is missing: ①'s view of the scene
    /// says which.
    teach: Option<Result<Teach, TeachError>>,
    /// The repository root a try runs in, and the scene its motion is re-posed on.
    root: PathBuf,
    scene: Option<PathBuf>,
    /// The tries' own child.
    launch: LaunchModel,
    /// A try started here whose result has not been read yet.
    running: bool,
    shown: Option<Shown>,
    /// What the last Save, Start over or try refused with.
    note: Option<String>,
}

/// The try shown: the newest.
struct Shown {
    dir: PathBuf,
    /// Started from this window, so [`State::launch`]'s lines are its.
    ours: bool,
    attempt: Option<Tile>,
    player: Option<Result<(RunDir, Player), String>>,
}

impl Shown {
    fn new(dir: PathBuf, ours: bool, attempt: Option<Tile>, scene: Option<&Path>) -> Self {
        let player = attempt.as_ref().map(|tile| {
            RunDir::open(&dir)
                .map(|run| {
                    let player = Player::open(&run, scene, tile.cell.clone());
                    (run, player)
                })
                .map_err(|e| e.to_string())
        });
        Self {
            dir,
            ours,
            attempt,
            player,
        }
    }
}

/// Once a frame: the try's child polled, and its attempt read the frame it ends.
pub(crate) fn tick(app: &mut EditorApp) {
    let s = &mut app.teach;
    s.launch.poll();
    if !s.running || s.launch.pid().is_some() {
        return;
    }
    s.running = false;
    let (Some(Ok(teach)), Some(shown)) = (&s.teach, &mut s.shown) else {
        return;
    };
    let attempt = teach.attempt(&shown.dir);
    let dir = std::mem::take(&mut shown.dir);
    *shown = Shown::new(dir, true, attempt, s.scene.as_deref());
}

/// ②'s step panel, centre and inspector; `false` for every other pane and step, and for a
/// project whose template the checkout does not have.
pub(crate) fn draw(app: &mut EditorApp, ui: &mut egui::Ui, pane: Pane) -> bool {
    let ours = matches!(pane, Pane::StepPanel | Pane::Viewport | Pane::Summary);
    if !ours || app.project.as_ref().is_none_or(|p| p.phase != Phase::Teach) {
        return false;
    }
    open(app);
    let lang = app.settings.lang;
    let read_only = match &app.teach.teach {
        None => return false,
        Some(Err(why)) => Some(why.text(lang)),
        Some(Ok(_)) => None,
    };
    if let Some(why) = read_only {
        let drawn = crate::ui::scene::draw(app, ui, pane);
        if pane == Pane::StepPanel {
            ui.separator();
            ui.label(RichText::new(why).color(colour(Mark::Optional)));
        }
        return drawn;
    }
    match pane {
        Pane::StepPanel => panel(app, ui),
        Pane::Viewport => centre(app, ui),
        _ => inspector(app, ui),
    }
    true
}

impl State {
    /// Read again on the next frame: what ② to ⑤ run has changed (packet M17/G9).
    pub(crate) fn forget(&mut self) {
        self.project = None;
    }
}

/// Reads the open project's program, once per project, with the first block selected and its
/// newest try shown.
fn open(app: &mut EditorApp) {
    let Some(open) = &app.project else { return };
    let s = &mut app.teach;
    if s.project.as_ref() == Some(&open.project.root) {
        return;
    }
    // The documents ② to ⑤ run: an authored project's own, in its own folder (packet M17/G9).
    let source = open.watch.source().as_ref().ok();
    let root = source.map(|(_, root)| root.clone());
    let template = source.map(|(t, _)| t.clone());
    let teach = template
        .as_ref()
        .zip(root.as_ref())
        .map(|(template, root)| {
            let mut teach = Teach::open(&open.project, template, root)?;
            teach.select(Some(0));
            Ok(teach)
        });
    let scene = results::scene(None, template.as_ref(), root.as_deref());
    let shown = (teach.as_ref().and_then(|t| t.as_ref().ok()))
        .and_then(Teach::latest_try)
        .map(|(dir, attempt)| Shown::new(dir, false, attempt, scene.as_deref()));
    *s = State {
        project: Some(open.project.root.clone()),
        teach,
        root: root.unwrap_or_default(),
        scene,
        launch: std::mem::take(&mut s.launch),
        running: false,
        shown,
        note: None,
    };
}

fn colour(mark: Mark) -> Color32 {
    let [r, g, b] = mark.colour();
    Color32::from_rgb(r, g, b)
}

/// A block's (or the program's) refusal and warnings, one line each.
fn notes(ui: &mut egui::Ui, notes: &Notes) {
    if let Some(error) = &notes.error {
        let line = format!("{} {error}", Mark::Missing.glyph());
        ui.label(RichText::new(line).color(colour(Mark::Missing)));
    }
    for warning in &notes.warnings {
        let line = format!("\u{26a0} {warning}");
        ui.label(RichText::new(line).color(colour(Mark::Optional)));
    }
}

/// A block's mark in the list and what its hover says: the refusal before the warnings.
fn mark(notes: &Notes) -> Option<(RichText, String)> {
    if let Some(error) = &notes.error {
        let glyph = RichText::new(Mark::Missing.glyph()).color(colour(Mark::Missing));
        return Some((glyph, error.clone()));
    }
    (!notes.warnings.is_empty()).then(|| {
        let glyph = RichText::new("\u{26a0}").color(colour(Mark::Optional));
        (glyph, notes.warnings.join("\n"))
    })
}

/// One of ②'s buttons, enabled as the model says; whether it was pressed.
fn button(ui: &mut egui::Ui, lang: Lang, teach: &Teach, action: Action) -> bool {
    let mut response = ui.add_enabled(teach.can(action), egui::Button::new(t(lang, action.key())));
    if let Some(hint) = action.hint_key() {
        response = response.on_hover_text(t(lang, hint));
    }
    response.clicked()
}

/// Left: what is wrong with the whole program, the blocks, and the buttons that change them.
fn panel(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    let count = (app.project.as_ref())
        .map(|p| p.watch.settings.demonstrations.to_string())
        .unwrap_or_default();
    let s = &mut app.teach;
    let Some(Ok(teach)) = &mut s.teach else {
        return;
    };
    ui.heading(t(lang, "teach.heading"));
    notes(ui, &teach.top(lang));
    let mut action = None;
    ui.horizontal_wrapped(|ui| {
        for a in [
            Action::AddMove,
            Action::AddGrip,
            Action::Delete,
            Action::Up,
            Action::Down,
        ] {
            if button(ui, lang, teach, a) {
                action = Some(a);
            }
        }
    });
    ui.separator();
    let selected = teach.selected();
    let mut pick = None;
    for (i, row) in teach.rows(lang).iter().enumerate() {
        let text = RichText::new(format!("{}. {}", i + 1, row.label)).atom_shrink(true);
        let chosen = selected == Some(i);
        let (widget, hover) = match mark(&row.notes) {
            Some((glyph, hover)) => (egui::Button::selectable(chosen, (glyph, text)), Some(hover)),
            None => (egui::Button::selectable(chosen, text), None),
        };
        let mut response = ui.add(widget.wrap());
        if let Some(hover) = hover {
            response = response.on_hover_text(hover);
        }
        if response.clicked() {
            pick = Some(i);
        }
        if chosen {
            ui.indent(("teach-notes", i), |ui| notes(ui, &row.notes));
        }
    }
    ui.separator();
    ui.horizontal(|ui| {
        for a in [Action::Reset, Action::Save] {
            if button(ui, lang, teach, a) {
                action = Some(a);
            }
        }
    });
    if let Some(note) = &s.note {
        ui.label(RichText::new(note).color(colour(Mark::Missing)));
    }
    ui.add_space(8.0);
    ui.weak(fill(lang, "teach.demonstrations", &[&count]));

    if pick.is_some() {
        teach.select(pick);
    }
    match action {
        Some(Action::AddMove) => teach.add_move(),
        Some(Action::AddGrip) => teach.add_grip(),
        Some(Action::Delete) => {
            teach.delete();
        }
        Some(Action::Up) => {
            teach.move_up();
        }
        Some(Action::Down) => {
            teach.move_down();
        }
        Some(Action::Reset) => s.note = teach.reset().err(),
        Some(Action::Save) => s.note = teach.save().err(),
        Some(Action::Try | Action::Reroll) | None => {}
    }
}

/// Centre: try it once, or somewhere else; while it runs, a line; then what it came to and the
/// attempt playing. Before any try, the scene.
fn centre(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    let dt = f64::from(ui.input(|i| i.stable_dt));
    let Some(open) = &app.project else { return };
    let (run, phases) = (app.launch.pid(), open.phases.clone());
    let s = &mut app.teach;
    let Some(Ok(teach)) = &mut s.teach else {
        return;
    };
    let busy = run.or(s.launch.pid());
    let refused = teach.refusal(busy, false, &phases).map(|r| r.text(lang));
    let mut go = None;
    ui.horizontal(|ui| {
        for a in [Action::Try, Action::Reroll] {
            let mut response =
                ui.add_enabled(refused.is_none(), egui::Button::new(t(lang, a.key())));
            if let Some(hint) = a.hint_key() {
                response = response.on_hover_text(t(lang, hint));
            }
            if let Some(why) = &refused {
                response = response.on_disabled_hover_text(why);
            }
            if response.clicked() {
                go = Some(a);
            }
        }
    });
    if let (Some(why), false) = (&refused, s.running) {
        ui.weak(why);
    }
    if let Some(a) = go {
        if a == Action::Reroll {
            teach.next_seed();
        }
        match teach.start_try(busy, false, &phases) {
            Ok(start) => {
                s.launch.start_in(&start.argv, &s.root);
                s.running = true;
                s.shown = Some(Shown::new(start.dir, true, None, None));
                s.note = None;
            }
            Err(refused) => s.note = Some(refused.text(lang)),
        }
    }
    if let Some(note) = &s.note {
        ui.label(RichText::new(note).color(colour(Mark::Missing)));
    }
    let Some(shown) = &mut s.shown else {
        crate::ui::scene::draw(app, ui, Pane::Viewport);
        return;
    };
    ui.separator();
    let line = teach.verdict(lang, shown.attempt.as_ref(), s.running);
    let tone = match &shown.attempt {
        _ if s.running => ui.visuals().strong_text_color(),
        Some(tile) if tile.success => colour(Mark::Have),
        _ => colour(Mark::Missing),
    };
    ui.horizontal(|ui| {
        if s.running {
            ui.spinner();
        }
        ui.label(RichText::new(line).strong().color(tone));
    });
    // A try that did not finish: `es`'s own last words say why (untranslated, as the console).
    let ended = matches!(
        s.launch.state(),
        LaunchState::Exited { .. } | LaunchState::Failed(_)
    );
    if shown.ours && ended && shown.attempt.is_none() {
        ui.weak(s.launch.status_line());
        let lines: Vec<&str> = s.launch.lines().collect();
        for line in &lines[lines.len().saturating_sub(LAST_LINES)..] {
            ui.monospace(*line);
        }
    }
    let mut pick = None;
    match &mut shown.player {
        Some(Ok((dir, player))) => {
            play(lang, ui, dt, dir, player, &mut pick);
            if let Some(cell) = pick {
                *player = Player::open(dir, s.scene.as_deref(), cell);
            }
        }
        Some(Err(e)) => {
            ui.weak(e.as_str());
        }
        None => {
            crate::ui::scene::draw(app, ui, Pane::Viewport);
        }
    }
}

/// Right: the selected block's fields - only those of its kind.
fn inspector(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    let Some(Ok(teach)) = &mut app.teach.teach else {
        return;
    };
    let Some(i) = teach.selected() else {
        if let Some(open) = &app.project {
            ui.heading(fill(lang, "shell.project", &[&open.project.file.name]));
        }
        return;
    };
    let block = teach.program().blocks[i].clone();
    if let Some(row) = teach.rows(lang).get(i) {
        ui.strong(format!("{}. {}", i + 1, row.label));
    }
    ui.separator();
    let mut edits = Vec::new();
    if let Some(target) = &block.target {
        target_field(ui, lang, teach, target, &mut edits);
        let (name, metres) = match block.height {
            Some(h) if block.above.is_none() => (FieldName::Height, h),
            _ => (FieldName::Above, block.above.unwrap_or_default()),
        };
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            for other in [FieldName::Above, FieldName::Height] {
                let chosen = other == name;
                if ui.selectable_label(chosen, t(lang, other.key())).clicked() && !chosen {
                    // The same number of centimetres, kept on the other slider's scale.
                    let (_, range) = other.unit().unwrap_or(("", 0.0..=0.0));
                    let cm = (metres * 100.0).clamp(*range.start(), *range.end());
                    edits.push(height(other, cm / 100.0));
                }
            }
        });
        if let Some(cm) = number(ui, lang, name, metres * 100.0, 0.5) {
            edits.push(height(name, cm / 100.0));
        }
        let pitch = block.pitch.unwrap_or_default();
        if let Some(deg) = number(ui, lang, FieldName::Pitch, pitch, 1.0) {
            edits.push(Field::Pitch(deg));
        }
    }
    ui.add_space(6.0);
    ui.label(t(lang, FieldName::Grip.key()));
    ui.horizontal(|ui| {
        for grip in [Grip::Open, Grip::Closed] {
            if ui
                .selectable_label(block.grip == grip, t(lang, grip_key(grip)))
                .clicked()
            {
                edits.push(Field::Grip(grip));
            }
        }
    });
    if block.target.is_none() {
        let step = teach.wait_step().unwrap_or(0.0);
        let wait = block.wait.unwrap_or_default();
        if let Some(s) = number(ui, lang, FieldName::Wait, wait, step) {
            // A whole number of steps, as a product: a sum of steps drifts off the grid.
            let s = if step > 0.0 {
                (s / step).round() * step
            } else {
                s
            };
            edits.push(Field::Wait(s));
        }
    }
    for edit in edits {
        teach.edit(edit);
    }
}

fn height(name: FieldName, metres: f64) -> Field {
    match name {
        FieldName::Height => Field::Height(metres),
        _ => Field::Above(metres),
    }
}

/// A number field on its slider, in the unit the model shows it in; the new value once moved.
/// Clamped only when edited: a value from the file is shown as it is, never rewritten.
fn number(ui: &mut egui::Ui, lang: Lang, name: FieldName, value: f64, step: f64) -> Option<f64> {
    let (unit, range) = name.unit()?;
    ui.add_space(6.0);
    ui.label(t(lang, name.key()));
    let mut shown = value;
    let slider = egui::Slider::new(&mut shown, range)
        .suffix(t(lang, unit))
        .step_by(step)
        .clamping(egui::SliderClamping::Edits);
    ui.add(slider).changed().then_some(shown)
}

/// The target: the object, a place of the scene, or a point on the floor in centimetres.
fn target_field(
    ui: &mut egui::Ui,
    lang: Lang,
    teach: &Teach,
    target: &Target,
    edits: &mut Vec<Field>,
) {
    ui.label(t(lang, FieldName::Target.key()));
    let kind = TargetKind::of(target);
    let places = teach.places();
    ui.horizontal(|ui| {
        for other in TargetKind::ALL {
            let to = match other {
                TargetKind::Object => Some(Target::Named(OBJECT.to_owned())),
                TargetKind::Place => places.first().map(|p| Target::Named((*p).to_owned())),
                TargetKind::Point => teach.point_of(target).map(Target::Point),
            };
            let chosen = other == kind;
            let widget = egui::Button::selectable(chosen, t(lang, other.key()));
            if ui.add_enabled(to.is_some(), widget).clicked() && !chosen {
                edits.extend(to.map(Field::Target));
            }
        }
    });
    match target {
        Target::Named(name) if kind == TargetKind::Place => {
            egui::ComboBox::from_id_salt("teach-place")
                .selected_text(name.as_str())
                .show_ui(ui, |ui| {
                    for place in &places {
                        if ui.selectable_label(name == place, *place).clicked() {
                            edits.push(Field::Target(Target::Named((*place).to_owned())));
                        }
                    }
                });
        }
        Target::Point([x, y]) => {
            let unit = t(lang, "teach.unit.cm");
            let mut cm = [x * 100.0, y * 100.0];
            let mut moved = false;
            ui.horizontal(|ui| {
                for v in &mut cm {
                    let drag = egui::DragValue::new(v)
                        .speed(0.5)
                        .fixed_decimals(1)
                        .suffix(unit);
                    moved |= ui.add(drag).changed();
                }
            });
            if moved {
                edits.push(Field::Target(Target::Point([cm[0] / 100.0, cm[1] / 100.0])));
            }
        }
        Target::Named(_) => {}
    }
}
