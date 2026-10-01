//! ①'s task as sentences (packet M17/G8, `docs/design/scene-authoring.md` section 4.6): in the
//! right pane beside the inspector, what success is, what fails, the time limit, what starts
//! where, what the robot sees and what is paid for, each sentence a drop-down or a number per
//! slot in the reader's word order.
//!
//! Drawing only. Which sentences and slots there are, the choices a slot offers, what an edit
//! implies, the weight levels, "say the task" and the camera check are
//! [`es_editor_scene::sentence`]'s, under test. What is decided here is when an edit is handed
//! over: once the person lets go, the edited specification as one command, so a drag of a number
//! is one undo step; a refused specification stays in its sentences with the reason above and is
//! not tried again until it changes.

use std::path::Path;

use eframe::egui;
use egui::{Color32, RichText};
use es_assets::esscene::ShapeDoc;
use es_assets::scene::SceneDesc;
use es_editor_scene::sentence::{
    self as s, vocab, At, Clause, Field, Level, Look, Relation, Sentence, Slot, TaskSpec, Unit,
};
use es_editor_scene::{new_body, new_region, Camera, Command, Item, Record, Refusal, SceneModel};

use crate::model::i18n::{fill, t, Lang};
use crate::model::template::templates_root;
use crate::ui::author::Author;

const RED: Color32 = Color32::from_rgb(220, 80, 70);

/// The task pane between frames.
#[derive(Default)]
pub(crate) struct Task {
    /// Whether the right pane shows the task rather than the selection's fields.
    shown: bool,
    /// The specification as edited so far, and the model revision it was read at.
    draft: Option<(usize, Option<TaskSpec>)>,
    /// The last refusal and the specification it refused.
    refusal: Option<(Refusal, TaskSpec)>,
    /// The camera check of the model revision it was made at.
    unseen: Option<(usize, Unseen)>,
}

/// Each observed camera and what of the sentences it does not see.
type Unseen = Vec<(String, Vec<String>)>;

/// A template's pieces: text, and the slot a `{n}` names.
enum Piece<'a> {
    Text(&'a str),
    Hole(usize),
}

fn pieces(template: &str) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let Some(close) = rest[open..].find('}').map(|c| open + c) else {
            break;
        };
        let Ok(i) = rest[open + 1..close].parse() else {
            break;
        };
        if open > 0 {
            out.push(Piece::Text(&rest[..open]));
        }
        out.push(Piece::Hole(i));
        rest = &rest[close + 1..];
    }
    if !rest.is_empty() {
        out.push(Piece::Text(rest));
    }
    out
}

/// A slot as the reader reads it.
fn said(lang: Lang, slot: &Slot) -> String {
    match slot {
        Slot::Name(n) => n.clone(),
        Slot::Relation(r, o) => t(lang, s::relation_key(*r, *o)).to_owned(),
        Slot::Draw(d) => t(lang, s::draw_key(*d)).to_owned(),
        Slot::Number(v, u) => s::show(v.unwrap_or(0.0), *u),
        Slot::Point(p) => s::show_point(*p),
    }
}

/// A sentence in `lang`'s words, 🎲 after a start item drawn anew every attempt.
pub fn words(lang: Lang, sentence: &Sentence) -> String {
    let mut out: String = (pieces(t(lang, sentence.key)).iter())
        .map(|p| match p {
            Piece::Text(x) => (*x).to_owned(),
            Piece::Hole(i) => {
                (sentence.slots.get(*i)).map_or_else(String::new, |(_, x)| said(lang, x))
            }
        })
        .collect();
    if sentence.dice {
        out.push_str(" \u{1f3b2}");
    }
    out
}

/// The right pane of ①: the selection's fields or the task, one tab each.
pub(crate) fn summary(ui: &mut egui::Ui, lang: Lang, author: &mut Author) {
    ui.horizontal(|ui| {
        let shown = &mut author.task.shown;
        ui.selectable_value(shown, false, t(lang, "author.task.tab.selected"));
        ui.selectable_value(shown, true, t(lang, "author.task.tab.task"));
    });
    ui.separator();
    if author.task.shown {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| panel(ui, lang, author));
    } else {
        author.inspector(ui, lang);
    }
}

fn num(ui: &mut egui::Ui, v: &mut f64, speed: f64, suffix: &str) -> bool {
    let w = egui::DragValue::new(v)
        .speed(speed)
        .suffix(suffix)
        .max_decimals(3);
    ui.add(w.update_while_editing(false)).changed()
}

fn level_word(lang: Lang, l: Level) -> String {
    match l {
        Level::Custom(v) => fill(lang, l.key(), &[&v.to_string()]),
        _ => t(lang, l.key()).to_owned(),
    }
}

/// A level picked from low, medium and high (and, with `off`, none): `now` unless changed.
fn level(
    ui: &mut egui::Ui,
    lang: Lang,
    salt: &str,
    now: Option<Level>,
    off: bool,
) -> Option<Level> {
    let word = |l: Option<Level>| {
        l.map_or_else(
            || t(lang, "author.task.level.off").to_owned(),
            |l| level_word(lang, l),
        )
    };
    let mut pick = now;
    egui::ComboBox::from_id_salt(salt)
        .selected_text(word(now))
        .show_ui(ui, |ui| {
            if off {
                ui.selectable_value(&mut pick, None, word(None));
            }
            for l in s::LEVELS {
                ui.selectable_value(&mut pick, Some(l), word(Some(l)));
            }
        });
    pick
}

/// One slot as a widget; the new slot when the person changed it.
fn widget(
    ui: &mut egui::Ui,
    lang: Lang,
    scene: &SceneDesc,
    of: &Sentence,
    field: Field,
    slot: &Slot,
) -> Option<Slot> {
    let salt = format!("{field:?}");
    let point = matches!(of.key, "author.task.near" | "author.task.farther");
    match slot {
        Slot::Name(n) => {
            let mut pick = Slot::Name(n.clone());
            egui::ComboBox::from_id_salt(salt)
                .selected_text(n.as_str())
                .show_ui(ui, |ui| {
                    for c in s::choices(scene, field, of) {
                        ui.selectable_value(&mut pick, Slot::Name(c.clone()), c);
                    }
                    if point {
                        let at = scene
                            .bodies
                            .iter()
                            .find(|b| b.name == *n)
                            .map(|b| b.pose.position);
                        let p = at.map_or([0.0; 3], |p| [p.x, p.y, p.z]);
                        ui.selectable_value(
                            &mut pick,
                            Slot::Point(p),
                            t(lang, "author.task.point"),
                        );
                    }
                });
            (pick != *slot).then_some(pick)
        }
        Slot::Point(p) => {
            let mut pick = None;
            egui::ComboBox::from_id_salt(salt)
                .selected_text(t(lang, "author.task.point"))
                .show_ui(ui, |ui| {
                    for c in s::choices(scene, field, of) {
                        if ui.selectable_label(false, &c).clicked() {
                            pick = Some(Slot::Name(c));
                        }
                    }
                });
            let mut q = *p;
            let mut changed = false;
            for v in &mut q {
                let mut shown = Unit::Cm.shown(*v);
                if num(ui, &mut shown, 0.5, " cm") {
                    (*v, changed) = (Unit::Cm.stored(shown), true);
                }
            }
            pick.or_else(|| changed.then_some(Slot::Point(q)))
        }
        Slot::Relation(r, o) => {
            let subject = match of.slots.first() {
                Some((_, Slot::Name(n))) => n.as_str(),
                _ => "",
            };
            let mut pick = (*r, *o);
            egui::ComboBox::from_id_salt(salt)
                .selected_text(t(lang, s::relation_key(*r, *o)))
                .show_ui(ui, |ui| {
                    for (rel, obj) in vocab::relations(scene, subject) {
                        let word = t(lang, s::relation_key(rel, obj));
                        if rel == Relation::Touches {
                            ui.add_enabled(false, egui::Button::selectable(false, word))
                                .on_disabled_hover_text(t(lang, s::TOUCHES));
                        } else {
                            ui.selectable_value(&mut pick, (rel, obj), word);
                        }
                    }
                });
            (pick != (*r, *o)).then_some(Slot::Relation(pick.0, pick.1))
        }
        Slot::Draw(d) => {
            let mut pick = *d;
            egui::ComboBox::from_id_salt(salt)
                .selected_text(t(lang, s::draw_key(*d)))
                .show_ui(ui, |ui| {
                    for x in s::DRAWS {
                        ui.selectable_value(&mut pick, x, t(lang, s::draw_key(x)));
                    }
                });
            (pick != *d).then_some(Slot::Draw(pick))
        }
        Slot::Number(v, u) => {
            let mut shown = u.shown(v.unwrap_or(0.0));
            let speed = if *u == Unit::Seconds { 0.1 } else { 0.5 };
            num(ui, &mut shown, speed, u.suffix()).then(|| Slot::Number(Some(u.stored(shown)), *u))
        }
    }
}

/// A sentence's slots as widgets between its words, in `lang`'s order; what the person changed.
fn sentence(
    ui: &mut egui::Ui,
    lang: Lang,
    scene: &SceneDesc,
    of: &Sentence,
) -> Option<(Field, Slot)> {
    let mut changed = None;
    for piece in pieces(t(lang, of.key)) {
        match piece {
            Piece::Text(x) if !x.trim().is_empty() => {
                ui.label(x.trim());
            }
            Piece::Text(_) => {}
            Piece::Hole(i) => {
                let Some((field, slot)) = of.slots.get(i) else {
                    continue;
                };
                if let Some(new) = widget(ui, lang, scene, of, *field, slot) {
                    changed = Some((*field, new));
                }
            }
        }
    }
    changed
}

/// The reason under the row `path` when the last refusal is about it.
fn reason(ui: &mut egui::Ui, lang: Lang, refused: Option<&Refusal>, path: &str) {
    if let Some(r) = refused.filter(|r| r.field == path || r.field.starts_with(&format!("{path}.")))
    {
        let args: Vec<&str> = r.args.iter().map(String::as_str).collect();
        ui.label(RichText::new(fill(lang, r.key, &args)).color(RED));
    }
}

enum Act {
    Up,
    Down,
    Remove,
}

fn clause_mut(spec: &mut TaskSpec, at: At) -> Option<&mut Clause> {
    if at.failure {
        spec.failure.as_mut()?.clauses.get_mut(at.index)
    } else {
        spec.success.clauses.get_mut(at.index)
    }
}

/// A section's clauses: each sentence, its reward for getting there, its order and removal.
fn clauses(
    ui: &mut egui::Ui,
    lang: Lang,
    scene: &SceneDesc,
    spec: &mut TaskSpec,
    failure: bool,
    refused: Option<&Refusal>,
) {
    let n = if failure {
        spec.failure.as_ref().map_or(0, |f| f.clauses.len())
    } else {
        spec.success.clauses.len()
    };
    let mut act = None;
    for index in 0..n {
        let at = At { failure, index };
        let Some(c) = clause_mut(spec, at) else {
            continue;
        };
        ui.push_id(at.path(), |ui| {
            ui.horizontal_wrapped(|ui| {
                if let Some((field, slot)) = sentence(ui, lang, scene, &s::clause(scene, c)) {
                    s::edit_clause(scene, c, field, slot);
                }
                if s::shapable(c).is_some() {
                    let mut on = c.shaping.is_some();
                    let hint = t(lang, "author.task.shaped.hint");
                    if ui
                        .checkbox(&mut on, t(lang, "author.task.shaped"))
                        .on_hover_text(hint)
                        .changed()
                    {
                        s::set_shaping(c, on.then_some(Level::Medium));
                    }
                    if let Some(now) = s::shaping_level(c) {
                        let pick = level(ui, lang, "shaping", Some(now), false);
                        if pick != Some(now) {
                            s::set_shaping(c, pick);
                        }
                    }
                }
                for (glyph, hint, a) in [
                    ("\u{2191}", "author.task.up.hint", Act::Up),
                    ("\u{2193}", "author.task.down.hint", Act::Down),
                    ("\u{1f5d1}", "author.task.remove.hint", Act::Remove),
                ] {
                    if ui
                        .small_button(glyph)
                        .on_hover_text(t(lang, hint))
                        .clicked()
                    {
                        act = Some((at, a));
                    }
                }
            });
            reason(ui, lang, refused, &at.path());
        });
    }
    match act {
        Some((at, Act::Up)) => s::move_clause(spec, at, false),
        Some((at, Act::Down)) => s::move_clause(spec, at, true),
        Some((at, Act::Remove)) => s::remove_clause(spec, at),
        None => {}
    }
    if ui.button(t(lang, "author.task.add")).clicked() {
        s::add_clause(spec, scene, failure);
    }
}

/// What starts where: the strength of the 🎲, each placement with its 🎲 toggle.
fn start(
    ui: &mut egui::Ui,
    lang: Lang,
    scene: &SceneDesc,
    spec: &mut TaskSpec,
    refused: Option<&Refusal>,
) {
    ui.horizontal(|ui| {
        ui.label(t(lang, "author.task.strength"))
            .on_hover_text(t(lang, "author.task.strength.hint"));
        let now = s::strength(spec);
        if let Some(l) = level(ui, lang, "strength", Some(now), false).filter(|l| *l != now) {
            s::set_strength(spec, l);
        }
    });
    let n = spec.start.as_ref().map_or(0, |st| st.items.len());
    let mut remove = None;
    for index in 0..n {
        let Some(item) = spec.start.as_mut().and_then(|st| st.items.get_mut(index)) else {
            continue;
        };
        ui.push_id(("start", index), |ui| {
            ui.horizontal_wrapped(|ui| {
                if let Some((field, slot)) = sentence(ui, lang, scene, &s::start_item(scene, item))
                {
                    s::edit_item(item, field, slot);
                }
                let dice = item.dice == Some(true);
                let hint = t(lang, "author.task.dice.hint");
                if ui
                    .selectable_label(dice, "\u{1f3b2}")
                    .on_hover_text(hint)
                    .clicked()
                {
                    item.dice = (!dice).then_some(true);
                }
                if ui
                    .small_button("\u{1f5d1}")
                    .on_hover_text(t(lang, "author.task.remove.hint"))
                    .clicked()
                {
                    remove = Some(index);
                }
            });
            reason(ui, lang, refused, &format!("start[{index}]"));
        });
    }
    if let Some(i) = remove {
        s::remove_item(spec, i);
    }
    if ui.button(t(lang, "author.task.add")).clicked() {
        s::add_item(spec, scene);
    }
}

/// What the robot sees: the scene's cameras checked (each with the camera check beside it),
/// the picture's size and look, and the senses the robot and the simulator give.
fn observe(
    ui: &mut egui::Ui,
    lang: Lang,
    scene: &SceneDesc,
    spec: &mut TaskSpec,
    unseen: &[(String, Vec<String>)],
) {
    let observed = |spec: &TaskSpec, name: &str| {
        let o = spec.observe.as_ref().and_then(|o| o.cameras.as_ref());
        o.is_some_and(|c| c.iter().any(|x| x == name))
    };
    for cam in &scene.cameras {
        ui.horizontal_wrapped(|ui| {
            let mut on = observed(spec, &cam.name);
            if ui.checkbox(&mut on, &cam.name).changed() {
                s::set_camera(spec, &cam.name, on);
            }
            match unseen.iter().find(|(c, _)| *c == cam.name) {
                Some((_, missing)) if !missing.is_empty() => {
                    let line = fill(lang, "author.task.unseen", &[&missing.join(", ")]);
                    ui.label(RichText::new(line).color(RED));
                }
                Some(_) => {
                    ui.weak(t(lang, "author.task.sees"));
                }
                None => {}
            }
        });
    }
    ui.horizontal_wrapped(|ui| {
        ui.label(t(lang, "author.task.camera_px"));
        let mut px = spec
            .observe
            .as_ref()
            .and_then(|o| o.camera_px)
            .unwrap_or(96);
        let drag = egui::DragValue::new(&mut px).range(16..=512).suffix(" px");
        if ui.add(drag.update_while_editing(false)).changed() {
            spec.observe.get_or_insert_with(Default::default).camera_px = Some(px);
        }
    });
    let now = s::look(spec);
    let mut pick = now;
    ui.horizontal_wrapped(|ui| {
        ui.label(t(lang, "author.task.look"));
        for (look, key) in [
            (Look::Quick, "author.task.look.quick"),
            (Look::Material, "author.task.look.material"),
            (Look::Traced, "author.task.look.traced"),
        ] {
            if look == Look::Material {
                ui.add_enabled(false, egui::Button::selectable(false, t(lang, key)))
                    .on_disabled_hover_text(t(lang, "author.task.look.material.hint"));
            } else {
                ui.selectable_value(&mut pick, look, t(lang, key));
            }
        }
        let render = (spec.observe.as_mut()).and_then(|o| o.render.as_mut());
        if let Some(r) = render.filter(|_| now == Look::Traced) {
            for (key, v) in [
                ("author.task.spp", &mut r.spp),
                ("author.task.bounces", &mut r.bounces),
            ] {
                let mut x = v.unwrap_or(1);
                ui.label(t(lang, key));
                if ui
                    .add(
                        egui::DragValue::new(&mut x)
                            .range(1..=4096)
                            .update_while_editing(false),
                    )
                    .changed()
                {
                    *v = Some(x);
                }
            }
        }
    });
    if pick != now {
        s::set_look(spec, pick);
    }
    let tables = spec
        .observe
        .as_ref()
        .map(|o| [o.state.clone(), o.privileged.clone()]);
    let mut act: Option<(String, bool)> = None;
    for (key, table) in ["author.task.state", "author.task.privileged"]
        .into_iter()
        .zip(tables.into_iter().flatten())
    {
        ui.strong(t(lang, key));
        for (channel, source) in table.iter().flatten() {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("{channel} \u{2190} {source}"));
                if ui
                    .small_button("\u{2194}")
                    .on_hover_text(t(lang, "author.task.swap.hint"))
                    .clicked()
                {
                    act = Some((channel.clone(), true));
                }
                if ui
                    .small_button("\u{1f5d1}")
                    .on_hover_text(t(lang, "author.task.remove.hint"))
                    .clicked()
                {
                    act = Some((channel.clone(), false));
                }
            });
        }
    }
    match act {
        Some((channel, true)) => s::swap_source(spec, &channel),
        Some((channel, false)) => s::remove_source(spec, &channel),
        None => {}
    }
    let mut add = None;
    egui::ComboBox::from_id_salt("add-source")
        .selected_text(t(lang, "author.task.add_source"))
        .show_ui(ui, |ui| {
            for source in s::sources(scene) {
                if ui.selectable_label(false, &source).clicked() {
                    add = Some(source);
                }
            }
        });
    if let Some(source) = add {
        s::add_source(spec, &source, false);
    }
}

/// The task: "say the task" without one; with one, every section of it.
fn panel(ui: &mut egui::Ui, lang: Lang, author: &mut Author) {
    let rev = author.model.revision();
    let task = &mut author.task;
    if task.draft.as_ref().is_none_or(|(r, _)| *r != rev) {
        task.draft = Some((rev, author.model.spec().cloned()));
        task.refusal = None;
    }
    if task.unseen.as_ref().is_none_or(|(r, _)| *r != rev) {
        task.unseen = Some((rev, author.model.unseen()));
    }
    let scene = std::sync::Arc::clone(author.model.scene());
    let refused = task.refusal.as_ref().map(|(r, _)| r.clone());
    if let Some(r) = &refused {
        let args: Vec<&str> = r.args.iter().map(String::as_str).collect();
        let why = fill(lang, r.key, &args);
        ui.label(RichText::new(fill(lang, "author.refused", &[&r.field, &why])).color(RED));
        if ui.button(t(lang, "author.task.put_back")).clicked() {
            task.draft = Some((rev, author.model.spec().cloned()));
            task.refusal = None;
        }
    }
    let unseen = task
        .unseen
        .as_ref()
        .map(|(_, u)| u.clone())
        .unwrap_or_default();
    let Some((_, draft)) = task.draft.as_mut() else {
        return;
    };
    match draft {
        None => {
            ui.label(t(lang, "author.task.none"));
            let say = ui
                .button(t(lang, "author.task.say"))
                .on_hover_text(t(lang, "author.task.say.hint"));
            if say.clicked() {
                let hz = (author.bundle.as_ref()).and_then(|[task, _]| s::control_hz(task));
                *draft = Some(s::new_spec(
                    &scene,
                    author.model.doc(),
                    hz.unwrap_or(s::CONTROL_HZ),
                ));
            }
        }
        Some(spec) => {
            let refused = refused.as_ref();
            ui.horizontal(|ui| {
                ui.label(t(lang, "author.task.robot"));
                egui::ComboBox::from_id_salt("robot")
                    .selected_text(spec.robot.as_str())
                    .show_ui(ui, |ui| {
                        for r in s::robots(&scene, author.model.doc()) {
                            ui.selectable_value(&mut spec.robot, r.clone(), r);
                        }
                    });
            });
            reason(ui, lang, refused, "robot");
            ui.heading(t(lang, "author.task.success"));
            clauses(ui, lang, &scene, spec, false, refused);
            ui.heading(t(lang, "author.task.failure"));
            clauses(ui, lang, &scene, spec, true, refused);
            ui.horizontal_wrapped(|ui| {
                if let Some((_, Slot::Number(Some(v), _))) =
                    sentence(ui, lang, &scene, &s::timeout(spec))
                {
                    spec.timeout_s = v;
                }
            });
            reason(ui, lang, refused, "timeout_s");
            ui.heading(t(lang, "author.task.start"));
            start(ui, lang, &scene, spec, refused);
            ui.heading(t(lang, "author.task.observe"));
            observe(ui, lang, &scene, spec, &unseen);
            reason(ui, lang, refused, "observe");
            ui.heading(t(lang, "author.task.reward"));
            ui.horizontal(|ui| {
                ui.label(t(lang, "author.task.bonus"));
                let now = s::bonus(spec);
                let pick = level(ui, lang, "bonus", now, true);
                if pick != now {
                    s::set_bonus(spec, pick);
                }
            });
        }
    }

    // Once the person lets go: the specification as one command, unless it was refused as is.
    let wanted = draft.clone();
    let settled = !ui.input(|i| i.pointer.any_down()) && !ui.ctx().wants_keyboard_input();
    let tried = (task.refusal.as_ref()).is_some_and(|(_, w)| Some(w) == wanted.as_ref());
    if wanted.as_ref() == author.model.spec() {
        task.refusal = None;
    } else if settled && !tried {
        match author
            .model
            .apply(&Command::Spec(wanted.clone().map(Box::new)))
        {
            Ok(()) => task.refusal = None,
            Err(r) => task.refusal = wanted.map(|w| (r, w)),
        }
    }
}

/// `es-editor --edit-demo sentences|refuse|new-task` (packet M17/G8's captures): ① shows the
/// task; `refuse` then tries an 8.01 s time limit, which is not whole control ticks and is
/// refused; `new-task` says the task of a project that has none, adds a region over SO-101's bin,
/// says "[cube] is inside [it]" and saves. `authored` builds a task on the empty project
/// (packet M17/G9's captures, [`authored`]). `false` for every other stage.
pub(crate) fn demo(author: &mut Author, stage: &str) -> bool {
    if !matches!(stage, "sentences" | "refuse" | "new-task" | "authored") {
        return false;
    }
    author.task.shown = true;
    let m = &mut author.model;
    if stage == "authored" {
        let _ = templates_root().map(|repo| authored(m, &repo));
        return true;
    }
    if stage == "refuse" {
        if let Some(mut spec) = m.spec().cloned() {
            spec.timeout_s = 8.01;
            if let Err(r) = m.apply(&Command::Spec(Some(Box::new(spec.clone())))) {
                author.task.draft = Some((m.revision(), Some(spec.clone())));
                author.task.refusal = Some((r, spec));
            }
        }
    }
    if stage == "new-task" {
        if m.spec().is_none() {
            let hz = (author.bundle.as_ref()).and_then(|[task, _]| s::control_hz(task));
            let said = s::new_spec(m.scene(), m.doc(), hz.unwrap_or(s::CONTROL_HZ));
            let _ = m.apply(&Command::Spec(Some(Box::new(said))));
        }
        let mut region = new_region(&m.unique("bin_area"));
        region.pos = Some([0.14, -0.1, 0.05]);
        let name = region.name.clone();
        let added = m.apply(&Command::Add(Record::Region(region))).is_ok();
        if let (true, Some(mut spec)) = (added, m.spec().cloned()) {
            let scene = std::sync::Arc::clone(m.scene());
            s::add_clause(&mut spec, &scene, false);
            let inside = Slot::Relation(Relation::Inside, true);
            s::edit_clause(
                &scene,
                &mut spec.success.clauses[0],
                Field::Relation,
                inside,
            );
            spec.success.clauses[0].object = Some(name);
            let _ = m.apply(&Command::Spec(Some(Box::new(spec))));
            let _ = m.save();
        }
    }
    true
}

/// A task built on the empty project as a person builds it in ① (packet M17/G9: its oracle 1
/// and `--edit-demo authored`): the library's SO-101 where a view of the origin looks, a 5 cm
/// box in front of it and a target area beside the box, "say the task" (whose default clause is
/// the box still), "[box] is inside [area]", and a save, which generates the documents.
pub fn authored(m: &mut SceneModel, repo: &Path) -> Result<(), String> {
    let refused = |r: Refusal| format!("{}: {} {:?}", r.field, r.key, r.args);
    let lib = es_editor_scene::add::library(repo)?;
    let so101 = (lib.into_iter().find(|r| r.id == "so101")).ok_or("no so101 in the library")?;
    let origin = Camera {
        eye: [0.6, -0.6, 0.8],
        look_at: [0.0; 3],
        fov_y: std::f64::consts::FRAC_PI_4,
        width: 640,
        height: 400,
    };
    m.add(&Item::Robot(so101), &origin, true).map_err(refused)?;
    let mut cube = new_body(&m.unique("box"), ShapeDoc::Box([0.025; 3]));
    cube.pos = Some([0.22, 0.0, 0.025]);
    cube.geoms[0].rgba = Some([0.85, 0.2, 0.15, 1.0]);
    let name = cube.name.clone();
    m.apply(&Command::Add(Record::Body(cube)))
        .map_err(refused)?;
    let mut area = new_region(&m.unique("target"));
    (area.pos, area.size) = (Some([0.22, 0.12, 0.025]), Some([0.04, 0.04, 0.03]));
    let target = area.name.clone();
    m.apply(&Command::Add(Record::Region(area)))
        .map_err(refused)?;
    let scene = std::sync::Arc::clone(m.scene());
    let mut spec = s::new_spec(&scene, m.doc(), s::CONTROL_HZ);
    s::add_clause(&mut spec, &scene, false);
    let first = &mut spec.success.clauses[1];
    first.subject = name;
    let inside = Slot::Relation(Relation::Inside, true);
    s::edit_clause(&scene, first, Field::Relation, inside);
    first.object = Some(target);
    m.apply(&Command::Spec(Some(Box::new(spec))))
        .map_err(refused)?;
    m.save()
}
