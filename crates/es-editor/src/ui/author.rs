//! ① of an editable project (packet M17/G5, `docs/design/scene-authoring.md` section 5): the
//! hierarchy on the left, the viewport's undo / redo / save row, and the inspector on the right,
//! over [`es_editor_scene::SceneModel`].
//!
//! Drawing only. Which rows there are, what a search keeps, what is hidden, which commands exist,
//! whether one is refused and why, how a size or an angle is shown — all the scene model's,
//! under test. What is decided here is when a typed value is handed over: once the person lets
//! go (no pointer button held, no text being typed), as one command, so a drag is one undo step;
//! a refused value stays in its field with the reason under it and is not tried again until it
//! changes.

use std::collections::BTreeSet;
use std::path::Path;

use eframe::egui;
use egui::{Color32, RichText};
use es_assets::esscene::{GeomDoc, JointDoc, JointKindDoc, ShapeDoc};
use es_editor_scene::inspect::{self, ShapeKind};
use es_editor_scene::{
    euler, new_body, new_camera, new_joint, new_light, new_region, tree, BackendKind, Command,
    Entity, Record, Refusal, Regen, RowKind, SceneModel,
};
use es_math::units::{DEG_TO_RAD, RAD_TO_DEG};

use crate::model::i18n::{fill, t, Lang};
use crate::model::scene_view::ScenePreview;

const RED: Color32 = Color32::from_rgb(220, 80, 70);
const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

/// The selected entity's fields as typed so far.
struct Draft {
    entity: Entity,
    /// The model revision it was read at: an edit, an undo or a redo reads it afresh.
    revision: usize,
    record: Record,
    name: String,
    rpy: [f64; 3],
    /// Whether an angle was changed: only then is the quaternion written (its bits kept otherwise).
    rotated: bool,
}

/// ① of one editable project between frames.
pub(crate) struct Author {
    pub(crate) model: SceneModel,
    preview: Option<(usize, Result<ScenePreview, String>)>,
    search: String,
    folded: BTreeSet<String>,
    renaming: Option<(Entity, String)>,
    draft: Option<Draft>,
    /// The last refusal and the record it refused.
    refusal: Option<(Refusal, Option<Record>)>,
    saved: Option<Result<(), String>>,
}

/// What the Add menu makes.
#[derive(Clone, Copy)]
enum Add {
    Shape(ShapeKind),
    Camera,
    Light,
    Region,
}

const ADD: [(Add, &str); 7] = [
    (Add::Shape(ShapeKind::Box), "author.shape.box"),
    (Add::Shape(ShapeKind::Sphere), "author.shape.sphere"),
    (Add::Shape(ShapeKind::Cylinder), "author.shape.cylinder"),
    (Add::Shape(ShapeKind::Capsule), "author.shape.capsule"),
    (Add::Camera, "author.kind.camera"),
    (Add::Light, "author.kind.light"),
    (Add::Region, "author.kind.region"),
];

fn fold_key(row: &es_editor_scene::Row) -> String {
    format!("{:?}/{}", row.kind, row.name)
}

impl Author {
    /// The project's scene under edit; an include starts folded.
    pub(crate) fn open(root: &Path, backends: Vec<BackendKind>) -> Result<Self, String> {
        let model = SceneModel::open(root, backends)?;
        let folded = (model.rows().iter())
            .filter(|r| r.kind == RowKind::Include)
            .map(fold_key)
            .collect();
        Ok(Self {
            model,
            preview: None,
            search: String::new(),
            folded,
            renaming: None,
            draft: None,
            refusal: None,
            saved: None,
        })
    }

    fn apply(&mut self, cmd: &Command, record: Option<Record>) {
        self.refusal = self.model.apply(cmd).err().map(|r| (r, record));
    }

    /// The edited scene as the viewport draws it (what is hidden left out), rebuilt when the
    /// document or what is hidden changes.
    pub(crate) fn preview(&mut self) -> Result<&ScenePreview, String> {
        let rev = self.model.revision();
        if self.preview.as_ref().is_none_or(|(r, _)| *r != rev) {
            let drawn = self.model.drawn();
            let built = (self.model.scene_file())
                .and_then(|path| ScenePreview::from_scene(path, drawn, rev));
            self.preview = Some((rev, built));
        }
        match &self.preview {
            Some((_, Ok(p))) => Ok(p),
            Some((_, Err(e))) => Err(e.clone()),
            None => unreachable!("built above"),
        }
    }

    /// `es-editor --edit-demo`: select the first free body (Shadow Hand's cube) and, from
    /// `edit` on, make it 9 cm and red; `undo` then undoes both (packet M17/G5's captures).
    pub(crate) fn demo(&mut self, stage: &str) {
        let doc = self.model.doc();
        let free = doc.bodies.iter().find(|b| {
            b.joint
                .as_ref()
                .is_some_and(|j| j.kind == JointKindDoc::Free)
                && !b.geoms.is_empty()
        });
        let Some(mut body) = free.cloned() else {
            return;
        };
        let e = Entity::Body(body.name.clone());
        self.model.select(Some(e.clone()));
        if stage == "select" {
            return;
        }
        body.geoms[0].shape = inspect::with_dims(&body.geoms[0].shape, &[0.09, 0.09, 0.09]);
        self.apply(&Command::Set(e.clone(), Record::Body(body.clone())), None);
        body.geoms[0].material = None;
        body.geoms[0].rgba = Some([0.9, 0.15, 0.1, 1.0]);
        self.apply(&Command::Set(e, Record::Body(body)), None);
        if stage == "undo" {
            self.model.undo();
            self.model.undo();
        }
    }

    /// Undo, redo, save with its unsaved mark, what the last save and generation did, and their
    /// keys: Ctrl+Z, Ctrl+Y (or Ctrl+Shift+Z), Ctrl+S, Delete — not while a field is typed in.
    pub(crate) fn toolbar(&mut self, ui: &mut egui::Ui, lang: Lang) {
        use egui::{Key, KeyboardShortcut as K, Modifiers as M};
        let typing = ui.ctx().wants_keyboard_input();
        let key = |k: K| !typing && ui.input_mut(|i| i.consume_shortcut(&k));
        let undo = key(K::new(M::COMMAND, Key::Z));
        let redo = key(K::new(M::COMMAND, Key::Y)) || key(K::new(M::COMMAND | M::SHIFT, Key::Z));
        let save = key(K::new(M::COMMAND, Key::S));
        let delete = key(K::new(M::NONE, Key::Delete));
        ui.horizontal_wrapped(|ui| {
            let m = &mut self.model;
            let b = |ui: &mut egui::Ui, on: bool, key: &'static str, hint: &'static str| {
                ui.add_enabled(on, egui::Button::new(t(lang, key)))
                    .on_hover_text(t(lang, hint))
                    .clicked()
            };
            if b(ui, m.can_undo(), "edit.undo", "author.undo.hint") || undo {
                m.undo();
            }
            if b(ui, m.can_redo(), "edit.redo", "author.redo.hint") || redo {
                m.redo();
            }
            let dirty = m.dirty();
            let word = if dirty {
                format!("{} \u{25cf}", t(lang, "edit.save"))
            } else {
                t(lang, "edit.save").to_owned()
            };
            let hint = if dirty {
                "author.unsaved"
            } else {
                "author.save.hint"
            };
            let clicked = ui
                .add_enabled(dirty, egui::Button::new(word))
                .on_hover_text(t(lang, hint))
                .on_disabled_hover_text(t(lang, hint))
                .clicked();
            if (clicked || save) && dirty {
                self.saved = Some(m.save());
            }
            let line = match (&self.saved, m.generated()) {
                (Some(Err(why)), _) => Some(("author.save_failed", vec![why.clone()])),
                (_, Regen::Written(files)) => {
                    Some(("author.generated", vec![files.len().to_string()]))
                }
                (_, Regen::Failed(why)) => Some(("author.generate_failed", vec![why.clone()])),
                (_, Regen::NoSpec) => Some(("author.no_spec", vec![])),
                (_, Regen::NotYet) => Some(("author.not_generated", vec![])),
            };
            if let Some((key, args)) = line {
                let args: Vec<&str> = args.iter().map(String::as_str).collect();
                ui.weak(fill(lang, key, &args));
            }
        });
        if delete {
            if let Some(e) = self.model.selection().cloned() {
                self.apply(&Command::Delete(e), None);
            }
        }
    }

    /// The hierarchy: the Add menu and the search, then the rows — fold arrow, eye, name and
    /// kind; a right click duplicates, deletes or renames. New things go where the viewport looks.
    pub(crate) fn hierarchy(&mut self, ui: &mut egui::Ui, lang: Lang, at: [f64; 3]) {
        ui.horizontal(|ui| {
            ui.menu_button(t(lang, "author.add"), |ui| {
                for (what, key) in ADD {
                    if ui.button(t(lang, key)).clicked() {
                        self.add(what, at);
                        ui.close();
                    }
                }
            });
            let search =
                egui::TextEdit::singleline(&mut self.search).hint_text(t(lang, "author.search"));
            ui.add(search);
        });
        ui.separator();
        let rows = self.model.rows();
        let show = tree::matches(&rows, &self.search);
        let searching = !self.search.trim().is_empty();
        let folded_above = |i: usize| {
            let mut at = rows[i].parent;
            while let Some(p) = at {
                if self.folded.contains(&fold_key(&rows[p])) {
                    return true;
                }
                at = rows[p].parent;
            }
            false
        };
        let visible: Vec<bool> = (0..rows.len())
            .map(|i| show[i] && (searching || !folded_above(i)))
            .collect();
        let mut toggle_fold = None;
        let mut picked = None;
        let mut command = None;
        let mut rename_done = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for (i, row) in rows.iter().enumerate() {
                    if !visible[i] {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        ui.add_space(row.depth as f32 * 14.0);
                        let parent = rows.get(i + 1).is_some_and(|r| r.parent == Some(i));
                        if parent {
                            let open = !self.folded.contains(&fold_key(row));
                            let arrow = if open { "\u{23f7}" } else { "\u{23f5}" };
                            if ui.small_button(arrow).clicked() {
                                toggle_fold = Some(fold_key(row));
                            }
                        } else {
                            ui.add_space(ui.spacing().interact_size.y);
                        }
                        let Some(e) = &row.entity else {
                            ui.weak(&row.name);
                            return;
                        };
                        if row.kind.drawn() {
                            let hidden = self.model.hidden(e);
                            let eye = RichText::new("\u{1f441}");
                            let eye = if hidden {
                                eye.weak().strikethrough()
                            } else {
                                eye
                            };
                            let r = ui
                                .small_button(eye)
                                .on_hover_text(t(lang, "author.eye.hint"));
                            if r.clicked() {
                                command = Some(Err(e.clone()));
                            }
                        }
                        if let Some((_, text)) = self.renaming.as_mut().filter(|(r, _)| r == e) {
                            let r = ui.text_edit_singleline(text);
                            r.request_focus();
                            if r.lost_focus() {
                                rename_done = Some(Command::Rename(e.clone(), text.clone()));
                            }
                            return;
                        }
                        let selected = self.model.selection() == Some(e);
                        let r = ui.selectable_label(selected, &row.name);
                        if r.clicked() {
                            picked = Some(e.clone());
                        }
                        r.context_menu(|ui| {
                            if !matches!(e, Entity::Physics) {
                                if ui.button(t(lang, "author.duplicate")).clicked() {
                                    command = Some(Ok(Command::Duplicate(e.clone())));
                                    ui.close();
                                }
                                if ui.button(t(lang, "author.rename")).clicked() {
                                    self.renaming = Some((e.clone(), e.label(self.model.doc())));
                                    ui.close();
                                }
                                if ui.button(t(lang, "author.delete")).clicked() {
                                    command = Some(Ok(Command::Delete(e.clone())));
                                    ui.close();
                                }
                            }
                        });
                        ui.weak(t(lang, row.kind.key()));
                    });
                }
            });
        if let Some(key) = toggle_fold {
            if !self.folded.remove(&key) {
                self.folded.insert(key);
            }
        }
        if let Some(e) = picked {
            self.model.select(Some(e));
        }
        match command {
            Some(Ok(cmd)) => self.apply(&cmd, None),
            Some(Err(e)) => self.model.toggle_hidden(&e),
            None => {}
        }
        if let Some(cmd) = rename_done {
            self.renaming = None;
            self.apply(&cmd, None);
        }
    }

    #[allow(clippy::many_single_char_names)] // a record per kind, and a point's x, y, z
    fn add(&mut self, what: Add, [x, y, z]: [f64; 3]) {
        let m = &self.model;
        let record = match what {
            Add::Shape(kind) => {
                let name = m.unique(kind.base());
                let shape = inspect::reshape(&ShapeDoc::Box([0.025; 3]), kind);
                let mut b = new_body(&name, shape);
                b.pos = Some([x, y, z + 0.05]);
                Record::Body(b)
            }
            Add::Camera => {
                let mut c = new_camera(&m.unique("camera"));
                c.pos = Some([x, y, z + 0.5]);
                Record::Camera(c)
            }
            Add::Light => {
                let mut l = new_light(&m.unique("light"));
                l.pos = Some([x, y, z + 1.0]);
                Record::Light(l)
            }
            Add::Region => {
                let mut r = new_region(&m.unique("region"));
                r.pos = Some([x, y, z]);
                Record::Region(r)
            }
        };
        self.apply(&Command::Add(record), None);
    }

    /// The selected entity's fields. Name and parent apply as they change (a rename, a
    /// re-parent); every other field is one `Set` once the person lets go.
    #[allow(clippy::many_single_char_names)] // one short name per record kind
    pub(crate) fn inspector(&mut self, ui: &mut egui::Ui, lang: Lang) {
        if let Some((r, _)) = &self.refusal {
            let args: Vec<&str> = r.args.iter().map(String::as_str).collect();
            let why = fill(lang, r.key, &args);
            let line = fill(lang, "author.refused", &[&r.field, &why]);
            ui.label(RichText::new(line).color(RED));
        }
        let Some(e) = self.model.selection().cloned() else {
            ui.weak(t(lang, "author.pick"));
            return;
        };
        let Some(current) = self.model.record(&e) else {
            return;
        };
        let rev = self.model.revision();
        if self
            .draft
            .as_ref()
            .is_none_or(|d| d.entity != e || d.revision != rev)
        {
            let mut record = current.clone();
            let quat = record.pose_mut().and_then(|(_, q)| *q).unwrap_or(IDENTITY);
            let rpy = euler::deg_from_quat(quat);
            self.draft = Some(Draft {
                name: e.label(self.model.doc()),
                entity: e.clone(),
                revision: rev,
                record,
                rpy,
                rotated: false,
            });
        }
        let doc = self.model.doc().clone();
        let parents = self.model.parents(&e);
        let refused = self.refusal.as_ref().map(|(r, _)| r.field.clone());
        let d = self.draft.as_mut().expect("set above");
        let prefix = e.field(&doc);
        let mut cmd = None;
        ui.heading(format!("{}  ", d.name))
            .on_hover_text(t(lang, row_kind(&e).key()));
        egui::Grid::new("author-fields")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                let f = Fields {
                    lang,
                    prefix: &prefix,
                    refused: refused.as_deref(),
                };
                if !matches!(e, Entity::Physics) {
                    f.label(ui, "author.name", "name");
                    let r = ui.text_edit_singleline(&mut d.name);
                    if r.lost_focus() && d.name != e.label(&doc) {
                        cmd = Some(Command::Rename(e.clone(), d.name.clone()));
                    }
                    ui.end_row();
                }
                if matches!(e, Entity::Body(_) | Entity::Camera(_) | Entity::Region(_)) {
                    let parent = match &d.record {
                        Record::Body(b) => b.parent.clone(),
                        Record::Camera(c) => c.parent.clone(),
                        Record::Region(r) => r.parent.clone(),
                        _ => None,
                    };
                    f.label(ui, "author.parent", "parent");
                    let world = t(lang, "author.world").to_owned();
                    let mut pick = parent.clone();
                    egui::ComboBox::from_id_salt("author-parent")
                        .selected_text(pick.clone().unwrap_or_else(|| world.clone()))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut pick, None, world);
                            for p in &parents {
                                ui.selectable_value(&mut pick, Some(p.clone()), p);
                            }
                        });
                    if pick != parent {
                        cmd = Some(Command::Reparent(e.clone(), pick));
                    }
                    ui.end_row();
                }
                if let Some((pos, _)) = d.record.pose_mut() {
                    f.label(ui, "author.position", "pos");
                    opt3(ui, pos, [0.0; 3], 0.005, " m");
                    ui.end_row();
                    f.label(ui, "author.rotation", "quat")
                        .on_hover_text(t(lang, "author.rotation.hint"));
                    ui.horizontal(|ui| {
                        for a in &mut d.rpy {
                            d.rotated |= num(ui, a, 1.0, "\u{b0}").changed();
                        }
                    });
                    ui.end_row();
                }
                // G8: the 🎲 toggles go in a third column of these rows.
                match &mut d.record {
                    Record::Physics(p) => {
                        f.label(ui, "author.timestep", "timestep");
                        opt1(ui, &mut p.timestep, 0.002, 0.0001, " s");
                        ui.end_row();
                        f.label(ui, "author.gravity", "gravity");
                        opt3(ui, &mut p.gravity, [0.0, 0.0, -9.81], 0.01, "");
                        ui.end_row();
                    }
                    Record::Include(i) => {
                        f.label(ui, "author.source", "source");
                        ui.weak(&i.source);
                        ui.end_row();
                    }
                    Record::Body(b) => {
                        joint(ui, &f, &mut b.joint);
                        let name = b.name.clone();
                        for (index, g) in b.geoms.iter_mut().enumerate() {
                            let ge = Entity::Geom {
                                body: name.clone(),
                                index,
                            };
                            let prefix = ge.field(&doc);
                            let f = Fields {
                                prefix: &prefix,
                                ..f
                            };
                            ui.strong(fill(lang, "author.part", &[&ge.label(&doc)]));
                            ui.end_row();
                            geom(ui, &f, g, &doc, index);
                        }
                    }
                    Record::Scenery(g) | Record::Geom(g) => geom(ui, &f, g, &doc, 0),
                    Record::Camera(c) => {
                        f.label(ui, "author.fovy", "fovy");
                        opt1(ui, &mut c.fovy, inspect::FOVY, 0.5, "\u{b0}");
                        ui.end_row();
                    }
                    Record::Light(l) => {
                        f.label(ui, "author.size", "size");
                        ui.horizontal(|ui| {
                            for v in &mut l.size {
                                whole(ui, v);
                            }
                        });
                        ui.end_row();
                        f.label(ui, "author.colour", "rgb");
                        let mut c = l.rgb.unwrap_or([1.0; 3]).map(|v| v as f32);
                        if ui.color_edit_button_rgb(&mut c).changed() {
                            l.rgb = Some(c.map(round));
                        }
                        ui.end_row();
                        f.label(ui, "author.brightness", "intensity");
                        opt1(ui, &mut l.intensity, 1.0, 0.05, "");
                        ui.end_row();
                    }
                    Record::Region(r) => {
                        f.label(ui, "author.size", "size");
                        let mut s = r.size.unwrap_or([0.005; 3]);
                        let changed = ui.horizontal(|ui| {
                            let mut changed = false;
                            for v in &mut s {
                                changed |= whole(ui, v);
                            }
                            changed
                        });
                        if changed.inner {
                            r.size = Some(s);
                        }
                        ui.end_row();
                    }
                }
            });
        ui.add_space(8.0);
        ui.weak(t(lang, "author.uses_template"));

        // Once the person lets go: the fields as one command, unless that value was refused.
        let mut record = d.record.clone();
        if d.rotated {
            if let Some((_, q)) = record.pose_mut() {
                *q = Some(euler::quat_from_deg(d.rpy));
            }
        }
        let settled = !ui.input(|i| i.pointer.any_down()) && !ui.ctx().wants_keyboard_input();
        let tried = (self.refusal.as_ref()).is_some_and(|(_, r)| r.as_ref() == Some(&record));
        if let Some(cmd) = cmd {
            self.apply(&cmd, None);
        } else if settled && record != current && !tried {
            self.apply(&Command::Set(e, record.clone()), Some(record));
        } else if record == current && self.refusal.as_ref().is_some_and(|(_, r)| r.is_some()) {
            // The refused value was put back: nothing is wrong any more.
            self.refusal = None;
        }
    }
}

fn row_kind(e: &Entity) -> RowKind {
    match e {
        Entity::Physics => RowKind::Physics,
        Entity::Include(_) => RowKind::Include,
        Entity::Scenery(_) => RowKind::Scenery,
        Entity::Body(_) => RowKind::Body,
        Entity::Geom { .. } => RowKind::Geom,
        Entity::Camera(_) => RowKind::Camera,
        Entity::Light(_) => RowKind::Light,
        Entity::Region(_) => RowKind::Region,
    }
}

/// One inspector's words and the field a refusal named.
#[derive(Clone, Copy)]
struct Fields<'a> {
    lang: Lang,
    /// The entity's field path (`body[cube]`).
    prefix: &'a str,
    refused: Option<&'a str>,
}

impl Fields<'_> {
    /// The row's label, red when the last refusal named this field.
    fn label(&self, ui: &mut egui::Ui, key: &'static str, leaf: &str) -> egui::Response {
        let path = format!("{}.{leaf}", self.prefix);
        let hit = self
            .refused
            .is_some_and(|r| r == path || r.starts_with(&format!("{path}.")));
        let text = RichText::new(t(self.lang, key));
        ui.label(if hit { text.color(RED).strong() } else { text })
    }
}

/// A colour channel as the document keeps it: four decimals, not the picker's `f32` noise.
fn round(v: f32) -> f64 {
    (f64::from(v) * 1e4).round() / 1e4
}

fn num(ui: &mut egui::Ui, v: &mut f64, speed: f64, suffix: &str) -> egui::Response {
    let w = egui::DragValue::new(v)
        .speed(speed)
        .suffix(suffix)
        .max_decimals(4);
    ui.add(w.update_while_editing(false))
}

/// A half-extent shown whole (exact in binary).
fn whole(ui: &mut egui::Ui, half: &mut f64) -> bool {
    let mut w = 2.0 * *half;
    let changed = num(ui, &mut w, 0.002, " m").changed();
    if changed {
        *half = w / 2.0;
    }
    changed
}

/// An optional value shown at its default and written only when changed.
fn opt1(ui: &mut egui::Ui, v: &mut Option<f64>, default: f64, speed: f64, suffix: &str) {
    let mut x = v.unwrap_or(default);
    if num(ui, &mut x, speed, suffix).changed() {
        *v = Some(x);
    }
}

fn opt3(ui: &mut egui::Ui, v: &mut Option<[f64; 3]>, default: [f64; 3], speed: f64, suffix: &str) {
    let mut x = v.unwrap_or(default);
    // Every widget drawn, then whether any changed (no short circuit).
    let changed = ui.horizontal(|ui| {
        let mut changed = false;
        for a in &mut x {
            changed |= num(ui, a, speed, suffix).changed();
        }
        changed
    });
    if changed.inner {
        *v = Some(x);
    }
}

const JOINTS: [(JointKindDoc, &str); 5] = [
    (JointKindDoc::Fixed, "author.joint.fixed"),
    (JointKindDoc::Free, "author.joint.free"),
    (JointKindDoc::Hinge, "author.joint.hinge"),
    (JointKindDoc::Slide, "author.joint.slide"),
    (JointKindDoc::Ball, "author.joint.ball"),
];

/// A body's joint: its kind, and for a hinge or slide its axis and limits (degrees or metres).
fn joint(ui: &mut egui::Ui, f: &Fields<'_>, j: &mut Option<JointDoc>) {
    let lang = f.lang;
    let kind = j.as_ref().map_or(JointKindDoc::Fixed, |j| j.kind);
    let mut pick = kind;
    f.label(ui, "author.joint", "joint");
    let word = |k: JointKindDoc| {
        JOINTS
            .iter()
            .find(|(x, _)| *x == k)
            .map_or("", |(_, w)| t(lang, w))
    };
    egui::ComboBox::from_id_salt("author-joint")
        .selected_text(word(kind))
        .show_ui(ui, |ui| {
            for (k, key) in JOINTS {
                ui.selectable_value(&mut pick, k, t(lang, key));
            }
        });
    ui.end_row();
    if pick != kind {
        match j {
            Some(j) => j.kind = pick,
            None => *j = Some(new_joint(pick)),
        }
    }
    let Some(j) = j.as_mut().filter(|j| {
        matches!(
            j.kind,
            JointKindDoc::Hinge | JointKindDoc::Slide | JointKindDoc::Ball
        )
    }) else {
        return;
    };
    if j.kind != JointKindDoc::Ball {
        f.label(ui, "author.axis", "joint.axis");
        opt3(ui, &mut j.axis, [0.0, 0.0, 1.0], 0.01, "");
        ui.end_row();
    }
    let metres = j.kind == JointKindDoc::Slide;
    let (key, suffix, speed) = if metres {
        ("author.range.m", " m", 0.005)
    } else {
        ("author.range.deg", "\u{b0}", 1.0)
    };
    f.label(ui, key, "joint.range");
    ui.horizontal(|ui| {
        let mut limited = j.range.is_some();
        if ui
            .checkbox(&mut limited, t(lang, "author.limited"))
            .changed()
        {
            j.range = limited.then_some([-1.0, 1.0]);
        }
        if let Some(range) = &mut j.range {
            for v in range {
                let mut shown = if metres { *v } else { *v * RAD_TO_DEG };
                if num(ui, &mut shown, speed, suffix).changed() {
                    *v = if metres { shown } else { shown * DEG_TO_RAD };
                }
            }
        }
    });
    ui.end_row();
}

/// A geom's shape and sizes, mass or density, friction, colour and material.
fn geom(
    ui: &mut egui::Ui,
    f: &Fields<'_>,
    g: &mut GeomDoc,
    doc: &es_assets::esscene::EsScene,
    salt: usize,
) {
    let lang = f.lang;
    let kind = ShapeKind::of(&g.shape);
    f.label(ui, "author.shape", "shape");
    let mut pick = kind;
    egui::ComboBox::from_id_salt(("author-shape", salt))
        .selected_text(t(lang, kind.key()))
        .show_ui(ui, |ui| {
            for k in ShapeKind::PICK {
                ui.selectable_value(&mut pick, k, t(lang, k.key()));
            }
        });
    if pick != kind {
        g.shape = inspect::reshape(&g.shape, pick);
    }
    ui.end_row();
    let dims = inspect::dims(&g.shape);
    if !dims.is_empty() {
        f.label(ui, "author.size", "shape");
        let mut v: Vec<f64> = dims.iter().map(|d| d.1).collect();
        let changed = ui
            .horizontal(|ui| {
                let mut changed = false;
                for ((key, _), x) in dims.iter().zip(&mut v) {
                    changed |= num(ui, x, 0.002, " m")
                        .on_hover_text(t(lang, key))
                        .changed();
                }
                changed
            })
            .inner;
        if changed {
            g.shape = inspect::with_dims(&g.shape, &v);
        }
        ui.end_row();
    }
    let mut given = g.mass.is_some();
    f.label(
        ui,
        if given {
            "author.mass"
        } else {
            "author.density"
        },
        if given { "mass" } else { "density" },
    );
    ui.horizontal(|ui| {
        if given {
            let mut m = g.mass.unwrap_or_default();
            if num(ui, &mut m, 0.005, " kg").changed() {
                g.mass = Some(m);
            }
        } else {
            opt1(ui, &mut g.density, inspect::DENSITY, 10.0, "");
        }
        if ui
            .checkbox(&mut given, t(lang, "author.mass.given"))
            .changed()
        {
            g.mass = given.then_some(0.1);
        }
    });
    ui.end_row();
    f.label(ui, "author.friction", "friction")
        .on_hover_text(t(lang, "author.friction.hint"));
    opt3(ui, &mut g.friction, inspect::FRICTION, 0.01, "");
    ui.end_row();
    f.label(ui, "author.colour", "rgba");
    let mut c = g.rgba.unwrap_or(inspect::RGBA).map(|v| v as f32);
    if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
        g.rgba = Some(c.map(round));
    }
    ui.end_row();
    f.label(ui, "author.material", "material");
    let none = t(lang, "author.material.none").to_owned();
    egui::ComboBox::from_id_salt(("author-material", salt))
        .selected_text(g.material.clone().unwrap_or_else(|| none.clone()))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut g.material, None, none);
            for m in &doc.materials {
                let label = match &m.texture {
                    Some(tex) => format!("{} ({tex})", m.name),
                    None => m.name.clone(),
                };
                ui.selectable_value(&mut g.material, Some(m.name.clone()), label);
            }
        });
    ui.end_row();
}
