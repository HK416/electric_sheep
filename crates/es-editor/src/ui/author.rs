//! ① of an editable project (packet M17/G5, `docs/design/scene-authoring.md` section 5): the
//! hierarchy on the left, the viewport's undo / redo / save row, and the inspector on the right,
//! over [`es_editor_scene::SceneModel`]. Over the viewport (packet M17/G6): the selection's tint,
//! its move / turn / size handles, a click to pick, and the policy camera's view in the corner.
//! ①'s Add menu and picture import, cameras and regions drawn as lines, and an include's
//! overrides in the inspector are packet M17/G7's, in this module's children.
//!
//! Drawing only. Which rows there are, what a search keeps, what is hidden, which commands exist,
//! whether one is refused and why, how a size or an angle is shown, what a click selects, where a
//! handle is and what its drag writes — all the scene model's, under test. What is decided here
//! is when a value is handed over: once the person lets go (no pointer button held, no text being
//! typed), as one command, so a drag — of a field or of a handle — is one undo step; a refused
//! value stays in its field with the reason under it and is not tried again until it changes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;
use egui::{Color32, Pos2, Rect, RichText, Stroke, Vec2};
use es_assets::esscene::{GeomDoc, JointDoc, JointKindDoc};
use es_editor_scene::inspect::{self, ShapeKind};
use es_editor_scene::policy;
use es_editor_scene::view::{project, ray};
use es_editor_scene::{
    euler, new_joint, tree, BackendKind, Camera, Command, Drag, Entity, Gizmo, PolicyCamera, Ray,
    Record, Refusal, Regen, RowKind, SceneModel, Step, Tool,
};
use es_math::units::{DEG_TO_RAD, RAD_TO_DEG};
use es_math::Vec3;

use crate::model::i18n::{fill, t, Lang};
use crate::model::scene_view::ScenePreview;
use crate::ui::corner::{self, Corner};

mod add;
mod markers;
mod overrides;

const RED: Color32 = Color32::from_rgb(220, 80, 70);
const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
/// The X, Y and Z handles, and the one held or under the pointer.
const AXES: [Color32; 3] = [
    Color32::from_rgb(230, 70, 60),
    Color32::from_rgb(90, 200, 80),
    Color32::from_rgb(70, 130, 240),
];
const HOT: Color32 = Color32::from_rgb(250, 210, 60);
/// The selection's tint, drawn over everything (an x-ray, not a depth-tested outline).
const TINT: Color32 = Color32::from_rgba_premultiplied(100, 64, 12, 96);
/// How near a handle the pointer must be to take it, in points.
const REACH: f64 = 8.0;

/// A handle held: the drag, the ray the pointer is on now, and whether a script holds it
/// (`--edit-demo drag`) rather than the pointer.
struct Held {
    drag: Drag,
    now: Ray,
    scripted: bool,
}

/// The selection's triangles for its tint, and the selection and revision they are of.
type Tint = (Option<Entity>, usize, Vec<[[f32; 3]; 3]>);

/// The picture's pixels on the canvas: the camera's own size, stretched over `rect`.
#[derive(Clone, Copy)]
struct Screen {
    rect: Rect,
    sx: f64,
    sy: f64,
}

impl Screen {
    fn new(rect: Rect, camera: &Camera) -> Self {
        Self {
            rect,
            sx: f64::from(camera.width) / f64::from(rect.width().max(1.0)),
            sy: f64::from(camera.height) / f64::from(rect.height().max(1.0)),
        }
    }

    fn px(self, p: Pos2) -> [f64; 2] {
        [
            f64::from(p.x - self.rect.min.x) * self.sx,
            f64::from(p.y - self.rect.min.y) * self.sy,
        ]
    }

    fn pt(self, q: [f64; 2]) -> Pos2 {
        let (x, y) = (q[0] / self.sx, q[1] / self.sy);
        Pos2::new(self.rect.min.x + x as f32, self.rect.min.y + y as f32)
    }
}

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
    preview: Option<(usize, Result<Arc<ScenePreview>, String>)>,
    search: String,
    folded: BTreeSet<String>,
    renaming: Option<(Entity, String)>,
    draft: Option<Draft>,
    /// The last refusal and the record it refused.
    refusal: Option<(Refusal, Option<Record>)>,
    saved: Option<Result<(), String>>,
    tool: Tool,
    snap: bool,
    held: Option<Held>,
    tint: Option<Tint>,
    corner: Corner,
    /// The template's bundle (Task IR, Observation IR): the corner's cameras while the project
    /// has no task specification of its own (it trains on the template's documents until G9).
    pub(crate) bundle: Option<[PathBuf; 2]>,
    /// The Add menu's and the overrides' state (packet M17/G7).
    add: add::State,
    /// The task as sentences (packet M17/G8).
    pub(crate) task: crate::ui::sentence::Task,
    /// "Save as template": the name being typed while its dialog is open, and what the last one
    /// did — the folder it wrote, or why not (packet M17/G9).
    pub(crate) save_as: Option<String>,
    pub(crate) saved_as: Option<Result<PathBuf, String>>,
}

fn fold_key(row: &es_editor_scene::Row) -> String {
    format!("{:?}/{}", row.kind, row.name)
}

impl Author {
    /// The project's scene under edit; an include starts folded. `bundle` is the template's Task
    /// IR and Observation IR, for the corner's cameras while the project has no specification.
    pub(crate) fn open(
        root: &Path,
        backends: Vec<BackendKind>,
        bundle: Option<[PathBuf; 2]>,
    ) -> Result<Self, String> {
        let mut model = SceneModel::open(root, backends)?;
        // What `generated/` holds against the saved documents (packet M17/G9).
        model.refresh();
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
            tool: Tool::Move,
            snap: true,
            held: None,
            tint: None,
            corner: Corner::default(),
            bundle,
            add: add::State::default(),
            task: crate::ui::sentence::Task::default(),
            save_as: None,
            saved_as: None,
        })
    }

    fn apply(&mut self, cmd: &Command, record: Option<Record>) {
        self.refusal = self.model.apply(cmd).err().map(|r| (r, record));
    }

    /// The edited scene as the viewport draws it (what is hidden left out), rebuilt when the
    /// document or what is hidden changes.
    pub(crate) fn preview(&mut self) -> Result<Arc<ScenePreview>, String> {
        let rev = self.model.revision();
        if self.preview.as_ref().is_none_or(|(r, _)| *r != rev) {
            let drawn = self.model.drawn();
            let built = (self.model.scene_file())
                .and_then(|path| ScenePreview::from_scene(path, drawn, rev));
            self.preview = Some((rev, built.map(Arc::new)));
        }
        match &self.preview {
            Some((_, Ok(p))) => Ok(Arc::clone(p)),
            Some((_, Err(e))) => Err(e.clone()),
            None => unreachable!("built above"),
        }
    }

    /// `es-editor --edit-demo`: select the first free body (Shadow Hand's cube) and, from
    /// `edit` on, make it 9 cm and red; `undo` then undoes both (packet M17/G5's captures).
    /// `drag` holds its move handle 5.37 cm along X (snapped: 5 cm) as seen from `camera`, and
    /// `corner` lets that drag go — one command — and selects the front camera, which the corner
    /// then shows (packet M17/G6's captures).
    /// The Add menu's stages (packet M17/G7) are [`Self::add_demo`]'s.
    pub(crate) fn demo(&mut self, stage: &str, camera: &Camera) {
        if self.add_demo(stage, camera) || crate::ui::sentence::demo(self, stage) {
            return;
        }
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
        if stage == "drag" || stage == "corner" {
            let Some(g) = self.model.gizmo(&e, Tool::Move, camera) else {
                return;
            };
            let at = |s: f64| ray(camera, project(camera, g.origin + g.axes[0].scale(s))?);
            let (Some(start), Some(now)) = (at(0.6 * g.size), at(0.6 * g.size + 0.0537)) else {
                return;
            };
            if let Some(drag) = self.model.drag(&e, g.clone(), 0, start) {
                self.held = Some(Held {
                    drag,
                    now,
                    scripted: true,
                });
            }
            if stage == "corner" {
                self.let_go();
                self.model.select(Some(Entity::Camera("front".into())));
            }
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
    /// Then the handles (packet M17/G6): move, turn, size (W, E, R), snapping, and frame the
    /// selection in `camera` (F).
    pub(crate) fn toolbar(&mut self, ui: &mut egui::Ui, lang: Lang, camera: &mut Camera) {
        use egui::{Key, KeyboardShortcut as K, Modifiers as M};
        let typing = ui.ctx().wants_keyboard_input();
        let key = |k: K| !typing && ui.input_mut(|i| i.consume_shortcut(&k));
        let undo = key(K::new(M::COMMAND, Key::Z));
        let redo = key(K::new(M::COMMAND, Key::Y)) || key(K::new(M::COMMAND | M::SHIFT, Key::Z));
        let save = key(K::new(M::COMMAND, Key::S));
        let delete = key(K::new(M::NONE, Key::Delete));
        let tool = (Tool::ALL.into_iter().zip([Key::W, Key::E, Key::R]))
            .filter(|(_, k)| key(K::new(M::NONE, *k)))
            .map(|(tool, _)| tool)
            .next_back();
        let mut frame = key(K::new(M::NONE, Key::F));
        self.tool = tool.unwrap_or(self.tool);
        ui.horizontal_wrapped(|ui| {
            for tool in Tool::ALL {
                ui.selectable_value(&mut self.tool, tool, t(lang, tool.key()))
                    .on_hover_text(t(lang, "author.tool.hint"));
            }
            ui.checkbox(&mut self.snap, t(lang, "author.snap"))
                .on_hover_text(t(lang, "author.snap.hint"));
            let selected = self.model.selection().is_some();
            frame |= (ui.add_enabled(selected, egui::Button::new(t(lang, "author.frame"))))
                .on_hover_text(t(lang, "author.frame.hint"))
                .clicked();
            ui.weak(t(lang, "author.view_hint"));
        });
        if frame {
            if let Some(c) = (self.model.selection()).and_then(|e| self.model.framed(e, camera)) {
                *camera = c;
            }
        }
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
            // A task whose documents are not what its saved documents make is saved and
            // generated again by the same button (packet M17/G9).
            let stale = m.spec().is_some() && !matches!(m.generated(), Regen::Written(_));
            let (word, hint) = match (dirty, stale) {
                (_, true) => ("author.save_generate", "author.save_generate.hint"),
                (true, false) => ("edit.save", "author.unsaved"),
                (false, false) => ("edit.save", "author.save.hint"),
            };
            let word = if dirty {
                format!("{} \u{25cf}", t(lang, word))
            } else {
                t(lang, word).to_owned()
            };
            let clicked = ui
                .add_enabled(dirty || stale, egui::Button::new(word))
                .on_hover_text(t(lang, hint))
                .on_disabled_hover_text(t(lang, hint))
                .clicked();
            if (clicked || save) && (dirty || stale) {
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
                (_, Regen::Stale) => Some(("author.stale", vec![])),
            };
            if let Some((key, args)) = line {
                let args: Vec<&str> = args.iter().map(String::as_str).collect();
                ui.weak(fill(lang, key, &args));
            }
            let as_template = ui
                .button(t(lang, "author.save_template"))
                .on_hover_text(t(lang, "author.save_template.hint"));
            if as_template.clicked() {
                self.save_as = Some(String::new());
            }
            let note = match &self.saved_as {
                Some(Ok(dir)) => fill(
                    lang,
                    "author.save_template.done",
                    &[&dir.display().to_string()],
                ),
                Some(Err(why)) => fill(lang, "author.save_template.failed", &[why]),
                None => String::new(),
            };
            if !note.is_empty() {
                ui.weak(note);
            }
        });
        if delete {
            if let Some(e) = self.model.selection().cloned() {
                self.apply(&Command::Delete(e), None);
            }
        }
    }

    /// The hierarchy: the Add menu and the search, then the rows — fold arrow, eye, name and
    /// kind; a right click duplicates, deletes or renames. New things go where `view` looks.
    pub(crate) fn hierarchy(&mut self, ui: &mut egui::Ui, lang: Lang, view: &Camera) {
        ui.horizontal(|ui| {
            self.add_menu(ui, lang, view);
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
        let brought = self.brought(&e);
        // A geom whose material is to be a picture (packet M17/G7).
        let mut picture = None;
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
                            if geom(ui, &f, g, &doc, index) {
                                picture = Some(index);
                            }
                        }
                    }
                    Record::Scenery(g) | Record::Geom(g) => {
                        if geom(ui, &f, g, &doc, 0) {
                            picture = Some(0);
                        }
                    }
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
        if let (Record::Include(i), Some(b)) = (&mut d.record, &brought) {
            overrides::show(ui, lang, &mut i.set, b, &doc);
        }
        ui.add_space(8.0);
        ui.weak(t(lang, "author.uses_template"));
        if let Some(index) = picture {
            self.picture(&e, index);
            return;
        }

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

impl Author {
    /// The held handle let go: its one command, applied (snapped as the row says).
    fn let_go(&mut self) {
        if let Some(h) = self.held.take() {
            if let Some(cmd) = h.drag.command(&h.now, self.snap) {
                self.apply(&cmd, None);
            }
        }
    }

    /// Over the viewport (packet M17/G6): a click picks what is under it (empty space picks
    /// nothing), a press on a handle drags it and letting go applies the drag as one command; the
    /// selection is tinted, its handles drawn, and the policy's camera shown in the corner.
    /// `true` while a handle is held, so the drag does not also turn the view.
    pub(crate) fn overlay(
        &mut self,
        lang: Lang,
        response: &egui::Response,
        painter: &egui::Painter,
        camera: &Camera,
    ) -> bool {
        let screen = Screen::new(response.rect, camera);
        let reach = REACH * screen.sx;
        let (press, pointer, down) = response.ctx.input(|i| {
            let p = &i.pointer;
            (p.press_origin(), p.latest_pos(), p.primary_down())
        });
        let selected = self.model.selection().cloned();
        let gizmo = (selected.as_ref()).and_then(|e| self.model.gizmo(e, self.tool, camera));
        let on_handle = |g: &Gizmo, at: Pos2| g.handle(camera, screen.px(at), reach);
        match &mut self.held {
            Some(h) if h.scripted => {
                if response.clicked() {
                    self.let_go();
                }
            }
            Some(h) => {
                if let Some(now) = pointer.and_then(|p| ray(camera, screen.px(p))) {
                    h.now = now;
                }
                if !down {
                    self.let_go();
                }
            }
            None if response.drag_started_by(egui::PointerButton::Primary) => {
                let grab = (gizmo.as_ref().zip(selected.as_ref()).zip(press))
                    .and_then(|((g, e), at)| Some((g, e, on_handle(g, at)?, at)));
                if let Some((g, e, axis, at)) = grab {
                    let start = ray(camera, screen.px(at));
                    let drag =
                        start.and_then(|s| Some((self.model.drag(e, g.clone(), axis, s)?, s)));
                    self.held = drag.map(|(drag, now)| Held {
                        drag,
                        now,
                        scripted: false,
                    });
                }
            }
            None if response.clicked() => {
                let at = response.interact_pointer_pos();
                let handle = (gizmo.as_ref().zip(at)).and_then(|(g, at)| on_handle(g, at));
                if let (Some(at), None) = (at, handle) {
                    // A camera's or region's lines first (packet M17/G7), then what is drawn.
                    let px = screen.px(at);
                    let hit = (self.model.marker_at(camera, px, reach))
                        .or_else(|| ray(camera, px).and_then(|r| self.model.hit(&r)));
                    self.model.select(hit);
                }
            }
            None => {}
        }
        let hover = pointer.filter(|p| response.rect.contains(*p));
        let hot = (gizmo.as_ref().zip(hover)).and_then(|(g, at)| on_handle(g, at));
        self.paint(painter, screen, camera, gizmo, hot);

        let scene = Arc::clone(self.model.scene());
        let (rev, selected) = (self.model.revision(), self.model.selection().cloned());
        let (model, bundle) = (&mut self.model, &self.bundle);
        let rect = response.rect;
        let cameras = || cameras(model, bundle.as_ref());
        (self.corner).paint(painter, rect, lang, &scene, rev, selected.as_ref(), cameras);
        if self.held.is_some() {
            response.ctx.request_repaint();
        }
        self.held.as_ref().is_some_and(|h| !h.scripted)
    }

    /// The selection's tint and handles; while a handle is held, both where the drag puts them,
    /// the held handle lit and how far it has gone beside it.
    fn paint(
        &mut self,
        painter: &egui::Painter,
        screen: Screen,
        camera: &Camera,
        gizmo: Option<Gizmo>,
        hot: Option<usize>,
    ) {
        let (rev, selected) = (self.model.revision(), self.model.selection().cloned());
        if (self.tint.as_ref()).is_none_or(|(e, r, _)| *e != selected || *r != rev) {
            let tris = (selected.as_ref()).map_or_else(Vec::new, |e| self.model.triangles(e));
            self.tint = Some((selected, rev, tris));
        }
        let step =
            (self.held.as_ref()).and_then(|h| Some((&h.drag, h.drag.step(&h.now, self.snap)?)));
        let at = |p: Vec3| match step {
            Some((drag, s)) => drag.moved(s, p),
            None => p,
        };
        let to = |p: Vec3| project(camera, at(p)).map(|q| screen.pt(q));
        self.paint_markers(painter, camera, |q| screen.pt(q));
        let mut mesh = egui::Mesh::default();
        for t in self.tint.iter().flat_map(|(_, _, tris)| tris) {
            let corners = t.map(|v| to(Vec3::new(v[0].into(), v[1].into(), v[2].into())));
            let [Some(a), Some(b), Some(c)] = corners else {
                continue;
            };
            let i = mesh.vertices.len() as u32;
            for p in [a, b, c] {
                mesh.colored_vertex(p, TINT);
            }
            mesh.add_triangle(i, i + 1, i + 2);
        }
        painter.add(mesh);

        let (shown, hot) = match (&self.held, step) {
            (Some(h), Some((drag, s))) => {
                let mut g = drag.gizmo.clone();
                g.origin = drag.moved(s, g.origin);
                (Some(g), Some(h.drag.axis))
            }
            (Some(h), None) => (Some(h.drag.gizmo.clone()), Some(h.drag.axis)),
            _ => (gizmo, hot),
        };
        let Some(g) = shown else {
            return;
        };
        for (k, line) in g.lines(camera) {
            let colour = if hot == Some(k) { HOT } else { AXES[k] };
            let points: Vec<Pos2> = line.into_iter().map(|q| screen.pt(q)).collect();
            // An arrow head on a move handle, a square on a size handle.
            if let (false, [.., before, end]) = (g.tool == Tool::Rotate, points.as_slice()) {
                let ahead = (*end - *before).normalized() * 9.0;
                let side = Vec2::new(-ahead.y, ahead.x) * 0.5;
                let head = if g.tool == Tool::Move {
                    vec![*end + ahead, *end + side, *end - side]
                } else {
                    let square = Rect::from_center_size(*end, Vec2::splat(9.0));
                    let [lt, rt] = [square.left_top(), square.right_top()];
                    vec![lt, rt, square.right_bottom(), square.left_bottom()]
                };
                painter.add(egui::Shape::convex_polygon(head, colour, Stroke::NONE));
            }
            let width: f32 = if hot == Some(k) { 4.0 } else { 3.0 };
            painter.line(points, Stroke::new(width, colour));
        }
        if let Some((_, s)) = step {
            let text = match s {
                Step::Move { from, to } => format!("{:+.1} cm", (to - from) * 100.0),
                Step::Rotate { angle } => format!("{:+.0}\u{b0}", angle * RAD_TO_DEG),
                Step::Scale { from, to } => {
                    format!("{:.1} \u{2192} {:.1} cm", from * 100.0, to * 100.0)
                }
            };
            if let Some(o) = project(camera, g.origin).map(|q| screen.pt(q)) {
                let at = o + Vec2::new(14.0, -14.0);
                corner::tag(painter, at, egui::Align2::LEFT_BOTTOM, text, HOT);
            }
        }
    }
}

/// The cameras the policy of `model`'s project sees: its specification's, else its template's
/// `bundle` (Task IR, Observation IR).
fn cameras(
    model: &mut SceneModel,
    bundle: Option<&[PathBuf; 2]>,
) -> Result<Vec<PolicyCamera>, String> {
    let own = model.policy_cameras()?;
    match bundle {
        Some([task, obs]) if own.is_empty() => policy::bundle(task, obs, model.scene()),
        _ => Ok(own),
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

/// A geom's shape and sizes, mass or density, friction, colour and material; `true` when its
/// material is to be a picture from a file (packet M17/G7).
fn geom(
    ui: &mut egui::Ui,
    f: &Fields<'_>,
    g: &mut GeomDoc,
    doc: &es_assets::esscene::EsScene,
    salt: usize,
) -> bool {
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
    let mut picture = false;
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
            picture = add::picture(ui, lang);
        });
    ui.end_row();
    picture
}
