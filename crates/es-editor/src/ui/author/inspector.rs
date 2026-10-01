//! ①'s inspector (packet M17/G5): the selected entity's fields, handed over as one command once
//! the person lets go, and the widgets they are drawn with: a body's joint, and a geom's shape,
//! size, mass, friction, colour and material.

use eframe::egui;
use egui::{Color32, RichText};
use es_assets::esscene::{GeomDoc, JointDoc, JointKindDoc};
use es_editor_scene::inspect::{self, ShapeKind};
use es_editor_scene::{euler, new_joint, Command, Entity, Record, RowKind};
use es_math::units::{DEG_TO_RAD, RAD_TO_DEG};

use super::{add, overrides, Author, Draft};
use crate::model::i18n::{fill, t, Lang};

const RED: Color32 = Color32::from_rgb(220, 80, 70);
const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

impl Author {
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
pub(super) fn round(v: f32) -> f64 {
    (f64::from(v) * 1e4).round() / 1e4
}

pub(super) fn num(ui: &mut egui::Ui, v: &mut f64, speed: f64, suffix: &str) -> egui::Response {
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
    // A mesh's size is its file's times its scale (packet M17/R3): one number while uniform.
    if let Some(mut v) = inspect::scale(&g.shape) {
        f.label(ui, "author.scale", "shape")
            .on_hover_text(t(lang, "author.scale.hint"));
        let speed = (v[0].abs() * 0.01).max(1e-6);
        let changed = ui
            .horizontal(|ui| {
                if v.iter().all(|x| x.to_bits() == v[0].to_bits()) {
                    let changed = num(ui, &mut v[0], speed, "").changed();
                    v = [v[0]; 3];
                    return changed;
                }
                let mut changed = false;
                for x in &mut v {
                    changed |= num(ui, x, speed, "").changed();
                }
                changed
            })
            .inner;
        if changed {
            g.shape = inspect::with_scale(&g.shape, v);
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
