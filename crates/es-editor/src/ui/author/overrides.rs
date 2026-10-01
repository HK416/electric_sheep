//! An include's overrides under its fields in the inspector (packet M17/G7): each joint, motor
//! and shape the included file declares, by the file's own names, showing the override or else the
//! file's value, with ↺ back to the file's. Drawing only: what the file says and the pruned set an
//! edit leaves are `es_editor_scene::overrides`'s; the inspector hands the include over as one
//! command once the person lets go, as for every other field.

use eframe::egui;
use es_assets::esscene::{ActuatorSet, EsScene, GeomSet, IncludeSet, JointSet};
use es_editor_scene::inspect;
use es_editor_scene::overrides::{prune, Brought};
use es_math::units::RAD_TO_DEG;

use super::{num, round};
use crate::model::i18n::{t, Lang};

/// The ↺ that drops an override, or an empty cell where there is none.
fn reset<T>(ui: &mut egui::Ui, lang: Lang, o: &mut Option<T>) {
    if o.is_none() {
        ui.label("");
    } else if (ui.small_button("\u{21ba}"))
        .on_hover_text(t(lang, "author.import.reset.hint"))
        .clicked()
    {
        *o = None;
    }
    ui.end_row();
}

/// One number: the override `o`, else the file's; a field the file lacks is not shown.
fn one(ui: &mut egui::Ui, lang: Lang, key: &'static str, o: &mut Option<f64>, file: Option<f64>) {
    let Some(mut v) = o.or(file) else { return };
    ui.label(t(lang, key));
    if num(ui, &mut v, 0.01, "").changed() {
        *o = Some(v);
    }
    reset(ui, lang, o);
}

/// Two numbers (a range), shown times `k` (degrees for radians).
fn pair(
    ui: &mut egui::Ui,
    lang: Lang,
    key: &'static str,
    (o, file): (&mut Option<[f64; 2]>, Option<[f64; 2]>),
    (k, speed, unit): (f64, f64, &str),
) {
    let Some(base) = o.or(file) else { return };
    ui.label(t(lang, key));
    let mut v = base.map(|x| x * k);
    let changed = ui.horizontal(|ui| {
        let a = num(ui, &mut v[0], speed, unit).changed();
        num(ui, &mut v[1], speed, unit).changed() | a
    });
    if changed.inner {
        *o = Some(v.map(|x| x / k));
    }
    reset(ui, lang, o);
}

/// A collapsing row per named thing, marked when it overrides anything.
fn row(ui: &mut egui::Ui, name: &str, changed: bool, body: impl FnOnce(&mut egui::Ui)) {
    let title = if changed {
        format!("{name} \u{25cf}")
    } else {
        name.to_owned()
    };
    egui::CollapsingHeader::new(title)
        .id_salt(("override", name))
        .show(ui, |ui| {
            egui::Grid::new(("override-grid", name))
                .num_columns(3)
                .show(ui, body);
        });
}

/// The overrides of an include that brings `b`, edited in `set`; `doc`'s materials may be named.
pub(super) fn show(
    ui: &mut egui::Ui,
    lang: Lang,
    set: &mut Option<IncludeSet>,
    b: &Brought,
    doc: &EsScene,
) {
    let mut s = set.clone().unwrap_or_default();
    ui.add_space(8.0);
    ui.strong(t(lang, "author.import.overrides"))
        .on_hover_text(t(lang, "author.import.overrides.hint"));
    ui.collapsing(t(lang, "author.import.joints"), |ui| {
        for (name, metres, file) in &b.joints {
            let o = s.joint.entry(name.clone()).or_default();
            row(ui, name, *o != JointSet::default(), |ui| {
                let (key, k, speed, unit) = if *metres {
                    ("author.range.m", 1.0, 0.005, " m")
                } else {
                    ("author.range.deg", RAD_TO_DEG, 1.0, "\u{b0}")
                };
                pair(ui, lang, key, (&mut o.range, file.range), (k, speed, unit));
                one(
                    ui,
                    lang,
                    "author.import.damping",
                    &mut o.damping,
                    file.damping,
                );
                one(
                    ui,
                    lang,
                    "author.import.armature",
                    &mut o.armature,
                    file.armature,
                );
                one(
                    ui,
                    lang,
                    "author.import.stiffness",
                    &mut o.stiffness,
                    file.stiffness,
                );
                let loss = (&mut o.frictionloss, file.frictionloss);
                one(ui, lang, "author.import.frictionloss", loss.0, loss.1);
            });
        }
    });
    ui.collapsing(t(lang, "author.import.actuators"), |ui| {
        for (name, file) in &b.actuators {
            let o = s.actuator.entry(name.clone()).or_default();
            row(ui, name, *o != ActuatorSet::default(), |ui| {
                one(ui, lang, "author.import.kp", &mut o.kp, file.kp);
                one(ui, lang, "author.import.kv", &mut o.kv, file.kv);
                let plain = (1.0, 0.01, "");
                let ctrl = (&mut o.ctrlrange, file.ctrlrange);
                pair(ui, lang, "author.import.ctrlrange", ctrl, plain);
                let force = (&mut o.forcerange, file.forcerange);
                pair(ui, lang, "author.import.forcerange", force, plain);
            });
        }
    });
    let materials: Vec<&String> = (b.materials.iter())
        .chain(doc.materials.iter().map(|m| &m.name))
        .collect();
    ui.collapsing(t(lang, "author.import.geoms"), |ui| {
        for (name, file) in &b.geoms {
            let o = s.geom.entry(name.clone()).or_default();
            row(ui, name, *o != GeomSet::default(), |ui| {
                ui.label(t(lang, "author.colour"));
                let rgba = o.rgba.or(file.rgba).unwrap_or(inspect::RGBA);
                let mut c = rgba.map(|v| v as f32);
                if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
                    o.rgba = Some(c.map(round));
                }
                reset(ui, lang, &mut o.rgba);
                ui.label(t(lang, "author.material"));
                let shown = o.material.clone().or_else(|| file.material.clone());
                let none = t(lang, "author.material.none");
                let mut pick = shown.clone();
                egui::ComboBox::from_id_salt(("override-material", name))
                    .selected_text(shown.as_deref().unwrap_or(none))
                    .show_ui(ui, |ui| {
                        for m in &materials {
                            ui.selectable_value(&mut pick, Some((*m).clone()), m.as_str());
                        }
                    });
                if pick != shown {
                    o.material = pick;
                }
                reset(ui, lang, &mut o.material);
            });
        }
    });
    *set = prune(Some(s));
}
