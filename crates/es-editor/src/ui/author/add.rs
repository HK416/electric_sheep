//! ①'s Add menu and picture import (packet M17/G7). Drawing only: what each item makes, where it
//! goes, what a file becomes and why one is refused are `es_editor_scene::add`'s, under test.

use std::sync::{Arc, OnceLock};

use eframe::egui;
use es_editor_scene::add::library;
use es_editor_scene::inspect::ShapeKind;
use es_editor_scene::overrides::Brought;
use es_editor_scene::{Camera, Entity, Item, Robot, RowKind};

use super::Author;
use crate::model::dialogs::{pick_file, AVAILABLE};
use crate::model::i18n::{fill, t, Lang};
use crate::model::template::templates_root;

const SHAPES: [ShapeKind; 4] = [
    ShapeKind::Box,
    ShapeKind::Sphere,
    ShapeKind::Cylinder,
    ShapeKind::Capsule,
];

/// The robot library, read once; empty outside a checkout.
fn robots() -> &'static [Robot] {
    static LIBRARY: OnceLock<Vec<Robot>> = OnceLock::new();
    LIBRARY.get_or_init(|| (templates_root().and_then(|r| library(&r).ok())).unwrap_or_default())
}

/// The Add menu and the overrides between frames.
#[derive(Default)]
pub(super) struct State {
    /// A script asked for the menu open (`--edit-demo add-menu`).
    open: bool,
    /// What the selected include brings, by its name and file, read once per include.
    brought: Option<(String, String, Option<Arc<Brought>>)>,
}

/// A button that opens the OS's file dialog: disabled, saying why, in a build without one.
fn file_button(ui: &mut egui::Ui, lang: Lang, key: &'static str) -> bool {
    ui.add_enabled(AVAILABLE, egui::Button::new(t(lang, key)))
        .on_disabled_hover_text(t(lang, "open.no_dialog.hint"))
        .clicked()
}

/// The material picker's last entry: a picture from a file.
pub(super) fn picture(ui: &mut egui::Ui, lang: Lang) -> bool {
    AVAILABLE
        && ui
            .selectable_label(false, t(lang, "author.import.image"))
            .clicked()
}

impl Author {
    /// Objects and fixed objects by shape, a mesh file, a robot of the library or from a file, a
    /// camera, a light and an area; each goes where `view` looks.
    pub(super) fn add_menu(&mut self, ui: &mut egui::Ui, lang: Lang, view: &Camera) {
        let mut chosen = None;
        let menu = ui.menu_button(t(lang, "author.add"), |ui| {
            for (key, fixed) in [("author.add.object", false), ("author.add.fixed", true)] {
                ui.label(t(lang, key));
                ui.horizontal(|ui| {
                    for k in SHAPES {
                        if ui.button(t(lang, k.key())).clicked() {
                            chosen = Some(if fixed {
                                Item::Fixed(k)
                            } else {
                                Item::Object(k)
                            });
                        }
                    }
                });
            }
            if file_button(ui, lang, "author.add.mesh") {
                chosen = pick_file(("STL, OBJ", &["stl", "obj"])).map(Item::Mesh);
            }
            ui.separator();
            ui.label(t(lang, "author.add.robot"));
            ui.horizontal_wrapped(|ui| {
                for r in robots() {
                    let word = r
                        .name
                        .as_deref()
                        .map_or(r.id.clone(), |k| fill(lang, k, &[]));
                    if ui.button(word).clicked() {
                        chosen = Some(Item::Robot(r.clone()));
                    }
                }
            });
            if file_button(ui, lang, "author.add.robot.file") {
                let kinds: &[&str] = &["xml", "urdf", "gltf", "glb"];
                chosen =
                    pick_file(("MJCF, URDF, glTF", kinds)).map(|p| Item::Robot(Robot::file(&p)));
            }
            ui.separator();
            ui.horizontal(|ui| {
                let others = [
                    (Item::Camera, "author.kind.camera"),
                    (Item::Light, "author.kind.light"),
                    (Item::Region, "author.kind.region"),
                ];
                for (item, key) in others {
                    if ui.button(t(lang, key)).clicked() {
                        chosen = Some(item);
                    }
                }
            });
            ui.weak(t(lang, "author.add.where"));
            if chosen.is_some() {
                ui.close();
            }
        });
        if std::mem::take(&mut self.add.open) {
            egui::Popup::open_id(ui.ctx(), egui::Popup::default_response_id(&menu.response));
        }
        if let Some(item) = chosen {
            self.add_item(&item, view);
        }
    }

    /// `item` added where `view` looks; a refusal is shown as every other one is. A new include
    /// starts folded, as the ones the project opened with do.
    fn add_item(&mut self, item: &Item, view: &Camera) {
        self.refusal = self
            .model
            .add(item, view, self.snap)
            .err()
            .map(|r| (r, None));
        if let (None, Some(Entity::Include(n))) = (&self.refusal, self.model.selection()) {
            self.folded.insert(format!("{:?}/{n}", RowKind::Include));
        }
    }

    /// `--edit-demo add-menu|add-box|add-robot` (packet M17/G7's captures): the menu open; a box
    /// where `view`'s eye sees a free spot of the SO-101 table, (0.3, 0.15); the library's first
    /// robot where `view` looks. `false` for any other stage.
    pub(super) fn add_demo(&mut self, stage: &str, view: &Camera) -> bool {
        let table = Camera {
            look_at: [0.3, 0.15, 0.0],
            ..*view
        };
        match stage {
            "add-menu" => self.add.open = true,
            "add-box" => self.add_item(&Item::Object(ShapeKind::Box), &table),
            "add-robot" => {
                if let Some(r) = robots().first() {
                    self.add_item(&Item::Robot(r.clone()), view);
                }
            }
            _ => return false,
        }
        true
    }

    /// A picture from a file as the material of geom `index` of `e`, one command.
    pub(super) fn picture(&mut self, e: &Entity, index: usize) {
        if let Some(path) = pick_file(("PNG", &["png"])) {
            self.refusal = self.model.texture(e, index, &path).err().map(|r| (r, None));
        }
    }

    /// What the include `e` brings, read when another include is selected or its file changes.
    pub(super) fn brought(&mut self, e: &Entity) -> Option<Arc<Brought>> {
        let Entity::Include(name) = e else {
            return None;
        };
        let source = (self.model.doc().includes.iter())
            .find(|i| i.name == *name)
            .map(|i| i.source.clone())?;
        let cached = self.add.brought.as_ref();
        if cached.is_none_or(|(n, s, _)| *n != *name || *s != source) {
            let read = self.model.brought(name).map(Arc::new);
            self.add.brought = Some((name.clone(), source, read));
        }
        self.add.brought.as_ref().and_then(|(_, _, b)| b.clone())
    }
}
