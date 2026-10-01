//! ①'s toolbar (packets M17/G5, G6 and G9): the handles' tool, snapping and framing; undo,
//! redo and save (and generate) with what the last save did; "save as template"; and their keys.

use eframe::egui;
use es_editor_scene::{Camera, Command, Regen, Tool};

use super::Author;
use crate::model::i18n::{fill, t, Lang};

impl Author {
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
}
