//! ①'s hierarchy (packet M17/G5): the Add menu and the search, then one row per entity with its
//! fold arrow, eye, name and kind, and its right-click menu.

use eframe::egui;
use egui::RichText;
use es_editor_scene::{tree, Camera, Command, Entity};

use super::{fold_key, Author};
use crate::model::i18n::{t, Lang};

impl Author {
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
}
