//! The five tabs the editor had before the dock (packets M7/E1-E8), each drawn as one
//! Advanced pane (packet M12/Y10): the Design graph, the Results table with its Start panel and
//! replay, Live signals, what the policy sees, and Problems. Moved out of `app.rs` unchanged in
//! behaviour; the one difference is that the Design graph's Edit switch, Save, Undo and Redo
//! now sit in its own pane instead of the top bar, and its inspector inside that pane.
//!
//! Nothing here decides *what* is drawn (`docs/design/editor-shell.md` section 2): every row,
//! every order and every word is a model's. The **Graph** pane has two modes. Read-only
//! (spec 23.4 stage 1) draws all four IRs stacked and moves nothing. `Edit` (stage 2) drives
//! one [`EditSession`] over the Task IR: every gesture becomes one [`Edit`], and nothing else.
//! A drag writes a position into the `.eslayout` sidecar, never into an IR (spec 4.2 rule 7).

use std::collections::BTreeMap;
use std::path::Path;

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use es_eval::run_dir::{Bucket, Rgb8Image, RunDir};
use es_ir::graph::PortRef;
use es_ir::NodeId;
use es_render::raster::{Camera, Projected, Raster, BACKGROUND};

use crate::app::EditorApp;
use crate::model::dialogs;
use crate::model::edit::{self, Edit, EditSession};
use crate::model::graph_view::{CrossEdge, LayerView, LayeredGraph, NodeView};
use crate::model::i18n::{self, Lang};
use crate::model::image_view::ImagePair;
use crate::model::inspector::{Field, Inspector, Widget};
use crate::model::labels::{self, Browse};
use crate::model::launch::{Kind as LaunchKind, State as LaunchState};
use crate::model::live_run::cell_key;
use crate::model::replay_view::{self, ReplayView};
use crate::model::run_view;
use crate::model::scene_view::ScenePreview;
use crate::model::search::Search;
use crate::model::telemetry_view;
use crate::model::train_view::{Plot, Series};
use crate::model::viewport::{Mode, Shot, Viewport};

const NODE_W: f32 = 178.0;
const NODE_H: f32 = 40.0;
/// Click radius of a port, in graph units.
const PORT_R: f32 = 7.0;
/// Width of the Launch section's flag labels, so the text boxes line up. Wider than the
/// flags it replaced: a plain name is longer than `--jobs` and may be Korean (packet M7/E6).
const FLAG_LABEL: Vec2 = Vec2::new(210.0, 18.0);

/// What the pointer is doing between press and release.
#[derive(Clone, Debug)]
pub(crate) enum Drag {
    /// Moving a node. `origin` is where it was when the drag started, so the single
    /// `Edit::MoveNode` pushed on release has the correct state to undo to.
    Node {
        id: NodeId,
        origin: [f32; 2],
        grab: Vec2,
    },
    /// Pulling a wire out of an output port.
    Link {
        from: PortRef,
    },
    Pan,
}

impl EditorApp {
    /// The Graph toolbar's search box (packet M7/E3). [`Search`] decides what matches and in
    /// what order; Enter and `Next` ask it for the following hit, and the only thing decided
    /// here is where the canvas has to be panned to put that hit in the middle.
    fn search_bar(&mut self, ui: &mut egui::Ui) {
        let mut jump = false;
        ui.horizontal(|ui| {
            ui.label(self.t("find.label"));
            let hint = self.t("find.hint");
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.search.query)
                    .hint_text(hint)
                    .desired_width(200.0),
            );
            if response.changed() {
                let hits = match (&self.edit, &self.opened) {
                    (Some(session), _) => Search::filter_session(&self.search.query, session),
                    (None, Some(opened)) => Search::filter(&self.search.query, &opened.graph),
                    (None, None) => Vec::new(),
                };
                self.search.set_hits(hits);
            }
            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                jump = true;
                response.request_focus();
            }
            if ui.button(self.t("find.next")).clicked() {
                jump = true;
            }
            ui.weak(self.search.summary());
        });
        if jump {
            self.jump_to_hit(ui.available_size());
        }
    }

    /// Puts the next hit in the middle of a `canvas`-sized view and selects it. Centring is
    /// the whole of it: `pan` is what the painter adds to every position.
    fn jump_to_hit(&mut self, canvas: Vec2) {
        let Some((layer, node)) = self.search.advance() else {
            return;
        };
        if let Some(pos) = self.node_pos(layer, node) {
            self.pan = canvas * 0.5
                - (Vec2::new(pos[0], pos[1]) + Vec2::new(NODE_W, NODE_H) * 0.5) * self.zoom;
        }
        if self.edit.is_some() {
            self.selected = Some(node);
        }
    }

    /// Where a hit sits: in the session's `.eslayout` while editing, in the layered view's own
    /// layout otherwise.
    fn node_pos(&self, layer: usize, node: NodeId) -> Option<[f32; 2]> {
        if let Some(session) = &self.edit {
            return session.layout.positions.get(&node).copied();
        }
        self.opened
            .as_ref()?
            .graph
            .layers
            .get(layer)?
            .nodes
            .iter()
            .find(|n| n.id == node)?
            .layout
    }

    /// The parameter inspector (packet M7/E3): one widget per field of the selected node.
    /// [`Inspector`] decides which widget, what the text means and whether it becomes an
    /// [`Edit`]; this draws and forwards.
    fn inspector_panel(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.lang;
        // Rebuilt when the selection moves or the session does - an undo behind the panel's
        // back would otherwise leave stale text in the boxes. Between those, the widgets own
        // their text, so typing survives a repaint.
        let key = (
            self.selected,
            self.edit.as_ref().map_or(0, |s| s.history().len()),
        );
        if self.inspector_key != key {
            self.inspector_key = key;
            self.inspector = match (&self.edit, self.selected) {
                (Some(session), Some(node)) => Inspector::for_node(session, node),
                _ => None,
            };
        }
        let Some(inspector) = &mut self.inspector else {
            ui.heading(self.t("inspector.title"));
            ui.weak(self.t("inspector.empty"));
            return;
        };
        ui.heading(format!("{} #{}", inspector.kind, inspector.node.0));
        let mut commit: Option<(String, String)> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for field in inspector.fields_mut() {
                ui.label(&field.name);
                if draw_field(ui, field) {
                    commit = Some((field.name.clone(), field.text.clone()));
                }
                if let Some(error) = &field.error {
                    ui.colored_label(BAD, error);
                }
                ui.add_space(4.0);
            }
        });
        let Some((name, text)) = commit else { return };
        let Some(edit) = self.inspector.as_mut().and_then(|i| i.edit(&name, &text)) else {
            // The reason is the field's, not a sentence invented here (spec 28.10 rule 3).
            let why = self
                .inspector
                .as_ref()
                .and_then(|i| i.field(&name))
                .and_then(|f| f.error.clone())
                .unwrap_or_default();
            self.status = format!("{name}: {why}");
            return;
        };
        if let Some(session) = self.edit.as_mut() {
            self.status = match session.apply(edit) {
                Ok(()) => i18n::fill(
                    lang,
                    "status.edits",
                    &[&session.history().len().to_string()],
                ),
                Err(diags) => diags.first().map_or_else(
                    || i18n::fill(lang, "status.refused", &[]),
                    |d| i18n::fill(lang, "status.refused", &[&d.code.to_string(), &d.message]),
                ),
            };
        }
    }

    /// The Design graph pane (spec 23.2, spec 23.4): the edit switch that used to sit in the
    /// top bar, the inspector while editing, and the canvas under them.
    pub(crate) fn graph_tab(&mut self, ui: &mut egui::Ui) {
        egui::TopBottomPanel::top("graph-tools").show_inside(ui, |ui| self.edit_bar(ui));
        // The inspector is only there in edit mode: a read-only graph has no parameter to set.
        if self.edit.is_some() {
            egui::SidePanel::right("inspector")
                .default_width(300.0)
                .show_inside(ui, |ui| self.inspector_panel(ui));
        }
        egui::CentralPanel::default().show_inside(ui, |ui| self.graph_canvas(ui));
    }

    /// Edit mode's switch and its Save, Undo and Redo, moved from the old top bar into the
    /// pane they act on (packet M12/Y10).
    fn edit_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let mut editing = self.edit.is_some();
            if ui
                .toggle_value(&mut editing, self.t("tab.edit"))
                .on_hover_text(self.t("tab.edit.hint"))
                .changed()
            {
                self.set_edit_mode(editing);
            }
            let Some(session) = &self.edit else {
                return;
            };
            let (undo, redo) = (session.can_undo(), session.can_redo());
            if ui.button(self.t("edit.save")).clicked() {
                self.save_edits();
            }
            if ui
                .add_enabled(undo, egui::Button::new(self.t("edit.undo")))
                .clicked()
            {
                if let Some(s) = self.edit.as_mut() {
                    s.undo();
                }
            }
            if ui
                .add_enabled(redo, egui::Button::new(self.t("edit.redo")))
                .clicked()
            {
                if let Some(s) = self.edit.as_mut() {
                    s.redo();
                }
            }
        });
    }

    fn graph_canvas(&mut self, ui: &mut egui::Ui) {
        // Nothing open: what to do about it, and no search box - there is nothing to find yet,
        // and an empty control is one more thing to wonder about (packet M7/E6).
        if self.edit.is_none() && self.opened.is_none() {
            super::shell::nothing_open(self, ui);
            return;
        }
        self.search_bar(ui);
        if self.edit.is_some() {
            self.edit_canvas(ui);
            return;
        }
        let hit = self.search.current();
        let Some(opened) = &self.opened else {
            return;
        };
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        if response.dragged() {
            self.pan += response.drag_delta();
        }
        if response.hovered() {
            let z = ui.input(|i| i.zoom_delta() * (1.0 + i.smooth_scroll_delta.y * 0.001));
            self.zoom = (self.zoom * z).clamp(0.2, 4.0);
        }
        let origin = response.rect.min + self.pan;
        let zoom = self.zoom;
        let at = |p: [f32; 2]| origin + Vec2::new(p[0], p[1]) * zoom;
        let size = Vec2::new(NODE_W, NODE_H) * zoom;

        for (i, layer) in opened.graph.layers.iter().enumerate() {
            let highlight = hit.filter(|(l, _)| *l == i).map(|(_, node)| node);
            paint_layer(&painter, layer, &at, size, zoom, highlight);
        }
        for edge in &opened.graph.cross_edges {
            paint_cross_edge(&painter, &opened.graph, edge, &at, size, zoom);
        }
    }

    /// Edit mode (spec 23.4 stage 2). Every branch below ends in exactly one
    /// [`EditSession::apply`], [`EditSession::undo`] or [`EditSession::redo`] call - this
    /// function decides nothing else, which is what keeps the untested half thin.
    fn edit_canvas(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.lang;
        let Self {
            edit: Some(session),
            palette,
            pan,
            zoom,
            drag,
            selected,
            status,
            ..
        } = self
        else {
            return;
        };
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        if response.hovered() {
            *zoom = (*zoom * ui.input(eframe::egui::InputState::zoom_delta)).clamp(0.2, 4.0);
        }
        let origin = response.rect.min + *pan;
        let z = *zoom;
        let to_graph = move |p: Pos2| [(p.x - origin.x) / z, (p.y - origin.y) / z];

        // An owned snapshot of the geometry the pointer is tested against, so that no borrow
        // of `session` is alive across the one `apply` below.
        let mut view = CanvasView::of(session, origin, z);

        let mut pending: Option<Edit> = None;
        // A plain click selects what is under it, and a click on the background clears the
        // selection - the same hit test the drag uses, so a node that can be dragged can be
        // clicked. `clicked()` is the primary button only, so the right-click that opens the
        // add-node menu below leaves the selection alone.
        if let (true, Some(pos)) = (response.clicked(), response.interact_pointer_pos()) {
            *selected = view.hit(pos);
        }
        if let (true, Some(pos)) = (response.drag_started(), response.interact_pointer_pos()) {
            let started = view.start_drag(pos);
            if let Drag::Node { id, .. } = &started {
                *selected = Some(*id);
            }
            *drag = Some(started);
        }
        match drag.clone() {
            Some(Drag::Pan) => *pan += response.drag_delta(),
            Some(Drag::Node {
                id,
                grab,
                origin: was,
            }) => {
                if let Some(pos) = response.interact_pointer_pos() {
                    let moved = to_graph(pos - grab);
                    if response.drag_stopped() {
                        // Put the node back where the gesture began and record one edit, so
                        // undo returns to the position before the whole drag.
                        session.layout.positions.insert(id, was);
                        pending = Some(Edit::MoveNode {
                            node: id,
                            pos: moved,
                        });
                    } else {
                        session.layout.positions.insert(id, moved);
                        view.positions.insert(id, moved);
                    }
                }
            }
            Some(Drag::Link { from }) => {
                if let (Some(pos), Some(start)) =
                    (response.interact_pointer_pos(), view.port_pos(&from, false))
                {
                    bezier(&painter, start, pos, true, LINK, z);
                    if response.drag_stopped() {
                        if let Some(to) = view.port_at(pos, true) {
                            pending = Some(Edit::Connect { from, to });
                        }
                    }
                }
            }
            None => {}
        }
        if response.drag_stopped() {
            *drag = None;
        }

        ui.input(|i| {
            if i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace) {
                if let Some(node) = *selected {
                    pending = Some(Edit::RemoveNode { node });
                }
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Z) {
                session.undo();
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Y) {
                session.redo();
            }
        });

        let ir = session.graph.kind();
        let menu_at = response.interact_pointer_pos().map(to_graph);
        response.context_menu(|ui| {
            ui.label(i18n::t(lang, "edit.add_node"));
            for (category, entries) in palette.by_category() {
                if entries.first().is_none_or(|e| e.ir != ir) {
                    continue;
                }
                ui.menu_button(category, |ui| {
                    for entry in entries {
                        if ui.button(&entry.kind).clicked() {
                            pending = Some(Edit::AddNode {
                                kind: entry.kind.clone(),
                                params: entry.defaults(),
                                pos: menu_at.unwrap_or([0.0, 0.0]),
                            });
                            ui.close_kind(egui::UiKind::Menu);
                        }
                    }
                });
            }
        });

        if let Some(edit) = pending {
            match session.apply(edit) {
                Ok(()) => {
                    *status = i18n::fill(
                        lang,
                        "status.edits",
                        &[&session.history().len().to_string()],
                    );
                }
                Err(diags) => {
                    *status = diags.first().map_or_else(
                        || i18n::fill(lang, "status.refused", &[]),
                        |d| i18n::fill(lang, "status.refused", &[&d.code.to_string(), &d.message]),
                    );
                }
            }
            view = CanvasView::of(session, origin, z);
        }
        if selected.is_some_and(|id| !session.graph.contains(id)) {
            *selected = None;
        }
        view.paint(&painter, *selected);
    }

    /// The Run tab (packet M7/E1): the cell table, the acceptance verdict, and for the
    /// selected cell its Safety Plane timeline and a filmstrip. Every number, every order and
    /// every decoded byte is [`RunDir`]'s; this turns them into widgets.
    pub(crate) fn run_tab(&mut self, ui: &mut egui::Ui) {
        // The Launch section is above the table and there whether or not anything is open:
        // starting a run is how the tab gets something to show (packet M7/E5).
        egui::TopBottomPanel::top("launch")
            .resizable(true)
            .default_height(400.0)
            .show_inside(ui, |ui| {
                // A pane shorter than the form scrolls it rather than cutting its flags off.
                egui::ScrollArea::vertical()
                    .id_salt("launch-form")
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.launch_panel(ui));
            });
        if self.run.is_none() && self.telemetry.live.is_empty() {
            ui.label(self.t("results.empty"));
            return;
        }
        // The replay of the selected cell shares the tab (packet M7/E2): the table picks the
        // episode, the panel plays it. The model decides how tall it is: its control rows
        // until a replay is loaded, a canvas afterwards. Two panel ids, because egui
        // remembers a panel's dragged height per id and the two states want their own.
        match replay_view::panel_height(self.replay.as_ref(), ui.available_height()) {
            Some(height) => {
                egui::TopBottomPanel::bottom("replay-canvas")
                    .resizable(true)
                    .default_height(height)
                    .show_inside(ui, |ui| self.replay_panel(ui));
            }
            None => {
                egui::TopBottomPanel::bottom("replay-controls")
                    .resizable(false)
                    .show_inside(ui, |ui| self.replay_panel(ui));
            }
        }
        egui::CentralPanel::default().show_inside(ui, |ui| self.run_table(ui));
    }

    fn run_table(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        // Captured before the destructure below, which borrows the rest of `self`.
        let lang = self.settings.lang;
        let Self {
            run,
            run_frames,
            telemetry,
            ..
        } = self;
        // **One table, two ends** (design note section 13). A finished run's rows come off
        // disk and a live one's off the wire, but both are `CellRow`s and a `Timeline`, so
        // everything below this match is the same code for either -- and none of it decides
        // anything: the rows, the headings and the strip are the models' (spec 28.10 rule 3).
        let live = &telemetry.live;
        // Which end the rows came from, which is the only thing the "joined late" mark means
        // anything for: a run read off disk is whole by the time it is opened.
        let is_live = run.is_none();
        let (columns, rows, selected, heading, acceptance) = match run.as_ref() {
            Some(run) => (
                run_view::columns(run),
                run.cells().to_vec(),
                run.selected_cell().map(|c| c.name.clone()),
                i18n::t(
                    lang,
                    if run.report.passed {
                        "results.passed"
                    } else {
                        "results.failed"
                    },
                )
                .to_owned(),
                run.acceptance().to_vec(),
            ),
            // A live run has no verdict yet: `report.json` is written after the last suite.
            None => (
                live.columns(),
                live.cells(),
                live.selected_cell(),
                live.status(),
                Vec::new(),
            ),
        };
        let timeline = selected.as_ref().map(|cell| match run.as_ref() {
            Some(run) => run.timeline(cell),
            None => live.timeline(cell),
        });
        let mut sort = None;
        let mut select = None;
        // Everything below is one scroll area, so a short window clips nothing: the table, the
        // acceptance rows, the strip and the filmstrip scroll together.
        let stages: Vec<(String, Option<f64>, bool)> = live
            .stages()
            .iter()
            .map(|s| (s.name.clone(), s.seconds, s.running()))
            .collect();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // A cycle publishes every stage on one socket (packet M7/E7), so the strip
                // above the table says which one these rows came from. A run that is one
                // command publishes no stage and the strip is not drawn at all.
                if !stages.is_empty() {
                    ui.heading(i18n::t(lang, "results.stages"));
                    ui.horizontal_wrapped(|ui| {
                        for (name, seconds, running) in &stages {
                            let plain = labels::stage_label(lang, name);
                            let word = if plain.is_empty() { name } else { plain };
                            let text = match seconds {
                                Some(s) => format!("{word}  {s:.1}s"),
                                None => word.to_owned(),
                            };
                            let colour = if *running {
                                Color32::from_rgb(240, 200, 80)
                            } else {
                                ui.visuals().text_color()
                            };
                            ui.colored_label(colour, text).on_hover_text(name);
                        }
                    });
                    ui.separator();
                }
                egui::Grid::new("run-cells").striped(true).show(ui, |ui| {
                    // A cycle's rows come from several stages and two of them name their
                    // episodes alike (packet M7/R12), so the stage is a column of its own. A
                    // run that is one command has no stages and no column. It does not sort:
                    // the rows are in the order the cycle ran them.
                    if !stages.is_empty() {
                        ui.label(i18n::t(lang, "column.stage"));
                    }
                    // The header is the metric's plain name and the hover is the raw one the
                    // report carries; a column this build has no word for keeps its raw name,
                    // which is still better than an empty heading (packet M7/E6).
                    for (i, name) in columns.iter().enumerate() {
                        let plain = labels::column_label(lang, name);
                        let heading = if plain.is_empty() { name } else { plain };
                        if ui.button(heading).on_hover_text(name).clicked() {
                            sort = Some(i);
                        }
                    }
                    ui.label(i18n::t(lang, "column.traj"));
                    ui.label(i18n::t(lang, "column.frames"));
                    ui.end_row();
                    for row in &rows {
                        // The row is the pair, not the name (packet M7/R12): selecting and
                        // the strip below both go by the key the model builds.
                        let key = cell_key(&row.stage, &row.name);
                        if !stages.is_empty() {
                            let plain = labels::stage_label(lang, &row.stage);
                            let word = if plain.is_empty() { &row.stage } else { plain };
                            ui.label(word).on_hover_text(&row.stage);
                        }
                        let is_selected = selected.as_deref() == Some(key.as_str());
                        // A cell whose beginning this viewer missed says so beside its name:
                        // what it shows is the part it heard, not the whole episode.
                        let name = if is_live && live.joined_late(&key) {
                            format!("{}  {}", row.name, i18n::t(lang, "results.joined_late"))
                        } else {
                            row.name.clone()
                        };
                        if ui.selectable_label(is_selected, name).clicked() {
                            select = Some(key);
                        }
                        ui.label(&row.suite);
                        ui.label(row.seed.map_or_else(|| "--".to_owned(), |s| s.to_string()));
                        for column in columns.iter().skip(3) {
                            ui.label(
                                row.metrics
                                    .get(column)
                                    .map_or_else(|| dash(lang), |v| metric_text(lang, v)),
                            );
                        }
                        ui.label(i18n::t(
                            lang,
                            if row.has_traj {
                                "value.yes"
                            } else {
                                "value.none"
                            },
                        ));
                        ui.label(row.frames.to_string());
                        ui.end_row();
                    }
                });

                ui.separator();
                ui.heading(&heading)
                    .on_hover_text(i18n::t(lang, "results.acceptance.hint"));
                for line in &acceptance {
                    let (text, colour) = acceptance_row(lang, line);
                    ui.colored_label(colour, text);
                }

                let (Some(cell), Some(timeline)) = (selected.as_ref(), timeline.as_ref()) else {
                    ui.separator();
                    ui.label(i18n::t(lang, "results.select_cell"));
                    return;
                };
                ui.separator();
                ui.heading(run_view::timeline_heading(timeline, cell));
                // One column per ~4 px of the strip; the model folds the frames into them.
                let n = (ui.available_width() / 4.0) as usize;
                paint_timeline(lang, ui, &timeline.buckets(n));
                for kind in timeline.kind_rows() {
                    ui.label(run_view::kind_label(&kind));
                }

                ui.separator();
                ui.heading(i18n::t(lang, "results.frames"));
                // A live run has no filmstrip on disk: what it has is the observation frame
                // the producer is publishing right now (stream 4), uploaded once per image
                // rather than once per repaint.
                let Some(run) = run.as_ref() else {
                    if let Some(image) = live.image() {
                        let key = format!("live#{}", live.images());
                        if !run_frames.contains_key(&key) {
                            run_frames.clear();
                            run_frames.insert(key.clone(), rgb_texture(&ctx, &key, image));
                        }
                        if let Some(texture) = run_frames.get(&key) {
                            let scale = (160.0 / texture.size_vec2().x).max(1.0);
                            ui.image(egui::load::SizedTexture::new(
                                texture.id(),
                                texture.size_vec2() * scale,
                            ));
                        }
                    } else {
                        ui.label(i18n::t(lang, "results.no_image"));
                    }
                    return;
                };
                // Eight thumbnails are wider than a narrow window; scroll them sideways rather
                // than cutting the last ones off.
                egui::ScrollArea::horizontal()
                    .id_salt("filmstrip")
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for index in run.filmstrip(cell, FILMSTRIP) {
                                let key = format!("{cell}#{index}");
                                let texture = run_frames.entry(key.clone()).or_insert_with(|| {
                                    let image = run.frame(cell, index).unwrap_or(Rgb8Image {
                                        width: 1,
                                        height: 1,
                                        data: vec![0, 0, 0],
                                    });
                                    rgb_texture(&ctx, &key, &image)
                                });
                                ui.vertical(|ui| {
                                    ui.label(format!("{index}"));
                                    let scale = (160.0 / texture.size_vec2().x).max(1.0);
                                    ui.image(egui::load::SizedTexture::new(
                                        texture.id(),
                                        texture.size_vec2() * scale,
                                    ));
                                });
                            }
                        });
                    });
            });
        // Sorting is a finished run's: a live table is in cell-name order and its rows are
        // still arriving. Selecting works on either.
        if let (Some(column), Some(run)) = (sort, run.as_mut()) {
            run.sort_by(column);
        }
        if let Some(name) = select {
            match run.as_mut() {
                Some(run) => run.select(&name),
                None => telemetry.live.select(&name),
            }
        }
    }

    /// The Launch section (packet M7/E5, spec 23.1): the editor is a **client**, so this
    /// starts a child process and attaches to it — it hosts nothing.
    ///
    /// Wiring only. Which flags the chosen kind has, what each is called, what the command
    /// line reads as, what the exit code means and how long to wait for the producer's socket
    /// are all [`LaunchModel`]'s, under test (spec 28.10 rule 3). There is no Pause and no
    /// Step: the run speaks no control protocol (spec 23.3), so Kill is the only control.
    fn launch_panel(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.lang;
        let mut start = false;
        let running = matches!(self.launch.state(), LaunchState::Running { .. });
        ui.horizontal(|ui| {
            // The selector says what the command *does*; the hover says what it is, which is
            // the command line itself (packet M7/E6).
            for kind in LaunchKind::ALL {
                ui.selectable_value(&mut self.launch.kind, kind, labels::kind_label(lang, kind))
                    .on_hover_text(kind.label());
            }
            ui.separator();
            let missing = self.launch.missing_required();
            start = ui
                .add_enabled(
                    !running && missing.is_empty(),
                    egui::Button::new(self.t("launch.start")),
                )
                .clicked();
            if ui
                .add_enabled(running, egui::Button::new(self.t("launch.stop")))
                .clicked()
            {
                self.launch.kill();
            }
            ui.separator();
            if missing.is_empty() {
                ui.label(self.launch.status_line());
            } else {
                // What to fill in, by the panel's own words for the fields (packet M7/E6).
                let names: Vec<&str> = missing
                    .iter()
                    .map(|f| labels::launch_label(lang, *f))
                    .collect();
                ui.label(format!("{} {}", self.t("launch.missing"), names.join(", ")));
            }
        });
        // One `horizontal` per flag rather than an `egui::Grid`: a grid caps a cell at the
        // column width it measured last frame, which squeezes a `TextEdit` down to the
        // default interact size and never lets it grow back.
        let mut picked: Option<(crate::model::launch::LaunchField, String)> = None;
        for field in self.launch.fields() {
            ui.horizontal(|ui| {
                // The label is the plain name and the hover is the flag the CLI is given, so
                // the panel still tells anyone who asks exactly what it will type.
                ui.add_sized(
                    FLAG_LABEL,
                    egui::Label::new(labels::launch_label(lang, *field)),
                )
                .on_hover_text(field.flag());
                ui.add(
                    egui::TextEdit::singleline(self.launch.field_mut(*field))
                        .hint_text(field.hint())
                        .desired_width(540.0),
                );
                if let Some(browse) = labels::browses(*field) {
                    if let Some(path) = self.browse(ui, browse) {
                        picked = Some((*field, path));
                    }
                }
            });
        }
        if let Some((field, path)) = picked {
            *self.launch.field_mut(field) = path;
        }
        ui.horizontal(|ui| {
            for flag in self.launch.flags() {
                let label = labels::launch_flag_label(lang, *flag);
                let raw = flag.flag();
                ui.checkbox(self.launch.flag_mut(*flag), label)
                    .on_hover_text(raw);
            }
        });
        // The command line, read-only: what is about to run, in one place, so nobody has to
        // guess which `es` or which flags the panel decided on. It is `es`'s own spelling and
        // stays untranslated - this is the line a person would paste into a terminal.
        ui.label(self.t("launch.command"));
        let mut command = self.launch.command_line();
        ui.add(
            egui::TextEdit::singleline(&mut command)
                .desired_width(f32::INFINITY)
                .interactive(false),
        );
        egui::ScrollArea::vertical()
            .id_salt("launch-lines")
            // A box of its own height inside the form's scroll, not one that fills it.
            .max_height(160.0)
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in self.launch.lines() {
                    ui.monospace(line);
                }
            });
        if start {
            self.start_launch();
        }
    }

    /// Start; the attach follows through [`LaunchModel::take_attached`] in `update`, when
    /// the dial the model runs on its own thread has answered. Never here: on a machine where
    /// a refused connect takes seconds, a dial on this thread held the whole window.
    fn start_launch(&mut self) {
        self.launch.start();
        self.status = self.launch.status_line();
    }

    /// The Replay panel (packet M7/E2): the selected cell's `.estraj`, posed and projected by
    /// [`ReplayView`] and painted as one mesh. The gestures map to the model's pure camera
    /// functions and to `advance`; nothing is decided here.
    fn replay_panel(&mut self, ui: &mut egui::Ui) {
        let dt = f64::from(ui.input(|i| i.stable_dt));
        let lang = self.settings.lang;
        let scene_browse = dialogs::AVAILABLE.then(|| self.browse_label(Browse::Scene));
        let Self {
            run,
            replay,
            replay_cell,
            replay_texture,
            scene_path,
            frames_path,
            camera,
            status,
            ..
        } = self;
        let selected = run
            .as_ref()
            .and_then(RunDir::selected_cell)
            .filter(|c| c.has_traj)
            .map(|c| c.name.clone());

        ui.horizontal(|ui| {
            ui.label(i18n::t(lang, "replay.scene"));
            ui.add(
                egui::TextEdit::singleline(scene_path)
                    .hint_text(i18n::t(lang, "replay.scene.hint"))
                    .desired_width(220.0),
            );
            if let Some(label) = scene_browse {
                if ui.button(label).clicked() {
                    if let Some(path) = dialogs::pick(Browse::Scene) {
                        *scene_path = path.display().to_string();
                    }
                }
            }
            // `es eval run --frames <dir>` writes wherever it was told, which is usually a
            // sibling of the run directory; the model re-scans when this is applied.
            ui.label(i18n::t(lang, "replay.frames"));
            let field = ui.add(
                egui::TextEdit::singleline(frames_path)
                    .hint_text(i18n::t(lang, "replay.frames.hint"))
                    .desired_width(220.0),
            );
            if field.lost_focus() {
                if let Some(run) = run.as_mut() {
                    run.set_frames_root(frames_path.trim());
                    *status = format!("{}: {}", run.dir.display(), run_view::status(run));
                }
            }
            let replay_label = i18n::t(lang, "replay.replay");
            let label = selected.as_ref().map_or_else(
                || replay_label.to_owned(),
                |name| format!("{replay_label} {name}"),
            );
            if ui
                .add_enabled(selected.is_some(), egui::Button::new(label))
                .clicked()
            {
                let (Some(run), Some(cell)) = (run.as_ref(), selected.as_ref()) else {
                    return;
                };
                match ReplayView::open(Path::new(scene_path.trim()), &run.traj_path(cell)) {
                    Ok(view) => {
                        *status = format!("{cell}: {} tick(s) replayed", view.ticks());
                        cell.clone_into(replay_cell);
                        *replay = Some(view);
                        *replay_texture = Canvas::default();
                    }
                    Err(e) => *status = e.to_string(),
                }
            }
            if let Some(view) = replay.as_mut() {
                if ui
                    .button(i18n::t(
                        lang,
                        if view.playing {
                            "replay.pause"
                        } else {
                            "replay.play"
                        },
                    ))
                    .clicked()
                {
                    view.playing = !view.playing;
                }
                if ui.button("|<").clicked() {
                    view.step(-1);
                }
                if ui.button(">|").clicked() {
                    view.step(1);
                }
                ui.add(egui::Slider::new(&mut view.speed, 0.1..=4.0).text("x"));
            }
        });

        let Some(view) = replay.as_mut() else {
            ui.label(i18n::t(lang, "replay.empty"));
            return;
        };
        view.advance(dt, REPLAY_RATE_HZ);
        let last = view.ticks().saturating_sub(1);
        ui.horizontal(|ui| {
            ui.label(format!(
                "{replay_cell}  tick {}/{last}  {:.2} s",
                view.tick,
                view.tick as f64 / REPLAY_RATE_HZ
            ));
            ui.add(egui::Slider::new(&mut view.tick, 0..=last).text("tick"));
        });

        replay_canvas(ui, lang, ui.available_size(), view, camera, replay_texture);
        if view.playing {
            ui.ctx().request_repaint();
        }
    }

    pub(crate) fn telemetry_tab(&mut self, ui: &mut egui::Ui) {
        // Spec 23.1: the editor attaches to a running process. The address and the token
        // are typed here and dialled by the model, which owns every error string.
        ui.horizontal(|ui| {
            ui.label(self.t("live.attach"))
                .on_hover_text(self.t("live.attach.hint"));
            ui.add(
                egui::TextEdit::singleline(&mut self.attach_addr)
                    .hint_text(crate::model::launch::DEFAULT_TELEMETRY)
                    .desired_width(160.0),
            );
            let token_hint = self.t("live.token");
            ui.add(
                egui::TextEdit::singleline(&mut self.attach_token)
                    .hint_text(token_hint)
                    .password(true)
                    .desired_width(160.0),
            );
            if ui.button(self.t("live.connect")).clicked() {
                match telemetry_view::attach(&self.attach_addr, &self.attach_token) {
                    Ok(source) => {
                        self.source = source;
                        self.status = format!("attached to {}", self.attach_addr.trim());
                    }
                    Err(e) => self.status = e,
                }
            }
        });
        ui.separator();
        // A training run's curves first: its speed table is all "not measured" (packet M16/H4).
        let learning_first = !self.telemetry.train.is_empty();
        if learning_first {
            self.training_section(ui);
            ui.separator();
        }
        ui.heading(self.t("live.performance"))
            .on_hover_text(self.t("live.performance.hint"));
        if self.telemetry.received == 0 {
            ui.label(self.t("live.empty"));
        }
        egui::Grid::new("metrics").striped(true).show(ui, |ui| {
            for (name, value) in self.telemetry.metric_rows() {
                // The row is named by the metric it is, and hovers the raw spelling the
                // producer sends (packet M7/E6).
                let plain = labels::metric_by_name(name)
                    .map_or(name, |m| labels::metric_label(self.settings.lang, m));
                ui.label(plain).on_hover_text(name);
                ui.label(value.map_or_else(
                    || self.t("value.not_measured").to_owned(),
                    |v| format!("{v:.3}"),
                ));
                ui.end_row();
            }
        });
        ui.separator();
        if !learning_first {
            self.training_section(ui);
            ui.separator();
        }
        ui.heading(self.t("live.streams"));
        egui::Grid::new("streams").striped(true).show(ui, |ui| {
            for (key, tick, value) in self.telemetry.latest() {
                ui.label(format!("stream {}[{}]", key.stream.0, key.index));
                ui.label(format!("tick {tick}"));
                ui.label(format!("{value:.4}"));
                ui.end_row();
            }
        });
        ui.separator();
        ui.heading(self.t("live.events"));
        for e in self.telemetry.events.iter().rev().take(200) {
            ui.label(format!("[{}] {} {:?}", e.tick, e.kind, e.fields));
        }
    }

    /// The Live tab's Training section (packet M7/E7): the learning curve, the learning rate,
    /// the checkpoint marks and the tensor the network is fitting.
    ///
    /// Wiring only. The curve's scale, its normalisation, the marks' positions and the ETA are
    /// [`crate::model::train_view::TrainView`]'s and are judged headlessly; this maps a
    /// rectangle and paints a polyline. **There is no plotting crate** — two `line_segment`
    /// runs over points already in the unit square is the whole of it (spec 28.10 rule 3).
    fn training_section(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.lang;
        let ctx = ui.ctx().clone();
        ui.heading(self.t("live.training"))
            .on_hover_text(self.t("live.training.hint"));
        if self.telemetry.train.is_empty() {
            ui.label(self.t("live.training.empty"));
            return;
        }
        let train = &self.telemetry.train;
        // Where the run has got to, how fast, and how much is left -- the model's numbers,
        // formatted by the tables (packet M7/E6).
        ui.horizontal_wrapped(|ui| {
            if let Some(step) = train.step() {
                ui.label(match train.total() {
                    Some(total) => {
                        i18n::fill(lang, "live.step", &[&step.to_string(), &total.to_string()])
                    }
                    None => i18n::fill(lang, "live.step_only", &[&step.to_string()]),
                });
            }
            if let Some(eta) = train.total().and_then(|t| train.eta(t)) {
                ui.label(i18n::fill(
                    lang,
                    "live.eta",
                    &[&format!("{:.0}s", eta.as_secs_f64())],
                ));
            }
            if let Some(rate) = train.throughput() {
                ui.label(i18n::fill(
                    lang,
                    "live.throughput",
                    &[&format!("{rate:.0}")],
                ));
            }
        });
        // The packed marks, one row each: step and `policy_hash` (packet M16/H4 -- a finished
        // run's folder is opened for these as much as for its curve).
        if !train.checkpoints().is_empty() {
            let title = i18n::fill(
                lang,
                "live.checkpoints",
                &[&train.checkpoints().len().to_string()],
            );
            egui::CollapsingHeader::new(title)
                .id_salt("train-marks")
                .show(ui, |ui| {
                    for (step, hash) in train.checkpoints() {
                        ui.monospace(format!("{step:>8}  {hash}"));
                    }
                });
        }
        let mut log = train.log_scale;
        ui.checkbox(&mut log, self.t("live.log_scale"));
        let loss = train.plot(Series::Loss, log);
        let lr = train.plot(Series::Lr, false);
        // An `[rl]` run's learning (packet M16/H4): drawn only for a run that has any.
        let rl: Vec<(Series, Option<Plot>)> = if train.rl().step.is_empty() {
            Vec::new()
        } else {
            Series::RL.map(|s| (s, train.plot(s, false))).to_vec()
        };
        let (sample, samples) = (train.sample().cloned(), train.samples());
        self.telemetry.train.log_scale = log;

        // An `[rl]` run is judged by its learning, not its loss, so that comes first; and a
        // rollout has no sample image to show, so its column is not offered.
        let picture = rl.is_empty() || sample.is_some();
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                if picture {
                    ui.set_max_width((ui.available_width() - 180.0).max(240.0));
                }
                if !rl.is_empty() {
                    ui.strong(i18n::t(lang, "live.rl"))
                        .on_hover_text(i18n::t(lang, "live.rl.hint"));
                }
                for (i, (series, plot)) in rl.into_iter().enumerate() {
                    ui.label(i18n::t(lang, series.key()));
                    match plot {
                        Some(plot) => {
                            paint_curve(ui, &plot, RL_COLOURS[i % RL_COLOURS.len()], 70.0);
                        }
                        None => {
                            ui.label(i18n::t(lang, "value.not_measured"));
                        }
                    }
                }
                for (key, plot, colour) in [
                    ("live.loss", loss, Color32::from_rgb(120, 200, 255)),
                    ("live.lr", lr, Color32::from_rgb(200, 160, 255)),
                ] {
                    ui.label(i18n::t(lang, key));
                    match plot {
                        Some(plot) => {
                            paint_curve(ui, &plot, colour, 90.0);
                        }
                        None => {
                            ui.label(i18n::t(lang, "results.no_events"));
                        }
                    }
                }
            });
            if !picture {
                return;
            }
            // What the network is looking at, beside the curve: the batch's own image input
            // after augmentation, uploaded once per sample rather than once per repaint.
            ui.vertical(|ui| {
                ui.label(i18n::t(lang, "live.sample"))
                    .on_hover_text(i18n::t(lang, "live.sample.hint"));
                let Some(image) = sample else {
                    ui.label(i18n::t(lang, "live.sample.empty"));
                    return;
                };
                let key = format!("sample#{samples}");
                if !self.run_frames.contains_key(&key) {
                    self.run_frames
                        .insert(key.clone(), rgb_texture(&ctx, &key, &image));
                }
                if let Some(texture) = self.run_frames.get(&key) {
                    let scale = (160.0 / texture.size_vec2().x).max(1.0);
                    ui.image(egui::load::SizedTexture::new(
                        texture.id(),
                        texture.size_vec2() * scale,
                    ));
                }
            });
        });
    }

    pub(crate) fn images_tab(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let empty = self.t("sees.empty");
        let (before, after) = (self.t("sees.before"), self.t("sees.after"));
        let count = |pairs: usize, outputs: usize| {
            i18n::fill(
                self.settings.lang,
                "sees.count",
                &[&pairs.to_string(), &outputs.to_string()],
            )
        };
        let Some(opened) = &mut self.opened else {
            ui.label(empty);
            return;
        };
        if let Some(err) = &opened.image_error {
            ui.label(err.as_str());
        }
        let outputs = opened.observation.outputs.len();
        ui.label(count(opened.pairs.len(), outputs));
        for pair in &opened.pairs {
            let (pair_before, pair_after) = opened
                .textures
                .entry(pair.name.clone())
                .or_insert_with(|| (texture(&ctx, pair, true), texture(&ctx, pair, false)));
            ui.heading(&pair.name);
            ui.horizontal(|ui| {
                for (label, tex) in [(before, &*pair_before), (after, &*pair_after)] {
                    ui.vertical(|ui| {
                        ui.label(label);
                        let scale = (240.0 / tex.size_vec2().x).max(1.0);
                        ui.image(egui::load::SizedTexture::new(
                            tex.id(),
                            tex.size_vec2() * scale,
                        ));
                    });
                }
            });
            ui.separator();
        }
    }

    pub(crate) fn diagnostics_tab(&mut self, ui: &mut egui::Ui) {
        // Editing has its own list, re-validated after every edit by `EditSession::apply`;
        // the opened bundle's is a snapshot of the four IRs as they were read from disk.
        let diagnostics: &[es_ir::Diagnostic] = match (&self.edit, &self.opened) {
            (Some(session), _) => session.diagnostics(),
            (None, Some(opened)) => &opened.graph.diagnostics,
            (None, None) => {
                ui.label(self.t("problems.empty"));
                return;
            }
        };
        if diagnostics.is_empty() {
            ui.label(self.t("problems.none"));
            return;
        }
        for d in diagnostics {
            ui.label(d.to_string());
        }
    }
}

fn texture(ctx: &egui::Context, pair: &ImagePair, before: bool) -> egui::TextureHandle {
    let img = if before { &pair.before } else { &pair.after };
    let name = format!("{}-{}", pair.name, if before { "before" } else { "after" });
    rgb_texture(ctx, &name, img)
}

pub(crate) fn rgb_texture(ctx: &egui::Context, name: &str, img: &Rgb8Image) -> egui::TextureHandle {
    let color = egui::ColorImage::from_rgb([img.width, img.height], &img.data);
    ctx.load_texture(name, color, egui::TextureOptions::NEAREST)
}

// --- the Run tab -------------------------------------------------------------------------------

/// Frames the filmstrip shows, sampled evenly over the cell by [`RunDir::filmstrip`].
const FILMSTRIP: usize = 8;

fn dash(lang: Lang) -> String {
    i18n::t(lang, "value.none").to_owned()
}

/// A metric cell. A histogram has no single number and an unmeasured metric has none at all
/// (spec 10.3): neither is rendered as `0`.
pub(crate) fn metric_text(lang: Lang, value: &es_ir::evaluation::MetricValue) -> String {
    match value {
        es_ir::evaluation::MetricValue::Scalar(v) => format!("{v:.4}"),
        es_ir::evaluation::MetricValue::Histogram(h) => {
            i18n::fill(lang, "value.histogram", &[&h.len().to_string()])
        }
        // The reason is the runner's own sentence and is shown verbatim: inventing a
        // translation for it would be the editor guessing at what `es` meant.
        es_ir::evaluation::MetricValue::Unavailable { reason } => {
            format!("{} ({reason})", i18n::t(lang, "value.not_measured"))
        }
    }
}

fn acceptance_row(lang: Lang, line: &es_ir::evaluation::AcceptanceResult) -> (String, Color32) {
    match line {
        es_ir::evaluation::AcceptanceResult::Determined {
            criterion,
            observed,
            passed,
        } => (
            format!(
                "{} {} {} ({}, {}): {observed:.4} -- {}",
                labels::metric_label(lang, criterion.metric),
                criterion.comparator.name(),
                criterion.threshold,
                criterion.aggregation.name(),
                criterion.suite.as_deref().unwrap_or("*"),
                i18n::t(
                    lang,
                    if *passed {
                        "results.pass"
                    } else {
                        "results.fail"
                    }
                )
            ),
            if *passed {
                Color32::from_rgb(120, 200, 120)
            } else {
                Color32::from_rgb(230, 120, 110)
            },
        ),
        es_ir::evaluation::AcceptanceResult::Unavailable { metric, reason } => (
            format!(
                "{}: {} ({reason})",
                labels::metric_label(lang, *metric),
                i18n::t(lang, "results.not_measured")
            ),
            Color32::from_gray(160),
        ),
    }
}

/// One colour per `EventSource`, violations as a tick beneath (spec 23.3).
pub(crate) fn paint_timeline(lang: Lang, ui: &mut egui::Ui, buckets: &[Bucket]) {
    if buckets.is_empty() {
        ui.label(i18n::t(lang, "results.no_events"));
        return;
    }
    let (response, painter) =
        ui.allocate_painter(Vec2::new(ui.available_width(), 30.0), Sense::hover());
    let rect = response.rect;
    let w = (rect.width() / buckets.len() as f32).max(1.0);
    for (i, bucket) in buckets.iter().enumerate() {
        let x = rect.left() + i as f32 * rect.width() / buckets.len() as f32;
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(w, 18.0)),
            0.0,
            source_colour(bucket.source),
        );
        if !bucket.counts.is_empty() {
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(x, rect.top() + 21.0), Vec2::new(w, 8.0)),
                0.0,
                Color32::from_rgb(240, 200, 80),
            );
        }
    }
}

/// One curve, painted (packet M7/E7). **No plotting crate**: the points arrive in the unit
/// square from [`crate::model::train_view::TrainView::plot`], and this maps them onto a
/// rectangle, joins them, and draws a vertical line where a checkpoint was packed. The two
/// numbers beside it are the range the model normalised against, in the series' own units.
/// `height` in points: a strip in the Live pane, the whole centre while ③ trains (M12/Y12).
/// Returns the rectangle the unit square was mapped onto, for ③'s preview marks (M13/Z5a).
/// One colour per `Series::RL` curve, in its order.
pub(crate) const RL_COLOURS: [Color32; 5] = [
    Color32::from_rgb(120, 220, 140),
    Color32::from_rgb(250, 210, 90),
    Color32::from_rgb(160, 200, 255),
    Color32::from_rgb(255, 150, 120),
    Color32::from_rgb(230, 120, 200),
];

pub(crate) fn paint_curve(ui: &mut egui::Ui, plot: &Plot, colour: Color32, height: f32) -> Rect {
    let (response, painter) =
        ui.allocate_painter(Vec2::new(ui.available_width(), height), Sense::hover());
    let rect = response.rect.shrink(4.0);
    let at = |p: &[f32; 2]| {
        Pos2::new(
            rect.left() + p[0] * rect.width(),
            rect.bottom() - p[1] * rect.height(),
        )
    };
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    for x in &plot.marks {
        let x = rect.left() + x * rect.width();
        painter.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(1.0_f32, Color32::from_rgb(240, 200, 80)),
        );
    }
    for pair in plot.points.windows(2) {
        painter.line_segment([at(&pair[0]), at(&pair[1])], Stroke::new(1.5_f32, colour));
    }
    let text = |pos: Pos2, anchor: Align2, value: f32| {
        painter.text(
            pos,
            anchor,
            format!("{value:.4}"),
            FontId::monospace(10.0),
            ui.visuals().weak_text_color(),
        );
    };
    text(rect.left_top(), Align2::LEFT_TOP, plot.max);
    text(rect.left_bottom(), Align2::LEFT_BOTTOM, plot.min);
    // The steps the two ends stand for, so a peak can be read off by where it falls.
    painter.text(
        rect.right_bottom(),
        Align2::RIGHT_BOTTOM,
        format!("{} … {}", plot.steps[0], plot.steps[1]),
        FontId::monospace(10.0),
        ui.visuals().weak_text_color(),
    );
    rect
}

// --- the Replay panel --------------------------------------------------------------------------

/// Where the replay camera starts; defined beside the scene preview it is tested with.
pub(crate) use crate::model::scene_view::SHOWCASE_CAMERA;

/// The control rate a recorded `.estraj` tick is worth. 50 Hz is the demo deployment's
/// `rate.control`; a run directory carries no Deployment IR to read it from, and playing at
/// the wrong rate only changes how fast the arm appears to move.
pub(crate) const REPLAY_RATE_HZ: f64 = 50.0;

/// Radians of orbit per point of drag, and zoom per point of scroll.
const ORBIT_PER_POINT: f64 = 0.008;
const ZOOM_PER_POINT: f64 = 0.002;

/// One viewport's pictures between frames: the fast raster and what it was drawn for
/// `(tick, camera)` (packet M7/E8), and the slower looks of packet M16/H8 with the texture of
/// their newest picture. Dropped with whatever it shows.
#[derive(Default)]
pub(crate) struct Canvas {
    raster: Option<((usize, Camera), egui::TextureHandle)>,
    look: Viewport,
    picture: Option<(u64, egui::TextureHandle)>,
    es: Option<std::path::PathBuf>,
}

/// What a canvas shows: a template's scene at its initial pose, or a replay at its tick.
#[derive(Clone, Copy)]
pub(crate) enum Posed<'a> {
    Scene(&'a ScenePreview),
    Replay(&'a ReplayView),
}

impl Posed<'_> {
    fn tick(self) -> usize {
        match self {
            Self::Scene(_) => 0,
            Self::Replay(view) => view.tick,
        }
    }

    fn project(self, camera: &Camera) -> Projected {
        match self {
            Self::Scene(preview) => preview.project(camera),
            Self::Replay(view) => view.project(view.tick, camera),
        }
    }

    fn shot(self, camera: &Camera) -> Shot {
        match self {
            Self::Scene(preview) => preview.shot(camera),
            Self::Replay(view) => view.shot(camera),
        }
    }

    fn tris(self) -> Result<es_render::TriScene, String> {
        match self {
            Self::Scene(preview) => Ok(preview.tris()),
            Self::Replay(view) => view.scene_at(view.tick).map_err(|e| e.to_string()),
        }
    }
}

/// The look every viewport draws, for the session: egui's own temporary memory, which no
/// store persists (packet M16/H8).
fn mode_id() -> egui::Id {
    egui::Id::new("viewport-look")
}

pub(crate) fn viewport_mode(ctx: &egui::Context) -> Mode {
    ctx.data(|d| d.get_temp(mode_id())).unwrap_or_default()
}

/// `es-editor --viewport <look>` and the selector above every viewport.
pub fn set_viewport_mode(ctx: &egui::Context, mode: Mode) {
    ctx.data_mut(|d| d.insert_temp(mode_id(), mode));
}

/// One replay frame in a `size` canvas: drag orbits, scroll zooms. Shared by the Replay panel
/// and the results screen's player (packet M12/Y13).
pub(crate) fn replay_canvas(
    ui: &mut egui::Ui,
    lang: Lang,
    size: Vec2,
    view: &ReplayView,
    camera: &mut Camera,
    canvas: &mut Canvas,
) {
    scene_canvas(ui, lang, size, Posed::Replay(view), camera, canvas);
}

/// [`replay_canvas`] for any posed scene, under the look selector (packet M16/H8). ① and ②
/// show a template's scene with it (packet M12/Y15).
pub(crate) fn scene_canvas(
    ui: &mut egui::Ui,
    lang: Lang,
    size: Vec2,
    posed: Posed<'_>,
    camera: &mut Camera,
    canvas: &mut Canvas,
) {
    let mut mode = viewport_mode(ui.ctx());
    let row = ui.horizontal(|ui| {
        ui.label(i18n::t(lang, "viewport.label"));
        for look in Mode::ALL {
            ui.selectable_value(&mut mode, look, i18n::t(lang, look.key()))
                .on_hover_text(i18n::t(lang, look.hint()));
        }
        if let Some(status) = canvas.look.status() {
            let (key, args) = status.line();
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            ui.weak(i18n::fill(lang, key, &args));
        }
    });
    set_viewport_mode(ui.ctx(), mode);
    let gap = row.response.rect.height() + ui.spacing().item_spacing.y;
    let size = Vec2::new(size.x, (size.y - gap).max(1.0));
    let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    (camera.width, camera.height) =
        Raster::size_for([response.rect.width(), response.rect.height()]);
    if response.dragged() {
        let drag = response.drag_delta();
        *camera = camera.orbit(
            f64::from(-drag.x) * ORBIT_PER_POINT,
            f64::from(drag.y) * ORBIT_PER_POINT,
        );
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            *camera = camera.zoom(f64::from(-scroll).mul_add(ZOOM_PER_POINT, 1.0));
        }
    }
    painter.rect_filled(response.rect, 0.0, Color32::from_gray(BACKGROUND));
    let es = (canvas.es).get_or_insert_with(|| crate::model::launch::es_binary().path);
    let shot = posed.shot(camera);
    let now = std::time::Instant::now();
    canvas.look.update(mode, &shot, now, es, || posed.tris());
    let texture = if let Some((picture, revision)) = canvas.look.picture() {
        if canvas.picture.as_ref().is_none_or(|(r, _)| *r != revision) {
            let px = [picture.width as usize, picture.height as usize];
            let image = egui::ColorImage::from_rgb(px, &picture.rgb);
            let options = egui::TextureOptions::LINEAR;
            let texture = ui.ctx().load_texture("viewport-look", image, options);
            canvas.picture = Some((revision, texture));
        }
        canvas.picture.as_ref().map(|(_, t)| t)
    } else {
        // One CPU frame per tick or camera change, never per repaint: the raster is the same
        // bytes until one of them moves, and re-drawing 2,700 triangles for a picture that
        // did not change would burn a core holding still.
        let key = (posed.tick(), *camera);
        if canvas.raster.as_ref().is_none_or(|(k, _)| *k != key) {
            let raster = Raster::draw(&posed.project(camera), camera.width, camera.height);
            let px = [raster.w as usize, raster.h as usize];
            let image = egui::ColorImage::from_rgb(px, &raster.rgb);
            let options = egui::TextureOptions::LINEAR;
            let texture = ui.ctx().load_texture("replay", image, options);
            canvas.raster = Some((key, texture));
        }
        canvas.raster.as_ref().map(|(_, t)| t)
    };
    if let Some(texture) = texture {
        // Stretched over the whole canvas: `size_for` kept the aspect, so this only ever
        // scales the picture up, and never by much.
        painter.image(
            texture.id(),
            response.rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    if canvas.look.busy() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(50));
    }
}

fn source_colour(source: es_eval::runner::EventSource) -> Color32 {
    match source {
        es_eval::runner::EventSource::Policy => Color32::from_rgb(70, 130, 180),
        es_eval::runner::EventSource::Human => Color32::from_rgb(150, 150, 200),
        es_eval::runner::EventSource::Clamped => Color32::from_rgb(220, 170, 60),
        es_eval::runner::EventSource::Fallback => Color32::from_rgb(210, 90, 80),
    }
}

fn paint_layer(
    painter: &egui::Painter,
    layer: &LayerView,
    at: &impl Fn([f32; 2]) -> Pos2,
    size: Vec2,
    zoom: f32,
    highlight: Option<NodeId>,
) {
    let font = FontId::proportional(12.0 * zoom);
    let band = layer.nodes.iter().find_map(|n| n.layout);
    if let Some(p) = band {
        painter.text(
            at(p) - Vec2::new(0.0, 18.0 * zoom),
            Align2::LEFT_BOTTOM,
            format!("{:?} IR", layer.kind),
            font.clone(),
            Color32::from_gray(150),
        );
    }
    for e in &layer.edges {
        let (Some(from), Some(to)) = (node_of(layer, e.from.node), node_of(layer, e.to.node))
        else {
            continue;
        };
        let (Some(a), Some(b)) = (from.layout, to.layout) else {
            continue;
        };
        let p0 = at(a) + Vec2::new(size.x, size.y * 0.5);
        let p3 = at(b) + Vec2::new(0.0, size.y * 0.5);
        bezier(painter, p0, p3, true, Color32::from_gray(120), zoom);
    }
    for node in &layer.nodes {
        let Some(p) = node.layout else { continue };
        let rect = Rect::from_min_size(at(p), size);
        let found = highlight == Some(node.id);
        painter.rect_filled(rect, 4.0, node_fill(found));
        painter.rect_stroke(
            rect,
            4.0,
            if found {
                Stroke::new(2.0_f32, HIT)
            } else {
                Stroke::new(1.0_f32, Color32::from_gray(90))
            },
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.min + Vec2::splat(6.0 * zoom),
            Align2::LEFT_TOP,
            &node.label,
            font.clone(),
            Color32::from_gray(220),
        );
        painter.text(
            rect.left_bottom() + Vec2::new(6.0 * zoom, -4.0 * zoom),
            Align2::LEFT_BOTTOM,
            ports_line(node),
            FontId::proportional(9.0 * zoom),
            Color32::from_gray(140),
        );
    }
}

fn ports_line(node: &NodeView) -> String {
    format!(
        "{} -> {}",
        node.ports.inputs.join(","),
        node.ports.outputs.join(",")
    )
}

fn paint_cross_edge(
    painter: &egui::Painter,
    graph: &LayeredGraph,
    edge: &CrossEdge,
    at: &impl Fn([f32; 2]) -> Pos2,
    size: Vec2,
    zoom: f32,
) {
    let (Some(a), Some(b)) = (graph.position(edge.from), graph.position(edge.to)) else {
        return;
    };
    let p0 = at(a) + Vec2::new(size.x * 0.5, size.y);
    let p3 = at(b) + Vec2::new(size.x * 0.5, 0.0);
    let colour = Color32::from_rgb(120, 170, 255);
    bezier(painter, p0, p3, false, colour, zoom);
    painter.text(
        p0.lerp(p3, 0.5),
        Align2::CENTER_CENTER,
        &edge.label,
        FontId::proportional(10.0 * zoom),
        colour,
    );
}

/// A cubic between two ports: control points offset along the flow direction — horizontally
/// inside a layer, vertically across layers.
fn bezier(
    painter: &egui::Painter,
    p0: Pos2,
    p3: Pos2,
    horizontal: bool,
    colour: Color32,
    zoom: f32,
) {
    let d = if horizontal {
        Vec2::new(((p3.x - p0.x).abs() * 0.5).max(20.0 * zoom), 0.0)
    } else {
        Vec2::new(0.0, ((p3.y - p0.y).abs() * 0.5).max(20.0 * zoom))
    };
    painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
        [p0, p0 + d, p3 - d, p3],
        false,
        Color32::TRANSPARENT,
        Stroke::new(1.5 * zoom, colour),
    ));
}

fn node_of(layer: &LayerView, id: es_ir::NodeId) -> Option<&NodeView> {
    layer.nodes.iter().find(|n| n.id == id)
}

// --- the edit canvas -------------------------------------------------------------------------

const LINK: Color32 = Color32::from_rgb(200, 180, 90);
const PIN_IN: Color32 = Color32::from_rgb(120, 170, 255);
/// The ring around a search hit, and the colour of a field that does not parse.
const HIT: Color32 = Color32::from_rgb(230, 190, 80);
const BAD: Color32 = Color32::from_rgb(230, 120, 110);

/// The first four bytes of a spec 5.3 content hash: enough to watch one change, and the
/// prefix `es evidence verify` would print. A hash an IR cannot produce says so.
pub(crate) fn short_hash(hash: Result<&[u8; 32], &es_ir::Diagnostic>) -> String {
    match hash {
        Ok([a, b, c, d, ..]) => format!("{a:02x}{b:02x}{c:02x}{d:02x}"),
        Err(d) => format!("unavailable ({})", d.code),
    }
}

/// A node body, selected or found versus plain.
fn node_fill(lit: bool) -> Color32 {
    if lit {
        Color32::from_rgb(60, 72, 96)
    } else {
        Color32::from_rgb(40, 44, 52)
    }
}

/// One parameter's widget; `true` once the person is done with it (Enter, focus lost, a drag
/// released, a variant picked). Which widget a field gets is [`Field::widget`]'s answer and
/// what its text means is [`Field::parse`]'s - neither is decided here.
fn draw_field(ui: &mut egui::Ui, field: &mut Field) -> bool {
    match field.widget() {
        Widget::Checkbox => {
            let mut on = field.flag();
            if ui.checkbox(&mut on, "").changed() {
                field.set_flag(on);
                return true;
            }
            false
        }
        widget @ (Widget::DragInt | Widget::DragFloat) => {
            let mut value = field.number();
            let speed = if widget == Widget::DragInt { 1.0 } else { 0.01 };
            let response = ui.add(egui::DragValue::new(&mut value).speed(speed));
            if response.changed() {
                field.set_number(value);
            }
            response.drag_stopped() || response.lost_focus()
        }
        Widget::Combo(variants) => {
            let mut picked = false;
            egui::ComboBox::from_id_salt(field.name.clone())
                .selected_text(field.text.clone())
                .show_ui(ui, |ui| {
                    for variant in &variants {
                        let value = variant.clone();
                        if ui
                            .selectable_value(&mut field.text, value, variant)
                            .clicked()
                        {
                            picked = true;
                        }
                    }
                });
            picked
        }
        widget @ (Widget::Text | Widget::ShapeText | Widget::TomlText) => ui
            .add(
                egui::TextEdit::singleline(&mut field.text)
                    .hint_text(widget.hint())
                    .desired_width(f32::INFINITY),
            )
            .lost_focus(),
    }
}

/// One frame of canvas geometry, owned. Built from the session, used for hit-testing and
/// painting, and rebuilt after an edit - so the mutation and the drawing never hold a borrow
/// of the session at the same time.
struct CanvasView {
    origin: Pos2,
    zoom: f32,
    positions: BTreeMap<NodeId, [f32; 2]>,
    kinds: BTreeMap<NodeId, &'static str>,
    /// `(inputs, outputs)` as the node itself declares them.
    ports: BTreeMap<NodeId, (Vec<String>, Vec<String>)>,
    edges: Vec<es_ir::Edge>,
}

impl CanvasView {
    fn of(session: &EditSession, origin: Pos2, zoom: f32) -> Self {
        let kinds = edit::kinds_by_id(&session.graph);
        let ports = kinds
            .keys()
            .map(|id| {
                (
                    *id,
                    (
                        session.graph.port_names(*id, es_ir::Dir::In),
                        session.graph.port_names(*id, es_ir::Dir::Out),
                    ),
                )
            })
            .collect();
        Self {
            origin,
            zoom,
            positions: session.layout.positions.clone(),
            kinds,
            ports,
            edges: session.graph.edges().to_vec(),
        }
    }

    fn rect(&self, id: NodeId) -> Option<Rect> {
        let p = self.positions.get(&id)?;
        Some(Rect::from_min_size(
            self.origin + Vec2::new(p[0], p[1]) * self.zoom,
            Vec2::new(NODE_W, NODE_H) * self.zoom,
        ))
    }

    fn names(&self, id: NodeId, input: bool) -> &[String] {
        self.ports.get(&id).map_or(
            &[][..],
            |(i, o)| {
                if input {
                    i.as_slice()
                } else {
                    o.as_slice()
                }
            },
        )
    }

    fn pin(rect: Rect, i: usize, n: usize, input: bool) -> Pos2 {
        let t = (i as f32 + 1.0) / (n as f32 + 1.0);
        Pos2::new(
            if input { rect.left() } else { rect.right() },
            rect.top() + rect.height() * t,
        )
    }

    fn port_pos(&self, port: &PortRef, input: bool) -> Option<Pos2> {
        let rect = self.rect(port.node)?;
        let names = self.names(port.node, input);
        let i = names.iter().position(|n| *n == port.port)?;
        Some(Self::pin(rect, i, names.len(), input))
    }

    /// The port whose pin is under `pos`, if any.
    fn port_at(&self, pos: Pos2, input: bool) -> Option<PortRef> {
        for id in self.kinds.keys() {
            let Some(rect) = self.rect(*id) else { continue };
            let names = self.names(*id, input);
            for (i, name) in names.iter().enumerate() {
                if Self::pin(rect, i, names.len(), input).distance(pos) <= PORT_R * self.zoom {
                    return Some(PortRef::new(*id, name.clone()));
                }
            }
        }
        None
    }

    /// The node whose body is under `pos`, if any. One hit test, shared by the click that
    /// selects and the drag that moves, so the two can never disagree about what was under
    /// the pointer (packet M7/E3).
    fn hit(&self, pos: Pos2) -> Option<NodeId> {
        self.kinds
            .keys()
            .copied()
            .find(|id| self.rect(*id).is_some_and(|rect| rect.contains(pos)))
    }

    /// Output pin first (a wire is pulled from a producer), then the node body, then the
    /// background.
    fn start_drag(&self, pos: Pos2) -> Drag {
        if let Some(from) = self.port_at(pos, false) {
            return Drag::Link { from };
        }
        let Some(id) = self.hit(pos) else {
            return Drag::Pan;
        };
        let rect = self.rect(id).unwrap_or(Rect::NOTHING);
        Drag::Node {
            id,
            origin: self.positions.get(&id).copied().unwrap_or_default(),
            grab: pos - rect.min,
        }
    }

    fn paint(&self, painter: &egui::Painter, selected: Option<NodeId>) {
        for edge in &self.edges {
            let (Some(p0), Some(p3)) = (
                self.port_pos(&edge.from, false),
                self.port_pos(&edge.to, true),
            ) else {
                continue;
            };
            bezier(painter, p0, p3, true, Color32::from_gray(140), self.zoom);
        }
        let font = FontId::proportional(12.0 * self.zoom);
        for (id, kind) in &self.kinds {
            let Some(rect) = self.rect(*id) else { continue };
            painter.rect_filled(rect, 4.0, node_fill(selected == Some(*id)));
            painter.rect_stroke(
                rect,
                4.0,
                Stroke::new(1.0_f32, Color32::from_gray(110)),
                egui::StrokeKind::Inside,
            );
            painter.text(
                rect.min + Vec2::splat(6.0 * self.zoom),
                Align2::LEFT_TOP,
                format!("{kind} #{}", id.0),
                font.clone(),
                Color32::from_gray(220),
            );
            for (input, colour) in [(true, PIN_IN), (false, LINK)] {
                let n = self.names(*id, input).len();
                for i in 0..n {
                    painter.circle_filled(
                        Self::pin(rect, i, n, input),
                        PORT_R * 0.5 * self.zoom,
                        colour,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CanvasView, Drag, NODE_H, NODE_W};

    use std::path::Path;

    use egui::{Pos2, Vec2};
    use es_ir::serial::Layout;
    use es_ir::NodeId;

    use crate::model::edit::{EditIr, EditSession};

    /// The canvas geometry over the demo bundle's Task IR with one node put somewhere known.
    /// `CanvasView` is plain data - positions, rectangles and names - so it needs no display.
    fn view_with_one_node_at(origin: Pos2, zoom: f32, pos: [f32; 2]) -> (CanvasView, NodeId) {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/visible-learning/task.toml");
        let toml = std::fs::read_to_string(&path).expect("the demo bundle's task.toml");
        let task = es_ir::serial::task_from_toml(&toml).expect("it parses");
        let mut session = EditSession::new(EditIr::Task(task), Layout::default());
        let (id, _) = session.graph.nodes()[0];
        session.layout.positions.insert(id, pos);
        (CanvasView::of(&session, origin, zoom), id)
    }

    /// Packet M7/E3: a plain click has to reach the inspector, and it does that through the
    /// same hit test the drag uses. A node's body is hit, the background is not, and the two
    /// callers agree.
    #[test]
    fn a_click_hits_the_node_under_it_and_nothing_on_the_background() {
        for (origin, zoom) in [
            (Pos2::ZERO, 1.0_f32),
            (Pos2::new(7.0, 11.0), 2.0),
            (Pos2::new(-40.0, 25.0), 0.5),
        ] {
            let at = [100.0_f32, 50.0_f32];
            let (view, id) = view_with_one_node_at(origin, zoom, at);
            let corner = origin + Vec2::new(at[0], at[1]) * zoom;
            let centre = corner + Vec2::new(NODE_W, NODE_H) * zoom * 0.5;

            assert_eq!(view.hit(centre), Some(id), "the body at {origin:?}/{zoom}");
            assert_eq!(view.hit(corner + Vec2::splat(1.0)), Some(id), "just inside");
            assert_eq!(view.hit(corner - Vec2::splat(1.0)), None, "just outside");
            assert_eq!(view.hit(corner + Vec2::new(0.0, 4000.0)), None, "far below");
            // Every other node is without a position, so nothing else can be hit.
            assert_eq!(
                view.hit(origin + Vec2::splat(-9999.0)),
                None,
                "empty canvas"
            );

            // The drag and the click read the same geometry: dragging from the centre grabs
            // the node the click would have selected, and the background pans.
            assert!(
                matches!(view.start_drag(centre), Drag::Node { id: dragged, .. } if dragged == id),
                "the drag grabs what the click selects"
            );
            assert!(matches!(
                view.start_drag(corner - Vec2::splat(1.0)),
                Drag::Pan
            ));
        }
    }
}
