//! The Design graph pane (spec 23.2, spec 23.4): its Edit switch with Save, Undo and Redo, the
//! search box, the read-only canvas of all four IRs, the parameter inspector while editing, and
//! the Problems pane's list. Edit mode's canvas is `edit_canvas`'s.

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use es_ir::NodeId;

use crate::app::EditorApp;
use crate::model::graph_view::{CrossEdge, LayerView, LayeredGraph, NodeView};
use crate::model::i18n;
use crate::model::inspector::{Field, Inspector, Widget};
use crate::model::search::Search;

pub(super) const NODE_W: f32 = 178.0;
pub(super) const NODE_H: f32 = 40.0;

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
            crate::ui::shell::nothing_open(self, ui);
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
pub(super) fn bezier(
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

/// The ring around a search hit, and the colour of a field that does not parse.
const HIT: Color32 = Color32::from_rgb(230, 190, 80);
const BAD: Color32 = Color32::from_rgb(230, 120, 110);

/// A node body, selected or found versus plain.
pub(super) fn node_fill(lit: bool) -> Color32 {
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
