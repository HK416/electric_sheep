//! Edit mode's canvas (spec 23.4 stage 2): one `EditSession` over the Task IR, every gesture one
//! `Edit`, and the owned geometry it hit-tests and paints between them, `CanvasView`.

use std::collections::BTreeMap;

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use es_ir::graph::PortRef;
use es_ir::NodeId;

use super::graph::{bezier, node_fill, NODE_H, NODE_W};
use crate::app::EditorApp;
use crate::model::edit::{self, Edit, EditSession};
use crate::model::i18n;

/// Click radius of a port, in graph units.
const PORT_R: f32 = 7.0;

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
    /// Edit mode (spec 23.4 stage 2). Every branch below ends in exactly one
    /// [`EditSession::apply`], [`EditSession::undo`] or [`EditSession::redo`] call - this
    /// function decides nothing else, which is what keeps the untested half thin.
    pub(super) fn edit_canvas(&mut self, ui: &mut egui::Ui) {
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
}

const LINK: Color32 = Color32::from_rgb(200, 180, 90);
const PIN_IN: Color32 = Color32::from_rgb(120, 170, 255);

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
