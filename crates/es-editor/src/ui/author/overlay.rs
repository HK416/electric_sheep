//! Over ①'s viewport (packet M17/G6): a click picks what is under it, a handle is dragged and
//! let go as one command, the selection is tinted and its handles drawn, and the policy camera's
//! view is shown in the corner.

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use es_editor_scene::policy;
use es_editor_scene::view::{project, ray};
use es_editor_scene::{Camera, Gizmo, PolicyCamera, SceneModel, Step, Tool};
use es_math::units::RAD_TO_DEG;
use es_math::Vec3;

use super::{Author, Held};
use crate::model::i18n::Lang;
use crate::ui::corner;

/// The X, Y and Z handles, and the one held or under the pointer.
const AXES: [Color32; 3] = [
    Color32::from_rgb(230, 70, 60),
    Color32::from_rgb(90, 200, 80),
    Color32::from_rgb(70, 130, 240),
];
pub(super) const HOT: Color32 = Color32::from_rgb(250, 210, 60);
/// The selection's tint, drawn over everything (an x-ray, not a depth-tested outline).
const TINT: Color32 = Color32::from_rgba_premultiplied(100, 64, 12, 96);
/// How near a handle the pointer must be to take it, in points.
const REACH: f64 = 8.0;

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

impl Author {
    /// The held handle let go: its one command, applied (snapped as the row says).
    pub(super) fn let_go(&mut self) {
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
