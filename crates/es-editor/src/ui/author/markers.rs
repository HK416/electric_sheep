//! Cameras and regions drawn over the viewport as lines (packet M17/G7): a camera's frustum, a
//! region's box, the selected one lit. Where the lines are and what a click near them picks are
//! `es_editor_scene::marker`'s.

use eframe::egui;
use egui::{Color32, Pos2, Stroke};
use es_editor_scene::{Camera, Entity};

use super::overlay::HOT;
use super::Author;

const CAMERA: Color32 = Color32::from_rgb(120, 200, 240);
const REGION: Color32 = Color32::from_rgb(120, 230, 150);

impl Author {
    /// Every camera's and region's lines as `view` sees them; `pt` puts a pixel on the canvas.
    pub(super) fn paint_markers(
        &self,
        painter: &egui::Painter,
        view: &Camera,
        pt: impl Fn([f64; 2]) -> Pos2,
    ) {
        let selected = self.model.selection();
        for (e, lines) in self.model.markers(view) {
            let colour = match &e {
                _ if Some(&e) == selected => HOT,
                Entity::Camera(_) => CAMERA,
                _ => REGION,
            };
            for [a, b] in lines {
                painter.line_segment([pt(a), pt(b)], Stroke::new(1.5_f32, colour));
            }
        }
    }
}
