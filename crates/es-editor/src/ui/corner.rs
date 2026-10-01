//! The policy camera's view in the viewport's corner (packet M17/G6, `docs/design/scene-authoring.md`
//! section 5): the camera the policy is given, at its declared size and render path, drawn in
//! process ([`crate::gpu::Sensor`]), over ① of a template and of an editable project alike.
//!
//! Drawing only: which cameras the policy sees, how each is declared and which one is shown are
//! [`es_editor_scene::policy`]'s, under test.

use std::sync::Arc;

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use es_assets::scene::SceneDesc;
use es_editor_scene::policy::shown;
use es_editor_scene::{Entity, PolicyCamera};
use es_ir::task::SensorPath;

use crate::gpu::{Health, Sensor, SensorJob};
use crate::model::i18n::{fill, t, Lang};

/// The share of the viewport's height the corner's picture takes, and its bounds in points.
const SHARE: f32 = 0.3;
const SMALLEST: f32 = 96.0;
const LARGEST: f32 = 260.0;
const MARGIN: f32 = 8.0;

/// The corner between frames: the cameras and the scene revision they were read for, the
/// device's frames, and the texture of the newest.
#[derive(Default)]
pub(crate) struct Corner {
    cameras: Option<(usize, Result<Vec<PolicyCamera>, String>)>,
    sensor: Sensor,
    texture: Option<(u64, egui::TextureHandle)>,
}

/// `text` in `colour` on a dark plate, placed by `align` at `at`: readable over any picture.
pub(crate) fn tag(painter: &egui::Painter, at: Pos2, align: Align2, text: String, colour: Color32) {
    let galley = painter.layout_no_wrap(text, FontId::proportional(13.0), colour);
    let plate = align.anchor_size(at, galley.size()).expand(3.0);
    painter.rect_filled(plate, 3.0, Color32::from_black_alpha(170));
    painter.galley(plate.min + Vec2::splat(3.0), galley, colour);
}

impl Corner {
    /// The shown camera's newest frame in `rect`'s lower right corner, its name and how it is
    /// drawn above it. `revision` is the scene's: the cameras are read again (`cameras`) and the
    /// frame drawn again when it moves. A policy that sees no camera has no corner.
    #[allow(clippy::too_many_arguments)] // the canvas, the scene and its selection
    pub(crate) fn paint(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        lang: Lang,
        scene: &Arc<SceneDesc>,
        revision: usize,
        selected: Option<&Entity>,
        cameras: impl FnOnce() -> Result<Vec<PolicyCamera>, String>,
    ) {
        if self.cameras.as_ref().is_none_or(|(r, _)| *r != revision) {
            self.cameras = Some((revision, cameras()));
        }
        let Some((_, listed)) = &self.cameras else {
            return;
        };
        let grey = Color32::from_gray(230);
        let say = |at: Pos2, align: Align2, text: String| tag(painter, at, align, text, grey);
        let corner = rect.right_bottom() - Vec2::splat(MARGIN);
        let cams = match listed {
            Ok(cams) => cams,
            Err(why) => {
                say(
                    corner,
                    Align2::RIGHT_BOTTOM,
                    fill(lang, "corner.failed", &[why]),
                );
                return;
            }
        };
        let Some(cam) = shown(cams, selected) else {
            return;
        };
        let key = (revision, cam.name.clone());
        let job = || {
            let cfg = es_env::render::sensor_cfg(cam.sensor, &cam.image, &cam.render, None);
            Some(SensorJob {
                scene: Arc::clone(scene),
                cfg,
            })
        };
        if !self.sensor.update(&key, job) {
            let why = match crate::gpu::health() {
                Some(Health::Failed(why)) => why,
                _ => String::new(),
            };
            say(
                corner,
                Align2::RIGHT_BOTTOM,
                fill(lang, "corner.no_device", &[&why]),
            );
            return;
        }

        let (w, h) = (
            cam.image.width.max(1) as f32,
            cam.image.height.max(1) as f32,
        );
        let tall = (rect.height() * SHARE).clamp(SMALLEST, LARGEST);
        let size = Vec2::new(tall * w / h, tall);
        let frame = Rect::from_min_size(corner - size, size);
        let how = match cam.render.path {
            SensorPath::Rs => t(lang, "corner.rs").to_owned(),
            SensorPath::Pt { spp, .. } => fill(lang, "corner.pt", &[&spp.to_string()]),
        };
        let (wide, high) = (cam.image.width.to_string(), cam.image.height.to_string());
        let title = fill(lang, "corner.title", &[&cam.name, &wide, &high, &how]);
        say(
            frame.right_top() - Vec2::new(0.0, 4.0),
            Align2::RIGHT_BOTTOM,
            title,
        );
        painter.rect_filled(frame, 0.0, Color32::from_gray(10));
        let newest = self.sensor.newest(&key);
        if let Some((Ok(picture), _, revision)) = newest {
            if self.texture.as_ref().is_none_or(|(r, _)| *r != revision) {
                let px = [picture.width as usize, picture.height as usize];
                let image = egui::ColorImage::from_rgb(px, &picture.rgb);
                // Nearest: the policy's own pixels, not a smoothed enlargement of them.
                let options = egui::TextureOptions::NEAREST;
                if let Some((r, texture)) = &mut self.texture {
                    texture.set(image, options);
                    *r = revision;
                } else {
                    let texture = painter.ctx().load_texture("corner", image, options);
                    self.texture = Some((revision, texture));
                }
            }
        }
        if let (Some((Ok(_), ..)), Some((_, texture))) = (newest, &self.texture) {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(texture.id(), frame, uv, Color32::WHITE);
        }
        painter.rect_stroke(
            frame,
            0.0,
            Stroke::new(1.0_f32, Color32::from_gray(200)),
            egui::StrokeKind::Outside,
        );
        match newest {
            Some((Err(why), true, _)) => {
                say(
                    frame.center(),
                    Align2::CENTER_CENTER,
                    fill(lang, "corner.failed", &[why]),
                );
            }
            Some((_, true, _)) => {}
            _ => {
                say(
                    frame.center(),
                    Align2::CENTER_CENTER,
                    t(lang, "corner.drawing").to_owned(),
                );
                painter
                    .ctx()
                    .request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
    }
}
