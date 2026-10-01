//! The scene canvas (packets M7/E8, M16/H8, M16/H9): a template's scene or a replay at its tick
//! under the look selector, a drag to orbit and the wheel to zoom, and ①'s overlay drawn over it.
//! The Replay panel, ①, ② and the results screen's player all draw with it.

use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Vec2};
use es_render::raster::{Camera, Projected, Raster, BACKGROUND};

use crate::model::i18n::{self, Lang};
use crate::model::replay_view::ReplayView;
use crate::model::scene_view::ScenePreview;
use crate::model::viewport::{Mode, Shot, Viewport};

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
    /// The in-process device's frames (packet M16/H9); `look` is the fallback without one.
    gpu: crate::gpu::Look,
    /// `--orbit-demo` / `--fps`: frames shown since `.0`, and their worker milliseconds.
    meter: Option<(std::time::Instant, u32, f32)>,
}

/// What a canvas shows: a template's scene at its initial pose, or a replay at its tick.
#[derive(Clone, Copy)]
pub(crate) enum Posed<'a> {
    Scene(&'a ScenePreview),
    Replay(&'a ReplayView),
}

impl Posed<'_> {
    /// A still scene's slot is its revision: an edit of ①'s document is a new picture.
    fn tick(self) -> usize {
        match self {
            Self::Scene(preview) => preview.revision(),
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

    fn source(self) -> crate::model::viewport::Source {
        match self {
            Self::Scene(preview) => preview.source(),
            Self::Replay(view) => view.source_at(view.tick),
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

/// `es-editor --fps` (`orbit: false`) and `--orbit-demo` (`true`): every viewport prints the
/// frames it showed each second, and with `orbit` turns its camera a little every frame - a
/// drag without a mouse, to measure and capture the look while it moves (packet M16/H9).
pub fn set_demo(ctx: &egui::Context, orbit: bool) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("viewport-demo"), orbit));
}

fn demo(ctx: &egui::Context) -> Option<bool> {
    ctx.data(|d| d.get_temp(egui::Id::new("viewport-demo")))
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
    scene_canvas(ui, lang, size, Posed::Replay(view), camera, canvas, None);
}

/// What ① draws over its viewport (packet M17/G6): the canvas, its painter and the camera at
/// the picture's size; `true` while it holds the pointer, so the drag does not also turn the view.
pub(crate) type Overlay<'a> = &'a mut dyn FnMut(&egui::Response, &egui::Painter, &Camera) -> bool;

/// [`replay_canvas`] for any posed scene, under the look selector (packet M16/H8). ① and ②
/// show a template's scene with it (packet M12/Y15), and draw their `overlay` over it.
pub(crate) fn scene_canvas(
    ui: &mut egui::Ui,
    lang: Lang,
    size: Vec2,
    posed: Posed<'_>,
    camera: &mut Camera,
    canvas: &mut Canvas,
    overlay: Option<Overlay<'_>>,
) {
    let mut mode = viewport_mode(ui.ctx());
    let row = ui.horizontal(|ui| {
        ui.label(i18n::t(lang, "viewport.label"));
        for look in Mode::ALL {
            ui.selectable_value(&mut mode, look, i18n::t(lang, look.key()))
                .on_hover_text(i18n::t(lang, look.hint()));
        }
        // The look's own line first (it is what changes), then which device draws it.
        let health = crate::gpu::health();
        let status = match &health {
            Some(crate::gpu::Health::Failed(_)) => canvas.look.status(),
            _ => canvas.gpu.status(mode, &posed.shot(camera)),
        };
        if let Some(status) = status {
            let (key, args) = status.line();
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            ui.weak(i18n::fill(lang, key, &args));
        }
        match health {
            Some(crate::gpu::Health::Failed(why)) => {
                ui.weak(i18n::fill(lang, "viewport.cpu_fallback", &[&why]));
            }
            Some(crate::gpu::Health::Ready { device, bytes }) => {
                let mb = (bytes / (1 << 20)).to_string();
                ui.weak(i18n::fill(lang, "viewport.gpu", &[&device, &mb]));
            }
            _ => {}
        }
    });
    set_viewport_mode(ui.ctx(), mode);
    let gap = row.response.rect.height() + ui.spacing().item_spacing.y;
    let size = Vec2::new(size.x, (size.y - gap).max(1.0));
    let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    (camera.width, camera.height) =
        Raster::size_for([response.rect.width(), response.rect.height()]);
    painter.rect_filled(response.rect, 0.0, Color32::from_gray(BACKGROUND));
    // The picture goes under whatever the overlay draws: its place is kept, filled below.
    let picture_at = painter.add(egui::Shape::Noop);
    let held = overlay.is_some_and(|o| o(&response, &painter, camera));
    if response.dragged() && !held {
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
    let demo = demo(ui.ctx());
    if demo == Some(true) {
        *camera = camera.orbit(0.01, 0.0);
    }
    let shot = posed.shot(camera);
    let gpu = canvas.gpu.update(mode, &shot, || posed.source());
    let picture = if gpu {
        canvas
            .gpu
            .newest()
            .map(|(f, r)| (&f.picture, r, Some(f.ms)))
    } else {
        let es = (canvas.es).get_or_insert_with(|| crate::model::launch::es_binary().path);
        let now = std::time::Instant::now();
        canvas.look.update(mode, &shot, now, es, || posed.tris());
        canvas.look.picture().map(|(p, r)| (p, r, None))
    };
    let texture = if let Some((picture, revision, ms)) = picture {
        if canvas.picture.as_ref().is_none_or(|(r, _)| *r != revision) {
            let px = [picture.width as usize, picture.height as usize];
            let image = egui::ColorImage::from_rgb(px, &picture.rgb);
            let options = egui::TextureOptions::LINEAR;
            // One texture per viewport, rewritten: a new one per frame of a drag would allocate
            // on the GL side every frame.
            if let Some((r, texture)) = &mut canvas.picture {
                texture.set(image, options);
                *r = revision;
            } else {
                let texture = ui.ctx().load_texture("viewport-look", image, options);
                canvas.picture = Some((revision, texture));
            }
            if let (Some(_), Some(ms)) = (demo, ms) {
                meter(&mut canvas.meter, mode, picture, ms);
            }
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
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let image = egui::Shape::image(texture.id(), response.rect, uv, Color32::WHITE);
        painter.set(picture_at, image);
    }
    if (gpu && canvas.gpu.busy(mode, &shot)) || demo == Some(true) {
        ui.ctx().request_repaint();
    } else if !gpu && canvas.look.busy() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(50));
    }
}

/// `--fps`: one line a second on stderr - frames shown, and the worker's mean milliseconds.
fn meter(
    meter: &mut Option<(std::time::Instant, u32, f32)>,
    mode: Mode,
    picture: &crate::model::viewport::Picture,
    ms: f32,
) {
    let (since, frames, total) = meter.get_or_insert((std::time::Instant::now(), 0, 0.0));
    *frames += 1;
    *total += ms;
    let secs = since.elapsed().as_secs_f32();
    if secs >= 1.0 {
        eprintln!(
            "viewport {mode:?} {}x{}: {:.1} frames/s shown, {:.1} ms per frame on the device, {} spp",
            picture.width,
            picture.height,
            *frames as f32 / secs,
            *total / *frames as f32,
            picture.samples
        );
        *meter = None;
    }
}
