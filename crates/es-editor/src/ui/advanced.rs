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
//!
//! One child module per pane: `graph` (its edit mode in `edit_canvas`), `run` (with the
//! `launch` section and the `replay` panel), `live` and `sees`. `viewport` is the scene canvas
//! the Replay panel, ①, ② and the results screen draw with.

use eframe::egui;
use es_eval::run_dir::Rgb8Image;

mod edit_canvas;
mod graph;
mod launch;
mod live;
mod replay;
mod run;
mod sees;
mod viewport;

pub(crate) use edit_canvas::Drag;
pub(crate) use live::{paint_curve, RL_COLOURS};
pub(crate) use replay::REPLAY_RATE_HZ;
pub(crate) use run::{metric_text, paint_timeline};
pub(crate) use viewport::{replay_canvas, scene_canvas, Canvas, Posed};
pub use viewport::{set_demo, set_viewport_mode};

/// Where the replay camera starts; defined beside the scene preview it is tested with.
pub(crate) use crate::model::scene_view::SHOWCASE_CAMERA;

pub(crate) fn rgb_texture(ctx: &egui::Context, name: &str, img: &Rgb8Image) -> egui::TextureHandle {
    let color = egui::ColorImage::from_rgb([img.width, img.height], &img.data);
    ctx.load_texture(name, color, egui::TextureOptions::NEAREST)
}

/// The first four bytes of a spec 5.3 content hash: enough to watch one change, and the
/// prefix `es evidence verify` would print. A hash an IR cannot produce says so.
pub(crate) fn short_hash(hash: Result<&[u8; 32], &es_ir::Diagnostic>) -> String {
    match hash {
        Ok([a, b, c, d, ..]) => format!("{a:02x}{b:02x}{c:02x}{d:02x}"),
        Err(d) => format!("unavailable ({})", d.code),
    }
}
