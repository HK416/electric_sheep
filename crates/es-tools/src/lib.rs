//! `es-tools` (layer 11): the `es` verbs that read a finished artifact rather than run one.
//!
//! `video` / `showcase` (frames), `backend` (a scene file), `evidence` (an `.esb` bundle),
//! `gap` (two dataset roots) and `bench` (a scene and an Observation IR) all take something
//! already on disk and report on it; none of them own a loop. They lived in `crates/es/src/cmd`
//! until packet `docs/packets/M10/W3b-es-tools-split.md` moved them here to bring `es` under
//! the spec 1.5 line target. [`error::CliError`] and [`util::hex`] came with them and `es`
//! re-exports all of it under the old module paths, so `crate::cmd::backend::load_scene`,
//! `crate::error::CliError` and the rest keep resolving and no caller changed.
//!
//! `es` stays the only binary (spec 26.2): this crate is a library it dispatches into.
//!
//! Layer rule (spec 4.2): layer 11, so it may depend on `es-data` / `es-eval` / `es-telemetry`
//! (layer 10) and below, and never on `es-editor` (layer 12).
#![forbid(unsafe_code)]

pub mod backend;
pub mod bench;
pub mod error;
pub mod evidence;
pub mod gap;
/// `es video showcase` (packet M5/V9): only with the `render` feature, because it is the one
/// subcommand that opens a Vulkan device.
#[cfg(feature = "render")]
pub mod showcase;
pub mod util;
pub mod video;

/// One image channel of a Task IR: its name, frame, declared `ImageSpec` and the `render` its
/// sensor declares (spec 7.4, packet M7/R5).
#[cfg(feature = "render")]
pub type ImageChannel = (
    String,
    es_ir::types::Frame,
    es_ir::image::ImageSpec,
    es_ir::task::SensorRender,
);

/// Every image channel of the Task IR's `ObservationSpec`, in channel-name order. A channel
/// whose source is not a sensor has no render declaration and gets the default one -- the
/// rasterizer, which is what it got before this field existed.
#[cfg(feature = "render")]
fn image_channels(task: &es_ir::task::TaskIr) -> Vec<ImageChannel> {
    task.observation_spec
        .channels
        .iter()
        .filter_map(|(name, c)| {
            let render = match c.source {
                es_ir::task::ObsSource::Sensor { render, .. } => render,
                _ => es_ir::task::SensorRender::default(),
            };
            c.ty.image
                .map(|spec| (name.clone(), c.ty.frame, spec, render))
        })
        .collect()
}

/// The one image channel a render is of -- `es video showcase` renders one view.
///
/// Lives here, beside [`frame_cameras`], so the showcase and the two `--frames` commands
/// cannot disagree about which channel is being rendered or about how.
#[cfg(feature = "render")]
pub fn image_channel(task: &es_ir::task::TaskIr) -> Result<ImageChannel, error::CliError> {
    let mut images = image_channels(task);
    if images.len() != 1 {
        return Err(error::CliError::Runtime(format!(
            "--frames needs exactly one image channel in the Task IR's ObservationSpec; it \
             declares {}",
            images.len()
        )));
    }
    Ok(images.remove(0))
}

/// The cameras `es loop collect --frames` and `es eval run --frames` render: every image
/// channel of the Task IR, by name, each from the camera its `Frame` names at the `ImageSpec`
/// it declares, on the path its sensor declares (packet M7/R5) -- there is no flag for either,
/// because the document decides, and a scene whose camera produces something else is refused
/// rather than resampled (`INV-14`).
///
/// With `frames`, each camera writes its frames there: one channel straight into `frames`
/// (today's layout, byte for byte), several into `<frames>/<channel>/` each, the layout the
/// `LeRobot` v3 export reads one camera per directory from (packet M15/N2).
#[cfg(feature = "render")]
pub fn frame_cameras(
    task: &es_ir::task::TaskIr,
    frames: Option<&std::path::Path>,
) -> Result<Vec<(String, es_env::EnvRendererCfg)>, error::CliError> {
    let images = image_channels(task);
    if images.is_empty() {
        return Err(error::CliError::Runtime(
            "--frames needs an image channel in the Task IR's ObservationSpec; it declares none"
                .to_owned(),
        ));
    }
    let several = images.len() > 1;
    images
        .into_iter()
        .map(|(name, frame, spec, render)| {
            let es_ir::types::Frame::Camera(camera) = frame else {
                return Err(error::CliError::Runtime(format!(
                    "image channel {name:?} is not in a camera frame, so there is no camera to \
                     render it from"
                )));
            };
            let dir = frames.map(|d| {
                if several {
                    d.join(&name)
                } else {
                    d.to_path_buf()
                }
            });
            let cfg = es_env::render::sensor_cfg(camera, &spec, &render, dir);
            Ok((name, cfg))
        })
        .collect()
}
