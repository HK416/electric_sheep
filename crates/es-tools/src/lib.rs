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

/// The one image channel a `--frames` render is of: its name, frame, declared `ImageSpec` and
/// the `render` its sensor declares (spec 7.4, packet M7/R5).
///
/// Lives here and is used by `es eval run --frames`, `es loop collect --frames` and
/// `es video showcase`, so the three cannot disagree about which channel is being rendered or
/// about how. A channel whose source is not a sensor has no render declaration and gets the
/// default one -- the rasterizer, which is what it got before this field existed.
#[cfg(feature = "render")]
pub fn image_channel(
    task: &es_ir::task::TaskIr,
) -> Result<
    (
        String,
        es_ir::types::Frame,
        es_ir::image::ImageSpec,
        es_ir::task::SensorRender,
    ),
    error::CliError,
> {
    let images: Vec<_> = task
        .observation_spec
        .channels
        .iter()
        .filter_map(|(name, c)| c.ty.image.map(|spec| (name, c, spec)))
        .collect();
    let [(name, channel, spec)] = images.as_slice() else {
        return Err(error::CliError::Runtime(format!(
            "--frames needs exactly one image channel in the Task IR's ObservationSpec; it \
             declares {}",
            images.len()
        )));
    };
    let render = match channel.source {
        es_ir::task::ObsSource::Sensor { render, .. } => render,
        _ => es_ir::task::SensorRender::default(),
    };
    Ok(((*name).clone(), channel.ty.frame, *spec, render))
}
