//! The renderer in the env loop (spec 15, `docs/design/visible-learning.md` section 7).
//!
//! Behind the off-by-default `render` feature, because it is the only thing in `es-env` that
//! links Vulkan: with the feature off this module does not exist and the crate builds exactly
//! as it did before (`es-runtime-embedded` and `es-ros2` never reach a GPU).
//!
//! What it adds is the wiring `es-render` deliberately does not have (it is layer 5 and knows
//! nothing of the physics state, spec 4.2):
//!
//! ```text
//! StateView.xpos/xquat ──▶ body_poses ──▶ TriScene::from_scene_with_poses ──▶ Renderer
//!                       └▶ camera_view (the camera's own body, when it has one)
//! ```
//!
//! Three deliberate limits, each stated rather than hidden:
//!
//! * the whole scene is re-tessellated and re-uploaded per frame (design note section 7.1);
//! * one camera, because `MultiViewPack` is rejected by plan lowering
//!   (`crates/es-compile/src/plan.rs:545`);
//! * the declared `ImageSpec` is **checked**, never resampled to fit ([`EnvRenderer::check`],
//!   spec 7.2, spec 26.1, `INV-14`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_ir::image::{CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageSpec};
use es_math::{Pose, Quat, Vec3};
use es_physics_core::backend::{ModelInfo, StateView};
use es_render::{CameraView, Channel, RenderConfig, RenderPath, Renderer, Tile, TileAtlasCfg};

use crate::EnvError;

/// `T_opencv_mujoco`: a rotation of pi about `+X`.
///
/// An MJCF camera looks along `-Z` with image-up along `+Y`; `es-render` reads the `OpenCV`
/// frame of spec 3.1 (`+Z` forward, `+Y` down). The importer stores the MJCF pose verbatim
/// (`crates/es-assets/src/mjcf/mod.rs:778`), so the conversion belongs here, once.
const MJCF_TO_OPENCV: Quat = Quat::from_xyzw(1.0, 0.0, 0.0, 0.0);

/// What to render, per env step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvRendererCfg {
    /// The scene camera to render from (`SceneDesc::cameras`).
    pub camera: StableId,
    pub width: u32,
    pub height: u32,
    pub channel: Channel,
    pub path: RenderPath,
    /// When set, every [`EnvRenderer::frame`] also writes `<dir>/<NNNNNN>.bin` + `.json`.
    pub frames_dir: Option<PathBuf>,
}

impl EnvRendererCfg {
    /// The `Rs` path on one camera, no frames on disk.
    pub fn rgb(camera: StableId, width: u32, height: u32) -> Self {
        Self {
            camera,
            width,
            height,
            channel: Channel::Rgb8,
            path: RenderPath::Rs,
            frames_dir: None,
        }
    }
}

/// The [`RenderConfig`] a [`EnvRendererCfg`] means. One function, so the GPU renderer and the
/// CPU reference (`es_render::cpu`) cannot drift apart — every golden and the GPU/CPU
/// comparison depend on them being the same config.
pub fn render_config(cfg: &EnvRendererCfg) -> RenderConfig {
    let atlas = TileAtlasCfg::row(cfg.width, cfg.height, 1);
    let mut out = match cfg.path {
        RenderPath::Rs => RenderConfig::rs(atlas),
        RenderPath::Pt {
            spp,
            bounces,
            restir,
            svgf,
        } => {
            let mut c = RenderConfig::pt(atlas, spp, bounces);
            c.path = RenderPath::Pt {
                spp,
                bounces,
                restir,
                svgf,
            };
            c
        }
    };
    // Only the channel the caller asked for: every other one is an atlas-sized buffer nobody
    // reads back.
    out.channels = BTreeSet::from([cfg.channel]);
    out
}

/// World pose of every body the model indexes, read out of one env's `xpos` / `xquat` rows.
///
/// A body the backend does not report — an empty `xpos`, a short row — is simply absent from
/// the map, and [`es_render::TriScene::from_scene_with_poses`] then keeps its scene pose. A
/// partially known state degrades to the static scene, never to the origin.
pub fn body_poses(model: &ModelInfo, state: &StateView<'_>, env: u32) -> BTreeMap<StableId, Pose> {
    let nbody = model.nbody as usize;
    let base = env as usize * nbody;
    let mut out = BTreeMap::new();
    for (id, range) in &model.body {
        let row = base + range.start as usize;
        let (Some(p), Some(q)) = (
            state.xpos.get(row * 3..row * 3 + 3),
            state.xquat.get(row * 4..row * 4 + 4),
        ) else {
            continue;
        };
        out.insert(
            *id,
            Pose::new(
                Vec3::new(p[0], p[1], p[2]),
                Quat::from_xyzw(q[0], q[1], q[2], q[3]),
            ),
        );
    }
    out
}

/// The scene camera `cfg.camera`, in the `OpenCV` frame `es-render` renders from.
///
/// A camera bolted to a body (a wrist camera) follows that body's pose from `world`; a camera
/// fixed to the world does not move. The intrinsics come from the camera's own `fovy` at the
/// requested size — nothing is rescaled here (`INV-14`).
pub fn camera_view(
    scene: &SceneDesc,
    cfg: &EnvRendererCfg,
    world: &BTreeMap<StableId, Pose>,
) -> Result<CameraView, EnvError> {
    let camera = scene
        .cameras
        .iter()
        .find(|c| c.id == cfg.camera)
        .ok_or_else(|| {
            EnvError::Task(format!(
                "camera {} is not in scene \"{}\"",
                cfg.camera, scene.name
            ))
        })?;
    let attached = camera
        .body
        .and_then(|b| world.get(&b).copied())
        .unwrap_or(Pose::IDENTITY);
    let pose = attached
        .compose(camera.pose)
        .compose(Pose::new(Vec3::ZERO, MJCF_TO_OPENCV));
    Ok(CameraView {
        pose,
        spec: es_render::ImageSpec::pinhole(cfg.width, cfg.height, camera.fovy),
    })
}

/// One camera rendered from a running env's state, frame after frame.
///
/// A concrete struct, not a trait: `INV-17` allows seven extension points and this is none of
/// them. It is not stored inside [`Env`](crate::Env) either — the `&Gpu` borrow would put a
/// lifetime on `Env<B>`, which `es-data` and `es-eval` name — so the caller owns it and hands
/// it the state it already has.
#[derive(Debug)]
pub struct EnvRenderer<'gpu> {
    renderer: Renderer<'gpu>,
    cfg: EnvRendererCfg,
    /// Kept whole: every frame re-poses the bodies and re-tessellates (design note 7.1).
    scene: SceneDesc,
    frame: u64,
}

impl<'gpu> EnvRenderer<'gpu> {
    pub fn new(
        gpu: &'gpu es_gpu::Gpu,
        scene: &SceneDesc,
        cfg: EnvRendererCfg,
    ) -> Result<Self, EnvError> {
        let renderer = Renderer::new(gpu, render_config(&cfg))
            .map_err(|e| EnvError::Unsupported(format!("renderer: {e}")))?;
        // Fail at construction rather than on the first frame: a camera the scene does not
        // have is a configuration error, not a run-time one.
        camera_view(scene, &cfg, &BTreeMap::new())?;
        Ok(Self {
            renderer,
            cfg,
            scene: scene.clone(),
            frame: 0,
        })
    }

    /// The renderer's `ImageSpec` subset, in Observation IR terms.
    pub fn image_spec(&self) -> ImageSpec {
        image_spec(&self.scene, &self.cfg).expect("the camera was resolved in `new`")
    }

    /// [`check_image_spec`] against this renderer's own spec.
    pub fn check(&self, declared: &ImageSpec) -> Result<(), EnvError> {
        check_image_spec(&self.image_spec(), declared)
    }

    /// Re-poses the scene from `state`'s `xpos` / `xquat` for `env`, renders it, and returns
    /// the tile. Writes `<frames_dir>/<NNNNNN>.bin` + `.json` when the config asks for it.
    ///
    /// Determinism (spec 3.5): same state, same device, same bits — the tessellation is a pure
    /// function of the poses, and the kernels carry the deterministic execution modes
    /// `es-gpu` compiled them with (spec 3.4).
    pub fn frame(
        &mut self,
        model: &ModelInfo,
        state: &StateView<'_>,
        env: u32,
    ) -> Result<Tile, EnvError> {
        let world = body_poses(model, state, env);
        let tri = es_render::TriScene::from_scene_with_poses(&self.scene, &world)
            .map_err(|e| EnvError::Unsupported(format!("tessellation: {e}")))?;
        let view = camera_view(&self.scene, &self.cfg, &world)?;
        self.renderer
            .upload_tris(tri)
            .map_err(|e| EnvError::Unsupported(format!("scene upload: {e}")))?;
        let mut atlas = self
            .renderer
            .render(&[view])
            .map_err(|e| EnvError::Unsupported(format!("render: {e}")))?;
        let tile = atlas
            .read_tile(0, self.cfg.channel)
            .map_err(|e| EnvError::Unsupported(format!("readback: {e}")))?;
        if let Some(dir) = &self.cfg.frames_dir {
            tile.write_to(dir, &format!("{:06}", self.frame))
                .map_err(|e| EnvError::Unsupported(format!("frame write: {e}")))?;
        }
        self.frame += 1;
        Ok(tile)
    }

    /// Frames produced so far — the next one's `<NNNNNN>` stem.
    pub fn frames(&self) -> u64 {
        self.frame
    }

    pub fn cfg(&self) -> &EnvRendererCfg {
        &self.cfg
    }
}

/// What a renderer configured this way produces, in Observation IR terms — the subset of
/// spec 7.2's `ImageSpec` `es-render` decides (`crates/es-render/src/view.rs:44-50`).
///
/// A free function, not only an [`EnvRenderer`] method: the check below is arithmetic-free and
/// must be runnable on a machine with no Vulkan device.
pub fn image_spec(scene: &SceneDesc, cfg: &EnvRendererCfg) -> Result<ImageSpec, EnvError> {
    let view = camera_view(scene, cfg, &BTreeMap::new())?;
    let i = view.spec.intrinsics;
    Ok(ImageSpec {
        width: cfg.width,
        height: cfg.height,
        channels: channel_format(cfg.channel),
        dtype: image_dtype(cfg.channel),
        color_space: color_space(cfg.channel),
        camera_model: CameraModel::Pinhole,
        intrinsics: es_ir::image::Intrinsics::new(
            f64::from(i.fx),
            f64::from(i.fy),
            f64::from(i.cx),
            f64::from(i.cy),
        ),
        extrinsics: Pose::IDENTITY,
        distortion: DistortionModel::None,
        shutter: es_ir::image::ShutterModel::Global,
        exposure: std::time::Duration::ZERO,
        rate_hz: 0.0,
        depth_scale: None,
    })
}

/// Checks the Observation IR's **declared** `ImageSpec` against what the renderer produces,
/// field by field (spec 7.2, spec 26.1, `INV-14`).
///
/// A mismatch is [`EnvError::ImageSpec`] naming the field, never a repair: resizing or
/// converting here would make `observation_hash` describe a pipeline nobody declared, and
/// `Resize` / `Crop` / `ColorTransform` are Observation IR nodes that carry the intrinsics
/// with them.
///
/// The fields not compared are the ones the renderer does not decide: `exposure`, `rate_hz`,
/// `shutter` and `extrinsics` are the sensor model (spec 18.3), and the intrinsics are a
/// *consequence* of the camera's `fovy` and the declared size, computed in `f32` through
/// `es_math::approx::tan` (`crates/es-render/src/view.rs:31-37`) — comparing them to an
/// authoring tool's `f64` `tan` would fail on the last bit and mean nothing.
pub fn check_image_spec(ours: &ImageSpec, declared: &ImageSpec) -> Result<(), EnvError> {
    let mismatch = |field: &'static str, declared: String, produced: String| {
        Err(EnvError::ImageSpec {
            field,
            declared,
            produced,
        })
    };
    if declared.width != ours.width {
        return mismatch("width", declared.width.to_string(), ours.width.to_string());
    }
    if declared.height != ours.height {
        return mismatch(
            "height",
            declared.height.to_string(),
            ours.height.to_string(),
        );
    }
    if declared.channels != ours.channels {
        return mismatch(
            "channels",
            format!("{:?}", declared.channels),
            format!("{:?}", ours.channels),
        );
    }
    if declared.dtype != ours.dtype {
        return mismatch(
            "dtype",
            format!("{:?}", declared.dtype),
            format!("{:?}", ours.dtype),
        );
    }
    if declared.color_space != ours.color_space {
        return mismatch(
            "color_space",
            format!("{:?}", declared.color_space),
            format!("{:?}", ours.color_space),
        );
    }
    if declared.camera_model != ours.camera_model {
        return mismatch(
            "camera_model",
            format!("{:?}", declared.camera_model),
            format!("{:?}", ours.camera_model),
        );
    }
    if declared.distortion != ours.distortion {
        return mismatch(
            "distortion",
            format!("{:?}", declared.distortion),
            format!("{:?}", ours.distortion),
        );
    }
    Ok(())
}

/// What spec 15.1's channel is in Observation IR terms.
fn channel_format(channel: Channel) -> ChannelFormat {
    match channel {
        Channel::Rgb8 | Channel::RgbF32Linear | Channel::PtRadiance => ChannelFormat::Rgb,
        Channel::Depth32 { .. } => ChannelFormat::Depth,
        Channel::SegmentationId => ChannelFormat::Seg,
        Channel::Normal => ChannelFormat::Normal,
        Channel::Flow => ChannelFormat::Flow,
    }
}

fn image_dtype(channel: Channel) -> es_ir::image::ImageDType {
    match channel {
        Channel::Rgb8 => es_ir::image::ImageDType::U8,
        _ => es_ir::image::ImageDType::F32,
    }
}

/// sRGB for the 8-bit colour channel, linear for everything else
/// (`crates/es-render/src/view.rs:44-50`).
fn color_space(channel: Channel) -> ColorSpace {
    match channel {
        Channel::Rgb8 => ColorSpace::SRgb,
        _ => ColorSpace::Linear,
    }
}
