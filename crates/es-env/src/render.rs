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
use es_ir::task::{SensorPath, SensorRender};
use es_math::{Pose, Quat, Vec3};
use es_physics_core::backend::{ModelInfo, StateView};
use es_render::{
    CameraView, Channel, RenderConfig, RenderPath, Renderer, Tile, TileAtlasCfg, Tonemap,
};

use crate::EnvError;

/// `T_opencv_mujoco`: a rotation of pi about `+X`.
///
/// An MJCF camera looks along `-Z` with image-up along `+Y`; `es-render` reads the `OpenCV`
/// frame of spec 3.1 (`+Z` forward, `+Y` down). The importer stores the MJCF pose verbatim
/// (`crates/es-assets/src/mjcf/mod.rs:778`), so the conversion belongs here, once.
const MJCF_TO_OPENCV: Quat = Quat::from_xyzw(1.0, 0.0, 0.0, 0.0);

/// What to render, per env step.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvRendererCfg {
    /// The scene camera to render from (`SceneDesc::cameras`).
    pub camera: StableId,
    pub width: u32,
    pub height: u32,
    pub channel: Channel,
    pub path: RenderPath,
    /// Linear multiplier before [`Self::tonemap`] on the `Pt` path's `Rgb8` output — a
    /// **pass-through of what the sensor declared** (packet M7/R5), never a CLI knob. `1.0`,
    /// the `RenderConfig` default, is what every `Rs` sensor carries.
    pub exposure: f32,
    pub tonemap: Tonemap,
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
            exposure: 1.0,
            tonemap: Tonemap::Reinhard,
            frames_dir: None,
        }
    }
}

/// The renderer a Task IR sensor asks for (spec 6, spec 15.3; packet M7/R5).
///
/// The one place a declared sensor becomes an [`EnvRendererCfg`], so collection, evaluation
/// and the showcase cannot render the same document three different ways. `spec` is the
/// declared [`ImageSpec`] and is **read, never rewritten** — the size is the document's and
/// `EnvRenderer::check` still refuses a camera that produces something else (`INV-14`).
///
/// [`SensorPath::Rs`] maps to exactly [`EnvRendererCfg::rgb`], field for field, which is what
/// keeps every committed frame and every render golden bitwise (spec 28.10 rule 1). `Pt` maps
/// to R3's estimator — NEE on, `ReSTIR` and `SVGF` off — and to **no accumulation** (R4): an
/// observation frame is a pure function of the pose it was rendered from, which is what the
/// collector/evaluator parity oracle needs.
pub fn sensor_cfg(
    camera: StableId,
    spec: &ImageSpec,
    render: &SensorRender,
    frames_dir: Option<PathBuf>,
) -> EnvRendererCfg {
    EnvRendererCfg {
        path: match render.path {
            SensorPath::Rs => RenderPath::Rs,
            SensorPath::Pt { spp, bounces } => RenderPath::Pt {
                spp,
                bounces,
                nee: true,
                restir: false,
                svgf: false,
            },
        },
        exposure: render.exposure,
        tonemap: match render.tonemap {
            es_ir::task::Tonemap::Reinhard => Tonemap::Reinhard,
            es_ir::task::Tonemap::Aces => Tonemap::Aces,
        },
        frames_dir,
        ..EnvRendererCfg::rgb(camera, spec.width, spec.height)
    }
}

/// The [`RenderConfig`] a [`EnvRendererCfg`] means. One function, so the GPU renderer and the
/// CPU reference (`es_render::cpu`) cannot drift apart — every golden and the GPU/CPU
/// comparison depend on them being the same config.
pub fn render_config(cfg: &EnvRendererCfg) -> RenderConfig {
    let mut out = config(cfg.width, cfg.height, cfg.channel, cfg.path);
    // The sensor's own two pass-through fields (packet M7/R5). Both default to what
    // `RenderConfig` already carried, so an `Rs` sensor is byte for byte today's config.
    out.exposure = cfg.exposure;
    out.tonemap = cfg.tonemap;
    out
}

/// [`render_config`] without an [`EnvRendererCfg`], for a camera that is not a scene camera
/// -- `es video showcase`'s free view (packet M5/V9). The same function underneath, so there
/// is still exactly one place a render path becomes a [`RenderConfig`].
pub fn config(width: u32, height: u32, channel: Channel, path: RenderPath) -> RenderConfig {
    let atlas = TileAtlasCfg::row(width, height, 1);
    // Every other `RenderConfig` field stays at its constructor default, which is what spec
    // 28.10 rule 1 asks of an observation path: packet M7/R2's `shading` and packet M7/R3's
    // `nee`, `light_rgb`, `exposure` and `tonemap` are carried through untouched, so a
    // committed observation document renders the same bytes it always did.
    let mut out = match path {
        RenderPath::Rs => RenderConfig::rs(atlas),
        RenderPath::Pt { spp, bounces, .. } => {
            let mut c = RenderConfig::pt(atlas, spp, bounces);
            c.path = path;
            c
        }
    };
    // Only the channel the caller asked for: every other one is an atlas-sized buffer nobody
    // reads back.
    out.channels = BTreeSet::from([channel]);
    out
}

/// World pose of every body the model indexes ([`crate::traj::body_poses`]).
///
/// Defined beside the `.estraj` trajectory rather than here, because a replay re-poses a
/// scene with no Vulkan device in the process and this module is the one that links one.
pub use crate::traj::body_poses;

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

/// A camera that is **not** in the scene: eye, aim point and vertical field of view.
///
/// The showcase render (packet M5/V9) needs a view the Observation IR does not declare, and
/// the scene file cannot grow one — `scene_hash` feeds `task_hash` feeds every trained
/// bundle, so adding a `<camera>` to the demo MJCF would invalidate the checkpoints the video
/// is meant to show. So the camera is built here, from the command line, in the same `OpenCV`
/// frame [`camera_view`] converts scene cameras into (`+X` right, `+Y` down, `+Z` forward,
/// spec 3.1). `INV-14` is not in play: nothing is resized: the intrinsics are computed from
/// this `fovy` at this size, exactly like every other camera.
///
/// World up is `+Z`. An eye that looks straight up or down along it is refused rather than
/// silently rolled.
pub fn look_at(
    eye: [f64; 3],
    target: [f64; 3],
    fovy_rad: f64,
    width: u32,
    height: u32,
) -> Result<CameraView, EnvError> {
    let (eye, target) = (
        Vec3::new(eye[0], eye[1], eye[2]),
        Vec3::new(target[0], target[1], target[2]),
    );
    let forward = (target - eye).normalize();
    let up = Vec3::new(0.0, 0.0, 1.0);
    let right = forward.cross(up);
    if !forward.norm().is_finite() || right.norm() < 1e-9 {
        return Err(EnvError::Task(format!(
            "camera at {eye:?} looking at {target:?}: the view direction is degenerate or              parallel to world up (+Z), so there is no roll-free orientation"
        )));
    }
    let right = right.normalize();
    let down = forward.cross(right);
    Ok(CameraView {
        pose: Pose::new(eye, quat_from_basis(right, down, forward)),
        spec: es_render::ImageSpec::pinhole(width, height, fovy_rad),
    })
}

/// The rotation whose matrix has `x`, `y`, `z` as its columns, by Shepperd's method: pick the
/// largest of the four denominators, so no branch divides by something near zero.
fn quat_from_basis(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let (m00, m01, m02) = (x.x, y.x, z.x);
    let (m10, m11, m12) = (x.y, y.y, z.y);
    let (m20, m21, m22) = (x.z, y.z, z.z);
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat::from_xyzw((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        Quat::from_xyzw(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        Quat::from_xyzw((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        Quat::from_xyzw((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    }
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
    /// Kept whole: every frame re-poses the bodies (design note 7.1).
    scene: SceneDesc,
    /// The per-geom local tessellation, computed once and re-posed per frame (design note
    /// `renderer.md` section 8.2). Bit-identical to tessellating every frame.
    cache: es_render::SceneCache,
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
            cache: es_render::SceneCache::default(),
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
        let tri = self
            .cache
            .tri_scene(&self.scene, &world)
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
        // `History` is packet M7/R4's renderer diagnostic, a u32 count per pixel. It is
        // unreachable here: `EnvRendererCfg` names one channel and the observation path never
        // names this one. It maps to the other single-u32 format so the match stays total.
        Channel::SegmentationId | Channel::History => ChannelFormat::Seg,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The showcase camera aims where it is told, in the `OpenCV` frame `es-render` reads:
    /// `+Z` is the view direction, `+Y` is image-down, `+X` is image-right (spec 3.1).
    #[test]
    fn look_at_puts_plus_z_on_the_view_direction() {
        // Three-quarter view: in front of, to the side of and above the origin.
        let eye = [0.6, -0.5, 0.4];
        let view = look_at(eye, [0.0; 3], 45f64.to_radians(), 320, 240).expect("a view");
        let eye = Vec3::new(eye[0], eye[1], eye[2]);
        let q = view.pose.orientation;
        let close = |a: Vec3, b: Vec3| (a - b).norm() < 1e-12;
        assert!(close(
            q.rotate(Vec3::new(0.0, 0.0, 1.0)),
            (Vec3::ZERO - eye).normalize()
        ));
        // Image-right is horizontal, and image-down points below the horizon.
        assert!(q.rotate(Vec3::new(1.0, 0.0, 0.0)).z.abs() < 1e-12);
        assert!(q.rotate(Vec3::new(0.0, 1.0, 0.0)).z < 0.0);
        assert_eq!((view.spec.width, view.spec.height), (320, 240));
    }

    /// Straight down the world-up axis has no roll-free orientation, and is refused rather
    /// than resolved by an arbitrary choice.
    #[test]
    fn a_camera_looking_along_world_up_is_refused() {
        let r = look_at([0.0, 0.0, 1.0], [0.0; 3], 0.8, 64, 64);
        assert!(matches!(r, Err(EnvError::Task(_))), "{r:?}");
    }
}
