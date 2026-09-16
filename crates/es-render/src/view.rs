//! Cameras and render configuration.
//!
//! Conventions are spec 3.1's: image origin top-left with x right and y down, camera frame
//! `OpenCV` (+Z forward, +X right, +Y down), sRGB the default colour space, metres.

use es_core::TickRate;
use es_math::{Pose, Vec3};
use es_sensor::{CameraContract, Channel, ContractError, Shutter};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Pinhole intrinsics in pixels, spec 7.2. No skew: every camera model this crate renders is
/// a plain pinhole, and a skew term that is always zero is a field nobody can trust.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Intrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl Intrinsics {
    /// Intrinsics from a vertical field of view, the form `es_assets::scene::Camera` carries.
    /// Principal point at the image centre.
    ///
    /// `fovy_rad` is `f64` because that is how the asset stores it; everything after the
    /// first cast is `f32` through [`es_math::approx::tan`], never the host `libm`. `fx`/`fy`
    /// sit on the golden camera path — every golden pixel is a function of them — so a
    /// platform's `tan` must not be able to move them (spec 3.2, 3.4). Op order is fixed:
    /// halve, `tan`, divide.
    pub fn from_fovy(width: u32, height: u32, fovy_rad: f64) -> Self {
        let f = (height as f32 * 0.5) / es_math::approx::tan(fovy_rad as f32 * 0.5);
        Self {
            fx: f,
            fy: f,
            cx: width as f32 * 0.5,
            cy: height as f32 * 0.5,
        }
    }
}

/// The subset of the spec 7.2 `ImageSpec` this crate honours.
///
/// `es-render` is layer 5 and `es-ir` (which owns the full `ImageSpec`) is layer 6, so the
/// full type is out of reach here by design (spec 4.2). The fields left out are not left
/// *open*: `camera_model` is `Pinhole`, `distortion` is `None`, `shutter` is `Global`,
/// `color_space` is sRGB for [`Channel::Rgb8`] and linear for the float colour channels, and
/// `depth_scale` is 1 m per unit. Sensor realism (spec 18.3) is what makes any of those a
/// choice, and it is a later pass over the atlas, not a renderer setting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageSpec {
    pub width: u32,
    pub height: u32,
    pub intrinsics: Intrinsics,
    /// Near clip, m. Hits at or before this are dropped.
    pub near: f32,
    /// Far clip, m. Also the background value of the depth channel.
    pub far: f32,
}

impl ImageSpec {
    pub fn pinhole(width: u32, height: u32, fovy_rad: f64) -> Self {
        Self {
            width,
            height,
            intrinsics: Intrinsics::from_fovy(width, height, fovy_rad),
            near: 0.01,
            far: 100.0,
        }
    }

    /// The spec 7.2 timing half of the contract, for the sensor layer downstream.
    /// Always [`Shutter::Global`] with zero exposure: see the type docs.
    pub fn to_contract(
        &self,
        channels: BTreeSet<Channel>,
        rate: TickRate,
    ) -> Result<CameraContract, ContractError> {
        CameraContract::new(channels, self.width, self.height, rate, Shutter::Global, 0)
    }
}

/// One camera to render: where it is and what it sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraView {
    /// `T_world_camera`. The camera frame is `OpenCV` (spec 3.1).
    pub pose: Pose,
    pub spec: ImageSpec,
}

/// Floats per view in the GPU parameter buffer.
pub const VIEW_STRIDE: usize = 16;

/// One view flattened to `f32`, exactly as the shaders read it. The CPU reference uses the
/// same struct, so neither path can quietly work in `f64` where the other works in `f32`
/// (spec 3.3: rendering is FP32).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewParams {
    pub pos: [f32; 3],
    /// `T_world_camera` rotation, xyzw.
    pub quat: [f32; 4],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub near: f32,
    pub far: f32,
}

impl ViewParams {
    pub fn new(view: &CameraView) -> Self {
        let q = view.pose.orientation.normalize();
        Self {
            pos: [
                view.pose.position.x as f32,
                view.pose.position.y as f32,
                view.pose.position.z as f32,
            ],
            quat: [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
            fx: view.spec.intrinsics.fx,
            fy: view.spec.intrinsics.fy,
            cx: view.spec.intrinsics.cx,
            cy: view.spec.intrinsics.cy,
            near: view.spec.near,
            far: view.spec.far,
        }
    }

    pub(crate) fn to_floats(self) -> [f32; VIEW_STRIDE] {
        [
            self.pos[0],
            self.pos[1],
            self.pos[2],
            self.quat[0],
            self.quat[1],
            self.quat[2],
            self.quat[3],
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.near,
            self.far,
            0.0,
            0.0,
            0.0,
        ]
    }
}

/// Which render path (spec 15.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPath {
    /// Rasterization. The spec 15.3 default for vision learning.
    Rs,
    /// Path tracing. Spec 1.9 item 2: the second thing cut under scope pressure.
    Pt {
        spp: u32,
        bounces: u32,
        /// `ReSTIR` DI (spec 28.6). Replaces the path-traced image with a direct-lighting
        /// estimate; see `docs/design/renderer.md` for what that means and what is skipped.
        restir: bool,
        /// Edge-aware a-trous denoise (spec 28.6). See the design doc: the variance half of
        /// SVGF is not implemented.
        svgf: bool,
    },
    // The spec 16.2 splat path lands here as a third variant writing the same channels into
    // the same atlas. `es-splat` owns the asset side today.
}

/// How the [`RenderPath::Rs`] path shades a hit (packet M7/R2).
///
/// [`Shading::Lambert`] is the default and is what every golden in `tests/golden/render/` and
/// every committed observation document pins, bit for bit: spec 28.10 rule 1 says a renderer
/// improvement arrives as a field whose default is today's output. [`Shading::Full`] is the
/// opt-in look — one shadow ray, a hemisphere ambient, a Blinn-Phong highlight and
/// supersampling — and is read by the `Rs` path only; the `Pt` path models all of it properly
/// and ignores this field.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Shading {
    /// Flat Lambert + constant [`RenderConfig::ambient`], one sample per pixel.
    #[default]
    Lambert,
    /// See `docs/design/renderer.md` section 9 for the equations and the accumulation order.
    Full {
        /// One shadow ray towards `light_dir` per shaded sub-sample (any-hit, R1's traversal).
        shadows: bool,
        /// Blinn-Phong specular weight in `[0, 1]`; `0.0` disables the highlight.
        specular: f32,
        /// Blinn-Phong exponent.
        shininess: f32,
        /// Hemisphere ambient: `sky_rgb` at `n.z = +1`, `ground_rgb` at `n.z = -1`, lerped on
        /// `(n.z + 1) / 2`. Replaces the constant [`RenderConfig::ambient`].
        sky_rgb: [f32; 3],
        ground_rgb: [f32; 3],
        /// Sub-samples per axis (`1` = off). Colour is the box filter of the `ssaa * ssaa`
        /// sub-samples; the geometry channels stay the centre ray's, so they do not move.
        ssaa: u32,
    },
}

impl Shading {
    /// The preset [`RenderConfig::rs_full`] and `es video showcase --look full` use.
    pub const FULL: Self = Self::Full {
        shadows: true,
        specular: 0.25,
        shininess: 32.0,
        sky_rgb: [0.55, 0.65, 0.85],
        ground_rgb: [0.25, 0.22, 0.20],
        ssaa: 2,
    };

    /// Sub-samples per axis, at least 1.
    pub(crate) fn ssaa(self) -> u32 {
        match self {
            Self::Lambert => 1,
            Self::Full { ssaa, .. } => ssaa.max(1),
        }
    }
}

/// Tile packing, spec 15.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileAtlasCfg {
    pub tile_w: u32,
    pub tile_h: u32,
    pub tiles_per_row: u32,
    /// How many tiles the atlas holds. Padding tiles (`n_tiles` past the last view) are
    /// cleared to the channel background and never read back.
    pub n_tiles: u32,
}

impl TileAtlasCfg {
    /// One row of `n` tiles.
    pub fn row(tile_w: u32, tile_h: u32, n: u32) -> Self {
        Self {
            tile_w,
            tile_h,
            tiles_per_row: n.max(1),
            n_tiles: n,
        }
    }
}

/// Everything the renderer needs that is not the scene or the cameras.
#[derive(Clone, Debug)]
pub struct RenderConfig {
    pub atlas: TileAtlasCfg,
    /// Channels the caller wants back. The renderer may compute more (one shader writes all
    /// of a path's channels); only these appear in the [`crate::Atlas`].
    pub channels: BTreeSet<Channel>,
    pub path: RenderPath,
    /// World-space unit direction **towards** the one directional light.
    pub light_dir: Vec3,
    /// Ambient floor in `[0, 1]`, so a surface facing away from the light is not pure black.
    /// [`Shading::Full`] replaces it with a hemisphere ambient.
    pub ambient: f32,
    /// How the `Rs` path shades. The default is today's look, byte for byte (spec 28.10
    /// rule 1).
    pub shading: Shading,
    /// Linear radiance a ray that hits nothing returns (path tracer), in `[0, inf)`.
    pub sky: [f32; 3],
    /// Seed of the counter-based RNG (spec 3.4: addressed, never global).
    pub seed: u32,
    /// A-trous iterations when `svgf` is on.
    pub svgf_iterations: u32,
}

impl RenderConfig {
    /// Rasterizing config emitting the four channels the `Rs` path produces.
    pub fn rs(atlas: TileAtlasCfg) -> Self {
        Self {
            atlas,
            channels: crate::RS_CHANNELS.iter().copied().collect(),
            path: RenderPath::Rs,
            light_dir: Vec3::new(0.3, 0.4, 0.866_025_4).normalize(),
            ambient: 0.15,
            shading: Shading::Lambert,
            sky: [0.0, 0.0, 0.0],
            seed: 0x5eed_1234,
            svgf_iterations: 4,
        }
    }

    /// [`Self::rs`] with the opt-in [`Shading::FULL`] look (packet M7/R2). Nothing else moves:
    /// same channels, same light, same atlas.
    pub fn rs_full(atlas: TileAtlasCfg) -> Self {
        Self {
            shading: Shading::FULL,
            ..Self::rs(atlas)
        }
    }

    /// Path-tracing config emitting `PtRadiance` plus the three geometry channels.
    pub fn pt(atlas: TileAtlasCfg, spp: u32, bounces: u32) -> Self {
        let mut cfg = Self::rs(atlas);
        cfg.path = RenderPath::Pt {
            spp,
            bounces,
            restir: false,
            svgf: false,
        };
        cfg.channels = crate::PT_CHANNELS.iter().copied().collect();
        cfg
    }

    pub(crate) fn spp(&self) -> u32 {
        match self.path {
            RenderPath::Rs => 1,
            RenderPath::Pt { spp, .. } => spp,
        }
    }

    pub(crate) fn bounces(&self) -> u32 {
        match self.path {
            RenderPath::Rs => 1,
            RenderPath::Pt { bounces, .. } => bounces,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins both the arithmetic and its order: no host `tan` anywhere on the camera path
    /// that produces the goldens (review M4 S-11).
    #[test]
    #[allow(clippy::float_cmp)]
    fn intrinsics_come_from_approx_tan_in_f32() {
        let i = Intrinsics::from_fovy(64, 48, 1.2);
        let want = (48.0f32 * 0.5) / es_math::approx::tan(1.2f32 * 0.5);
        assert_eq!(i.fy.to_bits(), want.to_bits(), "fy {} vs {want}", i.fy);
        assert_eq!(i.fx, i.fy);
        assert_eq!((i.cx, i.cy), (32.0, 24.0));
    }
}
