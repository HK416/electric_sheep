//! `ImageSpec` and its transform rules (spec 7.2, Appendix B.2, `INV-14`).
//!
//! Resize and crop change the geometry of an image, so they change its intrinsics. Doing one
//! without the other does not fail — it silently returns wrong 3D — which is why the transforms
//! are methods here and the Observation IR nodes are required to go through them. See
//! `docs/design/image-spec.md`.

use std::time::Duration;

use es_math::conventions::Pose;
use serde::{Deserialize, Serialize};

use crate::canon::CanonWriter;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelFormat {
    Rgb,
    Rgba,
    Gray,
    Depth,
    Seg,
    Normal,
    Flow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageDType {
    U8,
    U16,
    F16,
    F32,
}

/// sRGB is the default (spec 3.1); linearization is an explicit node, never implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    SRgb,
    Linear,
    Rec709,
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraModel {
    Pinhole,
    Fisheye,
    Equirect,
    OrthoDepth,
}

/// Pinhole intrinsics in pixels, `OpenCV` convention: origin top-left, `+x` right, `+y` down.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
}

impl Intrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            skew: 0.0,
        }
    }

    /// Scales for a resize by `(sx, sy)` (Appendix B.2).
    #[must_use]
    pub fn scaled(self, sx: f64, sy: f64) -> Self {
        Self {
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: self.cx * sx,
            cy: self.cy * sy,
            skew: self.skew * sx,
        }
    }

    fn canonical(&self, w: &mut CanonWriter) {
        for v in [self.fx, self.fy, self.cx, self.cy, self.skew] {
            w.f64(v);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DistortionModel {
    None,
    BrownConrady {
        k1: f64,
        k2: f64,
        k3: f64,
        p1: f64,
        p2: f64,
    },
    KannalaBrandt {
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadoutDir {
    TopToBottom,
    BottomToTop,
    LeftToRight,
    RightToLeft,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShutterModel {
    Global,
    Rolling { readout: Duration, dir: ReadoutDir },
}

/// Crop rectangle in pixels, origin top-left.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// The contract an image port carries (spec 7.2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageSpec {
    pub width: u32,
    pub height: u32,
    pub channels: ChannelFormat,
    pub dtype: ImageDType,
    pub color_space: ColorSpace,
    pub camera_model: CameraModel,
    pub intrinsics: Intrinsics,
    /// `T_body_camera`. The camera optical frame is `OpenCV`: `+Z` forward, `+X` right,
    /// `+Y` down (spec 3.1).
    pub extrinsics: Pose,
    pub distortion: DistortionModel,
    pub shutter: ShutterModel,
    pub exposure: Duration,
    pub rate_hz: f32,
    /// Metres per LSB, for a depth channel.
    pub depth_scale: Option<f32>,
}

impl ImageSpec {
    /// Resize scales the intrinsics. `rescale = false` is the caller's explicit opt-out and
    /// leaves them untouched — the check that reports it is
    /// [`ImageSpec::intrinsics_consistent_with`] (`OBS-034`).
    #[must_use]
    pub fn resized(&self, width: u32, height: u32, rescale: bool) -> Self {
        let (sx, sy) = self.scale_to(width, height);
        Self {
            width,
            height,
            intrinsics: if rescale {
                self.intrinsics.scaled(sx, sy)
            } else {
                self.intrinsics
            },
            ..*self
        }
    }

    /// Crop moves the principal point by the rectangle's origin.
    #[must_use]
    pub fn cropped(&self, rect: Rect, rescale: bool) -> Self {
        let intrinsics = if rescale {
            Intrinsics {
                cx: self.intrinsics.cx - f64::from(rect.x),
                cy: self.intrinsics.cy - f64::from(rect.y),
                ..self.intrinsics
            }
        } else {
            self.intrinsics
        };
        Self {
            width: rect.width,
            height: rect.height,
            intrinsics,
            ..*self
        }
    }

    /// Undistortion clears the distortion model and installs the rectified intrinsics.
    #[must_use]
    pub fn undistorted(&self, new_intrinsics: Intrinsics) -> Self {
        Self {
            intrinsics: new_intrinsics,
            distortion: DistortionModel::None,
            ..*self
        }
    }

    /// Are `self`'s intrinsics what `other`'s become under the resize between the two
    /// resolutions? A `false` here at a `CameraProjection` node is `OBS-034` (spec 7.2).
    pub fn intrinsics_consistent_with(&self, other: &Self) -> bool {
        let (sx, sy) = other.scale_to(self.width, self.height);
        let expected = other.intrinsics.scaled(sx, sy);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
        close(self.intrinsics.fx, expected.fx)
            && close(self.intrinsics.fy, expected.fy)
            && close(self.intrinsics.cx, expected.cx)
            && close(self.intrinsics.cy, expected.cy)
            && close(self.intrinsics.skew, expected.skew)
    }

    fn scale_to(&self, width: u32, height: u32) -> (f64, f64) {
        (
            f64::from(width) / f64::from(self.width),
            f64::from(height) / f64::from(self.height),
        )
    }

    /// Canonical bytes, for a port type that carries an image contract.
    pub fn canonical(&self, w: &mut CanonWriter) {
        w.u32(self.width);
        w.u32(self.height);
        for tag in [
            format!("{:?}", self.channels),
            format!("{:?}", self.dtype),
            format!("{:?}", self.color_space),
            format!("{:?}", self.camera_model),
            format!("{:?}", self.distortion),
            format!("{:?}", self.shutter),
        ] {
            w.str(&tag);
        }
        self.intrinsics.canonical(w);
        let p = self.extrinsics.position;
        let q = self.extrinsics.orientation;
        for v in [p.x, p.y, p.z, q.x, q.y, q.z, q.w] {
            w.f64(v);
        }
        w.u64(self.exposure.as_nanos().try_into().unwrap_or(u64::MAX));
        w.f32(self.rate_hz);
        match self.depth_scale {
            None => w.bool(false),
            Some(s) => {
                w.bool(true);
                w.f32(s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    /// A 640x480 sRGB pinhole camera.
    fn cam() -> ImageSpec {
        ImageSpec {
            width: 640,
            height: 480,
            channels: ChannelFormat::Rgb,
            dtype: ImageDType::U8,
            color_space: ColorSpace::SRgb,
            camera_model: CameraModel::Pinhole,
            intrinsics: Intrinsics::new(600.0, 600.0, 320.0, 240.0),
            extrinsics: Pose::IDENTITY,
            distortion: DistortionModel::BrownConrady {
                k1: -0.1,
                k2: 0.01,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
            shutter: ShutterModel::Global,
            exposure: Duration::from_micros(500),
            rate_hz: 30.0,
            depth_scale: None,
        }
    }

    #[test]
    fn resize_scales_the_intrinsics() {
        let half = cam().resized(320, 240, true);
        assert_eq!(half.width, 320);
        assert_eq!(half.height, 240);
        assert_eq!(half.intrinsics, Intrinsics::new(300.0, 300.0, 160.0, 120.0));
        assert!(half.intrinsics_consistent_with(&cam()));

        // Non-uniform: 640x480 -> 224x224 scales x and y differently.
        let square = cam().resized(224, 224, true);
        assert_close(square.intrinsics.fx, 600.0 * 224.0 / 640.0);
        assert_close(square.intrinsics.fy, 600.0 * 224.0 / 480.0);
        assert!(square.intrinsics_consistent_with(&cam()));
    }

    #[test]
    fn resize_without_rescale_leaves_intrinsics_untouched() {
        let stale = cam().resized(224, 224, false);
        assert_eq!(stale.intrinsics, cam().intrinsics);
        // This is exactly what OBS-034 reports; the caller emits the diagnostic.
        assert!(!stale.intrinsics_consistent_with(&cam()));
    }

    #[test]
    fn crop_shifts_the_principal_point_only() {
        let rect = Rect {
            x: 80,
            y: 60,
            width: 480,
            height: 360,
        };
        let cropped = cam().cropped(rect, true);
        assert_eq!(cropped.width, 480);
        assert_eq!(cropped.height, 360);
        assert_close(cropped.intrinsics.cx, 240.0);
        assert_close(cropped.intrinsics.cy, 180.0);
        assert_close(cropped.intrinsics.fx, 600.0);
        assert_close(cropped.intrinsics.fy, 600.0);

        let stale = cam().cropped(rect, false);
        assert_eq!(stale.intrinsics, cam().intrinsics);
    }

    #[test]
    fn undistort_clears_the_model() {
        let rectified = Intrinsics::new(590.0, 590.0, 319.0, 241.0);
        let out = cam().undistorted(rectified);
        assert_eq!(out.distortion, DistortionModel::None);
        assert_eq!(out.intrinsics, rectified);
        assert_eq!(out.width, 640);
    }

    #[test]
    fn same_resolution_requires_equal_intrinsics() {
        let a = cam();
        let mut b = cam();
        assert!(a.intrinsics_consistent_with(&b));
        b.intrinsics.cx += 1.0;
        assert!(!a.intrinsics_consistent_with(&b));
    }

    #[test]
    fn resize_then_crop_composes() {
        let resized = cam().resized(320, 240, true);
        let cropped = resized.cropped(
            Rect {
                x: 48,
                y: 8,
                width: 224,
                height: 224,
            },
            true,
        );
        assert_close(cropped.intrinsics.cx, 160.0 - 48.0);
        assert_close(cropped.intrinsics.cy, 120.0 - 8.0);
        assert_close(cropped.intrinsics.fx, 300.0);
    }

    #[test]
    fn canonical_encoding_tracks_every_field() {
        let encode = |s: &ImageSpec| {
            let mut w = CanonWriter::new();
            s.canonical(&mut w);
            w.finish().unwrap()
        };
        let base = encode(&cam());
        assert_eq!(base, encode(&cam()));
        assert_ne!(base, encode(&cam().resized(320, 240, true)));

        let mut other = cam();
        other.color_space = ColorSpace::Linear;
        assert_ne!(base, encode(&other));

        let mut other = cam();
        other.rate_hz = 60.0;
        assert_ne!(base, encode(&other));

        let mut other = cam();
        other.depth_scale = Some(0.001);
        assert_ne!(base, encode(&other));

        let mut other = cam();
        other.shutter = ShutterModel::Rolling {
            readout: Duration::from_millis(10),
            dir: ReadoutDir::TopToBottom,
        };
        assert_ne!(base, encode(&other));
    }

    #[test]
    fn serde_round_trip() {
        let json = serde_json::to_string(&cam()).unwrap();
        assert_eq!(serde_json::from_str::<ImageSpec>(&json).unwrap(), cam());
    }
}
