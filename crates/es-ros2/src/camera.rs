//! Camera ingest (W1c): a `sensor_msgs/CameraInfo` + `sensor_msgs/Image` pair becomes a
//! validated [`ImageSpec`], an HWC byte buffer, a [`PhysTick`] and the Safety Plane's
//! observation-age input. `docs/design/ros2-boundary.md` section 6 is the prescriptive
//! reference (6.1 field mapping, 6.2 ROI/binning, 6.3 encodings, 6.4 identity, 6.5 time);
//! spec 7.2 and Appendix B.2 own [`ImageSpec`] itself.
//!
//! Three rules shape everything here:
//!
//! - **INV-14.** This is the first code in the tree that builds an `ImageSpec` from outside
//!   data, so ROI, binning and a size mismatch are exactly where intrinsics go wrong. Every
//!   size change goes through [`ImageSpec::cropped`] / [`ImageSpec::resized`] with
//!   `rescale = true`; the only intrinsic values written directly are the `k`/`p` indexing of
//!   section 6.1.
//! - **Spec 26.1, "what is not validated is not executed".** [`check_declared`] compares the
//!   derived spec against the Observation IR's declared one and names the first differing
//!   field; it never substitutes one for the other.
//! - **Spec 25.1.** Image bytes cross a trust boundary: [`decode_image`] checks `step`,
//!   `data.len()` and the output size *before* it allocates, and is total (no panic) for any
//!   [`Image`] value.
//!
//! Time is integer throughout (spec 18.1, spec 3.4): [`StampClock::tick`] does its arithmetic
//! in `i128` and there is no `f32`/`f64` anywhere near a stamp.

use std::time::Duration;

use es_core::time::{PhysTick, TickRate};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    Rect, ShutterModel,
};
use es_math::conventions::Pose;
use es_safety::Micros;

pub use crate::error::CameraError;
use crate::msg::{CameraInfo, Image, Time};

/// Cap on one decoded frame, the same 64 MiB `es-ros2` caps a whole CDR message at
/// ([`crate::cdr::MAX_MESSAGE_BYTES`], spec 25.1).
pub const MAX_FRAME_BYTES: usize = crate::cdr::MAX_MESSAGE_BYTES;

/// What ROS 2 does not carry and no default may invent (design note section 6.1): the sensor's
/// identity, where it sits, and what its pixels mean.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIngestConfig {
    /// Safety Plane sensor name, for `SafetyPlane::sensor_seen` (spec 9.4).
    pub sensor: String,
    /// The `header.frame_id` both messages must carry (design note section 6.4).
    pub frame_id: String,
    /// `T_body_camera`. ROS optical frames already use the `ImageSpec` convention (`+Z`
    /// forward, `+X` right, `+Y` down), so no axis change is applied.
    pub extrinsics: Pose,
    pub color_space: ColorSpace,
    pub shutter: ShutterModel,
    pub exposure: Duration,
    pub rate_hz: f32,
    /// The stream is already rectified: read `p`, not `k`, and clear the distortion model.
    pub rectified: bool,
    /// Resize the calibration to the image size when they disagree, instead of `CAM-007`.
    pub rescale_to_image: bool,
}

// --- Encodings (design note section 6.3) --------------------------------------------------

/// How one row of source bytes becomes one row of decoded bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pixel {
    /// Already in the right order: copy the row.
    Copy,
    /// Swap channels 0 and 2 of every pixel (`bgr8`, `bgra8`).
    SwapRb,
    /// Multi-byte elements: byte-reverse each one when `is_bigendian` is set.
    Endian,
    /// Packed YUV 4:2:2. `y_idx` is the offset of the first luma byte in each 4-byte pair:
    /// `1` for `uyvy` (`U Y0 V Y1`), `0` for `yuyv` (`Y0 U Y1 V`).
    Yuv422 { y_idx: usize },
}

#[derive(Debug)]
struct Encoding {
    name: &'static str,
    channels: ChannelFormat,
    dtype: ImageDType,
    /// Bytes per pixel on the wire.
    src_bpp: u32,
    /// Bytes per pixel after decoding.
    out_bpp: u32,
    /// Metres per LSB (REP-118), for the depth encodings only.
    depth_scale: Option<f32>,
    pixel: Pixel,
}

const fn enc(
    name: &'static str,
    channels: ChannelFormat,
    dtype: ImageDType,
    src_bpp: u32,
    out_bpp: u32,
    depth_scale: Option<f32>,
    pixel: Pixel,
) -> Encoding {
    Encoding {
        name,
        channels,
        dtype,
        src_bpp,
        out_bpp,
        depth_scale,
        pixel,
    }
}

/// `sensor_msgs/image_encodings.hpp`'s strings this crate accepts, and nothing else
/// (`docs/api-notes/ros2-cdr.md`). A linear scan over 12 entries: no `HashMap` (spec 3.4).
const ENCODINGS: [Encoding; 12] = [
    enc(
        "rgb8",
        ChannelFormat::Rgb,
        ImageDType::U8,
        3,
        3,
        None,
        Pixel::Copy,
    ),
    enc(
        "bgr8",
        ChannelFormat::Rgb,
        ImageDType::U8,
        3,
        3,
        None,
        Pixel::SwapRb,
    ),
    enc(
        "rgba8",
        ChannelFormat::Rgba,
        ImageDType::U8,
        4,
        4,
        None,
        Pixel::Copy,
    ),
    enc(
        "bgra8",
        ChannelFormat::Rgba,
        ImageDType::U8,
        4,
        4,
        None,
        Pixel::SwapRb,
    ),
    enc(
        "mono8",
        ChannelFormat::Gray,
        ImageDType::U8,
        1,
        1,
        None,
        Pixel::Copy,
    ),
    enc(
        "mono16",
        ChannelFormat::Gray,
        ImageDType::U16,
        2,
        2,
        None,
        Pixel::Endian,
    ),
    // REP-118: "depth in millimeters"; "The value 0 denotes an invalid depth" (0 stays 0).
    enc(
        "16UC1",
        ChannelFormat::Depth,
        ImageDType::U16,
        2,
        2,
        Some(0.001),
        Pixel::Endian,
    ),
    // REP-118: depth in metres.
    enc(
        "32FC1",
        ChannelFormat::Depth,
        ImageDType::F32,
        4,
        4,
        Some(1.0),
        Pixel::Endian,
    ),
    enc(
        "uyvy",
        ChannelFormat::Rgb,
        ImageDType::U8,
        2,
        3,
        None,
        Pixel::Yuv422 { y_idx: 1 },
    ),
    enc(
        "yuv422",
        ChannelFormat::Rgb,
        ImageDType::U8,
        2,
        3,
        None,
        Pixel::Yuv422 { y_idx: 1 },
    ),
    enc(
        "yuyv",
        ChannelFormat::Rgb,
        ImageDType::U8,
        2,
        3,
        None,
        Pixel::Yuv422 { y_idx: 0 },
    ),
    enc(
        "yuv422_yuy2",
        ChannelFormat::Rgb,
        ImageDType::U8,
        2,
        3,
        None,
        Pixel::Yuv422 { y_idx: 0 },
    ),
];

fn encoding_of(name: &str) -> Result<&'static Encoding, CameraError> {
    ENCODINGS
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| CameraError::UnsupportedEncoding(name.to_owned()))
}

// OpenCV's fixed-point BT.601 constants, `modules/imgproc/src/color_yuv.simd.hpp` (OpenCV
// 5.0.0, as shipped by opencv-python-headless 5.0.0.93). Copied so the conversion is
// bit-exact against `cv::COLOR_YUV2RGB_UYVY` / `COLOR_YUV2RGB_YUY2`; the goldens decide
// (design note section 6.3).
const ITUR_BT601_CY: i32 = 1_220_542;
const ITUR_BT601_CUB: i32 = 2_116_026;
const ITUR_BT601_CUG: i32 = -409_993;
const ITUR_BT601_CVG: i32 = -852_492;
const ITUR_BT601_CVR: i32 = 1_673_527;
const ITUR_BT601_SHIFT: i32 = 20;

/// `uvToRGBuv`: the chroma terms shared by both luma samples of a 4:2:2 pair.
fn uv_terms(u: u8, v: u8) -> (i32, i32, i32) {
    let uu = i32::from(u) - 128;
    let vv = i32::from(v) - 128;
    let half = 1 << (ITUR_BT601_SHIFT - 1);
    (
        half + ITUR_BT601_CVR * vv,
        half + ITUR_BT601_CVG * vv + ITUR_BT601_CUG * uu,
        half + ITUR_BT601_CUB * uu,
    )
}

/// `yRGBuvToRGBA`'s luma term and its `saturate_cast<uchar>` of the shifted sum.
fn y_term(y: u8) -> i32 {
    (i32::from(y) - 16).max(0) * ITUR_BT601_CY
}

fn sat(v: i32) -> u8 {
    (v >> ITUR_BT601_SHIFT).clamp(0, 255) as u8
}

/// One decoded frame: HWC, tightly packed, multi-byte elements little-endian.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    pub channels: ChannelFormat,
    pub dtype: ImageDType,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

fn malformed(why: String) -> CameraError {
    CameraError::MalformedImage(why)
}

/// Decodes `image` into tightly packed HWC bytes, rejecting anything malformed **before**
/// allocating (spec 25.1). Total: no [`Image`] value panics here.
pub fn decode_image(image: &Image, max_bytes: usize) -> Result<DecodedFrame, CameraError> {
    let e = encoding_of(&image.encoding)?;
    if matches!(e.pixel, Pixel::Yuv422 { .. }) && image.width % 2 != 0 {
        return Err(malformed(format!(
            "`{}` needs an even width, got {}",
            e.name, image.width
        )));
    }
    // Sizes in `u128`: `width * height * bpp` cannot overflow it, so a hostile message is
    // rejected by comparison rather than by wrapping.
    let row_bytes = u128::from(image.width) * u128::from(e.src_bpp);
    if u128::from(image.step) < row_bytes {
        return Err(malformed(format!(
            "step {} is shorter than {} bytes of `{}` pixels",
            image.step, row_bytes, e.name
        )));
    }
    let want = u128::from(image.step) * u128::from(image.height);
    if u128::try_from(image.data.len()).unwrap_or(u128::MAX) != want {
        return Err(malformed(format!(
            "data is {} bytes, expected step * height = {want}",
            image.data.len()
        )));
    }
    let out_len = u128::from(image.width) * u128::from(image.height) * u128::from(e.out_bpp);
    if out_len > u128::try_from(max_bytes).unwrap_or(u128::MAX) {
        return Err(malformed(format!(
            "decoded frame would be {out_len} bytes, over the {max_bytes} byte cap"
        )));
    }

    // Every cast below is now bounded by `data.len()`, which is a real allocation.
    let out_len = out_len as usize;
    let row_bytes = row_bytes as usize;
    let step = image.step as usize;
    let swap = image.is_bigendian != 0;
    let mut data = Vec::with_capacity(out_len);
    for y in 0..image.height as usize {
        let row = &image.data[y * step..y * step + row_bytes];
        match e.pixel {
            Pixel::Copy => data.extend_from_slice(row),
            Pixel::SwapRb => {
                for px in row.chunks_exact(e.out_bpp as usize) {
                    data.push(px[2]);
                    data.push(px[1]);
                    data.push(px[0]);
                    data.extend_from_slice(&px[3..]);
                }
            }
            Pixel::Endian if !swap => data.extend_from_slice(row),
            Pixel::Endian => {
                for el in row.chunks_exact(e.out_bpp as usize) {
                    data.extend(el.iter().rev());
                }
            }
            Pixel::Yuv422 { y_idx } => {
                // OpenCV's `YUV422toRGB888Invoker` with `uIdx = 0`: `uidx = 1 - yIdx`,
                // `vidx = (2 + uidx) % 4`.
                let (u_idx, v_idx) = (1 - y_idx, 3 - y_idx);
                for pair in row.chunks_exact(4) {
                    let (ruv, guv, buv) = uv_terms(pair[u_idx], pair[v_idx]);
                    for k in [0, 2] {
                        let y = y_term(pair[y_idx + k]);
                        data.push(sat(y + ruv));
                        data.push(sat(y + guv));
                        data.push(sat(y + buv));
                    }
                }
            }
        }
    }
    debug_assert_eq!(data.len(), out_len);
    Ok(DecodedFrame {
        channels: e.channels,
        dtype: e.dtype,
        width: image.width,
        height: image.height,
        data,
    })
}

// --- CameraInfo -> ImageSpec (design note sections 6.1, 6.2, 6.4) -------------------------

fn check_frame_id(expected: &str, found: &str) -> Result<(), CameraError> {
    if expected == found {
        return Ok(());
    }
    Err(CameraError::FrameIdMismatch {
        expected: expected.to_owned(),
        found: found.to_owned(),
    })
}

/// Monocular only: `r` is the identity and `p`'s `Tx`, `Ty` are zero (design note 6.1).
// Exact comparison is the rule: a stereo `r`/`Tx` differs from the identity by far more than
// any tolerance, and "close to the identity" is not a thing ROS publishes.
#[allow(clippy::float_cmp)]
fn check_monocular(info: &CameraInfo) -> Result<(), CameraError> {
    const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    if info.r != IDENTITY {
        return Err(CameraError::NotMonocular(format!(
            "r is not the identity: {:?}",
            info.r
        )));
    }
    if info.p[3] != 0.0 || info.p[7] != 0.0 {
        return Err(CameraError::NotMonocular(format!(
            "p has a baseline: Tx = {}, Ty = {}",
            info.p[3], info.p[7]
        )));
    }
    Ok(())
}

fn coeffs<const N: usize>(model: &str, d: &[f64]) -> Result<[f64; N], CameraError> {
    <[f64; N]>::try_from(d).map_err(|_| {
        CameraError::UnsupportedDistortion(format!(
            "`{model}` needs {N} coefficients, got {}",
            d.len()
        ))
    })
}

/// `distortion_model` + `d` -> (`DistortionModel`, `CameraModel`), design note section 6.1.
/// There is no `RationalPolynomial` variant in `es-ir-types` (design note open question 3), so
/// a non-zero `k4..k6` tail is rejected rather than silently dropped.
#[allow(clippy::float_cmp)]
fn distortion_of(info: &CameraInfo) -> Result<(DistortionModel, CameraModel), CameraError> {
    let model = info.distortion_model.as_str();
    let d = info.d.as_slice();
    match model {
        "plumb_bob" => {
            let c = coeffs::<5>(model, d)?;
            Ok((brown_conrady(&c), CameraModel::Pinhole))
        }
        "rational_polynomial" => {
            let c = coeffs::<8>(model, d)?;
            if c[5] != 0.0 || c[6] != 0.0 || c[7] != 0.0 {
                return Err(CameraError::UnsupportedDistortion(format!(
                    "`rational_polynomial` with a non-zero k4..k6 tail ({}, {}, {})",
                    c[5], c[6], c[7]
                )));
            }
            Ok((
                brown_conrady(&[c[0], c[1], c[2], c[3], c[4]]),
                CameraModel::Pinhole,
            ))
        }
        // image_geometry rectifies `equidistant` with `cv::fisheye`, i.e. Kannala-Brandt.
        "equidistant" => {
            let c = coeffs::<4>(model, d)?;
            Ok((
                DistortionModel::KannalaBrandt {
                    k1: c[0],
                    k2: c[1],
                    k3: c[2],
                    k4: c[3],
                },
                CameraModel::Fisheye,
            ))
        }
        "" if d.iter().all(|v| *v == 0.0) => Ok((DistortionModel::None, CameraModel::Pinhole)),
        other => Err(CameraError::UnsupportedDistortion(format!(
            "`{other}` with d = {d:?}"
        ))),
    }
}

/// `d = [k1, k2, t1, t2, k3]` (`CameraInfo.msg`, verbatim).
fn brown_conrady(c: &[f64; 5]) -> DistortionModel {
    DistortionModel::BrownConrady {
        k1: c[0],
        k2: c[1],
        k3: c[4],
        p1: c[2],
        p2: c[3],
    }
}

/// ROI then binning, exactly `image_geometry`'s order, but expressed as `ImageSpec` transforms so
/// the intrinsics cannot be forgotten (design note section 6.2, INV-14).
fn roi_and_binning(calib: &ImageSpec, info: &CameraInfo) -> Result<ImageSpec, CameraError> {
    let roi = info.roi;
    let full = roi.x_offset == 0 && roi.y_offset == 0 && roi.width == 0 && roi.height == 0;
    let (spec, w, h) = if full {
        (*calib, calib.width, calib.height)
    } else {
        if roi.width == 0 || roi.height == 0 {
            return Err(CameraError::RoiBinning(format!(
                "roi {}x{} has a zero dimension",
                roi.width, roi.height
            )));
        }
        let rect = Rect {
            x: roi.x_offset,
            y: roi.y_offset,
            width: roi.width,
            height: roi.height,
        };
        (calib.cropped(rect, true), roi.width, roi.height)
    };
    // "binning_x = binning_y = 0 is considered the same as binning_x = binning_y = 1".
    let (bx, by) = (info.binning_x.max(1), info.binning_y.max(1));
    if w % bx != 0 || h % by != 0 {
        return Err(CameraError::RoiBinning(format!(
            "roi {w}x{h} is not divisible by binning {bx}x{by}"
        )));
    }
    if bx == 1 && by == 1 {
        return Ok(spec);
    }
    Ok(spec.resized(w / bx, h / by, true))
}

/// Builds the `ImageSpec` a `CameraInfo` + `Image` pair describes (design note section 6.1).
/// The result is a *candidate*: [`check_declared`] decides whether it may be executed.
pub fn derive_spec(
    info: &CameraInfo,
    image: &Image,
    cfg: &CameraIngestConfig,
) -> Result<ImageSpec, CameraError> {
    check_frame_id(&cfg.frame_id, &info.header.frame_id)?;
    check_frame_id(&cfg.frame_id, &image.header.frame_id)?;
    check_monocular(info)?;
    let e = encoding_of(&image.encoding)?;
    if info.width == 0 || info.height == 0 {
        return Err(CameraError::SizeMismatch(format!(
            "CameraInfo carries no calibration resolution ({}x{})",
            info.width, info.height
        )));
    }
    // The only intrinsic writes in this file: `k` for a raw stream, `p` for a rectified one.
    // A rectified stream's `d` is not read at all — it describes the raw image, which this
    // stream is not (design note section 6.1).
    let (intrinsics, distortion, camera_model) = if cfg.rectified {
        let p = &info.p;
        let i = Intrinsics {
            fx: p[0],
            fy: p[5],
            cx: p[2],
            cy: p[6],
            skew: p[1],
        };
        (i, DistortionModel::None, CameraModel::Pinhole)
    } else {
        let k = &info.k;
        let i = Intrinsics {
            fx: k[0],
            fy: k[4],
            cx: k[2],
            cy: k[5],
            skew: k[1],
        };
        let (distortion, camera_model) = distortion_of(info)?;
        (i, distortion, camera_model)
    };

    let calib = ImageSpec {
        width: info.width,
        height: info.height,
        channels: e.channels,
        dtype: e.dtype,
        color_space: cfg.color_space,
        camera_model,
        intrinsics,
        extrinsics: cfg.extrinsics,
        distortion,
        shutter: cfg.shutter,
        exposure: cfg.exposure,
        rate_hz: cfg.rate_hz,
        depth_scale: e.depth_scale,
    };
    let spec = roi_and_binning(&calib, info)?;
    if spec.width == image.width && spec.height == image.height {
        return Ok(spec);
    }
    if !cfg.rescale_to_image {
        return Err(CameraError::SizeMismatch(format!(
            "calibration is {}x{} after roi/binning but the image is {}x{}; set \
             rescale_to_image to resize",
            spec.width, spec.height, image.width, image.height
        )));
    }
    if image.width == 0 || image.height == 0 {
        return Err(CameraError::SizeMismatch(format!(
            "cannot rescale to a {}x{} image",
            image.width, image.height
        )));
    }
    Ok(spec.resized(image.width, image.height, true))
}

// --- Derived vs declared (design note section 6.4, spec 26.1) -----------------------------

/// The closeness rule [`ImageSpec::intrinsics_consistent_with`] uses, so the field this names
/// is the field that failed that check.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn intrinsic_field(a: &Intrinsics, b: &Intrinsics) -> &'static str {
    for (name, x, y) in [
        ("intrinsics.fx", a.fx, b.fx),
        ("intrinsics.fy", a.fy, b.fy),
        ("intrinsics.cx", a.cx, b.cx),
        ("intrinsics.cy", a.cy, b.cy),
        ("intrinsics.skew", a.skew, b.skew),
    ] {
        if !close(x, y) {
            return name;
        }
    }
    "intrinsics"
}

fn first_mismatch(fields: &[(&'static str, f64, f64)]) -> Option<&'static str> {
    fields
        .iter()
        .find(|(_, x, y)| !close(*x, *y))
        .map(|(name, _, _)| *name)
}

fn distortion_field(a: DistortionModel, b: DistortionModel) -> Option<&'static str> {
    use DistortionModel::{BrownConrady, KannalaBrandt};
    match (a, b) {
        (DistortionModel::None, DistortionModel::None) => None,
        (
            BrownConrady { k1, k2, k3, p1, p2 },
            BrownConrady {
                k1: q1,
                k2: q2,
                k3: q3,
                p1: r1,
                p2: r2,
            },
        ) => first_mismatch(&[
            ("distortion.k1", k1, q1),
            ("distortion.k2", k2, q2),
            ("distortion.k3", k3, q3),
            ("distortion.p1", p1, r1),
            ("distortion.p2", p2, r2),
        ]),
        (
            KannalaBrandt { k1, k2, k3, k4 },
            KannalaBrandt {
                k1: q1,
                k2: q2,
                k3: q3,
                k4: q4,
            },
        ) => first_mismatch(&[
            ("distortion.k1", k1, q1),
            ("distortion.k2", k2, q2),
            ("distortion.k3", k3, q3),
            ("distortion.k4", k4, q4),
        ]),
        _ => Some("distortion"),
    }
}

/// Checks the derived spec against the Observation IR's declared one and names the first
/// differing field (`CAM-008`). The declared spec is never replaced by the derived one.
#[allow(clippy::float_cmp)]
pub fn check_declared(derived: &ImageSpec, declared: &ImageSpec) -> Result<(), CameraError> {
    let mismatch = |field| Err(CameraError::DeclaredMismatch(field));
    if derived.width != declared.width {
        return mismatch("width");
    }
    if derived.height != declared.height {
        return mismatch("height");
    }
    if derived.channels != declared.channels {
        return mismatch("channels");
    }
    if derived.dtype != declared.dtype {
        return mismatch("dtype");
    }
    if derived.color_space != declared.color_space {
        return mismatch("color_space");
    }
    if derived.camera_model != declared.camera_model {
        return mismatch("camera_model");
    }
    if derived.shutter != declared.shutter {
        return mismatch("shutter");
    }
    if derived.exposure != declared.exposure {
        return mismatch("exposure");
    }
    if derived.rate_hz != declared.rate_hz {
        return mismatch("rate_hz");
    }
    if derived.depth_scale != declared.depth_scale {
        return mismatch("depth_scale");
    }
    if derived.extrinsics != declared.extrinsics {
        return mismatch("extrinsics");
    }
    if !derived.intrinsics_consistent_with(declared) {
        return mismatch(intrinsic_field(&derived.intrinsics, &declared.intrinsics));
    }
    match distortion_field(derived.distortion, declared.distortion) {
        Some(field) => mismatch(field),
        None => Ok(()),
    }
}

// --- Time (design note section 6.5, spec 18.1) --------------------------------------------

fn stamp_ns(t: Time) -> i128 {
    i128::from(t.sec) * 1_000_000_000 + i128::from(t.nanosec)
}

/// Turns a `builtin_interfaces/Time` header stamp into an integer [`PhysTick`]. The caller
/// picks the epoch (normally the first stamp it accepted) and records it; the arithmetic is
/// `i128` throughout, so no float ever touches a time value (spec 3.4).
#[derive(Debug, Clone, Copy)]
pub struct StampClock {
    epoch_ns: i128,
    rate: TickRate,
}

impl StampClock {
    pub fn new(epoch: Time, rate: TickRate) -> Self {
        Self {
            epoch_ns: stamp_ns(epoch),
            rate,
        }
    }

    /// `floor((stamp - epoch) * num / (den * 10^9))`. A stamp before the epoch, or with
    /// `nanosec >= 10^9`, is `CAM-009`.
    pub fn tick(&self, stamp: Time) -> Result<PhysTick, CameraError> {
        if stamp.nanosec >= 1_000_000_000 {
            return Err(CameraError::BadStamp(format!(
                "nanosec {} is not below 10^9",
                stamp.nanosec
            )));
        }
        let delta = stamp_ns(stamp) - self.epoch_ns;
        if delta < 0 {
            return Err(CameraError::BadStamp(format!(
                "stamp {}.{:09} is before the clock epoch",
                stamp.sec, stamp.nanosec
            )));
        }
        let ticks =
            delta * i128::from(self.rate.num()) / (i128::from(self.rate.den()) * 1_000_000_000);
        u64::try_from(ticks)
            .map(PhysTick)
            .map_err(|_| CameraError::BadStamp(format!("{ticks} ticks does not fit a u64")))
    }
}

/// How old `sample` is at `now`, saturating at zero for a stamp from the future. This is the
/// `obs_age` the Safety Plane's `StaleObservation` watchdog reads (spec 9.4); a camera that
/// stops publishing trips it, and `SensorDropout`, with no new watchdog.
pub fn obs_age(now: Time, sample: Time) -> Micros {
    let delta = stamp_ns(now) - stamp_ns(sample);
    Micros(u64::try_from(delta / 1_000).unwrap_or(0))
}

// --- The ingest itself --------------------------------------------------------------------

/// One frame that passed every check.
#[derive(Debug, Clone, PartialEq)]
pub struct Accepted {
    pub frame: DecodedFrame,
    pub tick: PhysTick,
    pub spec: ImageSpec,
}

/// The stateful half: the latest `CameraInfo` wins, and every image is re-derived and
/// re-checked against the declared spec before it is decoded (design note section 6.4).
#[derive(Debug)]
pub struct CameraIngest {
    cfg: CameraIngestConfig,
    declared: ImageSpec,
    clock: StampClock,
    info: Option<CameraInfo>,
}

impl CameraIngest {
    pub fn new(cfg: CameraIngestConfig, declared: ImageSpec, clock: StampClock) -> Self {
        Self {
            cfg,
            declared,
            clock,
            info: None,
        }
    }

    pub fn config(&self) -> &CameraIngestConfig {
        &self.cfg
    }

    /// Latest `CameraInfo` wins. Every frame re-derives from it, so a content change is picked
    /// up without comparing messages.
    pub fn on_camera_info(&mut self, info: CameraInfo) {
        self.info = Some(info);
    }

    /// Validate, then decode: nothing is allocated for an image whose spec was rejected.
    pub fn on_image(&mut self, image: &Image) -> Result<Accepted, CameraError> {
        let info = self.info.as_ref().ok_or(CameraError::NotCalibrated)?;
        let spec = derive_spec(info, image, &self.cfg)?;
        check_declared(&spec, &self.declared)?;
        let tick = self.clock.tick(image.header.stamp)?;
        let frame = decode_image(image, MAX_FRAME_BYTES)?;
        Ok(Accepted { frame, tick, spec })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msg::Header;

    fn header() -> Header {
        Header {
            stamp: Time { sec: 1, nanosec: 2 },
            frame_id: "cam".to_owned(),
        }
    }

    fn image(encoding: &str, width: u32, height: u32, step: u32, data: Vec<u8>) -> Image {
        Image {
            header: header(),
            height,
            width,
            encoding: encoding.to_owned(),
            is_bigendian: 0,
            step,
            data,
        }
    }

    #[test]
    fn every_listed_encoding_resolves_and_nothing_else_does() {
        for e in &ENCODINGS {
            assert_eq!(encoding_of(e.name).unwrap().name, e.name);
        }
        for name in ["bayer_rggb8", "nv12", "8UC3", "rgb8 ", ""] {
            assert_eq!(encoding_of(name).unwrap_err().code(), "CAM-001");
        }
    }

    #[test]
    fn size_checks_run_before_the_allocation() {
        // step shorter than a row, wrong data length, odd YUV width, and a size that cannot
        // fit the cap: all rejected, none of them indexed into `data`.
        let cases = [
            image("rgb8", 3, 2, 8, vec![0; 16]),
            image("rgb8", 3, 2, 9, vec![0; 17]),
            image("uyvy", 3, 2, 6, vec![0; 12]),
            image("rgb8", u32::MAX, u32::MAX, u32::MAX, vec![0; 4]),
        ];
        for img in cases {
            assert_eq!(
                decode_image(&img, MAX_FRAME_BYTES).unwrap_err().code(),
                "CAM-002"
            );
        }
        // A frame over the caller's cap is refused rather than allocated.
        let big = image("rgb8", 3, 2, 9, vec![0; 18]);
        assert_eq!(decode_image(&big, 17).unwrap_err().code(), "CAM-002");
        assert!(decode_image(&big, 18).is_ok());
    }

    #[test]
    fn ticks_are_floor_of_integer_arithmetic() {
        let epoch = Time {
            sec: 10,
            nanosec: 0,
        };
        let clock = StampClock::new(epoch, TickRate::hz(1000));
        assert_eq!(clock.tick(epoch).unwrap(), PhysTick(0));
        assert_eq!(
            clock
                .tick(Time {
                    sec: 10,
                    nanosec: 999_999,
                })
                .unwrap(),
            PhysTick(0),
        );
        assert_eq!(
            clock
                .tick(Time {
                    sec: 11,
                    nanosec: 500_000,
                })
                .unwrap(),
            PhysTick(1000),
        );
        assert_eq!(
            clock.tick(Time { sec: 9, nanosec: 0 }).unwrap_err().code(),
            "CAM-009"
        );
        assert_eq!(
            clock
                .tick(Time {
                    sec: 11,
                    nanosec: 1_000_000_000,
                })
                .unwrap_err()
                .code(),
            "CAM-009"
        );
    }

    #[test]
    fn obs_age_saturates_at_zero() {
        let t = |sec, nanosec| Time { sec, nanosec };
        assert_eq!(obs_age(t(5, 0), t(4, 500_000_000)), Micros(500_000));
        assert_eq!(obs_age(t(4, 0), t(5, 0)), Micros(0));
        assert_eq!(obs_age(t(5, 0), t(5, 0)), Micros(0));
    }

    #[test]
    fn an_image_without_a_camera_info_is_not_executed() {
        let cfg = CameraIngestConfig {
            sensor: "cam".to_owned(),
            frame_id: "cam".to_owned(),
            extrinsics: Pose::IDENTITY,
            color_space: ColorSpace::SRgb,
            shutter: ShutterModel::Global,
            exposure: Duration::from_micros(500),
            rate_hz: 30.0,
            rectified: false,
            rescale_to_image: false,
        };
        let declared = ImageSpec {
            width: 3,
            height: 2,
            channels: ChannelFormat::Rgb,
            dtype: ImageDType::U8,
            color_space: ColorSpace::SRgb,
            camera_model: CameraModel::Pinhole,
            intrinsics: Intrinsics::new(1.0, 1.0, 1.5, 1.0),
            extrinsics: Pose::IDENTITY,
            distortion: DistortionModel::None,
            shutter: ShutterModel::Global,
            exposure: Duration::from_micros(500),
            rate_hz: 30.0,
            depth_scale: None,
        };
        let clock = StampClock::new(Time { sec: 0, nanosec: 0 }, TickRate::hz(30));
        let mut ingest = CameraIngest::new(cfg, declared, clock);
        let err = ingest
            .on_image(&image("rgb8", 3, 2, 9, vec![0; 18]))
            .unwrap_err();
        assert_eq!(err.code(), "CAM-008");
    }
}
