//! W1c camera ingest, judged against the checked-in reference goldens
//! (`tests/golden/ros2/camera/`, produced only by `python/gen_camera_goldens.py` — rosbags for
//! the CDR messages, `image_geometry` for the intrinsics under ROI/binning, `OpenCV` and
//! `cv_bridge` for the YUV conversions). Nothing here recomputes an expected intrinsic or an
//! expected YUV byte; the fixtures come off disk and the oracle's answer is the expectation.
//!
//! `docs/packets/M3/W1c-camera-ingest.md` names every test below;
//! `docs/design/ros2-boundary.md` section 6 is the rule each one checks.

use std::path::PathBuf;
use std::time::Duration;

use es_core::time::{PhysTick, TickRate};
use es_ir::deployment::{
    ActionContract, ActionSpace, Deadlines, DeploymentIr, ExecutionMode as DepExecutionMode,
    FallbackPolicy, Limit, Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope,
    Watchdog, WatchdogSet, Workspace,
};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Rect,
    ShutterModel,
};
use es_math::conventions::Pose;
use es_ros2::camera::{
    check_declared, decode_image, derive_spec, obs_age, CameraError, CameraIngest,
    CameraIngestConfig, StampClock, MAX_FRAME_BYTES,
};
use es_ros2::msg::{CameraInfo, Header, Image, Time};
use es_safety::{ActionChunk, ExecutionMode, SafetyPlane, ViolationKind};
use proptest::prelude::*;

// --- Fixtures -----------------------------------------------------------------------------

fn golden(rel: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/ros2/camera")
        .join(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn info(name: &str) -> CameraInfo {
    CameraInfo::from_cdr(&golden(&format!("cdr/{name}.bin"))).expect("golden CameraInfo decodes")
}

fn img(name: &str) -> Image {
    Image::from_cdr(&golden(&format!("cdr/{name}.bin"))).expect("golden Image decodes")
}

/// `image_geometry`'s answer for one fixture: `intrinsic_matrix` or `projection_matrix`.
fn matrix(name: &str, which: &str) -> Vec<Vec<f64>> {
    let json: serde_json::Value =
        serde_json::from_slice(&golden("intrinsics.json")).expect("intrinsics.json parses");
    serde_json::from_value(json[name][which].clone()).expect("a matrix of numbers")
}

#[track_caller]
fn assert_close(got: f64, want: f64) {
    assert!((got - want).abs() <= 1e-12, "{got} != {want}");
}

/// Checks `fx`, `skew`, `cx`, `fy`, `cy` against a 3x3 or 3x4 row-major reference matrix.
#[track_caller]
fn assert_matches_matrix(spec: &ImageSpec, m: &[Vec<f64>]) {
    assert_close(spec.intrinsics.fx, m[0][0]);
    assert_close(spec.intrinsics.skew, m[0][1]);
    assert_close(spec.intrinsics.cx, m[0][2]);
    assert_close(spec.intrinsics.fy, m[1][1]);
    assert_close(spec.intrinsics.cy, m[1][2]);
}

fn cfg() -> CameraIngestConfig {
    CameraIngestConfig {
        sensor: "camera".to_owned(),
        frame_id: "cam".to_owned(),
        extrinsics: Pose::IDENTITY,
        color_space: ColorSpace::SRgb,
        shutter: ShutterModel::Global,
        exposure: Duration::from_micros(500),
        rate_hz: 30.0,
        rectified: false,
        rescale_to_image: false,
    }
}

/// An `Image` with only the fields [`derive_spec`] reads: size, encoding and `frame_id`.
fn sized(width: u32, height: u32, encoding: &str) -> Image {
    Image {
        header: Header {
            stamp: Time { sec: 1, nanosec: 2 },
            frame_id: "cam".to_owned(),
        },
        height,
        width,
        encoding: encoding.to_owned(),
        is_bigendian: 0,
        step: 0,
        data: Vec::new(),
    }
}

fn code(e: &CameraError) -> &'static str {
    e.code()
}

// --- CameraInfo -> ImageSpec (design note sections 6.1, 6.2) ------------------------------

#[test]
fn plumb_bob_k_and_d_map_to_brown_conrady() {
    let spec = derive_spec(&info("info_a"), &sized(640, 480, "rgb8"), &cfg()).unwrap();
    assert_eq!((spec.width, spec.height), (640, 480));
    assert_matches_matrix(&spec, &matrix("info_a", "intrinsic_matrix"));
    // `d = [k1, k2, t1, t2, k3]`.
    assert_eq!(
        spec.distortion,
        DistortionModel::BrownConrady {
            k1: -0.1,
            k2: 0.01,
            k3: 0.0005,
            p1: 0.001,
            p2: -0.002,
        }
    );
    assert_eq!(spec.camera_model, CameraModel::Pinhole);
    // Everything ROS does not carry comes from the configuration, never from a default.
    assert_eq!(spec.color_space, ColorSpace::SRgb);
    assert_eq!(spec.shutter, ShutterModel::Global);
    assert_eq!(spec.exposure, Duration::from_micros(500));
    assert_eq!(spec.extrinsics, Pose::IDENTITY);
}

#[test]
fn rectified_stream_uses_p_and_clears_distortion() {
    let f = info("info_f");
    let mut rectified = cfg();
    rectified.rectified = true;
    let spec = derive_spec(&f, &sized(640, 480, "rgb8"), &rectified).unwrap();
    assert_matches_matrix(&spec, &matrix("info_f", "projection_matrix"));
    assert_eq!(spec.distortion, DistortionModel::None);
    assert_eq!(spec.camera_model, CameraModel::Pinhole);

    // The same message read as a raw stream uses `k` and keeps the distortion.
    let raw = derive_spec(&f, &sized(640, 480, "rgb8"), &cfg()).unwrap();
    assert_matches_matrix(&raw, &matrix("info_f", "intrinsic_matrix"));
    assert_ne!(raw.intrinsics, spec.intrinsics);
    assert!(matches!(
        raw.distortion,
        DistortionModel::BrownConrady { .. }
    ));
}

#[test]
fn roi_then_binning_goes_through_cropped_then_resized() {
    let base = derive_spec(&info("info_a"), &sized(640, 480, "rgb8"), &cfg()).unwrap();
    let want = base
        .cropped(
            Rect {
                x: 80,
                y: 60,
                width: 480,
                height: 360,
            },
            true,
        )
        .resized(240, 180, true);
    let got = derive_spec(&info("info_b"), &sized(240, 180, "rgb8"), &cfg()).unwrap();
    assert_eq!(
        got, want,
        "ROI then binning must be exactly these two transforms"
    );
    assert_eq!((got.width, got.height), (240, 180));
    // ... and image_geometry, which applies the ROI and the binning to `k` directly, agrees.
    assert_matches_matrix(&got, &matrix("info_b", "intrinsic_matrix"));
}

#[test]
fn roi_width_not_divisible_by_binning_is_rejected() {
    let mut wide = info("info_b");
    wide.roi.width = 361;
    assert_eq!(
        code(&derive_spec(&wide, &sized(240, 180, "rgb8"), &cfg()).unwrap_err()),
        "CAM-006"
    );
    let mut tall = info("info_b");
    tall.roi.height = 361;
    assert_eq!(
        code(&derive_spec(&tall, &sized(240, 180, "rgb8"), &cfg()).unwrap_err()),
        "CAM-006"
    );
}

/// `ImageSpec::cropped` trusts its rectangle, so the bound has to be checked here (spec 26.1).
#[test]
fn an_roi_outside_the_calibration_is_rejected() {
    // info_b: a 640x480 calibration, ROI 480x360+80+60, binning 2.
    let mut right = info("info_b");
    right.roi.x_offset = 10_000;
    let err = derive_spec(&right, &sized(240, 180, "rgb8"), &cfg()).unwrap_err();
    assert_eq!(code(&err), "CAM-006");
    assert!(err.to_string().contains("640x480"), "{err}");

    let mut below = info("info_b");
    below.roi.y_offset = 300; // 300 + 360 > 480
    let err = derive_spec(&below, &sized(240, 180, "rgb8"), &cfg()).unwrap_err();
    assert_eq!(code(&err), "CAM-006");
    assert!(err.to_string().contains("640x480"), "{err}");

    // The rectangle that ends exactly on the last row/column is inside.
    let mut flush = info("info_b");
    flush.roi.x_offset = 160;
    flush.roi.y_offset = 120;
    derive_spec(&flush, &sized(240, 180, "rgb8"), &cfg()).expect("480x360+160+120 fits 640x480");
}

/// `CAM-006` is one code for three rules, so the message has to say which one refused
/// (design note section 6.2).
#[test]
fn roi_rejections_name_which_rule_failed() {
    let image = sized(240, 180, "rgb8");
    let message = |mutate: fn(&mut CameraInfo)| {
        let mut bad = info("info_b");
        mutate(&mut bad);
        let err = derive_spec(&bad, &image, &cfg()).unwrap_err();
        assert_eq!(code(&err), "CAM-006");
        err.to_string()
    };
    let zero = message(|i| i.roi.width = 0);
    let outside = message(|i| i.roi.x_offset = 10_000);
    let indivisible = message(|i| i.roi.width = 361);
    assert_ne!(zero, outside);
    assert_ne!(zero, indivisible);
    assert_ne!(outside, indivisible);
}

/// An uncalibrated monocular ROS driver publishes `r = [0; 9]`; it means the identity
/// (design note section 6.1).
#[test]
fn an_all_zero_r_is_the_identity() {
    let image = sized(640, 480, "rgb8");
    let want = derive_spec(&info("info_a"), &image, &cfg()).unwrap();

    let mut zeroed = info("info_a");
    zeroed.r = [0.0; 9];
    let got = derive_spec(&zeroed, &image, &cfg()).unwrap();
    assert_eq!(got, want);
    assert_matches_matrix(&got, &matrix("info_a", "intrinsic_matrix"));
}

#[test]
fn image_size_mismatch_needs_explicit_rescale() {
    let a = info("info_a");
    assert_eq!(
        code(&derive_spec(&a, &sized(320, 240, "rgb8"), &cfg()).unwrap_err()),
        "CAM-007"
    );

    let mut rescale = cfg();
    rescale.rescale_to_image = true;
    let got = derive_spec(&a, &sized(320, 240, "rgb8"), &rescale).unwrap();
    let calib = derive_spec(&a, &sized(640, 480, "rgb8"), &cfg()).unwrap();
    assert_eq!(got, calib.resized(320, 240, true));
    assert!(got.intrinsics_consistent_with(&calib));
}

#[test]
fn rational_polynomial_zero_tail_is_brown_conrady_nonzero_is_rejected() {
    let c = derive_spec(&info("info_c"), &sized(1280, 720, "rgb8"), &cfg()).unwrap();
    assert_eq!(
        c.distortion,
        DistortionModel::BrownConrady {
            k1: -0.05,
            k2: 0.02,
            k3: 0.001,
            p1: 0.0003,
            p2: -0.0007,
        }
    );
    assert_eq!(c.camera_model, CameraModel::Pinhole);
    assert_matches_matrix(&c, &matrix("info_c", "intrinsic_matrix"));

    // D is C with k4 = 0.01: there is no rational-polynomial variant to hold it, and dropping
    // it silently would change the projection (design note open question 3).
    let err = derive_spec(&info("info_d"), &sized(1280, 720, "rgb8"), &cfg()).unwrap_err();
    assert_eq!(code(&err), "CAM-004");
}

#[test]
fn equidistant_is_kannala_brandt_fisheye() {
    let e = derive_spec(&info("info_e"), &sized(848, 800, "rgb8"), &cfg()).unwrap();
    assert_eq!(
        e.distortion,
        DistortionModel::KannalaBrandt {
            k1: 0.1,
            k2: -0.02,
            k3: 0.003,
            k4: -0.0004,
        }
    );
    assert_eq!(e.camera_model, CameraModel::Fisheye);
    assert_matches_matrix(&e, &matrix("info_e", "intrinsic_matrix"));
}

#[test]
fn non_identity_r_or_nonzero_tx_is_rejected() {
    let image = sized(640, 480, "rgb8");
    for mutate in [
        |i: &mut CameraInfo| i.r[0] = 0.999,
        |i: &mut CameraInfo| i.r[1] = 0.01,
        |i: &mut CameraInfo| {
            i.r = {
                let mut r = [0.0; 9];
                r[8] = 1.0;
                r
            }
        },
        |i: &mut CameraInfo| i.p[3] = 0.05,
        |i: &mut CameraInfo| i.p[7] = -0.05,
    ] {
        let mut bad = info("info_a");
        mutate(&mut bad);
        assert_eq!(
            code(&derive_spec(&bad, &image, &cfg()).unwrap_err()),
            "CAM-005"
        );
    }
}

#[test]
fn frame_id_mismatch_is_rejected() {
    let mut other_info = info("info_a");
    other_info.header.frame_id = "other_optical_frame".to_owned();
    assert_eq!(
        code(&derive_spec(&other_info, &sized(640, 480, "rgb8"), &cfg()).unwrap_err()),
        "CAM-003"
    );

    let mut other_image = sized(640, 480, "rgb8");
    other_image.header.frame_id = "other_optical_frame".to_owned();
    assert_eq!(
        code(&derive_spec(&info("info_a"), &other_image, &cfg()).unwrap_err()),
        "CAM-003"
    );
}

// --- Encodings (design note section 6.3) --------------------------------------------------

fn decode(name: &str) -> es_ros2::camera::DecodedFrame {
    decode_image(&img(name), MAX_FRAME_BYTES).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn each_supported_encoding_decodes_to_hwc() {
    let rgb = img("image_rgb8");
    let out = decode("image_rgb8");
    assert_eq!((out.width, out.height), (3, 2));
    assert_eq!(
        (out.channels, out.dtype),
        (ChannelFormat::Rgb, ImageDType::U8)
    );
    assert_eq!(out.data, rgb.data);

    // `bgr8` carries the same bytes with R and B exchanged.
    let bgr = decode("image_bgr8");
    let swapped: Vec<u8> = rgb
        .data
        .chunks(3)
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    assert_eq!(bgr.data, swapped);
    assert_eq!(bgr.channels, ChannelFormat::Rgb);

    let rgba = img("image_rgba8");
    assert_eq!(decode("image_rgba8").data, rgba.data);
    assert_eq!(decode("image_rgba8").channels, ChannelFormat::Rgba);
    let bgra_swapped: Vec<u8> = rgba
        .data
        .chunks(4)
        .flat_map(|p| [p[2], p[1], p[0], p[3]])
        .collect();
    assert_eq!(decode("image_bgra8").data, bgra_swapped);

    let mono = decode("image_mono8");
    assert_eq!(mono.data, img("image_mono8").data);
    assert_eq!(
        (mono.channels, mono.dtype),
        (ChannelFormat::Gray, ImageDType::U8)
    );

    // A big-endian `mono16` arrives byte-swapped and leaves little-endian, i.e. identical to
    // the little-endian fixture of the same values.
    let le = decode("image_mono16_le");
    let be = decode("image_mono16_be");
    assert_eq!(le.data, img("image_mono16_le").data);
    assert_eq!(be.data, le.data);
    assert_ne!(be.data, img("image_mono16_be").data);
    assert_eq!(
        (le.channels, le.dtype),
        (ChannelFormat::Gray, ImageDType::U16)
    );

    let depth16 = decode("image_16uc1");
    assert_eq!(depth16.data, le.data);
    assert_eq!(
        (depth16.channels, depth16.dtype),
        (ChannelFormat::Depth, ImageDType::U16)
    );
    let depth32 = decode("image_32fc1");
    assert_eq!(depth32.data, img("image_32fc1").data);
    assert_eq!(
        (depth32.channels, depth32.dtype),
        (ChannelFormat::Depth, ImageDType::F32)
    );

    // `step = 3w + 2`: the two padding bytes per row are stripped.
    let padded = img("image_rgb8_padded");
    assert_eq!(padded.step, 3 * padded.width + 2);
    assert_eq!(decode("image_rgb8_padded").data, rgb.data);

    // `depth_scale` is REP-118's, and it lives on the spec, not on the frame.
    let mut rescale = cfg();
    rescale.rescale_to_image = true;
    let scale_of = |name: &str| {
        derive_spec(&info("info_a"), &img(name), &rescale)
            .unwrap()
            .depth_scale
    };
    assert_eq!(scale_of("image_16uc1"), Some(0.001));
    assert_eq!(scale_of("image_32fc1"), Some(1.0));
    assert_eq!(scale_of("image_rgb8"), None);
}

#[test]
fn uyvy_and_yuyv_match_opencv_bit_exact() {
    let bridge: serde_json::Value =
        serde_json::from_slice(&golden("cv_bridge.json")).expect("cv_bridge.json parses");
    for (name, file) in [
        ("image_uyvy", "yuv/uyvy_8x2.rgb"),
        ("image_yuyv", "yuv/yuyv_8x2.rgb"),
    ] {
        let out = decode(name);
        assert_eq!((out.width, out.height), (8, 2));
        assert_eq!(
            (out.channels, out.dtype),
            (ChannelFormat::Rgb, ImageDType::U8)
        );
        assert_eq!(out.data, golden(file), "{name} does not match OpenCV");
        let want_hex = bridge[name]["rgb8_hex"].as_str().unwrap();
        let want: Vec<u8> = (0..want_hex.len() / 2)
            .map(|i| u8::from_str_radix(&want_hex[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        assert_eq!(out.data, want, "{name} does not match cv_bridge");
    }
    // The two fixtures carry the same (u, y0, v, y1) quads in the two byte orders, so they
    // must decode identically.
    assert_eq!(decode("image_uyvy").data, decode("image_yuyv").data);

    // The deprecated spellings are the same layout and the same conversion; they are also the
    // only names this RoboStack cv_bridge accepts.
    for (modern, deprecated) in [("image_uyvy", "yuv422"), ("image_yuyv", "yuv422_yuy2")] {
        let mut alias = img(modern);
        alias.encoding = deprecated.to_owned();
        assert_eq!(
            decode_image(&alias, MAX_FRAME_BYTES).unwrap().data,
            decode(modern).data
        );
    }
}

#[test]
fn unsupported_encodings_are_rejected() {
    for encoding in ["bayer_rggb8", "nv12", "8UC3", "rgb16", "yuv420sp", ""] {
        let mut bad = img("image_rgb8");
        bad.encoding = encoding.to_owned();
        assert_eq!(
            code(&decode_image(&bad, MAX_FRAME_BYTES).unwrap_err()),
            "CAM-001",
            "{encoding}"
        );
        // The same string is refused before any spec is derived, too.
        assert_eq!(
            code(&derive_spec(&info("info_a"), &bad, &cfg()).unwrap_err()),
            "CAM-001"
        );
    }
}

#[test]
fn malformed_step_length_or_size_is_rejected_before_allocation() {
    let short_step = |i: &mut Image| i.step = 8;
    let short_data = |i: &mut Image| {
        i.data.pop();
    };
    let long_data = |i: &mut Image| i.data.push(0);
    for mutate in [short_step, short_data, long_data] {
        let mut bad = img("image_rgb8");
        mutate(&mut bad);
        assert_eq!(
            code(&decode_image(&bad, MAX_FRAME_BYTES).unwrap_err()),
            "CAM-002"
        );
    }

    // An odd width cannot carry 4:2:2 pairs.
    let mut odd = img("image_uyvy");
    odd.width = 7;
    assert_eq!(
        code(&decode_image(&odd, MAX_FRAME_BYTES).unwrap_err()),
        "CAM-002"
    );

    // `width * height` overflows every machine word: the size check is what rejects it, and it
    // runs before the allocation, so this returns instead of aborting.
    let mut huge = img("image_rgb8");
    huge.width = u32::MAX;
    huge.height = u32::MAX;
    huge.step = u32::MAX;
    assert_eq!(
        code(&decode_image(&huge, MAX_FRAME_BYTES).unwrap_err()),
        "CAM-002"
    );
    // A frame that fits the message but not the caller's cap is refused, not allocated.
    assert_eq!(
        code(&decode_image(&img("image_rgb8"), 17).unwrap_err()),
        "CAM-002"
    );
}

proptest! {
    /// Spec 25.1: image bytes are untrusted, so decoding is total.
    #[test]
    fn decoding_an_arbitrary_image_never_panics(
        width in prop_oneof![0u32..12, Just(u32::MAX), 60_000u32..70_000],
        height in prop_oneof![0u32..12, Just(u32::MAX), 60_000u32..70_000],
        step in prop_oneof![0u32..48, Just(u32::MAX)],
        is_bigendian in 0u8..3,
        encoding in prop::sample::select(vec![
            "rgb8", "bgr8", "rgba8", "bgra8", "mono8", "mono16", "16UC1", "32FC1", "uyvy",
            "yuv422", "yuyv", "yuv422_yuy2", "bayer_rggb8", "",
        ]),
        data in prop::collection::vec(any::<u8>(), 0..96),
    ) {
        let image = Image {
            header: Header {
                stamp: Time { sec: 1, nanosec: 2 },
                frame_id: "cam".to_owned(),
            },
            height,
            width,
            encoding: encoding.to_owned(),
            is_bigendian,
            step,
            data,
        };
        let _ = decode_image(&image, MAX_FRAME_BYTES);
    }
}

// --- Derived vs declared (design note section 6.4) ----------------------------------------

#[test]
fn derived_spec_must_match_the_declared_spec() {
    let derived = derive_spec(&info("info_a"), &sized(640, 480, "rgb8"), &cfg()).unwrap();
    check_declared(&derived, &derived).expect("a spec always matches itself");

    let named = |declared: &ImageSpec| match check_declared(&derived, declared).unwrap_err() {
        CameraError::DeclaredMismatch(field) => field,
        other => panic!("expected CAM-008, got {other}"),
    };

    let mut other = derived;
    other.color_space = ColorSpace::Linear;
    assert_eq!(named(&other), "color_space");

    let mut other = derived;
    other.intrinsics.fx += 1.0;
    assert_eq!(named(&other), "intrinsics.fx");

    let mut other = derived;
    let DistortionModel::BrownConrady { k1, .. } = &mut other.distortion else {
        panic!("fixture A is plumb_bob");
    };
    *k1 += 0.5;
    assert_eq!(named(&other), "distortion.k1");

    let mut other = derived;
    other.width = 320;
    assert_eq!(named(&other), "width");
    assert_eq!(
        check_declared(&derived, &other).unwrap_err().code(),
        "CAM-008"
    );
}

// --- Time (design note section 6.5) -------------------------------------------------------

fn ntsc_clock() -> StampClock {
    StampClock::new(
        Time {
            sec: 100,
            nanosec: 0,
        },
        TickRate::rational(30_000, 1001).unwrap(),
    )
}

#[test]
fn stamps_map_to_integer_ticks() {
    let clock = ntsc_clock();
    let at = |sec, nanosec| clock.tick(Time { sec, nanosec });

    // 1001 s at 30000/1001 Hz is exactly 30000 ticks, and one frame period lands on tick 1
    // the nanosecond it is reached, never before (floor, in i128).
    assert_eq!(at(1101, 0).unwrap(), PhysTick(30_000));
    assert_eq!(at(100, 0).unwrap(), PhysTick(0));
    assert_eq!(at(100, 33_366_666).unwrap(), PhysTick(0));
    assert_eq!(at(100, 33_366_667).unwrap(), PhysTick(1));

    assert_eq!(code(&at(99, 999_999_999).unwrap_err()), "CAM-009");
    assert_eq!(code(&at(0, 0).unwrap_err()), "CAM-009");
    assert_eq!(code(&at(101, 1_000_000_000).unwrap_err()), "CAM-009");
    assert_eq!(code(&at(101, u32::MAX).unwrap_err()), "CAM-009");
}

proptest! {
    #[test]
    fn ticks_are_monotone_in_the_stamp(a in 0i64..4_000_000_000i64, b in 0i64..4_000_000_000i64) {
        let clock = ntsc_clock();
        let at = |ns: i64| clock.tick(Time {
            sec: 100 + (ns / 1_000_000_000) as i32,
            nanosec: (ns % 1_000_000_000) as u32,
        }).unwrap();
        let (lo, hi) = (a.min(b), a.max(b));
        prop_assert!(at(lo) <= at(hi));
    }
}

// --- The Safety Plane sees a stalled camera (spec 9.4, design note section 6.5) ------------

/// A deliberately wide envelope: INV-12 forbids switching the plane off, so a test that needs
/// room widens the limits instead. Only the two watchdogs under test are armed.
fn deployment_ir() -> DeploymentIr {
    let rate = RateSpec {
        control: TickRate::hz(1000),
        inference: TickRate::hz(100),
    };
    DeploymentIr {
        schema_version: 1,
        robot: RobotRef {
            name: "camera_rig".to_owned(),
            target: RobotTarget::Physical {
                driver: "can0".to_owned(),
            },
            n_joints: 2,
        },
        action: ActionContract {
            space: ActionSpace::JointPosition,
            dim: 2,
            horizon: 4,
            execute_chunk: 4,
        },
        safety: SafetyEnvelope {
            position: vec![Limit::symmetric(10.0); 2],
            position_soft_margin: vec![0.05; 2],
            velocity_max: vec![100.0; 2],
            acceleration_max: vec![1000.0; 2],
            torque_max: vec![1000.0; 2],
            jerk_max: None,
            action_rate: RateLimit {
                first_diff_max: vec![10.0; 2],
                second_diff_max: vec![10.0; 2],
            },
            workspace: Workspace::Box {
                min: [-10.0, -10.0, -10.0],
                max: [10.0, 10.0, 10.0],
            },
            ee_velocity_max: 100.0,
            min_self_distance: 0.001,
            min_env_distance: 0.001,
            contact_force_max: 1000.0,
        },
        execution: DepExecutionMode::RecedingHorizon,
        deadlines: Deadlines {
            observation_age: Micros(100_000),
            inference_budget: Micros(50_000),
            actuation_budget: Micros(500),
        },
        watchdogs: WatchdogSet(vec![
            Watchdog::StaleObservation {
                max_age: Micros(100_000),
            },
            Watchdog::SensorDropout {
                sensor: "camera".to_owned(),
                max_gap: Micros(50_000),
            },
        ]),
        fallback: FallbackPolicy::HoldPosition,
        rate,
    }
}

#[test]
fn stale_and_missing_frames_trip_the_safety_plane() {
    let mut plane = SafetyPlane::<2, 4>::from_ir(&deployment_ir()).expect("a valid IR");
    let mut seq = 0u64;
    let mut step = |plane: &mut SafetyPlane<2, 4>, age, now| {
        seq += 1;
        let chunk = ActionChunk::<2, 4>::new([[0.0; 2]; 4], 4, ExecutionMode::RecedingHorizon)
            .with_seq(seq);
        plane.validate(&chunk, age, now)
    };

    let now = Time {
        sec: 10,
        nanosec: 0,
    };
    let fresh = Time {
        sec: 9,
        nanosec: 950_000_000,
    };
    let stale = Time { sec: 9, nanosec: 0 };
    assert_eq!(obs_age(now, fresh), Micros(50_000));
    assert_eq!(obs_age(now, stale), Micros(1_000_000));

    // A frame that just arrived: the camera is seen at this tick and the age is inside the
    // envelope, so neither watchdog fires.
    plane.sensor_seen("camera", PhysTick(0));
    let out = step(&mut plane, obs_age(now, fresh), PhysTick(0));
    assert!(!out.events.contains(ViolationKind::StaleObservation));
    assert!(!out.events.contains(ViolationKind::SensorDropout));
    assert!(out.is_clean());

    // The same camera, one second behind: `obs_age` alone raises StaleObservation.
    plane.sensor_seen("camera", PhysTick(1));
    let out = step(&mut plane, obs_age(now, stale), PhysTick(1));
    assert!(out.events.contains(ViolationKind::StaleObservation));
    assert!(!out.events.contains(ViolationKind::SensorDropout));

    // No frame at all for longer than `max_gap` (50 ms at a 1 kHz control period is 50 ticks;
    // the last frame was seen at tick 1): SensorDropout, without a camera-specific watchdog
    // anywhere in this crate.
    let out = step(&mut plane, obs_age(now, fresh), PhysTick(60));
    assert!(out.events.contains(ViolationKind::SensorDropout));
    assert!(!out.events.contains(ViolationKind::StaleObservation));

    // A frame arrives again and clears it.
    plane.sensor_seen("camera", PhysTick(61));
    let out = step(&mut plane, obs_age(now, fresh), PhysTick(61));
    assert!(!out.events.contains(ViolationKind::SensorDropout));
}

/// The public entry point end to end: validate against the declared spec, date the frame, then
/// decode it, and hand the tick to the Safety Plane as `sensor_seen` (design note section 6.5).
#[test]
fn camera_ingest_validates_then_decodes_and_dates_a_frame() {
    let mut cfg = cfg();
    cfg.rescale_to_image = true;
    let frame = img("image_rgb8");
    let declared = derive_spec(&info("info_a"), &frame, &cfg).unwrap();
    let clock = StampClock::new(Time { sec: 0, nanosec: 0 }, TickRate::hz(1000));
    let mut ingest = CameraIngest::new(cfg, declared, clock);

    // Nothing is executed before a CameraInfo has arrived -- and that is its own code, not
    // the "calibrated for another stream" one below (design note section 6.4).
    assert_eq!(code(&ingest.on_image(&frame).unwrap_err()), "CAM-010");

    ingest.on_camera_info(info("info_a"));
    let accepted = ingest.on_image(&frame).unwrap();
    assert_eq!(accepted.spec, declared);
    assert_eq!(accepted.tick, PhysTick(1000)); // stamp 1.000000002 s at 1 kHz
    assert_eq!(accepted.frame.data, frame.data);

    let mut plane = SafetyPlane::<2, 4>::from_ir(&deployment_ir()).expect("a valid IR");
    plane.sensor_seen("camera", accepted.tick);
    let chunk = ActionChunk::<2, 4>::new([[0.0; 2]; 4], 4, ExecutionMode::RecedingHorizon);
    let out = plane.validate(&chunk, Micros(0), accepted.tick);
    assert!(!out.events.contains(ViolationKind::SensorDropout));

    // A CameraInfo whose calibration no longer matches the declared spec stops the stream.
    ingest.on_camera_info(info("info_c"));
    assert_eq!(code(&ingest.on_image(&frame).unwrap_err()), "CAM-008");
}

/// "No `CameraInfo` yet" is a startup race a caller retries through; "calibrated for another
/// stream" stops the line. They are not the same condition and do not share a code
/// (design note section 6.4).
#[test]
fn no_camera_info_is_its_own_code() {
    let mut cfg = cfg();
    cfg.rescale_to_image = true;
    let frame = img("image_rgb8");
    let declared = derive_spec(&info("info_a"), &frame, &cfg).unwrap();
    let clock = StampClock::new(Time { sec: 0, nanosec: 0 }, TickRate::hz(1000));
    let mut ingest = CameraIngest::new(cfg, declared, clock);

    assert_eq!(code(&ingest.on_image(&frame).unwrap_err()), "CAM-010");

    ingest.on_camera_info(info("info_c"));
    assert_eq!(code(&ingest.on_image(&frame).unwrap_err()), "CAM-008");
}
