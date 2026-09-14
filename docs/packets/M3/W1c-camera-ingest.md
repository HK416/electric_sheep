# M3 W1c — camera ingest: `sensor_msgs/Image` + `CameraInfo` -> validated `ImageSpec`

Design note: `docs/design/ros2-boundary.md` section 6 (read 6.2 first: every size change goes through
`ImageSpec::cropped` / `resized`). Digest: `docs/api-notes/ros2-cdr.md` (layouts, encoding strings,
`CameraInfo` semantics). Depends on W1a (message structs); independent of W1b.

## context

```
crates/es-ros2/src/camera.rs
crates/es-ros2/src/lib.rs
crates/es-ros2/src/error.rs
crates/es-ros2/Cargo.toml
crates/es-ros2/tests/camera_ingest.rs
crates/es-ros2/tests/gen_camera_goldens.rs
crates/es-ros2/python/gen_camera_goldens.py
tests/golden/ros2/camera/**
Cargo.lock
docs/design/ros2-boundary.md
docs/packets/M3/W1c-camera-ingest.md
```

Notes: `camera.rs` carries its unit tests; `lib.rs` gets the module and re-exports only;
`error.rs` gets `CameraError` and the `CAM-*` codes; `Cargo.toml` adds `es-ir`, `es-math`,
`es-safety`. Goldens are additions only. The design note changes in section 6.3 only: record the
OpenCV fixed-point YUV constants used, with source path and OpenCV version.

## spec

- §7.2 and Appendix B.2, INV-14: resize and crop transform the intrinsics. This packet is the
  first code that builds an `ImageSpec` from outside data, so ROI, binning and size mismatches are
  exactly where INV-14 applies.
- §18.3: lens distortion follows the OpenCV convention; ROS `plumb_bob` and `equidistant` are
  OpenCV's pinhole and fisheye models.
- §18.1, §3.4: header stamps become integer `PhysTick`s; no float time.
- §9.4: stale or missing frames reach the existing `StaleObservation` / `SensorDropout` watchdogs
  through `obs_age` and `sensor_seen`; no new watchdog.
- §26.1: "what is not validated is not executed" -- the derived spec is checked against the
  declared one, never substituted.
- §25.1: image bytes cross a trust boundary.
- §28.5 W1 "real hardware camera drivers": met at the ROS 2 message boundary; an in-tree V4L2
  driver is not this packet (design note section 1).

## oracle

```
cargo fmt --check
cargo clippy -p es-ros2 --all-targets -- -D warnings
cargo test -p es-ros2 --test camera_ingest
cargo test -p es-ros2 --lib camera
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

Reference -- goldens are produced here, never by `es-ros2`:

```
ES_PYTHON=$HOME/envs/es-oracles/bin/python cargo test -p es-ros2 --test gen_camera_goldens -- --nocapture
ES_ROS2_ENV=$HOME/envs/ros2-kilted ES_PYTHON=$HOME/envs/es-oracles/bin/python \
  cargo test -p es-ros2 --test gen_camera_goldens -- --nocapture
```

`python/gen_camera_goldens.py <out_dir>` writes, with rosbags 0.11.5 and opencv-python-headless 5.0.0.93:

- CDR `.bin` + `.json` for every fixture below (`serialize_cdr(..., little_endian=True)`).
- `yuv/uyvy_8x2.rgb`, `yuv/yuyv_8x2.rgb`: `cv2.cvtColor(src, cv2.COLOR_YUV2RGB_UYVY)` / `COLOR_YUV2RGB_YUY2`.

With `--ros` (run as `scripts/ros2-env.sh "$ES_ROS2_ENV" python gen_camera_goldens.py --ros <out_dir>`):

- `intrinsics.json`: for each `CameraInfo` fixture, `image_geometry.PinholeCameraModel` ->
  `from_camera_info(msg)` -> `intrinsic_matrix()`, `projection_matrix()`.
- `cv_bridge.json`: `cv_bridge.CvBridge().imgmsg_to_cv2(msg, 'rgb8')` bytes for both YUV fixtures,
  which must equal the `cv2` files byte for byte (else the test fails and names the encoding).

`tests/gen_camera_goldens.rs` re-runs both modes into a temp dir and byte-compares with
`tests/golden/ros2/camera/`. A missing oracle prints `SKIP gen_camera_goldens: <why>` for that part;
when every part ran it prints `RAN gen_camera_goldens`. If `cv-bridge`/`image-geometry` cannot be
co-installed with the W1b environment, use `ES_ROS2_VISION_ENV` and record that here.

**Environment, measured 2026-09-14 (oracle server, `$HOME/envs/ros2-kilted`).** `cv-bridge` and
`image-geometry` *are* co-installed in the same prefix as the W1b `ros-base` / `rmw-zenoh-cpp`
environment — design note section 8's "co-installation is unverified" is now verified for this
prefix, and `ES_ROS2_VISION_ENV` is **not** needed. One real divergence was found and is recorded
in design note section 6.3: this `ros-kilted-cv-bridge 4.1.0` only recognizes the deprecated
`yuv422` / `yuv422_yuy2` encoding strings (`encoding_to_cvtype2("uyvy")` raises `Unrecognized
image encoding [uyvy]`), so `--ros` feeds cv_bridge those names for the same two fixtures.

**Deviation.** `crates/es-ros2/tests/gen_goldens.rs` (W1a's harness, outside this packet's
`context`) compares the *whole* `tests/golden/ros2/` file set against what `gen_ros2_goldens.py`
produces, so adding `tests/golden/ros2/camera/**` makes it fail wherever `ES_PYTHON` is set. One
line there now skips `camera/`, the way it already skips `rmw_zenoh/`.

Fixtures. `CameraInfo`: **A** 640x480 `plumb_bob`, `d = [-0.1, 0.01, 0.001, -0.002, 0.0005]`,
`k = [600,0,319.5, 0,610,239.5, 0,0,1]`, `r` identity, `p = [600,0,319.5,0, 0,610,239.5,0, 0,0,1,0]`;
**B** = A with `roi = {80, 60, 360, 480}` (x, y, height, width), `binning 2x2`, image 240x180;
**C** 1280x720 `rational_polynomial`, `k4..k6 = 0`; **D** = C with `k4 = 0.01`; **E** 848x800
`equidistant`, `d = [0.1, -0.02, 0.003, -0.0004]`; **F** = A with `p` != `k` (rectified stream).
`Image`: 3x2 `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`; `mono16` little- and big-endian; `16UC1`;
`32FC1`; 8x2 `uyvy` and `yuyv` covering Y 0..255 and U/V extremes; `rgb8` with `step = 3w + 2`.

`tests/camera_ingest.rs`:

- `plumb_bob_k_and_d_map_to_brown_conrady` -- A vs `intrinsics.json`, ≤ 1e-12.
- `rectified_stream_uses_p_and_clears_distortion` -- F.
- `roi_then_binning_goes_through_cropped_then_resized` -- B equals
  `calib.cropped(roi, true).resized(240, 180, true)` exactly, and image_geometry's `intrinsic_matrix()` within 1e-12.
- `roi_width_not_divisible_by_binning_is_rejected` (`CAM-006`).
- `image_size_mismatch_needs_explicit_rescale` (`CAM-007`; with `rescale_to_image` the result equals
  `spec.resized(w, h, true)` and satisfies `intrinsics_consistent_with`).
- `rational_polynomial_zero_tail_is_brown_conrady_nonzero_is_rejected` -- C ok, D `CAM-004`.
- `equidistant_is_kannala_brandt_fisheye` -- E.
- `non_identity_r_or_nonzero_tx_is_rejected` (`CAM-005`).
- `each_supported_encoding_decodes_to_hwc` -- exact bytes; `bgr` swapped; big-endian `mono16`
  swapped; step padding stripped; `depth_scale` 0.001 (`16UC1`) and 1.0 (`32FC1`).
- `uyvy_and_yuyv_match_opencv_bit_exact`.
- `unsupported_encodings_are_rejected` -- `bayer_rggb8`, `nv12`, `8UC3` (`CAM-001`).
- `malformed_step_length_or_size_is_rejected_before_allocation` (`CAM-002`; `width × height`
  overflow; proptest: decoding arbitrary `Image` values never panics).
- `frame_id_mismatch_is_rejected` (`CAM-003`).
- `derived_spec_must_match_the_declared_spec` (`CAM-008` naming `color_space`, `intrinsics.fx`, `distortion.k1`).
- `stamps_map_to_integer_ticks` -- proptest monotone; exact at 30000/1001 Hz; before epoch and
  `nanosec ≥ 10^9` rejected (`CAM-009`).
- `stale_and_missing_frames_trip_the_safety_plane` -- a `SafetyPlane` from a widened-envelope
  `DeploymentIr` with `StaleObservation` and `SensorDropout`; an old frame's `obs_age` raises
  `StaleObservation`; no `sensor_seen` beyond `max_gap` raises `SensorDropout`.

## acceptance

```rust
pub struct CameraIngestConfig {
    pub sensor: String, pub frame_id: String, pub extrinsics: es_math::conventions::Pose,
    pub color_space: ColorSpace, pub shutter: ShutterModel, pub exposure: Duration, pub rate_hz: f32,
    pub rectified: bool, pub rescale_to_image: bool,
}
pub fn derive_spec(info: &CameraInfo, image: &Image, cfg: &CameraIngestConfig) -> Result<ImageSpec, CameraError>;
pub fn check_declared(derived: &ImageSpec, declared: &ImageSpec) -> Result<(), CameraError>;
pub struct DecodedFrame { pub channels: ChannelFormat, pub dtype: ImageDType, pub width: u32, pub height: u32, pub data: Vec<u8> } // HWC, LE
pub fn decode_image(image: &Image, max_bytes: usize) -> Result<DecodedFrame, CameraError>;
pub struct StampClock { /* epoch, rate */ }
impl StampClock { pub fn new(epoch: Time, rate: TickRate) -> Self; pub fn tick(&self, stamp: Time) -> Result<PhysTick, CameraError>; }
pub fn obs_age(now: Time, sample: Time) -> es_safety::Micros;
pub struct CameraIngest { /* config, declared spec, latest CameraInfo */ }
impl CameraIngest {
    pub fn new(cfg: CameraIngestConfig, declared: ImageSpec, clock: StampClock) -> Self;
    pub fn on_camera_info(&mut self, info: CameraInfo);
    pub fn on_image(&mut self, image: &Image) -> Result<Accepted, CameraError>;   // Accepted { frame, tick, spec }
}
impl CameraError { pub fn code(&self) -> &'static str; }                         // "CAM-001" .. "CAM-009"
```

- Codes and mappings exactly as design note sections 6.1–6.5.
- The only intrinsic writes in `camera.rs` are the `k`/`p` indexing of section 6.1; every size
  change calls `ImageSpec::cropped` / `resized` with `rescale = true`.
- Stamp arithmetic in `i128`; no `f32`/`f64` time.
- No new trait, no `HashMap`, no new external dependency, ≤ ~700 source lines.

## forbidden

- `crates/es-ir-types`, `crates/es-ir` (no new `DistortionModel` variant -- design note open
  question 3), `crates/es-safety`, `crates/es-sensor`, every other crate.
- `session`, `config`, `actuator` (W1b); `hil` (W1d); the zenoh feature.
- TF, `CompressedImage`, pixel undistortion or rectification (the Observation IR `Undistort` node),
  Bayer demosaicing, an in-tree V4L2 driver.
- Editing goldens; computing expected values in Rust or in hand-written Python arithmetic instead of
  image_geometry, cv_bridge, OpenCV or rosbags.
