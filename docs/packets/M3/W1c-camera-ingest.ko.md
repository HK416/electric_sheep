<!-- Korean translation of docs/packets/M3/W1c-camera-ingest.md. The English file is the working copy; regenerate this when it changes. -->

# M3 W1c — 카메라 인제스트: `sensor_msgs/Image` + `CameraInfo` -> 검증된 `ImageSpec`

Design note: `docs/design/ros2-boundary.md` 섹션 6(6.2를 먼저 읽을 것: 모든 크기 변경은
`ImageSpec::cropped` / `resized`를 거친다). 다이제스트: `docs/api-notes/ros2-cdr.md`(레이아웃,
encoding 문자열, `CameraInfo` 의미론). W1a(메시지 struct)에 의존; W1b와는 독립적.

## context (범위)

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

참고: `camera.rs`는 자신의 단위 테스트를 갖는다; `lib.rs`는 모듈과 re-export만 얻는다;
`error.rs`는 `CameraError`와 `CAM-*` 코드를 얻는다; `Cargo.toml`은 `es-ir`, `es-math`,
`es-safety`를 추가한다. golden은 추가만 된다. 설계 노트는 섹션 6.3에서만 바뀐다: 사용된
OpenCV 고정소수점 YUV 상수를, 출처 경로와 OpenCV 버전과 함께 기록한다.

## spec (사양)

- §7.2와 부록 B.2, INV-14: resize와 crop은 intrinsics를 변환한다. 이 패킷은 외부 데이터로부터
  `ImageSpec`을 만드는 첫 코드이므로, ROI, binning, 크기 불일치가 정확히 INV-14가 적용되는
  지점이다.
- §18.3: 렌즈 왜곡은 OpenCV 컨벤션을 따른다; ROS의 `plumb_bob`과 `equidistant`는 OpenCV의
  pinhole과 fisheye 모델이다.
- §18.1, §3.4: header stamp는 정수 `PhysTick`이 된다; float 시간 없음.
- §9.4: 오래되거나 누락된 프레임은 `obs_age`와 `sensor_seen`을 통해 기존의
  `StaleObservation` / `SensorDropout` watchdog에 도달한다; 새 watchdog 없음.
- §26.1: "검증되지 않은 것은 실행되지 않는다" -- 유도된 spec은 선언된 것과 대조 검사되며,
  결코 대체되지 않는다.
- §25.1: 이미지 바이트는 신뢰 경계를 넘는다.
- §28.5 W1 "real hardware camera drivers": ROS 2 메시지 경계에서 충족됨; 인트리 V4L2
  드라이버는 이 패킷이 아니다(설계 노트 섹션 1).

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-ros2 --all-targets -- -D warnings
cargo test -p es-ros2 --test camera_ingest
cargo test -p es-ros2 --lib camera
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

레퍼런스 -- golden은 여기서 생성되며, `es-ros2`에서는 절대 생성되지 않는다:

```
ES_PYTHON=$HOME/envs/es-oracles/bin/python cargo test -p es-ros2 --test gen_camera_goldens -- --nocapture
ES_ROS2_ENV=$HOME/envs/ros2-kilted ES_PYTHON=$HOME/envs/es-oracles/bin/python \
  cargo test -p es-ros2 --test gen_camera_goldens -- --nocapture
```

`python/gen_camera_goldens.py <out_dir>`가 rosbags 0.11.5와 opencv-python-headless 5.0.0.93으로
쓰는 것:

- 아래의 모든 fixture에 대한 CDR `.bin` + `.json`(`serialize_cdr(..., little_endian=True)`).
- `yuv/uyvy_8x2.rgb`, `yuv/yuyv_8x2.rgb`: `cv2.cvtColor(src, cv2.COLOR_YUV2RGB_UYVY)` / `COLOR_YUV2RGB_YUY2`.

`--ros`와 함께(`scripts/ros2-env.sh "$ES_ROS2_ENV" python gen_camera_goldens.py --ros
<out_dir>`로 실행):

- `intrinsics.json`: 각 `CameraInfo` fixture에 대해, `image_geometry.PinholeCameraModel` ->
  `from_camera_info(msg)` -> `intrinsic_matrix()`, `projection_matrix()`.
- `cv_bridge.json`: 두 YUV fixture 각각에 대한 `cv_bridge.CvBridge().imgmsg_to_cv2(msg,
  'rgb8')` 바이트이며, `cv2` 파일과 바이트 단위로 같아야 한다(다르면 테스트가 실패하며
  encoding을 지목한다).

`tests/gen_camera_goldens.rs`는 두 모드를 임시 디렉터리에 다시 실행하고
`tests/golden/ros2/camera/`와 바이트 단위로 비교한다. 오라클이 없으면 그 부분에 대해
`SKIP gen_camera_goldens: <why>`를 출력한다; 모든 부분이 실행되면 `RAN gen_camera_goldens`를
출력한다. `cv-bridge`/`image-geometry`를 W1b 환경과 공존 설치할 수 없으면
`ES_ROS2_VISION_ENV`를 쓰고 여기에 기록한다.

**환경, 2026-09-14 측정(오라클 서버, `$HOME/envs/ros2-kilted`).** `cv-bridge`와
`image-geometry`는 W1b의 `ros-base` / `rmw-zenoh-cpp` 환경과 같은 prefix에 *실제로*
공존 설치되어 있다 — 설계 노트 섹션 8의 "co-installation is unverified"는 이제 이
prefix에 대해 검증되었으며, `ES_ROS2_VISION_ENV`는 **필요 없다**. 실제 차이 하나가
발견되었고 설계 노트 섹션 6.3에 기록되어 있다: 이 `ros-kilted-cv-bridge 4.1.0`은
deprecated된 `yuv422` / `yuv422_yuy2` encoding 문자열만 인식한다(`encoding_to_cvtype2("uyvy")`는
`Unrecognized image encoding [uyvy]`를 일으킨다), 그래서 `--ros`는 같은 두 fixture에
대해 cv_bridge에 그 이름들을 넘긴다.

**Deviation(벗어난 점).** `crates/es-ros2/tests/gen_goldens.rs`(W1a의 harness, 이 패킷의
`context` 밖)는 `tests/golden/ros2/` 파일 집합 *전체*를 `gen_ros2_goldens.py`가 만드는 것과
비교하므로, `tests/golden/ros2/camera/**`를 추가하면 `ES_PYTHON`이 설정된 곳마다 실패하게
된다. 그래서 거기에 한 줄을 추가해 이미 `rmw_zenoh/`를 건너뛰는 것과 같은 방식으로
`camera/`도 건너뛰게 했다.

Fixture. `CameraInfo`: **A** 640x480 `plumb_bob`, `d = [-0.1, 0.01, 0.001, -0.002, 0.0005]`,
`k = [600,0,319.5, 0,610,239.5, 0,0,1]`, `r` identity, `p = [600,0,319.5,0, 0,610,239.5,0,
0,0,1,0]`; **B** = A에 `roi = {80, 60, 360, 480}`(x, y, height, width), `binning 2x2`, 이미지
240x180를 더함; **C** 1280x720 `rational_polynomial`, `k4..k6 = 0`; **D** = C에 `k4 = 0.01`;
**E** 848x800 `equidistant`, `d = [0.1, -0.02, 0.003, -0.0004]`; **F** = A에 `p` != `k`(rectified
stream)를 더함. `Image`: 3x2 `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`; little-, big-endian
`mono16`; `16UC1`; `32FC1`; Y 0..255와 U/V 극단값을 아우르는 8x2 `uyvy`와 `yuyv`;
`step = 3w + 2`인 `rgb8`.

`tests/camera_ingest.rs`:

- `plumb_bob_k_and_d_map_to_brown_conrady` -- A 대 `intrinsics.json`, ≤ 1e-12.
- `rectified_stream_uses_p_and_clears_distortion` -- F.
- `roi_then_binning_goes_through_cropped_then_resized` -- B는
  `calib.cropped(roi, true).resized(240, 180, true)`와 정확히 같고, image_geometry의
  `intrinsic_matrix()`와 1e-12 이내로 같다.
- `roi_width_not_divisible_by_binning_is_rejected`(`CAM-006`).
- `image_size_mismatch_needs_explicit_rescale`(`CAM-007`; `rescale_to_image`을 쓰면 결과가
  `spec.resized(w, h, true)`와 같고 `intrinsics_consistent_with`를 만족한다).
- `rational_polynomial_zero_tail_is_brown_conrady_nonzero_is_rejected` -- C ok, D `CAM-004`.
- `equidistant_is_kannala_brandt_fisheye` -- E.
- `non_identity_r_or_nonzero_tx_is_rejected`(`CAM-005`).
- `each_supported_encoding_decodes_to_hwc` -- 정확한 바이트; `bgr` 교환됨; big-endian
  `mono16` 교환됨; step padding 제거됨; `depth_scale` 0.001(`16UC1`)과 1.0(`32FC1`).
- `uyvy_and_yuyv_match_opencv_bit_exact`.
- `unsupported_encodings_are_rejected` -- `bayer_rggb8`, `nv12`, `8UC3`(`CAM-001`).
- `malformed_step_length_or_size_is_rejected_before_allocation`(`CAM-002`; `width × height`
  overflow; proptest: 임의의 `Image` 값을 디코딩해도 절대 panic하지 않음).
- `frame_id_mismatch_is_rejected`(`CAM-003`).
- `derived_spec_must_match_the_declared_spec`(`CAM-008`이 `color_space`, `intrinsics.fx`,
  `distortion.k1`을 지목).
- `stamps_map_to_integer_ticks` -- proptest로 단조성 검증; 30000/1001 Hz에서 정확함; epoch
  이전과 `nanosec ≥ 10^9`는 거부됨(`CAM-009`).
- `stale_and_missing_frames_trip_the_safety_plane` -- `StaleObservation`과 `SensorDropout`이
  있는, envelope가 넓혀진 `DeploymentIr`로부터 만든 `SafetyPlane`; 오래된 프레임의
  `obs_age`가 `StaleObservation`을 일으킨다; `max_gap`을 넘도록 `sensor_seen`이 없으면
  `SensorDropout`을 일으킨다.

## acceptance (수용 기준)

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

- 코드와 매핑은 설계 노트 섹션 6.1–6.5 그대로.
- `camera.rs`에서 intrinsic을 쓰는 유일한 곳은 섹션 6.1의 `k`/`p` 인덱싱이다; 모든 크기
  변경은 `rescale = true`로 `ImageSpec::cropped` / `resized`를 호출한다.
- Stamp 연산은 `i128`로; `f32`/`f64` 시간 없음.
- 새 trait 없음, `HashMap` 없음, 새 외부 의존성 없음, 소스 코드 ~700줄 이하.

## forbidden (금지)

- `crates/es-ir-types`, `crates/es-ir`(새 `DistortionModel` variant 없음 -- 설계 노트 미결
  질문 3), `crates/es-safety`, `crates/es-sensor`, 그 외 모든 crate.
- `session`, `config`, `actuator`(W1b); `hil`(W1d); zenoh feature.
- TF, `CompressedImage`, pixel undistortion이나 rectification(Observation IR의 `Undistort`
  node), Bayer demosaicing, 인트리 V4L2 드라이버.
- golden 편집; image_geometry, cv_bridge, OpenCV, rosbags 대신 Rust나 손으로 작성한 Python
  연산으로 기댓값을 계산하는 것.
