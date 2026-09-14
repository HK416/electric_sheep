<!-- Korean translation of docs/design/ros2-boundary.md. The English file is the working copy; regenerate this when it changes. -->

# ROS 2 경계, 카메라 인제스트, HIL (`es-ros2`) — 설계

Spec: §24.1 (ROS 2 경계, 모드 A/B/C), §24.2 (HIL), §28.5 (M3 W1), §28.7 (게이트 14), §7.2와
부록 B.2 (`ImageSpec`), §18.1 (정수 시간), §18.3 (OpenCV 왜곡 컨벤션), §9.3–§9.5와
부록 B.4 (Safety Plane), §4.2 (계층화), §25.1 (보안), §26.2 (CI 등급), §1.4
(오라클), §1.9 (범위), §12.4 (미검증 수치).
불변식: INV-12, INV-13, INV-14, INV-17.
다이제스트: `docs/api-notes/rmw-zenoh.md`, `docs/api-notes/ros2-cdr.md`, `docs/api-notes/zenoh-rs.md`.
패킷: `docs/packets/M3/W1a-ros2-cdr-keyexpr.md`, `W1b-ros2-zenoh-session.md`,
`W1c-camera-ingest.md`, `W1d-hil.md`.

설계 노트의 섹션은 "section N"으로 인용하며, `§N`은 항상 spec을 가리킨다.

## 1. 범위

| 항목 (§28.5 W1) | 패킷 | 오라클 | W1 이후 |
|---|---|---|---|
| CDR, 메시지 서브셋, key expression, liveliness, attachment, 타입 해시 | W1a | rosbags, RoboStack JSON, rmw_zenoh design.md | 어디서나 검증 가능 |
| zenoh 세션, 모드 A, A ⊕ B 배타, actuator 경로 | W1b | loopback peer (PR 등급); 라이브 rmw_zenoh (`ES_ROS2_ENV`) | Linux에서 검증 가능, sudo 불필요 |
| 모드 B 데이터 경로 | W1b (코드만) | bridge key-mapping 단위 테스트 | 라이브: `Target / Status: unverified` |
| 모드 C (`RustDDS` + `ros2-client`) | 없음 | — | 빌드되지 않음; config가 거부한다(§24.1은 이를 experimental로 표시) |
| 카메라: `sensor_msgs/Image` + `CameraInfo` -> `ImageSpec` | W1c | rosbags, image_geometry, cv_bridge, OpenCV | 어디서나 / Linux에서 검증 가능 |
| HIL: UDP 링크, deadline/jitter 텔레메트리, 입력 로그, replay 게이트 | W1d | loopback 컨트롤러; live == replay | 어디서나 검증 가능 |
| `es hil replay`, `es --check-deps` 안의 ROS 2 항목 | W1e (미작성) | CLI 테스트 | — |
| 모드 B 라이브 오라클 | W1f (미작성) | zenoh-bridge-ros2dds + `rmw_cyclonedds_cpp` | — |
| 인트리 V4L2/USB 카메라 드라이버 | 없음 | 카메라가 필요 | `Target / Status: unverified` |
| 로봇 셀, 물리 컨트롤러 대상 HIL, 게이트 14 | 없음 | 하드웨어(§26.2 릴리스 등급) | `Target / Status: unverified` |
| 지연시간, jitter, 처리량 수치 전반 | — | 하드웨어 | `Target / Status: unverified` |

물리 카메라는 `sensor_msgs/Image`를 publish하는 기존 ROS 2 카메라 드라이버를 통해 들어온다;
W1은 그 메시지 경계에서 §28.5의 "real hardware camera drivers"를 충족한다. 인트리 V4L2
드라이버는 미뤄진다: 카메라 없이는 아무것도 그것을 판단할 수 없기 때문이다(§1.4).

## 2. 배치, 의존성, feature

**Crate.** `crates/es-ros2`, layer 11, `xtask/src/layering.rs`의 `LAYERS`에 이미 있음. 하위
계층 의존성만 가짐: `es-core` (1: `PhysTick`, `TickRate`), `es-math` (0: `Pose`), `es-ir` (6:
`es_ir::image::ImageSpec`, `DeploymentIr`), `es-safety` (8: `SafeAction`, `SafetyPlane`),
`es-runtime-embedded` (9, `default-features = false`: `es-policy` 없는 `core_rt::EmbeddedCore`),
`es-telemetry` (10). 외부: `serde`, `toml`, `thiserror`, `blake3` (workspace), `xxhash-rust`
(W1a), `zenoh` (W1b, optional).

**HIL은 `es-hil` crate가 아니라 모듈 `es_ros2::hil`이다.** §24.2는 `es-hil`이라는 이름을 쓰지만
§4.2의 표에는 없고, `cargo xtask layering`은 `LAYERS`에 없는 `es-*` crate를 모두 실패시킨다
("unknown es-* crate not in the spec 4.2 LAYERS table"). 새로 추가하는 것은 §4.2 개정이며, 사람의
결정이다(섹션 10). 기각된 후보지: `es-env` (9)는 §24.2가 요구하는 `es-telemetry` (10)를 쓸 수
없다; `es-transport` (11)는 §21의 텐서 전송이자 CUDA/HIP을 링크할 수 있는 유일한 crate인데,
HIL 장치가 그것을 물려받아서는 안 된다. `es-ros2`는 §24가 이미 지목한 layer-11 sim-to-real
crate다. 나중에 분리 추출할 때 파일 이동만으로 끝나도록, `hil`은 ROS 모듈로부터 아무것도
import하지 않는다(W1d 테스트가 이를 강제한다).

**Feature `zenoh`, 기본값 off.** 코덱, config, camera, HIL은 이것 없이 빌드된다; `es`와 임베디드
소비자는 절대 zenoh를 링크하지 않는다. W1b는 `cargo xtask ci`가 clippy와 테스트에
`--features es-ros2/zenoh`를 넘기게 만들어서, PR 등급이 **인프로세스 loopback peer**를 상대로
컴파일·린트·세션 실행을 한다(listen `tcp/127.0.0.1:<port>`, multicast scouting off, 라우터 없음,
외부 네트워크 없음). pin은 `zenoh = "=1.8.0"`(ROS 2의 `zenoh_cpp_vendor`가 빌드하는 버전)이며,
`default-features = false`, `features = ["transport_tcp"]`다: rustls/ring/quinn 없음(§25.1:
기본으로는 localhost 너머로 아무것도 listen하지 않는다; 원격 링크용 TLS는 이 crate의 일이
아니다). `.cargo/config.toml`은 `[resolver] incompatible-rust-versions = "fallback"`을 얻어서
새 의존성 트리가 `rust-version = "1.85"` 안에서 해석되게 한다.

**새 trait 없음(INV-17).** 메시지 디스패치는 `enum MsgType` / `enum Msg`다; 링크는 구체
struct다; plant는 매 틱마다 `q`, `qd`를 넘기는 호출자다.

## 3. 와이어 코덱 (W1a)

- **CDR.** `CdrWriter`: little-endian, header `00 01 00 00`, 정렬은 header 다음 바이트부터
  측정, trailing padding 없음. `CdrReader`: `00 00`/`00 01`을 받아들이고, options 바이트는
  무시하며, ≤ 3바이트의 trailing byte를 허용한다. 규칙과 golden byte: `ros2-cdr.md`.
- **신뢰 경계(§25.1).** 네트워크 바이트는 신뢰할 수 없다: 디코딩은 total이며(절대 panic하지
  않음, property test로 검증됨), 할당 **전에** `count × element_size ≤ remaining`을 검사하고,
  메시지를 `MAX_MESSAGE_BYTES = 64 MiB`(텔레메트리 프레임 상한)로 제한하며, NUL로 끝나는
  UTF-8 문자열을 요구한다.
- **Message**는 `to_cdr` / `from_cdr`를 갖는 평범한 struct다. `MsgType`은 const 테이블로부터
  `ros_name()`, `dds_name()`, `rihs01()`을 제공한다.
- **타입 해시는 계산이 아니라 테이블이다.** 서브셋은 고정되어 있다; 테이블은 RoboStack의
  `share/<pkg>/msg/<T>.json`과 rosbags의 `hash_rihs01`을 상대로 golden 검사되며, 11개 타입
  모두 일치한다. 임의 타입에 대한 REP-2016 해시 계산은 필요 없다.
- **이름.** 절대 ROS 이름만 허용(`/` 다음 `/`로 구분된 `[A-Za-z_][A-Za-z0-9_]*`); `~`와 상대
  이름은 호출자가 해석한다. Topic key: `<domain>/<앞뒤로 '/' 하나씩 제거한 name>/<dds type>/<RIHS01>`.
- **Liveliness.** `LivelinessToken` 인코딩 + total parse(`NN`, `MP`, `MS`; `SS`/`SC`는 parse만);
  rmw_zenoh 기본값 생략을 반영한 QoS chunk; `rmw-zenoh.md`에 있는 그대로의 접두사로
  `classify(key) -> TokenScheme::{RmwZenoh, Ros2DdsBridge, Other}`.
- **Attachment.** `Attachment { seq: i64, source_timestamp_ns: i64, gid: [u8; 16] }`, 정확히 33
  바이트; `gid_of(token_key_expr)` = XXH3-128, low 64비트 LE 다음 high 64비트 LE.

## 4. 세션과 모드 (W1b)

### 4.1 설정

```toml
[ros2]
domain_id = 0
node = "es_runtime"
namespace = "/"
enclave = "/"
liveliness_timeout_ms = 1000

[ros2.rmw_zenoh]              # mode A; implied when no mode table is present
mode = "peer"                 # "peer" | "client"
connect = ["tcp/localhost:7447"]
listen = ["tcp/localhost:0"]

# [ros2.dds_bridge]           # mode B; mutually exclusive with [ros2.rmw_zenoh]
# connect = ["tcp/localhost:7447"]
# bridge_namespace = "/"      # the bridge's own `namespace` setting

# [ros2.rust_dds]             # mode C; parsed only so the rejection is specific

[[ros2.actuator]]
topic = "/forward_position_controller/commands"
joints = ["j1", "j2", "j3"]
```

모드 A의 기본값은 `DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5`(peer, `tcp/localhost:7447`, multicast
off)를 그대로 따라서, 같은 호스트의 순정 `rmw_zenohd`가 아무 설정 없이도 발견된다. 알 수 없는
key는 거부된다.

### 4.2 A ⊕ B, 두 번 강제됨 (§24.1)

| 코드 | 조건 | 효과 |
|---|---|---|
| `ROS2-001` | `[ros2.rmw_zenoh]`와 `[ros2.dds_bridge]` 둘 다 있음 | config 거부, 세션 열리지 않음 |
| `ROS2-002` | `[ros2.rust_dds]` | config 거부: 모드 C는 빌드되지 않음 |
| `ROS2-003` | 모드 A이고 `liveliness().get("@/*/@ros2_lv/**")`가 토큰을 반환 | `Ros2Node::open` 실패 |
| `ROS2-004` | 모드 B이고 `liveliness().get("@/*/@ros2_lv")`가 아무것도 반환하지 않음 | `open` 실패: bridge 없음 |
| `ROS2-005` | 모드 B이고 `liveliness().get("@ros2_lv/<domain>/**")`가 토큰을 반환 | `open` 실패 |
| `ROS2-006` | 시작 후, 충돌하는 접두사에 대한 liveliness subscriber가 토큰을 봄 | node가 충돌을 래치함; 이후의 모든 `put`/`send`가 그것을 반환(fail closed) |
| `ROS2-010` | `[[ros2.actuator]]` topic에 대한 일반 `Publisher` | 거부됨(섹션 4.5) |
| `ROS2-011` | publisher의 타입이 아닌 `Msg`의 `put` | 거부됨 |
| `ROS2-012` | TRANSIENT_LOCAL 요청됨 | 거부됨(섹션 4.3) |

probe는 node가 자신의 토큰을 하나라도 선언하기 전에 실행되며, `liveliness_timeout_ms`로
제한된다. query는 그 순간에 도달 가능한 것만 본다, 그래서 `ROS2-006`이 존재한다. actuator
topic에서 fail closed하면 하위 컨트롤러는 명령을 받지 못하게 되는데, 이는 그쪽 자신의
timeout이 처리한다; 검사를 우회한 명령이 전송되는 일은 결코 없다. 코드는 `Ros2Error::code()`에서
나온다; 이들은 IR 진단이 아니라 런타임 조건이므로 `es-ir-types::codes`에 들어가지 않는다.

### 4.3 모드 A 데이터 경로

- `Ros2Node::open(&Ros2Config)`: zenoh `open(..).wait()`, probe, `NN` 토큰 선언.
- `publisher(name, MsgType, depth)`: topic key에 publisher를 선언하고 `MP` 토큰도 선언한다.
  `Publisher::put(&Msg)`는 타입을 검사하고, CDR을 인코딩하고, `(seq from 1, SystemTime ns, gid)`를
  attach하며, encoding은 설정하지 않는다.
- `subscriber(name, MsgType, depth)`: subscriber와 `MS` 토큰을 선언한다; 전달은 유계 FIFO를
  통한다(`recv_timeout`). 형식이 올바른 33바이트 attachment가 없는 sample은 rmw_zenoh와
  마찬가지로 버려지고 카운트된다.
- QoS: RELIABLE, VOLATILE, KEEP_LAST(depth). TRANSIENT_LOCAL은 zenoh-ext advanced pub/sub가
  필요한데 그 key 리터럴은 미검증이다 -> `ROS2-012`.
- wall-clock stamp는 wire 메타데이터다; Safety Plane 쪽에서는 아무것도 이를 읽지 않는다(§3.4).

### 4.4 모드 B 데이터 경로

Key = `bridge_namespace` + 앞의 `/`를 뺀 name(플러그인의 `ros2_name_to_key_expr`), CDR
payload, 우리 자신의 토큰 없음(bridge가 DDS discovery를 소유), attachment 없음. 인용된 매핑만을
상대로 단위 테스트됨; live: W1f까지 `Target / Status: unverified`.

### 4.5 Actuator 경로 (INV-12)

`[[ros2.actuator]]` topic은 예약되어 있다. 유일한 writer는
`ActuatorPublisher::<NJ>::send(&SafeAction<NJ>)`이며, `std_msgs/msg/Float64MultiArray { layout: {
dim: [], data_offset: 0 }, data: q }`를 publish한다: 이는 ros2_control의
`forward_command_controller`의 명령 타입이며(`using CmdType = std_msgs::msg::Float64MultiArray;`,
`"~/commands"`에 subscribe), 크기 불일치를 거부한다("command size (%zu) does not match number of
interfaces (%zu)"). `SafeAction`은 오직 Safety Plane에서만 나오므로, 이 crate는 검증되지 않은
action을 actuator topic에 올릴 방법을 전혀 제공하지 않는다. `NJ != joints.len()`은 생성 시점에
실패한다. 들어오는 `sensor_msgs/JointState`는 `SafetyPlane::observe_state` / `sensor_seen`에
도달하기 전에 설정된 joint 순서로 이름 기준 재정렬된다(joint가 하나라도 빠지면 sample이
거부된다). 이름은 있지만 sample이 그 joint의 `position`/`velocity` 값을 담고 있지 않은 경우도
같은 방식으로 거부된다(`ROS2-013`, `PartialJointState`) — 기본값은 측정값이 아니다(§25.1).
메시지 정의가 강제하는 유일한 예외: 완전히 비어 있는 `velocity`(`sensor_msgs/JointState`가
"may be empty"로 문서화한 배열)는 "이 드라이버는 velocity를 보고하지 않는다"로 읽어
`qd = [0.0; NJ]`가 되고, 비어 있지 않지만 길이가 모자란 `velocity`는 부분 sample이므로
거부된다. `position`에는 그런 해석이 없다: 짧거나 비어 있으면 항상 오류다.

## 5. 메시지 서브셋

| 타입 | 용도 |
|---|---|
| `std_msgs/String` | `demo_nodes_cpp` talker/listener와의 interop smoke test |
| `sensor_msgs/JointState` | 로봇 상태 입력 |
| `std_msgs/Float64MultiArray` (+ `MultiArrayLayout`, `MultiArrayDimension`) | 검증된 명령 출력 |
| `sensor_msgs/Image`, `sensor_msgs/CameraInfo` (+ `RegionOfInterest`) | 카메라 인제스트 |
| `std_msgs/Header`, `builtin_interfaces/Time` | 중첩(nested) |

W1에 없는 것: `trajectory_msgs/JointTrajectory`(해시는 나중을 위해 기록됨), TF(extrinsics는
설정에서 옴), `CompressedImage`, 서비스, 액션, TRANSIENT_LOCAL topic.

## 6. 카메라 인제스트 (W1c)

출력: Observation IR이 선언한 것과 **대조 검사되는**(결코 대체되지 않는) `ImageSpec`, HWC 바이트
버퍼, `PhysTick`, 그리고 Safety Plane 입력인 `obs_age`와 `sensor_seen`.

### 6.1 `CameraInfo` -> `ImageSpec`

| 필드 | 출처 |
|---|---|
| `width`, `height` | `CameraInfo.width/height`(캘리브레이션 해상도), 그다음 섹션 6.2 |
| `intrinsics`, raw stream | `k` row-major: `fx = k[0]`, `skew = k[1]`, `cx = k[2]`, `fy = k[4]`, `cy = k[5]` |
| `intrinsics`, `rectified = true` | `p`: `fx = p[0]`, `skew = p[1]`, `cx = p[2]`, `fy = p[5]`, `cy = p[6]`; `distortion = None` |
| `plumb_bob` | `d = [k1, k2, t1, t2, k3]` -> `BrownConrady { k1, k2, k3, p1: t1, p2: t2 }`, `Pinhole` |
| `rational_polynomial` | `d = [k1, k2, p1, p2, k3, k4, k5, k6]`; `k4 = k5 = k6 = 0.0`이면 정확히 `BrownConrady`, 아니면 `CAM-004`(rational variant는 존재하지 않음; 섹션 10) |
| `equidistant` | `d = [k1, k2, k3, k4]` -> `KannalaBrandt`, `Fisheye`(image_geometry는 이를 위해 `cv::fisheye`를 사용) |
| `d`가 all-zero인 `""` | `None` |
| `extrinsics`, `color_space`, `shutter`, `exposure`, `rate_hz` | `CameraIngestConfig`; ROS는 이들 중 어떤 것도 갖고 있지 않으며 default를 임의로 만들어내지 않는다 |
| `channels`, `dtype`, `depth_scale` | encoding, 섹션 6.3 |

단안(monocular)만: `r`은 identity여야 하고 `p[3]`, `p[7]`(`Tx`, `Ty`)은 0이어야 한다(`CAM-005`).
all-zero `r`—보정되지 않은 단안 카메라에 대해 ROS 드라이버가 publish하는 값—은 identity로
읽는다; 받아들이는 형태는 이 둘뿐이며, 허용 오차는 도입하지 않는다.

### 6.2 ROI, binning, 크기 (INV-14)

image_geometry(`pinhole_camera_model.cpp`, 그리고 Python `from_camera_info`)는 ROI를 먼저
적용하고 그다음 binning을 적용한다: `k[0,2] = (k[0,2] - roi.x_offset) / binning_x`,
`k[0,0] /= binning_x`; binning 0은 1이다; all-zero ROI는 전체 해상도다. Ingest는 **`ImageSpec`
transform을 통해** 동일하게 한다:

```
spec = calib.cropped(Rect { roi }, true)                          // skipped for an all-zero ROI
spec = spec.resized(roi.width / bx, roi.height / by, true)       // skipped when bx = by = 1
```

- `CAM-006`은 "ROI/binning 쌍을 쓸 수 없다"는 뜻이며, 세 가지 규칙을 포괄하고 각각 고유한
  메시지를 갖는다: `roi.width`/`roi.height`가 0인 경우; 사각형이 calibration 밖으로 나가는
  경우(`roi.x_offset + roi.width > CameraInfo.width`, `y`도 마찬가지, 넘침이 없도록 `u64`로
  더한다) — `ImageSpec::cropped`는 원점을 무조건 빼기 때문에, 검사하지 않은 사각형은 `cx`가
  음수인 그럴듯한 spec을 만들어낸다; 그리고 `roi.width % bx != 0`(resize 비율이 `1 / bx`가
  아니게 된다).
- 결과는 `Image.width/height`와 같아야 한다; 그렇지 않으면 `CAM-007`이다, 단 config가
  `rescale_to_image = true`를 설정한 경우는 예외이며, 그 경우 `resized(image.width, image.height,
  true)`를 한 번 더 적용한다.
- `rescale = false`는 절대 전달되지 않는다. `camera` 안의 어떤 코드도 섹션 6.1의 `k`/`p`
  인덱싱을 제외하면 intrinsic을 쓰지 않는다.
- 알려진 차이점: image_geometry는 binning 아래에서 `k[0,1]`(skew)을 스케일하지 않은 채로
  남겨둔다; 부록 B.2의 `Intrinsics::scaled`는 그것을 스케일한다. Fixture는 skew 0을 쓴다;
  skew가 0이 아닌 경우에는 spec의 규칙이 우선한다.

### 6.3 Encoding

| `encoding` | `channels` / `dtype` | 변환 | `depth_scale` |
|---|---|---|---|
| `rgb8` | `Rgb` / `U8` | 없음 | — |
| `bgr8` | `Rgb` / `U8` | R, B 교환 | — |
| `rgba8` / `bgra8` | `Rgba` / `U8` | 없음 / R, B 교환 | — |
| `mono8` | `Gray` / `U8` | 없음 | — |
| `mono16` | `Gray` / `U16` | `is_bigendian`이면 LE로 변환 | — |
| `16UC1` | `Depth` / `U16` | `is_bigendian`이면 LE로 변환; 0은 0으로 유지 | `0.001`(REP-118: "depth in millimeters"; "The value 0 denotes an invalid depth") |
| `32FC1` | `Depth` / `F32` | `is_bigendian`이면 LE로 변환 | `1.0`(REP-118: "depth ... in meters") |
| `uyvy`, `yuv422` | `Rgb` / `U8` | `cv::COLOR_YUV2RGB_UYVY`와 비트 단위로 동일 | — |
| `yuyv`, `yuv422_yuy2` | `Rgb` / `U8` | `cv::COLOR_YUV2RGB_YUY2`와 비트 단위로 동일 | — |
| 그 외 | `CAM-001` | | |

cv_bridge는 `yuv422 -> cv::COLOR_YUV2RGB_UYVY`와 `yuv422_yuy2 -> cv::COLOR_YUV2RGB_YUY2`를
매핑한다(`cv_bridge.cpp`). 할당 전 검사(§25.1): `step ≥ width × bpp`,
`data.len() == step × height`, `width × height × bpp ≤ max_bytes`, YUV는 width가 짝수여야
함(`CAM-002`). `width × bpp`를 넘는 행 패딩은 제거된다.

**YUV 4:2:2 상수(W1c).** `crates/es-ros2/src/camera.rs`는 OpenCV의 고정소수점 BT.601 정수를
`modules/imgproc/src/color_yuv.simd.hpp`(OpenCV 5.0.0, `opencv-python-headless 5.0.0.93` 안의
버전이며 `tests/golden/ros2/camera/yuv/*.rgb`를 만들어낸 바로 그 버전)에서 그대로 복사한다:

| 상수 | 값 |
|---|---|
| `ITUR_BT_601_SHIFT` | 20 |
| `ITUR_BT_601_CY` | 1220542 |
| `ITUR_BT_601_CUB` | 2116026 |
| `ITUR_BT_601_CUG` | -409993 |
| `ITUR_BT_601_CVG` | -852492 |
| `ITUR_BT_601_CVR` | 1673527 |

픽셀 쌍마다, `uIdx = 0`인 `YUV422toRGB888Invoker`(`uidx = 1 - yIdx`, `vidx = (2 + uidx) % 4`),
`uvToRGBuv`, `yRGBuvToRGBA`를 따른다: `ruv = 2^19 + CVR·(v-128)`, `guv = 2^19 + CVG·(v-128) +
CUG·(u-128)`, `buv = 2^19 + CUB·(u-128)`, `y = max(0, Y-16)·CY`이며, 각 채널은
`clamp((y + ·uv) >> 20, 0, 255)`이다. `yIdx`는 `uyvy`/`yuv422`에서 1, `yuyv`/`yuv422_yuy2`에서
0이다. golden이 판정하며, 바이트 단위로 정확히 일치한다(`uyvy_and_yuyv_match_opencv_bit_exact`).

**cv_bridge 철자(W1c, 실측).** RoboStack Kilted의 `ros-kilted-cv-bridge 4.1.0`은 최신 이름을
거부한다: `encoding_to_cvtype2("uyvy")`는 `Unrecognized image encoding [uyvy]`를 일으키고,
`"yuyv"`도 마찬가지다; deprecated된 `yuv422` / `yuv422_yuy2`만 해석된다(`CV_8UC2`로). 그래서
`--ros` cross-check은 같은 레이아웃의 deprecated 철자를 cv_bridge에 넘기며, ingest는 네 이름
모두를 받아들인다(위 표).

### 6.4 Identity

- `Image.header.frame_id`와 `CameraInfo.header.frame_id`는 설정된 `frame_id`와 같아야
  한다(`CAM-003`). ROS optical frame과 `ImageSpec::extrinsics`는 `+Z` forward, `+X` right, `+Y`
  down(§3.1) 컨벤션을 공유한다: 축 변경 없음.
- 유도된 spec은 선언된 것과 일치해야 한다(§26.1: "검증되지 않은 것은 실행되지 않는다"): enum
  필드와 size가 같고, intrinsics는 `intrinsics_consistent_with` 기준, distortion 계수는 상대
  오차 1e-9 이내. `CAM-008`은 처음으로 다른 필드의 이름을 알려준다.
- `CameraInfo`가 아직 하나도 도착하지 않은 것은 `CAM-008`이 아니라 `CAM-010`이다: 호출자가
  재시도로 넘길 수 있는 기동 시점의 경합은, 라인을 멈춰 세우는 "다른 스트림에 맞춰 보정된
  카메라"와 같은 조건이 아니다. 따라서 `CameraError::code`는 `CAM-001` .. `CAM-010`이다.
- 가장 최근의 `CameraInfo`가 우선한다; 내용 변경(header 제외)이 있으면 다시 유도하고 다시
  검사한다.

### 6.5 시간 (§18.1, §3.4)

`StampClock { epoch, rate: TickRate }`: `tick = floor((stamp_ns − epoch_ns) × num / (den × 10^9))`를
`i128`로 계산하며 float은 없다. `epoch`보다 이전인 stamp나 `nanosec ≥ 10^9`는 -> `CAM-009`.
호출자가 epoch를 고른다(보통 첫 stamp)이고 그것을 기록한다. `obs_age(now, stamp) -> Micros`는
saturating 정수 차이다. 받아들여진 프레임마다 `sensor_seen(cfg.sensor, tick)`가 발생한다; 그
age가 `validate`에 넘겨지는 `obs_age`다 -- 이것이 멈춰버린 카메라가 새 watchdog 없이
`StaleObservation` / `SensorDropout`(§9.4)을 유발하는 방법이다.

## 7. HIL (W1d)

### 7.1 토폴로지

```
controller under test --UDP Command/Heartbeat--> HilLink --> HilCore<NJ,H> --SafeAction--> plant (sim)
                      <--UDP State(q, qd)------                  |
                                                                 +--> .eshil log, HilStats -> es-telemetry
```

`HilCore`는 외부 컨트롤러와 시뮬레이션된 actuator 사이에서
`es_runtime_embedded::core_rt::EmbeddedCore::step`을 통해 Safety Plane을 실행한다. 이는 로봇이
실행하는 것과 같은 루프다(§9.5). chunk 실행 모드는 Deployment IR의 `execution`이며, 컨트롤러가
선택하지 않는다. `HilCore`는 디코딩된 이벤트를 소비하며 전송 방식에 무관하다; `HilLink`가 UDP
프론트엔드다. ROS 2 프론트엔드(§24.2 "ROS 2 or low-latency UDP")는 `HilCore`를 재사용할 것이다;
W1에는 없다.

### 7.2 와이어 프로토콜 (little-endian)

```
offset  size  field
0       4     magic "ESH1"
4       1     version = 1
5       1     kind: 1 Hello, 2 HelloAck, 3 State, 4 Command, 5 Heartbeat, 6 Bye
6       2     reserved = 0
8       8     session_id (0 in Hello)
16      8     seq: per sender and session, from 1, strictly increasing
24      8     sender_mono_ns: sender's monotonic clock (statistics only)
32      4     body_len
36      n     body
36+n    16    tag = blake3::keyed_hash(key, bytes[0 .. 36+n])[..16]
```

| kind | body |
|---|---|
| Hello | `u32 nj`, `u32 h`, `[u8; 32] deployment_hash` (`DeploymentIr::deployment_hash`) |
| HelloAck | `u64 rate_num`, `u64 rate_den`, `u64 start_tick` |
| State | `u64 tick`, `[f64; NJ] q`, `[f64; NJ] qd` |
| Command | `u64 obs_tick`(응답 대상 `State.tick`), `u16 rows`(≤ H), `u16 0`, `rows × NJ`개의 f64 |
| Heartbeat | 비어 있음 |
| Bye | `u16 reason`: 1 해시 불일치, 2 shape 불일치, 3 버전, 4 shutdown |

- datagram당 메시지 하나, ≤ 65,507바이트(1,500 MTU 링크에서 1,472바이트를 넘으면 조각남;
  문서화되어 있을 뿐 금지되지는 않음).
- `session_id`는 accept된 모든 `Hello`마다 새로 생성된다; `0`은 `Hello` 자신에서만 쓰인다.
  `Hello`를 accept하면 `last_seq`가 되감기므로, 기록된 `Hello` + `Command` 쌍의 재생을 막는 것은
  이 신선함이다: 새 id 아래에서는 기록된 모든 datagram이 session 검사에서 탈락한다(§25.1).
- `HilCore` 이전에 버려지고 카운트됨: 잘못된 magic, version, length, tag, 잘못된
  session(`rx_invalid`); `seq ≤` 마지막으로 accept된 값(`rx_stale`). `seq`에 gap이 있으면
  `rx_lost`에 더해진다.
- f64는 `to_bits`로 전달된다: NaN은 같은 NaN으로 도착하여 plane의 non-finite 규칙을 만난다.

### 7.3 Tick binning: 비결정성이 멈추는 지점

§24.2는 컨트롤러가 외부에 있다는 이유로 HIL에 tier-1 결정성을 부여하지 않는다. 비결정성은
datagram이 *언제* 도착하는가에 있다; `HilCore::tick(now, q, qd)`는 틱마다 한 번 그것을
확정한다:

1. `HilLink`는 소켓을 (non-blocking으로) 비우고, 지난 틱 이후 accept한 것을 도착 순서대로
   넘긴다.
2. heartbeat가 하나라도 있으면 -> `plane.heartbeat(now)`를 한 번 호출한다.
3. Command: 가장 높은 `seq`가 이긴다(나머지는 `superseded`로 카운트). 만약
   `now > obs_tick + ceil(inference_budget / control_period)`이면 **deadline miss**로
   카운트되지만 그래도 제출된다; 수용 여부는 link가 아니라 plane의 `InferenceDeadline`
   watchdog이 판단한다.
4. `obs_age = (now − 마지막으로 제출된 command의 obs_tick) × control_period_us`, saturating;
   첫 command 이전에는 0.
5. `EmbeddedCore::step(chunk_or_none, now, obs_age) -> SafeAction`.
6. 2–5단계의 입력과 결정이 로그에 추가된다; 그런 다음 `State { now, q, qd }`가 전송된다.

plane의 모든 입력은 정수와 비트 패턴으로 로그에 남는다. `Instant`/`SystemTime`은 통계에만
사용된다.

### 7.4 입력 로그 `.eshil` v1

```
header   "ESHIL\0\0\x01" | u32 nj | u32 h | u64 rate_num | u64 rate_den
         | [u8;32] deployment_hash | u32 ir_len | serde_json bytes of the DeploymentIr
records  u8 tag | u32 len | body
  0x01 ObserveState { u64 tick, [f64;NJ] q, [f64;NJ] qd }
  0x02 Heartbeat    { u64 tick }
  0x03 Step         { u64 tick, u64 obs_age_us, u8 has_chunk, [u16 rows, rows×NJ f64] }
  0x10 Decision     { u64 tick, [u64;NJ] q_bits, u8 source (0 policy, 1 clamped, 2 fallback),
                      u8 fallback (FallbackKind declaration index, 0xFF if none), u32 events }
  0x20 Stats        { non-normative, ignored by replay }
trailer  0xFF | u32 40 | [u8;32] blake3(concatenated Decision record bytes) | u64 steps
```

각 `Decision`은 그 `Step`을 뒤따른다. Replay는 임베딩된 IR로부터 `deployment_hash()`를 다시
유도하고 그것이 header와 같아야 한다(JSON 포맷팅은 무관하다; canonical 해시가 결정한다). trailer가
없는 로그(crash)는 마지막 완전한 레코드까지 replay하며 `truncated`를 보고한다.

### 7.5 Replay와 M3 게이트

`hil::replay::<NJ, H>(log) -> Result<ReplayReport, HilLogError>`: `SafetyPlane::from_ir`로 plane을
재구성하고, `EmbeddedCore::with_plane`으로 감싸고, `ObserveState` / `Heartbeat` / `Step`을
순서대로 적용하고, 각 `SafeAction`을 `Decision` 레코드로 다시 인코딩하여 바이트를 비교한다.
`ReplayReport { steps, identical, first_divergence: Option<(u64 index, PhysTick)>, live_hash,
replay_hash, truncated }`.

**게이트(§24.2, §28.5):** 실제 loopback UDP를 통한 live 실행이 delay, loss, 재정렬, 지연된
command, NaN row, heartbeat gap을 주입받은 채로 **바이트 단위로 동일한 결정**(`identical`,
`live_hash == replay_hash`)으로 replay되고, 그 실행은 공허하지 않다(non-vacuous): 최소 하나의
`Clamped`, 하나의 `Fallback`, 하나의 `NonFinite` 이벤트, 하나의 `HeartbeatLoss`, 하나의 deadline
miss가 있다.

이는 plane의 HIL 결정이 로그된 입력의 결정론적 함수이며, 같은 `EmbeddedCore` 코드로 오프라인
재현된다는 것을 증명한다. 물리 컨트롤러, 네트워크, 로봇이 그 입력들을 실제로 만들어낸다는 것,
또는 `thumbv7em` 빌드가 마이크로컨트롤러 위에서 동일하게 판단한다는 것(같은 소스, 다른
바이너리)은 증명하지 않는다: 게이트 14, `Target / Status: unverified`.

### 7.6 텔레메트리 (§24.2)

`HilStats`: `u64` 타입의 `ticks`, `commands`, `deadline_miss`, `rx_lost`, `rx_stale`,
`rx_invalid`, `superseded`, `heartbeats`; `i64` ns 타입의 `rtt_last`, `jitter`(State→Command
round trip에 대한 RFC 3550 estimator `J += (|D| − J) / 16`). `stats_every` 틱마다:
`Frame { tick, wall_ns, stream: HIL_STATS_STREAM, payload: Payload::Scalars(..) }`,
`HIL_STATS_STREAM = StreamId(0x4849_4C31)`, 필드 순서는 `hil::stats`에 나열되고 고정된 대로다.
`es_telemetry::transport::Server::publish`로 publish되며, 이는 루프를 절대 막지 않는다. CI
loopback 실행에서 나온 수치는 성능 주장이 아니라 관측값이다(§12.4).

### 7.7 보안 (§25.1)

기본 bind는 `127.0.0.1`이다; 그 외에는 명시적 설정이 필요하다. 모든 datagram은 16바이트의
keyed-blake3 tag를 갖는다; 32바이트 key는 저장소 밖의 파일에서 읽는다(`*.eshilkey`,
gitignore됨). 이 tag는 인증만 하며 암호화하지 않는다: HIL 상태는 비밀이 아니지만, 위조된
command가 plane에 도달해서는 안 된다. 테스트는 고정된 테스트 key를 쓴다.

## 8. 오라클과 CI 등급 (§1.4, §26.2)

| 오라클 | 산출 / 검사 | 필요 | 실행 위치 | 없으면 -> |
|---|---|---|---|---|
| rosbags 0.11.5 | CDR golden `tests/golden/ros2/cdr/**`, `rihs01.json` | `rosbags`가 있는 `ES_PYTHON` | 커밋된 golden(PR); provenance 재실행(오라클 job) | `SKIP` + 사유 |
| PyPI `xxhash` 4.0.1 | `tests/golden/ros2/gid.json` | `ES_PYTHON` | 동일 | `SKIP` |
| RoboStack JSON, `rclpy.serialization.serialize_message` | 해시 cross-check; 실제 RMW CDR 바이트 | `ES_ROS2_ENV` | 오라클 job / Linux 서버 | `SKIP` |
| rmw_zenoh design.md 예제 | token과 key 리터럴 | — | PR | — |
| 라이브 `rmw_zenohd`, `ros2` CLI, `demo_nodes_cpp` | 양방향 interop; `tests/golden/ros2/rmw_zenoh/**` capture | `ES_ROS2_ENV` | 오라클 job / Linux 서버 | `SKIP`; 실행 시 `RAN rmw_zenoh_interop` |
| image_geometry, cv_bridge (RoboStack) | ROI/binning 아래의 intrinsics; encoding 변환 | `ES_ROS2_ENV` | golden 생성, provenance | `SKIP` |
| opencv-python-headless 5.0.0.93 | YUV golden | `cv2`가 있는 `ES_PYTHON` | provenance | `SKIP` |
| HIL live == replay | 게이트 | — | PR | — |

`ES_ROS2_ENV`는 RoboStack 환경의 prefix다. 레퍼런스 명령은
`crates/es-ros2/scripts/ros2-env.sh <prefix> <cmd...>`를 통해 실행된다: `CONDA_PREFIX=<prefix>`를
export하고, `<prefix>/bin`을 `PATH` 앞에 붙이고, `<prefix>/etc/conda/activate.d/*.sh`를
source하고, 명령을 `exec`한다. 테스트 시점에 micromamba/pixi는 없다. 그 activation이 `ros2`에
충분한지는 W1b가 실행하기 전까지 **미검증**이다; `ES_ROS2_ENV`가 설정되어 있으면 어떤 실패든
테스트 실패이지 SKIP이 아니다. SKIP 사유는 `gpu`, `vulkan`, `render`, `slangc`, `device`를
피해야 한다(`xtask`가 이들을 GPU skip으로 분류하기 때문이다).

Linux x86_64, docker 없음, sudo 없음(RoboStack은 pixi를 권장한다; 여기서는 micromamba를
보이며, 채널은 동일하다):

```
micromamba create -y -p "$HOME/envs/ros2-kilted" -c conda-forge -c robostack-kilted \
  ros-kilted-ros-base=0.12.0 ros-kilted-rmw-zenoh-cpp=0.6.6 ros-kilted-demo-nodes-cpp=0.36.4 \
  ros-kilted-cv-bridge=4.1.0 ros-kilted-image-geometry=4.1.0
export ES_ROS2_ENV="$HOME/envs/ros2-kilted"
python3 -m venv "$HOME/envs/es-oracles"
"$HOME/envs/es-oracles/bin/pip" install rosbags==0.11.5 xxhash==4.0.1 opencv-python-headless==5.0.0.93
export ES_PYTHON="$HOME/envs/es-oracles/bin/python"
```

다섯 개의 RoboStack 패키지 모두 `robostack-kilted`의 `linux-64`에 존재한다(prefix.dev
repodata, 2026-09-14); kilted는 `python_abi 3.12`에 pin되어 있다. `cv-bridge`/`image-geometry`는
`np126` 빌드(`_10`)인 반면 나머지는 `np2` 빌드(`_21`)다: 하나의 prefix에 공존 설치할 수
있는지는 **미검증**이다. solver가 거부하면 W1c는 두 번째 prefix인 `ES_ROS2_VISION_ENV`를 쓰고
그것을 기록한다.

## 9. 패킷 순서

W1a -> W1b -> W1c -> W1d, 순차적: 넷 모두 `crates/es-ros2/src/lib.rs`, `crates/es-ros2/Cargo.toml`,
`Cargo.lock`을 편집한다. W1c와 W1d는 W1b의 세션이 필요 없으며 그보다 먼저 실행될 수 있다. 아직
작성되지 않음: W1e(`es hil replay <log>`, `es --check-deps` 안의 ROS 2 / `ES_ROS2_ENV` 항목)와
W1f(모드 B 라이브 오라클). §28.5는 W1에 패킷 9개를 배정한다; 나머지는 섹션 1의 하드웨어에
묶인 행들이다.

## 10. 사람에게 남기는 미결 질문

1. `es-hil`을 독자적인 layer-11 crate로 만들 것인가(§4.2와 `LAYERS` 개정), 아니면 설계된 대로
   `es_ros2::hil`로 둘 것인가?
2. W1b는 PR job이 zenoh를 컴파일하게 만든다(tcp만, ~270개 crate). cold job이 §26.2의 10분을
   넘으면 loopback 세션 테스트를 오라클 job으로 옮길 것인가?
3. `k4..k6`이 0이 아닌 `rational_polynomial`: 계속 거부할 것인가, 아니면 `es-ir-types`에
   `DistortionModel::RationalPolynomial`을 추가할 것인가(§25.3 아래의 스키마 변경)?
4. 게이트 14: M3를 `Target / Status: unverified`로 기록한 채 닫을 것인가, 아니면 로봇 셀을
   위해 보류할 것인가(이미 `docs/reviews/M3.md`에서 제기됨)?
5. 모드 C: v1.0에서 잘라낼 것인가, 아니면 미룰 것인가?
6. 로컬 key 파일 하나를 넘어서는 HIL key(rig별 key, rotation)?
