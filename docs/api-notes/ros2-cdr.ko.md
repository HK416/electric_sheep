<!-- Korean translation of docs/api-notes/ros2-cdr.md. The English file is the working copy; regenerate this when it changes. -->

# ROS 2 CDR payload와 RIHS01 타입 해시 -- `es_ros2::cdr` / `es_ros2::msg`를 위한 고정된 표면

`crates/es-ros2`가 wire 위에서 인코딩·디코딩하는 것, 그리고 key expression에 어떤 타입 해시
문자열을 넣는지 정리한 문서다(`docs/api-notes/rmw-zenoh.md`). `rosidl`도, Fast-CDR 바인딩도
없다: 고정된 메시지 서브셋 위에 손으로 짠 little-endian CDR writer/reader가 있을 뿐이다
(`docs/design/ros2-boundary.md` 섹션 5).

Spec: §24.1(모드 A는 `zenoh-rs`로 CDR을 구현한다), §1.4(golden은 reference에서 나온다).

## 버전

| | |
|---|---|
| CDR reference | **rosbags 0.11.5**(PyPI, `py3-none-any`, Apache-2.0, `Requires-Python >=3.10`; 바이너리 의존성 apsw, lz4, numpy, ruamel.yaml, zstandard), `get_typestore(Stores.ROS2_KILTED)`, 2026-09-14 조회 |
| Type-hash reference | RoboStack `robostack-kilted`: `ros-kilted-std-msgs-5.5.2`, `ros-kilted-sensor-msgs-5.5.2`, `ros-kilted-trajectory-msgs-5.5.2`(build `np2py312hf80f32c_21`), `ros-kilted-builtin-interfaces-2.3.1`; `share/<pkg>/msg/<T>.json` -> `type_hashes[0].hash_string`. rosbags의 `Typestore.hash_rihs01`을 상대로 cross-check: **11개 모두 일치**; rosbags는 `ROS2_JAZZY`, `ROS2_KILTED`, `ROS2_LYRICAL`에 대해 동일한 해시를 낸다 |
| Encapsulation | OMG DDS-XTypes 1.3, clause 7.6.3.1.2, Table 60, <https://www.omg.org/spec/DDS-XTypes/1.3/PDF> |
| Message definitions | <https://github.com/ros2/common_interfaces>(rolling), <https://github.com/ros2/rcl_interfaces>(`builtin_interfaces`) |
| Live ROS 2 byte capture | W1a는 오라클 서버(RoboStack ROS 2 Kilted)에서 `robostack_hashes_and_rclpy_bytes_agree_with_the_goldens`를 실행했다. `scripts/ros2-env.sh`(작업 패킷에 따라 POSIX `sh`)는 이 prefix를 무인으로 activate할 수 없다: `ros-kilted-ros-workspace_activate.sh`가 bash의 `source` builtin을 호출하는데, 오라클 서버의 `/bin/sh`(dash)는 이를 제공하지 않는다(`source: not found`) — 테스트를 약화시킨 것이 아니라 실제로 기록된 환경상의 공백이다. 대신 손으로 실행하여(`RIHS01` **PASS**, 11개 `share/<pkg>/msg/<T>.json` 해시 모두 `rihs01.json`과 일치; CDR 바이트 **FAIL**, 아래 "Encapsulation header" 참고) 이 행과 그 섹션이 기록하는 근거를 얻었다. **W1b, 2026-09-14:** `scripts/ros2-env.sh`의 shebang은 이제 `#!/usr/bin/env bash`이며(자신의 `source` 호출이 이제 해석된다), `crates/es-ros2/tests/rmw_zenoh_interop.rs`는 shebang/실행 비트가 checkout에서 살아남는 것에 의존하지 않고 이를 명시적으로 `bash scripts/ros2-env.sh <prefix> <cmd...>`로 호출한다 — 이것이 아래의 라이브 오라클이 실제로 거쳐가는 경로다. **W1b 후속, 2026-09-14:** `crates/es-ros2/tests/gen_goldens.rs`도 이제 `bash`를 통해 이를 호출하며, 그 CDR 검사는 더 이상 `serialize_message`와의 바이트 동일성을 요구하지 않는다(padding 내용은 정의되지 않았고 그 버퍼는 wire에는 없는 8바이트의 trailing byte를 갖는다): 대신 `rmw_zenoh_cpp` 아래의 `rclpy.serialization.deserialize_message(<golden>)`가 fixture 메시지와 필드 단위로 같기를 요구하며(`rosidl_runtime_py.convert.message_to_ordereddict`), 바이트 차이는 정보로만 출력한다. 오라클 서버 실행 결과: RIHS01 **PASS**, 6개 벡터 모두 CDR **PASS**(`string_*`는 바이트 단위로 동일; `joint_state`는 76 대 84, 나머지 셋은 padding만 다름); stamp 비트 하나를 뒤집은 golden은 실패한다. **실제 네트워크 바이트, 2026-09-14 캡처**(`rmw_zenoh_interop.rs`의 `ros2_topic_pub_joint_state_reaches_our_subscriber`와 `capture_reference_goldens`, 둘 다 RoboStack ROS 2 Kilted, `ros-kilted-rmw-zenoh-cpp 0.6.6`을 상대로 라이브 실행): `velocity`/`effort`가 빈 `ros2 topic pub` `JointState`는 **기존의, 수정되지 않은** `CdrReader`로 바이트 단위로 정확히 디코딩된다 — 아래 "Layout rules"의 "Empty sequence alignment: resolved by live capture" 참고. |

## Encapsulation header (모든 payload에 앞서는 4바이트)

- 바이트 0-1: representation id, wire 상에서는 big-endian. `CDR_BE = 00 00`,
  **`CDR_LE = 00 01`**(XTypes Table 60; Fast DDS `#define CDR_LE 0x0001`). XCDR2 id(`00 07`,
  `00 09`, `00 0b`)는 ROS 2의 plain 메시지에서 사용되지 않는다.
- ROS 2는 XCDR1 `PLAIN_CDR`을 직렬화한다: rmw_fastrtps `TypeSupport_impl.cpp`:
  `eprosima::fastcdr::Cdr ser(..., DEFAULT_ENDIAN, CdrVersion::XCDRv1); ser.set_encoding_flag(PLAIN_CDR);`.
  rmw_zenoh의 `type_support.cpp`는 `ser.serialize_encapsulation()`을 호출한다. W1a의 근거
  (`RMW_IMPLEMENTATION=rmw_zenoh_cpp` 아래의 `rclpy.serialization.serialize_message`, RoboStack
  Kilted, 오라클 서버): `String`, `Image`, `CameraInfo`에 대해, 채워진 모든 필드와 모든
  메시지의 전체 길이가 이 표의 plain XCDR1 레이아웃과 정확히 일치하며, XCDR2
  appendable/`DHEADER` 레이아웃이 아니라 XCDR1과 일관된다. 그 근거에서 벗어나 보이는 두
  가지는 모두 W1b의 *라이브 네트워크* 캡처(`rmw_zenoh_interop.rs`, 2026-09-14, RoboStack
  Kilted `ros-kilted-rmw-zenoh-cpp 0.6.6`)로 이제 해소되었다: (1) 정렬 padding 바이트는
  0이 **아니다** — Fast-CDR는 버퍼에 이미 있던 것을 그대로 남겨둔다(아래의 라이브
  `JointState` 캡처에서 무관한 문자열의 ASCII 조각, 예를 들어 `5f 72 6f`가 관측됨), 그래서
  디코더는 padding 내용을 절대 읽어서는 안 되고 건너뛰기만 해야 한다(이 crate의
  `CdrReader`는 이미 그렇게 하며, 변경 없음); (2) `JointState`의 두 *빈* `float64[]`
  필드(`velocity`, `effort`)는 wire 상에서 정렬 pad를 소비하지 **않는다** — W1a에서 발견된
  84 대 76 바이트 차이는 "네트워크 이전에 멈추는"(이 파일의 이전 표현) `rclpy.serialization
  .serialize_message`에 특유한 것이었다; 실제
  `ros2 topic pub ... JointState "{name: [j1, j2], position: [0.5, -1.0]}"`를
  `capture_reference_goldens`(`tests/golden/ros2/rmw_zenoh/talker_capture.json`, raw
  zenoh-rs만 사용)로 캡처하고 `ros2_topic_pub_joint_state_reaches_our_subscriber`로
  독립적으로 end-to-end 디코딩한 결과(두 테스트 모두 통과), 정확히 68바이트이며 `effort`의
  4바이트 zero count 다음에 trailing byte가 **0개**다 — 즉 rosbags의 "`count > 0`일 때만
  정렬" 규칙과 바이트 단위로 정확히 일치한다. **`CdrReader`에는 어떤 변경도 필요하지
  않았고 이루어지지도 않았다.**
- 바이트 2-3: options. XTypes: 두 번째 바이트의 하위 2비트는 다음 4바이트 경계까지의 trailing
  padding 개수를 담아야 한다(*shall*). rmw_fastrtps는 절대 그것을 설정하지 않는다; rosbags는
  `00 00`을 쓰고 trailing padding이 없다.
- **인코더 규칙:** `00 01 00 00`을 쓰고, trailing padding은 없음(rosbags와 일치).
  **디코더 규칙:** `00 00`(BE)과 `00 01`(LE)을 받아들이고, options 바이트는 무시하며, 마지막
  필드 다음 최대 3바이트까지 받아들인다(rosbags 리더: `assert pos + 4 + 3 >= len(rawdata)`).
  끝을 넘어서는 그 외의 것은 에러다.

## 레이아웃 규칙 (XCDR1 plain)

- **정렬 원점은 4바이트 header 다음의 첫 바이트다.** Fast-CDR의 `serialize_encapsulation()`은
  `reset_alignment()`(`origin_ = offset_`)로 끝난다; rosbags는 `rawdata[4:]`를 offset 0부터
  직렬화한다. 아래의 JointState 벡터에서 확인됨(pad 44->48).
- 크기 `n`(1, 2, 4, 8)의 primitive는 `n`에 정렬된다. `bool`, `uint8`, `char` = 1바이트,
  정렬 없음.
- `string`: **NUL을 포함한** `uint32` length, 그다음 바이트, 그다음 `0x00`. 빈 문자열 =
  `01 00 00 00 00`. (Fast-CDR는 `strlen + 1`; rosbags는 `len(bval) + 1`.)
- `T[]` / `sequence<T>`: `uint32` count, 그다음 원소. rosbags는 **count > 0일 때만** 원소
  크기에 정렬한다(빈 `float64[]`는 8바이트 pad를 추가하지 않는다) -- JointState 벡터(`velocity`,
  `effort`가 빔)로 확인됨, **그리고** 같은 두 필드가 빈 실제 `ros2 topic pub` `JointState`에
  대한 W1b의 라이브 `rmw_zenoh` 캡처로도 확인됨(위 "Encapsulation header" 참고): 이 규칙은 이
  파일의 정적 golden 벡터뿐 아니라 실제 네트워크 바이트에서도 성립한다.
- `T[N]` 고정 배열: count 없음, 원소들은 연속으로 정렬된다.
- 중첩된 message: struct 레벨 padding 없음; 첫 필드처럼 정렬된다.
- `u8[]` data(`Image.data`): count 다음 raw byte, 정렬 없음.

## 메시지 서브셋 (필드 순서, `.msg` 파일 그대로)

| Type | Fields |
|---|---|
| `builtin_interfaces/msg/Time` | `int32 sec`, `uint32 nanosec` |
| `std_msgs/msg/Header` | `builtin_interfaces/Time stamp`, `string frame_id` |
| `std_msgs/msg/String` | `string data` |
| `std_msgs/msg/MultiArrayDimension` | `string label`, `uint32 size`, `uint32 stride` |
| `std_msgs/msg/MultiArrayLayout` | `MultiArrayDimension[] dim`, `uint32 data_offset` |
| `std_msgs/msg/Float64MultiArray` | `MultiArrayLayout layout`, `float64[] data` |
| `sensor_msgs/msg/JointState` | `std_msgs/Header header`, `string[] name`, `float64[] position`, `float64[] velocity`, `float64[] effort` |
| `sensor_msgs/msg/Image` | `std_msgs/Header header`, `uint32 height`, `uint32 width`, `string encoding`, `uint8 is_bigendian`, `uint32 step`, `uint8[] data` |
| `sensor_msgs/msg/RegionOfInterest` | `uint32 x_offset`, `uint32 y_offset`, `uint32 height`, `uint32 width`, `bool do_rectify` |
| `sensor_msgs/msg/CameraInfo` | `std_msgs/Header header`, `uint32 height`, `uint32 width`, `string distortion_model`, `float64[] d`, `float64[9] k`, `float64[9] r`, `float64[12] p`, `uint32 binning_x`, `uint32 binning_y`, `RegionOfInterest roi` |

`sensor_msgs/include/sensor_msgs/image_encodings.hpp`(rolling)에서, camera ingest가 인식하는
문자열: `"rgb8"`, `"rgba8"`, `"bgr8"`, `"bgra8"`, `"mono8"`, `"mono16"`, `"16UC1"`, `"32FC1"`,
`"uyvy"`, `"yuv422"`(주석 `// deprecated`, `uyvy`와 같은 레이아웃), `"yuyv"`,
`"yuv422_yuy2"`(주석 `// deprecated`, `yuyv`와 같은 레이아웃). 그 외에도 존재하지만(bayer_*,
nv12, nv21, nv24, `<n>{U,S,F}C<k>`) ingest는 이를 거부한다.

`distortion_models.hpp`: `PLUMB_BOB = "plumb_bob"`, `RATIONAL_POLYNOMIAL = "rational_polynomial"`,
`EQUIDISTANT = "equidistant"`.

ingest가 의존하는 `CameraInfo.msg`의 주석(원문 그대로):
- height/width: "The image dimensions with which the camera was calibrated."
- `d`(plumb_bob의 경우): "the 5 parameters are: (k1, k2, t1, t2, k3)."
- `k`: "Intrinsic camera matrix for the raw (distorted) images. [fx 0 cx; 0 fy cy; 0 0 1]"(row-major).
- `p`: "the intrinsic (camera) matrix of the processed (rectified) image ... [fx' 0 cx' Tx; 0 fy' cy' Ty; 0 0 1 0]".
- binning: "binning_x = binning_y = 0 is considered the same as binning_x = binning_y = 1".
- roi: "(all values 0) is considered the same as full resolution"; "A particular ROI always
  denotes the same window of pixels on the camera sensor, regardless of binning settings."

## RIHS01 타입 해시

- **REP-2016** "ROS 2 Interface Type Description"에 정의됨
  (<https://github.com/ros-infrastructure/rep/pull/381>, 아직 열린 PR) -- REP-2011이 아니다,
  rosidl의 코드 주석이 "per REP-2011"이라고 말함에도 불구하고. "it omits field default values
  ... all other non-programmatic contents such as comments"; SHA-256; `RIHS01_` + 소문자 hex
  64자 = 71자.
- 두 구현이 "together form its specification by their agreement"다:
  `rosidl_generator_type_description`(`json.dumps(..., separators=(', ', ': '), sort_keys=False)`,
  `del field['default_value']`)와 `rcl/src/rcl/type_hash.c`.
- **`es-ros2`는 RIHS01을 계산하지 않는다.** 고정된 서브셋에 대한 const 테이블을 가지고
  있으며, W1a golden 테스트로 reference를 상대로 검사된다. 임의 타입에 대한 해시 계산은
  범위 밖이다(`docs/design/ros2-boundary.md` 섹션 3).

| Type | RIHS01 |
|---|---|
| `std_msgs/msg/String` | `RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18` |
| `std_msgs/msg/Header` | `RIHS01_f49fb3ae2cf070f793645ff749683ac6b06203e41c891e17701b1cb597ce6a01` |
| `builtin_interfaces/msg/Time` | `RIHS01_b106235e25a4c5ed35098aa0a61a3ee9c9b18d197f398b0e4206cea9acf9c197` |
| `std_msgs/msg/Float64MultiArray` | `RIHS01_1025ddc6b9552d191f89ef1a8d2f60f3d373e28b283d8891ddcc974e8c55397f` |
| `std_msgs/msg/MultiArrayLayout` | `RIHS01_4c66e6f78e740ac103a94cf63259f968e48c617e7699e829b63c21a5cb50dac6` |
| `std_msgs/msg/MultiArrayDimension` | `RIHS01_5e773a60a4c7fc8a54985f307c7837aa2994252a126c301957a24e31282c9cbe` |
| `sensor_msgs/msg/JointState` | `RIHS01_a13ee3a330e346c9d87b5aa18d24e11690752bd33a0350f11c5882bc9179260e` |
| `sensor_msgs/msg/Image` | `RIHS01_d31d41a9a4c4bc8eae9be757b0beed306564f7526c88ea6a4588fb9582527d47` |
| `sensor_msgs/msg/CameraInfo` | `RIHS01_b3dfd68ff46c9d56c80fd3bd4ed22c7a4ddce8c8348f2f59c299e73118e7e275` |
| `sensor_msgs/msg/RegionOfInterest` | `RIHS01_ad16bcba5f9131dcdba6fbded19f726f5440e3c513b4fb586dd3027eeed8abb1` |
| `trajectory_msgs/msg/JointTrajectory` | `RIHS01_179b33eba59d676f6d967ac71fe35e7ca2f64b2f3928f4a018cec115e213796e`(W1 서브셋에는 없음; 나중의 `JointTrajectory` 명령 경로를 위해 기록됨) |

이 String 해시는 rmw_zenoh의 `docs/design.md`(rolling) 144번째 줄에도 그대로 등장한다.

## Golden 벡터 (rosbags 0.11.5, `ROS2_KILTED`, 출력은 4바이트 header를 포함)

API: `Typestore.serialize_cdr(message, typename, *, little_endian=sys.byteorder == 'little') -> memoryview`,
`Typestore.deserialize_cdr(rawdata, typename) -> object`. `little_endian=True`를 명시적으로
전달한다.

| Vector | Bytes | Hex |
|---|---|---|
| String `data="hello"` | 14 | `000100000600000068656c6c6f00` |
| String `data=""` | 9 | `000100000100000000` |
| String `"hello"`, `little_endian=False` | 14 | `000000000000000668656c6c6f00` |
| JointState: stamp 1/2, frame `"base"`, name `["j1","j2"]`, position `[0.5,-1.0]`, velocity `[]`, effort `[]` | 76 | `00010000010000000200000005000000626173650000000002000000030000006a310000030000006a3200000200000000000000000000000000e03f000000000000f0bf0000000000000000` |
| Image: stamp 1/2, frame `"cam"`, 1x2 `rgb8`, `is_bigendian=0`, `step=6`, data `01..06` | 54 | `0001000001000000020000000400000063616d0001000000020000000500000072676238000000000600000006000000010203040506` |
| CameraInfo: stamp 1/2, frame `"cam"`, 480x640 `plumb_bob`, d `[0.1,0.01,0,0,0]`, k `[500,0,320,0,500,240,0,0,1]`, r identity, p `[500,0,320,0,0,500,240,0,0,0,1,0]`, binning 0, roi zero | 357 | 처음 64바이트: `0001000001000000020000000400000063616d00e0010000800200000a000000706c756d625f626f6200000005000000000000009a9999999999b93f7b14ae47`(payload는 payload offset 352에서 `do_rectify`로 끝남) |
| Float64MultiArray: `dim=[]`, `data_offset=0`, data `[0.25,-0.5,1.0]` | 44 | `0001000000000000000000000300000000000000000000000000d03f000000000000e0bf000000000000f03f` |

이것들은 W1a의 `cdr` 테스트가 출발점으로 삼는 리터럴 기댓값이다; `tests/golden/ros2/cdr/`
아래의 커밋된 파일들은 `crates/es-ros2/python/gen_ros2_goldens.py`로 재생성되며 이 표와
일치해야 한다.

재현 방법(Linux, sudo 불필요):

```
python3 -m venv /tmp/rosbags-venv && /tmp/rosbags-venv/bin/pip install rosbags==0.11.5
ES_PYTHON=/tmp/rosbags-venv/bin/python cargo test -p es-ros2 --test gen_goldens -- --nocapture
```
