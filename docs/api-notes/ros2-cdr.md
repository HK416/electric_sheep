# ROS 2 CDR payloads and RIHS01 type hashes -- pinned surface for `es_ros2::cdr` / `es_ros2::msg`

What `crates/es-ros2` encodes and decodes on the wire, and which type-hash strings it puts in
key expressions (`docs/api-notes/rmw-zenoh.md`). No `rosidl`, no Fast-CDR binding: a hand-rolled
little-endian CDR writer/reader over a fixed message subset (`docs/design/ros2-boundary.md` section 5).

Spec: §24.1 (mode A implements CDR with `zenoh-rs`), §1.4 (goldens come from the reference).

## Version

| | |
|---|---|
| CDR reference | **rosbags 0.11.5** (PyPI, `py3-none-any`, Apache-2.0, `Requires-Python >=3.10`; binary deps apsw, lz4, numpy, ruamel.yaml, zstandard), `get_typestore(Stores.ROS2_KILTED)`, retrieved 2026-09-14 |
| Type-hash reference | RoboStack `robostack-kilted`: `ros-kilted-std-msgs-5.5.2`, `ros-kilted-sensor-msgs-5.5.2`, `ros-kilted-trajectory-msgs-5.5.2` (build `np2py312hf80f32c_21`), `ros-kilted-builtin-interfaces-2.3.1`; `share/<pkg>/msg/<T>.json` -> `type_hashes[0].hash_string`. Cross-checked against rosbags `Typestore.hash_rihs01`: **all 11 agree**; rosbags gives identical hashes for `ROS2_JAZZY`, `ROS2_KILTED`, `ROS2_LYRICAL` |
| Encapsulation | OMG DDS-XTypes 1.3, clause 7.6.3.1.2, Table 60, <https://www.omg.org/spec/DDS-XTypes/1.3/PDF> |
| Message definitions | <https://github.com/ros2/common_interfaces> (rolling), <https://github.com/ros2/rcl_interfaces> (`builtin_interfaces`) |
| Live ROS 2 byte capture | **unverified** until W1b's interop oracle (`docs/packets/M3/W1b-ros2-zenoh-session.md`) records real rmw_zenoh payloads |

## Encapsulation header (4 bytes, precedes every payload)

- Bytes 0-1: representation id, big-endian on the wire. `CDR_BE = 00 00`, **`CDR_LE = 00 01`**
  (XTypes Table 60; Fast DDS `#define CDR_LE 0x0001`). XCDR2 ids (`00 07`, `00 09`, `00 0b`) are
  not used by ROS 2 plain messages.
- ROS 2 serializes XCDR1 `PLAIN_CDR`: rmw_fastrtps `TypeSupport_impl.cpp`:
  `eprosima::fastcdr::Cdr ser(..., DEFAULT_ENDIAN, CdrVersion::XCDRv1); ser.set_encoding_flag(PLAIN_CDR);`.
  rmw_zenoh's `type_support.cpp` calls `ser.serialize_encapsulation()`; its `CdrVersion` is
  **unverified** (settled by the W1b capture).
- Bytes 2-3: options. XTypes: the low 2 bits of the second byte *shall* carry the trailing
  padding count to the next 4-byte boundary. rmw_fastrtps never sets them; rosbags writes
  `00 00` and no trailing padding.
- **Encoder rule:** write `00 01 00 00`, no trailing padding (matches rosbags).
  **Decoder rule:** accept `00 00` (BE) and `00 01` (LE), ignore the options bytes, accept up to
  3 bytes past the last field (rosbags reader: `assert pos + 4 + 3 >= len(rawdata)`). Anything
  else past the end is an error.

## Layout rules (XCDR1 plain)

- **Alignment origin is the first byte after the 4-byte header.** Fast-CDR
  `serialize_encapsulation()` ends with `reset_alignment()` (`origin_ = offset_`); rosbags
  serializes `rawdata[4:]` from offset 0. Checked in the JointState vector below (pad 44->48).
- Primitive of size `n` (1, 2, 4, 8) aligns to `n`. `bool`, `uint8`, `char` = 1 byte, no align.
- `string`: `uint32` length **including the NUL**, then bytes, then `0x00`. Empty string =
  `01 00 00 00 00`. (Fast-CDR `strlen + 1`; rosbags `len(bval) + 1`.)
- `T[]` / `sequence<T>`: `uint32` count, then elements. rosbags aligns to the element size
  **only when count > 0** (an empty `float64[]` adds no 8-byte pad) -- confirmed by the
  JointState vector (`velocity`, `effort` empty).
- `T[N]` fixed array: no count, elements aligned as a run.
- Nested message: no struct-level padding; it aligns as its first field.
- `u8[]` data (`Image.data`): count then raw bytes, no alignment.

## Message subset (field order, verbatim from the `.msg` files)

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

`sensor_msgs/include/sensor_msgs/image_encodings.hpp` (rolling), the strings camera ingest
recognizes: `"rgb8"`, `"rgba8"`, `"bgr8"`, `"bgra8"`, `"mono8"`, `"mono16"`, `"16UC1"`,
`"32FC1"`, `"uyvy"`, `"yuv422"` (commented `// deprecated`, same layout as `uyvy`), `"yuyv"`,
`"yuv422_yuy2"` (commented `// deprecated`, same layout as `yuyv`). Others exist (bayer_*, nv12,
nv21, nv24, `<n>{U,S,F}C<k>`) and are rejected by ingest.

`distortion_models.hpp`: `PLUMB_BOB = "plumb_bob"`, `RATIONAL_POLYNOMIAL = "rational_polynomial"`,
`EQUIDISTANT = "equidistant"`.

`CameraInfo.msg` comments that ingest relies on (verbatim):
- height/width: "The image dimensions with which the camera was calibrated."
- `d` for plumb_bob: "the 5 parameters are: (k1, k2, t1, t2, k3)."
- `k`: "Intrinsic camera matrix for the raw (distorted) images. [fx 0 cx; 0 fy cy; 0 0 1]" (row-major).
- `p`: "the intrinsic (camera) matrix of the processed (rectified) image ... [fx' 0 cx' Tx; 0 fy' cy' Ty; 0 0 1 0]".
- binning: "binning_x = binning_y = 0 is considered the same as binning_x = binning_y = 1".
- roi: "(all values 0) is considered the same as full resolution"; "A particular ROI always
  denotes the same window of pixels on the camera sensor, regardless of binning settings."

## RIHS01 type hash

- Defined by **REP-2016** "ROS 2 Interface Type Description"
  (<https://github.com/ros-infrastructure/rep/pull/381>, open PR) -- not REP-2011, even though
  rosidl's code comment says "per REP-2011". "it omits field default values ... all other
  non-programmatic contents such as comments"; SHA-256; `RIHS01_` + 64 lowercase hex = 71 chars.
- Two implementations "together form its specification by their agreement":
  `rosidl_generator_type_description` (`json.dumps(..., separators=(', ', ': '), sort_keys=False)`,
  `del field['default_value']`) and `rcl/src/rcl/type_hash.c`.
- **`es-ros2` does not compute RIHS01.** It carries a const table for its fixed subset,
  checked against the reference by the W1a golden test. Computing hashes for arbitrary types
  is out of scope (`docs/design/ros2-boundary.md` section 3).

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
| `trajectory_msgs/msg/JointTrajectory` | `RIHS01_179b33eba59d676f6d967ac71fe35e7ca2f64b2f3928f4a018cec115e213796e` (not in the W1 subset; recorded for a later `JointTrajectory` command path) |

The String hash also appears verbatim in rmw_zenoh `docs/design.md` (rolling) line 144.

## Golden vectors (rosbags 0.11.5, `ROS2_KILTED`, output includes the 4-byte header)

API: `Typestore.serialize_cdr(message, typename, *, little_endian=sys.byteorder == 'little') -> memoryview`,
`Typestore.deserialize_cdr(rawdata, typename) -> object`. Pass `little_endian=True` explicitly.

| Vector | Bytes | Hex |
|---|---|---|
| String `data="hello"` | 14 | `000100000600000068656c6c6f00` |
| String `data=""` | 9 | `000100000100000000` |
| String `"hello"`, `little_endian=False` | 14 | `000000000000000668656c6c6f00` |
| JointState: stamp 1/2, frame `"base"`, name `["j1","j2"]`, position `[0.5,-1.0]`, velocity `[]`, effort `[]` | 76 | `00010000010000000200000005000000626173650000000002000000030000006a310000030000006a3200000200000000000000000000000000e03f000000000000f0bf0000000000000000` |
| Image: stamp 1/2, frame `"cam"`, 1x2 `rgb8`, `is_bigendian=0`, `step=6`, data `01..06` | 54 | `0001000001000000020000000400000063616d0001000000020000000500000072676238000000000600000006000000010203040506` |
| CameraInfo: stamp 1/2, frame `"cam"`, 480x640 `plumb_bob`, d `[0.1,0.01,0,0,0]`, k `[500,0,320,0,500,240,0,0,1]`, r identity, p `[500,0,320,0,0,500,240,0,0,0,1,0]`, binning 0, roi zero | 357 | first 64: `0001000001000000020000000400000063616d00e0010000800200000a000000706c756d625f626f6200000005000000000000009a9999999999b93f7b14ae47` (payload ends with `do_rectify` at payload offset 352) |
| Float64MultiArray: `dim=[]`, `data_offset=0`, data `[0.25,-0.5,1.0]` | 44 | `0001000000000000000000000300000000000000000000000000d03f000000000000e0bf000000000000f03f` |

These are the literal expectations W1a's `cdr` tests start from; the checked-in files under
`tests/golden/ros2/cdr/` are regenerated by `crates/es-ros2/python/gen_ros2_goldens.py` and
must match this table.

Reproduce (Linux, no sudo):

```
python3 -m venv /tmp/rosbags-venv && /tmp/rosbags-venv/bin/pip install rosbags==0.11.5
ES_PYTHON=/tmp/rosbags-venv/bin/python cargo test -p es-ros2 --test gen_goldens -- --nocapture
```
