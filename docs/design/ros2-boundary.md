# ROS 2 boundary, camera ingest and HIL (`es-ros2`) — design

Spec: §24.1 (ROS 2 boundary, modes A/B/C), §24.2 (HIL), §28.5 (M3 W1), §28.7 (gate 14), §7.2 and
Appendix B.2 (`ImageSpec`), §18.1 (integer time), §18.3 (OpenCV distortion convention), §9.3–§9.5
and Appendix B.4 (Safety Plane), §4.2 (layering), §25.1 (security), §26.2 (CI tiers), §1.4
(oracles), §1.9 (scope), §12.4 (unverified numbers).
Invariants: INV-12, INV-13, INV-14, INV-17.
Digests: `docs/api-notes/rmw-zenoh.md`, `docs/api-notes/ros2-cdr.md`, `docs/api-notes/zenoh-rs.md`.
Packets: `docs/packets/M3/W1a-ros2-cdr-keyexpr.md`, `W1b-ros2-zenoh-session.md`,
`W1c-camera-ingest.md`, `W1d-hil.md`.

Design-note sections are cited as "section N"; `§N` always means the spec.

## 1. Scope

| Item (§28.5 W1) | Packet | Oracle | After W1 |
|---|---|---|---|
| CDR, message subset, key expressions, liveliness, attachment, type hashes | W1a | rosbags, RoboStack JSON, rmw_zenoh design.md | verifiable anywhere |
| zenoh session, mode A, A ⊕ B exclusion, actuator path | W1b | loopback peers (PR tier); live rmw_zenoh (`ES_ROS2_ENV`) | verifiable on Linux, no sudo |
| Mode B data path | W1b (code only) | bridge key-mapping unit test | live: `Target / Status: unverified` |
| Mode C (`RustDDS` + `ros2-client`) | none | — | not built; config rejects it (§24.1 marks it experimental) |
| Camera: `sensor_msgs/Image` + `CameraInfo` -> `ImageSpec` | W1c | rosbags, image_geometry, cv_bridge, OpenCV | verifiable anywhere / Linux |
| HIL: UDP link, deadline/jitter telemetry, input log, replay gate | W1d | loopback controller; live == replay | verifiable anywhere |
| `es hil replay`, ROS 2 lines in `es --check-deps` | W1e (not written) | CLI tests | — |
| Mode B live oracle | W1f (not written) | zenoh-bridge-ros2dds + `rmw_cyclonedds_cpp` | — |
| In-tree V4L2/USB camera driver | none | needs a camera | `Target / Status: unverified` |
| Robot cell, HIL against a physical controller, gate 14 | none | hardware (§26.2 release tier) | `Target / Status: unverified` |
| Any latency, jitter or throughput figure | — | hardware | `Target / Status: unverified` |

Physical cameras enter through existing ROS 2 camera drivers publishing `sensor_msgs/Image`; W1
meets §28.5's "real hardware camera drivers" at that message boundary. An in-tree V4L2 driver is
deferred: nothing can judge it without a camera (§1.4).

## 2. Placement, dependencies, features

**Crate.** `crates/es-ros2`, layer 11, already in `xtask/src/layering.rs`'s `LAYERS`. Lower-layer
dependencies only: `es-core` (1: `PhysTick`, `TickRate`), `es-math` (0: `Pose`), `es-ir` (6:
`es_ir::image::ImageSpec`, `DeploymentIr`), `es-safety` (8: `SafeAction`, `SafetyPlane`),
`es-runtime-embedded` (9, `default-features = false`: `core_rt::EmbeddedCore` without `es-policy`),
`es-telemetry` (10). External: `serde`, `toml`, `thiserror`, `blake3` (workspace), `xxhash-rust`
(W1a), `zenoh` (W1b, optional).

**HIL is the module `es_ros2::hil`, not an `es-hil` crate.** §24.2 names `es-hil`, but §4.2's
table does not, and `cargo xtask layering` fails any `es-*` crate missing from `LAYERS` ("unknown
es-* crate not in the spec 4.2 LAYERS table"). Adding one is a §4.2 amendment, a human decision
(section 10). Rejected homes: `es-env` (9) cannot use `es-telemetry` (10), which §24.2 requires;
`es-transport` (11) is §21's tensor transport and the one crate allowed to link CUDA/HIP, which a
HIL rig must not inherit. `es-ros2` is the layer-11 sim-to-real crate §24 already names. To keep
a later extraction a file move, `hil` imports nothing from the ROS modules (a W1d test enforces it).

**Feature `zenoh`, off by default.** Codec, config, camera and HIL build without it; `es` and any
embedded consumer never link zenoh. W1b makes `cargo xtask ci` pass `--features es-ros2/zenoh` to
clippy and tests, so the PR tier compiles, lints and runs the session against **in-process loopback
peers** (listen `tcp/127.0.0.1:<port>`, multicast scouting off, no router, no outside network). The
pin is `zenoh = "=1.8.0"` (what ROS 2's `zenoh_cpp_vendor` builds), `default-features = false`,
`features = ["transport_tcp"]`: no rustls/ring/quinn (§25.1: nothing listens beyond localhost by
default; TLS for remote links is not this crate's job). `.cargo/config.toml` gains
`[resolver] incompatible-rust-versions = "fallback"` so the new dependency tree resolves inside
`rust-version = "1.85"`.

**No new trait (INV-17).** Message dispatch is `enum MsgType` / `enum Msg`; links are concrete
structs; the plant is the caller passing `q`, `qd` per tick.

## 3. Wire codec (W1a)

- **CDR.** `CdrWriter`: little-endian, header `00 01 00 00`, alignment measured from the byte after
  the header, no trailing padding. `CdrReader`: accepts `00 00`/`00 01`, ignores the options bytes,
  tolerates ≤ 3 trailing bytes. Rules and golden bytes: `ros2-cdr.md`.
- **Trust boundary (§25.1).** Network bytes are untrusted: decoding is total (never panics, property
  tested), checks `count × element_size ≤ remaining` **before** allocating, caps a message at
  `MAX_MESSAGE_BYTES = 64 MiB` (the telemetry frame cap), and requires NUL-terminated UTF-8 strings.
- **Messages** are plain structs with `to_cdr` / `from_cdr`. `MsgType` carries `ros_name()`,
  `dds_name()`, `rihs01()` from a const table.
- **Type hashes are a table, not a computation.** The subset is fixed; the table is golden-checked
  against RoboStack's `share/<pkg>/msg/<T>.json` and rosbags' `hash_rihs01`, which agree on all 11
  types. Computing REP-2016 hashes for arbitrary types is not needed.
- **Names.** Absolute ROS names only (`/`, then `/`-separated `[A-Za-z_][A-Za-z0-9_]*`); `~` and
  relative names are resolved by the caller. Topic key:
  `<domain>/<name minus one leading and one trailing '/'>/<dds type>/<RIHS01>`.
- **Liveliness.** `LivelinessToken` encode + total parse (`NN`, `MP`, `MS`; `SS`/`SC` parse only);
  QoS chunk with rmw_zenoh default elision; `classify(key) -> TokenScheme::{RmwZenoh,
  Ros2DdsBridge, Other}` by the verbatim prefixes in `rmw-zenoh.md`.
- **Attachment.** `Attachment { seq: i64, source_timestamp_ns: i64, gid: [u8; 16] }`, exactly 33
  bytes; `gid_of(token_key_expr)` = XXH3-128, low 64 bits LE then high 64 bits LE.

## 4. Session and modes (W1b)

### 4.1 Configuration

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

Mode A defaults mirror `DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5` (peer, `tcp/localhost:7447`,
multicast off), so a stock `rmw_zenohd` on the same host is found with no configuration. Unknown
keys are rejected.

### 4.2 A ⊕ B, enforced twice (§24.1)

| Code | Condition | Effect |
|---|---|---|
| `ROS2-001` | both `[ros2.rmw_zenoh]` and `[ros2.dds_bridge]` | config rejected, no session opened |
| `ROS2-002` | `[ros2.rust_dds]` | config rejected: mode C is not built |
| `ROS2-003` | mode A and `liveliness().get("@/*/@ros2_lv/**")` returns a token | `Ros2Node::open` fails |
| `ROS2-004` | mode B and `liveliness().get("@/*/@ros2_lv")` returns nothing | `open` fails: no bridge |
| `ROS2-005` | mode B and `liveliness().get("@ros2_lv/<domain>/**")` returns a token | `open` fails |
| `ROS2-006` | after start, a liveliness subscriber on the conflicting prefix sees a token | node latches the conflict; every later `put`/`send` returns it (fail closed) |
| `ROS2-010` | a generic `Publisher` on an `[[ros2.actuator]]` topic | refused (section 4.5) |
| `ROS2-011` | `put` of a `Msg` whose type is not the publisher's | refused |
| `ROS2-012` | TRANSIENT_LOCAL requested | refused (section 4.3) |

The probe runs before the node declares any token of its own, bounded by `liveliness_timeout_ms`. A
query only sees what is reachable at that instant, hence `ROS2-006`. Failing closed on an actuator
topic leaves the downstream controller without commands, which its own timeout handles; it never
sends a command that bypassed the checks. Codes come from `Ros2Error::code()`; they are runtime
conditions, not IR diagnostics, so they do not go in `es-ir-types::codes`.

### 4.3 Mode A data path

- `Ros2Node::open(&Ros2Config)`: zenoh `open(..).wait()`, probe, declare the `NN` token.
- `publisher(name, MsgType, depth)`: declare a publisher on the topic key and an `MP` token.
  `Publisher::put(&Msg)` checks the type, encodes CDR, attaches `(seq from 1, SystemTime ns, gid)`,
  sets no encoding.
- `subscriber(name, MsgType, depth)`: declare a subscriber and an `MS` token; delivery through a
  bounded FIFO (`recv_timeout`). Samples lacking a well-formed 33-byte attachment are dropped and
  counted, as rmw_zenoh does.
- QoS: RELIABLE, VOLATILE, KEEP_LAST(depth). TRANSIENT_LOCAL needs zenoh-ext advanced pub/sub whose
  key literals are unverified -> `ROS2-012`.
- Wall-clock stamps are wire metadata; nothing on the Safety Plane side reads them (§3.4).

### 4.4 Mode B data path

Key = `bridge_namespace` + name without its leading `/` (the plugin's `ros2_name_to_key_expr`), CDR
payload, no tokens of our own (the bridge owns DDS discovery), no attachment. Unit-tested against
the quoted mapping only; live: `Target / Status: unverified` until W1f.

### 4.5 Actuator path (INV-12)

`[[ros2.actuator]]` topics are reserved. The only writer is `ActuatorPublisher::<NJ>::send(&SafeAction<NJ>)`,
publishing `std_msgs/msg/Float64MultiArray { layout: { dim: [], data_offset: 0 }, data: q }`: the
command type of ros2_control's `forward_command_controller` (`using CmdType =
std_msgs::msg::Float64MultiArray;`, subscribed on `"~/commands"`), which refuses a size mismatch
("command size (%zu) does not match number of interfaces (%zu)"). A `SafeAction` only comes out of
the Safety Plane, so this crate offers no way to put an unvalidated action on an actuator topic.
`NJ != joints.len()` fails at construction. Inbound `sensor_msgs/JointState` is reordered by name
into the configured joint order (a missing joint rejects the sample) before it reaches
`SafetyPlane::observe_state` / `sensor_seen`.

## 5. Message subset

| Type | Purpose |
|---|---|
| `std_msgs/String` | interop smoke test with `demo_nodes_cpp` talker/listener |
| `sensor_msgs/JointState` | robot state in |
| `std_msgs/Float64MultiArray` (+ `MultiArrayLayout`, `MultiArrayDimension`) | validated command out |
| `sensor_msgs/Image`, `sensor_msgs/CameraInfo` (+ `RegionOfInterest`) | camera ingest |
| `std_msgs/Header`, `builtin_interfaces/Time` | nested |

Not in W1: `trajectory_msgs/JointTrajectory` (hash recorded for later), TF (extrinsics come from
configuration), `CompressedImage`, services, actions, TRANSIENT_LOCAL topics.

## 6. Camera ingest (W1c)

Output: an `ImageSpec` that is **checked against** the Observation IR's declared one (never
substituted for it), an HWC byte buffer, a `PhysTick`, and the Safety Plane inputs `obs_age` and
`sensor_seen`.

### 6.1 `CameraInfo` -> `ImageSpec`

| Field | Source |
|---|---|
| `width`, `height` | `CameraInfo.width/height` (calibration resolution), then section 6.2 |
| `intrinsics`, raw stream | `k` row-major: `fx = k[0]`, `skew = k[1]`, `cx = k[2]`, `fy = k[4]`, `cy = k[5]` |
| `intrinsics`, `rectified = true` | `p`: `fx = p[0]`, `skew = p[1]`, `cx = p[2]`, `fy = p[5]`, `cy = p[6]`; `distortion = None` |
| `plumb_bob` | `d = [k1, k2, t1, t2, k3]` -> `BrownConrady { k1, k2, k3, p1: t1, p2: t2 }`, `Pinhole` |
| `rational_polynomial` | `d = [k1, k2, p1, p2, k3, k4, k5, k6]`; `k4 = k5 = k6 = 0.0` exactly -> `BrownConrady`, else `CAM-004` (no rational variant exists; section 10) |
| `equidistant` | `d = [k1, k2, k3, k4]` -> `KannalaBrandt`, `Fisheye` (image_geometry uses `cv::fisheye` for it) |
| `""` with all-zero `d` | `None` |
| `extrinsics`, `color_space`, `shutter`, `exposure`, `rate_hz` | `CameraIngestConfig`; ROS carries none of them and no default is invented |
| `channels`, `dtype`, `depth_scale` | encoding, section 6.3 |

Monocular only: `r` must be identity and `p[3]`, `p[7]` (`Tx`, `Ty`) zero (`CAM-005`).

### 6.2 ROI, binning, size (INV-14)

image_geometry (`pinhole_camera_model.cpp`, and Python `from_camera_info`) applies the ROI first,
then binning: `k[0,2] = (k[0,2] - roi.x_offset) / binning_x`, `k[0,0] /= binning_x`; binning 0 is
1; an all-zero ROI is full resolution. Ingest does the same **through the `ImageSpec` transforms**:

```
spec = calib.cropped(Rect { roi }, true)                          // skipped for an all-zero ROI
spec = spec.resized(roi.width / bx, roi.height / by, true)       // skipped when bx = by = 1
```

- `roi.width % bx != 0` -> `CAM-006` (the resize ratio would not be `1 / bx`).
- The result must equal `Image.width/height`; otherwise `CAM-007`, unless the config sets
  `rescale_to_image = true`, which applies one more `resized(image.width, image.height, true)`.
- `rescale = false` is never passed. No code in `camera` writes an intrinsic except the `k`/`p`
  indexing of section 6.1.
- Known divergence: image_geometry leaves `k[0,1]` (skew) unscaled under binning; Appendix B.2's
  `Intrinsics::scaled` scales it. Fixtures use zero skew; for a non-zero skew the spec's rule wins.

### 6.3 Encodings

| `encoding` | `channels` / `dtype` | Conversion | `depth_scale` |
|---|---|---|---|
| `rgb8` | `Rgb` / `U8` | none | — |
| `bgr8` | `Rgb` / `U8` | swap R, B | — |
| `rgba8` / `bgra8` | `Rgba` / `U8` | none / swap R, B | — |
| `mono8` | `Gray` / `U8` | none | — |
| `mono16` | `Gray` / `U16` | to LE if `is_bigendian` | — |
| `16UC1` | `Depth` / `U16` | to LE if `is_bigendian`; 0 stays 0 | `0.001` (REP-118: "depth in millimeters"; "The value 0 denotes an invalid depth") |
| `32FC1` | `Depth` / `F32` | to LE if `is_bigendian` | `1.0` (REP-118: "depth ... in meters") |
| `uyvy`, `yuv422` | `Rgb` / `U8` | bit-exact `cv::COLOR_YUV2RGB_UYVY` | — |
| `yuyv`, `yuv422_yuy2` | `Rgb` / `U8` | bit-exact `cv::COLOR_YUV2RGB_YUY2` | — |
| other | `CAM-001` | | |

cv_bridge maps `yuv422 -> cv::COLOR_YUV2RGB_UYVY` and `yuv422_yuy2 -> cv::COLOR_YUV2RGB_YUY2`
(`cv_bridge.cpp`). Checks before any allocation (§25.1): `step ≥ width × bpp`,
`data.len() == step × height`, `width × height × bpp ≤ max_bytes`, even width for YUV (`CAM-002`).
Row padding beyond `width × bpp` is stripped.

**YUV 4:2:2 constants (W1c).** `crates/es-ros2/src/camera.rs` copies OpenCV's fixed-point BT.601
integers verbatim from `modules/imgproc/src/color_yuv.simd.hpp` (OpenCV 5.0.0, the version inside
`opencv-python-headless 5.0.0.93`, which is what produced `tests/golden/ros2/camera/yuv/*.rgb`):

| Constant | Value |
|---|---|
| `ITUR_BT_601_SHIFT` | 20 |
| `ITUR_BT_601_CY` | 1220542 |
| `ITUR_BT_601_CUB` | 2116026 |
| `ITUR_BT_601_CUG` | -409993 |
| `ITUR_BT_601_CVG` | -852492 |
| `ITUR_BT_601_CVR` | 1673527 |

Per pixel pair, following `YUV422toRGB888Invoker` with `uIdx = 0` (`uidx = 1 - yIdx`,
`vidx = (2 + uidx) % 4`), `uvToRGBuv` and `yRGBuvToRGBA`:
`ruv = 2^19 + CVR·(v-128)`, `guv = 2^19 + CVG·(v-128) + CUG·(u-128)`, `buv = 2^19 + CUB·(u-128)`,
`y = max(0, Y-16)·CY`, and each channel is `clamp((y + ·uv) >> 20, 0, 255)`. `yIdx` is 1 for
`uyvy`/`yuv422` and 0 for `yuyv`/`yuv422_yuy2`. The goldens decide, and they agree byte for byte
(`uyvy_and_yuyv_match_opencv_bit_exact`).

**cv_bridge spelling (W1c, measured).** RoboStack Kilted's `ros-kilted-cv-bridge 4.1.0` rejects
the modern names: `encoding_to_cvtype2("uyvy")` raises `Unrecognized image encoding [uyvy]`, and
so does `"yuyv"`; only the deprecated `yuv422` / `yuv422_yuy2` resolve (to `CV_8UC2`). The
`--ros` cross-check therefore hands cv_bridge the deprecated spelling of the same layout, and
ingest accepts all four names (the table above).

### 6.4 Identity

- `Image.header.frame_id` and `CameraInfo.header.frame_id` must equal the configured `frame_id`
  (`CAM-003`). ROS optical frames and `ImageSpec::extrinsics` share the convention `+Z` forward,
  `+X` right, `+Y` down (§3.1): no axis change.
- The derived spec must match the declared one (§26.1: "what is not validated is not executed"):
  enum fields and size equal, intrinsics per `intrinsics_consistent_with`, distortion coefficients
  within 1e-9 relative. `CAM-008` names the first differing field.
- Latest `CameraInfo` wins; a content change (header excluded) re-derives and re-checks.

### 6.5 Time (§18.1, §3.4)

`StampClock { epoch, rate: TickRate }`: `tick = floor((stamp_ns − epoch_ns) × num / (den × 10^9))`
in `i128`, no float. A stamp before `epoch` or `nanosec ≥ 10^9` -> `CAM-009`. The caller picks the
epoch (normally the first stamp) and records it. `obs_age(now, stamp) -> Micros` is a saturating
integer difference. Each accepted frame yields `sensor_seen(cfg.sensor, tick)`; its age is the
`obs_age` passed to `validate` -- which is how a stalled camera trips `StaleObservation` /
`SensorDropout` (§9.4) without a new watchdog.

## 7. HIL (W1d)

### 7.1 Topology

```
controller under test --UDP Command/Heartbeat--> HilLink --> HilCore<NJ,H> --SafeAction--> plant (sim)
                      <--UDP State(q, qd)------                  |
                                                                 +--> .eshil log, HilStats -> es-telemetry
```

`HilCore` runs the Safety Plane through `es_runtime_embedded::core_rt::EmbeddedCore::step`, the loop
the robot runs (§9.5), between the external controller and the simulated actuators. The chunk
execution mode is the Deployment IR's `execution`, never the controller's choice. `HilCore`
consumes decoded events and is transport-agnostic; `HilLink` is the UDP front end. A ROS 2 front
end (§24.2 "ROS 2 or low-latency UDP") would reuse `HilCore`; it is not in W1.

### 7.2 Wire protocol (little-endian)

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
| Command | `u64 obs_tick` (the `State.tick` it answers), `u16 rows` (≤ H), `u16 0`, `rows × NJ` f64 |
| Heartbeat | empty |
| Bye | `u16 reason`: 1 hash mismatch, 2 shape mismatch, 3 version, 4 shutdown |

- One message per datagram, ≤ 65,507 bytes (> 1,472 fragments on a 1,500 MTU link; documented,
  not forbidden).
- Dropped before `HilCore`, and counted: bad magic, version, length or tag, wrong session
  (`rx_invalid`); `seq ≤` last accepted (`rx_stale`). A `seq` gap adds to `rx_lost`.
- f64 travel as `to_bits`: a NaN arrives as the same NaN and meets the plane's non-finite rule.

### 7.3 Tick binning: where non-determinism stops

§24.2 withholds tier-1 determinism from HIL because the controller is external. The
non-determinism is *when* datagrams arrive; `HilCore::tick(now, q, qd)` fixes it once per tick:

1. `HilLink` drains the socket (non-blocking) and hands over what it accepted since the last tick,
   in arrival order.
2. Any heartbeat -> `plane.heartbeat(now)`, once.
3. Commands: highest `seq` wins (the rest count `superseded`). If
   `now > obs_tick + ceil(inference_budget / control_period)` it counts a **deadline miss** and is
   still submitted; acceptability is the plane's `InferenceDeadline` watchdog's call, not the link's.
4. `obs_age = (now − obs_tick of the last submitted command) × control_period_us`, saturating;
   0 before the first command.
5. `EmbeddedCore::step(chunk_or_none, now, obs_age) -> SafeAction`.
6. Steps 2–5's inputs and the decision are appended to the log; then `State { now, q, qd }` is sent.

Every plane input is in the log as integers and bit patterns. `Instant`/`SystemTime` feed statistics only.

### 7.4 Input log `.eshil` v1

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

Each `Decision` follows its `Step`. Replay re-derives `deployment_hash()` from the embedded IR and
requires it to equal the header (JSON formatting is irrelevant; the canonical hash decides). A log
without a trailer (crash) replays up to its last complete record and reports `truncated`.

### 7.5 Replay and the M3 gate

`hil::replay::<NJ, H>(log) -> Result<ReplayReport, HilLogError>`: rebuild the plane with
`SafetyPlane::from_ir`, wrap it in `EmbeddedCore::with_plane`, apply `ObserveState` / `Heartbeat` /
`Step` in order, re-encode each `SafeAction` as a `Decision` record, compare bytes.
`ReplayReport { steps, identical, first_divergence: Option<(u64 index, PhysTick)>, live_hash,
replay_hash, truncated }`.

**Gate (§24.2, §28.5):** a live run over real loopback UDP, with injected delay, loss, reordering,
a late command, a NaN row and a heartbeat gap, replays to **byte-identical decisions**
(`identical`, `live_hash == replay_hash`), and the run is non-vacuous: at least one `Clamped`, one
`Fallback`, one `NonFinite` event, one `HeartbeatLoss`, one deadline miss.

It proves the plane's HIL decisions are a deterministic function of the logged inputs, reproduced
offline by the same `EmbeddedCore` code. It does not prove that a physical controller, network or
robot produces those inputs, or that the `thumbv7em` build decides identically on a
microcontroller (same source, different binary): gate 14, `Target / Status: unverified`.

### 7.6 Telemetry (§24.2)

`HilStats`: `u64` `ticks`, `commands`, `deadline_miss`, `rx_lost`, `rx_stale`, `rx_invalid`,
`superseded`, `heartbeats`; `i64` ns `rtt_last`, `jitter` (RFC 3550 estimator
`J += (|D| − J) / 16` over the State→Command round trip). Every `stats_every` ticks:
`Frame { tick, wall_ns, stream: HIL_STATS_STREAM, payload: Payload::Scalars(..) }`,
`HIL_STATS_STREAM = StreamId(0x4849_4C31)`, field order as listed and fixed in `hil::stats`.
Published with `es_telemetry::transport::Server::publish`, which never blocks the loop. Numbers from
CI loopback runs are observations, not performance claims (§12.4).

### 7.7 Security (§25.1)

Default bind `127.0.0.1`; anything else is explicit configuration. Every datagram carries the
16-byte keyed-blake3 tag; the 32-byte key is read from a file outside the repo (`*.eshilkey`,
gitignored). The tag authenticates, it does not encrypt: HIL state is not secret, spoofed commands
must not reach the plane. Tests use a fixed test key.

## 8. Oracles and CI tiers (§1.4, §26.2)

| Oracle | Produces / checks | Needs | Runs in | Missing -> |
|---|---|---|---|---|
| rosbags 0.11.5 | CDR goldens `tests/golden/ros2/cdr/**`, `rihs01.json` | `ES_PYTHON` with `rosbags` | checked-in goldens (PR); provenance re-run (oracle job) | `SKIP` + reason |
| PyPI `xxhash` 4.0.1 | `tests/golden/ros2/gid.json` | `ES_PYTHON` | same | `SKIP` |
| RoboStack JSON, `rclpy.serialization.serialize_message` | hash cross-check; real-RMW CDR bytes | `ES_ROS2_ENV` | oracle job / Linux server | `SKIP` |
| rmw_zenoh design.md examples | token and key literals | — | PR | — |
| live `rmw_zenohd`, `ros2` CLI, `demo_nodes_cpp` | interop both ways; `tests/golden/ros2/rmw_zenoh/**` capture | `ES_ROS2_ENV` | oracle job / Linux server | `SKIP`; `RAN rmw_zenoh_interop` when run |
| image_geometry, cv_bridge (RoboStack) | intrinsics under ROI/binning; encoding conversions | `ES_ROS2_ENV` | golden generation, provenance | `SKIP` |
| opencv-python-headless 5.0.0.93 | YUV goldens | `ES_PYTHON` with `cv2` | provenance | `SKIP` |
| HIL live == replay | the gate | — | PR | — |

`ES_ROS2_ENV` is the prefix of a RoboStack environment. Reference commands run through
`crates/es-ros2/scripts/ros2-env.sh <prefix> <cmd...>`: export `CONDA_PREFIX=<prefix>`, prepend
`<prefix>/bin` to `PATH`, source `<prefix>/etc/conda/activate.d/*.sh`, `exec` the command. No
micromamba/pixi at test time. Whether that activation suffices for `ros2` is **unverified** until
W1b runs it; with `ES_ROS2_ENV` set, any failure is a test failure, never a SKIP. SKIP reasons must
avoid `gpu`, `vulkan`, `render`, `slangc`, `device` (`xtask` classifies those as GPU skips).

Linux x86_64, no docker, no sudo (RoboStack recommends pixi; micromamba shown, same channels):

```
micromamba create -y -p "$HOME/envs/ros2-kilted" -c conda-forge -c robostack-kilted \
  ros-kilted-ros-base=0.12.0 ros-kilted-rmw-zenoh-cpp=0.6.6 ros-kilted-demo-nodes-cpp=0.36.4 \
  ros-kilted-cv-bridge=4.1.0 ros-kilted-image-geometry=4.1.0
export ES_ROS2_ENV="$HOME/envs/ros2-kilted"
python3 -m venv "$HOME/envs/es-oracles"
"$HOME/envs/es-oracles/bin/pip" install rosbags==0.11.5 xxhash==4.0.1 opencv-python-headless==5.0.0.93
export ES_PYTHON="$HOME/envs/es-oracles/bin/python"
```

All five RoboStack packages exist for `linux-64` in `robostack-kilted` (prefix.dev repodata,
2026-09-14); kilted is pinned to `python_abi 3.12`. `cv-bridge`/`image-geometry` are `np126`
builds (`_10`) while the rest are `np2` builds (`_21`): co-installation in one prefix is
**unverified**. If the solver refuses, W1c uses a second prefix, `ES_ROS2_VISION_ENV`, and records it.

## 9. Packet order

W1a -> W1b -> W1c -> W1d, sequential: all four edit `crates/es-ros2/src/lib.rs`,
`crates/es-ros2/Cargo.toml` and `Cargo.lock`. W1c and W1d do not need W1b's session and may run
before it. Not yet written: W1e (`es hil replay <log>`, ROS 2 / `ES_ROS2_ENV` lines in
`es --check-deps`) and W1f (mode B live oracle). §28.5 budgets 9 packets for W1; the rest are the
hardware-bound rows of section 1.

## 10. Open questions for a human

1. `es-hil` as its own layer-11 crate (amend §4.2 and `LAYERS`), or `es_ros2::hil` as designed?
2. W1b makes the PR job compile zenoh (tcp only, ~270 crates). If the cold job exceeds §26.2's
   10 minutes, move the loopback session tests to the oracle job?
3. `rational_polynomial` with non-zero `k4..k6`: keep rejecting, or add
   `DistortionModel::RationalPolynomial` to `es-ir-types` (a schema change under §25.3)?
4. Gate 14: close M3 recorded `Target / Status: unverified`, or hold it for a robot cell (already
   raised in `docs/reviews/M3.md`)?
5. Mode C: cut from v1.0, or deferred?
6. HIL keys beyond one local key file (per-rig keys, rotation)?
