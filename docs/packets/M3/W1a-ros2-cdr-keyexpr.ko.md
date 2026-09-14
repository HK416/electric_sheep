<!-- Korean translation of docs/packets/M3/W1a-ros2-cdr-keyexpr.md. The English file is the working copy; regenerate this when it changes. -->

# M3 W1a — `es-ros2` wire 코덱: CDR, 메시지 서브셋, key expression, liveliness, attachment

Design note: `docs/design/ros2-boundary.md` 섹션 2, 3, 5, 8. 다이제스트: `docs/api-notes/ros2-cdr.md`
(바이트, 레이아웃, RIHS01 표, golden hex), `docs/api-notes/rmw-zenoh.md`(key, 토큰, attachment).
M3 W1의 첫 패킷; crate를 생성한다. zenoh는 여기 없음.

## context (범위)

```
Cargo.toml
Cargo.lock
crates/es-ros2/Cargo.toml
crates/es-ros2/src/lib.rs
crates/es-ros2/src/error.rs
crates/es-ros2/src/cdr.rs
crates/es-ros2/src/msg.rs
crates/es-ros2/src/names.rs
crates/es-ros2/src/attachment.rs
crates/es-ros2/tests/codec.rs
crates/es-ros2/tests/gen_goldens.rs
crates/es-ros2/python/gen_ros2_goldens.py
crates/es-ros2/scripts/ros2-env.sh
tests/golden/ros2/cdr/**
tests/golden/ros2/rihs01.json
tests/golden/ros2/gid.json
docs/api-notes/ros2-cdr.md
docs/packets/M3/W1a-ros2-cdr-keyexpr.md
```

참고: 루트 `Cargo.toml`은 `es-ros2 = { path = "crates/es-ros2" }` workspace dependency 항목만
얻는다. golden은 추가만 된다. `ros2-cdr.md`는 이 패킷의 오라클이 해결하는 "unverified" 행에서만
바뀐다.

## spec (사양)

- §24.1 모드 A: "Implements key expression, CDR, attachment, and liveliness with `zenoh-rs`".
  이 패킷은 네트워크가 필요 없는 절반이므로 어떤 머신에서든 판단할 수 있다.
- §1.4: golden은 reference에서 나온다(rosbags, RoboStack이 생성한 JSON, rmw_zenoh 자신의
  design.md, PyPI `xxhash`), 이 crate의 인코더에서는 절대 나오지 않는다.
- §25.1: 네트워크에서 온 바이트는 신뢰 경계를 넘는다; 디코더는 total이며 할당이 유계다.
- §4.2: `es-ros2`는 layer 11이다(이미 `LAYERS`에 있음); 이 패킷은 layer 1보다 위의 어떤 것에도
  의존하지 않는다.
- §3.4: `HashMap` 없음; INV-17: trait 없음(메시지 디스패치는 enum).

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-ros2 --all-targets -- -D warnings
cargo test -p es-ros2
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

레퍼런스(환경에 대해서는 설계 노트 섹션 8):

```
ES_PYTHON=$HOME/envs/es-oracles/bin/python  cargo test -p es-ros2 --test gen_goldens -- --nocapture
ES_ROS2_ENV=$HOME/envs/ros2-kilted ES_PYTHON=$HOME/envs/es-oracles/bin/python \
  cargo test -p es-ros2 --test gen_goldens -- --nocapture
```

`python/gen_ros2_goldens.py <out_dir>`(rosbags 0.11.5, `get_typestore(Stores.ROS2_KILTED)`,
`serialize_cdr(msg, typename, little_endian=True)`; xxhash 4.0.1)가 쓰는 것:

- `cdr/string_hello`, `cdr/string_empty`, `cdr/string_hello_be`(`little_endian=False`),
  `cdr/joint_state`, `cdr/image_rgb8_2x1`, `cdr/camera_info_plumb_bob`,
  `cdr/float64_multi_array` -- 각각 `.bin`과 `.json`(typename, 필드 값, rosbags 버전). 메시지는
  `ros2-cdr.md`의 golden 표에 있는 것과 정확히 같으며, 그 hex를 재현해야 한다.
- `ros2-cdr.md`의 11개 타입에 대한 `Typestore.hash_rihs01`로부터 `rihs01.json`.
- `gid.json`: `rmw-zenoh.md`의 두 design.md 토큰 각각에 대해,
  `xxhash.xxh3_128_intdigest(key)`를 `low64.to_bytes(8, 'little') + high64.to_bytes(8, 'little')`로
  나눈 것, hex.

`tests/gen_goldens.rs`(`crates/es-compile/tests/gen_goldens.rs` 패턴):

- `goldens_are_what_rosbags_and_xxhash_produce`: 스크립트를 임시 디렉터리에 다시 실행하고
  `tests/golden/ros2/`의 모든 파일과 바이트 단위로 비교한다. `rosbags`/`xxhash`가 없으면
  `SKIP goldens_are_what_rosbags_and_xxhash_produce: <why>`를, 실행되었으면 `RAN gen_goldens`를
  출력한다.
- `robostack_hashes_and_rclpy_bytes_agree_with_the_goldens`: `ES_ROS2_ENV`가 있을 때만 실행;
  `RMW_IMPLEMENTATION=rmw_zenoh_cpp`로
  `scripts/ros2-env.sh "$ES_ROS2_ENV" python python/gen_ros2_goldens.py --check-ros
  tests/golden/ros2`를 실행하며, 이는 (a) 각 `share/<pkg>/msg/<T>.json`의
  `type_hashes[0].hash_string`을 `rihs01.json`과 비교하고 (b) ROS 메시지 클래스로 같은
  메시지를 만들어 `rclpy.serialization.serialize_message(msg)`를 각 `.bin`과 비교하며,
  options-byte와 trailing-byte 차이를 따로 보고한다. 어떤 차이든 있으면 실패한다; 결과는
  `ros2-cdr.md`의 "Live ROS 2 byte capture" 행에 기록한다. `ES_ROS2_ENV`가 없으면: 사유와
  함께 SKIP.

## acceptance (수용 기준)

```rust
pub mod cdr {
    pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
    pub struct CdrWriter { /* .. */ }       // new(), u8/u32/i32/f64/bool/string/seq helpers, finish() -> Vec<u8>
    pub struct CdrReader<'a> { /* .. */ }   // new(&'a [u8]) -> Result<Self, CdrError>, mirrored getters, finish()
    pub enum CdrError { BadHeader, Truncated, TrailingBytes(usize), InvalidUtf8, MissingNul, TooLarge, BadLength }
}
pub mod msg {
    pub struct Time { pub sec: i32, pub nanosec: u32 }
    pub struct Header { pub stamp: Time, pub frame_id: String }
    pub struct StringMsg { pub data: String }
    pub struct MultiArrayDimension { pub label: String, pub size: u32, pub stride: u32 }
    pub struct MultiArrayLayout { pub dim: Vec<MultiArrayDimension>, pub data_offset: u32 }
    pub struct Float64MultiArray { pub layout: MultiArrayLayout, pub data: Vec<f64> }
    pub struct JointState { pub header: Header, pub name: Vec<String>, pub position: Vec<f64>, pub velocity: Vec<f64>, pub effort: Vec<f64> }
    pub struct Image { pub header: Header, pub height: u32, pub width: u32, pub encoding: String, pub is_bigendian: u8, pub step: u32, pub data: Vec<u8> }
    pub struct RegionOfInterest { pub x_offset: u32, pub y_offset: u32, pub height: u32, pub width: u32, pub do_rectify: bool }
    pub struct CameraInfo { pub header: Header, pub height: u32, pub width: u32, pub distortion_model: String, pub d: Vec<f64>,
                            pub k: [f64; 9], pub r: [f64; 9], pub p: [f64; 12], pub binning_x: u32, pub binning_y: u32, pub roi: RegionOfInterest }
    // every struct above: pub fn to_cdr(&self) -> Vec<u8>; pub fn from_cdr(bytes: &[u8]) -> Result<Self, CdrError>;
    pub enum MsgType { String, Float64MultiArray, JointState, Image, CameraInfo }
    impl MsgType { pub const fn ros_name(self) -> &'static str; pub const fn dds_name(self) -> &'static str; pub const fn rihs01(self) -> &'static str; }
    pub enum Msg { String(StringMsg), Float64MultiArray(Float64MultiArray), JointState(JointState), Image(Image), CameraInfo(CameraInfo) }
    impl Msg { pub fn msg_type(&self) -> MsgType; pub fn to_cdr(&self) -> Vec<u8>; pub fn from_cdr(ty: MsgType, bytes: &[u8]) -> Result<Self, CdrError>; }
}
pub mod names {
    pub fn validate_name(name: &str) -> Result<(), NameError>;
    pub fn topic_key_expr(domain_id: u32, name: &str, ty: MsgType) -> Result<String, NameError>;
    pub fn mangle(s: &str) -> String;                           // '/' -> '%'
    pub struct QosKey { /* each component Option<u64>; None = empty (rmw_zenoh default) */ }
    pub enum EntityKind { Node, Publisher, Subscriber, Service, Client }
    pub struct TopicPart { pub name: String, pub type_name: String, pub type_hash: String, pub qos: QosKey }
    pub struct LivelinessToken { pub domain_id: u32, pub zid: String, pub node_id: u64, pub entity_id: u64, pub kind: EntityKind,
                                 pub enclave: String, pub namespace: String, pub node_name: String,
                                 pub topic: Option<TopicPart>, pub extra: Vec<String> }
    impl LivelinessToken { pub fn to_key_expr(&self) -> String; pub fn parse(key: &str) -> Result<Self, NameError>; }
    pub enum TokenScheme { RmwZenoh, Ros2DdsBridge, Other }
    pub fn classify(key: &str) -> TokenScheme;
}
pub mod attachment {
    pub struct Attachment { pub seq: i64, pub source_timestamp_ns: i64, pub gid: [u8; 16] }
    impl Attachment { pub fn encode(&self) -> [u8; 33]; pub fn decode(bytes: &[u8]) -> Result<Self, AttachmentError>; }
    pub fn gid_of(token_key_expr: &str) -> [u8; 16];
}
```

- `crates/es-ros2/Cargo.toml`: `[lints] workspace = true`; 의존성 `es-core`, `thiserror`,
  `xxhash-rust = { version = "=0.8.18", default-features = false, features = ["xxh3"] }`;
  dev-dependency `proptest`, `serde_json`, `alloc-count`가 있는 `es-core`. zenoh도 tokio도
  없음(`cargo tree -p es-ros2 -e normal`에 둘 다 보이지 않음). `xxhash-rust`는
  `rust_version`을 선언하지 않는다; 1.85 툴체인이 설치되어 있으면 `cargo +1.85 check -p
  es-ros2`가 통과한다, 아니면 여기에 그렇게 적는다.
- `scripts/ros2-env.sh`는 설계 노트 섹션 8이 기술하는 그대로, `set -eu`, POSIX sh.
- RIHS01 문자열은 계산되지 않고 `const` 테이블이다.
- `.bin` golden은 `ros2-cdr.md`의 hex와 바이트 단위로 동일하다.
- 소스 코드 ~1,400줄 이하; 영어만.

## forbidden (금지)

- zenoh, 세션, config, actuator 경로(W1b); `camera`(W1c); `hil`(W1d); `crates/es`(W1e).
- `es-ros2` 외의 어떤 crate든; `xtask`; `.github/workflows/ci.yml`.
- RIHS01 계산; service/action payload; `JointTrajectory`(해시만 기록).
- 이 crate의 인코더로 golden을 생성하거나 편집하는 것, 또는 기존 golden을 편집하는 것.
- trait, `HashMap`, `unsafe`.
