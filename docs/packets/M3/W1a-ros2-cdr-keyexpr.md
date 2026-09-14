# M3 W1a — `es-ros2` wire codec: CDR, message subset, key expressions, liveliness, attachment

Design note: `docs/design/ros2-boundary.md` sections 2, 3, 5, 8. Digests: `docs/api-notes/ros2-cdr.md`
(bytes, layouts, RIHS01 table, golden hex), `docs/api-notes/rmw-zenoh.md` (keys, tokens, attachment).
First packet of M3 W1; creates the crate. No zenoh here.

## context

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

Notes: root `Cargo.toml` gets only the `es-ros2 = { path = "crates/es-ros2" }` workspace
dependency entry. Goldens are additions only. `ros2-cdr.md` changes only in the "unverified" rows
this packet's oracles settle.

## spec

- §24.1 mode A: "Implements key expression, CDR, attachment, and liveliness with `zenoh-rs`". This
  packet is the half that needs no network, so it can be judged on any machine.
- §1.4: goldens come from the reference (rosbags, RoboStack's generated JSON, rmw_zenoh's own
  design.md, PyPI `xxhash`), never from this crate's encoder.
- §25.1: bytes from a network cross a trust boundary; decoders are total and allocation-bounded.
- §4.2: `es-ros2` is layer 11 (already in `LAYERS`); this packet depends on nothing above layer 1.
- §3.4: no `HashMap`; INV-17: no trait (message dispatch is an enum).

## oracle

```
cargo fmt --check
cargo clippy -p es-ros2 --all-targets -- -D warnings
cargo test -p es-ros2
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

Reference (design note section 8 for the environments):

```
ES_PYTHON=$HOME/envs/es-oracles/bin/python  cargo test -p es-ros2 --test gen_goldens -- --nocapture
ES_ROS2_ENV=$HOME/envs/ros2-kilted ES_PYTHON=$HOME/envs/es-oracles/bin/python \
  cargo test -p es-ros2 --test gen_goldens -- --nocapture
```

`python/gen_ros2_goldens.py <out_dir>` (rosbags 0.11.5, `get_typestore(Stores.ROS2_KILTED)`,
`serialize_cdr(msg, typename, little_endian=True)`; xxhash 4.0.1) writes:

- `cdr/string_hello`, `cdr/string_empty`, `cdr/string_hello_be` (`little_endian=False`),
  `cdr/joint_state`, `cdr/image_rgb8_2x1`, `cdr/camera_info_plumb_bob`,
  `cdr/float64_multi_array` -- each `.bin` plus `.json` (typename, field values, rosbags version).
  The messages are exactly the ones in `ros2-cdr.md`'s golden table, whose hex they must reproduce.
- `rihs01.json` from `Typestore.hash_rihs01` for the 11 types in `ros2-cdr.md`.
- `gid.json`: for both design.md tokens in `rmw-zenoh.md`, `xxhash.xxh3_128_intdigest(key)` split
  into `low64.to_bytes(8, 'little') + high64.to_bytes(8, 'little')`, hex.

`tests/gen_goldens.rs` (the `crates/es-compile/tests/gen_goldens.rs` pattern):

- `goldens_are_what_rosbags_and_xxhash_produce`: re-runs the script into a temp dir and
  byte-compares every file with `tests/golden/ros2/`. Prints
  `SKIP goldens_are_what_rosbags_and_xxhash_produce: <why>` without `rosbags`/`xxhash`,
  `RAN gen_goldens` when it ran.
- `robostack_hashes_and_rclpy_bytes_agree_with_the_goldens`: only with `ES_ROS2_ENV`; runs
  `scripts/ros2-env.sh "$ES_ROS2_ENV" python python/gen_ros2_goldens.py --check-ros tests/golden/ros2`
  with `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, which (a) compares each `share/<pkg>/msg/<T>.json`
  `type_hashes[0].hash_string` with `rihs01.json` and (b) builds the same messages with the ROS
  message classes and compares `rclpy.serialization.serialize_message(msg)` with each `.bin`,
  reporting options-byte and trailing-byte differences separately. Any difference fails; record the
  outcome in `ros2-cdr.md`'s "Live ROS 2 byte capture" row. Without `ES_ROS2_ENV`: SKIP with reason.

`tests/codec.rs` (pure, PR tier):

- `cdr_goldens_match_rosbags` -- decode each golden to the `.json` values; re-encode to identical
  bytes (`string_hello_be` decode only).
- `cdr_empty_sequences_get_no_alignment_padding` -- JointState, 76 bytes.
- `cdr_camera_info_is_357_bytes_without_trailing_padding`.
- `cdr_decoder_accepts_options_bits_and_up_to_three_trailing_bytes`.
- `cdr_decoder_rejects_a_fourth_trailing_byte_and_unknown_representation_ids`.
- `cdr_decoding_is_total` -- proptest over arbitrary bytes for every `MsgType`: never panics; a
  `u32::MAX` sequence count in a 16-byte buffer returns `Truncated` inside
  `es_core::alloc_count::assert_no_alloc`.
- `msg_type_table_matches_the_rihs01_golden`.
- `topic_key_exprs_match_rmw_zenoh_design_md` -- `/chatter` and `/robot1/chatter`, domain 0.
- `design_md_liveliness_tokens_round_trip_byte_identical` -- the `NN` and `MP` examples: parse ->
  fields (`zid`, ids, kind, `%` namespace, `/chatter`, type, hash, depth 7) -> `to_key_expr()` equals input.
- `liveliness_parse_is_total_and_keeps_unknown_trailing_chunks` -- proptest; rolling's `/<backends>` kept.
- `token_scheme_separates_rmw_zenoh_from_the_dds_bridge` -- `@ros2_lv/0/...` vs `@/<zid>/@ros2_lv/MP/...`
  vs `0/chatter/...`.
- `attachment_is_seq_ts_leb128_len_gid` -- `seq = 1`, `ts = 1_700_000_000_123_456_789`,
  `gid = 00..0f` encode to the 33 bytes given by `rmw-zenoh.md`'s table (hand-assembled in the
  test); decode rejects 32 or 34 bytes and a length prefix other than `0x10`.
- `gid_matches_python_xxhash_for_the_design_md_tokens` -- `gid.json`.
- `ros_names_are_validated` -- `/chatter`, `/a/b_c` ok; `chatter`, `//x`, `/x/`, `~/x`, `/1x`, `/x y` rejected.

## acceptance

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

- `crates/es-ros2/Cargo.toml`: `[lints] workspace = true`; dependencies `es-core`, `thiserror`,
  `xxhash-rust = { version = "=0.8.18", default-features = false, features = ["xxh3"] }`;
  dev-dependencies `proptest`, `serde_json`, `es-core` with `alloc-count`. No zenoh, no tokio
  (`cargo tree -p es-ros2 -e normal` shows neither). `xxhash-rust` declares no `rust_version`;
  if a 1.85 toolchain is installed, `cargo +1.85 check -p es-ros2` passes, otherwise say so here.
- `scripts/ros2-env.sh` exactly as design note section 8 describes, `set -eu`, POSIX sh.
- RIHS01 strings are a `const` table, not computed.
- The `.bin` goldens are byte-identical to the hex in `ros2-cdr.md`.
- ≤ ~1,400 source lines; English only.

## forbidden

- zenoh, sessions, config, actuator path (W1b); `camera` (W1c); `hil` (W1d); `crates/es` (W1e).
- Any crate other than `es-ros2`; `xtask`; `.github/workflows/ci.yml`.
- Computing RIHS01; services/actions payloads; `JointTrajectory` (hash recorded only).
- Producing or editing goldens with this crate's encoder, or editing an existing golden.
- A trait, `HashMap`, `unsafe`.
