# M3 W1b — `es-ros2` zenoh session: mode A, A ⊕ B exclusion, actuator path, live rmw_zenoh oracle

Design note: `docs/design/ros2-boundary.md` sections 2, 4, 5, 8. Digests: `docs/api-notes/rmw-zenoh.md`,
`docs/api-notes/zenoh-rs.md`. Depends on W1a.

## context

```
.cargo/config.toml
Cargo.lock
crates/es-ros2/Cargo.toml
crates/es-ros2/src/lib.rs
crates/es-ros2/src/error.rs
crates/es-ros2/src/config.rs
crates/es-ros2/src/session.rs
crates/es-ros2/src/actuator.rs
crates/es-ros2/tests/config.rs
crates/es-ros2/tests/session_loopback.rs
crates/es-ros2/tests/rmw_zenoh_interop.rs
crates/es-ros2/scripts/ros2-env.sh
tests/golden/ros2/rmw_zenoh/**
xtask/src/main.rs
.github/workflows/ci.yml
docs/api-notes/rmw-zenoh.md
docs/api-notes/zenoh-rs.md
docs/packets/M3/W1b-ros2-zenoh-session.md
```

Notes:
`.cargo/config.toml` gets only `[resolver] incompatible-rust-versions = "fallback"`.
`crates/es-ros2/Cargo.toml` adds optional `zenoh`, the `zenoh` feature, `es-safety`, `serde` and `toml`.
`ros2-env.sh` gets fixes only.
`tests/golden/ros2/rmw_zenoh/**` is additions only, captured from the reference.
`xtask/src/main.rs` only adds `--features es-ros2/zenoh` to the `ci` clippy and test steps.
`ci.yml` changes only the oracle job: the RoboStack environment plus the interop step.
The api-notes change only the rows this packet settles, which become dated results.

## spec

- §24.1: mode A is the default and uses `zenoh-rs`; "A and B cannot be enabled at the same time.
  They are enforced as mutually exclusive in configuration and validated with liveliness at
  startup"; mode C is experimental and not built (design note section 1).
- §9.1 and INV-12: the actuator topic accepts only a `SafeAction`; no bypass exists.
- §25.1: default localhost, no TLS/QUIC stack linked; network bytes go through W1a's total decoders.
- §26.2: PR tier < 10 min, so only in-process loopback tests run there; the live oracle runs in the
  oracle job. §1.4: an absent oracle SKIPs with a printed reason, a present one never skips.
- §3.4: wall-clock stamps are wire metadata only.
- §4.2: feature off by default so `es` (and anything embedded) does not link zenoh.

## oracle

```
cargo fmt --check
cargo clippy -p es-ros2 --all-targets -- -D warnings
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
cargo test -p es-ros2 --features zenoh
cargo tree -p es-ros2 --features zenoh -e normal --prefix none | grep -cE '^(rustls|ring|quinn|rsa) '   # prints 0
cargo xtask ci
```

Live reference (Linux, no docker, no sudo; environment per design note section 8):

```
export ES_ROS2_ENV=$HOME/envs/ros2-kilted
cargo test -p es-ros2 --features zenoh --test rmw_zenoh_interop -- --nocapture --test-threads=1 2>&1 | tee interop.log
grep -q '^RAN rmw_zenoh_interop' interop.log
# one-time capture, committed by this packet:
ES_ROS2_CAPTURE_DIR=tests/golden/ros2/rmw_zenoh \
  cargo test -p es-ros2 --features zenoh --test rmw_zenoh_interop capture_reference_goldens -- --ignored --nocapture
```

Harness (`rmw_zenoh_interop.rs`). Without `ES_ROS2_ENV`, each test prints
`SKIP <test>: ES_ROS2_ENV is unset` and returns. With it: reserve a free port `P`; start
`scripts/ros2-env.sh "$ES_ROS2_ENV" ros2 run rmw_zenoh_cpp rmw_zenohd` with
`ZENOH_CONFIG_OVERRIDE='listen/endpoints=["tcp/127.0.0.1:P"]'`; wait ≤ 20 s for a TCP connect on
`P`. Every ROS process gets `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, `ROS_DOMAIN_ID=73`,
`ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/127.0.0.1:P"]'`; our node uses `domain_id = 73` and
`connect = ["tcp/127.0.0.1:P"]`. Children are killed on drop; every wait has a timeout.

- `ros2_topic_echo_receives_our_string` -- our `/es_chatter` publisher at 10 Hz;
  `ros2 topic echo --once /es_chatter std_msgs/msg/String` exits 0 within 30 s printing `data: hello from es`.
- `ros2_cli_lists_our_node_and_topic` -- `ros2 node list` contains `/es_interop`;
  `ros2 topic list -t` contains `/es_chatter [std_msgs/msg/String]` (bypass the ros2 daemon: record
  here whether `--no-daemon` or `ros2 daemon stop` was needed).
  **Recorded 2026-09-14:** `--no-daemon` on both `node list` and `topic list -t`; needed in
  practice (the harness starts a fresh `rmw_zenohd` + node per test-suite run, and a stale
  `ros2` daemon from an earlier invocation would otherwise miss it). Confirmed passing live
  against RoboStack Kilted, `ros-kilted-rmw-zenoh-cpp 0.6.6`.
- `demo_talker_reaches_our_subscriber` -- `ros2 run demo_nodes_cpp talker`; our `/chatter`
  subscriber decodes `Hello World: <n>`; attachments are 33 bytes with strictly increasing `seq`;
  the talker's `MP` token from `liveliness().get("@ros2_lv/73/**")` round-trips through
  `LivelinessToken::parse` / `to_key_expr` byte-identically, and its attachment `gid ==
  gid_of(<that token>)`.
- `ros2_topic_pub_joint_state_reaches_our_subscriber` -- `ros2 topic pub --once -w 1 /es_js
  sensor_msgs/msg/JointState "{name: [j1, j2], position: [0.5, -1.0]}"`; decoded names and positions match.
  **Recorded 2026-09-14 -- the known open CDR issue, closed:** `ros2-cdr.md`'s W1a evidence
  (`rclpy.serialization.serialize_message`) showed an empty `float64[]` (`velocity`/`effort`)
  consuming an extra 8-byte alignment pad it "stops before the network" to confirm. This test
  and `capture_reference_goldens` below are that confirmation, and it does **not** reproduce on
  the wire: the captured payload (`tests/golden/ros2/rmw_zenoh/talker_capture.json`'s
  `joint_state_pub_hex`) is exactly 68 bytes with zero trailing bytes after `effort`'s 4-byte
  zero count -- byte-for-byte the same "align only when `count > 0`" layout this crate's
  `CdrWriter`/`CdrReader` already implement. **No change was made to `cdr.rs`** (forbidden
  territory without a failing test; none materialized). Full trace: `docs/api-notes/ros2-cdr.md`
  "Encapsulation header" and "Layout rules".
- `our_image_camera_info_and_float64_multi_array_echo_in_ros2` -- `ros2 topic echo --once` of each;
  output contains `encoding: rgb8`, `distortion_model: plumb_bob`, and the three `data` values.
- the file prints `RAN rmw_zenoh_interop` once all non-ignored tests ran. **Implementation
  note:** a sixth test, `z_ran_rmw_zenoh_interop`, sorts alphabetically after the five scenarios
  above; rustc's test harness runs `--test-threads=1` tests in name order (verified empirically),
  so it prints the marker only after a shared counter confirms all five actually ran (not
  skipped).
- `capture_reference_goldens` (`#[ignore]`) -- uses **raw zenoh-rs only** (no `es_ros2` encoder or
  parser): records the talker's and a `demo_nodes_cpp listener`'s `NN`/`MP`/`MS` token strings,
  three talker attachments and payloads (hex), one `ros2 topic pub` JointState payload, and the
  RoboStack package versions, into `$ES_ROS2_CAPTURE_DIR/talker_capture.json`. **Recorded
  2026-09-14:** ran on the GPU server against RoboStack Kilted; `talker_capture.json` committed.
  Two implementation notes for a future re-run: (1) `ES_ROS2_CAPTURE_DIR` is resolved relative to
  the *workspace root* (`CARGO_MANIFEST_DIR/../..`), not the test binary's CWD (cargo runs test
  binaries with CWD set to the package directory, `crates/es-ros2`); (2) the one-shot
  `ros2 topic pub` for the JointState capture needs `-w 0` -- `ros2 topic pub --once` defaults to
  waiting for a matched RMW subscription otherwise (confirmed empirically: it still blocked with
  neither `-w` given nor `-w 1`), and the raw zenoh subscriber here deliberately declares no `MS`
  token to match against.

`session_loopback.rs` (PR tier; two in-process peers on `127.0.0.1`, multicast off, no router):

- `mode_a_publisher_sends_cdr_with_a_33_byte_attachment` -- a raw zenoh subscriber on the topic key
  receives `to_cdr()` bytes, `seq` 1, 2, 3, `gid == gid_of(own MP token)`, no encoding set.
- `mode_a_node_declares_nn_and_mp_tokens_that_parse`.
- `mode_a_subscriber_decodes_and_drops_samples_without_attachment` -- `dropped()` counts them.
- `mode_a_refuses_to_start_next_to_a_bridge_token` (`ROS2-003`).
- `mode_b_refuses_without_a_bridge_plugin_token` (`ROS2-004`).
- `mode_b_refuses_next_to_an_rmw_zenoh_token` (`ROS2-005`).
- `a_bridge_token_appearing_later_fails_closed` (`ROS2-006` on the next `put` and `send`).
- `mode_b_keys_follow_the_bridge_mapping` -- `/chatter` -> `chatter`; `bridge_namespace = "/robot1"`
  -> `robot1/chatter`; no attachment, no tokens of our own.
- `generic_publisher_refuses_actuator_topics` (`ROS2-010`).
- `actuator_publisher_sends_the_safe_action` -- a `SafeAction` from a real `SafetyPlane` (fixture
  envelope widened, never disabled) arrives as `Float64MultiArray { dim: [], data_offset: 0, data == q }`.
- `type_mismatch_is_rejected` (`ROS2-011`); `transient_local_is_rejected` (`ROS2-012`).
- `committed_reference_capture_round_trips` -- every token in `tests/golden/ros2/rmw_zenoh/talker_capture.json`
  parses and re-encodes byte-identically, every attachment decodes, every payload decodes as its type.

`config.rs` (no feature): `both_mode_tables_are_rejected` (`ROS2-001`), `rust_dds_is_rejected`
(`ROS2-002`), `defaults_mirror_the_rmw_zenoh_session_config`, `unknown_keys_are_rejected`,
`actuator_joint_count_mismatch_is_rejected`.

## acceptance

```rust
pub struct ZenohEndpoints { pub mode: ZenohMode, pub connect: Vec<String>, pub listen: Vec<String> }
pub enum ZenohMode { Peer, Client }
pub enum Ros2Mode { RmwZenoh(ZenohEndpoints), DdsBridge { endpoints: ZenohEndpoints, bridge_namespace: String } }
pub struct ActuatorTopic { pub topic: String, pub joints: Vec<String> }
pub struct Ros2Config { pub domain_id: u32, pub node: String, pub namespace: String, pub enclave: String,
                        pub liveliness_timeout: Duration, pub mode: Ros2Mode, pub actuators: Vec<ActuatorTopic> }
pub fn parse_config(toml: &str) -> Result<Ros2Config, Ros2Error>;
impl Ros2Error { pub fn code(&self) -> &'static str; }       // "ROS2-001" ..

#[cfg(feature = "zenoh")]
impl Ros2Node {
    pub fn open(cfg: &Ros2Config) -> Result<Self, Ros2Error>;
    pub fn publisher(&self, name: &str, ty: MsgType, depth: u32) -> Result<Publisher, Ros2Error>;
    pub fn subscriber(&self, name: &str, ty: MsgType, depth: u32) -> Result<Subscriber, Ros2Error>;
    pub fn actuator<const NJ: usize>(&self, topic: &str) -> Result<ActuatorPublisher<NJ>, Ros2Error>;
}
impl Publisher { pub fn put(&self, msg: &Msg) -> Result<(), Ros2Error>; }
impl Subscriber { pub fn recv_timeout(&self, t: Duration) -> Result<Option<Received>, Ros2Error>; pub fn dropped(&self) -> u64; }
pub struct Received { pub msg: Msg, pub attachment: Option<Attachment>, pub key_expr: String }
impl<const NJ: usize> ActuatorPublisher<NJ> { pub fn send(&self, action: &SafeAction<NJ>) -> Result<(), Ros2Error>; }
```

Implementation note (2026-09-14): the code above is present exactly as pinned. Three small
additive members exist alongside it, none changing a pinned signature: `Ros2Node::config(&self)
-> &Ros2Config` (a plain accessor `actuator.rs` needs); `Ros2Node::liveliness_tokens(&self,
pattern, timeout) -> Result<Vec<String>, Ros2Error>` (the live interop tests' only way to look up
a *remote* node's tokens, since `Self::open`'s own probe only needs the faster first-match
check); and `Ros2Node::publisher_with_durability` / `enum Durability { Volatile,
TransientLocal }`, with `publisher` calling it with `Volatile` -- the pinned `publisher` has no
durability parameter, so `transient_local_is_rejected` (`ROS2-012`) needed some way to actually
request `TRANSIENT_LOCAL` to reject. No new trait (`Durability` is a plain enum).

- `zenoh = { version = "=1.8.0", default-features = false, features = ["transport_tcp"], optional = true }`;
  `[features] default = []`, `zenoh = ["dep:zenoh"]`. No direct `tokio`; the public API is
  synchronous (`zenoh::Wait::wait`). No zenoh `unstable`/`internal`/`shared-memory`.
- Codes and conditions exactly as design note section 4.2.
- `cargo xtask ci` passes with the feature on, on Windows and Linux. Record here the cold wall time
  of the PR job on the GitHub runner; over 10 minutes, raise design note open question 2 before merging.
  **Recorded 2026-09-14:** `cargo xtask ci` PASSED on Windows (local dev machine) and PASSED
  (fmt-check + clippy + `cargo test -p es-ros2 --features zenoh`) on the Linux GPU server after
  `cargo clean -p es-ros2`: ~20 s wall for that crate's own cold compile + lint + full test run
  (dependencies, incl. `zenoh`'s ~270 crates, stayed warm in `target/`). This is a **crate-level**
  cold-build number, not the whole-workspace GitHub Actions runner number (`Target / Status:
  unverified` per spec 12.4 -- that number needs an actual GitHub Actions run, which this task did
  not push).
- `cargo +1.85 check -p es-ros2 --features zenoh` if a 1.85 toolchain exists; otherwise `zenoh-rs.md`
  keeps the MSRV row unverified. **Recorded 2026-09-14:** a `1.85-x86_64-pc-windows-msvc` toolchain
  is installed locally; `cargo +1.85 check -p es-ros2 --features zenoh` PASSED. `zenoh-rs.md`'s MSRV
  row updated to VERIFIED.
- `.github/workflows/ci.yml` oracle job: install micromamba without sudo, create the pinned
  environment of design note section 8, run the live command above, fail when `RAN rmw_zenoh_interop` is missing.
  Implemented for `ros-base`/`rmw-zenoh-cpp`/`demo-nodes-cpp` only (what this packet's oracle
  actually runs); `cv-bridge`/`image-geometry` are W1c's and design note section 8 flags their
  co-installation as unverified, so this job does not risk it. The workflow file was not run
  through actual GitHub Actions in this task (no push); its steps were replicated by hand on the
  GPU server (same micromamba-installed package set, same live command) and passed there.
- `talker_capture.json` is committed, produced by `capture_reference_goldens` on a machine with
  `ES_ROS2_ENV`. **Recorded 2026-09-14:** produced on the GPU server (RoboStack Kilted) and
  committed at `tests/golden/ros2/rmw_zenoh/talker_capture.json`.
- `rmw-zenoh.md` / `zenoh-rs.md`: `ZenohId` string, XXH3 GID, 1.8.0 <-> RoboStack 1.7.2 wire
  compatibility, `ros2-env.sh` sufficiency and 1.8.0 API differences each get a dated result.
  **Recorded 2026-09-14, all VERIFIED** -- see both files' updated tables/rows.
  `ros2-env.sh` sufficiency: **sufficient once invoked under bash** (its own shebang is now
  `#!/usr/bin/env bash`, and `rmw_zenoh_interop.rs` calls `bash scripts/ros2-env.sh ...`
  explicitly); the pre-existing `tests/gen_goldens.rs` (W1a, outside this packet's file scope)
  still invokes it via `Command::new("sh")` and hits the documented `source: not found` gap if
  `ES_ROS2_ENV` happens to be set for a plain `cargo test -p es-ros2` run -- left for a follow-up.
- No new trait, no `HashMap`, ≤ ~900 new source lines. **Recorded 2026-09-14:** new source lines
  (tests excluded) total ~955 across `config.rs` (231), `session.rs` (533), `actuator.rs` (82),
  `error.rs`'s additions (96) and `lib.rs`'s (13) -- about 6% over the `~900` target, kept because
  the overage is `actuator.rs`'s `reorder_joint_state` (design note section 4.5's inbound-state
  reordering, ~25 lines incl. doc) and the `Durability`/`liveliness_tokens`/`config()` additions
  above, all real design-note-scoped behavior rather than incidental scope creep. No new trait, no
  `HashMap` anywhere in the crate.

## forbidden

- W1a codec semantics (a bug fix needs a failing test first and a line in this file); `camera`
  (W1c); `hil` (W1d); `crates/es` (W1e); every other crate.
- Services, actions, TRANSIENT_LOCAL / zenoh-ext advanced pub/sub, mode C, TLS/QUIC transports.
- Any publish path to an actuator topic that does not take a `SafeAction`; a flag that skips the
  startup probe.
- Editing existing goldens; writing `tests/golden/ros2/rmw_zenoh/**` with `es_ros2` encoders.
