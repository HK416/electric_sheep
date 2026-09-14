# M3 W1d — HIL: UDP link, deadline/jitter telemetry, input log, byte-identical replay (M3 gate)

Design note: `docs/design/ros2-boundary.md` section 7 (7.3 is the core: the log records post-binning
Safety Plane inputs, so replay is deterministic although the live run is not). Depends on W1a only
for the crate skeleton; no ROS module is used.

## context

```
crates/es-ros2/src/hil/*.rs
crates/es-ros2/src/lib.rs
crates/es-ros2/Cargo.toml
crates/es-ros2/tests/hil_gate.rs
tests/fixtures/hil/**
.gitignore
Cargo.lock
docs/design/ros2-boundary.md
docs/packets/M3/W1d-hil.md
```

Notes: `src/hil/` is `mod.rs`, `wire.rs`, `core.rs`, `link.rs`, `log.rs`, `replay.rs`, `stats.rs`.
`lib.rs` gets the module and re-exports only. `Cargo.toml` adds `es-ir`, `es-safety`,
`es-runtime-embedded` (`default-features = false`), `es-telemetry`, `blake3`, `serde_json`.
`.gitignore` gets `*.eshilkey` only. Design note section 7 changes only if the implementation
forces a layout change; say why in this file.

## spec

- §24.2: connect a real controller to the simulation over low-latency UDP; expose deadline misses
  and jitter via telemetry; no tier-1 determinism, instead record input logs for post-hoc replay;
  "verifying that the Safety Plane behaves identically in HIL and on real hardware is the M3 gate."
- §9.5: the same Safety Plane code in simulation and on hardware -- `HilCore` drives
  `es_runtime_embedded::core_rt::EmbeddedCore::step`, it does not reimplement the loop.
- Appendix B.4, INV-13: `validate` returns no `Result`; INV-12: no bypass, and the only value
  handed toward the plant is a `SafeAction`.
- §3.4, §3.5: wall clock never enters the decision path; no `HashMap`.
- §25.1: localhost by default, authenticated datagrams, bounded parsing.
- §25.3: the log format is versioned; the v1 fixture must keep replaying.
- §28.7 gate 14 stays open: this packet proves the host-side half (design note section 7.5).
- §4.2: `es-hil` is not in `LAYERS`, hence a module (design note section 2).

## oracle

```
cargo fmt --check
cargo clippy -p es-ros2 --all-targets -- -D warnings
cargo test -p es-ros2 --test hil_gate -- --nocapture
cargo test -p es-ros2 --lib hil
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

There is no external reference: the oracle is §24.2's property (live == replay) plus hand-assembled
wire layouts. The gate test prints `RAN hil_gate steps=<n> clamped=<n> fallback=<n> deadline_miss=<n>`.

`tests/hil_gate.rs`:

- `hil_live_run_replays_to_byte_identical_decisions` -- **the gate.** `HilLink<3, 8>` on
  `127.0.0.1:0`, 1 kHz control, 2,000 ticks paced with `thread::sleep` to each tick's deadline; plane
  from a `DeploymentIr` fixture (inference budget, heartbeat timeout, violation-rate watchdog,
  `HoldPosition`). A controller thread over loopback sends `Hello`, then answers each `State` with a
  `Command` (a trajectory that periodically exceeds the velocity limit) after a 0–3 ms delay from a
  fixed-seed xorshift; it drops 5 % of datagrams, swaps one pair, sends one command 40 ticks late,
  puts a NaN in one row, and pauses heartbeats once for longer than the timeout. Plant: `q := action.q`.
  Then `replay::<3, 8>(&log)`: `identical`, `live_hash == replay_hash`, not `truncated`; and at least
  one `Clamped`, one `Fallback`, one `NonFinite` event, one `HeartbeatLoss`, `deadline_miss ≥ 1`.
- `v1_fixture_still_replays_identically` -- `tests/fixtures/hil/v1_small.eshil`, produced once by
  this packet's live run and committed as a **format fixture** (not a golden: no reference produces it).
- `tampered_step_diverges_at_its_tick` -- flip one action bit of one `Step`: `!identical`,
  `first_divergence` names that tick.
- `truncated_log_replays_its_complete_prefix` -- cut mid-record: `truncated`, prefix identical.
- `foreign_deployment_hash_is_rejected` (`HilLogError::DeploymentHash`).
- `shape_mismatch_is_rejected` -- `replay::<4, 8>` on an `NJ = 3` log (`HilLogError::Shape`).
- `bad_tag_wrong_session_and_stale_seq_never_reach_the_plane` -- counted in `rx_invalid` /
  `rx_stale`; the plane's counters and the log are unchanged.
- `hello_with_a_wrong_deployment_hash_gets_bye` -- `Bye { reason: 1 }`.
- `late_command_counts_a_deadline_miss_and_still_reaches_the_plane` -- socket-free: events injected
  into `HilCore` with explicit ticks.
- `stats_frames_reach_a_telemetry_client` -- `es_telemetry` loopback server; a client subscribed to
  `HIL_STATS_STREAM` receives `Payload::Scalars` in the design-note field order with `deadline_miss ≥ 1`.
- `hil_imports_no_ros_module` -- reads `src/hil/*.rs`; no `crate::` path into `cdr`, `msg`, `names`,
  `attachment`, `session`, `config`, `actuator` or `camera`.

Unit tests in `wire.rs`: every message kind round-trips; header offsets match design note section
7.2 on hand-assembled bytes; `rows > H`, `body_len` disagreeing with the datagram, and an oversize
datagram are rejected; proptest: decoding arbitrary bytes never panics.

## acceptance

```rust
pub struct HilConfig { pub bind: SocketAddr, pub key: [u8; 32], pub stats_every: u64, pub max_datagram: usize }
pub struct Command<const NJ: usize, const H: usize> { pub obs_tick: PhysTick, pub rows: u16, pub actions: [[f64; NJ]; H] }
pub struct HilCore<const NJ: usize, const H: usize> { /* EmbeddedCore, log writer, stats */ }
impl<const NJ: usize, const H: usize> HilCore<NJ, H> {
    pub fn from_ir(ir: &DeploymentIr, log: Box<dyn std::io::Write + Send>) -> Result<Self, HilError>;
    pub fn observe_state(&mut self, now: PhysTick, q: &[f64; NJ], qd: &[f64; NJ]);
    pub fn on_heartbeat(&mut self);
    pub fn on_command(&mut self, cmd: Command<NJ, H>);
    pub fn tick(&mut self, now: PhysTick) -> SafeAction<NJ>;
    pub fn stats(&self) -> HilStats;
    pub fn finish(self) -> Result<[u8; 32], HilError>;         // trailer; returns the decisions hash
}
pub struct HilLink<const NJ: usize, const H: usize> { /* UdpSocket, HilCore */ }
impl<const NJ: usize, const H: usize> HilLink<NJ, H> {
    pub fn bind(cfg: &HilConfig, core: HilCore<NJ, H>) -> std::io::Result<Self>;
    pub fn local_addr(&self) -> SocketAddr;
    pub fn tick(&mut self, now: PhysTick, q: &[f64; NJ], qd: &[f64; NJ]) -> SafeAction<NJ>;
    pub fn publish_stats(&self, server: &es_telemetry::Server, now: PhysTick);
    pub fn finish(self) -> Result<[u8; 32], HilError>;
}
pub fn replay<const NJ: usize, const H: usize>(log: &[u8]) -> Result<ReplayReport, HilLogError>;
pub struct ReplayReport { pub steps: u64, pub identical: bool, pub first_divergence: Option<(u64, PhysTick)>,
                          pub live_hash: [u8; 32], pub replay_hash: [u8; 32], pub truncated: bool }
pub const HIL_STATS_STREAM: StreamId = StreamId(0x4849_4C31);
```

- Wire layout, tick binning, log layout and stats order exactly as design note sections 7.2–7.6.
  The execution mode comes from `DeploymentIr::execution`; the controller cannot choose it.
- `Instant` / `SystemTime` appear only in `link.rs` and `stats.rs`.
- `HilLink`/`HilCore` expose no accessor that returns a received command un-validated.
- Any timing number the tests print is labelled an observation; reports use `Target / Status: unverified`.
- No new trait (`Box<dyn Write>` is `std`'s), no `HashMap`, no `unsafe`, ≤ ~1,300 source lines.

## forbidden

- `crates/es-safety`, `crates/es-runtime-embedded`, `crates/es-telemetry`, `crates/es-ir` sources:
  consume them as they are. If `EmbeddedCore` lacks something `HilCore` needs, stop and report.
- The ROS modules, a ROS 2 HIL front end, the zenoh feature; `crates/es` (`es hil replay` is W1e);
  `xtask`, `.github/workflows/ci.yml`.
- A wall-clock value as a Safety Plane input; a "HIL mode" flag in the plane (INV-12); a path that
  returns an unvalidated command toward the plant.
- Claims about a physical controller, a real network, RT scheduling or gate 14.
