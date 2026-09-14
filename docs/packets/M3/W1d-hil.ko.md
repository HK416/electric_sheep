<!-- Korean translation of docs/packets/M3/W1d-hil.md. The English file is the working copy; regenerate this when it changes. -->

# M3 W1d — HIL: UDP 링크, deadline/jitter 텔레메트리, 입력 로그, 바이트 단위로 동일한 replay (M3 게이트)

Design note: `docs/design/ros2-boundary.md` 섹션 7(핵심은 7.3이다: 로그는 binning 이후의
Safety Plane 입력을 기록하므로, live 실행은 그렇지 않더라도 replay는 결정론적이다). crate
골격에 대해서만 W1a에 의존; ROS 모듈은 쓰지 않는다.

## context (범위)

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

참고: `src/hil/`은 `mod.rs`, `wire.rs`, `core.rs`, `link.rs`, `log.rs`, `replay.rs`, `stats.rs`다.
`lib.rs`는 모듈과 re-export만 얻는다. `Cargo.toml`은 `es-ir`, `es-safety`,
`es-runtime-embedded`(`default-features = false`), `es-telemetry`, `blake3`, `serde_json`을
추가한다. `.gitignore`는 `*.eshilkey`만 얻는다. 설계 노트 섹션 7은 구현이 레이아웃 변경을
강제할 때만 바뀐다; 그럴 경우 이 파일에 이유를 적는다.

## spec (사양)

- §24.2: 저지연 UDP를 통해 실제 컨트롤러를 시뮬레이션에 연결한다; deadline miss와 jitter를
  텔레메트리로 노출한다; tier-1 결정성은 없으며, 대신 사후 replay를 위해 입력 로그를
  기록한다; "verifying that the Safety Plane behaves identically in HIL and on real hardware is
  the M3 gate."
- §9.5: 시뮬레이션과 하드웨어에서 같은 Safety Plane 코드 -- `HilCore`는
  `es_runtime_embedded::core_rt::EmbeddedCore::step`을 구동하며, 루프를 재구현하지 않는다.
- 부록 B.4, INV-13: `validate`는 `Result`를 반환하지 않는다; INV-12: 우회 없음, plant로
  넘겨지는 유일한 값은 `SafeAction`이다.
- §3.4, §3.5: wall clock은 절대 결정 경로에 들어가지 않는다; `HashMap` 없음.
- §25.1: 기본 localhost, 인증된 datagram, 유계 파싱.
- §25.3: 로그 포맷은 버전이 있다; v1 fixture는 계속 replay되어야 한다.
- §28.7 게이트 14는 열린 채로 남는다: 이 패킷은 호스트 쪽 절반을 증명한다(설계 노트 섹션 7.5).
- §4.2: `es-hil`은 `LAYERS`에 없으므로 모듈이다(설계 노트 섹션 2).

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-ros2 --all-targets -- -D warnings
cargo test -p es-ros2 --test hil_gate -- --nocapture
cargo test -p es-ros2 --lib hil
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

외부 reference는 없다: 오라클은 §24.2의 속성(live == replay)과 손으로 조립한 wire 레이아웃이다.
게이트 테스트는 `RAN hil_gate steps=<n> clamped=<n> fallback=<n> deadline_miss=<n>`를 출력한다.

`tests/hil_gate.rs`:

- `hil_live_run_replays_to_byte_identical_decisions` -- **게이트.** `127.0.0.1:0`의
  `HilLink<3, 8>`, 1 kHz 제어, 각 틱의 deadline에 맞춰 `thread::sleep`으로 페이싱한 2,000틱;
  `DeploymentIr` fixture(inference budget, heartbeat timeout, violation-rate watchdog,
  `HoldPosition`)로부터 만든 plane. loopback 너머의 컨트롤러 스레드가 `Hello`를 보낸 다음,
  고정 시드 xorshift에서 나온 0–3ms 지연 후 각 `State`에 `Command`(주기적으로 velocity
  한계를 넘는 궤적)로 응답한다; datagram의 5%를 버리고, 한 쌍을 바꿔치기하고, command 하나를
  40틱 늦게 보내고, 한 row에 NaN을 넣고, heartbeat를 한 번 timeout보다 길게 멈춘다. Plant:
  `q := action.q`. 그런 다음 `replay::<3, 8>(&log)`: `identical`, `live_hash == replay_hash`,
  `truncated` 아님; 그리고 최소 하나의 `Clamped`, 하나의 `Fallback`, 하나의 `NonFinite`
  이벤트, 하나의 `HeartbeatLoss`, `deadline_miss ≥ 1`.
- `v1_fixture_still_replays_identically` -- `tests/fixtures/hil/v1_small.eshil`, 이 패킷의
  live 실행으로 한 번 생성되어 **format fixture**로 커밋됨(golden이 아니다: 이를 만드는
  reference가 없다).
- `tampered_step_diverges_at_its_tick` -- 하나의 `Step`의 action 비트 하나를 뒤집음:
  `!identical`, `first_divergence`가 그 틱을 지목.
- `truncated_log_replays_its_complete_prefix` -- 레코드 중간에서 자름: `truncated`, prefix는
  동일.
- `foreign_deployment_hash_is_rejected`(`HilLogError::DeploymentHash`).
- `shape_mismatch_is_rejected` -- `NJ = 3` 로그에 대한 `replay::<4, 8>`(`HilLogError::Shape`).
- `bad_tag_wrong_session_and_stale_seq_never_reach_the_plane` -- `rx_invalid` / `rx_stale`로
  카운트됨; plane의 카운터와 로그는 바뀌지 않음.
- `hello_with_a_wrong_deployment_hash_gets_bye` -- `Bye { reason: 1 }`.
- `late_command_counts_a_deadline_miss_and_still_reaches_the_plane` -- 소켓 없이: 명시적 틱과
  함께 이벤트가 `HilCore`에 주입됨.
- `stats_frames_reach_a_telemetry_client` -- `es_telemetry` loopback server; `HIL_STATS_STREAM`을
  구독한 client가 설계 노트의 필드 순서로 `Payload::Scalars`를 받으며 `deadline_miss ≥ 1`.
- `hil_imports_no_ros_module` -- `src/hil/*.rs`를 읽는다; `cdr`, `msg`, `names`, `attachment`,
  `session`, `config`, `actuator`, `camera`로 향하는 `crate::` 경로가 없음.

`wire.rs`의 단위 테스트: 모든 메시지 종류가 왕복한다; header offset은 손으로 조립한
바이트에 대해 설계 노트 섹션 7.2와 일치한다; `rows > H`, datagram과 다른 `body_len`,
초과 크기 datagram은 거부된다; proptest: 임의 바이트를 디코딩해도 절대 panic하지 않는다.

## acceptance (수용 기준)

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

- Wire 레이아웃, tick binning, 로그 레이아웃, stats 순서는 설계 노트 섹션 7.2–7.6 그대로.
  실행 모드는 `DeploymentIr::execution`에서 온다; 컨트롤러는 그것을 선택할 수 없다.
- `Instant` / `SystemTime`은 `link.rs`와 `stats.rs`에만 나타난다.
- `HilLink`/`HilCore`는 검증되지 않은 채로 받은 command를 반환하는 accessor를 노출하지
  않는다.
- 테스트가 출력하는 모든 타이밍 수치는 관측값으로 표시된다; 보고서는 `Target / Status:
  unverified`를 쓴다.
- 새 trait 없음(`Box<dyn Write>`는 `std`의 것), `HashMap` 없음, `unsafe` 없음, 소스 코드
  ~1,300줄 이하.

## forbidden (금지)

- `crates/es-safety`, `crates/es-runtime-embedded`, `crates/es-telemetry`, `crates/es-ir`의
  소스: 있는 그대로 소비한다. `EmbeddedCore`에 `HilCore`가 필요로 하는 것이 없으면, 멈추고
  보고한다.
- ROS 모듈들, ROS 2 HIL 프론트엔드, zenoh feature; `crates/es`(`es hil replay`는 W1e);
  `xtask`, `.github/workflows/ci.yml`.
- Safety Plane 입력으로서의 wall-clock 값; plane 안의 "HIL mode" 플래그(INV-12); plant를
  향해 검증되지 않은 command를 반환하는 경로.
- 물리 컨트롤러, 실제 네트워크, RT 스케줄링, 게이트 14에 대한 주장.
