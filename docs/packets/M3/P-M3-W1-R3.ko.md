<!-- Korean translation of docs/packets/M3/P-M3-W1-R3.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-W1-R3 — `Hello`마다 새 HIL 세션

Spec: §25.1(인증된 데이터그램; 위조된 커맨드는 plane에 도달해서는 안 된다), §24.2(저지연 UDP 위의
HIL), INV-12. 디자인 노트 `docs/design/ros2-boundary.md` section 7.2와 7.7.
`docs/reviews/M3-W1.md`의 **S-2**를 닫는다.

## context (범위)

```
crates/es-ros2/src/hil/link.rs
crates/es-ros2/tests/hil_gate.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R3.md
```

## spec (사양)

`HilLink::bind`는 `session_id`를 한 번 계산하고(`link.rs:96-98`) `hello`는 그것을 다시 만들지
않는다(`:250-253`). `Hello`가 실제로 다시 만드는 것은 `last_seq`로, 해당 데이터그램의 `seq`로
되돌려진다. 이 둘이 합쳐져 replay 방어를 무너뜨린다: 녹음된 세션 — `Hello`와 그 뒤의 `Command`들 —
을 가진 쪽이 `Hello`를 replay하면 link가 이를 수락하고(같은 키, 같은 `session_id`, 요구대로
`hdr.session_id == 0`), `last_seq`가 뒤로 내려가며, 더 높은 `seq`를 가진 녹음된 `Command`들이 모두
새 것으로 수락되어 `HilCore::on_command`에 도달한다. 디자인 노트 7.7이 내건 목표, "위조된 커맨드는
plane에 도달해서는 안 된다"가 같은 호스트의 수동 녹음자에 대해 성립하지 않는다.

session id는 비밀이 아니고 그럴 필요도 없다. 필요한 것은 *새로움*이며, 그래야 이전 세션에서 인증된
데이터그램이 `:197`의 `hdr.session_id != self.session_id` 검사에서 거부된다. 수락된 `Hello`마다
다시 생성한다.

- `session_id` 계산을 `bind`에서 빼내 private `fn next_session_id(&self) -> u64`로 만든다. 이 함수는
  `stats::wall_ns()`, 로컬 포트, link별 단조 증가 카운터를 섞고, `0`(`Hello`의 예약값)이나 현재
  `session_id`를 절대 반환하지 않는다.
- `hello`(`:250`)은 `HelloAck`를 보내기 전에 새 id를 배정하므로 `HelloAck`가 그것을 실어 나르고,
  컨트롤러는 이미 하듯 그것을 채택한다(`hil_gate.rs:245`, `Peer::recv`).
- `tx_seq`도 함께 리셋된다. `sent`(왕복 슬롯)는 `u64::MAX`로 비워, 경계를 넘어 남은 슬롯이 엉뚱한
  RTT 샘플을 만들지 못하게 한다.
- `bind`는 필드가 초기화되지 않는 일이 없도록 초기 id를 계속 설정한다. 그 외에는 아무것도 움직이지
  않는다.

카운터는 link별이며 구조체 안에 산다. 전역 상태 없음, RNG 없음(전역 RNG 금지, §3.4), 할당 없음,
그리고 id는 모든 Safety Plane 입력과 `.eshil` 로그 밖에 있으므로 게이트의 byte-identical replay는
영향을 받지 않는다.

디자인 노트 7.2의 `session_id` 행에 한 줄을 추가한다: "수락된 `Hello`마다 재생성; `0`은 `Hello`
자신에서만".

## oracle (검증)

```
cargo test -p es-ros2 --test hil_gate -- --nocapture
cargo fmt --check
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
```

`tests/hil_gate.rs`에서 `bad_tag_wrong_session_and_stale_seq_never_reach_the_plane` 옆에 두는 새
테스트:

- `a_replayed_session_does_not_reach_the_plane` — `Peer`가 handshake하고 `small_command`를 한 번
  보내며, 테스트는 그 `Hello`와 `Command`의 **raw 바이트를 보관**한다. tick이 돌고 커맨드가 집계된
  뒤, 녹음된 `Hello` 바이트를 그대로 다시 보낸다: link의 `session_id`가 바뀌었고(새 `HelloAck`가
  다른 값을 싣는다) 그 뒤 녹음된 `Command` 바이트를 다시 보내도 `stats().commands`가 변하지 않으며
  `rx_invalid`가 증가함을 assert한다. 수정 전 **FAIL**: 지금은 녹음된 커맨드가 두 번째로도
  수락된다.
- `a_second_hello_starts_a_clean_session` — replay된 `Hello` 이후, *새* 세션에서 새로 프레이밍한
  `Command`가 수락되므로 수정이 재접속을 깨지 않는다.

게이트 테스트 `hil_live_run_replays_to_byte_identical_decisions`는 컨트롤러 자신의 재접속 경로에
대한 회귀 가드이며 모든 non-vacuity assertion과 함께 계속 통과해야 한다.

## acceptance (수용 기준)

- 두 테스트가 통과하고, 첫 번째가 수정 전 `hello`에서 실패함이 확인된다.
- 기존 `hil_gate` 테스트 11개가 변경 없이 통과하고, `RAN hil_gate steps=2000 …`이 여전히 출력되며,
  `tests/fixtures/hil/v1_small.eshil`이 동일하게 replay된다 — 로그 포맷은 session id를 싣지 않으므로
  fixture는 손대지 않는다(그리고 재생성해서도 안 된다).
- `HilLink`의 공개 시그니처 변경 없음; `session_id`는 private으로 유지된다.
- `Instant`/`SystemTime`은 `link.rs`/`stats.rs`에만 머물고 `hil_imports_no_ros_module`이 통과한다.
- 수락 경로에 할당 없음, `HashMap` 없음, 새 trait 없음, `unsafe` 없음.

## forbidden (금지)

- `crates/es-safety`, `crates/es-runtime-embedded`, `crates/es-telemetry`, `crates/es-ir`.
- `src/hil/wire.rs`(wire 레이아웃, tag, `MAX_DATAGRAM`은 올바르며 그대로), `log.rs`, `replay.rs`,
  `core.rs`, `wall_ns` 읽기를 넘는 `stats.rs`.
- ROS 모듈들, `crates/es`, `xtask`, `.github/workflows/ci.yml`.
- 암호화, nonce window, 키 교체 방식, 두 번째 키 추가(디자인 노트 10.6은 사람의 판단 사항이지 이
  패킷이 아니다).
- `tests/fixtures/hil/v1_small.eshil` 재생성. 그 밖의 모든 M3 W1 발견 사항.
