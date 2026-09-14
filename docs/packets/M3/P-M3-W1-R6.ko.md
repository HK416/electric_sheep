<!-- Korean translation of docs/packets/M3/P-M3-W1-R6.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-W1-R6 — 빈 `.eshil` 로그는 verify되지 않는다

Spec: §24.2(HIL 게이트는 "live == replay"이다), §25.3(버전이 있는 로그 포맷), §1.4(아무것도 없는
입력에서 통과할 수 있는 검사는 검사가 아니다).
`docs/reviews/M3-W1.md`의 **S-6**을 닫는다.

## context (범위)

```
crates/es-ros2/src/hil/replay.rs
crates/es-ros2/tests/hil_gate.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R6.md
```

## spec (사양)

`replay`(`replay.rs:124-133`)는 `identical: true`로 초기화된 `ReplayReport`를 만들고, 비교할 것이
아무것도 없을 때 이를 거짓으로 만들지 않는다. 그리고 두 해시는 모두 빈 입력의 blake3다. 따라서 유효한
헤더와 trailer뿐인 로그 — 첫 tick 전에 죽은 run이거나, 누군가 헤더까지 자르고 trailer를 다시 붙인
파일 — 이 `identical: true, live_hash == replay_hash, truncated: false`로 돌아온다. 게이트가
보고하는 모든 속성이 decision이 하나도 없는 로그로 충족된다.

게이트 테스트 자체에는 잘못이 없다: `hil_gate.rs:403-418`이 `commands > 0`,
`steps == GATE_TICKS`, 그리고 다섯 가지 plane 이벤트를 손으로 assert하므로 M3 게이트는
non-vacuous하다. 약한 곳은 *다른* 모든 호출자가 읽게 될 타입이다 — W1e의 `es hil replay`, 그리고
`ReplayReport`를 인용할 evidence 번들 — 이들은 빈 파일에 대해 깨끗한 검증을 보고하게 된다.
non-vacuity를 `replay` 자신 안으로 옮겨 호출자가 잊을 수 없게 한다.

- `steps == 0`일 때 `ReplayReport::identical`은 `false`다. 필드에 문서화한다: "`Decision` 레코드가
  없는 로그는 아무것도 검증하지 않으므로 결코 `identical`이 아니다."
- `ReplayReport::is_verified(&self) -> bool` = `self.identical && !self.truncated &&
  self.steps > 0 && self.live_hash == self.replay_hash`를 추가한다 — §24.2의 게이트가 뜻하는 단일
  술어이므로, 호출자가 네 필드에서 다시 조립하다 하나를 빠뜨리는 일이 없게 한다.
- `v1_fixture_still_replays_identically`와 `hil_live_run_replays_to_byte_identical_decisions`가 이미
  assert하는 것에 더해 `is_verified()`를 assert한다(필드 단위 assertion은 유지한다: *어느* 속성이
  실패했는지 말해 주는 것이 그것들이다).
- `truncated_log_replays_its_complete_prefix`는 `prefix.identical`과 `prefix.steps > 0`을 계속
  assert하고 `assert!(!prefix.is_verified())`를 추가한다 — 잘린 prefix는 불완전한 run의 올바른
  replay이지 verify된 run이 아니다.

`replay`의 시그니처, `.eshil` v1 포맷, trailer, decision 해시, 로그 레코드는 모두 그대로다;
`tests/fixtures/hil/v1_small.eshil`은 재생성하지 않는다. 디자인 노트 section 7.5에 이 술어를 적는
한 줄이 추가된다.

## oracle (검증)

```
cargo test -p es-ros2 --test hil_gate -- --nocapture
cargo fmt --check
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
```

`tests/hil_gate.rs`의 새 테스트:

- `a_log_with_no_decisions_does_not_verify` — `fixture_bytes()`에서 헤더만 남기고, 해시가 빈 입력의
  blake3이고 `steps`가 0인 trailer만 덧붙인다. `replay::<NJ, H>`가 `Ok`를 반환하고 `steps == 0`,
  `live_hash == replay_hash`(둘 다 빈 해시다 — 그것이 요점이다)이며 `identical`이 `false`,
  `is_verified()`가 `false`다. 수정 전 `identical`에서 **FAIL**.
- `a_header_only_log_does_not_verify` — 헤더만, trailer 없음: `truncated`, `steps == 0`,
  `!is_verified()`.
- `the_gate_predicate_is_the_gate` — v1 fixture는 `is_verified()`이고,
  `tampered_step_diverges_at_its_tick`의 변조된 로그는 아니며, 잘린 prefix도 아니다.

## acceptance (수용 기준)

- 세 테스트가 통과하고, 첫 번째가 수정 전 `replay`에서 실패함이 확인된다.
- 기존 `hil_gate` 테스트 11개가 모두 통과하고, `RAN hil_gate steps=2000 …`이 여전히 출력되며,
  fixture가 계속 replay된다(`identical`, `live_hash == replay_hash`, `steps == 120`).
- `ReplayReport`에 메서드 하나가 추가되고 필드는 늘지 않는다; `Copy`, `PartialEq`, `Eq`를 유지한다.
- `tests/fixtures/hil/v1_small.eshil`이 체크인된 것과 byte 단위로 동일하다.
- `HashMap` 없음, 새 trait 없음, `unsafe` 없음, replay hot path에 할당 없음.

## forbidden (금지)

- `crates/es-safety`, `crates/es-runtime-embedded`, `crates/es-telemetry`, `crates/es-ir`.
- `src/hil/wire.rs`, `link.rs`, `core.rs`, `stats.rs`; doc 주석을 넘는 `log.rs`(레코드 레이아웃,
  trailer, `LogWriter`는 올바르며 그대로 둔다).
- ROS 모듈들, `crates/es`(`es hil replay`는 W1e), `xtask`, `.github/workflows/ci.yml`.
- `tests/fixtures/hil/v1_small.eshil` 재생성, `.eshil` 버전 바이트 변경.
- 빈 로그에 대해 `replay`가 `Err`를 반환하게 만드는 것: decision이 없는 well-formed 로그는 아무것도
  검증하지 않는 유효한 파일이며, 그 사실을 말해야 하는 것은 report다.
- 그 밖의 모든 M3 W1 발견 사항.
