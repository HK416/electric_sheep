<!-- Korean translation of docs/packets/M3/P-M3-W1-R1.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-W1-R1 — violation-rate window는 window여야 한다

Spec: §9.4(`EnvelopeViolationRate { window, max_frac }`), §10.3(`envelope_violation_rate`, 그 자신의
예시 acceptance가 `<= 0.01`이다), §9.3(clamp vs fallback), Appendix B.4, INV-12.
`docs/reviews/M3-W1.md`의 블로커 **B-1**을 닫는다.

## context (범위)

```
crates/es-safety/src/counters.rs
crates/es-safety/src/plane.rs
crates/es-safety/tests/properties.rs
docs/design/safety-plane.md
docs/design/safety-plane.ko.md
docs/packets/M3/P-M3-W1-R1.md
```

## spec (사양)

`ViolationWindow::fraction()`(`counters.rs:59-64`)은 `ones`를 설정된 `window`가 아니라 지금까지
관측된 스텝 수인 `filled`로 나눈다. 사양대로 생긴 설정에서 모두 도달 가능한 결과는 다음과 같다.

- run의 첫 dirty 스텝이 rate를 정확히 `1.0`으로 만들고, 그래서 `plane.rs:268-273`이 **어떤**
  `max_frac < 1.0`에 대해서도 다음 스텝에서 `ViolationKind::ViolationRate`를 올린다.
- 그 trip은 fallback을 실행하고(`plane.rs:276-284`), fallback 스텝 자체가 dirty이므로
  (`finish` → `window.push(!events.is_empty())`, `plane.rs:460`) rate는 `1.0`에 머물고 plane은
  fallback을 벗어나지 못한다. `FallbackPolicy::EmergencyStop`이면 run의 두 번째 스텝에서 e-stop이
  latch된다(`plane.rs:282`).
- HIL cold start가 가장 빠른 진입 경로이지만(첫 tick에 chunk가 없어 `ChunkUnderrun`이 발생), HIL
  고유의 문제가 아니다: 어떤 deployment든 3번째 스텝의 일시적 clamp 하나면 같은 일이 벌어진다.

못 박아야 할 규칙이자 §10.3의 `<= 0.01`이 자연스럽게 읽히는 규칙: **rate는 마지막 `window` 스텝에
대한 것이며, `window` 스텝이 존재하기 전에는 watchdog이 trip하지 않는다.** 그 전의 window는 무엇의
표본도 아니다.

수정은 `counters.rs`에만 들어간다.

- `ViolationWindow::fraction()`은 `self.filled == self.len`이 된 뒤에만
  `self.ones as f64 / self.len as f64`를 반환하고, 그 전에는 `0.0`을 반환한다(이미 `filled == 0`에
  대해 `0.0`을 반환한다).
- `SafetyCounters::envelope_violation_rate()`는 시그니처와 §10.3의 의미를 유지한다. window가 찰
  때까지 `0.0`으로 읽힌다는 것, 그리고 에피소드 전체에 대해 rate를 계산하는 `es-eval`의 metric은
  이 ring과 무관하므로 영향을 받지 않는다는 것을 doc 주석에 적는다.

`plane.rs`의 watchdog 블록은 그대로다: 여전히 *이전* 스텝 기준으로 rate를 읽으며, 그것이 규칙을
비순환적으로 유지한다. 시그니처 변경 없음, 새 필드 없음, 할당 없음.

`docs/design/safety-plane.md`의 watchdog 절에 규칙을 기록한다: 부분적으로 채워진 window는 절대
trip하지 않으며, 그 이유(그렇지 않으면 이른 시점의 violation 하나가 100 % rate가 된다)를 적는다.

## oracle (검증)

```
cargo test -p es-safety
cargo test -p es-ros2 --test hil_gate
cargo fmt --check
cargo clippy -p es-safety --all-targets -- -D warnings
cargo xtask layering
```

수정 **전에** 작성하는 `crates/es-safety/tests/properties.rs`의 새 테스트:

- `a_partial_window_never_trips_the_rate_watchdog` — `EnvelopeViolationRate { window: 8,
  max_frac: 0.25 }`, `FallbackPolicy::HoldPosition`. 스텝 1은 `ActionChunk::empty(..)`를 제출하고
  (`ChunkUnderrun`, 정확히 HIL cold start), 스텝 2-8은 envelope 안의 깨끗한 chunk를 제출한다. 어떤
  스텝도 `ViolationKind::ViolationRate`를 올리지 않고 스텝 2부터 모두 `ActionSource::Policy`임을
  assert한다. 수정 전 **FAIL**(스텝 2부터 `Fallback`).
- `a_full_window_trips_at_the_threshold` — 같은 plane에서 깨끗한 스텝 8개를 돌린 뒤, 속도 한계를
  넘는 action을 가진 스텝 3개(3/8 = 0.375 > 0.25)를 돌리고, 세 번째 다음 스텝에서 `ViolationRate`가
  나타나며 그 전에는 나타나지 않음을 assert한다.
- `the_rate_falls_back_out_of_the_window` — 그 run을 깨끗한 chunk로 이어가 dirty 비트 3개가 밖으로
  밀려나면 `ViolationRate`가 멈추고 source가 `Policy`로 돌아옴을 assert한다. 즉 latch가 영구적이
  아니라 window의 함수임을 보인다.
- `an_estop_rate_watchdog_does_not_latch_on_step_two` — 같은 설정에 `FallbackPolicy::EmergencyStop`:
  이른 `ChunkUnderrun` 한 번 이후에도 `estop_latched`가 여전히 false다.

## acceptance (수용 기준)

- 네 테스트가 통과하고, 첫 번째와 마지막이 수정 전 `fraction()`에서 실패함이 확인된다.
- 기존 `es-safety` 스위트(`properties.rs:230`의 config 거부 테스트와
  `determinism_two_planes_same_inputs_same_outputs` 포함)가 변경 없이 통과한다.
- `es-runtime-embedded`와 `es-eval` 테스트 스위트가 변경 없이 통과한다.
- `cargo test -p es-ros2 --test hil_gate`가 fixture의 `max_frac: 1.0` 그대로 통과한다. 이 패킷은
  fixture를 **바꾸지 않는다**(게이트가 무엇을 arm해야 하는지는 별개의 판단이다).
- 공개 시그니처 변경 없음. `SafetyCounters`에 필드가 늘지 않고, 아무것도 할당하지 않는다.
- INV-12/INV-13 유지: 이 변경은 watchdog이 *덜* trip하게만 할 수 있고 clamp 단계를 끌 수 없으며,
  `validate`는 여전히 `Result`를 반환하지 않는다.

## forbidden (금지)

- `crates/es-ros2`(HIL fixture의 `max_frac`과 `hil_gate.rs`의 주석은 게이트가 무엇을 arm할지
  결정하는 쪽의 몫이다), `crates/es-ir`(`Watchdog` 스키마는 그대로), `crates/es-eval`
  (`metrics.rs:44`는 다른 출처에서 에피소드 단위 metric을 계산하며 그대로 둔다),
  `crates/es-runtime-embedded`.
- `WINDOW_CAP`, ring의 레이아웃, `window.push`의 "모든 이벤트는 dirty" 규칙 변경.
- "warm-up"이나 "grace period" 설정 필드 추가: window 길이가 이미 그것이다.
- 그 밖의 모든 M3 W1 발견 사항.
