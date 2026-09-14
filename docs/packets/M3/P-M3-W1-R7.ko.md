<!-- Korean translation of docs/packets/M3/P-M3-W1-R7.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-W1-R7 — rate watchdog는 자기 자신의 fallback을 측정하지 않는다

Spec: §9.4(`EnvelopeViolationRate`, `FallbackPolicy`), §18.5("Fallback은 실패가 아니라 정상
동작이다"), §10.3(`envelope_violation_rate`), §9.3, Appendix B.4, INV-12, INV-13.
디자인 노트 `docs/design/safety-plane.md`의 watchdog 표 7행과 "Known ceiling" 문단.
P-M3-W1-R1(`04e04b7`)이 발견했으나 건드리는 것이 금지되어 있던 2차 버그를 닫는다. 이 버그는
`crates/es-safety/tests/properties.rs::a_tripped_rate_watchdog_does_not_release_itself`가 고정하고
있다.

## context (범위)

```
crates/es-safety/src/plane.rs
crates/es-safety/src/types.rs
crates/es-safety/tests/properties.rs
docs/design/safety-plane.md
docs/design/safety-plane.ko.md
docs/packets/M3/P-M3-W1-R7.md
```

## spec (사양)

R1은 분모를 고쳤지만, 분자는 한 차수 뒤에서 여전히 순환한다. 일단 `ViolationKind::ViolationRate`가
trip하면 step 4a(`plane.rs:276-284`)가 이후 **모든** 스텝에서 fallback을 실행하고, `finish`는 그
각각을 `window.push(!events.is_empty())`로 기록한다(`plane.rs:460`). 그 스텝들의 이벤트는 watchdog
자신의 `ViolationRate`이므로, window는 watchdog의 메아리로 다시 채워지고 `ones`는 결코 줄지 않으며
trip이 영구화된다.

**§9.4의 어느 해석을 따를 것인가.** 세 가지가 이 사양이 latch가 아니라 복구를 의도함을 말한다.

1. §18.5가 명시적이다: "fallback이 발동한 env는 `Quarantined`가 아니라 `Ok`로 남는다 …
   **Fallback은 실패가 아니라 정상 동작이다.**" 끝날 수 없는 fallback은 어떤 해석으로도 실패다.
2. §9.4의 `FallbackPolicy`에서 "latch"라는 단어는 정확히 한 번 등장한다 — `EmergencyStop //
   immediate stop + latch`. latch는 그 *정책* 하나의 속성이다. §9.4의 어디에서도 latch를
   *watchdog*에 붙이지 않으며, `EnvelopeViolationRate`는 순수하게 windowed rate로 정의된다.
3. plane은 이미 그 해석대로 구현되어 있다. `estop_latched`는 `FallbackKind::EmergencyStop`에만
   설정되고(`plane.rs:279-283`), `is_latched()` / `reset_latch()`(`plane.rs:152-159`)가 거기에
   한정된 명시적 operator reset 경로다. 지금의 동작은 그 *외의* 모든 정책에 사실상 영구 latch를
   주는데, 그것은 `is_latched()`가 보고하지도 않고 `reset_latch()`가 지울 수도 없다 — 두 해석 중
   최악인 보이지 않는 latch다.

`plane.rs:268-270`의 주석과 디자인 노트 7행은 이 규칙이 비순환적이어야 함을 이미 적고 있다
("이 스텝 자신의 clamp는 아직 기록되지 않았다"). 이것은 한 스텝 뒤의 같은 순환이다.

**변경 내용.** window는 그 스텝이 *rate watchdog 자신의 trip이 아닌 이유로* dirty했는지를 기록한다.

- `types.rs`: `EventSet`에 `pub const fn without(self, kind: ViolationKind) -> Self`를 추가한다 —
  기존 newtype 위의 `u32` 마스크 하나, 새 타입 없음, trait 없음, 할당 없음.
- `plane.rs:460`: `self.counters.window.push(!events.without(ViolationKind::ViolationRate).is_empty());`

그러면 해제가 자동이고 유계가 된다: 마지막 진짜 violation이 ring을 빠져나간 뒤 정확히 `window`
스텝 만에 trip이 풀린다. **어떤** 진짜 이유로든 dirty한 스텝 — clamp, `NanInf`, `ChunkUnderrun`,
`HeartbeatLoss`, `SensorDropout`, `StaleObservation`, `InferenceDeadline` — 은 계속 집계되므로 rate는
여전히 실제 violation을 측정하고, 진짜로 계속 violation을 내는 plane은 watchdog이 계속 trip된 채로
남는다.

**바뀌어서는 안 되는 것.** `dirty_steps`, `clamped_steps`, `fallback_activations`,
`violations[ViolationRate]`, `steps`는 실제로 일어난 일을 기록하며 watchdog 자신의 것을 포함해 모든
fallback 스텝을 계속 세야 한다: §10.3의 에피소드 단위 `envelope_violation_rate`는 `es-eval`에서 이
ring이 아니라 `dirty_steps`/`steps`로 계산되므로, 함께 바뀌면 실제 fallback을 과소 보고하게 된다.
바뀌는 것은 watchdog의 **입력 ring**뿐이다.

**INV-12 / INV-13.** 모든 clamp 단계와 다른 모든 watchdog은 그대로이며 매 스텝 실행된다. rate
watchdog은 다른 watchdog들의 출력에 대한 메타 watchdog이지 그 자체로 하나의 단계가 아니므로, 전에
통과할 수 없던 것이 이제 envelope을 통과하는 일은 없다. `validate`는 시그니처를 유지하고 여전히
`Result`를 반환하지 않는다. 설정 스위치 없음, "HIL 모드" 없음, 우회 없음.

디자인 노트: "Known ceiling, not fixed here" 문단을 규칙과 해제 범위로 교체하고, watchdog 표 7행에
자기 자신을 측정하지 않는다는 절을 추가한다.

## oracle (검증)

```
cargo test -p es-safety
cargo test -p es-ros2 --test hil_gate
cargo test -p es-runtime-embedded
cargo fmt --check
cargo clippy -p es-safety --all-targets -- -D warnings
cargo xtask layering
cargo xtask nostd
cargo xtask check-spec-refs
```

`crates/es-safety/tests/properties.rs`에서 R1의 `rate_plane` / `hold_step` / `dirty_step`
헬퍼를 재사용한다(`window: 8`, `max_frac: 0.25`).

- `a_tripped_rate_watchdog_releases_after_a_clean_window` — 이 패킷이 고치는 버그를 고정하고 있는
  `a_tripped_rate_watchdog_does_not_release_itself`를 **대체한다**. R7이 그 테스트를 삭제하고 같은
  자리에 R7을 명시한 주석과 함께 이 테스트를 넣는다. 깨끗한 스텝 8개, dirty 3개(3/8 = 0.375 >
  0.25)로 trip시킨 뒤 깨끗한 스텝을 이어가면, 마지막 진짜 violation으로부터 `window` 스텝 안에
  `ViolationRate`가 멈추고 `source`가 `ActionSource::Policy`로 돌아온다. 수정 전 **FAIL**(기존
  테스트가 40스텝까지 정반대를 assert한다).
- `a_real_violation_during_a_trip_still_counts` — 위처럼 trip시킨 뒤 진짜로 dirty한 스텝을 계속
  먹인다: 그것이 계속되는 동안 watchdog은 해제되지 **않는다**. 수정이 window를 실제 violation에
  대해 눈멀게 해서는 안 된다.
- `the_metric_counters_still_count_every_fallback` — trip-해제 사이클 한 번 동안
  `fallback_activations`, `dirty_steps`, `violations[ViolationRate]`가 실제로 실행된 fallback 스텝
  수와 같다. ring과 함께 카운터까지 "고치는" 것을 막는 가드다.
- `an_estop_rate_trip_still_latches` — `FallbackPolicy::EmergencyStop`에서, 가득 찬 window 위의
  진짜 rate trip은 여전히 `is_latched()`를 설정하고 `reset_latch()`만이 그것을 지운다. 진짜 latch
  하나는 그대로 남는다(INV-12).
- `types.rs`의 `EventSet::without` 단위 테스트: 없는 kind를 제거하면 항등이고, 유일하게 있는 kind를
  제거하면 `EMPTY`이며, 다른 kind들은 살아남는다.

## acceptance (수용 기준)

- `properties.rs`의 네 테스트와 `types.rs` 단위 테스트가 통과하고, 첫 번째가 수정 전에 실패함이
  확인되며, `a_tripped_rate_watchdog_does_not_release_itself`는 사라지고 대체 테스트의 주석에 R7이
  명시된다.
- R1의 나머지 다섯 테스트와 `es-safety` 스위트의 나머지가 변경 없이 통과한다
  (`a_partial_window_never_trips_the_rate_watchdog`,
  `determinism_two_planes_same_inputs_same_outputs` 포함).
- `es-runtime-embedded`와 `es-eval` 스위트가 변경 없이 통과하고 `cargo xtask nostd`가 clean이다.
- `cargo test -p es-ros2 --test hil_gate`가 fixture의 `max_frac: 1.0`을 그대로 둔 채 계속 통과한다.
- 공개 API는 정확히 메서드 하나(`EventSet::without`)만 늘어난다. `SafetyCounters`나 `PlaneState`에
  새 필드 없음, 할당 없음, `HashMap` 없음, 새 trait 없음(INV-17).
- 디자인 노트와 `.ko.md` 형제 문서가 같은 커밋에서 갱신되고 "Known ceiling" 문단이 사라진다.

## human override (사람의 번복 선택지)

§9.4가 대신 **trip된 rate watchdog은 명시적 operator reset까지 latch된다**로 판정되면, 이 패킷은
이렇게 바뀐다: `PlaneState`에 `rate_latched: bool`을 추가하고 trip 시 설정하며, `ViolationRate`가
window가 아니라 그 플래그에서 다시 발화하게 하고, 이를 노출(`is_rate_latched()`)하고
`reset_latch()`에서 지우며, 그 규칙을 §9.4와 디자인 노트에 적는다. 이 해석은 위에 인용한 §18.5와
충돌하고 작업량도 엄격히 더 많으므로 기본값이 아니라 override다 — 다만 *어느* 판정이든 현재 동작은
틀렸다는 점에 유의할 것. 지금 만들어지는 latch는 `is_latched()`에 보이지 않고 `reset_latch()`로
지울 수도 없기 때문이다.

## forbidden (금지)

- `crates/es-ros2`(이것이 반영되면 HIL fixture의 `max_frac: 1.0`을 다시 쓸 수 있게 되지만, 게이트가
  무엇을 arm할지 바꾸는 것은 별개의 판단이다), `crates/es-ir`(`Watchdog` 스키마는 그대로),
  `crates/es-eval`, `crates/es-runtime-embedded`.
- `record_step`, `dirty_steps`, `clamped_steps`, `fallback_activations`, `steps`, 또는 `es-eval`이
  읽는 것을 바꾸는 것.
- `ViolationWindow`의 레이아웃, `WINDOW_CAP`, `fraction()`, 그 밖에 R1(`04e04b7`)이 정리한 것을
  바꾸는 것.
- `estop_latched`, `is_latched`, `reset_latch`, `EmergencyStop` 분기를 건드리는 것.
- hysteresis, cool-down, release threshold, grace period 설정 필드 추가: 이 watchdog이 가진 유일한
  기간은 window 길이다.
- 그 밖의 모든 M3 W1 발견 사항.
