<!-- Korean translation of docs/packets/M1/W5-safety-plane.md. The English file is the working copy; regenerate this when it changes. -->

# W5 — Safety Plane 런타임 (`es-safety`)

Spec: §9 (Deployment IR과 Safety Plane), §9.3 (envelope), §9.4 (watchdog/fallback),
§9.5 (시뮬/실기 동일성), §8.5–8.6 (action chunk, chunk underrun), §10.3
(`envelope_violation_rate`, `chunk_underrun_rate`), §18.5 (fallback은 정상 동작이다),
부록 B.4 (고정된 형태), §28.7 게이트 8 (type B).

설계: `docs/design/safety-plane.md` — 먼저 읽을 것. 이 문서가 이 패킷이 구현할
clamp 순서, watchdog 순서, fallback 의미론을 고정한다.

## context (범위)

```
crates/es-safety/Cargo.toml
crates/es-safety/src/lib.rs
crates/es-safety/src/types.rs
crates/es-safety/src/config.rs
crates/es-safety/src/counters.rs
crates/es-safety/src/plane.rs
crates/es-safety/tests/scenarios.rs
crates/es-safety/tests/properties.rs
tests/fixtures/safety/*.json
docs/design/safety-plane.md
docs/packets/M1/W5-safety-plane.md
```

## spec (사양)

`es-safety` (layer 8)는 `es-ir`가 이미 검증하는 Deployment IR을 위한
**런타임**이다.

- 부록 B.4의 필드들을 갖는 `SafetyPlane<const NJ: usize, const H: usize>`:
  `envelope`, `watchdogs`, `fallback`, `state: SafetyState<NJ, H>` (미리
  할당됨), `counters: SafetyCounters`.
- `SafetyPlane::from_ir(&DeploymentIr) -> Result<Self, SafetyConfigError>`가
  유일한 생성자다: `robot.n_joints == action.dim == NJ`, `action.horizon == H`,
  모든 한계값이 유한함, 중복 watchdog 없음, `EnvelopeViolationRate.window <=
  256`. `new`도, `Default`도, public 필드도 없다 — envelope 없는 plane은
  구성 불가능해야 한다 (INV-12).
- `pub fn validate(&mut self, chunk: &ActionChunk<NJ, H>, obs_age: Micros, now: PhysTick)
  -> SafeAction<NJ>` — `Result` 없음 (INV-13), 패닉 없음, 항상 실행 가능한
  액션.
- `ActionChunk<NJ, H>` = `[[f64; NJ]; H]` + 유효 길이 + `ExecutionMode`.
  `SafeAction<NJ>` = `{ q: [f64; NJ], source: Policy | Clamped | Fallback(FallbackKind),
  events: EventSet }` (`ViolationKind`에 대한 `u32` 비트셋).
- Watchdog (§9.4): `ChunkUnderrun`, `NanInf` (무조건), `StaleObservation`,
  `InferenceDeadline`, `ControllerHeartbeat`, `SensorDropout`,
  `EnvelopeViolationRate`, 이 평가 순서대로. `heartbeat(now)`와
  `sensor_seen(name, now)`가 마지막 둘에 값을 공급한다.
- Fallback (§9.4): `HoldPosition`, `ZeroVelocity` (`acceleration_max` 안에서
  감속), `RetractToHome` (미리 계산된 궤적을 결정적으로 진행하며 마지막
  waypoint에서 포화), `HandoffController` (hold + `Fallback(HandoffController)`
  플래그), `EmergencyStop` (`reset_latch()`까지 래치됨).
- Clamp 순서: NaN/Inf → position → velocity → acceleration → torque →
  workspace → rate limit, 그다음 hard-limit 정리.
- 결정성 (§3.4): (config, state, inputs)의 순수 함수. tick만 사용, float
  시간 누적 없음, `HashMap` 없음, RNG 없음, 전역 상태 없음.
- 구성 이후 핫 패스에서 힙 할당 0.

## oracle (오라클)

```
cargo fmt -p es-safety --check
cargo clippy -p es-safety --all-targets -- -D warnings
cargo test -p es-safety
cargo xtask layering
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `tests/scenarios.rs`가 모든 `tests/fixtures/safety/*.json` fixture(≥ 12개;
  17개가 실제로 실림)를 재생하며, 스텝마다 기대되는 `source`, 이벤트 집합,
  실행 종료 시점의 카운터를 확인한다: 초과된 각 한계 종류, NaN 액션, 오래된
  관측, chunk underrun, `execute_chunk`에서의 chunk 절단, inference-deadline
  미스, heartbeat 유실, sensor dropout, violation-rate 트립, e-stop 래치
  지속, retract 궤적 완료, 그리고 유효한 chunk가 비트 단위로 그대로 통과함.
  이것이 §28.7 게이트 8이다.
- `tests/properties.rs`: `NaN`/`±Inf`/범위 밖 값을 담은 임의의 chunk와
  임의의 tick, `obs_age` 시퀀스에 대해, `validate`는 절대 패닉하지 않고
  반환되는 `q`는 항상 유한하며 hard position 한계 안에 있다.
- 할당 테스트가 `es_core::alloc_count::assert_no_alloc`(dev-dep
  `es-core/alloc-count`) 안에서 10,000번의 `validate` 호출을 수행한다.
- `cargo xtask layering`은 여전히 `es-safety`를 `es-policy` 의존성 없이
  layer 8로 보고한다 (rule 8, INV-11).
- 크레이트 소스 ≤ 약 1,500줄, 영어만 사용, 새 trait 없음 (INV-17).

## forbidden (금지)

- `es-policy`에 대한 어떤 의존성이든, 또는 정책·네트워크·텐서를 따온
  이름의 어떤 타입이든 (INV-11).
- plane을 비활성화·우회·no-op화하는 어떤 방법이든: `enabled` 플래그 없음,
  `Option<Envelope>` 없음, `validate` 안의 `#[cfg(test)]` 지름길 없음,
  `unsafe` 없음 (INV-12). 여유가 필요한 테스트는 fixture의 envelope를
  넓힌다.
- `validate`를 `Result`나 `Option`을 반환하도록 바꾸거나 `&self`를 받도록
  바꾸는 것 (INV-13).
- `crates/es-compile`, `crates/es-env`, `crates/es-data`, `crates/es-ir`,
  또는 루트 `Cargo.toml`을 건드리는 것 — 동시 진행 중인 패킷들이 그것들을
  소유한다.
- 여기에 정기구학, 충돌 검사기, 접촉 모델을 추가하는 것. `ee_velocity_max`,
  `min_self_distance`, `min_env_distance`, `contact_force_max`는 해시를
  위해 실려 다닐 뿐이며(§9.5), 그 상태가 존재하는 곳에서 강제된다.
- 새 확장 trait (INV-17).
