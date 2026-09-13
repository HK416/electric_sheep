<!-- Korean translation of docs/design/safety-plane.md. The English file is the working copy; regenerate this when it changes. -->

# Safety Plane 런타임 (`es-safety`) — 설계

Spec 참조: §9 (Deployment IR과 Safety Plane), §9.3 (envelope), §9.4 (watchdog와
fallback), §9.5 (시뮬/실기 동일성), §8.5 (action chunk), §8.6 (비동기 추론, chunk
underrun), §10.3 (`envelope_violation_rate`, `chunk_underrun_rate`), §18.5 (fallback은
정상 동작이다), 부록 B.4 (고정된 타입 형태), §28.7 게이트 8.

불변식: INV-11 (`es-safety`는 `es-policy`에 절대 의존하지 않는다), INV-12 (어떤 코드
경로도 plane을 비활성화하지 않는다), INV-13 (`validate`는 `Result`를 반환하지 않는다).

## 이 크레이트는 무엇인가

`es-ir::deployment`는 **구성(configuration)**이다: 한계, watchdog, fallback을
기술하는 검증되고 해시 가능한 서술이다. 이 크레이트는 그 구성을 제어 틱마다 한 번씩
실행하는 **런타임**이다. 둘은 절대 합쳐지지 않는다. IR은 `Vec` 형태, serde 형태,
진단(diagnostic) 형태의 코드를 소유하고, 런타임은 고정 크기 배열과 실패할 수 없는
함수를 소유한다.

§9.5에 따라 동일한 코드가 시뮬레이션과 실제 하드웨어에서 실행된다. 이 크레이트 어디에도
시뮬레이션 모드도, "training" 모드도, `enabled` 플래그도 없다 (INV-12). 여유가 필요한
테스트는 자신의 fixture에서 envelope를 넓힌다.

## `no_std` 상태

부록 B.4는 "no_std 가능, 힙 할당 0"이라고 말한다. `es-ir`는 `std`이고 (`String`, `Vec`,
`BTreeSet`, `serde`를 사용한다), `from_ir`가 이를 읽어야 하므로 크레이트 전체는 M1
동안 `std`로 남는다. 지금 전달되는 것은 운영상 중요한 절반이다.

- **핫 패스는 아무것도 할당하지 않는다.** `validate`, `heartbeat`, `sensor_seen`은
  `SafetyPlane` 내부의 고정 크기 배열과, `from_ir`에서 한 번 크기가 정해진 뒤로는
  인덱싱만 되는 두 개의 `Vec`(retract 궤적과 sensor-dropout 테이블)만을 건드린다.
  한 테스트는 긴 `validate` 루프를 `es_core::alloc_count::assert_no_alloc`로 감싼다.
- **부동소수점 시간이 없다.** 모든 시간은 `PhysTick` (u64)과 `Micros` (u64)이다.
  시간에서 파생된 유일한 float는 상수 제어 주기 `dt_s`뿐이며, 이는 `from_ir`에서
  유리수 `TickRate`로부터 한 번 계산되고 절대 누적되지 않는다 (§3.4).
- **`HashMap`도, RNG도, 전역 상태도 없다.** sensor 테이블은 선형으로 스캔되고
  `&str`로 비교되는 `Vec`이다. 순회 순서는 구성될 때의 순서다.

크레이트를 문자 그대로 `no_std`로 만드는 분리 작업은 이후 패킷의 몫이다. `from_ir`를
`std` feature 뒤로 옮기고 런타임 코어는 기본 빌드에 남긴다. `[features] std =
["es-ir"]`가 그 형태이며, 아래의 런타임 코어는 구성 시점의 `From` 변환을 제외하고는
`es-ir` 타입을 전혀 참조하지 않으므로 그 분리는 기계적인 작업이다.

## 구성 — `SafetyPlane::from_ir`

```rust
pub fn from_ir(ir: &DeploymentIr) -> Result<Self, SafetyConfigError>
```

이것이 유일한 생성자다. `SafetyPlane::new()`도, `Default`도, public 필드도 없으므로
envelope 없는 plane은 구성할 수 없다 (INV-12). `from_ir`는 다음 순서로 거부한다.

1. `ir.validate()`가 진단을 반환한 경우 — IR이 내부적으로 일관되지 않다.
2. `ir.robot.n_joints != NJ` 또는 `ir.action.dim != NJ`인 경우 — const generic이
   구성과 불일치한다.
3. `ir.action.horizon != H`인 경우 — chunk 버퍼가 맞지 않게 된다.
4. 어떤 한계값이 유한하지 않거나, 관절별 벡터의 길이가 잘못된 경우. IR validator가
   이미 대부분을 검사하지만, 런타임은 고정 배열로 복사하며 `validate`를 건너뛴
   호출자가 만든 IR을 신뢰해서는 안 되므로 이 검사를 반복한다.
5. watchdog 종류가 두 번 나타나거나, `EnvelopeViolationRate`의 window가
   `WINDOW_CAP` (256)을 초과하는 경우 — ring은 그 상한으로 미리 할당된다.

그런 다음 모든 것이 고정 크기 배열로 복사된다: `[Limit; NJ]`, 속도·가속도·토크·저크를
위한 `[f64; NJ]`, rate-limit 두 행, 그리고 이미 유효한
`[lower + margin, upper - margin]` 쌍으로 접혀 들어간 soft margin (핫 패스가
관절당 세 번이 아니라 한 번만 비교하도록).

## 미리 할당된 상태 레이아웃

```rust
SafetyPlane<NJ, H>
├── envelope: Envelope<NJ>            fixed arrays, immutable after construction
│     soft:   [Limit; NJ]             hard limit shrunk by the soft margin
│     hard:   [Limit; NJ]             hard limit, used for the final scrub
│     vel_max, acc_max, tau_max, jerk_max: [f64; NJ]
│     d1_max, d2_max: [f64; NJ]       action rate limits (§9.3 rate_limit)
│     workspace: Workspace            box/cylinder/convex-hull, EE spaces only
│     dt_s: f64                       control period, constant
├── watchdogs: Watchdogs              one Option per kind; Vec<SensorWatch> for dropout
├── fallback: Fallback<NJ>            kind + Vec<[f64; NJ]> retract trajectory (sized once)
├── state: SafetyState<NJ, H>
│     chunk:        [[f64; NJ]; H]    the accepted chunk (copied in, never borrowed)
│     chunk_valid:  usize             valid rows of `chunk`
│     chunk_mode:   ExecutionMode
│     cursor:       usize             index of the next row to execute
│     last_safe:    [f64; NJ]         last emitted action (the hold target)
│     prev_safe:    [f64; NJ]         the one before it (second difference)
│     vel:          [f64; NJ]         velocity estimate, (last_safe - prev_safe) / dt_s
│     prev_vel:     [f64; NJ]         previous velocity estimate (acceleration)
│     retract_idx:  usize             cursor into the retract trajectory
│     last_chunk_tick / last_beat_tick: PhysTick
│     estop_latched: bool
└── counters: SafetyCounters
      violations: [u64; ViolationKind::COUNT]
      fallback_activations, steps, clamped_steps: u64
      window: ViolationWindow           [u64; 4] bit ring, len ≤ 256, plus head/filled/ones
```

여기서 자라나는 것은 아무것도 없다. `Vec<[f64; NJ]>` (retract)와 `Vec<SensorWatch>`
(dropout)만이 유일한 힙 객체이며 둘 다 `from_ir` 이후 확정된다.

`last_safe`는 각 관절의 soft-limit 중간값에서 시작하며, 이는 구성상 envelope 내부에
있다. 실제 자세를 아는 런타임은 첫 `validate` 이전에 `observe_state(&q, &qd)`를
호출한다 — 이것이 실제 로봇 팔에 필요한 보정 손잡이(calibration knob)다. 물리적
관절이 중간값이 말하는 위치에 있는 경우는 결코 없기 때문이다.

## `validate` — 알고리즘

```rust
pub fn validate(&mut self, chunk: &ActionChunk<NJ, H>, obs_age: Micros, now: PhysTick)
    -> SafeAction<NJ>
```

`Result`가 없다 (INV-13). 패닉도 없다. 모든 배열 인덱스는 이미 `NJ`/`H`로
경계지어진 `usize`이고, 모든 나눗셈은 상수 `dt_s > 0`으로 이루어지며, 유한하지 않은
입력은 산술에 도달하기 전에 정리(scrub)된다. `last_safe`가 항상 값을 하나 들고
있으므로 실행 가능한 액션은 항상 존재한다.

**0단계 — 래치.** `estop_latched`이면 정리된 hold 액션과 함께 즉시
`Fallback(EmergencyStop)`을 반환한다. 카운터는 이 스텝과 fallback 활성화를 기록하며,
어떤 watchdog도 평가되지 않고 어떤 chunk도 받아들여지지 않는다. `reset_latch()`만이
이를 해제한다.

**1단계 — chunk 수락.** 들어온 chunk를 저장된 chunk와 비교한다(`valid`, `mode`, 그리고
`valid`한 앞부분 행들을 비트 단위로). 다르면 새 chunk이므로 복사해 넣고
`cursor = 0`, `last_chunk_tick = now`로 둔다. 동일하면 커서는 계속 진행한다.

> 알려진 한계: 동일한 chunk를 두 번 연속 내보내는 정책은 오래된(stale) chunk로 읽혀
> 결국 `ChunkUnderrun`을 유발한다. 이는 안전한 쪽으로 치우치는 것이며, 부록 B.4의
> chunk 형태에 시퀀스 번호를 넣지 않기 위한 선택이다. 실제 워크로드가 이 문제에
> 부딪히면 해법은 휴리스틱이 아니라 `ActionChunk`에 `seq: u64`를 추가하는 것이다.

**2단계 — 실행 가능 길이.** `RecedingHorizon`에서는 `len = min(chunk_valid, K)`이며
`K = action.execute_chunk`이고, 그 외 모든 모드에서는 `min(chunk_valid, H)`이다
(§8.5). `len`을 넘는 행들은 유효한 예측이더라도 실행 가능하지 않다 — 이것이 spec이
말하는 절단(truncation)이며, 이를 넘어가는 것은 조용한 연장이 아니라 underrun이다.

**3단계 — watchdog**, 다음의 고정된 순서로 평가한다. 하나가 트립되어도 전부 평가되므로
모든 이벤트가 카운터와 `SafeAction::events`에 기록된다. 어떤 트립이든 설정된 단
하나의 fallback을 실행한다 — IR은 단일 `FallbackPolicy`만을 가지므로, 어떤
watchdog이 발동했는지는 기록되는 이벤트만 바꿀 뿐 응답을 바꾸지 않는다.

| # | Watchdog | 조건 |
|---|---|---|
| 1 | `ChunkUnderrun` | `cursor >= len` — 나머지가 후보 행을 필요로 하므로 가장 먼저 평가된다 |
| 2 | `NanInf` | 후보 행의 어떤 성분이든 유한하지 않은 경우 — **무조건적**이며 설정 불가능하다 (INV-12; IR에는 설계상 이런 variant가 없다) |
| 3 | `StaleObservation` | `obs_age > max_age` |
| 4 | `InferenceDeadline` | `micros_since(last_chunk_tick, now) > budget` — 이 예산은 plane이 새 chunk 없이 얼마나 오래 지났는지를 측정한다 |
| 5 | `ControllerHeartbeat` | `micros_since(last_beat_tick, now) > timeout` |
| 6 | `SensorDropout` | 설정된 어떤 sensor에 대해서든 `micros_since(last_seen, now) > max_gap` |
| 7 | `EnvelopeViolationRate` | `window.fraction() > max_frac`이며, **이전 스텝 시점**의 window를 사용한다 — 이번 스텝 자체의 clamp는 아직 기록되지 않았고, 그것이 이 규칙을 순환적이지 않게 만든다 |

`ChunkUnderrun`과 `NanInf`는 항상 활성화되어 있다. 나머지 다섯은 IR이 나열한
경우에만 활성화된다. 나열되지 않은 watchdog은 절대 트립되지 않는데, 이는 안전
계층의 비활성화가 아니라 설정 선택이다 — envelope의 clamp들은 아래에서 항상
실행된다.

`micros_since`는 정수 연산이다: `ticks * period_us`, 여기서
`period_us = 1e6 * den / num`은 제어 주기율(control rate)로부터 나온다.

**4a단계 — fallback 경로** (어떤 watchdog이든 트립된 경우). fallback 액션을
생성한 뒤(아래 참조), 그것을 최종 정리(non-finite → hold, 그다음 hard position
clamp)에 통과시켜 fallback 출력이 정책 출력과 동일한 규칙으로 envelope 안에
들어오게 한다. `source = Fallback(kind)`, `fallback_activations += 1`. violation
window는 이 스텝을 위반으로 기록한다. §18.5에 따라 이것은 *정상 동작*이다 — env는
`Ok` 상태를 유지하고 이벤트가 기록된다.

**4b단계 — clamp 경로** (아무것도 트립되지 않은 경우). 후보 행은 정확히 다음
순서로 clamp되며, 각 단계는 값을 바꾸었을 때 자신의 `ViolationKind`를 기록한다.

1. **NaN/Inf** — 여기서는 일어날 수 없다(watchdog 2가 이미 잡았다). 두 경로의
   단계 순서를 동일하게 유지하기 위해 이 정리는 방어적 no-op으로 남는다.
2. **position** — soft limit `[lower + margin, upper - margin]`로 clamp한다
   (§9.3 `position_limit`).
3. **velocity** — 암시된 속도 `(a - last_safe) / dt_s`. `|v| > vel_max[i]`이면
   `a = last_safe + sign(v) * vel_max[i] * dt_s`로 설정한다.
4. **acceleration** — 암시된 가속도 `(v - vel[i]) / dt_s`. `acc_max[i]`를
   초과하면 속도를 `vel[i] ± acc_max[i] * dt_s`로 clamp하고 그로부터 `a`를
   재구성한다.
5. **torque** — `action.space == JointTorque`일 때만, 즉 액션이 곧 토크인
   경우에만: `|a| <= tau_max[i]`로 clamp한다. 다른 space에서는 no-op 단계이며,
   그래서 별도의 분기가 아니라 여기 위치한다.
6. **workspace** — `action.space`가 `EePose` 또는 `EeDelta`이고 `NJ >= 3`일
   때만: 성분 `[0..3]`을 박스/실린더/convex hull로 투영한다 (§9.3 `workspace`).
   관절 공간 액션은 투영되지 않는다. 이 크레이트에는 정기구학(forward kinematics)이
   없으며, 여기서 하나를 새로 만드는 것은 두 번째의, 검증되지 않은 로봇 모델을
   만드는 것과 같다. 관절 공간 workspace 강제는 충돌 패킷의 몫이다.
7. **rate limit** — 1차 차분 `|a - last_safe| <= d1_max[i]`, 그다음 2차 차분
   `|a - 2*last_safe + prev_safe| <= d2_max[i]` (§9.3 `rate_limit`). 앞선 단계들이
   바꾼 것도 함께 매끄럽게 만들도록 마지막에 적용된다.

그런 다음 hard position clamp가 최종 정리로 한 번 더 실행된다. 7단계가 값을 soft
band 바깥으로 다시 밀어낼 수 있기 때문이다.

어떤 단계든 값을 바꾸었으면 `source = Clamped`, 아니면 `source = Policy`이다.
`cursor += 1`은 이 경로에서만 일어난다 — fallback 스텝은 chunk 행을 소비하지
않으므로, 조건이 해소되었을 때도 버퍼는 그대로 남아 있다.

**5단계 — 상태 갱신.** `prev_vel = vel`, `vel = (out - last_safe) / dt_s`,
`prev_safe = last_safe`, `last_safe = out`, `steps += 1`, window push. 두 경로에서
동일하므로, hold 목표는 항상 실제로 내보내진 것을 추적한다.

`collision_constraint`, `jerk_limit`, `ee_velocity_max`, `contact_force_max`
(§9.3)는 envelope에 실려 있지만 여기서 강제되지는 않는다. 앞의 셋은 이 크레이트에
주어지지 않은 상태(접촉 집합, FK, 측정된 힘)를 필요로 한다. 그것들은 그 상태가
존재하는 곳에서 강제된다. envelope는 그럼에도 그것들을 실어 나른다.
`deployment_hash`가 이들을 포함하도록 하기 위해서다 (§9.5).

## Fallback 의미론 (§9.4)

| 정책 | 생성되는 액션 |
|---|---|
| `HoldPosition` | `last_safe` 그대로 유지 |
| `ZeroVelocity` | 속도를 스텝당 최대 `acc_max[i] * dt_s`만큼 0을 향해 줄인다, `q = last_safe + v_new * dt_s`. 유한한 스텝 수 안에 정지에 도달해 유지하며 가속도 한계를 절대 넘지 않는다. |
| `RetractToHome` | `trajectory[retract_idx]`, `retract_idx`는 마지막 waypoint에서 포화(saturate)된다. 결정적으로 진행하며 fallback 스텝마다 waypoint를 하나씩 밟는다. 완료된 retraction은 마지막 waypoint를 영원히 유지한다. 조건이 해소되어도 인덱스는 리셋되지 **않는다** — 절반쯤 진행된 retraction은 처음부터 다시 시작하지 않고 이어서 진행된다. |
| `HandoffController` | `source = Fallback(HandoffController)`인 `last_safe` (hold). 그 variant 자체가 플래그다. 호출자는 이를 보고 자신의 고전적 컨트롤러로 전환한다. 이 크레이트는 그것을 실행하지 않는다. |
| `EmergencyStop` | `last_safe`이며 `estop_latched = true`. 이후의 모든 `validate`는 `reset_latch()`가 명시적으로 호출될 때까지 0단계에서 즉시 반환된다. `validate` 안의 어떤 것도 이 래치를 해제하지 않는다. |

fallback은 결코 신경망이 아니며 정책 런타임이 살아 있는지에도 전혀 의존하지 않는다
(§9.4). 다섯 가지 모두 미리 할당된 상태에 대한 테이블 조회다.

## 카운터 (§10.3)

```rust
pub struct SafetyCounters {
    pub violations: [u64; ViolationKind::COUNT],   // per kind
    pub fallback_activations: u64,
    pub steps: u64,
    pub clamped_steps: u64,
    window: ViolationWindow,
}
```

- `envelope_violation_rate()` — 슬라이딩 윈도 안에서 clamp되었거나, 투영되었거나,
  fallback으로 떨어진 스텝의 비율. 이것이 §10.3의 1급 지표이며, `EnvelopeViolationRate`
  watchdog이 읽는 것과 같은 수치다. window는 누적 1의 개수를 함께 들고 있는
  `[u64; 4]` 비트 ring(상한 256)이므로, 이 비율은 누적된 float가 아니라 정수 비다.
- `chunk_underrun_rate()` — `violations[ChunkUnderrun] / steps` (§8.6, §10.3).
- `violations[kind]` — §10.3의 `failure_mode_histogram`을 위한 종류별 카운트.

카운터는 단조 증가하는 `u64`이다. `reset_counters()` 외에는 public API의 어떤
것도 이를 리셋하지 않으며, 이 함수는 래치는 건드리지 않는다.

## 위반 시나리오 스위트 (§28.7 게이트 8)

`tests/fixtures/safety/*.json`을 `tests/scenarios.rs`가 `NJ = 3`, `H = 4`,
`K = 3`에 대해 재생한다. 각 fixture는 완전한 `DeploymentIr`와 스텝 목록을 싣고
있다. 각 스텝은 입력(chunk 또는 "이전 것을 재사용", `obs_age_us`, heartbeat,
관측된 sensor)과 기대되는 `source`, 기대되는 이벤트 집합, 선택적으로 기대되는
`q`를 선언한다. 카운터는 실행이 끝난 시점에 검사된다. 열일곱 개의 시나리오:

| # | Fixture | 무엇을 증명하는가 |
|---|---|---|
| 1 | `valid_chunk_passes` | 조건을 만족하는 chunk가 비트 단위로 그대로 나오고, `source = Policy`, 이벤트 없음, `envelope_violation_rate == 0` |
| 2 | `position_limit` | soft 관절 한계를 넘은 행이 그 한계로 clamp되고 `source = Clamped` |
| 3 | `velocity_limit` | `vel_max`에 비해 너무 큰 스텝이 `last_safe ± vel_max*dt`로 clamp됨 |
| 4 | `acceleration_limit` | `acc_max`에 비해 너무 큰 속도 변화가 clamp됨 |
| 5 | `torque_limit` | `JointTorque` action space에서 `|a| > tau_max`가 clamp됨 |
| 6 | `workspace_box` | workspace 박스 바깥의 `EePose` 액션이 그 위로 투영됨 |
| 7 | `rate_limit` | 1차/2차 차분 한계가 그 외엔 한계 내인 액션을 clamp함 |
| 8 | `nan_action` | `NaN` 성분이 즉시 fallback으로 떨어짐, 출력은 유한함 (§9.3 `action_validity`) |
| 9 | `stale_observation` | `obs_age > max_age`가 fallback으로 떨어졌다가 나이가 줄면 복구됨 |
| 10 | `chunk_underrun` | 커서가 `valid`를 넘어 진행됨; fallback, 카운터, 새 chunk에서 복구 (§8.6) |
| 11 | `chunk_truncation` | `RecedingHorizon` 하에서 `valid = 4 > K = 3`: 행 3은 절대 실행되지 않고, 스텝 4는 underrun이다 |
| 12 | `inference_deadline` | 예산 안에 새 chunk가 없으면 fallback으로 떨어졌다가 복구됨 |
| 13 | `heartbeat_loss` | timeout을 넘긴 `heartbeat()` 부재는 fallback으로 떨어짐; beat가 오면 `Policy`로 복원됨 |
| 14 | `sensor_dropout` | 설정된 sensor 하나가 `max_gap`을 넘겨 조용해짐; `sensor_seen`이 복원함 |
| 15 | `violation_rate` | 충분히 많은 clamp된 스텝이 window 비율을 `max_frac` 위로 밀어 rate watchdog을 트립시킴 |
| 16 | `estop_latch` | E-stop이 한 번 트립되면 이후 모든 스텝은 완벽한 chunk가 와도 `reset_latch()` 전까지 `Fallback(EmergencyStop)`으로 남음 |
| 17 | `retract_completes` | `RetractToHome`이 waypoint 단위로 진행되며 마지막 것을 유지함 |

거기에 더해 `tests/properties.rs`: `NaN`, `±Inf`, subnormal, 모든 한계를 크게
벗어난 값을 담은 임의의 chunk와, 임의의 `obs_age`/tick 시퀀스에 대해, 반환되는
`q`는 항상 유한하고 **hard** position 한계 안에 있으며, `validate`는 절대
패닉하지 않는다. 그리고 할당 테스트: `assert_no_alloc` 안에서 10,000번의
`validate` 호출.

## 왜 trait가 없는가

Safety Plane은 하나뿐이다. `PhysicsBackend`, `PolicyRuntime`, `TaskNodeFactory`,
`LearningNodeFactory`, `InferenceBackend`, `Scalar`, `DeterministicAcc`가 일곱
개의 확장 지점이며 (INV-17), 이것은 그중 하나가 아니다. `trait SafetyPolicy`는
호출자가 아무것도 하지 않는 plane을 설치할 수 있는 이음매(seam)가 될 것이며,
그것은 정확히 INV-12가 금지하는 것이다.
