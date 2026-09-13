<!-- Korean translation of docs/packets/M2/W2-batch-domains.md. The English file is the working copy; regenerate this when it changes. -->

# W2 — 배치 도메인: 라운드로빈, 비동기 추론, chunk 버퍼, 결정성 (`es-env`)

명세: §12.1(네 도메인), §12.2(`round_robin` 카메라 선택, obs→inf 배치 구성),
§12.3(비동기 추론 하에서의 결정성), §12.4(9개 지표 집합과 대역폭 예산),
§8.5(action chunk, `H` / `K = execute_chunk`), §8.6(비동기 추론, chunk 버퍼,
`ChunkBlendPolicy`, 안전 이벤트로서의 underrun), 부록 B.5(`ChunkArrival`),
§9.4(fallback), §28.4 W2. 설계 노트: `docs/design/batch-domains.md` §9–§12.

## context (범위)

```
crates/es-env/Cargo.toml
crates/es-env/src/lib.rs
crates/es-env/src/chunk_buffer.rs
crates/es-env/src/inference.rs
crates/es-env/src/domains.rs
crates/es-env/src/env.rs          (additive: Env::step_with_policy only)
docs/design/batch-domains.md      (append §9–§12)
docs/packets/M2/W2-batch-domains.md
```

`es-eval`이 기존 `es-env` 표면 위에서 동시에 만들어지고 있으므로, 이 패킷은
**추가 전용(additive only)**이다: 새 모듈, 새 `pub use` 줄, `Env`에 대한 새
메서드 하나. 기존의 어떤 `pub` 항목도 이름이 바뀌거나 제거되거나
재타입되지 않으며, 기존 테스트 기대값도 바뀌지 않는다.

## spec (사양)

- **`chunk_buffer`** — env마다 하나의 `ChunkBuffer<NJ, H>`:
  `new(execute_chunk, ChunkBlendPolicy)`, `push(&ActionChunk<NJ, H>,
  arrival_tick)`, `next_action(tick) -> Option<[f64; NJ]>`. `CHUNK_SLOTS =
  8`개의 인라인 슬롯, `next_action` 안에 힙 없음(겹침 집합은 인라인
  `[usize; CHUNK_SLOTS]`이며 push 순서로 삽입 정렬되어 reduction 순서가
  고정된다). 세 가지 블렌드: `HardSwitch`(최신이 승리), `LinearBlend
  { steps }`(이전 chunk로부터 램프), `TemporalEnsemble { weight_decay }`
  (ACT: `w_i = exp(-m·i)`이며 `i = 0`이 겹치는 예측 중 **가장 오래된 것**,
  `f64::exp`가 아니라 `es_math::approx::exp`를 통해 — §3.4). `K`는 앞의
  두 블렌드에서 chunk의 span을 제한한다; `TemporalEnsemble`은 `valid`한
  모든 행을 읽으며, 이것이 ACT가 평균 내는 대상이다. 커버링 chunk가 없으면
  ⇒ `None`과 underrun 카운터 tick. 절대 행동을 조작해내지 않는다.
- **`inference`** — `AsyncInference`: `Submission { env, submit_tick,
  inputs }`의 FIFO 큐이며, `poll(tick)`이 `submit_tick + latency_ticks <=
  tick`인 제출들을 제출 순서로 최대 `inference.batch`개까지 방출한다.
  `latency_ticks(expected_latency_ms, control)`은
  `RuntimeHints::expected_latency_ms`를 올림하여 정수 개의 control tick으로
  바꾸며, 단 한 번의 `f32 → µs` 변환이 그 경계에서 일어난다. 시뮬레이션된
  것일 뿐 wall-clock이 아니다(§12.3); 이 모듈은 실제 worker 스레드로의
  업그레이드 경로와, 방출 규칙이 왜 worker가 아니라 여기 남아 있어야
  하는지를 문서화한다.
- **`domains`** — `Schedule` 위의 `DomainRunner<NJ, H>`: `observe_window`는
  제어 window의 각 observation tick마다 `Schedule::observation_envs(t)`가
  선택하는 env들에 대해서만 정확히 observation plan을 실행한다(§12.2);
  `infer_window`는 inference tick에 env id 오름차순으로 제출하고, poll하며,
  방출된 배치를 하나의 `PolicyRuntime::infer` 호출로 묶은 뒤, 쪼갠
  chunk들을 `apply_at = submit_tick + latency_ticks`에 push한다(부록 B.5 —
  도착 tick은 적용 tick이 *아니다*); `emit_actions`는 env마다 한 행을
  꺼내어, underrun 시에는 빈 chunk를, 그렇지 않으면 그 행을
  `SafetyPlane::validate`에 건넨다. `DomainSizing`은 `GATE` 상수(4,096 ×
  512)를 포함해 어떤 구성에 대해서든 할당 없이 계산하는 §12.4 산술이다.
- **`Env::step_with_policy(runner, policy, planes, plans)`** — §12.1 위상
  순서로 진행되는 제어 스텝 하나. `planes`는 env마다 하나씩의
  `SafetyPlane`이다(hold target, rate-limit 이력, latch는 로봇별 상태다,
  §9.3); `plans`는 비어 있거나(원본 `qpos ‖ qvel` 관측) env마다 하나씩의
  `CpuPlan`이다(`TemporalWindow` 링은 env마다다, §7.5). 종료된 에피소드는
  자신의 버퍼를 비우고 대기 중인 추론을 폐기한다.

### 불변량

- **`INV-12`**: `emit_actions`는 chunk에서 `ctrl`로 가는 유일한 경로이며,
  그 두 갈래 — action과 underrun — 모두 `SafetyPlane::validate`를 거친다.
  우회는 없다, 테스트에서도 마찬가지다; 테스트 범위는 *넓어질* 뿐 절대
  비활성화되지 않는다.
- **§12.3**: 그 무엇도 시계를 읽지 않는다. 지연, 적용 시각, observation
  나이, round-robin 슬롯은 모두 tick의 정수 함수다.
- 계층: `es-env`(9)가 `es-policy`(8)를 새로 얻는다; `es-compile`(7)과
  `es-safety`(8)는 이미 있었다. `es-safety`는 여전히 policy 의존성이
  없다(rule 8, `INV-11`).

## oracle (오라클)

```
cargo fmt -p es-env --check
cargo clippy -p es-env --all-targets -- -D warnings
cargo test -p es-env
cargo xtask layering
cargo xtask context-budget
```

## acceptance (수용 기준)

- 손으로 계산한 사례에 대한 chunk 버퍼 블렌드 수학: ACT `m = 0.01`인 두
  개의 겹치는 chunk가 `(1·a_old + e^-0.01·a_new) / (1 + e^-0.01)`와
  1e-12까지 일치한다; `LinearBlend`는 0 → ½ → 1로 램프된다; `HardSwitch`는
  `K`개의 행을 서빙한 뒤 underrun된다; 고갈된 버퍼는 `None`을 반환한다;
  eviction과 슬롯 배치는 결과를 바꾸지 않는다.
- 비동기 추론: 배치가 따라잡을 수 있을 때 방출은 정확히 `submit_tick +
  latency_ticks`다; 방출된 *시퀀스*는 `batch ∈ {1, 2, 3, 4, 8, 16}` 전반에
  걸쳐 스케줄의 prefix-stable 함수다 — 더 좁은 배치는 처리량을 배급할 뿐
  절대 순서를 바꾸지 않는다; 실행은 비트 단위로 replay된다.
- 16-env `FakeBackend` + 상수 chunk `FakePolicy`로 200 제어 스텝을
  실행하면 `observation.batch ∈ {4, 8, 16}` 각각에 대해 두 번의 실행이
  비트 단위로 동일하다.
- **배치 독립성 불변량을 정확히 표현하면.** Round-robin은 *어느* env가
  *언제* 관측되는지를 바꾸고, 관측된 env만 chunk를 받으므로, env별 궤적은
  `observation.batch`에 **독립적이지 않다**. 실제로 성립하는 것: 스케줄이
  유도한 observation tick이 일치하는 env들은 구별 불가능하다는 것이다.
  각 구성의 16개 env를 observation-tick 집합(`16 / observation.batch`개
  그룹)으로 묶어, 그룹 내부에서는 동일함을, 그룹 사이에서는 다름을
  요구하는 방식으로 단언한다; 여기에 그 메커니즘인
  `underrun_rate(4) > underrun_rate(8) > underrun_rate(16)`을 더한다.
- `DomainSizing::GATE`는 할당 없이 §12.4의 산술을 재현한다: 30,720
  frames/s, 1.54 Gpixel/s, 4.6 GB/s RGB8, `f32` 정규화 후 18.5 GB/s, 4,096
  env에 대해 chunk 버퍼 36.7 MB; 모든 카메라를 켜면 §12.4의 245,760
  frames/s, 12.3 Gpixel/s, 37 GB/s와 148 GB/s — `round_robin`을 정당화하는
  8배 차이다. **`Target / Status: 미검증`**: §28.4 게이트(4,096 sim env ×
  512 obs env 안정)는 여기서는 측정이 아니라 예산이다.

## forbidden (금지)

- 기존 `es-env` `pub` 항목의 이름 변경·제거·재타입, 또는 기존 테스트
  기대값 변경(`es-eval`이 그 표면 위에서 동시에 만들어지고 있다).
- `crates/es-policy`, `crates/es-compile`, `crates/es-data`, `crates/es`,
  루트 `Cargo.toml`, 그리고 그 외의 모든 크레이트 — 인접 패킷들의
  소관이다.
- 실제 추론 스레드(§12.3는 결정성 우선이다; 스레딩은 이후의 처리량
  패킷이다), `EnvSelection::Subset`, 실제 sensor/camera 관측(M2 W3),
  배치화된 policy *lowering* 의미론, `EnvMetrics`를
  `es-telemetry`(layer 10)에 연결하는 것.
- `HashMap`, chunk 경로에서의 `std` 초월함수, 결정 로직 어디에서든의
  wall-clock, 그리고 새로운 extension-point trait(`INV-17`).
