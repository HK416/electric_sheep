<!-- Korean translation of docs/design/batch-domains.md. The English file is the working copy; regenerate this when it changes. -->

# 배치 도메인, 리셋, 그리고 env RNG (M1 W6)

Spec §12 (네 가지 배치 도메인, 도메인 간 정책, 결정성, 9개 지표), 부록 B.5
(스케줄러 스케치), §6.4 (실행 의미론), §18.1 (정수 시간), §18.5 (실패 의미론),
§13.1 (에피소드 기록이 학습 루프에 공급됨). 리뷰 등급 C.

이것은 `es-env`가 M1 W6에서 실제로 구현하는 것이며 — 못지않게 중요한 것으로 —
구현하지 않는 것이기도 하다.

## 1. 네 도메인, 네 크기

Spec §12.1은 "모든 것의 단위는 env 하나다"라는 생각을 거부한다. 각 도메인은
자신만의 배치 크기와 자신만의 주기를 가지며, `es-env`는 절대로 하나를 다른
것에서 유도하지 않는다.

| 도메인 | `batch` | `period` (sim tick 단위) | Device |
|---|---|---|---|
| simulation | `N_sim` (예: 4096) | **항상 1** — tick 자체를 정의한다 | `Cpu` / `Gpu(id)` |
| observation | `N_obs` (예: 512) | `sensor_dt / dt_phys`, 예를 들어 1 kHz에서 33 | |
| inference | `N_inf` (예: 256) | `dt_ctrl / dt_phys`, observation 주기의 배수 | |
| training | `N_train` (예: 64 시퀀스), 선택적 | inference 주기의 배수 | |

`DomainCfg.period`는 항상 simulation tick의 정수 개수이며 결코 지속 시간이
아니다 (§18.1: float 시간 없음, 타입으로 강제됨 — `BatchDomains` 어디에도 `f64`는
없다). 주기가 실제로 뜻하는 벽시계 시간은 backend의 `TickRate`에서 오며, `Env`가
`ModelInfo`에서 읽는다.

### 검증 (`Schedule::build`)

다음은 거부되며, 각각 고유한 `EnvError::Schedule` 메시지를 낸다.

- `batch == 0`이거나 `period == 0`인 경우;
- `simulation.period != 1`인 경우;
- `inference.period % observation.period != 0`인 경우 — inference tick은 반드시
  observation tick 위에 있어야 한다. 그렇지 않으면 정의되지 않은 tick의 observation을
  소비하게 된다;
- `training.period % inference.period != 0`인 경우;
- `observation.batch > simulation.batch`이거나 `inference.batch > observation.batch`인
  경우 (§12.1: 깔때기는 오직 좁아지기만 한다);
- `observation.batch == 0`인데 선택이 `All`이 아닌 경우.

## 2. 정적 계획

`Schedule`은 순수한 데이터다: 하이퍼주기 `H = lcm(obs, inf, train)`과, 길이 `H`인
`Vec<TickPlan>`이며, `TickPlan { offset, observation, inference, training }`는
sim tick `t`에서 어떤 도메인이 발화하는지를 말한다(`plan[t % H]`). 이 계획은
무엇이 얼마나 빠르게 도는지에 전혀 의존하지 않으므로, 두 대의 기계가 동일하게
인터리빙된다. `Schedule::ticks()`는 하나의 하이퍼주기를 순회한다.

tick **내부에서의** 발화 순서는 §6.4의 위상 순서에 의해 고정되며 데이터가 아니다.

```
PrePhysics -> Physics -> PostPhysics -> Observation -> Reward -> Termination -> Record
```

따라서 둘 다 발화하는 tick에서는 observation이 inference보다 먼저 오며,
inference는 절대 자신의 tick보다 새로운 observation을 보지 않는다.

### 카메라 env 선택 (§12.2, §12.3)

`N_obs < N_sim`일 때, 활성 카메라 집합은 `(tick, obs_period, N_obs, N_sim)`의
결정적 함수다 — RNG도, 부하 피드백도 없다.

```
cycle  = ceil(N_sim / N_obs)
slot   = (tick / obs_period) % cycle
active = [slot * N_obs, min((slot + 1) * N_obs, N_sim))
```

이는 `k = N_obs`인 §12.2의 `round_robin(k, period)`이다. 모든 env의 카메라는
`cycle`개의 observation tick마다 한 번씩 방문되며, 그 순서는 §12.3이 요구하는
대로 env-ID 오름차순이다.

## 3. 리셋 프로토콜

`reset(None)`은 배치 전체를 리셋하고, `reset(Some(&envs))`는 부분 집합을
리셋하며 `capabilities().supports_reset_subset`을 요구한다(그렇지 않으면
`EnvError::Physics(Unsupported)`가 되며, backend가 그 이름을 밝힌다 — 우리가
그것을 흉내내지 않는다). env마다, 다음 순서로 진행한다.

1. `episode[env] += 1` — 에피소드 카운터는 모든 RNG 키의 일부이므로, 재리셋이
   이전 에피소드의 추첨을 절대 재생하지 않는다.
2. 해당 env의 리셋 버퍼 행(`qpos`, `qvel`)을 0으로 만든 다음, `ResetState`와
   `Randomization` 노드를 오름차순 `NodeId` 순서로 적용한다 (§6.4: node ID가
   동점을 깬다. 즉 ID를 보존하는 그래프 재작성은 추첨 순서도 보존한다).
3. 무작위화된 행들로 `backend.reset(envs, Some(&state))`를 호출하며, 다른
   env들의 행은 변경 없이 그대로 넘어간다.
4. `EpisodeRecorder::start(env)` — 이전 에피소드가 마무리되어 반환된다.

자동 리셋: `step`은 physics step 이후 termination을 평가한다. 종료된 env들은
같은 `step` 호출의 *끝에서* 리셋되므로, 에피소드 경계는 tick 경계와 일치하고
호출자는 절대 반쯤 리셋된 배치를 보지 않는다. 반환되는 done 플래그는 새 에피소드가
아니라 종료된 tick에 속한다.

격리된(quarantined) env들(§18.5)은 여전히 backend에 의해 스텝되지만 보상
리덕션과 기록되는 에피소드에서는 제외된다. `StepReport`의
`FailureKind::NanDetected` / `Diverged`가 `es-core`의 `FailurePolicy`를 거쳐
이를 유도한다.

## 4. RNG

Spec §3.4는 전역 RNG를 금지하며 §6.6 `DET-001`은 모든 무작위 소스에 스트림
이름을 필수로 요구한다. 따라서 `EnvRng`는 순차-상태적(sequential-stateful)이
아니라 **카운터 기반**이다.

```
key     = splitmix64_chain(seed, env_index, episode, stream_id.bytes[0..8], bytes[8..16])
nth     = splitmix64(key ^ (counter * GOLDEN))
```

테스트가 고정하는 결과들:

- 같은 `(seed, env, episode, stream)`은 언제나 같은 시퀀스를 낸다. 어떤 기계에서든,
  다른 env들이 무엇을 했는지와 무관하게 — 공유 상태가 없으므로 env 순서와 스레드
  개수는 무관하다.
- 다른 `env`, `episode`, `stream`은 서로 무관한 시퀀스를 낸다. env들은 같은
  seed를 공유하더라도 서로 상관되지 않는다.
- 스트림은 그 이전 것들을 재생하지 않고도 어느 시점에서든 다시 유도될 수 있다.
  이것이 부분 집합 리셋을 저렴하게 만들고 재생을 정확하게 만드는 요인이다.

`stream_id: StableId`는 `Randomization`/`ResetState` 노드가 선언한 스트림 이름의
`StableId::from_path(stream_name)`이다. 따라서 스트림 이름을 바꾸면 추첨과
task hash가 바뀌고, 그 외에는 아무것도 바뀌지 않는다.

**분포** (`es_ir::task::Distribution`, 다섯 가지 모두 다룸):

| Variant | 방법 |
|---|---|
| `Constant(v)` | `v`, 추첨 소비 없음 |
| `Uniform{lo,hi}` | `lo + (hi - lo) * u`, `u = (x >> 11) as f64 * 2^-53` |
| `LogUniform{lo,hi}` | `exp(ln lo + (ln hi - ln lo) * u)`; `lo > 0` 필요 |
| `Normal{mean,std}` | Box–Muller, `mean + std * sqrt(-2 ln u1) * cos(2 pi u2)`, 고정된 연산 순서, 두 번째 변량은 버림 |
| `Choice(vs)` | `vs[(next_u64 % len)]`, 거부 없는 방식 (현실적인 어떤 len에 대해서도 편향 < 2^-64) |

초월함수는 `std`가 아니라 `es_math::approx`를 거친다(`ln`, `exp`, `cos`, `sqrt`)
(§3.2 `DET-010`). 이들은 `f32`다. 무작위화 추첨은 `f64` 정밀도를 필요로 하지
않으며, x86/ARM/GPU 사이에서 비트가 동일한 추첨이 마지막 몇 개의 가수 비트보다
더 중요하다. 상태 배열이 그러하므로 추첨값은 여전히 `f64`로 반환된다.

## 5. 무작위화 대상

`RandomizationPlan::compile(task, scene)`는 각 노드의 `target` 문자열을 구성
시점에 `SceneDesc`에 대해 **한 번만** 해석한다. 그래서 리셋마다 도는 경로에는
문자열 작업도, 풀어야 할 `Result`도 없다.

```
qpos[i]  qvel[i]                       state indices
joint.<name>.qpos  joint.<name>.qvel   resolved via ModelInfo's joint -> IndexRange maps
body.<name>.mass                       scale factor
geom.<name>.friction                   scale factor
actuator.<name>.gain                   scale factor
```

그 외의 것은 컴파일 시점에 `EnvError::Unsupported(target)`이 되며 절대 조용히
건너뛰지 않는다 (§17.2의 규칙을, 무작위화에 적용한 것).

**한계, 의도된 것:** `PhysicsBackend`에는 모델 파라미터 API가 없다
(`set_state`는 노출하지만 `set_body_mass`는 노출하지 않는다). 따라서 세 개의
scale 대상은 *추첨되고, `EpisodeMeta.param_scales`에 기록될 뿐, backend로
밀어 넣어지지 않는다*. `set_param`이 trait에 도착하는 순간 계획은 완전해지고
그 추첨은 hash chain의 일부가 된다. state 대상(`qpos`/`qvel`)은 오늘 실제로
적용된다.

## 6. 보상과 종료

`Reward`와 `Terminate`는 표현식 리터럴이 아니라 입력 엣지를 가진 그래프
싱크(sink)다. `es-env`는 각 싱크로 들어오는 *스칼라 원뿔(scalar cone)*을
구성 시점에 (`plan.rs`) `es_ir::task::Expr`로 한 번 낮춘 뒤, env마다 tick마다
이미 존재하며 결정적이라고 이미 명세된 `Expr::eval`로 평가한다(고정된 연산
순서, `NaN` 대신 `None`).

원뿔에서 지원되는 것: `GetJointState`, `GetSensor`, `GetTime`, `GetContact`
(잎, 이름 붙은 포트에 바인딩됨), `Arith`, `Compare`, `Clamp`. reward나
terminate 원뿔 안의 그 외 모든 노드 종류는
`EnvError::Unsupported("<kind> in a reward cone")`이다. 포트 이름은
`joint.<id>.<quantity>[i]`, `sensor.<id>[i]`, `time`, `contact.<a>.<b>`이다.

보상 집계는 `Reward` 노드들에 대해 오름차순 `NodeId`로 `sum(weight * term)`을
계산하는 것이다. `Mean`, `Min`, `Max`는 벡터 항의 원소들을 집계하며 여기서는
단일 원소 no-op이다.

## 7. 지표

`EnvMetrics`는 §12.4의 이름들을 싣고 있으며, 모든 필드가 `Option`이다 — 아무도
측정하지 않은 지표는 조작된 0이 아니라 항상 `None`이며, `step/s` 필드는
**아예 존재하지 않는다**. `es-env`는 layer 9이고 `es-telemetry`는 layer 10이므로
`EnvMetrics`는 telemetry 의존성이 전혀 없는 평범한 구조체다. `es-telemetry`가
위에서 그것을 `PerfMetrics`로 변환한다. W6는 실제로 관측하는
`physics_steps_per_sec`, `actions_per_sec`과 tick/wall-time 카운터를 채운다.
렌더, inference, VRAM 필드는 해당 도메인이 생기기 전까지 `None`으로 남는다.

## 8. 미룬 것

- **비동기 추론과 chunk 버퍼** (§12.3, §8.6, B.5 `ChunkArrival`). 스케줄은 이미
  inference tick을 표시해 두고 있다. `apply_at = computed_from + deterministic_delay`와
  `chunk_underrun_rate`는 `es-policy` (layer 8)와 함께 도착한다.
- **`EnvSelection::Subset(Vec<EnvId>)`** — 지금은 `All`과 위의 round-robin
  유도만 있다.
- **observation 도메인의 실제 작업.** W6는 observation tick을 스케줄할
  뿐이며, 렌더링과 Observation IR 파이프라인은 `es-render` / `es-compile`
  패킷의 몫이다.
- **Safety Plane 통합** (§9). `es-safety`는 이웃 패킷에서 진행 중이며, `Env`는
  아직 액션을 clamp하지 않는다. 도착하면 그 훅은 `step` 안, `set_ctrl` 이전의
  호출 하나가 될 것이며 — 선택적이지 않을 것이다 (INV-12).
- **training 도메인 실행** — 스케줄은 되어 있지만 `es-data`로의 인계는 이후
  W의 몫이다.
- **Backend 모델 파라미터 무작위화** — §5의 한계를 참조.
