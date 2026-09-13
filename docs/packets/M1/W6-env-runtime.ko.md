<!-- Korean translation of docs/packets/M1/W6-env-runtime.md. The English file is the working copy; regenerate this when it changes. -->

# W6 — env 런타임 기반 (`es-env`)

Spec: §12 (배치 도메인), 부록 B.5 (스케줄러), §6.3–6.4 (Randomization /
ResetState / Terminate / Reward, 실행 의미론), §18.1 (정수 시간), §18.5
(실패 의미론), §13.1 (에피소드 기록). 설계 노트: `docs/design/batch-domains.md`
(리뷰 등급 C).

## context (범위)

```
crates/es-env/src/lib.rs
crates/es-env/src/scheduler.rs
crates/es-env/src/rng.rs
crates/es-env/src/plan.rs
crates/es-env/src/randomize.rs
crates/es-env/src/episode.rs
crates/es-env/src/env.rs
docs/design/batch-domains.md
docs/packets/M1/W6-env-runtime.md
```

## spec (사양)

- **`scheduler`** — `BatchDomains { simulation, observation, inference,
  training: Option }`, `DomainCfg { batch, period (sim ticks), device }`.
  `Schedule::build`는 §12.1 / B.5를 검증한다 (sim period는 1; inference
  period는 observation period의 정수배; training은 inference의 배수; 배치
  깔때기는 오직 좁아지기만 함). 그리고 하나의 하이퍼주기
  `lcm(obs, inf, train)`를 정적 `Vec<TickPlan>`으로 펼친다. `ticks()`는
  이를 순회하고, `at(tick)`은 이를 인덱싱하며, `observation_envs(tick)`은
  §12.2의 결정적 `round_robin`이다. 순수 데이터 — backend 없음, clock
  없음, `f64` 없음.
- **`rng`** — `EnvRng`, `(seed, env, episode, stream_id)`로 키가 만들어지는
  카운터 기반 splitmix64. `rand` 의존성 없음, 전역 상태 없음(§3.4), env
  사이의 순차적 결합 없음. `es_ir::task::Distribution`의 다섯 variant를
  전부 다룬다. `ln`/`exp`/`cos`/`sqrt`는 `std`가 아니라 `es_math::approx`에서
  온다 (§3.2 `DET-010`).
- **`randomize`** — `RandomizationPlan::compile(task, scene, model)`은
  모든 `ResetState` / `Randomization` 노드의 대상 문자열을 한 번씩
  해석한다. 지원되는 것: `qpos[i]`, `qvel[i]`, `joint.<name>.qpos|qvel`,
  `body.<name>.mass`, `geom.<name>.friction`, `actuator.<name>.gain`. 그
  외의 것과 퇴화된 분포는 그것을 명명하는 `EnvError::Unsupported`가
  된다 — 절대 조용히 건너뛰지 않는다.
- **`plan`** — 각 `Reward` / `Terminate` 싱크로 들어오는 스칼라 원뿔을
  `es_ir::task::Expr`로 낮춘다 (`es-ir`가 이미 노출하는 평가기를 그대로
  쓰며, 두 번째 평가기는 작성하지 않는다). 원뿔에서 지원되는 노드:
  `GetJointState`, `GetSensor`, `GetTime`, `Arith`, `Compare`, `Clamp`.
- **`episode`** — `max_episode_steps`로부터 미리 크기가 정해진 열
  (columnar) 방식의 env별 `Vec`들을 갖는 `EpisodeRecorder`. `finish(env)
  -> Episode`는 tick, qpos, qvel, ctrl, sensordata, reward, done, failure,
  termination, 그리고 추첨된 파라미터 스케일을 싣는다. `Termination`은
  낮춰진 술어들로부터 평가된 다음, 에피소드 예산으로 평가된다.
- **`env`** — `new`, `reset(Option<&[u32]>)`,
  `step(&[f64]) -> StepOutcome { rewards, dones, failures, episodes }`,
  `metrics()`를 갖는 `Env<B: PhysicsBackend>`. 하나의 `step`은 하나의
  제어 스텝 = `inference.period`개의 시뮬레이션 tick이다. 종료된 env는
  같은 호출의 끝에서 자동으로 리셋된다. `EnvMetrics`는 §12.4의 이름들을
  전부 `Option`으로 싣고 있으며 `step/s` 필드는 없다.

제약: `BTreeMap`만 사용, 새 trait 없음 (INV-17), f64 시간 누적 없음, 영어만
사용.

## oracle (오라클)

```
cargo fmt -p es-env --check
cargo clippy -p es-env --all-targets -- -D warnings
cargo test -p es-env
cargo xtask layering
cargo xtask context-budget
```

## acceptance (수용 기준)

- 스케줄러: 좋고 나쁜 `BatchDomains`의 표가 빌드에 성공하거나, 문제가 되는
  도메인을 명명하는 메시지와 함께 거부된다; 계획은 도메인들의 순수 함수다;
  모든 inference tick은 observation tick이기도 하다; round-robin은 한
  cycle마다 모든 env를 정확히 한 번씩 다룬다.
- RNG: proptest — 같은 `(seed, env, episode, stream)`은 항상 같은 시퀀스를
  내며, 어떤 구성 요소를 바꾸든 시퀀스가 바뀐다; 스트림은 다른 곳의
  추첨에 영향받지 않고 개별적으로 주소 지정 가능하다.
- 무작위화: 지원되는 모든 대상이 fixture 씬에 대해 해석된다; 알 수 없는
  대상, 범위를 벗어난 인덱스, 퇴화된 분포는 대상을 명명하는 오류가 된다;
  `ResetState`는 `Randomization`보다 먼저 실행된다; 추첨은 env와 에피소드에
  따라 달라진다.
- Episode: 열 너비는 `steps() * width`다; 짧은 행은 패딩된다; `finish`는
  id를 굴린다(roll).
- Env: 같은 리셋 추첨을 한 두 env는 동일한 100-스텝 궤적을 내고, 다른
  seed는 다른 궤적을 낸다; 같은 seed는 정확히 재생된다; termination은
  자동 리셋을 유발하며 에피소드를 반환한다; 예산은 `Timeout`을 낸다;
  backend가 보고한 발산은 해당 env를 격리하고 보상에서 제외한다; 잘못된
  길이의 `ctrl`은 거부된다.

## forbidden (금지)

- `es-compile`과 `es-safety` (이웃 패킷에서 진행 중) — 의존성으로
  선언되지만 여기서는 의도적으로 사용되지 않는다. Safety Plane 훅은
  `es-safety`와 함께 도착한다 (INV-12: 선택적이지 않을 것이다).
- `crates/es-data`, 그 외 어떤 크레이트든, 루트 `Cargo.toml`, 그리고
  커밋하는 것.
- `es-telemetry` (layer 10)에 의존하는 것: `EnvMetrics`는 그것이 변환하는
  평범한 구조체다.
- 렌더링, Observation IR 파이프라인, 비동기 추론/chunk 버퍼, 그리고
  모델 파라미터 스케일을 backend로 밀어 넣는 것(`PhysicsBackend`에는
  파라미터 API가 없다) — 모두 `docs/design/batch-domains.md` §8에서 미룬
  것으로 나열되어 있다.
