<!-- Korean translation of docs/design/evaluation-execution.md. The English file is the working copy; regenerate this when it changes. -->

# Evaluation IR 실행 (`es-eval`) — 설계

Spec 참조: §10 (Evaluation IR), §10.2 (문서 구조), §10.3 (지표 정의), §10.4 (결정성과
공정성), §10.5 (산출물), §12.4 (아홉 개 성능 지표), §9.3–§9.4 (envelope violation rate는
1급 지표다), §5.3 (`execution_hash`), §8.6 (chunk underrun), §11.3 (CPU 레퍼런스 계획),
§18.5 (실패 의미론), §28.4 M2 W1, §28.7 게이트 11.

불변식: INV-15 (평가 중에는 augmentation이 꺼진다), INV-12 (어떤 경로도 Safety Plane을
비활성화하지 않는다), INV-17 (새로운 확장 지점 없음).

리뷰 등급: C — 코드보다 먼저 이것을 읽을 것.

## 1. 이 크레이트는 무엇인가

`es_ir::evaluation`은 **선언**이다: 스위트, perturbation, 지표, 수용 기준, 시드.
이 크레이트는 그것을 **실행**한다. 정확히 세 가지만을 소유한다.

1. 각 `PerturbationKind`를 실행 중에 실제로 일어나는 무언가로 바꾸는 것 (`perturb.rs`),
2. 끝난 `Episode`들과 `SafetyCounters`를 `MetricValue`로 바꾸는 것 (`metrics.rs`),
3. 셀 루프, 해시 체인, 그리고 §10.5의 두 산출물 (`runner.rs`).

trait는 추가하지 않는다. `PhysicsBackend`와 `PolicyRuntime`이 이 크레이트가 소비하는 두
확장 지점이며, 둘 다 이미 INV-17로 예약되어 있다.

## 2. 셀 루프

```
for suite in ir.suites                     # a row of the §10.1 table
  for episode in 0 .. ir.episodes.n_episodes
      overrides = plan.apply_at_reset(suite_id, episode)
      env.reset()
      cpu_plan.reset()                               # §2.4: the observation stream ends here
      loop
          inputs  = capture(env.backend().state())      # §2.3
          if plan step-drops this frame: reuse the held observation, age it
          obs     = cpu_plan.run(inputs)                 # §11.3
          if tick % replan == 0: inference.submit(obs, tick)   # section 2.6, packets V17 / T7
          for s in inference.poll(tick)                        # released after `latency` ticks
              buffer.push(policy.infer(s.inputs), s.tick + latency)   # App. B.5
          chunk   = plane_chunk(buffer, feed, tick)      # section 2.6, packet V6b — empty on an underrun
          action  = plane.validate(chunk, obs_age, tick) # INV-12, always runs
          ctrl    = plan.apply_per_step(action)          # delay, backlash, torque noise
          env.step(ctrl)
      until done or max_steps
```

`Env` 하나, `SafetyPlane` 하나, `CpuPlan` 하나가 실행 전체가 아니라 **셀마다** 있다:
plane의 카운터는 그 셀의 `envelope_violation_rate` 분모이며, plan의 `TemporalWindow`
ring들은 스위트 사이로 새어 나가서는 안 되는 에피소드별 스트림이기 때문이다.

`Env::new`는 자신의 backend를 소비하므로, `run`은 `B` 하나가 아니라
`new_backend: impl FnMut() -> B`를 받는다. 이는 패킷이 스케치한 시그니처로부터의
의도적인 이탈이며, 그 이유는 공정성(§10.4)이다: `Env`는 자신의 `RandomizationPlan` 추첨의
키가 되는 env별 에피소드 카운터를 유지한다. 따라서 셀들 사이에 `Env` 하나를 공유하면
`lighting_shift` 행이 `nominal` 행과 다른 기본 씬 집합을 받게 되어 두 표의 행이 더 이상
비교 가능하지 않게 된다. 셀마다 새로운 `Env`를 두면 그 카운터가 0에서 다시 시작한다.

`n_envs`는 1이다. 시뮬레이션 도메인 전체에 걸쳐 셀 루프를 배치하는 것은 이 패킷이
아니라 M2 W2(`round_robin`, §12.2)의 몫이다.

### 2.1 결정성 계약 (§10.4)

> 동일한 `evaluation_hash` + 동일한 `execution_hash` ⇒ 비트 단위로 동일한
> `report.json`.

여기서 그것을 성립시키는 것:

- **모든 perturbation 추첨**은 `EnvRng::new(seed, suite_id, episode_idx, stream)`에서
  나온다 — 그야말로 §10.4의 `TaskRng(seed_base, suite_id, episode_idx, stream)`이며,
  `seed`는 `SeedPlan`에서, `stream = StableId::from_path("es.eval.perturbation.<stream>")`은
  `Perturbation::stream`에서 온다. `EnvRng`는 카운터 기반이다: *n*번째 추첨은 오직 키와
  *n*에만 의존하므로, 그 시퀀스는 정책에도, backend에도, env가 몇 개 있는지에도, 어느
  셀이 먼저 실행되었는지에도 의존하지 않는다. 정책을 바꾼다고 perturbation이 바뀔 수는
  없다. 그것이 이 표 전체의 요지다.
- **리포트에는 벽시계(wall clock)가 없다.** 이 크레이트 안의 유일한 타임스탬프는
  `EvaluationLock::created`이며, 호출자가 공급하고(`RunConfig::created`, 기본값 0)
  `evaluation.lock`에만 쓰인다. `report.json`에는 시간이 전혀 없다.
- **결정적 순회와 순서.** 오직 `BTreeMap`만 사용한다. 에피소드는 `(cell_index, seed)`
  순서로 집계되고, 셀은 `(스위트 선언 순서, MetricSpec::ALL 순서)`로 출력되므로 JSON은
  바이트 단위로 안정적이다.
- 샘플링 경로에는 **전역 RNG도, `f64` 시간 누적도, `std` 초월함수도 없다** —
  `EnvRng::sample`은 이미 `es_math::approx`를 거친다 (§3.4).
- 두 개의 예외 통로는 숨겨지지 않고 lock에 이름이 명시된다: `execution_hash`는 런타임과
  하드웨어 능력(capability)을 포함하고, `evaluation_hash`는 조건을 포함한다.

### 2.2 INV-15 — augmentation 자동 비활성화, 그리고 거부

§10.4는 평가에서 augmentation이 *자동으로 비활성화*된다고 말하며, 계획이 실제로 하는 일이
바로 그것이다: `training_only` `Augment` 노드는 identity pass-through로 lowering되므로
(`docs/design/observation-lowering.md` §3), 여기서는 실행될 수 없고 allow-list 항목도
필요 없다. 거부는 그러고 남는 것을 다룬다.

무엇이든 실행되기 전에, Observation IR은 `training_only`가 **아닌**
`ObservationNode::Augment` — 즉 실제로 실행될 노드 — 를 찾아 스캔된다.
`AugmentationPolicy::AllowList`에 없는 그런 노드는 `EvalError::AugmentationEnabled`다.
그래프는 **다시 쓰이지 않고** 노드도 **제거되지 않는다**: observation 파이프라인을 조용히
편집하면 호출자가 평가했다고 생각하는 것과 다른 `observation_hash`가 되어 버리고, 리포트는
결코 선언된 적 없는 그래프를 증명하게 된다. 작성자는 Observation IR에서 그 노드를 빼거나,
근거와 함께 allow-list를 작성해야 하며, 그 근거는 결국 리포트의 조건(conditions)에
남는다.

allow-list 항목은 노드의 `NodeId`를 10진수(`"7"`)로 매칭한다. `es-ir`에는 노드
레이블이 없으므로(규칙 7: IR에는 UI 타입이 없다; 레이블은 `.eslayout`에 산다), id가
사용 가능한 유일한 안정적 키다. 레이블 사이드카가 연결되면 레이블로 매칭하는 것이 업그레이드
경로이며, allow-list 문자열의 형태는 바뀔 필요가 없다.

### 2.3 Observation 캡처

`CpuPlan`의 입력 버퍼는 `ImageInput` sensor 또는 `StateInput` source의
`StableId`(`Home::Input(id.to_string())`)로 이름이 붙는다. 캡처는 각 이름을 `ModelInfo`에
대해 해석한다:

| plan 입력 | source | 상태 |
|---|---|---|
| `ObservationNode::ImageInput` | 프레임 소스의 바이트, 변환 없이 | `--frames`와 함께 지원됨 |
| id가 `ModelInfo::qpos`에 있는 `StateInput` | 해당 env의 `qpos` 슬라이스 | 지원됨 |
| id가 `ModelInfo::sensor`에 있는 `StateInput` | 해당 env의 `sensordata` 슬라이스 | 지원됨 |
| id가 Task IR의 `ObsSource::JointState { body, dof }`인 `StateInput` | env 0의 앞쪽 `dof`개 관절 위치 | 지원됨 |
| 채널이 `quantity = Velocity`이고 id가 **관절**인 `StateInput` | 그 관절의 첫 dof부터 `dof`개, 즉 `qvel[start..start + dof]` | 지원됨 (패킷 M8/S4e) |
| 채널이 `quantity = Velocity`이고 id가 **body**인 `StateInput` | env 0의 앞쪽 `dof`개 관절 속도 | 지원됨 (패킷 M8/S4e) |
| 채널이 `ObsSource::BodyPose(b)`이고 `b`가 자유 관절이 아닌 `StateInput` | `xpos[row*3..][..3] ‖ xquat[row*4..][..4]`, 쿼터니언은 xyzw (§3.1) | 지원됨 (패킷 M8/S4e) |
| 그 외 | — | 첫 에피소드 이전에 `EvalError::Plan` |

모든 입력은 첫 에피소드 이전에 `input_sources`가 **한 번만** 해석한다. 입력이 이미지인
이유는 다른 것에 매칭되지 않아서가 아니라 *Observation IR*이 `ImageInput`이라고 말하기
때문이다: 이전 규칙("두 맵 어디에도 없는 id는 이미지")은 프레임 소스가 생기는 순간
27,648바이트짜리 카메라 타일을 6원소 관절 상태 버퍼에 먹였고, 그것이 패킷 M5/V3가 데모
자신의 문서에서 마주친 일이다.

`JointState { body, dof }` 행은 그 채널에 대해 가능한 유일한 해석이다: 관절이 아니라 body와
DoF 개수를 지칭하므로 캡처는 앞쪽 `dof`개 위치를 가져간다 — `joint_state::<NJ>`가 Safety
Plane에 먹이기 위해 이미 쓰는 것과 같은 규약이며, `run_episode`가 `NJ`개보다 적게 지닌
모델을 거부하는 이유(§2.5)이기도 하다.

**수량과 body 포즈 (패킷 M8/S4e).** `ObsSource::JointState`는 `GetJointState`가 늘 지녀온
`JointQuantity`를 지닌다. 없으면 `Position`이며, 그것이 이전의 모든 문서가 해시된 정규
형식이므로 커밋된 `task_hash`는 하나도 움직이지 않는다. 수량은 `qpos`와 `sensor` 맵보다
*먼저* 읽히므로, `Position` 채널은 이 패킷 이전과 정확히 같게 해석되고 `Velocity` 채널이
id가 우연히 겹치는 `qpos` 범위로 답해지는 일은 없다. **관절**을 지칭하는 속도 채널은 그
관절의 첫 dof에서 시작해 `dof`개를 가져간다 — 오프셋은 문서가 알 수 없는 유일한 것이고
폭은 모델이 알 수 없는 유일한 것이다. **body**를 지칭하면 행의 앞쪽 `dof`개를 가져가며,
이는 `Joints`의 거울상이다. `JointQuantity::Torque`는 이름으로 거부된다: `StateView`에는
관절 토크 배열이 없다.

`ObsSource::BodyPose(b)`는 `b`의 `xpos`/`xquat` 행을 읽는다. **자유 관절** body는 결코 그
갈래를 타지 않는다: 그 포즈는 `qpos`에 있고 `Qpos` 갈래가 먼저 답하며, 그것이 데모의 큐브
채널을 바이트 단위로 그대로 유지하는 것이다. 쿼터니언은 `StateView`가 문서화한 순서 —
xyzw (§3.1) — 그대로 쓰이고 여기서 재배열되지 않는다. 재배열은 Observation IR 노드의
일이지 캡처의 일이 아니다.

**source id 하나에 버퍼 하나.** `CpuPlan`은 입력 버퍼의 이름을
`Home::Input(id.to_string())`로 짓는다. 그래서 한 source id를 지칭하는 두 채널은 버퍼 하나를
나눠 갖게 되고 `exec::read_input`은 두 `StateInput` 노드에 같은 텐서를 건네게 된다 —
§10.1이 금지하는 조용히 틀린 숫자다. Cross-IR 검사는 그 쌍을 *받아들인다*(`XIR-002`는 한
id의 두 채널을 선언된 타입으로 구분하며, 그 타입의 단위가 곧 `JointQuantity::unit()`이다).
`input_sources`는 로워링이 버퍼 둘을 줄 수 있게 될 때까지 그 쌍을 이름으로 거부한다.
`tests/fixtures/rl/task-reach.toml`의 `joint_vel`이 body `base`가 아니라 블록의 첫 관절
`shoulder_pan`을 지칭하는 이유가 이것이다.

두 새 종류 모두 모델 없는 해석이 없다. 기록된 데이터셋의 `observation.state`는
`qpos ‖ qvel`이지만 후반부가 어디서 시작하는지는 `nq` — 실행된 모델의 성질이지 행의 성질이
아니다 — 이고, `xpos`/`xquat`는 애초에 행에 없다. 둘 다 추측된 오프셋으로 바뀌는 대신
`bake.rs`에서 이름 지어 거부된다.

프레임 소스가 없으면(`Evaluation::run`, 또는 `--frames` 없는 `es eval run`) 이미지
observation은 0으로 채워지는 대신 이름으로 거부된다. 검은 프레임에 대해 정책을 조용히
평가하는 셀은 숫자 하나를 만들어낼 것이고, 이 표에서 틀린 숫자는 표가 없는 것보다
나쁘다.

### 2.4 계획 상태는 에피소드별이다 (P-M2-R1)

`CpuPlan` 하나는 실행마다 컴파일되지만, `TemporalWindow` ring은 *스트림*이며, 스트림은
에피소드가 끝나는 곳에서 끝난다. 그래서 `run_episode`는 `env.reset` 직후에
`CpuPlan::reset()`을 호출하며, 이는 모든 ring을 `compile`이 남겨둔 상태로 다시 채운다
(`docs/design/observation-lowering.md` §9.1). 이것이 없다면 첫 번째 이후 모든 에피소드의
첫 프레임들은 이전 에피소드의 꼬리를 지니게 되고, 셀 2의 첫 에피소드는 셀 1의 것을 지니게
될 것이다 — 그러면 §10.1 표는 스위트가 선언된 순서에 의존하게 될 것이다. 오라클은 윈도우가
있는 observation에 대해 스위트 순서를 뒤집어 모든 셀이 변하지 않음을 단언한다.

이것이 다루지 *않는* 것에 유의하라: `Perturbation` 추첨은 스위트의 **위치**
(`EnvRng::new(seed, cell_index, episode, stream)`)로 키가 매겨지므로, 스위트 순서를
바꾸는 것은 perturbation이 있는 스위트의 추첨을 정당하게 바꾼다. 따라서 순서-무관성
오라클은 perturbation이 없는 스위트 두 개를 쓴다. 스트림을 스위트 *이름*으로 키잉하는
것은 계획 상태에 관한 것이 아니라 §10.4의 `suite_id`에 관한 별개의 질문이다.

### 2.5 `nu != NJ`는 오류다 (P-M2-R7)

`NJ`는 배포의 관절 수이며, 로드된 모델은 이에 동의해야 한다: `run_episode`는
`model.nu == NJ`이고 모델이 최소한 `NJ`개의 `qpos`/`qvel` 항목을 지니지 않는 한
`EvalError::JointMismatch`로 거부한다. 아무것도 브로드캐스트되지 않고 아무것도 패딩되지
않는다 — 관절 `NJ−1`의 복사본으로 채워진 `ctrl` 벡터나, `0.0`으로 패딩된 safety 입력은
§10.1 표에 잘못된 숫자를 만들어내며, 이는 거부된 실행보다 나쁘다.

### 2.6 추론 지연과 청크 버퍼 (패킷 M5/V6b, M5/V17, M7/T7)

세 개의 패킷이 "평가기가 정책을 호출한다"를 "평가기가, 배포 문서가 로봇에서 그렇게
돌아간다고 말하는 그대로 정책을 돌린다"로 바꿨다. 각각 규칙 하나씩이며, 셋 모두
*수집기*의 규칙이다 — 규칙을 다시 적어서가 아니라 수집기의 함수를 호출해서 도달했다:

| 무엇 | 양쪽 경로가 호출하는 `es-env` 함수 | 패킷 |
|---|---|---|
| 청크 하나가 `action.execute_chunk` 틱을 구동한다 | `es_env::plane_chunk` | M5/V6b |
| 정책은 `rate.control / rate.inference`마다 한 번 호출된다 | `es_env::replan_interval` | M5/V17 |
| 청크는 `computed_from + latency`에 실행된다 | `es_env::AsyncInference` + `latency_ticks` | M7/T7 |

세 번째는 §8.6의 정상 케이스다: 제어 주기보다 큰 추론 지연은 오류가 아니므로 런타임이
이를 모델링한다. `AsyncInference`는 벽시계가 아니라 *틱*에 대한 큐이며(§12.3), 틱 `t`의
제출은 `t + latency_ticks(expected_latency_ms, rate.control)`에 방출되고, 그것이 만든
청크는 `apply_at = t + latency`로 `ChunkBuffer`에 들어간다(App. B.5:
`apply_at = computed_from + deterministic latency`이지 결과가 우연히 돌아온 틱이 아니다).
따라서 빠른 호스트와 느린 호스트가 동일하게 재생되고, 수집 경로와 평가 경로도 그렇다.

분명히 적어 둘 결과 셋:

- **모든 에피소드의 틱 0은 청크 언더런이다.** 아직 계산된 것이 없으므로 `plane_chunk`는
  plane에 빈 청크를 건네고, plane은 자신의 fallback과 자신의
  `ViolationKind::ChunkUnderrun`으로 답한다. 이는 우회도 완화도 아니다(INV-12):
  언더런은 plane 자신의 이벤트이고, `chunk_underrun_rate`에 집계되며, `events.json`의
  프레임 0에 `source: "Fallback"`으로 나타난다. `es loop collect`는 M2 이래 정확히 이
  모습이었다.
- **정책은 결과가 적용되는 틱이 아니라 제출된 틱의 관측을 본다.** 실제 파이프라인이 하는
  일이 그것이고, 그래서 청크는 실행될 무렵 `latency` 틱만큼 낡아 있다.
- **`expected_latency_ms = 0`은 계속 합법이며** 0틱을 뜻한다: 제출과 poll이 같은 틱에
  일어나고 루프는 T7 이전과 정확히 같다. 이는 문서의 선택(`RuntimeHints`)이고, 어떤 실제
  로봇도 지킬 수 없는 주장이다.

그 숫자는 Learning IR의 것(`PolicyContract::runtime::expected_latency_ms`)이고,
`Evaluation::run`은 자신이 심판하는 네 IR만 받을 뿐 `LearningGraph`는 결코 받지 않는다 —
`hash_chain`이 `learning`과 `policy`를 로드된 `PolicyInfo`에서 가져오는 것과 같은 이유다.
그래서 `RunConfig::expected_latency_ms`로 건너오며, `es eval run`이 자신이 연 번들에서
채운다. 지연 *모델*은 여전히 정확히 하나다: `es_env::latency_ticks`를 `DomainRunner::new`와
셀 루프가 같은 두 인자로 호출한다.

오라클은 스케줄이 아니라 궤적이다:
`collection_and_evaluation_draw_the_same_trajectory`(`crates/es/tests/cli.rs`)가 한 시드를
`es loop collect`와 `es_eval::Evaluation`으로 각각 돌리고 두 `.estraj` 파일을 마지막 틱까지
비트 단위로 비교한다. T7 이전에는 틱 1에서 실패했다 — 측정된 값과 데모의 숫자가 어떻게
되었는지는 `docs/design/visible-learning.md` 7.30절에 있다.

### 2.7 샤딩: 단위는 셀이고, 왜 에피소드가 아닌가 (패킷 M7/T8)

`es eval run --jobs N`은 **셀**을 N개 워커 프로세스에 라운드로빈으로 분할한다
(`Evaluation::run_shard(.., shard: (index, count))`, 셀 `c`는 샤드 `c % count`). 그리고
`Evaluation::merge`가 셀 순서대로 되돌려 놓고 리포트·판정·해시 체인을 한 번만 계산한다. 분할은
셀 인덱스만으로 결정되므로 스위트가 얼마나 오래 걸리는지, 워커가 몇 개나 실제로 떴는지, 어떤
순회 순서인지에 의존하지 않는다(§3.4). 따라서 스위트가 하나뿐인 평가는 전혀 빨라지지 않는다.
데모의 16 에피소드 `nominal` 스위트는 셀 하나이고 `--jobs`가 아무리 커도 1-wide로 돈다.

패킷 M7/T8은 단위가 `(cell, episode)` 쌍이 될 수 있는지 물었다. `Env` 쪽은 정리되어 들어왔다.

**`Env::seek_episode(env, episode)`** 는 *다음* `reset`이 뽑을 카운터를 설정한다. 그 외에는 아무것도
건드리지 않는다. reset도, 백엔드 호출도, 레코더 상태도, 물리 틱도 건드리지 않는다. 그리고 스텝이
기록된 에피소드가 열려 있는 동안에는 거부되므로, seek이 기록된 에피소드를 조용히 버리는 일은 없다.
§6.3은 모든 reset과 randomization 추출을 `(seed, env, episode, stream)`만으로 키를 잡고,
`MuJoCoCpuBackend::reset`은 상태를 쓰기 전에 `mj_resetData`를 돌린다. 그래서 seek은 곧 재생이다.
측정값(`cargo test -p es-env --test seek -- --ignored`, 오라클 서버, 2026-09-21): 데모 씬과 커밋된
Task IR에서 `k ∈ {1, 3, 7}`에 대해, `k`로 seek한 새 `Env`와 `k`번 reset한 새 `Env`는 `qpos`,
`qvel`, 에피소드의 `ParamScales`, 그리고 전문가가 몬 에피소드의 45,784바이트 `.estraj` 전체가
비트 단위로 일치한다.

**평가기 쪽은 성립하지 않고, 분할은 들어가지 않았다.** 셀이 에피소드 `k-1`에서 `k`로 넘기는 것은
`Env` 외에 다음과 같다. `SafetyPlane`(`begin_episode`는 래치를 지우고 시드를 다시 무장시키지만
§9.4의 슬라이딩 `ViolationRate` 링인 `SafetyCounters::window`는 **지우지 않는다**),
`ChunkBuffer` + `PlaneFeed`와 `AsyncInference`(모두 에피소드마다 초기화됨), 단조 증가하는 `seq`
(순서만 판정됨), 그리고 plane과 env의 카운터(합계이므로 더해진다). 이 중 둘이 산출물에서 보이고,
둘 다 같은 커밋된 데모 문서 위에서 T8 이전 빌드와 T8 빌드의 `es eval run`을 돌려 측정했다 —
nominal 스위트, 시드 101–104, `--frames`, `v14/trained-20000.esb`, 오라클 서버, 2026-09-21
(`~/artifacts/plan-v/m7-t8/run-parity.sh`, `pre-j1/` 대 `post-j1/`).

| 무엇 | 셀 단위 (이전) | 에피소드 단위 (이후) |
|---|---|---|
| 처음 달라지는 `.estraj` | — | `nominal-01`, **틱 24** (`qpos[0]` 0.068231 → 0.071987) |
| `events.json` 첫 차이 | `nominal-01` 레코드 0, `tick: 7200` | 같은 레코드, `tick: 0` |
| `violation.rate` | 2318 | 2091 |
| `fallback` | 5740 | 5678 |
| `violation.position` / `.velocity` / `.acceleration` | 423 / 498 / 1456 | 261 / 421 / 1512 |
| `envelope_violation_rate` | 0.9998611 | 0.9990278 |
| `nominal-00` (에피소드 0) | — | 네 산출물 모두 바이트 단위로 동일 |

독립적인 두 가지 메커니즘이고, 그중 하나만 `es-safety`의 것이다.

1. **`ViolationRate` 윈도가 에피소드 경계를 넘어 살아남는다.** 데모는
   `envelope_violation_rate = { max_frac = 0.9, window = 200 }`을 선언하고 envelope 위반율 약
   0.999로 돈다. 그래서 T8 이전 plane은 에피소드 `k`의 틱 0에서 링이 에피소드 `k-1`의 dirty 스텝으로
   가득 차 있고, 약 1.0을 읽어 §9.4 워치독을 건드린다. 에피소드마다 새로 만든 plane은 빈 링에서
   시작하고, 빈 링은 200 스텝이 찰 때까지 `0.0`을 읽는다. 그래서 처음 200틱은 워치독이 걸리지 않고
   fallback 대신 clamp 경로를 타며, 궤적은 틱 24에서 갈라진다. 네 에피소드에 걸쳐 `violation.rate`
   227개 감소.
2. **`StepEvent::tick`은 셀에 누적되는 물리 클럭**인 `Env::tick()`이다. `nominal-01`이 시작할 때
   7200(제어 스텝 1800 × 서브스텝 4)이고, 에피소드 단위 env에서는 0이다. 이쪽은 plane이 아니라
   `es-eval`이 기록하는 값이다. 윈도를 지워도 `events.json`은 여전히 움직인다.

패킷대로 이 발견이 결론이고, 분할을 켰을 플래그는 존재하지 않는다. 두 질문 모두 M7 리뷰의 몫이다.

- `SafetyPlane::begin_episode`가 `SafetyCounters::window`를 지워야 하는가? 지우는 쪽이 "에피소드는
  스트림이 끝나는 곳"(§13.1, 그리고 §2.4가 observation plan에 대해 펴는 같은 논리)의 올바른 독해일
  수 있고, INV-12 관점에서도 깨끗하다 — envelope, 워치독, 카운터는 그대로 켜져 있고 워치독이 읽는
  링만 다시 무장한다. 다만 위 표의 커밋된 숫자를 전부 움직이므로 수정이 아니라 결정이다.
- `StepEvent::tick`이 env의 틱이 아니라 에피소드의 틱이어야 하는가? 셀의 프레임 `n`은 이미
  `frame: n`을 들고 있다. 절대 틱은 한 셀 안에서만 의미가 있고, 에피소드 상대 틱이면 `events.json`이
  실행이 어떻게 스케줄되었는지와 무관해진다.

둘 다 그쪽으로 가면 분할은 지금 들어간 것 위의 작은 변경이고(`ShardCell`이 에피소드별 레코드를
갖고, `record_cell`이 `merge`로 옮겨 가고, `run_shard`가 `suites × seeds`를 돌며 seek한다), 아래
벽시계 표가 그것이 무엇을 사는지 말해 준다.

#### 벽시계, nominal 스위트, 16 에피소드 (`Target / Status: measured`)

오라클 서버(16코어, RTX 4090), `--frames`, `v14/trained-20000.esb`, 스위트 하나, 시드 101–116.
각 행 옆의 1분 부하 평균은 그 실행이 시작된 순간의 박스 부하이고, 다른 에이전트가 같은 기계를 쓴다.

| 빌드 | `--jobs` | 실제로 뜬 워커 | 벽시계 (2차) | 시작 시 부하 | 1차 |
|---|---|---|---|---|---|
| 셀 단위 (적용됨) | 1 | 1 | 121.6 s | 0.55 | 120.3 s |
| 셀 단위 (적용됨) | 4 | 1 (스위트 수로 clamp) | 117.3 s | 2.31 | 120.0 s |
| 셀 단위 (적용됨) | 8 | 1 (clamp) | 120.6 s | 2.83 | 118.7 s |
| 에피소드 단위 (T8, 미적용) | 1 | 1 | 122.9 s | 2.89 | 124.2 s |
| 에피소드 단위 (T8, 미적용) | 4 | 4 | 60.6 s | 2.71 | 59.5 s |
| 에피소드 단위 (T8, 미적용) | 8 | 8 | 55.2 s | 2.96 | 55.2 s |

두 번 측정했고, 2차에는 각 행 앞에 "1분 부하가 4 아래가 될 때까지 기다린다" 게이트를 넣었다(1차는
세 행을 부하 3.6–6.0에서 시작했는데, 그것은 직전 행이 남긴 감쇠다). 두 측정은 모든 행에서 3초
이내로 일치하고, 그것이 여기서의 노이즈 바닥이다.

이 표가 말하는 것: 적용된 분할은 평평하다. `docs/design/visible-learning.ko.md` 7.11절이 예측한
그대로다 — 스위트 하나짜리 평가는 셀 하나이고 `--jobs`는 1로 clamp되며, 셀 단위 세 행은 노이즈만큼만
다르다. 에피소드 단위 분할은 워커 4개에서 **2.0배**, 8개에서 **2.2배**다. 두 번째 계단이 작은 이유는
박스가 16코어이고, 각 워커의 수학 라이브러리 스레드 풀이 `cores/N`으로 제한되며(7.11절), 워커마다
자기 MuJoCo와 torch 서브프로세스를 열기 때문이다. 에피소드 단위 `--jobs 1`은 셀 단위보다 1초쯤 더
드는데 노이즈 안이다. `Env::new` + `mj_resetData` + plane 생성을 한 번이 아니라 열여섯 번 하는 것이
병렬이 없을 때 에피소드 격리가 치르는 비용이고, 작다.

상금 전체가 숫자 하나다. nominal 평가가 약 5분이 아니라 약 2분이 되고, 데모의 6개 스위트 스윕에서는
셀 분할 위에 같은 2배가 한 번 더 붙는다. 그 대가는 위의 두 리뷰 결정이지 이 패킷이 아니다.

## 3. Perturbation 실현 (`perturb.rs`)

`PerturbationPlan::compile(&EvaluationIr, &SceneDesc, &ModelInfo, has_renderer)`는 어떤
에피소드가 실행되기 전에 모든 스위트의 모든 perturbation을 **한 번만** 해석하므로,
에피소드별 경로에는 문자열 매칭이 없고 실패할 수 없다. 이 런타임이 실현할 수 없는 종류는
컴파일 시점에 그것을 이름으로 지목하는 `EvalError::Unsupported(kind)`다 — 조용히 건너뛰는
일도, 근사하는 일도 결코 없다(`batch-domains.md` §5의 `RandomizationPlan`, §17.2의
`PhysicsBackend::load`와 같은 규칙).

`has_renderer`는 이 실행이 프레임 소스를 받았는가다(`Evaluation::run_with_frames`, 즉
`render` 피처로 빌드한 빌드의 `es eval run --frames`). 두 조명 종류는 그때만 실현
가능하며, 없을 때는 추첨해놓고 버리는 대신 이름으로 거부된다. `model`은 오늘은 쓰이지
않는다; 시그니처에 있는 이유는 아래 "상태 변형" 그룹의 모든 종류가 구현되는 순간 그것에
대해 대상을 해석하게 되기 때문이다.

### 3.1 지금 실현된 것

| 종류 | 방법 | 적용 |
|---|---|---|
| `action_delay` | 제어 벡터의 ring buffer, 깊이는 `round(ms / control_period_ms)`; ring은 reset 액션으로 채워지므로 첫 스텝들은 0이 아니라 hold pose를 명령한다 | 스텝마다 |
| `observation_delay` | 캡처된 observation을 `round(ms / control_period_ms)` 스텝만큼 붙들고 있으며, `SafetyPlane::validate`에 넘겨지는 `obs_age`가 그만큼 커지므로 `StaleObservation` watchdog이 그 지연을 본다 | 스텝마다 |
| `frame_drop` | 스텝마다 베르누이 `prob`; 적중하면 `[lo, hi]`개의 연속 프레임 버스트를 드롭하고, 그동안 이전 observation이 재사용되며 `obs_age`는 계속 커진다 | 스텝마다 |
| `torque_noise` | 각 제어 채널에 곱해지는 `1 + N(0, rel_sigma)`, 스텝마다 채널마다 추첨됨 | 스텝마다 |
| `backlash` | 에피소드별 `[lo, hi]` rad의 데드밴드: 그 밴드보다 작은 명령 변화는 액추에이터를 움직이지 않는다 | 스텝마다 |
| `light_intensity` | `range`에서 추첨한 이득을, `TriScene` 업로드 전에 씬 복사본의 모든 geom `rgba`에 곱한다. `Rs` 경로는 `albedo * (ambient + n.l * (1 - ambient)) + emission`을 쉐이딩하므로 `albedo`에 대해 *선형*이고, 따라서 색을 재는 것은 입사 복사휘도를 재는 것과 정확히 같다. `dist = "uniform"`만 커널이 있고 나머지 둘은 이름으로 거부된다. | 에피소드마다 |
| `light_direction` | `[-range_deg, range_deg]`에서 추첨한 yaw를 `es_math::approx::sin`/`cos`(결코 `std`의 것이 아니다, §3.4)로 `+Z` 둘레의 `RenderConfig::light_dir`에 적용한다 | 에피소드마다 |

두 조명 종류는 `LightOverride { intensity, yaw_deg }`이며, 다른 모든 에피소드별 노브처럼
`apply_at_reset`에서 추첨되어 프레임마다 프레임 소스에 전달된다. *렌더러*는 호출자의
것이므로(`es-eval`은 layer 10이고 Vulkan을 링크하지 않는다, `visible-learning.md` §7.4)
`LightOverride::scene`과 `::rotate_dir`이 커널이고 적용은 호출자가 한다. `es eval run`은
추첨이 바뀔 때만 렌더러를 다시 만들므로, 조명 perturbation이 없는 스위트는 실행 전체에
대해 정확히 하나만 만든다.

`ms` 목록(`observation_delay`, `action_delay`)은 `Choice` 분포다: 에피소드마다 값 하나가
추첨되므로, `ms: [0, 20, 50]`인 셀은 §10.2가 쓴 그대로 세 조건을 자신의 에피소드들에
걸쳐 섞는다.

에피소드마다 고정되는 추첨(`ms`, backlash 밴드)은 `apply_at_reset`에서 일어나
`ResetOverrides`에 담긴다. 스텝마다 일어나는 추첨(`frame_drop`, `torque_noise`)은
runner가 소유하는 `StepState`에 대해 `apply_per_step`에서 일어난다. `ResetOverrides`는
장차 될 hook의 이름을 미리 딴 것이다; 오늘은 상태 오버라이드를 하나도 싣지 않는데,
상태를 변경하는 모든 종류가 §3.2에 있기 때문이다.

### 3.2 오늘 `Unsupported`인 것

| 종류 | 무엇에 막혀 있나 |
|---|---|
| `light_intensity`, `light_direction` | **프레임 소스만 있으면 아무것도 막지 않는다.** 없으면(`Evaluation::run`, 또는 `--frames` 없는 `es eval run`) 흔들 렌더 이미지가 없으므로 그 이유를 들어 거부된다. |
| `color_temperature` | 색이 있는 조명. `Rs` 경로는 흰색 방향광 하나로 쉐이딩하고 `RenderConfig`에는 조명 색이 없으므로 설정할 대상 자체가 없다; 추가하는 것은 `es-render`(layer 5) 변경이다. |
| `camera_extrinsic`, `camera_intrinsic` | 캡처 시점의 `ImageSpec` intrinsics 재작성(INV-14) — `ImageSpec` 변환을 건너뛴 intrinsic perturbation은 카메라에 대한 조용한 거짓말이 될 것이다. 렌더러만으로는 풀리지 않는다. |
| `occluder` | 씬 그래프 삽입: occluder는 Task IR이 선언하지 않은 geom이고, 그것이 들어간 씬은 더 이상 `scene_hash`가 가리키는 씬이 아니다. |
| `object_pose` | 에피소드별 reset 오버라이드. `Env::reset`은 상태를 받지 않고 `Env`가 자신의 backend를 소유하므로, `es-eval`은 스텝 이전에 `qpos`를 쓸 수 없다. 그 hook은 이후 `es-env` 패킷의 `Env::reset_with(&ResetOverrides)`이며, `ResetOverrides`는 이미 그것을 실어 나를 형태로 되어 있다. **데모에는 필요 없다**: Task IR `Randomization`(§6.3)이 이미 모든 스위트의 모든 reset에서 큐브의 free joint를 움직인다(`visible-learning.md` section 2.7). |

12개 중 7개가 실현되었고, 그중 2개는 `--frames`가 있을 때만이다. 따라서 M2 W1의
게이트(§28.4, "Evaluation IR 전 스위트 동작")는 이 패킷들만으로는 여전히 충족되지
**않는다**. 거부는 요란하게 일어나므로, 리포트가 실행하지 않은 `lighting_shift` 행을
주장하는 일은 결코 없다.

## 4. 지표 (`metrics.rs`)

`compute(metric, episodes, counters, env_metrics) -> Measured`이며, `Measured`는
`Value(MetricValue)` 또는 `Unavailable(reason)`이다. **아무것도 지어내지 않는다**: 이
런타임이 측정하지 않는 지표는 이유 문자열과 함께 `Unavailable`이지, 결코 `0.0`이 아니다.

| 지표 (§10.3) | 무엇으로부터 계산되는가 | 상태 |
|---|---|---|
| `success_rate` | 셀에 대한 `Episode::termination == Success` 비율 | 측정됨 |
| `episode_length` | 평균 `Episode::steps()` | 측정됨 |
| `envelope_violation_rate` | 셀의 `SafetyCounters`에서 `(clamped_steps + fallback_activations) / steps`, 1로 상한 | 측정됨 |
| `chunk_underrun_rate` | `SafetyCounters::chunk_underrun_rate()` (§8.6) | 측정됨 |
| `action_smoothness` | 각 에피소드의 제어 궤적에 대한 `1 / (1 + mean |Δctrl| + mean |Δ²ctrl|)`, 평균 | 측정됨 |
| `failure_mode_histogram` | `Episode::termination`과 `Episode::failure` 버킷, 거기에 `SafetyCounters::fallback_activations`에서 나온 `fallback` 버킷과 0이 아닌 `ViolationKind`마다 하나씩의 버킷 | 측정됨 |
| `intervention_rate` | 실기/HIL에서의 사람 개입 (§24.2) | `Unavailable` — 이 빌드에는 HIL 경로가 없음 |
| `collision_rate` | 원치 않는 접촉 | `Unavailable` — `PhysicsBackend`가 아직 접촉을 보고하지 않음 |
| `domain_gap` | 실기 로그 재생 거리 (§24.3) | `Unavailable` — M3 |
| §12.4 성능 집합 | 필드마다 `Option`인 `EnvMetrics`의 통과(pass-through) | 필드가 `None`이면 필드별로 `Unavailable` |

정의에 대한 두 가지 참고.

- **`envelope_violation_rate`는 watchdog의 window가 아니라 셀 전체에 대한 누적값이다.**
  `SafetyCounters::envelope_violation_rate()`는 §9.4의 rate watchdog이 읽는 슬라이딩
  비율이다; §10.3의 지표는 셀 전체의 비율이므로, 대신 누적 카운터로부터 계산된다:
  `counters.dirty_steps / counters.steps`.
  **답변됨 (M2 W1b).** `clamped_steps`와 `fallback_activations`는 예전에는 `validate`의
  서로 다른 분기에서 세어져 합산되고 `steps`로 상한이 걸렸다; 나중에 둘 다를
  증가시키는 분기가 생겼다면 그 중첩분만큼 과소 보고했을 것이다. `SafetyCounters::record_step`
  (모든 `validate` 경로가 돌아가는 그 하나의 꼬리인 `SafetyPlane::finish`에서 한 번
  호출됨)은 이제 `clamped_steps`와/또는 `fallback_activations`를 세팅하고, 둘 중 몇 개가
  참이든 상관없이 `dirty_steps`를 최대 하나만 증가시키므로, 이 지표는 둘 다에 해당하는
  스텝에 대해서도 정확하다 — `es_safety::counters::tests::a_step_that_is_both_clamped_and_a_fallback_counts_once`를
  참고.
- **`step/s`는 없다.** 성능 행은 §12.4의 아홉 개 지표뿐이며 그 외에는 아무것도 없다.

에피소드에 걸친 집계는 `(cell_index, seed)`로 정렬된 `Vec<f64>`에 대한
`Aggregation::{Mean, Min, Max, P95}`다. `P95`는 정렬된 표본에 대한 nearest-rank 순서
통계량이다 — 보간이 없으므로 정확히 재현 가능하다. `mean`, `std`, `ci95`(정규 근사,
`1.96 * std / sqrt(n)`)는 여기서 세 개의 작은 함수로 산다; 두 리포트 사이의 실제 Welch
검정이 필요한 `es eval compare`(layer 11, 이후 패킷)는 이 크레이트보다 위에 있다.

## 5. 수용 기준

각 `AcceptanceCriterion`은 자신이 이름 붙인 스위트들에 대해 전개되며(`suite: None`은
모든 스위트를 뜻한다, §10.2) `Verdict` 하나를 낸다:

- 지표가 측정되었을 때는 `Pass { observed }` / `Fail { observed }`,
- 측정되지 않았을 때는 `Unavailable { reason }`.

**`Unavailable`은 통과가 아니다.** `EvaluationReport::passed`는 모든 `AcceptanceResult`가
`Determined { passed: true, .. }`일 때만 참이다.

> **답변됨 (M2 W1b).** `es_ir::evaluation`은 이제 `MetricValue::Unavailable { reason }`와
> `AcceptanceResult::Unavailable { metric, reason }`를 갖는다(후자는 `AcceptanceResult`를
> 단순 구조체에서 enum으로 바꾸었고, `#[serde(untagged)]`이므로 예전의
> `{criterion, observed, passed}` 형태도 여전히 `Determined` variant로 왕복한다).
> `run`은 `(EvaluationReport, EvaluationLock)`을 직접 반환한다; `EvalReport` 래퍼,
> `Unmeasured`, `Verdict`, `Outcome`은 `es-eval`에서 사라졌다 — 선언된 지표마다
> 하나의 `CellResult`(측정됨 또는 `MetricValue::Unavailable`)를, 수용 기준 한 줄마다
> 하나의 `AcceptanceResult`(`Determined` 또는 `Unavailable`)를 얻으므로, 래퍼가 더할
> 것이 남아 있지 않다.

## 6. 산출물 (§10.5)

```
write_artifacts(&report, &lock, dir)
  → report.json        es_ir::evaluation::EvaluationReport, written as-is (§10.5)
  → evaluation.lock    evaluation_hash + execution_hash + seeds + backend capabilities
```

`report.html`과 `episodes/`(리플레이, `ReplayPolicy`에 따라 실패분 우선)는 **이후
패킷**이다: HTML은 에디터가 이미 렌더링하는 표 레이아웃이 필요하고, 리플레이는 §23의
에피소드 직렬화 포맷이 필요하다. 그래서 `EvaluationReport::episodes`는 존재하지 않는
파일의 경로로 채워지는 대신 비어 있는 채로 쓰인다.

### `report.json`

```jsonc
{                                    // es_ir::evaluation::EvaluationReport, §10.5
  "schema_version": 1,
  "evaluation_hash": [32 bytes],
  "execution_hash":  [32 bytes],
  "cells": [
    { "suite": "nominal", "metric": "success_rate",
      "value": { "scalar": 0.92 }, "n_episodes": 100 },
    { "suite": "nominal", "metric": "collision_rate",
      "value": { "unavailable": { "reason": "no contact reporting in this backend" } },
      "n_episodes": 100 }
  ],
  "acceptance": [
    { "criterion": {...}, "observed": 0.92, "passed": true },
    { "metric": "collision_rate", "reason": "no contact reporting in this backend" }
  ],
  "passed": false,
  "episodes": []
}
```

두 가지 `acceptance` 형태는 `AcceptanceResult::Determined`와 `::Unavailable`이다; 이
enum은 `#[serde(untagged)]`이므로, 한 줄이 어느 것인지는 태그가 아니라 어떤 필드를
가지고 있는지로 읽힌다.

### `evaluation.lock`

TOML이 아니라 JSON이다: `evaluation_hash`가 이미 자신의 전송 보증(transport guarantee)을
위해 의존하고 있는 것과 같은 `float_roundtrip`을 쓰는 `serde_json`이며(`es_ir::evaluation`
모듈 문서), canonical하게 유지해야 할 serialiser가 하나 줄어드는 셈이다.

```jsonc
{
  "schema_version": 1,
  "evaluation_hash": "hex32",
  "execution_hash":  "hex32",
  "seeds": [20260912, ...],          // the resolved per-episode seeds, in order
  "backend": { "name": "mujoco-cpu", "determinism": "Bitwise", "float": "F64",
               "max_envs": 1024, "gpu_resident": false,
               "supports_reset_subset": true, "supports_state_get_set": true,
               "quirks": ["..."] },
  "created": 0                       // caller-supplied unix seconds; 0 = unset
}
```

해시는 lock에서는 16진수(사람이 읽는다)이고 리포트에서는 원시 바이트(`es-ir` 자체의
serde 형태)다. `created`는 두 산출물을 통틀어 유일하게 재현 불가능한 필드이며, 의도적으로
lock 안에만 갇혀 있다.

### `execution_hash` 조립 (§5.3)

`run`은 자신에게 주어진 것으로부터 `HashChain`을 구성한다:

| 슬롯 | 출처 |
|---|---|
| `asset`, `scene` | `TaskIr::scene.asset_hash` / `scene_hash` |
| `task_graph` | `canonical_hash(&task.graph)` |
| `task`, `observation`, `deployment`, `evaluation` | 각 IR 자신의 `*_hash()` |
| `learning`, `policy` | `PolicyInfo::lowering_hash` / `weights_hash` — 그래프 자체는 `run`에 넘겨지지 않으며, 이 둘은 로드된 런타임이 증명할 수 있는 두 다이제스트다 |
| `compiler` | `CpuPlan::compiler_hash()` |
| `runtime` | `PolicyRuntime::runtime_hash()` |
| `dataset`, `hardware` | `RunConfig` — 평가 실행은 어떤 데이터셋도 읽지 않으므로, 호출자가 0들이나 학습 세트의 다이제스트를 공급한다 |

`evaluation`은 체인 안에 있지만 의도적으로 `execution_hash`에는 들어가지 않는다(§5.3:
평가 조건은 무엇이 실행되는지를 바꾸지 않는다); 리포트는 둘 다 싣는다.
