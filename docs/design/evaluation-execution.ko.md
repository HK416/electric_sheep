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
          chunk   = policy.infer(obs)
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
