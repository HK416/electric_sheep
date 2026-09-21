# 가져온 정책의 강화학습 이어하기 (plan S, §13.4, §14.4, §28.11)

M8 캠페인을 위한 설계 노트. 스펙 절: §13.4(트레이너의 의미론), §14.4(가져오기), §28.11(사다리),
§8.3/§8.9(노드와 동등성 계층), §9.4(플레인은 롤아웃 중에도 켜져 있다), §19.3(`training/` 슬롯),
§12.4(아홉 개 지표). 패킷: `docs/packets/M8/S*.md`. 한국어 자매 문서: `rl-continuation.ko.md`.

이 노트의 모든 수치를 위한 상태 표시: **측정됨**(서버, 날짜, 경로) 또는
`Target / Status: unverified`.

## 1. 주장, 그리고 네 가지 규칙

프로젝트의 논지는 "다른 곳에서 설계되고 학습된 정책이 같은 의미론으로 여기서 돈다"(§8, §1.9)이다.
M5가 모방학습(LeRobot ACT)으로 그것을 증명했다. Plan S는 같은 논지를 두 번째 정책 계열(PPO MLP)과
두 번째 학습 신호(보상)로 증명하고, 한 걸음 더 나간다: 재현된 정책이 **우리 시뮬레이션에서 계속
학습하고**, 전후의 성공률이 하나의 Evaluation IR 아래 하나의 표에 담긴다.

§28.11이 고정하는 네 가지 규칙을, 이 노트가 구현하는 결정으로 다시 적으면:

1. **PPO는 트레이너이고, IR이 아니다.** 배포되는 그래프는 `Normalize(obs) → StateEncoder{Mlp} →
   PolicyHead{Regression, horizon 1, squash} → Normalizer{Inverse}`뿐이다. 가치 헤드, 로그
   표준편차(log-std), GAE, 옵티마이저, 엔트로피 계수는 `python/es/train_ppo.py`와 `training/`
   안에 산다. 그것들은 `training_hash`를 움직이고, `learning_hash`는 결코 움직이지 않는다.
2. **롤아웃은 `es-env`이고, Safety Plane은 켜져 있다.** 트레이너는 `es_native.Rollout`
   바인딩(S4a)을 통해 우리 `Env`를 밟으며, 자신만의 시뮬레이터를 부르는 일은 결코 없다. 샘플링된
   모든 행동은 액추에이터에 닿기 전에 `SafetyPlane::validate`를 거친다; 실행된 행동, 플레인의
   이벤트 비트, 샘플링된 행동이 모두 기록된다.
3. **어댑터가 선언하고, 코드는 결코 추측하지 않는다.** 관절 순서, 단위, 위치 목표 대 토크, 관측
   배치는 로봇별 어댑터 문서에서 온다. 불일치는 이름을 대는 거절(`IMP-0xx`)이다. Pickle과 orbax는
   학습 경로의 Python에서만 열린다(INV-16).
4. **새 trait 없음**(INV-17). `Rollout`은 기존 타입 위의 pyclass이고, 트레이너는 모듈이다.

## 2. 샘플링 모델 — S2b, S4a, S4b의 모양을 정하는 결정 하나

`lower_to_torch`는 헤드의 스쿼시와 `Normalizer{Inverse}`를 이미 포함하는 `forward(obs) →
action` 하나를 낸다, 그래서 모듈의 출력은 **액추에이터 단위의** 결정론적 행동이다. 그래서
트레이너는 자신의 가우시안을 **그 출력 주위에, 그 단위로** 정의한다:

```
mu   = module(obs)                       # the deployed function, unchanged
a    = mu + exp(log_std) * eps           # log_std: training-only, state-independent, [action_dim]
```

이것은 rsl_rl의 모델이다(행동 공간에서의 가우시안, 분포 안에 스쿼시가 없다). brax에서 가져온
정책은 `tanh`를 `mu` 안에 갖고 있다; `tanh(x)` 주위에서 샘플링하는 것은 `x`를 샘플링하고
스쿼시하는 것과는 brax의 `NormalTanhDistribution`과 다른 분포이고, 그것은 받아들여진다: 가져온
*결정론적* 정책은 정확히 재현되고(S2b), 이어하기는 우리 것인 분포를 갖는 새 학습 실행이다.
임포터는 소스의 log-std(brax의 두 번째 출력 절반, rsl_rl의 `std`)를 `import.json`에 보관해서
`train_ppo.py --init-log-std`가 상수 대신 그것으로부터 시작할 수 있게 한다.

가치 헤드는 Observation IR의 출력 포트들의 결합 위의 별도 MLP이고, `train_ppo.py`에서만 만들어지고
학습된다. 재개를 위해 `training/value.safetensors`에 쓰이며 번들로 패킹되는 일은 결코 없다.

## 3. 롤아웃에서의 지연시간과 청킹

PPO는 매 제어 스텝마다 horizon 1로 행동한다: 청크 버퍼도 없고, 선언된 지연시간도 없다. 롤아웃은
`expected_latency_ms = 0`으로 돌고 플레인은 스텝마다 한 행을 본다. 같은 정책의 **평가**(S4c)는
다른 모든 정책이 채점되는 것과 정확히 같이, 보통의 `es eval run` 경로를 통해 Deployment IR의
선언된 지연시간 아래서 돈다; 두 체제 사이의 차이는 다시 T7의 발견이며, 그것이 평가 수치를
트레이너 자신의 수치가 아니라 정직한 수치로 만드는 것이다. 아래 열린 질문 1.

## 3a. 증분 행동 (`ActionSpace::JointDelta`, packet M9/T1)

Deployment IR은 `action.space = "joint_delta"`를 선언할 수 있다: 정책이 목표 자체가 아니라
현재 관절 목표에 대한 *변화량*을 내보낸다. 이를 적분하는 함수는 하나뿐이고 —
`es_env::chunk_buffer::absolute_target` — 플레인에 행을 건네는 세 소비자(수집의
`DomainRunner::emit_actions`, 평가의 `es_eval::runner::run_episode`, 트레이너의
`Rollout::act`)가 chunk 행과 `validate` 사이에서 그것을 호출한다. 뒤따르는 네 가지가 규칙의
전부다:

* **플레인은 손대지 않는다.** 같은 엔벌로프에 대해, 같은 시그니처로(`INV-13`), 지금처럼
  *절대* 관절 목표를 검증한다. `tests/fixtures/rl/deployment-reach-delta.toml`은
  `deployment-reach.toml`에서 한 단어만 바뀐 문서이고 `safety` 블록은 동일하다.
* **증분은 실행된 값에 더해진다.** 원시 행이 아니다: `prev`는
  `SafetyPlane::last_safe_action()`이며, `observe_state` 뒤 `validate` 앞에서 읽는다.
  에피소드의 첫 tick에서 그 값은 플레인이 명령 사슬을 시드한 측정 자세(§9.3) 그 자체이므로,
  적분기는 그 숫자의 두 번째 사본을 소유하지 않고도 모든 에피소드 경계에서 재설정된다.
  따라서 클램프된 증분이 팔이 결코 도달할 수 없는 목표로 누적될 수 없다 — 이 규칙이 막으려는
  실패가 바로 그것이다.
* **단위는 증분이다.** 증분 정책의 `Normalizer { Inverse }` 통계는 제어 tick당 rad이므로
  행동 포트는 `Unit::AngularVelocity`다. `es_ir::cross`는 거기 `Unit::Angle`이 오면 이름을
  불러 거절한다(`XIR-031`). `JointDelta` deployment에 절대 목표 단위가 오는 것이야말로
  런타임이 위치에 위치를 더하게 만드는 유일한 실수다.
* **없으면 `JointPosition`이고**, 커밋된 모든 문서의 `deployment_hash`는 움직이지 않는다
  (`cargo test -p es-ir committed_deployment_hashes_are_unmoved_by_joint_delta`). deployment
  hash가 판별자를 쓰기 때문에 `JointDelta`는 두 `ActionSpace` enum 모두에서 마지막이다.

PPO 이어붙이기에서 증분이 가질 만한 이유: 정책의 잡음이 목표가 아니라 변화량에 걸리므로,
`action_rate` 감시견이 재는 tick당 움직임이 팔의 현재 위치와 훈련되지 않은 네트워크의 추측
사이의 거리가 아니라 정책 자신의 출력이 된다. 그 대가로 엔벌로프가 넓어지는 것은 아무것도
없다 — §28.12 규칙 1의 요점이다.

감추지 않고 적는 비용: 버퍼를 쓰는 경로에서 증분 갈래는 제어 tick당 한 행을 새 `seq`로
플레인에 건넨다. 플레인이 이미 수락한 행은 그 tick의 실행값에 대해 다시 적분할 수 없기
때문이다. 매 tick 새 `seq`는 매 tick `last_chunk_tick`을 찍으므로 증분 정책에서는
`ViolationKind::InferenceDeadline`이 발화할 수 없다. 죽은 정책은 재계획 구간 하나만큼 뒤에
`ChunkUnderrun`과 같은 폴백으로 여전히 잡힌다. 감시견을 되사려면 신선도를 다시 찍지 않고
행을 갱신할 수 있는 플레인이 필요한데, 플레인은 T1의 범위 밖이었다(`INV-11..13`).

## 4. 소스 프레임워크별 오라클 계층 (S2b)

| 소스 | 무엇을 비교하는가 | 계층 |
|---|---|---|
| rsl_rl(`.pt`), rl_games(`.pth`) | 소스의 torch 액터 대 우리 런타임(로워링된 모듈 위의 `TorchRuntime`), 무작위 관측 1,000개, f32 | **비트 단위**(§3.5 계층 1) |
| brax / MuJoCo Playground(orbax 또는 `source.npz`) | (a) 임포터의 numpy→torch 재구성 대 우리 런타임: **비트 단위**; (b) JAX의 결정론적 `tanh(loc)` 대 우리 런타임 | (a) 비트 단위; (b) §8.9 계층 4, 최대 절대오차 ≤ 1e-5, 기록됨 |

JAX와 torch는 `exp`/`tanh` 구현을 공유하지 않으므로, (b)는 구조적으로 비트 단위일 수 없다; (a)는
*가져오기*가 아무것도 잃지 않았음을 보이는 것이고, (b)는 *프레임워크*들이 스펙이 허용하는 계층까지
일치함을 보이는 것이다.

## 5. S2c와 S4b가 공유하는 reach 작업

작업 정의 하나를, 소스 트레이너(brax, S2c)와 우리 트레이너(S4b)가 쓰고 하나의 Evaluation
IR(S4c)이 채점한다. 커밋된 SO-101 장면과 그 에피소드별 큐브 무작위화를 재사용하므로, 새 무작위화
메커니즘이 필요 없다:

- 장면: `tests/fixtures/mjcf/so101_pick_place.xml`, 200 Hz 물리 위의 50 Hz 제어(데모의 V11
  카덴스, `n_substeps = 4`);
- 관측(26): `joint_pos[6] ‖ joint_vel[6] ‖ cube_pose[7] ‖ gripper_pose[7]`, 각 포즈는 몸체의 월드
  프레임 `pos[3] ‖ quat[4]`(쿼터니언 xyzw, §3.1; MuJoCo의 `xquat`은 wxyz라 원본 쪽이 재배열), 그리퍼 =
  몸체 `gripper`. 관측 IR이 상태 포트를 자르지 못하고(`ChannelSelect`는 로워링되지 않는다) Task IR의
  관측 캡처가 데모의 `sim_cube_pose`처럼 `GetBodyPose` 채널을 7로 묶기 때문에 포즈를 통째로 싣고,
  뺄셈은 네트워크가 배운다. 2026-09-21 15차원 차분 배치에서 개정;
- 행동(6): 위치 목표, 각 액추에이터의 `ctrlrange` 위에서 `[-1, 1]`로 정규화됨
  (`Normalizer{Inverse, MeanStd}`, `mean = centre`, `std = half-range`);
- 보상: 스텝마다 `−‖cube_pos − gripper_pos‖`, 성공 시 `+1`;
- 성공: 거리 `< 0.03` m; 타임아웃 200 제어 스텝.

**2026-09-21 S4e가 다시 개정:** 이제 관측 채널은 자신이 나르는 관절 *수량*을 말한다.
`joint_vel`은 `JointState { body = shoulder_pan, dof = 6, quantity = Velocity }`이고
`gripper_pose`는 `BodyPose(gripper)`이며, `es-eval`의 유일한 캡처 경로가 백엔드 자신의
`qvel`과 `xpos ‖ xquat`에서 둘 다 낸다(`docs/design/evaluation-execution.md` 2.3). body
`base`가 아니라 블록의 첫 *관절*을 지칭하는 이유는 `CpuPlan`이 source id마다 입력 버퍼를
하나만 잡기 때문이다: `base`를 지칭하는 두 채널은 같은 여섯 숫자를 받게 된다. IR들은 그
쌍을 받아들이지만 로워링이 아직 못 내므로, `input_sources`가 틀리게 내는 대신 이름으로
거부한다. 그것을 닫는 일은 `es-compile` 패킷이지 이 패킷이 아니다.

Task IR(`tests/fixtures/rl/task-reach.toml`)은 기존 노드(`GetBodyPose`, `Arith`, `Norm`,
`Compare`, `Reward`, `Terminate`)로 이것을 적는다. **S4b가 찾았다(2026-09-21):** `es-env`의 보상·종료 콘은 `GetBodyPose`도 `Norm`도 실행하지 않았고 `Expr`에는 제곱근이 없어, 보상은 적을 수는 있어도 실행할 수는 없었다. 패킷 **S4d**가 콘을 넓히고(`Source::Xpos` 레인, 레인별 `Arith`, `Norm{L2}` → `Expr::Sqrt` — IEEE 기본 연산이지 `DET-010`의 초월함수가 아니다, §6.6) reach 문서 네 개를 소유한다; S4b는 커밋된 데모 문서로 트레이너를 싣고 reach 오라클은 S4d를 기다린다. brax env가 벗어나야 한다면(`implicitfast`/
`elliptic`/`condim 6`에 대한 MJX 지원), 그 벗어남은 `docs/api-notes/brax-ppo-so101.md`의 이름
붙은 행이고 S4c 표가 재는 sim-to-sim 간극의 일부다.

## 6. `training/`이 얻는 것

- `init.lock`(S1): 소스 `policy_hash`, `learning_hash`, `copied`, `initialised`,
  `shape_mismatch` 목록. `[init]`이 없으면 부재하므로, 기존 레시피는 자기 해시를 유지한다.
- `config.json`은 `[rl]` 테이블을 그대로 나른다(S4b); `dataset.lock`은 RL 실행에서 `unset`을
  읽는다(§28.10 규칙 2: 실제 값 또는 `unset`, 결코 날조하지 않는다).
- `metrics/loss-curve.json`은 반복마다 `return`, `episode_len`, `envelope_violation_rate`,
  `executed_ne_sampled_rate`를 얻는다(S4b); 스트림 5가 정책 손실을 날라서 에디터의 Live 탭이
  그것을 그대로 그린다(E7).

## 7. 측정됨

*(패킷들이 채운다; 모든 행은 서버, 날짜, 경로를 이름 짓는다)*

### S4b — PPO 트레이너, 오라클 서버(Linux, 16코어 CPU), 2026-09-21

산출물: `~/artifacts/plan-s/s4b/` (`untrained.esb`, `run.toml`, `run/`).
인터프리터: `~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129, mujoco 3.13.0.
문서: `tests/fixtures/visible-learning/task.toml` + `tests/fixtures/rl/`
{`observation-state.toml`, `learning-state.toml`, `deployment-rl.toml`}, 레시피
`tests/fixtures/rl/training-rl-demo.toml`.

**경로는 돌고, 재현 가능하다.** `envs = 8`, `horizon = 64`, 200 반복, `seed = 0`, CPU 백엔드:

| | |
|---|---|
| 벽시계 | **20.8초** 전체(`es train`), 트레이너 내부 18.4초 |
| 반복당 | 92 ms (512행 = env 8 x 제어 스텝 64) |
| 제어 틱 | 12,800 |
| `identity_hash` | `c07c90aa09b48000…` |
| `training_hash` | `40da99eaa163a3c7…` |
| `dataset.lock` | `{"unset": true}` |

`Rollout.metrics()`가 주는 그대로의 §12.4 아홉 지표(`metrics/env-metrics.json`). 이 경로가
돌리지 않는 도메인은 지어낸 0이 아니라 `null`이고, 단일 `step/s`는 의도적으로 없다.

| 지표 | 값 |
|---|---|
| `physics_steps_per_sec` | 45,713 |
| `actions_per_sec` | 11,428 |
| `camera_frames_per_sec` | `null` — 이 경로에 렌더러 없음(§4.3) |
| `pixels_per_sec` | `null` — 위와 같음 |
| `observation_gb_per_sec` | `null` — `Env`가 계측하지 않음 |
| `policy_inferences_per_sec` | `null` — 추론은 `Env`가 아니라 트레이너 안에 있음 |
| `p50_end_to_end_latency` | `null` — 동기 롤아웃, 선언된 지연시간 없음(3절) |
| `p95_end_to_end_latency` | `null` — 위와 같음 |
| `gpu_memory_peak` | `null` — CPU 백엔드 |
| `chunk_underrun_rate` | `null` — horizon 1, 청크 버퍼 없음 |

나머지는 모두 `Target / Status: unverified`.

**비평가는 배우고, 행위자는 여기서 배울 것이 없다.** `value_loss`는 200 반복에 걸쳐 72.0 → 10.5로
떨어지고(반복 0 / 49 / 99 / 149 / 199: 72.0, 43.6, 29.7, 15.7, 10.5) `return`은 움직이지
않는다(−42.98 → −46.39, 구간 합의 잡음 범위 안). 이는 기대된 결과이지 트레이너의 결함이 아니다.
시험 대상 문서는 **데모** 작업이고, 그 보상은 큐브 x 위치의 `Normalize`인데, 무작위 자세에서
64 제어 스텝 동안 6자유도 팔이 하는 어떤 일도 그 큐브를 옮기지 못한다. PPO가 실제로 개선할 수
있는 작업은 reach 작업이고, 그것은 S4d의 몫이다 — 아래를 보라.

**플레인은 매 틱을 클램프한다.** `envelope_violation_rate`와 `executed_ne_sampled_rate`가 200
반복 전부에서 **1.00**으로 읽힌다. 이것이 이 측정의 발견이며 각주가 아니다.

* 이상 현상이 아니라 구조적이다. 행동은 관절 **위치 목표**이고, 플레인은 그 목표가 제어 틱마다
  측정된 관절에서 얼마나 멀어질 수 있는지를 제한한다(50 Hz에서 속도 3.0 rad/s, 가속도
  80 rad/s²). 학습되지 않은 신경망 출력 주위의 가우시안은 팔이 근처에도 없는 자세를 명령하므로,
  매 틱이 구성상 클램프된다. 데모 자신의 `deployment.toml` 헤더가 V0에서 이 워치독을 0.05에서
  0.9로 넓힌 이유와 같다.
* 그래서 `deployment-rl.toml`이 `max_frac`을 0.9 → 1.0으로 넓힌다. 0.9에서는 워치독이 반복 0의
  첫 몇 초 안에 폴백을 래치하고, 이후 모든 반복은 자기 자신이 아니라 붙들린 팔에 대해 최적화하게
  된다. 엔벨로프를 넓히는 것은 허용된 수이고 플레인을 끄는 것은 아니다(INV-12). 모든 클램프는
  여전히 세어지고 여전히 보고된다.
* 열린 질문 2를 가설이 아니라 구체적인 것으로 만든다. **1.00에서 정책은 env가 실행한 적 없는
  행동의 로그확률만으로 학습된다.** 대신 실행된 행동으로 학습할지는 이제 측정된 숫자가 뒷받침하는
  질문이고, 보상이 움직이는 작업이 생기는 즉시 가장 먼저 제거 실험할 대상이다.

**오라클.** 1(`train_rl_dry_run_plan`, 계획 골든과 다섯 거절)은 어디서나 통과한다. 2
(`train_rl_two_runs_are_bitwise`, `envs = 4`, `horizon = 16`, 3 반복, 두 번)와 3
(`train_rl_init_from_import`, 반복 0이 `[init] policy`와 텐서 단위로 일치)은 서버에서
`ES_PYTHON` 아래 통과한다. 그 과정에서 찾은 결함 둘은 테스트를 덮는 대신 트레이너에서 고쳤다.

* `samples_per_sec`는 `training_hash`로 가는 길에 `metrics.json`에서 빠진다. 그것은 기계에 대한
  측정이고, 남겨 두면 같은 레시피의 두 실행이 두 `training_hash`를 만들었다. §3.5 계층 1이 금지하는
  바로 그것이다. 디스크의 `metrics/loss-curve.json`에는 그대로 남는다.
* 트레이너 요약은 가치 파일을 썼는지와 init 가중치를 읽었는지를 *여부*로 보고하고 *위치*로 보고하지
  않는다. 거기에 절대 경로가 있으면 그것은 호출자가 고른 출력 디렉터리이고, 해시에 같은 영향을
  준다. (`train_act.py`에도 같은 잠복 문제가 있다. 거기서는 잡는 테스트가 없고, 고치는 것은 이
  패킷의 범위가 아니다.)

**오라클 4는 S4d로 미뤘고, 위의 S4e 행에서 측정되었다.** 5절의 reach 작업은 보상 cone 안에서 `GetBodyPose`와 `Norm`을
필요로 하는데, `es-env`의 `ScalarPlan`은 둘 다 lowering하지 않는다. 그것이 lowering하는 것은
`GetJointState`, `GetSensor`, `GetTime`, `Arith`, `Compare`, `Normalize`, `Logic`, `Clamp`이고,
모든 잎은 스칼라 하나를 바인딩하며, `es_ir_types::Expr`에는 설계상 제곱근이 없다(그 문서가 §6.6
`DET-010`을 인용한다). 따라서 −‖cube_pos − gripper_pos‖는 `es-env`와 무관하게 lowering해 들어갈
형태 자체가 없다. 패킷 **S4d**가 그 cone 확장과 네 개의 `*-reach.toml` 문서를 소유하며, 이 표의
성공률 행은 거기서 쓰인다. 위에서 측정된 것은 기반 구조다 — 레시피, 경로, 트레이너, 플레인,
재현성 — 이미 실행되는 문서 위에서.

### S4e — reach 작업이 학습되고 채점되다, 오라클 서버(Linux, 16코어 CPU), 2026-09-21

산출물: `~/artifacts/plan-s/s4e/` (`run-4000/`, `run-10000/`, 각각 `eval-<mark>/` 포함).
인터프리터: `~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129, mujoco 3.13.0.
문서: `tests/fixtures/rl/` {`task-reach.toml`, `observation-reach.toml`,
`learning-reach.toml`, `deployment-reach.toml`, `evaluation-reach.toml`}, 레시피
`tests/fixtures/rl/training-reach.toml`, 번들 `runs/reach-001/untrained.esb`
(`task_hash b5d3b813…`, `observation_hash 4ced8547…`, `learning_hash eb805f18…`,
`deployment_hash 7af05d88…`, `lowering_hash dce8d352…`).

**이것이 S4b가 미뤄둔 오라클 4이고, reach 작업은 학습된다.** `envs = 16`, `horizon = 64`,
`seed = 0`, CPU 백엔드; `evaluation-reach.toml`에 대해 `es eval run`이 채점, 홀드아웃 시드
201–216 16개, `nominal`:

| 예산(반복) | `success_rate` | `episode_length` | 롤아웃 `return` | 롤아웃 `entropy` |
|---|---|---|---|---|
| 1,000 | 0.0000 | 200.0 | −4.38 | 5.48 |
| 2,000 | 0.1875 | 171.0 | −4.68 | 5.31 |
| 2,500 | 0.2500 | 168.9 | −3.73 | 4.93 |
| 3,000 | 0.4375 | 136.6 | −3.64 | 4.63 |
| **4,000** | **0.5625** | **129.4** | −4.74 | 4.87 |
| 5,000 | 0.5000 | 135.0 | −3.82 | 5.04 |
| 7,500 | 0.3125 | 149.0 | −4.84 | 6.00 |
| 10,000 | 0.3125 | 157.3 | −4.92 | 6.06 |

**수용 기준은 충족되지 않았고, 그 이유는 예산이 아니다.** 0.8에는 한 번도 닿지 않았다.
4,000 반복에서의 0.5625가 정점이고, 그 너머에서 실행은 *퇴화한다* — 같은 레시피가 10,000
반복에서 0.3125를 받는데, 이는 2,500 반복 때보다 겨우 나은 수치이며, 롤아웃 엔트로피는
출발점보다도 높이 올라간다(반복 1에서 5.52, 4,000에서 4.87, 10,000에서 6.06). 예산이
늘어나는 동안 잊어가는 정책은 반복이 모자란 것이 아니다. 이 표가 제안하는 순서대로 세 가지를
어블레이션해야 한다: 상수 학습률(`schedule = "constant"` 내내), 엔트로피 계수(0.005 —
올라가는 엔트로피는 그 대가다), 그리고 아래 열린 질문 2 — **모든 틱이 클램프된다**, 그래서
모든 기울기가 env가 결코 실행하지 않은 행동의 로그 확률로부터 계산된다.

예산을 둘 돌린 이유는 첫 번째의 곡선이 끝에서도 오르고 있었기 때문이다: 4,000 반복(벽시계
12.6분, `training_hash 1933697d…`)과 10,000 반복(29.3분, `training_hash 69665845…`). 둘은
두 곡선이 아니라 하나다: 같은 시드에서 긴 실행의 반복 4,000이 짧은 실행의 마지막과 같은
`return`(−4.7397)과 `entropy`(4.8692)를 보고하므로, 위 여덟 행은 서로 끼워 맞춰진다.
커밋된 레시피는 측정된 정점인 4,000을 이름 짓는다.

`Rollout.metrics()`가 주는 대로의 §12.4 아홉 지표(`metrics/env-metrics.json`, 10,000 반복
실행). 이 경로가 결코 돌리지 않는 도메인은 날조된 0이 아니라 `null`이고, `step/s`는 일부러
없다:

| 지표 | 값 |
|---|---|
| `physics_steps_per_sec` | 32,076 |
| `actions_per_sec` | 8,019 |
| `camera_frames_per_sec` | `null` — 이 경로에 렌더러 없음(§4.3) |
| `pixels_per_sec` | `null` — 같음 |
| `observation_gb_per_sec` | `null` — `Env`가 계측하지 않음 |
| `policy_inferences_per_sec` | `null` — 추론은 `Env`가 아니라 트레이너 안에 있음 |
| `p50_end_to_end_latency` | `null` — 동기 롤아웃, 선언된 지연시간 없음(3절) |
| `p95_end_to_end_latency` | `null` — 같음 |
| `gpu_memory_peak` | `null` — CPU 백엔드 |
| `chunk_underrun_rate` | `null` — horizon 1, 청크 버퍼 없음 |

나머지는 모두 `Target / Status: unverified`. 10,000 반복 실행은 1,759.9초에 제어 틱 640,000회,
4,000 반복 실행은 755.2초에 256,000회.

**정점 체크포인트(4,000)에서의 교란 스위트** (게이트가 아니라 측정이다, §10.4):

| 스위트 | `success_rate` | `episode_length` |
|---|---|---|
| `nominal` | 0.5625 | 129.4 |
| `observation_delay` (20 ms, 40 ms) | 0.3125 | 170.3 |
| `torque_noise` (5 %) | 0.5000 | 127.6 |
| `backlash` (0–0.01 rad) | 0.5000 | 130.6 |

제어 한 스텝의 지연이 성공의 4분의 1을 앗아가고, 액추에이터 쪽 두 스위트는 16개 중 하나를
앗아간다. 그 순서는 50 Hz에서 관절 각도와 포즈 둘을 읽는 정책이 느껴야 할 바로 그것이며,
이 노트에서 트레이너가 아니라 *작업*에 관한 첫 번째 행이다.

**플레인은 여전히 모든 틱을 클램프한다.** `envelope_violation_rate`와
`executed_ne_sampled_rate`는 두 실행의 모든 반복에서, 그리고 모든 평가 셀에서 1.00을
읽는다 — S4b가 데모 작업에서 측정한 것과 정확히 같으며, 이제는 보상이 움직이고 정책이
실증적으로 학습하는 작업 위에서다. 그러니 그것은 평평한 보상의 증상이 아니다: 위치 목표
헤드 주위의 가우시안이 틱마다의 속도·가속도 봉투에 맞설 때 벌어지는 일이다. 열린 질문 2는
이제 나중이 아니라 *첫 번째로* 어블레이션할 것이다.

**오라클.** 1(`committed_task_hashes_are_unmoved_by_joint_quantity`)과
2(`capture_reads_qvel_and_body_pose`)는 어디서나 통과한다;
3(`rollout_observes_the_reach_documents`)은 `ES_PYTHON`에 MuJoCo가 있는 곳에서 통과한다 —
26폭 포트가 백엔드 자신의 `qpos[0..6]`, `qvel[0..6]`, 큐브 `qpos[6..13]`, 그리퍼
`xpos ‖ xquat`과 레인별로 비트까지 같다; 4(`reach_documents_validate`,
`train_reach_dry_run_plan`)는 어디서나 통과한다. S4a의 커밋된 롤아웃 골든
(`tests/golden/rollout/so101_100steps.json`)은 움직이지 않았고, 이는 기존 채널의 캡처가
하나도 움직이지 않았다는 가장 강한 진술이다.

### S2b — `es policy import-rl`, 오라클 서버 `renderer-14`(Linux, 16코어 CPU), 2026-09-21

산출물: `~/artifacts/plan-s/s2b/` (`brax/`, `rsl-rl/`, `rl-games/`). 소스 체크포인트:
`~/artifacts/plan-s/s2c/seed0-run1/` (`source.npz`, `meta.json`, `oracle-1000.npz`).
인터프리터: 오라클은 `~/venvs/es-lerobot-cuda/bin/python`(torch 2.11.0+cu129), 두 프레임워크
체크포인트는 `~/venvs/es-rl-import/bin/python`(torch 2.14.0+cpu, numpy 2.5.3,
**rsl-rl-lib 5.5.1**, **rl-games 1.6.5**). 문서:
`tests/fixtures/rl/{task-reach,deployment-reach,adapter-so101}.toml`, `task_hash`
`967ea2961d65f931…` — S4d가 커밋한 그대로의 reach Task IR이며 이 패킷은 건드리지 않았다.

**S2c 정책은 임포트되고, 임포트는 아무것도 잃지 않는다.** 26 → [256, 256] → 6, swish, `tanh`:

| 계층(`4절`) | 무엇을 비교하는가 | 측정 |
|---|---|---|
| (a) | 임포터의 numpy → torch 재구성 대 우리 런타임, 1,000 × 6 값 | **비트 동일** — 불일치 0개 |
| (b) | 우리 런타임 대 JAX의 결정적 `tanh(loc)`, `obs_scaled` 위에서 | 최대 절대오차 **9.704e-7**(허용 1e-5) |

| 슬롯 | 해시 |
|---|---|
| `weights_hash` | `37fc82a8811f07523c6de9dfcacf8220884c4e5b60189298568992d26acda6c5` |
| `observation_hash` | `fe391bef976df15ae747922463ad855e8eb1c4447f8d08aeefee8d76a7b7bb73` |
| `learning_hash` | `f5053f12c9227a658aecdf9f911a555616a0ff7761076572c1978434a5dd99d3` |
| `policy_hash` | `1a18cc4bccb49aa48169b4ec8aa2f966411b461d48b28766637bf11ca5b0cd60` |
| `deployment_hash` | `7af05d88891f5fe6e46717c29ccb46b97e917567e1860372f10301eb1c6fafe4` |

계층 (b)는 패킷의 `U(−1, 1)` 추출이 아니라 `obs_scaled` 위에서 돈다. `brax-ppo-so101.md` 6절을
따른 것으로, 스케일을 벗어난 입력은 좁은 채널을 지나며 모든 `swish`를 포화시키고 거기서는 어떤
올바른 임포터도 1e-5를 통과할 수 없다. 균등 집합은 게이트가 아니라 포화 탐침으로 남는다.

`observation_hash`는 `observation-reach.toml`의 것이 **아니다**. 임포터는 brax 자신의 러닝
통계를 나르는 `Normalize{MeanStd}`를 내보내고, 커밋된 문서는 항등 `Range{−1, 1}`을 나른다. 둘
다 같은 `task_hash`의 관측이며, 이것이 §7.4가 말하는 "여러 Observation IR이 하나의 Task IR을
공유한다"이다. 동시에 임포트된 정책을 자기가 정규화되지 않은 문서로 그냥 채점할 수 없는 이유이기도
하다.

**`rsl_rl`과 `rl_games`는 자기 체크포인트 위에서 비트 동일하다.** 프레임워크 자신의 클래스가
만들고 프레임워크가 저장하는 방식으로 저장한 무작위 가중치 액터(`import_rl.py --synth
<framework> --native`), 26 → [8, 8] → 6, elu, squash 없음:

| 프레임워크 | 버전 | 프레임워크 자신의 forward 대 재구성 | 재구성 대 우리 런타임 |
|---|---|---|---|
| `rsl_rl`(`actor_state_dict` 아래 `MLPModel`) | 5.5.1 | **비트 동일**, 최대 절대 0.0 | **비트 동일**, 6,000개 중 0개 |
| `rl_games`(`model` 아래 `a2c_network`) | 1.6.5 | **비트 동일**, 최대 절대 0.0 | **비트 동일**, 6,000개 중 0개 |

그 대가로 얻은 API 사실 두 가지. 둘 다 이제 `import_rl.py`의 독스트링에 있고 둘 다 패킷 초안에는
없었다. **rsl-rl ≥ 5.0은 액터의 `nn.Sequential`을 `actor`가 아니라 `mlp`로 이름 짓는다**
(`MLPModel.mlp`, `rsl_rl/models/mlp_model.py`; std는 `distribution.std_param` /
`distribution.log_std_param`으로 옮겨갔다). 그리고 `rl_games`의 `BaseModel.build`는 키워드가
아니라 **설정 딕셔너리 하나**를 받는다(`rl_games/algos_torch/models.py:29`). 두 레이아웃 모두
읽으며, 5.0 이전의 `actor.<i>` / 최상위 `std` 형태도 여전히 받아들인다.

**매핑 보고서는 실제 체크포인트에서 제 값을 한다.** `cube_pose`가 `severity = warning`으로
돌아온다. 큐브의 z는 `obs_std`가 1.79e-4이고 이웃들은 ~1.0인데, E4가 큐브 높이를 건드리지 않아
학습 내내 움직이지 않았기 때문이다. 이것은 brax의 1e-6 분산 바닥이 아니며(이 실행은 거기까지
가지 않았다), 그래서 검사는 하나의 숫자에 맞춘 임계값이 아니라 상대적이다(가장 넓은 채널보다
1,000배 넘게 좁은 성분).

**CI 오라클에는 Python이 필요 없다.** `import_rl_synthetic_three_frameworks`(es)와
`import_rl_refusals`(es-data)는 각 수 KB의 커밋된 픽스처 셋 위에서 돈다. 이 픽스처는 각
프레임워크의 네이티브 레이아웃에서 `import_rl.py --synth`가 생성한 것이고, 셋은 **바이트 동일한
리맵 가중치**를 낸다. 세 리더를 직접 돌리지 않고 Rust 쪽이 그들에 대해 말할 수 있는 유일한
것이다. 첫 번째 안의 `TorchRuntime` 열기는 `ES_PYTHON`이 없으면 이유를 찍고 건너뛴다.

### S4c — 이어붙이기 전과 후, 오라클 서버 `renderer-14`(Linux, 16코어 CPU), 2026-09-21

산출물: `~/artifacts/plan-s/s4c/` (`imported/`, `continued-seed{0,1,2}/`, `scratch-seed{0,1,2}/`,
`scratch64-seed{1,2}/`, 각각 자기 `eval/`을 가진다. 그리고 `logs/`와
`compare-imported-continued-seed0.txt`). 트리는 `~/Projects/es-s4c`의 `a977255` 더하기 이 패킷의
레시피 두 개, `cargo build --release -p es`, `es_native`는 이 트리에 맞게 다시 빌드했다.
인터프리터 `~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129, mujoco 3.13.0. 소스
체크포인트는 `~/artifacts/plan-s/s2c/seed0-run1/` (`source.npz` blake3 `8c0faf01…`). 레시피:
`tests/fixtures/rl/training-reach-continued.toml`, `…-scratch.toml`, `training-reach.toml`.

**이 표의 모든 행을 채점하는 Evaluation IR은 하나다** — `tests/fixtures/rl/evaluation-reach.toml`,
홀드아웃 시드 201–216 16개, `nominal`과 `observation_delay` / `torque_noise` / `backlash`,
그리고 `es eval run`을 통해 적용되는 Deployment IR의 선언된 지연시간(섹션 3). 아래 인용된 모든
리포트에 `evaluation_hash f15fe888…`이 찍혀 있고, 모든 번들은 `task_hash b5d3b813…`,
`observation_hash 4ced8547…`, `deployment_hash 7af05d88…`을 지닌다. 행들은 정책에서만 다르고 그
밖의 무엇에서도 다르지 않으며, 이것은 형식이 아니라 측정의 전제다(§13.3).

**표를 이 모양으로 만든 거절, 그리고 S2b는 그 일이 일어나기 전에 그 문단을 이미 썼다.** main의
문서들에 대해 다시 임포트하면 — S2b는 S4e 이전의 Task IR에 대해 임포트했다 — S2c 정책은
`task b5d3b813…`(S4e의 것이자 Evaluation IR의 것)과 `observation 21861ea5…`(brax의 러닝 통계를
실은 자기 자신의 `Normalize{MeanStd}`)으로 해시된다. `es eval run`은 그 번들을 백엔드를 열기도
전에, 5 ms 만에, 이름을 대며 거절한다:

```
error: tests/fixtures/rl/evaluation-reach.toml does not judge …/imported/bundle/policy.esb:
ERROR XIR-040  evaluation references a different Task or Observation IR

  evaluation observation reference is 4ced8547, the bundle hashes to 21861ea5

  hint: spec 10.4: equal evaluation_hash means equal conditions
```

이것은 올바른 동작이고, 그 결과 임포트된 정책에는 행이 아예 남지 않는다. Evaluation IR을 바꾸는
것은 §13.3이 금지하는 일이고 이 패킷이 금지당한 일이다. **그래서 이 표가 택한 편차:** brax의 입력
정규화기를 임포트 전에 중립 내보내기의 첫 Dense에 접어 넣는다 —
`y = W0·((x − mean)/std) + b0 = (W0/std)·x + (b0 − (W0/std)·mean)`, 날것의 관측에 대한 같은
함수다 — 그리고 `obs_mean` / `obs_std`가 `null`인 매니페스트로 임포트를 다시 돌린다. 이것을
편의가 아니라 정직으로 만드는 것은 세 가지다:

* 임포터가 만드는 항등 `Range{−1, 1}` Observation IR이 커밋된 `observation-reach.toml`과 **비트
  단위로 같다**: 다시 돌린 임포트는 `observation_hash 4ced8547…`을 찍고, 그것은
  `regenerate_reach_documents`가 쓴 바로 그 문서다. 두 반쪽은 맞추지 않았는데도 일치한다;
* 접기는 주장이 아니라 검사다: S2c 자신의 1,000행 오라클(`oracle-1000.npz`)에서 날것 관측 위의
  접힌 네트워크와 정규화된 관측 위의 원본은 스쿼시된 행동에서 최대 절대 **2.575e-5** 차이가
  난다 — 계층 (b)의 1e-5 허용오차보다 크고, 이유는 섹션 4가 이미 말한 것이다: 큐브의 z 채널은
  `obs_std`가 1.79e-4여서 `1/std`가 약 5,600이고 f32 반올림도 함께 증폭된다. 이것은 측정을 위한
  고쳐 쓰기이지 동치성 주장이 아니다;
* 증폭 자체는 아무것도 바뀌지 않는다. `crates/es-data/src/rl_import.rs`는 `MeanStd` 정규화기가
  "그 스케일을 벗어난 것을 1/std로 증폭한다"고 경고한다. 접든 접지 않든 같은 곱이 같은 `tanh`에
  도달하며 — 그것이 바로 이어붙인 행들이 곧 부딪히는 것이다.

**표.** `success_rate`와 `episode_length`는 `nominal` 스위트의 것이고, 학습이 있는 모든 행은 시드
세 개의 평균과 최소 / 최대다. `policy_hash`는 번들 자신의 것(§5.3)이지 `training.lock`의 §19.3
`H(training_hash, checkpoint_hash)`가 아니다 — 그것은 다른 것에 대한 다른 다이제스트다:

| 행 | 그래프, 초기화 | `success_rate` (nominal) | `episode_length` | 해시 |
|---|---|---|---|---|
| **source** — S2c 정책에 대한 brax 자신의 평가, `brax-ppo-so101.md` 5.2 / 5.5에서 인용(64 에피소드, 이 Evaluation IR이 **아니다**) | 26 → [256, 256] → 6, swish + `tanh` | 학습한 파생 MJX 장면에서 **1.00**; 커밋된 장면에서 **0.00** | — (최종 거리 8.4 mm / 173.9 mm) | `source.npz` blake3 `8c0faf01…`; `policy_hash`도 `execution_hash`도 없다 — 우리 런타임이 아니다 |
| **imported** — 같은 정책을 `es policy import-rl`로, 학습 없음 | 같은 그래프, 소스의 가중치 | **0.0000** | 200.0 (16개 중 16개 타임아웃) | `policy_hash 8aa810a7…`, `weights_hash 210894c0…`, `execution_hash c42adcd4…` |
| **continued** — `[init] = imported`, PPO 4,000 반복, 시드 0 / 1 / 2 | 같은 그래프, 임포트 초기화 | **0.0000** (0.0000 / 0.0000) | 200.0 | `policy_hash 29cfcd15…` — **세 시드가 하나의 해시**; `training_hash 65ae522a…` / `54a3b46a…` / `7822b3f3…`; `execution_hash d2667368…`, 이것도 셋이 하나 |
| **from scratch, 같은 아키텍처** — `[init]` 없는 같은 레시피, 시드 0 / 1 / 2 | 같은 그래프, 무작위 초기화 | **0.0833** (0.0000 / **0.2500**) | 189.9 (169.8 / 200.0) | `policy_hash 44223c82…` / `1cf1dbd9…` / `3dd1728d…`; `training_hash 6e366f4c…` / `5705eead…` / `eef2c569…`; `execution_hash 280ac541…` / `1111350d…` / `80102aa5…` |
| **from scratch, S4e의 그래프** — `training-reach.toml`, 시드 0 / 1 / 2 | 26 → [64, 64] → 6, relu | **0.4167** (0.3125 / **0.5625**) | 143.4 (129.4 / 153.6) | `policy_hash ea84966d…` / `a5321975…` / `7f736adb…`; `training_hash 1933697d…` / `17876c12…` / `d759d683…`; `execution_hash 9ff75635…` / `22b55e10…` / `d72bee1b…` |
| **expert** | — | **생략** | — | reach에는 스크립트 전문가가 없다 — `--expert`는 데모의 집어-놓기 시연자를 몬다 — 그래서 이 작업에서 §28.9 규칙 1의 하네스 점검은 전문가 게이트가 아니라 S4e의 0.5625 행이다 |

시드별로, 산포를 추론하지 않고 읽을 수 있도록:

| 실행 | 시드 | `success_rate` | `episode_length` | 롤아웃 `return`, 처음 → 마지막 | 롤아웃 `entropy`, 처음 → 마지막 | 벽시계 |
|---|---|---|---|---|---|---|
| `continued-seed0` | 0 | 0.0000 | 200.0 | −9.56 → −13.84 | 5.53 → **13.55** | 10m49.7s |
| `continued-seed1` | 1 | 0.0000 | 200.0 | −10.07 → −11.51 | 5.52 → **13.57** | 10m50.3s |
| `continued-seed2` | 2 | 0.0000 | 200.0 | −9.87 → −13.34 | 5.51 → **13.59** | 10m52.5s |
| `scratch-seed0` | 0 | 0.0000 | 200.0 | −14.15 → −6.58 | 5.52 → 7.57 | 11m38.9s |
| `scratch-seed1` | 1 | 0.0000 | 200.0 | −13.58 → −10.83 | 5.51 → 7.52 | 11m13.9s |
| `scratch-seed2` | 2 | 0.2500 | 169.8 | −15.24 → −3.77 | 5.51 → 6.07 | 12m10.9s |
| `scratch64-seed0` = S4e의 `run-4000` | 0 | 0.5625 | 129.4 | −15.76 → −4.74 | 5.52 → 4.87 | 12.6분 (단독) |
| `scratch64-seed1` | 1 | 0.3750 | 147.2 | −14.72 → −4.23 | 5.51 → 4.76 | 12m1.5s |
| `scratch64-seed2` | 2 | 0.3125 | 153.6 | −12.32 → −3.84 | 5.51 → 4.05 | 12m12.7s |

S4e의 것을 뺀 모든 실행은 16코어 상자에서 **두 개씩 동시에** 돌았고, 그것이 부풀린 것은 벽시계뿐
이다. 스레드 수는 트레이너 자신의 것이고 시드는 레시피의 것이다. 평가는 각각 한 프로세스,
11.3초. `envelope_violation_rate`와 `executed_ne_sampled_rate`는 여덟 실행의 모든 반복에서, 그리고
모든 평가 셀에서 **1.00**이다 — S4b와 S4e에서 그랬던 그대로.

**교란 스위트, 시드 세 개의 평균**(게이트가 아니라 측정, §10.4):

| 행 | `nominal` | `observation_delay` | `torque_noise` | `backlash` |
|---|---|---|---|---|
| imported | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| continued | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| from scratch, 같은 아키텍처 | 0.0833 | 0.0000 | 0.0625 | 0.0625 |
| from scratch, S4e의 그래프 | 0.4167 | 0.1667 | 0.5208 | 0.4583 |

**`es eval compare imported/eval/report.json continued-seed0/eval/report.json`**, 그대로. 네 개의
`failure_mode_histogram` 행에서 둘째 칸만 줄였다(첫째 칸을 그대로 반복한다):

```
SUITE                METRIC                                  A              B          DELTA  SIGNIFICANT
nominal              success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
nominal              envelope_violation_rate          1.000000       1.000000      +0.000000  n/a (aggregate-only report)
nominal              episode_length                 200.000000     200.000000      +0.000000  n/a (aggregate-only report)
nominal              failure_mode_histogram     {"fallback": 16, "timeout": 16, "violation.acceleration": 128, "violation.chunk_underrun": 16, "violation.position": 3184, "violation.velocity": 720} {the same}            n/a  n/a
observation_delay    success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
observation_delay    envelope_violation_rate          1.000000       1.000000      +0.000000  n/a (aggregate-only report)
observation_delay    episode_length                 200.000000     200.000000      +0.000000  n/a (aggregate-only report)
observation_delay    failure_mode_histogram     {the same six counts} {the same}            n/a  n/a
torque_noise         success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
torque_noise         envelope_violation_rate          1.000000       1.000000      +0.000000  n/a (aggregate-only report)
torque_noise         episode_length                 200.000000     200.000000      +0.000000  n/a (aggregate-only report)
torque_noise         failure_mode_histogram     {the same six counts} {the same}            n/a  n/a
backlash             success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
backlash             envelope_violation_rate          1.000000       1.000000      +0.000000  n/a (aggregate-only report)
backlash             episode_length                 200.000000     200.000000      +0.000000  n/a (aggregate-only report)
backlash             failure_mode_histogram     {the same six counts} {the same}            n/a  n/a

A: passed=false   B: passed=false
```

모든 델타가 `+0.000000`이고 모든 히스토그램이 개수까지 같은 이유는 **두 정책이 같은 함수**이기
때문이다. PPO 4,000 반복 뒤에도 이어붙인 네트워크의 텐서 여섯 개는 임포트된 것과 비트 단위로
같다: `weights/model-1000.safetensors`와 `weights/model-4000.safetensors`는 모든 텐서에서
`weights/init.safetensors`와 최대 절대 **0.0** 차이이고, 세 시드 모두 그렇다. 그래서 세 시드는
하나의 `weights_hash 16ba065f…`, 하나의 `policy_hash`, 하나의 `execution_hash`를 공유한다. 그런데도
imported 행과 continued 행들의 `execution_hash`는 *다르며*, 이것은 옳고 또 알아둘 가치가 있다:
체인은 가중치 **파일**을 해시하고(§5.3, `policy = weights_hash`), 임포터의 safetensors와 트레이너의
safetensors는 같은 숫자를 다른 바이트 배치로 담는다. 임포트된 정책은 또 `observation_delay`에서
`nominal`과 똑같은 궤적을 걷는다 — `traj/nominal-00.estraj`와 `traj/observation_delay-00.estraj`는
같은 바이트다 — 즉 관측을 한 틱, 두 틱 늦춰도 그것이 하는 일은 하나도 바뀌지 않는다.

**왜 아무것도 움직이지 않았는가, 추론이 아니라 측정으로.** 임포트된 액터에 도달하는 그래디언트는
정확히 0이다. 커밋된 reach 문서를 `es_native.Rollout`으로 홀드아웃 시드 201에서 스텝하며 각 관측을
접힌 네트워크에 통과시키면, 처음 50 제어 틱에서 여섯 개 스쿼시 이전 값 중 **가장 작은 것**이
**23.7**, 가장 큰 것이 **340.3**이다. f32에서 그 두 크기 모두에 대해 `d tanh/dx`는 **정확히
0.0**이다. 즉 여섯 출력 전부가 매 틱 포화해 있고, 스쿼시 뒤의 모든 가중치는 0 그래디언트를 받고,
PPO는 스쿼시 뒤에 있지 않은 단 하나의 파라미터 `log_std`를 갱신한다. 그것이 올라가면서 롤아웃
엔트로피는 5.53에서 13.55까지 오르고, 그동안 크리틱은 움직일 수 없는 정책을 상대로 평소의 일을
한다(`value_loss` 2.47 → 11.77, 정점은 19.5). `first_nonfinite_step`은 `null`이고 발산한 것은
없다. 트레이너의 실패가 아니라, 포화한 스쿼시에 PPO가 하는 일이다.

**이것이 말하는 것.** 이어붙이기는 도움이 되지 않았고, 발견은 그것이 *아무것도* 하지 않았다는
것이다. 이 작업에서, 이 예산에서, 임포트된 brax 정책은 출발점으로서 무(無)보다 못하다. 4,000
반복으로는 학습이 끝나지 않는 256 × 256 네트워크를 비용으로 치르고, 그래디언트를 0으로 만드는
스쿼시를 보태기 때문이다. 나머지는 대조군이 말한다: 같은 그래프도 무작위 초기화에서는 얼어붙지
않고 학습하며(`return` −14.15 → −6.58) 세 시드 중 하나는 0.2500에 도달한다. 그리고 커밋된 64 × 64
relu 그래프, 셋 중 가장 작은 것이 이 예산에서 가장 낫다 — 시드 세 개 평균 0.4167 — 이는 S4e가
측정한 단일 시드를 자기 산포의 가운데가 아니라 **꼭대기**(0.5625)에 놓기도 한다. 이 표가 제안하는
순서대로 절제(ablation)할 것 세 가지: 이어붙인 정책의 `tanh` 스쿼시(열린 질문 3 — `Squash`가 IR
파라미터가 아니라 로워링의 일이라면 이어붙이기가 그것을 떼어낼 수 있다), 접기가 드러낸 관측 스케일
(학습 중에 한 번도 움직이지 않았다는 이유로 5,600배 증폭된 채널을 입력으로 가진 정책은 E4가
잡았을 정책이다), 그리고 여전히 1.00으로 측정되고 이 노트의 다른 모든 행에서 여전히 앞서는 열린
질문 2.

### T2 — 델타 소스 정책, 오라클 서버 `renderer-14`, 2026-09-21

아티팩트: `~/artifacts/plan-t/t2/seed0-run{1,2}/`. venv `~/venvs/es-rl`
(`docs/api-notes/brax-ppo-so101.md` §1의 고정 버전 그대로). 상세와 학습 곡선은 그 노트의 7절에
있고, 여기 있는 것은 플랜 T가 측정되는 행들이다.

같은 brax 스택, 같은 파생 씬, 같은 2 M 스텝 예산에 행동만 틱당 증분으로 읽는다
(`target_t = clip(target_{t−1} + 0.05 · clip(a, −1, 1), ctrlrange)`, `target_0` = 리셋 자세):

| 항목 | 델타 | 위치(S2c) |
|---|---|---|
| 같은 시드 두 런의 `source.npz` 비트 동일성 | **예** (`473b4fde…`) | 예 (`8c0faf01…`) |
| 소스 프레임워크 64 에피소드 `success_reached` | **1.00** (64/64) | 1.00 |
| 최종 거리, 평균 / 최대 | **3.20 / 6.72 mm** | 8.41 / 14.65 mm |
| 리턴, 평균 | 126.81 | 188.11 |
| `check_export.py` 분포 내 최대 절대 오차 | 9.537e-07 | 1.580e-06 |
| wall clock, 2 M 스텝, 4,096 환경 | 561.3 s | 540.6 s |

리턴은 낮고 정책은 *더 좋다*. 델타 행동은 목표로 도약할 수 없으므로 매 에피소드 처음 ~24틱은
팔이 이동하는 동안 거리 페널티를 문다. 과제가 요구하는 성공률과 최종 정확도는 둘 다 나아졌다.

**틱당 명령 변화량, T3이 클램프 비율을 비교할 수** — 64 × 200 × 6 값에 대한
`|target_t − target_{t−1}|`:

| | 평균 | p95 | 최대 |
|---|---|---|---|
| 관절·틱별 | 0.01006 rad | 0.03219 rad | 0.04991 rad |
| 틱별, 여섯 관절 중 최대 | 0.02090 rad | 0.04303 rad | 0.04991 rad |

50 Hz에서 0.05 rad/틱은 2.5 rad/s로 엔벨로프의 3.0 rad/s 아래이고, 관측된 최대값은 소수 넷째
자리까지 그 상한이다. 즉 증분 자체는 플레인이 허용하는 것 이상을 요구하지 않으며, T3이 보게 될
클램프는 스텝이 아니라 **적분된** 목표와 위치 한계의 문제다.

## 8. 임포터와 어댑터

1절의 규칙 3은 어댑터가 선언하고 코드는 결코 추측하지 않는다고 말한다. `es policy import-rl`의
두 반쪽에서 그것이 무엇이 되는지가 이 절이다(스펙 §14.4, 패킷 M8/S2b).

**분할.** `python/es/import_rl.py`가 pickle이나 orbax 체크포인트를 여는 유일한 장소다(INV-16):
`rsl_rl`의 `.pt`와 `rl_games`의 `.pth`는 `torch.load`, orbax 디렉터리는
`brax.training.checkpoint.load`, 커밋된 내보내기는 S2c의 `source.npz` + `meta.json`. 거기서
나오는 것은 프레임워크 중립이고 불활성이다 — `weights.safetensors`와 `import.json` 매니페스트 —
그 뒤는 전부 Rust이고 경로에 Python이 없다.

**중립 형태, 그리고 네트워크를 어디서 자르는가.** *은닉* Dense마다 `mlp.<i>.weight|bias`
`[out, in]`, 그다음 출력 Dense가 `head.weight|bias`. 세 프레임워크 모두 모든 은닉 층을
활성화하고 출력 Dense는 선형으로 둔다. 그래서 `import.json`은 `activate_output = false`를
읽는다. 우리 그래프는 같은 네트워크를 한 층 앞에서 자른다. `StateEncoder{Mlp}`는 마지막 *은닉*
층에서 끝나며 — 그 층은 활성화되므로 S2a의 파라미터 `activate_output = true`를 나른다 —
`PolicyHead{Regression}`이 출력 Dense이고 그 뒤에 소스의 `squash`가 온다. 같은 함수, 다른
절단면이다. `crates/es-data/src/rl_import.rs::learning_graph`가 그 일을 하는 곳이고 그렇게
말한다.

**어댑터 문서**(`tests/fixtures/rl/adapter-so101.toml`)는 체크포인트가 우리 로봇에 대해 알 수
없는 네 가지를 나르고, 전체가 `deny_unknown_fields`다. 아무도 읽지 않는 키는 아무도 선언하지
않은 매핑이기 때문이다.

| 블록 | 말하는 것 |
|---|---|
| `[robot] name` | 이 어댑터가 어느 로봇의 것인가 |
| `[joints] source_order`, `units` | *우리* 액추에이터 이름으로 쓴 프레임워크의 행동 순서, 그리고 그것이 라디안이라는 것 |
| `[action] kind`, 선택적 `scale` / `offset` | 위치 목표인가 토크인가, 그리고 `ctrl = offset + scale · a` |
| `[[observation.channels]] source`, `slice`, `channel` | 평탄한 관측의 각 연속 블록이 어느 Task IR `ObservationSpec` 채널을 먹이는가 |

어댑터의 `scale` / `offset`은 매니페스트의 것을 덮어쓴다. 불일치는 경고이며 조용한 해소가
아니다. 해소된 쌍은 `mapping-report.json`에 쓰이고, `--reference` 오라클은 다시 결정하는 대신
임포트가 실제로 쓴 숫자를 읽는다.

**다섯 개의 거절.** 각각 아무것도 쓰지 않고 자기 코드의 이름을 댄다(심각도와 제목은 프로젝트의
다른 모든 코드처럼 `crates/es-ir-types/src/codes.rs`에 있다):

| 코드 | 언제 거절하는가 |
|---|---|
| `IMP-001` | 어댑터의 관절 수 ≠ Task IR의 `ActionSpec.dim`(또는 매니페스트의 `action_dim`) |
| `IMP-002` | 장면에 액추에이터가 없는 관절 이름 — 장면은 Task IR 자신의 저장소 상대 `scene.path`에서 읽는다 |
| `IMP-003` | `[joints] units = "deg"`; 조용한 도 → 라디안 변환이 바로 규칙 3이 금지하는 추측이다 |
| `IMP-004` | `[action] kind`가 Task IR의 `ActionSpec.space`와 어긋난다 |
| `IMP-005` | 채널 슬라이스가 `obs_dim`을 순서대로 정확히 덮지 않거나, `ObservationSpec`이 선언하지 않은 채널을 대거나, 너비가 어긋난다 |

**출력.** `observation.toml`(채널마다 `StateInput` → `Concat` → 매니페스트의
`Normalize{MeanStd}`, 소스에 정규화기가 없으면 항등 `Range{−1, 1}`), `learning.toml`(위 그래프
→ `ActionChunker` → `mean = offset`, `std = scale`인 `Normalizer{Inverse, MeanStd}`, 그래서
모듈의 출력은 액추에이터 단위다 — 2절), 가중치를 `lower_to_torch`가 선언하는 키로 리맵한
`policy.esb`, 그리고 §14.4의 의미 매핑 보고서인 `mapping-report.json`: 관절마다 그리고 관측
채널마다 한 행, 각각 소스 인덱스나 범위, 우리 이름, 단위, 심각도를 달고.

**brax 임포트에서 `log_std`는 `null`이며 그것은 누락이 아니다.** brax의 두 번째 출력 절반은
관측의 *함수*다(`std = softplus(x) + 0.001`, `brax/training/distribution.py:171`). 따라서
가져올 상태 독립적인 값이 없다. 그 행들은 `import.json`의 `source_std` 아래 메타데이터로
따라가고 아무것도 읽지 않는다. `rsl_rl`의 `std`와 `rl_games`의 `sigma`는 `[action_dim]`
파라미터가 *맞고*, 그 둘은 진짜 `log_std`를 가져온다. `python/es/train_ppo.py --init-log-std`는
`log_std`가 null이 아닐 때만 먹으며, 스칼라 하나를 받는다. 성분이 서로 다른 벡터는 임포터가 아니라
사람의 선택이다.

## 9. 사람을 위한 열린 질문

1. 롤아웃이 Deployment IR의 선언된 지연시간을 모델링해야 하는가(트레이너 안의 청크 버퍼),
   아니면 정직함을 평가가 나르는 채로 동기적으로 남아야 하는가? 이 노트는 동기식을 고른다.
2. 플레인이 샘플링된 행동을 클램프할 때, 로그 확률은 *샘플*의 것이다; env는 *실행된* 행동을
   보았다. 그것들이 다른 비율은 보고된다; 대신 실행된 행동으로 학습할지는 나중의 어블레이션이다.
3. Regression 헤드 위의 `Squash::Tanh`: IR 파라미터인가(이 노트), 아니면 로워링이 적용하는 행동
   단위의 속성인가(`quadruped-track.md` 3.6 질문 2)? 이 노트는 파라미터를 고르며, 부재 = 기본값
   = 오늘의 해시다.
