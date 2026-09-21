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

## 8. 사람을 위한 열린 질문

1. 롤아웃이 Deployment IR의 선언된 지연시간을 모델링해야 하는가(트레이너 안의 청크 버퍼),
   아니면 정직함을 평가가 나르는 채로 동기적으로 남아야 하는가? 이 노트는 동기식을 고른다.
2. 플레인이 샘플링된 행동을 클램프할 때, 로그 확률은 *샘플*의 것이다; env는 *실행된* 행동을
   보았다. 그것들이 다른 비율은 보고된다; 대신 실행된 행동으로 학습할지는 나중의 어블레이션이다.
3. Regression 헤드 위의 `Squash::Tanh`: IR 파라미터인가(이 노트), 아니면 로워링이 적용하는 행동
   단위의 속성인가(`quadruped-track.md` 3.6 질문 2)? 이 노트는 파라미터를 고르며, 부재 = 기본값
   = 오늘의 해시다.
