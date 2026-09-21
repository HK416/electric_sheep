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

**오라클 4는 S4d로 미룬다.** 5절의 reach 작업은 보상 cone 안에서 `GetBodyPose`와 `Norm`을
필요로 하는데, `es-env`의 `ScalarPlan`은 둘 다 lowering하지 않는다. 그것이 lowering하는 것은
`GetJointState`, `GetSensor`, `GetTime`, `Arith`, `Compare`, `Normalize`, `Logic`, `Clamp`이고,
모든 잎은 스칼라 하나를 바인딩하며, `es_ir_types::Expr`에는 설계상 제곱근이 없다(그 문서가 §6.6
`DET-010`을 인용한다). 따라서 −‖cube_pos − gripper_pos‖는 `es-env`와 무관하게 lowering해 들어갈
형태 자체가 없다. 패킷 **S4d**가 그 cone 확장과 네 개의 `*-reach.toml` 문서를 소유하며, 이 표의
성공률 행은 거기서 쓰인다. 위에서 측정된 것은 기반 구조다 — 레시피, 경로, 트레이너, 플레인,
재현성 — 이미 실행되는 문서 위에서.

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
