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

## 8. 사람을 위한 열린 질문

1. 롤아웃이 Deployment IR의 선언된 지연시간을 모델링해야 하는가(트레이너 안의 청크 버퍼),
   아니면 정직함을 평가가 나르는 채로 동기적으로 남아야 하는가? 이 노트는 동기식을 고른다.
2. 플레인이 샘플링된 행동을 클램프할 때, 로그 확률은 *샘플*의 것이다; env는 *실행된* 행동을
   보았다. 그것들이 다른 비율은 보고된다; 대신 실행된 행동으로 학습할지는 나중의 어블레이션이다.
3. Regression 헤드 위의 `Squash::Tanh`: IR 파라미터인가(이 노트), 아니면 로워링이 적용하는 행동
   단위의 속성인가(`quadruped-track.md` 3.6 질문 2)? 이 노트는 파라미터를 고르며, 부재 = 기본값
   = 오늘의 해시다.
