<!-- Korean translation of docs/api-notes/mujoco-playground-quadruped.md. The English file is the working copy; regenerate this when it changes. -->

# `mujoco_playground` 사족보행 조이스틱 — 로코모션 트랙(Track B) 고정 정보

고정 버전: `mujoco_playground`(PyPI 이름 `playground`) **0.2.0**
([태그 `v0.2.0`](https://github.com/google-deepmind/mujoco_playground/releases/tag/v0.2.0) =
커밋 [`124a73f`](https://github.com/google-deepmind/mujoco_playground/tree/124a73fa3303f75a62f8fe04d329b829ed0ebdfb),
Apache-2.0), `brax` **0.14.2**(`playground` 0.2.0이 요구하는 하한, Apache-2.0),
`mujoco_menagerie` 커밋
[`1b86ece`](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97)
(2025-11-07 — `playground` 0.2.0이 직접 고정하는 정확한 커밋, `mjx_env.MENAGERIE_COMMIT_SHA`),
로봇별 BSD-3-Clause. 이 문서의 어떤 부분도 우리가 직접 실행하지 않았다 — venv 설치도,
학습 실행도, 정책 임포트도 없다. 아래의 구조적 사실(관측 구성, config 기본값, XML geom
타입, PPO 하이퍼파라미터)은 모두 위 커밋의 원본 소스에서 직접 읽었고 파일 경로로 출처를
표시했다; 성능 수치(wall-clock, throughput)는 전부 논문 것이며 **상류 보고(reported by
upstream)**로 표시한다. 로드맵상 Track B: 외부에서 학습하고, MLP를 우리 Learning IR로
임포트하고, 우리 런타임에서 실행한다.

## 1. Go1 조이스틱 환경 — Go2는 상류에 존재하지 않는다

`mujoco_playground`는 사족보행 조이스틱 로코모션으로 **Go1만** 제공한다
(`mujoco_playground/_src/locomotion/go1/`); `go2/` 패키지도, `Go2Joystick*` 환경 클래스도
없고, 기술 보고서(technical report) PDF 전체를 텍스트 검색해도 "Go2"는 한 번도 나오지
않는다. [GitHub 이슈 #270](https://github.com/google-deepmind/mujoco_playground/issues/270)은
사용자가 Go1 태스크를 Go2 모델에 맞춰 어떻게 바꿀지 묻는 글이지, 상류 지원이 아니다. Go2
트랙을 하려면 Go1의 `Joystick` 태스크를 Go2 XML에 손으로 이식해야 하며, 바로 쓸 수 있는
환경을 불러오는 것이 아니다.

**관측**(`joystick.py::_get_obs`, `_post_init`, `default_config`): 정책 입력 `"state"`는
48차원으로, 노이즈가 섞인 `[local_linvel(3), gyro(3), gravity(3), joint_pos(12) -
default_pose, joint_vel(12), last_action(12), command(3)]`의 `hstack`이다. 노이즈는 채널별
스케일(`joint_pos 0.03, joint_vel 1.5, gyro 0.2, gravity 0.05, linvel 0.1`)에 `level=1.0`을
곱한 uniform additive 노이즈다. 두 번째 키 `"privileged_state"`(123차원: `state` +
노이즈 없는 gyro/accelerometer/gravity/linvel/angvel/joint_pos/joint_vel,
`actuator_force(12)`, `last_contact(4)`, 발별 속도 `(4×3)`, `feet_air_time(4)`, 몸통
`xfrc_applied(3)`, perturbation-active 플래그 1개)는 **critic 전용**이다
(§2의 `value_obs_key="privileged_state"`) — 배포되는 정책은 이를 전혀 보지 않는다.
`default_config()`의 `history_len=1`: Go1은 프레임을 쌓지 않는다(대조: Spot/H1 조이스틱
태스크는 같은 config 필드로 `history_len=3`).

**액션**: 관절당 하나의 위치 목표값으로 12차원. `motor_targets = default_pose + action *
action_scale`(`action_scale=0.5` rad, `default_pose`는 `home` 키프레임의 `qpos[7:]`). PD
게인은 환경 생성 시 모델에 기록된다(`go1/base.py::Go1Env.__init__`):
`actuator_gainprm[:,0]=Kp`, `actuator_biasprm[:,1]=-Kp`, `dof_damping[6:]=Kd`,
`Kp=35.0`, `Kd=0.5`(`default_config`). `ctrl_dt=0.02`(50 Hz 제어), `sim_dt=0.004`(250 Hz
물리, `n_substeps=5`), `action_repeat=1`. `episode_length=1000` 스텝 = 20초. 종료 조건: 위쪽
벡터의 z 성분이 `< 0`(완전히 뒤집힘).

**커맨드**: Ornstein–Uhlenbeck과 비슷한 리샘플(`sample_command`), 진폭 상한
`a=[1.5, 0.8, 1.2]`(전진 m/s, 측면 m/s, yaw rad/s), 축별 리샘플 확률 `b=[0.9, 0.25, 0.5]`,
리샘플 간 대기 시간 `~Exp(평균 5초)`.

**도메인 랜덤화**(`go1/randomize.py`, `--domain_randomization` CLI 플래그로 opt-in —
기본은 꺼짐): 바닥 마찰 `U(0.4, 1.0)`; 관절 `frictionloss *= U(0.9, 1.1)`;
`armature *= U(1.0, 1.05)`; 몸통 무게중심 `+= U(-0.05, 0.05)` m; 전체 바디 질량
`*= U(0.9, 1.1)`; 몸통 추가 질량 `+= U(-1.0, 1.0)` kg; `qpos0[7:] += U(-0.05, 0.05)` rad.
별도의 `pert_config`(몸통에 가하는 외부 속도-킥 힘, `0–3 m/s`를 `0.05–0.2초`에 걸쳐,
`1–3초`마다)도 있지만 **기본값은 꺼짐**(`enable=False`).

**XML**(`go1/xmls/scene_mjx_feetonly_flat_terrain.xml` → `go1_mjx_feetonly.xml`,
`meshdir` → `mujoco_menagerie/unitree_go1/assets`): **충돌 geom은 이미 primitive다.**
`collision` 디폴트 클래스는 `cylinder`/`capsule`/`sphere`/`box`만 쓴다 — 몸통: box 1개 +
cylinder 2개; 다리마다 hip cylinder 3개, thigh/calf는 `fromto`를 쓴 capsule, foot sphere
1개(`r=0.023`, 이 클래스에서 `contype=1`인 유일한 geom — 나머지는 발을 빼면 모두
`contype=0 conaffinity=0`). **시각 표현은 STL 메시 5개**(`trunk`, `hip`, `thigh_mirror`,
`thigh`, `calf`)로, 바디마다 `<geom class="visual" mesh=... group=2 contype=0
conaffinity=0>` 하나씩 — 이 파일에서 mesh geom은 이것뿐이다. 솔버:
`iterations=1 ls_iterations=5 integrator=Euler timestep=0.004`, cone 미지정(MuJoCo 기본값
`pyramidal`). self-collision geom을 추가한 `fullcollisions` XML 변형도 있지만 기본이 아니고
여기서는 필요 없다. rough-terrain은 같은 로봇 XML에 heightfield 씬과 더 큰
`naconmax`/`njmax`를 쓴다.

**보상**(`reward_config.scales`, `tracking_sigma=0.25`, `max_foot_height=0.1` m):
`tracking_lin_vel +1.0`과 `tracking_ang_vel +0.5`(둘 다 `exp(-err²/tracking_sigma)`);
비용 `lin_vel_z -0.5`, `ang_vel_xy -0.05`, `orientation -5.0`, `dof_pos_limits -1.0`,
`stand_still -1.0`, `termination -1.0`, `torques -0.0002`, `action_rate -0.01`,
`energy -0.001`, `feet_clearance -2.0`, `feet_height -0.2`, `feet_slip -0.1`;
`pose +0.5`(기본 관절 각도 근처 유지)와 `feet_air_time +0.1`. 합산 후 `sim.dt`를 곱하고
스텝당 `[0, 10000]`으로 clip한다.

출처: [`go1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/mujoco_playground/_src/locomotion/go1/joystick.py),
[`go1/base.py`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/mujoco_playground/_src/locomotion/go1/base.py),
[`go1/randomize.py`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/mujoco_playground/_src/locomotion/go1/randomize.py),
[`go1/xmls/go1_mjx_feetonly.xml`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/mujoco_playground/_src/locomotion/go1/xmls/go1_mjx_feetonly.xml).

## 2. 학습

명령(README + `learning/train_jax_ppo.py`, 둘 다 고정 커밋):
`train-jax-ppo --env_name Go1JoystickFlatTerrain --domain_randomization`(설치된 콘솔
스크립트) 또는 `python learning/train_jax_ppo.py --env_name Go1JoystickFlatTerrain
--domain_randomization`. 스크립트는 자체 CLI 플래그 기본값이 아니라
`locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")`(`train_jax_ppo.py:191`)에서
튜닝된 config를 자동으로 불러온다; `learning/notebooks/locomotion.ipynb`도 동일한 호출을
쓴다. `--domain_randomization` 없이 실행하면 랜덤화되지 않는다(opt-in, §1 참고).

**네트워크**(`config/locomotion_params.py`, 기술 보고서 Table 16 — 일반 기본값 Table 15의
`(128,128,128,128)`을 오버라이드): 정책 MLP `(512, 256, 128)`, 가치 MLP `(512, 256, 128)`,
`policy_obs_key="state"`, `value_obs_key="privileged_state"` — 비대칭 actor-critic.
`brax.training.agents.ppo.networks.make_ppo_networks`(`brax` 0.14.2)가
`brax.training.networks.MLP`로 둘을 만들며, 기본 `activation=linen.swish`,
`distribution_type="tanh_normal"`: 정책 head는 `2 × action_dim = 24`개 값을 낸다(대각
가우시안의 location, log-scale); 결정론적 추론은 `tanh(location)`이며
`NormalTanhDistribution.mode`와 일치한다.

**PPO 하이퍼파라미터**(`locomotion_params.py`, Table 16): `num_timesteps=200_000_000`,
`num_evals=10`, `num_resets_per_eval=1`, `reward_scaling=1.0`,
`normalize_observations=True`, `action_repeat=1`, `unroll_length=20`,
`num_minibatches=32`, `num_updates_per_batch=4`, `discounting=0.97`,
`learning_rate=3e-4`, `entropy_cost=1e-2`, `num_envs=8192`, `batch_size=256`,
`max_grad_norm=1.0`, `kernel_init=lecun_uniform`.

**Wall-clock — 상류 보고(reported by upstream)**, 우리가 실행한 값이 아니다: 기술
보고서의 실제 로봇 섹션은 Go1 평지 학습(제한된 커맨드 범위)이 "5분 이내(2x RTX 4090)"에
끝난다고 적었다; Figure 13(`Go1JoystickFlatTerrain`, 전체 200M 스텝 실행, Table 16 config)은
`1x 4090 / 2x 4090 / 1x A100 / 16x A100 / 1x H100 / 8x H100`에 대해 reward vs. wall-clock을
~500초까지 그리면서, contact가 적어서 "기기와 토폴로지가 달라도 학습 wall-clock 시간에
큰 차이가 없다"(Appendix C.4)고 적었다. 단일 4090의 정확한 초 단위 수치는 텍스트로는 없고
그래프뿐이다. 우리 단일 RTX 4090 기준 합리적 추정은 **몇 분에서 ~10분 사이(상류 보고
범위)**이며, 우리가 직접 실행하기 전까지는 `Target / Status: unverified`다.

**numpy로 내보내기**(`mujoco_playground/experimental/brax_network_to_onnx.ipynb`, 같은
커밋): 학습된 체크포인트는 `brax.training.checkpoint.load(ckpt_path)`(orbax)로
`params = (normalizer_params, policy_params, value_params)`로 복원된다; 추론에는
`(normalizer_params, policy_params)`만 필요하다. `policy_params['params']`는 flax
딕셔너리 `{"hidden_0": {"kernel", "bias"}, "hidden_1": {...}, "hidden_2": {...},
"hidden_3": {...}}`다 — hidden Dense 레이어 3개 `(512, 256, 128)`에 `2×action_dim` 출력
Dense 레이어 1개, `kernel` shape은 `[in, out]`(오른쪽 곱셈 관례이므로 PyTorch/우리 관례
`[out, in]`으로 쓰려면 전치 필요), `bias` shape은 `[out]`.
`normalizer_params.mean["state"]` / `.std["state"]`는 관측 정규화기의 실행 통계량(각각
shape `[48]`, `brax.training.acme.running_statistics`)이다 — 가중치와 **함께** 반드시
내보내야 하고, 첫 Dense 레이어 앞에서 `(x - mean) / std`로 적용해야 한다; 노트북의
`MLP.call`도 정확히 이렇게 한다. hidden 3개 레이어는 swish 활성화, 출력 레이어는 활성화
없음, 출력의 앞쪽 절반(`action_dim`)에만 `tanh`, 뒤쪽 절반(log-std)은 추론에서 쓰지 않는다.

## 3. 라이선스

`mujoco_playground` **0.2.0**: Apache-2.0
([`LICENSE`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/LICENSE),
GitHub API의 `license.spdx_id`로 확인). 예외 하나: rough-terrain 텍스처는 CC0
(Polyhaven)이며 flat-terrain 조이스틱과는 무관하다.

`mujoco_menagerie`의 `unitree_go1`과 `unitree_go2`(커밋 `1b86ece`, `playground` 0.2.0이
고정): 둘 다 **BSD-3-Clause**, 저작권자 HangZhou YuShu Technology Co.("Unitree
Robotics"), Unitree의 공개 URDF에서 파생
([go1](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go1_description),
[go2](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2_description)).
BSD-3-Clause는 세 조건 하에 소스/바이너리 재배포와 수정을 허용한다: 소스 재배포에
저작권 표시와 면책조항을 유지할 것, 바이너리 재배포에 이를 재현할 것, 허락 없이 "Unitree
Robotics"라는 이름으로 파생 제품을 보증/홍보하지 말 것. **primitives-only 파생 XML은
허용된다** — 이는 1항의 "소스 형태의 수정"에 정확히 해당한다; menagerie 자체의
`unitree_go2/go2_mjx.xml`이 이미 모든 충돌 geom을 sphere로 바꾸었으므로(§5 참고),
primitives-only *시각* 파생물을 추가하는 것은 menagerie 스스로가 쓰는 것과 같은 라이선스
경로를 따르는 것이다. 우리 저장소에는 직접적 선례가 있다:
`tests/fixtures/mjcf/so101_pick_place.xml`은 menagerie MJCF의 "hand-derived,
primitives-only, single-file derivative"이며, 고정된 상류 커밋에 대해 provenance를
추적한다(`crates/es-assets/tests/so101_provenance.rs`).

`brax` **0.14.2**: Apache-2.0(`google/brax`의 GitHub API `license.spdx_id`).

## 4. 대안

**`legged_gym`**(`leggedrobotics/legged_gym`, Isaac Gym Preview + `rsl_rl` PPO, PyTorch
actor-critic — numpy 내보내기는 쉽다). 우리에게 불리한 점: NVIDIA Isaac Gym *Preview* 위에
구축되어 있는데, NVIDIA는 이를 단종하고 Isaac Lab으로 대체했다 — 물리 백엔드(PhysX)는
MuJoCo/MJX와 계통이 전혀 다른 contact/solver 방식이므로, 학습 시점의 contact 거동이 우리
MuJoCo CPU 백엔드와 아무런 공통 혈통이 없다(MJX로 학습한 정책은 적어도 MuJoCo의 contact
모델을 공유하므로 그보다는 parity 위험이 낮다). `unitree_rl_gym`(BSD-3-Clause, Unitree
자체 fork)이 이미 튜닝된 Go1/Go2 config를 갖고 있다는 점은 실질적 이점이지만, 기반
프레임워크 자체는 상류에서 더 이상 유지보수되지 않는다.

**Isaac Lab**(`isaac-sim/IsaacLab`, manager-based `Isaac-Velocity-Flat-Unitree-Go2-v0`,
BSD-3-Clause, 활발히 유지보수됨). `legged_gym`보다 나은 점은 Go2 태스크를 기본으로 갖고
있다는 것(mujoco_playground는 없음)과 현재 NVIDIA가 유지보수하는 프레임워크라는 것.
다른 모든 축에서는 우리에게 불리하다: Isaac Sim/Omniverse가 필요하고(수 GB 설치, MJCF가
아닌 자체 USD 기반 애셋 파이프라인), 다시 PhysX다(`legged_gym`과 같은 parity 위험),
`pip install playground` + JAX/CUDA 스택보다 GPU 드라이버/툴킷 결합이 무겁다. Go2와 Go1의
로봇 형태 차이가 physics parity보다 더 중요하다고 판명될 때만 고려할 가치가 있다 — Go1
정책을 먼저 돌려보지 않고 판단할 문제는 아니다.

**`mujoco_menagerie` + 자체 PPO**(bare Go1/Go2 XML에 대해 직접 JAX/MJX나
PyTorch+MuJoCo-CPU 학습 루프를 짜는 것). 첫 시도로는 명백히 더 나쁘다: `mujoco_playground`가
이미 튜닝해 둔 것(보상 형태, 도메인 랜덤화 범위, PD 게인, 관측 노이즈)을 비교 기준도 없이
다시 구현하는 셈이고, 이 작업은 연구/임포트 과제이므로 이미 작동하는 것을 고정하는 쪽이
같은 목적지에 도달하는 더 쉬운 길이다. Go1 태스크 형태를 넘어서야 할 때(예: `mujoco_playground`에
없는 gait-conditioned 보상을 원할 때)라면 합리적인 *두 번째* 단계지만 진입점은 아니다.

## 5. 실현 가능성 판정

**mujoco_playground Go1로 진행한다**(Go2가 아니다 — 상류에 존재하지 않는다; Go2 트랙은
Go1이 파이프라인을 증명한 뒤, 이미 충돌 primitives인 `unitree_go2`의 XML에 `Joystick`을
손으로 이식하는 후속 작업이다). 다섯 제약 모두 충족 가능하다:

1. **Primitives-only 파생물** — 실현 가능, 직접적 선례 존재(`so101_provenance.rs`). 작업
   항목: `go1_mjx_feetonly.xml`에서 `go1_primitives.xml`을 손으로 파생 — 충돌 geom은 그대로
   가져온다(다리당 primitive geom 11개 × 4 + 몸통 3개 = 47개), STL 시각 메시 5개는 대응하는
   box/capsule/cylinder 근사물로 교체한다(SO-101 파생물과 같은 패턴), menagerie 커밋
   `1b86ece`에 대해 `so101_provenance.rs`가 자신의 고정 커밋을 검사하는 방식과 같은 방식으로
   provenance를 검사한다.
2. **관측 히스토리** — 사소함: Go1 자체가 `history_len=1`이라 스태킹이 전혀 필요 없고,
   Observation IR의 `TemporalWindow(n=1, stride, align)`(§7.5)이 이를 정확히 표현하며, 나중에
   다른 환경(Spot/H1)이 필요로 할 `history_len=3`도 같은 노드 타입으로 커버된다.
3. **Learning IR의 MLP 정책** — 실현 가능: `StateEncoder { kind: Mlp, out_dim }`(§8.3) →
   `RegressionHead`가 hidden 3층 swish MLP + tanh로 눌린 mean을 정확히 재현한다; CLAUDE.md의
   오라클 규칙에 따라 MLP 그래프의 torch 동치성은 이미 검증하고 있다. 작업 항목: brax
   체크포인트에서 `hidden_0..3`의 kernel/bias(`[out, in]`로 전치)와 관측 정규화기
   `mean`/`std`를 `safetensors` 파일로 내보내 우리 lowering이 읽게 한다(pickle 금지,
   INV-16) — 정규화기를 Observation IR의 기존 `Normalize` op에 포함시키면 새 노드 타입이
   필요 없다.
4. **Deployment IR 제어 주기** — 실현 가능: `ctrl_dt=0.02`(50 Hz)는 논문에서 실제 Go1
   배포가 도는 주기와 정확히 같다(Appendix C.5: "inferenced at 50 Hz"). 작업 항목: 우리
   Deployment IR의 추론 주기를 50 Hz, 청크 크기 1로 설정한다(Go1은 반응형 정책이며 상류에
   action chunking이 없다).
5. **MuJoCo CPU 3.13 서브프로세스** — 실현 가능, `mujoco>=3.6.0`은 Playground 자체의
   하한이고 우리 고정 버전 `mujoco==3.13.0`(`docs/api-notes/mujoco.md`)이 이를 만족한다;
   Go1 MJCF는 일반 MuJoCo에서 그대로 로드된다(표준 MJCF이며 MJX 지원은 추가적인 것일 뿐).

**위험 — MJX 학습과 우리 MuJoCo CPU 스테핑 사이의 physics parity.** 맞춰야 하는 것: 학습
XML 자체의 솔버 블록 `iterations=1 ls_iterations=5 integrator=Euler timestep=0.004`, cone
`pyramidal`(미지정 = 기본값) — 정책이 학습 때 겪어보지 못한 더 높은 iteration 수로 우리
CPU 백엔드를 돌리면 contact 해석이 달라지고, MuJoCo CPU와 MJX가 같은 contact 수식을
쓰더라도 발 미끄러짐/떨림 불일치 위험이 생긴다; `sim_dt=0.004`와 0.02초 제어 틱당
substep 5회는 더 큰 스텝 하나로 근사하지 말고 정확히 유지해야 한다; 마찰
(`geom_friction`, `frictionloss`, `armature`)은 `randomize.py` 범위의 중간값을 우리
배포 기본값으로 써야 한다 — DR은 분포를 대상으로 학습하지 평균을 대상으로 학습하지
않으므로, 우리의 고정된 물리 설정 하나는 그 분포의 표본 하나일 뿐이다; `home`
키프레임(기본 서 있는 자세)은 바이트 단위로 그대로 복사해야 한다 — `default_pose`가
액션의 영점이기 때문이다. 이 중 어느 것도 우리가 실제로 primitives 파생물을 MuJoCo CPU
백엔드에 로드하고 정책을 스텝해보기 전까지는 검증되지 않는다 — 어떤 학습 실행보다도
먼저 써야 할 첫 오라클이다.
