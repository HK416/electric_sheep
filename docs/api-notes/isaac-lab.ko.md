<!-- docs/api-notes/isaac-lab.md의 한국어 번역. 영어 파일이 원본이며, 바뀌면 이 파일도 갱신한다. -->

# Isaac Lab — `ManagerBasedRLEnv` 규약, reach 태스크, rsl_rl 체크포인트

`docs/api-notes/isaac-sim.md`의 동반 문서로 목적은 같다: **M11 X2**
(`docs/packets/M11/X2-adapter-v2.md`) 준비 자료. 그 패킷은 이미 바로 이 규약들 —
`joint_pos_rel`/`joint_vel_rel`, 텀별 `scale`, `last_action`, `history_length`/`clip`, "조인트는
articulation의 순서로 이름으로 resolve된다" — 를 v2 어댑터가 선언적으로 표현해야 할 것으로
지목하고 있다. 이 워크스페이스의 어떤 것도 아직 Isaac Lab을 돌리지 않는다; 아래의 모든 주장은
`verified (fetched)`(2026-09-23에 인용된 페이지/소스 파일에서 읽음) 또는 `unverified`다.
영어 원본: `isaac-lab.md`.

2026-09-23에 가져온 출처:

- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html>
- <https://github.com/isaac-sim/IsaacLab/releases>
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py` (raw, `main` 브랜치)
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/joint_pos_env_cfg.py` (raw, `main` 브랜치)
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py` (raw, `main` 브랜치)
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py` (raw, `main` 브랜치)
- `source/isaaclab/isaaclab/utils/string.py` (`resolve_matching_names`, raw, `main` 브랜치)
- <https://github.com/leggedrobotics/rsl_rl> (`algorithms/ppo.py`, `runners/on_policy_runner.py`, raw, `main` 브랜치)
- <https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html>

저장소의 `main` 브랜치에서 가져온 것들은 모두 2026-09-23 기준이지만 릴리스 태그에 고정된 것은
아니다; 정확한 줄 번호는 근사치로 취급하고, 바이트 단위로 의존하기 전에 아래 고정 버전에 대해
다시 확인할 것.

## 1. 버전, 설치, 헤드리스 학습

| | | 상태 |
|---|---|---|
| 고정 버전 | **Isaac Lab 2.3.2** (3.0 Early Access 계열 이전의 마지막 릴리스; 2026년 2월경 출시), **Isaac Sim 5.1**과 짝 | verified (fetched, 릴리스 목록) — 3.0.0-EA(Isaac Sim 6.1)를 고정하지 않는 이유는 `docs/api-notes/isaac-sim.md` §1 참고 |
| pip 경로 | `pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com` | verified (fetched) |
| pip 주의점 | Isaac Sim 자체를 함께 설치하지만(`[isaacsim,...]` extra) **standalone 스크립트는 포함하지 않는다** — `train.py`/`play.py`는 pip 패키지에 없고 라이브러리만 있다; 아래에 보인 reach 태스크의 `rsl_rl/train.py`를 실행하려면 pip 설치가 아니라 git 체크아웃(`isaaclab.sh`)이 필요하다 | verified (fetched) |
| 저장소 경로 | `git clone` + `./isaaclab.sh -i rsl_rl`(`rsl-rl-lib` extra 설치), 이어서 `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <task-id> --headless` | verified (fetched, 문서 예제가 정확히 이 형태를 씀) |
| Python | 3.11 (Isaac Sim 5.x 짝) | verified (fetched) |

## 2. `ManagerBasedRLEnv` 관측 규약

`isaaclab.envs.mdp.observations` (**verified (fetched)**, 모듈과 호출부에서 얻은 이름과 동작):

| 텀 | 공식 | 비고 |
|---|---|---|
| `mdp.joint_pos_rel` | `joint_pos − default_joint_pos` | `default_joint_pos`는 articulation에 설정된 휴지 자세; `asset_cfg.joint_ids`에 있는 조인트만 반환됨 |
| `mdp.joint_vel_rel` | `joint_vel − default_joint_vel` | 조인트 부분집합 규칙 동일 |
| `mdp.last_action` | 정책이 직전 제어 틱에 낸 **원본(raw)**(스케일/오프셋 적용 전) 액션 | 액션 매니저에서 가져오며, `data.ctrl`에 해당하는 적용된 타깃에서 가져오는 것이 아님 |
| `mdp.generated_commands` | 이름이 지정된 커맨드 텀(예: `ee_pose`)의 현재 값 | reach 태스크가 목표 자세에 사용 |

`ObservationTermCfg`(`ObsTerm`) 필드, **verified (fetched)**: `func`, `params`, `noise`
(노이즈 모델 인스턴스, 예: `Unoise(n_min, n_max)`, 그룹의 `enable_corruption = True`일 때만
적용), `clip`(노이즈 다음에 적용), `scale`(클립 다음에 적용), `history_length` +
`flatten_history_dim`(텀별 링 버퍼; 설정하면 텀의 shape에 history 축이 추가되고,
`flatten_history_dim=True`면 마지막 차원으로 평탄화됨). 관측 매니저의 파이프라인 순서:
**계산 → 커스텀 모디파이어 → 노이즈/corruption → clip → scale**.

**연결(concatenation) 순서**: `concatenate_terms = True`인 `ObsGroup`(예: `PolicyCfg`)은
자신의 텀들을 **선언 순서**로 이어붙인다 — `@configclass`에 필드가 적힌 순서이지, 정렬되거나
알파벳순이 아니다. reach 태스크의 policy 그룹(§4)은 `joint_pos, joint_vel, pose_command,
actions` 순서 그대로이므로, 평탄화된 관측 벡터는
`[joint_pos_rel(6 또는 N) | joint_vel_rel(N) | pose_command(7) | last_action(N)]`이다.

**조인트 순서** — M11 X2가 명시적으로 필요로 하는 바로 그것, 정확히 고정한다:
`isaaclab.utils.string.resolve_matching_names(keys, target_names, preserve_order=False)`가
`SceneEntityCfg`/액션 텀이 `joint_names` 정규식 목록을 실제 인덱스로 바꿀 때 쓰는 함수이며,
그 docstring은 명확하다: `preserve_order=False`(**기본값이며 `JointPositionActionCfg`가 쓰는
값**)일 때 "매칭된 인덱스와 이름의 순서는 제공된 문자열 목록의 순서와 같다" — 즉 **타깃
목록 자체의 순서**(에셋/USD가 저작한 그대로의 articulation 내부 조인트 순서)이지, config에
정규식 패턴이 적힌 순서가 *아니다*. `preserve_order=True`였다면 정규식 목록의 순서를 따랐을
것이나, reach 태스크는 이를 설정하지 않으므로 기본값에 머문다. **verified (fetched, docstring
+ 예시)**: `['a','b','c','d','e']`를 `['a|c', 'b']`로 매칭할 때 `preserve_order=False`는
`([0,1,2], ['a','b','c'])`를 반환한다 — 패턴 순서가 아니라 타깃 순서.

## 3. `JointPositionActionCfg`

**verified (fetched)**, `isaaclab.envs.mdp.actions.actions_cfg` + `JointAction.process_actions`:

```
processed_actions = raw_actions * scale + offset
```

- `scale`: float 또는 조인트-정규식별 dict, 기본값 `1.0`.
- `use_default_offset`: bool, 기본값 `True`. `True`면 env 생성 시 `offset`이 articulation의
  `default_joint_pos`로 **덮어써진다** — 즉 액션은 `scale`로 스케일된, *휴지 자세를 중심으로
  한 델타*가 되며, 이는 `mdp.joint_pos_rel` 자체의 중심 잡기와 일치한다. `False`면 `offset`은
  명시적으로 설정된 값(기본 `0.0`)이 그대로 쓰여, 액션은 스케일된 절대 타깃이 된다.
- `processed_actions`가 조인트 위치 타깃에 실제로 쓰이는 값이다(articulation 자체의 PD
  드라이브를 거치며, `qpos`를 직접 쓰는 것이 아니다).

Franka reach 태스크의 구체적 인스턴스(`config/franka/joint_pos_env_cfg.py`, verified fetched):
`JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5,
use_default_offset=True)`.

## 4. reach 태스크, 그대로 옮긴 구조 (`reach_env_cfg.py` + Franka 오버라이드)

기본 `ReachEnvCfg.__post_init__`(**verified fetched**): `decimation = 2`, `sim.dt = 1/60`
(약 16.667 ms 물리 스텝), `sim.render_interval = decimation`, `episode_length_s = 12.0`. 따라서
제어 주기는 `decimation * sim.dt = 1/30 s`(약 33.3 ms, 약 30 Hz)이고, 한 에피소드는
`12.0 / (1/30) = 360` 제어 스텝이다. (대조: M8의 brax/Playground SO-101 소스 정책,
`docs/api-notes/brax-ppo-so101.md` §2는 200 Hz 물리 위에서 50 Hz 제어로 돈다 — 다른 프로젝트의
숫자이지 Isaac Lab의 숫자가 아니며, M11 X2가 이미 계획한 `[timing] policy_dt` 체크
[`X2-adapter-v2.md:39`] 없이는 둘을 그냥 바꿔 쓸 수 없다.)

```python
# ObservationsCfg.PolicyCfg — 선언 순서가 곧 연결 순서
joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
actions = ObsTerm(func=mdp.last_action)
# __post_init__: enable_corruption = True, concatenate_terms = True

# ActionsCfg (기본은 추상; Franka가 채운다)
arm_action: ActionTerm = MISSING
gripper_action: ActionTerm | None = None
# Franka: arm_action = JointPositionActionCfg(asset_name="robot",
#   joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)

# RewardsCfg
end_effector_position_tracking        = RewTerm(mdp.position_command_error,       weight=-0.2)
end_effector_position_tracking_fine   = RewTerm(mdp.position_command_error_tanh,  weight=0.1, params={"std": 0.1})
end_effector_orientation_tracking     = RewTerm(mdp.orientation_command_error,    weight=-0.1)
action_rate                            = RewTerm(mdp.action_rate_l2,               weight=-0.0001)
joint_vel                              = RewTerm(mdp.joint_vel_l2,                 weight=-0.0001)

# TerminationsCfg
time_out = DoneTerm(mdp.time_out, time_out=True)   # 실패에 의한 조기 종료 없음

# EventCfg — 에피소드 리셋
reset_robot_joints = EventTerm(mdp.reset_joints_by_scale, mode="reset",
                                params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)})
```

모두 두 소스 파일에서 **verified (fetched)**. Franka 오버라이드는 세 추적 보상 텀과 커맨드
생성기 모두에서 추적 대상 엔드이펙터 바디로 `body_names="panda_hand"`를 지정한다.
`FrankaReachEnvCfg_PLAY`(평가/재생 변형)는 관측 그룹의 `enable_corruption = False`(평가 시
노이즈 없음)를 설정하고 env 개수를 줄인다 — SO-101 reach 태스크의 `-Play-v0` 변형을 만든다면
그대로 본뜰 형태다.

**이를 SO-101 reach 태스크에 그대로 본뜨려면**(여기서 만들지는 않음, 재사용할 형태만 제시):
SO-101의 링크 이름을 쓴 동일한 다섯 `RewardsCfg` 텀, 동일한 `ObservationsCfg.PolicyCfg` 네
텀, `JointPositionActionCfg(asset_name="robot", joint_names=[<SO-101의 6개 조인트 이름 또는
정규식>], scale=<미정>, use_default_offset=True)`, 동일하게 `time_out`뿐인 종료 조건.

## 5. rsl_rl 래퍼: `clip_actions`, 관측 정규화

`RslRlVecEnvWrapper`(`isaaclab_rl.rsl_rl.vecenv_wrapper`, **verified fetched**):
`clip_actions: float | None`를 받으며, 설정되면 매 `step()`마다
`torch.clamp(actions, -clip_actions, clip_actions)`가 적용되고 액션 공간의 경계도 그에 맞게
다시 쓰인다. 관측값은 `self.unwrapped.observation_manager.compute()`에서 온다(§2대로 이미
scale/clip/noise 적용됨) — **래퍼 자체는 추가 정규화를 하지 않는다**; 경험적(러닝 평균/표준편차)
정규화는 *정책 쪽*의 문제다:

- `RslRlOnPolicyRunnerCfg.empirical_normalization: bool`(**verified fetched, 이름과
  동작** — 여러 이슈 스레드로 뒷받침됨)은 rsl_rl의 `OnPolicyRunner`/`PPO` 안에
  `EmpiricalNormalization` 모듈을 켜며, 이는 관측값에 대해 러닝 평균/표준편차를 유지하다가
  actor/critic이 보기 전에 정규화한다(Isaac Lab 자체의 텀별 `scale`은 고정 상수이지 학습되는
  값이 아니라는 점에서 별개).
- rsl_rl 4.0 이상은 추가로 러너 config에 `obs_groups` 매핑(예: `{"actor": ["policy"],
  "critic": ["policy"]}`)을 요구하며, 없으면 `OnPolicyRunner.__init__`이 멈춘다 — Isaac Lab과
  너무 새로운/너무 오래된 `rsl_rl` pip 패키지 사이의 **verified (fetched, GitHub 이슈)** 버전
  호환성 함정.

## 6. 내보내기: `policy.pt` / `policy.onnx`

`isaaclab_rl.rsl_rl.exporter`(**verified fetched**, `_TorchPolicyExporter` /
`_OnnxPolicyExporter`): 둘 다 학습된 `actor` 모듈과 경험적 정규화기(또는
`empirical_normalization=False`면 `torch.nn.Identity()`)를 deep-copy하고, **내보낸 산출물이
이 둘을 합성한다**: `forward(obs) = actor(normalizer(obs))`. **정규화기는 내보낸
`policy.pt`/`policy.onnx`에 구워 넣어진다** — 내보낸 파일을 읽는 배포 측은 정규화 통계를 따로
알 필요가 없다; 필요한 것은 원본(Isaac Lab이 scale/clip한) 관측 벡터뿐이다. 순환 정책
(`memory_a.rnn`)은 명시적인 hidden/cell 상태 입출력과 함께 내보내진다(LSTM: `(obs, h_in,
c_in) → (actions, h_out, c_out)`; GRU: `(obs, h_in) → (actions, h_out)`); reach 태스크의
기본 MLP 정책은 비순환이다.

## 7. 체크포인트 형식 — 버전에 따라 다름, 둘 다 확인함

**verified (fetched)**, `leggedrobotics/rsl_rl` 소스, 고정된 `rsl_rl` 패키지 버전에 따라 두
가지 형태가 있다(Isaac Lab 2.3.2 자체가 고정하는 버전은 별도로 재확인하지 않았다 — 특정
체크포인트 파일에 대해 어느 쪽을 신뢰할지는 그 릴리스의 `requirements`/`setup.py`를 확인할 것):

- **클래식 (`rsl_rl` 2.x, 기존에 공개된 체크포인트 대부분이 쓰는 형식)**: 하나의
  `ActorCritic` 모듈의 `state_dict()`에서 나온 단일 결합 `model_state_dict`이며, 키는
  `actor.<layer>.weight`/`.bias`, `critic.<layer>.weight`/`.bias` 접두어를 쓰고, 가우시안
  정책의 액션 노이즈를 위한 최상위 `std`(또는 `log_std`) 파라미터가 있다. 이 부분은 뒷받침하는
  검색 결과 이상은 `unverified`다 — 이번 패스에서 2.x 태그 소스로부터 직접 재도출하지는
  않았다.
- **현재 `main` / rsl_rl ≥ 4.0–5.0**: `PPO.save()`는 별도의 `actor_state_dict` /
  `critic_state_dict`(각각 `self._raw_actor.state_dict()` /
  `self._raw_critic.state_dict()`에서)와 `optimizer_state_dict`를 반환하며, Random Network
  Distillation이 설정되어 있으면 조건부로 `rnd_state_dict`/`rnd_optimizer_state_dict`도
  반환한다. `OnPolicyRunner.save()`는 여기에 `iter`(현재 학습 iteration)와 `infos`를 더해
  `torch.save`한다. **체크포인트 딕셔너리에서 관측 정규화기 상태는 찾지 못했다** — 정규화
  통계는 `actor`/`critic` 모듈 자체에 있다(그것들 자신의 `state_dict()`의 일부로 저장되는
  `EmpiricalNormalization` 서브모듈), 이는 §6의 익스포터가 별도의 키가 아니라 로드된 actor에서
  바로 정규화기를 읽는다는 것과 일치한다.
- 이 분리는 **알려진 호환성 파괴 변경**이다: "rsl-rl ≥ 5.0은 별도의 `actor_state_dict`와
  `critic_state_dict` 항목을 요구하므로, 더 오래된 에셋 릴리스와 함께 나온 공개
  사전학습 체크포인트는 `KeyError: 'actor_state_dict'`로 로드에 실패한다" — **verified
  (fetched, GitHub PR/이슈 토론)**. M11 X2가 실제 rsl_rl 체크포인트용으로 만들 임포터는 하나의
  형태를 가정하지 말고 파일이 어느 형태인지(`"model_state_dict" in ckpt` vs.
  `"actor_state_dict" in ckpt`)로 분기해야 한다.

## 8. 아직 모르는 것

- ~~Isaac Lab 2.3.2가 고정하는 정확한 `rsl_rl` 버전~~ — 측정함(M11 I0, 2026-09-23):
  `isaaclab 2.3.2.post1` 휠의 메타데이터는 `[all]`과 `[rsl-rl]` 아래에 `rsl-rl-lib==3.0.1`을
  고정한다(번들된 `source/isaaclab_rl/setup.py`는 3.1.2라고 적지만 pip은 메타데이터를
  따른다). 3.x는 §7의 5.0 분리선 아래이므로 클래식 `model_state_dict` 형태가 예상된다 —
  M11 I3가 측정했다(§9): 그 형태가 맞다.

**측정된 설치(M11 I0, 2026-09-23, 자세한 내용은 `docs/api-notes/isaac-sim.md` §7).** pip
경로로 서버의 `~/venvs/es-isaac`에 설치했다(Python 3.11.16, `isaacsim 5.1.0.0`). 휠은
`isaaclab_rl`, `isaaclab_tasks`, `isaaclab_assets`, `isaaclab_mimic`, `isaaclab_contrib`를
`isaaclab/source/` 아래 소스 익스텐션으로 번들한다(별도 배포판 없음). Ubuntu 26.04에서
`libxml2.so.2`를 공급하면(isaac-sim §7.2) `isaaclab.app.AppLauncher(headless=True)`가 뜬다 —
이것은 `isaaclab.python.headless.kit`을 써서 RTX 렌더러 크래시를 피한다. `isaaclab_rl.rsl_rl`과
`isaaclab_tasks`가 임포트되고 `*Reach*` gym id 18개가 등록된다(`Isaac-Reach-Franka-v0`,
`Isaac-Reach-OpenArm-Bi-v0`, UR10e 배포 변형 등, SO-101용은 없음). 학습 스크립트(`train.py`)는
돌리지 않았고 Isaac Lab 리포도 클론하지 않았다.
- SO-101의 조인트 이름/개수가 커스텀 `joint_names` 정규식을 필요로 하는지, 아니면 Franka의
  `"panda_joint.*"`처럼 하나의 전체 매칭 패턴으로 되는지 — SO-101 USD 에셋이 저작한 조인트
  이름에 달려 있으며 여기서는 가져오지 않았다(이 프로젝트 자체의 MJCF가 Isaac 에셋이 아니라
  진실 소스다; `docs/api-notes/mujoco.md`의 Menagerie 절이 이미 MuJoCo용 `so101.xml`의 조인트
  이름을 고정하고 있다 — Isaac 쪽 USD 변환은 `docs/api-notes/isaac-sim.md` §3의 MJCF 임포터
  단서에 따라 이름을 바꿀 수 있다).
- `PhysxCfg` 반복 횟수의 정확한 필드명/기본값 — `docs/api-notes/isaac-sim.md` §5로 미룸, 같은
  공백. (2.3.2가 스테이지에 쓰는 값은 §9에 있다.)

## 9. M11 I3가 측정한 것 (2026-09-23, 오라클 서버, Isaac Lab 2.3.2.post1, rsl-rl-lib 3.0.1)

우리 장면에서 SO-101 reach 정책을 학습시키면서(`python/es/rl_source/isaac_so101_reach/`,
`docs/design/rl-continuation.md` 7절 I3) Isaac Lab 자체에 대해 확인한 것. 모든 항목은 실제로
실행했다; 산출물은 `~/artifacts/plan-x/i3/`.

- **체크포인트는 §7의 클래식 형태다.** `model_<it>.pt`는 `model_state_dict`,
  `optimizer_state_dict`, `iter`, `infos`(`None`)를 담는다. `model_state_dict`는 `ActorCritic`
  하나다: `std` `[6]`, `actor.{0,2,4}.{weight,bias}`(32 → 64 → 64 → 6; 인덱스는 ELU 모듈까지
  센다), `critic.{0,2,4}.*`, 그리고 `actor_obs_normalization = False`이면 정규화기 키는 없다.
  `OnPolicyRunner`는 `save_interval`마다 저장하고 **마지막 것은 `model_<N>.pt`가 아니라
  `model_<N-1>.pt`로** 저장한다. `import_rl.py --from rsl-rl`는 이를 그대로 읽는다.
- **rsl-rl-lib 3.0.1은 `obs_groups`가 필요하고**(`{"policy": ["policy"], "critic": ["policy"]}`),
  Isaac Lab 2.3의 `RslRlPpoActorCriticCfg`가 넘기는 `state_dependent_std` 필드는 무시한다
  (`ActorCritic.__init__ got unexpected arguments, which will be ignored`).
- **`params/env.yaml`은 안전한 YAML이 아니다.** `isaaclab.utils.io.dump_yaml`은 `yaml.dump`라서
  모든 튜플이 `!!python/tuple`로, `SceneEntityCfg`의 id가
  `!!python/object/apply:builtins.slice`로 쓰이고, `yaml.safe_load`는 파일을 거부한다.
  `import_rl.py`는 이제 모든 `!!python/...` 노드를 그 표기대로의 평범한 list, dict, 문자열로 읽고
  어떤 객체도 만들지 않는다.
- **보상 관리자는 모든 항에 `step_dt`를 곱한다**(`reward_manager.py`: `value = func(...) *
  weight * dt`); "제어 스텝당" 의미의 항은 `weight / step_dt`가 필요하다.
- **`JointPositionActionCfg.clip`**(`{조인트 정규식: (lo, hi)}` dict)은 *처리된* 행동을 자른다;
  `last_action`은 자르지 않은 원시 행동 그대로다.
- **MJCF 임포터의 베이스 용접은 복제되지 않는다.** `fix_base`에서 임포터는 `body0`이 빈
  `PhysicsFixedJoint rootJoint_<base>`를 쓴다 — MJCF 자체 좌표에서 월드에 고정된다.
  `InteractiveScene`이 `env_spacing = 2.0`으로 복제하자 모든 env의 팔이 월드 원점으로 끌려갔다
  (그리퍼가 자기 env 원점에서 ±1 m 떨어져 읽혔다; PhysX는 `Cloning joints …/rootJoint_base
  without a body rel may cause issues, since the localPose wont be updated`를 남긴다). 그래서
  env는 env 간 충돌을 거른 채 `env_spacing = 0`으로 돈다 — `physx_ref.py` 자신의
  `GridCloner(spacing = 0)`과 같다.
- **Kit은 `AppLauncher`로 시작한 스크립트에서 잡히지 않은 Python 예외 뒤에 0으로 종료한다**;
  드라이버가 직접 잡아 `os._exit(1)`해야 한다.
- **`torch.inference_mode()` 안에서 만든 텐서는 그 밖에서 쓸 수 없다**: inference-mode 롤아웃 두
  번 사이의 `env.reset()`은 둘을 한 inference-mode 블록에 넣지 않으면
  `write_joint_state_to_sim`에서 실패한다.
- **2.3.2가 스테이지에 쓰는 `PhysxCfg`**(괄호 안은 `physx_ref.py`의 `World`가 쓰는 값): 솔버 TGS
  (TGS); broadphase `GPU`(`MBP`); `enableCCD` false(true); GPU 파이프라인에서
  `enableGPUDynamics` true(false); `enableStabilization` false(false); `bounceThreshold` 0.5,
  `frictionOffsetThreshold` 0.04, `frictionCorrelationDistance` 0.025, 위치 반복 1..255, 속도
  반복 0..255(모두 `World`는 쓰지 않음); 초당 200 스텝(200).
- **GPU 파이프라인에서 같은 체크포인트, 같은 리셋이 실행마다 한 숫자가 아니다:** 시드 2의 선택된
  체크포인트가 같은 1,024개 리셋에서 0.6855, 이어서 0.6680을 냈다(한 프로세스에서
  `env.reset(seed=1002)` 두 번).
