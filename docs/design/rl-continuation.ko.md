# 가져온 정책의 강화학습 이어하기 (plan S, §13.4, §14.4, §28.11)

코드 위치: 패킷 `docs/packets/M10/W3a-es-import-split.md` 이후 가져오기 모듈은
`crates/es-import/src/rl_import.rs`(crate `es-import`, layer 9, spec §4.2)에 있다; `es-data`가
이를 `es_data::rl_import`로 재수출하므로 두 경로는 같은 모듈을 가리킨다.

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

### 2a. 그래디언트를 어느 행동에서 계산하는가(`[rl] estimator`, 패킷 M9/R5)

샘플링은 정책이 무엇을 *제안했는지*를 말하고, 플레인은 환경이 무엇을 *실행했는지*를 말한다. 둘이
다를 때 — 이 작업에서는 매 틱마다 — PPO에는 고를 수 있는 행동이 둘 있고, 그 선택은 플레인의 것이
아니라 추정량의 것이다:

```
a        = mu + exp(log_std) * eps       # the sample
executed = Rollout.act(a)                # what the plane let through to the actuator
```

| `[rl] estimator` | 롤아웃 버퍼의 행동 | 저장된 로그 확률과 비율(ratio)의 로그 확률 |
|---|---|---|
| `"sampled"` (기본값) | `a` | `log N(a; mu, sigma)` |
| `"executed"` | `executed` | `log N(executed; mu, sigma)` |

`"executed"`는 "플레인은 환경의 일부다"의 허용된 읽기다(§13.4, `docs/reviews/M9.md` S-7,
소유자의 선택지 B). 트레이너는 실제로 보상을 만들어 낸 행동으로부터 배우고, 엔벨로프는 Deployment
IR이 선언한 그대로 남는다(INV-11..13 — 다른 선택지인 엔벨로프 확장은 문서의 결정이지 이 패킷의
것이 아니다). 그리고 넷은 의도적으로 움직이지 **않는다**. 보상, done, 가치, GAE는 어느 쪽에서도
같은 숫자다 — 그것들은 언제나 실행된 행동의 것이었다. `executed_ne_sampled_rate`는 계속 플레인의
출력을 **샘플**과 비교한다 — 그것은 추정량이 아니라 엔벨로프에 관한 사실이다. 배포되는 정책은
어느 쪽에서도 `mu`이므로 `learning_hash`는 이 선택을 느낄 수 없다. 기본값은 `"sampled"`로 남아,
이 패킷 이전에 측정된 모든 행은 여전히 그때의 그 행이다. 이것이 *아닌* 것은 중요도 표본화 보정이다.
비율은 같은 가우시안을 같은 점에서 두 번 평가한 것 사이의 값이므로, `"executed"`는 클램프를
교정해야 할 절단(censoring)이 아니라 모델링되지 않은 환경의 일부로 두고, 환경이 실제로 본 분포
위에서 PPO를 도는 것이다. 7절의 R5 행이 그 측정이다.

### 2b. 롤아웃이 어느 엔진을 스텝하는가 (`[rl] backend`, 패킷 M11/X1)

`[rl] backend = "mjwarp"`이면 `es train`은 트레이너의 argv에 `--backend mjwarp`를 덧붙이고,
`train_ppo.py`는 그것을 `es_native.Rollout(…, backend=…)`에 넘기며, `Rollout`은 자신의 `Env`를
`MuJoCoCpuBackend` 대신 `MjWarpBackend` 위에 짓는다 — `Rollout` 안의, 단형화된 두 env의 닫힌
열거형이지 트레이트 객체도 새 트레이트도 아니다(INV-17). 없으면 `"mujoco-cpu"`이고, 적어 넣어도
없는 것과 정확히 같게 직렬화되므로 M11 이전에 측정된 모든 레시피는 자신의 `training_hash`를
유지한다. 그 밖의 값은 레시피 JSON과 `training/config.json`이 싣는 계획 줄에 들어가므로
`training_hash`에 들어간다(계획 골든 `tests/golden/train/plan-reach-mjwarp.txt`). `"newton"`과
`"physx"`는 레시피를 파싱할 때 거부된다: Newton 어댑터는 액추에이터를 선언하지 않으므로 그 자신의
`load`가 정책이 행동할 수 있는 모든 장면을 거부하고, PhysX는 M11/I1이다. Safety Plane은 어느
엔진에서나 같은 코드다(INV-11..13). MJWarp는 tier 2다(§3.5): 그 위의 실행은 CPU 백엔드에 대해
결코 비트 단위로 같지 않으며, `train_rl_two_runs_are_bitwise`는 계속 `mujoco-cpu`에 대한
진술로만 남는다.

**측정: 스펙 17.2 각주 이후 reach 장면은 `mjwarp`에서 돈다.** `so101_pick_place.xml`은
`cone="elliptic"`을 선언하고, MJWarp는 그것을 tier 2 행으로 매핑한다(이 패킷의 첫 측정 뒤 소유자의
결정). reach 레시피에 `[rl] backend = "mjwarp"`로 돌린 20 반복 PPO 스모크는 28 s에 끝났다(평균
리턴 −15.8 → −6.6). 수치와 그것에 필요했던 어댑터 수정 둘은 `evaluation-execution.ko.md` 2.8에
있다. `rollout_backend_mjwarp_steps_the_reach_documents`(ignored, `mujoco_warp` 필요)는 reach
문서를 두 엔진에서 스텝하고 거리를 출력한다.

### 2c. 롤아웃이 렌더링한다 (패킷 M11/X3)

es-py를 `render` 피처로 빌드하면 `es_native.Rollout`은 이미지를 관측한다(`python/es/pyproject.toml`의
maturin 빌드가 이 피처를 켠다). `Rollout`은 Observation IR의 단 하나의 `ImageInput`에 대해 env마다
`es_env::EnvRenderer`를 하나씩 가지며, 그 설정은 해당 센서를 선언한 Task IR 채널로부터
`es_env::render::sensor_cfg`가 만든다 — `es loop collect --frames`와 `es eval run --frames`가 쓰는
바로 그 함수다 — 그리고 그 `frame`을 같은 `es_eval::runner::capture`에 프레임 소스로 넘기므로 두
번째 관측 구현은 없다. 각 env의 리셋(명시적 리셋이든 done에 대한 `Env::step` 자신의 리셋이든)은 그
env의 `begin_episode`를 호출하므로, `seed = "tick"` 아래에서 샘플 키는 *그 env의* 에피소드에서 다시
시작한다(`renderer.ko.md` 12.8). `observe` 한 번은 env마다 프레임 하나이며, 그것이 틱이 세는 렌더
인덱스다. 누적도, env를 가로지르는 배치도(X3b) 없다. 피처가 없으면 이미지 입력은 예전처럼 `observe`에서
이름으로 거부된다. 상태만 있는 Observation IR은 렌더러를 짓지 않고 장치도 열지 않는다.

`train_ppo.py`는 모든 포트의 모양을 `contract.json`에서 읽고 이미지 포트를 `[n_envs, C, H, W]`로
쌓는다(여기서는 `[3, 96, 96]`, Observation IR 자신의 출력 레이아웃이다). 가치 MLP는 포트들을 평탄화해
읽는다. `Rollout.metrics()`는 `render_ms_per_frame`을 싣고 §12.4 집합의 `camera_frames_per_sec` /
`pixels_per_sec`를 채운다. `train_ppo.py`는 이것을 `env-metrics.json`에 쓰고 렌더 행을 stderr에
출력하며, `metrics.json`에는 결코 넣지 않는다.

카메라를 가진 reach 문서 — `task-`, `observation-`, `learning-`, `evaluation-reach-vision.toml` — 는
`regenerate_vision_reach_documents`(`crates/es-py/tests/vision_reach.rs`)가 reach 문서들과
`task-pt-tick.toml`의 카메라로부터 생성한다: reach 작업에 `Pt` 16 spp, 3 바운스, `seed = "tick"`의
`rgb_overhead`를 더한 것이다. Learning IR은 데모의 처음부터 학습하는 `VisionEncoder { ResNet18 }`(512)을
reach 상태 MLP(64) 곁에 두고 `Fusion { Concat }` 576, 그리고 reach 헤드다. `deployment-reach.toml`은
그대로 쓴다.

**측정.**

| 주장 | RTX 3060 (로컬) | RTX 4090 (오라클 서버) |
|---|---|---|
| `Rollout` 프레임 == `es loop collect --frames` 프레임(프로세스 내 수집기, env 0, 2 에피소드 × 4 틱), `Pt` 16 spp, `seed = "tick"` | 비트 동일, 8/8 | 비트 동일 |
| `Rollout` 프레임 == 수집기 배선의 쌍둥이 렌더러, 2 env × 8 틱, env 1은 env 0과 다른 위상에서 리셋 | 비트 동일, 16/16 | 비트 동일 |
| env 1의 `begin_episode`를 제거한 같은 오라클 | env 1의 첫 틱에서 실패(27,648바이트 중 27,054) | — |
| 상태만 있는 롤아웃(`so101_100steps.json`), `render` 유무 모두 | 골든 재현 | 골든 재현 |
| 상태만 있는 PPO 스모크, 이 패킷 전후의 `train_ppo.py`(3 반복, 4 env) | 체크포인트와 가치 비트 동일 | — |
| `train_rl_two_runs_are_bitwise` | 통과 | 통과 |
| 비전 작업 PPO 스모크, 10 반복 × 2 env × 16 스텝 | 실행됨, 손실 유한(0.51 → 1.37), 41.5 ms/프레임 | 실행됨, 손실 유한(0.51 → 1.37), 20.8 ms/프레임, 14 s |

`Rollout.render_ms_per_frame`, env 하나, 96×96, 프레임 전체(재배치, 업로드, 트레이스, 리드백),
릴리스 빌드, 워밍업 2프레임 후 팔이 움직이는 동안 16프레임; 각각 세 번 실행(3060 실행들은 ±0.4 ms
안에서 일치):

| 렌더 | RTX 3060 (로컬) | RTX 4090 (오라클 서버) |
|---|---|---|
| `Rs` | 2.02–2.42 ms | 2.44–2.45 ms |
| `Pt` 4 spp | 9.30–9.65 ms | 5.72–5.86 ms |
| `Pt` 4 spp + SVGF | 9.72–9.91 ms | 7.32–7.40 ms |
| `Pt` 16 spp | 31.06–31.15 ms | 15.67–15.73 ms |
| `Pt` 16 spp + SVGF | 31.40–31.42 ms | 17.17–17.32 ms |
| `Pt` 64 spp | 117.03–117.15 ms | 54.41–54.54 ms |
| `Pt` 64 spp + SVGF | 117.41–117.42 ms | 56.04–56.17 ms |

(`cargo test -p es-py --release --features render --test vision_reach -- --ignored
rollout_render_cost`.) env 하나가 프레임을 차례로 렌더링하므로, `envs × horizon` 행의 PPO 반복 하나는
`envs × horizon` 프레임을 직렬로 치른다: 16 env × 64 스텝, 16 spp이면 4090에서 학습 이전에 반복마다
렌더링만 ~16 초다. 그것이 X3b의 질문이다. 이 두 카드를 넘어서는 것은 `Target / Status: unverified`.

**2c가 건너뛰는 것.** `es train`의 `[rl]` 경로는 Observation IR에 이미지 입력이 있는 번들을 여전히
거부한다(`crates/es/src/cmd/train.rs`, 이 패킷의 범위 밖). 그래서 스모크는 `es policy lower` 모듈 위에서
`train_ppo.py`를 직접 돌린다; 그 거부를 푸는 것은 후속 작업이다. 관측마다 이미지 입력 하나(수집기의
규칙이기도 하다). env를 가로지르는 배치 렌더링(X3b).

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
* 증폭 자체는 아무것도 바뀌지 않는다. `crates/es-import/src/rl_import.rs`는 `MeanStd` 정규화기가
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

**임포트, 같은 서버 같은 날**(`ES_S2B_SOURCE=~/artifacts/plan-t/t2/seed0-run1`,
`ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129,
`cargo test --release -p es --test cli -- --ignored import_rl_reproduces_the_source_policy`).
`import.json`이 `action_kind = "joint_delta"`를 싣고, `adapter-so101-delta.toml`이 증분 단위와
함께 같은 것을 선언하며, Task IR과 Deployment IR이 `JointDelta`를 선언한다:

| 티어(4절) | 비교 대상 | 측정값 |
|---|---|---|
| (a) | 임포터의 numpy → torch 재구성 대 우리 런타임, 1,000 × 6 값 | **비트 단위 일치** — 불일치 0개 |
| (b) | 우리 런타임 대 JAX의 결정적 `tanh(loc)`, `obs_scaled` | **3.297e-7** 최대 절대 오차(허용 1e-5) |

| 슬롯 | 해시 |
|---|---|
| `weights_hash` | `c0199681d71b042332c2211590aef2a3c6a8020e965eb375652ec3dcb453c884` |
| `observation_hash` | `598ad2414fc5c9ffd414bc00658d029326ecde569a84909c0f47e53e34d35cc8` |
| `learning_hash` | `a408abec926306134b3df604ba813770d96bfc2fead3c9829d681d9beb9e6ded` |
| `policy_hash` | `17226acd48abd7288c1306cee492229fab38f874e1de52305e315e6ddb90b42a` |
| `deployment_hash` | `9c81278b496e51cba2aec3852f6543852c084906e2d9d48b934e37a60f0f993d` (`deployment-reach-delta.toml`) |

산출되는 Learning IR은 1절의 형태 그대로이고 차이는 하나다: `Normalizer{Inverse}`의 통계가
`mean = 0`, `std = 0.05`이고 출력 포트 단위가 `Unit::AngularVelocity` — 틱당 라디안이다.
`XIR-031`이 `JointDelta` 배치에 이를 요구하고 `Unit::Angle`을 이름으로 거부한다. 같은 오라클을
S2c의 **위치** 체크포인트에 돌리면 결과가 그대로다(`weights_hash 37fc82a8…`,
`policy_hash 1a18cc4b…`, 티어 (b) 9.704e-7). 즉 행동 종류가 바꾼 것은 그 숫자들의 의미뿐이다.

커밋된 델타 *Task* IR은 없다. T1이 커밋한 것은 `deployment-reach-delta.toml`이고 자기 교차
검사에서는 태스크를 메모리에서 바꾼다. 그래서 CLI 테스트는 `task-reach.toml`의 스크래치
사본에 다른 한 줄(`ActionSpec.space`)만 치환해 넣는다. 그 쌍을 커밋하는 것은 필요하다면 다음
패킷의 몫이다.

### P-M8-R1 — 탐색 노이즈, 오라클 서버(Linux, 16코어 CPU), 2026-09-21 UTC

산출물: `~/artifacts/plan-t/r1/` (`a1-seed0/`, `a2-seed0/`, `a3-seed0/`, `a4-seed0/`,
`a4-seed1/`, `logs/`). 각 실행은 `metrics/loss-curve.json`, `metrics/env-metrics.json`,
`checkpoints/{4000,10000}.esb`, `eval-4000/`, `eval-10000/`을 가진다. 트리
`~/Projects/es-r1-noise` (이 브랜치를 tarball로 올린 것이며 실행 후 삭제),
`cargo build --release -p es`, `es_native`를 그 트리에 맞게 재빌드. 인터프리터
`~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129, mujoco 3.13.0. 레시피
`tests/fixtures/rl/noise/{a1-log-std, a2-no-entropy, a3-cosine, a4-log-std-mid}.toml` —
`training-reach.toml`에서 행마다 필드 하나씩만 움직인 것, `seed = 0`, `steps = 10000`,
`checkpoint_at = [4000, 10000]`, `envs = 16`, `horizon = 64`, CPU 백엔드.
**A0는 S4e에서 인용한 것**(`~/artifacts/plan-s/s4e/run-4000/`, `run-10000/`)이며 다시 돌리지
않았다. 모든 체크포인트는 `es eval run --config tests/fixtures/rl/evaluation-reach.toml`로,
같은 `evaluation_hash f15fe888…`, 홀드아웃 시드 201–216 16개로 채점했고, 모든 번들은
`task_hash b5d3b813…`, `observation_hash 4ced8547…`, `learning_hash eb805f18…`,
`lowering_hash dce8d352…` — S4e의 것 그대로 — 를 지닌다. 이것이 이 표의 전제 조건이지
형식이 아니다(§13.3). A1/A2와 A3/A4는 각각 16코어 박스에서 `nice -n 10`으로 **한 번에 두
개씩** 돌았고, 이는 벽시계만 부풀릴 뿐 다른 것은 건드리지 않는다. A4 시드 1은 혼자 돌았다.
평가는 각각 한 프로세스, 10.9–11.8초.

**표.** 값이 셋인 칸은 `metrics/loss-curve.json`의 반복 1 / 4,000 / 10,000이고,
`success_rate`와 `episode_length`는 두 체크포인트에서의 `nominal` 스위트 값이다:

| id | `init_log_std` | `entropy` | `schedule` | `executed_ne_sampled_rate` | `envelope_violation_rate` | 롤아웃 엔트로피 | 롤아웃 `return` | 홀드아웃 `success_rate` | 홀드아웃 `episode_length` | 벽시계 | `training_hash` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A0** (S4e 인용) | −0.5 | 0.005 | constant | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 5.517 / 4.869 / 6.056 | −15.765 / −4.740 / −4.916 | **0.5625** / 0.3125 | 129.4 / 157.3 | 12m38.0s (4,000) · 29m22.3s (10,000), 둘 다 단독 | `1933697d…` / `69665845…` |
| **A1** | **−2.5** | 0.005 | constant | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −6.481 / −4.918 / −4.680 | −15.573 / −4.752 / −4.440 | 0.0000 / 0.0000 | 200.0 / 200.0 | 34m31.7s | `42632125…` |
| **A2** | −2.5 | **0.0** | constant | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −6.482 / −10.335 / −13.405 | −15.573 / −4.379 / −3.582 | 0.0000 / 0.0000 | 200.0 / 200.0 | 33m29.7s | `bbfd1464…` |
| **A3** | −2.5 | 0.0 | **`warmup_cosine`** (100, 3e-6) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −6.486 / −9.651 / −11.219 | −15.573 / −4.545 / −4.033 | 0.0000 / 0.0000 | 200.0 / 200.0 | 28m2.6s | `74f7d6a3…` |
| **A4** | **−1.5** | 0.0 | A3와 같음 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −0.486 / −6.195 / −8.591 | −15.622 / −4.268 / −3.512 | 0.3125 / **0.3750** | 155.3 / 137.9 | 31m10.5s | `e9556a2e…` |
| **A4, 시드 1** | −1.5 | 0.0 | A3와 같음 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −0.486 / −6.634 / −10.071 | −14.705 / −3.728 / −5.475 | **0.0000** / **0.0000** | 200.0 / 200.0 | 25m28.1s (단독) | `31a2db0b…` |

**클램프 비율은 떨어지지 않으며, 그것이 이 측정의 내용이다.** 세 지점 모두에서 모든 행이
여전히 정확히 1.00이므로 위 표가 무언가를 반올림해 숨기고 있는 것이 아니다. 실행 전체로
세어 보면 — 10,000 반복 × 1,024 행 = 10,240,000개의 샘플된 행동 — 플레인이 손대지 않고
통과시킨 행의 *절대 개수*는 다음과 같다:

| id | σ = exp(`init_log_std`) | 10,240,000 중 클램프되지 않은 행 | 실행 전체 평균 비율 |
|---|---|---|---|
| A0 | 0.607 rad | **0** | 1.0 |
| A1 | 0.082 rad | **8** | 0.99999921875 |
| A2 | 0.082 rad | **181** | 0.99998232421875 |
| A3 | 0.082 rad | **19** | 0.99999814453125 |
| A4 | 0.223 rad | **1** | 0.99999990234375 |
| A4, 시드 1 | 0.223 rad | **8** | 0.99999921875 |

가우시안을 7.4배 줄여서 천만 틱 중 8틱을 벌었다. 이유는 노이즈의 크기가 아니라 행동의
모양에 있다: 행동은 절대 관절 **위치 목표**이고, `deployment-reach.toml`은 그 목표가 *측정된*
관절로부터 50 Hz 틱 하나 동안 얼마나 멀어질 수 있는지를 `velocity_max = 3.0 rad/s` → 0.06
rad으로 묶는다(`action_rate.first_diff_max = 0.08` rad은 둘 중 느슨한 쪽). 그러므로 플레인이
클램프하는 것은 `μ − q`, 즉 네트워크가 명령한 자세와 팔이 실제로 있는 곳 사이의 거리이고,
σ는 이미 경계 밖에 있는 양을 흔들 뿐이다. **열린 질문 2는 노이즈를 줄여서 답할 수 없다** —
σ = 0에서도 비율은 여전히 ~1.00일 것이다 — 그러므로 남은 선택지는 리뷰가 이름 붙인 것들이다:
실행된 행동으로 학습하기(추정량), 엔벨로프를 넓히거나 다시 모양 잡기, 또는 헤드가 절대
목표가 아니라 델타를 내게 하기.

`Rollout.metrics()`가 주는 대로의 §12.4 아홉 지표(`metrics/env-metrics.json`). 이 경로가
결코 돌리지 않는 도메인은 조작된 0이 아니라 `null`이고, `step/s`는 의도적으로 없다:

| 지표 | A0 (S4e, 10,000) | A1 | A2 | A3 | A4 | A4, 시드 1 |
|---|---|---|---|---|---|---|
| `physics_steps_per_sec` | 32,076 | 33,591 | 35,093 | 34,225 | 30,156 | 36,479 |
| `actions_per_sec` | 8,019 | 8,398 | 8,773 | 8,556 | 7,539 | 9,120 |
| `camera_frames_per_sec` | `null` — 이 경로에 렌더러가 없다(§4.3) | `null` | `null` | `null` | `null` | `null` |
| `pixels_per_sec` | `null` — 같은 이유 | `null` | `null` | `null` | `null` | `null` |
| `observation_gb_per_sec` | `null` — `Env`가 계측하지 않는다 | `null` | `null` | `null` | `null` | `null` |
| `policy_inferences_per_sec` | `null` — 추론은 `Env`가 아니라 트레이너 안에 있다 | `null` | `null` | `null` | `null` | `null` |
| `p50_end_to_end_latency` | `null` — 동기 롤아웃, 선언된 지연 없음(3절) | `null` | `null` | `null` | `null` | `null` |
| `p95_end_to_end_latency` | `null` — 같은 이유 | `null` | `null` | `null` | `null` | `null` |
| `gpu_memory_peak` | `null` — CPU 백엔드 | `null` | `null` | `null` | `null` | `null` |
| `chunk_underrun_rate` | `null` — horizon 1, 청크 버퍼 없음 | `null` | `null` | `null` | `null` | `null` |

나머지는 전부 `Target / Status: unverified`. 실행마다 제어 틱 640,000개이고, `Rollout`
시간으로 1,759.9초(A0), 2,069.0초(A1), 2,006.9초(A2), 1,680.0초(A3), 1,867.9초(A4),
1,525.6초(A4 시드 1)이다. 두 개씩 돌린 행이 느린 것은 그 때문이지 트레이너 안의 어떤 이유
때문이 아니다 — 박스에 혼자 있던 A4 시드 1이 표에서 가장 빠른 행이고, 동시에 0점을 받은
행이다.

**A4 시드 0의 교란 스위트 — 교란할 성공이 있는 유일한 실행이다**(게이트가 아니라 측정값,
§10.4). 나머지 행 — A1, A2, A3, 그리고 시드 1의 A4 — 은 두 지점 모두에서 네 스위트 전부
0.0000이다:

| A4 | `nominal` | `observation_delay` | `torque_noise` | `backlash` |
|---|---|---|---|---|
| 4,000, 시드 0 | 0.3125 | 0.0625 | 0.4375 | 0.2500 |
| 10,000, 시드 0 | 0.3750 | 0.0000 | 0.1875 | 0.2500 |
| 10,000, 시드 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**변수마다 한 문장.**

* **`init_log_std`는 성공률을 조금이라도 움직인 유일한 손잡이이고, 클램프 비율은 움직이지
  못했다.** A3 → A4는 이 값만 −2.5에서 −1.5로 움직이는데 10,000에서의 홀드아웃
  `success_rate`가 0.0000 → 0.3750이 되는 동안 클램프 비율은 양쪽 다 1.00이다. A1 → A0는
  반대 방향으로 −2.5에서 −0.5로 움직이고, 작은 σ 행 어느 것도 근처에 못 가는 정점(4,000에서
  0.5625)을 되사온다. **시드 1 재실행은 그 해석을 어디까지 밀 수 있고 그 너머로는 안 되는지를
  말해 준다:** 같은 레시피가 시드 1에서 두 지점 모두 0.0000을 받으므로, A4의 0.3750은 0을
  포함하는 산포에서 뽑은 한 장이고, 다섯 실행이 뒷받침하는 유일한 주장은 부정형이다 —
  σ = exp(−2.5)에서는 두 예산 어느 쪽에서도, 세 가지 엔트로피/스케줄 설정 어느 쪽에서도
  작업이 학습되지 않는다.
* **엔트로피 계수는 롤아웃 엔트로피를 움직이고, 사람이 신경 쓰는 것은 아무것도 움직이지
  않는다.** A1 → A2는 이 값을 0으로 만들고 10,000에서의 엔트로피는 −4.680에서 −13.405로
  떨어진다 — S4e 행이 추측한 대로 보너스가 실제로 σ의 재팽창 비용을 대고 있었다 — 그러나
  `success_rate`는 변화 양쪽 모두 0.0000이므로, 이 작업에서 보너스는 퇴화의 원인도 아니었고
  그것만 떼어낼 가치가 있는 비용도 아니었다.
* **학습률 스케줄은 σ = exp(−2.5)에서 아무것도 바꾸지 않으며, A4의 모양을 그 공으로 돌릴 수
  없다.** A2 → A3는 코사인만 켜는데 둘 다 0.0000 / 0.0000이다(5분 반 빠른 것은 박스를 나눠
  쓴 스케줄링 잡음이지 스케줄의 성질이 아니다). A0가 퇴화하는 자리(0.5625 → 0.3125)에서 A4
  시드 0은 4,000 너머로 버티지만(0.3125 → 0.3750), A0와는 한 번에 세 필드가 다르고 **시드 1
  쌍둥이는 아무것도 버티지 못하므로**(0.0000 → 0.0000), "코사인이 퇴화를 멈췄다"는 이 표가
  뒷받침하는 주장이 *아니다*.

**수용 기준은 여전히 충족되지 않았고, A0를 이긴 변형은 없으며, 두 번째 시드가 동의하는
방식으로 4,000 너머를 버틴 변형도 없다** — 패킷은 그런 일이 벌어지면 그렇게 적으라고 했고,
그런 일이 벌어졌다. 0.8에는 근처도 못 갔다. 4,000에서의 A0의 0.5625가 reach 작업의 최고
수치로 남아 있고, 10,000에서의 A4의 0.3750이 새 레시피 넷 중 최고이며, 그것은 A0 자신의
퇴화한 꼬리(0.3125)와 16회 중 1회 차이이고, 시드 1에서는 0.0000이다. 이 패킷이 사온 것은 더
나은 정책이 아니라 설명 하나의 제거다: 1.00은 잘못 고른 탐색 σ의 부산물이 아니라 행동
공간에 구조적으로 박힌 것이고, 다음에 움직일 것은 레시피가 아니라 추정량, 엔벨로프, 또는
헤드다.

### T3 — 절대 공간 곁의 증분 공간, 오라클 서버(Linux, 16코어 CPU), 2026-09-22 UTC

산출물: `~/artifacts/plan-t/t3/` (`delta-scratch-seed{0,1,2}/`, `delta-continued-seed{0,1,2}/`,
`import-before/`, `imported/`, `neutral-folded/`, `recipes/`, `logs/`). 문서: 델타 삼종
`tests/fixtures/rl/{task,observation,evaluation}-reach-delta.toml`(task-reach에
`ActionSpec.space = JointDelta`를 얹은 것, 나머지 둘은 옮겨간 `task_hash`를 따른다),
`learning-reach-delta.toml`(64×64 relu 그래프, 행동 포트는 `AngularVelocity`이고
`Normalizer{Inverse}`는 틱당 0 / 0.05 rad), `deployment-reach-delta.toml`(T1), 레시피
`training-reach-delta.toml`과 `training-reach-delta-continued.toml`(A0의 `[rl]` 값에
`init_log_std = ln(0.02) = −3.912`만 더한 것 — 단위 변환일 뿐이다 — σ = 틱당 0.02 rad =
0.4 × `delta_scale`이고 새 손잡이가 아니다). 임포트된 델타 정책은 T2의
`~/artifacts/plan-t/t2/seed0-run1/`을 이 작업에 대해 다시 임포트한 것이며, S4c가 했던 대로
brax의 정규화기를 첫 Dense에 접어 넣었다. 패킷을 실행하던 에이전트가 API 장애로 끊긴 뒤,
오케스트레이터가 산출물로부터 이 절을 썼다; 아래 모든 수치는 서버의 `report.json` /
`metrics/loss-curve.json`에서 읽은 것이다.

**홀드아웃 `success_rate`, `nominal`, 시드 201–216 16개, 평균과 시드별.** 절대 공간 행들은
인용이다(S4e 시드 0; S4c `scratch64` 시드 1–2, 4,000에서); 시드 1–2의 절대 공간 10,000 지점은
서버에서 중단되어 `Target / Status: unverified`다.

| 행 | 4,000에서 | 10,000에서 | nominal `envelope_violation_rate`(평가) | 4,000에서 `episode_length` |
|---|---|---|---|---|
| 절대 A0(`training-reach.toml`) | **0.4167** (0.5625 / 0.3125 / 0.3750) | 0.3125 (시드 0) | 1.00 | 143.4 |
| 델타, from scratch | 0.1042 (0.0 / 0.0 / 0.3125) | 0.0833 (0.0 / 0.25 / 0.0) | 0.880 / 0.995 / 0.993 | 183.8 |
| 델타, imported, 학습 전 | 0.0 | — | 0.845 | 200.0 |
| 델타, `[init]` = 임포트 | 0.0 (0.0 / 0.0 / 0.0) | 0.0 (0.0 / 0.0 / 0.0) | 0.845 | 200.0 |

**롤아웃 통계**(`metrics/loss-curve.json`, 반복 1 / 4,000 / 10,000):

| 실행 | `executed_ne_sampled_rate` | `envelope_violation_rate` | 엔트로피 | `return` |
|---|---|---|---|---|
| delta-scratch 시드 0 | 0.994 / 1.00 / 1.00 | 0.801 / 0.997 / 1.00 | −14.95 / −14.07 / −6.92 | −15.73 / −9.33 / −5.89 |
| delta-scratch 시드 1 | 0.992 / 1.00 / 1.00 | 0.771 / 0.998 / 1.00 | −14.96 / −13.51 / −9.26 | −14.31 / −4.76 / −6.92 |
| delta-scratch 시드 2 | 0.992 / 1.00 / 1.00 | 0.783 / 0.996 / 1.00 | −14.96 / −14.00 / −9.66 | −11.79 / −6.32 / −5.22 |
| delta-continued 시드 0 | 1.00 / 1.00 / 1.00 | 0.969 / 1.00 / 1.00 | −14.95 / −8.81 / −2.85 | −8.38 / −11.12 / −9.46 |
| delta-continued 시드 1 | 1.00 / 1.00 / 1.00 | 0.981 / 1.00 / 1.00 | −14.95 / −8.82 / −2.65 | −8.80 / −11.77 / −8.85 |
| delta-continued 시드 2 | 1.00 / 1.00 / 1.00 | 0.973 / 1.00 / 1.00 | −14.96 / −7.26 / +1.12 | −8.53 / −17.04 / −10.21 |

**`es eval compare` 절대 시드 0 대 delta-scratch 시드 0, 둘 다 4,000에서**(`nominal`):
`success_rate` 0.5625 → 0.0(−0.5625), `envelope_violation_rate` 1.00 → 0.880, `episode_length`
129.4 → 200.0; 교란 스위트 전부 같은 방향; 둘 다 `passed = false`. 실패 히스토그램, `nominal`,
4,000에서:

| 정책 | `violation.position` | `violation.velocity` | `violation.acceleration` | 성공 |
|---|---|---|---|---|
| 절대 A0 시드 0 | 2,053 | 2,055 | 1,882 | 9 |
| delta-scratch 시드 2 | 2,303 | 0 | 984 | 5 |
| delta import, 학습 전 | 2,672 | 0 | 79 | 0 |
| delta-continued 시드 0 | 2,672 | 0 | 80 | 0 |

**이것이 말하는 것.** 증분 공간은 속도 상한에 대해 §8.5가 약속한 일을 해냈다 —
`violation.velocity`는 어디서나 0이고, 0.05 rad 증분은 `velocity_max`·dt를 결코 넘지 않는다 —
그러나 클램프 비율은 낮추지 못했는데, 클램프가 **위치 soft 엔벌로프**로 옮겨 갔기 때문이다.
적분기는 매 틱을 플레인의 실행된 목표에서 시작하므로(§8.5, "실행된 값에서 적분하라, 원시
행에서가 아니라"), soft 위치 한계로 밀고 들어가는 정책은 되밀리고 다음 틱에 다시 밀어붙인다:
영구적인 `violation.position`, 순 이동량 0, 그리고 PPO가 배울 수 없는 상수 행동 → 보상
짝이다. From scratch는 4,000에서 0.10으로 절대 공간의 0.42(각각 시드 세 개)에 맞서고 있다;
임포트된 델타 정책은 다시 M8의 S-5다 — 테이블에 접촉이 없는 파생 씬에서 학습되었으므로 우리
엔벌로프가 금지하는 작업공간으로 그리퍼를 몰아넣는다 — 그리고 거기서 이어지는 학습은 0.0에
머무르고, 그동안 엔트로피는 오르지만(−14.9 → −2.8) 아무것도 배우지 않는다. **M9 리뷰가
필요로 하는 문장:** 이 표들은 증분을 §13.4의 RL 기본 행동 공간으로 삼을 이유를 주지 않는다;
증분은 §8.5가 만들어 둔 그대로 남고(임포터가 진짜 상대-행동 정책을 위해 필요로 하는 추가물),
두 캠페인이 함께 가리키는 지렛대는 학습 정책에 대한 엔벌로프의 의미다 — 이 작업의 위치
soft margin, 실행된-행동 추정량, 또는 *측정된* 관절에 대해 적분된 증분 — 이 각각은
Deployment IR / 스펙 결정이지(INV-12: 넓히기만 하고 결코 끄지 않는다) 트레이너 변경이
아니다. 벽시계: 두세 개 학습이 동시에 돌았고, 10,000 반복 실행당 대략 30분; `Rollout.metrics()`가
보고하는 §12.4 아홉 지표는 각 실행의 `metrics/env-metrics.json`에 있고, 나머지는
`Target / Status: unverified`다. 여기서 측정되지 않은 것: 첫 반복의 액터 그래디언트
노름(S-13의 탐지기는 아직 존재하지 않는다).

### W0b — 플랫폼에 흔들리지 않는 `scene_hash` 아래에서 다시 측정한 A0, 오라클 서버(Linux, 16코어 CPU), 2026-09-22 UTC

패킷 `docs/packets/M10/W0b-scene-hash-libm.md`. MJCF/URDF 임포터가 `euler=`, `axisangle=`,
`zaxis=`, `rpy`에 대해 호스트의 libm을 부르지 않고 `es_math::approx::{sin_cos_f64, acos_f64}`를
지나게 되었고, 그래서 `scene_hash`는 Windows와 Linux에서 하나의 숫자다(§3.2의 `f64` 단락).
그것이 모든 SO-101 `task_hash`를 옮기고 reach 문서들도 함께 옮긴다: `task-reach.toml`
`b5d3b813…` → **`43a62f3f…`**, `observation-reach.toml` `4ced8547…` → **`ecabac79…`**,
`evaluation-reach.toml` `f15fe888…` → **`66ef84a5…`**; `learning-reach.toml` `eb805f18…`과
`deployment-reach.toml` `7af05d88…`은 씬을 읽지 않으므로 움직이지 않는다. 전체 옛 → 새 표는
패킷의 노트 절에 있다.

**옮겨진 해시 아래에서 A0를 처음부터 다시 돌렸고, 세 시드 모두 비트 단위로 같다.** 재생성한
reach 문서 넷으로 패킹한 미학습 번들, 시드 0·1·2의 `training-reach.toml`(서버 쪽 사본에서
`[run] seed`만 덮어썼고 그 밖에는 손대지 않았다), 각각 CPU 백엔드에서 4,000 반복, 그다음
재생성한 `evaluation-reach.toml`로 `es eval run`:

| 시드 | `weights/model-4000.safetensors` | S4e / S4c와 다른 텐서 수 | `nominal` `success_rate` | `episode_length` |
|---|---|---|---|---|
| 0 | `d79c5c3a…` | **0 / 8** | 0.5625 | 129.4375 |
| 1 | `935ca7b8…` | **0 / 8** | 0.3750 | 147.1875 |
| 2 | `35fc351a…` | **0 / 8** | 0.3125 | 153.6250 |

평균 **0.4167**, 곧 7절의 S4e·S4c 행과 `visible-learning.md` 7.34·7.35가 이미 지닌 숫자다.
모든 `report.json`의 모든 셀이 커밋된 것과 동일하다 — 네 스위트, 두 지표,
`observation_delay` 0.3125 / 0.0625 / 0.1250과 `torque_noise` 0.5000 / 0.4375 / 0.6250까지.
**이것이 이 저장소 어디에서도 가능한 §28.13 규칙 1의 가장 강한 형태다**: CPU 백엔드 위의 PPO는
비트 단위이므로(`train_rl_two_runs_are_bitwise`), 그 밖의 무엇도 옮기지 않은 해시 수정이라면
체크포인트를 바이트 단위로 재현해야 하고, 실제로 재현한다 — S4e와 S4c가 돌던 트리에서 네
마일스톤이 지난 트리 위에서. `visible-learning.md` 7.36이 데모 쪽 절반을 지닌다. 거기서는
CUDA 위의 ACT 학습이 비트 단위가 아니어서 주장을 한 층 위와 한 층 아래에서 해야 한다.

**움직이는 두 해시와, 각각이 움직이는 이유.** `evaluation_hash` `f15fe888…` → `66ef84a5…`는
`task`와 `observation`을 통한 이 수정 자체다. `execution_hash`는 두 겹으로 움직인다: 시드 0
`9ff75635…` → `145bc81a…`, 시드 1 `22b55e10…` → `7883ef24…`, 시드 2 `d72bee1b…` →
`63f91704…` — 한 번은 문서 때문에, 한 번은 W0a가 정책 런타임의 인트라옵 스레드 수를
`runtime_hash`에 넣었기 때문에. `evaluation.lock`이 이제 그것을 찍는다:
**`runtime_threads: 8`**, 이 상자의 물리 코어 8개에 대한 torch 자신의 기본값이고, `--jobs 1`의
`cores/N` 상한 16은 여기서 구속하지 않는다. S4e와 S4c의 락은 그 필드보다 앞서므로 `None`이다.
스레드 수가 둘이면 조건도 둘이지만(`evaluation-execution.md` 2.7), 여기서는 수가 같고 그것이
*기록된다*는 점만 새롭다.

**벽시계와 스케줄링 한 마디.** 시드 0 / 1 / 2는 **637초 / 618초 / 637초**에 학습하고 11 / 11 /
10초에 채점했으며, 한 번에 하나씩 돌렸다(단계 총 1,925초). 처음에는 패킷이 요구한 대로 셋을
동시에 띄웠다: `es train`은 `es eval run --jobs`가 평가기의 풀을 제한하는 식으로 학습기의
스레드 풀을 제한하지 않으므로, 세 런이 16코어 위에 3 × 8 torch 스레드를 올렸고 **60분** 동안
체크포인트를 하나도 쓰지 못했다 — 혼자 돌 때 755초였던 S4e에 대해서. 그 런들을 죽이고 순차로
다시 돌렸다; 스레드 수는 어느 쪽이든 torch의 기본값이므로 이것은 스케줄링 선택이지 다른 측정이
아니다 — 위의 비트 단위로 같은 체크포인트들이 그것을 가정이 아니라 확인해 준다. 산출물은
`~/artifacts/plan-w/w0b/reach/seed{0,1,2}/`, `training_hash` `2dfbc7c8…` / `6bf728e7…` /
`e3d030e8…`. §12.4의 아홉 지표는 각 런의 `metrics/env-metrics.json`에 있고, 나머지는
`Target / Status: unverified`다.

### P-M9-R5 — A0 곁의 실행된-행동 추정량, 오라클 서버(Linux, 16코어 CPU), 2026-09-22 UTC

산출물: `~/artifacts/plan-w/r5/` (`executed-seed{0,1,2}/`, `recipes/`, `logs/`, `run.sh`). 레시피:
R1 예산(`steps = 10000`, `checkpoint_at = [4000]`, `[run] seed` 0/1/2)의
`tests/fixtures/rl/training-reach-executed.toml`, A0와 같은 미학습 번들; `evaluation-reach.toml`
(held-out 시드 16개)로 채점. 패킷 에이전트가 끝난 뒤 오케스트레이터가 서버의 `report.json` /
`metrics/loss-curve.json`에서 읽어 작성했다.

**held-out `success_rate`, `nominal`, 시드 16개, 평균과 시드별.**

| 행 | 4,000에서 | 10,000에서 | nominal `envelope_violation_rate` (평가) | `episode_length` |
|---|---|---|---|---|
| 절대 A0, `estimator = "sampled"` (T3에서 인용) | **0.4167** (0.5625 / 0.3125 / 0.3750) | 0.3125 (시드 0) | 1.00 | 4,000에서 143.4 |
| 절대, `estimator = "executed"` | 0.0 (0.0 / 0.0 / 0.0) | 0.0 (0.0 / 0.0 / 0.0) | 1.00 | 둘 다 200.0 |

섭동 스위트(`observation_delay`, `torque_noise`, `backlash`)도 세 시드 모두 두 지점에서 0.0이다.

**롤아웃 통계** (`metrics/loss-curve.json`, 반복 1 / 4,000 / 10,000):

| 실행 | `executed_ne_sampled_rate` | 엔트로피 | `return` |
|---|---|---|---|
| executed 시드 0 | 1.00 / 1.00 / 1.00 | 5.50 / 2.38 / 2.37 | −15.76 / −21.04 / −27.29 |
| executed 시드 1 | 1.00 / 1.00 / 1.00 | 5.50 / 2.35 / 2.35 | −14.72 / −17.47 / −17.60 |
| executed 시드 2 | 1.00 / 1.00 / 1.00 | 5.50 / 1.76 / 1.76 | −12.32 / −7.85 / −11.37 |

**실패 히스토그램, `nominal`, 4,000에서** (A0 시드 0은 T3에서 인용): A0 `violation.position` 2,053,
`violation.velocity` 2,055, 성공 9; executed 시드 0/1/2 `violation.position` 2,866 / 3,184 / 2,280,
`violation.velocity`는 각각 3,184 — 16 × 199 틱 전부 — 성공 0.

**이것이 말하는 것.** 옵션 B는 도움이 되지 않는다; 아무것도 하지 않는 것보다 나쁘다. 로그 확률을
실행된 행동에서 취하면 정책은 뻗기를 멈춘다: 세 시드 중 둘에서 return이 떨어지고, 엔트로피는 한 번
내려간 뒤 얼어붙으며(시드 1은 4,000과 10,000에서 모두 2.3546, 즉 `log_std`가 더 이상 움직이지
않는다), 평가에서는 모든 틱이 속도 상한을 위반한다. 메커니즘은 엔벌로프가 아니라 추정량이다:
플레인의 클램프는 샘플의 반직선 전체를 하나의 경계값으로 보내므로, `log N(executed; mu, sigma)`는
가우시안이 거의 제안하지 않은 한 점의 밀도다 — 비율은 그 같은 점을 두 번 평가한 것이고,
그래디언트는 `mu`를 보상을 얻은 행동 쪽이 아니라 클램프 경계 쪽으로(양의 이점) 또는 그 반대로(음의
이점) 끈다. **다음 결정이 필요로 하는 문장:** 곡선은 4,000을 지나 유지되지 않고 어느 시드에서도
10,000에서 A0를 넘지 못한다; 기본값은 `"sampled"`로 남고 `"executed"`는 측정된 음성 결과로서
레시피 어휘에 남는다. 남은 선택지는 패킷의 A(서보 사양의 근거가 있을 때만, 더 넓은 위치 여유),
C(*측정된* 관절에 대한 증분), 또는 클램프를 검열(censoring)로 모델링하는 추정량 — clipped-action
policy gradient(Fujita & Maeda, 2018), 클램프된 차원에 대해 가우시안 꼬리 질량의 로그 — 이며,
마지막 것은 엔벌로프와 INV-11..13을 건드리지 않는 트레이너 변경이고 이 패킷이 할 일은 아니다.
벽시계: 세 학습이 같은 호스트에서 plan W의 W1a PT 실행과 동시에 돌았고(16코어에 부하 ≈ 24), 셋
모두 19,076 s ≈ 5.3 h; 평가 여섯 번 67 s. §12.4 아홉 지표는 각 실행의 `metrics/env-metrics.json`에
있고, 나머지는 `Target / Status: unverified`다.

### I3 — 우리 장면에서 학습한 Isaac Lab 정책과 A0, 세 엔진에서, 오라클 서버(RTX 4090, Ubuntu 26.04.1), 2026-09-23 UTC

패킷 `docs/packets/M11/I3-sim-to-sim-measured.md`. 산출물 `~/artifacts/plan-x/i3/`(`usd/`,
`train{0,1,2}/`, `select{0,1,2}.json`, `heldout{0,1,2}.json`, `delay{0,1,2}.json`,
`import/isaac-{0,1,2}/`, `eval/`, `scenes/`, `docs/`, `v1-xyzw/`, 단계 스크립트와 그
`<stage>.{start,end,done,log}` 마커). 트리 `~/Projects/es-i3`(`git archive`, I3 파일은 바뀔 때마다
복사), `cargo build --release -p es`. `~/venvs/es-isaac`에 Isaac Sim 5.1.0 + Isaac Lab 2.3.2.post1 +
rsl-rl-lib 3.0.1; `~/venvs/es`에 MuJoCo 3.13.0, mujoco_warp, torch 2.14 CPU.

**Isaac 작업은 Task IR을 따라 했고, 조정하지 않았다.** `python/es/rl_source/isaac_so101_reach/`는
manager-based env cfg(`env_cfg.py`)와 rsl_rl 드라이버(`main.py`)다. 로봇은 `main.py build-usd`가
`scene_to_mjcf`가 내보내는 MJCF(`--backend physx` 실행에서 `ES_PHYSX_DUMP_MJCF`로 덤프)를
`physx_ref.import_scene` — 이를 위해 `Sim._build_stage`에서 떼어낸 백엔드 자신의 임포트 함수 — 로
통과시켜 쓴 `robot.usd`이므로, 정책이 학습하는 스테이지와 `--backend physx`가 평가하는 스테이지는
같은 여덟 가지 수정을 가진다. 행별로 따라 한 것:

| Task IR (`task-reach-last-action.toml`) | Isaac env |
|---|---|
| MJCF의 5 ms 스텝 위 50 Hz, 200 제어 스텝 | `decimation 4`, `sim.dt 0.005`, `episode_length_s 4.0`(단언: `max_episode_length == 200`) |
| `joint_pos`, `joint_vel` | `mdp.joint_pos_rel`, `mdp.joint_vel_rel` × 0.05(어댑터: `offset = "default_pos"`, `scale = 0.05`) |
| `cube_pose` = `JointState { cube, dof 7 }`: 자유 조인트의 `qpos`, 쿼터니언 **w 먼저** | `free_joint_qpos`: 루트 좌표계 − env 원점, Isaac 자신의 w 먼저 쿼터니언 |
| `gripper_pose` = `BodyPose(gripper)`: `xpos ‖ xquat`, 쿼터니언 **x 먼저** | 링크 `gripper`의 `body_pose_xyzw` |
| `last_action`(`PreviousAction`, `initial` = 휴지 자세) | `mdp.last_action`(원시, 리셋 시 0); `default_joint_pos` = 그 자세 |
| `JointPosition`, ctrl은 `ctrlrange`로 클램프 | `JointPositionActionCfg(scale 0.5, use_default_offset, clip = ctrlrange)` |
| 제어 스텝당 `-1 · dist + 1 · (dist < 0.03)` | 같은 항을 가중치 ∓`1/step_dt`로(보상 관리자가 `dt`를 곱한다) |
| `Terminate Success` / `Timeout` | `DoneTerm(reached)` / `DoneTerm(time_out, time_out=True)` |
| `ResetState` 조인트 0; 큐브 x U[0.21, 0.27], y U[−0.03, 0.05], z 0.02 | `reset_joints_by_scale(0, 0)`; (0.24, 0.01, 0.02) 주위 ±(0.03, 0.04)의 `reset_root_state_uniform` |
| kp 998.22, kv 2.731, forcerange 2.94, armature 0.028, 조인트 감쇠 0.6, frictionloss 0.052 | `ImplicitActuator` stiffness kp, damping kv, `effort_limit_sim` 2.94, armature는 USD에서; 물리 스텝마다 명시적 effort `-0.6·q̇`(`physx_ref.py`처럼); frictionloss는 버림(`physx_ref.py`처럼) |
| 한 장면의 env들 | `env_spacing = 0`, env 간 충돌 필터(`physx_ref.py`의 `GridCloner(spacing = 0)`; 또한 강제됨, `isaac-lab.md` §9) |

가정하지 않고 확인했다: `q = 0`에서 env의 그리퍼 자세는 MuJoCo의 것과 1e-4 m 이내(0.2932,
−0.0002, 0.2344; x 먼저 쿼터니언 (0.0172, −0.7069, −0.0172, 0.7069))이고, 휴지 자세에서 안정된
뒤 2e-4 m 이내다. **따라 하지 않은 것** — 각각 Isaac 정책이 우리 런타임에서만, 또는 Isaac에서만
만나는 차이:

1. **Safety Plane**(Deployment IR): 속도 3 rad/s, 가속도 80 rad/s², 행동 변화율 틱당 0.08 / 0.04
   rad, 위치 소프트 여유, 작업 공간. Isaac에는 없다; A0는 그 아래에서 학습했다.
2. `es eval run`의 **제어 틱 하나의 행동 지연**: 임포트가 `expected_latency_ms = min(예산, 주기)
   = 20 ms`를 선언하고 `latency_ticks` = 1(A0의 2 ms도 1틱). Isaac은 0으로 학습하고 채점한다.
3. **장면 수준 PhysX 설정.** Isaac Lab의 `PhysxCfg` 대 `physx_ref.py`의 `World`: GPU broadphase 대
   MBP, CCD 끔 대 켬, GPU dynamics 켬 대 끔, 그리고 `World`가 쓰지 않는 bounce / friction-offset /
   반복 횟수 속성(`isaac-lab.md` §9). Isaac은 4,096 env로 GPU 파이프라인에서 학습한다.
4. **학습 쪽에만:** 첫 에피소드의 `init_at_random_ep_len`(rsl_rl), 관측 노이즈 없음(Task IR이
   선언하지 않음), 거리에 대한 Task IR의 `Normalize{0..1}`(1 m 아래에서는 동일).

**학습.** Isaac Lab 자신의 reach 러너 설정(`FrankaReachPPORunnerCfg`를 필드 그대로 복사: 24 스텝
× 4,096 env, [64, 64] ELU, lr 1e-3 adaptive, 경험적 정규화 없음)의 rsl_rl PPO, 시드당 1,500 반복 =
147.5 M env 스텝. 체크포인트는 **Isaac 쪽에서만** 고른다: 100번째마다의 체크포인트를 시드 2000+s의
리셋 1,024개에서 결정론적으로 채점해 가장 좋은 것을 택하고, 보고하는 수치는 두 번째 리셋
1,024개(시드 1000+s)의 것이다.

| 시드 | 학습 벽시계 | rsl_rl 성공률(확률적) 100 / 500 / 1,000 / 1,500 | 선택 | **Isaac 보류** | 1,000에서 | 1,499에서 |
|---|---|---|---|---|---|---|
| 0 | 2,648 s | 0.170 / 0.948 / 0.969 / 0.970 | `model_700` | **0.975** | 0.968 | 0.978 |
| 1 | 2,198 s | 0.238 / 0.583 / 0.623 / 0.609 | `model_900` | **0.631** | 0.619 | 0.623 |
| 2 | 2,244 s | 0.203 / 0.644 / 0.651 / 0.643 | `model_1000` | **0.686** | 0.686 | 0.648 |

셋 모두 정체한다(시드 0은 600 반복까지, 시드 1과 2는 900까지); 평균 **0.764**. 시드 2의 선택된
체크포인트는 한 프로세스에서 같은 리셋에 대해 0.6855, 이어서 0.6680을 냈다: Isaac의 GPU
파이프라인도 실행마다 재현되지 않는다.

**임포트.** 파일은 rsl-rl-lib 3.0.1의 클래식 형태다(`std`, `actor.{0,2,4}`, `critic.*`를 담은
`model_state_dict`; `isaac-lab.md` §9). `import_rl.py --from rsl-rl --activation elu --isaac-env-cfg
params/env.yaml --joint-names …` 다음 `es policy import-rl`을 `adapter-isaac-so101.toml`로
`task-reach-last-action.toml` + `deployment-reach.toml`에 대해: 셋 모두 `observation_hash
ace0eba4…`(`evaluation-reach-last-action.toml`이 이름 붙이는 것)와 `learning_hash 800a232c…`를
가진다; `policy_hash` `2b05615c…` / `79c80b31…` / `36898e15…`. 매핑 보고서의 timing 줄은
`decimation 4 x sim_dt 0.005 = 0.02 s == the Deployment IR's control period`이고, 여섯 `damping`
행은 경고다: 소스에서 2.731, 장면에서 3.331(kv + 조인트 감쇠) — Isaac 쪽에서 명시적 수동 항은 구동
게인이 아니다.

**표.** `nominal`의 `success_rate`, 보류 시드 16개(201–216); Isaac 행은
`evaluation-reach-last-action.toml`이, A0 행(W0b의 `4000.esb`)은 `evaluation-reach.toml`이 채점한다
— 같은 시드, 스위트, 합격 기준. 괄호 안은 `episode_length`. `envelope_violation_rate`는 모든 칸에서
1.0이다(모든 에피소드가 적어도 한 틱은 클램프된다).

| 정책 | Isaac 쪽 | physx CPU (r1 = r2) | physx GPU (r1 = r2) | mujoco-cpu | mjwarp r1 / r2 |
|---|---|---|---|---|---|
| Isaac 0 | 0.975 | 0.6875 (105.8) | 0.6875 (107.2) | **0.8750** (83.8) | 0.9375 (62.8) / 0.8125 (84.6) |
| Isaac 1 | 0.631 | 0.5000 (125.4) | 0.3750 (146.1) | 0.6250 (125.5) | 0.4375 (141.2) / 0.3125 (152.6) |
| Isaac 2 | 0.686 | 0.3750 (148.3) | 0.6250 (105.0) | 0.1875 (172.9) | 0.3750 (153.6) / 0.2500 (160.3) |
| A0 0 | — | 0.1250 (184.9) | 0.1250 (183.4) | 0.5625 (129.4) | 0.4375 (141.4) / 0.4375 (141.3) |
| A0 1 | — | 0.1875 (176.8) | 0.0000 (200.0) | 0.3750 (147.2) | 0.3750 (146.6) / 0.4375 (137.3) |
| A0 2 | — | 0.2500 (166.1) | 0.1250 (181.2) | 0.3125 (153.6) | 0.1875 (171.4) / 0.1875 (171.4) |
| **평균** Isaac / A0 | 0.764 / — | 0.521 / 0.188 | 0.563 / 0.083 | 0.563 / 0.417 | 0.583 / 0.333 (r1), 0.458 / 0.354 (r2) |

칸별 `execution_hash`(앞 8자리 16진수; 각 행의 physx 두 실행은 보고서 하나, 해시 하나를 냈다 —
I1이 측정한 대로 비트 동일; mjwarp 두 실행은 해시는 같고 수치는 다르다, X1의 tier 2 행):

| 정책 | physx CPU | physx GPU | mujoco-cpu | mjwarp |
|---|---|---|---|---|
| Isaac 0 | `9b7fad37` | `815f8064` | `f9eb7538` | `e52b4ed8` |
| Isaac 1 | `51a0dc8b` | `87cf5708` | `7ad08aa8` | `eba58084` |
| Isaac 2 | `d0b4b213` | `6e61f76d` | `c0e8f22a` | `a44788f7` |
| A0 0 | `37a6bc7a` | `e121afd9` | `08851281` | `7456e37d` |
| A0 1 | `9064ff63` | `409e93f4` | `e9566999` | `03a25132` |
| A0 2 | `2e352bc6` | `888b4ce0` | `808658ce` | `f9a760c0` |

**귀속**(`nominal` `success_rate`, 16 시드). physx(CPU 파이프라인)에서는 PhysX 어댑터가 켜고 끌 수
있는 근사 행 둘 — `ES_PHYSX_JOINT_DAMPING=none`(명시적 `-d·q̇` 없음)과
`ES_PHYSX_FRICTION_COMBINE=average` — 을 바꾸며, 각각 엔진 버전에, 따라서 해시에 기록된다.
mujoco-cpu에서는 PhysX가 버리거나 옮기는 행 둘을 거꾸로 MuJoCo에서 뺀다: `frictionloss` 없는,
`damping` 없는, 둘 다 없는 장면 사본(`scenes/`, sha256 `9f2769ed…`, `79b5fbaa…`, `752c725b…`).
"넓은 엔벌로프"는 진단용 Deployment IR이다(`docs/`, `deployment_hash 22473ac6…`): 속도 100 rad/s,
가속도 1e5, 행동 변화율 틱당 10 rad, ee 속도 100 m/s; 위치, 토크, 작업 공간은 그대로, plane은 켜진
채(INV-12). "지연 1"은 보류 리셋에서 `--action-delay 1`로 돌린 Isaac 자신의 평가다.

| 정책 | physx | physx, 조인트 감쇠 없음 | physx, 마찰 average | physx, 넓은 엔벌로프 | mujoco | mujoco, frictionloss 없음 | mujoco, 감쇠 없음 | mujoco, 둘 다 없음 | mujoco, 넓은 엔벌로프 | Isaac, 지연 0 → 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Isaac 0 | 0.6875 | 0.0000 | 0.6875 | 0.9375 | 0.8750 | **1.0000** | 0.0625 | 0.1250 | 0.9375 | 0.975 → 0.725 |
| Isaac 1 | 0.5000 | 0.0000 | 0.5000 | 0.4375 | 0.6250 | 0.4375 | 0.0625 | 0.1250 | 0.2500 | 0.631 → 0.585 |
| Isaac 2 | 0.3750 | 0.0000 | 0.3750 | 0.4375 | 0.1875 | 0.3125 | 0.0000 | 0.0000 | 0.4375 | 0.686 → 0.543 |
| A0 0 | 0.1250 | 0.0625 | 0.1250 | — | 0.5625 | 0.5000 | 0.1875 | 0.0000 | — | — |
| A0 1 | 0.1875 | 0.1875 | 0.1875 | — | 0.3750 | 0.4375 | 0.3125 | 0.1250 | — | — |
| A0 2 | 0.2500 | 0.0625 | 0.2500 | — | 0.3125 | 0.5625 | 0.2500 | 0.4375 | — | — |

행들이 수치로 말하는 것. **마찰 결합은 여기서 구조상 아무것도 설명하지 않는다**: 장면의 충돌 geom은
모두 μ = 1, 재질 하나이고 max = average = min = 1 — 열이 기본 칸과 칸마다 같다(해시는 그래도
움직인다). **조인트 감쇠는 양쪽 모두에서 하중을 받는다**: 명시적 `-0.6·q̇` 없이는 모든 Isaac 정책이
physx에서 0.0이고, MuJoCo의 감쇠를 빼도 mujoco-cpu에서 모든 정책이 떨어진다 — *근사된*(암시적 대신
명시적) 행이 격차가 아니라, 행 자체가 필수다. PhysX가 버리는 **frictionloss**는 MuJoCo에서 Isaac 0을
1.0으로, A0 2를 0.31에서 0.56으로 옮기고, 나머지 넷은 어느 쪽으로든 최대 0.19 옮긴다. **엔벌로프**는
Isaac 0을 physx에서 0.69에서 0.94로, MuJoCo에서 0.88에서 0.94로, Isaac 2를 MuJoCo에서 0.19에서
0.44로 옮기고, Isaac 1을 MuJoCo에서 0.63에서 0.25로 *내리며*, physx의 Isaac 1과 2는 에피소드 하나만큼
옮긴다. 여섯 정책 모두에 대해 "격차의 대부분"인 행은 없다: 에피소드 하나가 0.0625인 16 에피소드에서
엔진 격차는 정책마다 다르다.

**첫 번째 Isaac 정책 묶음은 잘못 따라 한 채널로 학습했고, 여기서 0.0을 냈다.** 첫 env cfg는
`cube_pose`를 x 먼저로 내보냈다 — 5절이 자세를 설명하는 방식대로. 런타임은 Task IR의 `cube_pose` —
`JointState { cube, dof = 7 }` 채널 — 를 자유 조인트의 `qpos`에서 가공 없이, **w 먼저**로 제공한다
(`es_eval::runner::Capture::Qpos`); `BodyPose`인 `gripper_pose`만 x 먼저다. 그 정책들(`v1-xyzw/`)은
Isaac 쪽에서 0.632 / 0.006 / 0.806을, 시드 0은 우리 쪽에서 physx CPU 0.0, mujoco-cpu 0.0625를 냈다 —
신경망이 `(0, 0, 0, 1)`을 배운 자리에 항등 쿼터니언이 `(1, 0, 0, 0)`으로 들어간다. 수정은 Isaac 쪽에
있다(`free_joint_qpos`); 5절의 문장은 `gripper_pose`에 대해서는 맞고 `cube_pose`에 대해서는 틀리며,
어댑터는 채널 안을 순열할 수 없으므로 어떤 소스든 이를 따라 해야 한다.

**런타임에 대한 두 발견, 여기서 고치지 않음.** (1) `es eval run --scene`은 Task IR의 `scene_hash`에
묶여 있지 않다: 위의 MuJoCo 장면 변형 셋은 `task-reach*.toml`에 대해 거부 없이 돌았고, 그
`execution_hash`는 기본 장면의 것이다(A0 0은 넷 모두 `08851281…`) — 다른 장면이 해시 체인에게는
다른 조건이 아니다. 그래서 귀속 행은 해시가 아니라 장면 파일의 sha256으로 식별한다. (2)
`task-reach-last-action.toml`은 `TaskIr::validate`에서 실패했다(`TASK-001`, "declared channel
last_action has no ObservationSpec node"). 그래서 `PreviousAction` 번들은 아예 만들 수 없었다; 이제
규칙은 루프가 제공하고 어떤 그래프 노드도 계산하지 않는 `PreviousAction` 채널을 건너뛴다
(`crates/es-ir/src/task.rs`, 이 패킷의 context 밖; 한 줄, 움직이는 해시 없음).

**답.** **아니다, Isaac Lab에서의 점수는 나오지 않는다 — 그러나 엔진 사이에서는 우리 정책보다 잘
버틴다:** 깨끗하게 임포트된(`observation_hash` 하나, timing 검사 통과) Isaac Lab 정책 셋은 우리
physx CPU 열에서 평균 **0.52**(0.69 / 0.50 / 0.38), Isaac 자신의 보류 리셋에서 **0.76**(0.975 /
0.631 / 0.686)이며, `es eval run`이 적용하는 제어 틱 하나의 행동 지연이 Isaac 쪽만으로 그중 0.15를
설명한다(같은 1,024개 리셋에서 `--action-delay 1`로 0.76 → 0.62); 엔진 사이에서는 physx CPU / physx
GPU / mujoco-cpu / mjwarp에서 평균 0.52 / 0.56 / 0.56 / 0.58–0.46인 반면, A0는 학습한 엔진인
mujoco-cpu의 0.42에서 physx CPU 0.19, physx GPU 0.08로 떨어진다(mjwarp 0.33–0.35) — PhysX에서
학습한 정책이 MuJoCo로 옮겨 가는 것이 MuJoCo에서 학습한 정책이 PhysX로 옮겨 가는 것보다 낫고, 귀속
행들은 조인트 감쇠를 두 엔진 모두에서 필수로, PhysX가 버리는 frictionloss를 MuJoCo에서 가장 크게
움직이는 단일 행으로 지목하되, 격차를 혼자 설명하는 한 행이 아니라 정책마다 그렇다.

벽시계: Isaac 학습은 GPU에서 학습 2,648 / 2,198 / 2,244 s(한 번에 시드 하나, GPU 큐 잠금 아래);
Isaac 쪽 선택과 보류 실행은 시드당 약 2.5분; `es eval run` 한 번(16 에피소드 × 4 스위트)은
mujoco-cpu ≈ 12 s, physx CPU ≈ 5.5분, physx GPU ≈ 7.5분, mjwarp ≈ 3.5분. 나머지는 모두
`Target / Status: unverified`.

### X7 — 경로 추적기 위의 비전 RL, 무작위화 켬과 끔, 오라클 서버(Linux, RTX 4090)

패킷 `docs/packets/M11/X7-vision-rl-pt.md`, spec 28.14 wave 3. 첫 실행은 예산에서 멈췄다(아래
"측정, 그리고 예산 정지"). P-M11-R3가 학습기를 CUDA로 옮기고 `-pix` 행을 더했으며, 2026-09-24
소유자 결정에 따른 재실행(이 절 끝의 "재실행")이 네 `-pix` 행에서 stage 1과 stage 2를 돌려
패킷의 질문에 답한다.

**문서.** `regenerate_x7_documents`(`crates/es/tests/cli.rs`)가 `task-reach-vision.toml`,
`observation-reach-vision.toml`, `evaluation-reach-vision.toml`로부터 네 행을 쓴다:
`pt-dr`, `rs-dr`, `pt`, `rs`(`task-`, `observation-`, `evaluation-reach-vision-<row>.toml`).
`Pt` 행은 X3의 센서(3 bounces, exposure 64, `seed = "tick"`)를 stage 1이 고른 spp와 SVGF로
유지하고, `Rs` 행은 기본 render 블록을 가진다. `-dr` 행은 다음 `Randomization` 대상을 더하며,
각각 자기 스트림 `dr.<target>` 위에 있고 모두 `Uniform`이다:

| 대상 | 범위 | 출처 |
|---|---|---|
| `light.radiance` (`Pt`의 태양; `Rs`에서는 거부되므로 `pt-dr`에만) | 0.5–2.0 | R2 |
| `light.intensity` | 0.7–1.3 | X5 |
| `light.direction` (yaw, 도) | −30–30 | X5 |
| `light.color` | 0.7–1.3 | X5 |
| `light.ambient` | 0.5–2.0 | X5 |
| `geom.bin_floor.rgba` | 0.5–1.5 | X5 |
| `camera.overhead.fov` | 0.85–1.15 | X5 |
| `camera.overhead.pose.{x,y,z}` (m) | −0.02–0.02 | X5 |
| `camera.overhead.pose.{roll,pitch,yaw}` (도) | −4–4 | X5 |
| `body.cube.mass` | 0.8–1.2 | X4 |
| `geom.cube_geom.friction` | 0.8–1.2 | X4 |
| `actuator.<servo>.gain`, 서보 여섯 개 모두 | 0.9–1.1 | X4 |

평가 문서는 `evaluation-reach-vision.toml`(시드 201–216, `nominal`, `observation_delay`,
`torque_noise`, `backlash`)에 데모의 `light_intensity`(0.5–1.5)와 `light_direction`(45°) 스위트를
더한 것이다. 레시피 `training-reach-vision-<row>.toml`은 `training-reach.toml`의 것(16 envs × 64
steps, epochs × minibatches 4 × 4, 4,000 iterations)을 그 행의 번들 위에 `device = "cuda"`로 둔
것이다.

**Stage 1의 선택 규칙, 돌기 전에 고정.** 여섯 개의 `pt-dr` 실행(spp ∈ {4, 8, 16} × SVGF {off,
on}, 1,000 iterations, seed 0) 각각을 **벽시계 시간당 return 이득**으로 채점한다:
`(final_return − initial_return) / (wall_clock_s / 3600)`. 여기서 `initial_return`과
`final_return`은 `train_ppo.py` 자신의 요약에서 처음과 마지막 10 % iteration의 평균이고,
`wall_clock_s`는 `metrics/env-metrics.json`의 값이다. 값이 가장 큰 행이 stage 2의 학습 설정이
된다. 어떤 행도 이득이 없으면(모든 값 ≤ 0), 이 예산에서 학습함을 보인 행이 없는 것이므로 가장
싼 행(`wall_clock_s`가 가장 작은 행)을 고른다. 행마다 시드 하나: 선택은 설정이지 주장이 아니며,
stage 1의 어떤 숫자도 학습에 관한 결과로 보고하지 않는다(§28.14 rule 7).

**측정, 그리고 예산 정지 (2026-09-23/24 UTC, `~/artifacts/plan-x/x7/`).** 인터프리터
`~/venvs/es-lerobot-cuda/bin/python`(torch 2.11.0+cu129), 이 커밋에서 `render`로 빌드한 `es`와
`es_native`, CPU 물리 백엔드, 16 env에 걸쳐 배치된 `Rollout`(X3b).

- **`--device cuda`는 돌지 않는다.** 커밋된 레시피로 돌린 5-iteration 스모크 두 개가 모두 첫
  forward에서 멈췄다: `RuntimeError: Expected all tensors to be on the same device … mat1 is on
  cpu`(`failed-cuda/smoke-*.log`). `train_ppo.py`는 actor를 장치로 옮기지만 관측과 모든 rollout
  버퍼는 CPU에 둔다. 상태 전용 실행은 모두 `cpu`로 돌았기 때문에 이것을 만난 적이 없다. 트레이너는
  이 패킷 밖이므로, 아래 모든 실행은 `device = "cpu"`로 덮어썼고(서버 스크립트의 레시피 단계),
  그래서 ResNet18 업데이트가 CPU에서 돈다.
- **스모크, 각 5 iteration, `cpu`:** `pt-dr`(16 spp) iteration당 28.0 s,
  `render_ms_per_frame` 5.44; `rs-dr` iteration당 23.0 s, 프레임당 0.52 ms. 두 경로 모두에서
  모든 대상 — render와 물리 — 이 컴파일되고 돌았다.
- **Stage 1, 돈 한 행** (`pt-dr`, 4 spp, SVGF off, seed 0, 1,000 iterations):

| spp | SVGF | 벽시계 | s / iteration | `render_ms_per_frame` | `initial_return` | `final_return` | 이득 / h | 엔트로피 처음 → 마지막 100 it. |
|---|---|---|---|---|---|---|---|---|
| 4 | off | 6.50 h | 23.4 | 1.56 | −8.91 | −10.64 | −0.27 | 5.63 → 7.49 |
| 4 | on | 돌지 않음 | | | | | | |
| 8 | off / on | 돌지 않음 | | | | | | |
| 16 | off / on | 돌지 않음 | | | | | | |

  1,000 iteration 동안 어떤 rollout 에피소드도 성공으로 끝나지 않았고(끝난 에피소드는 모두 200
  스텝 timeout까지 갔다), 엔트로피는 100-iteration 구간마다 올랐으며, `executed_ne_sampled_rate`는
  이전의 모든 reach 실행처럼 1.00이었다. §12.4 아홉 지표(`metrics/env-metrics.json`):
  `physics_steps_per_sec` 27,341; `actions_per_sec` 6,835; `camera_frames_per_sec` 639;
  `pixels_per_sec` 5.89e6; `observation_gb_per_sec`, `policy_inferences_per_sec`,
  `p50_end_to_end_latency`, `p95_end_to_end_latency`, `gpu_memory_peak`,
  `chunk_underrun_rate`는 `null`(S4e처럼 이 경로에서 계측되지 않음).
- **멈추게 한 추정.** 23.4 s iteration 중 렌더는 1.6 s이고 나머지는 CPU 학습기다. Stage 1의
  나머지 다섯 행은 ≈ 36 h가 더 걸린다(스모크의 프레임당 렌더 비용으로 4 spp ≈ 6.5 h에서 16 spp
  ≈ 7.8 h). 레시피의 4,000 iteration에서 stage 2는 평가 전에 12 실행 × ≈ 25–26 h ≈ **305 h**이고,
  1,000 iteration이면 ≈ 76 h다. 둘 다 이 패킷이 따른 60 GPU-시간 한도를 넘으므로, stage 1의
  남은 행은 건너뛰었고 stage 2는 시작하지 않았다. 측정된 속도 위의 산술, `Target / Status:
  unverified`.
- **문서 안의 교란 요인.** 비전 행은 `observation-reach-vision.toml`의 26폭 `state` 포트를
  그대로 가지며, 거기에 큐브의 자세(`cube_pose`, 7)가 들어 있다. 그래서 카메라는 상태가 이미
  정확히 주는 것 이상을 정책에 보여 주지 않고, 이 행들은 "*픽셀로부터* 배우는가"에 답할 수 없다.
  그 답에는 상태에 `cube_pose`가 없는 행이 필요하다.

커밋된 `Pt` 문서는 X3의 16 spp, SVGF off(`crates/es/tests/cli.rs`의 `X7_SPP`, `X7_SVGF`)에
머문다. 재실행은 새 파일인 4 spp 형제 문서로 학습한다(`X7_RERUN_SPP`, `x7_pt_pix_variants`).

**P-M11-R3: CUDA 위의 학습기와 `-pix` 행 (2026-09-24 UTC, `~/artifacts/plan-x/r3/`).**
`train_ppo.py`는 이제 actor나 value net에 닿는 모든 텐서를 `--device`에 둔다. 잡음과 순서
생성기는 CPU에 남고, 각 추출은 만들어진 뒤 옮겨진다. `--device cpu`는 이전과 같은 바이트를 쓴다.
옛 trainer와 새 trainer를 같은 인자로 돌리면 state reach 모듈과 vision 모듈(Windows, `.venv`)
모두에서 체크포인트(0, 1, 3, 최종), value 파일, stdout, loss curve(`samples_per_sec` 제외)가
같았고, `train_rl_*` cli 테스트가 통과한다. CPU가 아닌 곳에서는 `use_deterministic_algorithms`가
`warn_only`로 돌고(이번 실행에서 경고는 출력되지 않았다) cuBLAS에
`CUBLAS_WORKSPACE_CONFIG=:4096:8`이 주어진다. trainer는 stderr에
`seconds_per_iteration collect … update …`를 출력한다. X7의 각 행에 `-pix` 형제가 생겼다
(`observation-`, `evaluation-reach-vision-<row>-pix.toml`, `learning-reach-vision-pix.toml`,
`training-reach-vision-<row>-pix.toml`): `cube_pose`가 빠진 19폭 `state` 포트이고, task는 부모
행의 것이다.

서버, 커밋된 16 spp `pt-dr` 문서, recipe의 16 env × 64 step, seed 0, 인터프리터
`~/venvs/es-lerobot-cuda/bin/python`. 렌더 초는 `render_ms_per_frame` × 1,024 프레임이고,
rollout은 `collect`에서 렌더를 뺀 값이다.

| 실행 | device | iteration | 초 / iteration (wall) | 렌더 | rollout | 학습기 (`update`) |
|---|---|---|---|---|---|---|
| `pt-dr` smoke | cuda | 3 | 7.6 | 5.2 | 1.6 | 0.47 |
| `pt-dr-pix` smoke | cuda | 3 | 8.7 | 5.6 | 2.3 | 0.47 |
| `pt-dr` | cuda | 20 | 8.3 | 6.3 | 1.6 | 0.42 |
| `pt-dr` | cpu | 20 | 28.0 | 6.2 | 2.8 | 18.9 |

학습기는 GPU에서 45배 빠르다(18.9 → 0.42 초). 이제 iteration은 렌더가 좌우한다. 한 seed의
20 iteration return 곡선(recipe의 segment별 `return`):

- cuda: −11.12, −10.68, −10.41, −10.08, −10.73, −10.77, −10.36, −10.61, −10.58, −10.50, −10.86,
  −10.93, −10.39, −10.38, −10.42, −10.25, −10.66, −10.71, −10.60, −10.58 (평균 −10.58)
- cpu: −11.10, −10.50, −10.20, −10.00, −10.73, −10.78, −10.36, −10.62, −10.58, −10.51, −10.86,
  −10.93, −10.39, −10.39, −10.43, −10.26, −10.66, −10.71, −10.60, −10.58 (평균 −10.56)

둘은 비트 단위로 같지 않고 같다고 주장하지도 않는다(spec 3.5). iteration별 최대 차이는
0.21(iteration 3)이고 iteration 5부터는 0.015 안에서 일치한다. cuda smoke의 처음 세 return은
cuda 20 iteration 실행의 것과 같다. 어느 곡선도 20 iteration 안에 학습을 보이지 않으며, 기대한
것도 아니다. `-pix` smoke의 세 return(−8.91, −16.96, −18.86)은 세 iteration일 뿐 학습에 대해
아무것도 말하지 않는다.

**새 stage-2 추정** (위 속도에 대한 산술, `Target / Status: unverified`). iteration당 8.3 초인
`Pt` 16 spp 실행은 4,000 iteration에 ≈ 9.2 h다. `Rs` 행은 cuda에서 돌리지 않았다. X7이 잰 프레임당
0.52 ms(렌더 0.5 초)와 같은 rollout, 학습기로 보면 iteration당 ≈ 2.6 초, ≈ 2.9 h다. 12 실행
(4 행 × 3 seed)은 6 × 9.2 + 6 × 2.9 ≈ 평가 전 **73 GPU-시간**으로, CPU 학습기일 때의 ≈ 305 h와
대비된다. 네 `-pix` 행을 3 seed로 돌리면 같은 만큼이 더 들어, 24 실행 전체가 ≈ **145 h**다.
stage 1의 4 spp(프레임당 1.56 ms)라면 `Pt` 실행은 iteration당 ≈ 3.6 초, ≈ 4 h이고, 12 실행은
≈ 41 h, 24 실행은 ≈ 83 h가 된다.

**재실행: cuda 위의 네 `-pix` 행, `Pt`는 4 spp (소유자 결정 2026-09-24; 서버, 2026-09-24 10:02부터
2026-09-26 03:54 UTC까지, `~/artifacts/plan-x/x7b/`).** `cube_pose`가 든 행은 돌리지 않았고, 그
행들에 대해서는 위의 교란 요인 기록이 그대로 선다. 코드는 `0d9aedd`(`~/Projects/es-x7b`의
archive), `es`와 `es_native`는 `render`로 빌드, CPU 물리 백엔드, `--device cuda`, 인터프리터
`~/venvs/es-lerobot-cuda/bin/python`. `Pt` 행은 새 4 spp 파일로 학습한다
(`task-reach-vision-pt[-dr]-4spp[-svgf].toml`과 그 `-pix` observation·evaluation,
`training-reach-vision-pt[-dr]-4spp[-svgf]-pix.toml`). 16 spp 문서와 golden은 움직이지 않았다.
렌더 초는 `render_ms_per_frame` × 1,024, rollout은 `collect`에서 렌더를 뺀 값, 학습기는
`update`다. GPU lock은 모두 41.7 h 잡혔다(stage 1 1.8 h, stage 2 학습 39.4 h, 평가 0.5 h).
모든 stage 시작 시점에 외부 프로세스(`SSR_RENDER_GLTF`, ≈ 1 GB)가 GPU 위에 있었다
(`load.<stage>`). 아래 seed 사이의 시간 차이는 그것에도, 다른 어떤 것에도 귀속하지 않는다.

*Stage 1, SVGF 선택.* 돌기 전에 패킷에 적은 규칙: 마지막 100 iteration의 평균 return이 높은
실행이 이기되, 그 차이가 두 실행의 같은 100 iteration 표준편차 중 큰 값보다 작으면 SVGF
off(더 싼 쪽)가 이긴다. `pt-dr-pix`, 4 spp, seed 0, 1,000 iteration:

| SVGF | wall clock | 초 / it. | 렌더 | rollout | 학습기 | 처음 100 it. return | 마지막 100 it. return (std) | 엔트로피, 마지막 100 |
|---|---|---|---|---|---|---|---|---|
| off | 0.874 h | 3.15 | 1.30 | 1.43 | 0.415 | −18.79 | −18.86 (1.17) | 6.49 |
| on | 0.927 h | 3.34 | 1.50 | 1.42 | 0.415 | −20.01 | −20.06 (1.02) | 6.65 |

차이는 1.20으로 1.17보다 크므로 평균이 높은 쪽이 이긴다: **SVGF off**이고, 이것이 더 싼 쪽이기도
하다. 두 실행 모두 1,000 iteration 안에 배우지 않는다(처음과 마지막 100이 표준편차 안에서
일치). 9지표, off / on: `physics_steps_per_sec` 37,138 / 36,031; `actions_per_sec` 9,284 / 9,008;
`camera_frames_per_sec` 790 / 682; `pixels_per_sec` 7.28e6 / 6.28e6; `observation_gb_per_sec`,
`policy_inferences_per_sec`, `p50_end_to_end_latency`, `p95_end_to_end_latency`,
`gpu_memory_peak`, `chunk_underrun_rate`는 `null`(이 경로에서 계측하지 않음). 이전의 stage-1 행
(`cube_pose`가 있는 4 spp, CPU 학습기, iteration당 23.4 초)은 위 "측정, 그리고 예산 정지"의 표다.

*Stage 2, 학습.* 4 행 × seed 0, 1, 2, 각 4,000 iteration, recipe는 그대로, `Pt`는 4 spp에 SVGF
off. return은 `metrics/loss-curve.json`의 iteration별 `return`을 처음과 마지막 100 iteration에
걸쳐 평균한 값이고, 초는 iteration당이다.

| 행 | seed | wall clock | 초 / it. | 렌더 | rollout | 학습기 | 처음 100 return | 마지막 100 return (std) | 엔트로피, 마지막 100 |
|---|---|---|---|---|---|---|---|---|---|
| `pt-dr-pix` | 0 | 3.65 h | 3.29 | 1.34 | 1.53 | 0.415 | −18.79 | −18.66 (1.19) | 8.25 |
| `pt-dr-pix` | 1 | 4.11 h | 3.70 | 1.72 | 1.57 | 0.414 | −13.54 | −11.44 (1.46) | 12.85 |
| `pt-dr-pix` | 2 | 4.12 h | 3.71 | 1.73 | 1.57 | 0.414 | −9.15 | −8.75 (0.61) | 11.98 |
| `pt-pix` | 0 | 3.41 h | 3.07 | 1.10 | 1.55 | 0.415 | −18.81 | −18.66 (1.19) | 8.23 |
| `pt-pix` | 1 | 3.69 h | 3.32 | 1.36 | 1.55 | 0.415 | −13.95 | −11.53 (1.51) | 13.32 |
| `pt-pix` | 2 | 3.69 h | 3.32 | 1.38 | 1.53 | 0.414 | −9.09 | −7.53 (0.64) | 12.65 |
| `rs-dr-pix` | 0 | 2.76 h | 2.49 | 0.53 | 1.54 | 0.417 | −18.79 | −18.66 (1.19) | 8.28 |
| `rs-dr-pix` | 1 | 2.79 h | 2.51 | 0.56 | 1.53 | 0.417 | −18.40 | −15.92 (1.30) | 12.67 |
| `rs-dr-pix` | 2 | 2.78 h | 2.50 | 0.53 | 1.55 | 0.417 | −9.15 | −8.70 (0.63) | 12.14 |
| `rs-pix` | 0 | 2.77 h | 2.49 | 0.54 | 1.54 | 0.417 | −18.61 | −16.73 (1.07) | 12.56 |
| `rs-pix` | 1 | 2.80 h | 2.52 | 0.54 | 1.57 | 0.417 | −15.01 | −7.77 (0.65) | 10.77 |
| `rs-pix` | 2 | 2.80 h | 2.52 | 0.54 | 1.56 | 0.416 | −9.11 | −8.73 (0.69) | 11.93 |

return은 행보다 seed가 정한다: 네 행 모두에서 seed 0은 −18.8 근처, seed 2는 −9.1 근처에서
시작한다. 엔트로피는 모든 실행에서 올랐고(iteration 0에서 ≈ 5.5), `envelope_violation_rate`와
`executed_ne_sampled_rate`는 모든 실행의 모든 iteration에서 1.00이었다. 이전의 모든 reach 실행과
같다. seed 0의 `pt-dr-pix`와 `rs-dr-pix`(같은 물리 추출, 다른 렌더러)의 iteration별 return은
4,000 iteration에 걸쳐 최대 0.27, 평균 0.023 다르다. 실행별 9지표(`metrics/env-metrics.json`;
나머지 다섯은 stage 1처럼 `null`):

| 행 | seed | `physics_steps_per_sec` | `actions_per_sec` | `camera_frames_per_sec` | `pixels_per_sec` |
|---|---|---|---|---|---|
| `pt-dr-pix` | 0 / 1 / 2 | 37,548 / 32,275 / 32,753 | 9,387 / 8,069 / 8,188 | 765 / 595 / 593 | 7.05e6 / 5.48e6 / 5.46e6 |
| `pt-pix` | 0 / 1 / 2 | 39,144 / 32,704 / 36,751 | 9,786 / 8,176 / 9,188 | 928 / 755 / 744 | 8.55e6 / 6.96e6 / 6.85e6 |
| `rs-dr-pix` | 0 / 1 / 2 | 37,573 / 40,943 / 32,933 | 9,393 / 10,236 / 8,233 | 1,937 / 1,824 / 1,933 | 1.78e7 / 1.68e7 / 1.78e7 |
| `rs-pix` | 0 / 1 / 2 | 42,788 / 33,360 / 33,559 | 10,697 / 8,340 / 8,390 | 1,894 / 1,893 / 1,885 | 1.75e7 / 1.74e7 / 1.74e7 |

*평가.* 각 실행의 `checkpoints/4000.esb`를 자기 행의 평가 문서
(`evaluation-reach-vision-<row>.toml`, seed 201–216, `es eval run --jobs 4 --frames`, 각 153–179
초; 프레임은 각 보고서가 쓰인 뒤 지웠다)에서 평가했다. 스위트별 `success_rate`:

| 행 | seed | `nominal` | `light_intensity` | `light_direction` | `observation_delay` | `torque_noise` | `backlash` |
|---|---|---|---|---|---|---|---|
| `pt-dr-pix` | 0, 1, 2 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 |
| `pt-pix` | 0, 1, 2 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 |
| `rs-dr-pix` | 0, 1, 2 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 |
| `rs-pix` | 0, 1, 2 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 |

모든 held-out 에피소드가 200 step timeout까지 갔고 `envelope_violation_rate`는 1.00이었다. 예로
`pt-dr-pix` seed 1의 `nominal` 실패 히스토그램은 timeout 16, fallback 16, chunk underrun 16,
position 위반 3,184, velocity 720, acceleration 112다.

*Cross-render.* 열두 교차 평가(`Pt`로 학습한 것을 짝 `Rs` 문서에서, `Rs`로 학습한 것을 짝 `Pt`
문서에서) 모두 에피소드가 돌기 전에 exit 1로 거부되었다. 예로 `pt-dr-pix` seed 0을
`evaluation-reach-vision-rs-dr-pix.toml`에서:

```
error: tests/fixtures/rl/evaluation-reach-vision-rs-dr-pix.toml does not judge …/pt-dr-4spp-pix-s0/checkpoints/4000.esb:
ERROR XIR-040  evaluation references a different Task or Observation IR
  evaluation task reference is d3948e1b, the bundle hashes to eaa0d34b
ERROR XIR-040  evaluation references a different Task or Observation IR
  evaluation observation reference is 79495062, the bundle hashes to 1876c3f7
```

센서의 render 블록은 Task IR의 일부이므로, 해시 체인에게 `Pt` 행과 `Rs` 행은 서로 다른 두
task이자 두 observation이며, 평가 문서는 자신이 이름 붙인 task와 observation을 가진 번들만
판정한다(spec 10.4). 교차 렌더 평가에는 "같은 task를 다른 경로로 렌더한 것"을 평가 조건으로
적을 방법이 필요하다. 이것은 설계 질문이며, 이 패킷이 우회할 대상이 아니다.

*패킷의 질문.*

1. **아니다:** 4,000 iteration에서 `pt-dr-pix`의 세 seed 중 어느 것도(다른 세 행의 어느 seed도)
   여섯 스위트 어디에서도 held-out 성공을 한 번도 내지 못했고, 학습 return은 seed 사이의 차이
   이상으로 움직이지 않았다.
2. **4 spp에서 경로 추적기의 렌더는 iteration당 1.10–1.73 초로 `Rs`의 0.53–0.56 초 대비
   2.0–3.3배이고, 이로써 iteration은 2.49–2.52 초 대비 3.07–3.71 초, 4,000 iteration 실행은
   2.76–2.80 h 대비 3.41–4.12 h가 된다(stage 1에서 SVGF는 iteration당 렌더 0.2 초를 더한다).**
3. **답하지 못했다:** 배운 정책이 없어 비교할 전이가 없고, 렌더러가 task의 일부이므로 해시
   체인이 교차 렌더 평가를 XIR-040으로 거부한다.

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
절단면이다. `crates/es-import/src/rl_import.rs::learning_graph`가 그 일을 하는 곳이고 그렇게
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

### 8a. 어댑터 v2 — Isaac Lab 또는 Playground 정책의 입출력 규약 (packet M11/X2)

스펙 §28.14 규칙 3: 외부 정책의 입출력 규약은 IR이 소유한다. v2의 모든 필드는 선택이고
`deny_unknown_fields`는 그대로다. 그중 아무것도 선언하지 않은 어댑터는 이전과 정확히 같은
바이트로 변환된다(커밋된 v1 변환 네 개가 `crates/es-import/tests/adapter_v2.rs`에 해시로
고정되어 있다). 소스 규약마다 도착하는 곳은 한 군데다:

| 소스 규약 | 어댑터 v2 | 번들에서 도착하는 곳 |
|---|---|---|
| 관절을 아티큘레이션 순서에서 이름으로 해석 (Isaac `resolve_matching_names`) | `[joints] source_names` + `[joints.rename]` (`source_order`와 배타) | 관절별 채널마다 첫 Dense의 입력 열과 헤드의 행을 우리 액추에이터 순서로 치환 — 치환이므로 정확 |
| `default_joint_pos` | `[joints] default_pos`(소스 순서), 또는 매니페스트의 `default_joint_pos` | 그것을 읽는 곳에서만(아래) |
| `joint_pos_rel = q − default`, 항별 `scale` | `[[observation.channels]] offset = "default_pos"` 또는 벡터, `scale`(수 또는 벡터) | `Normalize{MeanStd}`에 접힌다: `mean = offset + mean_src / scale`, `std = std_src / scale` |
| `last_action` / `last_act` (원시 행동, 리셋 시 0) | Task IR 소스가 새 `ObsSource::PreviousAction { initial }`인 채널 | 루프가 이전 틱의 정책 행을 액추에이터 단위로 제공하고, 접기가 행동 꼬리의 역(`raw = (row − offset) / scale`)을 싣는다. Task IR은 `initial = offset`(우리 순서)을 선언해야 하며, 아니면 `IMP-005` |
| `history_length` | `history = N`, `history_order = "newest_last"`(Isaac의 평탄화) 또는 `"newest_first"` | N의 `TemporalWindowNode`(`Align::Hold`: 링이 찰 때까지 첫 프레임 반복 — 리셋 시 Isaac의 `CircularBuffer`). newest-first는 열 치환 |
| 항별 `clip`, 래퍼의 `clip_observations` | `clip = [lo, hi]` | **`IMP-009`**: 관측 IR에 clamp 노드가 없고 이 패킷은 추가하지 않는다. 정책이 만나는 상태에서 결코 걸리지 않는 clip은 선언하지 않는다(Isaac 오라클이 `max |obs| < 100`을 확인) |
| `JointPositionActionCfg`: `raw · scale + offset`, `use_default_offset` | `[action] scale`, `use_default_offset = true` (`offset = default_pos`) | 우리 순서의 `mean = offset`, `std = scale`인 `Normalizer{Inverse, MeanStd}` |
| `clip_actions` | `[action] clip = [lo, hi]` | 걸릴 수 없을 때만 수용 — `squash = tanh`이고 `[lo, hi] ⊇ [−1, 1]`. 아니면 `IMP-009` |
| `decimation × sim.dt`, Playground `ctrl_dt` | `[timing] policy_dt`, 또는 매니페스트의 `decimation` / `sim_dt` | 배치 IR의 제어 주기와 대조하고, 재샘플하지 않는다. 불일치는 **`IMP-006`**. 보고서의 `timing` 줄 |
| 액추에이터 `stiffness` / `damping` / `armature` / `effort_limit` | `[actuators]`(소스 순서) | 장면의 `kp`, `kv` + 관절 감쇠, armature, 힘 범위 옆의 `mapping-report.json` 행. 변환하지 않는다 |
| `projected_gravity`, `base_lin_vel`, `base_ang_vel`, `velocity_commands`, Task IR 채널이 없는 `generated_commands` | (채널의 `source` 항) | 항 이름을 댄 **`IMP-007`** |
| `source_order`와 `source_names`를 둘 다 / 둘 다 안 씀, 쓰이지 않는 rename, 치환이 아닌 해석 | — | **`IMP-008`** |

**`ObsSource::PreviousAction { initial }`**은 `ObsSource`의 마지막 변형이고 커밋된 어떤 Task IR에도
없으므로 어떤 `task_hash`도 움직이지 않았다(`crates/es-ir/tests/previous_action.rs`). 그 값은
정책이 이전 제어 틱에 낸 행이다 — 안전 평면 이전이며, `JointDelta`에서는 적분된 목표가 아니라
증분이다. `es_eval::runner`(평가와 수집, `capture_at`, 이전 틱의 `ChunkBuffer::action_at`에서;
언더런 틱은 행을 내지 않았으므로 마지막 행이 유지된다)와 `es_py::Rollout`(`act`에 마지막으로
건넨 행)에서 읽는다. `initial`(없으면 0)은 매 에피소드의 틱 0에 제공된다. bake는 기록된 행을
읽고 정책 출력을 갖고 있지 않으므로 이 채널을 이름으로 거절한다.

**`import_rl.py`.** rsl_rl의 `EmpiricalNormalization`은 버전이 둔 곳에서 읽는다 — ≥ 5.0
`actor_state_dict`의 `obs_normalizer.*`, 3.x `model_state_dict`의 `actor_obs_normalizer.*`
(비평가의 것은 결코 아님), 2.x 러너의 최상위 `obs_norm_state_dict` — `[1, D]`에서 평탄화하고,
`forward`가 합으로 나누므로 `obs_std = std + eps`(`eps = 1e-2`)다. `--isaac-env-cfg
params/env.yaml`은 `decimation`, `sim_dt`, 단일 행동 항의 `scale`과 `action_kind`, 그리고
`--joint-names`(아티큘레이션 순서. 없으면 정규식 표를 해석하지 않고 그렇다고 말한다)에 대해
해석한 `scene.robot.init_state.joint_pos`를 `default_joint_pos`로 기록한다.
`--playground-config`는 `ctrl_dt / sim_dt`를 `decimation`으로, `sim_dt`, `action_scale`, 덤프된
`default_pose`를 기록한다. 오라클의 생성기가 필요로 하는 피클 작성기(`save_native_rsl_rl`)도
여기에 산다(INV-16).

**측정(패킷 오라클 2, 이 워크스테이션, torch 2.14 CPU, 각 256 상태).**
`python/es/rl_source/isaac_reference.py`가 `isaac-lab.md` §§ 2–6과 `brax-ppo-so101.md` § 6에서
각 프레임워크의 관측 → 행동 사상을 NumPy로 계산하고, 임포트된 번들(`CpuPlan` 위 관측 IR,
`TorchRuntime` 위 학습 IR)과 액추에이터 단위로 비교한다:

| 소스 | 최대 절대 오차 |
|---|---|
| Isaac 스타일 rsl_rl, 고전 `model_state_dict` + `obs_norm_state_dict`, 관절 순서가 다르고 하나는 이름이 다름 | 1.216e-7 |
| 같은 액터의 ≥ 5.0 `actor_state_dict` 형태 | 1.216e-7 |
| Playground 스타일 brax (swish, tanh, `default_pose + 0.3·a`) | 7.605e-8 |
| 음성 대조: `joint_vel`의 `scale = 0.05`를 선언하지 않은 Isaac 소스 | 6.424e-1 |

이것이 정하지 않는 두 가지. rsl_rl 정규화기의 `eps = 1e-2`는 api-note에 없다(그 § 8의 열린
항목이 고정된 rsl_rl 버전이다). 리더와 레퍼런스가 둘 다 그것을 명시하므로, 고정 버전의 eps가
다르면 둘 다 움직인다. 그리고 실제 Isaac 체크포인트는 아직 임포트된 적이 없다 — 그것은 웨이브
3의 I3다.

## 9. 사람을 위한 열린 질문

1. 롤아웃이 Deployment IR의 선언된 지연시간을 모델링해야 하는가(트레이너 안의 청크 버퍼),
   아니면 정직함을 평가가 나르는 채로 동기적으로 남아야 하는가? 이 노트는 동기식을 고른다.
2. 플레인이 샘플링된 행동을 클램프할 때, 로그 확률은 *샘플*의 것이다; env는 *실행된* 행동을
   보았다. 그것들이 다른 비율은 보고된다. **트레이너 쪽은 패킷 M9/R5가 답했다**(2a절).
   `[rl] estimator = "executed"`가 실행된 행동으로 학습하고, 기본값은 `"sampled"`로 남으며,
   7절의 R5 행이 그것이 무엇을 측정했는지 말한다. 여전히 사람의 몫인 것은 같은 질문의 나머지
   절반이다 — 이 작업에서 *학습하는* 정책에게 엔벨로프가 무엇을 뜻해야 하는가
   (`docs/reviews/M9.md` S-7).
3. Regression 헤드 위의 `Squash::Tanh`: IR 파라미터인가(이 노트), 아니면 로워링이 적용하는 행동
   단위의 속성인가(`quadruped-track.md` 3.6 질문 2)? 이 노트는 파라미터를 고르며, 부재 = 기본값
   = 오늘의 해시다.
