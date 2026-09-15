# M5 V11 — 제어 한 스텝이 제어 주기 하나

설계 노트: `docs/design/visible-learning.ko.md` **7.19절**, 그리고 7.18(V10의 발견과 이 패킷이
답하는 미해결 질문 18), 7.16(V8의 수치, 비교 상대), 7.12–7.13(엔벌로프 의미론), 5절(스크립트
전문가). 스펙: §9.2, §12.1, §12.4, §13.2, 부록 B.5. V1c, V2, V6, V6b, V8, V9, V10에 의존한다.

## 이 패킷이 고치는 발견

V10이 측정했다(7.18절 측정 4). **데모 전체가 200 Hz로 돌았고 모든 문서는 50이라고 말했다.**
`es_env::Env::new`는 장면을 `LoadConfig { rate: None }`으로 적재하므로 물리는 MJCF의
`timestep="0.005"`를 유지하고, `Env::step`은 `schedule.domains().inference.period`만큼
시뮬레이션 틱을 전진시키는데, `Collector::run`과 `es_eval::runner`가 만들던 유일한 스케줄인
`BatchDomains::single_env()`가 그것을 1로 둔다. 그래서 기록된 액션 한 행이 5 ms 물리 스텝 하나였고,
그동안 `tests/fixtures/visible-learning/deployment.toml`은 `rate.control = 50`과
`rate.inference = 5`를 선언했으며 `SafetyPlane`의 `dt_s`는 Deployment IR에서 온다. 모든 동적
엔벌로프 한계가 스텝당 의도보다 네 배(가속도는 열여섯 배) 헐거웠고, 16행 청크는 320 ms가 아니라
80 ms를 덮었으며, 내보낸 데이터셋의 `fps = 50`은 5 ms 간격의 행을 가리켰다.

## 결정과 유도

미해결 질문 18의 기본값 **(b)** 채택: 물리는 두고 제어를 데시메이션한다.

```
BatchDomains::single_env_at(physics, control)     crates/es-env/src/scheduler.rs

    period = physics / control        정확한 유리수: (pn*cd) / (pd*cn)
    observation.period = inference.period = period
    simulation.period  = 1            그것이 틱을 정의한다(§12.1)
```

`LoadConfig::rate`는 `None`으로 남는다 — 장면 자신의 타임스텝은 건드리지 않는다. 시연의 접촉
거동과 M6 Go1 트랙이 모두 거기에 의존하기 때문이다. 데모에서는 `physics = 200 Hz`,
`control = 50 Hz`, `period = 4`다. `Env::step` 한 번이 `ctrl`을 한 번 설정하고 물리 틱 네 개를
전진시키며 한 행을 기록한다. 타임스텝이 제어 주기를 나누지 못하는 장면은 반올림되지 않고
**이름을 들어 거부**된다(부록 B.5의 검증 방식).

`observation.period`가 그것을 함께 쓰는 것은 의도적이다. `DomainRunner::observe_window`는
윈도우 *앞에서* 상태를 한 번 읽고 그 시뮬레이션 틱들을 순회하므로, `observation.period = 1`이면
같은 판독이 `TemporalWindow`에 네 번 들어가고 같은 프레임이 네 번 렌더링된다. 정책이 실제로
소비하는 것은 제어 스텝당 관측 하나다.

`rate.inference = 5`는 여전히 "제어 10스텝마다 재계획"을 뜻한다. 그것은 `ChunkBuffer`를 지나는
`action.execute_chunk`이고, V6b가 확립했으며 이 패킷은 건드리지 않는다.

**연결된 곳:** `Collector::run`(`es loop collect`), `es_eval::runner`(`es eval run`, 따라서
`expert_passes_the_evaluation_harness`), 그리고 `Env`를 직접 구동하는 `crates/es/tests/cli.rs`의
V10 테스트 두 개 — 공유 헬퍼 `demo_domains`를 통해. `es video showcase`의 `.estraj` 기록은
**제어 틱 단위**이고 그대로 유지된다. `Collector::run`과 `es_eval::runner` 모두
`step_with_policy` 반환 뒤에 제어 스텝당 한 행을 넣기 때문이다. 이제 그것은 진짜 50 Hz 기록이고,
`python/es/encode_video.py --fps 50`은 0.25배가 아니라 실시간으로 재생한다. 데이터셋의 `fps`는
이미 `rate.control`이었고 이제 참이다.

`TickRate::from_period_secs`는 타임스텝→틱 변환을 `es-core`로 옮긴다. `f64` 시간을 금지하는
스케줄러(§18.1)가 닿을 수 있는 곳이다. MuJoCo 백엔드의 `rate_from_timestep`은 사본을 두지 않고
그것에 위임한다.

## 속도 수정이 드러낸 두 번째 것

진짜 50 Hz에서 **스크립트 전문가가 동작을 멈췄다. `expert_solves_the_pinned_seeds` 0/8, 매
에피소드 타임아웃.** 추론이 아니라 측정이다(오라클 서버의 `python` + `mujoco` 3.13, 기록된 행을
툴 사이트에 대해 재생):

* 팔은 호버 자세에는 정확히 도달한다(툴이 래치된 큐브에서 xy로 0.0003 m, z 0.065).
* 하강에서 **파지 자세를 약 11 mm 지나친다** — 목표가 0.0146인데 툴 z가 0.0033까지 간다 —
  집게를 테이블에 박고, 제어 틱 하나에 큐브를 59 mm 밀어낸다.
* 그 뒤 팔은 테이블에 얹혀 있고 `shoulder_lift`는 유지된 명령보다 0.018 rad 아래에 앉으며,
  웨이포인트 기계의 `pos_tol = 0.01` 게이트는 다시는 닫히지 않는다. 접촉 없이 같은 `ctrl`을
  유지하면 순수 MuJoCo는 그 관절을 4.9e-4 rad로 안정시키므로, 0.018은 처짐이 아니라 테이블이다.

원인은 장면이 낼 수 없는 한계다. `forcerange` 2.94 N·m에 대해 `velocity_max = 3.0` rad/s와
`acceleration_max = 20` rad/s²는 하강 바닥에서 팔에 제동 여력을 남기지 않는다. 지금까지 그것이
드러날 수 없었던 이유는, 제어 한 스텝이 5 ms 물리 스텝 하나여서 팔이 에피소드 내내 속도 포화
상태였기 때문이다 — 램프를 추종할 수 없었으니 지나칠 수도 없었다.

`ExpertCfg::pace_to`는 전문가를 엔벌로프에 맞추려고 이미 있었다. 그 계수가 0.9에서
`PACE = 0.5`가 되었다. 진짜 속도에서 고정 시드 여덟 개로 스윕: **0.9 → 0/8, 0.75 → 2/8,
0.6 → 8/8, 0.5 → 8/8, 0.35 → 8/8**. 양쪽에 여유를 둔 절반. 엔벌로프 한계도, 수용 게이트도,
Deployment IR 값도 움직이지 않았다 — 이것은 장면 자신의 액추에이터에 대한 *전문가*의 보정이며,
이 패킷이 스케줄 외에 돌린 유일한 손잡이다.

## 범위

* **`context`** — `crates/es-core/src/time.rs`, `crates/es-env/src/scheduler.rs`,
  `crates/es-env/src/expert.rs`, `crates/es-data/src/collect.rs`, `crates/es-eval/src/runner.rs`,
  `crates/es-physics-backend/src/mujoco.rs`, `crates/es/tests/cli.rs`,
  `docs/design/visible-learning*.md` 7.19절과 미해결 질문 18,
  `docs/packets/M5/V11-control-period*.md`.
* **`forbidden`** — `LoadConfig::rate`(물리 타임스텝은 장면의 것으로 둔다),
  `tests/fixtures/**`(한계도, 게이트도, 문서도), `SafetyPlane`과 그 `validate` 시그니처
  (`INV-12`, `INV-13`), `docs/ARCHITECTURE*.md`.
* **`INV-17`** — 새 트레이트 없음. `single_env_at`은 기존 순수 데이터 타입의 생성자다.

## 오라클

1. **스케줄 유도** — `cargo test -p es-env --lib
   scheduler::tests::a_control_period_is_a_whole_number_of_substeps`. 0.005 s 장면에 50 Hz 제어는
   주기 4(관측도 마찬가지), 0.006 s는 두 속도를 이름으로 들며 거부, 0.02 s는 주기 1, 물리보다
   빠른 제어 속도는 같은 거부.
2. **V10의 측정이 뒤집힌다** — `ES_V10_DATASET=<new> ES_V10_DUMP=<dir> cargo test --release
   -p es --test cli recorded_actions_replay_to_the_same_outcome` 다음
   `$ES_PYTHON python/es/grasp_probe.py --dump <dir> --scene <scene> --substeps {4,1}`. 옛 V1c
   집합은 200 Hz 데이터라 50 Hz로 재생할 **수 없으므로** 데이터셋은 새로 수집했다.
3. **전문가는 여전히 하니스를 통과한다** — `cargo test --release -p es --test cli expert_ --
   --nocapture`, 측정된 `envelope_violation_rate`를 V6의 0.48–0.55 옆에 둔다.
4. **재수집·재학습·재측정** — V1c의 수집 명령, V2의 학습 손잡이, V8의 외부 ACT 파이프라인,
   V8의 0/16 · 0/16 · 1/16과 V6의 0/16에 견준다.
5. **중단 규칙** — 두 정책 모두 `success_rate 0.5` 아래에 머무르면 중단하고 보고한다. 이
   패킷에서 두 번째 변수를 움직이지 않는다.

## 측정

오라클 서버(RTX 4090, `es`는 `~/venvs/es`, 학습은 `~/venvs/es-lerobot-cuda`), mujoco 3.13,
2026-09-15. 산출물은 `~/artifacts/plan-v/v11/`.

### 1 — 스케줄

통과. 또한 `cargo xtask ci` 녹색이며 **골든도 픽스처 해시도 움직이지 않았다.** 스케줄은 정규
인코딩이 아니라 런타임 상태이므로 `regenerate_visible_learning_documents`도
`regenerate_quadruped_documents`도 재생성할 것이 없었다.

### 2 — 프로브가 뒤집힌다

| `grasp_probe.py` | `--substeps 4` | `--substeps 1` |
|---|---|---|
| `es` 재생 대비 최악 큐브 편차 | **0.0000 mm** | **195.0051 mm** |
| 큐브를 테이블에서 띄운 시연 | **50 / 50** | 0 / 50 |
| 들어올림, 중앙값 / 최대 | 120.75 / 130.77 mm | 3.85 / 12.10 mm |
| 양 집게 접촉 틱, 중앙값 | 90.0 | 0 |
| 양 집게가 큐브를 잡고 있을 때 그리퍼 관절 | 0.0934 rad | — |

정확히 V10의 표에서 열이 바뀐 것이다. 새 50 에피소드 집합에 대한
`recorded_actions_replay_to_the_same_outcome`: 기록된 큐브 50개가 통 안, `action` 재생이
50 재현(`Success` 50), `action_commanded` 재생이 50 재현.

### 3 — 하니스, 그리고 올바른 단위의 엔벌로프

| | V6 / V10 (200 Hz) | V11 (50 Hz) |
|---|---|---|
| `expert_passes_the_evaluation_harness` | 8 / 8 | **8 / 8** |
| 최악 `envelope_violation_rate` | 0.48 – 0.55 | **0.2044** (0.1535 – 0.2044) |
| `expert_solves_the_pinned_seeds` | 8 / 8 | **8 / 8** |
| `the_temporal_ensemble_survives_the_grasp_window` | `Success` | **`Success`** |
| 에피소드 길이, 제어 스텝 | ~351 | 225 – 235 |

위반율이 절반 넘게 떨어졌고, 그것이 "엔벌로프가 자신이 제한하는 스텝에 대해 측정된다"가
예측하는 바다.

### 4 — 이제 제어 한 스텝의 값어치

두 열 모두 같은 50 에피소드 명령(`--episodes 50 --seed 1`)에 같은 스크립트로 측정했으므로,
V10의 관절별 수치가 아니라 서로 견줄 수 있다.

| 시연 집합당 | V1c (200 Hz) | V11 (50 Hz) |
|---|---|---|
| 프레임 | 18,263 | 9,038 |
| 에피소드 중앙값, 제어 스텝 | 352 | 179 |
| 에피소드 중앙값, 시뮬레이션 초 | 1.76 | **3.58** |
| `max_j abs(action[t] − action[t−1])`, 중앙값 | 0.01687 rad | **0.03985 rad** |
| `max_j abs(action[t] − qpos[t−1])`, 중앙값 | 0.26506 rad | **0.06223 rad** |
| 16행 청크 이동, 중앙값 | 0.18931 rad | **0.47143 rad** |
| `meta/info.json` `fps` / 실제 행 간격 | 50 / 5 ms | **50 / 20 ms** |

명령이 더 이상 팔을 1/4 라디안 앞서지 않는다. 청크는 80 ms와 1/5 라디안이 아니라 320 ms와
0.5 라디안의 이동이다.

### 5 — 두 정책

**두 정책: 어느 쪽도 과제를 배우지 못했고, 중단 규칙이 발동한다.**

IR 소유 그래프, 새 50 Hz 집합에 V2의 정확한 손잡이(`es policy lower` →
`train_act.py --batch 8 --lr 1e-4 --seed 0 --device cuda --resident-gpu`, 최적화 20,000 스텝,
초기 손실 0.0565 → 최종 **0.0132**, RTX 4090에서 654 s. V1c의 200 Hz 실행은 0.0669에서
0.0180까지 갔다):

| 스위트 | `success_rate` | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|
| nominal (단독, 실행 a) | **0 / 16** | 0.2735 | 900 |
| nominal (단독, 실행 b) | **0 / 16** | 0.2735 | 900 |
| nominal (스위트 안) | 0 / 16 | 0.2706 | 900 |
| light_intensity | 0 / 16 | 0.2922 | 900 |
| light_direction | 0 / 16 | 0.2998 | 900 |
| observation_delay | 0 / 16 | 0.2940 | 900 |
| torque_noise | 0 / 16 | 0.5132 | 900 |
| backlash | 0 / 16 | 0.2003 | 900 |

두 nominal 실행이 마지막 자리까지 일치하므로 이 수치는 스케줄이 아니라 정책의 것이다. 모든
에피소드가 900 스텝 예산을 다 쓴다. 처음 여섯 nominal 셀의 `.estraj` 기록을 읽으면, 팔은 가장 크게
움직이는 관절에서 **1.66 – 1.69 rad** 움직이는데 큐브는 여섯 개 모두 **정확히 시작 자세**로 끝나고
그리퍼는 활짝 열린 채다. 정책은 파지를 놓치는 것이 아니라 큐브에 도달하지 못한다.

외부 ACT, 새 내보내기에 V8의 파이프라인을 그대로(`es dataset export --lerobot-v3
--drop action_commanded,action_source --state-dim 6` → `lerobot-train --policy.type=act
--steps=100000 --batch_size=8 --seed=0`, 2,033 s → `es policy import-lerobot` →
`es eval run`): **nominal 0 / 16**, `envelope_violation_rate` 0.0523, 모든 에피소드 900 스텝.
외부 ACT의 6개 스위트 스윗은 쓰지 **않았다**. 중단 규칙은 nominal 수치에서 발동하고, V8의
0/16 · 0/16 · 1/16도 그 수치이며, 섭동 없이도 0점인 정책의 섭동 스윗은 아무것도 측정하지 않는다.

| | V6 (200 Hz) | V8 (200 Hz) | **V11 (50 Hz)** |
|---|---|---|---|
| IR 소유 그래프, 20,000 스텝, nominal | 0 / 16 | — | **0 / 16** |
| LeRobot ACT, 20,000 / 50,000 / 100,000 스텝, nominal | — | 0/16 · 0/16 · 1/16 | 100,000에서 **0 / 16** |

**벽시계**(처리량 주장이 아니라 관측 — §12.4): 프레임 포함 50 에피소드 수집 50 s, 베이크 1 s,
lower + 20,000 스텝 학습 654 s, 프레임 포함 6개 스위트 평가(96 셀, 렌더 프레임 86,400개,
`--jobs 6`) 393 s, `lerobot-train` 100,000 스텝 2,033 s, 16셀 nominal 실행 하나는 경합에 따라
10 – 25분.

**판정.** 속도는 틀렸었고 이제 맞다. 제어 한 스텝이 제어 주기 하나이고, 엔벌로프는 자신이
제한하는 스텝에 대해 측정되며(전문가 위반율 0.48–0.55 → 0.2044), 청크는 실제 운동 320 ms이고,
데이터셋의 `fps`는 참이다. 속도 수정은 다른 무엇도 찾지 못한 것을 찾았다. **플랜 V가 수집한 모든
시연은 구동된 것이 아니라 끌려다닌 팔이 만든 것이다.** Deployment IR이 장면의 2.94 N·m
액추에이터로 제동할 수 없는 가속도를 선언하고 있고, 200 Hz에서는 팔이 속도 포화 상태라 추종할 수
없는 것을 지나칠 수도 없었기 때문이다. 그것을 고치자 — 전문가가 이제 엔벌로프의 절반에 맞춘다 —
시연 50/50, 전문가 게이트 두 개 모두 8/8, 위반율은 V6의 절반 이하로 돌아왔다.

**그리고 어느 정책도 여전히 배우지 못한다. 0/16과 0/16, V6의 0/16과 V8의 1/16에 대해.** 중단
규칙이 발동한다. 수정은 학습 문제를 오히려 *거칠게* 만들었는데 — 스텝당 17이 아니라 40 mrad,
청크당 1/5가 아니라 0.5 라디안 — 결과는 움직이지 않았다. 잘못된 것이 무엇이든 제어 속도도,
시연도(50/50, 재생과 프로브로 확인), 파지도(122 mm 들어올림, 양 집게, 에피소드의 절반), 앙상블도,
모델도(V8 자신의 ACT, 100,000 스텝) 아니다. **여기서 두 번째 변수를 움직이지 말 것.** 다음 패킷은
가설 하나와 변수 하나 — 해상도 또는 시연 개수 — 를 받고, 비교 기준은 V11의 수치다.


## 수용

1. **스케줄 유도.** 충족 — 단위 테스트 통과, 거부 메시지가 두 속도를 이름으로 든다.
2. **프로브가 뒤집힌다.** 충족 — 새로 수집한 집합에서 서브스텝 4에 0.0000 mm, 1에 195.0051 mm,
   들어올림 50/50, 재생 50/50.
3. **전문가가 하니스를 통과한다.** 8/8로 충족, `envelope_violation_rate`는 V6의 0.48–0.55에서
   0.2044로 하락 — 그러나 **두 번째 발견 뒤에야**. 선언된 한계에서는 팔이 파지 자세를 11 mm
   지나쳐 큐브를 쳐내므로 전문가의 페이싱을 엔벌로프의 0.9에서 0.5로 재보정해야 했다.
   Deployment IR 값도 게이트도 움직이지 않았다.
4. **재수집·재학습·재측정.** 충족, 그리고 답은 부정적이다. IR 소유 그래프는 모든 스위트에서
   **0/16**, 외부 ACT는 100,000 스텝에서 **0/16**. V6의 0/16과 V8의 1/16에 대해.
5. **중단 규칙.** 발동. 이 패킷에서 다른 것은 움직이지 않았다.
6. `cargo xtask ci` 녹색. 골든도 픽스처 해시도 움직이지 않았다.

## 다음 패킷

가설 하나, 변수 하나, 비교 기준은 V11. 96x96이 너무 작거나 시연 50개가 너무 적거나. V8이
의도적으로 둘 다 고정했고 V11도 다시 고정했다. 잘못될 수 있는 다섯 가지 중 넷은 이제 측정되어
닫혔다.
