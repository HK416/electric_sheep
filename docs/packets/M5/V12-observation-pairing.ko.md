# M5 V12 — 정책이 보고 행동하는 관측이 곧 시연이 기록하는 관측

설계 노트: `docs/design/visible-learning.ko.md` **7.20절**, 그리고 7.19(V11의 수치, 비교 대상),
7.18(V10이 이 페어링을 측정하고 잡음으로 읽은 곳), 7.12–7.13(엔벌로프 의미론), 5절(스크립트
전문가). 스펙: §12.1, §13.2, §18.5. 의존: V1c, V2, V6, V6b, V8, V9, V10, V11.

## 이 패킷이 고치는 결함

`crates/es-env/src/env.rs`의 `Env::step`, 이 패킷 이전:

```
set_ctrl(ctrl) -> backend.step(substeps) -> let state = self.backend.state()
              -> StepRow { qpos/qvel/sensordata: <스텝 이후 상태>, ctrl: last_ctrl, ... }
              -> recorder.push
```

그래서 시연의 행 `t`는 `observation.state[t]` — `action[t]`가 **실행된 뒤의** 상태 — 를
`action[t]`와 짝지었다. `crates/es-data/src/collect.rs`는 같은 스텝-이후 상태에서 프레임을
렌더하고 `.estraj` 자세를 기록했다("스텝이 끝난 상태에서 — 위의 행이 기록한 바로 그 상태").

그러나 어떤 정책도 아직 도달하지 않은 상태에서 행동을 계산하지는 않는다.
`Env::step_with_policy`는 `observe_window -> infer_window -> emit_actions -> step` 순서이고,
`es_eval::runner`는 "관측 -> 계획 -> 추론 -> 검증 -> 스텝"이며 프레임은 추론 **전에** 잡힌다.
따라서 학습은 `(s_{t+1}, image_{t+1}) -> a_t`를 배우는데 추론은 `(s_t, image_t) -> a_t`를
묻는다 — **플랜 V가 수집한 모든 시연의 모든 입력에 제어 주기 하나만큼의 지연**이 있었다.
LeRobot의 관례는 반대다. `observation[t]`는 `action[t]`를 고른 근거이고, 행동은 그것을 본 뒤에
실행된다.

V10은 이것을 측정하고 잡음으로 읽었다. 당시에는 옳았다 — 200 Hz에서 제어 한 스텝은 5 ms이고
이동량은 0.7 mrad였다. 진짜 50 Hz(V11)에서 스텝당 증분은 **중앙값 0.04 rad**, 정책이 재현해야
하는 추종 오차 중앙값의 두 배다.

## 수정, 그리고 왜 env에 넣었는가

한 곳, `Env::step`. 모든 소비자가 상속한다. `backend.step` **이전에** `qpos/qvel/sensordata`와
틱을 스냅샷해 방금 적용한 `ctrl`과 함께 `StepRow`에 넣는다. 보상·종료·실패는 전이의 것으로
남는다 — 그것들은 스텝이 만들어낸 결과에 대한 진술이고, `done`은 여전히 에피소드의 마지막
프레임을 표시한다(LeRobot의 의미론). 스냅샷은 `reset_qpos`·`last_ctrl` 옆의 한 번만 할당되는
스크래치(`PreStep`)다.

수집기는 프레임 렌더와 `.estraj` 푸시를 같은 순간 — 스텝 이후가 아니라 이전 — 으로 옮긴다.
이미지·상태 행·궤적이 하나의 상태, 곧 추론에서 정책이 받게 될 그 상태가 된다. 덕분에 아무도
이름 붙이지 않았던 더 작은 결함도 사라진다. 종료 스텝에서 `Env::step`은 반환 전에 env를 자동
리셋하므로, 수집기의 스텝-이후 읽기는 모든 에피소드의 마지막 프레임에 **다음 에피소드의 리셋
상태**를 렌더하고 있었다.

건드리지 않은 것: Safety Plane과 `validate` 시그니처(`INV-12`, `INV-13`), IR 문서와 그 해시,
그리고 `dataset_schema_hash` — 스키마로서의 열과 의미는 그대로이고 내용만 움직인다.

`es_eval::runner`는 고칠 것이 없었다. 이미 추론 전에 프레임과 궤적 행을 잡고 있었다. 그것이
요점이다 — 평가기가 옳았고 기록기가 틀렸다.

## scope

* **`context`** — `crates/es-env/src/env.rs`, `crates/es-data/src/collect.rs`,
  `crates/es-data/tests/loop_learning.rs`, `crates/es/tests/cli.rs`,
  `docs/design/visible-learning*.md` 7.20절과 미해결 질문 19,
  `docs/packets/M5/V12-observation-pairing*.md`.
* **`forbidden`** — `SafetyPlane`과 `validate` 시그니처(`INV-12`, `INV-13`),
  `tests/fixtures/**`(문서도 한계도 게이트도 아님), 데이터셋 스키마, `docs/ARCHITECTURE*.md`.
* **`INV-17`** — 새 트레이트 없음. `PreStep`은 `Vec<f64>` 셋을 가진 비공개 구조체다.

## 오라클

1. **행은 그 행동이 계산된 상태다** — `cargo test -p es-env --lib
   the_recorded_row_is_the_state_the_action_was_computed_from`: 움직이는 장면에서 `step(ctrl)`
   한 번 뒤, 기록된 행의 `qpos`는 스텝 이전 상태와 같고 이후의 `backend.state()`와는 다르며,
   `ctrl`은 준 값 그대로다. `crates/es-data/tests/loop_learning.rs::
   frames_are_written_once_per_control_step`이 수집기 쪽 절반이다: 프레임 `i`와
   `observation.state[i]`는 비트까지 같은 순간이다.
2. **페어링 측정이 뒤집힌다** — 새로 수집한 세트에서 `max_j |action[t] - qpos[t+k]|`의 프레임
   중앙값, `k = -1, 0, +1`. 수정 전에는 `k = 0`이 최소(행이 이미 `s_{t+1}`이므로 그것이 추종
   오차), 수정 후에는 `k = +1`이 최소여야 하고 `k = 0`이 명령 선행이어야 한다.
   `~/artifacts/plan-v/v12/pairing.py`를 V11과 V12의 baked 세트 양쪽에 돌린다.
3. **그 밖에는 움직이지 않는다** — `recorded_actions_replay_to_the_same_outcome`가 새 세트에서
   모든 시연을 재현하고, V9의 `showcase_replay_of_a_real_run_is_bit_identical`이 궤적을 기록된
   프레임으로 비트까지 똑같이 재생한다.
4. **재수집·재학습·재측정** — V11의 명령 그대로, 그리고 held-out 101–116뿐 아니라 **학습
   시드**(1–16)에서도 평가. V11의 양쪽 0/16과 비교한다.
5. **중단 규칙** — 수정 후에도 IR 소유 그래프가 *자기 학습 시드에서* 0/16이면 그것은 폐루프
   결함이고, 다음 패킷은 데이터를 더 모으는 것이 아니라 정책의 예측과 전문가 행동을 틱 단위로
   비교하는 것이다.

## 측정

오라클 서버(RTX 4090), `es`는 `~/venvs/es`, 학습은 `~/venvs/es-lerobot-cuda`, 2026-09-16.
산출물은 `~/artifacts/plan-v/v12/`, V11의 것은 옆의 `v11/`.

### 1 — 두 단위 오라클

둘 다 통과, `cargo xtask ci` 그린, 골든도 픽스처 해시도 움직이지 않았다.

### 2 — 페어링이 정확히 제어 한 스텝만큼 뒤집힌다

`pairing.py`를 각 패킷의 50 에피소드 baked 세트에(양쪽 다 9,038 프레임 — 수정은 행이 *담는
것*을 바꾸지 행 수를 바꾸지 않는다):

| `max_j abs(action[t] − qpos[t+k])`의 프레임 중앙값 | V11(스텝 이후 행) | **V12(스텝 이전 행)** |
|---|---|---|
| `k = −1` | 0.06159 rad | 0.11149 rad |
| `k = 0` | **0.02094 rad** | 0.06159 rad |
| `k = +1` | 0.06753 rad | **0.02137 rad** |
| 최소 | `k = 0` | **`k = +1`** |

V12의 `k = 0`은 V11의 `k = −1`과 소수점 다섯 자리까지 같다 — 같은 측정이 정확히 한 행 밀린
것이고, 그것이 이 수정이 주장하는 바다. 명령 선행 0.062 rad, 추종 오차 0.021 rad.

### 3 — 시연은 그대로 재생된다

`ES_V10_DATASET=<새 세트> cargo test --release -p es --test cli
recorded_actions_replay_to_the_same_outcome`: **시연 50개가 큐브 50개를 상자에 넣었고, `action`
재생이 50개(50 `Success`), `action_commanded`가 50개를 재현**한다. V11과 같다. 수집 자체도
같다: 50/50 `Success`, 9,038 프레임, `dataset_schema_hash` `1f5ddafc…8777c`(**불변**),
`dataset_content_hash` `b13af58e…1a35` → `3bfaf41f…3d24`.

V9의 오라클 `showcase_replay_of_a_real_run_is_bit_identical`을 새 실행의 첫 nominal 셀에:
**900 프레임 동일**(오라클은 `<run>/frames/<cell>`을 읽으므로 실행의 `--frames` 디렉터리를
`nominal-20000-a/frames`로 링크해 두었다). `.estraj` 기록과 기록된 프레임이 함께 움직였다 — 이제 상태를 두 번이 아니라
한 번 읽어 둘 다 쓰기 때문이다.

### 4 — 두 정책, V11과 견주어

IR 소유 그래프, 새 50 에피소드 세트에 V2의 노브 그대로(`es policy lower` →
`train_act.py --batch 8 --lr 1e-4 --seed 0 --device cuda --resident-gpu`, 20,000 스텝, 초기
손실 0.0583 → 최종 **0.0141**, RTX 4090에서 654 s; V11은 0.0565에서 0.0132):

| IR 소유 그래프, 20,000 스텝 | V11 | **V12** |
|---|---|---|
| nominal, 시드 101–116 (실행 a) | 0 / 16 | **0 / 16** |
| nominal, 시드 101–116 (실행 b) | 0 / 16 | **0 / 16** |
| **학습 시드 1–16** | 0 / 16 | **0 / 16** |
| `envelope_violation_rate`, nominal | 0.2735 | **0.0371** |
| `envelope_violation_rate`, 학습 시드 | 0.3152 | **0.0365** |
| 실패 히스토그램, 학습 시드 | fallback 280, position 3402, acceleration 949, velocity 498, rate 280 | **position 6, acceleration 467, velocity 342, fallback 없음** |
| `episode_length` | 900 | 900 |

실행 a와 b는 마지막 자리까지 일치하므로 이 수치는 정책의 것이다. **엔벌로프 위반율이 8배,
position 위반이 560배 줄었다** — 정책이 내는 명령이 이제 그 명령을 낸 상태에서 도달 가능하다.
입력에서 한 스텝 지연을 없앴을 때 기대되는 바로 그 변화다. 그래도 큐브에는 도달하지 못한다.

**중단 규칙이 발동한다: 자기 학습 시드에서 0/16.** 시드 1은 학습 에피소드 0이다.

### 5 — 이유는 이제 측정되었다: 학습된 적합이 배포된 적합이 아니다

오케스트레이터의 개루프 분석(`~/artifacts/plan-v/v11/openloop/`)이 이 패킷보다 먼저 이유를
찾았고, 그것은 페어링이 **아니다**. `train_act.py --batch 8`은 **단일 샘플** 순전파 여덟 번을
누적해 한 옵티마이저 스텝을 만든다. 그래서 낮춰진 ResNet18의 모든 `BatchNorm2d`가 `N = 1`
통계로 학습되는데, 추론은 `model.eval()`(`crates/es-policy/python/torch_ref.py`)로 러닝 통계를
쓴다. V12 자신의 체크포인트와 baked 세트로 다시 돌린 결과(학습 에피소드 3개):

| 10행 청크 L1, 학습 에피소드 0 / 1 / 2 | V11 | **V12** |
|---|---|---|
| `train()` 모드 — 손실이 측정한 것 | 0.0107 / 0.0122 / 0.0110 | 0.0129 / 0.0127 / 0.0130 |
| `eval()` 모드 — **`es eval run`이 실행하는 경로** | 0.0313 / 0.0373 / 0.0385 | 0.0337 / 0.0438 / 0.0441 |
| 기준선: 모든 행을 현재 자세로 유지 | 0.0484 / 0.0488 / 0.0481 | 0.0565 / 0.0571 / 0.0561 |

배포된 적합은 학습된 적합의 **세 배**이고, 학습에 쓴 에피소드에서조차 "아무것도 하지 않기"보다
4분의 1만 낫다. 이 간극은 페어링 수정 앞뒤로 크기가 같고, 그래서 페어링이 결함의 전부가
아니었음을 말해준다 — 다만 개루프 표에는 더 이상 지연의 흔적이 없다. 청크의 0행이 `action[t]`에
가장 가깝고, 평가기가 실제로 실행하는 블렌드는 이제 *정렬된* 쪽이다. 모듈은 계약의 16이 아니라
**10**행을 낸다(`v5_actions = v4_chunk[:10]`).

### 6 — 외부 ACT

LeRobot 자신의 ACT는 위 발견의 대조군이다. 백본이 `FrozenBatchNorm2d`를 쓰고 실제 배치 8로
학습하므로 train/eval 간극이 없다. **새** 내보내기에 V8 파이프라인 그대로:

| LeRobot ACT, 100,000 스텝, nominal | V8 (200 Hz) | V11 (50 Hz) | **V12 (50 Hz, 페어링 수정)** |
|---|---|---|---|
| `success_rate` | 1 / 16 | 0 / 16 | **0 / 16** |
| `envelope_violation_rate` | -- | 0.0523 | **0.4267** |
| `episode_length` | 900 | 900 | 900 |

RTX 4090에서 `lerobot-train` 2,175 s. **여전히 0/16이고, 이 수치가 다음 패킷을 정직하게
만든다.** 외부 ACT에는 train/eval 정규화 간극이 없으므로 `BatchNorm`이 이야기의 전부일 수도
없다. 엔벌로프 위반율은 IR 소유 그래프와 *반대* 방향으로 움직였다(0.0523 -> 0.4267, fallback
110, acceleration 위반 5,981). 수정된 페어링으로 학습한 ACT는 플레인이 더 세게 클램프해야 하는
명령을 낸다 — 한 스텝 늦게 보여주던 상태를 더 이상 그렇게 보지 않게 된 정책의 흔적이다. 어느
쪽이든 두 정책 모두 큐브에 도달하지 못한다.

**판정.** 페어링은 실제 결함이었고 뿌리에서 고쳐졌다. `Env`의 모든 소비자가 올바른
`observation[t] -> action[t]`를 상속하고, 측정은 정확히 한 행만큼 뒤집히며, 파이프라인의 다른
것은 움직이지 않았다(재생 50/50, V9 비트 동일, 스키마 해시 불변). 그러나 충분하지는 **않았다**:
IR 소유 그래프는 held-out 0/16, 학습 시드 0/16, 외부 ACT도 0/16. 수정이 사준 것은 측정 가능하되
성공은 아니다 — IR 그래프의 엔벌로프 위반율 8분의 1, position 위반 560분의 1. 중단 규칙이
발동하고, 다음 변수는 추측이 아니라 이름과 측정치를 가진다: 미해결 질문 19.

## acceptance

1. **행은 그 행동이 계산된 상태다.** 충족 — 단위 오라클 둘 다 통과, `cargo xtask ci` 그린,
   골든 불변.
2. **페어링이 뒤집힌다.** 충족, 그것도 정확히 한 행만큼.
3. **그 밖에는 움직이지 않는다.** 충족 — 시연 50/50 재생, V9 재생 900 프레임 비트 동일.
4. **재수집·재학습·재측정.** 충족, 그리고 IR 소유 그래프에 대해서는 부정적이다: held-out
   0/16, **자기 학습 시드에서 0/16**, 엔벌로프 위반율은 8분의 1.
5. **중단 규칙.** 발동.

## 다음 패킷

중단 규칙이 이름을 대고 개루프 측정이 확인해준다. **IR 소유 그래프의 배포된 적합이 학습된
적합이 아니다. `train_act.py`가 `BatchNorm2d`에 한 번에 한 샘플씩 먹이기 때문이다.** 그것이
미해결 질문 19이고, 변수는 하나다: 배치 낮추기를 벡터화해 BatchNorm이 진짜 배치를 보게 하거나,
IR의 ResNet18이 frozen/group 정규화를 쓰게 하거나. 어느 쪽도 이 패킷의 수정을 건드리지 않는다.
다음 패킷은 그 수정을 물려받는다.
