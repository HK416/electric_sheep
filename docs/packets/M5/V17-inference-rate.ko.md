# M5 V17 — 평가기가 Deployment IR이 선언한 주기로 재계획한다

설계 노트: `docs/design/visible-learning.ko.md` **7.25절**, 그리고 7.24(V16, 릴리스는 관측의
함수가 아니다), 7.13(V6b, 평가가 수집과 같은 방식으로 청크를 실행한다), 7.19(V11, 한 제어
스텝은 한 제어 주기).
스펙: §9.2, §9.4, §8.5, §8.6, §12.1, 부록 B.5, §1.4. 의존: V6b, V11, V15, V16.

## 질문

`tests/fixtures/visible-learning/deployment.toml`은 `rate.control = 50 Hz`,
`rate.inference = 5 Hz`(제어 10틱마다 재계획)와 `action.execute_chunk = 10`을 선언하고, 낮춰진
청크는 `[10, 6]`이다. 두 루프 모두 두 번째 레이트를 읽지 않았다. `es_eval::runner`는 매 제어
틱마다 `infer_chunk`를 불렀고 `es_env::DomainRunner`는 매 제어 스텝마다 추론을 제출했다
(`BatchDomains::single_env_at`이 추론 도메인을 제어 주기마다 발화시킨다). 그래서 모든 청크의
1..9행은 한 번도 실행되지 않았다 — 템포럴 앙상블은 최근 여덟 청크의 0..7행을 평균했고,
`HardSwitch`였다면 최신 청크의 0행만 냈을 것이다.

V16은 그 대가를 홀드 지점에서 측정했다. 시연의 타깃이 상승하는 동안 정책 입력 전체가 약 6
제어 틱 동안 얼어 있고, 그래서 틱마다의 재계획은 얼어붙은 입력의 조건부 중앙값(+0.007 rad)을
계속 실행하며 집게는 큐브를 놓지 못한다 — 반면 **같은 청크**의 7·8·9행은 0.105, 0.150,
0.203 rad이고 8개 운반 홀드 전부에서 0.093 rad 정지각을 넘는다. 미해결 질문 22의 기본값은
(iii)이고 질문 23의 결정이 이 패킷이다. 선언된 재계획 주기를 두 경로 모두에서 지키고, 재학습
없이 기존 최고 체크포인트를 다시 측정한다.

## 산출물

1. **하나의 규칙, 두 경로.** `es_env::replan_interval(rate)`은 `rate.control /
   rate.inference`를 정수 제어 틱으로 주고, 나누어떨어지지 않는 레이트는 반올림하지 않고
   **이름을 붙여 거부**한다. `es_eval::runner`는 `step % replan == 0`에서만 추론하고 그
   사이는 V6b가 연결해 둔 `es_env::plane_chunk` / `ChunkBuffer` 경로로 청크의 다음 행들을
   실행한다. `DomainRunner::infer_window`도 같은 수로 제출을 막으므로 `es loop collect`와
   `es eval run`이 청크를 동일하게 실행한다(V6b의 계약). 버퍼는 특수 처리하지 않고 그대로
   둔다. 10행 청크를 5 Hz로 쓰면 살아 있는 청크가 정확히 하나이고 앙상블은 그것으로
   축퇴하는데, 이는 분기가 아니라 산술이다. 추론 지연 모델링은 손대지 않았다.
   `ExpertCfg::pace_to`도 두 경로에서 같은 수를 받는다.
2. **플레인의 주기가 쓰는 시계.** `SafetyPlane`은 틱 차이를 Deployment IR의 *제어* 주기로
   마이크로초로 바꾸는데, 두 루프 모두 **시뮬레이션** 틱인 `Env::tick`을 넘기고 있었다 —
   V11 이후 데모 씬에서는 제어 스텝당 4틱이다. 매 제어 틱마다 청크가 도착하는 동안에는
   드러날 수 없었다. `accept`가 마감을 재기 전에 `last_chunk_tick = now`를 찍어 간격이 항상
   0이었기 때문이다. `rate.inference`를 지키자 간격이 실제가 되었고, 첫 측정은
   `violation.inference_deadline`이 28,800 제어 틱 중 25,920틱에서 발화하고 폴백이 에피소드
   전체를 붙잡은 결과로 돌아왔다. `DomainRunner::emit_actions`는 자신의 `control_tick`을,
   `es_eval::runner`는 에피소드 스텝을 쓴다. Safety Plane 자체는 그대로다.
3. **문서가 스스로를 말한다.** 5 Hz 재계획을 선언한 배치는 청크 도착 간격이 최대 180 ms이고
   40 ms `inference_deadline`을 만족시킬 수 없다. 데모의 `inference_budget`과 그 워치독은
   240 ms가 된다 — 문서가 스스로 요구하는 200 ms에 추론 자체를 위해 이미 허용하던 40 ms를
   더한 값이다. `deadlines.observation_age`는 `DEP_021`이 `inference_budget`보다 작은 값을
   거부하기 때문에만 따라간다. 실제로 강제되는 신선도 한계(`stale_observation`, 80 ms), 모든
   안전 한계, 0.5 수용 임계값은 그대로다. `deployment_hash`가 움직이므로 V15의 가중치 파일을
   현재 문서로 만든 번들에 다시 담는다. 가중치는 바이트 단위로 같은 파일이다.
4. **재학습 없는 재측정.** V15의 40,000스텝 체크포인트를 1,800스텝 예산에서 학습 시드 1–16과
   홀드아웃 101–116(두 번)으로, 에피소드별 들어올림 / 운반 / **릴리스** / 하니스 성공을
   V15·V16의 행 옆에 둔다.

## 범위

* **`context`** — `crates/es-env/src/{domains,expert,env,lib}.rs`,
  `crates/es-eval/src/runner.rs`, `crates/es-data/src/collect.rs`, `crates/es/src/cmd/loop.rs`,
  `crates/es-eval/tests/evaluation.rs`, `crates/es/tests/cli.rs`,
  `tests/fixtures/visible-learning/deployment.toml`,
  `docs/design/visible-learning*.md` 7.25절과 미해결 질문 22–23,
  `docs/packets/M5/V17-inference-rate*.md`.
* **`forbidden`** — 모든 IR 스키마, `crates/es-safety`, Deployment IR의 안전 한계와 수용
  임계값(0.5, 불변), 재학습, 7.21–7.24절, `docs/ARCHITECTURE*.md`.
* **`INV-17`** — 새 트레잇도 새 확장점도 없다. 자유 함수 하나와 필드 하나.
* **`INV-12`** — 비활성화한 것은 없다. 모든 틱은 여전히 버퍼 → `plane_chunk` →
  `SafetyPlane::validate` → `ctrl`을 지나고, 선언된 레이트에서 만족될 수 없던 워치독은
  제거된 것이 아니라 그 레이트에 맞는 수치를 갖게 되었다.

## 오라클

1. **`the_runner_infers_once_per_declared_replan_period`**(`crates/es-eval/tests/evaluation.rs`,
   모든 CI 티어): 호출을 세는 목 `PolicyRuntime`으로 100 제어 스텝 평가를 돌리면 10:1
   비율에서 정확히 **10회**, 1:1에서 정확히 **100회** 추론한다.
2. **`an_inference_rate_that_does_not_divide_the_control_rate_is_refused`**(같은 파일):
   나누어떨어지지 않는 레이트는 `rate.control`, `rate.inference`와 소수 주기를 이름으로 말하는
   `EvalError::Env`다.
3. **`collection_and_evaluation_ask_the_policy_at_the_same_cadence`**(`crates/es/tests/cli.rs`,
   `mujoco` 필요): 같은 스크립트 전문가, 같은 시드, 120 제어 틱에서 정책이 **각 경로마다**
   정확히 `ceil(120 / replan)`번 호출된다. 아울러 두 경로의 틱별 `qpos ‖ qvel` 궤적이 원시
   `f64` 비트로 처음 갈라지는 틱을 출력하되 단언으로 덮지 않는다. `es loop collect`는
   `RuntimeHints::expected_latency_ms`(15 ms, 50 Hz에서 제어 1틱)를 `AsyncInference`로
   적용하고 `es_eval::runner`에는 지연 모델이 없어서, 수집의 첫 청크는 틱 1에, 평가의 첫
   청크는 틱 0에 도달한다.
4. **`expert_passes_the_evaluation_harness`**가 여덟 고정 시드에서 여전히 ≥ 0.875이고
   `envelope_violation_rate`를 보고한다. `expert_solves_the_pinned_seeds`,
   `collection_and_evaluation_draw_the_same_scene_for_a_seed`, V9의
   `a_showcase_replay_reproduces_the_frames_the_policy_saw`, V10의
   `recorded_actions_replay_to_the_same_outcome`도 그대로 통과한다.
5. **`cargo xtask ci`** 통과 — fmt, clippy `-D warnings`, 컨텍스트 예산, 레이어링, 스펙 참조,
   골든.

## 수용 기준

케이던스 오라클이 통과하고, 두 경로가 같은 주기로 정책을 호출하며, 전문가가 여전히 하니스를
통과하고, V15 체크포인트가 1,800 예산에서 두 스위트로 재측정되어 개선 여부와 무관하게
V15·V16 옆에 보고된다. 홀드아웃 `success_rate` ≥ 0.5이면 6스위트 스윕과 쇼케이스 영상으로
이어지고, 미만이면 정지 규칙이 발동해 에피소드별 표를 보고한다 — 릴리스는 일어나는데 성공이
여전히 실패하면 어느 술어 항이 실패하는지 정확히 말한다. `cargo xtask ci`가 통과한다.
