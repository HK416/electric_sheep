# M7 T7 — 평가기가 컬렉터와 같은 방식으로 추론 지연시간을 모델링한다

스펙: §8.6(비동기 추론과 청크 버퍼: 제어 주기보다 큰 지연시간이 정상적인 경우다),
§12.3(결정성: `apply_at = computed_from + deterministic latency`, 부록 B.5), §9.4
(`ChunkUnderrun`은 플레인 이벤트다), §10.4(평가는 문서의 재현 가능한 함수다), §28.9 L22 /
사다리 5단, M5 리뷰 S-1 / R2, §28.10(T7). 확장할 설계 노트: `docs/design/evaluation-execution.md`
(+ `.ko.md`) 2절(셀 루프)과 `docs/design/visible-learning.md`의 열린 질문 24(답할 것; 7.30절을
쓸 것 — 착륙한 것에 맞춰 번호를 확인할 것). 선행: V17(두 경로 모두
`es_env::replan_interval`을 통해 `rate.inference`를 지킨다; 케이던스-동등성 오라클),
V6b(`es_env::plane_chunk`는 두 경로가 공유한다).

## the question

`DomainRunner`(컬렉터)는 틱 `t`에 청크를 제출하고 `AsyncInference`를 통해
`t + latency_ticks(expected_latency_ms, rate.control)`에 그것을 방출한다; 틱 0은 기록된
`ChunkUnderrun`이고 첫 청크는 틱 1에 도착한다. `es_eval::runner`는 정책을 호출하고 같은
틱에 행 0을 실행한다. 하나의 시드에 대한 `qpos ‖ qvel` 궤적은 틱 0에서는 일치하고 틱
1에서 갈라진다(열린 질문 24). **평가기가 컬렉터 자신의 지연시간 모델을 쓸 수 있어서,
"두 경로가 청크를 동일하게 실행한다"는 것이 스케줄뿐 아니라 궤적에 대해서도 참이 될 수
있는가 — 그리고 그 아래에서 데모의 수치는 무엇이 되는가?**

## spec

* `es_eval::runner`는 `es_env::inference::AsyncInference`를 통해 정책 호출을 몰되,
  `latency_ticks(contract.runtime.expected_latency_ms, rate.control)` — **`DomainRunner::new`가
  만드는 것과 같은 생성자 호출** — 을 쓰고, `plane_chunk` 전에 방출된 결과를 `ChunkBuffer`에
  먹인다. 두 번째 지연시간 모델은 없다: 헬퍼 하나를 `DomainRunner`에서 뽑아내 두 쪽이 함수
  하나를 호출하게 해야 한다면, `es-env`로 뽑아낼 것(그것은 범위 안이다); 복사하지 말 것.
* 그러므로 모든 에피소드의 틱 0은 빈 버퍼로 플레인에 도달하고, 플레인은 컬렉터에 대해
  하는 것과 정확히 같이 `ChunkUnderrun`을 기록하고 그 폴백에 따라 행동한다. `events.json`이
  그것을 보여준다; `envelope_violation_rate`와 언더런 카운터가 움직인다; 설계 기록 안의
  모든 평가 수치가 이 패킷에 의해 다시 날짜가 매겨진다(규칙: 지워지지 않고 표시된다).
* `expected_latency_ms = 0`은 여전히 합법이며 틱 0개를 뜻한다 — 문서의 선택
  (`RuntimeHints`)이며, 노트에는 "실제 로봇이라면 지킬 수 없는 주장"(열린 질문 24 자신의
  표현)이라고 적는다.
* **빠져 있던 오라클**: V17의 케이던스-동등성 테스트를 *궤적*-동등성 테스트로 확장한다 —
  하나의 시드, 스크립트화된 전문가(또는 데모의 학습되지 않은 번들)를 `es loop collect`와
  `es_eval::Evaluation` 양쪽으로 통과시켜, 두 `.estraj` 파일의 틱별 `qpos ‖ qvel`을 마지막
  틱까지 비트 단위로 비교한다. 이 패킷 이전에는 틱 1에서 실패한다; 이후에는 통과한다.
  실패했던 수치를 노트에 남겨 둔다.
* 서버에서 재측정, 재학습 없이: V19b의 `w13-060000-a80.esb`를 커밋된 문서들 위에서, 배제된
  명목 시드 101–116과 여섯 스위트 스윕(`--jobs 6`)으로, V19b의 표 옆에; 그리고 V18b의
  IR-그래프 체크포인트를 명목상 배제된 것에 대해. `success_rate`, `envelope_violation_rate`,
  언더런 카운트, 폴백 틱을 보고한다. `passed`가 여전히 유지되는지는 발견 사항이다;
  수용 임계값은 어느 방향으로도 건드리지 않는다.

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
crates/es-eval/src/runner.rs
crates/es-eval/tests/*.rs
crates/es-env/src/domains.rs
crates/es-env/src/inference.rs
crates/es-env/src/lib.rs
crates/es-env/tests/*.rs
crates/es/tests/cli.rs
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/T7-eval-latency.md
docs/packets/M7/T7-eval-latency.ko.md
```

`crates/es-eval/src/runner.rs`, `crates/es-env/src/{domains.rs,inference.rs,lib.rs}`(공유
헬퍼뿐; 컬렉터의 동작 변경 없음 — 어떤 시드에 대한 그것의 `.estraj`는 비트 단위로
불변이고, 테스트가 그렇게 말한다), `crates/es/tests/cli.rs`(동등성 오라클; 틱-0 실행을
단언하는 기존 테스트들이 갱신되고 노트에 이름이 적힌다), `crates/es-eval/tests/*.rs`,
`docs/design/evaluation-execution*.md`, `docs/design/visible-learning*.md`(열린 질문 24, 새
절), `docs/packets/M7/T7-eval-latency*.md`.

## oracle

1. `cargo test -p es-eval tick_zero_is_a_chunk_underrun_under_a_declared_latency` — 가짜
   백엔드에서 `expected_latency_ms`가 제어 주기 하나일 때, 모든 에피소드의 첫 `StepEvent`가
   `ChunkUnderrun` 비트를 담고 `source != Policy`다; `0`일 때는 담지 않는다.
2. `cargo test -p es --test cli collection_and_evaluation_draw_the_same_trajectory -- --ignored`
   — 전문가(또는 학습되지 않은 데모 번들)를 하나의 시드로 두 경로 모두에 통과시킨다: 모든
   틱에서 `.estraj` 행이 비트 단위로 같다(mujoco 필요; `ES_PYTHON` 없이는 이유와 함께
   `SKIP`). 수정 전의 첫 발산 틱과 값을 노트에 기록한다.
3. `cargo test -p es --test cli` — 기존의 전문가-하네스 오라클
   (`expert_passes_the_evaluation_harness`, 임계값 0.875)이 지연시간 모델 아래에서도 여전히
   통과한다(전문가는 수집에서 이미 그 아래에서 돈다).
4. `cargo test -p es-env` — 컬렉터 자신의 테스트는 불변; 공유 헬퍼로 인해 한 시드의
   `.estraj` 바이트가 전후로 불변임을 고정하는 새 `collector_trajectory_is_unchanged_by_the_shared_helper`
   (기대값은 먼저 `main`에서 생성한다).
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T7-eval-latency.md`.

## acceptance

오라클 1–5(2–3은 오라클 서버에서, `ES_PYTHON=~/venvs/es/bin/python`). 서버의
`~/artifacts/plan-v/m7-t7/` 아래와 설계 노트 안의 재측정 표가 V19b와 V18b의 것 옆에 놓이며,
`passed`는 발견된 대로 보고된다. §28.9 L22의 행이 "T7로 고쳐짐"이라는 한 줄 포인터를
얻는다(`docs/ARCHITECTURE*.md`에 대한 그 수정은 오케스트레이터의 몫이다 — 그 대신 보고서에
그것을 나열할 것).

## forbidden

`crates/es-safety/**`(INV-12/13: 플레인은 넓혀지지도 우회되지도 않는다 — 언더런은 그
자신의 이벤트다); 어떤 픽스처에서든 `expected_latency_ms`를 바꾸는 것; 수용 임계값; 두
번째 지연시간 구현; `docs/ARCHITECTURE*.md`; 골든; `crates/es-policy/**`. INV-17: 새
트레이트 없음.
