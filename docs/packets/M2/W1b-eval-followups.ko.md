<!-- Korean translation of docs/packets/M2/W1b-eval-followups.md. The English file is the working copy; regenerate this when it changes. -->

# W1b — Evaluation 실행 후속 작업 (follow-ups)

Spec: §10 (Evaluation IR), §10.3 (지표 정의), §10.5 (산출물), §9.3-§9.4
(envelope violation rate), §28.4 M2 W1, §28.7 게이트 11.
설계 노트: `docs/design/evaluation-execution.md` 4절과 5절 (이제 답변됨).
패킷: `docs/packets/M2/W1-evaluation-execution.md` (이 패킷이 후속하는 패킷).
불변식: INV-12, INV-13, INV-17.

W1의 리뷰어 질문 두 가지가 여기서 답변된다: `es_ir::evaluation`에 빠져 있던
`Unavailable` variant(설계 노트 5절), 그리고 `envelope_violation_rate`에 대한
`clamped_steps` / `fallback_activations` 이중 계산 상한(설계 노트 4절).

## context (범위)

```
crates/es-ir/src/evaluation.rs
crates/es-eval/src/lib.rs
crates/es-eval/src/metrics.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
crates/es-safety/src/counters.rs
crates/es-safety/src/plane.rs
crates/es/src/cmd/eval.rs
docs/design/evaluation-execution.md
docs/packets/M2/W1b-eval-followups.md
```

## spec (사양)

- **`es_ir::evaluation`** — `MetricValue`에 `Unavailable { reason: String }`가 추가된다.
  `AcceptanceResult`는 단순 구조체에서 enum으로 바뀐다(`Determined { criterion,
  observed, passed }` — 기존 구조체의 필드 그대로 — 그리고 `Unavailable { metric,
  reason }`), `#[serde(untagged)]`이므로 이 패킷 이전에 쓰인 `report.json`도 여전히
  파싱된다: 그 평평한 `{criterion, observed, passed}` 객체는 `Determined`와 일치한다.
  `validate`도 `evaluation_hash`도 두 타입 중 어느 것도 열거하지 않으므로(둘 다
  리포트 전용이며 해시되는 문서에 속하지 않는다), 어느 쪽도 상응하는 업데이트가
  필요하지 않았다.
- **`es-eval`** — `metrics::compute`는 `es_ir::evaluation::MetricValue`를 직접
  반환한다(크레이트 로컬 `Measured` 래퍼는 IR이 표현할 수 없던 "측정되지 않음" 경우를
  나르기 위해서만 존재했다; 이제 삭제된다). `runner::Evaluation::run`은
  `(es_ir::evaluation::EvaluationReport, EvaluationLock)`을 반환한다; `EvalReport` 래퍼와
  그 `Unmeasured` / `Verdict` / `Outcome` 기계 장치는 삭제된다. `record_cell`은 선언된
  지표마다, 스위트마다 정확히 하나의 `CellResult`를 밀어 넣으며, 어떤 코드 경로도
  측정하지 않는 것에는 `Unavailable`을 넣는다. `judge`는 (criterion, 일치하는 suite)
  쌍마다 정확히 하나의 `AcceptanceResult`를 밀어 넣으며, 지표가 스칼라로 귀결되면
  `Determined`, 그렇지 않으면(측정되지 않음, 히스토그램이 스칼라로 비교됨, 또는
  스위트가 그 지표를 아예 선언하지 않음) `Unavailable`이다.
- **`es-safety`** — `SafetyCounters`에 `dirty_steps: u64`와 크레이트 전용
  `record_step(clamped: bool, fell_back: bool)`가 추가된다; 이전처럼 `clamped_steps`
  와/또는 `fallback_activations`를 증가시키고, 둘 중 몇 개가 참이든 상관없이
  `dirty_steps`는 최대 하나만 증가시킨다. `SafetyPlane::finish` — 모든 `validate` 반환
  경로가 거쳐 가는 그 하나의 꼬리 — 가 유일한 호출 지점이므로, 기존의 모든 카운터
  증가는 스텝이 clamped로 또는 fallback으로 계산되는 시점을 바꾸지 않은 채 그곳으로
  옮겨간다. `es-eval`의 `envelope_violation_rate` 지표는 `dirty_steps / steps`가 된다
  (더 이상 발생할 수 없는 이중 계산을 막던 `saturating_add(...).min(steps)` 가드는
  없어진다). `SafetyPlane::validate`의 시그니처는 바뀌지 않는다(INV-13); 핫 패스는
  할당이 없는 채로 남는다(`record_step`은 `Copy` 구조체에 대한 단순 산술이다).
- **`es`** — `es eval compare`의 `scalar`와 `value_repr`은 완전성(exhaustiveness)을
  위해 필요한 `MetricValue::Unavailable` arm을 얻는다; 비교 테이블은 unavailable
  셀을 `unavailable (<reason>)`으로 출력하고 히스토그램과 마찬가지로 수치 델타에서
  제외한다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-ir -p es-eval -p es-safety -p es --all-targets --features es-ir/testing -- -D warnings
cargo test -p es-ir -p es-eval -p es-safety -p es --features es-ir/testing
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `MetricValue`와 `AcceptanceResult`는 serde 왕복이 되며, 이 패킷 이전에 쓰인 JSON으로
  만들어진 값도 포함한다(평평한 `{criterion, observed, passed}` `AcceptanceResult`는
  `Determined`로 역직렬화된다);
- `Evaluation::run`은 `(EvaluationReport, EvaluationLock)`을 반환한다; `es-eval`에
  `EvalReport`라는 이름의 타입은 더 이상 남아 있지 않다;
- `report.report.*`, `report.unmeasured` 또는 `report.verdicts`를 읽던 모든 `es-eval`
  테스트는 이제 `EvaluationReport`의 동등한 필드를 직접 읽으며, 전부 그린 상태를
  유지한다;
- `SafetyCounters` 단위 테스트는 clamped이면서 동시에 fallback인 스텝을 구성하고
  (`record_step(true, true)`) `clamped_steps == 1`이고 `fallback_activations == 1`인
  채로 `dirty_steps == 1`임을 단언한다;
- `assert_no_alloc`로 지켜지는 `SafetyPlane::validate` 핫 패스 테스트는 여전히
  통과한다;
- `es eval compare`는 `Unavailable` 셀을 담은 리포트에 대해서도 여전히 컴파일되고
  실행된다.

## forbidden (금지)

- `crates/es-env`, `crates/es-policy`, `crates/es-compile`, `crates/es-data` — 다른
  패킷들이 이들을 소유한다.
- `SafetyPlane::validate`의 시그니처를 바꾸는 것, 또는 어떤 envelope/watchdog/fallback
  로직이든 바꾸는 것: 이 패킷은 이미 결정된 결과가 어떻게 계산되는지만 바꿀 뿐,
  어떤 결과가 결정되는지는 결코 바꾸지 않는다.
- 새 확장 지점(INV-17), 이름 붙은 두 variant를 넘어서는 새 `es-ir` 스키마 필드,
  그리고 어떤 `es_ir::evaluation` 해시 경로 변경이든(`evaluation_hash`는 리포트
  타입을 포함하지 않는다).
- Golden 파일, 그리고 `xtask`에 대한 어떤 변경도.
