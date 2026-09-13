<!-- Korean translation of docs/packets/M2/W1-evaluation-execution.md. The English file is the working copy; regenerate this when it changes. -->

# W1 — Evaluation IR 실행 (`es-eval`)

Spec: §10 (Evaluation IR), §10.3 (지표 정의), §10.4 (결정성과 공정성), §10.5 (산출물),
§12.4 (아홉 개 성능 지표), §9.4 (envelope violation rate는 1급 지표다), §5.3
(`execution_hash`), §28.4 M2 W1, §28.7 게이트 11.
설계 노트: `docs/design/evaluation-execution.md` (리뷰 등급 C).
불변식: INV-15 (평가 중에는 augmentation이 꺼짐), INV-12, INV-17.

## context (범위)

```
crates/es-eval/Cargo.toml
crates/es-eval/src/lib.rs
crates/es-eval/src/perturb.rs
crates/es-eval/src/metrics.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
docs/design/evaluation-execution.md
docs/packets/M2/W1-evaluation-execution.md
Cargo.toml                        # the `es-eval` workspace-dependency line only
```

## spec (사양)

- **`perturb`** — `PerturbationPlan::compile(&EvaluationIr, &SceneDesc, &ModelInfo)`은
  모든 스위트의 perturbation을 한 번 해석한다. 실현된 것: `action_delay`,
  `observation_delay`, `frame_drop`, `torque_noise`, `backlash`. 그 외 전부는 그 종류를
  지목하는 `EvalError::Unsupported { kind, reason }`다 — 절대 건너뛰지 않고, 절대
  근사하지 않는다. `apply_at_reset`은 에피소드별 손잡이(knob)를 `ResetOverrides`로
  추첨한다; `StepState::{drop_observation, apply_per_step}`은 스텝별 프로세스(action-delay
  ring, dropout burst, deadband, actuator noise)를 실행한다. 모든 추첨은
  §10.4의 `TaskRng(seed_base, suite_id, episode_idx, stream)`인
  `EnvRng::new(seed, suite_id, episode_idx, stream)`이다.
- **`metrics`** — 18개 `MetricSpec` variant 전부에 대해
  `compute(&MetricSpec, &[Episode], &SafetyCounters, &EnvMetrics) -> Measured`. 측정됨:
  `success_rate`, `episode_length`, `action_smoothness`, `envelope_violation_rate`,
  `chunk_underrun_rate`, `failure_mode_histogram`. `Unavailable(reason)`:
  `intervention_rate`, `collision_rate`, `domain_gap`, 그리고 `EnvMetrics` 슬롯이
  `None`인 각 §12.4 필드. 지어낸 `0.0`도 없고 단일 `step/s`도 없다. `aggregate` /
  `mean` / `std` / `ci95`는 로컬이다; 두 리포트 사이의 Welch 비교는 이 크레이트 위에
  있는 `es eval compare`다.
- **`runner`** — `Evaluation::run::<B, F, NJ, H>`은 suite x episode를 순회한다: 새
  `Env`, 필수인 `SafetyPlane::from_ir(deploy)`, `CpuPlan`(§11.3), 그다음 완료되거나 스텝
  예산에 도달할 때까지 `capture -> plan.run -> policy.infer -> plane.validate ->
  env.step`. INV-15는 무엇이든 실행되기 전에 검사되며, 그래프를 다시 쓰는 대신 거부한다.
  `EvalReport`(§10.5의 `EvaluationReport`에 `unmeasured`와 `verdicts`를 더한 것)와
  `EvaluationLock`을 만든다; `write_artifacts`는 `report.json`과 `evaluation.lock`을
  쓴다. `execution_hash`는 IR들, `CpuPlan::compiler_hash`, `PolicyRuntime::runtime_hash`,
  `RunConfig`로부터 조립된 `HashChain`에서 나온다.

제약: 오직 `BTreeMap`만, 새 trait 없음(INV-17), 영어만, 소스 라인 <= 약 1600줄.

### 패킷 스케치로부터의 이탈과 그 이유

1. `run`은 `B` 하나가 아니라 `new_backend: impl FnMut() -> B`를 받는다. `Env::new`는
   자신의 backend를 소비하며, 각 셀은 새 `Env`가 필요하다. 그래야 태스크 자신의
   randomization의 키가 되는 에피소드 카운터가 0에서 다시 시작하고 표의 행들이 비교
   가능하게 남는다(§10.4).
2. `run`은 `EvaluationReport`가 아니라 `(EvalReport, EvaluationLock)`을 반환한다.
   `es_ir::evaluation`에는 `MetricValue::Unavailable`도 `AcceptanceResult::Unavailable`도
   없고, "측정되지 않음"은 지어내지 않고서는 `f64`로 쓸 수 없다. 설계 노트 5절의
   리뷰어 질문을 참고.
3. `apply_per_step`은 `PerturbationPlan`이 아니라 `StepState`에 산다: 이 프로세스들은
   전부 상태를 가지며, `&self` 메서드는 ring이나 RNG 커서를 보관할 곳이 없다.

## oracle (오라클)

```
cargo fmt -p es-eval --check
cargo clippy -p es-eval --all-targets -- -D warnings
cargo test -p es-eval
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- 스위트 x 측정된 지표마다 셀 하나, 각각 올바른 `n_episodes`;
- 같은 문서를 두 번 실행하면 바이트 단위로 동일한 `report.json`이 나온다;
- 다른 `seed_base`는 최소 한 셀을 바꾼다;
- perturbation이 적용된 스위트는 `nominal`과 다르다(컴파일은 되었지만 아무것도 하지
  않는 커널은 실패한다);
- 수용 기준이 이름 붙였지만 측정되지 않은 지표는 `Outcome::Unavailable`을 내고, 실행은
  통과하지 않으며, 지어낸 `observed`를 가진 `AcceptanceResult`는 쓰이지 않는다;
- allow-list 밖의 `Augment` 노드는 실행을 거부시킨다(INV-15); allow-list에 있는
  것은 검사를 통과한다;
- 지원되지 않는 `PerturbationKind`는 그 종류를 지목하며 실행을 거부시킨다;
- 정책이 envelope 밖의 액션을 명령할 때 `envelope_violation_rate`는 0이 아니고 더
  높다;
- `report.json`과 `evaluation.lock`이 쓰이며, `report.episodes`는 비어 있는 채로
  남는다.

## forbidden (금지)

- `crates/es-env`, `crates/es-policy`, `crates/es-compile`, `crates/es-data`, `crates/es`
  — 다른 패킷들이 이들을 소유하고 있으며 동시에 편집 중이다.
- `crates/es-ir` — `MetricValue::Unavailable`을 추가하는 것은 별도의 패킷이다.
- `report.html`, `episodes/` 리플레이, `es eval compare`, 시뮬레이션 도메인 전체에
  걸친 셀 루프 배치(M2 W2), 렌더러에 의존하는 어떤 perturbation 커널도.
- Golden 파일, 그리고 `xtask`에 대한 어떤 변경도.
