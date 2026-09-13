<!-- Korean translation of docs/packets/M2/P-M2-R1.md. The English file is the working copy; regenerate this when it changes. -->

# P-M2-R1 — 에피소드마다 계획의 temporal ring을 리셋하기

Spec: §7.5 (3계층 시간 모델), §10.1 (평가 표), §10.4 (공정성과 결정성), §11.3
(`CpuPlan` 실행).
설계 노트: `docs/design/observation-lowering.md` §9.1,
`docs/design/evaluation-execution.md` §2.4.
리뷰 발견 사항: `docs/reviews/M2.md` — Blocker, `crates/es-eval/src/runner.rs:126`.

`CpuPlan` 하나가 평가 실행 전체에 대해 컴파일되며, 그 `TemporalWindow` ring들(`plan.rs`,
`exec.rs`에서 변형됨)은 결코 비워지지 않았다. 그래서 에피소드 N의 첫 프레임들은 에피소드
N−1의 꼬리를 보았고, 셀 2의 것은 셀 1의 것을 보았다. 그러면 §10.1 표는 스위트가 선언된
순서에 의존하게 되는데 — 이는 정확히 §10.4가 막기 위해 존재하는 것이다.

## context (범위)

```
crates/es-compile/src/plan.rs
crates/es-compile/tests/observation_cpu.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
docs/design/observation-lowering.md
docs/design/evaluation-execution.md
docs/packets/M2/P-M2-R1.md
```

## spec (사양)

- **`es-compile`** — `CpuPlan::reset(&mut self)`는 모든 상태를 지니는 버퍼를 `compile`이
  남겨둔 상태로 되돌린다: 각 `Ring`의 `data`는 0으로 다시 채워지고, `cursor`와 `pushed`는
  0으로 돌아간다. ring들은 오늘날 계획이 지니는 유일한 상태다; 나중에 이 경로에 추가되는
  상태를 지니는 것은 무엇이든 여기서도 지워진다. 새 타입 없음, 새 trait 없음(INV-17),
  `compile`이나 `run`의 시그니처 변경 없음.
- **`es-eval`** — `run_episode`는 `env.reset(None)` 직후, 이미 있는
  `safety.reset_latch()` 옆에서 `plan.reset()`을 호출한다. 에피소드는 observation
  스트림이 끝나는 지점이다; 따라서 셀의 첫 에피소드도 깨끗하게 시작한다.

## oracle (오라클)

```
cargo test -p es-compile reset_returns_the_plan_to_a_freshly_compiled_one
cargo test -p es-eval reversing_the_suite_order_leaves_every_cell_unchanged
```

- `es-compile`: `StateInput -> TemporalWindow(n = 2)` 계획을 네 번 실행하고, `reset`한
  다음, 같은 입력으로 다시 실행한다; 그 결과는 그 입력에 대한 새로 컴파일된 계획의 첫
  `run`과 같아야 하며, reset 이전 실행과는 *달라야* 한다(그렇지 않으면 픽스처가 히스토리에
  민감하지 않은 것이고 테스트는 아무것도 증명하지 못한다).
- `es-eval`: 윈도우가 있는 observation에 대한 동일한 평가를, 한 번은 스위트가 한 순서로
  선언된 채로, 한 번은 뒤집힌 채로 실행한다; 모든 (스위트, 지표) 셀이 동일해야 한다.
  perturbation이 없는 스위트 두 개를 쓰는데, `Perturbation` 추첨은 스위트의
  *위치*(`EnvRng::new(seed, cell_index, episode, stream)`)로 키가 매겨지므로
  perturbation이 있는 스위트는 구성상 순서에 의존하기 때문이다 — 이는 §10.4의
  `suite_id`에 관한 별개의 질문이다.

## acceptance (수용 기준)

- 두 오라클 테스트 모두 통과하며, es-eval 쪽은 `plan.reset()`을 제거하면 실패한다(주석
  처리해서 확인함: `envelope_violation_rate`가 두 순서 사이에서 0.46 → 0.44로
  이동했다).
- `cargo test -p es-compile -p es-eval`가 green이다; 바뀐 golden 파일 없음.
- `CpuPlan::compiler_hash`는 변하지 않는다: 커널이 추가되거나 번호가 다시 매겨지지
  않았다.

## forbidden (금지)

- `crates/es-compile/src/budget.rs`(P-M2-R5 소관), `crates/es-env`, `crates/es-policy`,
  `crates/es-telemetry`, `crates/es`.
- ring과 함께 `SafetyPlane`의 카운터나 envelope를 지우는 것: `reset_latch`가 이미 그
  경계를 그어 두었고 INV-12는 그것을 넓히는 것을 금지한다.
- 재할당하거나 크기를 바꾸는 `reset`: arena와 ring 길이는 컴파일 시점의 결정이다(§11.1
  `Memory Plan`).
