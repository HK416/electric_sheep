# W1 — Evaluation IR execution (`es-eval`)

Spec: §10 (Evaluation IR), §10.3 (metric definitions), §10.4 (determinism and fairness),
§10.5 (artifacts), §12.4 (the nine performance metrics), §9.4 (envelope violation rate is a
first-class metric), §5.3 (`execution_hash`), §28.4 M2 W1, §28.7 gate 11.
Design note: `docs/design/evaluation-execution.md` (review class C).
Invariants: INV-15 (augmentation off during evaluation), INV-12, INV-17.

## context

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

## spec

- **`perturb`** — `PerturbationPlan::compile(&EvaluationIr, &SceneDesc, &ModelInfo)` resolves
  every suite's perturbations once. Realised: `action_delay`, `observation_delay`,
  `frame_drop`, `torque_noise`, `backlash`. Everything else is
  `EvalError::Unsupported { kind, reason }` naming the kind — never skipped, never
  approximated. `apply_at_reset` draws the per-episode knobs into `ResetOverrides`;
  `StepState::{drop_observation, apply_per_step}` runs the per-step processes (action-delay
  ring, dropout burst, deadband, actuator noise). Every draw is
  `EnvRng::new(seed, suite_id, episode_idx, stream)` = §10.4's
  `TaskRng(seed_base, suite_id, episode_idx, stream)`.
- **`metrics`** — `compute(&MetricSpec, &[Episode], &SafetyCounters, &EnvMetrics) -> Measured`
  for all 18 `MetricSpec` variants. Measured: `success_rate`, `episode_length`,
  `action_smoothness`, `envelope_violation_rate`, `chunk_underrun_rate`,
  `failure_mode_histogram`. `Unavailable(reason)`: `intervention_rate`, `collision_rate`,
  `domain_gap`, and each §12.4 field whose `EnvMetrics` slot is `None`. No fabricated `0.0`
  and no single `step/s`. `aggregate` / `mean` / `std` / `ci95` are local; the Welch
  comparison of two reports is `es eval compare`, above this crate.
- **`runner`** — `Evaluation::run::<B, F, NJ, H>` loops suite x episode: fresh `Env`,
  mandatory `SafetyPlane::from_ir(deploy)`, `CpuPlan` (§11.3), then
  `capture -> plan.run -> policy.infer -> plane.validate -> env.step` until done or the step
  budget. INV-15 is checked before anything runs and refuses rather than rewriting the graph.
  Produces `EvalReport` (the §10.5 `EvaluationReport` plus `unmeasured` and `verdicts`) and
  `EvaluationLock`; `write_artifacts` writes `report.json` and `evaluation.lock`.
  `execution_hash` comes from a `HashChain` assembled from the IRs, `CpuPlan::compiler_hash`,
  `PolicyRuntime::runtime_hash` and `RunConfig`.

Constraints: `BTreeMap` only, no new traits (INV-17), English only, <= ~1600 source lines.

### Deviations from the packet sketch, and why

1. `run` takes `new_backend: impl FnMut() -> B`, not one `B`. `Env::new` consumes its
   backend and each cell needs a fresh `Env` so its episode counter — which keys the task's
   own randomization — restarts at 0 and the table rows stay comparable (§10.4).
2. `run` returns `(EvalReport, EvaluationLock)`, not `EvaluationReport`.
   `es_ir::evaluation` has no `MetricValue::Unavailable` and no
   `AcceptanceResult::Unavailable`, and "not measured" cannot be written as an `f64` without
   inventing one. See the reviewer question in the design note section 5.
3. `apply_per_step` lives on `StepState`, not on `PerturbationPlan`: every one of these
   processes is stateful and a `&self` method has nowhere to keep the ring or the RNG cursor.

## oracle

```
cargo fmt -p es-eval --check
cargo clippy -p es-eval --all-targets -- -D warnings
cargo test -p es-eval
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

## acceptance

- one cell per suite x measured metric, `n_episodes` correct on each;
- two runs of the same document produce byte-identical `report.json`;
- a different `seed_base` changes at least one cell;
- a perturbed suite differs from `nominal` (a kernel that compiled but did nothing fails);
- a metric named by acceptance but not measured yields `Outcome::Unavailable`, the run does
  not pass, and no `AcceptanceResult` with a fabricated `observed` is written;
- an `Augment` node outside the allow-list refuses the run (INV-15); an allow-listed one gets
  past the check;
- an unsupported `PerturbationKind` refuses the run naming the kind;
- `envelope_violation_rate` is non-zero and higher when the policy commands out-of-envelope
  actions;
- `report.json` and `evaluation.lock` are written, and `report.episodes` stays empty.

## forbidden

- `crates/es-env`, `crates/es-policy`, `crates/es-compile`, `crates/es-data`, `crates/es` —
  other packets own these and are editing them concurrently.
- `crates/es-ir` — adding `MetricValue::Unavailable` is a separate packet.
- `report.html`, `episodes/` replay, `es eval compare`, batching the cell loop across the
  simulation domain (M2 W2), and any renderer-dependent perturbation kernel.
- Golden files, and any change to `xtask`.
