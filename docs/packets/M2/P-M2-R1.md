# P-M2-R1 — reset the plan's temporal rings per episode

Spec: §7.5 (three-layer time model), §10.1 (the evaluation table), §10.4 (fairness and
determinism), §11.3 (`CpuPlan` execution).
Design notes: `docs/design/observation-lowering.md` §9.1,
`docs/design/evaluation-execution.md` §2.4.
Review finding: `docs/reviews/M2.md` — Blocker, `crates/es-eval/src/runner.rs:126`.

One `CpuPlan` is compiled for the whole evaluation run and its `TemporalWindow` rings
(`plan.rs`, mutated in `exec.rs`) were never cleared, so episode N's first frames saw episode
N−1's tail and cell 2's saw cell 1's. The §10.1 table then depended on the order the suites
were declared in — exactly what §10.4 exists to prevent.

## context

```
crates/es-compile/src/plan.rs
crates/es-compile/tests/observation_cpu.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
docs/design/observation-lowering.md
docs/design/evaluation-execution.md
docs/packets/M2/P-M2-R1.md
```

## spec

- **`es-compile`** — `CpuPlan::reset(&mut self)` puts every stateful buffer back to the state
  `compile` left it in: each `Ring`'s `data` refilled with zeros, `cursor` and `pushed` back
  to 0. The rings are the only state a plan carries today; anything stateful added to this
  path later is cleared here too. No new type, no new trait (INV-17), no signature change to
  `compile` or `run`.
- **`es-eval`** — `run_episode` calls `plan.reset()` immediately after `env.reset(None)`,
  next to the existing `safety.reset_latch()`. An episode is where an observation stream
  ends; the first episode of a cell therefore also starts clean.

## oracle

```
cargo test -p es-compile reset_returns_the_plan_to_a_freshly_compiled_one
cargo test -p es-eval reversing_the_suite_order_leaves_every_cell_unchanged
```

- `es-compile`: a `StateInput -> TemporalWindow(n = 2)` plan is run four times, `reset`, then
  run again on the same input; the result must equal a freshly compiled plan's first `run` on
  that input, and must *differ* from the pre-reset run (otherwise the fixture is not
  history-sensitive and the test proves nothing).
- `es-eval`: the same evaluation over a windowed observation, once with the suites declared
  in one order and once reversed; every (suite, metric) cell must be identical. Two
  perturbation-free suites, because a `Perturbation` draw is keyed by the suite's *position*
  (`EnvRng::new(seed, cell_index, episode, stream)`) and a perturbed suite is therefore
  order-dependent by construction — a separate question about §10.4's `suite_id`.

## acceptance

- Both oracle tests pass, and the es-eval one fails if `plan.reset()` is removed (verified by
  commenting it out: `envelope_violation_rate` moved 0.46 → 0.44 between the two orders).
- `cargo test -p es-compile -p es-eval` green; no golden file changed.
- `CpuPlan::compiler_hash` is unchanged: no kernel was added or renumbered.

## forbidden

- `crates/es-compile/src/budget.rs` (P-M2-R5 owns it), `crates/es-env`, `crates/es-policy`,
  `crates/es-telemetry`, `crates/es`.
- Clearing the `SafetyPlane`'s counters or envelope along with the rings: `reset_latch`
  already draws that line and INV-12 forbids widening it.
- A `reset` that reallocates or resizes: the arena and the ring lengths are compile-time
  decisions (§11.1 `Memory Plan`).
