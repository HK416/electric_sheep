# M7 T7 — the evaluator models inference latency the way the collector does

Spec: §8.6 (async inference and the chunk buffer: latency larger than the control period is the
normal case), §12.3 (determinism: `apply_at = computed_from + deterministic latency`, App. B.5),
§9.4 (`ChunkUnderrun` is a plane event), §10.4 (an evaluation is a reproducible function of the
document), §28.9 L22 / ladder rung 5, M5 review S-1 / R2, §28.10 (T7). Design note to extend:
`docs/design/evaluation-execution.md` (+ `.ko.md`) section 2 (the cell loop) and
`docs/design/visible-learning.md` open question 24 (answer it; write section 7.30 — check the
numbering against what has landed). Predecessors: V17 (both paths honour `rate.inference`
through `es_env::replan_interval`; the cadence-parity oracle), V6b (`es_env::plane_chunk` shared
by both paths).

## the question

`DomainRunner` (the collector) submits a chunk at tick `t` and releases it at `t +
latency_ticks(expected_latency_ms, rate.control)` through `AsyncInference`; tick 0 is a recorded
`ChunkUnderrun` and the first chunk lands at tick 1. `es_eval::runner` calls the policy and
executes row 0 in the same tick. The `qpos ‖ qvel` traces of one seed agree at tick 0 and diverge
at tick 1 (open question 24). **Can the evaluator use the collector's own latency model, so that
"the two paths execute chunks identically" is true of the trajectory and not only of the
schedule — and what do the demo's numbers become under it?**

## spec

* `es_eval::runner` drives its policy calls through `es_env::inference::AsyncInference` with
  `latency_ticks(contract.runtime.expected_latency_ms, rate.control)` — the **same constructor
  call `DomainRunner::new` makes** — and feeds released results into the `ChunkBuffer` before
  `plane_chunk`. No second latency model: if a helper must be extracted from `DomainRunner` so
  both call one function, extract it into `es-env` (that is in scope); do not copy it.
* Tick 0 of every episode therefore reaches the plane with an empty buffer, and the plane
  records `ChunkUnderrun` and acts on its fallback exactly as it does for the collector.
  `events.json` shows it; `envelope_violation_rate` and the underrun counters move; every
  evaluation number in the design record is re-dated by this packet (rule: marked, not deleted).
* `expected_latency_ms = 0` stays legal and means zero ticks — a document choice
  (`RuntimeHints`), stated in the note as "a claim no real robot can honour" (open question 24's
  own words).
* **The oracle that was missing**: extend V17's cadence-parity test into a *trajectory*-parity
  test — one seed, the scripted expert (or the demo's untrained bundle) through `es loop
  collect` and through `es_eval::Evaluation`, compare the per-tick `qpos ‖ qvel` from the two
  `.estraj` files bitwise to the last tick. Before this packet it fails at tick 1; after, it
  passes. Keep the failing number in the note.
* Re-measure on the server, no retraining: V19b's `w13-060000-a80.esb` on the committed
  documents, held-out nominal seeds 101–116 and the six-suite sweep (`--jobs 6`), beside V19b's
  table; and V18b's IR-graph checkpoint on nominal held-out. Report `success_rate`,
  `envelope_violation_rate`, underrun counts, fallback ticks. Whether `passed` still holds is
  the finding; the acceptance threshold is **not** touched either way.

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

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

`crates/es-eval/src/runner.rs`, `crates/es-env/src/{domains.rs,inference.rs,lib.rs}` (a shared
helper only; no behaviour change on the collector — its `.estraj` for a seed is bitwise
unchanged, and a test says so), `crates/es/tests/cli.rs` (the parity oracle; existing tests that
assert tick-0 execution are updated and named in the note), `crates/es-eval/tests/*.rs`,
`docs/design/evaluation-execution*.md`, `docs/design/visible-learning*.md` (open question 24, new
section), `docs/packets/M7/T7-eval-latency*.md`.

## oracle

1. `cargo test -p es-eval tick_zero_is_a_chunk_underrun_under_a_declared_latency` — with
   `expected_latency_ms` = one control period on a fake backend, the first `StepEvent` of every
   episode carries the `ChunkUnderrun` bit and `source != Policy`; with `0` it does not.
2. `cargo test -p es --test cli collection_and_evaluation_draw_the_same_trajectory -- --ignored`
   — the expert (or the untrained demo bundle) on one seed through both paths: the `.estraj`
   rows equal bitwise for every tick (needs mujoco; `SKIP` with reason without `ES_PYTHON`).
   Record the pre-fix first divergent tick and values in the note.
3. `cargo test -p es --test cli` — the existing expert-harness oracle
   (`expert_passes_the_evaluation_harness`, threshold 0.875) still passes with the latency model
   (the expert already runs under it in collection).
4. `cargo test -p es-env` — the collector's own tests unchanged; a new
   `collector_trajectory_is_unchanged_by_the_shared_helper` pins one seed's `.estraj` bytes
   before/after (generate the expectation from `main` first).
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T7-eval-latency.md`.

## acceptance

Oracles 1–5 (2–3 on the oracle server, `ES_PYTHON=~/venvs/es/bin/python`). The re-measurement
table on the server under `~/artifacts/plan-v/m7-t7/` and in the design note beside V19b's and
V18b's, with `passed` reported as found. §28.9 L22's row gets a one-line "fixed by T7" pointer
(that edit to `docs/ARCHITECTURE*.md` is the orchestrator's — list it in your report instead).

## forbidden

`crates/es-safety/**` (INV-12/13: the plane is neither widened nor bypassed — an underrun is its
own event); changing `expected_latency_ms` in any fixture; the acceptance threshold; a second
latency implementation; `docs/ARCHITECTURE*.md`; goldens; `crates/es-policy/**`. INV-17: no new
trait.
