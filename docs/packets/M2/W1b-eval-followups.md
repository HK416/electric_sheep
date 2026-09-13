# W1b — Evaluation execution follow-ups

Spec: §10 (Evaluation IR), §10.3 (metric definitions), §10.5 (artifacts), §9.3-§9.4
(envelope violation rate), §28.4 M2 W1, §28.7 gate 11.
Design note: `docs/design/evaluation-execution.md` sections 4 and 5 (now answered).
Packet: `docs/packets/M2/W1-evaluation-execution.md` (the packet this follows up on).
Invariants: INV-12, INV-13, INV-17.

Two reviewer questions from W1 are answered here: the missing `Unavailable` variants in
`es_ir::evaluation` (design note section 5), and the `clamped_steps` /
`fallback_activations` double-count ceiling on `envelope_violation_rate` (design note
section 4).

## context

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

## spec

- **`es_ir::evaluation`** — `MetricValue` gains `Unavailable { reason: String }`.
  `AcceptanceResult` changes from a bare struct to an enum (`Determined { criterion,
  observed, passed }` — the old struct's fields, unchanged — plus `Unavailable { metric,
  reason }`), `#[serde(untagged)]` so a `report.json` written before this packet still
  parses: its flat `{criterion, observed, passed}` object matches `Determined`. Neither
  `validate` nor `evaluation_hash` enumerates either type (both are report-only, not part
  of the hashed document), so neither needed a matching update.
- **`es-eval`** — `metrics::compute` returns `es_ir::evaluation::MetricValue` directly (the
  crate-local `Measured` wrapper existed only to carry the "not measured" case the IR
  could not; it is deleted). `runner::Evaluation::run` returns
  `(es_ir::evaluation::EvaluationReport, EvaluationLock)`; the `EvalReport` wrapper and its
  `Unmeasured` / `Verdict` / `Outcome` machinery are deleted. `record_cell` pushes exactly
  one `CellResult` per declared metric per suite, `Unavailable` for the ones no code path
  measures. `judge` pushes exactly one `AcceptanceResult` per (criterion, matching suite)
  pair, `Determined` when the metric resolved to a scalar, `Unavailable` otherwise
  (unmeasured, a histogram compared as a scalar, or the suite never declaring the metric).
- **`es-safety`** — `SafetyCounters` gains `dirty_steps: u64` and a crate-private
  `record_step(clamped: bool, fell_back: bool)` that increments `clamped_steps` and/or
  `fallback_activations` as before and `dirty_steps` by at most one regardless of how many
  of the two are true. `SafetyPlane::finish` — the single tail every `validate` return path
  goes through — is the one call site, so every existing counter increment moves there
  without changing when a step counts as clamped or as a fallback. `es-eval`'s
  `envelope_violation_rate` metric becomes `dirty_steps / steps` (no more
  `saturating_add(...).min(steps)` guard against a double-count that can no longer happen).
  `SafetyPlane::validate`'s signature does not change (INV-13); the hot path stays
  allocation-free (`record_step` is plain arithmetic on a `Copy` struct).
- **`es`** — `es eval compare`'s `scalar` and `value_repr` gain the `MetricValue::Unavailable`
  arm they need for exhaustiveness; the comparison table prints an unavailable cell as
  `unavailable (<reason>)` and excludes it from the numeric delta, same as a histogram.

## oracle

```
cargo fmt --check
cargo clippy -p es-ir -p es-eval -p es-safety -p es --all-targets --features es-ir/testing -- -D warnings
cargo test -p es-ir -p es-eval -p es-safety -p es --features es-ir/testing
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

## acceptance

- `MetricValue` and `AcceptanceResult` serde round-trip, including a value built from JSON
  written before this packet (a bare `{criterion, observed, passed}` `AcceptanceResult`
  deserializes as `Determined`);
- `Evaluation::run` returns `(EvaluationReport, EvaluationLock)`; no type named `EvalReport`
  remains in `es-eval`;
- every `es-eval` test that used to read `report.report.*`, `report.unmeasured` or
  `report.verdicts` now reads the equivalent field on `EvaluationReport` directly, and all
  of them stay green;
- a `SafetyCounters` unit test constructs a step that is both clamped and a fallback
  (`record_step(true, true)`) and asserts `dirty_steps == 1` while `clamped_steps == 1` and
  `fallback_activations == 1`;
- `assert_no_alloc`-guarded `SafetyPlane::validate` hot-path test still passes;
- `es eval compare` still compiles and runs against a report carrying an `Unavailable`
  cell.

## forbidden

- `crates/es-env`, `crates/es-policy`, `crates/es-compile`, `crates/es-data` — other
  packets own these.
- Changing `SafetyPlane::validate`'s signature, or any envelope/watchdog/fallback logic:
  this packet only changes how an already-decided outcome is counted, never which outcome
  is decided.
- New extension points (INV-17), new `es-ir` schema fields beyond the two named variants,
  and any `es_ir::evaluation` hash-path change (`evaluation_hash` does not cover report
  types).
- Golden files, and any change to `xtask`.
