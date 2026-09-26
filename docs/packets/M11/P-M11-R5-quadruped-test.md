# M11 R5 — the quadruped observation-gap test reaches the gap it pins

Spec: §1.4 (a test either runs or says why), §8.7 (the runtime checks a checkpoint against the
lowered graph). Found by M10 S-1 and re-read by `docs/reviews/M11.md` S-7:
`quadruped_eval_run_names_the_observation_gap` (`crates/es/tests/cli.rs`) fails under
`ES_PYTHON`, which stops `cargo xtask ci`'s fail-fast test stage. The test writes
`b"es-m6-b1-untrained-placeholder"` as the weights. The runtime refuses that checkpoint against
the lowered graph (8 tensors missing) before it reaches the observation gap the test exists to
pin. Type B.

## the question

**Does the test build a checkpoint whose tensors match the lowered Learning IR (names, shapes,
dtypes), so that `es eval run` gets past the checkpoint check and is refused by the observation
gap's own message, or completes?**

## spec

* The test's weights are a well-formed safetensors that holds every tensor the lowered graph
  asks for, at its lowered shape and dtype (zeros are fine). Build it from the lowering itself,
  through whatever function the runtime or `es-compile` already uses to list the graph's
  parameters. Do not hand-write shapes.
* The three arms stay: `3` = SKIPPED, `1` = refused by the observation gap's message, `0` = the
  gap closed. A `1` for any other reason, the checkpoint refusal included, still fails the test.
* If a helper is needed, it goes in the test file. No production code changes unless listing the
  lowered parameters is impossible without one. In that case, add the smallest `pub` function and
  say why in the report.

## context

```
crates/es/tests/cli.rs
crates/es/tests/common/**
docs/packets/M11/P-M11-R5-quadruped-test.md
docs/packets/M11/P-M11-R5-quadruped-test.ko.md
```

## oracle

1. With `ES_PYTHON` set to the project venv:
   `cargo test -p es --test cli quadruped_eval_run_names_the_observation_gap -- --nocapture`
   passes and prints `RAN ... (refused by name)` or `RAN ... (the run completed)`.
2. Without `ES_PYTHON`, it still passes (the SKIPPED arm or the same refusal).
3. `cargo xtask ci` runs its test stage end to end with `ES_PYTHON` set.
4. fmt, clippy `-D warnings`, `cargo xtask check-scope docs/packets/M11/P-M11-R5-quadruped-test.md`.

## acceptance

Oracles 1–4. The report quotes the refusal message the test now reaches.

## forbidden

Deleting the test, `#[ignore]`, or widening the `1` arm's message match to accept the checkpoint
refusal. Committed goldens and fixtures.
