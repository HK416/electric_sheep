# P-M2-R4 — bound the inference submission queue

Spec: spec 12.1, spec 12.2, spec 12.3, spec 20.3, spec 28.4. Design note:
`docs/design/batch-domains.md` §10. Review finding: `docs/reviews/M2.md` "Should-fix"
`crates/es-env/src/inference.rs:65`.

## context

```
crates/es-env/src/inference.rs       (the queue + its tests)
crates/es-env/src/lib.rs             (re-exports `default_max_pending`)
docs/design/batch-domains.md         (§10)
docs/packets/M2/P-M2-R4.md           (this file)
```

## spec

`AsyncInference::queue` was an unbounded `VecDeque`. Whenever `inference.batch` is under the
arrival rate — the normal over-subscribed case, which is why spec 12.2 has a camera
round-robin at all — it grew without limit. At the spec 28.4 gate configuration that is the
out-of-memory spec 20.3 says must not happen.

- `default_max_pending(latency_ticks, batch) = batch × latency_ticks + batch`: exactly the
  work in flight when the pipeline keeps up (one batch released per tick for the whole
  latency, plus the batch being filled). Saturating, and never below one batch.
- `AsyncInference::with_max_pending(n)` overrides it, clamped up to `batch`.
- `submit` drops a submission that would exceed the bound and counts it in
  `dropped_submissions()`. The **newest** is dropped, so the queue stays FIFO and
  back-pressure still delays work in schedule order rather than reshuffling it (spec 12.3).
- A dropped observation produces no chunk, so it surfaces downstream exactly where every other
  missing chunk does: an underrun, then the Safety Plane's fallback (spec 8.6, spec 9.4).
  `submitted() + dropped_submissions()` is every attempt.

## oracle

```
cargo test -p es-env inference
```

`a_full_queue_drops_the_newest_instead_of_growing`: `batch = 1`, `latency_ticks = 2` (so
`max_pending == 3`), 4,096 envs × 10,000 ticks = 40.96 M submissions. `pending()` stays at or
below `max_pending()` on every tick, the excess is counted rather than held, and the test
completes without allocating a backlog. The release-order tests lift the bound explicitly
(`with_max_pending(usize::MAX)`): they are about order, which drops would confound.

## acceptance

- `cargo fmt --check`, `cargo clippy -p es-env --all-targets -- -D warnings`,
  `cargo test -p es-env`, `cargo xtask layering`, `cargo xtask check-spec-refs`.
- No new trait (INV-17). Determinism unchanged: the drop rule is a pure function of the queue
  length, so a run still replays bitwise.

## forbidden

- Real threading behind `submit`/`poll` — a later packet; the release rule stays here.
- `crates/es-env/src/domains.rs`'s action phase (P-M2-R3) and the sizing model (P-M2-R5).
