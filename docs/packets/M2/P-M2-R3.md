# P-M2-R3 — the inference deadline survives the domain runner

Spec: spec 8.5, spec 8.6, spec 9.4, spec 12.3, spec 13.1. Design note:
`docs/design/batch-domains.md` §9. Review finding: `docs/reviews/M2.md` "Should-fix"
`crates/es-env/src/domains.rs:271,275`.

## context

```
crates/es-env/src/domains.rs         (the action phase + its tests)
crates/es-env/src/chunk_buffer.rs    (adds `arrivals`, `action_at`)
docs/design/batch-domains.md         (§9, §11)
docs/packets/M2/P-M2-R3.md           (this file)
```

## spec

`DomainRunner::emit_actions` stamped `ActionChunk::with_seq(self.control_tick)` on a freshly
synthesized one-row chunk every control tick. `SafetyPlane::accept` treats any unseen `seq` as
a new chunk and moves `last_chunk_tick` to it, and `last_chunk_tick` is what the
`InferenceDeadline` watchdog measures — so through `DomainRunner` that watchdog could never
fire, and a dead policy showed up only as `ChunkUnderrun`.

`seq` now advances **once per policy invocation result** — the inference completion that
produced the chunk — as `es-runtime-embedded/src/runtime.rs` already does:

- `ChunkBuffer::arrivals()` counts pushes (one per result delivered to that env).
- When `arrivals` has moved and the buffer covers the current tick, `emit_actions` stamps a
  fresh `seq` and fills the chunk with the rows that result will drive until the next one,
  read through the new `ChunkBuffer::action_at` (the counter-free `next_action`). The
  lookahead is exact: nothing reaches the buffer without moving `arrivals`, which is what
  triggers the next rebuild, so the plane's cursor walks exactly the rows per-tick blending
  would have produced.
- Every other tick resubmits the previous `seq`. `accept` copies nothing for a `seq` it holds,
  so the payload is never read, the plane keeps consuming its chunk, and `last_chunk_tick`
  stays where the last real result put it. Past the chunk's rows the plane's own
  `ChunkUnderrun` produces the fallback.
- `reset_env` bumps `seq` with no rows behind it, so the plane drops the finished episode's
  chunk instead of consuming it (spec 13.1).

`next_action` keeps being called once per env per control tick, so §12.4's
`chunk_underrun_rate` is unchanged.

## oracle

```
cargo test -p es-env domains::tests::a_policy_that_stops_producing_trips_the_inference_deadline
```

12 control steps with a live policy, then 8 with none: `InferenceDeadline` is counted 0 times
while the policy runs and fires on the 4th silent step — a 50 ms budget at 16 ms of plane time
per control step, so "within the budget", not late.

## acceptance

- `cargo fmt --check`, `cargo clippy -p es-env --all-targets -- -D warnings`,
  `cargo test -p es-env`, `cargo xtask layering`, `cargo xtask check-spec-refs`.
- No new trait (INV-17) and no heap in the action phase: the chunk is an inline
  `[[f64; NJ]; H]`, as before.
- The plane is still the only actuator path and is never disabled (INV-12, INV-13).

## forbidden

- `crates/es-safety/**` — `SafetyPlane::validate`'s signature and `accept`'s rule are fixed
  (INV-13); this packet adapts the caller to them.
- `crates/es-runtime-embedded/**` — it already gets this right; it is the reference here.
- P-M2-R4's queue bound and P-M2-R5's sizing model.
