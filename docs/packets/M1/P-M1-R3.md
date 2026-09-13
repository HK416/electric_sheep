# P-M1-R3 — chunk freshness from the caller

Fixes an M1 review Should-fix (`docs/reviews/M1.md`, Should-fix, `crates/es-safety/src/plane.rs:288-302`):
`SafetyPlane::accept` judged chunk freshness by content equality over the valid prefix, so a
policy that re-emits a bit-identical chunk on a genuine replan (a stationary hold, a saturated
output) never advanced `last_chunk_tick`, and `InferenceDeadline` could trip on a healthy
policy. Freshness now comes from the caller: `ActionChunk::seq`, a monotonic per-invocation
counter.

Spec: spec 8.5, spec 8.6, spec 9.4, `INV-13` (unchanged: `validate` still returns `SafeAction`,
no `Result`).

## context

```
crates/es-safety/src/types.rs                              (ActionChunk gains seq: u64)
crates/es-safety/src/plane.rs                               (accept() compares seq, not bytes)
crates/es-safety/tests/scenarios.rs                          (fixture Chunk.seq, auto-increment)
crates/es-runtime-embedded/src/runtime.rs                    (increments seq once per infer)
crates/es-runtime-embedded/tests/embedded.rs                 (comment fix only)
crates/es-env/src/domains.rs                                 (feeds control_tick as seq at the
                                                               two direct SafetyPlane::validate
                                                               call sites — es-env already
                                                               constructed ActionChunk before
                                                               this packet)
tests/fixtures/safety/identical_chunks_fresh_seq.json        (new)
tests/fixtures/safety/repeated_seq_is_stale.json             (new)
docs/design/safety-plane.md                                  (chunk-identity section rewritten;
                                                               fixture table +2)
docs/packets/M1/P-M1-R3.md                                   (new)
```

## forbidden

`crates/es-compile`, `crates/es-telemetry`, `crates/es-eval`, `crates/es-data`, `.github/**` —
owned by concurrent M1 follow-up work. `es-eval/src/runner.rs` also constructs an `ActionChunk`
but is out of scope; it is unaffected because `ActionChunk::new`/`::empty` keep their existing
signatures and default `seq` to `0` ("unknown"), so nothing outside this packet's scope needed
to change to keep compiling. No new trait (`SafetyPlane` extension points are frozen at
`INV-17`); `SafetyPlane::validate`'s signature is unchanged.

## spec

- `ActionChunk<NJ, H>` gains `pub seq: u64`. `new`/`empty` default it to `0` ("unknown"); a
  caller that cares about freshness chains `.with_seq(seq)`.
- `SafetyPlane::accept` treats a chunk as new iff `last_seq.is_none() || chunk.seq > last_seq`
  — the very first chunk a plane ever sees is always accepted, whatever its `seq`. Content is
  never inspected. `last_seq` is stored in `SafetyState` (`Option<u64>`, `None` initially).
- `es-runtime-embedded::EmbeddedRuntime` adds a `seq: u64` field, incremented once per
  `infer()` call (a replan tick) and attached via `.with_seq(self.seq)` to the chunk it hands
  the plane; ticks that reuse the buffered chunk resubmit the same `ActionChunk` (same `seq`)
  unchanged, exactly as before.
- `es-env::DomainRunner::emit_actions` synthesizes one single-row `ActionChunk` per control
  tick from the chunk buffer; it now tags each with `self.control_tick` (strictly increasing
  per call), replacing its former reliance on content differing tick to tick.

## oracle

```
cargo fmt --check
cargo clippy -p es-safety -p es-runtime-embedded -p es-env --all-targets -- -D warnings
cargo test -p es-safety -p es-runtime-embedded
cargo check --workspace --all-targets
```

## acceptance

- `tests/fixtures/safety/identical_chunks_fresh_seq.json`: three replans submit the same
  content with a strictly increasing `seq` each time; every step asserts `source: policy`.
- `tests/fixtures/safety/repeated_seq_is_stale.json`: a chunk is resubmitted with the *same*
  `seq` (even with different content); that step is `Fallback(HoldPosition)` with
  `ChunkUnderrun`, and a later fresh `seq` recovers `Policy`.
- All seventeen pre-existing scenarios and the property suite pass unchanged (`cargo test -p
  es-safety`).
- `cargo test -p es-runtime-embedded` passes, including `the_replan_cadence_is_honored` and
  `a_chunk_reuse_tick_does_not_allocate`, which exercise many reuse ticks between replans.
- `cargo check --workspace --all-targets` — `es-eval`'s existing `ActionChunk::new` call sites
  compile unmodified.
