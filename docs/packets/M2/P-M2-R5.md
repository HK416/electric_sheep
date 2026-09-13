# P-M2-R5 — one chunk-buffer sizing model

Spec: spec 4.2, spec 8.6, spec 12.4, spec 20.2, spec 28.7. Design notes:
`docs/design/memory-budget.md` §3, `docs/design/batch-domains.md` §11. Review finding:
`docs/reviews/M2.md` "Should-fix" `crates/es-compile/src/budget.rs:420` vs
`crates/es-env/src/domains.rs:495`.

## context

```
crates/es-core/src/sizing.rs         (new: CHUNK_SLOTS + chunk_buffer_bytes)
crates/es-core/src/lib.rs            (adds `pub mod sizing;`)
crates/es-env/src/chunk_buffer.rs    (re-exports the constant)
crates/es-env/src/domains.rs         (`DomainSizing::chunk_buffer_bytes` delegates)
crates/es-compile/src/budget.rs      (the `chunk_buffers` item + its tests)
docs/design/memory-budget.md         (§2, §3)
docs/packets/M2/P-M2-R5.md           (this file)
```

## spec

Two in-tree models of the same quantity disagreed 8×: the budget used spec 20.2's
`n_sim_envs × H × NJ × 4B × 2`, the runtime allocated `CHUNK_SLOTS = 8` slots of `f64`. Gate
13's ±10% is unreachable while both exist.

`es-compile` is layer 7 and `es-env` layer 9, so neither can call the other. The formula
therefore lives at layer 1:

```rust
es_core::sizing::CHUNK_SLOTS: usize            // 8
es_core::sizing::chunk_buffer_bytes(n_envs, nj, h) -> u64
    = n_envs * CHUNK_SLOTS * h * nj * 8        // saturating
```

`es_env::CHUNK_SLOTS` is now a re-export of it, `DomainSizing::chunk_buffer_bytes` delegates
to it (the `chunk_slots` field is gone — it was a second copy of the constant), and
`MemoryBudget::estimate`'s `chunk_buffers` item calls the same function.

The **deviation from spec 20.2 is deliberate and stated in the formula string** the budget
prints: the runtime keeps `CHUNK_SLOTS` *overlapping* chunks of `f64` per env, because ACT
temporal ensembling averages every live chunk (spec 8.6) and the control path is f64
throughout. That is 4× the spec's f32 double-buffered line, and it is the implementation that
is right — the spec's line predates the ensembling buffer.

## oracle

```
cargo test -p es-compile budget
cargo test -p es-core sizing
```

`chunk_buffers_agree_with_the_runtime_sizing_at_the_gate_configuration` asserts
`MemoryBudget::estimate(..).chunk_buffers` equals `es_core::sizing::chunk_buffer_bytes` for the
spec 12.4 gate configuration (4,096 sim envs, `H = 20`, `NJ = 7`) — 36,700,160 B, the same
number `es_env::DomainSizing::GATE` asserts — and that the printed formula names the
deviation.

## acceptance

- `cargo fmt --check`,
  `cargo clippy -p es-core -p es-env -p es-compile --all-targets -- -D warnings`,
  `cargo test -p es-core -p es-env -p es-compile`, `cargo xtask layering`,
  `cargo xtask check-spec-refs`.
- No new dependency edge: `es-core` is already below both callers (spec 4.2).
- No new trait (INV-17).

## forbidden

- Changing `CHUNK_SLOTS` itself, or the runtime's f64 chunk storage — this packet unifies the
  two models, it does not re-tune either.
- `crates/es-compile/src/plan.rs`, `exec.rs` and the other budget items.
