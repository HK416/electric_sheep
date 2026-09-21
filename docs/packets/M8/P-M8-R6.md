# M8 R6 — `es-ir` under the target: `graph` and `hash` move down to `es-ir-types`

Spec: §1.5 (target ≤ 6,000 code lines, hard cap 10,000; over the target a split packet is
mandatory), §4.2 (`es-ir-types` is layer 2, `es-ir` layer 6 — a move *down* changes no
dependency direction), §5.3 (the hash chain's definitions do not move), §28.12 wave 0 (the
prerequisite of plan T's `ActionSpace` change). Review: `docs/reviews/M8.md` R6. Precedent: the
first split that created `es-ir-types` (`Expr`, `codes`, `diag`).

## the question

`es-ir` is at 6,148 code lines. `graph.rs` (402) and `hash.rs` (718) import only `codes`, `diag`
and each other — no IR struct — and `es-ir-types` already holds `codes` and `diag`. **Does moving
them down, with `es-ir` re-exporting them under the same paths, bring `es-ir` under the target
with no caller changed and no committed hash moved?**

## spec

* Move `crates/es-ir/src/graph.rs` and `crates/es-ir/src/hash.rs` to `crates/es-ir-types/src/`
  (as `graph` and `hash` modules), byte-for-byte except the `use` lines. If `hash.rs` turns out
  to import anything from `es-ir` proper beyond `codes`/`diag`/`graph`, move only `graph.rs` and
  say so; `norm.rs` stays (it imports the three IRs).
* `es-ir` re-exports: `pub use es_ir_types::{graph, hash};` so `es_ir::graph::…` and
  `es_ir::hash::…` keep resolving. No caller in the workspace changes; `git diff --stat` outside
  the two crates and their `Cargo.toml`s is empty.
* Tests that lived inside the moved files move with them; tests elsewhere in `es-ir` that pin
  hashes (`committed_*_hash*`, `hash_chain_*`, the `testing` property tests) run unchanged.
* `docs/design/hash-canonicalization.md` (+ `.ko.md`) and `docs/design/ir-types.md` (+ `.ko.md`)
  each gain one sentence saying where the module now lives.

## context

```
crates/es-ir/src/lib.rs
crates/es-ir/src/graph.rs
crates/es-ir/src/hash.rs
crates/es-ir/Cargo.toml
crates/es-ir-types/src/**
crates/es-ir-types/Cargo.toml
docs/design/hash-canonicalization.md
docs/design/hash-canonicalization.ko.md
docs/design/ir-types.md
docs/design/ir-types.ko.md
docs/packets/M8/P-M8-R6.md
docs/packets/M8/P-M8-R6.ko.md
```

## oracle

1. `cargo xtask context-budget` — `es-ir` below 6,000 code lines (report the number; expected
   ≈ 5,030 with both modules moved, ≈ 5,750 with `graph` alone).
2. `cargo test --workspace` — every existing test green, in particular the pinned committed
   hashes (`task eb6efefa…`, `task-pt d546b808…`, `learning 5dac0a46…`, `fdb5178a…`, the
   evaluation and deployment pins) and `es-ir/testing`'s property tests.
3. `cargo xtask layering` — no violation; `cargo xtask check-spec-refs`; `cargo xtask
   verify-goldens` (nothing changes); `cargo xtask check-scope docs/packets/M8/P-M8-R6.md`.
4. `git diff --stat main -- crates ':!crates/es-ir' ':!crates/es-ir-types'` is empty.

## acceptance

Oracles 1–4; the two notes' sentences with Korean siblings.

## forbidden

Changing any function's body, name or visibility; touching `norm.rs`, the five IR modules,
`cross.rs`, `factory.rs`, `serial.rs`; a new crate; any caller edit; `docs/ARCHITECTURE*.md`;
`tests/golden/**`.
