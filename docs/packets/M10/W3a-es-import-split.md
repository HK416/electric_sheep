# M10 W3a — `es-data` under the target: the three importers move down to `es-import`

Spec: §1.5 (target ≤ 6,000 code lines, hard cap 10,000; over the target a split packet is
mandatory), §4.2 (the row added 2026-09-22: `es-import` at layer 9 — below `es-data` (10), above
`es-policy` (8), which `rl_import` needs; a move *down* changes no dependency direction), §5.3
(no hash definition moves), §28.13 wave 1. Review: `docs/reviews/M8.md` R6 / human decision "the
line budget". Precedent: `docs/packets/M8/P-M8-R6.md` (`graph` / `hash` → `es-ir-types`, byte-for-byte
except `use` lines, re-exported, zero caller edits). Type B.

## the question

`es-data` reports 7,321 code lines (≈ 6,900 real — see the miscount below). `lerobot_config.rs`
(935), `roboverse.rs` (688) and `rl_import.rs` (829) import nothing from the crate but the LeRobot
feature specs and each other, touch only `es-core` / `es-ir` / `es-math` / `es-policy`, have their
tests in separate files, and are used only by `crates/es/src/cmd/{import,policy}.rs`. **Does moving
them to a layer-9 crate `es-import`, re-exported by `es-data` under the same paths, bring `es-data`
under the target with no caller changed and no committed hash moved?**

## spec

* New crate `crates/es-import` (layer 9; deps `es-core`, `es-ir`, `es-math`, `es-policy`, `serde`,
  `serde_json`, `thiserror`, and whatever the three files already use). Move
  `crates/es-data/src/{lerobot_config,roboverse,rl_import}.rs` and
  `crates/es-data/tests/{lerobot_config,roboverse,rl_import}.rs` **byte-for-byte except the
  `use` lines**. If `lerobot_config` needs something from `es_data::lerobot` (feature specs),
  move the smallest self-contained piece it needs into `es-import` and re-export it from
  `es-data` too, or leave the dependency the other way if it is only a type — state which.
* Error type: the three modules' errors become the new crate's (`ImportError` or per-module),
  and `es_data::DataError` gains a `#[from]` so every existing `?` site keeps compiling.
* `es-data` re-exports: `pub use es_import::{lerobot_config, roboverse, rl_import};` so
  `es_data::lerobot_config::…` etc. keep resolving. `git diff --stat main -- crates ':!crates/es-data'
  ':!crates/es-import'` is empty except `Cargo.toml` / `Cargo.lock` lines.
* `xtask/src/layering.rs` `LAYERS` gains `("es-import", 9)` (the spec table already has the
  row). **The context-budget miscount**: `xtask/src/context_budget.rs`'s `brace_delta` does not
  handle raw strings, and `crates/es-data/src/training.rs:2194` opens `r#"` inside `mod tests`
  with braces in its TOML body, so ≈ 419 test lines count as code; fix the counter to skip raw
  string bodies (a few lines) — it is this packet's own oracle, so it is in scope — and report
  the numbers before and after the fix.
* Design notes: `docs/design/policy-bundle.md` or `python-builder.md` — wherever `rl_import`
  is described — and `lerobot-config`'s note gain one sentence saying where the module lives
  (+ `.ko.md`).

## context

```
crates/es-import/**
crates/es-data/src/lib.rs
crates/es-data/src/lerobot_config.rs
crates/es-data/src/roboverse.rs
crates/es-data/src/rl_import.rs
crates/es-data/src/lerobot/**
crates/es-data/tests/**
crates/es-data/Cargo.toml
Cargo.toml
Cargo.lock
xtask/src/layering.rs
xtask/src/context_budget.rs
docs/design/policy-bundle.md
docs/design/policy-bundle.ko.md
docs/design/python-builder.md
docs/design/python-builder.ko.md
docs/packets/M10/W3a-es-import-split.md
docs/packets/M10/W3a-es-import-split.ko.md
```

## oracle

1. `cargo xtask context-budget` — `es-data` < 6,000 (expected ≈ 4,450 after the counter fix,
   ≈ 4,870 before), `es-import` ≈ 2,450; every other crate's number unchanged except by the
   counter fix, which is listed.
2. `cargo test --workspace` — every pinned hash green; `es policy import-rl` and `es import`
   CLI tests unchanged.
3. `cargo xtask layering`; `cargo xtask check-spec-refs`; `cargo xtask verify-goldens`.
4. `git diff --stat main -- crates ':!crates/es-data' ':!crates/es-import'` — Cargo files only.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W3a-es-import-split.md`.

## acceptance

Oracles 1–5; the note sentences with their Korean siblings.

## forbidden

Changing any function body, name or visibility beyond `use` lines and the error `#[from]`;
editing a caller outside the two crates; touching `training.rs`, `collect.rs`, `identity.rs`,
`intervention.rs` (P-M9-R5 owns `training.rs` in this wave); `tests/golden/**`;
`docs/ARCHITECTURE*.md`; a non-`es-` crate name.
