# M12 R4 — `es-editor-model` (layer 12) and `es-editor` (layer 13)

Owner decision H-2, 2026-09-29 (`docs/reviews/M12.md`): `es-editor` is at 9,763 of §1.5's 10,000
code lines, and S2–S6 cannot land in it. The headless view-models become their own crate; the egui
shell sits one layer above them. Spec §1.5, §4.2 (the layer table and its rules), Appendix C.8
(`LAYERS`); `docs/design/editor-shell.md` section 2 (the model/egui split is already the rule).

## spec

- New crate `crates/es-editor-model` (layer **12**): everything under
  `crates/es-editor/src/model/` today, the i18n tables (`crates/es-editor/i18n/` →
  `crates/es-editor-model/i18n/`) and every test that needs no display. No egui dependency except
  what `model/` already names (`fonts.rs` builds `egui::FontDefinitions`; keep that dependency and
  say so in the crate doc, or move `fonts.rs` to the shell if it is the only reason — whichever
  leaves `es-editor-model` without `eframe`).
- `crates/es-editor` (layer **13**): `app.rs`, `ui/`, `main.rs`, `lib.rs`; it depends on
  `es-editor-model`. Nothing depends on `es-editor` (§4.2 rule 4 stays); **nothing depends on
  `es-editor-model` except `es-editor`** (a new sentence in rule 4).
- Spec: §4.2's table gains row `| 13 | es-editor |` and row 12 becomes `es-editor-model`; rule 4
  names both; Appendix C.8's `LAYERS` gains `("es-editor-model", 12)` and `("es-editor", 13)`.
  `docs/ARCHITECTURE.ko.md` and `docs/ARCHITECTURE.md` in the **same commit** (the pre-commit hook
  checks). `xtask/src/layering.rs` `LAYERS` and its rule-4 check follow; `CLAUDE.md`'s layering
  line says `es-editor-model`(12) → `es-editor`(13).
- A pure move otherwise: no behaviour change, every test keeps passing under its new crate, the
  goldens under `tests/golden/editor/` byte-identical, `include_str!`/fixture paths adjusted.
- Both crates under the §1.5 target where possible; report both numbers.

## context

```
crates/es-editor/**
crates/es-editor-model/**
Cargo.toml
Cargo.lock
xtask/src/layering.rs
xtask/src/context_budget.rs
docs/ARCHITECTURE.md
docs/ARCHITECTURE.ko.md
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/design/editor-redesign.md
docs/design/editor-redesign.ko.md
CLAUDE.md
.githooks/pre-commit
docs/packets/M12/P-M12-R4-editor-split.md
docs/packets/M12/P-M12-R4-editor-split.ko.md
```

## oracle

1. `cargo test -p es-editor-model` and `cargo test -p es-editor` together pass exactly the tests
   `cargo test -p es-editor` passes on main today (same names, same count).
2. `cargo xtask layering` green with the new rows; a unit test in `layering.rs` that
   `es-editor-model` may not depend on `es-editor` and that no third crate may depend on
   `es-editor-model`.
3. `cargo xtask context-budget`: both crates under 10,000.
4. `cargo xtask verify-goldens`, `check-spec-refs`, fmt, clippy `-D warnings`, `check-scope`;
   `cargo build -p es-editor` (the binary).

## forbidden

Any behaviour change; renaming model types; connecting to any remote server.
