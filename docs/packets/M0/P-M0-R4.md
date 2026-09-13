# P-M0-R4 — split `es-ir` back under the spec 1.5 context budget

Spec: spec 1.5 (context budget), spec 4.2 (crate layering), Appendix B.8 (`LAYERS`). Follow-up
to the `[HUMAN]` item on `es-ir` line count in `docs/reviews/M0.md`. Review class A.

## context

```
crates/es-ir-types/**
crates/es-ir/src/lib.rs
crates/es-ir/src/graph.rs
crates/es-ir/src/hash.rs
crates/es-ir/Cargo.toml
Cargo.toml
xtask/src/layering.rs
docs/packets/M0/P-M0-R4.md
```

## spec

`es-ir` was 6,589 code lines (6,669 after P-M0-R2), over the spec 1.5 target of 6,000.

Two options from the review were rejected:

- **Moving the `testing` modules out is a no-op.** `xtask/src/context_budget.rs` already drops
  any `#[cfg(...test...)] mod … { }` block, and the proptest strategies are all inline
  `#[cfg(any(test, feature = "testing"))] pub mod testing` blocks. The 6,589 figure is already
  net of them (3,465 excluded lines); moving them to files would change nothing.
- **A sibling at layer 6 (`es-ir-deploy`) is forbidden** — it would have to depend on `es-ir`
  for `Graph`/`IrNode`, which is a same-layer dependency (spec 4.2 rule 1).
- **`es-diag` (just `codes.rs` + `diag.rs`) is too small**: 290 code lines, leaving `es-ir` at
  ~6,380, still WARN.

So: a new crate **`es-ir-types` at layer 2** (deps `es-math` layer 0, `es-core` layer 1, so 2 is
the lowest layer it can sit at) holding the IR vocabulary that does not know what a graph is:

| moved | from |
|---|---|
| `codes.rs`, `diag.rs`, `types.rs`, `image.rs` | `crates/es-ir/src/` verbatim |
| `canon.rs` — `CanonWriter` | `crates/es-ir/src/hash.rs` |
| `NodeId` (`lib.rs`) | `crates/es-ir/src/graph.rs` — `Diagnostic` points at one |

Every public path stays where it was, by re-export: `es-ir`'s `lib.rs` does
`pub use es_ir_types::{codes, diag, image, types};`, `hash.rs` does
`pub use es_ir_types::canon::CanonWriter;`, `graph.rs` does `pub use es_ir_types::NodeId;`. A
`pub use` of a module at the crate root also keeps `crate::codes::…` resolving inside `es-ir`,
so no module body changed apart from three `use` lines in the moved files.

`xtask/src/layering.rs` gains `("es-ir-types", 2)`. This is a deviation from the Appendix B.8
table, which predates the split; the spec should pick it up when Appendix B.8 is next revised.

## oracle

```
cargo xtask context-budget && cargo xtask layering && cargo test --workspace --features es-ir/testing
```

## acceptance

- `context-budget` reports `OK` for every crate: `es-ir` 5,774 (was 6,669), `es-ir-types` 909.
- `layering` passes with 18 crates, no violations.
- `cargo check --workspace --all-targets --features es-ir/testing` is clean with **no** change
  to any file in `es-compile`, `es-env`, `es-policy`, `es-data`, `es-safety`, `es`,
  `es-runtime-embedded` or the `es-ir` integration tests — every `es_ir::…` path still resolves.
- `cargo test --workspace --features es-ir/testing` passes.

## forbidden

Any file outside `context`. Moving anything that mentions `Graph`, `IrNode` or an IR node kind
into `es-ir-types` — the split line is "knows nothing about graphs". Changing a public path
without a re-export at the old one. Editing `docs/ARCHITECTURE.ko.md`.
