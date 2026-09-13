# P-M1-R6 — freeze the node kind lists

Fixes an M1 review Should-fix (`docs/reviews/M1.md`, Should-fix, `crates/es-ir/src/factory.rs:93,103`;
also spec 28.7 gate 10, listed "partially" met): `BUILTIN_TASK_KINDS` / `BUILTIN_LEARNING_KINDS`
were declared "frozen" only in a doc comment — nothing in CI failed if a kind was renamed,
reordered, added, or removed, even though doing so silently changes `task_hash` /
`learning_hash` for every graph that uses it.

Spec: spec 6.3, spec 8.3, spec 28.7 gate 10.

## context

```
crates/es-ir/src/factory.rs   (kinds_hash helper, two frozen [u8; 32] consts, test module)
docs/packets/M1/P-M1-R6.md    (new)
```

## forbidden

`crates/es-compile`, `crates/es-telemetry`, `crates/es-eval`, `crates/es-data`, `.github/**` —
owned by concurrent M1 follow-up work. No new trait — `kinds_hash` is a private free function,
not an extension point (`INV-17` unaffected). `docs/design/ir-types.md` is not edited by this
packet; its migration-note requirement is documented for whoever next changes the lists.

## spec

- `kinds_hash(kinds: &[&str]) -> [u8; 32]`: sorts the kinds (so declaration order in the
  source array is not part of the contract), joins with `\n`, hashes with `blake3::hash`.
- `BUILTIN_TASK_KINDS_HASH` / `BUILTIN_LEARNING_KINDS_HASH`: hardcoded `[u8; 32]` hex literals,
  computed once by running the hash and pasting the result (see below). Doc comment on each:
  "changing this hash is a schema change: bump `schema_version` and add a migration note in
  docs/design/ir-types.md".
- A test in `factory.rs` recomputes `kinds_hash` over the live `BUILTIN_TASK_KINDS` /
  `BUILTIN_LEARNING_KINDS` and asserts equality with the hardcoded constant, so a rename,
  reorder-that-changes-membership, addition, or removal fails `cargo test -p es-ir`.

Computed hashes (blake3 over the sorted kind list joined by `\n`), each 32 bytes as hex:

```
BUILTIN_TASK_KINDS_HASH     = 079283852d9d3faf85c7da5d48fa0c88210ab5c37c82effc3bb236e2603e144c
BUILTIN_LEARNING_KINDS_HASH = a17d0653f6c0160cfce352a486ca040bbdbdf4b52139a12ffcf0920e1caffe51
```

The authoritative form is the `[u8; 32]` hex-array literal in `factory.rs` (one `0xNN` per
byte, in this same order).

## oracle

```
cargo fmt --check
cargo clippy -p es-ir --all-targets -- -D warnings
cargo test -p es-ir
```

## acceptance

- `cargo test -p es-ir factory::frozen_kind_lists::` passes: both hardcoded hashes match a
  fresh `kinds_hash` computation.
- Manually renaming one entry in either `BUILTIN_TASK_KINDS` or `BUILTIN_LEARNING_KINDS` (not
  committed — a local check only) makes the corresponding test fail, confirming gate 10 is now
  machine-checked rather than a comment.
- `cargo test -p es-ir` in full still passes (`factory_basic.rs`'s existing coverage tests are
  unaffected — they still exercise the lists directly, not the hash).
