# P-M4-R3 — USD reader trust-boundary hardening (review S-1, S-2)

Spec: spec 1.9 item 6 (the native USD reader is the first thing cut under scope pressure, but
while it exists it reads untrusted `.usda` text, so spec 1.4's oracle-first rule applies to its
failure modes too). Follow-up to `docs/reviews/M4.md` should-fixes S-1 and S-2. Review class A.

## context

```
crates/es-usd/src/lib.rs
crates/es-usd/src/parse.rs
crates/es-physics-core/src/usd.rs
crates/es/src/cmd/import.rs
crates/es/Cargo.toml
crates/es/tests/cli.rs
docs/packets/M4/P-M4-R3.md
```

## spec

Three holes, all "attacker-controlled `.usda` file", none of them a semantics bug:

- **S-1** (`es-physics-core/src/usd.rs`, was `:514`) — `mesh_asset` read a `faceVertexCounts`
  entry as `count as usize`. Rust's float-to-int cast saturates rather than wrapping, so
  `1e20` became `usize::MAX`, not a panic by itself — but `cursor + n > indices.len()` then
  overflowed the *addition* (`cursor + n`) before the comparison ran, which wraps in release
  and can pass the guard, reaching `indices[cursor + offset]` with a `cursor` nowhere near a
  valid range. Fix: `usd_face_vertex_count` rejects anything that is not a finite integer
  fitting `u32` and at least 3 (`UsdSceneError::Invalid`, prim path attached), and the running
  `cursor` advances through `checked_add(n).filter(|c| *c <= indices.len())` instead of a raw
  `+`.
- **S-2** (`es-usd/src/parse.rs`, was `:635,650`) — `MAX_DEPTH` only counts `def`-inside-`def`
  prim nesting (`prim`/`prim_body`, checked at `:472`). `raw_value` and `raw_sequence` recurse
  into each other for every `[...]`/`(...)` literal with no counter of their own, so
  `[[[[...]]]]` deep enough overflows the native stack — an abort, not a `UsdError`, and not
  catchable by the caller. Fix: a `value_depth: usize` field on `Parser`, incremented and
  checked against a new `MAX_VALUE_DEPTH = 64` in a `nested_sequence` wrapper both recursion
  sites now call, `UsdError::Syntax` with the offending line once it is exceeded. Separately,
  `lex` (`:180`, was un-numbered before this packet) expands the whole layer into a `Vec<char>`
  over whatever `es/src/cmd/import.rs` (`:297`) read from disk with no size check at all — a
  large file drives an uncapped allocation before a single token exists. Fix: a new
  `es_usd::MAX_USDA_BYTES` (64 MiB) checked against `text.len()` at the top of `parse_usda`,
  before `strip_header`/`lex` ever run, and the same constant checked against
  `std::fs::metadata(&path).len()` in the CLI's `usd_import` before it calls
  `read_to_string` — refused by file size, not read-then-rejected.

Not done: relexing over `char_indices`/bytes instead of `Vec<char>`. `lex_string`,
`lex_delimited` and `lex_number` all index `chars: &[char]` by element position and compare
`char` values directly; converting the whole tokenizer to byte offsets touches every one of
them for a memory-shape win the new size cap already bounds (64 MiB of UTF-8 becomes at most
~64M `char`s, a fixed ceiling either way). Out of scope for a should-fix packet; flagged as a
possible follow-up, not attempted here.

## oracle

```
cargo test -p es-usd -p es-physics-core
cargo test -p es usd
```

Three cases the fix must turn into a clean `Err`, none of them a crash:

- `usd::tests::a_face_vertex_count_that_overflows_u32_is_an_error_not_ub`
  (`es-physics-core/src/usd.rs`) — `faceVertexCounts = [3, 1e20, ...]` on the existing
  `mesh_cube.usda` fixture (edited in-test, no new fixture file) returns
  `UsdSceneError::Invalid` naming `faceVertexCounts`, not a panic or an out-of-bounds index.
- `parse::tests::deep_value_nesting_errors_instead_of_overflowing` (`es-usd/src/parse.rs`) —
  100 KB of `[` (generated in-test, not a fixture file) returns `UsdError::Syntax` rather than
  aborting the process. Ran under the default `cargo test` stack (no `--test-threads=1` special
  case needed) and completes in the same test binary in well under a second, which is the
  in-process form of "wrap in a thread with a timeout": a real stack overflow would abort the
  whole test binary instead of returning an `Err`, so the test passing at all is the proof.
- `parse::tests::oversized_layer_is_rejected_before_the_char_vec` (`es-usd/src/parse.rs`) and
  `import_usd_refuses_an_oversized_file` (`es/tests/cli.rs`) — a `MAX_USDA_BYTES + 1`-byte
  layer (built in-test with `"a".repeat(...)` / `vec![b'a'; ...]`, no fixture file, keeping
  `tests/fixtures/usd/` under 200 KB) is refused by both the parser directly and the CLI's
  `usd_import` (by file metadata length, before the read).

## acceptance

- `cargo fmt -p es-usd -p es-physics-core -p es --check` clean.
- `cargo clippy -p es-usd -p es-physics-core -p es --all-targets -- -D warnings` clean.
- `cargo test -p es-usd -p es-physics-core` — 19 (`es-physics-core`) + 15 (`es-usd`) tests
  pass, all pre-existing tests unchanged and still green, three new tests added.
- `cargo test -p es usd` — the two existing `import_usd_*` CLI tests plus the new
  `import_usd_refuses_an_oversized_file` pass.
- No new fixture file added to `tests/fixtures/usd/` (all three hostile inputs are generated
  in-test); the directory stays at its pre-packet size.
- No change to `UsdError`'s or `UsdSceneError`'s variant shapes — both hostile inputs map onto
  the existing `Syntax`/`Invalid` variants with a message, not a new error kind.

## forbidden

`es-script`, `es-eval`, `es-data`, `es-env`, `es-ir`, `es-gpu`, `es-render`, `es-core` (other
review follow-ups in flight against those crates). `es/src/cmd/import.rs`'s RoboVerse branch
(`roboverse_import`, S-9/S-10) — this packet touches only `usd_import`'s read. Relexing
`es-usd::parse::lex` off `Vec<char>` (noted above, not attempted). Any change to the mapping
`mesh_asset` performs beyond the count/cursor checks — geometry semantics are out of scope.
