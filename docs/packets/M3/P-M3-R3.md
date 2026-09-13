# P-M3-R3 — bound the PLY reader's reserve

Spec: §1.4 (oracle-first), §3.4/App. D (no unbounded allocation from untrusted input), §16
(this crate's scope). Follow-up to `docs/reviews/M3.md` Should-fix `ply.rs:441` and to
`W3-splat-importer.md`, which this packet stays inside the scope of.

`import_ply`'s ascii path reserved `3 * count`, `4 * count` and `per_rest * count` elements
straight from the header's *declared* `element vertex` count, before the per-line field-count
check (`fields.len() != header.props.len()`) ever ran. A hostile ascii header can declare far
more vertices than the file has bytes for — the file's own non-blank-line count only bounds
`count` loosely, at 2 bytes/line — so a ~1 MB file declaring `element vertex 500000` against a
full SH-degree-3 (59-property) header asked for roughly 118 MB before the first record was
validated. The binary path was already safe: its `Truncated` check (`available < count *
stride`) runs *before* the reserve and rejects any `count` the file cannot actually hold.

## context

```
crates/es-splat/src/ply.rs
crates/es-splat/tests/splat.rs
crates/es-splat/Cargo.toml
docs/packets/M3/P-M3-R3.md
```

## spec

1. **`reserve_count(format, props, stride, count, available) -> usize`** — a private helper
   in `ply.rs` used by `import_ply` in place of the header's raw `count` for every
   `SplatScene` array's `Vec::with_capacity`. Bounds the reserve by how many full vertex
   records `available` bytes could possibly hold: `count.min(available / (2 * props))` for
   ascii (a record needs at least one digit and one separator per property), `count.min(
   available / stride)` for binary (already implied by the pre-existing `Truncated` check;
   this makes it an explicit, direct bound rather than an incidental one).
2. No change to `import_ply`'s control flow, error variants, or the loop that decodes each
   vertex: `reserve_count` only sizes the up-front capacity, so a file with more real records
   than the conservative estimate still decodes correctly via ordinary `Vec` growth.
3. `es-splat`'s `[dev-dependencies]` gain `es-core` with the `alloc-count` feature (the same
   pattern `es-safety` and `es-runtime-embedded` already use), so a test can observe
   allocation counts under `cfg(test)`.

## oracle

```
cargo fmt -p es-splat --check
cargo clippy -p es-splat --all-targets -- -D warnings
cargo test -p es-splat
cargo xtask check-spec-refs
```

Specifically: `ply::reserve_tests::ascii_reserve_is_bounded_by_file_size_not_declared_count`
pins `reserve_count`'s output directly — for the hostile parameters (`count = 500_000`,
`available = 1_000_000`, `props = 59`) the worst-case reserve across all six arrays (59 `f32`
slots per reserved vertex) is under 8 MB, versus the ~118 MB the pre-fix `3 * count` /
`per_rest * count` calls implied. `a_hostile_ascii_header_does_not_amplify_the_reserve` in
`tests/splat.rs` imports a generated ~1 MB ascii PLY declaring `element vertex 500000` with
one token per line and asserts `SplatError::BadFieldCount { vertex: 0, found: 1, expected: 59
}`, plus (via `es_core::alloc_count::allocation_count`) that the import allocates on the order
of a hundred times, not the hundreds of thousands an unreserved, per-push growth path would
need — the byte bound itself is pinned by the pure-function test above, since the counter
counts calls, not bytes.

## acceptance

- The hostile-header test passes; the byte-exact round-trip and ascii/binary cross-decode
  tests are unaffected (they exercise `reserve_count` on well-formed input, where it equals
  the true `count` since a legitimate file always has more than `2 * props` bytes available
  per real record).
- `binary_reserve_is_bounded_by_stride` pins that the binary path's bound is unchanged
  (`available / stride`, matching what the pre-existing `Truncated` check already
  guaranteed).

## forbidden

- Changing `SplatError` variants, the per-line field-count check, or the binary `Truncated`
  check's condition — this packet only changes *how much* is reserved up front, not what
  counts as valid input.
- Anything outside `crates/es-splat/**` and this packet file. `es-core`'s `alloc-count`
  feature is consumed as a dev-dependency, never edited.
- Committing. The oracle is run and reported, not landed.
