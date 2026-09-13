# P-M3-R7 — mismatched metadata does not panic

Spec: §13.2 (intervention labels), §19.2 (dataset identity), W7
(`docs/packets/M3/W7-learning-loop.md`).
Review finding: `docs/reviews/M3.md` Should-fix, `crates/es-data/src/intervention.rs:280` —
`m[i]` indexed a mask sized from `meta.length` with `i` running over `ep.len()`; a dataset
whose episode metadata disagrees with its parquet would panic there instead of reporting the
inconsistency the rest of the crate already treats as a `DataError`.

## context

```
crates/es-data/src/intervention.rs
crates/es-data/tests/loop_learning.rs
docs/packets/M3/P-M3-R7.md
```

## spec

- `label`'s per-episode loop compares the mask's length source (`meta.length`, from
  `dataset.episodes()`) against `ep.len()` (the parquet episode `read_episode` actually
  returned) **before** using either to index the other, and returns
  `DataError::Inconsistent` naming the episode and both lengths when they disagree — not
  `m.get(i).copied().unwrap_or(false)`, which the review explicitly rejects: a mismatch is a
  bad dataset, and papering over it with a default silently drops real intervention frames
  instead of reporting the corruption (spec 13.2's whole point is a trustworthy label).
  `DataError::Inconsistent` stays the existing `String`-payload variant (used a dozen places
  across `es-data` already, including three lines earlier in this same function) rather than
  growing a new struct-shaped variant — one more call site formatting a descriptive message
  is a smaller diff than restructuring every existing caller of `Inconsistent`.
- The guard runs once per episode, ahead of the `values` vector that indexes `masks`, so it
  covers every index in that episode's frame range regardless of whether the episode has any
  intervention segments at all.

## oracle

```
cargo test -p es-data label_on_mismatched_metadata_is_an_error_not_a_panic
```

Builds a dataset with `collect_into`, hand-edits `meta/episodes.jsonl` so episode 0's
declared `length` is shorter than the parquet file `collect_into` actually wrote, then calls
`es_data::label(&root, &[])` and asserts it returns `DataError::Inconsistent` rather than
panicking.

## acceptance

- `label` on a dataset whose `meta/episodes.jsonl` length is shorter than its parquet
  returns `Err(DataError::Inconsistent { .. })` (the existing `String`-payload variant), not
  a panic.
- No `unwrap`, `expect`, or `.get(..).unwrap_or(..)` was introduced on the mask-indexing
  path.
- `cargo test -p es-data` stays green, including the existing `loop_learning.rs` and
  `lerobot.rs` fixtures.

## forbidden

- No change to the `DataError` enum's shape (no new struct-style `Inconsistent` variant) —
  every existing `DataError::Inconsistent(String)` call site across `es-data` would need
  updating for no behavioral gain.
- No edits to `crates/es-eval`, `crates/es-splat`, or any other `crates/es-*` crate.
- Tests are append-only in `crates/es-data/tests/loop_learning.rs` — no edits to existing
  tests or fixtures in that file.
