# P-M3-R2 — `es gap` survives a hostile dataset

Follow-up to the M3 review should-fixes (`docs/reviews/M3.md`: `domain_gap.rs:174`,
`cmd/gap.rs:168`, `domain_gap.rs:350`).

Spec: §24.3 (domain-gap diagnostics), §19.1 (LeRobot dataset layout), §28.7 gate 15,
§1.2 (work packets), §1.4 (oracle first).

Design note: `docs/design/domain-gap.md` §3, §4, §5, §7.

## context

```
crates/es-eval/src/domain_gap.rs   (GapReport::to_json, FeatureSamples, ChannelGap, compute)
crates/es/src/cmd/gap.rs           (build_input)
crates/es/tests/cli.rs             (append tests)
docs/design/domain-gap.md          (§3, §4, §5, §7)
docs/packets/M3/P-M3-R2.md         (this file)
```

Three trust-boundary holes, all on the path from a dataset this repo did not write to
`gap_report.json`.

## spec

1. **Non-finite samples.** `build_input` keeps a frame only when every one of its dims is
   finite; a dropped frame increments `FeatureSamples::nonfinite_dropped`, which `compute`
   carries into `ChannelGap::sim_nonfinite_dropped` / `real_nonfinite_dropped` and `Display`
   prints as a total. Neither KS nor Wasserstein-1 has a meaning over `NaN`, and `serde_json`
   refuses to encode one.
2. **`to_json` returns `Result`.** `GapReport::to_json(&self) -> Result<String,
   serde_json::Error>`; the `.expect` is gone. The CLI maps the error to `CliError::Runtime`.
   A command that has already printed its table must not panic on the way to disk.
3. **The column slice is checked.** `flat.get(local * dims..(local + 1) * dims)` with a
   `DataError::Inconsistent` on the miss, naming the episode, the channel, how many values the
   column holds and how many the declared `elem_count` implies. `dims` comes from
   `info.json`; a column that disagrees is an inconsistent dataset, not an index panic.
4. **Mismatched widths are listed, not clamped.** A channel both sides carry under the same
   name with different `dims` becomes a `GapReport::dims_mismatch` entry (`DimsMismatch
   { channel, sim_dims, real_dims }`) and is not scored. `compare_channel` is only ever called
   on equal widths, so the `d.min(dims - 1)` clamp — which compared two unrelated components —
   is gone.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-eval -p es
cargo xtask check-spec-refs
```

## acceptance

- `gap_hostile_dataset_errors_instead_of_panicking`: an in-test `LeRobot` dataset with a `NaN`
  column *and* an `info.json` declaring a feature wider than its parquet column makes `es gap`
  exit 1 with `error: ... inconsistent ...` on stderr — no panic, no index out of bounds.
- `gap_nonfinite_samples_are_dropped_and_counted`: a `NaN` alone is not fatal — the run
  completes, `gap_report.json` parses, the affected channel reports its dropped-frame count,
  and its `mean`/`std` are finite.
- `a_channel_whose_width_disagrees_is_listed_not_clamped`: a 2-wide sim channel against a
  3-wide real one produces no `ChannelGap` row and one `dims_mismatch` entry.
- `report_json_round_trips` still holds through the new `Result` signature.

## forbidden

- Adding a variant to `es_data::DataError` or editing `crates/es-data/**`: `Inconsistent` is
  exactly what a self-contradicting dataset is, and another packet owns that crate.
- Silently substituting a value for a non-finite sample, or a `0.0` for a channel that was
  never compared — the design doc's "nothing is invented" rule.
- Touching `crates/es-splat/**`, `xtask/**`, `.github/**`, `crates/es/src/cmd/evidence.rs`.
- A new extension-point trait (INV-17) or a `HashMap` in the report path (§18.4).
