# P-M3-R6 — collinear correspondences are a degenerate fit

Spec: §16.2 (position alignment, `Similarity::fit`), §1.4 (oracle-first). Follow-up to
`docs/reviews/M3.md` Should-fix `align.rs:97` and to `W3-splat-importer.md`, which this packet
stays inside the scope of.

`Similarity::fit`'s doc already promised "degenerate input is an error, never a silent
identity" and named three non-collinear points as what fixes a rotation. The only check
implemented was `src_spread <= 0.0`, which catches coincident points (zero spread in every
direction) but not collinear ones: four points on a line have positive `src_spread` but no
well-defined rotation *about* that line, so Horn's method silently returned a finite,
arbitrary rotation instead of the promised error.

## context

```
crates/es-splat/src/align.rs
crates/es-splat/tests/splat.rs
docs/packets/M3/P-M3-R6.md
```

## spec

1. **Rank check in `Similarity::fit`** — after the existing coincident-points check
   (`src_spread <= 0.0` → `SplatError::DegenerateFit`, unchanged), compute the centred `src`
   point of largest norm (`axis`, a fixed, order-independent choice: first point reached in
   iteration order with the running-maximum norm) and the sum of every centred point's squared
   component perpendicular to `axis` (`perp_spread`) — the second principal spread. When
   `perp_spread` is negligible relative to `src_spread` (`<= src_spread *
   f64::EPSILON.sqrt()`), every correspondence lies on `axis`'s line through the centroid:
   `SplatError::DegenerateFit`.
2. Fixed operation order throughout (accumulate `axis`/`axis_norm2` in the same pass as `s`
   and `src_spread`; the perpendicular-spread pass iterates `src` once, in order), matching
   the file's existing "no data-dependent iteration count, no reassociation" discipline — this
   is a reproducibility property, not a performance one.
3. No change to the coincident-points check, to `TooFewPoints`/`PointCountMismatch`/
   `NonFiniteSample`, or to `horn_rotation`/`jacobi_eigen`.

## oracle

```
cargo fmt -p es-splat --check
cargo clippy -p es-splat --all-targets -- -D warnings
cargo test -p es-splat
cargo xtask check-spec-refs
```

Specifically: `similarity_fit_refuses_collinear_input` in `tests/splat.rs` fits four points on
a line (axis-aligned, `src == dst`) and asserts `SplatError::DegenerateFit`, then repeats with
a non-axis-aligned collinear `src` against a different (still collinear) `dst`, so the check is
not a special case of "identity" or "along a coordinate axis". The existing
`similarity_fit_refuses_degenerate_input` (coincident points) and
`similarity_fit_recovers_a_known_transform` (64 seeds of 10 pseudorandom, generically
non-collinear points) are unchanged and still pass.

## acceptance

- Four collinear points are `SplatError::DegenerateFit`, not a finite rotation.
- The coincident-points case and the 64-seed recovery property are unaffected: random 3-D
  points are essentially never collinear, so `perp_spread` stays far above the epsilon
  threshold for genuinely well-posed input.

## forbidden

- Changing the coincident-points check's threshold or error variant.
- A new error variant — collinear input reuses `SplatError::DegenerateFit`, per the doc's own
  wording ("degenerate input is an error").
- `ColorAffine::fit`, `Binding::bind`/`skin`, or anything in `ply.rs` — this packet is
  `align.rs`'s rank check only.
- Anything outside `crates/es-splat/**` and this packet file.
- Committing. The oracle is run and reported, not landed.
