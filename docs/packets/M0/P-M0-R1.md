# P-M0-R1 — `BinnedAcc` poison ordering (M0 review blocker)

Spec: spec 18.4 (deterministic reduction), spec 3.4 (deterministic execution contract),
Appendix B.8 (signature), Appendix D `DET-011`. Follow-up to the M0 review
(`docs/reviews/M0.md`, Blocker). Review class A.

## context

```
crates/es-math/src/reduce.rs
docs/design/deterministic-reduce.md
docs/packets/M0/P-M0-R1.md
```

## spec

- `add` records a non-finite input in a dedicated poison state **before** any bin arithmetic
  and returns; nothing non-finite ever enters `bins`. The old code wrote `bins[0] += v` and
  then called `rescale_to(0)`, which shifted the poison off the bottom of the bin array
  whenever a prior deposit had already set a coarser index — `add(1e-300); add(NAN)` finished
  as `0.0`, `add(NAN); add(1e-300)` as `NaN`.
- The state is a `u8` bit set (`+Inf` / `-Inf` / `NaN`), so `merge` combines it with `|`:
  commutative, associative, idempotent. No allocation, still `Copy`.
- `finish` resolves it with IEEE `sum` semantics, ahead of the bin sum: `NaN` dominates;
  `+Inf` together with `-Inf` is `NaN`; a lone infinity dominates the finite terms; `NaN` is
  returned as the `f64::NAN` constant so `to_bits()` is stable across permutations.
- `merge` must also carry poison when either side is empty — an empty accumulator merging a
  poisoned one becomes poisoned, and a poisoned one merging an empty one stays poisoned.
- Reproducibility is now claimed for non-finite inputs too; the design doc's "only for finite
  inputs" caveat is replaced by the table in `docs/design/deterministic-reduce.md`.

## oracle

```
cargo test -p es-math reduce
```

Written before the fix: the two new proptests fail on the old implementation.

## acceptance

- Proptest `poisoned_permutation_is_bit_identical`: over vectors drawn from `NaN`, `±Inf`,
  `±f64::MIN_POSITIVE`, the smallest subnormals and normal values, `finish().to_bits()` is
  equal for the original order and for a shuffle.
- Proptest `poisoned_split_merge_tree_is_bit_identical`: the same vectors reduced through an
  arbitrary split-merge tree give the same bits as a linear reduction.
- `non_finite_poisons_whatever_the_order`: both orders of `[1e-300, NAN]` are `NaN`;
  `[+Inf, -Inf]` in either order is `NaN`; `[1.0, +Inf]` is `+Inf`; `[1e300, -Inf]` is `-Inf`.
- `poison_survives_a_merge_from_either_side`, including merging into a default accumulator.
- The pre-existing finite properties (permutation, split-merge, commutativity, associativity,
  accuracy) still pass unchanged.

## forbidden

Any file outside `context`. Changing the `DeterministicAcc` trait signature or adding an
extension point (INV-17). Adding a `KahanAcc`. Heap allocation in the accumulator. Making
`finish` return a `Result`. Touching `crates/es-math/src/approx.rs`, `scalar.rs` or `simd.rs`.
