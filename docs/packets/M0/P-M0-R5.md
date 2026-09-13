# P-M0-R5 — `BinnedAcc` term-count ceiling assertion

Spec: spec 18.4 (deterministic reduction), spec 3.4, Appendix D `DET-011`. Follow-up to the
M0 review (`docs/reviews/M0.md`, Nit on `reduce.rs:33-35`). Review class C. Builds on
P-M0-R1.

## context

```
crates/es-math/src/reduce.rs
docs/design/deterministic-reduce.md
docs/packets/M0/P-M0-R5.md
```

## spec

- The exactness argument holds only below a term count that was documented and never checked.
  The exact bound: a deposit adds less than `2^W` quanta to a bin (the slice is smaller than
  the bin's top boundary, `2^W` quanta), and a bin is exact while it holds under `2^53`
  quanta, so `MAX_TERMS = 2^(53 - W) = 2^27 = 134_217_728` at `W = 26`.
- `terms: u32` counts finite deposits. A zero and a non-finite input touch no bin and are not
  counted. The counter saturates rather than wrapping.
- `merge` **adds** the two counts: the budget is per reduction, not per accumulator, because
  merging adds bin contents. Addition is commutative and associative, so the counter cannot
  break the `DET-011` invariant, and the count is a function of the multiset, so the assert
  cannot fire for one permutation and stay quiet for another.
- `add` and `merge` both `debug_assert!(terms <= MAX_TERMS, "... term ceiling ...")`. Debug
  only: the counter must not cost anything in a release kernel. Past the ceiling bins round
  and order independence lapses, so the breach has to be loud in debug runs and tests.
- A `#[cfg(test)] set_terms_for_test` helper exists so the ceiling can be reached without
  performing `2^27` real deposits.

## oracle

```
cargo test -p es-math reduce
```

## acceptance

- `depositing_past_the_term_ceiling_trips_the_assert`: `#[should_panic(expected = "term
  ceiling")]`, seeds the counter at `MAX_TERMS` and calls `add` once.
- `merging_past_the_term_ceiling_trips_the_assert`: the same through `merge`, proving the
  budget is shared across accumulators.
- Both tests are `#[cfg(debug_assertions)]`, so a release test run does not fail them.
- `docs/design/deterministic-reduce.md` states the bound's derivation, not just its value.
- Every pre-existing `reduce` test still passes; no test is slowed down measurably.

## forbidden

Any file outside `context`. Turning the ceiling into a runtime `assert!` or a `Result`.
Changing `W`, `K` or the `DeterministicAcc` signature. Adding an extension point (INV-17).
Heap allocation in the accumulator.
