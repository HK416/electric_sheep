# P-M0-R2 — decide and enforce the `canonical_hash` completeness ceiling

Spec: spec 5.3 (hash chain), spec 11.2 (`*_hash`), Appendix B.7. Follow-up to the `[SHOULD]`
finding on `crates/es-ir/src/hash.rs` in `docs/reviews/M0.md`. Review class A.

## context

```
crates/es-ir/src/hash.rs
crates/es-ir-types/src/codes.rs
docs/design/hash-canonicalization.md
docs/packets/M0/P-M0-R2.md
```

## spec

The M0 review confirmed that `canonical_hash` was exactly a 1-WL invariant and collides on
eight identical nodes wired as one 8-cycle versus two disjoint 4-cycles. The decision is to
**lift the ceiling, not document it**: add individualization-refinement.

- `refine_from(g, order, pos, colour, budget)` — one colour-refinement run to the fixpoint,
  spending one unit of `budget`.
- `ambiguous_class(colour)` — the smallest colour class with more than one member, ties broken
  by the colour value. A property of the colouring, never of `NodeId`.
- `canonical_encoding(...)` — encode a discrete colouring directly; otherwise try every member
  of the ambiguous class in turn as the distinguished node (`distinguish` recolours it with a
  domain-separated digest), re-refine, recurse, and take the lexicographically smallest
  encoding. Taking the minimum over *all* choices is what preserves Appendix B.7.
- `REFINE_CAP = 10_000` refinement passes per call, shared across the recursion. Exhausting it
  returns the new `HASH-002` "graph too symmetric to canonicalize" rather than a hash that
  might be wrong.
- The module and `canonical_hash` doc comments state the conditional guarantee instead of an
  unqualified "semantic identity"; `docs/design/hash-canonicalization.md` carries the
  algorithm, the cap and the complexity table.

## oracle

```
cargo test -p es-ir --features testing --lib hash::
```

## acceptance

- `eight_cycle_differs_from_two_four_cycles` asserts both halves: the plain refinement
  encodings of the two graphs are **equal** (the fixture really is the 1-WL counterexample),
  and their `canonical_hash` values **differ**. Both graphs stay relabelling-invariant.
- `exhausted_refinement_budget_is_reported` drives `canonical_encoding` with a spent budget and
  asserts `HASH-002`.
- The existing `hash_independent_of_node_ids` and `param_change_changes_hash` proptests, and
  every other `hash::` test, still pass unchanged.
- `HASH-002` is in `codes::CODES` with a title and section, so `codes::` tests still pass.

## forbidden

Any file outside `context`. Changing `CANON_TAG` or the encoding of an already-discrete
colouring — stored hashes for unambiguous graphs must not move. Widening the completeness claim
beyond "whenever the search finishes inside the cap". Returning a hash when the cap is hit.
