# Graph canonicalization for `canonical_hash` — design

Spec refs: spec 5.3 (hash chain), spec 11.2 (`*_hash`), Appendix B.7 (relabelling invariant).
Packet: `docs/packets/M0/P-M0-R2.md`. Code: `crates/es-ir/src/hash.rs`.

## The problem

`canonical_hash` must be a *semantic* identity: two graphs get the same hash exactly when they
are the same graph up to node renaming. The original implementation was colour refinement
alone (1-dimensional Weisfeiler-Leman): colour each node by kind + params, then repeatedly
recolour it by the sorted multiset of `(direction, near port, neighbour colour, far port)` over
its edges until no cell splits; encode the sorted multisets of node colours and of
colour-labelled edges; hash that.

1-WL is a sound *invariant* — isomorphic graphs always agree — but it is provably incomplete.
The M0 review's counterexample is in `hash::tests::eight_cycle_differs_from_two_four_cycles`:
four identical `Const` and four identical `Add` (ports `a`/`b`) wired as one 8-cycle, versus the
same eight nodes wired as two disjoint 4-cycles. Every node is regular in both directions in
both graphs, so refinement never splits a cell and the two encodings are byte-identical. Two
different task graphs would have shared one `task_hash`, i.e. a silent reproducibility bug.

## The algorithm: individualization-refinement

`canonical_hash` now computes a canonical *form*, not just an invariant:

1. Refine to the 1-WL fixpoint (`refine_from`).
2. If the colouring is discrete (every colour unique), encode and hash it.
3. Otherwise pick the **ambiguous class**: the smallest colour class with more than one member,
   ties broken by the colour value itself (`ambiguous_class`). Both criteria are properties of
   the colouring, never of `NodeId`, so every relabelling of a graph picks the same class.
4. For **each** member of that class in turn: recolour just that node
   (`distinguish` = `H(INDIV_TAG || colour)`), re-refine, and recurse.
5. Take the lexicographically smallest encoding over all branches.

Trying every member and taking the minimum is what keeps the result independent of which node
happened to be chosen, so Appendix B.7 still holds: isomorphic graphs enumerate corresponding
branch sets and therefore agree on the minimum. Node ids are still never written into the
encoding — they only index the colour vector.

## The cap

`REFINE_CAP = 10_000` refinement passes per `canonical_hash` call, shared across the whole
recursion. A graph that exhausts it returns `Diagnostic` **`HASH-002` "graph too symmetric to
canonicalize"** instead of a hash. Refusing is the only safe failure: a wrong hash is a
reproducibility bug that surfaces months later, a diagnostic is a build error now.

## Complexity

One refinement pass is `O(n · |E| · n)` — a full edge scan per node per round (unchanged from
before; noted as a `[HUMAN]` item in the M0 review and still not worth fixing for authored
graphs). Refinement passes per call:

| Graph | Passes |
|---|---|
| Discrete after plain refinement (the common case: distinct kinds or params) | 1 |
| One ambiguous class of size `k`, discrete after one individualization | `1 + k` |
| Nested ambiguity, depth `d`, class sizes `k₁ … k_d` | `O(∏ kᵢ)` |

So the worst case is exponential, which is why the cap exists. In practice authored IR graphs
are a few hundred nodes and ambiguity is rare — a class only survives refinement when several
nodes are *structurally* interchangeable, which usually means they really are interchangeable.

## What "semantic identity" now guarantees

- **Sound (unconditional).** Isomorphic graphs — same structure, same kinds, same params, same
  port names, same boundary order, any `NodeId` assignment — always produce the same hash.
  Proved by the `hash_independent_of_node_ids` proptest.
- **Complete (conditional).** Non-isomorphic graphs produce different hashes whenever the
  search finishes inside `REFINE_CAP`, which is the guarantee a canonical form gives. Past the
  cap the answer is `HASH-002`, never a hash that might be wrong.
- **Still not claimed.** Nothing here says two graphs with the same hash compute the same
  thing: `f32` fields are widened to `f64` in the encoding (`CanonWriter::f32`), so a precision
  change that alters runtime behaviour does not invalidate a stored hash.
