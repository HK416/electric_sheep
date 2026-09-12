# Job partitioning (P14)

Spec §3.4 lists *deterministic scheduling* — static partitioning, single queue — as one of the
eight conditions of the deterministic execution contract, and §18.4 requires reductions whose
result does not depend on how work was split across hardware.

## Rule

`partition_static(n_items, n_workers)` is a pure function of its two arguments.

```
base = n_items / n_workers
rem  = n_items % n_workers
chunk i gets base + 1 items for i < rem, else base items
```

Ranges are contiguous, ascending, non-overlapping, and cover `0..n_items` exactly once. Empty
ranges are dropped, so the chunk count is `min(n_items, n_workers)`.

The extra item goes to the *low* chunks, not the high ones — an arbitrary choice, fixed here so
that a replay on another machine partitions identically.

## What is forbidden

- Deriving `n_workers` from the machine (`available_parallelism`, CPU count, GPU SM count).
  The caller passes it in; a replay reproduces it from the run record.
- Work stealing, dynamic chunking, or any partition that depends on how fast a worker got
  through its items.
- Merging reduction results in completion order. `par_reduce` collects per-chunk results into a
  slot indexed by chunk and folds them with `merge` in chunk-index order, so the result is
  identical for a given `n_workers` regardless of thread scheduling.

## Worker count and the result

Chunk boundaries depend on `n_workers`, so a result is invariant across worker counts only when
`merge` is associative (§18.4: the key invariant is that `merge` is associative and
commutative). Floating point sums are not; `es-math`'s `DeterministicAcc` (RFA / binned
summation, INV-17) is the accumulator that makes them so. `es-core` stays generic over the
merge closure and does not know about floats.

## Ceiling

`JobSystem` spawns scoped threads per call (`std::thread::scope`) rather than keeping a pool
alive. Spawn cost is irrelevant at the chunk sizes here; if per-call spawn ever shows up in a
profile, a persistent pool can be swapped in behind the same API without changing the
partitioning contract.
