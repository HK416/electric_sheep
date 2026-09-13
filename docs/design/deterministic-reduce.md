# Deterministic reduction (`es_math::reduce`) — design

Spec refs: §18.4 (결정적 리덕션), §3.4 (결정적 실행 계약), Appendix B.8 (signature),
Appendix D `DET-011`.

## The invariant that matters

From §18.4: **`merge` must be associative and commutative.** If it is, the result does not
depend on subgroup order, workgroup merge order, or SM count, and the same reduction on CPU
and GPU produces the same bits. Everything below exists only to make that true.

`KahanAcc` is explicitly **not** provided: compensated summation is still order-dependent
(the compensation term depends on the arrival order), so it fails the invariant.

## Representation

Appendix B.8 pins the shape:

```rust
pub trait DeterministicAcc<T>: Default + Clone {
    fn add(&mut self, v: T);
    fn merge(&mut self, other: &Self);
    fn finish(&self) -> T;
}
pub struct BinnedAcc<const K: usize = 3> { bins: [f64; K], index: Option<i32> }
```

The implementation carries two more private, `Copy` fields — `poison: u8` and `terms: u32`
(both below) — which the public signature does not show. The layout stays allocation-free.

`index` is the bin index of the largest magnitude deposited so far; `None` means empty.
Bin `j` of an accumulator with index `i` covers the exponent window

```
[ 2^(-W·(i+j+1)) , 2^(-W·(i+j)) )        W = 26 bits per bin, K = 3 bins (spec default)
```

so all bin boundaries of every accumulator lie on one global grid `{ 2^(-W·n) }`. That is the
property that makes rescaling exact (below).

## Deposit — exact, no rounding

Binned/RFA summation in the Demmel–Nguyen sense, with the slicing done by **truncation on the
significand** rather than by the `1.5·2^e` splitting trick:

```
add(x):  grow index so that |x| < 2^(-W·index)
         r = x
         for j in 0..K:
             part    = truncate r at exponent (-W·(index+j+1))   // clear low bits, toward zero
             bins[j] += part
             r       -= part                                      // exact: part is a bit-prefix of r
```

Both operations are exact:

- `part` is `r` with low significand bits cleared, so `r - part` is exactly representable —
  it is the discarded suffix of `r`'s significand.
- `bins[j] += part` is exact because every value ever added to bin `j` is a multiple of
  `2^(-W·(index+j+1))` and smaller than `2^(-W·(index+j))`; the sum stays a multiple of the
  same quantum, and a `f64` represents it exactly while `|bin| < 2^(53-W)` quanta.

Exact addition is trivially associative and commutative, so the invariant holds by
construction rather than by a rounding-error argument. Truncation is toward zero and depends
only on `x` and the index, never on the accumulator's current contents — so no
round-to-nearest tie ever depends on arrival order.

**The W trade-off.** `W` buys precision and spends term budget:

| W | bits guaranteed below the largest term (`W·(K-1)`) | terms before a bin can round (`2^(53-W)`) |
|---|---|---|
| 20 | 40 | 8.6·10^9 |
| **26** | **52** — what `f64` itself carries | **1.3·10^8** |
| 32 | 64 | 2.1·10^6 |

`W = 26` is the chosen point: full `f64` significand coverage, and a ceiling of ~1.3·10^8
terms. That ceiling is on the *whole* reduction, merges included — `merge` adds bin
contents, so splitting the work across accumulators does not buy more budget. Past it the
exactness argument lapses and bins start to round; a reduction that large needs a larger `K`
(and a proportionally smaller `W`), which is why `K` is a const parameter.

### The ceiling is checked, not just documented

The exact bound: one deposit adds **less than `2^W` quanta** to a bin (the slice `part` is
smaller than the bin's top boundary `2^(-W·(i+j))`, which is `2^W` times that bin's quantum
`2^(-W·(i+j+1))`), and a bin holds an exact integer count of quanta only while that count is
under `2^53`. So

```
MAX_TERMS = 2^53 / 2^W = 2^(53 - W) = 2^27 = 134_217_728      (W = 26)
```

`terms: u32` counts finite deposits (a zero or a non-finite input touches no bin, so neither
is counted), saturating rather than wrapping, and `merge` **adds** the two counts — the
budget is per reduction, not per accumulator. Both `add` and `merge` then
`debug_assert!(terms <= MAX_TERMS)`. It is a `debug_assert` on purpose — the counter would be
pure overhead in a release kernel — so the guarantee it buys is that a reduction that large
is caught in a debug run or a test, not that release builds refuse it. Past the ceiling bins
round and order independence lapses, which is exactly why the breach must be loud somewhere.
The count is itself order-independent (it is the multiset size), so the assert cannot fire
for one permutation and stay quiet for another.

## Rescale and merge

Rescaling to a smaller index (a larger value arrived) shifts bins down by `d` positions and
drops what falls off the bottom. Because all boundaries sit on the common grid, this is
exactly the same as having truncated every earlier input at the *new, coarser* boundary from
the start — so the result depends only on the final index, not on the order in which the
magnitudes arrived.

`merge` rescales both sides to `min(index)` and adds the bins element-wise: associative and
commutative because `+` on the aligned bins is exact.

## finish

`finish` sums the bins from the smallest window to the largest, in that fixed order. This is
the only rounding in the whole algorithm, and it is a pure function of the bin contents, so it
is reproducible.

## What is lost

The `K` bins span `W·K = 78` binary orders of magnitude below the top boundary. Because the
index quantises in steps of `W`, the guaranteed coverage below the largest *term* is
`W·(K-1) = 52` bits; contributions below that are truncated away. That is the deliberate
trade of binned summation — it buys order-independence, and `K` is the knob
(`BinnedAcc<4>` buys another 26 orders of magnitude for one more `f64` per accumulator).

Truncation is toward zero, so a long sum of same-sign values is biased slightly toward zero
rather than being unbiased. Deterministic rounding-to-nearest would remove the bias but
reintroduce an order-dependent tie-break, so it is not used.

## Non-finite inputs — poison bits, not a bin

A non-finite input never reaches the bins. It sets a bit in `poison: u8`
(`+Inf` / `-Inf` / `NaN`) **before** any bin arithmetic, and `add` returns.

That placement is the whole point. Depositing a non-finite value into bin 0 and then
rescaling (what the first implementation did) loses it whenever the accumulator's index is
already coarser than 0: `rescale_to` shifts the bins down, and `NaN` falls off the bottom —
so `add(1e-300); add(NAN)` finished as `0.0` while `add(NAN); add(1e-300)` finished as `NaN`.
Order-dependent, i.e. a direct breach of the `DET-011` contract this module exists to keep.
A flag outside the bin array cannot be shifted away.

`merge` combines the states with `|`: commutative, associative and idempotent, so the
non-finite path inherits the same invariant as the bins with no extra argument.

`finish` reports the poison before it sums, matching what IEEE `sum` would have produced:

| state | `finish` |
|---|---|
| `NaN` seen | `NaN` |
| both `+Inf` and `-Inf` seen | `NaN` (`Inf + (-Inf)`) |
| one infinity only | that infinity, whatever the finite terms are |
| no poison | the bin sum |

`NaN` is returned as the `f64::NAN` constant, so `finish().to_bits()` is identical across
permutations — poisoned reductions are bit-reproducible, not merely "some NaN".

## Oracle

`cargo test -p es-math reduce::` — proptest over random `f64` vectors asserts that the
`finish()` bits are identical for (a) the original order, (b) an arbitrary permutation,
(c) an arbitrary split-merge tree over the same elements. A second pair of proptests runs the
same two properties over vectors that mix `NaN` / `±Inf` with subnormals and normals, and a
debug-only `#[should_panic]` test deposits past `MAX_TERMS`. Bit equality of a CPU `BinnedAcc`
against a GPU subgroup reduction is **Target / Status: unverified** (needs the M2 Vulkan path).
