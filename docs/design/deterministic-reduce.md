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

Non-finite inputs poison the accumulator (they are added straight into bin 0 and propagate
through `merge`/`finish`); reproducibility is only claimed for finite inputs.

## Oracle

`cargo test -p es-math reduce::` — proptest over random `f64` vectors asserts that the
`finish()` bits are identical for (a) the original order, (b) an arbitrary permutation,
(c) an arbitrary split-merge tree over the same elements. Bit equality of a CPU `BinnedAcc`
against a GPU subgroup reduction is **Target / Status: unverified** (needs the M2 Vulkan path).
