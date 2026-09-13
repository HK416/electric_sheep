//! Reproducible floating-point reduction (spec §18.4, Appendix B.8, `DET-011`).
//!
//! The invariant that matters is that [`DeterministicAcc::merge`] is associative and
//! commutative: then subgroup order, workgroup merge order and SM count cannot change the
//! result, and a CPU reduction matches a GPU one. [`BinnedAcc`] gets there by making every
//! deposit *exact* — values are sliced by truncation onto a global exponent grid, and exact
//! addition is trivially associative and commutative.
//!
//! There is deliberately no `KahanAcc`: compensated summation is order-dependent (the
//! compensation term depends on arrival order) and fails the invariant.
//!
//! Full argument, the term-count ceiling and the truncation trade: `docs/design/deterministic-reduce.md`.

// Exponent-field surgery on IEEE-754 f64; the casts are intentional and cannot wrap for the
// masked values used here.
#![allow(clippy::cast_possible_wrap)]

/// An accumulator whose result does not depend on the order values arrived in.
///
/// One of the seven allowed extension points (INV-17).
pub trait DeterministicAcc<T>: Default + Clone {
    /// Deposit one value.
    fn add(&mut self, v: T);
    /// Combine with another accumulator. **Must be associative and commutative.**
    fn merge(&mut self, other: &Self);
    /// Reduce to a single value. Pure function of the accumulator state.
    fn finish(&self) -> T;
}

/// Bits of exponent range per bin.
///
/// `W * (K - 1) = 52` bits of guaranteed precision below the largest term (about what `f64`
/// itself carries) and `2^(53 - W)` = ~1.3e8 terms per accumulator before the exactness
/// argument lapses. Raising `K` buys range; lowering `W` buys term budget.
const W: i32 = 26;

/// Deposits before the exactness argument lapses.
///
/// One deposit adds less than `2^W` quanta to a bin (a slice is smaller than the bin's top
/// boundary, which is `2^W` times its quantum), and a bin stays exact while it holds fewer
/// than `2^53` quanta — so `2^53 / 2^W = 2^(53 - W)` deposits, `2^27` at `W = 26`. The
/// ceiling is on the whole reduction: `merge` adds bin contents *and* term counts, so
/// splitting the work across accumulators does not buy more budget.
const MAX_TERMS: u32 = 1 << (53 - W);

/// Poison bits. A non-finite input is recorded here, never in the bins, so a later rescale
/// cannot drop it and `merge` can combine two states with a commutative, idempotent `|`.
const POISON_POS_INF: u8 = 1;
const POISON_NEG_INF: u8 = 2;
const POISON_NAN: u8 = 4;

/// Binned / RFA accumulator in the Demmel–Nguyen sense (spec §18.4, signature from
/// Appendix B.8).
///
/// Bin `j` of an accumulator with index `i` holds the part of the sum in
/// `[2^(-W*(i+j+1)), 2^(-W*(i+j)))`. All bin boundaries of all accumulators lie on the one
/// grid `2^(-W*n)`, which is what makes rescaling and merging exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinnedAcc<const K: usize = 3> {
    bins: [f64; K],
    index: Option<i32>,
    /// Non-finite inputs seen so far, as `POISON_*` bits — outside the bin array on purpose.
    poison: u8,
    /// Deposits so far, for the [`MAX_TERMS`] ceiling. Saturating, so it cannot wrap.
    terms: u32,
}

impl<const K: usize> Default for BinnedAcc<K> {
    fn default() -> Self {
        Self {
            bins: [0.0; K],
            index: None,
            poison: 0,
            terms: 0,
        }
    }
}

/// `floor(log2(|x|))` for a finite non-zero `x`.
fn exponent_of(x: f64) -> i32 {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: the leading significand bit sets the exponent.
        let m = bits & 0x000f_ffff_ffff_ffff;
        63 - m.leading_zeros() as i32 - 1074
    } else {
        biased - 1023
    }
}

/// `x` with every significand bit below `2^e` cleared (truncation toward zero). Exact: the
/// discarded remainder `x - truncate_at(x, e)` is itself representable.
fn truncate_at(x: f64, e: i32) -> f64 {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    // Exponent of the least significant significand bit.
    let low = if biased == 0 { -1074 } else { biased - 1075 };
    let shift = e - low;
    if shift <= 0 {
        return x;
    }
    if shift > 52 {
        return 0.0;
    }
    f64::from_bits(bits & !((1u64 << shift) - 1))
}

/// The largest bin index whose top boundary still covers `|x|`.
fn index_for(x: f64) -> i32 {
    (-(exponent_of(x) + 1)).div_euclid(W)
}

impl<const K: usize> BinnedAcc<K> {
    /// Widen the covered range to `need` (a smaller index). Shifting is an exact bin move
    /// because every boundary sits on the shared `2^(-W*n)` grid, so the bins that fall off
    /// the bottom are exactly the bits a deposit at the new, coarser index would have
    /// truncated anyway.
    fn rescale_to(&mut self, need: i32) {
        match self.index {
            None => self.index = Some(need),
            Some(cur) if need < cur => {
                let d = (cur - need) as usize;
                let mut bins = [0.0; K];
                if d < K {
                    bins[d..].copy_from_slice(&self.bins[..K - d]);
                }
                self.bins = bins;
                self.index = Some(need);
            }
            Some(_) => {}
        }
    }

    /// Books `n` deposits against the exactness ceiling.
    fn count(&mut self, n: u32) {
        self.terms = self.terms.saturating_add(n);
        debug_assert!(
            self.terms <= MAX_TERMS,
            "BinnedAcc term ceiling exceeded: {} deposits > 2^{} (docs/design/deterministic-reduce.md)",
            self.terms,
            53 - W
        );
    }

    #[cfg(test)]
    fn set_terms_for_test(&mut self, terms: u32) {
        self.terms = terms;
    }
}

impl<const K: usize> DeterministicAcc<f64> for BinnedAcc<K> {
    fn add(&mut self, v: f64) {
        if v == 0.0 {
            return;
        }
        if !v.is_finite() {
            // Poison, recorded before any bin arithmetic so no rescale can lose it.
            self.poison |= if v.is_nan() {
                POISON_NAN
            } else if v > 0.0 {
                POISON_POS_INF
            } else {
                POISON_NEG_INF
            };
            return;
        }
        self.count(1);
        self.rescale_to(index_for(v));
        let index = self.index.unwrap_or(0);
        let mut r = v;
        for j in 0..K {
            let bottom = -W * (index + j as i32 + 1);
            let part = truncate_at(r, bottom);
            self.bins[j] += part;
            r -= part;
        }
    }

    fn merge(&mut self, other: &Self) {
        // Both are commutative and associative, and both must survive an empty other side.
        self.poison |= other.poison;
        self.count(other.terms);
        let Some(other_index) = other.index else {
            return;
        };
        let Some(index) = self.index else {
            self.bins = other.bins;
            self.index = other.index;
            return;
        };
        let mut other = *other;
        if other_index < index {
            self.rescale_to(other_index);
        } else {
            other.rescale_to(index);
        }
        for (bin, add) in self.bins.iter_mut().zip(other.bins) {
            *bin += add;
        }
    }

    fn finish(&self) -> f64 {
        // Poison first, matching IEEE `sum`: NaN dominates, +Inf with -Inf is NaN, and a lone
        // infinity dominates every finite term.
        if self.poison != 0 {
            let both = POISON_POS_INF | POISON_NEG_INF;
            if self.poison & POISON_NAN != 0 || self.poison & both == both {
                return f64::NAN;
            }
            return if self.poison & POISON_POS_INF != 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        // Fixed order, smallest bin first: the only rounding in the algorithm, and a pure
        // function of the bin contents.
        self.bins.iter().rev().sum()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use proptest::prelude::*;

    type Acc = BinnedAcc<3>;

    fn sum_in_order(values: &[f64]) -> u64 {
        let mut acc = Acc::default();
        for &v in values {
            acc.add(v);
        }
        acc.finish().to_bits()
    }

    /// Recursive split-merge over `values`, splitting at a seed-driven point.
    fn sum_by_tree(values: &[f64], seed: &mut u64) -> Acc {
        if values.len() <= 1 {
            let mut acc = Acc::default();
            for &v in values {
                acc.add(v);
            }
            return acc;
        }
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let cut = 1 + (*seed as usize) % (values.len() - 1);
        let (a, b) = values.split_at(cut);
        let mut left = sum_by_tree(a, seed);
        let right = sum_by_tree(b, seed);
        left.merge(&right);
        left
    }

    fn shuffled(values: &[f64], mut seed: u64) -> Vec<f64> {
        let mut out = values.to_vec();
        for i in (1..out.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            out.swap(i, (seed as usize) % (i + 1));
        }
        out
    }

    /// Mixed magnitudes, both signs — the case where naive summation is order-dependent.
    fn values() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(
            prop_oneof![-1.0..1.0f64, -1e6..1e6f64, -1e-6..1e-6f64, -1e12..1e12f64,],
            1..256,
        )
    }

    /// `NaN` / `±Inf` mixed with subnormals and normals: the poison case that used to be
    /// order-dependent (`P-M0-R1`).
    fn poisoned_values() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(
            prop_oneof![
                Just(f64::NAN),
                Just(f64::INFINITY),
                Just(f64::NEG_INFINITY),
                Just(f64::MIN_POSITIVE),
                Just(-f64::MIN_POSITIVE),
                // Smallest subnormal, both signs.
                Just(f64::from_bits(1)),
                Just(-f64::from_bits(1)),
                -1.0..1.0f64,
                -1e12..1e12f64,
            ],
            1..64,
        )
    }

    proptest! {
        #[test]
        fn permutation_is_bit_identical(v in values(), seed in 1u64..u64::MAX) {
            prop_assert_eq!(sum_in_order(&v), sum_in_order(&shuffled(&v, seed)));
        }

        #[test]
        fn split_merge_tree_is_bit_identical(v in values(), seed in 1u64..u64::MAX) {
            let mut s = seed;
            prop_assert_eq!(sum_in_order(&v), sum_by_tree(&v, &mut s).finish().to_bits());
        }

        #[test]
        fn merge_is_commutative(a in values(), b in values()) {
            let (mut x, mut y) = (Acc::default(), Acc::default());
            for &v in &a { x.add(v); }
            for &v in &b { y.add(v); }
            let (mut ab, mut ba) = (x, y);
            ab.merge(&y);
            ba.merge(&x);
            prop_assert_eq!(ab.finish().to_bits(), ba.finish().to_bits());
        }

        #[test]
        fn merge_is_associative(a in values(), b in values(), c in values()) {
            let acc = |vs: &[f64]| { let mut t = Acc::default(); for &v in vs { t.add(v); } t };
            let (x, y, z) = (acc(&a), acc(&b), acc(&c));

            let mut left = x;
            left.merge(&y);
            left.merge(&z);

            let mut yz = y;
            yz.merge(&z);
            let mut right = x;
            right.merge(&yz);

            prop_assert_eq!(left.finish().to_bits(), right.finish().to_bits());
        }

        /// Non-finite inputs must be as order-independent as finite ones: the poison state is
        /// what `finish` reports, and it is a function of the multiset, not of the order.
        #[test]
        fn poisoned_permutation_is_bit_identical(v in poisoned_values(), seed in 1u64..u64::MAX) {
            prop_assert_eq!(sum_in_order(&v), sum_in_order(&shuffled(&v, seed)));
        }

        #[test]
        fn poisoned_split_merge_tree_is_bit_identical(
            v in poisoned_values(),
            seed in 1u64..u64::MAX,
        ) {
            let mut s = seed;
            prop_assert_eq!(sum_in_order(&v), sum_by_tree(&v, &mut s).finish().to_bits());
        }

        /// Within the covered range the result must still be a good sum, not just a stable one.
        #[test]
        fn accuracy_tracks_a_plain_sum(v in prop::collection::vec(-1e6..1e6f64, 1..256)) {
            let magnitude: f64 = v.iter().map(|x| x.abs()).sum();
            let want: f64 = v.iter().sum();
            let got = { let mut a = Acc::default(); for &x in &v { a.add(x); } a.finish() };
            prop_assert!((got - want).abs() <= 1e-9 * magnitude, "{got} vs {want}");
        }
    }

    #[test]
    fn empty_and_single() {
        assert_eq!(Acc::default().finish(), 0.0);
        let mut acc = Acc::default();
        acc.add(0.0);
        assert_eq!(acc.finish(), 0.0);

        // A single value is recovered exactly: its 53 significand bits fit in the 78 bits of
        // range three W=26 bins cover.
        for v in [1.0, -1.0, 1e-300, 1e300, f64::MIN_POSITIVE, 0.1, 12345.678] {
            let mut acc = Acc::default();
            acc.add(v);
            assert_eq!(acc.finish(), v, "single add of {v} was not exact");
        }
    }

    #[test]
    fn merging_an_empty_accumulator_is_a_no_op() {
        let mut a = Acc::default();
        a.add(1.5);
        let before = a;
        a.merge(&Acc::default());
        assert_eq!(a, before);

        let mut empty = Acc::default();
        empty.merge(&before);
        assert_eq!(empty.finish(), 1.5);
    }

    #[test]
    fn out_of_range_terms_are_dropped_not_reordered() {
        // 2^-100 relative to 1.0 is below the covered range; it must vanish the same way
        // whichever order it arrives in.
        let big_first = sum_in_order(&[1.0, 2f64.powi(-100)]);
        let small_first = sum_in_order(&[2f64.powi(-100), 1.0]);
        assert_eq!(big_first, small_first);
        assert_eq!(f64::from_bits(big_first), 1.0);
    }

    #[test]
    fn non_finite_poisons_whatever_the_order() {
        let nan = |vs: &[f64]| f64::from_bits(sum_in_order(vs)).is_nan();
        let sum = |vs: &[f64]| f64::from_bits(sum_in_order(vs));

        assert!(nan(&[1.0, f64::NAN]) && nan(&[f64::NAN, 1.0]));
        // The regression: a tiny earlier value made the rescale swallow the poison.
        assert!(nan(&[1e-300, f64::NAN]) && nan(&[f64::NAN, 1e-300]));
        // IEEE `sum` semantics: Inf + (-Inf) is NaN, in either order.
        assert!(nan(&[f64::INFINITY, f64::NEG_INFINITY]));
        assert!(nan(&[f64::NEG_INFINITY, 1e-300, f64::INFINITY]));
        // NaN dominates a lone infinity; a lone infinity dominates the finite terms.
        assert!(nan(&[f64::INFINITY, f64::NAN]) && nan(&[f64::NAN, f64::INFINITY]));
        assert_eq!(sum(&[1.0, f64::INFINITY]), f64::INFINITY);
        assert_eq!(sum(&[f64::INFINITY, 1.0]), f64::INFINITY);
        assert_eq!(sum(&[1e300, f64::NEG_INFINITY]), f64::NEG_INFINITY);
    }

    #[test]
    fn poison_survives_a_merge_from_either_side() {
        let poisoned = {
            let mut a = Acc::default();
            a.add(f64::NAN);
            a
        };
        let mut finite = Acc::default();
        finite.add(1.0);

        let mut left = finite;
        left.merge(&poisoned);
        let mut right = poisoned;
        right.merge(&finite);
        assert!(left.finish().is_nan() && right.finish().is_nan());

        // An empty accumulator must pick the poison up too.
        let mut empty = Acc::default();
        empty.merge(&poisoned);
        assert!(empty.finish().is_nan());
    }

    /// The exactness ceiling is a `debug_assert`, so this only fires in a debug build.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "term ceiling")]
    fn depositing_past_the_term_ceiling_trips_the_assert() {
        let mut acc = Acc::default();
        // Cheap path to the ceiling: 2^27 real deposits would take minutes.
        acc.set_terms_for_test(MAX_TERMS);
        acc.add(1.0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "term ceiling")]
    fn merging_past_the_term_ceiling_trips_the_assert() {
        let mut a = Acc::default();
        let mut b = Acc::default();
        a.set_terms_for_test(MAX_TERMS);
        b.set_terms_for_test(1);
        a.merge(&b);
    }
}
