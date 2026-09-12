//! Scalar representations (spec §3.3) — `Scalar` is one of the seven allowed extension
//! points (INV-17). The set is closed: `f32` (`F32`), `f64` (`F64Native`) and [`DoubleF32`]
//! (`DoubleFloat`, for devices without native `shaderFloat64`). A fourth representation is a
//! spec change, so the trait is sealed.

use core::cmp::Ordering;
use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for super::DoubleF32 {}
}

/// The numeric representation a kernel is compiled for.
///
/// Deliberately small: no transcendentals (those are `f32`-only and live in
/// [`crate::approx`]), no numeric tower, no conversions beyond `f64`.
pub trait Scalar:
    sealed::Sealed
    + Copy
    + Debug
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Difference between `ONE` and the next larger representable value.
    const EPSILON: Self;

    /// Nearest representable value to `v`.
    fn from_f64(v: f64) -> Self;
    /// Value as `f64`, rounded once.
    fn to_f64(self) -> f64;
    /// Square root. Correctly rounded for `f32`/`f64`.
    #[must_use]
    fn sqrt(self) -> Self;
    /// Absolute value.
    #[must_use]
    fn abs(self) -> Self;
    /// Neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f32::EPSILON;

    fn from_f64(v: f64) -> Self {
        v as Self
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }
    fn abs(self) -> Self {
        Self::abs(self)
    }
    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f64::EPSILON;

    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }
    fn abs(self) -> Self {
        Self::abs(self)
    }
    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
}

// --- error-free transformations (Dekker 1971, Knuth 1969) ------------------------------

/// `(s, e)` with `s = fl(a + b)` and `a + b == s + e` exactly.
#[inline]
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// [`two_sum`] for the case `|a| >= |b|` (three operations instead of six).
#[inline]
fn quick_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    (s, b - (s - a))
}

/// `(p, e)` with `p = fl(a * b)` and `a * b == p + e` exactly.
///
/// This is the **only** `mul_add` in the crate, and it is required here: a single-rounding
/// fused multiply-add is what makes the product split exact. `f32::mul_add` is guaranteed
/// fused in Rust (software fallback where the hardware lacks FMA), so the result is the same
/// on every host — unlike an implicitly contracted `a * b + c`, which spec §3.4 forbids.
#[inline]
fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// A pair of `f32` representing `hi + lo` with `|lo| <= 0.5 * ulp(hi)` — roughly 48 bits of
/// significand.
///
/// The `DoubleFloat` row of the spec §3.3 precision table: GPU physics on devices whose
/// `shaderFloat64` is emulated (Intel Arc Alchemist) runs this instead of `f64`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DoubleF32 {
    hi: f32,
    lo: f32,
}

impl DoubleF32 {
    /// Exact: every `f32` is a `DoubleF32`.
    #[must_use]
    pub const fn from_f32(hi: f32) -> Self {
        Self { hi, lo: 0.0 }
    }

    /// High and low limb.
    #[must_use]
    pub const fn parts(self) -> (f32, f32) {
        (self.hi, self.lo)
    }

    /// Renormalise an unevaluated pair into the `|lo| <= 0.5 * ulp(hi)` invariant.
    #[inline]
    fn renorm(hi: f32, lo: f32) -> Self {
        let (hi, lo) = quick_two_sum(hi, lo);
        Self { hi, lo }
    }
}

impl Add for DoubleF32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (s1, s2) = two_sum(self.hi, rhs.hi);
        let (t1, t2) = two_sum(self.lo, rhs.lo);
        let (s1, s2) = quick_two_sum(s1, s2 + t1);
        Self::renorm(s1, s2 + t2)
    }
}

impl Sub for DoubleF32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Neg for DoubleF32 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Mul for DoubleF32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (p1, p2) = two_prod(self.hi, rhs.hi);
        Self::renorm(p1, p2 + (self.hi * rhs.lo + self.lo * rhs.hi))
    }
}

impl Div for DoubleF32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let q1 = self.hi / rhs.hi;
        if !q1.is_finite() {
            return Self::from_f32(q1);
        }
        let r = self - rhs * Self::from_f32(q1);
        let q2 = r.hi / rhs.hi;
        let r2 = r - rhs * Self::from_f32(q2);
        let q3 = r2.hi / rhs.hi;
        let (s1, s2) = quick_two_sum(q1, q2);
        Self::renorm(s1, s2 + q3)
    }
}

impl PartialEq for DoubleF32 {
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd for DoubleF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&other.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&other.lo),
            ord => Some(ord),
        }
    }
}

impl Scalar for DoubleF32 {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    // ~2^-47: the product of the two limbs' precisions.
    const EPSILON: Self = Self {
        hi: f32::EPSILON * f32::EPSILON * 0.5,
        lo: 0.0,
    };

    fn from_f64(v: f64) -> Self {
        let hi = v as f32;
        Self {
            hi,
            lo: (v - f64::from(hi)) as f32,
        }
    }

    fn to_f64(self) -> f64 {
        f64::from(self.hi) + f64::from(self.lo)
    }

    fn sqrt(self) -> Self {
        if self.hi <= 0.0 {
            // 0 -> 0, negative -> NaN, matching IEEE.
            return Self::from_f32(self.hi.sqrt());
        }
        // One Newton step on the double-float residual, as in Bailey's double-double sqrt.
        let inv = 1.0 / self.hi.sqrt();
        let ax = self.hi * inv;
        let (p, e) = two_prod(ax, ax);
        let residual = (self - Self { hi: p, lo: e }).hi;
        Self::renorm(ax, residual * inv * 0.5)
    }

    fn abs(self) -> Self {
        if self.hi < 0.0 {
            -self
        } else {
            self
        }
    }

    fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use proptest::prelude::*;

    /// Values with an exponent range that keeps every intermediate well inside `f32`.
    fn tame() -> impl Strategy<Value = f32> {
        (-1e6f32..1e6f32).prop_filter("finite", |v| v.is_finite())
    }

    proptest! {
        #[test]
        fn two_sum_is_error_free(a in tame(), b in tame()) {
            let (s, e) = two_sum(a, b);
            prop_assert_eq!(f64::from(s) + f64::from(e), f64::from(a) + f64::from(b));
        }

        #[test]
        fn two_prod_is_error_free(a in tame(), b in tame()) {
            let (p, e) = two_prod(a, b);
            prop_assert_eq!(f64::from(p) + f64::from(e), f64::from(a) * f64::from(b));
        }

        #[test]
        fn round_trip_keeps_48_bits(v in -1e30f64..1e30f64) {
            let back = DoubleF32::from_f64(v).to_f64();
            prop_assert!((back - v).abs() <= v.abs() * 2f64.powi(-47));
        }

        /// The whole point of the representation: one `f32` op loses ~2^-24, the pair ~2^-47.
        #[test]
        fn add_beats_f32(a in tame(), b in tame()) {
            let exact = f64::from(a) + f64::from(b);
            prop_assume!(exact != 0.0);
            let pair = (DoubleF32::from_f32(a) + DoubleF32::from_f32(b)).to_f64();
            prop_assert!((pair - exact).abs() <= (f64::from(a + b) - exact).abs());
        }

        #[test]
        fn mul_beats_f32(a in tame(), b in tame()) {
            let exact = f64::from(a) * f64::from(b);
            prop_assume!(exact != 0.0);
            let pair = (DoubleF32::from_f32(a) * DoubleF32::from_f32(b)).to_f64();
            prop_assert!((pair - exact).abs() <= (f64::from(a * b) - exact).abs());
        }

        #[test]
        fn div_then_mul_round_trips(a in tame(), b in tame()) {
            prop_assume!(b.abs() > 1e-3);
            let a = DoubleF32::from_f32(a);
            let b = DoubleF32::from_f32(b);
            let back = (a / b) * b;
            prop_assert!((back.to_f64() - a.to_f64()).abs() <= a.to_f64().abs() * 2f64.powi(-44));
        }

        #[test]
        fn sqrt_squared_round_trips(v in 1e-6f32..1e6f32) {
            let d = DoubleF32::from_f32(v);
            let r = d.sqrt();
            prop_assert!(((r * r).to_f64() - f64::from(v)).abs() <= f64::from(v) * 2f64.powi(-44));
        }

        #[test]
        fn ordering_matches_f64(a in tame(), b in tame()) {
            let (da, db) = (DoubleF32::from_f32(a), DoubleF32::from_f32(b));
            prop_assert_eq!(da.partial_cmp(&db), f64::from(a).partial_cmp(&f64::from(b)));
        }
    }

    #[test]
    fn consts_are_consistent() {
        assert_eq!(DoubleF32::ZERO.to_f64(), 0.0);
        assert_eq!(DoubleF32::ONE.to_f64(), 1.0);
        assert!(DoubleF32::EPSILON.to_f64() < f64::from(f32::EPSILON));
        assert!(DoubleF32::ONE.is_finite());
        assert_eq!(DoubleF32::from_f32(-2.0).abs(), DoubleF32::from_f32(2.0));
    }

    #[test]
    fn generic_over_scalar() {
        fn hypot<T: Scalar>(a: T, b: T) -> T {
            (a * a + b * b).sqrt()
        }
        assert_eq!(hypot(3.0f32, 4.0f32), 5.0);
        assert_eq!(hypot(3.0f64, 4.0f64), 5.0);
        assert_eq!(
            hypot(DoubleF32::from_f32(3.0), DoubleF32::from_f32(4.0)).to_f64(),
            5.0
        );
    }
}
