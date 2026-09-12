//! Deterministic transcendentals (spec §3.2, `DET-010`).
//!
//! `GLSL.std.450` only bounds the ULP error, so the same shader gives different bits on
//! different vendors and drivers — which breaks the tier-1 bitwise guarantee of §3.5. These
//! functions are the single implementation both sides use: Rust here,
//! `crates/es-math/slang/approx.slang` on the GPU, sharing coefficients (`coeffs`) *and*
//! evaluation order.
//!
//! Rules, enforced by review and by the packet's `forbidden` list:
//! Horner form written out as one expression, no reassociation, no `mul_add` (an FMA would
//! change the result on hosts that have one), no lookup tables, no branching on device
//! properties.
//!
//! Accuracy, primary domains and coefficient provenance: `docs/design/transcendental.md`.

// The bit manipulation below is exponent-field surgery on IEEE-754 f32; the casts are
// intentional and cannot wrap for the masked values used here.
// Single-letter locals mirror the Slang source and the Cephes reference line for line;
// renaming them would break that correspondence.
#![allow(clippy::cast_possible_wrap, clippy::many_single_char_names)]

pub mod coeffs;
#[cfg(test)]
mod slang_mirror;

use coeffs::{
    ATAN_C0, ATAN_C1, ATAN_C2, ATAN_C3, COS_C0, COS_C1, COS_C2, EXP_C0, EXP_C1, EXP_C2, EXP_C3,
    EXP_C4, EXP_C5, LN2_HI, LN2_LO, LOG2E, LOG_C0, LOG_C1, LOG_C2, LOG_C3, LOG_C4, LOG_C5, LOG_C6,
    LOG_C7, LOG_C8, PI, PIO2, PIO2_1, PIO2_2, PIO2_3, PIO2_LO, PIO4, PIO4_LO, PI_LO, SIN_C0,
    SIN_C1, SIN_C2, SQRT_HALF, TAN_PIO8, TWO_OVER_PI,
};

/// `2^k` for `k` in `[-126, 127]`, built directly in the exponent field (exact).
#[inline]
fn pow2i(k: i32) -> f32 {
    f32::from_bits(((k + 127) as u32) << 23)
}

/// `y * 2^n`, split into two exact steps so `n` may leave the single-exponent range.
#[inline]
fn scale2(y: f32, n: i32) -> f32 {
    let n = n.clamp(-250, 250);
    let half = n / 2;
    y * pow2i(half) * pow2i(n - half)
}

/// `(m, e)` with `x == m * 2^e` and `m` in `[0.5, 1)`. `x` must be finite and positive.
#[inline]
fn frexp(x: f32) -> (f32, i32) {
    // Subnormals have no implicit leading bit; scale them into the normal range first.
    let (x, adj) = if x < f32::MIN_POSITIVE {
        (x * 16_777_216.0, -24)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 126 + adj;
    (f32::from_bits((bits & 0x807f_ffff) | 0x3f00_0000), e)
}

/// True for negative values *and* for `-0.0`, which `x < 0.0` misses.
#[inline]
fn is_neg(x: f32) -> bool {
    x.to_bits() >> 31 == 1
}

/// Cody-Waite reduction: `(r, quadrant)` with `x = r + quadrant * pi/2 (mod 2pi)`,
/// `|r| <= pi/4`.
#[inline]
fn reduce_quadrant(x: f32) -> (f32, i32) {
    let n = (x * TWO_OVER_PI + 0.5).floor();
    let r = ((x - n * PIO2_1) - n * PIO2_2) - n * PIO2_3;
    (r, n as i32 & 3)
}

/// `sin(r)` for `|r| <= pi/4`.
#[inline]
fn sin_kernel(r: f32) -> f32 {
    let z = r * r;
    ((SIN_C0 * z + SIN_C1) * z + SIN_C2) * z * r + r
}

/// `cos(r)` for `|r| <= pi/4`.
#[inline]
fn cos_kernel(r: f32) -> f32 {
    let z = r * r;
    ((COS_C0 * z + COS_C1) * z + COS_C2) * z * z - 0.5 * z + 1.0
}

/// Sine. Primary domain `|x| <= 1e3`; non-finite input gives `NaN`.
#[must_use]
pub fn sin(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (r, quadrant) = reduce_quadrant(x);
    match quadrant {
        0 => sin_kernel(r),
        1 => cos_kernel(r),
        2 => -sin_kernel(r),
        _ => -cos_kernel(r),
    }
}

/// Cosine. Primary domain `|x| <= 1e3`; non-finite input gives `NaN`.
#[must_use]
pub fn cos(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (r, quadrant) = reduce_quadrant(x);
    match quadrant {
        0 => cos_kernel(r),
        1 => -sin_kernel(r),
        2 => -cos_kernel(r),
        _ => sin_kernel(r),
    }
}

/// `e^x`. Primary domain `[-88, 88]`; saturates to `0` / `inf` outside the f32 range.
#[must_use]
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 88.72284 {
        return f32::INFINITY;
    }
    if x < -103.97208 {
        return 0.0;
    }
    let n = (LOG2E * x + 0.5).floor();
    let r = x - n * LN2_HI - n * LN2_LO;
    let z = r * r;
    let y = (((((EXP_C0 * r + EXP_C1) * r + EXP_C2) * r + EXP_C3) * r + EXP_C4) * r + EXP_C5) * z
        + r
        + 1.0;
    scale2(y, n as i32)
}

/// Natural logarithm. `ln(0) == -inf`, `ln(x < 0) == NaN`.
#[must_use]
pub fn ln(x: f32) -> f32 {
    if x.is_nan() || is_neg(x) {
        return if x == 0.0 {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        };
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    let (m, e) = frexp(x);
    // Centre the mantissa on 1 so the polynomial argument stays in [-0.29, 0.41].
    let (m, e) = if m < SQRT_HALF {
        (m + m - 1.0, e - 1)
    } else {
        (m - 1.0, e)
    };
    let z = m * m;
    let y = ((((((((LOG_C0 * m + LOG_C1) * m + LOG_C2) * m + LOG_C3) * m + LOG_C4) * m + LOG_C5)
        * m
        + LOG_C6)
        * m
        + LOG_C7)
        * m
        + LOG_C8)
        * m
        * z;
    let fe = e as f32;
    let y = y + LN2_LO * fe - 0.5 * z;
    m + y + LN2_HI * fe
}

/// `atan(t)` for `t` in `[0, 1]`.
#[inline]
fn atan_unit(t: f32) -> f32 {
    if t > TAN_PIO8 {
        let x = (t - 1.0) / (t + 1.0);
        let z = x * x;
        let p = (((ATAN_C0 * z + ATAN_C1) * z + ATAN_C2) * z + ATAN_C3) * z * x + x;
        PIO4 + (p + PIO4_LO)
    } else {
        let z = t * t;
        (((ATAN_C0 * z + ATAN_C1) * z + ATAN_C2) * z + ATAN_C3) * z * t + t
    }
}

/// Four-quadrant arctangent, in `(-pi, pi]`. `atan2(0, 0) == 0`.
#[must_use]
pub fn atan2(y: f32, x: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    let (ax, ay) = (x.abs(), y.abs());
    let mut a = if ax == 0.0 && ay == 0.0 {
        0.0
    } else if ax.is_infinite() && ay.is_infinite() {
        PIO4
    } else if ay <= ax {
        atan_unit(ay / ax)
    } else {
        (PIO2 - atan_unit(ax / ay)) + PIO2_LO
    };
    if is_neg(x) {
        a = (PI - a) + PI_LO;
    }
    if is_neg(y) {
        a = -a;
    }
    a
}

/// Square root — a single correctly-rounded IEEE-754 operation, identical everywhere.
#[must_use]
pub fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

/// Reciprocal square root: two correctly-rounded operations, no polynomial, no
/// vendor-specific fast-rsqrt instruction. `rsqrt(0) == inf`.
#[must_use]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    /// Monotone integer image of a `f32`, so ULP distance is a subtraction.
    fn ordered(x: f32) -> i64 {
        let b = x.to_bits();
        if b & 0x8000_0000 == 0 {
            i64::from(b)
        } else {
            -i64::from(b & 0x7fff_ffff)
        }
    }

    fn ulp(a: f32, b: f32) -> u64 {
        if a.is_nan() && b.is_nan() {
            return 0;
        }
        (ordered(a) - ordered(b)).unsigned_abs()
    }

    /// xorshift64*: deterministic, local, no global RNG (spec §3.4).
    struct Rng(u64);

    impl Rng {
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / 9_007_199_254_740_992.0
        }

        fn range(&mut self, lo: f64, hi: f64) -> f32 {
            (lo + (hi - lo) * self.unit()) as f32
        }
    }

    const POINTS: usize = 200_000;

    /// Dense linear sweep plus a pseudo-random sweep over `[lo, hi]`, referenced against the
    /// `f64` `std` implementation rounded to `f32`.
    fn sweep(
        name: &str,
        lo: f64,
        hi: f64,
        ours: impl Fn(f32) -> f32,
        reference: impl Fn(f64) -> f64,
    ) -> u64 {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut worst = 0;
        let mut worst_at = 0.0f32;
        for i in 0..POINTS {
            for x in [
                (lo + (hi - lo) * i as f64 / POINTS as f64) as f32,
                rng.range(lo, hi),
            ] {
                let d = ulp(ours(x), reference(f64::from(x)) as f32);
                if d > worst {
                    worst = d;
                    worst_at = x;
                }
            }
        }
        println!("approx::{name:<6} max {worst} ULP over [{lo}, {hi}] (worst at {worst_at:e})");
        worst
    }

    #[test]
    fn ulp_within_target() {
        assert!(sweep("sin", -1e3, 1e3, sin, f64::sin) <= 2);
        assert!(sweep("cos", -1e3, 1e3, cos, f64::cos) <= 2);
        assert!(sweep("sin", -6.5, 6.5, sin, f64::sin) <= 2);
        assert!(sweep("cos", -6.5, 6.5, cos, f64::cos) <= 2);
        assert!(sweep("exp", -88.0, 88.0, exp, f64::exp) <= 2);
        assert!(sweep("ln", 1e-30, 1e30, ln, f64::ln) <= 2);
        assert!(sweep("ln", 0.5, 2.0, ln, f64::ln) <= 2);
        assert!(sweep("sqrt", 0.0, 1e30, sqrt, f64::sqrt) <= 2);
        assert!(sweep("rsqrt", 1e-30, 1e30, rsqrt, |v| 1.0 / v.sqrt()) <= 2);
    }

    #[test]
    fn atan2_ulp_within_target() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut worst = 0;
        for i in 0..POINTS {
            // Dense sweep around the unit circle, plus random pairs across the magnitude range.
            let t = 2.0 * f64::from(PI) * i as f64 / POINTS as f64 - f64::from(PI);
            for (y, x) in [
                (t.sin() as f32, t.cos() as f32),
                (rng.range(-1e6, 1e6), rng.range(-1e6, 1e6)),
                (rng.range(-1e-6, 1e-6), rng.range(-1e6, 1e6)),
                (rng.range(-1e6, 1e6), rng.range(-1e-6, 1e-6)),
            ] {
                let want = f64::from(y).atan2(f64::from(x)) as f32;
                worst = worst.max(ulp(atan2(y, x), want));
            }
        }
        println!("approx::atan2  max {worst} ULP");
        assert!(worst <= 2, "atan2 max {worst} ULP");
    }

    #[test]
    fn specials() {
        assert!(sin(f32::NAN).is_nan());
        assert!(sin(f32::INFINITY).is_nan());
        assert_eq!(sin(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert!(exp(f32::NAN).is_nan());
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert_eq!(ln(-0.0), f32::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert!(ln(f32::NAN).is_nan());
        assert_eq!(ln(f32::INFINITY), f32::INFINITY);
        assert_eq!(atan2(0.0, 0.0), 0.0);
        assert_eq!(atan2(0.0, 1.0), 0.0);
        assert_eq!(atan2(1.0, 0.0), PIO2);
        assert_eq!(atan2(0.0, -1.0), PI);
        assert_eq!(sqrt(0.0), 0.0);
        assert_eq!(sqrt(4.0), 2.0);
        assert!(sqrt(-1.0).is_nan());
        assert_eq!(rsqrt(0.0), f32::INFINITY);
        assert_eq!(rsqrt(4.0), 0.5);
    }

    #[test]
    fn subnormal_ln() {
        let tiny = f32::from_bits(1);
        assert!(ulp(ln(tiny), f64::from(tiny).ln() as f32) <= 2);
    }
}
