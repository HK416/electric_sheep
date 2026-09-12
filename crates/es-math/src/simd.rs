//! SIMD primitives (spec §2.2) with a determinism contract (spec §3.4).
//!
//! The contract is not throughput, it is association order: [`dot`] keeps [`LANES`]
//! independent lane accumulators, folds the tail into the lanes in order, then reduces the
//! lanes in a fixed sequence. A scalar implementation that does the same thing gets the same
//! bits, which is what the tests assert.
//!
//! There is no per-ISA function cloning (no `multiversion`): `wide`'s `*` and `+` are plain
//! IEEE operations with no contraction, so an AVX2 clone would produce the same bits as the
//! SSE2 one. [`simd_level`] therefore only *reports* what the host supports — it feeds
//! `hardware_capability` in the §5.3 execution hash.

use wide::f64x4;

/// The crate-wide vector width for `f64` work.
pub type F64xN = f64x4;

/// Lanes in [`F64xN`].
pub const LANES: usize = 4;

/// Host SIMD capability, for the `hardware_capability` term of the execution hash (§5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    Scalar,
    Sse2,
    Avx,
    Avx2,
    Neon,
}

/// Detect the host's widest supported instruction set. Reporting only — it must never select
/// a different algorithm, or the §3.4 "fixed algorithm" clause is broken.
#[must_use]
pub fn simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else if std::is_x86_feature_detected!("avx") {
            SimdLevel::Avx
        } else {
            SimdLevel::Sse2
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        SimdLevel::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdLevel::Scalar
    }
}

fn vector(chunk: &[f64]) -> F64xN {
    F64xN::from(<[f64; LANES]>::try_from(chunk).expect("chunks_exact yields LANES elements"))
}

/// Inner product. Bit-identical to the scalar reduction with the same association order.
///
/// # Panics
/// If the slices have different lengths.
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut acc = F64xN::ZERO;
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        acc += vector(x) * vector(y);
    }
    let mut lanes = acc.to_array();
    for (i, (x, y)) in ca.remainder().iter().zip(cb.remainder()).enumerate() {
        lanes[i] += x * y;
    }
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// `y += alpha * x`, elementwise.
///
/// # Panics
/// If the slices have different lengths.
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "axpy: length mismatch");
    let va = F64xN::splat(alpha);
    let mut cx = x.chunks_exact(LANES);
    let mut cy = y.chunks_exact_mut(LANES);
    for (sx, sy) in cx.by_ref().zip(cy.by_ref()) {
        let updated = vector(sy) + va * vector(sx);
        sy.copy_from_slice(&updated.to_array());
    }
    for (xi, yi) in cx.remainder().iter().zip(cy.into_remainder()) {
        *yi += alpha * xi;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use proptest::prelude::*;

    /// The association order [`dot`] must reproduce, written out scalar.
    fn dot_reference(a: &[f64], b: &[f64]) -> f64 {
        let mut lanes = [0.0f64; LANES];
        let full = a.len() - a.len() % LANES;
        for i in (0..full).step_by(LANES) {
            for (l, lane) in lanes.iter_mut().enumerate() {
                *lane += a[i + l] * b[i + l];
            }
        }
        for (l, i) in (full..a.len()).enumerate() {
            lanes[l] += a[i] * b[i];
        }
        lanes[0] + lanes[1] + lanes[2] + lanes[3]
    }

    fn axpy_reference(alpha: f64, x: &[f64], y: &mut [f64]) {
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi += alpha * xi;
        }
    }

    fn slice() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-1e6..1e6f64, 0..512)
    }

    proptest! {
        #[test]
        fn dot_matches_scalar_bitwise(a in slice()) {
            let b: Vec<f64> = a.iter().rev().map(|v| v * 0.5 + 1.0).collect();
            prop_assert_eq!(dot(&a, &b).to_bits(), dot_reference(&a, &b).to_bits());
        }

        #[test]
        fn axpy_matches_scalar_bitwise(x in slice(), alpha in -10.0..10.0f64) {
            let y0: Vec<f64> = x.iter().map(|v| v * -0.25).collect();
            let mut got = y0.clone();
            let mut want = y0;
            axpy(alpha, &x, &mut got);
            axpy_reference(alpha, &x, &mut want);
            for (g, w) in got.iter().zip(&want) {
                prop_assert_eq!(g.to_bits(), w.to_bits());
            }
        }
    }

    #[test]
    fn every_tail_length() {
        // The tail is where a lane-parallel reduction usually diverges from the scalar one.
        for n in 0..=(3 * LANES + 1) {
            let a: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 3.0)).collect();
            let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 + 1.0).collect();
            assert_eq!(
                dot(&a, &b).to_bits(),
                dot_reference(&a, &b).to_bits(),
                "n = {n}"
            );
        }
    }

    #[test]
    fn empty_and_trivial() {
        assert_eq!(dot(&[], &[]), 0.0);
        let mut y = vec![1.0, 2.0, 3.0];
        axpy(2.0, &[1.0, 1.0, 1.0], &mut y);
        assert_eq!(y, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn dot_rejects_length_mismatch() {
        let _ = dot(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    fn level_is_reported() {
        assert_ne!(format!("{:?}", simd_level()), "");
    }
}
