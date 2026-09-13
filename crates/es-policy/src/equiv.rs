//! Tier-4 policy equivalence (spec 8.9, spec 3.5): does the same Learning IR produce the same
//! action under two different `PolicyRuntime`s?
//!
//! The comparison is over the action chunk `[H, action_dim]` in the graph's **output** space,
//! after the unnormalizer, because that is the tensor the Safety Plane and the actuator see
//! (spec 9). Comparing normalized features would pass on a transposed unnormalizer.

use es_compile::Tensor;
use es_ir::types::ElemType;

/// How far apart two runtimes may be. The constants are spec 8.9's table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    pub abs: f64,
    /// Relative to the **second** argument of [`compare_actions`], which is the reference.
    pub rel: f64,
    /// Also require identical bit patterns, so `-0.0` against `+0.0` fails.
    pub bitwise: bool,
}

impl Tolerance {
    /// `PyTorch`(fp32) against ONNX(fp32): max absolute action error <= 1e-5.
    pub const TIER4_FP32: Self = Self {
        abs: 1e-5,
        rel: 1e-5,
        bitwise: false,
    };
    /// `PyTorch`(fp32) against Vulkan(fp32), small policies only: <= 1e-4.
    pub const TIER4_VULKAN: Self = Self {
        abs: 1e-4,
        rel: 1e-4,
        bitwise: false,
    };
    /// The same runtime re-run with deterministic kernels: bitwise.
    pub const BITWISE: Self = Self {
        abs: 0.0,
        rel: 0.0,
        bitwise: true,
    };
}

/// What [`compare_actions`] measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equivalence {
    pub max_abs: f64,
    pub max_rel: f64,
    pub pass: bool,
}

impl Equivalence {
    /// Shape, dtype or length disagreement. Not "a large error": a harness that broadcasts is
    /// a harness that passes when it should not.
    fn incomparable() -> Self {
        Self {
            max_abs: f64::INFINITY,
            max_rel: f64::INFINITY,
            pass: false,
        }
    }
}

/// Compare two action chunks. `b` is the reference side.
pub fn compare_actions(a: &Tensor, b: &Tensor, tol: Tolerance) -> Equivalence {
    if a.shape != b.shape || a.dtype != ElemType::F32 || b.dtype != ElemType::F32 {
        return Equivalence::incomparable();
    }
    let (Some(xs), Some(ys)) = (as_f32(a), as_f32(b)) else {
        return Equivalence::incomparable();
    };
    if xs.len() != ys.len() {
        return Equivalence::incomparable();
    }

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut pass = true;
    for (x, y) in xs.iter().zip(&ys) {
        // A NaN on either side is never equivalent, whatever the tolerance says.
        if x.is_nan() || y.is_nan() {
            return Equivalence::incomparable();
        }
        let d = f64::from(*x - *y).abs();
        let r = if *y == 0.0 {
            0.0
        } else {
            d / f64::from(*y).abs()
        };
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(r);
        if !(d <= tol.abs || d <= tol.rel * f64::from(*y).abs()) {
            pass = false;
        }
        if tol.bitwise && x.to_bits() != y.to_bits() {
            pass = false;
        }
    }
    Equivalence {
        max_abs,
        max_rel,
        pass,
    }
}

/// The f32 view of a tensor's bytes, or `None` if the payload is not a whole number of them.
fn as_f32(t: &Tensor) -> Option<Vec<f32>> {
    if t.data.len() % 4 != 0 {
        return None;
    }
    Some(
        t.data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Build an `[H, action_dim]` f32 chunk — the shape every tier-4 comparison is over.
pub fn action_chunk(horizon: u64, action_dim: u64, values: &[f32]) -> Tensor {
    Tensor {
        dtype: ElemType::F32,
        shape: vec![horizon, action_dim],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_chunks_pass_every_tolerance() {
        let a = action_chunk(2, 2, &[0.1, -0.2, 3.0, 4.5]);
        for tol in [
            Tolerance::TIER4_FP32,
            Tolerance::TIER4_VULKAN,
            Tolerance::BITWISE,
        ] {
            let e = compare_actions(&a, &a, tol);
            assert!(e.pass && e.max_abs == 0.0, "{e:?}");
        }
    }

    #[test]
    fn the_spec_8_9_threshold_is_where_it_says() {
        let b = action_chunk(1, 1, &[1.0]);
        // 5e-6 is inside the fp32 tier, 1e-3 is outside both the absolute and relative terms.
        assert!(compare_actions(&action_chunk(1, 1, &[1.000_005]), &b, Tolerance::TIER4_FP32).pass);
        assert!(!compare_actions(&action_chunk(1, 1, &[1.001]), &b, Tolerance::TIER4_FP32).pass);
        // The Vulkan tier is looser by an order of magnitude, as spec 8.9's table says: 5e-5
        // fails the fp32 tier and passes that one.
        let near = action_chunk(1, 1, &[1.000_05]);
        assert!(!compare_actions(&near, &b, Tolerance::TIER4_FP32).pass);
        assert!(compare_actions(&near, &b, Tolerance::TIER4_VULKAN).pass);
    }

    #[test]
    fn a_large_value_passes_on_the_relative_term() {
        let b = action_chunk(1, 1, &[1.0e6]);
        let a = action_chunk(1, 1, &[1.000_000_5e6]); // 0.5 absolute, 5e-7 relative
        let e = compare_actions(&a, &b, Tolerance::TIER4_FP32);
        assert!(e.pass && e.max_abs > 0.1, "{e:?}");
    }

    #[test]
    fn signed_zero_passes_the_value_tiers_and_fails_bitwise() {
        let a = action_chunk(1, 1, &[-0.0]);
        let b = action_chunk(1, 1, &[0.0]);
        assert!(compare_actions(&a, &b, Tolerance::TIER4_FP32).pass);
        assert!(!compare_actions(&a, &b, Tolerance::BITWISE).pass);
    }

    #[test]
    fn shape_dtype_and_nan_are_incomparable_not_merely_wrong() {
        let a = action_chunk(2, 2, &[0.0; 4]);
        for other in [
            action_chunk(4, 1, &[0.0; 4]),
            action_chunk(2, 2, &[f32::NAN, 0.0, 0.0, 0.0]),
            Tensor {
                dtype: ElemType::F64,
                shape: vec![2, 2],
                data: vec![0u8; 32],
            },
        ] {
            let e = compare_actions(&a, &other, Tolerance::TIER4_FP32);
            assert!(!e.pass && e.max_abs.is_infinite(), "{e:?}");
        }
    }
}
