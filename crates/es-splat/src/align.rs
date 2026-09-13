//! Position and colour alignment (spec 16.2).
//!
//! Both are closed-form fits over correspondences someone else chose. ICP and RANSAC are not
//! here: spec 16.3's own standard is that exact alignment is unnecessary and distribution
//! match is the goal, and spec 1.9 puts this whole path third on the cut list.

use es_math::{Quat, Vec3};

use crate::{SplatError, SplatScene, SH_C0};

/// Cyclic Jacobi sweeps for the 4x4 symmetric eigenproblem. Fixed, not
/// convergence-dependent: a data-dependent iteration count is a data-dependent result.
/// Six rotations per sweep; a 4x4 is converged to `f64` well inside this many.
const JACOBI_SWEEPS: usize = 16;

/// A similarity transform: uniform scale, then rotation, then translation.
///
/// `T_robot_scan` of spec 16.2 — the map from the capture's arbitrary reconstruction frame
/// into the robot's frame. Scale is part of it because a 3DGS world has no metric: monocular
/// `SfM` fixes geometry only up to a scale factor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    pub scale: f64,
    pub rot: Quat,
    pub trans: Vec3,
}

impl Default for Similarity {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Similarity {
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        rot: Quat::IDENTITY,
        trans: Vec3::ZERO,
    };

    /// `scale * (rot * p) + trans`.
    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.rot.rotate(p).scale(self.scale) + self.trans
    }

    /// Fits the similarity taking `src` onto `dst` by Umeyama's closed form, with Horn's
    /// quaternion method for the rotation.
    ///
    /// Needs at least 3 correspondences: three non-collinear points fix a rotation, two do
    /// not. Degenerate input is an error, never a silent identity.
    ///
    /// The 4x4 symmetric eigenproblem is solved by a fixed-sweep cyclic Jacobi rotation,
    /// which needs `sqrt` and division and no transcendental at all — so there is neither a
    /// 3x3 SVD to hand-roll nor an `es_math::approx` dependency to justify — and its fixed
    /// operation order makes the fit reproducible without a determinism contract of its own.
    /// The quaternion parameterisation also cannot produce a reflection, which is the failure
    /// an SVD route needs a determinant correction for.
    pub fn fit(src: &[Vec3], dst: &[Vec3]) -> Result<Self, SplatError> {
        if src.len() != dst.len() {
            return Err(SplatError::PointCountMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        if src.len() < 3 {
            return Err(SplatError::TooFewPoints(src.len()));
        }
        for (i, (a, b)) in src.iter().zip(dst).enumerate() {
            let finite = |v: &Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
            if !finite(a) || !finite(b) {
                return Err(SplatError::NonFiniteSample(i));
            }
        }

        let n = src.len() as f64;
        let mean = |pts: &[Vec3]| {
            pts.iter()
                .fold(Vec3::ZERO, |acc, p| acc + *p)
                .scale(1.0 / n)
        };
        let (src_mean, dst_mean) = (mean(src), mean(dst));

        // `s[a][b] = sum_i src_centred[a] * dst_centred[b]`, accumulated in index order.
        let mut s = [[0.0f64; 3]; 3];
        let mut src_spread = 0.0f64;
        for (a, b) in src.iter().zip(dst) {
            let a = *a - src_mean;
            let b = *b - dst_mean;
            let (av, bv) = ([a.x, a.y, a.z], [b.x, b.y, b.z]);
            for i in 0..3 {
                for j in 0..3 {
                    s[i][j] += av[i] * bv[j];
                }
            }
            src_spread += a.dot(a);
        }
        if src_spread <= 0.0 {
            return Err(SplatError::DegenerateFit);
        }

        let rot = horn_rotation(&s);
        let mut num = 0.0f64;
        for (a, b) in src.iter().zip(dst) {
            num += (*b - dst_mean).dot(rot.rotate(*a - src_mean));
        }
        let scale = num / src_spread;
        Ok(Self {
            scale,
            rot,
            trans: dst_mean - rot.rotate(src_mean).scale(scale),
        })
    }

    /// Applies this transform to a whole capture: centres through [`Similarity::apply`],
    /// orientations pre-multiplied by `rot`, standard deviations multiplied by `scale`.
    pub fn apply_scene(&self, scene: &mut SplatScene) {
        for i in 0..scene.len() {
            let p = self.apply(scene.position(i));
            scene.positions[3 * i] = p.x as f32;
            scene.positions[3 * i + 1] = p.y as f32;
            scene.positions[3 * i + 2] = p.z as f32;

            let q = Quat::from_xyzw(
                f64::from(scene.rotations[4 * i]),
                f64::from(scene.rotations[4 * i + 1]),
                f64::from(scene.rotations[4 * i + 2]),
                f64::from(scene.rotations[4 * i + 3]),
            );
            let q = (self.rot * q).normalize();
            scene.rotations[4 * i] = q.x as f32;
            scene.rotations[4 * i + 1] = q.y as f32;
            scene.rotations[4 * i + 2] = q.z as f32;
            scene.rotations[4 * i + 3] = q.w as f32;

            for c in 0..3 {
                scene.scales[3 * i + c] = (f64::from(scene.scales[3 * i + c]) * self.scale) as f32;
            }
        }
        scene.recompute_bounds();
        scene.refresh_asset_hash();
    }
}

/// Horn's quaternion method: the eigenvector of the largest eigenvalue of the 4x4 symmetric
/// `n` built from the cross-covariance `s` is the rotation taking `src` onto `dst`.
fn horn_rotation(s: &[[f64; 3]; 3]) -> Quat {
    let (sxx, sxy, sxz) = (s[0][0], s[0][1], s[0][2]);
    let (syx, syy, syz) = (s[1][0], s[1][1], s[1][2]);
    let (szx, szy, szz) = (s[2][0], s[2][1], s[2][2]);
    // Ordered (w, x, y, z).
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let (values, vectors) = jacobi_eigen(n);
    let mut best = 0usize;
    for i in 1..4 {
        if values[i] > values[best] {
            best = i;
        }
    }
    Quat::from_xyzw(
        vectors[1][best],
        vectors[2][best],
        vectors[3][best],
        vectors[0][best],
    )
    .normalize()
}

/// Cyclic Jacobi eigen-decomposition of a symmetric 4x4. Returns `(eigenvalues, eigenvectors)`
/// with eigenvector `j` in column `j`.
fn jacobi_eigen(mut m: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    /// Mixes columns `p` and `q` of every row by the plane rotation `(cos, sin)`.
    fn mix_columns(rows: &mut [[f64; 4]], p: usize, q: usize, cos: f64, sin: f64) {
        for row in rows {
            let (a, b) = (row[p], row[q]);
            row[p] = cos * a - sin * b;
            row[q] = sin * a + cos * b;
        }
    }

    let mut vecs = [[0.0f64; 4]; 4];
    for (i, row) in vecs.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..JACOBI_SWEEPS {
        for p in 0..3 {
            for q in (p + 1)..4 {
                // An already-zero off-diagonal would make the rotation angle 0/0.
                if m[p][q].abs() <= f64::MIN_POSITIVE {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let root = (theta * theta + 1.0).sqrt();
                let tan = if theta >= 0.0 {
                    1.0 / (theta + root)
                } else {
                    -1.0 / (-theta + root)
                };
                let cos = 1.0 / (tan * tan + 1.0).sqrt();
                let sin = tan * cos;
                mix_columns(&mut m, p, q, cos, sin);
                // Rows `p` and `q`, which `mix_columns` cannot reach: two disjoint borrows.
                let (upper, lower) = m.split_at_mut(q);
                for (pk, qk) in upper[p].iter_mut().zip(lower[0].iter_mut()) {
                    let (a, b) = (*pk, *qk);
                    *pk = cos * a - sin * b;
                    *qk = sin * a + cos * b;
                }
                mix_columns(&mut vecs, p, q, cos, sin);
            }
        }
    }
    ([m[0][0], m[1][1], m[2][2], m[3][3]], vecs)
}

/// Per-channel affine colour map: `out_c = gain_c * in_c + bias_c`, in RGB space.
///
/// Spec 16.1 singles this step out — mapping the scan's colour space onto the robot's actual
/// camera is what put the policy back in-distribution. The spec calls for a polynomial map;
/// this is its first useful term, and adding gamma or a cross-channel matrix before there is
/// a real capture to fit against would be fitting to nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAffine {
    pub gain: [f64; 3],
    pub bias: [f64; 3],
}

impl Default for ColorAffine {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ColorAffine {
    pub const IDENTITY: Self = Self {
        gain: [1.0; 3],
        bias: [0.0; 3],
    };

    /// Ordinary least squares per channel, independently: `gain = cov / var`,
    /// `bias = mean(dst) - gain * mean(src)`.
    ///
    /// A channel with no variance in `src` degrades to a pure offset (`gain = 1`) rather
    /// than dividing by zero.
    pub fn fit(src: &[[f32; 3]], dst: &[[f32; 3]]) -> Result<Self, SplatError> {
        if src.len() != dst.len() {
            return Err(SplatError::PointCountMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        if src.len() < 2 {
            return Err(SplatError::TooFewColorSamples(src.len()));
        }
        for (i, (a, b)) in src.iter().zip(dst).enumerate() {
            if a.iter().chain(b).any(|v| !v.is_finite()) {
                return Err(SplatError::NonFiniteSample(i));
            }
        }

        let n = src.len() as f64;
        let mut out = Self::IDENTITY;
        for c in 0..3 {
            let sm = src.iter().map(|s| f64::from(s[c])).sum::<f64>() / n;
            let dm = dst.iter().map(|d| f64::from(d[c])).sum::<f64>() / n;
            let mut cov = 0.0f64;
            let mut var = 0.0f64;
            for (a, b) in src.iter().zip(dst) {
                let a = f64::from(a[c]) - sm;
                cov += a * (f64::from(b[c]) - dm);
                var += a * a;
            }
            out.gain[c] = if var > 0.0 { cov / var } else { 1.0 };
            out.bias[c] = dm - out.gain[c] * sm;
        }
        Ok(out)
    }

    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = (self.gain[c] * f64::from(rgb[c]) + self.bias[c]) as f32;
        }
        out
    }

    /// Samples the base colours of `scene` at `indices`, for fitting against a reference
    /// capture's samples at the same points.
    pub fn sample(scene: &SplatScene, indices: &[usize]) -> Vec<[f32; 3]> {
        indices
            .iter()
            .filter(|i| **i < scene.len())
            .map(|i| scene.rgb(*i))
            .collect()
    }
}

/// `rgb -> degree-0 SH coefficient`, the inverse of [`SplatScene::rgb`].
pub(crate) fn rgb_to_dc(rgb: f32) -> f32 {
    (rgb - 0.5) / SH_C0
}
