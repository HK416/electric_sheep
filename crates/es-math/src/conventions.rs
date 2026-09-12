//! Coordinate, unit and rigid-body conventions (spec §3.1), as types.
//!
//! Right-handed, **Z-up**, X-forward (ROS/MJCF/URDF). Lengths in m, angles in rad, mass in
//! kg, time in s — SI is the storage unit everywhere; conversions happen at I/O boundaries
//! only, with [`units`]. Quaternions are stored **xyzw**, unit norm, canonicalised to
//! `w >= 0`. Inertia is a body-frame symmetric 3x3 with a principal-axis decomposition cache.
//!
//! This is not a linear algebra library: it carries the conventions and nothing more.
//! Image and camera conventions (the last three rows of the §3.1 table) belong to
//! `ImageSpec` in `es-ir` (§7.2), not here.

use serde::{Deserialize, Serialize};

/// Unit conversions. SI is the storage unit; these exist for parsing and display only.
pub mod units {
    /// Degrees to radians.
    pub const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;
    /// Radians to degrees.
    pub const RAD_TO_DEG: f64 = 180.0 / core::f64::consts::PI;
}

/// A point or direction in a right-handed frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// The §3.1 world axes: Z-up, X-forward, right-handed (so +Y is left).
pub mod axis {
    use super::Vec3;

    /// +Z.
    pub const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    /// +X.
    pub const FORWARD: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    /// +Y. `FORWARD x LEFT == UP`.
    pub const LEFT: Vec3 = Vec3::new(0.0, 1.0, 0.0);
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    #[must_use]
    pub fn scale(self, k: f64) -> Self {
        Self::new(self.x * k, self.y * k, self.z * k)
    }

    #[must_use]
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Unit vector, or [`Vec3::ZERO`] if `self` has no direction.
    #[must_use]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n > 0.0 {
            self.scale(1.0 / n)
        } else {
            Self::ZERO
        }
    }
}

impl core::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl core::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl core::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Rotation as a unit quaternion, stored **xyzw** with `w >= 0` (spec §3.1).
///
/// `q` and `-q` are the same rotation; [`Quat::normalize`] picks the `w >= 0` representative
/// so that a hash of the quaternion is a hash of the rotation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    #[must_use]
    pub const fn from_xyzw(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    #[must_use]
    pub fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Unit norm and `w >= 0`. Maps `q` and `-q` to exactly the same bits, and is a fixed
    /// point to within a rounding (see the test). A zero-length or non-finite quaternion
    /// normalises to [`Quat::IDENTITY`].
    #[must_use]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n <= 0.0 || !n.is_finite() {
            return Self::IDENTITY;
        }
        // Divide rather than multiply by the reciprocal: one correctly-rounded operation
        // per component instead of two roundings. Negate the divisor rather than the
        // components, so `q` and `-q` land on exactly the same bits.
        let k = if self.w < 0.0 { -n } else { n };
        Self {
            x: self.x / k,
            y: self.y / k,
            z: self.z / k,
            w: self.w / k,
        }
    }

    /// Inverse of a unit quaternion.
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Rotate `v` by this (unit) quaternion.
    #[must_use]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let t = u.cross(v).scale(2.0);
        v + t.scale(self.w) + u.cross(t)
    }

    /// Columns of the rotation matrix, `[e0, e1, e2]`.
    fn from_columns(m: [[f64; 3]; 3]) -> Self {
        // Shepperd's method: pick the largest of the four candidates for numerical stability.
        let (m00, m11, m22) = (m[0][0], m[1][1], m[2][2]);
        let trace = m00 + m11 + m22;
        let q = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            [
                (m[1][2] - m[2][1]) / s,
                (m[2][0] - m[0][2]) / s,
                (m[0][1] - m[1][0]) / s,
                0.25 * s,
            ]
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            [
                0.25 * s,
                (m[1][0] + m[0][1]) / s,
                (m[2][0] + m[0][2]) / s,
                (m[1][2] - m[2][1]) / s,
            ]
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            [
                (m[1][0] + m[0][1]) / s,
                0.25 * s,
                (m[2][1] + m[1][2]) / s,
                (m[2][0] - m[0][2]) / s,
            ]
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            [
                (m[2][0] + m[0][2]) / s,
                (m[2][1] + m[1][2]) / s,
                0.25 * s,
                (m[0][1] - m[1][0]) / s,
            ]
        };
        Self::from_xyzw(q[0], q[1], q[2], q[3]).normalize()
    }
}

/// Hamilton product. `a * b` applies `b` first, then `a` — the same order as
/// `a.rotate(b.rotate(v))`. The result is not renormalised; call [`Quat::normalize`] when it
/// is about to be stored.
impl core::ops::Mul for Quat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }
}

/// Rigid-body placement: position in m, orientation as a canonical unit quaternion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub position: Vec3,
    pub orientation: Quat,
}

impl Pose {
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        orientation: Quat::IDENTITY,
    };

    #[must_use]
    pub fn new(position: Vec3, orientation: Quat) -> Self {
        Self {
            position,
            orientation: orientation.normalize(),
        }
    }

    /// `local`, which is expressed in this pose's frame, expressed in this pose's parent.
    #[must_use]
    pub fn compose(self, local: Self) -> Self {
        Self {
            position: self.position + self.orientation.rotate(local.position),
            orientation: (self.orientation * local.orientation).normalize(),
        }
    }

    #[must_use]
    pub fn inverse(self) -> Self {
        let inv = self.orientation.conjugate();
        Self {
            position: inv.rotate(-self.position),
            orientation: inv.normalize(),
        }
    }

    /// Point `p`, expressed in this pose's frame, expressed in this pose's parent.
    #[must_use]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        self.position + self.orientation.rotate(p)
    }
}

/// Body-frame inertia tensor: symmetric 3x3 in kg*m^2, with the principal-axis decomposition
/// cached (spec §3.1).
///
/// Serialises as the six unique components only; the cache is recomputed on load, so it can
/// never disagree with them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "InertiaRepr", into = "InertiaRepr")]
pub struct Inertia {
    ixx: f64,
    iyy: f64,
    izz: f64,
    ixy: f64,
    ixz: f64,
    iyz: f64,
    principal: Vec3,
    axes: Quat,
}

/// Wire format for [`Inertia`]: the six independent components.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename = "Inertia")]
struct InertiaRepr {
    ixx: f64,
    iyy: f64,
    izz: f64,
    ixy: f64,
    ixz: f64,
    iyz: f64,
}

impl From<InertiaRepr> for Inertia {
    fn from(r: InertiaRepr) -> Self {
        Self::new(r.ixx, r.iyy, r.izz, r.ixy, r.ixz, r.iyz)
    }
}

impl From<Inertia> for InertiaRepr {
    fn from(i: Inertia) -> Self {
        Self {
            ixx: i.ixx,
            iyy: i.iyy,
            izz: i.izz,
            ixy: i.ixy,
            ixz: i.ixz,
            iyz: i.iyz,
        }
    }
}

impl Inertia {
    /// Build from the six independent components and compute the principal-axis cache.
    #[must_use]
    pub fn new(ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64) -> Self {
        let (principal, axes) = principal_axes([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]);
        Self {
            ixx,
            iyy,
            izz,
            ixy,
            ixz,
            iyz,
            principal,
            axes,
        }
    }

    /// Diagonal inertia, principal axes aligned with the body frame.
    #[must_use]
    pub fn diagonal(ixx: f64, iyy: f64, izz: f64) -> Self {
        Self::new(ixx, iyy, izz, 0.0, 0.0, 0.0)
    }

    /// Rows of the symmetric tensor.
    #[must_use]
    pub fn matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.ixx, self.ixy, self.ixz],
            [self.ixy, self.iyy, self.iyz],
            [self.ixz, self.iyz, self.izz],
        ]
    }

    /// Principal moments, sorted descending (cached).
    #[must_use]
    pub fn principal_moments(&self) -> Vec3 {
        self.principal
    }

    /// Rotation whose columns are the principal axes, body frame to principal frame (cached).
    #[must_use]
    pub fn principal_axes(&self) -> Quat {
        self.axes
    }
}

/// Cyclic Jacobi eigendecomposition of a symmetric 3x3.
///
/// Fixed sweep count rather than a convergence test: the iteration count must not depend on
/// the data (spec §3.4 forbids data-derived control flow in deterministic paths), and 12
/// sweeps is far beyond the ~6 a 3x3 needs to reach machine precision.
// p/q/c/s/t are the textbook Jacobi rotation names; spelling them out would make the
// algorithm harder to check against the reference, not easier.
#[allow(clippy::many_single_char_names)]
fn principal_axes(m: [[f64; 3]; 3]) -> (Vec3, Quat) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..12 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let sign = if theta < 0.0 { -1.0 } else { 1.0 };
            let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in &mut a {
                let (rp, rq) = (row[p], row[q]);
                row[p] = c * rp - s * rq;
                row[q] = s * rp + c * rq;
            }
            let (mut rp, mut rq) = (a[p], a[q]);
            for (x, y) in rp.iter_mut().zip(rq.iter_mut()) {
                let (ox, oy) = (*x, *y);
                *x = c * ox - s * oy;
                *y = s * ox + c * oy;
            }
            a[p] = rp;
            a[q] = rq;
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }

    // Sort eigenpairs by moment, descending. Three elements: an explicit network, no sort
    // key allocation, no order-dependent tie-break.
    let mut order = [0usize, 1, 2];
    for i in 0..3 {
        for j in i + 1..3 {
            if a[order[j]][order[j]] > a[order[i]][order[i]] {
                order.swap(i, j);
            }
        }
    }
    let moments = Vec3::new(
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    );
    let mut cols = [
        [v[0][order[0]], v[1][order[0]], v[2][order[0]]],
        [v[0][order[1]], v[1][order[1]], v[2][order[1]]],
        [v[0][order[2]], v[1][order[2]], v[2][order[2]]],
    ];
    // Jacobi can return a reflection; make it a rotation so it is expressible as a quaternion.
    let det = Vec3::new(cols[0][0], cols[0][1], cols[0][2])
        .cross(Vec3::new(cols[1][0], cols[1][1], cols[1][2]))
        .dot(Vec3::new(cols[2][0], cols[2][1], cols[2][2]));
    if det < 0.0 {
        for e in &mut cols[2] {
            *e = -*e;
        }
    }
    (moments, Quat::from_columns(cols))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use proptest::prelude::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn quat_parts() -> impl Strategy<Value = (f64, f64, f64, f64)> {
        (
            -10.0..10.0f64,
            -10.0..10.0f64,
            -10.0..10.0f64,
            -10.0..10.0f64,
        )
    }

    proptest! {
        /// Spec §3.1: unit norm, w >= 0.
        #[test]
        fn normalize_is_canonical((x, y, z, w) in quat_parts()) {
            let q = Quat::from_xyzw(x, y, z, w);
            prop_assume!(q.norm() > 1e-6);
            let n = q.normalize();
            prop_assert!(n.w >= 0.0, "w = {}", n.w);
            prop_assert!(close(n.norm(), 1.0, 8.0 * f64::EPSILON), "norm = {}", n.norm());
        }

        /// Not bitwise idempotent: `norm()` of an already-normalised quaternion is only 1.0
        /// to within a rounding, so the second division moves the last bit. The invariant
        /// §3.1 asks for — unit norm, `w >= 0` — is a fixed point; the bits are stable to
        /// ~1 ULP. Anything that hashes a rotation must hash the normalised form once.
        #[test]
        fn normalize_is_idempotent_to_one_ulp((x, y, z, w) in quat_parts()) {
            let q = Quat::from_xyzw(x, y, z, w).normalize();
            let again = q.normalize();
            prop_assert!(again.w >= 0.0);
            for (a, b) in [(again.x, q.x), (again.y, q.y), (again.z, q.z), (again.w, q.w)] {
                prop_assert!((a - b).abs() <= 2.0 * f64::EPSILON * b.abs(), "{a} vs {b}");
            }
        }

        /// q and -q are the same rotation and must share one representation.
        #[test]
        fn double_cover_is_collapsed((x, y, z, w) in quat_parts()) {
            let q = Quat::from_xyzw(x, y, z, w);
            prop_assume!(q.norm() > 1e-6);
            let neg = Quat::from_xyzw(-x, -y, -z, -w);
            prop_assert_eq!(q.normalize(), neg.normalize());
        }

        #[test]
        fn rotation_preserves_length((x, y, z, w) in quat_parts(), v in (-5.0..5.0f64, -5.0..5.0f64, -5.0..5.0f64)) {
            let q = Quat::from_xyzw(x, y, z, w);
            prop_assume!(q.norm() > 1e-6);
            let v = Vec3::new(v.0, v.1, v.2);
            let r = q.normalize().rotate(v);
            prop_assert!(close(r.norm(), v.norm(), 1e-9 * (1.0 + v.norm())));
        }

        #[test]
        fn pose_inverse_round_trips((x, y, z, w) in quat_parts(), p in (-5.0..5.0f64, -5.0..5.0f64, -5.0..5.0f64)) {
            let q = Quat::from_xyzw(x, y, z, w);
            prop_assume!(q.norm() > 1e-6);
            let pose = Pose::new(Vec3::new(p.0, p.1, p.2), q);
            let back = pose.inverse().transform_point(pose.transform_point(Vec3::new(1.0, 2.0, 3.0)));
            prop_assert!(close(back.x, 1.0, 1e-9) && close(back.y, 2.0, 1e-9) && close(back.z, 3.0, 1e-9));
        }

        /// R * diag(principal) * R^T must reproduce the original tensor.
        #[test]
        fn inertia_decomposition_reconstructs(
            d in (0.1..10.0f64, 0.1..10.0f64, 0.1..10.0f64),
            o in (-0.05..0.05f64, -0.05..0.05f64, -0.05..0.05f64),
        ) {
            let i = Inertia::new(d.0, d.1, d.2, o.0, o.1, o.2);
            let p = i.principal_moments();
            let r = i.principal_axes();
            let cols = [
                r.rotate(Vec3::new(1.0, 0.0, 0.0)),
                r.rotate(Vec3::new(0.0, 1.0, 0.0)),
                r.rotate(Vec3::new(0.0, 0.0, 1.0)),
            ];
            let moments = [p.x, p.y, p.z];
            let got = |a: usize, b: usize| -> f64 {
                (0..3).map(|k| {
                    let c = [cols[k].x, cols[k].y, cols[k].z];
                    moments[k] * c[a] * c[b]
                }).sum()
            };
            let want = i.matrix();
            for (a, row) in want.iter().enumerate() {
                for (b, expected) in row.iter().enumerate() {
                    prop_assert!(close(got(a, b), *expected, 1e-9), "[{a}][{b}] {} vs {expected}", got(a, b));
                }
            }
            prop_assert!(p.x >= p.y && p.y >= p.z, "moments not descending: {p:?}");
        }

        #[test]
        fn serde_round_trip_is_bitwise(
            (x, y, z, w) in quat_parts(),
            d in (0.1..10.0f64, 0.1..10.0f64, 0.1..10.0f64),
        ) {
            let pose = Pose::new(Vec3::new(1.5, -2.5, 3.5), Quat::from_xyzw(x, y, z, w));
            let json = serde_json::to_string(&pose).unwrap();
            prop_assert_eq!(serde_json::from_str::<Pose>(&json).unwrap(), pose);

            let inertia = Inertia::new(d.0, d.1, d.2, 0.01, -0.02, 0.03);
            let json = serde_json::to_string(&inertia).unwrap();
            prop_assert_eq!(serde_json::from_str::<Inertia>(&json).unwrap(), inertia);
        }
    }

    #[test]
    fn axes_match_spec_3_1() {
        assert_eq!(axis::UP, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(axis::FORWARD, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(axis::FORWARD.cross(axis::LEFT), axis::UP);
        assert_eq!(units::DEG_TO_RAD * 180.0, core::f64::consts::PI);
        assert!(close(
            units::RAD_TO_DEG * core::f64::consts::PI,
            180.0,
            1e-12
        ));
    }

    #[test]
    fn identity_and_diagonal_inertia() {
        assert_eq!(Quat::default(), Quat::IDENTITY);
        assert_eq!(Quat::IDENTITY.rotate(axis::FORWARD), axis::FORWARD);
        assert_eq!(Pose::IDENTITY.transform_point(axis::UP), axis::UP);

        let i = Inertia::diagonal(3.0, 2.0, 1.0);
        assert_eq!(i.principal_moments(), Vec3::new(3.0, 2.0, 1.0));
        assert_eq!(i.principal_axes(), Quat::IDENTITY);
    }

    #[test]
    fn inertia_json_carries_six_components() {
        let json = serde_json::to_string(&Inertia::diagonal(1.0, 2.0, 3.0)).unwrap();
        assert!(
            !json.contains("principal"),
            "cache leaked into the wire format: {json}"
        );
        assert!(json.contains("ixx"));
    }
}
