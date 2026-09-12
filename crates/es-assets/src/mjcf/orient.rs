//! MJCF orientation attributes to a canonical spec 3.1 quaternion (xyzw, unit, `w >= 0`).
//!
//! MJCF offers five mutually exclusive spellings — `quat`, `euler`, `axisangle`, `xyaxes`,
//! `zaxis` — and `MuJoCo` takes the first one present. `quat` is wxyz there and xyzw here, which
//! is the single most common way an importer goes quietly wrong.
//!
//! `sin`/`cos` here are `std`'s: this is an offline import path, not a physics, observation or
//! reward kernel, so the spec 3.2 restriction to `es_math::approx` does not apply.

use es_math::{Quat, Vec3};

use super::attrs::Attrs;
use super::MjcfError;

/// The five orientation attributes, in the order `MuJoCo` resolves them.
const ATTRS: [&str; 5] = ["quat", "axisangle", "xyaxes", "zaxis", "euler"];

/// Reads whichever orientation attribute is present. `angle_scale` is 1 for
/// `<compiler angle="radian">` and `pi/180` for `degree`; `eulerseq` is the `<compiler>` value.
pub(crate) fn read(attrs: &Attrs<'_>, angle_scale: f64, eulerseq: &str) -> Result<Quat, MjcfError> {
    // Every spelling counts as read: MuJoCo takes the first one present and ignores the rest.
    for name in ATTRS {
        let _ = attrs.get(name);
    }
    if let Some([w, x, y, z]) = attrs.fixed::<4>("quat")? {
        // MJCF is wxyz, spec 3.1 is xyzw.
        return Ok(Quat::from_xyzw(x, y, z, w).normalize());
    }
    if let Some([x, y, z, angle]) = attrs.fixed::<4>("axisangle")? {
        return Ok(axis_angle(Vec3::new(x, y, z), angle * angle_scale));
    }
    if let Some([xx, xy, xz, yx, yy, yz]) = attrs.fixed::<6>("xyaxes")? {
        return from_xy(Vec3::new(xx, xy, xz), Vec3::new(yx, yy, yz))
            .ok_or_else(|| attrs.bad("xyaxes", attrs.get_or("xyaxes", "")));
    }
    if let Some([x, y, z]) = attrs.fixed::<3>("zaxis")? {
        let axis = Vec3::new(x, y, z);
        return from_zaxis(axis).ok_or_else(|| attrs.bad("zaxis", attrs.get_or("zaxis", "")));
    }
    if let Some([a, b, c]) = attrs.fixed::<3>("euler")? {
        return euler(
            eulerseq,
            [a * angle_scale, b * angle_scale, c * angle_scale],
        )
        .ok_or(MjcfError::BadEulerSeq {
            line: attrs.line(),
            seq: eulerseq.to_owned(),
        });
    }
    Ok(Quat::IDENTITY)
}

/// Rotation of `angle` rad about `axis`.
pub(crate) fn axis_angle(axis: Vec3, angle: f64) -> Quat {
    let axis = axis.normalize();
    let (s, c) = (angle * 0.5).sin_cos();
    Quat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c).normalize()
}

/// Shortest rotation taking `+Z` onto `dir`. `None` when `dir` has no direction.
pub(crate) fn from_zaxis(dir: Vec3) -> Option<Quat> {
    let z = dir.normalize();
    if z.norm() == 0.0 {
        return None;
    }
    let up = Vec3::new(0.0, 0.0, 1.0);
    let dot = up.dot(z);
    if dot > 1.0 - 1e-12 {
        return Some(Quat::IDENTITY);
    }
    if dot < -1.0 + 1e-12 {
        // Antiparallel: a half turn about any axis orthogonal to +Z.
        return Some(Quat::from_xyzw(1.0, 0.0, 0.0, 0.0).normalize());
    }
    Some(axis_angle(up.cross(z), dot.clamp(-1.0, 1.0).acos()))
}

/// Frame whose x axis is `x` and whose y axis is `y` after Gram-Schmidt, as `MuJoCo` does.
fn from_xy(x: Vec3, y: Vec3) -> Option<Quat> {
    let ex = x.normalize();
    if ex.norm() == 0.0 {
        return None;
    }
    let ey = (y - ex.scale(ex.dot(y))).normalize();
    if ey.norm() == 0.0 {
        return None;
    }
    Some(from_columns(ex, ey, ex.cross(ey)))
}

/// Quaternion of the rotation whose matrix has these columns (Shepperd: pick the largest
/// candidate so the square root never loses precision).
fn from_columns(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let (m00, m11, m22) = (x.x, y.y, z.z);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, 0.25 * s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [0.25 * s, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(y.x + x.y) / s, 0.25 * s, (z.y + y.z) / s, (z.x - x.z) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(z.x + x.z) / s, (z.y + y.z) / s, 0.25 * s, (x.y - y.x) / s]
    };
    Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize()
}

/// Euler angles in the `<compiler eulerseq>` order. Lowercase axes rotate about the moving
/// (intrinsic) frame, uppercase about the fixed (extrinsic) one, as in `MuJoCo`.
fn euler(seq: &str, angles: [f64; 3]) -> Option<Quat> {
    if seq.chars().count() != 3 {
        return None;
    }
    let mut q = Quat::IDENTITY;
    for (letter, angle) in seq.chars().zip(angles) {
        let axis = match letter.to_ascii_lowercase() {
            'x' => Vec3::new(1.0, 0.0, 0.0),
            'y' => Vec3::new(0.0, 1.0, 0.0),
            'z' => Vec3::new(0.0, 0.0, 1.0),
            _ => return None,
        };
        let step = axis_angle(axis, angle);
        q = if letter.is_ascii_uppercase() {
            step * q
        } else {
            q * step
        };
    }
    Some(q.normalize())
}
