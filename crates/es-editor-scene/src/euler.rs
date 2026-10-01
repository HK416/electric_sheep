//! The inspector's rotation in degrees (design section 3.1): the document stores quaternions,
//! the bits that enter `scene_hash`; a person reads and types roll, pitch and yaw — fixed axes X,
//! then Y, then Z, URDF's `rpy`. Both directions go through `es_math::approx`'s `f64` port, so
//! no host `libm` reaches a quaternion the document stores (spec 3.2). A quaternion is only
//! written when a person changes an angle; one shown and left alone keeps its bits.

use es_math::approx::{acos_f64, sin_cos_f64};
use es_math::units::{DEG_TO_RAD, RAD_TO_DEG};

/// `atan2` from `acos`: exact in sign and quadrant, `atan2(0, 0) = 0`.
fn atan2(y: f64, x: f64) -> f64 {
    let h = (x * x + y * y).sqrt();
    if h == 0.0 {
        return 0.0;
    }
    let a = acos_f64((x / h).clamp(-1.0, 1.0));
    if y < 0.0 {
        -a
    } else {
        a
    }
}

/// Roll, pitch, yaw in degrees of the rotation `[x, y, z, w]`; pitch in [-90, 90].
#[allow(clippy::many_single_char_names)] // the quaternion's own lane names
pub fn deg_from_quat([x, y, z, w]: [f64; 4]) -> [f64; 3] {
    let roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    let s = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0);
    let pitch = atan2(s, (1.0 - s * s).sqrt());
    let yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    [roll, pitch, yaw].map(|a| a * RAD_TO_DEG)
}

/// The unit quaternion `[x, y, z, w]`, `w >= 0`, of roll, pitch, yaw in degrees: `qz * qy * qx`.
pub fn quat_from_deg(rpy: [f64; 3]) -> [f64; 4] {
    let [(sr, cr), (sp, cp), (sy, cy)] = rpy.map(|a| sin_cos_f64(a * DEG_TO_RAD * 0.5));
    let q = [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ];
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let sign = if q[3] < 0.0 { -1.0 } else { 1.0 };
    q.map(|v| sign * v / n)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn degrees_and_quaternions_round_trip() {
        for rpy in [
            [0.0, 0.0, 0.0],
            [30.0, -20.0, 75.0],
            [-170.0, 45.0, 120.0],
            [90.0, 0.0, -90.0],
            [0.0, 89.0, 0.0],
        ] {
            let back = deg_from_quat(quat_from_deg(rpy));
            for (a, b) in rpy.iter().zip(back) {
                assert!((a - b).abs() < 1e-9, "{rpy:?} -> {back:?}");
            }
        }
        // Identity is exactly zero, and a quarter turn about Z is exactly what MJCF writes.
        assert_eq!(deg_from_quat([0.0, 0.0, 0.0, 1.0]), [0.0; 3]);
        let q = quat_from_deg([0.0, 0.0, 90.0]);
        assert!(
            (q[2] - q[3]).abs() < 1e-15 && q[0] == 0.0 && q[1] == 0.0,
            "{q:?}"
        );
        // The Shadow Hand's side camera reads as angles and back within rounding.
        let side = [
            0.445_885_483_867_369_87,
            -0.289_778_636_094_924_4,
            -0.461_490_007_498_030_9,
            0.710_099_605_913_701_5,
        ];
        let again = quat_from_deg(deg_from_quat(side));
        for (a, b) in side.iter().zip(again) {
            assert!((a - b).abs() < 1e-12, "{again:?}");
        }
    }
}
