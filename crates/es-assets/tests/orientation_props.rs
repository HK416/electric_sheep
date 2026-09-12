//! P32 property: whatever a file writes in an orientation attribute, what lands in the scene
//! is a canonical spec 3.1 rotation — unit norm, `w >= 0`, never a `NaN`.

use es_assets::mjcf::parse_str;
use proptest::prelude::*;

fn orientation(attribute: &str) -> es_math::Quat {
    let xml = format!("<mujoco><worldbody><body name=\"b\" {attribute}/></worldbody></mujoco>");
    let import = parse_str(&xml).unwrap_or_else(|e| panic!("{attribute}: {e}"));
    import.scene.bodies[1].pose.orientation
}

fn assert_canonical(q: es_math::Quat, attribute: &str) {
    assert!(q.w >= 0.0, "{attribute}: w < 0 ({q:?})");
    assert!(
        (q.norm() - 1.0).abs() < 1e-12,
        "{attribute}: not unit norm ({q:?})"
    );
}

proptest! {
    #[test]
    fn quat_attribute_is_canonicalised(
        w in -1e3f64..1e3,
        x in -1e3f64..1e3,
        y in -1e3f64..1e3,
        z in -1e3f64..1e3,
    ) {
        // MJCF writes wxyz; a zero-length quaternion falls back to the identity.
        let attribute = format!("quat=\"{w} {x} {y} {z}\"");
        assert_canonical(orientation(&attribute), &attribute);
    }

    #[test]
    fn axisangle_and_euler_are_canonicalised(
        x in -10f64..10.0,
        y in -10f64..10.0,
        z in -10f64..10.0,
        angle in -720f64..720.0,
    ) {
        for attribute in [
            format!("axisangle=\"{x} {y} {z} {angle}\""),
            format!("euler=\"{x} {y} {angle}\""),
            format!("zaxis=\"{x} {y} {z}\""),
        ] {
            assert_canonical(orientation(&attribute), &attribute);
        }
    }
}
