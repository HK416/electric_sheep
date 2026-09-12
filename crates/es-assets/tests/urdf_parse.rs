//! P34 oracle: hand-written URDF fixtures, asserted value by value.
//!
//! Fixtures live in `tests/fixtures/urdf/` at the workspace root (inputs, not golden outputs,
//! so they are not under `tests/golden/`), same convention as `tests/fixtures/mjcf/`.

use std::collections::BTreeMap;
use std::f64::consts::FRAC_PI_2;
use std::path::PathBuf;

use es_assets::scene::{scene_id, ActuatorTarget, JointKind, Shape};
use es_assets::urdf::{parse_urdf, Import, PackageResolver, UrdfError};

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/../../tests/fixtures/urdf/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn no_packages() -> PackageResolver {
    PackageResolver::default()
}

fn with_robo_pkg() -> PackageResolver {
    PackageResolver::new(BTreeMap::from([(
        "robo_pkg".to_owned(),
        PathBuf::from("/pkgs/robo_pkg"),
    )]))
}

fn load(name: &str, resolver: &PackageResolver) -> Import {
    let import = parse_urdf(&fixture(name), resolver).unwrap_or_else(|e| panic!("{name}: {e}"));
    import
        .scene
        .validate()
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    import
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn urdf_arm2_values_and_root_detection() {
    let scene = load("arm2.urdf", &no_packages()).scene;
    assert_eq!(scene.name, "arm2");
    assert_eq!(scene.bodies.len(), 3);

    let base = scene.bodies.iter().find(|b| b.name == "base_link").unwrap();
    assert_eq!(base.parent, None, "root link must have no parent");
    assert_eq!(base.id, scene_id("body", "base_link"));
    assert!(close(base.inertial.unwrap().mass, 2.0));

    let link1 = scene.bodies.iter().find(|b| b.name == "link1").unwrap();
    assert_eq!(link1.parent, Some(scene_id("body", "base_link")));
    assert!(close(link1.pose.position.z, 0.1));
    assert!(close(link1.inertial.unwrap().mass, 1.0));

    let link2 = scene.bodies.iter().find(|b| b.name == "link2").unwrap();
    assert_eq!(link2.parent, Some(scene_id("body", "link1")));
    assert!(close(link2.pose.position.x, 0.3));

    let shoulder = scene.joints.iter().find(|j| j.name == "shoulder").unwrap();
    assert_eq!(shoulder.kind, JointKind::Hinge);
    assert_eq!(shoulder.body, link1.id);
    let (lo, hi) = shoulder.range.expect("revolute joint keeps its limit");
    assert!(close(lo, -FRAC_PI_2) && close(hi, FRAC_PI_2), "{lo} {hi}");
    assert!(close(shoulder.damping, 0.1) && close(shoulder.friction_loss, 0.01));
    assert!(close(shoulder.axis.z, 1.0));

    let elbow = scene.joints.iter().find(|j| j.name == "elbow").unwrap();
    assert!(close(elbow.axis.y, 1.0));
    assert_eq!(elbow.range, Some((-1.0, 1.0)));

    // A box visual/collision half-extent is half the URDF <box size>.
    let base_box = base
        .geoms
        .iter()
        .find(|g| !g.visual_only)
        .expect("base_link has a collision geom");
    assert_eq!(
        base_box.shape,
        Shape::Box {
            half_extents: es_math::Vec3::new(0.1, 0.1, 0.05)
        }
    );
}

#[test]
fn urdf_mixed_joint_kinds_transmission_and_mesh() {
    let import = load("mixed_joints.urdf", &with_robo_pkg());
    let scene = import.scene;

    let weld = scene
        .joints
        .iter()
        .find(|j| j.name == "mount_weld")
        .unwrap();
    assert_eq!(weld.kind, JointKind::Fixed);

    // `continuous` never carries a range, even though this fixture writes no lower/upper.
    let spin = scene.joints.iter().find(|j| j.name == "spin").unwrap();
    assert_eq!(spin.kind, JointKind::Hinge);
    assert_eq!(spin.range, None);

    let lift = scene.joints.iter().find(|j| j.name == "lift").unwrap();
    assert_eq!(lift.kind, JointKind::Slide);
    assert_eq!(lift.range, Some((0.0, 0.5)));

    let actuator = scene
        .actuators
        .iter()
        .find(|a| a.name == "spin_motor")
        .unwrap();
    assert_eq!(actuator.target, ActuatorTarget::Joint(spin.id));
    assert!(close(actuator.gear[0], 50.0));
    assert_eq!(actuator.force_range, Some((-10.0, 10.0)));

    let mount = scene
        .bodies
        .iter()
        .find(|b| b.name == "sensor_mount")
        .unwrap();
    assert!(matches!(mount.geoms[0].shape, Shape::Mesh { .. }));
    assert_eq!(scene.assets.len(), 1);
    assert_eq!(
        scene.assets[0].path,
        with_robo_pkg()
            .resolve("package://robo_pkg/meshes/mount.stl")
            .unwrap()
            .to_string_lossy()
    );

    // <gazebo> is real URDF but carries nothing SceneDesc represents: a warning, not a failure.
    assert!(
        import.warnings.iter().any(|w| w.message.contains("gazebo")),
        "{:?}",
        import.warnings
    );
}

#[test]
fn urdf_unresolvable_package_is_an_error_listing_known_roots() {
    let err = parse_urdf(&fixture("mixed_joints.urdf"), &no_packages()).unwrap_err();
    match err {
        UrdfError::UnknownPackage { package, known } => {
            assert_eq!(package, "robo_pkg");
            assert!(known.is_empty());
        }
        other => panic!("expected UnknownPackage, got {other}"),
    }
}

#[test]
fn urdf_two_roots_is_an_error() {
    let err = parse_urdf(&fixture("two_roots.urdf"), &no_packages()).unwrap_err();
    assert!(matches!(err, UrdfError::MultipleRoots(_)), "{err}");
}

#[test]
fn urdf_malformed_xml_fails_to_parse_not_panics() {
    let err = parse_urdf(&fixture("malformed.urdf"), &no_packages()).unwrap_err();
    assert!(matches!(err, UrdfError::Xml(_)), "{err}");
}

#[test]
fn urdf_scene_hash_ignores_link_and_joint_order() {
    let forward = "<robot name=\"r\">\
        <link name=\"a\"><inertial><mass value=\"1\"/><inertia ixx=\"1\" iyy=\"1\" izz=\"1\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial></link>\
        <link name=\"b\"><inertial><mass value=\"2\"/><inertia ixx=\"1\" iyy=\"1\" izz=\"1\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial></link>\
        <joint name=\"j\" type=\"revolute\"><parent link=\"a\"/><child link=\"b\"/><axis xyz=\"0 0 1\"/><limit lower=\"-1\" upper=\"1\" effort=\"1\" velocity=\"1\"/></joint>\
    </robot>";
    let reordered = "<robot name=\"r\">\
        <joint name=\"j\" type=\"revolute\"><parent link=\"a\"/><child link=\"b\"/><axis xyz=\"0 0 1\"/><limit lower=\"-1\" upper=\"1\" effort=\"1\" velocity=\"1\"/></joint>\
        <link name=\"b\"><inertial><mass value=\"2\"/><inertia ixx=\"1\" iyy=\"1\" izz=\"1\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial></link>\
        <link name=\"a\"><inertial><mass value=\"1\"/><inertia ixx=\"1\" iyy=\"1\" izz=\"1\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial></link>\
    </robot>";
    let a = parse_urdf(forward, &no_packages()).unwrap().scene;
    let b = parse_urdf(reordered, &no_packages()).unwrap().scene;
    assert_eq!(a.scene_hash(), b.scene_hash());
}

#[test]
fn urdf_unknown_element_is_a_warning_not_an_error() {
    let xml = "<robot name=\"r\"><link name=\"a\"/><totally_unknown foo=\"bar\"/></robot>";
    let import = parse_urdf(xml, &no_packages()).unwrap();
    assert!(import
        .warnings
        .iter()
        .any(|w| w.message.contains("totally_unknown")));
}
