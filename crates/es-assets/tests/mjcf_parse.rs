//! P32 / P33 oracle: hand-written MJCF fixtures, asserted value by value.
//!
//! The fixtures live in `tests/fixtures/mjcf/` at the workspace root (they are inputs, not
//! golden outputs, so they are not under `tests/golden/`).

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use es_assets::mjcf::{parse_str, Import, MjcfError};
use es_assets::scene::{
    scene_id, ActuatorKind, ActuatorTarget, AssetKind, FrictionCone, Integrator, JointKind,
    SensorKind, SensorTarget, Shape, Solver, TendonKind,
};
use es_math::Vec3;

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/../../tests/fixtures/mjcf/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn load(name: &str) -> Import {
    let import = parse_str(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
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
fn pendulum_values_are_si() {
    let scene = load("pendulum.xml").scene;
    assert_eq!(scene.name, "pendulum");
    // world + rod.
    assert_eq!(scene.bodies.len(), 2);
    let rod = scene.bodies.iter().find(|b| b.name == "rod").unwrap();
    assert_eq!(rod.parent, Some(scene_id("body", "world")));
    assert_eq!(rod.id, scene_id("body", "world/rod"));
    assert!(close(rod.pose.position.z, 1.0));

    let joint = &scene.joints[0];
    assert_eq!(joint.kind, JointKind::Hinge);
    assert_eq!(joint.id, scene_id("joint", "world/rod/hinge"));
    let (lo, hi) = joint.range.expect("autolimits turns a range into limits");
    assert!(close(lo, -FRAC_PI_2) && close(hi, FRAC_PI_2), "{lo} {hi}");
    assert!(close(joint.damping, 0.1) && close(joint.armature, 0.01));

    // `fromto` fixes both the length and the pose; the shaft points down -Z.
    let shaft = rod.geoms.iter().find(|g| g.name == "shaft").unwrap();
    assert_eq!(
        shaft.shape,
        Shape::Capsule {
            radius: 0.02,
            half_length: 0.25
        }
    );
    assert!(close(shaft.pose.position.z, -0.25));
    let down = shaft.pose.orientation.rotate(Vec3::new(0.0, 0.0, 1.0));
    assert!(close(down.z, -1.0), "{down:?}");

    let inertial = rod.inertial.expect("<inertial> is explicit here");
    assert!(close(inertial.mass, 1.5) && close(inertial.com.z, -0.25));
    assert!(close(inertial.inertia.matrix()[2][2], 0.002));

    let options = scene.options;
    assert!(close(options.timestep, 0.001) && close(options.impratio, 2.0));
    assert_eq!(options.integrator, Integrator::Rk4);
    assert_eq!(options.cone, FrictionCone::Elliptic);
    assert_eq!(options.solver, Solver::Cg);
    assert_eq!(options.iterations, 50);

    let floor = &scene.bodies[0].geoms[0];
    assert!(matches!(floor.shape, Shape::Plane { half_x: 5.0, .. }));
    assert!(close(floor.friction[0], 0.8));
}

#[test]
fn default_classes_merge_in_the_right_order() {
    let scene = load("arm2.xml").scene;
    let joint = |name: &str| scene.joints.iter().find(|j| j.name == name).unwrap();
    let geom = |name: &str| {
        scene
            .bodies
            .iter()
            .flat_map(|b| &b.geoms)
            .find(|g| g.name == name)
            .unwrap()
    };

    // main default only.
    assert!(close(joint("j1").damping, 0.5));
    assert_eq!(joint("j1").kind, JointKind::Hinge);
    // angle="radian": the range is not rescaled.
    assert_eq!(joint("j1").range, Some((-1.0, 1.0)));
    // explicit > childclass "heavy" (5) > main (0.5).
    assert!(close(joint("j2").damping, 1.0));
    // childclass reaches the geom.
    assert_eq!(
        geom("g2").shape,
        Shape::Capsule {
            radius: 0.06,
            half_length: 0.125
        }
    );
    assert!(close(geom("g2").density, 2000.0));
    // class= on the element overrides the inherited childclass.
    assert_eq!(geom("g3").shape, Shape::Sphere { radius: 0.03 });
    assert!(close(geom("g3").density, 1000.0));
    assert!(close(geom("g1").density, 1000.0));

    let link2 = scene.bodies.iter().find(|b| b.name == "link2").unwrap();
    assert_eq!(link2.id, scene_id("body", "world/link1/link2"));
}

#[test]
fn unknown_class_is_an_error() {
    let xml = r#"<mujoco><worldbody><body name="b" class="nope"/></worldbody></mujoco>"#;
    assert!(matches!(
        parse_str(xml).unwrap_err(),
        MjcfError::UnknownClass { .. }
    ));
}

#[test]
fn actuators_sensors_and_tendons() {
    let import = load("actuated.xml");
    let scene = &import.scene;

    // Slide limits are lengths and stay as written; hinge limits are angles.
    let slide = scene.joints.iter().find(|j| j.name == "slide1").unwrap();
    assert_eq!(slide.kind, JointKind::Slide);
    assert_eq!(slide.range, Some((-0.5, 0.5)));
    let hinge = scene.joints.iter().find(|j| j.name == "hinge1").unwrap();
    let (lo, hi) = hinge.range.unwrap();
    assert!(close(lo, 0.0) && close(hi, FRAC_PI_4), "{lo} {hi}");

    assert_eq!(scene.actuators.len(), 4);
    let actuator = |name: &str| scene.actuators.iter().find(|a| a.name == name).unwrap();
    assert_eq!(actuator("m1").kind, ActuatorKind::Motor);
    assert_eq!(actuator("m1").target, ActuatorTarget::Joint(slide.id));
    assert!(close(actuator("m1").gear[0], 10.0));
    assert_eq!(actuator("m1").ctrl_range, Some((-1.0, 1.0)));
    assert_eq!(actuator("m1").force_range, Some((-50.0, 50.0)));
    assert_eq!(
        actuator("p1").kind,
        ActuatorKind::Position { kp: 30.0, kv: 2.0 }
    );
    assert_eq!(actuator("v1").kind, ActuatorKind::Velocity { kv: 4.0 });
    assert_eq!(
        actuator("g1").kind,
        ActuatorKind::General {
            gain: [5.0, 0.0, 0.0],
            bias: [0.0, -3.0, 0.0]
        }
    );
    let coupler = scene.tendons.iter().find(|t| t.name == "coupler").unwrap();
    assert_eq!(actuator("g1").target, ActuatorTarget::Tendon(coupler.id));

    match &coupler.kind {
        TendonKind::Fixed { joints } => {
            assert_eq!(joints.len(), 2);
            assert_eq!(joints[0], (slide.id, 1.0));
            assert!(close(joints[1].1, -0.5));
        }
        other @ TendonKind::Spatial { .. } => panic!("{other:?}"),
    }
    let cable = scene.tendons.iter().find(|t| t.name == "cable").unwrap();
    assert_eq!(cable.range, Some((0.0, 0.4)));
    assert!(close(cable.stiffness, 7.0) && close(cable.damping, 0.3));
    match &cable.kind {
        TendonKind::Spatial { sites } => assert_eq!(sites.len(), 2),
        other @ TendonKind::Fixed { .. } => panic!("{other:?}"),
    }

    assert_eq!(scene.sensors.len(), 11);
    let kinds: Vec<SensorKind> = scene.sensors.iter().map(|s| s.kind).collect();
    assert!(kinds.contains(&SensorKind::Touch) && kinds.contains(&SensorKind::RangeFinder));
    // An unnamed sensor is named after its target, so its id does not depend on element order.
    let jointpos = scene
        .sensors
        .iter()
        .find(|s| s.name == "jointpos_hinge1")
        .unwrap();
    assert_eq!(jointpos.kind, SensorKind::JointPos);
    assert_eq!(jointpos.target, SensorTarget::Joint(hinge.id));
    assert!(close(jointpos.noise, 0.001));
    let named = scene.sensors.iter().find(|s| s.name == "hv").unwrap();
    assert!(close(named.cutoff, 10.0));
    let framequat = scene
        .sensors
        .iter()
        .find(|s| s.kind == SensorKind::FrameQuat)
        .unwrap();
    assert_eq!(
        framequat.target,
        SensorTarget::Body(scene_id("body", "world/base/finger"))
    );

    // `fovy` is degrees in MJCF and radians here.
    let camera = &scene.cameras[0];
    assert!(close(camera.fovy, 60_f64.to_radians()));
    assert_eq!(camera.body, Some(scene_id("body", "world/base")));
}

#[test]
fn dangling_reference_is_an_error_with_a_line() {
    let xml =
        "<mujoco>\n<worldbody/>\n<actuator>\n<motor joint=\"ghost\"/>\n</actuator>\n</mujoco>";
    match parse_str(xml).unwrap_err() {
        MjcfError::UnknownRef { line, name, .. } => {
            assert_eq!((line, name.as_str()), (4, "ghost"));
        }
        other => panic!("{other}"),
    }
}

#[test]
fn every_orientation_spelling_agrees() {
    let scene = load("orientations.xml").scene;
    let quat = |name: &str| {
        scene
            .bodies
            .iter()
            .find(|b| b.name == name)
            .unwrap()
            .pose
            .orientation
    };
    // 90 degrees about +Z, however it is spelled.
    let turn = quat("b_quat");
    assert!(close(turn.z, FRAC_PI_4.sin()) && close(turn.w, FRAC_PI_4.cos()));
    for name in ["b_euler", "b_axisangle", "b_xyaxes"] {
        let q = quat(name);
        assert!(close(q.x, turn.x) && close(q.y, turn.y), "{name}: {q:?}");
        assert!(close(q.z, turn.z) && close(q.w, turn.w), "{name}: {q:?}");
    }
    // zaxis takes +Z onto +X.
    let moved = quat("b_zaxis").rotate(Vec3::new(0.0, 0.0, 1.0));
    assert!(close(moved.x, 1.0), "{moved:?}");
    // MJCF wxyz with a negative w canonicalises to w >= 0 (spec 3.1).
    let negated = quat("b_negative_w");
    assert!(
        negated.w >= 0.0 && close(negated.z, -FRAC_PI_4.sin()),
        "{negated:?}"
    );
}

#[test]
fn assets_are_references_not_files() {
    let import = load("assets.xml");
    let scene = &import.scene;
    assert_eq!(scene.assets.len(), 5);
    let asset = |name: &str| scene.assets.iter().find(|a| a.name == name).unwrap();
    assert_eq!(asset("arm").path, "meshes/arm.stl");
    assert_eq!(asset("arm").kind, AssetKind::Mesh);
    // An unnamed mesh takes the file stem, directories and extension stripped.
    assert_eq!(asset("gripper").path, "meshes/parts/gripper.obj");
    assert_eq!(asset("wood").path, "textures/wood.png");
    assert_eq!(asset("woodmat").kind, AssetKind::Material);

    let tool = scene.bodies.iter().find(|b| b.name == "tool").unwrap();
    let arm_mesh = tool.geoms.iter().find(|g| g.name == "armmesh").unwrap();
    assert_eq!(
        arm_mesh.shape,
        Shape::Mesh {
            asset: asset("arm").id
        }
    );
    assert_eq!(arm_mesh.material, Some(asset("woodmat").id));
    assert!(
        tool.geoms
            .iter()
            .find(|g| g.name == "grip")
            .unwrap()
            .visual_only
    );
    assert!(matches!(
        scene.bodies[0].geoms[0].shape,
        Shape::HeightField { .. }
    ));

    // Sections with no SceneDesc representation are reported, never dropped in silence.
    let reported = |needle: &str| import.warnings.iter().any(|w| w.message.contains(needle));
    assert!(
        reported("<equality>") && reported("<keyframe>"),
        "{:?}",
        import.warnings
    );
}

#[test]
fn unknown_elements_and_attributes_become_warnings() {
    let xml = "<mujoco>\n  <worldbody>\n    <light name=\"l\"/>\n    <body name=\"b\" \
               gravcomp=\"1\"/>\n  </worldbody>\n</mujoco>";
    let import = parse_str(xml).unwrap();
    assert!(import
        .warnings
        .iter()
        .any(|w| w.message.contains("<light>")));
    assert!(import
        .warnings
        .iter()
        .any(|w| w.message.contains("gravcomp") && w.line == 4));
}

#[test]
fn scene_hash_survives_element_reordering() {
    let body = |name: &str, z: f64| {
        format!(
            "<body name=\"{name}\" pos=\"0 0 {z}\">\
             <joint name=\"j_{name}\" type=\"hinge\" axis=\"0 0 1\"/>\
             <geom name=\"g_{name}\" type=\"sphere\" size=\"0.1\"/></body>"
        )
    };
    let wrap = |inner: String| format!("<mujoco><worldbody>{inner}</worldbody></mujoco>");
    let forward = parse_str(&wrap(body("a", 1.0) + &body("b", 2.0))).unwrap();
    let reversed = parse_str(&wrap(body("b", 2.0) + &body("a", 1.0))).unwrap();
    assert_eq!(
        forward.scene.scene_hash(),
        reversed.scene.scene_hash(),
        "element order must not change the scene hash (spec 5.3)"
    );

    // A value change must change it.
    let moved = parse_str(&wrap(body("a", 1.5) + &body("b", 2.0))).unwrap();
    assert_ne!(forward.scene.scene_hash(), moved.scene.scene_hash());
}

#[test]
fn parsing_is_deterministic_across_runs() {
    let first = load("actuated.xml").scene.scene_hash();
    let second = load("actuated.xml").scene.scene_hash();
    assert_eq!(first, second);
}
