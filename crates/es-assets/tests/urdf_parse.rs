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

// --- packet M18/K8: `<mesh scale>` -------------------------------------------------------

/// A fresh directory holding `part.stl`, a 100 x 60 x 20 mm box written in millimetres
/// (outward-wound, so `MuJoCo` takes its volume), and nothing else.
fn mm_box_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("es-urdf-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    let h = [50.0f32, 30.0, 10.0];
    let corner = |i: usize| [0, 1, 2].map(|k| if i >> k & 1 == 1 { h[k] } else { -h[k] });
    let quads = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    let mut stl = vec![0u8; 80];
    stl.extend(12u32.to_le_bytes());
    for [a, b, c, d] in quads {
        for tri in [[a, b, c], [a, c, d]] {
            stl.extend([0u8; 12]);
            for v in tri.into_iter().flat_map(corner) {
                stl.extend(v.to_le_bytes());
            }
            stl.extend([0u8; 2]);
        }
    }
    std::fs::write(dir.join("part.stl"), stl).unwrap();
    dir
}

fn one_mesh_urdf(scale: Option<&str>) -> String {
    let scale = scale.map_or_else(String::new, |s| format!(" scale=\"{s}\""));
    format!(
        "<robot name=\"r\"><link name=\"a\">\
         <inertial><mass value=\"1\"/><inertia ixx=\"1\" iyy=\"1\" izz=\"1\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial>\
         <collision><geometry><mesh filename=\"part.stl\"{scale}/></geometry></collision>\
         </link></robot>"
    )
}

fn mesh_of(scene: &es_assets::SceneDesc) -> es_core::StableId {
    match scene.bodies.iter().find(|b| b.name == "a").unwrap().geoms[0].shape {
        Shape::Mesh { asset } => asset,
        ref other => panic!("{other:?}"),
    }
}

/// A millimetre STL at `scale="0.001 0.001 0.001"` reads as the MJCF `<mesh>` G2 writes for it:
/// the same asset (`part@0.001`), the same `mesh_scales` entry, the same scaled vertices.
#[test]
#[allow(clippy::float_cmp)] // the same bits are the point
fn urdf_mesh_scale_reads_as_mjcf_mesh_scale() {
    let dir = mm_box_dir("scale");
    let urdf = |scale| {
        let mut import = parse_urdf(&one_mesh_urdf(scale), &no_packages()).unwrap();
        es_assets::mesh::load(&mut import.scene, &dir).unwrap();
        import
    };
    let import = urdf(Some("0.001 0.001 0.001"));
    assert!(import.warnings.iter().all(|w| !w.message.contains("scale")));
    let u = import.scene;
    let mjcf = r#"<mujoco><asset><mesh name="part@0.001" file="part.stl" scale="0.001 0.001 0.001"/>
        </asset><worldbody><body name="a"><geom name="g" type="mesh" mesh="part@0.001"/></body>
        </worldbody></mujoco>"#;
    let mut m = es_assets::parse_mjcf(mjcf).unwrap().scene;
    es_assets::mesh::load(&mut m, &dir).unwrap();

    let (ua, ma) = (mesh_of(&u), mesh_of(&m));
    assert_eq!(ua, ma);
    let asset = |s: &es_assets::SceneDesc| {
        let a = s.assets.iter().find(|a| a.id == ua).unwrap();
        (a.name.clone(), a.path.clone(), a.hash)
    };
    assert_eq!(asset(&u).0, "part@0.001");
    assert_eq!(asset(&u), asset(&m));
    assert_eq!(u.mesh_scales, m.mesh_scales);
    assert_eq!(u.mesh_scales[&ua], [0.001; 3]);
    assert_eq!(u.meshes[&ua].positions, m.meshes[&ua].positions);
    let positions = u.mesh_positions(ua).unwrap();
    assert_eq!(*positions, *m.mesh_positions(ua).unwrap());
    assert_eq!(
        extents(&positions),
        [0.1, 0.06, 0.02].map(|e: f64| e as f32)
    );

    // Not uniform: every axis in the name. Absent and 1 add nothing: no `scene_hash` moves.
    let u = urdf(Some("0.001 0.002 0.001")).scene;
    let id = mesh_of(&u);
    let name = &u.assets.iter().find(|a| a.id == id).unwrap().name;
    assert_eq!(name, "part@0.001,0.002,0.001");
    let (absent, one) = (urdf(None).scene, urdf(Some("1 1 1")).scene);
    assert!(one.mesh_scales.is_empty());
    assert_eq!(absent.scene_hash(), one.scene_hash());
    assert_eq!(
        one.assets[0].name, "collision1",
        "an unscaled mesh keeps its geom's name"
    );
}

fn extents(p: &[[f32; 3]]) -> [f32; 3] {
    [0, 1, 2].map(|k| {
        let (lo, hi) = p.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| {
            (lo.min(v[k]), hi.max(v[k]))
        });
        hi - lo
    })
}

/// A zero, negative or non-finite scale has no mesh to give; the parse fails naming `scale`.
#[test]
fn urdf_non_positive_mesh_scale_is_refused_by_name() {
    for scale in ["0 0 0", "0.001 -0.001 0.001", "1 inf 1", "1 1"] {
        let err = parse_urdf(&one_mesh_urdf(Some(scale)), &no_packages()).unwrap_err();
        assert!(
            matches!(err, UrdfError::BadAttr { attr: "scale", .. }),
            "{err}"
        );
        assert!(err.to_string().contains("`scale`"), "{err}");
    }
}

/// `MuJoCo` reading the URDF itself and reading the MJCF `write_mjcf` makes of our reading give
/// the millimetre box its size in metres. Needs `ES_PYTHON` with `mujoco`; prints `SKIP` without.
#[test]
fn urdf_scaled_mesh_loads_on_mujoco_at_its_size() {
    let Some(python) = std::env::var("ES_PYTHON")
        .ok()
        .filter(|p| !p.trim().is_empty())
    else {
        println!("SKIP urdf_scaled_mesh_loads_on_mujoco_at_its_size: ES_PYTHON is not set");
        return;
    };
    let dir = mm_box_dir("mujoco");
    let urdf = one_mesh_urdf(Some("0.001 0.001 0.001"));
    std::fs::write(dir.join("r.urdf"), &urdf).unwrap();
    let mut scene = parse_urdf(&urdf, &no_packages()).unwrap().scene;
    es_assets::mesh::load(&mut scene, &dir).unwrap();
    let out = dir.join("export");
    std::fs::create_dir_all(&out).unwrap();
    let xml = es_assets::mjcf::write_mjcf(&scene, &out).unwrap();
    std::fs::write(out.join("scene.xml"), xml).unwrap();

    let script = "import sys, mujoco\n\
                  for p in sys.argv[1:]:\n\
                  \x20   m = mujoco.MjModel.from_xml_path(p)\n\
                  \x20   a, n = m.mesh_vertadr[0], m.mesh_vertnum[0]\n\
                  \x20   v = m.mesh_vert[a:a + n]\n\
                  \x20   print(*sorted(float(x) for x in v.max(0) - v.min(0)))\n";
    let run = std::process::Command::new(&python)
        .args(["-c", script])
        .arg(dir.join("r.urdf"))
        .arg(out.join("scene.xml"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let ours = scene.mesh_positions(mesh_of(&scene)).unwrap();
    let mut want = extents(&ours).map(f64::from);
    want.sort_by(f64::total_cmp);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "{stdout}");
    for (what, line) in ["MuJoCo's URDF reader", "write_mjcf"].iter().zip(lines) {
        let got: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        println!("{what}: extents {got:?} (ours {want:?})");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{what}: {got:?} vs {want:?}");
        }
    }
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
