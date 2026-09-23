//! Packet M11/I1 oracle 1: what `PhysXBackend` declares, and what its mapping report says about
//! the MJCF importer, hold without Isaac Sim (spec 28.14 rule 6: every importer gap is a
//! mapping-report row; spec 17.3: a backend declares honestly).
//!
//!     cargo test -p es-physics-backend physx_

use es_assets::scene::SceneDesc;
use es_physics_backend::mapping::{lookup, MjcfRow, Status};
use es_physics_backend::{
    mapping_report, physx, BackendKind, MappingReport, PhysXBackend, TaskFeature,
};
use es_physics_core::caps::{DeterminismTier, FloatPrecision};
use es_physics_core::{Feature, LoadConfig, PhysicsBackend, PhysicsError, StateView};

fn fixture(name: &str) -> SceneDesc {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/mjcf/");
    let text = std::fs::read_to_string(format!("{dir}{name}")).unwrap();
    let mut scene = es_assets::parse_mjcf(&text).unwrap().scene;
    es_assets::mesh::load(&mut scene, std::path::Path::new(dir)).unwrap();
    scene
}

fn inline(xml: &str) -> SceneDesc {
    es_assets::parse_mjcf(xml).unwrap().scene
}

/// `(feature name, status label)` of every row of `report`.
fn rows(report: &MappingReport) -> Vec<(String, &'static str)> {
    report
        .rows
        .iter()
        .map(|r| (r.feature.to_string(), r.mapping.status.label()))
        .collect()
}

fn row<'a>(report: &'a MappingReport, name: &str) -> &'a es_physics_backend::MappingRow {
    report
        .rows
        .iter()
        .find(|r| r.feature.to_string() == name)
        .unwrap_or_else(|| panic!("no `{name}` row in\n{report}"))
}

#[test]
fn physx_capabilities_are_declared_honestly() {
    let cpu = physx::capabilities_on(false);
    let gpu = physx::capabilities_on(true);
    assert_eq!(
        physx::capabilities(),
        cpu,
        "the CPU pipeline is the default"
    );
    for caps in [&cpu, &gpu] {
        assert_eq!(caps.name, "physx");
        assert_eq!(BackendKind::from_name(&caps.name), Some(BackendKind::PhysX));
        // spec 17.3: only mujoco-cpu may claim bitwise. PhysX CPU reruns were bitwise in I0,
        // but CPU vs GPU differ by up to 0.27 rad: tier 2 is what is declared.
        assert_eq!(caps.determinism, DeterminismTier::CrossBackend);
        assert_eq!(caps.float, FloatPrecision::F32);
        // No per-env model parameters: set_params is refused by name.
        assert!(!caps.has(Feature::ModelParams));
        for kept in [
            Feature::JointHinge,
            Feature::JointFree,
            Feature::JointFixed,
            Feature::JointLimit,
            Feature::JointArmature,
            Feature::ActuatorPosition,
            Feature::ActuatorOnJoint,
            Feature::ContactPyramidal,
            Feature::ContactMesh,
        ] {
            assert!(caps.has(kept), "{kept} is mapped and must be declared");
        }
        for dropped in [
            Feature::JointFrictionLoss,
            Feature::JointBall,
            Feature::ContactSoftParams,
            Feature::ContactCondim6,
            Feature::ContactElliptic,
            Feature::ActuatorVelocity,
            Feature::ActuatorGeneral,
            Feature::Tendon,
            Feature::SensorJointPos,
            Feature::SensorTouch,
            Feature::SensorForce,
        ] {
            assert!(
                !caps.has(dropped),
                "{dropped} is not mapped and must not be declared"
            );
        }
        // The repairs the adapter makes are declared, each where it bites.
        let quirks: Vec<String> = caps
            .quirks
            .iter()
            .map(|q| format!("{}: {}", q.feature, q.description))
            .collect();
        let quirks = quirks.join("\n");
        for word in [
            "kp",
            "kv",
            "ctrlrange",
            "reset",
            "free",
            "OBJ",
            "angular damping",
            "prototype",
            "pipeline",
        ] {
            assert!(
                quirks.contains(word),
                "no quirk mentions `{word}`:\n{quirks}"
            );
        }
    }
    assert!(!cpu.batch.gpu_resident);
    assert!(gpu.batch.gpu_resident);

    // The declaration is the `PhysX` column of the mapping, feature by feature.
    for feature in TaskFeature::all() {
        let TaskFeature::Capability(capability) = feature else {
            continue;
        };
        let mapped = matches!(
            lookup(feature, BackendKind::PhysX).status,
            Status::Native(_) | Status::Approximated(_)
        );
        assert_eq!(cpu.has(capability), mapped, "{capability}");
    }
}

#[test]
fn physx_mapping_report_names_every_dropped_feature() {
    // SO-101: every importer gap I0 measured that the adapter does not repair is a row, never
    // a silent drop (spec 28.14 rule 6).
    let so101 = mapping_report(&fixture("so101_pick_place.xml"), BackendKind::PhysX);
    assert!(!so101.blocked, "{so101}");
    let got = rows(&so101);
    for (name, status) in [
        ("JointFrictionLoss", "unsupported"),
        ("joint.damping", "approximated"),
        ("geom.friction", "approximated"),
        ("option.solver", "unsupported"),
        ("body.mass_from_geoms", "unsupported"),
        ("ContactCondim6", "unsupported"),
        ("ContactElliptic", "unsupported"),
        ("actuator.pd", "approximated"),
        ("ActuatorPosition", "approximated"),
    ] {
        assert!(
            got.contains(&(name.to_owned(), status)),
            "`{name}` is not `{status}` in\n{so101}"
        );
    }
    // A spec 17.2 row keeps the spec's status, and its note says what this adapter does.
    let soft = row(&so101, "contact.soft_params");
    assert!(soft.mapping.status.note().contains("dropped"), "{so101}");

    // The MJCF rows are the PhysX column's business: other backends' reports do not move.
    for kind in [
        BackendKind::MuJoCoCpu,
        BackendKind::MjWarp,
        BackendKind::Newton,
    ] {
        let report = mapping_report(&fixture("so101_pick_place.xml"), kind);
        assert!(
            !report
                .rows
                .iter()
                .any(|r| matches!(r.feature, TaskFeature::Mjcf(_))),
            "{report}"
        );
    }
    for row in MjcfRow::ALL {
        let cell = lookup(TaskFeature::Mjcf(row), BackendKind::MuJoCoCpu);
        assert!(matches!(cell.status, Status::Native(_)), "{}", row.name());
    }

    // mesh_box: a mesh collides as its convex hull; a body without <inertial> takes PhysX's mass.
    let mesh = mapping_report(&fixture("mesh_box.xml"), BackendKind::PhysX);
    assert!(!mesh.blocked, "{mesh}");
    assert_eq!(
        row(&mesh, "ContactMesh").mapping.status.label(),
        "approximated"
    );
    assert!(row(&mesh, "ContactMesh")
        .mapping
        .status
        .note()
        .contains("hull"));
    assert_eq!(
        row(&mesh, "body.mass_from_geoms").mapping.status.label(),
        "unsupported"
    );
    assert!(
        !rows(&mesh).iter().any(|(n, _)| n == "joint.damping"),
        "{mesh}"
    );

    // contype / conaffinity bitmasks other than 1/1 and 0/0.
    let masked = mapping_report(
        &inline(
            r#"<mujoco><worldbody><body name="b"><joint name="j"/>
                 <geom name="g" type="sphere" size="0.1" contype="2" conaffinity="1"/>
               </body></worldbody></mujoco>"#,
        ),
        BackendKind::PhysX,
    );
    assert_eq!(
        row(&masked, "geom.contype_conaffinity")
            .mapping
            .status
            .label(),
        "unsupported"
    );

    // What the adapter cannot run at all is blocked, by name, before a process spawns.
    for (xml, name) in [
        (
            r#"<mujoco><worldbody><body name="b"><joint name="j" type="ball"/>
                 <geom name="g" type="sphere" size="0.1"/></body></worldbody></mujoco>"#,
            "JointBall",
        ),
        (
            r#"<mujoco><worldbody><body name="b"><joint name="j"/>
                 <geom name="g" type="sphere" size="0.1"/></body></worldbody>
               <sensor><jointpos name="s" joint="j"/></sensor></mujoco>"#,
            "SensorJointPos",
        ),
        (
            r#"<mujoco><worldbody><body name="b"><joint name="j"/>
                 <geom name="g" type="sphere" size="0.1"/></body></worldbody>
               <actuator><velocity name="v" joint="j" kv="1"/></actuator></mujoco>"#,
            "ActuatorVelocity",
        ),
    ] {
        let scene = inline(xml);
        let report = mapping_report(&scene, BackendKind::PhysX);
        assert!(report.blocked, "{report}");
        assert!(
            report.blocking().any(|r| r.feature.to_string() == name),
            "{report}"
        );
        let err = PhysXBackend::new()
            .load(&scene, &LoadConfig::default())
            .unwrap_err();
        let PhysicsError::Unsupported(message) = &err else {
            panic!("expected the mapping report, got {err:?}");
        };
        assert!(
            message.contains(name) && message.contains("blocked: yes"),
            "{message}"
        );
    }
}

/// Without `ES_ISAAC_PYTHON` the backend is unavailable with a reason (the documented SKIPPED
/// path of every caller), and nothing is spawned.
#[test]
fn physx_is_unavailable_without_es_isaac_python() {
    if std::env::var_os("ES_ISAAC_PYTHON").is_some() {
        eprintln!("SKIP physx_is_unavailable_without_es_isaac_python: ES_ISAAC_PYTHON is set");
        return;
    }
    let why = PhysXBackend::is_available().unwrap_err();
    assert!(why.contains("ES_ISAAC_PYTHON"), "{why}");
    let why = es_physics_backend::is_available(BackendKind::PhysX).unwrap_err();
    assert!(why.contains("ES_ISAAC_PYTHON"), "{why}");
}

/// Oracle 2's shape, runnable wherever Isaac Sim is: SO-101 and `mesh_box` on `PhysX` (the
/// `ES_PHYSX_DEVICE` pipeline) against mujoco-cpu under one control sequence, printed (numbers
/// recorded, no tolerance claimed); then a `PhysX` rerun against the first run, and on a GPU
/// pipeline the CPU pipeline against it.
#[test]
fn physx_against_mujoco_cpu_is_measured() {
    const TICKS: u32 = 500;
    for probe in [
        PhysXBackend::is_available(),
        es_physics_backend::MuJoCoCpuBackend::is_available(),
    ] {
        if let Err(reason) = probe {
            eprintln!("SKIP physx_against_mujoco_cpu_is_measured: {reason}");
            return;
        }
    }
    for name in ["so101_pick_place.xml", "mesh_box.xml"] {
        let scene = fixture(name);
        // Every actuator swings across the middle half of its ctrlrange, out of phase.
        let ctrl: Vec<Vec<f64>> = (0..TICKS)
            .map(|k| {
                scene
                    .actuators
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        let (lo, hi) = a.ctrl_range.unwrap_or((-1.0, 1.0));
                        let phase = std::f64::consts::TAU * f64::from(k) / 100.0 + i as f64;
                        f64::midpoint(lo, hi) + 0.25 * (hi - lo) * phase.sin()
                    })
                    .collect()
            })
            .collect();
        let ctrl: &[Vec<f64>] = if scene.actuators.is_empty() {
            &[]
        } else {
            &ctrl
        };
        let compare = |a: &mut dyn PhysicsBackend, b: &mut dyn PhysicsBackend| {
            es_physics_backend::compare_backends(a, b, &scene, ctrl, TICKS).unwrap()
        };
        let mut px = PhysXBackend::new();
        let report = compare(&mut es_physics_backend::MuJoCoCpuBackend::new(), &mut px);
        eprintln!("{name}: {}\n{report}", px.engine_version().unwrap());
        assert!(report.max_dqpos.is_finite());
        let rerun = compare(&mut PhysXBackend::new(), &mut PhysXBackend::new());
        eprintln!(
            "{name}: PhysX rerun max |dqpos| {:e}, max |dqvel| {:e}, diverged at {:?}",
            rerun.max_dqpos, rerun.max_dqvel, rerun.divergence_tick
        );
        if px.capabilities().batch.gpu_resident {
            let pipelines = compare(&mut PhysXBackend::with_device("cpu"), &mut px);
            eprintln!(
                "{name}: PhysX CPU vs GPU pipeline max |dqpos| {:e}, diverged at {:?}",
                pipelines.max_dqpos, pipelines.divergence_tick
            );
        }
    }
}

/// The env's reset writes only the position of a free joint it randomizes and leaves the
/// quaternion zero; `MuJoCo` reads that as the identity. `PhysX` silently drops a transform with a
/// zero quaternion, so before the fix the cube never moved and reach A0 scored 0 (I1).
#[test]
fn physx_reset_state_moves_a_free_body_with_a_zero_quaternion() {
    if let Err(reason) = PhysXBackend::is_available() {
        eprintln!("SKIP physx_reset_state_moves_a_free_body_with_a_zero_quaternion: {reason}");
        return;
    }
    let scene = fixture("so101_pick_place.xml");
    let mut px = PhysXBackend::new();
    let info = px.load(&scene, &LoadConfig::default()).unwrap();
    let cube = scene.joints.iter().find(|j| j.name == "cube_free").unwrap();
    let at = info.qpos[&cube.id].start as usize;
    let mut qpos = vec![0.0; info.nq as usize];
    qpos[at..at + 3].copy_from_slice(&[0.3, 0.05, 0.02]);
    let qvel = vec![0.0; info.nv as usize];
    px.reset(
        None,
        Some(&StateView {
            n_envs: 1,
            qpos: &qpos,
            qvel: &qvel,
            ..StateView::default()
        }),
    )
    .unwrap();
    let got = &px.state().qpos[at..at + 7];
    let want = [0.3, 0.05, 0.02, 1.0, 0.0, 0.0, 0.0];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "cube qpos {got:?}, want {want:?}");
    }
}
