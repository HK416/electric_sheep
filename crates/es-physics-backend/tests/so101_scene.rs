//! The SO-101 demo scene loads whole, rather than warning (spec 4.3, spec 17.2).
//!
//! `es_assets::parse_mjcf` -> `mjcf_out::scene_to_mjcf` -> `MuJoCoCpuBackend::load`. The
//! primitives-only derivative exists so that **nothing in it is unmappable**: a scene item a
//! backend cannot map is `PhysicsError::Unsupported` by name, and this test proves there is
//! no such item rather than tolerating a warning.
//!
//! Without a Python interpreter carrying `mujoco`, the load half prints
//! `SKIP so101_scene: <why>`; `every_scene_item_is_mappable` needs no Python and always runs.
//!
//!     ES_PYTHON=$HOME/venvs/es/bin/python cargo test -p es-physics-backend \
//!       --test so101_scene -- --nocapture

use es_assets::scene::{JointKind, SceneDesc};
use es_core::TickRate;
use es_physics_backend::{scene_to_mjcf, MuJoCoCpuBackend};
use es_physics_core::{LoadConfig, ModelInfo, PhysicsBackend};

const FIXTURE: &str = "so101_pick_place.xml";

fn scene() -> SceneDesc {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/mjcf/");
    let xml = std::fs::read_to_string(format!("{path}{FIXTURE}"))
        .unwrap_or_else(|e| panic!("cannot read {FIXTURE}: {e}"));
    let import = es_assets::parse_mjcf(&xml).unwrap_or_else(|e| panic!("{FIXTURE}: {e}"));
    assert!(
        import.warnings.is_empty(),
        "{FIXTURE} parses with warnings: {:#?}",
        import.warnings
    );
    import.scene
}

/// `nq` / `nv` `MuJoCo` must report, derived from the parsed scene rather than assumed: a hinge
/// is one coordinate and one dof, a free joint is seven coordinates and six dofs.
fn expected_dofs(scene: &SceneDesc) -> (u32, u32) {
    scene
        .joints
        .iter()
        .fold((0, 0), |(nq, nv), j| match j.kind {
            JointKind::Free => (nq + 7, nv + 6),
            JointKind::Ball => (nq + 4, nv + 3),
            JointKind::Fixed => (nq, nv),
            JointKind::Hinge | JointKind::Slide => (nq + 1, nv + 1),
        })
}

fn load() -> Option<(SceneDesc, ModelInfo)> {
    if let Err(reason) = MuJoCoCpuBackend::is_available() {
        println!("SKIP so101_scene: {reason}");
        return None;
    }
    let scene = scene();
    let mut backend = MuJoCoCpuBackend::new();
    let info = backend
        .load(
            &scene,
            &LoadConfig {
                n_envs: 1,
                rate: Some(TickRate::hz(200)),
                seed: 1,
            },
        )
        .expect("the demo scene loads in MuJoCo");
    println!("RAN so101_scene");
    Some((scene, info))
}

#[test]
fn the_demo_scene_loads_in_mujoco() {
    let Some((scene, info)) = load() else { return };

    let (nq, nv) = expected_dofs(&scene);
    assert_eq!(info.nq, nq, "nq (6 arm hinges + the cube's free joint)");
    assert_eq!(info.nv, nv);
    assert_eq!(info.nu, 6, "the six position servos");
    assert_eq!(info.n_envs, 1);

    // Every actuator the scene declares resolves to a `ctrl` slot.
    assert_eq!(info.actuator.len(), scene.actuators.len());
    for a in &scene.actuators {
        let range = info
            .actuator
            .get(&a.id)
            .unwrap_or_else(|| panic!("actuator `{}` has no ctrl slot", a.name));
        assert_eq!(range.len, 1, "actuator `{}`", a.name);
        assert!(range.start < info.nu, "actuator `{}` out of range", a.name);
    }

    // The cube's free joint owns the last seven `qpos` entries: the arm comes first in the
    // body tree, so `qpos[6]`, `qpos[7]` and `qpos[8]` are the cube's x, y and z -- the
    // indices the Task IR's randomization names.
    let cube = scene
        .joints
        .iter()
        .find(|j| j.name == "cube_free")
        .expect("the scene has a `cube_free` joint");
    let range = info.qpos[&cube.id];
    assert_eq!((range.start, range.len), (6, 7), "cube free joint qpos");
    assert_eq!(info.dof[&cube.id].start, 6, "cube free joint qvel");

    // Every body, including the world, has a row in `xpos`/`xquat`.
    assert_eq!(info.nbody as usize, scene.bodies.len());
    for b in &scene.bodies {
        assert!(
            info.body.contains_key(&b.id),
            "body `{}` has no row",
            b.name
        );
    }
}

#[test]
fn every_scene_item_is_mappable() {
    let scene = scene();
    let mjcf = scene_to_mjcf(&scene).expect("no geom, actuator or sensor is Unsupported");
    // The emitted MJCF is what MuJoCo actually reads: it must carry no mesh and no include.
    assert!(!mjcf.contains("mesh"), "the emitted MJCF references a mesh");
    assert!(!mjcf.contains("<include"));
    for name in ["cube_free", "shoulder_pan", "gripper", "table", "bin_floor"] {
        assert!(mjcf.contains(name), "the emitted MJCF lost `{name}`");
    }
}
