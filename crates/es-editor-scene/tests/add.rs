//! Packet G7 (`docs/packets/M17/plan-g.md`): ①'s Add, headless. The oracles of the packet: (1)
//! each Add item, on the Shadow Hand copy and on `empty.esscene`, expands and maps onto
//! `mujoco-cpu`, ends up selected and is one undo step whose undo restores the bytes; (2) a box
//! added on a scripted centre ray rests on the SO-101 table, and on no hit at the origin; (3)
//! files are imported by content: once, with the source's asset hashes, a broken one refused with
//! nothing left behind, and a document naming an import round-trips; (4) the library's SO-101 is
//! `so101.xml` at its pose; (5) an include's override is written and cleared; (6) cameras and
//! regions are picked by their lines on screen. And a picture becomes a geom's material in one
//! command.

// Placements are exact by construction; geometry keeps its own letters.
#![allow(clippy::float_cmp, clippy::many_single_char_names)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use es_assets::esscene::{expand, EsScene, Include};
use es_assets::scene::{AssetKind, SceneDesc};
use es_editor_scene::add::library;
use es_editor_scene::gizmo::snap_cm;
use es_editor_scene::import::{self, ASSETS};
use es_editor_scene::inspect::ShapeKind;
use es_editor_scene::overrides::prune;
use es_editor_scene::view::{project, ray};
use es_editor_scene::{
    make_editable, BackendKind, Camera, Command, Entity, Item, Record, Robot, SceneModel,
};
use es_math::{Pose, Vec3};

const SO101_SCENE: &str = "tests/fixtures/esscene/so101_pick_place.esscene";
const SO101_ARM: &str = "tests/fixtures/esscene/so101.xml";
const EMPTY_SCENE: &str = "tests/fixtures/esscene/empty.esscene";
const HAND_SCENE: &str = "tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.esscene";
const HAND_FILE: &str = "tests/fixtures/mjcf/shadow_hand/shadow_hand.xml";
const HAND_SPEC: &str = "tests/fixtures/estask/shadow_hand_repose.estask";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-g7-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open(name: &str, scene: &str, spec: Option<&str>) -> SceneModel {
    let root = scratch(name);
    let spec = spec.map(|s| repo().join(s));
    make_editable(&root, &repo().join(scene), spec.as_deref()).expect("the copy");
    SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).expect("it opens")
}

fn bytes(m: &SceneModel) -> String {
    m.doc().to_toml().unwrap()
}

/// The hand template's own viewport (`templates/shadow-hand-repose.toml`).
fn hand_view() -> Camera {
    Camera {
        eye: [1.5, 0.4, 0.6],
        look_at: [1.02, 0.95, 0.2],
        fov_y: std::f64::consts::FRAC_PI_4,
        width: 640,
        height: 400,
    }
}

/// A view whose centre ray goes from `eye` through `at`.
fn towards(eye: [f64; 3], at: [f64; 3]) -> Camera {
    Camera {
        eye,
        look_at: at,
        fov_y: std::f64::consts::FRAC_PI_4,
        width: 640,
        height: 400,
    }
}

fn centre(view: &Camera) -> es_editor_scene::Ray {
    ray(view, [320.0, 200.0]).expect("a view")
}

/// Every file under `dir`, relative to it.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                todo.push(p);
            } else {
                out.push(p.strip_prefix(dir).unwrap().to_path_buf());
            }
        }
    }
    out.sort();
    out
}

fn library_robot(id: &str) -> Robot {
    let lib = library(&repo()).expect("the library reads");
    lib.into_iter().find(|r| r.id == id).expect(id)
}

/// Every item the menu offers, files from the repository's fixtures.
fn items() -> Vec<Item> {
    let fixtures = repo().join("tests/fixtures");
    let mut out: Vec<Item> = [
        ShapeKind::Box,
        ShapeKind::Sphere,
        ShapeKind::Cylinder,
        ShapeKind::Capsule,
    ]
    .into_iter()
    .map(Item::Object)
    .collect();
    out.extend([
        Item::Fixed(ShapeKind::Box),
        Item::Mesh(fixtures.join("mjcf/meshes/box.stl")),
        Item::Robot(library_robot("so101")),
        Item::Robot(library_robot("shadow_hand")),
        Item::Robot(Robot::file(&fixtures.join("urdf/arm2.urdf"))),
        Item::Robot(Robot::file(
            &fixtures.join("gltf/textured_box/textured_box.gltf"),
        )),
        Item::Camera,
        Item::Light,
        Item::Region,
    ]);
    out
}

/// Whether `e` is what `item` makes.
fn made_by(item: &Item, e: &Entity) -> bool {
    matches!(
        (item, e),
        (Item::Object(_) | Item::Mesh(_), Entity::Body(_))
            | (Item::Fixed(_), Entity::Scenery(_))
            | (Item::Robot(_), Entity::Include(_))
            | (Item::Camera, Entity::Camera(_))
            | (Item::Light, Entity::Light(_))
            | (Item::Region, Entity::Region(_))
    )
}

/// Oracle 1: every item, on the Shadow Hand copy and on the empty scene.
#[test]
fn every_item_is_one_selected_step_that_expands() {
    let cases = [
        ("hand", HAND_SCENE, hand_view()),
        ("empty", EMPTY_SCENE, towards([0.0, -1.2, 1.2], [0.0; 3])),
    ];
    for (name, scene, view) in cases {
        let mut m = open(&format!("every-{name}"), scene, None);
        for item in items() {
            let before = bytes(&m);
            m.add(&item, &view, true)
                .unwrap_or_else(|r| panic!("{name}: {item:?}: {r:?}"));
            let after = bytes(&m);
            let e = m.selection().cloned().expect("selected");
            assert!(made_by(&item, &e), "{name}: {item:?} selected {e:?}");
            assert!(m.record(&e).is_some(), "{name}: {e:?}");
            // The document as it now is expands and maps onto the reference backend.
            let again = es_editor_scene::check::check(m.doc(), m.root(), &[BackendKind::MuJoCoCpu]);
            assert!(again.is_ok(), "{name}: {item:?}: {again:?}");
            assert!(m.undo(), "{name}: {item:?} is a step");
            assert_eq!(
                bytes(&m),
                before,
                "{name}: {item:?}: undo restores the bytes"
            );
            assert!(m.redo());
            assert_eq!(bytes(&m), after, "{name}: {item:?}: redo");
            m.undo();
        }
    }
}

/// Oracle 2: resting on the SO-101 table under a scripted centre ray; at the origin on no hit.
#[test]
fn a_new_box_rests_where_the_view_looks() {
    let mut m = open("rests", SO101_SCENE, None);
    let view = towards([0.6, 0.6, 0.4], [0.353, 0.247, 0.0]);
    let (hit, n) = m.surface(&centre(&view)).expect("the table");
    assert_eq!([n.x, n.y, n.z], [0.0, 0.0, 1.0], "the table is level");
    assert!(hit.z.abs() < 1e-9, "on the table: {hit:?}");
    for snap in [false, true] {
        m.add(&Item::Object(ShapeKind::Box), &view, snap).unwrap();
        let Some(Entity::Body(b)) = m.selection().cloned() else {
            panic!("a body")
        };
        let body = m.doc().bodies.iter().find(|x| x.name == b).unwrap();
        let [x, y, z] = body.pos.unwrap();
        let half = 0.025; // the new box's half height
        assert!(
            (z - half - hit.z).abs() < 1e-9,
            "its bottom at the hit: {z}"
        );
        if snap {
            assert_eq!([x, y], [snap_cm(hit.x), snap_cm(hit.y)]);
            assert_eq!([x, y], [0.35, 0.25]);
        } else {
            assert_eq!([x, y], [hit.x, hit.y]);
        }
        // The expansion agrees: the box's lowest point is on the table.
        let s = m.scene();
        let sb = s.bodies.iter().find(|x| x.name == b).unwrap();
        assert!((sb.pose.position.z - half - hit.z).abs() < 1e-9);
        // Taken away again: the next one would rest on it.
        m.undo();
    }
    // Into the sky: nothing is hit, so the origin, resting on the ground plane.
    let up = towards([0.0, 0.0, 1.0], [1.0, 0.0, 2.0]);
    assert_eq!(m.surface(&centre(&up)), None);
    m.add(&Item::Object(ShapeKind::Box), &up, true).unwrap();
    let Some(Entity::Body(b)) = m.selection().cloned() else {
        panic!("a body")
    };
    let body = m.doc().bodies.iter().find(|x| x.name == b).unwrap();
    assert_eq!(body.pos, Some([0.0, 0.0, 0.025]));
}

/// Each mesh asset's content hash, by name.
fn mesh_hashes(s: &SceneDesc) -> BTreeMap<String, [u8; 32]> {
    (s.assets.iter())
        .filter(|a| a.kind == AssetKind::Mesh)
        .map(|a| (a.name.clone(), a.hash))
        .collect()
}

/// `file` (in `dir`) included alone, as its reader gives it.
fn alone(dir: &Path, file: &str) -> SceneDesc {
    let mut doc = import::empty();
    doc.includes.push(Include {
        name: "x".into(),
        source: file.into(),
        prefix: None,
        pos: None,
        quat: None,
        set: None,
    });
    expand(&doc, dir).unwrap()
}

/// Oracle 3: by content, once; the source's hashes; nothing left by a refusal; a round trip.
#[test]
fn imports_are_copied_by_content() {
    let root = scratch("import");
    for src in [SO101_ARM, HAND_FILE] {
        let src = repo().join(src);
        let (path, first) = import::robot(&root, &src).unwrap();
        let hash = blake3::hash(&std::fs::read(&src).unwrap()).to_hex();
        let name = src.file_name().unwrap().to_string_lossy();
        assert_eq!(path, format!("{ASSETS}/{hash}/{name}"));
        assert!(first.0.iter().any(|p| p.ends_with(&*name)), "{:?}", first.0);
        let listed = files(&root);
        let (again, second) = import::robot(&root, &src).unwrap();
        assert_eq!(again, path);
        assert!(
            second.0.is_empty(),
            "the second copy writes nothing: {:?}",
            second.0
        );
        assert_eq!(files(&root), listed);
    }
    // The hand's twelve meshes came along under `meshes/`, as its `meshdir` names them.
    let meshes = files(&root)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "stl"));
    assert_eq!(meshes.count(), 12);

    // Added to an empty scene, the hand's meshes hash as the source's do.
    let mut m = open("import-hash", EMPTY_SCENE, None);
    let view = towards([0.0, -1.2, 1.2], [0.0; 3]);
    m.add(&Item::Robot(library_robot("shadow_hand")), &view, true)
        .unwrap();
    let dir = repo().join(HAND_FILE);
    let source = alone(dir.parent().unwrap(), "shadow_hand.xml");
    assert_eq!(mesh_hashes(m.scene()), mesh_hashes(&source));
    assert_eq!(mesh_hashes(&source).len(), 12);
    for a in m
        .scene()
        .assets
        .iter()
        .filter(|a| a.kind == AssetKind::Mesh)
    {
        assert!(a.path.starts_with(ASSETS), "{}", a.path);
    }

    // Broken files: refused, nothing added, nothing written.
    let junk = scratch("junk");
    std::fs::write(junk.join("broken.xml"), "<mujoco><worldbody><body name=").unwrap();
    std::fs::write(junk.join("broken.stl"), b"not a mesh").unwrap();
    let (before, listed) = (bytes(&m), files(m.root()));
    let can_undo = m.can_undo();
    m.undo();
    let undone = bytes(&m);
    m.redo();
    for item in [
        Item::Robot(Robot::file(&junk.join("broken.xml"))),
        Item::Mesh(junk.join("broken.stl")),
    ] {
        let refused = m.add(&item, &view, true).expect_err("refused");
        assert!(!refused.args.is_empty(), "{refused:?} says why");
        assert_eq!(bytes(&m), before, "{item:?}: nothing added");
        assert_eq!(files(m.root()), listed, "{item:?}: no new file");
        assert_eq!(m.can_undo(), can_undo);
    }
    m.undo();
    assert_eq!(bytes(&m), undone, "the refusals made no undo step");
    m.redo();

    // A document naming an imported mesh survives a save and a read.
    let stl = repo().join("tests/fixtures/mjcf/meshes/box.stl");
    m.add(&Item::Mesh(stl.clone()), &view, true).unwrap();
    let Some(Entity::Body(b)) = m.selection().cloned() else {
        panic!("a body")
    };
    let geom = &m.doc().bodies.iter().find(|x| x.name == b).unwrap().geoms[0];
    let hash = blake3::hash(&std::fs::read(&stl).unwrap()).to_hex();
    assert_eq!(
        geom.shape,
        es_assets::esscene::ShapeDoc::Mesh(format!("{ASSETS}/{hash}.stl"))
    );
    m.save().unwrap();
    let text = std::fs::read_to_string(m.root().join(es_editor_scene::SCENE_FILE)).unwrap();
    assert_eq!(&EsScene::from_toml(&text).unwrap(), m.doc());
    let reopened = SceneModel::open(m.root(), vec![BackendKind::MuJoCoCpu]).unwrap();
    assert_eq!(reopened.doc(), m.doc());
    assert_eq!(mesh_hashes(reopened.scene()), mesh_hashes(m.scene()));
}

/// Oracle 4: the library's SO-101 on the empty scene is `so101.xml`, placed at its pose.
#[test]
fn the_library_so101_is_its_file_at_its_pose() {
    let mut m = open("so101", EMPTY_SCENE, None);
    let view = towards([0.5, -0.5, 0.6], [0.1, 0.1, 0.0]);
    m.add(&Item::Robot(library_robot("so101")), &view, true)
        .unwrap();
    let Some(Entity::Include(name)) = m.selection().cloned() else {
        panic!("an include")
    };
    assert_eq!(name, "so101");
    let inc = &m.doc().includes[0];
    let [x, y, z] = inc.pos.unwrap();
    assert_eq!(
        [x, y, z],
        [0.1, 0.1, 0.0],
        "where the view looks, on the floor"
    );

    let src = es_assets::parse_mjcf(&std::fs::read_to_string(repo().join(SO101_ARM)).unwrap())
        .unwrap()
        .scene;
    let s = m.scene();
    let names = |s: &SceneDesc| -> Vec<String> {
        s.bodies.iter().skip(1).map(|b| b.name.clone()).collect()
    };
    assert_eq!(names(s), names(&src));
    let joints =
        |s: &SceneDesc| -> Vec<_> { s.joints.iter().map(|j| (j.name.clone(), j.range)).collect() };
    assert_eq!(joints(s), joints(&src));
    let actuators = |s: &SceneDesc| -> Vec<_> {
        (s.actuators.iter())
            .map(|a| (a.name.clone(), a.ctrl_range, a.force_range))
            .collect()
    };
    assert_eq!(actuators(s), actuators(&src));
    let place = Pose::new(Vec3::new(x, y, z), es_math::Quat::IDENTITY);
    let base = |s: &SceneDesc| s.bodies.iter().find(|b| b.name == "base").unwrap().pose;
    assert_eq!(base(s), place.compose(base(&src)));

    // A second one meets the first's names: it gets its handle as a prefix.
    m.add(&Item::Robot(library_robot("so101")), &view, true)
        .unwrap();
    let second = &m.doc().includes[1];
    assert_eq!(
        (second.name.as_str(), second.prefix.as_deref()),
        ("so101_2", Some("so101_2:"))
    );
}

/// Oracle 5: an included joint's range overridden, shown by the expansion, and cleared.
#[test]
fn an_override_is_written_and_cleared() {
    let mut m = open("override", SO101_SCENE, None);
    let e = Entity::Include("arm".into());
    let file = m.brought("arm").expect("the arm reads");
    let pan = |b: &es_editor_scene::overrides::Brought| {
        (b.joints.iter())
            .find(|j| j.0 == "shoulder_pan")
            .map(|j| j.2.range)
            .unwrap()
    };
    let range = pan(&file).expect("the file limits it");
    assert!(
        file.actuators.iter().any(|a| a.1.kp.is_some()),
        "position gains"
    );
    assert!(!file.geoms.is_empty());

    let Some(Record::Include(mut inc)) = m.record(&e) else {
        panic!("the include")
    };
    let mut set = inc.set.clone().unwrap_or_default();
    set.joint.entry("shoulder_pan".into()).or_default().range = Some([-1.0, 1.0]);
    inc.set = prune(Some(set.clone()));
    m.apply(&Command::Set(e.clone(), Record::Include(inc.clone())))
        .unwrap();
    assert!(
        bytes(&m).contains("[include.set.joint.shoulder_pan]"),
        "{}",
        bytes(&m)
    );
    let expanded = |m: &SceneModel| {
        (m.scene().joints.iter())
            .find(|j| j.name == "shoulder_pan")
            .unwrap()
            .range
    };
    assert_eq!(expanded(&m), Some((-1.0, 1.0)));
    // What the file says is still what the file says.
    assert_eq!(pan(&m.brought("arm").unwrap()), Some(range));

    // Cleared, the field drops its override and the empty table goes with it.
    set.joint.get_mut("shoulder_pan").unwrap().range = None;
    inc.set = prune(Some(set));
    assert_eq!(inc.set, None);
    m.apply(&Command::Set(e, Record::Include(inc))).unwrap();
    assert!(!bytes(&m).contains("include.set"), "{}", bytes(&m));
    assert_eq!(expanded(&m), Some((range[0], range[1])));
}

/// Oracle 6: a camera's and a region's lines pick them on screen; elsewhere picks neither.
#[test]
fn markers_are_picked_on_screen() {
    let mut m = open("markers", HAND_SCENE, Some(HAND_SPEC));
    let view = hand_view();
    let marked: Vec<Entity> = m.markers(&view).into_iter().map(|(e, _)| e).collect();
    for c in ["top", "front", "side"] {
        assert!(
            marked.contains(&Entity::Camera(c.into())),
            "{c}: {marked:?}"
        );
    }
    // At a camera's eye, where its frustum's lines meet.
    let (parent, local) = m.placement(&Entity::Camera("front".into())).unwrap();
    let at = project(&view, parent.compose(local).position).expect("in view");
    assert_eq!(
        m.marker_at(&view, at, 4.0),
        Some(Entity::Camera("front".into()))
    );
    // A region's corner, and a point a few pixels off its edge.
    let region = es_editor_scene::new_region("zone");
    let region = es_assets::esscene::RegionDoc {
        pos: Some([1.0, 0.8, 0.1]),
        size: Some([0.05; 3]),
        ..region
    };
    m.apply(&Command::Add(Record::Region(region))).unwrap();
    let corner = project(&view, Vec3::new(1.05, 0.85, 0.15)).unwrap();
    let zone = Some(Entity::Region("zone".into()));
    assert_eq!(m.marker_at(&view, corner, 4.0), zone);
    assert_eq!(m.marker_at(&view, [corner[0] + 3.0, corner[1]], 4.0), zone);
    // A region draws nothing, so the ray through it finds no region.
    let through = ray(&view, corner).unwrap();
    assert_ne!(m.hit(&through), zone);
    // Far from every line: nothing.
    assert_eq!(m.marker_at(&view, [2.0, 398.0], 4.0), None);
}

/// A picture made the cube's material: one command, a texture and a material named by the file,
/// reused when the same picture comes again.
#[test]
fn a_picture_becomes_a_material_in_one_step() {
    let mut m = open("picture", SO101_SCENE, None);
    let png = repo().join("tests/fixtures/gltf/textured_box/base.png");
    let before = bytes(&m);
    let cube = Entity::Body("cube".into());
    m.texture(&cube, 0, &png).unwrap();
    let hash = blake3::hash(&std::fs::read(&png).unwrap()).to_hex();
    let doc = m.doc();
    assert_eq!(doc.textures.len(), 1);
    assert_eq!(
        doc.textures[0].file.as_deref(),
        Some(&*format!("{ASSETS}/{hash}.png"))
    );
    assert_eq!(doc.materials[0].texture.as_deref(), Some("base"));
    let geom = &doc.bodies.iter().find(|b| b.name == "cube").unwrap().geoms[0];
    assert_eq!(geom.material.as_deref(), Some("base"));
    let decoded = (m.scene().assets.iter())
        .find(|a| a.kind == AssetKind::Texture && a.name == "base")
        .unwrap();
    // Decoded by content: the hash of the same picture read where it was.
    let mut source = import::empty();
    let json = serde_json::json!({ "name": "base", "file": "base.png" });
    source.textures.push(serde_json::from_value(json).unwrap());
    let source = expand(&source, png.parent().unwrap()).unwrap();
    assert_eq!(decoded.hash, source.assets[0].hash, "decoded by content");
    assert!(m.undo());
    assert_eq!(bytes(&m), before, "one step");
    m.redo();
    // The same picture on the table: the same texture and material.
    m.texture(&Entity::Scenery(0), 0, &png).unwrap();
    assert_eq!((m.doc().textures.len(), m.doc().materials.len()), (1, 1));
    assert_eq!(m.doc().geoms[0].material.as_deref(), Some("base"));
    // Not a picture: refused, nothing written.
    let listed = files(m.root());
    let stl = repo().join("tests/fixtures/mjcf/meshes/box.stl");
    assert!(m.texture(&cube, 0, &stl).is_err());
    assert_eq!(files(m.root()), listed);
}
