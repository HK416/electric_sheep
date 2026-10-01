//! Packet G5 (`docs/packets/M17/plan-g.md`): the editor's scene model, headless. The oracles of
//! the packet: (1) any command sequence undone returns the identical document and redone the
//! same as applying it (a property test); (2) an invalid edit is refused with the field named and
//! changes nothing; (3) a rename rewrites `task.estask` and the regenerated documents still
//! validate; (4) a saved document re-reads equal; (5) an editable copy of a template moves no
//! hash but the ones that name the scene file itself.

use std::path::{Path, PathBuf};

use es_assets::esscene::{EsScene, ShapeDoc};
use es_editor_scene::{
    make_editable, BackendKind, Command, Entity, Record, Regen, SceneModel, GENERATED_DIR,
    SCENE_FILE, SPEC_FILE,
};
use es_ir::deployment::RobotTarget;
use es_ir::serial as s;
use proptest::prelude::*;

const SO101_SCENE: &str = "tests/fixtures/esscene/so101_pick_place.esscene";
const HAND_SCENE: &str = "tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.esscene";
const HAND_SPEC: &str = "tests/fixtures/estask/shadow_hand_repose.estask";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-g5-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A project holding an editable copy of the SO-101 cube scene (no task specification).
fn so101(name: &str) -> SceneModel {
    let root = scratch(name);
    make_editable(&root, &repo().join(SO101_SCENE), None).expect("the copy");
    SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).expect("it opens")
}

/// A project holding an editable copy of the Shadow Hand scene and its task specification.
fn hand(name: &str) -> SceneModel {
    let root = scratch(name);
    make_editable(
        &root,
        &repo().join(HAND_SCENE),
        Some(&repo().join(HAND_SPEC)),
    )
    .expect("copy");
    SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).expect("it opens")
}

fn bytes(m: &SceneModel) -> String {
    m.doc().to_toml().unwrap()
}

/// One abstract edit; its numbers pick among whatever the document holds when it is applied.
#[derive(Clone, Debug)]
enum Op {
    AddBox(f64),
    AddCamera,
    AddLight,
    AddRegion,
    Delete(usize),
    Duplicate(usize),
    Mass(usize, f64),
    Pose(usize, [f64; 3], [f64; 3]),
    Reparent(usize, Option<usize>),
    Rename(usize, u8),
    Hide(usize),
    Undo,
}

fn op() -> impl Strategy<Value = Op> {
    let xyz = || prop::array::uniform3(-1.0..1.0f64);
    prop_oneof![
        (0.005..0.1f64).prop_map(Op::AddBox),
        Just(Op::AddCamera),
        Just(Op::AddLight),
        Just(Op::AddRegion),
        any::<usize>().prop_map(Op::Delete),
        any::<usize>().prop_map(Op::Duplicate),
        (any::<usize>(), -0.5..2.0f64).prop_map(|(i, m)| Op::Mass(i, m)),
        (
            any::<usize>(),
            xyz(),
            prop::array::uniform3(-170.0..170.0f64)
        )
            .prop_map(|(i, p, e)| Op::Pose(i, p, e)),
        (any::<usize>(), prop::option::of(any::<usize>())).prop_map(|(i, p)| Op::Reparent(i, p)),
        (any::<usize>(), 0..6u8).prop_map(|(i, n)| Op::Rename(i, n)),
        any::<usize>().prop_map(Op::Hide),
        Just(Op::Undo),
    ]
}

fn pick<T: Clone>(list: &[T], i: usize) -> Option<T> {
    (!list.is_empty()).then(|| list[i % list.len()].clone())
}

/// The concrete command `op` means on `m`'s document now; `None` for nothing to pick.
fn command(m: &SceneModel, op: &Op) -> Option<Command> {
    let entities: Vec<Entity> = m.rows().into_iter().filter_map(|r| r.entity).collect();
    let bodies: Vec<String> = m.doc().bodies.iter().map(|b| b.name.clone()).collect();
    Some(match op {
        Op::AddBox(h) => Command::Add(Record::Body(es_editor_scene::new_body(
            &m.unique("box"),
            ShapeDoc::Box([*h; 3]),
        ))),
        Op::AddCamera => Command::Add(Record::Camera(es_editor_scene::new_camera(
            &m.unique("camera"),
        ))),
        Op::AddLight => Command::Add(Record::Light(es_editor_scene::new_light(
            &m.unique("light"),
        ))),
        Op::AddRegion => Command::Add(Record::Region(es_editor_scene::new_region(
            &m.unique("region"),
        ))),
        Op::Delete(i) => Command::Delete(pick(&entities, *i)?),
        Op::Duplicate(i) => Command::Duplicate(pick(&entities, *i)?),
        Op::Mass(i, mass) => {
            let name = pick(&bodies, *i)?;
            let mut body = m.doc().bodies.iter().find(|b| b.name == name)?.clone();
            body.geoms.first_mut()?.mass = Some(*mass);
            Command::Set(Entity::Body(name), Record::Body(body))
        }
        Op::Pose(i, pos, rpy) => Command::SetPose {
            entity: pick(&entities, *i)?,
            pos: *pos,
            quat: es_editor_scene::euler::quat_from_deg(*rpy),
        },
        Op::Reparent(i, p) => Command::Reparent(
            Entity::Body(pick(&bodies, *i)?),
            p.and_then(|p| pick(&bodies, p)),
        ),
        Op::Rename(i, n) => Command::Rename(pick(&entities, *i)?, format!("thing{n}")),
        Op::Hide(_) | Op::Undo => return None,
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 40, ..ProptestConfig::default() })]

    /// Oracle 1: whatever sequence of commands (refused ones included, undos mixed in), the
    /// model's stack is the reference stack below: undoing every step gives back the document's
    /// bytes and a clean model, and redoing every step gives the bytes of applying every command
    /// that was not undone, each step in between the same document both ways.
    #[test]
    fn undo_all_is_the_start_and_redo_all_is_the_end(ops in prop::collection::vec(op(), 1..14)) {
        let mut m = so101("prop");
        let start = bytes(&m);
        // The reference: the documents on the undo path (the start first), and the ones a redo
        // reaches (the next one last).
        let (mut done, mut undone) = (vec![start.clone()], Vec::<String>::new());
        for op in &ops {
            match op {
                Op::Undo => {
                    if m.undo() {
                        undone.push(done.pop().expect("a step"));
                    }
                    prop_assert_eq!(&bytes(&m), done.last().unwrap());
                }
                Op::Hide(i) => {
                    let entities: Vec<Entity> =
                        m.rows().into_iter().filter_map(|r| r.entity).collect();
                    if let Some(e) = pick(&entities, *i) {
                        m.toggle_hidden(&e);
                    }
                    prop_assert_eq!(&bytes(&m), done.last().unwrap());
                }
                op => {
                    let Some(cmd) = command(&m, op) else { continue };
                    let before = bytes(&m);
                    match m.apply(&cmd) {
                        // Refused: nothing changed.
                        Err(_) => prop_assert_eq!(bytes(&m), before),
                        Ok(()) if bytes(&m) != before => {
                            done.push(bytes(&m));
                            undone.clear();
                        }
                        Ok(()) => {}
                    }
                }
            }
        }
        let tip: Vec<String> = done.iter().chain(undone.iter().rev()).cloned().collect();
        let mut seen = vec![bytes(&m)];
        while m.redo() {
            seen.push(bytes(&m));
        }
        while m.undo() {
            seen.push(bytes(&m));
        }
        prop_assert_eq!(&bytes(&m), &start);
        prop_assert!(!m.dirty());
        let mut forth = vec![start.clone()];
        while m.redo() {
            forth.push(bytes(&m));
        }
        prop_assert_eq!(&forth, &tip);
        prop_assert_eq!(m.dirty(), tip.last() != Some(&start));
        // Down from the tip, every state in reverse.
        let down: Vec<String> = seen[seen.len() - tip.len()..].to_vec();
        prop_assert_eq!(down, tip.iter().rev().cloned().collect::<Vec<_>>());
        let _ = std::fs::remove_dir_all(m.root());
    }
}

/// Oracle 2: each invalid edit is refused naming the field, in words of the editor's tables, and
/// leaves the document, the undo stack and the dirty mark as they were.
#[test]
fn invalid_edits_are_refused_by_field_and_change_nothing() {
    let mut m = so101("refuse");
    let start = bytes(&m);
    let cube = m.doc().bodies[0].clone();
    assert_eq!(cube.name, "cube");

    let mut heavy = cube.clone();
    heavy.geoms[0].mass = Some(-0.2);
    let mut painted = cube.clone();
    painted.geoms[0].material = Some("no-such-paint".into());
    let mut thin = cube.clone();
    thin.geoms[0].shape = ShapeDoc::Box([0.01, 0.0, 0.01]);
    let cases = [
        (
            Command::Set(Entity::Body("cube".into()), Record::Body(heavy)),
            "body[cube].geom[cube_geom].mass",
            "author.refused.negative",
        ),
        (
            Command::Reparent(Entity::Body("cube".into()), Some("nowhere".into())),
            "body[cube].parent",
            "author.refused.parent",
        ),
        (
            Command::Set(Entity::Body("cube".into()), Record::Body(painted)),
            "body[cube].geom[cube_geom].material",
            "author.refused.missing",
        ),
        (
            Command::Set(Entity::Body("cube".into()), Record::Body(thin)),
            "body[cube].geom[cube_geom].shape",
            "author.refused.positive",
        ),
        (
            Command::Rename(Entity::Light("ceiling".into()), String::new()),
            "light[ceiling].name",
            "author.refused.empty",
        ),
    ];
    for (cmd, field, key) in cases {
        let refusal = m.apply(&cmd).expect_err(field);
        assert_eq!(
            (refusal.field.as_str(), refusal.key),
            (field, key),
            "{cmd:?}"
        );
        assert_eq!(bytes(&m), start, "{cmd:?} changed the document");
        assert!(!m.dirty() && !m.can_undo(), "{cmd:?}");
    }

    // A duplicate name: a copy of the cube, renamed back to the cube.
    m.apply(&Command::Duplicate(Entity::Body("cube".into())))
        .unwrap();
    assert_eq!(m.selection(), Some(&Entity::Body("cube_2".into())));
    let before = bytes(&m);
    let refusal = m
        .apply(&Command::Rename(
            Entity::Body("cube_2".into()),
            "cube".into(),
        ))
        .unwrap_err();
    assert_eq!(
        (refusal.field.as_str(), refusal.key, refusal.args.as_slice()),
        (
            "body[cube].name",
            "author.refused.duplicate",
            &["cube".to_owned()][..]
        )
    );
    assert_eq!(bytes(&m), before);

    // A body hung from its own child.
    m.apply(&Command::Reparent(
        Entity::Body("cube_2".into()),
        Some("cube".into()),
    ))
    .unwrap();
    let refusal = m
        .apply(&Command::Reparent(
            Entity::Body("cube".into()),
            Some("cube_2".into()),
        ))
        .unwrap_err();
    assert_eq!(
        (refusal.field.as_str(), refusal.key),
        ("body[cube].parent", "author.refused.cycle")
    );
    let _ = std::fs::remove_dir_all(m.root());
}

/// Oracle 3: renaming the cube rewrites every place `task.estask` names it, and after the save
/// the documents regenerate (every one validated by `generate`) from the renamed scene.
#[test]
fn a_rename_rewrites_the_spec_and_the_documents_regenerate() {
    let mut m = hand("rename");
    m.apply(&Command::Rename(
        Entity::Body("object".into()),
        "cube".into(),
    ))
    .unwrap();
    let spec = m.spec().expect("the spec").clone();
    let text = spec.to_toml().unwrap();
    // Every name of the cube, and nothing else (`object_vel` is a channel, not the cube).
    assert!(
        !text.contains("\"object\"") && !text.contains("\"object."),
        "{text}"
    );
    assert_eq!(spec.success.clauses[0].subject, "cube");
    assert_eq!(spec.failure.as_ref().unwrap().clauses[0].subject, "cube");
    let start = spec.start.as_ref().unwrap();
    assert!(start.items.iter().any(|i| i.what == "cube.x"));
    assert!(start.items.iter().any(|i| i.what == "cube.orientation"));
    let observe = spec.observe.as_ref().unwrap();
    assert_eq!(
        observe.privileged.as_ref().unwrap()["cube_pose"],
        "cube.pose"
    );
    // The goal and the region on the cube follow; nothing else of the scene moved.
    let region = m.doc().regions.iter().find(|r| r.name == "object:center");
    assert_eq!(region.unwrap().parent.as_deref(), Some("cube"));
    assert_eq!(spec.success.clauses[0].object.as_deref(), Some("target"));

    m.save().expect("saved");
    assert!(!m.dirty());
    match m.generated() {
        Regen::Written(files) => assert!(files.iter().any(|f| f == "task.toml"), "{files:?}"),
        other => panic!("{other:?}"),
    }
    let task = std::fs::read_to_string(m.root().join(GENERATED_DIR).join("task.toml")).unwrap();
    let task = s::task_from_toml(&task).unwrap();
    assert!(task.validate().is_empty());
    assert_eq!(task.scene.path, SCENE_FILE);

    // Undo puts the old names back in both documents.
    assert!(m.undo());
    assert_eq!(m.spec().unwrap().success.clauses[0].subject, "object");
    assert!(m.dirty());
    let _ = std::fs::remove_dir_all(m.root());
}

/// Oracle 4: what is saved re-reads to the same document and specification, clean; the write is
/// atomic (no temporary is left beside it).
#[test]
fn a_saved_document_reads_back_equal() {
    let mut m = hand("save");
    m.apply(&Command::Rename(
        Entity::Body("object".into()),
        "cube".into(),
    ))
    .unwrap();
    let mut body = m.doc().bodies[0].clone();
    body.geoms[0].shape = ShapeDoc::Box([0.05; 3]);
    body.geoms[0].material = None;
    body.geoms[0].rgba = Some([0.9, 0.2, 0.15, 1.0]);
    m.apply(&Command::Set(
        Entity::Body("cube".into()),
        Record::Body(body),
    ))
    .unwrap();
    m.apply(&Command::Add(Record::Camera(es_editor_scene::new_camera(
        "extra",
    ))))
    .unwrap();
    assert!(m.dirty());
    m.save().unwrap();
    let again = SceneModel::open(m.root(), vec![BackendKind::MuJoCoCpu]).unwrap();
    assert_eq!(again.doc(), m.doc());
    assert_eq!(again.spec(), m.spec());
    assert!(!again.dirty());
    let text = std::fs::read_to_string(m.root().join(SCENE_FILE)).unwrap();
    assert_eq!(&EsScene::from_toml(&text).unwrap(), m.doc());
    let leftovers: Vec<_> = std::fs::read_dir(m.root())
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    let _ = std::fs::remove_dir_all(m.root());
}

fn hex(h: [u8; 32]) -> String {
    blake3::Hash::from_bytes(h).to_hex().to_string()
}

/// Oracle 5: an editable copy of the Shadow Hand template, nothing edited, generates the
/// template's committed documents but for what names the scene *file*: the Task IR's
/// `scene.path` (`scene.esscene` now) and `asset_hash` (that file's bytes) — its `scene_hash` and
/// everything else equal — and the Deployment IR's simulated robot target (the same path); and,
/// through the hash chain, the Task IR's hash where the others cite it (an Observation IR's
/// `task_ref`, an Evaluation IR's `task` and `observation`). The Learning IRs cite neither and
/// hash as committed. The SO-101 cube template has no task specification; its copied scene
/// expands to the template scene's `scene_hash`.
#[test]
fn an_editable_copy_moves_nothing_but_the_scene_files_name() {
    let mut m = hand("copy");
    assert!(es_editor_scene::is_editable(m.root()));
    m.regenerate();
    let Regen::Written(files) = m.generated() else {
        panic!("{:?}", m.generated())
    };
    assert_eq!(files.len(), 13, "{files:?}");
    let got = |f: &str| std::fs::read_to_string(m.root().join(GENERATED_DIR).join(f)).unwrap();
    let committed = |f: &str| {
        std::fs::read_to_string(repo().join("tests/fixtures/shadow-hand").join(f)).unwrap()
    };

    let task = s::task_from_toml(&got("task.toml")).unwrap();
    let mut template = s::task_from_toml(&committed("task-repose.toml")).unwrap();
    assert_eq!(task.scene.scene_hash, template.scene.scene_hash);
    assert_eq!(task.scene.path, SCENE_FILE);
    let copied = std::fs::read(m.root().join(SCENE_FILE)).unwrap();
    assert_eq!(copied, std::fs::read(repo().join(HAND_SCENE)).unwrap());
    template.scene.path = task.scene.path.clone();
    template.scene.asset_hash = task.scene.asset_hash;
    let task_hash = task.task_hash().unwrap();
    assert_eq!(task_hash, template.task_hash().unwrap());

    for (file, against) in [
        ("observation-teacher.toml", "observation-teacher.toml"),
        ("observation-student.toml", "observation-student.toml"),
    ] {
        let ours = s::observation_from_toml(&got(file)).unwrap();
        let mut theirs = s::observation_from_toml(&committed(against)).unwrap();
        assert_eq!(ours.task_ref, task_hash, "{file}");
        theirs.task_ref = task_hash;
        let (a, b) = (ours.observation_hash(), theirs.observation_hash());
        assert_eq!(a.unwrap(), b.unwrap(), "{file}");
    }
    for (file, against) in [
        ("learning-teacher.toml", "learning-teacher.toml"),
        ("learning-student.toml", "learning-student.toml"),
    ] {
        let h = |t: &str| hex(s::learning_from_toml(t).unwrap().learning_hash().unwrap());
        assert_eq!(h(&got(file)), h(&committed(against)), "{file}");
    }
    for (file, against) in [
        ("deployment-teacher.toml", "deployment-hand.toml"),
        ("deployment-student.toml", "deployment-student.toml"),
    ] {
        let ours = s::deployment_from_toml(&got(file)).unwrap();
        let mut theirs = s::deployment_from_toml(&committed(against)).unwrap();
        let scene = RobotTarget::Simulated {
            scene: SCENE_FILE.to_owned(),
        };
        assert_eq!(ours.robot.target, scene, "{file}");
        theirs.robot.target = scene;
        let (a, b) = (ours.deployment_hash(), theirs.deployment_hash());
        assert_eq!(a.unwrap(), b.unwrap(), "{file}");
    }
    for (file, against) in [
        ("evaluation-teacher.toml", "evaluation-teacher-64.toml"),
        ("evaluation-student.toml", "evaluation-student.toml"),
        (
            "evaluation-student-nominal.toml",
            "evaluation-student-nominal.toml",
        ),
    ] {
        let ours = s::evaluation_from_toml(&got(file)).unwrap();
        let mut theirs = s::evaluation_from_toml(&committed(against)).unwrap();
        assert_eq!(ours.task, hex(task_hash), "{file}");
        theirs.task.clone_from(&ours.task);
        theirs.observation.clone_from(&ours.observation);
        assert_eq!(
            ours.evaluation_hash().unwrap(),
            theirs.evaluation_hash().unwrap(),
            "{file}"
        );
    }
    let _ = std::fs::remove_dir_all(m.root());

    let cube = so101("copy-cube");
    let template =
        es_tools_free_scene_hash(&repo().join("tests/fixtures/mjcf/so101_pick_place.xml"));
    assert_eq!(cube.scene().scene_hash(), template);
    assert!(cube.spec().is_none() && cube.root().join("so101.xml").is_file());
    assert!(!cube.root().join(SPEC_FILE).exists());
    let _ = std::fs::remove_dir_all(cube.root());
}

/// The template's MJCF scene as `load_scene` reads it: parsed, meshes and textures loaded.
fn es_tools_free_scene_hash(xml: &Path) -> [u8; 32] {
    let text = std::fs::read_to_string(xml).unwrap();
    let mut scene = es_assets::parse_mjcf(&text).unwrap().scene;
    es_assets::mesh::load(&mut scene, xml.parent().unwrap()).unwrap();
    scene.scene_hash()
}

/// A copy refuses a project that is already editable, and a project made editable once is
/// left as it was.
#[test]
fn a_second_copy_is_refused() {
    let m = so101("twice");
    let err = make_editable(m.root(), &repo().join(SO101_SCENE), None).unwrap_err();
    assert!(err.contains(SCENE_FILE), "{err}");
    let _ = std::fs::remove_dir_all(m.root());
}

/// The hierarchy of the Shadow Hand copy: the include folded over the hand's bodies (read-only
/// parts), the cube and the goal with their geoms and the regions hanging from them, the three
/// cameras and the light; a search keeps the rows a match folds under. Hiding changes what the
/// viewport draws and never the document.
#[test]
fn the_hierarchy_folds_includes_and_hiding_is_the_viewports_only() {
    use es_editor_scene::RowKind as K;
    let mut m = hand("rows");
    let rows = m.rows();
    let at = |name: &str| rows.iter().position(|r| r.name == name).expect(name);
    let hand = at("hand");
    assert_eq!(
        (rows[hand].kind, rows[hand].depth),
        (K::Include, 0),
        "{rows:?}"
    );
    let mount = at("robot0:hand mount");
    assert_eq!(
        (rows[mount].kind, rows[mount].entity.as_ref()),
        (K::Part, None)
    );
    assert_eq!(rows[mount].parent, Some(hand));
    let palm = at("robot0:palm");
    assert!(rows[palm].depth > rows[mount].depth, "{:?}", rows[palm]);
    let cube = at("object");
    assert_eq!(rows[cube].kind, K::Body);
    // The cube's geom is named as the body is; the row under the body is the geom.
    let geom = rows
        .iter()
        .position(|r| r.parent == Some(cube) && r.kind == K::Geom);
    assert!(geom.is_some(), "{rows:?}");
    assert_eq!(rows[at("object:center")].parent, Some(cube));
    for camera in ["top", "front", "side"] {
        assert_eq!(rows[at(camera)].kind, K::Camera);
    }
    assert_eq!(rows[at("ceiling")].kind, K::Light);
    assert_eq!(rows[0].kind, K::Physics);

    let show = es_editor_scene::tree::matches(&rows, "PALM");
    assert!(
        show[palm] && show[mount] && show[hand],
        "the path down to the match"
    );
    assert!(!show[cube] && !show[0]);
    assert!(es_editor_scene::tree::matches(&rows, " ")
        .iter()
        .all(|s| *s));

    let before = m.doc().clone();
    let all = m.drawn();
    let drawn_geoms =
        |s: &es_assets::scene::SceneDesc| -> usize { s.bodies.iter().map(|b| b.geoms.len()).sum() };
    m.toggle_hidden(&Entity::Include("hand".into()));
    let no_hand = m.drawn();
    m.toggle_hidden(&Entity::Body("object".into()));
    let neither = m.drawn();
    let cube_geoms = all
        .bodies
        .iter()
        .find(|b| b.name == "object")
        .unwrap()
        .geoms
        .len();
    assert!(drawn_geoms(&no_hand) < drawn_geoms(&all) / 2);
    assert_eq!(drawn_geoms(&no_hand) - drawn_geoms(&neither), cube_geoms);
    assert!(no_hand
        .bodies
        .iter()
        .any(|b| b.name == "target" && !b.geoms.is_empty()));
    assert_eq!(m.doc(), &before);
    assert!(!m.dirty() && !m.can_undo());
    m.toggle_hidden(&Entity::Include("hand".into()));
    m.toggle_hidden(&Entity::Body("object".into()));
    assert_eq!(drawn_geoms(&m.drawn()), drawn_geoms(&all));
    let _ = std::fs::remove_dir_all(m.root());
}
