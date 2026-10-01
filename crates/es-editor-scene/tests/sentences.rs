//! Packet G8 (`docs/packets/M17/plan-g.md`): ①'s task as sentences, headless. The oracles: (1)
//! each committed specification read as sentences and written back slot by slot is the same
//! specification (its words are pinned in `crates/es-editor/tests/sentences.rs`); (2) every kind
//! of edit is one undo step that compiles; (3) a relation the subject does not take is not
//! offered, and a missing field, a time limit off the ticks and `touches` are refused by clause
//! and field, changing nothing; (4) "say the task" on the SO-101 copy, a region, two sentences,
//! and the documents generate; (5) the camera check sees, and misses what is turned away from or
//! hidden behind something.

use std::path::{Path, PathBuf};

use es_assets::esscene::ShapeDoc;
use es_editor_scene::sentence::{self as s, At, Field, Level, Look, Slot};
use es_editor_scene::{
    make_editable, new_geom, new_region, BackendKind, Command, Entity, Record, Regen, SceneModel,
};
use es_script::spec::vocab;
use es_script::spec::{compile_task, Relation, TaskSpec};

const SO101_SCENE: &str = "tests/fixtures/esscene/so101_pick_place.esscene";
const HAND_SCENE: &str = "tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.esscene";
const HAND_SPEC: &str = "tests/fixtures/estask/shadow_hand_repose.estask";
const VIEWS_SPEC: &str = "tests/fixtures/estask/so101_views.estask";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-g8-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

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

fn so101(name: &str) -> SceneModel {
    let root = scratch(name);
    make_editable(&root, &repo().join(SO101_SCENE), None).expect("copy");
    SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).expect("it opens")
}

fn spec(m: &SceneModel) -> TaskSpec {
    m.spec().expect("a specification").clone()
}

fn put(m: &mut SceneModel, spec: TaskSpec) -> Result<(), es_editor_scene::Refusal> {
    m.apply(&Command::Spec(Some(Box::new(spec))))
}

/// Every slot of every sentence written back as it reads, through the editor's own edits.
fn written_back(spec: &TaskSpec, scene: &es_assets::scene::SceneDesc) -> TaskSpec {
    let mut out = spec.clone();
    let sections = [(false, out.success.clauses.len())]
        .into_iter()
        .chain(out.failure.as_ref().map(|f| (true, f.clauses.len())));
    for (failure, n) in sections.collect::<Vec<_>>() {
        for index in 0..n {
            let c = if failure {
                &mut out.failure.as_mut().unwrap().clauses[index]
            } else {
                &mut out.success.clauses[index]
            };
            for (field, slot) in s::clause(scene, c).slots {
                s::edit_clause(scene, c, field, slot);
            }
            s::set_shaping(c, s::shaping_level(c));
        }
    }
    for item in out.start.iter_mut().flat_map(|st| st.items.iter_mut()) {
        for (field, slot) in s::start_item(scene, item).slots {
            s::edit_item(item, field, slot);
        }
    }
    s::set_bonus(&mut out, s::bonus(spec));
    s::set_strength(&mut out, s::strength(spec));
    s::set_look(&mut out, s::look(spec));
    out
}

/// Oracle 1: the two committed specifications, read as sentences and written back without an
/// edit, are equal to themselves field for field, and their weights read as levels.
#[test]
fn committed_specifications_round_trip_through_their_sentences() {
    for (scene, file) in [
        (HAND_SCENE, HAND_SPEC),
        ("tests/fixtures/mjcf/so101_pick_place_views.xml", VIEWS_SPEC),
    ] {
        let text = std::fs::read_to_string(repo().join(file)).unwrap();
        let spec = TaskSpec::from_toml(&text).unwrap();
        let path = repo().join(scene);
        let desc = if scene.ends_with(".esscene") {
            let doc =
                es_assets::esscene::EsScene::from_toml(&std::fs::read_to_string(&path).unwrap())
                    .unwrap();
            es_assets::esscene::expand(&doc, path.parent().unwrap()).unwrap()
        } else {
            let mut d = es_assets::parse_mjcf(&std::fs::read_to_string(&path).unwrap())
                .unwrap()
                .scene;
            es_assets::mesh::load(&mut d, path.parent().unwrap()).unwrap();
            d
        };
        let back = written_back(&spec, &desc);
        assert_eq!(back, spec, "{file}");
        assert_eq!(TaskSpec::from_toml(&back.to_toml().unwrap()).unwrap(), spec);
    }

    let text = std::fs::read_to_string(repo().join(HAND_SPEC)).unwrap();
    let hand = TaskSpec::from_toml(&text).unwrap();
    assert_eq!(
        s::shaping_level(&hand.success.clauses[0]),
        Some(Level::Medium)
    );
    let fail = &hand.failure.as_ref().unwrap().clauses[0];
    assert_eq!(s::shaping_level(fail), Some(Level::High));
    assert_eq!(s::bonus(&hand), Some(Level::High));
    assert_eq!(s::strength(&hand), Level::Medium);
    assert_eq!(s::look(&hand), Look::Traced);
    let text = std::fs::read_to_string(repo().join(VIEWS_SPEC)).unwrap();
    let views = TaskSpec::from_toml(&text).unwrap();
    assert_eq!(
        s::shaping_level(&views.success.clauses[0]),
        Some(Level::Medium)
    );
    assert_eq!(s::shaping_level(&views.success.clauses[1]), None);
    assert_eq!(s::bonus(&views), None);
    // A weight at no level is custom and kept as written.
    let mut odd = hand.success.clauses[0].clone();
    odd.weight = Some(0.37);
    assert_eq!(s::shaping_level(&odd), Some(Level::Custom(0.37)));
    s::set_shaping(&mut odd, Some(Level::Custom(0.37)));
    assert_eq!(odd.weight, Some(0.37));
}

/// One `Spec` command: applied, one more undo step, compiled; undo gives the previous back.
fn step(m: &mut SceneModel, edit: impl FnOnce(&mut TaskSpec, &es_assets::scene::SceneDesc)) {
    let before = spec(m);
    let mut next = before.clone();
    edit(&mut next, &m.scene().clone());
    assert_ne!(next, before, "the edit changes something");
    let rev = m.revision();
    put(m, next.clone()).unwrap_or_else(|r| panic!("{r:?}\n{}", next.to_toml().unwrap()));
    assert_eq!(m.spec(), Some(&next));
    assert!(m.revision() > rev, "the corner follows: the revision moves");
    assert!(m.undo());
    assert_eq!(m.spec(), Some(&before), "one undo step");
    assert!(m.redo());
    assert_eq!(m.spec(), Some(&next));
}

/// The specification's `task_hash` and `scene_hash` (only the specification is edited here, so
/// the saved scene is the scene).
fn hashes(m: &SceneModel) -> ([u8; 32], [u8; 32]) {
    let task = compile_task(&spec(m), m.root()).expect("compiles");
    (task.task_hash().unwrap(), task.scene.scene_hash)
}

/// Oracle 2: every kind of edit is one undo step that compiles; the success angle moves the
/// `task_hash` and not the `scene_hash`.
#[test]
fn every_edit_is_one_undo_step_that_compiles() {
    let mut m = hand("edits");
    let fail = At {
        failure: true,
        index: 0,
    };
    // Clauses: one added, moved, its relation, removed.
    step(&mut m, |sp, scene| s::add_clause(sp, scene, true));
    step(&mut m, |sp, _| {
        s::move_clause(
            sp,
            At {
                failure: true,
                index: 1,
            },
            false,
        );
    });
    step(&mut m, |sp, scene| {
        let c = &mut sp.failure.as_mut().unwrap().clauses[1];
        s::edit_clause(
            scene,
            c,
            Field::Relation,
            Slot::Relation(Relation::Near, true),
        );
    });
    step(&mut m, |sp, _| s::remove_clause(sp, fail));
    // A field, through its shown unit.
    let (task0, scene0) = hashes(&m);
    step(&mut m, |sp, scene| {
        let c = &mut sp.success.clauses[0];
        let v = s::Unit::DegAsIs.stored(8.0);
        s::edit_clause(
            scene,
            c,
            Field::WithinDeg,
            Slot::Number(Some(v), s::Unit::DegAsIs),
        );
    });
    let (task1, scene1) = hashes(&m);
    assert_ne!(task0, task1, "the angle moves the task_hash");
    assert_eq!(scene0, scene1, "and not the scene_hash");
    // Start items: the dice, the noise, the strength.
    step(&mut m, |sp, _| {
        sp.start.as_mut().unwrap().items[1].dice = Some(true);
    });
    step(&mut m, |sp, _| {
        let item = &mut sp.start.as_mut().unwrap().items[1];
        s::edit_item(item, Field::Noise, Slot::Number(Some(0.02), s::Unit::Cm));
    });
    step(&mut m, |sp, _| s::set_strength(sp, Level::High));
    // Observe: a camera, the look; reward: the bonus and a shaping level.
    step(&mut m, |sp, _| s::set_camera(sp, "side", false));
    step(&mut m, |sp, _| s::set_look(sp, Look::Quick));
    step(&mut m, |sp, _| s::add_source(sp, "object.qpos", true));
    step(&mut m, |sp, _| s::swap_source(sp, "object_qpos"));
    step(&mut m, |sp, _| s::remove_source(sp, "object_qpos"));
    step(&mut m, |sp, _| s::set_bonus(sp, Some(Level::Medium)));
    step(&mut m, |sp, _| {
        s::set_shaping(&mut sp.success.clauses[0], Some(Level::Low));
    });
    step(&mut m, s::add_item);
    let _ = std::fs::remove_dir_all(m.root());
}

/// A scripted edit of a specification.
type Edit = Box<dyn Fn(&mut TaskSpec)>;

/// Oracle 3: what a subject does not take is not offered; a missing field, a time limit off the
/// control ticks, `touches` and an empty success are refused naming the clause and the field,
/// and nothing changes. A scene edit that would break a sentence is refused too.
#[test]
fn refusals_name_the_clause_and_field_and_change_nothing() {
    let mut m = hand("refuse");
    let scene = m.scene().clone();
    let offered = |subject: &str| -> Vec<Relation> {
        vocab::relations(&scene, subject)
            .into_iter()
            .map(|(r, _)| r)
            .collect()
    };
    let coordinate = offered("object.x");
    assert!(coordinate.contains(&Relation::Inside) && coordinate.contains(&Relation::Still));
    for not in [
        Relation::Near,
        Relation::OrientationMatches,
        Relation::Touches,
    ] {
        assert!(!coordinate.contains(&not), "{not:?}");
    }
    let body = offered("object");
    assert!(body.contains(&Relation::OrientationMatches) && body.contains(&Relation::Touches));
    let palm = offered("robot0:palm");
    assert!(
        !palm.contains(&Relation::Still),
        "the palm has no free joint"
    );
    assert!(vocab::subjects(&scene).iter().any(|x| x == "object.z"));

    let start = spec(&m);
    let cases: [(&str, &str, Edit); 4] = [
        (
            "success[0].within_deg",
            s::REQUIRED,
            Box::new(|sp: &mut TaskSpec| sp.success.clauses[0].within_deg = None),
        ),
        (
            "timeout_s",
            s::TICKS,
            Box::new(|sp: &mut TaskSpec| sp.timeout_s = 8.01),
        ),
        (
            "success[0].relation",
            s::TOUCHES,
            Box::new(|sp: &mut TaskSpec| {
                let c = &mut sp.success.clauses[0];
                (c.relation, c.within_deg, c.shaping, c.weight, c.term) =
                    (Relation::Touches, None, None, None, None);
            }),
        ),
        (
            "success.clauses",
            s::NO_SUCCESS,
            Box::new(|sp: &mut TaskSpec| sp.success.clauses.clear()),
        ),
    ];
    for (field, key, edit) in cases {
        let mut next = start.clone();
        edit(&mut next);
        let r = put(&mut m, next).expect_err(field);
        assert_eq!((r.field.as_str(), r.key), (field, key), "{r:?}");
        assert_eq!(m.spec(), Some(&start), "{field}: nothing changed");
        assert!(!m.can_undo() && !m.dirty(), "{field}");
    }
    // The goal is a sentence's object: deleting it would break the specification.
    let r = m
        .apply(&Command::Delete(Entity::Body("target".into())))
        .expect_err("refused");
    assert_eq!(r.field, "observe.state.goal_pose", "{r:?}");
    assert!(!m.can_undo() && m.doc().bodies.iter().any(|b| b.name == "target"));
    // No specification at all is allowed: there is none, then.
    m.apply(&Command::Spec(None)).unwrap();
    assert!(m.spec().is_none());
    let _ = std::fs::remove_dir_all(m.root());
}

/// Oracle 4: a new task on the SO-101 copy — "say the task", a region over the bin, "[cube] is
/// inside [region]" and "[cube] is still" — compiles and generates its documents.
#[test]
fn a_new_task_on_the_so101_copy_compiles_and_generates() {
    let mut m = so101("new");
    assert!(m.spec().is_none());
    let rate = s::control_hz(&repo().join("tests/fixtures/visible-learning/task.toml"));
    assert_eq!(rate, Some(50.0));
    let said = s::new_spec(m.scene(), m.doc(), rate.unwrap());
    assert_eq!(said.robot, "arm");
    assert_eq!(
        (
            said.success.clauses[0].subject.as_str(),
            said.success.clauses[0].relation
        ),
        ("cube", Relation::Still)
    );
    m.apply(&Command::Spec(Some(Box::new(said))))
        .expect("it compiles");
    assert!(m.can_undo());

    let mut region = new_region("bin_area");
    region.pos = Some([0.14, -0.1, 0.05]);
    region.size = Some([0.05, 0.05, 0.05]);
    m.apply(&Command::Add(Record::Region(region))).unwrap();
    let scene = m.scene().clone();
    assert!(vocab::regions(&scene).iter().any(|r| r == "bin_area"));

    let mut next = spec(&m);
    s::add_clause(&mut next, &scene, false);
    let first = &mut next.success.clauses[0];
    s::edit_clause(
        &scene,
        first,
        Field::Relation,
        Slot::Relation(Relation::Inside, true),
    );
    assert_eq!(first.object.as_deref(), Some("bin_area"));
    put(&mut m, next).expect("cube inside bin_area, cube still");
    let sp = spec(&m);
    let words: Vec<(String, Relation)> = (sp.success.clauses.iter())
        .map(|c| (c.subject.clone(), c.relation))
        .collect();
    assert_eq!(
        words,
        [
            ("cube".into(), Relation::Inside),
            ("cube".into(), Relation::Still)
        ]
    );
    m.save().unwrap();
    match m.generated() {
        Regen::Written(files) => assert_eq!(files, &["task.toml"]),
        other => panic!("{other:?}"),
    }
    let _ = std::fs::remove_dir_all(m.root());
}

/// Oracle 5: a camera level with the cube, looking at it, sees it at its start; turned away, or
/// with a board between them, it does not, until the start lifts the cube over the board. The
/// overhead camera does not see it either: at the arm's start pose the lower arm is on the line
/// to the cube's centre.
#[test]
fn the_camera_check_sees_and_misses() {
    let mut m = so101("camera");
    let said = s::new_spec(m.scene(), m.doc(), 50.0);
    assert_eq!(
        said.observe.as_ref().unwrap().cameras.as_deref(),
        Some(&["overhead".to_owned()][..])
    );
    m.apply(&Command::Spec(Some(Box::new(said)))).unwrap();
    assert_eq!(
        m.unseen(),
        [("overhead".to_owned(), vec!["cube".to_owned()])]
    );

    // An eye 35 cm in front of the cube looking along +Y (a quarter turn about X).
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let mut eye = es_editor_scene::new_camera("eye");
    (eye.pos, eye.quat) = (Some([0.24, -0.35, 0.02]), Some([h, 0.0, 0.0, h]));
    m.apply(&Command::Add(Record::Camera(eye))).unwrap();
    let mut next = spec(&m);
    s::set_camera(&mut next, "overhead", false);
    s::set_camera(&mut next, "eye", true);
    put(&mut m, next).unwrap();
    assert_eq!(m.unseen(), [("eye".to_owned(), vec![])]);

    // Turned to look along −Y, away from the cube.
    let away = Command::SetPose {
        entity: Entity::Camera("eye".into()),
        pos: [0.24, -0.35, 0.02],
        quat: [-h, 0.0, 0.0, h],
    };
    m.apply(&away).unwrap();
    assert_eq!(m.unseen()[0].1, ["cube"]);
    m.undo();
    assert_eq!(m.unseen()[0].1, Vec::<String>::new());

    // A board between them.
    let mut board = new_geom(ShapeDoc::Box([0.05, 0.005, 0.05]));
    board.pos = Some([0.24, -0.2, 0.02]);
    m.apply(&Command::Add(Record::Scenery(board))).unwrap();
    assert_eq!(m.unseen()[0].1, ["cube"]);

    // Where the start puts the cube counts: 15 cm up, the line to it clears the board.
    let mut next = spec(&m);
    s::add_item(&mut next, &m.scene().clone());
    let item = &mut next.start.as_mut().unwrap().items[0];
    (item.what, item.value) = ("cube.z".into(), Some(0.15));
    put(&mut m, next).unwrap();
    assert_eq!(m.unseen()[0].1, Vec::<String>::new());
    let _ = std::fs::remove_dir_all(m.root());
}
