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
    // Packet R5: a newly picked "inside a region" pays its distance to the centre, at medium.
    assert_eq!(
        (first.shaping, s::shaping_level(first)),
        (
            Some(es_script::spec::Shaping::Distance),
            Some(Level::Medium)
        )
    );
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
    // Packet M17/G9: "say the task" also says who learns it, so the whole set is generated.
    let all = [
        "task.toml",
        "observation-teacher.toml",
        "learning-teacher.toml",
        "deployment-teacher.toml",
        "evaluation-teacher.toml",
        "training-teacher.toml",
        "observation-student.toml",
        "learning-student.toml",
        "deployment-student.toml",
        "evaluation-student.toml",
        "evaluation-student-nominal.toml",
        "training-student.toml",
        "cycle-student.toml",
    ];
    match m.generated() {
        Regen::Written(files) => assert_eq!(files, &all),
        other => panic!("{other:?}"),
    }
    // What is on disk is what the saved documents generate; a changed one on disk is not.
    assert_eq!(es_editor_scene::on_disk(m.root()), *m.generated());
    let spec_path = m.root().join(es_editor_scene::SPEC_FILE);
    let text = std::fs::read_to_string(&spec_path).unwrap();
    std::fs::write(
        &spec_path,
        text.replace("timeout_s = 8.0", "timeout_s = 9.0"),
    )
    .unwrap();
    assert_eq!(es_editor_scene::on_disk(m.root()), Regen::Stale);
    m.refresh();
    assert_eq!(m.generated(), &Regen::Stale);
    std::fs::write(&spec_path, text.replace("execute = 10", "execute = 3")).unwrap();
    let failed = es_editor_scene::on_disk(m.root());
    assert!(
        matches!(&failed, Regen::Failed(why) if why.contains("execute")),
        "{failed:?}"
    );
    std::fs::write(&spec_path, &text).unwrap();
    assert!(matches!(
        es_editor_scene::on_disk(m.root()),
        Regen::Written(_)
    ));
    std::fs::remove_file(&spec_path).unwrap();
    assert_eq!(es_editor_scene::on_disk(m.root()), Regen::NoSpec);
    let _ = std::fs::remove_dir_all(m.root());
}

/// Packet R5 (design note section 4.7.1): an "inside a region" clause offers the distance
/// toggle; turning it on at a level and off is one undo step that compiles, it reads back at
/// its level through its sentence, and re-picking the relation it has does not turn it back on.
#[test]
fn the_region_distance_toggle_is_one_undo_step() {
    let mut m = so101("region");
    let said = s::new_spec(m.scene(), m.doc(), 50.0);
    m.apply(&Command::Spec(Some(Box::new(said)))).unwrap();
    let mut region = new_region("bin_area");
    region.pos = Some([0.14, -0.1, 0.05]);
    m.apply(&Command::Add(Record::Region(region))).unwrap();
    let scene = m.scene().clone();
    let inside = Slot::Relation(Relation::Inside, true);
    step(&mut m, |sp, scene| {
        s::edit_clause(
            scene,
            &mut sp.success.clauses[0],
            Field::Relation,
            inside.clone(),
        );
    });
    assert!(s::shapable(&spec(&m).success.clauses[0]).is_some());
    for level in [Some(Level::High), None, Some(Level::Low)] {
        step(&mut m, |sp, _| {
            s::set_shaping(&mut sp.success.clauses[0], level);
        });
        let now = spec(&m);
        assert_eq!(s::shaping_level(&now.success.clauses[0]), level);
        assert_eq!(written_back(&now, &scene), now, "{level:?}");
    }
    let mut off = spec(&m).success.clauses[0].clone();
    s::set_shaping(&mut off, None);
    s::edit_clause(&scene, &mut off, Field::Relation, inside);
    assert_eq!(off.shaping, None, "the same relation picked again");
    let _ = std::fs::remove_dir_all(m.root());
}

/// Packet M17/G9: "say the task" says who learns it — the robot's joints, velocities and last
/// action and the first free body's pose and velocity observed (that body's for the teacher
/// alone), a medium success bonus at plan H's scale, plan H's PPO teacher over every channel, a
/// camera student over every observed camera reading the joints, and a cycle the trained
/// teacher demonstrates, its successes kept.
#[test]
fn say_the_task_says_who_learns_it() {
    let m = so101("learners");
    let said = s::new_spec(m.scene(), m.doc(), 50.0);
    let o = said.observe.as_ref().unwrap();
    let names = |t: &Option<std::collections::BTreeMap<String, String>>| -> Vec<(String, String)> {
        t.iter()
            .flatten()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    };
    let pair = |k: &str, v: &str| (k.to_owned(), v.to_owned());
    assert_eq!(
        names(&o.state),
        [
            pair("joint_pos", "robot.joint_pos"),
            pair("joint_vel", "robot.joint_vel"),
            pair("previous_action", "robot.previous_action"),
        ]
    );
    assert_eq!(
        names(&o.privileged),
        [pair("cube_pose", "cube.pose"), pair("cube_vel", "cube.vel")]
    );
    let r = said.reward.as_ref().unwrap();
    assert_eq!(
        (r.scale, r.success),
        (Some(s::REWARD_SCALE), Some(s::BONUS[1]))
    );
    assert_eq!(said.teacher, Some(es_script::spec::Teacher::default()));
    let st = said.student.as_ref().unwrap();
    assert_eq!(
        (st.views.clone(), st.state.clone(), st.horizon, st.execute),
        (None, Some(vec!["joint_pos".to_owned()]), 16, 10)
    );
    let c = said.cycle.as_ref().unwrap();
    assert_eq!(
        (c.expert.clone(), c.episodes, c.seed, c.success_only),
        (None, s::EPISODES, s::SEED, Some(true))
    );
    // At 60 Hz the chunk still replans at a whole rate; at 25 Hz too.
    for (hz, execute) in [(60.0, 10), (25.0, 5), (7.0, 7)] {
        let said = s::new_spec(m.scene(), m.doc(), hz);
        assert_eq!(said.student.unwrap().execute, execute, "{hz}");
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

/// Packet M17/G9: on the empty scene, the library's SO-101 and a box from the Add menu — whose
/// free joint is unnamed, so it is named as the box — "say the task" compiles: the box still is
/// the box's speed, not a joint's (the compiler reads only a hinge or slide joint by name).
#[test]
fn say_the_task_on_an_object_from_the_add_menu() {
    let root = scratch("added");
    let empty = repo().join("tests/fixtures/esscene/empty.esscene");
    make_editable(&root, &empty, None).expect("copy");
    let mut m = SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).unwrap();
    let lib = es_editor_scene::add::library(&repo()).unwrap();
    let so101 = lib.into_iter().find(|r| r.id == "so101").unwrap();
    let view = |at: [f64; 3]| es_editor_scene::Camera {
        eye: [at[0] + 0.5, at[1] - 0.5, 0.6],
        look_at: at,
        fov_y: std::f64::consts::FRAC_PI_4,
        width: 640,
        height: 400,
    };
    m.add(&es_editor_scene::Item::Robot(so101), &view([0.0; 3]), true)
        .unwrap();
    let object = es_editor_scene::inspect::ShapeKind::Box;
    m.add(
        &es_editor_scene::Item::Object(object),
        &view([0.3, 0.3, 0.0]),
        true,
    )
    .unwrap();
    let Some(Entity::Body(name)) = m.selection().cloned() else {
        panic!("a body")
    };
    assert!(
        m.scene().joints.iter().any(|j| j.name == name),
        "named as its body"
    );
    let said = s::new_spec(m.scene(), m.doc(), s::CONTROL_HZ);
    let c = &said.success.clauses[0];
    assert_eq!(
        (c.subject.as_str(), c.relation),
        (name.as_str(), Relation::Still)
    );
    put(&mut m, said).expect("it compiles");
    let _ = std::fs::remove_dir_all(root);
}
