//! Packet G6 (`docs/packets/M17/plan-g.md`): the viewport's decisions, headless. The oracles of
//! the packet: (1) a pick on scripted rays selects the entity the ray meets first — the
//! segmentation channel's geom, mapped to its body, include, scenery or light — and passes
//! through what is hidden; (2) a gizmo drag on scripted rays makes the pose or size it should,
//! snapped to whole centimetres and 15 degrees; (3) a drag is exactly one undo step; and the
//! corner's camera is the one the documents declare.

// Snapped values are exact by construction; geometry keeps its own letters.
#![allow(clippy::float_cmp, clippy::many_single_char_names)]

use std::path::{Path, PathBuf};

use es_assets::esscene::ShapeDoc;
use es_editor_scene::gizmo::snap_cm;
use es_editor_scene::policy::{bundle, shown};
use es_editor_scene::view::{project, ray};
use es_editor_scene::{
    euler, make_editable, BackendKind, Camera, Command, Entity, Ray, Record, SceneModel, Step, Tool,
};
use es_ir::serial as s;
use es_ir::task::{ObsSource, SeedStream, SensorPath};
use es_math::approx::sin_cos_f64;
use es_math::units::DEG_TO_RAD;
use es_math::{Quat, Vec3};

const SO101_SCENE: &str = "tests/fixtures/esscene/so101_pick_place.esscene";
const HAND_SCENE: &str = "tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.esscene";
const HAND_SPEC: &str = "tests/fixtures/estask/shadow_hand_repose.estask";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-g6-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn so101(name: &str) -> SceneModel {
    let root = scratch(name);
    make_editable(&root, &repo().join(SO101_SCENE), None).expect("the copy");
    SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).expect("it opens")
}

fn hand(name: &str) -> SceneModel {
    let root = scratch(name);
    let spec = repo().join(HAND_SPEC);
    make_editable(&root, &repo().join(HAND_SCENE), Some(&spec)).expect("the copy");
    SceneModel::open(&root, vec![BackendKind::MuJoCoCpu]).expect("it opens")
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

fn v(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3::new(x, y, z)
}

fn down(x: f64, y: f64, z: f64) -> Ray {
    Ray {
        origin: v(x, y, z),
        dir: v(0.0, 0.0, -1.0),
    }
}

/// The centroid of `e`'s first drawn triangle.
fn centroid(m: &SceneModel, e: &Entity) -> Vec3 {
    let t = m.triangles(e)[0];
    let c = |k: usize| f64::from(t[0][k] + t[1][k] + t[2][k]) / 3.0;
    v(c(0), c(1), c(2))
}

/// The ray the viewport casts through the point of its picture where `p` lands.
fn through(camera: &Camera, p: Vec3) -> Ray {
    ray(camera, project(camera, p).expect("in front of the eye")).expect("a view")
}

/// Oracle 1: what a click selects, on scripted rays and on rays through the picture.
#[test]
fn a_pick_selects_what_the_ray_meets_first() {
    let mut m = hand("pick");
    let (object, target, hand) = (
        Entity::Body("object".into()),
        Entity::Body("target".into()),
        Entity::Include("hand".into()),
    );
    let light = Entity::Light("ceiling".into());
    // From the far side of the goal cube, at its height: the goal, whose face is nearest.
    let side = Ray {
        origin: v(2.0, 0.87, 0.2),
        dir: v(-1.0, 0.0, 0.0),
    };
    assert_eq!(m.hit(&side), Some(target.clone()));
    // Under the ceiling panel, looking up: the light. Above it: nothing at all.
    let up = |z| Ray {
        origin: v(1.0, 0.95, z),
        dir: v(0.0, 0.0, 1.0),
    };
    assert_eq!(m.hit(&up(0.85)), Some(light.clone()));
    assert_eq!(m.hit(&up(0.95)), None);

    // Each thing alone in view, through the viewport's own ray at a point of it: the hand's
    // parts select the include, a body's geom the body.
    let all = [&object, &target, &hand, &light];
    for e in all {
        for other in all.iter().filter(|o| **o != e) {
            m.toggle_hidden(other);
        }
        let at = centroid(&m, e);
        assert_eq!(m.hit(&through(&hand_view(), at)), Some(e.clone()), "{e:?}");
        // Hidden itself, the same ray goes through: nothing else is drawn.
        m.toggle_hidden(e);
        assert_eq!(m.hit(&through(&hand_view(), at)), None, "{e:?} hidden");
        for o in all {
            m.toggle_hidden(o);
        }
    }

    // The SO-101 copy: static scenery by its place in `[[geom]]`, a body, the arm include.
    let m = so101("pick");
    assert_eq!(
        m.hit(&down(0.4, 0.4, 0.3)),
        Some(Entity::Scenery(0)),
        "table"
    );
    assert_eq!(
        m.hit(&down(0.14, -0.1, 0.3)),
        Some(Entity::Scenery(1)),
        "bin floor"
    );
    // The gripper hangs over the cube: from above the ray meets the arm, from the side the cube.
    assert_eq!(
        m.hit(&down(0.24, 0.0, 0.3)),
        Some(Entity::Include("arm".into()))
    );
    let side = Ray {
        origin: v(0.6, 0.0, 0.01),
        dir: v(-1.0, 0.0, 0.0),
    };
    assert_eq!(m.hit(&side), Some(Entity::Body("cube".into())));
    // Straight down onto the arm's highest triangle, from under the ceiling panel.
    let arm = Entity::Include("arm".into());
    let tris = m.triangles(&arm);
    assert!(tris.len() > 100, "{} triangles", tris.len());
    let mid = |t: &[[f32; 3]; 3], k: usize| f64::from(t[0][k] + t[1][k] + t[2][k]) / 3.0;
    let top = (tris.iter())
        .max_by(|a, b| mid(a, 2).total_cmp(&mid(b, 2)))
        .unwrap();
    assert_eq!(m.hit(&down(mid(top, 0), mid(top, 1), 0.75)), Some(arm));
}

/// The cube of the hand copy, its move handle along X grabbed at 5 cm from its origin and
/// dragged on to `to` (metres along the handle), through the viewport's own rays.
fn move_drag(m: &SceneModel, axis: usize, to: f64) -> (es_editor_scene::Drag, Ray) {
    let e = Entity::Body("object".into());
    let cam = hand_view();
    let g = m.gizmo(&e, Tool::Move, &cam).expect("a gizmo");
    let at = |s: f64| through(&cam, g.origin + g.axes[axis].scale(s));
    let drag = m.drag(&e, g.clone(), axis, at(0.05)).expect("a drag");
    (drag, at(0.05 + to))
}

/// Oracle 2: a drag's pose and size, unsnapped and snapped; the handle under the pointer.
#[test]
fn a_gizmo_drag_moves_turns_and_sizes_with_snapping() {
    let m = hand("drag");
    let e = Entity::Body("object".into());
    let cam = hand_view();

    // The cube sits at (1.0, 0.87, 0.2), unturned, on the world: the handles are the world's.
    let g = m.gizmo(&e, Tool::Move, &cam).expect("a gizmo");
    assert_eq!(g.origin, v(1.0, 0.87, 0.2));
    assert_eq!(
        g.axes,
        [v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(0.0, 0.0, 1.0)]
    );
    // The pointer on 70 % of the X handle is on it; well off every handle it is on none.
    let on_x = project(&cam, g.origin + g.axes[0].scale(0.7 * g.size)).unwrap();
    assert_eq!(g.handle(&cam, on_x, 6.0), Some(0));
    assert_eq!(g.handle(&cam, [5.0, 5.0], 6.0), None);

    // Move: 5.37 cm along X is 1.0537 as dragged and 1.05 snapped, written as such.
    let (drag, now) = move_drag(&m, 0, 0.0537);
    let Some(Step::Move { from, to }) = drag.step(&now, false) else {
        panic!("a move")
    };
    assert_eq!(from, 1.0);
    assert!((to - 1.0537).abs() < 1e-9, "{to}");
    let snapped = drag.command(&now, true).expect("a command");
    assert_eq!(
        snapped,
        Command::SetPose {
            entity: e.clone(),
            pos: [1.05, 0.87, 0.2],
            quat: [0.0, 0.0, 0.0, 1.0],
        }
    );
    // Along Y, back 2.62 cm: 0.8438, snapped 0.84; within half a centimetre of where it was,
    // snapped, it changes nothing and makes no command.
    let (drag, now) = move_drag(&m, 1, -0.0262);
    let Some(Command::SetPose { pos, .. }) = drag.command(&now, true) else {
        panic!("a move")
    };
    assert_eq!(pos, [1.0, 0.84, 0.2]);
    assert_eq!(snap_cm(0.8438), 0.84);
    let (drag, now) = move_drag(&m, 1, 0.002);
    assert_eq!(drag.command(&now, true), None);
    // The ghost the viewport draws while dragging moves by what the command will write.
    let (drag, now) = move_drag(&m, 0, 0.0537);
    let step = drag.step(&now, true).unwrap();
    let ghost = drag.moved(step, v(1.03, 0.87, 0.23));
    assert!((ghost - v(1.08, 0.87, 0.23)).norm() < 1e-12, "{ghost:?}");

    // Turn about Z: grabbed on the ring at 10 degrees, let go at 47; 37 as dragged, 30 snapped.
    let g = m.gizmo(&e, Tool::Rotate, &cam).expect("a gizmo");
    let ring = |deg: f64| {
        let (s, c) = sin_cos_f64(deg * DEG_TO_RAD);
        through(
            &cam,
            g.origin + (g.axes[0].scale(c) + g.axes[1].scale(s)).scale(g.size),
        )
    };
    // Half way between X and Y: on the Z ring only (the rings cross on the axes).
    let half = (g.axes[0] + g.axes[1]).normalize().scale(g.size);
    assert_eq!(
        g.handle(&cam, project(&cam, g.origin + half).unwrap(), 4.0),
        Some(2)
    );
    let drag = m.drag(&e, g.clone(), 2, ring(10.0)).expect("a drag");
    let Some(Step::Rotate { angle }) = drag.step(&ring(47.0), false) else {
        panic!("a turn")
    };
    assert!((angle - 37.0 * DEG_TO_RAD).abs() < 1e-9, "{angle}");
    let Some(Command::SetPose { pos, quat, .. }) = drag.command(&ring(47.0), true) else {
        panic!("a turn")
    };
    assert_eq!(pos, [1.0, 0.87, 0.2]);
    let (s, c) = sin_cos_f64(30.0 * DEG_TO_RAD * 0.5);
    let want = Quat::from_xyzw(0.0, 0.0, s, c).normalize();
    assert_eq!(quat, [want.x, want.y, want.z, want.w]);
    let rpy = euler::deg_from_quat(quat);
    assert!(
        (rpy[2] - 30.0).abs() < 1e-9 && rpy[0] == 0.0 && rpy[1] == 0.0,
        "{rpy:?}"
    );

    // Size along X: grabbed at 4 cm, let go at 6 cm, so 1.5 times as wide: 6 cm to 9 cm.
    let g = m.gizmo(&e, Tool::Scale, &cam).expect("a gizmo");
    assert_eq!(g.on, [true; 3]);
    let at = |s: f64| through(&cam, g.origin + g.axes[0].scale(s));
    let drag = m.drag(&e, g.clone(), 0, at(0.04)).expect("a drag");
    let Some(Step::Scale { from, to }) = drag.step(&at(0.06), true) else {
        panic!("a size")
    };
    assert_eq!((from, to), (0.06, 0.09));
    let Some(Command::Set(_, Record::Body(b))) = drag.command(&at(0.06), true) else {
        panic!("a size")
    };
    assert_eq!(b.geoms[0].shape, ShapeDoc::Box([0.045, 0.03, 0.03]));
    // What has no size has no size handles; a camera turns and moves all the same.
    assert!(m
        .gizmo(&Entity::Include("hand".into()), Tool::Scale, &cam)
        .is_none());
    let top = Entity::Camera("top".into());
    assert!(m.gizmo(&top, Tool::Scale, &cam).is_none());
    assert_eq!(
        m.gizmo(&top, Tool::Move, &cam).map(|g| g.origin),
        Some(v(1.04, 0.88, 0.56))
    );
    // A region hangs from the cube: its handles start at the cube's place.
    let centre = Entity::Region("object:center".into());
    assert_eq!(
        m.gizmo(&centre, Tool::Move, &cam).map(|g| g.origin),
        Some(v(1.0, 0.87, 0.2))
    );
}

/// Oracle 3: a drag is one command, so one undo step, and undoing it restores the document byte
/// for byte; the snapped value is the number the document writes.
#[test]
fn a_drag_is_exactly_one_undo_step() {
    let mut m = hand("undo");
    let before = m.doc().to_toml().unwrap();
    let (drag, now) = move_drag(&m, 0, 0.0537);
    m.apply(&drag.command(&now, true).unwrap())
        .expect("applied");
    let after = m.doc().to_toml().unwrap();
    assert!(after.contains("pos = [1.05, 0.87, 0.2]"), "{after}");
    assert!(m.can_undo() && m.undo());
    assert!(!m.can_undo(), "more than one step");
    assert_eq!(m.doc().to_toml().unwrap(), before);
    assert!(m.redo());
    assert_eq!(m.doc().to_toml().unwrap(), after);
    // A size drag is one `Set`, likewise one step.
    let e = Entity::Body("object".into());
    let g = m.gizmo(&e, Tool::Scale, &hand_view()).unwrap();
    let at = |s: f64| through(&hand_view(), g.origin + g.axes[2].scale(s));
    let drag = m.drag(&e, g.clone(), 2, at(0.04)).unwrap();
    m.apply(&drag.command(&at(0.05), true).unwrap())
        .expect("applied");
    assert!(m.undo());
    assert_eq!(m.doc().to_toml().unwrap(), after);
}

/// Frame-selected: the camera turns to the middle of what is selected, from where it looked,
/// and all of it is in the picture.
#[test]
fn frame_selected_puts_the_selection_in_the_middle_of_the_picture() {
    let m = hand("frame");
    let e = Entity::Body("object".into());
    let cam = hand_view();
    let (centre, radius) = m.bounds(&e).expect("bounds");
    assert!((centre - v(1.0, 0.87, 0.2)).norm() < 1e-6, "{centre:?}");
    assert!((radius - 0.03 * 3f64.sqrt()).abs() < 1e-6, "{radius}");
    let framed = m.framed(&e, &cam).expect("framed");
    assert!(
        (Vec3::new(framed.look_at[0], framed.look_at[1], framed.look_at[2]) - centre).norm()
            < 1e-12
    );
    let middle = project(&framed, centre).unwrap();
    assert!((middle[0] - 320.0).abs() < 1e-6 && (middle[1] - 200.0).abs() < 1e-6);
    for t in m.triangles(&e) {
        for p in t {
            let [x, y] = project(&framed, v(p[0].into(), p[1].into(), p[2].into())).unwrap();
            assert!(
                (0.0..640.0).contains(&x) && (0.0..400.0).contains(&y),
                "{x} {y}"
            );
        }
    }
    // A camera draws nothing: framing it is framing its place.
    let top = m.bounds(&Entity::Camera("top".into())).unwrap();
    assert_eq!(top, (v(1.04, 0.88, 0.56), 0.1));
}

/// The corner: the hand project's policy sees its three cameras exactly as the committed Task IR
/// declares them (96 x 96, path traced at 32 samples, exposure 8, a seed per tick); the template's
/// bundle names the same three; the corner shows the selected one when the policy sees it.
#[test]
fn the_corner_shows_the_camera_the_documents_declare() {
    let mut m = hand("corner");
    let cams = m.policy_cameras().expect("the specification compiles");
    let names: Vec<&str> = cams.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["top", "front", "side"]);
    let read = |p: &str| std::fs::read_to_string(repo().join(p)).unwrap();
    let task = s::task_from_toml(&read("tests/fixtures/shadow-hand/task-repose.toml")).unwrap();
    for c in &cams {
        let ch = &task.observation_spec.channels[&format!("rgb_{}", c.name)];
        let ObsSource::Sensor { id, render, .. } = &ch.source else {
            panic!("{}", c.name)
        };
        assert_eq!((c.sensor, c.render), (*id, *render), "{}", c.name);
        assert_eq!(Some(&c.image), ch.ty.image.as_ref(), "{}", c.name);
        assert_eq!((c.image.width, c.image.height), (96, 96));
        assert_eq!(
            c.render.path,
            SensorPath::Pt {
                spp: 32,
                bounces: 3
            }
        );
        assert_eq!((c.render.exposure, c.render.seed), (8.0, SeedStream::Tick));
    }
    let pick = |e: Option<Entity>| shown(&cams, e.as_ref()).map(|c| c.name.clone());
    assert_eq!(
        pick(Some(Entity::Camera("front".into()))).as_deref(),
        Some("front")
    );
    assert_eq!(
        pick(Some(Entity::Body("object".into()))).as_deref(),
        Some("top")
    );
    assert_eq!(pick(None).as_deref(), Some("top"));

    // The template: what its bundle's Observation IR reads, as its Task IR declares it.
    let docs = |task: &str, obs: &str| [repo().join(task), repo().join(obs)];
    let [task, obs] = docs(
        "tests/fixtures/shadow-hand/task-repose.toml",
        "tests/fixtures/shadow-hand/observation-student.toml",
    );
    assert_eq!(bundle(&task, &obs, m.scene()), Ok(cams));
    // The SO-101 cube card's camera-only bundle.
    let so = so101("corner");
    let [task, obs] = docs(
        "tests/fixtures/visible-learning/task.toml",
        "tests/fixtures/visible-learning/observation-v8.toml",
    );
    let cams = bundle(&task, &obs, so.scene()).expect("the bundle reads");
    let names: Vec<&str> = cams.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["overhead"]);
    // A file that is not there is said by its path.
    let missing = bundle(Path::new("no/such/task.toml"), &obs, so.scene());
    assert!(missing.is_err_and(|e| e.contains("no/such/task.toml")));
}
