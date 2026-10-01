//! Packet G3a (`docs/packets/M17/plan-g.md`): the task specification `*.estask` compiles to the
//! Task IR. The committed documents are the oracle: the Shadow Hand and SO-101 views specs in
//! `tests/fixtures/estask/` compile to the `task_hash` of `tests/fixtures/shadow-hand/
//! task-repose.toml` and `tests/fixtures/visible-learning/task-views.toml`, which this file only
//! reads.

use std::path::{Path, PathBuf};

use es_ir::task::TaskIr;
use es_script::spec::{compile_task, TaskSpec};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    let path = repo().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn spec(rel: &str) -> TaskSpec {
    TaskSpec::from_toml(&read(rel)).expect("the spec reads")
}

fn hex(d: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*d).to_hex().to_string()
}

/// Every node and channel of one task that the other does not have, for a failing hash.
fn diff(got: &TaskIr, want: &TaskIr) -> String {
    let lines = |t: &TaskIr| -> Vec<String> {
        let mut v: Vec<String> = t.graph.nodes.values().map(|n| format!("{n:?}")).collect();
        v.extend(t.observation_spec.channels.iter().map(|c| format!("{c:?}")));
        v.push(format!("{:?}", t.config));
        v.push(format!("{:?}", t.scene));
        v.sort();
        v
    };
    let (mut g, mut w) = (lines(got), lines(want));
    for x in lines(want) {
        if let Some(i) = g.iter().position(|y| *y == x) {
            g.remove(i);
            w.remove(w.iter().position(|y| *y == x).expect("present"));
        }
    }
    format!(
        "{} vs {} nodes, {} vs {} edges\nonly compiled:\n  {}\nonly committed:\n  {}",
        got.graph.nodes.len(),
        want.graph.nodes.len(),
        got.graph.edges.len(),
        want.graph.edges.len(),
        g.join("\n  "),
        w.join("\n  ")
    )
}

fn same_task(spec_rel: &str, committed_rel: &str) {
    let got = compile_task(&spec(spec_rel), &repo()).expect("the spec compiles");
    let want = es_ir::serial::task_from_toml(&read(committed_rel)).expect("committed");
    let (g, w) = (got.task_hash().unwrap(), want.task_hash().unwrap());
    assert!(
        g == w,
        "{spec_rel}: task_hash {} != {}\n{}",
        hex(&g),
        hex(&w),
        diff(&got, &want)
    );
    let graph = got.task_graph_hash().unwrap() == want.task_graph_hash().unwrap();
    println!(
        "{spec_rel}: task_hash {} equal; task_graph_hash equal: {graph}",
        hex(&g)
    );
}

#[test]
fn the_shadow_hand_spec_compiles_to_the_committed_task_hash() {
    same_task(
        "tests/fixtures/estask/shadow_hand_repose.estask",
        "tests/fixtures/shadow-hand/task-repose.toml",
    );
}

#[test]
fn the_so101_views_spec_compiles_to_the_committed_task_hash() {
    same_task(
        "tests/fixtures/estask/so101_views.estask",
        "tests/fixtures/visible-learning/task-views.toml",
    );
}

const FIXTURES: [&str; 2] = [
    "tests/fixtures/estask/shadow_hand_repose.estask",
    "tests/fixtures/estask/so101_views.estask",
];

#[test]
fn compiling_is_a_function_of_the_document_and_the_scene() {
    for rel in FIXTURES {
        let once = |_| es_ir::serial::task_to_toml(&compile_task(&spec(rel), &repo()).unwrap());
        assert_eq!(once(0).unwrap(), once(1).unwrap(), "{rel}");
    }
}

#[test]
fn read_then_write_is_the_identity_on_the_fixtures() {
    for rel in FIXTURES {
        let doc = spec(rel);
        let again = TaskSpec::from_toml(&doc.to_toml().unwrap()).unwrap();
        assert_eq!(again, doc, "{rel}");
    }
}

/// `doc` with `from` replaced by `to` (exactly once) must be refused naming every `needle`.
fn refused(doc: &str, from: &str, to: &str, needles: &[&str]) {
    assert_eq!(doc.matches(from).count(), 1, "`{from}` is not unique");
    let text = doc.replace(from, to);
    let err = TaskSpec::from_toml(&text)
        .and_then(|s| compile_task(&s, &repo()))
        .expect_err(to)
        .to_string();
    for n in needles {
        assert!(err.contains(n), "`{to}`: `{err}` does not name `{n}`");
    }
}

#[test]
fn refusals_name_the_clause_and_the_field() {
    let so = read(FIXTURES[1]);
    let inside = "relation = \"inside\", range = [0.09, 0.19]";
    let cases: [(&str, &str, &[&str]); 14] = [
        (
            "speed = 0.05",
            "speed = 0.05, colour = \"red\"",
            &["colour"],
        ),
        (
            "robot = \"base\"",
            "robot = \"base\"\nrobots = 2",
            &["robots"],
        ),
        ("robot = \"base\"", "robot = \"bse\"", &["robot", "bse"]),
        (
            "relation = \"still\"",
            "relation = \"touches\"",
            &["success[1]", "relation", "GetContact"],
        ),
        (
            "subject = \"gripper\"",
            "subject = \"griper\"",
            &["success[2] (griper above)", "subject", "griper"],
        ),
        (
            "value = 0.55 }",
            "value = 0.55, m = 0.1 }",
            &["failure[0]", "`m`", "above"],
        ),
        (
            inside,
            "relation = \"inside\"",
            &["success[0]", "range", "required"],
        ),
        (
            "speed = 0.05",
            "speed = 0.05, weight = 1.0",
            &["success[1]", "weight", "shaping"],
        ),
        (
            "shaping = \"ramp\"",
            "shaping = \"distance\"",
            &["success[0]", "shaping", "ramp"],
        ),
        (
            "subject = \"cube.x\", relation = \"still\"",
            "subject = \"cube.w\", relation = \"still\"",
            &["success[1]", "subject", "x, y or z"],
        ),
        (
            "what = \"cube.z\"",
            "what = \"cube.w\"",
            &["start[6]", "what"],
        ),
        (
            "{ what = \"gripper\", range = [0.0, 0.0]",
            "{ what = \"gripper\", range = [0.0, 0.0], tilt = 1.0",
            &["start[5]", "tilt"],
        ),
        (
            "\"overhead\", \"wrist\"",
            "\"overhead\", \"wristcam\"",
            &["observe", "cameras", "wristcam"],
        ),
        (
            "\"cube.qpos\"",
            "\"cube.qpose\"",
            &["observe.privileged.sim_cube_pose", "cube.qpose"],
        ),
    ];
    for (from, to, needles) in cases {
        refused(&so, from, to, needles);
    }
    let hand = read(FIXTURES[0]);
    refused(
        &hand,
        "object = \"target\"",
        "object = \"targt\"",
        &["success[0] (object orientation_matches)", "object", "targt"],
    );
    refused(&hand, "m = 0.24", "m = 1.5", &["failure[0]", "`m`", "1 m"]);
}

/// GV's finding (design note section 4.7): a coordinate of a free body no start item sets starts
/// where the scene puts the body, not at zero. The scene is `empty.esscene` with the SO-101 arm
/// included, as an authored project's, and the bodies written here.
mod scene_pose {
    // The poses are the scene's bits, compared as such.
    #![allow(clippy::float_cmp)]

    use std::path::{Path, PathBuf};

    use es_assets::scene::{JointKind, SceneDesc};
    use es_env::{BatchDomains, Env};
    use es_ir::task::{Distribution, TaskIr, TaskNode};
    use es_physics_backend::MuJoCoCpuBackend;
    use es_physics_core::backend::PhysicsBackend;
    use es_script::spec::{compile_task, TaskSpec};

    use super::{read, repo};

    /// GV's box: 5 cm, on the floor in front of the arm.
    const BOX: &str = r#"
[[body]]
name = "box"
pos = [0.22, 0.0, 0.025]
joint = { kind = "free" }

[[body.geom]]
name = "box_geom"
shape = { box = [0.025, 0.025, 0.025] }
"#;

    /// A box turned 45° about Z, `[x, y, z, w]` (canonical, so `expand` keeps the bits).
    const TURNED: &str = r#"
[[body]]
name = "turned"
pos = [0.3, 0.1, 0.025]
quat = [0.0, 0.0, 0.3826834323650898, 0.9238795325112867]
joint = { kind = "free" }

[[body.geom]]
name = "turned_geom"
shape = { box = [0.025, 0.025, 0.025] }
"#;

    /// GV's start: the box's x and y drawn, nothing else.
    const XY: &str = r#"
[start]
items = [
  { what = "box.x", value = 0.22, noise = 0.02, dice = true },
  { what = "box.y", value = 0.0, noise = 0.02, dice = true },
]
"#;

    type Reset = (bool, String, Distribution, String);

    /// A scratch project: `scene.esscene` (the empty scene, the arm, `bodies`) and the arm's file.
    fn project(tag: &str, bodies: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("estask-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        let arm = "tests/fixtures/esscene/so101.xml";
        std::fs::copy(repo().join(arm), dir.join("so101.xml")).unwrap();
        let empty = read("tests/fixtures/esscene/empty.esscene");
        let text =
            format!("{empty}\n[[include]]\nname = \"arm\"\nsource = \"so101.xml\"\n{bodies}");
        std::fs::write(dir.join("scene.esscene"), text).unwrap();
        dir
    }

    fn scene(dir: &Path) -> SceneDesc {
        let text = std::fs::read_to_string(dir.join("scene.esscene")).unwrap();
        let doc = es_assets::esscene::EsScene::from_toml(&text).unwrap();
        es_assets::esscene::expand(&doc, dir).unwrap()
    }

    fn spec(start: &str) -> TaskSpec {
        let text = format!(
            "kind = \"task-spec\"\nschema = 1\nscene = \"scene.esscene\"\nrobot = \"arm\"\n\
             control_hz = 50\ntimeout_s = 8.0\n\n[success]\n\
             clauses = [{{ subject = \"box\", relation = \"still\", speed = 0.05 }}]\n{start}"
        );
        TaskSpec::from_toml(&text).unwrap()
    }

    /// Every reset node in node order: `(dice, target, dist, stream)`.
    fn resets(task: &TaskIr) -> Vec<Reset> {
        let row = |dice, t: &String, d: &Distribution, s: &String| {
            Some((dice, t.clone(), d.clone(), s.clone()))
        };
        (task.graph.nodes.values())
            .filter_map(|n| match n {
                TaskNode::ResetState {
                    target,
                    dist,
                    stream,
                } => row(false, target, dist, stream),
                TaskNode::Randomization {
                    target,
                    dist,
                    stream,
                } => row(true, target, dist, stream),
                _ => None,
            })
            .collect()
    }

    fn node(dice: bool, lane: u32, dist: Distribution, stream: &str) -> Reset {
        (dice, format!("qpos[{lane}]"), dist, stream.to_owned())
    }

    /// Oracles 2 and 4: the box placed by x and y gets constant `ResetState`s for its z and its
    /// quaternion, the scene's (`qpos[6..13]`: the arm's six hinges come first); none for x or
    /// y beyond the items' own. `zero_unset = true` gives the items' nodes alone.
    #[test]
    fn unset_coordinates_start_at_the_scenes_pose() {
        let dir = project("compile", BOX);
        let task = compile_task(&spec(XY), &dir).unwrap();
        let (x, y) = (
            Distribution::Uniform {
                lo: 0.22 - 0.02,
                hi: 0.22 + 0.02,
            },
            Distribution::Uniform {
                lo: -0.02,
                hi: 0.02,
            },
        );
        let items = vec![node(true, 6, x, "box.x"), node(true, 7, y, "box.y")];
        let c = Distribution::Constant;
        let mut want = items.clone();
        want.extend([
            node(false, 8, c(0.025), "scene.box.pos.2"),
            node(false, 9, c(1.0), "scene.box.quat.0"),
            node(false, 10, c(0.0), "scene.box.quat.1"),
            node(false, 11, c(0.0), "scene.box.quat.2"),
            node(false, 12, c(0.0), "scene.box.quat.3"),
        ]);
        assert_eq!(resets(&task), want);
        for s in ["scene.box.pos.2", "scene.box.quat.0", "scene.box.quat.3"] {
            assert!(task.config.rng_streams.contains(s), "{s}");
        }

        let with = |v: &str| spec(&XY.replace("[start]", &format!("[start]\nzero_unset = {v}")));
        let old = compile_task(&with("true"), &dir).unwrap();
        assert_eq!(resets(&old), items);
        // `false` is the rule, as absent is.
        let rule = compile_task(&with("false"), &dir).unwrap();
        assert_eq!(rule.task_hash().unwrap(), task.task_hash().unwrap());
    }

    /// The free joint's seven `qpos` lanes of body `name` at tick 0.
    fn lanes(env: &Env<MuJoCoCpuBackend>, scene: &SceneDesc, name: &str) -> Vec<f64> {
        let body = scene.bodies.iter().find(|b| b.name == name).unwrap();
        let joint = (scene.joints.iter())
            .find(|j| j.body == body.id && j.kind == JointKind::Free)
            .unwrap();
        let at = &env.model().qpos[&joint.id];
        let state = env.backend().state();
        state.qpos_of(0)[at.start as usize..(at.start + at.len) as usize].to_vec()
    }

    /// Oracle 3, on `mujoco-cpu`: the box placed by x and y starts at `(x, y, 0.025)` with the
    /// identity; the turned box, which no item names, keeps its place and its turn; with no
    /// start items at all, both start at the scene's pose.
    #[test]
    fn an_env_starts_free_bodies_at_the_scenes_pose() {
        if let Err(why) = MuJoCoCpuBackend::is_available() {
            eprintln!("SKIP an_env_starts_free_bodies_at_the_scenes_pose: {why}");
            return;
        }
        let dir = project("env", &format!("{BOX}{TURNED}"));
        let scene = scene(&dir);
        let turned = [
            0.3,
            0.1,
            0.025,
            0.923_879_532_511_286_7,
            0.0,
            0.0,
            0.382_683_432_365_089_8,
        ];
        for start in [XY, ""] {
            let task = compile_task(&spec(start), &dir).unwrap();
            let domains = BatchDomains::single_env();
            let env = Env::new(&task, &scene, MuJoCoCpuBackend::new(), &domains, 7).unwrap();
            let b = lanes(&env, &scene, "box");
            if start.is_empty() {
                assert_eq!(b, [0.22, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]);
            } else {
                let drawn = (0.2..=0.24).contains(&b[0]) && (-0.02..=0.02).contains(&b[1]);
                assert!(drawn, "{b:?}");
                assert_eq!(b[2..], [0.025, 1.0, 0.0, 0.0, 0.0], "{b:?}");
            }
            assert_eq!(lanes(&env, &scene, "turned"), turned, "{start}");
        }
    }
}

mod generated {
    use es_script::spec::{
        Clause, Clauses, Draw, Observe, Relation, RenderDoc, RenderPath, RewardDoc, Shaping, Start,
        StartItem, TaskSpec,
    };
    use proptest::option::of;
    use proptest::prelude::*;

    fn name() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_.:]{0,8}"
    }

    fn num() -> impl Strategy<Value = f64> {
        -1.0e3..1.0e3
    }

    fn clause() -> impl Strategy<Value = Clause> {
        let relation = prop_oneof![
            Just(Relation::Inside),
            Just(Relation::Above),
            Just(Relation::Near),
            Just(Relation::FartherThan),
            Just(Relation::Still),
            Just(Relation::OrientationMatches),
        ];
        let shaping = prop_oneof![
            Just(Shaping::Distance),
            Just(Shaping::Ramp),
            Just(Shaping::InverseAngle)
        ];
        (
            (name(), relation, of(name()), of([num(), num(), num()])),
            (
                of([num(), num()]),
                of(num()),
                of(num()),
                of(num()),
                of(num()),
                of(num()),
            ),
            (of(shaping), of(num()), of([num(), num()]), of(name())),
        )
            .prop_map(|(a, b, c)| Clause {
                subject: a.0,
                relation: a.1,
                object: a.2,
                point: a.3,
                range: b.0,
                value: b.1,
                m: b.2,
                speed: b.3,
                angular: b.5,
                within_deg: b.4,
                shaping: c.0,
                weight: c.1,
                ramp: c.2,
                term: c.3,
            })
    }

    fn item() -> impl Strategy<Value = StartItem> {
        let draw = prop_oneof![Just(Draw::Yaw), Just(Draw::Tilt), Just(Draw::Any)];
        (
            name(),
            (of(num()), of(num()), of([num(), num()]), of(num())),
            (of(draw), of(num())),
            (of(any::<bool>()), of(any::<bool>()), of(name())),
        )
            .prop_map(|(what, a, d, b)| StartItem {
                what,
                value: a.0,
                noise: a.1,
                range: a.2,
                draw: d.0,
                tilt: a.3,
                tilt_max_deg: d.1,
                coupled: b.0,
                dice: b.1,
                stream: b.2,
            })
    }

    fn observe() -> impl Strategy<Value = Observe> {
        let render = (
            of(Just(RenderPath::Pt)),
            of(1u32..64),
            of(0.1f32..10.0),
            of(any::<bool>()),
        )
            .prop_map(|(path, spp, exposure, svgf)| RenderDoc {
                path,
                spp,
                bounces: spp,
                exposure,
                svgf,
                ..RenderDoc::default()
            });
        let channels = || of(proptest::collection::btree_map(name(), name(), 0..4));
        (
            of(proptest::collection::vec(name(), 0..3)),
            of(1u32..512),
            of(render),
            channels(),
            channels(),
        )
            .prop_map(|(cameras, camera_px, render, state, privileged)| Observe {
                cameras,
                camera_px,
                render,
                state,
                privileged,
            })
    }

    fn spec() -> impl Strategy<Value = TaskSpec> {
        let clauses =
            || proptest::collection::vec(clause(), 0..3).prop_map(|clauses| Clauses { clauses });
        let start = (
            of(num()),
            of(any::<bool>()),
            proptest::collection::vec(item(), 0..3),
        )
            .prop_map(|(strength, zero_unset, items)| Start {
                strength,
                zero_unset,
                items,
            });
        let reward =
            (of(num()), of(num()), of(num())).prop_map(|(scale, success, failure)| RewardDoc {
                scale,
                success,
                failure,
            });
        (
            (name(), name(), 1.0f64..500.0, 0.1f64..100.0),
            (
                clauses(),
                of(clauses()),
                of(start),
                of(observe()),
                of(reward),
            ),
        )
            .prop_map(|(a, b)| TaskSpec {
                kind: "task-spec".to_owned(),
                schema: 1,
                scene: a.0,
                robot: a.1,
                control_hz: a.2,
                timeout_s: a.3,
                success: b.0,
                failure: b.1,
                start: b.2,
                observe: b.3,
                reward: b.4,
                // G3b's sections: their round trip is `estask_generate.rs`'s, on the fixtures.
                teacher: None,
                student: None,
                deploy: None,
                evaluate: None,
                cycle: None,
            })
    }

    proptest! {
        #[test]
        fn read_then_write_is_the_identity_on_generated_documents(doc in spec()) {
            let text = doc.to_toml().unwrap();
            prop_assert_eq!(TaskSpec::from_toml(&text).unwrap(), doc, "{}", text);
        }
    }
}
