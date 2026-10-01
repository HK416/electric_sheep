//! Plan G, packet G1: the scene document `*.esscene`.
//!
//! (1) Mirror: a document that includes the robot and writes the rest of a committed MJCF scene
//! natively expands to the `SceneDesc` that MJCF file reads to — the same values in the same
//! order, the same `scene_hash`, the same asset content hashes. The one list compared as a set
//! is `assets`: the Shadow Hand file interleaves the cube's texture and materials with the hand's,
//! and an include's assets come before the document's; nothing reads that order (`scene_hash`
//! sorts it, the loaders and the renderer look assets up by id).
//! (2) Round trip: `from_toml(to_toml(doc)) == doc`, `to_toml` stable, a property test.
//! (3) Refusals name the field. (4) Placement, prefix and `[include.set]`.

// Bits are the point: every float here is compared exactly.
#![allow(clippy::float_cmp)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use es_assets::esscene::{expand, EsScene, EsSceneError};
use es_assets::SceneDesc;
use proptest::prelude::*;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// What `es_tools::load_scene` does with an MJCF file.
fn load_mjcf(path: &Path) -> SceneDesc {
    let mut s = es_assets::parse_mjcf(&std::fs::read_to_string(path).unwrap())
        .unwrap()
        .scene;
    es_assets::mesh::load(&mut s, path.parent().unwrap()).unwrap();
    s
}

fn read_doc(path: &Path) -> EsScene {
    EsScene::from_toml(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn expand_file(path: &Path) -> SceneDesc {
    expand(&read_doc(path), path.parent().unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

/// Every value and every order, `assets` as a set (module docs).
fn canon(s: &SceneDesc) -> String {
    let mut s = s.clone();
    s.assets.sort_by_key(|a| a.id);
    format!("{s:?}")
}

fn assert_mirror(mjcf: &Path, doc: &Path) -> SceneDesc {
    let (want, got) = (load_mjcf(mjcf), expand_file(doc));
    assert_eq!(hex(&got.scene_hash()), hex(&want.scene_hash()));
    let hashes = |s: &SceneDesc| {
        let mut v: Vec<_> = s.assets.iter().map(|a| (a.id, a.hash)).collect();
        v.sort();
        v
    };
    assert_eq!(hashes(&got), hashes(&want));
    // Line by line, so a difference is readable.
    let (g, w) = (canon(&got), canon(&want));
    for (a, b) in g.split(", ").zip(w.split(", ")) {
        assert_eq!(a, b);
    }
    assert_eq!(g, w);
    got
}

#[test]
fn so101_document_reads_to_the_mjcf_scene() {
    let got = assert_mirror(
        &fixtures().join("mjcf/so101_pick_place.xml"),
        &fixtures().join("esscene/so101_pick_place.esscene"),
    );
    // The digest every committed SO-101 document carries (scene_hash_pins.rs).
    assert_eq!(
        hex(&got.scene_hash()),
        "882e7d0b67d32fe8df9260d753d5e985a2bd641de557882bc4e654b5a03b5ba4"
    );
}

#[test]
fn shadow_hand_document_reads_to_the_mjcf_scene() {
    let got = assert_mirror(
        &fixtures().join("mjcf/shadow_hand/shadow_hand_repose.xml"),
        &fixtures().join("mjcf/shadow_hand/shadow_hand_repose.esscene"),
    );
    assert!(!got.meshes.is_empty() && got.textures.values().all(|t| t.data.is_some()));
    // The digest tests/fixtures/shadow-hand/task-repose.toml carries.
    assert_eq!(
        hex(&got.scene_hash()),
        "a900841d2a2bfb8f7faed123bd4f8f4bc7bce9a61d82f06f4e4a54126004cc87"
    );
}

#[test]
fn the_committed_documents_round_trip() {
    for path in [
        "esscene/so101_pick_place.esscene",
        "mjcf/shadow_hand/shadow_hand_repose.esscene",
    ] {
        let doc = read_doc(&fixtures().join(path));
        let text = doc.to_toml().unwrap();
        let again = EsScene::from_toml(&text).unwrap();
        assert_eq!(again, doc, "{path}");
        assert_eq!(again.to_toml().unwrap(), text, "{path}");
    }
}

// ---- refusals name the field -------------------------------------------------------------

fn refusal(text: &str) -> String {
    let doc = match EsScene::from_toml(text) {
        Ok(doc) => doc,
        Err(e) => return e.to_string(),
    };
    match expand(&doc, &fixtures().join("esscene")) {
        Ok(_) => panic!("expanded:\n{text}"),
        Err(e) => e.to_string(),
    }
}

const HEAD: &str = "kind = \"scene\"\nschema = 1\n";

#[test]
fn refusals_name_the_field() {
    let cases = [
        ("[[body]]\nname = \"a\"\ncolour = 1\n", "colour"),
        ("[physics]\nsubsteps = 2\n", "substeps"),
        ("[[body]]\nname = \"a\"\nparent = \"ghost\"\n", "`body[a].parent`"),
        (
            "[[body]]\nname = \"a\"\n[[body.geom]]\nname = \"g\"\nshape = { sphere = 0.1 }\nmaterial = \"ghost\"\n",
            "`body[a].geom[g].material`",
        ),
        ("[[material]]\nname = \"m\"\ntexture = \"ghost\"\n", "`material[m].texture`"),
        ("[[include]]\nname = \"r\"\nsource = \"ghost.xml\"\n", "`include[r].source`"),
        ("[[body]]\nname = \"a\"\n[[body]]\nname = \"a\"\n", "`body[a].name`"),
        ("[[camera]]\nname = \"c\"\nparent = \"ghost\"\n", "`camera[c].parent`"),
        (
            "[[include]]\nname = \"r\"\nsource = \"so101.xml\"\n[include.set.joint.ghost]\ndamping = 1.0\n",
            "`include[r].set.joint.ghost`",
        ),
        (
            "[[include]]\nname = \"r\"\nsource = \"so101.xml\"\n[include.set.actuator.gripper]\nstiffness = 1.0\n",
            "stiffness",
        ),
    ];
    for (body, field) in cases {
        let msg = refusal(&format!("{HEAD}{body}"));
        assert!(msg.contains(field), "`{field}` not in: {msg}");
    }
    let msg = refusal("kind = \"task\"\nschema = 1\n");
    assert!(msg.contains("`kind`"), "{msg}");
    // A native body named like an included one is two elements on one name path.
    let msg = refusal(&format!(
        "{HEAD}[[include]]\nname = \"r\"\nsource = \"so101.xml\"\n[[body]]\nname = \"base\"\n"
    ));
    assert!(msg.contains("`base`"), "{msg}");
}

// ---- include placement, prefix, overrides ------------------------------------------------

#[test]
fn a_prefixed_placed_include_keeps_its_wiring() {
    let text = format!(
        "{HEAD}[[include]]\nname = \"left\"\nsource = \"so101.xml\"\nprefix = \"left/\"\npos = [0.0, 0.5, 0.0]\n\
         [include.set.joint.gripper]\nrange = [-0.1, 1.0]\n\
         [include.set.actuator.gripper]\nkp = 50.0\n\
         [include.set.geom.base_shell]\nrgba = [0.0, 1.0, 0.0, 1.0]\n\
         [[include]]\nname = \"right\"\nsource = \"so101.xml\"\n\
         [[body]]\nname = \"tool\"\nparent = \"left/gripper\"\n"
    );
    let s = expand(
        &EsScene::from_toml(&text).unwrap(),
        &fixtures().join("esscene"),
    )
    .unwrap();
    let body = |n: &str| s.bodies.iter().find(|b| b.name == n).unwrap();
    assert_eq!(body("left/base").pose.position.y, 0.5);
    assert_eq!(body("base").pose.position.y, 0.0);
    assert_eq!(
        body("left/base").id,
        es_assets::scene_id("body", "world/left/base")
    );
    assert_eq!(
        body("tool").id,
        es_assets::scene_id(
            "body",
            "world/left/base/left/shoulder/left/upper_arm/left/lower_arm/left/wrist/left/gripper/tool"
        )
    );
    let joint = s.joints.iter().find(|j| j.name == "left/gripper").unwrap();
    assert_eq!(joint.range, Some((-0.1, 1.0)));
    let act = s
        .actuators
        .iter()
        .find(|a| a.name == "left/gripper")
        .unwrap();
    assert_eq!(
        act.target,
        es_assets::scene::ActuatorTarget::Joint(joint.id)
    );
    assert!(matches!(
        act.kind,
        es_assets::scene::ActuatorKind::Position { kp, .. } if kp == 50.0
    ));
    let shell = body("left/base")
        .geoms
        .iter()
        .find(|g| g.name == "left/base_shell")
        .unwrap();
    assert_eq!(shell.rgba, [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(s.actuators.len(), 12);
}

// ---- round trip over generated documents -------------------------------------------------

fn finite() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(-0.0),
        Just(1.0),
        -1e6..1e6f64,
        prop::num::f64::NORMAL,
        prop::num::f64::SUBNORMAL,
    ]
}

fn name() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_:/ .-]{0,8}"
}

fn arr<const N: usize>() -> impl Strategy<Value = [f64; N]> {
    prop::collection::vec(finite(), N).prop_map(|v| v.try_into().unwrap())
}

fn opt<T: std::fmt::Debug + Clone>(
    s: impl Strategy<Value = T>,
) -> impl Strategy<Value = Option<T>> {
    prop::option::of(s)
}

fn geom_doc() -> impl Strategy<Value = es_assets::esscene::GeomDoc> {
    use es_assets::esscene::{GeomDoc, ShapeDoc};
    let shape = prop_oneof![
        arr::<3>().prop_map(ShapeDoc::Plane),
        finite().prop_map(ShapeDoc::Sphere),
        arr::<2>().prop_map(ShapeDoc::Capsule),
        arr::<3>().prop_map(ShapeDoc::Box),
        name().prop_map(ShapeDoc::Mesh),
    ];
    (
        opt(name()),
        shape,
        opt(arr::<3>()),
        opt(arr::<4>()),
        opt(finite()),
        opt(arr::<3>()),
        opt(0u32..8),
        opt(any::<i32>()),
        opt(arr::<5>()),
        opt(arr::<4>()),
        opt(name()),
    )
        .prop_map(
            |(name, shape, pos, quat, mass, friction, condim, priority, solimp, rgba, material)| {
                GeomDoc {
                    name,
                    shape,
                    pos,
                    quat,
                    mass,
                    density: None,
                    friction,
                    condim,
                    contype: condim,
                    conaffinity: None,
                    priority,
                    margin: mass,
                    gap: None,
                    solref: None,
                    solimp,
                    rgba,
                    material,
                }
            },
        )
}

fn doc() -> impl Strategy<Value = EsScene> {
    use es_assets::esscene::{
        BodyDoc, CameraDoc, Include, InertialDoc, IntegratorDoc, JointDoc, JointKindDoc, LightDoc,
        MaterialDoc, Physics, RegionDoc, TextureDoc,
    };
    let joint = (
        prop_oneof![
            Just(JointKindDoc::Fixed),
            Just(JointKindDoc::Free),
            Just(JointKindDoc::Hinge),
            Just(JointKindDoc::Slide),
            Just(JointKindDoc::Ball),
        ],
        opt(name()),
        opt(arr::<3>()),
        opt(arr::<2>()),
        opt(finite()),
    )
        .prop_map(|(kind, name, axis, range, damping)| JointDoc {
            kind,
            name,
            axis,
            pos: axis,
            range,
            damping,
            armature: damping,
            stiffness: None,
            frictionloss: None,
            springref: damping,
        });
    let body = (
        name(),
        opt(name()),
        opt(arr::<3>()),
        opt(arr::<4>()),
        opt(finite()),
        opt(joint),
        opt((finite(), arr::<3>())),
        prop::collection::vec(geom_doc(), 0..3),
    )
        .prop_map(
            |(name, parent, pos, quat, gravcomp, joint, inertial, geoms)| BodyDoc {
                name,
                parent,
                pos,
                quat,
                gravcomp,
                joint,
                inertial: inertial.map(|(mass, d)| InertialDoc {
                    mass,
                    pos: None,
                    quat: None,
                    diaginertia: Some(d),
                    fullinertia: None,
                }),
                geoms,
            },
        );
    let material = (
        name(),
        opt(arr::<4>()),
        opt(finite()),
        opt(name()),
        opt(any::<bool>()),
    )
        .prop_map(|(name, rgba, specular, texture, texuniform)| MaterialDoc {
            name,
            rgba,
            emission: None,
            specular,
            shininess: specular,
            metallic: None,
            roughness: None,
            texrepeat: None,
            texuniform,
            texture: texture.clone(),
            orm: None,
            metallic_map: None,
            roughness_map: None,
            normal_map: texture,
            normal_scale: specular,
            emissive_map: None,
        });
    let texture = (name(), opt(name()), opt([1u32..5, 1u32..5]), opt(name())).prop_map(
        |(name, file, gridsize, gridlayout)| TextureDoc {
            name,
            kind: None,
            colorspace: None,
            file,
            gridsize,
            gridlayout,
            cubefiles: None,
            builtin: None,
            rgb1: None,
            rgb2: None,
            mark: None,
            markrgb: None,
            random: None,
            width: None,
            height: None,
        },
    );
    let camera = (
        name(),
        opt(name()),
        opt(arr::<3>()),
        opt(arr::<4>()),
        opt(finite()),
    )
        .prop_map(|(name, parent, pos, quat, fovy)| CameraDoc {
            name,
            parent,
            pos,
            quat,
            fovy,
        });
    let light = (name(), opt(arr::<3>()), arr::<2>(), opt(finite())).prop_map(
        |(name, pos, size, intensity)| LightDoc {
            name,
            kind: None,
            pos,
            quat: None,
            size,
            rgb: None,
            intensity,
        },
    );
    let region =
        (name(), opt(name()), opt(arr::<3>())).prop_map(|(name, parent, size)| RegionDoc {
            name,
            parent,
            pos: size,
            quat: None,
            size,
        });
    let include =
        (name(), name(), opt(name()), opt(arr::<3>())).prop_map(|(name, source, prefix, pos)| {
            Include {
                name,
                source,
                prefix,
                pos,
                quat: None,
                set: None,
            }
        });
    let physics = opt(
        (opt(finite()), opt(0u32..100)).prop_map(|(timestep, iterations)| Physics {
            timestep,
            iterations,
            integrator: Some(IntegratorDoc::ImplicitFast),
            ..Physics::default()
        }),
    );
    (
        opt(name()),
        physics,
        prop::collection::vec(include, 0..2),
        prop::collection::vec(geom_doc(), 0..3),
        prop::collection::vec(body, 0..4),
        prop::collection::vec(texture, 0..2),
        prop::collection::vec(material, 0..2),
        prop::collection::vec(camera, 0..2),
        prop::collection::vec(light, 0..2),
        prop::collection::vec(region, 0..2),
    )
        .prop_map(
            |(
                name,
                physics,
                includes,
                geoms,
                bodies,
                textures,
                materials,
                cameras,
                lights,
                regions,
            )| {
                EsScene {
                    kind: "scene".to_owned(),
                    schema: 1,
                    name,
                    physics,
                    includes,
                    geoms,
                    bodies,
                    textures,
                    materials,
                    cameras,
                    lights,
                    regions,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn read_after_write_is_the_identity(d in doc()) {
        let text = d.to_toml().map_err(|e| TestCaseError::fail(e.to_string()))?;
        let again = EsScene::from_toml(&text).map_err(|e| TestCaseError::fail(format!("{e}\n{text}")))?;
        prop_assert_eq!(&again, &d, "{}", text);
        prop_assert_eq!(again.to_toml().unwrap(), text);
    }
}

#[test]
fn an_unreadable_document_is_a_toml_error() {
    assert!(matches!(
        EsScene::from_toml("kind = "),
        Err(EsSceneError::Toml(_))
    ));
}
