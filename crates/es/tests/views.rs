//! Packet M15/N1 (`docs/packets/M15/plan-n.md`, `docs/design/multi-camera.md`): the three-view
//! scene `tests/fixtures/mjcf/so101_pick_place_views.xml` and the documents beside it; packet
//! M15/N8b: experiment 2's MAD documents (`*-mad.toml`).
//!
//! The generated documents come from the committed ones by the `derive_*` functions below, and
//! `views_documents_are_what_the_generator_derives` holds the committed files to them, so no
//! hash in them is typed in and none can go stale. The recipes and cycles are hand-written and
//! held to the hint card's by `views_recipes_and_cycles_mirror_the_hint_card`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_ir::evaluation::EvaluationIr;
use es_ir::graph::{NodeId, Port, PortRef};
use es_ir::learning::{LearningGraph, LearningNode};
use es_ir::observation::{ObservationIr, ObservationOutput};
use es_ir::task::{TaskIr, TaskNode};

const VIEWS_SCENE: &str = "tests/fixtures/mjcf/so101_pick_place_views.xml";
const DEMO_SCENE: &str = "tests/fixtures/mjcf/so101_pick_place.xml";

/// The two cameras this packet adds, with the Task IR channel and the Learning IR fusion port
/// each one feeds.
const NEW_VIEWS: [(&str, &str, &str); 2] = [
    ("wrist", "rgb_wrist", "wrist"),
    ("side", "rgb_side", "side"),
];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn vl(name: &str) -> PathBuf {
    repo().join("tests/fixtures/visible-learning").join(name)
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hex(d: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*d).to_hex().to_string()
}

fn parse_scene(rel: &str) -> SceneDesc {
    let parsed = es_assets::parse_mjcf(&read(&repo().join(rel))).expect("the scene parses");
    assert!(parsed.warnings.is_empty(), "{rel}: {:?}", parsed.warnings);
    parsed.scene
}

fn camera(scene: &SceneDesc, name: &str) -> es_assets::scene::Camera {
    scene
        .cameras
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("the scene has a `{name}` camera"))
        .clone()
}

/// `fx = fy` of a pinhole camera `width` pixels wide with vertical field of view `fovy`: the
/// square image's focal length, which is how the committed `rgb_overhead` channel was declared.
fn focal(width: u32, fovy: f64) -> f64 {
    f64::from(width) / 2.0 / (fovy / 2.0).tan()
}

/// `value` with every occurrence of camera `from` replaced by `to` (a sensor, a `Camera` frame,
/// a `Sensor` time reference) and every `fx`/`fy` equal to `fx_from` replaced by `fx_to`.
///
/// Done through the serialized form, so every place a camera appears in a node, a port type or
/// a channel is caught without naming each one -- a copy of the overhead chain that kept one
/// overhead id would be a document that reads the wrong camera somewhere.
fn retarget<T: serde::Serialize + serde::de::DeserializeOwned>(
    value: &T,
    from: StableId,
    to: StableId,
    fx_from: f64,
    fx_to: f64,
) -> T {
    fn walk(
        v: &mut serde_json::Value,
        from: &serde_json::Value,
        to: &serde_json::Value,
        fx: (f64, f64),
    ) {
        match v {
            serde_json::Value::String(_) if v == from => *v = to.clone(),
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, from, to, fx);
                }
            }
            serde_json::Value::Object(map) => {
                for (key, item) in map.iter_mut() {
                    if (key == "fx" || key == "fy") && item.as_f64() == Some(fx.0) {
                        *item = serde_json::json!(fx.1);
                    } else {
                        walk(item, from, to, fx);
                    }
                }
            }
            _ => {}
        }
    }
    let mut v = serde_json::to_value(value).expect("serializes");
    let (from, to) = (
        serde_json::to_value(from).expect("id"),
        serde_json::to_value(to).expect("id"),
    );
    walk(&mut v, &from, &to, (fx_from, fx_to));
    serde_json::from_value(v).expect("deserializes")
}

// --- the derivations ---------------------------------------------------------------------------

/// task.toml on the views scene, plus `rgb_wrist` and `rgb_side`: each a `GetSensor ->
/// ObservationSpec` pair and a channel, copied from `rgb_overhead`'s with the camera retargeted.
fn derive_task() -> TaskIr {
    let xml = std::fs::read(repo().join(VIEWS_SCENE)).expect("the views scene");
    let scene = parse_scene(VIEWS_SCENE);
    let mut task = es_ir::serial::task_from_toml(&read(&vl("task.toml"))).expect("task.toml");
    VIEWS_SCENE.clone_into(&mut task.scene.path);
    task.scene.scene_hash = scene.scene_hash();
    task.scene.asset_hash = *blake3::hash(&xml).as_bytes();

    let overhead = camera(&scene, "overhead");
    let channel = task.observation_spec.channels["rgb_overhead"].clone();
    let image = channel.ty.image.expect("rgb_overhead is an image");
    let fx = image.intrinsics.fx;
    // The committed channel's focal length is this formula on the overhead camera's own fovy,
    // so the new channels are declared the way it was.
    assert!(
        (focal(image.width, overhead.fovy) - fx).abs() < 1e-9,
        "{fx}"
    );
    let (get, decl) = {
        let find = |want: fn(&TaskNode) -> bool| {
            *task
                .graph
                .nodes
                .iter()
                .find(|(_, n)| want(n))
                .expect("the overhead's node")
                .0
        };
        (
            find(|n| matches!(n, TaskNode::GetSensor { .. })),
            find(
                |n| matches!(n, TaskNode::ObservationSpec { channel, .. } if channel == "rgb_overhead"),
            ),
        )
    };
    for (name, port, _) in NEW_VIEWS {
        let cam = camera(&scene, name);
        let fx_new = focal(image.width, cam.fovy);
        let next = task.graph.nodes.keys().map(|n| n.0).max().expect("nodes") + 1;
        let (g, d) = (NodeId(next), NodeId(next + 1));
        task.graph.insert(
            g,
            retarget(&task.graph.nodes[&get], overhead.id, cam.id, fx, fx_new),
        );
        let mut spec = retarget(&task.graph.nodes[&decl], overhead.id, cam.id, fx, fx_new);
        match &mut spec {
            TaskNode::ObservationSpec { channel, .. } => port.clone_into(channel),
            other => panic!("{other:?}"),
        }
        task.graph.insert(d, spec);
        task.graph.connect(g, "value", d, "value");
        task.observation_spec.channels.insert(
            port.to_owned(),
            retarget(&channel, overhead.id, cam.id, fx, fx_new),
        );
    }
    let diags = task.validate();
    assert!(diags.is_empty(), "{diags:?}");
    task
}

/// `ir` without output `name` and the nodes that exist only to produce it.
fn drop_output(ir: &mut ObservationIr, name: &str) {
    let out = ir.outputs.remove(name).expect("the output");
    let mut gone = vec![out.port.node];
    while let Some(e) = ir
        .graph
        .edges
        .iter()
        .find(|e| gone.contains(&e.to.node) && !gone.contains(&e.from.node))
    {
        gone.push(e.from.node);
    }
    ir.graph
        .edges
        .retain(|e| !gone.contains(&e.from.node) && !gone.contains(&e.to.node));
    for id in &gone {
        ir.graph.nodes.remove(id);
    }
}

/// observation-augmented.toml (U3's) on task-views.toml, without `sim_cube_pose`.
fn derive_observation_cam(task: &TaskIr) -> ObservationIr {
    let mut obs = es_ir::serial::observation_from_toml(&read(&vl("observation-augmented.toml")))
        .expect("observation-augmented.toml");
    obs.task_ref = task.task_hash().expect("task hash");
    drop_output(&mut obs, "sim_cube_pose");
    let diags = obs.validate();
    assert!(diags.is_empty(), "{diags:?}");
    obs
}

/// observation-cam.toml plus the overhead chain once per new camera, retargeted.
fn derive_observation_views(cam: &ObservationIr) -> ObservationIr {
    let scene = parse_scene(VIEWS_SCENE);
    let overhead = camera(&scene, "overhead");
    let mut obs = cam.clone();
    let out = obs.outputs["rgb_overhead"].clone();
    let fx = out.ty.image.as_ref().expect("an image port").intrinsics.fx;
    // The chain: every node upstream of the output, in the order the walk meets them.
    let mut chain = vec![out.port.node];
    while let Some(e) = obs
        .graph
        .edges
        .iter()
        .find(|e| chain.contains(&e.to.node) && !chain.contains(&e.from.node))
    {
        chain.push(e.from.node);
    }
    let edges: Vec<_> = obs
        .graph
        .edges
        .iter()
        .filter(|e| chain.contains(&e.to.node))
        .cloned()
        .collect();
    for (name, port, _) in NEW_VIEWS {
        let cam = camera(&scene, name);
        let fx_new = focal(out.ty.image.as_ref().expect("image").width, cam.fovy);
        let base = obs.graph.nodes.keys().map(|n| n.0).max().expect("nodes") + 1;
        // Old ids in ascending order map onto `base..`, so the copy keeps the chain's shape.
        let mut sorted = chain.clone();
        sorted.sort();
        let map: BTreeMap<NodeId, NodeId> = sorted
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, NodeId(base + i as u32)))
            .collect();
        for (old, new) in &map {
            let node = retarget(&obs.graph.nodes[old], overhead.id, cam.id, fx, fx_new);
            obs.graph.insert(*new, node);
        }
        for e in &edges {
            obs.graph
                .connect(map[&e.from.node], &e.from.port, map[&e.to.node], &e.to.port);
        }
        obs.outputs.insert(
            port.to_owned(),
            ObservationOutput {
                port: PortRef::new(map[&out.port.node], &out.port.port),
                ty: retarget(&out.ty, overhead.id, cam.id, fx, fx_new),
            },
        );
    }
    let diags = obs.validate();
    assert!(diags.is_empty(), "{diags:?}");
    obs
}

/// learning-pretrained.toml (U3's) without the `sim_cube_pose` branch.
fn derive_learning_cam() -> LearningGraph {
    const CUBE: &str = "sim_cube_pose";
    let mut g = es_ir::serial::learning_from_toml(&read(&vl("learning-pretrained.toml")))
        .expect("learning-pretrained.toml");
    let node = g
        .nodes
        .inputs
        .iter()
        .find(|p| p.port == CUBE)
        .expect("the cube input")
        .node;
    let fused = g
        .nodes
        .edges
        .iter()
        .find(|e| e.from.node == node)
        .expect("the cube encoder feeds the fusion")
        .to
        .clone();
    g.nodes.nodes.remove(&node);
    g.nodes.edges.retain(|e| e.from.node != node);
    g.nodes.inputs.retain(|p| p.node != node);
    g.inputs.retain(|p| p.name != CUBE);
    g.policy.contract.inputs.remove(CUBE);
    match g.nodes.nodes.get_mut(&fused.node).expect("the fusion") {
        LearningNode::Fusion { inputs, .. } => inputs.retain(|p| p.name != fused.port),
        other => panic!("{other:?}"),
    }
    let diags = g.validate();
    assert!(diags.is_empty(), "{diags:?}");
    g
}

/// learning-cam.toml plus one `VisionEncoder` per new view (node 0's, input renamed), each a new
/// `Fusion{Concat}` input.
fn derive_learning_views(cam: &LearningGraph) -> LearningGraph {
    let mut g = cam.clone();
    let vision = g
        .nodes
        .inputs
        .iter()
        .find(|p| p.port == "rgb_overhead")
        .expect("the overhead input")
        .node;
    let fusion = *g
        .nodes
        .nodes
        .iter()
        .find(|(_, n)| matches!(n, LearningNode::Fusion { .. }))
        .expect("the fusion")
        .0;
    let image = g
        .inputs
        .iter()
        .find(|p| p.name == "rgb_overhead")
        .expect("the image input")
        .clone();
    let feature = match &g.nodes.nodes[&fusion] {
        LearningNode::Fusion { inputs, .. } => inputs[0].ty.clone(),
        other => panic!("{other:?}"),
    };
    for (_, channel, port) in NEW_VIEWS {
        let id = NodeId(g.nodes.nodes.keys().map(|n| n.0).max().expect("nodes") + 1);
        let mut encoder = g.nodes.nodes[&vision].clone();
        match &mut encoder {
            LearningNode::VisionEncoder { inputs, .. } => channel.clone_into(&mut inputs[0].name),
            other => panic!("{other:?}"),
        }
        g.nodes.insert(id, encoder);
        g.nodes.connect(id, "out", fusion, port);
        g.nodes.inputs.push(PortRef::new(id, channel));
        match g.nodes.nodes.get_mut(&fusion).expect("the fusion") {
            LearningNode::Fusion { inputs, .. } => inputs.push(Port::new(port, feature.clone())),
            other => panic!("{other:?}"),
        }
        let input = Port::new(channel, image.ty.clone());
        g.inputs.push(input.clone());
        g.policy.contract.inputs.insert(channel.to_owned(), input);
    }
    let diags = g.validate();
    assert!(diags.is_empty(), "{diags:?}");
    g
}

/// learning-views.toml made MAD's (packet M15/N8b, the shape `crates/es-policy/tests/subset.rs`
/// builds inline): the wrist and side encoders `share` the overhead's, the three features meet
/// in a new `Fusion{Sum}` (ports named by camera), and the sum takes the `Concat`'s `image` port.
fn derive_learning_mad(views: &LearningGraph) -> LearningGraph {
    let mut g = views.clone();
    let encoder = |g: &LearningGraph, channel: &str| {
        g.nodes
            .inputs
            .iter()
            .find(|p| p.port == channel)
            .expect("the view's encoder")
            .node
    };
    let concat = *g
        .nodes
        .nodes
        .iter()
        .find(|(_, n)| matches!(n, LearningNode::Fusion { .. }))
        .expect("the fusion")
        .0;
    let sum = NodeId(g.nodes.nodes.keys().map(|n| n.0).max().expect("nodes") + 1);
    let owner = encoder(&g, "rgb_overhead");
    let feature = match g.nodes.nodes.get_mut(&concat).expect("the fusion") {
        LearningNode::Fusion { inputs, .. } => {
            inputs.retain(|p| p.name == "image" || p.name == "state");
            inputs[0].ty.clone()
        }
        other => panic!("{other:?}"),
    };
    g.nodes
        .edges
        .retain(|e| e.to.node != concat || e.to.port == "state");
    let terms = std::iter::once(("overhead", "rgb_overhead", "image")).chain(NEW_VIEWS);
    let mut ports = Vec::new();
    for (camera, channel, _) in terms {
        let id = encoder(&g, channel);
        g.nodes.connect(id, "out", sum, camera);
        ports.push(Port::new(camera, feature.clone()));
        if id != owner {
            match g.nodes.nodes.get_mut(&id).expect("the encoder") {
                LearningNode::VisionEncoder { share, .. } => *share = Some(owner),
                other => panic!("{other:?}"),
            }
        }
    }
    g.nodes.insert(
        sum,
        LearningNode::Fusion {
            inputs: ports,
            kind: es_ir::learning::FusionKind::Sum,
            out_dim: 512,
            token_count: 0,
        },
    );
    g.nodes.connect(sum, "out", concat, "image");
    let diags = g.validate();
    assert!(diags.is_empty(), "{diags:?}");
    g
}

/// evaluation-augmented.toml naming `task` and `observation`.
fn derive_evaluation(task: &TaskIr, obs: &ObservationIr) -> EvaluationIr {
    let mut ev = es_ir::serial::evaluation_from_toml(&read(&vl("evaluation-augmented.toml")))
        .expect("evaluation-augmented.toml");
    ev.task = hex(&task.task_hash().expect("task hash"));
    ev.observation = hex(&obs.observation_hash().expect("observation hash"));
    let diags = ev.validate();
    assert!(diags.is_empty(), "{diags:?}");
    ev
}

struct Derived {
    task: TaskIr,
    obs_cam: ObservationIr,
    obs_views: ObservationIr,
    learning_cam: LearningGraph,
    learning_views: LearningGraph,
    learning_mad: LearningGraph,
    eval_cam: EvaluationIr,
    eval_views: EvaluationIr,
    eval_mad: EvaluationIr,
}

fn derive() -> Derived {
    let task = derive_task();
    let obs_cam = derive_observation_cam(&task);
    let obs_views = derive_observation_views(&obs_cam);
    let learning_cam = derive_learning_cam();
    let learning_views = derive_learning_views(&learning_cam);
    let learning_mad = derive_learning_mad(&learning_views);
    let eval_cam = derive_evaluation(&task, &obs_cam);
    let eval_views = derive_evaluation(&task, &obs_views);
    // The MAD bundle reads the three-view arm's Task and Observation IR (packet M15/N8b).
    let eval_mad = derive_evaluation(&task, &obs_views);
    Derived {
        task,
        obs_cam,
        obs_views,
        learning_cam,
        learning_views,
        learning_mad,
        eval_cam,
        eval_views,
        eval_mad,
    }
}

// --- the generator -----------------------------------------------------------------------------

const GENERATED: &str = "`ES_GENERATE_GOLDENS=1 cargo test -p es --test views -- --ignored \
                         generate_views_documents`";

const TASK_VIEWS_HEADER: &str = "\
# Task IR (spec 6) for the SO-101 cube-into-bin demo seen by three cameras -- plan N, packet
# M15/N1 (docs/design/multi-camera.md, docs/packets/M15/plan-n.md).
#
# Generated by GENERATED
# from task.toml, which is unchanged. This file is that document with the scene moved to
# tests/fixtures/mjcf/so101_pick_place_views.xml -- `scene_hash` and `asset_hash` derived from
# the file, never typed in -- and two more image channels, each a `GetSensor -> ObservationSpec`
# pair and an `observation_spec` channel copied from `rgb_overhead`'s with the camera replaced:
#
#   rgb_overhead  the committed `overhead` camera, untouched (its id is its path in the MJCF body
#                 tree, the same in both scenes)
#   rgb_wrist     the `wrist` camera in SO-101's `camera_mount`, fovy 70 deg:
#                 fx = fy = 48 / tan(35 deg)
#   rgb_side      the world-fixed `side` camera, fovy 45 deg: the overhead's intrinsics
#
# All three are 96x96 Rgb8 as the renderer delivers it (U8, HWC, sRGB, pinhole, 50 Hz).
#
# Everything else -- success, rewards, termination, reset, randomization, `joint_state` and the
# simulator-privileged `sim_cube_pose` -- is task.toml's node for node; task.toml's header says
# what each is and why. `sim_cube_pose` stays declared so the graph is task.toml's: the
# camera-only Observation IRs beside this file (observation-cam.toml, observation-views.toml)
# do not read it. Spec 7.4 lets several Observation IRs share one Task IR, and XIR-002 checks
# only that what an Observation IR reads is declared -- so one collection under this document
# serves both arms of plan N's experiment 1.
#
# `views_documents_are_what_the_generator_derives` (crates/es/tests/views.rs) holds this file to
# its derivation; `the_views_task_declares_what_the_scene_renders` checks each channel's
# ImageSpec against the renderer's (INV-14).
";

const OBSERVATION_CAM_HEADER: &str = "\
# Observation IR (spec 7), camera only, one view -- plan N, packet M15/N1: experiment 1's
# baseline arm (docs/design/multi-camera.md section 4).
#
# Generated by GENERATED
# from observation-augmented.toml (M7/U row U3's Observation IR, unchanged): that document with
# `task_ref` set to task-views.toml's `task_hash` and the simulator-privileged `sim_cube_pose`
# branch -- its `StateInput`, its `Normalize` and its output -- removed. What is left is U3's
# image chain on `rgb_overhead`, byte for byte,
#
#   ImageInput (U8 HWC) -> Dequantize -> Normalize{0..1} -> Pad{4} -> Crop{Random 96x96}
#                       -> Augment{ColorJitter brightness 0.2 contrast 0.2, training only}
#
# and `joint_state`. observation-augmented.toml's header says what the chain does in training
# and in evaluation (INV-14, INV-15).
#
# It reads one of task-views.toml's three image channels, which spec 7.4 allows (several
# Observation IRs share one Task IR; XIR-002 checks only what is read), so a collection
# rendered under task-views.toml serves this arm and the three-view arm alike.
";

const OBSERVATION_VIEWS_HEADER: &str = "\
# Observation IR (spec 7), camera only, three views -- plan N, packet M15/N1: experiment 1's
# three-view arm (docs/design/multi-camera.md section 4).
#
# Generated by GENERATED
# from observation-cam.toml: that document plus its image chain twice more, once for
# `rgb_wrist` and once for `rgb_side`. Each copy is an independent `ImageInput` chain with its
# own output port -- spec 7.4's multi-view form, not `MultiViewPack` -- and is the overhead
# chain with the camera id (sensor, frame, time reference) and the intrinsics (fx, fy, from the
# camera's fovy) replaced. Nothing else differs, so each view is padded, randomly shifted and
# colour-jittered in training and centre-cropped in evaluation exactly as `rgb_overhead` is.
";

const LEARNING_CAM_HEADER: &str = "\
# Learning IR (spec 8), camera only, one view -- plan N, packet M15/N1: experiment 1's baseline
# arm (docs/design/multi-camera.md section 4).
#
# Generated by GENERATED
# from learning-pretrained.toml (M7/U row U3's graph, unchanged) with the simulator-privileged
# `sim_cube_pose` branch removed: its input, its `StateEncoder`, its `Fusion` input `cube` and
# its contract entry. What is left is U3's graph on the overhead camera and the arm's joints:
#
#   VisionEncoder{ResNet18, ImageNet-pretrained, fine-tuned} -.
#                                                            Fusion{Concat} -> TemporalEncoder
#   StateEncoder{Mlp[256]}                                   -'   -> PolicyHead -> ActionChunker
#
# horizon 16, execute_chunk 10, 5 Hz, TemporalEnsemble and the all-zero placeholder weights are
# learning-pretrained.toml's (see its header). The fusion's projection takes 2 x 512 features
# instead of 3 x 512, so this graph trains from the ImageNet backbone, never from a U3
# checkpoint.
";

const LEARNING_VIEWS_HEADER: &str = "\
# Learning IR (spec 8), camera only, three views -- plan N, packet M15/N1: experiment 1's
# three-view arm (docs/design/multi-camera.md section 4).
#
# Generated by GENERATED
# from learning-cam.toml: that graph plus one `VisionEncoder` per new image port -- node 0's,
# with its input renamed to `rgb_wrist` and `rgb_side` -- each fused as one more
# `Fusion{Concat}` input (`wrist`, `side`). Four 512-wide features meet in the fusion (image,
# state, wrist, side), which projects 2048 -> 512; everything after it is learning-cam.toml's.
#
# Three independent ResNet18s: `train_act.py --init-backbone` loads the ImageNet weights into
# every lowered backbone, so they start equal and train apart. Sharing one encoder across the
# views is plan N's `share` (packet N5), not this document.
";

const EVALUATION_HEADER: &str = "\
# Evaluation IR (spec 10) for plan N's experiment 1, ARM -- packet M15/N1.
#
# Generated by GENERATED;
# evaluation-augmented.toml (the document M7/U row U3 was judged by) with `task` and
# `observation` replaced by task-views.toml's and OBSERVATION's own hashes. Same 16 held-out
# seeds 101-116, same six suites, same perturbation parameters, same metrics, same acceptance
# (`success_rate >= 0.5` on nominal) -- a diff with the two fields put back is empty, and
# `views_documents_are_what_the_generator_derives` checks exactly that.
#
# A separate document because an Evaluation IR names the Task and Observation IR it judges
# (XIR-040, spec 13.3): each arm of the experiment reads its own Observation IR, so each is
# judged under its own document, and a report says so. The conditions a person compares --
# suites, seeds, thresholds -- are identical across the two arms and to U3's.
";

const LEARNING_MAD_HEADER: &str = "\
# Learning IR (spec 8), camera only, three views in MAD's form -- plan N, packet M15/N8b:
# experiment 2 (docs/design/multi-camera.md sections 3.3-3.5 and 4, docs/packets/M15/plan-n.md
# task E2). MAD: Almuzairee, Patil, Bhatt, Christensen, CoRL 2025, arXiv 2505.04619.
#
# Generated by GENERATED
# from learning-views.toml (experiment 1's three-view graph): the same three
# `VisionEncoder{ResNet18, ImageNet-pretrained, fine-tuned, out 512}`, the wrist's and the side's
# with `share = 0` -- one module, the overhead encoder's, applied to each view (LRN-033) -- and
# their features summed by a new `Fusion{Sum}` node 8, one port per camera (LRN-032), instead of
# being three `Concat` inputs. The sum takes the `Concat`'s `image` port:
#
#   rgb_overhead -> VisionEncoder 0 -----------.
#   rgb_wrist    -> VisionEncoder 6, share 0 --- Fusion{Sum} 8 --.
#   rgb_side     -> VisionEncoder 7, share 0 --'                 Fusion{Concat} 2 -> TemporalEncoder
#   joint_state  -> StateEncoder{Mlp[256]} 1 ---------------------'   -> PolicyHead -> ActionChunker
#
# The `Concat` projects 2 x 512 -> 512 (learning-views.toml's 4 x 512); everything after it is
# learning-views.toml's, and so are the placeholder weights. Its Observation IR is
# observation-views.toml, unchanged, so it is judged under evaluation-mad.toml, whose body is
# evaluation-views.toml's.
#
# Why a sum: a view's term can be left out without retraining the fusion -- the other terms keep
# their meaning -- so `[run] single_view` (training-mad.toml) trains each view's term alone beside
# the sum (MAD's disentangle), and `es policy subset --views <names>` derives a bundle that reads
# fewer cameras with the same weights (design note section 3.4).
";

const EVALUATION_MAD_HEADER: &str = "\
# Evaluation IR (spec 10) for plan N's experiment 2, the MAD policy with all three cameras --
# packet M15/N8b (docs/design/multi-camera.md sections 3.5 and 4).
#
# Generated by GENERATED;
# evaluation-augmented.toml (the document M7/U row U3 was judged by) with `task` and
# `observation` replaced by task-views.toml's and observation-views.toml's own hashes -- the two
# documents the MAD bundle reads, since learning-mad.toml changes the Learning IR only. So this
# body is evaluation-views.toml's, field for field, and so is `evaluation_hash`: the MAD policy
# with three cameras is judged under experiment 1's three-view conditions. Same 16 held-out seeds
# 101-116, same six suites, same metrics, same acceptance (`success_rate >= 0.5` on nominal);
# `views_documents_are_what_the_generator_derives` checks all of that.
#
# A file of its own, as training-cam.toml and training-views.toml are, so that cycle-mad.toml
# names its arm's document. A bundle `es policy subset` derives reads fewer cameras -- another
# Observation IR -- and is judged under a document naming that one (XIR-040, spec 13.3); this
# is not that document.
";

fn header(text: &str) -> String {
    text.replace("GENERATED", GENERATED)
}

/// Writes the seven generated documents of packet M15/N1 and the two of M15/N8b. Run explicitly:
///
///     ES_GENERATE_GOLDENS=1 cargo test -p es --test views -- --ignored generate_views_documents
#[test]
#[ignore = "fixture generator; run explicitly"]
fn generate_views_documents() {
    use es_ir::serial::{evaluation_to_toml, learning_to_toml, observation_to_toml, task_to_toml};
    // Spec 1.4: goldens and fixtures are CI read-only, and `cargo test -- --include-ignored`
    // runs every ignored test; a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_views_documents: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    let d = derive();
    let write = |name: &str, head: String, body: String| {
        std::fs::write(vl(name), format!("{head}\n{body}")).expect("write");
    };
    let eval_head = |arm: &str, obs: &str| {
        header(EVALUATION_HEADER)
            .replace("ARM", arm)
            .replace("OBSERVATION", obs)
    };
    write(
        "task-views.toml",
        header(TASK_VIEWS_HEADER),
        task_to_toml(&d.task).expect("toml"),
    );
    write(
        "observation-cam.toml",
        header(OBSERVATION_CAM_HEADER),
        observation_to_toml(&d.obs_cam).expect("toml"),
    );
    write(
        "observation-views.toml",
        header(OBSERVATION_VIEWS_HEADER),
        observation_to_toml(&d.obs_views).expect("toml"),
    );
    write(
        "learning-cam.toml",
        header(LEARNING_CAM_HEADER),
        learning_to_toml(&d.learning_cam).expect("toml"),
    );
    write(
        "learning-views.toml",
        header(LEARNING_VIEWS_HEADER),
        learning_to_toml(&d.learning_views).expect("toml"),
    );
    write(
        "evaluation-cam.toml",
        eval_head("the one-view arm", "observation-cam.toml"),
        evaluation_to_toml(&d.eval_cam).expect("toml"),
    );
    write(
        "evaluation-views.toml",
        eval_head("the three-view arm", "observation-views.toml"),
        evaluation_to_toml(&d.eval_views).expect("toml"),
    );
    write(
        "learning-mad.toml",
        header(LEARNING_MAD_HEADER),
        learning_to_toml(&d.learning_mad).expect("toml"),
    );
    write(
        "evaluation-mad.toml",
        header(EVALUATION_MAD_HEADER),
        evaluation_to_toml(&d.eval_mad).expect("toml"),
    );
    println!(
        "task-views {}\nobservation-cam {}\nobservation-views {}\nlearning-cam {}\n\
         learning-views {}\nlearning-mad {}\nevaluation-cam {}\nevaluation-views {}\n\
         evaluation-mad {}",
        hex(&d.task.task_hash().expect("hash")),
        hex(&d.obs_cam.observation_hash().expect("hash")),
        hex(&d.obs_views.observation_hash().expect("hash")),
        hex(&d.learning_cam.learning_hash().expect("hash")),
        hex(&d.learning_views.learning_hash().expect("hash")),
        hex(&d.learning_mad.learning_hash().expect("hash")),
        hex(&d.eval_cam.evaluation_hash().expect("hash")),
        hex(&d.eval_views.evaluation_hash().expect("hash")),
        hex(&d.eval_mad.evaluation_hash().expect("hash")),
    );
}

// --- the documents ------------------------------------------------------------------------------

/// The committed documents are exactly what the generator derives from the committed U3
/// documents and the views scene -- and each derivation undone gives back its source: the Task
/// IR without the two channels is task.toml, the Evaluation IRs with the two references put back
/// are evaluation-augmented.toml, the Observation and Learning IRs add up to U3's minus the
/// privileged branch.
#[test]
fn views_documents_are_what_the_generator_derives() {
    let d = derive();
    let task = es_ir::serial::task_from_toml(&read(&vl("task-views.toml"))).expect("task-views");
    assert_eq!(task, d.task, "task-views.toml is stale");
    let obs = |n: &str| es_ir::serial::observation_from_toml(&read(&vl(n))).expect(n);
    assert_eq!(obs("observation-cam.toml"), d.obs_cam);
    assert_eq!(obs("observation-views.toml"), d.obs_views);
    let learning = |n: &str| es_ir::serial::learning_from_toml(&read(&vl(n))).expect(n);
    assert_eq!(learning("learning-cam.toml"), d.learning_cam);
    assert_eq!(learning("learning-views.toml"), d.learning_views);
    assert_eq!(learning("learning-mad.toml"), d.learning_mad);
    let eval = |n: &str| es_ir::serial::evaluation_from_toml(&read(&vl(n))).expect(n);
    let (cam, views, mad) = (
        eval("evaluation-cam.toml"),
        eval("evaluation-views.toml"),
        eval("evaluation-mad.toml"),
    );
    assert_eq!(cam, d.eval_cam);
    assert_eq!(views, d.eval_views);
    assert_eq!(mad, d.eval_mad);
    // The MAD bundle reads the three-view arm's documents, so its evaluation is that arm's.
    assert_eq!(mad, views);
    assert_eq!(mad.evaluation_hash(), views.evaluation_hash());

    // Undone: task.toml, node for node.
    let committed = es_ir::serial::task_from_toml(&read(&vl("task.toml"))).expect("task.toml");
    let mut undone = task.clone();
    undone.scene = committed.scene.clone();
    for (_, channel, _) in NEW_VIEWS {
        undone.observation_spec.channels.remove(channel);
    }
    undone
        .graph
        .nodes
        .retain(|id, _| committed.graph.nodes.contains_key(id));
    undone
        .graph
        .edges
        .retain(|e| committed.graph.nodes.contains_key(&e.to.node));
    assert_eq!(
        undone, committed,
        "the rest of the task graph is not task.toml's"
    );
    assert_eq!(task.observation_spec.channels.len(), 5);

    // Undone: evaluation-augmented.toml, suite for suite.
    let u3 = eval("evaluation-augmented.toml");
    for mut ev in [cam, views, mad] {
        ev.task.clone_from(&u3.task);
        ev.observation.clone_from(&u3.observation);
        assert_eq!(ev, u3);
    }
    println!("RAN views_documents_are_what_the_generator_derives");
}

/// Each arm's four documents and its Evaluation IR validate, cross-check (`es ir check`, the
/// XIR pass, spec 11.1) and lower to a CPU plan (`es task compile`) -- the three independent
/// image chains included.
#[test]
fn views_documents_validate_check_and_compile() {
    let es = |args: &[&str]| -> String {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_es"));
        cmd.current_dir(repo());
        for a in args {
            if Path::new(a).extension().is_some_and(|e| e == "toml") {
                cmd.arg(vl(a));
            } else {
                cmd.arg(a);
            }
        }
        let out = cmd.output().expect("run es");
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            out.status.success() && !text.contains("ERROR"),
            "es {args:?}\n{text}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        text
    };
    for arm in ["cam", "views", "mad"] {
        let (obs, learning, eval) = (
            format!("observation-{}.toml", observation_of(arm)),
            format!("learning-{arm}.toml"),
            format!("evaluation-{arm}.toml"),
        );
        let docs = ["task-views.toml", &obs, &learning, "deployment.toml", &eval];
        let valid = es(&[&["ir", "validate"][..], &docs].concat());
        assert!(valid.contains("evaluation_hash: "), "{valid}");
        es(&[&["ir", "check"][..], &docs].concat());
        let plan = es(&["task", "compile", "task-views.toml", &obs]);
        let images = if arm == "cam" { 1 } else { 3 };
        assert_eq!(
            plan.matches("U8  shape=[96, 96, 3]").count(),
            images,
            "{plan}"
        );
    }
    println!("RAN views_documents_validate_check_and_compile");
}

/// The second of the expert gate's two checks (`XIR-040` against the Evaluation IR), errors
/// only -- as `crates/es/tests/cli.rs`'s `gate_errors` makes it.
fn gate_errors(bundle: &es_compile::PolicyBundle, evaluation: &str) -> Vec<String> {
    let eval_ir = es_ir::serial::evaluation_from_toml(&read(&vl(evaluation))).expect("parses");
    xir_errors(bundle, &eval_ir)
}

fn xir_errors(bundle: &es_compile::PolicyBundle, eval_ir: &EvaluationIr) -> Vec<String> {
    es_ir::cross::check(&es_ir::cross::IrBundle {
        task: &bundle.task,
        observation: &bundle.observation,
        learning: &bundle.learning,
        deployment: &bundle.deployment,
        evaluation: Some(eval_ir),
    })
    .into_iter()
    .filter(es_ir::diag::Diagnostic::is_error)
    .map(|d| d.to_string())
    .collect()
}

fn scratch(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("es-views-{tag}-{nanos}"))
}

/// The Observation IR an arm reads: MAD (experiment 2) reads the three-view arm's.
fn observation_of(arm: &str) -> &str {
    if arm == "mad" {
        "views"
    } else {
        arm
    }
}

/// `es policy init` of one arm's bundle into `<dir>/<arm>.esb`: its Observation and Learning IR
/// over task-views.toml and the committed deployment.toml, as the cycles' headers write it.
fn policy_init(dir: &Path, arm: &str) -> PathBuf {
    let out = dir.join(format!("{arm}.esb"));
    let fixture = |n: &str| format!("tests/fixtures/visible-learning/{n}");
    let o = Command::new(env!("CARGO_BIN_EXE_es"))
        .current_dir(repo())
        .args(["policy", "init", "--task", &fixture("task-views.toml")])
        .args([
            "--observation",
            &fixture(&format!("observation-{}.toml", observation_of(arm))),
        ])
        .args(["--learning", &fixture(&format!("learning-{arm}.toml"))])
        .args(["--deployment", &fixture("deployment.toml")])
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run es policy init");
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    out
}

/// `es policy init` builds each arm's untrained bundle from its four documents (the committed
/// `deployment.toml` reused), and the bundle passes the checks the expert gate makes before it
/// opens a backend: `PolicyBundle::open` (XIR-010) and XIR-040 against the arm's own Evaluation
/// IR -- and not against the other arm's (packet M12/Y5b's two calls, no Python).
#[test]
fn policy_init_builds_each_arms_bundle_and_the_gate_accepts_it() {
    let dir = scratch("policy-init");
    let init = |arm: &str| -> es_compile::PolicyBundle {
        let bytes = std::fs::read(policy_init(&dir, arm)).expect("the bundle was written");
        es_compile::PolicyBundle::open(&bytes).expect("the bundle opens (XIR-010 included)")
    };
    let (cam, views) = (init("cam"), init("views"));
    assert_eq!(
        gate_errors(&cam, "evaluation-cam.toml"),
        Vec::<String>::new()
    );
    assert_eq!(
        gate_errors(&views, "evaluation-views.toml"),
        Vec::<String>::new()
    );
    for (bundle, other) in [
        (&cam, "evaluation-views.toml"),
        (&views, "evaluation-cam.toml"),
    ] {
        let refused = gate_errors(bundle, other);
        assert!(refused.iter().any(|d| d.contains("XIR-040")), "{refused:?}");
    }
    // One Task IR under both arms: one collection serves both.
    assert_eq!(cam.manifest.hashes.task, views.manifest.hashes.task);
    assert_eq!(cam.learning.policy.contract.inputs.len(), 2);
    assert_eq!(views.learning.policy.contract.inputs.len(), 4);
    // Both graphs lower to the module the trainer optimizes: one pretrained backbone per view,
    // and a fusion projecting the concatenated features to 512.
    for (bundle, views, fused) in [(&cam, 1, 2 * 512), (&views, 3, 4 * 512)] {
        let module = es_policy::lower_to_torch(&bundle.learning).expect("the graph lowers");
        assert_eq!(
            module.source.matches("= _frozen_backbone(").count(),
            views,
            "{}",
            module.source
        );
        assert!(
            module.source.contains(&format!("nn.Linear({fused}, 512)")),
            "{}",
            module.source
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    println!("RAN policy_init_builds_each_arms_bundle_and_the_gate_accepts_it");
}

/// Packet M15/N8b: `es policy init` builds the MAD bundle, the gate accepts it under
/// evaluation-mad.toml, it lowers to **one** backbone applied to the three views and summed,
/// `[run] single_view` accepts its graph, and `es policy subset` keeps each camera alone --
/// with, packet M15/N8c, the Evaluation IR each subset is judged by: evaluation-mad.toml with
/// only its references changed, and a wrong parent document refused by name.
///
/// The init bundle's weights are a placeholder the subset refuses (it rewrites a checkpoint
/// that fits the lowering), so the subset runs on a bundle of the same four documents with
/// weights that fit -- as N8's own CLI test does. No Python.
#[test]
fn the_mad_bundle_builds_trains_one_view_at_a_time_and_subsets() {
    use es_ir::learning::WeightsRef;
    use es_policy::weights::{write_safetensors, Checkpoint};

    let dir = scratch("mad");
    let bytes = std::fs::read(policy_init(&dir, "mad")).expect("the bundle was written");
    let mad = es_compile::PolicyBundle::open(&bytes).expect("the bundle opens (XIR-010)");
    assert_eq!(
        gate_errors(&mad, "evaluation-mad.toml"),
        Vec::<String>::new()
    );
    assert_eq!(mad.learning.policy.contract.inputs.len(), 4);
    let module = es_policy::lower_to_torch(&mad.learning).expect("the graph lowers");
    assert_eq!(
        module.source.matches("= _frozen_backbone(").count(),
        1,
        "{}",
        module.source
    );
    assert!(
        module.source.contains("nn.Linear(1024, 512)"),
        "{}",
        module.source
    );
    es_data::training::check_single_view(&mad.learning).expect("single_view takes the graph");

    let mut file: Checkpoint = module
        .weight_shapes
        .iter()
        .map(|(k, s)| {
            let n = s.iter().product::<u64>() as usize;
            (k.clone(), (s.clone(), vec![0.25; n]))
        })
        .collect();
    for claim in module.weight_keys.iter().filter(|k| k.ends_with(".*")) {
        file.insert(
            format!("{}fc.bias", claim.trim_end_matches('*')),
            (vec![1], vec![1.0]),
        );
    }
    let weights = write_safetensors(&file);
    let mut learning = mad.learning.clone();
    learning.policy.weights = WeightsRef::Safetensors {
        path: learning.policy.weights.path().to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let fitted = dir.join("mad-fitted.esb");
    let built = es_compile::PolicyBundle::build(
        &mad.task,
        &mad.observation,
        &learning,
        &mad.deployment,
        &weights,
    )
    .expect("the MAD documents pack with fitting weights");
    std::fs::write(&fitted, built).expect("write");
    let parent_policy = hex(&learning.policy_hash().expect("policy hash"));
    let judged_by = vl("evaluation-mad.toml");
    let committed = read(&judged_by);
    let parent_ev = es_ir::serial::evaluation_from_toml(&committed).expect("parses");
    let subset = |view: &str, out: &Path, evaluation: &Path| -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_es"));
        cmd.args(["policy", "subset", "--policy"])
            .arg(&fitted)
            .args(["--views", view, "--out"])
            .arg(out)
            .arg("--evaluation")
            .arg(evaluation);
        cmd
    };
    for view in ["rgb_overhead", "rgb_wrist", "rgb_side"] {
        let out = dir.join(format!("{view}.esb"));
        // One names where its Evaluation IR goes; the others take the default beside the bundle.
        let mut cmd = subset(view, &out, &judged_by);
        let written = if view == "rgb_overhead" {
            let to = dir.join("judged").join("overhead.toml");
            cmd.arg("--evaluation-out").arg(&to);
            to
        } else {
            dir.join(format!("{view}.evaluation.toml"))
        };
        let o = cmd.output().expect("run es policy subset");
        let text = String::from_utf8_lossy(&o.stdout);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{view}: {text}{}",
            String::from_utf8_lossy(&o.stderr)
        );
        // Kept alone, a sharer takes the overhead encoder's tensors under its own id.
        assert_eq!(
            text.contains("node 0 -> node"),
            view != "rgb_overhead",
            "{text}"
        );
        let sub = es_compile::PolicyBundle::open(&std::fs::read(&out).expect("written"))
            .expect("the subset opens");
        let inputs: Vec<&String> = sub.learning.policy.contract.inputs.keys().collect();
        assert_eq!(inputs, ["joint_state", view]);

        // Packet M15/N8c (design note 3.5): the parent's document no longer judges the subset,
        // the written one does, and it is the parent's with the references changed.
        assert!(
            xir_errors(&sub, &parent_ev)
                .iter()
                .any(|d| d.contains("XIR-040")),
            "{view}"
        );
        assert!(
            text.contains(&format!("evaluation:    {}", written.display())),
            "{text}"
        );
        let doc = read(&written);
        let ev = es_ir::serial::evaluation_from_toml(&doc).expect("the written evaluation parses");
        assert_eq!(xir_errors(&sub, &ev), Vec::<String>::new(), "{view}");
        assert!(ev.validate().is_empty(), "{view}");
        assert_eq!(ev.task, hex(&sub.task.task_hash().expect("hash")));
        assert_eq!(
            ev.observation,
            hex(&sub.observation.observation_hash().expect("hash"))
        );
        assert_ne!(ev.observation, parent_ev.observation, "{view}");
        let mut undone = ev.clone();
        undone.task.clone_from(&parent_ev.task);
        undone.observation.clone_from(&parent_ev.observation);
        assert_eq!(undone, parent_ev, "{view}: more than the references differ");
        // The overhead camera alone reads the one-view arm's Observation IR, so experiment 2
        // judges it under experiment 1's own document.
        if view == "rgb_overhead" {
            let cam = es_ir::serial::evaluation_from_toml(&read(&vl("evaluation-cam.toml")));
            assert_eq!(ev, cam.expect("evaluation-cam.toml"));
        }
        // Byte for byte below the headers: the Task IR is carried over, so only the observation
        // line differs.
        let body = |t: &str| -> Vec<String> {
            t.lines()
                .filter(|l| !l.starts_with('#'))
                .map(str::to_owned)
                .collect()
        };
        let (mine, theirs) = (body(&doc), body(&committed));
        assert_eq!(mine.len(), theirs.len(), "{view}");
        let differ: Vec<&String> = mine
            .iter()
            .zip(&theirs)
            .filter(|(a, b)| a != b)
            .map(|(a, _)| a)
            .collect();
        assert_eq!(differ, [&format!("observation = \"{}\"", ev.observation)]);
        for line in [
            format!("# parent document:        {}", judged_by.display()),
            format!(
                "# parent evaluation_hash: {}",
                hex(&parent_ev.evaluation_hash().expect("hash"))
            ),
            format!("# parent policy_hash:     {parent_policy}"),
            format!("# views kept:             {view}"),
        ] {
            assert!(doc.lines().any(|l| l == line), "{line}\n{doc}");
        }
    }

    // A wrong parent document -- the one-view arm's -- is refused by name, and nothing is written.
    let wrong = dir.join("wrong.esb");
    let o = subset("rgb_wrist", &wrong, &vl("evaluation-cam.toml"))
        .output()
        .expect("run es policy subset");
    let err = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(1), "{err}");
    assert!(
        err.contains("evaluation-cam.toml does not judge the parent bundle")
            && err.contains("XIR-040"),
        "{err}"
    );
    assert!(!wrong.exists() && !dir.join("wrong.evaluation.toml").exists());
    let _ = std::fs::remove_dir_all(&dir);
    println!("RAN the_mad_bundle_builds_trains_one_view_at_a_time_and_subsets");
}

fn toml_value(name: &str) -> toml::Value {
    toml::from_str(&read(&vl(name))).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Both arms train on U3's run settings -- each recipe's body is training-hint-u3.toml's, value
/// for value -- and each cycle is cycle-hint-u3.toml with the scene, the recipe and the
/// Evaluation IR moved and `[eval.preview]` added, nothing else. `--dry-run` prints each plan.
#[test]
fn views_recipes_and_cycles_mirror_the_hint_card() {
    let u3 = toml_value("training-hint-u3.toml");
    for arm in ["cam", "views"] {
        assert_eq!(
            toml_value(&format!("training-{arm}.toml")),
            u3,
            "training-{arm}.toml"
        );

        let mut want = toml_value("cycle-hint-u3.toml");
        let t = want.as_table_mut().expect("a table");
        t.insert("scene".into(), VIEWS_SCENE.into());
        let fixture = |n: String| format!("tests/fixtures/visible-learning/{n}");
        t["train"].as_table_mut().expect("[train]").insert(
            "recipe".into(),
            fixture(format!("training-{arm}.toml")).into(),
        );
        let eval = t["eval"].as_table_mut().expect("[eval]");
        eval.insert(
            "config".into(),
            fixture(format!("evaluation-{arm}.toml")).into(),
        );
        eval.insert("preview".into(), toml::Value::Table(toml::Table::new()));
        assert_eq!(
            toml_value(&format!("cycle-{arm}.toml")),
            want,
            "cycle-{arm}.toml"
        );

        let out = std::env::temp_dir().join(format!("es-views-cycle-{arm}-dry"));
        let o = Command::new(env!("CARGO_BIN_EXE_es"))
            .current_dir(repo())
            .env_remove("ES_PYTHON")
            .args(["loop", "cycle", "--recipe"])
            .arg(format!("tests/fixtures/visible-learning/cycle-{arm}.toml"))
            .arg("--out")
            .arg(&out)
            .arg("--dry-run")
            .output()
            .expect("run es loop cycle");
        let plan = String::from_utf8_lossy(&o.stdout);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{plan}{}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(plan.contains(&format!("evaluation-{arm}.toml")), "{plan}");
        assert!(plan.contains(VIEWS_SCENE), "{plan}");
        // The bake reads the directory the collection writes: `collect/frames`, under which
        // each of task-views.toml's three channels has its own `<channel>/` (packet M15/N3).
        assert!(
            plan.contains("--out collect/ds --frames collect/frames")
                && plan.contains("--frames collect/frames collect/ds\n"),
            "{plan}"
        );
        assert!(plan.contains("preview"), "{plan}");
        assert!(!out.exists(), "a dry run writes nothing");
    }
    println!("RAN views_recipes_and_cycles_mirror_the_hint_card");
}

/// Packet M15/N8b: experiment 2's recipe is training-views.toml with 60,000 steps and their
/// marks and MAD's single-view loss -- U3's lr 4e-4 and no `grad_clip`, since packet P-M15-R2
/// found the camera-only divergence in the attention kernel and not in the rate; its cycle is
/// cycle-views.toml with the recipe and the Evaluation IR moved, nothing else.
///
/// The dry run plans `--single-view 0.5`. `[run] single_view` is checked against the recipe's
/// `[policy] bundle` before any plan, so the dry run runs in a scratch directory holding the
/// cycle's three documents at their relative paths and the MAD bundle `es policy init` builds
/// at `runs/collect-001/untrained.esb`.
#[test]
fn mad_recipe_and_cycle_are_the_views_arms_with_single_view() {
    let mut want = toml_value("training-views.toml");
    let run = want["run"].as_table_mut().expect("[run]");
    run.insert("steps".into(), 60_000.into());
    run.insert(
        "checkpoint_at".into(),
        toml::Value::Array([1000, 5000, 20000, 60000].map(toml::Value::from).to_vec()),
    );
    run.insert(
        "single_view".into(),
        toml::Value::Table(toml::Table::from_iter([("weight".to_owned(), 0.5.into())])),
    );
    assert_eq!(toml_value("training-mad.toml"), want, "training-mad.toml");

    let fixture = |n: &str| format!("tests/fixtures/visible-learning/{n}");
    let mut want = toml_value("cycle-views.toml");
    let t = want.as_table_mut().expect("a table");
    t["train"]
        .as_table_mut()
        .expect("[train]")
        .insert("recipe".into(), fixture("training-mad.toml").into());
    t["eval"]
        .as_table_mut()
        .expect("[eval]")
        .insert("config".into(), fixture("evaluation-mad.toml").into());
    assert_eq!(toml_value("cycle-mad.toml"), want, "cycle-mad.toml");

    let dir = scratch("mad-cycle");
    let docs = dir.join("tests/fixtures/visible-learning");
    std::fs::create_dir_all(&docs).expect("scratch");
    for name in ["cycle-mad.toml", "training-mad.toml", "evaluation-mad.toml"] {
        std::fs::copy(vl(name), docs.join(name)).expect("copy");
    }
    std::fs::create_dir_all(dir.join("runs/collect-001")).expect("scratch");
    std::fs::rename(
        policy_init(&dir, "mad"),
        dir.join("runs/collect-001/untrained.esb"),
    )
    .expect("the bundle in place");
    let o = Command::new(env!("CARGO_BIN_EXE_es"))
        .current_dir(&dir)
        .env_remove("ES_PYTHON")
        .args(["loop", "cycle", "--recipe"])
        .arg(fixture("cycle-mad.toml"))
        .args(["--out", "out", "--dry-run"])
        .output()
        .expect("run es loop cycle");
    let plan = String::from_utf8_lossy(&o.stdout);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{plan}{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let train: Vec<&str> = plan
        .lines()
        .filter(|l| l.contains("train_act.py"))
        .collect();
    assert!(
        train.len() == 1 && train[0].contains("--single-view 0.5"),
        "{plan}"
    );
    for want in ["--lr 0.0004 ", "--checkpoint-at 1000,5000,20000,60000 "] {
        assert!(train[0].contains(want), "{want}: {plan}");
    }
    assert!(!train[0].contains("--grad-clip"), "{plan}");
    assert!(
        plan.contains("evaluation-mad.toml") && plan.contains("preview"),
        "{plan}"
    );
    assert!(!dir.join("out").exists(), "a dry run writes nothing");
    println!("train line: {}", train[0]);
    let _ = std::fs::remove_dir_all(&dir);
    println!("RAN mad_recipe_and_cycle_are_the_views_arms_with_single_view");
}

// --- baking several cameras (packet M15/N3) -------------------------------------------------------

/// task-views.toml's image channels, in channel-name order.
const CHANNELS: [&str; 3] = ["rgb_overhead", "rgb_side", "rgb_wrist"];

/// The one byte every pixel of camera `k`'s tile at global frame `g` holds in collection
/// `salt`: a tensor baked from it says which camera, which frame and which collection it is.
fn tile_byte(salt: usize, k: usize, g: usize) -> u8 {
    (7 + 50 * k + 3 * g + 100 * salt) as u8
}

/// A collection as `es loop collect --frames` writes it for task-views.toml: the columns the
/// bake reads, a video feature per image channel, and each camera's tiles in its own
/// `<frames>/<channel>/` (packet M15/N2).
fn write_views_collection(root: &Path, frames: &Path, lengths: &[usize], salt: usize) {
    use es_data::{Column, Dtype, Episode, FeatureSpec, Info, LeRobotWriter};

    // `qpos || qvel` of the views scene, as `es_data::collect::to_lerobot` writes it.
    let (state, dof) = (13 + 12, 6);
    let mut features = BTreeMap::from([
        (
            "observation.state".to_owned(),
            FeatureSpec::new(Dtype::Float32, [state as u64]),
        ),
        (
            "action".to_owned(),
            FeatureSpec::new(Dtype::Float32, [dof as u64]),
        ),
    ]);
    for channel in CHANNELS {
        features.insert(
            format!("observation.images.{channel}"),
            FeatureSpec::new(Dtype::Video, [96, 96, 3]),
        );
    }
    // The labels `es loop distill` remaps.
    for label in [es_data::INTERVENTION, es_data::ACTION_SOURCE] {
        features.insert(label.to_owned(), FeatureSpec::new(Dtype::Int64, [1]));
    }
    let mut writer = LeRobotWriter::create(root, Info::new(50.0, features)).expect("create");
    let mut g = 0;
    for (index, n) in lengths.iter().copied().enumerate() {
        let ramp = |w: usize| Column::F32((0..n * w).map(|i| (i % 7) as f32 / 7.0 - 0.5).collect());
        writer
            .write_episode(&Episode {
                index: index as u32,
                tasks: vec!["views".to_owned()],
                timestamps: (0..n).map(|i| i as f64 / 50.0).collect(),
                task_index: vec![0; n],
                columns: BTreeMap::from([
                    ("observation.state".to_owned(), ramp(state)),
                    ("action".to_owned(), ramp(dof)),
                    (es_data::INTERVENTION.to_owned(), Column::I64(vec![0; n])),
                    (es_data::ACTION_SOURCE.to_owned(), Column::I64(vec![0; n])),
                ]),
                video: BTreeMap::new(),
            })
            .expect("write episode");
        for _ in 0..n {
            for (k, channel) in CHANNELS.iter().enumerate() {
                let dir = frames.join(channel);
                std::fs::create_dir_all(&dir).expect("camera dir");
                let tile = vec![tile_byte(salt, k, g); 96 * 96 * 3];
                std::fs::write(dir.join(format!("{g:06}.bin")), tile).expect("tile");
            }
            g += 1;
        }
    }
    writer.finish().expect("finish");
}

/// One tensor of a safetensors file: its shape and its F32 values.
fn tensor(bytes: &[u8], name: &str) -> (Vec<u64>, Vec<f32>) {
    let header = es_policy::weights::parse_header(bytes).expect("safetensors");
    let entry = header.get(name).unwrap_or_else(|| panic!("no {name}"));
    let base = 8 + u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let data = &bytes[base + entry.offsets.0 as usize..base + entry.offsets.1 as usize];
    let values = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
        .collect();
    (entry.shape.clone(), values)
}

/// Packet M15/N3's oracle: `es dataset bake` hands each image port its own camera's frames.
/// Each of the three-view arm's three image tensors is, frame by frame, the one byte its own
/// camera's tile holds at that global frame -- three different tensors, not one camera read
/// three times -- and the one-view arm, baked from the same three-camera collection, reads the
/// overhead camera's directory: its tensors are the three-view arm's, bit for bit.
///
/// task-views.toml has two `JointState` channels, so the bake resolves `joint_state` against
/// the scene (packet M5/V7a) and needs the `MuJoCo` backend: `ES_PYTHON` with `mujoco`, or a
/// skip.
#[test]
fn bake_reads_each_image_port_from_its_own_camera() {
    if let Err(reason) = es_physics_backend::MuJoCoCpuBackend::is_available() {
        println!("SKIP bake_reads_each_image_port_from_its_own_camera: {reason}");
        return;
    }
    let dir = scratch("bake");
    let (root, frames) = (dir.join("ds"), dir.join("frames"));
    let lengths = [3, 2];
    write_views_collection(&root, &frames, &lengths, 0);
    let bake = |arm: &str, flags: &[&str]| -> (Vec<Vec<u8>>, serde_json::Value) {
        let out = dir.join(format!("baked-{arm}{}", flags.concat()));
        let o = Command::new(env!("CARGO_BIN_EXE_es"))
            .current_dir(repo())
            .args(["dataset", "bake"])
            .args(flags)
            .arg("--policy")
            .arg(policy_init(&dir, arm))
            .arg("--out")
            .arg(&out)
            .arg("--frames")
            .arg(&frames)
            .arg(&root)
            .output()
            .expect("run es dataset bake");
        assert_eq!(
            o.status.code(),
            Some(0),
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        let manifest = std::fs::read_to_string(out.join("manifest.json")).expect("manifest");
        let episodes = (0..lengths.len())
            .map(|e| std::fs::read(out.join(format!("episode_{e:06}.safetensors"))).expect("ep"))
            .collect();
        (
            episodes,
            serde_json::from_str(&manifest).expect("manifest.json"),
        )
    };
    let ((views, _), (cam, _)) = (bake("views", &[]), bake("cam", &[]));

    let mut g = 0;
    for (e, n) in lengths.iter().copied().enumerate() {
        let mut seen = Vec::new();
        for (k, channel) in CHANNELS.iter().enumerate() {
            let (shape, values) = tensor(&views[e], channel);
            assert_eq!(shape, [n as u64, 3, 96, 96], "{channel}");
            for (t, frame) in values.chunks_exact(3 * 96 * 96).enumerate() {
                let want = f32::from(tile_byte(0, k, g + t)) / 255.0;
                assert!(
                    frame.iter().all(|v| (v - want).abs() < 1e-6),
                    "episode {e} frame {t} {channel}: {} is not camera {k}'s {want}",
                    frame[0]
                );
            }
            seen.push(values);
        }
        assert!(seen[0] != seen[1] && seen[1] != seen[2] && seen[0] != seen[2]);
        for name in ["rgb_overhead", "joint_state", "action"] {
            assert_eq!(tensor(&cam[e], name), tensor(&views[e], name), "{name}");
        }
        assert!(
            es_policy::weights::parse_header(&cam[e])
                .expect("safetensors")
                .keys()
                .all(|k| !k.contains("wrist") && !k.contains("side")),
            "the one-view arm bakes one image"
        );
        g += n;
    }

    // What `es train` bakes for the three-view arm: `--for-training`, each view at its `Pad(4)`
    // boundary. Its bytes per frame are the host RAM `train_act.py` holds per frame of the set.
    let (_, manifest) = bake("views", &["--for-training"]);
    let tensors = manifest["tensors"].as_object().expect("tensors");
    for channel in CHANNELS {
        assert_eq!(tensors[channel]["shape"], serde_json::json!([3, 104, 104]));
    }
    let per_frame: u64 = tensors
        .values()
        .map(|t| {
            let shape = t["shape"].as_array().expect("a shape");
            4 * shape
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .product::<u64>()
        })
        .sum();
    println!("three-view training bake: {per_frame} bytes per frame");
    let _ = std::fs::remove_dir_all(&dir);
    println!("RAN bake_reads_each_image_port_from_its_own_camera");
}

/// `es loop distill --frames` (a cycle's `[collect] merge`) merges a three-camera collection
/// camera by camera: the second input's tile `i` of each camera lands at `offset + i` in that
/// camera's own directory, never in another's (packet M15/N3).
#[test]
fn distill_merges_each_cameras_tiles_into_its_own_directory() {
    let dir = scratch("distill");
    let (a, b) = (dir.join("a/ds"), dir.join("b/ds"));
    let (fa, fb, merged) = (
        dir.join("a/frames"),
        dir.join("b/frames"),
        dir.join("merged"),
    );
    write_views_collection(&a, &fa, &[2], 0);
    write_views_collection(&b, &fb, &[3], 1);
    let o = Command::new(env!("CARGO_BIN_EXE_es"))
        .args(["loop", "distill", "--in"])
        .arg(&a)
        .arg("--in-frames")
        .arg(&fa)
        .arg("--in")
        .arg(&b)
        .arg("--in-frames")
        .arg(&fb)
        .args(["--train", "1", "--val", "0", "--test", "0", "--out"])
        .arg(&merged)
        .arg("--frames")
        .arg(&fa)
        .output()
        .expect("run es loop distill");
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    for (k, channel) in CHANNELS.iter().enumerate() {
        for (g, want) in [(0, tile_byte(0, k, 0)), (1, tile_byte(0, k, 1))]
            .into_iter()
            .chain((0..3).map(|i| (2 + i, tile_byte(1, k, i))))
        {
            let tile = std::fs::read(fa.join(channel).join(format!("{g:06}.bin"))).expect("tile");
            assert!(tile.iter().all(|b| *b == want), "{channel} {g}");
        }
    }
    assert!(!fa.join("000000.bin").exists(), "no flat tile");
    let _ = std::fs::remove_dir_all(&dir);
    println!("RAN distill_merges_each_cameras_tiles_into_its_own_directory");
}

// --- the scene ------------------------------------------------------------------------------------

/// The views scene is the demo scene plus two cameras and nothing else: without them it hashes
/// to the committed scene's `scene_hash`, which covers every body, geom, joint, actuator, option
/// and the overhead camera (spec 5.3). The wrist camera rides on `camera_mount`; the side camera
/// is fixed to the world.
#[test]
fn the_views_scene_is_the_demo_scene_plus_two_cameras() {
    let demo = parse_scene(DEMO_SCENE);
    let views = parse_scene(VIEWS_SCENE);
    assert_ne!(views.scene_hash(), demo.scene_hash());
    let mut stripped = views.clone();
    stripped
        .cameras
        .retain(|c| !NEW_VIEWS.iter().any(|(name, ..)| *name == c.name));
    assert_eq!(stripped.cameras.len(), 1);
    assert_eq!(stripped.scene_hash(), demo.scene_hash());

    let mount = views
        .bodies
        .iter()
        .find(|b| b.name == "camera_mount")
        .expect("camera_mount")
        .id;
    assert_eq!(camera(&views, "wrist").body, Some(mount));
    assert_eq!(camera(&views, "side").body, None);
    assert_eq!(camera(&views, "overhead").id, camera(&demo, "overhead").id);

    // The provenance names the file it was derived from by content.
    let provenance: serde_json::Value = serde_json::from_str(&read(
        &repo().join("tests/fixtures/mjcf/so101_pick_place_views.PROVENANCE.json"),
    ))
    .expect("the provenance is JSON");
    let demo_xml = std::fs::read(repo().join(DEMO_SCENE)).expect("the demo scene");
    assert_eq!(
        provenance["derived_from_blake3"],
        blake3::hash(&demo_xml).to_hex().as_str()
    );
    println!("RAN the_views_scene_is_the_demo_scene_plus_two_cameras");
}

#[cfg(feature = "render")]
mod render {
    use super::*;
    use es_assets::scene::JointKind;
    use es_env::render::{camera_view, check_image_spec, image_spec, render_config};
    use es_env::EnvRendererCfg;
    use es_math::{Pose, Quat, Vec3};
    use es_render::{cpu, Channel, TriScene};

    /// Every body's world pose with the named hinges at the given angles and every other joint at
    /// zero -- the Task IR's reset pose when `q` is empty. An MJCF hinge turns the child frame about
    /// `axis` through `anchor`, both in the child's frame: `body.pose * T(a) * R * T(-a)`.
    fn forward(scene: &SceneDesc, q: &[(&str, f64)]) -> BTreeMap<StableId, Pose> {
        let mut world: BTreeMap<StableId, Pose> = BTreeMap::new();
        for body in &scene.bodies {
            let mut local = body.pose;
            for joint in scene.joints.iter().filter(|j| j.body == body.id) {
                let angle = q
                    .iter()
                    .find(|(n, _)| *n == joint.name)
                    .map_or(0.0, |p| p.1);
                if joint.kind != JointKind::Hinge || angle == 0.0 {
                    continue;
                }
                let axis = joint.axis.normalize();
                let (s, c) = ((angle * 0.5).sin(), (angle * 0.5).cos());
                local = local
                    .compose(Pose::new(
                        joint.anchor,
                        Quat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c),
                    ))
                    .compose(Pose::new(-joint.anchor, Quat::IDENTITY));
            }
            let pose = body
                .parent
                .and_then(|p| world.get(&p).copied())
                .map_or(local, |p| p.compose(local));
            world.insert(body.id, pose);
        }
        world
    }

    const SIDE: u32 = 96;

    /// The cube's positions the oracle pins: the four corners of task.toml's draw
    /// (x 0.21-0.27, y -0.03-0.05, z 0.02) and a point inside it.
    const CUBES: [(f64, f64); 5] = [
        (0.21, -0.03),
        (0.21, 0.05),
        (0.27, -0.03),
        (0.27, 0.05),
        (0.24, 0.01),
    ];

    /// The fewest cube-coloured pixels a camera must show. `ResNet18`'s stem (a stride-2
    /// convolution, then a stride-2 max-pool) turns a 96x96 frame into a 24x24 grid of 4x4-pixel
    /// cells: at 16 pixels the cube covers at least one cell's worth of the frame, so it is more
    /// than a speck the stem can average away. The measured counts are printed beside it.
    const FLOOR: usize = 16;

    fn body(scene: &SceneDesc, name: &str) -> StableId {
        scene
            .bodies
            .iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("the views scene has a `{name}` body"))
            .id
    }

    /// The CPU reference's `Rgb8` frame of camera `name` for `world` -- the golden path every
    /// observation frame is compared against (`crates/es-env/tests/render_loop.rs`).
    fn frame(scene: &SceneDesc, name: &str, world: &BTreeMap<StableId, Pose>) -> Vec<u8> {
        let cfg = EnvRendererCfg::rgb(camera(scene, name).id, SIDE, SIDE);
        let tri = TriScene::from_scene_with_poses(scene, world).expect("the scene tessellates");
        let view = camera_view(scene, &cfg, world).expect("the camera resolves");
        cpu::rasterize(&tri, &view, &render_config(&cfg), 0)
            .tile(Channel::Rgb8)
            .expect("Rgb8 was requested")
            .as_u8()
            .expect("Rgb8 is bytes")
            .to_vec()
    }

    /// Pixels of the cube's colour (`rgba="0.85 0.2 0.15 1"`) under the `Rs` path's flat
    /// shading: every lit or shadowed face keeps green and blue at about half of red (measured:
    /// 173/88/77, 225/117/102, 100/48/41). Nothing else in the scene is near that hue -- the
    /// jaws' collision boxes are pure red (green = blue = 0), the arm yellow, the table beige,
    /// the bin blue -- so the count is the cube's.
    fn cube_pixels(rgb: &[u8]) -> usize {
        rgb.chunks(3)
            .filter(|p| {
                let (r, g, b) = (u32::from(p[0]), u32::from(p[1]), u32::from(p[2]));
                r >= 64
                    && (35 * r..=65 * r).contains(&(100 * g))
                    && (30 * r..=60 * r).contains(&(100 * b))
                    && b < g
            })
            .count()
    }

    /// `ES_VIEWS_DUMP=<dir>` writes the frame there as a binary PPM, for a person to look at.
    fn dump(stem: &str, rgb: &[u8]) {
        let Ok(dir) = std::env::var("ES_VIEWS_DUMP") else {
            return;
        };
        std::fs::create_dir_all(&dir).expect("dump dir");
        let mut ppm = format!("P6\n{SIDE} {SIDE}\n255\n").into_bytes();
        ppm.extend_from_slice(rgb);
        std::fs::write(Path::new(&dir).join(format!("{stem}.ppm")), ppm).expect("write ppm");
    }

    fn with_cube(
        scene: &SceneDesc,
        mut world: BTreeMap<StableId, Pose>,
        (x, y): (f64, f64),
    ) -> BTreeMap<StableId, Pose> {
        world.insert(
            body(scene, "cube"),
            Pose::new(Vec3::new(x, y, 0.02), Quat::IDENTITY),
        );
        world
    }

    /// The first waypoint of the demonstration program (`templates/teach/so101-pick-place.toml`,
    /// block 1): the tool site 45 mm above the cube's centre, pitched -85 deg, jaws open -- the
    /// pose from which the arm descends onto the cube.
    fn approach(scene: &SceneDesc, (x, y): (f64, f64)) -> BTreeMap<StableId, Pose> {
        let links = es_env::Links::from_scene(scene).expect("the SO-101 chain");
        let target = Vec3::new(x, y, 0.02 + 0.045);
        let q = es_env::so101_ik(&links, target, (-85f64).to_radians()).expect("reachable");
        let names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"];
        let mut joints: Vec<(&str, f64)> = names.iter().copied().zip(q).collect();
        joints.push(("gripper", 0.9));
        let world = forward(scene, &joints);
        // `forward` is this file's own algebra; the closed-form IK is MuJoCo-checked
        // (`crates/es-env/tests/expert.rs`), so the tool site landing on the IK's target is
        // what says the rendered arm is the program's pose.
        let tool = scene
            .bodies
            .iter()
            .find(|b| b.sites.iter().any(|s| s.name == "gripperframe"))
            .expect("the tool body");
        let site = tool
            .sites
            .iter()
            .find(|s| s.name == "gripperframe")
            .expect("the tool site");
        let at = world[&tool.id].transform_point(site.pose.position);
        assert!((at - target).norm() < 1e-6, "{at:?} vs {target:?}");
        with_cube(scene, world, (x, y))
    }

    /// task-views.toml's three image channels name the scene's three cameras, and each declared
    /// `ImageSpec` is what the renderer produces for that camera (`check_image_spec`, INV-14).
    #[test]
    fn the_views_task_declares_what_the_scene_renders() {
        let scene = parse_scene(VIEWS_SCENE);
        let task = es_ir::serial::task_from_toml(&read(&vl("task-views.toml"))).expect("task");
        for (name, channel) in [
            ("overhead", "rgb_overhead"),
            ("wrist", "rgb_wrist"),
            ("side", "rgb_side"),
        ] {
            let ch = &task.observation_spec.channels[channel];
            match &ch.source {
                es_ir::task::ObsSource::Sensor { id, .. } => {
                    assert_eq!(*id, camera(&scene, name).id, "{channel}");
                }
                other => panic!("{channel}: {other:?}"),
            }
            let declared = ch.ty.image.as_ref().expect("an image channel");
            let cfg = EnvRendererCfg::rgb(camera(&scene, name).id, SIDE, SIDE);
            let ours = image_spec(&scene, &cfg).expect("the camera resolves");
            check_image_spec(&ours, declared).unwrap_or_else(|e| panic!("{channel}: {e}"));
            assert!(
                (ours.intrinsics.fx - declared.intrinsics.fx).abs() < 1e-3,
                "{channel}: fx {} rendered, {} declared",
                ours.intrinsics.fx,
                declared.intrinsics.fx
            );
        }
        println!("RAN the_views_task_declares_what_the_scene_renders");
    }

    /// The cameras see the cube. The side camera at the Task IR's reset pose (every joint at 0),
    /// the wrist camera at the program's approach waypoint above the cube -- where a wrist
    /// camera is for -- each at the five pinned cube positions, at least [`FLOOR`] cube pixels.
    /// The wrist view moves with the gripper; the side and overhead cameras do not.
    ///
    /// The overhead camera is the committed one (pinned by
    /// `the_views_scene_is_the_demo_scene_plus_two_cameras`) and is measured, not asserted: at
    /// the reset pose the outstretched arm lies between it and most of the draw.
    #[test]
    fn the_cameras_see_the_cube_and_the_wrist_moves_with_the_gripper() {
        let scene = parse_scene(VIEWS_SCENE);
        let reset = forward(&scene, &[]);
        let mut report = Vec::new();
        for (i, cube) in CUBES.into_iter().enumerate() {
            let at_reset = with_cube(&scene, reset.clone(), cube);
            let at_approach = approach(&scene, cube);
            let mut counts = Vec::new();
            for (pose, world) in [("reset", &at_reset), ("approach", &at_approach)] {
                for name in ["overhead", "wrist", "side"] {
                    let rgb = frame(&scene, name, world);
                    dump(&format!("{pose}-{i}-{name}"), &rgb);
                    counts.push((pose, name, cube_pixels(&rgb)));
                }
            }
            let count = |p: &str, n: &str| {
                counts
                    .iter()
                    .find(|(pose, name, _)| *pose == p && *name == n)
                    .expect("rendered")
                    .2
            };
            report.push(format!("cube {cube:?}: {counts:?}"));
            assert!(
                count("reset", "side") >= FLOOR,
                "side at reset, cube {cube:?}: {counts:?}"
            );
            assert!(
                count("approach", "wrist") >= FLOOR,
                "wrist at the approach, cube {cube:?}: {counts:?}"
            );
            assert!(count("approach", "side") >= FLOOR, "{counts:?}");

            // Two arm poses, one cube: the wrist camera's pose and picture change with the
            // gripper; a world-fixed camera's pose does not.
            let pose_of = |name: &str, world: &BTreeMap<StableId, Pose>| {
                let cfg = EnvRendererCfg::rgb(camera(&scene, name).id, SIDE, SIDE);
                camera_view(&scene, &cfg, world).expect("resolves").pose
            };
            assert_ne!(pose_of("wrist", &at_reset), pose_of("wrist", &at_approach));
            assert_ne!(
                frame(&scene, "wrist", &at_reset),
                frame(&scene, "wrist", &at_approach)
            );
            for fixed in ["overhead", "side"] {
                assert_eq!(pose_of(fixed, &at_reset), pose_of(fixed, &at_approach));
            }
        }
        println!("floor {FLOOR} cube pixels of 96x96; (pose, camera, count):");
        println!("{}", report.join("\n"));
        println!("RAN the_cameras_see_the_cube_and_the_wrist_moves_with_the_gripper");
    }
}
