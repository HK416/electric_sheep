//! Oracles for packet M11/X3 (`docs/packets/M11/X3-rollout-render.md`): `Rollout` renders.
//!
//! * The **vision reach documents** -- `tests/fixtures/rl/{task,observation,learning,
//!   evaluation}-reach-vision.toml` -- are generated here, from the committed reach documents
//!   and the demo's camera, never typed in (`regenerate_vision_reach_documents`, and
//!   `vision_reach_documents_validate` holds the committed files to what it writes).
//! * `rollout_frame_equals_collector_frame` (feature `render`, GPU + `MuJoCo`): two envs, two
//!   episodes each, `Pt` 16 spp under `seed = "tick"`. A `Rollout` frame at `(env, episode,
//!   tick)` is compared bitwise with (a) `es loop collect --frames` run in-process -- the
//!   collector, wired to its renderer exactly as `crates/es/src/cmd/loop.rs` wires it -- for
//!   env 0, and (b) a collector-wired renderer over a twin `Env` stepping the same executed
//!   commands, for both envs. Env 1 resets off env 0's phase, so the seed restarting per env is
//!   on the trace, not assumed. The collector runs one env (its episodes draw
//!   `(seed, env 0, episode)`), which is why env 1 has only the twin.
//! * Without the feature the same test name refuses the image input by name.
//!
//! `MuJoCo` comes through `ES_PYTHON` (spec 1.4's reference backend); with no `MuJoCo`, no
//! `slangc` or no Vulkan device each test prints `SKIP <test>: <reason>` and returns.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use es_ir::graph::{NodeId, Port, PortRef};
use es_ir::learning::{FusionKind, LearningGraph, LearningNode};
use es_ir::observation::{ObservationIr, ObservationOutput};
use es_ir::task::{ObsSource, SensorPath, TaskIr, TaskNode};
use es_ir::types::Shape;

/// `Pt` at 16 samples per pixel: the packet's oracle setting, between the 4 a PPO run can
/// afford and the 64 the demo's `task-pt-tick.toml` renders at.
const VISION_SPP: u32 = 16;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/es-py is two levels under the repo root")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n")
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

// --- the generator --------------------------------------------------------------------------

/// `task-reach.toml` plus the demo's overhead camera: its `GetSensor` and `ObservationSpec`
/// nodes and its `rgb_overhead` channel, copied out of `task-pt-tick.toml` (`Pt`, 3 bounces,
/// exposure 64, `seed = "tick"`) with `spp` set to [`VISION_SPP`]. Nothing else moves: the
/// reward cone, the terminations, the reset draws and the four state channels are the reach
/// task's.
fn vision_task() -> TaskIr {
    let mut task =
        es_ir::serial::task_from_toml(&read("tests/fixtures/rl/task-reach.toml")).expect("reach");
    let demo =
        es_ir::serial::task_from_toml(&read("tests/fixtures/visible-learning/task-pt-tick.toml"))
            .expect("the demo's tick-seeded Pt task");
    let spec_node = demo
        .graph
        .nodes
        .iter()
        .find(|(_, n)| matches!(n, TaskNode::ObservationSpec { channel, .. } if channel == "rgb_overhead"))
        .map(|(id, _)| *id)
        .expect("the demo declares rgb_overhead");
    let edge = demo
        .graph
        .edges
        .iter()
        .find(|e| e.to.node == spec_node)
        .expect("the camera's ObservationSpec is fed")
        .clone();
    let next = task.graph.nodes.keys().map(|n| n.0).max().expect("nodes") + 1;
    task.graph
        .insert(NodeId(next), demo.graph.nodes[&edge.from.node].clone());
    task.graph
        .insert(NodeId(next + 1), demo.graph.nodes[&spec_node].clone());
    task.graph.connect(
        NodeId(next),
        &edge.from.port,
        NodeId(next + 1),
        &edge.to.port,
    );
    let mut channel = demo.observation_spec.channels["rgb_overhead"].clone();
    match &mut channel.source {
        ObsSource::Sensor { render, .. } => match &mut render.path {
            SensorPath::Pt { spp, .. } => *spp = VISION_SPP,
            SensorPath::Rs => panic!("task-pt-tick.toml's camera is Pt"),
        },
        other => panic!("rgb_overhead is {other:?}"),
    }
    task.observation_spec
        .channels
        .insert("rgb_overhead".to_owned(), channel);
    task
}

/// `observation-reach.toml` plus the demo's image branch (`ImageInput -> Dequantize ->
/// Normalize`, `[3, 96, 96]` in `[0, 1]`), copied node for node out of
/// `observation-pt-tick.toml` and renumbered after the reach nodes. `task_ref` follows the
/// vision task.
fn vision_observation(task: &TaskIr) -> ObservationIr {
    let mut obs =
        es_ir::serial::observation_from_toml(&read("tests/fixtures/rl/observation-reach.toml"))
            .expect("reach observation");
    obs.task_ref = task.task_hash().expect("the vision task hashes");
    let demo = es_ir::serial::observation_from_toml(&read(
        "tests/fixtures/visible-learning/observation-pt-tick.toml",
    ))
    .expect("the demo observation");
    let out = demo.outputs["rgb_overhead"].clone();
    // Backwards closure over the edges from the image output: its branch and nothing else.
    let mut keep = vec![out.port.node];
    let mut i = 0;
    while i < keep.len() {
        for e in &demo.graph.edges {
            if e.to.node == keep[i] && !keep.contains(&e.from.node) {
                keep.push(e.from.node);
            }
        }
        i += 1;
    }
    keep.sort();
    let base = obs.graph.nodes.keys().map(|n| n.0).max().expect("nodes") + 1;
    let map: BTreeMap<NodeId, NodeId> = keep
        .iter()
        .enumerate()
        .map(|(k, id)| (*id, NodeId(base + k as u32)))
        .collect();
    for (old, new) in &map {
        obs.graph.insert(*new, demo.graph.nodes[old].clone());
    }
    for e in &demo.graph.edges {
        if let (Some(f), Some(t)) = (map.get(&e.from.node), map.get(&e.to.node)) {
            obs.graph.connect(*f, &e.from.port, *t, &e.to.port);
        }
    }
    let port = PortRef::new(map[&out.port.node], out.port.port.clone());
    obs.graph.outputs.push(port.clone());
    obs.outputs.insert(
        "rgb_overhead".to_owned(),
        ObservationOutput { port, ty: out.ty },
    );
    obs
}

/// `learning-reach.toml` with a camera branch: the demo's `VisionEncoder { ResNet18 }`
/// (`learning.toml`'s, from scratch, 512 wide) beside the reach `StateEncoder` (64 wide),
/// `Fusion { Concat }` of the two at 576 -- the plain concatenation, so the lowering adds no
/// projection -- feeding the reach head, chunker and unnormalizer unchanged.
fn vision_learning() -> LearningGraph {
    let mut g = es_ir::serial::learning_from_toml(&read("tests/fixtures/rl/learning-reach.toml"))
        .expect("reach learning");
    let demo =
        es_ir::serial::learning_from_toml(&read("tests/fixtures/visible-learning/learning.toml"))
            .expect("the demo learning");
    let image = demo
        .inputs
        .iter()
        .find(|p| p.name == "rgb_overhead")
        .expect("the demo reads rgb_overhead")
        .clone();
    let encoder = demo
        .nodes
        .nodes
        .values()
        .find(|n| matches!(n, LearningNode::VisionEncoder { .. }))
        .expect("the demo has a VisionEncoder")
        .clone();
    let LearningNode::VisionEncoder {
        out_dim: vision_dim,
        ..
    } = encoder
    else {
        unreachable!()
    };
    let find = |want: fn(&LearningNode) -> bool| {
        g.nodes
            .nodes
            .iter()
            .find(|(_, n)| want(n))
            .map(|(id, _)| *id)
            .expect("the reach graph has the node")
    };
    let state = find(|n| matches!(n, LearningNode::StateEncoder { .. }));
    let head = find(|n| matches!(n, LearningNode::PolicyHead { .. }));
    let LearningNode::StateEncoder {
        out_dim: state_dim, ..
    } = g.nodes.nodes[&state]
    else {
        unreachable!()
    };
    let (head_port, feat_ty) = match &g.nodes.nodes[&head] {
        LearningNode::PolicyHead { inputs, .. } => (inputs[0].name.clone(), inputs[0].ty.clone()),
        _ => unreachable!(),
    };
    let feat = |dim: u32| {
        let mut ty = feat_ty.clone();
        ty.shape = Shape::new([u64::from(dim)]);
        ty
    };
    let fused = vision_dim + state_dim;
    let next = g.nodes.nodes.keys().map(|n| n.0).max().expect("nodes") + 1;
    let (vision, fusion) = (NodeId(next), NodeId(next + 1));
    g.nodes.insert(vision, encoder);
    g.nodes.insert(
        fusion,
        LearningNode::Fusion {
            inputs: vec![
                Port::new("image", feat(vision_dim)),
                Port::new("state", feat(state_dim)),
            ],
            kind: FusionKind::Concat,
            out_dim: fused,
            token_count: 0,
        },
    );
    if let Some(LearningNode::PolicyHead { inputs, .. }) = g.nodes.nodes.get_mut(&head) {
        inputs[0].ty = feat(fused);
    }
    g.nodes
        .edges
        .retain(|e| !(e.from.node == state && e.to.node == head));
    g.nodes.connect(vision, "out", fusion, "image");
    g.nodes.connect(state, "out", fusion, "state");
    g.nodes.connect(fusion, "out", head, &head_port);
    // `inputs` and `nodes.inputs` are positional twins (spec 8, Appendix B.3).
    g.inputs.push(image.clone());
    g.nodes
        .inputs
        .push(PortRef::new(vision, image.name.clone()));
    g.policy.contract.inputs.insert(image.name.clone(), image);
    g
}

const TASK_HEADER: &str = "\
# Task IR (spec 6) for the SO-101 reach task seen through a camera -- packet M11/X3.
#
# Generated by `ES_GENERATE_GOLDENS=1 cargo test -p es-py --test vision_reach -- --ignored
# regenerate_vision_reach_documents` from tests/fixtures/rl/task-reach.toml and
# tests/fixtures/visible-learning/task-pt-tick.toml; no hash in it is typed in by hand.
#
# task-reach.toml plus ONE CHANNEL: the demo's `rgb_overhead` -- its `GetSensor` and
# `ObservationSpec` nodes and its channel, copied -- rendered on the `Pt` path at 16 samples
# per pixel, 3 bounces, exposure 64, `seed = \"tick\"` (docs/design/renderer.md 12.8). The
# reward cone, the two terminations, the reset draws and the four state channels are
# task-reach.toml's, value for value; its header argues them.
#
# 16 spp is the packet's oracle setting (docs/packets/M11/X3-rollout-render.md), not a tuned
# one: the render cost table in docs/design/rl-continuation.md section 2c is what a vision PPO
# run picks its setting from (spec 28.14 wave 3, X7).
";

const OBSERVATION_HEADER: &str = "\
# Observation IR (spec 7) for the camera-bearing SO-101 reach task -- packet M11/X3.
#
# Generated by `regenerate_vision_reach_documents` (crates/es-py/tests/vision_reach.rs);
# `task_ref` is task-reach-vision.toml's own `task_hash`.
#
# observation-reach.toml's 26-wide `state` port, node for node, plus the demo's image branch
# copied out of tests/fixtures/visible-learning/observation-pt-tick.toml: `ImageInput` (u8,
# 96x96x3, the declared ImageSpec) -> `Dequantize` -> `Normalize` into `rgb_overhead`,
# `[3, 96, 96]` f32 in [0, 1]. No Resize and no Crop, so nothing owes an intrinsics
# transform (spec 7.2, INV-14).
";

const LEARNING_HEADER: &str = "\
# Learning IR (spec 8) for the camera-bearing SO-101 reach task under PPO -- packet M11/X3.
#
# Generated by `regenerate_vision_reach_documents` (crates/es-py/tests/vision_reach.rs) from
# tests/fixtures/rl/learning-reach.toml and tests/fixtures/visible-learning/learning.toml.
#
#   VisionEncoder{ResNet18, from scratch, 512} -.
#                                                Fusion{Concat, 576} -> PolicyHead{Regression,
#   StateEncoder{Mlp [64, 64], 64}            -'      tanh, 1 x 6} -> ActionChunker
#                                                                   -> Normalizer{Inverse}
#
# The encoder is the demo's own node; the state branch, head, chunker and unnormalizer are
# learning-reach.toml's, so the action is in actuator units exactly as there. The fusion is the
# plain concatenation (576 = 512 + 64), so the lowering adds no projection.
";

const EVALUATION_HEADER: &str = "\
# Evaluation IR (spec 10) for the camera-bearing SO-101 reach task -- packet M11/X3.
#
# evaluation-reach.toml with `task` and `observation` following the vision documents' hashes
# and nothing else: the same 16 held-out seeds 201-216, the same suites, the same
# `success_rate >= 0.8` on `nominal`. Generated by `regenerate_vision_reach_documents`.
";

/// `(file name, header, body)` for every vision document -- what the generator writes and what
/// `vision_reach_documents_validate` holds the committed files to. The deployment is
/// `deployment-reach.toml`, unchanged: a camera says nothing about the envelope.
fn vision_document_set() -> Vec<(&'static str, &'static str, String)> {
    let task = vision_task();
    let observation = vision_observation(&task);
    let mut evaluation =
        es_ir::serial::evaluation_from_toml(&read("tests/fixtures/rl/evaluation-reach.toml"))
            .expect("reach evaluation");
    evaluation.task = hex(&task.task_hash().expect("hash"));
    evaluation.observation = hex(&observation.observation_hash().expect("hash"));
    let toml = |r: Result<String, _>| r.expect("the document serializes");
    vec![
        (
            "task-reach-vision.toml",
            TASK_HEADER,
            toml(es_ir::serial::task_to_toml(&task)),
        ),
        (
            "observation-reach-vision.toml",
            OBSERVATION_HEADER,
            toml(es_ir::serial::observation_to_toml(&observation)),
        ),
        (
            "learning-reach-vision.toml",
            LEARNING_HEADER,
            toml(es_ir::serial::learning_to_toml(&vision_learning())),
        ),
        (
            "evaluation-reach-vision.toml",
            EVALUATION_HEADER,
            toml(es_ir::serial::evaluation_to_toml(&evaluation)),
        ),
    ]
}

/// Writes the vision reach documents. Run explicitly:
///
///     ES_GENERATE_GOLDENS=1 cargo test -p es-py --test vision_reach -- --ignored \
///         regenerate_vision_reach_documents
#[test]
#[ignore = "fixture generator; run explicitly"]
fn regenerate_vision_reach_documents() {
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP regenerate_vision_reach_documents: set ES_GENERATE_GOLDENS=1");
        return;
    }
    for (name, header, body) in vision_document_set() {
        let path = repo_root().join("tests/fixtures/rl").join(name);
        std::fs::write(&path, format!("{header}\n{body}")).expect("write");
        println!("wrote {}", path.display());
    }
}

/// The committed vision documents parse, validate, agree across every Cross-IR boundary with
/// `deployment-reach.toml`, and are exactly what the generator writes.
#[test]
fn vision_reach_documents_validate() {
    let rl = |name: &str| read(&format!("tests/fixtures/rl/{name}"));
    let task = es_ir::serial::task_from_toml(&rl("task-reach-vision.toml")).expect("task");
    let observation = es_ir::serial::observation_from_toml(&rl("observation-reach-vision.toml"))
        .expect("observation");
    let learning =
        es_ir::serial::learning_from_toml(&rl("learning-reach-vision.toml")).expect("learning");
    let evaluation = es_ir::serial::evaluation_from_toml(&rl("evaluation-reach-vision.toml"))
        .expect("evaluation");
    let deployment =
        es_ir::serial::deployment_from_toml(&rl("deployment-reach.toml")).expect("deployment");
    for (name, diags) in [
        ("task", task.validate()),
        ("observation", observation.validate()),
        ("learning", learning.validate()),
        ("evaluation", evaluation.validate()),
    ] {
        assert!(diags.is_empty(), "{name}-reach-vision.toml: {diags:#?}");
    }
    let diags = es_ir::cross::check(&es_ir::cross::IrBundle {
        task: &task,
        observation: &observation,
        learning: &learning,
        deployment: &deployment,
        evaluation: Some(&evaluation),
    });
    assert!(diags.is_empty(), "cross-IR: {diags:#?}");
    assert_eq!(
        observation.outputs["rgb_overhead"].ty.shape.dims(),
        [3, 96, 96]
    );

    for (name, _, body) in vision_document_set() {
        let on_disk = rl(name);
        let committed = on_disk
            .split_once("\nes_schema")
            .map(|(_, rest)| format!("es_schema{rest}"))
            .unwrap_or(on_disk);
        assert_eq!(
            committed, body,
            "{name} is not what the generator writes; rerun `ES_GENERATE_GOLDENS=1 cargo test \
             -p es-py --test vision_reach -- --ignored regenerate_vision_reach_documents`"
        );
    }
}

// --- the rollout --------------------------------------------------------------------------

/// The reach deployment's joint count and horizon.
const NJ: usize = 6;
const H: usize = 1;
const SEED: u64 = 3;

/// `(task, observation, deployment, scene)` as text, in `Rollout::new`'s order.
fn rollout_documents() -> (String, String, String, String) {
    (
        read("tests/fixtures/rl/task-reach-vision.toml"),
        read("tests/fixtures/rl/observation-reach-vision.toml"),
        read("tests/fixtures/rl/deployment-reach.toml"),
        read("tests/fixtures/mjcf/so101_pick_place.xml"),
    )
}

fn mujoco(test: &str) -> bool {
    match es_physics_backend::MuJoCoCpuBackend::is_available() {
        Ok(()) => true,
        Err(why) => {
            println!("SKIP {test}: {why}");
            false
        }
    }
}

/// Without the `render` feature es-py links no renderer, and an image input is refused by
/// name at `observe` -- today's behaviour, never a zero-filled frame.
#[cfg(not(feature = "render"))]
#[test]
fn rollout_frame_equals_collector_frame_needs_the_render_feature() {
    const TEST: &str = "rollout_frame_equals_collector_frame_needs_the_render_feature";
    if !mujoco(TEST) {
        return;
    }
    let (task, obs, deploy, scene) = rollout_documents();
    let mut roll = es_native::rollout::Rollout::<NJ, H>::new(&task, &obs, &deploy, &scene, SEED, 2)
        .expect("the vision documents build a rollout");
    let err = roll
        .observe()
        .expect_err("no renderer in this build")
        .to_string();
    assert!(
        err.contains("is an image") && err.contains("renderer"),
        "the refusal names the image input: {err}"
    );
    assert_eq!(roll.render_ms_per_frame(), None, "nothing was rendered");
    println!("RAN {TEST}: {err}");
}

#[cfg(feature = "render")]
mod render {
    use super::*;

    use std::cell::Cell;

    use es_compile::{BundleKind, BundleManifest, PolicyBundle};
    use es_core::TickRate;
    use es_data::collect::CollectEvent;
    use es_data::{CollectSpec, Collector, Intervention};
    use es_env::scheduler::BatchDomains;
    use es_env::{Env, EnvRenderer};
    use es_gpu::{Gpu, GpuOptions, SlangCompiler};
    use es_native::rollout::Rollout;
    use es_physics_backend::MuJoCoCpuBackend;
    use es_physics_core::backend::{ModelInfo, PhysicsBackend};
    use es_policy::{PolicyError, PolicyInfo, PolicyRuntime, WeightsSource};

    /// Env 0's episodes, in control ticks: the collector's `max_steps`.
    const L: u32 = 4;
    /// Env 1 is reset after this many ticks of its first episode, so its seed clock runs off
    /// env 0's phase.
    const ENV1_RESET_AT: u32 = 3;
    const STEPS: u32 = 2 * L;

    fn gpu(test: &str) -> Option<Gpu> {
        if let Err(e) = SlangCompiler::new() {
            println!("SKIP {test}: no slangc ({e})");
            return None;
        }
        match Gpu::open(GpuOptions::default()) {
            Ok(g) => {
                println!("{test} on {}", g.capabilities().device_name);
                Some(g)
            }
            Err(e) => {
                println!("SKIP {test}: no Vulkan device ({e})");
                None
            }
        }
    }

    /// The command a tick sends: the episode's reset pose plus a slow ramp, rounded to `f32`
    /// because the collector's action tensor is `f32` (`Intervened::infer`) and both runs must
    /// hand the plane the same number. 0.004 rad per 20 ms tick is 0.2 rad/s, inside the
    /// envelope's rate and acceleration bounds, so the plane lets it through on both paths.
    fn command(q0: &[f64], tick: u32) -> [f64; NJ] {
        std::array::from_fn(|j| f64::from((q0[j] + 0.004 * f64::from(tick)) as f32))
    }

    #[derive(Debug, Default)]
    struct NoPolicy;

    impl PolicyRuntime for NoPolicy {
        fn load(
            &mut self,
            _: &LearningGraph,
            _: &WeightsSource,
        ) -> Result<PolicyInfo, PolicyError> {
            Err(PolicyError::Backend("the oracle drives every tick".into()))
        }
        fn infer(
            &mut self,
            _: &BTreeMap<String, es_compile::Tensor>,
        ) -> Result<BTreeMap<String, es_compile::Tensor>, PolicyError> {
            Err(PolicyError::Backend("the oracle drives every tick".into()))
        }
        fn info(&self) -> Option<&PolicyInfo> {
            None
        }
        fn runtime_hash(&self) -> [u8; 32] {
            [0; 32]
        }
    }

    /// Packet M11/X3 oracle 1.
    #[test]
    fn rollout_frame_equals_collector_frame() {
        const TEST: &str = "rollout_frame_equals_collector_frame";
        if !mujoco(TEST) {
            return;
        }
        let Some(gpu) = gpu(TEST) else { return };
        let (task_toml, obs_toml, deploy_toml, scene_xml) = rollout_documents();
        let task = es_ir::serial::task_from_toml(&task_toml).expect("task");
        let deploy = es_ir::serial::deployment_from_toml(&deploy_toml).expect("deployment");
        let scene = es_assets::parse_mjcf(&scene_xml).expect("scene").scene;

        // --- the rollout, two envs ------------------------------------------------------
        let mut roll =
            Rollout::<NJ, H>::new(&task_toml, &obs_toml, &deploy_toml, &scene_xml, SEED, 2)
                .expect("the vision documents build a rendering rollout");
        // The twin: the same documents and seed, stepped with the commands the rollout's
        // plane executed. `rollout_matches_es_eval_loop` is why this is the same env.
        let mut domains = BatchDomains::single_env_at(
            TickRate::from_period_secs(scene.options.timestep).expect("rate"),
            deploy.rate.control,
        )
        .expect("domains");
        for d in [
            &mut domains.simulation,
            &mut domains.observation,
            &mut domains.inference,
        ] {
            d.batch = 2;
        }
        let mut twin: Env<MuJoCoCpuBackend> =
            Env::new(&task, &scene, MuJoCoCpuBackend::new(), &domains, SEED).expect("twin");
        let (_, _, _, cfg) = image_channel(&task);
        let mut twin_cams: Vec<EnvRenderer<'_>> = (0..2)
            .map(|_| EnvRenderer::new(&gpu, &scene, cfg.clone()).expect("renderer"))
            .collect();
        let mut twin_new = [true, true];

        let mut frames: Vec<[Vec<u8>; 2]> = Vec::new();
        let mut q0: [Vec<f64>; 2] = [roll.qpos(0), roll.qpos(1)];
        let mut env0_q0 = vec![q0[0].clone()];
        let mut tick = [0u32; 2];
        for step in 0..STEPS {
            if step > 0 && step % L == 0 {
                roll.reset(Some(&[0])).expect("reset env 0");
                twin.reset(Some(&[0])).expect("twin reset");
                (twin_new[0], tick[0], q0[0]) = (true, 0, roll.qpos(0));
                env0_q0.push(q0[0].clone());
            }
            if step == ENV1_RESET_AT {
                roll.reset(Some(&[1])).expect("reset env 1");
                twin.reset(Some(&[1])).expect("twin reset");
                (twin_new[1], tick[1], q0[1]) = (true, 0, roll.qpos(1));
            }
            roll.observe().expect("observe renders");
            let got = [0, 1].map(|i| roll.frame(i).expect("an image was captured").to_vec());
            {
                let model = twin.model();
                let state = twin.backend().state();
                for i in 0..2 {
                    if std::mem::take(&mut twin_new[i]) {
                        twin_cams[i].begin_episode();
                    }
                    let want = twin_cams[i]
                        .frame(model, &state, i as u32)
                        .expect("twin frame")
                        .to_bytes();
                    assert_eq!(
                        roll.qpos(i),
                        state.qpos_of(i as u32),
                        "step {step} env {i}: the twin left the rollout's state"
                    );
                    assert!(
                        got[i] == want,
                        "step {step} env {i} (tick {}): the rollout's frame differs from the \
                         collector-wired renderer's in {} of {} bytes",
                        tick[i],
                        got[i].iter().zip(&want).filter(|(a, b)| a != b).count(),
                        want.len()
                    );
                }
            }
            frames.push(got);
            let actions: Vec<f64> = (0..2).flat_map(|i| command(&q0[i], tick[i])).collect();
            let act = roll.act(&actions).expect("act");
            assert_eq!(
                act.executed, actions,
                "step {step}: the plane clamped the ramp"
            );
            assert_eq!(
                act.dones,
                [false, false],
                "no episode ends inside the oracle"
            );
            twin.step(&act.executed).expect("twin step");
            tick = tick.map(|t| t + 1);
        }

        // Grain moves with the tick and not only with the pose: consecutive frames differ.
        assert_ne!(
            frames[0][0], frames[1][0],
            "tick 0 and tick 1 rendered alike"
        );

        // --- the collector, env 0 ------------------------------------------------------
        let dir = std::env::temp_dir().join(format!("es-py-x3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let frames_dir = dir.join("frames");
        let mut learning = es_ir::serial::learning_from_toml(&read(
            "tests/fixtures/rl/learning-reach-vision.toml",
        ))
        .expect("learning");
        // Horizon 1, synchronous, like `Rollout`: the chunk the intervener hands in is the
        // command of the same tick (`rl-continuation.md` section 3).
        learning.policy.contract.runtime.expected_latency_ms = 0.0;
        let bundle = PolicyBundle {
            manifest: BundleManifest::new(BundleKind::Policy, es_compile::BundleHashes::default()),
            task: task.clone(),
            observation: es_ir::serial::observation_from_toml(&obs_toml).expect("obs"),
            learning,
            deployment: deploy.clone(),
            weights: Vec::new(),
            evaluation: None,
        };
        let (_, _, _, cfg) = image_channel(&task);
        let mut renderer = EnvRenderer::new(
            &gpu,
            &scene,
            es_env::EnvRendererCfg {
                frames_dir: Some(frames_dir.clone()),
                ..cfg
            },
        )
        .expect("collector renderer");
        // `crates/es/src/cmd/loop.rs::collect_typed`'s wiring, line for line: the episode
        // flag set on `EpisodeBegin`, `begin_episode` on the first frame after it.
        let new_episode = Cell::new(true);
        let mut publish = |event: CollectEvent| {
            if matches!(event, CollectEvent::EpisodeBegin { .. }) {
                new_episode.set(true);
            }
        };
        let mut frame_sink =
            |model: &ModelInfo, state: &es_physics_core::backend::StateView<'_>| {
                if new_episode.replace(false) {
                    renderer.begin_episode();
                }
                renderer.frame(model, state, 0).map_err(|e| e.to_string())?;
                Ok(())
            };
        let mut hook = |episode: u32, frame: u32, _: &ModelInfo, _: &[f64]| {
            Intervention::Action(command(&env0_q0[episode as usize], frame))
        };
        let mut policy = NoPolicy;
        Collector::run_with_sink::<MuJoCoCpuBackend, _, NJ, H>(
            &CollectSpec {
                bundle: &bundle,
                scene: &scene,
                n_episodes: 2,
                seed: SEED,
                max_steps: L,
                out_root: &dir.join("dataset"),
                traj_dir: None,
            },
            &mut policy,
            MuJoCoCpuBackend::new,
            &mut hook,
            Some(&mut frame_sink),
            Some(&mut publish),
        )
        .expect("the collector runs");

        for (step, pair) in frames.iter().enumerate() {
            let path = frames_dir.join(format!("{step:06}.bin"));
            let theirs = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let (episode, t) = (step as u32 / L, step as u32 % L);
            assert!(
                pair[0] == theirs,
                "episode {episode} tick {t}: the rollout's env-0 frame differs from \
                 `es loop collect --frames` in {} of {} bytes",
                pair[0].iter().zip(&theirs).filter(|(a, b)| a != b).count(),
                theirs.len()
            );
        }
        let ms = roll.render_ms_per_frame().expect("render cost measured");
        let m = roll.metrics();
        assert!(m.camera_frames_per_sec.is_some() && m.pixels_per_sec.is_some());
        let _ = std::fs::remove_dir_all(&dir);
        println!(
            "RAN {TEST}: {STEPS} ticks x 2 envs bitwise vs the collector-wired twin; env 0's \
             2 episodes x {L} ticks bitwise vs the collector; {} bytes/frame; render \
             {ms:.2} ms/frame",
            frames[0][0].len()
        );
    }

    /// The render cost table of `docs/design/rl-continuation.md` section 2c: `Rollout`'s own
    /// `render_ms_per_frame` on the vision reach task, one env, `Rs` and `Pt` at 4 / 16 / 64
    /// spp with SVGF off and on. Two warm-up frames, then [`COST_FRAMES`] frames while the arm
    /// moves, the warm-up subtracted out of the running mean. Run explicitly, in release:
    ///
    ///     cargo test -p es-py --release --features render --test vision_reach -- \
    ///         --ignored rollout_render_cost --nocapture
    #[test]
    #[ignore = "measurement; run explicitly"]
    fn rollout_render_cost() {
        const TEST: &str = "rollout_render_cost";
        const WARM: u32 = 2;
        const COST_FRAMES: u32 = 16;
        if !mujoco(TEST) {
            return;
        }
        let Some(gpu) = gpu(TEST) else { return };
        drop(gpu);
        let (task_toml, obs_toml, deploy_toml, scene_xml) = rollout_documents();
        let rows: Vec<(&str, SensorPath, bool)> = std::iter::once(("Rs", SensorPath::Rs, false))
            .chain([4, 16, 64].into_iter().flat_map(|spp| {
                [false, true].map(|svgf| ("Pt", SensorPath::Pt { spp, bounces: 3 }, svgf))
            }))
            .collect();
        println!("| path | spp | SVGF | ms/frame |");
        println!("|---|---|---|---|");
        for (label, path, svgf) in rows {
            let mut task = es_ir::serial::task_from_toml(&task_toml).expect("task");
            let ch = task
                .observation_spec
                .channels
                .get_mut("rgb_overhead")
                .expect("camera");
            let ObsSource::Sensor { render, .. } = &mut ch.source else {
                unreachable!()
            };
            render.path = path;
            render.svgf = svgf;
            if path == SensorPath::Rs {
                render.exposure = 1.0;
                render.seed = es_ir::task::SeedStream::Fixed;
            }
            let task_text = es_ir::serial::task_to_toml(&task).expect("serializes");
            let mut roll =
                Rollout::<NJ, H>::new(&task_text, &obs_toml, &deploy_toml, &scene_xml, SEED, 1)
                    .expect("rollout");
            let q0 = roll.qpos(0);
            let mut warm = 0.0;
            for t in 0..WARM + COST_FRAMES {
                roll.observe().expect("observe");
                if t + 1 == WARM {
                    warm = roll.render_ms_per_frame().expect("rendered") * f64::from(WARM);
                }
                roll.act(&command(&q0, t)).expect("act");
            }
            let total =
                roll.render_ms_per_frame().expect("rendered") * f64::from(WARM + COST_FRAMES);
            let ms = (total - warm) / f64::from(COST_FRAMES);
            let spp = match path {
                SensorPath::Pt { spp, .. } => spp.to_string(),
                SensorPath::Rs => "-".to_owned(),
            };
            println!(
                "| {label} | {spp} | {} | {ms:.2} |",
                if svgf { "on" } else { "off" }
            );
        }
    }

    /// The one image channel, as `es_tools::image_channel` reads it, and the renderer config
    /// `es_env::render::sensor_cfg` makes of it.
    fn image_channel(
        task: &TaskIr,
    ) -> (
        String,
        es_core::StableId,
        es_ir::image::ImageSpec,
        es_env::EnvRendererCfg,
    ) {
        let (name, ch) = task
            .observation_spec
            .channels
            .iter()
            .find(|(_, c)| c.ty.image.is_some())
            .expect("an image channel");
        let es_ir::types::Frame::Camera(camera) = ch.ty.frame else {
            panic!("{name} is not in a camera frame")
        };
        let ObsSource::Sensor { render, .. } = ch.source else {
            panic!("{name} is not a sensor")
        };
        let spec = ch.ty.image.expect("image");
        (
            name.clone(),
            camera,
            spec,
            es_env::render::sensor_cfg(camera, &spec, &render, None),
        )
    }
}
