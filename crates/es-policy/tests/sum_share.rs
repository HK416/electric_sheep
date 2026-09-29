//! Packet M15/N6 oracles: `VisionEncoder.share` and `FusionKind::Sum` lowered to `PyTorch`
//! (spec 8.3, `docs/design/multi-camera.md` sections 3.3 and 6).
//!
//! Python-free, in the PR tier:
//!
//! - every committed Learning IR lowers to the byte-identical source and contract it did on
//!   `main` before the packet (`lowering_hash` is the `compiler` slot of `execution_hash`);
//! - a three-view graph whose encoders share one weight group lowers to **one** backbone
//!   member, applied to each view, and a `Sum` in declared input order;
//! - a checkpoint that still carries a sharer's own tensors is refused, naming the sharer.
//!
//! Python-gated (`ES_PYTHON` with `torch` and `torchvision`; a loud SKIP otherwise): the
//! lowered module equals `python/sum_share_ref.py` -- one hand-written `ResNet18` applied to
//! each camera and the features added with `+` -- on fixed inputs, **bitwise** on CPU; the
//! weight file it writes holds one copy of the encoder and loads through `TorchRuntime`
//! (`validate_keys` included); `train_act.py`'s `init_backbone` finds exactly one backbone,
//! the owner's, to initialise; and its single-view loss from one forward over a list of view
//! subsets is N7's `1 + V` separate forwards bitwise, with the encoder run `V` times instead
//! of `V * (1 + V)` (packet M15/N7b).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use es_compile::Tensor;
use es_ir::graph::{Graph, NodeId, Port, PortRef};
use es_ir::learning::{
    action_unit, ActionExecutionMode, Activation, ArchKind, ChunkBlendPolicy, FusionKind, HeadKind,
    LearningGraph, LearningNode, PolicyContract, PolicyHandle, RuntimeHints, Squash,
    StateEncoderKind, VisionBackbone, WeightsRef,
};
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_policy::lower::Contract;
use es_policy::weights::{parse_header, validate_keys, write_safetensors, Checkpoint};
use es_policy::{
    compare_actions, lower_to_torch, PolicyError, PolicyRuntime, Tolerance, TorchRuntime,
    WeightsSource,
};

// --- the committed documents ----------------------------------------------------------------

/// `(document, lowering_hash, blake3 of the contract's JSON)`, computed on `main` (d18e08f)
/// before this packet touched the lowering, and typed in here on purpose: the one place the
/// numbers are asserted rather than derived.
const COMMITTED: [(&str, &str, &str); 8] = [
    (
        "quadruped/learning.toml",
        "752dde6f6429798a7b49e9c34bc72f73fca3b0552bd0aa7ab37a63e7a568767e",
        "cd7ffda318c414838bf78d07d9f4be43fc2fb94bb7f817d24a2a2a2cf74422c0",
    ),
    (
        "rl/learning-reach-delta.toml",
        "12a044ac19a6efd7996648e318a91e5f18daca992f25df5b17b1e348c5250541",
        "f8700364ece758cfc5e5e6db9100b333a21ac451fa0ee57ac59bbaecb5487a6e",
    ),
    (
        "rl/learning-reach-vision-pix.toml",
        "ae34a21b45ab9d74ddef104a1eb82fbe865355b189b9082f8716feeada0c550a",
        "39855007b70b67ac25c657392917ee2911e68e9674d1ac64c1a2cda6beebba38",
    ),
    (
        "rl/learning-reach-vision.toml",
        "7a6522eca6b84321bed77f150d24f264a0d92ce2f13ab977c32c98c783f559aa",
        "198fc0cf33f0cbf23cfc0039d29453be9d2e72b6810c9825943af8b92cb9cd90",
    ),
    (
        "rl/learning-reach.toml",
        "dce8d35268fe74378d55d4f8a065e53d95c81f7559f8693b8f3e5f7abc902a0e",
        "29ac91fd0f32f87807f21edea980ec61c165bcf4b530261be78593e770921705",
    ),
    (
        "rl/learning-state.toml",
        "7a90048ff4576f3f8dcfd7d78af3e1fcdaaeec6ee1c2b5d6a7c239bbbff1a375",
        "b3880c8a2efd61ad0a24d545ac78ae4bb9b70c1c1747de2a04e01b2b696390fa",
    ),
    (
        "visible-learning/learning-pretrained.toml",
        "41d11a06615b22b0fa2d41c31481c191a7721529806a44a48edaeaf1b06a6841",
        "525a71f2eac45cfa2f296e6fb1df925d5e41e179da667ce457bff53e861d9af1",
    ),
    (
        "visible-learning/learning.toml",
        "3d06811c52b6887f021440d4ce9a1a061e4eb27a0abb97c79822acd4d8a2d394",
        "c4c1cb0d45a9d8ff8d3b9fcb5d67e2b61ba7211bd9b7a7fbffa87f0cbcfb94db",
    ),
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[test]
fn committed_learning_irs_lower_as_on_main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for (name, lowering, contract) in COMMITTED {
        let path = root.join(name);
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let g = es_ir::serial::learning_from_toml(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        let m = lower_to_torch(&g).unwrap_or_else(|e| panic!("{name}: {e}"));
        let json = serde_json::to_string(&Contract::new(&m, &g)).expect("serialises");
        assert_eq!(
            (
                hex(&m.lowering_hash),
                hex(blake3::hash(json.as_bytes()).as_bytes())
            ),
            (lowering.to_owned(), contract.to_owned()),
            "{name}: the lowering or its contract moved"
        );
        assert!(m.sharers.is_empty(), "{name}");
    }
    println!(
        "RAN committed_learning_irs_lower_as_on_main: {} documents",
        COMMITTED.len()
    );
}

// --- the three-view graph (design note section 3.3) ------------------------------------------

/// Feature width, and the state MLP's hidden width.
const W: u32 = 64;
const STATE: u64 = 6;
const HORIZON: u32 = 4;
const ACTION_DIM: u32 = 6;
const EXECUTE: u32 = 2;
/// Small enough that three `ResNet18` applications stay quick on a CPU.
const IMAGE: [u64; 3] = [3, 32, 32];
const VIEWS: [&str; 3] = ["rgb_wrist", "rgb_overhead", "rgb_side"];
/// The owner sits between its two sharers, so neither "owner first" nor "owner last" is what
/// the lowering relies on.
const OWNER: u32 = 1;
const STATE_NODE: u32 = 3;
const HEAD: u32 = 6;

fn ty(shape: &[u64], unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new(shape.to_vec()),
        unit,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    }
}

fn feature(name: &str, dim: u32) -> Port {
    Port::new(name, ty(&[u64::from(dim)], Unit::Dimensionless))
}

fn image(name: &str) -> Port {
    Port::new(name, ty(&IMAGE, Unit::Normalized { lo: 0.0, hi: 1.0 }))
}

/// Wrist (0), overhead (1, the owner) and side (2) through one `ResNet18`, summed (4),
/// concatenated with the joint state's MLP (3 -> 5), a regression head (6) and a chunker (7).
fn three_views(pretrained: bool) -> LearningGraph {
    let state = Port::new("joint_state", ty(&[STATE], action_unit()));
    let chunk = |name: &str, h: u32| {
        Port::new(
            name,
            ty(&[u64::from(h), u64::from(ACTION_DIM)], action_unit()),
        )
    };
    let encoder = |view: &str, share: Option<u32>| LearningNode::VisionEncoder {
        inputs: vec![image(view)],
        backbone: VisionBackbone::ResNet18,
        pretrained,
        frozen: false,
        out_dim: W,
        token_count: 0,
        share: share.map(NodeId),
    };
    let mut g = Graph::new(1);
    g.insert(NodeId(0), encoder(VIEWS[0], Some(OWNER)));
    g.insert(NodeId(OWNER), encoder(VIEWS[1], None));
    g.insert(NodeId(2), encoder(VIEWS[2], Some(OWNER)));
    g.insert(
        NodeId(STATE_NODE),
        LearningNode::StateEncoder {
            inputs: vec![state.clone()],
            kind: StateEncoderKind::Mlp {
                hidden: vec![W],
                activation: Activation::Relu,
                activate_output: false,
            },
            out_dim: W,
        },
    );
    g.insert(
        NodeId(4),
        LearningNode::Fusion {
            inputs: vec![
                feature("wrist", W),
                feature("overhead", W),
                feature("side", W),
            ],
            kind: FusionKind::Sum,
            out_dim: W,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(5),
        LearningNode::Fusion {
            inputs: vec![feature("views", W), feature("state", W)],
            kind: FusionKind::Concat,
            out_dim: 2 * W,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(HEAD),
        LearningNode::PolicyHead {
            inputs: vec![feature("feat", 2 * W)],
            kind: HeadKind::Regression,
            action_dim: ACTION_DIM,
            horizon: HORIZON,
            squash: Squash::None,
        },
    );
    g.insert(
        NodeId(7),
        LearningNode::ActionChunker {
            inputs: vec![chunk("chunk", HORIZON)],
            horizon: HORIZON,
            execute_chunk: EXECUTE,
            replan_hz: 5.0,
            mode: ActionExecutionMode::RecedingHorizon,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 2,
        },
    );
    for (from, to, port) in [
        (0, 4, "wrist"),
        (OWNER, 4, "overhead"),
        (2, 4, "side"),
        (4, 5, "views"),
        (STATE_NODE, 5, "state"),
        (5, HEAD, "feat"),
    ] {
        g.connect(NodeId(from), "out", NodeId(to), port);
    }
    g.connect(NodeId(HEAD), "chunk", NodeId(7), "chunk");
    let inputs: Vec<Port> = VIEWS.iter().map(|v| image(v)).chain([state]).collect();
    g.inputs = vec![
        PortRef::new(NodeId(0), VIEWS[0]),
        PortRef::new(NodeId(OWNER), VIEWS[1]),
        PortRef::new(NodeId(2), VIEWS[2]),
        PortRef::new(NodeId(STATE_NODE), "joint_state"),
    ];
    g.outputs = vec![PortRef::new(NodeId(7), "actions")];
    LearningGraph {
        schema_version: 1,
        outputs: vec![chunk("actions", EXECUTE)],
        nodes: g,
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0; 32],
            },
            contract: PolicyContract {
                inputs: inputs.iter().map(|p| (p.name.clone(), p.clone())).collect(),
                observation_window: 1,
                action_dim: ACTION_DIM,
                horizon: HORIZON,
                execute_chunk: EXECUTE,
                replanning_hz: 5.0,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 5.0,
                    deadline_ms: 50.0,
                },
            },
        },
        inputs,
    }
}

#[test]
fn a_share_group_lowers_to_one_module_applied_per_view() {
    let g = three_views(false);
    let m = lower_to_torch(&g).expect("the three-view graph lowers");

    // One backbone member, the owner's; each view's forward line applies it.
    assert_eq!(
        m.source.matches("_backbone(\"resnet18\"").count(),
        1,
        "{}",
        m.source
    );
    assert!(m.source.contains("self.n1 = _backbone(\"resnet18\", 64)"));
    for (node, view) in [(0, VIEWS[0]), (OWNER, VIEWS[1]), (2, VIEWS[2])] {
        assert!(
            m.source
                .contains(&format!("v{node}_out = self.n1(inputs[{view:?}])")),
            "{}",
            m.source
        );
    }
    assert!(!m.source.contains("self.n0"), "{}", m.source);
    assert!(!m.source.contains("self.n2"), "{}", m.source);
    // The Sum, in the node's declared input order (wrist, overhead, side), not a map's, each
    // term with the camera it came from (packet M15/N7) and N6's stacked sum as the body.
    assert!(
        m.source.contains(
            "v4_out = _sum([v0_out, v1_out, v2_out], [\"rgb_wrist\", \"rgb_overhead\", \
             \"rgb_side\"], keep_views)"
        ),
        "{}",
        m.source
    );
    assert!(
        m.source.contains("return torch.stack(terms, 0).sum(0)"),
        "{}",
        m.source
    );
    // `keep_views` is a keyword only a `Sum` graph has, and it defaults to every view.
    assert!(
        m.source
            .contains("def forward(self, keep_views=None, **inputs):"),
        "{}",
        m.source
    );
    assert_eq!(Contract::new(&m, &g).sum_views, VIEWS);
    assert_eq!(es_policy::lower::sum_views(&g), VIEWS);
    // Packet M15/N7b: the encoders and the state path run once, before `post`; the `Sum` and
    // the head are inside it, which a list of view subsets runs once per subset.
    let (once, per_subset) = m
        .source
        .split_once("def post(keep_views):\n")
        .expect("a Sum graph has an inner post");
    assert!(once.contains("v0_out = self.n1(") && once.contains("v3_out = self.n3("));
    assert!(per_subset.contains("v4_out = _sum(") && !per_subset.contains("self.n1("));
    assert!(per_subset.contains("return [post(keep) for keep in keep_views]"));
    // One claim for the group, none for the sharers.
    let claims: Vec<&String> = m.weight_keys.iter().filter(|k| k.ends_with(".*")).collect();
    assert_eq!(claims, ["nodes.1.*"]);
    assert!(!m.weight_keys.iter().any(|k| k.starts_with("nodes.0.")));
    assert!(!m.weight_keys.iter().any(|k| k.starts_with("nodes.2.")));
    assert_eq!(m.sharers, BTreeMap::from([(0, OWNER), (2, OWNER)]));
    // `Sum` has no weight of its own.
    assert!(!m.weight_keys.iter().any(|k| k.starts_with("nodes.4.")));

    // The graph without the `share`s is three backbones and a different architecture.
    let mut unshared = g.clone();
    for node in unshared.nodes.nodes.values_mut() {
        if let LearningNode::VisionEncoder { share, .. } = node {
            *share = None;
        }
    }
    let three = lower_to_torch(&unshared).expect("lowers");
    assert_eq!(three.source.matches("_backbone(\"resnet18\"").count(), 3);
    assert_ne!(three.lowering_hash, m.lowering_hash);
    assert!(three.sharers.is_empty());
    println!(
        "RAN a_share_group_lowers_to_one_module_applied_per_view: lowering_hash {}",
        hex(&m.lowering_hash)
    );
}

/// A checkpoint for the graph: every exact key at its shape plus one tensor under each claim.
fn conforming(m: &es_policy::TorchModule) -> Checkpoint {
    let mut file: Checkpoint = m
        .weight_shapes
        .iter()
        .map(|(k, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            (k.clone(), (shape.clone(), vec![0.5; n]))
        })
        .collect();
    for claim in m.weight_keys.iter().filter(|k| k.ends_with(".*")) {
        file.insert(
            format!("{}conv1.weight", claim.trim_end_matches('*')),
            (vec![1], vec![1.0]),
        );
    }
    file
}

#[test]
fn a_checkpoint_with_a_sharers_own_tensors_is_refused_by_name() {
    let m = lower_to_torch(&three_views(false)).expect("lowers");
    let good = conforming(&m);
    validate_keys(&m, &parse_header(&write_safetensors(&good)).unwrap())
        .expect("one copy of the group, under the owner, fits");

    // A second copy under the wrist encoder: what a file trained against the unshared graph,
    // or an old export, still carries.
    let mut bad = good.clone();
    bad.insert("nodes.0.conv1.weight".to_owned(), (vec![1], vec![1.0]));
    bad.insert("nodes.0.fc.bias".to_owned(), (vec![1], vec![1.0]));
    let err = validate_keys(&m, &parse_header(&write_safetensors(&bad)).unwrap()).unwrap_err();
    let PolicyError::WeightMismatch {
        unexpected,
        missing,
        shape,
    } = &err
    else {
        panic!("{err}")
    };
    assert!(missing.is_empty() && shape.is_empty(), "{err:?}");
    assert_eq!(
        unexpected,
        &["nodes.0.* (node 0 shares node 1's weights; they are stored once, under nodes.1.*)"],
        "one entry naming the sharer, not two anonymous strays"
    );
    // A stray under no encoder is still listed by its own key.
    let mut stray = good;
    stray.insert("nodes.9.weight".to_owned(), (vec![1], vec![1.0]));
    let err = validate_keys(&m, &parse_header(&write_safetensors(&stray)).unwrap()).unwrap_err();
    assert!(
        matches!(&err, PolicyError::WeightMismatch { unexpected, .. } if unexpected == &["nodes.9.weight"]),
        "{err:?}"
    );
    println!("RAN a_checkpoint_with_a_sharers_own_tensors_is_refused_by_name");
}

// --- the Python-gated oracle -----------------------------------------------------------------

const REF: &str = include_str!("../python/sum_share_ref.py");

/// `ES_PYTHON` with `torch` and `torchvision`, or the reason to SKIP. Everything after this
/// probe is an error, never a skip (`docs/reviews/M4.md:67`, S-7).
fn python() -> Result<String, String> {
    let python = std::env::var("ES_PYTHON")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .ok_or("ES_PYTHON is unset")?;
    let out = Command::new(&python)
        .args(["-c", "import torch, torchvision"])
        .output()
        .map_err(|e| format!("`{python}`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`{python}` has no torch/torchvision: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("")
        ));
    }
    Ok(python)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-sum-share-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn f32_tensor(shape: &[u64], values: &[f64]) -> Tensor {
    Tensor {
        dtype: ElemType::F32,
        shape: shape.to_vec(),
        data: values
            .iter()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect(),
    }
}

#[test]
fn the_lowered_share_and_sum_equal_a_hand_written_module() {
    let python = match python() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP the_lowered_share_and_sum_equal_a_hand_written_module: {why}");
            return;
        }
    };
    let dir = scratch("oracle");
    let g = three_views(false);
    let m = lower_to_torch(&g).expect("lowers");
    let source = dir.join("es_policy.py");
    std::fs::write(&source, &m.source).expect("write the module");
    let pretrained = dir.join("es_policy_pretrained.py");
    std::fs::write(
        &pretrained,
        lower_to_torch(&three_views(true)).expect("lowers").source,
    )
    .expect("write the pretrained module");
    let weights = dir.join("policy.safetensors");
    let train_act = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/es/train_act.py");
    let spec = serde_json::json!({
        "source": source, "weights": weights, "pretrained_source": pretrained,
        "train_act": train_act,
        "owner": OWNER, "state": STATE_NODE, "head": HEAD,
        "views": VIEWS, "image": IMAGE, "state_dim": STATE, "width": W, "hidden": W,
        "horizon": HORIZON, "action_dim": ACTION_DIM, "execute": EXECUTE, "batch": 4,
        "keep_views": true, "single_view": 0.5,
    });
    let out = Command::new(&python)
        .args(["-c", REF, &spec.to_string()])
        .output()
        .expect("run the reference");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let reply: serde_json::Value = serde_json::from_str(stdout.lines().last().unwrap_or(""))
        .unwrap_or_else(|e| panic!("{e}: {stdout}\n{}", String::from_utf8_lossy(&out.stderr)));
    assert_eq!(reply["ok"], true, "{reply}");

    // 1. Bitwise, on CPU: the stacked sum and the hand-written `a + b + c` agree to the bit.
    assert_eq!(
        reply["bitwise"], true,
        "the lowered module and the hand-written one differ by {} (torch {})",
        reply["max_abs"], reply["torch"]
    );

    // 2. One copy of the encoder: the sharers hold no tensor, the owner a whole ResNet18.
    let per_node = &reply["keys_per_node"];
    assert!(
        per_node.get("0").is_none() && per_node.get("2").is_none(),
        "{per_node}"
    );
    let backbone = per_node["1"].as_u64().expect("the owner holds tensors");
    assert_eq!(
        backbone,
        reply["reference_backbone_tensors"]
            .as_u64()
            .expect("a count"),
        "the owner holds exactly one ResNet18"
    );

    // 3. The file loads through the runtime's own path -- `validate_keys` against this very
    //    lowering -- and infers sample 0 as the reference did.
    let bytes = std::fs::read(&weights).expect("the reference wrote the weights");
    let header = parse_header(&bytes).expect("safetensors");
    validate_keys(&m, &header).expect("one copy under the owner fits the lowering");
    let mut loaded = g.clone();
    loaded.policy.weights = WeightsRef::Safetensors {
        path: weights.to_string_lossy().into_owned(),
        hash: *blake3::hash(&bytes).as_bytes(),
    };
    let mut rt = TorchRuntime::new();
    rt.load(&loaded, &WeightsSource::Safetensors(weights.clone()))
        .expect("torch is here, so the load succeeds");
    let sample: BTreeMap<String, Tensor> = g
        .inputs
        .iter()
        .map(|p| {
            let values: Vec<f64> = reply["sample0_inputs"][&p.name]
                .as_array()
                .expect("the reference returns every input")
                .iter()
                .map(|v| v.as_f64().expect("a number"))
                .collect();
            (p.name.clone(), f32_tensor(p.ty.shape.dims(), &values))
        })
        .collect();
    let got = rt.infer(&sample).expect("one forward");
    let want: Vec<f32> = reply["sample0_actions"]
        .as_array()
        .expect("the reference's sample 0")
        .iter()
        .map(|v| v.as_f64().expect("a number") as f32)
        .collect();
    let want = Tensor {
        dtype: ElemType::F32,
        shape: vec![u64::from(EXECUTE), u64::from(ACTION_DIM)],
        data: want.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    // A batch of 1 against a row of a batch of 4: spec 8.9's tier-4 fp32, not bitwise.
    let e = compare_actions(&got["actions"], &want, Tolerance::TIER4_FP32);
    assert!(e.pass, "TorchRuntime vs the reference, sample 0: {e:?}");

    // 4. `--init-backbone` initialises the owner, and there is nothing else to initialise.
    let report = reply["init_backbone"].as_array().expect("a report");
    assert_eq!(report.len(), 1, "{report:?}");
    assert_eq!(report[0]["member"], "n1", "{report:?}");
    assert_eq!(reply["init_backbone_loaded_exactly"], true, "{reply}");

    // 5. Packet M15/N7: one camera kept is the hand-written module fed that camera alone,
    //    and keeping every camera is the default forward -- both bitwise.
    for view in VIEWS {
        assert_eq!(reply["keep_views_bitwise"][view], true, "{view}: {reply}");
    }
    assert_eq!(reply["keep_every_view_is_default"], true, "{reply}");

    // 6. Packet M15/N7b: `train_act.py`'s single-view loss from one forward (a list of view
    //    subsets) is N7's `1 + V` separate forwards to the bit, its gradient within spec 8.9's
    //    tier-4 fp32 (only the sum order moved), and the shared encoder runs once per view
    //    instead of once per view per pass.
    let sv = &reply["single_view"];
    assert_eq!(sv["loss_bitwise"], true, "{sv}");
    assert_eq!(sv["grad_tensors"][0], sv["grad_tensors"][1], "{sv}");
    let rel = sv["grad_max_rel"].as_f64().expect("a number");
    assert!(rel <= Tolerance::TIER4_FP32.rel, "{sv}");
    let views = VIEWS.len() as u64;
    assert_eq!(sv["encoder_calls"]["separate"], views * (1 + views), "{sv}");
    assert_eq!(sv["encoder_calls"]["shared"], views, "{sv}");

    println!(
        "RAN the_lowered_share_and_sum_equal_a_hand_written_module: torch {}, bitwise over a \
         batch of 4 (and with each camera alone); one ResNet18 copy ({backbone} tensors under \
         nodes.1); TorchRuntime sample 0 max_abs {:.3e}; init_backbone -> {}; single-view \
         loss {} bitwise, grad max_rel {rel:.3e}, encoder calls {} -> {}, seconds {} -> {}",
        reply["torch"],
        e.max_abs,
        report[0]["member"],
        sv["loss"][1],
        sv["encoder_calls"]["separate"],
        sv["encoder_calls"]["shared"],
        sv["seconds"]["separate"],
        sv["seconds"]["shared"]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// --- single-view training (packet M15/N7) -----------------------------------------------------

/// `es policy lower`'s two files for `g`, written directly: the trainer needs the module and
/// its contract, not a bundle.
fn module_dir(dir: &Path, g: &LearningGraph) -> PathBuf {
    let m = lower_to_torch(g).expect("lowers");
    let out = dir.join("module");
    std::fs::create_dir_all(&out).expect("module dir");
    std::fs::write(out.join("es_policy.py"), &m.source).expect("es_policy.py");
    std::fs::write(
        out.join("contract.json"),
        serde_json::to_string_pretty(&Contract::new(&m, g)).expect("serialises"),
    )
    .expect("contract.json");
    out
}

/// A synthetic bake in `es dataset bake`'s layout: two episodes of six frames, each camera
/// its own pixel pattern (so dropping one changes the input), a ramp of states and actions.
fn baked_dir(dir: &Path) -> PathBuf {
    const FRAMES: u64 = 6;
    let out = dir.join("baked");
    std::fs::create_dir_all(&out).expect("baked dir");
    let pixels: u64 = IMAGE.iter().product();
    let mut episodes = Vec::new();
    for e in 0..2u64 {
        let mut file = Checkpoint::new();
        for (i, view) in VIEWS.iter().enumerate() {
            let values = (0..FRAMES * pixels)
                .map(|j| ((j * 7 + 29 * i as u64 + 13 * e) % 256) as f32 / 255.0)
                .collect();
            let mut shape = vec![FRAMES];
            shape.extend(IMAGE);
            file.insert((*view).to_owned(), (shape, values));
        }
        let ramp = |w: u64| -> Vec<f32> {
            (0..FRAMES * w)
                .map(|j| 0.1 * ((j / w + 2 * e) as f32) * (1.0 + (j % w) as f32 / 4.0))
                .collect()
        };
        file.insert("joint_state".to_owned(), (vec![FRAMES, STATE], ramp(STATE)));
        let action = u64::from(ACTION_DIM);
        file.insert("action".to_owned(), (vec![FRAMES, action], ramp(action)));
        let name = format!("episode_{e:06}.safetensors");
        std::fs::write(out.join(&name), write_safetensors(&file)).expect("episode");
        episodes.push(serde_json::json!({"file": name, "frames": FRAMES}));
    }
    let mut tensors = serde_json::Map::new();
    for view in VIEWS {
        tensors.insert(
            view.to_owned(),
            serde_json::json!({"dtype": "F32", "shape": IMAGE}),
        );
    }
    tensors.insert(
        "joint_state".to_owned(),
        serde_json::json!({"dtype": "F32", "shape": [STATE]}),
    );
    let manifest = serde_json::json!({
        "schema_version": 1, "observation_hash": null, "episodes": episodes, "tensors": tensors,
    });
    std::fs::write(out.join("manifest.json"), manifest.to_string()).expect("manifest.json");
    out
}

/// `train_act.py` for three steps at batch 2, a progress line per step, plus `extra`.
/// Returns (exit code, the JSON lines on stdout, stderr).
fn train(
    python: &str,
    module: &Path,
    baked: &Path,
    out: &Path,
    extra: &[&str],
) -> (Option<i32>, Vec<serde_json::Value>, String) {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/es/train_act.py");
    let run = Command::new(python)
        .arg(&script)
        .args(["--module", &module.to_string_lossy()])
        .args(["--baked", &baked.to_string_lossy()])
        .args(["--out", &out.to_string_lossy()])
        .args([
            "--checkpoint-at",
            "3",
            "--seed",
            "0",
            "--batch",
            "2",
            "--lr",
            "1e-3",
        ])
        .args(["--device", "cpu", "--progress-every", "1"])
        .args(extra)
        .output()
        .expect("run train_act.py");
    let lines = String::from_utf8_lossy(&run.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    (
        run.status.code(),
        lines,
        String::from_utf8_lossy(&run.stderr).into_owned(),
    )
}

/// Packet M15/N7's oracle: `--single-view 0.5` runs, and every step's logged loss is
/// `loss_all + 0.5 * loss_single`, with the single-view term a different number (a camera's
/// term really was dropped). Without the flag no line carries either term. And the trainer
/// refuses the flag on a module whose contract lists no `sum_views`, or with a weight <= 0.
#[test]
fn single_view_training_logs_both_terms_of_its_loss() {
    let python = match python() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP single_view_training_logs_both_terms_of_its_loss: {why}");
            return;
        }
    };
    let dir = scratch("single-view");
    let module = module_dir(&dir, &three_views(false));
    let baked = baked_dir(&dir);

    let (code, lines, err) = train(
        &python,
        &module,
        &baked,
        &dir.join("sv.safetensors"),
        &["--single-view", "0.5"],
    );
    assert_eq!(code, Some(0), "{err}");
    let progress: Vec<&serde_json::Value> =
        lines.iter().filter_map(|l| l.get("progress")).collect();
    assert_eq!(progress.len(), 3, "{lines:?}");
    for p in &progress {
        let f = |k: &str| p[k].as_f64().unwrap_or_else(|| panic!("{k} in {p}"));
        let (loss, all, single) = (f("loss"), f("loss_all"), f("loss_single"));
        // f32 arithmetic in torch against this f64 recomputation: a relative 1e-6.
        assert!(
            (loss - (all + 0.5 * single)).abs() <= 1e-6 * loss.abs().max(1.0),
            "{p}: loss {loss} != {all} + 0.5 * {single}"
        );
        assert!(
            (all - single).abs() > 1e-6,
            "{p}: dropping two cameras' terms changed nothing"
        );
    }
    let summary = lines.last().expect("a summary line");
    let block = &summary["single_view"];
    assert_eq!(block["weight"], 0.5, "{summary}");
    assert_eq!(block["views"], serde_json::json!(VIEWS), "{summary}");
    for key in [
        "initial_loss_all",
        "final_loss_all",
        "initial_loss_single",
        "final_loss_single",
    ] {
        assert!(block[key].as_f64().is_some(), "{key}: {summary}");
    }

    // Without the flag: the loop of before, so neither term appears anywhere.
    let (code, lines, err) = train(
        &python,
        &module,
        &baked,
        &dir.join("plain.safetensors"),
        &[],
    );
    assert_eq!(code, Some(0), "{err}");
    for line in &lines {
        let text = line.to_string();
        assert!(
            !text.contains("loss_all") && !text.contains("single_view"),
            "{text}"
        );
    }

    // Refused by name: a weight that is not above 0, and a module with no camera `Sum`.
    let (code, _, err) = train(
        &python,
        &module,
        &baked,
        &dir.join("z.safetensors"),
        &["--single-view", "0"],
    );
    assert_ne!(code, Some(0));
    assert!(err.contains("--single-view is 0.0"), "{err}");
    let contract = module.join("contract.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&contract).expect("read")).expect("json");
    json.as_object_mut().expect("an object").remove("sum_views");
    std::fs::write(&contract, json.to_string()).expect("write");
    let (code, _, err) = train(
        &python,
        &module,
        &baked,
        &dir.join("n.safetensors"),
        &["--single-view", "0.5"],
    );
    assert_ne!(code, Some(0));
    assert!(err.contains("lists no `sum_views`"), "{err}");

    println!(
        "RAN single_view_training_logs_both_terms_of_its_loss: {} steps, first step loss {} = \
         {} + 0.5 * {}",
        progress.len(),
        progress[0]["loss"],
        progress[0]["loss_all"],
        progress[0]["loss_single"]
    );
    let _ = std::fs::remove_dir_all(&dir);
}
