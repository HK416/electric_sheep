//! Packet M15/N5 oracle: `FusionKind::Sum` and `VisionEncoder.share` (spec 8.3,
//! `docs/design/multi-camera.md` sections 3.3 and 6).
//!
//! Three things are claimed. A three-view graph whose encoders share one weight group and meet
//! in a `Sum` validates clean; every way of getting either wrong is refused by name
//! (`LRN-032`, `LRN-033`); and adding the two changes no committed document — every Learning IR
//! under `tests/fixtures/` hashes and serializes exactly as it did on `main` before this packet.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use es_ir::codes::{LRN_032, LRN_033};
use es_ir::graph::{Graph, NodeId, Port, PortRef};
use es_ir::learning::{
    action_unit, ActionExecutionMode, Activation, ArchKind, ChunkBlendPolicy, FusionKind, HeadKind,
    LearningGraph, LearningNode, PolicyContract, PolicyHandle, RuntimeHints, Squash,
    StateEncoderKind, TemporalKind, VisionBackbone, WeightsRef,
};
use es_ir::serial::{learning_from_toml, learning_to_toml};
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

// --- The committed documents ----------------------------------------------------------------

/// `(document, learning_hash, policy_hash, blake3 of learning_to_toml, blake3 of serde_json)`,
/// every one computed on `main` (e0c3403) before this packet touched anything, and typed in
/// here on purpose: this is the one place the numbers are asserted rather than derived. The
/// two serializations are the documents' canonical encodings; an unset `share` that leaked
/// into either (`share = ...`, `"share":null`) would move its digest.
const COMMITTED: [(&str, &str, &str, &str, &str); 8] = [
    (
        "quadruped/learning.toml",
        "daac51f8fbc97490b00a845fc47ad1e997dd2b7b2ead8ca2df89310ac002056a",
        "04a7f3c97b212db4488cb1572a3fa46b2436aa26c51ba78388dbebced3d8e5f8",
        "19cfbb1068474468e7e736bff86f4647f3bec9f74b67575265093442d2f8bfe1",
        "2bd230ccb9cb7f25c62c31f42963292a965db339e4840ddb118ab255d83de099",
    ),
    (
        "rl/learning-reach-delta.toml",
        "e79acf42f01855d845211b2df0e0dc006d04c7984eb8d5b8e4cbf1f7dffa46ba",
        "7372b5cece2e3be31425781caf2932b16d039a0716420f5e3701a180eea0fe2d",
        "4ec958edba72f78d0569521d01278edae94de26063df3a8e3b2ea38e84fdc530",
        "9d96fcf2558fb44fcb55028eff5d3b633e7a5178623cda8858fee92fbc9f5ff8",
    ),
    (
        "rl/learning-reach-vision-pix.toml",
        "1b42a6ff5fdc35c17b723cd28c08a961d9e9120a2a38ff2245ffedd45acc6039",
        "deea490824f8728ae1016ee69b1c1817e9631daefe455e4be7c6134f1cf35840",
        "427f8b78ed5150d9085617ed48e40447a6219995918e3b436977c6832531e88b",
        "bcf7e797cf65c7b54c9b8760b854afc3be2e1e8f815f05ba11dd16fba6d7dee1",
    ),
    (
        "rl/learning-reach-vision.toml",
        "df1119f0b9d2eb8e8a4278d752ad4a8604e486f6729ee013cf99d198671f6d9d",
        "281b2360bd43acf64f47800060e3459d98d4f0210ab36079a0f318a34385d7a8",
        "d02973964e909d0a66f9caf3f8cca14a4eb8d04f3984875fae6a9cb227841851",
        "d6806843dc43ddc0acd69244e4aa9bf6a10c7eeb68ca5c488f553a6aa61023de",
    ),
    (
        "rl/learning-reach.toml",
        "eb805f185d711b0f9b319fcb0079f741e566971b7edeb109d93cb888e1ff1070",
        "18bddeb5c6a5a01d8288d52c8151ccfbf279006d87ecfaaa43b6c40264b781bf",
        "11a706bb00d2b3038318776d17a264cae3b52e0c7a929c9c6587dc5f7dd7259e",
        "1fe262480d7c80b44f41663bd6cad717f5d1665b46c582834581a911aa36c766",
    ),
    (
        "rl/learning-state.toml",
        "a670b686e47b0e010767b264a27487915ec52d73dc9e09e4515251c60bee78c8",
        "2bebf347b40491fbfab1e9672d5f740e93b1499233338f8bf22dfb9f23be21a3",
        "4a2ed3203d1f22de23e1fb07f7c40a029299d1decd4a21d5009c0f0d7a596534",
        "6564ee89121ba2bb1eeee63bd04d529745aea94c88cda991129c9990b481d17d",
    ),
    (
        "visible-learning/learning-pretrained.toml",
        "fdb5178a52c24507f583b5cfc54cd832dfcda0901216d1da705060c264ac699e",
        "d5e152b96abb9d8f85bc4194226a4f78cb1e632c8c1d443e144e78af9ebc0976",
        "fbd7d8992a2edb6383dce9de78c9503e0734808de3443c65f2f0baf18f564531",
        "91ea91c3d2c4fb636bb17731defadc352f63b28583d3141bc65ec01d2e4292a1",
    ),
    (
        "visible-learning/learning.toml",
        "5dac0a46f56198b1a6d04ead8ee43b18913356c56ee99b01a35decc66dc446f0",
        "c94c2732e215e4d8ac39cc0bd280c2d5495e5e36dbbc2da39f41ac952bd84a07",
        "7cfb2f46d61a187b06b0ff129d25a8657ff17ab9097fa0ad661d682f9bded865",
        "1010b72f597dd979a615a62a79ecf83e62f1818760bc8d432590bb6bb2a915ee",
    ),
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[test]
fn committed_learning_irs_hash_and_serialize_as_on_main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for (name, learning, policy, toml, json) in COMMITTED {
        let path = root.join(name);
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let g = learning_from_toml(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        let written = learning_to_toml(&g).expect("writes");
        let actual = (
            hex(&g.learning_hash().expect("hashes")),
            hex(&g.policy_hash().expect("hashes")),
            hex(blake3::hash(written.as_bytes()).as_bytes()),
            hex(blake3::hash(serde_json::to_string(&g).expect("json").as_bytes()).as_bytes()),
        );
        assert_eq!(
            actual,
            (
                learning.to_owned(),
                policy.to_owned(),
                toml.to_owned(),
                json.to_owned()
            ),
            "{name} moved"
        );
        assert!(
            !written.contains("share"),
            "{name}: an unset share was written"
        );
    }
}

// --- A three-view graph (MAD, design note section 3.3) ---------------------------------------

const W: u32 = 64;

/// A node renaming, and an edit of one encoder, as plain function pointers.
type Relabel = fn(u32) -> u32;
type Edit = fn(&mut LearningNode);

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
    Port::new(
        name,
        ty(&[3, 96, 96], Unit::Normalized { lo: 0.0, hi: 1.0 }),
    )
}

fn encoder(view: &str, share: Option<u32>) -> LearningNode {
    LearningNode::VisionEncoder {
        inputs: vec![image(view)],
        backbone: VisionBackbone::ResNet18,
        pretrained: true,
        frozen: false,
        out_dim: W,
        token_count: 0,
        share: share.map(NodeId),
    }
}

/// Wrist (0), overhead (1) and side (2) cameras through one `ResNet18` owned by the overhead
/// encoder — between its two sharers, so neither "owner first" nor "owner last" is assumed —
/// summed, then concatenated with the joint state.
fn three_views() -> LearningGraph {
    let state = Port::new("joint_state", ty(&[6], action_unit()));
    let chunk = |name: &str, h: u64| Port::new(name, ty(&[h, 6], action_unit()));
    let mut g = Graph::new(1);
    g.insert(NodeId(0), encoder("rgb_wrist", Some(1)));
    g.insert(NodeId(1), encoder("rgb_overhead", None));
    g.insert(NodeId(2), encoder("rgb_side", Some(1)));
    g.insert(
        NodeId(3),
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
            out_dim: W,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(6),
        LearningNode::TemporalEncoder {
            inputs: vec![feature("seq", W)],
            kind: TemporalKind::Transformer,
            n_frames: 1,
            out_dim: W,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(7),
        LearningNode::PolicyHead {
            inputs: vec![feature("feat", W)],
            kind: HeadKind::Regression,
            action_dim: 6,
            horizon: 16,
            squash: Squash::None,
        },
    );
    g.insert(
        NodeId(8),
        LearningNode::ActionChunker {
            inputs: vec![chunk("chunk", 16)],
            horizon: 16,
            execute_chunk: 10,
            replan_hz: 5.0,
            mode: ActionExecutionMode::TemporalEnsemble,
            blend: ChunkBlendPolicy::TemporalEnsemble { weight_decay: 0.01 },
            buffer_chunks: 2,
        },
    );
    for (from, to, port) in [
        (0, 4, "wrist"),
        (1, 4, "overhead"),
        (2, 4, "side"),
        (4, 5, "views"),
        (3, 5, "state"),
        (5, 6, "seq"),
        (6, 7, "feat"),
    ] {
        g.connect(NodeId(from), "out", NodeId(to), port);
    }
    g.connect(NodeId(7), "chunk", NodeId(8), "chunk");
    let inputs = vec![
        image("rgb_wrist"),
        image("rgb_overhead"),
        image("rgb_side"),
        state,
    ];
    g.inputs = inputs
        .iter()
        .enumerate()
        .map(|(i, p)| PortRef::new(NodeId(i as u32), p.name.clone()))
        .collect();
    g.outputs = vec![PortRef::new(NodeId(8), "actions")];
    LearningGraph {
        schema_version: 1,
        outputs: vec![chunk("actions", 10)],
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
                action_dim: 6,
                horizon: 16,
                execute_chunk: 10,
                replanning_hz: 5.0,
                execution_mode: ActionExecutionMode::TemporalEnsemble,
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

fn errors(g: &LearningGraph) -> Vec<String> {
    g.validate()
        .into_iter()
        .filter(es_ir::Diagnostic::is_error)
        .map(|d| format!("{} {}", d.code.as_str(), d.message))
        .collect()
}

/// `code` is reported, on `node`, with `needle` in its message.
fn refused(g: &LearningGraph, code: &str, node: u32, needle: &str) {
    let diags = g.validate();
    let found = diags.iter().find(|d| {
        d.code.as_str() == code && d.node == Some(NodeId(node)) && d.message.contains(needle)
    });
    let d = found
        .unwrap_or_else(|| panic!("no {code} on node {node} saying {needle:?} among {diags:#?}"));
    println!("RAN {code}: {}", d.message);
}

fn node_mut(g: &mut LearningGraph, id: u32) -> &mut LearningNode {
    g.nodes.nodes.get_mut(&NodeId(id)).expect("the node exists")
}

/// Renames every node by `f`, carrying edges, the boundary and `share` along.
fn relabel(g: &LearningGraph, f: impl Fn(u32) -> u32) -> LearningGraph {
    let at = |id: NodeId| NodeId(f(id.0));
    let port = |p: &PortRef| PortRef::new(at(p.node), p.port.clone());
    let mut out = g.clone();
    out.nodes.nodes = g
        .nodes
        .nodes
        .iter()
        .map(|(id, n)| {
            let mut n = n.clone();
            if let LearningNode::VisionEncoder {
                share: Some(owner), ..
            } = &mut n
            {
                *owner = at(*owner);
            }
            (at(*id), n)
        })
        .collect();
    for e in &mut out.nodes.edges {
        (e.from, e.to) = (port(&e.from), port(&e.to));
    }
    out.nodes.inputs = g.nodes.inputs.iter().map(port).collect();
    out.nodes.outputs = g.nodes.outputs.iter().map(port).collect();
    out
}

#[test]
fn three_views_with_one_shared_encoder_and_a_sum_validate() {
    let g = three_views();
    assert!(errors(&g).is_empty(), "{:#?}", errors(&g));
}

#[test]
fn the_owner_may_come_before_or_after_its_sharers() {
    let g = three_views();
    let hash = g.learning_hash().expect("hashes");
    // Owner first (the overhead encoder becomes node 0), owner last (node 8), and every id
    // moved far away: the relation is by id, never by position.
    let cases: [(&str, Relabel); 3] = [
        ("owner first", |i| match i {
            0 => 1,
            1 => 0,
            i => i,
        }),
        ("owner last", |i| match i {
            1 => 8,
            8 => 1,
            i => i,
        }),
        ("ids reversed", |i| 100 - i),
    ];
    for (what, f) in cases {
        let moved = relabel(&g, f);
        assert!(errors(&moved).is_empty(), "{what}: {:#?}", errors(&moved));
        assert_eq!(moved.learning_hash().expect("hashes"), hash, "{what}");
    }
}

#[test]
fn normalization_carries_share_to_the_new_ids() {
    let g = three_views();
    let canon = es_ir::canon_learning(&g).expect("normalizes");
    assert!(errors(&canon).is_empty(), "{:#?}", errors(&canon));
    assert_eq!(canon.learning_hash().unwrap(), g.learning_hash().unwrap());
    let owner_of = |g: &LearningGraph, view: &str| -> Option<NodeId> {
        g.nodes.nodes.values().find_map(|n| match n {
            LearningNode::VisionEncoder { inputs, share, .. } if inputs[0].name == view => *share,
            _ => None,
        })
    };
    let overhead = canon
        .nodes
        .nodes
        .iter()
        .find_map(|(id, n)| match n {
            LearningNode::VisionEncoder { inputs, .. } if inputs[0].name == "rgb_overhead" => {
                Some(*id)
            }
            _ => None,
        })
        .expect("the overhead encoder survives");
    assert_eq!(owner_of(&canon, "rgb_wrist"), Some(overhead));
    assert_eq!(owner_of(&canon, "rgb_side"), Some(overhead));
}

#[test]
fn share_is_written_and_read_back() {
    let g = three_views();
    let text = learning_to_toml(&g).expect("writes");
    assert!(text.contains("share = 1"), "{text}");
    assert!(text.contains("kind = \"Sum\""), "{text}");
    assert_eq!(learning_from_toml(&text).expect("reads"), g);
    let json = serde_json::to_string(&g).expect("json");
    assert_eq!(
        serde_json::from_str::<LearningGraph>(&json).expect("json"),
        g
    );
}

#[test]
fn which_encoders_share_is_part_of_the_learning_hash() {
    let with = |shares: [Option<u32>; 3]| {
        let mut g = three_views();
        for (id, share) in shares.into_iter().enumerate() {
            let LearningNode::VisionEncoder { share: s, .. } = node_mut(&mut g, id as u32) else {
                unreachable!()
            };
            *s = share.map(NodeId);
        }
        g.learning_hash().expect("hashes")
    };
    let hashes = [
        with([None, None, None]),
        with([Some(1), None, None]),
        with([None, None, Some(1)]),
        with([Some(1), None, Some(1)]),
        with([None, Some(0), Some(0)]),
    ];
    let distinct: BTreeSet<_> = hashes.iter().collect();
    assert_eq!(
        distinct.len(),
        hashes.len(),
        "two sharing patterns hash alike"
    );
}

// --- Sum refusals (LRN-032) -------------------------------------------------------------------

#[test]
fn a_sum_term_of_another_width_is_lrn_032() {
    let mut g = three_views();
    let LearningNode::Fusion { inputs, .. } = node_mut(&mut g, 4) else {
        unreachable!()
    };
    inputs[2] = feature("side", 32);
    refused(&g, LRN_032, 4, "\"side\" is [32]");
}

#[test]
fn a_sum_whose_out_dim_is_not_its_terms_width_is_lrn_032() {
    let mut g = three_views();
    let LearningNode::Fusion { out_dim, .. } = node_mut(&mut g, 4) else {
        unreachable!()
    };
    *out_dim = 3 * W;
    refused(&g, LRN_032, 4, "out_dim 192");
}

#[test]
fn a_sum_of_tokens_needs_the_same_token_count() {
    let mut g = three_views();
    let LearningNode::Fusion { token_count, .. } = node_mut(&mut g, 4) else {
        unreachable!()
    };
    *token_count = 4;
    refused(&g, LRN_032, 4, "token_count 4");

    // The same Sum over token features that all agree is fine by LRN-032.
    let LearningNode::Fusion { inputs, .. } = node_mut(&mut g, 4) else {
        unreachable!()
    };
    for p in inputs.iter_mut() {
        p.ty.shape = Shape::new([4, u64::from(W)]);
    }
    assert!(
        !g.validate().iter().any(|d| d.code.as_str() == LRN_032),
        "{:#?}",
        errors(&g)
    );
}

// --- share refusals (LRN-033) -----------------------------------------------------------------

fn sharer_with(edit: impl FnOnce(&mut LearningNode)) -> LearningGraph {
    let mut g = three_views();
    edit(node_mut(&mut g, 0));
    g
}

fn set_share(n: &mut LearningNode, to: u32) {
    let LearningNode::VisionEncoder { share, .. } = n else {
        unreachable!()
    };
    *share = Some(NodeId(to));
}

#[test]
fn share_naming_no_node_is_lrn_033() {
    refused(
        &sharer_with(|n| set_share(n, 42)),
        LRN_033,
        0,
        "node 42 does not exist",
    );
}

#[test]
fn share_naming_a_node_that_is_not_a_vision_encoder_is_lrn_033() {
    refused(
        &sharer_with(|n| set_share(n, 3)),
        LRN_033,
        0,
        "node 3 is a StateEncoder, not a VisionEncoder",
    );
}

#[test]
fn share_naming_itself_is_lrn_033() {
    refused(
        &sharer_with(|n| set_share(n, 0)),
        LRN_033,
        0,
        "cannot share its own weights",
    );
}

#[test]
fn share_does_not_chain() {
    // The side camera shares the wrist camera, which itself shares the overhead one.
    let mut g = three_views();
    set_share(node_mut(&mut g, 2), 0);
    refused(&g, LRN_033, 2, "node 0 itself shares node 1");
}

#[test]
fn share_across_different_encoders_is_lrn_033() {
    let cases: [(&str, Edit); 5] = [
        ("backbone is ResNet34 here but ResNet18 on node 1", |n| {
            if let LearningNode::VisionEncoder { backbone, .. } = n {
                *backbone = VisionBackbone::ResNet34;
            }
        }),
        ("out_dim is 32 here but 64 on node 1", |n| {
            if let LearningNode::VisionEncoder { out_dim, .. } = n {
                *out_dim = 32;
            }
        }),
        ("token_count is 4 here but 0 on node 1", |n| {
            if let LearningNode::VisionEncoder { token_count, .. } = n {
                *token_count = 4;
            }
        }),
        ("pretrained is false here but true on node 1", |n| {
            if let LearningNode::VisionEncoder { pretrained, .. } = n {
                *pretrained = false;
            }
        }),
        ("frozen is true here but false on node 1", |n| {
            if let LearningNode::VisionEncoder { frozen, .. } = n {
                *frozen = true;
            }
        }),
    ];
    for (needle, edit) in cases {
        refused(&sharer_with(edit), LRN_033, 0, needle);
    }
}
