//! Packet M15/N8 oracles: `es policy subset` (`es_policy::subset`, `docs/design/multi-camera.md`
//! section 3.4).
//!
//! The parent is plan N's three-view documents (`task-views.toml`, `observation-views.toml`,
//! `deployment.toml`) with `learning-views.toml` made MAD's: the three `VisionEncoder`s one
//! `share` group owned by the overhead camera's, their features summed by a new `Sum` node, the
//! sum concatenated with the joint state.
//!
//! Python-free, in the PR tier: each subset's documents validate and cross-validate -- against
//! an Evaluation IR whose observation reference is the subset's, derived here from
//! `evaluation-views.toml` -- every kept tensor's bytes are the parent's, the provenance is in
//! the weights, and the refusals are by name.
//!
//! Python-gated (`ES_PYTHON` with `torch` and `torchvision`; a loud SKIP otherwise): with random
//! weights, each subset bundle's lowered module equals the full bundle's run with
//! `keep_views=kept` (packet M15/N7) on the same inputs, **bitwise** on CPU -- keeping the
//! owner's view, keeping only a sharer's (the re-homing case), and keeping two.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use es_compile::PolicyBundle;
use es_ir::cross::{self, IrBundle};
use es_ir::graph::{NodeId, Port};
use es_ir::learning::{FusionKind, LearningGraph, LearningNode, WeightsRef};
use es_ir::observation::ObservationNode;
use es_ir::serial;
use es_policy::lower::Contract;
use es_policy::subset::{subset, views, SubsetError, SUBSET_OF_KEY, SUBSET_VIEWS_KEY};
use es_policy::weights::{metadata, parse_header, write_safetensors, Checkpoint};
use es_policy::{lower_to_torch, SafetensorsEntry};

const VIEWS: [&str; 3] = ["rgb_overhead", "rgb_wrist", "rgb_side"];
/// The overhead encoder, the group's owner; the wrist's and the side's.
const OWNER: u32 = 0;
const WRIST: u32 = 6;
const SIDE: u32 = 7;
/// The `Sum` this file adds, and the `Concat` it feeds.
const SUM: u32 = 8;
const CONCAT: u32 = 2;

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// `learning-views.toml` made MAD's (design note section 3.3): encoders 6 and 7 share node 0's
/// weights, the three features meet in `Sum` node 8, and node 8 takes the `image` port of the
/// `Concat` that used to take all three.
fn mad() -> LearningGraph {
    let mut g = serial::learning_from_toml(&fixture("learning-views.toml")).expect("parses");
    let Some(LearningNode::Fusion { inputs, .. }) = g.nodes.nodes.get_mut(&NodeId(CONCAT)) else {
        panic!("node 2 is the Concat")
    };
    let feature = inputs[0].clone();
    inputs.retain(|p| p.name == "image" || p.name == "state");
    let port = |name: &str| Port::new(name, feature.ty.clone());
    g.nodes.insert(
        NodeId(SUM),
        LearningNode::Fusion {
            inputs: vec![port("overhead"), port("wrist"), port("side")],
            kind: FusionKind::Sum,
            out_dim: 512,
            token_count: 0,
        },
    );
    g.nodes
        .edges
        .retain(|e| e.to.node != NodeId(CONCAT) || e.to.port == "state");
    for (from, to) in [(OWNER, "overhead"), (WRIST, "wrist"), (SIDE, "side")] {
        g.nodes.connect(NodeId(from), "out", NodeId(SUM), to);
    }
    g.nodes.connect(NodeId(SUM), "out", NodeId(CONCAT), "image");
    for id in [WRIST, SIDE] {
        set_share(&mut g, id, Some(OWNER));
    }
    g
}

fn set_share(g: &mut LearningGraph, id: u32, to: Option<u32>) {
    let Some(LearningNode::VisionEncoder { share, .. }) = g.nodes.nodes.get_mut(&NodeId(id)) else {
        panic!("node {id} is an encoder")
    };
    *share = to.map(NodeId);
}

fn share_of(g: &LearningGraph, id: u32) -> Option<u32> {
    match &g.nodes.nodes[&NodeId(id)] {
        LearningNode::VisionEncoder { share, .. } => share.map(|n| n.0),
        other => panic!("node {id} is {other:?}"),
    }
}

/// A checkpoint that fits `g`'s lowering: every exact key at its shape and two tensors under
/// each prefix claim, every value distinct enough that a swapped tensor would show.
fn weights_for(g: &LearningGraph) -> Vec<u8> {
    let m = lower_to_torch(g).expect("lowers");
    let mut file = Checkpoint::new();
    for (i, (key, shape)) in m.weight_shapes.iter().enumerate() {
        let n = shape.iter().product::<u64>() as usize;
        let values = (0..n)
            .map(|j| ((i * 131 + j * 7) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        file.insert(key.clone(), (shape.clone(), values));
    }
    for (i, claim) in m
        .weight_keys
        .iter()
        .filter(|k| k.ends_with(".*"))
        .enumerate()
    {
        let prefix = claim.trim_end_matches('*');
        let v = i as f32;
        file.insert(
            format!("{prefix}conv1.weight"),
            (vec![3], vec![v, v + 0.25, -v]),
        );
        file.insert(
            format!("{prefix}fc.bias"),
            (vec![2], vec![v * 0.5, 1.0 - v]),
        );
    }
    write_safetensors(&file)
}

/// The parent bundle: plan N's three-view documents, `g`, and `weights`.
fn parent(g: &LearningGraph, weights: &[u8]) -> PolicyBundle {
    let mut g = g.clone();
    g.policy.weights = WeightsRef::Safetensors {
        path: g.policy.weights.path().to_owned(),
        hash: *blake3::hash(weights).as_bytes(),
    };
    let bytes = PolicyBundle::build(
        &serial::task_from_toml(&fixture("task-views.toml")).expect("task"),
        &serial::observation_from_toml(&fixture("observation-views.toml")).expect("observation"),
        &g,
        &serial::deployment_from_toml(&fixture("deployment.toml")).expect("deployment"),
        weights,
    )
    .expect("the MAD graph makes a bundle with the three-view documents");
    PolicyBundle::open(&bytes).expect("reopens")
}

fn keep(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| (*s).to_owned()).collect()
}

/// Keeping the owner's view; only a sharer's (the owner re-homed); two, one of them a sharer.
const CASES: [&[&str]; 3] = [
    &["rgb_overhead"],
    &["rgb_wrist"],
    &["rgb_wrist", "rgb_side"],
];

fn tensor<'a>(file: &'a [u8], entry: &SafetensorsEntry) -> &'a [u8] {
    let base = 8 + u64::from_le_bytes(file[..8].try_into().expect("8 bytes")) as usize;
    &file[base + entry.offsets.0 as usize..base + entry.offsets.1 as usize]
}

#[test]
fn each_subset_validates_and_cross_validates_against_its_own_evaluation() {
    let g = mad();
    let p = parent(&g, &weights_for(&g));
    let evaluation = serial::evaluation_from_toml(&fixture("evaluation-views.toml")).expect("ev");
    for kept in CASES {
        let s = subset(&p, &keep(kept)).unwrap_or_else(|e| panic!("{kept:?}: {e}"));
        // `open` re-validates the four IRs, re-runs the cross-IR pass and every hash.
        let sub = PolicyBundle::open(&s.bytes).expect("the subset opens");
        assert_eq!(s.views, kept);
        assert_eq!(views(&sub.learning), kept);
        let mut ports: Vec<&str> = kept.to_vec();
        ports.push("joint_state");
        ports.sort_unstable();
        for names in [
            sub.observation
                .outputs
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            sub.learning
                .policy
                .contract
                .inputs
                .keys()
                .map(String::as_str)
                .collect(),
        ] {
            assert_eq!(names, ports, "{kept:?}");
        }
        // The dropped chains went, nothing else: a camera chain is six nodes, the state two.
        let cameras = sub
            .observation
            .graph
            .nodes
            .values()
            .filter(|n| matches!(n, ObservationNode::ImageInput { .. }))
            .count();
        assert_eq!(cameras, kept.len(), "{kept:?}");
        assert_eq!(
            sub.observation.graph.nodes.len(),
            2 + 6 * kept.len(),
            "{kept:?}"
        );
        // The Task and Deployment IR are the parent's; the hashes that read the views move.
        assert_eq!((&sub.task, &sub.deployment), (&p.task, &p.deployment));
        let (h, ph) = (sub.manifest.hashes, p.manifest.hashes);
        assert_eq!(h.task, ph.task);
        for (slot, mine, theirs) in [
            ("observation", h.observation, ph.observation),
            ("learning", h.learning, ph.learning),
            ("policy", h.policy, ph.policy),
        ] {
            assert_ne!(mine, theirs, "{kept:?}: {slot}_hash did not move");
        }

        // The parent's Evaluation IR names the parent's observation (XIR-040); the derived one,
        // with the subset's, cross-validates.
        let check = |ev| {
            cross::check(&IrBundle {
                task: &sub.task,
                observation: &sub.observation,
                learning: &sub.learning,
                deployment: &sub.deployment,
                evaluation: Some(ev),
            })
            .into_iter()
            .filter(es_ir::Diagnostic::is_error)
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
        };
        let stale = check(&evaluation);
        assert!(stale.iter().any(|d| d.contains("XIR-040")), "{stale:?}");
        let mut derived = evaluation.clone();
        derived.observation = hex(&h.observation.expect("an observation hash"));
        assert!(derived.validate().iter().all(|d| !d.is_error()));
        assert_eq!(check(&derived), Vec::<String>::new(), "{kept:?}");
    }
    println!("RAN each_subset_validates_and_cross_validates_against_its_own_evaluation: {CASES:?}");
}

#[test]
fn kept_tensors_are_the_parents_bytes_and_the_share_group_is_rehomed() {
    let g = mad();
    let p = parent(&g, &weights_for(&g));
    let parent_file = parse_header(&p.weights).expect("safetensors");
    let parent_hash = hex(&p.learning.policy_hash().expect("hashes"));
    for kept in CASES {
        let s = subset(&p, &keep(kept)).expect("subsets");
        let sub = PolicyBundle::open(&s.bytes).expect("opens");
        let file = parse_header(&sub.weights).expect("safetensors");

        // Old name -> new name, from what the subset reports it did; then byte for byte.
        let mut want = BTreeMap::new();
        for key in parent_file.keys() {
            let node: u32 = key
                .split('.')
                .nth(1)
                .expect("nodes.<id>.")
                .parse()
                .expect("id");
            let renamed = match s.rehomed.get(&node) {
                Some(to) => key.replacen(&format!("nodes.{node}."), &format!("nodes.{to}."), 1),
                None if s.dropped.contains(&node) => continue,
                None => key.clone(),
            };
            want.insert(renamed, key);
        }
        assert_eq!(
            file.keys().collect::<Vec<_>>(),
            want.keys().collect::<Vec<_>>(),
            "{kept:?}"
        );
        for (new, old) in &want {
            let (new_entry, old_entry) = (&file[new], &parent_file[*old]);
            assert_eq!(
                (&new_entry.dtype, &new_entry.shape),
                (&old_entry.dtype, &old_entry.shape),
                "{new}"
            );
            assert_eq!(
                tensor(&sub.weights, new_entry),
                tensor(&p.weights, old_entry),
                "{new} <- {old}"
            );
        }
        assert_eq!(
            metadata(&sub.weights, SUBSET_OF_KEY).expect("reads"),
            Some(parent_hash.clone())
        );
        assert_eq!(
            metadata(&sub.weights, SUBSET_VIEWS_KEY).expect("reads"),
            Some(kept.join(","))
        );
        assert_eq!(hex(&s.parent), parent_hash);

        // What happened to the group and the Sum, case by case.
        let sum = sub.learning.nodes.nodes.get(&NodeId(SUM));
        match kept {
            ["rgb_overhead"] => {
                assert!(s.rehomed.is_empty());
                assert_eq!(share_of(&sub.learning, OWNER), None);
                // The sharers held nothing, so nothing was removed.
                assert_eq!(file.len(), parent_file.len());
                assert!(sum.is_none(), "a Sum of one term is that term");
            }
            ["rgb_wrist"] => {
                assert_eq!(s.rehomed, BTreeMap::from([(OWNER, WRIST)]));
                assert_eq!(share_of(&sub.learning, WRIST), None);
                assert!(file.keys().any(|k| k.starts_with("nodes.6.")));
                assert!(!file.keys().any(|k| k.starts_with("nodes.0.")));
                assert!(sum.is_none());
            }
            _ => {
                assert_eq!(s.rehomed, BTreeMap::from([(OWNER, WRIST)]));
                assert_eq!(share_of(&sub.learning, WRIST), None);
                assert_eq!(share_of(&sub.learning, SIDE), Some(WRIST));
                let Some(LearningNode::Fusion { inputs, kind, .. }) = sum else {
                    panic!("two terms stay a Sum: {sum:?}")
                };
                assert_eq!(*kind, FusionKind::Sum);
                let names: Vec<&str> = inputs.iter().map(|p| p.name.as_str()).collect();
                assert_eq!(
                    names,
                    ["wrist", "side"],
                    "declared order, the dropped port out"
                );
            }
        }
    }

    // An unshared encoder whose only path is its Sum term goes with its own tensors.
    let mut unshared = mad();
    set_share(&mut unshared, SIDE, None);
    let p = parent(&unshared, &weights_for(&unshared));
    assert!(parse_header(&p.weights)
        .expect("safetensors")
        .keys()
        .any(|k| k.starts_with("nodes.7.")));
    let s = subset(&p, &keep(&["rgb_overhead", "rgb_wrist"])).expect("an unshared term drops");
    let sub = PolicyBundle::open(&s.bytes).expect("opens");
    assert!(s.rehomed.is_empty());
    assert!(!parse_header(&sub.weights)
        .expect("safetensors")
        .keys()
        .any(|k| k.starts_with("nodes.7.")));
    println!("RAN kept_tensors_are_the_parents_bytes_and_the_share_group_is_rehomed");
}

#[test]
fn refusals_are_by_name() {
    let g = mad();
    let p = parent(&g, &weights_for(&g));
    for (asked, view) in [
        (&["rgb_top"][..], "rgb_top"),
        (&["joint_state"], "joint_state"),
    ] {
        let err = subset(&p, &keep(asked)).unwrap_err();
        assert!(
            matches!(&err, SubsetError::UnknownView { view: v, views } if v == view && views == &VIEWS),
            "{err:?}"
        );
        assert!(err.to_string().contains(view), "{err}");
    }
    assert!(matches!(subset(&p, &[]), Err(SubsetError::NoViewKept)));

    // Experiment 1's own graph: three encoders into one Concat.
    let concat = serial::learning_from_toml(&fixture("learning-views.toml")).expect("parses");
    let err = subset(
        &parent(&concat, &weights_for(&concat)),
        &keep(&["rgb_wrist"]),
    )
    .unwrap_err();
    assert!(matches!(err, SubsetError::NoSum), "{err:?}");
    assert!(err.to_string().contains("Sum"), "{err}");

    // The overhead and the wrist summed, the side concatenated: only the summed two can go.
    let mut mixed = mad();
    let Some(LearningNode::Fusion { inputs, .. }) = mixed.nodes.nodes.get_mut(&NodeId(SUM)) else {
        panic!("the Sum")
    };
    let side = inputs.pop().expect("the side term");
    let Some(LearningNode::Fusion { inputs, .. }) = mixed.nodes.nodes.get_mut(&NodeId(CONCAT))
    else {
        panic!("the Concat")
    };
    inputs.push(side);
    for e in &mut mixed.nodes.edges {
        if e.from.node == NodeId(SIDE) {
            e.to.node = NodeId(CONCAT);
        }
    }
    let p = parent(&mixed, &weights_for(&mixed));
    let err = subset(&p, &keep(&["rgb_overhead", "rgb_wrist"])).unwrap_err();
    assert!(
        matches!(&err, SubsetError::NotASumTerm { view, through }
            if view == "rgb_side" && through == "node 2 (Fusion Concat)"),
        "{err:?}"
    );
    assert!(err.to_string().contains("rgb_side"), "{err}");
    let s = subset(&p, &keep(&["rgb_overhead", "rgb_side"])).expect("a summed view drops");
    PolicyBundle::open(&s.bytes).expect("opens");
    println!("RAN refusals_are_by_name");
}

// --- the Python-gated oracle -----------------------------------------------------------------

const REF: &str = include_str!("../python/subset_ref.py");

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

fn run(python: &str, spec: &serde_json::Value) -> serde_json::Value {
    let out = Command::new(python)
        .args(["-c", REF, &spec.to_string()])
        .output()
        .expect("run the reference");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let reply: serde_json::Value = serde_json::from_str(stdout.lines().last().unwrap_or(""))
        .unwrap_or_else(|e| panic!("{e}: {stdout}\n{}", String::from_utf8_lossy(&out.stderr)));
    assert_eq!(reply["ok"], true, "{reply}");
    reply
}

#[test]
fn a_subset_equals_the_full_graph_with_the_dropped_terms_left_out() {
    let python = match python() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP a_subset_equals_the_full_graph_with_the_dropped_terms_left_out: {why}");
            return;
        }
    };
    let dir = std::env::temp_dir().join(format!("es-subset-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let g = mad();
    let m = lower_to_torch(&g).expect("lowers");
    let (source, weights) = (dir.join("full.py"), dir.join("full.safetensors"));
    std::fs::write(&source, &m.source).expect("write the module");

    // Random weights, from the lowered module's own initialisation.
    run(
        &python,
        &serde_json::json!({"mode": "init", "source": source, "weights": weights}),
    );
    let p = parent(
        &g,
        &std::fs::read(&weights).expect("the reference wrote the weights"),
    );

    let mut cases = Vec::new();
    for (i, kept) in CASES.iter().enumerate() {
        let s = subset(&p, &keep(kept)).expect("subsets");
        let sub = PolicyBundle::open(&s.bytes).expect("opens");
        let (sub_source, sub_weights) = (
            dir.join(format!("subset-{i}.py")),
            dir.join(format!("subset-{i}.safetensors")),
        );
        let lowered = lower_to_torch(&sub.learning).expect("the subset lowers");
        std::fs::write(&sub_source, &lowered.source).expect("write");
        std::fs::write(&sub_weights, &sub.weights).expect("write");
        cases.push(serde_json::json!({
            "kept": kept, "source": sub_source, "weights": sub_weights,
            "inputs": sub.learning.policy.contract.inputs.keys().collect::<Vec<_>>(),
        }));
    }
    let reply = run(
        &python,
        &serde_json::json!({
            "mode": "compare", "source": source, "weights": weights, "batch": 2,
            "inputs": Contract::new(&m, &g).inputs, "cases": cases,
        }),
    );
    let results = reply["cases"].as_array().expect("one result per case");
    assert_eq!(results.len(), CASES.len());
    for r in results {
        assert_eq!(r["finite"], true, "{r}");
        assert_eq!(
            r["differs_from_every_view"], true,
            "a term was really dropped: {r}"
        );
        assert_eq!(r["bitwise"], true, "max_abs {}: {r}", r["max_abs"]);
    }
    println!(
        "RAN a_subset_equals_the_full_graph_with_the_dropped_terms_left_out: torch {}, bitwise \
         for {CASES:?}",
        reply["torch"]
    );
    let _ = std::fs::remove_dir_all(&dir);
}
