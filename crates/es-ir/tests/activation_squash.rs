//! Packet M8/S2a oracle 1 and 2: an `Mlp` says which activation it uses and whether the last
//! layer is activated, a `Regression` head says whether its output is squashed, and declaring
//! the **defaults** is not a document change (spec 5.3, spec 14.4).
//!
//! The committed `learning.toml` and `learning-pretrained.toml` are the whole point. Their
//! `learning_hash`es are named by `docs/design/learning-lowering.md` section 5.3, by every
//! bundle in `docs/design/visible-learning.md` section 7 and by every `execution_hash` derived
//! from them; if three new node parameters moved them, every recorded number would stop being
//! about the documents in this repository.

use std::fmt::Write as _;
use std::path::PathBuf;

use es_ir::graph::NodeId;
use es_ir::learning::{Activation, HeadKind, LearningGraph, LearningNode, Squash};

/// `learning_hash` of `tests/fixtures/visible-learning/learning.toml`, computed on `main`
/// before this packet touched anything. Typed in here on purpose: this test is the one place
/// the number is asserted rather than derived.
const COMMITTED_LEARNING_HASH: &str =
    "5dac0a46f56198b1a6d04ead8ee43b18913356c56ee99b01a35decc66dc446f0";

/// The same, for `learning-pretrained.toml` (one `pretrained` flag apart, packet M7/T5).
const PRETRAINED_LEARNING_HASH: &str =
    "fdb5178a52c24507f583b5cfc54cd832dfcda0901216d1da705060c264ac699e";

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hash_of(ir: &LearningGraph) -> String {
    ir.learning_hash()
        .expect("the graph hashes")
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// The committed document with the three **defaults** written into its text, so the claim is
/// about what a person could type and not only about what `Default` returns. The demo has two
/// `Mlp` encoders and one head.
fn with_explicit_defaults(text: &str) -> String {
    let mut out = String::new();
    let (mut mlps, mut heads) = (0, 0);
    for line in text.lines() {
        out.push_str(line);
        out.push('\n');
        match line.trim() {
            "hidden = [256]" => {
                out.push_str("activation = \"Relu\"\nactivate_output = false\n");
                mlps += 1;
            }
            "kind = \"Regression\"" => {
                out.push_str("squash = \"None\"\n");
                heads += 1;
            }
            _ => {}
        }
    }
    assert_eq!((mlps, heads), (2, 1), "the demo's shape changed");
    out
}

/// Every way of spelling a non-default, applied to the committed graph.
fn non_defaults(ir: &LearningGraph) -> Vec<(&'static str, LearningGraph)> {
    let mut out = Vec::new();
    for (what, activation, activate_output) in [
        ("activation = Elu", Activation::Elu, false),
        ("activation = Swish", Activation::Swish, false),
        ("activation = Tanh", Activation::Tanh, false),
        ("activate_output = true", Activation::Relu, true),
    ] {
        let mut g = ir.clone();
        let LearningNode::StateEncoder { kind, .. } = g
            .nodes
            .nodes
            .get_mut(&NodeId(1))
            .expect("node 1 is the MLP")
        else {
            panic!("node 1 is not a StateEncoder");
        };
        *kind = es_ir::learning::StateEncoderKind::Mlp {
            hidden: vec![256],
            activation,
            activate_output,
        };
        out.push((what, g));
    }

    let mut g = ir.clone();
    let LearningNode::PolicyHead { squash, .. } = g
        .nodes
        .nodes
        .get_mut(&NodeId(4))
        .expect("node 4 is the head")
    else {
        panic!("node 4 is not a PolicyHead");
    };
    *squash = Squash::Tanh;
    out.push(("squash = Tanh", g));
    out
}

#[test]
fn committed_learning_hashes_are_unmoved_by_activation_and_squash() {
    for (name, pinned) in [
        ("learning.toml", COMMITTED_LEARNING_HASH),
        ("learning-pretrained.toml", PRETRAINED_LEARNING_HASH),
    ] {
        let text = fixture(name);
        let committed = es_ir::serial::learning_from_toml(&text).expect("the fixture parses");
        assert_eq!(hash_of(&committed), pinned, "{name}: learning_hash moved");

        // Absent = default = today's canonical form, in the text as well as in the type.
        let explicit = es_ir::serial::learning_from_toml(&with_explicit_defaults(&text))
            .expect("explicit defaults parse");
        assert_eq!(
            hash_of(&explicit),
            pinned,
            "{name}: explicitly written defaults moved the hash"
        );
        assert_eq!(
            explicit, committed,
            "{name}: the documents differ as values"
        );
        assert!(explicit.validate().is_empty(), "{:?}", explicit.validate());

        // ... and a value that is not the default does move it, or the parameter is decorative.
        for (what, changed) in non_defaults(&committed) {
            let moved = hash_of(&changed);
            assert_ne!(moved, pinned, "{name}: {what} did not move the hash");
            // It round-trips through TOML with the same hash.
            let back = es_ir::serial::learning_from_toml(
                &es_ir::serial::learning_to_toml(&changed).expect("it serializes"),
            )
            .expect("it parses again");
            assert_eq!(hash_of(&back), moved, "{name}: {what} did not round trip");
            assert!(back.validate().is_empty(), "{:?}", back.validate());
        }
    }
    println!(
        "RAN committed_learning_hashes_are_unmoved_by_activation_and_squash: \
         learning.toml {COMMITTED_LEARNING_HASH}, learning-pretrained.toml \
         {PRETRAINED_LEARNING_HASH}"
    );
}

/// The inspector's parameter table has to know the field exists even though a default node
/// does not serialize it (packet M7/E3, `factory::optional_params`).
#[test]
fn the_policy_head_schema_declares_squash() {
    let registry = es_ir::factory::LearningNodeRegistry::with_builtins();
    let schema = registry.schema("PolicyHead").expect("a builtin schema");
    let squash = schema
        .params
        .iter()
        .find(|p| p.name == "squash")
        .expect("the schema declares squash");
    assert!(!squash.required, "squash is optional: absent = None");
    assert_eq!(
        squash.default,
        Some(toml::Value::String("None".to_owned())),
        "the starting point a fresh head offers is the default"
    );
}

#[test]
fn squash_is_refused_off_a_regression_head() {
    let committed = es_ir::serial::learning_from_toml(&fixture("learning.toml")).expect("parses");
    let mut g = committed.clone();
    let LearningNode::PolicyHead { kind, squash, .. } = g
        .nodes
        .nodes
        .get_mut(&NodeId(4))
        .expect("node 4 is the head")
    else {
        panic!("node 4 is not a PolicyHead");
    };
    *kind = HeadKind::FlowMatching { n_steps: 10 };
    *squash = Squash::Tanh;

    let diags = g.validate();
    let found = diags
        .iter()
        .find(|d| d.code.as_str() == es_ir::codes::LRN_031)
        .unwrap_or_else(|| panic!("no LRN-031 among {diags:#?}"));
    assert!(found.node.is_some(), "the diagnostic names the node");
    println!(
        "RAN squash_is_refused_off_a_regression_head: {}",
        found.message
    );

    // The same head with the default squash is not refused by LRN-031.
    let LearningNode::PolicyHead { squash, .. } = g
        .nodes
        .nodes
        .get_mut(&NodeId(4))
        .expect("node 4 is the head")
    else {
        unreachable!()
    };
    *squash = Squash::None;
    assert!(
        !g.validate()
            .iter()
            .any(|d| d.code.as_str() == es_ir::codes::LRN_031),
        "a default squash is not a diagnostic on any head"
    );
}
