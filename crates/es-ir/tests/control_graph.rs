//! IR-C at the crate boundary (spec 6.2): a Task IR that carries a control graph must
//! validate, round-trip through the `.esgraph` TOML envelope, and keep `task_hash` a semantic
//! identity — relabel-invariant, but sensitive to every IR-C parameter.
//!
//! The node-level rules (`CTRL-001`..`CTRL-006`, `TYPE-030`) are covered by the unit tests in
//! `src/control.rs`; this file is about the composition with the rest of the Task IR.

use es_ir::control::{ControlGraph, ControlNode, RepeatUntil};
use es_ir::graph::NodeId;
use es_ir::norm::{canon_task, testing::shuffle_task_ids};
use es_ir::task::testing::{control_tree, task_ir};
use es_ir::task::TaskIr;

fn terms() -> Vec<(String, f64, f64)> {
    vec![
        ("reach".to_owned(), 1.0, 0.5),
        ("lift".to_owned(), 2.0, 0.5),
    ]
}

fn staged() -> TaskIr {
    let terms = terms();
    let mut ir = task_ir(&terms, 3);
    ir.control = Some(control_tree(&terms));
    ir
}

#[test]
fn control_a_staged_task_validates_clean() {
    assert_eq!(staged().validate(), Vec::new());
}

#[test]
fn control_survives_the_toml_envelope_with_its_hash() {
    let ir = staged();
    let text = es_ir::serial::task_to_toml(&ir).expect("writes");
    let back = es_ir::serial::task_from_toml(&text).expect("parses");
    assert_eq!(back, ir);
    assert_eq!(back.task_hash().unwrap(), ir.task_hash().unwrap());
    assert!(text.contains("SubTask"), "the control graph is in the file");
}

#[test]
fn control_survives_serde_json() {
    let ir = staged();
    let json = serde_json::to_string(&ir).expect("writes");
    assert_eq!(serde_json::from_str::<TaskIr>(&json).unwrap(), ir);
}

/// A file written before IR-C has no `control` key at all and must still parse.
#[test]
fn control_defaults_to_none_when_the_file_omits_it() {
    let plain = task_ir(&terms(), 3);
    let text = es_ir::serial::task_to_toml(&plain).expect("writes");
    assert!(
        !text.contains("[body.control]"),
        "None writes no control section"
    );
    assert_eq!(
        es_ir::serial::task_from_toml(&text)
            .expect("parses")
            .control,
        None
    );
}

#[test]
fn control_hash_is_relabel_invariant_and_normalization_stable() {
    let ir = staged();
    let base = ir.task_hash().unwrap();
    // Both graphs are renumbered: IR-D by `shuffle_ids`, IR-C by its own permutation.
    for seed in [1u64, 7, 99] {
        let shuffled = shuffle_task_ids(&ir, seed);
        assert_ne!(shuffled.control, ir.control, "seed {seed} renumbered IR-C");
        assert_eq!(shuffled.task_hash().unwrap(), base, "seed {seed}");
    }
    assert_eq!(canon_task(&ir).unwrap().task_hash().unwrap(), base);
}

#[test]
fn adding_or_editing_a_control_graph_moves_the_task_hash() {
    let plain = task_ir(&terms(), 3);
    let ir = staged();
    assert_ne!(ir.task_hash().unwrap(), plain.task_hash().unwrap());

    let base = ir.task_hash().unwrap();
    let mut edited = ir.clone();
    let control = edited.control.as_mut().expect("a control graph");
    let ControlNode::SubTask { timeout_ticks, .. } =
        control.nodes.get_mut(&NodeId(1)).expect("the abort stage")
    else {
        unreachable!("node 1 is a SubTask")
    };
    *timeout_ticks += 1;
    assert_ne!(edited.task_hash().unwrap(), base, "a timeout is semantic");

    let mut repeat = ir.clone();
    let control = repeat.control.as_mut().expect("a control graph");
    control.nodes.insert(
        NodeId(2),
        ControlNode::Repeat {
            body: NodeId(3),
            until: RepeatUntil::Count(3),
        },
    );
    assert_ne!(repeat.task_hash().unwrap(), base, "a count is semantic");
}

/// The control graph is not the IR-D graph: a stage names rewards, it does not own nodes.
#[test]
fn a_control_graph_does_not_change_the_task_graph_hash_input() {
    let plain = task_ir(&terms(), 3);
    let ir = staged();
    assert_eq!(ir.graph, plain.graph);
    assert_ne!(
        ir.task_graph_hash().unwrap(),
        plain.task_graph_hash().unwrap()
    );
}

/// The five B.7 properties run over `arbitrary_task_ir`, which now emits control trees; this
/// pins the piece those properties rest on — an empty graph is still rejected, not hashed.
#[test]
fn an_empty_control_graph_is_reported_not_hashed() {
    let mut ir = staged();
    ir.control = Some(ControlGraph {
        root: NodeId(0),
        nodes: std::collections::BTreeMap::new(),
    });
    let diags = ir.validate();
    assert!(
        diags
            .iter()
            .any(|d| d.code.as_str() == es_ir::codes::CTRL_001),
        "{diags:?}"
    );
}
