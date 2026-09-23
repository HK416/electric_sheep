//! Oracle 3 of packet M11/X2: `ObsSource::PreviousAction` is an addition, not a change.
//!
//! The rule is spec 28.14's rule 1: a document that does not declare the new channel keeps its
//! bytes and its hash. Every committed Task IR is checked against the `task_ref` the committed
//! Observation IR beside it names -- a number written before this packet existed -- and the one
//! new document is checked to differ from its parent by exactly the new channel.

use std::path::PathBuf;

use es_ir::serial::{observation_from_toml, task_from_toml, task_to_toml};
use es_ir::task::{ObsSource, TaskIr};

fn fixture(path: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn task(path: &str) -> TaskIr {
    task_from_toml(&fixture(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn previous_action_leaves_committed_task_hashes_unmoved() {
    for (task_path, observation_path) in [
        ("rl/task-reach.toml", "rl/observation-reach.toml"),
        (
            "rl/task-reach-delta.toml",
            "rl/observation-reach-delta.toml",
        ),
        (
            "visible-learning/task.toml",
            "visible-learning/observation.toml",
        ),
        (
            "visible-learning/task-pt.toml",
            "visible-learning/observation-pt.toml",
        ),
        (
            "visible-learning/task-pt-tick.toml",
            "visible-learning/observation-pt-tick.toml",
        ),
    ] {
        let named = observation_from_toml(&fixture(observation_path))
            .unwrap_or_else(|e| panic!("{observation_path}: {e}"))
            .task_ref;
        assert_eq!(
            task(task_path).task_hash().expect("hashes"),
            named,
            "{task_path} no longer hashes to the task_ref {observation_path} names"
        );
    }
}

#[test]
fn previous_action_channel_moves_the_hash() {
    let parent = task("rl/task-reach.toml");
    let child = task("rl/task-reach-last-action.toml");
    assert_ne!(
        parent.task_hash().expect("hashes"),
        child.task_hash().expect("hashes"),
        "a PreviousAction channel is a new document"
    );
    // ...and it is the only difference: without it the child is its parent, hash for hash.
    let mut stripped = child.clone();
    stripped.observation_spec.channels.remove("last_action");
    assert_eq!(
        stripped.task_hash().expect("hashes"),
        parent.task_hash().expect("hashes")
    );

    // `initial` is part of the document: a different reset value is a different task.
    let mut other = child.clone();
    if let ObsSource::PreviousAction { initial } = &mut other
        .observation_spec
        .channels
        .get_mut("last_action")
        .expect("the channel")
        .source
    {
        *initial = None;
    }
    assert_ne!(
        other.task_hash().expect("hashes"),
        child.task_hash().expect("hashes")
    );
}

#[test]
fn previous_action_initial_round_trips() {
    let child = task("rl/task-reach-last-action.toml");
    let initial = match &child.observation_spec.channels["last_action"].source {
        ObsSource::PreviousAction { initial } => initial.clone(),
        other => panic!("last_action is {other:?}"),
    };
    assert_eq!(initial, Some(vec![0.1, -0.4, 0.7, 0.25, -0.05, 0.3]));

    let text = task_to_toml(&child).expect("writes");
    let back = task_from_toml(&text).expect("reads back");
    assert_eq!(back, child);
    assert_eq!(back.task_hash().ok(), child.task_hash().ok());

    // Absent `initial` is absent in the text too (zeros at reset), and reads back as absent.
    let mut zero = child.clone();
    zero.observation_spec
        .channels
        .get_mut("last_action")
        .expect("the channel")
        .source = ObsSource::PreviousAction { initial: None };
    let text = task_to_toml(&zero).expect("writes");
    assert!(
        !text.contains("initial"),
        "an absent initial is not written"
    );
    assert_eq!(task_from_toml(&text).expect("reads back"), zero);
}
