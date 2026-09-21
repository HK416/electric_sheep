//! Packet M8/S4e oracle 1: an observation channel names the joint quantity it carries, and
//! naming the **default** one is not a document change (spec 5.3, spec 28.10 rule 1, the rule
//! packet M7/R5 set for `SensorRender`).
//!
//! The committed `task.toml` is the whole point, for the reason `sensor_render.rs` gives: its
//! `task_hash` is named by the committed `evaluation.toml`, by every bundle in
//! `docs/design/visible-learning.md` 7.31 and by every `execution_hash` derived from them.

use std::path::PathBuf;

use es_ir::cross::{self, IrBundle};
use es_ir::task::{JointQuantity, ObsSource, TaskIr};

/// `tests/fixtures/visible-learning/task.toml`'s `task_hash`, the same literal
/// `sensor_render.rs` pins, typed in on purpose.
const COMMITTED_TASK_HASH: &str =
    "eb6efefa2010befff0c73a5cf8c089f1818e374a7c61a73a64dcd78849102127";
/// `tests/fixtures/visible-learning/task-pt.toml`'s.
const COMMITTED_PT_HASH: &str = "d546b80804dd035ed6c696ddc251f8ce2d68d1a3964407d00f76af746c95e2fa";

fn fixture(dir: &str, name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(dir)
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hash_of(ir: &TaskIr) -> String {
    use std::fmt::Write;
    ir.task_hash()
        .expect("the task hashes")
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// The committed document with an **explicit** `Position` written into every `JointState`
/// channel, so the claim is about what a person could type and not only about what serde's
/// `default` returns.
fn with_explicit_position(text: &str) -> String {
    let mut out = String::new();
    let mut in_state = false;
    for line in text.lines() {
        out.push_str(line);
        out.push('\n');
        let trimmed = line.trim();
        if trimmed.ends_with(".source.JointState]") {
            in_state = true;
        } else if in_state && trimmed.starts_with("dof = ") {
            out.push_str("quantity = \"Position\"\n");
            in_state = false;
        }
    }
    assert!(!in_state, "a JointState table has no `dof` key");
    out
}

#[test]
fn committed_task_hashes_are_unmoved_by_joint_quantity() {
    for (name, want) in [
        ("task.toml", COMMITTED_TASK_HASH),
        ("task-pt.toml", COMMITTED_PT_HASH),
    ] {
        let text = fixture("visible-learning", name);
        let committed =
            es_ir::serial::task_from_toml(&text).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(hash_of(&committed), want, "{name}: the task_hash moved");
        for ch in committed.observation_spec.channels.values() {
            if let ObsSource::JointState { quantity, .. } = ch.source {
                assert_eq!(
                    quantity,
                    JointQuantity::Position,
                    "{name}: an absent `quantity` is Position"
                );
            }
        }

        // Absent = Position = today's canonical form, in the text as well as in the type.
        let explicit = es_ir::serial::task_from_toml(&with_explicit_position(&text))
            .expect("an explicit `quantity = \"Position\"` parses");
        assert_eq!(
            hash_of(&explicit),
            want,
            "{name}: an explicitly written `Position` moved the hash"
        );
        assert!(explicit.validate().is_empty(), "{:?}", explicit.validate());

        // ... and `Velocity` does move it, on every channel that can carry it.
        let mut moved = 0;
        let keys: Vec<String> = committed
            .observation_spec
            .channels
            .keys()
            .cloned()
            .collect();
        for key in keys {
            let mut ir = committed.clone();
            let ch = ir.observation_spec.channels.get_mut(&key).expect("the key");
            let ObsSource::JointState { quantity, .. } = &mut ch.source else {
                continue;
            };
            *quantity = JointQuantity::Velocity;
            assert_ne!(
                hash_of(&ir),
                want,
                "{name}: `quantity = Velocity` on \"{key}\" did not move the hash"
            );
            // ... and it round-trips through TOML with that same moved hash.
            let text = es_ir::serial::task_to_toml(&ir).expect("toml");
            let back = es_ir::serial::task_from_toml(&text).expect("it parses again");
            assert_eq!(hash_of(&back), hash_of(&ir), "{name}: \"{key}\" round trip");
            moved += 1;
        }
        assert!(moved > 0, "{name} declares no JointState channel");
        println!("RAN committed_task_hashes_are_unmoved_by_joint_quantity: {name} {want}");
    }
}

/// The reach documents, as five parsed IRs. The one place in `es-ir`'s tests that reads them:
/// a five-sided bundle is what `cross::check` takes, and building one by hand here would be a
/// second copy of `crates/es/tests/cli.rs`'s generator.
fn reach() -> (
    TaskIr,
    es_ir::observation::ObservationIr,
    es_ir::learning::LearningGraph,
    es_ir::deployment::DeploymentIr,
    es_ir::evaluation::EvaluationIr,
) {
    (
        es_ir::serial::task_from_toml(&fixture("rl", "task-reach.toml")).expect("task"),
        es_ir::serial::observation_from_toml(&fixture("rl", "observation-reach.toml"))
            .expect("observation"),
        es_ir::serial::learning_from_toml(&fixture("rl", "learning-reach.toml")).expect("learning"),
        es_ir::serial::deployment_from_toml(&fixture("rl", "deployment-reach.toml"))
            .expect("deployment"),
        es_ir::serial::evaluation_from_toml(&fixture("rl", "evaluation-reach.toml"))
            .expect("evaluation"),
    )
}

/// `XIR-002` keys a state channel by `(id, quantity)`: a position channel and a velocity
/// channel may name one joint block, and two channels that declare the same thing twice may
/// not (`StateInput` carries no quantity of its own, so what tells the pair apart is the
/// declared type, whose unit *is* `JointQuantity::unit()`).
#[test]
fn two_quantities_on_one_joint_block() {
    use es_ir::observation::ObservationNode;

    let (task, observation, learning, deployment, _evaluation) = reach();
    // No Evaluation IR: this test moves `task_hash` on purpose, and `XIR-040` (the evaluation
    // names another task) would then fire on every mutation and mask the rule under test.
    // `crates/es/tests/cli.rs`'s `reach_documents_validate` checks the five-sided bundle.
    let check = |t: &TaskIr, o: &es_ir::observation::ObservationIr| {
        cross::check(&IrBundle {
            task: t,
            observation: o,
            learning: &learning,
            deployment: &deployment,
            evaluation: None,
        })
    };
    // The committed documents give every channel its own id, and they agree.
    let base = match task.observation_spec.channels["joint_pos"].source {
        ObsSource::JointState { body, .. } => body,
        ref other => panic!("joint_pos is not a JointState channel: {other:?}"),
    };
    assert!(
        check(&task, &observation).is_empty(),
        "{:#?}",
        check(&task, &observation)
    );

    // Both channels on `base`, at two quantities: accepted. `task_ref` is re-pointed because
    // moving a channel moves `task_hash`, and `XIR-001` would otherwise mask the answer.
    let moved = |quantity: JointQuantity, ty: Option<es_ir::types::PortType>| {
        let mut t = task.clone();
        let ch = t
            .observation_spec
            .channels
            .get_mut("joint_vel")
            .expect("joint_vel");
        ch.source = ObsSource::JointState {
            body: base,
            dof: 6,
            quantity,
        };
        if let Some(ty) = ty {
            ch.ty = ty;
        }
        let mut o = observation.clone();
        let node = o.graph.nodes.get_mut(&es_ir::graph::NodeId(1)).expect("n1");
        let ObservationNode::StateInput { source, .. } = node else {
            panic!("node 1 is not a StateInput");
        };
        *source = base;
        o.task_ref = t.task_hash().expect("the moved task hashes");
        (t, o)
    };

    let (t, o) = moved(JointQuantity::Velocity, None);
    let diags = check(&t, &o);
    assert!(
        diags.is_empty(),
        "a position channel and a velocity channel on one joint block: {diags:#?}"
    );

    // The same quantity twice, declared identically: refused, rather than resolved by
    // whichever name sorts first.
    let joint_pos_ty = task.observation_spec.channels["joint_pos"].ty.clone();
    let (t, o) = moved(JointQuantity::Position, Some(joint_pos_ty));
    let diags = check(&t, &o);
    let said = diags.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert!(
        said.iter().any(|d| d.contains("XIR-002")),
        "two identical channels on one id were accepted: {said:#?}"
    );
    println!(
        "RAN two_quantities_on_one_joint_block: {} refusals",
        said.len()
    );
}
