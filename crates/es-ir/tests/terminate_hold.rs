//! Packet M18/K7 (spec 6.3, design note `scene-authoring.md` section 4.9): `Terminate` may hold
//! for `n` control ticks, and an absent hold is today's document. Oracle 2's IR half: every
//! committed Task IR keeps its `task_hash`, serializes without the field and parses back to the
//! same hash; a hold moves the hash and round-trips; `0` is `TASK-004`.

use std::path::{Path, PathBuf};

use es_ir::task::{TaskIr, TaskNode};

/// `tests/fixtures/visible-learning/task.toml`'s, as `sensor_render.rs` pins it.
const COMMITTED_TASK_HASH: &str =
    "86a7f3a3410f2ae70b48db7fa8d746552745aa3b75c4d3c5cb178549e47ab5ba";

fn hex(ir: &TaskIr) -> String {
    use std::fmt::Write;
    (ir.task_hash().expect("hashes").iter()).fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Every committed Task IR document: a `task*.toml` under `tests/fixtures` that parses as one.
fn committed() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "toml")
                && (path.file_name().unwrap().to_string_lossy()).starts_with("task")
            {
                out.push(path);
            }
        }
    }
    let mut paths = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
        &mut paths,
    );
    paths.sort();
    (paths.into_iter())
        .filter_map(|p| Some((p.clone(), std::fs::read_to_string(&p).ok()?)))
        .filter(|(_, text)| es_ir::serial::task_from_toml(text).is_ok())
        .collect()
}

fn holds(ir: &mut TaskIr) -> Vec<&mut Option<u32>> {
    (ir.graph.nodes.values_mut())
        .filter_map(|n| match n {
            TaskNode::Terminate { hold_ticks, .. } => Some(hold_ticks),
            _ => None,
        })
        .collect()
}

#[test]
fn an_absent_hold_is_todays_document() {
    let docs = committed();
    assert!(docs.len() >= 10, "{} committed Task IRs", docs.len());
    let mut terminates = 0;
    for (path, text) in &docs {
        let mut ir = es_ir::serial::task_from_toml(text).unwrap();
        terminates += holds(&mut ir).len();
        assert!(holds(&mut ir).iter().all(|h| h.is_none()), "{path:?}");
        let written = es_ir::serial::task_to_toml(&ir).unwrap();
        assert!(
            !written.contains("hold_ticks"),
            "{path:?}: absent is not written"
        );
        let back = es_ir::serial::task_from_toml(&written).unwrap();
        assert_eq!(hex(&back), hex(&ir), "{path:?}");
        if path.ends_with("visible-learning/task.toml") {
            assert_eq!(
                hex(&ir),
                COMMITTED_TASK_HASH,
                "the committed task_hash moved"
            );
        }
    }
    assert!(terminates > 0);
    println!(
        "{} committed Task IRs, {terminates} Terminate nodes",
        docs.len()
    );
}

#[test]
fn a_hold_is_hash_input_and_zero_is_refused() {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/visible-learning/task.toml"),
    )
    .unwrap();
    let ir = es_ir::serial::task_from_toml(&text).unwrap();
    let mut held = ir.clone();
    *holds(&mut held).remove(0) = Some(60);
    assert!(held.validate().is_empty(), "{:?}", held.validate());
    assert_ne!(hex(&held), hex(&ir), "a hold moves the task_hash");
    let mut other = ir.clone();
    *holds(&mut other).remove(0) = Some(61);
    assert_ne!(hex(&other), hex(&held), "and its length does");

    let written = es_ir::serial::task_to_toml(&held).unwrap();
    assert!(written.contains("hold_ticks = 60"), "{written}");
    let mut back = es_ir::serial::task_from_toml(&written).unwrap();
    assert_eq!(hex(&back), hex(&held));
    assert_eq!(holds(&mut back).remove(0), &mut Some(60));

    let mut zero = ir;
    *holds(&mut zero).remove(0) = Some(0);
    let codes: Vec<_> = (zero.validate().iter())
        .map(|d| d.code.as_str().to_owned())
        .collect();
    assert_eq!(codes, [es_ir::codes::TASK_004]);
}
