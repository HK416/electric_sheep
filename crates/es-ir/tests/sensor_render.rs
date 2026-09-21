//! Packet M7/R5 oracle 1: a sensor declares its render path, and declaring the **default**
//! one is not a document change (spec 5.3, spec 28.10 rule 1).
//!
//! The committed `task.toml` is the whole point. Its `task_hash` is named by the committed
//! `evaluation.toml`, by every bundle in `docs/design/visible-learning.md` 7.31 and by every
//! `execution_hash` derived from them; if adding `SensorRender` moved it, every one of those
//! documents would have to be regenerated and every recorded number would stop being about
//! the documents in this repository.

use std::path::PathBuf;

use es_ir::task::{ObsSource, SensorPath, SensorRender, TaskIr, Tonemap};

/// The `task_hash` of `tests/fixtures/visible-learning/task.toml`, recorded in
/// `docs/design/visible-learning.md` 7.31 and in the committed `evaluation.toml`'s `task`
/// field. Typed in here on purpose: this test is the one place the number is asserted rather
/// than derived.
const COMMITTED_TASK_HASH: &str =
    "eb6efefa2010befff0c73a5cf8c089f1818e374a7c61a73a64dcd78849102127";

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning")
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

/// The one image channel's `render`, by the name both fixtures give it.
fn render_of(ir: &TaskIr) -> SensorRender {
    match &ir
        .observation_spec
        .channels
        .get("rgb_overhead")
        .expect("the demo declares one image channel")
        .source
    {
        ObsSource::Sensor { render, .. } => *render,
        other => panic!("rgb_overhead is not a sensor channel: {other:?}"),
    }
}

fn with_render(ir: &TaskIr, render: SensorRender) -> TaskIr {
    let mut ir = ir.clone();
    match &mut ir
        .observation_spec
        .channels
        .get_mut("rgb_overhead")
        .expect("the demo declares one image channel")
        .source
    {
        ObsSource::Sensor { render: r, .. } => *r = render,
        other => panic!("rgb_overhead is not a sensor channel: {other:?}"),
    }
    ir
}

/// The committed document with an **explicit** default `render` written into its text, so the
/// claim is about what a person could type and not only about what `Default` returns.
fn with_explicit_default_render(text: &str) -> String {
    let mut out = String::new();
    let mut in_sensor = false;
    for line in text.lines() {
        out.push_str(line);
        out.push('\n');
        let line = line.trim();
        if line == "[body.observation_spec.channels.rgb_overhead.source.Sensor]" {
            in_sensor = true;
        } else if in_sensor && line.starts_with("id = ") {
            out.push_str("render = { path = \"rs\", exposure = 1.0, tonemap = \"reinhard\" }\n");
            in_sensor = false;
        }
    }
    assert!(!in_sensor, "the sensor table has no `id` key");
    out
}

#[test]
fn committed_task_hash_is_unmoved_by_sensor_render() {
    let text = fixture("task.toml");
    let committed = es_ir::serial::task_from_toml(&text).expect("task.toml parses");
    assert_eq!(
        hash_of(&committed),
        COMMITTED_TASK_HASH,
        "the committed task_hash moved"
    );
    assert_eq!(
        render_of(&committed),
        SensorRender::default(),
        "an absent `render` is the default one"
    );

    // Absent = default = today's canonical form, in the text as well as in the type.
    let explicit = es_ir::serial::task_from_toml(&with_explicit_default_render(&text))
        .expect("an explicit default `render` parses");
    assert_eq!(render_of(&explicit), SensorRender::default());
    assert_eq!(
        hash_of(&explicit),
        COMMITTED_TASK_HASH,
        "an explicitly written default `render` moved the hash"
    );
    assert!(explicit.validate().is_empty(), "{:?}", explicit.validate());

    // A `Pt` sensor is a different document (spec 13.3), and differs in nothing else: put the
    // default back and the committed hash returns.
    let pt = es_ir::serial::task_from_toml(&fixture("task-pt.toml")).expect("task-pt.toml parses");
    let declared = render_of(&pt);
    assert_eq!(
        declared.path,
        SensorPath::Pt {
            spp: 64,
            bounces: 3
        },
        "task-pt.toml declares the path tracer"
    );
    assert_eq!(declared.tonemap, Tonemap::Reinhard);
    let pt_hash = hash_of(&pt);
    assert_ne!(
        pt_hash, COMMITTED_TASK_HASH,
        "a Pt sensor must move the hash"
    );
    assert_eq!(
        hash_of(&with_render(&pt, SensorRender::default())),
        COMMITTED_TASK_HASH,
        "task-pt.toml differs from task.toml in more than the sensor's render path"
    );
    assert!(pt.validate().is_empty(), "{:?}", pt.validate());
    println!(
        "RAN committed_task_hash_is_unmoved_by_sensor_render: task-pt task_hash {pt_hash}, \
         exposure {}",
        declared.exposure
    );
}

/// Every knob of the block moves the hash when it is set, and none of them moves it while the
/// block as a whole is the default: the `if not default` rule in `ObsSource::canonical` is not
/// allowed to swallow a field.
#[test]
fn every_render_field_is_hash_input() {
    let committed = es_ir::serial::task_from_toml(&fixture("task.toml")).expect("task.toml parses");
    let base = SensorRender::default();
    for changed in [
        SensorRender {
            path: SensorPath::Pt {
                spp: 64,
                bounces: 3,
            },
            ..base
        },
        SensorRender {
            path: SensorPath::Pt {
                spp: 65,
                bounces: 3,
            },
            ..base
        },
        SensorRender {
            path: SensorPath::Pt {
                spp: 64,
                bounces: 4,
            },
            ..base
        },
        SensorRender {
            exposure: 32.0,
            ..base
        },
        SensorRender {
            tonemap: Tonemap::Aces,
            ..base
        },
    ] {
        let hash = hash_of(&with_render(&committed, changed));
        assert_ne!(
            hash, COMMITTED_TASK_HASH,
            "{changed:?} did not move the hash"
        );
        // ... and it round-trips through TOML with the same hash.
        let text = es_ir::serial::task_to_toml(&with_render(&committed, changed)).expect("toml");
        let back = es_ir::serial::task_from_toml(&text).expect("it parses again");
        assert_eq!(render_of(&back), changed, "{changed:?} did not round trip");
        assert_eq!(hash_of(&back), hash);
    }
}
