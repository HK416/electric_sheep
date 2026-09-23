//! Oracle 2 of packet M11/X2: an Isaac-Lab-style `rsl_rl` actor (both checkpoint shapes
//! `rsl_rl` has shipped) and a Playground-style brax actor, each imported through adapter v2
//! and run through our runtime -- Observation IR on `CpuPlan`, Learning IR on `TorchRuntime` --
//! against `python/es/rl_source/isaac_reference.py`, which computes what the source framework
//! would command on the same 256 states from the api-notes' formulas in `NumPy`.
//!
//!     ES_PYTHON=<python with torch, numpy, yaml> cargo test -p es-import --test isaac_reference
//!
//! Skips with a printed reason without `ES_PYTHON`: the generator and the importer's Python
//! half need torch.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use es_compile::{CpuPlan, PlanMode, TensorRef};
use es_import::rl_import::{convert, Adapter, ImportManifest};
use es_ir::learning::WeightsRef;
use es_ir::serial::{deployment_from_toml, task_from_toml};
use es_ir::task::ObsSource;
use es_ir::types::ElemType;
use es_policy::{PolicyRuntime, TorchRuntime, WeightsSource};

/// Max abs error allowed, in actuator units (the packet's number).
const TOLERANCE: f64 = 1e-6;

const ACTUATORS: [&str; 6] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rl(name: &str) -> String {
    let path = root().join("tests/fixtures/rl").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn python(py: &str, script: &str, args: &[String]) -> String {
    let out = Command::new(py)
        .current_dir(root())
        .arg(script)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("run {script}: {e}"));
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{script} {args:?}:\n{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    text
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("json file")).expect("json")
}

fn rows(v: &serde_json::Value) -> Vec<Vec<f64>> {
    v.as_array()
        .expect("rows")
        .iter()
        .map(|r| {
            r.as_array()
                .expect("row")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect()
}

/// One source: generate, import (both halves), run 256 states, return the max abs error.
/// `edit` rewrites the adapter's text first (the negative control).
fn max_error(py: &str, kind: &str, adapter: &str, edit: (&str, &str)) -> f64 {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("x2-{kind}"));
    let _ = std::fs::remove_dir_all(&dir);
    let out = dir.to_string_lossy().into_owned();
    print!(
        "{}",
        python(
            py,
            "python/es/rl_source/isaac_reference.py",
            &["--make".into(), kind.into(), "--out".into(), out.clone()],
        )
    );
    let mut args: Vec<String> =
        serde_json::from_value(json(&dir.join("import_args.json"))).expect("import args");
    args.extend(["--out".into(), format!("{out}/neutral")]);
    python(py, "python/es/import_rl.py", &args);

    let manifest = ImportManifest::parse(
        &std::fs::read_to_string(dir.join("neutral/import.json")).expect("import.json"),
    )
    .expect("manifest");
    let weights = std::fs::read(dir.join("neutral/weights.safetensors")).expect("weights");
    let text = rl(adapter);
    assert!(text.contains(edit.0), "the edit's anchor is in {adapter}");
    let adapter = Adapter::parse(&text.replace(edit.0, edit.1)).expect("adapter");
    let task = task_from_toml(&rl("task-reach-last-action.toml")).expect("task");
    let deployment = deployment_from_toml(&rl("deployment-reach.toml")).expect("deployment");
    let actuators: Vec<String> = ACTUATORS.iter().map(|s| (*s).to_owned()).collect();
    let imported = convert(
        &manifest,
        &adapter,
        &task,
        &deployment,
        &weights,
        &actuators,
    )
    .unwrap_or_else(|e| panic!("{kind}: {e}"));
    for w in &imported.report.warnings {
        println!("{kind} warning: {w}");
    }

    let mut learning = imported.learning;
    learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: es_policy::weights::weights_hash(&imported.weights),
    };
    let mut runtime = TorchRuntime::new();
    runtime
        .load(
            &learning,
            &WeightsSource::InMemory(imported.weights.clone()),
        )
        .expect("TorchRuntime loads the imported bundle");
    let mut plan = CpuPlan::compile(&imported.observation, PlanMode::Release)
        .unwrap_or_else(|d| panic!("{kind}: the observation compiles: {d:?}"));

    // Plan input name -> the Task IR channel whose values feed it.
    let by_input: BTreeMap<String, String> = task
        .observation_spec
        .channels
        .iter()
        .map(|(name, ch)| {
            let id = match &ch.source {
                ObsSource::JointState { body, .. } => *body,
                ObsSource::BodyPose(id) => *id,
                ObsSource::PreviousAction { .. } => ObsSource::previous_action_id(),
                other => panic!("{name}: {other:?}"),
            };
            (id.to_string(), name.clone())
        })
        .collect();
    // The source's `clip_observations = 100` is not declarable (IMP-009); it is exact to leave
    // it out only because it never binds on these states.
    let max_obs = json(&dir.join("reference.json"))["max_abs_obs"]
        .as_f64()
        .expect("max_abs_obs");
    assert!(
        max_obs < 100.0,
        "{kind}: the observation clip binds ({max_obs})"
    );
    let states = json(&dir.join("states.json"));
    let reference = rows(&json(&dir.join("reference.json"))["actions"]);
    let columns: BTreeMap<&str, Vec<Vec<f64>>> =
        ["joint_pos", "joint_vel", "cube_pose", "last_action"]
            .into_iter()
            .map(|c| (c, rows(&states[c])))
            .collect();

    let mut worst = 0.0f64;
    for (r, want) in reference.iter().enumerate() {
        let mut bytes: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
        for (input, id) in &plan.inputs {
            let channel = by_input
                .get(input)
                .unwrap_or_else(|| panic!("plan input {input} is no channel"));
            let values = &columns[channel.as_str()][r];
            let data = values
                .iter()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect();
            bytes.push((input.clone(), plan.buffers[id.0].shape.clone(), data));
        }
        let inputs: BTreeMap<String, TensorRef<'_>> = bytes
            .iter()
            .map(|(n, shape, data)| {
                (
                    n.clone(),
                    TensorRef::new(ElemType::F32, shape.clone(), data),
                )
            })
            .collect();
        plan.reset();
        let ports = plan.run(&inputs).expect("the plan runs");
        let state = ports.get("state").expect("the state port").clone();
        let outputs = runtime
            .infer(&BTreeMap::from([("state".to_owned(), state)]))
            .expect("infer");
        let got: Vec<f32> = outputs
            .values()
            .next()
            .expect("one output")
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(got.len(), want.len(), "{kind}: action width");
        for (g, w) in got.iter().zip(want) {
            worst = worst.max((f64::from(*g) - w).abs());
        }
    }
    println!(
        "{kind}: {} states, max abs error {worst:.3e} (tolerance {TOLERANCE:.0e}), obs max |x| {}",
        reference.len(),
        json(&dir.join("reference.json"))["max_abs_obs"]
    );
    worst
}

#[test]
fn isaac_reference() {
    let Ok(py) = std::env::var("ES_PYTHON") else {
        println!("SKIP isaac_reference: set ES_PYTHON to a python with torch, numpy and yaml");
        return;
    };
    let mut failed = Vec::new();
    for (kind, adapter) in [
        ("isaac-classic", "adapter-isaac-reach.toml"),
        ("isaac-split", "adapter-isaac-reach.toml"),
        ("playground", "adapter-playground-reach.toml"),
    ] {
        let worst = max_error(&py, kind, adapter, ("[robot]", "[robot]"));
        if worst.is_nan() || worst > TOLERANCE {
            failed.push(format!("{kind}: {worst:.3e}"));
        }
    }
    assert!(failed.is_empty(), "over {TOLERANCE:.0e}: {failed:?}");

    // The negative control: the same Isaac source with ONE convention left undeclared -- the
    // joint velocity's `scale = 0.05` -- must miss by far more than the tolerance, or the
    // comparison above could be passing on a function that ignores the observation.
    let undeclared = max_error(
        &py,
        "isaac-classic",
        "adapter-isaac-reach.toml",
        ("scale = 0.05\n", ""),
    );
    println!("negative control (joint_vel scale undeclared): max abs error {undeclared:.3e}");
    assert!(
        undeclared > 1e-3,
        "an undeclared scale went unnoticed: {undeclared:.3e}"
    );
}
