//! The M5 V2 training oracle: the three-step loop `es policy lower` -> `train_act.py` ->
//! `es policy pack`, judged by the four executable facts of `docs/design/visible-learning.md`
//! section 6.3.
//!
//! Spec 1.4 cannot judge training by "matches a reference" — two runs of the same optimizer on
//! the same CPU do not agree bitwise across `torch` versions (design note section 9). So none
//! of the facts below is a human reading a curve:
//!
//! 1. **the module is the IR's** — the `es_policy.py` the optimizer loaded is byte-equal to
//!    `lower_to_torch(&bundle.learning)?.source`, and `train_act.py` defines no layer;
//! 2. **the weights fit the contract** — `pack` refuses a missing key, an unknown key and a
//!    disagreeing shape, naming each one, with exit code 1;
//! 3. **it learns something** — on a fixed dataset and a fixed seed, the final loss is below a
//!    pinned fraction of the initial loss;
//! 4. **it round-trips** — the packed bundle opens, `TorchRuntime::load` accepts it (which
//!    means it went through `lower_to_torch`, not `load_lowered`) and `infer` returns a finite
//!    chunk. That `es eval run --policy` gets past the *same* load is asserted one layer up, in
//!    `crates/es/tests/cli.rs`, where the Evaluation IR fixture lives.
//!
//! Facts 1 and 2 need no Python and run in the PR tier. Facts 3 and 4 are `#[ignore]`d and need
//! `torch`:
//!
//! ```text
//! ES_PYTHON=$HOME/venvs/es-lerobot/bin/python \
//!   cargo test -p es-policy --test ir_training -- --ignored --nocapture
//! ```
//!
//! A missing `torch` is a `SKIP` with its reason. A *failure inside* `PyTorch` is not: the
//! interpreter is probed once, up front, and every error after that probe is reported as an
//! error (`docs/reviews/M4.md:67`, S-7 — the thing `act_checkpoint.rs:113` got wrong).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use es_compile::{PolicyBundle, Tensor};
use es_ir::learning::WeightsRef;
use es_ir::types::ElemType;
use es_policy::lower::{lower_to_torch, Contract};
use es_policy::weights::{write_safetensors, Checkpoint};
use es_policy::{PolicyRuntime, Tolerance, TorchRuntime, WeightsSource};

/// Final training loss must be at most this fraction of the initial loss. A threshold, not a
/// golden: lowering it to make a change pass is the same as editing a golden file.
const LOSS_MUST_FALL_TO: f64 = 0.7;

/// How many optimizer steps the oracle's own run takes. Small on purpose — this measures that
/// the loop descends, not how well it ends up (spec 12.4: no training time is claimed).
const ORACLE_STEPS: usize = 40;

// --- the demo documents, the bundle, and the binary -----------------------------------------

fn vl_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning")
        .join(name)
}

/// The demo scene, for `es dataset bake --scene` (the bundle's `scene.path` is relative to the
/// repository root, which is not this test's working directory).
fn demo_scene() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf/so101_pick_place.xml")
}

/// `es`, next to this test's own executable.
///
/// `CARGO_BIN_EXE_es` is only defined inside `es`'s own integration tests, and spec 4.2 forbids
/// `es-policy` (layer 8) from depending on the top-level binary (layer 12) to get it. Cargo
/// puts both under the same profile directory, so the binary is one level up from `deps/`.
fn es_bin() -> PathBuf {
    let mut dir = std::env::current_exe().expect("the test has an executable path");
    dir.pop();
    if dir.file_name().is_some_and(|n| n == "deps") {
        dir.pop();
    }
    let exe = dir.join(format!("es{}", std::env::consts::EXE_SUFFIX));
    assert!(
        exe.is_file(),
        "{} is not built; run `cargo build -p es` (or `cargo test --workspace`, which is what \
         `cargo xtask ci` does) before this test",
        exe.display()
    );
    exe
}

fn scratch_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("es-ir-training-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// The plan V demo bundle: V0's four documents with a placeholder checkpoint, exactly as
/// `crates/es/tests/cli.rs` builds it for the V1 expert.
fn demo_bundle(dir: &Path) -> (PathBuf, PolicyBundle) {
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let mut learning =
        es_ir::serial::learning_from_toml(&read("learning.toml")).expect("learning.toml");
    let weights = b"es-v2-untrained-placeholder".to_vec();
    learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = PolicyBundle::build(
        &es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the four demo documents pack into a bundle");
    let path = dir.join("untrained.esb");
    std::fs::write(&path, &bytes).expect("write untrained.esb");
    let opened = PolicyBundle::open(&bytes).expect("the bundle just written opens");
    (path, opened)
}

fn es(args: &[&str]) -> std::process::Output {
    Command::new(es_bin())
        .args(args)
        .output()
        .expect("run the es binary")
}

fn text(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// `es policy lower` into a fresh directory, returning it and the parsed contract.
fn lowered(dir: &Path, bundle: &Path) -> (PathBuf, Contract) {
    let build = dir.join("build");
    let out = es(&[
        "policy",
        "lower",
        "--policy",
        &bundle.to_string_lossy(),
        "--out",
        &build.to_string_lossy(),
    ]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out));
    let contract: Contract =
        serde_json::from_str(&std::fs::read_to_string(build.join("contract.json")).expect("read"))
            .expect("contract.json parses back into the type that wrote it");
    (build, contract)
}

/// A checkpoint that fits the contract exactly: every declared shape, plus one tensor standing
/// in for each opaque sub-module a prefix claim covers.
fn conforming(contract: &Contract) -> Checkpoint {
    let mut file: Checkpoint = contract
        .weight_shapes
        .iter()
        .map(|(k, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            (k.clone(), (shape.clone(), vec![0.01f32; n]))
        })
        .collect();
    for claim in contract.weight_keys.iter().filter(|k| k.ends_with(".*")) {
        let prefix = claim.trim_end_matches('*');
        file.insert(format!("{prefix}conv1.weight"), (vec![2], vec![1.0, 2.0]));
    }
    file
}

/// `es policy pack` with `file` as the checkpoint. Returns (exit code, combined output).
fn pack(dir: &Path, bundle: &Path, tag: &str, file: &Checkpoint) -> (Option<i32>, String) {
    let weights = dir.join(format!("{tag}.safetensors"));
    std::fs::write(&weights, write_safetensors(file)).expect("write the checkpoint");
    let out = es(&[
        "policy",
        "pack",
        "--policy",
        &bundle.to_string_lossy(),
        "--weights",
        &weights.to_string_lossy(),
        "--out",
        &dir.join(format!("{tag}.esb")).to_string_lossy(),
    ]);
    (out.status.code(), text(&out))
}

// --- fact 1: the module is the IR's ---------------------------------------------------------

#[test]
fn the_trained_module_is_the_lowering() {
    let dir = scratch_dir("module-identity");
    let (path, bundle) = demo_bundle(&dir);
    let (build, contract) = lowered(&dir, &path);

    // The whole reason this packet exists: what PyTorch optimizes is byte-for-byte what
    // `TorchRuntime::load` re-lowers and infers with (design note section 2.5).
    let module = lower_to_torch(&bundle.learning).expect("the demo graph lowers");
    let written = std::fs::read_to_string(build.join("es_policy.py")).expect("es_policy.py");
    assert_eq!(written, module.source, "es policy lower rewrote the module");
    assert!(written.contains("class EsPolicy(nn.Module):"), "{written}");

    assert_eq!(contract, Contract::new(&module, &bundle.learning));
    assert_eq!(contract.action_dim, 6);
    assert_eq!(contract.horizon, 16);
    assert_eq!(
        contract.inputs["rgb_overhead"],
        vec![3, 96, 96],
        "the contract must carry the shape `forward` expects"
    );
    println!(
        "RAN module_identity: {} bytes of module, {} weight keys, lowering_hash {}",
        written.len(),
        contract.weight_keys.len(),
        contract.lowering_hash
    );
}

#[test]
fn train_act_defines_no_layer() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/es/train_act.py");
    let source = std::fs::read_to_string(&script).expect("python/es/train_act.py");
    for forbidden in [
        "nn.Linear",
        "nn.Conv",
        "nn.Transformer",
        "nn.Sequential",
        "nn.Module)",
        "torchvision",
        "torch.load",
        "pickle",
        // Packet M5/V2b: no Observation IR node has a second implementation here. `permute` and
        // `/ 255` were `Op::Dequantize` by hand, and `pyarrow` was the raw dataset read that
        // bypassed the plan. The observation comes from `es dataset bake` or it does not come.
        "permute(",
        "/ 255",
        "pyarrow",
    ] {
        assert!(
            !source.contains(forbidden),
            "python/es/train_act.py contains `{forbidden}`: the architecture comes from the \
             lowering and the observation from the Observation IR, or they do not come (and \
             weights are safetensors only, INV-16)"
        );
    }
    println!("RAN no_layer: {} lines scanned", source.lines().count());
}

// --- fact 2: the weights fit the contract ---------------------------------------------------

#[test]
fn pack_accepts_a_conforming_checkpoint() {
    let dir = scratch_dir("pack-ok");
    let (path, _) = demo_bundle(&dir);
    let (_, contract) = lowered(&dir, &path);
    let (code, out) = pack(&dir, &path, "ok", &conforming(&contract));
    assert_eq!(code, Some(0), "{out}");
    assert!(out.contains("weights_hash:"), "{out}");
}

#[test]
fn pack_refuses_a_missing_key() {
    let dir = scratch_dir("pack-missing");
    let (path, _) = demo_bundle(&dir);
    let (_, contract) = lowered(&dir, &path);
    let mut file = conforming(&contract);
    file.remove("nodes.4.bias");
    let (code, out) = pack(&dir, &path, "missing", &file);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("missing key    nodes.4.bias"), "{out}");
    assert!(
        !dir.join("missing.esb").exists(),
        "a refusal wrote a bundle"
    );
}

#[test]
fn pack_refuses_an_extra_key() {
    let dir = scratch_dir("pack-extra");
    let (path, _) = demo_bundle(&dir);
    let (_, contract) = lowered(&dir, &path);
    let mut file = conforming(&contract);
    file.insert("nodes.9.weight".to_owned(), (vec![1], vec![0.0]));
    let (code, out) = pack(&dir, &path, "extra", &file);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("unexpected key nodes.9.weight"), "{out}");
    assert!(!dir.join("extra.esb").exists(), "a refusal wrote a bundle");
}

#[test]
fn pack_refuses_a_wrong_shape() {
    let dir = scratch_dir("pack-shape");
    let (path, _) = demo_bundle(&dir);
    let (_, contract) = lowered(&dir, &path);
    let mut file = conforming(&contract);
    file.insert("nodes.4.weight".to_owned(), (vec![95, 512], vec![0.0; 1]));
    let (code, out) = pack(&dir, &path, "shape", &file);
    assert_eq!(code, Some(1), "{out}");
    assert!(
        out.contains("wrong shape    nodes.4.weight: want [96, 512], got [95, 512]"),
        "{out}"
    );
    assert!(!dir.join("shape.esb").exists(), "a refusal wrote a bundle");
}

#[test]
fn pack_recomputes_the_manifest_rather_than_patching_it() {
    let dir = scratch_dir("pack-manifest");
    let (path, before) = demo_bundle(&dir);
    let (_, contract) = lowered(&dir, &path);
    let (code, out) = pack(&dir, &path, "packed", &conforming(&contract));
    assert_eq!(code, Some(0), "{out}");

    let after = PolicyBundle::open(&std::fs::read(dir.join("packed.esb")).expect("read"))
        .expect("the packed bundle opens, which re-checks every hash in it");
    assert_eq!(after.task, before.task);
    assert_eq!(after.observation, before.observation);
    assert_eq!(after.deployment, before.deployment);
    assert_ne!(after.weights, before.weights, "the weights did not change");
    // Only the weights reference moved inside the Learning IR, and it moved because the bytes
    // did (spec 5.3). `policy_hash` is the slot that covers `WeightsRef`
    // (`es_ir::learning::LearningGraph::policy_hash`), so that is the one that moves;
    // `learning_hash` covers the graph, which is untouched, and neither the task nor the
    // observation slot has any business moving because a checkpoint did.
    assert_ne!(
        after.learning.policy.weights.hash(),
        before.learning.policy.weights.hash()
    );
    assert_eq!(
        after.learning.policy.weights.path(),
        before.learning.policy.weights.path()
    );
    assert_eq!(after.learning.nodes, before.learning.nodes);
    assert_ne!(after.manifest.hashes.policy, before.manifest.hashes.policy);
    assert_eq!(
        after.manifest.hashes.learning,
        before.manifest.hashes.learning
    );
    assert_eq!(after.manifest.hashes.task, before.manifest.hashes.task);
    assert_eq!(
        after.manifest.hashes.observation,
        before.manifest.hashes.observation
    );
}

// --- facts 3 and 4: torch ---------------------------------------------------------------

/// The one probe that separates "no `torch` here" from "`PyTorch` failed". Everything after it is
/// an error, never a skip (`docs/reviews/M4.md:67`, S-7).
fn python_with_torch() -> Result<String, String> {
    let mut tried = Vec::new();
    let candidates = match std::env::var("ES_PYTHON") {
        Ok(p) if !p.trim().is_empty() => vec![p],
        _ => vec!["python".to_owned(), "python3".to_owned()],
    };
    for python in candidates {
        match Command::new(&python)
            .args(["-c", "import torch, torchvision, pyarrow"])
            .output()
        {
            Ok(out) if out.status.success() => return Ok(python),
            Ok(out) => tried.push(format!(
                "`{python}`: {}",
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .last()
                    .unwrap_or("import failed")
                    .trim()
                    .to_owned()
            )),
            Err(e) => tried.push(format!("`{python}`: {e}")),
        }
    }
    Err(format!(
        "no Python with `torch`, `torchvision` and `pyarrow` (set ES_PYTHON to choose one): {}",
        tried.join("; ")
    ))
}

/// A `LeRobot` v2.1 dataset of two short episodes plus the flat `<NNNNNN>.bin` tiles
/// `es loop collect --frames` writes beside one, in the layout
/// `docs/api-notes/lerobot-dataset.md` records: `meta/` plus one 3-level-list parquet per
/// episode. argv is `<dataset root> <tiles dir>`.
///
/// Written through `pyarrow` rather than `es-data`, because `es-data` is layer 10 and this
/// crate is layer 8 (spec 4.2). The rows are a ramp, so there is something for L1 to fit, and
/// the tiles are a per-frame pattern so the image branch is not a constant either.
const MINI_DATASET_PY: &str = r#"
import json, os, sys
import pyarrow as pa, pyarrow.parquet as pq
root, tiles, episodes, frames, state_w, action_w = sys.argv[1], sys.argv[2], 2, 24, 25, 6
tile_bytes = 96 * 96 * 3
os.makedirs(os.path.join(root, "meta"), exist_ok=True)
os.makedirs(os.path.join(root, "data", "chunk-000"), exist_ok=True)
os.makedirs(tiles, exist_ok=True)
feature = lambda w: {"dtype": "float32", "shape": [w]}
info = {
    "codebase_version": "v2.1", "robot_type": "es", "total_episodes": episodes,
    "total_frames": episodes * frames, "total_tasks": 1, "total_videos": 0,
    "total_chunks": 1, "chunks_size": 1000, "fps": 50.0, "splits": {"train": "0:%d" % episodes},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": None,
    "features": {"observation.state": feature(state_w), "action": feature(action_w)},
}
json.dump(info, open(os.path.join(root, "meta", "info.json"), "w"))
with open(os.path.join(root, "meta", "episodes.jsonl"), "w") as h:
    for e in range(episodes):
        h.write(json.dumps({"episode_index": e, "tasks": ["es:v2:oracle"], "length": frames}) + "\n")
with open(os.path.join(root, "meta", "tasks.jsonl"), "w") as h:
    h.write(json.dumps({"task_index": 0, "task": "es:v2:oracle"}) + "\n")
for e in range(episodes):
    state, action = [], []
    for t in range(frames):
        phase = (t + 4 * e) / float(frames)
        state.append([0.30 * phase * (1 + (i % 3)) for i in range(state_w)])
        action.append([0.30 * phase * (1 + (i % 3)) for i in range(action_w)])
    column = lambda rows: pa.array(rows, type=pa.list_(pa.float32()))
    # The five reserved scalar columns as well as the features: `es dataset bake` reads this
    # through `es_data::LeRobotDataset`, which refuses a file with no `timestamp`
    # (crates/es-data/src/lerobot/columns.rs). `compression="NONE"` for the same reason --
    # es-data enables no parquet codec on purpose, and pyarrow defaults to snappy.
    table = pa.table({
        "observation.state": column(state),
        "action": column(action),
        "timestamp": pa.array([t / 50.0 for t in range(frames)], type=pa.float64()),
        "frame_index": pa.array(list(range(frames)), type=pa.int64()),
        "episode_index": pa.array([e] * frames, type=pa.int64()),
        "index": pa.array([e * frames + t for t in range(frames)], type=pa.int64()),
        "task_index": pa.array([0] * frames, type=pa.int64()),
    })
    pq.write_table(
        table,
        os.path.join(root, "data", "chunk-000", "episode_%06d.parquet" % e),
        compression="NONE",
    )
for i in range(episodes * frames):
    tile = bytes(((j + 13 * i) % 256) for j in range(tile_bytes))
    open(os.path.join(tiles, "%06d.bin" % i), "wb").write(tile)
"#;

fn run(python: &str, args: &[&str]) -> std::process::Output {
    let out = Command::new(python)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("run {python} {args:?}: {e}"));
    assert!(
        out.status.success(),
        "{python} {args:?} failed with {:?}\n{}",
        out.status.code(),
        text(&out)
    );
    out
}

fn train_act_py() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/es/train_act.py")
}

/// Facts 3 and 4 in one run, because fact 4 needs fact 3's checkpoint: bake a tiny fixed
/// dataset through the bundle's own Observation IR, train on **that** with a fixed seed,
/// require the loss to fall, pack the result, and put it through `TorchRuntime`.
///
/// The bake is not a step in the way; it is what makes the run honest (packet M5/V2b). V2
/// trained on the raw `observation.state` row and a Python copy of `Op::Dequantize`, so the
/// module saw one observation in training and another at inference. What `train_act.py` reads
/// here is byte-for-byte what `capture` computes, which `es-eval`'s
/// `a_baked_frame_is_bit_identical_to_what_capture_serves` pins without an interpreter.
#[test]
#[ignore = "needs torch, torchvision and pyarrow"]
fn act_training_uses_baked_observations() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP ir_training: {why}");
            return;
        }
    };
    let dir = scratch_dir("train");
    let (path, bundle) = demo_bundle(&dir);
    let (build, contract) = lowered(&dir, &path);

    let (dataset, tiles, baked) = (dir.join("ds"), dir.join("tiles"), dir.join("baked"));
    run(
        &python,
        &[
            "-c",
            MINI_DATASET_PY,
            &dataset.to_string_lossy(),
            &tiles.to_string_lossy(),
        ],
    );

    // --- the observation, exactly once, in Rust ------------------------------------------
    let bake = es(&[
        "dataset",
        "bake",
        "--policy",
        &path.to_string_lossy(),
        "--out",
        &baked.to_string_lossy(),
        "--frames",
        &tiles.to_string_lossy(),
        // The demo's second JointState channel sends the bake to the scene (packet M5/V7a),
        // and the Task IR's `scene.path` is repository-relative while this test does not run
        // from the root: name the file the way `es eval run --scene` does.
        "--scene",
        &demo_scene().to_string_lossy(),
        &dataset.to_string_lossy(),
    ]);
    assert_eq!(bake.status.code(), Some(0), "{}", text(&bake));

    // --- fact 3: it learns something ---------------------------------------------------
    let weights = dir.join("model.safetensors");
    let out = run(
        &python,
        &[
            &train_act_py().to_string_lossy(),
            "--module",
            &build.to_string_lossy(),
            "--baked",
            &baked.to_string_lossy(),
            "--out",
            &weights.to_string_lossy(),
            "--batch",
            "4",
            "--seed",
            "0",
            "--checkpoint-at",
            &ORACLE_STEPS.to_string(),
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().last().unwrap_or_default();
    let report: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(line).unwrap_or_else(|e| panic!("{e} in `{line}`"));
    let number = |key: &str| {
        report[key]
            .as_f64()
            .unwrap_or_else(|| panic!("{key}: {line}"))
    };
    let (initial, final_) = (number("initial_loss"), number("final_loss"));
    assert!(
        final_ <= LOSS_MUST_FALL_TO * initial,
        "the loss did not fall: {initial} -> {final_}, which is more than {LOSS_MUST_FALL_TO} \
         of the initial loss. Lowering the threshold is editing a golden."
    );
    assert_eq!(report["steps"].as_u64(), Some(ORACLE_STEPS as u64));

    // --- fact 2 again, on a checkpoint the optimizer actually wrote --------------------
    let packed = dir.join("trained.esb");
    let pack_out = es(&[
        "policy",
        "pack",
        "--policy",
        &path.to_string_lossy(),
        "--weights",
        &weights.to_string_lossy(),
        "--out",
        &packed.to_string_lossy(),
    ]);
    assert_eq!(pack_out.status.code(), Some(0), "{}", text(&pack_out));

    // --- fact 4: it round-trips --------------------------------------------------------
    let bytes = std::fs::read(&packed).expect("read trained.esb");
    let trained = PolicyBundle::open(&bytes).expect("the trained bundle opens");
    let mut runtime = TorchRuntime::new();
    // `load`, not `load_lowered`: this goes through `lower_to_torch` on the bundle's own
    // Learning IR, which is the assertion a `lower_act` bundle would fail (design note 2.5).
    let info = runtime
        .load(
            &trained.learning,
            &WeightsSource::InMemory(trained.weights.clone()),
        )
        .expect("TorchRuntime::load accepts the packed bundle");
    assert_eq!(info.action_dim, contract.action_dim);

    // A held-out observation, written to disk so both sides read the same numbers rather than
    // recomputing them from two copies of one formula.
    //
    // Every port gets real values now. V2 had to zero the image port here because the dataset
    // carried no pixels, so every `BatchNorm2d` running variance in the backbone had collapsed
    // towards zero and a non-zero image at inference turned into `inf` and then `NaN`. The bake
    // feeds the image port the tiles, so there is no port left that saw only zeros in training
    // and no reason to hand inference one.
    let held_out: BTreeMap<String, Vec<f32>> = contract
        .inputs
        .iter()
        .map(|(port, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            let values = (0..n).map(|i| (i % 23) as f32 / 23.0).collect();
            (port.clone(), values)
        })
        .collect();
    // The bake is what the module trained on, and the report says which plan produced it.
    assert_eq!(
        report["ports"].as_array().map(Vec::len),
        Some(contract.inputs.len()),
        "training fed a different set of ports than the contract declares: {line}"
    );
    assert!(
        report["observation_hash"]
            .as_str()
            .is_some_and(|h| h.len() == 64),
        "the baked manifest carried no observation_hash: {line}"
    );
    let observation = dir.join("observation.json");
    std::fs::write(
        &observation,
        serde_json::to_string(&held_out).expect("serialize the observation"),
    )
    .expect("write observation.json");

    let inputs: BTreeMap<String, Tensor> = contract
        .inputs
        .iter()
        .map(|(port, shape)| {
            (
                port.clone(),
                Tensor {
                    dtype: ElemType::F32,
                    shape: shape.clone(),
                    data: held_out[port]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                },
            )
        })
        .collect();
    let outputs = runtime
        .infer(&inputs)
        .expect("infer on a held-out observation");
    let chunk = &outputs["actions"];
    let execute = u64::from(bundle.learning.policy.contract.execute_chunk);
    assert_eq!(chunk.shape, vec![execute, u64::from(contract.action_dim)]);
    let values: Vec<f32> = chunk
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()), "{values:?}");

    // ...and it is the *same* chunk the optimizer's own process would produce. This is what
    // makes "the packed bundle round-trips" a number rather than a shape check: it covers the
    // safetensors writer, `pack`'s validation, the `nodes.N` -> `nN` rename and the wire
    // protocol, all at once. Spec 8.9's tier-4 fp32 tolerance is the pinned one.
    let direct = run(
        &python,
        &[
            "-c",
            REFERENCE_FORWARD_PY,
            &build.to_string_lossy(),
            &weights.to_string_lossy(),
            &observation.to_string_lossy(),
        ],
    );
    let flat: Vec<f32> = serde_json::from_slice(&direct.stdout)
        .unwrap_or_else(|e| panic!("{e} in `{}`", String::from_utf8_lossy(&direct.stdout)));
    let reference = Tensor {
        dtype: ElemType::F32,
        shape: chunk.shape.clone(),
        data: flat.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    let equivalence = es_policy::compare_actions(chunk, &reference, Tolerance::TIER4_FP32);
    assert!(
        equivalence.pass,
        "the Rust runtime and a direct PyTorch forward disagree: {equivalence:?}"
    );

    println!(
        "RAN act_training_uses_baked_observations: loss {initial:.4} -> {final_:.4} over \
         {ORACLE_STEPS} steps ({:.3}x), chunk {:?}, max_abs vs a direct forward {:e} \
         (tol {:e}), observation_hash {}, torch {}",
        final_ / initial,
        chunk.shape,
        equivalence.max_abs,
        Tolerance::TIER4_FP32.abs,
        report["observation_hash"].as_str().unwrap_or("?"),
        info.version
    );
}

/// Packet M5/V5. `--resident-gpu` moves the baked set onto the device once instead of copying
/// one sample per forward. It must move **where the tensors live and nothing else**: same batch
/// order, same dtype, same arithmetic, so the loss curve is bit-identical at the same `--seed`.
///
/// Run here on the CPU, where `.to(device)` is a no-op either way — which is exactly the point.
/// If the resident path ever changed the dtype, the sample order or the accumulation, this
/// would catch it without a GPU, and the two `--loss-curve` files are compared as **bytes**,
/// not with a tolerance.
///
/// `--amp bf16` and `--compile` are deliberately not in this comparison: they change the bits,
/// which is why they are opt-in. The gate that stays either way is
/// `act_training_uses_baked_observations`' tier-4 fp32 round-trip, at the defaults.
#[test]
#[ignore = "needs torch, torchvision and pyarrow"]
fn resident_gpu_does_not_move_the_loss() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP ir_training: {why}");
            return;
        }
    };
    let dir = scratch_dir("resident");
    let (path, _bundle) = demo_bundle(&dir);
    let (build, _contract) = lowered(&dir, &path);

    let baked = mini_baked(&python, &dir, &path);

    let train = |tag: &str, resident: bool| -> (Vec<u8>, String) {
        let curve = dir.join(format!("{tag}.json"));
        let out = dir.join(format!("{tag}.safetensors"));
        let mut args = vec![
            train_act_py().to_string_lossy().into_owned(),
            "--module".to_owned(),
            build.to_string_lossy().into_owned(),
            "--baked".to_owned(),
            baked.to_string_lossy().into_owned(),
            "--out".to_owned(),
            out.to_string_lossy().into_owned(),
            "--batch".to_owned(),
            "4".to_owned(),
            "--seed".to_owned(),
            "0".to_owned(),
            "--checkpoint-at".to_owned(),
            ORACLE_STEPS.to_string(),
            "--loss-curve".to_owned(),
            curve.to_string_lossy().into_owned(),
        ];
        if resident {
            args.push("--resident-gpu".to_owned());
        }
        let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
        let done = run(&python, &borrowed);
        let line = String::from_utf8_lossy(&done.stdout)
            .lines()
            .last()
            .unwrap_or_default()
            .to_owned();
        (
            std::fs::read(&curve).unwrap_or_else(|e| panic!("{}: {e}", curve.display())),
            line,
        )
    };

    let (plain, plain_report) = train("plain", false);
    let (resident, resident_report) = train("resident", true);
    assert_eq!(
        String::from_utf8_lossy(&resident),
        String::from_utf8_lossy(&plain),
        "--resident-gpu moved the loss; it may only move where the tensors live"
    );
    // ...and the curve has to be over something: 40 steps that are not all the same number.
    let losses: Vec<f64> = serde_json::from_slice(&plain).expect("the loss curve is JSON");
    assert_eq!(losses.len(), ORACLE_STEPS, "{plain_report}");
    assert!(
        losses.iter().any(|v| (v - losses[0]).abs() > 0.0),
        "the loss never moved at all: {losses:?}"
    );
    assert!(
        resident_report.contains("\"resident_gpu\": true"),
        "the report must record the mode it ran in: {resident_report}"
    );
    println!(
        "RAN resident_gpu_does_not_move_the_loss: {ORACLE_STEPS} bit-identical steps, \
         {} bytes of curve\n  plain:    {plain_report}\n  resident: {resident_report}",
        plain.len()
    );
}

/// `MINI_DATASET_PY` through `es dataset bake`: the few-frame baked set both training oracles
/// optimize over. Shared because building it twice is the expensive half of either test.
fn mini_baked(python: &str, dir: &Path, bundle: &Path) -> PathBuf {
    let (dataset, tiles, baked) = (dir.join("ds"), dir.join("tiles"), dir.join("baked"));
    run(
        python,
        &[
            "-c",
            MINI_DATASET_PY,
            &dataset.to_string_lossy(),
            &tiles.to_string_lossy(),
        ],
    );
    let bake = es(&[
        "dataset",
        "bake",
        "--policy",
        &bundle.to_string_lossy(),
        "--out",
        &baked.to_string_lossy(),
        "--frames",
        &tiles.to_string_lossy(),
        // The demo's second JointState channel sends the bake to the scene (packet M5/V7a),
        // and the Task IR's `scene.path` is repository-relative while this test does not run
        // from the root: name the file the way `es eval run --scene` does.
        "--scene",
        &demo_scene().to_string_lossy(),
        &dataset.to_string_lossy(),
    ]);
    assert_eq!(bake.status.code(), Some(0), "{}", text(&bake));
    baked
}

/// `--channel-weight` (packet M5/V16) scales one action channel's absolute error and nothing
/// else about the run.
///
/// Two properties, and the first is the one that protects every number already measured: an
/// **all-ones** weighting must be bit-identical to no weighting, because `(|d| * 1).mean()` is
/// `l1_loss`. If it ever is not, every loss curve in design note section 7 was taken under a
/// different objective than the one the script now runs. The second is that a weight that is
/// not one *does* move the curve, and that the run summary records the vector it ran under --
/// the flag enters no hash slot (spec 8.1), so the summary is the only place it is written
/// down.
#[test]
#[ignore = "needs torch, torchvision and pyarrow"]
fn channel_weight_of_one_is_the_unweighted_loss() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP ir_training: {why}");
            return;
        }
    };
    let dir = scratch_dir("chanweight");
    let (path, _bundle) = demo_bundle(&dir);
    let (build, _contract) = lowered(&dir, &path);
    let baked = mini_baked(&python, &dir, &path);

    let train = |tag: &str, weight: Option<&str>| -> (Vec<u8>, String) {
        let curve = dir.join(format!("{tag}.json"));
        let out = dir.join(format!("{tag}.safetensors"));
        let mut args = vec![
            train_act_py().to_string_lossy().into_owned(),
            "--module".to_owned(),
            build.to_string_lossy().into_owned(),
            "--baked".to_owned(),
            baked.to_string_lossy().into_owned(),
            "--out".to_owned(),
            out.to_string_lossy().into_owned(),
            "--batch".to_owned(),
            "4".to_owned(),
            "--seed".to_owned(),
            "0".to_owned(),
            "--checkpoint-at".to_owned(),
            ORACLE_STEPS.to_string(),
            "--loss-curve".to_owned(),
            curve.to_string_lossy().into_owned(),
        ];
        if let Some(w) = weight {
            args.push("--channel-weight".to_owned());
            args.push(w.to_owned());
        }
        let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
        let done = run(&python, &borrowed);
        let line = String::from_utf8_lossy(&done.stdout)
            .lines()
            .last()
            .unwrap_or_default()
            .to_owned();
        (
            std::fs::read(&curve).unwrap_or_else(|e| panic!("{}: {e}", curve.display())),
            line,
        )
    };

    let (plain, plain_report) = train("plain", None);
    let (ones, _) = train("ones", Some("5=1"));
    assert_eq!(
        String::from_utf8_lossy(&ones),
        String::from_utf8_lossy(&plain),
        "--channel-weight 5=1 moved the loss; an all-ones weighting must be `l1_loss`"
    );
    let (five, five_report) = train("five", Some("5=5"));
    assert_ne!(
        String::from_utf8_lossy(&five),
        String::from_utf8_lossy(&plain),
        "--channel-weight 5=5 did not move the loss at all"
    );
    assert!(
        five_report.contains("\"channel_weight\": [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]"),
        "the report must record the weighting it ran under: {five_report}"
    );
    assert!(
        plain_report.contains("\"channel_weight\": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"),
        "an unweighted run records all ones: {plain_report}"
    );
    println!(
        "RAN channel_weight_of_one_is_the_unweighted_loss: {ORACLE_STEPS} steps\n  \
         plain: {plain_report}\n  five:  {five_report}"
    );
}

/// `EsPolicy` in a plain interpreter, with no `TorchRuntime` between it and the checkpoint:
/// argv is `<module dir> <model.safetensors> <observation.json>` and stdout is the flattened
/// action chunk as JSON. Deliberately not importing `torch_ref.py` — the point is to be a
/// second implementation, not the same one twice.
const REFERENCE_FORWARD_PY: &str = r#"
import json, struct, sys
import torch
module_dir, weights_path, observation_path = sys.argv[1], sys.argv[2], sys.argv[3]
namespace = {}
exec(compile(open(module_dir + "/es_policy.py").read(), "<es-policy>", "exec"), namespace)
model = namespace["EsPolicy"]()
blob = open(weights_path, "rb").read()
size = struct.unpack_from("<Q", blob, 0)[0]
header, base, state = json.loads(blob[8 : 8 + size]), 8 + size, {}
for name, entry in header.items():
    if name == "__metadata__":
        continue
    a, b = entry["data_offsets"]
    raw = blob[base + a : base + b]
    values = struct.unpack("<%df" % (len(raw) // 4), raw)
    head, _, tail = name[len("nodes.") :].partition(".")
    state["n" + head + ("." + tail if tail else "")] = torch.tensor(
        values, dtype=torch.float32
    ).reshape(entry["shape"])
model.load_state_dict(state, strict=True)
model.eval()
shapes = json.load(open(module_dir + "/contract.json"))["inputs"]
observation = json.load(open(observation_path))
inputs = {
    port: torch.tensor(values, dtype=torch.float32).reshape([1] + shapes[port])
    for port, values in observation.items()
}
with torch.inference_mode():
    out = next(iter(model(**inputs).values()))
sys.stdout.write(json.dumps([float(v) for v in out.reshape(-1).tolist()]))
"#;

/// Packet M5/V13, the executable half of the fix: the lowered backbone has **no training
/// mode**. `GroupNorm` normalizes over channel groups of the one sample in front of it, so
/// `train()` and `eval()` are bit-for-bit the same function and no running-statistic buffer
/// exists to drift. `BatchNorm2d` failed both halves: the lowering is single-sample (spec 8.3
/// has no batch axis), so every one of the 20 layers fitted N = 1 statistics during training
/// and then used running ones at inference — the train/deploy gap design note section 7.21
/// measured at 0.011 vs 0.031 chunk L1.
///
/// Dropout inside `nn.TransformerEncoderLayer` is the remaining, intended train/eval
/// difference — the three `nn.Dropout` members plus `MultiheadAttention`'s attention dropout,
/// which is a float read off `self.training` rather than a module — which is why this asserts
/// over the backbone rather than the whole module.
#[test]
#[ignore = "needs torch and torchvision"]
fn the_backbone_computes_the_same_function_in_train_and_eval() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP backbone_train_eval: {why}");
            return;
        }
    };
    let dir = scratch_dir("backbone-train-eval");
    let (path, _) = demo_bundle(&dir);
    let (build, _) = lowered(&dir, &path);

    let script = dir.join("train_eval.py");
    std::fs::write(&script, BACKBONE_TRAIN_EVAL_PY).expect("write the probe");
    let out = run(
        &python,
        &[&script.to_string_lossy(), &build.to_string_lossy()],
    );
    println!("RAN backbone_train_eval: {}", text(&out).trim());
}

/// argv is `<module dir>`; exits non-zero with the offending keys or the deviation.
const BACKBONE_TRAIN_EVAL_PY: &str = r#"
import json, sys
import torch
module_dir = sys.argv[1]
namespace = {}
exec(compile(open(module_dir + "/es_policy.py").read(), "<es-policy>", "exec"), namespace)
model = namespace["EsPolicy"]()

stats = [
    k
    for k in model.state_dict()
    if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
]
assert not stats, "running statistics survived the lowering: %s" % stats[:4]

backbones = [(n, m) for n, m in model.named_children() if type(m).__name__ == "ResNet"]
assert backbones, "the demo graph has a VisionEncoder, so a backbone must be instantiated"

shapes = json.load(open(module_dir + "/contract.json"))["inputs"]
image = [s for s in shapes.values() if len(s) == 3][0]
x = torch.rand([1] + image, generator=torch.Generator().manual_seed(0))
for name, backbone in backbones:
    with torch.no_grad():
        model.train()
        trained = backbone(x)
        model.eval()
        deployed = backbone(x)
    assert torch.equal(trained, deployed), "%s: train() and eval() differ by %g" % (
        name,
        (trained - deployed).abs().max(),
    )
print(
    "%d backbone(s) bit-identical in train() and eval() on %s, %d state_dict keys, "
    "no running statistics" % (len(backbones), image, len(model.state_dict()))
)
"#;

// --- packet M7/T4: the learning-rate schedule ------------------------------------------------

/// `tests/golden/train/lr_warmup_cosine.json`: the first 1,000 values of the demo's own
/// large-batch schedule, generated once by [`generate_lr_golden`] from the script's `lr_at`.
fn lr_golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/train/lr_warmup_cosine.json")
}

/// A re-implementation of `python/es/train_act.py::lr_at`, expression for expression.
///
/// This is what pins the schedule **without** an interpreter: the golden was produced by the
/// script, and this function has to reproduce it bit for bit in `f64`. Keep the two in the
/// same order — `lr * step / warmup`, and `lr_min + (lr - lr_min) * 0.5 * (1 + cos(...))` —
/// because a re-association is a different number in the last bit, which is the whole point.
fn lr_at(step: u32, total: u32, lr: f64, lr_min: f64, warmup: u32) -> f64 {
    if warmup > 0 && step < warmup {
        return lr * f64::from(step) / f64::from(warmup);
    }
    let span = total.saturating_sub(warmup).max(1);
    lr_min
        + (lr - lr_min)
            * 0.5
            * (1.0 + (std::f64::consts::PI * f64::from(step - warmup) / f64::from(span)).cos())
}

/// argv is `<train_act.py> <total> <lr> <lr_min> <warmup> <count>`; stdout is
/// `{"values": [..], "lr_curve_hash": "…" | null}`.
///
/// The script is `exec`'d rather than imported so that this reads the file the repository
/// ships, from wherever the test runs. `__name__` is set because the file ends in the usual
/// `if __name__ == "__main__"` guard.
const LR_PROBE_PY: &str = r#"
import json, sys
namespace = {"__name__": "es_train_act_probe"}
source = open(sys.argv[1], encoding="utf-8").read()
exec(compile(source, sys.argv[1], "exec"), namespace)
total, lr, lr_min, warmup, count = (
    int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
)
values = [namespace["lr_at"](s, total, lr, lr_min, warmup) for s in range(count)]
sys.stdout.write(
    json.dumps({"values": values, "lr_curve_hash": namespace["lr_curve_hash"](values)})
)
"#;

/// Regenerates the lr golden from the script itself. Run once, explicitly, on a machine with
/// `ES_PYTHON`; the file is then read-only (spec 1.4), like every other golden here.
#[test]
#[ignore = "golden generator; run explicitly with ES_PYTHON"]
fn generate_lr_golden() {
    let python = python_with_torch().expect("the generator needs an interpreter");
    let (total, lr, lr_min, warmup, count) = (20000u32, 4e-4, 1e-6, 250u32, 1000usize);
    let probe = run(
        &python,
        &[
            "-c",
            LR_PROBE_PY,
            &train_act_py().to_string_lossy(),
            &total.to_string(),
            &lr.to_string(),
            &lr_min.to_string(),
            &warmup.to_string(),
            &count.to_string(),
        ],
    );
    let reply: serde_json::Value = serde_json::from_slice(&probe.stdout).expect("the probe");
    let golden = serde_json::json!({
        "schema_version": 1,
        "source": "python/es/train_act.py::lr_at",
        "total": total, "lr": lr, "lr_min": lr_min, "warmup": warmup,
        "values": reply["values"],
    });
    std::fs::create_dir_all(lr_golden().parent().expect("a parent")).expect("golden dir");
    std::fs::write(
        lr_golden(),
        serde_json::to_string_pretty(&golden).expect("serialize") + "\n",
    )
    .expect("write the golden");
    println!(
        "RAN generate_lr_golden: {count} values -> {}",
        lr_golden().display()
    );
}

/// Oracle 1 of packet M7/T4. The Rust re-implementation above equals the golden bitwise, and
/// with an interpreter the script's own `lr_at` equals it bitwise too.
///
/// Both halves are exact `f64` comparisons on purpose. A schedule is five numbers and three
/// operations; if two implementations of it disagree in the last bit, one of them has been
/// re-associated, and a training run is then not reproducible from its `scheduler.json`.
#[test]
fn lr_schedule_matches_the_golden() {
    let text = std::fs::read_to_string(lr_golden())
        .unwrap_or_else(|e| panic!("{}: {e}", lr_golden().display()));
    let golden: serde_json::Value = serde_json::from_str(&text).expect("the golden is JSON");
    let number = |key: &str| golden[key].as_f64().unwrap_or_else(|| panic!("{key}"));
    let count = |key: &str| golden[key].as_u64().unwrap_or_else(|| panic!("{key}")) as u32;
    let (total, warmup) = (count("total"), count("warmup"));
    let (lr, lr_min) = (number("lr"), number("lr_min"));
    let want: Vec<f64> = golden["values"]
        .as_array()
        .expect("values")
        .iter()
        .map(|v| v.as_f64().expect("a value"))
        .collect();
    assert!(want.len() >= 1000, "the golden is {} values", want.len());

    let mut differ = Vec::new();
    for (step, want) in want.iter().enumerate() {
        let got = lr_at(step as u32, total, lr, lr_min, warmup);
        if got.to_bits() != want.to_bits() {
            differ.push((step, got, *want));
        }
    }
    assert!(
        differ.is_empty(),
        "the Rust re-implementation is not the golden at {} of {} steps, first {:?}",
        differ.len(),
        want.len(),
        &differ[..differ.len().min(4)]
    );
    // The schedule is what the lowering's own oracle assumes: it starts at 0, peaks at `lr`
    // at the end of the warmup and decays. Cheap, and it catches a golden regenerated from a
    // formula that happens to agree with a broken re-implementation.
    assert_eq!(want[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(want[warmup as usize].to_bits(), lr.to_bits());
    assert!(want[warmup as usize + 1] < lr && want[999] > lr_min);

    match python_with_torch() {
        Err(why) => println!("SKIP the interpreter half of lr_schedule_matches_the_golden: {why}"),
        Ok(python) => {
            let probe = run(
                &python,
                &[
                    "-c",
                    LR_PROBE_PY,
                    &train_act_py().to_string_lossy(),
                    &total.to_string(),
                    &lr.to_string(),
                    &lr_min.to_string(),
                    &warmup.to_string(),
                    &want.len().to_string(),
                ],
            );
            let reply: serde_json::Value =
                serde_json::from_slice(&probe.stdout).expect("the probe replies JSON");
            let from_python: Vec<f64> = reply["values"]
                .as_array()
                .expect("values")
                .iter()
                .map(|v| v.as_f64().expect("a value"))
                .collect();
            assert_eq!(from_python.len(), want.len());
            for (step, (a, b)) in from_python.iter().zip(&want).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "step {step}: the script gives {a:e}, the golden {b:e}"
                );
            }
            // ...and `lr_curve_hash` is blake3 over those same values as little-endian f64,
            // which is what makes the number in a run summary checkable from the outside.
            let hashed: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();
            let here = blake3::hash(&hashed).to_hex().to_string();
            match reply["lr_curve_hash"].as_str() {
                Some(there) => assert_eq!(there, here, "lr_curve_hash disagrees"),
                None => println!("SKIP lr_curve_hash: `blake3` is not importable by {python}"),
            }
            println!("RAN lr_schedule_matches_the_golden: interpreter half too, {here}");
        }
    }
    println!(
        "RAN lr_schedule_matches_the_golden: {} values bitwise, warmup {warmup} of {total}, \
         lr {lr:e} -> {lr_min:e}",
        want.len()
    );
}

/// Oracle 2 of packet M7/T4. `--schedule constant` is the default **and** it is the run of
/// before: passing every new flag at its documented default gives a byte-identical loss
/// curve, the optimizer block is the one the measured runs were taken under, and the applied
/// learning rate never moves off `--lr`.
///
/// What this cannot check from inside the repository is equality with the *previous* script,
/// because that file is not in the tree. That comparison is the server measurement in
/// `docs/design/training-recipe.md` section 10: `git archive main` into a scratch directory,
/// the same 40 steps, and the two curves compared as bytes.
#[test]
#[ignore = "needs torch, torchvision and pyarrow"]
fn the_default_schedule_is_the_old_run() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP the_default_schedule_is_the_old_run: {why}");
            return;
        }
    };
    let dir = scratch_dir("default-schedule");
    let (path, _bundle) = demo_bundle(&dir);
    let (build, _contract) = lowered(&dir, &path);
    let baked = mini_baked(&python, &dir, &path);

    let train = |tag: &str, extra: &[&str]| -> (Vec<u8>, serde_json::Value) {
        let curve = dir.join(format!("{tag}.json"));
        let out = dir.join(format!("{tag}.safetensors"));
        let mut args = vec![
            train_act_py().to_string_lossy().into_owned(),
            "--module".to_owned(),
            build.to_string_lossy().into_owned(),
            "--baked".to_owned(),
            baked.to_string_lossy().into_owned(),
            "--out".to_owned(),
            out.to_string_lossy().into_owned(),
            "--batch".to_owned(),
            "4".to_owned(),
            "--seed".to_owned(),
            "0".to_owned(),
            "--checkpoint-at".to_owned(),
            ORACLE_STEPS.to_string(),
            "--loss-curve".to_owned(),
            curve.to_string_lossy().into_owned(),
        ];
        args.extend(extra.iter().map(|w| (*w).to_owned()));
        let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
        let done = run(&python, &borrowed);
        let line = String::from_utf8_lossy(&done.stdout)
            .lines()
            .last()
            .unwrap_or_default()
            .to_owned();
        (
            std::fs::read(&curve).unwrap_or_else(|e| panic!("{}: {e}", curve.display())),
            serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e} in `{line}`")),
        )
    };

    let (default, report) = train("default", &[]);
    let (explicit, _) = train(
        "explicit",
        &[
            "--schedule",
            "constant",
            "--warmup-steps",
            "0",
            "--lr-min",
            "0",
            "--weight-decay",
            "0.01",
            "--grad-clip",
            "0",
        ],
    );
    assert_eq!(
        String::from_utf8_lossy(&explicit),
        String::from_utf8_lossy(&default),
        "the new flags at their defaults moved the loss curve"
    );

    // The knobs themselves, and not only their agreement with each other: this is the
    // assertion that fails if a default is ever edited (packet M7/T4's `forbidden`).
    assert_eq!(
        report["optimizer"],
        serde_json::json!({
            "kind": "AdamW", "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0.01,
        }),
        "the default optimizer moved: every measured run in the design notes was at this one"
    );
    assert_eq!(report["schedule"], "constant");
    assert_eq!(report["batch"], 4, "the flag this test passes");
    assert_eq!(report["grad_clip"], 0.0);
    assert_eq!(report["first_nonfinite_step"], serde_json::Value::Null);

    // The applied rate never moved off `--lr`, stated as the hash the summary reports.
    let flat: Vec<u8> = std::iter::repeat_n(1e-4f64.to_le_bytes(), ORACLE_STEPS)
        .flatten()
        .collect();
    let constant = blake3::hash(&flat).to_hex().to_string();
    match report["lr_curve_hash"].as_str() {
        Some(reported) => assert_eq!(
            reported, constant,
            "a constant schedule applied something other than {ORACLE_STEPS} copies of --lr"
        ),
        None => println!("SKIP the lr_curve_hash half: `blake3` is not importable by {python}"),
    }

    // ...and a schedule that is not constant *is* a different run, so the equality above is
    // a property of the default and not of a flag that does nothing.
    let (cosine, cosine_report) = train(
        "cosine",
        &["--schedule", "warmup_cosine", "--warmup-steps", "10"],
    );
    assert_ne!(
        String::from_utf8_lossy(&cosine),
        String::from_utf8_lossy(&default),
        "--schedule warmup_cosine did not move the loss at all"
    );
    println!(
        "RAN the_default_schedule_is_the_old_run: {ORACLE_STEPS} steps, default curve \
         {} bytes, lr_curve_hash {}\n  default: {report}\n  cosine:  {cosine_report}",
        default.len(),
        report["lr_curve_hash"]
    );
}

/// Packet M7/T3. The batch axis has to be an axis and nothing more: the same eight
/// observations through the lowered module as one `[8, ..]` batch and as eight `[1, ..]` calls
/// must be the same eight chunks.
///
/// Every node of the demo graph is per-sample by construction — `GroupNorm` normalizes over
/// channel groups of the sample in front of it (packet M5/V13, which is *why* it is
/// `GroupNorm`), `nn.Linear` is a row-wise affine, and the transformer sees a one-token
/// sequence per row — so nothing in the architecture can make a row depend on its neighbours.
/// What is left to differ is which matmul kernel torch dispatches on `[8, 512]` against
/// `[1, 512]`, which is a floating-point difference and not a semantic one, so the bar is spec
/// 8.9's tier-4 fp32 rather than bitwise. The measured difference is printed: a *large* one
/// would mean a node that mixes rows, and that is the failure this exists to catch.
#[test]
#[ignore = "needs torch and torchvision"]
fn the_batch_is_invariant() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP batch_invariance: {why}");
            return;
        }
    };
    let dir = scratch_dir("batch-invariance");
    let (path, _) = demo_bundle(&dir);
    let (build, contract) = lowered(&dir, &path);
    assert!(
        contract.batch_axis,
        "the contract must declare the axis this test is about"
    );

    let script = dir.join("batch_invariance.py");
    std::fs::write(&script, BATCH_INVARIANCE_PY).expect("write the probe");
    let out = run(
        &python,
        &[&script.to_string_lossy(), &build.to_string_lossy(), "8"],
    );
    let report: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e} in `{}`", String::from_utf8_lossy(&out.stdout)));
    let max_abs = report["max_abs"].as_f64().expect("max_abs is a number");
    assert!(
        max_abs <= Tolerance::TIER4_FP32.abs,
        "a batch of 8 and 8 single calls disagree by {max_abs:e}, over spec 8.9's tier-4 fp32 \
         tolerance of {:e}: a node in the lowering mixes rows",
        Tolerance::TIER4_FP32.abs
    );
    println!(
        "RAN the_batch_is_invariant: max_abs {max_abs:e} (tol {:e}) over a batched chunk {}",
        Tolerance::TIER4_FP32.abs,
        report["shape"]
    );
}

/// argv is `<module dir> <N>`; stdout is `{"max_abs": f, "shape": [..]}`.
///
/// The weights are the module's own fixed-seed initialisation rather than a checkpoint: what
/// is compared is one function against itself at two batch sizes, and an untrained
/// `EsPolicy()` is that function as surely as a trained one. No `torch_ref.py` either — this
/// is a second implementation of the call, not the same one twice.
const BATCH_INVARIANCE_PY: &str = r#"
import json, sys
import torch
module_dir, n = sys.argv[1], int(sys.argv[2])
namespace = {}
exec(compile(open(module_dir + "/es_policy.py").read(), "<es-policy>", "exec"), namespace)
torch.manual_seed(0)
model = namespace["EsPolicy"]()
model.eval()
contract = json.load(open(module_dir + "/contract.json"))
assert contract["batch_axis"], "the contract does not declare a batch axis"
generator = torch.Generator().manual_seed(7)
batch = {
    port: torch.rand([n] + shape, generator=generator, dtype=torch.float32)
    for port, shape in contract["inputs"].items()
}
with torch.inference_mode():
    together = next(iter(model(**batch).values()))
    apart = torch.cat(
        [
            next(iter(model(**{p: t[i : i + 1] for p, t in batch.items()}).values()))
            for i in range(n)
        ]
    )
assert together.shape[0] == n, "the module did not return a batch of %d: %s" % (
    n,
    list(together.shape),
)
assert together.shape == apart.shape, "%s vs %s" % (list(together.shape), list(apart.shape))
sys.stdout.write(
    json.dumps(
        {"max_abs": float((together - apart).abs().max()), "shape": list(together.shape)}
    )
)
"#;

// --- packet M7/T5: the pretrained backbone ---------------------------------------------------

/// The demo bundle built from `learning-pretrained.toml` instead of `learning.toml` — the same
/// graph with `pretrained = true` — with `frozen` overridden when the caller asks.
fn pretrained_bundle(dir: &Path, frozen: bool) -> (PathBuf, PolicyBundle) {
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let mut learning = es_ir::serial::learning_from_toml(&read("learning-pretrained.toml"))
        .expect("learning-pretrained.toml");
    for node in learning.nodes.nodes.values_mut() {
        if let es_ir::learning::LearningNode::VisionEncoder { frozen: f, .. } = node {
            *f = frozen;
        }
    }
    let weights = b"es-t5-untrained-placeholder".to_vec();
    learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = PolicyBundle::build(
        &es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the pretrained demo documents pack into a bundle");
    let path = dir.join(if frozen {
        "pretrained-frozen.esb"
    } else {
        "pretrained.esb"
    });
    std::fs::write(&path, &bytes).expect("write the pretrained bundle");
    let opened = PolicyBundle::open(&bytes).expect("the bundle just written opens");
    (path, opened)
}

/// The pinned blake3, read out of `crates/es-data/src/training.rs`.
///
/// One copy of that number exists in this repository and this is not it: `es-data` is layer 10
/// and this crate is layer 8 (spec 4.2), so the constant cannot be imported, and a second
/// literal would be a second thing to forget. The file is read as text.
fn pinned_blake3() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../es-data/src/training.rs");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let after = text
        .split_once("pub const RESNET18_IMAGENET1K_V1_BLAKE3: &str =")
        .unwrap_or_else(|| panic!("{}: the pin is gone", path.display()))
        .1;
    after
        .split('"')
        .nth(1)
        .expect("the pin is a string literal")
        .to_owned()
}

/// `python/es/fetch_backbone.py`, next to the trainer.
fn fetch_backbone_py() -> PathBuf {
    train_act_py()
        .parent()
        .expect("python/es")
        .join("fetch_backbone.py")
}

/// The `ImageNet` artifact, fetched once and reused.
///
/// `ES_BACKBONE_DIR` names where it lives — `~/artifacts/plan-v/m7-t5` on the oracle server.
/// Without it the script is run into a scratch directory; torchvision caches the `.pth`, so
/// the second test of a session pays for serialization and not for the download.
fn backbone_artifact(python: &str, dir: &Path) -> Result<PathBuf, String> {
    let out = std::env::var("ES_BACKBONE_DIR").map_or_else(|_| dir.join("backbone"), PathBuf::from);
    let file = out.join("resnet18-imagenet1k-v1.safetensors");
    if file.is_file() {
        return Ok(file);
    }
    let done = Command::new(python)
        .arg(fetch_backbone_py())
        .args(["--arch", "resnet18", "--out"])
        .arg(&out)
        .args(["--expect", &pinned_blake3()])
        .output()
        .map_err(|e| format!("{python}: {e}"))?;
    if !done.status.success() {
        return Err(format!(
            "fetch_backbone.py: {}",
            String::from_utf8_lossy(&done.stderr).trim_end()
        ));
    }
    Ok(file)
}

/// Oracle 2 of packet M7/T5. The pretrained backbone is initialised from the artifact and
/// **still has no training mode**: `FrozenBatchNorm2d`'s affine constants and running
/// statistics do not read `self.training`, so `train()` and `eval()` are one function, bit for
/// bit. That is V13's rule kept — what the optimizer minimizes is what `torch_ref.py` deploys.
///
/// The second half is what makes the hash chain mean something here: every tensor the module
/// ends up holding is bitwise the tensor in the file whose blake3 `es train` verified. A loader
/// that transposed, cast or averaged anything would pass the first half and fail this.
#[test]
#[ignore = "needs torch and torchvision"]
fn the_pretrained_backbone_has_no_training_mode() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP the_pretrained_backbone_has_no_training_mode: {why}");
            return;
        }
    };
    let dir = scratch_dir("pretrained-train-eval");
    let artifact = match backbone_artifact(&python, &dir) {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP the_pretrained_backbone_has_no_training_mode: {why}");
            return;
        }
    };
    let (path, _) = pretrained_bundle(&dir, false);
    let (build, _) = lowered(&dir, &path);

    let script = dir.join("pretrained_probe.py");
    std::fs::write(&script, PRETRAINED_PROBE_PY).expect("write the probe");
    let out = run(
        &python,
        &[
            &script.to_string_lossy(),
            &train_act_py().to_string_lossy(),
            &build.to_string_lossy(),
            &artifact.to_string_lossy(),
        ],
    );
    let report = text(&out);
    assert!(
        out.status.success(),
        "the pretrained probe failed:\n{report}"
    );
    println!(
        "RAN the_pretrained_backbone_has_no_training_mode: {}",
        report.trim()
    );
}

/// argv is `<train_act.py> <module dir> <artifact.safetensors>`; exits non-zero with what
/// disagreed. `train_act.py` is `exec`'d rather than reimplemented — the loader under test is
/// the one the trainer uses, not a second copy of it.
const PRETRAINED_PROBE_PY: &str = r#"
import json, sys
from pathlib import Path
import torch
trainer, module_dir, artifact = sys.argv[1], sys.argv[2], sys.argv[3]
train_act = {"__name__": "es_train_act_probe"}
exec(compile(open(trainer, encoding="utf-8").read(), trainer, "exec"), train_act)
namespace = {}
exec(compile(open(module_dir + "/es_policy.py").read(), "<es-policy>", "exec"), namespace)

model = namespace["EsPolicy"]()
# No network at construction: the module was built before any of this, and what follows is the
# only thing that gives it ImageNet tensors (spec 2.5).
tensors = train_act["read_safetensors"](Path(artifact))
report = train_act["init_backbone"](model, tensors)

backbones = [(n, m) for n, m in model.named_children() if type(m).__name__ == "ResNet"]
assert backbones, "the pretrained graph must instantiate a backbone"

# Bitwise, not close: a checkpoint is bytes, and `es train` verified the blake3 of these.
loaded = 0
for name, member in backbones:
    own = member.state_dict()
    for key, value in tensors.items():
        if key.startswith("fc."):
            continue
        assert torch.equal(own[key].cpu(), value), "%s.%s is not the file's tensor" % (name, key)
        loaded += 1

# FrozenBatchNorm2d is an affine pair and a statistics pair, all four constants -- which is
# exactly why the two modes agree.
stats = [k for k in model.state_dict() if k.endswith(("running_mean", "running_var"))]
assert stats, "a frozen BatchNorm keeps its statistics as constants; none survived"
assert not [
    k for k in model.state_dict() if k.endswith("num_batches_tracked")
], "num_batches_tracked has no place in a frozen norm"

shapes = json.load(open(module_dir + "/contract.json"))["inputs"]
image = [s for s in shapes.values() if len(s) == 3][0]
x = torch.rand([2] + image, generator=torch.Generator().manual_seed(0))
for name, backbone in backbones:
    with torch.no_grad():
        model.train()
        trained = backbone(x)
        model.eval()
        deployed = backbone(x)
    assert torch.equal(trained, deployed), "%s: train() and eval() differ by %g" % (
        name,
        (trained - deployed).abs().max(),
    )
sys.stdout.write(
    json.dumps(
        {
            "backbones": report,
            "tensors_bitwise_equal": loaded,
            "frozen_statistics": len(stats),
            "train_equals_eval": True,
        }
    )
)
"#;

/// Oracle 5 of packet M7/T5. `frozen: true` means the optimizer never sees `nodes.<k>.*`: after
/// 20 steps every backbone tensor is bitwise what the artifact held, and the head moved.
///
/// Both halves matter. Unchanged alone would also be true of a run that did nothing; a moved
/// head is what says the 20 steps happened. And "bitwise" is the right comparison because
/// `AdamW` with a non-zero weight decay moves a parameter that has no gradient, so a backbone
/// merely left in the optimizer would drift — quietly, and only in the last bits.
#[test]
#[ignore = "needs torch, torchvision and pyarrow"]
fn frozen_excludes_the_backbone_from_the_optimizer() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP frozen_excludes_the_backbone_from_the_optimizer: {why}");
            return;
        }
    };
    let dir = scratch_dir("frozen-backbone");
    let artifact = match backbone_artifact(&python, &dir) {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP frozen_excludes_the_backbone_from_the_optimizer: {why}");
            return;
        }
    };
    let (path, _) = pretrained_bundle(&dir, true);
    let (build, _) = lowered(&dir, &path);
    let baked = mini_baked(&python, &dir, &path);

    let trained = dir.join("frozen.safetensors");
    let done = run(
        &python,
        &[
            &train_act_py().to_string_lossy(),
            "--module",
            &build.to_string_lossy(),
            "--baked",
            &baked.to_string_lossy(),
            "--out",
            &trained.to_string_lossy(),
            "--init-backbone",
            &artifact.to_string_lossy(),
            "--batch",
            "4",
            "--seed",
            "0",
            "--checkpoint-at",
            "20",
            // A weight decay a frozen parameter would feel if it were in the optimizer.
            "--weight-decay",
            "0.1",
        ],
    );
    assert!(
        done.status.success(),
        "the trainer failed:\n{}",
        text(&done)
    );
    let summary: serde_json::Value = serde_json::from_str(
        String::from_utf8_lossy(&done.stdout)
            .lines()
            .last()
            .unwrap_or_default(),
    )
    .expect("the trainer prints one JSON line");
    assert!(
        summary["frozen_parameters"].as_u64().unwrap_or(0) > 1_000_000,
        "a frozen ResNet18 is millions of parameters: {summary}"
    );
    assert!(
        summary["trainable_parameters"].as_u64().unwrap_or(0) > 0,
        "{summary}"
    );

    let script = dir.join("compare.py");
    std::fs::write(&script, FROZEN_COMPARE_PY).expect("write the comparison");
    let out = run(
        &python,
        &[
            &script.to_string_lossy(),
            &train_act_py().to_string_lossy(),
            &artifact.to_string_lossy(),
            &trained.to_string_lossy(),
        ],
    );
    let report = text(&out);
    assert!(out.status.success(), "{report}");
    println!(
        "RAN frozen_excludes_the_backbone_from_the_optimizer: {}\n  trainer: {summary}",
        report.trim()
    );
}

/// argv is `<train_act.py> <artifact.safetensors> <trained.safetensors>`; exits non-zero when a
/// frozen tensor moved or when nothing outside the backbone is there at all.
const FROZEN_COMPARE_PY: &str = r#"
import json, sys
from pathlib import Path
import torch
trainer, artifact, trained = sys.argv[1], sys.argv[2], sys.argv[3]
train_act = {"__name__": "es_train_act_probe"}
exec(compile(open(trainer, encoding="utf-8").read(), trainer, "exec"), train_act)
read = train_act["read_safetensors"]
before, after = read(Path(artifact)), read(Path(trained))

# The checkpoint is keyed `nodes.<k>.<rest>`; the artifact is keyed torchvision's way.
node = None
for key in after:
    if key.startswith("nodes.") and key.endswith(".conv1.weight"):
        node = key.split(".")[1]
assert node is not None, "no backbone in the checkpoint: %s" % sorted(after)[:8]

held = 0
for key, value in before.items():
    if key.startswith("fc."):
        continue
    mapped = "nodes.%s.%s" % (node, key)
    assert mapped in after, "%s is not in the checkpoint" % mapped
    assert torch.equal(after[mapped], value), "%s moved under frozen = true" % mapped
    held += 1

head = [k for k in after if not k.startswith("nodes.%s." % node)]
assert head, "the checkpoint holds nothing but the backbone"
sys.stdout.write(
    json.dumps({"frozen_tensors_unchanged": held, "other_tensors": len(head), "node": node})
)
"#;

/// The packet's hash rule, without an interpreter: `lowering_hash` moves for the pretrained
/// graph and for no other, and the committed `learning.toml` is exactly where it was.
///
/// The literal is the demo's own, recorded in `docs/design/learning-lowering.md` section 5.3.
/// It is the one that matters: every measured run in `docs/design/visible-learning.md`
/// section 7 names it, and this packet is not allowed to move it.
#[test]
fn the_demo_lowering_hash_moves_only_for_the_pretrained_graph() {
    let dir = scratch_dir("lowering-hash");
    let (_, scratch) = demo_bundle(&dir);
    let (_, pre) = pretrained_bundle(&dir, false);
    let from_scratch = lower_to_torch(&scratch.learning).expect("the demo graph lowers");
    let pretrained = lower_to_torch(&pre.learning).expect("the pretrained graph lowers");
    let hex = |d: &[u8; 32]| {
        use std::fmt::Write as _;
        d.iter().fold(String::new(), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        })
    };

    assert_eq!(
        hex(&from_scratch.lowering_hash),
        FROM_SCRATCH_LOWERING_HASH,
        "packet M7/T5 moved the from-scratch lowering; every number in design note \
         `visible-learning.md` section 7 was measured under the old one"
    );
    assert_ne!(
        from_scratch.lowering_hash, pretrained.lowering_hash,
        "the two graphs lower to the same source"
    );
    println!(
        "from-scratch lowering_hash {}\npretrained   lowering_hash {}",
        hex(&from_scratch.lowering_hash),
        hex(&pretrained.lowering_hash)
    );
}

/// The demo's `lowering_hash` as packet M7/T3 left it (design note `learning-lowering.md`
/// section 5.2's table).
const FROM_SCRATCH_LOWERING_HASH: &str =
    "3d06811c52b6887f021440d4ce9a1a061e4eb27a0abb97c79822acd4d8a2d394";

// --- packet M7/T6: the augmentation, in Rust and in Python -------------------------------

fn augment_py() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/es/augment.py")
}

fn augment_golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/train/augment_seed0.json")
}

/// Murmur3's `fmix32` — `es_render::rng::mix32` and `python/es/augment.py`'s, the same ten
/// lines of integer arithmetic in the third language that has to agree about them.
///
/// `es-policy` does not depend on `es-render` (layer 8 does not need layer 5 for a test), so
/// this is a copy on purpose; what makes it honest is that the golden below was produced by
/// the *Python* one and this has to reproduce it bitwise.
fn mix32(mut z: u32) -> u32 {
    z ^= z >> 16;
    z = z.wrapping_mul(0x85eb_ca6b);
    z ^= z >> 13;
    z = z.wrapping_mul(0xc2b2_ae35);
    z ^ (z >> 16)
}

/// `(augmentation_seed, sample_index, step, node_index)` — the stream one node draws from for
/// one sample at one optimizer step (design note `training-recipe.md` section 13).
fn aug_key(seed: u64, sample: u32, step: u32, node: u32) -> u32 {
    let mut k = mix32(seed as u32);
    k = mix32(k ^ ((seed >> 32) as u32));
    k = mix32(k ^ sample);
    k = mix32(k ^ step);
    mix32(k ^ node)
}

/// Draw `i` of stream `k`, in `[0, 1)`: the top 24 bits, so the value is exact in `f32`.
fn aug_uniform(k: u32, i: u32) -> f64 {
    let bits = mix32(k ^ i.wrapping_mul(0x9e37_79b9));
    f64::from(bits >> 8) * (1.0 / 16_777_216.0)
}

/// The Rust re-implementation of `python/es/augment.py::apply_chain`, node for node.
///
/// `values` is one sample, `shape` its `(channels, height, width)`; the return is the
/// augmented sample and its shape. Every scalar is computed in `f64` and rounded to `f32`
/// before it touches a value, which is what lets the two implementations agree without
/// agreeing about widening.
fn apply_chain(
    chain: &[serde_json::Value],
    mut values: Vec<f32>,
    mut shape: (usize, usize, usize),
    sample: u32,
    step: u32,
    seed: u64,
) -> (Vec<f32>, (usize, usize, usize)) {
    for node in chain {
        let stream = aug_key(
            seed,
            sample,
            step,
            node["node"].as_u64().expect("node id") as u32,
        );
        let param = |key: &str| node[key].as_f64().unwrap_or_else(|| panic!("{key}"));
        let (planes, height, width) = shape;
        match node["kind"].as_str().expect("kind") {
            "RandomCrop" => {
                let (crop_w, crop_h) = (param("width") as usize, param("height") as usize);
                let span = |u: f64, from: usize, to: usize| {
                    ((u * (from - to + 1) as f64) as usize).min(from - to)
                };
                let left = span(aug_uniform(stream, 0), width, crop_w);
                let top = span(aug_uniform(stream, 1), height, crop_h);
                let mut out = Vec::with_capacity(planes * crop_h * crop_w);
                for plane in 0..planes {
                    for row in 0..crop_h {
                        let at = plane * height * width + (row + top) * width + left;
                        out.extend_from_slice(&values[at..at + crop_w]);
                    }
                }
                values = out;
                shape = (planes, crop_h, crop_w);
            }
            "ColorJitter" => {
                for refused in ["saturation", "hue"] {
                    // Exactly zero is the condition `augment.py` refuses on, so the
                    // comparison here has to be the same exact one.
                    assert!(
                        param(refused) == 0.0,
                        "the Rust half refuses what Python refuses"
                    );
                }
                let gain =
                    (1.0 + (2.0 * aug_uniform(stream, 0) - 1.0) * param("brightness")) as f32;
                for value in &mut values {
                    *value *= gain;
                }
                // The one reduction: f64, then rounded once to f32. Python's is torch's
                // `.double().mean()`, whose summation order is its own — see section 13.
                let mean = (values.iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64)
                    as f32;
                let contrast =
                    (1.0 + (2.0 * aug_uniform(stream, 1) - 1.0) * param("contrast")) as f32;
                for value in &mut values {
                    *value = (*value - mean) * contrast + mean;
                }
            }
            "GaussianNoise" => {
                let sigma = param("sigma");
                for (at, value) in values.iter_mut().enumerate() {
                    let at = at as u32;
                    // `1 - u` is in (0, 1], so the logarithm is always defined -- the same
                    // guard `augment.py` uses, and it has to be the same one.
                    let first = 1.0 - aug_uniform(stream, 2 * at);
                    let second = aug_uniform(stream, 2 * at + 1);
                    let radius = (-2.0 * first.ln()).sqrt();
                    let angle = (2.0 * std::f64::consts::PI * second).cos();
                    *value += (sigma * radius * angle) as f32;
                }
            }
            other => panic!("the golden names {other}, which this oracle does not implement"),
        }
    }
    (values, shape)
}

/// argv is `<augment.py> <request.json>`; stdout is `{"output": [[...], ...]}`.
///
/// The module is `exec`'d rather than imported for the reason the lr probe `exec`s
/// `train_act.py`: this reads the file the repository ships, from wherever the test runs.
const AUGMENT_PROBE_PY: &str = r#"
import json, sys
import torch
namespace = {"__name__": "es_augment_probe"}
source = open(sys.argv[1], encoding="utf-8").read()
exec(compile(source, sys.argv[1], "exec"), namespace)
request = json.loads(open(sys.argv[2], encoding="utf-8").read())
shape = request["shape"]
x = torch.tensor(request["input"], dtype=torch.float32).reshape(shape)
rows = [x.clone() for _ in request["samples"]]
out = namespace["apply_chain"](
    request["chain"],
    torch.stack(rows),
    request["samples"],
    request["step"],
    request["seed"],
)
sys.stdout.write(json.dumps({"output": [row.reshape(-1).tolist() for row in out]}))
"#;

/// The request the golden was produced from, and the one the Python half is asked to repeat.
fn augment_request(golden: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "shape": golden["shape"], "input": golden["input"], "chain": golden["chain"],
        "samples": golden["samples"], "step": golden["step"], "seed": golden["seed"],
    })
}

/// Generates `tests/golden/train/augment_seed0.json` from `python/es/augment.py`. Run once,
/// explicitly, with `ES_PYTHON` **and** `ES_GENERATE_GOLDENS=1`; the file is then read-only
/// (spec 1.4).
///
/// The second variable is not belt and braces: `cargo test -- --include-ignored` runs every
/// `#[ignore]`d test in the workspace, and an M7 review found exactly that rewriting a golden
/// nobody meant to touch. A generator has to refuse to run by accident.
#[test]
#[ignore = "golden generator; run explicitly with ES_PYTHON and ES_GENERATE_GOLDENS=1"]
fn generate_augment_golden() {
    assert_eq!(
        std::env::var("ES_GENERATE_GOLDENS").as_deref(),
        Ok("1"),
        "refusing to rewrite a golden without ES_GENERATE_GOLDENS=1"
    );
    let python = python_with_torch().expect("the generator needs an interpreter");
    let dir = scratch_dir("augment-golden");
    let (c, h, w) = (3usize, 12usize, 12usize);
    // A ramp with per-channel structure, in f32 from the start: every value the golden
    // carries is exactly an f32, so reading it back on either side is lossless.
    let input: Vec<f32> = (0..c * h * w)
        .map(|i| ((i * 37 % 211) as f32) / 211.0)
        .collect();
    let chain = serde_json::json!([
        {"node": 8, "kind": "RandomCrop", "width": 8, "height": 8},
        {"node": 9, "kind": "ColorJitter", "brightness": 0.2, "contrast": 0.2,
         "saturation": 0.0, "hue": 0.0},
        {"node": 10, "kind": "GaussianNoise", "sigma": 0.05},
    ]);
    let mut golden = serde_json::json!({
        "schema_version": 1,
        "source": "python/es/augment.py::apply_chain",
        "seed": 0, "step": 0, "samples": [0, 1, 2, 3, 4],
        "shape": [c, h, w],
        "chain": chain,
        "input": input,
    });
    let request = dir.join("request.json");
    std::fs::write(&request, augment_request(&golden).to_string()).expect("write the request");
    let probe = run(
        &python,
        &[
            "-c",
            AUGMENT_PROBE_PY,
            &augment_py().to_string_lossy(),
            &request.to_string_lossy(),
        ],
    );
    let reply: serde_json::Value = serde_json::from_slice(&probe.stdout).expect("the probe");
    golden["output"] = reply["output"].clone();
    std::fs::create_dir_all(augment_golden().parent().expect("a parent")).expect("golden dir");
    std::fs::write(
        augment_golden(),
        serde_json::to_string_pretty(&golden).expect("serialize") + "\n",
    )
    .expect("write the golden");
    println!(
        "RAN generate_augment_golden: {} samples -> {}",
        golden["samples"].as_array().expect("samples").len(),
        augment_golden().display()
    );
}

/// Oracle 1 of packet M7/T6. The Rust re-implementation above equals the golden bitwise at
/// `f32`, with no interpreter; with `ES_PYTHON`, `python/es/augment.py` equals it too.
///
/// Both halves are exact comparisons. The draws are integer arithmetic, so they cannot
/// legitimately differ at all; the values are `f32` results of scalars rounded from `f64`,
/// which is the design that makes "exact" a reasonable thing to ask for across two languages
/// and two libms (the same measurement T4 section 10 made for `cos`).
#[test]
fn augmentation_matches_the_golden() {
    let text = std::fs::read_to_string(augment_golden())
        .unwrap_or_else(|e| panic!("{}: {e}", augment_golden().display()));
    let golden: serde_json::Value = serde_json::from_str(&text).expect("the golden is JSON");
    let floats = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .expect("an array")
            .iter()
            .map(|x| x.as_f64().expect("a number") as f32)
            .collect()
    };
    let input = floats(&golden["input"]);
    let dims = golden["shape"].as_array().expect("shape");
    let shape = (
        dims[0].as_u64().expect("c") as usize,
        dims[1].as_u64().expect("h") as usize,
        dims[2].as_u64().expect("w") as usize,
    );
    let chain = golden["chain"].as_array().expect("chain").clone();
    let (seed, step) = (
        golden["seed"].as_u64().expect("seed"),
        golden["step"].as_u64().expect("step") as u32,
    );
    let samples: Vec<u32> = golden["samples"]
        .as_array()
        .expect("samples")
        .iter()
        .map(|s| s.as_u64().expect("a sample") as u32)
        .collect();
    let want: Vec<Vec<f32>> = golden["output"]
        .as_array()
        .expect("output")
        .iter()
        .map(floats)
        .collect();
    assert_eq!(want.len(), samples.len());

    let mut differ = Vec::new();
    for (row, sample) in samples.iter().enumerate() {
        let (got, out_shape) = apply_chain(&chain, input.clone(), shape, *sample, step, seed);
        assert_eq!(out_shape, (3, 8, 8), "the chain's output shape");
        assert_eq!(got.len(), want[row].len(), "sample {sample}: length");
        for (at, (mine, theirs)) in got.iter().zip(&want[row]).enumerate() {
            if mine.to_bits() != theirs.to_bits() {
                differ.push((*sample, at, *mine, *theirs));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "the Rust re-implementation is not the golden at {} of {} values, first {:?}",
        differ.len(),
        want.len() * want[0].len(),
        &differ[..differ.len().min(4)]
    );
    // Non-vacuity: the five samples must actually differ from one another, or the comparison
    // above would pass on an implementation that ignores `sample_index` entirely.
    assert!(
        want.windows(2).all(|pair| pair[0] != pair[1]),
        "two samples were augmented identically"
    );

    match python_with_torch() {
        Err(why) => println!("SKIP the interpreter half of augmentation_matches_the_golden: {why}"),
        Ok(python) => {
            let dir = scratch_dir("augment-check");
            let request = dir.join("request.json");
            std::fs::write(&request, augment_request(&golden).to_string()).expect("the request");
            let probe = run(
                &python,
                &[
                    "-c",
                    AUGMENT_PROBE_PY,
                    &augment_py().to_string_lossy(),
                    &request.to_string_lossy(),
                ],
            );
            let reply: serde_json::Value =
                serde_json::from_slice(&probe.stdout).expect("the probe replies JSON");
            let from_python: Vec<Vec<f32>> = reply["output"]
                .as_array()
                .expect("output")
                .iter()
                .map(floats)
                .collect();
            assert_eq!(from_python.len(), want.len());
            for (row, (a, b)) in from_python.iter().zip(&want).enumerate() {
                for (at, (mine, theirs)) in a.iter().zip(b).enumerate() {
                    assert_eq!(
                        mine.to_bits(),
                        theirs.to_bits(),
                        "sample {row} value {at}: the script gives {mine:e}, the golden                          {theirs:e}"
                    );
                }
            }
            println!("RAN augmentation_matches_the_golden: interpreter half too, {python}");
        }
    }
    println!(
        "RAN augmentation_matches_the_golden: {} samples x {} values bitwise at f32",
        want.len(),
        want[0].len()
    );
}

/// `demo_bundle` with `observation-augmented.toml` in place of `observation.toml` — the same
/// Task, Learning and Deployment IR, so `learning_hash` is unmoved and only the observation
/// changed (packet M7/T6).
fn augmented_bundle(dir: &Path) -> PathBuf {
    let read = |name: &str| std::fs::read_to_string(vl_fixture(name)).expect(name);
    let mut learning =
        es_ir::serial::learning_from_toml(&read("learning.toml")).expect("learning.toml");
    let weights = b"es-t6-untrained-placeholder".to_vec();
    learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = PolicyBundle::build(
        &es_ir::serial::task_from_toml(&read("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&read("observation-augmented.toml"))
            .expect("observation-augmented.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the augmented documents pack into a bundle");
    let path = dir.join("untrained-augmented.esb");
    std::fs::write(&path, &bytes).expect("write the bundle");
    path
}

/// Oracle 5 of packet M7/T6. The augmented document trains: the bake writes the boundary, the
/// trainer applies the chain, the loss is finite, and two runs at one `augmentation_seed`
/// give **byte-identical** loss curves while two seeds do not.
///
/// That pair is the whole claim of a counter-based RNG. A `torch.Generator` would pass the
/// first half and fail to be reproducible anywhere else; what makes this meaningful is that
/// the values are also the ones `augmentation_matches_the_golden` pins without an interpreter.
#[test]
#[ignore = "needs torch, torchvision and pyarrow"]
fn augmented_training_runs() {
    let python = match python_with_torch() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP augmented_training_runs: {why}");
            return;
        }
    };
    let dir = scratch_dir("augmented-train");
    let bundle = augmented_bundle(&dir);
    let opened = PolicyBundle::open(&std::fs::read(&bundle).expect("read")).expect("open");
    let chains = es_compile::plan::augmentation_chains(&opened.observation).expect("one chain");
    assert_eq!(chains.len(), 1, "{chains:?}");
    let chain = &chains["rgb_overhead"];
    assert_eq!(chain.len(), 2, "the fixture declares a crop and a jitter");

    // The file `es train` writes, written here by hand because `es-data` is layer 10 and this
    // crate is layer 8 (spec 4.2). `crates/es/tests/cli.rs` is where the *real* writer's
    // shape is pinned; what this needs is a file `python/es/augment.py` can read.
    let json_chain: Vec<serde_json::Value> = chain
        .iter()
        .map(|step| match step.kind {
            es_ir::observation::AugmentKind::RandomCrop { width, height } => serde_json::json!({
                "node": step.node.0, "kind": "RandomCrop", "width": width, "height": height,
            }),
            es_ir::observation::AugmentKind::ColorJitter {
                brightness,
                contrast,
                saturation,
                hue,
            } => serde_json::json!({
                "node": step.node.0, "kind": "ColorJitter", "brightness": brightness,
                "contrast": contrast, "saturation": saturation, "hue": hue,
            }),
            other => panic!("the fixture grew a {other:?}"),
        })
        .collect();
    let augmentation = |seed: u64| -> serde_json::Value {
        serde_json::json!({
            "kind": "observation-ir",
            "observation_hash":
                es_policy::weights::hex(&opened.observation.observation_hash().expect("hash")),
            "seed": seed,
            "chains": {"rgb_overhead": json_chain.clone()},
        })
    };

    // The bake at the boundary: the image port is the `104x104` canvas the `Pad` produced,
    // and the module still declares `3x96x96` -- which is what the chain brings it back to.
    let (dataset, tiles, baked) = (dir.join("ds"), dir.join("tiles"), dir.join("baked"));
    run(
        &python,
        &[
            "-c",
            MINI_DATASET_PY,
            &dataset.to_string_lossy(),
            &tiles.to_string_lossy(),
        ],
    );
    let bake = es(&[
        "dataset",
        "bake",
        "--policy",
        &bundle.to_string_lossy(),
        "--out",
        &baked.to_string_lossy(),
        "--frames",
        &tiles.to_string_lossy(),
        "--scene",
        &demo_scene().to_string_lossy(),
        "--for-training",
        &dataset.to_string_lossy(),
    ]);
    assert_eq!(bake.status.code(), Some(0), "{}", text(&bake));
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(baked.join("manifest.json")).expect("the manifest"),
    )
    .expect("the manifest is JSON");
    assert_eq!(
        manifest["tensors"]["rgb_overhead"]["shape"],
        serde_json::json!([3, 104, 104]),
        "the bake did not write the chain's input"
    );
    assert_eq!(
        manifest["augmentation"]["rgb_overhead"][0]["kind"],
        "RandomCrop"
    );

    let (build, contract) = lowered(&dir, &bundle);
    assert_eq!(
        contract.inputs["rgb_overhead"],
        vec![3, 96, 96],
        "the module's input is the network's, not the boundary's"
    );

    let train = |tag: &str, seed: u64| -> (Vec<u8>, serde_json::Value) {
        let config = dir.join(format!("{tag}-augmentation.json"));
        std::fs::write(&config, augmentation(seed).to_string()).expect("write the config");
        let curve = dir.join(format!("{tag}.json"));
        let out = dir.join(format!("{tag}.safetensors"));
        let done = run(
            &python,
            &[
                &train_act_py().to_string_lossy(),
                "--module",
                &build.to_string_lossy(),
                "--baked",
                &baked.to_string_lossy(),
                "--out",
                &out.to_string_lossy(),
                "--batch",
                "4",
                "--seed",
                "0",
                "--checkpoint-at",
                &ORACLE_STEPS.to_string(),
                "--loss-curve",
                &curve.to_string_lossy(),
                "--augmentation",
                &config.to_string_lossy(),
            ],
        );
        let report: serde_json::Value = serde_json::from_str(
            String::from_utf8_lossy(&done.stdout)
                .lines()
                .last()
                .unwrap_or_default(),
        )
        .expect("the trainer's summary line is JSON");
        (
            std::fs::read(&curve).unwrap_or_else(|e| panic!("{}: {e}", curve.display())),
            report,
        )
    };

    let (first, report) = train("seed0-a", 0);
    let (again, _) = train("seed0-b", 0);
    let (other, _) = train("seed1", 1);
    assert_eq!(
        first, again,
        "two runs at one augmentation_seed gave different loss curves; the draws are not \
         addressed by (seed, sample, step, node)"
    );
    assert_ne!(
        first, other,
        "two augmentation seeds gave one loss curve; nothing was augmented"
    );
    let losses: Vec<f64> = serde_json::from_slice(&first).expect("the curve is JSON");
    assert_eq!(losses.len(), ORACLE_STEPS, "{report}");
    assert!(losses.iter().all(|v| v.is_finite()), "{losses:?}");
    assert_eq!(report["augmentation"]["rgb_overhead"][0], "RandomCrop");
    assert_eq!(report["augmentation_seed"], 0);
    println!(
        "RAN augmented_training_runs: {} steps, loss {} -> {}, two seeds two curves",
        losses.len(),
        report["initial_loss"],
        report["final_loss"]
    );
}
