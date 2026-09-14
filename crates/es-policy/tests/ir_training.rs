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
    pq.write_table(
        pa.table({"observation.state": column(state), "action": column(action)}),
        os.path.join(root, "data", "chunk-000", "episode_%06d.parquet" % e),
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
        "RAN ir_training: loss {initial:.4} -> {final_:.4} over {ORACLE_STEPS} steps \
         ({:.3}x), chunk {:?}, max_abs vs a direct forward {:e} (tol {:e}), torch {}",
        final_ / initial,
        chunk.shape,
        equivalence.max_abs,
        Tolerance::TIER4_FP32.abs,
        info.version
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
    port: torch.tensor(values, dtype=torch.float32).reshape(shapes[port])
    for port, values in observation.items()
}
with torch.inference_mode():
    out = next(iter(model(**inputs).values()))
sys.stdout.write(json.dumps([float(v) for v in out.reshape(-1).tolist()]))
"#;
