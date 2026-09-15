//! Spec 8.9's M1 gate: load `LeRobot`'s ACT checkpoint and produce the same action chunk.
//!
//! The two sides of the comparison are:
//!
//! - **reference** — `python/act_ref.py` runs the real
//!   `lerobot.policies.act.modeling_act.ACTPolicy` on a fixed synthetic observation;
//! - **ours** — `es_policy::lerobot` reads the checkpoint's own `config.json`, lowers it
//!   (`lower_act`), remaps the `LeRobot` safetensors keys into our `nodes.<id>.<param>` scheme
//!   (`remap_checkpoint`), and runs the result through `TorchRuntime` on the same observation.
//!
//! The observation is built the same way on both sides from integer arithmetic, so no tensor
//! crosses the boundary and every value is exact in f32 (see `act_ref.py`'s docstring).
//!
//! **This test SKIPs, loudly, when `ES_ACT_CHECKPOINT` is unset or `lerobot` is missing.**
//! Spec 1.4 wants the harness to exist whether or not the oracle is on a given machine; an
//! absent checkpoint must never be reported as a passing equivalence. To run it for real:
//!
//! ```text
//! ES_PYTHON=<venv>/python ES_ACT_CHECKPOINT=<dir> cargo test -p es-policy act -- --nocapture
//! ```
//!
//! where `<dir>` holds `config.json` and `model.safetensors` — from
//! `lerobot/act_aloha_sim_transfer_cube_human`, or from a `lerobot-train` run of our own
//! (packet `docs/packets/M5/V8-external-act.md`). The checkpoint is never committed — it lives
//! under `target/lerobot-cache/` (see `.gitignore`).
//!
//! `ES_ACT_OBSERVATION=<file.json>` replaces the synthetic ramp with a **recorded frame**,
//! `{"state": [...], "image": [...]}`, flat and row-major at the shapes `config.json` declares.
//! That is what makes the gate meaningful for a checkpoint trained here rather than upstream:
//! agreement on a ramp says the two modules compute the same function, agreement on a frame the
//! policy was trained on says it on the input that decides the demo. The file is written with
//! shortest-round-trip decimals, so the two sides still hold bit-identical f32.

use std::collections::BTreeMap;
use std::process::Command;

use es_compile::Tensor;
use es_ir::types::ElemType;
use es_policy::lerobot::{act_policy, lower_act, remap_checkpoint, ActConfig};
use es_policy::runtime::PolicyRuntime;
use es_policy::weights::weights_hash;
use es_policy::{compare_actions, Tolerance, TorchRuntime};

const REF: &str = include_str!("../python/act_ref.py");

/// `((i * step + offset) % 4096) / scale + shift`, the fixed observation both sides build.
/// Integer arithmetic over a power of two, so the f32 values are bit-identical to Python's.
fn ramp(count: usize, step: u64, offset: u64, scale: f32, shift: f32) -> Vec<f32> {
    (0..count as u64)
        .map(|i| ((i * step + offset) % 4096) as f32 / scale + shift)
        .collect()
}

fn tensor(shape: Vec<u64>, values: &[f32]) -> Tensor {
    Tensor {
        dtype: ElemType::F32,
        shape,
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

#[derive(serde::Deserialize)]
struct RefReply {
    ok: bool,
    #[serde(default)]
    error: String,
    #[serde(default)]
    lerobot: String,
    #[serde(default)]
    torch: String,
    #[serde(default)]
    shape: Vec<u64>,
    #[serde(default)]
    actions: Vec<f32>,
}

fn python() -> String {
    std::env::var("ES_PYTHON").unwrap_or_else(|_| "python".to_owned())
}

/// `ES_ACT_OBSERVATION`'s contents: one recorded frame, flat and row-major.
#[derive(serde::Deserialize)]
struct Obs {
    state: Vec<f32>,
    image: Vec<f32>,
}

/// The recorded frame both sides read, when one was named. A panic is a real failure: an
/// observation that was asked for and could not be used must never fall back to the ramp.
fn recorded(state_dim: usize, pixels: usize) -> Option<(Vec<f32>, Vec<f32>, String)> {
    let path = std::env::var("ES_ACT_OBSERVATION").ok()?;
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let obs: Obs = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(
        (obs.state.len(), obs.image.len()),
        (state_dim, pixels),
        "{path}: the config declares {state_dim} state values and {pixels} pixels"
    );
    Some((obs.state, obs.image, path))
}

/// Run the real `LeRobot` policy. `Err` carries a reason to SKIP on, never a silent pass.
fn reference(dir: &str, observation: Option<&str>) -> Result<RefReply, String> {
    let mut args = vec!["-c".to_owned(), REF.to_owned(), dir.to_owned()];
    args.extend(observation.map(str::to_owned));
    let out = Command::new(python())
        .args(&args)
        .output()
        .map_err(|e| format!("cannot start `{}`: {e}", python()))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().last().ok_or_else(|| {
        format!(
            "no output; stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        )
    })?;
    let reply: RefReply = serde_json::from_str(line)
        .map_err(|e| format!("{e} in `{}`", &line[..line.len().min(300)]))?;
    if !reply.ok {
        return Err(reply.error);
    }
    Ok(reply)
}

/// Whether a reference failure means "this machine does not have the oracle" (SKIP) rather
/// than "the oracle ran and something is wrong" (FAIL). Only an absent interpreter or an
/// absent module qualifies; every other exception is a real result.
fn is_missing_dependency(why: &str) -> bool {
    why.starts_with("cannot start `")
        || why.starts_with("ModuleNotFoundError:")
        || why.starts_with("ImportError:")
}

#[test]
fn a_real_lerobot_act_checkpoint_reproduces_its_actions() {
    let Ok(dir) = std::env::var("ES_ACT_CHECKPOINT") else {
        println!("SKIP: ES_ACT_CHECKPOINT is unset (spec 8.9's M1 gate needs a real checkpoint)");
        return;
    };

    let config = std::fs::read_to_string(format!("{dir}/config.json"))
        .unwrap_or_else(|e| panic!("{dir}/config.json: {e}"));
    let cfg = ActConfig::parse(&config).expect("the checkpoint's config.json must lower");
    let module = lower_act(&cfg).expect("the ACT lowering must accept its own config");

    let original = std::fs::read(format!("{dir}/model.safetensors"))
        .unwrap_or_else(|e| panic!("{dir}/model.safetensors: {e}"));
    let source_hash = weights_hash(&original);
    // LeRobot 0.6.x's second checkpoint layout keeps the normalization statistics in a
    // processor state file rather than in `model.safetensors`; both are read, neither is a flag.
    let stats = std::fs::read_to_string(format!("{dir}/policy_preprocessor.json"))
        .ok()
        .and_then(|raw| {
            let pipeline: serde_json::Value = serde_json::from_str(&raw).ok()?;
            let name = pipeline["steps"]
                .as_array()?
                .iter()
                .find(|s| s["registry_name"] == "normalizer_processor")?["state_file"]
                .as_str()?
                .to_owned();
            std::fs::read(format!("{dir}/{name}")).ok()
        });
    let remapped = remap_checkpoint(&cfg, &original, stats.as_deref())
        .expect("the remap must produce safetensors");
    let policy = act_policy(&cfg, &dir, source_hash, weights_hash(&remapped)).unwrap();

    let state_dim = cfg.state_dim().unwrap();
    let image_shape = cfg.image_shape();
    let pixels = image_shape.iter().product::<u64>() as usize;
    let frame = recorded(state_dim as usize, pixels);

    // The reference first: if `lerobot` is missing there is nothing to compare against.
    // A *reference error* is not the same thing (review `docs/reviews/M4.md` S-7): an
    // installed LeRobot that raised is a failure, not an environment this machine lacks.
    let reference = match reference(&dir, frame.as_ref().map(|(_, _, p)| p.as_str())) {
        Ok(r) => r,
        Err(why) if is_missing_dependency(&why) => {
            println!("SKIP: the LeRobot reference is not installed ({why})");
            return;
        }
        Err(why) => panic!("the LeRobot reference failed: {why}"),
    };

    let mut runtime = TorchRuntime::new();
    let info = match runtime.load_lowered(
        &module,
        &policy,
        &es_policy::WeightsSource::InMemory(remapped),
    ) {
        Ok(info) => info,
        Err(es_policy::PolicyError::Unavailable(why)) => {
            println!("SKIP: no Python with `torch` ({why})");
            return;
        }
        Err(e) => panic!("load failed: {e}"),
    };

    let camera = cfg.cameras()[0].replace('.', "_");
    let (state, image) = match &frame {
        Some((state, image, _)) => (state.clone(), image.clone()),
        None => (
            ramp(state_dim as usize, 911, 3, 2048.0, -1.0),
            ramp(pixels, 37, 11, 4096.0, 0.0),
        ),
    };
    let inputs: BTreeMap<String, Tensor> = [
        (
            camera,
            tensor(
                std::iter::once(1)
                    .chain(image_shape.iter().copied())
                    .collect(),
                &image,
            ),
        ),
        (
            "observation_state".to_owned(),
            tensor(vec![1, state_dim], &state),
        ),
    ]
    .into_iter()
    .collect();

    let outputs = runtime.infer(&inputs).expect("inference must run");
    let ours = &outputs["actions"];
    let theirs = tensor(reference.shape.clone(), &reference.actions);

    let e = compare_actions(ours, &theirs, Tolerance::TIER4_FP32);
    println!(
        "RAN act_checkpoint: lerobot {} torch {} observation {} shape {:?} max_abs {:e} max_rel \
         {:e}",
        reference.lerobot,
        reference.torch,
        frame.as_ref().map_or("ramp", |(_, _, p)| p.as_str()),
        reference.shape,
        e.max_abs,
        e.max_rel
    );

    assert_eq!(
        (info.action_dim, info.horizon),
        (cfg.action_dim().unwrap() as u32, cfg.chunk_size),
        "the contract must be the one the config declares (spec 8.4)"
    );
    assert_eq!(
        ours.shape, reference.shape,
        "the chunk shape must be [n_action_steps, action_dim]"
    );
    // Measured at the pinned version: bitwise identical. The lowering is the same graph over
    // the same weights, evaluated by the same torch, so anything above zero here means a
    // module diverged — see docs/packets/M1/P-M1-R2.md.
    assert!(
        e.pass,
        "spec 8.9 tier 4 (<= 1e-5) failed: max_abs {:e}, max_rel {:e}",
        e.max_abs, e.max_rel
    );
    assert!(
        compare_actions(ours, &theirs, Tolerance::BITWISE).pass,
        "the ACT lowering was bitwise identical when this test was written (max_abs {:e}); a \
         difference is a real regression, not float noise",
        e.max_abs
    );
}
