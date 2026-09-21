//! `es policy lower / pack` — the two Rust ends of the training split (spec 2.3, spec 8.7).
//!
//! There is deliberately no `es policy train`: the optimizer is on the Python side of spec
//! 2.3's split, exactly as `es loop train` is not a command (`cmd/loop.rs:5-7`). What is here
//! is the pair of steps that *are* the core's: emitting the module the Learning IR lowers to
//! (`lower`) and admitting a checkpoint back into a bundle (`pack`).
//!
//! `pack` is a spec 25.1 trust boundary — `model.safetensors` came from outside the Rust core.
//! Every key and every shape is checked against the lowering before a byte is written, an
//! unknown key is refused rather than ignored, and nothing is ever dropped, padded or
//! reshaped to make a checkpoint fit. `safetensors` only; no pickle path exists (`INV-16`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use es_compile::PolicyBundle;
use es_data::rl_import::{Adapter, ImportManifest};
use es_ir::graph::Graph;
use es_ir::learning::{ActionExecutionMode, LearningGraph, WeightsRef};
use es_ir::serial::{deployment_from_toml, observation_from_toml, task_from_toml};
use es_policy::lerobot::{act_policy, remap_checkpoint, ActConfig};
use es_policy::lower::{lower_to_torch, Contract};
use es_policy::weights::{parse_header, validate_keys, weights_hash};
use es_policy::PolicyError;

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es policy lower --policy <in.esb> --out <dir>
es policy pack  --policy <in.esb> --weights <model.safetensors> --out <out.esb>
es policy import-lerobot --checkpoint <dir> --task <t.toml> --observation <o.toml>
                         --deployment <d.toml> --out <out.esb>
es policy import-rl --manifest <import.json> --weights <w.safetensors>
                    --adapter <adapter.toml> --task <t.toml> --deployment <d.toml>
                    --out <dir>

The Rust half of the spec 2.3 training split. No subcommand needs Python.

lower   Opens the policy bundle (spec 9.6), lowers its Learning IR to PyTorch
        (`es_policy::lower_to_torch`, spec 8.7) and writes, under <dir>:
          es_policy.py    the generated module, verbatim -- the same bytes
                          `es eval run` re-lowers and infers with
          contract.json   { lowering_hash, weight_keys, weight_shapes,
                            action_dim, horizon, inputs }
        `python/es/train_act.py --module <dir>` optimizes that module and nothing else:
        the architecture comes from the IR or it does not come.

pack    Reads <model.safetensors>, checks every key and every shape against the same
        lowering, and rewrites the bundle with the new weights and a recomputed manifest
        (spec 5.3 -- the manifest is rebuilt by `PolicyBundle::build`, never patched, so
        `learning_hash` and `policy_hash` move because the weights moved). A missing key,
        an unknown key or a disagreeing shape is named and refused; nothing is dropped,
        padded or reshaped. The Task, Observation and Deployment IR are carried across
        unchanged, so `es eval run --policy <out.esb>` loads it with no change to the
        eval path.

import-lerobot
        Admits a policy designed and trained **outside** this project -- a LeRobot ACT
        checkpoint (config.json + model.safetensors) -- into a bundle `es eval run` loads
        with no change to the eval path (spec 8.9's M1 gate, packet M5/V8). The LeRobot
        keys are remapped into this project's scheme and the training-only CVAE tensors
        dropped (`es_policy::lerobot`; safetensors only, INV-16), and the checkpoint's own
        config.json travels in the output's safetensors `__metadata__`, because spec 8.3's
        node parameters cannot express a CVAE and a DETR decoder and the architecture has
        to reach the runtime somehow.

        The Learning IR written into the bundle is spec 8.1's shape for an external policy:
        an opaque PolicyHandle with a fully typed contract, and no preprocessor nodes --
        the Observation IR owns all of that. Chunk scheduling (execute_chunk,
        replanning_hz, execution mode, deadline) comes from the Deployment IR, which spec 9
        says owns it; chunk_size must already equal the deployment's horizon, and a
        checkpoint that disagrees is named and refused rather than reshaped.

import-rl
        Admits a PPO actor trained by brax / MuJoCo Playground, rsl_rl or rl_games
        (spec 14.4 \"RL policy import\", packet M8/S2b). `python/es/import_rl.py` has
        already turned the checkpoint into <w.safetensors> with framework-neutral keys
        and an <import.json> manifest -- the pickle and the orbax reader live there and
        only there (INV-16). This half reads those two plus a per-robot **adapter
        document**, and writes under <dir>:
          observation.toml     StateInput per adapter channel -> Concat -> Normalize
          learning.toml        StateEncoder{Mlp} -> PolicyHead{Regression, squash}
                               -> ActionChunker -> Normalizer{Inverse}
          policy.esb           the bundle, weights remapped onto the lowered keys
          mapping-report.json  spec 14.4's Semantic Mapping Report: one row per joint
                               and per observation channel
        Nothing is guessed: joint order, units, action kind and the observation layout
        come from the adapter or the import is refused by name (`IMP-001` .. `IMP-005`).
        The Task IR's own `scene.path` is opened, repository-relative, for the actuator
        names `IMP-002` checks against.

Exit codes: 0 success, 1 runtime failure, 2 usage error.
";

pub fn dispatch(args: &[String]) -> i32 {
    let result = match args.first().map(String::as_str) {
        Some("lower") => lower(&args[1..]),
        Some("pack") => pack(&args[1..]),
        Some("import-lerobot") => import_lerobot(&args[1..]),
        Some("import-rl") => import_rl(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            return 0;
        }
        Some(other) => Err(CliError::Usage(format!(
            "es policy: unknown subcommand '{other}'\n\n{HELP}"
        ))),
    };
    match result {
        Ok(code) => i32::from(code),
        Err(CliError::Usage(m)) => {
            eprintln!("{m}");
            2
        }
        Err(CliError::Runtime(m)) => {
            eprintln!("error: {m}");
            1
        }
    }
}

/// `--flag value` pairs, rejecting anything unexpected — the same shape `es loop` uses.
fn parse(args: &[String], flags: &[&str]) -> Result<BTreeMap<String, String>, CliError> {
    let mut out = BTreeMap::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--help" || a == "-h" {
            return Err(CliError::Usage(HELP.to_owned()));
        }
        if !flags.contains(&a.as_str()) {
            return Err(CliError::Usage(format!("unknown flag '{a}'\n\n{HELP}")));
        }
        let value = it
            .next()
            .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{HELP}")))?;
        out.insert(a.clone(), value.clone());
    }
    for flag in flags {
        if !out.contains_key(*flag) {
            return Err(CliError::Usage(format!("{flag} is required\n\n{HELP}")));
        }
    }
    Ok(out)
}

fn open_bundle(path: &str) -> Result<PolicyBundle, CliError> {
    let bytes = std::fs::read(path).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    PolicyBundle::open(&bytes).map_err(|e| CliError::Runtime(format!("{path}: {e}")))
}

pub(crate) fn lower(args: &[String]) -> Result<u8, CliError> {
    let a = parse(args, &["--policy", "--out"])?;
    let bundle = open_bundle(&a["--policy"])?;
    let module = lower_to_torch(&bundle.learning)
        .map_err(|e| CliError::Runtime(format!("lowering the Learning IR: {e}")))?;
    let contract = Contract::new(&module, &bundle.learning);

    let dir = PathBuf::from(&a["--out"]);
    std::fs::create_dir_all(&dir)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", dir.display())))?;
    let write = |name: &str, body: &str| -> Result<(), CliError> {
        let path = dir.join(name);
        std::fs::write(&path, body)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))
    };
    write("es_policy.py", &module.source)?;
    write(
        "contract.json",
        &serde_json::to_string_pretty(&contract)
            .map_err(|e| CliError::Runtime(format!("contract.json: {e}")))?,
    )?;

    println!("module:        {}", dir.join("es_policy.py").display());
    println!("contract:      {}", dir.join("contract.json").display());
    println!("lowering_hash: {}", contract.lowering_hash);
    println!(
        "weights:       {} keys ({} exact, {} prefix claims)",
        contract.weight_keys.len(),
        contract.weight_shapes.len(),
        contract.weight_keys.len() - contract.weight_shapes.len()
    );
    Ok(0)
}

pub(crate) fn pack(args: &[String]) -> Result<u8, CliError> {
    let a = parse(args, &["--policy", "--weights", "--out"])?;
    let bundle = open_bundle(&a["--policy"])?;
    let module = lower_to_torch(&bundle.learning)
        .map_err(|e| CliError::Runtime(format!("lowering the Learning IR: {e}")))?;

    let weights_path = &a["--weights"];
    let weights = std::fs::read(weights_path)
        .map_err(|e| CliError::Runtime(format!("{weights_path}: {e}")))?;
    // Before anything is written: the file is safetensors (never a pickle, `INV-16`) and it is
    // the checkpoint *this* lowering declared (spec 25.1).
    let header =
        parse_header(&weights).map_err(|e| CliError::Runtime(format!("{weights_path}: {e}")))?;
    validate_keys(&module, &header).map_err(|e| CliError::Runtime(mismatch(weights_path, &e)))?;

    // The declaration has to move with the bytes, or `PolicyBundle::build` refuses them and
    // `TorchRuntime::load` would refuse them again at run time (spec 5.3).
    let mut learning = bundle.learning.clone();
    learning.policy.weights = WeightsRef::Safetensors {
        path: learning.policy.weights.path().to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let packed = PolicyBundle::build(
        &bundle.task,
        &bundle.observation,
        &learning,
        &bundle.deployment,
        &weights,
    )
    .map_err(|e| CliError::Runtime(format!("rebuilding the bundle: {e}")))?;

    let out = PathBuf::from(&a["--out"]);
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", parent.display())))?;
    }
    std::fs::write(&out, &packed)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;

    // Reopened rather than trusted: `open` re-validates the four IRs, re-runs the cross-IR
    // pass and recomputes every hash, which is the only honest way to say it round-trips.
    let reopened = PolicyBundle::open(&packed)
        .map_err(|e| CliError::Runtime(format!("the bundle just written does not open: {e}")))?;
    println!("bundle:        {}", out.display());
    println!("tensors:       {}", header.len());
    println!("weights_hash:  {}", hex(learning.policy.weights.hash()));
    println!("lowering_hash: {}", hex(&module.lowering_hash));
    for (slot, value) in [
        ("task", reopened.manifest.hashes.task),
        ("observation", reopened.manifest.hashes.observation),
        ("learning", reopened.manifest.hashes.learning),
        ("policy", reopened.manifest.hashes.policy),
    ] {
        if let Some(h) = value {
            println!("{slot}_hash: {}", hex(&h));
        }
    }
    Ok(0)
}

/// The bytes of the preprocessor's `normalizer_processor` state file, when the checkpoint has
/// one (`docs/api-notes/lerobot-act.md`: `LeRobot` 0.6.x's two checkpoint layouts).
///
/// `policy_preprocessor.json` names it, so the step index is read rather than guessed — it is
/// `_step_3_` for ACT today and a pipeline change would move it.
fn normalizer_state(dir: &str) -> Result<Option<Vec<u8>>, CliError> {
    let Ok(raw) = std::fs::read_to_string(format!("{dir}/policy_preprocessor.json")) else {
        return Ok(None);
    };
    let pipeline: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| CliError::Runtime(format!("{dir}/policy_preprocessor.json: {e}")))?;
    let Some(name) = pipeline["steps"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|s| s["registry_name"] == "normalizer_processor")
        .and_then(|s| s["state_file"].as_str())
    else {
        return Ok(None);
    };
    std::fs::read(format!("{dir}/{name}"))
        .map(Some)
        .map_err(|e| CliError::Runtime(format!("{dir}/{name}: {e}")))
}

/// `es policy import-lerobot` — a policy designed and trained outside, under our Deployment IR.
///
/// The packet's whole point (`docs/packets/M5/V8-external-act.md`): spec 8 says "we do not
/// invent a proprietary policy architecture", and the falsifiable form of that is a real
/// `lerobot-train` ACT running through `es eval run`, the Safety Plane and the Evaluation IR
/// with no change to any of them.
pub(crate) fn import_lerobot(args: &[String]) -> Result<u8, CliError> {
    let a = parse(
        args,
        &[
            "--checkpoint",
            "--task",
            "--observation",
            "--deployment",
            "--out",
        ],
    )?;
    let read = |flag: &str| -> Result<String, CliError> {
        std::fs::read_to_string(&a[flag])
            .map_err(|e| CliError::Runtime(format!("{}: {e}", a[flag])))
    };

    let dir = &a["--checkpoint"];
    let config_raw = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| CliError::Runtime(format!("{dir}/config.json: {e}")))?;
    let cfg = ActConfig::parse(&config_raw)
        .map_err(|e| CliError::Runtime(format!("{dir}/config.json: {e}")))?;
    let original = std::fs::read(format!("{dir}/model.safetensors"))
        .map_err(|e| CliError::Runtime(format!("{dir}/model.safetensors: {e}")))?;

    let task = task_from_toml(&read("--task")?)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a["--task"])))?;
    let observation = observation_from_toml(&read("--observation")?)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a["--observation"])))?;
    let deployment = deployment_from_toml(&read("--deployment")?)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a["--deployment"])))?;

    // `lower_act` returns `actions[0][:n_action_steps]`, and the runtime buffers a whole
    // `horizon`-row chunk. LeRobot's `n_action_steps` is the *scheduling* number, which spec 9
    // gives to the Deployment IR — so the module has to hand back everything it predicted, and
    // the training run has to have said so. Refused, never reshaped.
    let action = deployment.action;
    if cfg.chunk_size as usize != action.horizon || cfg.n_action_steps != cfg.chunk_size {
        return Err(CliError::Runtime(format!(
            "the checkpoint predicts chunk_size {} and returns n_action_steps {}; this \
             deployment buffers a horizon of {} and executes {} of it.\nTrain with \
             `--policy.chunk_size={} --policy.n_action_steps={}`: the Deployment IR owns the \
             execution cadence (spec 9.2), the policy owns the prediction length (spec 8.5).",
            cfg.chunk_size,
            cfg.n_action_steps,
            action.horizon,
            action.execute_chunk,
            action.horizon,
            action.horizon,
        )));
    }

    // LeRobot 0.6.x moved normalization out of `ACTPolicy` into a processor pipeline, so a
    // checkpoint `lerobot-train` writes today keeps the statistics in a separate state file that
    // `policy_preprocessor.json` names. An older checkpoint keeps them in `model.safetensors`
    // and has no such file; both layouts load, and neither needs a flag.
    let stats = normalizer_state(dir)?;
    let remapped = remap_checkpoint(&cfg, &original, stats.as_deref())
        .map_err(|e| CliError::Runtime(format!("remapping the checkpoint: {e}")))?;
    let mut policy = act_policy(&cfg, dir, weights_hash(&original), weights_hash(&remapped))
        .map_err(|e| CliError::Runtime(format!("the checkpoint's config.json: {e}")))?;

    // The checkpoint decides the *shape* of each input; the Observation IR decides its **unit**
    // (spec 5.1 rule 6 and spec 7.4: preprocessing is the Observation IR's, and the unit is what
    // preprocessing produced). `act_policy` can only guess the unit from `config.json`, which
    // records none, so the contract adopts the one the Observation IR declares — after checking
    // that the two agree about what they are talking about, which is a better error here than
    // `XIR-010`'s at the far end.
    for (name, port) in &mut policy.contract.inputs {
        let produced = observation.outputs.get(name).ok_or_else(|| {
            CliError::Runtime(format!(
                "the checkpoint wants an input named \"{name}\"; {} produces {:?}.\nThe names \
                 are the checkpoint's feature names with dots replaced by underscores (spec 8.4, \
                 XIR-010).",
                a["--observation"],
                observation.outputs.keys().collect::<Vec<_>>()
            ))
        })?;
        if produced.ty.elem != port.ty.elem || produced.ty.shape != port.ty.shape {
            return Err(CliError::Runtime(format!(
                "observation output \"{name}\" is {:?}{:?} but the checkpoint declares {:?}{:?}",
                produced.ty.elem,
                produced.ty.shape.dims(),
                port.ty.elem,
                port.ty.shape.dims()
            )));
        }
        port.ty.unit = produced.ty.unit.clone();
    }

    // Spec 9 owns the cadence; `act_policy` leaves these at zero because a `config.json` has no
    // control rate to read them from.
    let control_hz = deployment.rate.control.as_hz_f64();
    policy.contract.execute_chunk = action.execute_chunk as u32;
    policy.contract.replanning_hz = (control_hz / action.execute_chunk.max(1) as f64) as f32;
    policy.contract.execution_mode = match deployment.execution {
        es_ir::deployment::ExecutionMode::OpenLoopChunk => ActionExecutionMode::OpenLoopChunk,
        es_ir::deployment::ExecutionMode::RecedingHorizon => ActionExecutionMode::RecedingHorizon,
        es_ir::deployment::ExecutionMode::TemporalEnsemble { .. } => {
            ActionExecutionMode::TemporalEnsemble
        }
        es_ir::deployment::ExecutionMode::RealTimeChunking => ActionExecutionMode::RealTimeChunking,
    };
    let budget_ms = deployment.deadlines.inference_budget.0 as f32 / 1000.0;
    policy.contract.runtime.deadline_ms = budget_ms;
    policy.contract.runtime.expected_latency_ms =
        declared_latency_ms(budget_ms, policy.contract.replanning_hz);
    policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: weights_hash(&remapped),
    };

    // Spec 8.3, verbatim: "pi0 3.5B is not decomposed into nodes. It is referenced whole via
    // `PolicyBundle`, but its input/output contract (spec 8.4) is type-checked." That is this
    // situation — the spec's own risk register even names it as the fallback for "Learning IR
    // cannot express real policies" — so the graph is one `PolicyBundle` node carrying the
    // contract's ports, and no decomposition is invented.
    //
    // The graph *boundary* stays empty, and that is the deliberate part. Spec 5.4's "a network
    // must not be fed raw" is enforced on `LearningGraph::inputs`, the ports that feed
    // IR-described nodes; a whole-VLA reference has none, and its normalization is inside the
    // referenced artefact — for ACT, `normalize_inputs`, which our lowering carries as nodes
    // 9/10 and which is the first operation of the forward pass. What is type-checked is the
    // contract, by `XIR-010` against the Observation IR's outputs, exactly as spec 8.3 says.
    let mut nodes = Graph::new(1);
    nodes.insert(
        es_ir::graph::NodeId(0),
        es_ir::learning::LearningNode::PolicyBundle {
            inputs: policy.contract.inputs.values().cloned().collect(),
            weights: policy.weights.clone(),
            action_dim: policy.contract.action_dim,
            horizon: policy.contract.horizon,
        },
    );
    let learning = LearningGraph {
        schema_version: 1,
        inputs: Vec::new(),
        nodes,
        outputs: Vec::new(),
        policy,
    };

    let bytes = PolicyBundle::build(&task, &observation, &learning, &deployment, &remapped)
        .map_err(|e| CliError::Runtime(format!("building the bundle: {e}")))?;
    let out = PathBuf::from(&a["--out"]);
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", parent.display())))?;
    }
    std::fs::write(&out, &bytes)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;

    // Reopened rather than trusted, like `pack`: `open` re-validates the four IRs, re-runs the
    // cross-IR pass and recomputes every hash.
    let reopened = PolicyBundle::open(&bytes)
        .map_err(|e| CliError::Runtime(format!("the bundle just written does not open: {e}")))?;
    println!("bundle:        {}", out.display());
    println!(
        "source:        {dir} ({} tensors -> {})",
        parse_header(&original).map_or(0, |h| h.len()),
        parse_header(&remapped).map_or(0, |h| h.len())
    );
    println!("source_hash:   {}", hex(&weights_hash(&original)));
    println!("weights_hash:  {}", hex(&weights_hash(&remapped)));
    println!(
        "contract:      action_dim {} horizon {} execute_chunk {} replanning_hz {}",
        learning.policy.contract.action_dim,
        learning.policy.contract.horizon,
        learning.policy.contract.execute_chunk,
        learning.policy.contract.replanning_hz
    );
    for (slot, value) in [
        ("task", reopened.manifest.hashes.task),
        ("observation", reopened.manifest.hashes.observation),
        ("learning", reopened.manifest.hashes.learning),
        ("policy", reopened.manifest.hashes.policy),
        ("deployment", reopened.manifest.hashes.deployment),
    ] {
        if let Some(h) = value {
            println!("{slot}_hash: {}", hex(&h));
        }
    }
    Ok(0)
}

/// `es policy import-rl` — a PPO actor trained elsewhere, under our documents (spec 14.4).
///
/// The division of labour is the packet's (M8/S2b): Python opened the checkpoint and left a
/// neutral `safetensors` + `import.json`, a human wrote the adapter, and this reads the three
/// and builds the two documents, the bundle and the Semantic Mapping Report. Every mapping
/// decision is the adapter's; every refusal names its `IMP-0xx` code and writes nothing.
pub(crate) fn import_rl(args: &[String]) -> Result<u8, CliError> {
    let a = parse(
        args,
        &[
            "--manifest",
            "--weights",
            "--adapter",
            "--task",
            "--deployment",
            "--out",
        ],
    )?;
    let read = |flag: &str| -> Result<String, CliError> {
        std::fs::read_to_string(&a[flag])
            .map_err(|e| CliError::Runtime(format!("{}: {e}", a[flag])))
    };
    let named = |flag: &str, e: es_data::rl_import::ImportError| {
        CliError::Runtime(format!("{}: {e}", a[flag]))
    };

    let manifest =
        ImportManifest::parse(&read("--manifest")?).map_err(|e| named("--manifest", e))?;
    let adapter = Adapter::parse(&read("--adapter")?).map_err(|e| named("--adapter", e))?;
    let task = task_from_toml(&read("--task")?)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a["--task"])))?;
    let deployment = deployment_from_toml(&read("--deployment")?)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a["--deployment"])))?;
    let weights = std::fs::read(&a["--weights"])
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a["--weights"])))?;

    // `IMP-002` checks the adapter's joint names against the actuators the scene actually has,
    // so the scene is read here rather than guessed at. Repository-relative, the same contract
    // `es train` states for the same field.
    let scene_path = task.scene.path.clone();
    let xml = std::fs::read_to_string(&scene_path).map_err(|e| {
        CliError::Runtime(format!(
            "{scene_path}: {e}\nThe Task IR's `scene.path` is repository-relative; run \
             `es policy import-rl` from the repository root."
        ))
    })?;
    let scene = es_assets::parse_mjcf(&xml)
        .map_err(|e| CliError::Runtime(format!("{scene_path}: {e}")))?
        .scene;
    let actuators: Vec<String> = scene.actuators.iter().map(|x| x.name.clone()).collect();

    let imported = match es_data::rl_import::convert(
        &manifest,
        &adapter,
        &task,
        &deployment,
        &weights,
        &actuators,
    ) {
        Ok(imported) => imported,
        Err(e) => {
            // Decided before the first file, like `es import roboverse`: a refusal leaves no
            // half-written output directory behind.
            eprintln!("error: {e}");
            return Ok(1);
        }
    };

    let mut learning = imported.learning;
    learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: weights_hash(&imported.weights),
    };
    let bundle = PolicyBundle::build(
        &task,
        &imported.observation,
        &learning,
        &deployment,
        &imported.weights,
    )
    .map_err(|e| CliError::Runtime(format!("building the bundle: {e}")))?;

    let dir = PathBuf::from(&a["--out"]);
    std::fs::create_dir_all(&dir)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", dir.display())))?;
    let write = |name: &str, body: &[u8]| -> Result<(), CliError> {
        let path = dir.join(name);
        std::fs::write(&path, body)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))
    };
    write(
        "observation.toml",
        es_ir::serial::observation_to_toml(&imported.observation)
            .map_err(|e| CliError::Runtime(e.to_string()))?
            .as_bytes(),
    )?;
    write(
        "learning.toml",
        es_ir::serial::learning_to_toml(&learning)
            .map_err(|e| CliError::Runtime(e.to_string()))?
            .as_bytes(),
    )?;
    write("policy.esb", &bundle)?;
    write(
        "mapping-report.json",
        serde_json::to_string_pretty(&imported.report)
            .map_err(|e| CliError::Runtime(format!("mapping-report.json: {e}")))?
            .as_bytes(),
    )?;

    // Reopened rather than trusted, like `pack` and `import-lerobot`.
    let reopened = PolicyBundle::open(&bundle)
        .map_err(|e| CliError::Runtime(format!("the bundle just written does not open: {e}")))?;
    for w in &imported.report.warnings {
        println!("warning: {w}");
    }
    println!("out:           {}", dir.display());
    println!(
        "source:        {} ({} joints, {} channels, obs_dim {}, hidden {:?}, {} + {})",
        imported.report.framework,
        imported.report.joints.len(),
        imported.report.channels.len(),
        imported.report.obs_dim,
        imported.report.hidden,
        imported.report.activation,
        imported.report.squash,
    );
    println!(
        "log_std:       {}",
        imported.report.log_std.as_ref().map_or_else(
            || "null (the source carries no state-independent log-std; \
                 `train_ppo.py --init-log-std` is not fed from this import)"
                .to_owned(),
            |v| format!("{v:?} (training-only; it never enters the bundle)"),
        )
    );
    println!("weights_hash:  {}", hex(learning.policy.weights.hash()));
    for (slot, value) in [
        ("task", reopened.manifest.hashes.task),
        ("observation", reopened.manifest.hashes.observation),
        ("learning", reopened.manifest.hashes.learning),
        ("policy", reopened.manifest.hashes.policy),
        ("deployment", reopened.manifest.hashes.deployment),
    ] {
        if let Some(h) = value {
            println!("{slot}_hash: {}", hex(&h));
        }
    }
    Ok(0)
}

/// The mismatch, key by key. `PolicyError`'s own `Display` counts them; a refusal that does
/// not say *which* key is a refusal the caller cannot act on (spec 17.2).
fn mismatch(path: &str, error: &PolicyError) -> String {
    let PolicyError::WeightMismatch {
        missing,
        unexpected,
        shape,
    } = error
    else {
        return format!("{path}: {error}");
    };
    let mut lines = vec![format!("{path} does not fit the lowered Learning IR:")];
    for key in missing {
        lines.push(format!("  missing key    {key}"));
    }
    for key in unexpected {
        lines.push(format!("  unexpected key {key}"));
    }
    for detail in shape {
        lines.push(format!("  wrong shape    {detail}"));
    }
    lines.push(
        "nothing was written: a checkpoint is never dropped, padded or reshaped to fit".to_owned(),
    );
    lines.join("\n")
}

/// What `import-lerobot` may honestly declare for `RuntimeHints::expected_latency_ms`.
///
/// `es_ir::learning::RuntimeHints` documents that field as "a measurement on the reference
/// device of spec 0.3, not a promise", and a `config.json` carries no measurement, so what the
/// import declares is a **bound**: the largest latency this deployment tolerates, which is the
/// tighter of its `deadlines.inference_budget` and its own re-plan period (spec 8.4, `LRN-052`,
/// which checks exactly `latency <= min(deadline, 1000 / replanning_hz)`).
///
/// Until packet M5/V19 this was the budget alone. That is the same number whenever the budget
/// fits inside the re-plan period -- V8's 40 ms against 200 ms -- and a contradiction the
/// moment it does not: V17 states a 240 ms `inference_budget` for the 5 Hz re-plan the same
/// document declares, and `LRN-052` then refused every imported checkpoint for claiming a
/// latency that deployment's own rate forbids. The refusal was about the invented number, not
/// about the policy.
///
/// So `LRN-052` has nothing left to catch on this path, and that is the honest state of it: a
/// rule cannot check a number the same command made up. A measured latency would come from
/// `es bench` (spec 12.4) and is not something a checkpoint directory can supply.
fn declared_latency_ms(budget_ms: f32, replanning_hz: f32) -> f32 {
    budget_ms.min(1000.0 / replanning_hz)
}

#[cfg(test)]
mod tests {
    use super::declared_latency_ms;

    /// The two plan-V deployments, and the rule `LRN-052` applies to the result.
    #[test]
    fn the_declared_latency_is_the_tighter_of_the_budget_and_the_replan_period() {
        // Compared as bits: every value here is exactly representable, and the point of the
        // assertion is that the number did not move at all (clippy's `float_cmp`).
        // V8's deployment: a 40 ms inference budget against a 200 ms re-plan period. The
        // budget is the bound, and this is the value V8 imported.
        assert_eq!(declared_latency_ms(40.0, 5.0).to_bits(), 40.0f32.to_bits());
        // V17's: a 240 ms budget for the same 5 Hz re-plan. The period is the bound.
        assert_eq!(
            declared_latency_ms(240.0, 5.0).to_bits(),
            200.0f32.to_bits()
        );
        // Whatever the two numbers are, the result never exceeds either -- which is what
        // `LRN-052` checks (`crates/es-ir/src/learning.rs`: latency > min(deadline, replan)).
        for (budget, hz) in [(40.0, 5.0), (240.0, 5.0), (20.0, 50.0), (1000.0, 1.0)] {
            let latency = declared_latency_ms(budget, hz);
            assert!(latency <= budget && latency <= 1000.0 / hz, "{budget} {hz}");
        }
    }
}
