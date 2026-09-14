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
use es_ir::learning::WeightsRef;
use es_policy::lower::{lower_to_torch, Contract};
use es_policy::weights::{parse_header, validate_keys};
use es_policy::PolicyError;

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es policy lower --policy <in.esb> --out <dir>
es policy pack  --policy <in.esb> --weights <model.safetensors> --out <out.esb>

The Rust half of the spec 2.3 training split. Neither subcommand needs Python.

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

Exit codes: 0 success, 1 runtime failure, 2 usage error.
";

pub fn dispatch(args: &[String]) -> i32 {
    let result = match args.first().map(String::as_str) {
        Some("lower") => lower(&args[1..]),
        Some("pack") => pack(&args[1..]),
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

fn lower(args: &[String]) -> Result<u8, CliError> {
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

fn pack(args: &[String]) -> Result<u8, CliError> {
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
