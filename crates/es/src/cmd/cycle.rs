//! `es loop cycle` — collect, train, evaluate and showcase under one document and one ledger
//! (spec 13.1, spec 13.3; packet M7/T2, design note `docs/design/training-recipe.md`).
//!
//! Thin, like `es train` and for the same reason: the cycle document, the stage plan and the
//! ledger steps are `es_data::training` and `es_data::collect`, which are headless and
//! unit-tested. What is here is the order, the refusals that need a file open, and the
//! in-process calls. **Every stage is the function the command already is** —
//! `cmd::r#loop::collect`, `cmd::eval::run`, `cmd::train::run`, `cmd::showcase::run` — called
//! with the same words the plan prints, so a printed line and an executed stage cannot drift.
//! Only what those already spawn (the physics subprocess, the trainer, `--jobs` workers) is a
//! process.

use std::path::{Path, PathBuf};

use es_compile::PolicyBundle;
use es_data::collect::{
    append_loop_step, last_evaluation_hash, read_loop_steps, LoopKind, LoopStep, CHECKPOINT,
};
use es_data::training::{Cycle, CyclePlan, CycleStep, Stage};
use es_data::{DatasetIdentity, LeRobotDataset, Split};
use es_ir::evaluation::{AcceptanceResult, EvaluationReport};
use serde_json::Value;

use crate::error::CliError;
use crate::util::hex;

pub const HELP: &str = "\
es loop cycle --recipe <cycle.toml> [--out <dir>] [--dry-run] [--from <stage>]
              [--allow-new-evaluation] [--skip-expert-gate]

Runs spec 13.1's loop -- collect, train, evaluate, showcase -- from one document, and appends
every stage to `loop.jsonl` with the hashes spec 13.3 wants: the dataset a policy was trained
on, and the checkpoint a report judged.

The document names the stages; it does not re-describe them. `[train]` is `es train`'s recipe
by path or inline (its `[dataset]` is overridden by this cycle's collect output), `[eval]` is
an Evaluation IR by path, `[collect]` and `[showcase]` are optional. Every stage runs
in-process: nothing here re-implements what `es loop collect`, `es train`, `es eval run` or
`es video showcase` compute.

  kind = \"cycle\"
  scene = \"tests/fixtures/mjcf/so101_pick_place.xml\"
  [collect]  policy, expert, episodes, seed, frames     # or  dataset = \"<root>\"
  [train]    recipe = \"training.toml\"
  [eval]     config, checkpoint = \"last\", jobs, frames
  [showcase] cell, eye, look_at, fov, width, height

Two refusals are the point of the command:

* **The harness passes the expert first** (spec 28.9 rule 1). With `[collect] expert` set, the
  expert is run through `es eval run` on the *same* `[eval] config` before anything trains, and
  a failed acceptance stops the cycle: a harness the expert fails is a harness no policy can
  pass. `--skip-expert-gate` runs anyway and records the deviation in the ledger.
* **A moved `evaluation_hash` is refused by name** (spec 13.3). If `<out>/loop.jsonl` already
  holds an evaluate step under different evaluation conditions, the comparison the ledger
  invites would be invalid; both hashes are printed and `--allow-new-evaluation` is the
  deliberate act that proceeds.

--dry-run     print the stage plan -- one line per command, paths relative to <out>, the
              training plan nested under `train` -- and run nothing.
--from <stage>  collect | train | eval | showcase: resume an interrupted cycle, refusing if
              the earlier stages' outputs are missing or disagree with the ledger.

Exit codes: 0 success, 1 the expert gate or the final acceptance failed (both printed) or a
runtime failure, 2 usage error, 3 skipped (a backend or runtime this machine does not have).
";

fn bad(msg: impl Into<String>) -> CliError {
    CliError::Runtime(msg.into())
}

fn read(path: &Path) -> Result<String, CliError> {
    std::fs::read_to_string(path).map_err(|e| bad(format!("{}: {e}", path.display())))
}

pub(crate) fn run(args: &[String]) -> Result<u8, CliError> {
    let (mut document, mut out, mut from) = (None, None, None);
    let (mut dry_run, mut allow_new, mut skip_gate) = (false, false, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let slot = match a.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(0);
            }
            "--dry-run" => {
                dry_run = true;
                continue;
            }
            "--allow-new-evaluation" => {
                allow_new = true;
                continue;
            }
            "--skip-expert-gate" => {
                skip_gate = true;
                continue;
            }
            "--recipe" => &mut document,
            "--out" => &mut out,
            "--from" => &mut from,
            other => {
                return Err(CliError::Usage(format!(
                    "es loop cycle: unknown flag '{other}'\n\n{HELP}"
                )))
            }
        };
        *slot = Some(
            it.next()
                .ok_or_else(|| {
                    CliError::Usage(format!("es loop cycle: {a} needs a value\n\n{HELP}"))
                })?
                .clone(),
        );
    }
    let Some(document) = document else {
        return Err(CliError::Usage(format!(
            "es loop cycle: --recipe <cycle.toml> is required\n\n{HELP}"
        )));
    };
    let document = PathBuf::from(document);
    let out = out.map_or_else(
        || PathBuf::from(document.file_stem().unwrap_or_default()),
        PathBuf::from,
    );
    let from = match &from {
        Some(name) => Some(Stage::parse(name).ok_or_else(|| {
            CliError::Usage(format!(
                "es loop cycle: --from {name:?} is not a stage; it is collect, train, eval or \
                 showcase\n\n{HELP}"
            ))
        })?),
        None => None,
    };

    let cycle = Cycle::parse(&read(&document)?).map_err(|e| bad(e.to_string()))?;
    let recipe_text = match &cycle.train.recipe {
        Some(path) => Some(read(Path::new(path))?),
        None => None,
    };
    let recipe = cycle
        .training(recipe_text.as_deref(), &out)
        .map_err(|e| bad(e.to_string()))?;
    let train_plan = crate::cmd::train::plan_of(&recipe, &out.join("train"))?;
    let plan =
        CyclePlan::build(&cycle, &recipe, train_plan, &out).map_err(|e| bad(e.to_string()))?;

    // Spec 13.3, before anything runs and `--dry-run` included: the evaluation conditions are
    // a property of the document, so a moved `evaluation_hash` is knowable without a GPU.
    let evaluation_hash = evaluation_hash(&cycle.eval.config)?;
    check_evaluation_hash(&out, &evaluation_hash, allow_new)?;

    if dry_run {
        print!("{}", plan.render(&out));
        return Ok(0);
    }

    std::fs::create_dir_all(&out).map_err(|e| bad(format!("{}: {e}", out.display())))?;
    let dataset_root = PathBuf::from(&plan.dataset_root);
    if let Some(stage) = from {
        check_resume(&plan, &out, &dataset_root, stage)?;
    }

    let mut gate = None;
    let mut code = 0u8;
    for step in &plan.steps {
        if from.is_some_and(|f| step.stage < f) {
            println!("- {} (skipped by --from)", step.stage.as_str());
            continue;
        }
        if step.stage == Stage::ExpertGate && skip_gate {
            println!("- expert-gate (skipped by --skip-expert-gate)");
            gate = Some("skipped (--skip-expert-gate)".to_owned());
            continue;
        }
        println!("$ {}", one_line(step));
        match step.stage {
            Stage::Collect => {
                let c = crate::cmd::r#loop::collect(&step.args)?;
                if c != 0 {
                    return Ok(c);
                }
                mirror_collect(&dataset_root, &out)?;
            }
            Stage::ExpertGate => {
                let c = crate::cmd::eval::run(&step.args)?;
                if c == 3 {
                    return Ok(3);
                }
                let dir = out.join("eval-expert");
                let expert = cycle
                    .collect
                    .as_ref()
                    .and_then(|c| c.expert.clone())
                    .unwrap_or_default();
                let bundle = cycle.collect.as_ref().map(|c| c.policy.clone());
                let step = evaluate_step(&dir, bundle.as_deref(), None, Some(&expert))?;
                append_both(&out, &dataset_root, &step)?;
                if c != 0 {
                    return Err(bad(format!(
                        "the expert did not pass the evaluation harness ({}/report.json): a \
                         harness the expert fails is a harness no policy can pass, so nothing \
                         is trained (spec 28.9 rule 1). Fix the harness, or pass \
                         --skip-expert-gate to train anyway and record the deviation",
                        dir.display()
                    )));
                }
                gate = Some("passed".to_owned());
            }
            Stage::Train => {
                crate::cmd::train::run(&recipe, &out.join("train"), false, None)?;
                let step = train_step(&out.join("train"), &dataset_root, gate.as_deref())?;
                append_both(&out, &dataset_root, &step)?;
            }
            Stage::Eval => {
                // Iteration >= 2 reuses `--out`, so the previous report is moved aside before
                // this one overwrites it -- `es eval compare` needs both to exist.
                let dir = out.join("eval");
                let (report, previous) = (dir.join("report.json"), dir.join("report-prev.json"));
                if report.exists() {
                    std::fs::rename(&report, &previous)
                        .map_err(|e| bad(format!("{}: {e}", previous.display())))?;
                }
                let c = crate::cmd::eval::run(&step.args)?;
                if c == 3 {
                    return Ok(3);
                }
                code = c;
                let policy = hex(&policy_hash_of(&out.join("train"), plan.mark)?);
                let step = evaluate_step(&dir, None, Some(&policy), None)?;
                append_both(&out, &dataset_root, &step)?;
                if previous.exists() {
                    println!("\n$ es eval compare eval/report-prev.json eval/report.json");
                    crate::cmd::eval::compare(&[
                        previous.to_string_lossy().into_owned(),
                        report.to_string_lossy().into_owned(),
                    ])?;
                }
            }
            Stage::Showcase => showcase(&step.args)?,
        }
    }

    let ledger = out.join(es_data::collect::LOOP_FILE);
    es_data::check_chain(&read_loop_steps(&out).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    println!("\nledger: {} (chained)", ledger.display());
    Ok(code)
}

// --- the stages that are not a plain call ----------------------------------------------------

#[cfg(feature = "render")]
fn showcase(args: &[String]) -> Result<(), CliError> {
    crate::cmd::showcase::run(args).map(|_| ())
}

#[cfg(not(feature = "render"))]
fn showcase(_args: &[String]) -> Result<(), CliError> {
    Err(bad(
        "[showcase] needs the `render` feature; this build links no renderer (spec 4.2: \
         es-render is layer 5 and the default build of `es` does not pull it in). Rebuild with \
         `cargo build -p es --features render`, or delete [showcase] from the cycle.",
    ))
}

fn one_line(step: &CycleStep) -> String {
    step.prefix
        .iter()
        .chain(&step.args)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

// --- the ledger (spec 13.3) ------------------------------------------------------------------

/// Appends one step to the cycle's ledger and to the dataset root's, which is the thing that
/// trained or was judged. Two files, one line each, never rewritten.
fn append_both(out: &Path, dataset_root: &Path, step: &LoopStep) -> Result<(), CliError> {
    append_loop_step(out, step).map_err(|e| bad(e.to_string()))?;
    if dataset_root != out {
        append_loop_step(dataset_root, step).map_err(|e| bad(e.to_string()))?;
    }
    Ok(())
}

/// `Collector::run` appends its own step to the dataset's ledger; the cycle's ledger gets the
/// same line, so `<out>/loop.jsonl` holds the whole chain and not only the half this command
/// wrote itself.
fn mirror_collect(dataset_root: &Path, out: &Path) -> Result<(), CliError> {
    let steps = read_loop_steps(dataset_root).map_err(|e| bad(e.to_string()))?;
    let last = steps
        .iter()
        .rev()
        .find(|s| s.kind == LoopKind::Collect)
        .ok_or_else(|| {
            bad(format!(
                "{}: the collect stage wrote no collect step",
                dataset_root.display()
            ))
        })?;
    append_loop_step(out, last).map_err(|e| bad(e.to_string()))
}

/// The `train` step of spec 13.3: what the run read, and what it produced.
fn train_step(
    train_out: &Path,
    dataset_root: &Path,
    gate: Option<&str>,
) -> Result<LoopStep, CliError> {
    let lock = json_of(&train_out.join("training.lock"))?;
    let collect = read_loop_steps(dataset_root)
        .map_err(|e| bad(e.to_string()))?
        .into_iter()
        .rev()
        .find(|s| s.kind == LoopKind::Collect);
    let dataset = json_of(&train_out.join("training").join("dataset.lock"))?;
    let slot = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).unwrap_or("-").to_owned();
    let mut step = LoopStep::new(LoopKind::Train)
        .input("content", &slot(&dataset, "content"))
        .input("schema", &slot(&dataset, "schema"))
        .input("split", &slot(&dataset, "split"))
        .input("identity_hash", &slot(&lock, "identity_hash"))
        .input("dataset", &dataset_root.display())
        .output("training_hash", &slot(&lock, "training_hash"));
    if let Some(gate) = gate {
        // Spec 28.9 rule 1: a cycle that trained without the harness having passed the expert
        // says so in the record, rather than looking like one that did.
        step = step.input("expert_gate", &gate);
    }
    // The ledger's own view of what collection produced, so the chain is checkable from one
    // file even when the dataset came from elsewhere.
    if let Some(c) = collect {
        if let Some(content) = c.outputs.get("content") {
            if *content != slot(&dataset, "content") {
                return Err(bad(format!(
                    "the training run read dataset content {} and collection wrote {content}",
                    slot(&dataset, "content")
                )));
            }
        }
    }
    for ckpt in lock
        .get("checkpoints")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
    {
        let mark = ckpt.get("step").and_then(Value::as_u64).unwrap_or(0);
        let hash = ckpt
            .get("policy_hash")
            .and_then(Value::as_str)
            .unwrap_or("-");
        step = step.output(&format!("{CHECKPOINT}{mark}"), &hash);
    }
    Ok(step)
}

/// The `evaluate` step of spec 13.3: which policy, under which conditions, and what came out.
///
/// `policy_hash` is spec 19.3's `H(training_hash, checkpoint_hash)` for a trained checkpoint —
/// the same number the train step's `checkpoint.<mark>` output carries, which is what makes
/// the chain checkable. The expert gate has no checkpoint, so it names the bundle whose four
/// documents the harness ran with and the expert that drove it.
fn evaluate_step(
    eval_out: &Path,
    bundle_path: Option<&str>,
    policy_hash: Option<&str>,
    expert: Option<&str>,
) -> Result<LoopStep, CliError> {
    let lock = json_of(&eval_out.join("evaluation.lock"))?;
    let path = eval_out.join("report.json");
    let bytes = std::fs::read(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let report: EvaluationReport = serde_json::from_slice(&bytes)
        .map_err(|e| bad(format!("{}: not an EvaluationReport: {e}", path.display())))?;
    let acceptance = report.acceptance.iter().find_map(|r| match r {
        AcceptanceResult::Determined {
            criterion,
            observed,
            ..
        } if criterion.metric.name() == "success_rate" => Some(*observed),
        _ => None,
    });

    let mut step = LoopStep::new(LoopKind::Evaluate)
        .input(
            "evaluation_hash",
            &lock
                .get("evaluation_hash")
                .and_then(Value::as_str)
                .unwrap_or("-"),
        )
        .output("report", &hex(blake3::hash(&bytes).as_bytes()))
        .output("passed", &report.passed);
    if let Some(rate) = acceptance {
        step = step.output("success_rate", &rate);
    }
    if let Some(name) = expert {
        step = step.input("expert", &name);
    }
    if let Some(hash) = policy_hash {
        step = step.input("policy_hash", &hash);
    }
    // The documents the run judged against, from the bundle it was pointed at.
    if let Some(path) = bundle_path {
        let bytes = std::fs::read(path).map_err(|e| bad(format!("{path}: {e}")))?;
        let bundle = PolicyBundle::open(&bytes).map_err(|e| bad(format!("{path}: {e}")))?;
        let h = &bundle.manifest.hashes;
        let slot = |d: Option<[u8; 32]>| d.as_ref().map_or_else(|| "-".to_owned(), hex);
        step = step
            .input("deployment", &slot(h.deployment))
            .input("observation", &slot(h.observation));
        if policy_hash.is_none() {
            step = step.input("policy_hash", &slot(h.policy));
        }
    }
    Ok(step)
}

/// Spec 19.3's `policy_hash` of one checkpoint, out of the lock `es train` wrote.
fn policy_hash_of(train_out: &Path, mark: u32) -> Result<[u8; 32], CliError> {
    let lock = json_of(&train_out.join("training.lock"))?;
    let hex = lock
        .get("checkpoints")
        .and_then(Value::as_array)
        .and_then(|list| {
            list.iter()
                .find(|c| c.get("step").and_then(Value::as_u64) == Some(u64::from(mark)))
        })
        .and_then(|c| c.get("policy_hash"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            bad(format!(
                "{}: no checkpoint at step {mark}",
                train_out.join("training.lock").display()
            ))
        })?;
    from_hex(hex).ok_or_else(|| bad(format!("policy_hash {hex} is not 32 bytes of hex")))
}

fn json_of(path: &Path) -> Result<Value, CliError> {
    serde_json::from_str(&read(path)?).map_err(|e| bad(format!("{}: {e}", path.display())))
}

fn from_hex(text: &str) -> Option<[u8; 32]> {
    let bytes: Vec<u8> = (0..text.len() / 2)
        .filter_map(|i| u8::from_str_radix(text.get(i * 2..i * 2 + 2)?, 16).ok())
        .collect();
    bytes.try_into().ok()
}

// --- the two refusals -------------------------------------------------------------------------

/// The Evaluation IR's own `evaluation_hash` (spec 10.4), from the document alone.
fn evaluation_hash(config: &str) -> Result<String, CliError> {
    let ir = es_ir::serial::evaluation_from_toml(&read(Path::new(config))?)
        .map_err(|e| bad(format!("{config}: {e}")))?;
    let hash = ir.evaluation_hash().map_err(|e| bad(e.to_string()))?;
    Ok(hex(&hash))
}

/// Spec 13.3's discipline, as a refusal: "keeping the evaluation conditions fixed while
/// changing only data·policy is the discipline", and a ledger whose evaluate steps were taken
/// under two different conditions invites a comparison that is not one.
fn check_evaluation_hash(out: &Path, want: &str, allow_new: bool) -> Result<(), CliError> {
    let steps = read_loop_steps(out).map_err(|e| bad(e.to_string()))?;
    let Some(previous) = last_evaluation_hash(&steps) else {
        return Ok(());
    };
    if previous == want || allow_new {
        return Ok(());
    }
    Err(bad(format!(
        "the evaluation conditions moved: {} already holds an evaluate step under \
         evaluation_hash\n  {previous}\nand this cycle's [eval] config hashes to\n  {want}\n\
         Spec 13.3: with a changed evaluation_hash the comparison between the two iterations \
         is invalid. Pass --allow-new-evaluation to start a new comparison here, or point \
         --out at a fresh directory.",
        out.join(es_data::collect::LOOP_FILE).display()
    )))
}

/// `--from <stage>`: the earlier stages' outputs have to be on disk and agree with the ledger.
fn check_resume(
    plan: &CyclePlan,
    out: &Path,
    dataset_root: &Path,
    from: Stage,
) -> Result<(), CliError> {
    if from <= Stage::Collect {
        return Ok(());
    }
    let dataset = LeRobotDataset::open(dataset_root).map_err(|e| {
        bad(format!(
            "--from {}: {}: {e}",
            from.as_str(),
            dataset_root.display()
        ))
    })?;
    let n = dataset.episodes().len() as u32;
    let identity = DatasetIdentity::compute(&dataset, &Split::deterministic(n, [1.0, 0.0, 0.0], 0))
        .map_err(|e| bad(e.to_string()))?;
    let content = hex(&identity.to_dataset_hash().content);
    let steps = read_loop_steps(out).map_err(|e| bad(e.to_string()))?;
    let recorded = steps
        .iter()
        .rev()
        .find(|s| s.kind == LoopKind::Collect)
        .and_then(|s| s.outputs.get("content"));
    if let Some(recorded) = recorded {
        if *recorded != content {
            return Err(bad(format!(
                "--from {}: {} holds dataset content {content} and the ledger's collect step \
                 wrote {recorded}; the stages under {} are not one cycle",
                from.as_str(),
                dataset_root.display(),
                out.display()
            )));
        }
    }
    if from >= Stage::Eval {
        let bundle = out.join(format!("train/checkpoints/{}.esb", plan.mark));
        if !bundle.exists() {
            return Err(bad(format!(
                "--from {}: {} is not on disk; the train stage did not finish",
                from.as_str(),
                bundle.display()
            )));
        }
        policy_hash_of(&out.join("train"), plan.mark)?;
    }
    if from >= Stage::Showcase {
        let report = out.join("eval").join("report.json");
        if !report.exists() {
            return Err(bad(format!(
                "--from showcase: {} is not on disk; the eval stage did not finish",
                report.display()
            )));
        }
    }
    Ok(())
}
