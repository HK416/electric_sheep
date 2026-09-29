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

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

use es_compile::PolicyBundle;
use es_data::collect::{
    append_loop_step, last_evaluation_hash, read_loop_steps, LoopKind, LoopStep, CHECKPOINT,
};
use es_data::training::{
    collect_root, preview_evaluation, Backbone, Cycle, CyclePlan, CycleStep, PreviewRef,
    PreviewStep, Stage,
};
use es_data::{DatasetIdentity, LeRobotDataset, Split};
use es_ir::evaluation::{AcceptanceResult, EvaluationReport};
use serde_json::{json, Value};

use crate::cmd::telemetry::{
    TelemetryArgs, Voice, STAGE_COLLECT, STAGE_CYCLE, STAGE_EVAL, STAGE_EXPERT_GATE, STAGE_PREVIEW,
    STAGE_SHOWCASE, STAGE_TRAIN,
};
use crate::cmd::train::TrainWatch;
use crate::error::CliError;
use crate::util::hex;

pub const HELP: &str = "\
es loop cycle --recipe <cycle.toml> [--out <dir>] [--dry-run] [--from <stage>]
              [--allow-new-evaluation] [--skip-expert-gate]
              [--telemetry <addr>] [--telemetry-token <t>] [--telemetry-image-every <N>]
              [--progress-every <N>]

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
             perturb = { config, suites }, merge = [\"<root>\", ...]     # optional
  [train]    recipe = \"training.toml\"
             init = \"<bundle | lerobot pretrained_model dir>\"           # optional
  [eval]     config, checkpoint = \"last\", jobs, frames
  [eval.preview]  episodes = 4, suite = <the first>, frames = true      # optional
  [showcase] cell, eye, look_at, fov, width, height

Training again on what failed (packet M13/Z3): `perturb` collects under those suites of an
Evaluation IR (`es loop collect --perturb --suites`), and a collection whose seeds meet those
of [eval] config or of `perturb`'s own config is refused, `--dry-run` included (spec 13.3).
`merge` is a merge stage after collect: `es loop distill` of the new dataset and those roots
into <out>/collect/merged, all-train, which is what trains; with `frames`, each root's tiles
are the `frames` beside it and are linked into collect/frames after the new ones. `init` is
where training starts: the IR and RL routes' `[init] policy`, the lerobot route's
`--policy.path`. Both have to be on disk before anything runs (not checked by `--dry-run`).

With [eval.preview], each checkpoint gets a short test as soon as its bundle is on disk, while
the trainer is still running (the last mark after it exits): this same `es`, as a child, runs
`es eval run --jobs 1` on the bundle under a document derived from [eval] config (that suite
alone, its first `episodes` seeds, the same metrics, **no acceptance**), written to
<out>/preview/<mark>/evaluation.toml with the run's artifacts and `eval.log` beside it. One
preview at a time; the eval stage waits for the last. A preview judges nothing (spec 13.3): its
results go to <out>/preview/index.jsonl, one row per mark, and never into loop.jsonl.

A recipe's `[policy] base_model` is checked against the pin before the first stage, and with
`base_model_fetch` a missing one is fetched then (the plan's `# fetch:` line) -- never after
collect.

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
--telemetry <addr>
              publish the whole cycle live on this address (spec 23.1). **One socket, bound
              once**, before the first stage: every stage speaks on it and every event carries
              the stage it came from, bracketed by `stage.begin` / `stage.end`. The address is
              the cycle's, not a stage's, so it is on no line of the plan.
--telemetry-token <t>   required in every client's Hello (spec 25.1); none by default
--telemetry-image-every <N>
              how often a picture goes out: the observation image every N control ticks in
              collect and the evaluations, and the training sample every N optimizer steps.
--progress-every <N>    the trainer's progress lines, in optimizer steps (default 10)

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
    let (mut addr, mut token, mut image_every, mut progress_every) = (None, None, None, None);
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
            "--telemetry" => &mut addr,
            "--telemetry-token" => &mut token,
            "--telemetry-image-every" => &mut image_every,
            "--progress-every" => &mut progress_every,
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
    // Derived before anything runs, so a suite the Evaluation IR does not declare is refused by
    // `--dry-run` too (packet M13/Z1).
    let preview = match &cycle.eval.preview {
        Some(p) => Some(PreviewIr::derive(&cycle.eval.config, p)?),
        None => None,
    };
    check_perturb(&cycle)?;

    if dry_run {
        print!("{}", plan.render(&out));
        return Ok(0);
    }
    cycle
        .check_inputs(plan.train.route)
        .map_err(|e| bad(e.to_string()))?;
    // The step that wrote the data the cycle trains on, for the ledger's own checks.
    let data_kind = if cycle.merge().is_empty() {
        LoopKind::Collect
    } else {
        LoopKind::Distill
    };

    // **One socket for the whole cycle** (packet M7/E7), bound before the first stage opens
    // anything. The stages are in-process calls, so they are handed this publisher rather
    // than a `--telemetry` on their argv -- which is why the address is on no plan line.
    let telemetry = TelemetryArgs {
        addr: match &addr {
            Some(v) => Some(TelemetryArgs::parse_addr(v, HELP)?),
            None => None,
        },
        token,
        image_every: match &image_every {
            Some(v) => TelemetryArgs::parse_image_every(v, HELP)?,
            None => 0,
        },
    };
    let progress_every = match &progress_every {
        Some(v) => v.parse().map_err(|_| {
            CliError::Usage(format!(
                "es loop cycle: --progress-every {v:?} is not a number\n\n{HELP}"
            ))
        })?,
        None => crate::cmd::train::DEFAULT_PROGRESS_EVERY,
    };
    let mut publisher = telemetry.bind(STAGE_CYCLE)?;

    std::fs::create_dir_all(&out).map_err(|e| bad(format!("{}: {e}", out.display())))?;
    let dataset_root = PathBuf::from(&plan.dataset_root);
    if let Some(stage) = from {
        check_resume(&plan, &out, &dataset_root, data_kind, stage)?;
    }
    // Packet M12/R8: the backbone the train stage reads is fetched when missing and checked
    // against the pin now, not after collect and the expert gate (26 minutes on the hint card).
    if from.is_none_or(|f| f <= Stage::Train) {
        crate::cmd::train::fetch_base_model(&plan.train, &recipe)?;
        if let Some(path) = &recipe.policy.base_model {
            Backbone::verify(Path::new(path)).map_err(|e| bad(e.to_string()))?;
        }
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
        // Every event of this stage carries its name, and `stage.begin` / `stage.end` bracket
        // it with the wall-clock the table in the design note is made of.
        if let Some(p) = publisher.as_mut() {
            p.stage_begin(stage_name(step.stage));
        }
        match step.stage {
            Stage::Collect => {
                let c = crate::cmd::r#loop::collect(&step.args, publisher.as_mut())?;
                if c != 0 {
                    return Ok(c);
                }
                mirror(Path::new(&collect_root(&out)), &out, LoopKind::Collect)?;
            }
            Stage::Merge => {
                crate::cmd::r#loop::distill(&step.args)?;
                mirror(&dataset_root, &out, LoopKind::Distill)?;
            }
            Stage::ExpertGate => {
                let c = crate::cmd::eval::run(&step.args, publisher.as_mut())?;
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
                // Packet M13/Z1: one worker previews each mark as its bundle lands, and the stage
                // ends after the last preview -- the eval stage never shares the machine with one.
                let queue = preview.as_ref().map(|ir| {
                    let voice = publisher.as_ref().map(|p| p.voice(STAGE_PREVIEW));
                    Previews::start(plan.previews.clone(), ir.clone(), voice, &out)
                });
                let mut push = |mark: u32| {
                    if let Some(q) = &queue {
                        q.push(mark);
                    }
                };
                let watch = TrainWatch {
                    publisher: publisher.as_mut(),
                    progress_every,
                    // One number for "how often a picture": control ticks in collect and the
                    // evaluations, optimizer steps here.
                    sample_every: telemetry.image_every,
                    on_checkpoint: queue.is_some().then_some(&mut push as &mut dyn FnMut(u32)),
                };
                let trained =
                    crate::cmd::train::run(&recipe, &out.join("train"), false, None, watch);
                if let Some(q) = queue {
                    q.finish();
                }
                trained?;
                let step = train_step(
                    &out.join("train"),
                    &dataset_root,
                    data_kind,
                    gate.as_deref(),
                )?;
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
                let c = crate::cmd::eval::run(&step.args, publisher.as_mut())?;
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
        if let Some(p) = publisher.as_mut() {
            p.stage_end(code);
        }
    }

    let ledger = out.join(es_data::collect::LOOP_FILE);
    es_data::check_chain(&read_loop_steps(&out).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    println!("\nledger: {} (chained)", ledger.display());
    if let Some(p) = &publisher {
        println!("{}", p.summary());
    }
    Ok(code)
}

/// The `stage` field a stage's events carry. `Stage::as_str` is the plan's own word for it,
/// and the wire wants a `'static` one, so the two are matched here rather than allocated.
fn stage_name(stage: Stage) -> &'static str {
    match stage {
        // The merge is the collect stage's second half on the wire: a viewer reads the data as
        // still being made while it runs, and a failed merge as a failed collection.
        Stage::Collect | Stage::Merge => STAGE_COLLECT,
        Stage::ExpertGate => STAGE_EXPERT_GATE,
        Stage::Train => STAGE_TRAIN,
        Stage::Eval => STAGE_EVAL,
        Stage::Showcase => STAGE_SHOWCASE,
    }
}

// --- the stages that are not a plain call ----------------------------------------------------

#[cfg(feature = "render")]
fn showcase(args: &[String]) -> Result<(), CliError> {
    crate::cmd::showcase::run(args).map(|_| ())
}

#[cfg(not(feature = "render"))]
fn showcase(_args: &[String]) -> Result<(), CliError> {
    Err(bad(
        "[showcase] needs the `render` feature; this build links no renderer (it was built \
         with `--no-default-features`). Rebuild with the default features \
         (`cargo build -p es`), or delete [showcase] from the cycle.",
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

/// `Collector::run` and `es_data::distill` append their own step to the dataset's ledger; the
/// cycle's ledger gets the same line, so `<out>/loop.jsonl` holds the whole chain and not only
/// the half this command wrote itself.
fn mirror(root: &Path, out: &Path, kind: LoopKind) -> Result<(), CliError> {
    let steps = read_loop_steps(root).map_err(|e| bad(e.to_string()))?;
    let last = steps.iter().rev().find(|s| s.kind == kind).ok_or_else(|| {
        bad(format!(
            "{}: the stage wrote no {kind:?} step",
            root.display()
        ))
    })?;
    append_loop_step(out, last).map_err(|e| bad(e.to_string()))
}

/// The `train` step of spec 13.3: what the run read, and what it produced. `data` is the kind
/// of step that wrote the dataset: the collection, or the merge (packet M13/Z3).
fn train_step(
    train_out: &Path,
    dataset_root: &Path,
    data: LoopKind,
    gate: Option<&str>,
) -> Result<LoopStep, CliError> {
    let lock = json_of(&train_out.join("training.lock"))?;
    let collect = read_loop_steps(dataset_root)
        .map_err(|e| bad(e.to_string()))?
        .into_iter()
        .rev()
        .find(|s| s.kind == data);
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

// --- the checkpoint previews (packet M13/Z1) ------------------------------------------------

/// The one document every preview of a cycle runs, derived from `[eval] config`.
#[derive(Clone)]
struct PreviewIr {
    /// `evaluation.toml`, as written beside each preview.
    text: String,
    /// Its own `evaluation_hash`, for the index row -- never the cycle's (spec 13.3).
    hash: String,
    suite: String,
}

impl PreviewIr {
    fn derive(config: &str, preview: &PreviewRef) -> Result<Self, CliError> {
        let ir = es_ir::serial::evaluation_from_toml(&read(Path::new(config))?)
            .map_err(|e| bad(format!("{config}: {e}")))?;
        let derived = preview_evaluation(&ir, preview).map_err(|e| bad(e.to_string()))?;
        Ok(Self {
            text: es_ir::serial::evaluation_to_toml(&derived)
                .map_err(|e| bad(format!("the preview's Evaluation IR: {e}")))?,
            hash: hex(&derived.evaluation_hash().map_err(|e| bad(e.to_string()))?),
            suite: derived.suites[0].name.clone(),
        })
    }
}

/// One worker thread, one preview at a time, in the order the marks land.
///
/// ponytail: a killed cycle leaves a running preview child to finish its few episodes alone; a
/// job object (Windows) / process group (Unix) would take it down too, if that ever matters.
struct Previews {
    tx: mpsc::Sender<u32>,
    worker: thread::JoinHandle<()>,
}

impl Previews {
    fn start(steps: Vec<PreviewStep>, ir: PreviewIr, voice: Option<Voice>, out: &Path) -> Self {
        let (tx, rx) = mpsc::channel::<u32>();
        let out = out.to_path_buf();
        let worker = thread::spawn(move || {
            for mark in rx {
                if let Some(step) = steps.iter().find(|s| s.mark == mark) {
                    preview(step, &ir, voice.as_ref(), &out);
                }
            }
        });
        Self { tx, worker }
    }

    /// Queues a mark and returns at once: the trainer may still be running.
    fn push(&self, mark: u32) {
        let _ = self.tx.send(mark);
    }

    /// Waits for the last queued preview.
    fn finish(self) {
        drop(self.tx);
        let _ = self.worker.join();
    }
}

/// One mark's preview: the derived document into `<out>/preview/<mark>/`, the running `es` as
/// the child that evaluates the bundle (its console into `eval.log` there, not into the
/// trainer's), and what came of it said three ways -- a line, `preview.end`, and a row of
/// `<out>/preview/index.jsonl`. A preview that fails is a row with its exit code, never the
/// cycle's failure: it judges nothing.
fn preview(step: &PreviewStep, ir: &PreviewIr, voice: Option<&Voice>, out: &Path) {
    let dir = Path::new(&step.dir);
    let rel = format!("preview/{}", step.mark);
    if let Some(v) = voice {
        v.preview_begin(step.mark, &rel);
    }
    let code = preview_child(step, &ir.text, dir).unwrap_or_else(|e| {
        println!("preview {}: {e}", step.mark);
        1
    });
    let rows = es_eval::episodes::read_episodes(dir)
        .ok()
        .flatten()
        .unwrap_or_default();
    let successes = rows.iter().filter(|r| r.termination == "success").count();
    println!(
        "preview {}: {successes} of {} episode(s) succeeded, exit {code} ({})",
        step.mark,
        rows.len(),
        dir.display()
    );
    if let Some(v) = voice {
        v.preview_end(step.mark, &rel, successes, rows.len(), code);
    }
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let row = json!({
        "step": step.mark,
        "dir": rel,
        "bundle": format!("train/checkpoints/{}.esb", step.mark),
        "suite": ir.suite,
        "evaluation_hash": ir.hash,
        "successes": successes,
        "episodes": rows.len(),
        "code": code,
        "created": created,
    });
    let index = out.join("preview").join("index.jsonl");
    let appended = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&index)
        .and_then(|mut f| writeln!(f, "{row}"));
    if let Err(e) = appended {
        println!("preview {}: {}: {e}", step.mark, index.display());
    }
}

/// `es eval run` on one mark, as a child of this very binary.
fn preview_child(step: &PreviewStep, document: &str, dir: &Path) -> std::io::Result<i32> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("evaluation.toml"), document)?;
    let log = std::fs::File::create(dir.join("eval.log"))?;
    let status = Command::new(std::env::current_exe()?)
        .args(["eval", "run"])
        .args(&step.args)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .status()?;
    Ok(status.code().unwrap_or(-1))
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

/// `[collect] perturb`, refused on the documents before anything runs, `--dry-run` included
/// (packet M13/Z3): a suite `config` does not declare, or collect seeds that meet the seeds of
/// the Evaluation IR this cycle is judged by or of the one it draws from (spec 13.3). Z2's own
/// checks, so the words are `es loop collect`'s.
fn check_perturb(cycle: &Cycle) -> Result<(), CliError> {
    let Some((collect, p)) = cycle
        .collect
        .as_ref()
        .and_then(|c| Some((c, c.perturb.as_ref()?)))
    else {
        return Ok(());
    };
    let open = |config: &str| {
        es_ir::serial::evaluation_from_toml(&read(Path::new(config))?)
            .map_err(|e| bad(format!("{config}: {e}")))
    };
    crate::cmd::r#loop::suite_cells(&open(&p.config)?, &p.config, &p.suites)?;
    for config in [&cycle.eval.config, &p.config] {
        crate::cmd::r#loop::refuse_evaluation_seeds(
            &open(config)?,
            config,
            collect.seed,
            collect.episodes,
        )?;
    }
    Ok(())
}

/// `--from <stage>`: the earlier stages' outputs have to be on disk and agree with the ledger.
/// `data` is the kind of step that wrote `dataset_root`: the collection, or the merge.
fn check_resume(
    plan: &CyclePlan,
    out: &Path,
    dataset_root: &Path,
    data: LoopKind,
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
        .find(|s| s.kind == data)
        .and_then(|s| s.outputs.get("content"));
    if let Some(recorded) = recorded {
        if *recorded != content {
            return Err(bad(format!(
                "--from {}: {} holds dataset content {content} and the ledger's {} step \
                 wrote {recorded}; the stages under {} are not one cycle",
                from.as_str(),
                dataset_root.display(),
                format!("{data:?}").to_lowercase(),
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
