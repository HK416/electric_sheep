//! `es train` — one document, one command, a real `training_hash` (spec 13.1, 19.3;
//! packet M7/T1, design note `docs/design/training-recipe.md`).
//!
//! Thin on purpose. The recipe schema, the command plan and spec 19.3's `training/` bundle
//! live in `es_data::training`, which is headless and unit-tested; what is here is the
//! execution order and the process boundary. Every `es` step of the plan is called
//! **in-process** — they are functions in this module's siblings — and the only subprocess is
//! the Python trainer, which is exactly where spec 2.3 draws the line.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use es_compile::PolicyBundle;
use es_data::training::{
    camera_suffix, has_image_input, has_pretrained_backbone, init_from, init_weights, state_dim,
    Backbone, DatasetFacts, Plan, Recipe, Route, Step, StepKind, Training, TRAIN_ACT,
};
use es_data::{DatasetIdentity, LeRobotDataset, Split};
use es_ir::observation::ObservationIr;
use es_ir::serial::{observation_from_toml, task_from_toml};
use serde_json::{json, Value};

use crate::cmd::telemetry::{Publisher, TelemetryArgs, STAGE_TRAIN};
use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es train --recipe <training.toml> [--out <dir>] [--dry-run] [--allow-retired-task <hex>]
         [--telemetry <addr>] [--telemetry-token <t>] [--telemetry-image-every <N>]
         [--progress-every <N>] [--sample-every <N>]

Runs one training document end to end and writes spec 19.3's `training/` from what the run
actually used (spec 13.1: each step of the loop is a command *and* an artifact).

The recipe names a dataset, exactly one policy -- `[policy] bundle` (this project's Learning
IR) or `[policy] lerobot` (a policy designed outside it) -- and a `[run]` block. Which of the
two it names decides the route:

  bundle   es dataset bake -> es policy lower -> python/es/train_act.py -> es policy pack
  lerobot  es dataset export -> lerobot-train -> es policy import-lerobot

A bundle whose Learning IR declares `VisionEncoder { pretrained = true }` also needs
`[policy] base_model = \"<dir>/resnet18-imagenet1k-v1.safetensors\"`, the artifact
`python/es/fetch_backbone.py` writes. Its blake3 is checked against the lock file beside it
*and* against the pin this build carries, its licence is copied into
training/base_model.lock, and the tensors reach the trainer as `--init-backbone` -- never
over the network at construction (spec 2.5, 19.3).

An optional `[init] policy = \"<bundle.esb>\"` says which policy this run starts from (IR route
only). Its safetensors is compared to the lowered module's contract, the tensors whose name
and shape match are copied into <out>/weights/init.safetensors and reach the trainer as
--init-weights, and what was copied, initialised or left behind over a shape is recorded in
<out>/training/init.lock -- a slot of the run's identity, like base_model.lock. Sharing no
tensor at all is refused. `[run] steps = 0` is legal beside it and means \"checkpoint
immediately\": the module's initial state is written as 0.esb without an optimizer step.

Every `es` step above runs in-process; only the trainer is a subprocess. Run `es train` from
the repository root: the IR route's trainer is `python/es/train_act.py` and a Task IR's
`scene.path` is repository-relative.

--out <dir>   where everything is written; defaults to the recipe's file stem. Holds
              baked/ module/ weights/ checkpoints/ metrics/ (or ds-v3/ lerobot/), plus
              training/ and training.lock.
--dry-run     print the command plan -- one line per step, every path relative to <out> --
              write <out>/training/plan.txt, and run nothing.
--allow-retired-task <hex>
              accept a dataset recorded under a task_hash that is not the policy's, naming
              the hash being accepted (M5 review S-3/R4). Without it the mismatch is refused:
              demonstrations of one predicate and documents of another measure nothing.

--telemetry <addr>
              publish the run live on this address (spec 23.1), e.g. 127.0.0.1:7777. The
              trainer's stdout is then read line by line rather than at exit, and what it
              says goes out on stream 5 as [step, loss, lr, samples_per_s], with a
              `checkpoint` event per packed mark on stream 1 and the sample image on stream 4.
              The summary is still the trainer's last stdout line and still what
              training.lock records.
--telemetry-token <t>
              required in every client's Hello (spec 25.1); none by default
--progress-every <N>
              ask the trainer for one progress line every N optimizer steps (default 10 with
              --telemetry, 0 without)
--sample-every <N>
              ask the trainer to write one image input of the batch, after augmentation, as
              Rgb8 beside metrics/ every N steps (default 0, never). --telemetry-image-every
              is the same number under the spelling `es eval run` and the editor use.

Both trainer flags are passed **only** with --telemetry, and neither enters the plan: they
change nothing the run computes, so `training.lock`, the checkpoints and metrics/loss.json
are byte-identical with and without them.

ES_PYTHON overrides `[run] interpreter` when it is set.

<out>/training.lock carries two digests: `identity_hash`, over the nine slots that are known
before the trainer starts, and `training_hash`, over all twelve. A slot the run genuinely
does not know is the file {\"unset\": true} and is hashed as such -- never a zero digest and
never a fabricated one (spec 28.10 rule 2).

Exit codes: 0 success, 1 runtime failure, 2 usage error.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    let (mut recipe, mut out, mut retired) = (None, None, None);
    let (mut addr, mut token, mut image_every) = (None, None, None);
    let (mut progress_every, mut sample_every) = (None, None);
    let mut dry_run = false;
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
            "--recipe" => &mut recipe,
            "--out" => &mut out,
            "--allow-retired-task" => &mut retired,
            "--telemetry" => &mut addr,
            "--telemetry-token" => &mut token,
            "--telemetry-image-every" => &mut image_every,
            "--progress-every" => &mut progress_every,
            "--sample-every" => &mut sample_every,
            other => {
                return Err(CliError::Usage(format!(
                    "es train: unknown flag '{other}'\n\n{HELP}"
                )))
            }
        };
        *slot = Some(
            it.next()
                .ok_or_else(|| CliError::Usage(format!("es train: {a} needs a value\n\n{HELP}")))?
                .clone(),
        );
    }
    let Some(recipe_path) = recipe else {
        return Err(CliError::Usage(format!(
            "es train: --recipe <training.toml> is required\n\n{HELP}"
        )));
    };
    let recipe_path = PathBuf::from(recipe_path);
    let out = out.map_or_else(
        || PathBuf::from(recipe_path.file_stem().unwrap_or_default()),
        PathBuf::from,
    );
    let recipe = Recipe::parse(&read(&recipe_path)?).map_err(|e| bad(e.to_string()))?;
    if recipe.kind != es_data::training::KIND {
        return Err(bad(format!(
            "{}: kind = {:?}; `es train` reads a {:?} document",
            recipe_path.display(),
            recipe.kind,
            es_data::training::KIND
        )));
    }
    // Bound before the bundle, the dataset or Python is opened, so a viewer that attaches on
    // the printed address is subscribed before the first optimizer step (packet M7/E7).
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
    let number = |flag: &str, v: &Option<String>| -> Result<Option<u64>, CliError> {
        v.as_ref()
            .map(|v| {
                v.parse().map_err(|_| {
                    CliError::Usage(format!("es train: {flag} {v:?} is not a number\n\n{HELP}"))
                })
            })
            .transpose()
    };
    let watching = telemetry.addr.is_some();
    let mut owned = telemetry.bind(STAGE_TRAIN)?;
    let watch = TrainWatch {
        // The trainer says nothing nobody is listening for: without `--telemetry` neither
        // flag is passed and its command line is the one every measured run used.
        progress_every: number("--progress-every", &progress_every)?.unwrap_or(if watching {
            DEFAULT_PROGRESS_EVERY
        } else {
            0
        }),
        sample_every: number("--sample-every", &sample_every)?.unwrap_or(telemetry.image_every),
        publisher: owned.as_mut(),
    };
    let code = run(&recipe, &out, dry_run, retired.as_deref(), watch)?;
    if let Some(p) = &owned {
        println!("{}", p.summary());
    }
    Ok(code)
}

/// Progress lines a `--telemetry` run asks for when nobody said. Ten steps is a curve that
/// moves without a line per step on a 20,000-step run.
pub(crate) const DEFAULT_PROGRESS_EVERY: u64 = 10;

/// What `es train` publishes, and how often it asks the trainer to say something.
///
/// A `--telemetry`-less run carries the default: no publisher and two zeroes, which is the
/// trainer command line of every measured run (packet M7/E7).
#[derive(Default)]
pub(crate) struct TrainWatch<'a> {
    pub publisher: Option<&'a mut Publisher>,
    pub progress_every: u64,
    pub sample_every: u64,
}

impl TrainWatch<'_> {
    /// The flags appended to `train_act.py`'s command line — **not** to the plan.
    ///
    /// The plan is `config.json`'s `plan` and therefore part of `identity_hash` (spec 19.3):
    /// a flag that changes nothing the run computes must not move a run's identity, and a
    /// `training.lock` that differed by whether someone was watching would make two identical
    /// runs look like two runs. `--dry-run` prints the same plan either way.
    fn trainer_flags(&self, ir_route: bool) -> Vec<String> {
        let mut out = Vec::new();
        if self.publisher.is_none() || !ir_route {
            return out;
        }
        for (flag, n) in [
            ("--progress-every", self.progress_every),
            ("--sample-every", self.sample_every),
        ] {
            if n > 0 {
                out.push(flag.to_owned());
                out.push(n.to_string());
            }
        }
        out
    }
}

fn bad(msg: impl Into<String>) -> CliError {
    CliError::Runtime(msg.into())
}

fn read(path: &Path) -> Result<String, CliError> {
    std::fs::read_to_string(path).map_err(|e| bad(format!("{}: {e}", path.display())))
}

/// The interpreter the run uses: `ES_PYTHON` first, as every other oracle in this repository
/// resolves it, then the recipe's declared one.
fn interpreter_of(recipe: &Recipe) -> String {
    std::env::var("ES_PYTHON").unwrap_or_else(|_| recipe.run.interpreter.clone())
}

/// The program that runs `lerobot-train`: its entry point beside the interpreter when there
/// is one, otherwise `-m lerobot.scripts.lerobot_train` (packet M7/T1 spec).
fn lerobot_trainer(interpreter: &str) -> Vec<String> {
    let dir = Path::new(interpreter).parent().unwrap_or(Path::new(""));
    for name in ["lerobot-train", "lerobot-train.exe"] {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return vec![candidate.to_string_lossy().into_owned()];
        }
    }
    vec![
        interpreter.to_owned(),
        "-m".to_owned(),
        "lerobot.scripts.lerobot_train".to_owned(),
    ]
}

/// The plan `es train --recipe` would run this recipe with, built without touching the
/// dataset, the bundle or Python — what `--dry-run` prints and what `es loop cycle` nests
/// under its own `train` line.
///
/// The external route carries no bundle, so its Observation IR is a file and can be read
/// before anything else; the IR route's lives inside the bundle and is only opened for a real
/// run, which is what lets `--dry-run` be judged with neither dataset nor bundle on disk
/// (packet M7/T1 oracle 1).
pub(crate) fn plan_of(recipe: &Recipe, out: &Path) -> Result<Plan, CliError> {
    plan_with(recipe, out, None)
}

/// [`plan_of`] for a caller that has already opened the policy's Observation IR.
///
/// It is the same plan plus the two flags packet M7/T6's `training_only` chain adds — the
/// bake's `--for-training` and the trainer's `--augmentation`. The IR route only reaches
/// this with `Some` on a real run, because `--dry-run` deliberately does not open the bundle.
fn plan_with(
    recipe: &Recipe,
    out: &Path,
    observation: Option<&ObservationIr>,
) -> Result<Plan, CliError> {
    let route = recipe.route().map_err(|e| bad(e.to_string()))?;
    let interpreter = interpreter_of(recipe);
    let trainer = lerobot_trainer(&interpreter);
    let external_obs = match (route, observation) {
        (Route::External, None) => Some(open_observation(recipe)?),
        _ => None,
    };
    let observation = observation.or(external_obs.as_ref());
    let augmented = match observation {
        Some(obs) => !es_compile::plan::augmentation_chains(obs)
            .map_err(|d| {
                bad(d
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("\n"))
            })?
            .is_empty(),
        None => false,
    };
    Plan::build(
        recipe,
        out,
        &interpreter,
        &trainer,
        observation.map(state_dim),
        augmented,
    )
    .map_err(|e| bad(e.to_string()))
}

/// One training run, from a recipe that is already parsed.
///
/// The parsed recipe rather than its path, because `es loop cycle` overrides the `[dataset]`
/// slot with its own collect output before handing it over (packet M7/T2): the words the two
/// commands run are the same words, and there is no second parser.
pub(crate) fn run(
    recipe: &Recipe,
    out: &Path,
    dry_run: bool,
    retired: Option<&str>,
    mut watch: TrainWatch<'_>,
) -> Result<u8, CliError> {
    let recipe = recipe.clone();
    let route = recipe.route().map_err(|e| bad(e.to_string()))?;
    let interpreter = interpreter_of(&recipe);

    if dry_run {
        let text = plan_of(&recipe, out)?.render(out);
        print!("{text}");
        let path = out.join("training").join("plan.txt");
        write_file(&path, text.as_bytes())?;
        // stdout is the plan and nothing else, so it and `plan.txt` are one golden.
        eprintln!("plan: {}", path.display());
        return Ok(0);
    }

    // --- the refusals that need the documents -------------------------------------------
    let bundle = match route {
        Route::Ir => {
            let path = recipe.policy.bundle.clone().unwrap_or_default();
            let bytes = std::fs::read(&path).map_err(|e| bad(format!("{path}: {e}")))?;
            Some(PolicyBundle::open(&bytes).map_err(|e| bad(format!("{path}: {e}")))?)
        }
        Route::External => None,
    };
    let external_obs = match route {
        Route::External => Some(open_observation(&recipe)?),
        Route::Ir => None,
    };
    let observation = match (&bundle, &external_obs) {
        (Some(b), _) => &b.observation,
        (None, Some(o)) => o,
        (None, None) => unreachable!("one of the two routes always resolves an Observation IR"),
    };
    // The plan, now that the document the augmentation chain lives in is open (packet M7/T6).
    // `--dry-run` above builds it without one on the IR route, which is what lets a plan be
    // printed on a machine that has neither the bundle nor the dataset.
    let plan = plan_with(&recipe, out, Some(observation))?;
    if has_image_input(observation) && recipe.dataset.frames.is_none() {
        return Err(bad(
            "the Observation IR has an image input and [dataset] `frames` is not set; a run \
             with a zero-filled image channel trains a policy that looks fine",
        ));
    }
    // Packet M7/T5: the recipe and the Learning IR must agree about where this run starts.
    // Either disagreement writes a `base_model.lock` that does not describe the run -- a
    // bundle that wants ImageNet and gets none trains from scratch under a document saying
    // otherwise, and a recipe that names weights the module never loads claims a provenance
    // out of thin air (spec 28.10 rule 2).
    let backbone = match (
        bundle
            .as_ref()
            .map(|b| has_pretrained_backbone(&b.learning)),
        &recipe.policy.base_model,
    ) {
        (Some(true), None) => {
            return Err(bad(
                "the bundle's Learning IR declares `VisionEncoder { pretrained = true }` and \
                 [policy] `base_model` names no weights to start from. The ImageNet tensors \
                 enter through a checkpoint, never over the network at construction (spec \
                 2.5): fetch them with `python/es/fetch_backbone.py --arch resnet18 --out \
                 <dir>` and point `base_model` at the safetensors file it writes.",
            ))
        }
        (Some(false), Some(path)) => {
            return Err(bad(format!(
                "[policy] `base_model` names {path}, and the bundle's Learning IR has no \
                 `VisionEncoder {{ pretrained = true }}` to load it into. Nothing would read \
                 those weights, and `training/base_model.lock` would claim a provenance this \
                 run does not have.",
            )))
        }
        (Some(true), Some(path)) => {
            Some(Backbone::verify(Path::new(path)).map_err(|e| bad(e.to_string()))?)
        }
        _ => None,
    };
    if let Some(lock) = &backbone {
        println!("base_model:    {} ({})", lock.source, lock.license);
    }

    // Packet M8/S1: what this run starts from, decided here -- before the probe, before the
    // bake, before a single GPU-second -- because `init.lock` enters `identity_hash`, and the
    // identity of a run exists before the run does. `[init]` is refused on the lerobot route,
    // so the bundle is always open by now.
    let init = match (&recipe.init, &bundle) {
        (Some(from), Some(target)) => {
            let bytes = std::fs::read(&from.policy)
                .map_err(|e| bad(format!("[init] `policy` {}: {e}", from.policy)))?;
            let built = init_from(&from.policy, &bytes, &target.learning)
                .map_err(|e| bad(e.to_string()))?;
            println!(
                "init:          {} tensor(s) copied from {}, {} initialised",
                built.copied, from.policy, built.initialised
            );
            Some(built)
        }
        _ => None,
    };

    let task_hash = match (&bundle, &recipe.policy.task) {
        (Some(b), _) => b.task.task_hash().map_err(|e| bad(e.to_string()))?,
        (None, Some(path)) => task_from_toml(&read(Path::new(path))?)
            .map_err(|e| bad(format!("{path}: {e}")))?
            .task_hash()
            .map_err(|e| bad(e.to_string()))?,
        (None, None) => unreachable!("the external route requires [policy] task"),
    };

    let dataset = LeRobotDataset::open(&recipe.dataset.root).map_err(|e| bad(e.to_string()))?;
    let facts = dataset_facts(&dataset)?;
    check_task(&facts, &task_hash, retired)?;

    // --- the identity, before a single GPU-second ----------------------------------------
    let training_dir = out.join("training");
    let mut training = Training::pre_run(
        &recipe,
        &plan,
        out,
        &interpreter,
        &facts,
        backbone.as_ref(),
        Some(observation),
    )
    .map_err(|e| bad(e.to_string()))?;
    if let Some(init) = &init {
        training
            .set_init(&init.lock)
            .map_err(|e| bad(e.to_string()))?;
    }
    training
        .write(&training_dir)
        .map_err(|e| bad(e.to_string()))?;
    let identity_hash = training.hash().map_err(|e| bad(e.to_string()))?;
    write_lock(out, &identity_hash, None, &training, &[])?;
    println!("route:         {}", plan.route.as_str());
    println!("identity_hash: {}", hex(&identity_hash));

    // Before the bake or the export, not after: an interpreter that cannot import what the
    // route needs is a two-second answer, and finding it out after a ten-minute bake is the
    // kind of thing this command exists to stop. Its reply is also `hardware.json`.
    let hardware_probe = probe(&interpreter, route)?;

    // The external route's exporter wants one directory per camera and `es loop collect
    // --frames` writes a flat one; packet M5/V19 bridged the two with `mkdir` and `ln -s`.
    if route == Route::External {
        if let Some(flat) = &recipe.dataset.frames {
            mirror_frames(Path::new(flat), &dataset, &out.join("frames-in"))?;
        }
    }

    // `train_act.py` writes its checkpoints and its loss curve where it is told and creates
    // no directory: making them is the caller's job, and it is this caller.
    for dir in ["weights", "metrics", "checkpoints"] {
        let dir = out.join(dir);
        std::fs::create_dir_all(&dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?;
    }
    // The intersection `init_from` kept, as the file the trainer's `--init-weights` names.
    // Written after `weights/` exists and before the plan runs, because the trainer is one of
    // the plan's steps and this is its input (packet M8/S1).
    if let Some(init) = &init {
        write_file(
            Path::new(&init_weights(out)),
            &es_policy::weights::write_safetensors(&init.weights),
        )?;
    }

    // --- the plan ------------------------------------------------------------------------
    let mut checkpoints = Vec::new();
    let mut summary = Value::Null;
    for step in &plan.steps {
        println!("$ {}", one_line(step));
        match step.kind {
            StepKind::DatasetBake => {
                crate::cmd::dataset::bake(&step.args)?;
            }
            StepKind::DatasetExport => {
                crate::cmd::dataset::export(&step.args)?;
            }
            StepKind::PolicyLower => {
                crate::cmd::policy::lower(&step.args)?;
            }
            StepKind::Trainer => {
                // How long this run is, said right before the trainer and not before the
                // bake: a viewer that dials in during a long bake still hears the total
                // before the first progress line, so it can draw an ETA (packet M7/E7).
                if let Some(p) = watch.publisher.as_deref_mut() {
                    p.train_begin(recipe.run.steps);
                }
                summary = spawn(step, route == Route::Ir, &mut watch)?;
            }
            StepKind::PolicyPack => {
                crate::cmd::policy::pack(&step.args)?;
                checkpoints.push(published_row(step, out, &mut watch)?);
            }
            StepKind::PolicyImportLerobot => {
                crate::cmd::policy::import_lerobot(&step.args)?;
                checkpoints.push(published_row(step, out, &mut watch)?);
            }
        }
    }

    // --- the three post-run slots --------------------------------------------------------
    let manifest = json!({"schema_version": 1, "checkpoints": checkpoints});
    training.finish(
        &manifest,
        &metrics(out, &summary),
        &hardware(&hardware_probe, route, &recipe.run.device, &interpreter),
    );
    training
        .write(&training_dir)
        .map_err(|e| bad(e.to_string()))?;
    let training_hash = training.hash().map_err(|e| bad(e.to_string()))?;
    write_lock(
        out,
        &identity_hash,
        Some(&training_hash),
        &training,
        &checkpoints,
    )?;
    warn_on_optimizer(&training, &summary);
    println!("training_hash: {}", hex(&training_hash));
    println!("lock:          {}", out.join("training.lock").display());
    Ok(0)
}

// --- helpers ----------------------------------------------------------------------------

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?;
    }
    std::fs::write(path, bytes).map_err(|e| bad(format!("{}: {e}", path.display())))
}

fn one_line(step: &Step) -> String {
    step.prefix
        .iter()
        .chain(&step.args)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

fn open_observation(recipe: &Recipe) -> Result<ObservationIr, CliError> {
    let path = recipe.policy.observation.clone().unwrap_or_default();
    observation_from_toml(&read(Path::new(&path))?).map_err(|e| bad(format!("{path}: {e}")))
}

/// The dataset's own numbers: the spec 19.2 three-way hash, the split `es loop distill` left
/// beside it when there is one, and the `es:task:<hex>` name collection recorded.
fn dataset_facts(dataset: &LeRobotDataset) -> Result<DatasetFacts, CliError> {
    let n = dataset.episodes().len() as u32;
    let split_file = dataset.root().join("split.json");
    let (split, split_source) = match std::fs::read_to_string(&split_file) {
        Ok(text) => (
            serde_json::from_str::<Split>(&text)
                .map_err(|e| bad(format!("{}: {e}", split_file.display())))?,
            "loop distill split.json",
        ),
        // `es train` trains on every episode of the root it was pointed at, so the honest
        // split for a root with no `split.json` is the all-train partition -- the same
        // display-only convention `es dataset info` and `es dataset bake` use.
        Err(_) => (Split::deterministic(n, [1.0, 0.0, 0.0], 0), "all-train"),
    };
    let identity = DatasetIdentity::compute(dataset, &split).map_err(|e| bad(e.to_string()))?;
    Ok(DatasetFacts {
        hashes: identity.to_dataset_hash(),
        episodes: n,
        frames: dataset.episodes().iter().map(|m| m.length).sum(),
        recorded_task: dataset
            .tasks()
            .iter()
            .map(|t| t.task.clone())
            .find(|t| t.starts_with("es:task:")),
        split_source,
    })
}

/// M5 review S-3/R4, surfaced here: demonstrations of one predicate and documents of another
/// measure nothing, and packet M5/V19 measured 0/16 by construction because of it.
fn check_task(
    facts: &DatasetFacts,
    task_hash: &[u8; 32],
    retired: Option<&str>,
) -> Result<(), CliError> {
    let Some(recorded) = &facts.recorded_task else {
        return Ok(());
    };
    let want = format!("es:task:{}", hex(task_hash));
    if *recorded == want {
        return Ok(());
    }
    let recorded_hex = recorded.trim_start_matches("es:task:");
    if retired == Some(recorded_hex) {
        println!("note: accepting the retired task {recorded_hex} (--allow-retired-task)");
        return Ok(());
    }
    Err(bad(format!(
        "the dataset was collected under task_hash {recorded_hex} and this recipe's Task IR \
         is {}.\nA policy trained on demonstrations of one predicate and judged against \
         another measures nothing (M5 review S-3).\nPass \
         `--allow-retired-task {recorded_hex}` to accept it deliberately.",
        hex(task_hash)
    )))
}

/// Hard-links the flat `<NNNNNN>.bin` tiles into the `<camera>/` layout `es dataset export`
/// reads. Nothing is copied unless the link fails, and an existing file is left alone.
///
/// ponytail: one link per frame, ~60k for a 200-demonstration set; a directory symlink would
/// be one syscall, but it needs privileges on Windows and this runs on both.
fn mirror_frames(flat: &Path, dataset: &LeRobotDataset, into: &Path) -> Result<(), CliError> {
    let frames: u64 = dataset.episodes().iter().map(|m| m.length).sum();
    for camera in dataset.info().cameras() {
        let dir = into.join(camera_suffix(camera));
        std::fs::create_dir_all(&dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?;
        for i in 0..frames {
            let (src, dst) = (
                flat.join(format!("{i:06}.bin")),
                dir.join(format!("{i:06}.bin")),
            );
            if dst.exists() {
                continue;
            }
            if std::fs::hard_link(&src, &dst).is_err() {
                std::fs::copy(&src, &dst)
                    .map_err(|e| bad(format!("{} -> {}: {e}", src.display(), dst.display())))?;
            }
        }
        println!("frames:        {} <- {}", dir.display(), flat.display());
    }
    Ok(())
}

/// The one interpreter probe: it is both the refusal the packet names (an interpreter that
/// cannot import what the route needs, reported in the interpreter's own words) and the
/// source of `hardware.json`'s device and library versions.
fn probe(interpreter: &str, route: Route) -> Result<Value, CliError> {
    let script = "\
import json, sys
d = {}
try:
    import torch
    d['torch'] = torch.__version__
    d['cuda'] = torch.version.cuda
    d['device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
except Exception as e:
    sys.stderr.write('%s: %s\\n' % (type(e).__name__, e)); raise SystemExit(1)
if len(sys.argv) > 1 and sys.argv[1] == 'lerobot':
    try:
        import lerobot
        d['lerobot'] = getattr(lerobot, '__version__', 'unknown')
    except Exception as e:
        sys.stderr.write('%s: %s\\n' % (type(e).__name__, e)); raise SystemExit(1)
print(json.dumps(d))
";
    let mut cmd = Command::new(interpreter);
    cmd.args(["-c", script]);
    if route == Route::External {
        cmd.arg("lerobot");
    }
    let out = cmd.output().map_err(|e| {
        bad(format!(
            "{interpreter}: {e}\nSet ES_PYTHON or [run] interpreter."
        ))
    })?;
    if !out.status.success() {
        return Err(bad(format!(
            "{interpreter} cannot import what the {} route needs:\n{}",
            route.as_str(),
            String::from_utf8_lossy(&out.stderr).trim_end()
        )));
    }
    serde_json::from_slice(&out.stdout)
        .map_err(|e| bad(format!("{interpreter}: the probe printed {e}")))
}

/// Runs the one subprocess. `capture` is for `train_act.py`, whose whole report is a single
/// JSON line on stdout; `lerobot-train` streams a progress log instead and inherits.
///
/// With a publisher the captured path becomes a *streamed* one: the same stdout, read line by
/// line so a `{"progress": ...}` line reaches a viewer while the run is still going. The
/// summary is still the last line and still parsed the same way, which is what keeps
/// `training.lock` byte-identical (packet M7/E7).
fn spawn(step: &Step, capture: bool, watch: &mut TrainWatch<'_>) -> Result<Value, CliError> {
    let mut cmd = Command::new(&step.prefix[0]);
    cmd.args(&step.prefix[1..]).args(&step.args);
    let extra = watch.trainer_flags(capture);
    if !extra.is_empty() {
        println!("  + {}", extra.join(" "));
        cmd.args(&extra);
    }
    let named = || format!("{}: ", step.prefix[0]);
    let (ok, code, summary) = if capture && watch.publisher.is_some() {
        stream(&mut cmd, watch).map_err(|e| bad(format!("{}{e}", named())))?
    } else if capture {
        let out = cmd.output().map_err(|e| bad(format!("{}{e}", named())))?;
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        print!("{text}");
        // Captured means captured: a trainer that failed has to be able to say why.
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        let last = text.lines().last().unwrap_or_default();
        (
            out.status.success(),
            out.status.code(),
            serde_json::from_str(last).unwrap_or(Value::Null),
        )
    } else {
        let status = cmd.status().map_err(|e| bad(format!("{}{e}", named())))?;
        (status.success(), status.code(), Value::Null)
    };
    if ok {
        Ok(summary)
    } else {
        Err(bad(format!(
            "the trainer exited with {}",
            code.unwrap_or(-1)
        )))
    }
}

/// The trainer's stdout, line by line, published as it arrives.
///
/// stderr is inherited rather than piped: reading two pipes from one thread deadlocks when
/// either fills, and the non-streamed path's `eprint!` of the captured stderr and this go to
/// the same place. A line that is not one of the two the trainer publishes is printed and
/// remembered as a candidate summary, so the last non-progress line is the report — exactly
/// what `Command::output`'s `text.lines().last()` picks.
fn stream(
    cmd: &mut Command,
    watch: &mut TrainWatch<'_>,
) -> std::io::Result<(bool, Option<i32>, Value)> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("the trainer's stdout was not piped"))?;
    let mut last = String::new();
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        println!("{line}");
        let parsed = serde_json::from_str::<Value>(&line).unwrap_or(Value::Null);
        let publisher = watch.publisher.as_deref_mut();
        match (publisher, parsed.get("progress"), parsed.get("sample")) {
            (Some(p), Some(progress), _) => p.progress(progress),
            (Some(p), None, Some(Value::String(path))) => p.sample(Path::new(path)),
            _ => {}
        }
        if parsed.get("progress").is_none() && parsed.get("sample").is_none() {
            last = line;
        }
    }
    let status = child.wait()?;
    Ok((
        status.success(),
        status.code(),
        serde_json::from_str(&last).unwrap_or(Value::Null),
    ))
}

/// [`manifest_row`], announced: a viewer learns that a mark was packed and which policy it
/// is, at the moment the bundle exists on disk (packet M7/E7).
fn published_row(step: &Step, out: &Path, watch: &mut TrainWatch<'_>) -> Result<Value, CliError> {
    let row = manifest_row(step, out)?;
    if let Some(p) = watch.publisher.as_deref_mut() {
        p.checkpoint(
            &row["step"].to_string(),
            row["bundle_policy_hash"].as_str().unwrap_or("-"),
        );
    }
    Ok(row)
}

/// One `checkpoint.manifest` row, read back from the bundle that was just written so the
/// digest names bytes on disk rather than bytes in memory.
fn manifest_row(step: &Step, out: &Path) -> Result<Value, CliError> {
    let path = step
        .args
        .iter()
        .skip_while(|a| *a != "--out")
        .nth(1)
        .ok_or_else(|| bad("a checkpoint step with no --out"))?;
    let bytes = std::fs::read(path).map_err(|e| bad(format!("{path}: {e}")))?;
    let bundle = PolicyBundle::open(&bytes).map_err(|e| bad(format!("{path}: {e}")))?;
    let relative = Path::new(path)
        .strip_prefix(out)
        .unwrap_or(Path::new(path))
        .to_string_lossy()
        .replace('\\', "/");
    Ok(json!({
        "step": step.step,
        "bundle": relative,
        // The bundle's own spec 5.3 `policy_hash`, not spec 19.3's: the latter is
        // H(training_hash, checkpoint) and cannot live inside a `training_hash` input.
        // `training.lock` carries that one.
        "bundle_policy_hash": bundle.manifest.hashes.policy.as_ref().map(hex),
        "weights_blake3": hex(blake3::hash(&bytes).as_bytes()),
    }))
}

fn metrics(out: &Path, summary: &Value) -> Value {
    let curve: Option<Value> = std::fs::read_to_string(out.join("metrics/loss.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    match (curve, summary) {
        (None, Value::Null) => json!({
            "loss": {"unset": true},
            "note": "the external trainer reports its curve to its own logs, not to a file \
                     this command reads",
        }),
        (curve, summary) => json!({
            "loss": curve.unwrap_or(Value::Null),
            "summary": summary.clone(),
        }),
    }
}

fn hardware(probe: &Value, route: Route, device: &str, interpreter: &str) -> Value {
    json!({
        "device": device,
        "device_name": probe.get("device_name").cloned(),
        "torch": probe.get("torch").cloned(),
        "cuda": probe.get("cuda").cloned(),
        "lerobot": probe.get("lerobot").cloned(),
        // No build script and no committed Cargo.lock (M5 review S-8/R7), so a build cannot
        // be named here yet; claiming a revision would be exactly the fabricated slot spec
        // 28.10 rule 2 forbids.
        "driver": {"unset": true},
        "git_describe": {"unset": true},
        "es_version": env!("CARGO_PKG_VERSION"),
        "interpreter": interpreter,
        "trainer": match route { Route::Ir => TRAIN_ACT, Route::External => "lerobot-train" },
    })
}

/// `optimizer.json` declares the betas and eps `train_act.py`'s `AdamW` leaves at torch's
/// defaults, beside the lr and weight decay the recipe tells it to use (packet M7/T4). If
/// torch ever moves a default the declaration is stale, so the trainer reports what it built
/// and the two are compared out loud.
fn warn_on_optimizer(training: &Training, summary: &Value) {
    let Some(reported) = summary.get("optimizer") else {
        return;
    };
    let Ok(declared) = serde_json::from_str::<Value>(training.file("optimizer.json")) else {
        return;
    };
    let same = ["lr", "betas", "eps", "weight_decay"]
        .iter()
        .all(|k| declared.get(*k) == reported.get(*k));
    if !same {
        println!(
            "warning: the trainer reports {reported} and optimizer.json declares {declared}; \
             training_hash names the declaration, so it is now stale"
        );
    }
}

fn write_lock(
    out: &Path,
    identity: &[u8; 32],
    training_hash: Option<&[u8; 32]>,
    training: &Training,
    checkpoints: &[Value],
) -> Result<(), CliError> {
    let id = training.identity();
    let per_checkpoint: Vec<Value> = checkpoints
        .iter()
        .map(|c| {
            let digest = c["weights_blake3"]
                .as_str()
                .and_then(from_hex)
                .unwrap_or([0; 32]);
            json!({
                "step": c["step"],
                "bundle": c["bundle"],
                // Spec 19.3: policy_hash = H(training_hash, checkpoint_hash).
                "policy_hash": id.policy_hash(&digest).ok().as_ref().map(hex),
            })
        })
        .collect();
    let lock = json!({
        "schema_version": 1,
        "identity_hash": hex(identity),
        "training_hash": training_hash.map_or(json!({"unset": true}), |h| json!(hex(h))),
        "files": training.digests(),
        "checkpoints": per_checkpoint,
    });
    let mut text = lock.to_string();
    text.push('\n');
    write_file(&out.join("training.lock"), text.as_bytes())
}

fn from_hex(text: &str) -> Option<[u8; 32]> {
    let bytes: Vec<u8> = (0..text.len() / 2)
        .filter_map(|i| u8::from_str_radix(text.get(i * 2..i * 2 + 2)?, 16).ok())
        .collect();
    bytes.try_into().ok()
}
