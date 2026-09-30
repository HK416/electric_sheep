//! `es train` — one document, one command, a real `training_hash` (spec 13.1, 19.3;
//! packet M7/T1, design note `docs/design/training-recipe.md`).
//!
//! Thin on purpose. The recipe schema, the command plan and spec 19.3's `training/` bundle
//! live in `es_data::training`, which is headless and unit-tested; what is here is the
//! execution order and the process boundary. Every `es` step of the plan is called
//! **in-process** — they are functions in this module's siblings — and the only subprocess is
//! the Python trainer, which is exactly where spec 2.3 draws the line.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use es_compile::PolicyBundle;
use es_core::child::{retry_start, transient_start_failure};
use es_data::training::{
    camera_dirs, has_image_input, has_pretrained_backbone, init_from, init_weights, rollout_docs,
    state_dim, Backbone, DatasetFacts, Plan, Recipe, Route, Step, StepKind, Training, TRAIN_ACT,
    TRAIN_PPO,
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

An `[rl]` table beside `[policy] bundle` picks a third route -- PPO against the Task IR's
reward instead of recorded demonstrations (spec 13.4):

  rl       es policy lower -> python/es/train_ppo.py -> es policy pack

There is no bake, because there is no dataset: `[dataset]` is optional and training/
dataset.lock reads {\"unset\": true}. `[run] steps` is the iteration count, `[run] batch` is
refused by name (a PPO batch is [rl] envs * horizon, derived), and the Task, Observation and
Deployment IR come out of the bundle -- written to <out>/docs/ and handed to the trainer as
--rollout-docs, so the env stepped is the one the policy was declared against. Every sampled
action goes through the Safety Plane before the actuator; there is no path around it
(INV-12). The value MLP and the Gaussian's log_std are training-only state: they live in
<out>/training/value.safetensors and are never packed into a bundle.

A bundle whose Learning IR declares `VisionEncoder { pretrained = true }` also needs
`[policy] base_model = \"<dir>/resnet18-imagenet1k-v1.safetensors\"`, the artifact
`python/es/fetch_backbone.py` writes. Its blake3 is checked against the lock file beside it
*and* against the pin this build carries, its licence is copied into
training/base_model.lock, and the tensors reach the trainer as `--init-backbone` -- never
over the network at construction (spec 2.5, 19.3). With `[policy] base_model_fetch =
\"resnet18\"` a missing file is fetched first: `fetch_backbone.py --out <its dir> --expect <the
pin>` with the run's interpreter, shown as the plan's `# fetch:` line (`es loop cycle` runs it
before collect).

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
              training.lock records. On the lerobot route each checkpoint is imported as
              soon as `lerobot-train` has finished writing it -- `model.safetensors` and
              `config.json` on disk when a training bar past its step is read -- instead of
              all after the trainer exits; the bundles and training.lock are the same bytes.
--telemetry-token <t>
              required in every client's Hello (spec 25.1); none by default
--progress-every <N>
              ask the trainer for one progress line every N optimizer steps (default 10 with
              --telemetry, 0 without)
--sample-every <N>
              ask the trainer to write one image input of the batch, after augmentation, as
              Rgb8 beside metrics/ every N steps (default 0, never). --telemetry-image-every
              is the same number under the spelling `es eval run` and the editor use.

Both trainer flags are passed **only** with --telemetry (and --progress-every also when `es
loop cycle` previews checkpoints), and neither enters the plan: they change nothing the run
computes, so `training.lock`, the checkpoints and metrics/loss.json are byte-identical with and
without them. With either listener each checkpoint is bundled as soon as the trainer is past
it -- its weights on disk when a progress line (a training bar on the lerobot route) past the
mark is read -- instead of all after the trainer exits; the bundles are the same bytes.

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
    // Before the plan, `--dry-run` included (packet M15/N7): a `single_view` over a graph with
    // no camera `Sum` is refused by name here, as `Cycle::training` refuses it for a cycle.
    recipe.check_single_view().map_err(|e| bad(e.to_string()))?;
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
        on_checkpoint: None,
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
    /// Told each checkpoint mark as soon as its bundle is on disk (packet M13/Z1): `es loop
    /// cycle`'s preview queue. It must return at once -- the trainer may still be running.
    pub on_checkpoint: Option<&'a mut dyn FnMut(u32)>,
}

impl TrainWatch<'_> {
    /// Someone wants to hear from the run while it is going, not only from its end.
    fn listening(&self) -> bool {
        self.publisher.is_some() || self.on_checkpoint.is_some()
    }

    /// The flags appended to `train_act.py`'s command line — **not** to the plan.
    ///
    /// The plan is `config.json`'s `plan` and therefore part of `identity_hash` (spec 19.3):
    /// a flag that changes nothing the run computes must not move a run's identity, and a
    /// `training.lock` that differed by whether someone was watching would make two identical
    /// runs look like two runs. `--dry-run` prints the same plan either way.
    fn trainer_flags(&self, route: Route) -> Vec<String> {
        let mut out = Vec::new();
        if !self.listening() || !route.captures_trainer_stdout() {
            return out;
        }
        for (flag, n) in [
            // The progress line is also how a mark is known to be written (packet M13/Z1),
            // so the preview queue asks for it too.
            ("--progress-every", self.progress_every),
            // A rollout has no image batch to draw one from: `--sample-every` is
            // `train_act.py`'s, and passing it to `train_ppo.py` would name a flag that
            // trainer does not have (packet M8/S4b). And a picture is for a viewer: without
            // one it would be a file written for nobody.
            (
                "--sample-every",
                if route == Route::Rl || self.publisher.is_none() {
                    0
                } else {
                    self.sample_every
                },
            ),
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
        Route::Ir | Route::Rl => {
            let path = recipe.policy.bundle.clone().unwrap_or_default();
            let bytes = std::fs::read(&path).map_err(|e| bad(format!("{path}: {e}")))?;
            Some(PolicyBundle::open(&bytes).map_err(|e| bad(format!("{path}: {e}")))?)
        }
        Route::External => None,
    };
    let external_obs = match route {
        Route::External => Some(open_observation(&recipe)?),
        Route::Ir | Route::Rl => None,
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
    match route {
        // `es_native.Rollout` renders an image input since packet M11/X3 (its maturin build
        // has the `render` feature), so a render build passes the documents through and the
        // trainer writes the render cost to `metrics/env-metrics.json` (packet M11/R1). A
        // build without the feature is refused here rather than on the first step of the
        // first iteration -- after the lowering and the interpreter probe (packet M8/S4b).
        Route::Rl if !cfg!(feature = "render") && has_image_input(observation) => {
            return Err(bad(
                "the policy's Observation IR has an image input and this build of `es` has no \
                 `render` feature, so `[rl]` cannot render it through `es_native.Rollout`. \
                 Rebuild with the default features, or train against a state-only Observation \
                 IR of the same task",
            ))
        }
        Route::Rl => {}
        _ if has_image_input(observation)
            && recipe
                .dataset
                .as_ref()
                .and_then(|d| d.frames.as_ref())
                .is_none() =>
        {
            return Err(bad(
                "the Observation IR has an image input and [dataset] `frames` is not set; a \
                 run with a zero-filled image channel trains a policy that looks fine",
            ))
        }
        _ => {}
    }
    fetch_base_model(&plan, &recipe)?;
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

    // An RL run has no dataset to open, and says so rather than inventing one: `dataset.lock`
    // reads `{"unset": true}` and is hashed as such (spec 28.10 rule 2, packet M8/S4b). The
    // `check_task` comparison goes with it -- there are no demonstrations whose `task_hash`
    // could disagree with the documents, because the rollout *is* the documents.
    let (facts, opened) = if route == Route::Rl {
        (
            DatasetFacts {
                hashes: es_ir::DatasetHash {
                    content: [0; 32],
                    schema: [0; 32],
                    split: [0; 32],
                },
                episodes: 0,
                frames: 0,
                recorded_task: None,
                split_source: "unset",
            },
            None,
        )
    } else {
        let root = recipe
            .dataset
            .as_ref()
            .map(|d| d.root.clone())
            .unwrap_or_default();
        let dataset = LeRobotDataset::open(&root).map_err(|e| bad(e.to_string()))?;
        let facts = dataset_facts(&dataset)?;
        check_task(&facts, &task_hash, retired)?;
        (facts, Some(dataset))
    };

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
    // --frames` writes a flat one for one camera; packet M5/V19 bridged the two with `mkdir`
    // and `ln -s`.
    if let (Route::External, Some(dataset), Some(frames)) = (
        route,
        &opened,
        recipe.dataset.as_ref().and_then(|d| d.frames.as_ref()),
    ) {
        mirror_frames(Path::new(frames), dataset, &out.join("frames-in"))?;
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
    // The three documents the rollout is built from, taken out of the bundle rather than off
    // the recipe (packet M8/S4b): `es_native.Rollout` reads them as text, and a trainer
    // stepping an env declared by anything but the policy's own bundle is the disagreement
    // this writes out of existence. The scene is not among them -- the Task IR's `scene.path`
    // names it, and the trainer resolves it relative to the repository root, which is where
    // `es train` is run from.
    if route == Route::Rl {
        let bundle = bundle.as_ref().expect("the rl route opened its bundle");
        let docs = PathBuf::from(rollout_docs(out));
        for (name, text) in [
            ("task.toml", es_ir::serial::task_to_toml(&bundle.task)),
            (
                "observation.toml",
                es_ir::serial::observation_to_toml(&bundle.observation),
            ),
            (
                "deployment.toml",
                es_ir::serial::deployment_to_toml(&bundle.deployment),
            ),
        ] {
            let text = text.map_err(|e| bad(format!("{name} does not serialise: {e}")))?;
            write_file(&docs.join(name), text.as_bytes())?;
        }
        // And the scene beside them, copied rather than referenced, so `--rollout-docs` is
        // one self-contained directory and the trainer parses no TOML to find an XML file.
        // `scene.path` is repository-relative and `es train` runs from the repository root.
        let scene = &bundle.task.scene.path;
        let bytes = std::fs::read(scene).map_err(|e| {
            bad(format!(
                "{scene}: {e}\nThe Task IR's `scene.path` is repository-relative; run `es \
                 train` from the repository root."
            ))
        })?;
        write_file(&docs.join("scene.xml"), &bytes)?;
        // The mesh and texture files the scene names, at the same scene-relative paths, so the
        // rollout resolves them against `docs` exactly as `load_scene` does against the scene's
        // own directory (packets M16/H2, H1b). A primitives-only scene names none; a builtin
        // texture (`builtin="checker"`) names no file.
        let parsed = std::str::from_utf8(&bytes)
            .ok()
            .and_then(|xml| es_assets::parse_mjcf(xml).ok());
        let scene_dir = Path::new(scene).parent().unwrap_or(Path::new("."));
        for asset in parsed.iter().flat_map(|p| &p.scene.assets) {
            use es_assets::scene::AssetKind;
            let texture_file =
                asset.kind == AssetKind::Texture && scene_dir.join(&asset.path).is_file();
            if asset.kind == AssetKind::Mesh || texture_file {
                let mesh = std::fs::read(scene_dir.join(&asset.path))
                    .map_err(|e| bad(format!("{}: {e}", scene_dir.join(&asset.path).display())))?;
                write_file(&docs.join(&asset.path), &mesh)?;
            }
        }
        println!("rollout docs:  {}", docs.display());
        // The one route that steps physics, so the one that has a tier to say (S-4).
        let kind = match recipe.rl.as_ref().map(|rl| &rl.backend) {
            Some(es_data::training::RlBackend::MjWarp) => es_physics_backend::BackendKind::MjWarp,
            Some(es_data::training::RlBackend::PhysX) => es_physics_backend::BackendKind::PhysX,
            _ => es_physics_backend::BackendKind::MuJoCoCpu,
        };
        println!("{}", super::eval::determinism_tier(kind));
    }

    // --- the plan ------------------------------------------------------------------------
    let mut checkpoints = Vec::new();
    let mut summary = Value::Null;
    for step in &plan.steps {
        // Bundled while the trainer was still running (packet M13/Z1).
        if matches!(
            step.kind,
            StepKind::PolicyPack | StepKind::PolicyImportLerobot
        ) && checkpoints
            .iter()
            .any(|r: &Value| r["step"] == json!(step.step))
        {
            continue;
        }
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
                let mut early = Early {
                    plan: &plan,
                    out,
                    marks: CheckpointWatch::of(&plan),
                    rows: &mut checkpoints,
                };
                summary = spawn(step, route, recipe.run.batch, &mut watch, &mut early)?;
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
    // In mark order whichever way each was bundled: a no-op unless an early bundle failed
    // and was redone after the trainer exited, and the manifest is a `training_hash` slot.
    checkpoints.sort_by_key(|r| r["step"].as_u64());
    let manifest = json!({"schema_version": 1, "checkpoints": checkpoints});
    training.finish(
        &manifest,
        &metrics(out, &summary, route),
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

/// Packet M12/R8: the plan's `# fetch:` line, run when `[policy] base_model` is not on disk --
/// with the run's interpreter, before anything that takes time. What it wrote is then
/// `Backbone::verify`'s to judge, exactly as a file fetched by hand always was.
pub(crate) fn fetch_base_model(plan: &Plan, recipe: &Recipe) -> Result<(), CliError> {
    let (Some(words), Some(path)) = (&plan.fetch, &recipe.policy.base_model) else {
        return Ok(());
    };
    if Path::new(path).exists() {
        return Ok(());
    }
    let command = words.join(" ");
    println!("$ {command}");
    let why = match Command::new(&words[0]).args(&words[1..]).status() {
        Ok(status) if status.success() => return Ok(()),
        Ok(status) => format!("it exited with {}", status.code().unwrap_or(-1)),
        Err(e) => format!("{}: {e}", words[0]),
    };
    Err(bad(format!(
        "the pretrained backbone {path} is missing and could not be fetched ({why}).\n\
         It is downloaded once, by `{command}`: the interpreter needs torchvision and blake3 \
         (set ES_PYTHON to one that has them) and, the first time, the network. Nothing else \
         has run."
    )))
}

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

/// Hard-links each camera's `<NNNNNN>.bin` tiles into the `<camera>/` layout `es dataset
/// export` reads: the flat tiles of a one-camera collection, `<frames>/<camera>/` for each
/// camera of several (packet M15/N3, [`camera_dirs`]) -- never one camera's frames under
/// another's name. Nothing is copied unless the link fails, and an existing file is left
/// alone.
///
/// ponytail: one link per frame, ~60k for a 200-demonstration set; a directory symlink would
/// be one syscall, but it needs privileges on Windows and this runs on both.
fn mirror_frames(frames: &Path, dataset: &LeRobotDataset, into: &Path) -> Result<(), CliError> {
    let total: u64 = dataset.episodes().iter().map(|m| m.length).sum();
    for (camera, from) in camera_dirs(dataset.info(), frames) {
        let dir = into.join(camera);
        std::fs::create_dir_all(&dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?;
        for i in 0..total {
            let (src, dst) = (
                from.join(format!("{i:06}.bin")),
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
        println!("frames:        {} <- {}", dir.display(), from.display());
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
    // Many interpreters importing CUDA torch at once on Windows sometimes fail to start (packet
    // M15/R1); such a failure is retried with a fresh interpreter, which changes nothing.
    retry_start(interpreter, || {
        let out = cmd.output().map_err(|e| {
            let msg = format!("{interpreter}: {e}\nSet ES_PYTHON or [run] interpreter.");
            (msg, false)
        })?;
        let stderr = String::from_utf8_lossy(&out.stderr);
        let printed = !out.stdout.trim_ascii().is_empty();
        let transient = transient_start_failure(out.status.code(), &stderr, printed);
        if !out.status.success() {
            // The exit status too: a process the OS ends (an access violation, a failed DLL
            // load) leaves no Python traceback, and an empty message says nothing (review M14
            // N-5).
            let msg = format!(
                "{interpreter} cannot import what the {} route needs ({}):\n{}",
                route.as_str(),
                exit_words(out.status),
                stderr.trim_end()
            );
            return Err((msg, transient));
        }
        serde_json::from_slice(&out.stdout)
            .map_err(|e| (format!("{interpreter}: the probe printed {e}"), transient))
    })
    .map_err(|(msg, attempts)| match attempts {
        1 => bad(msg),
        n => bad(format!("{msg}\n(after {n} start attempts)")),
    })
}

/// An exit status in words, the Windows NTSTATUS in hex where it is one (0xC0000005 is an
/// access violation, 0xC0000135 a DLL that did not load).
fn exit_words(status: std::process::ExitStatus) -> String {
    match status.code() {
        Some(code) if code < 0 => format!("exit status 0x{:08X}", code as u32),
        Some(code) => format!("exit status {code}"),
        None => "ended by a signal".to_owned(),
    }
}

/// Runs the one subprocess. `capture` is for `train_act.py`, whose whole report is a single
/// JSON line on stdout; `lerobot-train` streams a progress log instead and inherits.
///
/// With someone listening -- a publisher, or `es loop cycle`'s preview queue -- the run is
/// read while it goes: the captured path becomes a *streamed* one, the same stdout read line
/// by line so a `{"progress": ...}` line reaches a viewer, and `lerobot-train`'s console is
/// relayed and read for its bar and metric lines (packet M12/R2). Either way each checkpoint
/// is bundled as soon as the trainer is past it (`early`, packet M13/Z1). The summary is still
/// the last line and still parsed the same way, which is what keeps `training.lock`
/// byte-identical (packet M7/E7).
fn spawn(
    step: &Step,
    route: Route,
    batch: Option<u32>,
    watch: &mut TrainWatch<'_>,
    early: &mut Early<'_>,
) -> Result<Value, CliError> {
    let capture = route.captures_trainer_stdout();
    let mut cmd = Command::new(&step.prefix[0]);
    cmd.args(&step.prefix[1..]).args(&step.args);
    let extra = watch.trainer_flags(route);
    if !extra.is_empty() {
        println!("  + {}", extra.join(" "));
        cmd.args(&extra);
    }
    let named = || format!("{}: ", step.prefix[0]);
    let (ok, code, summary) = if capture && watch.listening() {
        stream(&mut cmd, watch, early).map_err(|e| bad(format!("{}{e}", named())))?
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
    } else if watch.listening() {
        let mut progress = LerobotProgress::default();
        let status = stream_lerobot(&mut cmd, |piece| {
            if let (Some(p), Some(row)) = (watch.publisher.as_deref(), progress.read(piece, batch))
            {
                p.train_row(row);
            }
            if let Some((past, _, _)) = training_bar(piece) {
                early.past(past, watch);
            }
        })
        .map_err(|e| bad(format!("{}{e}", named())))?;
        (status.success(), status.code(), Value::Null)
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
/// the same place. stdout is read on its own thread, so the trainer never waits on a full pipe
/// while a mark is packed here. A line that is not one of the two the trainer publishes is
/// printed and remembered as a candidate summary, so the last non-progress line is the report
/// -- exactly what `Command::output`'s `text.lines().last()` picks.
fn stream(
    cmd: &mut Command,
    watch: &mut TrainWatch<'_>,
    early: &mut Early<'_>,
) -> std::io::Result<(bool, Option<i32>, Value)> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("the trainer's stdout was not piped"))?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut last = String::new();
    for line in rx {
        println!("{line}");
        let parsed = serde_json::from_str::<Value>(&line).unwrap_or(Value::Null);
        let publisher = watch.publisher.as_deref_mut();
        match (publisher, parsed.get("progress"), parsed.get("sample")) {
            (Some(p), Some(progress), _) => {
                if let Some(row) = progress_row(progress) {
                    p.train_row(row);
                }
            }
            (Some(p), None, Some(Value::String(path))) => p.sample(Path::new(path)),
            _ => {}
        }
        if let Some(past) = progress_step(&parsed) {
            early.past(past, watch);
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

/// One `{"progress": {...}}` object of `train_act.py` / `train_ppo.py` as the stream-5 row
/// `[step, loss, lr, samples_per_s]` (packet M7/E7); a line without a step or a loss is none.
///
/// A `null` loss is NaN (packet P-M14-R1): JSON has no NaN, so the trainers write a non-finite
/// loss as `null`, and a diverged run must be drawn as one -- the editor's light reads a
/// non-finite loss as broken. An absent `lr` or rate is NaN, never a stand-in.
fn progress_row(progress: &Value) -> Option<[f64; 4]> {
    let at = |k: &str| progress.get(k).and_then(Value::as_f64);
    let loss = match progress.get("loss")? {
        Value::Null => f64::NAN,
        v => v.as_f64()?,
    };
    Some([
        at("step")?,
        loss,
        at("lr").unwrap_or(f64::NAN),
        at("samples_per_s").unwrap_or(f64::NAN),
    ])
}

/// The step `train_act.py` or `train_ppo.py` says it has finished: `{"progress": {"step": N}}`.
fn progress_step(line: &Value) -> Option<u64> {
    line.get("progress")?.get("step")?.as_u64()
}

/// The checkpoints bundled while the trainer is still running (packet M13/Z1): the plan's own
/// pack or import step for a mark, run as soon as [`CheckpointWatch`] says the mark is written,
/// and skipped by the plan loop afterwards.
///
/// A bundle that fails here is not the run's failure: the mark stays in the plan, and the loop
/// after the trainer bundles it the way it always has, and fails there if it still does.
struct Early<'p> {
    plan: &'p Plan,
    out: &'p Path,
    marks: CheckpointWatch,
    rows: &'p mut Vec<Value>,
}

impl Early<'_> {
    /// The trainer says it is past optimizer step (or iteration) `step`.
    fn past(&mut self, step: u64, watch: &mut TrainWatch<'_>) {
        for mark in self.marks.past(step) {
            let Some(s) = self.plan.steps.iter().find(|s| {
                s.step == Some(mark)
                    && matches!(s.kind, StepKind::PolicyPack | StepKind::PolicyImportLerobot)
            }) else {
                continue;
            };
            println!("$ {}", one_line(s));
            let made = if s.kind == StepKind::PolicyPack {
                crate::cmd::policy::pack(&s.args)
            } else {
                crate::cmd::policy::import_lerobot(&s.args)
            };
            match made.and_then(|_| published_row(s, self.out, watch)) {
                Ok(row) => self.rows.push(row),
                Err(e) => println!(
                    "note: checkpoint {mark} was not bundled while training ({e}); it is \
                     bundled again after the trainer exits"
                ),
            }
        }
    }
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
    if let (Some(hook), Some(mark)) = (watch.on_checkpoint.as_deref_mut(), step.step) {
        hook(mark);
    }
    Ok(row)
}

/// The word after `flag` on a step's command line.
fn arg<'s>(step: &'s Step, flag: &str) -> Option<&'s String> {
    step.args.iter().skip_while(|a| *a != flag).nth(1)
}

/// One `checkpoint.manifest` row, read back from the bundle that was just written so the
/// digest names bytes on disk rather than bytes in memory.
fn manifest_row(step: &Step, out: &Path) -> Result<Value, CliError> {
    let path = arg(step, "--out").ok_or_else(|| bad("a checkpoint step with no --out"))?;
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

fn metrics(out: &Path, summary: &Value, route: Route) -> Value {
    // The RL route's curve is per iteration and carries nine fields per row, not one loss
    // per optimizer step, so it is a different file rather than the same name holding two
    // shapes (packet M8/S4b).
    let name = match route {
        Route::Rl => "metrics/loss-curve.json",
        _ => "metrics/loss.json",
    };
    let mut curve: Option<Value> = std::fs::read_to_string(out.join(name))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    // `metrics.json` is a `training_hash` slot, so what goes in it has to be what the run
    // *computed* -- and `samples_per_sec` is a measurement of the machine, not of the run.
    // The same recipe on a faster box is the same run, and leaving a wall-clock number in
    // here would make `training_hash` unreproducible by construction, which is the one thing
    // spec 3.5 tier 1 asks of it. The number stays in `metrics/loss-curve.json` on disk,
    // where a person reads it, and the machine it describes is `hardware.json`'s (M8/S4b).
    if route == Route::Rl {
        if let Some(Value::Array(rows)) = &mut curve {
            for row in rows {
                if let Some(row) = row.as_object_mut() {
                    row.remove("samples_per_sec");
                }
            }
        }
    }
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
        "trainer": match route {
            Route::Ir => TRAIN_ACT,
            Route::Rl => TRAIN_PPO,
            Route::External => "lerobot-train",
        },
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

// --- `lerobot-train`'s console (packet M12/R2) ------------------------------------------

/// What one `\r`- or `\n`-delimited piece of `lerobot-train`'s console says. tqdm's bar and
/// the logger both write stderr, so a metric line can land on the end of the bar it
/// interrupted, in the same piece.
#[derive(Debug, Default, PartialEq)]
struct LerobotSaid {
    /// `Training:  70%|███████   | 3510/5000 [03:51<01:39, 14.94step/s]`: the exact step, the
    /// total, and steps per second once tqdm has measured one.
    tqdm: Option<(u64, u64, Option<f64>)>,
    /// `… step:3K smpl:26K ep:49 epch:0.25 loss:0.131 grdn:12.233 lr:1.0e-04 …`, every
    /// `log_freq` steps. Its `step:` is abbreviated and is not read.
    loss: Option<f64>,
    lr: Option<f64>,
}

fn lerobot_said(piece: &str) -> LerobotSaid {
    let token = |key: &str| {
        piece
            .split_whitespace()
            .find_map(|t| t.strip_prefix(key)?.parse().ok())
    };
    LerobotSaid {
        tqdm: training_bar(piece),
        loss: token("loss:"),
        lr: token("lr:"),
    }
}

/// The training bar only: the pretrained backbone's download is a tqdm bar too.
fn training_bar(piece: &str) -> Option<(u64, u64, Option<f64>)> {
    let bar = piece.split_once("Training:")?.1;
    let (counts, timing) = bar.splitn(3, '|').nth(2)?.trim_start().split_once(" [")?;
    let (step, total) = counts.split_once('/')?;
    let rate = timing.split_once(", ")?.1.split(']').next()?.trim();
    let rate = match (rate.strip_suffix("step/s"), rate.strip_suffix("s/step")) {
        (Some(per_s), _) => per_s.parse().ok(),
        (_, Some(s_per)) => s_per.parse().ok().map(|s: f64| 1.0 / s),
        _ => None,
    };
    Some((step.parse().ok()?, total.parse().ok()?, rate))
}

/// The latest bar, and the stream-5 rows the metric lines complete.
#[derive(Default)]
struct LerobotProgress {
    step: Option<u64>,
    rate: Option<f64>,
}

impl LerobotProgress {
    /// `[step, loss, lr, samples_per_s]` when this piece gives a loss and a bar has already
    /// given the exact step; nothing otherwise, never a stand-in for the loss.
    fn read(&mut self, piece: &str, batch: Option<u32>) -> Option<[f64; 4]> {
        let said = lerobot_said(piece);
        if let Some((step, _, rate)) = said.tqdm {
            (self.step, self.rate) = (Some(step), rate);
        }
        Some([
            self.step? as f64,
            said.loss?,
            said.lr.unwrap_or(f64::NAN),
            self.rate
                .zip(batch)
                .map_or(f64::NAN, |(r, b)| r * f64::from(b)),
        ])
    }
}

/// Copies one of the trainer's pipes to ours as it arrives, byte for byte, and hands each
/// `\r`- or `\n`-delimited piece on. It keeps draining after a failed write, so the trainer
/// never blocks on a full pipe.
fn relay(mut from: impl Read, mut to: impl Write, mut on_piece: impl FnMut(String)) {
    let (mut buf, mut piece) = ([0u8; 4096], Vec::new());
    loop {
        let n = match from.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        let _ = to.write_all(&buf[..n]).and_then(|()| to.flush());
        for &b in &buf[..n] {
            if b != b'\r' && b != b'\n' {
                piece.push(b);
            } else if !piece.is_empty() {
                on_piece(String::from_utf8_lossy(&piece).into_owned());
                piece.clear();
            }
        }
    }
    if !piece.is_empty() {
        on_piece(String::from_utf8_lossy(&piece).into_owned());
    }
}

/// `lerobot-train` with someone listening: both pipes relayed to ours, and every piece of what
/// it said handed to `on_piece`. Nothing it prints is written anywhere new.
fn stream_lerobot(
    cmd: &mut Command,
    mut on_piece: impl FnMut(&str),
) -> std::io::Result<std::process::ExitStatus> {
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let (tx, rx) = std::sync::mpsc::channel();
    if let Some(out) = child.stdout.take() {
        let tx = tx.clone();
        std::thread::spawn(move || relay(out, std::io::stdout(), |p| drop(tx.send(p))));
    }
    if let Some(err) = child.stderr.take() {
        let tx = tx.clone();
        std::thread::spawn(move || relay(err, std::io::stderr(), |p| drop(tx.send(p))));
    }
    drop(tx);
    // Ends when both relays have seen their pipe close.
    for piece in rx {
        on_piece(&piece);
    }
    child.wait()
}

/// Which checkpoint marks the trainer has finished writing (packet M13/Z1). One rule on every
/// route: a mark is complete when its files are on disk **and** the trainer has said it is
/// past the mark -- because each trainer says where it is *before* it writes the mark:
///
/// * `train_act.py` / `train_ppo.py` print `{"progress": {"step": N}}` and then write
///   `weights/model-N.safetensors`, so the first progress line past N comes after the write
///   returned;
/// * `lerobot-train` 0.6.1 draws the bar for step N (`progbar.update(1)`) and then saves step
///   N, writing `model.safetensors` and `config.json` before the processor files
///   `import-lerobot` also reads -- a directory holding both can still be half written, and a
///   stats file missing then is read as the older layout. The next bar past N is drawn after
///   the save returned.
///
/// The last mark is the run's length: nothing is said past it, so it is bundled after the
/// trainer exits, as before.
struct CheckpointWatch {
    /// `(mark, the files it is written as)`, not yet reported.
    pending: Vec<(u32, Vec<PathBuf>)>,
}

impl CheckpointWatch {
    /// Every mark the plan bundles, with what the trainer writes for it.
    fn of(plan: &Plan) -> Self {
        let pending = plan
            .steps
            .iter()
            .filter_map(|s| {
                let files = match s.kind {
                    StepKind::PolicyPack => vec![PathBuf::from(arg(s, "--weights")?)],
                    StepKind::PolicyImportLerobot => {
                        let dir = Path::new(arg(s, "--checkpoint")?);
                        vec![dir.join("model.safetensors"), dir.join("config.json")]
                    }
                    _ => return None,
                };
                Some((s.step?, files))
            })
            .collect();
        Self { pending }
    }

    /// The marks complete now that the trainer is past `step`, each reported once.
    fn past(&mut self, step: u64) -> Vec<u32> {
        let (ready, pending): (Vec<_>, Vec<_>) =
            self.pending.drain(..).partition(|(mark, files)| {
                step > u64::from(*mark) && files.iter().all(|f| f.is_file())
            });
        self.pending = pending;
        ready.into_iter().map(|(mark, _)| mark).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use es_data::training::{ir_checkpoint, lerobot_checkpoint, Plan, Recipe};
    use serde_json::{json, Value};

    use super::{
        lerobot_said, progress_row, progress_step, relay, training_bar, CheckpointWatch,
        LerobotProgress, LerobotSaid,
    };

    // Copied from a real `lerobot-train` 0.6.1 run (Y-V item 1, `target/yv/cube-cam/runs/001/
    // es.log`). tqdm and the logger both write stderr, so a metric line lands on the end of
    // the bar it interrupted, with no separator between them.
    const BAR: &str = "Training:  70%|███████   | 3510/5000 [03:51<01:39, 14.94step/s]";
    const METRIC: &str = "INFO 2026-09-29 11:29:52 ot_train.py:641 step:3K smpl:26K ep:49 \
                          epch:0.25 loss:0.131 grdn:12.233 lr:1.0e-04 updt_s:0.062 \
                          data_s:0.001 smp/s:127 mem_gb:0.94 l1_loss:0.092 kld_loss:0.004";
    const BAR_THEN_METRIC: &str = "Training:  64%|██████▍   | 3200/5000 [03:31<01:55, \
                                   15.62step/s]INFO 2026-09-29 11:29:52 ot_train.py:641 \
                                   step:3K smpl:26K ep:49 epch:0.25 loss:0.131 grdn:12.233 \
                                   lr:1.0e-04 updt_s:0.062 data_s:0.001 smp/s:127 \
                                   mem_gb:0.94 l1_loss:0.092 kld_loss:0.004";
    const FIRST: &str = "Training:   0%|          | 0/5000 [00:00<?, ?step/s]INFO 2026-09-29 \
                         11:26:21 ot_train.py:597 Start offline training on a fixed dataset, \
                         with effective batch size: 8";
    const DOWNLOAD: &str = " 84%|████████▎ | 37.4M/44.7M [00:00<00:00, 58.9MB/s]";
    const RUN: &str = "\rTraining:   0%|          | 1/5000 [00:12<17:29:51, 12.60s/step]\
                       \rTraining:   0%|          | 3/5000 [00:12<4:36:11,  3.32s/step] \
                       \rTraining:   0%|          | 5/5000 [00:12<2:17:01,  1.65s/step]";

    #[test]
    fn the_tqdm_form_is_step_total_rate() {
        let said = lerobot_said(BAR);
        assert_eq!(said.tqdm, Some((3510, 5000, Some(14.94))));
        assert_eq!(said.loss, None);
        // Before tqdm has a rate it prints `?step/s`: the step is known, the rate is not.
        assert_eq!(lerobot_said(FIRST).tqdm, Some((0, 5000, None)));
    }

    #[test]
    fn the_metric_form_is_loss_and_lr() {
        let said = lerobot_said(METRIC);
        assert_eq!(
            said,
            LerobotSaid {
                tqdm: None,
                loss: Some(0.131),
                lr: Some(1.0e-4),
            }
        );
        // Both in one piece, the way stderr carries them.
        let both = lerobot_said(BAR_THEN_METRIC);
        assert_eq!(both.tqdm, Some((3200, 5000, Some(15.62))));
        assert_eq!((both.loss, both.lr), (Some(0.131), Some(1.0e-4)));
    }

    /// Packet P-M14-R1: a diverged run is drawn as one. The trainer writes a non-finite loss as
    /// `null` (JSON has no NaN) and `lerobot-train` prints `loss:nan`; both reach the published
    /// row as NaN, which the editor's light judges `Broken`. An absent loss is still no row.
    #[test]
    #[allow(clippy::float_cmp)] // exact: the row carries the parsed values themselves
    fn a_nonfinite_loss_is_published_as_nan() {
        let row = progress_row(&json!({"step": 1130, "loss": null, "lr": 4e-4,
                                       "samples_per_s": 8.0, "elapsed_s": 1.0}))
        .expect("a null loss is a row");
        assert_eq!(row[0], 1130.0);
        assert!(row[1].is_nan(), "{row:?}");
        assert_eq!((row[2], row[3]), (4e-4, 8.0));
        let fine = progress_row(&json!({"step": 10, "loss": 0.25, "lr": 1e-4}))
            .expect("a finite loss is a row");
        assert_eq!(fine[..2], [10.0, 0.25]);
        assert!(
            fine[3].is_nan(),
            "no samples_per_s is NaN, never a stand-in"
        );
        assert_eq!(progress_row(&json!({"step": 10})), None);

        let diverged = METRIC.replace("loss:0.131", "loss:nan");
        assert!(lerobot_said(&diverged).loss.is_some_and(f64::is_nan));
        let mut progress = LerobotProgress::default();
        progress.read(BAR, Some(8));
        let row = progress.read(&diverged, Some(8)).expect("a row");
        assert!(row[1].is_nan(), "{row:?}");
    }

    #[test]
    fn a_piece_with_neither_says_nothing() {
        // The pretrained backbone's download bar is a tqdm bar too, but not the training one.
        assert_eq!(lerobot_said(DOWNLOAD), LerobotSaid::default());
        let start = &FIRST[FIRST.find("INFO").unwrap()..];
        assert_eq!(lerobot_said(start), LerobotSaid::default());
        assert_eq!(lerobot_said(""), LerobotSaid::default());
    }

    #[test]
    fn a_run_of_tqdm_updates_the_last_one_wins() {
        // `\r\n`: the Windows log's own line end after a metric line.
        let console = format!("{RUN}\r{METRIC}\r\n");
        let (mut echoed, mut progress, mut rows) = (Vec::new(), LerobotProgress::default(), vec![]);
        relay(console.as_bytes(), &mut echoed, |p| {
            rows.extend(progress.read(&p, Some(8)));
        });
        // What the trainer wrote reaches the console byte for byte.
        assert_eq!(echoed, console.as_bytes());
        // `1.65s/step` is 1 / 1.65 steps per second, times the batch of 8.
        assert_eq!(rows, [[5.0, 0.131, 1.0e-4, (1.0 / 1.65) * 8.0]]);
    }

    #[test]
    fn a_metric_line_before_any_bar_is_no_row_yet() {
        let mut progress = LerobotProgress::default();
        assert_eq!(progress.read(METRIC, Some(8)), None);
        assert_eq!(
            progress.read(BAR, Some(8)),
            None,
            "a bar alone carries no loss"
        );
        assert_eq!(
            progress.read(METRIC, Some(8)),
            Some([3510.0, 0.131, 1.0e-4, 14.94 * 8.0])
        );
    }

    // --- packet M13/Z1: which marks are written --------------------------------------------

    /// A fresh directory for one test, gone before it starts.
    fn scratch(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("es-z1-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// The plan `es train` runs for `recipe` under `out` -- the steps whose files are watched.
    fn plan(recipe: &str, out: &Path) -> Plan {
        let recipe = Recipe::parse(recipe).expect("the recipe parses");
        let trainer = ["lerobot-train".to_owned()];
        Plan::build(&recipe, out, "python", &trainer, Some(6), false).expect("the plan builds")
    }

    const TRAIN_RUN: &str = "steps = 2000\nlr = 1e-4\nseed = 0\ncheckpoint_at = [1000]\n\
                       device = \"cpu\"\n";

    /// What the runtime does with one piece of `lerobot-train`'s console, and with one stdout
    /// line of `train_act.py` / `train_ppo.py`.
    fn bar(watch: &mut CheckpointWatch, piece: &str) -> Vec<u32> {
        training_bar(piece).map_or_else(Vec::new, |(step, _, _)| watch.past(step))
    }
    fn line(watch: &mut CheckpointWatch, line: &Value) -> Vec<u32> {
        progress_step(line).map_or_else(Vec::new, |step| watch.past(step))
    }
    fn progress(step: u64) -> Value {
        json!({"progress": {"step": step, "loss": 0.1, "lr": 1e-4, "samples_per_s": 8.0,
               "elapsed_s": 1.0}})
    }

    /// Review focus 2: a checkpoint `lerobot-train` is still writing is never imported. The
    /// directory is written in stages, the way `save_checkpoint` writes it.
    #[test]
    fn a_lerobot_mark_is_complete_once_a_bar_past_it_is_read_with_both_files_on_disk() {
        let out = scratch("lerobot");
        let recipe = format!(
            "kind = \"training\"\n[dataset]\nroot = \"ds\"\n[policy]\ntask = \"t.toml\"\n\
             observation = \"o.toml\"\ndeployment = \"d.toml\"\n\
             lerobot = {{ type = \"act\", chunk_size = 16, n_action_steps = 16 }}\n\
             [run]\nbatch = 8\n{TRAIN_RUN}"
        );
        let mut watch = CheckpointWatch::of(&plan(&recipe, &out));
        let dir = PathBuf::from(lerobot_checkpoint(&out, 1000));
        let at = |n: u64| format!("Training:  20%|██        | {n}/2000 [01:00<04:00, 16.00step/s]");

        assert!(bar(&mut watch, &at(1001)).is_empty(), "nothing on disk yet");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert!(bar(&mut watch, &at(1002)).is_empty(), "config.json alone");
        std::fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        // The bar for the mark itself is drawn before the save starts, and a metric line is
        // no bar at all.
        assert!(bar(&mut watch, &at(1000)).is_empty(), "the mark's own bar");
        assert!(bar(&mut watch, METRIC).is_empty(), "a metric line");
        assert_eq!(bar(&mut watch, &at(1003)), vec![1000]);
        assert!(bar(&mut watch, &at(1004)).is_empty(), "reported once");
        assert_eq!(
            watch.pending.len(),
            1,
            "2000, the run's length, waits for the exit"
        );
        let _ = std::fs::remove_dir_all(&out);
    }

    /// The IR route: `train_act.py` prints step N's progress line and then writes
    /// `model-N.safetensors`, so only a line past N, with the file there, completes N.
    #[test]
    fn an_ir_mark_is_complete_once_a_progress_line_past_it_is_read_with_its_weights() {
        let out = scratch("ir");
        let recipe = format!(
            "kind = \"training\"\n[dataset]\nroot = \"ds\"\n[policy]\nbundle = \"b.esb\"\n\
             [run]\nbatch = 8\n{TRAIN_RUN}"
        );
        let mut watch = CheckpointWatch::of(&plan(&recipe, &out));
        assert!(
            line(&mut watch, &progress(1010)).is_empty(),
            "no weights yet"
        );
        let weights = PathBuf::from(ir_checkpoint(&out, 1000));
        std::fs::create_dir_all(weights.parent().unwrap()).unwrap();
        std::fs::write(&weights, b"weights").unwrap();
        // Step 1000's own line is printed before the write; a sample line and the summary
        // say nothing about where the trainer is.
        assert!(
            line(&mut watch, &progress(1000)).is_empty(),
            "the mark's own line"
        );
        assert!(line(&mut watch, &json!({"sample": "metrics/sample.bin"})).is_empty());
        assert!(line(&mut watch, &json!({"initial_loss": 1.0, "steps": 2000})).is_empty());
        assert_eq!(line(&mut watch, &progress(1010)), vec![1000]);
        assert!(
            line(&mut watch, &progress(1020)).is_empty(),
            "reported once"
        );
        assert_eq!(
            watch.pending.len(),
            1,
            "2000, the run's length, waits for the exit"
        );
        let _ = std::fs::remove_dir_all(&out);
    }

    /// The RL route: `train_ppo.py` numbers its progress lines by iteration, as its marks are,
    /// and writes mark 0 before the first iteration -- which the first line then completes.
    #[test]
    fn an_rl_mark_is_complete_once_a_progress_line_past_it_is_read_with_its_weights() {
        let out = scratch("rl");
        let recipe = "kind = \"training\"\n[policy]\nbundle = \"b.esb\"\n\
                      [rl]\nalgo = \"ppo\"\nenvs = 4\nhorizon = 8\nepochs = 1\n\
                      minibatches = 2\ngamma = 0.99\nlam = 0.95\nclip = 0.2\nentropy = 0.0\n\
                      value_coef = 0.5\n\
                      [run]\nsteps = 300\nlr = 3e-4\nseed = 0\ncheckpoint_at = [0, 100]\n\
                      device = \"cpu\"\n";
        let mut watch = CheckpointWatch::of(&plan(recipe, &out));
        let write = |mark: u32| {
            let weights = PathBuf::from(ir_checkpoint(&out, mark));
            std::fs::create_dir_all(weights.parent().unwrap()).unwrap();
            std::fs::write(&weights, b"weights").unwrap();
        };
        write(0);
        assert_eq!(line(&mut watch, &progress(10)), vec![0]);
        assert!(
            line(&mut watch, &progress(110)).is_empty(),
            "100 is not written yet"
        );
        write(100);
        assert!(
            line(&mut watch, &progress(100)).is_empty(),
            "the mark's own line"
        );
        assert_eq!(line(&mut watch, &progress(110)), vec![100]);
        assert_eq!(
            watch.pending.len(),
            1,
            "300, the run's length, waits for the exit"
        );
        let _ = std::fs::remove_dir_all(&out);
    }

    /// Packet M15/N3: the external route's export gets each camera's own frames. One camera
    /// mirrors the flat tiles, as before; two mirror `<frames>/<channel>/` each, and the v3
    /// export then carries two image features, each with its own camera's pixels -- where the
    /// old mirror linked one flat directory under every camera's name.
    #[test]
    fn mirror_frames_gives_each_camera_its_own_frames() {
        use std::collections::BTreeMap;

        use es_data::{Column, Dtype, Episode, FeatureSpec, Info, LeRobotDataset, LeRobotWriter};

        let byte = |k: usize, g: usize| (10 + 100 * k + g) as u8;
        for cameras in [&["top"][..], &["top", "wrist"]] {
            let dir = scratch(&format!("mirror-{}", cameras.len()));
            let (root, frames, into) = (dir.join("ds"), dir.join("frames"), dir.join("in"));
            let mut features =
                BTreeMap::from([("action".to_owned(), FeatureSpec::new(Dtype::Float32, [1]))]);
            for c in cameras {
                features.insert(
                    format!("observation.images.{c}"),
                    FeatureSpec::new(Dtype::Video, [2, 2, 3]),
                );
            }
            let mut writer = LeRobotWriter::create(&root, Info::new(50.0, features)).unwrap();
            writer
                .write_episode(&Episode {
                    index: 0,
                    tasks: vec!["t".to_owned()],
                    timestamps: vec![0.0, 0.02, 0.04],
                    task_index: vec![0; 3],
                    columns: BTreeMap::from([("action".to_owned(), Column::F32(vec![0.0; 3]))]),
                    video: BTreeMap::new(),
                })
                .unwrap();
            writer.finish().unwrap();
            // The layout `es loop collect --frames` writes: flat for one image channel.
            for (k, c) in cameras.iter().enumerate() {
                let at = if cameras.len() > 1 {
                    frames.join(c)
                } else {
                    frames.clone()
                };
                std::fs::create_dir_all(&at).unwrap();
                for g in 0..3 {
                    std::fs::write(at.join(format!("{g:06}.bin")), [byte(k, g); 12]).unwrap();
                }
            }

            let dataset = LeRobotDataset::open(&root).unwrap();
            super::mirror_frames(&frames, &dataset, &into).unwrap();
            for (k, c) in cameras.iter().enumerate() {
                for g in 0..3 {
                    let tile = std::fs::read(into.join(c).join(format!("{g:06}.bin"))).unwrap();
                    assert_eq!(tile, [byte(k, g); 12], "{c} {g}");
                }
            }
            let out = dir.join("v3");
            let report = es_data::export_v3(
                &dataset,
                &out,
                Some(&into),
                &es_data::ExportOptions::default(),
            )
            .unwrap();
            assert_eq!(report.cameras.len(), cameras.len(), "{report:?}");
            let stats: Value = serde_json::from_str(
                &std::fs::read_to_string(out.join("meta/stats.json")).unwrap(),
            )
            .unwrap();
            for (k, c) in cameras.iter().enumerate() {
                let max = &stats[format!("observation.images.{c}")]["max"][0][0][0];
                assert_eq!(max.as_f64(), Some(f64::from(byte(k, 2)) / 255.0), "{c}");
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Packet M15/R1's reproduction: 26 interpreter probes at once, which on Windows with CUDA
    /// torch lost one or more starts to the DLL loader. Every probe must now answer; a retried
    /// start prints `start attempt n of 3 failed` (run with `--nocapture` to count them).
    /// Ignored: it needs `ES_PYTHON` with torch and loads CUDA 26 times.
    #[test]
    #[ignore = "needs ES_PYTHON with torch; loads CUDA 26 times"]
    fn twenty_six_probes_at_once_all_answer() {
        let python = std::env::var("ES_PYTHON").expect("ES_PYTHON names an interpreter");
        let failed: Vec<String> = std::thread::scope(|s| {
            let probes: Vec<_> = (0..26)
                .map(|_| s.spawn(|| super::probe(&python, super::Route::Ir)))
                .collect();
            probes
                .into_iter()
                .filter_map(|p| p.join().unwrap().err().map(|e| format!("{e:?}")))
                .collect()
        });
        assert!(
            failed.is_empty(),
            "{} of 26 failed: {failed:#?}",
            failed.len()
        );
    }
}
