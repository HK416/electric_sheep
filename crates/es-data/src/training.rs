//! The training recipe (spec 13.1, 19.3; packet M7/T1) — one document, one command.
//!
//! Read `docs/design/training-recipe.md` first. Everything here is headless: this module
//! parses the recipe, decides which of the two routes it describes, builds the **command
//! plan**, and assembles spec 19.3's `training/` bundle. It runs nothing. `crates/es/src/
//! cmd/train.rs` is the thin shell that executes the plan — the `es` steps in-process and
//! only the Python trainer as a subprocess (spec 2.3: Python is on the learning path only).
//!
//! Two digests, and the split between them is the point (spec 19.3, §28.10 rule 2):
//!
//! * `identity_hash` — the nine slots a run knows **before** the trainer starts (`config`,
//!   `optimizer`, `scheduler`, `seed`, `dataset`, `base_model`, `augmentation`, `precision`,
//!   `topology`). The same recipe, the same inputs and the same interpreter give one
//!   `identity_hash` in any output directory, so the name of a run exists before a single
//!   GPU-second is spent on it.
//! * `training_hash` — the same twelve slots with `checkpoint.manifest`, `metrics.json` and
//!   `hardware.json` filled in from what actually ran.
//!
//! A slot the run genuinely does not know is the file [`UNSET`], hashed as such. It is never
//! an all-zero digest and never a fabricated one: `es loop distill` writes zeros because it
//! is on the other side of spec 2.3's boundary and knows nothing; `es train` is on this side
//! and knows almost everything, so "unknown" has to be a value it can defend.

use std::collections::BTreeMap;
use std::path::Path;

use es_ir::observation::{ObservationIr, ObservationNode};
use es_ir::DatasetHash;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::collect::hex;
use crate::identity::{BaseModel, TrainingIdentity};
use crate::DataError;

/// `kind = "training"`.
pub const KIND: &str = "training";

/// The canonical body of a slot whose value this run does not know.
pub const UNSET: &str = "{\"unset\":true}\n";

/// The trainer of the IR route, relative to the repository root.
pub const TRAIN_ACT: &str = "python/es/train_act.py";

/// The columns `lerobot` would turn into extra action heads (packet M5/V8, V19).
pub const EXPORT_DROP: &str = "action_commanded,action_source,intervention";

/// The twelve files of spec 19.3's `training/`, in the order `TrainingIdentity` hashes them.
pub const FILES: [&str; 12] = [
    "config.json",
    "optimizer.json",
    "scheduler.json",
    "seed.json",
    "dataset.lock",
    "base_model.lock",
    "augmentation.json",
    "precision.json",
    "topology.json",
    "checkpoint.manifest",
    "metrics.json",
    "hardware.json",
];

fn refuse(msg: impl Into<String>) -> DataError {
    DataError::Loop(msg.into())
}

// --- the recipe ------------------------------------------------------------------------------

/// `training.toml` — the one document `es train` reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub kind: String,
    pub dataset: DatasetRef,
    pub policy: PolicyRef,
    pub run: Run,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRef {
    /// A `LeRobot` v2.1 root, as `es loop collect` writes it.
    pub root: String,
    /// The flat `<NNNNNN>.bin` tiles beside it (`es loop collect --frames`). Required when
    /// the Observation IR has an image input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<String>,
}

/// Exactly one of `bundle` (the IR route) and `lerobot` (the external route).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lerobot: Option<Lerobot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment: Option<String>,
}

/// A policy designed outside this project, trained by its own trainer (spec 8.3, packet
/// M5/V8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lerobot {
    /// `lerobot`'s `--policy.type`.
    #[serde(rename = "type")]
    pub kind: String,
    pub chunk_size: u32,
    pub n_action_steps: u32,
    /// Passed to `lerobot-train` verbatim, after everything this module derives.
    #[serde(default)]
    pub extra: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Run {
    pub steps: u32,
    pub batch: u32,
    pub lr: f64,
    pub seed: u64,
    /// Optimizer steps to also keep a checkpoint at. `steps` is always one of them.
    #[serde(default)]
    pub checkpoint_at: Vec<u32>,
    pub device: String,
    /// Overridden by `ES_PYTHON` when it is set (the same override every other oracle uses).
    #[serde(default = "default_interpreter")]
    pub interpreter: String,
    /// The learning-rate schedule (packet M7/T4). Absent is `constant`, and absent means
    /// **absent**: no flag on the trainer's command line and the same `scheduler.json` as
    /// before T4, so a recipe written for the measured runs keeps its `identity_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<Schedule>,
    /// `AdamW`'s weight decay. Absent is torch's own `1e-2`, which is what `optimizer.json`
    /// has always declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,
    /// Gradient-norm clip. Absent is off, and `optimizer.json` then names no clip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip: Option<f64>,
}

/// `[run] schedule = { kind = "warmup_cosine", warmup = 250, lr_min = 1e-6 }` (packet M7/T4,
/// spec 19.3's `scheduler.json`).
///
/// The shape of the schedule is the trainer's — `python/es/train_act.py::lr_at`, pinned by
/// `tests/golden/train/lr_warmup_cosine.json` — and this is the document that names it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schedule {
    /// `constant` or `warmup_cosine`.
    pub kind: String,
    /// Optimizer steps of linear warmup from 0 to `[run] lr`.
    #[serde(default)]
    pub warmup: u32,
    /// The floor the cosine decays to.
    #[serde(default)]
    pub lr_min: f64,
}

fn default_interpreter() -> String {
    "python".to_owned()
}

/// Which of the two paths of §28.10's "as built" table this recipe describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// `bake -> lower -> train_act.py -> pack`: the Learning IR is the architecture.
    Ir,
    /// `export -> lerobot-train -> import-lerobot`: the architecture is `lerobot`'s.
    External,
}

impl Route {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ir => "ir",
            Self::External => "external",
        }
    }
}

impl Recipe {
    /// Parses and refuses by name. TOML comes through `es_ir::serial`, which is where this
    /// repository's TOML reader already lives — a recipe is a document like every other one.
    pub fn parse(text: &str) -> Result<Self, DataError> {
        let recipe: Self = es_ir::serial::parse_toml(text)
            .map_err(|e| refuse(format!("the recipe does not parse: {e}")))?;
        let route = recipe.route()?;
        recipe.marks()?;
        // Unconditionally, not behind the route test: a schedule that cannot run is refused
        // on both routes, and on this one for a second reason as well.
        let schedule_args = recipe.schedule_args()?;
        if route == Route::External && !schedule_args.is_empty() {
            return Err(refuse(
                "[run] `schedule`, `weight_decay` and `grad_clip` are the IR route's: they \
                 are `python/es/train_act.py`'s flags, and `lerobot-train` carries its own \
                 optimizer and scheduler configuration. Declaring one here would put a \
                 schedule into `scheduler.json` that the run never applied; reach lerobot's \
                 own through `[policy] lerobot.extra`",
            ));
        }
        Ok(recipe)
    }

    /// The trainer flags `[run] schedule`, `weight_decay` and `grad_clip` add (packet M7/T4).
    ///
    /// Empty when the recipe sets none of them — which is what keeps the command plan, and
    /// its golden, byte-identical for a recipe written before this packet.
    pub fn schedule_args(&self) -> Result<Vec<String>, DataError> {
        let run = &self.run;
        let mut args = Vec::new();
        match run.schedule.as_ref().map(|s| (s.kind.as_str(), s)) {
            None => {}
            Some(("constant", schedule)) => {
                if schedule.warmup != 0 || schedule.lr_min != 0.0 {
                    return Err(refuse(
                        "[run] `schedule.kind` is \"constant\" and it sets `warmup` or \
                         `lr_min`; a constant schedule has neither. \
                         `kind = \"warmup_cosine\"` is the one that does",
                    ));
                }
            }
            Some(("warmup_cosine", schedule)) => {
                if schedule.warmup >= run.steps {
                    return Err(refuse(format!(
                        "[run] `schedule.warmup` is {} and the run is {} steps: the learning \
                         rate would never leave the ramp",
                        schedule.warmup, run.steps
                    )));
                }
                if !schedule.lr_min.is_finite() || !(0.0..run.lr).contains(&schedule.lr_min) {
                    return Err(refuse(format!(
                        "[run] `schedule.lr_min` is {} and `lr` is {}: the floor of the \
                         cosine has to be finite, not negative, and below the peak",
                        schedule.lr_min, run.lr
                    )));
                }
                args.extend([
                    s("--schedule"),
                    s("warmup_cosine"),
                    s("--warmup-steps"),
                    schedule.warmup.to_string(),
                    s("--lr-min"),
                    schedule.lr_min.to_string(),
                ]);
            }
            Some((other, _)) => {
                return Err(refuse(format!(
                    "[run] `schedule.kind` is {other:?}; it is \"constant\" or \
                     \"warmup_cosine\""
                )))
            }
        }
        for (field, value) in [
            ("weight-decay", run.weight_decay),
            ("grad-clip", run.grad_clip),
        ] {
            let Some(value) = value else { continue };
            if !value.is_finite() || value < 0.0 {
                return Err(refuse(format!(
                    "[run] `{}` is {value}",
                    field.replace('-', "_")
                )));
            }
            args.push(format!("--{field}"));
            args.push(value.to_string());
        }
        Ok(args)
    }

    /// `bundle` xor `lerobot`, and the external route needs the three documents the import
    /// will carry into the bundle.
    pub fn route(&self) -> Result<Route, DataError> {
        match (&self.policy.bundle, &self.policy.lerobot) {
            (Some(_), Some(_)) => Err(refuse(
                "[policy] sets both `bundle` and `lerobot`; a recipe describes one route -- \
                 `bundle` lowers this project's Learning IR, `lerobot` trains an external \
                 policy. Delete one",
            )),
            (None, None) => Err(refuse(
                "[policy] sets neither `bundle` nor `lerobot`; one of them names the policy \
                 to train",
            )),
            (Some(_), None) => Ok(Route::Ir),
            (None, Some(_)) => {
                for (flag, value) in [
                    ("task", &self.policy.task),
                    ("observation", &self.policy.observation),
                    ("deployment", &self.policy.deployment),
                ] {
                    if value.is_none() {
                        return Err(refuse(format!(
                            "[policy] `{flag}` is required on the lerobot route: an external \
                             checkpoint carries no IR, so `es policy import-lerobot` needs the \
                             Task, Observation and Deployment IR that will travel with it"
                        )));
                    }
                }
                Ok(Route::External)
            }
        }
    }

    /// The checkpoint marks, `run.steps` always among them and always last.
    ///
    /// `train_act.py` caps its run at the largest mark, so making `steps` a mark is what
    /// keeps `run.steps` the authority on how long the run is.
    pub fn marks(&self) -> Result<Vec<u32>, DataError> {
        if self.run.steps == 0 {
            return Err(refuse("[run] `steps` is 0; there is nothing to train"));
        }
        if self.run.batch == 0 {
            return Err(refuse("[run] `batch` is 0"));
        }
        if !self.run.lr.is_finite() || self.run.lr <= 0.0 {
            return Err(refuse(format!("[run] `lr` is {}", self.run.lr)));
        }
        let mut marks: Vec<u32> = self.run.checkpoint_at.clone();
        marks.push(self.run.steps);
        marks.sort_unstable();
        marks.dedup();
        if let Some(bad) = marks.iter().find(|m| **m == 0 || **m > self.run.steps) {
            return Err(refuse(format!(
                "[run] `checkpoint_at` holds {bad}, which is not an optimizer step of a \
                 {}-step run",
                self.run.steps
            )));
        }
        Ok(marks)
    }

    /// `lerobot-train` saves at one frequency, not at a list, so the marks have to be its
    /// multiples. Refused rather than rounded: a checkpoint that is not on disk cannot be
    /// imported, and silently moving a mark would put the wrong step number in
    /// `checkpoint.manifest`.
    pub fn save_freq(&self) -> Result<u32, DataError> {
        let marks = self.marks()?;
        let freq = marks[0];
        if let Some(bad) = marks.iter().find(|m| *m % freq != 0) {
            return Err(refuse(format!(
                "[run] `checkpoint_at` is {marks:?} on the lerobot route, and `lerobot-train` \
                 saves at a single --save_freq: {bad} is not a multiple of {freq}"
            )));
        }
        Ok(freq)
    }
}

/// How many leading values of the recorded `observation.state` row the Observation IR reads.
///
/// The row is `qpos ‖ qvel` (`es_data::collect::to_lerobot`), and every `StateInput` reads a
/// slice of it; the widths add up because the demo's two channels — six joint angles and the
/// cube's seven-wide free joint — are exactly `qpos[0..6]` and `qpos[6..13]`, which is the 13
/// packet M5/V19 exported by hand. It is a **sum, not a resolved range**: resolving a source
/// id against `ModelInfo` needs a scene, and `es-data` is layer 10 beside `es-eval`, so this
/// side cannot ask. Two channels that overlap would over-count, and the export would then
/// carry columns nothing reads — wasteful, never wrong.
pub fn state_dim(obs: &ObservationIr) -> usize {
    obs.graph
        .nodes
        .values()
        .filter_map(|node| match node {
            ObservationNode::StateInput { io, .. } => Some(io.output.shape.dims()),
            _ => None,
        })
        .map(|dims| dims.iter().product::<u64>() as usize)
        .sum()
}

/// Does the Observation IR read pixels? Then the recipe owes `[dataset] frames`.
pub fn has_image_input(obs: &ObservationIr) -> bool {
    obs.graph
        .nodes
        .values()
        .any(|n| matches!(n, ObservationNode::ImageInput { .. }))
}

/// The camera directory `es dataset export --frames` looks in, for a feature key like
/// `observation.images.rgb_overhead` (`lerobot::v3::camera_source`).
pub fn camera_suffix(feature: &str) -> &str {
    feature.rsplit_once('.').map_or(feature, |(_, tail)| tail)
}

// --- the command plan ------------------------------------------------------------------------

/// Which command a [`Step`] is, so the shell can call it in-process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    DatasetBake,
    DatasetExport,
    PolicyLower,
    PolicyPack,
    PolicyImportLerobot,
    /// The one subprocess (spec 2.3).
    Trainer,
}

/// One line of the plan: what it is, the words in front of its flags, and its flags.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub kind: StepKind,
    /// `["es", "dataset", "bake"]`, or the interpreter and the trainer.
    pub prefix: Vec<String>,
    pub args: Vec<String>,
    /// The optimizer step this checkpoint step belongs to, for `checkpoint.manifest`.
    pub step: Option<u32>,
}

/// The whole run as commands, before any of them is executed.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub route: Route,
    pub steps: Vec<Step>,
    pub marks: Vec<u32>,
}

fn s(v: impl AsRef<str>) -> String {
    v.as_ref().to_owned()
}

/// `out.join(rest)` as a string — the plan holds real paths and only [`Plan::render`] makes
/// them relative.
fn under(out: &Path, rest: &str) -> String {
    out.join(rest).to_string_lossy().into_owned()
}

/// `<out>/lerobot/checkpoints/<NNNNNN>/pretrained_model` (`docs/api-notes/lerobot-act.md`).
pub fn lerobot_checkpoint(out: &Path, step: u32) -> String {
    under(
        out,
        &format!("lerobot/checkpoints/{step:06}/pretrained_model"),
    )
}

/// `<out>/weights/model-<step>.safetensors` — `train_act.py --checkpoint-at` writes the mark
/// into the stem.
pub fn ir_checkpoint(out: &Path, step: u32) -> String {
    under(out, &format!("weights/model-{step}.safetensors"))
}

impl Plan {
    /// The plan is a function of the recipe, the output directory and the resolved
    /// interpreter — and, on the external route, of the Observation IR's state width.
    ///
    /// `trainer` is the program that runs the external trainer: the `lerobot-train` entry
    /// point beside the interpreter when there is one, otherwise the interpreter and
    /// `-m lerobot.scripts.lerobot_train`. Nothing on disk is read here, so `--dry-run` works
    /// on a machine that has neither the dataset nor the bundle.
    pub fn build(
        recipe: &Recipe,
        out: &Path,
        interpreter: &str,
        trainer: &[String],
        state_dim: Option<usize>,
    ) -> Result<Self, DataError> {
        let route = recipe.route()?;
        let marks = recipe.marks()?;
        let run = &recipe.run;
        let mut steps = Vec::new();
        let es = |a: &[&str]| -> Vec<String> { a.iter().map(|w| s(*w)).collect() };

        match route {
            Route::Ir => {
                let bundle = recipe.policy.bundle.clone().unwrap_or_default();
                let mut bake = vec![
                    s("--policy"),
                    bundle.clone(),
                    s("--out"),
                    under(out, "baked"),
                ];
                if let Some(frames) = &recipe.dataset.frames {
                    bake.push(s("--frames"));
                    bake.push(frames.clone());
                }
                bake.push(recipe.dataset.root.clone());
                steps.push(Step {
                    kind: StepKind::DatasetBake,
                    prefix: es(&["es", "dataset", "bake"]),
                    args: bake,
                    step: None,
                });
                steps.push(Step {
                    kind: StepKind::PolicyLower,
                    prefix: es(&["es", "policy", "lower"]),
                    args: vec![
                        s("--policy"),
                        bundle.clone(),
                        s("--out"),
                        under(out, "module"),
                    ],
                    step: None,
                });
                let marks_arg = marks
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                steps.push(Step {
                    kind: StepKind::Trainer,
                    prefix: vec![s(interpreter), s(TRAIN_ACT)],
                    args: vec![
                        s("--module"),
                        under(out, "module"),
                        s("--baked"),
                        under(out, "baked"),
                        s("--out"),
                        under(out, "weights/model.safetensors"),
                        s("--checkpoint-at"),
                        marks_arg,
                        s("--seed"),
                        run.seed.to_string(),
                        s("--batch"),
                        run.batch.to_string(),
                        s("--lr"),
                        run.lr.to_string(),
                        s("--device"),
                        run.device.clone(),
                        s("--loss-curve"),
                        under(out, "metrics/loss.json"),
                    ]
                    .into_iter()
                    // Appended, and only when the recipe asks for them (packet M7/T4): a
                    // recipe that names no schedule renders the plan it always did.
                    .chain(recipe.schedule_args()?)
                    .collect(),
                    step: None,
                });
                for mark in &marks {
                    steps.push(Step {
                        kind: StepKind::PolicyPack,
                        prefix: es(&["es", "policy", "pack"]),
                        args: vec![
                            s("--policy"),
                            bundle.clone(),
                            s("--weights"),
                            ir_checkpoint(out, *mark),
                            s("--out"),
                            under(out, &format!("checkpoints/{mark}.esb")),
                        ],
                        step: Some(*mark),
                    });
                }
            }
            Route::External => {
                let lerobot = recipe.policy.lerobot.clone().unwrap_or_else(|| Lerobot {
                    kind: String::new(),
                    chunk_size: 0,
                    n_action_steps: 0,
                    extra: Vec::new(),
                });
                let dim = state_dim.ok_or_else(|| {
                    refuse(
                        "the lerobot route needs the Observation IR's state width and it was \
                         not resolved",
                    )
                })?;
                let mut export = vec![
                    s("--lerobot-v3"),
                    recipe.dataset.root.clone(),
                    s("--out"),
                    under(out, "ds-v3"),
                ];
                if recipe.dataset.frames.is_some() {
                    // Not the recipe's path: `export` wants one directory per camera and
                    // `es loop collect --frames` writes a flat one, so the shell mirrors the
                    // tiles into `<out>/frames-in/<camera>` first. Packet M5/V19 did that by
                    // hand with `mkdir` and `ln -s`; a recipe that needed it would not be one
                    // command.
                    export.push(s("--frames"));
                    export.push(under(out, "frames-in"));
                }
                export.push(s("--drop"));
                export.push(s(EXPORT_DROP));
                export.push(s("--state-dim"));
                export.push(dim.to_string());
                steps.push(Step {
                    kind: StepKind::DatasetExport,
                    prefix: es(&["es", "dataset", "export"]),
                    args: export,
                    step: None,
                });

                let mut train = vec![
                    s("--dataset.repo_id=es/train"),
                    format!("--dataset.root={}", under(out, "ds-v3")),
                    format!("--policy.type={}", lerobot.kind),
                    format!("--policy.chunk_size={}", lerobot.chunk_size),
                    format!("--policy.n_action_steps={}", lerobot.n_action_steps),
                    format!("--policy.device={}", run.device),
                    s("--policy.push_to_hub=false"),
                    format!("--policy.optimizer_lr={}", run.lr),
                    format!("--steps={}", run.steps),
                    format!("--batch_size={}", run.batch),
                    format!("--seed={}", run.seed),
                    format!("--save_freq={}", recipe.save_freq()?),
                    format!("--output_dir={}", under(out, "lerobot")),
                    s("--job_name=es-train"),
                    s("--wandb.enable=false"),
                ];
                train.extend(lerobot.extra.iter().cloned());
                steps.push(Step {
                    kind: StepKind::Trainer,
                    prefix: trainer.to_vec(),
                    args: train,
                    step: None,
                });
                for mark in &marks {
                    steps.push(Step {
                        kind: StepKind::PolicyImportLerobot,
                        prefix: es(&["es", "policy", "import-lerobot"]),
                        args: vec![
                            s("--checkpoint"),
                            lerobot_checkpoint(out, *mark),
                            s("--task"),
                            recipe.policy.task.clone().unwrap_or_default(),
                            s("--observation"),
                            recipe.policy.observation.clone().unwrap_or_default(),
                            s("--deployment"),
                            recipe.policy.deployment.clone().unwrap_or_default(),
                            s("--out"),
                            under(out, &format!("checkpoints/{mark}.esb")),
                        ],
                        step: Some(*mark),
                    });
                }
            }
        }
        Ok(Self {
            route,
            steps,
            marks,
        })
    }

    /// One line per step, every path under `<out>` written relative to it and every separator
    /// a `/` — so two machines with different scratch directories, and Windows and Linux,
    /// render one plan and the golden is a property of the recipe alone.
    pub fn render(&self, out: &Path) -> String {
        // `<out>/` is stripped wherever it appears in a word, not only at its head: the
        // external trainer takes `--flag=value` and the value is the path.
        let prefix = format!("{}/", out.to_string_lossy().replace('\\', "/"));
        let mut text = format!("# route: {}\n", self.route.as_str());
        for step in &self.steps {
            let words = step
                .prefix
                .iter()
                .chain(&step.args)
                .map(|w| w.replace('\\', "/").replace(&prefix, ""));
            text.push_str(&words.collect::<Vec<_>>().join(" "));
            text.push('\n');
        }
        text
    }
}

// --- spec 19.3's `training/` -------------------------------------------------------------------

/// Compact, key-sorted, `float_roundtrip` JSON with a trailing newline.
///
/// `serde_json::Map` is a `BTreeMap` in this workspace (no `preserve_order` feature), so key
/// order is a property of the names and not of the code that built the value — which is what
/// makes the digest of one of these files reproducible (spec 3.4).
pub fn canon_json(value: &Value) -> String {
    let mut text = value.to_string();
    text.push('\n');
    text
}

/// What the dataset says about itself, read once by the shell and passed in here.
#[derive(Clone, Debug)]
pub struct DatasetFacts {
    pub hashes: DatasetHash,
    pub episodes: u32,
    pub frames: u64,
    /// The `es:task:<hex>` name `es loop collect` records, when there is one.
    pub recorded_task: Option<String>,
    /// Whether the split came from `es loop distill`'s `split.json` or is the all-train one.
    pub split_source: &'static str,
}

/// The spec 19.3 `training/` bundle held in memory: twelve files, each canonical JSON.
#[derive(Clone, Debug)]
pub struct Training {
    files: BTreeMap<String, String>,
    dataset: DatasetHash,
    base_source: String,
    base_license: String,
}

impl Training {
    /// The nine pre-run slots, from the recipe and the dataset alone. The three post-run ones
    /// are [`UNSET`] until [`Training::finish`].
    pub fn pre_run(
        recipe: &Recipe,
        plan: &Plan,
        out: &Path,
        interpreter: &str,
        data: &DatasetFacts,
    ) -> Result<Self, DataError> {
        let run = &recipe.run;
        let route = plan.route;
        // The plan is stored `<out>`-relative, exactly as `--dry-run` prints it: config.json
        // must not carry the scratch directory, or the same recipe would have two identities.
        let rendered = plan.render(out);
        let plan_lines: Vec<&str> = rendered.lines().skip(1).collect();
        let recipe_json = serde_json::to_value(recipe)
            .map_err(|e| refuse(format!("the recipe does not serialise: {e}")))?;

        // The betas and weight decay `train_act.py`'s `AdamW(params, lr=lr)` leaves at
        // torch's documented defaults: declared here, and compared against what the trainer
        // reports when the run ends. `lerobot`'s own optimizer block is not this side's to
        // declare, so it stays unset (T4 owns the optimizer and the schedule).
        let optimizer = match route {
            Route::Ir => {
                let mut optimizer = json!({
                    "kind": "AdamW", "lr": run.lr, "betas": [0.9, 0.999], "eps": 1e-8,
                    // Torch's own default until packet M7/T4, and now the number the trainer
                    // is *told* to use -- the same value, declared instead of assumed.
                    "weight_decay": run.weight_decay.unwrap_or(0.01),
                    "declared_by": "train_act.py",
                });
                if let Some(clip) = run.grad_clip {
                    // Only when there is one: no clip is the absence of a clip, and a key
                    // that appeared unconditionally would move every pre-T4 identity_hash.
                    optimizer["grad_clip"] = json!(clip);
                }
                optimizer
            }
            Route::External => json!({
                "kind": "AdamW", "lr": run.lr, "betas": {"unset": true},
                "weight_decay": {"unset": true}, "declared_by": "lerobot-train",
            }),
        };
        let base_model = match route {
            Route::Ir => json!({"source": "none"}),
            // A *declared* provenance: `lerobot`'s ACT builds `vision_backbone` with these
            // weights unless `extra` overrides it (docs/api-notes/lerobot-config.md). The
            // weights' own digest and licence are packet T5's; claiming them here would be
            // the fabrication spec 28.10 rule 2 forbids.
            Route::External => {
                let extra = recipe.policy.lerobot.as_ref().map(|l| &l.extra);
                let pick = |flag: &str, default: &str| -> String {
                    extra
                        .into_iter()
                        .flatten()
                        .find_map(|a| a.strip_prefix(flag).map(s))
                        .unwrap_or_else(|| s(default))
                };
                json!({
                    "source": "lerobot:torchvision",
                    "vision_backbone": pick("--policy.vision_backbone=", "resnet18"),
                    "pretrained_backbone_weights": pick(
                        "--policy.pretrained_backbone_weights=",
                        "ResNet18_Weights.IMAGENET1K_V1",
                    ),
                    "license": {"unset": true},
                    "weights_hash": {"unset": true},
                })
            }
        };
        let base_source = base_model["source"].as_str().unwrap_or("none").to_owned();

        let mut files = BTreeMap::new();
        let mut put = |name: &str, v: Value| {
            files.insert(s(name), canon_json(&v));
        };
        put(
            "config.json",
            json!({
                "schema_version": 1, "route": route.as_str(), "recipe": recipe_json,
                "interpreter": interpreter, "plan": plan_lines,
            }),
        );
        put("optimizer.json", optimizer);
        // `total_steps` is part of the schedule and not decoration: the cosine's period is
        // the length of the run, so two runs of one `warmup`/`lr_min` pair at different
        // `steps` are two schedules (packet M7/T4).
        put(
            "scheduler.json",
            match run.schedule.as_ref().filter(|s| s.kind == "warmup_cosine") {
                None => json!({"kind": "constant", "lr": run.lr}),
                Some(schedule) => json!({
                    "kind": "warmup_cosine", "lr": run.lr, "lr_min": schedule.lr_min,
                    "warmup": schedule.warmup, "total_steps": run.steps,
                }),
            },
        );
        put(
            "seed.json",
            json!({
                "global": run.seed, "dataloader": run.seed,
                "augmentation": {"unset": true},
            }),
        );
        put(
            "dataset.lock",
            json!({
                "root": recipe.dataset.root, "episodes": data.episodes, "frames": data.frames,
                "content": hex(&data.hashes.content), "schema": hex(&data.hashes.schema),
                "split": hex(&data.hashes.split), "split_source": data.split_source,
                "recorded_task": data.recorded_task,
            }),
        );
        put("base_model.lock", base_model);
        put("augmentation.json", json!({"kind": "none"}));
        put(
            "precision.json",
            json!({
                "dtype": "fp32", "amp": "off",
                "gradient_accumulation": match route { Route::Ir => run.batch, Route::External => 1 },
            }),
        );
        put("topology.json", json!({"world_size": 1}));
        for name in ["checkpoint.manifest", "metrics.json", "hardware.json"] {
            files.insert(s(name), s(UNSET));
        }
        Ok(Self {
            files,
            dataset: data.hashes,
            base_source,
            base_license: s("unset"),
        })
    }

    /// The three post-run slots, from what the run actually produced.
    pub fn finish(&mut self, checkpoints: &Value, metrics: &Value, hardware: &Value) {
        self.files
            .insert(s("checkpoint.manifest"), canon_json(checkpoints));
        self.files.insert(s("metrics.json"), canon_json(metrics));
        self.files.insert(s("hardware.json"), canon_json(hardware));
    }

    pub fn file(&self, name: &str) -> &str {
        self.files.get(name).map_or(UNSET, String::as_str)
    }

    fn digest(&self, name: &str) -> [u8; 32] {
        *blake3::hash(self.file(name).as_bytes()).as_bytes()
    }

    /// Every file's digest, for `training.lock`.
    pub fn digests(&self) -> BTreeMap<String, String> {
        FILES
            .iter()
            .map(|n| (s(*n), hex(&self.digest(n))))
            .collect()
    }

    /// Spec 19.3's twelve slots, each the blake3 of the file that holds it.
    ///
    /// `base_model` carries the declared provenance strings and the digest of
    /// `base_model.lock` itself; the backbone weights' own hash is inside that file, unset
    /// until packet T5.
    pub fn identity(&self) -> TrainingIdentity {
        TrainingIdentity {
            config: self.digest("config.json"),
            optimizer: self.digest("optimizer.json"),
            scheduler: self.digest("scheduler.json"),
            seed: self.digest("seed.json"),
            dataset: self.dataset,
            base_model: BaseModel {
                source: self.base_source.clone(),
                hash: self.digest("base_model.lock"),
                license: self.base_license.clone(),
            },
            augmentation: self.digest("augmentation.json"),
            precision: self.digest("precision.json"),
            topology: self.digest("topology.json"),
            checkpoint_manifest: self.digest("checkpoint.manifest"),
            metrics: self.digest("metrics.json"),
            hardware: self.digest("hardware.json"),
        }
    }

    pub fn hash(&self) -> Result<[u8; 32], DataError> {
        self.identity().training_hash()
    }

    /// Writes the twelve files under `<dir>`.
    pub fn write(&self, dir: &Path) -> Result<(), DataError> {
        for name in FILES {
            crate::write_file(&dir.join(name), self.file(name).as_bytes())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IR: &str = r#"
kind = "training"
[dataset]
root = "runs/collect-001/ds"
frames = "runs/collect-001/frames"
[policy]
bundle = "untrained.esb"
[run]
steps = 20000
batch = 8
lr = 1e-4
seed = 0
checkpoint_at = [1000, 5000]
device = "cuda"
interpreter = "python"
"#;

    const EXTERNAL: &str = r#"
kind = "training"
[dataset]
root = "runs/collect-001/ds"
[policy]
task = "task.toml"
observation = "observation.toml"
deployment = "deployment.toml"
lerobot = { type = "act", chunk_size = 16, n_action_steps = 16 }
[run]
steps = 2000
batch = 8
lr = 1e-4
seed = 0
checkpoint_at = [1000]
device = "cuda"
"#;

    fn facts() -> DatasetFacts {
        DatasetFacts {
            hashes: DatasetHash {
                content: [1; 32],
                schema: [2; 32],
                split: [3; 32],
            },
            episodes: 2,
            frames: 6,
            recorded_task: Some("es:task:abc".to_owned()),
            split_source: "all-train",
        }
    }

    fn plan_of(text: &str, out: &str) -> (Recipe, Plan) {
        let recipe = Recipe::parse(text).expect("recipe parses");
        let trainer = vec![s("python"), s("-m"), s("lerobot.scripts.lerobot_train")];
        let plan = Plan::build(&recipe, Path::new(out), "python", &trainer, Some(13))
            .expect("plan builds");
        (recipe, plan)
    }

    /// The marks always end at `run.steps`, because `train_act.py` caps the run at the
    /// largest of them.
    #[test]
    fn steps_is_always_the_last_mark() {
        let recipe = Recipe::parse(IR).expect("parses");
        assert_eq!(recipe.marks().unwrap(), vec![1000, 5000, 20000]);
    }

    #[test]
    fn a_recipe_names_one_route() {
        let both = IR.replace(
            "bundle = \"untrained.esb\"",
            "bundle = \"a.esb\"\nlerobot = { type = \"act\", chunk_size = 1, \
             n_action_steps = 1 }",
        );
        let e = Recipe::parse(&both).expect_err("both is refused");
        assert!(e.to_string().contains("both"), "{e}");
        let neither = IR.replace("bundle = \"untrained.esb\"", "");
        let e = Recipe::parse(&neither).expect_err("neither is refused");
        assert!(e.to_string().contains("neither"), "{e}");
    }

    #[test]
    fn the_external_route_needs_the_three_documents() {
        let no_task = EXTERNAL.replace("task = \"task.toml\"\n", "");
        let e = Recipe::parse(&no_task).expect_err("refused");
        assert!(e.to_string().contains("`task` is required"), "{e}");
    }

    /// `lerobot-train` has one `--save_freq`, so the marks have to be its multiples.
    #[test]
    fn a_mark_lerobot_cannot_save_at_is_refused() {
        let recipe = Recipe::parse(&EXTERNAL.replace("[1000]", "[700]")).expect("parses");
        let e = recipe.save_freq().expect_err("refused");
        assert!(e.to_string().contains("save_freq"), "{e}");
        assert_eq!(
            Recipe::parse(EXTERNAL).unwrap().save_freq().unwrap(),
            1000,
            "1000 and 2000 are both multiples of 1000"
        );
    }

    /// The rendered plan holds no absolute path, on either separator.
    #[test]
    fn the_render_is_relative_to_out() {
        let (_, plan) = plan_of(IR, "/tmp/scratch-a");
        let a = plan.render(Path::new("/tmp/scratch-a"));
        let (_, plan_b) = plan_of(IR, "/var/other-b");
        let b = plan_b.render(Path::new("/var/other-b"));
        assert_eq!(a, b, "the plan is a property of the recipe, not of --out");
        assert!(!a.contains("scratch-a"), "{a}");
        assert!(a.contains("es policy pack --policy untrained.esb"), "{a}");
        assert!(a.contains("--out checkpoints/20000.esb"), "{a}");
    }

    /// Two output directories, one `identity_hash` — the property oracle 2 checks end to end.
    #[test]
    fn the_identity_is_a_function_of_the_recipe_and_not_of_out() {
        let (recipe, plan_a) = plan_of(IR, "/tmp/a");
        let (_, plan_b) = plan_of(IR, "/tmp/b");
        let a =
            Training::pre_run(&recipe, &plan_a, Path::new("/tmp/a"), "python", &facts()).unwrap();
        let b =
            Training::pre_run(&recipe, &plan_b, Path::new("/tmp/b"), "python", &facts()).unwrap();
        assert_eq!(a.hash().unwrap(), b.hash().unwrap());

        // ... and it moves with the two knobs the packet names.
        for edit in ["seed = 1", "lr = 2e-4"] {
            let (field, _) = edit.split_once(' ').unwrap();
            let text = IR
                .lines()
                .map(|l| if l.starts_with(field) { edit } else { l })
                .collect::<Vec<_>>()
                .join("\n");
            let (r2, p2) = plan_of(&text, "/tmp/a");
            let c = Training::pre_run(&r2, &p2, Path::new("/tmp/a"), "python", &facts()).unwrap();
            assert_ne!(a.hash().unwrap(), c.hash().unwrap(), "{edit}");
        }
    }

    /// No slot is a zero digest and every file parses (packet M7/T1 oracle 2, headless half).
    #[test]
    fn every_slot_is_a_real_digest_of_a_real_file() {
        let (recipe, plan) = plan_of(EXTERNAL, "/tmp/x");
        let t = Training::pre_run(&recipe, &plan, Path::new("/tmp/x"), "python", &facts()).unwrap();
        for name in FILES {
            serde_json::from_str::<Value>(t.file(name))
                .unwrap_or_else(|e| panic!("{name} is not canonical JSON: {e}"));
        }
        let id = t.identity();
        for (slot, d) in [
            ("config", id.config),
            ("optimizer", id.optimizer),
            ("scheduler", id.scheduler),
            ("seed", id.seed),
            ("base_model", id.base_model.hash),
            ("augmentation", id.augmentation),
            ("precision", id.precision),
            ("topology", id.topology),
            ("checkpoint_manifest", id.checkpoint_manifest),
            ("metrics", id.metrics),
            ("hardware", id.hardware),
        ] {
            assert_ne!(d, [0u8; 32], "{slot} is an all-zero digest");
        }
        // The unset slots are the digest of one known file, not of nothing.
        let unset = *blake3::hash(UNSET.as_bytes()).as_bytes();
        assert_eq!(id.metrics, unset);
    }

    /// Filling the post-run slots moves `training_hash` away from `identity_hash`.
    #[test]
    fn finishing_moves_the_hash() {
        let (recipe, plan) = plan_of(IR, "/tmp/x");
        let mut t =
            Training::pre_run(&recipe, &plan, Path::new("/tmp/x"), "python", &facts()).unwrap();
        let before = t.hash().unwrap();
        t.finish(
            &json!({"checkpoints": []}),
            &json!({"loss": [1.0]}),
            &json!({"device": "cpu"}),
        );
        assert_ne!(before, t.hash().unwrap());
    }

    /// Packet M7/T4. The schedule is absent until a recipe asks for it: no flag on the
    /// trainer's line, the `scheduler.json` of before, and therefore the same
    /// `identity_hash` — the property that keeps the measured runs reproducible.
    #[test]
    fn a_recipe_without_a_schedule_is_the_run_of_before() {
        let (recipe, plan) = plan_of(IR, "/tmp/a");
        let rendered = plan.render(Path::new("/tmp/a"));
        for flag in [
            "--schedule",
            "--warmup-steps",
            "--lr-min",
            "--weight-decay",
            "--grad-clip",
        ] {
            assert!(
                !rendered.contains(flag),
                "{flag} is on a plan that asked for none"
            );
        }
        let before =
            Training::pre_run(&recipe, &plan, Path::new("/tmp/a"), "python", &facts()).unwrap();
        assert_eq!(
            before.file("scheduler.json"),
            "{\"kind\":\"constant\",\"lr\":0.0001}\n"
        );
        assert!(before
            .file("optimizer.json")
            .contains("\"weight_decay\":0.01"));
        assert!(!before.file("optimizer.json").contains("grad_clip"));

        let asked = IR.replace(
            "device = \"cuda\"",
            "schedule = { kind = \"warmup_cosine\", warmup = 250, lr_min = 1e-6 }\n\
             weight_decay = 0.05\ngrad_clip = 1.0\ndevice = \"cuda\"",
        );
        let (r2, p2) = plan_of(&asked, "/tmp/a");
        let rendered = p2.render(Path::new("/tmp/a"));
        assert!(
            rendered.contains(
                "--schedule warmup_cosine --warmup-steps 250 --lr-min 0.000001 \
                 --weight-decay 0.05 --grad-clip 1"
            ),
            "{rendered}"
        );
        let after = Training::pre_run(&r2, &p2, Path::new("/tmp/a"), "python", &facts()).unwrap();
        assert_eq!(
            after.file("scheduler.json"),
            "{\"kind\":\"warmup_cosine\",\"lr\":0.0001,\"lr_min\":1e-6,\"total_steps\":20000,\
             \"warmup\":250}\n"
        );
        assert!(after.file("optimizer.json").contains("\"grad_clip\":1.0"));
        assert_ne!(before.hash().unwrap(), after.hash().unwrap());
    }

    /// Each of the three is refused by the name of the field that is wrong.
    #[test]
    fn a_schedule_that_cannot_run_is_refused_by_name() {
        let with =
            |body: &str| IR.replace("device = \"cuda\"", &format!("{body}\ndevice = \"cuda\""));
        for (body, word) in [
            ("schedule = { kind = \"cosine\" }", "schedule.kind"),
            (
                "schedule = { kind = \"warmup_cosine\", warmup = 20000 }",
                "schedule.warmup",
            ),
            (
                "schedule = { kind = \"warmup_cosine\", warmup = 10, lr_min = 1.0 }",
                "schedule.lr_min",
            ),
            (
                "schedule = { kind = \"constant\", warmup = 10 }",
                "constant",
            ),
            ("grad_clip = -1.0", "grad_clip"),
        ] {
            let e = Recipe::parse(&with(body)).expect_err("refused");
            assert!(e.to_string().contains(word), "{body}: {e}");
        }
        // The external route has its own optimizer and scheduler: declaring one here would
        // put a schedule into `scheduler.json` that the run never applied.
        let e = Recipe::parse(&EXTERNAL.replace(
            "device = \"cuda\"",
            "schedule = { kind = \"warmup_cosine\", warmup = 10 }\ndevice = \"cuda\"",
        ))
        .expect_err("refused");
        assert!(e.to_string().contains("IR route's"), "{e}");
    }

    #[test]
    fn a_camera_feature_names_its_directory() {
        assert_eq!(
            camera_suffix("observation.images.rgb_overhead"),
            "rgb_overhead"
        );
        assert_eq!(camera_suffix("rgb"), "rgb");
    }
}
