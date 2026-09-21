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

use es_compile::plan::{augmentation_chains, AugmentStep};
use es_ir::learning::{LearningGraph, LearningNode};
use es_ir::observation::{AugmentKind, ObservationIr, ObservationNode};
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

/// The one pretrained backbone this repository has approved (spec 29 licence row, owner
/// decision 2026-09-15: torchvision's `ImageNet` `ResNet18` weights, BSD-3).
pub const BASE_MODEL_SOURCE: &str = "torchvision.models.ResNet18_Weights.IMAGENET1K_V1";

/// **The pin** (packet M7/T5): the blake3 of the `resnet18-imagenet1k-v1.safetensors`
/// `python/es/fetch_backbone.py` writes, and the only `base_model` `es train` accepts.
///
/// The 45 MB file is not committed — it lives at `~/artifacts/plan-v/m7-t5/` on the oracle
/// server — so this string is what the repository knows about it, the way
/// `tests/fixtures/mjcf/*.PROVENANCE.json` pins the upstream MJCF it is derived from. It is
/// also the single copy: `crates/es-policy/tests/backbone_provenance.rs` reads it out of this
/// file rather than keeping a second one, and `fetch_backbone.py` is *given* it with
/// `--expect` rather than holding its own.
///
/// Measured identical under torchvision 0.26.0+cu129 and 0.29.0+cpu, which is what makes it a
/// property of the weights and not of the interpreter that fetched them.
pub const RESNET18_IMAGENET1K_V1_BLAKE3: &str =
    "8511928e7ca6e3b07355e8b66284294ba692093fd0dfe246cdf96a6c9e801899";

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
    /// Required by the two routes that read demonstrations, optional beside `[rl]`: PPO's
    /// data is the rollout it generates, so there is nothing on disk to name (packet M8/S4b).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<DatasetRef>,
    pub policy: PolicyRef,
    pub run: Run,
    /// `[init] policy = "<bundle.esb>"` — the policy this run starts from (packet M8/S1).
    /// Absent is absent: no `training/init.lock`, no `--init-weights`, and the same
    /// `identity_hash` every recipe written before the packet had.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init: Option<InitRef>,
    /// `[rl]` — the reinforcement-learning route (packet M8/S4b). Absent is absent: every
    /// recipe written before this packet parses, plans and hashes exactly as it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rl: Option<Rl>,
}

/// `[init]` — one field, because one is the whole question (packet M8/S1).
///
/// The bundle named here is opened, its safetensors compared to the lowered module's
/// contract, and the tensors whose name *and* shape match are what the trainer starts from.
/// Everything else about the run — the dataset, the architecture, the optimizer — is still
/// `[dataset]`, `[policy]` and `[run]`: this table says where the numbers come from, not what
/// they are.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitRef {
    /// A `.esb` bundle, as `es policy pack` or `es train` writes one.
    pub policy: String,
}

/// `[rl]` — PPO over rollouts in our own `Env` (packet M8/S4b, spec 13.4).
///
/// **What is not here is the point** (`docs/design/rl-continuation.md` rule 1). PPO is a
/// trainer, not an IR: the value head, the state-independent `log_std`, GAE and the entropy
/// coefficient live in `python/es/train_ppo.py` and in spec 19.3's `training/`, and they move
/// `training_hash` and never `learning_hash`. The deployed graph is whatever `[policy] bundle`
/// already says it is, and this table does not touch it.
///
/// `[run] steps` is the iteration count beside this table, `[run] batch` is refused by name
/// (PPO's batch is `envs * horizon`, derived, not declared), and `[dataset]` is optional
/// because a rollout generates its own data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rl {
    /// `"ppo"`, and nothing else yet. Named rather than assumed, so a second algorithm is a
    /// value here and not a new table.
    pub algo: String,
    /// Environments stepped in lockstep — the simulation batch (spec 5.2).
    pub envs: u32,
    /// Control steps per env per iteration. One iteration collects `envs * horizon` rows.
    pub horizon: u32,
    /// Passes over each iteration's rows.
    pub epochs: u32,
    /// Minibatches per epoch; it has to divide `envs * horizon`.
    pub minibatches: u32,
    /// Discount.
    pub gamma: f64,
    /// GAE's `lambda`.
    pub lam: f64,
    /// The clipped objective's `epsilon`.
    pub clip: f64,
    /// Entropy bonus coefficient.
    pub entropy: f64,
    /// Value-loss coefficient.
    pub value_coef: f64,
    /// The Gaussian's initial `log_std`, which is training-only state and never enters a
    /// document or a bundle. Absent is the importer's own when `[init]` carries one, and
    /// `-0.5` otherwise — the trainer decides, because it is the side that can see both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_log_std: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    /// The `resnet18-imagenet1k-v1.safetensors` a `VisionEncoder { pretrained: true }` is
    /// initialised from (packet M7/T5), with its `.lock.json` beside it. Required by such a
    /// bundle and refused by any other, so the recipe and the IR cannot disagree about
    /// whether this run started from `ImageNet`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
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
    /// Optimizer steps on the two demonstration routes; **PPO iterations** beside `[rl]`.
    pub steps: u32,
    /// Samples per optimizer step. Required by the two demonstration routes and refused by
    /// name beside `[rl]`, where the batch is `envs * horizon` and is therefore derived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<u32>,
    pub lr: f64,
    pub seed: u64,
    /// The seed of the augmentation RNG (packet M7/T6), `[run] seed` when it is absent.
    /// Separate because it is separable: re-drawing the augmentation of an otherwise
    /// identical run is a different run, and spec 19.3 gives it its own `seed.json` slot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub augmentation_seed: Option<u64>,
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
    /// Passed to whichever trainer the route runs, verbatim, after everything this module
    /// derives (and after `[policy.lerobot] extra` on the external route). `--resident-gpu`
    /// is the one packet M7/T3 measured; a flag that changes the bits is a deliberate,
    /// recorded act, which is why it lives in the recipe and enters `identity_hash`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra: Vec<String>,
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
    /// `lower -> train_ppo.py -> pack`: the same Learning IR, optimized against a reward
    /// instead of demonstrations (packet M8/S4b). No bake, because there is no dataset to
    /// bake: the rollout is the data.
    Rl,
}

impl Route {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ir => "ir",
            Self::External => "external",
            Self::Rl => "rl",
        }
    }

    /// Does this route run a trainer whose stdout is one JSON line (and, when someone is
    /// watching, `{"progress": ...}` lines before it)? Both of ours do; `lerobot-train`
    /// streams a log and inherits instead.
    pub fn captures_trainer_stdout(self) -> bool {
        matches!(self, Self::Ir | Self::Rl)
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
        recipe.rl_args()?;
        match (route, recipe.run.batch) {
            // Refused **by name**, not silently ignored: PPO's batch is `envs * horizon` and
            // is therefore derived. A recipe that declared one would put a number into
            // `precision.json` that nothing in the run ever used (spec 28.10 rule 2).
            (Route::Rl, Some(batch)) => {
                return Err(refuse(format!(
                    "[run] `batch` is {batch} and this recipe has an `[rl]` table; a PPO \
                     iteration's batch is `envs * horizon` ({} here), which is derived from \
                     [rl] and not declared. Delete `[run] batch`",
                    recipe
                        .rl
                        .as_ref()
                        .map_or(0, |rl| u64::from(rl.envs) * u64::from(rl.horizon)),
                )))
            }
            (Route::Ir | Route::External, None) => {
                return Err(refuse(
                    "[run] `batch` is required: it is how many samples one optimizer step \
                     covers. Only an `[rl]` recipe leaves it out, because a rollout's batch \
                     is `envs * horizon`",
                ))
            }
            _ => {}
        }
        if route != Route::Rl && recipe.dataset.is_none() {
            return Err(refuse(
                "[dataset] `root` is required: this route trains on recorded demonstrations \
                 and has to be told where they are. Only an `[rl]` recipe leaves it out, \
                 because its data is the rollout it generates",
            ));
        }
        if recipe.rl.is_some() && recipe.policy.lerobot.is_some() {
            return Err(refuse(
                "[rl] is the IR route's: it optimizes the graph `[policy] bundle` names \
                 against a reward, and `lerobot-train` is an imitation trainer over a \
                 dataset. A recipe describes one route",
            ));
        }
        if route == Route::External && recipe.policy.base_model.is_some() {
            return Err(refuse(
                "[policy] `base_model` is the IR route's: it initialises a \
                 `VisionEncoder { pretrained = true }` in this project's Learning IR, and \
                 `lerobot`'s ACT builds its own backbone from its own \
                 `pretrained_backbone_weights`. Reach lerobot's through \
                 `[policy] lerobot.extra`",
            ));
        }
        if route == Route::External && recipe.init.is_some() {
            return Err(refuse(
                "[init] `policy` is the IR route's: this side copies the tensors of a bundle \
                 into the module `es policy lower` emitted, and `lerobot-train` starts from \
                 its own `--policy.path`. Mixing the two would write an `init.lock` \
                 describing tensors the run never loaded",
            ));
        }
        if route == Route::External && recipe.run.steps == 0 {
            return Err(refuse(
                "[run] `steps` is 0 on the lerobot route; `lerobot-train` saves at a \
                 `--save_freq` and has no step 0 to save at. \"Checkpoint immediately\" is \
                 the IR route's, where the trainer writes the module's initial state",
            ));
        }
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

    /// The `[rl]` table validated, as `train_ppo.py`'s flags (packet M8/S4b).
    ///
    /// Empty when there is no `[rl]`, which is what keeps every pre-S4b plan and its golden
    /// byte-identical. Every bound here is a refusal rather than a clamp: a coefficient
    /// silently moved is a run nobody can reproduce from its own recipe.
    pub fn rl_args(&self) -> Result<Vec<String>, DataError> {
        let Some(rl) = &self.rl else {
            return Ok(Vec::new());
        };
        if rl.algo != "ppo" {
            return Err(refuse(format!(
                "[rl] `algo` is {:?}; this packet implements \"ppo\"",
                rl.algo
            )));
        }
        for (field, value) in [("envs", rl.envs), ("horizon", rl.horizon)] {
            if value == 0 {
                return Err(refuse(format!("[rl] `{field}` is 0")));
            }
        }
        for (field, value) in [("epochs", rl.epochs), ("minibatches", rl.minibatches)] {
            if value == 0 {
                return Err(refuse(format!("[rl] `{field}` is 0")));
            }
        }
        let rows = u64::from(rl.envs) * u64::from(rl.horizon);
        if rows % u64::from(rl.minibatches) != 0 {
            return Err(refuse(format!(
                "[rl] `minibatches` is {} and an iteration collects {rows} rows \
                 (`envs` {} * `horizon` {}); it has to divide them, because a trailing short \
                 minibatch would weight the last rows of every epoch differently",
                rl.minibatches, rl.envs, rl.horizon
            )));
        }
        for (field, value) in [("gamma", rl.gamma), ("lam", rl.lam)] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(refuse(format!(
                    "[rl] `{field}` is {value}; it is in [0, 1]"
                )));
            }
        }
        if !rl.clip.is_finite() || rl.clip <= 0.0 {
            return Err(refuse(format!(
                "[rl] `clip` is {}; the clipped objective's epsilon is positive",
                rl.clip
            )));
        }
        for (field, value) in [("entropy", rl.entropy), ("value_coef", rl.value_coef)] {
            if !value.is_finite() || value < 0.0 {
                return Err(refuse(format!("[rl] `{field}` is {value}")));
            }
        }
        if let Some(log_std) = rl.init_log_std {
            if !log_std.is_finite() {
                return Err(refuse(format!("[rl] `init_log_std` is {log_std}")));
            }
        }
        let mut args = vec![
            s("--envs"),
            rl.envs.to_string(),
            s("--horizon"),
            rl.horizon.to_string(),
            s("--epochs"),
            rl.epochs.to_string(),
            s("--minibatches"),
            rl.minibatches.to_string(),
            s("--gamma"),
            rl.gamma.to_string(),
            s("--lam"),
            rl.lam.to_string(),
            s("--clip"),
            rl.clip.to_string(),
            s("--entropy"),
            rl.entropy.to_string(),
            s("--value-coef"),
            rl.value_coef.to_string(),
        ];
        // Only when the recipe names one: absent means the trainer decides between the
        // importer's own `log_std` and the constant, and a flag carrying a default would
        // take that decision away from the side that can see both.
        if let Some(log_std) = rl.init_log_std {
            args.push(s("--init-log-std"));
            args.push(log_std.to_string());
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
            (Some(_), None) if self.rl.is_some() => Ok(Route::Rl),
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
    ///
    /// `steps = 0` is the one run with a mark of 0: **checkpoint immediately**, the module's
    /// initial state written without an optimizer step (packet M8/S1). With `[init]` that is
    /// a re-pack of the policy it started from; without one it is the untrained module, which
    /// is a real thing to want and a refusal nobody could defend.
    pub fn marks(&self) -> Result<Vec<u32>, DataError> {
        if self.run.batch == Some(0) {
            return Err(refuse("[run] `batch` is 0"));
        }
        if !self.run.lr.is_finite() || self.run.lr <= 0.0 {
            return Err(refuse(format!("[run] `lr` is {}", self.run.lr)));
        }
        if self.run.steps == 0 {
            if !self.run.checkpoint_at.is_empty() {
                return Err(refuse(format!(
                    "[run] `steps` is 0 and `checkpoint_at` is {:?}; a run that takes no \
                     optimizer step has no step to also checkpoint at. Its one mark is 0",
                    self.run.checkpoint_at
                )));
            }
            return Ok(vec![0]);
        }
        let mut marks: Vec<u32> = self.run.checkpoint_at.clone();
        marks.push(self.run.steps);
        marks.sort_unstable();
        marks.dedup();
        // Mark 0 is "before the first update". On the two demonstration routes that state
        // only exists for a `steps = 0` run, which is handled above; an `[rl]` run passes
        // through it on the way to iteration 1, and it is the state a continuation run is
        // judged against -- `checkpoints/0.esb` has to be `[init] policy` bit for bit
        // (packet M8/S4b oracle 3). So it is a legal mark here and nowhere else.
        let floor = u32::from(self.rl.is_none());
        if let Some(bad) = marks.iter().find(|m| **m < floor || **m > self.run.steps) {
            return Err(refuse(format!(
                "[run] `checkpoint_at` holds {bad}, which is not an {} of a {}-step run",
                if self.rl.is_some() {
                    "iteration, or 0 for the state before the first one,"
                } else {
                    "optimizer step"
                },
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

/// One `training_only` chain as JSON — the list `training/augmentation.json` carries and the
/// list `es dataset bake --for-training` writes into its manifest, written once here so the
/// trainer reads one shape from either file (packet M7/T6).
///
/// `node` is the Observation IR node id, and it is not decoration: it is the `node_index`
/// coordinate of the augmentation RNG's key, so two ports' chains draw different streams at
/// the same sample and step (`python/es/augment.py`, design note section 13).
pub fn augmentation_json(chain: &[AugmentStep]) -> Value {
    Value::Array(
        chain
            .iter()
            .map(|step| {
                let mut entry = match step.kind {
                    AugmentKind::RandomCrop { width, height } => {
                        json!({"kind": "RandomCrop", "width": width, "height": height})
                    }
                    AugmentKind::ColorJitter {
                        brightness,
                        contrast,
                        saturation,
                        hue,
                    } => json!({
                        "kind": "ColorJitter", "brightness": brightness, "contrast": contrast,
                        "saturation": saturation, "hue": hue,
                    }),
                    AugmentKind::RandomErasing { probability } => {
                        json!({"kind": "RandomErasing", "probability": probability})
                    }
                    AugmentKind::GaussianNoise { sigma } => {
                        json!({"kind": "GaussianNoise", "sigma": sigma})
                    }
                };
                entry["node"] = json!(step.node.0);
                entry
            })
            .collect(),
    )
}

/// Does the Learning IR declare a pretrained backbone? Then the recipe owes
/// `[policy] base_model` (packet M7/T5).
pub fn has_pretrained_backbone(learning: &LearningGraph) -> bool {
    learning
        .nodes
        .nodes
        .values()
        .any(|n| matches!(n, LearningNode::VisionEncoder { pretrained, .. } if *pretrained))
}

/// `<weights>.lock.json` as `python/es/fetch_backbone.py` writes it, reduced to the fields
/// that identify **the weights** (spec 19.3's `base_model.lock`).
///
/// The lock file beside the artifact also records the `torch` and `torchvision` that fetched
/// it; those are deliberately not here. Two machines fetching the same upstream file write
/// byte-identical tensors and different version strings, and carrying the strings into
/// `base_model.lock` would put the fetching machine into `identity_hash` — so one recipe
/// would have two identities depending on where its backbone was produced.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Backbone {
    /// `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`.
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub url: String,
    /// torchvision's own hash of the `.pth` it downloaded.
    #[serde(default)]
    pub sha256_upstream: String,
    /// blake3 of the safetensors file — the pin.
    #[serde(default)]
    pub blake3: String,
    /// The key the artifact does not carry, named rather than left to be noticed.
    #[serde(default)]
    pub dropped: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub license_url: String,
}

impl Backbone {
    /// Reads `weights` and the `.lock.json` beside it, and refuses by name unless the three
    /// claims agree: the file's blake3, the lock file's, and [`RESNET18_IMAGENET1K_V1_BLAKE3`].
    ///
    /// A licence field that is empty is a refusal of its own: spec 19.3 makes this file the
    /// basis for licence tracking, and a provenance record with nothing in that slot tracks
    /// nothing.
    pub fn verify(weights: &Path) -> Result<Self, DataError> {
        let bytes = std::fs::read(weights)
            .map_err(|e| refuse(format!("[policy] `base_model` {}: {e}", weights.display())))?;
        let digest = hex(blake3::hash(&bytes).as_bytes());
        let lock_path = weights.with_extension("lock.json");
        let text = std::fs::read_to_string(&lock_path).map_err(|e| {
            refuse(format!(
                "{}: {e}\nEvery `base_model` travels with the lock file \
                 `python/es/fetch_backbone.py` writes beside it; it is where the licence and \
                 the upstream URL come from (spec 19.3)",
                lock_path.display()
            ))
        })?;
        let lock: Self = serde_json::from_str(&text)
            .map_err(|e| refuse(format!("{}: {e}", lock_path.display())))?;

        if lock.blake3 != digest {
            return Err(refuse(format!(
                "{} does not hash to what {} claims:\n  file  {digest}\n  lock  {}",
                weights.display(),
                lock_path.display(),
                lock.blake3
            )));
        }
        if lock.source != BASE_MODEL_SOURCE {
            return Err(refuse(format!(
                "{} names the source {:?}; the one this repository has approved is {:?} \
                 (spec 29 licence row)",
                lock_path.display(),
                lock.source,
                BASE_MODEL_SOURCE
            )));
        }
        if lock.license.trim().is_empty() {
            return Err(refuse(format!(
                "{}: `license` is empty. Spec 19.3 makes base_model.lock the basis for \
                 licence tracking, and a run whose backbone carries no licence cannot be \
                 shipped anywhere",
                lock_path.display()
            )));
        }
        // Last, because the three above ask whether the lock file is a well-formed provenance
        // record for these bytes and this one asks whether these bytes are the artifact the
        // repository has actually measured. Both matter; they are different questions.
        if digest != RESNET18_IMAGENET1K_V1_BLAKE3 {
            return Err(refuse(format!(
                "[policy] `base_model` {} is not the pinned backbone:\n  file    \
                 {digest}\n  pinned  {RESNET18_IMAGENET1K_V1_BLAKE3}\nThe pin is \
                 `RESNET18_IMAGENET1K_V1_BLAKE3` in crates/es-data/src/training.rs. Re-fetch \
                 with `python/es/fetch_backbone.py --arch resnet18 --out <dir>`, or move the \
                 pin deliberately (`--repin`) together with the design note",
                weights.display()
            )));
        }
        Ok(lock)
    }
}

// --- starting from a policy (packet M8/S1) -----------------------------------------------

/// `training/init.lock`, the thirteenth file of `<out>/training/` — written only by a recipe
/// that names `[init] policy`, and its digest is what enters `config.json` and therefore
/// `identity_hash` / `training_hash`.
pub const INIT_LOCK: &str = "init.lock";

/// What [`init_from`] decided: the lock to record and the tensors to hand the trainer.
#[derive(Clone, Debug)]
pub struct Init {
    /// `training/init.lock`'s body, sorted names.
    pub lock: Value,
    /// The copied tensors, ready for `es_policy::weights::write_safetensors`.
    pub weights: es_policy::weights::Checkpoint,
    /// `copied.len()`, so the shell can print it without re-reading the lock.
    pub copied: usize,
    pub initialised: usize,
}

/// Compare `[init] policy`'s bundle to the module `target` lowers to, and copy what fits.
///
/// Three buckets and no fourth (the packet's schema): **copied** is a tensor whose name and
/// shape the lowered module declares, **initialised** is a name the module declares and the
/// bundle does not carry, and **`shape_mismatch`** is a name both know at two shapes — recorded
/// and *not* copied, because reshaping a trained tensor is guessing and the failure mode of a
/// guess here is a policy that still trains and still looks fine.
///
/// A `nodes.<k>.*` prefix claim covers a sub-module whose parameter names belong to
/// torchvision or `torch.nn` (`es_policy::weights::validate_keys`). Its tensors are copied by
/// name alone, because there is no declared shape on this side to check them against;
/// `train_act.py --init-weights` checks each one against the real module before loading it,
/// which is where the shape actually lives.
///
/// Zero copied is a refusal. A warm start that shares nothing with what it starts from is not
/// a warm start, and letting it through would write a provenance record whose whole content
/// is "none of this was used".
pub fn init_from(source: &str, bundle: &[u8], target: &LearningGraph) -> Result<Init, DataError> {
    let opened = es_compile::PolicyBundle::open(bundle)
        .map_err(|e| refuse(format!("[init] `policy` {source}: {e}")))?;
    let header = es_policy::weights::parse_header(&opened.weights)
        .map_err(|e| refuse(format!("[init] `policy` {source}: {e}")))?;
    let module = es_policy::lower_to_torch(target)
        .map_err(|e| refuse(format!("the bundle being trained does not lower: {e}")))?;

    let claims: Vec<&str> = module
        .weight_keys
        .iter()
        .filter_map(|k| k.strip_suffix('*'))
        .collect();
    // Safetensors is eight bytes of header length, the header, then the data segment; an
    // entry's offsets are relative to the end of the header. `parse_header` succeeded, so
    // these eight bytes are there and the segment behind them is long enough.
    let data_at = 8 + u64::from_le_bytes(
        opened.weights[..8]
            .try_into()
            .expect("parse_header already read these eight bytes"),
    ) as usize;

    let (mut copied, mut mismatch) = (Vec::new(), Vec::new());
    let mut weights = es_policy::weights::Checkpoint::new();
    for (name, entry) in &header {
        let declared = module.weight_shapes.get(name);
        if declared.is_none() && !claims.iter().any(|p| name.starts_with(p)) {
            // A tensor the module does not declare at all: not copied, and not one of the
            // three buckets either -- the module has no slot to put it in.
            continue;
        }
        if let Some(want) = declared {
            if entry.shape != *want {
                mismatch.push(json!({
                    "name": name, "expected": want, "found": entry.shape,
                }));
                continue;
            }
        }
        if entry.dtype != "F32" {
            return Err(refuse(format!(
                "[init] `policy` {source}: {name} is {}, and every tensor on this path is \
                 F32 (spec 8.4). A second dtype here would be a second reader of one format",
                entry.dtype
            )));
        }
        let (a, b) = entry.offsets;
        let bytes = &opened.weights[data_at + a as usize..data_at + b as usize];
        let values = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        weights.insert(name.clone(), (entry.shape.clone(), values));
        copied.push(name.clone());
    }

    let initialised: Vec<String> = module
        .weight_keys
        .iter()
        .filter(|key| match key.strip_suffix('*') {
            Some(prefix) => !header.keys().any(|k| k.starts_with(prefix)),
            None => !header.contains_key(*key),
        })
        .cloned()
        .collect();

    if copied.is_empty() {
        return Err(refuse(format!(
            "[init] `policy` {source} shares no tensor with the module this recipe trains: \
             {} name(s) the module declares are absent from the bundle and {} more disagree \
             about a shape. Starting from a policy that shares nothing is a mistake, not a \
             warm start -- the run would be `steps` steps from scratch under a document \
             saying otherwise",
            initialised.len(),
            mismatch.len()
        )));
    }

    // `copied` and `initialised` come out of a BTreeMap and a key list that is already
    // sorted, but say it rather than rely on it: the lock's digest is a hash slot.
    copied.sort();
    let mut initialised = initialised;
    initialised.sort();
    Ok(Init {
        copied: copied.len(),
        initialised: initialised.len(),
        lock: json!({
            "schema_version": 1,
            "source": source,
            "policy_hash": opened.manifest.hashes.policy.as_ref().map(hex),
            "learning_hash": opened.manifest.hashes.learning.as_ref().map(hex),
            "copied": copied,
            "initialised": initialised,
            "shape_mismatch": mismatch,
        }),
        weights,
    })
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

/// The trainer of the RL route, relative to the repository root (packet M8/S4b).
pub const TRAIN_PPO: &str = "python/es/train_ppo.py";

/// `<out>/docs` — the bundle's Task, Observation and Deployment IR, written back out as the
/// three `.toml` files `es_native.Rollout` is constructed from (packet M8/S4b).
///
/// The trainer is handed a directory rather than three paths because the three documents are
/// one thing: they come out of one bundle, and a run that mixed a task from one with a
/// deployment from another would be stepping an env nothing declared. The scene is not among
/// them — the Task IR's `scene.path` already names it, and a second copy on the command line
/// is a second thing that can disagree with the document.
pub fn rollout_docs(out: &Path) -> String {
    under(out, "docs")
}

/// `<out>/training/value.safetensors` — the value MLP, for resumption (spec 19.3, S4b).
///
/// Under `training/` and never under `checkpoints/`: it is training-only state, like the
/// optimizer's moments, and `es policy pack` would refuse it anyway because the lowered
/// module declares no such tensor (`docs/design/rl-continuation.md` rule 1).
pub fn value_weights(out: &Path) -> String {
    under(out, "training/value.safetensors")
}

/// `<out>/weights/init.safetensors` — the tensors [`init_from`] copied out of `[init] policy`,
/// and the file the trainer is handed as `--init-weights` (packet M8/S1).
///
/// It is written, rather than the bundle handed over directly, because what the trainer loads
/// has to be exactly what `init.lock` says was copied: a file holding the intersection cannot
/// disagree with the list, and a whole checkpoint plus a list can.
pub fn init_weights(out: &Path) -> String {
    under(out, "weights/init.safetensors")
}

impl Plan {
    /// The plan is a function of the recipe, the output directory and the resolved
    /// interpreter — and, on the external route, of the Observation IR's state width.
    ///
    /// `trainer` is the program that runs the external trainer: the `lerobot-train` entry
    /// point beside the interpreter when there is one, otherwise the interpreter and
    /// `-m lerobot.scripts.lerobot_train`. Nothing on disk is read here, so `--dry-run` works
    /// on a machine that has neither the dataset nor the bundle.
    ///
    /// `augmented` is whether the policy's Observation IR carries a `training_only` chain
    /// (packet M7/T6). It is a *fact about the bundle*, not a recipe field: the bake is told
    /// to write the chain's boundary and the trainer is told to apply it, or neither is and
    /// the plan is the plan of before, word for word. `--dry-run` on the IR route does not
    /// open the bundle, so it passes `false` and prints the un-augmented line.
    pub fn build(
        recipe: &Recipe,
        out: &Path,
        interpreter: &str,
        trainer: &[String],
        state_dim: Option<usize>,
        augmented: bool,
    ) -> Result<Self, DataError> {
        let route = recipe.route()?;
        let marks = recipe.marks()?;
        let run = &recipe.run;
        let mut steps = Vec::new();
        let es = |a: &[&str]| -> Vec<String> { a.iter().map(|w| s(*w)).collect() };

        match route {
            Route::Ir => {
                let dataset = recipe.dataset.clone().unwrap_or_default();
                let bundle = recipe.policy.bundle.clone().unwrap_or_default();
                let mut bake = vec![
                    s("--policy"),
                    bundle.clone(),
                    s("--out"),
                    under(out, "baked"),
                ];
                if let Some(frames) = &dataset.frames {
                    bake.push(s("--frames"));
                    bake.push(frames.clone());
                }
                if augmented {
                    bake.push(s("--for-training"));
                }
                bake.push(dataset.root.clone());
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
                        run.batch.unwrap_or_default().to_string(),
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
                    // Likewise for the pretrained backbone (packet M7/T5). `frozen` is *not*
                    // here: it is a field of the IR, so the lowered module carries it and the
                    // trainer reads it off `requires_grad` -- a copy of it on this line would
                    // be a second thing that can disagree with the Learning IR.
                    .chain(
                        recipe
                            .policy
                            .base_model
                            .iter()
                            .flat_map(|path| [s("--init-backbone"), path.clone()]),
                    )
                    // Beside it, and for the same reason (packet M8/S1): the module's initial
                    // state is a checkpoint, not a construction-time download. The file is
                    // the *intersection* `init_from` wrote, so the trainer loads exactly the
                    // names `training/init.lock` lists as copied.
                    .chain(
                        recipe
                            .init
                            .iter()
                            .flat_map(|_| [s("--init-weights"), init_weights(out)]),
                    )
                    // The chain the bake left at the boundary (packet M7/T6). The file is
                    // spec 19.3's own `augmentation.json`, so what the trainer applies and
                    // what `identity_hash` names are one file, not two descriptions of one.
                    .chain(
                        augmented
                            .then(|| {
                                [
                                    s("--augmentation"),
                                    under(out, "training/augmentation.json"),
                                ]
                            })
                            .into_iter()
                            .flatten(),
                    )
                    // Last, because `[run] extra` is by definition what comes after everything
                    // this module derives (packet M7/T2).
                    .chain(run.extra.iter().cloned())
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
            // `lower -> train_ppo.py -> pack` (packet M8/S4b). No bake: the rollout is the
            // data, so there is nothing on disk to run through the Observation IR ahead of
            // time -- `es_native.Rollout` runs the same `CpuPlan` per step instead.
            Route::Rl => {
                let bundle = recipe.policy.bundle.clone().unwrap_or_default();
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
                    prefix: vec![s(interpreter), s(TRAIN_PPO)],
                    args: vec![
                        s("--module"),
                        under(out, "module"),
                        // The three documents the bundle carries, written back out by the
                        // shell: `Rollout` takes them as text, and taking them from the
                        // bundle is what stops a run from stepping an env that the policy
                        // being trained was not declared against.
                        s("--rollout-docs"),
                        rollout_docs(out),
                        s("--out"),
                        under(out, "weights/model.safetensors"),
                        s("--value-out"),
                        value_weights(out),
                        s("--checkpoint-at"),
                        marks_arg,
                        s("--iterations"),
                        run.steps.to_string(),
                        s("--seed"),
                        run.seed.to_string(),
                        s("--lr"),
                        run.lr.to_string(),
                        s("--device"),
                        run.device.clone(),
                        s("--loss-curve"),
                        under(out, "metrics/loss-curve.json"),
                    ]
                    .into_iter()
                    .chain(recipe.rl_args()?)
                    .chain(recipe.schedule_args()?)
                    .chain(
                        recipe
                            .init
                            .iter()
                            .flat_map(|_| [s("--init-weights"), init_weights(out)]),
                    )
                    .chain(run.extra.iter().cloned())
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
                let dataset = recipe.dataset.clone().unwrap_or_default();
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
                    dataset.root.clone(),
                    s("--out"),
                    under(out, "ds-v3"),
                ];
                if dataset.frames.is_some() {
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
                    format!("--batch_size={}", run.batch.unwrap_or_default()),
                    format!("--seed={}", run.seed),
                    format!("--save_freq={}", recipe.save_freq()?),
                    format!("--output_dir={}", under(out, "lerobot")),
                    s("--job_name=es-train"),
                    s("--wandb.enable=false"),
                ];
                train.extend(lerobot.extra.iter().chain(&run.extra).cloned());
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

// --- the cycle (packet M7/T2) -----------------------------------------------------------------

/// `kind = "cycle"`.
pub const CYCLE_KIND: &str = "cycle";

/// `cycle.toml` — collect, train, evaluate and showcase under one document and one ledger
/// (spec 13.1, spec 13.3).
///
/// It names the stages; it does not re-describe them. `[train]` is T1's recipe by path or
/// inline, `[eval]` is an Evaluation IR by path, and the words each stage runs with are the
/// flags those commands already take (design note `docs/design/training-recipe.md`, "the
/// cycle").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cycle {
    pub kind: String,
    pub scene: String,
    /// An existing dataset root to train on, instead of `[collect]`. Exactly one of the two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collect: Option<CollectRef>,
    pub train: TrainRef,
    pub eval: EvalRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub showcase: Option<ShowcaseRef>,
}

/// `[collect]` — what `es loop collect` is told.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectRef {
    /// The bundle whose Deployment IR is the plane (`es loop collect --policy`).
    pub policy: String,
    /// The scripted demonstrator, or absent for a trained policy's own rollouts. Setting it
    /// is what arms the expert gate of spec 28.9 rule 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert: Option<String>,
    pub episodes: u32,
    #[serde(default)]
    pub seed: u64,
    /// Render the Task IR's image channel beside the dataset (`es loop collect --frames`).
    #[serde(default)]
    pub frames: bool,
}

/// `[train]` — T1's recipe by path (`recipe`) or inline (`dataset`/`policy`/`run`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<DatasetRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<PolicyRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<Run>,
}

/// `[eval]` — what `es eval run` is told, for the expert gate and for the trained policy
/// alike. One config, because a gate the expert passed under other conditions gates nothing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalRef {
    pub config: String,
    /// `"last"` or one of the recipe's checkpoint marks.
    #[serde(default = "default_checkpoint")]
    pub checkpoint: String,
    #[serde(default = "one_job")]
    pub jobs: u32,
    /// Render the observation frames the run needs. An Observation IR with an image input is
    /// refused without them, and the expert gate reads its own state through the same source.
    #[serde(default)]
    pub frames: bool,
}

fn default_checkpoint() -> String {
    "last".to_owned()
}

fn one_job() -> u32 {
    1
}

/// `[showcase]` — the human-facing re-render of one evaluated episode (packet M5/V9).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShowcaseRef {
    /// A cell of the evaluation, `<suite>-<NN>`, as its `.estraj` is named.
    pub cell: String,
    pub eye: [f64; 3],
    pub look_at: [f64; 3],
    #[serde(default = "default_fov")]
    pub fov: f64,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
}

fn default_fov() -> f64 {
    45.0
}

fn default_width() -> u32 {
    1280
}

fn default_height() -> u32 {
    720
}

/// One stage of the cycle. The order here is the order they run in and the order `--from`
/// compares against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Collect,
    /// Spec 28.9 rule 1: the same harness, on the expert, before anything trains.
    ExpertGate,
    Train,
    Eval,
    Showcase,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Collect => "collect",
            Self::ExpertGate => "expert-gate",
            Self::Train => "train",
            Self::Eval => "eval",
            Self::Showcase => "showcase",
        }
    }

    /// The four `--from` names. The expert gate is not one of them: it belongs to the collect
    /// stage's data, and resuming *at* it would train on a dataset nothing judged.
    pub fn parse(name: &str) -> Option<Self> {
        [Self::Collect, Self::Train, Self::Eval, Self::Showcase]
            .into_iter()
            .find(|s| s.as_str() == name)
    }
}

/// One stage as the command that runs it: the same words the plan prints and the same words
/// the in-process entry point is handed, so a printed line and an executed stage cannot drift.
#[derive(Clone, Debug, PartialEq)]
pub struct CycleStep {
    pub stage: Stage,
    /// `["es", "loop", "collect"]` and so on — printed, never spawned as a process.
    pub prefix: Vec<String>,
    pub args: Vec<String>,
}

/// The whole cycle as commands, with T1's plan nested under `train`.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclePlan {
    pub steps: Vec<CycleStep>,
    pub train: Plan,
    /// The checkpoint mark `[eval] checkpoint` resolved to.
    pub mark: u32,
    /// Where the dataset the cycle trains on lives.
    pub dataset_root: String,
}

impl Cycle {
    pub fn parse(text: &str) -> Result<Self, DataError> {
        let cycle: Self = es_ir::serial::parse_toml(text)
            .map_err(|e| refuse(format!("the cycle does not parse: {e}")))?;
        if cycle.kind != CYCLE_KIND {
            return Err(refuse(format!(
                "kind = {:?}; `es loop cycle` reads a {CYCLE_KIND:?} document",
                cycle.kind
            )));
        }
        match (&cycle.collect, &cycle.dataset) {
            (Some(_), Some(_)) => Err(refuse(
                "the cycle sets both `[collect]` and `dataset`; one collects the data and the \
                 other reuses a root that already holds it. Delete one",
            )),
            (None, None) => Err(refuse(
                "the cycle sets neither `[collect]` nor `dataset`; there is nothing to train on",
            )),
            _ => Ok(cycle),
        }
    }

    /// The dataset root this cycle trains on; `<out>/collect/ds` when it collects its own.
    pub fn dataset_root(&self, out: &Path) -> String {
        match &self.dataset {
            Some(root) => root.clone(),
            None => under(out, "collect/ds"),
        }
    }

    /// T1's recipe for this cycle: the document `[train] recipe` names (read by the caller —
    /// this module opens no path it was not handed) or the inline tables, with the dataset
    /// slot **overridden by the cycle's own collect output**. A cycle that trained on another
    /// directory's data would chain nothing (spec 13.3).
    pub fn training(&self, recipe_text: Option<&str>, out: &Path) -> Result<Recipe, DataError> {
        let mut recipe = match recipe_text {
            Some(text) => Recipe::parse(text)?,
            None => Recipe {
                kind: s(KIND),
                dataset: Some(self.train.dataset.clone().unwrap_or_default()),
                policy: self.train.policy.clone().ok_or_else(|| {
                    refuse(
                        "[train] names neither `recipe` nor `policy`: one of them says what is \
                         trained",
                    )
                })?,
                run: self.train.run.clone().ok_or_else(|| {
                    refuse("[train] is inline and has no `run` block: steps, batch, lr, seed")
                })?,
                // An inline `[train]` names no policy to start from: a cycle that continues
                // one reaches `[init]` through `[train] recipe`, where the whole recipe --
                // and its `init.lock` -- is one document (packet M8/S1).
                init: None,
                // A cycle collects demonstrations and trains on them; `[rl]` generates its
                // own data and has no collect stage to chain to. Reached, like `[init]`,
                // through `[train] recipe`, where the whole recipe is one document.
                rl: None,
            },
        };
        let dataset = recipe.dataset.get_or_insert_with(DatasetRef::default);
        if let Some(collect) = &self.collect {
            dataset.root = under(out, "collect/ds");
            dataset.frames = collect.frames.then(|| under(out, "collect/frames"));
        } else if let Some(root) = &self.dataset {
            dataset.root.clone_from(root);
        }
        if dataset.root.is_empty() {
            return Err(refuse(
                "[train] is inline with no `[train.dataset] root`, and the cycle collects \
                 nothing to put there",
            ));
        }
        recipe.route()?;
        recipe.marks()?;
        Ok(recipe)
    }

    /// The checkpoint `[eval]` judges: `"last"` is the recipe's largest mark, and anything
    /// else has to be one of them — a checkpoint that is not on disk cannot be evaluated.
    pub fn mark(&self, recipe: &Recipe) -> Result<u32, DataError> {
        let marks = recipe.marks()?;
        if self.eval.checkpoint == "last" {
            return Ok(*marks.last().expect("marks always holds run.steps"));
        }
        let want: u32 = self.eval.checkpoint.parse().map_err(|_| {
            refuse(format!(
                "[eval] `checkpoint` is {:?}; it is \"last\" or one of the recipe's marks \
                 {marks:?}",
                self.eval.checkpoint
            ))
        })?;
        if !marks.contains(&want) {
            return Err(refuse(format!(
                "[eval] `checkpoint` is {want}, which the recipe does not write: its marks are \
                 {marks:?}"
            )));
        }
        Ok(want)
    }
}

impl CyclePlan {
    /// Every stage's command line, from the cycle, the resolved T1 recipe and its plan.
    /// Nothing on disk is read here, so `--dry-run` works on a machine that has neither the
    /// dataset, the bundle nor Python.
    pub fn build(
        cycle: &Cycle,
        recipe: &Recipe,
        train: Plan,
        out: &Path,
    ) -> Result<Self, DataError> {
        let mark = cycle.mark(recipe)?;
        let dataset_root = cycle.dataset_root(out);
        let es = |a: &[&str]| -> Vec<String> { a.iter().map(|w| s(*w)).collect() };
        let mut steps = Vec::new();

        if let Some(collect) = &cycle.collect {
            let mut args = vec![
                s("--policy"),
                collect.policy.clone(),
                s("--scene"),
                cycle.scene.clone(),
                s("--episodes"),
                collect.episodes.to_string(),
                s("--seed"),
                collect.seed.to_string(),
                s("--out"),
                dataset_root.clone(),
            ];
            if collect.frames {
                args.push(s("--frames"));
                args.push(under(out, "collect/frames"));
            }
            if let Some(expert) = &collect.expert {
                args.push(s("--expert"));
                args.push(expert.clone());
            }
            steps.push(CycleStep {
                stage: Stage::Collect,
                prefix: es(&["es", "loop", "collect"]),
                args,
            });
            // Spec 28.9 rule 1. The expert's own report is kept beside the policy's, under its
            // own directory: a harness the expert fails is a harness no policy can pass, and
            // the evidence for that has to survive the run that comes after it.
            if let Some(expert) = &collect.expert {
                steps.push(CycleStep {
                    stage: Stage::ExpertGate,
                    prefix: es(&["es", "eval", "run"]),
                    args: eval_args(cycle, &collect.policy, "eval-expert", out)
                        .into_iter()
                        .chain([s("--expert"), expert.clone()])
                        .collect(),
                });
            }
        }

        steps.push(CycleStep {
            stage: Stage::Train,
            prefix: es(&["es", "train"]),
            args: vec![
                s("--recipe"),
                // The cycle's own word, verbatim (T1's rule 2), or `(inline)` when the tables
                // are in this document. The dataset override is visible in the nested plan
                // below rather than in a rewritten path.
                cycle.train.recipe.clone().unwrap_or_else(|| s("(inline)")),
                s("--out"),
                under(out, "train"),
            ],
        });

        let checkpoint = under(out, &format!("train/checkpoints/{mark}.esb"));
        steps.push(CycleStep {
            stage: Stage::Eval,
            prefix: es(&["es", "eval", "run"]),
            args: eval_args(cycle, &checkpoint, "eval", out),
        });

        if let Some(show) = &cycle.showcase {
            steps.push(CycleStep {
                stage: Stage::Showcase,
                prefix: es(&["es", "video", "showcase"]),
                args: vec![
                    s("--run"),
                    under(out, "eval"),
                    s("--scene"),
                    cycle.scene.clone(),
                    s("--out"),
                    under(out, "showcase"),
                    s("--cell"),
                    show.cell.clone(),
                    s("--eye"),
                    triple(show.eye),
                    s("--look-at"),
                    triple(show.look_at),
                    s("--fov"),
                    show.fov.to_string(),
                    s("--width"),
                    show.width.to_string(),
                    s("--height"),
                    show.height.to_string(),
                ],
            });
        }

        Ok(Self {
            steps,
            train,
            mark,
            dataset_root,
        })
    }

    /// One line per stage, T1's plan indented under the `train` line, every path under `<out>`
    /// written relative to it and every separator a `/` — the same three rules that make the
    /// training plan a property of the recipe alone (design note section 3).
    pub fn render(&self, out: &Path) -> String {
        let prefix = format!("{}/", out.to_string_lossy().replace('\\', "/"));
        let rel = |w: &String| w.replace('\\', "/").replace(&prefix, "");
        let stages: Vec<&str> = self.steps.iter().map(|s| s.stage.as_str()).collect();
        let mut text = format!("# cycle: {}\n", stages.join(" -> "));
        for step in &self.steps {
            let words: Vec<String> = step.prefix.iter().chain(&step.args).map(rel).collect();
            text.push_str(&words.join(" "));
            text.push('\n');
            if step.stage == Stage::Train {
                // Nested, and relative to the *cycle's* `<out>`: the training plan reaches out
                // of `<out>/train` into the collect output, so rendering it against its own
                // directory would leave an absolute path in the golden.
                for line in self.train.render(out).lines() {
                    text.push_str("  ");
                    text.push_str(line);
                    text.push('\n');
                }
            }
        }
        text
    }
}

/// `es eval run`'s words for one policy, shared by the expert gate and the trained policy so
/// that the two runs differ in exactly one thing: what is driving.
fn eval_args(cycle: &Cycle, policy: &str, out_dir: &str, out: &Path) -> Vec<String> {
    let mut args = vec![
        s("--config"),
        cycle.eval.config.clone(),
        s("--policy"),
        s(policy),
        s("--scene"),
        cycle.scene.clone(),
        s("--out"),
        under(out, out_dir),
        s("--jobs"),
        cycle.eval.jobs.to_string(),
    ];
    if cycle.eval.frames {
        args.push(s("--frames"));
        args.push(under(out, &format!("{out_dir}/frames")));
    }
    args
}

fn triple(v: [f64; 3]) -> String {
    format!("{},{},{}", v[0], v[1], v[2])
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
    ///
    /// `observation` is the policy's Observation IR when the run has one on disk — the source
    /// of `augmentation.json` and of `seed.json`'s augmentation slot (packet M7/T6). `None`
    /// is the honest answer for a caller that has not opened it, and it writes the same two
    /// slots this command wrote before the packet.
    pub fn pre_run(
        recipe: &Recipe,
        plan: &Plan,
        out: &Path,
        interpreter: &str,
        data: &DatasetFacts,
        backbone: Option<&Backbone>,
        observation: Option<&ObservationIr>,
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
            // `train_ppo.py` builds `Adam`, not `AdamW`: PPO's reference implementations use
            // it, and decoupled weight decay on a policy that is already entropy-regularised
            // is a second regulariser nobody asked for. Its decay is therefore torch's
            // `Adam` default -- zero -- unless the recipe names one (packet M8/S4b).
            Route::Rl => {
                let mut optimizer = json!({
                    "kind": "Adam", "lr": run.lr, "betas": [0.9, 0.999], "eps": 1e-8,
                    "weight_decay": run.weight_decay.unwrap_or(0.0),
                    "declared_by": "train_ppo.py",
                });
                if let Some(clip) = run.grad_clip {
                    optimizer["grad_clip"] = json!(clip);
                }
                optimizer
            }
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
            // A PPO run starts from `[init]` or from the lowering's own draw, and either way
            // not from ImageNet: `[policy] base_model` needs a `VisionEncoder`, and the RL
            // route's graph reads state.
            Route::Rl => json!({"source": "none"}),
            // Packet M7/T5: a *verified* provenance. `es train` hashed the file, agreed with
            // the lock beside it and with the pin, and what goes into the slot is what the
            // lock said about the weights -- not about the machine that fetched them.
            Route::Ir => match backbone {
                Some(lock) => serde_json::to_value(lock)
                    .map_err(|e| refuse(format!("base_model.lock does not serialise: {e}")))?,
                None => json!({"source": "none"}),
            },
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
        // Packet M7/T6: real when the document declares a chain, and `{"unset": true}` when it
        // does not -- a run with nothing to randomise has no augmentation seed, and inventing
        // one would move every measured run's `training_hash` for a number nothing read.
        let chains = match observation {
            Some(obs) => augmentation_chains(obs).map_err(|d| {
                refuse(
                    d.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            })?,
            None => BTreeMap::new(),
        };
        let augmentation_seed = run.augmentation_seed.unwrap_or(run.seed);
        put(
            "seed.json",
            json!({
                "global": run.seed, "dataloader": run.seed,
                "augmentation": if chains.is_empty() {
                    json!({"unset": true})
                } else {
                    json!(augmentation_seed)
                },
            }),
        );
        // `{"unset": true}`, and hashed as such, for a run whose data is the rollout it
        // generates (spec 28.10 rule 2: real or unset, never fabricated). A zero digest here
        // would claim a dataset of thirty-two zero bytes, and a synthesised one would claim a
        // dataset that does not exist (packet M8/S4b).
        match &recipe.dataset {
            Some(dataset) if route != Route::Rl => put(
                "dataset.lock",
                json!({
                    "root": dataset.root, "episodes": data.episodes, "frames": data.frames,
                    "content": hex(&data.hashes.content), "schema": hex(&data.hashes.schema),
                    "split": hex(&data.hashes.split), "split_source": data.split_source,
                    "recorded_task": data.recorded_task,
                }),
            ),
            // `canon_json` of this is [`UNSET`] byte for byte, which is what makes the slot
            // the same "this run does not know" every other unset slot is.
            _ => put("dataset.lock", json!({"unset": true})),
        }
        put("base_model.lock", base_model);
        // The slot the trainer is *given* (`--augmentation`), not a description of one: the
        // chains here are the nodes `python/es/augment.py` applies, and the seed is the one it
        // keys its counter RNG with. `{"kind": "none"}` stays the whole file for a document
        // that declares no augmentation, so an unaugmented run's digest is the digest it had.
        put(
            "augmentation.json",
            if chains.is_empty() {
                json!({"kind": "none"})
            } else {
                let observation_hash = observation
                    .expect("a chain came from a document")
                    .observation_hash()
                    .map_err(|e| refuse(format!("the Observation IR does not hash: {e}")))?;
                json!({
                    "kind": "observation-ir",
                    "observation_hash": hex(&observation_hash),
                    "seed": augmentation_seed,
                    "chains": chains.iter()
                        .map(|(port, chain)| (port.clone(), augmentation_json(chain)))
                        .collect::<serde_json::Map<_, _>>(),
                })
            },
        );
        put(
            "precision.json",
            json!({
                "dtype": "fp32", "amp": "off",
                "gradient_accumulation": match route {
                    Route::Ir => run.batch.unwrap_or_default(),
                    // One optimizer step per minibatch, nothing accumulated across them.
                    Route::External | Route::Rl => 1,
                },
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
            // The licence slot of `TrainingIdentity`, real at last on the IR route: spec 19.3
            // makes the backbone's licence part of the run's identity, so a run that changed
            // nothing but the licence of its base model is a different run.
            base_license: backbone.map_or_else(|| s("unset"), |b| b.license.clone()),
        })
    }

    /// The thirteenth slot (packet M8/S1): `training/init.lock`, and its digest inside
    /// `config.json`.
    ///
    /// **Why the digest lives in `config.json`.** Spec 19.3 names twelve files and
    /// `TrainingIdentity` has twelve fields; a thirteenth field would be a change to
    /// `es_data::identity`, which this packet does not own. `config.json` is the "this is the
    /// run as configured" slot, and a run that starts from a policy is configured by that
    /// lock as much as by its recipe — so the lock's digest goes in there, the way
    /// `TrainingIdentity.base_model.hash` is the digest of `base_model.lock` rather than the
    /// file's contents. `identity_hash`, `training_hash` and §19.3's
    /// `policy_hash = H(training_hash, checkpoint_hash)` all follow from it, and a recipe
    /// without `[init]` never calls this, so its `config.json` is byte-for-byte the one it
    /// always was.
    ///
    /// Called before [`Training::hash`] is read for the first time, i.e. while the identity
    /// is still the pre-run one: what a run starts from is known before it starts.
    pub fn set_init(&mut self, lock: &Value) -> Result<(), DataError> {
        let text = canon_json(lock);
        let digest = hex(blake3::hash(text.as_bytes()).as_bytes());
        let mut config: Value = serde_json::from_str(self.file("config.json"))
            .map_err(|e| refuse(format!("config.json does not parse: {e}")))?;
        config["init"] = json!(digest);
        self.files.insert(s("config.json"), canon_json(&config));
        self.files.insert(s(INIT_LOCK), text);
        Ok(())
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

    /// Every file's digest, for `training.lock` — the twelve, and `init.lock` when the run
    /// has one (packet M8/S1).
    pub fn digests(&self) -> BTreeMap<String, String> {
        FILES
            .iter()
            .copied()
            .chain(self.files.contains_key(INIT_LOCK).then_some(INIT_LOCK))
            .map(|n| (s(n), hex(&self.digest(n))))
            .collect()
    }

    /// Spec 19.3's twelve slots, each the blake3 of the file that holds it.
    ///
    /// `base_model` carries the provenance strings and the digest of `base_model.lock`
    /// itself; the backbone weights' own blake3 is inside that file — real on the IR route
    /// when the recipe names one (packet M7/T5), still a *declared* string on the external
    /// route, where nothing on this side downloaded or verified `lerobot`'s backbone.
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

    /// Writes the twelve files under `<dir>`, and `init.lock` beside them when there is one.
    pub fn write(&self, dir: &Path) -> Result<(), DataError> {
        for name in FILES
            .iter()
            .copied()
            .chain(self.files.contains_key(INIT_LOCK).then_some(INIT_LOCK))
        {
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
        let plan = Plan::build(&recipe, Path::new(out), "python", &trainer, Some(13), false)
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
        let a = Training::pre_run(
            &recipe,
            &plan_a,
            Path::new("/tmp/a"),
            "python",
            &facts(),
            None,
            None,
        )
        .unwrap();
        let b = Training::pre_run(
            &recipe,
            &plan_b,
            Path::new("/tmp/b"),
            "python",
            &facts(),
            None,
            None,
        )
        .unwrap();
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
            let c = Training::pre_run(
                &r2,
                &p2,
                Path::new("/tmp/a"),
                "python",
                &facts(),
                None,
                None,
            )
            .unwrap();
            assert_ne!(a.hash().unwrap(), c.hash().unwrap(), "{edit}");
        }
    }

    /// No slot is a zero digest and every file parses (packet M7/T1 oracle 2, headless half).
    #[test]
    fn every_slot_is_a_real_digest_of_a_real_file() {
        let (recipe, plan) = plan_of(EXTERNAL, "/tmp/x");
        let t = Training::pre_run(
            &recipe,
            &plan,
            Path::new("/tmp/x"),
            "python",
            &facts(),
            None,
            None,
        )
        .unwrap();
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
        let mut t = Training::pre_run(
            &recipe,
            &plan,
            Path::new("/tmp/x"),
            "python",
            &facts(),
            None,
            None,
        )
        .unwrap();
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
        let before = Training::pre_run(
            &recipe,
            &plan,
            Path::new("/tmp/a"),
            "python",
            &facts(),
            None,
            None,
        )
        .unwrap();
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
        let after = Training::pre_run(
            &r2,
            &p2,
            Path::new("/tmp/a"),
            "python",
            &facts(),
            None,
            None,
        )
        .unwrap();
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

    // --- the cycle (packet M7/T2) -----------------------------------------------------------

    const CYCLE: &str = r#"
kind = "cycle"
scene = "scene.xml"
[collect]
policy = "untrained.esb"
expert = "so101-pick-place"
episodes = 200
seed = 1
frames = true
[train]
recipe = "training.toml"
[eval]
config = "evaluation.toml"
jobs = 6
frames = true
[showcase]
cell = "nominal-00"
eye = [0.66, -0.46, 0.52]
look_at = [0.14, -0.04, 0.04]
fov = 36
"#;

    fn cycle_plan(text: &str, out: &str) -> CyclePlan {
        let cycle = Cycle::parse(text).expect("the cycle parses");
        let out = Path::new(out);
        let recipe = cycle.training(Some(IR), out).expect("the recipe resolves");
        let trainer = vec![s("python"), s("-m"), s("lerobot.scripts.lerobot_train")];
        let plan = Plan::build(
            &recipe,
            &out.join("train"),
            "python",
            &trainer,
            Some(13),
            false,
        )
        .expect("plan builds");
        CyclePlan::build(&cycle, &recipe, plan, out).expect("the cycle plan builds")
    }

    /// The cycle's collect output is what the training recipe reads, whatever the recipe's own
    /// `[dataset]` says: otherwise the ledger chains nothing (spec 13.3).
    #[test]
    fn the_collect_output_overrides_the_recipes_dataset() {
        let cycle = Cycle::parse(CYCLE).expect("parses");
        let recipe = cycle
            .training(Some(IR), Path::new("/tmp/run"))
            .expect("resolves");
        let dataset = recipe
            .dataset
            .as_ref()
            .expect("the IR route names a dataset");
        assert!(dataset.root.ends_with("collect/ds"), "{recipe:?}");
        assert!(
            dataset
                .frames
                .as_deref()
                .unwrap()
                .ends_with("collect/frames"),
            "{recipe:?}"
        );
        // "last" is the recipe's largest mark; a mark it does not write is refused by name.
        assert_eq!(cycle.mark(&recipe).unwrap(), 20000);
        let other = CYCLE.replace("jobs = 6", "jobs = 6\ncheckpoint = \"7000\"");
        let e = Cycle::parse(&other)
            .unwrap()
            .mark(&recipe)
            .expect_err("refused");
        assert!(e.to_string().contains("7000"), "{e}");
    }

    /// The rendered cycle holds no absolute path, on either separator, and the training plan
    /// is nested under its own stage.
    #[test]
    fn the_cycle_render_is_relative_to_out() {
        let a = cycle_plan(CYCLE, "/tmp/scratch-a").render(Path::new("/tmp/scratch-a"));
        let b = cycle_plan(CYCLE, "/var/other-b").render(Path::new("/var/other-b"));
        assert_eq!(a, b, "the plan is a property of the document, not of --out");
        assert!(!a.contains("scratch-a"), "{a}");
        assert!(
            a.contains("# cycle: collect -> expert-gate -> train -> eval -> showcase"),
            "{a}"
        );
        assert!(a.contains("es loop collect --policy untrained.esb"), "{a}");
        assert!(a.contains("--expert so101-pick-place"), "{a}");
        assert!(
            a.contains("\n  # route: ir\n"),
            "the T1 plan is nested: {a}"
        );
        assert!(a.contains("  es dataset bake"), "{a}");
        assert!(a.contains("--policy train/checkpoints/20000.esb"), "{a}");
        assert!(
            a.contains("--eye 0.66,-0.46,0.52 --look-at 0.14,-0.04,0.04"),
            "{a}"
        );
    }

    /// A cycle names one source of data.
    #[test]
    fn a_cycle_collects_or_reuses_but_not_both() {
        let both = CYCLE.replace("[collect]", "dataset = \"runs/ds\"\n[collect]");
        let e = Cycle::parse(&both).expect_err("refused");
        assert!(e.to_string().contains("both"), "{e}");
        let block = CYCLE
            .split_once("[collect]\n")
            .and_then(|(_, rest)| rest.split_once("[train]"))
            .map(|(block, _)| block.to_owned())
            .expect("the fixture has a [collect] block");
        let neither = CYCLE.replace(&format!("[collect]\n{block}"), "");
        let e = Cycle::parse(&neither).expect_err("refused");
        assert!(e.to_string().contains("neither"), "{e}");
    }

    /// `[run] extra` reaches the trainer's command line on the IR route.
    #[test]
    fn a_trainer_flag_from_the_recipe_is_on_the_command_line() {
        let text = IR.replace(
            "device = \"cuda\"",
            "device = \"cuda\"\nextra = [\"--resident-gpu\"]",
        );
        let (_, plan) = plan_of(&text, "/tmp/x");
        assert!(
            plan.render(Path::new("/tmp/x"))
                .contains("--loss-curve metrics/loss.json --resident-gpu"),
            "{}",
            plan.render(Path::new("/tmp/x"))
        );
    }

    /// Packet M7/T5. `[policy] base_model` is the IR route's, it puts `--init-backbone` on
    /// the trainer's line, and a recipe that names none renders the plan it always did.
    #[test]
    fn base_model_is_the_ir_routes_and_reaches_the_trainer() {
        let (_, plan) = plan_of(IR, "/tmp/a");
        assert!(
            !plan.render(Path::new("/tmp/a")).contains("--init-backbone"),
            "a recipe that named no base_model got one"
        );

        let named = IR.replace(
            "bundle = \"untrained.esb\"",
            "bundle = \"untrained.esb\"\nbase_model = \"backbones/resnet18.safetensors\"",
        );
        let (_, plan) = plan_of(&named, "/tmp/a");
        let rendered = plan.render(Path::new("/tmp/a"));
        assert!(
            rendered.contains("--init-backbone backbones/resnet18.safetensors"),
            "{rendered}"
        );
        // `frozen` is the IR's and stays there: the lowered module carries it.
        assert!(
            !rendered.contains("--frozen") && !rendered.contains("--freeze"),
            "{rendered}"
        );

        let external = EXTERNAL.replace(
            "task = \"task.toml\"",
            "task = \"task.toml\"\nbase_model = \"backbones/resnet18.safetensors\"",
        );
        let e = Recipe::parse(&external).expect_err("refused");
        assert!(e.to_string().contains("base_model"), "{e}");
    }

    /// The verified lock is what `base_model.lock` holds, and it moves `training_hash`.
    #[test]
    fn a_verified_backbone_fills_the_base_model_slot() {
        let (recipe, plan) = plan_of(IR, "/tmp/a");
        let none = Training::pre_run(
            &recipe,
            &plan,
            Path::new("/tmp/a"),
            "python",
            &facts(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(none.file("base_model.lock"), "{\"source\":\"none\"}\n");

        let lock = Backbone {
            source: s(BASE_MODEL_SOURCE),
            url: s("https://download.pytorch.org/models/resnet18-f37072fd.pth"),
            sha256_upstream: s("f37072fd"),
            blake3: s(RESNET18_IMAGENET1K_V1_BLAKE3),
            dropped: s("*.num_batches_tracked"),
            license: s("BSD-3-Clause"),
            license_url: s("https://github.com/pytorch/vision/blob/main/LICENSE"),
        };
        let with = Training::pre_run(
            &recipe,
            &plan,
            Path::new("/tmp/a"),
            "python",
            &facts(),
            Some(&lock),
            None,
        )
        .unwrap();
        let written: Value = serde_json::from_str(with.file("base_model.lock")).unwrap();
        assert_eq!(written["blake3"], RESNET18_IMAGENET1K_V1_BLAKE3);
        assert_eq!(written["license"], "BSD-3-Clause");
        // The fetching machine's torch version is deliberately absent: it would make one
        // recipe have two identities depending on where its backbone was produced.
        assert!(written.get("torch").is_none(), "{written}");
        assert_eq!(with.identity().base_model.license, "BSD-3-Clause");
        assert_ne!(none.hash().unwrap(), with.hash().unwrap());
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
