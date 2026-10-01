//! `training.toml`, the recipe (packet M7/T1): its tables, the route it describes, the
//! checkpoint marks, and every refusal [`Recipe::parse`] makes before anything runs -- the
//! `[run]` schedule (packet M7/T4) and single-view weight (packet M15/N7) among them.

use es_ir::learning::LearningGraph;
use serde::{Deserialize, Serialize};

use super::{refuse, s, Rl};
use crate::DataError;

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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRef {
    /// A `LeRobot` v2.1 root, as `es loop collect` writes it.
    pub root: String,
    /// The `<NNNNNN>.bin` tiles beside it (`es loop collect --frames`): flat for one camera,
    /// `<frames>/<channel>/` each for several ([`camera_dirs`]). Required when the Observation
    /// IR has an image input.
    ///
    /// [`camera_dirs`]: super::camera_dirs
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
    /// How `base_model` is obtained when it is not on disk: `fetch_backbone.py --arch` of it
    /// (packet M12/R8). `es train` -- and `es loop cycle`, before it collects -- runs that script
    /// with the run's interpreter and verifies what it wrote. The pin stays
    /// [`RESNET18_IMAGENET1K_V1_BLAKE3`]; the document holds no second copy of it.
    ///
    /// [`RESNET18_IMAGENET1K_V1_BLAKE3`]: super::RESNET18_IMAGENET1K_V1_BLAKE3
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_fetch: Option<String>,
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
    /// `--policy.path` (packet M13/Z3): the `pretrained_model` directory this run fine-tunes,
    /// in place of `--policy.type`, which `lerobot-train` refuses beside it. The other
    /// `--policy.*` flags then override the loaded config. A cycle's `[train] init` sets it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
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
    /// `single_view = { weight = alpha }` — MAD's single-view loss (packet M15/N7, design note
    /// `multi-camera.md` section 3.3): `train_act.py --single-view alpha`. The IR route's only,
    /// and only for a graph whose cameras meet in a `Sum` ([`Recipe::check_single_view`]).
    /// Absent is absent: no flag, and the plan and `identity_hash` of before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub single_view: Option<SingleView>,
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
    /// Linear warmup from 0 to `[run] lr`, in `[run] steps`' unit: optimizer steps on the IR
    /// route, PPO iterations on the `[rl]` route, where every epoch and minibatch of an
    /// iteration uses that iteration's rate (packet M11/R11).
    #[serde(default)]
    pub warmup: u32,
    /// The floor the cosine decays to.
    #[serde(default)]
    pub lr_min: f64,
}

/// `[run] single_view` (packet M15/N7). A table, not a bare number, so what is weighted is
/// named where it is written.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SingleView {
    /// `alpha`: the weight of the mean single-view loss beside the all-views loss. Above 0.
    pub weight: f64,
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
        recipe.fetch_args()?;
        recipe.single_view_args()?;
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

    /// `[run] single_view` validated, as `train_act.py`'s `--single-view <alpha>` (packet
    /// M15/N7). Empty when it is absent, which keeps every plan and its golden byte-identical.
    ///
    /// Refused by name off the IR route: MAD (arXiv 2505.04619) weights the single-view
    /// features into both the actor and the critic of an RL agent, and `train_ppo.py`
    /// implements neither; `lerobot-train` has no such loss at all. The graph half of the
    /// check needs the bundle and is [`Recipe::check_single_view`].
    pub fn single_view_args(&self) -> Result<Vec<String>, DataError> {
        let Some(single) = &self.run.single_view else {
            return Ok(Vec::new());
        };
        if !single.weight.is_finite() || single.weight <= 0.0 {
            return Err(refuse(format!(
                "[run] `single_view.weight` is {}; it weights the single-view loss beside the \
                 all-views one and must be above 0. Delete `single_view` to train on all views \
                 together only",
                single.weight
            )));
        }
        match self.route()? {
            Route::Ir => Ok(vec![s("--single-view"), single.weight.to_string()]),
            Route::Rl => Err(refuse(
                "[run] `single_view` is the imitation trainer's (`train_act.py`): MAD's RL form \
                 weights the single-view features into both the actor and the critic, and \
                 `train_ppo.py` implements neither. Delete it beside `[rl]`",
            )),
            Route::External => Err(refuse(
                "[run] `single_view` is the IR route's: it drops a camera's term from the \
                 Learning IR's `Sum` fusion, and `lerobot-train` trains lerobot's own \
                 architecture, which has none",
            )),
        }
    }

    /// `[run] single_view` against the graph it would train (packet M15/N7): the bundle `[policy]
    /// bundle` names must have cameras that meet in a `Sum` fusion
    /// ([`es_policy::lower::sum_views`]), or there is no camera's term to drop and the recipe
    /// is refused by name. Reads the bundle only when the field is set, so every other recipe
    /// is still judged, `--dry-run` included, without one on disk. `es train` calls this before
    /// its plan, and [`Cycle::training`] before a cycle's.
    ///
    /// [`Cycle::training`]: super::Cycle::training
    pub fn check_single_view(&self) -> Result<(), DataError> {
        if self.run.single_view.is_none() {
            return Ok(());
        }
        let path = self.policy.bundle.as_deref().unwrap_or_default();
        let bytes = std::fs::read(path).map_err(|e| {
            refuse(format!(
                "[run] `single_view` is checked against the bundle's Learning IR before \
                 anything runs, and [policy] `bundle` {path:?} cannot be read: {e}"
            ))
        })?;
        let bundle = es_compile::PolicyBundle::open(&bytes)
            .map_err(|e| refuse(format!("[policy] `bundle` {path}: {e}")))?;
        check_single_view(&bundle.learning)
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

/// The graph half of [`Recipe::check_single_view`] (packet M15/N7): single-view training
/// drops a camera's term from a `Sum` fusion, so a Learning IR with no `Sum` fed by a
/// `VisionEncoder` has nothing to drop and is refused by name.
pub fn check_single_view(learning: &LearningGraph) -> Result<(), DataError> {
    if es_policy::lower::sum_views(learning).is_empty() {
        return Err(refuse(
            "[run] `single_view` trains each camera alone by leaving the others' terms out of \
             a `Sum` fusion, and this bundle's Learning IR has no `Sum` whose inputs come from \
             `VisionEncoder`s (a `Concat` of views cannot lose one without retraining). Give \
             the views a `Sum` fusion, or delete `single_view`",
        ));
    }
    Ok(())
}
