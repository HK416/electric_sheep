//! The command plan (packet M7/T1): a run as the commands that execute it, the paths those
//! commands name under `<out>`, and the facts about the documents it is built from.

use std::path::Path;

use es_ir::observation::{ObservationIr, ObservationNode};

use super::{refuse, s, under, Lerobot, Recipe, Route, EXPORT_DROP, FETCH_BACKBONE, TRAIN_ACT};
use crate::DataError;

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
    /// [`FETCH_BACKBONE`]'s whole command line when the recipe declares `base_model_fetch`
    /// (packet M12/R8), run only when `base_model` is not on disk -- so a comment in the
    /// rendered plan, not a step.
    pub fetch: Option<Vec<String>>,
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
///
/// [`init_from`]: super::init_from
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
                    // `[run] single_view` (packet M15/N7): absent, nothing.
                    .chain(recipe.single_view_args()?)
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
                    // The IR route's pretrained backbone, the same flag and the same function
                    // in `train_act.py` (packet M11/R10); absent, the plan of before.
                    .chain(
                        recipe
                            .policy
                            .base_model
                            .iter()
                            .flat_map(|path| [s("--init-backbone"), path.clone()]),
                    )
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
                    path: None,
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
                    // `es loop collect --frames` writes a flat one for one camera, so the shell
                    // mirrors each camera's tiles into `<out>/frames-in/<camera>` first. Packet
                    // M5/V19 did that by hand with `mkdir` and `ln -s`; a recipe that needed it
                    // would not be one command.
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
                    match &lerobot.path {
                        Some(path) => format!("--policy.path={path}"),
                        None => format!("--policy.type={}", lerobot.kind),
                    },
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
        let fetch_args = recipe.fetch_args()?;
        let fetch = (!fetch_args.is_empty()).then(|| {
            [s(interpreter), s(FETCH_BACKBONE)]
                .into_iter()
                .chain(fetch_args)
                .collect()
        });
        Ok(Self {
            route,
            steps,
            marks,
            fetch,
        })
    }

    /// `# fetch: <command>` and a newline, when the recipe declares one (packet M12/R8).
    pub fn fetch_line(&self) -> Option<String> {
        let words = self.fetch.as_ref()?;
        Some(format!("# fetch: {}\n", words.join(" ").replace('\\', "/")))
    }

    /// One line per step, every path under `<out>` written relative to it and every separator
    /// a `/` — so two machines with different scratch directories, and Windows and Linux,
    /// render one plan and the golden is a property of the recipe alone.
    pub fn render(&self, out: &Path) -> String {
        // `<out>/` is stripped wherever it appears in a word, not only at its head: the
        // external trainer takes `--flag=value` and the value is the path.
        let prefix = format!("{}/", out.to_string_lossy().replace('\\', "/"));
        let mut text = format!("# route: {}\n", self.route.as_str());
        text.push_str(&self.fetch_line().unwrap_or_default());
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

/// Each camera of a dataset `es loop collect` wrote, by [`camera_suffix`] -- which is its Task
/// IR channel's name -- with the directory its tiles are in under the collection's `--frames`:
/// the root itself for one camera (the flat layout), `<frames>/<channel>/` each for several
/// (packet M15/N2). What `es train` mirrors for the export and `es loop distill` merges.
pub fn camera_dirs(info: &crate::Info, frames: &Path) -> Vec<(String, std::path::PathBuf)> {
    let cameras: Vec<&str> = info.cameras().map(|c| camera_suffix(c)).collect();
    let several = cameras.len() > 1;
    cameras
        .into_iter()
        .map(|c| {
            let dir = if several {
                frames.join(c)
            } else {
                frames.to_path_buf()
            };
            (c.to_owned(), dir)
        })
        .collect()
}
