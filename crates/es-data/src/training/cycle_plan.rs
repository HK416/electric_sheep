//! The cycle's command plan (packet M7/T2): one line per stage, T1's plan nested under
//! `train`, and one checkpoint preview per mark (packet M13/Z1).

use std::path::Path;

use super::{collect_root, frames_beside, s, under, Cycle, Plan, PreviewRef, Recipe, Stage};
use crate::DataError;

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
    /// `[eval.preview]`, and one preview per training mark; both empty without it.
    pub preview: Option<PreviewRef>,
    pub previews: Vec<PreviewStep>,
    /// `[train] init`, said under the train line: the IR route's plan names only the file
    /// `init_from` writes, not where its tensors came from.
    pub init: Option<String>,
}

/// One checkpoint's preview: `es eval run`'s words, run as a child of `es loop cycle` once the
/// mark's bundle is on disk (packet M13/Z1). `--config` is `<dir>/evaluation.toml`, the
/// [`preview_evaluation`] document the cycle writes there first.
///
/// [`preview_evaluation`]: super::preview_evaluation
#[derive(Clone, Debug, PartialEq)]
pub struct PreviewStep {
    pub mark: u32,
    /// `<out>/preview/<mark>`: the derived IR, then the run's usual artifacts.
    pub dir: String,
    pub args: Vec<String>,
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
            let frames = collect.frames.then(|| under(out, "collect/frames"));
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
                collect_root(out),
            ];
            if let Some(frames) = &frames {
                args.push(s("--frames"));
                args.push(frames.clone());
            }
            if let Some(expert) = &collect.expert {
                args.push(s("--expert"));
                args.push(expert.clone());
            }
            if let Some(p) = &collect.perturb {
                args.extend([s("--perturb"), p.config.clone()]);
                args.extend([s("--suites"), p.suites.join(",")]);
            }
            steps.push(CycleStep {
                stage: Stage::Collect,
                prefix: es(&["es", "loop", "collect"]),
                args,
            });
            // Packet M13/Z3: the new dataset first, then the earlier roots, into the root that
            // trains. All-train, because `es train` trains on every episode of its root and the
            // split `split.json` records should say so. The tiles go into `collect/frames`
            // after the new collection's own, which is the merged root's global frame order.
            // Packet M16/H3: `success_only` is the same stage keeping the successes, its tiles
            // renumbered into a directory of their own beside the root it writes.
            if cycle.merges() {
                let mut args = Vec::new();
                for root in std::iter::once(collect_root(out)).chain(collect.merge.clone()) {
                    let tiles = frames.as_ref().map(|_| frames_beside(&root));
                    args.extend([s("--in"), root]);
                    args.extend(tiles.into_iter().flat_map(|t| [s("--in-frames"), t]));
                }
                args.extend(["--train", "1", "--val", "0", "--test", "0"].map(s));
                args.extend([s("--out"), dataset_root.clone()]);
                let into = frames.as_ref().map(|_| cycle.dataset_frames(out));
                args.extend(into.into_iter().flat_map(|f| [s("--frames"), f]));
                if collect.success_only {
                    args.push(s("--success-only"));
                }
                steps.push(CycleStep {
                    stage: Stage::Merge,
                    prefix: es(&["es", "loop", "distill"]),
                    args,
                });
            }
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

        let preview = cycle.eval.preview.clone();
        let previews = match &preview {
            Some(p) => train
                .marks
                .iter()
                .map(|&m| preview_step(cycle, p, m, out))
                .collect(),
            None => Vec::new(),
        };

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
            preview,
            previews,
            init: cycle.train.init.clone(),
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
        // Above every stage and not under `train`: `es loop cycle` fetches a missing backbone
        // before it collects, so it is never the train stage that finds it gone (M12/R8).
        text.push_str(&self.train.fetch_line().unwrap_or_default());
        for step in &self.steps {
            let words: Vec<String> = step.prefix.iter().chain(&step.args).map(rel).collect();
            text.push_str(&words.join(" "));
            text.push('\n');
            if step.stage == Stage::Train {
                // Nested, and relative to the *cycle's* `<out>`: the training plan reaches out
                // of `<out>/train` into the collect output, so rendering it against its own
                // directory would leave an absolute path in the golden.
                let rendered = self.train.render(out);
                for line in rendered.lines().filter(|l| !l.starts_with("# fetch: ")) {
                    text.push_str("  ");
                    text.push_str(line);
                    text.push('\n');
                }
                if let Some(init) = &self.init {
                    text.push_str("  # init: from ");
                    text.push_str(&rel(init));
                    text.push('\n');
                }
                // Beside the training plan and not in it: a preview runs as the cycle's child
                // while the trainer goes on, and the trainer's plan is `training.lock`'s.
                if let Some(p) = &self.preview {
                    let suite = p
                        .suite
                        .as_ref()
                        .map_or_else(|| s("the first suite"), |n| format!("suite {n}"));
                    text.push_str("  # preview: after each checkpoint, one at a time, ");
                    text.push_str(&p.episodes.to_string());
                    text.push_str(" episode(s) of ");
                    text.push_str(&suite);
                    text.push('\n');
                }
                for p in &self.previews {
                    let words: Vec<String> = ["es", "eval", "run"]
                        .iter()
                        .map(s)
                        .chain(p.args.iter().map(rel))
                        .collect();
                    text.push_str("  ");
                    text.push_str(&words.join(" "));
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

/// `es eval run`'s words for one mark's preview: the derived document, the mark's bundle, one
/// worker -- the preview shares the machine with the trainer.
fn preview_step(cycle: &Cycle, preview: &PreviewRef, mark: u32, out: &Path) -> PreviewStep {
    let dir = under(out, &format!("preview/{mark}"));
    let mut args = vec![
        s("--config"),
        under(out, &format!("preview/{mark}/evaluation.toml")),
        s("--policy"),
        under(out, &format!("train/checkpoints/{mark}.esb")),
        s("--scene"),
        cycle.scene.clone(),
        s("--out"),
        dir.clone(),
        s("--jobs"),
        s("1"),
    ];
    if preview.frames {
        args.push(s("--frames"));
        args.push(under(out, &format!("preview/{mark}/frames")));
    }
    PreviewStep { mark, dir, args }
}

fn triple(v: [f64; 3]) -> String {
    format!("{},{},{}", v[0], v[1], v[2])
}
