//! `cycle.toml`, the cycle (packet M7/T2): collect, train, evaluate and showcase under one
//! document -- its tables and refusals, the dataset and recipe it resolves to, the checkpoint
//! `[eval]` judges, and the Evaluation IR a checkpoint preview runs (packet M13/Z1).

use std::path::Path;

use es_ir::evaluation::{EpisodeBatch, EvaluationIr, SeedPlan};
use serde::{Deserialize, Serialize};

use super::{refuse, s, under, DatasetRef, InitRef, PolicyRef, Recipe, Route, Run, KIND};
use crate::DataError;

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
    /// The bundle whose Deployment IR is the plane (`es loop collect --policy`). Required;
    /// defaulted only so that a `[collect]` holding nothing but `merge` is refused by name.
    #[serde(default)]
    pub policy: String,
    /// The scripted demonstrator, or absent for a trained policy's own rollouts. Setting it
    /// is what arms the expert gate of spec 28.9 rule 1. A built-in name or a demonstration
    /// program's `.toml` path (packet M14/Q2), passed through verbatim to both `es loop
    /// collect --expert` and the gate's `es eval run --expert`, which resolve it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert: Option<String>,
    /// Required and never 0; defaulted for the same reason as `policy`.
    #[serde(default)]
    pub episodes: u32,
    #[serde(default)]
    pub seed: u64,
    /// Render the Task IR's image channel beside the dataset (`es loop collect --frames`).
    #[serde(default)]
    pub frames: bool,
    /// Collect under an Evaluation IR's own perturbations (packet M13/Z3): `es loop collect
    /// --perturb <config> --suites <a,b>` (packet M13/Z2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perturb: Option<PerturbRef>,
    /// Earlier dataset roots, merged with this collection by `es loop distill` into
    /// `<out>/collect/merged`, which is then what trains (packet M13/Z3). Each root's frame
    /// tiles are the `frames` directory beside it -- the layout a cycle writes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub merge: Vec<String>,
    /// Train on the successful episodes only (packet M16/H3): a merge stage, `es loop distill
    /// --success-only`, of this collection (and `merge`'s roots) into
    /// `<out>/collect/successes/ds` with its tiles in `collect/successes/frames`, which is then
    /// what trains. For a demonstrator that is itself a policy and succeeds only sometimes.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub success_only: bool,
}

/// `[collect] perturb = { config, suites }` (packet M13/Z3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerturbRef {
    /// An Evaluation IR by path; its suites are what the collection runs under.
    pub config: String,
    /// Episode `i` runs under `suites[i % len]`.
    pub suites: Vec<String>,
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
    /// What the run starts from (packet M13/Z3): a bundle, which becomes the IR and RL routes'
    /// `[init] policy`, or a `lerobot` `pretrained_model` directory, which becomes the lerobot
    /// route's `--policy.path` -- its fine-tuning path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init: Option<String>,
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
    /// A short test of every checkpoint while the run trains; absent, none (packet M13/Z1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<PreviewRef>,
}

fn default_checkpoint() -> String {
    "last".to_owned()
}

fn one_job() -> u32 {
    1
}

/// `[eval.preview]` — after each checkpoint bundle is written, `es loop cycle` runs a child
/// `es eval run` on it, so a person sees the policy get better while it trains (packet M13/Z1).
///
/// Never the evaluation the run is judged by (spec 13.3): [`preview_evaluation`] derives its
/// own document from `[eval] config` -- one suite, a few of its seeds, **no acceptance** -- and
/// the preview's results stay under `<out>/preview/<mark>/`, out of the ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewRef {
    /// The suite's first `episodes` seeds.
    #[serde(default = "four_episodes")]
    pub episodes: u32,
    /// A suite of `[eval] config`; absent, its first -- the nominal one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suite: Option<String>,
    /// Render the frames, which is what makes a preview something to watch.
    #[serde(default = "rendered")]
    pub frames: bool,
}

impl Default for PreviewRef {
    fn default() -> Self {
        Self {
            episodes: four_episodes(),
            suite: None,
            frames: rendered(),
        }
    }
}

fn four_episodes() -> u32 {
    4
}

fn rendered() -> bool {
    true
}

/// The Evaluation IR a preview runs: `[eval] config`'s chosen suite alone, its first
/// `episodes` seeds (never more than the document has), the same metrics, and no acceptance --
/// a preview is watched, never judged (spec 13.3).
///
/// ponytail: a suite other than the first moves to index 0, which is its `suite_id` in the
/// perturbation draws (spec 10.4), so its draws are not the full evaluation's for the same
/// seed; keep the other suites and run one if a preview must equal its rows.
pub fn preview_evaluation(
    ir: &EvaluationIr,
    preview: &PreviewRef,
) -> Result<EvaluationIr, DataError> {
    let suite = match &preview.suite {
        Some(name) => ir.suites.iter().find(|s| s.name == *name).ok_or_else(|| {
            let declared: Vec<&str> = ir.suites.iter().map(|s| s.name.as_str()).collect();
            refuse(format!(
                "[eval.preview] `suite` is \"{name}\", which [eval] config does not declare: \
                 its suites are {declared:?}"
            ))
        })?,
        None => ir
            .suites
            .first()
            .ok_or_else(|| refuse("[eval] config declares no suite to preview"))?,
    };
    let n = preview.episodes.min(ir.episodes.n_episodes);
    let seeds = match &ir.episodes.seeds {
        SeedPlan::Base(base) => SeedPlan::Base(*base),
        SeedPlan::Explicit(list) => {
            SeedPlan::Explicit(list.iter().copied().take(n as usize).collect())
        }
    };
    Ok(EvaluationIr {
        episodes: EpisodeBatch {
            n_episodes: n,
            seeds,
        },
        suites: vec![suite.clone()],
        acceptance: Vec::new(),
        ..ir.clone()
    })
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
    /// `[collect] merge`: `es loop distill` of the new dataset with the earlier roots (packet
    /// M13/Z3). Part of the collect stage's data, so, like the gate, not a `--from` name.
    Merge,
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
            Self::Merge => "merge",
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
        if cycle.eval.preview.as_ref().is_some_and(|p| p.episodes == 0) {
            return Err(refuse(
                "[eval.preview] `episodes` is 0; a preview with no episodes shows nothing. \
                 Delete [eval.preview] to run none",
            ));
        }
        if let Some(c) = &cycle.collect {
            if !c.merge.is_empty() && (c.policy.is_empty() || c.episodes == 0) {
                return Err(refuse(
                    "[collect] `merge` merges earlier roots with what this cycle collects, and \
                     this [collect] collects nothing (no `policy`, or `episodes` 0). To train on \
                     earlier roots alone, `es loop distill` them and name the result in \
                     `dataset`",
                ));
            }
            if c.policy.is_empty() || c.episodes == 0 {
                return Err(refuse(
                    "[collect] needs `policy` (the bundle whose Deployment IR the collection \
                     runs under) and `episodes` above 0",
                ));
            }
            if c.perturb.as_ref().is_some_and(|p| p.suites.is_empty()) {
                return Err(refuse(
                    "[collect] perturb `suites` is empty; name the suites of `config` to collect \
                     under, or delete `perturb`",
                ));
            }
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

    /// The dataset root this cycle trains on: `<out>/collect/ds` when it collects its own,
    /// `<out>/collect/merged` when that is merged with `[collect] merge` (packet M13/Z3), and
    /// `<out>/collect/successes/ds` when only the successes are kept (packet M16/H3).
    pub fn dataset_root(&self, out: &Path) -> String {
        match &self.dataset {
            Some(root) => root.clone(),
            None if self.success_only() => under(out, "collect/successes/ds"),
            None if self.merge().is_empty() => collect_root(out),
            None => under(out, "collect/merged"),
        }
    }

    /// The frame tiles of [`Self::dataset_root`] when the cycle collects them: the `frames`
    /// beside it.
    pub(super) fn dataset_frames(&self, out: &Path) -> String {
        if self.success_only() {
            under(out, "collect/successes/frames")
        } else {
            under(out, "collect/frames")
        }
    }

    /// `[collect] success_only`.
    pub fn success_only(&self) -> bool {
        self.collect.as_ref().is_some_and(|c| c.success_only)
    }

    /// Whether a merge stage (`es loop distill`) writes the root that trains.
    pub fn merges(&self) -> bool {
        !self.merge().is_empty() || self.success_only()
    }

    /// `[collect] merge`, empty when there is none.
    pub fn merge(&self) -> &[String] {
        self.collect.as_ref().map_or(&[], |c| &c.merge)
    }

    /// What `[train] init` and `[collect] merge` name has to be on disk before a real run
    /// starts, or the collection before it is wasted (packet M13/Z3). `--dry-run` does not call
    /// this: it opens nothing, which is what keeps the goldens judgeable without a run.
    pub fn check_inputs(&self, route: Route) -> Result<(), DataError> {
        if let Some(init) = &self.train.init {
            let at = Path::new(init);
            let (ok, what) = match route {
                Route::External => (
                    at.join("model.safetensors").is_file() && at.join("config.json").is_file(),
                    "a lerobot `pretrained_model` directory (model.safetensors, config.json)",
                ),
                Route::Ir | Route::Rl => (at.is_file(), "a bundle"),
            };
            if !ok {
                return Err(refuse(format!(
                    "[train] `init` is {init:?}, which is not {what} on disk"
                )));
            }
        }
        let frames = self.collect.as_ref().is_some_and(|c| c.frames);
        for root in self.merge() {
            crate::LeRobotDataset::open(Path::new(root))
                .map_err(|e| refuse(format!("[collect] `merge` {root}: {e}")))?;
            let tiles = frames_beside(root);
            if frames && !Path::new(&tiles).is_dir() {
                return Err(refuse(format!(
                    "[collect] `merge` {root}: this cycle trains on frames and {tiles} is not a \
                     directory; a merged root's tiles are the `frames` beside it"
                )));
            }
        }
        Ok(())
    }

    /// T1's recipe for this cycle: the document `[train] recipe` names (read by the caller —
    /// this module opens no path it was not handed, except the `[policy] bundle` a
    /// `[run] single_view` is checked against) or the inline tables, with the dataset
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
                // `[train] init` below, or `[init]` through `[train] recipe` (packet M8/S1).
                init: None,
                // A cycle collects demonstrations and trains on them; `[rl]` generates its
                // own data and has no collect stage to chain to. Reached, like `[init]`,
                // through `[train] recipe`, where the whole recipe is one document.
                rl: None,
            },
        };
        let dataset = recipe.dataset.get_or_insert_with(DatasetRef::default);
        if let Some(collect) = &self.collect {
            // The merge extends `collect/frames` with the earlier roots' tiles, so one
            // directory serves `collect/ds` and `collect/merged` alike (packet M13/Z3).
            dataset.root = self.dataset_root(out);
            dataset.frames = collect.frames.then(|| self.dataset_frames(out));
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
        if let Some(init) = &self.train.init {
            // The lerobot route is the recipe with `[policy] lerobot`.
            let taken = match recipe.policy.lerobot.as_mut() {
                Some(lerobot) => lerobot.path.replace(init.clone()).is_some(),
                None => recipe
                    .init
                    .replace(InitRef {
                        policy: init.clone(),
                    })
                    .is_some(),
            };
            if taken {
                return Err(refuse(format!(
                    "[train] `init` is {init:?} and the recipe already names what it starts \
                     from; one of the two, not both"
                )));
            }
        }
        recipe.marks()?;
        recipe.fetch_args()?;
        // An inline `[train] run` never went through `Recipe::parse`, so both halves of
        // `single_view`'s check run here (packet M15/N7); the second reads the bundle only
        // when the field is set.
        recipe.single_view_args()?;
        recipe.check_single_view()?;
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

/// Where a cycle's `[collect]` writes its dataset.
pub fn collect_root(out: &Path) -> String {
    under(out, "collect/ds")
}

/// A dataset root's frame tiles, in the layout a cycle writes: the `frames` directory
/// beside it (`collect/ds`, `collect/merged` and `collect/frames`).
///
/// ponytail: a convention, not a declaration; a merged root from elsewhere needs a
/// `{ root, frames }` entry in `[collect] merge` if that ever comes up.
pub fn frames_beside(root: &str) -> String {
    let parent = Path::new(root).parent().unwrap_or(Path::new(""));
    parent.join("frames").to_string_lossy().into_owned()
}
