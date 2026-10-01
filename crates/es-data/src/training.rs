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

use std::path::Path;

use crate::DataError;

mod backbone;
mod cycle;
mod cycle_plan;
mod files;
mod init;
mod plan;
mod recipe;
mod rl;

pub use backbone::{has_pretrained_backbone, Backbone, FETCH_BACKBONE};
pub use cycle::{
    collect_root, frames_beside, preview_evaluation, CollectRef, Cycle, EvalRef, PerturbRef,
    PreviewRef, ShowcaseRef, Stage, TrainRef, CYCLE_KIND,
};
pub use cycle_plan::{CyclePlan, CycleStep, PreviewStep};
pub use files::{augmentation_json, canon_json, DatasetFacts, Training};
pub use init::{init_from, untrained_bundle, Init, INIT_LOCK};
pub use plan::{
    camera_dirs, camera_suffix, has_image_input, init_weights, ir_checkpoint, lerobot_checkpoint,
    rollout_docs, state_dim, value_weights, Plan, Step, StepKind, TRAIN_PPO,
};
pub use recipe::{
    check_single_view, DatasetRef, InitRef, Lerobot, PolicyRef, Recipe, Route, Run, Schedule,
    SingleView,
};
pub use rl::{Critic, Estimator, Rl, RlBackend};

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

fn s(v: impl AsRef<str>) -> String {
    v.as_ref().to_owned()
}

/// `out.join(rest)` as a string — the plan holds real paths and only [`Plan::render`] makes
/// them relative.
fn under(out: &Path, rest: &str) -> String {
    out.join(rest).to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_ir::evaluation::{EvaluationIr, SeedPlan};
    use es_ir::DatasetHash;
    use serde_json::{json, Value};

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

    /// Packet M14/Q2: `[collect] expert` may be a program's path, and the plan carries it
    /// verbatim to the collection and to the expert gate; nothing else in the plan moves.
    #[test]
    fn a_cycle_passes_an_expert_program_path_through() {
        let path = "projects/cube/teach.toml";
        let named = cycle_plan(CYCLE, "/tmp/q2").render(Path::new("/tmp/q2"));
        let text = CYCLE.replace("\"so101-pick-place\"", &format!("{path:?}"));
        let file = cycle_plan(&text, "/tmp/q2").render(Path::new("/tmp/q2"));
        let with = format!("--expert {path}");
        assert_eq!(
            file.matches(&with).count(),
            2,
            "collect and the gate: {file}"
        );
        assert_eq!(file.replace(&with, "--expert so101-pick-place"), named);
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

    /// Packet M12/R8. `base_model_fetch` is one `# fetch:` line -- the run's interpreter,
    /// `base_model`'s directory, the pin -- and nothing else in the plan moves; a cycle prints
    /// it once, above collect. Refused by name with no `base_model`, for an arch that has no
    /// pin, and when the script would write a file the run does not read.
    #[test]
    fn base_model_fetch_is_one_line_above_collect() {
        let named = IR.replace(
            "bundle = \"untrained.esb\"",
            "bundle = \"untrained.esb\"\n\
             base_model = \"target/backbone/resnet18-imagenet1k-v1.safetensors\"",
        );
        let fetching = named.replace("[run]", "base_model_fetch = \"resnet18\"\n[run]");
        let (_, before) = plan_of(&named, "/tmp/a");
        let (_, plan) = plan_of(&fetching, "/tmp/a");
        let line = format!(
            "# fetch: python python/es/fetch_backbone.py --arch resnet18 --out target/backbone \
             --expect {RESNET18_IMAGENET1K_V1_BLAKE3}\n"
        );
        assert_eq!(before.fetch, None);
        assert_eq!(plan.fetch_line().as_deref(), Some(line.as_str()));
        assert_eq!(plan.steps, before.steps);
        let out = Path::new("/tmp/a");
        assert_eq!(
            plan.render(out),
            before.render(out).replacen('\n', &format!("\n{line}"), 1)
        );

        let cycle = Cycle::parse(CYCLE).expect("parses");
        let run = Path::new("/tmp/run");
        let recipe = cycle.training(Some(&fetching), run).expect("resolves");
        let train = Plan::build(&recipe, &run.join("train"), "python", &[], None, false)
            .expect("plan builds");
        let rendered = CyclePlan::build(&cycle, &recipe, train, run)
            .expect("the cycle plan builds")
            .render(run);
        assert_eq!(rendered.matches("# fetch: ").count(), 1, "{rendered}");
        let (_, rest) = rendered.split_once('\n').expect("a header line");
        assert!(
            rest.starts_with(&format!("{line}es loop collect ")),
            "{rendered}"
        );

        for (text, word) in [
            (
                IR.replace("[run]", "base_model_fetch = \"resnet18\"\n[run]"),
                "names no file",
            ),
            (
                fetching.replace("= \"resnet18\"", "= \"resnet50\""),
                "\"resnet50\"",
            ),
            (
                fetching.replace("target/backbone/resnet18-imagenet1k-v1", "b/resnet18"),
                "resnet18-imagenet1k-v1.safetensors",
            ),
        ] {
            let e = Recipe::parse(&text).expect_err("refused");
            assert!(e.to_string().contains("base_model"), "{e}");
            assert!(e.to_string().contains(word), "{e}");
        }
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

    // --- the checkpoint preview (packet M13/Z1) ----------------------------------------------

    const EVALUATION: &str =
        include_str!("../../../tests/fixtures/visible-learning/evaluation.toml");

    fn demo_evaluation() -> EvaluationIr {
        es_ir::serial::evaluation_from_toml(EVALUATION).expect("the demo Evaluation IR parses")
    }

    /// The committed demo evaluation, previewed with the defaults: the nominal suite alone, its
    /// first four seeds, the same metrics and no acceptance -- a document that validates, round
    /// trips through the TOML written beside the preview, and hashes as other conditions.
    #[test]
    fn a_preview_is_the_first_suite_with_its_first_seeds_and_no_acceptance() {
        let ir = demo_evaluation();
        let derived = preview_evaluation(&ir, &PreviewRef::default()).expect("derives");
        assert_eq!(derived.suites, ir.suites[..1]);
        assert_eq!(derived.suites[0].name, "nominal");
        assert_eq!(derived.episodes.n_episodes, 4);
        assert_eq!(
            derived.episodes.seeds,
            SeedPlan::Explicit(vec![101, 102, 103, 104])
        );
        assert_eq!(derived.metrics, ir.metrics);
        assert!(derived.acceptance.is_empty(), "{:?}", derived.acceptance);
        assert_eq!(
            (&derived.task, &derived.observation, derived.replay),
            (&ir.task, &ir.observation, ir.replay)
        );
        assert!(derived.validate().is_empty(), "{:?}", derived.validate());
        assert_ne!(derived.evaluation_hash(), ir.evaluation_hash());
        let text = es_ir::serial::evaluation_to_toml(&derived).expect("serialises");
        assert_eq!(
            es_ir::serial::evaluation_from_toml(&text).expect("parses back"),
            derived
        );
    }

    /// A named suite is that suite; more episodes than the evaluation has are its episodes; a
    /// `seed_base` plan keeps its base; a suite the document does not declare is refused by
    /// name.
    #[test]
    fn a_preview_names_its_suite_and_takes_no_seed_the_evaluation_lacks() {
        let ir = demo_evaluation();
        let named = PreviewRef {
            episodes: 40,
            suite: Some("torque_noise".to_owned()),
            frames: false,
        };
        let derived = preview_evaluation(&ir, &named).expect("derives");
        assert_eq!(derived.suites.len(), 1);
        assert_eq!(derived.suites[0], ir.suites[4]);
        assert_eq!(derived.episodes, ir.episodes);

        let mut based = ir.clone();
        based.episodes.seeds = SeedPlan::Base(7);
        let derived = preview_evaluation(&based, &PreviewRef::default()).expect("derives");
        assert_eq!(derived.episodes.seeds, SeedPlan::Base(7));
        assert_eq!(derived.episodes.n_episodes, 4);

        let unknown = PreviewRef {
            suite: Some("fog".to_owned()),
            ..PreviewRef::default()
        };
        let e = preview_evaluation(&ir, &unknown).expect_err("refused");
        assert!(e.to_string().contains("\"fog\""), "{e}");
        assert!(e.to_string().contains("light_intensity"), "{e}");
    }

    /// `[eval.preview]` is optional: absent, the plan has no preview and renders as before;
    /// present and empty, it is four framed episodes of the first suite after every mark.
    #[test]
    fn eval_preview_is_optional_and_previews_every_mark() {
        let cycle = Cycle::parse(CYCLE).expect("parses");
        assert_eq!(cycle.eval.preview, None);
        let before = cycle_plan(CYCLE, "/tmp/run");
        assert!(before.previews.is_empty());
        assert!(!before.render(Path::new("/tmp/run")).contains("preview"));

        let text = format!("{CYCLE}[eval.preview]\n");
        let cycle = Cycle::parse(&text).expect("parses");
        assert_eq!(cycle.eval.preview, Some(PreviewRef::default()));
        assert_eq!(
            PreviewRef::default(),
            PreviewRef {
                episodes: 4,
                suite: None,
                frames: true
            }
        );
        let plan = cycle_plan(&text, "/tmp/run");
        let marks: Vec<u32> = plan.previews.iter().map(|p| p.mark).collect();
        assert_eq!(marks, plan.train.marks);
        let rendered = plan.render(Path::new("/tmp/run"));
        assert!(
            rendered.contains(
                "\n  es eval run --config preview/5000/evaluation.toml --policy \
                 train/checkpoints/5000.esb --scene scene.xml --out preview/5000 --jobs 1 \
                 --frames preview/5000/frames\n"
            ),
            "{rendered}"
        );
        // Everything else is the plan of before, line for line.
        let without: Vec<&str> = rendered
            .lines()
            .filter(|l| !l.contains("preview"))
            .collect();
        let old = before.render(Path::new("/tmp/run"));
        assert_eq!(without, old.lines().collect::<Vec<_>>());

        let e =
            Cycle::parse(&format!("{CYCLE}[eval.preview]\nepisodes = 0\n")).expect_err("refused");
        assert!(e.to_string().contains("episodes"), "{e}");
    }

    // --- the "again" cycle (packet M13/Z3) ---------------------------------------------------

    /// `CYCLE` going again: perturbed, merged with an earlier root, started from its checkpoint.
    fn again() -> String {
        CYCLE
            .replace(
                "frames = true\n[train]",
                "frames = true\nperturb = { config = \"evaluation.toml\", suites = \
                 [\"light_intensity\", \"torque_noise\"] }\nmerge = [\"runs/001/collect/ds\"]\n\
                 [train]",
            )
            .replace(
                "recipe = \"training.toml\"",
                "recipe = \"training.toml\"\ninit = \"runs/001/train/checkpoints/20000.esb\"",
            )
    }

    /// Z2's flags on the collect line, a merge stage into `collect/merged` with the frames beside
    /// each root, the merged root and its frames as what trains, and `[init] policy` on the IR
    /// route -- said under the train line.
    #[test]
    fn an_again_cycle_perturbs_merges_and_starts_from_init() {
        let text = again();
        let cycle = Cycle::parse(&text).expect("parses");
        let recipe = cycle.training(Some(IR), Path::new("/tmp/run")).unwrap();
        let dataset = recipe.dataset.as_ref().unwrap();
        assert!(dataset.root.ends_with("collect/merged"), "{recipe:?}");
        assert!(dataset
            .frames
            .as_deref()
            .unwrap()
            .ends_with("collect/frames"));
        assert_eq!(
            recipe.init,
            Some(InitRef {
                policy: s("runs/001/train/checkpoints/20000.esb")
            })
        );

        let plan = cycle_plan(&text, "/tmp/run");
        assert!(plan.dataset_root.ends_with("collect/merged"));
        let stages: Vec<Stage> = plan.steps.iter().map(|s| s.stage).collect();
        assert_eq!(
            stages[..3],
            [Stage::Collect, Stage::Merge, Stage::ExpertGate]
        );
        let rendered = plan.render(Path::new("/tmp/run"));
        for line in [
            "# cycle: collect -> merge -> expert-gate -> train -> eval -> showcase\n",
            " --out collect/ds --frames collect/frames --expert so101-pick-place --perturb \
             evaluation.toml --suites light_intensity,torque_noise\n",
            "\nes loop distill --in collect/ds --in-frames collect/frames --in \
             runs/001/collect/ds --in-frames runs/001/collect/frames --train 1 --val 0 --test 0 \
             --out collect/merged --frames collect/frames\n",
            "--frames collect/frames collect/merged\n",
            "--init-weights train/weights/init.safetensors",
            "\n  # init: from runs/001/train/checkpoints/20000.esb\n",
        ] {
            assert!(rendered.contains(line), "{line:?} not in\n{rendered}");
        }

        // Without frames there are no tiles to merge.
        let bare = text.replace("frames = true\nperturb", "perturb");
        let plan = cycle_plan(&bare, "/tmp/run");
        assert_eq!(
            plan.steps[1].args.join(" ").replace('\\', "/"),
            "--in /tmp/run/collect/ds --in runs/001/collect/ds --train 1 --val 0 --test 0 \
             --out /tmp/run/collect/merged"
        );
    }

    /// Packet M16/H3: `success_only` is a merge stage of the collection alone with
    /// `--success-only`, into `collect/successes/ds` with its tiles in
    /// `collect/successes/frames` -- which is what trains. Without it a cycle that merges
    /// nothing has no merge stage, as before.
    #[test]
    fn a_success_only_cycle_trains_on_the_kept_episodes() {
        let text = CYCLE.replace(
            "frames = true\n[train]",
            "frames = true\nsuccess_only = true\n[train]",
        );
        let cycle = Cycle::parse(&text).expect("parses");
        let recipe = cycle.training(Some(IR), Path::new("/tmp/run")).unwrap();
        let dataset = recipe.dataset.as_ref().unwrap();
        assert!(dataset
            .root
            .replace('\\', "/")
            .ends_with("collect/successes/ds"));
        let frames = dataset.frames.as_deref().unwrap().replace('\\', "/");
        assert!(frames.ends_with("collect/successes/frames"), "{frames}");
        let plan = cycle_plan(&text, "/tmp/run");
        assert_eq!(plan.steps[1].stage, Stage::Merge);
        let rendered = plan.render(Path::new("/tmp/run"));
        let line =
            "\nes loop distill --in collect/ds --in-frames collect/frames --train 1 --val 0 \
                    --test 0 --out collect/successes/ds --frames collect/successes/frames \
                    --success-only\n";
        assert!(rendered.contains(line), "{line:?} not in\n{rendered}");
        let plain = cycle_plan(CYCLE, "/tmp/run");
        assert!(plain.steps.iter().all(|s| s.stage != Stage::Merge));
    }

    /// The lerobot route starts from `--policy.path`, in place of `--policy.type`, which
    /// `lerobot-train` refuses beside it (`lerobot/configs/parser.py`, 0.6.1).
    #[test]
    fn init_is_the_lerobot_routes_policy_path() {
        let cycle = Cycle::parse(&again()).expect("parses");
        let out = Path::new("/tmp/run");
        let recipe = cycle.training(Some(EXTERNAL), out).unwrap();
        assert_eq!(recipe.init, None);
        let trainer = vec![s("lerobot-train")];
        let plan = Plan::build(
            &recipe,
            &out.join("train"),
            "python",
            &trainer,
            Some(6),
            false,
        )
        .unwrap();
        let rendered = plan.render(&out.join("train"));
        assert!(
            rendered.contains("--policy.path=runs/001/train/checkpoints/20000.esb --policy.chunk"),
            "{rendered}"
        );
        assert!(!rendered.contains("--policy.type"), "{rendered}");
    }

    /// Refused by name: a `[collect]` that merges and collects nothing, a perturbation with no
    /// suite, and an `init` beside a recipe that already starts from something.
    #[test]
    fn an_again_cycle_refuses_what_it_cannot_do() {
        let block = CYCLE
            .split_once("[collect]\n")
            .and_then(|(_, rest)| rest.split_once("[train]"))
            .map(|(block, _)| block.to_owned())
            .expect("the fixture has a [collect] block");
        let merge_only = CYCLE.replace(
            &format!("[collect]\n{block}"),
            "dataset = \"runs/ds\"\n[collect]\nmerge = [\"runs/001/collect/ds\"]\n",
        );
        let e = Cycle::parse(&merge_only).expect_err("refused");
        assert!(e.to_string().contains("`merge`"), "{e}");
        let none = again().replace("episodes = 200", "episodes = 0");
        let e = Cycle::parse(&none).expect_err("refused");
        assert!(e.to_string().contains("`merge`"), "{e}");
        let e = Cycle::parse(&CYCLE.replace("episodes = 200\n", "")).expect_err("refused");
        assert!(e.to_string().contains("episodes"), "{e}");
        let empty = again().replace("[\"light_intensity\", \"torque_noise\"]", "[]");
        let e = Cycle::parse(&empty).expect_err("refused");
        assert!(e.to_string().contains("suites"), "{e}");

        let cycle = Cycle::parse(&again()).expect("parses");
        let with_init = format!("{IR}\n[init]\npolicy = \"other.esb\"\n");
        let e = cycle
            .training(Some(&with_init), Path::new("/tmp/run"))
            .expect_err("refused");
        assert!(e.to_string().contains("not both"), "{e}");
    }

    /// `[train] init` and `[collect] merge` have to be on disk before a real run starts: a bundle
    /// on the IR route, a `pretrained_model` directory on the lerobot route, a dataset root.
    #[test]
    fn a_real_run_refuses_an_init_or_a_merge_root_that_is_not_there() {
        let dir = std::env::temp_dir().join(format!("es-data-again-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bundle = dir.join("20000.esb");
        let at = |init: &Path, merge: &[&str]| {
            let mut cycle = Cycle::parse(&again()).unwrap();
            cycle.train.init = Some(init.to_string_lossy().into_owned());
            cycle.collect.as_mut().unwrap().merge = merge.iter().map(|m| s(*m)).collect();
            cycle
        };

        let e = at(&bundle, &[])
            .check_inputs(Route::Ir)
            .expect_err("no bundle");
        assert!(e.to_string().contains("20000.esb"), "{e}");
        std::fs::write(&bundle, b"esb").unwrap();
        at(&bundle, &[])
            .check_inputs(Route::Ir)
            .expect("a file is there");
        at(&bundle, &[])
            .check_inputs(Route::Rl)
            .expect("a file is there");

        let model = dir.join("pretrained_model");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("config.json"), b"{}").unwrap();
        let e = at(&model, &[])
            .check_inputs(Route::External)
            .expect_err("half a checkpoint");
        assert!(e.to_string().contains("pretrained_model"), "{e}");
        std::fs::write(model.join("model.safetensors"), b"st").unwrap();
        at(&model, &[])
            .check_inputs(Route::External)
            .expect("a whole checkpoint");
        let e = at(&bundle, &[])
            .check_inputs(Route::External)
            .expect_err("a file, not a dir");
        assert!(e.to_string().contains("lerobot"), "{e}");

        let missing = dir.join("runs/001/collect/ds");
        let e = at(&bundle, &[&missing.to_string_lossy()])
            .check_inputs(Route::Ir)
            .expect_err("no root");
        assert!(e.to_string().contains("`merge`"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
