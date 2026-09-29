//! Packet M15/N7 — `[run] single_view = { weight = alpha }`, Python-free.
//!
//! The field reaches `train_act.py` as `--single-view <alpha>` and nothing else changes; absent,
//! the plan is the plan of before (the committed `tests/golden/train/` plans are judged by
//! `crates/es/tests/cli.rs`). It is refused by name at recipe check time -- `Recipe::parse`,
//! `Recipe::check_single_view` (what `es train` calls before its plan, `--dry-run` included)
//! and `Cycle::training` -- for a weight that is not above 0, off the IR route, and for a
//! bundle whose Learning IR has no `Sum` fed by `VisionEncoder`s.

use std::path::{Path, PathBuf};

use es_compile::PolicyBundle;
use es_data::training::{check_single_view, Cycle, Plan, Recipe};
use es_ir::learning::{FusionKind, LearningGraph, LearningNode, WeightsRef};

const IR: &str = r#"
kind = "training"
[dataset]
root = "runs/collect-001/ds"
frames = "runs/collect-001/frames"
[policy]
bundle = "untrained.esb"
[run]
steps = 200
batch = 8
lr = 1e-4
seed = 0
device = "cuda"
"#;

fn with_single_view(recipe: &str, weight: &str) -> String {
    recipe.replace(
        "device = \"cuda\"",
        &format!("device = \"cuda\"\nsingle_view = {{ weight = {weight} }}"),
    )
}

fn plan_line(text: &str) -> String {
    let recipe = Recipe::parse(text).expect("the recipe parses");
    Plan::build(&recipe, Path::new("/tmp/o"), "python", &[], None, false)
        .expect("the plan builds")
        .render(Path::new("/tmp/o"))
}

#[test]
fn single_view_reaches_the_trainer_line_and_absent_is_absent() {
    let plain = plan_line(IR);
    assert!(!plain.contains("single-view"), "{plain}");
    let asked = plan_line(&with_single_view(IR, "0.5"));
    let trainer = asked
        .lines()
        .find(|l| l.contains("train_act.py"))
        .expect("a trainer line");
    assert!(trainer.contains(" --single-view 0.5"), "{trainer}");
    // The only difference between the two plans is that flag.
    assert_eq!(asked.replace(" --single-view 0.5", ""), plain);
    println!("RAN single_view_reaches_the_trainer_line: {trainer}");
}

#[test]
fn a_weight_not_above_zero_is_refused_by_name() {
    for weight in ["0.0", "-0.5", "nan", "inf"] {
        let e = Recipe::parse(&with_single_view(IR, weight)).expect_err(weight);
        assert!(
            e.to_string().contains("`single_view.weight`"),
            "{weight}: {e}"
        );
    }
}

#[test]
fn single_view_off_the_ir_route_is_refused_by_name() {
    let rl = r#"
kind = "training"
[policy]
bundle = "untrained.esb"
[rl]
algo = "ppo"
envs = 4
horizon = 8
epochs = 1
minibatches = 1
gamma = 0.99
lam = 0.95
clip = 0.2
entropy = 0.0
value_coef = 0.5
[run]
steps = 10
lr = 3e-4
seed = 0
device = "cuda"
"#;
    let e = Recipe::parse(&with_single_view(rl, "0.5")).expect_err("rl");
    assert!(
        e.to_string().contains("`single_view`") && e.to_string().contains("train_ppo.py"),
        "{e}"
    );
    let lerobot = IR.replace(
        "bundle = \"untrained.esb\"",
        "task = \"t.toml\"\nobservation = \"o.toml\"\ndeployment = \"d.toml\"\n\
         lerobot = { type = \"act\", chunk_size = 16, n_action_steps = 16 }",
    );
    Recipe::parse(&lerobot).expect("the lerobot recipe parses without the field");
    let e = Recipe::parse(&with_single_view(&lerobot, "0.5")).expect_err("lerobot");
    assert!(
        e.to_string().contains("`single_view`") && e.to_string().contains("lerobot-train"),
        "{e}"
    );
}

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn demo_learning() -> LearningGraph {
    es_ir::serial::learning_from_toml(&fixture("learning.toml")).expect("learning.toml")
}

/// The committed demo graph (one camera, `Concat`) is refused; the same graph with its fusion
/// made a `Sum` -- every term is 512 wide, so it validates -- is accepted: the camera's term is
/// one a `Sum` can drop.
#[test]
fn a_graph_without_a_camera_sum_is_refused_by_name() {
    let concat = demo_learning();
    let e = check_single_view(&concat).expect_err("a Concat of views is refused");
    assert!(
        e.to_string().contains("`single_view`") && e.to_string().contains("no `Sum`"),
        "{e}"
    );

    let mut sum = concat;
    for node in sum.nodes.nodes.values_mut() {
        if let LearningNode::Fusion { kind, .. } = node {
            *kind = FusionKind::Sum;
        }
    }
    let errors: Vec<String> = sum
        .validate()
        .into_iter()
        .filter(es_ir::Diagnostic::is_error)
        .map(|d| d.to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    check_single_view(&sum).expect("a Sum fed by a VisionEncoder has a camera term to drop");
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-single-view-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

/// The demo's four documents in a bundle on disk, as `es train` would open it.
fn demo_bundle(dir: &Path) -> PathBuf {
    let mut learning = demo_learning();
    let weights = b"es-n7-placeholder".to_vec();
    learning.policy.weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: *blake3::hash(&weights).as_bytes(),
    };
    let bytes = PolicyBundle::build(
        &es_ir::serial::task_from_toml(&fixture("task.toml")).expect("task.toml"),
        &es_ir::serial::observation_from_toml(&fixture("observation.toml"))
            .expect("observation.toml"),
        &learning,
        &es_ir::serial::deployment_from_toml(&fixture("deployment.toml")).expect("deployment.toml"),
        &weights,
    )
    .expect("the demo documents pack");
    let path = dir.join("untrained.esb");
    std::fs::write(&path, bytes).expect("write the bundle");
    path
}

/// What `es train` (before its plan, `--dry-run` included) and `Cycle::training` run: the
/// bundle is opened only when the field is set, and a `Concat` graph is refused by name.
#[test]
fn the_recipe_and_the_cycle_check_the_bundles_graph() {
    // Absent: nothing is read, so a bundle that is not on disk is no refusal.
    Recipe::parse(IR)
        .expect("parses")
        .check_single_view()
        .expect("no field, no check");

    let dir = scratch("check");
    let bundle = demo_bundle(&dir);
    let bundle_arg = bundle.to_string_lossy().replace('\\', "/");
    let recipe = with_single_view(IR, "0.5").replace("untrained.esb", &bundle_arg);
    let e = Recipe::parse(&recipe)
        .expect("the text alone is fine")
        .check_single_view()
        .expect_err("the demo graph has no camera Sum");
    assert!(e.to_string().contains("no `Sum`"), "{e}");

    let missing = with_single_view(IR, "0.5").replace("untrained.esb", "nowhere/x.esb");
    let e = Recipe::parse(&missing)
        .expect("parses")
        .check_single_view()
        .expect_err("the bundle is needed to judge the field");
    assert!(e.to_string().contains("cannot be read"), "{e}");

    let cycle = format!(
        r#"
kind = "cycle"
scene = "scene.xml"
dataset = "runs/ds"
[train.policy]
bundle = "{bundle_arg}"
[train.run]
steps = 10
batch = 2
lr = 1e-4
seed = 0
device = "cpu"
single_view = {{ weight = 0.5 }}
[eval]
config = "evaluation.toml"
"#
    );
    let parsed = Cycle::parse(&cycle).expect("the cycle parses");
    let e = parsed
        .training(None, Path::new("/tmp/o"))
        .expect_err("refused before anything runs");
    assert!(e.to_string().contains("no `Sum`"), "{e}");
    let e = Cycle::parse(&cycle.replace("weight = 0.5", "weight = 0.0"))
        .expect("parses")
        .training(None, Path::new("/tmp/o"))
        .expect_err("an inline run is checked too");
    assert!(e.to_string().contains("`single_view.weight`"), "{e}");
    println!("RAN the_recipe_and_the_cycle_check_the_bundles_graph");
    let _ = std::fs::remove_dir_all(&dir);
}
