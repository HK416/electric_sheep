//! The training recipes and the cycle (`es_data::training::{Recipe, Cycle}`): presets with the
//! committed runs' values, the paths a project's runs take, then the specification's override
//! tables merged on top (any field of a `training.toml`; an unknown one is refused by name).

use es_data::training::{CollectRef, Cycle, EvalRef, PreviewRef, Recipe, TrainRef, CYCLE_KIND};

use super::project::{CycleDoc, Family, Student, Teacher};
use super::{refuse, SpecError, TaskSpec};

/// Plan H's teacher run (`training-teacher-v2.toml`): `rl_games`' Shadow Hand PPO with no entropy
/// bonus, an initial sigma of e^-1.6 ≈ 0.2 in actuator units, lr decaying to 1e-5 over 3,000
/// iterations on `MJWarp`, a checkpoint every 250.
const PPO: &str = r#"
kind = "training"
[policy]
[rl]
algo = "ppo"
envs = 2048
horizon = 16
epochs = 5
minibatches = 4
gamma = 0.99
lam = 0.95
clip = 0.2
entropy = 0.0
value_coef = 0.5
init_log_std = -1.6
backend = "mjwarp"
[run]
steps = 3000
lr = 5e-4
schedule = { kind = "warmup_cosine", warmup = 50, lr_min = 1e-5 }
seed = 0
checkpoint_at = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750]
device = "cuda"
interpreter = "python"
"#;

/// Plan U's U3 run as plan N's arms train it (`training-views.toml`): batch 64, lr 4e-4,
/// `warmup_cosine` 250 / 1e-6, the `ImageNet` `ResNet18` fetched, 20,000 steps.
const ACT: &str = r#"
kind = "training"
[dataset]
[policy]
base_model = "target/backbone/resnet18-imagenet1k-v1.safetensors"
base_model_fetch = "resnet18"
[run]
steps = 20000
batch = 64
lr = 4e-4
seed = 0
checkpoint_at = [1000, 5000, 20000]
device = "cuda"
interpreter = "python"
schedule = { kind = "warmup_cosine", warmup = 250, lr_min = 1e-6 }
"#;

/// Plan N's MAD run over `ACT` (`training-mad.toml`): three times the steps and MAD's
/// single-view loss at weight 0.5.
const MAD: &str = r"
[run]
steps = 60000
checkpoint_at = [1000, 5000, 20000, 60000]
single_view = { weight = 0.5 }
";

/// `over` merged into `base`: tables key by key, anything else replaced.
fn merge(base: &mut toml::Table, over: &toml::Table) {
    for (k, v) in over {
        match (base.get_mut(k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge(b, o),
            _ => {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

fn table(text: &str) -> toml::Table {
    text.parse().expect("the presets are TOML")
}

/// `[section] key = value`.
fn set(t: &mut toml::Table, section: &str, key: &str, value: String) {
    if let Some(toml::Value::Table(s)) = t.get_mut(section) {
        s.insert(key.to_owned(), toml::Value::String(value));
    }
}

fn recipe(at: &str, mut doc: toml::Table, over: Option<&toml::Table>) -> Result<Recipe, SpecError> {
    if let Some(over) = over {
        merge(&mut doc, over);
    }
    toml::Value::Table(doc)
        .try_into()
        .or_else(|e: toml::de::Error| refuse(at, "training", e.to_string()))
}

/// The teacher's PPO recipe; its bundle is `<runs>/teacher-untrained.esb`.
pub(super) fn teacher(t: &Teacher, runs: &str) -> Result<Recipe, SpecError> {
    let mut doc = table(PPO);
    set(
        &mut doc,
        "policy",
        "bundle",
        format!("{runs}/teacher-untrained.esb"),
    );
    recipe("teacher", doc, t.training.as_ref())
}

/// The student's imitation recipe: its bundle `<runs>/<arm>-untrained.esb`, its dataset what
/// the first cycle (`--out <runs>/<arm>-001`) collects — the kept successes with
/// `success_only`.
pub(super) fn student(
    s: &Student,
    arm: &str,
    runs: &str,
    success_only: bool,
) -> Result<Recipe, SpecError> {
    let mut doc = table(ACT);
    if s.family.unwrap_or_default() == Family::Mad {
        merge(&mut doc, &table(MAD));
    }
    let data = format!(
        "{runs}/{arm}-001/collect{}",
        if success_only { "/successes" } else { "" }
    );
    set(&mut doc, "dataset", "root", format!("{data}/ds"));
    set(&mut doc, "dataset", "frames", format!("{data}/frames"));
    set(
        &mut doc,
        "policy",
        "bundle",
        format!("{runs}/{arm}-untrained.esb"),
    );
    recipe("student", doc, s.training.as_ref())
}

/// The cycle: demonstrate (the expert, or the trained teacher `<runs>/teacher.esb`), keep the
/// successes if asked, train `training-<arm>.toml`, evaluate on `evaluation-<arm>.toml` (or its
/// nominal sibling) with a preview of each checkpoint. An expert's collection is recorded
/// under the student's own bundle, the Task IR it carries.
pub(super) fn cycle(
    spec: &TaskSpec,
    c: &CycleDoc,
    arm: &str,
    bundle: &str,
    runs: &str,
    path: &dyn Fn(&str) -> String,
) -> Result<Cycle, SpecError> {
    if c.expert.is_none() && spec.teacher.is_none() {
        return refuse(
            "cycle",
            "expert",
            "required without a `[teacher]`: who demonstrates?",
        );
    }
    let policy = match &c.expert {
        Some(_) => bundle.to_owned(),
        None => format!("{runs}/teacher.esb"),
    };
    let config = if c.nominal_only == Some(true) {
        format!("evaluation-{arm}-nominal.toml")
    } else {
        format!("evaluation-{arm}.toml")
    };
    Ok(Cycle {
        kind: CYCLE_KIND.to_owned(),
        scene: spec.scene.clone(),
        dataset: None,
        collect: Some(CollectRef {
            policy,
            expert: c.expert.clone(),
            episodes: c.episodes,
            seed: c.seed,
            frames: true,
            perturb: None,
            merge: Vec::new(),
            success_only: c.success_only.unwrap_or(false),
        }),
        train: TrainRef {
            recipe: Some(path(&format!("training-{arm}.toml"))),
            dataset: None,
            policy: None,
            run: None,
            init: None,
        },
        eval: EvalRef {
            config: path(&config),
            checkpoint: "last".to_owned(),
            jobs: c.jobs.unwrap_or(1),
            frames: true,
            preview: (c.preview != Some(false)).then(PreviewRef::default),
        },
        showcase: c.showcase.clone(),
    })
}
