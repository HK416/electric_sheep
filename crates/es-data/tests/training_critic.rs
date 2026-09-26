//! Packet M11/R10 oracle 1 — `[rl] critic` and `[policy] base_model` on the `[rl]` route.
//!
//! The pin comes first. `tests/fixtures/rl/training-reach-vision-rs-pix.toml` names no
//! `critic`, and the digest below is what it hashed to *before* this packet existed: a field that
//! is absent, or spelled out at its default, has to leave every recipe written before it exactly
//! where it was (spec 28.10 rule 2), as `[rl] estimator` did (packet M9/R5).

use std::path::Path;

use es_data::training::{
    Backbone, DatasetFacts, Plan, Recipe, Training, BASE_MODEL_SOURCE,
    RESNET18_IMAGENET1K_V1_BLAKE3,
};
use es_ir::DatasetHash;

/// `identity_hash` of the committed X7 `rs-pix` recipe, computed before this packet.
const RS_PIX_IDENTITY: &str = "7b3e80191f7509cac978e0340777454cc2cce9e226b2ed69fea7fedee8c2ce75";

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rl")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n")
}

/// An `[rl]` run opens no dataset; the facts are a constant so the digest is the recipe's.
fn facts() -> DatasetFacts {
    DatasetFacts {
        hashes: DatasetHash {
            content: [0; 32],
            schema: [0; 32],
            split: [0; 32],
        },
        episodes: 0,
        frames: 0,
        recorded_task: None,
        split_source: "unset",
    }
}

fn hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    digest.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// The recipe's plan (rendered) and its pre-run `training/` bundle.
fn pre_run(text: &str, backbone: Option<&Backbone>) -> (String, Training) {
    let recipe = Recipe::parse(text).expect("the recipe parses");
    let out = Path::new("out");
    let plan = Plan::build(&recipe, out, "python", &[], None, false).expect("the plan builds");
    let training = Training::pre_run(&recipe, &plan, out, "python", &facts(), backbone, None)
        .expect("the pre-run slots build");
    (plan.render(out), training)
}

#[test]
fn a_recipe_without_critic_is_the_recipe_of_before() {
    let text = fixture("training-reach-vision-rs-pix.toml");
    let (plan, training) = pre_run(&text, None);
    assert_eq!(
        hex(&training.hash().expect("identity_hash")),
        RS_PIX_IDENTITY,
        "a recipe that names no critic moved its identity_hash"
    );
    assert!(!plan.contains("--critic"), "{plan}");
    assert!(!training.file("config.json").contains("critic"));

    // Spelled out at the default, it serialises like absence.
    let spelled = text.replace(
        "value_coef  = 0.5",
        "value_coef  = 0.5\ncritic      = \"observation\"",
    );
    assert_ne!(spelled, text, "the anchor line moved");
    let (plan, training) = pre_run(&spelled, None);
    assert_eq!(hex(&training.hash().expect("hash")), RS_PIX_IDENTITY);
    assert!(!plan.contains("--critic"), "{plan}");
}

#[test]
fn a_privileged_critic_moves_training_hash_and_reaches_the_trainer() {
    let text = fixture("training-reach-vision-rs-pix-critic.toml");
    let (plan, training) = pre_run(&text, None);
    assert_ne!(
        hex(&training.hash().expect("identity_hash")),
        RS_PIX_IDENTITY,
        "the critic is not in training_hash: two different trainings claim one identity"
    );
    assert!(
        training
            .file("config.json")
            .contains("\"critic\":\"privileged\""),
        "{}",
        training.file("config.json")
    );
    let trainer = plan
        .lines()
        .find(|l| l.contains("train_ppo.py"))
        .expect("the trainer line");
    assert!(trainer.ends_with(" --critic privileged"), "{trainer}");

    // An unknown value is refused with the word the recipe used.
    let bad = text.replace("\"privileged\"", "\"banana\"");
    let said = Recipe::parse(&bad).expect_err("refused").to_string();
    assert!(said.contains("banana"), "{said}");
}

/// `[policy] base_model` on the `[rl]` route reaches `train_ppo.py` as `--init-backbone`, as it
/// reaches `train_act.py` on the IR route, and its verified lock fills `base_model.lock`.
#[test]
fn base_model_reaches_the_ppo_trainer() {
    let text = fixture("training-reach-vision-rs-pix-critic.toml");
    let (plain, none) = pre_run(&text, None);
    assert!(!plain.contains("--init-backbone"), "{plain}");
    assert_eq!(none.file("base_model.lock"), "{\"source\":\"none\"}\n");

    let named = text.replace(
        "bundle = \"runs/reach-vision-rs-pix/untrained.esb\"",
        "bundle = \"runs/reach-vision-rs-pix/untrained.esb\"\n\
         base_model = \"backbones/resnet18.safetensors\"",
    );
    assert_ne!(named, text, "the anchor line moved");
    let lock = Backbone {
        source: BASE_MODEL_SOURCE.to_owned(),
        blake3: RESNET18_IMAGENET1K_V1_BLAKE3.to_owned(),
        license: "BSD-3-Clause".to_owned(),
        ..Backbone::default()
    };
    let (plan, with) = pre_run(&named, Some(&lock));
    assert!(
        plan.contains("--init-backbone backbones/resnet18.safetensors"),
        "{plan}"
    );
    assert!(
        with.file("base_model.lock")
            .contains(RESNET18_IMAGENET1K_V1_BLAKE3),
        "{}",
        with.file("base_model.lock")
    );
    assert_ne!(none.hash().unwrap(), with.hash().unwrap());
}
