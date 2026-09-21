//! Packet M8/S1 — `[init] policy` and `training/init.lock`.
//!
//! The pin comes first. `tests/fixtures/visible-learning/training.toml` names no `[init]`,
//! and the two digests below are what it hashed to *before* this packet existed: a slot that
//! is absent has to leave every recipe written before it exactly where it was (spec 28.10
//! rule 2). Everything else in this file is about the slot being real when it is present.

use std::path::Path;

use es_data::training::{DatasetFacts, Plan, Recipe, Training};
use es_ir::DatasetHash;
use serde_json::{json, Value};

/// The committed IR-route recipe, read from the repository root.
fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Facts a dataset would report, fixed so the digests below are a function of the recipe
/// alone. They are not the fixture dataset's — no dataset is opened here — they are a
/// constant this file owns, exactly as `training.rs`'s own unit tests use one.
fn facts() -> DatasetFacts {
    DatasetFacts {
        hashes: DatasetHash {
            content: [1; 32],
            schema: [2; 32],
            split: [3; 32],
        },
        episodes: 12,
        frames: 3600,
        recorded_task: None,
        split_source: "all-train",
    }
}

fn hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    digest.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// The spec 19.3 bundle of a recipe, with `lock` set as its `init.lock` when there is one.
fn bundle(text: &str, lock: Option<&Value>) -> Training {
    let recipe = Recipe::parse(text).expect("the recipe parses");
    let out = Path::new("out");
    let plan = Plan::build(&recipe, out, "python", &[], Some(13), false).expect("the plan builds");
    let mut training = Training::pre_run(&recipe, &plan, out, "python", &facts(), None, None)
        .expect("the pre-run slots build");
    if let Some(lock) = lock {
        training.set_init(lock).expect("init.lock is set");
    }
    training
}

/// `(identity_hash, training_hash)` of a recipe, with the three post-run slots filled from
/// fixed values so the second digest is a function of the recipe too.
fn hashes(text: &str, lock: Option<&Value>) -> (String, String) {
    let mut training = bundle(text, lock);
    let identity = hex(&training.hash().expect("identity_hash"));
    training.finish(
        &json!({"schema_version": 1, "checkpoints": []}),
        &json!({"loss": []}),
        &json!({"torch": "2.9.0"}),
    );
    (identity, hex(&training.hash().expect("training_hash")))
}

/// An `init.lock` as `es_data::training::init_from` writes one, with `copied` replaced so a
/// test can move the lock's *content* without moving the recipe.
fn lock_of(copied: &[&str]) -> Value {
    json!({
        "schema_version": 1,
        "source": "runs/train-001/checkpoints/20000.esb",
        "policy_hash": "1".repeat(64),
        "learning_hash": "2".repeat(64),
        "copied": copied,
        "initialised": ["nodes.0.*"],
        "shape_mismatch": [],
    })
}

/// **The pin.** `tests/fixtures/visible-learning/training.toml` without `[init]`, measured
/// before packet M8/S1 changed a line. If either of these moves, a recipe that predates the
/// packet has a new identity and every measured run in `docs/design/training-recipe.md`
/// section 7 is describing a run nobody can name any more.
const FIXTURE_IDENTITY: &str = "a7531245795d713cc520c6b93a3a8c851ce9a4f5fb7512265c5c072d98ff75ac";
const FIXTURE_TRAINING: &str = "577a3f4a141e27cf65014461912c271909b87a50ae391af2db6edd237679c01d";

#[test]
fn init_lock_enters_training_hash_and_is_absent_without_init() {
    let base = fixture("training.toml");
    let (identity, training) = hashes(&base, None);
    assert_eq!(
        identity, FIXTURE_IDENTITY,
        "the fixture recipe's identity_hash moved"
    );
    assert_eq!(
        training, FIXTURE_TRAINING,
        "the fixture recipe's training_hash moved"
    );

    // Absent means absent: no thirteenth file, and nothing about it in `training.lock`.
    let plain = bundle(&base, None);
    assert!(!plain.digests().contains_key("init.lock"));
    let dir = std::env::temp_dir().join("es-s1-training-init");
    let _ = std::fs::remove_dir_all(&dir);
    plain.write(&dir.join("plain")).expect("write the bundle");
    assert!(!dir.join("plain").join("init.lock").exists());

    // Present, it is a slot: the same twelve files, one more digest, and two digests that are
    // no longer the ones above.
    let lock = lock_of(&["nodes.2.weight", "nodes.4.weight"]);
    let (with_identity, with_training) = hashes(&base, Some(&lock));
    assert_ne!(
        with_identity, FIXTURE_IDENTITY,
        "an init.lock that changed nothing is not a slot"
    );
    assert_ne!(with_training, FIXTURE_TRAINING);
    let started = bundle(&base, Some(&lock));
    assert_eq!(
        started.file("init.lock"),
        lock.to_string() + "\n",
        "init.lock is not the canonical JSON of what it was given"
    );
    assert_eq!(
        started.digests()["init.lock"],
        hex(blake3::hash(started.file("init.lock").as_bytes()).as_bytes())
    );
    started.write(&dir.join("started")).expect("write");
    assert!(dir.join("started").join("init.lock").exists());

    // The *content* of the lock, not merely the recipe's path to a bundle: two runs that
    // named one bundle and copied different tensors out of it are two runs.
    let (other_identity, _) = hashes(&base, Some(&lock_of(&["nodes.2.weight"])));
    assert_ne!(
        with_identity, other_identity,
        "the lock's content does not reach identity_hash"
    );

    // And nothing else moved with it: the eleven slots that are not `config.json` are the
    // bytes they were, so the lock is one addition and not a re-shuffle.
    for name in es_data::training::FILES {
        if name == "config.json" {
            continue;
        }
        assert_eq!(
            plain.file(name),
            started.file(name),
            "{name} moved with [init]"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The two refusals and the one new legality the table brings with it (spec 17.2: a refusal
/// names what caused it).
#[test]
fn init_is_the_ir_routes_and_zero_steps_is_one_mark() {
    let external = fixture("training-lerobot.toml");
    let with_init = external.replace(
        "[run]\n",
        "[init]\npolicy = \"runs/train-001/checkpoints/20000.esb\"\n[run]\n",
    );
    let said = Recipe::parse(&with_init)
        .expect_err("[init] on the lerobot route")
        .to_string();
    assert!(
        said.contains("[init]") && said.contains("IR route"),
        "{said}"
    );

    let zero = external
        .replace("steps         = 20000", "steps         = 0")
        .replace("checkpoint_at = [10000, 20000]\n", "");
    assert!(
        zero.contains("steps         = 0"),
        "the fixture's steps moved"
    );
    let said = Recipe::parse(&zero)
        .expect_err("steps = 0 on the lerobot route")
        .to_string();
    assert!(said.contains("save_freq"), "{said}");

    // On the IR route it is legal, and its one mark is 0.
    let recipe = Recipe::parse(&fixture("training-init.toml")).expect("training-init.toml parses");
    assert_eq!(recipe.marks().expect("marks"), [0]);
    assert_eq!(
        recipe.init.as_ref().expect("[init]").policy,
        "runs/train-001/checkpoints/20000.esb"
    );

    // `checkpoint_at` beside it is a refusal: there is no optimizer step to also stop at.
    let both = fixture("training-init.toml")
        .replace("steps       = 0", "steps       = 0\ncheckpoint_at = [1]");
    let said = Recipe::parse(&both)
        .expect_err("steps = 0 with checkpoint_at")
        .to_string();
    assert!(said.contains("checkpoint_at"), "{said}");
}
