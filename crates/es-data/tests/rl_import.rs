//! Oracle 2 of packet M8/S2b: the five named refusals of `es_data::rl_import`.
//!
//! Spec 13.4's rule is that the import never guesses, and the falsifiable form of that is this
//! file: five ways an adapter can disagree with the Task IR it targets, five `IMP-0xx` codes,
//! and a sixth case that must still *pass* so the test cannot succeed by refusing everything.

use std::path::{Path, PathBuf};

use es_data::rl_import::{convert, Adapter, ImportManifest};
use es_ir::serial::{deployment_from_toml, task_from_toml};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rl")
        .join(name)
}

fn read(name: &str) -> String {
    std::fs::read_to_string(fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The six SO-101 actuator names, in the scene's declaration order -- what the CLI reads out of
/// the Task IR's own `scene.path` and hands to `convert`.
const ACTUATORS: [&str; 6] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
];

/// Runs the conversion with `adapter` in place of the committed one and returns what it said.
fn run(adapter_toml: &str) -> Result<String, String> {
    let manifest = ImportManifest::parse(&read("import/playground/import.json")).expect("manifest");
    let adapter = Adapter::parse(adapter_toml).map_err(|e| e.to_string())?;
    let task = task_from_toml(&read("task-reach.toml")).expect("task-reach.toml");
    let deployment =
        deployment_from_toml(&read("deployment-reach.toml")).expect("deployment-reach.toml");
    let weights = std::fs::read(fixture("import/playground/weights.safetensors")).expect("weights");
    let actuators: Vec<String> = ACTUATORS.iter().map(|s| (*s).to_owned()).collect();
    match convert(
        &manifest,
        &adapter,
        &task,
        &deployment,
        &weights,
        &actuators,
    ) {
        Ok(imported) => Ok(format!(
            "{} joints, {} channels",
            imported.report.joints.len(),
            imported.report.channels.len()
        )),
        Err(e) => Err(e.to_string()),
    }
}

fn refused(adapter_toml: &str, code: &str) {
    let said = run(adapter_toml).expect_err(&format!("{code} was not refused"));
    assert!(
        said.starts_with(code),
        "expected {code}, got:\n{said}\n(a refusal that does not name its code is a refusal \
         the caller cannot act on -- spec 17.2)"
    );
    assert!(
        es_ir::codes::lookup(code).is_some(),
        "{code} is not in the diagnostic dictionary"
    );
}

#[test]
fn import_rl_refusals() {
    let base = read("adapter-so101.toml");

    // The control: the committed adapter converts. Without this the five below could be
    // passing because *everything* is refused.
    assert_eq!(
        run(&base).expect("the committed adapter"),
        "6 joints, 4 channels"
    );

    // IMP-001 -- five joints for a six-dimensional ActionSpec.
    refused(&base.replace("    \"gripper\",\n", ""), "IMP-001");

    // IMP-002 -- a joint the scene has no actuator for.
    refused(
        &base.replace("\"elbow_flex\"", "\"elbow_flexx\""),
        "IMP-002",
    );

    // IMP-003 -- degrees. Converting them silently is the guess rule 3 forbids.
    refused(
        &base.replace("units = \"rad\"", "units = \"deg\""),
        "IMP-003",
    );

    // IMP-004 -- torques for a `joint_position` task.
    refused(
        &base.replace("kind = \"position_target\"", "kind = \"torque\""),
        "IMP-004",
    );

    // IMP-005, twice: a tiling with a hole, and a channel the Task IR never declared.
    refused(
        &base.replace("slice = [6, 12]", "slice = [7, 12]"),
        "IMP-005",
    );
    refused(
        &base.replace("channel = \"gripper_pose\"", "channel = \"tool_pose\""),
        "IMP-005",
    );
}

/// The adapter is `deny_unknown_fields` end to end: a key nobody reads is a mapping nobody
/// declared, and a typo that is silently ignored is the same bug as a guess.
#[test]
fn import_rl_adapter_refuses_unknown_fields() {
    let base = read("adapter-so101.toml");
    for mutation in [
        ("[robot]\nname", "[robot]\nmake = \"waveshare\"\nname"),
        ("units = \"rad\"", "units = \"rad\"\ngear = 345.0"),
        (
            "kind = \"position_target\"",
            "kind = \"position_target\"\nclip = true",
        ),
        (
            "source = \"qvel[0:6]\"",
            "source = \"qvel[0:6]\"\nsign = -1",
        ),
    ] {
        let said = run(&base.replace(mutation.0, mutation.1)).expect_err(mutation.1);
        assert!(said.starts_with("adapter:"), "{said}");
    }
}

/// The three frameworks' fixtures carry the same synthetic network in three native layouts, so
/// the remapped weights must come out byte-identical. That is the readers agreeing, which is
/// the part of `import_rl.py` no Rust test can otherwise see.
#[test]
fn import_rl_three_frameworks_agree_on_the_weights() {
    let base = read("adapter-so101.toml");
    let adapter = Adapter::parse(&base).expect("adapter");
    let task = task_from_toml(&read("task-reach.toml")).expect("task");
    let deployment = deployment_from_toml(&read("deployment-reach.toml")).expect("deployment");
    let actuators: Vec<String> = ACTUATORS.iter().map(|s| (*s).to_owned()).collect();

    let mut hashes = Vec::new();
    for framework in ["playground", "rsl-rl", "rl-games"] {
        let manifest = ImportManifest::parse(&read(&format!("import/{framework}/import.json")))
            .expect(framework);
        let weights = std::fs::read(fixture(&format!("import/{framework}/weights.safetensors")))
            .expect(framework);
        let imported = convert(
            &manifest,
            &adapter,
            &task,
            &deployment,
            &weights,
            &actuators,
        )
        .unwrap_or_else(|e| panic!("{framework}: {e}"));
        hashes.push(*blake3::hash(&imported.weights).as_bytes());
    }
    assert!(
        hashes.windows(2).all(|w| w[0] == w[1]),
        "the three readers of the same synthetic network disagree"
    );
}
