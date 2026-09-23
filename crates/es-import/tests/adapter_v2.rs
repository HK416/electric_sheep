//! Oracle 1 of packet M11/X2: adapter v2 -- each new field parses and lands where the packet
//! says, each new refusal (`IMP-006` .. `IMP-009`) is by name, and every committed v1 adapter
//! converts to exactly the bytes it converted to before the packet (spec 28.14 rule 1).

use std::path::{Path, PathBuf};

use es_compile::{CpuPlan, PlanMode};
use es_import::rl_import::{
    actuator_rows, convert, Adapter, HistoryOrder, ImportManifest, Offset, RlImport, Scale,
    SceneActuator, Severity,
};
use es_ir::learning::{LearningNode, StatsSource};
use es_ir::observation::{NormalizeStats, ObservationNode};
use es_ir::serial::{deployment_from_toml, task_from_toml};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rl")
        .join(name)
}

fn read(name: &str) -> String {
    std::fs::read_to_string(fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

const ACTUATORS: [&str; 6] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
];

/// `task-reach-last-action.toml`'s `initial`: the rest pose, our order.
const DEFAULT_OURS: [f64; 6] = [0.1, -0.4, 0.7, 0.25, -0.05, 0.3];

fn actuators() -> Vec<String> {
    ACTUATORS.iter().map(|s| (*s).to_owned()).collect()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn h(bytes: &[u8]) -> String {
    hex(blake3::hash(bytes).as_bytes())
}

// --- 1. every v1 conversion, byte for byte ----------------------------------------------------

/// One conversion's four outputs, each as a blake3 of its bytes:
/// `observation.toml | learning.toml | weights | mapping-report.json`.
fn digest(dir: &str, adapter: &str, task: &str, deployment: &str) -> String {
    let manifest =
        ImportManifest::parse(&read(&format!("import/{dir}/import.json"))).expect("manifest");
    let adapter = Adapter::parse(&read(adapter)).expect("adapter");
    let task = task_from_toml(&read(task)).expect("task");
    let deployment = deployment_from_toml(&read(deployment)).expect("deployment");
    let weights =
        std::fs::read(fixture(&format!("import/{dir}/weights.safetensors"))).expect("weights");
    let out = convert(
        &manifest,
        &adapter,
        &task,
        &deployment,
        &weights,
        &actuators(),
    )
    .unwrap_or_else(|e| panic!("{dir}: {e}"));
    let obs = es_ir::serial::observation_to_toml(&out.observation).expect("observation toml");
    let learning = es_ir::serial::learning_to_toml(&out.learning).expect("learning toml");
    let report = serde_json::to_string_pretty(&out.report).expect("report json");
    [
        h(obs.as_bytes()),
        h(learning.as_bytes()),
        h(&out.weights),
        h(report.as_bytes()),
    ]
    .join(" ")
}

/// The committed v1 conversions, recorded on `main` (e6f75b0) before this packet's first line
/// of `rl_import.rs` changed. Typed in on purpose: this is the one place they are asserted.
const V1_PINS: [(&str, &str, &str, &str, &str); 4] = [
    (
        "playground",
        "adapter-so101.toml",
        "task-reach.toml",
        "deployment-reach.toml",
        "6cf83d034bdd9828b47131cf6802d5cfc2ca59beda5f085bbf86eb1e98e12b58 \
         ba7ca55db02eade1740bad696ecb60e6a055514900011190f7c0634fe98d3cb5 \
         5d9ea7b55e90accd311e611ec01a2ee2ea925deaf9155bb598ae3ee0292162e0 \
         e7152b4ca7bf36d8736afe538073f6294edf2b5281535ffbb1185ac41bec6c54",
    ),
    (
        "rsl-rl",
        "adapter-so101.toml",
        "task-reach.toml",
        "deployment-reach.toml",
        "4bb7bf9dea126cc6e0ca7e8d01c207905d43d9220815e20c77081d452f001e05 \
         7bdd8192992c5a2df319325546557d0280cc3cac8f1290d83cf40c0e05e9c185 \
         5d9ea7b55e90accd311e611ec01a2ee2ea925deaf9155bb598ae3ee0292162e0 \
         5a7e0d030fb33e5ca2eef44a840bfa38a492b5b847f2dd0de451c8b9f513dffb",
    ),
    (
        "rl-games",
        "adapter-so101.toml",
        "task-reach.toml",
        "deployment-reach.toml",
        "68d038823eb4bf5801433f94138df5486872819e37a79a2f4ab1403d2e62eda8 \
         7bdd8192992c5a2df319325546557d0280cc3cac8f1290d83cf40c0e05e9c185 \
         5d9ea7b55e90accd311e611ec01a2ee2ea925deaf9155bb598ae3ee0292162e0 \
         61890c5b59863278abebcfc9c80fe3f4fd3319fec6047798e53ca2a5c1714fce",
    ),
    (
        "playground-delta",
        "adapter-so101-delta.toml",
        "task-reach-delta.toml",
        "deployment-reach-delta.toml",
        "3fc7c0ffd9aac78c99af400a06c99b0c6c34f1f4bac78e68df9f4e99a0b549d0 \
         0649c09aedf1ac66ec381f1dca3e41de32369667160b23e65c54f212c3522f96 \
         5d9ea7b55e90accd311e611ec01a2ee2ea925deaf9155bb598ae3ee0292162e0 \
         fb633c323c2291cc765b9970c82eab69e55f07d3c18e3a95f21cbfeb32c3fce5",
    ),
];

#[test]
fn adapter_v2_leaves_every_v1_conversion_byte_identical() {
    let mut moved = Vec::new();
    for (dir, adapter, task, deployment, pinned) in V1_PINS {
        let pinned = pinned.split_whitespace().collect::<Vec<_>>().join(" ");
        let got = digest(dir, adapter, task, deployment);
        if got != pinned {
            moved.push(format!(
                "{dir} {adapter}\n  pinned {pinned}\n  got    {got}"
            ));
        }
    }
    assert!(
        moved.is_empty(),
        "a v1 conversion moved:\n{}",
        moved.join("\n")
    );
}

// --- a synthetic neutral source, so the v2 cases need no Python -------------------------------

/// `import.json` + `weights.safetensors` for an `obs_dim -> [8, 8] -> 6` ELU actor with no
/// normalizer, built in memory. `extra` is spliced into the manifest's JSON object.
fn synthetic(obs_dim: u64, squash: &str, extra: &str) -> (ImportManifest, Vec<u8>) {
    let manifest = ImportManifest::parse(&format!(
        r#"{{"framework": "synthetic", "obs_dim": {obs_dim}, "action_dim": 6, "hidden": [8, 8],
            "activation": "elu", "squash": "{squash}"{extra}}}"#
    ))
    .expect("synthetic manifest");
    let mut ckpt = es_policy::weights::Checkpoint::new();
    let mut put = |name: &str, shape: Vec<u64>| {
        let n = shape.iter().product::<u64>() as usize;
        let values = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        ckpt.insert(name.to_owned(), (shape, values));
    };
    put("mlp.0.weight", vec![8, obs_dim]);
    put("mlp.0.bias", vec![8]);
    put("mlp.1.weight", vec![8, 8]);
    put("mlp.1.bias", vec![8]);
    put("head.weight", vec![6, 8]);
    put("head.bias", vec![6]);
    (manifest, es_policy::weights::write_safetensors(&ckpt))
}

/// The Isaac adapter, edited, against the synthetic 25-wide source and the last-action task.
fn isaac(edit: &[(&str, &str)], squash: &str, extra: &str) -> Result<RlImport, String> {
    let mut text = read("adapter-isaac-reach.toml");
    for (from, to) in edit {
        assert!(
            text.contains(from),
            "the edit's anchor {from:?} is in the adapter"
        );
        text = text.replace(from, to);
    }
    let obs_dim = if text.contains("history = 2") { 31 } else { 25 };
    let (manifest, weights) = synthetic(obs_dim, squash, extra);
    let adapter = Adapter::parse(&text).map_err(|e| e.to_string())?;
    let task = task_from_toml(&read("task-reach-last-action.toml")).expect("task");
    let deployment = deployment_from_toml(&read("deployment-reach.toml")).expect("deployment");
    convert(
        &manifest,
        &adapter,
        &task,
        &deployment,
        &weights,
        &actuators(),
    )
    .map_err(|e| e.to_string())
}

fn refused(result: Result<RlImport, String>, code: &str, mentions: &str) {
    let said = result
        .err()
        .unwrap_or_else(|| panic!("{code} was not refused"));
    assert!(
        said.starts_with(code) && said.contains(mentions),
        "expected {code} naming {mentions:?}, got:\n{said}"
    );
    assert!(
        es_ir::codes::lookup(code).is_some(),
        "{code} is not in the diagnostic dictionary"
    );
}

fn normalize(out: &RlImport) -> (Vec<f64>, Vec<f64>) {
    out.observation
        .graph
        .nodes
        .values()
        .find_map(|n| match n {
            ObservationNode::Normalize {
                stats: NormalizeStats::MeanStd { mean, std },
                ..
            } => Some((mean.clone(), std.clone())),
            _ => None,
        })
        .expect("a MeanStd Normalize")
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
}

// --- 2. every new field parses and lands --------------------------------------------------------

#[test]
fn adapter_v2_fields_parse() {
    let a = Adapter::parse(&read("adapter-isaac-reach.toml")).expect("the Isaac adapter");
    assert!(a.joints.source_order.is_empty());
    assert_eq!(a.joints.source_names.as_ref().map(Vec::len), Some(6));
    assert_eq!(
        a.joints.rename.get("gripper_jaw").map(String::as_str),
        Some("gripper")
    );
    assert_eq!(a.joints.default_pos.as_ref().map(Vec::len), Some(6));
    assert!(a.action.use_default_offset);
    assert_eq!(a.timing.as_ref().map(|t| t.policy_dt), Some(0.02));
    let ch = &a.observation.channels;
    assert_eq!(ch[0].offset, Some(Offset::Named("default_pos".to_owned())));
    assert_eq!(ch[1].scale, Some(Scale::One(0.05)));

    // The fields the committed adapters do not use, in both their forms.
    let more = read("adapter-isaac-reach.toml")
        .replace(
            "scale = 0.05",
            "scale = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]\nhistory = 2\nhistory_order = \"newest_first\"",
        )
        .replace(
            "use_default_offset = true",
            "use_default_offset = true\nclip = [-1.0, 1.0]",
        )
        .replace(
            "\n[joints.rename]\n",
            "\n[actuators]\nstiffness = [17.8, 17.8, 17.8, 17.8, 17.8, 17.8]\n\n[joints.rename]\n",
        );
    let b = Adapter::parse(&more).expect("every v2 field");
    assert_eq!(
        b.observation.channels[1].scale,
        Some(Scale::Each(vec![0.05; 6]))
    );
    assert_eq!(b.observation.channels[1].history, Some(2));
    assert_eq!(
        b.observation.channels[1].history_order,
        Some(HistoryOrder::NewestFirst)
    );
    assert_eq!(b.action.clip, Some([-1.0, 1.0]));
    assert_eq!(
        b.actuators.and_then(|x| x.stiffness).map(|v| v.len()),
        Some(6)
    );

    // `deny_unknown_fields` still holds in every new block.
    for (from, to) in [
        ("policy_dt = 0.02", "policy_dt = 0.02\ndecimation = 4"),
        ("scale = 0.05", "scale = 0.05\nnoise = 0.01"),
        (
            "gripper_jaw = \"gripper\"",
            "gripper_jaw = \"gripper\"\n[joints.alias]",
        ),
    ] {
        let said = Adapter::parse(&read("adapter-isaac-reach.toml").replace(from, to))
            .expect_err(to)
            .to_string();
        assert!(said.starts_with("adapter:"), "{said}");
    }
}

#[test]
fn adapter_v2_folds_into_the_normalizer_and_the_unnormalizer() {
    let out = isaac(&[], "none", "").expect("the Isaac adapter converts");
    let (mean, std) = normalize(&out);
    // joint_pos: `q - default` -> mean = default, in OUR order.
    assert!(close(&mean[0..6], &DEFAULT_OURS), "{:?}", &mean[0..6]);
    assert!(close(&std[0..6], &[1.0; 6]));
    // joint_vel: `qd * 0.05` -> std = 1 / 0.05.
    assert!(close(&std[6..12], &[20.0; 6]), "{:?}", &std[6..12]);
    // cube_pose: untouched.
    assert!(close(&mean[12..19], &[0.0; 7]) && close(&std[12..19], &[1.0; 7]));
    // last_action: `(row - default) / 0.5` -> mean = default, std = 0.5.
    assert!(close(&mean[19..25], &DEFAULT_OURS), "{:?}", &mean[19..25]);
    assert!(close(&std[19..25], &[0.5; 6]), "{:?}", &std[19..25]);

    // The unnormalizer, in our actuator order: `ctrl = default + 0.5 * a`.
    let (m, s) = out
        .learning
        .nodes
        .nodes
        .values()
        .find_map(|n| match n {
            LearningNode::Normalizer {
                stats: StatsSource::MeanStd { mean, std },
                ..
            } => Some((mean.clone(), std.clone())),
            _ => None,
        })
        .expect("the unnormalizer");
    assert!(
        close(&m, &DEFAULT_OURS) && close(&s, &[0.5; 6]),
        "{m:?} {s:?}"
    );
    assert!(out
        .report
        .timing
        .as_deref()
        .is_some_and(|t| t.contains("0.02")));
    assert!(
        out.report.joints[1].note.contains("gripper_jaw") && out.report.joints[1].name == "gripper",
        "{:?}",
        out.report.joints[1]
    );
    assert!(out.report.channels[3].note.contains("action tail"));

    // The manifest may state the pose instead of the adapter; the same pose converts the same.
    let from_manifest = isaac(
        &[("default_pos = [0.7, 0.3, -0.4, 0.1, 0.25, -0.05]\n", "")],
        "none",
        r#", "default_joint_pos": [0.7, 0.3, -0.4, 0.1, 0.25, -0.05], "decimation": 4, "sim_dt": 0.005"#,
    )
    .expect("the pose from the manifest");
    assert_eq!(normalize(&from_manifest), (mean, std));

    // A clip that cannot bind under tanh is admitted, and the action is unchanged by it.
    isaac(
        &[(
            "use_default_offset = true",
            "use_default_offset = true\nclip = [-1.0, 1.0]",
        )],
        "tanh",
        "",
    )
    .expect("a non-binding clip under tanh");
}

#[test]
fn adapter_v2_history_becomes_a_temporal_window() {
    for order in ["newest_last", "newest_first"] {
        let out = isaac(
            &[
                (
                    "scale = 0.05",
                    &format!("scale = 0.05\nhistory = 2\nhistory_order = \"{order}\""),
                ),
                ("slice = [6, 12]", "slice = [6, 18]"),
                ("slice = [12, 19]", "slice = [18, 25]"),
                ("slice = [19, 25]", "slice = [25, 31]"),
            ],
            "none",
            "",
        )
        .unwrap_or_else(|e| panic!("{order}: {e}"));
        let windows = out
            .observation
            .graph
            .nodes
            .values()
            .filter(|n| {
                matches!(n, ObservationNode::TemporalWindowNode { window, .. } if window.n_steps == 2)
            })
            .count();
        assert_eq!(windows, 1, "{order}");
        let (_, std) = normalize(&out);
        assert!(
            close(&std[6..18], &[20.0; 12]),
            "{order}: {:?}",
            &std[6..18]
        );
        let plan = CpuPlan::compile(&out.observation, PlanMode::Release);
        assert!(
            plan.is_ok(),
            "{order}: the windowed observation compiles: {:?}",
            plan.err()
        );
    }
}

#[test]
fn adapter_v2_actuator_rows_compare_and_never_convert() {
    let text = read("adapter-isaac-reach.toml").replace(
        "\n[joints.rename]\n",
        "\n[actuators]\nstiffness = [17.8, 17.8, 17.8, 17.8, 17.8, 17.8]\n\n[joints.rename]\n",
    );
    let adapter = Adapter::parse(&text).expect("adapter");
    let mut scene: Vec<SceneActuator> = ACTUATORS
        .iter()
        .map(|n| SceneActuator {
            name: (*n).to_owned(),
            stiffness: Some(17.8),
            ..SceneActuator::default()
        })
        .collect();
    scene[5].stiffness = Some(10.0); // our gripper, the source's `gripper_jaw` (index 1)
    let rows = actuator_rows(&adapter, &scene);
    assert_eq!(rows.len(), 6);
    let jaw = rows
        .iter()
        .find(|r| r.name == "gripper")
        .expect("the gripper row");
    assert_eq!(jaw.severity, Severity::Warning);
    assert_eq!((jaw.source, jaw.scene), (17.8, Some(10.0)));
    assert_eq!(
        rows.iter().filter(|r| r.severity == Severity::Ok).count(),
        5
    );
}

// --- 3. every new refusal, by name ----------------------------------------------------------------

#[test]
fn adapter_v2_refusals_are_by_name() {
    // The control: the committed adapter converts, so the refusals below are not "everything".
    isaac(&[], "none", "").expect("the control converts");

    // IMP-006: Isaac's own reach task runs at 30 Hz; the deployment runs 50.
    refused(
        isaac(
            &[("policy_dt = 0.02", "policy_dt = 0.03333333333")],
            "none",
            "",
        ),
        "IMP-006",
        "policy_dt",
    );
    refused(
        isaac(
            &[],
            "none",
            r#", "decimation": 2, "sim_dt": 0.016666666666666666"#,
        ),
        "IMP-006",
        "decimation 2",
    );

    // IMP-007: a locomotion term, and a command nothing in the Task IR computes.
    refused(
        isaac(
            &[("mdp.joint_vel_rel", "mdp.projected_gravity")],
            "none",
            "",
        ),
        "IMP-007",
        "projected_gravity",
    );
    refused(
        isaac(
            &[("channel = \"cube_pose\"", "channel = \"ee_goal\"")],
            "none",
            "",
        ),
        "IMP-007",
        "generated_commands",
    );

    // IMP-008: two declarations of the order, none, and a rename nobody uses.
    refused(
        isaac(
            &[(
                "units = \"rad\"",
                "units = \"rad\"\nsource_order = [\"shoulder_pan\"]",
            )],
            "none",
            "",
        ),
        "IMP-008",
        "both",
    );
    refused(
        isaac(
            &[(
                "gripper_jaw = \"gripper\"",
                "gripper_jaw = \"gripper\"\njaw = \"gripper\"",
            )],
            "none",
            "",
        ),
        "IMP-008",
        "jaw",
    );

    // IMP-002 through v2: a source name that resolves to no actuator.
    refused(
        isaac(&[("gripper_jaw = \"gripper\"", "")], "none", ""),
        "IMP-002",
        "gripper_jaw",
    );

    // IMP-009: a per-term clip (no Observation IR clamp node), and an action clip that binds.
    refused(
        isaac(
            &[("scale = 0.05", "scale = 0.05\nclip = [-100.0, 100.0]")],
            "none",
            "",
        ),
        "IMP-009",
        "joint_vel",
    );
    refused(
        isaac(
            &[(
                "use_default_offset = true",
                "use_default_offset = true\nclip = [-1.0, 1.0]",
            )],
            "none",
            "",
        ),
        "IMP-009",
        "squash",
    );

    // IMP-005: the previous action's `initial` must fold to Isaac's zero at reset.
    refused(
        isaac(
            &[(
                "use_default_offset = true",
                "offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
            )],
            "none",
            "",
        ),
        "IMP-005",
        "initial",
    );

    // Not a code, because it never got as far as a mapping: a pose nobody states.
    let said = isaac(
        &[("default_pos = [0.7, 0.3, -0.4, 0.1, 0.25, -0.05]\n", "")],
        "none",
        "",
    )
    .expect_err("use_default_offset without a pose");
    assert!(said.contains("does not guess"), "{said}");
}
