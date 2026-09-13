//! Integration tests for `es_script::generate` (spec 14.5).

use std::collections::BTreeMap;

use es_ir::observation::ObservationIr;
use es_ir::task::SceneRef;
use es_script::generate::{
    build_prompt, parse_candidate, repair_loop, GenerateError, Palette, Prompt, TaskSpecPrompt,
    GEN_NO_REWARD,
};

fn palette_kinds() -> Vec<&'static str> {
    es_ir::factory::BUILTIN_TASK_KINDS
        .iter()
        .chain(es_ir::factory::BUILTIN_LEARNING_KINDS)
        .copied()
        .collect()
}

#[test]
fn palette_lists_every_builtin_kind() {
    let json = Palette::from_builtins().to_json().to_owned();
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    let entries = parsed["entries"].as_array().expect("an array");
    assert_eq!(entries.len(), palette_kinds().len());
    for kind in palette_kinds() {
        assert!(
            entries.iter().any(|e| e["kind"] == kind),
            "{kind} is exported"
        );
    }
    let reward = entries
        .iter()
        .find(|e| e["kind"] == "Reward")
        .expect("Reward is exported");
    assert_eq!(reward["ir"], "task");
    assert!(reward["params"]
        .as_array()
        .expect("params")
        .iter()
        .any(|p| p["name"] == "weight"));
}

#[test]
fn palette_is_deterministic() {
    assert_eq!(
        Palette::from_builtins().to_json(),
        Palette::from_builtins().to_json()
    );
}

#[test]
fn prompt_contains_every_builtin_kind() {
    let prompt = build_prompt(
        &Palette::from_builtins(),
        &TaskSpecPrompt {
            description: "pick up the red block".to_owned(),
            ..Default::default()
        },
    );
    for kind in palette_kinds() {
        assert!(prompt.system.contains(kind), "{kind} missing from prompt");
    }
    assert!(prompt.user.contains("pick up the red block"));
}

#[test]
fn build_prompt_carries_scene_and_constraints() {
    let mut constraints = BTreeMap::new();
    constraints.insert("max_episode_steps".to_owned(), "500".to_owned());
    let prompt = build_prompt(
        &Palette::from_builtins(),
        &TaskSpecPrompt {
            description: "reach the target".to_owned(),
            scene: Some(SceneRef {
                path: "scenes/reach.xml".to_owned(),
                scene_hash: [1; 32],
                asset_hash: [2; 32],
            }),
            constraints,
        },
    );
    assert!(prompt.user.contains("scenes/reach.xml"));
    assert!(prompt.user.contains("max_episode_steps: 500"));
}

fn fenced(kind: &str, body: &str) -> String {
    format!("Here is the IR:\n\n```toml\n{body}\n```\n\nkind={kind}")
}

#[test]
fn parse_candidate_reads_a_fenced_block() {
    let ir = es_ir::task::testing::task_ir(&[("dist".to_owned(), 1.0, 1.0)], 1);
    let toml = es_ir::serial::task_to_toml(&ir).unwrap();
    let (parsed, obs) = parse_candidate(&fenced("task", &toml)).unwrap();
    assert_eq!(parsed.task_hash().unwrap(), ir.task_hash().unwrap());
    assert!(obs.is_none());
}

#[test]
fn parse_candidate_reads_a_bare_block_with_no_fences() {
    let ir = es_ir::task::testing::task_ir(&[("dist".to_owned(), 1.0, 1.0)], 1);
    let toml = es_ir::serial::task_to_toml(&ir).unwrap();
    let (parsed, _) = parse_candidate(&toml).unwrap();
    assert_eq!(parsed.task_hash().unwrap(), ir.task_hash().unwrap());
}

#[test]
fn parse_candidate_reads_both_task_and_observation_blocks() {
    let task = es_ir::task::testing::task_ir(&[("dist".to_owned(), 1.0, 1.0)], 1);
    let obs = ObservationIr::new(1, task.task_hash().unwrap());
    let reply = format!(
        "{}\n\n{}",
        fenced("task", &es_ir::serial::task_to_toml(&task).unwrap()),
        fenced(
            "observation",
            &es_ir::serial::observation_to_toml(&obs).unwrap()
        ),
    );
    let (parsed_task, parsed_obs) = parse_candidate(&reply).unwrap();
    assert_eq!(parsed_task.task_hash().unwrap(), task.task_hash().unwrap());
    assert_eq!(
        parsed_obs.unwrap().observation_hash().unwrap(),
        obs.observation_hash().unwrap()
    );
}

#[test]
fn parse_candidate_reports_no_task_candidate() {
    let err = parse_candidate("not TOML at all").unwrap_err();
    assert!(matches!(err, GenerateError::NoTaskCandidate(_)));
}

/// The spec 14.5 loop: an invalid candidate (no Reward node) gets its diagnostic fed back, the
/// next candidate is valid, and the report shows exactly two rounds.
#[test]
fn repair_loop_fixes_a_missing_reward_in_two_rounds() {
    // `task_ir(&[], _)` wires GetTime -> Terminate with no reward terms: exactly the "missing
    // reward" defect this test wants, with nothing else wrong.
    let invalid = es_ir::task::testing::task_ir(&[], 1);
    let valid = es_ir::task::testing::task_ir(&[("dist".to_owned(), 1.0, 1.0)], 1);

    let invalid_reply = fenced("task", &es_ir::serial::task_to_toml(&invalid).unwrap());
    let valid_reply = fenced("task", &es_ir::serial::task_to_toml(&valid).unwrap());

    let mut replies = vec![invalid_reply, valid_reply].into_iter();
    let mut provider = move |_: &Prompt| {
        replies
            .next()
            .ok_or_else(|| GenerateError::Provider("no more scripted replies".to_owned()))
    };

    let palette = Palette::from_builtins();
    let prompt = build_prompt(
        &palette,
        &TaskSpecPrompt {
            description: "reach the target, then stop".to_owned(),
            ..Default::default()
        },
    );
    let report = repair_loop(&prompt, &mut provider, 3);

    assert_eq!(report.rounds.len(), 2, "{report:?}");
    assert!(
        report.rounds[0]
            .diagnostics
            .iter()
            .any(|d| d.contains(GEN_NO_REWARD)),
        "{:?}",
        report.rounds[0].diagnostics
    );
    assert!(report.rounds[1].diagnostics.is_empty());
    let result = report.result.expect("round 2 validated");
    assert_eq!(result.task_hash().unwrap(), valid.task_hash().unwrap());
}

#[test]
fn repair_loop_gives_up_after_max_rounds() {
    let invalid = es_ir::task::testing::task_ir(&[], 1);
    let reply = fenced("task", &es_ir::serial::task_to_toml(&invalid).unwrap());
    let mut provider = move |_: &Prompt| Ok(reply.clone());
    let palette = Palette::from_builtins();
    let prompt = build_prompt(
        &palette,
        &TaskSpecPrompt {
            description: "reach the target".to_owned(),
            ..Default::default()
        },
    );
    let report = repair_loop(&prompt, &mut provider, 2);
    assert_eq!(report.rounds.len(), 2);
    assert!(report.result.is_none());
}
