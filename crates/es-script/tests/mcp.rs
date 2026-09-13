//! In-process MCP harness test (spec 14.5): feeds `Server::run` a request sequence over an
//! in-memory pipe and asserts the JSON-RPC responses, the same way `crates/es/tests/cli.rs`
//! spawns the `es mcp` subprocess but without paying for a process per test.

use std::collections::{BTreeMap, BTreeSet};

use es_ir::graph::Graph;
use es_ir::observation::{ObservationIr, TemporalModel};
use es_ir::serial::{observation_to_toml, task_to_toml};
use es_ir::task::{ObservationSpec, SceneRef, TaskConfig, TaskIr};
use es_script::mcp::Server;
use serde_json::{json, Value};

/// A minimal Task IR: empty graph, no declared channels. `TaskIr::validate` has nothing to
/// complain about with no nodes and no RNG streams to cross-check, so this is a clean fixture
/// for the `validate` tool without pulling in the full spec 11.1 cross-IR bundle (`validate`
/// checks one IR at a time; only `es ir check` / `es_ir::cross` needs all five to agree).
fn minimal_task_toml() -> String {
    let ir = TaskIr {
        schema_version: 1,
        scene: SceneRef {
            path: "scenes/fixture.usda".to_owned(),
            scene_hash: [1; 32],
            asset_hash: [2; 32],
        },
        graph: Graph::new(1),
        observation_spec: ObservationSpec {
            channels: BTreeMap::default(),
        },
        control: None,
        config: TaskConfig {
            max_episode_steps: 200,
            control_rate_hz: 50.0,
            deterministic: true,
            rng_streams: BTreeSet::new(),
        },
    };
    task_to_toml(&ir).expect("task toml")
}

/// A minimal Observation IR: no nodes, no outputs. Enough for `estimate_cost` (every camera/
/// history item reports `unavailable`, never a panic) without a physics scene or a policy.
fn minimal_observation_toml() -> String {
    let ir = ObservationIr {
        schema_version: 1,
        task_ref: [0; 32],
        graph: Graph::new(1),
        temporal: TemporalModel::default(),
        outputs: BTreeMap::default(),
    };
    observation_to_toml(&ir).expect("observation toml")
}

fn request(id: i64, method: &str, params: &Value) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string()
}

/// Runs one request sequence through `Server::run` over an in-memory pipe and returns the
/// parsed JSON-RPC responses in order.
fn run_sequence(requests: &[String]) -> Vec<Value> {
    let input = requests.join("\n") + "\n";
    let mut out = Vec::new();
    Server::new()
        .run(std::io::BufReader::new(input.as_bytes()), &mut out)
        .expect("server run should never error on malformed input");
    String::from_utf8(out)
        .expect("utf8 output")
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON: {l}: {e}")))
        .collect()
}

#[test]
fn initialize_then_tools_list_then_validate_then_estimate_cost_then_malformed() {
    let task_toml = minimal_task_toml();
    let obs_toml = minimal_observation_toml();

    let responses = run_sequence(&[
        request(1, "initialize", &json!({"protocolVersion": "2025-06-18"})),
        request(2, "tools/list", &json!({})),
        request(
            3,
            "tools/call",
            &json!({"name": "validate", "arguments": {"kind": "task", "toml": task_toml}}),
        ),
        request(
            4,
            "tools/call",
            &json!({
                "name": "estimate_cost",
                "arguments": {
                    "observation_toml": obs_toml,
                    "sim_envs": 4, "obs_envs": 2, "views": 1, "inference_batch": 8,
                    "precision": "f32",
                }
            }),
        ),
        // Malformed: `tools/call` naming a real tool but omitting its required argument.
        request(
            5,
            "tools/call",
            &json!({"name": "compile", "arguments": {}}),
        ),
    ]);
    assert_eq!(responses.len(), 5);

    // 1. initialize
    assert_eq!(responses[0]["id"], 1);
    assert_eq!(responses[0]["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(responses[0]["result"]["capabilities"]["tools"], json!({}));

    // 2. tools/list
    let tools = responses[1]["result"]["tools"]
        .as_array()
        .expect("tools array");
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    for expect in [
        "validate",
        "compile",
        "estimate_cost",
        "eval",
        "eval_run",
        "hash_chain",
    ] {
        assert!(
            names.contains(&expect),
            "missing tool '{expect}' in {names:?}"
        );
    }
    assert_eq!(tools[0]["inputSchema"]["type"], "object");

    // 3. tools/call validate
    assert_eq!(responses[2]["result"]["isError"], false);
    let validate_text = responses[2]["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    let validate_body: Value = serde_json::from_str(validate_text).expect("json content");
    assert_eq!(validate_body["kind"], "task");
    assert_eq!(validate_body["has_error"], false);
    assert!(validate_body["hash"].is_string());

    // 4. tools/call estimate_cost
    assert_eq!(responses[3]["result"]["isError"], false);
    let cost_text = responses[3]["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    let cost_body: Value = serde_json::from_str(cost_text).expect("json content");
    assert!(cost_body["total_bytes"].is_u64());
    assert!(cost_body["items"].as_array().is_some());

    // 5. malformed tools/call (missing required 'mode') is a JSON-RPC protocol error, never a
    // crash and never silently accepted (spec 25.1: malformed input -> -32602).
    assert_eq!(responses[4]["error"]["code"], -32602);
    assert!(responses[4].get("result").is_none());
}

#[test]
fn a_line_that_is_not_json_at_all_is_reported_not_fatal() {
    let responses = run_sequence(&[
        request(1, "initialize", &json!({})),
        "not json at all".to_owned(),
        request(2, "tools/list", &json!({})),
    ]);
    // The bad line gets its own error response; the server keeps serving the request after it.
    assert_eq!(responses.len(), 3);
    assert_eq!(responses[0]["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(responses[1]["error"]["code"], -32700);
    assert!(responses[2]["result"]["tools"].is_array());
}
