//! MCP server (spec 14.5): JSON-RPC 2.0 over stdio, `initialize` / `tools/list` / `tools/call`
//! only -- no resources, no prompts, no sampling. Spec 25.1 pins the transport: localhost/stdio
//! only, never a network listener, so this module never opens a socket.
//!
//! Framing: newline-delimited JSON-RPC 2.0 messages, no embedded newlines, per the MCP stdio
//! transport (`Status: unverified` against a live client -- confirmed from
//! <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>, see
//! `docs/api-notes/mcp.md`). `Content-Length` framing (the alternative the spec allows for
//! other transports) is not implemented: stdio mandates newline delimiting, not a choice.

use std::io::{BufRead, Write};

use serde_json::{json, Value};

use crate::tools::{self, ToolError};

/// The MCP protocol version this server speaks (2025-06-18, the latest at implementation time).
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// Default cap on one JSON-RPC request line (spec 25.1 stdio transport, `docs/api-notes/mcp.md`
/// "Request size cap"): an untrusted line this big must be rejected before it is allocated in
/// full, not after being parsed as JSON/TOML. Override via [`ServerConfig::max_request_bytes`].
pub const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;

/// Tunable knobs for [`Server`]. `Default` matches the spec-pinned defaults.
#[derive(Clone, Copy, Debug)]
pub struct ServerConfig {
    /// Requests longer than this (bytes, before the trailing newline) get a `-32600` error
    /// instead of being buffered and parsed. See [`MAX_REQUEST_BYTES`].
    pub max_request_bytes: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_request_bytes: MAX_REQUEST_BYTES,
        }
    }
}

/// One MCP tool's name, description and JSON Schema input shape, for `tools/list`.
struct ToolDef {
    name: &'static str,
    description: &'static str,
    input_schema: Value,
}

fn tool_defs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "validate",
            description: "Validate a Task/Observation/Learning/Deployment/Evaluation IR TOML \
                document and report its diagnostics plus content hash (spec 11.1).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["task", "observation", "learning", "deployment", "evaluation"],
                        "description": "sniffed from the TOML envelope's `kind` field if omitted"
                    },
                    "toml": {"type": "string", "description": "IR file contents (spec 14.3)"}
                },
                "required": ["toml"]
            }),
        },
        ToolDef {
            name: "compile",
            description: "Compile an Observation IR to the CPU reference execution plan and \
                report its compiler_hash (spec 11.1 Lower).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "observation_toml": {"type": "string"},
                    "mode": {"type": "string", "enum": ["debug", "release"]}
                },
                "required": ["observation_toml", "mode"]
            }),
        },
        ToolDef {
            name: "estimate_cost",
            description: "Estimate the spec 20.2 memory budget for an Observation (plus \
                optional Learning) IR at given batch-domain sizes, and report spec 20.3 rule \
                violations.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "observation_toml": {"type": "string"},
                    "learning_toml": {"type": "string"},
                    "sim_envs": {"type": "integer", "minimum": 0},
                    "obs_envs": {"type": "integer", "minimum": 0},
                    "views": {"type": "integer", "minimum": 0},
                    "inference_batch": {"type": "integer", "minimum": 0},
                    "precision": {"type": "string", "enum": ["f32", "f16"]}
                },
                "required": [
                    "observation_toml", "sim_envs", "obs_envs", "views", "inference_batch",
                    "precision"
                ]
            }),
        },
        ToolDef {
            name: "eval",
            description: "Compare two EvaluationReport JSON documents cell by cell (spec 10.5 \
                `es eval compare`).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "report_a_json": {"type": "string"},
                    "report_b_json": {"type": "string"}
                },
                "required": ["report_a_json", "report_b_json"]
            }),
        },
        ToolDef {
            name: "eval_run",
            description: "Run an Evaluation IR against a policy bundle (spec 10.5 `es eval \
                run`). This MCP server links no PhysicsBackend/PolicyRuntime, so it always \
                reports SKIPPED with the reason (spec 1.4: never fake a result); use the `es \
                eval run` CLI for an actual run.",
            input_schema: json!({"type": "object", "properties": {}}),
        },
        ToolDef {
            name: "hash_chain",
            description: "Report the spec 5.3 hash chain slots knowable from the given IR TOML \
                documents at authoring time (task/observation/learning/policy/deployment); \
                dataset/runtime/hardware are always \"unset\" here.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "observation": {"type": "string"},
                    "learning": {"type": "string"},
                    "deployment": {"type": "string"}
                }
            }),
        },
    ]
}

/// A JSON-RPC 2.0 protocol-level error: `(code, message)`.
type RpcError = (i64, String);

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;

fn error_response(id: &Value, code: i64, message: &str) -> String {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}).to_string()
}

/// One line read off `input`: a complete line's raw bytes (newline stripped), or a marker that
/// the line exceeded the configured cap before a newline was found (the bytes were discarded as
/// they streamed in -- this never buffers an oversized line in full, per S-3).
enum RawLine {
    Bytes(Vec<u8>),
    TooLong,
}

/// Reads one newline-delimited line from `reader`, capping how many bytes of it are retained at
/// `max_bytes`. Returns `Ok(None)` only at true EOF with nothing left to read. A line at or
/// under the cap is returned in full (`RawLine::Bytes`); over the cap, bytes past the cap are
/// consumed and dropped rather than appended, so memory use stays bounded regardless of how
/// large the offending line is, and `RawLine::TooLong` is returned once its terminating newline
/// (or EOF) is reached.
fn read_raw_line<R: BufRead>(reader: &mut R, max_bytes: usize) -> std::io::Result<Option<RawLine>> {
    let mut buf = Vec::new();
    let mut too_long = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(if buf.is_empty() && !too_long {
                None
            } else {
                Some(if too_long {
                    RawLine::TooLong
                } else {
                    RawLine::Bytes(buf)
                })
            });
        }
        // `chunk_len` is the part of `available` that belongs to the current line (up to the
        // newline, if this fill_buf call happens to contain one -- otherwise all of it). The cap
        // check below must see this uniformly regardless of which case it is, or a line whose
        // entire length (including the newline) lands in one fill_buf call would skip the cap
        // entirely.
        let newline_pos = available.iter().position(|&b| b == b'\n');
        let chunk_len = newline_pos.unwrap_or(available.len());
        if !too_long {
            if buf.len() + chunk_len > max_bytes {
                too_long = true;
                buf.clear();
                buf.shrink_to_fit();
            } else {
                buf.extend_from_slice(&available[..chunk_len]);
            }
        }
        let consumed = newline_pos.map_or(available.len(), |pos| pos + 1);
        reader.consume(consumed);
        if newline_pos.is_some() {
            return Ok(Some(if too_long {
                RawLine::TooLong
            } else {
                RawLine::Bytes(buf)
            }));
        }
    }
}

/// The MCP server: reads newline-delimited JSON-RPC requests from `input`, writes
/// newline-delimited responses to `output`, until `input` reaches EOF. A malformed request
/// never crashes the loop -- it gets a JSON-RPC error response (or, for a notification with no
/// `id`, no response at all, per JSON-RPC 2.0). Two request-level faults are handled the same
/// way, before any JSON parsing: a line over [`ServerConfig::max_request_bytes`] gets `-32600`
/// (S-3), and a line that is not valid UTF-8 gets `-32700` instead of ending the session
/// (`docs/api-notes/mcp.md` "Request size cap").
#[derive(Debug)]
pub struct Server {
    config: ServerConfig,
}

impl Server {
    pub fn new() -> Self {
        Self {
            config: ServerConfig::default(),
        }
    }

    /// A server with non-default limits, e.g. a smaller `max_request_bytes` for a constrained
    /// embedding.
    pub fn with_config(config: ServerConfig) -> Self {
        Self { config }
    }

    pub fn run<R: BufRead, W: Write>(&self, mut input: R, mut output: W) -> std::io::Result<()> {
        loop {
            let Some(raw) = read_raw_line(&mut input, self.config.max_request_bytes)? else {
                return Ok(());
            };
            let response = match raw {
                RawLine::TooLong => Some(error_response(
                    &Value::Null,
                    INVALID_REQUEST,
                    &format!(
                        "request exceeds max_request_bytes ({})",
                        self.config.max_request_bytes
                    ),
                )),
                RawLine::Bytes(bytes) => match std::str::from_utf8(&bytes) {
                    Err(e) => Some(error_response(
                        &Value::Null,
                        PARSE_ERROR,
                        &format!("invalid UTF-8 in request: {e}"),
                    )),
                    Ok(line) => {
                        let line = line.trim();
                        if line.is_empty() {
                            None
                        } else {
                            Self::handle_line(line)
                        }
                    }
                },
            };
            if let Some(response) = response {
                writeln!(output, "{response}")?;
                output.flush()?;
            }
        }
    }

    fn handle_line(line: &str) -> Option<String> {
        let req: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => return Some(error_response(&Value::Null, PARSE_ERROR, &e.to_string())),
        };
        // A JSON-RPC notification (no "id") gets no response, successful or not.
        let id = req.get("id")?.clone();
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let empty = json!({});
        let params = req.get("params").unwrap_or(&empty);
        Some(match Self::dispatch(method, params) {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
            Err((code, message)) => error_response(&id, code, &message),
        })
    }

    fn dispatch(method: &str, params: &Value) -> Result<Value, RpcError> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "electric-sheep", "version": env!("CARGO_PKG_VERSION")},
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({
                "tools": tool_defs().into_iter().map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })).collect::<Vec<_>>(),
            })),
            "tools/call" => Self::call_tool(params),
            other => Err((METHOD_NOT_FOUND, format!("method not found: {other}"))),
        }
    }

    fn call_tool(params: &Value) -> Result<Value, RpcError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| (INVALID_PARAMS, "tools/call: missing 'name'".to_owned()))?;
        let empty = json!({});
        let args = params.get("arguments").unwrap_or(&empty);

        let result = match name {
            "validate" => tools::validate(args),
            "compile" => tools::compile(args),
            "estimate_cost" => tools::estimate_cost(args),
            "eval" => tools::eval_compare(args),
            "eval_run" => tools::eval_run(args),
            "hash_chain" => tools::hash_chain(args),
            other => return Err((INVALID_PARAMS, format!("unknown tool '{other}'"))),
        };
        match result {
            Ok(v) => Ok(json!({
                "content": [{"type": "text", "text": v.to_string()}],
                "isError": false,
            })),
            Err(ToolError::BadParams(msg)) => Err((INVALID_PARAMS, msg)),
            Err(ToolError::Failed(msg)) => Ok(json!({
                "content": [{"type": "text", "text": msg}],
                "isError": true,
            })),
        }
    }
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    fn run_lines(lines: &[&str]) -> Vec<Value> {
        let input = lines.join("\n") + "\n";
        let mut out = Vec::new();
        Server::new()
            .run(BufReader::new(input.as_bytes()), &mut out)
            .expect("run");
        String::from_utf8(out)
            .expect("utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid json-rpc response"))
            .collect()
    }

    #[test]
    fn initialize_reports_the_protocol_version_and_tools_capability() {
        let resp = run_lines(&[r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#]);
        assert_eq!(resp.len(), 1);
        assert_eq!(resp[0]["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(resp[0]["result"]["capabilities"]["tools"], json!({}));
    }

    #[test]
    fn tools_list_names_all_six_tools() {
        let resp = run_lines(&[r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#]);
        let tools = resp[0]["result"]["tools"].as_array().expect("array");
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for expect in [
            "validate",
            "compile",
            "estimate_cost",
            "eval",
            "eval_run",
            "hash_chain",
        ] {
            assert!(names.contains(&expect), "{names:?}");
        }
    }

    #[test]
    fn a_notification_gets_no_response() {
        let resp = run_lines(&[r#"{"jsonrpc":"2.0","method":"initialized"}"#]);
        assert!(resp.is_empty());
    }

    #[test]
    fn unknown_method_is_method_not_found() {
        let resp = run_lines(&[r#"{"jsonrpc":"2.0","id":1,"method":"nope"}"#]);
        assert_eq!(resp[0]["error"]["code"], METHOD_NOT_FOUND);
    }

    #[test]
    fn malformed_tool_call_is_invalid_params() {
        let resp = run_lines(&[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"validate","arguments":{}}}"#,
        ]);
        assert_eq!(resp[0]["error"]["code"], INVALID_PARAMS);
    }

    #[test]
    fn malformed_json_line_is_a_parse_error_not_a_crash() {
        let resp = run_lines(&["not json"]);
        assert_eq!(resp[0]["error"]["code"], PARSE_ERROR);
    }

    /// S-3 oracle: an over-cap line gets `-32600` and is discarded, and the session survives to
    /// answer a normal request on the next line -- a 64 MiB line against the real 16 MiB default,
    /// not a shrunk-down cap, so the fix is exercised at the size the review flagged.
    #[test]
    fn oversized_line_is_invalid_request_and_the_session_stays_alive() {
        let huge = "x".repeat(64 * 1024 * 1024);
        let input = format!(
            "{huge}\n{}\n",
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#
        );
        let mut out = Vec::new();
        Server::new()
            .run(BufReader::new(input.as_bytes()), &mut out)
            .expect("run");
        let responses: Vec<Value> = String::from_utf8(out)
            .expect("utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid json-rpc response"))
            .collect();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["error"]["code"], INVALID_REQUEST);
        assert_eq!(responses[1]["result"]["protocolVersion"], PROTOCOL_VERSION);
    }

    /// Same shape as above but against a small configured cap, so the "discarded, loop alive"
    /// behavior is also pinned without allocating tens of megabytes per test run.
    #[test]
    fn oversized_line_with_a_small_configured_cap_is_invalid_request() {
        let input = format!(
            "{}\n{}\n",
            "x".repeat(100),
            r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#
        );
        let mut out = Vec::new();
        Server::with_config(ServerConfig {
            max_request_bytes: 64,
        })
        .run(BufReader::new(input.as_bytes()), &mut out)
        .expect("run");
        let responses: Vec<Value> = String::from_utf8(out)
            .expect("utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid json-rpc response"))
            .collect();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["error"]["code"], INVALID_REQUEST);
        assert_eq!(responses[1]["result"], json!({}));
    }

    /// S-3 oracle: a non-UTF-8 byte gets `-32700` (parse error) instead of killing the loop via
    /// `line?` propagating `InvalidData` out of `run()`.
    #[test]
    fn invalid_utf8_byte_is_a_parse_error_and_the_session_stays_alive() {
        let mut input = vec![0xFFu8, b'\n'];
        input.extend_from_slice(br#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#);
        input.push(b'\n');
        let mut out = Vec::new();
        Server::new()
            .run(BufReader::new(&input[..]), &mut out)
            .expect("run");
        let responses: Vec<Value> = String::from_utf8(out)
            .expect("utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid json-rpc response"))
            .collect();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["error"]["code"], PARSE_ERROR);
        assert_eq!(responses[1]["result"], json!({}));
    }
}
