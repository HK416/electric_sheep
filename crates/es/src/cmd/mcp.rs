//! `es mcp` (spec 14.5): runs the Electric Sheep MCP server on stdin/stdout so external tools
//! (LLM task-generation loops, editors) can use `validate` / `compile` / `estimate_cost` /
//! `eval` as MCP tools without shelling out to `es` itself. Spec 25.1: stdio only, this never
//! opens a network listener.

use std::io::{stdin, stdout, BufReader};

use es_script::mcp::Server;

use crate::error::CliError;

const HELP: &str = "\
es mcp

Runs the Electric Sheep MCP server (spec 14.5): reads newline-delimited JSON-RPC 2.0
requests from stdin and writes newline-delimited responses to stdout until stdin closes
(MCP stdio transport). Exposes `validate`, `compile`, `estimate_cost`, `eval`, `eval_run`
and `hash_chain` as MCP tools (`tools/list` describes each one's JSON Schema).

`eval_run` always reports SKIPPED: this server links no PhysicsBackend/PolicyRuntime, so
an actual evaluation run still needs `es eval run` directly.

Spec 25.1: localhost/stdio only -- this command never opens a network socket.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    if !args.is_empty() {
        return Err(CliError::Usage(format!(
            "es mcp takes no arguments\n\n{HELP}"
        )));
    }
    Server::new()
        .run(BufReader::new(stdin()), stdout())
        .map_err(|e| CliError::Runtime(e.to_string()))?;
    Ok(0)
}
