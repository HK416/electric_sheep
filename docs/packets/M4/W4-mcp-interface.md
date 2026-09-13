# W4 — MCP interface

Spec: spec 14.5 (line ~1759: "MCP 인터페이스로 `validate` / `compile` / `estimate_cost` /
`eval`을 노출한다" -- expose Electric Sheep as a backend for external tools, no evolution-loop
orchestrator of our own), spec 11.1 (Cross-IR Check / Lower, reused by `validate`/`compile`),
spec 20 (memory budget, reused by `estimate_cost`), spec 10.5 (`eval compare`/`eval run`
artifacts, reused by `eval`/`eval_run`), spec 25.1 (security: localhost/stdio only, no network
listener here).

Design note: `docs/design/mcp-interface.md`. Pinned protocol digest: `docs/api-notes/mcp.md`.

## context

```
crates/es-script/src/lib.rs             `pub mod mcp; pub mod tools;`
crates/es-script/src/mcp.rs             NEW -- JSON-RPC 2.0 stdio Server
crates/es-script/src/tools.rs           NEW -- validate/compile/estimate_cost/eval/eval_run/
                                         hash_chain tool bodies
crates/es-script/tests/mcp.rs           NEW -- in-process request-sequence harness
crates/es/src/cmd/mcp.rs                NEW -- `es mcp` subcommand
crates/es/src/cmd/mod.rs                `pub mod mcp;`
crates/es/src/main.rs                   `Some("mcp") => cmd::mcp::dispatch(&args[1..])`
crates/es/tests/cli.rs                  appended -- `es mcp` spawned with piped stdin
docs/api-notes/mcp.md                   NEW
docs/design/mcp-interface.md            NEW
docs/packets/M4/W4-mcp-interface.md     this file
```

`crates/es-script/src/generate.rs` (LLM task generation, spec 14.5's other half) is a
concurrent packet's file and is not touched here.

## spec

1. `es_script::mcp::Server`: reads newline-delimited JSON-RPC 2.0 requests from a `BufRead`,
   writes newline-delimited responses to a `Write`, until the input reaches EOF (MCP stdio
   transport -- messages are delimited by newlines and must not contain embedded newlines).
   A JSON-RPC *notification* (no `id`) gets no response, successful or not. A line that is not
   valid JSON gets a `-32700` parse-error response, never a panic.
2. `initialize` returns `protocolVersion` (`"2025-06-18"`), `capabilities: {"tools": {}}`,
   `serverInfo`. `tools/list` returns the six tool schemas (`ToolDef` table, JSON Schema
   `inputSchema` per tool). `tools/call { name, arguments }` dispatches to
   `es_script::tools::{validate, compile, estimate_cost, eval_compare, eval_run, hash_chain}`.
3. Two distinct error paths, matching the MCP spec's own split: a JSON-RPC protocol error
   (`-32602 Invalid params`) for a malformed `tools/call` -- an unknown tool name, or a
   `ToolError::BadParams` (missing/mis-typed required argument); an `isError: true` tool
   result for a `ToolError::Failed` -- the arguments parsed but what they named (bad TOML, an
   unparseable report) did not work. Neither path crashes the server.
4. Tool bodies call the same library functions the CLI already calls (`es_ir::serial`,
   `es_compile::CpuPlan`, `es_compile::budget::MemoryBudget`) and return JSON, never shell out
   to the `es` binary -- `es-script` (layer 11) cannot depend on `es`'s binary crate (layer 12)
   anyway.
5. `eval_run` always reports `{"status": "SKIPPED", "reason": ...}`: an actual run needs a
   `PhysicsBackend` and a `PolicyRuntime` (spec 9.6, spec 17.1), neither of which `es-script`
   links (its `Cargo.toml` deps are `es-core`, `es-ir`, `es-ir-types`, `es-compile`, `es-eval`,
   `es-data` only) -- mirrors `es eval run`'s own `SKIPPED` when a backend is unavailable
   (spec 1.4: never fake a result this process cannot produce).
6. `es mcp` (spec 2.5) runs `Server` on stdin/stdout, no arguments, no network socket
   (spec 25.1).

## oracle

```
cargo fmt -p es-script -p es --check
cargo clippy -p es-script -p es --all-targets -- -D warnings
cargo test -p es-script -p es
cargo xtask layering
cargo xtask check-spec-refs
```

## acceptance

- `initialize` -> `tools/list` -> `tools/call validate` (on a minimal Task IR) ->
  `tools/call estimate_cost` -> a malformed `tools/call` (missing a required argument), fed as
  one newline-delimited sequence to `Server::run`, produces five responses in order: three
  successful results, and the malformed call as a `-32602` error -- asserted by
  `crates/es-script/tests/mcp.rs`.
- `es mcp` spawned as a subprocess with piped stdin/stdout answers the same `initialize` /
  `tools/list` / `tools/call validate` sequence and exits 0 when stdin closes -- asserted by
  `crates/es/tests/cli.rs`.
- A line that fails to parse as JSON gets a `-32700` response and the server keeps serving
  the requests after it (never terminates the loop early).
- `validate` on a bad TOML document, and `compile { observation_toml, mode: "not-a-mode" }`,
  never panic: the former is a tool-execution error (`isError: true`), the latter a
  `-32602` protocol error (missing/mis-typed `mode` value is caught by argument validation,
  not left to the compiler to reject).

## forbidden

- `crates/es-script/src/generate.rs` -- a concurrent packet's file.
- Any new extension-point trait (`INV-17`): none is needed here.
- A network listener of any kind (spec 25.1): stdio only.
- Shelling out to the `es` binary from `es-script` (layer ordering forbids the dependency
  anyway; the tool bodies call the library crates directly).
- `HashMap` (this crate uses `BTreeMap` only, though the MCP tool tables here need neither).
- Editing `crates/es-ir/**`, `crates/es-compile/**`, `crates/es-eval/**`, `crates/es-data/**`
  -- other packets own them; this packet only calls their public API.
- Adding a new external crate dependency to `es-script`'s `Cargo.toml` beyond the ones already
  listed there (`es-core`, `es-ir`, `es-ir-types`, `es-compile`, `es-eval`, `es-data`, `serde`,
  `serde_json`, `blake3`, `thiserror`); IR-kind sniffing reuses `es_ir::serial`'s own
  `KindMismatch` diagnostic instead of adding a `toml` dependency for that alone.
