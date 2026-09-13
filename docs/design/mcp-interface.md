# MCP interface (spec 14.5)

Spec 14.5: existing tools (LLM task-generation loops, editors, evolutionary-search harnesses)
should be able to use Electric Sheep as a backend without Electric Sheep growing an
orchestrator of its own (spec 0.5 -- not cited by number here since it is a design principle,
not a section this doc pins against). The interface it names is small on purpose: `validate`,
`compile`, `estimate_cost`, `eval`.

## Shape

`es_script::mcp::Server` speaks JSON-RPC 2.0 over stdio, newline-delimited (one message per
line, no embedded newlines -- the MCP stdio transport's own framing rule, `docs/api-notes/
mcp.md`). Three methods: `initialize`, `tools/list`, `tools/call`. No resources, no prompts, no
sampling, no `notifications/tools/list_changed` -- the tool set here is fixed at compile time,
so there is nothing to notify about.

## Six tools, not four

Spec 14.5 names four (`validate` / `compile` / `estimate_cost` / `eval`); this packet exposes
six, splitting `eval` in two:

- `eval` -- compares two already-produced `EvaluationReport` JSON documents (spec 10.5
  `es eval compare`). Pure data in, pure data out: no backend needed, so this is the literal
  `eval` of spec 14.5.
- `eval_run` -- would run an Evaluation IR against a policy bundle (spec 10.5 `es eval run`).
  Always reports `{"status": "SKIPPED", "reason": ...}`: an actual run needs a `PhysicsBackend`
  and a `PolicyRuntime` (spec 9.6, spec 17.1), and `es-script` (layer 11) links neither --
  adding them here would pull `es-physics-backend` and `es-policy` into a crate whose only job
  is to translate IR/report JSON, and would make this server capable of long-running,
  resource-heavy work over a protocol with no cancellation or progress reporting built in
  (out of scope for this packet; spec 1.4 also forbids faking a result this process cannot
  produce). `hash_chain` is the sixth: spec 14.5 does not name it, but every one of the other
  five tools' output already carries a `*_hash`, and an external caller comparing runs needs
  the spec 5.3 hash chain slots without re-deriving them by hand.

## Why not shell out to `es`

`crates/es/src/cmd/{ir,task,bench,eval}.rs` already implement `validate`/`compile`/
`estimate_cost`/`eval compare` as CLI commands that print tables. `es-script` cannot depend on
the `es` binary crate (layer 11 is below layer 12), so `crates/es-script/src/tools.rs` calls the
same library functions those commands call (`es_ir::serial`, `es_compile::CpuPlan`,
`es_compile::budget::MemoryBudget`, `es_ir::evaluation::EvaluationReport`) directly and returns
JSON instead of a printed table. Shelling out to a subprocess was considered and rejected: it
would need an `es` binary on `PATH` (a hidden runtime dependency an MCP client would not expect
from a library crate), double the process count per call, and reintroduce exactly the parsing
problem library functions exist to avoid.

## Error handling

Two paths, matching the MCP spec's own split between "Protocol Errors" and "Tool Execution
Errors":

- `ToolError::BadParams` -> JSON-RPC `-32602 Invalid params`. The `tools/call` arguments
  themselves are wrong shape: an unknown tool name, a missing required field, a value outside
  its enum (`mode` not `"debug"`/`"release"`, `precision` not `"f32"`/`"f16"`).
- `ToolError::Failed` -> a successful JSON-RPC response whose `result.isError` is `true`. The
  arguments parsed, but what they named did not work: TOML that fails to parse, an
  `EvaluationReport` JSON that fails to deserialize, an Observation IR that fails to compile.

Neither path panics. A line that is not JSON at all gets `-32700 Parse error` with `id: null`
(JSON-RPC 2.0's own rule for an error with no request to attach to) and the server keeps
reading the next line.

## Ceiling

- No pagination on `tools/list` (`cursor`/`nextCursor`): six tools fit one response. Add it if
  the tool count ever grows past what a client reasonably wants in one call.
- No `outputSchema` per tool, only `inputSchema`: every tool's JSON result shape is documented
  in this file and `crates/es-script/src/tools.rs`'s doc comments, not machine-checked against
  a schema. Add `outputSchema` if a client needs to validate results without reading the source.
- `eval`'s comparison table has no significance test (the CLI's Welch-t-test, spec 10.5): a
  duplicate would have to live in `es-script` since `es` cannot be a dependency of it, and spec
  14.5 only asks for "the compare table JSON". Move the test into a library crate both `es` and
  `es-script` can call if an MCP caller ever needs it.
