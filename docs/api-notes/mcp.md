# Model Context Protocol -- pinned surface for `es_script::mcp`

What `crates/es-script/src/mcp.rs` implements against, so a protocol revision is a diff
against this file rather than an archaeology session. No MCP SDK crate is used (this crate is
layer 11 and hand-rolls the JSON-RPC framing over `serde_json`) -- companion to
`docs/api-notes/torch.md` and `docs/api-notes/mujoco.md`, same shape, same reason.

Spec: spec 14.5 (line ~1759, the MCP interface requirement).

## Version

| | |
|---|---|
| verified against | protocol version **`2025-06-18`**, fetched from <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports> and <https://modelcontextprotocol.io/specification/2025-06-18/server/tools> on 2026-09-13 |
| MCP SDK crate | **none** -- hand-rolled JSON-RPC 2.0 over `serde_json`, per the work packet |
| Live client interop | **unverified** -- checked against the written spec pages only, never against a real MCP client (Claude Desktop, an IDE's MCP integration, `@modelcontextprotocol/inspector`); `Target / Status: unverified` per spec 12.4's convention for anything not measured |

## Transport -- stdio framing

Confirmed from the spec's "Transports" page:

- JSON-RPC messages **MUST** be UTF-8 encoded.
- stdio: the server reads JSON-RPC messages from stdin and writes them to stdout.
- **Messages are delimited by newlines, and MUST NOT contain embedded newlines.** This is the
  one framing rule that matters for `Server::run`: read one line, parse one JSON-RPC message,
  write one line back. There is no `Content-Length:` header framing on stdio -- that is
  Streamable HTTP's problem, not implemented here (spec 25.1: this server is stdio-only, never
  a network listener).
- The server MAY write to stderr for logging; `es_script::mcp` does not do this yet (nothing
  here needs it).
- The server MUST NOT write anything to stdout that is not a valid MCP message.

## `initialize`

Request carries `protocolVersion`, `capabilities`, `clientInfo` (not otherwise checked by this
server -- `Server::dispatch` ignores `params` for `initialize` entirely, since a v1
implementation has exactly one behavior regardless of what the client asks for).

Response (this server's exact shape, `crates/es-script/src/mcp.rs`):

```json
{
  "protocolVersion": "2025-06-18",
  "capabilities": {"tools": {}},
  "serverInfo": {"name": "electric-sheep", "version": "<CARGO_PKG_VERSION>"}
}
```

`capabilities.tools` is the empty object, not `{"listChanged": true}`: the tool set never
changes at runtime, so `notifications/tools/list_changed` is never sent and `listChanged` is
left unset (falsy) rather than promising a notification that never comes.

## `tools/list`

No pagination (`cursor`/`nextCursor`) -- six tools, one response, confirmed from the spec's
"Listing Tools" section. Each tool: `name`, `description`, `inputSchema` (JSON Schema, `type:
"object"` with `properties`/`required`). No `outputSchema`, no `title`, no `annotations` --
optional per spec, not produced here (see `docs/design/mcp-interface.md` "Ceiling").

## `tools/call`

Request: `{"name": "<tool>", "arguments": {...}}`. Response on success:

```json
{"content": [{"type": "text", "text": "<JSON-encoded tool result>"}], "isError": false}
```

`structuredContent` (the spec's alternative, machine-typed result field) is not produced --
`content[0].text` is the entire JSON result as a string, decoded by the caller. On a
tool-execution failure: the same shape with `isError: true` and `text` set to the error
message.

Confirmed error-code split from the spec's "Error Handling" section: unknown tool / invalid
arguments are **protocol errors** (a JSON-RPC `error` object, this server always uses
`-32602 Invalid params`); a tool that ran but failed reports `isError: true` in a normal
`result`, never a JSON-RPC error. `es_script::tools::ToolError::{BadParams, Failed}` map to
exactly this split.

## JSON-RPC error codes actually used

| code | meaning | when `es_script::mcp` sends it |
|---|---|---|
| `-32700` | Parse error | a stdio line is not valid JSON, or is not valid UTF-8 at all |
| `-32600` | Invalid Request | a stdio line is longer than `ServerConfig::max_request_bytes` |
| `-32601` | Method not found | `dispatch`'s method is none of `initialize`/`ping`/`tools/list`/`tools/call` |
| `-32602` | Invalid params | unknown tool name, or a `ToolError::BadParams` from a tool body |

These four are standard JSON-RPC 2.0 reserved codes, not MCP-specific; MCP itself defines no
additional codes beyond what its "Error Handling" section labels a protocol error.

## Request size cap

M4 review finding **S-3**: `Server::run` used to read lines with `std::io::BufRead::lines()`,
which has no length limit (an attacker-or-bug-controlled line is fully buffered, converted to a
`String`, and only then handed to `serde_json` -- for `validate`'s `toml` argument, allocated
*again* as the decoded JSON string, then parsed as TOML, so one huge line cost multiples of its
own size before anything checked it) and turns a non-UTF-8 byte into an `io::Error` that
`line?` propagates straight out of `run()`, ending the whole session on one bad byte.

Fixed by reading raw bytes with a length cap before any `String`/JSON conversion is attempted:

- `ServerConfig::max_request_bytes` (default [`MAX_REQUEST_BYTES`], 16 MiB) bounds one line.
  `Server::run`'s internal reader tracks the running length as bytes stream in from the
  `BufRead` and, once the running total would exceed the cap, stops appending to the buffer
  (dropping what it already has) for the rest of that line -- so the memory cost of an
  oversized line is bounded by the cap, not by how large the line actually is. Once the line's
  terminating newline (or EOF) is reached, the server writes `-32600 Invalid Request` with
  `id: null` and moves on to the next line; nothing is parsed as JSON.
- A line at or under the cap is decoded with `std::str::from_utf8` (not `String::from_utf8`,
  which would consume the buffer on success but still needed a fallible path) rather than
  relying on `io::Read`/`io::BufRead`'s own (fallible, session-ending) UTF-8 conversion. A
  decode failure gets `-32700 Parse error` with `id: null` -- same shape as a line that decodes
  fine but fails `serde_json::from_str`, just one step earlier -- and the loop keeps running.

Both faults are exercised in `crates/es-script/src/mcp.rs`'s `mod tests`: a line built at the
real 16 MiB default (not a shrunk-down cap) confirms the size check fires at the size the
review actually flagged, and a lone `0xFF` byte followed by a valid `initialize` confirms the
loop survives invalid UTF-8 and answers the next request normally.
