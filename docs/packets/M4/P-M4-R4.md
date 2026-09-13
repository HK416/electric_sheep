# P-M4-R4 — MCP request cap and generate provider hygiene

Spec: spec 14.5 (MCP interface, LLM generation), spec 25.1 (stdio-only transport, but the
process on the other end of the pipe is untrusted input). Closes should-fixes **S-3** and
**S-8** of `docs/reviews/M4.md`.

## context

```
crates/es-script/src/mcp.rs
crates/es-script/src/generate.rs
crates/es-script/tests/
crates/es/src/cmd/generate.rs
docs/api-notes/mcp.md
docs/design/mcp-interface.md
docs/design/llm-task-generation.md
docs/packets/M4/P-M4-R4.md
```

## spec

**S-3** (`es-script/src/mcp.rs:144`): `Server::run` read requests with
`std::io::BufRead::lines()`, which has no length cap (a huge `toml` argument was fully
buffered, decoded to a `String`, re-encoded as the `serde_json::Value`, then parsed as TOML --
multiples of its own size allocated before any check ran) and turns one non-UTF-8 byte into an
`io::Error` that `line?` propagates out of `run()`, ending the whole session, contradicting the
module doc's claim that a malformed request never crashes the loop.

Fix: `ServerConfig { max_request_bytes: usize }` (`Default` = `MAX_REQUEST_BYTES`, 16 MiB;
`Server::with_config` to override). `Server::run`'s new internal reader (`read_raw_line`) reads
raw bytes off the `BufRead` and tracks the running length as it streams in; once the running
total would cross the cap it stops appending to the buffer (dropping the rest) for the
remainder of that line, so memory use is bounded by the cap rather than by the line's actual
size. Once that line's terminating newline (or EOF) is reached:

- over cap -> `-32600 Invalid Request`, `id: null`, line discarded, loop continues.
- at or under cap -> decoded with `std::str::from_utf8`; a decode failure ->
  `-32700 Parse error`, `id: null`, loop continues; success -> the existing `handle_line` path
  (trim, empty-line skip, `serde_json::from_str`, dispatch) is unchanged.

**S-8** (`es-script/src/generate.rs:427`, `es/src/cmd/generate.rs:49,437`): three separate
gaps in `AnthropicProvider`/`es task generate`:

1. `ureq::post` used the default agent, which has a connect timeout but **no read timeout** --
   a stalled provider hung the round, and therefore the whole CLI invocation, forever.
2. `--rounds` parsed into a bare `u32` with no range check -- `0` can never produce a result,
   and there was no upper bound on how many provider calls (each itself unbounded in time,
   see 1) one invocation could make.
3. `format!("unexpected response shape: {resp}")` (and the two other `GenerateError::Provider`
   construction sites) put the entire provider response/error text into the message unbounded
   and unredacted.

Fix:

- `AnthropicProvider` now builds its own `ureq::Agent` via `AgentBuilder::timeout_connect` +
  `timeout_read`, both set to `AnthropicProvider::DEFAULT_TIMEOUT` (60 s), overridable through
  `with_timeout` / `with_endpoint` (the latter also lets a test point the client at a
  `TcpListener` instead of the real API). A `ureq` error whose source downcasts to
  `io::Error(ErrorKind::TimedOut)` (`ureq` normalizes both connect- and read-timeout failures
  to this) maps to a new `GenerateError::Timeout` variant, kept distinct from
  `GenerateError::Provider(String)` so a caller does not need to string-match a message to
  branch on "the network was slow" vs. any other provider failure.
- `es task generate --rounds` is checked against `1..=10` in `dispatch` before any provider is
  constructed; outside that range it is `CliError::Usage` (exit 2), matching every other bad
  argument in this CLI. `--timeout SECS` (default 60) is new, forwarded to
  `AnthropicProvider::with_timeout`; `--provider stdin` ignores it.
- Every `GenerateError::Provider` message goes through `redact_and_truncate`: any literal
  occurrence of the API key is replaced with `<redacted>` (defense in depth -- the key is not
  expected to appear in a response, but a proxy or error page could echo a request header
  back), then the result is capped at 256 bytes (`MAX_PROVIDER_ERROR_BYTES`) at a UTF-8 char
  boundary.

## oracle

```
cargo fmt -p es-script -p es --check
cargo clippy -p es-script -p es --all-targets --all-features -- -D warnings
cargo test -p es-script
cargo test -p es -- generate mcp
```

`AnthropicProvider` and its tests are behind the `llm` feature (off by default, per the crate's
existing TLS-free-default-build rule), so the S-8 timeout test also needs (run separately, not
part of the default gate above):

```
cargo test -p es-script --features llm
```

Written before the fix, confirmed to fail on the unfixed code:

- `crates/es-script/src/mcp.rs` `mod tests`:
  - `oversized_line_is_invalid_request_and_the_session_stays_alive` -- a 64 MiB line (the real
    `MAX_REQUEST_BYTES` default, not a shrunk-down cap) gets `-32600`, and a following
    `initialize` on the next line still gets answered.
  - `oversized_line_with_a_small_configured_cap_is_invalid_request` -- same shape at a small
    `ServerConfig::max_request_bytes` so the "discarded, loop alive" behavior is pinned without
    allocating tens of megabytes per test run.
  - `invalid_utf8_byte_is_a_parse_error_and_the_session_stays_alive` -- a lone `0xFF` byte gets
    `-32700`, and a following `ping` on the next line still gets answered.
- `crates/es-script/src/generate.rs` `mod llm_tests` (feature `llm`):
  - `a_stalled_provider_times_out_instead_of_hanging` -- a `std::net::TcpListener` that accepts
    the connection and never writes a response (the accepted socket is kept alive for the
    sleep, since dropping it immediately would reset the connection instead of timing out);
    `AnthropicProvider::with_endpoint` points at it with a 1 s timeout; asserts
    `Err(GenerateError::Timeout)`.
  - `redact_and_truncate_removes_the_secret_and_caps_the_length`.
- `crates/es/src/cmd/generate.rs` `mod tests`:
  - `rounds_outside_one_to_ten_is_a_usage_error` -- `--rounds 0` and `--rounds 11` both return
    `Err(CliError::Usage(_))` (exit 2 at the `main` level) without touching a provider.
  - `rounds_at_the_boundary_passes_the_usage_check` -- `--rounds 1` and `--rounds 10` are not
    rejected by the rounds check itself (an unknown `--provider` name is used so `dispatch`
    still returns `Usage`, but from the provider-name branch, confirming the rounds check let
    them through without going near stdin or the network).

These CLI-level checks are unit tests inside `crates/es/src/cmd/generate.rs`, calling
`dispatch()` directly, rather than integration tests spawning the `es` binary from
`crates/es/tests/cli.rs` -- that file is outside this packet's declared `context` and was
concurrently being edited by another packet's work while this one was in flight.

## acceptance

- Both oracle commands above are clean; `cargo test -p es-script --features llm` passes.
- `es-script` compiles and its existing test suite passes both with and without `--features
  llm` (the default build stays TLS-free, unchanged from before this packet).
- No public signature of an existing function changed; `Server::new()`, `Server::run`,
  `AnthropicProvider::from_env`, and `AnthropicProvider::call` all keep their old signatures.
  New public surface only: `ServerConfig`, `MAX_REQUEST_BYTES`, `Server::with_config`,
  `GenerateError::Timeout`, `AnthropicProvider::{DEFAULT_TIMEOUT, with_timeout,
  with_endpoint}`, `es task generate --timeout`.
- `docs/api-notes/mcp.md`, `docs/design/mcp-interface.md`, `docs/design/llm-task-generation.md`
  describe the new cap/timeout/truncation behavior; no `docs/ARCHITECTURE.*` change needed
  (nothing here changes a pinned type signature, error code list, or invariant beyond adding
  two already-reserved JSON-RPC codes to the table this crate uses).

## forbidden

Anything outside `context` -- every other M4 review finding (S-1, S-2, S-4 through S-7, S-9,
S-10) belongs to its own packet; in particular `es/src/cmd/import.rs` (S-9) and
`crates/es/tests/cli.rs` (shared integration-test file, actively being edited by other
in-flight packets) are not touched here. Changing `SafetyPlane::validate`'s signature (not
applicable to this packet, restated per `AGENTS.md`). Changing the MCP tool set, `tools/list`
shape, or the six existing tool schemas. Changing `AnthropicProvider`'s wire format (model,
headers, request/response body shape) beyond the agent-level timeout. Raising or removing the
`--rounds` cap, or making `MAX_REQUEST_BYTES` unbounded.
