# P-M1-R5 — transport handshake hardening (`es-telemetry::transport`)

M1 review follow-up (`docs/reviews/M1.md`, Should-fix): the auth token was compared with a
variable-time `String` equality, the handshake read had no timeout so a silent peer could pin a
server thread forever, and there was no cap on concurrent connections (a thread per accepted
TCP connection, unbounded). Spec: spec 25.1 (security: token auth, resource limits on an
untrusted peer), spec 25.3 (versioning, unchanged by this packet). Design note:
`docs/design/telemetry-protocol.md`.

## context

```
crates/es-telemetry/src/transport.rs
docs/design/telemetry-protocol.md
docs/packets/M1/P-M1-R5.md
```

Tests live inside `transport.rs`'s own `#[cfg(test)] mod tests` — no new test file.

## spec

- **Constant-time token comparison.** `reject_reason` no longer compares `hello.token` to the
  configured token via `PartialEq` on `String`/`Option<String>` (variable-time, short-circuits
  on the first differing byte). A new `ct_eq(a: &[u8], b: &[u8]) -> bool` folds every byte
  position up to `max(a.len(), b.len())` into one `u8` accumulator with no early return, and a
  length mismatch is folded in up front rather than compared with `!=`. No new dependency
  (`subtle` or otherwise) — the whole function is five lines.
- **Handshake timeout.** `Server::bind_with(addr, token, cfg: ServerConfig)` is a new
  constructor; `Server::bind` is unchanged in signature and now calls
  `Self::bind_with(addr, token, ServerConfig::default())`. `ServerConfig::handshake_timeout`
  (default `HANDSHAKE_TIMEOUT = Duration::from_secs(5)`) is applied with
  `TcpStream::set_read_timeout` before `serve_client` reads the client's `Hello`; a peer that
  never completes one is dropped (the thread returns) once the timeout elapses. The timeout is
  cleared (`set_read_timeout(None)`) immediately after a successful `Hello` read, so a
  long-lived, authenticated session's later reads (the `Subscribe` loop) block normally again.
- **Connection cap.** `ServerConfig::max_clients` (default `DEFAULT_MAX_CLIENTS = 64`) is
  checked in `serve_client`, after the existing version/token checks (so a bad token still gets
  its own specific `Bye` reason) and before the client is registered in the shared map: a
  connection arriving once the server already holds `max_clients` clients gets
  `Bye { reason: "too many clients" }` and is never added to `clients`.
- ponytail: the cap check and the registration insert are not one atomic step, so two
  connections arriving at exactly the capacity boundary can both pass the check before either
  registers, briefly exceeding `max_clients` by one. Acceptable for a loopback/trusted-network
  shim (`docs/design/telemetry-protocol.md` §6/§8 already documents this transport is not the
  final one); a single mutex held across "check and insert" is the fix if this ever needs to be
  exact.

## oracle

```
cargo fmt -p es-telemetry --check
cargo clippy -p es-telemetry --all-targets -- -D warnings
cargo test -p es-telemetry
```

## acceptance

- A client that connects and sends nothing observes the server close the connection (read
  returns EOF) within `handshake_timeout + 1s`; the test uses a short configured timeout via
  `Server::bind_with` to stay fast.
- A wrong token (`a_bad_token_is_rejected`, unchanged) and a token of a different length than
  the configured one (`a_token_of_different_length_is_still_rejected`, new — exercises `ct_eq`'s
  length-mismatch path) are both rejected with `TransportError::Rejected`.
- The 65th concurrent client (with the default `max_clients = 64`) is refused with
  `TransportError::Rejected { reason }` containing `"too many"`, while the first 64 all
  handshake successfully and are held open for the duration of the check.
- Every existing `transport::tests` case (no-token handshake, matching-token handshake,
  unsupported version, publish/subscribe fan-out, slow-client drop counting, close/reap) passes
  unchanged.

## forbidden

- `crates/es-telemetry/src/protocol.rs`, `crates/es-telemetry/src/ring.rs`,
  `crates/es-telemetry/src/lib.rs`, `crates/es-runtime-embedded/*` — other packets' scope; in
  particular `ServerConfig` is reachable as `es_telemetry::transport::ServerConfig` and is
  deliberately not re-exported at the crate root, since that touches `lib.rs`.
- `crates/es-compile/*`, `docs/design/policy-bundle.md` — P-M1-R4's scope.
- TLS, real authentication beyond a constant-time-compared shared token, QUIC, and any claim
  that spec 28.7 gate 9's `< 1%` overhead is *measured* by this packet — it is not, and stays
  `Target / Status: unverified`.
- Any new external dependency and any new trait (INV-17).
- Committing.
