# W8 — telemetry transport (`es-core::ring`, `es-telemetry::transport`)

Spec: §23.1–23.2 (editor is a client of a running process; backend-neutral protocol), §25.1
(security: token auth, TLS later, localhost-default bind), §25.3 (API versioning: handshake
negotiation, N-1 support), §28.7 gate 9 (telemetry + graph view overhead < 1%, M1 —
unverifiable from a unit test, reported below as `Target / Status: unverified`). Design note:
`docs/design/telemetry-protocol.md` (spec 28.3's M1 prerequisite doc).

## context

```
crates/es-core/src/ring.rs
crates/es-core/src/lib.rs
crates/es-telemetry/src/ring.rs
crates/es-telemetry/src/transport.rs
crates/es-telemetry/src/lib.rs
crates/es-telemetry/Cargo.toml
crates/es-runtime-embedded/src/lib.rs
crates/es-runtime-embedded/src/runtime.rs
docs/design/telemetry-protocol.md
docs/packets/M1/W8-telemetry-transport.md
```

(`crates/es-runtime-embedded/src/ring.rs` is deleted by this packet, not edited.)

## spec

- **Ring buffer, moved not duplicated.** `RingBuffer<T>` (sequence numbers, `push`,
  `iter_newest`, `drain_since`, `dropped`) moves verbatim into `es_core::ring` (layer 1), the
  one place both `es-telemetry` (layer 10) and `es-runtime-embedded` (layer 9) can reach
  without either crossing the other's layer boundary (spec 4.2). `es_telemetry::ring` becomes
  `pub use es_core::ring::RingBuffer;` so no existing caller of `es_telemetry::ring::RingBuffer`
  changes. `es-runtime-embedded`'s private `TelemetryRing` copy and its `ponytail:` note about
  the duplication are deleted; `EmbeddedRuntime` now holds a `RingBuffer<TickRecord>` (the
  `TickRecord` type itself stays in `es-runtime-embedded`, since it is specific to that crate's
  tick loop).
- **`es_telemetry::transport::Server`** — `bind(addr: SocketAddr, token: Option<String>) ->
  io::Result<Server>` opens a `std::net::TcpListener` and runs its accept loop on a background
  thread; each accepted client gets its own reader/writer thread pair. `publish(&self, frame:
  Frame)` fans a frame out to every client currently subscribed to `frame.stream`, using a
  bounded (`CLIENT_QUEUE_CAPACITY = 16`) per-client channel and a non-blocking `try_send` — a
  full queue drops the frame and counts it, the producer never waits. `stats() -> ServerStats`
  reports `{ sent, dropped }` per client, keyed by session id.
- **`es_telemetry::transport::Client`** — `connect(addr, token, client_name) ->
  Result<Client, TransportError>` performs the `Hello`/`HelloAck` handshake (or reads back a
  `Bye { reason }` and returns `TransportError::Rejected`), `subscribe(streams)` sends a new
  subscription set, `recv()` blocks for the next message, `try_recv()` returns
  `TransportError::WouldBlock` instead of blocking, `close()` sends a `Bye` and ends the
  session.
- **Handshake and auth (spec 25.1, 25.3).** A client must send `Hello` before anything else.
  The server rejects (with `Bye { reason }`, then closes) when: the client's
  `versions_supported` shares nothing with the server's own `{PROTOCOL_VERSION,
  PROTOCOL_VERSION - 1}` (N-1, spec 25.3's `negotiate` — already existed, reused unchanged); or
  the server has a `token` and the client's `Hello.token` does not match it exactly. On
  success the server replies `HelloAck { version, session_id, execution_hash: None }` (this
  layer does not carry a hash chain, so the field starts empty; a caller that has one can wrap
  it in). Binding `127.0.0.1:0` is the recommended default (spec 25.1's localhost-default) —
  the function does not hardcode it, since a caller may need a fixed port.
- **Framing.** Reuses `crate::protocol::{encode, decode}` (length-prefixed JSON) unchanged;
  `transport` only adds the loop that reads/writes those frames off a `TcpStream`.
- **No TLS, no QUIC (spec 23.4, 25.1).** This is a loopback/trusted-network shim for M1;
  `Target / Status: unverified` beyond that. `es-transport` (layer 11) is where QUIC + TLS land
  in a later packet — documented in `docs/design/telemetry-protocol.md`, not implemented here.

Constraints: English only, `BTreeMap` only (no `HashMap`), no new external dependency (`std`
`TcpListener`/`TcpStream` only), no new trait (INV-17), ≤ ~500 new lines, do not edit the root
`Cargo.toml`.

## oracle

```
cargo fmt -p es-core -p es-telemetry -p es-runtime-embedded --check
cargo clippy -p es-core -p es-telemetry -p es-runtime-embedded --all-targets -- -D warnings
cargo test -p es-core -p es-telemetry -p es-runtime-embedded
cargo xtask layering
cargo xtask context-budget
```

## acceptance

- `es-core::ring::RingBuffer<T>` carries every existing test (push/wrap/drop counting,
  `drain_since` clamping, the allocation-free `push` assertion) unchanged; `es-telemetry` and
  `es-runtime-embedded` compile and test green against it with no behavior change at either
  call site.
- Loopback handshake: connecting with a shared (or absent, when the server has none) token and
  a supported version succeeds and returns a `HelloAck`; a wrong or missing token is rejected
  with `Bye`; an unsupported version is rejected with `Bye` naming the mismatch.
- Publishing 1000 frames to a client whose queue (16 deep) is never drained yields
  `stats().dropped > 0` for that client, and the 1000-frame publish loop completes in well
  under a second — i.e. `publish` did not block on the full queue.
- `es-runtime-embedded`'s existing test suite (bundle round-trip, replan cadence, chunk-reuse
  allocation-free tick, telemetry length) passes unchanged against the new ring type.

## forbidden

- `crates/es-editor`, `crates/es/src/cmd/backend.rs`, and any doc translation — other packets'
  scope.
- The root `Cargo.toml` and any new external dependency (QUIC/zenoh is a later, `es-transport`
  packet — noted, not built).
- TLS, real authentication beyond a plaintext shared token, and anything claiming spec 28.7
  gate 9's `< 1%` overhead number is *measured* — it is not, and is reported as
  `Target / Status: unverified`.
- Committing.
