# Telemetry protocol — wire shape and the M1 loopback transport

Design note for `es-telemetry::protocol` and `es-telemetry::transport`. Spec: spec 23.1
(backend-neutral, editor is a client), spec 23.2–23.4 (layer graph view, performance gate),
spec 25.1 (security), spec 25.3 (API versioning), spec 12.4 (performance metric set). This is
the M1 prerequisite design doc spec 28.3 names for the W8 packet
(`docs/packets/M1/W8-telemetry-transport.md`).

## 1. Why this exists

Spec 23.1: **"the editor does not host training — it is a client of a running process."** That
only works if there is something to connect *to*: a process (the simulator, the embedded
runtime, or a bare Python training loop) opens a socket and streams state; any number of
readers — the egui editor, a CLI, a logging script — attach to it without changing the
producer's code path. Spec 23.1 also insists the protocol is not Electric Sheep-specific
("a thin Python adapter... is the wedge for adoption"), so the wire schema
(`es_telemetry::protocol`) carries no ES-only concept: a stream id, a tick, a payload.

This document covers two layers that are deliberately separate:

- **the schema** (`protocol.rs`): what a message *is*, independent of how it travels;
- **the transport** (`transport.rs`, added in W8): a `std::net::TcpListener`/`TcpStream`
  implementation good enough to exercise the schema end to end on localhost. It is explicitly
  not the final transport — see §6.

## 2. Message flow

```
   client (editor / lerobot-adapter / CLI)              server (sim / embedded runtime)
            |                                                       |
            |  TCP connect ------------------------------------->   |
            |                                                       |
            |  Hello { versions_supported, token, client } ----->   |  negotiate() + token check
            |                                                       |
            |  <----------------------- HelloAck { version,         |  (accepted)
            |                            session_id,                |
            |                            execution_hash }           |
            |                    or                                 |
            |  <----------------------- Bye { reason }               |  (rejected, socket closes)
            |                                                       |
            |  Subscribe { streams: [..] } ----------------------->  |  replaces subscription set
            |                                                       |
            |  <====================================================|  Frame, Frame, Frame, ...
            |                                                       |  (only for subscribed streams)
            |  Subscribe { streams: [..] } ----------------------->  |  (may re-subscribe any time)
            |                                                       |
            |  Bye { reason } ------------------------------------>  |  (either side may end it)
```

One TCP connection is one session. After the handshake the relationship is asymmetric: the
client's only outbound messages are `Subscribe` (as often as its view changes) and `Bye`; the
server's only outbound messages are `Frame`s (fanned out to whichever clients are subscribed to
that frame's stream) — see `crates/es-telemetry/src/transport.rs`'s module doc for exactly
which thread writes what, since two threads writing the same socket unsynchronized would
interleave bytes.

## 3. Framing

`protocol::encode`/`decode`: `codec (1 byte) | body_len (u32, LE) | body`. `codec = 0` is JSON
today; a binary codec is an additive variant, not a wire break, so `Codec` staying an enum
matters more than JSON being fast. `decode_with_max` rejects an oversized `body_len` before
touching the bytes, so a corrupt length prefix cannot be used to force a large allocation.

This is intentionally the same framing whether the transport is the M1 TCP shim or a later
QUIC stream: `transport.rs` only adds the "read until one `decode` succeeds, then loop" logic
around a `TcpStream`; nothing about the frame bytes assumes TCP.

## 4. Auth (spec 25.1)

`Server::bind(addr, token: Option<String>)`. When `token` is `Some`, every `Hello.token` must
equal it exactly or the server replies `Bye { reason: "missing or invalid token" }` and closes
the connection without registering the client. When `token` is `None`, no check runs — this is
the local-development shape (`es --check-deps` running the editor against a local sim with no
token configured).

**What this is not**: the token travels in clear text inside the JSON body, and there is no
channel encryption. That is fine on `127.0.0.1` (the spec 25.1 default bind) between processes
owned by the same user, and not fine across an untrusted network. TLS is spec 25.1's other
requirement for this surface and is not implemented here — see §6.

## 5. Versioning (spec 25.3)

`negotiate(client_versions, server_versions) -> Option<u32>` (unchanged from before this
packet) picks the highest version both sides list. The server offers
`{PROTOCOL_VERSION, PROTOCOL_VERSION - 1}` — "N and N-1" — so that an editor built against the
previous release still connects to a newer runtime during a rollout; when `PROTOCOL_VERSION` is
`1` there is no N-1 yet and the server offers just `{1}`. No shared version is a handshake
rejection (`Bye`, reason names both lists), never a silent downgrade or a panic.

`Message`, `Frame`, `Payload` and `PerfMetrics` themselves have no version field: spec 25.3
places the version *in the handshake*, not per-message, because a session negotiates once and
then speaks one dialect for its lifetime. A future incompatible schema change bumps
`PROTOCOL_VERSION`; an additive one (a new `Payload` variant, a new optional field) does not
need to, by the same reasoning `serde`'s `Option` fields already buy the manifest format in
`docs/design/policy-bundle.md`.

## 6. Backpressure

Spec 23.4's gate 9 — telemetry overhead on the training/sim loop must stay under 1% — is a
statement about the *producer*, not the network. `Server::publish` must return in bounded time
regardless of what any subscriber is doing:

- each client has its own bounded queue (`CLIENT_QUEUE_CAPACITY = 16` frames);
- `publish` does a non-blocking `try_send`; a full queue means the frame is dropped and counted
  in that client's `ClientStats::dropped`, never waited for;
- a slow client — one that stopped reading, or is behind on a busy image stream — only ever
  loses its own data. It cannot slow the simulation, other subscribers, or even its own
  connection's liveness (the queue drains independently on that client's writer thread).

Sixteen is a guess sized for "a few ticks of slack," not a measurement — the M1 packet's test
(`a_slow_client_drops_frames_without_blocking_the_publisher`) only proves the *shape* of the
policy (drops happen, the producer never blocks), not that 16 is the right depth for a real
graph-view session. Tuning that number, and answering gate 9's actual "< 1%" question, needs a
running editor against a running training loop — outside what a unit test can produce.
**`Target / Status: unverified`** for the < 1% figure itself; this document only establishes
that the mechanism which would let it be true (non-blocking fan-out) is in place.

## 7. What the Python adapter needs

Spec 23.1's backend-neutral requirement means a LeRobot/Isaac Lab/Newton training loop should
be able to emit frames without linking any Rust code. Everything in `protocol.rs`'s crate-level
doc comment already describes the JSON body shape; W8 adds one more requirement on top for a
Python client that wants to *push* into a running `Server` rather than just being parsed by a
downstream Rust reader:

1. Open a TCP connection to the address the runtime printed (or was configured with).
2. Send one length-prefixed `Hello` (`{"type": "hello", "versions_supported": [1], "token":
   null_or_the_configured_token, "client": "lerobot-adapter"}`), using the 5-byte
   `codec | len` header from §3 (`codec = 0`).
3. Read one length-prefixed reply. `"type": "hello_ack"` means proceed; `"type": "bye"` means
   the connection is closing and `reason` says why (bad token or version mismatch — retry with
   a corrected `Hello` on a fresh connection, since the server does not keep listening on a
   rejected socket).
4. Send `Frame` messages directly — a producer does not need to `Subscribe` to anything; only a
   *reader* of a `Server`'s fan-out subscribes. A training loop that only ever pushes scalars
   and `Metrics` payloads never needs the tensor/image encoding at all.

No protobuf, no schema compiler, no dependency beyond a socket and a JSON encoder any of those
frameworks already have on hand.

## 8. Known ceilings (this packet)

- **No shutdown handshake for `Server`.** `Server::bind`'s accept thread runs for the process's
  life; there is no `Server::close` that joins it. Fine for an embedded runtime or simulator
  process that lives exactly as long as its listener should; a caller that needs to rebind the
  same address later needs a shutdown flag plus a self-connect wakeup, not built here.
- **No TLS, no QUIC.** Spec 23.4 wants QUIC for the remote editor path and spec 25.1 wants TLS;
  neither exists in `es-telemetry::transport`. Both are `es-transport` (layer 11, per
  `CLAUDE.md` the only crate allowed to link CUDA/HIP) work, later — this transport is the
  loopback/trusted-network shim that lets the M1 gates exercise the schema and the backpressure
  policy before that crate exists.
- **Fixed queue depth, not a budget.** `CLIENT_QUEUE_CAPACITY` is a constant, not the
  "20 µs/step sampling budget" spec 23.3 describes for state streaming. Turning that budget
  into an actual rate limiter (as opposed to a depth limiter) is later work once there is a
  real producer to measure against.
- **No reconnect/resume.** A dropped TCP connection loses the client's subscription state; it
  reconnects and re-subscribes from scratch. There is no `session_id`-keyed resume.
