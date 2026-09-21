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

The comparison itself is `transport::ct_eq`, a constant-time fold over `max(a.len(), b.len())`
bytes with no early return (no `subtle` dependency — five lines), not `PartialEq` on `String`:
a naive `!=` compare stops at the first differing byte, which leaks how much of a guessed token
was correct through timing. A length mismatch is folded into the same accumulator up front
rather than short-circuited with `!=`.

Two more resource limits sit in front of the token check, both on `Server::bind_with(addr,
token, cfg: ServerConfig)` (`Server::bind` calls it with `ServerConfig::default()`):

- **Connection cap** (`ServerConfig::max_clients`, default `DEFAULT_MAX_CLIENTS = 64`): checked
  in the accept loop itself, before a thread is spawned for the connection at all — so before
  `Hello` is even read, which means before the version/token checks. `in_flight =
  handshaking.load() + clients.len()` counts every connection currently occupying a "slot",
  whether it is mid-handshake or already registered; a connection arriving once that is
  `>= max_clients` gets `Bye { reason: "too many clients" }`, written synchronously by the
  accept loop thread, and its socket is dropped (closed) without ever spawning a thread or
  reading a byte from it. This is a behavior change from the original shape (which checked the
  cap inside the per-connection thread, after the token/version checks): an over-cap
  connection's `Hello` — even one with a bad token — is never read, so it only ever gets "too
  many clients", not a token- or version-specific reason. The accept loop is single-threaded,
  so this check-then-increment has no race with itself — the earlier `ponytail` note about two
  connections racing past the boundary no longer applies; the sequencing here (not a held
  mutex) is what makes it exact.
- **Handshake timeout** (`ServerConfig::handshake_timeout`, default `HANDSHAKE_TIMEOUT = 5s`):
  an absolute deadline, not a per-read one. The accept loop records `Instant::now() +
  handshake_timeout` when the connection is admitted; `read_message_until` re-derives the
  *remaining* time before every socket read that would block and uses that as the read's
  timeout, erroring out once no time is left. A peer dribbling a single byte every few seconds
  — each individual read comfortably inside its own per-read timeout — is still cut off once
  the total elapsed time crosses the deadline, which a single `set_read_timeout(Some(duration))`
  applied once (the original shape) could not do: that call bounds one read, not the sum of
  many partial ones. Cleared (`set_read_timeout(None)`) immediately after a successful `Hello`
  read, so a session's later reads (the `Subscribe` loop) block normally for as long as the
  connection lives.
- **Observability**: `Server::stats()` reports `handshaking` (connections admitted past the cap
  check but not yet past the handshake) and `threads_live` (every `serve_client` thread
  currently running, handshaking or registered) alongside the per-client `sent`/`dropped`
  counters, so a caller — or a test — can see the cap actually holding rather than trusting it
  blindly. Both are shared `AtomicUsize` counters; `threads_live` is decremented by an RAII
  guard on thread exit so every path (clean return, an early `?`, a panic unwind) accounts for
  itself once.

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

## 9. Producers in this repository (packet M7/E4)

§7 says what a *foreign* producer must send. This is what the one in this repository sends:
`es eval run --telemetry <addr> [--telemetry-token <t>] [--telemetry-image-every <N>]` binds a
`Server` before it opens the bundle, the scene or either Python interpreter — so a viewer that
attaches on the printed address is subscribed before the first cell — and publishes on four
streams:

| Stream | Payload | When |
|---|---|---|
| `1` | `Event { kind: "cell.begin" \| "cell.end" \| "suite.end", fields }` | `cell.begin` at each episode's start (`cell`, `suite`, `seed`, `episode`), `cell.end` once its files are written (`cell`, `outcome`, `steps`, `frames`, `traj`), `suite.end` once the §10.1 row is measured (`suite`, `n_episodes`, `metric.<name>` as the `MetricValue`'s own JSON) |
| `2` | `Scalars([frame, tick, source, violation bits])` | every control tick that captured an observation |
| `3` | `Metrics(PerfMetrics)` | at each `cell.end`: `actions_per_sec`, `policy_inferences_per_sec`, `chunk_underrun_rate`. Everything else is `None` — this run does not measure end-to-end latency or GPU memory, and §12.4 would rather have a hole than a zero |
| `4` | `Image { format: "rgb8" }` | every `--telemetry-image-every N` ticks; `0` (the default) publishes none |

Stream ids are **data, not schema**: nothing in `protocol.rs` names them, a consumer that does
not recognize one ignores it (`es_editor::model::live_run` does exactly that), and a second
producer choosing other numbers breaks nothing here. `Frame::stream` is a `u32`.

Stream 2 is one `es_eval::runner::StepEvent` per sample, flattened: the frame index inside the
cell, the `PhysTick` it ran at, the action's source as `es_data::ActionSourceCode`'s number
(`Policy 0, Clamped 1, Fallback 2, Human 3`), and `es_safety::EventSet::bits()`. It is the
record `events.json` gets, not a second measurement of the same step, which is what lets the
editor rebuild a finished run's own rows from the wire (`docs/design/editor-shell.md` §13).

Two rules the producer keeps and a third the wire cannot:

- **Nothing is computed for telemetry's sake.** The evaluator hands a closure what it already
  had; the image bytes are *borrowed* from the plan's own input buffer rather than rendered or
  copied. With no `--telemetry` nothing binds and the two §10.5 artifacts are byte-identical.
- **Nothing waits.** Publishing is §6's non-blocking fan-out, so an editor that stalls on a
  repaint loses frames and the run does not notice. The §6 note about `CLIENT_QUEUE_CAPACITY`
  being a guess still stands — what M7/E4 adds is a measurement of the producer-side cost
  (`editor-shell.md` §13, gate 9), not a tuned queue.
- **The token is still clear text on a trusted socket** (§4). `--telemetry 127.0.0.1:7777` is
  the shape this is for; a bind on `0.0.0.0` with a token is not a secure remote channel, and
  §6/§8's TLS/QUIC ceiling is where that gets fixed.

### 9.1 The whole loop, on one address (packet M7/E7)

After E4 only `es eval run` published. Three more producers now do, and one of them is a
*cycle* — so a person who presses Start on the whole loop watches it rather than watching log
lines. The frame construction lives in one file, `crates/es/src/cmd/telemetry.rs`, which is
the only place in the repository that turns anything into a `Frame`.

**Every stream-1 event carries a `stage` field**: `collect`, `expert-gate`, `train`, `eval`,
`showcase` or `cycle`. That is what makes one socket enough. `es loop cycle --telemetry` binds
**once**, before its first stage, and hands the same publisher to every in-process stage
(`cmd::r#loop::collect`, `cmd::eval::run`, `cmd::train::run` each take an
`Option<&mut Publisher>`); the standalone commands bind their own and name themselves. A
cycle's stages are bracketed by `stage.begin { name }` and
`stage.end { name, seconds, code }`, so the editor's strip is the plan's own order with the
wall-clock each stage took.

The address is the **cycle's, not a stage's**: the stages are in-process calls, so
`--telemetry` is on no line of `es loop cycle --dry-run`'s plan and `plan-cycle.txt` is
unmoved (`cycle_telemetry_is_one_address`).

| Stream | Payload | Sent by | When |
|---|---|---|---|
| `1` | `Event { kind, fields }` | all | `cell.begin` / `cell.end` / `suite.end` (eval, §9), `episode.begin { episode, seed }` / `episode.end { episode, outcome, steps }` (collect), `train.begin { total_steps }` and `checkpoint { step, policy_hash }` (train), `stage.begin` / `stage.end` (cycle). Every one of them also carries `stage` |
| `2` | `Scalars([frame, tick, source, violation bits])` | eval, collect | every control tick |
| `3` | `Metrics(PerfMetrics)` | eval | at each `cell.end` |
| `4` | `Image { format: "rgb8" }` | eval, collect, train | every `--telemetry-image-every` ticks; for a training run that is the **sample image** every `--sample-every` optimizer steps |
| `5` | `Scalars([step, loss, lr, samples_per_s])` | train | every `--progress-every` optimizer steps |

Stream 5 is the one new id and it is still data, not schema: `protocol.rs` is frozen at its
version, and a consumer that does not know 5 ignores it.

**Collection publishes what it already had.** `Collector::run_with_sink` calls one closure
(`es_data::collect::CollectSink`, a closure and not an eighth extension point, `INV-17`) with
the episode boundaries and, per control tick, the `action_source` the dataset's own column
records plus the `EventSet` bits of that step. The bits are the **delta of the plane's own
per-kind counters** across the step: one env means exactly one `validate` per step, so a kind
whose count moved is a kind that step raised — the same bitset `SafeAction::events` carries,
read from the side the collector can see (`es_env::DomainRunner::emit_actions` keeps the
`SafeAction` itself). That array is only read when a sink is there, and the dataset under
`--out` is byte-identical with and without the flag
(`collect_telemetry_publishes_every_episode` compares every file of both trees).

**Training publishes its own stdout.** `train_act.py --progress-every N` prints one
`{"progress": {…}}` line every `N` optimizer steps and `--sample-every N` writes
`sample-<step>.bin` + `.json` beside `--loss-curve` and prints `{"sample": "<path>"}`. The
summary stays the **last** line and stays byte-identical, and `es train` still parses it the
same way — with `--telemetry` it reads stdout through `Stdio::piped` + `BufReader::lines`
instead of at exit, and stderr is inherited rather than piped, because reading two pipes from
one thread deadlocks when either fills.

Three rules this producer keeps:

- **The two trainer flags are never in the plan.** `config.json` carries the plan and
  `identity_hash` covers `config.json` (§19.3), so a flag that changes nothing the run
  computes must not move a run's identity — a `training.lock` that differed by whether
  somebody was watching would make two identical runs look like two runs. They are appended
  to the trainer's argv at spawn time and printed on their own line. `train_telemetry_streams_the_curve`
  compares `training.lock`, `training/config.json`, `metrics/loss.json` and the packed
  `checkpoints/40.esb` byte for byte against a run without the flag.
- **No stream-3 frame.** §12.4's `PerfMetrics` has no training slot, and `protocol.rs` is
  frozen; mapping training samples/s onto `actions_per_sec` would be a lie about which
  quantity was measured. The throughput is the fourth scalar of stream 5 and nowhere else.
- **A training frame's `tick` is zero.** An optimizer step is not a physics tick, and the step
  is the first scalar of the payload.

**The sample image is for display and says so.** It is one image input of the current batch,
*after* augmentation — the tensor the network is actually fitting — mapped to `Rgb8` with
`clamp(v, 0, 1) * 255`, which is the exact inverse of the demo's chain (`Dequantize` ÷255 then
`Normalize{Range 0..1}`, the identity). Neither `contract.json` nor the bake's `manifest.json`
carries a normalisation's numbers, so the mapping is **recorded in the sidecar** rather than
assumed: a document normalising by mean/std would show through as a washed-out picture with
`"mapping": "clamp(v, 0, 1) * 255"` beside it, rather than as a wrong one nobody can question.
Undoing an arbitrary chain would need the chain, and that is a manifest change this packet
does not make.
