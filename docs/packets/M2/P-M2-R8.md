# P-M2-R8 — accept-loop connection cap and an absolute handshake deadline

Fixes the M2 review should-fix (`docs/reviews/M2.md`, Should-fix,
`crates/es-telemetry/src/transport.rs:173,255`): the client cap was applied inside
`serve_client`, after `thread::spawn` had already run, and the handshake timeout was a
per-read `set_read_timeout` rather than a budget over the whole handshake — so a peer
dribbling one byte at a time never tripped it and held a server thread indefinitely. Design
note: `docs/design/telemetry-protocol.md` §4 (updated by this packet).

## context

```
crates/es-telemetry/src/transport.rs   (cap moved into the accept loop, absolute deadline,
                                         Server::stats().{handshaking,threads_live}, tests)
docs/design/telemetry-protocol.md      (§4 resource-limits section)
docs/packets/M2/P-M2-R8.md             (new)
```

## spec

- **Cap before spawn.** The accept loop (the single thread running
  `listener.incoming()`) checks `handshaking.load() + clients.lock().len() >=
  cfg.max_clients` *before* calling `thread::spawn` for a newly accepted connection. Over the
  cap: the accept loop itself writes `Bye { reason: "too many clients" }` synchronously, then
  drops the socket (closing it) — no thread is spawned, `Hello` is never read. Because the
  accept loop is single-threaded, this check-then-increment has no race with itself, unlike
  the old check inside `serve_client` (which ran concurrently per connection).
- **Absolute handshake deadline.** The accept loop records `deadline = Instant::now() +
  cfg.handshake_timeout` at admission and passes it into `serve_client`, which reads `Hello`
  via `read_message_until(stream, buf, Some(deadline))`: before every socket read that would
  block, the function recomputes the *remaining* time until `deadline` and uses that as the
  read's timeout, erroring out once none is left. This bounds the total handshake time
  regardless of how many partial reads a slow or adversarial peer causes — a single
  `set_read_timeout` call (the prior shape) only bounds one read, not their sum.
- **Observable thread accounting.** `Server` holds two `Arc<AtomicUsize>` — `handshaking`
  (connections admitted past the cap but not yet past `Hello`) and `threads_live` (every
  `serve_client` thread currently running). `ServerStats` gains `handshaking` and
  `threads_live` fields, populated by `Server::stats()`. Both counters are released via a
  small RAII `CounterGuard` on thread/handshake exit, so every path — clean return, an early
  `?`/`else return`, or a panic unwind — accounts for itself exactly once. The handshaking
  guard is released only once the connection is inserted into the client map (not right after
  the `HelloAck` write), so a connection is never invisible to the cap check in between.
- **`ServerConfig` unchanged in shape.** `handshake_timeout` and `max_clients` already existed
  and are reused for both the cap and the deadline; no new config field.

Constraints: English only, no new external dependency, no new trait (INV-17), no new
`HashMap` (`BTreeMap` only).

## oracle

```
cargo fmt -p es-telemetry --check
cargo clippy -p es-telemetry --all-targets -- -D warnings
cargo test -p es-telemetry
```

- `connections_past_the_cap_get_bye_without_spawning_a_thread`
  (`crates/es-telemetry/src/transport.rs`): with `max_clients = 8`, opens 8 silent sockets to
  fill the cap, then 8 more; each of the extra 8 reads back `Bye { reason: "too many
  clients" }` without ever completing a handshake, and `Server::stats().threads_live` stays
  `<= max_clients + 2` both while the first 8 are still waiting out their handshake deadline
  and after it passes.
- `a_dribbling_client_is_dropped_at_the_absolute_handshake_deadline`: sends a valid `Hello`
  one byte every 30 ms (comfortably inside any single read's own timeout) against a 300 ms
  `handshake_timeout`; the connection is dropped within `handshake_timeout + 1s` regardless of
  the partial reads.

## acceptance

- Every pre-existing `es-telemetry` test stays green, including
  `a_connection_past_max_clients_is_refused` and `a_silent_client_is_dropped_after_the_
  handshake_timeout`.
- Both new tests are deterministic across repeated local runs and the whole `cargo test
  -p es-telemetry` suite finishes in well under 3 seconds (measured: ~0.8s).
- `docs/design/telemetry-protocol.md` §4 describes the cap-before-Hello ordering (a behavior
  change: an over-cap connection now gets "too many clients" even with a bad token, since its
  `Hello` is never read), the absolute-deadline mechanism, and the new `ServerStats` fields.

## forbidden

- Any other should-fix or nit from `docs/reviews/M2.md` (they are separate packets).
- `crates/es-telemetry/src/protocol.rs`, `crates/es-telemetry/src/ring.rs`, and any other
  crate — this packet is transport-only.
- New dependencies, new traits, TLS/QUIC (still `es-transport`, layer 11, later).
- Committing.
