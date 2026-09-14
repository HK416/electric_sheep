# P-M3-W1-R3 — a fresh HIL session per `Hello`

Spec: §25.1 (authenticated datagrams; a spoofed command must not reach the plane), §24.2 (HIL over
low-latency UDP), INV-12. Design note `docs/design/ros2-boundary.md` sections 7.2 and 7.7.
Closes **S-2** of `docs/reviews/M3-W1.md`.

## context

```
crates/es-ros2/src/hil/link.rs
crates/es-ros2/tests/hil_gate.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R3.md
```

## spec

`HilLink::bind` computes `session_id` once (`link.rs:96-98`) and `hello` never regenerates it
(`:250-253`); what a `Hello` does regenerate is `last_seq`, resetting it to that datagram's `seq`.
The two together defeat the replay protection: a party holding a recorded session — the `Hello`
and the `Command`s that followed it — replays the `Hello`, the link accepts it (same key, same
`session_id`, `hdr.session_id == 0` as required), `last_seq` drops back, and every recorded
`Command` with a higher `seq` is then accepted as new and reaches `HilCore::on_command`. Design
note 7.7's stated goal, "spoofed commands must not reach the plane", does not hold against a
passive recorder on the same host.

The session id is not a secret and does not need to be; it needs to be *fresh*, so that a datagram
authenticated under an earlier session is refused by the `hdr.session_id != self.session_id` check
at `:197`. Regenerate it on every accepted `Hello`:

- Move the `session_id` derivation out of `bind` into a private `fn next_session_id(&self) -> u64`
  that mixes `stats::wall_ns()`, the local port and a monotonically increasing per-link counter, and
  never returns `0` (0 is `Hello`'s reserved value) or the current `session_id`.
- `hello` (`:250`) assigns the new id before it sends the `HelloAck`, so the `HelloAck` carries it
  and the controller adopts it as it already does (`hil_gate.rs:245`, `Peer::recv`).
- `tx_seq` resets with it; `sent` (the round-trip slots) is cleared to `u64::MAX` so a stale slot
  cannot produce a bogus RTT sample across the boundary.
- `bind` keeps setting an initial id so the field is never uninitialised; nothing else moves.

The counter is per link and lives in the struct; no global state, no RNG (INV: no global RNG,
§3.4), nothing allocates, and the id stays out of every Safety Plane input and out of the `.eshil`
log, so the gate's byte-identical replay is unaffected.

Design note 7.2 gains one line under the `session_id` row: "regenerated on every accepted `Hello`;
`0` only in `Hello` itself".

## oracle

```
cargo test -p es-ros2 --test hil_gate -- --nocapture
cargo fmt --check
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
```

New test in `tests/hil_gate.rs`, alongside
`bad_tag_wrong_session_and_stale_seq_never_reach_the_plane`:

- `a_replayed_session_does_not_reach_the_plane` — `Peer` handshakes, sends one
  `small_command`, and the test **keeps the raw bytes** of that `Hello` and that `Command`. A tick
  runs and the command is counted. Then the recorded `Hello` bytes are resent verbatim: assert the
  link's `session_id` changed (the new `HelloAck` carries a different one) and that resending the
  recorded `Command` bytes afterwards leaves `stats().commands` unchanged and increments
  `rx_invalid`. **FAILS before the fix**: today the recorded command is accepted a second time.
- `a_second_hello_starts_a_clean_session` — after the replayed `Hello`, a freshly framed
  `Command` on the *new* session is accepted, so the fix does not break reconnection.

The gate test `hil_live_run_replays_to_byte_identical_decisions` is the regression guard for the
controller's own reconnect path and must still pass with all its non-vacuity assertions.

## acceptance

- Both tests pass; the first is confirmed to fail on the unfixed `hello`.
- All 11 existing `hil_gate` tests pass unchanged, `RAN hil_gate steps=2000 …` still printed, and
  `tests/fixtures/hil/v1_small.eshil` still replays identically — the log format does not carry a
  session id, so the fixture is untouched (and must not be regenerated).
- `HilLink`'s public signatures unchanged; `session_id` stays private.
- `Instant`/`SystemTime` stay confined to `link.rs`/`stats.rs`; `hil_imports_no_ros_module` passes.
- No allocation on the accept path, no `HashMap`, no new trait, no `unsafe`.

## forbidden

- `crates/es-safety`, `crates/es-runtime-embedded`, `crates/es-telemetry`, `crates/es-ir`.
- `src/hil/wire.rs` (the wire layout, the tag, and `MAX_DATAGRAM` are correct and stay), `log.rs`,
  `replay.rs`, `core.rs`, `stats.rs` beyond reading `wall_ns`.
- The ROS modules, `crates/es`, `xtask`, `.github/workflows/ci.yml`.
- Adding encryption, a nonce window, a key-rotation scheme, or a second key (design note 10.6 is a
  human decision, not this packet).
- Regenerating `tests/fixtures/hil/v1_small.eshil`. Any other M3 W1 finding.
