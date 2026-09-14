# P-M3-W1-R6 — an empty `.eshil` log does not verify

Spec: §24.2 (the HIL gate is "live == replay"), §25.3 (a versioned log format), §1.4 (a check that
can pass on nothing is not a check).
Closes **S-6** of `docs/reviews/M3-W1.md`.

## context

```
crates/es-ros2/src/hil/replay.rs
crates/es-ros2/tests/hil_gate.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R6.md
```

## spec

`replay` (`replay.rs:124-133`) builds `ReplayReport` with `identical: true` initialised and never
falsified when there is nothing to compare, and both hashes are blake3 of the empty input. A log
that is nothing but a valid header and a trailer — a run that crashed before its first tick, or a
file someone truncated to the header and re-trailered — therefore comes back
`identical: true, live_hash == replay_hash, truncated: false`. Every property the gate reports is
satisfied by a log containing no decisions at all.

Nothing is wrong in the gate test: `hil_gate.rs:403-418` asserts `commands > 0`,
`steps == GATE_TICKS` and five distinct plane events by hand, so the M3 gate is non-vacuous. The
weakness is in the type every *other* caller will read — `es hil replay` (W1e) and any evidence
bundle that quotes a `ReplayReport` — which would report a clean verification for an empty file.
Move the non-vacuity into `replay` itself so the caller cannot forget it:

- `ReplayReport::identical` is `false` when `steps == 0`. Document it on the field: "a log with no
  `Decision` record verifies nothing, so it is never `identical`."
- Add `ReplayReport::is_verified(&self) -> bool` = `self.identical && !self.truncated &&
  self.steps > 0 && self.live_hash == self.replay_hash` — the single predicate §24.2's gate means,
  so a caller does not re-assemble it from four fields and drop one.
- `v1_fixture_still_replays_identically` and
  `hil_live_run_replays_to_byte_identical_decisions` assert `is_verified()` in addition to what they
  already assert (the existing field-level assertions stay: they are what says *which* property
  failed).
- `truncated_log_replays_its_complete_prefix` keeps asserting `prefix.identical` and
  `prefix.steps > 0`, and gains `assert!(!prefix.is_verified())` — a truncated prefix is a correct
  replay of an incomplete run, not a verified one.

`replay`'s signature, the `.eshil` v1 format, the trailer, the decision hash and the log records are
all unchanged; `tests/fixtures/hil/v1_small.eshil` is not regenerated. Design note section 7.5 gains
one line stating the predicate.

## oracle

```
cargo test -p es-ros2 --test hil_gate -- --nocapture
cargo fmt --check
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
```

New tests in `tests/hil_gate.rs`:

- `a_log_with_no_decisions_does_not_verify` — take `fixture_bytes()`, keep the header, and append
  only a trailer whose hash is blake3 of nothing and whose `steps` is 0. `replay::<NJ, H>` returns
  `Ok`, `steps == 0`, `live_hash == replay_hash` (they are both the empty hash — that is the point),
  and `identical` is `false`, `is_verified()` is `false`. **FAILS before the fix** on `identical`.
- `a_header_only_log_does_not_verify` — the header alone, no trailer: `truncated`, `steps == 0`,
  `!is_verified()`.
- `the_gate_predicate_is_the_gate` — the v1 fixture is `is_verified()`; the tampered log of
  `tampered_step_diverges_at_its_tick` is not; the truncated prefix is not.

## acceptance

- The three tests pass; the first is confirmed to fail on the unfixed `replay`.
- All 11 existing `hil_gate` tests pass, `RAN hil_gate steps=2000 …` still printed, and the
  fixture still replays (`identical`, `live_hash == replay_hash`, `steps == 120`).
- `ReplayReport` gains one method and no field; it stays `Copy`, `PartialEq`, `Eq`.
- `tests/fixtures/hil/v1_small.eshil` is byte-identical to what is checked in.
- No `HashMap`, no new trait, no `unsafe`, no allocation on the replay hot path.

## forbidden

- `crates/es-safety`, `crates/es-runtime-embedded`, `crates/es-telemetry`, `crates/es-ir`.
- `src/hil/wire.rs`, `link.rs`, `core.rs`, `stats.rs`; `log.rs` beyond a doc comment (the record
  layout, the trailer and `LogWriter` are correct and stay).
- The ROS modules, `crates/es` (`es hil replay` is W1e), `xtask`, `.github/workflows/ci.yml`.
- Regenerating `tests/fixtures/hil/v1_small.eshil`, or changing the `.eshil` version byte.
- Making `replay` return `Err` for an empty log: a well-formed log with no decisions is a valid
  file that verifies nothing, and the report is what must say so.
- Any other M3 W1 finding.
