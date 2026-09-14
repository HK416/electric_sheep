# P-M3-W1-R1 — the violation-rate window is a window

Spec: §9.4 (`EnvelopeViolationRate { window, max_frac }`), §10.3 (`envelope_violation_rate`, whose
own example acceptance is `<= 0.01`), §9.3 (clamp vs fallback), Appendix B.4, INV-12.
Closes blocker **B-1** of `docs/reviews/M3-W1.md`.

## context

```
crates/es-safety/src/counters.rs
crates/es-safety/src/plane.rs
crates/es-safety/tests/properties.rs
docs/design/safety-plane.md
docs/design/safety-plane.ko.md
docs/packets/M3/P-M3-W1-R1.md
```

## spec

`ViolationWindow::fraction()` (`counters.rs:59-64`) divides `ones` by `filled` — the steps observed
so far — not by the configured `window`. The consequences, all reachable from a spec-shaped
configuration:

- The first dirty step of a run makes the rate exactly `1.0`, so `plane.rs:268-273` raises
  `ViolationKind::ViolationRate` on the next step for **any** `max_frac < 1.0`.
- That trip runs the fallback (`plane.rs:276-284`); the fallback step is itself dirty
  (`finish` → `window.push(!events.is_empty())`, `plane.rs:460`), so the rate stays `1.0` and the
  plane never leaves fallback. With `FallbackPolicy::EmergencyStop` it latches the e-stop
  (`plane.rs:282`) on the second step of the run.
- A HIL cold start is the quickest way in (its first tick has no chunk, so `ChunkUnderrun` fires),
  but it is not HIL-specific: one transient clamp at step 3 of any deployment does the same.

The rule to pin, and the one §10.3's `<= 0.01` reads naturally: **the rate is over the last
`window` steps, and the watchdog does not trip until `window` steps exist.** Before then the window
is not a sample of anything.

The fix is in `counters.rs` only:

- `ViolationWindow::fraction()` returns `self.ones as f64 / self.len as f64` once
  `self.filled == self.len`, and `0.0` before that (it already returns `0.0` for `filled == 0`).
- `SafetyCounters::envelope_violation_rate()` keeps its signature and its §10.3 meaning; document
  in its doc comment that it reads `0.0` until the window is full, and that `es-eval`'s metric —
  which computes the rate over a whole episode, not over this ring — is unaffected.

`plane.rs`'s watchdog block is unchanged: it still reads the rate as of the *previous* step, which
is what keeps the rule non-circular. No signature changes, no new field, no allocation.

`docs/design/safety-plane.md` records the rule in the watchdog section: a partially filled window
never trips, and why (a single early violation is otherwise a 100 % rate).

## oracle

```
cargo test -p es-safety
cargo test -p es-ros2 --test hil_gate
cargo fmt --check
cargo clippy -p es-safety --all-targets -- -D warnings
cargo xtask layering
```

New tests in `crates/es-safety/tests/properties.rs`, written before the fix:

- `a_partial_window_never_trips_the_rate_watchdog` — `EnvelopeViolationRate { window: 8,
  max_frac: 0.25 }` with `FallbackPolicy::HoldPosition`. Step 1 submits
  `ActionChunk::empty(..)` (a `ChunkUnderrun`, exactly the HIL cold start); steps 2-8 submit clean
  in-envelope chunks. Assert no step raises `ViolationKind::ViolationRate` and every step from 2 on
  has `ActionSource::Policy`. **FAILS before the fix** (step 2 onward are `Fallback`).
- `a_full_window_trips_at_the_threshold` — same plane; drive 8 clean steps, then 3 steps whose
  action exceeds the velocity limit (3/8 = 0.375 > 0.25), and assert `ViolationRate` appears on the
  step after the third and not before.
- `the_rate_falls_back_out_of_the_window` — continue that run with clean chunks until the three
  dirty bits have slid out, and assert `ViolationRate` stops firing and the source returns to
  `Policy`, i.e. the latch is a function of the window and not permanent.
- `an_estop_rate_watchdog_does_not_latch_on_step_two` — the same configuration with
  `FallbackPolicy::EmergencyStop`: after one early `ChunkUnderrun`, `estop_latched` is still false.

## acceptance

- The four tests pass; the first and the last are confirmed to fail on the unfixed `fraction()`.
- The existing `es-safety` suite (including `properties.rs:230`'s config-rejection test and
  `determinism_two_planes_same_inputs_same_outputs`) still passes unchanged.
- `es-runtime-embedded` and `es-eval` test suites unchanged and passing.
- `cargo test -p es-ros2 --test hil_gate` still passes with the fixture's `max_frac: 1.0`; the
  fixture is **not** changed by this packet (that is a separate judgement about what the gate
  should arm).
- No public signature changes. `SafetyCounters` gains no field; nothing allocates.
- INV-12/INV-13 untouched: the change can only make the watchdog trip *less*, never disable a
  clamp stage, and `validate` still returns no `Result`.

## forbidden

- `crates/es-ros2` (the HIL fixture's `max_frac` and `hil_gate.rs`'s comment belong to whoever
  decides what the gate arms), `crates/es-ir` (`Watchdog`'s schema is unchanged), `crates/es-eval`
  (`metrics.rs:44` computes the episode-level metric from a different source and stays as it is),
  `crates/es-runtime-embedded`.
- Changing `WINDOW_CAP`, the ring's layout, or `window.push`'s "any event is dirty" rule.
- Adding a "warm-up" or "grace period" configuration field: the window length already is one.
- Any other M3 W1 finding.
