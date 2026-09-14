# P-M3-W1-R7 — the rate watchdog does not measure its own fallbacks

Spec: §9.4 (`EnvelopeViolationRate`, `FallbackPolicy`), §18.5 ("Fallback is normal behavior, not a
failure"), §10.3 (`envelope_violation_rate`), §9.3, Appendix B.4, INV-12, INV-13.
Design note `docs/design/safety-plane.md`, watchdog table row 7 and the "Known ceiling" paragraph.
Closes the second-order bug P-M3-W1-R1 (`04e04b7`) found and was forbidden to touch, pinned by
`crates/es-safety/tests/properties.rs::a_tripped_rate_watchdog_does_not_release_itself`.

## context

```
crates/es-safety/src/plane.rs
crates/es-safety/src/types.rs
crates/es-safety/tests/properties.rs
docs/design/safety-plane.md
docs/design/safety-plane.ko.md
docs/packets/M3/P-M3-W1-R7.md
```

## spec

R1 fixed the denominator; the numerator is still circular one order later. Once
`ViolationKind::ViolationRate` trips, step 4a (`plane.rs:276-284`) runs the fallback on **every**
later step, and `finish` records each of them with `window.push(!events.is_empty())`
(`plane.rs:460`). The events on those steps are the watchdog's own `ViolationRate`, so the window
refills with the watchdog's echo, `ones` never decays, and the trip is permanent.

**Which reading of §9.4 applies.** Three things say the spec intends recovery, not a latch:

1. §18.5 is explicit: "An env in which the fallback triggered remains `Ok` rather than
   `Quarantined` … **Fallback is normal behavior, not a failure.**" A fallback that can never end
   is a failure under any reading.
2. §9.4's `FallbackPolicy` uses the word "latch" exactly once — `EmergencyStop // immediate stop +
   latch`. Latching is a property of that one *policy*. Nothing in §9.4 attaches a latch to a
   *watchdog*, and `EnvelopeViolationRate` is defined purely as a windowed rate.
3. The plane already implements that reading. `estop_latched` is set only for
   `FallbackKind::EmergencyStop` (`plane.rs:279-283`) and `is_latched()` / `reset_latch()`
   (`plane.rs:152-159`) are the explicit operator-reset path scoped to it. Today's behaviour gives
   every *other* policy a de facto permanent latch that `is_latched()` does not report and
   `reset_latch()` cannot clear — an invisible latch, which is the worst of both readings.

The plane's own comment at `plane.rs:268-270` and design note row 7 already state that the rule is
meant to be non-circular ("this step's own clamp is not recorded yet"). This is the same
circularity, one step later.

**The change.** The window records whether the step was dirty *for a reason other than the rate
watchdog's own trip*:

- `types.rs`: add `pub const fn without(self, kind: ViolationKind) -> Self` to `EventSet` — one
  `u32` mask on the existing newtype, no new type, no trait, no allocation.
- `plane.rs:460`: `self.counters.window.push(!events.without(ViolationKind::ViolationRate).is_empty());`

Release is then automatic and bounded: the trip clears exactly `window` steps after the last
genuine violation leaves the ring. A step that is dirty for **any** real reason — a clamp,
`NanInf`, `ChunkUnderrun`, `HeartbeatLoss`, `SensorDropout`, `StaleObservation`,
`InferenceDeadline` — still counts, so the rate still measures real violations and a plane that
keeps genuinely violating keeps the watchdog tripped.

**What must not change.** `dirty_steps`, `clamped_steps`, `fallback_activations`,
`violations[ViolationRate]` and `steps` record what actually happened and must keep counting every
fallback step, including the watchdog's own: §10.3's episode-level `envelope_violation_rate` in
`es-eval` is computed from `dirty_steps`/`steps`, not from this ring, and would under-report real
fallbacks if it moved too. Only the watchdog's **input ring** changes.

**INV-12 / INV-13.** Every clamp stage and every other watchdog is untouched and still runs on
every step; nothing can now pass the envelope that could not before, because the rate watchdog is a
meta-watchdog over the others' output and never a stage of its own. `validate` keeps its signature
and still returns no `Result`. No configuration switch, no "HIL mode", no bypass.

Design note: replace the "Known ceiling, not fixed here" paragraph with the rule and the release
bound, and add the non-self-measurement clause to watchdog table row 7.

## oracle

```
cargo test -p es-safety
cargo test -p es-ros2 --test hil_gate
cargo test -p es-runtime-embedded
cargo fmt --check
cargo clippy -p es-safety --all-targets -- -D warnings
cargo xtask layering
cargo xtask nostd
cargo xtask check-spec-refs
```

In `crates/es-safety/tests/properties.rs`, reusing R1's `rate_plane` / `hold_step` / `dirty_step`
helpers (`window: 8`, `max_frac: 0.25`):

- `a_tripped_rate_watchdog_releases_after_a_clean_window` — **replaces**
  `a_tripped_rate_watchdog_does_not_release_itself`, which pins the bug this packet fixes; R7
  deletes it and this test takes its place at the same position, with a comment naming R7. 8 clean
  steps, 3 dirty (3/8 = 0.375 > 0.25) to trip, then clean steps: `ViolationRate` stops firing
  within `window` steps of the last genuine violation and `source` returns to `ActionSource::Policy`.
  **FAILS before the fix** (the existing test asserts the opposite through step 40).
- `a_real_violation_during_a_trip_still_counts` — trip as above, then keep feeding genuinely dirty
  steps: the watchdog does **not** release while they continue. The fix must not blind the window
  to real violations.
- `the_metric_counters_still_count_every_fallback` — over one trip-and-release cycle,
  `fallback_activations`, `dirty_steps` and `violations[ViolationRate]` equal the number of
  fallback steps actually run. The guard against "fixing" the counters along with the ring.
- `an_estop_rate_trip_still_latches` — with `FallbackPolicy::EmergencyStop`, a genuine rate trip on
  a full window still sets `is_latched()`, and only `reset_latch()` clears it. The one real latch
  stays (INV-12).
- `EventSet::without` unit test in `types.rs`: removing an absent kind is the identity, removing
  the only present kind is `EMPTY`, other kinds survive.

## acceptance

- The four `properties.rs` tests and the `types.rs` unit test pass; the first is confirmed to fail
  before the fix, and `a_tripped_rate_watchdog_does_not_release_itself` is gone with R7 named in the
  replacement's comment.
- R1's other five tests and the rest of the `es-safety` suite pass unchanged, including
  `a_partial_window_never_trips_the_rate_watchdog` and
  `determinism_two_planes_same_inputs_same_outputs`.
- `es-runtime-embedded` and `es-eval` suites unchanged and passing; `cargo xtask nostd` clean.
- `cargo test -p es-ros2 --test hil_gate` still passes with the fixture's `max_frac: 1.0` untouched.
- Public API grows by exactly one method, `EventSet::without`. No new field on `SafetyCounters` or
  `PlaneState`, no allocation, no `HashMap`, no new trait (INV-17).
- Design note and its `.ko.md` sibling updated in the same commit; the "Known ceiling" paragraph is
  gone.

## human override

If §9.4 is instead ruled to mean **a tripped rate watchdog latches until an explicit operator
reset**, this packet becomes: add `rate_latched: bool` to `PlaneState`, set it at the trip, make
`ViolationRate` re-fire from the flag rather than from the window, expose it (`is_rate_latched()`)
and clear it in `reset_latch()`, and write the rule into §9.4 and the design note. That reading
contradicts §18.5 as quoted above and is strictly more work, which is why it is the override and
not the default — but note that under *either* ruling the present behaviour is wrong, because the
latch it produces today is invisible to `is_latched()` and unclearable by `reset_latch()`.

## forbidden

- `crates/es-ros2` (the HIL fixture's `max_frac: 1.0` becomes usable again once this lands, but
  changing what the gate arms is a separate judgement), `crates/es-ir` (`Watchdog`'s schema is
  unchanged), `crates/es-eval`, `crates/es-runtime-embedded`.
- Changing `record_step`, `dirty_steps`, `clamped_steps`, `fallback_activations`, `steps`, or what
  `es-eval` reads.
- Changing `ViolationWindow`'s layout, `WINDOW_CAP`, `fraction()`, or anything else R1 (`04e04b7`)
  settled.
- Touching `estop_latched`, `is_latched`, `reset_latch`, or the `EmergencyStop` branch.
- Adding a hysteresis, cool-down, release-threshold or grace-period configuration field: the window
  length is the only period this watchdog has.
- Any other M3 W1 finding.
