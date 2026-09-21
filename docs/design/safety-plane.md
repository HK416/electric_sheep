# Safety Plane runtime (`es-safety`) — design

Spec refs: §9 (Deployment IR and Safety Plane), §9.3 (envelope), §9.4 (watchdogs and
fallback), §9.5 (sim/real identity), §8.5 (action chunk), §8.6 (async inference, chunk
underrun), §10.3 (`envelope_violation_rate`, `chunk_underrun_rate`), §18.5 (a fallback is
normal operation), Appendix B.4 (pinned type shape), §28.7 gate 8.

Invariants: INV-11 (`es-safety` never depends on `es-policy`), INV-12 (no code path disables
the plane), INV-13 (`validate` returns no `Result`).

## What this crate is

`es-ir::deployment` is the **configuration**: a validated, hashable description of limits,
watchdogs and fallback. This crate is the **runtime** that executes that configuration once
per control tick. The two never merge: the IR owns `Vec`-shaped, serde-shaped, diagnostic-
shaped code; the runtime owns fixed-size arrays and a function that cannot fail.

Per §9.5 the same code runs in simulation and on hardware. There is no simulation mode, no
"training" mode, and no `enabled` flag anywhere in this crate (INV-12). A test that needs
room widens the envelope in its fixture.

## `no_std` status

> **Superseded (M3 W2).** The split described at the end of this section has been done: the
> crate *is* `no_std` under `--no-default-features`, the `Vec`s named below are now fixed-size
> arrays, and `SafetyPlane::from_config` is the single construction path `from_ir` delegates
> to. See `docs/design/embedded-runtime.md` §2–3. The rest of this section is kept for the
> reasoning, not as a statement of the current build.

Appendix B.4 says "no_std 가능, 힙 할당 0". `es-ir` is `std` (it uses `String`, `Vec`,
`BTreeSet`, `serde`), and `from_ir` must read it, so the crate as a whole stays `std` for
M1. What is delivered now is the half that matters operationally:

- **the hot path allocates nothing.** `validate`, `heartbeat` and `sensor_seen` touch only
  fixed-size arrays inside `SafetyPlane` plus two `Vec`s that are sized once in `from_ir`
  (the retract trajectory and the sensor-dropout table) and only indexed afterwards. A test
  wraps a long `validate` loop in `es_core::alloc_count::assert_no_alloc`.
- **no floating-point time.** All time is `PhysTick` (u64) and `Micros` (u64); the only
  float derived from time is the constant control period `dt_s`, computed once in `from_ir`
  from the rational `TickRate` and never accumulated (§3.4).
- **no `HashMap`, no RNG, no global state.** The sensor table is a `Vec` scanned linearly
  and compared by `&str`; iteration order is its construction order.

The split that would make the crate literally `no_std` is a later packet: move `from_ir`
behind a `std` feature and keep the runtime core in the default build. `[features] std =
["es-ir"]` is the shape, and nothing in the runtime core below refers to `es-ir` types
except by `From` conversion at construction, so that split is mechanical.

## Construction — `SafetyPlane::from_ir`

```rust
pub fn from_ir(ir: &DeploymentIr) -> Result<Self, SafetyConfigError>
```

This is the only constructor. There is no `SafetyPlane::new()`, no `Default`, and no public
field, so an envelope-less plane is not constructible (INV-12). `from_ir` rejects, in this
order:

1. `ir.validate()` returned diagnostics — the IR is not internally consistent.
2. `ir.robot.n_joints != NJ` or `ir.action.dim != NJ` — the const generic disagrees with the
   configuration.
3. `ir.action.horizon != H` — the chunk buffer would not match.
4. Any limit is non-finite, or any per-joint vector has the wrong length. The IR validator
   already covers most of this; the check is repeated because the runtime copies into fixed
   arrays and must not trust a caller-built IR that skipped `validate`.
5. A watchdog kind appears twice, or an `EnvelopeViolationRate` window exceeds
   `WINDOW_CAP` (256) — the ring is pre-allocated at that cap.

Everything is then copied into fixed-size arrays: `[Limit; NJ]`, `[f64; NJ]` for velocity,
acceleration, torque, jerk, the two rate-limit rows, and the soft margins (already folded
into an effective `[lower + margin, upper - margin]` pair so the hot path does one compare
per joint, not three).

## Pre-allocated state layout

```
SafetyPlane<NJ, H>
├── envelope: Envelope<NJ>            fixed arrays, immutable after construction
│     soft:   [Limit; NJ]             hard limit shrunk by the soft margin
│     hard:   [Limit; NJ]             hard limit, used for the final scrub
│     vel_max, acc_max, tau_max, jerk_max: [f64; NJ]
│     d1_max, d2_max: [f64; NJ]       action rate limits (§9.3 rate_limit)
│     workspace: Workspace            box/cylinder/convex-hull, EE spaces only
│     dt_s: f64                       control period, constant
├── watchdogs: Watchdogs              one Option per kind; Vec<SensorWatch> for dropout
├── fallback: Fallback<NJ>            kind + Vec<[f64; NJ]> retract trajectory (sized once)
├── state: SafetyState<NJ, H>
│     chunk:        [[f64; NJ]; H]    the accepted chunk (copied in, never borrowed)
│     chunk_valid:  usize             valid rows of `chunk`
│     chunk_mode:   ExecutionMode
│     cursor:       usize             index of the next row to execute
│     last_safe:    [f64; NJ]         last emitted action (the hold target, and the
│                                     reference every derivative stage measures against)
│     prev_safe:    [f64; NJ]         the one before it (second difference)
│     vel:          [f64; NJ]         velocity estimate, (last_safe - prev_safe) / dt_s
│     prev_vel:     [f64; NJ]         previous velocity estimate (acceleration)
│     retract_idx:  usize             cursor into the retract trajectory
│     last_chunk_tick / last_beat_tick: PhysTick
│     estop_latched: bool
│     seeded:       bool              observe_state has put the chain on the real pose
└── counters: SafetyCounters
      violations: [u64; ViolationKind::COUNT]
      fallback_activations, steps, clamped_steps: u64
      window: ViolationWindow           [u64; 4] bit ring, len ≤ 256, plus head/filled/ones
```

Nothing here grows. `Vec<[f64; NJ]>` (retract) and `Vec<SensorWatch>` (dropout) are the only
heap objects and both are final after `from_ir`.

`last_safe` starts at each joint's soft-limit midpoint, which is inside the envelope by
construction. A runtime that knows the real pose calls `observe_state(&q, &qd)` before the
first `validate` — that is the calibration knob a real arm needs, since a physical joint is
never where the midpoint says it is.

**Measurement enters the envelope exactly once per episode, and only there** (packet
`docs/packets/M5/V6-envelope-semantics.md`). `observe_state` seeds `last_safe`, `prev_safe`,
`vel` and `prev_vel` on the first call after `begin_episode` (or after construction) and
returns without touching anything on every call after that, so a caller may — and every caller
does — call it before *every* `validate`. `begin_episode` clears the e-stop latch and re-arms
the seed; it is what a collector or an evaluation runner calls at an episode boundary, and it
weakens nothing (INV-12).

The reason is §9.3 and §9.5, not convenience. §9.3's table constrains the *policy output* and
every dynamic row of it says **clamp**, which only a value the plane is about to emit can be;
so stages 3, 4 and 7 below are differences of the plane's own commands, never of the servo's
following error. And §9.5 requires the same `deployment_hash` to give the same safe action in
simulation and on the robot — which a clamp computed from feedback, exact in MuJoCo and
quantized and late over a serial bus, cannot. A caller that re-seeded every tick would be
bounding the position error that *generates* the servo's torque, which is `torque_limit`'s row
and is already enforced by the drive's own saturation.

## `validate` — the algorithm

```rust
pub fn validate(&mut self, chunk: &ActionChunk<NJ, H>, obs_age: Micros, now: PhysTick)
    -> SafeAction<NJ>
```

No `Result` (INV-13). No panic: every array index is a `usize` already bounded by `NJ`/`H`,
every division is by the constant `dt_s > 0`, and non-finite input is scrubbed before it
reaches arithmetic. There is always an executable action because `last_safe` always holds
one.

**Step 0 — latch.** If `estop_latched`, return `Fallback(EmergencyStop)` immediately with
the scrubbed hold action. Counters record the step and the fallback activation; no watchdog
is evaluated, no chunk is accepted. Only `reset_latch()` clears it.

**Step 1 — chunk acceptance.** The chunk-identity question — is this the same chunk as last
tick, or a new one? — is answered by `ActionChunk::seq` (P-M1-R3), a caller-supplied,
monotonic-per-policy-invocation counter (spec 8.6): a chunk is new iff `seq` is strictly
greater than the last one accepted; content is never compared. If it is new, copy it in,
`cursor = 0`, `last_chunk_tick = now`, remember `seq`. If `seq` is not greater (including a
repeat of the same value), the cursor keeps advancing over the stored chunk. The very first
chunk a plane ever sees is accepted unconditionally, whatever its `seq`.

This means a policy that re-emits a bit-identical chunk on a genuine replan (a stationary
hold, a saturated output) still reads as fresh — `ActionSource::Policy`, not a slow slide
into `ChunkUnderrun` — because the caller advanced `seq`, even though the bytes did not.
Conversely, a caller that resubmits the same `seq` twice (it failed to advance its own
counter) is read as stale and the cursor eventually runs past the stored chunk into
`ChunkUnderrun` — the safe-by-default outcome for a caller bug, not a special case in the
plane. `es-runtime-embedded::EmbeddedRuntime` increments its counter once per `infer` call
(replan tick) and reuses the same `seq` across the ticks that merely consume the buffered
chunk, which is exactly "new chunk" vs. "still consuming" in caller terms.

**Step 2 — executable length.** `len = min(chunk_valid, K)` under `RecedingHorizon`, where
`K = action.execute_chunk`; `min(chunk_valid, H)` under every other mode (§8.5). Rows past
`len` are not executable even though they are valid predictions — this is the truncation the
spec describes, and running off the end of it is an underrun, not a silent extension.

**Step 3 — watchdogs**, in this fixed order. All of them are evaluated even after one trips,
so every event is recorded in the counters and in `SafeAction::events`. Any trip runs the one
configured fallback — the IR carries a single `FallbackPolicy`, so which watchdog fired
changes the recorded event, never the response.

| # | Watchdog | Condition |
|---|---|---|
| 1 | `ChunkUnderrun` | `cursor >= len` — evaluated first because the rest need a candidate row |
| 2 | `NanInf` | any component of the candidate row is not finite — **unconditional**, not configurable (INV-12; the IR has no such variant by design) |
| 3 | `StaleObservation` | `obs_age > max_age` |
| 4 | `InferenceDeadline` | `micros_since(last_chunk_tick, now) > budget` — the budget measures how long the plane has gone without a fresh chunk |
| 5 | `ControllerHeartbeat` | `micros_since(last_beat_tick, now) > timeout` |
| 6 | `SensorDropout` | for any configured sensor, `micros_since(last_seen, now) > max_gap` |
| 7 | `EnvelopeViolationRate` | `window.fraction() > max_frac`, using the window **as of the previous step** — this step's own clamp has not been recorded yet, which is what keeps the rule non-circular — and the window never records this watchdog's own trip, which keeps it non-circular one step later too (P-M3-W1-R7) |

**A partially filled window never trips** (P-M3-W1-R1). `window.fraction()` is
`ones / window`, not `ones / steps-seen-so-far`, and reads `0.0` until the ring holds `window`
steps. Dividing by the steps seen so far makes the *first* dirty step of a run a rate of
exactly `1.0`, so any `max_frac < 1.0` trips on step two — and because the fallback step is
itself dirty, the rate stays `1.0` and the plane never leaves the fallback (with
`EmergencyStop`, it latches). A HIL cold start is the quickest way in (its first tick has no
chunk, so `ChunkUnderrun` fires), but one transient clamp at step 3 of any deployment does the
same. `window` is the warm-up period; there is no separate grace-period field, and §10.3's
own example acceptance of `<= 0.01` only reads as written if the denominator is the window.

**The watchdog does not measure its own fallbacks** (P-M3-W1-R7). The window records a step as
a violation iff it was dirty for a reason *other than* `ViolationRate`:
`window.push(!events.without(ViolationRate).is_empty())`. Without that clause every step after
the trip is a fallback step whose only event is the watchdog's own, the ring refills with that
echo, `ones` never decays and the trip is permanent — a latch, for *every* fallback policy, that
`is_latched()` does not report and `reset_latch()` cannot clear. With it, release is automatic
and bounded: the watchdog stops firing at most `window` steps after the last genuine violation
leaves the ring, which is what §18.5 ("a fallback is normal behaviour, not a failure") asks for.
A step dirty for any real reason — a clamp, `NanInf`, `ChunkUnderrun`, `HeartbeatLoss`,
`SensorDropout`, `StaleObservation`, `InferenceDeadline` — still counts, so a plane that keeps
genuinely violating keeps the watchdog tripped. Only the watchdog's *input ring* changes:
`dirty_steps`, `clamped_steps`, `fallback_activations`, `steps` and `violations[ViolationRate]`
still count every fallback step including the watchdog's own, which is what §10.3's
episode-level `envelope_violation_rate` in `es-eval` reads. `EmergencyStop` is untouched — §9.4
attaches "latch" to that one policy, so a genuine rate trip still latches and only
`reset_latch()` clears it (`tests/properties.rs:an_estop_rate_trip_still_latches`).

`ChunkUnderrun` and `NanInf` are armed always. The other five are armed only if the IR lists
them; an unlisted watchdog never trips, which is a configuration choice, not a disabled
safety layer — the envelope clamps below always run.

`micros_since` is integer: `ticks * period_us`, where `period_us = 1e6 * den / num` of the
control rate.

**Step 4a — fallback path** (any watchdog tripped). Produce the fallback action (below),
then run it through the final scrub (non-finite → hold, then hard position clamp) so the
fallback output is inside the envelope by the same rule as the policy output.
`source = Fallback(kind)`, `fallback_activations += 1`. The violation window records this
step as a violation unless its only event was the rate watchdog's own trip (see row 7). Per
§18.5 this is *normal operation*: the env stays `Ok` and the event is recorded.

**Step 4b — clamp path** (nothing tripped). The candidate row is clamped in exactly this
order, each stage recording its `ViolationKind` if it changed the value:

1. **NaN/Inf** — cannot happen here (watchdog 2 already caught it); the scrub stays as a
   defensive no-op so the stage order is the same on both paths.
2. **position** — clamp to the soft limit `[lower + margin, upper - margin]` (§9.3
   `position_limit`).
3. **velocity** — implied velocity `(a - last_safe) / dt_s`, where `last_safe` is the plane's
   own last command and never the measured joint (§9.3, §9.5 — see the seed rule above); if
   `|v| > vel_max[i]`, set `a = last_safe + sign(v) * vel_max[i] * dt_s`.
4. **acceleration** — implied acceleration `(v - vel[i]) / dt_s`; if over `acc_max[i]`,
   clamp the velocity to `vel[i] ± acc_max[i] * dt_s` and rebuild `a` from it.
5. **torque** — only when `action.space == JointTorque`, where the action *is* a torque:
   clamp `|a| <= tau_max[i]`. A no-op stage in the other spaces, which is why it sits here
   rather than in a separate branch.
6. **workspace** — only when `action.space` is `EePose` or `EeDelta` and `NJ >= 3`: project
   components `[0..3]` into the box / cylinder / convex hull (§9.3 `workspace`). Joint-space
   actions are not projected: this crate has no forward kinematics, and inventing one here
   would be a second, unvalidated robot model. Joint-space workspace enforcement belongs to
   the collision packet.
7. **rate limit** — first difference `|a - last_safe| <= d1_max[i]`, then second difference
   `|a - 2*last_safe + prev_safe| <= d2_max[i]` (§9.3 `rate_limit`). Applied last so it also
   smooths whatever the earlier stages changed.

Then the hard position clamp runs once more as a final scrub, because stage 7 can push a
value back out of the soft band.

`source = Clamped` if any stage changed a value, else `Policy`. `cursor += 1` only on this
path — a fallback step does not consume a chunk row, so the buffer is still there when the
condition clears.

**Step 5 — state update.** `prev_vel = vel`, `vel = (out - last_safe) / dt_s`,
`prev_safe = last_safe`, `last_safe = out`, `steps += 1`, window push. Identical on both
paths, so the hold target always tracks what was actually emitted.

`collision_constraint`, `jerk_limit`, `ee_velocity_max`, `contact_force_max` (§9.3) are
carried in the envelope but not enforced here: the first three need state this crate is not
given (contact set, FK, measured force). They are enforced where that state exists. The
envelope still transports them so `deployment_hash` covers them (§9.5).

## Fallback semantics (§9.4)

| Policy | Action produced |
|---|---|
| `HoldPosition` | `last_safe` unchanged |
| `ZeroVelocity` | velocity reduced toward zero by at most `acc_max[i] * dt_s` per step, `q = last_safe + v_new * dt_s`. Reaches and stays at rest in a bounded number of steps and never exceeds the acceleration limit. |
| `RetractToHome` | `trajectory[retract_idx]`, `retract_idx` saturating at the last waypoint. Deterministic stepping, one waypoint per fallback step; a completed retraction holds the final waypoint forever. The index does **not** reset when the condition clears — a half-finished retraction resumes rather than restarting. |
| `HandoffController` | `last_safe` (hold) with `source = Fallback(HandoffController)`. That variant *is* the flag: the caller switches to its classical controller on seeing it. This crate never runs one. |
| `EmergencyStop` | `last_safe`, and `estop_latched = true`. Every subsequent `validate` short-circuits at step 0 until `reset_latch()` is called explicitly. Nothing in `validate` clears the latch. |

A fallback is never a neural network and never depends on the policy runtime being alive
(§9.4): all five are table lookups over pre-allocated state.

## Counters (§10.3)

```rust
pub struct SafetyCounters {
    pub violations: [u64; ViolationKind::COUNT],   // per kind
    pub fallback_activations: u64,
    pub steps: u64,
    pub clamped_steps: u64,
    window: ViolationWindow,
}
```

- `envelope_violation_rate()` — fraction of steps in the sliding window that were clamped,
  projected or fell back for a reason other than the rate watchdog itself (row 7). This is
  the §10.3 first-class metric, and it is the same number the `EnvelopeViolationRate`
  watchdog reads; `es-eval`'s episode-level metric of the same name is `dirty_steps / steps`
  and does count every fallback step. The window is a `[u64; 4]` bit ring (cap 256)
  with a running ones-count, so the fraction is an integer ratio, not an accumulated float.
- `chunk_underrun_rate()` — `violations[ChunkUnderrun] / steps` (§8.6, §10.3).
- `violations[kind]` — per-kind counts for the `failure_mode_histogram` of §10.3.

Counters are monotone `u64`; nothing in the public API resets them except `reset_counters()`,
which does not touch the latch.

**The ring is per episode; the sums are not** (§9.4, packet M7/R1). `begin_episode` empties
`window` — and only `window`, beside the latch and the seed. The reason is §10.3's own rule for
the ring: the watchdog judges *a full window or nothing*, so a window straddling an episode
boundary is half of one stream and half of another, and episode `k` is judged partly on episode
`k-1`'s steps. §13.1 says an episode is where a stream ends, and this is the same boundary the
observation plan (`CpuPlan::reset`) and the chunk buffer (`PlaneFeed::end_episode`) already
take. Emptying the ring **re-arms** the watchdog, it does not disarm it (INV-12): the envelope,
every watchdog and every summed counter are untouched and the next violation latches again.

The sums stay the cell's on purpose. `violations`, `steps`, `clamped_steps`, `dirty_steps` and
`fallback_activations` are what `es-eval` divides into `envelope_violation_rate` and
`chunk_underrun_rate` over a whole cell (§10.3), so zeroing them at an episode boundary would
move every §10.1 row; `reset_counters()` remains the only way to zero one. What this cost, when
it was not done: `docs/design/evaluation-execution.md` 2.7 has the measurement — 227 `violation.rate`
events over four demo episodes, and a trajectory that parted company at tick 24.

## Violation scenario suite (§28.7 gate 8)

`tests/fixtures/safety/*.json`, replayed by `tests/scenarios.rs` against `NJ = 3`, `H = 4`,
`K = 3`. Each fixture carries a full `DeploymentIr` plus a step list; each step declares the
inputs (chunk or "reuse the previous one", `obs_age_us`, heartbeat, sensors seen) and the
expected `source`, expected event set, and optionally the expected `q`. Counters are checked
at the end of the run. Nineteen scenarios:

| # | Fixture | What it proves |
|---|---|---|
| 1 | `valid_chunk_passes` | A conforming chunk is emitted bit-identical, `source = Policy`, no events, `envelope_violation_rate == 0` |
| 2 | `position_limit` | A row past the soft joint limit is clamped to it, `source = Clamped` |
| 3 | `velocity_limit` | A step too large for `vel_max` is clamped to `last_safe ± vel_max*dt` |
| 4 | `acceleration_limit` | A velocity change too large for `acc_max` is clamped |
| 5 | `torque_limit` | `JointTorque` action space, `|a| > tau_max` clamped |
| 6 | `workspace_box` | `EePose` action outside the workspace box is projected onto it |
| 7 | `rate_limit` | First/second difference bounds clamp an otherwise in-limit action |
| 8 | `nan_action` | A `NaN` component falls back immediately; output finite (§9.3 `action_validity`) |
| 9 | `stale_observation` | `obs_age > max_age` falls back, then recovers when the age drops |
| 10 | `chunk_underrun` | Cursor runs past `valid`; fallback, counter, recovery on a new chunk (§8.6) |
| 11 | `chunk_truncation` | `valid = 4 > K = 3` under `RecedingHorizon`: row 3 is never executed, step 4 is an underrun |
| 12 | `inference_deadline` | No fresh chunk within the budget falls back, then recovers |
| 13 | `heartbeat_loss` | Missing `heartbeat()` past the timeout falls back; a beat restores `Policy` |
| 14 | `sensor_dropout` | One configured sensor goes quiet past `max_gap`; `sensor_seen` restores |
| 15 | `violation_rate` | Enough clamped steps to push the window fraction over `max_frac`, which trips the rate watchdog |
| 16 | `estop_latch` | E-stop trips once and every later step stays `Fallback(EmergencyStop)` even with a perfect chunk, until `reset_latch()` |
| 17 | `retract_completes` | `RetractToHome` steps waypoint by waypoint and holds the last one |
| 18 | `identical_chunks_fresh_seq` | A bit-identical chunk resubmitted with a strictly greater `seq` on each replan is still `source = Policy` (P-M1-R3) |
| 19 | `repeated_seq_is_stale` | A chunk resubmitted with the *same* `seq` as before is not accepted; the cursor runs past it into `ChunkUnderrun` (P-M1-R3) |

Plus `tests/properties.rs`: for arbitrary chunks containing `NaN`, `±Inf`, subnormals and
values far outside every limit, and arbitrary `obs_age`/tick sequences, the returned `q` is
always finite and inside the **hard** position limits, and `validate` never panics. And an
allocation test: 10,000 `validate` calls inside `assert_no_alloc`.

## Why no trait

There is one Safety Plane. `PhysicsBackend`, `PolicyRuntime`, `TaskNodeFactory`,
`LearningNodeFactory`, `InferenceBackend`, `Scalar`, `DeterministicAcc` are the seven
extension points (INV-17), and this is not one of them. A `trait SafetyPolicy` would be a
seam through which a caller could install a plane that does nothing, which is exactly what
INV-12 forbids.
