# W5 — Safety Plane runtime (`es-safety`)

Spec: §9 (Deployment IR과 Safety Plane), §9.3 (envelope), §9.4 (watchdog/fallback), §9.5
(sim/real identity), §8.5–8.6 (action chunk, chunk underrun), §10.3
(`envelope_violation_rate`, `chunk_underrun_rate`), §18.5 (fallback is normal operation),
Appendix B.4 (pinned shape), §28.7 gate 8 (type B).

Design: `docs/design/safety-plane.md` — read it first; it pins the clamping order, the
watchdog order and the fallback semantics this packet implements.

## context

```
crates/es-safety/Cargo.toml
crates/es-safety/src/lib.rs
crates/es-safety/src/types.rs
crates/es-safety/src/config.rs
crates/es-safety/src/counters.rs
crates/es-safety/src/plane.rs
crates/es-safety/tests/scenarios.rs
crates/es-safety/tests/properties.rs
tests/fixtures/safety/*.json
docs/design/safety-plane.md
docs/packets/M1/W5-safety-plane.md
```

## spec

`es-safety` (layer 8) is the **runtime** for the Deployment IR `es-ir` already validates.

- `SafetyPlane<const NJ: usize, const H: usize>` with the Appendix B.4 fields: `envelope`,
  `watchdogs`, `fallback`, `state: SafetyState<NJ, H>` (pre-allocated), `counters:
  SafetyCounters`.
- `SafetyPlane::from_ir(&DeploymentIr) -> Result<Self, SafetyConfigError>` is the only
  constructor: `robot.n_joints == action.dim == NJ`, `action.horizon == H`, every limit
  finite, no duplicate watchdog, `EnvelopeViolationRate.window <= 256`. No `new`, no
  `Default`, no public field — an envelope-less plane must not be constructible (INV-12).
- `pub fn validate(&mut self, chunk: &ActionChunk<NJ, H>, obs_age: Micros, now: PhysTick)
  -> SafeAction<NJ>` — no `Result` (INV-13), no panic, always an executable action.
- `ActionChunk<NJ, H>` = `[[f64; NJ]; H]` + valid length + `ExecutionMode`.
  `SafeAction<NJ>` = `{ q: [f64; NJ], source: Policy | Clamped | Fallback(FallbackKind),
  events: EventSet }` (a `u32` bitset over `ViolationKind`).
- Watchdogs (§9.4): `ChunkUnderrun`, `NanInf` (unconditional), `StaleObservation`,
  `InferenceDeadline`, `ControllerHeartbeat`, `SensorDropout`, `EnvelopeViolationRate`, in
  that evaluation order. `heartbeat(now)` and `sensor_seen(name, now)` feed the last two.
- Fallbacks (§9.4): `HoldPosition`, `ZeroVelocity` (decelerate within `acceleration_max`),
  `RetractToHome` (pre-computed trajectory stepped deterministically, saturating at the last
  waypoint), `HandoffController` (hold + the `Fallback(HandoffController)` flag),
  `EmergencyStop` (latched until `reset_latch()`).
- Clamp order: NaN/Inf → position → velocity → acceleration → torque → workspace → rate
  limit, then a hard-limit scrub.
- Determinism (§3.4): pure function of (config, state, inputs). Ticks only, no float time
  accumulation, no `HashMap`, no RNG, no global state.
- Zero heap allocation after construction on the hot path.

## oracle

```
cargo fmt -p es-safety --check
cargo clippy -p es-safety --all-targets -- -D warnings
cargo test -p es-safety
cargo xtask layering
cargo xtask check-spec-refs
```

## acceptance

- `tests/scenarios.rs` replays every `tests/fixtures/safety/*.json` fixture (≥ 12; 17
  shipped) and asserts the expected `source`, event set and end-of-run counters per step:
  each limit kind exceeded, NaN action, stale observation, chunk underrun, chunk truncation
  at `execute_chunk`, inference-deadline miss, heartbeat loss, sensor dropout,
  violation-rate trip, e-stop latch persistence, retract trajectory completion, and a valid
  chunk passing through bit-identical. This is §28.7 gate 8.
- `tests/properties.rs`: for arbitrary chunks containing `NaN`/`±Inf`/out-of-range values
  and arbitrary tick and `obs_age` sequences, `validate` never panics and the returned `q`
  is always finite and inside the hard position limits.
- An allocation test drives 10,000 `validate` calls inside
  `es_core::alloc_count::assert_no_alloc` (dev-dep `es-core/alloc-count`).
- `cargo xtask layering` still reports `es-safety` at layer 8 with no `es-policy` dependency
  (rule 8, INV-11).
- Crate source ≤ ~1,500 lines, English only, no new trait (INV-17).

## forbidden

- Any dependency on `es-policy`, or any type named after a policy, network or tensor
  (INV-11).
- Any way to disable, bypass or no-op the plane: no `enabled` flag, no `Option<Envelope>`,
  no `#[cfg(test)]` shortcut in `validate`, no `unsafe` (INV-12). A test that needs room
  widens the fixture's envelope.
- Changing `validate` to return `Result`, `Option`, or to take `&self` (INV-13).
- Touching `crates/es-compile`, `crates/es-env`, `crates/es-data`, `crates/es-ir`, or the
  root `Cargo.toml` — concurrent packets own those.
- Adding forward kinematics, a collision checker or a contact model here. `ee_velocity_max`,
  `min_self_distance`, `min_env_distance` and `contact_force_max` are transported for the
  hash (§9.5) and enforced where that state exists.
- New extension traits (INV-17).
