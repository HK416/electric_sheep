# W2 — batch domains: round-robin, async inference, chunk buffer, determinism (`es-env`)

Spec: §12.1 (four domains), §12.2 (`round_robin` camera selection, obs→inf batch
composition), §12.3 (determinism under asynchronous inference), §12.4 (the 9-metric set and
the bandwidth budget), §8.5 (action chunks, `H` / `K = execute_chunk`), §8.6 (async inference,
chunk buffer, `ChunkBlendPolicy`, underrun as a safety event), App. B.5 (`ChunkArrival`),
§9.4 (fallback), §28.4 W2. Design note: `docs/design/batch-domains.md` §9–§12.

## context

```
crates/es-env/Cargo.toml
crates/es-env/src/lib.rs
crates/es-env/src/chunk_buffer.rs
crates/es-env/src/inference.rs
crates/es-env/src/domains.rs
crates/es-env/src/env.rs          (additive: Env::step_with_policy only)
docs/design/batch-domains.md      (append §9–§12)
docs/packets/M2/W2-batch-domains.md
```

`es-eval` is being built concurrently on the existing `es-env` surface, so this packet is
**additive only**: new modules, new `pub use` lines, one new method on `Env`. No existing
`pub` item is renamed, removed or re-typed, and no existing test expectation changes.

## spec

- **`chunk_buffer`** — `ChunkBuffer<NJ, H>` per env: `new(execute_chunk, ChunkBlendPolicy)`,
  `push(&ActionChunk<NJ, H>, arrival_tick)`, `next_action(tick) -> Option<[f64; NJ]>`.
  `CHUNK_SLOTS = 8` inline slots, no heap in `next_action` (the overlap set is an inline
  `[usize; CHUNK_SLOTS]`, insertion-sorted by push sequence so the reduction order is fixed).
  Three blends: `HardSwitch` (latest wins), `LinearBlend { steps }` (ramp from the previous
  chunk), `TemporalEnsemble { weight_decay }` (ACT: `w_i = exp(-m·i)` with `i = 0` the
  **oldest** overlapping prediction, via `es_math::approx::exp`, never `f64::exp` — §3.4).
  `K` bounds a chunk's span under the first two blends; `TemporalEnsemble` reads all `valid`
  rows, which is what ACT averages. No covering chunk ⇒ `None` and an underrun counter tick.
  It never fabricates an action.
- **`inference`** — `AsyncInference`: a FIFO queue of `Submission { env, submit_tick, inputs }`
  with `poll(tick)` releasing at most `inference.batch` submissions whose
  `submit_tick + latency_ticks <= tick`, in submit order. `latency_ticks(expected_latency_ms,
  control)` turns `RuntimeHints::expected_latency_ms` into whole control ticks, rounding up,
  with the single `f32 → µs` conversion at the edge. Simulated, never wall-clock (§12.3); the
  module documents the upgrade path to a real worker thread and why the release rule must stay
  here rather than in the worker.
- **`domains`** — `DomainRunner<NJ, H>` over a `Schedule`:
  `observe_window` runs the observation plan for exactly the envs
  `Schedule::observation_envs(t)` selects on each observation tick of the control window
  (§12.2); `infer_window` submits ascending by env id on inference ticks, polls, stacks the
  released batch into one `PolicyRuntime::infer` call and pushes the split chunks at
  `apply_at = submit_tick + latency_ticks` (App. B.5 — the arrival tick is *not* the apply
  tick); `emit_actions` pulls one row per env and hands it, or an empty chunk on underrun, to
  `SafetyPlane::validate`. `DomainSizing` is the allocation-free §12.4 arithmetic for a
  configuration, including the `GATE` constant (4,096 × 512).
- **`Env::step_with_policy(runner, policy, planes, plans)`** — one control step in §12.1 phase
  order. `planes` is one `SafetyPlane` per env (its hold target, rate-limit history and latch
  are per-robot state, §9.3); `plans` is empty (raw `qpos ‖ qvel` observation) or one
  `CpuPlan` per env (`TemporalWindow` rings are per-env, §7.5). Episodes that ended clear
  their buffer and drop their queued inference.

### Invariants

- **`INV-12`**: `emit_actions` is the only path from a chunk to `ctrl`, and both its branches
  — action and underrun — go through `SafetyPlane::validate`. There is no bypass, not in
  tests; the test envelope is *widened*, never disabled.
- **§12.3**: nothing reads a clock. Latency, apply time, observation age and the round-robin
  slot are all integer functions of the tick.
- Layering: `es-env` (9) gains `es-policy` (8); `es-compile` (7) and `es-safety` (8) were
  already present. `es-safety` still has no policy dependency (rule 8, `INV-11`).

## oracle

```
cargo fmt -p es-env --check
cargo clippy -p es-env --all-targets -- -D warnings
cargo test -p es-env
cargo xtask layering
cargo xtask context-budget
```

## acceptance

- Chunk-buffer blend math on hand-computed cases: two overlapping chunks with ACT `m = 0.01`
  match `(1·a_old + e^-0.01·a_new) / (1 + e^-0.01)` to 1e-12; `LinearBlend` ramps 0 → ½ → 1;
  `HardSwitch` serves `K` rows then underruns; an exhausted buffer returns `None`; eviction
  and slot placement do not change the result.
- Async inference: release is exactly `submit_tick + latency_ticks` when the batch keeps up;
  the released *sequence* is a prefix-stable function of the schedule across
  `batch ∈ {1, 2, 3, 4, 8, 16}` — a narrower batch rations throughput, never reorders; a run
  replays bitwise.
- A 16-env `FakeBackend` + constant-chunk `FakePolicy` run of 200 control steps is bitwise
  identical across two runs for each `observation.batch ∈ {4, 8, 16}`.
- **The batch-independence invariant, stated precisely.** Round-robin changes *which* envs
  observe *when*, and only an observed env gets a chunk, so per-env trajectories are **not**
  independent of `observation.batch`. What holds: envs whose schedule-derived observation
  ticks coincide are indistinguishable. Asserted by grouping the 16 envs of each configuration
  by their observation-tick set (`16 / observation.batch` groups) and requiring equality inside
  a group and inequality between groups; plus `underrun_rate(4) > underrun_rate(8) >
  underrun_rate(16)`, which is the mechanism.
- `DomainSizing::GATE` reproduces §12.4's arithmetic without allocating: 30,720 frames/s,
  1.54 Gpixel/s, 4.6 GB/s RGB8, 18.5 GB/s after `f32` normalization, 36.7 MB of chunk buffers
  for 4,096 envs; with every camera on, §12.4's 245,760 frames/s, 12.3 Gpixel/s, 37 GB/s and
  148 GB/s — the 8× that justifies `round_robin`. **`Target / Status: unverified`**: the
  §28.4 gate (4,096 sim env × 512 obs env stable) is a budget here, not a measurement.

## forbidden

- Renaming, removing or re-typing any existing `es-env` `pub` item, or changing an existing
  test expectation (`es-eval` is being built on that surface concurrently).
- `crates/es-policy`, `crates/es-compile`, `crates/es-data`, `crates/es`, the root
  `Cargo.toml`, and every other crate — neighbouring packets own them.
- A real inference thread (§12.3 is determinism-first; threading is a later throughput
  packet), `EnvSelection::Subset`, real sensor/camera observations (M2 W3), batched policy
  *lowering* semantics, wiring `EnvMetrics` to `es-telemetry` (layer 10).
- `HashMap`, `std` transcendentals on the chunk path, wall-clock anywhere in the decision
  logic, and any new extension-point trait (`INV-17`).
