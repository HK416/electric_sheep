# Evaluation IR execution (`es-eval`) — design

Spec refs: §10 (Evaluation IR), §10.2 (document structure), §10.3 (metric definitions),
§10.4 (determinism and fairness), §10.5 (artifacts), §12.4 (the nine performance metrics),
§9.3–§9.4 (envelope violation rate is a first-class metric), §5.3 (`execution_hash`),
§8.6 (chunk underrun), §11.3 (CPU reference plan), §18.5 (failure semantics),
§28.4 M2 W1, §28.7 gate 11.

Invariants: INV-15 (augmentation is off during evaluation), INV-12 (no path disables the
Safety Plane), INV-17 (no new extension points).

Review class: C — read this before the code.

## 1. What this crate is

`es_ir::evaluation` is the **declaration**: suites, perturbations, metrics, acceptance,
seeds. This crate **runs** it. It owns exactly three things:

1. turning each `PerturbationKind` into something that happens during a run (`perturb.rs`),
2. turning finished `Episode`s plus `SafetyCounters` into `MetricValue`s (`metrics.rs`),
3. the cell loop, the hash chain, and the two artifacts of §10.5 (`runner.rs`).

It adds no trait. `PhysicsBackend` and `PolicyRuntime` are the two extension points it
consumes, both already reserved by INV-17.

## 2. The cell loop

```
for suite in ir.suites                     # a row of the §10.1 table
  for episode in 0 .. ir.episodes.n_episodes
      overrides = plan.apply_at_reset(suite_id, episode)
      env.reset()
      loop
          inputs  = capture(env.backend().state())      # §2.3
          if plan step-drops this frame: reuse the held observation, age it
          obs     = cpu_plan.run(inputs)                 # §11.3
          chunk   = policy.infer(obs)
          action  = plane.validate(chunk, obs_age, tick) # INV-12, always runs
          ctrl    = plan.apply_per_step(action)          # delay, backlash, torque noise
          env.step(ctrl)
      until done or max_steps
```

One `Env`, one `SafetyPlane` and one `CpuPlan` **per cell**, not per run: the plane's
counters are the cell's `envelope_violation_rate` denominator, and the plan's
`TemporalWindow` rings are a per-episode stream that must not leak between suites.

`Env::new` consumes its backend, so `run` takes `new_backend: impl FnMut() -> B` rather
than one `B`. This is a deliberate deviation from the packet's sketched signature and the
reason is fairness (§10.4): `Env` keeps a per-env episode counter that keys its
`RandomizationPlan` draws, so a single `Env` shared across cells would give the
`lighting_shift` row a different set of base scenes than the `nominal` row and the two
table rows would no longer be comparable. A fresh `Env` per cell restarts that counter at 0.

`n_envs` is 1. Batching the cell loop across the simulation domain is M2 W2
(`round_robin`, §12.2), not this packet.

### 2.1 Determinism contract (§10.4)

> equal `evaluation_hash` + equal `execution_hash` ⇒ bitwise-identical `report.json`.

What makes that hold here:

- **Every perturbation draw** comes from `EnvRng::new(seed, suite_id, episode_idx, stream)`
  — literally §10.4's `TaskRng(seed_base, suite_id, episode_idx, stream)`, with `seed` from
  `SeedPlan` and `stream = StableId::from_path("es.eval.perturbation.<stream>")` from
  `Perturbation::stream`. `EnvRng` is counter-based: the *n*th draw depends on the key and
  *n* only, so the sequence does not depend on the policy, on the backend, on how many
  envs exist, or on which cell ran first. Changing the policy cannot change the
  perturbations; that is the whole point of the table.
- **No wall clock in the report.** The only timestamp in the crate is
  `EvaluationLock::created`, supplied by the caller (`RunConfig::created`, default 0) and
  written only to `evaluation.lock`. `report.json` has no time in it.
- **Deterministic iteration and ordering.** `BTreeMap` only. Episodes are aggregated in
  `(cell_index, seed)` order, cells are emitted in
  `(suite declaration order, MetricSpec::ALL order)`, so the JSON is byte-stable.
- **No global RNG, no `f64` time accumulation, no `std` transcendentals** on the sampling
  path — `EnvRng::sample` already routes through `es_math::approx` (§3.4).
- The two escape hatches are named in the lock, not hidden: `execution_hash` covers the
  runtime and the hardware capability, `evaluation_hash` covers the conditions.

### 2.2 INV-15 — augmentation refusal

Before anything runs, the Observation IR is scanned for `ObservationNode::Augment`. A node
that is not in `AugmentationPolicy::AllowList` is `EvalError::AugmentationEnabled`. The
graph is **not** rewritten and the node is **not** stripped: silently editing the
observation pipeline would change `observation_hash` relative to what the caller thinks it
evaluated, and the report would then attest to a graph that was never declared. The author
either takes the node out of the Observation IR or writes the allow-list with its
justification, which is what ends up in the report's conditions.

The allow-list entry is matched against the node's `NodeId` in decimal (`"7"`). There are
no node labels in `es-ir` (rule 7: no UI types in the IR; labels live in `.eslayout`), so
the id is the only stable key available. When a label sidecar is wired in, matching on the
label is the upgrade path and the allow-list strings do not have to change shape.

### 2.3 Observation capture

The `CpuPlan`'s input buffers are named by the `StableId` of the `ImageInput` sensor or
`StateInput` source (`Home::Input(id.to_string())`). Capture resolves each name against
`ModelInfo`:

| plan input | source | status |
|---|---|---|
| `StateInput` whose id is in `ModelInfo::qpos` | that env's `qpos` slice | supported |
| `StateInput` whose id is in `ModelInfo::sensor` | that env's `sensordata` slice | supported |
| `ImageInput` (any id not in either map) | a rendered frame | `EvalError::Unsupported` |

`es-render` (layer 5) does not exist yet, so there is no camera in this build and an image
observation is refused by name rather than fed zeros. A cell that silently evaluated a
policy on black frames would produce a number, and a wrong number in this table is worse
than no table.

## 3. Perturbation realisation (`perturb.rs`)

`PerturbationPlan::compile(&EvaluationIr, &SceneDesc, &ModelInfo)` resolves every
perturbation of every suite **once**, before any episode runs, so the per-episode path has
no matching on strings and cannot fail. A kind this runtime cannot realise is
`EvalError::Unsupported(kind)` naming it at compile time — never skipped silently, and
never approximated (same rule as `RandomizationPlan` in `batch-domains.md` §5 and as
`PhysicsBackend::load` in §17.2).

`scene` and `model` are unused today; they are in the signature because every kind in the
"scene mutation" group below resolves a target against them the moment it is implemented.

### 3.1 Realised now

| kind | how | applied |
|---|---|---|
| `action_delay` | ring buffer of control vectors, depth `round(ms / control_period_ms)`; the ring is filled with the reset action, so the first steps command the hold pose rather than zeros | per step |
| `observation_delay` | the captured observation is held for `round(ms / control_period_ms)` steps and `obs_age` handed to `SafetyPlane::validate` grows accordingly, so the `StaleObservation` watchdog sees the delay | per step |
| `frame_drop` | Bernoulli `prob` per step; a hit drops a burst of `[lo, hi]` consecutive frames, during which the previous observation is reused and `obs_age` keeps growing | per step |
| `torque_noise` | multiplicative `1 + N(0, rel_sigma)` on each control channel, drawn per step per channel | per step |
| `backlash` | a per-episode deadband of `[lo, hi]` rad: a commanded change smaller than the band does not move the actuator | per step |

`ms` lists (`observation_delay`, `action_delay`) are a `Choice` distribution: one value is
drawn per episode, so a cell with `ms: [0, 20, 50]` mixes the three conditions across its
episodes exactly as §10.2 writes it.

The draws that are fixed for an episode (`ms`, backlash band) happen in `apply_at_reset`
and land in `ResetOverrides`. The draws that are per step (`frame_drop`, `torque_noise`)
happen in `apply_per_step` against a `StepState` the runner owns. `ResetOverrides` is
named for the hook it will become; today it carries no state override, because every
state-mutating kind is in §3.2.

### 3.2 `Unsupported` today

| kind | blocked on |
|---|---|
| `light_intensity`, `light_direction`, `color_temperature` | a renderer. `es-render` (layer 5) is not implemented; there is no light to perturb. |
| `camera_extrinsic`, `camera_intrinsic` | the same, plus `ImageSpec` intrinsics rewriting at capture (INV-14) — an intrinsic perturbation that skipped the `ImageSpec` transform would be a silent lie about the camera. |
| `occluder` | a renderer and scene-graph insertion. |
| `object_pose` | a per-episode reset override. `Env::reset` takes no state and `Env` owns its backend, so `es-eval` cannot write `qpos` before a step. The hook is an `Env::reset_with(&ResetOverrides)` in a follow-up `es-env` packet; `ResetOverrides` is already shaped to carry it. |

That is 5 realised of 12. The gate for M2 W1 (§28.4, "Evaluation IR 전 스위트 동작") is
therefore **not** met by this packet alone; it needs the renderer waves. The refusal is
loud so that a report can never claim a `lighting_shift` row it did not run.

## 4. Metrics (`metrics.rs`)

`compute(metric, episodes, counters, env_metrics) -> Measured`, where
`Measured` is `Value(MetricValue)` or `Unavailable(reason)`. **Nothing is invented**: a
metric this runtime does not measure is `Unavailable` with a reason string, never `0.0`.

| metric (§10.3) | computed from | status |
|---|---|---|
| `success_rate` | `Episode::termination == Success` over the cell | measured |
| `episode_length` | mean `Episode::steps()` | measured |
| `envelope_violation_rate` | `(clamped_steps + fallback_activations) / steps` of the cell's `SafetyCounters`, capped at 1 | measured |
| `chunk_underrun_rate` | `SafetyCounters::chunk_underrun_rate()` (§8.6) | measured |
| `action_smoothness` | `1 / (1 + mean |Δctrl| + mean |Δ²ctrl|)` over each episode's control trace, averaged | measured |
| `failure_mode_histogram` | `Episode::termination` and `Episode::failure` buckets, plus a `fallback` bucket from `SafetyCounters::fallback_activations` and one bucket per non-zero `ViolationKind` | measured |
| `intervention_rate` | human intervention on hardware/HIL (§24.2) | `Unavailable` — no HIL path in this build |
| `collision_rate` | unwanted contact | `Unavailable` — `PhysicsBackend` reports no contacts yet |
| `domain_gap` | real-log replay distance (§24.3) | `Unavailable` — M3 |
| the §12.4 performance set | pass-through of `EnvMetrics`, which is `Option` per field | `Unavailable` per field when the field is `None` |

Two notes on the definitions.

- **`envelope_violation_rate` is cumulative over the cell, not the watchdog's window.**
  `SafetyCounters::envelope_violation_rate()` is the sliding fraction the §9.4 rate
  watchdog reads; the §10.3 metric is the whole-cell rate, so it is computed from the
  cumulative counters instead. `clamped_steps` and `fallback_activations` are counted on
  different branches of `validate`; the sum is capped at `steps` so a future branch that
  increments both cannot produce a rate above 1. **ceiling:** if the plane ever grows a
  step that is both clamped and a fallback, this under-reports by the overlap; the fix is a
  single `dirty_steps` counter in `es-safety`, which is a one-line change there.
- **No `step/s`.** The performance row is the nine metrics of §12.4 and nothing else.

Aggregation across episodes is `Aggregation::{Mean, Min, Max, P95}` over a `Vec<f64>`
sorted by `(cell_index, seed)`. `P95` is the nearest-rank order statistic on the sorted
sample — no interpolation, so it is exactly reproducible. `mean`, `std` and `ci95` (normal
approximation, `1.96 * std / sqrt(n)`) live here as three small functions;
`es eval compare` (layer 11, a later packet) is what needs a real Welch test between two
reports, and it is above this crate.

## 5. Acceptance

Each `AcceptanceCriterion` expands over the suites it names (`suite: None` means every
suite, §10.2) and yields one `Verdict`:

- `Pass { observed }` / `Fail { observed }` when the metric was measured,
- `Unavailable { reason }` when it was not.

**`Unavailable` is not a pass.** `EvalReport::passed` is true only when every verdict is
`Pass`. This is the part of the schema that does not fit `es-ir`'s
`AcceptanceResult { criterion, observed: f64, passed: bool }`: there is no way to say
"not measured" in an `f64`, and writing `observed: 0.0, passed: false` invents a
measurement. So `es-eval` owns `Verdict` and `report.json` is `EvalReport`, which embeds
the §10.5 `EvaluationReport` (carrying only the measured cells and the resolvable
acceptance lines) and adds `unmeasured` and `verdicts`.

> **Reviewer question.** The clean fix is `MetricValue::Unavailable` and an
> `AcceptanceResult::Unavailable` variant in `es_ir::evaluation`, which would delete
> `EvalReport` and let `run` return `EvaluationReport` directly. That is an `es-ir` change
> and out of this packet's scope.

## 6. Artifacts (§10.5)

```
write_artifacts(&report, &lock, dir)
  → report.json        EvalReport: cells x suites, unmeasured, verdicts
  → evaluation.lock    evaluation_hash + execution_hash + seeds + backend capabilities
```

`report.html` and `episodes/` (replay, failures first per `ReplayPolicy`) are **a later
packet**: the HTML needs the table layout the editor already renders, and replay needs the
episode serialisation format of §23. `EvaluationReport::episodes` is therefore written
empty, not populated with paths to files that do not exist.

### `report.json`

```jsonc
{
  "report": {                       // es_ir::evaluation::EvaluationReport, §10.5
    "schema_version": 1,
    "evaluation_hash": [32 bytes],
    "execution_hash":  [32 bytes],
    "cells":      [ { "suite": "nominal", "metric": "success_rate",
                      "value": { "scalar": 0.92 }, "n_episodes": 100 } ],
    "acceptance": [ { "criterion": {...}, "observed": 0.92, "passed": true } ],
    "passed": false,
    "episodes": []
  },
  "unmeasured": [ { "suite": "nominal", "metric": "collision_rate",
                    "reason": "no contact reporting in this backend" } ],
  "verdicts":   [ { "criterion": {...}, "suite": "nominal",
                    "outcome": { "pass": { "observed": 0.92 } } } ]
}
```

### `evaluation.lock`

JSON, not TOML: the same `serde_json` with `float_roundtrip` that `evaluation_hash` already
depends on for its transport guarantee (`es_ir::evaluation` module docs), and one fewer
serialiser to keep canonical.

```jsonc
{
  "schema_version": 1,
  "evaluation_hash": "hex32",
  "execution_hash":  "hex32",
  "seeds": [20260912, ...],          // the resolved per-episode seeds, in order
  "backend": { "name": "mujoco-cpu", "determinism": "Bitwise", "float": "F64",
               "max_envs": 1024, "gpu_resident": false,
               "supports_reset_subset": true, "supports_state_get_set": true,
               "quirks": ["..."] },
  "created": 0                       // caller-supplied unix seconds; 0 = unset
}
```

Hashes are hex in the lock (a human reads it) and raw bytes in the report (`es-ir`'s own
serde shape). `created` is the only non-reproducible field in either artifact and it is
deliberately confined to the lock.

### `execution_hash` assembly (§5.3)

`run` builds the `HashChain` from what it is given:

| slot | source |
|---|---|
| `asset`, `scene` | `TaskIr::scene.asset_hash` / `scene_hash` |
| `task_graph` | `canonical_hash(&task.graph)` |
| `task`, `observation`, `deployment`, `evaluation` | the IRs' own `*_hash()` |
| `learning`, `policy` | `PolicyInfo::lowering_hash` / `weights_hash` — the graph itself is not passed to `run`, and these are the two digests the loaded runtime can attest to |
| `compiler` | `CpuPlan::compiler_hash()` |
| `runtime` | `PolicyRuntime::runtime_hash()` |
| `dataset`, `hardware` | `RunConfig` — an evaluation run reads no dataset, so the caller supplies zeros or the training set's digest |

`evaluation` is in the chain but not in `execution_hash` by design (§5.3: the evaluation
conditions do not change what is executed); the report carries both.
