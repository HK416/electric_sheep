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
      cpu_plan.reset()                               # §2.4: the observation stream ends here
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

### 2.2 INV-15 — augmentation auto-disable, then refusal

§10.4 says augmentation is *auto-disabled* in evaluation, and that is what the plan does: a
`training_only` `Augment` node lowers to an identity pass-through
(`docs/design/observation-lowering.md` §3), so it cannot run here and needs no allow-list
entry. The refusal is what is left over.

Before anything runs, the Observation IR is scanned for an `ObservationNode::Augment` that is
**not** `training_only` — one that would actually execute. A node like that which is
not in `AugmentationPolicy::AllowList` is `EvalError::AugmentationEnabled`. The
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
| an `ObservationNode::ImageInput` | the frame source's bytes, unconverted | supported with `--frames` |
| `StateInput` whose id is in `ModelInfo::qpos` | that env's `qpos` slice | supported |
| `StateInput` whose id is in `ModelInfo::sensor` | that env's `sensordata` slice | supported |
| `StateInput` whose id is a Task IR `ObsSource::JointState { body, dof }` | the leading `dof` joint positions of env 0 | supported |
| anything else | — | `EvalError::Plan`, before the first episode |

Every input is resolved **once**, before the first episode, by `input_sources`. An input is
an image because the *Observation IR* says `ImageInput`, not because nothing else matched
it: the earlier rule ("any id in neither map is an image") served a 27,648-byte camera tile
to a 6-element joint-state buffer the moment a frame source existed, which is what packet
M5/V3 hit on the demo's own documents.

The `JointState { body, dof }` row is the one reading available for that channel: it names a
body and a DoF count, not joints, so capture takes the leading `dof` positions — the same
convention `joint_state::<NJ>` already uses to feed the Safety Plane, and the reason
`run_episode` refuses a model carrying fewer than `NJ` of them (§2.5).

Without a frame source (`Evaluation::run`, or `es eval run` with no `--frames`) an image
observation is refused by name rather than fed zeros. A cell that silently evaluated a
policy on black frames would produce a number, and a wrong number in this table is worse
than no table.

### 2.4 Plan state is per episode (P-M2-R1)

One `CpuPlan` is compiled per run, but a `TemporalWindow` ring is a *stream*, and a stream
ends where an episode does. `run_episode` therefore calls `CpuPlan::reset()` right after
`env.reset`, which refills every ring to the state `compile` left it in
(`docs/design/observation-lowering.md` §9.1). Without it the first frames of every episode
after the first would carry the previous episode's tail, and the first episode of cell 2 would
carry cell 1's — so the §10.1 table would depend on the order the suites were declared in.
The oracle reverses the suite order over a windowed observation and asserts every cell is
unchanged.

Note what this does *not* cover: a `Perturbation` draw is keyed by the suite's **position**
(`EnvRng::new(seed, cell_index, episode, stream)`), so reordering suites does legitimately
change a perturbed suite's draws. The order-independence oracle therefore uses two
perturbation-free suites. Keying the stream on the suite *name* instead is a separate
question about §10.4's `suite_id`, not about plan state.

### 2.5 `nu != NJ` is an error (P-M2-R7)

`NJ` is the deployment's joint count, and the loaded model has to agree: `run_episode` refuses
with `EvalError::JointMismatch` unless `model.nu == NJ` and the model carries at least `NJ`
`qpos`/`qvel` entries. Nothing is broadcast and nothing is padded — a `ctrl` vector filled
with copies of joint `NJ−1`, or a safety input padded with `0.0`, is a wrong number in the
§10.1 table, which is worse than a refused run.

## 3. Perturbation realisation (`perturb.rs`)

`PerturbationPlan::compile(&EvaluationIr, &SceneDesc, &ModelInfo, has_renderer)` resolves
every perturbation of every suite **once**, before any episode runs, so the per-episode path
has no matching on strings and cannot fail. A kind this runtime cannot realise is
`EvalError::Unsupported(kind)` naming it at compile time — never skipped silently, and
never approximated (same rule as `RandomizationPlan` in `batch-domains.md` §5 and as
`PhysicsBackend::load` in §17.2).

`has_renderer` is whether the run was given a frame source (`Evaluation::run_with_frames`,
i.e. `es eval run --frames` on a build with the `render` feature). The two lighting kinds are
realisable only then; without one they are refused by name rather than drawn and dropped.
`model` is unused today; it is in the signature because every kind in the "state mutation"
group below resolves a target against it the moment it is implemented.

### 3.1 Realised now

| kind | how | applied |
|---|---|---|
| `action_delay` | ring buffer of control vectors, depth `round(ms / control_period_ms)`; the ring is filled with the reset action, so the first steps command the hold pose rather than zeros | per step |
| `observation_delay` | the captured observation is held for `round(ms / control_period_ms)` steps and `obs_age` handed to `SafetyPlane::validate` grows accordingly, so the `StaleObservation` watchdog sees the delay | per step |
| `frame_drop` | Bernoulli `prob` per step; a hit drops a burst of `[lo, hi]` consecutive frames, during which the previous observation is reused and `obs_age` keeps growing | per step |
| `torque_noise` | multiplicative `1 + N(0, rel_sigma)` on each control channel, drawn per step per channel | per step |
| `backlash` | a per-episode deadband of `[lo, hi]` rad: a commanded change smaller than the band does not move the actuator | per step |
| `light_intensity` | a gain drawn from `range`, applied to every geom's `rgba` in a clone of the scene before the `TriScene` upload. The `Rs` path shades `albedo * (ambient + n.l * (1 - ambient)) + emission`, which is *linear* in `albedo`, so scaling the colours is exactly scaling the incident radiance. Only `dist = "uniform"` has a kernel; the other two are refused by name. | per episode |
| `light_direction` | a yaw drawn from `[-range_deg, range_deg]`, applied to `RenderConfig::light_dir` about `+Z` through `es_math::approx::sin`/`cos` (never `std`'s, §3.4) | per episode |

The two lighting kinds are `LightOverride { intensity, yaw_deg }`, drawn in `apply_at_reset`
like every other per-episode knob and handed to the frame source with every frame. The
*renderer* is the caller's (`es-eval` is layer 10 and links no Vulkan, `visible-learning.md`
§7.4), so `LightOverride::scene` and `::rotate_dir` are the kernels and the caller applies
them; `es eval run` rebuilds its renderer only when the draw changes, so a suite with no
light perturbation builds exactly one for the whole run.

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
| `light_intensity`, `light_direction` | **nothing, given a frame source.** Without one (`Evaluation::run`, or `es eval run` with no `--frames`) there is no rendered image to perturb, and they are refused with that reason. |
| `color_temperature` | a coloured light. The `Rs` path shades from one white directional light and `RenderConfig` carries no light colour, so there is nothing to set; adding one is an `es-render` change (layer 5). |
| `camera_extrinsic`, `camera_intrinsic` | `ImageSpec` intrinsics rewriting at capture (INV-14) — an intrinsic perturbation that skipped the `ImageSpec` transform would be a silent lie about the camera. A renderer alone does not unblock these. |
| `occluder` | scene-graph insertion: an occluder is a geom the Task IR did not declare, and a scene with one would no longer be the scene `scene_hash` names. |
| `object_pose` | a per-episode reset override. `Env::reset` takes no state and `Env` owns its backend, so `es-eval` cannot write `qpos` before a step. The hook is an `Env::reset_with(&ResetOverrides)` in a follow-up `es-env` packet; `ResetOverrides` is already shaped to carry it. **The demo does not need it**: Task IR `Randomization` (§6.3) already moves the cube's free joint at every reset, in every suite (`visible-learning.md` section 2.7). |

That is 7 realised of 12, 2 of them only with `--frames`. The gate for M2 W1 (§28.4,
"Evaluation IR 전 스위트 동작") is therefore still **not** met by these packets alone. The
refusal is loud so that a report can never claim a `lighting_shift` row it did not run.

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
  cumulative counters instead: `counters.dirty_steps / counters.steps`.
  **Answered (M2 W1b).** `clamped_steps` and `fallback_activations` used to be counted on
  different branches of `validate`, summed and capped at `steps`; a future branch that
  incremented both would have under-reported by the overlap. `SafetyCounters::record_step`
  (called once from `SafetyPlane::finish`, the single tail every `validate` path returns
  through) now sets `clamped_steps` and/or `fallback_activations` and increments
  `dirty_steps` by at most one regardless of how many of the two are true, so the metric is
  exact even for a step that is both at once — see
  `es_safety::counters::tests::a_step_that_is_both_clamped_and_a_fallback_counts_once`.
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

**`Unavailable` is not a pass.** `EvaluationReport::passed` is true only when every
`AcceptanceResult` is `Determined { passed: true, .. }`.

> **Answered (M2 W1b).** `es_ir::evaluation` now has `MetricValue::Unavailable { reason }`
> and `AcceptanceResult::Unavailable { metric, reason }` (the latter turned `AcceptanceResult`
> from a bare struct into an enum, `#[serde(untagged)]` so the old `{criterion, observed,
> passed}` shape still round-trips as the `Determined` variant). `run` returns
> `(EvaluationReport, EvaluationLock)` directly; the `EvalReport` wrapper, `Unmeasured`,
> `Verdict` and `Outcome` are gone from `es-eval` — every declared metric gets one
> `CellResult` (measured or `MetricValue::Unavailable`) and every acceptance line one
> `AcceptanceResult` (`Determined` or `Unavailable`), so there is nothing left for a wrapper
> to add.

## 6. Artifacts (§10.5)

```
write_artifacts(&report, &lock, dir)
  → report.json        es_ir::evaluation::EvaluationReport, written as-is (§10.5)
  → evaluation.lock    evaluation_hash + execution_hash + seeds + backend capabilities
```

`report.html` and `episodes/` (replay, failures first per `ReplayPolicy`) are **a later
packet**: the HTML needs the table layout the editor already renders, and replay needs the
episode serialisation format of §23. `EvaluationReport::episodes` is therefore written
empty, not populated with paths to files that do not exist.

### `report.json`

```jsonc
{                                    // es_ir::evaluation::EvaluationReport, §10.5
  "schema_version": 1,
  "evaluation_hash": [32 bytes],
  "execution_hash":  [32 bytes],
  "cells": [
    { "suite": "nominal", "metric": "success_rate",
      "value": { "scalar": 0.92 }, "n_episodes": 100 },
    { "suite": "nominal", "metric": "collision_rate",
      "value": { "unavailable": { "reason": "no contact reporting in this backend" } },
      "n_episodes": 100 }
  ],
  "acceptance": [
    { "criterion": {...}, "observed": 0.92, "passed": true },
    { "metric": "collision_rate", "reason": "no contact reporting in this backend" }
  ],
  "passed": false,
  "episodes": []
}
```

The two `acceptance` shapes are `AcceptanceResult::Determined` and `::Unavailable`; the
enum is `#[serde(untagged)]`, so which one a line is is read off which fields it has, not
a tag.

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
