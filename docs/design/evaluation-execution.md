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
          if tick % replan == 0: inference.submit(obs, tick)   # section 2.6, packets V17 / T7
          for s in inference.poll(tick)                        # released after `latency` ticks
              buffer.push(policy.infer(s.inputs), s.tick + latency)   # App. B.5
          chunk   = plane_chunk(buffer, feed, tick)      # section 2.6, packet V6b — empty on an underrun
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
| `StateInput` whose channel says `quantity = Velocity` and whose id is a **joint** | `dof` values from that joint's first dof, `qvel[start..start + dof]` | supported (packet M8/S4e) |
| `StateInput` whose channel says `quantity = Velocity` and whose id is a **body** | the leading `dof` joint velocities of env 0 | supported (packet M8/S4e) |
| `StateInput` whose channel is `ObsSource::BodyPose(b)`, `b` not a free joint | `xpos[row*3..][..3] ‖ xquat[row*4..][..4]`, the quaternion xyzw (§3.1) | supported (packet M8/S4e) |
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

**Quantity and body pose (packet M8/S4e).** `ObsSource::JointState` carries the
`JointQuantity` `GetJointState` has always carried; absent is `Position`, which is the
canonical form every earlier document was hashed in, so no committed `task_hash` moves. The
quantity is read *before* the `qpos` and `sensor` maps, so a `Position` channel resolves
exactly as it did and a `Velocity` channel is never answered by a `qpos` range that happens to
share its id. A velocity channel naming a **joint** starts at that joint's first dof and takes
`dof` values from there — the offset is the one thing the documents cannot know, and the width
is the one thing the model cannot; naming a **body** it takes the leading `dof` of the row, the
mirror of `Joints`. `JointQuantity::Torque` is refused by name: `StateView` has no joint torque
array.

`ObsSource::BodyPose(b)` reads `b`'s row of `xpos` and `xquat`. A **free-joint** body never
takes that arm: its pose is in `qpos`, the `Qpos` arm answers it first, and that is what keeps
the demo's cube channel byte for byte what it was. The quaternion is written in the order
`StateView` documents — xyzw (§3.1) — and is not reordered here; reordering is an Observation
IR node's job, not capture's.

**One buffer per source id.** `CpuPlan` names an input buffer `Home::Input(id.to_string())`, so
two channels on one source id would be allocated one buffer between them and `exec::read_input`
would hand both `StateInput` nodes the same tensor — a silent wrong number, which is what §10.1
forbids. The Cross-IR check *accepts* the pair (`XIR-002` tells two channels on one id apart by
their declared type, whose unit is `JointQuantity::unit()`); `input_sources` refuses it by name
until the lowering can give them two buffers. That is why `tests/fixtures/rl/task-reach.toml`'s
`joint_vel` names the block's first joint, `shoulder_pan`, and not the body `base`.

Neither new kind has a model-free reading. `observation.state` in a recorded dataset is
`qpos ‖ qvel`, but where the second half begins is `nq` — a property of the model that ran, not
of the row — and `xpos`/`xquat` are not in the row at all. Both are named and refused in
`bake.rs` rather than turned into a guessed offset.

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

### 2.6 Inference latency and the chunk buffer (packets M5/V6b, M5/V17, M7/T7)

Three packets turned "the evaluator calls the policy" into "the evaluator runs the policy
the way the deployment says it runs on the robot". They are one rule each, and each one is
the *collector's* rule, reached by calling the collector's function rather than by
restating it:

| what | `es-env` function both paths call | packet |
|---|---|---|
| a chunk drives `action.execute_chunk` ticks | `es_env::plane_chunk` | M5/V6b |
| the policy is asked once per `rate.control / rate.inference` | `es_env::replan_interval` | M5/V17 |
| a chunk is executed at `computed_from + latency` | `es_env::AsyncInference` + `latency_ticks` | M7/T7 |

The third is §8.6's normal case: inference latency larger than the control period is not an
error, so the runtime models it. `AsyncInference` is a queue over *ticks*, never wall-clock
(§12.3) — a submission at tick `t` is released at `t + latency_ticks(expected_latency_ms,
rate.control)`, and the chunk it produces is pushed into the `ChunkBuffer` with `apply_at =
t + latency` (App. B.5: `apply_at = computed_from + deterministic latency`, never the tick
the result happened to come back on). A fast host and a slow host therefore replay
identically, and so do the collection and evaluation paths.

Three consequences worth stating plainly:

- **Tick 0 of every episode is a chunk underrun.** Nothing has been computed yet, so
  `plane_chunk` hands the plane an empty chunk and the plane answers with its own fallback
  and its own `ViolationKind::ChunkUnderrun`. That is not a bypass or a widening (INV-12):
  the underrun is the plane's event, it is counted in `chunk_underrun_rate`, and it shows in
  `events.json` as `source: "Fallback"` on frame 0. `es loop collect` has looked exactly
  like this since M2.
- **The policy sees the observation of the tick it was submitted on**, not of the tick the
  result is applied on. That is what the real pipeline does, and it is why the chunk is
  stale by `latency` ticks by the time it executes.
- **`expected_latency_ms = 0` stays legal** and means zero ticks: submit and poll happen in
  the same tick and the loop is exactly what it was before T7. It is a document choice
  (`RuntimeHints`), and a claim no real robot can honour.

The number is the Learning IR's (`PolicyContract::runtime::expected_latency_ms`), and
`Evaluation::run` is handed the four IRs it judges and never the `LearningGraph` — the same
reason `hash_chain` takes `learning` and `policy` off the loaded `PolicyInfo`. So it crosses
on `RunConfig::expected_latency_ms`, which `es eval run` fills from the bundle it opened.
There is still exactly one latency *model*: `es_env::latency_ticks`, called by
`DomainRunner::new` and by the cell loop with the same two arguments.

The oracle is a trajectory, not a schedule: `collection_and_evaluation_draw_the_same_trajectory`
(`crates/es/tests/cli.rs`) runs one seed through `es loop collect` and through
`es_eval::Evaluation` and compares the two `.estraj` files bitwise to the last tick. Before
T7 it failed at tick 1 — see `docs/design/visible-learning.md` section 7.30 for the measured
values and for what the demo's numbers became.

### 2.7 Sharding: the unit is the cell, and why it is not the episode (packet M7/T8)

`es eval run --jobs N` partitions the **cells** round-robin over N worker processes
(`Evaluation::run_shard(.., shard: (index, count))`, cell `c` to shard `c % count`), and
`Evaluation::merge` puts them back in cell order and computes the report, the judgement and
the hash chain once. The partition is fixed by the cell index alone, so it does not depend on
how long a suite takes, on how many workers started, or on any iteration order (§3.4). A
one-suite evaluation therefore gets no speedup at all: the demo's 16-episode `nominal` suite
is one cell and runs one-wide however large `--jobs` is.

Packet M7/T8 asked whether the unit could be the `(cell, episode)` pair instead. The `Env`
side of that is now settled and shipped:

**`Env::seek_episode(env, episode)`** sets the counter the *next* `reset` draws from. It
touches nothing else — no reset, no backend call, no recorder state, not the physics tick —
and it is refused while an episode is open with steps recorded on it, so a seek is never a
silent discard. §6.3 keys every reset and randomization draw by `(seed, env, episode, stream)`
alone, and `MuJoCoCpuBackend::reset` runs `mj_resetData` before writing the state, so a seek
*is* the replay. Measured (`cargo test -p es-env --test seek -- --ignored`, oracle server,
2026-09-21): for `k ∈ {1, 3, 7}` on the demo scene and the committed Task IR, a fresh `Env`
seeked to `k` and a fresh `Env` reset `k` times agree bitwise on `qpos`, `qvel`, the episode's
`ParamScales` and the whole 45,784-byte `.estraj` of an expert-driven episode.

**The evaluator side does not hold, and the partition is not shipped.** What a cell carries
from episode `k-1` into episode `k`, beyond the `Env`: the `SafetyPlane` (`begin_episode`
clears the latch and re-arms the seed, but **not** `SafetyCounters::window`, the sliding
`ViolationRate` ring of §9.4), the `ChunkBuffer` + `PlaneFeed` and `AsyncInference` (all
cleared per episode), the monotonic `seq` (only ordering is judged), and the plane's and env's
counters (sums, which add). Two of those are visible in the artifacts, and both were measured
by running the pre-T8 and the T8 build of `es eval run` over the same committed demo documents
— nominal suite, seeds 101–104, `--frames`, `v14/trained-20000.esb`, oracle server,
2026-09-21 (`~/artifacts/plan-v/m7-t8/run-parity.sh`, `pre-j1/` vs `post-j1/`):

| what | cell-level (pre) | per-episode (post) |
|---|---|---|
| first differing `.estraj` | — | `nominal-01`, **tick 24** (`qpos[0]` 0.068231 → 0.071987) |
| `events.json` first difference | `nominal-01` record 0, `tick: 7200` | same record, `tick: 0` |
| `violation.rate` | 2318 | 2091 |
| `fallback` | 5740 | 5678 |
| `violation.position` / `.velocity` / `.acceleration` | 423 / 498 / 1456 | 261 / 421 / 1512 |
| `envelope_violation_rate` | 0.9998611 | 0.9990278 |
| `nominal-00` (episode 0) | — | byte-identical, all four artifacts |

Two independent mechanisms, and only one of them is `es-safety`'s:

1. **The `ViolationRate` window carries across the episode boundary.** The demo declares
   `envelope_violation_rate = { max_frac = 0.9, window = 200 }` and runs at an envelope
   violation rate of ~0.999, so at tick 0 of episode `k` the pre-T8 plane's ring is full of
   episode `k-1`'s dirty steps, reads ~1.0, and trips the §9.4 watchdog. A per-episode plane
   starts with an empty ring, which reads `0.0` until 200 steps are in it — so its first 200
   ticks do not trip, take the clamp path instead of the fallback path, and the trajectory
   diverges at tick 24. 227 fewer `violation.rate` events over four episodes.
2. **`StepEvent::tick` is the cell's cumulative physics clock**, `Env::tick()`, which is 7200
   at the start of `nominal-01` (1800 control steps × 4 substeps) and 0 for a per-episode env.
   This one is `es-eval`'s record, not the plane's: it would still move `events.json` even if
   the window were cleared.

Per the packet, the finding stands and the flag that would have enabled the partition does not
exist. Both questions are the M7 review's:

- should `SafetyPlane::begin_episode` clear `SafetyCounters::window`? Clearing it is arguably
  the correct reading of "an episode is where a stream ends" (§13.1, and the same argument
  §2.4 makes for the observation plan), and it would be an INV-12-clean change — the envelope,
  the watchdogs and the counters stay on, only the ring the watchdog reads is re-armed. It
  moves every committed number in the table above, so it is a decision and not a fix.
- should `StepEvent::tick` be the episode's tick rather than the env's? Frame `n` of a cell
  already carries `frame: n`; the absolute tick is only meaningful inside one cell, and an
  episode-relative tick would make `events.json` independent of how the run was scheduled.

If both go that way the partition is a small change on top of what is here (`ShardCell` gains
per-episode records, `record_cell` moves to `merge`, `run_shard` walks `suites × seeds` and
seeks) and the wall-clock table below says what it would buy.

#### Wall-clock, nominal suite, 16 episodes (`Target / Status: measured`)

Oracle server (Ryzen, 16 cores, RTX 4090), `--frames`, `v14/trained-20000.esb`, one suite,
seeds 101–116. The 1-minute load average beside each row is the box's at the moment the run
started; other agents share the machine.

| build | `--jobs` | workers actually spawned | wall (pass 2) | load at start | pass 1 |
|---|---|---|---|---|---|
| cell-level (shipped) | 1 | 1 | 121.6 s | 0.55 | 120.3 s |
| cell-level (shipped) | 4 | 1 (clamped to the suite count) | 117.3 s | 2.31 | 120.0 s |
| cell-level (shipped) | 8 | 1 (clamped) | 120.6 s | 2.83 | 118.7 s |
| per-episode (T8, not shipped) | 1 | 1 | 122.9 s | 2.89 | 124.2 s |
| per-episode (T8, not shipped) | 4 | 4 | 60.6 s | 2.71 | 59.5 s |
| per-episode (T8, not shipped) | 8 | 8 | 55.2 s | 2.96 | 55.2 s |

Two passes, the second with a "wait until the 1-minute load is below 4" gate before each row
(pass 1 took three rows at a load of 3.6–6.0, the decay of the row before it). The two agree
to within 3 s on every row, which is the noise floor here.

What it says: the shipped partition is flat, exactly as `docs/design/visible-learning.md`
section 7.11 predicted — a one-suite evaluation is one cell, `--jobs` clamps to 1, and the
three cell-level rows differ only by noise. The per-episode partition is **2.0×** at four
workers and **2.2×** at eight; the second step is small because the box has 16 cores, each
worker's math-library pool is capped at `cores/N` (section 7.11 again), and every worker opens
its own MuJoCo and torch subprocess. Per-episode `--jobs 1` costs about a second more than
cell-level, inside the noise: sixteen `Env::new` + `mj_resetData` + plane constructions
instead of one is what episode isolation costs when nothing is parallel, and it is small.

That is the whole prize in one number: ~2 min of nominal evaluation instead of ~2 min × the
suites, or for the demo's six-suite sweep the same 2× again on top of the cell split. It is
bought with the two review decisions above, not with this packet.

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
