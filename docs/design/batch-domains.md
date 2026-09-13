# Batch domains, reset and the env RNG (M1 W6)

Spec §12 (four batch domains, inter-domain policy, determinism, the 9 metrics), Appendix B.5
(scheduler sketch), §6.4 (execution semantics), §18.1 (integer time), §18.5 (failure
semantics), §13.1 (episode recording feeds the learning loop). Review class C.

This is what `es-env` actually implements in M1 W6, and — as importantly — what it does not.

## 1. Four domains, four sizes

Spec §12.1 rejects "one env is the unit of everything". Each domain has its own batch size and
its own period, and `es-env` never derives one from another:

| Domain | `batch` | `period` (sim ticks) | Device |
|---|---|---|---|
| simulation | `N_sim` (e.g. 4096) | **always 1** — it defines the tick | `Cpu` / `Gpu(id)` |
| observation | `N_obs` (e.g. 512) | `sensor_dt / dt_phys`, e.g. 33 at 1 kHz | |
| inference | `N_inf` (e.g. 256) | `dt_ctrl / dt_phys`, a multiple of the obs period | |
| training | `N_train` (e.g. 64 sequences), optional | a multiple of the inference period | |

`DomainCfg.period` is an integer count of simulation ticks, never a duration (§18.1: no float
time, enforced by the type — there is no `f64` anywhere in `BatchDomains`). The wall-clock
meaning of a period comes from the backend's `TickRate`, which `Env` reads from `ModelInfo`.

### Validation (`Schedule::build`)

Rejected, each with its own `EnvError::Schedule` message:

- any `batch == 0`, or a `period == 0`;
- `simulation.period != 1`;
- `inference.period % observation.period != 0` — an inference tick must land on an observation
  tick, otherwise it would consume an observation from an undefined tick;
- `training.period % inference.period != 0`;
- `observation.batch > simulation.batch`, `inference.batch > observation.batch` (§12.1: the
  funnel only ever narrows);
- `observation.batch == 0` with a non-`All` selection.

## 2. The static plan

`Schedule` is pure data: a hyper-period `H = lcm(obs, inf, train)` and a `Vec<TickPlan>` of
length `H`, where `TickPlan { offset, observation, inference, training }` says which domains
fire on sim tick `t` (`plan[t % H]`). Nothing about the plan depends on how fast anything runs,
so two machines interleave identically. `Schedule::ticks()` iterates one hyper-period.

Firing order **within** a tick is fixed by §6.4's phase order and is not data:

```
PrePhysics -> Physics -> PostPhysics -> Observation -> Reward -> Termination -> Record
```

so on a tick where both fire, observation precedes inference, and inference never sees an
observation newer than its own tick.

### Camera env selection (§12.2, §12.3)

When `N_obs < N_sim` the active-camera set is a deterministic function of
`(tick, obs_period, N_obs, N_sim)` — no RNG, no load feedback:

```
cycle  = ceil(N_sim / N_obs)
slot   = (tick / obs_period) % cycle
active = [slot * N_obs, min((slot + 1) * N_obs, N_sim))
```

That is §12.2's `round_robin(k, period)` with `k = N_obs`; every env's camera is visited once
per `cycle` observation ticks, and the order is env-ID ascending, as §12.3 requires.

## 3. Reset protocol

`reset(None)` resets the whole batch; `reset(Some(&envs))` resets a subset and requires
`capabilities().supports_reset_subset` (otherwise `EnvError::Physics(Unsupported)`; the backend
names it, we do not emulate it). Per env, in this order:

1. `episode[env] += 1` — the episode counter is part of every RNG key, so a re-reset never
   replays the previous episode's draws.
2. Zero that env's row of the reset buffer (`qpos`, `qvel`), then apply the `ResetState` and
   `Randomization` nodes in ascending `NodeId` order (§6.4: node ID breaks ties, so a graph
   rewrite that preserves IDs preserves the draw sequence).
3. `backend.reset(envs, Some(&state))` with the randomized rows, other envs' rows carried
   through unchanged.
4. `EpisodeRecorder::start(env)` — the previous episode is finished and handed back.

Auto-reset: `step` evaluates termination after the physics step; envs that terminated are
reset at the *end* of the same `step` call, so the episode boundary is a tick boundary and the
caller never observes a half-reset batch. The done flag returned belongs to the terminating
tick, not to the fresh episode.

Quarantined envs (§18.5) are still stepped by the backend but are excluded from reward
reduction and from the recorded episode; a `FailureKind::NanDetected` / `Diverged` from
`StepReport` drives that, via `FailurePolicy` from `es-core`.

## 4. RNG

Spec §3.4 forbids a global RNG and §6.6 `DET-001` makes the stream name mandatory on every
random source. `EnvRng` is therefore **counter-based**, not sequential-stateful:

```
key     = splitmix64_chain(seed, env_index, episode, stream_id.bytes[0..8], bytes[8..16])
nth     = splitmix64(key ^ (counter * GOLDEN))
```

Consequences that the tests pin:

- The same `(seed, env, episode, stream)` gives the same sequence, always, on any machine and
  regardless of what other envs did — no shared state, so env order and thread count are
  irrelevant.
- Different `env`, `episode` or `stream` give unrelated sequences; envs are not correlated
  even though they share a seed.
- A stream can be re-derived at any point without replaying the ones before it, which is what
  makes subset reset cheap and replay exact.

`stream_id: StableId` is `StableId::from_path(stream_name)` of the `Randomization`/`ResetState`
node's declared stream, so renaming a stream changes the draws (and the task hash) and nothing
else does.

**Distributions** (`es_ir::task::Distribution`, all five covered):

| Variant | Method |
|---|---|
| `Constant(v)` | `v`, no draw consumed |
| `Uniform{lo,hi}` | `lo + (hi - lo) * u`, `u = (x >> 11) as f64 * 2^-53` |
| `LogUniform{lo,hi}` | `exp(ln lo + (ln hi - ln lo) * u)`; `lo > 0` required |
| `Normal{mean,std}` | Box–Muller, `mean + std * sqrt(-2 ln u1) * cos(2 pi u2)`, fixed op order, second variate discarded |
| `Choice(vs)` | `vs[(next_u64 % len)]`, rejection-free (bias < 2^-64 for any realistic len) |

Transcendentals go through `es_math::approx` (`ln`, `exp`, `cos`, `sqrt`), not `std`
(§3.2 `DET-010`). Those are `f32`; a randomization draw does not need `f64` precision, and
bit-identical draws across x86/ARM/GPU matter more than the last few mantissa bits. Draws are
still returned as `f64` because the state arrays are.

## 5. Randomization targets

`RandomizationPlan::compile(task, scene)` resolves each node's `target` string against
`SceneDesc` **once**, at construction, so the per-reset path has no string work and no
`Result` to unwrap:

```
qpos[i]  qvel[i]                       state indices
joint.<name>.qpos  joint.<name>.qvel   resolved via ModelInfo's joint -> IndexRange maps
body.<name>.mass                       scale factor
geom.<name>.friction                   scale factor
actuator.<name>.gain                   scale factor
```

Anything else is `EnvError::Unsupported(target)` at compile time — never silently skipped
(§17.2's rule, applied to randomization).

**Ceiling, deliberate:** `PhysicsBackend` has no model-parameter API (it exposes `set_state`,
not `set_body_mass`). The three scale targets are therefore *sampled, recorded into
`EpisodeMeta.param_scales`, and not pushed into the backend*. The plan is complete and the
draws are part of the hash chain the moment a `set_param` lands on the trait; state targets
(`qpos`/`qvel`) are applied for real today.

## 6. Reward and termination

`Reward` and `Terminate` are graph sinks with an input edge, not expression literals. `es-env`
lowers the *scalar cone* feeding each sink into an `es_ir::task::Expr` once, at construction
(`plan.rs`), then evaluates it per env per tick with `Expr::eval`, which already exists and is
already specified as deterministic (fixed op order, `None` instead of `NaN`).

Supported in a cone: `GetJointState`, `GetSensor`, `GetTime`, `GetContact` (leaves, bound to
named ports), `Arith`, `Compare`, `Clamp`. Every other node kind in a reward or terminate cone
is `EnvError::Unsupported("<kind> in a reward cone")`. Port names are
`joint.<id>.<quantity>[i]`, `sensor.<id>[i]`, `time`, `contact.<a>.<b>`.

Reward aggregation is `sum(weight * term)` over `Reward` nodes in ascending `NodeId`; `Mean`,
`Min`, `Max` aggregate the elements of a vector term and are single-element no-ops here.

## 7. Metrics

`EnvMetrics` carries the §12.4 names, every field `Option` — a metric nobody measured is
`None`, never a fabricated zero, and there is **no `step/s` field at all**. `es-env` is layer 9
and `es-telemetry` is layer 10, so `EnvMetrics` is a plain struct with no telemetry dependency;
`es-telemetry` converts it into `PerfMetrics` from above. W6 fills
`physics_steps_per_sec`, `actions_per_sec` and the tick/wall-time counters it actually
observes; render, inference and VRAM fields stay `None` until those domains exist.

## 8. Deferred

- **Async inference and the chunk buffer** (§12.3, §8.6, B.5 `ChunkArrival`). The schedule
  already marks inference ticks; `apply_at = computed_from + deterministic_delay` and
  `chunk_underrun_rate` arrive with `es-policy` (layer 8).
- **`EnvSelection::Subset(Vec<EnvId>)`** — only `All` and the round-robin derivation above.
- **The observation domain's actual work.** W6 schedules observation ticks; rendering and the
  Observation IR pipeline are `es-render` / `es-compile` packets.
- **Safety Plane integration** (§9). `es-safety` is in-flight in a neighbouring packet; `Env`
  does not clamp actions yet. When it lands the hook is one call in `step`, before
  `set_ctrl` — and it will not be optional (INV-12).
- **Training domain execution** — scheduled, but the hand-off to `es-data` is W-later.
- **Backend model-parameter randomization** — see §5's ceiling.

---

# M2 W2 — round-robin execution, async inference, chunk buffer

§8 above listed four of these as deferred. This section replaces those entries; §1–§7 still
describe the M1 W6 foundation and are unchanged.

## 9. The runner

`DomainRunner<NJ, H>` executes a `Schedule` over an `Env`. Its clock is the **control tick**:
one `inference.period` window of simulation ticks, which is exactly what one `Env::step`
advances. One `Env::step_with_policy` is one control tick, in §12.1 phase order.

```
observation ticks in [t, t+inference.period)
    └─ Schedule::observation_envs(t)        round_robin, §12.2 — only these envs
         └─ CpuPlan::run (or the raw qpos‖qvel row)  → latest[env]
inference tick in the same window
    └─ submit(env, control_tick) ascending by env id, §12.3
poll(control_tick)
    └─ up to inference.batch, FIFO → one PolicyRuntime::infer over the stacked batch
         └─ ChunkBuffer::push(chunk, submit_tick + latency_ticks)     ← App. B.5 apply_at
every control tick
    └─ ChunkBuffer::next_action → SafetyPlane::validate → Env::step(ctrl)
```

The last arrow is the **only** path from a chunk to an actuator, and both of its branches take
it: an underrun hands the plane `ActionChunk::empty` and the plane answers with the fallback
(§8.6, §9.4). No branch reaches `set_ctrl` around the plane, including in tests — the test
envelope is widened, never disabled (INV-12).

Two things are arrays rather than singletons, both because the state behind them is per-env:
`planes: &mut [SafetyPlane]` (hold target, rate-limit history and the e-stop latch are
per-robot, §9.3) and `plans: &mut [CpuPlan]` (a `TemporalWindow` ring is a per-env history,
§7.5). A shared instance would interleave one env's history into another's.

## 10. Simulated latency (§12.3)

`latency_ticks = ceil(expected_latency_ms × control_rate)`, computed once, in integers, from
`RuntimeHints::expected_latency_ms`. It is the whole of the async model:

- A submission is released when `submit_tick + latency_ticks <= control_tick`, never earlier
  even if a real backend finished sooner — §12.3's deterministic mode waits.
- The chunk is applied at `submit_tick + latency_ticks`, **not** at the tick it arrived
  (App. B.5 `ChunkArrival::apply_at`). If a narrow `inference.batch` delivered it late, the
  rows already in the past are simply never served; that shows up as an underrun in the
  counters rather than as a silent time shift.
- Nothing reads a clock. `Instant` appears in `es-env` only in `EnvMetrics::simulation_wall`,
  which is a measurement, not an input to any decision.

Real threading is a later packet and belongs behind this same interface: `submit` hands work
to a worker, `poll` takes it back, and the release rule stays here so that swapping the worker
in cannot change semantics. Deterministic first (§12.3); real-time mode releases early and
records the divergence in the replay.

## 11. The chunk buffer (§8.5, §8.6)

`ChunkBuffer<NJ, H>` holds `CHUNK_SLOTS = 8` chunks inline. `next_action` allocates nothing:
the covering set is an inline `[usize; 8]`, insertion-sorted by push sequence, so the
combination order is arrival order and never slot order — which is what makes the ensemble sum
bitwise reproducible.

| `ChunkBlendPolicy` | rule | span of one chunk |
|---|---|---|
| `HardSwitch` | newest covering chunk wins | `min(valid, K)` |
| `LinearBlend { steps }` | ramp previous → newest over `steps` ticks | `min(valid, K)` |
| `TemporalEnsemble { weight_decay }` | `w_i = exp(-m·i)`, `i = 0` **oldest** | `valid` |

ACT indexes its exponential weights from the oldest overlapping prediction, so a smaller `m`
incorporates a new observation faster; `es_math::approx::exp` is used, not `f64::exp` (§3.4
forbids std transcendentals on a deterministic path). `K = execute_chunk` bounds the first two
blends because §8.5 replans after `K`; `TemporalEnsemble` deliberately reads all `valid` rows,
because averaging the overlap *is* the method.

`next_action` returns `None` on underrun and counts it. It never fabricates a row — a
fabricated action would reach the actuator having been checked against nothing.

## 12. What batch-independence actually means

A 16-env run replays bitwise for a fixed `observation.batch`. Across different
`observation.batch` it does **not**, and that is correct rather than a defect: round-robin
changes which envs are observed on which tick, only an observed env submits, only a submitting
env gets a chunk, and an env without a chunk gets the Safety Plane's fallback. Halving the
observation batch halves each env's observation rate and raises its `chunk_underrun_rate`.

The invariant that does hold, and the one the tests assert:

> Two envs are indistinguishable exactly when their observation ticks coincide.

Concretely, with `simulation.batch = 16` the envs partition into `16 / observation.batch`
round-robin groups; trajectories are identical inside a group and different between groups,
and `underrun_rate(4) > underrun_rate(8) > underrun_rate(16)`.

### Sizing the §28.4 gate

`DomainSizing` is integer arithmetic over a configuration — no allocation, so the gate
configuration can be costed on a laptop that cannot run it. For `GATE` (4,096 sim env,
512 obs env, 2 views, 224×224, 30 Hz, `H = 20`, `NJ = 7`):

| quantity | round-robin 512 | every camera on |
|---|---|---|
| `camera_frames_per_sec` | 30,720 | 245,760 |
| `pixels_per_sec` | 1.54 G | 12.3 G |
| render output (RGB8) | 4.6 GB/s | 37 GB/s |
| after `f32` normalization | 18.5 GB/s | 148 GB/s |
| chunk buffers (8 slots × `H` × `NJ` × f64) | 36.7 MB | 36.7 MB |

The 8× in the last two columns is the whole argument for §12.2. All of it is
**`Target / Status: unverified`** (§12.4): the §28.4 W2 gate — 4,096 sim env × 512 obs env
stable — is a budget here, not a measurement, and the executed test is the scaled 16-env
version.
