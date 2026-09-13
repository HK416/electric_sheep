# W6 — env runtime foundations (`es-env`)

Spec: §12 (batch domains), App. B.5 (scheduler), §6.3–6.4 (Randomization / ResetState /
Terminate / Reward, execution semantics), §18.1 (integer time), §18.5 (failure semantics),
§13.1 (episode recording). Design note: `docs/design/batch-domains.md` (review class C).

## context

```
crates/es-env/src/lib.rs
crates/es-env/src/scheduler.rs
crates/es-env/src/rng.rs
crates/es-env/src/plan.rs
crates/es-env/src/randomize.rs
crates/es-env/src/episode.rs
crates/es-env/src/env.rs
docs/design/batch-domains.md
docs/packets/M1/W6-env-runtime.md
```

## spec

- **`scheduler`** — `BatchDomains { simulation, observation, inference, training: Option }`,
  `DomainCfg { batch, period (sim ticks), device }`. `Schedule::build` validates §12.1 / B.5
  (sim period 1; inference period an integer multiple of the observation period; training a
  multiple of inference; the batch funnel only narrows) and expands one hyper-period
  `lcm(obs, inf, train)` into a static `Vec<TickPlan>`. `ticks()` iterates it; `at(tick)`
  indexes it; `observation_envs(tick)` is §12.2's deterministic `round_robin`. Pure data — no
  backend, no clock, no `f64`.
- **`rng`** — `EnvRng`, counter-based splitmix64 keyed by `(seed, env, episode, stream_id)`.
  No `rand` dependency, no global state (§3.4), no sequential coupling between envs. Covers
  all five `es_ir::task::Distribution` variants; `ln`/`exp`/`cos`/`sqrt` come from
  `es_math::approx`, not `std` (§3.2 `DET-010`).
- **`randomize`** — `RandomizationPlan::compile(task, scene, model)` resolves every
  `ResetState` / `Randomization` node's target string once. Supported:
  `qpos[i]`, `qvel[i]`, `joint.<name>.qpos|qvel`, `body.<name>.mass`,
  `geom.<name>.friction`, `actuator.<name>.gain`. Anything else, and a degenerate
  distribution, is `EnvError::Unsupported` naming it — never silently skipped.
- **`plan`** — lowers the scalar cone feeding each `Reward` / `Terminate` sink into
  `es_ir::task::Expr` (the evaluator `es-ir` already exposes; no second one is written).
  Supported cone nodes: `GetJointState`, `GetSensor`, `GetTime`, `Arith`, `Compare`, `Clamp`.
- **`episode`** — `EpisodeRecorder` with columnar per-env `Vec`s pre-sized from
  `max_episode_steps`; `finish(env) -> Episode` carrying ticks, qpos, qvel, ctrl, sensordata,
  reward, done, failure, termination and the drawn parameter scales. `Termination` is
  evaluated from the lowered predicates, then the episode budget.
- **`env`** — `Env<B: PhysicsBackend>` with `new`, `reset(Option<&[u32]>)`,
  `step(&[f64]) -> StepOutcome { rewards, dones, failures, episodes }`, `metrics()`. One
  `step` is one control step = `inference.period` simulation ticks; terminated envs auto-reset
  at the end of the same call. `EnvMetrics` carries the §12.4 names, all `Option`, and has no
  `step/s` field.

Constraints: `BTreeMap` only, no new traits (INV-17), no f64 time accumulation, English only.

## oracle

```
cargo fmt -p es-env --check
cargo clippy -p es-env --all-targets -- -D warnings
cargo test -p es-env
cargo xtask layering
cargo xtask context-budget
```

## acceptance

- Scheduler: a table of good and bad `BatchDomains` builds or is rejected with a message
  naming the offending domain; the plan is a pure function of the domains; every inference
  tick is also an observation tick; round-robin covers every env exactly once per cycle.
- RNG: proptest — the same `(seed, env, episode, stream)` always gives the same sequence, and
  changing any component changes it; a stream is addressable, not affected by draws elsewhere.
- Randomization: every supported target resolves against the fixture scene; unknown targets,
  out-of-range indices and degenerate distributions are errors that name the target;
  `ResetState` runs before `Randomization`; draws vary with env and episode.
- Episode: column widths are `steps() * width`; a short row is padded; `finish` rolls the id.
- Env: two envs with the same reset draw produce identical 100-step trajectories, and
  different seeds produce different ones; the same seed replays exactly; termination
  auto-resets and hands back the episode; the budget yields `Timeout`; a backend-reported
  divergence quarantines the env and drops it from the reward; a bad `ctrl` length is rejected.

## forbidden

- `es-compile` and `es-safety` (in flight in neighbouring packets) — declared as dependencies,
  deliberately unused here. The Safety Plane hook lands with `es-safety` (INV-12: it will not
  be optional).
- `crates/es-data`, any other crate, the root `Cargo.toml`, and committing.
- Depending on `es-telemetry` (layer 10): `EnvMetrics` is a plain struct it converts.
- Rendering, the Observation IR pipeline, async inference / chunk buffers, and pushing model
  parameter scales into the backend (`PhysicsBackend` has no parameter API) — all listed as
  deferred in `docs/design/batch-domains.md` §8.
