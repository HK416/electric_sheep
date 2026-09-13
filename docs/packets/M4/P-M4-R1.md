# P-M4-R1 — IR-C port map is per env

Spec: spec 6.2 / spec 6.4 (IR-C execution semantics), spec 6.6 (determinism: the batch answer
may not depend on the order envs are stepped in), spec 5.2 (independent batch domains).
Closes blocker **B-1** of `docs/reviews/M4.md`.

## context

```
crates/es-env/src/control.rs
crates/es-env/src/env.rs
docs/design/control-graph.md
docs/design/control-graph.ko.md
docs/packets/M4/P-M4-R1.md
```

## spec

`Env` keeps **one** `BTreeMap<String, f64>` port map for the whole batch and refills it per env
in `bind_ports`. Every IR-D binding (`qpos[i]`, `qvel[i]`, `sensor[i]`, `time`, `time.episode`)
is overwritten there, so those are per env already. The three stage ports
`stage.index` / `stage.ticks` / `stage.done` are not in that set: `ControlExecutor::write_ports`
inserts them, nothing removes them, so for env *e* they still hold env *e−1*'s values (for
env 0, env *n−1*'s from the previous step) at the two points that read the map **before** this
env's own `write_ports` runs:

1. `episode::evaluate` — the task-level `Terminate` cone (`env.rs`, `evaluate_env`).
2. the `Branch` evaluated once on graph entry (`control.rs`, `step` → `run(Some(root))`).

Any `Terminate` or entry `Branch` naming `stage.*` was therefore a function of a different env,
and swapping the order envs are stepped in changed the result.

The fix, at the one place both readers route through:

- `control.rs` names the three keys once (`STAGE_INDEX` / `STAGE_TICKS` / `STAGE_DONE`) and
  exposes `pub fn clear_stage_ports(&mut BTreeMap<String, f64>)`, which removes them.
- `Env::bind_ports` calls it first, before binding this env's IR-D sources. It is a no-op for an
  IR-D-only task (the keys are never present) and costs three `BTreeMap::remove`s per env.

Consequence, documented in `control-graph.md` §3.1: on the step a graph is entered no stage has
run yet for that env, so `stage.*` is **unbound** and an entry `Branch` naming one evaluates to
`None` — falsy, the `else_` arm — deterministically, for every env and every order. Per-env
executor scratch was the alternative; it buys nothing here, since the same "unbound at entry"
answer falls out of it, and it would duplicate the map.

`ControlExecutor` keeps its per-env `StageState` vector and its `step` signature; no stage
state moved, and IR-C still gates reward and termination only (INV-12 / INV-13 untouched).

## oracle

```
cargo test -p es-env control
```

Two tests, written before the fix:

- `stage_ports_do_not_leak_into_the_next_env` — two envs, root `Branch` on
  `stage.ticks > 0.5`, arms `then` / `else`. Neither env has spent a tick in a stage, so both
  must take `else`. On the old code env 0 takes `else` and env 1 takes `then`, because env 0's
  `stage.ticks = 1` was still in the map: **FAILED** before, `ok` after.
- `a_branch_is_per_env` — the R1 acceptance shape: two envs whose conditions genuinely
  disagree (`qpos[0] > 0`, driven by opposite torques). Each env gets the arm **its own** value
  selects, and swapping which env holds which torque swaps the stages rather than leaving them
  where the evaluation order put them.

## acceptance

- Both tests pass; the first one is confirmed to fail on the unfixed `bind_ports`.
- The eight pre-existing `control::tests` and the rest of `es-env` (62 tests) still pass,
  `two_runs_are_bitwise_identical` included.
- `cargo fmt --check -p es-env`, `cargo clippy -p es-env --all-targets -- -D warnings`,
  `cargo xtask layering`, `cargo xtask context-budget` clean.
- No public signature changed except the new free function `control::clear_stage_ports`.

## forbidden

Anything outside `context` — B-2 (`es-eval/src/evidence.rs`) and every other M4 finding belong
to their own packets. Changing `ControlExecutor::step`'s signature or its `StageState`.
Giving `Env` a second port map. Adding a stage port, renaming one, or changing what
`write_ports` computes. The `control.rs:158` timeout-vs-complete nit (a separate finding).
