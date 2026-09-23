# M11 X4 — mass, friction and gain draws reach the physics

Spec: §28.14 rule 4 (model parameters reach physics through one method on the existing
`PhysicsBackend` trait; draws keyed by `(seed, env, episode, stream)` and recorded in the episode)
and rule 1 and wave 1, §6.3 (`Randomization`), §5 "Randomization targets", §17.2 (a backend without
the capability refuses by name), INV-17 (no new trait — a method on one of the seven is allowed).
Design notes: `batch-domains.md` (+ko) around line 148 (the ceiling "`PhysicsBackend` has no
model-parameter API") is replaced by what this packet builds. Type B.

## the question

`body.<n>.mass`, `geom.<n>.friction` and `actuator.<n>.gain` draws are resolved and recorded in
`EpisodeMeta.param_scales` but never pushed (`crates/es-env/src/randomize.rs:24`); MuJoCo CPU shares
one `MjModel` across envs (`crates/es-physics-backend/python/mujoco_ref.py:39-47`), so per-env
parameters cannot exist today. **With `PhysicsBackend::set_params`, does a drawn scale reach each
env's model at reset, reproduce a direct MuJoCo model edit bit for bit, and leave every scene that
declares no parameter target unmoved?**

## spec

* `PhysicsBackend::set_params(&mut self, envs: &[u32], params: &[(Param, StableId, f64)]) ->
  Result<(), PhysicsError>` — scales relative to the loaded model's value (`mass·s`,
  `friction[0..3]·s`, `gainprm[0]·s` and, for a position actuator, `biasprm[1]·s` so the servo
  stays a servo — state which in the doc). A default implementation returns
  `PhysicsError::Unsupported("set_params")`; `Capabilities` gains `Feature::ModelParams`.
  `Param` moves to (or is re-exported from) `es-physics-core` if it lives above it.
* MuJoCo CPU: `mujoco_ref.py` keeps **one `MjModel` per env** (copies of the loaded model, created
  lazily on the first `set_params`, so a run that never calls it keeps one model and today's
  bytes); a `SetParams` request writes the scaled values from the *original* model (never
  compounding across episodes) and calls `mj_setConst` only if the field requires it (document which).
  `Env::reset` calls `set_params` for the reset envs before pushing the state, when the plan has any
  `Scale` entry; the draw order and the recorded `param_scales` are unchanged.
* MJWarp: per-world model fields if `mujoco_warp` supports batched model arrays in the pinned
  version (the agent checks `docs/api-notes/mujoco-warp*.md` or the installed package and records
  it); otherwise no `ModelParams` capability and a Task IR with a scale target is refused by name on
  `mjwarp`. Newton: not declared.

## context

```
crates/es-physics-core/src/backend.rs
crates/es-physics-core/src/caps.rs
crates/es-physics-core/tests/**
crates/es-physics-backend/src/mujoco.rs
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/src/proc.rs
crates/es-physics-backend/src/mapping.rs
crates/es-physics-backend/python/mujoco_ref.py
crates/es-physics-backend/python/mjwarp_ref.py
crates/es-physics-backend/tests/**
crates/es-env/src/randomize.rs
crates/es-env/src/env.rs
crates/es-env/tests/**
tests/fixtures/mjcf/**
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/packets/M11/X4-set-params.md
docs/packets/M11/X4-set-params.ko.md
```

## oracle

1. `cargo test -p es-physics-core set_params_default_is_unsupported`.
2. `ES_PYTHON=… cargo test -p es-physics-backend --test set_params` — on a two-env scene, env 1's
   body mass ×1.5, a geom's friction ×0.5, an actuator gain ×1.2: 500 steps equal a direct Python
   `mujoco` run with the same model edits **bitwise** (qpos, qvel); env 0 equals an unedited run
   bitwise; applying the same scales twice (two resets) does not compound.
3. `cargo test -p es-env randomized_params_reach_the_backend` — a Task IR with `body.<n>.mass
   Uniform(0.8, 1.2)` over 4 envs: each env's measured `body_mass` (read back through a test hook or
   the mujoco reply) equals `nominal × recorded scale`; `seed` replays exactly.
4. Every committed task / scene / trajectory golden unmoved (the demo declares no scale target).
5. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

Oracles 1–5 locally and on the server (CPU queue), the `batch-domains.md` section rewritten (+ko),
the MJWarp capability decision recorded with its evidence.

## forbidden

A new trait; `es-safety`; compounding scales; a per-env model created when no parameter target
exists; claiming MJWarp support without running it.
