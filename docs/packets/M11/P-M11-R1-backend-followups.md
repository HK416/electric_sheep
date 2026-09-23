# M11 R1 — backend follow-ups: the adapter script in the identity, PhysX in `Rollout`, vision RL through `es train`

Spec: §28.14 rule 2 (the backend is a condition of the run) and rule 1, §5.3, §13.4. Found by the
X1 follow-up, I1 and X3 reports (2026-09-23): (1) two fixes to `mjwarp_ref.py` and one to
`physx_ref.py` changed evaluation results without moving any `execution_hash`, because
`backend_identity` hashes the engine version and not the adapter script; (2) `es_native.Rollout`
still refuses `backend = "physx"`; (3) `es train`'s `[rl]` route refuses a bundle whose Observation
IR has an image input (`crates/es/src/cmd/train.rs`), so vision RL runs only by calling
`train_ppo.py` by hand. Type B.

## the question

**Does `execution_hash` move when a non-reference backend's adapter script changes, can `Rollout`
run on PhysX, and does `es train` run the vision reach recipe end to end — with every committed
`mujoco-cpu` hash unmoved?**

## spec

* `backend_identity(caps, engine_version, script)` hashes, after the existing fields, the blake3 of
  the adapter script the backend embeds (`mjwarp::SCRIPT`, `newton::SCRIPT`, `physx::SCRIPT`), under
  the same length-prefix rule. `mujoco-cpu` keeps its all-zero slot (§28.14 rule 2: the reference
  is pinned by its goldens instead). `evaluation.lock`'s `backend` block gains `script_blake3`.
* `Rollout(backend = "physx")`: the closed enum in `crates/es-py/src/rollout.rs` gains
  `Env<PhysXBackend>`; `[rl] backend = "physx"` parses; no other change to the rollout loop.
* `es train` `[rl]` accepts an image input when the build has the `render` feature (the maturin
  build now always has it — X3), passes the rollout documents as today, and records render cost in
  `metrics/env-metrics.json`; without the feature it still refuses by name.

## context

```
crates/es-physics-backend/src/lib.rs
crates/es-physics-backend/tests/**
crates/es-eval/src/runner.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
crates/es-data/src/training.rs
crates/es-py/src/rollout.rs
crates/es-py/src/pybind.rs
crates/es-py/tests/**
tests/golden/train/**
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/packets/M11/P-M11-R1-backend-followups.md
docs/packets/M11/P-M11-R1-backend-followups.ko.md
```

## oracle

1. `cargo test -p es-physics-backend backend_identity` — changing only the script bytes changes the
   identity; `mujoco-cpu`'s slot is still all zeros; every committed `mujoco-cpu` lock unmoved.
2. `cargo test -p es --test cli train_rl_vision_dry_run_plan` — a plan golden (addition) for
   `tests/fixtures/rl/training-reach-vision.toml`; and with `ES_PYTHON` + a GPU, a 3-iteration
   `es train` of it completes and packs a checkpoint.
3. Server (`ES_ISAAC_PYTHON`, GPU lock): a 5-iteration `Rollout` PPO smoke with `[rl] backend =
   "physx"` on the state reach recipe completes.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

Oracles 1–4; `evaluation-execution.md` §2.8 (+ko) states the script rule.

## forbidden

Moving a `mujoco-cpu` hash; a new trait; changing the X3 frame bytes.
