# M11 X3 — `Rollout` renders: RL observes images, with the per-tick seed per env

Spec: §28.14 rules 1, 4, 5 and wave 2, §4.3 (Python is a first-class dependency on the learning
path; the renderer is Rust), §13.4 (rollouts through `es-env`, the plane on), §15, §3.4. Design
notes: `renderer.md` 12.3 / 12.8 (the tick seed and its three callers), `rl-continuation.md`
section 2 (+ko). Depends on X6 (`render.svgf`) being merged, and on X1 (the backend enum in
`Rollout`) — `git merge --ff-only main` first. Type B.

## the question

`es_native.Rollout::observe` refuses an image input because es-py links no renderer
(`crates/es-py/src/rollout.rs:235`), and es-py builds `es-env` without its `render` feature. PPO
already executes the lowered Learning IR, so a `VisionBackbone` encoder works once images arrive.
**With an es-py `render` feature, does `Rollout` render each env's sensor — `Rs` or `Pt`, with
`seed = "tick"` restarting at each env's own reset and `svgf` as declared — so that a `Rollout`
frame is bit-identical to the collector's frame at the same `(episode, tick)`?**

## spec

* `es-py` gains a `render` feature (`es-env/render`); the maturin build used by `train_ppo.py`
  enables it. Without the feature, an image input is still refused by name (today's behaviour).
* `Rollout` owns one `EnvRenderer` per env built through the same `es_env::render::sensor_cfg` the
  collector uses; `observe` captures image ports through the same `capture` path as
  `es_eval::runner` (no second implementation of the Observation IR); each env's reset calls that
  env's `EnvRenderer::begin_episode`, so the tick seed is episode-relative per env.
* The per-frame render cost is measured and returned in `Rollout.metrics()` (render ms/frame, the
  nine-metric set's render row; no single `step/s` figure).
* `train_ppo.py` stacks image ports as `[n_envs, C, H, W]` float tensors in the lowered module's
  expected layout (read it from the contract, never assume).

## context

```
crates/es-py/Cargo.toml
crates/es-py/src/rollout.rs
crates/es-py/src/lib.rs
crates/es-py/tests/**
crates/es-env/src/render.rs
crates/es-env/tests/render_loop.rs
python/es/train_ppo.py
python/es/pyproject.toml
tests/fixtures/rl/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/X3-rollout-render.md
docs/packets/M11/X3-rollout-render.ko.md
```

## oracle

1. `cargo test -p es-py --features render rollout_frame_equals_collector_frame` (GPU) — a
   camera-bearing reach task (`tests/fixtures/rl/task-reach-vision.toml`, generated like the other
   reach documents), `Pt` 16 spp, `seed = "tick"`: for 2 envs × 2 episodes, `Rollout`'s frame at
   `(episode, tick)` == `es loop collect --frames` at the same `(episode, tick)` bitwise; without the
   feature the image input is refused by name.
2. `cargo test -p es-py rollout_state_only_is_unchanged` — the state-only reach rollouts and
   `train_rl_two_runs_are_bitwise` unchanged.
3. `ES_PYTHON=… python/es/train_ppo.py` 10-iteration smoke on the vision reach task with a ResNet18
   `VisionBackbone` graph: runs, finite loss, render ms/frame reported.
4. fmt, clippy `-D warnings` (with and without `render`), check-scope, verify-goldens.

## acceptance

Oracles 1–4 on both GPUs; the render cost table (Rs, Pt 4/16/64 spp, SVGF on/off) in
`rl-continuation.md` (+ko).

## forbidden

A second observation implementation in es-py; temporal accumulation; changing `EnvRenderer`'s
frame bytes; batching across envs (X3b's).
