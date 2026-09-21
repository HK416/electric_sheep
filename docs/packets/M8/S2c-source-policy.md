# M8 S2c — the source policy: brax PPO on our SO-101 scene, pinned so it can be imported

Spec: §28.11 wave 1 and its "owner decision 2" (brax PPO, MuJoCo Playground 0.2.0 pins, Isaac
second), §14.4, §2.5 (external tools are pinned by version and digest), §1.7 (an unrecognized
upstream fact is a warning, never a guess). Precedent: `docs/api-notes/mujoco-playground-quadruped.md`
(the brax parameter format, section 2 "Export to numpy" — read it first). The task is defined
once, in `docs/design/rl-continuation.md` section 5; this packet implements the **source side**
of it. New api-note: `docs/api-notes/brax-ppo-so101.md` (+ `.ko.md`). Type D: the numbers are
observations.

## the question

No pretrained SO-101 PPO policy exists upstream (Playground ships Go1, Panda, Aloha, Leap …).
**Can a reach policy be trained on our own `so101_pick_place.xml` with the upstream stack (MJX +
brax PPO), on the 4090, from a recipe that a second person can re-run — and exported in a form
S2b can import without ever opening orbax or pickle in Rust?**

## spec

* **Environment** `python/es/rl_source/so101_reach_env.py`: an `mjx_env.MjxEnv` over the committed
  scene, control 50 Hz (`ctrl_dt = 0.02`, `sim_dt = 0.005`, `n_substeps = 4`), observation /
  action / reward / success / timeout exactly as `rl-continuation.md` section 5 (gripper point =
  site `gripperframe`; cube pose randomized per episode the way the committed Task IR's
  `Randomization` node does — read `tests/fixtures/visible-learning/task.toml`'s header and
  reproduce the same distribution; say in the api-note if it cannot be identical). Actions are
  normalized `[-1, 1]` → `ctrlrange` (centre + half-range × a).
* **MJX compatibility, measured not assumed.** The scene declares `integrator="implicitfast"`,
  `cone="elliptic"`, `condim="3"`/`"6"`, `frictionloss`, `armature`, `<position>` actuators. Try the
  scene unmodified first; if MJX refuses or diverges, write a derived
  `python/es/rl_source/so101_reach_mjx.xml` with the **minimum** edits and a table of every edit
  in the api-note — each edit is a sim-to-sim gap S4c will measure.
* **Training** `python/es/rl_source/train_brax_so101.py --seed S --timesteps N --out <dir>`:
  `brax.training.agents.ppo.train` with a config in the style of Playground's manipulation params
  (networks `(256, 256)` or whatever converges — record it), `normalize_observations = True`.
  Writes: the orbax checkpoint, `source.npz` (framework-neutral: `obs_mean`, `obs_std`, hidden
  `kernel_i`/`bias_i` as stored, the output Dense split into `mean_kernel`/`mean_bias` and
  `logstd_kernel`/`logstd_bias`), `meta.json` (`framework`, versions, `obs_dim`, `action_dim`,
  `hidden`, `activation = "swish"`, `activate_output = true`, `squash = "tanh"`, obs layout, action
  `scale`/`offset`, joint order from the MJCF, blake3 of the scene used, the config), a reward
  curve JSON, and **`oracle-1000.npz`**: 1,000 obs vectors drawn `U(−1, 1)` per channel with seed
  0 and the deterministic actions `tanh(loc)` JAX computed for them (f32). That file is S2b's
  oracle input.
* **Evaluation in the source framework** `eval_brax_so101.py`: success rate over 64 episodes
  with the deterministic policy — the number S4c puts in the "source framework" column.
* **Venv** `~/venvs/es-rl` on the server (uv, Python 3.12): `jax[cuda12]`, `mujoco`, `mujoco-mjx`,
  `brax==0.14.2`, `playground==0.2.0`, `orbax-checkpoint`, `numpy`. Every version in the api-note.
  Share the GPU politely: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=.5`;
  another job (U4 evaluation, `~/artifacts/plan-v/m7-r5/`) may be running — never kill anything.
* **Determinism** is an observation: seed 0 twice with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
  — record whether `source.npz` is bitwise (blake3 of both). If not, say so and keep both hashes.
* **Artifacts** under `~/artifacts/plan-s/s2c/` (checkpoint, `source.npz`, `meta.json`,
  `oracle-1000.npz`, logs, curve); the api-note names the paths and the blake3 of `source.npz`.

## context

```
python/es/rl_source/**
docs/api-notes/brax-ppo-so101.md
docs/api-notes/brax-ppo-so101.ko.md
docs/packets/M8/S2c-source-policy.md
docs/packets/M8/S2c-source-policy.ko.md
```

Learning-path Python only. No Rust file changes. Nothing in `python/es/` outside `rl_source/`.

## oracle

1. Server: `train_brax_so101.py --seed 0 --timesteps <N>` finishes; `eval_brax_so101.py` reports
   success rate ≥ 0.8 over 64 episodes (the target; if not reached, the row says what was
   reached and the run still ships — S4c needs *a* policy, and a weak one is more informative for
   continuation than none).
2. `source.npz` / `meta.json` / `oracle-1000.npz` written; a tiny reader
   `python/es/rl_source/check_export.py` re-computes the 1,000 actions from `source.npz` in numpy
   (swish, tanh) and matches JAX's to ≤ 1e-5 (numpy vs XLA; record the max abs error).
3. Two seeds-0 runs: `source.npz` blake3 recorded, bitwise or not.
4. The api-note exists with pins, the env definition, the MJX findings table, the recipe, the
   numbers (each `measured (server, date, path)`), and the export format S2b will read.

## acceptance

Oracles 1–4; `cargo xtask ci` unaffected (no Rust), `cargo xtask check-scope` on this packet.

## forbidden

Any Rust change; opening the checkpoint anywhere but this Python; changing the committed scene
(`tests/fixtures/mjcf/**`) — a derived XML lives under `rl_source/`; a task definition that
differs from `rl-continuation.md` section 5 without a row in the api-note saying how and why.
