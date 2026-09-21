# M8 S4b — the PPO trainer: `es train --recipe` with `[rl]`, rollouts through `es_native.Rollout`

Spec: §13.4 (all of it), §19.3, §12.4, §3.5 tier 1 (CPU backend bitwise), §28.10 rule 2,
§28.11 wave 3, INV-12 (the plane is on), INV-16, INV-17. Depends on **S4a** (the binding), **S1**
(`[init]`), **S2a** (squash lowering). Design note: `docs/design/rl-continuation.md` sections 2, 3,
5, 6 are the decisions; this packet fills section 7. `training-recipe.md` gains "The RL route".

## the question

**Can one recipe run PPO — rollouts in our `Env` with the Safety Plane on, GAE, the clipped
objective, entropy — from scratch or from `[init]`, write §19.3's `training/` with real values,
and be bitwise reproducible on the CPU backend?**

## spec

* **Recipe.** `[rl] algo = "ppo"`, `envs`, `horizon` (steps per env per iteration), `epochs`,
  `minibatches`, `gamma`, `lam`, `clip`, `entropy`, `value_coef`, `init_log_std` (absent → the
  importer's `log_std` if `[init]` carries one, else `−0.5`). With `[rl]`: `[run] steps` = PPO
  iterations; `[run] batch` refused by name; `[dataset]` optional and `dataset.lock` reads
  `unset`; `[policy] bundle` names the graph (a fresh `es policy pack` of the documents or an
  imported bundle); `task` / `observation` / `deployment` come from the bundle; `[run] seed`,
  `device`, `lr`, `grad_clip`, `checkpoint_at`, `schedule` keep their meaning.
* **Route** `rl`: `es policy lower` → `python/es/train_ppo.py --module … --rollout-docs … --scene …
  [--init-weights …] --envs … --out …` → `es policy pack` per checkpoint (as the IR route does).
  The trainer builds the actor from the lowered module (deterministic action in actuator units),
  a state-independent `log_std`, and a value MLP over the concatenated Observation IR ports;
  samples `a = mu + exp(log_std)·eps`; steps `Rollout.act`; stores `obs`, sampled `a`, executed
  action, event bits, reward, done, value, log-prob; GAE; clipped objective + value loss +
  entropy; Adam with `[run] lr`, `grad_clip`; `torch.use_deterministic_algorithms(True)` and
  seeded RNGs on `device = cpu`.
* **What is recorded.** `metrics/loss-curve.json` per iteration: `loss`, `policy_loss`,
  `value_loss`, `entropy`, `return`, `episode_len`, `envelope_violation_rate` (from the event
  bits), `executed_ne_sampled_rate`, `samples_per_sec`; stream 5 `[step, loss, lr, samples/s]`
  so the editor's Live tab draws it (E7); `training/config.json` carries `[rl]`;
  `training/value.safetensors` for resumption, never packed.
* **Determinism** on the CPU backend: two runs of the same recipe → every checkpoint bitwise and
  the same `training_hash`. GPU backends are a later throughput packet (§28.11 "not on the
  ladder").
* **The oracle task** `tests/fixtures/rl/task-reach.toml` + `observation-reach.toml` +
  `deployment-reach.toml` + `evaluation-reach.toml` (`rl-continuation.md` section 5; the
  deployment envelope wide enough that a random policy is clamped, not latched — INV-12: widen,
  never disable) and `training-reach.toml` (`[rl]`, small: `envs = 8`, `horizon = 64`).

## context

```
crates/es-data/src/training.rs
crates/es-data/tests/**
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
python/es/train_ppo.py
python/es/README.md
python/es/README.ko.md
tests/fixtures/rl/**
tests/golden/train/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M8/S4b-ppo-trainer.md
docs/packets/M8/S4b-ppo-trainer.ko.md
```

## oracle

1. `cargo test -p es --test cli train_rl_dry_run_plan` — the plan golden (an addition) for
   `training-reach.toml`; `[run] batch` with `[rl]` refused by name; `[dataset]` absent accepted.
2. `cargo test -p es --test cli train_rl_two_runs_are_bitwise` (`ES_PYTHON`; SKIP with reason):
   `envs = 4`, `horizon = 16`, 3 iterations, seed 0, twice → checkpoints bitwise, `training_hash`
   equal, `dataset.lock` reads `unset`, `loss-curve.json` has the listed fields.
3. `cargo test -p es --test cli train_rl_init_from_import` (`ES_PYTHON`): `[init] policy =` the
   synthetic playground fixture bundle from S2b → iteration 0's actor equals the import bitwise
   before the first update (`0.esb`).
4. Server: `training-reach.toml` with a real budget (`envs = 16`, `horizon = 64`, N iterations)
   → `es eval run --config evaluation-reach.toml` success rate ≥ 0.8; N, wall-clock and the nine
   metrics in section 7 (`Target / Status: unverified` for the ones the run does not measure).
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M8/S4b-ppo-trainer.md`.

## acceptance

Oracles 1–5. Section 7 rows with dates and paths; `training-recipe.md` "The RL route".

## forbidden

A simulator other than `es-env` (rule 2); any plane state skipped (INV-12); the value head or
`log_std` in a document or a bundle (rule 1); a Task IR change for RL (rule 6); pickle (INV-16);
a new trait (INV-17); `docs/ARCHITECTURE*.md`; `tests/golden/**` modifications.
