# M9 R5 — the trainer learns from the action the plane executed (owner decision B, 2026-09-22)

Spec: §13.4 (rollouts through `es-env`, the plane on; today: "the policy's sample is the action,
the plane's output is what the actuator gets, both are recorded" — this packet changes what the
*gradient* is computed from, not what is recorded), §9.4, INV-11..13 (the envelope and the plane
do not change), §28.9 rule 3 (one variable), §12.4. Review: `docs/reviews/M9.md` S-7 and the
human decision "§13.4's default action space" — the owner chose **B**: the envelope stays as
declared; the trainer uses the executed action. Design note: `docs/design/rl-continuation.md`
section 2 (the sampling model) gains the rule; section 7 the rows.

## the question

Every rollout tick of every RL run so far was clamped (`executed_ne_sampled_rate = 1.00`), so
PPO's gradient was computed from the log-probability of an action the env never executed, and
neither smaller noise (P-M8-R1) nor the increment space (T3) changed that. **If the trainer treats
the plane as part of the environment — the action in the buffer is the one the plane executed,
and the log-probability is evaluated at that action — does the reach policy learn past its
4,000-iteration peak, with the envelope untouched?**

## spec

* `train_ppo.py --estimator executed` (default stays `sampled`, so every committed recipe and
  its `training_hash` are unmoved; `[rl] estimator = "executed"` in the recipe, `deny_unknown_fields`,
  recorded in `training/config.json` and therefore in `training_hash`): in the rollout, store
  `executed` (from `Rollout.act`) as the action of the step, and compute `logp` under the current
  Gaussian **at the executed action** (both the stored old log-probability and the ratio in the
  update); rewards, dones, values, GAE unchanged. `executed_ne_sampled_rate` keeps reporting the
  plane's clamp rate — it is a fact about the envelope, not about the estimator.
* Nothing in `es-env`, `es-py`, `es-safety` or the Deployment IR changes; the plane's envelope in
  `deployment-reach.toml` is the one every earlier row used.
* Recipes: `tests/fixtures/rl/training-reach-executed.toml` = `training-reach.toml` + the one
  line; `--dry-run` plan golden as an addition.

## context

```
python/es/train_ppo.py
crates/es-data/src/training.rs
crates/es-data/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/training-reach-executed.toml
tests/golden/train/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M9/P-M9-R5.md
docs/packets/M9/P-M9-R5.ko.md
```

## oracle

1. `cargo test -p es --test cli train_rl_estimator_dry_run_plan` — the plan golden (an addition);
   `estimator = "sampled"` spelled out hashes the same as absent; an unknown value is refused by
   name; `training-reach.toml`'s `training_hash` is unmoved.
2. `cargo test -p es --test cli train_rl_two_runs_are_bitwise` (`ES_PYTHON`) also with
   `estimator = "executed"`: two runs bitwise.
3. Server: `training-reach-executed.toml`, seeds 0/1/2, 4,000 and 10,000 iterations, evaluated
   on `evaluation-reach.toml`; beside T3's absolute A0 row (0.4167 at 4,000; 0.3125 seed 0 at
   10,000) in one table — `success_rate`, `episode_length`, `executed_ne_sampled_rate`, rollout
   entropy, `return`, wall-clock, the nine metrics. The sentence the next decision needs: does
   the curve hold past 4,000 and does it beat A0 at 10,000 on three seeds?
4. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M9/P-M9-R5.md`.

## acceptance

Oracles 1–4; section 2's rule and section 7's rows (+ `.ko.md`). If B does not help, the rows say
so and the next decision (A with a servo-spec reason, or C) is the owner's.

## forbidden

Any change to the envelope, `es-safety`, `es-env`'s integrator or `Rollout` (INV-11..13; one
variable); making `executed` the default (a recipe-visible choice, hashed); pickle; a new trait.
