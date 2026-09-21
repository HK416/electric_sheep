# M9 T3 — reach by PPO in the increment space, beside the best absolute-space run

Spec: §13.4, §28.9 rule 3 (one variable), §28.12 wave 2, §12.4. Depends on **T1**, **T2** (the
delta deployment and a delta bundle to start from) and **P-M8-R1** (the best absolute-space
recipe, the control row). Type D with one small Rust/Python surface: `train_ppo.py` needs
nothing new if `Rollout` integrates (T1); the recipe names the delta bundle.

## the question

**With the same budget, seeds and Evaluation IR, does PPO in the increment space clamp less and
score higher than the best absolute-space recipe R1 found — and does continuation from T2's
imported delta policy do anything, where S4c's absolute one did nothing?**

## spec

Rows, three seeds each, 4,000 and 10,000 iterations, `evaluation-reach.toml` (the declared
latency applies through `es eval run`):

| row | bundle | recipe |
|---|---|---|
| absolute, best of R1 | `learning-reach.toml` graph, `deployment-reach.toml` | R1's best (quoted where already measured) |
| delta, from scratch | same graph, `deployment-reach-delta.toml` | R1's best `[rl]` values |
| delta, `[init]` = T2's import | T2's imported bundle | same |

Per row: held-out `success_rate` (mean; min / max), `episode_length`,
`executed_ne_sampled_rate`, `envelope_violation_rate`, rollout entropy, the first-iteration
actor gradient norm (S-13's detector, if S4f/R7 has landed; else `unverified`), wall-clock,
the nine metrics. `es eval compare` between the absolute and delta rows at seed 0.

## context

```
tests/fixtures/rl/training-reach-delta.toml
tests/fixtures/rl/training-reach-delta-continued.toml
tests/golden/train/**
crates/es/tests/cli.rs
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M9/T3-delta-reach-rl.md
docs/packets/M9/T3-delta-reach-rl.ko.md
```

## oracle

1. `cargo test -p es --test cli train_rl_delta_dry_run_plan` — the plan golden (an addition).
2. Server: the table, every cell measured or `unverified` with a reason; artifacts under
   `~/artifacts/plan-t/t3/`.
3. `cargo xtask check-spec-refs`; `check-scope`.

## acceptance

The table in section 7 and a row 7.35 in `visible-learning.md`, with the sentence the M9 review
needs: whether §13.4's default action space for RL should become the increment.

## forbidden

Changing the Evaluation IR, the envelope, or the trainer between rows; more than one variable
between the absolute and delta rows (same `[rl]` values, same graph, same seeds).
