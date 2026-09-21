# M8 S4c — the measurement: the imported policy before and after continuation, one table

Spec: §28.11 wave 3 (Type D), §13.3 (the evaluation is fixed; only data and policy move),
§10.5 (`es eval compare`), §28.9 rules 2–3, §12.4. Depends on **S2b** (the imported bundle),
**S1**, **S4b**. Design note: `docs/design/rl-continuation.md` section 7; a row in
`docs/design/visible-learning.md` (7.34, "the RL track") pointing there.

## the question

**Does continuing the imported policy by PPO in our simulation raise its success rate on the
same Evaluation IR — and by how much against training from scratch for the same budget?**

## spec

One Evaluation IR (`tests/fixtures/rl/evaluation-reach.toml`: 16 held-out seeds, nominal + the
demo's perturbation suites that apply to a state policy), one scene, one Deployment IR (the
declared latency applies — `rl-continuation.md` section 3). Rows, each with three training seeds
where training is involved (mean and min / max, §28.9 "confidence intervals"):

| row | policy | what it shows |
|---|---|---|
| source | brax's own evaluation of the S2c policy (`eval_brax_so101.py`) | the source framework's number |
| imported | the S2b bundle, no training | the honest sim-to-sim number |
| continued | `[init] = imported`, PPO for budget B | the campaign's claim |
| from scratch | PPO for budget B, random init | the control |
| expert | the scripted expert, if one exists for reach; else omitted and said so | the harness check (§28.9 rule 1) |

Budget B is set from S4b's measured convergence (owner decision 3 in §28.11). Every run's
`policy_hash`, `training_hash`, `evaluation_hash` and `execution_hash` in the table; `es eval
compare imported.json continued.json` output beside it.

## context

```
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
tests/fixtures/rl/**
docs/packets/M8/S4c-continuation-measurement.md
docs/packets/M8/S4c-continuation-measurement.ko.md
```

## oracle

The table exists with every cell either a measured number (server, date, path under
`~/artifacts/plan-s/s4c/`) or `Target / Status: unverified` with the reason; the hashes chain
(`evaluation_hash` identical across all rows); `cargo xtask check-scope`.

## acceptance

The table, the compare output, and one paragraph of what it says — including if continuation
did not help.

## forbidden

Changing the Evaluation IR between rows (§13.3); quoting a number without its hashes; any
training-side change (S4b owns the trainer).
