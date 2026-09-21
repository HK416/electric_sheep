# M8 R1 — exploration noise: what the clamp rate and the reach curve do when only the recipe moves

Spec: §13.4 (rollouts through `es-env`, the plane on; the sampled ≠ executed rate is reported),
§28.9 rule 3 (one variable at a time), §28.12 wave 0, §12.4. Review: `docs/reviews/M8.md` S-3 and
the human decision "the plane in RL rollouts" — the owner chose (2026-09-22) to measure the
trainer's noise before deciding between the estimator, the envelope and delta actions. Type D:
no code. Design note: `docs/design/rl-continuation.md` section 7 gains the table.

## the question

S4b and S4e measured `executed_ne_sampled_rate = 1.00` on every tick of every run: with
`init_log_std = −0.5` the Gaussian's σ ≈ 0.6 rad per 50 Hz tick is a 30 rad/s commanded velocity
and the plane clamps it by construction; the entropy bonus then pays for σ to grow (4.87 → 6.06)
and the reach run decays after its 4,000-iteration peak. **With the plane and the envelope
untouched, how far does the clamp rate fall and does the curve hold when only `init_log_std`,
the entropy coefficient and the learning-rate schedule move?**

## spec

Recipes derived from `tests/fixtures/rl/training-reach.toml` (committed under
`tests/fixtures/rl/noise/`, header comments saying what each changes), seed 0, `envs = 16`,
`horizon = 64`, evaluated with `es eval run --config tests/fixtures/rl/evaluation-reach.toml`
at 4,000 and 10,000 iterations (checkpoints at both):

| id | `[rl] init_log_std` | `[rl] entropy` | `[run] schedule` |
|---|---|---|---|
| A0 (S4e, quoted) | −0.5 | 0.005 | constant |
| A1 | −2.5 | 0.005 | constant |
| A2 | −2.5 | 0.0 | constant |
| A3 | −2.5 | 0.0 | `warmup_cosine` (warmup 100, `lr_min` = lr/100) |
| A4 | −1.5 | 0.0 | as A3 |

The best of A1–A4 by held-out `success_rate` at 10,000 iterations is re-run with seed 1. Recorded
per row: `executed_ne_sampled_rate` and `envelope_violation_rate` at iterations 1, 4,000,
10,000 (from `metrics/loss-curve.json`); rollout entropy at the same points; held-out
`success_rate` and `episode_length` at 4,000 and 10,000; wall-clock; the nine §12.4 metrics as
`Rollout.metrics()` reports them and `Target / Status: unverified` for the rest. One table,
one paragraph saying which variable moved which number.

## context

```
tests/fixtures/rl/noise/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M8/P-M8-R1.md
docs/packets/M8/P-M8-R1.ko.md
```

## oracle

1. Server: the runs above (`~/artifacts/plan-t/r1/<id>-seed<n>/`), the table with server, date and
   path; `cargo test -p es --test cli train_rl_dry_run_plan` still green (no golden moves — the
   noise recipes are not added to the plan golden).
2. `cargo xtask check-spec-refs`; `cargo xtask check-scope docs/packets/M8/P-M8-R1.md`.

## acceptance

The table in section 7 with every cell measured or `unverified` with a reason, and one sentence
per variable. If no variant beats A0's 0.5625 or holds past 4,000, the table says so — the
measurement is the deliverable.

## forbidden

Any code change (`train_ppo.py`, `es-env`, `es-safety`, the envelope in `deployment-reach.toml`);
changing the Evaluation IR; more than one variable per row beyond the table above.
