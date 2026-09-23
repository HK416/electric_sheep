# M11 U6 — the demo's path-traced row with SVGF, two training seeds

Spec: §28.14 rules 5, 7 and wave 3; `visible-learning.md` 7.37 (U5: held-out 0.25, training seeds
0.50, one CUDA training); M10 review S-2 (the cycle recipe is under-trained for today's 103,881-frame
collect) and S-4 (single CUDA runs carry ±0.3). Depends on X6. Type D.

## the question

**With `svgf = true` on top of U5's `seed = "tick"`, does the demo's path-traced ACT row move, and
by more than the spread of two training seeds?**

## spec

* Documents: `task-pt-tick-svgf.toml` and its observation/evaluation siblings (X6), plus the
  augmented observation and evaluation derived the way W1a derived U5's.
* Run W1a's pipeline under `task-pt-tick-svgf`: collect 200 (seed 1, `--expert`) → train with
  `training-u5.toml`'s settings (one variable: the document) at **two training seeds** → held-out
  and training-seed evaluation. Also run U5 again at its second training seed, so U5 has two seeds.
  Server, GPU lock, nohup + markers under `~/artifacts/plan-x/u6/`.
* Parity gate first, as W1a did: collector and evaluator tick-0..2 frames md5-identical.

## context

```
tests/fixtures/visible-learning/**
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M11/U6-demo-svgf.md
docs/packets/M11/U6-demo-svgf.ko.md
```

## oracle

1. The parity gate on the server.
2. The table: U5 seed 0 (committed), U5 seed 1, U6 seed 0, U6 seed 1 — held-out `success_rate` per
   suite, training-seed success, final loss, the hashes.
3. check-scope, verify-goldens; raw frame trees deleted after reading.

## acceptance

`visible-learning.md` 7.38 (+ko) with the table and one sentence: does SVGF move the row by more
than the two-seed spread.

## forbidden

Changing the recipe (step count included: S-2 is the owner's decision, not this packet's); code
changes.
