# M7 T4 — a learning-rate schedule, and the batch that stopped diverging

Spec: §19.3 (`scheduler.json`, `optimizer.json`, `precision.json` are identity slots), §8.1 (the
optimizer is not in the IR — it is the trainer's, and the trainer's identity is §19.3), §28.9
"what would remove each" ("large batches with warmup and an lr schedule: linear scaling was
measured to diverge, so what is needed is a schedule, not a convention"), §28.10 (T4). Design
note to edit: `docs/design/training-recipe.md` (+ `.ko.md`, T1's) — a "schedule" section.
Depends on **T1** (`training/scheduler.json` is written from the recipe) and **T3** (real
batches; a schedule over accumulated single samples answers a different question).

## the question

V5 measured `--batch 64 --lr 8e-4` (linear scaling from batch 8 / 1e-4): `final_loss` NaN after
1 h 24 min (design note 7.11). With T3 a batch of 64 is one forward, so the wall-clock argument
for large batches is real for the first time. **With warmup and a cosine decay recorded as
identity, does batch 64 train to a loss at or below batch 8's at equal samples seen?**

## spec

* `python/es/train_act.py` gains `--schedule constant|warmup_cosine` (default `constant` —
  the measured runs stay reproducible), `--warmup-steps N` (default `0`), `--lr-min F` (default
  `0.0`) and `--weight-decay F` (AdamW's, default torch's `1e-2` **made explicit**, so
  `optimizer.json` can name it), `--grad-clip F` (default `0` = off; a clipped run says so).
  The schedule is `lr(step) = lr * step / warmup` for `step < warmup`, then `lr_min + (lr -
  lr_min) * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))` — implemented as a plain
  function `lr_at(step, total, lr, lr_min, warmup) -> float` in the script and called every step
  (`torch.optim.lr_scheduler` is not used: its float sequence is an implementation detail of a
  torch version, and the golden below pins ours).
* The JSON summary reports `schedule`, `warmup_steps`, `lr_min`, `weight_decay`, `grad_clip`
  and `lr_curve_hash` (blake3 over the `f64` lr values actually applied, little-endian).
* `es train` (T1) passes the recipe's `[run] schedule = { kind = "warmup_cosine", warmup = 500,
  lr_min = 1e-6 }` through, writes it as `training/scheduler.json`, and `optimizer.json` gains
  `weight_decay` and `grad_clip`. `identity_hash` moves with any of them.
* Server measurement (acceptance, on V15's baked set, 20,000 optimizer steps each unless noted):
  | run | batch | lr | schedule | steps |
  |---|---|---|---|---|
  | A (baseline, T3) | 8 | 1e-4 | constant | 20,000 |
  | B | 64 | 1e-4 | constant | 2,500 (equal samples) |
  | C | 64 | 4e-4 | warmup_cosine, warmup 250 | 2,500 |
  | D | 64 | 4e-4 | warmup_cosine, warmup 250 | 20,000 |
  Report `final_loss`, wall-clock, whether any step was non-finite. Keep D's checkpoint at
  `~/artifacts/plan-v/m7-t4/model-D.safetensors` for the U-measurement. **Do not evaluate** the
  checkpoints here.

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

```
python/es/train_act.py
crates/es-policy/tests/ir_training.rs
tests/golden/train/**
crates/es-data/src/training.rs
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/training.toml
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M7/T4-lr-schedule.md
docs/packets/M7/T4-lr-schedule.ko.md
```

`python/es/train_act.py`, `crates/es-policy/tests/ir_training.rs` (schedule tests),
`tests/golden/train/lr_warmup_cosine.json` (new: the first 1,000 values of `lr_at` for `total
= 20000, lr = 4e-4, lr_min = 1e-6, warmup = 250`, generated once by a `#[ignore]`d test that
calls the script's function through `ES_PYTHON`; the checked-in test compares the Python
output to the golden **and** to a Rust re-implementation of the same formula in the test file,
which is what pins it without Python), `crates/es-data/src/training.rs` and
`crates/es/src/cmd/train.rs` (the recipe fields and the two JSON files), `crates/es/tests/cli.rs`
(`train_*` tests only), `tests/fixtures/visible-learning/training.toml` (if the fixture recipe
should show the field — say so), `docs/design/training-recipe*.md`, `docs/packets/M7/T4-lr-schedule*.md`.

## oracle

1. `cargo test -p es-policy --test ir_training lr_schedule_matches_the_golden` — the Rust
   re-implementation equals the golden bitwise (f64); with `ES_PYTHON`, the script's `lr_at`
   equals it bitwise too (else `SKIP` with reason).
2. `cargo test -p es-policy --test ir_training -- --ignored the_default_schedule_is_the_old_run`
   — 40 steps at `--schedule constant` produce the loss curve T3's oracle 4 recorded (bitwise,
   same seed, same device: CPU), so the measured runs are unmoved.
3. `cargo test -p es --test cli train_identity_moves_with_the_schedule` — changing `warmup`
   changes `identity_hash`; `scheduler.json` and `optimizer.json` carry the fields.
4. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T4-lr-schedule.md`.

## acceptance

Oracles 1–4; the four-row table on the server in the design note, each row's `training.lock`
kept under `~/artifacts/plan-v/m7-t4/`. A NaN in any row is reported as the finding it is, with
the step at which it appeared.

## forbidden

Changing the default schedule, lr, batch or optimizer (the measured runs must reproduce);
`torch.optim.lr_scheduler`; `crates/es-policy/src/**` (T3/T5 own the lowering);
`crates/es-ir/**`; evaluating any checkpoint; `docs/ARCHITECTURE*.md`; goldens other than the new
one. INV-16: safetensors only.
