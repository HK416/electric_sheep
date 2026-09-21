# M7 U — the measurement: the IR graph after T3–T6, on the committed documents

Spec: §28.10 ("one sentence on accuracy": T3·T4·T5 change the lowering, so the IR-graph policy is
re-measured **once** on the committed documents; the stop rule — if held-out stays at or below
V18b's 0.625 after T5, IR-graph tuning ends and the product is the external route's speed), §28.9
rules 2–3 (an invalidated measurement is marked, never erased; no performance claim without a
reproducible metric), §12.4 (nine metrics, never `step/s`), §13.3 (a moved `evaluation_hash` is
a new comparison, said out loud), §10.5. Type D. Design note to extend:
`docs/design/visible-learning.md` (+ `.ko.md`) — section 7.31 "as built (M7/U)", beside V18b (7.28)
and V19b (7.29/7.30). Depends on **T2, T3, T4, T5, T6, T7** for the accuracy half and **R2, R3,
R4** for the showcase half; each half runs when its inputs have landed and says which commit it ran from.

## the question

Every T-packet reported a loss. **Does any of it move `success_rate` on the documents the
project committed — and by the §28.10 stop rule, does IR-graph work continue after M7?**

## spec

* **The documents.** Task/Deployment/Evaluation IR: the committed `tests/fixtures/visible-learning/`
  set (`evaluation.toml` = 16 episodes on the held-out seeds 101–116 + the six suites; the training
  seeds' document is `~/artifacts/plan-v/v15/eval-trainseeds.toml` on the server). Data: V15's
  200 demonstrations (`~/artifacts/plan-v/v15/ds-train`, baked at `v15/baked` — re-bake with
  `--for-training` for U2/U3). Every run: 20,000 optimizer steps at T4's row-D settings (batch 64,
  lr 4e-4, `warmup_cosine` warmup 250, `lr_min` 1e-6, seed 0, `--resident-gpu`), through `es train`
  with a recipe, so `training.lock` exists for each.

  | run | Learning IR | Observation IR | what it measures |
  |---|---|---|---|
  | U0 | committed `learning.toml` | committed `observation.toml` | T3 + T4 alone — **T4's row-D checkpoint, already trained** (`~/artifacts/plan-v/m7-t4/model-D.safetensors`, `lr_curve_hash c01d5185…`): pack and evaluate, do not retrain |
  | U1 | `learning-pretrained.toml` (T5) | committed | + the ImageNet backbone, `frozen = false` unless T5's note says otherwise |
  | U2 | committed | `observation-augmented.toml` (T6) | + random shift and brightness/contrast |
  | U3 | `learning-pretrained.toml` | `observation-augmented.toml` | both |

  For each: `success_rate` on the held-out 16, on the training seeds 16, and the six-suite sweep
  from the committed `evaluation.toml` with `--jobs 6`, all under T7's latency model (the bundle's
  `expected_latency_ms` as `es policy pack` writes it — say what it is; open question 32). `report.json`,
  `evaluation.lock`, `training.lock` and the checkpoint under `~/artifacts/plan-v/m7-u/U<n>/`.
* **U2/U3 are a new comparison.** Their Observation IR is a different document, so the Evaluation
  IR that names it is a different `evaluation_hash` (§13.3). The table prints both hashes and the
  note says in one sentence that U2/U3 rows are comparable to U0/U1 only by the reader's judgement,
  not by the chain — the perturbation suites, seeds and metrics are the same bytes; the observation
  document is not.
* **The stop rule, applied.** The note's first paragraph states the U1 held-out number against
  0.625 and the consequence §28.10 fixes. It is written after the measurement and not before.
* **The cycle's wall-clock.** T2's acceptance cycle (collect 200 → expert gate → train → eval →
  showcase) is cited, not re-run, unless T2's number was taken before T4/T5 landed — then one
  re-run at the U3 configuration, per-stage times beside T2's.
* **The showcase half.** From U0's `nominal-00` (or V19b's if U0 fails every episode — say which):
  `es video showcase` frames at 1280×720 with R2's `--look full`, R3's `--path pt --spp 64`, and R4's
  `--path pt --spp 4 --accumulate` if R4 has landed (else the row reads `not run: R4 not landed at
  <commit>`), each encoded to H.264 by `python/es/encode_video.py`, ms/frame beside the R-packets'
  own tables, the mp4s under `~/artifacts/plan-v/m7-u/showcase/` and copied to the requester's
  `target/plan-u/m7-u/`.
* **Nothing is tuned here.** A row that diverges, times out or fails acceptance is a row with a
  number in it. A second seed, a longer run or a changed lr is a new packet.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
tests/fixtures/visible-learning/training-u*.toml
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/U-measurement.md
docs/packets/M7/U-measurement.ko.md
```

Four recipes `training-u0.toml` … `training-u3.toml` (committed, so the runs are reproducible
from the repository — U0's names the row-D settings but its checkpoint is T4's), the design note
section, this packet. **No source change.** A defect found on the way is reported, not fixed here.

## oracle

1. Every row of the table has a `report.json` whose `evaluation_hash` matches the printed one and a
   `training.lock` whose `identity_hash` matches the recipe's `--dry-run` identity on the requester's
   machine (`es train --dry-run` is judged locally without Python).
2. `es eval compare U0/report.json U1/report.json` (and U0 vs U2, U1 vs U3) printed into the note —
   the tool's own significance column, not a reading by eye.
3. `cargo xtask ci` (no source moved; the four recipes' `train_dry_run` plans are not goldens, but the
   recipes parse); `cargo xtask check-scope docs/packets/M7/U-measurement.md`.

## acceptance

The 7.31 table (four rows × three columns + the sweep's six), the stop-rule paragraph, the cycle
wall-clock, the showcase ms/frame table and the three mp4s. Each number names the commit and the
tree it ran from.

## forbidden

Any source change; retraining U0; a fifth configuration; evaluating on seeds other than the two
committed sets; `docs/ARCHITECTURE*.md` (the review moves the spec, not this packet); goldens.
