# Packet M5/V18b — the full Evaluation IR sweep and the showcase videos at the V18 levels

Design note: `docs/design/visible-learning.md` section 7.28. Predecessors: V18 (the
`acceleration_max` ablation, section 7.26) and V9 (the showcase pipeline, section 7.17).

## Context

- Allowed scope: no repository file. Server tree `~/Projects/es-v18b`, artifacts
  `~/artifacts/plan-v/v18b/`, small results mirrored to `target/plan-v/v18b/` (untracked).
- Inputs: V18's packed bundles `trained-L40.esb` and `trained-L80.esb` (V15's
  `model-40000.safetensors`, byte for byte), the checked-in
  `tests/fixtures/visible-learning/evaluation.toml`, V15's `eval-holdout.toml`.

## Spec

Run the Evaluation IR as checked in — every suite it declares, its own seed plan (16 held-out
episodes, seeds 101–116), budget 1800 — at `acceleration_max` 40 and 80, report `passed` and the
per-suite metrics; reproduce V18's held-out nominal numbers at both levels; at the level that
passes, render two held-out successes and a 16-episode mosaic with the V9 pipeline.

## Oracle

- `es eval run --config tests/fixtures/visible-learning/evaluation.toml --policy trained-L<N>.esb
  --scene tests/fixtures/mjcf/so101_pick_place.xml --jobs 4 --out … --frames … --traj …` per
  level; `report.json` carries `passed`.
- The held-out nominal rerun reproduces V18's `report.json` (`evaluation_hash 40623e01…`, every
  metric and the `failure_mode_histogram`) bit for bit.
- `es ir check task.toml observation.toml learning.toml deployment-L80.toml` prints the same
  `deployment_hash` as the committed fixture after commit `74b01a3` (orchestrator's check:
  `f2f9a510…` for both; the level-40 copy is `38e56f4a…`).
- `es video showcase` / `es video mosaic` produce mp4s whose frame counts `ffprobe` confirms.

## Acceptance

`passed = true` at the adopted level; the held-out reproduction is bit-identical; the three
videos exist and a sampled frame sequence shows grasp, carry and release.

## Forbidden

Editing `docs/`, `tests/` or any fixture (the fixture decision is the owner's and was made in
parallel, commit `74b01a3`); retraining; disabling or bypassing the Safety Plane; promoting the
level-40 numbers to anything but the alternative not taken.

## Results (2026-09-16, RTX 4090, `ES_PYTHON=~/venvs/es-lerobot-cuda`)

| level | `passed` | nominal | light_intensity | light_direction | observation_delay | torque_noise | backlash |
|---|---|---|---|---|---|---|---|
| 40 | true | 0.625 | 0.3125 | 0.4375 | 0.25 | 0.125 | 0.5625 |
| **80 (adopted)** | **true** | **0.625** | 0.625 | 0.5 | 0.0625 | 0.0625 | 0.625 |

`evaluation_hash e52e8360…` at both levels (the deployment is not part of it);
`execution_hash c2322c2c…` (40), `52dfeba9…` (80). Held-out nominal reproduces V18 at both levels.
Videos: `v18b-L80-holdout-nominal-00.mp4` (seed 101, 206 ticks), `v18b-L80-holdout-nominal-12.mp4`
(seed 113, 183 ticks), both 1280×720 at 30 fps; `v18b-L80-holdout-mosaic.mp4` (4×4 tiles of the
96×96 observation, 1,800 frames at 50 fps). Server `~/artifacts/plan-v/v18b/showcase/`, local
`target/plan-v/v18b/showcase/`; full tables and commands in `target/plan-v/v18b/RESULTS.md`.

Left open: reproducing level 40 while the level-80 render ran on the same GPU gave identical
numbers and a different `execution_hash` (`745c9777…` against `c2322c2c…` solo) — open question 26.
