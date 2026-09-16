# Packet M5/V19b — the external ACT on the committed documents

Design note: `docs/design/visible-learning.md` section 7.29. Predecessors: V19 (section 7.27, the
checkpoint and the merged Observation IR), V18/V18b (sections 7.26, 7.28, the fixture decision).

## Context

- Allowed scope: no repository file; the run is `~/artifacts/plan-v/v19b/run.sh` and
  `showcase.sh` on the oracle server, in the tree `~/Projects/es-main` (an archive of `ebde599`).
- Inputs: `~/artifacts/plan-v/v19/train-w13/checkpoints/060000/pretrained_model` (LeRobot 0.6.1 ACT
  trained on V15's 200 demonstrations), `~/artifacts/plan-v/v19/observation-v19.toml`
  (`observation_hash 07fad282…`), the committed `task.toml`, `learning.toml`, `deployment.toml`
  (`acceleration_max = 80`, commit `74b01a3`), V19's `eval-w13-{train,holdout}.toml` and
  `evaluation-s13.toml`.

## Spec

Re-import the checkpoint against the committed deployment with the merged tree's
`es policy import-lerobot` and evaluate it, with nothing else changed, on the nominal suite over
training seeds 1–16 and held-out seeds 101–116 and on the six-suite sweep over held-out seeds;
render two held-out successes and the mosaic.

## Oracle

- `es policy import-lerobot --checkpoint … --task task.toml --observation observation-v19.toml
  --deployment deployment.toml --out w13-060000-a80.esb` prints `deployment_hash f2f9a510…`, the
  hash `es ir check` prints for the committed fixture.
- `es eval run` on the three documents; `report.json` carries `passed`.
- `es video showcase` / `es video mosaic` + `python/es/encode_video.py` produce three mp4s.

## Acceptance

`passed = true` on the held-out nominal document (threshold 0.5, unchanged); the videos exist and a
sampled frame sequence shows grasp, carry and release.

## Forbidden

Retraining; any change to a document or to code; disabling or bypassing the Safety Plane; choosing a
different checkpoint than the one V19 reported.

## Results (2026-09-16, RTX 4090, `ES_PYTHON=~/venvs/es`)

| document | `passed` | `success_rate` | `envelope_violation_rate` | fallback ticks |
|---|---|---|---|---|
| nominal, training 1–16 | true | 1.0000 | 0.232 | 0 |
| nominal, held-out 101–116 | true | 0.9375 | 0.208 | 0 |
| six suites, held-out | true | nominal 0.9375 · light_intensity 0.9375 · light_direction 0.8125 · observation_delay 1.0 · torque_noise 0.3125 · backlash 0.9375 | 0.18–0.50 | 4 (light_direction) |

Hashes: `task eb6efefa…`, `observation 07fad282…`, `learning 17613075…`, `policy 308a1f53…`,
`deployment f2f9a510…`. Videos `v19b-a80-holdout-nominal-00.mp4` (seed 101), `-nominal-11.mp4` (seed
112), `-mosaic.mp4`, under `~/artifacts/plan-v/v19b/` and `target/plan-v/v19b/showcase/`; reports
mirrored to `target/plan-v/v19b/reports/`. The single held-out nominal failure (`nominal-04`, seed 105)
is a grasp miss with the jaw open for 1,475 of 1,800 ticks.
