# M11 X7 — vision RL on the path tracer, with randomization on

Spec: §28.14 rules 4, 5, 7 and wave 3, §12.4 (the nine metrics), §13.4, §3.5. Depends on X3, X3b,
X5, X6, P-M11-R1 (`es train` accepts image inputs). Design notes: `rl-continuation.md` 2c (the
vision rollout), `renderer.md` 12.9 and 13 (cost). Type D.

## the question

**Does PPO learn the SO-101 reach task from path-traced 96×96 pixels, with `seed = "tick"`, SVGF,
and every visual and physical randomization target on? What does the path tracer cost against
the rasterizer, and which training renderer transfers better to the other one?**

## spec

* **Documents:** `tests/fixtures/rl/task-reach-vision.toml` (X3) gets variants, generated the same
  way as the other documents:
  * `…-pt-dr.toml`: `Pt`, `seed = "tick"`, `svgf` per row, with X5's light, colour, ambient,
    geom-rgba, camera-pose and fov targets and X4's mass/friction/gain targets at moderate ranges
    (the ranges are stated in the report).
  * `…-rs-dr.toml`: the same with `Rs`.
  * `…-pt.toml` and `…-rs.toml`: randomization off.
* **Stage 1, sweep** (server, GPU lock, batched render from X3b): `Pt` spp ∈ {4, 8, 16} ×
  SVGF {off, on} for 1,000 iterations on seed 0. Record ms/frame, samples/s (the nine metrics) and
  return. The row with the best return per wall-clock hour becomes the training setting. The
  choice rule is written down before the sweep runs.
* **Stage 2, rows**, 3 seeds each, same budget: `Pt`+DR, `Pt` no-DR, `Rs`+DR, `Rs` no-DR.
* **Evaluation:** held-out 16 seeds on `evaluation-reach-vision-*.toml` with the eval perturbation
  suites. Cross-render: a policy trained on `Pt` is evaluated on the `Rs` document and vice versa.
  That comparison is the "which renderer transfers" answer.

**Owner decision 2026-09-24 (the rerun).** After R3 moved the learner to CUDA, the rerun covers
only the four pixel-only rows (`-pix`, no `cube_pose` in the state): `pt-dr-pix`, `pt-pix`,
`rs-dr-pix`, `rs-pix`, 3 seeds each, 4,000 iterations, `--device cuda`, the `Pt` rows at **4 spp**.
The rows with `cube_pose` are not run; they stay in the report as the confound finding. Stage 1
shrinks to `pt-dr-pix` at 4 spp, SVGF off against on (1,000 iterations, seed 0): the higher mean
return over the last 100 iterations wins, unless the difference is smaller than the larger of the
two runs' std over those iterations, in which case SVGF off (the cheaper one) wins. The 4 spp
documents are new files beside the committed 16 spp ones (`x7_pt_pix_variants`). Budget ≈ 50–55
GPU-hours; a projection above 65 h stops the run.

## context

```
tests/fixtures/rl/**
tests/golden/train/**
crates/es/tests/cli.rs
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/X7-vision-rl-pt.md
docs/packets/M11/X7-vision-rl-pt.ko.md
```

## oracle

1. `cargo test -p es --test cli vision_reach_dr_documents_check` and dry-run plan goldens
   (additions).
2. Server stage 1 table (ms/frame, samples/s, return at 1,000 iterations, per spp × SVGF).
3. Server stage 2 table: 4 rows × 3 seeds, held-out `success_rate` per suite, cross-render
   columns, wall clock, the nine metrics.
4. check-scope, verify-goldens.

## acceptance

Oracles 1–4; `rl-continuation.md` (+ko) section 7 gains X7 with a one-sentence answer per question.

## forbidden

Changing code (this is a measurement packet: a bug found is a new packet); retuning PPO per row
beyond the one recipe; single-seed claims.
