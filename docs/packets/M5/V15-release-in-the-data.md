# M5 V15 — the release, in the data

Design note: `docs/design/visible-learning.md` **section 7.23**, and 7.22 (V14, the hand that
never opens), 7.21 (V13, the backbone), 5.4 (why the cone's success predicate is x-only).
Spec: §6.2, §6.5 (`Terminate`), §9.3, §10.4, §1.4. Depends on V13, V14.

## the question

V14's policy carries the cube into the bin volume on 9 of 16 held-out seeds and then holds it
there for up to 33 s with the jaw stalled on it. Zero releases in 96 evaluated episodes. The
orchestrator's hypothesis was that the recorded demonstrations end *before* the expert's
`Release` stage — that the policy has never seen a hand open and is reproducing its data
exactly. This packet measures that first and only then changes anything.

## Part 1 — measure, no training

1. **Do the demonstrations contain the release?** Over V14's 200-demonstration `ds-train`
   (and the `.estraj` beside each episode): per episode the gripper column of the recorded
   `action`, the frames that follow the last command that closes the jaw, how many of them
   command it open, the peak command reached, the episode length, and the cube's z at the last
   frame — held in the air, or resting.
2. **Why does the harness not fire for a held cube?** On V14's 1,800-budget evaluation, per
   tick of the carrying episodes (held-out 01 and 04, training 03/11/15): the cube's x, its
   x-velocity, and every term of the Task IR's success predicate, so the blocking term is named
   rather than guessed.

## Part 2 — the predicate

Whatever Part 1 shows, the predicate has to end the episode on a release the data can teach.
The node set is fixed (`Compare`, `Logic`, `Normalize`, `GetJointState`, `GetTime`); no new
`TaskNode` kind, no `es-ir` change. The documents move only through the sanctioned generator
(`regenerate_visible_learning_documents`), so `task_hash` moves and `deployment_hash` must not.

## Part 3 — re-collect, re-train, re-measure

V14's exact recipe with the predicate as the single variable: 200 demonstrations
(`--episodes 200 --seed 1`, 50 Hz, `PACE = 0.5`, with frames), bake, the V13 GroupNorm graph
for 40,000 steps (batch 8, lr 1e-4, seed 0, `--resident-gpu`; 20,000 saved too), pack, and
evaluate at the 1,800-step budget on the training seeds 1–16 and the held-out seeds 101–116,
the final checkpoint twice.

## scope

* **`context`** — `crates/es/tests/cli.rs` (`add_gripper_open_term`'s threshold,
  `expert_solves_the_pinned_seeds`'s new assertion), `crates/es-env/src/plan.rs` (the
  cone-lowering test's cases only), `tests/fixtures/visible-learning/{task,observation,
  observation-v8,evaluation,evaluation-v8}.toml` (regenerated, never hand-edited),
  `docs/design/visible-learning*.md` section 7.23 and open question 21,
  `docs/packets/M5/V15-release-in-the-data*.md`.
* **`forbidden`** — the Deployment IR's limits, the Safety Plane, the acceptance threshold
  (0.5, unchanged), the expert's stages and `demo_cfg`, `crates/es-policy`'s lowering,
  sections 7.21 and 7.22, `docs/ARCHITECTURE*.md`.
* **`INV-17`** — no new trait, no new extension point. One constant and one dataset.

## oracles

1. **The fixture is generated, not typed.** `cargo test -p es --test cli -- --ignored
   regenerate_visible_learning_documents` reproduces every document byte for byte; `cargo xtask
   ci` is green on the result; `deployment.toml` is untouched.
2. **The expert still solves the task.** `expert_solves_the_pinned_seeds` and
   `expert_passes_the_evaluation_harness`, 8 of 8 each, unchanged thresholds.
3. **The release is in the data, and it is asserted.** `expert_solves_the_pinned_seeds` counts,
   per demonstration, the recorded gripper commands that follow the last closed one and exceed
   0.6 rad, and fails below `RELEASE_FRAMES`. A pin of the same kind as `THRESHOLD`: lowering it
   to make a change pass is editing a golden.
4. **A held cube is not a success.** `the_demo_task_cones_lower` evaluates the *lowered*
   success `Expr` on a jaw that is shut on the cube, on a jaw that is opening but has not let
   go, and on a released one — through the very expression `es-env` runs, with no backend.
5. **Determinism.** The held-out suite is run twice on the final checkpoint; the two reports
   must agree to the last digit.
6. **Stop rule.** One variable. Held-out `success_rate ≥ 0.5` (unchanged) opens the six-suite
   sweep and the showcase videos; below it the packet stops and reports the per-episode table.

## measured

Oracle server (RTX 4090), `~/venvs/es` for `es`, `~/venvs/es-lerobot-cuda` for training,
2026-09-16, tree `~/Projects/es-v15`. Artifacts under `~/artifacts/plan-v/v15/`, small results
mirrored to `target/plan-v/v15/`. The numbers, the tables and the verdict are in design note
section 7.23; the two headline answers are:

1. **The demonstrations did contain the release** — and only the first 14 control steps of it.
   The episode ends the instant the success predicate goes true, so the gripper term's 0.6 rad
   threshold cut the expert's opening ramp with the *command* still at 0.614 of the 0.9 it was
   aiming at. Median per demonstration: 14 frames after the last closed command, 8 above 0.30,
   **1** above 0.60, and none at `grip_open`.
2. **The harness was already counting only released cubes.** The gripper term has been in the
   predicate since V1. On held-out 01, 1,691 of 1,800 ticks have the cube in the bin's x span
   and 1,676 of those also satisfy `|vx| < 0.05`; the settling term never fails on contact
   jitter. The single blocking term is the gripper, at 0.078 – 0.093 rad for the whole hold.
3. **Repairing the predicate changed the data and not the policy.** The threshold moved 0.6 →
   0.85, which took each demonstration's opening tail from 14 frames to 19 and its frames
   commanded above 0.6 from 1 to 6, ending on a released cube under a jaw at 0.834 rad. The
   200-demonstration, 40,000-step policy trained on that data scores `success_rate` **0 / 16**
   held out (twice, identical) and **0 / 16** on the training seeds, carries the cube into the
   bin volume on 4 and 5 of 16, and **releases it in 0 of 64 evaluated episodes** — 128 across
   V14 and V15. Held-out `success_rate` is below the unchanged acceptance threshold of 0.5, so
   the stop rule fires: no six-suite sweep, no showcase videos, no second variable.
