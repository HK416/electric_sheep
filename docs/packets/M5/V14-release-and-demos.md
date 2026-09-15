# M5 V14 — the release, and four times the demonstrations

Design note: `docs/design/visible-learning.md` **section 7.22**, and 7.19 (V11, the control
period), 7.20 (V12, the pairing fix and the 50-demonstration set), 7.21 (V13, the GroupNorm
backbone and the baseline this packet is measured against), 5.4 (why the success predicate is
x-only). Spec: §6.2 (`max_episode_steps`), §9.3 (the position clamp), §10.4, §1.4.
Depends on V11, V12, V13.

## the two questions

V13's policy — the IR-owned graph with the cube's pose in its state, trained 20,000 steps on
V12's 50 demonstrations — **grasps the cube, lifts it 118 – 124 mm and carries it to the bin in
3 of 16 training-seed episodes, and then holds it there until the 900-step budget ends**. Held
out, 0 of 16. `violation.position` dominates the failure histogram and the gripper joint carries
the largest open-loop residual.

Two things follow, and this packet is one each.

1. **Does the policy release if it is given time?** The 900-step budget was hiding the answer:
   all three carries were still holding the cube in the right place when the episode ran out.
   Part 1 raises the budget to 1,800 control steps (36 s) and re-runs V13's own weights.
2. **One variable: 50 demonstrations → 200.** Nothing else moves. V13's stop rule named
   resolution or demonstration count as the next single variable; this is the second.

## Part 1 — the budget, on V13's own weights

`max_episode_steps` is not hand-written: `tests/fixtures/visible-learning/task.toml` is generated
by `regenerate_visible_learning_documents`, so the change is the generator's constant
(`(max_steps, timeout_s)`, 900/18.0 → **1800/36.0** — the Task IR's `Timeout` leaf compares
elapsed seconds, so both move together) followed by a regeneration.

Every hash downstream of the Task IR moves and **the weights do not**: V13's
`model-20000.safetensors` is repacked with `es policy pack` into a bundle built from the
regenerated documents. That is the honest statement of what the comparison is — the same
function, a different episode budget.

## Part 2 — 200 demonstrations

V12's exact collection command at 50 Hz with the `PACE = 0.5` expert, `--episodes 200 --seed 1`
(seeds beyond 50 are new draws of the same `Randomization` streams, not repeats), then bake,
then the V13 graph with V2's knobs: `--batch 8 --lr 1e-4 --seed 0 --device cuda --resident-gpu`,
20,000 steps, and 40,000 as well if the loss is still falling at 20,000.

Evaluated on the training seeds 1–16 and the held-out seeds 101–116, **at the 900-step budget**,
because Part 1's answer is that no episode releases at 1,800 either — so 1,800 buys nothing and
900 is what V13's table is on.

## scope

* **`context`** — `crates/es/tests/cli.rs` (`regenerate_visible_learning_documents`'s
  `max_steps`/`timeout_s` constant only), `tests/fixtures/visible-learning/{task,observation,
  observation-v8,evaluation,evaluation-v8}.toml` (regenerated, never hand-edited),
  `docs/design/visible-learning*.md` section 7.22 and one new open question,
  `docs/packets/M5/V14-release-and-demos*.md`.
* **`forbidden`** — the Deployment IR's limits, the Safety Plane, the acceptance threshold
  (0.5, unchanged), `crates/es-policy`'s lowering (V13 owns it), sections 7.20 and 7.21,
  `docs/ARCHITECTURE*.md`.
* **`INV-17`** — no new trait, no new extension point. One constant and one dataset.

## oracles

1. **The fixture is generated, not typed.** `cargo test -p es --test cli -- --ignored
   regenerate_visible_learning_documents` reproduces every document byte for byte from the
   scene and from the constant; `cargo xtask ci` (goldens, spec-refs, layering, context budget)
   is green on the result.
2. **The weights did not move.** `es policy pack` on the regenerated bundle prints the same 86
   tensors and the same `weights_hash` and `lowering_hash` as V13, and a different
   `task_hash` / `observation_hash` / `policy_hash`.
3. **Release, measured per episode.** The `.estraj` record of every evaluated episode, read
   against the 3-D bin volume `cube_in_the_bin` (`crates/es/tests/cli.rs`) and the gripper's
   own open threshold (0.6 rad, the Task IR's own): lifted / carried / released, and the tick
   of each, beside the harness's verdict.
4. **`violation.position`, by joint.** `events.json` carries the `ViolationKind` bitset per
   frame and no joint index, so the breakdown is derived by replaying the `.estraj` positions
   against the Deployment IR's soft bounds (`[lower + margin, upper - margin]`, the exact
   interval `SafetyPlane::clamp` stage 2 clamps the command to).
5. **Determinism.** The held-out suite is run twice on the final checkpoint; the two reports
   must agree to the last digit.
6. **Stop rule.** One variable. Held-out `success_rate ≥ 0.5` (unchanged) opens the six-suite
   sweep and two showcase videos; below it, the packet stops and reports the per-episode
   carry/release table so the next packet can be chosen.

## measured

Oracle server (RTX 4090), `~/venvs/es` for `es`, `~/venvs/es-lerobot-cuda` for training,
2026-09-16, tree `~/Projects/es-v14` built from the V12 merge (`Env::step` snapshots the state
before the backend step). Artifacts under `~/artifacts/plan-v/v14/`, small results mirrored to
`target/plan-v/v14/`.

### 1 — the documents

| | V13 | **V14** |
|---|---|---|
| `max_episode_steps` / `Timeout` leaf | 900 / 18.0 s | **1800 / 36.0 s** |
| `task_hash` | `d7a7c061…` | **`78814eb4…`** |
| `observation_hash` | `4b069f6c…` | **`72b8609a…`** |
| `learning_hash` | `5dac0a46…` | `5dac0a46…` |
| `lowering_hash` | `70a8fec7…` | `70a8fec7…` |
| weights: tensors / `weights_hash` | 86 / `0dc1f930…` | 86 / `0dc1f930…` |
| `policy_hash` | `840aef94…`-lineage | **`8dbff824…`** |

The Learning IR and its lowering are untouched and the weight bytes are V13's own; the task,
observation, evaluation and policy hashes all move, because the episode budget is a Task IR
value and the hash chain is doing its job (§5.3).

### 2 — Part 1: the policy does not release, and 18 more seconds do not change that

`es eval run --frames`, the nominal suite, 16 episodes, budget 1,800.

| | training seeds 1–16 | held-out 101–116 |
|---|---|---|
| `success_rate` (harness) | 1 / 16 | **0 / 16** |
| cube lifted clear of the table | 3 / 16 | 3 / 16 |
| **cube carried into the bin volume** | **3 / 16** (03 @ 917, 11 @ 147, 15 @ 172) | **2 / 16** (01 @ 143, 04 @ 142) |
| **cube released inside the bin** | **0 / 16** | **0 / 16** |
| cube still inside the bin at tick 1800 | 3 / 16 | 2 / 16 |
| `envelope_violation_rate` | 0.4227 | 0.5457 |
| `episode_length` | 1691.5 | 1800.0 |

Every carrying episode holds the cube inside the bin volume for the whole remainder — 883 to
1,653 control steps, 17.7 to 33.1 s — with the gripper at 0.078 – 0.093 rad, which is the jaw
stalled on the 30 mm cube. **Not one episode opens the hand.** The extra 900 steps changed no
verdict on either suite; they only proved the hold is indefinite.

The one `success` is the same episode V13 reported: training seed 3 (`nominal-02`), 64 ticks, a
1.8 mm lift. It is not a carry.

### 3 — Part 1: `violation.position` is the gripper, and only the gripper

Replaying the `.estraj` positions against the Deployment IR's soft bounds, per joint:

| joint | training seeds, ticks outside the soft bound | held-out |
|---|---|---|
| `shoulder_pan` / `shoulder_lift` / `elbow_flex` / `wrist_roll` | 0 | 0 |
| `wrist_flex` | 23 (0.1%) | 18 (0.1%) |
| **`gripper`** | **10,145 of 27,064 (37.5%)** | **14,690 of 28,800 (51.0%)** |
| harness `violation.position` | 9,265 | 13,380 |

The gripper's soft interval is `[-0.1245, 1.6953]` rad; the pinned ticks sit within 1 mrad of
the **lower** bound. The policy is commanding the jaw shut harder than the envelope allows, for
half of every episode, and the Safety Plane clamps it there — which is the same fact as "it
never opens", seen from the other side.

### 4 — Part 2: 200 demonstrations

| | V12 / V13 (50) | **V14 (200)** |
|---|---|---|
| demonstrations, `es loop collect --episodes N --seed 1` | 50 / 50 `Success` | **200 / 200 `Success`** |
| frames | 9,038 | **35,918** |
| collect / bake wall clock | 50 s / 1 s | **208 s / 8 s** |
| training loss, initial → 20,000 → 40,000 | 0.0632 → 0.0162 → — | 0.1167 → **0.0186** → **0.0148** |
| training wall clock, 40,000 steps | 719 s (20,000) | **1,411 s** |

The loss is still falling at 20,000, so 40,000 is the final checkpoint and both are packed.

### 5 — Part 2: the success table, beside V13's

Budget 900, the nominal suite, `es eval run --frames`.

| | V13, 50 demos, 20k | **V14, 200 demos, 40k** |
|---|---|---|
| `success_rate`, training seeds 1–16 | 1 / 16 | **1 / 16** |
| `success_rate`, held-out 101–116 | 0 / 16 | **0 / 16** (twice, identical) |
| **cube carried into the bin, training** | **3 / 16** | **8 / 16** |
| **cube carried into the bin, held-out** | **0 / 16** | **9 / 16** |
| **cube released inside the bin** | 0 / 16 | **0 / 16** |
| `envelope_violation_rate`, training / held-out | 0.5396 / 0.6392 | **0.1587 / 0.2060** |
| `violation.position`, training / held-out | 5,655 / 7,542 | **657 / 1,357** |
| `episode_length`, training / held-out | 847.75 / 900 | 855.125 / 900 |

The two held-out runs on the 40,000-step checkpoint agree to the last digit, so the numbers are
the policy's and not the schedule's.

Held-out `success_rate` is 0, which is below the acceptance threshold of 0.5, so **the stop rule
fires**: no six-suite sweep and no showcase videos, and no second variable in this packet.

### verdict

Four times the demonstrations is the largest single-variable move plan V has measured. It turns
**0 / 16 held-out carries into 9 / 16** and cuts the envelope violation rate by a factor of
three; `violation.position` falls from 7,542 frames to 1,357. The policy now finds the cube,
grasps it, lifts it ~125 mm and puts it over the bin on more than half of the seeds it has never
seen.

And the harness still scores **0 / 16**, because the last thing the task needs is the one thing
the policy has never done: **open the hand.** Zero releases in 96 evaluated episodes across both
parts, at 900 steps and at 1,800. The failure is not reach, not grasp, not generalization and
not the budget; it is a single joint that is commanded shut for the whole episode and clamped
there by the Safety Plane.
