# M5 V16 — why the hand never opens

Design note: `docs/design/visible-learning.md` **section 7.24**, and 7.23 (V15, the release was
in the data), 7.22 (V14, the hand that never opens), 7.21 (V13, the backbone).
Spec: §8.1, §8.5, §8.6, §9.2, §1.4. Depends on V13, V14, V15.

## the question

V15 closed the last two hypotheses about the data. The demonstrations contain the release —
all 200 of them, ending with the cube at rest in the bin under a jaw commanded to 0.846 rad —
and the harness's success predicate can see a released cube. The policy still holds: **0
releases in 128 evaluated episodes**, carrying the cube into the bin on 9 of 16 held-out seeds
and then stalling the jaw on it at 0.078 – 0.093 rad for up to 33 s.

What is left is one channel of a six-wide L1 regression. This packet diagnoses it on the
learning and execution side before changing anything, and then moves exactly one variable.

## Part 1 — diagnose, no training

V15's artefacts only (`~/artifacts/plan-v/v15/`): the baked 200-demonstration set,
`model-40000.safetensors`, the lowered `build/`, and the `.estraj` and frame tiles of the
evaluation runs. Every input port is rebuilt the way the Observation IR builds it, so the
tensors the analysis feeds the module are the tensors `es eval run` fed it.

1. **Open-loop at the release.** Over 20 training demonstrations, locate the release onset —
   one past the last frame commanding the jaw shut — and run the module on the baked inputs of
   the 15 frames either side. Per relative frame: the predicted gripper rows 0..9 against the
   target, and whether the arm rows predict "keep lowering" or "stationary".
2. **Where the closed-loop hold sits.** From the `.estraj` of V15's carrying episodes, the hold
   state (arm qpos, gripper qpos, cube pose) against the demonstrations' release-onset states
   in the policy's normalized input space, per dimension; then those hold observations —
   including the run's own rendered tile — through the module open-loop. If it predicts closed
   at the hold and open at the onset, the smallest perturbation of the hold that flips it.
3. **The commanded gripper value.** The raw gripper output distribution at hold and at grasp,
   and what the action port's unit actually is, so "is the gripper channel's scale such that
   closed regresses beyond the demonstrations' value?" is answered from the document and not
   from memory.
4. **The ensemble on the gripper.** One carrying episode's chunk predictions replayed through
   `plane_chunk`'s temporal ensemble, against executing only the newest chunk's row 0, to
   quantify the lag and attenuation the blend imposes on a closed → open step.

## Part 2 — move one variable

Whatever Part 1 names, and nothing else. The candidates the orchestrator listed are
(a) a per-channel weight in the training objective, (b) the execution blend, (c) the hold state
being off the release manifold; Part 1 decides between them by measurement and the two that
lose are reported with their numbers rather than left as opinions.

Re-measure at the 1,800-step budget: training seeds 1–16 and held-out seeds 101–116 twice, per
episode lifted / carried / released. Held-out `success_rate` ≥ 0.5 continues to the six-suite
sweep and the showcase videos; below it the stop rule fires and the packet reports.

## scope

* **`context`** — `python/es/train_act.py` (the loss weighting flag),
  `crates/es-policy/tests/ir_training.rs` (its oracle), `docs/design/visible-learning*.md`
  section 7.24 and the open questions, `docs/packets/M5/V16-the-hand-opens*.md`.
* **`forbidden`** — every IR schema; the Deployment IR's limits, the Safety Plane and the
  acceptance threshold (0.5, unchanged); `crates/es-policy`'s lowering; the fixtures under
  `tests/fixtures/visible-learning/`; sections 7.21 – 7.23; `docs/ARCHITECTURE*.md`.
* **`INV-17`** — no new trait, no new extension point. One optimizer flag and four
  measurements.
* Nothing the packet adds enters a hash slot: `--channel-weight` is an optimizer knob like
  `--batch` and `--lr` (spec 8.1), so `learning_hash`, `lowering_hash`, `task_hash`,
  `observation_hash` and `deployment_hash` are all V15's, and only the weight bytes move.

## oracles

1. **`channel_weight_of_one_is_the_unweighted_loss`**
   (`crates/es-policy/tests/ir_training.rs`, `--ignored`, needs torch): an all-ones weighting
   produces a **byte-identical** `--loss-curve` to no weighting, because `(|d| * 1).mean()` is
   `l1_loss` — which is what protects every loss number already in design note section 7. A
   weight that is not one moves the curve, and the run summary records the vector it ran under.
2. **`cargo xtask ci`** green — fmt, clippy `-D warnings`, context budget, layering,
   spec-refs, goldens.
3. **The measurement is the oracle for Part 1.** Each of the four questions is a printed table
   over V15's artefacts, reproducible from `~/artifacts/plan-v/v16/part1*.py`, and section 7.24
   carries all four.

## acceptance

Part 1's four tables exist and name one cause. Part 2 moves exactly that one variable, and the
1,800-budget evaluation on both suites is reported beside V15's whether it improves or not.
`cargo xtask ci` is green. No IR document and no golden file changes.
