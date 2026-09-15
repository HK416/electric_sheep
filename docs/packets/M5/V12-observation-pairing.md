# M5 V12 — the observation a policy acts on is the one the demonstration records

Design note: `docs/design/visible-learning.md` **section 7.20**, and 7.19 (V11's numbers, what
this is compared against), 7.18 (V10's measurement of the pairing, which was read as noise),
7.12–7.13 (envelope semantics), section 5 (the scripted expert). Spec: §12.1, §13.2, §18.5.
Depends on V1c, V2, V6, V6b, V8, V9, V10, V11.

## the defect this packet fixes

`crates/es-env/src/env.rs`, `Env::step`, before this packet:

```
set_ctrl(ctrl) -> backend.step(substeps) -> let state = self.backend.state()
              -> StepRow { qpos/qvel/sensordata: <post-step>, ctrl: last_ctrl, ... }
              -> recorder.push
```

So a demonstration row `t` paired `observation.state[t]` — the state **after** `action[t]` was
executed — with `action[t]`. `crates/es-data/src/collect.rs` then rendered the frame and pushed
the `.estraj` pose from that same post-step state ("from the state the step ended in — the same
state the row above recorded").

But nothing computes an action from a state it has not reached yet. `Env::step_with_policy`
runs `observe_window -> infer_window -> emit_actions -> step`, and `es_eval::runner` runs
"observe -> plan -> infer -> validate -> step" with the frame captured before the inference.
Training therefore learned `(s_{t+1}, image_{t+1}) -> a_t` while inference asks
`(s_t, image_t) -> a_t`: **one control period of lag in every input of every demonstration plan
V ever collected.** LeRobot's convention is the other one — `observation[t]` is what `action[t]`
was chosen from, executed after observing it.

V10 measured this and read it as noise: at 200 Hz one control step was 5 ms and the shift was
0.7 mrad. At the true 50 Hz (V11) the per-step increment is **0.04 rad median**, which is twice
the median following error the policy is asked to reproduce.

## the fix, and why it is in the env

One place, `Env::step`, so every consumer inherits it: snapshot `qpos/qvel/sensordata` (and the
tick) **before** `backend.step` and put that in the `StepRow` beside the `ctrl` that was just
applied. Reward, termination and failure stay the transition's — they are statements about what
the step produced, and `done` still marks the last frame of the episode, which is LeRobot's
semantics. The snapshot is a scratch buffer allocated once (`PreStep`), like `reset_qpos` and
`last_ctrl` beside it.

The collector moves its frame render and its `.estraj` push to the same instant — before the
step rather than after — so image, state row and trajectory are one state: the one the policy
will be handed at inference. That also removes a smaller defect nobody had named: on a terminal
step `Env::step` auto-resets the env before returning, so the collector's post-step read
rendered the **next episode's reset state** into the last frame of every episode.

Not touched: the Safety Plane and `validate`'s signature (`INV-12`, `INV-13`), the IR documents
and their hashes, and `dataset_schema_hash` — the columns and their meaning as a schema are
unchanged; only the content moves.

`es_eval::runner` needed no change: it already captured the frame and the trajectory row before
inference. That is the whole point — the evaluator was right and the recorder was wrong.

## scope

* **`context`** — `crates/es-env/src/env.rs`, `crates/es-data/src/collect.rs`,
  `crates/es-data/tests/loop_learning.rs`, `crates/es/tests/cli.rs`,
  `docs/design/visible-learning*.md` section 7.20 and open question 19,
  `docs/packets/M5/V12-observation-pairing*.md`.
* **`forbidden`** — `SafetyPlane` and its `validate` signature (`INV-12`, `INV-13`),
  `tests/fixtures/**` (no document, no limit, no gate), the dataset schema,
  `docs/ARCHITECTURE*.md`.
* **`INV-17`** — no new trait. `PreStep` is a private struct of three `Vec<f64>`.

## oracles

1. **The row is the state its action was computed from** — `cargo test -p es-env --lib
   the_recorded_row_is_the_state_the_action_was_computed_from`: after one `step(ctrl)` on a
   scene that moves, the recorded row's `qpos` equals the state before the step, differs from
   `backend.state()` after it, and its `ctrl` is the one that was given.
   `crates/es-data/tests/loop_learning.rs::frames_are_written_once_per_control_step` is the
   collector's half: frame `i` and `observation.state[i]` are bit-identical instants.
2. **The pairing measurement flips** — on a freshly collected set, the median over frames of
   `max_j |action[t] - qpos[t+k]|` for `k = -1, 0, +1`. Before the fix the smallest is `k = 0`
   (the row is already `s_{t+1}`, so that is the following error); after it the smallest must be
   `k = +1`, and `k = 0` must be the command lead. `~/artifacts/plan-v/v12/pairing.py`, run on
   both V11's and V12's baked sets.
3. **Nothing else moved** — `recorded_actions_replay_to_the_same_outcome` still reproduces every
   demonstration on the new set, and `showcase_replay_of_a_real_run_is_bit_identical` (V9) still
   replays the trajectory into the frames that were recorded, bit for bit.
4. **Re-collect, re-train, re-measure** — V11's exact commands, and the evaluation run **on the
   training seeds** (1–16) as well as on the held-out 101–116, against V11's 0/16 on both.
5. **Stop rule** — if the IR-owned graph still scores 0/16 *on its own training seeds* after
   the fix, that is a closed-loop defect and the next packet is a per-tick comparison of the
   policy's prediction against the expert's action, not more data.

## measured

Oracle server (RTX 4090), `~/venvs/es` for `es`, `~/venvs/es-lerobot-cuda` for training,
2026-09-16. Artifacts under `~/artifacts/plan-v/v12/`, V11's beside them under `v11/`.

### 1 — the two unit oracles

Both pass, and `cargo xtask ci` is green with no golden and no fixture hash moved.

### 2 — the pairing flips, by exactly one control step

`pairing.py` over the 50-episode baked set of each packet (9,038 frames both times — the fix
changes what a row *contains*, not how many there are):

| median over frames of `max_j abs(action[t] − qpos[t+k])` | V11 (post-step row) | **V12 (pre-step row)** |
|---|---|---|
| `k = −1` | 0.06159 rad | 0.11149 rad |
| `k = 0` | **0.02094 rad** | 0.06159 rad |
| `k = +1` | 0.06753 rad | **0.02137 rad** |
| the smallest | `k = 0` | **`k = +1`** |

V12's `k = 0` is V11's `k = −1` to the last digit — the table is the same measurement shifted by
exactly one row, which is what the fix claims to be. The command lead is 0.062 rad and the
following error 0.021; the policy is now asked to map the state it will actually be given to the
command that state was answered with.

### 3 — the demonstrations still replay

`ES_V10_DATASET=<the new set> cargo test --release -p es --test cli
recorded_actions_replay_to_the_same_outcome`: **50 demonstrations recorded 50 cubes in the bin,
replaying `action` reproduces 50 (50 `Success`), `action_commanded` 50**, unchanged from V11.
Collection itself is unchanged too: 50/50 `Success`, 9,038 frames, `dataset_schema_hash`
`1f5ddafc…8777c` (**unmoved**), `dataset_content_hash` `b13af58e…1a35` → `3bfaf41f…3d24`.

`showcase_replay_of_a_real_run_is_bit_identical` (V9's oracle) on the new run's first nominal
cell: **900 frames identical** (the oracle reads `<run>/frames/<cell>`, so the run's
`--frames` directory is linked in as `nominal-20000-a/frames`). The `.estraj` record and the recorded frames moved together, as
they must — they are now written from one read of the state instead of two.

### 4 — the two policies, against V11

The IR-owned graph, V2's exact knobs on the new 50-episode set (`es policy lower` ->
`train_act.py --batch 8 --lr 1e-4 --seed 0 --device cuda --resident-gpu`, 20,000 steps, initial
loss 0.0583 -> final **0.0141**, 654 s on the RTX 4090; V11 reached 0.0132 from 0.0565):

| IR-owned graph, 20,000 steps | V11 | **V12** |
|---|---|---|
| nominal, seeds 101-116 (run a) | 0 / 16 | **0 / 16** |
| nominal, seeds 101-116 (run b) | 0 / 16 | **0 / 16** |
| **training seeds 1-16** | 0 / 16 | **0 / 16** |
| `envelope_violation_rate`, nominal | 0.2735 | **0.0371** |
| `envelope_violation_rate`, training seeds | 0.3152 | **0.0365** |
| failure histogram, training seeds | fallback 280, position 3402, acceleration 949, velocity 498, rate 280 | **position 6, acceleration 467, velocity 342, no fallback** |
| `episode_length` | 900 | 900 |

Runs a and b agree to the last digit, so the number is the policy's. The **envelope violation
rate fell by a factor of eight and position violations by a factor of 560** -- the commands the
policy emits are now nearly always reachable from the state it emitted them in, which is exactly
what removing a one-step lag from the input should do. It still never reaches the cube.

**The stop rule fires: 0/16 on its own training seeds.** Seed 1 is training episode 0.

### 5 -- why, measured: the trained fit is not the deployed fit

The orchestrator's open-loop analysis of V11's checkpoint (`~/artifacts/plan-v/v11/openloop/`)
found the reason before this packet ran, and it is **not** the pairing:
`train_act.py --batch 8` accumulates eight **single-sample** forwards, so every `BatchNorm2d` in
the lowered ResNet18 trains on `N = 1` statistics, while inference runs `model.eval()`
(`crates/es-policy/python/torch_ref.py`) and uses the running statistics. Re-run against V12's
own checkpoint and baked set (`~/artifacts/plan-v/v12/openloop.py`, three training episodes):

| chunk L1 over the 10 emitted rows, training episodes 0 / 1 / 2 | V11 | **V12** |
|---|---|---|
| the module in `train()` -- what the loss measured | 0.0107 / 0.0122 / 0.0110 | 0.0129 / 0.0127 / 0.0130 |
| the module in `eval()` -- **what `es eval run` executes** | 0.0313 / 0.0373 / 0.0385 | 0.0337 / 0.0438 / 0.0441 |
| baseline: hold the current pose for every row | 0.0484 / 0.0488 / 0.0481 | 0.0565 / 0.0571 / 0.0561 |

The deployed fit is **three times** the trained fit and only a quarter better than "do nothing",
on episodes the policy was trained on. That gap is the same size before and after the pairing
fix, which is what says the pairing was never the whole defect -- but the open-loop tables also
show no lag left to find: row 0 of the emitted chunk is closest to `action[t]`, and the blend
the evaluator actually executes is now the *aligned* one (before the fix the evaluator executed
the chunk predicted from `s_{t-1}`, the row V11's table marks "as evaluated"). The module emits
**10** rows, not the contract's 16 (`v5_actions = v4_chunk[:10]`).

### 6 -- the external ACT

LeRobot's own ACT is the control for the finding above: its backbone uses `FrozenBatchNorm2d`
and it trains on real batches of 8, so it has no train/eval gap. V8's pipeline unchanged on the
**new** export (`es dataset export --lerobot-v3 --drop action_commanded,action_source
--state-dim 6` -> `lerobot-train --policy.type=act --steps=100000 --batch_size=8 --seed=0` ->
`es policy import-lerobot` -> `es eval run`):

| LeRobot ACT, 100,000 steps, nominal | V8 (200 Hz) | V11 (50 Hz) | **V12 (50 Hz, paired)** |
|---|---|---|---|
| `success_rate` | 1 / 16 | 0 / 16 | **0 / 16** |
| `envelope_violation_rate` | -- | 0.0523 | **0.4267** |
| `episode_length` | 900 | 900 | 900 |

2,175 s of `lerobot-train` on the RTX 4090. **It is still 0/16, and this is the number that
keeps the next packet honest**: the external ACT has no train/eval normalization gap, so
`BatchNorm` cannot be the whole story either. Its envelope violation rate moved the *opposite*
way from the IR-owned graph's (0.0523 -> 0.4267, with 110 fallbacks and 5,981 acceleration
violations) -- an ACT trained on the corrected pairing emits commands the plane has to clamp
harder, which is the signature of a policy that is chasing a state it is no longer being shown
one step late. Neither policy reaches the cube either way.

**Verdict.** The pairing was a real defect and is fixed at its root: every consumer of `Env`
inherits the correct `observation[t] -> action[t]`, the measurement flips by exactly one row,
and nothing else in the pipeline moved (replay 50/50, V9 bit-identical, schema hash unmoved).
It was **not** sufficient: 0/16 held out and 0/16 on the training seeds for the IR-owned graph,
0/16 for the external ACT. What the fix bought is measurable but not success -- the IR graph's
envelope violation rate fell 8x and its position violations 560x. The stop rule fires, and the
next variable is named and measured rather than guessed: open question 19.

## acceptance

1. **The row is the state its action was computed from.** Met -- both unit oracles pass and
   `cargo xtask ci` is green with no golden moved.
2. **The pairing flips.** Met, and by exactly one row: V12's `k = 0` is V11's `k = -1` to five
   decimals, and the smallest pairing moves from `k = 0` to `k = +1`.
3. **Nothing else moved.** Met -- 50/50 demonstrations replay to 50 cubes in the bin, and V9's
   replay is 900 frames bit-identical.
4. **Re-collect, re-train, re-measure.** Met, and the answer is negative for the IR-owned graph:
   0/16 held out, 0/16 **on its own training seeds**, with the envelope violation rate down 8x.
5. **Stop rule.** Fired.

## the next packet

The stop rule names it and the open-loop measurement confirms it: **the IR-owned graph's
deployed fit is not its trained fit, because `train_act.py` feeds `BatchNorm2d` one sample at a
time.** That is open question 19, and it is one variable: vectorize the batch lowering so the
BatchNorms see real batches, or make the IR's ResNet18 use frozen / group normalization. Neither
touches this packet's fix, which is a correctness fix the next packet inherits.
