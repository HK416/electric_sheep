# M14 Q-V — plan Q's real runs (S3, action blocks)

Plan Q's Task Q5, on this PC (Windows 11, RTX 3060 12 GB, 64 GB RAM), 2026-09-29, with the editor
open on each run as the owner asked. Every project was made by the editor's own model
(`Project::create`, `write_run`) and every run launched with the argv the editor's Start uses.

## 1. A try from ② (packet Q4)

A scratch hint-card project; block 1 "above the cube" changed from 4.5 cm to 6 cm in ②; "한 번 해
보기" ran `es eval run --jobs 1` on one episode at seed 1001 (a seed outside every evaluation
seed, review of Q3): **succeeded**, 511 frames, 16.5 s from the derived `evaluation.toml` to
`report.json`. ② marked block 1 with the wrist-angle warning ("at this wrist angle the arm cannot
always reach") — the one try happened to be a reachable position.

## 2. A short run of the edited program (Q5): three bugs found

The same edit, a new project (`target/yv/cube-hint-q5`), the short preset, 200 demonstrations.

| stage | result |
|---|---|
| collect | **10 success, 187 failure, 3 timeout**; episode lengths `511, 520, 520, 1800, …, 11, 1, 1, 1, …` — 186 episodes of **one frame** |
| expert gate | passed, nominal 11/16 (0.6875 ≥ 0.5) |
| train | loss **NaN from step 1,130**; the editor's light stayed green; stopped by hand at step 3,800 |

- **The collector leaked an abort into every later episode** (P-M14-R1, bug 1). Episode 13 was
  a genuine failure — block 1 out of reach, the demonstration ends by name (spec 17.2) — and the
  `Intervened` wrapper's `aborted` flag was never cleared, so each later episode ended after its
  first frame. Measured on the same 16 seeds (1001–1016) the edited program succeeds 11 of 16
  whether the evaluation or the collector drives it; before episode 13 the collection had 10 of
  13. The same wrapper also kept a stale `injected` flag, so every expert dataset recorded the
  first frames of each episode after the first as human-driven: re-collected datasets now
  differ in the `intervention` column and content hash. The original program never aborts
  (200/200), which is why no earlier run showed it.
- **A diverged training was invisible** (bug 2). `train_act.py` printed `"loss": NaN`, which is not
  JSON; `es train` dropped those lines and the telemetry wire rejected a NaN frame too, so no point
  after step 1,130 reached the editor and its "broken" rule never fired. Now non-finite values are
  written as `null`, read back as NaN, the trainer stops at the first non-finite loss with a named
  message, and the light's *broken* outranks *stopped*.
- **The cycle trained on a collection that mostly failed** (bug 3). Now `es loop cycle` refuses,
  before the train stage, when fewer than half of the demonstrations succeeded, and says to fix
  the program in ②.
- **The editor, watched:** opening it mid-collection counted only the demonstrations heard since
  attaching ("2 of 200" when 25 were done) — fixed (`9ef9b24`: the count comes from the episode
  index; successes read "of the N the editor saw").

## 3. The grip wait (the finding of M13/Z7, measured)

Z7 found every demonstration of the committed program holding still for 221 frames (4.4 s, 43 %
of the episode) between closing the gripper and lifting, and the long-trained policy failing 65
of 90 timed-out attempts as *never lifted*. With both grip waits at 1 s the demonstrator still
succeeds 4 of 4 (seeds 5000–5003) with the same plane counts, and its demonstrations are 38 %
shorter.

The experiment: the hint card at *long* (60,000 steps), the same seeds and settings as Z7, one
change made in ② — both "wait" blocks from 5 s to 1 s. Project `target/yv/cube-hint-wait1`.

| | Z7 (5 s waits) | 1 s waits |
|---|---|---|
| demonstrations | 200/200, 103,881 frames | 200/200, 63,901 frames |
| expert gate | passed | passed |
| previews 1,000 / 5,000 / 20,000 / 60,000 | 0/4 each; the cube never lifted in any of the 16 | 0/4 each; lifted in 1, 2 and 2 of the last three, carried to the bin, never let go |
| train (60,000 steps) | 54.8 min | about 55 min (1,150 samples/s) |
| eval, nominal | 0/16 | **2/16** |
| eval, all suites | 6/96 | 4/96 |
| attempts that lifted the cube (of 96) | 30 | 38 |
| ended with the cube in the bin / held above it | 14 / 12 | 17 / 9 |
| `envelope_violation_rate`, nominal (torque noise) | 0.117 (0.436) | 0.135 (0.707) |
| whole run | 1 h 38 min | 1 h 24 min |

**What it says, and what it does not.** The shorter wait moved nominal from 0/16 to 2/16 — the
hint card's first nominal successes on today's data — and lifting from 30 to 38 of 96, but all
suites fell from 6 to 4, and one training per arm carries the ±0.3 run-to-run spread the M10
review measured (S-4). So the 4.4 s still segment is **not established** as the reason the policy
fails; one more training seed per arm is what would settle it. The previews over-stated the
difference: Z7's four preview positions happened never to be lifted, while its final evaluation
lifted in 30 of 96 — four episodes are for watching progress, not for comparing runs.

**What both runs share: the robot does not let go.** In both, about one attempt in five (20 and
22 of 96, successes excluded) ended with the cube in or over the bin and the task not counting
it — the ones checked by hand had the gripper shut, short of the task's `gripper > 0.85` clause. Every demonstration holds only **27 frames** (0.54 s) of letting
go, whatever the release block's wait — the episode ends on success the moment the gripper
passes 0.85 — so the release is the thinnest part of the data. The editor named such attempts
*too late* (fixed: `NotReleased`, P-M13-R4) and, when held above the bin, *left outside* (being
fixed); R4's first advice ("wait longer after opening") could not help and was corrected
(`e88f920`).
