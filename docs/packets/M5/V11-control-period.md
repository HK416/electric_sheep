# M5 V11 — one control step is one control period

Design note: `docs/design/visible-learning.md` **section 7.19**, and 7.18 (V10's finding, and
open question 18 which this packet answers), 7.16 (V8's numbers, what this is compared against),
7.12–7.13 (envelope semantics), section 5 (the scripted expert). Spec: §9.2, §12.1, §12.4,
§13.2, App. B.5. Depends on V1c, V2, V6, V6b, V8, V9, V10.

## the finding this packet fixes

V10 measured it (section 7.18, measurement 4): **the whole demo ran at 200 Hz while every
document declared 50**. `es_env::Env::new` loads the scene with `LoadConfig { rate: None }`, so
the physics keeps the MJCF's `timestep="0.005"`; `Env::step` advances
`schedule.domains().inference.period` simulation ticks, which `BatchDomains::single_env()` —
the only schedule `Collector::run` and `es_eval::runner` ever built — sets to 1. So one recorded
action row was one 5 ms physics step, while `tests/fixtures/visible-learning/deployment.toml`
declares `rate.control = 50` and `rate.inference = 5`, and `SafetyPlane`'s `dt_s` comes from the
Deployment IR. Every dynamic envelope limit was four times looser per step than intended
(sixteen for acceleration), a 16-row chunk spanned 80 ms rather than 320, and the exported
dataset's `fps = 50` described rows 5 ms apart.

## the decision, and the derivation

Open question 18's default **(b)**, adopted: keep the physics, decimate the control.

```
BatchDomains::single_env_at(physics, control)     crates/es-env/src/scheduler.rs

    period = physics / control        exact rational: (pn*cd) / (pd*cn)
    observation.period = inference.period = period
    simulation.period  = 1            it defines the tick (§12.1)
```

`LoadConfig::rate` stays `None` — the scene's own timestep is untouched, because the
demonstrations' contact behaviour and the M6 Go1 track both depend on it. For the demo,
`physics = 200 Hz`, `control = 50 Hz`, `period = 4`: one `Env::step` sets `ctrl` once, advances
four physics ticks and records one row. A scene whose timestep does not divide the control
period is **refused by name** (App. B.5 validation style), never rounded.

`observation.period` shares it deliberately. `DomainRunner::observe_window` reads the state once
*before* the window and then loops over its simulation ticks, so at `observation.period = 1` the
same reading would be pushed into a `TemporalWindow` four times and the same frame rendered four
times. One observation per control step is what the policy actually consumes.

`rate.inference = 5` keeps meaning "re-plan every 10 control steps": that is
`action.execute_chunk` through the `ChunkBuffer`, which V6b established and this packet does not
touch.

**Wired through:** `Collector::run` (`es loop collect`), `es_eval::runner` (`es eval run`, and
therefore `expert_passes_the_evaluation_harness`), and the two V10 tests in `crates/es/tests/cli.rs`
that drive `Env` directly, via a shared `demo_domains` helper. `es video showcase`'s `.estraj`
record is **per control tick** and stays that way — `Collector::run` and `es_eval::runner` both
push one row per control step, after `step_with_policy` returns — so it is now a true 50 Hz
record, and `python/es/encode_video.py --fps 50` plays it in real time instead of at 0.25x.
The dataset's `fps` was already `rate.control`; it is now true.

`TickRate::from_period_secs` moves the timestep-to-ticks conversion into `es-core`, where the
scheduler (which forbids `f64` time, §18.1) can reach it; `rate_from_timestep` in the MuJoCo
backend delegates to it rather than keeping a second copy.

## the second thing the rate fix exposed

At the true 50 Hz **the scripted expert stopped working: 0/8 on `expert_solves_the_pinned_seeds`,
every episode a timeout.** Measured, not inferred (`python` + `mujoco` 3.13 on the oracle server,
replaying the collected rows against the tool site):

* The arm reaches the hover pose correctly (tool 0.0003 m from the latched cube in xy, z 0.065).
* On the descent it **overshoots the grasp pose by about 11 mm** — tool z reaches 0.0033 where
  the target is 0.0146 — drives the jaws into the table, and shoves the cube 59 mm out of reach
  in one control tick.
* The arm then rests on the table, `shoulder_lift` sits 0.018 rad below its held command, and
  the waypoint machine's `pos_tol = 0.01` gate never closes again. Held at the same `ctrl` with
  no contact, plain MuJoCo settles that joint to 4.9e-4 rad, so the 0.018 is the table, not droop.

The cause is a limit the scene cannot deliver: `velocity_max = 3.0` rad/s and
`acceleration_max = 20` rad/s² against a `forcerange` of 2.94 N·m leaves the arm no braking
authority at the bottom of the descent. Until now that could not show, because one control step
was one 5 ms physics step and the arm was velocity-saturated for the whole episode — it could
not track the ramp, so it could not overshoot it either.

`ExpertCfg::pace_to` already existed to pace the expert to the envelope; its factor is now
`PACE = 0.5` instead of 0.9. Swept on the eight pinned seeds at the true rate: **0.9 → 0/8,
0.75 → 2/8, 0.6 → 8/8, 0.5 → 8/8, 0.35 → 8/8**. Half, with margin on both sides. No envelope
limit, no acceptance gate and no Deployment IR value moved — this is a calibration of the
*expert* against the scene's own actuators, and it is the only knob this packet turned beyond
the schedule.

## scope

* **`context`** — `crates/es-core/src/time.rs`, `crates/es-env/src/scheduler.rs`,
  `crates/es-env/src/expert.rs`, `crates/es-data/src/collect.rs`, `crates/es-eval/src/runner.rs`,
  `crates/es-physics-backend/src/mujoco.rs`, `crates/es/tests/cli.rs`,
  `docs/design/visible-learning*.md` section 7.19 and open question 18,
  `docs/packets/M5/V11-control-period*.md`.
* **`forbidden`** — `LoadConfig::rate` (the physics timestep stays the scene's),
  `tests/fixtures/**` (no limit, no gate, no document), `SafetyPlane` and its `validate`
  signature (`INV-12`, `INV-13`), `docs/ARCHITECTURE*.md`.
* **`INV-17`** — no new trait. `single_env_at` is a constructor on an existing plain-data type.

## oracles

1. **Schedule derivation** — `cargo test -p es-env --lib
   scheduler::tests::a_control_period_is_a_whole_number_of_substeps`. 50 Hz control with a
   0.005 s scene gives period 4 (observation too); 0.006 s is refused with both rates named;
   0.02 s gives period 1; a control rate faster than the physics is the same refusal.
2. **V10's measurement flips** — `ES_V10_DATASET=<new> ES_V10_DUMP=<dir> cargo test --release
   -p es --test cli recorded_actions_replay_to_the_same_outcome`, then
   `$ES_PYTHON python/es/grasp_probe.py --dump <dir> --scene <scene> --substeps {4,1}`. The old
   V1c set is 200 Hz data and **cannot** be replayed at 50 Hz, so the dataset is newly collected.
3. **The expert still passes the harness** — `cargo test --release -p es --test cli expert_ --
   --nocapture`, with the measured `envelope_violation_rate` beside V6's 0.48–0.55.
4. **Re-collect, re-train, re-measure** — V1c's collection command, V2's training knobs, V8's
   external-ACT pipeline, compared against V8's 0/16 · 0/16 · 1/16 and V6's 0/16.
5. **Stop rule** — if both policies stay below `success_rate 0.5`, stop and report; do not move
   a second variable in this packet.

## measured

Oracle server (RTX 4090, `~/venvs/es` for `es`, `~/venvs/es-lerobot-cuda` for training),
mujoco 3.13, 2026-09-15. Artifacts under `~/artifacts/plan-v/v11/`.

### 1 — the schedule

Passes. Also `cargo xtask ci` green with **no golden and no fixture hash moved**: the schedule is
runtime state, not canonical encoding, so neither `regenerate_visible_learning_documents` nor
`regenerate_quadruped_documents` had anything to regenerate.

### 2 — the probe flips

| `grasp_probe.py` | `--substeps 4` | `--substeps 1` |
|---|---|---|
| worst cube drift from the `es` replay | **0.0000 mm** | **195.0051 mm** |
| demonstrations that lift the cube clear of the table | **50 / 50** | 0 / 50 |
| lift, median / max | 120.75 / 130.77 mm | 3.85 / 12.10 mm |
| two-jaw contact ticks, median | 90.0 | 0 |
| gripper joint while both jaws hold the cube | 0.0934 rad | — |

Exactly V10's table with the columns exchanged. `recorded_actions_replay_to_the_same_outcome`
on the new 50-episode set: 50 recorded cubes in the bin, `action` reproduces 50 (50 `Success`),
`action_commanded` reproduces 50.

### 3 — the harness, and the envelope in the right unit

| | V6 / V10 (200 Hz) | V11 (50 Hz) |
|---|---|---|
| `expert_passes_the_evaluation_harness` | 8 / 8 | **8 / 8** |
| worst `envelope_violation_rate` | 0.48 – 0.55 | **0.2044** (0.1535 – 0.2044) |
| `expert_solves_the_pinned_seeds` | 8 / 8 | **8 / 8** |
| `the_temporal_ensemble_survives_the_grasp_window` | `Success` | **`Success`** |
| episode length, control steps | ~351 | 225 – 235 |

The violation rate fell by well over half, which is what "the envelope is measured against the
step it bounds" predicts.

### 4 — what a control step is now worth

Both columns measured with the same script over the same 50-episode command
(`--episodes 50 --seed 1`), so the two are comparable to each other rather than to V10's
per-joint figures.

| per demonstration set | V1c (200 Hz) | V11 (50 Hz) |
|---|---|---|
| frames | 18,263 | 9,038 |
| median episode, control steps | 352 | 179 |
| median episode, simulated seconds | 1.76 | **3.58** |
| `max_j abs(action[t] − action[t−1])`, median | 0.01687 rad | **0.03985 rad** |
| `max_j abs(action[t] − qpos[t−1])`, median | 0.26506 rad | **0.06223 rad** |
| 16-row chunk travel, median | 0.18931 rad | **0.47143 rad** |
| `meta/info.json` `fps` / actual row spacing | 50 / 5 ms | **50 / 20 ms** |

The command no longer leads the arm by a quarter of a radian; a chunk is 320 ms and half a
radian of travel instead of 80 ms and a fifth.

### 5 — the two policies

**The two policies: neither learns the task, and the stop rule fires.**

The IR-owned graph, V2's exact knobs on the new 50 Hz set (`es policy lower` →
`train_act.py --batch 8 --lr 1e-4 --seed 0 --device cuda --resident-gpu`, 20,000 optimizer
steps, initial loss 0.0565 → final **0.0132**, 654 s on the RTX 4090; V1c's 200 Hz run reached
0.0180 from 0.0669):

| suite | `success_rate` | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|
| nominal (standalone, run a) | **0 / 16** | 0.2735 | 900 |
| nominal (standalone, run b) | **0 / 16** | 0.2735 | 900 |
| nominal (inside the suite) | 0 / 16 | 0.2706 | 900 |
| light_intensity | 0 / 16 | 0.2922 | 900 |
| light_direction | 0 / 16 | 0.2998 | 900 |
| observation_delay | 0 / 16 | 0.2940 | 900 |
| torque_noise | 0 / 16 | 0.5132 | 900 |
| backlash | 0 / 16 | 0.2003 | 900 |

The two nominal runs agree to the last digit, so the number is the policy's and not the
schedule's. Every episode runs the full 900-step budget. Reading the `.estraj` records of the
first six nominal cells: the arm moves **1.66 – 1.69 rad** on its widest joint and the cube
finishes at **exactly its start pose** in all six, gripper wide open. The policy is not
fumbling the grasp; it never arrives at the cube.

The external ACT, V8's pipeline unchanged on the new export (`es dataset export --lerobot-v3
--drop action_commanded,action_source --state-dim 6` → `lerobot-train --policy.type=act
--steps=100000 --batch_size=8 --seed=0`, 2,033 s → `es policy import-lerobot` →
`es eval run`): **0 / 16 nominal**, `envelope_violation_rate` 0.0523, every episode 900 steps.
The six-suite sweep on the external ACT was **not** spent: the stop rule fires on the nominal
number, which is what V8's 0/16 · 0/16 · 1/16 are, and a perturbation sweep of a policy that
scores zero unperturbed measures nothing.

| | V6 (200 Hz) | V8 (200 Hz) | **V11 (50 Hz)** |
|---|---|---|---|
| IR-owned graph, 20,000 steps, nominal | 0 / 16 | — | **0 / 16** |
| LeRobot ACT, 20,000 / 50,000 / 100,000 steps, nominal | — | 0/16 · 0/16 · 1/16 | **0 / 16** at 100,000 |

**Wall clocks** (observations, not a throughput claim — §12.4): collect 50 episodes with frames
50 s; bake 1 s; lower + train 20,000 steps 654 s; the six-suite evaluation with frames
(96 cells, 86,400 rendered frames, `--jobs 6`) 393 s; `lerobot-train` 100,000 steps 2,033 s;
one 16-cell nominal run 10 – 25 min depending on contention.

**Verdict.** The rate was wrong and is now right: one control step is one control period, the
envelope is measured against the step it bounds (violation rate 0.48–0.55 → 0.2044 on the
expert), a chunk is 320 ms of real motion, and the dataset's `fps` is true. The rate fix also
found what nothing else had: **every demonstration plan V ever collected was produced by an arm
being dragged rather than driven**, because the Deployment IR declares an acceleration the
scene's 2.94 N·m actuators cannot brake against, and at 200 Hz the arm was velocity-saturated
and could not overshoot what it could not track. Fixing that — the expert now paces to half the
envelope — restores 50/50 demonstrations, 8/8 on both expert gates, and a violation rate less
than half of V6's.

**And neither policy learns it anyway: 0/16 and 0/16, against V6's 0/16 and V8's 1/16.** The
stop rule fires. The learning problem was made *coarser* by the fix — 40 mrad per step instead
of 17, half a radian per chunk instead of a fifth — and the result did not move. Whatever is
wrong is not the control rate, not the demonstrations (50/50, replayed and probed), not the
grasp (122 mm lifts, both jaws, half the episode), not the ensemble, and not the model (V8's
own ACT, 100,000 steps). **Do not move a second variable here.** The next packet gets one
hypothesis and one variable — resolution or demonstration count — and V11's numbers are the
baseline it is compared against.


## acceptance

1. **Schedule derivation.** Met — the unit test passes and the refusal names both rates.
2. **The probe flips.** Met — 0.0000 mm at 4 substeps, 195.0051 mm at 1, on a newly collected
   set, with 50/50 lifts and 50/50 replays.
3. **The expert passes the harness.** Met at 8/8, with `envelope_violation_rate` falling from
   V6's 0.48–0.55 to 0.2044 — but **only after a second finding**: the expert's pace had to be
   recalibrated from 0.9 to 0.5 of the envelope, because at the declared limits the arm
   overshoots the grasp pose by 11 mm and swats the cube. No Deployment IR value and no gate
   moved.
4. **Re-collect, re-train, re-measure.** Met, and the answer is negative: **0/16** for the
   IR-owned graph on every suite and **0/16** for the external ACT at 100,000 steps, against
   V6's 0/16 and V8's 1/16.
5. **Stop rule.** Fired. Nothing else moved in this packet.
6. `cargo xtask ci` green; no golden and no fixture hash moved.

## the next packet

One hypothesis, one variable, against V11's baseline: 96x96 is too small, or 50 demonstrations
are too few. V8 held both fixed on purpose and V11 held both fixed again; four of the five
things that could be wrong are now measured and closed.
