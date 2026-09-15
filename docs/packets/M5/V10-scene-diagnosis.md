# M5 V10 — the scene diagnosis

Design note: `docs/design/visible-learning.md` **section 7.18**, and 7.15 (finding 8, the three
suspects), 7.16 (V8), 7.12–7.13 (envelope semantics, `ChunkBuffer` routing, seeding), 7.10 (what
`action` is), section 5 and 5.4 (the expert and its success predicate). Spec: §8.5, §8.6, §9.3,
§9.5, §12.1, §13.2. Depends on V1, V1c, V6, V6b and V8; changes none of them.

## the question this packet exists to answer

**Why can the SO-101 cube-into-bin demo not be learned?** The harness is proven — the scripted
expert passes 8/8 through `es eval run` (V6) and both paths draw the same scene for a seed
(V6b). The model is ruled out — V8 imported LeRobot's own ACT, verified it bitwise, trained it
100,000 steps and scored 0/16, 0/16, 1/16. Finding 8 of section 7.15 named the three things left:

1. the demonstrations do not actually contain a solved task,
2. the expert pushes the cube rather than grasping it,
3. the temporal ensemble destroys the grasp before it reaches the actuator.

V10 measures all three, cheapest first. **No training, no model change, no fixture change, no
hash moves.** Every number below comes from an executable command.

## scope

* **`context`** — `crates/es/tests/cli.rs` (two new tests), `python/es/grasp_probe.py` (new),
  `docs/design/visible-learning*.md` section 7.18 and open question 16,
  `docs/packets/M5/V10-scene-diagnosis*.md`.
* **`forbidden`** — every fixture under `tests/fixtures/`, every crate's `src/`,
  `docs/ARCHITECTURE*.md`, and any retraining. V10 measures; it does not fix.
* **`INV-17`** — no new trait. The replay is a plain loop over `es_env::Env`, not a second
  `PolicyRuntime` impl; the grasp probe is a Python script outside the runtime entirely.
* **`INV-12`** — the replay still routes every action through `SafetyPlane::validate`. Nothing
  is disabled and no envelope number moves.

## the three measurements

### 1. Do the recorded actions reproduce success?

`recorded_actions_replay_to_the_same_outcome` (`crates/es/tests/cli.rs`). Opens a collected
`LeRobotDataset`, and for each episode replays its recorded action rows open-loop from the same
reset, through the same `Env` and the same `SafetyPlane`, scoring by the bin's three-dimensional
interior — the honest predicate `expert_solves_the_pinned_seeds` uses, not the Task IR's x-only
cone leaf (section 5.4). Both action columns are replayed: `action`, the executed `SafeAction`,
and `action_commanded`, the raw pre-plane command (section 7.10).

Two decisions worth stating. **A plain replay loop, not a `PolicyRuntime`** — `INV-17` allows
seven extension points and a recorded-action player is none of them, and `Env::step(ctrl)` takes
the row directly. **One chunk `seq` per run, not per episode** — `SafetyPlane::accept` takes a
chunk only for a `seq` greater than the last it saw and `begin_episode` deliberately does not
reset that (§8.6), so a per-episode counter makes every episode after the first a permanent
underrun. Measured: 1/50 with the counter per episode, 50/50 with it per run.

**Oracle.** `ES_V10_DATASET=<root> cargo test --release -p es --test cli
recorded_actions_replay_to_the_same_outcome -- --nocapture`. Asserts that replaying `action`
puts exactly as many cubes in the bin as the demonstrations themselves recorded. Skips with a
printed reason without `mujoco` or without `ES_V10_DATASET` — a collected dataset is an input,
not a fixture.

**Result (V1c `ds-train`, 50 episodes, seed 1, 18,263 frames).** The demonstrations recorded
50/50 cubes in the bin. Replaying `action` reproduces **50/50** (49 `Success` — one episode's
cube is in the bin and the x-only predicate disagrees, which is section 5.4's ceiling), with the
plane correcting 10,787 ticks by at most `9.004e-4` rad. Replaying `action_commanded` reproduces
**50/50** (50 `Success`), with the plane correcting 11,589 ticks by up to `3.258e-1` rad. **The
data is sound, and suspect 1 is closed.**

### 2. Does the expert grasp or push?

`python/es/grasp_probe.py`, run with `~/venvs/es/bin/python` (mujoco 3.13). Replays the same
action sequences on the fixture scene with plain `mujoco` — no `es` runtime in the loop — and
logs per control tick the contact normal force between each jaw geom and `cube_geom`
(`mj_contactForce`), the cube's height, and the gripper joint's commanded versus measured
position.

Its input is the replay's `ES_V10_DUMP` directory: line 1 of each `ep-NNN.txt` is the reset
`qpos ‖ qvel` and every line after it is `action[0..6] ‖ cube_xyz[0..3]`. Only the `es` side
knows the Task IR's randomization draw, so the reset state has to be handed over rather than
guessed; the cube column is the self-check, and it is what turns `--substeps` from a setting
into a measurement.

**Oracle.** `$ES_PYTHON python/es/grasp_probe.py --dump <dir> --scene
tests/fixtures/mjcf/so101_pick_place.xml`. Prints the table and the verdict line; without
`mujoco` it prints the reason and exits 0.

**Result.** **50/50** demonstrations lift the cube clear of the table (> 20 mm, its
half-height); median lift 122.65 mm, max 127.67 mm. Both jaws are in contact for a median of
161.5 ticks — 47.9 % of an episode — and the gripper joint stalls at 0.0950 rad while they hold
it, which is the geometric closure the scene's 25 mm cube allows. 50/50 end with the cube in the
bin. **The expert grasps, and suspect 2 is closed.**

### 3. Does the temporal ensemble survive the grasp window?

`the_temporal_ensemble_survives_the_grasp_window` (`crates/es/tests/cli.rs`). Drives one
demonstration through `es_env::plane_chunk` exactly as `es loop collect` and `es eval run` do,
and feeds a second `ChunkBuffer` the identical chunks under `ChunkBlendPolicy::HardSwitch`. The
second buffer's row is the raw command — the newest chunk's own row for the tick — so the
difference between the two is the ensemble and nothing else. No second definition of "raw", and
no reimplementation of the blend.

**Oracle.** `cargo test --release -p es --test cli
the_temporal_ensemble_survives_the_grasp_window -- --nocapture`. Asserts the episode ends in
`Success` **and** that every joint's blended command still spans both ends of what the newest
chunk asked for — an equality, not a tolerance, because the extremes are held long enough for
all eight overlapping chunks to agree on them. A `decay` or `CHUNK_SLOTS` that did open the
gripper fails it.

**Result (seed 1, `decay = 0.01`, 8 overlapping chunks of a 16-row horizon).** The episode is
351 ticks and ends in `Success`. Over ticks 118–338 the worst per-joint deviation is 0.35651 rad
and all of it is the gripper, but `blend min == raw min == −0.05000` and
`blend max == raw max == 0.90000` on every joint. The jaw joint measures 0.0678 rad at its
tightest, inside the 0.0950 rad closure measurement 2 reports. **The ensemble is a lag, not a
loss of range, and suspect 3 is closed.**

## what the three measurements found instead

`--substeps 1` keeps the probe's cube on the `es` replay's to **0.0002 mm** over fifty episodes;
`--substeps 4` diverges by **202.98 mm**. So one recorded action row is exactly one MuJoCo step
of the scene's own `timestep="0.005"` — **200 Hz** — while `deployment.toml` declares
`rate.control = 50`. `Env::new` loads with `LoadConfig { rate: None }` and `Env::step` advances
`schedule.domains().inference.period` ticks, which `BatchDomains::single_env()` sets to 1, and
nothing derives either from the Deployment IR.

Consequences, all arithmetic: every dynamic Safety Plane limit is four (acceleration: sixteen)
times looser than the step it bounds, because `dt_s` comes from the Deployment IR; the
per-tick commanded increment a policy has to resolve has a median of 0.00322 rad on an action
space of ±1.7 rad; and a 16-row chunk spans 80 ms rather than 320 ms.

Two further hypotheses were measured and closed in passing. The dataset pairs
`observation.state[t]` with the `action[t]` that *produced* it rather than the row it was
computed from, the opposite of LeRobot's pairing — but at 5 ms the two readings differ by 0.7
milliradians (0.00251 against 0.00322), which is noise. And a chunk's L1 against the trivial
predictor "repeat the joints you can already see" is 0.0982 rad over a whole trajectory against
0.0964 rad inside the grasp window, so the grasp is neither a vanishing fraction of the
objective nor a harder part of it.

## acceptance

* `cargo xtask ci` clean.
* Both tests run on the oracle server and skip with a printed reason without `mujoco`; the
  replay also skips without `ES_V10_DATASET`.
* Section 7.18 carries the three tables, the fourth finding and a verdict; open question 16 is
  added for the human decision.
* No fixture, no crate `src/`, no hash and no golden moved.

## the next packet

**V11 — one control step is one control period.** Derive `BatchDomains::inference.period` from
the Deployment IR's `rate.control` and the scene's physics rate, refuse a scene whose timestep
does not divide the control period, re-collect with V1c's command and retrain with V2's knobs,
and compare against V8's 100,000-step numbers. Open question 16 states the alternative
(`LoadConfig::rate`, which changes the physics rather than the schedule) and why the default is
the schedule.
