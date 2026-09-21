# RL continuation of imported policies (plan S, §13.4, §14.4, §28.11)

Design note for the M8 campaign. Spec sections: §13.4 (the trainer's semantics), §14.4 (the
import), §28.11 (the ladder), §8.3/§8.9 (the nodes and the equivalence tiers), §9.4 (the plane is
on during rollouts), §19.3 (the `training/` slots), §12.4 (nine metrics). Packets:
`docs/packets/M8/S*.md`. Korean sibling: `rl-continuation.ko.md`.

Status legend for every number in this note: **measured** (server, date, path) or
`Target / Status: unverified`.

## 1. The claim, and the four rules

The project's thesis is "a policy designed and trained elsewhere runs here with the same
semantics" (§8, §1.9). M5 proved it for imitation (LeRobot ACT). Plan S proves it for a second
policy family (PPO MLP) and a second learning signal (reward), and goes one step further: the
reproduced policy **keeps training in our simulation**, and the success rate before and after is
one table under one Evaluation IR.

The four rules §28.11 fixes, restated as the decisions this note implements:

1. **PPO is a trainer, not an IR.** The deployed graph is `Normalize(obs) → StateEncoder{Mlp} →
   PolicyHead{Regression, horizon 1, squash} → Normalizer{Inverse}` and nothing else. The value
   head, the log-std, GAE, the optimizer and the entropy coefficient live in
   `python/es/train_ppo.py` and in `training/`. They move `training_hash`, never `learning_hash`.
2. **Rollouts are `es-env`, and the Safety Plane is on.** The trainer steps our `Env` through the
   `es_native.Rollout` binding (S4a) and never calls a simulator of its own. Every sampled action
   goes through `SafetyPlane::validate` before the actuator; the executed action, the plane's
   event bits and the sampled action are all recorded.
3. **The adapter declares; code never guesses.** Joint order, units, position-target versus
   torque, and the observation layout come from a per-robot adapter document. A mismatch is a
   named refusal (`IMP-0xx`). Pickle and orbax are opened only in learning-path Python (INV-16).
4. **No new trait** (INV-17). `Rollout` is a pyclass over existing types; the trainer is a module.

## 2. The sampling model — one decision that shapes S2b, S4a and S4b

`lower_to_torch` emits one `forward(obs) → action` that already includes the head's squash and
the `Normalizer{Inverse}`, so the module's output is the deterministic action **in actuator
units**. The trainer therefore defines its Gaussian **around that output, in those units**:

```
mu   = module(obs)                       # the deployed function, unchanged
a    = mu + exp(log_std) * eps           # log_std: training-only, state-independent, [action_dim]
```

This is rsl_rl's model (a Gaussian in action space, no squash inside the distribution). A
brax-imported policy carries `tanh` inside `mu`; sampling around `tanh(x)` rather than sampling
`x` and squashing is a different distribution from brax's `NormalTanhDistribution`, and that is
accepted: the imported *deterministic* policy is reproduced exactly (S2b), and continuation is a
new training run whose distribution is ours. The importer keeps the source's log-std (brax's
second output half, rsl_rl's `std`) in `import.json` so `train_ppo.py --init-log-std` can start
from it instead of a constant.

The value head is a separate MLP over the concatenation of the Observation IR's output ports,
built and trained only in `train_ppo.py`. It is written to `training/value.safetensors` for
resumption and is never packed into a bundle.

## 3. Latency and chunking in rollouts

PPO acts every control step with horizon 1: no chunk buffer, no declared latency. The rollout
runs at `expected_latency_ms = 0` and the plane sees one row per step. The **evaluation** of the
same policy (S4c) runs under the Deployment IR's declared latency through the ordinary
`es eval run` path, exactly as every other policy is scored; the difference between the two
regimes is the T7 finding again, and it is what makes the evaluation number honest rather than
the trainer's own. Open question 1 below.

## 4. Oracle tiers by source framework (S2b)

| source | what is compared | tier |
|---|---|---|
| rsl_rl (`.pt`), rl_games (`.pth`) | the source's torch actor vs our runtime (`TorchRuntime` over the lowered module), 1,000 random obs, f32 | **bitwise** (§3.5 tier 1) |
| brax / MuJoCo Playground (orbax or `source.npz`) | (a) the importer's numpy→torch re-construction vs our runtime: **bitwise**; (b) JAX's deterministic `tanh(loc)` vs our runtime | (a) bitwise; (b) §8.9 tier 4, max abs error ≤ 1e-5, recorded |

JAX and torch do not share `exp`/`tanh` implementations, so (b) cannot be bitwise by
construction; (a) is what shows the *import* lost nothing, (b) what shows the *frameworks* agree
to the tier the spec allows.

## 5. The reach task shared by S2c and S4b

One task definition, used by the source trainer (brax, S2c) and by our trainer (S4b) and scored
by one Evaluation IR (S4c). It reuses the committed SO-101 scene and its per-episode cube
randomization, so no new randomization mechanism is needed:

- scene: `tests/fixtures/mjcf/so101_pick_place.xml`, 50 Hz control over 200 Hz physics (the demo's
  V11 cadence, `n_substeps = 4`);
- observation (26): `joint_pos[6] ‖ joint_vel[6] ‖ cube_pose[7] ‖ gripper_pose[7]`, each pose
  the body's world-frame `pos[3] ‖ quat[4]` (quaternion xyzw, §3.1; MuJoCo's `xquat` is wxyz
  and the source side reorders), gripper = body `gripper`. The poses go in whole because the
  Observation IR cannot slice a state port (`ChannelSelect` is not lowered) and the Task IR's
  observation capture binds a `GetBodyPose` channel as the demo's `sim_cube_pose` does (7);
  the network learns the subtraction. Revised 2026-09-21 from the 15-dim difference layout.
- action (6): position targets, normalized `[-1, 1]` over each actuator's `ctrlrange`
  (`Normalizer{Inverse, MeanStd}` with `mean = centre`, `std = half-range`);
- reward: `−‖cube_pos − gripper_pos‖` per step, `+1` on success;
- success: distance `< 0.03` m; timeout 200 control steps.

**Revised again 2026-09-21 by S4e:** the observation channels now say which joint *quantity*
they carry. `joint_vel` is `JointState { body = shoulder_pan, dof = 6, quantity = Velocity }`
and `gripper_pose` is `BodyPose(gripper)`, and `es-eval`'s one capture path serves both from
the backend's own `qvel` and `xpos ‖ xquat` (`docs/design/evaluation-execution.md` 2.3). It
names the block's first *joint* rather than the body `base` because `CpuPlan` allocates one
input buffer per source id: two channels on `base` would be handed the same six numbers. The
IRs accept that pair; the lowering cannot serve it yet, and `input_sources` refuses it by name
rather than serving it wrong. Closing that is an `es-compile` packet, not this one.

The Task IR (`tests/fixtures/rl/task-reach.toml`) spells this with existing nodes
(`GetBodyPose`, `Arith`, `Norm`, `Compare`, `Reward`, `Terminate`). **Found by S4b
(2026-09-21):** `es-env`'s reward/termination cone executed neither `GetBodyPose` nor `Norm`,
and `Expr` had no square root, so the reward was spellable but not executable. Packet **S4d**
extends the cone (`Source::Xpos` lanes, lane-wise `Arith`, `Norm{L2}` → `Expr::Sqrt`, an IEEE
basic operation and not a `DET-010` transcendental — §6.6) and owns the four reach documents;
S4b lands the trainer against the committed demo documents and its reach oracle waits on S4d. If the brax env has to
deviate (MJX support for `implicitfast`/`elliptic`/`condim 6`), the deviation is a named row in
`docs/api-notes/brax-ppo-so101.md` and is part of the sim-to-sim gap the S4c table measures.

## 6. What `training/` gains

- `init.lock` (S1): source `policy_hash`, `learning_hash`, `copied`, `initialised`,
  `shape_mismatch` lists. Absent when `[init]` is absent, so existing recipes keep their hash.
- `config.json` carries the `[rl]` table verbatim (S4b); `dataset.lock` reads `unset` for an RL
  run (§28.10 rule 2: real or `unset`, never fabricated).
- `metrics/loss-curve.json` gains per-iteration `return`, `episode_len`,
  `envelope_violation_rate`, `executed_ne_sampled_rate` (S4b); stream 5 carries the policy loss
  so the editor's Live tab draws it unchanged (E7).

## 7. Measured

*(filled by the packets; every row names server, date and path)*

### S4b — the PPO trainer, oracle server (Linux, 16-core CPU), 2026-09-21

Artifacts: `~/artifacts/plan-s/s4b/` (`untrained.esb`, `run.toml`, `run/`).
Interpreter: `~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129, mujoco 3.13.0.
Documents: `tests/fixtures/visible-learning/task.toml` + `tests/fixtures/rl/`
{`observation-state.toml`, `learning-state.toml`, `deployment-rl.toml`}, recipe
`tests/fixtures/rl/training-rl-demo.toml`.

**The route runs, and it is reproducible.** `envs = 8`, `horizon = 64`, 200 iterations,
`seed = 0`, CPU backend:

| | |
|---|---|
| wall clock | **20.8 s** total (`es train`), 18.4 s inside the trainer |
| per iteration | 92 ms (512 rows: 8 envs x 64 control steps) |
| control ticks | 12,800 |
| `identity_hash` | `c07c90aa09b48000…` |
| `training_hash` | `40da99eaa163a3c7…` |
| `dataset.lock` | `{"unset": true}` |

The nine §12.4 metrics as `Rollout.metrics()` gives them (`metrics/env-metrics.json`). A domain
this path never runs is `null`, not a fabricated zero, and there is deliberately no `step/s`:

| metric | value |
|---|---|
| `physics_steps_per_sec` | 45,713 |
| `actions_per_sec` | 11,428 |
| `camera_frames_per_sec` | `null` — no renderer on this path (§4.3) |
| `pixels_per_sec` | `null` — same |
| `observation_gb_per_sec` | `null` — not instrumented by `Env` |
| `policy_inferences_per_sec` | `null` — inference is in the trainer, not in `Env` |
| `p50_end_to_end_latency` | `null` — synchronous rollout, no declared latency (section 3) |
| `p95_end_to_end_latency` | `null` — same |
| `gpu_memory_peak` | `null` — CPU backend |
| `chunk_underrun_rate` | `null` — horizon 1, no chunk buffer |

Everything else is `Target / Status: unverified`.

**The critic learns; the actor has nothing to learn here.** `value_loss` falls 72.0 → 10.5 over
the 200 iterations (iteration 0 / 49 / 99 / 149 / 199: 72.0, 43.6, 29.7, 15.7, 10.5) and
`return` does not move (−42.98 → −46.39, inside the noise of a segment sum). That is the
expected result and not a defect of the trainer: the document under test is the **demo** task,
whose reward is a `Normalize` of the cube's x position, and nothing a 6-DoF arm does in 64
control steps from a random pose moves that cube. The task PPO can actually improve on is the
reach task, and it is S4d's — see below.

**The plane clamps every single tick.** `envelope_violation_rate` and
`executed_ne_sampled_rate` both read **1.00** in every one of the 200 iterations. This is the
finding of the measurement, not a footnote:

* It is structural, not anomalous. The action is a joint **position target**; the plane bounds
  how far that target may move from the measured joint per control tick (velocity 3.0 rad/s,
  acceleration 80 rad/s² at 50 Hz). A Gaussian around an untrained network's output commands
  poses the arm is nowhere near, so every tick is clamped by construction — the same reason the
  demo's own `deployment.toml` header gives for widening this watchdog from 0.05 to 0.9 at V0.
* It is why `deployment-rl.toml` widens `max_frac` 0.9 → 1.0. At 0.9 the watchdog latches the
  fallback within the first seconds of iteration 0, and every later iteration would optimize
  against a held arm rather than against itself. Widening the envelope is the sanctioned move;
  disabling the plane is not (INV-12), and every clamp is still counted and still reported.
* It makes open question 2 concrete rather than hypothetical. **At 1.00 the policy is trained
  entirely on log-probabilities of actions the env never executed.** Whether to train on the
  executed action instead is now a question with a measured number behind it, and it is the
  first thing to ablate once a task with a moving reward exists.

**Oracles.** 1 (`train_rl_dry_run_plan`, the plan golden plus five refusals) passes anywhere; 2
(`train_rl_two_runs_are_bitwise`, `envs = 4`, `horizon = 16`, 3 iterations, twice) and 3
(`train_rl_init_from_import`, iteration 0 equals `[init] policy` tensor for tensor) pass on the
server under `ES_PYTHON`. Two defects they found, both fixed in the trainer rather than papered
over in the test:

* `samples_per_sec` is dropped from `metrics.json` on the way into `training_hash`. It is a
  measurement of the machine; leaving it in made two runs of one recipe produce two
  `training_hash`es, which is exactly what §3.5 tier 1 forbids. It stays in
  `metrics/loss-curve.json` on disk.
* the trainer's summary reports *whether* it wrote a value file and loaded init weights, not
  *where*. An absolute path there is the output directory the caller chose, and it had the same
  effect on the hash. (`train_act.py` has the same latent issue; no test catches it there, and
  fixing it is not this packet's scope.)

**Oracle 4 was deferred to S4d and is measured above, in the S4e row.** The reach task of section 5 needs `GetBodyPose` and `Norm` in a
reward cone, and `es-env`'s `ScalarPlan` lowers neither — it lowers `GetJointState`,
`GetSensor`, `GetTime`, `Arith`, `Compare`, `Normalize`, `Logic` and `Clamp`, every leaf binds a
single scalar, and `es_ir_types::Expr` has no square root by design (its doc cites §6.6
`DET-010`). So −‖cube_pos − gripper_pos‖ has no form to lower *into*, independently of
`es-env`. Packet **S4d** owns that cone extension and the four `*-reach.toml` documents, and the
success-rate row of this table is written there. What is measured above is the infrastructure —
the recipe, the route, the trainer, the plane and the reproducibility — on documents that
already execute.

### S4e — the reach task trained and scored, oracle server (Linux, 16-core CPU), 2026-09-21

Artifacts: `~/artifacts/plan-s/s4e/` (`run-4000/`, `run-10000/`, each with `eval-<mark>/`).
Interpreter: `~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129, mujoco 3.13.0.
Documents: `tests/fixtures/rl/` {`task-reach.toml`, `observation-reach.toml`,
`learning-reach.toml`, `deployment-reach.toml`, `evaluation-reach.toml`}, recipe
`tests/fixtures/rl/training-reach.toml`, bundle `runs/reach-001/untrained.esb`
(`task_hash b5d3b813…`, `observation_hash 4ced8547…`, `learning_hash eb805f18…`,
`deployment_hash 7af05d88…`, `lowering_hash dce8d352…`).

**This is S4b's deferred oracle 4, and the reach task trains.** `envs = 16`, `horizon = 64`,
`seed = 0`, CPU backend; scored by `es eval run` on `evaluation-reach.toml`, 16 held-out
seeds 201–216, `nominal`:

| budget (iterations) | `success_rate` | `episode_length` | rollout `return` | rollout `entropy` |
|---|---|---|---|---|
| 1,000 | 0.0000 | 200.0 | −4.38 | 5.48 |
| 2,000 | 0.1875 | 171.0 | −4.68 | 5.31 |
| 2,500 | 0.2500 | 168.9 | −3.73 | 4.93 |
| 3,000 | 0.4375 | 136.6 | −3.64 | 4.63 |
| **4,000** | **0.5625** | **129.4** | −4.74 | 4.87 |
| 5,000 | 0.5000 | 135.0 | −3.82 | 5.04 |
| 7,500 | 0.3125 | 149.0 | −4.84 | 6.00 |
| 10,000 | 0.3125 | 157.3 | −4.92 | 6.06 |

**The acceptance criterion is not met, and the reason is not the budget.** 0.8 was never
reached; 0.5625 at 4,000 iterations is the peak, and past it the run *decays* — the same
recipe at 10,000 iterations scores 0.3125, barely better than it did at 2,500, and the
rollout entropy climbs back past where it started (5.52 at iteration 1, 4.87 at 4,000, 6.06
at 10,000). A policy that is unlearning while its budget grows is not short of iterations.
The three things to ablate, in the order this table suggests them: the constant learning rate
(`schedule = "constant"` throughout), the entropy coefficient (0.005, which is what the rising
entropy is paid for), and open question 2 below — **every single tick is clamped**, so every
gradient is computed from the log-probability of an action the env never executed.

Two budgets were run because the first one's curve was still climbing at its end: 4,000
iterations (12.6 min wall clock, `training_hash 1933697d…`) and 10,000 (29.3 min,
`training_hash 69665845…`). They are one curve and not two: at the same seed the longer run's
iteration 4,000 reports the same `return` (−4.7397) and `entropy` (4.8692) as the shorter
run's last, so the eight rows above interleave. The committed recipe names 4,000, the
measured peak.

The nine §12.4 metrics as `Rollout.metrics()` gives them (`metrics/env-metrics.json`, the
10,000-iteration run). A domain this path never runs is `null`, not a fabricated zero, and
there is deliberately no `step/s`:

| metric | value |
|---|---|
| `physics_steps_per_sec` | 32,076 |
| `actions_per_sec` | 8,019 |
| `camera_frames_per_sec` | `null` — no renderer on this path (§4.3) |
| `pixels_per_sec` | `null` — same |
| `observation_gb_per_sec` | `null` — not instrumented by `Env` |
| `policy_inferences_per_sec` | `null` — inference is in the trainer, not in `Env` |
| `p50_end_to_end_latency` | `null` — synchronous rollout, no declared latency (section 3) |
| `p95_end_to_end_latency` | `null` — same |
| `gpu_memory_peak` | `null` — CPU backend |
| `chunk_underrun_rate` | `null` — horizon 1, no chunk buffer |

Everything else is `Target / Status: unverified`. 640,000 control ticks over 1,759.9 s for the
10,000-iteration run; 256,000 over 755.2 s for the 4,000-iteration one.

**The perturbed suites, at the peak checkpoint** (4,000; measurements, not gates, §10.4):

| suite | `success_rate` | `episode_length` |
|---|---|---|
| `nominal` | 0.5625 | 129.4 |
| `observation_delay` (20 ms, 40 ms) | 0.3125 | 170.3 |
| `torque_noise` (5 %) | 0.5000 | 127.6 |
| `backlash` (0–0.01 rad) | 0.5000 | 130.6 |

One control step of delay costs a quarter of the successes; the two actuator suites cost one
episode out of sixteen. That ordering is what a policy reading joint angles and two poses at
50 Hz should feel, and it is the first row in this note that is about the *task* rather than
about the trainer.

**The plane still clamps every single tick.** `envelope_violation_rate` and
`executed_ne_sampled_rate` read 1.00 in every iteration of both runs and in every evaluation
cell — exactly what S4b measured on the demo task, now on a task whose reward moves and whose
policy demonstrably learns. So it is not a symptom of a flat reward: it is what a Gaussian
around a position-target head does against a per-tick velocity and acceleration envelope. Open
question 2 is now the *first* thing to ablate rather than a later one.

**Oracles.** 1 (`committed_task_hashes_are_unmoved_by_joint_quantity`) and 2
(`capture_reads_qvel_and_body_pose`) pass anywhere; 3
(`rollout_observes_the_reach_documents`) passes wherever `ES_PYTHON` has MuJoCo — the 26-wide
port equals the backend's own `qpos[0..6]`, `qvel[0..6]`, cube `qpos[6..13]` and gripper
`xpos ‖ xquat` lane for lane and bit for bit; 4 (`reach_documents_validate`,
`train_reach_dry_run_plan`) pass anywhere. S4a's committed rollout golden
(`tests/golden/rollout/so101_100steps.json`) is unmoved, which is the strongest statement that
no existing channel's capture moved.

## 8. Open questions for a human

1. Should rollouts model the Deployment IR's declared latency (a chunk buffer in the trainer),
   or stay synchronous with the evaluation carrying the honesty? This note chooses synchronous.
2. When the plane clamps a sampled action, the log-probability is that of the *sample*; the env
   saw the *executed* action. The rate at which they differ is reported; whether to train on the
   executed action instead is a later ablation.
3. `Squash::Tanh` on a Regression head: an IR parameter (this note) or a property of the action
   unit the lowering applies (`quadruped-track.md` 3.6 question 2)? This note chooses the
   parameter, with absent = default = today's hash.
