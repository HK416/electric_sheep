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

## 3a. Incremental actions (`ActionSpace::JointDelta`, packet M9/T1)

A Deployment IR may declare `action.space = "joint_delta"`: the policy emits a *change* to the
current joint target rather than the target itself. One function integrates it —
`es_env::chunk_buffer::absolute_target` — and the three consumers that hand the plane a row
(`DomainRunner::emit_actions` for collection, `es_eval::runner::run_episode` for evaluation,
`Rollout::act` for the trainer) call it between the chunk row and `validate`. Four things
follow, and they are the whole rule:

* **The plane is untouched.** It goes on validating an *absolute* joint target, against the
  same envelope, with the same signature (`INV-13`). `tests/fixtures/rl/deployment-reach-delta.toml`
  is `deployment-reach.toml` with one word changed, and its `safety` block is identical.
* **The increment is added to what was executed**, never to the raw row: `prev` is
  `SafetyPlane::last_safe_action()`, read after `observe_state` and before `validate`. At the
  first tick of an episode that value *is* the measured pose the plane seeded the command
  chain with (§9.3), so the integrator resets at every episode boundary without owning a
  second copy of the number. A clamped increment therefore cannot accumulate into a target the
  arm can never reach — which is the failure this rule exists against.
* **The units are increments.** A delta policy's `Normalizer { Inverse }` statistics are rad
  per control tick, so its action port is `Unit::AngularVelocity`; `es_ir::cross` refuses
  `Unit::Angle` there by name (`XIR-031`). The absolute-target unit on a `JointDelta`
  deployment is the one mistake that would make the runtime add a position to a position.
* **Absent is `JointPosition`**, and every committed document's `deployment_hash` is unmoved
  (`cargo test -p es-ir committed_deployment_hashes_are_unmoved_by_joint_delta`). `JointDelta`
  is last in both `ActionSpace` enums because the deployment hash writes the discriminant.

Why an increment is worth having for PPO continuation: the policy's noise lands on the change
rather than on the target, so the per-tick motion the `action_rate` watchdog measures is the
policy's own output and not the distance between where the arm is and where an untrained
network guessed. Nothing in the envelope widens to pay for it — §28.12 rule 1's point.

The cost, named rather than hidden: on the buffered paths the delta arm hands the plane one
row per control tick under a fresh `seq`, because a row the plane has already accepted cannot
be re-integrated against the tick's executed value. A fresh `seq` every tick stamps
`last_chunk_tick` every tick, so `ViolationKind::InferenceDeadline` cannot fire for a delta
policy; a dead policy is still caught, one replan window later, by `ChunkUnderrun` and the
same fallback. Buying the watchdog back needs a plane that can refresh rows without restamping
freshness, and the plane was out of scope for T1 (`INV-11..13`).

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

### S2b — `es policy import-rl`, oracle server `renderer-14` (Linux, 16-core CPU), 2026-09-21

Artifacts: `~/artifacts/plan-s/s2b/` (`brax/`, `rsl-rl/`, `rl-games/`). Source checkpoint:
`~/artifacts/plan-s/s2c/seed0-run1/` (`source.npz`, `meta.json`, `oracle-1000.npz`).
Interpreters: `~/venvs/es-lerobot-cuda/bin/python` (torch 2.11.0+cu129) for the oracle,
`~/venvs/es-rl-import/bin/python` (torch 2.14.0+cpu, numpy 2.5.3, **rsl-rl-lib 5.5.1**,
**rl-games 1.6.5**) for the two framework checkpoints. Documents:
`tests/fixtures/rl/{task-reach,deployment-reach,adapter-so101}.toml`, `task_hash`
`967ea2961d65f931…` — the reach Task IR as S4d committed it, unchanged by this packet.

**The S2c policy imports, and the import loses nothing.** 26 → [256, 256] → 6, swish, `tanh`:

| tier (`section 4`) | what is compared | measured |
|---|---|---|
| (a) | the importer's numpy → torch reconstruction vs our runtime, 1,000 × 6 values | **bitwise** — 0 mismatching values |
| (b) | our runtime vs JAX's deterministic `tanh(loc)` on `obs_scaled` | **9.704e-7** max abs error (tolerance 1e-5) |

| slot | hash |
|---|---|
| `weights_hash` | `37fc82a8811f07523c6de9dfcacf8220884c4e5b60189298568992d26acda6c5` |
| `observation_hash` | `fe391bef976df15ae747922463ad855e8eb1c4447f8d08aeefee8d76a7b7bb73` |
| `learning_hash` | `f5053f12c9227a658aecdf9f911a555616a0ff7761076572c1978434a5dd99d3` |
| `policy_hash` | `1a18cc4bccb49aa48169b4ec8aa2f966411b461d48b28766637bf11ca5b0cd60` |
| `deployment_hash` | `7af05d88891f5fe6e46717c29ccb46b97e917567e1860372f10301eb1c6fafe4` |

Tier (b) is run on `obs_scaled` and not on the packet's `U(−1, 1)` draw, following
`brax-ppo-so101.md` section 6: off-scale inputs saturate every `swish` through the narrow
channels and no correct importer can pass 1e-5 there. The uniform set remains a saturation
probe, not a gate.

The `observation_hash` is **not** `observation-reach.toml`'s. The importer emits
`Normalize{MeanStd}` carrying brax's own running statistics, where the committed document
carries the identity `Range{−1, 1}`; both are observations of the same `task_hash`, which is
what §7.4 means by several Observation IRs sharing one Task IR. It is also the reason an
imported policy cannot simply be scored against a document it was not normalized by.

**`rsl_rl` and `rl_games`, bitwise, on their own checkpoints.** A random-weights actor built by
the framework's own classes and saved the way the framework saves (`import_rl.py --synth
<framework> --native`), 26 → [8, 8] → 6, elu, no squash:

| framework | version | framework's own forward vs the reconstruction | the reconstruction vs our runtime |
|---|---|---|---|
| `rsl_rl` (`MLPModel` under `actor_state_dict`) | 5.5.1 | **bitwise**, max abs 0.0 | **bitwise**, 0 of 6,000 values |
| `rl_games` (`a2c_network` under `model`) | 1.6.5 | **bitwise**, max abs 0.0 | **bitwise**, 0 of 6,000 values |

Two API facts this cost, both now in `import_rl.py`'s docstrings and neither in the packet's
draft: **rsl-rl ≥ 5.0 names the actor's `nn.Sequential` `mlp`, not `actor`** (`MLPModel.mlp`,
`rsl_rl/models/mlp_model.py`; the std moved to `distribution.std_param` /
`distribution.log_std_param`), and `rl_games`' `BaseModel.build` takes **one config dict**, not
keywords (`rl_games/algos_torch/models.py:29`). Both layouts are read; the pre-5.0
`actor.<i>` / top-level `std` shape is still accepted.

**The mapping report earns its keep on the real checkpoint.** `cube_pose` comes back
`severity = warning`: the cube's z has `obs_std` 1.79e-4 against ~1.0 for its neighbours,
because E4 leaves the cube's height untouched and it never moved in training. That is not
brax's 1e-6 variance floor — this run did not reach it — so the check is relative (a component
over 1,000× narrower than the widest) rather than a threshold tuned to one number.

**The CI oracles need no Python.** `import_rl_synthetic_three_frameworks` (es) and
`import_rl_refusals` (es-data) run off three committed fixtures of a few KB each, generated by
`import_rl.py --synth` out of each framework's native layout; the three produce **byte-identical
remapped weights**, which is the only thing the Rust side can say about the three readers
without running them. The `TorchRuntime` open inside the first skips with a printed reason when
`ES_PYTHON` is unset.

### S4c — before and after continuation, oracle server `renderer-14` (Linux, 16-core CPU), 2026-09-21

Artifacts: `~/artifacts/plan-s/s4c/` (`imported/`, `continued-seed{0,1,2}/`, `scratch-seed{0,1,2}/`,
`scratch64-seed{1,2}/`, each with its own `eval/`, plus `logs/` and
`compare-imported-continued-seed0.txt`). Tree `~/Projects/es-s4c` at `a977255` plus this packet's
two recipes, `cargo build --release -p es`, `es_native` rebuilt for it. Interpreter
`~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129, mujoco 3.13.0. Source checkpoint
`~/artifacts/plan-s/s2c/seed0-run1/` (`source.npz` blake3 `8c0faf01…`). Recipes:
`tests/fixtures/rl/training-reach-continued.toml`, `…-scratch.toml`, `training-reach.toml`.

**One Evaluation IR judges every row of this table** — `tests/fixtures/rl/evaluation-reach.toml`,
16 held-out seeds 201–216, `nominal` plus `observation_delay` / `torque_noise` / `backlash`, the
Deployment IR's declared latency applying through `es eval run` (section 3). `evaluation_hash
f15fe888…` is on every report quoted below, and every bundle carries `task_hash b5d3b813…`,
`observation_hash 4ced8547…` and `deployment_hash 7af05d88…`. The rows differ in the policy and in
nothing else, which is the measurement's precondition and not a formality (§13.3).

**The refusal that shaped the table, and S2b wrote its paragraph before it happened.** Re-imported
against main's documents — S2b imported against a pre-S4e Task IR — the S2c policy hashes `task
b5d3b813…`, S4e's and the Evaluation IR's, and `observation 21861ea5…`, its own
`Normalize{MeanStd}` carrying brax's running statistics. `es eval run` refuses that bundle by name,
in 5 ms, before it opens a backend:

```
error: tests/fixtures/rl/evaluation-reach.toml does not judge …/imported/bundle/policy.esb:
ERROR XIR-040  evaluation references a different Task or Observation IR

  evaluation observation reference is 4ced8547, the bundle hashes to 21861ea5

  hint: spec 10.4: equal evaluation_hash means equal conditions
```

That is correct behaviour, and it leaves an imported policy with no row at all: changing the
Evaluation IR is what §13.3 forbids and what this packet was forbidden to do. **The deviation this
table took instead:** brax's input normalizer is folded into the first Dense of the neutral export
before the import — `y = W0·((x − mean)/std) + b0 = (W0/std)·x + (b0 − (W0/std)·mean)`, the same
function of the raw observation — and the import is re-run from a manifest whose `obs_mean` /
`obs_std` are `null`. Three things make that honest rather than convenient:

* the importer's identity-`Range{−1, 1}` Observation IR **is** the committed
  `observation-reach.toml`, bit for bit: the re-import prints `observation_hash 4ced8547…`, the
  document `regenerate_reach_documents` wrote. The two halves agree without being made to;
* the fold is checked, not asserted: on S2c's own 1,000-row oracle (`oracle-1000.npz`) the folded
  network on raw observations and the original on normalized ones differ by **2.575e-5** max abs
  in the squashed action — above tier (b)'s 1e-5 tolerance, for the reason section 4 already
  gives: the cube's z channel has `obs_std` 1.79e-4, so `1/std` is ~5,600 and f32 rounding is
  amplified with it. It is a measurement's rewrite, not an equivalence claim;
* it changes nothing about the amplification itself. `crates/es-data/src/rl_import.rs` warns that
  a `MeanStd` normalizer "amplifies anything off that scale by 1/std"; folded or not, the same
  product reaches the same `tanh` — which is exactly what the continued rows then run into.

**The table.** `success_rate` and `episode_length` are the `nominal` suite's; every training row is
three seeds, mean with min / max; `policy_hash` is the bundle's own (§5.3), not `training.lock`'s
§19.3 `H(training_hash, checkpoint_hash)`, which is a different digest of a different thing:

| row | graph, init | `success_rate` (nominal) | `episode_length` | hashes |
|---|---|---|---|---|
| **source** — brax's own evaluation of the S2c policy, quoted from `brax-ppo-so101.md` 5.2 / 5.5 (64 episodes, **not** this Evaluation IR) | 26 → [256, 256] → 6, swish + `tanh` | **1.00** in the derived MJX scene it trained in; **0.00** on the committed scene | — (final distance 8.4 mm / 173.9 mm) | `source.npz` blake3 `8c0faf01…`; no `policy_hash` and no `execution_hash` — it is not our runtime |
| **imported** — the same policy through `es policy import-rl`, no training | the same graph, the source's weights | **0.0000** | 200.0 (timeout, 16 of 16) | `policy_hash 8aa810a7…`, `weights_hash 210894c0…`, `execution_hash c42adcd4…` |
| **continued** — `[init] = imported`, PPO 4,000 iterations, seeds 0 / 1 / 2 | the same graph, imported init | **0.0000** (0.0000 / 0.0000) | 200.0 | `policy_hash 29cfcd15…` — **one hash for all three seeds**; `training_hash 65ae522a…` / `54a3b46a…` / `7822b3f3…`; `execution_hash d2667368…`, also one for all three |
| **from scratch, same architecture** — the same recipe without `[init]`, seeds 0 / 1 / 2 | the same graph, random init | **0.0833** (0.0000 / **0.2500**) | 189.9 (169.8 / 200.0) | `policy_hash 44223c82…` / `1cf1dbd9…` / `3dd1728d…`; `training_hash 6e366f4c…` / `5705eead…` / `eef2c569…`; `execution_hash 280ac541…` / `1111350d…` / `80102aa5…` |
| **from scratch, S4e's graph** — `training-reach.toml`, seeds 0 / 1 / 2 | 26 → [64, 64] → 6, relu | **0.4167** (0.3125 / **0.5625**) | 143.4 (129.4 / 153.6) | `policy_hash ea84966d…` / `a5321975…` / `7f736adb…`; `training_hash 1933697d…` / `17876c12…` / `d759d683…`; `execution_hash 9ff75635…` / `22b55e10…` / `d72bee1b…` |
| **expert** | — | **omitted** | — | there is no scripted expert for reach — `--expert` drives the demo's pick-and-place demonstrator — so §28.9 rule 1's harness check on this task is S4e's 0.5625 row and not an expert gate |

Per seed, so the spread is read rather than inferred:

| run | seed | `success_rate` | `episode_length` | rollout `return`, first → last | rollout `entropy`, first → last | wall clock |
|---|---|---|---|---|---|---|
| `continued-seed0` | 0 | 0.0000 | 200.0 | −9.56 → −13.84 | 5.53 → **13.55** | 10m49.7s |
| `continued-seed1` | 1 | 0.0000 | 200.0 | −10.07 → −11.51 | 5.52 → **13.57** | 10m50.3s |
| `continued-seed2` | 2 | 0.0000 | 200.0 | −9.87 → −13.34 | 5.51 → **13.59** | 10m52.5s |
| `scratch-seed0` | 0 | 0.0000 | 200.0 | −14.15 → −6.58 | 5.52 → 7.57 | 11m38.9s |
| `scratch-seed1` | 1 | 0.0000 | 200.0 | −13.58 → −10.83 | 5.51 → 7.52 | 11m13.9s |
| `scratch-seed2` | 2 | 0.2500 | 169.8 | −15.24 → −3.77 | 5.51 → 6.07 | 12m10.9s |
| `scratch64-seed0` = S4e's `run-4000` | 0 | 0.5625 | 129.4 | −15.76 → −4.74 | 5.52 → 4.87 | 12.6 min (alone) |
| `scratch64-seed1` | 1 | 0.3750 | 147.2 | −14.72 → −4.23 | 5.51 → 4.76 | 12m1.5s |
| `scratch64-seed2` | 2 | 0.3125 | 153.6 | −12.32 → −3.84 | 5.51 → 4.05 | 12m12.7s |

Every run but S4e's ran **two at a time** on the 16-core box, which inflates its wall clock and
nothing else: the thread count is the trainer's own and the seed is the recipe's. Each evaluation
is one process, 11.3 s. `envelope_violation_rate` and `executed_ne_sampled_rate` read **1.00** in
every iteration of all eight runs and in every evaluation cell, as they did in S4b and in S4e.

**The perturbed suites, mean over the three seeds** (measurements, not gates, §10.4):

| row | `nominal` | `observation_delay` | `torque_noise` | `backlash` |
|---|---|---|---|---|
| imported | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| continued | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| from scratch, same architecture | 0.0833 | 0.0000 | 0.0625 | 0.0625 |
| from scratch, S4e's graph | 0.4167 | 0.1667 | 0.5208 | 0.4583 |

**`es eval compare imported/eval/report.json continued-seed0/eval/report.json`**, verbatim, with
the four `failure_mode_histogram` rows' second cell abbreviated (it repeats the first):

```
SUITE                METRIC                                  A              B          DELTA  SIGNIFICANT
nominal              success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
nominal              envelope_violation_rate          1.000000       1.000000      +0.000000  n/a (aggregate-only report)
nominal              episode_length                 200.000000     200.000000      +0.000000  n/a (aggregate-only report)
nominal              failure_mode_histogram     {"fallback": 16, "timeout": 16, "violation.acceleration": 128, "violation.chunk_underrun": 16, "violation.position": 3184, "violation.velocity": 720} {the same}            n/a  n/a
observation_delay    success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
observation_delay    envelope_violation_rate          1.000000       1.000000      +0.000000  n/a (aggregate-only report)
observation_delay    episode_length                 200.000000     200.000000      +0.000000  n/a (aggregate-only report)
observation_delay    failure_mode_histogram     {the same six counts} {the same}            n/a  n/a
torque_noise         success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
torque_noise         envelope_violation_rate          1.000000       1.000000      +0.000000  n/a (aggregate-only report)
torque_noise         episode_length                 200.000000     200.000000      +0.000000  n/a (aggregate-only report)
torque_noise         failure_mode_histogram     {the same six counts} {the same}            n/a  n/a
backlash             success_rate                     0.000000       0.000000      +0.000000  n/a (aggregate-only report)
backlash             envelope_violation_rate          1.000000       1.000000      +0.000000  n/a (aggregate-only report)
backlash             episode_length                 200.000000     200.000000      +0.000000  n/a (aggregate-only report)
backlash             failure_mode_histogram     {the same six counts} {the same}            n/a  n/a

A: passed=false   B: passed=false
```

Every delta is `+0.000000` and every histogram is identical to the count, because **the two
policies are the same function**. After 4,000 PPO iterations the continued network's six tensors
are bit-identical to the imported ones: `weights/model-1000.safetensors` and
`weights/model-4000.safetensors` differ from `weights/init.safetensors` by **0.0** max abs in every
tensor, in all three seeds, so the three seeds share one `weights_hash 16ba065f…`, one
`policy_hash` and one `execution_hash`. The imported row and the continued rows nevertheless carry
*different* `execution_hash`es, which is correct and worth knowing: the chain hashes the weights
**file** (§5.3, `policy = weights_hash`), and the importer's safetensors and the trainer's carry
the same numbers in a different byte layout. The imported policy also walks the identical
trajectory under `observation_delay` as under `nominal` — `traj/nominal-00.estraj` and
`traj/observation_delay-00.estraj` are the same bytes — so delaying its observation by one and two
control steps changes nothing it does.

**Why nothing moved, measured rather than reasoned.** The gradient that reaches the imported actor
is exactly zero. Stepping the committed reach documents through `es_native.Rollout` on held-out
seed 201 and pushing each observation through the folded network, the **smallest** of the six
pre-squash values over the first 50 control ticks is **23.7** and the largest **340.3**; in f32,
`d tanh/dx` at both of those magnitudes is **exactly 0.0**. So every one of the six outputs is
saturated at every tick, every weight behind the squash gets a zero gradient, and PPO updates the
one parameter that is not behind it: `log_std`, which climbs until the rollout entropy reads 13.55
(from 5.53) while the critic does its ordinary work (`value_loss` 2.47 → 11.77 through a peak of
19.5) against a policy that cannot move. `first_nonfinite_step` is `null`; nothing diverged. It is
not a failure of the trainer — it is what PPO does to a saturated squash.

**What it says.** Continuation did not help, and the finding is that it did not do *anything*: on
this task, at this budget, an imported brax policy is worth less than nothing as a starting point.
It costs a 256 × 256 network that 4,000 iterations do not finish training, and it contributes a
squash that zeroes the gradient. The controls say the rest: the same graph from random init is not
frozen — it learns (`return` −14.15 → −6.58) and one seed in three reaches 0.2500 — and the
committed 64 × 64 relu graph, the smallest of the three, is the best of them at this budget, 0.4167
mean over three seeds, which also places S4e's single measured seed at the **top** of its own
spread (0.5625) rather than in the middle of it. Three things to ablate, in the order this table
suggests them: the `tanh` squash on a continued policy (open question 3 — were `Squash` the
lowering's business rather than an IR parameter, a continuation could drop it); the observation
scale the fold exposes (a policy whose input carries a channel amplified 5,600× because it never
moved in training is a policy E4 would have caught); and open question 2, still measured at 1.00
and still ahead of both in every other row of this note.

### T2 — the delta source policy, oracle server `renderer-14`, 2026-09-21

Artifacts: `~/artifacts/plan-t/t2/seed0-run{1,2}/`. Venv `~/venvs/es-rl` (the §1 pins of
`docs/api-notes/brax-ppo-so101.md`, unchanged). Full detail and the training curve are that
note's section 7; these are the rows plan T is measured on.

The same brax stack, the same derived scene and the same 2 M-step budget, with the action read
as a per-tick increment (`target_t = clip(target_{t−1} + 0.05 · clip(a, −1, 1), ctrlrange)`,
`target_0` = the reset pose):

| row | delta | position (S2c) |
|---|---|---|
| `source.npz` bitwise across two same-seed runs | **yes** (`473b4fde…`) | yes (`8c0faf01…`) |
| `success_reached` over 64 episodes, source framework | **1.00** (64/64) | 1.00 |
| final distance, mean / max | **3.20 / 6.72 mm** | 8.41 / 14.65 mm |
| return, mean | 126.81 | 188.11 |
| `check_export.py`, in-distribution max abs error | 9.537e-07 | 1.580e-06 |
| wall clock, 2 M steps, 4,096 envs | 561.3 s | 540.6 s |

The return is lower and the policy is *better*: a delta action cannot jump to the target, so the
first ~24 ticks of every episode pay the distance penalty while the arm travels. Success and
final accuracy are what the task asks for, and both improved.

**The per-tick command change, the number T3 compares the clamp rate against** —
`|target_t − target_{t−1}|` over 64 × 200 × 6 values:

| | mean | p95 | max |
|---|---|---|---|
| per joint and tick | 0.01006 rad | 0.03219 rad | 0.04991 rad |
| per tick, largest of the six joints | 0.02090 rad | 0.04303 rad | 0.04991 rad |

0.05 rad/tick at 50 Hz is 2.5 rad/s against the envelope's 3.0 rad/s, and the observed maximum
is the cap to four digits: the increment itself never asks for more than the plane allows, so
what T3 will see clamped is the *integrated* target against the position limits, not the step.

**The import, same server and day** (`ES_S2B_SOURCE=~/artifacts/plan-t/t2/seed0-run1`,
`ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129,
`cargo test --release -p es --test cli -- --ignored import_rl_reproduces_the_source_policy`).
`import.json` carries `action_kind = "joint_delta"`, `adapter-so101-delta.toml` declares the
same with the increment unit, and the Task and Deployment IR declare `JointDelta`:

| tier (section 4) | what is compared | measured |
|---|---|---|
| (a) | the importer's numpy → torch reconstruction vs our runtime, 1,000 × 6 values | **bitwise** — 0 mismatching values |
| (b) | our runtime vs JAX's deterministic `tanh(loc)` on `obs_scaled` | **3.297e-7** max abs error (tolerance 1e-5) |

| slot | hash |
|---|---|
| `weights_hash` | `c0199681d71b042332c2211590aef2a3c6a8020e965eb375652ec3dcb453c884` |
| `observation_hash` | `598ad2414fc5c9ffd414bc00658d029326ecde569a84909c0f47e53e34d35cc8` |
| `learning_hash` | `a408abec926306134b3df604ba813770d96bfc2fead3c9829d681d9beb9e6ded` |
| `policy_hash` | `17226acd48abd7288c1306cee492229fab38f874e1de52305e315e6ddb90b42a` |
| `deployment_hash` | `9c81278b496e51cba2aec3852f6543852c084906e2d9d48b934e37a60f0f993d` (`deployment-reach-delta.toml`) |

The emitted Learning IR is the section 1 shape with one difference: the `Normalizer{Inverse}`
statistics are `mean = 0`, `std = 0.05` and its output port is `Unit::AngularVelocity` — rad per
control tick, which `XIR-031` requires of a `JointDelta` deployment and refuses `Unit::Angle`
for. The same oracle on the S2c **position** checkpoint is unmoved (`weights_hash
37fc82a8…`, `policy_hash 1a18cc4b…`, tier (b) 9.704e-7), so the action kind changed what the
numbers mean and nothing else.

There is no committed delta *Task* IR: T1 committed `deployment-reach-delta.toml` and mutates
the task in memory for its own cross-check, so the CLI tests substitute the one line that
differs (`ActionSpec.space`) into a scratch copy of `task-reach.toml`. Committing the pair is
the next packet's to do if it wants one.

### P-M8-R1 — exploration noise, oracle server (Linux, 16-core CPU), 2026-09-21 UTC

Artifacts: `~/artifacts/plan-t/r1/` (`a1-seed0/`, `a2-seed0/`, `a3-seed0/`, `a4-seed0/`,
`a4-seed1/`, `logs/`), each run carrying `metrics/loss-curve.json`, `metrics/env-metrics.json`,
`checkpoints/{4000,10000}.esb`, `eval-4000/` and `eval-10000/`. Tree `~/Projects/es-r1-noise`
(this branch, pushed as a tarball and deleted after the runs), `cargo build --release -p es`,
`es_native` rebuilt for it. Interpreter `~/venvs/es-lerobot-cuda/bin/python`, torch
2.11.0+cu129, mujoco 3.13.0. Recipes `tests/fixtures/rl/noise/{a1-log-std, a2-no-entropy,
a3-cosine, a4-log-std-mid}.toml` — `training-reach.toml` with one field moved per row, seed 0,
`steps = 10000`, `checkpoint_at = [4000, 10000]`, `envs = 16`, `horizon = 64`, CPU backend.
**A0 is quoted from S4e** (`~/artifacts/plan-s/s4e/run-4000/`, `run-10000/`) and was not re-run.
Every checkpoint is scored by `es eval run --config tests/fixtures/rl/evaluation-reach.toml`,
the same `evaluation_hash f15fe888…`, 16 held-out seeds 201–216, and every bundle carries
`task_hash b5d3b813…`, `observation_hash 4ced8547…`, `learning_hash eb805f18…` and
`lowering_hash dce8d352…` — S4e's, unmoved, which is this table's precondition (§13.3).
A1/A2 and A3/A4 each ran **two at a time** on the 16-core box under `nice -n 10`, which
inflates their wall clocks and nothing else; A4 seed 1 ran alone. Each evaluation is one
process, 10.9–11.8 s.

**The table.** Three-value cells are iterations 1 / 4,000 / 10,000 of
`metrics/loss-curve.json`; `success_rate` and `episode_length` are the `nominal` suite's at the
two checkpoints:

| id | `init_log_std` | `entropy` | `schedule` | `executed_ne_sampled_rate` | `envelope_violation_rate` | rollout entropy | rollout `return` | held-out `success_rate` | held-out `episode_length` | wall clock | `training_hash` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A0** (S4e, quoted) | −0.5 | 0.005 | constant | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 5.517 / 4.869 / 6.056 | −15.765 / −4.740 / −4.916 | **0.5625** / 0.3125 | 129.4 / 157.3 | 12m38.0s (4,000) · 29m22.3s (10,000), both alone | `1933697d…` / `69665845…` |
| **A1** | **−2.5** | 0.005 | constant | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −6.481 / −4.918 / −4.680 | −15.573 / −4.752 / −4.440 | 0.0000 / 0.0000 | 200.0 / 200.0 | 34m31.7s | `42632125…` |
| **A2** | −2.5 | **0.0** | constant | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −6.482 / −10.335 / −13.405 | −15.573 / −4.379 / −3.582 | 0.0000 / 0.0000 | 200.0 / 200.0 | 33m29.7s | `bbfd1464…` |
| **A3** | −2.5 | 0.0 | **`warmup_cosine`** (100, 3e-6) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −6.486 / −9.651 / −11.219 | −15.573 / −4.545 / −4.033 | 0.0000 / 0.0000 | 200.0 / 200.0 | 28m2.6s | `74f7d6a3…` |
| **A4** | **−1.5** | 0.0 | as A3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −0.486 / −6.195 / −8.591 | −15.622 / −4.268 / −3.512 | 0.3125 / **0.3750** | 155.3 / 137.9 | 31m10.5s | `e9556a2e…` |
| **A4, seed 1** | −1.5 | 0.0 | as A3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | −0.486 / −6.634 / −10.071 | −14.705 / −3.728 / −5.475 | **0.0000** / **0.0000** | 200.0 / 200.0 | 25m28.1s (alone) | `31a2db0b…` |

**The clamp rate does not fall, and that is the measurement.** At the three marks every row
still reads exactly 1.00, so the table above is not rounding anything away. Counted over the
whole run instead — 10,000 iterations × 1,024 rows = 10,240,000 sampled actions — the
*absolute* number of rows the plane let through unchanged is:

| id | σ = exp(`init_log_std`) | rows the plane did not clamp, of 10,240,000 | whole-run mean rate |
|---|---|---|---|
| A0 | 0.607 rad | **0** | 1.0 |
| A1 | 0.082 rad | **8** | 0.99999921875 |
| A2 | 0.082 rad | **181** | 0.99998232421875 |
| A3 | 0.082 rad | **19** | 0.99999814453125 |
| A4 | 0.223 rad | **1** | 0.99999990234375 |
| A4, seed 1 | 0.223 rad | **8** | 0.99999921875 |

Shrinking the Gaussian by a factor of 7.4 bought 8 unclamped ticks in ten million. The reason
is in the shape of the action and not in the size of the noise: the action is an absolute joint
**position target**, and `deployment-reach.toml` bounds how far that target may travel from the
*measured* joint in one 50 Hz tick at `velocity_max = 3.0 rad/s` → 0.06 rad (with
`action_rate.first_diff_max = 0.08` rad the looser of the two). What the plane clamps is
therefore `μ − q`, the distance between the network's commanded pose and where the arm actually
is, and σ only perturbs a quantity that is already outside the bound. **Open question 2 cannot
be answered by turning the noise down** — at σ = 0 the rate would still be ~1.00 — so the
choice left is the one the review names: train on the executed action (the estimator), widen or
reshape the envelope, or make the head emit a delta rather than an absolute target.

The nine §12.4 metrics as `Rollout.metrics()` gives them (`metrics/env-metrics.json`). A domain
this path never runs is `null`, not a fabricated zero, and there is deliberately no `step/s`:

| metric | A0 (S4e, 10,000) | A1 | A2 | A3 | A4 | A4, seed 1 |
|---|---|---|---|---|---|---|
| `physics_steps_per_sec` | 32,076 | 33,591 | 35,093 | 34,225 | 30,156 | 36,479 |
| `actions_per_sec` | 8,019 | 8,398 | 8,773 | 8,556 | 7,539 | 9,120 |
| `camera_frames_per_sec` | `null` — no renderer on this path (§4.3) | `null` | `null` | `null` | `null` | `null` |
| `pixels_per_sec` | `null` — same | `null` | `null` | `null` | `null` | `null` |
| `observation_gb_per_sec` | `null` — not instrumented by `Env` | `null` | `null` | `null` | `null` | `null` |
| `policy_inferences_per_sec` | `null` — inference is in the trainer, not in `Env` | `null` | `null` | `null` | `null` | `null` |
| `p50_end_to_end_latency` | `null` — synchronous rollout, no declared latency (section 3) | `null` | `null` | `null` | `null` | `null` |
| `p95_end_to_end_latency` | `null` — same | `null` | `null` | `null` | `null` | `null` |
| `gpu_memory_peak` | `null` — CPU backend | `null` | `null` | `null` | `null` | `null` |
| `chunk_underrun_rate` | `null` — horizon 1, no chunk buffer | `null` | `null` | `null` | `null` | `null` |

Everything else is `Target / Status: unverified`. 640,000 control ticks per run, over
1,759.9 s (A0), 2,069.0 s (A1), 2,006.9 s (A2), 1,680.0 s (A3), 1,867.9 s (A4) and
1,525.6 s (A4 seed 1) of `Rollout` time; the two-at-a-time rows are slower for that reason and
not for a reason inside the trainer — A4 seed 1, alone on the box, is the fastest row in the
table and is also the row that scores zero.

**The perturbed suites for A4 seed 0, the only run with a success to perturb** (measurements,
not gates, §10.4). Every other row — A1, A2, A3 and A4 at seed 1 — is 0.0000 in all four
suites at both marks:

| A4 | `nominal` | `observation_delay` | `torque_noise` | `backlash` |
|---|---|---|---|---|
| 4,000, seed 0 | 0.3125 | 0.0625 | 0.4375 | 0.2500 |
| 10,000, seed 0 | 0.3750 | 0.0000 | 0.1875 | 0.2500 |
| 10,000, seed 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**One sentence per variable.**

* **`init_log_std` is the only knob that moved the success rate at all, and it did not move
  the clamp rate.** A3 → A4 changes it and nothing else, −2.5 → −1.5, and the held-out
  `success_rate` at 10,000 goes 0.0000 → 0.3750 while the clamp rate stays at 1.00 in both;
  A1 → A0 changes it the other way, −2.5 → −0.5, and buys back the 0.5625 peak that no
  small-σ row comes near. **The seed-1 re-run says how far that reading may be pushed and no
  further:** the same recipe at seed 1 scores 0.0000 at both marks, so A4's 0.3750 is one draw
  from a spread that includes zero, and the only claim the five runs support is the negative
  one — at σ = exp(−2.5) the task is not learned at either budget, at any of the three
  entropy/schedule settings.
* **The entropy coefficient moves the rollout entropy and nothing a person cares about.**
  A1 → A2 zeroes it and the entropy at 10,000 falls from −4.680 to −13.405 — the bonus was
  indeed paying for σ to re-inflate, exactly as the S4e row guessed — but `success_rate` is
  0.0000 on both sides of the change, so on this task the bonus was neither the cause of the
  decay nor a cost worth removing on its own.
* **The learning-rate schedule changes nothing at σ = exp(−2.5) and cannot be credited for
  A4's shape.** A2 → A3 turns the cosine on alone and both score 0.0000 / 0.0000 (it is 5½
  minutes faster, which is scheduling noise from sharing the box, not a property of the
  schedule); A4 seed 0 holds past 4,000 rather than decaying as A0 does (0.3125 → 0.3750
  against 0.5625 → 0.3125), but it differs from A0 in three fields at once **and its seed-1
  twin does not hold anything** (0.0000 → 0.0000), so "the cosine stopped the decay" is *not*
  a claim this table supports.

**The acceptance criterion is still not met, no variant beats A0, and no variant holds past
4,000 in a way a second seed agrees with** — the packet said to write that down if it
happened, and it happened. 0.8 was never approached; A0's 0.5625 at 4,000 remains the best
number on the reach task; A4's 0.3750 at 10,000 is the best of the four new recipes, ties A0's
own decayed tail (0.3125) within one episode of sixteen, and is 0.0000 at seed 1. What the
packet bought is not a better policy, it is the elimination of an explanation: the 1.00 is
structural in the action space, not an artefact of a badly chosen exploration σ, and the next
thing to move is the estimator, the envelope or the head — not the recipe.

## 8. The importer and the adapter

Rule 3 of section 1 says the adapter declares and code never guesses. This is what that comes
to in the two halves of `es policy import-rl` (spec §14.4, packet M8/S2b).

**The split.** `python/es/import_rl.py` is the only place a pickle or an orbax checkpoint is
opened (INV-16): `torch.load` for `rsl_rl`'s `.pt` and `rl_games`' `.pth`,
`brax.training.checkpoint.load` for an orbax directory, S2c's `source.npz` + `meta.json` for the
committed export. What leaves it is framework-neutral and inert — `weights.safetensors` and an
`import.json` manifest — and everything after that is Rust with no Python on the path.

**The neutral form, and where the network is cut.** `mlp.<i>.weight|bias` `[out, in]` for each
*hidden* Dense, then `head.weight|bias` for the output Dense. All three frameworks activate
every hidden layer and leave the output Dense linear, so `import.json` reads
`activate_output = false`. Our graph cuts the same network one layer earlier:
`StateEncoder{Mlp}` ends at the last *hidden* layer — and therefore carries
`activate_output = true`, the S2a parameter, because that layer *is* activated — and
`PolicyHead{Regression}` is the output Dense, with the source's `squash` after it. Same
function, different cut; `crates/es-data/src/rl_import.rs::learning_graph` is where it happens
and says so.

**The adapter document** (`tests/fixtures/rl/adapter-so101.toml`) carries the four things a
checkpoint cannot know about our robot, and `deny_unknown_fields` throughout, because a key
nobody reads is a mapping nobody declared:

| block | says |
|---|---|
| `[robot] name` | which robot this adapter is for |
| `[joints] source_order`, `units` | the framework's action order by *our* actuator names, and that they are radians |
| `[action] kind`, optional `scale` / `offset` | position targets or torques, and `ctrl = offset + scale · a` |
| `[[observation.channels]] source`, `slice`, `channel` | which Task IR `ObservationSpec` channel each contiguous block of the flat observation feeds |

`scale` / `offset` from the adapter override the manifest's; a disagreement is a warning, never
a silent resolution. The resolved pair is written into `mapping-report.json` so the
`--reference` oracle reads the numbers the import actually used instead of deciding again.

**The five refusals.** Each writes nothing and names its code (`crates/es-ir-types/src/codes.rs`
holds severity and title, like every other code in the project):

| code | refused when |
|---|---|
| `IMP-001` | the adapter's joint count ≠ the Task IR's `ActionSpec.dim` (or the manifest's `action_dim`) |
| `IMP-002` | a joint name the scene has no actuator for — the scene is read from the Task IR's own repository-relative `scene.path` |
| `IMP-003` | `[joints] units = "deg"`; a silent degree conversion is exactly the guess rule 3 forbids |
| `IMP-004` | `[action] kind` disagrees with the Task IR's `ActionSpec.space` |
| `IMP-005` | the channel slices do not tile `obs_dim` exactly and in order, or name a channel the `ObservationSpec` does not declare, or one whose width disagrees |

**The output.** `observation.toml` (one `StateInput` per channel → `Concat` →
`Normalize{MeanStd}` from the manifest, or an identity `Range{−1, 1}` when the source carried no
normalizer), `learning.toml` (the graph above → `ActionChunker` → `Normalizer{Inverse, MeanStd}`
with `mean = offset`, `std = scale`, so the module's output is in actuator units — section 2),
`policy.esb` with the weights remapped onto the keys `lower_to_torch` declares, and
`mapping-report.json`, §14.4's Semantic Mapping Report: one row per joint and per observation
channel, each with its source index or range, our name, the unit and a severity.

**`log_std` is `null` for a brax import, and that is not an omission.** brax's second output
half is a *function of the observation* (`std = softplus(x) + 0.001`,
`brax/training/distribution.py:171`), so there is no state-independent value to import; its
rows travel in `import.json` under `source_std` as metadata and nothing reads them. `rsl_rl`'s
`std` and `rl_games`' `sigma` *are* `[action_dim]` parameters, so those two import a real
`log_std`. `python/es/train_ppo.py --init-log-std` is fed only when `log_std` is non-null, and
it takes one scalar: a vector whose entries differ is a human's choice, not the importer's.

## 9. Open questions for a human

1. Should rollouts model the Deployment IR's declared latency (a chunk buffer in the trainer),
   or stay synchronous with the evaluation carrying the honesty? This note chooses synchronous.
2. When the plane clamps a sampled action, the log-probability is that of the *sample*; the env
   saw the *executed* action. The rate at which they differ is reported; whether to train on the
   executed action instead is a later ablation.
3. `Squash::Tanh` on a Regression head: an IR parameter (this note) or a property of the action
   unit the lowering applies (`quadruped-track.md` 3.6 question 2)? This note chooses the
   parameter, with absent = default = today's hash.
