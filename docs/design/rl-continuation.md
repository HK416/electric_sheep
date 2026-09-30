# RL continuation of imported policies (plan S, §13.4, §14.4, §28.11)

Where the code lives: since packet `docs/packets/M10/W3a-es-import-split.md` the importer is
`crates/es-import/src/rl_import.rs` (crate `es-import`, layer 9, spec §4.2); `es-data` re-exports
it as `es_data::rl_import`, so both paths name the same module.

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
resumption and is never packed into a bundle. `[rl] critic = "privileged"` (packet M11/R10)
makes it read every non-image port and `Rollout.qpos` instead, so simulator state a deployment
never has (the cube's free joint) reaches the baseline and no pixel does; absent is
`"observation"`, which serialises like absence. Section 7's R10 row is the measurement.

### 2a. Which action the gradient is computed at (`[rl] estimator`, packet M9/R5)

Sampling says what the policy *proposed*; the plane says what the environment *ran*. When they
differ — on this task, on every single tick — PPO has two actions to choose between and the
choice is the estimator's, not the plane's:

```
a        = mu + exp(log_std) * eps       # the sample
executed = Rollout.act(a)                # what the plane let through to the actuator
```

| `[rl] estimator` | the action in the rollout buffer | the log-probability, stored and in the ratio |
|---|---|---|
| `"sampled"` (default) | `a` | `log N(a; mu, sigma)` |
| `"executed"` | `executed` | `log N(executed; mu, sigma)` |

`"executed"` is the sanctioned reading of "the plane is part of the environment" (§13.4,
`docs/reviews/M9.md` S-7, the owner's option B): the trainer learns from the action that
actually produced the reward, and the envelope stays exactly as the Deployment IR declares it
(INV-11..13 — the alternative, widening the envelope, is a document decision and not this
one). Four things deliberately do **not** move with it: rewards, dones, values and GAE are the
same numbers either way, because they were always the executed action's; `executed_ne_sampled_rate`
keeps comparing the plane's output with the **sample**, because it is a fact about the envelope
and not about the estimator; the deployed policy is `mu` under either, so `learning_hash` cannot
feel the choice; and the default stays `"sampled"`, so every row measured before this packet is
still the row it was. What it is *not* is an importance-sampling correction: the ratio is
between two evaluations of the same Gaussian at the same point, so `"executed"` is PPO on the
distribution the environment actually saw, with the clamp treated as an unmodelled part of the
env rather than as a censoring to be un-biased. Section 7's R5 row is the measurement.

### 2b. Which engine the rollout steps (`[rl] backend`, packet M11/X1)

`[rl] backend = "mjwarp"` makes `es train` append `--backend mjwarp` to the trainer's argv, and
`train_ppo.py` hands it to `es_native.Rollout(…, backend=…)`, which builds its `Env` on
`MjWarpBackend` instead of `MuJoCoCpuBackend` — a closed enum of the two monomorphized envs
inside `Rollout`, not a trait object and not a new trait (INV-17). Absent is `"mujoco-cpu"`, and
spelled out it serialises exactly like absence, so every recipe measured before M11 keeps its
`training_hash`; any other value is in the recipe JSON and the plan lines `training/config.json`
carries, so it is in `training_hash` (plan golden `tests/golden/train/plan-reach-mjwarp.txt`).
`"newton"` and `"physx"` are refused when the recipe is parsed: Newton's adapter declares no
actuators, so its own `load` refuses any scene a policy could act in, and PhysX is M11/I1. The
Safety Plane is the same code on either engine (INV-11..13). MJWarp is tier 2 (§3.5): a run on
it is never bitwise against the CPU backend, and `train_rl_two_runs_are_bitwise` stays a
statement about `mujoco-cpu` only.

**Measured: the reach scene runs on `mjwarp` since the spec 17.2 footnote.**
`so101_pick_place.xml` declares `cone="elliptic"`; MJWarp maps it as a tier 2 row (the owner's
decision after this packet's first measurement). A 20-iteration PPO smoke with
`[rl] backend = "mjwarp"` on the reach recipe ran in 28 s (mean return −15.8 → −6.6); the
numbers and the two adapter fixes it needed are in `evaluation-execution.md` 2.8.
`rollout_backend_mjwarp_steps_the_reach_documents` (ignored; needs `mujoco_warp`) steps the reach
documents on both engines and prints the distance.

### 2c. The rollout renders (packet M11/X3)

`es_native.Rollout` observes images when es-py is built with its `render` feature (the maturin
build in `python/es/pyproject.toml` turns it on). It owns one `es_env::EnvRenderer` per env for
the Observation IR's one `ImageInput`, configured by `es_env::render::sensor_cfg` from the Task
IR channel that declares that sensor — the function `es loop collect --frames` and
`es eval run --frames` use — and hands its `frame` to the same `es_eval::runner::capture` as a
frame source, so there is no second observation implementation. Each env's reset (explicit or
`Env::step`'s own on a done) calls that env's `begin_episode`, so under `seed = "tick"` the
sample keys restart at *that env's* episode (`renderer.md` 12.8); one `observe` is one frame per
env, which is the render index the tick counts. No accumulation, no batching across envs (X3b).
Without the feature an image input is refused by name at `observe`, as before. A state-only
Observation IR builds no renderer and opens no device.

`train_ppo.py` reads every port's shape from `contract.json` and stacks an image port as
`[n_envs, C, H, W]` (`[3, 96, 96]` here, which is the Observation IR's own output layout); the
value MLP reads the ports flattened. `Rollout.metrics()` carries `render_ms_per_frame` and fills
`camera_frames_per_sec` / `pixels_per_sec` of the §12.4 set; `train_ppo.py` writes them into
`env-metrics.json` and prints the render row on stderr, never into `metrics.json`.

The camera-bearing reach documents — `task-`, `observation-`, `learning-` and
`evaluation-reach-vision.toml` — are generated by `regenerate_vision_reach_documents`
(`crates/es-py/tests/vision_reach.rs`) from the reach documents and `task-pt-tick.toml`'s camera:
the reach task plus `rgb_overhead` on `Pt` 16 spp, 3 bounces, `seed = "tick"`; the Learning IR
is the demo's from-scratch `VisionEncoder { ResNet18 }` (512) beside the reach state MLP (64),
`Fusion { Concat }` at 576, and the reach head. `deployment-reach.toml` is used unchanged.

**Measured.**

| claim | RTX 3060 (local) | RTX 4090 (oracle server) |
|---|---|---|
| `Rollout` frame == `es loop collect --frames` frame (in-process collector, env 0, 2 episodes × 4 ticks), `Pt` 16 spp, `seed = "tick"` | bit-identical, 8 of 8 | bit-identical |
| `Rollout` frame == collector-wired twin renderer, 2 envs × 8 ticks, env 1 reset off env 0's phase | bit-identical, 16 of 16 | bit-identical |
| the same oracle with env 1's `begin_episode` removed | fails at env 1's first tick (27,054 of 27,648 bytes) | — |
| state-only rollout (`so101_100steps.json`), with and without `render` | golden reproduced | golden reproduced |
| PPO smoke state-only, `train_ppo.py` before vs after this packet (3 iterations, 4 envs) | checkpoint and value bitwise | — |
| `train_rl_two_runs_are_bitwise` | passes | passes |
| PPO smoke on the vision task, 10 iterations × 2 envs × 16 steps | runs, loss finite (0.51 → 1.37), 41.5 ms/frame | runs, loss finite (0.51 → 1.37), 20.8 ms/frame, 14 s |

`Rollout.render_ms_per_frame`, one env, 96×96, whole frame (re-pose, upload, trace, readback),
release build, two warm-up frames then 16 while the arm moves; three runs each (the 3060 runs
agreed to ±0.4 ms):

| render | RTX 3060 (local) | RTX 4090 (oracle server) |
|---|---|---|
| `Rs` | 2.02–2.42 ms | 2.44–2.45 ms |
| `Pt` 4 spp | 9.30–9.65 ms | 5.72–5.86 ms |
| `Pt` 4 spp + SVGF | 9.72–9.91 ms | 7.32–7.40 ms |
| `Pt` 16 spp | 31.06–31.15 ms | 15.67–15.73 ms |
| `Pt` 16 spp + SVGF | 31.40–31.42 ms | 17.17–17.32 ms |
| `Pt` 64 spp | 117.03–117.15 ms | 54.41–54.54 ms |
| `Pt` 64 spp + SVGF | 117.41–117.42 ms | 56.04–56.17 ms |

(`cargo test -p es-py --release --features render --test vision_reach -- --ignored
rollout_render_cost`.) One env renders one frame after another, so a PPO iteration of
`envs × horizon` rows pays `envs × horizon` frames serially: 16 envs × 64 steps at 16 spp is
~16 s of rendering per iteration on the 4090 before any learning. That is X3b's question.
`Target / Status: unverified` beyond these two cards.

**What 2c skips.** `es train`'s `[rl]` route still refuses a bundle whose Observation IR has an
image input (`crates/es/src/cmd/train.rs`, outside this packet's scope), so the smoke runs
`train_ppo.py` directly on an `es policy lower` module; lifting that refusal is a follow-up.
One image input per observation (the collector's rule too). Batched multi-env rendering (X3b).

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
  the body's world-frame `pos[3] ‖ quat[4]`, gripper = body `gripper`. Quaternion order differs
  between the two (corrected 2026-09-24, M11/I3): `gripper_pose` is a `BodyPose`, served x-first
  (xyzw, §3.1); `cube_pose` is the free joint's raw `qpos`, served **w-first** as MuJoCo stores it
  (`es_eval::runner::Capture::Qpos`). A source trainer must match each. The poses go in whole because the
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
  so the editor's Live tab draws it unchanged (E7). Watched, stream 6 carries `return`,
  `episode_len`, the success fraction, `entropy` and `envelope_violation_rate`, and a finished
  `es train` folder opens as the same plots from `loss-curve.json` (M16/H4,
  `telemetry-protocol.md` section 9.2).

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
* it changes nothing about the amplification itself. `crates/es-import/src/rl_import.rs` warns that
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

### T3 — the increment space beside the absolute one, oracle server (Linux, 16-core CPU), 2026-09-22 UTC

Artifacts: `~/artifacts/plan-t/t3/` (`delta-scratch-seed{0,1,2}/`, `delta-continued-seed{0,1,2}/`,
`import-before/`, `imported/`, `neutral-folded/`, `recipes/`, `logs/`). Documents: the delta trio
`tests/fixtures/rl/{task,observation,evaluation}-reach-delta.toml` (task-reach with
`ActionSpec.space = JointDelta`, the other two following the moved `task_hash`), `learning-reach-delta.toml`
(the 64×64 relu graph with its action port in `AngularVelocity` and `Normalizer{Inverse}` 0 / 0.05 rad
per tick), `deployment-reach-delta.toml` (T1), recipes `training-reach-delta.toml` and
`training-reach-delta-continued.toml` (A0's `[rl]` values with `init_log_std = ln(0.02) = −3.912`, a unit
conversion — σ = 0.02 rad per tick = 0.4 × `delta_scale` — not a new knob). The imported delta policy is
T2's `~/artifacts/plan-t/t2/seed0-run1/` re-imported against this task with brax's normalizer folded
into the first Dense, as S4c did. Written by the orchestrator from the artifacts after the agent
running the packet was cut off by API outages; every number below is read from `report.json` /
`metrics/loss-curve.json` on the server.

**Held-out `success_rate`, `nominal`, 16 seeds (201–216), mean and per seed.** The absolute rows are
quoted (S4e seed 0; S4c `scratch64` seeds 1–2 at 4,000); the absolute 10,000 mark for seeds 1–2 was
aborted on the server and is `Target / Status: unverified`.

| row | at 4,000 | at 10,000 | nominal `envelope_violation_rate` (eval) | `episode_length` at 4,000 |
|---|---|---|---|---|
| absolute A0 (`training-reach.toml`) | **0.4167** (0.5625 / 0.3125 / 0.3750) | 0.3125 (seed 0) | 1.00 | 143.4 |
| delta, from scratch | 0.1042 (0.0 / 0.0 / 0.3125) | 0.0833 (0.0 / 0.25 / 0.0) | 0.880 / 0.995 / 0.993 | 183.8 |
| delta, imported, before training | 0.0 | — | 0.845 | 200.0 |
| delta, `[init]` = the import | 0.0 (0.0 / 0.0 / 0.0) | 0.0 (0.0 / 0.0 / 0.0) | 0.845 | 200.0 |

**Rollout statistics** (`metrics/loss-curve.json`, iterations 1 / 4,000 / 10,000):

| run | `executed_ne_sampled_rate` | `envelope_violation_rate` | entropy | `return` |
|---|---|---|---|---|
| delta-scratch seed 0 | 0.994 / 1.00 / 1.00 | 0.801 / 0.997 / 1.00 | −14.95 / −14.07 / −6.92 | −15.73 / −9.33 / −5.89 |
| delta-scratch seed 1 | 0.992 / 1.00 / 1.00 | 0.771 / 0.998 / 1.00 | −14.96 / −13.51 / −9.26 | −14.31 / −4.76 / −6.92 |
| delta-scratch seed 2 | 0.992 / 1.00 / 1.00 | 0.783 / 0.996 / 1.00 | −14.96 / −14.00 / −9.66 | −11.79 / −6.32 / −5.22 |
| delta-continued seed 0 | 1.00 / 1.00 / 1.00 | 0.969 / 1.00 / 1.00 | −14.95 / −8.81 / −2.85 | −8.38 / −11.12 / −9.46 |
| delta-continued seed 1 | 1.00 / 1.00 / 1.00 | 0.981 / 1.00 / 1.00 | −14.95 / −8.82 / −2.65 | −8.80 / −11.77 / −8.85 |
| delta-continued seed 2 | 1.00 / 1.00 / 1.00 | 0.973 / 1.00 / 1.00 | −14.96 / −7.26 / +1.12 | −8.53 / −17.04 / −10.21 |

**`es eval compare` absolute seed 0 vs delta-scratch seed 0, both at 4,000** (`nominal`): `success_rate`
0.5625 → 0.0 (−0.5625), `envelope_violation_rate` 1.00 → 0.880, `episode_length` 129.4 → 200.0; every
perturbed suite the same direction; both `passed = false`. Failure histograms, `nominal`, at 4,000:

| policy | `violation.position` | `violation.velocity` | `violation.acceleration` | success |
|---|---|---|---|---|
| absolute A0 seed 0 | 2,053 | 2,055 | 1,882 | 9 |
| delta-scratch seed 2 | 2,303 | 0 | 984 | 5 |
| delta import, before training | 2,672 | 0 | 79 | 0 |
| delta-continued seed 0 | 2,672 | 0 | 80 | 0 |

**What it says.** The increment space did what §8.5 promised on the rate bounds — `violation.velocity`
is zero everywhere, the 0.05 rad increments never exceed `velocity_max`·dt — and it did not lower the
clamp rate, because the clamp moved to the **position soft envelope**. The integrator seeds every tick
from the plane's executed target (§8.5, "integrate from the executed value, never the raw row"), so a
policy that pushes into a soft position limit is pushed back and pushes again on the next tick: a
permanent `violation.position`, zero net motion, and a constant action → reward pairing PPO cannot learn
from. From scratch that is 0.10 at 4,000 against the absolute space's 0.42 (three seeds each); the
imported delta policy is M8's S-5 again — trained in the derived scene where the table has no contact,
it drives the gripper into the workspace our envelope forbids — and continuation from it stays at 0.0
while its entropy climbs (−14.9 → −2.8) with nothing learning. **The sentence the M9 review needs:**
the tables give no reason to make the increment §13.4's default action space for RL; the increment
stays what §8.5 made it (an addition the importer needs for real relative-action policies), and the
lever both campaigns point at is the envelope's meaning for a learning policy — the position soft
margin for this task, the executed-action estimator, or an increment integrated over the *measured*
joint — each a Deployment IR / spec decision (INV-12: widen, never disable), not a trainer change.
Wall-clocks: two to three trainings concurrent, ≈ 30 min per 10,000-iteration run; the nine §12.4
metrics as `Rollout.metrics()` reports them are in each run's `metrics/env-metrics.json`, the rest
`Target / Status: unverified`. Not measured here: the first-iteration actor gradient norm (S-13's
detector does not exist yet).

### W0b — A0 re-measured under the platform-stable `scene_hash`, oracle server (Linux, 16-core CPU), 2026-09-22 UTC

Packet `docs/packets/M10/W0b-scene-hash-libm.md`. The MJCF/URDF importers stopped calling the
host's libm for `euler=`, `axisangle=`, `zaxis=` and `rpy` and now go through
`es_math::approx::{sin_cos_f64, acos_f64}`, so `scene_hash` is one number on Windows and Linux
(§3.2's `f64` paragraph). That moves every SO-101 `task_hash`, and the reach documents with it:
`task-reach.toml` `b5d3b813…` → **`43a62f3f…`**, `observation-reach.toml` `4ced8547…` →
**`ecabac79…`**, `evaluation-reach.toml` `f15fe888…` → **`66ef84a5…`**; `learning-reach.toml`
`eb805f18…` and `deployment-reach.toml` `7af05d88…` do not move, because neither reads a scene.
The full old → new table is the packet's note section.

**A0 re-run from scratch under the moved hashes: bit for bit, all three seeds.** An untrained
bundle packed from the four regenerated reach documents, `training-reach.toml` at seeds 0, 1 and
2 (`[run] seed` overridden in server-side copies; nothing else touched), 4,000 iterations each on
the CPU backend, then `es eval run` on the regenerated `evaluation-reach.toml`:

| seed | `weights/model-4000.safetensors` | tensors differing from S4e / S4c | `nominal` `success_rate` | `episode_length` |
|---|---|---|---|---|
| 0 | `d79c5c3a…` | **0 / 8** | 0.5625 | 129.4375 |
| 1 | `935ca7b8…` | **0 / 8** | 0.3750 | 147.1875 |
| 2 | `35fc351a…` | **0 / 8** | 0.3125 | 153.6250 |

Mean **0.4167**, the number section 7's S4e and S4c rows and `visible-learning.md` 7.34 and 7.35
already carry. Every cell of every `report.json` is identical to the committed one — all four
suites, both metrics, `observation_delay` 0.3125 / 0.0625 / 0.1250 and `torque_noise` 0.5000 /
0.4375 / 0.6250 included. **This is the strongest form of §28.13 rule 1 available anywhere in
this repository**: PPO on the CPU backend is bitwise (`train_rl_two_runs_are_bitwise`), so a
hash fix that moved nothing else has to reproduce the checkpoint byte for byte, and it does —
on a tree four milestones after the one S4e and S4c ran on. `visible-learning.md` 7.36 carries
the demo's half, where CUDA ACT training is not bitwise and the claim has to be made one level
up and one level down instead.

**The two hashes that do move, and why each moves.** `evaluation_hash` `f15fe888…` →
`66ef84a5…` is the fix itself, through `task` and `observation`. `execution_hash` moves twice
over: seed 0 `9ff75635…` → `145bc81a…`, seed 1 `22b55e10…` → `7883ef24…`, seed 2 `d72bee1b…` →
`63f91704…` — once for the documents and once because W0a put the policy runtime's intra-op
thread count into `runtime_hash`. `evaluation.lock` now prints it: **`runtime_threads: 8`**,
torch's own default on this box's 8 physical cores, where `--jobs 1`'s `cores/N` cap of 16 does
not bind. S4e's and S4c's locks predate the field and read `None`. Two reports at two thread
counts are two conditions (`evaluation-execution.md` 2.7); here the count is the same and only
its being *recorded* is new.

**Wall clocks and a scheduling note.** Seeds 0 / 1 / 2 trained in **637 s / 618 s / 637 s** and
scored in 11 / 11 / 10 s, run one at a time (stage total 1,925 s). They were first launched
three concurrent, as the packet asked: `es train` does not cap the trainer's thread pool the way
`es eval run --jobs` caps the evaluator's, so three runs put 3 × 8 torch threads on 16 cores and
had written no checkpoint after **60 minutes**, against S4e's 755 s alone. The runs were killed
and re-run sequentially; the thread count is torch's default either way, so this is a scheduling
choice and not a different measurement — which the bit-identical checkpoints above then confirm
rather than assume. Artifacts `~/artifacts/plan-w/w0b/reach/seed{0,1,2}/`, `training_hash`
`2dfbc7c8…` / `6bf728e7…` / `e3d030e8…`. The nine §12.4 metrics are in each run's
`metrics/env-metrics.json`; everything else is `Target / Status: unverified`.

### P-M9-R5 — the executed-action estimator beside A0, oracle server (Linux, 16-core CPU), 2026-09-22 UTC

Artifacts: `~/artifacts/plan-w/r5/` (`executed-seed{0,1,2}/`, `recipes/`, `logs/`, `run.sh`). Recipe:
`tests/fixtures/rl/training-reach-executed.toml` at the R1 budget (`steps = 10000`, `checkpoint_at =
[4000]`, `[run] seed` 0/1/2), the same untrained bundle as A0; scored on `evaluation-reach.toml`
(16 held-out seeds). Written by the orchestrator from `report.json` / `metrics/loss-curve.json` on the
server after the packet's agent had ended.

**Held-out `success_rate`, `nominal`, 16 seeds, mean and per seed.**

| row | at 4,000 | at 10,000 | nominal `envelope_violation_rate` (eval) | `episode_length` |
|---|---|---|---|---|
| absolute A0, `estimator = "sampled"` (quoted from T3) | **0.4167** (0.5625 / 0.3125 / 0.3750) | 0.3125 (seed 0) | 1.00 | 143.4 at 4,000 |
| absolute, `estimator = "executed"` | 0.0 (0.0 / 0.0 / 0.0) | 0.0 (0.0 / 0.0 / 0.0) | 1.00 | 200.0 at both |

Every perturbed suite (`observation_delay`, `torque_noise`, `backlash`) is 0.0 at both marks on
all three seeds as well.

**Rollout statistics** (`metrics/loss-curve.json`, iterations 1 / 4,000 / 10,000):

| run | `executed_ne_sampled_rate` | entropy | `return` |
|---|---|---|---|
| executed seed 0 | 1.00 / 1.00 / 1.00 | 5.50 / 2.38 / 2.37 | −15.76 / −21.04 / −27.29 |
| executed seed 1 | 1.00 / 1.00 / 1.00 | 5.50 / 2.35 / 2.35 | −14.72 / −17.47 / −17.60 |
| executed seed 2 | 1.00 / 1.00 / 1.00 | 5.50 / 1.76 / 1.76 | −12.32 / −7.85 / −11.37 |

**Failure histogram, `nominal`, at 4,000** (A0 seed 0 quoted from T3): A0 `violation.position` 2,053,
`violation.velocity` 2,055, success 9; executed seeds 0/1/2 `violation.position` 2,866 / 3,184 / 2,280,
`violation.velocity` 3,184 on each — every one of the 16 × 199 ticks — success 0.

**What it says.** Option B does not help; it is worse than doing nothing. With the log-probability
taken at the executed action the policy stops reaching: the return falls on two of three seeds, the
entropy drops once and then freezes (seed 1 reads 2.3546 at both 4,000 and 10,000, i.e. `log_std`
no longer moves), and in evaluation every tick violates the velocity bound. The mechanism is the
estimator, not the envelope: the plane's clamp maps a whole half-line of samples onto one boundary
value, so `log N(executed; mu, sigma)` is the density of a point the Gaussian almost never
proposed — the ratio is between two evaluations of that same point, and the gradient pulls `mu`
towards (positive advantage) or away from (negative advantage) the clamp boundary rather than
towards the actions that earned the reward. **The sentence the next decision needs:** the curve
does not hold past 4,000 and does not beat A0 at 10,000 on any seed; `"sampled"` stays the default
and `"executed"` stays in the recipe vocabulary as a measured negative. The remaining options are
the packet's A (a wider position margin, only with a servo-spec reason), C (increments over the
*measured* joint), or an estimator that models the clamp as censoring — the clipped-action policy
gradient (Fujita & Maeda, 2018), the log of the Gaussian's tail mass for a clamped dimension — which
is a trainer change, keeps the envelope and INV-11..13 untouched, and is not this packet's to make.
Wall-clock: the three trainings concurrent with plan W's W1a PT run on the same host (load ≈ 24 on
16 cores), 19,076 s ≈ 5.3 h for all three; the six evaluations 67 s. The nine §12.4 metrics are in
each run's `metrics/env-metrics.json`, the rest `Target / Status: unverified`.

### I3 — an Isaac Lab policy trained on our scene, and A0, on three engines, oracle server (RTX 4090, Ubuntu 26.04.1), 2026-09-23 UTC

Packet `docs/packets/M11/I3-sim-to-sim-measured.md`. Artifacts `~/artifacts/plan-x/i3/` (`usd/`,
`train{0,1,2}/`, `select{0,1,2}.json`, `heldout{0,1,2}.json`, `delay{0,1,2}.json`,
`import/isaac-{0,1,2}/`, `eval/`, `scenes/`, `docs/`, `v1-xyzw/`, the stage scripts and their
`<stage>.{start,end,done,log}` markers). Tree `~/Projects/es-i3` (`git archive`, the I3 files copied
as they changed), `cargo build --release -p es`. Isaac Sim 5.1.0 + Isaac Lab 2.3.2.post1 +
rsl-rl-lib 3.0.1 in `~/venvs/es-isaac`; MuJoCo 3.13.0, mujoco_warp, torch 2.14 CPU in `~/venvs/es`.

**The Isaac task mirrors the Task IR; it was not tuned.** `python/es/rl_source/isaac_so101_reach/`
is a manager-based env cfg (`env_cfg.py`) and an rsl_rl driver (`main.py`). Its robot is
`robot.usd`, written by `main.py build-usd` from the MJCF `scene_to_mjcf` emits (dumped by
`ES_PHYSX_DUMP_MJCF` from a `--backend physx` run) through `physx_ref.import_scene` — the backend's
own import function, split out of `Sim._build_stage` for this, so the stage the policy trains on
and the one `--backend physx` evaluates on carry the same eight fixups. Mirrored, row by row:

| Task IR (`task-reach-last-action.toml`) | Isaac env |
|---|---|
| 50 Hz over the MJCF's 5 ms step, 200 control steps | `decimation 4`, `sim.dt 0.005`, `episode_length_s 4.0` (asserted: `max_episode_length == 200`) |
| `joint_pos`, `joint_vel` | `mdp.joint_pos_rel`, `mdp.joint_vel_rel` × 0.05 (adapter: `offset = "default_pos"`, `scale = 0.05`) |
| `cube_pose` = `JointState { cube, dof 7 }`: the free joint's `qpos`, quaternion **w-first** | `free_joint_qpos`: root frame minus env origin, Isaac's own w-first quaternion |
| `gripper_pose` = `BodyPose(gripper)`: `xpos ‖ xquat`, quaternion **x-first** | `body_pose_xyzw` of link `gripper` |
| `last_action` (`PreviousAction`, `initial` = the rest pose) | `mdp.last_action` (raw, zero at reset); `default_joint_pos` = that pose |
| `JointPosition`, ctrl clamped to `ctrlrange` | `JointPositionActionCfg(scale 0.5, use_default_offset, clip = ctrlrange)` |
| `-1 · dist + 1 · (dist < 0.03)` per control step | the same terms at weight ∓`1/step_dt` (the reward manager multiplies by `dt`) |
| `Terminate Success` / `Timeout` | `DoneTerm(reached)` / `DoneTerm(time_out, time_out=True)` |
| `ResetState` joints 0; cube x U[0.21, 0.27], y U[−0.03, 0.05], z 0.02 | `reset_joints_by_scale(0, 0)`; `reset_root_state_uniform` ±(0.03, 0.04) around (0.24, 0.01, 0.02) |
| kp 998.22, kv 2.731, forcerange 2.94, armature 0.028, joint damping 0.6, frictionloss 0.052 | `ImplicitActuator` stiffness kp, damping kv, `effort_limit_sim` 2.94, armature from the USD; `-0.6·q̇` as an explicit effort every physics step (as `physx_ref.py`); frictionloss dropped (as `physx_ref.py`) |
| envs in one scene | `env_spacing = 0`, inter-env collisions filtered (`physx_ref.py`'s `GridCloner(spacing = 0)`; also forced, `isaac-lab.md` §9) |

Checked, not assumed: at `q = 0` the env's gripper pose is MuJoCo's to 1e-4 m (0.2932, −0.0002,
0.2344; quaternion x-first (0.0172, −0.7069, −0.0172, 0.7069)), and settled at the rest pose to
2e-4 m.
**Not mirrored**, each a difference the Isaac policy meets only in our runtime or only in Isaac:

1. **The Safety Plane** (Deployment IR): velocity 3 rad/s, acceleration 80 rad/s², action rate
   0.08 / 0.04 rad per tick, position soft margins, workspace. Isaac has none; A0 trained under it.
2. **One control tick of action latency** in `es eval run`: the import declares
   `expected_latency_ms = min(budget, period) = 20 ms`, `latency_ticks` = 1 (A0's 2 ms is 1 tick
   too). Isaac trains and scores at 0.
3. **Scene-level PhysX settings.** Isaac Lab's `PhysxCfg` against `physx_ref.py`'s `World`: GPU
   broadphase vs MBP, CCD off vs on, GPU dynamics on vs off, and bounce / friction-offset /
   iteration-count attributes `World` does not author (`isaac-lab.md` §9). Isaac trains on the GPU
   pipeline with 4,096 envs.
4. **Training-side only:** `init_at_random_ep_len` on the first episode (rsl_rl), no observation
   noise (the Task IR declares none), the Task IR's `Normalize{0..1}` on the distance (identical
   below 1 m).

**Training.** rsl_rl PPO with Isaac Lab's own reach runner config (`FrankaReachPPORunnerCfg`, copied
field for field: 24 steps × 4,096 envs, [64, 64] ELU, lr 1e-3 adaptive, no empirical
normalization), 1,500 iterations = 147.5 M env steps per seed. The checkpoint is chosen **on Isaac's
side only**: every 100th checkpoint scored deterministically on 1,024 resets from seed 2000+s, the
best taken; the number reported is a second set of 1,024 resets (seed 1000+s).

| seed | learn wall | rsl_rl success (stochastic) at 100 / 500 / 1,000 / 1,500 | selected | **Isaac held-out** | at 1,000 | at 1,499 |
|---|---|---|---|---|---|---|
| 0 | 2,648 s | 0.170 / 0.948 / 0.969 / 0.970 | `model_700` | **0.975** | 0.968 | 0.978 |
| 1 | 2,198 s | 0.238 / 0.583 / 0.623 / 0.609 | `model_900` | **0.631** | 0.619 | 0.623 |
| 2 | 2,244 s | 0.203 / 0.644 / 0.651 / 0.643 | `model_1000` | **0.686** | 0.686 | 0.648 |

All three plateau (seed 0 by 600 iterations, seeds 1 and 2 by 900); mean **0.764**. Seed 2's
selected checkpoint scored 0.6855 and then 0.6680 on the same resets in one process: Isaac's GPU
pipeline is not reproducible run to run either.

**Import.** The files are rsl-rl-lib 3.0.1's classic shape (`model_state_dict` with `std`,
`actor.{0,2,4}`, `critic.*`; `isaac-lab.md` §9). `import_rl.py --from rsl-rl --activation elu
--isaac-env-cfg params/env.yaml --joint-names …` then `es policy import-rl` with
`adapter-isaac-so101.toml` against `task-reach-last-action.toml` + `deployment-reach.toml`: all
three carry `observation_hash ace0eba4…` (the one `evaluation-reach-last-action.toml` names) and
`learning_hash 800a232c…`; `policy_hash` `2b05615c…` / `79c80b31…` / `36898e15…`. The mapping
report's timing line reads `decimation 4 x sim_dt 0.005 = 0.02 s == the Deployment IR's control
period`, and its six `damping` rows are warnings: 2.731 in the source, 3.331 (kv + joint damping)
in the scene — the explicit passive term is not a drive gain on the Isaac side.

**The table.** `success_rate` on `nominal`, 16 held-out seeds (201–216); Isaac rows are scored by
`evaluation-reach-last-action.toml`, A0 rows (W0b's `4000.esb`) by `evaluation-reach.toml` — the
same seeds, suites and acceptance. `episode_length` in brackets. `envelope_violation_rate` is 1.0
in every cell (every episode clamps at least one tick).

| policy | Isaac side | physx CPU (r1 = r2) | physx GPU (r1 = r2) | mujoco-cpu | mjwarp r1 / r2 |
|---|---|---|---|---|---|
| Isaac 0 | 0.975 | 0.6875 (105.8) | 0.6875 (107.2) | **0.8750** (83.8) | 0.9375 (62.8) / 0.8125 (84.6) |
| Isaac 1 | 0.631 | 0.5000 (125.4) | 0.3750 (146.1) | 0.6250 (125.5) | 0.4375 (141.2) / 0.3125 (152.6) |
| Isaac 2 | 0.686 | 0.3750 (148.3) | 0.6250 (105.0) | 0.1875 (172.9) | 0.3750 (153.6) / 0.2500 (160.3) |
| A0 0 | — | 0.1250 (184.9) | 0.1250 (183.4) | 0.5625 (129.4) | 0.4375 (141.4) / 0.4375 (141.3) |
| A0 1 | — | 0.1875 (176.8) | 0.0000 (200.0) | 0.3750 (147.2) | 0.3750 (146.6) / 0.4375 (137.3) |
| A0 2 | — | 0.2500 (166.1) | 0.1250 (181.2) | 0.3125 (153.6) | 0.1875 (171.4) / 0.1875 (171.4) |
| **mean** Isaac / A0 | 0.764 / — | 0.521 / 0.188 | 0.563 / 0.083 | 0.563 / 0.417 | 0.583 / 0.333 (r1), 0.458 / 0.354 (r2) |

`execution_hash` per cell (first 8 hex digits; the two physx runs of each row printed one report
and one hash — bitwise, as I1 measured; the two mjwarp runs share a hash and not a number, the X1
tier-2 row):

| policy | physx CPU | physx GPU | mujoco-cpu | mjwarp |
|---|---|---|---|---|
| Isaac 0 | `9b7fad37` | `815f8064` | `f9eb7538` | `e52b4ed8` |
| Isaac 1 | `51a0dc8b` | `87cf5708` | `7ad08aa8` | `eba58084` |
| Isaac 2 | `d0b4b213` | `6e61f76d` | `c0e8f22a` | `a44788f7` |
| A0 0 | `37a6bc7a` | `e121afd9` | `08851281` | `7456e37d` |
| A0 1 | `9064ff63` | `409e93f4` | `e9566999` | `03a25132` |
| A0 2 | `2e352bc6` | `888b4ce0` | `808658ce` | `f9a760c0` |

**Attribution** (`nominal` `success_rate`, 16 seeds). On physx (CPU pipeline) the two
approximated rows the PhysX adapter can toggle — `ES_PHYSX_JOINT_DAMPING=none` (no explicit
`-d·q̇`) and `ES_PHYSX_FRICTION_COMBINE=average` — each written into the engine version, so into
the hash. On mujoco-cpu the two rows PhysX drops or moves, removed from MuJoCo instead: scene
copies without `frictionloss`, without `damping`, without both (`scenes/`, sha256 `9f2769ed…`,
`79b5fbaa…`, `752c725b…`). "Wide envelope" is a diagnostic Deployment IR (`docs/`,
`deployment_hash 22473ac6…`): velocity 100 rad/s, acceleration 1e5, action rate 10 rad/tick,
ee velocity 100 m/s; positions, torque and workspace kept, the plane on (INV-12). "Latency 1" is
Isaac's own eval with `--action-delay 1` on the held-out resets.

| policy | physx | physx, no joint damping | physx, friction average | physx, wide envelope | mujoco | mujoco, no frictionloss | mujoco, no damping | mujoco, neither | mujoco, wide envelope | Isaac, latency 0 → 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Isaac 0 | 0.6875 | 0.0000 | 0.6875 | 0.9375 | 0.8750 | **1.0000** | 0.0625 | 0.1250 | 0.9375 | 0.975 → 0.725 |
| Isaac 1 | 0.5000 | 0.0000 | 0.5000 | 0.4375 | 0.6250 | 0.4375 | 0.0625 | 0.1250 | 0.2500 | 0.631 → 0.585 |
| Isaac 2 | 0.3750 | 0.0000 | 0.3750 | 0.4375 | 0.1875 | 0.3125 | 0.0000 | 0.0000 | 0.4375 | 0.686 → 0.543 |
| A0 0 | 0.1250 | 0.0625 | 0.1250 | — | 0.5625 | 0.5000 | 0.1875 | 0.0000 | — | — |
| A0 1 | 0.1875 | 0.1875 | 0.1875 | — | 0.3750 | 0.4375 | 0.3125 | 0.1250 | — | — |
| A0 2 | 0.2500 | 0.0625 | 0.2500 | — | 0.3125 | 0.5625 | 0.2500 | 0.4375 | — | — |

What the rows say, as numbers. **Friction combine explains nothing here, by construction**: every
colliding geom of the scene has μ = 1, one material, and max = average = min = 1 — the column equals
the default cell for cell (and the hash still moves). **Joint damping is load-bearing on both
sides**: without the explicit `-0.6·q̇` every Isaac policy scores 0.0 on physx, and removing
MuJoCo's damping drops every policy on mujoco-cpu too — the one row that is *approximated* (explicit
instead of implicit) is not the gap, the row itself is essential. **frictionloss**, the row PhysX
drops, moves Isaac 0 to 1.0 on MuJoCo and A0 2 from 0.31 to 0.56, and the other four by at most
0.19 either way. **The envelope** moves Isaac 0 from 0.69 to 0.94 on physx and from 0.88 to 0.94
on MuJoCo, Isaac 2 from 0.19 to 0.44 on MuJoCo, Isaac 1 *down* from 0.63 to 0.25 on MuJoCo, and
Isaac 1 and 2 on physx by one episode. None of the rows is "most of the gap" for all six policies:
at 16 episodes, where one episode is 0.0625, the engine gap is policy-specific.

**A first set of Isaac policies was trained on a mis-mirrored channel, and scored 0.0 here.** The
first env cfg served `cube_pose` x-first, as `docs/design/rl-continuation.md` section 5 describes
the poses. The runtime serves the Task IR's `cube_pose` — a `JointState { cube, dof = 7 }` channel
— from the free joint's `qpos`, raw, **w-first** (`es_eval::runner::Capture::Qpos`); only
`gripper_pose`, a `BodyPose`, is x-first. Those policies (`v1-xyzw/`) scored 0.632 / 0.006 / 0.806
on Isaac's side and, for seed 0, 0.0 on physx CPU and 0.0625 on mujoco-cpu in ours — the identity
quaternion lands as `(1, 0, 0, 0)` where the network learned `(0, 0, 0, 1)`. The fix is on the
Isaac side (`free_joint_qpos`); section 5's sentence is right for `gripper_pose` and wrong for
`cube_pose`, and the adapter cannot permute inside a channel, so any source must mirror it.

**Two findings about the runtime, not fixed here.** (1) `es eval run --scene` is not bound to the
Task IR's `scene_hash`: the three MuJoCo scene variants above ran against `task-reach*.toml`
unrefused, and their `execution_hash` is the default scene's (`08851281…` for A0 0 on all four)
— a different scene is not a different condition to the hash chain. The attribution rows are
therefore identified by the scene file's sha256, not by the hash. (2) `task-reach-last-action.toml`
failed `TaskIr::validate` (`TASK-001`, "declared channel last_action has no ObservationSpec node"),
so no `PreviousAction` bundle could be built at all; the rule now skips a `PreviousAction`
channel, which the loop serves and no graph node computes (`crates/es-ir/src/task.rs`, outside this
packet's context; one line, no hash moves).

**The answer.** **No, not what it scores in Isaac Lab — but it holds up across engines better than our
own policy does:** imported cleanly (one `observation_hash`, the timing check passes), the three Isaac
Lab policies average **0.52** on our physx CPU column (0.69 / 0.50 / 0.38) against **0.76** on
Isaac's own held-out resets (0.975 / 0.631 / 0.686), and the one-tick action latency `es eval run`
applies accounts for 0.15 of that on Isaac's side alone (0.76 → 0.62 on the same 1,024 resets with
`--action-delay 1`); across engines they average 0.52 / 0.56 / 0.56 / 0.58–0.46 on physx CPU /
physx GPU / mujoco-cpu / mjwarp, while A0 falls from 0.42 on mujoco-cpu, the engine it trained on,
to 0.19 on physx CPU and 0.08 on physx GPU (0.33–0.35 on mjwarp) — the PhysX-trained policy
transfers to MuJoCo better than the MuJoCo-trained one transfers to PhysX, and the attribution rows
name joint damping as essential on both engines and frictionloss, the row PhysX drops, as the
largest single mover on MuJoCo, policy by policy rather than as one row that explains the gap.

Wall clock: Isaac training 2,648 / 2,198 / 2,244 s of learning on the GPU (one seed at a time,
under the GPU queue lock); the Isaac-side selection and held-out runs about 2.5 min per seed; one
`es eval run` (16 episodes × 4 suites) ≈ 12 s on mujoco-cpu, ≈ 5.5 min on physx CPU, ≈ 7.5 min on
physx GPU, ≈ 3.5 min on mjwarp. Everything else `Target / Status: unverified`.

### X7 — vision RL on the path tracer, randomization on and off, oracle server (Linux, RTX 4090)

Packet `docs/packets/M11/X7-vision-rl-pt.md`, spec 28.14 wave 3. The first run stopped on budget
("Measured, and a budget stop" below); P-M11-R3 moved the learner to CUDA and added the `-pix`
rows; the rerun under the owner decision of 2026-09-24 ("The rerun" at the end of this section)
ran stage 1 and stage 2 on the four `-pix` rows and answers the packet's questions.

**Documents.** `regenerate_x7_documents` (`crates/es/tests/cli.rs`) writes four rows from
`task-reach-vision.toml`, `observation-reach-vision.toml` and `evaluation-reach-vision.toml`:
`pt-dr`, `rs-dr`, `pt`, `rs` (`task-`, `observation-`, `evaluation-reach-vision-<row>.toml`).
`Pt` rows keep X3's sensor (3 bounces, exposure 64, `seed = "tick"`) at stage 1's spp and SVGF;
`Rs` rows carry the default render block. The `-dr` rows add these `Randomization` targets, each
on its own stream `dr.<target>`, every one `Uniform`:

| target | range | from |
|---|---|---|
| `light.radiance` (the `Pt` sun; refused on `Rs`, so `pt-dr` only) | 0.5–2.0 | R2 |
| `light.intensity` | 0.7–1.3 | X5 |
| `light.direction` (yaw, degrees) | −30–30 | X5 |
| `light.color` | 0.7–1.3 | X5 |
| `light.ambient` | 0.5–2.0 | X5 |
| `geom.bin_floor.rgba` | 0.5–1.5 | X5 |
| `camera.overhead.fov` | 0.85–1.15 | X5 |
| `camera.overhead.pose.{x,y,z}` (m) | −0.02–0.02 | X5 |
| `camera.overhead.pose.{roll,pitch,yaw}` (degrees) | −4–4 | X5 |
| `body.cube.mass` | 0.8–1.2 | X4 |
| `geom.cube_geom.friction` | 0.8–1.2 | X4 |
| `actuator.<servo>.gain`, all six servos | 0.9–1.1 | X4 |

The evaluation documents are `evaluation-reach-vision.toml` (seeds 201–216, `nominal`,
`observation_delay`, `torque_noise`, `backlash`) plus the demo's `light_intensity` (0.5–1.5) and
`light_direction` (45°) suites. The recipes `training-reach-vision-<row>.toml` are
`training-reach.toml`'s (16 envs × 64 steps, 4 × 4 epochs × minibatches, 4,000 iterations) on
the row's bundle with `device = "cuda"`.

**Stage 1's choice rule, fixed before it ran.** Each of the six `pt-dr` runs (spp ∈ {4, 8, 16} ×
SVGF {off, on}, 1,000 iterations, seed 0) is scored by its **return gain per wall-clock hour**:
`(final_return − initial_return) / (wall_clock_s / 3600)`, where `initial_return` and
`final_return` are `train_ppo.py`'s own summary means over the first and the last 10 % of the
iterations and `wall_clock_s` is `metrics/env-metrics.json`'s. The row with the largest value is
the training setting of stage 2. If no row gains (every value ≤ 0), no row has shown it learns
at this budget and the cheapest row (the smallest `wall_clock_s`) is chosen. One seed per row:
the choice is a setting, not a claim, and no stage-1 number is reported as a result about
learning (§28.14 rule 7).

**Measured, and a budget stop (2026-09-23/24 UTC, `~/artifacts/plan-x/x7/`).** Interpreter
`~/venvs/es-lerobot-cuda/bin/python` (torch 2.11.0+cu129), `es` and `es_native` built from this
commit with `render`, the CPU physics backend, `Rollout` batched over 16 envs (X3b).

- **`--device cuda` does not run.** Both 5-iteration smokes with the committed recipes stopped
  at the first forward: `RuntimeError: Expected all tensors to be on the same device … mat1 is on
  cpu` (`failed-cuda/smoke-*.log`). `train_ppo.py` moves the actor to the device but keeps the
  observation and every rollout buffer on the CPU; the state-only runs never met this because
  they all ran on `cpu`. The trainer is outside this packet, so every run below overrode
  `device = "cpu"` (the server script's recipe step), which puts the ResNet18 update on the CPU.
- **Smokes, 5 iterations each, `cpu`:** `pt-dr` (16 spp) 28.0 s per iteration,
  `render_ms_per_frame` 5.44; `rs-dr` 23.0 s per iteration, 0.52 ms per frame. Every target —
  render and physics, on both paths — compiled and ran.
- **Stage 1, the one row that ran** (`pt-dr`, 4 spp, SVGF off, seed 0, 1,000 iterations):

| spp | SVGF | wall clock | s / iteration | `render_ms_per_frame` | `initial_return` | `final_return` | gain / h | entropy first → last 100 it. |
|---|---|---|---|---|---|---|---|---|
| 4 | off | 6.50 h | 23.4 | 1.56 | −8.91 | −10.64 | −0.27 | 5.63 → 7.49 |
| 4 | on | not run | | | | | | |
| 8 | off / on | not run | | | | | | |
| 16 | off / on | not run | | | | | | |

  No rollout episode ended in success in 1,000 iterations (every episode that ended ran to the
  200-step timeout), the entropy rose in every 100-iteration block, and `executed_ne_sampled_rate`
  was 1.00, as on every reach run before. The nine §12.4 metrics (`metrics/env-metrics.json`):
  `physics_steps_per_sec` 27,341; `actions_per_sec` 6,835; `camera_frames_per_sec` 639;
  `pixels_per_sec` 5.89e6; `observation_gb_per_sec`, `policy_inferences_per_sec`,
  `p50_end_to_end_latency`, `p95_end_to_end_latency`, `gpu_memory_peak`,
  `chunk_underrun_rate` `null` (not instrumented on this path, as in S4e).
- **The estimate that stopped it.** The render is 1.6 s of the 23.4 s iteration; the rest is the
  CPU learner. Stage 1's other five rows would take ≈ 36 h more (≈ 6.5 h at 4 spp to ≈ 7.8 h at
  16 spp, from the smokes' per-frame render cost). Stage 2 at the recipes' 4,000 iterations is
  12 runs × ≈ 25–26 h ≈ **305 h** before evaluations; at 1,000 iterations it is ≈ 76 h. Both are
  over the 60 GPU-hour limit the packet was run under, so the remaining stage-1 rows were
  skipped and stage 2 was not started. Arithmetic on measured rates, `Target / Status:
  unverified`.
- **A confound in the documents.** The vision rows keep `observation-reach-vision.toml`'s
  26-wide `state` port, which carries the cube's pose (`cube_pose`, 7). The camera therefore
  shows the policy nothing the state does not already give it exactly, so these rows cannot answer
  "learns *from pixels*"; a row without `cube_pose` in the state is needed for that.

The committed `Pt` documents stay at X3's 16 spp, SVGF off (`X7_SPP`, `X7_SVGF` in
`crates/es/tests/cli.rs`); the rerun trains on 4 spp siblings that are new files
(`X7_RERUN_SPP`, `x7_pt_pix_variants`).

**P-M11-R3: the learner on CUDA, and the `-pix` rows (2026-09-24 UTC, `~/artifacts/plan-x/r3/`).**
`train_ppo.py` now puts every tensor that meets the actor or the value net on `--device`; the
noise and order generators stay on the CPU and each draw is moved after it is made. `--device
cpu` writes the bytes it wrote before: the old and the new trainer, same arguments, gave equal
checkpoints (0, 1, 3, final), value file, stdout and loss curve (without `samples_per_sec`) on
the state reach module and on the vision module (Windows, `.venv`), and the `train_rl_*` cli tests
pass. Off the CPU, `use_deterministic_algorithms` runs with `warn_only` (no warning was printed in
these runs) and cuBLAS gets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The trainer prints
`seconds_per_iteration collect … update …` on stderr. Each X7 row has a `-pix` sibling
(`observation-`, `evaluation-reach-vision-<row>-pix.toml`, `learning-reach-vision-pix.toml`,
`training-reach-vision-<row>-pix.toml`): the `state` port without `cube_pose`, 19 wide; the task is
the parent row's.

Server, the committed 16 spp `pt-dr` documents, the recipe's 16 envs × 64 steps, seed 0,
interpreter `~/venvs/es-lerobot-cuda/bin/python`. Render seconds are `render_ms_per_frame` ×
1,024 frames; rollout is `collect` minus render.

| run | device | iterations | s / iteration (wall) | render | rollout | learner (`update`) |
|---|---|---|---|---|---|---|
| `pt-dr` smoke | cuda | 3 | 7.6 | 5.2 | 1.6 | 0.47 |
| `pt-dr-pix` smoke | cuda | 3 | 8.7 | 5.6 | 2.3 | 0.47 |
| `pt-dr` | cuda | 20 | 8.3 | 6.3 | 1.6 | 0.42 |
| `pt-dr` | cpu | 20 | 28.0 | 6.2 | 2.8 | 18.9 |

The learner is 45× faster on the GPU (18.9 → 0.42 s); the iteration is now render-bound. The
20-iteration return curves from one seed (the recipe's per-segment `return`):

- cuda: −11.12, −10.68, −10.41, −10.08, −10.73, −10.77, −10.36, −10.61, −10.58, −10.50, −10.86,
  −10.93, −10.39, −10.38, −10.42, −10.25, −10.66, −10.71, −10.60, −10.58 (mean −10.58)
- cpu: −11.10, −10.50, −10.20, −10.00, −10.73, −10.78, −10.36, −10.62, −10.58, −10.51, −10.86,
  −10.93, −10.39, −10.39, −10.43, −10.26, −10.66, −10.71, −10.60, −10.58 (mean −10.56)

They are not bitwise equal and are not claimed to be (spec 3.5): the largest per-iteration gap
is 0.21 (iteration 3) and from iteration 5 on they agree to within 0.015. The cuda smoke's first
three returns equal the cuda 20-iteration run's. Neither curve shows learning in 20 iterations,
and none was expected. The `-pix` smoke's three returns (−8.91, −16.96, −18.86) are three
iterations and say nothing about learning.

**A new stage-2 estimate** (arithmetic on the rates above, `Target / Status: unverified`). A `Pt`
16 spp run at 8.3 s per iteration is ≈ 9.2 h for 4,000 iterations. An `Rs` row was not run on
cuda; with X7's measured 0.52 ms per frame (0.5 s of render), the same rollout and learner it
is ≈ 2.6 s per iteration, ≈ 2.9 h. The 12 runs (4 rows × 3 seeds) are 6 × 9.2 + 6 × 2.9 ≈
**73 GPU-hours** before evaluations, against ≈ 305 h with the CPU learner. The four `-pix` rows
at 3 seeds cost the same again, ≈ **145 h** for all 24 runs. At stage 1's 4 spp (1.56 ms per
frame) a `Pt` run would be ≈ 3.6 s per iteration, ≈ 4 h, which would put the 12 runs at ≈ 41 h
and all 24 at ≈ 83 h.

**The rerun: the four `-pix` rows on cuda, `Pt` at 4 spp (owner decision 2026-09-24; server,
2026-09-24 10:02 to 2026-09-26 03:54 UTC, `~/artifacts/plan-x/x7b/`).** The rows with `cube_pose`
were not run; the confound note above stands for them. Code from `0d9aedd` (archive in
`~/Projects/es-x7b`), `es` and `es_native` built with `render`, the CPU physics backend,
`--device cuda`, interpreter `~/venvs/es-lerobot-cuda/bin/python`. The `Pt` rows train on new 4 spp
files (`task-reach-vision-pt[-dr]-4spp[-svgf].toml` and their `-pix` observation and evaluation,
`training-reach-vision-pt[-dr]-4spp[-svgf]-pix.toml`); the 16 spp documents and goldens did not
move. Render seconds are `render_ms_per_frame` × 1,024; rollout is `collect` minus render;
learner is `update`. The GPU lock was held for 41.7 h in total (stage 1 1.8 h, stage 2 training
39.4 h, evaluations 0.5 h). A foreign process (`SSR_RENDER_GLTF`, ≈ 1 GB) was on the GPU at every
stage's start (`load.<stage>`); the timing spread between seeds below is not attributed to it or to
anything else.

*Stage 1, SVGF choice.* The rule, written in the packet before it ran: the run with the higher
mean return over the last 100 iterations wins, unless the difference is smaller than the larger of
the two runs' std over those iterations, in which case SVGF off (the cheaper one) wins. `pt-dr-pix`
at 4 spp, seed 0, 1,000 iterations:

| SVGF | wall clock | s / it. | render | rollout | learner | first 100 it. return | last 100 it. return (std) | entropy, last 100 |
|---|---|---|---|---|---|---|---|---|
| off | 0.874 h | 3.15 | 1.30 | 1.43 | 0.415 | −18.79 | −18.86 (1.17) | 6.49 |
| on | 0.927 h | 3.34 | 1.50 | 1.42 | 0.415 | −20.01 | −20.06 (1.02) | 6.65 |

The difference is 1.20, larger than 1.17, so the higher mean wins: **SVGF off**, which is also the
cheaper one. Neither run learns in 1,000 iterations (first and last 100 agree within the std).
The nine metrics, off / on: `physics_steps_per_sec` 37,138 / 36,031; `actions_per_sec` 9,284 /
9,008; `camera_frames_per_sec` 790 / 682; `pixels_per_sec` 7.28e6 / 6.28e6;
`observation_gb_per_sec`, `policy_inferences_per_sec`, `p50_end_to_end_latency`,
`p95_end_to_end_latency`, `gpu_memory_peak`, `chunk_underrun_rate` `null` (not instrumented on this
path). The earlier stage-1 row (16 spp documents' task at 4 spp with `cube_pose`, CPU learner,
23.4 s per iteration) is the table in "Measured, and a budget stop" above.

*Stage 2, training.* 4 rows × seeds 0, 1, 2, 4,000 iterations each, the recipe unchanged, `Pt` at
4 spp with SVGF off. Returns are the per-iteration `return` of `metrics/loss-curve.json`, averaged
over the first and the last 100 iterations; seconds are per iteration.

| row | seed | wall clock | s / it. | render | rollout | learner | first 100 return | last 100 return (std) | entropy, last 100 |
|---|---|---|---|---|---|---|---|---|---|
| `pt-dr-pix` | 0 | 3.65 h | 3.29 | 1.34 | 1.53 | 0.415 | −18.79 | −18.66 (1.19) | 8.25 |
| `pt-dr-pix` | 1 | 4.11 h | 3.70 | 1.72 | 1.57 | 0.414 | −13.54 | −11.44 (1.46) | 12.85 |
| `pt-dr-pix` | 2 | 4.12 h | 3.71 | 1.73 | 1.57 | 0.414 | −9.15 | −8.75 (0.61) | 11.98 |
| `pt-pix` | 0 | 3.41 h | 3.07 | 1.10 | 1.55 | 0.415 | −18.81 | −18.66 (1.19) | 8.23 |
| `pt-pix` | 1 | 3.69 h | 3.32 | 1.36 | 1.55 | 0.415 | −13.95 | −11.53 (1.51) | 13.32 |
| `pt-pix` | 2 | 3.69 h | 3.32 | 1.38 | 1.53 | 0.414 | −9.09 | −7.53 (0.64) | 12.65 |
| `rs-dr-pix` | 0 | 2.76 h | 2.49 | 0.53 | 1.54 | 0.417 | −18.79 | −18.66 (1.19) | 8.28 |
| `rs-dr-pix` | 1 | 2.79 h | 2.51 | 0.56 | 1.53 | 0.417 | −18.40 | −15.92 (1.30) | 12.67 |
| `rs-dr-pix` | 2 | 2.78 h | 2.50 | 0.53 | 1.55 | 0.417 | −9.15 | −8.70 (0.63) | 12.14 |
| `rs-pix` | 0 | 2.77 h | 2.49 | 0.54 | 1.54 | 0.417 | −18.61 | −16.73 (1.07) | 12.56 |
| `rs-pix` | 1 | 2.80 h | 2.52 | 0.54 | 1.57 | 0.417 | −15.01 | −7.77 (0.65) | 10.77 |
| `rs-pix` | 2 | 2.80 h | 2.52 | 0.54 | 1.56 | 0.416 | −9.11 | −8.73 (0.69) | 11.93 |

The return is set by the seed more than by the row: seed 0 starts near −18.8 and seed 2 near −9.1
on all four rows. The entropy rose in every run (≈ 5.5 at iteration 0), `envelope_violation_rate`
and `executed_ne_sampled_rate` were 1.00 in every iteration of every run, as on every reach run
before. The per-iteration returns of `pt-dr-pix` and `rs-dr-pix` at seed 0 (same physics draws,
different renderer) differ by at most 0.27 and by 0.023 on average over the 4,000 iterations. The
nine metrics, per run (`metrics/env-metrics.json`; the other five `null` as in stage 1):

| row | seed | `physics_steps_per_sec` | `actions_per_sec` | `camera_frames_per_sec` | `pixels_per_sec` |
|---|---|---|---|---|---|
| `pt-dr-pix` | 0 / 1 / 2 | 37,548 / 32,275 / 32,753 | 9,387 / 8,069 / 8,188 | 765 / 595 / 593 | 7.05e6 / 5.48e6 / 5.46e6 |
| `pt-pix` | 0 / 1 / 2 | 39,144 / 32,704 / 36,751 | 9,786 / 8,176 / 9,188 | 928 / 755 / 744 | 8.55e6 / 6.96e6 / 6.85e6 |
| `rs-dr-pix` | 0 / 1 / 2 | 37,573 / 40,943 / 32,933 | 9,393 / 10,236 / 8,233 | 1,937 / 1,824 / 1,933 | 1.78e7 / 1.68e7 / 1.78e7 |
| `rs-pix` | 0 / 1 / 2 | 42,788 / 33,360 / 33,559 | 10,697 / 8,340 / 8,390 | 1,894 / 1,893 / 1,885 | 1.75e7 / 1.74e7 / 1.74e7 |

*Evaluation.* Each run's `checkpoints/4000.esb` on its own row's evaluation document
(`evaluation-reach-vision-<row>.toml`, seeds 201–216, `es eval run --jobs 4 --frames`, 153–179 s
each; the frames were deleted after each report). `success_rate` per suite:

| row | seed | `nominal` | `light_intensity` | `light_direction` | `observation_delay` | `torque_noise` | `backlash` |
|---|---|---|---|---|---|---|---|
| `pt-dr-pix` | 0, 1, 2 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 |
| `pt-pix` | 0, 1, 2 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 |
| `rs-dr-pix` | 0, 1, 2 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 |
| `rs-pix` | 0, 1, 2 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00 |

Every held-out episode ran to the 200-step timeout with `envelope_violation_rate` 1.00; the
`nominal` failure histogram of `pt-dr-pix` seed 1, for one, is 16 timeouts, 16 fallbacks, 16 chunk
underruns, 3,184 position, 720 velocity and 112 acceleration violations.

*Cross-render.* All twelve cross evaluations (`Pt`-trained on the matching `Rs` document,
`Rs`-trained on the matching `Pt` document) were refused before any episode ran, exit 1, e.g.
`pt-dr-pix` seed 0 on `evaluation-reach-vision-rs-dr-pix.toml`:

```
error: tests/fixtures/rl/evaluation-reach-vision-rs-dr-pix.toml does not judge …/pt-dr-4spp-pix-s0/checkpoints/4000.esb:
ERROR XIR-040  evaluation references a different Task or Observation IR
  evaluation task reference is d3948e1b, the bundle hashes to eaa0d34b
ERROR XIR-040  evaluation references a different Task or Observation IR
  evaluation observation reference is 79495062, the bundle hashes to 1876c3f7
```

The sensor's render block is part of the Task IR, so a `Pt` and an `Rs` row are two tasks and two
observations to the hash chain, and an evaluation document judges only the bundle whose task and
observation it names (spec 10.4). A cross-render evaluation needs a way to state "the same task
rendered by the other path" as an evaluation condition; that is a design question, not something
this packet works around.

*The packet's questions.*

1. **No:** at 4,000 iterations none of the three `pt-dr-pix` seeds (nor any seed of the other three
   rows) scored a single held-out success on any of the six suites, and the training return did
   not move beyond the spread between seeds.
2. **At 4 spp the path tracer's render is 1.10–1.73 s per iteration against 0.53–0.56 s on `Rs`
   (2.0–3.3×), which makes an iteration 3.07–3.71 s against 2.49–2.52 s and a 4,000-iteration run
   3.41–4.12 h against 2.76–2.80 h (SVGF adds 0.2 s of render per iteration in stage 1).**
3. **Unanswered:** no policy learned, so there is no transfer to compare, and the hash chain
   refuses the cross-render evaluation with XIR-040 because the renderer is part of the task.

### R10 — pixel PPO with a privileged critic, oracle server (Linux, RTX 4090), 2026-09-26 UTC

Packet `docs/packets/M11/P-M11-R10-vision-rl-critic.md`. The packet asked two questions, and it
fixed the stage-1 rule before anything ran.

**Code.** `[rl] critic = "observation" | "privileged"` sits in the training recipe beside
`estimator`. Absent, or spelled out at `"observation"`, it serialises like absence, so X7's
`rs-pix` recipe keeps the `identity_hash` it had before the packet (`7b3e8019…`, pinned in
`crates/es-data/tests/training_critic.rs`). `"privileged"` is in `training/config.json` and moves
the hash, and it reaches `train_ppo.py` as `--critic privileged`, last in the argv. The privileged
value network is the same 64-64 tanh MLP, but its input is every non-image port (a `[dim]` shape
in `contract.json`) plus `Rollout.qpos(env)`, both read at the same state. On the `-pix` rows that
is 19 + `nq` 13 = 32 inputs. The actor is unchanged, and the value network is still never packed.
`[policy] base_model` on the `[rl]` route now reaches `train_ppo.py` as `--init-backbone`, through
`train_act.py`'s own `init_backbone`, and fills `base_model.lock`. Before this packet the recipe
was accepted and the backbone verified, but the plan passed no flag and the lock read `none`.

Oracles, all on Windows with `.venv`:

* `training_critic.rs` passes 3 of 3.
* `train_rl_privileged_critic_is_bitwise_and_reads_state`: 2 iterations × 2 envs × 8 steps on
  the CPU, run twice. The checkpoint and the value file are bitwise equal, and
  `value.net.0.weight` is `[64, 32]`.
* `train_r10_dry_run_plans`: each arm-P plan is X7's plan for the row plus the one flag.
  `plan-reach-vision-rs[-dr]-pix-critic.txt` are new files.
* HEAD's `train_ppo.py` and this one, with the same arguments and the default critic on the
  vision module, write equal checkpoints (0, 3, final), value file, stdout and loss curve (the
  curve without `samples_per_sec`).
* Every `train_rl_*` test passes. `verify-goldens` reports two additions and nothing modified.

**Arm PI stopped before it ran.** Two parts of the pipeline would have to scale the pixels, and
neither does:

* The `-pix` Observation IR ends the image chain with `Dequantize → Normalize { Range 0..1 }`.
* The lowering calls `_frozen_backbone(...)` on that tensor directly (`lower/torch.rs`), with no
  ImageNet mean/std in between.

So an ImageNet ResNet18 would read `[0, 1]` pixels instead of the standardised input its
FrozenBatchNorm statistics were fitted to. Adding that normalization is an IR decision outside
this packet. Therefore no `-imagenet` document and no PI recipe were written.

**Stage 1** (`~/artifacts/plan-x/r10/`, code `82db6db`, `rs-pix`, seed 0, 1,000 iterations,
`--device cuda`, interpreter `~/venvs/es-lerobot-cuda/bin/python`). The baseline is X7's
`rs-pix-s0` curve (`~/artifacts/plan-x/x7b/out/rs-pix-s0/metrics/loss-curve.json`), which was not
rerun. The windows are iterations 900–999, with the population std.

| arm | wall clock | s / it. | render | rollout | learner | mean return 900–999 (std) | margin over baseline | passes? |
|---|---|---|---|---|---|---|---|---|
| baseline (X7 `rs-pix-s0`) | — | 2.49 | 0.54 | 1.54 | 0.417 | −18.623 (0.360) | — | — |
| P (privileged critic, from-scratch encoder) | 0.721 h | 2.60 | 0.56 | 1.62 | 0.414 | −18.620 (0.361) | +0.003 | no (needs > 0.361) |
| PI (privileged critic, frozen ImageNet encoder) | not run | | | | | | | stopped (normalization) |

**No arm passes, so under the rule the packet stops here, and the next packet is state-to-pixel
DAgger (arXiv:2412.13662).** Stage 2 was not started. Had P passed, stage 2 was projected at
18.4 GPU-hours, inside the 30-hour budget. The GPU lock was held for 0.73 h in total: stage 1
took 0.72 h, and the diagnostic below took 17 s.

The run beside X7's, value loss / entropy / return:

| iteration | 0 | 1 | 100 | 500 | 999 | mean 900–999 |
|---|---|---|---|---|---|---|
| X7 `rs-pix-s0` | 0.42 / 5.52 / −8.90 | 7.82 / 5.52 / −16.08 | 2.90 / 5.69 / −18.70 | 2.16 / 6.45 / −18.24 | 35.06 / 7.36 / −18.73 | 13.07 / 7.27 / −18.62 |
| P | 2.07 / 5.52 / −8.90 | 9.90 / 5.52 / −16.07 | 2.88 / 5.70 / −18.70 | 2.16 / 6.46 / −18.24 | 35.03 / 7.38 / −18.74 | 13.06 / 7.29 / −18.62 |

The two runs differ only in what the value network reads. Their per-iteration returns agree to
within 0.036, and to 0.0056 on average, over all 1,000 iterations. Their value losses differ at
the start (0.42 against 2.07 at iteration 0) and agree to within 2.7 % per iteration from
iteration 100 on (0.23 % on average). Entropy rises in both. `envelope_violation_rate` and
`executed_ne_sampled_rate` were 1.00 in every iteration.

The nine metrics (`metrics/env-metrics.json`) were:

| metric | value |
|---|---|
| `physics_steps_per_sec` | 43,642 |
| `actions_per_sec` | 10,911 |
| `camera_frames_per_sec` | 1,840 |
| `pixels_per_sec` | 1.70e7 |
| `observation_gb_per_sec`, `policy_inferences_per_sec`, `p50_end_to_end_latency`, `p95_end_to_end_latency`, `gpu_memory_peak`, `chunk_underrun_rate` | `null` (not instrumented) |

**Why the critic cannot matter here** (`~/artifacts/plan-x/r10/diag/`, `diag.py`). Each actor
drove 16 envs × 64 steps with its deterministic `mu`, from seed 0 on the `rs-pix` documents. A
hook on the head's `Linear` recorded `z`, the value before `tanh`.

| actor | segment return | plane changed the action | mean \|z\| | share of \|z\| > 3 |
|---|---|---|---|---|
| untrained (the lowering's draw, seed 0) | −8.58 | 0.995 | 0.24 | 0.00 |
| P, iteration 1,000 | −18.02 | 1.00 | 20.9 | 1.00 |
| X7 `rs-pix-s0`, iteration 4,000 | −18.02 | 1.00 | 19.1 | 1.00 |
| constant action 0 (no policy) | −14.36 | 0.00 | — | — |
| constant action 0.5 (no policy) | −8.37 | 0.14 | — | — |

Both trained actors have pushed every channel's `tanh` into saturation. At \|z\| > 3, `tanh`
is within 0.5 % of its bound and its gradient is under 1 %. So `mu` sits at a corner of the
action range, the plane clamps it the same way each time, and the two actors produce the same
return.

The environment does respond to the action: two constants that ignore the observation score
−14.4 and −8.4, and the untrained actor scores −8.6. The collapse happens in the first update.
The return falls from −8.9 at iteration 0 to −16.1 at iteration 1 in both runs, before the two
critics could differ much.

This is S4c's failure again (a saturated `tanh` with zero gradient), now on a from-scratch
ResNet18. The value network is downstream of it: a better baseline cannot move an actor whose
output gradient is zero. This is an observation made after the rule was applied, and the rule's
decision stands. It does also say that R10's premise ("the cause is in the recipe, not in the
envelope") holds only in part. The cause is in the actor's first update, not in the critic.

*The packet's questions.*

1. **No.** With a critic that reads `qpos` and the state port, pixel-only PPO on the rasterizer
   followed X7's pixel-critic run to within 0.036 return per iteration for 1,000 iterations
   (−18.620 against −18.623 over iterations 900–999). Both actors had saturated their `tanh` by
   then.
2. **Not measured.** The ImageNet arm stopped because nothing between the `Normalized{0,1}` port
   and a pretrained backbone applies ImageNet's input normalization.

### R11 — the pixel actor's first update, oracle server (Linux, RTX 4090), 2026-09-26/27 UTC

Packet `docs/packets/M11/P-M11-R11-first-update-saturation.md`. Type D: recipes only, no code
changed. Everything ran under `~/artifacts/plan-x/r11/` (`r11.sh`, `sumA.py`, `decideB.py`) with
R10's build: `es` and `es_native` from `82db6db`, the tree `~/Projects/es-r10`, R10's two
untrained bundles, R10's lowered `rs-pix` module and its `diag.py`. Interpreter
`~/venvs/es-lerobot-cuda/bin/python`, `--device cuda`, CPU physics. Every arm is `rs-pix` with
`critic = "privileged"`, and every recipe is R10's `training-reach-vision-rs-pix-critic.toml`
with only the named field changed (`training-reach-vision-rs-pix-critic-{lr1e-4, lr3e-5, lr1e-5,
warmup, clip}.toml`). The server script rewrites `steps`, `checkpoint_at` and the bundle path per
stage, as R10's did. A foreign process (`SSR_RENDER_GLTF`, 994 MiB) was on the GPU at every
stage's start.

**AW's `warmup` is 100, counted in iterations.** `train_ppo.py` evaluates the schedule once per
PPO iteration (`lr_at(iteration, iterations, lr, lr_min, warmup_steps)`), and all 4 epochs × 4
minibatches = 16 Adam steps of that iteration use it. So "100 iterations of optimizer steps" is
100 × 4 × 4 = 1,600 Adam steps, and in the trainer's unit the field is `warmup = 100`.
`warmup = 1600` would have been 1,600 iterations. Two consequences, reported rather than fixed:

* `Schedule::warmup` (`crates/es-data/src/training.rs`) is documented as "optimizer steps". That
  is true on the IR route (`train_act.py`), and on the `[rl]` route it is iterations.
* `es train` refuses `warmup >= steps`, so AW's 20-iteration probe cannot be written. AW's probe
  ran the same recipe at `steps = 101` with `checkpoint_at = [1, 5, 20]` and reads iterations
  1–20. On the ramp the rate is `3e-4 × iteration / 100`, whatever the run's length, and the
  probe's first 100 returns equal stage B's to the last digit.

The ramp starts at 0 (`lr_at(0) = 0`), so AW's first update does not move the actor. Its
checkpoint 1 measures exactly as the untrained actor does.

**Stage A, the probe** (20 iterations, seed 0). `diag.py` drove 16 envs × 64 steps of each
checkpoint's deterministic `mu` from seed 0 and hooked the head's `Linear` for `z`, the value
before `tanh`. "Segment" is that rollout's return. "Train" is the loss curve's `return` at
0-based index 1, 5 and 19, the segment collected by the actor after that many updates (index 20
does not exist in a 20-iteration run). The untrained actor has mean |z| 0.239, share of |z| > 3
0.00 and segment return −8.58.

| arm | change | lr at it. 20 | mean \|z\| 1 / 5 / 20 | share \|z\| > 3, 1 / 5 / 20 | segment return 1 / 5 / 20 | train return 1 / 5 / 19 |
|---|---|---|---|---|---|---|
| A0 | none | 3e-4 | 6.95 / 8.38 / 9.19 | 0.83 / 1.00 / 1.00 | −18.02 / −18.02 / −18.02 | −16.07 / −19.21 / −18.69 |
| A1 | `lr = 1e-4` | 1e-4 | 2.50 / 2.78 / 3.05 | 0.17 / 0.33 / 0.33 | −21.43 / −21.56 / −21.88 | −20.55 / −20.40 / −21.07 |
| A2 | `lr = 3e-5` | 3e-5 | 0.69 / 0.69 / 0.76 | 0.00 / 0.00 / 0.00 | −14.17 / −11.95 / −7.66 | −9.29 / −21.02 / −13.01 |
| A3 | `lr = 1e-5` | 1e-5 | 0.21 / 0.24 / 0.21 | 0.00 / 0.00 / 0.00 | −7.90 / −6.89 / −5.51 | −8.57 / −6.93 / −5.39 |
| AW | warmup-cosine, `warmup = 100` | 6e-5 | 0.24 / 0.22 / 0.28 | 0.00 / 0.00 / 0.00 | −8.58 / −6.99 / −8.94 | −8.85 / −7.21 / −10.10 |
| AG | `grad_clip = 0.5` | 3e-4 | 7.40 / 8.77 / 8.70 | 1.00 / 1.00 / 1.00 | −18.02 / −18.02 / −18.02 | −16.08 / −19.21 / −18.71 |

A0 reproduces R10's collapse. Its first 20 training returns equal R10's `s1-p` run's to the last
digit (the largest difference is 0.0), including −8.90 → −16.07 between index 0 and index 1. Its
actor already has 83 % of |z| beyond 3 after one update, so the stage stands. A1's shares
(1/6, 2/6) are one and then two of the six channels saturated on every step. Clipping the global
gradient norm at 0.5 changes nothing measurable: AG's returns stay within 0.03 of A0's.

*Rule A.* The stable arms (share of |z| > 3 at iteration 20 below 0.05) are A2 (0.00), A3 (0.00)
and AW (0.00). A0 (1.00), A1 (0.33) and AG (1.00) are not. Among the stable arms the largest rate
at iteration 20 is AW's 6e-5 (A2 3e-5, A3 1e-5), so **AW goes to stage B**. The GPU lock was held
for 629 s: 58–60 s per 20-iteration run, 262 s for AW's 101 iterations and 74 s for the
diagnostic.

AW's probe also recorded iterations 21–100. They were not part of the rule, but they were
already on disk. The return stays between −4.7 and −12.2 through iteration 44, then drops to
−22.1 at iteration 46, where the ramp is at 1.38e-4.

**Stage B, the screen** (AW, `rs-pix`, seed 0, 1,000 iterations; the schedule ramps over
iterations 0–100 and then decays by cosine to 0 at 1,000). The windows are iterations 900–999
with the population std, and the baseline is X7's `rs-pix-s0` curve, as in R10.

| run | wall clock | s / it. | render | rollout | learner | mean return 900–999 (std) | margin | passes? |
|---|---|---|---|---|---|---|---|---|
| baseline (X7 `rs-pix-s0`) | — | 2.49 | 0.54 | 1.54 | 0.417 | −18.623 (0.360) | — | — |
| AW | 0.734 h | 2.64 | 0.56 | 1.67 | 0.414 | −3.950 (0.295) | +14.673 | **yes** (needs > 0.360) |

| iteration | 0 | 1 | 46 | 100 | 150 | 500 | 999 |
|---|---|---|---|---|---|---|---|
| value loss / entropy / return | 3.27 / 5.51 / −8.90 | 3.82 / 5.51 / −8.85 | — / — / −22.08 | 3.28 / 5.57 / −18.20 | — / — / −7.42 | 0.57 / 5.90 / −4.73 | 0.73 / 5.88 / −3.49 |

The return is below −15 on 87 of iterations 45–132. It recovers after 132, while the rate is
still near its 3e-4 peak: −7.2 by iteration 152, −5.0 on average over iterations 200–299 and −4.4
over 400–499. `diag.py`:
checkpoint 100 has mean |z| 1.31, share 0.00 and segment return −17.48. Checkpoint 1,000 has mean
|z| 1.51, share 0.069 and segment return −4.04, better than both constant actions (−14.36 and
−8.37). `envelope_violation_rate` and `executed_ne_sampled_rate` were 1.00 in every iteration.
The nine metrics were `physics_steps_per_sec` 28,331, `actions_per_sec` 7,083,
`camera_frames_per_sec` 1,838 and `pixels_per_sec` 1.69e7; the other five were `null` (not
instrumented).

*Rule B passes*, so the budget was projected before stage C: 0.91 GPU-hours used, plus six runs
at four times stage B's lock time, six evaluations at 0.05 h and two diagnostics, gives
18.9 GPU-hours, inside the 25-hour budget. Stage C ran.

**Stage C, the rows.** AW × seeds {0, 1, 2} × `rs-pix` and `rs-dr-pix`
(`training-reach-vision-rs[-dr]-pix-critic-warmup.toml`, 4,000 iterations, plan goldens
`plan-reach-vision-rs[-dr]-pix-critic-warmup.txt` from `train_r11_dry_run_plans`). The final
value loss, entropy and return are at iteration 3,999. The last-100 return is the mean (std) over
iterations 3,900–3,999. |z| and the segment return come from `diag.py` on `checkpoints/4000.esb`,
run on the row's own rollout documents.

| row | seed | wall clock | s / it. | final value loss / entropy / return | last 100 return (std) | X7's last 100 | mean \|z\| / share > 3 | segment return |
|---|---|---|---|---|---|---|---|---|
| `rs-pix` | 0 | 2.94 h | 2.65 | 0.61 / 5.91 / −3.72 | −3.41 (0.21) | −16.73 | 5.12 / 0.58 | −3.83 |
| `rs-pix` | 1 | 2.93 h | 2.64 | 0.34 / 5.71 / −3.90 | −4.04 (0.31) | −7.77 | 4.08 / 0.53 | −4.43 |
| `rs-pix` | 2 | 2.94 h | 2.64 | 0.58 / 7.14 / −4.13 | −4.43 (0.39) | −8.73 | 7.76 / 0.85 | −4.60 |
| `rs-dr-pix` | 0 | 2.96 h | 2.66 | 0.36 / 5.69 / −3.77 | −4.01 (0.28) | −18.66 | 2.28 / 0.35 | −4.08 |
| `rs-dr-pix` | 1 | 2.95 h | 2.65 | 0.66 / 7.05 / −5.49 | −5.78 (0.31) | −15.92 | 3.55 / 0.50 | −5.94 |
| `rs-dr-pix` | 2 | 2.93 h | 2.64 | 2.33 / 7.14 / −4.86 | −5.18 (0.74) | −8.70 | 14.88 / 0.88 | −6.42 |

Render was 0.54–0.56 s, rollout 1.66–1.69 s and learner 0.414 s per iteration in every run.
`envelope_violation_rate` and `executed_ne_sampled_rate` were 1.00 in every iteration of every
run. The nine metrics (`metrics/env-metrics.json`; `observation_gb_per_sec`,
`policy_inferences_per_sec`, `p50_end_to_end_latency`, `p95_end_to_end_latency`,
`gpu_memory_peak` and `chunk_underrun_rate` were `null`, as before):

| row | seeds | `physics_steps_per_sec` | `actions_per_sec` | `camera_frames_per_sec` | `pixels_per_sec` |
|---|---|---|---|---|---|
| `rs-pix` | 0 / 1 / 2 | 27,363 / 28,919 / 29,255 | 6,841 / 7,230 / 7,314 | 1,859 / 1,900 / 1,896 | 1.71e7 / 1.75e7 / 1.75e7 |
| `rs-dr-pix` | 0 / 1 / 2 | 26,732 / 31,006 / 31,528 | 6,683 / 7,752 / 7,882 | 1,839 / 1,827 / 1,822 | 1.69e7 / 1.68e7 / 1.68e7 |

*Evaluation.* Each run's `checkpoints/4000.esb` on its row's document
(`evaluation-reach-vision-<row>.toml`, seeds 201–216, `es eval run --jobs 4 --frames`, 138–154 s
each; the frames were deleted after each report). `success_rate` per suite:

| row | seed | `nominal` | `light_intensity` | `light_direction` | `observation_delay` | `torque_noise` | `backlash` |
|---|---|---|---|---|---|---|---|
| `rs-pix` | 0 | 0.1875 | 0.125 | 0.3125 | 0.1875 | 0.4375 | 0.1875 |
| `rs-pix` | 1 | 0.125 | 0.0625 | 0.00 | 0.0625 | 0.1875 | 0.1875 |
| `rs-pix` | 2 | 0.0625 | 0.00 | 0.125 | 0.125 | 0.0625 | 0.125 |
| `rs-dr-pix` | 0 | 0.3125 | 0.1875 | 0.3125 | 0.00 | 0.25 | 0.25 |
| `rs-dr-pix` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `rs-dr-pix` | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

These are the first held-out successes of pixel-only PPO in this project, where X7 scored 0.00 on
all 12 runs. No run meets the documents' acceptance (`nominal` success ≥ 0.8). The
`envelope_violation_rate` is still 1.00 in every suite. The `nominal` histogram of `rs-dr-pix`
seed 0, for one, is 5 successes, 11 timeouts, 16 fallbacks, 16 chunk underruns, and 2,426
position, 2,262 velocity and 2,085 acceleration violations.

Two limits on what stage C shows:

* Against X7, AW changes both the critic and the schedule. Against R10 it changes only the
  schedule, and R10's arm P followed X7 within 0.036 return per iteration for 1,000 iterations.
  The schedule is also a warmup *and* a decay to 0, and this packet does not separate the two.
* At 4,000 iterations the head's `tanh` is partly saturated again in every run (share of
  |z| > 3 of 0.35–0.88). This time it comes with the best returns measured, so late saturation
  here is not the first-update collapse.

The GPU lock was held for 18.82 h in total: stage A 0.17 h, stage B 0.74 h (2,645 s training,
14 s diagnostic), and stage C 17.90 h (63,536 s training, 881 s evaluation, 36 s diagnostics).

*The packet's questions.*

1. **A lower rate of 3e-5 or 1e-5, or a 100-iteration warmup, keeps the pixel actor's `tanh` out of
   saturation through 20 iterations (share of |z| > 3 of 0.00 at iteration 20). `lr = 1e-4`
   (0.33) and `grad_clip = 0.5` (1.00, the same as no change) do not. Rule A chose the warmup.**
2. **Yes, partly.** With the warmup-cosine schedule and R10's privileged critic, pixel-only PPO
   learns `rs-pix`. The 1,000-iteration screen beat X7's baseline by 14.7 return (−3.95 against
   −18.62), and at 4,000 iterations all three `rs-pix` seeds score held-out `nominal` successes
   (0.19, 0.13, 0.06) where X7's scored 0.00. So does one of three `rs-dr-pix` seeds (0.31).
   None comes near the 0.8 acceptance, and the envelope still clamps every step.

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
function, different cut; `crates/es-import/src/rl_import.rs::learning_graph` is where it happens
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

### 8a. Adapter v2 — an Isaac Lab or Playground policy's I/O conventions (packet M11/X2)

Spec §28.14 rule 3: the IR owns an external policy's I/O conventions. Every v2 field is optional
and `deny_unknown_fields` still holds; an adapter that declares none of them converts to exactly
the bytes it converted to before (the four committed v1 conversions are pinned by hash in
`crates/es-import/tests/adapter_v2.rs`). Each source convention lands in one place:

| source convention | adapter v2 | where it lands in the bundle |
|---|---|---|
| joints resolved by name in the articulation's order (Isaac `resolve_matching_names`) | `[joints] source_names` + `[joints.rename]` (exclusive with `source_order`) | the first Dense's input columns of every per-joint channel and the head's rows are permuted into our actuator order — a permutation, exact |
| `default_joint_pos` | `[joints] default_pos` (source order), or the manifest's `default_joint_pos` | only where something reads it (below) |
| `joint_pos_rel = q − default`, per-term `scale` | `[[observation.channels]] offset = "default_pos"` or a vector, `scale` (number or vector) | folded into `Normalize{MeanStd}`: `mean = offset + mean_src / scale`, `std = std_src / scale` |
| `last_action` / `last_act` (the raw action, zero at reset) | a channel whose Task IR source is the new `ObsSource::PreviousAction { initial }` | the loop serves the previous tick's policy row in actuator units; the fold carries the action tail's inverse (`raw = (row − offset) / scale`); the Task IR must declare `initial = offset` (our order), else `IMP-005` |
| `history_length` | `history = N`, `history_order = "newest_last"` (Isaac's flattening) or `"newest_first"` | a `TemporalWindowNode` of N (`Align::Hold`: the first frame repeated until the ring fills, Isaac's `CircularBuffer` on reset); newest-first is a column permutation |
| per-term `clip`, the wrapper's `clip_observations` | `clip = [lo, hi]` | **`IMP-009`**: the Observation IR has no clamp node and this packet adds none. A clip that never binds on the states the policy meets is left undeclared (the Isaac oracle checks `max |obs| < 100`) |
| `JointPositionActionCfg`: `raw · scale + offset`, `use_default_offset` | `[action] scale`, `use_default_offset = true` (`offset = default_pos`) | `Normalizer{Inverse, MeanStd}` with `mean = offset`, `std = scale`, in our order |
| `clip_actions` | `[action] clip = [lo, hi]` | accepted only where it cannot bind — `squash = tanh` and `[lo, hi] ⊇ [−1, 1]`; else `IMP-009` |
| `decimation × sim.dt`, Playground `ctrl_dt` | `[timing] policy_dt`, or the manifest's `decimation` / `sim_dt` | checked against the Deployment IR's control period, never resampled; a mismatch is **`IMP-006`**; the report's `timing` line |
| actuator `stiffness` / `damping` / `armature` / `effort_limit` | `[actuators]` (source order) | `mapping-report.json` rows next to the scene's `kp`, `kv` + joint damping, armature and force range; never converted |
| `projected_gravity`, `base_lin_vel`, `base_ang_vel`, `velocity_commands`, a `generated_commands` with no Task IR channel | (the channel's `source` term) | **`IMP-007`**, naming the term |
| both / neither of `source_order`, `source_names`; an unused rename; a resolution that is not a permutation | — | **`IMP-008`** |

**`ObsSource::PreviousAction { initial }`** is the last variant of `ObsSource`, and absent from
every committed Task IR, so no `task_hash` moved (`crates/es-ir/tests/previous_action.rs`). Its
value is the row the policy emitted for the previous control tick — before the Safety Plane, and
for `JointDelta` the increment, not the integrated target — read in `es_eval::runner` (evaluation
and collection, `capture_at`, from `ChunkBuffer::action_at` of the previous tick; an underrun tick
emitted no row and the last one stands) and in `es_py::Rollout` (the row last handed to `act`).
`initial` (absent = zeros) is served on tick 0 of every episode. A bake reads recorded rows and
keeps no policy output, so it refuses the channel by name.

**`import_rl.py`.** rsl_rl's `EmpiricalNormalization` is read wherever that version kept it —
`obs_normalizer.*` in a ≥ 5.0 `actor_state_dict`, `actor_obs_normalizer.*` (never the critic's)
in a 3.x `model_state_dict`, the runner's top-level `obs_norm_state_dict` in 2.x — flattened from
`[1, D]`, and `obs_std = std + eps` (`eps = 1e-2`), because `forward` divides by the sum.
`--isaac-env-cfg params/env.yaml` records `decimation`, `sim_dt`, the single action term's
`scale` and `action_kind`, and `scene.robot.init_state.joint_pos` resolved into
`default_joint_pos` against `--joint-names` (the articulation's order; without it the regex
table is not resolved and says so). `--playground-config` records `ctrl_dt / sim_dt` as
`decimation`, `sim_dt`, `action_scale` and a dumped `default_pose`. The pickle writer the
oracle's generator needs (`save_native_rsl_rl`) lives here too (INV-16).

**Measured (packet oracle 2, this workstation, torch 2.14 CPU, 256 states each).**
`python/es/rl_source/isaac_reference.py` computes each framework's observation → action map in
NumPy from `isaac-lab.md` §§ 2–6 and `brax-ppo-so101.md` § 6, and the imported bundle
(Observation IR on `CpuPlan`, Learning IR on `TorchRuntime`) is compared in actuator units:

| source | max abs error |
|---|---|
| Isaac-style rsl_rl, classic `model_state_dict` + `obs_norm_state_dict`, joints in another order and one renamed | 1.216e-7 |
| the same actor in the ≥ 5.0 `actor_state_dict` shape | 1.216e-7 |
| Playground-style brax (swish, tanh, `default_pose + 0.3·a`) | 7.605e-8 |
| negative control: the Isaac source with `joint_vel`'s `scale = 0.05` left undeclared | 6.424e-1 |

Two things this does not settle. The `eps = 1e-2` of rsl_rl's normalizer is not in the api-note
(its § 8 open item is the pinned rsl_rl version); both the reader and the reference state it, so a
different eps in the pinned version would move both. And no real Isaac checkpoint has been
imported yet — that is wave 3's I3.

## 9. Open questions for a human

1. Should rollouts model the Deployment IR's declared latency (a chunk buffer in the trainer),
   or stay synchronous with the evaluation carrying the honesty? This note chooses synchronous.
2. When the plane clamps a sampled action, the log-probability is that of the *sample*; the env
   saw the *executed* action. The rate at which they differ is reported. **Answered for the
   trainer by packet M9/R5** (section 2a): `[rl] estimator = "executed"` trains on the executed
   action, the default stays `"sampled"`, and section 7's R5 row says what it measured. What is
   still a human's is the other half of the same question — what the envelope should mean for a
   *learning* policy on this task (`docs/reviews/M9.md` S-7).
3. `Squash::Tanh` on a Regression head: an IR parameter (this note) or a property of the action
   unit the lowering applies (`quadruped-track.md` 3.6 question 2)? This note chooses the
   parameter, with absent = default = today's hash.
