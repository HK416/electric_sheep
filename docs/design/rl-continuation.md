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

**Oracle 4 is deferred to S4d.** The reach task of section 5 needs `GetBodyPose` and `Norm` in a
reward cone, and `es-env`'s `ScalarPlan` lowers neither — it lowers `GetJointState`,
`GetSensor`, `GetTime`, `Arith`, `Compare`, `Normalize`, `Logic` and `Clamp`, every leaf binds a
single scalar, and `es_ir_types::Expr` has no square root by design (its doc cites §6.6
`DET-010`). So −‖cube_pos − gripper_pos‖ has no form to lower *into*, independently of
`es-env`. Packet **S4d** owns that cone extension and the four `*-reach.toml` documents, and the
success-rate row of this table is written there. What is measured above is the infrastructure —
the recipe, the route, the trainer, the plane and the reproducibility — on documents that
already execute.

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
