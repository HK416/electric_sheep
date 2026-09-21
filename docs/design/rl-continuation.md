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
- observation (15): `joint_pos[6] ‖ joint_vel[6] ‖ (cube_pos − gripper_pos)[3]`, gripper =
  body `gripper` (site `gripperframe` on the brax side, the same point);
- action (6): position targets, normalized `[-1, 1]` over each actuator's `ctrlrange`
  (`Normalizer{Inverse, MeanStd}` with `mean = centre`, `std = half-range`);
- reward: `−‖cube_pos − gripper_pos‖` per step, `+1` on success;
- success: distance `< 0.03` m; timeout 200 control steps.

The Task IR (`tests/fixtures/rl/task-reach.toml`) spells this with existing nodes
(`GetBodyPose`, `Arith`, `Norm`, `Compare`, `Reward`, `Terminate`). If the brax env has to
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

## 8. Open questions for a human

1. Should rollouts model the Deployment IR's declared latency (a chunk buffer in the trainer),
   or stay synchronous with the evaluation carrying the honesty? This note chooses synchronous.
2. When the plane clamps a sampled action, the log-probability is that of the *sample*; the env
   saw the *executed* action. The rate at which they differ is reported; whether to train on the
   executed action instead is a later ablation.
3. `Squash::Tanh` on a Regression head: an IR parameter (this note) or a property of the action
   unit the lowering applies (`quadruped-track.md` 3.6 question 2)? This note chooses the
   parameter, with absent = default = today's hash.
