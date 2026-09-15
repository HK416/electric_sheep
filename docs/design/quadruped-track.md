# Quadruped track (Track B): train externally, import, run here

Status: sections 1–3 written by packet M6/B1, which is the track's first and LOCAL-ONLY packet —
a scene, four IR documents and the oracles that judge them. Nothing has been trained and nothing
has been imported yet. Pinned upstream facts live in
`docs/api-notes/mujoco-playground-quadruped.md` (mujoco_playground 0.2.0 @ `124a73f`, menagerie
@ `1b86ece`, brax 0.14.2); this note does not restate them, it says what we do about them.

Spec sections this track leans on: §8 (Learning IR), §1.9 (what is never cut and what is cut
first), §9 (Deployment IR and the Safety Plane), §17 (backends), §28 (why M6 exists).

## 1. Why external RL and an import, rather than training here

Electric Sheep is a **compiler and runtime**, not a training framework. §1.9 puts a native
physics solver first in line to be cut; a native PPO implementation is not even on the list,
because it was never in scope. The Learning IR (§8) is the import surface: it represents a
policy — encoders, head, chunker, unnormalizer — as typed nodes, and §8.7 lowers those nodes to
PyTorch so that the same IR run in PyTorch is the ground truth. Nothing in §8 says the weights
have to have been produced here.

So Track B is: **`mujoco_playground` trains a Go1 joystick policy in MJX/JAX PPO; we import the
weights into a Learning IR graph and run the policy through our own Observation IR, Deployment
IR and Safety Plane.** That is the honest division of labour and it also happens to be the
cheapest path to a walking robot in this repo:

- Upstream has already tuned the part that is hard and that we have no opinion about: fifteen
  shaped reward terms, the domain-randomization ranges, the PD gains, the observation noise.
  Re-deriving those with no reference to check against is the harder path to the same place.
- Upstream's training physics is **MuJoCo** (MJX). Our reference backend is MuJoCo (CPU). The
  contact model, the solver and the integrator are the same family, which is the single biggest
  reason to prefer `mujoco_playground` over `legged_gym` or Isaac Lab, both of which train under
  PhysX (api-note section 4).
- It exercises the part of our stack that has never been exercised: a **floating-base** robot
  with twelve actuated joints. Every fixture in this repo before it — the pendulum, the two-link
  arm, the SO-101 — is fixed-base. Section 3 is mostly a list of what that turned up.

What this repo owns, and what the import cannot skip: the scene our loader accepts, the four IR
documents, the hash chain over them, the Safety Plane the policy's actions pass through, and the
Evaluation IR that judges the result. None of those are `mujoco_playground`'s.

## 2. The scene derivative

`tests/fixtures/mjcf/go1_primitives.xml` is a single-file, primitives-only derivative of
mujoco_playground's `go1_mjx_feetonly.xml` plus the two elements of
`scene_mjx_feetonly_flat_terrain.xml` we need. The full enumeration is
`tests/fixtures/mjcf/go1_primitives.PROVENANCE.json`, which the provenance oracle reads; the
shape of it is:

- **Substituted:** the 13 `class="visual"` geoms that referenced 5 Unitree STL meshes are boxes.
  That is the only change to the robot, and it cannot change the dynamics: those geoms are
  `contype=0 conaffinity=0`, and every body carries an explicit `<inertial>`, which MuJoCo
  prefers over geom-derived inertia. Collision geoms were already primitives upstream — that is
  what `feetonly` means — so **no collision geometry was approximated at all**, which is the
  thing that would have been a physics change.
- **Inlined:** the flat floor geom and the `home` keyframe, because our importer refuses
  `<include>` (`MjcfError::Include`) and `home` is both the reset pose and the zero-point of
  every action (`motor_targets = default_pose + action * 0.5`).
- **Dropped:** the `<sensor>` block, the five `<site>`s it referenced, and one `<light>`.
  `crates/es-physics-backend/src/mjcf_out.rs` emits only `jointpos`/`jointvel` sensors, so a
  scene declaring a gyro or a velocimeter is refused at load by `check_requirements` before
  MuJoCo sees it. Sensors carry no dynamics, so this costs no physics — but it costs the
  observation, see section 3.
- **Byte-for-byte:** the `<option>` block, the whole `<default>` tree, every joint, every
  actuator, every collision geom, every `<inertial>`, and `home`. `cargo test -p es-assets
  --test go1_provenance` proves it against the pinned commit, verifying blake3 **before**
  parsing the downloaded bytes (§25.1).

Why not vendor the STLs: 5 binary meshes we would have to keep, licence and all, for geometry
that draws nothing in any oracle this track runs. The SO-101 fixture set the precedent
(`so101_provenance.rs`) and this follows it exactly.

Licence: the robot is menagerie's `unitree_go1`, BSD-3-Clause, HangZhou YuShu TECHNOLOGY CO.,LTD.
mujoco_playground's edits and the scene file are Apache-2.0. Both permit a modified
redistribution in source form. `go1_primitives.LICENSE` is menagerie's file, unmodified.

## 3. The four documents, the parity risk, and what is still open

`tests/fixtures/quadruped/{task,observation,learning,deployment}.toml` are generated by
`cargo test -p es --test cli -- --ignored regenerate_quadruped_documents` from the scene and
from each other, so no hash and no joint limit in them is typed in by hand — the arm demo's
`regenerate_visible_learning_documents` is the precedent. Each file's header says what it is;
this section says what is *wrong* with them, which is the part worth writing down.

### 3.1 Physics parity, and how the oracle pins it

The risk: a policy trained in MJX under `iterations=1 ls_iterations=5 timestep=0.004
integrator=Euler` with `eulerdamp` disabled, deployed by us through `parse_mjcf` →
`scene_to_mjcf` → MuJoCo. **Everything that round trip drops is physics the policy never saw.**
Running our CPU backend at MuJoCo's default 50 line-search iterations, or with implicit damping,
changes contact resolution and integration in ways that show up as foot slip and jitter.

Two of those were real. Before this packet, `SceneDesc::PhysicsOptions` had no field for
`ls_iterations` and `<option><flag>` was warned about and dropped, so the re-emitted MJCF told
MuJoCo `ls_iterations = 50` (its default) and left `eulerdamp` on. Both are now carried and
emitted; `the_emitted_mjcf_keeps_the_playground_option_block` is the regression test and it needs
no Python, so it runs in PR CI.

What is still dropped on the round trip, and known: geom `priority` (so MuJoCo mixes friction
pairwise, `max(1.0, 0.6) = 1.0`, instead of letting the floor's `0.6` win) and `<position
inheritrange>` (so the actuators lose the `ctrlrange` MuJoCo would have derived). Neither moves a
zero-action standing trajectory, which is why the step oracle's tolerance is the backend's
declared `DeterminismTier::PhysicsMeaning` and not bitwise — §4.3 forbids an external backend
declaring tier 1 in any case. Both are cheap to fix and neither is fixed here: a `priority`
field is an `es-assets` change with its own tests.

`crates/es-physics-backend/tests/go1_step.rs` (`#[ignore]`, needs `ES_PYTHON` with `mujoco`)
steps 250 control ticks × 5 substeps from `home` with the action at zero through our backend,
asserts no NaN and that the trunk is still above 0.20 m — i.e. **it stands** — then steps the
same XML through `mujoco` directly and compares `qpos`. It also asserts that MuJoCo read
`ls_iterations = 5` and `eulerdamp = false` out of our fixture rather than its own defaults, and
prints wall-clock per 1,000 physics steps for both paths as an observation (§12.4 — nine metrics
or none; a single `step/s` figure is not a claim this repo makes).

### 3.2 The observation cannot be served yet — the track's first blocking item

Playground's policy input is 48 wide: `local_linvel(3)`, `gyro(3)`, `gravity(3)`,
`joint_pos(12) - default_pose`, `joint_vel(12)`, `last_action(12)`, `command(3)`. Our
Observation IR declares exactly that, in that order, and `es ir check` is green over the four
documents. But nothing in our runtime can *fill* it:

- `crates/es-eval/src/runner.rs::input_sources` resolves an observation input to a `qpos` slice,
  an MJCF sensor range, or an image. Joint velocities are not reachable (there is no `qvel`
  capture), base linear velocity and the gyro are not reachable (we dropped the sensors, and
  `mjcf_out` could not emit them anyway), projected gravity is a rotation of the base quaternion
  that no node computes, and `last_action` and `command` are runtime state, not sensor readings.
- Task IR-D has no source for most of it either, which is why the `ObservationSpec` node in
  `task.toml` is deliberately unfed. Declaring the channel is still correct — §7.4 is exactly
  "Task IR declares, Observation IR implements" — but IR-D computes none of it.

So `es eval run` on these four documents refuses **by name** today rather than faking a run
(§1.4). `quadruped_eval_run_names_the_observation_gap` pins that refusal's message, so the day
the capture path grows base state the test fails and gets updated instead of quietly staying
green over a gap.

### 3.3 The Safety Plane is fed a fixed-base robot's state — the second blocking item

`crates/es-eval/src/runner.rs::joint_state` hands the plane `qpos[0..NJ]` and `qvel[0..NJ]`. For
a fixed-base arm that is the actuated joints. For Go1 it is the trunk's free joint plus the first
five or six hinges: the plane would judge a base position against a knee's limit. The envelope in
`deployment.toml` is correct for the robot; feeding it the right twelve rows is an `es-eval`
change (the leading-`NJ` convention has to become "the actuated joints' indices out of
`ModelInfo`"), and it is out of this packet's scope. It is **not** worked around, and certainly
not worked around by disabling anything (INV-12).

What can be proved today, and is: `quadruped_bundle_runs_100_ticks_through_the_safety_plane`
builds a `policy.esb` from the four documents and runs 100 control ticks through the real
`SafetyPlane::<12, 1>` built from `deployment.toml`. Two passes. An untrained policy's action
(`tanh`-bounded, so up to ±0.5 rad of command jump per 20 ms tick, ~50 rad/s against a 21 rad/s
joint) is clamped on all 100 ticks and never latches a fallback and never leaves the position
limits; a command the robot can physically follow (0.02 rad/tick) passes through as
`ActionSource::Policy` on all 100. The second pass is the one that matters: without it, an
envelope clamped to a constant would also "pass" the first — the defect packet M5/V6 found on the
arm.

### 3.4 The Learning IR is not yet the network upstream trained

Three gaps, all of them in the *lowering*, not in the document:

1. **Activation.** brax's `MLP` uses `linen.swish`; `lower_to_torch` emits `nn.ReLU`. Imported
   weights will not reproduce the trained policy until one of the two moves. This is the largest
   single import risk and it is the next packet's first question.
2. **Activation placement.** Upstream is `Dense(512) swish, Dense(256) swish, Dense(128) swish,
   Dense(24)`. Ours is `StateEncoder{Mlp hidden=[512,256], out_dim=128}` → `Linear(48,512) ReLU
   Linear(512,256) ReLU Linear(256,128)` plus the head's `Linear(128,12)`: four linear layers, as
   upstream, but no activation before the head. (`hidden = [512, 256]` with `out_dim = 128` is
   how "an MLP of 512, 256, 128" is spelled in our node set; `hidden = [512,256,128]` would be a
   *fifth* linear layer.)
3. **`tanh`.** Deterministic inference upstream is `tanh(location)` over the first half of the
   24-wide output (`NormalTanhDistribution.mode`); the log-scale half is unused. Our node set has
   no activation node at all, so `tanh` is not represented. The action port's `Normalized { lo =
   -1, hi = 1 }` unit and the Safety Plane's clamp give the *range* of `tanh` but not its shape.

What *is* exact: the observation normalizer is `Normalizer { Forward, MeanStd }` with placeholder
`mean = 0, std = 1` — the shape brax's `running_statistics` exports, filled from the checkpoint
by the import packet — and the action unnormalizer is `Normalizer { Inverse, MeanStd }` with
`mean = default_pose` (out of the scene's `home` keyframe) and `std = 0.5` (`action_scale`),
which is `motor_targets = default_pose + action * action_scale` exactly. Weights are a
`safetensors` reference with an all-zero hash; no pickle path exists anywhere (INV-16).

### 3.5 What else the documents cannot say

- **Termination.** Upstream terminates when the upright vector's z-component goes negative; the
  usual second guard is base height. Neither is expressible: `es-env`'s reward/termination cone
  binds a joint's *first* `qpos` index (`crates/es-env/src/plan.rs::joint_leaf`), and a free
  joint's first index is x. `task.toml` terminates on timeout and on "the base left the arena"
  (|x| > 3 m), and says so in its header.
- **Reward.** Upstream's `exp(-err² / tracking_sigma)` needs `MathFn` (not in the cone) and a
  constant leaf to subtract the commanded velocity with (IR-D has none). Ours is forward speed
  normalized over the command bound. This costs nothing today — upstream's fifteen terms train
  the policy, in `mujoco_playground` — but it is the reward *our* evaluation harness would use.
- **The joystick command.** Upstream resamples it mid-episode on an OU-like schedule. IR-D can
  say "drawn once per episode", which is what the three `Randomization` nodes do. A mid-episode
  resample is an IR-C (control graph, §6.2) question.
- **Observation noise.** Upstream adds per-channel *uniform* noise inside its env.
  `AugmentKind` has `GaussianNoise` and no uniform variant, and INV-15 would keep an `Augment`
  node off during evaluation anyway. Not represented; it belongs to the trainer, not to us.

### 3.6 Open questions for a human

1. Do we move `lower_to_torch` to swish for `StateEncoder{Mlp}`, add an activation parameter to
   the node (an IR change, and §1.9's "only 7 extension points" makes us think twice), or accept
   a re-tune of the imported policy under ReLU? (3.4 #1.)
2. Is `tanh` a Learning IR node, a `PolicyHead` parameter, or a property of the action unit that
   the lowering applies? (3.4 #3.)
3. Who owns "the actuated joints' indices" — `ModelInfo`, the Deployment IR, or a new field on
   the Task IR's `ObsSource::JointState`? Both 3.2 and 3.3 are waiting on the same answer.
4. Go1 or Go2? Upstream has only Go1 (api-note section 1). A Go2 track means hand-porting
   `Joystick` onto menagerie's `unitree_go2`, which is itself already primitives-collision. Not a
   call to make before the Go1 policy has run once.
