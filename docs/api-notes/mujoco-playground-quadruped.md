# `mujoco_playground` quadruped joystick, pinned for the locomotion track (Track B)

Pins: `mujoco_playground` (PyPI name `playground`) **0.2.0**
([tag `v0.2.0`](https://github.com/google-deepmind/mujoco_playground/releases/tag/v0.2.0) =
commit [`124a73f`](https://github.com/google-deepmind/mujoco_playground/tree/124a73fa3303f75a62f8fe04d329b829ed0ebdfb),
Apache-2.0), `brax` **0.14.2** (the floor `playground` 0.2.0 requires; Apache-2.0),
`mujoco_menagerie` commit
[`1b86ece`](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97)
(2025-11-07 — the exact commit `playground` 0.2.0 vendors, `mjx_env.MENAGERIE_COMMIT_SHA`),
BSD-3-Clause per robot. No part of this was executed by us — no venv install, no training run,
no policy import. Every structural fact below (observation composition, config defaults, XML
geom types, PPO hyperparameters) was read directly out of the pinned source at the commit
above and is cited by file path; every performance number (wall-clock, throughput) is the
paper's, marked **reported by upstream**. Track B per the roadmap: train externally, import the
MLP into our Learning IR, run our runtime.

## 1. Go1 joystick environment — Go2 does not exist upstream

`mujoco_playground` ships **only Go1** for quadruped joystick locomotion
(`mujoco_playground/_src/locomotion/go1/`); there is no `go2/` package, no `Go2Joystick*` env
class, and the technical report never mentions "Go2" (checked by full-text search of the PDF).
[GitHub issue #270](https://github.com/google-deepmind/mujoco_playground/issues/270) is a user
asking how to adapt the Go1 task to a Go2 model; it is not upstream support. Any Go2 track
means porting the Go1 `Joystick` task onto the Go2 XML by hand, not loading a ready-made env.

**Observation** (`joystick.py::_get_obs`, `_post_init`, `default_config`): the policy input
`"state"` is 48-dim, `hstack` of noisy `[local_linvel(3), gyro(3), gravity(3), joint_pos(12) -
default_pose, joint_vel(12), last_action(12), command(3)]`. Noise is additive uniform,
`level=1.0` scaled per-channel (`joint_pos 0.03, joint_vel 1.5, gyro 0.2, gravity 0.05, linvel
0.1`). A second key, `"privileged_state"` (123-dim: `state` plus un-noised
gyro/accelerometer/gravity/linvel/angvel/joint_pos/joint_vel, `actuator_force(12)`,
`last_contact(4)`, per-foot velocity `(4×3)`, `feet_air_time(4)`, torso `xfrc_applied(3)`, one
perturbation-active flag), feeds the **critic only** (`value_obs_key="privileged_state"` in
§2) — the deployed policy never sees it. `history_len=1` in `default_config()`: Go1 stacks no
frames (contrast Spot/H1 joystick tasks, `history_len=3`, same config field).

**Action**: 12-dim, one position target per joint. `motor_targets = default_pose + action *
action_scale` (`action_scale=0.5` rad, `default_pose` = the `home` keyframe's `qpos[7:]`).
PD gains are written into the model at env construction (`go1/base.py::Go1Env.__init__`):
`actuator_gainprm[:,0]=Kp`, `actuator_biasprm[:,1]=-Kp`, `dof_damping[6:]=Kd`, with
`Kp=35.0`, `Kd=0.5` (`default_config`). `ctrl_dt=0.02` (50 Hz control), `sim_dt=0.004` (250 Hz
physics, `n_substeps=5`), `action_repeat=1`. `episode_length=1000` steps = 20 s. Termination:
upright-vector z-component `< 0` (fully flipped).

**Command**: an Ornstein–Uhlenbeck-like resample (`sample_command`), amplitude bounds
`a=[1.5, 0.8, 1.2]` (m/s forward, m/s lateral, rad/s yaw), per-axis resample probability
`b=[0.9, 0.25, 0.5]`, waiting time `~Exp(mean 5 s)` between resamples.

**Domain randomization** (`go1/randomize.py`, opt-in via the `--domain_randomization` CLI
flag — off by default): floor friction `U(0.4, 1.0)`; joint `frictionloss *= U(0.9, 1.1)`;
`armature *= U(1.0, 1.05)`; torso CoM `+= U(-0.05, 0.05)` m; all body mass `*= U(0.9, 1.1)`;
extra torso mass `+= U(-1.0, 1.0)` kg; `qpos0[7:] += U(-0.05, 0.05)` rad. A separate
`pert_config` (external velocity-kick force on the torso, `0–3 m/s` over `0.05–0.2 s`, every
`1–3 s`) exists but is **disabled by default** (`enable=False`).

**XML** (`go1/xmls/scene_mjx_feetonly_flat_terrain.xml` → `go1_mjx_feetonly.xml`, `meshdir` →
`mujoco_menagerie/unitree_go1/assets`): **collision geoms are already primitives.** The
`collision` default class uses only `cylinder`/`capsule`/`sphere`/`box` — trunk: 1 box + 2
cylinders; each leg: 3 hip cylinders, thigh/calf as capsules via `fromto`, one foot sphere
(`r=0.023`, the only geom with `contype=1`, everything else in that class is `contype=0
conaffinity=0` except the feet). **Visuals are 5 STL meshes** (`trunk`, `hip`,
`thigh_mirror`, `thigh`, `calf`), one `<geom class="visual" mesh=... group=2 contype=0
conaffinity=0>` per body — these are the only mesh geoms in the file. Solver:
`iterations=1 ls_iterations=5 integrator=Euler timestep=0.004`, cone unspecified (MuJoCo
default `pyramidal`). A `fullcollisions` XML variant also exists (adds self-collision geoms);
not the default and not needed here. Rough-terrain uses the same robot XML with a heightfield
scene and larger `naconmax`/`njmax`.

**Rewards** (`reward_config.scales`, `tracking_sigma=0.25`, `max_foot_height=0.1` m):
`tracking_lin_vel +1.0` and `tracking_ang_vel +0.5` (both `exp(-err²/tracking_sigma)`);
costs `lin_vel_z -0.5`, `ang_vel_xy -0.05`, `orientation -5.0`, `dof_pos_limits -1.0`,
`stand_still -1.0`, `termination -1.0`, `torques -0.0002`, `action_rate -0.01`, `energy
-0.001`, `feet_clearance -2.0`, `feet_height -0.2`, `feet_slip -0.1`; `pose +0.5` (stay near
default joint angles) and `feet_air_time +0.1`. Summed, scaled by `sim.dt`, clipped
`[0, 10000]` per step.

Sources: [`go1/joystick.py`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/mujoco_playground/_src/locomotion/go1/joystick.py),
[`go1/base.py`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/mujoco_playground/_src/locomotion/go1/base.py),
[`go1/randomize.py`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/mujoco_playground/_src/locomotion/go1/randomize.py),
[`go1/xmls/go1_mjx_feetonly.xml`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/mujoco_playground/_src/locomotion/go1/xmls/go1_mjx_feetonly.xml).

## 2. Training

Command (README + `learning/train_jax_ppo.py`, both pinned commit):
`train-jax-ppo --env_name Go1JoystickFlatTerrain --domain_randomization` (installed console
script) or `python learning/train_jax_ppo.py --env_name Go1JoystickFlatTerrain
--domain_randomization`. The script loads the tuned config from
`locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")` automatically
(`train_jax_ppo.py:191`), not from its own CLI flag defaults; `learning/notebooks/
locomotion.ipynb` uses the same call. Without `--domain_randomization` the run is
un-randomized (opt-in, see §1).

**Network** (`config/locomotion_params.py`, Table 16 of the technical report — overrides the
generic Table 15 default of `(128,128,128,128)`): policy MLP `(512, 256, 128)`, value MLP
`(512, 256, 128)`, `policy_obs_key="state"`, `value_obs_key="privileged_state"` — an
asymmetric actor-critic. `brax.training.agents.ppo.networks.make_ppo_networks` (`brax`
0.14.2) builds both from `brax.training.networks.MLP`, default `activation=linen.swish`,
`distribution_type="tanh_normal"`: the policy head emits `2 × action_dim = 24` numbers
(location, log-scale of a diagonal Gaussian); deterministic inference is
`tanh(location)`, matching `NormalTanhDistribution.mode`.

**PPO hyperparameters** (`locomotion_params.py`, Table 16): `num_timesteps=200_000_000`,
`num_evals=10`, `num_resets_per_eval=1`, `reward_scaling=1.0`, `normalize_observations=True`,
`action_repeat=1`, `unroll_length=20`, `num_minibatches=32`, `num_updates_per_batch=4`,
`discounting=0.97`, `learning_rate=3e-4`, `entropy_cost=1e-2`, `num_envs=8192`,
`batch_size=256`, `max_grad_norm=1.0`, `kernel_init=lecun_uniform`.

**Wall-clock — reported by upstream**, not run by us: the technical report's real-world
section states Go1 flat-ground training (restricted command ranges) finishes "within 5
minutes (2x RTX 4090)"; Figure 13 (`Go1JoystickFlatTerrain`, full 200 M-step run, Table 16
config) plots reward vs. wall-clock for `1x 4090 / 2x 4090 / 1x A100 / 16x A100 / 1x H100 /
8x H100` out to ~500 s, noting "different devices and topologies do not make material
difference in training wallclock time" (Appendix C.4) because contacts are few. No exact
single-4090 second count is given as text, only the plot. On our single RTX 4090 the honest
estimate is **a few minutes to ~10 minutes reported-by-upstream range**, `Target / Status:
unverified` until we run it.

**Export to numpy** (`mujoco_playground/experimental/brax_network_to_onnx.ipynb`, same
commit): a trained checkpoint restores via `brax.training.checkpoint.load(ckpt_path)`
(orbax) as `params = (normalizer_params, policy_params, value_params)`; only
`(normalizer_params, policy_params)` are needed for inference. `policy_params['params']` is a
flax dict `{"hidden_0": {"kernel", "bias"}, "hidden_1": {...}, "hidden_2": {...}, "hidden_3":
{...}}` — 3 hidden Dense layers `(512, 256, 128)` plus one output Dense to `2×action_dim`,
`kernel` shape `[in, out]` (right-multiply convention, transpose for a `[out, in]` PyTorch/our
convention), `bias` shape `[out]`. `normalizer_params.mean["state"]` /
`.std["state"]` are the running observation-normalizer statistics (shape `[48]` each,
`brax.training.acme.running_statistics`) — **must** be exported alongside the weights and
applied as `(x - mean) / std` before the first Dense layer; the notebook's own `MLP.call`
does exactly this. Swish activation on the 3 hidden layers, no activation on the output layer,
`tanh` on the first half (`action_dim`) of the output, second half (log-std) unused at
inference.

## 3. Licenses

`mujoco_playground` **0.2.0**: Apache-2.0
([`LICENSE`](https://github.com/google-deepmind/mujoco_playground/blob/124a73fa3303f75a62f8fe04d329b829ed0ebdfb/LICENSE),
confirmed via the GitHub API `license.spdx_id`). One carve-out: the rough-terrain texture is
CC0 (Polyhaven), irrelevant to flat-terrain joystick.

`mujoco_menagerie` `unitree_go1` and `unitree_go2` (commit `1b86ece`, pinned by `playground`
0.2.0): both **BSD-3-Clause**, copyright HangZhou YuShu Technology Co. ("Unitree Robotics"),
derived from Unitree's public URDF
([go1](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go1_description),
[go2](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2_description)).
BSD-3-Clause permits redistribution and modification, source or binary, on three conditions:
keep the copyright notice and disclaimer in source redistributions, reproduce them in binary
redistributions, and do not use "Unitree Robotics" to endorse a derived product without
permission. **A primitives-only derivative XML is permitted** — it is exactly "modification …
in source form" under clause 1; menagerie's own `unitree_go2/go2_mjx.xml` already replaces
every collision geom with spheres (see §5), so a further primitives-only *visual* derivative
follows the same license path menagerie itself uses. Our repo has direct precedent:
`tests/fixtures/mjcf/so101_pick_place.xml` is "a hand-derived, primitives-only, single-file
derivative" of a menagerie MJCF, provenance-tracked against a pinned upstream commit
(`crates/es-assets/tests/so101_provenance.rs`).

`brax` **0.14.2**: Apache-2.0 (GitHub API `license.spdx_id` on `google/brax`).

## 4. Alternatives

**`legged_gym`** (`leggedrobotics/legged_gym`, Isaac Gym Preview + `rsl_rl` PPO,
PyTorch actor-critic — trivial numpy export). Worse for us: built on NVIDIA Isaac Gym
*Preview*, which NVIDIA discontinued in favor of Isaac Lab — the physics backend (PhysX) is a
different contact/solver family from MuJoCo/MJX, so training-time contact behavior has no
shared lineage with our MuJoCo CPU backend at all (more parity risk than an MJX-trained
policy, which at least shares MuJoCo's contact model). `unitree_rl_gym`
(BSD-3-Clause, Unitree's own fork) already has tuned Go1/Go2 configs, which is the one
practical upside, but the base framework is unmaintained upstream.

**Isaac Lab** (`isaac-sim/IsaacLab`, manager-based `Isaac-Velocity-Flat-Unitree-Go2-v0`,
BSD-3-Clause, actively maintained). Better than `legged_gym` in that it natively ships a Go2
task (mujoco_playground does not) and is the currently-maintained NVIDIA framework. Worse for
us on every other axis: requires Isaac Sim/Omniverse (multi-GB install, its own USD-based
asset pipeline, not MJCF), PhysX again (same parity risk as `legged_gym`), and heavier GPU
driver/toolkit coupling than a `pip install playground` + JAX/CUDA stack. Only worth it if the
Go2-vs-Go1 embodiment gap turns out to matter more than physics parity — not our call to make
without running the Go1 policy first.

**`mujoco_menagerie` + custom PPO** (write our own JAX/MJX or PyTorch+MuJoCo-CPU training
loop against the bare Go1/Go2 XML). Strictly worse for a first pass: reimplements exactly what
`mujoco_playground` already tuned (reward shaping, domain randomization ranges, PD gains,
observation noise) with no reference to check against, and this is a research/import task —
building a second training stack is the harder path for the same destination we get from
pinning one that already works. Reasonable *second* step if we outgrow the Go1 task shape
(e.g. want gait-conditioned rewards `mujoco_playground` doesn't have) but not the entry point.

## 5. Feasibility verdict

**Go with mujoco_playground Go1** (not Go2 — it does not exist upstream; a Go2 track means
hand-porting `Joystick` onto `unitree_go2`'s XML, itself already primitives-collision, as a
follow-up once Go1 proves the pipeline). All five constraints are satisfiable:

1. **Primitives-only derivative** — feasible, direct precedent exists
   (`so101_provenance.rs`). Work item: hand-derive
   `go1_primitives.xml` from `go1_mjx_feetonly.xml` — collision geoms carry over unchanged (11
   primitive geoms per leg × 4 + 3 on the trunk = 47), replace the 5 STL visual meshes with
   matching box/capsule/cylinder approximations (same pattern as the SO-101 derivative),
   provenance-checked against menagerie commit `1b86ece` the way `so101_provenance.rs` checks
   against its pin.
2. **Observation history** — trivial: Go1's own `history_len=1` needs no stacking at all;
   Observation IR's `TemporalWindow(n=1, stride, align)` (§7.5) represents it exactly, and the
   same node type covers `history_len=3` if a later env (Spot/H1) needs it.
3. **MLP policy in Learning IR** — feasible: `StateEncoder { kind: Mlp, out_dim }` (§8.3) →
   `RegressionHead` reproduces the 3-hidden-layer swish MLP + tanh-squashed mean exactly; we
   already verify MLP-graph torch equivalence per CLAUDE.md's oracle rule. Work item: export
   `hidden_0..3` kernel/bias (transpose to `[out, in]`) and the `mean`/`std` observation
   normalizer from the brax checkpoint into a `safetensors` file our lowering can read (no
   pickle, INV-16) — no new node type needed if the normalizer is folded into the Observation
   IR's existing `Normalize` op.
4. **Deployment IR control rate** — feasible: `ctrl_dt=0.02` (50 Hz) is exactly what the real
   Go1 deployment in the paper also runs at (Appendix C.5: "inferenced at 50 Hz"). Work item:
   set our Deployment IR's inference rate to 50 Hz, chunk size 1 (Go1 is a reactive policy, no
   action chunking upstream).
5. **MuJoCo CPU 3.13 subprocess** — feasible, `mujoco>=3.6.0` is Playground's own floor, our
   pin `mujoco==3.13.0` (`docs/api-notes/mujoco.md`) satisfies it; the Go1 MJCF loads under
   plain MuJoCo (it is standard MJCF, MJX support is additive).

**Risks — physics parity between MJX training and our MuJoCo CPU stepping.** Must match: the
training XML's own solver block, `iterations=1 ls_iterations=5 integrator=Euler
timestep=0.004`, cone `pyramidal` (unspecified = default) — running our CPU backend at a
*higher* iteration count than what the policy trained under changes contact resolution the
policy never saw, risking foot-slip/jitter mismatches even though MuJoCo CPU and MJX share the
same contact math; `sim_dt=0.004` with 5 substeps per 0.02 s control tick must be preserved
exactly, not approximated by a single larger step; friction (`geom_friction`, `frictionloss`,
`armature`) must use the *undomain-randomized* nominal values (mid-range of the `randomize.py`
bounds) as our deployment defaults, since DR trains for a distribution, not a mean, and our
one fixed physics config is one sample from it; the `home` keyframe (default standing pose)
must be copied byte-for-byte, since `default_pose` is the action's zero-point. None of this is
exercised until we actually load the primitives derivative in our MuJoCo CPU backend and step
a policy — first oracle to write, before any training run.
