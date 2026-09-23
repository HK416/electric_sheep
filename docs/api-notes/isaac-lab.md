# Isaac Lab — ManagerBasedRLEnv conventions, the reach task, rsl_rl checkpoints

Companion to `docs/api-notes/isaac-sim.md`, same purpose: prep for **M11 X2**
(`docs/packets/M11/X2-adapter-v2.md`), which already names these exact conventions —
`joint_pos_rel`/`joint_vel_rel`, per-term `scale`, `last_action`, `history_length`/`clip`,
joints "resolved by name in the articulation's order", `decimation × sim.dt` — as what a v2
adapter must express declaratively. Nothing in this workspace runs Isaac Lab; every claim below
is `verified (fetched)` (read from the cited page/source file on 2026-09-23) or `unverified`.
Korean sibling: `isaac-lab.ko.md`.

Sources fetched 2026-09-23:

- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html>
- <https://github.com/isaac-sim/IsaacLab/releases>
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py` (raw, `main` branch)
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/joint_pos_env_cfg.py` (raw, `main` branch)
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py` (raw, `main` branch)
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py` (raw, `main` branch)
- `source/isaaclab/isaaclab/utils/string.py` (`resolve_matching_names`, raw, `main` branch)
- <https://github.com/leggedrobotics/rsl_rl> (`algorithms/ppo.py`, `runners/on_policy_runner.py`, raw, `main` branch)
- <https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html>

Everything pulled from a repo's `main` branch is dated 2026-09-23 but not pinned to a release
tag; treat exact line numbers as approximate and re-check against the pinned version below
before depending on them byte-for-byte.

## 1. Version, install, headless training

| | | status |
|---|---|---|
| pinned version | **Isaac Lab 2.3.2** (last release before the 3.0 Early Access line; released ~February 2026), paired with **Isaac Sim 5.1** | verified (fetched, release list) — see `docs/api-notes/isaac-sim.md` §1 for why 3.0.0-EA (Isaac Sim 6.1) is not pinned |
| pip route | `pip install isaaclab[isaacsim,all]==2.3.2.post1 --extra-index-url https://pypi.nvidia.com` | verified (fetched) |
| pip caveat | bundles Isaac Sim itself (`[isaacsim,...]` extra) but **ships no standalone scripts** — `train.py`/`play.py` are not part of the pip package, only the library; running the reach task's `rsl_rl/train.py` as shown below needs the git checkout (`isaaclab.sh`), not the pip install | verified (fetched) |
| repo route | `git clone` + `./isaaclab.sh -i rsl_rl` (installs the `rsl-rl-lib` extra), then `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <task-id> --headless` | verified (fetched, doc example uses this exact form) |
| Python | 3.11 (Isaac Sim 5.x pairing) | verified (fetched) |

## 2. `ManagerBasedRLEnv` observation conventions

`isaaclab.envs.mdp.observations` (`verified (fetched)`, names and behavior from the module and
its call sites):

| term | formula | notes |
|---|---|---|
| `mdp.joint_pos_rel` | `joint_pos − default_joint_pos` | `default_joint_pos` is the articulation's configured rest pose; only joints in `asset_cfg.joint_ids` are returned |
| `mdp.joint_vel_rel` | `joint_vel − default_joint_vel` | same joint-subset rule |
| `mdp.last_action` | the previous control-tick's **raw** (pre-scale/offset) action the policy emitted | sourced from the action manager, not from `data.ctrl`-equivalent applied targets |
| `mdp.generated_commands` | the current value of a named command term (e.g. `ee_pose`) | used by the reach task for its goal pose |

`ObservationTermCfg` (`ObsTerm`) fields, **verified (fetched)**: `func`, `params`, `noise`
(a noise-model instance, e.g. `Unoise(n_min, n_max)`, applied only if the group's
`enable_corruption = True`), `clip` (applied after noise), `scale` (applied after clip),
`history_length` + `flatten_history_dim` (a per-term ring buffer; when set, the term's shape
gains a history axis that is flattened into the trailing dimension if
`flatten_history_dim=True`). The pipeline order, per the observation manager: **compute → custom
modifiers → noise/corruption → clip → scale**.

**Concatenation order**: an `ObsGroup` (e.g. `PolicyCfg`) with `concatenate_terms = True`
concatenates its terms in **declaration order** — the order the fields are written in the
`@configclass`, not a sorted or alphabetical order. The reach task's policy group (§4) is
`joint_pos, joint_vel, pose_command, actions`, in that literal order, so the flat observation
vector is `[joint_pos_rel(6 or N) | joint_vel_rel(N) | pose_command(7) | last_action(N)]`.

**Joint ordering** — the one M11 X2 explicitly needs, pinned precisely:
`isaaclab.utils.string.resolve_matching_names(keys, target_names, preserve_order=False)` is
what `SceneEntityCfg`/action terms use to turn a `joint_names` regex list into concrete
indices, and its docstring is explicit: with `preserve_order=False` (**the default, and what
`JointPositionActionCfg` uses**), "the ordering of the matched indices and names is the same as
the order of the provided list of strings" — i.e. **the target list's own order** (the
articulation's internal joint order, however the asset/USD authored it), *not* the order the
regex patterns are written in the config. `preserve_order=True` would instead follow the regex
list's order; the reach task does not set it, so it stays at the default. **verified (fetched,
docstring + worked example)**: `['a','b','c','d','e']` matched by `['a|c', 'b']` under
`preserve_order=False` returns `([0,1,2], ['a','b','c'])` — target order, not pattern order.

## 3. `JointPositionActionCfg`

**verified (fetched)**, `isaaclab.envs.mdp.actions.actions_cfg` + `JointAction.process_actions`:

```
processed_actions = raw_actions * scale + offset
```

- `scale`: float or per-joint-regex dict, default `1.0`.
- `use_default_offset`: bool, default `True`. When `True`, `offset` is **overwritten** with the
  articulation's `default_joint_pos` at env creation — i.e. the action is a *delta around the
  rest pose* scaled by `scale`, matching `mdp.joint_pos_rel`'s own centering. When `False`,
  `offset` is whatever was explicitly configured (default `0.0`), making the action an absolute
  scaled target.
- `processed_actions` is what is written to the joint position targets (subject to the
  articulation's own PD drive, not a direct `qpos` write).

Franka reach task's concrete instance (`config/franka/joint_pos_env_cfg.py`, verified fetched):
`JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5,
use_default_offset=True)`.

## 4. The reach task, verbatim structure (`reach_env_cfg.py` + Franka override)

Base `ReachEnvCfg.__post_init__` (**verified fetched**): `decimation = 2`, `sim.dt = 1/60`
(≈16.667 ms physics step), `sim.render_interval = decimation`, `episode_length_s = 12.0`. So the
control period is `decimation * sim.dt = 1/30 s` (~33.3 ms, ~30 Hz), and an episode is `12.0 /
(1/30) = 360` control steps. (Contrast M8's brax/Playground SO-101 source policy,
`docs/api-notes/brax-ppo-so101.md` §2, which runs 50 Hz control over 200 Hz physics — a
different project's numbers, not Isaac Lab's; the two are not interchangeable without the
adapter's `[timing] policy_dt` check M11 X2 already plans, `X2-adapter-v2.md:39`.)

```python
# ObservationsCfg.PolicyCfg — declaration order is the concatenation order
joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
actions = ObsTerm(func=mdp.last_action)
# __post_init__: enable_corruption = True, concatenate_terms = True

# ActionsCfg (base is abstract; Franka fills it in)
arm_action: ActionTerm = MISSING
gripper_action: ActionTerm | None = None
# Franka: arm_action = JointPositionActionCfg(asset_name="robot",
#   joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)

# RewardsCfg
end_effector_position_tracking        = RewTerm(mdp.position_command_error,       weight=-0.2)
end_effector_position_tracking_fine   = RewTerm(mdp.position_command_error_tanh,  weight=0.1, params={"std": 0.1})
end_effector_orientation_tracking     = RewTerm(mdp.orientation_command_error,    weight=-0.1)
action_rate                            = RewTerm(mdp.action_rate_l2,               weight=-0.0001)
joint_vel                              = RewTerm(mdp.joint_vel_l2,                 weight=-0.0001)

# TerminationsCfg
time_out = DoneTerm(mdp.time_out, time_out=True)   # no early termination on failure

# EventCfg — episode reset
reset_robot_joints = EventTerm(mdp.reset_joints_by_scale, mode="reset",
                                params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)})
```

All **verified (fetched)** from the two source files. The Franka override names
`body_names="panda_hand"` as the tracked end-effector body in all three tracking reward terms
and the command generator. `FrankaReachEnvCfg_PLAY` (the eval/play variant) sets
`enable_corruption = False` on the observation group (no noise at eval time) and reduces env
count — the shape a mirrored SO-101 reach task's `-Play-v0` variant would copy.

**Mirroring this for an SO-101 reach task** (not built here, just the shape to reuse): same
five `RewardsCfg` terms with SO-101's link names, same `ObservationsCfg.PolicyCfg` four terms,
`JointPositionActionCfg(asset_name="robot", joint_names=[<SO-101's 6 joint names or a regex>],
scale=<tbd>, use_default_offset=True)`, same `time_out`-only termination.

## 5. rsl_rl wrapper: `clip_actions`, observation normalization

`RslRlVecEnvWrapper` (`isaaclab_rl.rsl_rl.vecenv_wrapper`, **verified fetched**): takes
`clip_actions: float | None`; when set, `torch.clamp(actions, -clip_actions, clip_actions)` is
applied every `step()`, and the action space bounds are rewritten to match. Observations come
from `self.unwrapped.observation_manager.compute()` (already scaled/clipped/noised per §2) —
**the wrapper itself does no additional normalization**; empirical (running-mean/std)
normalization is a *policy-side* concern:

- `RslRlOnPolicyRunnerCfg.empirical_normalization: bool` (**verified fetched, name and
  behavior** — search corroborated by multiple issue threads) turns on an `EmpiricalNormalization`
  module inside rsl_rl's `OnPolicyRunner`/`PPO`, which maintains a running mean/std over
  observations and normalizes them before the actor/critic ever see them (distinct from Isaac
  Lab's own per-`ObsTerm` `scale`, which is a fixed constant, not learned).
- rsl_rl ≥ 4.0 additionally requires an `obs_groups` mapping (e.g. `{"actor": ["policy"],
  "critic": ["policy"]}`) in the runner config or `OnPolicyRunner.__init__` hangs — a
  **verified (fetched, GitHub issue)** version-compatibility trap between Isaac Lab and a
  too-new/too-old `rsl_rl` pip package.

## 6. Export: `policy.pt` / `policy.onnx`

`isaaclab_rl.rsl_rl.exporter` (**verified fetched**, `_TorchPolicyExporter` /
`_OnnxPolicyExporter`): both deep-copy the trained `actor` module and the empirical normalizer
(or `torch.nn.Identity()` if `empirical_normalization=False`), and **the exported artifact
composes them**: `forward(obs) = actor(normalizer(obs))`. **The normalizer is baked into the
exported `policy.pt`/`policy.onnx`** — a deployment reading the exported file does not need to
know normalization stats separately; it needs only the raw (Isaac-Lab-scaled/clipped) observation
vector. Recurrent policies (`memory_a.rnn`) export with explicit hidden/cell-state
inputs/outputs (LSTM: `(obs, h_in, c_in) → (actions, h_out, c_out)`; GRU: `(obs, h_in) →
(actions, h_out)`); the reach task's default MLP policy is non-recurrent.

## 7. Checkpoint format — version-dependent, both confirmed

**verified (fetched)**, `leggedrobotics/rsl_rl` source, two shapes depending on the pinned
`rsl_rl` package version (Isaac Lab 2.3.2's own pin was not independently re-fetched — confirm
against that release's `requirements`/`setup.py` before trusting one over the other for a given
checkpoint file):

- **Classic (`rsl_rl` 2.x, the format most existing published checkpoints use)**: a single
  combined `model_state_dict` from one `ActorCritic` module's `state_dict()`, with keys prefixed
  `actor.<layer>.weight`/`.bias`, `critic.<layer>.weight`/`.bias`, and a top-level `std` (or
  `log_std`) parameter for the Gaussian policy's action noise. `unverified` here beyond
  corroborating search results — not independently re-derived from a 2.x tag's source in this
  pass.
- **Current `main` / rsl_rl ≥ 4.0–5.0**: `PPO.save()` returns separate
  `actor_state_dict` / `critic_state_dict` (from `self._raw_actor.state_dict()` /
  `self._raw_critic.state_dict()`) plus `optimizer_state_dict`, and conditionally
  `rnd_state_dict`/`rnd_optimizer_state_dict` if Random Network Distillation is configured. The
  `OnPolicyRunner.save()` wraps this with `iter` (current learning iteration) and `infos`
  before `torch.save`. **No observation-normalizer state was found in the checkpoint dict** —
  normalization stats live on the `actor`/`critic` modules themselves (an `EmpiricalNormalization`
  submodule saved as part of their own `state_dict()`), which is consistent with §6's exporter
  reading the normalizer straight off the loaded actor rather than from a separate key.
- This split is a **known breaking change**: "rsl-rl ≥ 5.0 expects separate `actor_state_dict`
  and `critic_state_dict` entries, so published pretrained checkpoints shipped with older asset
  releases fail to load with `KeyError: 'actor_state_dict'`" — **verified (fetched, GitHub PR/
  issue discussion)**. Any importer M11 X2 builds for a real rsl_rl checkpoint must branch on
  which shape the file has (`"model_state_dict" in ckpt` vs. `"actor_state_dict" in ckpt"`)
  rather than assume one.

## 8. What we don't know yet

- Exact `rsl_rl` version Isaac Lab 2.3.2 pins (classic vs. split checkpoint shape) —
  unverified in this pass, needed before M11 X2 commits to one checkpoint reader.
- Whether SO-101's joint names/count need a custom `joint_names` regex or can use one
  catch-all pattern the way Franka's `"panda_joint.*"` does — depends on the SO-101 USD asset's
  authored joint names, not fetched here (this project's own MJCF is the source of truth, not an
  Isaac asset; `docs/api-notes/mujoco.md`'s Menagerie section already pins `so101.xml`'s joint
  names for MuJoCo — an Isaac-side USD conversion could rename them, per
  `docs/api-notes/isaac-sim.md` §3's MJCF-importer caveats).
- Precise field names/defaults for `PhysxCfg` iteration counts — deferred to
  `docs/api-notes/isaac-sim.md` §5, same gap.
