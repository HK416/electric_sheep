"""The SO-101 reach task as an Isaac Lab manager-based env (packet M11/I3, spec 28.14 rule 2).

Every term mirrors `tests/fixtures/rl/task-reach-last-action.toml`; nothing here is tuned to a
number our runtime produced. Import only after `isaaclab.app.AppLauncher` has started Kit.

| Task IR | here |
|---|---|
| scene `so101_pick_place.xml` | the USD `main.py build-usd` writes from the MJCF `scene_to_mjcf` emits, through `physx_ref.import_scene` |
| 50 Hz control over the MJCF's 5 ms step | `decimation = 4`, `sim.dt = 0.005` |
| `joint_pos` / `joint_vel` channels | `mdp.joint_pos_rel` / `mdp.joint_vel_rel` (scale 0.05), folded back by the adapter |
| `cube_pose`: `JointState { cube, dof 7 }`, the free joint's `qpos` -- `pos | quat` **w-first** | `free_joint_qpos`: the root frame minus the env origin, then Isaac's own w-first quaternion |
| `gripper_pose`: `BodyPose(gripper)`, `xpos | xquat` -- quaternion **x-first** (spec 3.1) | `body_pose_xyzw`: the body frame minus the env origin, x-first |
| `last_action` (`PreviousAction`, `initial` = rest pose) | `mdp.last_action` (the raw action, zero at reset), with `default_joint_pos` = that pose |
| `ActionSpec JointPosition`, MuJoCo clamps ctrl to `ctrlrange` | `JointPositionActionCfg(scale 0.5, use_default_offset)`, `clip` = `ctrlrange` |
| reward `-1 * dist` + `1 * (dist < 0.03)` per control step | the same terms at weight `-1/dt`, `+1/dt`: Isaac Lab multiplies every term by `dt` |
| `Terminate Success` / `Timeout` at 4.0 s | `DoneTerm(reached)`, `DoneTerm(mdp.time_out, time_out=True)`, `episode_length_s = 4.0` |
| `ResetState` joints = 0; cube x U[0.21, 0.27], y U[-0.03, 0.05], z 0.02 | `reset_joints_by_scale(0, 0)`; `reset_root_state_uniform` around (0.24, 0.01, 0.02) +- (0.03, 0.04) |
| position servo kp, kv, forcerange; joint damping 0.6 | `ImplicitActuator` stiffness kp, damping kv, effort limit; `-0.6 q'` as an explicit effort before every physics step, as `physx_ref.py` applies it |
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# `task-reach-last-action.toml`'s `PreviousAction.initial`, our order.
DEFAULT_POS = [0.1, -0.4, 0.7, 0.25, -0.05, 0.3]
# so101_pick_place.xml `<position ctrlrange>`, per actuator.
CTRLRANGE = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.17453, 1.74533),
}
# class sts3215: `<position kp kv forcerange>`, `<joint damping armature>`.
KP, KV, FORCE, JOINT_DAMPING = 998.22, 2.731, 2.94, 0.60
SIM_DT, DECIMATION = 0.005, 4
STEP_DT = SIM_DT * DECIMATION
SUCCESS_M = 0.03
ACTION_SCALE = 0.5
JOINT_VEL_SCALE = 0.05


# --- the actuator: an implicit PD drive plus MuJoCo's passive joint damping ------------------


class PassiveDampedImplicitActuator(ImplicitActuator):
    """PhysX's implicit drive (stiffness kp, damping kv, max force = forcerange) plus
    `-d * q'` as a feed-forward joint effort, recomputed before every physics step
    (`Articulation.write_data_to_sim` runs the actuator model once per substep) -- outside the
    drive's force clamp, as MuJoCo applies passive damping and as `physx_ref.py` does."""

    def compute(self, control_action, joint_pos, joint_vel):
        control_action = super().compute(control_action, joint_pos, joint_vel)
        control_action.joint_efforts = -self.cfg.passive_damping * joint_vel
        return control_action


@configclass
class PassiveDampedImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type: type = PassiveDampedImplicitActuator
    passive_damping: float = 0.0


# --- observation, reward and termination terms ------------------------------------------------


def _pose(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None or isinstance(asset_cfg.body_ids, slice):
        pos, quat = asset.data.root_link_pos_w, asset.data.root_link_quat_w
    else:
        pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0]]
        quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
    return pos - env.scene.env_origins, quat


def free_joint_qpos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """A free body's `qpos` in MuJoCo's layout, `pos[3] | quat[4]` with the quaternion
    **w-first** -- what the Task IR's `cube_pose` channel (`JointState { cube, dof = 7 }`) is
    served from (`es_eval::runner::Capture::Qpos` reads the free joint's `qpos` raw). Not the
    x-first order of a `BodyPose` channel: measured in M11/I3, a first set of policies trained
    on x-first here scored 0.0 in our runtime against 0.63 on Isaac's side."""
    pos, quat = _pose(env, asset_cfg)
    return torch.cat([pos, quat], dim=-1)


def body_pose_xyzw(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The body frame's pose in the env frame, `pos[3] | quat[4]` x-first (spec 3.1) -- the
    Task IR's `GetBodyPose` / `BodyPose` channel. The robot base is at the env origin with the
    identity rotation, so this is also the base frame."""
    pos, quat = _pose(env, asset_cfg)
    return torch.cat([pos, quat[:, 1:4], quat[:, 0:1]], dim=-1)


def distance(env, cube_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """`Norm{L2}(cube_pos - gripper_pos)`, the Task IR's node 12."""
    return torch.linalg.norm(_pose(env, cube_cfg)[0] - _pose(env, gripper_cfg)[0], dim=-1)


def reached(env, cube_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """`Compare{Lt 0.03}`, node 15: the success reward and the success termination."""
    return distance(env, cube_cfg, gripper_cfg) < SUCCESS_M


CUBE = SceneEntityCfg("cube")
GRIPPER = SceneEntityCfg("robot", body_names=["gripper"])
PAIR = {"cube_cfg": CUBE, "gripper_cfg": GRIPPER}


# --- the env ----------------------------------------------------------------------------------


def make_env_cfg(usd_path: str, cube_path: str, num_envs: int, seed: int, device: str):
    """`usd_path` / `cube_path` come from `build-usd`'s `scene.json`: the robot USD and the cube
    body's prim path below the robot prim."""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=dict(zip(JOINTS, DEFAULT_POS)),
                joint_vel={".*": 0.0},
            ),
            actuators={
                "sts3215": PassiveDampedImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=KP,
                    damping=KV,
                    effort_limit_sim=FORCE,
                    passive_damping=JOINT_DAMPING,
                ),
            },
        )
        # The cube is a body of the same USD (the MJCF's free body), not a second spawn.
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + cube_path,
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.01, 0.02), rot=(1.0, 0.0, 0.0, 0.0)),
        )

    @configclass
    class ActionsCfg:
        arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=ACTION_SCALE,
            use_default_offset=True,
            clip=dict(CTRLRANGE),
        )

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            # Declaration order is the concatenation order (isaac-lab.md section 2).
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=JOINT_VEL_SCALE)
            cube_pose = ObsTerm(func=free_joint_qpos, params={"asset_cfg": CUBE})
            gripper_pose = ObsTerm(func=body_pose_xyzw, params={"asset_cfg": GRIPPER})
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class RewardsCfg:
        # Isaac Lab's reward manager multiplies every term by `step_dt`; the weights undo it so
        # a control step earns what the Task IR's `Reward` nodes give it.
        reach_distance = RewTerm(func=distance, weight=-1.0 / STEP_DT, params=PAIR)
        reach_success = RewTerm(func=reached, weight=1.0 / STEP_DT, params=PAIR)

    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        success = DoneTerm(func=reached, params=PAIR)

    @configclass
    class EventCfg:
        reset_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
        )
        reset_cube = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.03, 0.03), "y": (-0.04, 0.04)},
                "velocity_range": {},
                "asset_cfg": CUBE,
            },
        )

    @configclass
    class So101ReachEnvCfg(ManagerBasedRLEnvCfg):
        # Every env at one origin, collisions filtered between envs -- `physx_ref.py`'s
        # `GridCloner(spacing = 0)`. Also forced: the importer welds the base to the world at
        # the MJCF's own coordinates, so a spaced env's arm is pulled back to the origin.
        scene: SceneCfg = SceneCfg(num_envs=num_envs, env_spacing=0.0)
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventCfg = EventCfg()

        def __post_init__(self):
            self.decimation = DECIMATION
            self.sim.dt = SIM_DT
            self.sim.render_interval = DECIMATION
            self.sim.gravity = (0.0, 0.0, -9.81)
            self.sim.device = device
            self.episode_length_s = 4.0
            self.seed = seed

    return So101ReachEnvCfg()


def make_runner_cfg(seed: int, max_iterations: int, device: str):
    """Isaac Lab's own reach runner config (`FrankaReachPPORunnerCfg`, reach/config/franka/
    agents/rsl_rl_ppo_cfg.py in isaaclab 2.3.2.post1), copied field for field; only the seed,
    the budget, the device and the observation-group map (which rsl-rl-lib 3.x requires) are
    set here."""
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

    @configclass
    class So101ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 24
        max_iterations = 1000
        save_interval = 50
        experiment_name = "so101_reach"
        run_name = ""
        obs_groups = {"policy": ["policy"], "critic": ["policy"]}
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64],
            activation="elu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=8,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

    cfg = So101ReachPPORunnerCfg()
    cfg.seed = seed
    cfg.max_iterations = max_iterations
    cfg.device = device
    return cfg
