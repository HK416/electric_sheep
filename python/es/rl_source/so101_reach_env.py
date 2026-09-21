"""SO-101 reach task on MJX -- the source-side env of packet M8/S2c.

The task is defined once, in `docs/design/rl-continuation.md` section 5, and is shared with
our own trainer (S4b) and one Evaluation IR (S4c):

  * scene   `tests/fixtures/mjcf/so101_pick_place.xml`, unmodified (MJX loads it as committed,
            see `docs/api-notes/brax-ppo-so101.md` section 3);
  * control 50 Hz over 200 Hz physics (`ctrl_dt = 0.02`, `sim_dt = 0.005`, 4 substeps);
  * obs     26 = joint_pos[6] || joint_vel[6] || cube_pose[7] || gripper_pose[7], each pose
            world-frame `pos[3] || quat[4]` with the quaternion in **xyzw** order (spec 3.1;
            MuJoCo's `xquat` is wxyz and is reordered here), gripper = the *body* `gripper`;
  * action  6 normalized position targets, `ctrl = centre + half_range * clip(a, -1, 1)`;
  * reward  -dist + 1 on success, `dist = ||cube_pos - gripper_body_pos||` (positions only);
            success = dist < 0.03 m; episode = 200 control steps.

The per-episode cube draw reproduces the committed Task IR's `Randomization` nodes
(`tests/fixtures/visible-learning/task.toml` nodes 28-30): x ~ U(0.21, 0.27), y ~ U(-0.03,
0.05), z = 0.02, every arm joint reset to 0.

Run `python so101_reach_env.py` for the self-check: it is also the MJX-compatibility
measurement (MJX vs MuJoCo CPU over one 200-step episode) quoted in the api-note.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

# Joint order as the MJCF declares it; the actuators are in the same order.
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CUBE_X = (0.21, 0.27)  # task.toml node 29, stream `cube.x`, target `qpos[6]`
CUBE_Y = (-0.03, 0.05)  # task.toml node 30, stream `cube.y`, target `qpos[7]`
SCENE_NAME = "so101_pick_place.xml"

OBS_DIM = 26
QUAT_XYZW = np.array([1, 2, 3, 0])  # MuJoCo stores wxyz
CUBE_POS = slice(12, 15)
CUBE_QUAT = slice(15, 19)
GRIPPER_POS = slice(19, 22)
GRIPPER_QUAT = slice(22, 26)
OBS_LAYOUT = (
    ("joint_pos", 0, 6, "rad"),
    ("joint_vel", 6, 6, "rad/s"),
    ("cube_pos", 12, 3, "m"),
    ("cube_quat_xyzw", 15, 4, "unit quaternion, xyzw"),
    ("gripper_pos", 19, 3, "m"),
    ("gripper_quat_xyzw", 22, 4, "unit quaternion, xyzw"),
)


def scene_path(explicit: str | None = None) -> Path:
    """An explicit scene, else the committed fixture (the server mirrors the repo layout)."""
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "mjcf" / SCENE_NAME


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=200,
        success_dist=0.03,
        success_bonus=1.0,
    )


class SO101Reach(mjx_env.MjxEnv):
    """Reach the cube with the gripper frame."""

    def __init__(self, xml_path=None, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        self._xml_path = str(scene_path(xml_path))
        mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        mj_model.opt.timestep = self._config.sim_dt
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)

        jid = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        if any(i < 0 for i in jid):
            raise ValueError(f"missing joint in {self._xml_path}: {JOINTS}")
        self._qpos_adr = np.array([mj_model.jnt_qposadr[i] for i in jid])
        self._qvel_adr = np.array([mj_model.jnt_dofadr[i] for i in jid])
        cube_joint = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self._cube_qadr = int(mj_model.jnt_qposadr[cube_joint])
        self._cube_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._gripper_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

        lo, hi = mj_model.actuator_ctrlrange[:, 0], mj_model.actuator_ctrlrange[:, 1]
        self._act_offset = jp.asarray((hi + lo) / 2.0)
        self._act_scale = jp.asarray((hi - lo) / 2.0)
        self._init_q = jp.asarray(mj_model.qpos0)

    # -- task -----------------------------------------------------------------

    def _obs(self, data) -> jax.Array:
        # Poses go in whole, as the Observation IR binds them: pos[3] || quat[4], and the
        # quaternion in xyzw (spec 3.1) where MuJoCo's `xquat` is wxyz.
        return jp.concatenate([
            data.qpos[self._qpos_adr],
            data.qvel[self._qvel_adr],
            data.xpos[self._cube_body],
            data.xquat[self._cube_body][QUAT_XYZW],
            data.xpos[self._gripper_body],
            data.xquat[self._gripper_body][QUAT_XYZW],
        ])

    @staticmethod
    def _dist(obs: jax.Array) -> jax.Array:
        """Gripper-to-cube distance: the two world positions inside the observation."""
        return jp.linalg.norm(obs[CUBE_POS] - obs[GRIPPER_POS])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, kx, ky = jax.random.split(rng, 3)
        qpos = self._init_q.at[self._cube_qadr].set(
            jax.random.uniform(kx, minval=CUBE_X[0], maxval=CUBE_X[1])
        )
        qpos = qpos.at[self._cube_qadr + 1].set(
            jax.random.uniform(ky, minval=CUBE_Y[0], maxval=CUBE_Y[1])
        )
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=jp.zeros(self._mjx_model.nv),
            ctrl=jp.zeros(self._mjx_model.nu),
        )
        data = mjx.forward(self._mjx_model, data)
        obs = self._obs(data)
        metrics = {"dist": self._dist(obs), "success": jp.zeros(())}
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, {"rng": rng})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        ctrl = self._act_offset + self._act_scale * jp.clip(action, -1.0, 1.0)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        obs = self._obs(data)
        dist = self._dist(obs)
        success = (dist < self._config.success_dist).astype(jp.float32)
        reward = -dist + success * self._config.success_bonus
        state.metrics.update(dist=dist, success=success)
        # No early termination: the episode ends on the wrapper's 200-step timeout only.
        return state.replace(data=data, obs=obs, reward=reward, done=jp.zeros(()))

    # -- MjxEnv ---------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def _bench(xml: str | None, num_envs: int, steps: int = 100) -> None:
    """Batched throughput -- the number that decides the training budget."""
    env = SO101Reach(xml)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    t0 = time.time()
    state = reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
    jax.block_until_ready(state.obs)
    print(f"compile reset ({num_envs} envs): {time.time() - t0:.1f} s", flush=True)
    action = jp.zeros((num_envs, env.action_size))
    t0 = time.time()
    state = step(state, action)
    jax.block_until_ready(state.obs)
    print(f"compile step ({num_envs} envs): {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    for _ in range(steps):
        state = step(state, action)
    jax.block_until_ready(state.obs)
    dt = time.time() - t0
    print(
        f"bench {num_envs} envs x {steps} control steps: {num_envs * steps / dt:.0f} "
        f"env-steps/s ({dt:.1f} s)",
        flush=True,
    )


def _self_check(xml: str | None, num_steps: int) -> None:
    t0 = time.time()
    env = SO101Reach(xml)
    m = env.mj_model
    print(f"scene {env.xml_path}", flush=True)
    print(
        f"nq={m.nq} nv={m.nv} nu={m.nu} integrator={int(m.opt.integrator)} "
        f"cone={int(m.opt.cone)} iterations={m.opt.iterations} "
        f"ls_iterations={m.opt.ls_iterations} substeps={env.n_substeps} "
        f"act={env.action_size} put_model_s={time.time() - t0:.1f}",
        flush=True,
    )
    t0 = time.time()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state.obs.block_until_ready()
    print(f"compile reset: {time.time() - t0:.1f} s", flush=True)
    assert state.obs.shape == (OBS_DIM,), state.obs.shape
    assert abs(float(jp.linalg.norm(state.obs[CUBE_QUAT])) - 1.0) < 1e-5
    x = float(state.data.qpos[env._cube_qadr])
    y = float(state.data.qpos[env._cube_qadr + 1])
    assert CUBE_X[0] <= x <= CUBE_X[1] and CUBE_Y[0] <= y <= CUBE_Y[1], (x, y)

    # Action mapping: +-1 must land exactly on the actuator's ctrlrange.
    lo = np.asarray(m.actuator_ctrlrange[:, 0])
    hi = np.asarray(m.actuator_ctrlrange[:, 1])
    got_hi = np.asarray(env._act_offset + env._act_scale * jp.ones(m.nu))
    got_lo = np.asarray(env._act_offset - env._act_scale * jp.ones(m.nu))
    assert np.allclose(got_hi, hi, atol=1e-6) and np.allclose(got_lo, lo, atol=1e-6)

    # MJX vs MuJoCo CPU over one 200-step episode, same start state, same ctrl sequence.
    # When `--xml` is a derived scene, the committed scene is stepped on CPU too, so the
    # backend gap (MJX vs CPU) and the edit's own gap (CPU vs CPU) are separate numbers.
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.3, 0.3, size=(num_steps, m.nu))
    data = mujoco.MjData(m)
    data.qpos[:] = np.asarray(state.data.qpos)
    data.qvel[:] = 0.0
    mujoco.mj_forward(m, data)
    ref = None
    if Path(env.xml_path) != scene_path(None):
        ref_model = mujoco.MjModel.from_xml_path(str(scene_path(None)))
        ref_model.opt.timestep = env.sim_dt
        ref = mujoco.MjData(ref_model)
        ref.qpos[:] = np.asarray(state.data.qpos)
        ref.qvel[:] = 0.0
        mujoco.mj_forward(ref_model, ref)
    step = jax.jit(env.step)
    t0 = time.time()
    warm = step(state, jp.zeros(m.nu))
    warm.obs.block_until_ready()
    print(f"compile step: {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    worst_q = 0.0
    worst_obs = 0.0
    worst_edit = 0.0
    trace = {}
    for i, a in enumerate(actions):
        state = step(state, jp.asarray(a))
        ctrl = np.asarray(env._act_offset + env._act_scale * np.clip(a, -1, 1))
        data.ctrl[:] = ctrl
        for _ in range(env.n_substeps):
            mujoco.mj_step(m, data)
        if ref is not None:
            ref.ctrl[:] = ctrl
            for _ in range(env.n_substeps):
                mujoco.mj_step(ref_model, ref)
            worst_edit = max(worst_edit, float(np.abs(ref.qpos - data.qpos).max()))
        worst_q = max(worst_q, float(np.abs(np.asarray(state.data.qpos) - data.qpos).max()))
        cpu_obs = np.concatenate([
            data.qpos[env._qpos_adr],
            data.qvel[env._qvel_adr],
            data.xpos[env._cube_body],
            data.xquat[env._cube_body][QUAT_XYZW],
            data.xpos[env._gripper_body],
            data.xquat[env._gripper_body][QUAT_XYZW],
        ])
        worst_obs = max(worst_obs, float(np.abs(np.asarray(state.obs) - cpu_obs).max()))
        if i + 1 in (1, 10, 50):
            trace[i + 1] = (worst_q, worst_edit)
    dt = time.time() - t0
    for n, (q, e) in trace.items():
        print(f"after {n:3d} steps: mjx-vs-cpu {q:.3e}  edited-vs-committed {e:.3e}", flush=True)
    print(f"mjx-vs-cpu max|dqpos|={worst_q:.3e} max|dobs|={worst_obs:.3e}", flush=True)
    if ref is not None:
        print(f"cpu-edited-vs-cpu-committed max|dqpos|={worst_edit:.3e}", flush=True)
    print(f"1-env loop: {len(actions) / dt:.1f} control steps/s ({dt:.1f} s)", flush=True)
    print(f"final dist={float(state.metrics['dist']):.4f} reward={float(state.reward):.4f}")
    print("self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MJX SO-101 reach env self-check")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--steps", type=int, default=200, help="control steps compared vs CPU")
    ap.add_argument("--bench", type=int, default=0, help="batched throughput, N envs")
    args = ap.parse_args()
    if args.bench:
        _bench(args.xml, args.bench)
    else:
        _self_check(args.xml, args.steps)
