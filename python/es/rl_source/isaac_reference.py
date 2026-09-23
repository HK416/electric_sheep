"""Oracle 2 of packet M11/X2: two synthetic source policies and what their frameworks would do.

    isaac_reference.py --make isaac-classic|isaac-split|playground --out <dir> [--rows 256]

Writes, under <dir>:

* the source in its framework's **native** layout -- an rsl_rl checkpoint (`native/model_0.pt`,
  either the classic `model_state_dict` + `obs_norm_state_dict` shape or the >= 5.0
  `actor_state_dict` shape, `isaac-lab.md` section 7) with Isaac Lab's `params/env.yaml`, or a
  brax export (`native/source.npz` + `meta.json`, `brax-ppo-so101.md` section 6) with a
  Playground config;
* `import_args.json` -- the `import_rl.py` arguments that read it;
* `states.json` -- `rows` states of **our** robot, by our Task IR channel names, in our joint
  order and our units, including the previous action row in actuator units;
* `reference.json` -- for each state, the action the source framework would command, in our
  actuator order and units, computed below in NumPy (float64).

THE REFERENCE IS WRITTEN FROM THE API NOTES, NOT FROM OUR IMPORTER. Nothing here reads
`crates/es-import` or the adapter: the observation is assembled term by term the way
`docs/api-notes/isaac-lab.md` sections 2-4 describe the ObservationManager (declaration order,
`clip` then `scale`, `joint_pos_rel = q - default_joint_pos`, `last_action` = the raw action),
the actor is the exporter's `actor(normalizer(obs))` (section 6), and the action is
`JointPositionAction`'s `raw * scale + offset` with `offset = default_joint_pos` (section 3);
the brax side is `brax-ppo-so101.md` section 6 plus Playground's `default_pose + action_scale *
a`. The arrays are the generator's own, held in memory -- the pickle this script asks
`import_rl.py` to write is never read back here (INV-16: pickle lives in `import_rl.py`).

One convention is not in either api-note and is stated here instead: rsl_rl's
`EmpiricalNormalization.forward` is `(x - mean) / (std + eps)` with `eps = 1e-2` (rsl_rl
`modules/normalizer.py`, unverified against a pinned tag -- the api-note's section 8 open item).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import import_rl  # noqa: E402  -- only for its pickle *writer*

# Our robot, as our Task IR and scene name it (tests/fixtures/rl/task-reach-last-action.toml).
OURS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Our rest pose in our order; `initial` of the Task IR's `last_action` channel is this pose,
# because the action offset of both sources is their default pose.
DEFAULT_OURS = [0.1, -0.4, 0.7, 0.25, -0.05, 0.3]

# The Isaac source: an articulation whose joints come out of the USD importer in another order
# and with the jaw under another name (`resolve_matching_names` keeps the articulation's order).
ISAAC_NAMES = ["elbow_flex", "gripper_jaw", "shoulder_lift", "shoulder_pan", "wrist_flex", "wrist_roll"]
ISAAC_RENAME = {"gripper_jaw": "gripper"}
ISAAC_DECIMATION, ISAAC_SIM_DT = 4, 0.005  # 0.02 s: the Deployment IR's 50 Hz
ISAAC_ACTION_SCALE = 0.5
ISAAC_JOINT_VEL_SCALE = 0.05
ISAAC_CLIP_OBSERVATIONS = 100.0
RSL_EPS = 1e-2

PLAYGROUND_CTRL_DT, PLAYGROUND_SIM_DT = 0.02, 0.004
PLAYGROUND_ACTION_SCALE = 0.3

HIDDEN = (32, 32)
OBS_DIM = 6 + 6 + 7 + 6
ACT = 6


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def swish(x):
    return x / (1.0 + np.exp(-x))


def mlp(x, layers, act):
    """Hidden Dense + activation for all but the last layer, which is linear (both notes)."""
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = act(x)
    return x


def random_layers(rng):
    dims = [OBS_DIM, *HIDDEN, ACT]
    return [
        (
            rng.normal(0, 1 / np.sqrt(dims[i]), (dims[i + 1], dims[i])).astype(np.float32),
            rng.normal(0, 0.1, dims[i + 1]).astype(np.float32),
        )
        for i in range(len(dims) - 1)
    ]


def f64(layers):
    return [(w.astype(np.float64), b.astype(np.float64)) for w, b in layers]


# --- Isaac Lab + rsl_rl ---------------------------------------------------------------------


def isaac_observation(q, qd, command, last_raw, default):
    """`ObservationsCfg.PolicyCfg`, declaration order (isaac-lab.md section 2): per term
    compute -> (noise: off at eval) -> clip (none declared) -> scale; then concatenate."""
    joint_pos = (q - default) * 1.0  # mdp.joint_pos_rel
    joint_vel = (qd - 0.0) * ISAAC_JOINT_VEL_SCALE  # mdp.joint_vel_rel, default_joint_vel = 0
    pose_command = command  # mdp.generated_commands("ee_pose")
    actions = last_raw  # mdp.last_action: the raw action the policy emitted
    obs = np.concatenate([joint_pos, joint_vel, pose_command, actions], axis=-1)
    # The rsl_rl wrapper's observation clip, applied to the whole vector.
    return np.clip(obs, -ISAAC_CLIP_OBSERVATIONS, ISAAC_CLIP_OBSERVATIONS)


def isaac_policy(obs, layers, mean, std):
    """The exporter's `actor(normalizer(obs))` (section 6), `EmpiricalNormalization` inside."""
    return mlp((obs - mean) / (std + RSL_EPS), layers, elu)


def isaac_action(raw, default):
    """`JointPositionAction.process_actions`: `raw * scale + offset`, `offset = default_joint_pos`
    under `use_default_offset = True` (section 3)."""
    return raw * ISAAC_ACTION_SCALE + default


def make_isaac(out: Path, rows: int, shape: str) -> None:
    rng = np.random.default_rng(20260923)
    layers = random_layers(rng)
    mean = rng.normal(0, 0.2, OBS_DIM).astype(np.float32)
    var = rng.uniform(0.3, 2.0, OBS_DIM).astype(np.float32)
    std = np.sqrt(var).astype(np.float32)

    src_of_ours = [OURS.index(ISAAC_RENAME.get(n, n)) for n in ISAAC_NAMES]  # source i -> ours
    default_src = np.array([DEFAULT_OURS[j] for j in src_of_ours])

    native = out / "native"
    native.mkdir(parents=True, exist_ok=True)
    norm = {
        "_mean": mean[None, :],
        "_var": var[None, :],
        "_std": std[None, :],
        "count": np.array(1000, dtype=np.int64),
    }
    if shape == "classic":
        state = {}
        for i, (w, b) in enumerate(layers):
            state[f"actor.{2 * i}.weight"], state[f"actor.{2 * i}.bias"] = w, b
            state[f"critic.{2 * i}.weight"] = np.zeros((1 if i == len(layers) - 1 else w.shape[0], w.shape[1]), np.float32)
        state["std"] = np.ones(ACT, np.float32)
        payload = {"model_state_dict": state, "obs_norm_state_dict": norm, "iter": 0}
    else:
        state = {}
        for i, (w, b) in enumerate(layers):
            state[f"mlp.{2 * i}.weight"], state[f"mlp.{2 * i}.bias"] = w, b
        state.update({f"obs_normalizer.{k}": v for k, v in norm.items()})
        state["distribution.std_param"] = np.ones(ACT, np.float32)
        payload = {"actor_state_dict": state, "critic_state_dict": {}, "iter": 0}
    import_rl.save_native_rsl_rl(native / "model_0.pt", payload)

    # Isaac Lab's `params/env.yaml` shape, the fields the importer reads.
    import yaml  # noqa: PLC0415

    env = {
        "decimation": ISAAC_DECIMATION,
        "sim": {"dt": ISAAC_SIM_DT},
        "actions": {
            "arm_action": {
                "class_type": "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction",
                "asset_name": "robot",
                "joint_names": [".*"],
                "scale": ISAAC_ACTION_SCALE,
                "use_default_offset": True,
            }
        },
        "scene": {
            "robot": {
                "init_state": {
                    # Regex keys, as Isaac configs write them; one per joint here.
                    "joint_pos": {n: float(default_src[i]) for i, n in enumerate(ISAAC_NAMES)}
                }
            }
        },
    }
    (out / "env.yaml").write_text(yaml.safe_dump(env, sort_keys=False))
    (out / "import_args.json").write_text(
        json.dumps(
            [
                "--from", "rsl-rl",
                "--checkpoint", str(native / "model_0.pt"),
                "--activation", "elu",
                "--isaac-env-cfg", str(out / "env.yaml"),
                "--joint-names", ",".join(ISAAC_NAMES),
            ]
        )
    )

    # Our states, and what Isaac would do in each.
    q_ours = np.array(DEFAULT_OURS) + rng.uniform(-0.5, 0.5, (rows, ACT))
    qd_ours = rng.uniform(-2.0, 2.0, (rows, ACT))
    command = rng.uniform(-1.0, 1.0, (rows, 7))
    last_raw_src = rng.uniform(-1.0, 1.0, (rows, ACT))
    prev_src = isaac_action(last_raw_src, default_src)  # what the policy commanded last tick
    prev_ours = np.empty_like(prev_src)
    prev_ours[:, src_of_ours] = prev_src

    q_src, qd_src = q_ours[:, src_of_ours], qd_ours[:, src_of_ours]
    obs = isaac_observation(q_src, qd_src, command, last_raw_src, default_src)
    raw = isaac_policy(obs, f64(layers), mean.astype(np.float64), std.astype(np.float64))
    act_src = isaac_action(raw, default_src)
    act_ours = np.empty_like(act_src)
    act_ours[:, src_of_ours] = act_src
    write_case(out, q_ours, qd_ours, command, prev_ours, act_ours, float(np.abs(obs).max()))


# --- MuJoCo Playground + brax ---------------------------------------------------------------


def playground_observation(q, qd, target, last_act, default):
    """A Playground manipulation env's `_get_obs`: `qpos - default_pose`, `qvel`, the target,
    `last_act` (the raw action), concatenated."""
    return np.concatenate([q - default, qd, target, last_act], axis=-1)


def playground_policy(obs, layers, mean, std):
    """brax-ppo-so101.md section 6: `(obs - mean) / std`, swish MLP, `tanh(loc)`."""
    return np.tanh(mlp((obs - mean) / std, layers, swish))


def playground_action(a, default):
    return default + PLAYGROUND_ACTION_SCALE * a


def make_playground(out: Path, rows: int) -> None:
    rng = np.random.default_rng(20260924)
    layers = random_layers(rng)
    mean = rng.normal(0, 0.2, OBS_DIM).astype(np.float32)
    std = rng.uniform(0.5, 1.5, OBS_DIM).astype(np.float32)
    default = np.array(DEFAULT_OURS)

    native = out / "native"
    native.mkdir(parents=True, exist_ok=True)
    flat = {"obs_mean": mean, "obs_std": std}
    *hidden, (hw, hb) = layers
    for i, (w, b) in enumerate(hidden):
        flat[f"kernel_{i}"], flat[f"bias_{i}"] = w.T.copy(), b  # brax kernels are [in, out]
    flat["mean_kernel"], flat["mean_bias"] = hw.T.copy(), hb
    flat["logstd_kernel"] = np.zeros_like(flat["mean_kernel"])
    flat["logstd_bias"] = np.zeros(ACT, np.float32)
    np.savez(native / "source.npz", **flat)
    (native / "meta.json").write_text(
        json.dumps({"framework": "brax", "activation": "swish", "squash": "tanh", "action": {}})
    )
    (out / "playground.json").write_text(
        json.dumps(
            {
                "ctrl_dt": PLAYGROUND_CTRL_DT,
                "sim_dt": PLAYGROUND_SIM_DT,
                "action_scale": PLAYGROUND_ACTION_SCALE,
                "default_pose": DEFAULT_OURS,
            }
        )
    )
    (out / "import_args.json").write_text(
        json.dumps(
            [
                "--from", "mujoco-playground",
                "--checkpoint", str(native),
                "--playground-config", str(out / "playground.json"),
            ]
        )
    )

    q = default + rng.uniform(-0.5, 0.5, (rows, ACT))
    qd = rng.uniform(-2.0, 2.0, (rows, ACT))
    target = rng.uniform(-1.0, 1.0, (rows, 7))
    last_act = rng.uniform(-1.0, 1.0, (rows, ACT))
    prev = playground_action(last_act, default)
    obs = playground_observation(q, qd, target, last_act, default)
    a = playground_policy(obs, f64(layers), mean.astype(np.float64), std.astype(np.float64))
    write_case(out, q, qd, target, prev, playground_action(a, default), float(np.abs(obs).max()))


def write_case(out, q, qd, pose, prev, actions, max_abs_obs) -> None:
    (out / "states.json").write_text(
        json.dumps(
            {
                "joint_pos": q.tolist(),
                "joint_vel": qd.tolist(),
                "cube_pose": pose.tolist(),
                "last_action": prev.tolist(),
            }
        )
    )
    (out / "reference.json").write_text(
        json.dumps({"actions": actions.tolist(), "max_abs_obs": max_abs_obs})
    )
    print(json.dumps({"out": str(out), "rows": len(q), "max_abs_obs": max_abs_obs}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make", required=True, choices=("isaac-classic", "isaac-split", "playground"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if args.make == "playground":
        make_playground(out, args.rows)
    else:
        make_isaac(out, args.rows, args.make.removeprefix("isaac-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
