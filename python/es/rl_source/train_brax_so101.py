"""Train the brax PPO source policy for the SO-101 reach task (packet M8/S2c).

    python train_brax_so101.py --seed 0 --timesteps 6000000 --out ~/artifacts/plan-s/s2c/seed0

Writes into `--out`:

    checkpoints/            orbax, written by brax itself (`save_checkpoint_path`)
    source.npz              framework-neutral weights, the file S2b imports
    meta.json               everything S2b needs to rebuild the function, plus provenance
    curve.json              the reward curve (one row per brax eval)
    oracle-1000.npz         `obs`/`actions` (U(-1, 1), numpy seed 0) and `obs_scaled`/
                            `actions_scaled` (the same draw on the observation's own scale)

`source.npz` holds only the deterministic policy: `obs_mean`, `obs_std`, `kernel_i`/`bias_i`
for each hidden Dense as stored (`[in, out]`, right-multiply), and the output Dense split into
`mean_*` (the first `action_dim` columns, brax's `loc`) and `logstd_*` (the second half, which
brax turns into a standard deviation with `softplus(x) + min_std`, NOT `exp(x)` -- see
`docs/api-notes/brax-ppo-so101.md`). The value network is training-only and is not exported.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata as md
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax import linen
from mujoco_playground import wrapper

import so101_reach_env as reach

HIDDEN = (256, 256)
PACKAGES = (
    "jax",
    "jaxlib",
    "brax",
    "playground",
    "mujoco",
    "mujoco-mjx",
    "flax",
    "optax",
    "orbax-checkpoint",
    "numpy",
)


def ppo_config(timesteps: int, seed: int, num_envs: int) -> dict:
    """In the style of Playground's `PandaPickCube` manipulation params, wider network."""
    return dict(
        num_timesteps=timesteps,
        num_evals=10,
        episode_length=200,
        reward_scaling=1.0,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=1e-3,
        entropy_cost=2e-2,
        num_envs=num_envs,
        batch_size=256,
        max_grad_norm=1.0,
        num_eval_envs=64,
        deterministic_eval=True,
        seed=seed,
    )


def blake3_hex(path: Path) -> str:
    try:
        import blake3
    except ImportError:
        return "unavailable (blake3 not installed)"
    h = blake3.blake3()
    h.update(path.read_bytes())
    return h.hexdigest()


def versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "absent"
    return out


def export_source_npz(params, out: Path) -> dict:
    """brax params -> `source.npz`; returns the layer shapes for `meta.json`."""
    normalizer, policy = params[0], params[1]
    dense = policy["params"]
    names = sorted(dense, key=lambda k: int(k.split("_")[-1]))
    *hidden, last = names

    flat = {
        "obs_mean": np.asarray(normalizer.mean, np.float32),
        "obs_std": np.asarray(normalizer.std, np.float32),
    }
    for i, name in enumerate(hidden):
        flat[f"kernel_{i}"] = np.asarray(dense[name]["kernel"], np.float32)
        flat[f"bias_{i}"] = np.asarray(dense[name]["bias"], np.float32)
    kernel = np.asarray(dense[last]["kernel"], np.float32)
    bias = np.asarray(dense[last]["bias"], np.float32)
    half = kernel.shape[1] // 2
    flat["mean_kernel"] = kernel[:, :half]
    flat["mean_bias"] = bias[:half]
    flat["logstd_kernel"] = kernel[:, half:]
    flat["logstd_bias"] = bias[half:]

    np.savez(out / "source.npz", **flat)
    return {
        "n_hidden": len(hidden),
        "shapes": {k: list(v.shape) for k, v in flat.items()},
        "brax_layer_names": names,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="brax PPO on the SO-101 reach task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--xml",
        default=str(Path(__file__).with_name("so101_reach_mjx.xml")),
        help="the MJX scene; the committed one is too slow to train on (see the api-note)",
    )
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    xml = reach.scene_path(args.xml)
    committed = reach.scene_path(None)

    env = reach.SO101Reach(str(xml))
    eval_env = reach.SO101Reach(str(xml))
    cfg = ppo_config(args.timesteps, args.seed, args.num_envs)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=HIDDEN,
        value_hidden_layer_sizes=HIDDEN,
        activation=linen.swish,
    )

    curve = []
    t0 = time.time()

    def progress(step: int, metrics: dict) -> None:
        row = {"step": int(step), "wall_clock_s": round(time.time() - t0, 1)}
        row.update(
            {k: float(v) for k, v in metrics.items() if k.startswith("eval/episode_")}
        )
        curve.append(row)
        print(json.dumps(row), flush=True)
        (out / "curve.json").write_text(json.dumps(curve, indent=2))

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        # Playground's default auto-reset: each env replays its own first reset, so the cube
        # is drawn once per env rather than once per episode. With `num_envs` in the thousands
        # and ~2 episodes per env over the whole run, that is `num_envs` independent draws
        # from the Task IR's distribution and costs no extra `mjx.forward` per step, which
        # `full_reset=True` would (measured: the GPU is the binding constraint here).
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        progress_fn=progress,
        save_checkpoint_path=str(out / "checkpoints"),
        **cfg,
    )
    wall = time.time() - t0
    print(f"trained {args.timesteps} steps in {wall:.1f} s", flush=True)

    layout = export_source_npz(params, out)

    # The oracle S2b imports against: 1,000 obs per channel and the deterministic action JAX
    # computes for them (the distribution's mode, tanh(loc)). Two sets:
    #   obs         U(-1, 1), numpy seed 0 -- the packet's draw, kept verbatim;
    #   obs_scaled  obs_mean + obs_std * U(-1, 1), numpy seed 1 -- the same draw put on the
    #               observation's own scale. It exists because three channels of this task are
    #               constant (cube z, two quaternion components), so their normalizer std is
    #               brax's 1e-6 floor: a U(-1, 1) value in those channels normalizes to ~1e6,
    #               saturates every swish and lands tanh on +-1, and two float32
    #               implementations cannot then agree to 1e-5 on the few rows that are not
    #               saturated. `obs_scaled` is the set the 1e-5 tier is meaningful on.
    source = np.load(out / "source.npz")
    obs = np.random.default_rng(0).uniform(
        -1.0, 1.0, size=(1000, reach.OBS_DIM)
    ).astype(np.float32)
    scaled = np.random.default_rng(1).uniform(-1.0, 1.0, size=obs.shape).astype(np.float32)
    obs_scaled = (source["obs_mean"] + source["obs_std"] * scaled).astype(np.float32)
    # The actions are computed at `highest` matmul precision. XLA's default on an Ampere-class
    # GPU is TF32 (10-bit mantissa), which differs from a true float32 matmul by ~1e-3 -- far
    # above the 1e-5 tier, and a property of the hardware path rather than of the exported
    # weights. The delta is measured and recorded in `meta.json` instead of being baked in.
    key = jax.random.PRNGKey(0)
    with jax.default_matmul_precision("highest"):
        policy = jax.jit(make_inference_fn(params, deterministic=True))
        actions = np.asarray(policy(jp.asarray(obs), key)[0], np.float32)
        actions_scaled = np.asarray(policy(jp.asarray(obs_scaled), key)[0], np.float32)
    tf32 = jax.jit(make_inference_fn(params, deterministic=True))
    tf32_delta = float(
        np.abs(np.asarray(tf32(jp.asarray(obs_scaled), key)[0]) - actions_scaled).max()
    )
    np.savez(
        out / "oracle-1000.npz",
        obs=obs,
        actions=actions,
        obs_scaled=obs_scaled,
        actions_scaled=actions_scaled,
    )

    mj = env.mj_model
    lo = mj.actuator_ctrlrange[:, 0]
    hi = mj.actuator_ctrlrange[:, 1]
    meta = {
        "framework": "brax",
        "packet": "M8/S2c",
        "versions": versions(),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "wall_clock_s": round(wall, 1),
        "obs_dim": reach.OBS_DIM,
        "action_dim": 6,
        "hidden": list(HIDDEN),
        "activation": "swish",
        # brax's `MLP(activate_final=False)`: the output Dense is linear. The only output
        # nonlinearity is the distribution's `tanh` below (`squash`).
        "activate_output": False,
        "squash": "tanh",
        "normalizer": {
            "formula": "(x - obs_mean) / obs_std",
            "clip": None,
            "source": "brax.training.acme.running_statistics",
        },
        "log_std": {
            "stored": "logstd_kernel/logstd_bias = second half of the output Dense",
            "formula": "std = softplus(x) + 0.001",
            "used_at_inference": False,
        },
        "obs_layout": [
            {"name": name, "start": start, "len": n, "unit": unit}
            for name, start, n, unit in reach.OBS_LAYOUT
        ],
        "quaternion_order": "xyzw (spec 3.1); MuJoCo xquat is wxyz and is reordered",
        "joint_order": list(reach.JOINTS),
        "action": {
            "kind": "position_target",
            "order": list(reach.JOINTS),
            "offset": [float(v) for v in (hi + lo) / 2.0],
            "scale": [float(v) for v in (hi - lo) / 2.0],
            "formula": "ctrl = offset + scale * clip(a, -1, 1)",
        },
        "scene": {
            "used_xml": str(xml),
            "blake3": blake3_hex(xml),
            "committed_scene": str(committed),
            "committed_blake3": blake3_hex(committed),
        },
        "env": {
            "ctrl_dt": 0.02,
            "sim_dt": 0.005,
            "n_substeps": 4,
            "episode_length": 200,
            "success_dist": 0.03,
            "success_bonus": 1.0,
            "distance": "||obs[12:15] - obs[19:22]|| (cube body vs gripper body position)",
            "cube_x": list(reach.CUBE_X),
            "cube_y": list(reach.CUBE_Y),
        },
        "ppo": cfg,
        "export": layout,
        "oracle": {
            "obs": "U(-1, 1) per channel, numpy seed 0",
            "obs_scaled": "obs_mean + obs_std * U(-1, 1), numpy seed 1",
            "actions": "tanh(loc), jax.default_matmul_precision('highest')",
            "tf32_delta": tf32_delta,
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/source.npz, meta.json, oracle-1000.npz, curve.json", flush=True)
    print(f"source.npz blake3 {blake3_hex(out / 'source.npz')}", flush=True)


if __name__ == "__main__":
    main()
