"""Success rate of the exported source policy, in the source framework (packet M8/S2c).

    python eval_brax_so101.py --out ~/artifacts/plan-s/s2c/seed0 [--episodes 64]

The deterministic policy is rebuilt from `source.npz` in JAX -- the same arithmetic S2b will
rebuild in torch -- and run on the MJX env, so this number scores the *exported* file, not a
checkpoint only orbax can open. It is the "source framework" column of S4c's table.

Success is per the task definition (`rl-continuation.md` section 5): gripper-to-cube distance
< 0.03 m. Two readings are printed, because an episode is 200 steps with no early termination:
`reached` (any step inside 0.03 m) and `final` (the last step is inside).

The action kind comes from `meta.json`, so a `joint_delta` export is evaluated as one. For
either kind the per-tick **command change** |target_t - target_{t-1}| is measured here (mean,
p95, max, over every joint and tick): for a delta policy that is the increment the Safety
Plane's clamp is compared against (packet M9/T3), and for a position policy it is what the
same number looks like without an integrator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

import so101_reach_env as reach


def load_policy(npz_path: Path):
    """(obs) -> action, exactly the function `source.npz` describes."""
    npz = np.load(npz_path)
    mean = jp.asarray(npz["obs_mean"])
    std = jp.asarray(npz["obs_std"])
    n = sum(1 for k in npz.files if k.startswith("kernel_"))
    hidden = [(jp.asarray(npz[f"kernel_{i}"]), jp.asarray(npz[f"bias_{i}"])) for i in range(n)]
    mean_k = jp.asarray(npz["mean_kernel"])
    mean_b = jp.asarray(npz["mean_bias"])

    def policy(obs: jax.Array) -> jax.Array:
        x = (obs - mean) / std
        for kernel, bias in hidden:
            x = jax.nn.swish(x @ kernel + bias)
        return jp.tanh(x @ mean_k + mean_b)

    return policy


def _stats(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate the exported SO-101 reach policy")
    ap.add_argument("--out", required=True, help="dir holding source.npz")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xml", default=None)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    xml = args.xml or meta.get("scene", {}).get("used_xml")
    action = meta.get("action", {})
    kind = action.get("kind", "position_target")
    env = reach.SO101Reach(
        xml,
        action="delta" if kind == "joint_delta" else "position",
        delta_scale=action.get("delta_scale", reach.DELTA_SCALE),
    )
    policy = jax.jit(jax.vmap(load_policy(out / "source.npz")))
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    state = reset(keys)
    reached = np.zeros(args.episodes)
    returns = np.zeros(args.episodes)
    prev = np.asarray(state.data.ctrl)
    changes = []
    for _ in range(int(env._config.episode_length)):
        state = step(state, policy(state.obs))
        ctrl = np.asarray(state.data.ctrl)
        changes.append(np.abs(ctrl - prev))
        prev = ctrl
        reached = np.maximum(reached, np.asarray(state.metrics["success"]))
        returns += np.asarray(state.reward)
    final = np.asarray(state.metrics["success"])
    dist = np.asarray(state.metrics["dist"])

    result = {
        "episodes": args.episodes,
        "seed": args.seed,
        "scene": env.xml_path,
        "success_reached": float(reached.mean()),
        "success_final": float(final.mean()),
        "final_dist_mean_m": float(dist.mean()),
        "final_dist_max_m": float(dist.max()),
        "return_mean": float(returns.mean()),
        "action_kind": kind,
        # |target_t - target_{t-1}| in rad: over every (episode, tick, joint) triple, and then
        # per tick as the largest of the six joints -- the number a per-tick clamp acts on.
        "command_change_rad": _stats(np.asarray(changes)),
        "command_change_rad_max_joint": _stats(np.asarray(changes).max(axis=2)),
    }
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
