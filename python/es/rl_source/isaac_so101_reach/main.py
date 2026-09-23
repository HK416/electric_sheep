"""Train and score the SO-101 reach policy in Isaac Lab on our own scene (packet M11/I3).

    main.py build-usd --mjcf emitted.xml --out <dir>
    main.py train --scene <dir>/scene.json --seed N --iterations N --out <run>
    main.py eval  --scene <dir>/scene.json --checkpoint <run>/model_N.pt [...] --seed N --episodes N --out <json>

Run with the Isaac venv (`~/venvs/es-isaac/bin/python`), `OMNI_KIT_ACCEPT_EULA=YES` and the
compat `LD_LIBRARY_PATH` of isaac-sim.md 7.2. `AppLauncher(headless=True)` starts Isaac Lab's
physics-only experience (`isaaclab.python.headless.kit`); nothing here renders.

`build-usd` runs `physx_ref.import_scene` -- the backend's own importer path and repairs, not a
copy -- on the MJCF `scene_to_mjcf` emits (`ES_PHYSX_DUMP_MJCF` on any `--backend physx` run
writes it), flattens the stage and writes the robot prim as `robot.usd`, plus `scene.json`: the
fixups, the cube's prim path, and the PhysX scene attributes `physx_ref.py`'s `World` authors,
so a mismatch with Isaac Lab's `PhysxCfg` is a listed row and not a guess. `train` is rsl_rl's
`OnPolicyRunner` on `env_cfg.make_env_cfg` (Isaac Lab's `train.py` minus Hydra) and writes
`params/env.yaml` + `params/agent.yaml` as `train.py` does. `eval` rolls each checkpoint's
deterministic policy (the Gaussian's mean) for one episode in each of `--episodes` envs, all
reset from one seed the training run never used, and writes the Isaac-side success rates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PHYSX_REF = ROOT / "crates" / "es-physics-backend" / "python" / "physx_ref.py"


def launch(device: str):
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=True, device=device).app


def physx_ref():
    spec = importlib.util.spec_from_file_location("physx_ref", PHYSX_REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scene_attrs(stage) -> dict:
    """Every authored attribute of the stage's physics scene prim(s)."""
    from pxr import UsdPhysics

    out = {}
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene):
            out[str(p.GetPath())] = {
                a.GetName(): repr(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValue()
            }
    return out


def build_usd(args) -> None:
    app = launch("cpu")
    ref = physx_ref()
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    mjcf = Path(args.mjcf).read_text()
    spec = ref.parse(mjcf)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="es-i3-")
    path, fixups = ref.externalize_meshes(mjcf, tmp)
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    fixups += ref.import_scene(path, spec, "/World/robot", "/World/robot")

    free = sorted(spec["free"])
    cube = [p for p in stage.Traverse() if p.GetName() in free and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(cube) != 1:
        raise SystemExit("expected one free body, found %s" % [str(p.GetPath()) for p in cube])
    cube_path = str(cube[0].GetPath())[len("/World/robot/"):]

    flat = stage.Flatten()
    usd = out / "robot.usd"
    layer = Sdf.Layer.CreateNew(str(usd))
    if not Sdf.CopySpec(flat, "/World/robot", layer, "/robot"):
        raise SystemExit("Sdf.CopySpec /World/robot -> /robot failed")
    layer.Save()
    s = Usd.Stage.Open(str(usd))
    s.SetDefaultPrim(s.GetPrimAtPath("/robot"))
    UsdGeom.SetStageUpAxis(s, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(s, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(s, 1.0)
    s.GetRootLayer().Save()
    colliders = [str(p.GetPath()) for p in s.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    roots = [str(p.GetPath()) for p in s.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]

    # The scene `physx_ref.py` simulates in: `World(physics_dt = the MJCF timestep)`.
    from isaacsim.core.api import World

    ctx.new_stage()
    World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.005, backend="numpy", device="cpu")
    app.update()
    info = {
        "usd": str(usd),
        "cube_path": cube_path,
        "articulation_roots": roots,
        "colliders": len(colliders),
        "fixups": fixups,
        "physx_ref_world_scene": scene_attrs(ctx.get_stage()),
        "mjcf_blake2b": __import__("hashlib").blake2b(mjcf.encode()).hexdigest()[:32],
    }
    (out / "scene.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "physx_ref_world_scene"}, indent=2))
    sys.stdout.flush()
    os._exit(0)


def make_env(scene: dict, num_envs: int, seed: int, device: str):
    from isaaclab.envs import ManagerBasedRLEnv

    import env_cfg

    cfg = env_cfg.make_env_cfg(scene["usd"], scene["cube_path"], num_envs, seed, device)
    env = ManagerBasedRLEnv(cfg=cfg)
    joints = list(env.scene["robot"].joint_names)
    if joints != env_cfg.JOINTS:
        raise SystemExit("articulation joint order %s != the adapter's %s" % (joints, env_cfg.JOINTS))
    if env.max_episode_length != 200:
        raise SystemExit("max_episode_length %d != 200 control steps" % env.max_episode_length)
    return cfg, env


def train(args) -> None:
    launch(args.device)
    sys.path.insert(0, str(HERE))
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    import env_cfg

    scene = json.loads(Path(args.scene).read_text())
    torch.manual_seed(args.seed)
    cfg, env = make_env(scene, args.num_envs, args.seed, args.device)
    agent = env_cfg.make_runner_cfg(args.seed, args.iterations, args.device)
    out = Path(args.out)
    (out / "params").mkdir(parents=True, exist_ok=True)
    dump_yaml(str(out / "params" / "env.yaml"), cfg)
    dump_yaml(str(out / "params" / "agent.yaml"), agent)
    (out / "isaaclab_scene.json").write_text(json.dumps(scene_attrs(env.sim.stage), indent=2) + "\n")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(out), device=args.device)
    t0 = time.time()
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
    (out / "wall.json").write_text(json.dumps({
        "learn_s": time.time() - t0, "iterations": agent.max_iterations, "num_envs": args.num_envs,
        "steps_per_env": agent.num_steps_per_env,
        "env_steps": agent.max_iterations * agent.num_steps_per_env * args.num_envs,
    }) + "\n")
    sys.stdout.flush()
    os._exit(0)


def evaluate(args) -> None:
    launch(args.device)
    sys.path.insert(0, str(HERE))
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    import env_cfg

    scene = json.loads(Path(args.scene).read_text())
    _, env = make_env(scene, args.episodes, args.seed, args.device)
    agent = env_cfg.make_runner_cfg(args.seed, 1, args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=args.device)
    n = args.episodes
    results = []
    # One inference-mode block: tensors made inside it cannot be written outside it, and a
    # reset between checkpoints writes the articulation state.
    with torch.inference_mode():
        for ckpt in args.checkpoint:
            runner.load(ckpt, load_optimizer=False)
            policy = runner.get_inference_policy(device=args.device)
            env.reset(seed=args.seed)  # the same resets for every checkpoint
            obs = wrapped.get_observations()
            # `--action-delay k`: the env executes the row the policy emitted k ticks ago (raw 0,
            # the rest pose, before that) -- the latency `es eval run` applies (diagnostic).
            queue = [torch.zeros(n, env.action_manager.total_action_dim, device=args.device)] * args.action_delay
            done = torch.zeros(n, dtype=torch.bool, device=args.device)
            success = torch.zeros(n, dtype=torch.bool, device=args.device)
            length = torch.zeros(n, dtype=torch.long, device=args.device)
            for t in range(1, 201):
                queue.append(policy(obs))
                obs, _, dones, _ = wrapped.step(queue.pop(0))
                ended = dones.bool() & ~done
                success |= ended & env.termination_manager.get_term("success")
                length[ended] = t
                done |= ended
                if bool(done.all()):
                    break
            results.append({
                "checkpoint": ckpt, "reset_seed": args.seed, "episodes": n, "action_delay": args.action_delay,
                "success_rate": success.float().mean().item(),
                "episode_length": length.float().mean().item(),
                "unfinished": int((~done).sum().item()),
            })
            print(json.dumps(results[-1]))
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    sys.stdout.flush()
    os._exit(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-usd")
    b.add_argument("--mjcf", required=True)
    b.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--scene", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--iterations", type=int, default=1000)
    t.add_argument("--num-envs", type=int, default=4096)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--scene", required=True)
    e.add_argument("--checkpoint", required=True, nargs="+")
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--episodes", type=int, default=1024)
    e.add_argument("--action-delay", type=int, default=0)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        {"build-usd": build_usd, "train": train, "eval": evaluate}[args.cmd](args)
    except BaseException:
        # Kit's own exit path reports 0 after an uncaught exception (measured, M11/I3).
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
