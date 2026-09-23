"""M11 I0 smoke: import an MJCF into headless Isaac Sim (PhysX), step it with a fixed
position-target sequence, print versions and the joint trajectory as JSON.

Not a backend (I1 owns that). The same script also runs the reference on MuJoCo CPU and
compares two trajectories, so the |dq| numbers in docs/api-notes/isaac-sim.md come from one file:

    OMNI_KIT_ACCEPT_EULA=YES ~/venvs/es-isaac/bin/python python/physx_smoke.py --mjcf so101.xml \
        --out physx.json --stage-report stage.json
    ~/venvs/es/bin/python python/physx_smoke.py --engine mujoco --mjcf so101.xml --out mujoco.json
    python python/physx_smoke.py --compare mujoco.json physx.json

The control: every position actuator's target is
    mid + 0.5 * half * sin(2*pi*k/steps + i)   (k = step, i = actuator index, [mid±half] = ctrlrange)
applied before each of `--steps` physics steps at the MJCF's own <option timestep>.
Row k is the state *after* step k. `q` holds every 1-dof joint by MJCF name (radians/metres),
`pos` the world position of every body carrying a free joint.
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET


def parse_mjcf(path):
    """What both engines need from the MJCF: timestep, position actuators, free-joint bodies."""
    root = ET.parse(path).getroot()
    opt = root.find("option")
    dt = float(opt.get("timestep", "0.002")) if opt is not None else 0.002
    acts = []
    actuator = root.find("actuator")  # not root.iter: <default> classes carry <position> too
    for i, a in enumerate(actuator.iter("position") if actuator is not None else []):
        lo, hi = (float(x) for x in a.get("ctrlrange").split())
        joint = next(j for j in root.iter("joint") if j.get("name") == a.get("joint"))
        # kp/kv/damping as written inline (the emitted MJCF); class defaults are not resolved.
        acts.append({"name": a.get("name"), "joint": a.get("joint"), "lo": lo, "hi": hi, "i": i,
                     "kp": float(a.get("kp", "1")), "kv": float(a.get("kv", "0")),
                     "damping": float(joint.get("damping", "0"))})
    free = []
    for body in root.iter("body"):
        for child in body:
            if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free"):
                free.append(body.get("name"))
    return {"dt": dt, "actuators": acts, "free_bodies": free}


def targets(acts, k, steps):
    return [
        0.5 * (a["lo"] + a["hi"]) + 0.25 * (a["hi"] - a["lo"]) * math.sin(2 * math.pi * k / steps + a["i"])
        for a in acts
    ]


def run_mujoco(args, spec):
    import mujoco

    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    hinge = [
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), m.jnt_qposadr[j])
        for j in range(m.njnt)
        if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    ]
    bodies = [(b, m.body(b).id) for b in spec["free_bodies"]]
    # MJCF actuator order == ctrl order; our parse walks the same document order.
    rows = []
    for k in range(args.steps):
        d.ctrl[:] = targets(spec["actuators"], k, args.steps)
        mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)  # mj_step leaves xpos at the pre-integration state
        rows.append({
            "step": k,
            "q": {n: float(d.qpos[a]) for n, a in hinge},
            "pos": {b: [float(x) for x in d.xpos[i]] for b, i in bodies},
        })
    return {"engine": "mujoco-cpu", "versions": {"mujoco": mujoco.__version__}, "dt": m.opt.timestep, "rows": rows}


def stage_report(stage):
    """Every physics-relevant attribute the importer wrote, per prim — the fidelity table's source."""
    keep = ("physics:", "physx", "drive:", "limit:", "mjcf:", "isaac")
    out = []
    for prim in stage.Traverse():
        attrs = {}
        for a in prim.GetAttributes():
            n = a.GetName()
            if n.startswith(keep):
                v = a.Get()
                if v is None:
                    continue
                try:
                    json.dumps(v)
                except TypeError:
                    try:
                        v = [float(x) for x in v]  # Gf.Vec3f and friends
                    except TypeError:
                        v = str(v)  # Gf.Quatf, tokens
                attrs[n] = v
        rels = {r.GetName(): [str(t) for t in r.GetTargets()] for r in prim.GetRelationships()
                if r.GetName().startswith(("physics:", "material:", "collection:"))}
        schemas = list(prim.GetAppliedSchemas())
        if attrs or rels or schemas or prim.GetTypeName() in ("Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Plane"):
            out.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(), "schemas": schemas,
                        "attrs": attrs, "rels": rels})
    return out


def externalize_meshes(path):
    """The importer cannot read an inline <mesh vertex= face=> (what scene_to_mjcf emits): it
    looks for a file named after the mesh and aborts the process. Write each as an OBJ beside a
    rewritten copy of the MJCF. Returns (path to import, list of what was rewritten)."""
    tree = ET.parse(path)
    inline = [m for m in tree.getroot().iter("mesh") if m.get("vertex") is not None]
    if not inline:
        return path, []
    out_dir = os.path.abspath(os.path.splitext(path)[0] + ".meshes")
    os.makedirs(out_dir, exist_ok=True)
    for m in inline:
        v = [float(x) for x in m.attrib.pop("vertex").split()]
        f = [int(x) for x in m.attrib.pop("face").split()]
        name = os.path.join(out_dir, m.get("name") + ".obj")
        with open(name, "w") as obj:
            obj.writelines("v %r %r %r\n" % tuple(v[i:i + 3]) for i in range(0, len(v), 3))
            obj.writelines("f %d %d %d\n" % (f[i] + 1, f[i + 1] + 1, f[i + 2] + 1) for i in range(0, len(f), 3))
        m.set("file", os.path.basename(name))  # relative: the importer prefixes the MJCF's dir
    new = os.path.join(out_dir, os.path.basename(path))
    tree.write(new)
    return new, ["inline mesh %s written to %s" % (m.get("name"), m.get("file")) for m in inline]


def run_physx(args, spec):
    from isaacsim import SimulationApp

    exp = args.experience
    if exp is None:
        import isaaclab
        cand = os.path.join(os.path.dirname(isaaclab.__file__), "apps", "isaaclab.python.headless.kit")
        # ponytail: Isaac Lab's physics-only experience; the default Isaac Sim one crashes in
        # librtx.scenedb on Ubuntu 26.04 / driver 610 (api-note §7). `--experience ""` retries it.
        exp = cand if os.path.exists(cand) else ""
    app = SimulationApp({"headless": True}, experience=exp)
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.asset.importer.mjcf")
        app.update()
        import numpy as np
        import omni.kit.commands
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.version import get_version
        from pxr import UsdPhysics

        import torch

        gpu = args.device.startswith("cuda")  # the GPU pipeline needs the torch backend

        def arr(x, dtype):
            return torch.tensor(x, dtype=dtype, device=args.device) if gpu else np.array(x)

        world = World(stage_units_in_meters=1.0, physics_dt=spec["dt"], rendering_dt=spec["dt"],
                      backend="torch" if gpu else "numpy", device=args.device)
        _, cfg = omni.kit.commands.execute("MJCFCreateImportConfig")
        cfg.set_fix_base(args.fix_base)
        cfg.set_import_inertia_tensor(True)
        cfg.set_create_physics_scene(False)  # World owns the scene and its dt
        cfg.set_make_default_prim(False)
        mjcf, fixups = externalize_meshes(args.mjcf)
        omni.kit.commands.execute("MJCFCreateAsset", mjcf_path=os.path.abspath(mjcf),
                                  import_config=cfg, prim_path="/World/robot")
        app.update()
        stage = omni.usd.get_context().get_stage()

        # Two more importer gaps are repaired on the stage so MJCF semantics hold; every repair
        # is recorded in the output and is a mapping-report row (api-note §7).
        for p in list(stage.Traverse()):
            # (a) the MJCF <worldbody> becomes an Xform carrying ArticulationRootAPI but no rigid
            # body; isaacsim.core's articulation view then fails in World.reset().
            if p.HasAPI(UsdPhysics.ArticulationRootAPI) and not p.HasAPI(UsdPhysics.RigidBodyAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                fixups.append("removed ArticulationRootAPI from bodiless " + str(p.GetPath()))
            # (b) fix_base welds *every* root, including bodies with an MJCF free joint.
            if p.GetTypeName() == "PhysicsFixedJoint" and p.GetName() in ["rootJoint_" + b for b in spec["free_bodies"]]:
                stage.RemovePrim(p.GetPath())
                fixups.append("removed fix_base weld " + str(p.GetPath()) + " (MJCF free joint)")
        # Diagnostic only (not a fix the importer owes us): rewrite each drive from the MJCF
        # actuator, stiffness = kp, damping = kv + joint damping, either as written ("rad") or
        # converted to USD's per-degree angular units ("deg"), to attribute |dq| to the gains.
        if args.drive_gains != "asis":
            unit = math.pi / 180 if args.drive_gains == "deg" else 1.0
            gains = {a["joint"]: a for a in spec["actuators"]}
            for p in stage.Traverse():
                a = gains.get(p.GetName())
                if a and p.HasAPI(UsdPhysics.DriveAPI, "angular"):
                    drive = UsdPhysics.DriveAPI.Get(p, "angular")
                    drive.GetStiffnessAttr().Set(a["kp"] * unit)
                    drive.GetDampingAttr().Set((a["kv"] + a["damping"]) * unit)
            fixups.append("drive gains rewritten from MJCF (%s)" % args.drive_gains)
        roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        arts = [world.scene.add(SingleArticulation(prim_path=r, name="art%d" % i)) for i, r in enumerate(roots)]
        # A free body is found by MJCF body name anywhere under the import root.
        free = {}
        for p in stage.Traverse():
            if p.GetName() in spec["free_bodies"] and p.HasAPI(UsdPhysics.RigidBodyAPI):
                free.setdefault(p.GetName(), str(p.GetPath()))
        if args.stage_report:  # before reset: a failing reset must still leave the report
            with open(args.stage_report, "w") as f:
                json.dump({"articulation_roots": roots, "free_bodies": free, "fixups": fixups, "prims": stage_report(stage)}, f, indent=1)
        # Registered with the scene so reset() binds them to the physics view (otherwise the
        # pose read back is the stale USD transform).
        bodies = {b: world.scene.add(SingleRigidPrim(prim_path=path, name="fb_" + b)) for b, path in free.items()}
        world.reset()

        # Actuators are matched to DOFs by joint name (the importer's DOF order is its own).
        plan = []
        for art in arts:
            names = list(art.dof_names)
            idx = [(names.index(a["joint"]), a["i"]) for a in spec["actuators"] if a["joint"] in names]
            plan.append((art, names, idx))
        # The state reset() leaves (reset steps physics itself; a free body shows how far).
        initial = {b: [float(x) for x in prim.get_world_pose()[0]] for b, prim in bodies.items()}
        masses = {b: float(prim.get_mass()) for b, prim in bodies.items()}  # as PhysX resolved them
        rows = []
        for k in range(args.steps):
            t = targets(spec["actuators"], k, args.steps)
            for art, names, idx in plan:
                if idx:
                    art.apply_action(ArticulationAction(joint_positions=arr([t[i] for _, i in idx], torch.float32),
                                                        joint_indices=arr([d for d, _ in idx], torch.long)))
            world.step(render=False)
            q = {}
            for art, names, _ in plan:
                q.update({n: float(v) for n, v in zip(names, art.get_joint_positions())})
            pos = {b: [float(x) for x in prim.get_world_pose()[0]] for b, prim in bodies.items()}
            rows.append({"step": k, "q": q, "pos": pos})
        result = {"engine": "physx", "device": args.device,
                  "versions": {"isaacsim": get_version()[0] if isinstance(get_version(), (list, tuple)) else str(get_version()),
                               "experience": os.path.basename(exp) or "default", "torch": torch.__version__},
                  "dt": spec["dt"], "articulations": {a.prim_path: list(a.dof_names) for a in arts},
                  "fix_base": args.fix_base, "fixups": fixups, "initial_pos": initial, "free_body_mass": masses, "rows": rows}
        emit(args, result)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # app.close() would exit 0 and hide the failure
    finally:
        sys.stdout.flush()
        app.close()  # may exit the process; everything is already written


def emit(args, result):
    text = json.dumps(result)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print("TRAJECTORY " + text, flush=True)
    print("ROWS %d" % len(result["rows"]), flush=True)


def compare(a_path, b_path):
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    n = min(len(a["rows"]), len(b["rows"]))
    out = {"a": a["engine"], "b": b["engine"], "steps": n, "max_abs_dq": {}, "final_dq": {}, "max_abs_dpos": {}}
    for j in a["rows"][0]["q"]:
        if j in b["rows"][0]["q"]:
            d = [abs(a["rows"][k]["q"][j] - b["rows"][k]["q"][j]) for k in range(n)]
            out["max_abs_dq"][j] = max(d)
            out["final_dq"][j] = b["rows"][n - 1]["q"][j] - a["rows"][n - 1]["q"][j]
    for body in a["rows"][0]["pos"]:
        if body in b["rows"][0]["pos"]:
            out["max_abs_dpos"][body] = max(
                max(abs(x - y) for x, y in zip(a["rows"][k]["pos"][body], b["rows"][k]["pos"][body])) for k in range(n))
    out["missing_in_b"] = sorted(set(a["rows"][0]["q"]) - set(b["rows"][0]["q"]))
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf")
    p.add_argument("--engine", choices=["physx", "mujoco"], default="physx")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out")
    p.add_argument("--stage-report")
    p.add_argument("--device", default="cpu", help="PhysX pipeline: cpu or cuda:0")
    p.add_argument("--fix-base", type=int, default=1, help="importer fix_base (MJCF: a jointless root is welded)")
    p.add_argument("--experience", default=None)
    p.add_argument("--drive-gains", choices=["asis", "rad", "deg"], default="asis",
                   help="diagnostic: keep the importer's drives, or rewrite them from the MJCF")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()
    if args.compare:
        return compare(*args.compare)
    spec = parse_mjcf(args.mjcf)
    if args.engine == "mujoco":
        emit(args, run_mujoco(args, spec))
    else:
        run_physx(args, spec)


if __name__ == "__main__":
    main()
