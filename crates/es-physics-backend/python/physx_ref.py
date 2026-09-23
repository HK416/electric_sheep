"""PhysX (Isaac Sim, headless) process for `PhysXBackend` (packet M11/I1, spec 28.14 rule 6).

The same line-delimited JSON protocol as mujoco_ref.py (load / reset / set_ctrl / step / state /
set_state / quit), answered from an Isaac Sim stage built by Isaac Sim's own MJCF importer from
the MJCF `scene_to_mjcf` emits. The import path and its repairs are python/physx_smoke.py's
(packet M11/I0); every importer gap measured there (docs/api-notes/isaac-sim.md section 7.3) is
either repaired here or declared by the Rust side's mapping report, and the list of repairs is
in the load reply (`fixups`).

State is MuJoCo's layout, rebuilt from the MJCF: bodies depth-first in document order with
`world` first, joints in body order, hinge/slide -> 1 qpos / 1 qvel, free -> 7 / 6 with the
quaternion w-first (MuJoCo's order), its linear velocity that of the body origin in the world
frame and its angular velocity in the body frame (MuJoCo's convention; PhysX reports the
centre-of-mass velocity and a world-frame angular velocity, converted both ways here).
`xquat` is x-first like every other backend's reply (spec 3.1).

stdout is the protocol. Kit logs to stdout, so fd 1 is duplicated for the protocol before
anything is imported and then pointed at stderr (which the Rust side discards).
"""

import json
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

PROTO = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

JOINT_DIMS = {"free": (7, 6), "ball": (4, 3), "slide": (1, 1), "hinge": (1, 1)}


def floats(text, n=None):
    values = [float(x) for x in text.split()]
    if n is not None and len(values) != n:
        raise ValueError("expected %d numbers in %r" % (n, text))
    return values


def parse(mjcf):
    """What the backend needs from the emitted MJCF, in MuJoCo's own order."""
    root = ET.fromstring(mjcf)
    opt = root.find("option")
    gravity = floats(opt.get("gravity", "0 0 -9.81"), 3) if opt is not None else [0.0, 0.0, -9.81]
    bodies, joints, geoms = ["world"], [], []
    parent = {}
    nq = nv = 0

    def walk(element, owner):
        nonlocal nq, nv
        for g in element.findall("geom"):
            collides = not (g.get("contype", "1") == "0" and g.get("conaffinity", "1") == "0")
            friction = floats(g.get("friction", "1 0.005 0.0001"))
            geoms.append({"name": g.get("name"), "body": owner, "collides": collides, "mu": friction[0]})
        for b in element.findall("body"):
            name = b.get("name")
            bodies.append(name)
            parent[name] = owner
            for j in b:
                if j.tag == "freejoint":
                    kind = "free"
                elif j.tag == "joint":
                    kind = j.get("type", "hinge")
                else:
                    continue
                if kind == "ball":
                    raise ValueError("joint `%s`: ball joints are not mapped by this adapter" % j.get("name"))
                dq, dv = JOINT_DIMS[kind]
                joints.append({
                    "name": j.get("name"), "kind": kind, "body": name, "qpos": [nq, dq], "dof": [nv, dv],
                    "damping": float(j.get("damping", "0")), "stiffness": float(j.get("stiffness", "0")),
                    "springref": float(j.get("springref", "0")),
                })
                nq, nv = nq + dq, nv + dv
            walk(b, name)

    walk(root.find("worldbody"), "world")
    free = {j["body"] for j in joints if j["kind"] == "free"}
    for j in joints:
        if j["kind"] == "free" and any(parent.get(b) == j["body"] for b in bodies):
            raise ValueError("body `%s`: a free joint on a body with children (a floating-base "
                             "articulation) is not mapped by this adapter" % j["body"])
    actuators = []
    for a in (root.find("actuator") if root.find("actuator") is not None else []):
        if a.tag not in ("position", "motor"):
            raise ValueError("actuator `%s`: <%s> is not mapped (position and motor are)" % (a.get("name"), a.tag))
        gear = floats(a.get("gear", "1"))[0]
        ctrl = floats(a.get("ctrlrange")) if a.get("ctrlrange") else None
        force = floats(a.get("forcerange")) if a.get("forcerange") else None
        if force is not None and a.tag == "position" and force[0] != -force[1]:
            raise ValueError("actuator `%s`: an asymmetric forcerange has no PhysX drive analogue" % a.get("name"))
        if a.tag == "position" and gear != 1.0:
            raise ValueError("actuator `%s`: a position actuator with gear %r is not mapped" % (a.get("name"), gear))
        actuators.append({"name": a.get("name"), "kind": a.tag, "joint": a.get("joint"), "gear": gear,
                          "kp": float(a.get("kp", "1")), "kv": float(a.get("kv", "0")),
                          "ctrl": ctrl, "force": force})
    if root.find("sensor") is not None and len(root.find("sensor")):
        raise ValueError("<sensor> is not mapped by this adapter")
    return {"gravity": gravity, "bodies": bodies, "joints": joints, "geoms": geoms, "free": free,
            "actuators": actuators, "nq": nq, "nv": nv}


def externalize_meshes(mjcf, out_dir):
    """The importer cannot read an inline <mesh vertex= face=> and aborts the process with
    exit 0 (api-note 7.3): write each as an OBJ beside a rewritten copy of the MJCF. The
    scene's hashed structs never see these paths."""
    tree = ET.ElementTree(ET.fromstring(mjcf))
    inline = [m for m in tree.getroot().iter("mesh") if m.get("vertex") is not None]
    for m in inline:
        v = floats(m.attrib.pop("vertex"))
        f = [int(x) for x in m.attrib.pop("face").split()]
        # Named after the mesh: the importer looks a mesh up by its name, not by file= (I1).
        name = m.get("name").replace("/", "_") + ".obj"
        with open(os.path.join(out_dir, name), "w") as obj:
            obj.writelines("v %r %r %r\n" % tuple(v[k:k + 3]) for k in range(0, len(v), 3))
            obj.writelines("f %d %d %d\n" % (f[k] + 1, f[k + 1] + 1, f[k + 2] + 1) for k in range(0, len(f), 3))
        m.set("file", name)  # relative: the importer resolves it against the MJCF's directory
    path = os.path.join(out_dir, "scene.xml")
    tree.write(path)
    return path, ["inline mesh %s -> %s" % (m.get("name"), m.get("file")) for m in inline]


# Quaternions are (w, x, y, z) lists below.
def qrot(q, v):
    w, x, y, z = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return [v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx]


def qconj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


class Sim(object):
    def __init__(self, mjcf, n_envs, timestep, device):
        self.spec = parse(mjcf)
        self.n = n_envs
        self.device = device
        self.gpu = device.startswith("cuda")
        self.fixups = []
        self._tmp = tempfile.TemporaryDirectory(prefix="es-physx-")
        path, fix = externalize_meshes(mjcf, self._tmp.name)
        self.fixups += fix
        self._start_app()
        self._build_stage(path, timestep)

    # ---- process / stage ------------------------------------------------------------------

    def _start_app(self):
        from isaacsim import SimulationApp

        exp = os.environ.get("ES_ISAAC_EXPERIENCE")
        if exp is None:
            # Isaac Lab's physics-only experience: the default Isaac Sim one segfaults in
            # librtx.scenedb on the oracle host (api-note 7.2). Set ES_ISAAC_EXPERIENCE="" to
            # use Isaac Sim's default.
            try:
                import isaaclab
                cand = os.path.join(os.path.dirname(isaaclab.__file__), "apps", "isaaclab.python.headless.kit")
                exp = cand if os.path.exists(cand) else ""
            except ImportError:
                exp = ""
        self.experience = os.path.basename(exp) or "default"
        self.app = SimulationApp({"headless": True}, experience=exp)

    def _build_stage(self, path, timestep):
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.asset.importer.mjcf")
        self.app.update()
        import omni.kit.commands
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

        spec = self.spec
        self.world = World(stage_units_in_meters=1.0, physics_dt=timestep, rendering_dt=timestep,
                           backend="torch" if self.gpu else "numpy", device=self.device)
        _, cfg = omni.kit.commands.execute("MJCFCreateImportConfig")
        cfg.set_fix_base(True)  # MJCF: a jointless root body is welded to the world
        cfg.set_import_inertia_tensor(True)
        cfg.set_create_physics_scene(False)  # World owns the scene and its dt
        cfg.set_make_default_prim(False)
        env0 = "/World/envs/env_0"
        omni.kit.commands.execute("MJCFCreateAsset", mjcf_path=path, import_config=cfg, prim_path=env0 + "/robot")
        self.app.update()
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(env0 + "/robot").IsValid():
            raise RuntimeError("the MJCF importer produced no prim at %s/robot" % env0)

        # The importer writes each body's colliders once as a prototype under /collisions (and
        # meshes / visuals under /meshes, /visuals), then references them into the body as an
        # instance. The prototypes are defined, active, collision-enabled prims: PhysX
        # simulates every one as a static collider at the world origin (measured, I1). They
        # are deactivated at their root; the references into the bodies still compose.
        for top in stage.GetPseudoRoot().GetChildren():
            if top.GetPath() != Sdf.Path("/World") and not top.IsA(UsdPhysics.Scene):
                top.SetActive(False)
                self.fixups.append("deactivated importer prototype root %s" % top.GetPath())
        # Instances are made ordinary prims so a collider can carry its own material binding.
        for p in [p for p in stage.Traverse() if p.IsInstanceable()]:
            p.SetInstanceable(False)

        by_name = {}
        for p in stage.Traverse():
            by_name.setdefault(p.GetName(), []).append(p)

        for p in list(stage.Traverse()):
            if not p.IsValid():
                continue
            # The <worldbody> becomes an Xform with ArticulationRootAPI but no rigid body.
            if p.HasAPI(UsdPhysics.ArticulationRootAPI) and not p.HasAPI(UsdPhysics.RigidBodyAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                self.fixups.append("removed ArticulationRootAPI from bodiless " + str(p.GetPath()))
            # fix_base welds every root, MJCF free bodies included; a free body is a plain
            # rigid body here, not a 0-dof articulation.
            if p.GetTypeName() == "PhysicsFixedJoint" and p.GetName() in ["rootJoint_" + b for b in spec["free"]]:
                stage.RemovePrim(p.GetPath())
                self.fixups.append("removed fix_base weld " + str(p.GetPath()))
            elif p.GetName() in spec["free"] and p.HasAPI(UsdPhysics.ArticulationRootAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                self.fixups.append("free body %s is a rigid body, not an articulation" % p.GetName())
            # The importer's collision groups filter nothing (api-note 7.3); envs are isolated
            # by the cloner's own groups below.
            if p.IsValid() and p.GetTypeName() == "PhysicsCollisionGroup":
                stage.RemovePrim(p.GetPath())
                self.fixups.append("removed importer collision group " + str(p.GetPath()))

        # Every rigid body: no PhysX-only damping, no sleeping (MuJoCo has neither).
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
                rb.CreateAngularDampingAttr().Set(0.0)
                rb.CreateLinearDampingAttr().Set(0.0)
                rb.CreateSleepThresholdAttr().Set(0.0)
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(p).CreateSleepThresholdAttr().Set(0.0)
        self.fixups.append("angular/linear damping 0 and sleep threshold 0 on every rigid body")

        # Friction: one PhysX material per distinct MJCF sliding coefficient, static = dynamic,
        # combined by max as MuJoCo combines geom friction. Torsional / rolling (condim 4/6) have
        # no PhysX analogue and are declared by the mapping report.
        # A geom's collider sits at `<body>/collisions/<geom>/<geom>` (a world geom's at
        # `worldBody/<geom>/collisions/...`), so the material is bound on the prims named after
        # the geom and reaches the collider by inheritance.
        def has_collider(p):
            return any(q.HasAPI(UsdPhysics.CollisionAPI) for q in Usd.PrimRange(p))

        mats = {}
        missing = []
        for g in spec["geoms"]:
            if not g["collides"]:
                continue
            prims = [p for p in by_name.get(g["name"], []) if p.IsValid() and str(p.GetPath()).startswith(env0)
                     and "/visuals/" not in str(p.GetPath()) and has_collider(p)]
            if not prims:
                missing.append(g["name"])
                continue
            mu = g["mu"]
            if mu not in mats:
                mpath = Sdf.Path("%s/physics_materials/mu_%d" % (env0, len(mats)))
                mat = UsdShade.Material.Define(stage, mpath)
                api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                api.CreateStaticFrictionAttr().Set(mu)
                api.CreateDynamicFrictionAttr().Set(mu)
                api.CreateRestitutionAttr().Set(0.0)
                px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
                px.CreateFrictionCombineModeAttr().Set("max")
                mats[mu] = mat
            for p in prims:
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mats[mu], UsdShade.Tokens.weakerThanDescendants, "physics")
        if missing:
            for p in stage.Traverse():
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    sys.stderr.write("collider %s\n" % p.GetPath())
            raise RuntimeError("no collider prim for geom(s) %s after import" % ", ".join(missing))
        self.fixups.append("%d friction material(s), static = dynamic = mu, combine max" % len(mats))

        # Gravity from <option>, not World's default.
        g = spec["gravity"]
        mag = math.sqrt(sum(x * x for x in g))
        for p in stage.Traverse():
            if p.IsA(UsdPhysics.Scene):
                s = UsdPhysics.Scene(p)
                if mag > 0:
                    s.CreateGravityDirectionAttr().Set(Gf.Vec3f(*[x / mag for x in g]))
                s.CreateGravityMagnitudeAttr().Set(mag)

        roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        free_paths = {}
        for b in spec["free"]:
            hits = [p for p in by_name.get(b, []) if p.IsValid() and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(hits) != 1:
                raise RuntimeError("free body `%s`: %d rigid-body prims after import" % (b, len(hits)))
            free_paths[b] = str(hits[0].GetPath())
        # The pose each free body was imported at: MuJoCo's qpos0 for its free joint.
        from pxr import UsdGeom
        self.free0 = {}
        for b, fp in free_paths.items():
            m = UsdGeom.Xformable(stage.GetPrimAtPath(fp)).ComputeLocalToWorldTransform(0)
            t, r = m.ExtractTranslation(), m.ExtractRotationQuat()
            i = r.GetImaginary()
            self.free0[b] = [t[0], t[1], t[2], r.GetReal(), i[0], i[1], i[2]]

        if self.n > 1:
            from isaacsim.core.cloner import GridCloner

            cloner = GridCloner(spacing=0.0)
            paths = cloner.generate_paths("/World/envs/env", self.n)
            cloner.clone(source_prim_path=env0, prim_paths=paths, replicate_physics=False)
            scene_path = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)][0]
            cloner.filter_collisions(scene_path, "/World/collisions", paths, global_paths=[])
            self.fixups.append("%d envs cloned at one origin, collisions filtered between envs" % self.n)

        self.world.reset()  # steps physics twice (api-note 7.3); _initial() below undoes it
        import omni.physics.tensors as tensors

        self.view = tensors.create_simulation_view("torch" if self.gpu else "numpy")
        self.view.set_subspace_roots("/")
        star = lambda p: p.replace(env0, "/World/envs/env_*", 1)
        self.arts = []
        for r in roots:
            art = self.view.create_articulation_view(star(r))
            self.arts.append({"view": art, "rows": self._rows(art.prim_paths),
                              "dofs": list(art.shared_metatype.dof_names), "links": list(art.shared_metatype.link_names)})
        self.frees = []
        for b in sorted(free_paths):
            rb = self.view.create_rigid_body_view(star(free_paths[b]))
            self.frees.append({"body": b, "view": rb, "rows": self._rows(rb.prim_paths)})
        self._plan()
        self.ctrl = [[0.0] * len(spec["actuators"]) for _ in range(self.n)]
        self.reset(list(range(self.n)), None)

    def _rows(self, paths):
        """`rows[env]` = the view row holding that env (view order is the view's own)."""
        env = [int(p.split("/World/envs/env_", 1)[1].split("/", 1)[0]) for p in paths]
        if sorted(env) != list(range(self.n)):
            raise RuntimeError("a view covers envs %s, expected 0..%d" % (sorted(env), self.n - 1))
        rows = [0] * self.n
        for row, e in enumerate(env):
            rows[e] = row
        return rows

    def _plan(self):
        """Where every MJCF joint, actuator and body lives among the views; the drives."""
        spec = self.spec
        self.where = {}  # hinge/slide joint name -> (art index, dof index)
        for a, art in enumerate(self.arts):
            for d, name in enumerate(art["dofs"]):
                self.where[name] = (a, d)
        lost = [j["name"] for j in spec["joints"] if j["kind"] != "free" and j["name"] not in self.where]
        if lost:
            raise RuntimeError("joint(s) %s have no PhysX dof after import" % ", ".join(lost))
        self.links = {}
        for a, art in enumerate(self.arts):
            for k, name in enumerate(art["links"]):
                self.links[name] = (a, k)
        free = {f["body"]: i for i, f in enumerate(self.frees)}
        unplaced = [b for b in spec["bodies"][1:] if b not in self.links and b not in free]
        if unplaced:
            raise RuntimeError("body(ies) %s are neither an articulation link nor a free body" % ", ".join(unplaced))
        self.free_index = free
        # Drives: a position actuator's kp / kv as stiffness / damping, written through the
        # tensor API in SI units (N m / rad), never USD's per-degree attribute; forcerange as the
        # drive's max force. Every other dof has no drive.
        for a, art in enumerate(self.arts):
            nd = len(art["dofs"])
            if nd == 0:
                continue
            stiff, damp, fmax = [0.0] * nd, [0.0] * nd, [3.4e38] * nd
            for act in spec["actuators"]:
                if act["kind"] == "position" and self.where[act["joint"]][0] == a:
                    d = self.where[act["joint"]][1]
                    stiff[d], damp[d] = act["kp"], act["kv"]
                    if act["force"] is not None:
                        fmax[d] = act["force"][1]
            v = art["view"]
            idx = self._idx(v.count, cpu=True)
            v.set_dof_stiffnesses(self._t([stiff] * v.count, cpu=True), idx)
            v.set_dof_dampings(self._t([damp] * v.count, cpu=True), idx)
            v.set_dof_max_forces(self._t([fmax] * v.count, cpu=True), idx)
        self.fixups.append("drives authored from the MJCF: stiffness kp, damping kv (SI), max force forcerange")

    # ---- tensors --------------------------------------------------------------------------

    def _t(self, rows, cpu=False):
        if self.gpu:
            import torch
            return torch.tensor(rows, dtype=torch.float32, device="cpu" if cpu else self.device)
        import numpy as np
        return np.asarray(rows, dtype=np.float32)

    def _idx(self, n, cpu=False):
        if self.gpu:
            import torch
            return torch.arange(n, dtype=torch.int32, device="cpu" if cpu else self.device)
        import numpy as np
        return np.arange(n, dtype=np.int32)

    @staticmethod
    def _list(t):
        if hasattr(t, "detach"):
            t = t.detach().cpu().numpy()
        return t.tolist()

    # ---- state ----------------------------------------------------------------------------

    def _read(self):
        """Per-env (qpos, qvel, xpos, xquat) in MuJoCo's layout."""
        dof_q = [self._list(a["view"].get_dof_positions()) if a["dofs"] else None for a in self.arts]
        dof_v = [self._list(a["view"].get_dof_velocities()) if a["dofs"] else None for a in self.arts]
        links = [self._list(a["view"].get_link_transforms()) for a in self.arts]
        tf = [self._list(f["view"].get_transforms()) for f in self.frees]
        vel = [self._list(f["view"].get_velocities()) for f in self.frees]
        com = [self._list(f["view"].get_coms()) for f in self.frees]
        out = []
        for e in range(self.n):
            pose = {}  # body -> (pos, quat wxyz)
            for b, (a, k) in self.links.items():
                x = links[a][self.arts[a]["rows"][e]][k]
                pose[b] = (x[0:3], [x[6], x[3], x[4], x[5]])
            fvel = {}
            for i, f in enumerate(self.frees):
                row = f["rows"][e]
                x = tf[i][row]
                q = [x[6], x[3], x[4], x[5]]
                pose[f["body"]] = (x[0:3], q)
                c = qrot(q, com[i][row][0:3])  # origin -> centre of mass, world frame
                w = vel[i][row][3:6]
                v_origin = [vc - wc for vc, wc in zip(vel[i][row][0:3], cross(w, c))]
                fvel[f["body"]] = v_origin + qrot(qconj(q), w)
            qpos, qvel = [], []
            for j in self.spec["joints"]:
                if j["kind"] == "free":
                    p, q = pose[j["body"]]
                    qpos += list(p) + list(q)
                    qvel += fvel[j["body"]]
                else:
                    a, d = self.where[j["name"]]
                    row = self.arts[a]["rows"][e]
                    qpos.append(dof_q[a][row][d])
                    qvel.append(dof_v[a][row][d])
            xpos, xquat = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
            for b in self.spec["bodies"][1:]:
                p, q = pose[b]
                xpos += list(p)
                xquat += [q[1], q[2], q[3], q[0]]
            out.append((qpos, qvel, xpos, xquat))
        return out

    def _write(self, envs, qpos, qvel):
        """Writes MuJoCo-layout state rows (one per env in `envs`) into the views."""
        nq, nv = self.spec["nq"], self.spec["nv"]
        for a, art in enumerate(self.arts):
            if not art["dofs"]:
                continue
            v = art["view"]
            q = self._list(v.get_dof_positions())
            qd = self._list(v.get_dof_velocities())
            for k, e in enumerate(envs):
                row = art["rows"][e]
                for j in self.spec["joints"]:
                    if j["kind"] != "free" and self.where[j["name"]][0] == a:
                        d = self.where[j["name"]][1]
                        q[row][d] = qpos[k * nq + j["qpos"][0]]
                        qd[row][d] = qvel[k * nv + j["dof"][0]]
            idx = self._idx(v.count)
            v.set_dof_positions(self._t(q), idx)
            v.set_dof_velocities(self._t(qd), idx)
        for i, f in enumerate(self.frees):
            v = f["view"]
            tf = self._list(v.get_transforms())
            vel = self._list(v.get_velocities())
            com = self._list(v.get_coms())
            j = next(j for j in self.spec["joints"] if j["kind"] == "free" and j["body"] == f["body"])
            for k, e in enumerate(envs):
                row = f["rows"][e]
                p = qpos[k * nq + j["qpos"][0]: k * nq + j["qpos"][0] + 7]
                u = qvel[k * nv + j["dof"][0]: k * nv + j["dof"][0] + 6]
                q = p[3:7]
                w = qrot(q, u[3:6])
                c = qrot(q, com[row][0:3])
                tf[row] = p[0:3] + [q[1], q[2], q[3], q[0]]
                vel[row] = [vo + wc for vo, wc in zip(u[0:3], cross(w, c))] + w
            idx = self._idx(v.count)
            v.set_transforms(self._t(tf), idx)
            v.set_velocities(self._t(vel), idx)
        if hasattr(self.view, "update_articulations_kinematic"):
            self.view.update_articulations_kinematic()

    def _initial(self):
        """MuJoCo's qpos0 / zero velocity: joints at 0, free bodies at their imported pose."""
        qpos, qvel = [], []
        for j in self.spec["joints"]:
            qpos += self.free0[j["body"]] if j["kind"] == "free" else [0.0]
            qvel += [0.0] * j["dof"][1]
        return qpos, qvel

    def state(self):
        rows = self._read()
        return {"qpos": [x for r in rows for x in r[0]], "qvel": [x for r in rows for x in r[1]],
                "act": [], "sensordata": [], "xpos": [x for r in rows for x in r[2]],
                "xquat": [x for r in rows for x in r[3]]}

    def reset(self, envs, state):
        for e in envs:
            self.ctrl[e] = [0.0] * len(self.spec["actuators"])
        if state is not None and state.get("qpos"):
            qpos = state["qpos"]
            qvel = state.get("qvel") or [0.0] * (self.spec["nv"] * len(envs))
        else:
            q0, v0 = self._initial()
            qpos, qvel = q0 * len(envs), v0 * len(envs)
        self._write(envs, qpos, qvel)
        self._apply_targets()

    def set_state(self, state):
        envs = list(range(self.n))
        if not state.get("qpos"):
            return
        qvel = state.get("qvel") or [0.0] * (self.spec["nv"] * self.n)
        self._write(envs, state["qpos"], qvel)

    def set_ctrl(self, ctrl):
        nu = len(self.spec["actuators"])
        if len(ctrl) != nu * self.n:
            raise ValueError("ctrl has %d values, expected %d" % (len(ctrl), nu * self.n))
        self.ctrl = [list(ctrl[e * nu:(e + 1) * nu]) for e in range(self.n)]
        self._apply_targets()

    def _clamped(self, e, i):
        act, u = self.spec["actuators"][i], self.ctrl[e][i]
        if act["ctrl"] is not None:  # MuJoCo clamps ctrl to ctrlrange; the importer drops it
            u = min(max(u, act["ctrl"][0]), act["ctrl"][1])
        return u

    def _apply_targets(self):
        for a, art in enumerate(self.arts):
            if not art["dofs"]:
                continue
            v = art["view"]
            t = [[0.0] * len(art["dofs"]) for _ in range(v.count)]
            for e in range(self.n):
                for i, act in enumerate(self.spec["actuators"]):
                    if act["kind"] == "position" and self.where[act["joint"]][0] == a:
                        t[art["rows"][e]][self.where[act["joint"]][1]] = self._clamped(e, i)
            v.set_dof_position_targets(self._t(t), self._idx(v.count))

    def _apply_efforts(self):
        """Joint forces PhysX has no passive analogue for, recomputed before every physics step
        from the current state (explicit): MuJoCo's passive joint damping and spring, and motor
        actuators (gear * ctrl, clamped to forcerange)."""
        for a, art in enumerate(self.arts):
            if not art["dofs"]:
                continue
            passive = [j for j in self.spec["joints"] if j["kind"] != "free" and self.where[j["name"]][0] == a
                       and (j["damping"] or j["stiffness"])]
            motors = [(i, act) for i, act in enumerate(self.spec["actuators"])
                      if act["kind"] == "motor" and self.where[act["joint"]][0] == a]
            if not passive and not motors:
                continue
            v = art["view"]
            q = self._list(v.get_dof_positions()) if passive else None
            qd = self._list(v.get_dof_velocities()) if passive else None
            f = [[0.0] * len(art["dofs"]) for _ in range(v.count)]
            for e in range(self.n):
                row = art["rows"][e]
                for j in passive:
                    d = self.where[j["name"]][1]
                    f[row][d] -= j["damping"] * qd[row][d] + j["stiffness"] * (q[row][d] - j["springref"])
                for i, act in motors:
                    d = self.where[act["joint"]][1]
                    u = act["gear"] * self._clamped(e, i)
                    if act["force"] is not None:
                        u = min(max(u, act["force"][0]), act["force"][1])
                    f[row][d] += u
            v.set_dof_actuation_forces(self._t(f), self._idx(v.count))

    def step(self, n):
        for _ in range(n):
            self._apply_efforts()
            self.world.step(render=False)
        nonfinite = []
        for e, (qpos, qvel, _, _) in enumerate(self._read()):
            if not all(math.isfinite(x) for x in qpos + qvel):
                nonfinite.append(e)
        return nonfinite

    def info(self):
        from isaacsim.core.version import get_version
        import omni.kit.app

        ver = get_version()
        isaac = ver[0] if isinstance(ver, (list, tuple)) else str(ver)
        physx = "unknown"
        for ext in omni.kit.app.get_app().get_extension_manager().get_extensions():
            if ext.get("name") == "omni.physx":
                physx = str(ext.get("version"))
                if isinstance(ext.get("version"), (list, tuple)):
                    physx = ".".join(str(x) for x in ext["version"][:3])
        spec = self.spec
        return {
            "nq": spec["nq"], "nv": spec["nv"], "nu": len(spec["actuators"]), "nsensordata": 0,
            "nbody": len(spec["bodies"]),
            "joints": [{"name": j["name"], "qpos": j["qpos"], "dof": j["dof"]} for j in spec["joints"]],
            "actuators": [a["name"] for a in spec["actuators"]],
            "sensors": [],
            "bodies": spec["bodies"],
            # The pipeline is part of the engine: CPU and GPU PhysX differ (api-note 7.4), so
            # the version string -- and with it backend_identity -- names it.
            "engine_version": "isaacsim %s physx %s %s" % (isaac, physx, "gpu" if self.gpu else "cpu"),
            "experience": self.experience,
            "fixups": self.fixups,
        }


def handle(sim, req):
    cmd = req.get("cmd")
    if cmd == "load":
        if sim is not None:
            raise ValueError("this process already holds a stage; start a new one per load")
        timestep = req.get("timestep")
        if timestep is None:
            opt = ET.fromstring(req["mjcf"]).find("option")
            timestep = float(opt.get("timestep", "0.002")) if opt is not None else 0.002
        device = os.environ.get("ES_PHYSX_DEVICE", "cpu")
        sim = Sim(req["mjcf"], int(req["n_envs"]), float(timestep), device)
        return sim, sim.info()
    if sim is None:
        raise ValueError("no model loaded")
    if cmd == "reset":
        envs = req.get("envs")
        sim.reset(list(range(sim.n)) if envs is None else [int(e) for e in envs], req.get("state"))
        return sim, {}
    if cmd == "set_ctrl":
        sim.set_ctrl(req["ctrl"])
        return sim, {}
    if cmd == "step":
        return sim, {"nonfinite": sim.step(int(req["n"]))}
    if cmd == "state":
        return sim, sim.state()
    if cmd == "set_state":
        sim.set_state(req["state"])
        return sim, {}
    raise ValueError("unknown command %r" % (cmd,))


def main():
    sim = None
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            sim, payload = handle(sim, req)
            payload["ok"] = True
        except Exception as exc:  # Any failure is a protocol response, never a crash.
            import traceback

            traceback.print_exc()
            payload = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        PROTO.write(json.dumps(payload) + "\n")
        PROTO.flush()
    PROTO.flush()
    # SimulationApp.close() can hang or exit on its own; nothing is left to say.
    os._exit(0)


if __name__ == "__main__":
    main()
