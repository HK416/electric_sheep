"""MuJoCo (CPU) reference process for `MuJoCoCpuBackend` (spec 17.1: the CI oracle).

Line-delimited JSON on stdin, one JSON object per line on stdout. Requests:

    {"cmd": "load", "mjcf": str, "n_envs": int, "timestep": float|null, "seed": int}
    {"cmd": "reset", "envs": [int]|null, "state": {...}|null}
    {"cmd": "set_ctrl", "ctrl": [float]}          # n_envs * nu, env-major
    {"cmd": "step", "n": int}
    {"cmd": "state"}
    {"cmd": "set_state", "state": {"qpos": [...], "qvel": [...], "act": [...]}}
    {"cmd": "set_params", "envs": [int], "params": [{"field", "index", "sub", "scale"}]}
    {"cmd": "quit"}

Every response is {"ok": true, ...} or {"ok": false, "error": str}. Floats cross as JSON
numbers: `repr` is shortest-round-trip on both sides, so the values are exact.

Correctness over speed. Envs are independent `MjData` stepped in a loop; this process is the
reference, not a throughput path. See docs/api-notes/mujoco.md for the pinned API surface.
"""

import copy
import json
import sys

try:
    import mujoco
    import numpy as np
except ImportError as exc:  # Reported as a protocol response, not a traceback on stderr.
    sys.stdout.write(json.dumps({"ok": False, "error": "import failed: %s" % exc}) + "\n")
    sys.stdout.flush()
    raise SystemExit(1)

# qpos / dof width per joint type, indexed by mjtJoint (free, ball, slide, hinge).
JOINT_DIMS = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}


def name_of(model, objtype, index):
    return mujoco.mj_id2name(model, objtype, index) or ""


def write_param(orig, model, p):
    """Writes `orig`'s value times the scale into `model`; returns the (field, row, column) read
    back for the reply. Always from `orig`, the loaded model, so scales never compound."""
    field, i, scale = p["field"], int(p["index"]), float(p["scale"])
    if field == "body_mass":
        model.body_mass[i] = orig.body_mass[i] * scale
        return ("body_mass", i, None)
    if field == "geom_friction":
        sub = int(p["sub"])
        if sub >= int(orig.body_geomnum[i]):
            raise ValueError("body %d has no geom %d" % (i, sub))
        g = int(orig.body_geomadr[i]) + sub
        # Sliding, torsional and rolling together.
        model.geom_friction[g, 0:3] = orig.geom_friction[g, 0:3] * scale
        return ("geom_friction", g, 0)
    if field == "actuator_gain":
        gain = orig.actuator_gainprm[i, 0]
        model.actuator_gainprm[i, 0] = gain * scale
        # A servo's bias mirrors its gain (position: biasprm[1] = -kp; velocity:
        # biasprm[2] = -kv), so it is scaled with it and the servo stays a servo.
        for k in (1, 2):
            if gain != 0 and orig.actuator_biasprm[i, k] == -gain:
                model.actuator_biasprm[i, k] = orig.actuator_biasprm[i, k] * scale
                break
        return ("actuator_gainprm", i, 0)
    raise ValueError("unknown parameter field %r" % (field,))


def read_param(model, slot):
    field, row, col = slot
    array = getattr(model, field)
    return float(array[row] if col is None else array[row, col])


def version_of(module, dist):
    """`module.__version__`, else the installed distribution's; an engine with neither fails
    the load, because the version is hashed into the run's condition (packet M11/X1)."""
    version = getattr(module, "__version__", None)
    if not version:
        from importlib import metadata

        version = metadata.version(dist)
    return str(version)


class Sim(object):
    def __init__(self, mjcf, n_envs, timestep, seed):
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.seed = seed
        self.datas = [mujoco.MjData(self.model) for _ in range(n_envs)]
        # One model per env exists only once `set_params` has been called; until then every
        # env steps the one loaded model, byte for byte what it did before per-env models.
        self.models = None
        for data in self.datas:
            mujoco.mj_forward(self.model, data)

    def model_of(self, env):
        return self.model if self.models is None else self.models[env]

    def set_params(self, envs, params):
        if self.models is None:
            self.models = [copy.copy(self.model) for _ in self.datas]
        values = []
        for env in envs:
            model = self.models[env]
            slots = [write_param(self.model, model, p) for p in params]
            # Mass feeds derived constants (body_subtreemass, the invweight0s, actuator_acc0,
            # meaninertia); friction and gain feed none.
            if any(p["field"] == "body_mass" for p in params):
                mujoco.mj_setConst(model, mujoco.MjData(model))
            for slot in slots:
                values.append([read_param(self.model, slot), read_param(model, slot)])
        return values

    def info(self):
        model = self.model
        joints = []
        for i in range(model.njnt):
            nq, nv = JOINT_DIMS[int(model.jnt_type[i])]
            joints.append(
                {
                    "name": name_of(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                    "qpos": [int(model.jnt_qposadr[i]), nq],
                    "dof": [int(model.jnt_dofadr[i]), nv],
                }
            )
        sensors = [
            {
                "name": name_of(model, mujoco.mjtObj.mjOBJ_SENSOR, i),
                "adr": int(model.sensor_adr[i]),
                "dim": int(model.sensor_dim[i]),
            }
            for i in range(model.nsensor)
        ]
        return {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nsensordata": int(model.nsensordata),
            "nbody": int(model.nbody),
            "joints": joints,
            "actuators": [
                name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
            ],
            "sensors": sensors,
            "bodies": [
                name_of(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
            ],
            "engine_version": "mujoco %s" % version_of(mujoco, "mujoco"),
        }

    def state(self):
        qpos, qvel, act, sensordata, xpos, xquat = [], [], [], [], [], []
        for data in self.datas:
            qpos.extend(data.qpos.tolist())
            qvel.extend(data.qvel.tolist())
            act.extend(data.act.tolist())
            sensordata.extend(data.sensordata.tolist())
            xpos.extend(data.xpos.reshape(-1).tolist())
            # MuJoCo stores wxyz; spec 3.1 is xyzw.
            wxyz = data.xquat.reshape(-1, 4)
            xquat.extend(wxyz[:, [1, 2, 3, 0]].reshape(-1).tolist())
        return {
            "qpos": qpos,
            "qvel": qvel,
            "act": act,
            "sensordata": sensordata,
            "xpos": xpos,
            "xquat": xquat,
        }

    def write_state(self, state, envs):
        for row, env in enumerate(envs):
            data = self.datas[env]
            for field in ("qpos", "qvel", "act"):
                values = state.get(field)
                if not values:
                    continue
                width = len(getattr(data, field))
                if width:
                    chunk = values[row * width : (row + 1) * width]
                    getattr(data, field)[:] = np.asarray(chunk, dtype=np.float64)
            mujoco.mj_forward(self.model_of(env), data)

    def reset(self, envs, state):
        for env in envs:
            mujoco.mj_resetData(self.model_of(env), self.datas[env])
        if state is not None:
            self.write_state(state, envs)
        else:
            for env in envs:
                mujoco.mj_forward(self.model_of(env), self.datas[env])

    def set_ctrl(self, ctrl):
        nu = self.model.nu
        expected = nu * len(self.datas)
        if len(ctrl) != expected:
            raise ValueError("ctrl has %d values, expected %d" % (len(ctrl), expected))
        for env, data in enumerate(self.datas):
            if nu:
                data.ctrl[:] = np.asarray(ctrl[env * nu : (env + 1) * nu], dtype=np.float64)

    def step(self, n):
        nonfinite = []
        for env, data in enumerate(self.datas):
            model = self.model_of(env)
            for _ in range(n):
                mujoco.mj_step(model, data)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                nonfinite.append(env)
        return nonfinite


def handle(sim, req):
    cmd = req.get("cmd")
    if cmd == "load":
        sim = Sim(req["mjcf"], int(req["n_envs"]), req.get("timestep"), int(req.get("seed", 0)))
        return sim, sim.info()
    if sim is None:
        raise ValueError("no model loaded")
    if cmd == "reset":
        envs = req.get("envs")
        sim.reset(range(len(sim.datas)) if envs is None else [int(e) for e in envs], req.get("state"))
        return sim, {}
    if cmd == "set_ctrl":
        sim.set_ctrl(req["ctrl"])
        return sim, {}
    if cmd == "step":
        return sim, {"nonfinite": sim.step(int(req["n"]))}
    if cmd == "state":
        return sim, sim.state()
    if cmd == "set_state":
        sim.write_state(req["state"], range(len(sim.datas)))
        return sim, {}
    if cmd == "set_params":
        return sim, {"values": sim.set_params([int(e) for e in req["envs"]], req["params"])}
    raise ValueError("unknown command %r" % (cmd,))


def main():
    sim = None
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                return
            sim, payload = handle(sim, req)
            payload["ok"] = True
        except Exception as exc:  # Any failure is a protocol response, never a crash.
            payload = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
