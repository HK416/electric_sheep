"""MuJoCo Warp reference process for `MjWarpBackend` (spec 17.1: the batched GPU path).

Same line-delimited JSON protocol as `mujoco_ref.py`, so the two backends are interchangeable
from Rust:

    {"cmd": "load", "mjcf": str, "n_envs": int, "timestep": float|null, "seed": int}
    {"cmd": "reset", "envs": [int]|null, "state": {...}|null}
    {"cmd": "set_ctrl", "ctrl": [float]}          # n_envs * nu, env-major
    {"cmd": "step", "n": int}
    {"cmd": "state"}
    {"cmd": "set_state", "state": {"qpos": [...], "qvel": [...], "act": [...]}}
    {"cmd": "quit"}

Every response is {"ok": true, ...} or {"ok": false, "error": str}.

`n_envs` is `nworld` here: unlike the CPU reference, this is a real batch on one device
(spec 12.1). Model metadata still comes from the CPU `MjModel`, which `put_model` is built
from, so the `load` reply has exactly the fields `mujoco_ref.py` returns.

`mujoco_warp` is young: every call this script makes is listed in docs/api-notes/mujoco-warp.md
as **unverified** (no GPU in CI). Failures surface as protocol errors, never as a traceback.
"""

import json
import sys

# `warp` prints an initialization banner ("Warp x.y.z initialized:" and a device table) on
# stdout, and stdout is the protocol. The real handle is taken first and everything else --
# imports, `wp.init()`, any future chatter -- is pointed at stderr, which Rust discards.
_OUT = sys.stdout
sys.stdout = sys.stderr

try:
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp
except ImportError as exc:  # Reported as a protocol response, not a traceback on stderr.
    _OUT.write(json.dumps({"ok": False, "error": "import failed: %s" % exc}) + "\n")
    _OUT.flush()
    raise SystemExit(1)

# qpos / dof width per joint type, indexed by mjtJoint (free, ball, slide, hinge).
JOINT_DIMS = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}


def name_of(model, objtype, index):
    return mujoco.mj_id2name(model, objtype, index) or ""


def flat(arr):
    """A warp array as a flat list of float64, env-major (its first axis is nworld)."""
    return np.asarray(arr.numpy(), dtype=np.float64).reshape(-1).tolist()


class Sim(object):
    def __init__(self, mjcf, n_envs, timestep, seed):
        self.mjm = mujoco.MjModel.from_xml_string(mjcf)
        if timestep is not None:
            self.mjm.opt.timestep = timestep
        self.seed = seed
        self.n_envs = n_envs
        # The CPU world is kept as the reset reference: mj_resetData defines "initial state"
        # for both backends, so a reset here means the same thing as on mujoco-cpu.
        self.mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, self.mjd)
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.put_data(self.mjm, self.mjd, nworld=n_envs)

    def info(self):
        model = self.mjm
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
        }

    def state(self):
        # MuJoCo stores wxyz; spec 3.1 is xyzw.
        wxyz = np.asarray(self.d.xquat.numpy(), dtype=np.float64).reshape(-1, 4)
        return {
            "qpos": flat(self.d.qpos),
            "qvel": flat(self.d.qvel),
            "act": flat(self.d.act),
            "sensordata": flat(self.d.sensordata),
            "xpos": flat(self.d.xpos),
            "xquat": wxyz[:, [1, 2, 3, 0]].reshape(-1).tolist(),
        }

    def write_field(self, name, values, envs, width):
        """Overwrites rows `envs` of one state field from a flat env-major list."""
        if not values or width == 0:
            return
        array = getattr(self.d, name)
        rows = np.array(array.numpy(), copy=True).reshape(self.n_envs, width)
        for row, env in enumerate(envs):
            chunk = values[row * width : (row + 1) * width]
            if len(chunk) != width:
                raise ValueError("%s row %d has %d values, expected %d" % (name, row, len(chunk), width))
            rows[env] = np.asarray(chunk, dtype=rows.dtype)
        array.assign(np.ascontiguousarray(rows.reshape(array.shape)))

    def widths(self):
        return {
            "qpos": int(self.mjm.nq),
            "qvel": int(self.mjm.nv),
            "act": int(self.mjm.na),
        }

    def write_state(self, state, envs):
        widths = self.widths()
        for field in ("qpos", "qvel", "act"):
            self.write_field(field, state.get(field), envs, widths[field])
        mjw.forward(self.m, self.d)

    def reset(self, envs, state):
        mujoco.mj_resetData(self.mjm, self.mjd)
        mujoco.mj_forward(self.mjm, self.mjd)
        widths = self.widths()
        for field in ("qpos", "qvel", "act"):
            width = widths[field]
            if width == 0:
                continue
            initial = np.asarray(getattr(self.mjd, field), dtype=np.float64).reshape(-1).tolist()
            self.write_field(field, initial * len(envs), envs, width)
        if self.mjm.nu:
            self.write_field("ctrl", [0.0] * (self.mjm.nu * len(envs)), envs, int(self.mjm.nu))
        if state is not None:
            self.write_state(state, envs)
        else:
            mjw.forward(self.m, self.d)

    def set_ctrl(self, ctrl):
        nu = int(self.mjm.nu)
        expected = nu * self.n_envs
        if len(ctrl) != expected:
            raise ValueError("ctrl has %d values, expected %d" % (len(ctrl), expected))
        self.write_field("ctrl", ctrl, range(self.n_envs), nu)

    def step(self, n):
        for _ in range(n):
            mjw.step(self.m, self.d)
        qpos = np.asarray(self.d.qpos.numpy(), dtype=np.float64).reshape(self.n_envs, -1)
        qvel = np.asarray(self.d.qvel.numpy(), dtype=np.float64).reshape(self.n_envs, -1)
        finite = np.isfinite(qpos).all(axis=1) & np.isfinite(qvel).all(axis=1)
        return [int(env) for env in np.flatnonzero(~finite)]


def handle(sim, req):
    cmd = req.get("cmd")
    if cmd == "load":
        sim = Sim(req["mjcf"], int(req["n_envs"]), req.get("timestep"), int(req.get("seed", 0)))
        return sim, sim.info()
    if sim is None:
        raise ValueError("no model loaded")
    if cmd == "reset":
        envs = req.get("envs")
        sim.reset(range(sim.n_envs) if envs is None else [int(e) for e in envs], req.get("state"))
        return sim, {}
    if cmd == "set_ctrl":
        sim.set_ctrl(req["ctrl"])
        return sim, {}
    if cmd == "step":
        return sim, {"nonfinite": sim.step(int(req["n"]))}
    if cmd == "state":
        return sim, sim.state()
    if cmd == "set_state":
        sim.write_state(req["state"], range(sim.n_envs))
        return sim, {}
    raise ValueError("unknown command %r" % (cmd,))


def main():
    wp.init()
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
        _OUT.write(json.dumps(payload) + "\n")
        _OUT.flush()


if __name__ == "__main__":
    main()
