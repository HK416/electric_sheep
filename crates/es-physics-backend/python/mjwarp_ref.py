"""MuJoCo Warp reference process for `MjWarpBackend` (spec 17.1: the batched GPU path).

Same line-delimited JSON protocol as `mujoco_ref.py`, so the two backends are interchangeable
from Rust:

    {"cmd": "load", "mjcf": str, "n_envs": int, "timestep": float|null, "seed": int}
    {"cmd": "reset", "envs": [int]|null, "state": {...}|null}
    {"cmd": "set_ctrl", "ctrl": [float]}          # n_envs * nu, env-major
    {"cmd": "set_ctrl", "frame": int}             # the same, as float64 bytes after the line
    {"cmd": "step", "n": int}
    {"cmd": "state"}
    {"cmd": "set_state", "state": {"qpos": [...], "qvel": [...], "act": [...]}}
    {"cmd": "set_params", "envs": [int], "params": [{"field", "index", "sub", "scale"}]}
    {"cmd": "quit"}

Every response is {"ok": true, ...} or {"ok": false, "error": str}; the `state` reply is a
line {"ok": true, "frame": [six lengths]} followed by the arrays' float64 bytes (`answer`,
packet M16/H0).

`n_envs` is `nworld` here: unlike the CPU reference, this is a real batch on one device
(spec 12.1). Model metadata still comes from the CPU `MjModel`, which `put_model` is built
from, so the `load` reply has exactly the fields `mujoco_ref.py` returns.

`mujoco_warp` is young: every call this script makes is listed in docs/api-notes/mujoco-warp.md
as **unverified** (no GPU in CI). Failures surface as protocol errors, never as a traceback.
"""

import copy
import json
import os
import sys

# `warp` prints an initialization banner ("Warp x.y.z initialized:" and a device table) on
# stdout, and stdout is the protocol. The real handle is taken first and everything else --
# imports, `wp.init()`, any future chatter -- is pointed at stderr, which Rust discards. At the
# *descriptor* level, not just `sys.stdout`: a kernel's `printf` (mujoco_warp's "nefc overflow"
# warning, say) writes to fd 1 from C and would otherwise land mid-protocol (packet M11/X1).
_OUT = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr

# Per-world constraint and contact capacity. mujoco_warp sizes these from the model when not
# told, and on the SO-101 scene that is too small (`nefc overflow ... njmax beyond 64`), which
# drops constraints silently. A declared bound, not a measurement (packet M11/X1).
NJMAX = 1024
NCONMAX = 256

try:
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp
except ImportError as exc:  # Reported as a protocol response, not a traceback on stderr.
    _OUT.write(json.dumps({"ok": False, "error": "import failed: %s" % exc}) + "\n")
    _OUT.flush()
    raise SystemExit(1)


def patch_ccd_grid_size():
    """Works around an upstream failure (mujoco_warp 3.13.0, warp-lang 1.16.0; found by packet
    M16/H1 on the Shadow Hand scene, reproduced in H0): with the CCD module already in warp's
    kernel cache, `collision_convex._ccd_grid_size` asks `wp.get_suggested_block_size` about a
    CCD kernel the loaded module's metadata does not list, and `mjw.step` raises
    `KeyError: 'ccd_kernel_builder__locals__ccd_kernel_<hash>_cuda_kernel_forward_smem_bytes'`.
    On that error the kernel's module is unloaded (warp rehashes it on the next load) and the
    query retried once; if it fails again the grid is `naconmax`, the CPU branch's width -- the
    kernel grid-strides over the candidates, so the contacts are the same and only the launch
    width differs. A mujoco_warp without the function is left alone."""
    try:
        from mujoco_warp._src import collision_convex
    except ImportError:
        return
    upstream = getattr(collision_convex, "_ccd_grid_size", None)
    if upstream is None:
        return

    def ccd_grid_size(kernel, naconmax, device):
        try:
            return upstream(kernel, naconmax, device)
        except KeyError:
            kernel.module.unload()
        try:
            return upstream(kernel, naconmax, device)
        except KeyError:
            return naconmax

    collision_convex._ccd_grid_size = ccd_grid_size


patch_ccd_grid_size()

# qpos / dof width per joint type, indexed by mjtJoint (free, ball, slide, hinge).
JOINT_DIMS = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}

# MuJoCo's mjMAXVAL: a qpos, qvel or qacc past it is "bad" to `mj_step` (`Sim.step`).
MAXVAL = 1e10


def name_of(model, objtype, index):
    return mujoco.mj_id2name(model, objtype, index) or ""


def write_param(orig, model, p):
    """`mujoco_ref.py`'s edit, verbatim: `orig`'s value times the scale into `model`, returning
    the (field, row, column) to read back."""
    field, i, scale = p["field"], int(p["index"]), float(p["scale"])
    if field == "body_mass":
        model.body_mass[i] = orig.body_mass[i] * scale
        return ("body_mass", i, None)
    if field == "geom_friction":
        sub = int(p["sub"])
        if sub >= int(orig.body_geomnum[i]):
            raise ValueError("body %d has no geom %d" % (i, sub))
        g = int(orig.body_geomadr[i]) + sub
        model.geom_friction[g, 0:3] = orig.geom_friction[g, 0:3] * scale
        return ("geom_friction", g, 0)
    if field == "actuator_gain":
        gain = orig.actuator_gainprm[i, 0]
        model.actuator_gainprm[i, 0] = gain * scale
        for k in (1, 2):
            if gain != 0 and orig.actuator_biasprm[i, k] == -gain:
                model.actuator_biasprm[i, k] = orig.actuator_biasprm[i, k] * scale
                break
        return ("actuator_gainprm", i, 0)
    raise ValueError("unknown parameter field %r" % (field,))


def model_arrays(model):
    """Every array field of a CPU `MjModel`, by name, plus `stat.meaninertia`."""
    names = [n for n in dir(model) if not n.startswith("_")]
    out = {n: getattr(model, n) for n in names if isinstance(getattr(model, n), np.ndarray)}
    out["stat.meaninertia"] = np.asarray([model.stat.meaninertia])
    return out


def flat(arr):
    """A warp array as a flat float64 array, env-major (its first axis is nworld)."""
    return np.asarray(arr.numpy(), dtype=np.float64).reshape(-1)


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
        self.d = mjw.put_data(
            self.mjm, self.mjd, nworld=n_envs, nconmax=NCONMAX, njmax=NJMAX
        )
        # Per-env CPU models and the model fields made per-world, both only once `set_params`
        # is called; until then every world shares the one loaded model.
        self.models = None
        self.batched = set()
        # `replay`'s captured graphs, by key. A graph holds the arrays it was captured on, so
        # `set_params` drops every graph when it replaces a model array.
        self.graphs = {}

    def replay(self, key, launch):
        """Runs `launch` (mjw calls on `self.m` / `self.d`) as a CUDA graph captured right
        after its first eager run, which builds every kernel module: the same kernels on the
        same arrays, without the Python cost of issuing each launch -- ~30x for `step` on SO-101
        at 1,024 worlds (packet M16/H0). Capturing enqueues nothing, so each call runs once."""
        graph = self.graphs.get(key)
        if graph is not None:
            wp.capture_launch(graph)
            return
        launch()
        device = self.d.qpos.device
        if device.is_cuda and wp.is_mempool_enabled(device):
            with wp.ScopedCapture(device=device) as capture:
                launch()
            self.graphs[key] = capture.graph

    def forward(self):
        self.replay("forward", lambda: mjw.forward(self.m, self.d))

    def warp_field(self, name):
        if name == "stat.meaninertia":
            return self.m.stat, "meaninertia"
        return self.m, name

    def set_params(self, envs, params):
        if self.models is None:
            self.models = [copy.copy(self.mjm) for _ in range(self.n_envs)]
        pristine = model_arrays(self.mjm)
        slots, edited = [], {}
        for env in envs:
            model = self.models[env]
            slots = [write_param(self.mjm, model, p) for p in params]
            # Mass feeds derived constants; `mj_setConst` derives them exactly as on mujoco-cpu.
            if any(p["field"] == "body_mass" for p in params):
                mujoco.mj_setConst(model, mujoco.MjData(model))
            edited[env] = model_arrays(model)
            for name, value in edited[env].items():
                if not np.array_equal(value, pristine[name], equal_nan=value.dtype.kind == "f"):
                    owner, attr = self.warp_field(name)
                    # A derived field mujoco_warp does not hold (e.g. dof_M0) feeds nothing.
                    if getattr(owner, attr, None) is not None:
                        self.batched.add(name)
        for name in sorted(self.batched):
            owner, attr = self.warp_field(name)
            array = getattr(owner, attr)
            host = np.array(array.numpy(), copy=True)
            if host.shape[0] == 1 and self.n_envs > 1:
                host = np.repeat(host, self.n_envs, axis=0)
            elif host.shape[0] != self.n_envs:
                raise ValueError("%s is not a per-world field in mujoco_warp (shape %s)" % (name, host.shape))
            for env in envs:
                value = edited[env][name]
                if value.size != host[env].size:
                    raise ValueError("%s: %d values per world, the CPU model has %d" % (name, host[env].size, value.size))
                host[env] = np.asarray(value).reshape(host[env].shape)
            if host.shape == tuple(array.shape):
                # Already per-world: written in place, so the captured graphs stay valid.
                array.assign(host)
            else:
                setattr(owner, attr, wp.array(host, dtype=array.dtype, device=array.device))
                self.graphs = {}
        values = []
        for env in envs:
            for field, row, col in slots:
                world = getattr(self.m, field).numpy()
                nominal, applied = getattr(self.mjm, field)[row], world[env % world.shape[0]][row]
                if col is not None:
                    nominal, applied = nominal[col], applied[col]
                values.append([float(nominal), float(applied)])
        return values

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
            "engine_version": "mujoco_warp %s; warp %s; mujoco %s"
            % (
                version_of(mjw, "mujoco-warp"),
                version_of(wp, "warp-lang"),
                version_of(mujoco, "mujoco"),
            ),
        }

    def state(self):
        # MuJoCo stores wxyz; spec 3.1 is xyzw.
        wxyz = np.asarray(self.d.xquat.numpy(), dtype=np.float64).reshape(-1, 4)
        return {
            "frame": [
                flat(self.d.qpos),
                flat(self.d.qvel),
                flat(self.d.act),
                flat(self.d.sensordata),
                flat(self.d.xpos),
                wxyz[:, [1, 2, 3, 0]].reshape(-1),
            ]
        }

    def write_field(self, name, values, envs, width):
        """Overwrites rows `envs` of one state field from a flat env-major list."""
        if len(values) == 0 or width == 0:
            return
        array = getattr(self.d, name)
        rows = np.array(array.numpy(), copy=True).reshape(self.n_envs, width)
        envs = list(envs)
        if len(values) != len(envs) * width:
            raise ValueError("%s has %d values, expected %d rows of %d" % (name, len(values), len(envs), width))
        # One fancy assignment, not a Python loop over rows (packet M16/H0): the same
        # per-element float64 -> float32 conversion, a repeated env still last-wins.
        rows[envs] = np.asarray(values, dtype=np.float64).reshape(len(envs), width).astype(rows.dtype)
        array.assign(np.ascontiguousarray(rows.reshape(array.shape)))

    def widths(self):
        return {
            "qpos": int(self.mjm.nq),
            "qvel": int(self.mjm.nv),
            "act": int(self.mjm.na),
        }

    def unit_quaternions(self, qpos, rows):
        """A zero free- or ball-joint quaternion becomes MuJoCo's identity `(1, 0, 0, 0)`.

        `mj_normalizeQuat` on mujoco-cpu turns a zero quaternion into wxyz identity; mujoco_warp
        normalises it to `(0, 0, 0, 1)` read as wxyz -- a half turn about z. The env writes zeros
        for every coordinate a reset does not draw (a free joint's orientation included), so
        without this the same reset is two different poses on the two backends (packet M11/X1,
        measured on the SO-101 cube).
        """
        model = self.mjm
        nq = int(model.nq)
        for i in range(model.njnt):
            kind = int(model.jnt_type[i])
            if kind not in (0, 1):  # free, ball
                continue
            adr = int(model.jnt_qposadr[i]) + (3 if kind == 0 else 0)
            for row in range(rows):
                quat = qpos[row * nq + adr : row * nq + adr + 4]
                if len(quat) == 4 and not any(quat):
                    qpos[row * nq + adr : row * nq + adr + 4] = [1.0, 0.0, 0.0, 0.0]
        return qpos

    def write_state(self, state, envs):
        widths = self.widths()
        for field in ("qpos", "qvel", "act"):
            values = state.get(field)
            if field == "qpos" and values:
                values = self.unit_quaternions(list(values), len(envs))
            self.write_field(field, values, envs, widths[field])
        self.forward()

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
            self.forward()

    def set_ctrl(self, ctrl):
        nu = int(self.mjm.nu)
        expected = nu * self.n_envs
        if len(ctrl) != expected:
            raise ValueError("ctrl has %d values, expected %d" % (len(ctrl), expected))
        self.write_field("ctrl", ctrl, range(self.n_envs), nu)

    def step(self, n):
        def launch():
            for _ in range(n):
                mjw.step(self.m, self.d)

        self.replay(("step", n), launch)
        # MuJoCo's own test (`mj_checkPos` / `mj_checkVel` / `mj_checkAcc`): NaN or past
        # mjMAXVAL. mujoco_warp neither checks nor resets, and a blow-up in float32 can stay
        # finite for many steps (qpos ~1e15), so finiteness alone misses it (packet M16/H2b).
        bad = np.zeros(self.n_envs, dtype=bool)
        for field in (self.d.qpos, self.d.qvel, self.d.qacc):
            values = np.asarray(field.numpy(), dtype=np.float64).reshape(self.n_envs, -1)
            bad |= ~(np.abs(values) <= MAXVAL).all(axis=1)
        return [int(env) for env in np.flatnonzero(bad)]


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
    if cmd == "set_params":
        return sim, {"values": sim.set_params([int(e) for e in req["envs"]], req["params"])}
    raise ValueError("unknown command %r" % (cmd,))


def main():
    wp.init()
    sim = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if "frame" in req:
                req["ctrl"] = read_frame(sys.stdin.buffer, int(req["frame"]))
            if req.get("cmd") == "quit":
                return
            sim, payload = handle(sim, req)
            payload["ok"] = True
        except Exception as exc:  # Any failure is a protocol response, never a crash.
            payload = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        answer(_OUT, payload)


def read_frame(stream, n):
    """`mujoco_ref.py`'s `read_frame`, verbatim: the `n` float64 values a request line's
    `frame` announced, read as raw little-endian bytes right after the line (packet M16/H0)."""
    data = stream.read(8 * n)
    if len(data) != 8 * n:
        raise EOFError("frame cut short: %d of %d bytes" % (len(data), 8 * n))
    return np.frombuffer(data, dtype="<f8")


def answer(out, payload):
    """`mujoco_ref.py`'s `answer`, verbatim: a payload with a `frame` (the `state` reply) goes
    as a line naming each array's length, then the arrays as raw little-endian float64 bytes
    (packet M16/H0)."""
    arrays = payload.pop("frame", None)
    if arrays is not None:
        payload["frame"] = [int(a.size) for a in arrays]
    out.write(json.dumps(payload) + "\n")
    out.flush()
    if arrays is not None:
        for a in arrays:
            out.buffer.write(np.ascontiguousarray(a, dtype="<f8").tobytes())
        out.buffer.flush()


if __name__ == "__main__":
    main()
