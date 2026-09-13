"""Newton reference process for `NewtonBackend` (spec 4.3, spec 17.2).

Same line-delimited JSON protocol as `mujoco_ref.py` and `mjwarp_ref.py`, so the three
backends are interchangeable from Rust. See `mjwarp_ref.py` for the command list.

What is different about Newton, all of it verified against newton 1.6.0 and written up in
docs/api-notes/newton.md:

* The batch is `world_count`: one `ModelBuilder` per env is replicated with `add_world`, so
  `joint_q` comes out world-major already -- the env-major layout spec 12.1 asks for.
* Coordinates are Newton's, not MuJoCo's. `joint_q_start` addresses `joint_q` per joint, and a
  free joint is (pos xyz, quat xyzw) where MuJoCo writes (pos xyz, quat wxyz).
* `add_mjcf` imports no `<actuator>` and no `<sensor>` (`Model.actuators` comes back empty), so
  `nu` and `nsensordata` are 0 here and the Rust side refuses such a scene by name rather than
  running it unactuated.
* The solver is `SolverFeatherstone`, not `SolverMuJoCo`: newton 1.6.0 pins
  `mujoco-warp~=3.12.0` and this environment needs 3.13.0 for `MjWarpBackend`, so importing
  `SolverMuJoCo` fails to compile its kernels. Featherstone is Newton's own reduced-coordinate
  articulated-body solver, which is what makes the cross-backend comparison worth running.
"""

import json
import sys

# `warp` prints an initialization banner on stdout, and stdout is the protocol. The real handle
# is taken first and everything else -- imports, kernel chatter -- is pointed at stderr.
_OUT = sys.stdout
sys.stdout = sys.stderr

try:
    import newton
    import numpy as np
except ImportError as exc:  # Reported as a protocol response, not a traceback on stderr.
    _OUT.write(json.dumps({"ok": False, "error": "import failed: %s" % exc}) + "\n")
    _OUT.flush()
    raise SystemExit(1)


def leaf(label):
    """The MJCF name inside a Newton label path (`model/worldbody/rod/hinge` -> `hinge`)."""
    return label.rsplit("/", 1)[-1]


def flat(arr):
    """A warp array as a flat list of float64, world-major (its first axis is the batch)."""
    return np.asarray(arr.numpy(), dtype=np.float64).reshape(-1).tolist()


class Sim(object):
    def __init__(self, mjcf, n_envs, timestep, seed):
        one = newton.ModelBuilder()
        one.add_mjcf(mjcf)
        batch = newton.ModelBuilder()
        for _ in range(n_envs):
            batch.add_world(one)
        self.model = batch.finalize()
        self.n_envs = n_envs
        self.seed = seed
        # `add_mjcf(parse_mujoco_options=True)` already took the MJCF timestep; an explicit one
        # from the Rust side wins, because spec 18.1 makes the tick rate the model.
        self.dt = timestep if timestep is not None else 1.0 / 60.0
        self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.control = self.model.control()
        self.state = self.model.state()
        self.scratch = self.model.state()
        # The finalized model carries the scene's initial coordinates; `reset` means these,
        # so a reset here is the same event as `mj_resetData` on the other two backends.
        self.q0 = np.array(self.model.joint_q.numpy(), copy=True)
        self.qd0 = np.zeros_like(np.array(self.model.joint_qd.numpy(), copy=True))
        self.nq = self.model.joint_coord_count // n_envs
        self.nv = self.model.joint_dof_count // n_envs
        self.nbody = self.model.body_count // n_envs
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def info(self):
        model = self.model
        starts = model.joint_q_start.numpy()
        dof_starts = model.joint_qd_start.numpy()
        labels = list(model.joint_label)
        joints = []
        # One world's worth: the replicas repeat the same names at the same offsets.
        for i in range(model.joint_count // self.n_envs):
            joints.append(
                {
                    "name": leaf(labels[i]),
                    "qpos": [int(starts[i]), int(starts[i + 1] - starts[i])],
                    "dof": [int(dof_starts[i]), int(dof_starts[i + 1] - dof_starts[i])],
                }
            )
        return {
            "nq": int(self.nq),
            "nv": int(self.nv),
            # `add_mjcf` imports neither actuators nor sensors (newton 1.6.0).
            "nu": 0,
            "nsensordata": 0,
            "nbody": int(self.nbody),
            "joints": joints,
            "actuators": [],
            "sensors": [],
            "bodies": [leaf(b) for b in list(model.body_label)[: self.nbody]],
        }

    def state_payload(self):
        # A Newton transform is (pos xyz, quat xyzw); spec 3.1 is xyzw, so no swizzle.
        body_q = np.asarray(self.state.body_q.numpy(), dtype=np.float64).reshape(-1, 7)
        return {
            "qpos": flat(self.state.joint_q),
            "qvel": flat(self.state.joint_qd),
            "act": [],
            "sensordata": [],
            "xpos": body_q[:, :3].reshape(-1).tolist(),
            "xquat": body_q[:, 3:].reshape(-1).tolist(),
        }

    def write_field(self, name, values, envs, width):
        """Overwrites rows `envs` of one state field from a flat env-major list."""
        if not values or width == 0:
            return
        array = getattr(self.state, name)
        rows = np.array(array.numpy(), copy=True).reshape(self.n_envs, width)
        for row, env in enumerate(envs):
            chunk = values[row * width : (row + 1) * width]
            if len(chunk) != width:
                raise ValueError(
                    "%s row %d has %d values, expected %d" % (name, row, len(chunk), width)
                )
            rows[env] = np.asarray(chunk, dtype=rows.dtype)
        array.assign(np.ascontiguousarray(rows.reshape(array.shape)))

    def write_state(self, state, envs):
        self.write_field("joint_q", state.get("qpos"), envs, self.nq)
        self.write_field("joint_qd", state.get("qvel"), envs, self.nv)
        if state.get("act"):
            raise ValueError("newton has no actuator activation state; `act` must be empty")
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def reset(self, envs, state):
        self.write_field("joint_q", self.q0.reshape(self.n_envs, self.nq)[list(envs)]
                         .reshape(-1).tolist(), envs, self.nq)
        self.write_field("joint_qd", self.qd0.reshape(self.n_envs, self.nv)[list(envs)]
                         .reshape(-1).tolist(), envs, self.nv)
        self.control.clear()
        if state is not None:
            self.write_state(state, envs)
        else:
            newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def set_ctrl(self, ctrl):
        if ctrl:
            raise ValueError(
                "newton's MJCF importer maps no actuators, so nu is 0 and ctrl must be empty"
            )

    def step(self, n):
        for _ in range(n):
            self.state.clear_forces()
            # Contacts are not wired in this adapter: `NewtonBackend` declares no contact
            # capability, so a scene that needs one is refused before it gets here.
            self.solver.step(self.state, self.scratch, self.control, None, self.dt)
            self.state, self.scratch = self.scratch, self.state
        qpos = np.asarray(self.state.joint_q.numpy(), dtype=np.float64).reshape(self.n_envs, -1)
        qvel = np.asarray(self.state.joint_qd.numpy(), dtype=np.float64).reshape(self.n_envs, -1)
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
        return sim, sim.state_payload()
    if cmd == "set_state":
        sim.write_state(req["state"], range(sim.n_envs))
        return sim, {}
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
        _OUT.write(json.dumps(payload) + "\n")
        _OUT.flush()


if __name__ == "__main__":
    main()
