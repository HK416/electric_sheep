# `mujoco` (Python), pinned to 3.13.0

Surface used by `crates/es-physics-backend/python/mujoco_ref.py`, the reference process behind
`MuJoCoCpuBackend` (spec 17.1). Checked against the wheel `mujoco==3.13.0` on CPython 3.12 /
3.13 (Windows x86-64) by running the crate's own oracle tests. Not a pinned dependency of the
workspace: the package is optional, and `MuJoCoCpuBackend::is_available()` reports its absence
so CI skips the oracle instead of failing (spec 2.4 — nothing on the runtime path needs
Python).

Anything in the 3.x line should work; refresh this file when the version the CI image installs
changes.

## Calls used

| Call | Signature as used | Notes |
|---|---|---|
| `mujoco.MjModel.from_xml_string(xml)` | `(str) -> MjModel` | Raises `ValueError` on a bad model; the message is forwarded verbatim as `PhysicsError::Backend`. |
| `mujoco.MjData(model)` | `(MjModel) -> MjData` | One per env; `n_envs` is emulated by holding a list of them. |
| `mujoco.mj_step(model, data)` | `(MjModel, MjData) -> None` | One physics tick. Called `n` times per `step(n)`. |
| `mujoco.mj_forward(model, data)` | `(MjModel, MjData) -> None` | After a reset or a state write, so that `sensordata` / `xpos` / `xquat` match `qpos`. |
| `mujoco.mj_resetData(model, data)` | `(MjModel, MjData) -> None` | Back to the model's initial state. |
| `mujoco.mj_id2name(model, objtype, id)` | `(MjModel, mjtObj, int) -> str \| None` | `None` for an unnamed element; the emitter always writes names, so a `None` here is a bug and surfaces as `PhysicsError::Protocol`. |
| `mujoco.mjtObj.mjOBJ_JOINT` / `mjOBJ_ACTUATOR` / `mjOBJ_SENSOR` / `mjOBJ_BODY` | enum | Only these four namespaces are looked up. |

## Fields read

`MjData`: `qpos`, `qvel`, `act`, `ctrl`, `sensordata`, `xpos`, `xquat` — all `numpy` arrays of
`float64`. `xpos` is `(nbody, 3)` and `xquat` is `(nbody, 4)`; **`xquat` is wxyz** and is
reordered to the spec 3.1 xyzw before it crosses the wire.

`MjModel`: `nq`, `nv`, `nu`, `nsensordata`, `nbody`, `njnt`, `nsensor`, `jnt_type`,
`jnt_qposadr`, `jnt_dofadr`, `sensor_adr`, `sensor_dim`, `opt.timestep`.

`jnt_type` is `mjtJoint`: `0 = free` (7 qpos / 6 dof), `1 = ball` (4 / 3), `2 = slide` (1 / 1),
`3 = hinge` (1 / 1). The script carries that table itself rather than deriving widths from
address differences.

## Behaviour worth pinning

- **Sensor noise is off by default.** `noise` and `cutoff` on a sensor are written into the
  MJCF but `MuJoCo` applies noise only with the `sensornoise` enable flag, which this adapter
  does not set. Declared as a `BackendQuirk` on the backend's capabilities (spec 17.2).
- **`autolimits` defaults to true** in 3.x, so a joint `range` with no `limited` attribute is
  limited. The emitter relies on this and writes no `limited`.
- **`fullinertia` overwrites the inertial frame:** `MuJoCo` eigendecomposes it and sets the
  frame from the result, so `quat` and `fullinertia` cannot be combined. The emitter writes
  `quat` + `diaginertia` when the inertia is diagonal, `fullinertia` when the frame is
  identity, and refuses the remaining case by name rather than losing the rotation.
- A plane's third `size` component is the rendering grid spacing and must be positive.

## Protocol

`python -c "<embedded script>"`, one JSON object per line each way. Requests carry `cmd`;
replies are `{"ok": true, ...}` or `{"ok": false, "error": "<ExcType>: <message>"}` — a
modelling error is a value, never a dead process. Arrays are JSON lists of numbers, env-major.
`json.dumps` on a Python float and `serde_json` with `float_roundtrip` are both
shortest-round-trip, so `f64` values cross exactly; the bitwise run-to-run test depends on it.

| Request | Reply |
|---|---|
| `{"cmd":"load","mjcf":str,"n_envs":int,"timestep":float\|null,"seed":int}` | `nq, nv, nu, nsensordata, nbody, joints[{name,qpos:[adr,dim],dof:[adr,dim]}], actuators[name], sensors[{name,adr,dim}], bodies[name]` |
| `{"cmd":"reset","envs":[int]\|null,"state":{...}\|null}` | `{}` |
| `{"cmd":"set_ctrl","ctrl":[float]}` | `{}` |
| `{"cmd":"step","n":int}` | `{"nonfinite":[env]}` |
| `{"cmd":"state"}` | `qpos, qvel, act, sensordata, xpos, xquat` |
| `{"cmd":"set_state","state":{"qpos":[],"qvel":[],"act":[]}}` | `{}` |
| `{"cmd":"quit"}` | *(no reply; the process exits)* |

`ES_PYTHON` selects the interpreter; otherwise `python` then `python3` are tried.
