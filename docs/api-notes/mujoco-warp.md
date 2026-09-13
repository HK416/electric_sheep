# `mujoco_warp` + `warp` (Python) — **verified**, pinned to `mujoco-warp==3.13.0`

Surface used by `crates/es-physics-backend/python/mjwarp_ref.py`, the reference process behind
`MjWarpBackend` (spec 17.1: the batched GPU backend, spec 12.1: the simulation batch domain).

> **Status: verified 2026-09-13.** Every call in the "Calls used" table below was run against a
> real installation and a real device; the **unverified** banner this file used to carry is
> gone, and what is still unchecked says so row by row. The earlier pin of `mujoco-warp==0.1.0`
> was wrong: the package versions in lockstep with `mujoco` itself.

Verified on: `mujoco-warp 3.13.0`, `warp-lang 1.17.0`, `mujoco 3.13.0`, CPython 3.12 on
Windows 11, CUDA Toolkit 12.9 / driver 13.1, NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB, sm_89).
`mujoco_warp` is not a workspace dependency: it is optional, and nothing on the runtime path may
need Python (spec 2.4).

## Calls used — verified unless the row says otherwise

| Call | Signature as verified | Notes |
|---|---|---|
| `mujoco_warp.put_model(mjm, batch_sizes=None)` | `(mujoco.MjModel, dict[str, int] \| None) -> Model` | A CPU `MjModel` is uploaded once. `batch_sizes` left at its default. |
| `mujoco_warp.put_data(mjm, mjd, nworld=1, ...)` | `(MjModel, MjData, int, ...) -> Data` | `nworld` is the batch width, the whole reason this backend exists. `nconmax` / `njmax` / `nccdmax` / `naconmax` / `nvmax` exist and are left at their defaults. |
| `mujoco_warp.step(m, d)` | `(Model, Data) -> None` | One physics tick for every world at once. Called `n` times per `step(n)`. |
| `mujoco_warp.forward(m, d)` | `(Model, Data) -> None` | After a reset or a state write, so `sensordata` / `xpos` / `xquat` match `qpos`. The counterpart of `mujoco.mj_forward`. |
| `warp.init()` | `() -> None` | Called once at process start. See the stdout note below — this is the call that made the protocol fail. |
| `mujoco.MjModel.from_xml_string`, `mujoco.MjData`, `mujoco.mj_forward`, `mujoco.mj_resetData`, `mujoco.mj_id2name` | see `mujoco.md` | Model building and the reset reference stay on the CPU side. |

`get_data_into(result, mjm, d, world_id=0)` exists with that signature and is deliberately
**not** used: it reads one world at a time through a CPU `MjData` and would serialise the batch.
State is read from the device arrays directly.

## `warp` writes to stdout — the one real trap

`warp` prints an initialization banner on **stdout**:

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.1
   Devices: ...
```

and it keeps printing there (module compile/load lines, deprecation warnings). Stdout is the
JSON-lines protocol, so this corrupted the very first reply and every live test failed with
`Protocol("expected value at line 1 column 1 in `Warp 1.17.0 initialized:`")`. `mjwarp_ref.py`
therefore captures the real handle first and points `sys.stdout` at `sys.stderr`, which the
Rust side discards:

```python
_OUT = sys.stdout
sys.stdout = sys.stderr
```

Any future out-of-process backend that imports `warp` needs the same two lines;
`newton_ref.py` has them.

## Fields read and written — verified

`Data`: `qpos`, `qvel`, `act`, `ctrl`, `sensordata`, `xpos`, `xquat`. Each is a `warp.array`
whose **first axis is `nworld`**, which is what makes the wire format env-major with no
transposition. Read with `.numpy()`, written with `.assign(ndarray)`. Both were exercised: a
two-world batch set to `qpos = [0.3, -0.2]` stepped to two different, independent answers.

- The arrays are **float32**. Values are widened to f64 at the process boundary, so
  `MjWarpBackend` declares `FloatPrecision::F32` and determinism tier 2 (spec 3.5): agreement
  with `mujoco-cpu` is a tolerance, never a guarantee of bit equality. Declaring otherwise would
  be the dishonest state spec 17.3 warns about — see the measurement below.
- `xquat` is **wxyz**, as on the CPU, and is reordered to the spec 3.1 xyzw before it crosses
  the wire.
- Model metadata (`nq`, `nv`, `nu`, `nsensordata`, `nbody`, `jnt_qposadr`, `sensor_adr`, …) is
  read from the CPU `MjModel` that `put_model` was built from, so the `load` reply has exactly
  the fields `mujoco_ref.py` returns and the three adapters share one Rust decoder.

## Measured (spec 12.4: numbers, with what produced them)

Hardware above, 1 kHz pendulum with `armature = 0.01`, `damping = 0.1`.

| Measurement | Value | How |
|---|---|---|
| Run-to-run reproducibility, 200 ticks, 2 worlds | **bit-identical**, `max \|delta\| = 0` | `mjwarp::tests::mjwarp_runs_agree_to_the_declared_tier`, two fresh processes from the same written state |
| `max \|dqpos\|` vs `mujoco-cpu`, 200 ticks | **7.41e-8** | `mjwarp::tests::mjwarp_against_mujoco_cpu` on a rod that starts horizontal |
| `max \|dqvel\|` vs `mujoco-cpu`, 200 ticks | 7.76e-7 | same |
| energy proxy delta | 3.16e-6 | same |

**The declared tier stays 2.** One scene reproducing bit for bit on one GPU and one driver is an
observation, not a guarantee: spec 17.3 says only `mujoco-cpu` declares tier 1, and the test
asserts the tier-2 contract (`delta < 1e-9`) while merely *printing* whether the run was
bitwise. A GPU backend that happens to be reproducible today must not become a test that fails
on the next driver.

The comparison scene matters: `compare_backends` resets and steps without control, so a
pendulum hanging straight down sits at its equilibrium and scores a vacuous `max |dqpos| = 0`.
The compare tests use a rod that starts horizontal for this reason.

## Semantics pinned by spec 17.2

| Task IR | MJWarp mapping | Status |
|---|---|---|
| `actuator.pd(kp, kd)` | position actuator gain | native |
| `contact.friction_cone` | pyramidal | native; an **elliptic** scene is refused by name (`severity: error`, spec 14.4) rather than silently re-coned |
| `contact.soft_params` | `solref` / `solimp` impedance | native |
| `joint.armature` | armature | native |
| `sensor.contact_force` | — | blocked: the shared MJCF emitter writes no force / touch sensor |
| `contact.condim = 6` | — | `TODO(api-notes)`: still unverified, declared unsupported with a warning |

The table lives in code as `crates/es-physics-backend/src/mapping.rs`, and
`MjWarpBackend::capabilities()` is *derived* from it, so the declaration and the report cannot
drift apart.

## Behaviour worth pinning

- **Reset means `mj_resetData`.** The process keeps one CPU `MjData` as the reset reference and
  copies its `qpos` / `qvel` / `act` into the selected worlds, so "initial state" means the same
  thing on both backends and a cross-backend comparison starts from the same place.
- **A subset reset touches only its rows**, and only a whole-batch reset rewinds the tick.
- **Divergence is reported, not raised**: non-finite worlds come back in `step`'s `nonfinite`
  list and become `StepReport::failures` (spec 18.5).
- A Python-side failure of any kind is one `{"ok": false, "error": ...}` line, so a wrong API
  name shows up as `PhysicsError::Backend` naming the attribute.
- **First call is slow.** `put_model` / the first `step` compile and cache warp kernels; the
  cache lives under `%LOCALAPPDATA%\NVIDIA\warp\Cache\<version>`. A cold cache costs seconds,
  a warm one milliseconds.

## Still unverified

- `contact.condim = 6`.
- Batch widths beyond 2 worlds. `MAX_ENVS = 8192` is a *declared* bound, not a measurement
  (spec 12.4); the real ceiling is device memory.
- Contact-rich scenes. Everything measured above is a single hinge with no collisions.
- `nconmax` / `njmax` sizing, which a contact-rich scene will need.
