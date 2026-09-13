# `mujoco_warp` + `warp` (Python) — **unverified**, pinned to `mujoco-warp==0.1.0`

Surface used by `crates/es-physics-backend/python/mjwarp_ref.py`, the reference process behind
`MjWarpBackend` (spec 17.1: the batched GPU backend, spec 12.1: the simulation batch domain).

> **Status: unverified.** Unlike `docs/api-notes/mujoco.md`, nothing in this file has been run.
> The machine this packet was implemented on has no CUDA device and neither `mujoco_warp` nor
> `warp` installed, so `MjWarpBackend::is_available()` fails and every live test skips with a
> printed reason. `mujoco_warp` is a young package whose API still moves; spec 1.7 names an
> invented-but-plausible API name as a standing agent failure mode, so **every call below is
> marked with how it was derived** and must be re-checked on the first machine with a GPU.
> Until then `MjWarpBackend` is protocol-complete and engine-unproven.

Pinned for the first verification pass: `mujoco-warp==0.1.0`, `warp-lang>=1.7`, `mujoco==3.13.0`
on CPython 3.12, CUDA 12.x. `mujoco_warp` is not a workspace dependency: it is optional, and
nothing on the runtime path may need Python (spec 2.4).

## Calls used — all unverified

| Call | Signature as used | Derived from |
|---|---|---|
| `mujoco_warp.put_model(mjm)` | `(mujoco.MjModel) -> Model` | The package's documented entry point: a CPU `MjModel` is uploaded once. |
| `mujoco_warp.put_data(mjm, mjd, nworld=n)` | `(MjModel, MjData, int) -> Data` | Same; `nworld` is the batch width, the whole reason this backend exists. Other keyword arguments (`nconmax`, `njmax`) exist and are left at their defaults. |
| `mujoco_warp.step(m, d)` | `(Model, Data) -> None` | One physics tick for every world at once. Called `n` times per `step(n)`. |
| `mujoco_warp.forward(m, d)` | `(Model, Data) -> None` | After a reset or a state write, so `sensordata` / `xpos` / `xquat` match `qpos`. The counterpart of `mujoco.mj_forward`. |
| `warp.init()` | `() -> None` | Called once at process start; a missing CUDA device fails here and is reported as a protocol error, not a traceback. |
| `mujoco.MjModel.from_xml_string`, `mujoco.MjData`, `mujoco.mj_forward`, `mujoco.mj_resetData`, `mujoco.mj_id2name` | see `mujoco.md` | Verified — model building and the reset reference stay on the CPU side. |

`get_data_into(mjd, mjm, d)` is deliberately **not** used: reading through a CPU `MjData` would
serialise the batch. State is read from the device arrays directly.

## Fields read and written — all unverified

`Data`: `qpos`, `qvel`, `act`, `ctrl`, `sensordata`, `xpos`, `xquat`. Each is a `warp.array`
whose **first axis is `nworld`**, which is what makes the wire format env-major with no
transposition. Read with `.numpy()`, written with `.assign(ndarray)`.

- The arrays are **float32**. Values are widened to f64 at the process boundary, so
  `MjWarpBackend` declares `FloatPrecision::F32` and determinism tier 2 (spec 3.5): agreement
  with `mujoco-cpu` is a tolerance, never bit-for-bit. Declaring otherwise would be the
  dishonest state spec 17.3 warns about.
- `xquat` is **wxyz**, as on the CPU, and is reordered to the spec 3.1 xyzw before it crosses
  the wire.
- Model metadata (`nq`, `nv`, `nu`, `nsensordata`, `nbody`, `jnt_qposadr`, `sensor_adr`, …) is
  read from the CPU `MjModel` that `put_model` was built from, so the `load` reply has exactly
  the fields `mujoco_ref.py` returns and the two adapters share one Rust decoder.

## Semantics pinned by spec 17.2

| Task IR | MJWarp mapping | Status |
|---|---|---|
| `actuator.pd(kp, kd)` | position actuator gain | native |
| `contact.friction_cone` | pyramidal | native; an **elliptic** scene is refused by name (`severity: error`, spec 14.4) rather than silently re-coned |
| `contact.soft_params` | `solref` / `solimp` impedance | native |
| `joint.armature` | armature | native |
| `sensor.contact_force` | — | blocked: the shared MJCF emitter writes no force / touch sensor |
| `contact.condim = 6` | — | `TODO(api-notes)`: unverified, declared unsupported with a warning |

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
  name shows up as `PhysicsError::Backend` naming the attribute — which is how the first GPU
  run will tell us which rows of this file were wrong.

## First-GPU-run checklist

1. `python -c "import mujoco_warp, warp"` — then `MjWarpBackend::is_available()` returns `Ok`.
2. `cargo test -p es-physics-backend mjwarp_pendulum -- --nocapture` — stops printing `SKIP`.
3. `cargo test -p es-physics-backend mjwarp_against_mujoco_cpu` — the spec 3.5 tier 3
   comparison against the CPU oracle. Its `max |dqpos| < 1e-2` bound is a smoke bound, not a
   validated tolerance; replace it with a measured one and record the 9-metric set (spec 12.4).
4. Re-check every row above and delete the **unverified** banner only for what actually ran.
