# `newton` (Python) — **partly verified**, pinned to `newton==1.6.0`

Surface used by `crates/es-physics-backend/python/newton_ref.py`, the reference process behind
`NewtonBackend` (spec 4.3: Newton is the M2 backend, spec 17.2: backend semantic mapping).

> **Status: the load / step / state path is verified; contacts and actuation are not.** Every
> call in the table below was run against a real installation and a real device. What was *not*
> verified says so, and `NewtonBackend` declares no capability for it — spec 1.7: an
> invented-but-plausible API name is the cheapest mistake an agent can make.

Verified on: `newton 1.6.0`, `warp-lang 1.17.0`, CPython 3.12 on Windows 11, CUDA Toolkit 12.9 /
driver 13.1, NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB, sm_89). `newton` is not a workspace
dependency: it is optional, and nothing on the runtime path may need Python (spec 2.4).

## Getting the package

The PyPI name is **`newton`**, not `newton-physics`. `pip install newton-physics` installs a
1.6 kB stub whose only behaviour is to raise:

```
ImportError: The 'newton-physics' package has been renamed to 'newton'.
```

`pip install newton` pulls a 6.2 MB pure-Python wheel and needs only `warp-lang>=1.17.0`, which
this workspace already has for `MjWarpBackend`. No CUDA-specific wheel is required — `warp`
supplies the device layer, and the same wheel runs CPU-only.

## Calls used — verified

| Call | Signature as verified | Notes |
|---|---|---|
| `newton.ModelBuilder(up_axis=Axis.Z, gravity=None)` | `-> ModelBuilder` | One builder per env template. |
| `ModelBuilder.add_mjcf(source, *, parse_mujoco_options=True, ...)` | `(str, ...) -> None` | Reads MJCF text, so the same `scene_to_mjcf` emitter feeds all three backends. See the gaps below. |
| `ModelBuilder.add_world(builder)` | `(ModelBuilder, ...) -> None` | Replication. Called `n_envs` times; `Model.world_count` becomes the batch of spec 12.1. |
| `ModelBuilder.finalize()` | `(...) -> Model` | Uploads to the device. |
| `Model.state()` / `Model.control()` | `-> State` / `-> Control` | Two `State`s are kept and swapped, per the solver's in/out signature. |
| `newton.solvers.SolverFeatherstone(model)` | `(Model) -> SolverBase` | The solver actually used. See the `SolverMuJoCo` blocker below. |
| `SolverBase.step(state_in, state_out, control, contacts, dt)` | `(State, State, Control \| None, Contacts \| None, float) -> None` | `contacts=None` here — not wired. |
| `newton.eval_fk(model, joint_q, joint_qd, state)` | `(Model, array, array, State) -> None` | After a reset or a state write, so `body_q` matches `joint_q`. The counterpart of `mujoco.mj_forward`. |
| `State.clear_forces()` | `() -> None` | Before each solver step. |

`warp` prints its banner on stdout; `newton_ref.py` carries the same `sys.stdout` guard as
`mjwarp_ref.py`. See `mujoco-warp.md` for why.

## Fields read and written — verified

| Field | Shape | Notes |
|---|---|---|
| `State.joint_q` | flat, world-major | Generalized coordinates. `Model.joint_coord_count / world_count` per env. |
| `State.joint_qd` | flat, world-major | Generalized velocities. |
| `State.body_q` | `(nbody, 7)` | A Newton transform is **(pos xyz, quat xyzw)** — already the spec 3.1 order, unlike MuJoCo's wxyz, so no swizzle. |
| `Model.joint_q_start` / `joint_qd_start` | `njoint + 1` | Per-joint address into `joint_q` / `joint_qd`, fencepost-terminated: joint `i` occupies `[start[i], start[i+1])`. |
| `Model.joint_label` | `list[str]` | Path-shaped, e.g. `zoo/worldbody/rod/hinge`. The MJCF name is the **last `/` segment**. `Model.body_label` likewise. |
| `Model.joint_type` | `njoint` | `JointType`: `PRISMATIC=0, REVOLUTE=1, BALL=2, FIXED=3, FREE=4, DISTANCE=5, D6=6, ROD=7`. |
| `Model.joint_armature`, `joint_limit_lower/upper`, `joint_damping` | per dof | Carry the MJCF values. |

Arrays are **float32**, widened to f64 at the process boundary, so `NewtonBackend` declares
`FloatPrecision::F32` and determinism tier 2 (spec 17.3: a GPU backend never declares tier 1).

**Verified imported by `add_mjcf`:** all four MJCF joint kinds (`freejoint`, `hinge`, `slide`,
`ball`), joint limits (MuJoCo's `angle="degree"` default is honoured — `range="-1 1"` became
`-1.745e-2` rad), and `armature`.

## Gaps — verified *absent*, and what the adapter does about them

These are the reason `NewtonBackend` refuses scenes rather than running them:

| Gap | Evidence | Adapter behaviour |
|---|---|---|
| `add_mjcf` imports **no `<actuator>`** | `Model.actuators == []`, `Model.joint_target_mode == [0]` after importing an MJCF with `<motor>` | `nu = 0`; every actuator feature is `severity: error`, so an actuated scene is **refused by name** (spec 14.4) rather than run unactuated |
| `add_mjcf` imports **no `<sensor>`** | no sensor arrays on `Model` | `nsensordata = 0`; every sensor feature is blocked |
| **Contacts are not wired** | this adapter steps with `contacts=None`; `newton.CollisionPipeline` exists but is unused | only the MJCF default cone is declared, as `approximated` with a quirk saying bodies pass through each other. `ContactElliptic` / `ContactSoftParams` / `ContactCondim6` / `ContactMesh` / `ContactHeightField` are blocked |
| Joint **spring** and **friction loss** | never checked | `TODO(api-notes)`: unverified, declared unsupported with a warning |

## `SolverMuJoCo` is blocked here — version conflict

`newton.solvers.SolverMuJoCo` would be the natural choice for an MJCF scene, but newton 1.6.0's
`[sim]` extra pins **`mujoco-warp~=3.12.0`** while this workspace needs **3.13.0** for
`MjWarpBackend`. Against 3.13.0, constructing `SolverMuJoCo` fails while compiling its kernels:

```
WarpCodegenError: Error while parsing function "convert_mjw_contacts_to_newton_kernel"
  ... Couldn't find function overload for 'contact_force_fn' ...
```

One environment cannot hold both pins, and `MjWarpBackend` wins the tie: it is the default
backend (spec 4.3) and the one spec 17.1 names. So `NewtonBackend` steps with
**`SolverFeatherstone`**, Newton's own reduced-coordinate articulated-body solver.

That is arguably the better test anyway — spec 17.2 exists because the same Task IR must not
behave differently across backends, and a genuinely different solver is what makes the
comparison worth running. It also means Newton's disagreement with `mujoco-cpu` is physics, not
rounding; the numbers below say so.

Revisit when newton relaxes the pin, or when `mujoco-warp` 3.13 support lands in newton.

## Measured (spec 12.4: numbers, with what produced them)

Hardware above, 1 kHz pendulum with `armature = 0.01`, `damping = 0.1`, rod starting horizontal.

| Measurement | Value | How |
|---|---|---|
| `max \|dqpos\|` vs `mujoco-cpu`, 200 ticks | **5.88e-4** | `newton::tests::newton_against_mujoco_cpu` |
| `max \|dqvel\|` vs `mujoco-cpu`, 200 ticks | 5.02e-3 | same |
| energy proxy delta | 2.06e-2 | same |
| first tick past the 1e-6 tolerance | **tick 6** | same |
| `mjwarp` on the same scene, for scale | 7.41e-8 | `mujoco-warp.md` |

Four orders of magnitude between the two GPU backends is the point: `mjwarp` runs MuJoCo's
solver and agrees with MuJoCo; Newton runs Featherstone and does not. The assertion bound
(`< 1e-2`) records that the two stay on the same trajectory. It is **not** a validated
agreement tolerance.

## Semantics pinned by spec 17.2

| Task IR | Newton mapping | Status in `mapping.rs` |
|---|---|---|
| `actuator.pd(kp, kd)` | a joint controller (`ControllerPD` / `DrivePD`, `Control.joint_target_q` with `Model.joint_target_ke/kd`) | approximated — the *engine* has it; `add_mjcf` does not wire it, so the capability row blocks |
| `contact.friction_cone` | selectable pyramidal or elliptic | native per the spec table: `SolverMuJoCo(cone=...)` takes it. Not reachable through `SolverFeatherstone` |
| `contact.soft_params` | solver-dependent | approximated |
| `joint.armature` | armature | **native, verified**: `Model.joint_armature` carries the MJCF value |
| `sensor.contact_force` | read from the contact buffer | approximated — not wired in this adapter |

The Spec17 rows above state what the *engine* does; the capability rows state what
`NewtonBackend` was verified to *deliver*. Where they differ, the capability row is what gates
execution (spec 14.4). Both live in `crates/es-physics-backend/src/mapping.rs`, and
`newton::tests::the_declaration_is_the_newton_column_of_the_mapping` asserts the declaration and
the table agree feature by feature.

## Next-run checklist

1. Wire `newton.CollisionPipeline` and re-declare the contact features, or keep refusing them.
2. Map MuJoCo actuators onto `Control.joint_f` / `joint_target_q` explicitly, since `add_mjcf`
   will not; only then may `nu` stop being 0.
3. Re-try `SolverMuJoCo` once the `mujoco-warp` pin allows 3.13, and compare the two solvers.
4. Measure a batch wider than 2 worlds. `MAX_ENVS = 8192` is declared, not measured.
