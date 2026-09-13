# W4 — Newton backend adapter

Spec: spec 4.3 (`NewtonBackend` is the M2 backend — "Kamino / VBD / hydroelastic when needed"),
spec 17.2 (backend semantic mapping and `es backend compare` — "the new core work"), spec 17.3
(determinism tiers: a GPU backend declares tier 2 or 3), spec 14.4 (an unmapped item with
`severity: error` blocks execution), spec 11.6 (capability check), spec 12.1 (the simulation
batch domain), spec 12.4 (measured numbers, never a single figure), spec 1.7 (hallucinated API
names are a standing agent failure mode), spec 18.1 / spec 18.5 (integer ticks, reported
divergence), spec 2.4 (nothing on the runtime path links Python). M2 Wave 4. Review class B.

## context

```
crates/es-physics-backend/src/newton.rs        (new)
crates/es-physics-backend/python/newton_ref.py (new)
crates/es-physics-backend/src/mapping.rs       (the Newton column + two tests whose premise changed)
crates/es-physics-backend/src/lib.rs           (one `pub mod` line + one re-export)
crates/es-physics-backend/src/proc.rs          (shared spawn / call / drop; see W1-mjwarp-live.md)
docs/api-notes/newton.md
docs/packets/M2/W4-newton-backend.md
```

## forbidden

`mjwarp.rs` and `mujoco.rs` semantics, `mjcf_out.rs`, and every crate outside
`es-physics-backend`. **The `es` CLI is out of scope**: see "CLI note" below.

## spec

`NewtonBackend` mirrors `MjWarpBackend` — a Python subprocess running `newton_ref.py`, speaking
the line-delimited JSON protocol of `proc.rs`, loading through Newton's MJCF importer so the one
`scene_to_mjcf` emitter feeds all three backends, stepping with Newton's solver, and returning
env-major state arrays.

- `capabilities()`: determinism tier 2 (`CrossBackend`), `gpu_resident: true`,
  `max_envs = 8192`, `FloatPrecision::F32`.
- The Newton column of `mapping.rs` filled from what was **verified**, not from what the engine
  can do in principle.
- `BackendKind::Newton` already existed and `compare_backends` resolves a column from
  `Capabilities::name`, so naming the backend `"newton"` wires it in with no further change.

## oracle

```
cargo fmt -p es-physics-backend --check
cargo clippy -p es-physics-backend --all-targets -- -D warnings
ES_PYTHON=<venv python> cargo test -p es-physics-backend -- --nocapture | grep -E 'SKIP|RAN|qpos|test result'
cargo xtask check-spec-refs
```

Tests: mapping rows, protocol round-trip on canned JSON (no Python, no GPU), and live tests that
print `RAN` when Newton imports and `SKIP <reason>` otherwise.

## findings

### the package is `newton`, not `newton-physics`

`pip install newton-physics` installs a 1.6 kB stub that raises `ImportError: ... has been
renamed to 'newton'`. The real wheel is `newton==1.6.0`, 6.2 MB, pure Python, requiring only
`warp-lang>=1.17.0` — already present for `MjWarpBackend`. **No blocker: it installed and ran on
Windows + CUDA.**

### `SolverMuJoCo` is unusable in this environment

newton 1.6.0's `[sim]` extra pins `mujoco-warp~=3.12.0`; this workspace needs **3.13.0** for
`MjWarpBackend`. Constructing `SolverMuJoCo` against 3.13.0 dies compiling its kernels
(`WarpCodegenError` in `convert_mjw_contacts_to_newton_kernel`). One environment cannot hold
both pins and `MjWarpBackend` wins — it is the default backend (spec 4.3).

So the adapter steps with **`SolverFeatherstone`**, Newton's own reduced-coordinate
articulated-body solver. This is arguably the better outcome: spec 17.2 exists because the same
Task IR must not behave differently across backends, and a genuinely different solver is what
makes that comparison worth running. Newton's disagreement with `mujoco-cpu` below is physics,
not rounding.

### `add_mjcf` imports no actuators and no sensors

Verified: after importing an MJCF containing `<motor>`, `Model.actuators == []` and
`Model.joint_target_mode == [0]`. No sensor arrays exist at all.

This is the packet's most consequential finding, because the tempting response — run the scene
anyway and report zeros — is exactly the silent-wrong-robot failure the mapping report exists to
prevent. Instead every actuator and sensor feature is `severity: error` in the Newton column, so
an actuated scene is **refused by name at `load`, before a process is spawned** (spec 14.4).
`nu = 0` and `nsensordata = 0` are honest, and `set_ctrl` with a non-empty vector is an error.

The Spec17 row `actuator.pd → controller` stays `approximated`: the *engine* has `ControllerPD`
and `Control.joint_target_q`. The distinction the code now draws is between what the engine can
do and what this adapter was verified to deliver — the capability row is what gates execution.

### contacts are not wired

The adapter steps with `contacts = None`. `newton.CollisionPipeline` exists and is unused.
Blocking the default pyramidal cone would block *every* scene (every MJCF carries one), so it is
declared `approximated` with a blunt quirk — bodies pass through each other — while
`ContactElliptic`, `ContactSoftParams`, `ContactCondim6`, `ContactMesh` and `ContactHeightField`
are blocked outright.

### verified imported by `add_mjcf`

All four MJCF joint kinds, joint limits (MuJoCo's `angle="degree"` default honoured), and
`armature` — so `joint.armature → armature` is **native, verified**, matching spec 17.2.
Joint springs and friction loss were never checked and stay `TODO(api-notes)`.

### coordinates are Newton's, not MuJoCo's

`Model.joint_q_start` addresses `joint_q` per joint, fencepost-terminated. `Model.joint_label` is
path-shaped (`model/worldbody/rod/hinge`); the MJCF name is the last `/` segment. A Newton
transform is `(pos xyz, quat xyzw)` — already spec 3.1 order — but a **free joint's generalized
coordinates are xyzw where MuJoCo writes wxyz**, so `qpos` is not element-wise comparable with
`mujoco-cpu` for a floating base. Declared as a quirk.

## accepted

- `newton_pendulum` **RAN**: `nq = nv = 1`, `n_envs = 2` independent, 100 ticks finite, resets.
- `newton_against_mujoco_cpu` **RAN**.
- `an_actuated_scene_is_refused_rather_than_run_unactuated` passes with or without Newton
  installed — the report is the gate and it runs before anything is spawned.
- `the_declaration_is_the_newton_column_of_the_mapping` asserts the capability declaration and
  the spec 17.2 table agree feature by feature, so they cannot drift.
- 48 tests pass in `es-physics-backend`.

### measured

1 kHz pendulum, rod starting horizontal, 200 ticks, RTX 4060 Laptop GPU, `newton 1.6.0`.

| Measurement | Newton | `mjwarp`, same scene |
|---|---|---|
| `max \|dqpos\|` vs `mujoco-cpu` | **5.88e-4** | 7.41e-8 |
| `max \|dqvel\|` vs `mujoco-cpu` | 5.02e-3 | 7.76e-7 |
| energy proxy delta | 2.06e-2 | 3.16e-6 |
| first tick past the 1e-6 tolerance | **6** | none |

Four orders of magnitude between the two GPU backends, and that is the expected result:
`mjwarp` runs MuJoCo's solver and agrees with MuJoCo, Newton runs Featherstone and does not. The
`< 1e-2` assertion records that the two stay on the same trajectory; it is **not** a validated
agreement tolerance (spec 12.4).

## CLI note — out of scope, needs a follow-up

`BackendKind::Newton` and `compare_backends` need nothing further: the backend names itself
`"newton"` and `BackendKind::from_name` resolves the column. What is **not** done is the `es`
CLI's `es backend compare --task T --backends mjwarp,newton,mujoco-cpu` (spec 17.2), which lives
outside `crates/es-physics-backend`. Whoever owns `docs/packets/M1/CLI-backend-compare.md` needs
to add `"newton" => Box::new(NewtonBackend::new())` to its backend-name switch;
`es_physics_backend::NewtonBackend` is re-exported and ready.

## still open

- Wire `newton.CollisionPipeline` and re-declare the contact features.
- Map MuJoCo actuators onto `Control.joint_f` / `joint_target_q` explicitly, since `add_mjcf`
  will not; only then may `nu` stop being 0.
- Re-try `SolverMuJoCo` when the `mujoco-warp` pin allows 3.13.
- Batch widths beyond 2 worlds; `MAX_ENVS = 8192` is declared, not measured.
- `docs/api-notes/newton.ko.md` and `docs/packets/M2/W4-newton-backend.ko.md` do not exist;
  the repo keeps Korean counterparts for these. Outside this packet's declared scope.
