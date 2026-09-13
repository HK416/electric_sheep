# W1 — MJWarp backend adapter + backend semantic-mapping report

Spec: spec 17.2 (backend semantic mapping and `es backend compare` — "the new core work"),
spec 17.1 / spec 17.3 (backend layer, determinism tiers: GPU backends declare tier 2 or 3, only
`mujoco-cpu` may claim tier 1), spec 4.3 (backends are the default path), spec 14.4 (an unmapped
item with `severity: error` blocks execution), spec 11.6 (capability check at compile time),
spec 12.1 (MJWarp is the batched GPU path), spec 3.5 (determinism tiers), spec 1.7 (hallucinated
API names are a standing agent failure mode), spec 18.1 / spec 18.5 (integer ticks, reported
divergence). M1 Wave 1. Review class B.

## context

```
crates/es-physics-backend/src/mapping.rs
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/src/lib.rs          (two `pub mod` lines + re-exports)
crates/es-physics-backend/python/mjwarp_ref.py
docs/api-notes/mujoco-warp.md
docs/packets/M1/W1-mjwarp-adapter.md
```

## spec

### `mapping.rs` — the spec 17.2 table

- `BackendKind { MuJoCoCpu, MjWarp, Newton, PhysX }` with the `--backends` spelling, and
  `from_name` so a `Capabilities::name` maps back to a column.
- `TaskFeature = Spec17(Spec17Row) | Capability(Feature)`: the five rows the spec 17.2 table
  spells out (`actuator.pd`, `contact.friction_cone`, `contact.soft_params`, `joint.armature`,
  `sensor.contact_force`) plus every `Feature` variant in `es-physics-core`. The two kinds are
  separate because `actuator.pd` is a pair of gains, not an actuator kind, and
  `sensor.contact_force` is one question asked of two sensor kinds.
- `lookup(TaskFeature, BackendKind) -> Mapping { status, severity }`, total by construction;
  `SemanticMapping` materialises the cross product for iteration.
  `Status = Native(note) | Approximated(note) | Unsupported(note)`,
  `Severity = Info | Warning | Error`.
- Columns: `mujoco-cpu` is derived from `crate::mujoco::capabilities()` and `mjwarp` from the
  same set minus what spec 17.2 pins narrower, so a declaration and its mapping cannot drift.
  `newton` and `physx` carry exactly the spec 17.2 cells; every other row is
  `Unsupported("TODO(api-notes): unverified against the engine")` with `Warning` — **a mapping
  is never guessed native** (spec 1.7).
- `mapping_report(&SceneDesc, BackendKind) -> MappingReport { backend, rows, blocked }` scans
  the scene — actuator kinds, sensor kinds, `armature != 0`, friction cone, non-default
  `solref` / `solimp` — and lists each used feature's mapping. Feature rows come from
  `Requirements::from_scene`, so the report and the spec 11.6 capability check see the same
  scene. `blocked` is any row that is both `Unsupported` and `severity: error` (spec 14.4).
  `Display` is the fixed-width table `es backend compare` prints in its header.
- `compare_backends(a, b, &SceneDesc, ctrl_seq, n_ticks) -> Result<CompareReport, PhysicsError>`
  runs both backends from the same reset state with the same (cycled) control sequence and
  scores spec 3.5 tier 3 metrics: `max |dqpos|`, `max |dqvel|`, an energy drift **proxy**
  (`sum ½ qvel²` — kinetic only, no mass matrix, no potential term; documented as a proxy and
  comparable across backends because it needs nothing but `qvel`), the first tick where
  `|dqpos|` exceeds `DIVERGENCE_TOL = 1e-6`, both declared determinism tiers, and both mapping
  reports. `Display` is a table. The `es backend compare` CLI wiring is another packet.

### `mjwarp.rs` — `MjWarpBackend`

- MuJoCo Warp behind `PhysicsBackend` through `python/mjwarp_ref.py`, embedded with
  `include_str!` and run as `python -c`, speaking the same line-delimited JSON as
  `mujoco_ref.py` (`load` / `reset` / `set_ctrl` / `step` / `state` / `set_state` / `quit`), so
  the Rust request and reply types in `proc.rs` are reused unchanged. `load` carries
  `n_envs`, which the script passes as `put_data(..., nworld=n_envs)` — a real batch on one
  device (spec 12.1), not the CPU adapter's emulation loop.
- `is_available()` runs `python -c "import mujoco_warp, warp"`; `ES_PYTHON` overrides the
  interpreter.
- Declared capabilities: **determinism tier 2 (`CrossBackend`), never tier 1** (spec 17.3),
  `gpu_resident: true`, `max_envs: 8192` (a declared bound, not a measurement — spec 12.4),
  `float: F32` (the device arrays are float32; agreement with the CPU oracle is a tolerance),
  `supports_reset_subset` and `supports_state_get_set`. The feature sets *are* the MJWarp
  column of the mapping table, and a test asserts that feature by feature.
- `load()` runs `mapping_report` **before anything is spawned** and refuses a blocked scene
  with `PhysicsError::Unsupported(report.to_string())` (spec 14.4): the error message is the
  table, naming the rows. Features are the report's business, so the capability check that
  follows covers only the run-level shape of the request (batch size, spec 11.6).
- `docs/api-notes/mujoco-warp.md` pins the version and marks **every `mujoco_warp` call as
  unverified**, with a first-GPU-run checklist.

### Deviations from the packet brief, and why

- `Status::Native` and `Status::Unsupported` carry a note as well as `Approximated`. An
  unexplained `Unsupported` cell is exactly what spec 17.2 exists to prevent, and the brief's
  own `TODO(api-notes)` requirement needs somewhere to live.
- `compare_backends` returns `Result<CompareReport, _>` rather than `CompareReport`: a load or
  step failure is not a comparison result and must not be swallowed into a report with a hole
  in it.
- `MappingReport` carries its `BackendKind`, so the rendered table can name the backend.
- `proc.rs::Process` hardcodes `mujoco_ref.py` and this packet may not edit it, so the
  spawn / call / drop trio is duplicated in `mjwarp.rs` behind a `ponytail:` comment. Fold both
  into one `Process::spawn_with(SCRIPT)` when a third out-of-process backend appears.
- No `Capabilities` or `Feature` addition was needed in `es-physics-core`.

## oracle

```
cargo fmt -p es-physics-backend --check
cargo clippy -p es-physics-backend --all-targets -- -D warnings
cargo test -p es-physics-backend
cargo xtask layering && cargo xtask check-spec-refs && cargo xtask context-budget
```

Everything except the two live tests runs with no GPU and no Python. `mjwarp_pendulum` and
`mjwarp_against_mujoco_cpu` print `SKIP <name>: <reason>` and pass when `is_available()` fails.

## acceptance

- Every `(TaskFeature, BackendKind)` pair has a row, every row has a non-empty note, and the
  five spec 17.2 rows assert their exact statuses cell by cell — including PhysX's armature,
  which the spec marks unsupported with a *warning*, not a block.
- An unchecked row is `Unsupported` + `Warning` + `TODO(api-notes)` and does not block.
- `mapping_report` on `tests/fixtures/mjcf/pendulum.xml` blocks on MJWarp (the fixture asks for
  an elliptic cone) and does not block on `mujoco-cpu`; on `actuated.xml` it reports
  `actuator.pd`, `sensor.contact_force` and `Tendon`, blocks on MJWarp, and only warns on
  Newton. A plain hinge scene blocks on no backend.
- `MjWarpBackend::load` on an elliptic scene fails with `PhysicsError::Unsupported` whose
  message contains the rendered table, `ContactElliptic` and `blocked: yes`; a batch above
  `MAX_ENVS` fails with `Unsupported::BatchSize`; calls before `load` are `NotLoaded`.
- `compare_backends` on two identical fake integrators reports no divergence and zero deltas;
  perturbing one by `1e-9` finds the divergence tick (tick 44 for that integrator) and a
  non-zero energy-proxy delta.
- The `mjwarp_ref.py` protocol round-trips against canned JSON — `load`, batched `state`,
  `step` with a non-finite world, `Ack`, a Python-side error, a malformed line — with no Python
  process involved.

## forbidden

Any file outside `context`. `crates/es`, `crates/es-policy`, `crates/es-sensor`,
`crates/es-actuator` (other packets in flight). `mujoco.rs`, `proc.rs`, `mjcf_out.rs` — reused,
never edited. The root `Cargo.toml`. Any new trait (`INV-17`: the seven extension points are
fixed). `HashMap` / `HashSet` (spec 3.4). Declaring determinism tier 1 from a GPU backend
(spec 17.3). Guessing a native mapping for an engine nobody has run (spec 1.7). Implementing
physics: this crate maps onto an engine, it is not one (spec 17.4 is M4+).
