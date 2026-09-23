# M11 I1 — `PhysXBackend`: Isaac Sim runs as a `PhysicsBackend`

Spec: §28.14 rule 6 (tier `CrossBackend`; scenes through `scene_to_mjcf` and Isaac Sim's MJCF
importer; every importer gap is a mapping-report row) and rules 1–2 and wave 2, §17.1–17.2, §14.4
(a mapping blocked by severity error never runs), §3.5, INV-17 (`PhysicsBackend` is one of the seven;
no new trait). API notes: `docs/api-notes/isaac-sim.md` (W0 + I0's measured fidelity table).
Depends on I0 (the venv and the importer table) and X1 (the dispatch that makes `physx` selectable).
Type B/D.

## the question

`BackendKind::PhysX` exists and prints "not implemented" (`crates/es-physics-backend/src/mapping.rs`).
MJWarp and Newton already run as Python subprocesses behind the `proc.rs` JSON-lines protocol
(`mjwarp.rs`, `newton.rs`, `python/*_ref.py`). **Does a `PhysXBackend` built the same way — the
emitted MJCF imported by Isaac Sim headless, position targets through joint drives, state read back
— implement the whole trait with capabilities and quirks that say exactly what the importer kept, so
that `es backend compare` and X1's `es eval run --backend physx` run on it?**

## spec

* `crates/es-physics-backend/src/physx.rs` + `python/physx_ref.py`: the `proc.rs` protocol
  (`Load`, `Reset`, `SetCtrl`, `Step`, `State`, `SetState`, and `SetParams` if I0 shows the fields
  are writable per env — else no `ModelParams` capability). `ES_ISAAC_PYTHON` names the
  interpreter (the Isaac venv is not `ES_PYTHON`'s); unavailable = the documented SKIPPED exit 3.
* `load`: `scene_to_mjcf` → a temp file → the MJCF importer → an articulation per env (cloned);
  the physics dt = the scene timestep, substeps and solver (TGS/PGS, iterations) from the scene's
  options where they map, else a quirk row; CPU vs GPU pipeline chosen by `LoadConfig` and recorded.
* Actuators: MuJoCo `position` actuators → joint drives with the actuator's `kp` / `kv` as
  stiffness / damping (unit conversion stated), `motor` → effort; anything else refused by name.
  Every other MJCF feature I0's table marks as dropped or changed is a mapping-report row.
* State ordering: joints mapped by name to our `qpos` / `qvel` layout (free joints: position +
  quaternion order converted; document it).
* Capabilities: tier `CrossBackend`, `f32`, `gpu_resident` as measured; `engine_version` =
  Isaac Sim + PhysX versions (X1's hash rule).

## context

```
crates/es-physics-backend/src/physx.rs
crates/es-physics-backend/src/lib.rs
crates/es-physics-backend/src/mapping.rs
crates/es-physics-backend/src/proc.rs
crates/es-physics-backend/python/physx_ref.py
crates/es-physics-backend/tests/**
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/check_deps.rs
crates/es-tools/src/backend.rs
crates/es/tests/cli.rs
docs/api-notes/isaac-sim.md
docs/api-notes/isaac-sim.ko.md
docs/packets/M11/I1-physx-backend.md
docs/packets/M11/I1-physx-backend.ko.md
```

## oracle

1. `cargo test -p es-physics-backend physx_capabilities_are_declared_honestly` and
   `physx_mapping_report_names_every_dropped_feature` (no Isaac needed).
2. Server, `ES_ISAAC_PYTHON=~/venvs/es-isaac/bin/python`, GPU queue lock:
   `es backend compare --scene tests/fixtures/mjcf/so101_pick_place.xml --backends mujoco-cpu,physx
   --ctrl-random --ticks 500` and the same on `mesh_box.xml` — numbers (max |Δqpos|, divergence tick,
   energy proxy) recorded; run twice (run-to-run reproducibility, CPU and GPU pipelines).
3. Server: `es eval run --backend physx` of reach A0 on `evaluation-reach.toml` (X1's dispatch) —
   completes; `success_rate` beside mujoco-cpu and mjwarp.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens; local build without Isaac SKIPs cleanly.

## acceptance

Oracles 1–4; the api-note's backend section (+ko) with the numbers and every quirk.

## forbidden

A new trait; Isaac Sim imported by any Rust crate (subprocess only); tolerances claimed rather than
measured; hiding an importer drop; `es-safety`.
