# M11 I0 — Isaac Sim and Isaac Lab on the oracle server, headless, measured

Spec: §28.14 rule 6 (Isaac Sim is installed only after the owner accepts NVIDIA's EULA; tier
`CrossBackend`; scenes through `scene_to_mjcf` and its MJCF importer, every gap a mapping-report row)
and wave 1, §2 (Python is a first-class dependency on the learning path only; the core runtime never
needs Isaac), §17.1. API digests: `docs/api-notes/isaac-sim.md`, `isaac-lab.md` (M11 W0). Type D.

**Precondition:** the owner's explicit acceptance of the NVIDIA Omniverse / Isaac Sim EULA, recorded
in this packet's report with the date. Without it the packet does not start.

## the question

**Does a headless Isaac Sim, installed from pip into its own venv on the RTX 4090 server, import
our SO-101 scene as MJCF (the text `scene_to_mjcf` emits), step it with position targets, and report
joint states — and what does its MJCF importer keep, change or drop?**

## spec

* A uv venv `~/venvs/es-isaac` with the Python version the api-note pins, `isaacsim` (the pinned
  version and extras) and Isaac Lab (the matching release), plus `rsl-rl-lib`. No system packages;
  no sudo. Install log and `pip freeze` under `~/artifacts/plan-x/i0/`.
* `python/physx_smoke.py` (repo, not yet a backend): starts `SimulationApp({"headless": True})`,
  imports an MJCF file through the MJCF importer, creates the articulation, steps 100 physics steps
  at the scene's timestep with a fixed position-target sequence, prints versions and the joint
  trajectory as JSON, exits cleanly. Run on `tests/fixtures/mjcf/so101_pick_place.xml` (via
  `es`-emitted MJCF — dump it with a small `es backend` or test helper) and on `mesh_box.xml`.
* An importer fidelity table in `docs/api-notes/isaac-sim.md` (+ko), filled from the imported USD
  stage: joints and limits, actuators → drive type / stiffness / damping, masses and inertias vs
  MuJoCo's, collision shapes (mesh → convex?), friction, contype/conaffinity, sensors, what was
  dropped with warnings.
* A same-control comparison against MuJoCo CPU (the 100-step joint trajectory, max |Δq|) — numbers
  only; no tolerance claimed.

## context

```
python/physx_smoke.py
docs/api-notes/isaac-sim.md
docs/api-notes/isaac-sim.ko.md
docs/api-notes/isaac-lab.md
docs/api-notes/isaac-lab.ko.md
docs/packets/M11/I0-isaac-sim-install.md
docs/packets/M11/I0-isaac-sim-install.ko.md
```

## oracle

1. On the server: `~/venvs/es-isaac/bin/python python/physx_smoke.py --mjcf <so101.xml>` exits 0 and
   prints the versions and a 100-row trajectory; the same for `mesh_box`.
2. The fidelity table and the Δq numbers are in the api-note with the date and versions.
3. Nothing in `crates/` changes; `cargo xtask check-scope docs/packets/M11/I0-isaac-sim-install.md`.

## acceptance

Oracles 1–3; disk use and install time recorded; the GPU queue lock held during runs.

## forbidden

Installing without the recorded EULA acceptance; sudo; touching other venvs; any Rust change
(I1 owns the backend).
