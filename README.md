# Electric Sheep

**A robot learning compiler and runtime.** The task, the observation pipeline, the policy, the
deployment envelope and the evaluation are each written as a typed intermediate representation
(IR). They are compiled to execution plans and GPU kernels, run with the same semantics in
simulation and on a robot, and tracked end to end by a hash chain. That chain is the condition
for bitwise reproducibility. Physics is one backend among several, not the product: MuJoCo (CPU),
MuJoCo Warp and Newton are adapters behind one trait.

The canonical specification is [`docs/ARCHITECTURE.ko.md`](docs/ARCHITECTURE.ko.md) (Korean),
with an English translation in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). It is
prescriptive: it pins type signatures, error codes and invariants, and the code follows it.

## The five IRs

```
Task IR ──► Observation IR ──► Learning IR ──► Deployment IR + Safety Plane ──► actuators
   │  scene, goals, reward,     sensor → tensor     tensor → action chunk   limits, deadlines,
   │  termination, reset,       (ImageSpec, time    (ACT, Diffusion, flow,  fallback; independent
   │  randomization             model, resize with  PPO actors)             of the policy
   │                            intrinsics)
   └──────────────────────────────► Evaluation IR: perturbation suites × metrics × acceptance
```

```
execution_hash = H(task, observation, learning, policy, dataset, deployment,
                   compiler, runtime, hardware_capability)
```

Every artifact the runtime writes carries the hashes it was produced under: datasets, trained
bundles (`.esb`), `training.lock`, `evaluation.lock` and reports. A dataset collected under a
different `task_hash` is refused by name. Two reports are comparable exactly when their
`evaluation_hash` is equal.

## What works today

| Area | Status |
|---|---|
| IRs, validators, canonical hashing | the five IRs, TOML round-trip, cross-IR checks (`es ir check`) |
| Physics | MuJoCo CPU (the reference), MuJoCo Warp and Newton adapters; `es backend compare` runs one scene on two backends from a shared reset (MJWarp vs CPU max \|Δqpos\| 7.4e-8, Newton 5.9e-4) |
| Scenes | MJCF, URDF, glTF and USD readers; STL / OBJ mesh geoms that load, hash by content and simulate (upstream Menagerie SO-101 with its 19 meshes) |
| Rendering | Vulkan rasterizer (`Rs`) and path tracer (`Pt`: NEE, ReSTIR DI, SVGF, still-camera temporal accumulation), each bit-identical to a pure-Rust CPU reference |
| Imitation learning | `es loop collect` (expert or policy, LeRobot v2.1 datasets) → `es train` (ACT through PyTorch) → `es eval run`; an externally trained LeRobot ACT runs bitwise through the runtime |
| Reinforcement learning | PPO through `es_native.Rollout` with the Safety Plane on, bitwise on the CPU backend; `es policy import-rl` imports brax / rsl_rl / rl_games actors (state observations only) |
| Evaluation | perturbation suites (observation / action delay, frame drop, torque noise, backlash, light intensity / direction), `success_rate`, envelope violations, failure histograms, `es eval compare` |
| Safety | the Safety Plane validates every action, and no code path can disable it (widen the envelope instead) |
| Editor | egui run browser, replay, inspector, live telemetry and launch panel (`es-editor`) |
| Real robot | **open**: the hardware interface, HIL and cameras need hardware |

Known limits: `es eval run`, `es loop collect` and the RL rollout run on the `mujoco-cpu` backend
only, and the RL rollout accepts no image input. Milestone reviews in
[`docs/reviews/`](docs/reviews/) list what is open and which decisions the owner still has to
make. The latest is [`M10.md`](docs/reviews/M10.md).

## Quick start

Requirements: Rust ≥ 1.85, a Vulkan 1.3 GPU and the Vulkan SDK (`slangc`) for
the GPU paths. The learning path and the reference oracles need Python with `torch` and
`mujoco`. The core runtime, the policy inference and the Safety Plane do not need Python.

```bash
git config core.hooksPath .githooks          # once per clone: commit-message and language hooks
export ES_PYTHON=/path/to/venv/bin/python     # torch + mujoco; oracles SKIP loudly without it
export ES_SLANGC=/path/to/slangc              # Vulkan SDK's Slang compiler, for GPU shaders

cargo run -p es -- --check-deps               # what this machine can run
cargo xtask ci                                # the full gate (see below)
```

The demo task is SO-101 pick-and-place, with its documents in
`tests/fixtures/visible-learning/`. The loop is:

```bash
es loop collect --policy untrained.esb --scene tests/fixtures/mjcf/so101_pick_place.xml \
    --episodes 200 --seed 1 --expert so101-pick-place --out ds-train --frames frames-train
es train --recipe tests/fixtures/visible-learning/training-u3.toml --out runs/u3
es eval run --config tests/fixtures/visible-learning/evaluation.toml \
    --policy runs/u3/checkpoints/20000.esb --scene tests/fixtures/mjcf/so101_pick_place.xml \
    --out report --jobs 6
es eval compare report-a/report.json report-b/report.json
```

`es loop cycle --recipe <cycle.toml>` chains the same stages with an expert gate. Run
`es --help` for every verb (`ir`, `task`, `loop`, `train`, `policy`, `eval`, `evidence`,
`backend`, `gap`, `bench`, `video`, `dataset`, `import`).

## Repository layout

```
crates/        29 crates in 13 layers, upper depends on lower only (spec 4.2, CI-enforced)
  es-math, es-core                    numerics, deterministic approximations, ids
  es-gpu, es-assets, es-usd           Vulkan, scene importers (MJCF/URDF/glTF/USD, STL/OBJ)
  es-physics-{core,backend}, ...      the PhysicsBackend trait and its adapters
  es-render, es-splat                 rasterizer, path tracer, CPU reference
  es-ir-types, es-ir                  the five IRs, validation, hashing
  es-compile                          observation lowering, policy bundles
  es-policy, es-safety                the policy runtime (torch), the Safety Plane
  es-env                              environments, randomization, rendering in the loop
  es-data, es-import, es-eval, ...    datasets, importers, evaluation, telemetry
  es-py, es-tools, es-ros2, ...       Python bindings, CLI verbs, ROS 2 boundary
  es-editor                           the egui editor
  es                                  the `es` CLI
python/es/     training scripts (train_act.py, train_ppo.py, import_rl.py) and the IR builder
xtask/         `cargo xtask`: every CI gate
tests/         fixtures (IR documents, scenes) and read-only goldens
docs/          spec, design notes, API digests, work packets, milestone reviews (each with .ko.md)
```

## How work is done here

Implementation is done by AI coding agents. People write the spec, design the oracles and
judge the results. See [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md).

- **Oracle first.** A work packet (`docs/packets/<milestone>/`) declares its allowed file
  scope, its spec, a runnable pass/fail oracle, its acceptance criteria and what it must not
  touch. The test comes before the implementation.
- **Reference oracles.** The references are MuJoCo for physics, PyTorch for Learning IR
  lowering, LeRobot for observation and policy equivalence, and CPU-reference golden images for
  the renderer. Goldens are read-only in CI.
- **Invariants.** `es-safety` never depends on `es-policy`. Resize and crop always transform the
  intrinsics. Weights load through `safetensors` only, never pickle. The only traits allowed
  are seven single-implementation extension points. Deterministic mode forbids atomic FP sums,
  fast-math and host transcendentals on hash paths.
- **The gate.** `cargo xtask ci` runs fmt, clippy `-D warnings`, the workspace tests
  (including property tests), the per-crate context budget (≤ 6,000 target / 10,000 cap
  lines), layering, `no_std`, spec cross-references and golden immutability.
- **Conventions.** Source code and commit messages are in English (Conventional Commits, no
  trailers). Every document under `docs/` has a Korean sibling.

## License

Apache-2.0 (workspace `Cargo.toml`).
