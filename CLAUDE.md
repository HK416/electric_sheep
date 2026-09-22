# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

`docs/ARCHITECTURE.ko.md` (the v1.0 technical spec, ~3000 lines, in Korean) is the canonical
source of truth; `docs/ARCHITECTURE.md` is its English translation. M0–M5 and M7 are implemented and reviewed (`docs/reviews/M0.md` .. `M5.md`, `M7.md`; review
follow-ups are `docs/packets/<M>/P-<M>-R<n>.md` or the review's R-list). M5 (plan V, `docs/design/visible-learning.md`) proved the thesis: an
externally trained LeRobot ACT runs bitwise through the runtime and passes the demo's Evaluation IR. **M7 plan U** (spec §28.10, packets in
`docs/packets/M7/`) closed 2026-09-21: `es train` / `es loop cycle`, the editor's run browser/replay/inspector/live telemetry/launch panel/plain-language UI,
batched lowering + lr schedule + pretrained backbone + augmentation, and the RS/PT render quality ladder; on the committed documents the IR-graph policy
U3 passes the demo's acceptance under the declared latency (held-out 0.5625); the addendum in `docs/reviews/M7.md` records the three
follow-ups E7 (collect/train/cycle publish telemetry, the editor draws the loss curve), E8 (replay depth buffer) and R5 (a Task IR
sensor declares `render = { path = "pt", … }`; absent = default = today's hash). **M8 plan S** (spec §28.11, packets in `docs/packets/M8/`, `docs/design/rl-continuation.md`) closed 2026-09-22: the episode-boundary
decision (yes/yes) shipped the `(cell, episode)` partition (2× evaluation, nothing moved); `es policy import-rl` brings a brax/rsl_rl/rl_games
PPO actor into a bundle (bitwise / 9.7e-7); `[init] policy` + `init.lock`; `[rl]` PPO through `es_native.Rollout` with the Safety Plane on,
bitwise on the CPU backend; the reach task trains from scratch to 0.42 (three seeds) but the imported brax policy scores 0.00 here and
continuation cannot move it (saturated tanh, zero gradient). **M9 plan T** (spec §28.12, `docs/packets/M9/`) closed 2026-09-22: `es-ir` split under the line target; `ActionSpace::JointDelta` with one
`es-env` integrator and the plane untouched; a brax increment policy imported bitwise; measured, the increment space clamps as much as the
absolute one (on the position envelope instead of the rate bounds) and learns less (0.10 vs 0.42 at 4,000 iterations), so §13.4's default stays
`JointPosition`. The next campaign waits on the human decisions in `docs/reviews/M9.md` (the envelope's meaning for a learning policy, the
buffered-path watchdog for delta policies) and `docs/reviews/M8.md` (the worker thread pool in `execution_hash`, `scene_hash` across
platforms, where the next source policy is trained), and the older ones in `docs/reviews/M7.md` (the stop rule's reading, the SSIM threshold). M6 (quadruped, `docs/design/quadruped-track.md`) is parked pending the owner's decision. GPU paths (es-gpu, es-render, Observation
IR GPU lowering, MJWarp/Newton adapters) were verified on an RTX 4060 with the Vulkan SDK;
the Python oracles (MuJoCo, PyTorch, LeRobot ACT checkpoint, diffusers) run from the project
venv `.venv` (set `ES_PYTHON` to its interpreter). Still open: M3 W1 real-robot
interface/HIL/cameras (hardware), the native physics solver prototype (cut by spec §1.9 #1),
and the human decisions listed in the reviews. Work packets live in `docs/packets/<milestone>/`, design notes
in `docs/design/`, pinned external API digests in `docs/api-notes/`, milestone reviews in
`docs/reviews/`. Every document under `docs/` has a Korean sibling `<name>.ko.md`; keep
the pairing when adding or changing docs. Before adding anything, read the relevant
`docs/ARCHITECTURE.ko.md` section (§ numbers below) — it is prescriptive, not aspirational,
and pins exact type signatures, error codes, and invariants. Gate every change with
`cargo xtask ci` (fmt, clippy `-D warnings`, tests incl. `es-ir/testing` property tests,
context-budget, layering, spec-refs, goldens). Reference oracles that need Python packages
(MuJoCo, MJWarp, PyTorch) skip with a printed reason when the package is missing; set
`ES_PYTHON` to an interpreter that has them to run them for real.

## What this is

Electric Sheep (formerly Keystone) is a **Robot Learning Compiler & Runtime**: task,
observation, learning, and deployment are described as typed intermediate representations,
compiled to GPU kernels and execution plans, run with identical semantics in simulation and
on real robots, and tracked end-to-end via a hash chain. Physics simulation is one backend,
not the product — MuJoCo Warp / Newton / MJX are adopted as default backends (§4.3); a native
solver is optional and the first thing cut under scope pressure (§1.9).

## Development model (§1) — read before doing anything

Implementation is done entirely by AI coding agents; humans write specs, design oracles, and
judge. This shapes how you must work here:

- **Oracle-first, no exceptions (§1.4):** the verification harness comes before the
  implementation. Work whose pass/fail cannot be judged by an executable command is a design
  task, not an implementation task.
- **Work packets (§1.2):** the unit of work has `context` (allowed file scope), `spec`,
  `oracle` (runnable pass/fail command), `acceptance`, and `forbidden` (what neighboring
  packets own). Stay inside the declared scope — no incidental refactoring.
- **Context budget (§1.5):** each core crate must fit one context window — target ≤ 6,000,
  hard cap ≤ 10,000 source lines (tests excluded). Over the cap, a split packet is mandatory.
- **Reference oracles (§1.4):** MuJoCo (CPU) for physics, **PyTorch for Learning IR lowering**
  (same IR run in PyTorch is the ground truth), LeRobot for Observation IR / policy
  equivalence, MPFR/`rug` for transcendentals, golden images/tensors for the renderer. Golden
  files are CI read-only — never edit outputs to make a test pass.

## Language & toolchain (§2)

Pure Rust core, no C++ (C via `bindgen` only; Slang runs at build time). Python is a
first-class dependency on the *learning* path only (PyTorch/JAX, LeRobot datasets, USD bake,
MuJoCo reference); the core runtime, observation pipeline, policy inference, Safety Plane, and
real-robot deployment must run without Python. Key crates: `ash` (Vulkan), `gpu-allocator`,
`wide`/`multiversion`, `crossbeam`, `loom`, `PyO3`+`maturin`, `egui`+`winit`+`egui-snarl`,
`serde`, `proptest`, `blake3`, `ort` (ONNX, optional), `safetensors`. Shaders: Slang → SPIR-V,
offline-compiled with a content-hash cache.

## Commands (spec-defined; see §26.2, §1.5)

All verification routes through a single entry point, `cargo xtask` (`xtask/`), which CI
enforces. `es` is the runtime/CLI (not yet implemented).

- `cargo xtask context-budget` — enforce the per-crate line cap (§1.5)
- `cargo xtask` layering check — enforce the crate-layer rules in §4.2 (see the `LAYERS`
  table in Appendix C.8: `es-safety` ⇏ `es-policy`, `es-ir` knows neither compiler nor torch
  nor egui, only `es-transport` links CUDA/HIP)
- `xtask verify-goldens` — confirm golden files unchanged via git history
- `xtask check-scope` — diff must stay within the packet's declared scope
- `xtask check-spec-refs` — spec `§` cross-references resolve
- Standard Rust gates in PR CI (< 10 min): `clippy`, `fmt`, unit tests, IR
  schema/normalization-hash/cycle/type checks, determinism lints, Cross-IR fixtures, Safety
  Plane violation scenarios
- `es --check-deps` — print environment capabilities; `es task compile`, `es eval compare
  A.json B.json`, `es backend compare --task T --backends mjwarp,newton,mujoco-cpu`,
  `es evidence verify bundle.esb`, `es bench --memory-report`, `es deploy` (§9.6, §10.5, §17.2)

## Architecture (§4, §5)

Five typed IRs pipeline authoring → actuator:

1. **Task IR** (§6) — scene ref, goals, rewards, termination, randomization, reset; *declares*
   `ObservationSpec` but does not implement preprocessing. No neural nets. Split IR-D
   (dataflow DAG, M1) / IR-C (control, M4).
2. **Observation IR** (§7) — sensor → tensor. Carries `ImageSpec` (resolution, color space,
   camera model, intrinsics/extrinsics, distortion, shutter, exposure). Three-layer time
   model: `History` (system buffer) / `TemporalWindow` (learning-input meaning) /
   `TemporalEncoder` (network). One Task IR can carry several Observation IRs (same
   `task_hash`, different `observation_hash`).
3. **Learning IR** (§8) — tensor → action chunk. Encoders / fusion / temporal / policy head
   (Regression·Diffusion·FlowMatching) / chunker / unnormalizer. Represents ACT, Diffusion
   Policy, SmolVLA, π₀. Batch semantics are free (decoupled from env batch).
4. **Deployment IR + Safety Plane** (§9) — safety limits, execution mode, deadlines, fallback.
5. **Evaluation IR** (§10) — perturbation suite × metric × acceptance.

Everything is tracked by a hash chain (§5.3): `execution_hash = H(task, observation, learning,
policy, dataset, deployment, compiler, runtime, hardware_capability)`, the condition for
bitwise reproducibility (§3.5 tier 1). Four batch domains — simulation / observation /
inference / training — have independent sizes and schedules (§5.2, §12).

**Crate layering (§4.2, layers 0–12, CI-enforced):** `es-math`(0) → `es-core`(1) → `es-gpu`
/`es-assets`/`es-usd`(2) → `es-actuator`/`es-sensor`/`es-physics-core`(3) →
`es-physics-backend`/`-cpu`/`-gpu`(4) → `es-render`/`es-splat`(5) → `es-ir`(6) →
`es-compile`(7) → `es-policy`/`es-safety`(8) → `es-env`(9) → `es-data`/`es-telemetry`/
`es-eval`(10) → `es-ros2`/`es-py`/`es-script`/`es-transport`(11) → `es-editor`(12). Upper
depends on lower only; no same-layer deps.

## Non-negotiable invariants (§4.2, §7.2, §9, App. D) — do not violate

- **Safety is independent of policy.** `es-safety` must never depend on `es-policy` (rule 8,
  INV-11). No code path may disable the Safety Plane — not even for tests; widen the envelope
  instead (INV-12). `SafetyPlane::validate` does not return `Result`; do not change its
  signature (INV-13).
- **IR boundaries (§5.1):** no neural nets in Task IR (Learning IR owns them); Task IR only
  *declares* `ObservationSpec` (Observation IR owns preprocessing); no UI/layout types in
  `es-ir` — they live in `.eslayout` sidecars (rules 6, 7).
- **Intrinsics on resize/crop:** `Resize`/`Crop` must transform `ImageSpec` intrinsics; never
  skip it. `rescale_intrinsics=false` only when the user explicitly chose it (§7.2, INV-14).
- **Determinism (§3.4):** Vulkan float-controls is a capability *query*, not a setting;
  determinism is applied via SPIR-V execution modes. Forbidden in deterministic mode: atomic
  FP sums, fast-math, `HashMap`-iteration dependence, f64 time accumulation, standard
  transcendentals in physics/observation/reward kernels (use `es-math::approx`), global RNG.
- **Policy runtime (§2.4, §8.7):** IR owns pre/post-processing — do not push it into
  `PolicyRuntime`. Weights via `safetensors`; never add pickle-based loading (INV-16).
- **Only 7 extension points** (single-impl traits) allowed: `PhysicsBackend`, `PolicyRuntime`,
  `TaskNodeFactory`, `LearningNodeFactory`, `InferenceBackend`, `Scalar`, `DeterministicAcc`
  (INV-17). No speculative abstractions beyond these.
- **Never cut (§1.9):** the five IRs + validators, Safety Plane, Evaluation IR, hash chain,
  oracle infra, LeRobot compatibility.

Repository conventions live in the root `AGENTS.md` and `docs/` (`conventions.md`,
`invariants.md`, `design/`, `api-notes/`, `packets/`) once created; `docs/ARCHITECTURE.ko.md`
Appendix C summarizes the agent rules. Report unverified performance as `Target / Status:
unverified`, and use the 9-metric set, never a single `step/s` figure (§12.4).

## Conventions

- **Language.** Source code and commit messages are English only. Documentation
  (`*.md`, `*.txt`, `*.rst`, anything under `docs/`) may be multilingual — Korean is fine
  there (e.g. `docs/ARCHITECTURE.ko.md` is the canonical Korean spec by design; `docs/ARCHITECTURE.md`
  will be its English translation). A crate's UI string tables (`crates/*/i18n/*.toml`) may
  likewise hold non-English text — they exist so that the text a person reads lives in one
  place and no source file has to.
- **Commit format.** Conventional Commits: `<type>[(scope)][!]: <summary>`. Allowed types:
  `feat`, `fix`, `refactor`, `docs`, `init`, `test`, `chore`, `build`, `ci`, `perf`, `style`.
- **No trailers/footers.** Do not append `Co-Authored-By:`, `Claude-Session:`,
  `Signed-off-by:`, or `Generated with ...` lines to commit messages or pull request
  descriptions. (`.claude/settings.json` disables Claude Code's automatic attribution.)
- **Hooks.** Enforcement lives in `.githooks/` (`commit-msg` checks the format, bans
  trailers, and rejects non-English text (non-Latin letters) in commit messages; `pre-commit`
  rejects non-English text (non-Latin letters) in staged non-documentation files). Accented
  Latin (é, ü), symbols (→, ✓, —) and emoji are allowed. A fresh clone must run
  `git config core.hooksPath .githooks`
  once to activate them.
- `docs/ARCHITECTURE.ko.md` (Korean, canonical) and `docs/ARCHITECTURE.md` (English translation) must be staged in the same commit; the pre-commit hook rejects one without the other. Bypass with `SKIP_DOC_SYNC=1 git commit ...` only for a follow-up `docs:` commit that catches the translation up.
