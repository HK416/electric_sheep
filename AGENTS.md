# AGENTS.md — rules for coding agents

Canonical spec: `docs/ARCHITECTURE.ko.md` (Korean). Read the section a packet cites before
writing code. `CLAUDE.md` summarizes the architecture and invariants.

## Workflow
- Every packet has `context / spec / oracle / acceptance / forbidden` (spec 1.2). Stay in scope.
- Oracle first (spec 1.4): write the test/property/fixture before the implementation.
- Gate before reporting done: `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace && cargo xtask context-budget && cargo xtask layering`.
- Source code and commit messages are English only (git hooks reject non-Latin letters).
- Commit format: Conventional Commits, no trailers.
- Core crate ≤ 6,000 target / ≤ 10,000 hard cap source lines (tests excluded).

## Electric Sheep rules (spec Appendix C.1)

### IR boundaries (spec 5.1)
No neural nets in Task IR; Learning IR owns them. Task IR only *declares* `ObservationSpec`;
preprocessing belongs to Observation IR. No UI/layout types in IR (`.eslayout` sidecar).

### Safety (spec 9, 4.2 rule 8)
`es-safety` never depends on `es-policy`. No code path disables the Safety Plane, not even in
tests (widen the envelope instead). `SafetyPlane::validate` does not return `Result`.

### Images (spec 7.2)
Resize/Crop go through `ImageSpec::resized / cropped`; never skip intrinsics transforms.
`rescale_intrinsics=false` only on explicit user choice.

### Policy runtime (spec 2.4, 8.7)
Pre/post-processing is owned by the IR, not `PolicyRuntime`. Weights via safetensors; no pickle.

### Determinism (spec 3.4)
Vulkan float-controls is a capability query, not a setting; apply via SPIR-V execution modes.
Forbidden in deterministic paths: atomic FP sums, fast-math, `HashMap` iteration order, f64
time accumulation, std transcendentals in physics/observation/reward kernels (use
`es_math::approx`), global RNG.

### Extension points (INV-17)
Only these single-impl traits are allowed: `PhysicsBackend`, `PolicyRuntime`,
`TaskNodeFactory`, `LearningNodeFactory`, `InferenceBackend`, `Scalar`, `DeterministicAcc`.

### Performance (spec 12.4)
Never a single `step/s` figure; use the 9-metric set. Unverified numbers are
`Target / Status: unverified`.
