# Electric Sheep

Robot Learning Compiler & Runtime. Task, observation, learning, deployment and evaluation
are typed IRs, compiled to execution plans, run with identical semantics in simulation and
on real robots, and tracked end-to-end by a hash chain.

Canonical spec: `docs/ARCHITECTURE.ko.md`. Agent rules: `AGENTS.md`, `CLAUDE.md`.

## Status

M0 (contracts) in progress. Crates: `es-math`, `es-core`, `es-ir`, `es-assets`,
`es-physics-core`, `es-physics-backend`, `es-telemetry`; verification via `cargo xtask ci`.

```bash
git config core.hooksPath .githooks
cargo xtask ci
```
