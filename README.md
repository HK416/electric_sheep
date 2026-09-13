# Electric Sheep

Robot Learning Compiler & Runtime. Task, observation, learning, deployment and evaluation
are typed IRs, compiled to execution plans, run with identical semantics in simulation and
on real robots, and tracked end-to-end by a hash chain.

Canonical spec: `docs/ARCHITECTURE.ko.md`. Agent rules: `AGENTS.md`, `CLAUDE.md`.

## Status

M0 (contracts) complete; M1 (vision vertical slice) implemented for everything that runs
without a GPU or network. See `docs/reviews/` for milestone reviews and `docs/packets/`
for work packets. Verification via `cargo xtask ci`; `es --check-deps` lists which
optional oracles (MuJoCo, MJWarp, PyTorch) are available.

```bash
git config core.hooksPath .githooks
cargo xtask ci
```
