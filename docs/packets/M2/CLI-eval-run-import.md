# CLI-eval-run-import — `es eval run` and `es import lerobot-config`

Spec: §10.5 (`es eval run --config ... --policy policy.esb` -> `report.json`, `report.html`,
`episodes/`, `evaluation.lock`), §14.4 (external conversion, LeRobot config is the priority
target), §1.4 (oracle-first: an evaluation this machine cannot really run is refused, never
faked), §9.6 (`policy.esb`), §5.3 (hash chain).

## context

```
crates/es/src/cmd/eval.rs      (+ `run` subcommand, alongside the existing `compare`)
crates/es/src/cmd/import.rs    (new: `es import lerobot-config`)
crates/es/src/cmd/mod.rs       (+ `pub mod import;`)
crates/es/src/cmd/backend.rs   (`load_scene` made `pub(crate)`, reused by `eval run`)
crates/es/src/main.rs          (+ `import` dispatch, usage text)
crates/es/Cargo.toml           (+ `es-eval`, `es-env`, `es-safety`, `es-policy` deps)
crates/es/tests/cli.rs
docs/packets/M2/CLI-eval-run-import.md
```

## spec

1. `es eval run --config <eval.toml> --policy <policy.esb> --scene <file.xml|urdf> [--out
   <dir>] [--backend mujoco-cpu] [--runtime torch]`:
   - Opens the bundle (`es_compile::PolicyBundle::open`, spec 9.6 — re-validates every IR and
     hash) and parses the Evaluation IR from `--config`
     (`es_ir::serial::evaluation_from_toml`).
   - Checks `MuJoCoCpuBackend::is_available()` and `es_policy::torch_runtime::is_available()`
     *before* touching the scene file. Either unavailable prints `SKIPPED (<reason>)` and
     exits **3** — a distinct code from usage (2) and failure (1), because nothing ran.
   - Otherwise loads the scene, loads the policy (`TorchRuntime::load` against
     `WeightsSource::InMemory(bundle.weights)`), and calls `es_eval::Evaluation::run`. Its
     `NJ`/`H` const generics are resolved at runtime from
     `bundle.deployment.{robot.n_joints, action.horizon}` through a small fixed dispatch
     table (ponytail: enumerated pairs, not a runtime-generic solver — add a pair for a new
     robot/horizon combination); an unlisted pair is a `CliError::Runtime`, not a panic.
   - Writes `report.json` and `evaluation.lock` via `es_eval::write_artifacts`, and
     `report.html` — a hand-written, escaped HTML table over the same data, no template
     crate. **`episodes/` is not produced by this build** (no renderer yet, `--help` and this
     packet both say so).
   - Exit 0 iff every `AcceptanceResult` is `Determined { passed: true }`; exit 1 if any is
     `Determined { passed: false }` or `Unavailable` (both printed).

2. `es import lerobot-config --config <config.json> [--stats <stats.json>] [--dataset <root>]
   --out <dir>`: runs `es_data::lerobot_config::convert`, writes `<out>/observation.toml` and
   `<out>/learning.toml` via `es_ir::serial`, prints every conversion warning and both content
   hashes (`observation_hash`, `learning_hash`). Exit 1 on a `ConfigError` (bad JSON,
   unsupported policy type, missing feature) or an I/O error; 2 on a usage error.

## oracle

```
cargo fmt -p es --check
cargo clippy -p es --all-targets -- -D warnings
cargo test -p es
```

## acceptance

- `eval run` against a bundle built in-test with `PolicyBundle::build` (the M1
  `crates/es/tests/cli.rs` cross-IR fixture, weights re-pointed at a hand-written byte blob
  exactly as `crates/es-runtime-embedded/tests/embedded.rs`'s `deployable()` does it) exits
  **3** and prints `SKIPPED` when `mujoco`/`torch` are absent, which is always true in CI's PR
  job (spec 1.4) — the backend/runtime check runs before the (nonexistent) `--scene` file is
  ever read.
- `import lerobot-config` on `tests/fixtures/lerobot_config/act_config.json` writes
  `observation.toml` / `learning.toml` that round-trip clean (no `ERROR`) through
  `es ir validate`, and prints both hashes.

## forbidden

- Any other crate's `src/` (`es-eval`, `es-compile`, `es-policy`, `es-physics-backend`,
  `es-data`, `es-ir` are consumed, not modified).
- `episodes/` replay — no renderer exists yet; this packet documents the gap, it does not
  close it.
- `HashMap`/`HashSet` — `BTreeMap` only.
- A new extension-point trait (INV-17); this packet adds none.
- A template-engine dependency for `report.html` — hand-written and escaped instead.
- Committing — the oracle is run and reported, not landed, by this packet.
