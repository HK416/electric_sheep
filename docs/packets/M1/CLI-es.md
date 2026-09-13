# CLI-es — the `es` runtime/CLI

Spec: §2.5 (`es --check-deps`, external tool dependency table), §5.3 (hash chain), §10.5
(Evaluation IR artifacts, `es eval compare`), §11.1 (unified compiler pipeline, Cross-IR
Check), §17.2 (`es backend compare`, out of scope for this packet), §9.6 (`policy.esb`,
out of scope for this packet), §26.2 (CI command surface).

## context

```
crates/es/Cargo.toml
crates/es/src/main.rs
crates/es/src/error.rs
crates/es/src/util.rs
crates/es/src/cmd/mod.rs
crates/es/src/cmd/check_deps.rs
crates/es/src/cmd/ir.rs
crates/es/src/cmd/task.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/dataset.rs
crates/es/tests/cli.rs
docs/packets/M1/CLI-es.md
```

## spec

A new binary crate, `es`, at the language-binding layer (layer 11 of §4.2's table --
allowed to depend on anything layer <= 10; nothing may depend back on it). No `clap`: a
hand-rolled `std::env::args` dispatcher, matching the convention `xtask` already set.
`CliError { Usage(String), Runtime(String) }` is the one error type every subcommand
returns; exit code 2 for a usage error, 1 for a runtime failure or a diagnostic of
`Severity::Error`, 0 otherwise. No `unwrap` on user-supplied input (file paths, file
contents, CLI arguments).

- **`es --check-deps`** (§2.5) -- never fails. Probes: a Python interpreter (`python`,
  `python3`, or `$ES_PYTHON`, mirroring `es-physics-backend::proc::python_candidates`),
  `mujoco` importable (delegates to `MuJoCoCpuBackend::is_available`, which already does
  its own interpreter search), `torch` and `lerobot` importable (`python -c "import
  ..."`, spawned with a 10s timeout via polling `try_wait` so a wedged interpreter cannot
  hang the CLI), and a Vulkan loader (`vulkan-1.dll` on `%PATH%`/System32, or
  `libvulkan.so.1` under the usual lib dirs on Unix -- a file-existence heuristic only,
  labelled as such in the output, never a driver probe). Reports which §2.5 capabilities
  follow: the PyTorch learning path, the MuJoCo (CPU) oracle, LeRobot dataset export
  (dataset *read* is Rust-native and always available), and Slang/Vulkan GPU lowering
  (also noting a cache hit needs none of this).
- **`es ir validate <file.toml>...`** -- detects the file's `IrKind` from its envelope
  (`es_ir::serial`: peek the `kind` field via `toml::Value` before choosing which typed
  `*_from_toml` to call), runs that IR's own `validate()`, prints each `Diagnostic` via
  its spec block-format `Display`, then prints the IR's own `*_hash` as lowercase hex.
- **`es ir check <task> <obs> <learning> <deploy> [eval]`** -- parses the five files
  positionally (their own `*_from_toml` already rejects a `KindMismatch`), builds an
  `IrBundle`, runs `es_ir::cross::check`, prints diagnostics, then prints every hash
  chain slot §5.3 defines that authoring time can know: task, observation, learning,
  policy (`LearningGraph::policy_hash`), deployment, evaluation (`unset` if the file was
  omitted), and compiler (`CpuPlan::compile` the observation IR in `PlanMode::Debug`
  purely to read `compiler_hash` off the result -- that hash depends only on crate
  version, plan mode and the kernel table, not on IR content, but the method lives on a
  compiled `CpuPlan`). `dataset`/`runtime`/`hardware` are always `unset`: only known at
  run time, never at authoring time.
- **`es task compile <task> <obs> [--release]`** -- validates the Task IR structurally
  (`--help` says plainly that the task *executor* lives in `es-env`, not here) and
  compiles the Observation IR via `CpuPlan::compile`, printing the node execution order,
  the buffer table (dtype, shape, home, byte size), the total arena size, and
  `compiler_hash`.
- **`es eval compare <A.json> <B.json>`** -- both files are `es_ir::evaluation::
  EvaluationReport` JSON. Prints a per-suite/per-metric table: A's value, B's value,
  delta, and a significance column. `EvaluationReport` (spec 10.5) carries only per-cell
  aggregates (`MetricValue::Scalar`/`::Histogram`), never raw per-episode samples, so the
  command also re-parses each file as a generic `serde_json::Value` and looks for a
  non-standard `cells[i].samples: [f64, ...]` extension; only when *both* sides carry it
  for a matching `(suite, metric)` cell does it compute a two-sided Welch t-test p-value
  (plain Rust: Welch-Satterthwaite df, then the regularized incomplete beta via Lentz's
  continued fraction and a Lanczos `ln_gamma` -- no stats crate) and flag `|p| < 0.05`.
  Otherwise the column reads `n/a (aggregate-only report)`.
- **`es dataset info <root>`** -- opens a `LeRobotDataset`, prints every feature as the
  `PortType` it presents at an Observation IR port (`FeatureSpec::port_type`, `Err` for
  `string` features, which have none), the episode count, and the three `DatasetIdentity`
  hashes (content/schema/split) computed over a display-only all-train `Split` (`dataset
  info` has no caller-supplied split to identify with).
- `es --version`, `es --help`, and `es <subcommand> --help` for every subcommand.

## oracle

```
cargo fmt -p es --check
cargo clippy -p es --all-targets -- -D warnings
cargo test -p es
cargo xtask layering
cargo xtask context-budget
```

## acceptance

- `--check-deps` exits 0 in every environment, including one with no Python and no
  Vulkan loader, and correctly labels the Vulkan check as a heuristic.
- `ir validate` on each of the five IR kinds finds the right `validate()` and prints the
  matching `*_hash`; an unknown `kind` or a file that isn't valid TOML is a runtime error
  (exit 1), not a panic.
- `ir check` on a cross-referenced, fully valid bundle (built the way
  `crates/es-ir/tests/cross_fixture.rs`'s `Fixture::new()` does) produces zero
  diagnostics and every one of task/observation/learning/policy/deployment/compiler as a
  real hash, `evaluation` as a hash when given and `unset` when omitted, and
  dataset/runtime/hardware always `unset`.
- `task compile` on that same fixture's task+observation pair succeeds, and its
  `compiler_hash` matches calling `CpuPlan::compile` directly.
- `eval compare` on two hand-authored `EvaluationReport` JSON files prints one row per
  distinct `(suite, metric)`, correctly marks cells present in only one side, and only
  emits a `p=` significance value when both sides carry a `samples` array for that cell.
- `dataset info` on a fixture LeRobot dataset lists every non-`string` feature's
  `PortType`, the right episode count, and three non-zero identity hashes.
- No subcommand's implementation calls `.unwrap()`/`.expect()` on data that originates
  from a CLI argument, a file's contents, or a spawned process's output.
- `cargo xtask layering` reports no violation for the `es` crate (it does not match the
  `es-*` pattern the LAYERS table gates on, so no table entry is required; it is a
  binary, so nothing can depend back on it either way).
- `crates/es/**` stays at or under ~1,200 source lines (tests excluded).

## forbidden

- `crates/es-policy/**` and any file under `docs/ARCHITECTURE*.md` -- owned by
  concurrently in-flight packets.
- Editing `xtask/src/layering.rs` unless the layering check actually rejects a crate
  named `es`; it currently does not (`LAYERS` only gates names starting `es-`), so this
  packet does not touch it.
- Editing the root `Cargo.toml` -- the `crates/*` glob already covers `crates/es`, and
  every dependency here is `{ workspace = true }`.
- `clap` or any other CLI-argument-parsing crate.
- Modifying `crates/es-ir/tests/cross_fixture.rs` -- `crates/es/tests/cli.rs` copies the
  fixture-building functions it needs (not the violation-scenario tests) rather than
  changing the original.
- Committing.
