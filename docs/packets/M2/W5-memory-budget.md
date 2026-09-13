# W5 — Memory budget model + `es bench --memory-report` (offline part)

Spec: spec 20.1, spec 20.2, spec 20.3, spec 15.2, spec 12.4, spec 5.2, spec 28.4,
spec 28.7 gate 13. Design note: `docs/design/memory-budget.md`.

## context

```
crates/es-compile/src/budget.rs      (new)
crates/es-compile/src/lib.rs         (adds `pub mod budget;` and its re-exports)
crates/es-compile/Cargo.toml         (no change)
crates/es/src/cmd/bench.rs           (new)
crates/es/src/cmd/mod.rs             (adds `pub mod bench;`)
crates/es/src/main.rs                (adds `bench` dispatch + usage line)
crates/es/Cargo.toml                 (adds `es-telemetry` dependency)
crates/es/tests/cli.rs               (appends bench tests)
docs/design/memory-budget.md         (new)
docs/packets/M2/W5-memory-budget.md  (new, this file)
```

Other M2 W5 sub-packets (GPU lowering optimization, the live measurement loop behind the
9-metric table, wiring the budget model into `es-env`'s auto-shrink and `es-editor`'s
compile-time check) are **out of scope**: this packet is the static/offline half only, and it
cannot be otherwise — there is no GPU in this environment to measure against (spec 28.4's
±10% accuracy gate is `unverified` here, see design note §6).

## spec

- `es_compile::budget::MemoryBudget::estimate(&BudgetInputs) -> MemoryReport` — every spec
  20.2 line item (`physics_state`, `render_tile_atlas`, `observation_intermediates`,
  `history_buffers`, `policy_weights`, `inference_activations`, `chunk_buffers`), each a
  `BudgetItem { name, bytes, formula }` with the formula spelled out for audit. `bytes == 0`
  with a formula starting `"unavailable: ..."` means "no data", not "free" — `policy_weights`
  is always unavailable (`WeightsRef` carries no byte size, `INV-16`).
- `MemoryReport { items, total_bytes, per_domain: BTreeMap<String, u64>, bandwidth_per_tick }`
  and its `Display` (a GiB table).
- `MemoryReport::violations(&BudgetInputs) -> Vec<BudgetViolation>` — spec 20.3's
  `obs_batch <= sim_batch` rule (always checked) and `total <= device_bytes - reserve` rule
  (checked when `device_bytes` is given), plus a `maxImageDimension2D` (spec 15.2) check when
  a `TileAtlasCfg` is given.
- `es bench --memory-report --obs <obs.toml> [--learning <learning.toml>] [--scene <mjcf|urdf>]
  --sim-envs N --obs-envs N --views N --inference-batch N [--precision f16|f32]
  [--device-gib G] [--tile-w N --tile-h N --tiles-per-row N]` — prints the table and any
  violations; exits 1 on a violation, 2 on a usage error, 0 otherwise. `--scene` is
  best-effort: an unavailable/unparseable backend prints a note and leaves `physics_state`
  `unavailable` rather than failing the command.
- `es bench` (no `--memory-report`) prints the spec 12.4 nine-metric table via
  `es_telemetry::PerfMetrics::default()` — every field `unmeasured` — plus one line:
  `Target / Status: unverified` (the measurement loop is a separate M2 W5 sub-packet). No new
  metric type is defined: `es-compile` (layer 7) does not need one, and the CLI (layer 11)
  already has `es-telemetry` (layer 10) available.
- `ModelSizes { nq, nv, nu, nsensordata }` mirrors `es_physics_core::backend::ModelInfo`'s
  fields but is defined locally in `es-compile`: adding an `es-physics-core` (layer 3)
  dependency to `es-compile` (layer 7) for four `u32`s is out of this packet's declared file
  scope, and the CLI (which already depends on both) copies them across at the call site.
- `BudgetDomains { n_sim_envs, n_obs_envs, n_views, inference_batch }` is shaped like
  `es_env::scheduler::BatchDomains` but defined locally for the same layering reason
  (`es-env` is layer 9, above `es-compile`).

## oracle

```
cargo fmt -p es-compile -p es --check
cargo clippy -p es-compile -p es --all-targets -- -D warnings
cargo test -p es-compile -p es
cargo xtask check-spec-refs
```

## acceptance

- `budget.rs` unit tests: a hand-built `ObservationIr`/`LearningGraph`/`ModelSizes` fixture
  with every byte count asserted exactly (not just "> 0"); `render_tile_atlas` and
  `policy_weights` asserted `unavailable` (bytes `0`, formula starts `"unavailable"`) on that
  fixture (no camera node, no weight size); both spec 20.3 rules exercised as violations;
  every item's formula string asserted non-empty; `Display` asserted to contain `"GiB"`,
  `"total"` and `"per domain:"`.
- `crates/es/tests/cli.rs`: `es bench` (no flag) contains all nine spec 12.4 metric names,
  `unmeasured`, and the `Target / Status: unverified` line. `es bench --memory-report` against
  the existing cross-IR fixture's Observation/Learning IR (written to TOML via
  `write_fixture_toml`, spec 5.1's `es_ir::serial`) prints every budget item name, `GiB`, and
  `per domain:`, and exits 0 with no violations at a balanced `--obs-envs <= --sim-envs`; the
  same fixture with `--obs-envs > --sim-envs` exits 1 and names
  `obs_batch_le_sim_batch`; `--memory-report` without `--obs` is a usage error (exit 2).

## forbidden

- `crates/es-env/**`, `crates/es-eval/**`, `crates/es-policy/**`, `crates/es-data/**` — other
  agents' concurrent M2 packets.
- Any change to `WeightsRef` (adding a byte-size field): `policy_weights` stays `unavailable`
  until a packet in `es-ir`'s scope decides to widen the schema.
- The live measurement loop behind the nine `PerfMetrics` fields, the `N_obs`/`N_inf`
  auto-shrink loop, and the editor's compile-time budget-exceeded error (spec 20.3) — all need
  a running backend/GPU or `es-editor` (layer 12) and are separate packets.
- Root `Cargo.toml`.
