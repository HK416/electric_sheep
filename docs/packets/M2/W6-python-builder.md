# M2 W6 (remainder) — Python authoring builder (`es-py`, spec 14.2)

Implements the Python authoring frontend spec 14.1/14.2 describes: five (here, four —
Evaluation IR out of scope) typed-IR builders reachable from Python, backed by the same node
factories the editor and CLI use, so `es.Task(...)`/`.save(...)` and a hand-written `task.toml`
produce identical IR. Design note: `docs/design/python-builder.md`. API digest:
`docs/api-notes/pyo3.md`.

## context

```
crates/es-py/**                         (new crate, layer 11)
Cargo.toml                              (add `es-py` member entry, `pyo3` workspace dependency)
python/es/**                            (new: __init__.py, builder.py, math.py, pyproject.toml,
                                          examples/pick_place.py)
docs/api-notes/pyo3.md                  (new)
docs/design/python-builder.md           (new)
docs/packets/M2/W6-python-builder.md    (new)
```

## spec

- **`crates/es-py/src/builder.rs`**: language-neutral core, no `pyo3`, compiled and tested by
  default (`cargo test -p es-py` with zero features). `TaskBuilder`, `ObservationBuilder`,
  `LearningBuilder` wrap `TaskNodeRegistry`/`LearningNodeRegistry`/`ObservationNode`'s own tagged
  enum with `add(kind, params: serde_json::Value) -> Result<NodeId, Diagnostic>`, `connect`,
  a `declare_*` method for that IR's boundary declarations, `build(self) -> Result<Ir,
  Vec<Diagnostic>>` (assembles the IR and calls its own `validate()`), and a `save_toml(&Ir,
  &Path)` associated function using `es_ir::serial`. `DeploymentBuilder` is direct setters
  instead (`DeploymentIr` is a struct, not a graph — see the design note). No per-node-kind
  match arm anywhere in this file.
- **`crates/es-py/src/pybind.rs`** (`#[cfg(feature = "python")]`): pyo3 `es_native` module.
  `Task`/`Observation`/`Learning`/`Deployment` pyclasses expose the generic `add`/`connect`/
  `declare_*`/`build`/`save`, plus the handful of spec-14.2-named sugar methods (`Task.observe`,
  `.reward`, `.terminate`; `Learning.act` as a static convenience constructor) — every one of
  them still just calls `add`/`connect`. Errors become `EsDiagnosticError` carrying
  `(display_text, diagnostics_json)`.
- **`python/es/`**: pure-Python package (`Task`, `Observation`, `Learning`, `Deployment`,
  `Expr` with operator overloads, `es.math.norm`) matching the spec 14.2 snippet's names and
  call shapes, built on `es_native`. `pyproject.toml` (`[tool.maturin]`, `manifest-path` into
  `crates/es-py`, `features = ["python"]`) is what `maturin develop` reads.
  `examples/pick_place.py` reproduces the spec 14.2 snippet and writes `task.toml` /
  `observation.toml` / `learning.toml` / `deployment.toml` (spec 14.3 format).
- **`pyo3` is optional, off by default** (`es-py`'s `python` feature, `dep:pyo3`): the default
  workspace build never links libpython. `maturin` enables it explicitly.
- Constraints: English only, `BTreeMap` only (none needed directly — inherited from `es-ir`
  types), no new trait (`INV-17`: `TaskNodeFactory`/`LearningNodeFactory` are reused, not
  extended), roughly 1000 new Rust lines across `builder.rs` + `pybind.rs`.

## oracle

```
cargo fmt -p es-py --check
cargo clippy -p es-py --all-targets -- -D warnings
cargo test -p es-py
cargo xtask layering
```

- `crates/es-py/src/builder.rs` unit tests (no Python): building a minimal valid Task graph
  through `add`/`connect` and round-tripping it through `save_toml`/`task_from_toml`; an unknown
  kind reporting `FACTORY-001` as a `Diagnostic` rather than panicking; `declare_obs` round-
  tripping an `ObsChannel`; building an `ObservationIr` with no registry involved; both
  `LearningBuilder`/`DeploymentBuilder` reporting a clear diagnostic instead of panicking when
  required state (a policy handle; every `DeploymentIr` field) is missing.
- `crates/es-py/tests/python_example.rs`: **SKIPs** unless `ES_PYTHON` is set (no `es` CLI exists
  yet to shell out to — `CLAUDE.md`). With `ES_PYTHON=<python with `es` installed>`, runs
  `python/es/examples/pick_place.py`, parses the four TOML files it writes with
  `es_ir::serial::*_from_toml`, and asserts each IR's own `.validate()` reports zero errors —
  the same validators an `es ir validate` CLI would eventually call.

## acceptance

- `cargo fmt -p es-py --check && cargo clippy -p es-py --all-targets -- -D warnings && cargo
  test -p es-py && cargo xtask layering` all green.
- `cargo check --workspace --all-targets` compiles `es-py` cleanly (workspace-wide breakage in
  other, concurrently-edited crates is out of this packet's control and gate).
- Verified live in this session: `maturin develop --release` (from `python/es`, `.venv` at
  repo root) builds and installs `es` + the `es_native` extension; running
  `python/es/examples/pick_place.py` produces `task.toml`/`observation.toml`/`learning.toml`/
  `deployment.toml`, and `ES_PYTHON=<venv python> cargo test -p es-py --test python_example`
  passes (all four validate with zero errors).

## forbidden

- Any crate other than `crates/es-py` (new), plus the two listed edits to the root `Cargo.toml`
  (the `es-py` member entry and the `pyo3` workspace dependency line) — every other crate is
  owned by a concurrent packet.
- Evaluation IR — spec 14.2's own snippet does not builder it either; out of scope here.
- Replaying cross-IR checks (`es-ir::cross`, spec 11.1) client-side — that is the compiler's
  job, not this builder's.
- Committing.
