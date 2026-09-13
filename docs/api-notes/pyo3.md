# pyo3 (pinned API digest)

Pinned for `crates/es-py` (M2 W6, spec 14.2 Python authoring builder). Version and feature set
live in the workspace `Cargo.toml`; this note is what to remember about the API surface itself,
not a substitute for `docs.rs/pyo3/0.29.2`.

## Pin

```toml
# workspace Cargo.toml
pyo3 = { version = "0.29.2", features = ["extension-module", "abi3-py312"] }
```

- **`abi3-py312`**: compiles against CPython's stable ABI as of 3.12. One wheel
  (`*-cp312-abi3-*.whl`) then loads on 3.12, 3.13, 3.14, ... — no per-minor-version rebuild.
  Matches this repo's pinned Python (3.12 system + `.venv`). Losing this pin (bumping the
  `abi3-py3xx` tag) is a compatibility decision, not a routine version bump.
- **`extension-module`**: tells pyo3 to link against libpython lazily (the symbols come from
  the host interpreter at import time, not the build). Required for a `cdylib` Python loads;
  wrong for embedding Python *inside* a Rust binary (not what this packet does).
- Both features are requested at the *workspace* level (so `cargo metadata`/lockfile stay
  single-featured for the one dependent), but only reach the build when `es-py`'s own `python`
  feature is on (`pyo3 = { workspace = true, optional = true }` + `python = ["dep:pyo3"]`). The
  default `cargo build`/`cargo test --workspace` never activates it, so CI never needs libpython
  headers or a Python interpreter to build the rest of the workspace (CLAUDE.md, spec 2.1).

## Module shape used here

- One `#[pymodule] fn es_native(m: &Bound<'_, PyModule>) -> PyResult<()>` (0.21+ "Bound" API —
  the pre-0.21 `&PyModule` signature is gone). Its name **must** match the last path segment of
  maturin's `module-name` (`python/es/pyproject.toml` sets `module-name = "es.es_native"`),
  because the macro's generated `PyInit_<name>` C symbol is derived from the function name, not
  from where the function is declared or whether it is `pub`.
- `#[pyclass(unsendable)]` on every builder wrapper (`Task`, `Observation`, `Learning`,
  `Deployment`): pyo3 requires a plain `#[pyclass]` to be `Send + Sync` (so it can, in
  principle, cross Python's GIL-releasing boundary safely); the node factories these wrap
  (`Box<dyn TaskNodeFactory>` etc.) are neither, and an authoring script only ever touches one
  builder from one thread, so `unsendable` (skip the check, panic if it is ever used from a
  different OS thread than it was created on) is the honest fit rather than adding `Mutex`
  wrapping nothing needs.
- Consuming methods (`build`, `save`) are modeled as `Option<Builder>` fields with `.take()`,
  not `self` by value: `#[pymethods]` cannot take a receiver by value for a `#[pyclass(unsendable)]`
  type instance the Python side still holds a reference to.
- Custom exception: `pyo3::create_exception!(es_native, EsDiagnosticError, PyException)`. Raised
  with a 2-tuple `(display_text, diagnostics_json)` so a caller can either print it as-is or
  `json.loads(e.args[1])` for structured `Diagnostic`s.
- Params cross the FFI boundary as **JSON text**, not a hand-walked `PyObject` tree: Python's
  stdlib `json.dumps`/`json.loads` already does exactly that conversion, and no
  `pythonize`/`serde-pyobject`-style crate is on this packet's approved dependency list (`pyo3`
  only) — see `crates/es-py/src/pybind.rs`'s module doc comment.

## Gotchas hit while building this packet

- A `#[pymethods]` `impl` block exposes **every** function in it to Python, including ones meant
  as private helpers — there is no `pub(crate)`-style opt-out inside the block. Helpers like
  `fn inner(&mut self) -> PyResult<&mut Builder>` (not a valid Python return type — `Builder`
  isn't a `pyclass`) must live in a second, plain `impl Task { .. }` block instead.
- `#[pyclass]` without `unsendable` fails at compile time (not runtime) with a `Send`/`Sync`
  trait-bound error whose root cause is buried several `note:`s down a long trait-resolution
  chain — worth checking first for any pyclass wrapping a `Box<dyn Trait>`.
