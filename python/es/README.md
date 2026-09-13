# `es` — Python authoring builder (spec 14.2)

Thin Python frontend over the `es_native` pyo3 extension (`crates/es-py`), which itself calls
the language-neutral Rust builder core. See `docs/design/python-builder.md` for the layering.

## The Rust side is canonical

A handful of small formulas in `builder.py` reimplement Rust logic locally instead of calling
into `es_native`, because the values are needed for client-side type inference before a node
has been built (an editor needs to know a port's shape before running the graph once):

- `stable_id(path)` mirrors `es_core::StableId::from_path` (`crates/es-core/src/id.rs`):
  `blake3(path)` truncated to 16 bytes, hex-encoded.
- `_feature_ty(dim, tokens)` and `_chunk_ty(horizon, action_dim)` mirror `feature()`/`chunk()`
  in `crates/es-ir/src/learning.rs`.

**Rust is the source of truth for all three.** If a value here ever disagrees with the Rust
implementation, the Rust implementation is right and this file is the one to fix. `python/es/
selfcheck.py` pins one golden vector per function, matching the literal pinned in
`crates/es-core/src/id.rs`'s `from_path_is_stable_and_distinct` test, so a drift on either side
fails loudly instead of producing IR that validates against the wrong bodies (M4 review S-14).

## Running the self-check

```
maturin develop --release --features python   # from this directory, once, to build es_native
PYTHONPATH=python <venv>/Scripts/python.exe -m es.selfcheck
```
