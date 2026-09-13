# P-M4-R8 — a shared golden vector for the Python builder's Rust mirrors

Closes M4 review should-fix S-14 (`docs/reviews/M4.md`): `python/es/builder.py:35`'s
`stable_id` reimplements `es_core::StableId::from_path` (`crates/es-core/src/id.rs:19-24`) and
`builder.py:476-486`'s `_feature_ty`/`_chunk_ty` reimplement `crates/es-ir/src/learning.rs`'s
private `feature()`/`chunk()`. `id.rs:81`'s test asserted only self-consistency (equal inputs
give equal ids), so nothing pinned either side to the *other* side's actual bytes; a drift
would validate IR against the wrong bodies with no test failing anywhere.

Spec: spec §1.4 (oracle-first — a Python mirror with no shared oracle is an implementation gap,
not a design question), §5.3 (hash chain integrity depends on `StableId` agreeing everywhere),
§3.4 (determinism). No IR or hashing behavior changes; this packet only adds the missing pin.

## context

```
crates/es-core/src/id.rs      (golden hex literal added to the existing test)
python/es/selfcheck.py        (new — recomputes the same vectors in Python)
python/es/README.md           (new — documents the Rust side as canonical)
docs/packets/M4/P-M4-R8.md    (new)
```

## spec

- `StableId::from_path("robot/arm/joint_1")` is `blake3(path)` truncated to 16 bytes, hex
  form `6d2e29e8077ed3c571f21602d29c7145`. Pinned as a literal in
  `id::tests::from_path_is_stable_and_distinct` (Rust) and `python/es/selfcheck.py`'s
  `STABLE_ID_VECTOR` (Python) — both recompute from the same string and must agree byte for
  byte.
- `_feature_ty(dim, tokens)` mirrors `feature()`: `tokens == 0` gives shape `[dim]`, otherwise
  `[tokens, dim]`, always `Dimensionless`/`Policy`. Pinned at `(512, 0) -> shape [512]`
  (`FEATURE_TY_VECTOR`).
- `_chunk_ty(horizon, action_dim)` mirrors `chunk()`: shape `[horizon, action_dim]`,
  `Normalized(-1, 1)` (`action_unit()`), `Policy` frame. Pinned at ACT's own `(100, 14)`
  (`CHUNK_TY_VECTOR`), matching `docs/api-notes/lerobot-act.md`'s `chunk_size`/`action_dim`.
- `python/es/selfcheck.py` is a standalone, dependency-free check: it imports only
  `python/es/builder.py` and asserts the three vectors above. No test framework, no fixtures —
  three `assert`s and a print, matching the rest of `python/es`'s style.
- `python/es/README.md` states plainly that `crates/es-core/src/id.rs` and
  `crates/es-ir/src/learning.rs` are canonical; the Python functions are convenience mirrors
  that exist so the authoring builder doesn't shell out to Rust for every id and port type.

## oracle

```
cargo fmt -p es-core --check
cargo clippy -p es-core --all-targets -- -D warnings
cargo test -p es-core
PYTHONPATH=python <venv>/Scripts/python.exe -m es.selfcheck
```

(`es.selfcheck` imports `es.builder`, which imports `es_native` — build it once with
`maturin develop --release --features python` from `python/es` if it isn't already installed
in the venv.)

## acceptance

- `cargo test -p es-core` passes with the pinned hex literal in place.
- `python -m es.selfcheck` prints `OK` and exits 0 against the same venv used for the ACT
  oracle (spec §1.4).
- A deliberate one-character edit to either `StableId::from_path` or `stable_id` (or to
  `feature()`/`_feature_ty`, `chunk()`/`_chunk_ty`) fails exactly one of the two checks above —
  confirming the vectors are load-bearing, not decorative.
- `docs/packets/M4/P-M4-R8.md` and `python/es/README.md` exist and cross-reference each other.

## forbidden

- `crates/es-py/**`, `crates/es-ir/**` — no behavior change to the pyo3 extension or the IR
  crate; this packet only pins existing outputs.
- Any other packet's scope (`es-env`, `es-eval`, `es-data`, `es-usd`, `es-script`, `es-gpu`,
  `es-render`, and S-15's `docs/api-notes/lerobot-act.md` beyond what P-M4-S15 owns).
