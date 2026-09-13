"""Golden-vector check for the Python mirrors of Rust logic (spec §1.4, M4 review S-14).

`stable_id` (`builder.py`) reimplements `es_core::StableId::from_path` and `_feature_ty`/
`_chunk_ty` reimplement `es-ir/src/learning.rs`'s private `feature()`/`chunk()` — none of them
call into Rust, so nothing catches the two sides drifting apart except a shared golden vector.
`crates/es-core/src/id.rs`'s `from_path_is_stable_and_distinct` test pins the same hex literal
this file checks; the Rust side is canonical (see `python/es/README.md`).

Run with the venv interpreter from the repo root:

    PYTHONPATH=python <venv>/python -m es.selfcheck
"""

from .builder import _chunk_ty, _feature_ty, stable_id

# Pinned in crates/es-core/src/id.rs's `from_path_is_stable_and_distinct` test.
STABLE_ID_VECTOR = ("robot/arm/joint_1", "6d2e29e8077ed3c571f21602d29c7145")

# Mirrors `feature()` in crates/es-ir/src/learning.rs (untokenized: shape is just `[dim]`).
FEATURE_TY_VECTOR = (
    (512, 0),
    {"elem": "F32", "shape": [512], "unit": "Dimensionless", "frame": "Policy", "time": "Tick", "image": None},
)

# Mirrors `chunk()` in crates/es-ir/src/learning.rs (ACT's own horizon/action_dim, §8.9).
CHUNK_TY_VECTOR = (
    (100, 14),
    {
        "elem": "F32",
        "shape": [100, 14],
        "unit": {"Normalized": {"lo": -1.0, "hi": 1.0}},
        "frame": "Policy",
        "time": "Tick",
        "image": None,
    },
)


def main() -> None:
    path, want = STABLE_ID_VECTOR
    got = stable_id(path)
    assert got == want, f"stable_id({path!r}) = {got}, expected {want} (see crates/es-core/src/id.rs)"

    (dim, tokens), want = FEATURE_TY_VECTOR
    got = _feature_ty(dim, tokens)
    assert got == want, f"_feature_ty{dim, tokens} = {got}, expected {want}"

    (horizon, action_dim), want = CHUNK_TY_VECTOR
    got = _chunk_ty(horizon, action_dim)
    assert got == want, f"_chunk_ty{horizon, action_dim} = {got}, expected {want}"

    print("OK: stable_id, _feature_ty, _chunk_ty match their Rust-side golden vectors")


if __name__ == "__main__":
    main()
