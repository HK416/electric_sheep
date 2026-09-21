"""Golden-vector check for the Python mirrors of Rust logic (spec §1.4, M4 review S-14).

`stable_id` (`builder.py`) reimplements `es_core::StableId::from_path` and `_feature_ty`/
`_chunk_ty` reimplement `es-ir/src/learning.rs`'s private `feature()`/`chunk()` — none of them
call into Rust, so nothing catches the two sides drifting apart except a shared golden vector.
`crates/es-core/src/id.rs`'s `from_path_is_stable_and_distinct` test pins the same hex literal
this file checks; the Rust side is canonical (see `python/es/README.md`).

`--env` is a second, unrelated check (packet M8/S4a): the `es_native.Rollout` binding driven
100 control steps by the same scripted control `crates/es-py/tests/rollout.rs` drives, compared
against the golden that test writes. It is the Python half of that oracle — the Rust struct and
the pyclass are pinned to one trace, so the binding cannot quietly step a different runtime than
`es eval run` does. It needs `MuJoCo` (the reference backend runs out of process, spec 17.1) and
prints `SKIP <reason>` when the extension or the interpreter is not there.

Run with the venv interpreter from the repo root:

    PYTHONPATH=python <venv>/python -m es.selfcheck [--env]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


# --- `--env`: the rollout binding against its golden (packet M8/S4a) -------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "tests/golden/rollout/so101_100steps.json"
DOCUMENTS = {
    "task": "tests/fixtures/visible-learning/task.toml",
    "observation": "crates/es-py/tests/fixtures/observation-state.toml",
    "deployment": "tests/fixtures/visible-learning/deployment.toml",
    "scene": "tests/fixtures/mjcf/so101_pick_place.xml",
}
NJ, STEPS, N_ENVS, SEED, RESET_AT = 6, 100, 2, 7, 50


def _scripted(step: int, env: int, joint: int) -> float:
    """Mirrors `scripted()` in `crates/es-py/tests/rollout.rs`. Integers and division by 100,
    so both sides produce the identical f64; the golden is what catches a drift."""
    phase = (step + env * 5 + joint * 3) % 40
    tri = phase if phase < 20 else 40 - phase
    base = (tri - 10) / 100.0
    if step % 10 == 0 and joint == (step // 10) % NJ:
        return base + 0.5
    return base


def _trace(rollout) -> dict:
    steps = []
    for step in range(STEPS):
        if step == RESET_AT:
            rollout.reset([0])
        assert rollout.tick() == step, f"tick {rollout.tick()} at step {step}"
        qpos: list[float] = []
        for env in range(N_ENVS):
            qpos.extend(rollout.qpos(env))
        observations = rollout.observe()
        actions = [_scripted(step, e, j) for e in range(N_ENVS) for j in range(NJ)]
        executed, events, rewards, dones = rollout.act(actions)
        steps.append(
            {
                "qpos": qpos,
                "observations": observations,
                "executed": executed,
                "events": events,
                "rewards": rewards,
                "dones": dones,
            }
        )
    return {"schema": 1, "n_envs": N_ENVS, "seed": SEED, "steps": steps}


def check_env() -> None:
    try:
        from . import es_native
    except ImportError as e:  # the extension was never built into this interpreter
        print(f"SKIP es.selfcheck --env: {e} (run `maturin develop --release --features python`)")
        return
    if not hasattr(es_native, "Rollout"):
        print("SKIP es.selfcheck --env: this es_native has no Rollout (rebuild the extension)")
        return
    missing = [p for p in DOCUMENTS.values() if not (REPO_ROOT / p).is_file()]
    if missing or not GOLDEN.is_file():
        print(f"SKIP es.selfcheck --env: missing {missing or [str(GOLDEN)]}")
        return
    text = {k: (REPO_ROOT / p).read_text(encoding="utf-8") for k, p in DOCUMENTS.items()}
    try:
        rollout = es_native.Rollout(
            text["task"], text["observation"], text["deployment"], text["scene"], SEED, N_ENVS
        )
    except ValueError as e:
        # No `mujoco` in this interpreter is the ordinary case on a machine without the
        # reference backend, and it is a skip, not a failure (spec 1.4).
        print(f"SKIP es.selfcheck --env: {e}")
        return

    got = _trace(rollout)
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for key in ("schema", "n_envs", "seed"):
        assert got[key] == want[key], f"{key}: {got[key]} != {want[key]}"
    assert len(got["steps"]) == len(want["steps"]), "step count"
    for i, (a, b) in enumerate(zip(got["steps"], want["steps"])):
        for field in ("qpos", "observations", "executed", "events", "rewards", "dones"):
            assert a[field] == b[field], (
                f"step {i} {field} differs from {GOLDEN.name}:\n  got  {a[field]}\n  want {b[field]}"
            )
    print(f"RAN es.selfcheck --env: {STEPS} steps x {N_ENVS} envs match {GOLDEN.name}")


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
    # `--env` is the rollout oracle and nothing else; bare `-m es.selfcheck` keeps doing
    # exactly what it did before this packet.
    if "--env" in sys.argv[1:]:
        check_env()
    else:
        main()
