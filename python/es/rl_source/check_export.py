"""Oracle 2 of packet M8/S2c: `source.npz` recomputed in numpy must match JAX.

    python check_export.py --out ~/artifacts/plan-s/s2c/seed0 [--tol 1e-5]

Reads `source.npz` and `oracle-1000.npz` (two 1,000-row obs sets and the actions JAX produced
for them), recomputes the actions in plain numpy -- normalize, swish, linear, tanh -- and
reports the max abs error for each. This is the same arithmetic S2b must reproduce in torch,
so if this file drifts from the exporter the import breaks here first, in numpy, with no
framework to blame. Exit code 1 if the in-distribution error exceeds the tolerance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def swish(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), in the branch form that does not overflow (JAX's is stable too)."""
    pos = np.exp(-np.abs(x, dtype=np.float32), dtype=np.float32)
    sigmoid = np.where(x >= 0, 1.0 / (1.0 + pos), pos / (1.0 + pos)).astype(np.float32)
    return (x * sigmoid).astype(np.float32)


def actions_from_npz(npz, obs: np.ndarray) -> np.ndarray:
    x = ((obs - npz["obs_mean"]) / npz["obs_std"]).astype(np.float32)
    n = sum(1 for k in npz.files if k.startswith("kernel_"))
    for i in range(n):
        x = swish(x @ npz[f"kernel_{i}"] + npz[f"bias_{i}"]).astype(np.float32)
    return np.tanh(x @ npz["mean_kernel"] + npz["mean_bias"]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="numpy re-computation of the exported policy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    source = np.load(out / "source.npz")
    oracle = np.load(out / "oracle-1000.npz")

    errors = {}
    for name in ("", "_scaled"):
        obs, want = oracle["obs" + name], oracle["actions" + name]
        assert obs.dtype == np.float32 and want.dtype == np.float32, obs.dtype
        got = actions_from_npz(source, obs)
        errors[name or "_uniform"] = float(np.abs(got - want).max())
        saturated = float((np.abs(want) > 0.999).mean())
        print(
            f"obs{name} {obs.shape} max_abs_error {errors[name or '_uniform']:.3e} "
            f"(|action| > 0.999 in {saturated:.1%} of rows)"
        )

    # The gate is the in-distribution set. `obs` is the packet's U(-1, 1) draw, which is far
    # outside what the policy ever sees: three observation channels are constant in this task
    # (cube z and two quaternion components), so their normalizer std is brax's 1e-6 floor and
    # a U(-1, 1) sample normalizes to ~1e6. Two float32 implementations of the same matmul
    # cannot agree to 1e-5 after that; its error is recorded, not asserted.
    err = errors["_scaled"]
    print(f"gate: in-distribution max_abs_error {err:.3e} tol {args.tol:.0e}")
    if err > args.tol:
        print("FAIL: numpy and JAX disagree beyond the tolerance")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
