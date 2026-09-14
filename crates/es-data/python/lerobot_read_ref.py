#!/usr/bin/env python3
"""Read a dataset this workspace wrote with the *real* `lerobot` package (spec 1.4).

Usage: `lerobot_read_ref.py <dataset-root>`; prints one JSON object on stdout.

A test tool, never on the runtime path (spec 2.5): `es-data` writes the dataset, and this
says what `lerobot.datasets.lerobot_dataset.LeRobotDataset` makes of it -- episode count,
frames per episode, every feature with its dtype and shape, and the first and last
`observation.state` row. `crates/es-data/tests/lerobot_oracle.rs` compares that with what
`es-data`'s own reader says. Any failure is reported as `{"skip": "<why>"}` so a machine
without `lerobot[dataset]` skips with a reason instead of failing CI.
"""
import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"skip": "usage: lerobot_read_ref.py <dataset-root>"}))
        return 0
    root = sys.argv[1]
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:  # noqa: BLE001 - any import failure is a skip, with its reason
        print(json.dumps({"skip": f"import lerobot.datasets: {type(e).__name__}: {e}"}))
        return 0

    try:
        import lerobot

        version = getattr(lerobot, "__version__", "unknown")
        ds = LeRobotDataset(repo_id="es/oracle", root=root)
        meta = ds.meta
        features = {}
        for name, spec in meta.features.items():
            shape = spec.get("shape", [])
            features[name] = {
                "dtype": spec.get("dtype", "?"),
                "shape": [int(d) for d in shape],
            }
        lengths = [int(meta.episodes[i]["length"]) for i in range(meta.total_episodes)]
        first = ds[0]["observation.state"]
        last = ds[len(ds) - 1]["observation.state"]
        out = {
            "lerobot_version": version,
            "codebase_version": str(meta._version if hasattr(meta, "_version") else ""),
            "episodes": int(meta.total_episodes),
            "frames": int(meta.total_frames),
            "lengths": lengths,
            "features": features,
            "first_state": [float(v) for v in first],
            "last_state": [float(v) for v in last],
        }
    except Exception as e:  # noqa: BLE001 - a refusal is a finding, and it is the reason
        print(json.dumps({"skip": f"{type(e).__name__}: {e}"}))
        return 0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
