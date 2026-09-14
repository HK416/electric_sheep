#!/usr/bin/env python3
"""Read a dataset this workspace wrote with the *real* `lerobot` package (spec 1.4).

Usage: `lerobot_read_ref.py <dataset-root>`; prints one JSON object on stdout.

A test tool, never on the runtime path (spec 2.5): `es-data` writes the dataset, and this
says what `lerobot.datasets.lerobot_dataset.LeRobotDataset` makes of it -- episode count,
frames per episode, every feature with its dtype and shape, and, iterating *every* frame,
the flattened `observation.state` and `action` values plus each camera's tensor shape and
pixel sum. `crates/es-data/tests/lerobot_oracle.rs` (v2.1) and
`crates/es-data/tests/lerobot_v3.rs` (v3.0) compare that with what `es-data` wrote.

Every field is flat on purpose: the Rust side scans this JSON without a JSON dependency.

Any failure is reported as `{"skip": "<why>"}` so a machine without `lerobot[dataset]` skips
with a reason instead of failing CI. A *refusal* by lerobot is also reported that way, and it
is up to the caller to decide whether a refusal is a finding or a failure.
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
        cameras = sorted(k for k, v in features.items() if v["dtype"] in ("image", "video"))

        # Every frame, not just the two ends.
        states, actions, image_shape, image_sum = [], [], {}, {}
        for i in range(len(ds)):
            item = ds[i]
            if "observation.state" in item:
                states.extend(float(v) for v in item["observation.state"])
            if "action" in item:
                actions.extend(float(v) for v in item["action"])
            for cam in cameras:
                tensor = item[cam]
                image_shape[cam] = [int(d) for d in tensor.shape]
                image_sum[cam] = image_sum.get(cam, 0.0) + float(tensor.sum())

        first = ds[0]["observation.state"]
        last = ds[len(ds) - 1]["observation.state"]
        out = {
            "lerobot_version": version,
            "codebase_version": str(meta._version if hasattr(meta, "_version") else ""),
            "episodes": int(meta.total_episodes),
            "frames": int(meta.total_frames),
            "iterated": len(ds),
            "lengths": lengths,
            "features": features,
            "cameras": cameras,
            "first_state": [float(v) for v in first],
            "last_state": [float(v) for v in last],
            "states_flat": states,
            "actions_flat": actions,
            "image_shape": [d for cam in cameras for d in image_shape.get(cam, [])],
            "image_sum": [image_sum.get(cam, 0.0) for cam in cameras],
            "tasks": sorted({str(ds[i]["task"]) for i in range(len(ds))}),
        }
    except Exception as e:  # noqa: BLE001 - a refusal is a finding, and it is the reason
        print(json.dumps({"skip": f"{type(e).__name__}: {e}"}))
        return 0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
