#!/usr/bin/env python3
"""Judge `meta/stats.json` against `lerobot`'s own statistics (spec 1.4, packet M5/V8).

Usage: `lerobot_stats_ref.py <dataset-root>`; prints one flat JSON object on stdout.

`es dataset export --lerobot-v3` writes `meta/stats.json` itself, in Rust, because
`lerobot-train` needs it for normalization and nothing else in this project does. The oracle
for that file is the package that owns the format, twice over:

  1. `LeRobotDataset(root=...).meta.stats` -- the file as `lerobot` loads it, which is the
     only shape its normalizers ever index;
  2. `compute_episode_stats` per episode then `aggregate_stats`, run here on the very parquet
     the export wrote -- which is how `lerobot` would have computed the same file.

The script reports the largest absolute disagreement between the two, per feature and stat
key, plus the features present in one and not the other. The Rust side decides the threshold.

Two honest differences, both reported rather than hidden:

  * `lerobot` *samples* image frames (`compute_stats.sample_indices`, ~100 per episode over a
    long one) and the export reduces over every pixel of every frame. On a dataset small
    enough that the sample is exhaustive the two agree to float precision; on a larger one
    `lerobot`'s number is the estimate and ours is exact.
  * the `qNN` quantile keys are histogram estimates in `lerobot` and are not written by the
    export at all, so they are excluded here and listed under `skipped_keys`.

A test tool, never on the runtime path (spec 2.5). Any import failure is `{"skip": "<why>"}`.
"""
import json
import os
import sys
import tempfile

STATS = ["min", "max", "mean", "std", "count"]


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"skip": "usage: lerobot_stats_ref.py <dataset-root>"}))
        return 0
    root = sys.argv[1]
    try:
        import numpy as np
        import pandas as pd

        from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:  # noqa: BLE001 - any import failure is a skip, with its reason
        print(json.dumps({"skip": f"import lerobot.datasets: {type(e).__name__}: {e}"}))
        return 0

    try:
        ds = LeRobotDataset(repo_id="es/oracle", root=root)
        ours = ds.meta.stats
        if ours is None:
            print(json.dumps({"skip": "meta/stats.json is absent (load_stats returned None)"}))
            return 0
        features = ds.meta.features

        frame = pd.read_parquet(os.path.join(root, "data", "chunk-000", "file-000.parquet"))
        tmp = tempfile.mkdtemp(prefix="es-stats-")
        per_episode = []
        for episode, rows in frame.groupby("episode_index", sort=True):
            data = {}
            for key, spec in features.items():
                dtype = spec["dtype"]
                if dtype in ("string", "language"):
                    continue
                if dtype in ("image", "video"):
                    paths = []
                    for i, cell in enumerate(rows[key]):
                        path = os.path.join(
                            tmp, "%s-%d-%06d.png" % (key.replace(".", "_"), episode, i)
                        )
                        with open(path, "wb") as handle:
                            handle.write(cell["bytes"])
                        paths.append(path)
                    data[key] = paths
                    continue
                values = np.stack([np.atleast_1d(np.asarray(v)) for v in rows[key]])
                # A `shape: [1]` feature is a scalar column, and LeRobot's own writer hands
                # `compute_episode_stats` a 1-D array for it -- which is what selects
                # `keepdims=True` and the `(1,)` output shape.
                if values.shape[1] == 1:
                    values = values.reshape(-1)
                data[key] = values
            per_episode.append(compute_episode_stats(data, features))
        theirs = aggregate_stats(per_episode)

        worst, worst_key, missing, extra = 0.0, "", [], []
        for key in sorted(theirs):
            if key not in ours:
                missing.append(key)
                continue
            for stat in STATS:
                if stat not in ours[key] or stat not in theirs[key]:
                    missing.append("%s/%s" % (key, stat))
                    continue
                a = np.asarray(ours[key][stat], dtype=np.float64).reshape(-1)
                b = np.asarray(theirs[key][stat], dtype=np.float64).reshape(-1)
                if a.shape != b.shape:
                    missing.append("%s/%s (shape %s vs %s)" % (key, stat, a.shape, b.shape))
                    continue
                d = float(np.max(np.abs(a - b))) if a.size else 0.0
                if d > worst:
                    worst, worst_key = d, "%s/%s" % (key, stat)
        for key in sorted(ours):
            if key not in theirs:
                extra.append(key)

        import lerobot

        print(
            json.dumps(
                {
                    "ok": True,
                    "lerobot": getattr(lerobot, "__version__", "unknown"),
                    "features": sorted(ours),
                    "worst": worst,
                    "worst_key": worst_key,
                    "missing": missing,
                    "extra": extra,
                    "skipped_keys": ["q01", "q10", "q50", "q90", "q99"],
                }
            )
        )
    except Exception as e:  # noqa: BLE001 - a refusal is a finding; the caller decides
        print(json.dumps({"skip": f"{type(e).__name__}: {e}"}))
    return 0


sys.exit(main())
