"""Merge the demo's two state ports into the one `observation.state` a LeRobot policy takes
(packet `docs/packets/M5/V19-external-act-current-data.md`, design note section 7.27).

`tests/fixtures/visible-learning/observation.toml` declares two state ports -- the arm's six
joint angles and V7a's simulator-privileged `sim_cube_pose` (the cube free joint's `qpos[6..13]`)
-- because the IR-graph policy of V13/V15/V17 has a `Fusion{Concat}` node that takes both.
`lerobot.utils.feature_utils.dataset_to_policy_features` gives a policy exactly **one**
`observation.state` feature, so an external ACT trained by `lerobot-train` has one place to put
them: concatenated, in the order the recorded `observation.state` row (`qpos || qvel`) already
holds them. `es dataset export --lerobot-v3 --state-dim 13` keeps exactly `qpos[0..13]`, which
is those six angles followed by that cube pose, so the vector the network is trained on and the
vector this graph produces at inference are the same thing in the same order.

So this writes observation-v8.toml's graph (spec 7; packet M5/V8 explains every other choice in
it, above all why the state `Normalize` is the identity `Range{0..1}`: ACT carries its own
`MEAN_STD` statistics in the checkpoint and applies them as the first operation of its forward
pass, and a second affine map here would have to be applied to the exported parquet as well)
plus observation.toml's cube branch, joined by an `ObservationNode::Concat` on axis 0.

Three things the merge has to get right, and each is checked below:

*   the join order is joints-then-cube, matching `qpos`;
*   the cube branch's `Normalize` becomes the identity too, for V8's reason -- observation.toml
    normalizes it to +-0.3 m for a policy whose statistics are the IR's, and the exported parquet
    carries metres;
*   the `Concat` ports are `Frame::Policy`. An arm-frame `[6]` and a world-frame `[7]` have no
    common frame, and `Frame::Policy` is the one the IR already defines as compatible with any
    other (`es_ir_types::Frame::compatible`) -- which is what a feature vector handed to a
    network is.

Usage:

    python make_v19_observation.py <observation.toml> <observation-v8.toml> <out.toml>

Prints one JSON line: {"state_dim": 13, "nodes": n, "edges": m}.
"""

from __future__ import annotations

import json
import sys
import tomllib

import toml

JOINTS, CUBE, CONCAT = "3", "5", "7"


def norm01() -> dict:
    """`Unit::Normalized{0..1}`. A fresh dict each call: `toml`'s encoder reports a shared one
    as a circular reference."""
    return {"Normalized": {"lo": 0.0, "hi": 1.0}}


def port(dim: int) -> dict:
    """One `Concat` port: a policy-frame feature vector on the control tick."""
    return {"elem": "F32", "shape": [dim], "time": "Tick", "frame": "Policy", "unit": norm01()}


def merge(full: dict, v8: dict) -> dict:
    """observation-v8's graph + observation.toml's cube branch, joined on axis 0."""
    g8, gf = v8["body"]["graph"], full["body"]["graph"]
    nodes, edges = g8["nodes"], g8["edges"]

    # observation.toml's nodes 5 (StateInput, the cube's free joint) and 6 (its Normalize),
    # carried over by id so the two documents stay readable side by side.
    cube_in = gf["nodes"][CUBE]
    cube_norm = gf["nodes"]["6"]
    assert "StateInput" in cube_in and cube_in["StateInput"]["io"]["output"]["shape"] == [7]
    # The identity, for the reason in the module docstring.
    cube_norm["Normalize"]["stats"] = {"Range": {"lo": 0.0, "hi": 1.0}}
    cube_norm["Normalize"]["io"]["output"]["unit"] = norm01()
    nodes[CUBE], nodes["6"] = cube_in, cube_norm

    joints_out = nodes[JOINTS]["Normalize"]["io"]["output"]
    assert joints_out["shape"] == [6] and joints_out["unit"] == norm01(), joints_out

    nodes[CONCAT] = {
        "Concat": {
            "axis": 0,
            "time_align": "Hold",
            "io": {"inputs": [port(6), port(7)], "output": port(13)},
        }
    }
    edges += [
        {"from": {"node": 5, "port": "out"}, "to": {"node": 6, "port": "in0"}},
        {"from": {"node": 3, "port": "out"}, "to": {"node": 7, "port": "in0"}},
        {"from": {"node": 6, "port": "out"}, "to": {"node": 7, "port": "in1"}},
    ]
    v8["body"]["outputs"]["observation_state"] = {
        "port": {"node": 7, "port": "out"},
        "ty": port(13),
    }
    return v8


def main() -> int:
    full, v8, out = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(full, "rb") as f:
        a = tomllib.load(f)
    with open(v8, "rb") as f:
        b = tomllib.load(f)
    doc = merge(a, b)
    with open(out, "w", encoding="utf-8") as f:
        f.write(toml.dumps(doc))

    g = doc["body"]["graph"]
    state = doc["body"]["outputs"]["observation_state"]
    # in0 is the arm, in1 is the cube: the order `qpos` and `--state-dim 13` already have.
    ins = g["nodes"][CONCAT]["Concat"]["io"]["inputs"]
    assert [t["shape"][0] for t in ins] == [6, 7], ins
    assert state["ty"]["shape"] == [13], state
    assert {"node": 3, "port": "out"} in [e["from"] for e in g["edges"]]
    assert {"node": 6, "port": "out"} in [e["from"] for e in g["edges"]]
    print(json.dumps({"state_dim": 13, "nodes": len(g["nodes"]), "edges": len(g["edges"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
