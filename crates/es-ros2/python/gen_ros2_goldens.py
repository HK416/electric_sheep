#!/usr/bin/env python3
"""Generate/check the `es-ros2` W1a goldens from the reference oracles (spec 1.4).

Two independent oracles, two modes:

    gen_ros2_goldens.py <out_dir>
        rosbags 0.11.5 (`Stores.ROS2_KILTED`) + PyPI xxhash. Writes `<out_dir>/cdr/*.bin`,
        `<out_dir>/cdr/*.json`, `<out_dir>/rihs01.json`, `<out_dir>/gid.json`. Needs `rosbags`
        and `xxhash` importable (`ES_PYTHON` in `docs/design/ros2-boundary.md` section 8).

    gen_ros2_goldens.py --check-ros <golden_dir>
        A live RoboStack ROS 2 Kilted install (`rmw-zenoh-cpp`, `ros-base`). Cross-checks
        `<golden_dir>/rihs01.json` against `share/<pkg>/msg/<T>.json`'s `type_hashes[0]
        .hash_string`, and `<golden_dir>/cdr/*.bin` against `rclpy.serialization
        .serialize_message`. Run through `scripts/ros2-env.sh <prefix> python
        python/gen_ros2_goldens.py --check-ros tests/golden/ros2` with `RMW_IMPLEMENTATION=
        rmw_zenoh_cpp`, per `docs/design/ros2-boundary.md` section 8. Never used to produce a
        golden — this crate's goldens come only from rosbags/xxhash (spec 1.4); this mode is a
        second, independent oracle that either agrees or documents where it does not
        (`docs/api-notes/ros2-cdr.md`'s "Live ROS 2 byte capture" row).

Neither mode imports `es_ros2` or anything from this crate: the goldens must come from the
reference, never from our own encoder (packet `forbidden`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROSBAGS_VERSION = "0.11.5"
XXHASH_VERSION = "4.0.1"

# The exact vectors in docs/api-notes/ros2-cdr.md's "Golden vectors" table. `fields` is the
# single source of truth for both the rosbags message (this file) and the rclpy message
# (`--check-ros`): whatever is written into the `.json` sidecar is what gets rebuilt on the ROS
# 2 side later, so the two oracles are provably testing the same input.
VECTORS: list[dict[str, Any]] = [
    {
        "name": "string_hello",
        "typename": "std_msgs/msg/String",
        "little_endian": True,
        "check_ros": True,
        "fields": {"data": "hello"},
    },
    {
        "name": "string_empty",
        "typename": "std_msgs/msg/String",
        "little_endian": True,
        "check_ros": True,
        "fields": {"data": ""},
    },
    {
        "name": "string_hello_be",
        "typename": "std_msgs/msg/String",
        "little_endian": False,
        # rclpy always serializes native-endian (effectively LE on every RoboStack target this
        # crate cares about); a big-endian golden has nothing on the ROS 2 side to compare to.
        "check_ros": False,
        "fields": {"data": "hello"},
    },
    {
        "name": "joint_state",
        "typename": "sensor_msgs/msg/JointState",
        "little_endian": True,
        "check_ros": True,
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "base"},
            "name": ["j1", "j2"],
            "position": [0.5, -1.0],
            "velocity": [],
            "effort": [],
        },
    },
    {
        "name": "image_rgb8_2x1",
        "typename": "sensor_msgs/msg/Image",
        "little_endian": True,
        "check_ros": True,
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": 1,
            "width": 2,
            "encoding": "rgb8",
            "is_bigendian": 0,
            "step": 6,
            "data": [1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "camera_info_plumb_bob",
        "typename": "sensor_msgs/msg/CameraInfo",
        "little_endian": True,
        "check_ros": True,
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": 480,
            "width": 640,
            "distortion_model": "plumb_bob",
            "d": [0.1, 0.01, 0.0, 0.0, 0.0],
            "k": [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
            "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "p": [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "binning_x": 0,
            "binning_y": 0,
            "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
        },
    },
    {
        "name": "float64_multi_array",
        "typename": "std_msgs/msg/Float64MultiArray",
        "little_endian": True,
        "check_ros": True,
        "fields": {
            "layout": {"dim": [], "data_offset": 0},
            "data": [0.25, -0.5, 1.0],
        },
    },
]

# The 11 REP-2016 type hashes `docs/api-notes/ros2-cdr.md` pins (5 dispatchable message types
# plus the nested/only-hash-recorded ones).
RIHS01_TYPES = [
    "std_msgs/msg/String",
    "std_msgs/msg/Header",
    "builtin_interfaces/msg/Time",
    "std_msgs/msg/Float64MultiArray",
    "std_msgs/msg/MultiArrayLayout",
    "std_msgs/msg/MultiArrayDimension",
    "sensor_msgs/msg/JointState",
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CameraInfo",
    "sensor_msgs/msg/RegionOfInterest",
    "trajectory_msgs/msg/JointTrajectory",
]

# The two rmw_zenoh `docs/design.md` liveliness token examples GID is derived from
# (`docs/api-notes/rmw-zenoh.md` "Liveliness tokens", verbatim).
GID_TOKENS = [
    "@ros2_lv/0/aac3178e146ba6f1fc6e6a4085e77f21/0/0/NN/%/%/listener",
    "@ros2_lv/0/8b20917502ee955ac4476e0266340d5c/0/10/MP/%/%/talker/%chatter/"
    "std_msgs::msg::dds_::String_/"
    "RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18/::,7:,:,:,,",
]


def _dump_json(obj: Any, path: Path) -> None:
    # `newline="\n"` pins the on-disk line ending to LF on every platform (default text-mode
    # writing translates "\n" to the platform's line ending, which would make a Windows-
    # generated golden byte-differ from a Linux-generated one for no semantic reason).
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


# --- Mode 1: rosbags + xxhash -> goldens -------------------------------------------------


def _build_rosbags_message(ts: Any, name: str, typename: str, f: dict[str, Any]) -> Any:
    """One explicit builder per vector name (the message subset is small and fixed; a
    reflection-based generic builder would hide, not simplify, the numpy-dtype requirements
    rosbags' generated `__init__`s have for numeric sequences)."""
    import numpy as np

    if name in ("string_hello", "string_empty", "string_hello_be"):
        String = ts.types[typename]
        return String(data=f["data"])

    if name == "joint_state":
        Time = ts.types["builtin_interfaces/msg/Time"]
        Header = ts.types["std_msgs/msg/Header"]
        JointState = ts.types[typename]
        header = Header(stamp=Time(**f["header"]["stamp"]), frame_id=f["header"]["frame_id"])
        return JointState(
            header=header,
            name=list(f["name"]),
            position=np.array(f["position"], dtype=np.float64),
            velocity=np.array(f["velocity"], dtype=np.float64),
            effort=np.array(f["effort"], dtype=np.float64),
        )

    if name == "image_rgb8_2x1":
        Time = ts.types["builtin_interfaces/msg/Time"]
        Header = ts.types["std_msgs/msg/Header"]
        Image = ts.types[typename]
        header = Header(stamp=Time(**f["header"]["stamp"]), frame_id=f["header"]["frame_id"])
        return Image(
            header=header,
            height=f["height"],
            width=f["width"],
            encoding=f["encoding"],
            is_bigendian=f["is_bigendian"],
            step=f["step"],
            data=np.array(f["data"], dtype=np.uint8),
        )

    if name == "camera_info_plumb_bob":
        Time = ts.types["builtin_interfaces/msg/Time"]
        Header = ts.types["std_msgs/msg/Header"]
        CameraInfo = ts.types[typename]
        RegionOfInterest = ts.types["sensor_msgs/msg/RegionOfInterest"]
        header = Header(stamp=Time(**f["header"]["stamp"]), frame_id=f["header"]["frame_id"])
        roi_f = f["roi"]
        roi = RegionOfInterest(
            x_offset=roi_f["x_offset"],
            y_offset=roi_f["y_offset"],
            height=roi_f["height"],
            width=roi_f["width"],
            do_rectify=roi_f["do_rectify"],
        )
        return CameraInfo(
            header=header,
            height=f["height"],
            width=f["width"],
            distortion_model=f["distortion_model"],
            d=np.array(f["d"], dtype=np.float64),
            k=np.array(f["k"], dtype=np.float64),
            r=np.array(f["r"], dtype=np.float64),
            p=np.array(f["p"], dtype=np.float64),
            binning_x=f["binning_x"],
            binning_y=f["binning_y"],
            roi=roi,
        )

    if name == "float64_multi_array":
        MultiArrayLayout = ts.types["std_msgs/msg/MultiArrayLayout"]
        Float64MultiArray = ts.types[typename]
        layout = MultiArrayLayout(dim=[], data_offset=f["layout"]["data_offset"])
        return Float64MultiArray(layout=layout, data=np.array(f["data"], dtype=np.float64))

    raise ValueError(f"unknown vector {name!r}")


def generate(out_dir: Path) -> None:
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_KILTED)

    cdr_dir = out_dir / "cdr"
    cdr_dir.mkdir(parents=True, exist_ok=True)

    for spec in VECTORS:
        msg = _build_rosbags_message(ts, spec["name"], spec["typename"], spec["fields"])
        raw = bytes(ts.serialize_cdr(msg, spec["typename"], little_endian=spec["little_endian"]))
        (cdr_dir / f"{spec['name']}.bin").write_bytes(raw)
        _dump_json(
            {
                "typename": spec["typename"],
                "little_endian": spec["little_endian"],
                "fields": spec["fields"],
                "rosbags_version": ROSBAGS_VERSION,
            },
            cdr_dir / f"{spec['name']}.json",
        )

    rihs01 = {t: ts.hash_rihs01(t) for t in RIHS01_TYPES}
    _dump_json(rihs01, out_dir / "rihs01.json")

    import xxhash

    gid_entries = []
    for token in GID_TOKENS:
        h = xxhash.xxh3_128_intdigest(token.encode())
        low64 = h & 0xFFFF_FFFF_FFFF_FFFF
        high64 = h >> 64
        gid_hex = (low64.to_bytes(8, "little") + high64.to_bytes(8, "little")).hex()
        gid_entries.append({"key_expr": token, "gid_hex": gid_hex})
    _dump_json(
        {"tokens": gid_entries, "xxhash_version": XXHASH_VERSION},
        out_dir / "gid.json",
    )


# --- Mode 2: --check-ros -> cross-check against a live RoboStack install -----------------


def _diff_report(want: bytes, got: bytes) -> list[str]:
    lines: list[str] = []
    if want[:2] != got[:2]:
        lines.append(f"  header id: golden={want[:2].hex()} rclpy={got[:2].hex()}")
    if len(want) >= 4 and len(got) >= 4 and want[2:4] != got[2:4]:
        lines.append(f"  options bytes: golden={want[2:4].hex()} rclpy={got[2:4].hex()}")
    if len(got) > len(want):
        lines.append(f"  rclpy is {len(got) - len(want)} byte(s) longer than the golden")
    elif len(got) < len(want):
        lines.append(f"  rclpy is {len(want) - len(got)} byte(s) shorter than the golden")
    n = min(len(want), len(got))
    i = 4
    while i < n:
        if want[i] != got[i]:
            j = i
            while j < n and want[j] != got[j]:
                j += 1
            lines.append(f"  offset {i}..{j}: golden={want[i:j].hex()} rclpy={got[i:j].hex()}")
            i = j
        else:
            i += 1
    if len(got) > len(want):
        lines.append(f"  trailing (rclpy only) offset {len(want)}..{len(got)}: {got[len(want):].hex()}")
    return lines


def _build_rclpy_message(name: str, typename: str, f: dict[str, Any]) -> Any:
    import array

    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import CameraInfo, Image, JointState, RegionOfInterest
    from std_msgs.msg import Float64MultiArray, Header, MultiArrayLayout, String

    if name in ("string_hello", "string_empty"):
        return String(data=f["data"])

    if name == "joint_state":
        header = Header(
            stamp=Time(sec=f["header"]["stamp"]["sec"], nanosec=f["header"]["stamp"]["nanosec"]),
            frame_id=f["header"]["frame_id"],
        )
        return JointState(
            header=header,
            name=list(f["name"]),
            position=array.array("d", f["position"]),
            velocity=array.array("d", f["velocity"]),
            effort=array.array("d", f["effort"]),
        )

    if name == "image_rgb8_2x1":
        header = Header(
            stamp=Time(sec=f["header"]["stamp"]["sec"], nanosec=f["header"]["stamp"]["nanosec"]),
            frame_id=f["header"]["frame_id"],
        )
        return Image(
            header=header,
            height=f["height"],
            width=f["width"],
            encoding=f["encoding"],
            is_bigendian=f["is_bigendian"],
            step=f["step"],
            data=array.array("B", f["data"]),
        )

    if name == "camera_info_plumb_bob":
        header = Header(
            stamp=Time(sec=f["header"]["stamp"]["sec"], nanosec=f["header"]["stamp"]["nanosec"]),
            frame_id=f["header"]["frame_id"],
        )
        roi_f = f["roi"]
        roi = RegionOfInterest(
            x_offset=roi_f["x_offset"],
            y_offset=roi_f["y_offset"],
            height=roi_f["height"],
            width=roi_f["width"],
            do_rectify=roi_f["do_rectify"],
        )
        return CameraInfo(
            header=header,
            height=f["height"],
            width=f["width"],
            distortion_model=f["distortion_model"],
            d=array.array("d", f["d"]),
            k=array.array("d", f["k"]),
            r=array.array("d", f["r"]),
            p=array.array("d", f["p"]),
            binning_x=f["binning_x"],
            binning_y=f["binning_y"],
            roi=roi,
        )

    if name == "float64_multi_array":
        layout = MultiArrayLayout(dim=[], data_offset=f["layout"]["data_offset"])
        return Float64MultiArray(layout=layout, data=array.array("d", f["data"]))

    raise ValueError(f"unknown vector {name!r}")


def _check_rihs01(golden_dir: Path) -> tuple[bool, list[str]]:
    import os

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return False, ["CONDA_PREFIX is not set (expected from scripts/ros2-env.sh)"]

    want = json.loads((golden_dir / "rihs01.json").read_text())
    share = Path(conda_prefix) / "share"
    ok = True
    lines = []
    for typename, want_hash in sorted(want.items()):
        pkg, _, rest = typename.partition("/msg/")
        json_path = share / pkg / "msg" / f"{rest}.json"
        try:
            data = json.loads(json_path.read_text())
            got_hash = data["type_hashes"][0]["hash_string"]
        except OSError as e:
            ok = False
            lines.append(f"{typename}: cannot read {json_path}: {e}")
            continue
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            ok = False
            lines.append(f"{typename}: cannot parse {json_path}: {e}")
            continue
        if got_hash == want_hash:
            lines.append(f"{typename}: match")
        else:
            ok = False
            lines.append(f"{typename}: MISMATCH golden={want_hash} robostack={got_hash}")
    return ok, lines


def _check_cdr_bytes(golden_dir: Path) -> tuple[bool, list[str]]:
    import rclpy.serialization

    ok = True
    lines = []
    for spec in VECTORS:
        if not spec["check_ros"]:
            continue
        name = spec["name"]
        want = (golden_dir / "cdr" / f"{name}.bin").read_bytes()
        msg = _build_rclpy_message(name, spec["typename"], spec["fields"])
        got = bytes(rclpy.serialization.serialize_message(msg))
        if got == want:
            lines.append(f"{name}: match ({len(got)} bytes)")
            continue
        ok = False
        lines.append(f"{name}: MISMATCH (golden {len(want)} bytes, rclpy {len(got)} bytes)")
        lines.extend(_diff_report(want, got))
    return ok, lines


def check_ros(golden_dir: Path) -> int:
    hash_ok, hash_lines = _check_rihs01(golden_dir)
    print("RIHS01 cross-check (RoboStack share/<pkg>/msg/<T>.json vs rihs01.json):")
    for line in hash_lines:
        print(f"  {line}")
    print(f"RIHS01: {'PASS' if hash_ok else 'FAIL'}")
    print()

    cdr_ok, cdr_lines = _check_cdr_bytes(golden_dir)
    print("rclpy CDR byte cross-check (rclpy.serialization.serialize_message vs cdr/*.bin):")
    for line in cdr_lines:
        print(line)
    print(f"CDR: {'PASS' if cdr_ok else 'FAIL'}")

    return 0 if (hash_ok and cdr_ok) else 1


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "--check-ros":
        return check_ros(Path(argv[1]))
    if len(argv) == 1:
        generate(Path(argv[0]))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
