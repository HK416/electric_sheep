#!/usr/bin/env python3
"""Generate/check the `es-ros2` W1c camera goldens from the reference oracles (spec 1.4).

Two independent oracles, two modes:

    gen_camera_goldens.py <out_dir>
        rosbags 0.11.5 (`Stores.ROS2_KILTED`) + opencv-python-headless 5.0.0.93. Writes
        `<out_dir>/cdr/*.bin` + `*.json` for every `CameraInfo` / `Image` fixture, and
        `<out_dir>/yuv/{uyvy,yuyv}_8x2.rgb` from `cv2.cvtColor(src, cv2.COLOR_YUV2RGB_UYVY)`
        / `COLOR_YUV2RGB_YUY2`. Needs `rosbags` and `cv2` importable (`ES_PYTHON`,
        `docs/design/ros2-boundary.md` section 8).

    gen_camera_goldens.py --ros <out_dir>
        A live RoboStack ROS 2 Kilted install with `image_geometry` and `cv_bridge`. Writes
        `<out_dir>/intrinsics.json` (`PinholeCameraModel.from_camera_info(msg)` ->
        `intrinsic_matrix()`, `projection_matrix()`, i.e. the reference for ROI/binning under
        INV-14) and `<out_dir>/cv_bridge.json` (`CvBridge().imgmsg_to_cv2(msg, 'rgb8')` for the
        two YUV fixtures), and fails if `cv_bridge` and `cv2` disagree on a single byte, naming
        the encoding. Run through
        `scripts/ros2-env.sh <prefix> python python/gen_camera_goldens.py --ros <out_dir>`.

Neither mode imports `es_ros2` or anything from this crate, and no expected value is computed
by hand here: intrinsics come from image_geometry, YUV from OpenCV, bytes from rosbags (packet
`forbidden`).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

ROSBAGS_VERSION = "0.11.5"
OPENCV_VERSION = "5.0.0.93"

IDENTITY_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
ZERO_ROI = {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False}

# `docs/packets/M3/W1c-camera-ingest.md` "Fixtures", verbatim. `k`/`p` are row-major; `roi` is
# (x_offset, y_offset, height, width) as the `.msg` orders them.
_A_K = [600.0, 0.0, 319.5, 0.0, 610.0, 239.5, 0.0, 0.0, 1.0]
_A_P = [600.0, 0.0, 319.5, 0.0, 0.0, 610.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
_C_K = [900.0, 0.0, 639.5, 0.0, 900.0, 359.5, 0.0, 0.0, 1.0]
_C_P = [900.0, 0.0, 639.5, 0.0, 0.0, 900.0, 359.5, 0.0, 0.0, 0.0, 1.0, 0.0]
_E_K = [285.7, 0.0, 423.5, 0.0, 286.1, 399.5, 0.0, 0.0, 1.0]
_E_P = [285.7, 0.0, 423.5, 0.0, 0.0, 286.1, 399.5, 0.0, 0.0, 0.0, 1.0, 0.0]

CAMERA_INFO: list[dict[str, Any]] = [
    {
        "name": "info_a",
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": 480,
            "width": 640,
            "distortion_model": "plumb_bob",
            "d": [-0.1, 0.01, 0.001, -0.002, 0.0005],
            "k": _A_K,
            "r": IDENTITY_R,
            "p": _A_P,
            "binning_x": 0,
            "binning_y": 0,
            "roi": ZERO_ROI,
        },
    },
    {
        "name": "info_b",
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": 480,
            "width": 640,
            "distortion_model": "plumb_bob",
            "d": [-0.1, 0.01, 0.001, -0.002, 0.0005],
            "k": _A_K,
            "r": IDENTITY_R,
            "p": _A_P,
            "binning_x": 2,
            "binning_y": 2,
            "roi": {
                "x_offset": 80,
                "y_offset": 60,
                "height": 360,
                "width": 480,
                "do_rectify": False,
            },
        },
    },
    {
        "name": "info_c",
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": 720,
            "width": 1280,
            "distortion_model": "rational_polynomial",
            "d": [-0.05, 0.02, 0.0003, -0.0007, 0.001, 0.0, 0.0, 0.0],
            "k": _C_K,
            "r": IDENTITY_R,
            "p": _C_P,
            "binning_x": 0,
            "binning_y": 0,
            "roi": ZERO_ROI,
        },
    },
    {
        "name": "info_d",
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": 720,
            "width": 1280,
            "distortion_model": "rational_polynomial",
            "d": [-0.05, 0.02, 0.0003, -0.0007, 0.001, 0.01, 0.0, 0.0],
            "k": _C_K,
            "r": IDENTITY_R,
            "p": _C_P,
            "binning_x": 0,
            "binning_y": 0,
            "roi": ZERO_ROI,
        },
    },
    {
        "name": "info_e",
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": 800,
            "width": 848,
            "distortion_model": "equidistant",
            "d": [0.1, -0.02, 0.003, -0.0004],
            "k": _E_K,
            "r": IDENTITY_R,
            "p": _E_P,
            "binning_x": 0,
            "binning_y": 0,
            "roi": ZERO_ROI,
        },
    },
    {
        "name": "info_f",
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": 480,
            "width": 640,
            "distortion_model": "plumb_bob",
            "d": [-0.1, 0.01, 0.001, -0.002, 0.0005],
            "k": _A_K,
            "r": IDENTITY_R,
            # Rectified stream: p != k, so ingest must read p and drop the distortion.
            "p": [590.0, 0.0, 318.0, 0.0, 0.0, 600.0, 242.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "binning_x": 0,
            "binning_y": 0,
            "roi": ZERO_ROI,
        },
    },
]

# 3x2 payloads: a distinct value per byte, so a channel swap or a dropped pad row shows up.
_RGB_3X2 = list(range(1, 19))
_RGBA_3X2 = list(range(1, 25))
_MONO16_VALUES = [0, 1, 256, 4095, 65535, 12345]
_F32_VALUES = [0.0, 0.001, 0.5, 1.25, 3.5, 10.0]

# (u, y0, v, y1) per pixel pair: Y at both ends of the range and U/V at 0 / 128 / 255. The same
# quads build both the UYVY and the YUYV fixture, so the two must decode to identical RGB.
_YUV_QUADS = [
    (0, 0, 0, 16),
    (255, 235, 255, 255),
    (128, 128, 128, 64),
    (0, 255, 255, 0),
    (255, 0, 0, 255),
    (128, 16, 235, 128),
    (64, 100, 192, 200),
    (0, 128, 255, 255),
]
_UYVY_8X2 = [b for (u, y0, v, y1) in _YUV_QUADS for b in (u, y0, v, y1)]
_YUYV_8X2 = [b for (u, y0, v, y1) in _YUV_QUADS for b in (y0, u, y1, v)]


def _image(name: str, width: int, height: int, encoding: str, step: int, data: list[int],
           is_bigendian: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "fields": {
            "header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "cam"},
            "height": height,
            "width": width,
            "encoding": encoding,
            "is_bigendian": is_bigendian,
            "step": step,
            "data": data,
        },
    }


IMAGES: list[dict[str, Any]] = [
    _image("image_rgb8", 3, 2, "rgb8", 9, _RGB_3X2),
    _image("image_bgr8", 3, 2, "bgr8", 9, _RGB_3X2),
    _image("image_rgba8", 3, 2, "rgba8", 12, _RGBA_3X2),
    _image("image_bgra8", 3, 2, "bgra8", 12, _RGBA_3X2),
    _image("image_mono8", 3, 2, "mono8", 3, list(range(1, 7))),
    _image("image_mono16_le", 3, 2, "mono16", 6,
           list(struct.pack("<6H", *_MONO16_VALUES))),
    _image("image_mono16_be", 3, 2, "mono16", 6,
           list(struct.pack(">6H", *_MONO16_VALUES)), is_bigendian=1),
    _image("image_16uc1", 3, 2, "16UC1", 6, list(struct.pack("<6H", *_MONO16_VALUES))),
    _image("image_32fc1", 3, 2, "32FC1", 12, list(struct.pack("<6f", *_F32_VALUES))),
    # step = 3w + 2: two padding bytes per row that ingest must strip.
    _image("image_rgb8_padded", 3, 2, "rgb8", 11,
           _RGB_3X2[0:9] + [0xAA, 0xBB] + _RGB_3X2[9:18] + [0xCC, 0xDD]),
    _image("image_uyvy", 8, 2, "uyvy", 16, _UYVY_8X2),
    _image("image_yuyv", 8, 2, "yuyv", 16, _YUYV_8X2),
]

# (fixture name, OpenCV conversion code, golden file under `<out_dir>/yuv/`, the encoding
# string cv_bridge is given in `--ros` mode). RoboStack Kilted's `ros-kilted-cv-bridge 4.1.0`
# only recognizes the deprecated `yuv422` / `yuv422_yuy2` spellings -- `encoding_to_cvtype2`
# raises `Unrecognized image encoding [uyvy]` for the modern ones -- and they are the same
# layout and the same conversion (`docs/design/ros2-boundary.md` section 6.3), so the
# cross-check feeds cv_bridge the name it accepts.
YUV_FIXTURES = [
    ("image_uyvy", "COLOR_YUV2RGB_UYVY", "uyvy_8x2.rgb", "yuv422"),
    ("image_yuyv", "COLOR_YUV2RGB_YUY2", "yuyv_8x2.rgb", "yuv422_yuy2"),
]


def _dump_json(obj: Any, path: Path) -> None:
    # `newline="\n"`: pin the on-disk line ending to LF on every platform.
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


# --- Mode 1: rosbags + OpenCV -> goldens -------------------------------------------------


def _rosbags_camera_info(ts: Any, f: dict[str, Any]) -> Any:
    import numpy as np

    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    RegionOfInterest = ts.types["sensor_msgs/msg/RegionOfInterest"]
    CameraInfo = ts.types["sensor_msgs/msg/CameraInfo"]
    return CameraInfo(
        header=Header(stamp=Time(**f["header"]["stamp"]), frame_id=f["header"]["frame_id"]),
        height=f["height"],
        width=f["width"],
        distortion_model=f["distortion_model"],
        d=np.array(f["d"], dtype=np.float64),
        k=np.array(f["k"], dtype=np.float64),
        r=np.array(f["r"], dtype=np.float64),
        p=np.array(f["p"], dtype=np.float64),
        binning_x=f["binning_x"],
        binning_y=f["binning_y"],
        roi=RegionOfInterest(**f["roi"]),
    )


def _rosbags_image(ts: Any, f: dict[str, Any]) -> Any:
    import numpy as np

    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    Image = ts.types["sensor_msgs/msg/Image"]
    return Image(
        header=Header(stamp=Time(**f["header"]["stamp"]), frame_id=f["header"]["frame_id"]),
        height=f["height"],
        width=f["width"],
        encoding=f["encoding"],
        is_bigendian=f["is_bigendian"],
        step=f["step"],
        data=np.array(f["data"], dtype=np.uint8),
    )


def generate(out_dir: Path) -> None:
    import numpy as np
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_KILTED)
    cdr_dir = out_dir / "cdr"
    cdr_dir.mkdir(parents=True, exist_ok=True)

    for spec, typename, build in (
        [(s, "sensor_msgs/msg/CameraInfo", _rosbags_camera_info) for s in CAMERA_INFO]
        + [(s, "sensor_msgs/msg/Image", _rosbags_image) for s in IMAGES]
    ):
        msg = build(ts, spec["fields"])
        raw = bytes(ts.serialize_cdr(msg, typename, little_endian=True))
        (cdr_dir / f"{spec['name']}.bin").write_bytes(raw)
        _dump_json(
            {
                "typename": typename,
                "little_endian": True,
                "fields": spec["fields"],
                "rosbags_version": ROSBAGS_VERSION,
            },
            cdr_dir / f"{spec['name']}.json",
        )

    import cv2

    yuv_dir = out_dir / "yuv"
    yuv_dir.mkdir(parents=True, exist_ok=True)
    by_name = {s["name"]: s["fields"] for s in IMAGES}
    for name, code, golden, _bridge_encoding in YUV_FIXTURES:
        f = by_name[name]
        src = np.array(f["data"], dtype=np.uint8).reshape(f["height"], f["width"], 2)
        rgb = cv2.cvtColor(src, getattr(cv2, code))
        assert rgb.shape == (f["height"], f["width"], 3), rgb.shape
        (yuv_dir / golden).write_bytes(rgb.tobytes())


# --- Mode 2: --ros -> image_geometry + cv_bridge ------------------------------------------


def _rclpy_camera_info(f: dict[str, Any]) -> Any:
    import array

    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import CameraInfo, RegionOfInterest
    from std_msgs.msg import Header

    return CameraInfo(
        header=Header(
            stamp=Time(sec=f["header"]["stamp"]["sec"], nanosec=f["header"]["stamp"]["nanosec"]),
            frame_id=f["header"]["frame_id"],
        ),
        height=f["height"],
        width=f["width"],
        distortion_model=f["distortion_model"],
        d=array.array("d", f["d"]),
        k=array.array("d", f["k"]),
        r=array.array("d", f["r"]),
        p=array.array("d", f["p"]),
        binning_x=f["binning_x"],
        binning_y=f["binning_y"],
        roi=RegionOfInterest(**f["roi"]),
    )


def _rclpy_image(f: dict[str, Any]) -> Any:
    import array

    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header

    return Image(
        header=Header(
            stamp=Time(sec=f["header"]["stamp"]["sec"], nanosec=f["header"]["stamp"]["nanosec"]),
            frame_id=f["header"]["frame_id"],
        ),
        height=f["height"],
        width=f["width"],
        encoding=f["encoding"],
        is_bigendian=f["is_bigendian"],
        step=f["step"],
        data=array.array("B", f["data"]),
    )


def ros(out_dir: Path) -> int:
    import image_geometry
    from cv_bridge import CvBridge

    intrinsics: dict[str, Any] = {}
    for spec in CAMERA_INFO:
        model = image_geometry.PinholeCameraModel()
        model.from_camera_info(_rclpy_camera_info(spec["fields"]))
        intrinsics[spec["name"]] = {
            "intrinsic_matrix": [list(row) for row in model.intrinsic_matrix()],
            "projection_matrix": [list(row) for row in model.projection_matrix()],
        }
    _dump_json(intrinsics, out_dir / "intrinsics.json")

    bridge = CvBridge()
    by_name = {s["name"]: s["fields"] for s in IMAGES}
    bridged: dict[str, Any] = {}
    ok = True
    for name, code, golden, bridge_encoding in YUV_FIXTURES:
        f = dict(by_name[name], encoding=bridge_encoding)
        got = bytes(bridge.imgmsg_to_cv2(_rclpy_image(f), "rgb8").tobytes())
        bridged[name] = {"encoding": bridge_encoding, "rgb8_hex": got.hex()}
        want_path = out_dir / "yuv" / golden
        if not want_path.exists():
            print(f"{f['encoding']}: {want_path} is missing (run mode 1 into this directory first)")
            ok = False
            continue
        want = want_path.read_bytes()
        if got == want:
            print(f"{f['encoding']}: cv_bridge matches cv2.{code} ({len(got)} bytes)")
        else:
            ok = False
            print(f"{f['encoding']}: MISMATCH cv_bridge vs cv2.{code}")
            print(f"  cv2       {want.hex()}")
            print(f"  cv_bridge {got.hex()}")
    _dump_json(bridged, out_dir / "cv_bridge.json")

    print(f"cv_bridge: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--ros":
        return ros(Path(argv[1]))
    if len(argv) == 1:
        generate(Path(argv[0]))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
