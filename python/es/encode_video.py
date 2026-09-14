"""Encode `es video mosaic` output frames into an mp4 (design note
`docs/design/visible-learning.md` sections 2.9, 9; spec 2.4, 5.3).

The only Python in the M5 V4 packet, and a presentation step, not a runtime path: the mp4 it
writes is not part of the execution hash chain. It reads the raw `NNNNNN.bin` + `layout.json`
frames `es video mosaic` wrote -- it never guesses a shape -- and writes them with
`cv2.VideoWriter`. The oracle server has no `ffmpeg` binary and only the `mp4v` fourcc (MPEG-4
Part 2) opens there (measured 2026-09-14; `avc1`/`H264` fail: the only linked H.264 encoder finds
no device), so `mp4v` is the default codec, not a placeholder for something better.

Usage:

    python encode_video.py --frames <mosaic dir> --out demo.mp4 --fps N [--codec mp4v]

Prints one JSON line on success: {"frames": n, "width": w, "height": h, "codec": "mp4v"}, and
nothing else on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_layout(frames_dir: Path) -> tuple[int, int, int]:
    layout_path = frames_dir / "layout.json"
    layout = json.loads(layout_path.read_text())
    shape = layout["shape"]
    if not (isinstance(shape, list) and len(shape) == 3) or layout.get("dtype") != "u8":
        raise ValueError(f"{layout_path}: expected a 3-element u8 shape, got {layout!r}")
    h, w, c = shape
    return int(h), int(w), int(c)


def frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.bin"), key=lambda p: p.stem)


def encode(frames_dir: Path, out: Path, fps: float, codec: str) -> dict:
    h, w, c = load_layout(frames_dir)
    if c != 3:
        raise ValueError(f"encode_video only reads 3-channel u8 frames, got {c} channel(s)")
    paths = frame_paths(frames_dir)
    if not paths:
        raise ValueError(f"no *.bin frames under {frames_dir}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out} with codec {codec!r}")
    try:
        expected = h * w * c
        for p in paths:
            raw = p.read_bytes()
            if len(raw) != expected:
                raise ValueError(f"{p}: {len(raw)} byte(s), layout.json declares {expected}")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, c)
            # Our sidecar frames are RGB (`es-render`'s `Channel::Rgb8` byte order);
            # `cv2.VideoWriter` wants BGR.
            writer.write(frame[:, :, ::-1])
    finally:
        writer.release()

    return {"frames": len(paths), "width": w, "height": h, "codec": codec}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", required=True, type=float)
    ap.add_argument("--codec", default="mp4v")
    args = ap.parse_args(argv)

    try:
        result = encode(args.frames, args.out, args.fps, args.codec)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
