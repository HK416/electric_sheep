"""Generate `tests/golden/observation/**` from PyTorch/torchvision (spec 1.4).

The goldens are the *reference oracle* for the Observation IR CPU plan, so they must not be
produced by the code they check (M1 review finding, packet `docs/packets/M1/P-M1-R1.md`).
Every tensor below comes from torch or torchvision; nothing here imports or mirrors
`es-compile`.

Run:

    ES_PYTHON=<venv>/Scripts/python cargo test -p es-compile --test gen_goldens -- --ignored

or directly:

    python crates/es-compile/python/gen_observation_goldens.py <out-dir>

Output: little-endian, tightly packed `.bin` plus a `.json` sidecar naming shape, dtype,
kernel, the exact torch call that produced it, and — where bit-equality with `es-math`'s
polynomial transcendentals is impossible — a `tolerance_ulp`.

Pinned versions: `docs/api-notes/torchvision.md`. Semantics: `docs/design/observation-lowering.md`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

# Determinism (spec 3.4): CPU, one thread, seeded, no nondeterministic kernels. Nothing here
# actually draws from the RNG — the seed is set so that stays true if a golden ever does.
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
torch.set_num_threads(1)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DTYPE_NAME = {torch.float32: "f32", torch.uint8: "u8"}


def gradient_8x6() -> np.ndarray:
    """The 8x6 RGB u8 HWC synthetic gradient every image golden starts from.

    Every channel varies on a different axis, so a channel swap or an HWC/CHW slip shows up.
    """
    px = np.zeros((6, 8, 3), dtype=np.uint8)
    for y in range(6):
        for x in range(8):
            px[y, x] = (x * 32, y * 40, (x + y) * 16)
    return px


def write(out: Path, name: str, tensor: torch.Tensor, kernel: str, pins: str, oracle: str,
          tolerance_ulp: int = 0) -> None:
    tensor = tensor.contiguous()
    out.mkdir(parents=True, exist_ok=True)
    # The sidecar promises little-endian; a no-op on every host we build on, but the promise
    # should not depend on the host.
    arr = tensor.numpy()
    (out / f"{name}.bin").write_bytes(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())
    sidecar = {
        "name": name,
        "dtype": DTYPE_NAME[tensor.dtype],
        "layout": "row-major, little-endian, tightly packed",
        "shape": list(tensor.shape),
        "kernel": kernel,
        "pins": pins,
        "oracle": oracle,
        "generator": "crates/es-compile/python/gen_observation_goldens.py",
        "spec": "docs/design/observation-lowering.md",
    }
    if tolerance_ulp:
        sidecar["tolerance_ulp"] = tolerance_ulp
    (out / f"{name}.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def main(out: Path) -> None:
    # 1. ToTensor: HWC u8 -> CHW f32 / 255.
    chw = TF.to_tensor(gradient_8x6())
    write(out, "dequantize_8x6_rgb", chw, "cast_u8_hwc_to_f32_chw.v1",
          "the HWC u8 -> CHW f32 /255 boundary conversion",
          "torchvision.transforms.functional.to_tensor")

    # 2. Bilinear downscale, align_corners=false, antialias=false.
    small = torch.nn.functional.interpolate(
        chw.unsqueeze(0), size=(3, 4), mode="bilinear", align_corners=False, antialias=False
    ).squeeze(0)
    write(out, "resize_bilinear_8x6_to_4x3", small, "resize_bilinear.v1",
          "half-pixel centres, align_corners=false, no antialias, PyTorch's tap association",
          'torch.nn.functional.interpolate(mode="bilinear", align_corners=False, antialias=False)')

    # 3. Crop: rect (x=2, y=1, w=4, h=4), origin top-left.
    write(out, "crop_8x6_at_2_1_4x4", chw[:, 1:5, 2:6], "crop.v1",
          "origin top-left (OpenCV, spec 3.1); pairs with ImageSpec::cropped for the intrinsics",
          "tensor slicing chw[:, y:y+h, x:x+w]")

    # 4. The sRGB EOTF at every u8 input, in float64, rounded once to f32.
    #    IEC 61966-2-1: the reference formula, not es-math's polynomial `exp(2.4 * ln t)`.
    k = torch.arange(256, dtype=torch.float64) / 255.0
    lut = torch.where(k <= 0.04045, k / 12.92, ((k + 0.055) / 1.055) ** 2.4).to(torch.float32)
    write(out, "srgb_to_linear_lut256", lut, "srgb_to_linear.v1",
          "sRGB EOTF at k/255; the reference f64 formula rounded to f32, not a polynomial fit",
          "IEC 61966-2-1 in torch.float64, cast to float32",
          # `srgb_eotf` must use es_math::approx (no std powf, DET-010), so the kernel is a
          # polynomial exp/ln pair evaluated in f32 and cannot be bit-equal to the f64 formula.
          # 7 is the measured worst case over all 256 entries (4.8e-7 relative), not a margin;
          # see observation-lowering.md section 6. Every other golden is bit-exact.
          tolerance_ulp=7)

    # 5. Per-channel normalize with the ImageNet statistics.
    write(out, "normalize_imagenet_4x3", TF.normalize(small, MEAN, STD), "normalize_mean_std.v1",
          "(x - mean[c]) / std[c] by division, not by a reciprocal multiply",
          "torchvision.transforms.functional.normalize")

    # 6. A two-frame history window over a 4-element state, after three pushes, stride 1:
    #    oldest -> newest with the current frame last, so the window is frames 1 and 2.
    frames = [torch.tensor([i, i + 0.5, i + 1.0, i + 1.5], dtype=torch.float32) for i in range(3)]
    write(out, "history_window_n2_s1", torch.stack(frames[1:], dim=0), "window_gather.v1",
          "oldest -> newest, current frame last; Align::Hold before the ring fills",
          "torch.stack")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/golden/observation"))
