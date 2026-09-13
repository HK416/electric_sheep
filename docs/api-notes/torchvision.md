# torchvision / torch — pinned API surface for the observation goldens

What `crates/es-compile/python/gen_observation_goldens.py` actually calls, so a torch upgrade
is a diff against this file rather than an archaeology session. Companion to
`docs/api-notes/torch.md` (the `TorchRuntime` surface), same shape, same reason.

Spec: spec 1.4 (the goldens must come from a reference oracle, not from the code under test),
spec 7.7 (LeRobot preprocessing equivalence), spec 11.3 (the CPU plan is ground truth). The
semantics these calls pin are `docs/design/observation-lowering.md`; the packet is
`docs/packets/M1/P-M1-R1.md`.

## Version

| | |
|---|---|
| verified against | **torch 2.14.0+cpu**, **torchvision 0.29.0+cpu** (`--index-url https://download.pytorch.org/whl/cpu`) |
| Python | 3.12.10 |
| `numpy` | **2.5.2** — required, and pulled in by the torchvision wheel; `to_tensor` takes a PIL image or an ndarray, not a tensor |

Install for a local run:

```
python -m venv <short path>            # a deep path fails on Windows without long-path support
<venv>/Scripts/python -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu
ES_PYTHON=<venv>/Scripts/python cargo test -p es-compile --test gen_goldens
```

`ES_PYTHON` is the same variable the MuJoCo and `TorchRuntime` oracles use, so one venv serves
all three. Without it, `python` then `python3` are tried in order; with no torchvision
anywhere, `gen_goldens` prints `SKIPPED` and passes, and every other test still runs.

## The API this depends on

| API | golden | notes |
|---|---|---|
| `torchvision.transforms.functional.to_tensor(ndarray HWC u8)` | `dequantize_8x6_rgb` | `permute(2,0,1).contiguous().float().div(255)`. Bit-equal to the kernel's `f32::from(u8) / 255.0` — the division is f32 on both sides |
| `torch.nn.functional.interpolate(x, size, mode="bilinear", align_corners=False, antialias=False)` | `resize_bilinear_8x6_to_4x3` | half-pixel centres. `antialias` **must** be passed explicitly: torchvision's `Resize` has defaulted to `antialias=True` since 0.17, and that is a different filter, not a refinement (design note §11) |
| tensor slicing `chw[:, y:y+h, x:x+w]` | `crop_8x6_at_2_1_4x4` | deliberately *not* `TF.crop`, which pads out-of-bounds rectangles; the IR rejects those at compile time (`COMPILE-003`) |
| `torch.float64` arithmetic, `.to(torch.float32)` | `srgb_to_linear_lut256` | IEC 61966-2-1 EOTF, `x/12.92` below 0.04045 and `((x+0.055)/1.055) ** 2.4` above, evaluated in f64 and rounded once. Not a torchvision call — torchvision has no sRGB EOTF, and `**` in f64 is the most accurate reference available |
| `torchvision.transforms.functional.normalize(t, mean, std)` | `normalize_imagenet_4x3` | `sub_(mean).div_(std)`: division, not a reciprocal multiply |
| `torch.stack(frames, dim=0)` | `history_window_n2_s1` | oldest → newest, current frame last |

## Determinism

The script sets `torch.use_deterministic_algorithms(True)`, `torch.manual_seed(0)` and
`torch.set_num_threads(1)`. Nothing in it draws from the RNG today; the seed is set so that
stays true if a golden ever does. Every value is CPU f32 or f64 — no CUDA, no TF32, no
autocast.

## What is not bit-equal, and why

Five of the six goldens are reproduced bit-for-bit by the Rust kernels. The sixth,
`srgb_to_linear_lut256`, carries `"tolerance_ulp": 7` in its sidecar: `DET-010` forbids `powf`
in an observation kernel, so `srgb_eotf` evaluates the 2.4 power as
`es_math::approx::exp(2.4 * ln t)` in f32 and cannot land on the f64 reference's bits. 7 ULP
(4.8e-7 relative) is the measured worst case over all 256 entries, not a margin chosen to make
a test pass.

`torch.nn.functional.interpolate` agrees bit-for-bit at the pinned golden size but not at every
size: across a sweep of 4,624 (source, target) size pairs, 1,585 are bit-equal and the rest
differ by 1 to 4 ULP. The half-pixel indices agree exactly; the tap weights do not. Recorded,
not matched — see design note §12 item 6.

## Upgrade procedure

1. Install the new wheels, run `cargo test -p es-compile --test gen_goldens` with `ES_PYTHON`
   set. It fails, naming the file whose bytes moved.
2. Decide whether the move is a torch bug fix (accept) or a semantics change (a new kernel id
   and a design-note change first).
3. Regenerate with the script pointed at `tests/golden/observation`, update the version table
   above, and gate the commit with `GOLDEN_UPDATE=1 cargo xtask verify-goldens`.
