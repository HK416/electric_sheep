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

## Pretrained weights (packet M7/T5)

A second, unrelated use of torchvision: `python/es/fetch_backbone.py` turns its ImageNet
ResNet18 weights into a hashed, licensed artifact that `es train` can verify. Spec §19.3
(`base_model.lock`), §2.5 (no network at instantiation), §29 licence row (owner decision
2026-09-15: these weights, BSD-3, are allowed). The semantics are
`docs/design/learning-lowering.md` section 5.3; the recipe side is
`docs/design/training-recipe.md` section 11.

| | |
|---|---|
| API | `torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)` |
| URL (`ResNet18_Weights.IMAGENET1K_V1.url`) | `https://download.pytorch.org/models/resnet18-f37072fd.pth` |
| upstream sha256 of that `.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| produced with | **torchvision 0.26.0+cu129**, torch 2.11.0+cu129 (oracle server); cross-checked under torchvision 0.29.0+cpu, torch 2.14.0+cpu |
| artifact | `resnet18-imagenet1k-v1.safetensors`, 46,805,551 B, 102 F32 tensors |
| blake3 of the artifact (**the pin**) | `8511928e7ca6e3b07355e8b66284294ba692093fd0dfe246cdf96a6c9e801899` |
| licence | BSD-3-Clause, `https://github.com/pytorch/vision/blob/main/LICENSE` |

Four notes on that table.

* **The pin is a property of the weights, not of the interpreter.** The two torchvision
  versions above write byte-identical safetensors, because the tensors are the upstream
  `.pth`'s and the writer is deterministic (keys sorted, F32, little-endian). The pin lives in
  `crates/es-data/src/training.rs` as `RESNET18_IMAGENET1K_V1_BLAKE3` and nowhere else; the
  script is given it with `--expect`.
* **`num_batches_tracked` is dropped.** It is an int64 count of the batches upstream training
  saw, not a weight; `FrozenBatchNorm2d` — what both `lower/torch.rs` and `lerobot.rs` build —
  has no such buffer, and this project's safetensors layout is F32 only
  (`crates/es-policy/src/weights.rs`). The lock file names it under `dropped` rather than
  leaving a reader to notice. 102 tensors is 100 backbone tensors plus `fc.weight`/`fc.bias`,
  and the `fc` pair is skipped at load time because the lowering replaces `fc` with
  `Linear(512, out_dim)`.
* **This is the only pickle read in the project**, and torchvision does it, inside
  `load_state_dict_from_url`, on the learning path (§2.3). `es` never opens a `.pth`; INV-16 is
  about what this project's loaders accept, and they accept safetensors.
* **The `.pth` download is cached** by `torch.hub` under `torch.hub.get_dir()/checkpoints`, and
  `fetch_backbone.py` hashes the cached file for `sha256_upstream` rather than re-downloading —
  torchvision itself only checks the first eight hex digits, which is what the file name
  carries, and a provenance record with eight digits is not one anyone can verify.

### Upgrade procedure

The same shape as the golden one above, with one difference: nothing here is a golden file, so
a torchvision upgrade that moves the artifact moves a *constant*. Run
`cargo test -p es-policy --test backbone_provenance -- --ignored` with `ES_PYTHON` set; it
fails naming both digests. Then decide whether upstream republished the weights (a new URL and
sha256, so a new pin) or the writer changed (a bug, fix it), and only then run
`fetch_backbone.py --repin`, moving `RESNET18_IMAGENET1K_V1_BLAKE3`, this table and
`docs/design/learning-lowering.md` section 5.3 in the same commit. Every `training_hash` taken
under the old pin is a different run, and saying so is the point of the pin existing.
