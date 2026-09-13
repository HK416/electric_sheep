# Observation IR → CPU reference execution plan

Design note for `crates/es-compile` (layer 7). Spec: spec 7.2, spec 7.3, spec 7.5, spec 7.6,
spec 7.7, spec 11.1, spec 11.3, spec 11.5, spec 3.1, spec 3.4. Work packet:
`docs/packets/M1/W3-observation-cpu-ref.md`.

Review class C (human review of the numerics): everything below fixes bit patterns that
goldens and, later, a GPU lowering must reproduce. Getting the resize convention wrong is
silent — the images still look right and the policy is fed something it was not trained on.

## 1. What this is and is not

Spec 11.3: **the CPU path is not a performance path, it is the ground truth.** It is the
oracle the Slang/SPIR-V lowering (spec 11.4) is judged against, it runs without Slang so CI
and macOS can gate on it, and it can show every node's intermediate value.

Not in this note: fusion, GPU buffers, the `PolicyRuntime` call plan. Those are later waves.

## 2. Tensor model

```rust
Tensor { dtype: ElemType, shape: Vec<u64>, data: Vec<u8> }
```

Row-major, tightly packed, no padding, no batch axis (spec 5.4: the batch axis belongs to a
batch domain, not to a shape).

**Layout rule, two layouts only:**

| where | layout | dtype |
|---|---|---|
| sensor boundary (`ImageInput`) | **HWC** | `U8` |
| everything downstream | **CHW** | `F32` (or `F16`/`Bf16` at an output) |

CHW f32 is what PyTorch and therefore LeRobot feed a policy, and matching it is the whole
point of spec 7.7 ("LeRobot preprocessing equivalence"). HWC u8 is what every camera driver
and every LeRobot parquet/video frame actually hands over, so converting anywhere but at the
boundary would mean converting twice.

The layout change happens in one node kind: `Dequantize` — and its fused twin, a
`ColorTransform` that reads a u8 port directly (§6), which is the same conversion with the
EOTF folded in. `Dequantize` is therefore the IR spelling of torchvision `ToTensor()` —
HWC u8 → CHW f32 divided by 255. Nothing else in the plan ever permutes axes.

State tensors (`StateInput`) are 1-D and have no layout question.

## 3. Node → kernel table

This is the M1 W3 part-1 subset. A node outside it is a `COMPILE-002` diagnostic at compile
time, never a silent no-op.

| Observation IR node | kernel | notes |
|---|---|---|
| `ImageInput` | — | names a plan input buffer; u8 HWC |
| `StateInput` | — | names a plan input buffer; f32 1-D |
| `Dequantize` | `cast_u8_hwc_to_f32_chw` | `/255`, the one layout change |
| `Resize{Bilinear}` | `resize_bilinear` | §4 |
| `Resize{Nearest}` | `resize_nearest` | `floor(scale * (d + 0.5))`, PyTorch's rule |
| `Crop{Rect\|Center\|Random}` | `crop` | §5 |
| `ColorTransform{SRgb→Linear}` | `srgb_to_linear` / LUT | §6 |
| `Normalize{MeanStd\|Range}` | `normalize` | §7 |
| `Concat{axis}` | `concat` | §8 |
| `Stack{axis}` | `stack` | §8 |
| `TemporalWindowNode` | `history_push` + `window_gather` | §9 |
| `Augment{training_only}` | — | identity pass-through, §9.1 |
| output elem ≠ `F32` | `cast_f32_to_f16` / `_bf16` | `half`, round-to-nearest-even |
| `Pad`, `Undistort`, `Rectify`, `Warp`, `CameraProjection`, `ToGray`, `ChannelSelect`, `QuantizeU8`, `FrameStack`, `Delta`, `Mask`, `MultiViewPack`, `LanguageInput`, `Augment` (not `training_only`) | — | `COMPILE-002`, later waves |

### `Augment` and INV-15 (spec 7.3, spec 10.4)

This plan is the evaluation and deployment path: there are no augmentation kernels on it and
there is no training mode from which to reach them, so there is no `PlanOptions { training }`
flag — the flag belongs in the packet that brings the kernels, and inventing it now would be a
switch with one position. What the two node flavours lower to:

| node | lowering |
|---|---|
| `Augment { training_only: true }` | identity: no step, no buffer; consumers read the producer's buffer |
| `Augment { training_only: false }` | `COMPILE-002` |

The identity case *is* how "evaluation disables augmentation" is realised (§10.4's
auto-disable). The node is **not** removed from the graph: stripping it would make
`observation_hash` describe a graph the author never declared. `es-eval` therefore no longer
has to refuse a `training_only` node — the plan has already disabled it — and still refuses
any other `Augment` node outside its allow-list.

### Determinism rules every kernel obeys (spec 3.4)

Single-threaded, fixed loop order (outer→inner exactly as written), no atomics, no `HashMap`
iteration, no `f64` accumulators, no `std` transcendentals — `es_math::approx` only, so Rust
and the future Slang kernel share coefficients *and* evaluation order. Accumulation in these
kernels is over at most four terms, so no `es_math::reduce` is needed yet; a future `Area`
resize or a `ToGray` over many taps will need one.

## 4. Resize: bilinear, `align_corners = false`, half-pixel centres

Target semantics: `torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear",
align_corners=False, antialias=False)`, which is also torchvision
`Resize(..., antialias=False)`.

For each output index `d` on an axis of length `D`, with `scale = S / D`:

```
s      = scale * (d + 0.5) - 0.5
s      = max(s, 0.0)                  // PyTorch clamps the negative edge, not the positive
i0     = floor(s)                     // as integer
i1     = min(i0 + 1, S - 1)
l1     = s - i0                       // in [0, 1]
l0     = 1 - l1
```

and the sample, in this exact association (it is PyTorch's, and reassociating it changes the
last bits):

```
out = h0 * (w0 * v[y0][x0] + w1 * v[y0][x1])
    + h1 * (w0 * v[y1][x0] + w1 * v[y1][x1])
```

All arithmetic f32. `s` is computed in f32 from f32 `scale`; computing it in f64 and
narrowing gives different bits at some sizes, so the f32 path is normative.

This is **not** exact on a constant image: the taps are combined with `l0 = 1 - l1`, whose
f32 sum is not exactly 1, so a flat input can come back off by an ulp. PyTorch is not exact
there either — `interpolate` on a constant 5×5 → 7×7 returns three distinct values — so the
proptest asserts 2 ulp, not equality. An association that is exact on constants would be a
different kernel.

Measured against torch 2.14 at the golden size (8×6 → 4×3) this kernel is **bit-equal**. It is
not bit-equal at every size: see §12 item 6, which records where and by how much.

**Antialiasing is not implemented.** For a downscale, `antialias=True` is a different
algorithm (a support-widened filter), not a refinement of this one. See §11.

## 5. Crop

`Crop` copies `rect.width × rect.height` starting at `(rect.x, rect.y)`, per channel plane.
`CropMode::Center` and `CropMode::Random` resolve to the centred rectangle
`((W - w) / 2, (H - h) / 2, w, h)` — the same rectangle `CropMode::rect` gives the type
system, so the plan's geometry and the propagated `ImageSpec` cannot disagree. Random offsets
are an augmentation and belong with `Augment` (§3).

A rectangle that leaves the image is `COMPILE-003` at compile time, not a clamp at run time.

**INV-14.** The plan never touches `Intrinsics`. Every geometric node takes the `ImageSpec`
that `ObservationIr::propagate_image_specs` produced, which is built by `ImageSpec::resized` /
`ImageSpec::cropped`. Chained crops therefore compose the same way the plan's pixels do:
`cropped(a).cropped(b)` shifts the principal point by `a.x + b.x`, and cropping once by the
combined rectangle gives the same intrinsics and the same pixels. That equality is a proptest.

## 6. Colour: sRGB → linear

Spec 3.1: **sRGB is the default and linearisation happens only through an explicit node.** The
Observation IR already refuses a `ColorTransform` whose declared `src` is not the incoming
`ColorSpace` (`OBS-021`), so the kernel never has to guess.

The scalar EOTF, applied per channel, alpha untouched (there is no alpha in the supported
channel formats yet):

```
srgb_eotf(x) = x <= 0.04045 ? x / 12.92
                            : approx::exp(2.4 * approx::ln((x + 0.055) / 1.055))
```

`powf` is forbidden in an observation kernel (spec 3.2, `DET-010`), so the 2.4 power is
`exp(2.4 * ln(t))` through `es_math::approx`, whose coefficients the Slang mirror shares.
That costs a little accuracy against `libm`; it buys the only thing that matters here, which
is that the CPU oracle and the GPU kernel produce the *same* bits.

**The one golden with a tolerance.** The oracle for this table is the IEC 61966-2-1 formula
evaluated in f64 and rounded once to f32 — the most accurate reference available, and by
construction *not* reachable by a f32 polynomial fit. So `srgb_to_linear_lut256.json` carries
`"tolerance_ulp": 7` and `observation_cpu.rs` reads it from the sidecar. 7 ULP (4.8e-7
relative, worst entry `k = 12`) is the measured maximum over all 256 entries, not a margin
picked to pass: 38 entries are exact, 167 are within 2 ULP, 3 reach 7. Every other golden in
the set has no tolerance and is compared byte-for-byte. The tolerance lives in the sidecar
rather than the test because the sidecar is CI read-only — widening it means modifying a
golden, which `cargo xtask verify-goldens` refuses.

**LUT-256.** When the node's input is u8 (a `ColorTransform` wired straight to an
`ImageInput`, ahead of any `Dequantize`), the input takes only 256 values, so the plan
evaluates `srgb_eotf(k / 255)` once per `k` into a
`[f32; 256]` table and indexes it, writing CHW f32 — the dequantize is folded in, so the
two-step form and this fused one produce the same tensor. The table is built from the same
`srgb_eotf`, so the two paths agree by construction rather than by review. The golden for
this node is that table.

`Linear → SRgb` (the inverse, needed by the spec 7.7 round-trip oracle) is not implemented in
this part — the round-trip oracle is the next wave, and the LUT does not invert exactly.

## 7. Normalize

CHW, per channel `c`:

- `MeanStd { mean, std }`: `(x - mean[c]) / std[c]`. Division, not multiplication by a
  reciprocal — PyTorch's `normalize` divides, and the reciprocal form differs in the last bit.
  `mean`/`std` are `f64` in the IR and are narrowed to `f32` once, at plan time.
  `len(mean) == len(std) == C` or `COMPILE-004`.
- `Range { lo, hi }`: `(x - lo) / (hi - lo)`, the same scalar for every channel.

`Normalize`'s output unit must be `Unit::Normalized` — the IR already enforces that
(`OBS-040`), so the plan does not repeat the check.

## 8. Concat and stack

`Concat { axis }` joins inputs along an existing axis; every other axis must match.
`Stack { axis }` inserts a new axis of length `n_inputs` at `axis`; all inputs must have the
same shape. Both are pure byte copies in declared input-port order (`in0`, `in1`, …), which is
the order the edge list is sorted into, so the result does not depend on how the graph was
authored.

`time_align` is not a run-time operation here: alignment across clocks is decided by the IR
type check (`TYPE-014`) and by whoever fills the input buffers. The plan copies.

## 9. History and window (spec 7.5)

Three layers, and only the middle one is a node:

- `History<T, N>` — the ring buffer, owned by `es-core`/`es-sensor`, outside the IR. The plan
  owns a *ring per windowed stream* only because the CPU reference has to run standalone; a
  real deployment hands the ring in.
- `TemporalWindow(n_steps, stride, align)` — the node. Shape-bearing.
- `TemporalEncoder` — Learning IR. Not here.

```
history_push(ring, slot_len, depth, cursor, frame)   // writes slot cursor % depth
window_gather(ring, slot_len, depth, cursor, n, stride, dst)
```

`window_gather` emits oldest→newest, i.e. slot `cursor - (n - 1 - k) * stride` for
`k = 0..n`, so the last row of the output is always the current frame — the convention
LeRobot's `delta_timestamps` uses for a negative-offset observation window. Before the ring
has filled, the oldest available frame is repeated (`Align::Hold`); `Align::Interpolate` and
`Align::Reject` are `COMPILE-002` for now.

Output shape is `[n_steps, ...frame_shape]`.

### 9.1 `CpuPlan::reset` — the rings are the plan's only state

A stream ends at an episode boundary. `CpuPlan::reset()` refills every ring with zeros and
puts `cursor` and `pushed` back to 0, which is exactly the state `compile` leaves them in, so
`reset(); run(x)` equals a freshly compiled plan's first `run(x)` (asserted in
`crates/es-compile/tests/observation_cpu.rs`). `es-eval` calls it after every `env.reset`
(P-M2-R1): without it, episode N's first frames see episode N−1's tail and cell 2's see cell
1's, which makes the §10.1 table depend on suite order — the one thing §10.4 exists to
prevent. Any future stateful buffer on this path must be cleared here too.

## 10. Plan, buffers, debug vs release (spec 11.5)

`CpuPlan::compile` runs: `ObservationIr::validate` (errors abort, warnings are kept),
`propagate_image_specs`, `topo_order`, then one pass assigning each node's `out` port a
`BufferId` whose dtype and shape come from the node's **declared** `PortType` — the IR is the
contract, the plan does not re-derive shapes it could disagree with. Sizes are summed into one
flat arena that `run` allocates once; a `BufferId` is `(offset, len, dtype, shape)`.

Spec 11.5 asks for two plans:

| | release | debug |
|---|---|---|
| fusion | maximal | node boundaries kept |
| per-node values | sampled | all |

On this path **both modes keep every intermediate addressable and the arena is not aliased.**
There is no fusion on the CPU reference path (fusing is the GPU lowering's job, and the CPU
plan exists to *have* node boundaries), so liveness-based aliasing would save memory on a path
that is not a memory path while removing the only thing it is for. `PlanMode` is recorded on
the plan and reaches `compiler_hash`, so a future release plan that does alias gets a
different hash instead of silently reusing a cached one.

`compiler_hash()` = blake3 over (`es-compile` crate version, plan mode, every kernel id in the
table, in order). It fills the `compiler` slot of `execution_hash` (spec 5.3, spec 11.2):
changing a kernel's numerics must change it, which is why the ids — not the source — are
hashed and why a new kernel is appended, never inserted.

## 11. GPU lowering (delivered) - how each kernel mirrors

`GpuPlan::compile` (`crates/es-compile/src/gpu/`) runs `CpuPlan::compile` and mirrors it:
the same topological order, the same buffer table, the same kernel per node. It is not a
second compiler, so the two paths cannot drift apart by *deciding* differently - only by
computing differently, which is what `tests/observation_gpu.rs` measures.

Kernels: `crates/es-compile/slang/observation.slang`, one entry point per kernel id,
specialised entirely by `-D` defines (dtype, channels, sizes and buffer offsets). A pipeline
is therefore a pure function of `(kernel id, defines)`: two resizes to the same size share
one, two to different sizes do not.

| kernel | GPU form | measured |
|---|---|---|
| `cast_u8_hwc_to_f32_chw` | one thread per output element, same `ch/y/x` decomposition, `float(byte) / 255.0` | bit-equal to the CPU kernel and to `dequantize_8x6_rgb` |
| `resize_bilinear` | one thread per output pixel, the section 4 index arithmetic and PyTorch's association verbatim. Texture samplers are **not** used: their filtering precision is vendor-defined | bit-equal at the golden size and on 50/50 random `(source, target)` pairs up to 17x17, worst 0 ULP |
| `resize_nearest` | same, `min((uint)(scale * d), S - 1)` | bit-equal to the CPU kernel |
| `crop` | its own dispatch, an index offset per plane. Not fused into the consumer: debug mode keeps node boundaries (section 10), and fusion is a later packet | bit-equal to `crop_8x6_at_2_1_4x4` |
| `srgb_to_linear` | two entry points behind one id: `srgb_to_linear_u8` indexes the LUT, `srgb_to_linear` evaluates the EOTF through `es-math/slang/approx.slang`. The LUT is a 256-entry **storage** buffer uploaded from `kernels::srgb_to_linear_lut()` - the CPU's own bytes, not a re-derivation, and not a texture | LUT path within the sidecar's 7 ULP of the golden (the CPU kernel's own distance from the f64 formula); elementwise path bit-equal to the CPU kernel, 0 ULP |
| `normalize_mean_std` / `normalize_range` | element-wise; mean/std ride in the same `aux` buffer as the LUT, `lo`/`hi` are f32 **bit patterns** in the defines so a decimal round-trip cannot round twice. Not fused with its producer, for the reason `crop` is not | bit-equal to `normalize_imagenet_4x3` |
| `concat` / `stack` | one dispatch per input, copying that input's `outer x chunk` elements into its fixed slot. Not elided into the producers: that is fusion. The Slang entries are `concat_copy` / `stack_copy` - `concat` and `stack` are taken by the Slang core module - while the kernel ids stay `concat.v1` / `stack.v1` | bit-equal to the CPU kernel |
| `history_push` / `window_gather` | the rings live in device memory and survive across `run`s; `cursor`/`pushed` ride in a small `state` buffer uploaded per run, because a pipeline's defines are compile-time and a cursor is not. `window_gather` is an index remap | bit-equal to `history_window_n2_s1`; `GpuPlan::reset` equals `CpuPlan::reset` |
| `cast_f32_to_f16` / `_bf16` | element-wise, the `half` crate's round-to-nearest-even mirrored in integer ops. The narrowed **bits** go to a `uint` region, not an f16 buffer: the device is not opened with 16-bit storage, and the arena keeps the f32 so spec 11.5's per-node view survives | bit-equal to the CPU kernel |

Five bindings serve every kernel - `arena` (f32 intermediates + f32 inputs), `words` (u8
inputs, then narrowed output bits), `aux` (LUT + normalize statistics), `rings`, `state` -
so one descriptor layout covers the whole plan and every buffer offset can be a define.

Determinism (spec 3.4): the execution modes come from
`Capabilities::deterministic_execution_modes()` - a *compile* input derived from the
capability query, never a device setting - and `es_gpu::apply_exec_modes` patches
`NoContraction` onto the float arithmetic, without which the resize's
`scale * (d + 0.5) - 0.5` becomes an fma and the last bits move. One queue, one submission
per `run`, a full barrier between dispatches, no atomics, no shared memory, no subgroup
operations, no dependence on the workgroup count. The whole arena is re-uploaded each `run`
rather than patched, which removes the only way two runs of one plan could differ.

`GpuPlan::compiler_hash()` = the CPU plan's hash (crate version, plan mode, kernel ids) plus
every pipeline's SPIR-V content hash, so editing a `.slang` file moves it even when no kernel
id did (spec 3.4 item 7). Denormals are the one documented divergence: the deterministic
execution modes flush them to zero on the device and the CPU kernels do not.

Fusion boundary (spec 11.4): reductions, shape changes, sensor reads, and - in debug mode -
every node boundary. **No fusion is implemented yet**; every node is its own dispatch in both
modes, exactly as the CPU plan keeps every node its own step (section 10). Release-mode
fusion and arena aliasing are a packet of their own, and they are what `PlanMode` reaching
the hash is there to protect.

## 12. What is `unverified` against LeRobot

Marked per spec 12.4's rule that unverified is stated, not implied. Item 2 moved from
`unverified` to **measured** when the goldens were regenerated from torch (§13); item 6 is
what that measurement turned up.

1. **Which resize LeRobot actually calls — `unverified`, and the top question for review.**
   This note implements `interpolate(..., align_corners=False, antialias=False)`. torchvision's
   `Resize` has defaulted to `antialias=True` since 0.17, and different LeRobot policies
   (ACT, Diffusion Policy, SmolVLA, π₀) resize in different places — dataset transform,
   processor, or not at all. Until one of those call sites is read and pinned, the agreement
   claimed by spec 7.7 is untested for any downscale. If the answer is `antialias=True`, an
   antialiased kernel is an additional kernel id, not a change to this one.
2. **u8 → f32 scaling — measured.** `cast_u8_hwc_to_f32_chw` is bit-equal to
   `torchvision.transforms.functional.to_tensor` on the golden image: same permute, same f32
   division by 255. What stays `unverified` is which LeRobot path is in play — some hand over
   an already-float video frame decoded by torchcodec/ffmpeg, where the u8 quantisation never
   happened.
3. **`Normalize` statistics — `unverified`.** The plan applies whatever the IR carries. Whether
   those are LeRobot's per-dataset `mean`/`std` or ImageNet's is a dataset question, one layer
   up.
4. **f16 rounding — `unverified`.** `half` rounds to nearest-even; PyTorch's `.half()` does
   too, but this is asserted, not measured.
5. **Ring-buffer fill behaviour before `n_steps` frames exist — `unverified`.** LeRobot's
   `delta_timestamps` clamps to the first frame of the episode, which is what `Align::Hold`
   does here, but the equality has not been run.
6. **`resize_bilinear` away from the golden size — measured, and it disagrees.** A sweep of
   4,624 (source, target) size pairs against torch 2.14 CPU: 1,585 bit-equal (the golden size
   among them), the rest off by **1 to 4 ULP**. The half-pixel convention is *not* the cause —
   the source indices agree exactly, and computing them in f64 and narrowing makes the
   disagreement worse, not better, which is why §4 keeps the f32 path normative. What differs
   is how the two tap weights are formed: this kernel takes `l0 = 1 - l1`, torch's CPU kernel
   appears to normalise both by their sum, which is the only reading that explains a source
   axis of length 1 — there torch returns the input value exactly (one tap, weight exactly 1)
   while this kernel returns `l0 * v + l1 * v`, off by 1 ULP. Matching it means a new kernel id
   (weights are part of the numerics `compiler_hash` covers) and a matching Slang change, so it
   is a packet of its own, not a patch, and it needs torch's source read rather than inferred.
   Until then, "matches PyTorch bit for bit" is true at the pinned golden sizes and
   4-ULP-true elsewhere.

## 13. Goldens

`tests/golden/observation/*.bin` + a `.json` sidecar per file (shape, dtype, kernel, what it
pins, the oracle call that produced it, and an optional `tolerance_ulp`).

**They come from PyTorch and torchvision, never from the kernels they check** (spec 1.4):
`crates/es-compile/python/gen_observation_goldens.py` imports nothing from this workspace, and
the versions it was pinned against are in `docs/api-notes/torchvision.md`. Generating them
from `es-compile` itself was the M1 review's blocker — a wrong half-pixel convention would
have been enshrined rather than caught. Packet: `docs/packets/M1/P-M1-R1.md`.

`cargo test -p es-compile --test gen_goldens` re-runs that script into a temp directory and
fails if a byte differs, so provenance is machine-checked whenever torch is installed
(`ES_PYTHON` points at it) and prints `SKIPPED` when it is not. Replacing a golden means
running the script at `tests/golden/observation` and gating the commit with
`GOLDEN_UPDATE=1 cargo xtask verify-goldens` — a deliberate act with a spec change behind it,
never a way to make a test pass.

Comparison is byte-for-byte unless the sidecar declares a `tolerance_ulp`; only
`srgb_to_linear_lut256` does, at 7 ULP, for the reason in §6.

Little-endian f32/u8, tightly packed — the same bytes the arena holds.
