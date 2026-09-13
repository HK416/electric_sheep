# Renderer — tile atlas, compute rasterizer, compute path tracer

Spec: §15 (vision data plane: render → observation, tile atlas, render paths, acceleration
structures), §3.1 (OpenCV camera and image conventions, sRGB default), §3.3 (rendering is
FP32, camera-relative coordinates), §3.4 (deterministic execution contract), §7.2
(`ImageSpec`), §16.2 (the splat path is a *render path*, not a separate feature), §28.3 W2
(tile atlas + channel contract), §28.6 (PT + ReSTIR + SVGF), §1.4 (golden images are the
oracle), §1.9 item 2 (the path tracer is the second thing cut).

Crate: `es-render`, layer 5 (§4.2). It may use `es-core`, `es-math`, `es-gpu`, `es-assets`,
`es-sensor` and nothing above. In particular it cannot see `es-ir` (layer 6), so the full
§7.2 `ImageSpec` is *not* available here — see [§2.2](#22-imagespec-at-layer-5).

## 0. What this is and what it is not

`es-render` produces the §15.1 channel contract for many cameras at once, on the GPU, with no
host round trip inside a frame. Two render paths, both **compute shaders**:

| path | what it is | status |
|---|---|---|
| `Rs` | screen-space scan rasterizer | the §15.3 default for vision learning |
| `Pt` | path tracer, optional ReSTIR DI and SVGF passes | §1.9 item 2, cuttable |

`es-gpu` offers compute pipelines only — no graphics pipeline, no `VK_KHR_ray_tracing_
pipeline`, no acceleration-structure extension (see `docs/design/gpu-foundation.md`). So the
§15.4 TLAS/BLAS design is **not implemented**: both paths walk a flat triangle array in index
order. That is the honest ceiling of this packet and it is why the scene size this crate is
tested at is "a few hundred triangles", not a robot cell.

Not here, deliberately:

- **Splats.** §16.2 places 3DGS as a third render path in this crate. `es-splat` (layer 5,
  same layer, no dependency either way) owns the asset side today. The hook for later is
  `RenderPath`: a `Splat` variant lands next to `Rs`/`Pt` and writes the same channels into
  the same atlas. Nothing in the atlas, channel or `Atlas::read_tile` API assumes triangles.
- **Sensor realism** (§18.3: distortion, rolling shutter, motion blur, exposure, shot noise,
  depth holes). The renderer emits the clean channels; realism is a later pass over the
  atlas. Consequently only `DistortionModel::None` and `ShutterModel::Global` of §7.2 are
  honoured, and the renderer *rejects* a view that asks for anything else rather than
  silently ignoring it.
- **Optical flow and velocity channels.** `Channel::Flow` needs a previous-frame transform
  per primitive; the renderer has no notion of a previous frame yet. Asking for it is an
  error, not an empty buffer.
- **Textures and materials.** One flat albedo per geom, from `Geom::rgba`. No UVs, no
  texture sampling, no PBR.

## 1. Tile atlas (§15.2)

### 1.1 Layout

`maxMultiviewViewCount` is 32 on desktop GPUs, so hundreds of cameras cannot be multiview
(§15.2). Every camera gets one **tile**; tiles are packed `tiles_per_row` wide in row-major
order into one atlas per channel:

```
TileAtlasCfg { tile_w, tile_h, tiles_per_row }

rows          = ceil(n_tiles / tiles_per_row)
atlas_w       = tiles_per_row * tile_w
atlas_h       = rows          * tile_h
tile_origin(i) = ( (i % tiles_per_row) * tile_w,
                   (i / tiles_per_row) * tile_h )
```

`tile_origin` is a closed form, not a table lookup: the layout is deterministic, so a
downstream consumer reconstructs a per-env tensor by arithmetic with no host transfer
(§15.2). The last row is padded when `n_tiles` does not divide evenly; padded tiles are
cleared to the channel's background value and are never read by `read_tile`.

One atlas **per channel**, not one interleaved RGBA+depth atlas. Reason: the channels of
§15.1 have different element types (`u8` ×3, `f32` ×1, `u32` ×1, `f32` ×3) and a compute
shader writes `StructuredBuffer<T>`, so interleaving would cost a bitcast layer and a packing
rule for no gain. `Atlas` therefore holds `BTreeMap<Channel, Buffer>`.

Constraint check (§15.2): `atlas_w` and `atlas_h` must both be ≤ `maxImageDimension2D`
(16384 on the desktop GPUs §15.2 names). The atlas is a buffer here, not a `VkImage`, so the
limit does not bind physically — it is still enforced, because `es-compile`'s budget model
enforces it and a layout this crate accepts but the budget rejects would be a lie.

### 1.2 `atlas_bytes()`

The formula is a **mirror** of the `render_tile_atlas` item in
`crates/es-compile/src/budget.rs` (§20.2). `es-compile` is layer 7 and cannot be depended on
from layer 5, so the arithmetic is duplicated and pinned by a test:

```
atlas_bytes(cfg, n_tiles, channel)
  = rows * tiles_per_row * tile_w * tile_h * components * dtype_bytes * 2
                                                                       ^ double buffer
```

with `components = Channel::n_components()` and `dtype_bytes` from `Channel::dtype()`. The
`* 2` is the double-buffer factor `budget.rs` applies.

`Renderer` does **not** allocate that: it allocates `device_bytes()`, single-buffered and one
32-bit word per component, because a compute shader writes word-addressed storage buffers.
`Rgb8` is therefore one packed `RGBA8` word per pixel on the device (a shader cannot write
three bytes of a word without racing its neighbour) and the unused alpha is dropped in
`Atlas::read_tile`. So `device_bytes` is not `atlas_bytes / 2`, and both names exist so that
neither number is quietly reported as the other: reporting `atlas_bytes` as the allocation
would overstate it, reporting `device_bytes` to the budget would understate the
steady-state plan.

Reference figure to check against §15.2: 512 tiles of 224×224 `Rgb8`, `tiles_per_row = 23`
→ rows 23, atlas 5152×5152, `atlas_bytes` = 23·23·224·224·3·1·2 = 159,258,624 B ≈ 152 MiB
(the spec's 87 MB is the single-buffered 512-tile figure, 5376×5376 padded differently; both
are the same formula at different `tiles_per_row`).

## 2. Inputs

### 2.1 Scene

`Renderer::upload_scene(&SceneDesc)` walks bodies in `SceneDesc` order, composes each body's
world pose from its parent chain, and tessellates every `Geom` into world-space triangles:

| `Shape` | tessellation |
|---|---|
| `Box` | 12 triangles |
| `Plane` | 2 triangles over the finite half-extents; an infinite plane (`half_x == 0`) uses a fixed 100 m half-extent |
| `Sphere` | UV sphere, fixed 16×8 |
| `Capsule` | cylinder 16 segments + two 16×4 hemisphere caps |
| `Cylinder`, `Ellipsoid` | as `Capsule`/`Sphere` with the respective radii |
| `Mesh` | `RenderError::Unsupported` — `SceneDesc` carries an `AssetRef`, not vertices; wiring `es_assets::gltf::MeshData` in needs an asset resolver this packet does not own |
| `HeightField` | `RenderError::Unsupported` |

Tessellation counts are constants, not a quality setting: a changed count changes every
golden, so it must be a deliberate edit, not a knob.

Each triangle carries: 3 world positions (f32), a world geometric normal (f32×3, from the
triangle winding), an albedo (f32×3, `Geom::rgba` RGB), an emission (f32×3, non-zero only for
geoms whose name ends in `_light` — see [§4.2](#42-restir-di)), and a segmentation id (u32).
The segmentation id is `1 + the geom's index in traversal order`; `0` means background. Ids
are dense and start at 1 so that `0` is unambiguously "no hit" in the `SegmentationId`
channel.

Buffer layout is a flat `f32` array, stride 20 floats (80 B) per triangle — `v0 v1 v2 n
albedo emission seg pad` — the segmentation id `asuint`-bitcast into slot 18. One buffer, one
stride, the same on both sides.

### 2.2 `ImageSpec` at layer 5

§7.2's `ImageSpec` lives in `es-ir` (layer 6). `es-render` defines its own **subset**,
`es_render::ImageSpec`, carrying exactly the fields the renderer honours:

```rust
pub struct ImageSpec {
    pub width: u32,
    pub height: u32,
    pub intrinsics: Intrinsics, // fx, fy, cx, cy — no skew
    pub near: f32,
    pub far: f32,
}
```

`color_space` is fixed to `SRgb` for `Rgb8` and `Linear` for `RgbF32Linear`/`PtRadiance`
(§3.1: sRGB is the default and a linear conversion is an explicit node, never implicit).
`camera_model` is fixed to `Pinhole`, `distortion` to `None`, `shutter` to `Global`. Those are
not fields because a field implies a choice, and there is no choice here; the *Observation IR*
`ImageSpec` that `es-ir` attaches downstream keeps the full set, and
`ImageSpec::to_contract()` produces the `es_sensor::CameraContract` that carries the timing
half of §7.2 across the boundary.

The renderer validates `width == tile_w && height == tile_h` for every view. A view whose
resolution differs from the tile is an error, not a resample: a silent resample would change
intrinsics (§7.2 `OBS-034`, INV-14) and this crate has no `ImageSpec::resized` to do it
correctly.

### 2.3 Camera convention (§3.1)

`CameraView { pose, spec }`. `pose` is `T_world_camera`. The camera frame is OpenCV: **+Z
forward, +X right, +Y down**. Image origin is top-left, x right, y down. The primary ray for
pixel `(px, py)` is, in camera space,

```
d_cam = ( (px + 0.5 - cx) / fx,
          (py + 0.5 - cy) / fy,
          1 )                         // not normalised
```

and in world space `o = pose.position`, `d = pose.orientation.rotate(d_cam)`. `d_cam.z == 1`
exactly, so the ray parameter `t` **is** the camera-space depth in metres — no division, no
separate depth reconstruction, and `Depth32`'s `unit_m` is `1.0`.

§3.3 asks for camera-relative coordinates. The renderer takes them literally: triangle
positions are uploaded in world space as `f64`→`f32` once, and the shader subtracts the
camera origin before intersecting. Vertices far from the world origin therefore lose no more
f32 mantissa than their distance *from the camera* costs.

## 3. `Rs` — compute rasterizer

One thread per atlas pixel; `[numthreads(8, 8, 1)]`; the dispatch covers the whole atlas so
padded tiles are cleared by the same kernel. Per pixel:

1. Read the view index from the tile coordinate (`tile_origin` inverted), then the view's
   pose and intrinsics from a per-view parameter buffer.
2. Build the primary ray as in §2.3.
3. **Scan** triangles `0..n_tri` in ascending index order. For each: Möller–Trumbore
   intersection; reject `t <= near` or `t >= far` (this is the near/far clip); keep the hit
   with the smallest `t`, ties broken by the lower triangle index.
4. Shade and write the four channels.

Three things about that loop:

- **It has no binning stage.** `ponytail:` O(pixels × triangles) with no spatial structure —
  the ceiling is a few hundred triangles per frame. The upgrade path is the §15.4 TLAS, which
  needs a Vulkan extension `es-gpu` does not expose yet; a per-tile triangle-bin compaction
  pass is the intermediate step and slots in between (2) and (3) without changing any output.
- **The traversal order is the output's identity.** Ascending index, strict `<` on `t`, is a
  total order over hits, so the winner does not depend on thread scheduling. There is no depth
  buffer and no read-modify-write, hence no atomics (§3.4 forbids FP atomics anyway).
- **Coverage and depth come from the same ray-triangle intersection the path tracer uses.**
  Not from screen-space edge functions and interpolated `1/z`. This is the whole reason the
  §15.3 requirement "depth, seg and normal are bit-identical between RS and PT" holds *by
  construction* rather than to a tolerance — see [§6](#6-pt--rs-channel-agreement-14).
  Calling it a rasterizer is then a statement about the *loop structure* (one thread per
  pixel, scan the primitives, depth test) rather than about a fixed-function triangle setup.

### 3.1 Shading

Flat + Lambert, one directional light `L` (a `RenderConfig` field, world-space direction *to*
the light, plus an ambient term):

```
n        = geometric normal, flipped to face the ray
lambert  = ambient + max(0, dot(n, L)) * (1 - ambient)
rgb_lin  = albedo * lambert
```

No shadow ray in `Rs` — a shadow ray is a second scan of the whole triangle array per pixel,
which doubles the cost of the default vision path for an effect the `Pt` path models properly.
`ponytail:` no shadows in `Rs`; add a shadow scan when a golden shows the missing contact
shadow actually hurts a policy.

`Rgb8` is `rgb_lin` encoded with the **exact piecewise sRGB transfer** (§3.1: sRGB is the
default colour space):

```
srgb(c) = 12.92 * c                              c <= 0.0031308
        = 1.055 * c^(1/2.4) - 0.055              otherwise
u8      = round_half_away_from_zero(255 * clamp(srgb, 0, 1))
```

`c^(1/2.4)` is `es_exp(es_ln(c) * (1/2.4))` on both sides — `es_math::approx` on the CPU,
`approx.slang` on the GPU (§3.2 `DET-010`: no `std`/`GLSL.std.450` transcendental on an
observation path). §28.7 gate 3 already proves those two agree bit for bit, which is what
makes `Rgb8` bit-comparable against a CPU-generated golden at all.

### 3.2 Channels written

| channel | value | background |
|---|---|---|
| `Rgb8` | the above, 3 × u8 | `0, 0, 0` |
| `Depth32 { unit_m: 1.0 }` | `t` in metres, camera space | `far` |
| `SegmentationId` | geom id, 1-based | `0` |
| `Normal` | face normal **in camera space**, unit length (`Channel::Normal` is documented as camera-space), 3 × f32 | `0, 0, 0` |

A `RenderConfig` that asks for a channel outside this set (`Flow`, `RgbF32Linear` on the `Rs`
path, `PtRadiance` on the `Rs` path) is `RenderError::UnsupportedChannel`.

## 4. `Pt` — compute path tracer (§28.6, §1.9 item 2)

`Pt { spp, bounces, restir, svgf }`. One thread per pixel, `spp` samples, `bounces` non-
specular bounces, all diffuse. Writes `PtRadiance` (linear f32×3) plus the same
`Depth32`/`SegmentationId`/`Normal` channels from the **primary hit of sample 0** — the ones
§15.3 requires to match `Rs`.

The estimator, per sample:

```
throughput = 1
radiance   = 0
ray        = primary
for bounce in 0..bounces:
    hit = scan(ray)                       // same routine as Rs
    if !hit: radiance += throughput * sky; break
    radiance += throughput * emission(hit)
    throughput *= albedo(hit)             // cosine-weighted sampling cancels
                                          // the cos term and the 1/pi BRDF
    ray = cosine_hemisphere(n, rng)
radiance /= spp                           // accumulated in a fixed sample order
```

Cosine-weighted hemisphere sampling makes the diffuse throughput update a plain multiply by
albedo, so there is no division by a pdf and no chance of a 0/0. Russian roulette is **not**
used: it would make the work per pixel data-dependent and the bounce count is fixed and small
anyway.

`spp` accumulation is a plain sequential `+=` in ascending sample index — not a tree, not a
`DeterministicAcc`. §18.4's binned accumulator is for reductions whose *order* is not fixed;
here it is fixed by the loop, so the cheap thing is also the reproducible thing.

### 4.1 RNG

Counter-based, addressed not stepped, exactly the design of `es_env::rng` (§3.4: no global
RNG): the value is a pure function of its coordinates.

```
key(view, px, py, sample, bounce, stream)
    = mix32( mix32( mix32( mix32( mix32(seed ^ view) ^ (px*73856093 ^ py*19349663) )
                           ^ sample ) ^ bounce ) ^ stream )
```

`mix32` is the 32-bit finalizer `(z ^= z>>16) *= 0x85eb_ca6b; (z ^= z>>13) *= 0xc2b2_ae35;
z ^= z>>16`. **32-bit, not `es_env::rng`'s splitmix64**: Slang's `uint64_t` needs
`shaderInt64`, which §3.3 does not guarantee on every target this crate must run on (MoltenVK
in particular), and a 32-bit mixer is bit-identical between Rust and Slang with no capability
check at all. Statistical quality beyond "decorrelated enough for a fixed-bounce diffuse path
tracer" is `unverified`; it is not used for anything but rendering.

`f32` in `[0, 1)` is `(x >> 8) as f32 * (1 / 16777216)` — exact on both sides, never 1.0.

### 4.2 ReSTIR DI

A textbook ReSTIR DI, on by `Pt { restir: true }`, off by default. Three passes, three
dispatches, one reservoir buffer per pass (no in-place update, so no read/write hazard and no
dependence on dispatch order within a pass):

1. **Initial candidates.** `M = 8` uniform samples over the emissive triangles (a geom whose
   name ends in `_light`; its `Geom::rgba` RGB is the emitted radiance). Weighted reservoir
   sampling on the unshadowed target function `p̂ = |albedo/π · Le · G|`, then **one** shadow
   scan on the survivor. Un-shadowed survivors keep `W = 0`.
2. **Temporal reuse**, one pass: combine with the reservoir the previous `render()` call left
   at the same pixel. No motion-vector reprojection — the previous reservoir is read at the
   *same* pixel, which is correct only for a static camera and static geometry. On the first
   `render()` the previous buffer is all-zero, so the pass is a no-op and the frame is
   spatial-only. Documented, not hidden: a moving camera gets stale reuse, visible as lag.
3. **Spatial reuse**, one pass: combine with 4 neighbours at the *fixed* offsets
   `(±3, 0), (0, ±3)`, each accepted only if its depth is within 10 % and its normal within
   25° (the standard geometric similarity test). Fixed offsets, not RNG-jittered: jitter buys
   less correlated noise and costs the ability to state that the result is a function of the
   pixel grid alone.

Skipped, and this is the list §28.6 will have to fill in: MIS weights (the reservoirs use the
biased `1/M` combination, not the GRIS/pairwise-MIS weights that make reuse unbiased), the
bias-correction visibility re-test on reuse, ReSTIR **GI** entirely (this is direct lighting
only; indirect bounces go through the plain path-traced estimator above), multiple light
types (emissive triangles only — no environment map, no analytic lights), and any sort of
reservoir ageing beyond `M`-clamping at 20.

### 4.3 SVGF

On by `Pt { svgf: true }`, off by default. **One** à-trous pass set, `n` iterations
(`svgf_iterations`, default 4), the standard 5-tap B-spline wavelet kernel
`(1, 4, 6, 4, 1)/16` separated over 2D with stride `1 << i`, edge-stopping on depth and
normal:

```
w = w_depth * w_normal
w_depth  = exp(-|z_p - z_q| / (sigma_z * |grad z| + eps))
w_normal = max(0, dot(n_p, n_q))^sigma_n
```

Skipped: the **V** in SVGF. There is no temporal accumulation of colour, no per-pixel variance
estimate, no variance-guided `w_luminance` term, no 7×7 variance prefilter for low-sample
regions, no disocclusion handling, no history-length-driven kernel widening. What is left is
an edge-aware à-trous filter — a real and useful denoiser, and not the algorithm in the SVGF
paper. It is named `svgf` because §28.6 names it; the doc comment on the kernel says the same
thing this paragraph does.

## 5. CPU references (§1.4)

The oracle is not the GPU. `es_render::cpu` is pure Rust, no `es-gpu`, no `unsafe`, and it is
what generates every golden in `tests/golden/render/`.

```rust
pub fn rasterize(scene: &TriScene, view: &CameraView, cfg: &RenderConfig) -> CpuFrame
pub fn path_trace(scene: &TriScene, view: &CameraView, cfg: &RenderConfig) -> CpuFrame
```

`CpuFrame` holds one `Vec` per channel plus the tile shape. Both functions are *the same
algorithm* as the shaders, expression for expression and in the same order — same
intersection routine, same traversal order, same `es_math::approx` calls, same accumulation
order. Where the two texts diverge the Slang is wrong, not the Rust.

This is why the goldens are generated by the CPU path **once** and never by the GPU path: a
golden produced on an RTX 4060 would bake that driver's arithmetic into the repository and the
next device would "fail" a test that is really a device difference. Golden files are CI
read-only (§1.4); `xtask verify-goldens` enforces it.

Scene for the goldens: a Cornell box built in Rust (`es_render::cornell`) — five box walls
(white, red left, green right), two boxes inside, one emissive ceiling panel
`ceiling_light`, 64×64 tiles, one camera, one directional light. Eight geoms, 96 triangles.

**No two faces in that scene are coplanar**, and the two inner boxes are sunk into the floor.
That is a correctness requirement, not styling: where two triangles share a plane, a ray hits
both at exactly the same `t` and the winner is decided by which side of `u + v <= 1` the
barycentrics fall on — an answer the CPU and the GPU can give differently by one ULP. With
walls that merely abutted the floor and ceiling, 9 of 4096 pixels picked a different face on
the GPU (a whole-pixel colour difference, not a rounding one) while depth and segmentation
still agreed. Overlapping the walls into the floor and ceiling removed all nine.

| golden | channel | dtype | shape |
|---|---|---|---|
| `cornell_rs_rgb8` | `Rgb8` | u8 | 64×64×3 |
| `cornell_rs_depth` | `Depth32` | f32 | 64×64 |
| `cornell_rs_seg` | `SegmentationId` | u32 | 64×64 |
| `cornell_pt1spp` | `PtRadiance` | f32 | 64×64×3 |

`cornell_pt1spp` is 1 spp, 2 bounces, ReSTIR and SVGF off — the smallest thing that exercises
the RNG, the bounce loop and the emissive hit, and the only PT configuration a golden can pin
bit-exactly without also pinning the noise of 64 samples.

### 5.1 Tolerances

Claimed tolerance, and what was **measured** on an NVIDIA RTX 4060 Laptop GPU (driver
592.82, Slang 2026.8) at 64×64:

| comparison | claim | measured |
|---|---|---|
| CPU `Rs`/`Pt` vs golden | bit-identical (it generated them) | bit-identical, all 4 goldens |
| GPU `Rs` `Rgb8` vs golden | bit-identical | 0 of 12288 bytes differ |
| GPU `Rs` `SegmentationId` vs golden | bit-identical | bit-identical |
| GPU `Rs` `Depth32` vs golden | ≤ 1 ULP | **0 ULP** |
| GPU `Rs` `Normal` vs golden | ≤ 1 ULP | **0 ULP** |
| GPU `Pt` 1 spp vs CPU `Pt` 1 spp | bit-identical | 0 ULP |
| GPU `Pt` + SVGF vs CPU | bit-identical | 0 ULP |
| GPU `Rs` run A vs run B | bit-identical, every channel | bit-identical |
| GPU `Pt` + ReSTIR (+ SVGF) vs CPU | ≤ 1e-5 normalized | **1.3e-7 normalized, max 10 ULP** |

`Depth32` and `Normal` are allowed 1 ULP rather than 0 because the Möller–Trumbore divide is
the one place the two texts can legitimately differ — `(1/det) * x` on one side and `x / det`
on the other is a compiler choice, not a source difference. This device happens to give 0.
The tests print the measured ULP, so a regression from 0 to 1 is visible even though 1 passes.

**ReSTIR is the one path that is not bit-equal**, at 10 ULP / 1.3e-7 of the image peak. The
reservoir passes sum `p̂ · W · M` over neighbours, and the shader compiler is free to
reassociate within a single expression even under `NoContraction`. The consequence is worse
than the number looks: a resampling decision is the threshold `rand · w_sum < weight`, so a
one-ULP difference in a weight can select a *different light sample* at a pixel. It did not
happen on this device at this resolution, and there is nothing in the design that prevents it
— which is why the tolerance is stated against the image peak rather than per pixel (most of
a rendered image is near zero, and a 1e-7 difference on a 1e-7 pixel is a 100% relative error
for an image nobody could tell apart).

## 6. PT ↔ RS channel agreement (§1.4, §15.3)

§15.3's output contract says depth, segmentation and normal are **bit-identical** between the
render paths and only RGB is a similarity threshold. The test `pt_and_rs_agree_on_geometry`
renders the same scene through both paths and asserts exactly that, on the CPU (so it runs
everywhere) and on the GPU (so a device-specific divergence is caught too):

```
rs.depth == pt.depth        bitwise
rs.seg   == pt.seg          bitwise
rs.normal == pt.normal      bitwise
```

It holds by construction because both paths call the same `nearest_hit` over the same
triangle array in the same order for the primary ray, and `Pt` writes these three channels
from sample 0's primary hit before any bounce or RNG draw. If someone later replaces the `Rs`
scan with a real fixed-function-style rasterizer, this test is the one that will fail, and the
fix is a documented tolerance on depth — never a change to the goldens.

RGB is *not* compared between paths: `Rs` is a one-bounce analytic shade and `Pt` is a Monte
Carlo estimate of a different integral. An SSIM threshold between them (which §15.3 asks for)
needs a converged PT render and an SSIM implementation; both are `unverified` here and belong
with the §28.6 packet that makes the path tracer real.

## 7. Determinism (§3.4)

- One compute queue, one recorder, a full barrier after every dispatch — all inherited from
  `es-gpu`, none of it optional here.
- No atomics anywhere. No shared memory. No subgroup operations. No dependence on the
  workgroup count: every kernel indexes from `SV_DispatchThreadID` alone and bounds-checks.
- Every transcendental goes through `approx.slang` / `es_math::approx`.
- `ExecModes` come from `Gpu::deterministic_execution_modes()` and are passed to every
  `SlangCompiler::compile_file` call, so `NoContraction` and the §3.4 step-3 execution modes
  are in every module.
- No `HashMap`. `BTreeMap` for the channel map, `Vec` indexed by view for everything else.
- The RNG is addressed by `(view, pixel, sample, bounce, stream)`, so sample order, thread
  order and dispatch shape cannot change a draw.

What this does **not** buy: bit equality *across devices*. §3.5 tier 1 needs the same
`hardware_capability` in the execution hash, and the `Depth32` 1-ULP tolerance above is
already an admission that the GPU and the CPU are two different implementations of the same
text. The claim this crate makes is: same device, same binary → same bits; and CPU → GPU
agreement at the stated tolerances.
