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
§15.4 TLAS/BLAS design is **not implemented** as hardware. Since M7/R1 both paths traverse a
software BVH built on the CPU per frame (section 8); before it they walked a flat triangle
array in index order, which is why the goldens are "a few hundred triangles" scenes and why
the hit rule is stated as a scan's.

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

The curved shapes' vertex directions are computed in `f32` through `es_math::approx::{sin,
cos}`, never the host `libm` (§3.2 `DET-010`), then widened exactly to `f64` and scaled by the
`f64` radii. The tessellation runs on the CPU for **both** paths — `Renderer::upload_tris`
uploads `TriScene::to_floats()` and `es_render::cpu` traverses the same `TriScene` — so a
host-dependent `sin` here would hand the GPU different geometry than the reference on a
different machine, silently. The longitude angle folds `seg == SPHERE_SEGMENTS` onto segment
0 (`phi_of`) so the seam closes on identical bits rather than on a `sin(2π)` residue, which in
`f32` is ~1e-7 rather than the ~1e-16 of the `f64` version it replaced. The unit test
`curved_shapes_upload_the_vertices_the_cpu_reference_traverses` asserts the upload buffer
carries the CPU's vertices bit for bit, and that a repeat tessellation reproduces them.

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

`Intrinsics::from_fovy` is on the golden camera path: `fx`/`fy` feed the per-view parameter
buffer, and every golden pixel is a function of them. It therefore takes the `f64` `fovy` the
asset stores, casts once, and does the rest in `f32` through `es_math::approx::tan` in a fixed
order — halve, `tan`, divide. The host `libm` `tan` it replaced was not pinned by anything
(review M4 S-11); moving to `approx::tan` shifted `fy` at 64×64 / `fovy = 1.2` by exactly 1
ULP (`46.774269` → `46.774265`), which is why the `Rs` goldens were regenerated — see
[§5](#5-cpu-references-14).

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

- **The scan is now a BVH traversal** (M7/R1, section 8) that returns exactly the scan's
  answer: nearest `t`, ties to the lower global index, compared on every candidate. Step (3)
  above is the *rule*; the tree is how the candidates are found. The §15.4 two-level TLAS is
  the upgrade path and needs no Vulkan extension either.
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

This is `Shading::Lambert`, the `#[derive(Default)]` default, and it is what every golden and
every committed observation document pins, byte for byte. No shadow ray: when this was the
only shading, a shadow ray meant a second scan of the whole triangle array per pixel, which
doubled the cost of the default vision path for an effect the `Pt` path models properly.

Since M7/R2 there is a second, **opt-in** shading — `Shading::Full`, one shadow ray through
R1's any-hit traversal, a hemisphere ambient, a Blinn-Phong highlight and supersampling:

```
n        = geometric normal, flipped to face the ray
vis      = shadows ? (any_hit(p + n * RAY_EPS, L, 0, SHADOW_FAR) ? 0 : 1) : 1
diffuse  = max(0, dot(n, L)) * vis
h        = normalize(L - d)
spec     = specular * pow(max(0, dot(n, h)), shininess) * vis      // 0 when dot(n, h) <= 0
hemi     = ground + (sky - ground) * ((n.z + 1) * 0.5)
rgb_lin  = albedo * (hemi + diffuse * (1 - hemi)) + spec + emission   // per channel
```

then the same sRGB transfer below. `pow` is `es_exp(es_ln(x) * shininess)`, never `std` or
`GLSL.std.450`. Nothing about `Lambert` moved to make room for it: one kernel branches on a
flag read from the parameter buffer, and the new slots were appended after the existing ones.
[§9](#9-the-rs-look-m7r2) has the whole of it — the accumulation order of the box filter, the
parameter layout, why the observation path was not given the option, and what the GPU and the
CPU do and do not agree about.

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

`Pt { spp, bounces, nee, restir, svgf }`. One thread per pixel, `spp` samples, `bounces` non-
specular bounces, all diffuse. Writes `PtRadiance` (linear f32×3), the tone-mapped `Rgb8` of
[§10](#10-the-path-tracer-grows-up-m7r3), and the same `Depth32`/`SegmentationId`/`Normal`
channels from the **primary hit of sample 0** — the ones §15.3 requires to match `Rs`.

The estimator, per sample. `nee` is `false` by default and the whole right-hand column is
then a multiply by `1.0`, which is why `cornell_pt1spp` is byte-identical either way:

```
throughput = 1
radiance   = 0
ray        = primary
prev_pdf   = 0                            // the camera ray: no light strategy made it
for bounce in 0..bounces:
    hit = scan(ray)                       // same routine as Rs
    if !hit:
        radiance += throughput * sky * w_sky      // w_sky = 1, or 1/2 under nee
        break
    radiance += throughput * emission(hit) * w_bsdf   // w_bsdf = 1, or the power
                                                      // heuristic vs the light pdf
    n = face_forward(hit)
    p = hit_point + n * RAY_EPS
    if nee:
        radiance += throughput * nee_direct(p, n, albedo, last = bounce + 1 == bounces)
    throughput *= albedo(hit)             // cosine-weighted sampling cancels
                                          // the cos term and the 1/pi BRDF
    ray = cosine_hemisphere(n, rng)
    prev_pdf = dot(n, ray) / pi
radiance /= spp                           // accumulated in a fixed sample order
```

Cosine-weighted hemisphere sampling makes the diffuse throughput update a plain multiply by
albedo, so there is no division by a pdf and no chance of a 0/0. Russian roulette is **not**
used: it would make the work per pixel data-dependent and the bounce count is fixed and small
anyway — and neither does NEE introduce one: its three light kinds are three `if`s on
*config*, uniform across the dispatch, so every pixel of a given render does the same work.

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

**The stream table.** `stream` is what separates independent uses at the same coordinates,
and a new use takes a **new id** rather than reusing one at a different `index`: reusing one
would silently correlate two draws that the estimator assumes are independent. The table is
pinned in `crates/es-render/src/rng.rs` and here, and both are the contract:

| id | use | drawn per |
|---|---|---|
| 0 | the cosine-weighted BSDF bounce direction | sample, bounce |
| 1 | `ReSTIR` initial candidates (light pick, area sample, reservoir accept) | pixel |
| 2 | `ReSTIR` spatial reuse (the reservoir accept per neighbour) | pixel |
| 3 | `ReSTIR` temporal reuse (the reservoir accept) | pixel |
| 4 | NEE: which emissive triangle (M7/R3) | sample, bounce |
| 5 | NEE: the uniform point on that triangle | sample, bounce |
| 6 | NEE: the cosine-weighted sky direction | sample, bounce |

The NEE shadow test has no stream: it draws no random number. Streams 4–6 are only drawn
when `nee` is on, so a `nee: false` render makes exactly the draws it made before M7/R3.

### 4.2 ReSTIR DI

A textbook ReSTIR DI, on by `Pt { restir: true }`, off by default. Three passes, three
dispatches, one reservoir buffer per pass (no in-place update, so no read/write hazard and no
dependence on dispatch order within a pass):

1. **Initial candidates.** `M = 8` uniform samples over the emissive triangles (a geom whose
   name ends in `_light`; its `Geom::rgba` RGB is the emitted radiance). Weighted reservoir
   sampling on the unshadowed target function `p̂ = |albedo/π · Le · G|`. **No shadow ray
   since M7/R3**: visibility is tested once, in the spatial pass, on the sample that pass
   actually selects — see [§10.2](#102-restir-is-unbiased-now-and-here-is-the-number).
2. **Temporal reuse**, one pass: combine with the reservoir the previous `render()` call left
   at the same pixel. No motion-vector reprojection — the previous reservoir is read at the
   *same* pixel, which is correct only for a static camera and static geometry. On the first
   `render()` the previous buffer is all-zero, so the pass is a no-op and the frame is
   spatial-only. Documented, not hidden: a moving camera gets stale reuse, visible as lag.
3. **Spatial reuse**, one pass: combine with 4 neighbours at the *fixed* offsets
   `(±3, 0), (0, ±3)`, each accepted only if its depth is within 10 % and its normal within
   25° (the standard geometric similarity test). Fixed offsets, not RNG-jittered: jitter buys
   less correlated noise and costs the ability to state that the result is a function of the
   pixel grid alone. The combination is **pairwise MIS** (M7/R3), not `1/M`, and the pass
   ends with one shadow ray towards the sample it selected.

Skipped, still: ReSTIR **GI** entirely (this is direct lighting only; indirect bounces go
through the path-traced estimator above), light types other than emissive triangles inside
the reservoirs (the NEE path of [§10.1](#101-next-event-estimation-and-mis) has three, the
reservoirs have one — no environment map, no analytic light in a reservoir), and any sort of
reservoir ageing beyond `M`-clamping at 20. What is **no longer** skipped, and what §28.6
asked for: the MIS weights that make reuse unbiased, and the bias-correction placement of
the visibility test. [§10.2](#102-restir-is-unbiased-now-and-here-is-the-number) has the
before/after number.

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

Regenerated once since they were first written, for review M4 S-11 (`Intrinsics::from_fovy`
moving from the host `tan` to `es_math::approx::tan`, [§2.3](#23-camera-convention-31)). The
1 ULP shift in `fy` is a sub-pixel change of the ray directions, so the diff is confined to
silhouette edges:

| golden | diff |
|---|---|
| `cornell_rs_rgb8` | 57 of 12288 bytes differ (edge pixels, so up to 183 on a red-wall/white-box boundary) |
| `cornell_rs_depth` | 1604 of 4096 floats differ, max abs 4.05e-6 m, max 17 ULP |
| `cornell_rs_seg` | 19 of 4096 pixels take the neighbouring geom |
| `cornell_pt1spp` | **byte-identical** |

The GPU still matches the regenerated files exactly (`Rgb8`/`SegmentationId` bit-equal,
`Depth32`/`Normal` 0 ULP on an RTX 4060), which is the point: the shift is in the shared
input, not in either path's arithmetic.

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
needs a converged PT render and an SSIM implementation. M7/R3 built both —
`es_render::ssim` and a tone-mapped `Rgb8` from the `Pt` path — and
[§10.4](#104-the-1531-ssim-number-finally-exists-and-it-is-low) is the measurement. The
**threshold is still not set**: the number turns out to say more about the two lighting
models than about either renderer, and choosing a threshold from it is an owner decision, not
a packet's.

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

## 8. Where the frame goes, and the software BVH (M7/R1)

`docs/packets/M7/R1-bvh.md`. §28.10 records **80 ms/frame** at 1280×720 for the SO-101 demo
cell (`visible-learning.md` 7.17 item 4, one RTX 4090) and names two suspects: the scene is
re-tessellated and re-uploaded into a freshly allocated buffer every frame, and the shader
scans every triangle per pixel. Neither was measured. §1.4 says measure first.

### 8.1 The baseline, measured

`cargo test -p es-render --release -- --ignored --nocapture frame_profile` times four phases
of one frame — tessellate, upload, dispatch+wait, readback — over 100 frames on the demo
scene (`tests/fixtures/mjcf/so101_pick_place.xml`, **2,754** triangles after tessellation;
§28.10's "2,978" is the earlier count) through `es video showcase`'s own camera
(`--eye 0.66,-0.46,0.52 --look-at 0.14,-0.04,0.04 --fov 36`). `upload` is
`Renderer::upload_tris`; `dispatch+wait` is `Renderer::render`, which also uploads the
parameter buffer and blocks on the fence; `readback` is `Atlas::read_tile`.

**Before, RTX 3060 (driver 591.12, Slang 2026.8), median / p95 ms:**

| phase | 1280×720 | 96×96 |
|---|---|---|
| tessellate | 0.232 / 0.326 | 0.255 / 0.321 |
| upload | 0.334 / 0.490 | 0.203 / 0.345 |
| dispatch+wait | **38.633 / 41.374** | 0.913 / 1.354 |
| readback | **83.575 / 86.850** | 1.135 / 1.311 |
| frame total | 122.856 / 141.384 | 2.512 / 3.106 |
| `camera_frames_per_sec` (median) | 8.1 | 398.1 |
| `pixels_per_sec` (median) | 7.50e6 | 3.67e6 |

Two conclusions, and one of them is not this packet's.

1. **The scan is 38.6 ms of the 1280×720 frame** — 921,600 pixels × 2,754 triangles = 2.5 G
   Möller–Trumbore tests. That is what a BVH removes, and it is the only one of the two
   suspects §28.10 named that the measurement supports.
2. **The readback is 83.6 ms, and it is not the renderer.** The same test reads a
   host-visible buffer of the same size directly: **80.7 ms for 3,600 KiB, 44 MiB/s**.
   `es_gpu::Buffer::download` on a device-local buffer allocates a `Usage::Staging`
   (`MemoryLocation::CpuToGpu`) buffer, copies into it and then `to_vec()`s it — and
   CpuToGpu memory is *write-combined*, so the host read is uncached. The fix is a
   `GpuToCpu` readback buffer in `es-gpu`, which `crates/es-gpu/**` being out of this
   packet's scope puts in the next one. Recorded here so it is not re-discovered.
   `Target / Status: unverified` for any 1280×720 frame-total target below ~84 ms until that
   lands.

Tessellation is 0.23 ms and the upload 0.33 ms: together 0.5 % of the frame. They are still
cached and made persistent, because they are cheap to fix and because at 96×96 — the size
every observation frame in an evaluation is rendered at — 0.46 ms of a 2.5 ms frame is 18 %.

### 8.2 What changed, and why not one output bit did

Three edits, each pinned by a test that compares against the code it replaced.

1. **`SceneCache`** (`scene.rs`) keeps every geom's *local* tessellation across frames — the
   `approx::{sin, cos}` vertices `tessellate()` produces — and applies the pose per frame in
   `f64` through the same `push_geom` as before: the same `pose.transform_point`, the same
   `to_f32`, the same `face_normal`, the same push order. `cached_tessellation_is_bit_identical`
   compares the cached and the uncached `TriScene` with `PartialEq` on Cornell and on the
   SO-101 scene at three poses. `TriScene::from_scene_with_poses` is now one call into an
   empty cache, so no caller changed meaning.
2. **Persistent buffers** (`renderer.rs`): the triangle+BVH buffer, the params buffer and the
   per-pixel `hit_tri` buffer are kept across frames and grown, never shrunk (`grow`). The
   four channel buffers stay per frame because `Atlas` owns them and hands them to the caller.
3. **A single-level BVH** (`bvh.rs`; `es_nearest` / `es_any_hit` in `common.slang`;
   `nearest_hit` / `any_hit` in `cpu.rs`): median split on the centroid along the longest
   axis of the centroid bounds, leaves of ≤ 4 triangles, nodes in a flat array uploaded
   *after* the triangles in the same buffer (slots 17–19 of `params` carry the offsets, so no
   descriptor moved). The traversal is a fixed-order descent with a 64-entry stack; the build
   bounds the depth (11 for the 2,754-triangle scene, 6 for Cornell — printed by the test).

**Why the BVH's answer is the flat scan's answer.** The flat scan's hit rule is "nearest `t`
strictly inside `(near, far)`, ties to the lower global triangle index". The traversal keeps
that rule literally — it compares `(t, index)` on every candidate, never trusting the visit
order — so it returns the scan's winner as long as it *visits* every triangle the scan would
accept. Two details make that certain rather than likely: every node box is padded outward
by `PAD = 1e-4` m, because Möller–Trumbore can accept a ray that passes a hair outside a
triangle's exact bounds; and the slab test uses Ize's robust factors (`1 ∓ 2^-22`) so its
own rounding never rejects a node the ray enters. Both only *add* candidates, never remove
one, and an extra candidate cannot change a `(t, index)` minimum. The intersection routine
itself is untouched, so `t` is the same bits — which is why `Depth32` and `Normal` stayed at
0 ULP against the CPU on both GPUs. `bvh_traversal_is_the_flat_scan` fires 10,000 rays per
scene through both routines (the flat scan is kept as `nearest_hit_flat`): all equal, on
Cornell and on SO-101, nearest and any-hit alike. The tree enters no hash: it is a search
structure over data that is already hashed.

The two-level TLAS/BLAS of §15.4 stays the upgrade path (`ponytail:` at `Bvh::build`): the
per-frame `f64` posing is what keeps the vertices bit-identical to every committed golden,
and a 3,000-triangle rebuild costs 0.1–0.2 ms.

### 8.3 After, measured

Same test, same scene, same cameras. Medians in ms (p95 in the raw tables under
`~/artifacts/plan-v/m7-r1/` on the server; the local ones are in this section's history).

| phase, 1280×720 | RTX 3060 before | RTX 3060 after | RTX 4090 before | RTX 4090 after |
|---|---|---|---|---|
| tessellate | 0.232 | 0.200 | 0.257 | 0.084 |
| upload | 0.334 | 0.890 | 0.635 | 0.995 |
| dispatch+wait | **38.633** | **0.981** | **6.934** | **1.108** |
| readback | 83.575 | 85.082 | 64.183 | 63.830 |
| frame total | 122.856 | 87.445 | 71.871 | 65.967 |
| `render()` sustained, no readback | — | 0.925 | — | 1.076 |
| `camera_frames_per_sec`, sustained | — | 1,081 | — | 929 |
| `pixels_per_sec`, sustained | — | 9.97e8 | — | 8.56e8 |

| phase, 96×96 | RTX 3060 before | RTX 3060 after | RTX 4090 before | RTX 4090 after |
|---|---|---|---|---|
| dispatch+wait | 0.913 | 0.354 | 3.506 | 1.451 |
| readback | 1.135 | 0.808 | 1.741 | 1.510 |
| frame total | 2.512 | 2.168 | 6.372 | 4.234 |
| `camera_frames_per_sec` (median frame) | 398 | 461 | 157 | 236 |

The scan went from 38.6 ms to 1.0 ms on the 3060 and from 6.9 ms to 1.1 ms on the 4090 —
the 4090 was never scan-bound the way the laptop-class card was, which the baseline table
alone could not have told. The `upload` row grew by ~0.6 ms on both cards: it now carries the
BVH build (CPU) and the extra 2,047 nodes; a cost that is visible only because everything
around it shrank.

**The showcase, RTX 4090, V19b `nominal-00` (224 ticks, 1280×720):** 122.2 ms/frame before,
118.6 ms/frame after. The 96×96 `overhead` replay of the same cell reproduces the recorded
observation frames **224 / 224 bit-identical** through the BVH path (V9's oracle, re-run on
this branch). Nothing about the picture moved; almost nothing about the wall-clock did either,
because of what section 8.1 already said:

### 8.4 What is left is the readback, and it is `es-gpu`'s

At 1280×720 the frame is now **64–85 ms of readback around a 1 ms render**.
`es_gpu::Buffer::download` stages a device-local buffer through `Usage::Staging`, which is
`MemoryLocation::CpuToGpu` — host-visible, *write-combined*, uncached for host reads — and the
test's direct read of such a buffer runs at 27–44 MiB/s. A readback buffer in
`MemoryLocation::GpuToCpu` (host-cached) is the fix, and `crates/es-gpu/**` is outside this
packet's scope, so it is packet **M7/R1b**. Until it lands: `Target / Status: unverified` for
any 1280×720 frame-total below ~64 ms, and the §28.10 target of `< 5 ms/frame` is a target
for R1b, not a claim of R1. At 96×96 — the size every observation frame is rendered at — the
frame is already 2.2 ms on the 3060 and 4.2 ms on the 4090, of which readback is a third.

**R1b landed** (`docs/packets/M7/R1b-readback.md`, commit `7c8fe39`): `Buffer::download` now
stages through `Usage::Readback` (`MemoryLocation::GpuToCpu`, host-cached), and
`frame_profile` times that path directly (`device-local download of 3600 KiB via
Buffer::download`). Every render test and golden unchanged on both GPUs, the 96×96 replay of
V19b `nominal-00` still **224 / 224 bit-identical** to the recorded frames.

| 1280×720, median ms | RTX 3060 R1 | RTX 3060 R1b | RTX 4090 R1 | RTX 4090 R1b |
|---|---|---|---|---|
| readback | 85.082 | **7.229** | 63.830 | **3.001** |
| frame total | 87.445 | **12.586** | 65.967 | **5.207** |
| `Buffer::download`, 3,600 KiB | (86 ms path) | 1.228 (2.9 GiB/s) | (68 ms path) | 1.682 (2.1 GiB/s) |
| `camera_frames_per_sec` (median frame) | 11.4 | 79 | 15.2 | 192 |
| `pixels_per_sec` (median frame) | 1.05e7 | 7.3e7 | 1.40e7 | 1.77e8 |

| 96×96, median ms | RTX 3060 R1 | RTX 3060 R1b | RTX 4090 R1 | RTX 4090 R1b |
|---|---|---|---|---|
| readback | 0.808 | 0.285 | 1.510 | 0.446 |
| frame total | 2.168 | 4.296 (p95 15.5, a busy desktop) | 4.234 | **2.410** |

**The showcase, RTX 4090, V19b `nominal-00` at 1280×720: 122.2 → 118.6 → 6.2 ms/frame** — the
224-frame cell renders in 1.4 s where V9 measured 80 ms/frame; the 96×96 replay is 2.6 ms/frame.
The §28.10 target of `< 5 ms/frame` at 1280×720 reads **5.2 ms** on the 4090's profile — not
met, by 0.2 ms; the 3.0 ms readback that remains is mostly `Atlas::read_tile`'s host-side
unpack of the packed `RGBA8` words into `Rgb8` bytes (the copy itself is 1.7 ms), which is
where the next millisecond lives if anyone needs it. The 3060 does not reach it (12.6 ms).

## 9. The `Rs` look (M7/R2)

`docs/packets/M7/R2-rs-look.md`. §28.10 records the `Rs` path as "Lambert plus a constant
ambient; no shadows, no highlights, no textures" — honest, and it looks like 1995. The
question the packet asks is whether it can have contact shadows, a sky, highlights and
anti-aliasing **without changing a byte of what the committed observation documents render**
(§28.10 rule 1).

The answer is a `RenderConfig` field:

```rust
pub enum Shading {
    #[default] Lambert,                       // §3.1, unchanged, every golden pins it
    Full { shadows, specular, shininess, sky_rgb, ground_rgb, ssaa },
}
```

`RenderConfig::rs()` keeps `Lambert`. `RenderConfig::rs_full()` is `Shading::FULL`, the
opinionated preset: `shadows: true, specular: 0.25, shininess: 32.0, sky [0.55, 0.65, 0.85],
ground [0.25, 0.22, 0.20], ssaa: 2`. `es video showcase --look lambert|full` selects it and
defaults to `lambert`.

### 9.1 The shading, exactly

[§3.1](#31-shading) has the equations; this is what is behind them.

- **The shadow ray is R1's any-hit traversal** (`cpu::any_hit` / `es_any_hit`), from
  `p + n * RAY_EPS` towards the light on `(0, SHADOW_FAR)` with `SHADOW_FAR = 1e30`. The
  light is *directional* — infinitely far — so the occluder may stand outside the camera's
  far plane and the view frustum's `far` is the wrong bound. `1e30` rather than infinity
  because `rcp_safe` clamps reciprocals to ±1e30 to keep `0 * inf` out of the slab test, and
  an infinite `far` would put it back in.
- **`vis` multiplies the diffuse *and* the specular term, never the ambient.** A surface in
  shadow keeps its hemisphere ambient, which is what stops the shadowed floor of the Cornell
  box from going black.
- **Blinn-Phong, not Phong**: `h = normalize(L - d)`, where `d` is the view ray as the camera
  built it — not normalised, `z = 1` in camera space. Normalising `d` first would be one more
  `sqrt` per sub-sample for a highlight that is a look, not a measurement.
- **`pow` is `es_exp(es_ln(x) * shininess)`**, guarded at `x <= 0` on both sides (§3.2
  `DET-010`: no `std`, no `GLSL.std.450` on a path that generates a golden).
- **The hemisphere is world-space**: `sky` at `n.z = +1`, `ground` at `n.z = -1`, lerped on
  `(n.z + 1) * 0.5`. Written out as `ground + (sky - ground) * t` on both sides rather than
  through `lerp`/`mix`, because HLSL's `lerp(x, y, s) = x + s * (y - x)` and GLSL's
  `mix(x, y, a) = x * (1 - a) + y * a` are not the same expression and the two texts have to
  round the same way.
- **A ray that hits nothing still returns black**, not `sky_rgb`. `sky_rgb` is the *ambient*
  from above, not a background: giving the background a colour would change what
  `SegmentationId == 0` looks like and is a scene decision, not a shading one.
- **The ambient and the diffuse mix is energy-conserving** — `hemi + diffuse * (1 - hemi)`,
  per channel, the same shape `Lambert` uses for its constant ambient
  (`ambient + ndl * (1 - ambient)`). A surface in full light returns exactly its albedo and a
  shadowed one returns `albedo * hemi`, so nothing clips before the sRGB transfer.
  **Amended at review**: the packet's original `albedo * (hemi + diffuse)` exceeds the albedo
  on every lit surface — with `sky` at 0.85 and `ndl` at 0.87 the SO-101 table rendered pure
  white — and that is a defect of the equation, not something a tone map should be asked to
  hide. `spec` stays additive: a highlight is allowed to blow out, that is what a highlight
  is.

What `Full` is still not: no textures, no soft shadows (one ray, one directional light), no
multiple lights, no tone map (that is R3's), no global illumination (that is `Pt`).

**What the Cornell golden does not pin.** The box is a closed room, so every shadow ray from
inside it hits the ceiling: under the preset `vis = 0` at every pixel of
`cornell_rs_full_rgb8`, the `diffuse` term is zero, and the golden is byte-identical before
and after the amendment above. It pins the hemisphere, the shadow ray and the box filter, and
it is blind to the mix. Two tests cover what it cannot: `full_shading_is_energy_conserving`
calls `shade_full` directly and asserts the lit surface returns its albedo and the shadowed
one `albedo * hemi`, and `gpu_full_shading_matches_the_cpu` runs its GPU/CPU comparison a
second time with `shadows: false`, which is the only way the device executes the lit branch
on this scene at all.

### 9.2 SSAA is a fixed-order box filter

`ssaa: k` shades `k × k` sub-samples per pixel at `px + (sx + 0.5) / k`, and box-filters them:

```
acc = 0
for sy in 0..k:            // row-major, sy outer, sx inner, one sequential +=
    for sx in 0..k:
        acc += shade(sub_sample(sx, sy))     // a sub-sample that hits nothing adds 0
lin = acc * (1 / (k * k))  // one multiply, then the sRGB transfer
```

Spec §3.4 forbids an accumulation whose order depends on scheduling, so the order is written
into both texts and the sum is sequential in `f32` — no pairwise tree, no atomic, no shared
memory. It is **one kernel**: the supersampling happens inside the thread that owns the output
pixel, not in a bigger atlas resolved by a second pass. That is also why no atlas, buffer or
`Atlas::read_tile` shape changed: at `ssaa: 2` each thread casts 4 primary rays and 4 shadow
rays instead of 1 and 0.

`k = 1` puts the single sub-sample at `px + 0.5` — the pixel centre, bit for bit — so
`primary_dir` is now `primary_dir_sub(.., 0, 0, 1.0)` and `Lambert` did not move.

**The geometry channels are the centre ray's, under any shading.** `Depth32`,
`SegmentationId` and `Normal` come from `es_primary_dir(vw, px)` exactly as they always did,
and the tests assert they are bitwise what `Lambert` writes — on the CPU
(`cpu_full_shading_reproduces_its_golden`) and on the GPU (`gpu_full_shading_matches_the_cpu`).
This is a **deviation from the packet's wording**, which says they come from "the first
sub-sample of each block". They cannot: at `ssaa: 2` the first sub-sample of a block sits at
`px + 0.25`, and its depth is not the centre's. The centre ray is what makes the equality the
packet actually demands true, and it keeps §15.3's `RS`/`PT` channel agreement (which is
against `Pt`'s sample-0 primary hit, also a centre ray) intact for free.

### 9.3 What the GPU and the CPU agree about

Measured at 64×64 on the Cornell box, NVIDIA RTX 3060 (Slang 2026.8) and RTX 4090:

| comparison | claim | measured |
|---|---|---|
| CPU `Full` `Rgb8` vs `cornell_rs_full_rgb8` | bit-identical (it generated it) | bit-identical |
| CPU `Full` vs `Lambert`, depth/seg/normal | bit-identical | bit-identical |
| GPU `Full` vs `Lambert`, depth/seg/normal | bit-identical | bit-identical |
| GPU `Full` `Rgb8` vs CPU, preset | bit-identical **except at edge pixels** | 3 of 12,288 bytes, 1 of 4,096 pixels |
| GPU `Full` `Rgb8` vs CPU, `shadows: false` | the same | 3 of 12,288 bytes, the same pixel |

That last row is the one finding worth keeping. The differing pixel is (37, 49), where the
short box's top face and its far face meet: one of its four sub-samples grazes the shared
edge, the CPU gives the hit to the far face and the GPU to the top face, and the two faces
have different hemisphere ambients — so a quarter of the pixel's colour changes, 5 to 9 levels
of 255 (bluest, because `sky - ground` is largest in blue). It is a **coverage** tie, not a
shading difference: `dot`'s summation order is not pinned across the two implementations
(SPIR-V's `OpDot` may associate as it likes, even under `NoContraction`), and where a ray
passes within an ULP of a triangle boundary the winner of "nearest hit, ties to the lower
index" can differ. One sample per pixel never landed on such an edge in this scene; 2×2
sub-samples do, at one pixel in four thousand.

So `gpu_full_shading_matches_the_cpu` asserts the honest thing rather than a number that is
true on one device: every differing pixel must be a pixel whose four sub-samples do **not**
all hit the same triangle, must differ by at most 64 levels (255/4, the ceiling on what one
sub-sample of four can move a channel by — a backstop; the sub-sample assertion is the
discriminating one), and at most 0.1% of pixels may differ. A pixel whose four sub-samples
agree on the triangle and still differs is a shading divergence and fails the test.

The alternative — pinning `dot` by writing out `a.x*b.x + a.y*b.y + a.z*b.z` in
`es_intersect` — would change the arithmetic of the *default* path's kernel, which this packet
forbids. It is worth doing on its own, with the goldens re-verified afterwards.

### 9.4 The parameter layout

The shading block was appended **after** M4's globals, so no existing slot moved:

| slot | meaning |
|---|---|
| 0..20 | M4's globals: triangle count, atlas shape, light, ambient, sky, seed, spp, bounces, light count, BVH base/nodes/prim-base |
| 20 | shading: `0` = `Lambert`, `1` = `Full` |
| 21 | shadows, `0`/`1` |
| 22, 23 | specular, shininess |
| 24..27, 27..30 | `sky_rgb`, `ground_rgb` |
| 30 | `ssaa` |
| 31.. | the per-view records, 16 floats each (`ES_PARAM_VIEW_BASE`) |

`PARAM_VIEW_BASE` moved from 20 to 31 in `renderer.rs` and `common.slang` together; it is one
constant on each side and every kernel reads views through `es_view(v)`. `Lambert` writes the
flag and leaves 21..31 zero. `u32` values ride in the buffer as `f32::from_bits`, the same way
the triangle count always has.

### 9.5 Why the observation path did not get the knob

`EnvRendererCfg` has no `shading` field and `es_env::render::config` does not set one: an
observation renders `Lambert` and only `Lambert`. §28.10 rule 1 is why — the pixels a
committed document describes belong to that document, and a flag that lets a caller re-render
them differently is a flag that lets a caller silently invalidate a checkpoint. Offering the
look on the observation path means moving `observation_hash`, re-collecting the dataset and
retraining; that is a hash-aware packet with the owner's decision behind it, not a field.

`es video showcase --look full` is exactly the case where none of that applies: the showcase
camera is in no scene and in no IR, and its frames are watched by people, not by policies.

### 9.6 Cost

`Shading::Full` at `ssaa: 2` is 4 primary rays and up to 4 shadow rays per pixel where
`Lambert` casts one primary ray — 8× the traversal work per shaded pixel, before the extra
`exp`/`ln` of the highlight. Measured on the SO-101 demo cell (V19b `nominal-00`, 224 ticks,
1280×720, RTX 4090, `es video showcase`), with the readback of §8.4 still in every frame:

| look | ms/frame, interleaved runs of 224 frames |
|---|---|
| `lambert` | 6.7, 6.6 |
| `full` (`ssaa: 2`, shadows on) | 10.1, 9.9, 9.9 |

**+3.1 ms/frame, about +46%**, for 8× the rays — because the frame is not ray-bound. This is
the whole-command wall clock divided by the frames (tessellate, upload, dispatch, readback,
write to disk), on a card that had no other compute process on it (`nvidia-smi`: no compute
apps, P8, 0%). Section 8.4 is why the multiplier is so much smaller than the ray count: most
of a 1280×720 frame is the readback and the file write, and R1b's host-cached staging buffer
is what brought this camera from §8.3's 118.6 ms to the 6–7 ms the `lambert` row measures
(§8.4's R1b run recorded 6.2 on the same cell). The `Rs` scan itself was about 1 ms
of it, and `Full` at `ssaa: 2` makes that about 4.

At 96×96 — the observation size — none of this applies: the observation path renders
`Lambert` and only `Lambert` (§9.5), so its cost did not move by a nanosecond.

These are the amended mix's numbers; it costs nothing to conserve energy (one subtract and
one multiply replace one add), and the frame `show-full/000120.bin` has **no** saturated
pixel where the additive form left the whole table at 255.

## 10. The path tracer grows up (M7/R3)

`docs/packets/M7/R3-pt-quality.md`. Before this packet the `Pt` path could not produce a
picture: no next-event estimation (a bounce had to *land* on a light for anything to be lit),
`ReSTIR` combined reservoirs with the biased `1/M`, and the output was linear radiance with
no tone map, so `Channel::Rgb8` was not in `PT_CHANNELS` and `es video showcase` had no way
to use it (`visible-learning.md` 7.17 item 5). Four additions, each behind a `RenderConfig`
field whose default reproduces today's bytes (§28.10 rule 1) — `cargo xtask verify-goldens`
reports **0 changed**, and the `tests/fixtures/visible-learning/frames` fixture is untouched
because the observation path never sees any of these fields ([§9.5](#95-why-the-observation-
path-did-not-get-the-knob) applies verbatim: `EnvRendererCfg` has no `nee`, no `exposure`,
no `tonemap`).

### 10.1 Next-event estimation and MIS

`Pt { nee }`, default `false`. At every diffuse hit the estimator samples the lights directly
and MIS-weights the result against the BSDF strategy with the **power heuristic**, β = 2 (PBR
4e 13.10). Three light kinds, all through one code path on each side:

| light | sampled as | MIS | shadow ray |
|---|---|---|---|
| emissive triangles (a geom named `*_light`) | one pick uniform over the light list, then a uniform point on that triangle; the solid-angle pdf is `d² / (\|cos_l\| · A · n_lights)` | yes, against `cos_s/π` | one, to `dist · (1 − 1e-3)` |
| the directional light `light_dir`, radiance `light_rgb` (new, default `[0,0,0]`) | a delta: the direction *is* `light_dir` | no — no pdf to weight against | one, to `SHADOW_FAR` |
| the sky `sky` (a miss still returns it) | a cosine-weighted hemisphere direction | yes, and the pdf is the BSDF's, so the heuristic gives each exactly 1/2 | one, and it has to **escape** |

`light_rgb` defaults to zero so a `Pt` render that does not ask for a directional light gets
one that contributes nothing; `sky` already defaulted to zero. Each of the three is an `if`
on *config*, uniform across the dispatch, so the work per pixel is still fixed (§3.4 forbids
a data-dependent loop bound, not a compile-time-uniform branch).

**The last vertex drops the MIS weights, and this is the one non-obvious thing here.** At
`bounce == bounces - 1` the continuation ray is sampled and never traced, so the
complementary `w_bsdf` share of that vertex's direct lighting is not estimated anywhere. With
the weights left in, NEE at `B` bounces is *darker* than the BSDF-only tracer at `B` by a
measurable amount; with them dropped, NEE at `B` bounces is an estimator of exactly what the
BSDF-only tracer estimates at `B + 1` — the same set of path lengths. Measured on Cornell,
16×16, 4,096 spp:

| comparison | difference of image means |
|---|---|
| NEE @ 3 bounces vs no-NEE @ 3 bounces, MIS at the last vertex | **+3.72 %** (a truncation difference, not a bias) |
| NEE @ 3 bounces vs no-NEE @ 4 bounces, MIS dropped at the last vertex | **−0.45 %** |

`nee_converges_to_the_same_image` asserts the second row under 1 %. It deviates from the
packet twice, and both deviations are the same finding: it compares `B` against `B + 1`
rather than `B` against `B`, and it asserts on the **image means** rather than the per-pixel
mean absolute difference, because at 4,096 spp each estimator still carries ~0.0015 of its
own noise and the per-pixel figure is 3.0 % of the mean no matter how right the estimators
are. The per-pixel number is printed beside the assertion with a 10 % backstop, so a genuine
mismatch — a missing weight, a wrong pdf — still fails loudly.

**What NEE buys**, `nee_at_low_spp_has_lower_variance`, Cornell 16×16, 16 spp, 3 bounces,
RMSE against the 4,096 spp NEE reference:

| estimator | RMSE at 16 spp |
|---|---|
| NEE off | 0.024110 |
| NEE on | **0.016565** (1.46× lower) |

1.46× the RMSE is ~2.1× the samples for the same noise, on a scene whose single light is a
large ceiling panel that a cosine bounce finds fairly often. A small light, which is where
NEE usually wins by an order of magnitude, is not in any committed scene; that number is
`Target / Status: unverified`.

### 10.2 `ReSTIR` is unbiased now, and here is the number

Two changes, both in the spatial pass:

1. **Pairwise MIS** (Wyman et al. 2023 §5) replaces the biased `1/M`. For the canonical
   (centre) technique `c` and `N` neighbours,
   `m_i(X) = (1/N)·(M_i p̂_i(X)) / (M_i p̂_i(X) + M_c p̂_c(X))` and
   `m_c(X) = (1/N)·Σ_i (M_c p̂_c(X)) / (M_i p̂_i(X) + M_c p̂_c(X))`, which sum to one term by
   term for *any* `X`. The resampling weight of a candidate becomes `m · p̂_destination(X) · W`
   and the combined contribution weight is `w_sum / p̂_destination(Y)` — no `1/M` and no `1/Z`
   left to divide by, because the `m`s already carry the normalization. This is the packet's
   `W = (1/p̂_y)·(w_sum/Z)` with `Z` folded into the per-candidate weights, which is the form
   that also gives a neighbour who *could* have generated the sample more weight than one who
   barely could, rather than only excluding those who could not. It costs three extra target
   evaluations per neighbour (its own target at its own sample, and at the canonical sample)
   and no extra rays; the neighbour's shading point comes from the g-buffer the same way the
   centre's does.
2. **The visibility test moved** from the survivor of the initial pass to the finally
   selected sample at the destination. Same one shadow ray per pixel. It matters because
   zeroing `W` at the *source* makes a neighbour's reservoir carry a visibility that was
   decided at a different shading point; testing it at the destination makes the estimator
   `f_shadowed(y) · W(y)` with `W` built from the unshadowed target, which is unbiased. The
   reservoir stores the **unshadowed** `W`, so temporal reuse next frame still reuses the
   quantity the target function is defined over.

`restir_is_unbiased_within_tolerance`: 256 independent 1 spp `ReSTIR` frames (temporal off,
spatial on), averaged, against a 4,096 spp path trace of the *same* integral — direct
lighting only, which is two bounces without NEE, because that is what `ReSTIR` DI estimates.
Cornell, 16×16, reference mean radiance 0.037287:

| weighting | mean | bias | oracle |
|---|---|---|---|
| `1/M` + source-side visibility (**before M7/R3**) | 0.036631 | **−1.759 %** | fails |
| `1/M` + destination-side visibility | 0.036616 | −1.798 % | fails |
| pairwise MIS + destination-side visibility (**now**) | 0.037324 | **+0.099 %** | passes |

The middle row is the attribution: essentially all of the bias was the `1/M` weight, and
moving the visibility test contributes ~0.04 % — inside the noise of a 256-seed average. The
packet's own `unverified` TODO now has a number on both sides of it.

`gpu_restir_and_svgf_match_the_cpu_within_tolerance` keeps its 1e-5 normalized tolerance and
still passes on both cards; it pins the GPU against the CPU, not against a golden, which is
what made replacing the weighting legal at all.

### 10.3 The tone map

`RenderConfig::{exposure, tonemap}`, defaults `1.0` and `Tonemap::Reinhard`:

```
Reinhard(c) = (c·e) / (1 + c·e)                                    // Reinhard et al. 2002
Aces(c)     = clamp( x(2.51x + 0.03) / (x(2.43x + 0.59) + 0.14) )  // Narkowicz 2015, x = c·e
```

then the existing exact piecewise sRGB transfer and the existing rounding. **Neither operator
contains a transcendental** — additions, multiplies and one division per channel — so the
oracle is not a ULP budget but bit equality: `tonemap_is_bitwise_on_both_sides` renders a
64×64 Cornell `PtRadiance` tile at 1 spp (the configuration
`gpu_path_tracer_matches_the_cpu_reference_at_1spp` already pins bit-equal, so any difference
is the tone map's) through both operators at exposure 1 and 8, and compares the `Rgb8`:
**0 of 12,288 bytes differ** in all four cases, on the RTX 3060 and on the RTX 4090. The
same test asserts monotonicity on the CPU over `[0, 64)` at three exposures, which is the
property that makes `Rgb8` a usable image rather than a lookup table.

`Channel::Rgb8` therefore joins `PT_CHANNELS`. On the GPU it is a **separate dispatch**,
`pt.slang`'s `tonemap` entry point, recorded last so it reads whatever `ReSTIR` and `SVGF`
left in the radiance buffer rather than what the tracer wrote. On the CPU it is the last
block of `cpu::path_trace`. One new golden, `cornell_pt_nee_rgb8` (4 spp, 3 bounces, NEE,
Reinhard, exposure 1, generated by the CPU reference): the GPU reproduces it **bit for bit**
on both cards, and the underlying `PtRadiance` agrees to 9.1e-7 normalized / 216 ULP — the
ULP number is large and the normalized one is tiny for the reason §5.1 already gives, that
most of the image is near zero.

`Rs` `Full` is **not** given the tone map. It clamps, as it did in M7/R2, and its golden is
untouched; the two operators are `pub` in `es_render::cpu` and a later packet can hand them
to the rasterizer with its own golden.

### 10.4 The §15.3 SSIM number finally exists, and it is low

`es_render::ssim(a, b, w, h) -> f64` — the simple windowed form of Wang et al. 2004: an 8×8
window slid one pixel at a time over the Rec.709 luma plane, uniform weights (not the 11×11
Gaussian), `K1 = 0.01`, `K2 = 0.03`, `L = 255`, the mean of the per-window score, all in
`f64`. `ssim(a, a)` is **exactly** `1.0`, not `1.0 ± 1e-12`: the variance, the covariance and
the two means are written so that identical input makes the numerator and the denominator the
same bits.

`rs_pt_ssim` (an `--ignored` measurement, not an assertion) compares `Rs` `Full` at the R2
preset (`ssaa 2`) against a converged `Pt` NEE render, sweeping the exposure:

| exposure | Cornell 64×64 SSIM | Cornell PT mean byte | SO-101 320×180 SSIM | SO-101 PT mean byte |
|---|---|---|---|---|
| 0.5 | 0.2282 | 23 | 0.2207 | 14 |
| **1** (default) | **0.3247** | 33 | **0.2910** | 23 |
| 2 | 0.4246 | 47 | 0.3775 | 35 |
| 4 | 0.5099 | 65 | 0.4742 | 50 |
| 8 | 0.5646 | 87 | 0.5685 | 69 |
| 16 | 0.5861 | 112 | 0.6470 | 92 |
| 32 | 0.5865 | 140 | 0.7069 | 118 |
| 64 | 0.5861 | 168 | 0.7562 | 145 |
| 128 | 0.5912 | 193 | 0.8016 | 168 |

(`Rs` `Full`'s own mean byte is **130** on Cornell and **173** on SO-101. Cornell is the CPU
reference at 1,024 spp; SO-101 is the GPU in 16 chunks of 64 spp averaged on the host, so no
single dispatch can trip a desktop driver watchdog — a different estimator from one 1,024 spp
dispatch, equally unbiased, and the only one that runs on Windows.)

**The threshold is not set here, and the table is why.** At the default exposure the two
images are 0.29–0.32 similar, which looks like a failure and is not: `Rs` `Full` is a
hemisphere ambient plus a directional light with no interreflection, and `Pt` is one 0.24 m
emissive panel at ~1 W/sr/m² with global illumination and no directional light at all. They
are pictures of *different lighting*, and most of the SSIM deficit is the brightness offset —
the score climbs monotonically with exposure right up to the point where Reinhard has
flattened everything. Three things follow, and they are decisions for the M7 review rather
than for this packet:

1. a §15.3 threshold set from these numbers would be a threshold on the scene's lighting
   setup, not on the renderers;
2. the honest comparison gives the `Pt` side the `Rs` side's lights — `light_rgb` set to the
   directional light's radiance and `sky` to the hemisphere — which R3 made *possible*
   (§10.1's light table) but did not tune;
3. or §15.3's contract is restated as "the geometry channels are bit-identical and RGB is
   whatever the lighting model says", which is what the code actually guarantees today.

`Target / Status: unverified` for any §15.3 SSIM threshold until one of the three is chosen.

### 10.5 `es video showcase --path pt`

```
es video showcase ... --path rs|pt [--spp 64] [--bounces 3] [--exposure 1.0]
                      [--tonemap reinhard|aces]
```

`rs` stays the default, so V9's bit-identity oracle keeps meaning: a re-render of a committed
run reproduces its recorded frames. `pt` runs with **NEE on, `ReSTIR` and `SVGF` off** —
`ReSTIR`'s temporal reuse assumes a camera that does not move, and the showcase re-renders a
whole trajectory. `es_env::render::config` is still the one place a render path becomes a
`RenderConfig`; `--exposure` and `--tonemap` are set on the returned config beside `--look`,
the same way M7/R2 set `shading`. There is no `--svgf` flag: nobody asked for one and the
field is reachable from Rust.

### 10.6 Cost, RTX 4090, V19b `nominal-00` (224 ticks, 1280×720)

The same cell, the same camera (`--eye 0.66,-0.46,0.52 --look-at 0.14,-0.04,0.04 --fov 36`)
and the same interleaved-runs method as [§9.6](#96-cost) — whole-command wall clock divided
by the 224 frames, so tessellate, upload, dispatch, readback and the write to disk are all in
it.

**The card was not idle.** `nvidia-smi` reported 6–7 other compute processes throughout
(another agent training); §9.6's numbers were taken on a card with none. The `lambert` row
below is 3.0× §9.6's, which is the size of the contention, and every row carries it equally
because the runs are interleaved. The absolute figures are therefore a **floor**, not the
card's capability:

| path | ms/frame, this run (loaded card) | §9.6, idle card |
|---|---|---|
| `Rs` `lambert` | 20.8, 20.0 | 6.7, 6.6 |
| `Rs` `full` (`ssaa 2`) | 25.0, 26.3 | 10.1, 9.9, 9.9 |
| `Pt` NEE, 64 spp, 3 bounces, Reinhard e=1 | **562.3, 561.2** | — |
| `Pt` NEE, 64 spp, 3 bounces, Reinhard e=32 | 560.8 | — |
| `Pt` NEE, 256 spp, 3 bounces, Reinhard e=32 | **2264.3** | — |

Three things the table says. **The tone map is free**: 562.3 vs 560.8 ms/frame is the same
number with a different `exposure`, which it should be — one dispatch of one multiply, one
add and one divide per pixel over a frame that spends half a second in the tracer.
**`spp` is linear**: 4× the samples is 4.03× the time (2264.3 / 561.2), so there is no
per-frame overhead worth naming and a caller can read "how long do I have?" straight off.
And **`Pt` is ~27.5× `Rs` `lambert`** at 64 spp on the same loaded card; on an idle one the
ratio will be *larger*, because `lambert` at 6.7 ms is mostly readback and file write while
`Pt` at 560 ms is almost entirely the tracer. Any idle-card `Pt` figure is
`Target / Status: unverified` — the honest reading of this run is "the showcase's PT path is
half a second a frame at 64 spp on a 4090, and 64 spp is visibly noisy; 256 spp at 2.3 s a
frame is what the pictures under `~/artifacts/plan-v/m7-r3/show-pt-256/` look like."

### 10.7 What R3 still skips

The way [§4.2](#42-restir-di) lists its skips, and for the same reason — §1.9 item 2 makes
the path tracer the second thing cut, so every step of it says what it is not:

- **`ReSTIR` GI.** Still direct lighting only. Indirect bounces go through the path-traced
  estimator, which is why `restir: true` and a high `spp` are two different tools rather than
  one.
- **`ReSTIR` over the other two light kinds.** The reservoirs hold emissive triangles only.
  The directional light and the sky are NEE-only; a reservoir over a mixed light list needs
  a light *type* in the reservoir stride, which is a buffer-layout change.
- **Environment maps.** `sky` is one constant colour sampled with a cosine pdf. A real
  environment map needs an image, an importance-sampling distribution built over it, and a
  pdf that is not the BSDF's — at which point the MIS here starts to earn its keep.
- **Glossy and specular BSDFs.** Everything is still Lambertian. `Shading::Full`'s
  Blinn-Phong highlight has no `Pt` counterpart, there is no microfacet model, no Fresnel, no
  refraction, no perfectly specular path. The MIS machinery is exactly the machinery a
  microfacet BSDF would need, which is the point of building it now.
- **Spectral anything.** Three channels, RGB, no wavelength, no dispersion, no colour
  management beyond §3.1's sRGB transfer.
- **Textures.** Still one flat albedo per geom (§0).
- **Adaptive sampling, firefly clamping, Russian roulette.** All three make the work or the
  estimator data-dependent, which §3.4 forbids on a path that has to reproduce bit for bit.
  A converged image is `spp`, and `spp` is a fixed number.
- **A tone map on the `Rs` path.** §10.3.
- **The temporal `ReSTIR` pass is still unbiased-by-accident**: with no motion-vector
  reprojection its two candidates share one shading point, so `1/M` *is* the correct MIS
  weight there. Give it reprojection and it needs the same pairwise treatment the spatial
  pass got.
