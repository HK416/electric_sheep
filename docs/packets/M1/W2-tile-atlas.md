# W2-tile-atlas — `es-render`: tile atlas, channel contract, compute rasterizer

Spec: §15.1 (render → observation path, channel contract), §15.2 (tile atlas for mass-parallel
cameras), §15.3 (render paths; `RS` is the vision-learning default), §15.4 (acceleration
structures — *not* implemented, see below), §3.1 (OpenCV camera frame, top-left image origin,
sRGB default), §3.3 (rendering is FP32, camera-relative), §3.4 (deterministic execution),
§7.2 (`ImageSpec`), §20.2 (the `render_tile_atlas` budget item this mirrors), §28.3 W2,
§1.4 (golden images are the oracle). Design: `docs/design/renderer.md`.

Pairs with `docs/packets/M4/W8-renderer.md`, which owns the path tracer in the same crate.

## context

```
crates/es-render/**                  (src, slang/raster.slang, slang/common.slang, slang/rng.slang, tests)
tests/golden/render/cornell_rs_*     (new goldens, generated once by the CPU reference)
docs/design/renderer.md
docs/packets/M1/W2-tile-atlas.md
```

## spec

- `TileAtlasCfg { tile_w, tile_h, tiles_per_row, n_tiles }` →
  `AtlasLayout { rows, width, height }` with
  `tile_origin(i) = ((i % tiles_per_row) * tile_w, (i / tiles_per_row) * tile_h)` — a closed
  form, so a consumer reconstructs per-env tensors by arithmetic and never by a host transfer
  (§15.2). An atlas wider or taller than `maxImageDimension2D` (16384) is refused.
- `AtlasLayout::atlas_bytes(channel)` mirrors the `render_tile_atlas` item of
  `crates/es-compile/src/budget.rs` **term for term**, double-buffer factor included:
  `rows * tiles_per_row * tile_w * tile_h * components * dtype_bytes * 2`. `es-compile` is
  layer 7 and unreachable from layer 5, so the duplication is pinned by a test that recomputes
  the §15.2 reference figure by hand. `device_bytes()` is the separate, smaller number the
  renderer actually allocates: single-buffered, one 32-bit word per component, `Rgb8` packed
  as one `RGBA8` word.
- Channel contract (§15.1): the `Rs` path writes `Rgb8`, `Depth32 { unit_m: 1.0 }`, `Normal`
  (camera space), `SegmentationId` (1-based geom id, `0` = background). `Flow` and
  `RgbF32Linear` are `RenderError::UnsupportedChannel`, not silently empty buffers.
- `Renderer::new(gpu, RenderConfig)` compiles `raster.slang` through
  `es_gpu::SlangCompiler` with `Gpu::deterministic_execution_modes()` (§3.4 step 3).
  `upload_scene(&SceneDesc)` tessellates box / plane / sphere / capsule / cylinder /
  ellipsoid into world-space triangles with fixed, constant subdivision; `Shape::Mesh` and
  `Shape::HeightField` are `RenderError::UnsupportedShape`.
  `render(&[CameraView]) -> Atlas`, `Atlas::read_tile(cam, channel) -> Tile`.
- `es_render::ImageSpec` is the layer-5 **subset** of §7.2: width, height, pinhole
  intrinsics, near, far. `camera_model` is `Pinhole`, `distortion` `None`, `shutter` `Global`,
  colour space sRGB for `Rgb8` — fixed, not fields, because §18.3 sensor realism is a later
  pass over the atlas. A view whose resolution differs from the tile is refused rather than
  resampled (resampling without transforming intrinsics is §7.2 `OBS-034` / `INV-14`).
- One thread per atlas pixel. The rasterizer scans triangles in ascending index order and
  keeps the nearest hit, ties to the lower index; coverage and depth come from the same
  Möller–Trumbore intersection the path tracer uses, which is what makes §15.3's
  "depth/seg/normal are bit-identical between the paths" hold by construction. No binning,
  no acceleration structure: §15.4's TLAS needs a Vulkan extension `es-gpu` does not expose.
- Determinism: no atomics, no shared memory, no subgroup ops, no dependence on workgroup
  count; `es_math::approx` / `approx.slang` for every transcendental (§3.2 `DET-010`),
  including the sRGB transfer's `c^(1/2.4)`; `BTreeMap` only.

## oracle

```
cargo fmt -p es-render --check
cargo clippy -p es-render --all-targets -- -D warnings
cargo test -p es-render -- --nocapture
cargo xtask layering && cargo xtask verify-goldens && cargo xtask context-budget && cargo xtask check-spec-refs
```

Goldens are regenerated only by `cargo test -p es-render -- --ignored generate_goldens`, which
runs the **CPU** reference. Never the GPU: a golden produced on one driver would bake that
driver's arithmetic into the repository. Every GPU test prints `SKIP <test>: <reason>` and
returns when there is no Vulkan device or no `slangc`, so the suite passes on a CI box without
a GPU and says so.

## acceptance

Measured on an NVIDIA RTX 4060 Laptop GPU (driver 592.82, Slang 2026.8), 64×64 tiles,
96-triangle Cornell box:

- `cpu_reference_reproduces_the_goldens_bit_for_bit` — all four goldens.
- `gpu_rasterizer_matches_the_cpu_goldens` — `Rgb8` 0 of 12288 bytes differ,
  `SegmentationId` bit-equal, `Depth32` **max ULP 0**, `Normal` **max ULP 0**.
- `gpu_renders_are_bit_identical_across_runs` — all four channels, two runs.
- `gpu_atlas_packs_several_cameras` — 3 cameras + 1 padding tile in a 2×2 atlas; every tile
  bit-equal to rendering that camera alone; `read_tile(3, ..)` is an error.
- Unit tests: tile origins and padding, the budget-formula mirror, the
  `maxImageDimension2D` refusal, tessellation winding and dense 1-based segmentation ids,
  nested body pose composition, RNG stream separation.

## forbidden

Any file outside `context` — in particular `crates/es-usd`, `es-script`, `es-data`, `es-eval`,
`es-py`, `es-compile`, `crates/es`, and the root `Cargo.toml`. The path tracer, ReSTIR and
SVGF (that is `docs/packets/M4/W8-renderer.md`, same crate, separate acceptance). Editing a
golden to make a test pass (§1.4 — goldens are CI read-only; report the ULP instead). New
extension-point traits (`INV-17`). `HashMap`/`HashSet` (§3.4). Graphics pipelines, ray-tracing
extensions or acceleration structures — `es-gpu` exposes none and adding them is its packet,
not this one. Splat rendering (§16.2; the hook is a `RenderPath` variant). Sensor realism
(§18.3). Pushing resize/crop into the renderer (§7.2 — Observation IR owns preprocessing).
