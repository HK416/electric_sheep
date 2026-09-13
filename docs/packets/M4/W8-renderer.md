# W8-renderer — `es-render`: compute path tracer, ReSTIR DI, a-trous denoise

Spec: §15.3 (`PT` render path; the output contract the paths share), §15.4 (acceleration
structures — *not* implemented), §28.6 (path tracer + ReSTIR + SVGF), §1.9 item 2 (the path
tracer is the second thing cut under scope pressure — build the minimal honest version),
§1.4 (golden images are the oracle, PT/RS channel agreement), §3.1, §3.2 `DET-010`, §3.3,
§3.4 (no global RNG, deterministic execution), §16.2 (the splat path is a third render path
in this crate — hook only). Design: `docs/design/renderer.md`.

Builds on `docs/packets/M1/W2-tile-atlas.md`, which owns the atlas, the channel contract and
the rasterizer in the same crate.

## context

```
crates/es-render/src/{cpu,rng,view,renderer}.rs
crates/es-render/slang/{pt,restir,svgf,rng,common}.slang
crates/es-render/tests/render.rs
tests/golden/render/cornell_pt1spp.{bin,json}
docs/design/renderer.md
docs/packets/M4/W8-renderer.md
```

## spec

- `RenderPath::Pt { spp, bounces, restir, svgf }`. Diffuse only, cosine-weighted hemisphere
  sampling (the weighting cancels the `cos` and the `1/pi`, so throughput is a plain multiply
  by albedo — no pdf division, no 0/0), fixed bounce count, **no Russian roulette** (it makes
  the work per pixel data-dependent for nothing at these bounce counts). Samples accumulate
  sequentially in ascending index: the order is fixed by the loop, so the cheap sum is also
  the reproducible one (§18.4 is for reductions whose order is *not* fixed).
- Depth, segmentation and normal come from the primary hit of sample 0, before any RNG draw,
  through the same `nearest_hit` the rasterizer uses. §15.3's "depth/seg/normal are
  bit-identical between the paths" therefore holds by construction, and
  `pt_and_rs_agree_on_geometry` is the test that would catch someone breaking it.
- RNG (`rng.rs` + `rng.slang`): counter-based and addressed by
  `(seed, view, pixel, sample, bounce, stream)`, the design of `es_env::rng` (§3.4: no global
  RNG). **32-bit Murmur3 `fmix32`, not splitmix64** — Slang's `uint64_t` needs `shaderInt64`,
  which §3.3 does not guarantee on every target. Statistical quality beyond a fixed-bounce
  diffuse path tracer is `unverified`.
- ReSTIR DI (`restir.slang`, three entry points, three dispatches, one buffer written per
  pass): 8 uniform candidates over the emissive triangles with weighted reservoir sampling on
  the unshadowed target function, one shadow scan on the survivor; temporal reuse against the
  previous `render()` call's reservoirs at the same pixel; spatial reuse over four *fixed*
  neighbour offsets with the standard depth/normal similarity test. Emissive = a geom whose
  name ends in `_light` (`SceneDesc` has no emissive field and `es-assets` is not in scope).
  Output replaces `PtRadiance` with the direct-lighting estimate.
  **Skipped, and named as skipped in the code:** MIS weights (the biased `1/M` combination,
  not GRIS pairwise MIS), the bias-correction visibility re-test on reuse, ReSTIR GI
  entirely, every light type but emissive triangles, motion-vector reprojection (so temporal
  reuse is a no-op on frame 1 and assumes a static camera), reservoir ageing beyond the `M`
  clamp.
- `svgf.slang`: `n` a-trous iterations at stride `1 << i` with the 5-tap B-spline wavelet and
  depth/normal edge stopping; the normal weight is `n^32` by five squarings, so no
  transcendental. **This is not SVGF**: no temporal accumulation, no variance estimate, no
  variance-guided luminance weight, no variance prefilter, no disocclusion handling, no
  history-driven kernel widening. It is named for §28.6's deliverable and the kernel's own
  doc comment says exactly this.
- No acceleration structure (§15.4): `es-gpu` exposes compute pipelines only. Both paths scan
  a flat triangle array in index order — the honest ceiling is a few hundred triangles.
- The golden scene has **no coplanar faces**. Where two triangles share a plane a ray hits
  both at the same `t` and the winner turns on which side of `u + v <= 1` the barycentrics
  land — which the CPU and the GPU can answer differently by one ULP.

## oracle

```
cargo fmt -p es-render --check
cargo clippy -p es-render --all-targets -- -D warnings
cargo test -p es-render -- --nocapture
cargo xtask layering && cargo xtask verify-goldens && cargo xtask context-budget && cargo xtask check-spec-refs
```

`cornell_pt1spp` is generated once, from `es_render::cpu::path_trace` — 1 spp, 2 bounces,
ReSTIR and SVGF off: the smallest configuration that exercises the RNG, the bounce loop and
the emissive hit, and the only one a golden can pin bit-exactly without also pinning the
noise of 64 samples. GPU tests `SKIP` with a printed reason when there is no device or no
`slangc`.

## acceptance

Measured on an NVIDIA RTX 4060 Laptop GPU (driver 592.82, Slang 2026.8), 64×64, 96 triangles:

- `gpu_path_tracer_matches_the_cpu_reference_at_1spp` — **bit-identical, max ULP 0**.
- `gpu_pt_and_rs_agree_on_geometry` — `Depth32`, `SegmentationId`, `Normal` bit-equal between
  the paths (§15.3), on the GPU; the same assertion runs on the CPU in
  `cpu::tests::pt_and_rs_agree_on_geometry` so it is checked with no device too.
- `gpu_restir_and_svgf_match_the_cpu_within_tolerance` — claim ≤ 1e-5 of the image peak,
  **measured 1.3e-7 (max 10 ULP)**. SVGF alone is bit-identical; ReSTIR is the one path that
  is not, because the reuse passes sum `p̂ · W · M` over neighbours and the compiler may
  reassociate within an expression even under `NoContraction`. The tolerance is stated against
  the image peak, not per pixel: most of a rendered image is near zero and a 1e-7 difference
  on a 1e-7 pixel is a 100% relative error for an image nobody could tell apart.
- CPU unit tests: determinism of `path_trace`, finiteness and non-negativity with ReSTIR and
  SVGF on, cosine samples inside the hemisphere and unit length, RNG stream separation.

## forbidden

Any file outside `context` — in particular `crates/es-usd`, `es-script`, `es-data`, `es-eval`,
`es-py`, `es-compile`, `crates/es`, and the root `Cargo.toml`. Editing a golden to make a test
pass (§1.4); a divergence is reported as a ULP count. Claiming ReSTIR or SVGF is the published
algorithm — the skipped list above stays in the code. Adding the missing halves (MIS weights,
variance estimation, ReSTIR GI) under this packet: §1.9 item 2 says this path is cuttable, so
it stays minimal until a §28.6 packet asks for more. New extension-point traits (`INV-17`).
`HashMap`/`HashSet`, FP atomics, `std` transcendentals in a kernel (§3.4, §3.2). Ray-tracing
or graphics Vulkan extensions — that is `es-gpu`'s packet. Splat rendering (§16.2). Sensor
realism (§18.3). An SSIM comparison between `RS` and `PT` RGB: it needs a converged PT render
and an SSIM implementation, both `unverified` here.
