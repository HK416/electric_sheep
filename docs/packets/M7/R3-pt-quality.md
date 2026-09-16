# M7 R3 — the path tracer grows up: next-event estimation, MIS, an unbiased ReSTIR, a tone map

Spec: §15.3 (PT: photorealistic datasets and goldens; RGB between RS and PT is an SSIM
threshold — never measured), §15.1 (same channel contract), §28.6 (ReSTIR/SVGF named), §3.2/§3.4
(`approx` transcendentals, fixed accumulation order, counter-based RNG), §1.9 item 2 (PT is
cuttable — so every step here is honest about what it skips), §28.10 rule 1. Design note to
extend: `docs/design/renderer.md` (+ `.ko.md`) sections 4, 4.2, 6 and a new section 10. Depends
on **R1** (traversal) and **R2** (`Shading::Full` is the RS side of the SSIM comparison; the
tone map is shared). References to read first: Pharr, Jakob, Humphreys, *Physically Based
Rendering* 4th ed. §13.10 (light sampling, MIS with the balance/power heuristic); Bitterli et al.
2020 "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting"
§4.3 (the unbiased `1/Z` combination) and Wyman et al. 2023 "A Gentle Introduction to ReSTIR"
§5 (pairwise MIS weights); Reinhard et al. 2002 (the `x/(1+x)` operator).

## the question

PT today: all-diffuse, emissive triangles the only light, a hit accumulates `Le` only when a
bounce happens to land on a light (no next-event estimation), `ReSTIR` combines reservoirs with
the biased `1/M`, and the output is linear radiance with no tone map, so PT cannot produce
`Rgb8` and the showcase never uses it (design note 7.17 item 5). **Can PT produce an `Rgb8`
frame that a person would call photoreal, converge fast enough to be usable for the showcase,
and let the §15.3 SSIM contract be measured for the first time?**

## spec

* **Lights.** Three kinds, all through one `Light` enumeration on the CPU and a small light
  table in the params/scene buffers on the GPU: emissive triangles (as today), the directional
  light `light_dir` (`RenderConfig` already has it; give it a radiance `light_rgb`, default
  `[0,0,0]` so PT's default output is unchanged), and the sky (`sky` already exists: a miss
  returns it — keep that; in `Full` PT it is also sampled as a light with a cosine-weighted
  hemisphere pdf).
* **NEE + MIS.** At each diffuse hit: sample one light (uniform over the light list for
  emissive triangles, then uniform over the triangle's area; the directional light is a delta —
  no MIS needed, one shadow ray), trace a shadow ray with R1's any-hit, add
  `f * Le * G / p_light * w_light`; continue the BSDF bounce as today and, when it hits an
  emissive triangle, add `Le * w_bsdf` where `w_light`/`w_bsdf` are the **power heuristic**
  (β = 2) over the two strategies' pdfs. Behind `Pt { nee: bool }`, default `false` so
  `cornell_pt1spp` is untouched. Fixed sample order; the RNG streams are
  `key(…, bounce, stream)` with new stream ids for the light pick, the area sample and the
  shadow test — document the stream table.
* **ReSTIR unbiased.** Replace `1/M` with the unbiased contribution weight `W = (1/p̂_y) ·
  (w_sum / Z)` where `Z` counts the reused reservoirs whose target function is non-zero at the
  destination, and use pairwise MIS in the spatial pass (the "Gentle Introduction" form) so a
  neighbour that cannot have generated the sample contributes zero weight. Behind
  `Pt { restir: true }` as before — the current biased path is *replaced*, not kept: it was the
  packet's own `unverified` TODO, and the test that pins ReSTIR (`gpu_restir_and_svgf_match_the
  _cpu_within_tolerance`) has a tolerance, not a golden, so the change is legal. Record the
  before/after mean-radiance bias against a 4,096 spp reference on Cornell (an observation with
  a number; `unverified` → measured).
* **Tone map → `Rgb8`.** `Channel::Rgb8` joins `PT_CHANNELS`. `RenderConfig` gains `exposure:
  f32` (default 1.0) and `tonemap: Tonemap { Reinhard, Aces }` (default `Reinhard`):
  `Reinhard(c) = c·e / (1 + c·e)` per channel; `Aces` is the Narkowicz 2015 rational fit
  `(x(2.51x + 0.03)) / (x(2.43x + 0.59) + 0.14)` clamped — both are additions, multiplies and
  one division per channel, no transcendental, so CPU == GPU **bitwise** is the oracle; then the
  existing exact sRGB transfer and rounding. The same two functions are usable by R2's `Full`
  shading (it clamps today); do not change R2's default.
* **SSIM.** `es_render::ssim(a: &[u8], b: &[u8], w, h) -> f64` — the standard 8×8 windowed SSIM
  on the luma of two `Rgb8` tiles (Wang et al. 2004 constants `K1 = 0.01, K2 = 0.03`), pure
  Rust, f64, deterministic. Measured between `Rs` `Full` (R2 preset, `ssaa 2`) and `Pt` at 1,024
  spp with NEE on Cornell and on the SO-101 scene from V9's showcase camera; the numbers go in
  the design note; §15.3's threshold is **set from them afterwards**, not asserted here (`Target
  / Status: unverified` until then — the packet only makes the number exist).
* `es video showcase --path pt --spp N [--bounces B] [--exposure E] [--tonemap reinhard|aces]`
  writes `Rgb8` frames from PT (NEE on, ReSTIR off by default, SVGF optional).

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

```
crates/es-render/src/view.rs
crates/es-render/src/cpu.rs
crates/es-render/src/renderer.rs
crates/es-render/src/lib.rs
crates/es-render/src/rng.rs
crates/es-render/src/ssim.rs
crates/es-render/slang/common.slang
crates/es-render/slang/pt.slang
crates/es-render/slang/restir.slang
crates/es-render/tests/render.rs
tests/golden/render/cornell_pt_nee_rgb8.*
crates/es/src/cmd/showcase.rs
crates/es/tests/video.rs
crates/es-env/src/render.rs
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M7/R3-pt-quality.md
docs/packets/M7/R3-pt-quality.ko.md
```

`crates/es-render/src/{view.rs,cpu.rs,renderer.rs,lib.rs,rng.rs}`, `crates/es-render/src/ssim.rs`
(new), `crates/es-render/slang/{common.slang,pt.slang,restir.slang}`, `crates/es-render/tests/render.rs`,
`tests/golden/render/cornell_pt_nee_rgb8.*` (new, generated: 4 spp, 3 bounces, NEE, Reinhard,
exposure 1 — pinned bitwise on the CPU; the GPU matches at the stated tolerance), `crates/es/src/cmd/showcase.rs`
(the flags), `crates/es/tests/video.rs`, `crates/es-env/src/render.rs` **only** to pass the new
`RenderConfig` fields through `config()` unchanged (defaults), `docs/design/renderer*.md`,
`docs/packets/M7/R3-pt-quality*.md`.

## oracle

1. Every existing golden and GPU test unchanged; `cargo xtask verify-goldens` 0 changed
   (`cornell_pt1spp` has NEE off, `light_rgb` zero, no tone map in `PtRadiance`).
2. `cargo test -p es-render tonemap_is_bitwise_on_both_sides` (GPU; `SKIP` without a device) —
   a 64×64 `PtRadiance` tile through `Reinhard` and `Aces` on the CPU and on the GPU: `Rgb8`
   bitwise equal, and `RgbF32Linear`-in → `Rgb8`-out is monotone per channel on the CPU.
3. `cargo test -p es-render nee_converges_to_the_same_image` — Cornell, CPU, NEE on vs off at
   4,096 spp (small tile, 16×16): mean absolute difference of linear radiance < 1 % of the mean
   radiance, printed — the estimators agree in expectation.
4. `cargo test -p es-render nee_at_low_spp_has_lower_variance` — 16 spp, NEE on vs off, each
   against the 4,096 spp NEE reference: NEE's RMSE is smaller (printed).
5. `cargo test -p es-render restir_is_unbiased_within_tolerance` — ReSTIR (temporal off, spatial
   on) at 1 spp averaged over 256 independent seeds vs the 4,096 spp reference: |bias| < 1 % of
   mean radiance (printed; the old `1/M` weight fails this — record its number too).
6. `cargo test -p es-render gpu_pt_nee_matches_the_cpu` — the new golden reproduced bitwise by the
   CPU; the GPU within the existing ReSTIR-style tolerance, printed.
7. `cargo test -p es-render ssim_is_one_for_identical_and_falls_with_noise` — `ssim(a, a) == 1.0`,
   `ssim(a, a + noise)` decreases monotonically with noise amplitude.
8. `cargo test -p es --test video showcase_pt_flags_are_parsed`; `cargo xtask ci`;
   `cargo xtask check-scope docs/packets/M7/R3-pt-quality.md`.

## acceptance

Oracles 1–8 (GPU ones on the RTX 3060 and the server). On the server: a PT showcase of V19b's
`nominal-00` (`--spp 64 --bounces 3`, Reinhard, R1's camera) under `~/artifacts/plan-v/m7-r3/`,
its ms/frame beside R1/R2's, and the SSIM table (Cornell, SO-101) in design note section 10.
Every skipped thing (ReSTIR GI, environment maps, glossy BSDFs, spectral anything) listed in the
note the way section 4.2 lists its skips today.

## forbidden

Moving `cornell_pt1spp` or any golden; changing `Rs` `Lambert`; a transcendental outside
`approx`; Russian roulette or any data-dependent loop bound (§3.4: fixed work per pixel);
atomics/shared memory; `crates/es-gpu/**`; giving `EnvRendererCfg` the PT knobs (the observation
path stays where R2 left it); `docs/ARCHITECTURE*.md`. INV-17: no new trait.
