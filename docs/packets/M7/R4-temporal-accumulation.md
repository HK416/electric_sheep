# M7 R4 — temporal accumulation for a still camera, and the V in SVGF

Spec: §15.3 (PT for photorealistic datasets and the showcase), §28.6 (SVGF named), §3.4 (fixed
accumulation order, no data-dependent loop bounds, counter-based RNG), §28.10 rule 1 and (R4).
Design note to extend: `docs/design/renderer.md` (+ `.ko.md`) — section 4.3 rewritten to say what
is now true, a new section 11. Depends on **R3** (NEE, the tone map → `Rgb8`, `ssim`, the PT
showcase) and **R1** (persistent buffers). Read first: Schied et al. 2017 "Spatiotemporal
Variance-Guided Filtering" sections 4.1–4.3 (temporal accumulation, moments, the luminance weight,
the 7×7 variance prefilter for short histories), and `docs/design/renderer.md` 4.3's list of what
was skipped — that list is this packet's checklist.

## the question

The showcase camera does not move; the arm does. Every frame today starts from zero samples.
**Can PT keep, per camera and per pixel, a history that replaces spp on the pixels the scene did
not change, drop it exactly where it did, and let the per-pixel variance drive the à-trous
filter — with `N` frames of 1 spp accumulating to the bytes `1` frame of `N` spp produces?**

## spec

* **`RenderConfig.temporal: Option<Temporal { max_history: u32 }>`**, default `None` (today's
  bytes; every golden and the `frames` fixture unmoved). Read by the `Pt` path only. Per camera
  slot the renderer keeps: the radiance sum, the first and second luminance moments, the history
  length `n ≤ max_history`, and the previous frame's depth / normal / primitive id.
* **Still camera, per-pixel validity.** No reprojection: the history for a pixel is valid when the
  camera slot's `CameraView` is bitwise the previous one **and** the pixel's depth, normal and
  primitive id are bitwise the previous frame's; otherwise `n := 0` for that pixel. A changed
  `CameraView` resets the whole slot. This is the disocclusion rule for a scene that moves under a
  camera that does not; a moving camera is *out of scope* and the note says so.
* **Sample indexing makes accumulation exact.** Frame `f` of a slot draws its `spp` samples with
  `sample = n · spp + s` in `rng::key` — so `N` frames of 1 spp draw the same samples, in the same
  order, as one frame of `N` spp. The sum is kept as a running **sum** (not a running mean) in the
  same left-to-right order the in-kernel loop uses, divided once at output — which is what makes
  oracle 2 a *bitwise* statement on the CPU and not a tolerance.
* **The variance.** From the moments: `var = max(0, E[l²] − E[l]²)`; for `n < 4` the spatial 7×7
  bilateral estimate of the paper (depth/normal-weighted) stands in. The à-trous pass gains
  `w_l = es_exp(−|l_p − l_q| / (σ_l · sqrt(var_p) + ε))`, `σ_l = 4`, multiplied into the existing
  `w_depth · w_normal`; the variance is filtered alongside the colour with the squared weights as
  the paper does. `sqrt` is IEEE-exact; `es_exp` is `approx`'s. History-length-driven kernel widening
  stays skipped (listed).
* **The showcase.** `es video showcase --path pt --accumulate [--max-history N]` (R3's flags plus
  this); the persistent history lives in the one `Renderer` the showcase already keeps across ticks.
* **Observability.** `Channel::History` (`u32` per pixel, the `n` after this frame) joins the PT
  channels — data, not schema — so a test and a person can see where the history was dropped.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
crates/es-render/src/view.rs
crates/es-render/src/cpu.rs
crates/es-render/src/renderer.rs
crates/es-render/src/atlas.rs
crates/es-render/src/lib.rs
crates/es-render/slang/common.slang
crates/es-render/slang/pt.slang
crates/es-render/slang/svgf.slang
crates/es-render/slang/accumulate.slang
crates/es-render/tests/render.rs
tests/golden/render/cornell_pt_accum8_rgb8.*
crates/es/src/cmd/showcase.rs
crates/es/tests/video.rs
crates/es-sensor/src/channel.rs
crates/es-env/src/render.rs
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M7/R4-temporal-accumulation.md
docs/packets/M7/R4-temporal-accumulation.ko.md
```

`view.rs` (`Temporal`, `Channel::History`), `cpu.rs` (the reference: accumulation, moments,
variance, the luminance weight), `renderer.rs` (per-slot history buffers, the validity pass, the
dispatch order), `atlas.rs` **only** if the `u32` channel needs a slot, `slang/{pt,svgf,accumulate}.slang`
(`accumulate.slang` new: validity + sum + moments), `tests/render.rs`, the new golden (8 frames × 1 spp,
NEE on, Reinhard — the CPU generator's bytes), `showcase.rs`/`video.rs` (the flags), the design
note, this packet.

**Two files the packet did not foresee, added at implementation time.** `Channel` is
`es-sensor`'s (layer 3), not `es-render`'s — `es-render` re-exports it — so `Channel::History`
is four match arms in `crates/es-sensor/src/channel.rs`, and one exhaustive `match` over
`Channel` in `crates/es-env/src/render.rs` (`channel_format`) needs one arm to keep compiling.
Neither is a PT knob in `EnvRendererCfg` and neither is reachable from the observation path;
the `forbidden` list stands.

## oracle

1. Every existing golden and GPU test unchanged; `cargo xtask verify-goldens` 0 changed.
2. `cargo test -p es-render accumulation_of_n_frames_is_n_spp` — Cornell, CPU: 8 frames × 1 spp
   accumulated equals 1 frame × 8 spp **bitwise** in `PtRadiance`; and 8 × 4 spp equals 1 × 32 spp.
3. `cargo test -p es-render history_drops_where_the_scene_moved` — Cornell with the tall block
   translated at frame 5: `Channel::History` is 5 on every pixel whose depth/normal/id did not change
   and 1 on every pixel where it did, with at least one pixel in each set; a changed `CameraView`
   gives 1 everywhere.
4. `cargo test -p es-render variance_falls_with_history` — the mean per-pixel `var` at `n = 1, 4,
   16` is strictly decreasing (printed).
5. `cargo test -p es-render gpu_accumulation_matches_the_cpu` — the new golden reproduced bitwise
   by the CPU; the GPU within the existing PT tolerance, printed; the GPU `History` channel bitwise.
6. `cargo test -p es-render luminance_weight_narrows_the_filter_where_variance_is_low` — on a
   half-converged / half-noisy synthetic tile, the filtered output moves less on the converged half
   than without the luminance term (printed RMSE per half).
7. `cargo test -p es --test video showcase_accumulate_flags_are_parsed`; `cargo xtask ci`;
   `cargo xtask check-scope docs/packets/M7/R4-temporal-accumulation.md`.

## acceptance

Oracles 1–7 (GPU on the RTX 3060 and the server). On the server: the PT showcase of V19b's
`nominal-00` with `--spp 4 --accumulate --max-history 32` beside R3's `--spp 64` (ms/frame both;
SSIM of each against R3's 1,024 spp reference for the *static* pixels and for the *arm* pixels
separately — the History channel is the mask), under `~/artifacts/plan-v/m7-r4/`; PNGs of frame
0, 8, 32 in the worktree's `target/plan-u/r4/`. Section 11 records the sample-indexing rule, the
validity rule, the numbers and the remaining skips (reprojection, kernel widening, ReSTIR-GI).

## forbidden

Moving any golden or R3's default; a running mean or any accumulation order other than the
kernel's; a moving-camera reprojection; a transcendental outside `approx`; data-dependent loop
bounds; atomics; `crates/es-gpu/**`; PT knobs in `EnvRendererCfg`; `docs/ARCHITECTURE*.md`.
INV-17: no new trait.
