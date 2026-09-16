# M7 R2 — the RS look: shadows, a hemisphere sky, highlights and supersampling, opt-in

Spec: §15.3 (RS is the vision default; RGB between paths is a similarity threshold), §15.1 (the
channel contract is unchanged), §3.2/§3.4 (every transcendental through `es_math::approx`;
fixed accumulation order), §28.10 rule 1 (the default look does not move one byte). Design note
to extend: `docs/design/renderer.md` (+ `.ko.md`) — section 3.1 gains the `Full` shading, new
section 9. Depends on **R1** (the any-hit traversal a shadow ray needs is the BVH's).

## the question

RS shades `albedo * (ambient + max(0, n·L) * (1 - ambient))`: no shadow, no highlight, a constant
ambient, one sample per pixel. It is honest and it looks like 1995. **Can RS have contact
shadows, a sky, highlights and anti-aliasing without changing a byte of what the committed
observation documents render?**

## spec

* `RenderConfig` gains `shading: Shading` with `#[derive(Default)]`:

  ```rust
  pub enum Shading {
      /// Today's flat Lambert + constant ambient. The default; every golden pins it.
      #[default] Lambert,
      Full {
          /// One shadow ray towards `light_dir` per shaded pixel (any-hit, R1's traversal).
          shadows: bool,
          /// Blinn-Phong: specular weight in [0, 1] and shininess exponent; 0.0 disables.
          specular: f32,
          shininess: f32,
          /// Hemisphere ambient: sky colour for n.z = +1, ground colour for n.z = -1, lerped
          /// on (n.z + 1) / 2. Replaces the constant `ambient` when `Full`.
          sky_rgb: [f32; 3],
          ground_rgb: [f32; 3],
          /// Supersampling factor per axis (1 = off). The atlas is rendered at
          /// (w*ssaa, h*ssaa) and box-filtered in fixed row-major order before encoding.
          ssaa: u32,
      },
  }
  ```

  `RenderConfig::rs()` keeps `Shading::Lambert`; `RenderConfig::rs_full(atlas)` is the opinionated
  preset (`shadows: true, specular: 0.25, shininess: 32.0, sky [0.55, 0.65, 0.85], ground
  [0.25, 0.22, 0.20], ssaa: 2`).
* Shading in `Full`, identical on the CPU (`cpu.rs`) and in `common.slang`, in this order and
  with these functions: `n` face-forwarded; `vis = shadows ? (any_hit(p + n*RAY_EPS, L) ? 0 : 1)
  : 1`; `diffuse = max(0, n·L) * vis`; `h = normalize(L - d)`; `spec = specular * pow(max(0, n·h),
  shininess) * vis` with `pow` as `es_exp(es_ln(x) * shininess)` guarded at `x <= 0`;
  `hemi = lerp(ground, sky, (n.z + 1) * 0.5)`; `rgb_lin = albedo * (hemi + diffuse * (1 - hemi))
  + spec + emission` per channel (**amended at review: energy-conserving mix** — the additive
  form exceeded the albedo on every lit surface and rendered the table pure white). Then the existing exact sRGB transfer and `u8` rounding. SSAA: the supersampled
  linear colour is averaged in **row-major order over the `ssaa × ssaa` block** (a fixed
  sequential `+=` then one multiply by `1/(ssaa*ssaa)`), and only then encoded — on both sides.
* The params buffer grows at the **end** (new slots after the existing ones, so `Lambert`'s
  indices do not move); the `Rs` kernel branches on a shading flag read from params — one
  kernel, not two.
* `es video showcase --look lambert|full` (default `lambert`, so V9's bit-identity oracle keeps
  meaning) selects the preset. `EnvRendererCfg` and the observation path are **not** given the
  option in this packet: rule 1 — the look of a committed document is the document's, and
  offering the knob there is a later, hash-aware packet.
* Goldens: `cornell_rs_full_rgb8` (u8, 64×64, the `rs_full` preset with `ssaa: 2`) generated once
  by the CPU generator (`generate_goldens`, extended); depth/seg/normal of `Full` are the same
  buffers as `Lambert` and need no new golden — the test asserts they are bitwise equal to
  `cornell_rs_*`. With `ssaa: 2` the geometry channels are written from the **first** sub-sample
  of each block so the equality above still holds (state this in the note).

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

```
crates/es-render/src/view.rs
crates/es-render/src/cpu.rs
crates/es-render/src/renderer.rs
crates/es-render/src/lib.rs
crates/es-render/slang/common.slang
crates/es-render/slang/raster.slang
crates/es-render/tests/render.rs
tests/golden/render/cornell_rs_full_rgb8.*
crates/es/src/cmd/showcase.rs
crates/es/tests/video.rs
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M7/R2-rs-look.md
docs/packets/M7/R2-rs-look.ko.md
```

`crates/es-render/src/{view.rs,cpu.rs,renderer.rs,lib.rs}`, `crates/es-render/slang/{common.slang,
raster.slang}`, `crates/es-render/tests/render.rs` (new tests + the extended generator),
`tests/golden/render/cornell_rs_full_rgb8.*` (new, generated), `crates/es/src/cmd/showcase.rs`
(the `--look` flag), `crates/es/tests/video.rs` (a `--look` argument test), `docs/design/renderer*.md`,
`docs/packets/M7/R2-rs-look*.md`.

## oracle

1. Every existing golden and GPU test unchanged; `cargo xtask verify-goldens` 0 changed. First
   and last.
2. `cargo test -p es-render lambert_is_the_default_and_is_byte_identical` — `RenderConfig::rs()`
   has `Shading::Lambert`, and a render through the new code path with `Lambert` equals the golden
   bitwise (this is oracle 1 stated as a unit test at the API).
3. `cargo test -p es-render cpu_full_shading_reproduces_its_golden` and
   `gpu_full_shading_matches_the_cpu` (GPU; `SKIP` without a device) — `Rgb8` bitwise, and the
   three geometry channels of `Full` bitwise equal to the `Lambert` ones.
4. `cargo test -p es-render a_shadow_ray_darkens_only_occluded_pixels` — on Cornell with
   `shadows: true` vs `false` (all else equal), the set of pixels that changed is non-empty and
   every changed pixel's `any_hit` towards `L` is true on the CPU.
5. `cargo test -p es-render ssaa_is_a_fixed_order_box_filter` — `ssaa: 2` on a one-triangle scene
   equals a hand-computed average of the four sub-samples in row-major order (bitwise on the
   linear value, then encoded).
6. `cargo test -p es --test video showcase_look_flag_is_parsed`; `cargo xtask ci`;
   `cargo xtask check-scope docs/packets/M7/R2-rs-look.md`.

## acceptance

Oracles 1–6 (3 on the local RTX 3060 and on the server). One showcase frame set of V19b's
`nominal-00` rendered with `--look full` on the server under `~/artifacts/plan-v/m7-r2/`, its
ms/frame beside R1's `lambert` number (observation). Design note section 9 records the shading
equations, the SSAA order, the params layout and why the observation path did not get the knob.

## forbidden

Changing `Shading::Lambert`'s arithmetic or the default; giving `EnvRendererCfg` / the
observation path the new look; any existing golden or fixture; `pow`/`exp`/`log` from the host
or `GLSL.std.450`; atomics or shared memory; a second kernel; `crates/es-gpu/**`;
`docs/ARCHITECTURE*.md`. INV-17: no new trait.
