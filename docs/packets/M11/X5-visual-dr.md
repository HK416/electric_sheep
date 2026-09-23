# M11 X5 — visual randomization: light, colour, ambient, geom colour, camera pose and field of view

Spec: §28.14 rule 4 (draws keyed by `(seed, env, episode, stream)`, recorded in the episode; the same
draw renders the same frame bitwise, GPU == CPU; a drawn field of view moves that episode's `ImageSpec`
intrinsics, INV-14) and rule 1 and wave 2, §6.3 (`Randomization`), §7.2 (`ImageSpec`), §10.2 (the eval
perturbations stay as they are), §15. Review: `docs/reviews/M10.md` S-6 (the `Pt` sensor has no
directional light, so `light_direction` measures nothing there). Design notes: `renderer.md` (+ko),
`batch-domains.md` (+ko). Depends on X3 (render overrides reach `Rollout`) and X4 (the randomization
plan's shape after `set_params`). Type B.

## the question

Visual variation exists only as two evaluation perturbations — `LightOverride { intensity, yaw_deg }`
drawn by `EnvRng` in `crates/es-eval/src/perturb.rs:64-257` — and the Task IR's `Randomization` targets
are physical only (`crates/es-env/src/randomize.rs:141-200`). The owner chose lighting (intensity,
direction, colour, ambient), geom colours, camera extrinsics and camera intrinsics. **Can each be a
Task IR `Randomization` target, drawn per episode per env, applied at render time on `Rs` and `Pt`
(with a directional light on `Pt`), bit-reproducible and GPU == CPU, with every document that
declares none of them unmoved?**

## spec

* Targets (the grammar in `randomize.rs` gains a render class; an unknown target stays
  `Unsupported` by name): `light.intensity` (scale), `light.direction` (yaw and pitch, degrees,
  two streams), `light.color` (RGB scale, or a colour temperature in kelvin through a fixed
  table), `light.ambient` (scale), `geom.<name>.rgba` (per-channel scale or HSV jitter, with the
  choice stated in the doc), `camera.<name>.pose` (translation metres and rotation degrees about
  the camera's own axes), `camera.<name>.fov` (vertical fov scale). Draws go into a per-env
  `RenderOverrides` that the frame source reads at render time; the physics step never sees them.
* The `LightOverride` mechanism moves from es-eval into es-env as the one implementation; es-eval's
  `LightIntensity` / `LightDirection` perturbations call it, and their committed reports stay
  byte-identical.
* es-render: `RenderConfig` gains light colour and a directional light on the `Pt` path (NEE
  toward a direction, beside the emissive geoms), in the Slang shaders and the CPU reference
  together. Defaults reproduce today's frames bitwise (white, the `Rs` default direction, and
  `Pt` directional intensity 0 unless declared).
* Camera: the pose draw composes with the sensor's extrinsics; the fov draw rescales `fx`, `fy`
  about the principal point, and the episode's effective `ImageSpec` (both) is written into the
  episode meta and the frame's `layout.json`, so a consumer of intrinsics sees the drawn ones
  (INV-14). `Resize` / `Crop` downstream still transform them.
* Every draw is recorded in `EpisodeMeta` beside `param_scales`.

## context

```
crates/es-env/src/randomize.rs
crates/es-env/src/render.rs
crates/es-env/src/env.rs
crates/es-env/src/episode.rs
crates/es-env/tests/**
crates/es-eval/src/perturb.rs
crates/es-eval/tests/**
crates/es-ir/src/task.rs
crates/es-ir/tests/**
crates/es-render/src/**
crates/es-render/shaders/**
crates/es-render/tests/**
crates/es-py/src/rollout.rs
tests/golden/render/dr_*
tests/fixtures/rl/**
docs/design/renderer.md
docs/design/renderer.ko.md
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/packets/M11/X5-visual-dr.md
docs/packets/M11/X5-visual-dr.ko.md
```

## oracle

1. `cargo test -p es-env visual_randomization_` — each target parses and resolves; the same
   `(seed, env, episode, stream)` draws the same values; different episodes differ; undeclared
   targets move no hash.
2. `cargo test -p es-render --test render dr_` (GPU) — for each target, a drawn frame: GPU == CPU
   reference at the parity rule of its path (0 ULP where today's rules say 0); the default
   (undrawn) frame byte-identical to every existing golden; new `dr_*` goldens from the CPU
   reference only; `Pt` with a directional light is lit (non-zero where the sky alone was zero).
3. `cargo test -p es-eval` — the committed reports that use `light_intensity` / `light_direction`
   are byte-identical after the move.
4. `cargo test -p es-env camera_fov_draw_moves_intrinsics` — the episode's recorded `fx`, `fy` equal
   the nominal × the drawn scale; a downstream `Resize` transforms them (INV-14).
5. fmt, clippy `-D warnings`, check-scope, verify-goldens; both GPUs.

## acceptance

Oracles 1–5; `renderer.md` gains the directional-light and colour sections and `batch-domains.md`
the render-target grammar (+ko).

## forbidden

Textures / materials (M7 R6); occluders; changing any committed frame, report or hash; the
physics seeing a render draw; a vendor denoiser.
