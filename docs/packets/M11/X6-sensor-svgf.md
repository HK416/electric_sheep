# M11 X6 — the `Pt` sensor declares SVGF

Spec: §28.14 rule 5 (`Pt` observation noise is handled by declared document fields only —
`render.seed = "tick"` and `render.svgf` — never by temporal accumulation on the observation path or
by a vendor denoiser) and rule 1 and wave 1, §6 (the `render` table), §15.3, §3.4. Design notes:
`renderer.md` 4.3 (SVGF; without a history the luminance weight is exactly 1.0) and 12.3 (why the
observation path has no accumulation), 12.8 (the tick seed); `visible-learning.md` 7.37 (U5).
Depends on W1a (merged). Type B.

## the question

`es_env::render::sensor_cfg` maps `SensorPath::Pt { spp, bounces }` to
`RenderPath::Pt { nee: true, restir: false, svgf: false }` — SVGF is hard-coded off. A single-frame
SVGF (depth/normal edge-stopping, luminance weight 1.0 without history) is already bit-identical GPU
== CPU. **If the sensor can declare `svgf = true`, does the observation frame stay a pure function of
`(pose, episode, tick)` — collector == evaluator bitwise, GPU == CPU — with every committed task hash
unmoved?**

## spec

* `es_ir::task::SensorRender` gains `svgf: bool` (default `false`),
  `#[serde(default, skip_serializing_if = "is_false")]`, written by `canonical` **only when true**,
  after `seed` — exactly W1a's `seed` pattern (`crates/es-ir/src/task.rs:752-804`); `is_default()`
  includes it. Valid only on `Pt`; `svgf = true` on `Rs` is a validation error with a new code.
* `sensor_cfg` maps it to `RenderPath::Pt { svgf }` with `svgf_iterations` at `RenderConfig`'s
  default (4) and `temporal: None` (unchanged: no accumulation on this path).
* `es video showcase --task` follows the sensor (as it does for `seed`).

## context

```
crates/es-ir/src/task.rs
crates/es-ir/src/validate*.rs
crates/es-ir-types/src/codes.rs
crates/es-ir/tests/sensor_render.rs
crates/es-env/src/render.rs
crates/es-env/tests/render_loop.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/task-pt-tick-svgf.toml
tests/fixtures/visible-learning/observation-pt-tick-svgf.toml
tests/fixtures/visible-learning/evaluation-pt-tick-svgf.toml
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M11/X6-sensor-svgf.md
docs/packets/M11/X6-sensor-svgf.ko.md
```

## oracle

1. `cargo test -p es-ir committed_task_hashes_are_unmoved_by_svgf` — `task.toml`, `task-pt.toml`,
   `task-pt-tick.toml` unmoved; `svgf = false` spelled out hashes as absent; `true` moves it; `Rs` +
   `svgf` is refused by code.
2. `cargo test -p es-env --features render --test render_loop pt_svgf_sensor_` (GPU): collector-path
   and evaluator-path frames at the same `(episode, tick)` bitwise with `seed = "tick"` + `svgf`;
   GPU == CPU reference bitwise on one frame; the SVGF frame differs from the non-SVGF frame (it did
   something); ms/frame measured with and without SVGF.
3. `cargo test -p es --test cli` unchanged; `es ir check` accepts the three new fixtures.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

Oracles 1–4 on the RTX 3060 and the RTX 4090 (GPU queue); `renderer.md` gains 12.9 (+ko) with the
cost table.

## forbidden

`temporal: Some(_)` on the observation path; ReSTIR on the sensor (not bitwise); a vendor denoiser;
moving a committed hash; changing SVGF's kernel or constants.
