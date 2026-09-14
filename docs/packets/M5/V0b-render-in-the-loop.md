# M5 V0b — the renderer in the env loop: animation, frames to disk, image observation port

Design note: `docs/design/visible-learning.md` section 7; read section 2.2 first — `es-render` is a
complete headless island that **no crate depends on**, and the four gaps listed there are what this
packet closes. Independent of V0 and V4; blocks V1, V2, V3.

This packet exists because plan V's requested order had no home for it: nothing in the repo connects a
renderer to the env loop, writes a frame to disk, or accepts an Observation IR image input.

## context

```
crates/es-render/src/scene.rs
crates/es-render/src/atlas.rs
crates/es-render/src/lib.rs
crates/es-render/tests/render.rs
crates/es-env/src/render.rs
crates/es-env/src/env.rs
crates/es-env/src/domains.rs
crates/es-env/src/lib.rs
crates/es-env/Cargo.toml
crates/es-eval/src/runner.rs
crates/es-eval/Cargo.toml
crates/es-eval/tests/evaluation.rs
crates/es-env/tests/render_loop.rs
tests/golden/render/so101_frame0.bin
tests/golden/render/so101_frame0.json
Cargo.lock
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V0b-render-in-the-loop.md
docs/packets/M5/V0b-render-in-the-loop.ko.md
```

Notes: `es-render/src/scene.rs` gains one function and `atlas.rs` one `write_to`; `es-env/src/render.rs`
is new and is the only file that names `es_render`; `env.rs` and `domains.rs` gain the optional field and
the image port; `runner.rs` changes one `EvalError::Plan` arm. `es-env` and `es-eval` gain an optional
`render` feature only — with it off, both crates build exactly as today.

## spec

- §4.2: `es-env` is layer 9 and `es-eval` layer 10; `es-render` (5) and `es-gpu` (2) are below both, so
  `cargo xtask layering` passes. `es-render` gains **no** dependency: it takes a pose map, not a
  `StateView`, so layer 5 still knows nothing of `es-physics-core` (3).
- §15.2: one atlas, N tiles, closed-form `tile_origin`; frames come out of the existing
  `Atlas::read_tile`.
- §15.4 is not implemented in this build (no TLAS/BLAS, `crates/es-render/src/lib.rs:11-14`); this packet
  does not change that, and the design note states the re-tessellation ceiling.
- §7.2, §26.1, INV-14: the renderer's `ImageSpec` subset is **checked against** the Observation IR's
  declared `ImageSpec`, never substituted for it and never resampled to fit
  (`crates/es-render/src/renderer.rs:292-302`). A mismatch is an error naming the field.
- §3.4, §3.5: the CPU reference path is the determinism claim; the GPU path is bit-compared against it on
  a device that already passes the existing GPU oracles, and nothing else is claimed.
- §10.1: an image input the runner cannot serve stays an error rather than becoming zeros — this packet
  narrows the condition, it does not remove the refusal
  (`crates/es-eval/src/runner.rs:421-428`).
- §1.5: `es-env` 2,060 and `es-eval` 2,139 code lines today; this packet is budgeted under ~800 across
  the four crates.
- INV-17: `EnvRenderer` is a concrete struct. No trait is added.

## oracle

```
cargo fmt --check
cargo clippy -p es-render -p es-env -p es-eval --all-targets -- -D warnings
cargo clippy -p es-env -p es-eval --features render --all-targets -- -D warnings
cargo test -p es-render
cargo test -p es-env
cargo test -p es-env --features render --test render_loop
cargo test -p es-eval --features render
cargo xtask context-budget
cargo xtask layering
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

Reference — the GPU leg, on a machine with a Vulkan device and `slangc`:

```
cargo test -p es-env --features render --test render_loop -- --nocapture
```

The CPU path runs everywhere and is the golden. The GPU path prints `SKIP <test>: <reason>` with no
device or no `slangc`, exactly as `crates/es-render/tests/render.rs:57-72` already does; when it ran it
prints `RAN render_loop_gpu`.

Golden generation (ignored by default, as `generate_goldens` already is at
`crates/es-render/tests/render.rs:165-195`): the new `tests/golden/render/so101_frame0.{bin,json}` is
produced by the **CPU** reference from V0's fixture scene at a fixed `qpos`, so no driver's arithmetic is
committed.

`crates/es-env/tests/render_loop.rs`:

- `poses_from_state_move_the_triangles` — with a body translated by a known offset in the pose map, every
  triangle of that body's geoms moves by exactly that offset and no other triangle moves.
- `an_absent_body_keeps_its_scene_pose` — a pose map missing a body reproduces `from_scene` for it, so a
  partially-known state degrades to the static scene rather than to the origin.
- `cpu_frame_matches_the_golden` — the frame rendered from V0's fixture at the fixed `qpos` is byte-equal
  to `tests/golden/render/so101_frame0.bin`.
- `two_runs_of_the_same_state_are_byte_identical` — the determinism claim of design note section 9,
  on the CPU path.
- `gpu_frame_matches_the_cpu_frame` — bit equality on this device; `SKIP` otherwise.
- `frames_written_to_disk_round_trip` — `write_to` then re-read equals `Tile::to_bytes`, and the sidecar's
  `dtype`, `shape` and `layout` match the existing `tests/golden/render/*.json` schema.
- `the_declared_image_spec_is_checked_not_coerced` — an Observation IR `ImageInput` whose `ImageSpec`
  differs from the renderer's in size, channel count or colour space is an error naming the field; no
  resize happens.
- `without_a_renderer_an_image_input_is_still_refused` — with the feature off (or `EnvRenderer` absent),
  `crates/es-eval/src/runner.rs` returns the same `EvalError::Plan` it returns today, with the same
  message. Nothing is ever zero-filled.
- `an_env_without_a_renderer_is_unchanged` — an existing `es-env` test suite run with `--features render`
  but no `EnvRenderer` produces byte-identical episode records to the run without the feature.

## acceptance

```rust
// crates/es-render/src/scene.rs
impl TriScene {
    pub fn from_scene_with_poses(
        scene: &es_assets::scene::SceneDesc,
        world: &std::collections::BTreeMap<es_core::StableId, es_math::Pose>,
    ) -> Result<Self, RenderError>;
}

// crates/es-render/src/atlas.rs
impl Tile {
    /// Writes `<dir>/<stem>.bin` (this tile's `to_bytes`) and `<dir>/<stem>.json`
    /// (`dtype`, `shape`, `layout`), the layout `tests/golden/render/*` already uses.
    pub fn write_to(&self, dir: &std::path::Path, stem: &str) -> std::io::Result<()>;
}

// crates/es-env/src/render.rs   (cfg(feature = "render"))
pub struct EnvRendererCfg {
    pub camera: es_core::StableId,
    pub width: u32,
    pub height: u32,
    pub channel: es_render::Channel,
    pub path: es_render::RenderPath,
    pub frames_dir: Option<std::path::PathBuf>,
}
pub struct EnvRenderer<'gpu> { /* Renderer, cfg, cached TriScene, frame counter */ }
impl<'gpu> EnvRenderer<'gpu> {
    pub fn new(gpu: &'gpu es_gpu::Gpu, scene: &SceneDesc, cfg: EnvRendererCfg) -> Result<Self, EnvError>;
    /// Re-poses the scene from `state`'s `xpos`/`xquat` for `env`, renders, and returns the tile.
    /// Writes `<frames_dir>/<NNNNNN>.bin` + `.json` when `frames_dir` is set.
    pub fn frame(&mut self, model: &ModelInfo, state: &StateView<'_>, env: u32)
        -> Result<es_render::Tile, EnvError>;
    /// The renderer's `ImageSpec` subset, for the caller to check against the declared one.
    pub fn image_spec(&self) -> es_render::ImageSpec;
}
```

- `Env` gains `render: Option<EnvRenderer<'_>>` behind the feature, and `domains.rs`'s observation input
  map gains the image port when it is `Some`. With `None`, byte-identical behaviour to today.
- `es-eval`'s runner accepts an `ImageInput` **only** when a renderer is present and the declared
  `ImageSpec` matches; otherwise the existing `EvalError::Plan` message is unchanged.
- The image tensor handed to `CpuPlan::run` is the tile's bytes in the declared layout. No resampling, no
  colour conversion, no channel reorder happens in `es-env`: a mismatch is an error, and any conversion
  is an Observation IR node (§7.2).
- No new trait; no `HashMap`; no new external dependency in any of the four crates; `cargo xtask layering`
  and `cargo xtask context-budget` pass with and without `--features render`.
- ≤ ~800 source lines across `es-render`, `es-env` and `es-eval`.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` — no new observation node, no `ImageSpec` field.
- Implementing `MultiViewPack` (`crates/es-compile/src/plan.rs:545` rejects it; the demo uses one camera).
- Resolving `Shape::Mesh` or `HeightField` in `es-render` (`crates/es-render/src/scene.rs:201-202`): V0's
  fixture is primitives only, and an asset resolver is its own packet.
- A PNG, JPEG or video encoder, or any new image dependency: the frame format is the existing raw
  `.bin` + `.json` sidecar (design note section 7.2).
- `crates/es-eval/src/perturb.rs` — enabling `LightIntensity` / `LightDirection` is V3.
- `crates/es-data`, `crates/es/src/cmd/*.rs` — the CLI `--frames` flag is V3, dataset frames are V1.
- Making `es-env`'s `render` feature default, or making `es-runtime-embedded` or `es-ros2` reach Vulkan.
- Editing `tests/golden/render/cornell_*`; generating the new golden on the GPU path.
