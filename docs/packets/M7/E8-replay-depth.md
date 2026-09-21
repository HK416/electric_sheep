# M7 E8 — the Replay panel draws with a depth buffer

Spec: §23.3 (the 3D view; "the editor clones the scene locally and renders the selected env's
poses itself"), §28.10 rule 3 (headless first), §1.4 (goldens). Owner note 2026-09-21: *the
polygons in the rendered view look wrong — as if there were no depth buffer.* Design note:
`docs/design/editor-shell.md` section 11 (E2, the Replay panel) is amended in place. Depends on
**E2** (`ReplayView`, `Camera`, `TriScene::from_scene_with_poses`, the projection).

## the question

E2 projects every triangle, sorts them back to front by centroid depth and hands `app.rs` an
`egui::Mesh` — the painter's algorithm, marked `ponytail:` at the time. The SO-101 arm is links
that interpenetrate at every joint, so a whole-triangle sort is wrong wherever two links overlap:
a base plate is drawn over the shoulder, a finger through the wrist. **Can the panel rasterise
with a per-pixel depth test on the CPU, at panel resolution, fast enough to play at 50 Hz, with
the same camera, the same shading and a golden that pins the pixels?**

## spec

* **`model/replay_view.rs` rasterises.** `Projected` (screen-space triangles, kept — it is what
  the tests and the golden of E2 pin) gains `Raster::draw(&Projected, w, h) -> Raster { w, h,
  rgb: Vec<u8>, depth: Vec<f32> }`: a scanline / edge-function rasteriser with a per-pixel
  depth test (nearest wins, interpolated depth from the three vertices' camera-space `z`, not the
  centroid), flat Lambert per triangle exactly as `project` shades today, the same near-plane clip,
  `f32` throughout, fixed iteration order (triangles in `Projected` order, pixels row-major), so
  the image is a pure function of `(trajectory, tick, camera, w, h)`. No SIMD, no threads — a
  ~3,000-triangle scene at 960×540 is a few million edge tests, which is milliseconds.
* **The panel shows a texture.** `app.rs` uploads the `Raster` as an `egui::ColorImage` texture
  once per tick change (or camera change) and draws it scaled to the panel; the raster's
  resolution is the panel's size capped at 960×540 (`Raster::size_for(panel)`, a model function).
  The orbit/zoom gestures, the scrubber, play/pause are untouched. `egui::Mesh` and the sort go;
  the `ponytail:` comment goes with them.
* **Golden.** `tests/golden/editor/replay-tick0-320x180.bin` (+ `.json` with the camera): the
  demo fixture trajectory (E2's `tests/fixtures/visible-learning/run/traj/*.estraj` or whichever
  E2's tests use) at tick 0 from the default `Camera`, 320×180 `Rgb8`, generated once by an
  `#[ignore]`d generator behind `ES_GENERATE_GOLDENS=1`, then read-only.
* **Back faces.** Draw both sides as the `Rs` path does (the scene's tessellation has no
  guaranteed winding); say so in the note.
* **Not here.** Anti-aliasing, shadows, textures, a GPU path in the editor (the editor links no
  Vulkan — E2's decision stands), picking.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
crates/es-editor/src/model/replay_view.rs
crates/es-editor/src/app.rs
crates/es-editor/tests/**
tests/golden/editor/replay-*.bin
tests/golden/editor/replay-*.json
tests/golden/editor/replay-sort-order.json
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E8-replay-depth.md
docs/packets/M7/E8-replay-depth.ko.md
```

`replay_view.rs` (the rasteriser and its tests; `project` and the sort-order golden stay, since
`Projected` is still the rasteriser's input — if the sort is now dead code, delete the sort and
retire its golden in the same commit, saying so), `app.rs` (texture instead of mesh), the new
golden pair, the design note, this packet.

## oracle

1. `cargo test -p es-editor a_depth_test_beats_the_painters_sort` — two triangles that cross
   (an X seen edge-on: each is nearer on one side) in two colours: with the sort, one whole
   triangle covers the other; with `Raster::draw` the left half shows one colour and the right half
   the other, asserted on pixels.
2. `cargo test -p es-editor replay_raster_reproduces_its_golden` — the fixture at tick 0 from the
   default camera at 320×180 equals the golden bitwise.
3. `cargo test -p es-editor replay_raster_is_a_function_of_its_inputs` — two draws of the same
   inputs are bitwise equal; a different tick or camera is not.
4. `cargo test -p es-editor replay_raster_is_fast_enough -- --nocapture` — the fixture scene at
   960×540: the median of 20 draws printed; `< 16 ms` on this box is the target (an observation,
   not an assertion — print it).
5. `cargo build -p es-editor`; `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/E8-replay-depth.md`.

## acceptance

Oracles 1–5. Screenshots before/after on the demo run `target/plan-u/demo/out/eval` (scene
`tests/fixtures/mjcf/so101_pick_place.xml`, case `nominal-00`, a tick where the arm is folded)
under the worktree's `target/plan-u/e8/`; the orchestrator repeats the check. Section 11's
`ponytail:` paragraph is replaced by what is now true.

## forbidden

A GPU path or any new dependency in the editor; changing `Camera`, `project`'s projection or the
shading numbers (the sort-order golden of E2 must still pass while it exists); goldens other than
the new pair (and the retired one, if retired); `docs/ARCHITECTURE*.md`. INV-17: no new trait.
