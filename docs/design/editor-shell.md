# Editor shell — read-only layered graph, telemetry, before/after images

Design note for `crates/es-editor` (layer 12). Spec: §23 (editor and integrated debugger),
§23.2 (layered graph view), §23.3 (watching during training), §23.4 (the three graph-feature
stages — read-only is M1), §14.3 (`.eslayout` sidecar), §1.9 (cut 5: the *editable* visual
graph goes, read-only stays), §12.4 (the metric set).

## 1. What the crate is

A **client** (§23.1). It hosts no training, owns no simulation and holds no IR authority: it
opens a bundle, asks each IR to validate itself, and draws what it is told. §4.2 rule 4 makes
that structural — nothing may depend on `es-editor`, so nothing can grow a dependency on a
view.

Stage 1 of §23.4 only: read-only. Editing needs layout persistence, undo/redo, search and
large-graph performance; read-only needs none of them, and §23.4 says most of the debugging
value is already there. It is also the half that survives §1.9 cut 5.

## 2. Split: view-model / egui

| Half | Where | Tested |
|---|---|---|
| view-model | `src/model/{graph_view,telemetry_view,image_view}.rs` | fully, headless |
| egui shell | `src/app.rs`, `src/main.rs` | compiled only |

CI has no display, so everything that *decides* anything lives in `model` and is judged by
`cargo test -p es-editor`; `app.rs` turns positions into rectangles and is judged by
`cargo build -p es-editor`. Keeping a decision out of `app.rs` is the rule, not a preference:
an untested file is allowed to be thin and nothing else.

## 3. No `egui-snarl`

§23.4 names `egui-snarl` for the node graph. A **read-only** view does not need it: snarl
exists for interactive wiring — drag, connect, disconnect, pin hit-testing, an editing-time
node model — and stage 1 draws rectangles and cubic beziers with the `egui::Painter` that
`egui` already ships. One fewer dependency, and no third idea of what a node is beside
`es_ir::Graph` and `LayeredGraph`.

Upgrade path for §23.4 stage 2 (M3, editing): `LayeredGraph` stays exactly as it is — snarl
would replace the ~120 painting lines in `app.rs`, taking `NodeView`/`Ports` as its node model
and `Layout` as its position store. Nothing in the view-model is shaped by the painter's
absence, so adopting snarl is a change to one file.

## 4. The layered graph (§23.2)

`LayeredGraph::from_bundle(task, observation, learning, deployment)` produces four stacked
`LayerView`s in that order, each with its nodes (id, the IR's own `IrNode::kind` tag, label,
port names) and its edges taken verbatim from `es_ir::Graph`.

The Deployment IR is a record, not a graph (§9.2), so its band is four synthetic nodes in
execution order — `ActionContract → SafetyEnvelope → Watchdogs → Fallback` — which is the
action/safety part of §23.2's picture.

**Cross-IR edges mirror `es_ir::cross`.** The rules stay there; this draws the same joins
`cross::check` diagnoses:

| Edge | Joined on | Rule |
|---|---|---|
| Task `ObservationSpec` → Observation source | sensor / body id, or `Language` | §7.4 (`XIR-001/002`) |
| Observation output → Learning contract input | tensor name | §8.4 (`XIR-010`) |
| Learning action output → deployed action contract | port name | §8.5, §9.2 (`XIR-020..022`) |

`diagnostics` is every IR's own `validate()` plus `cross::check`, so the picture and the
complaint list come from one pass. A bundle that does not validate is exactly the one someone
opens the editor to look at, so `from_bundle` never fails.

### Layout

`auto_layout` is Sugiyama-lite: longest-path rank → `x`, position within the rank in the
graph's canonical `NodeId` order → `y`, one horizontal band per layer. Deterministic (the same
bundle lays out identically, which is a test) and non-overlapping. Crossing reduction is
skipped — it is what makes such a layout *pretty*, and it is also what needs a careful stable
tie-break to stay deterministic.

`apply_layout(&Layout)` overrides positions from an `.eslayout` sidecar (§14.3). **The editor
reads layout and never writes it into an IR** (§4.2 rule 7): `Layout` lives in
`es_ir::serial`, no IR type has a position field, and an IR's `*_hash` cannot see one. A
sidecar is keyed by `NodeId` alone, which the four IRs do not share a namespace for, so a
position applies to that id in every layer — per-IR sidecars are a stage-2 question, when
*saving* layout starts to matter.

## 5. Telemetry (§23.3)

`TelemetryModel` consumes `es_telemetry::protocol::Message` from a
`Box<dyn FnMut() -> Option<Message>>`. That indirection is the whole point: the transport is
another packet's (`es_telemetry::transport`), and plugging it in is one line in `main.rs`.

Kept: a capped `(tick, value)` history per `(stream, component)`, the latest `PerfMetrics`,
and an event log. Histories are plain `Vec`s with a cap, not `es_core::ring` — a viewer
dropping its oldest sample is not on the determinism path, and the ring belongs to the
producer. `pump` is bounded per frame (§23.3 runs the viewer on a budget).

`metric_rows()` is the §12.4 set in a fixed order, end-to-end latency split into p50 and p95.
A single `step/s` figure is forbidden; an unmeasured metric renders as `--`, never `0`.

## 6. Before/after images (§23.3)

§23.3: *"seeing the difference with your own eyes is half of vision debugging"*.
`BeforeAfter::run(obs, input)` compiles the Observation IR with `es_compile::CpuPlan` — the
same CPU reference the GPU lowering is judged against (§11.3), not a second preprocessing
implementation — runs it on one frame, and pairs the raw input with each image output.

Display decoding follows what the CPU kernels define: a `u8` tensor is the sensor's HWC
frame, anything else is the pipeline's CHW float. Float values outside `[0, 1]` (a `Normalize`
output) are rescaled to the tensor's own min/max, so a normalized image is visible instead of
clipped to black; values already in range keep their absolute scale.

A bundle on disk carries no sample frame, so `BeforeAfter::sample(obs)` synthesizes a
deterministic gradient shaped like the graph's declared `ImageInput`. When a telemetry image
stream is attached (§23.3), that frame replaces the gradient and nothing else changes.

## 7. Dependencies and build

`eframe` / `egui` `0.32.3`, pinned in the workspace. Not the newest release (0.36.2): 0.32 is
the last one whose MSRV matches the workspace `rust-version` (1.85). Features are minimal —
`glow` + `default_fonts` + the two Linux windowing backends `x11` / `wayland`; no accesskit, no
wgpu, no persistence — which keeps the incremental rebuild of this crate at ~2.4 s.

`x11` / `wayland` are required on Linux, not only for a runnable binary: with neither, winit
0.30 stops at its own `compile_error!` ("The platform you're compiling for is not supported by
winit"), so even `cargo clippy --workspace` fails there (§26.1 makes Linux x86_64 the primary
platform). Both are enabled because a native Wayland session and a plain X11 host each need
their own backend. Neither needs a system `-dev` package at build time — `cargo check -p
es-editor` succeeds with the pkg-config search path emptied — and on Windows/macOS both are
no-ops. Opening a window on Linux has not been exercised (no display on the verification
host): `Status: unverified`. Packet: `docs/packets/M4/P-M4-R9.md`.

## 8. Not here

- 3D scene view, state streaming, pause/step/rewind, hot-patching (§23.3) — they need a
  running process on the other end of the transport.
- Node-level live values and the reward → exposure back-trace of §23.2. The structure they
  hang on (`LayeredGraph` + `TelemetryModel` in one crate) is in place; wiring a stream to a
  node is the next packet.
- The §28.7 gate-9 number (telemetry + graph view cost < 1% of training throughput):
  `Target / Status: unverified` — it cannot be measured from a viewer with no producer.

---

## 9. Stage 2: the editable graph (§23.4, M3 W6)

Read-only stayed; editing was added beside it, not on top of it. `LayeredGraph` is untouched —
the prediction in §3 held: adopting an editing model was a change to `app.rs` plus two new
view-model files.

| Half | Where | Tested |
|---|---|---|
| edit model | `src/model/edit.rs`, `src/model/palette.rs` | fully, headless |
| canvas | `src/app.rs` (`edit_canvas`, `CanvasView`) | compiled only |

`EditSession` owns one IR (`EditIr::{Task, Observation, Learning}` — Deployment and Evaluation
are records, not graphs), its `.eslayout` `Layout`, the two node registries, the diagnostics
list, and the undo/redo stacks. Six edits cover everything the UI can do: `AddNode`,
`RemoveNode`, `Connect`, `Disconnect`, `SetParam`, `MoveNode`.

### What editing must not do

- **`MoveNode` is layout-only.** It writes `Layout::positions` and nothing else; a test moves a
  node three times and asserts `task_hash` is unchanged, and a second test asserts the
  `.esgraph` half of `save()` is byte-identical before and after a move (§4.2 rule 7, §14.3).
- **No per-kind code in the editor.** Nodes come from `TaskNodeRegistry::create` /
  `LearningNodeRegistry::create`, parameters are replaced by re-serializing the node and handing
  the table back to the factory, and the add-node menu is `Palette::from_registries`. Adding a
  node kind to `es-ir` is visible here with no edit at all. See `docs/design/node-sdk.md`.
- **No new abstraction.** `INV-17` allows two node factories and the SDK is exactly those two
  plus `NodeSchema`. The consequence is that Observation IR has no factory, so `AddNode` and
  `SetParam` on an Observation graph report `FACTORY-001` while its other four edits work.

### Validation, and what "refused" means

Every edit runs both checks: `EditIr::validate()` (the IR's own full pass) fills the advisory
diagnostics list, and `Graph::validate_declared_ports()` decides the edit's fate. An edit that
introduces a **new error** there — a port type mismatch (`TYPE-003`), an unknown port
(`GRAPH-010`), a second edge into one input (`GRAPH-003`) — is reverted and returned, and it
never enters the history. Errors that were already in the graph stay: an editor that refuses to
work on a broken graph is an editor nobody can fix a graph with, and a half-authored graph is
exactly the one someone opens the editor for.

`RemoveNode` takes every incident edge with it, so a removal cannot leave the dangling endpoint
that `GRAPH-002` would then refuse the removal for.

### Undo

Whole-state snapshots, one per edit, not per-edit inverses. Undoing `RemoveNode` by inverse has
to restore the node, its parameters, its layout entry and every incident edge — four chances to
be subtly wrong, and subtly wrong undo is worse than fat undo. An authored graph is hundreds of
nodes; the `ponytail:` comment in `edit.rs` names the upgrade path if that ever stops being
true. A test round-trips three edits through undo and redo and compares the IR, the layout and
the hash.

### Still no `egui-snarl`

Re-decided with editing in hand, and the answer did not change. What snarl would replace is
`CanvasView` plus the painting loop: ~190 lines of hit-testing pins, dragging rectangles and
drawing beziers, which is not enough to earn a dependency that brings its own node model,
its own layout store and its own idea of a port — three concepts that already exist here as
`es_ir::Graph`, `es_ir::serial::Layout` and `es_ir::Port`. The gesture-to-`Edit` mapping, which
is the part that would actually be hard to get right, is not something snarl does for us: it is
`EditSession`, and it is tested.

### Not here

- Search, minimap, multi-select, copy/paste, box-select (§23.4 lists search and large-graph
  performance as stage-2 needs; the single-selection canvas is what the six edits need).
- A parameter inspector panel. `Edit::SetParam` and `NodeSchema` are both in place and tested;
  the widget that fills a `ParamType` in is UI work with no model behind it left to design.
- Editing Observation IR nodes, and the Control Graph (IR-C) — stage 3, M4.

---

## 10. The Run tab: a finished run, opened (§23.3, §10.5, M7/E1)

Every `es eval run` / `es loop collect` writes `report.json`, `evaluation.lock`, `events.json`,
`traj/<cell>.estraj` and — with `--frames` — `frames/<cell>/NNNNNN.bin`. Until now the only
readers were `es video mosaic` and a person with `jq`. The **Run** tab opens the directory.

`es-editor <run-dir>` and the File field take one path; a directory holding `report.json` is a
run and anything else is a bundle (`RunView::is_run_dir`). Told apart by what is on disk, not
by a flag: a run directory and a bundle directory cannot be confused, and a person who typed
the wrong one gets the other view's error, not a mode.

### `model/run_view.rs`

| Call | Gives |
|---|---|
| `RunView::open(dir)` | the `EvaluationReport`, the `BTreeMap<String, Vec<es_eval::runner::StepEvent>>` of `events.json`, and a listing (never a load) of `traj/` and `frames/` |
| `cells()` | one `CellRow` per **episode**: name, suite, seed, the suite's metrics by the report's own names, whether a trajectory and frames exist |
| `columns()` / `sort_by(i)` | the table's headers, and a stable sort by any of them |
| `timeline(cell)` | per tick the `EventSource` and the decoded `EventSet`, plus per-kind totals and each kind's first tick |
| `Timeline::buckets(n)` | the same ticks folded into `n` columns, for drawing at any width |
| `acceptance()` | `report.acceptance` verbatim |
| `filmstrip(cell, 8)` / `frame(cell, i)` | at most eight evenly spread frame indices, and one decoded `Rgb8Image` |
| `set_frames_root(dir)` / `frames_root()` | where `<cell>/NNNNNN.bin` is looked for; `<run>/frames` on open |
| `selected_cell()` / `select(name)` | the selection, which is the whole coupling to the replay panel (§11) |

**Two things are called a cell** and the file keeps them apart. `es_ir::evaluation::CellResult`
is one *suite × metric* of the §10.1 table; a cell on disk — an `events.json` key, a
`frames/<cell>/`, a `traj/<cell>.estraj` — is one *episode*, named `<suite>-<NN>` by
`Evaluation::run_shard`. A `CellRow` is the episode and carries the metrics its suite measured,
so the same three numbers appear on both rows of a two-episode suite. The alternative — a row
per `CellResult` — is the report table, and it is not what someone who wants to watch episode
`nominal-01` is looking for.

**Metric names are the report's.** `columns()` is the union of the metric names the report
carries, so a metric added to `MetricSpec` appears with no change here. Nothing is hard-coded,
and a `Histogram` or an `Unavailable` value is shown as what it is, never as `0`.

**Seeds come from `evaluation.lock`.** `report.json` carries none. The lock's `seeds[i]` is
selected by the `NN` of the cell name, which is the episode index `run_shard` counts with.
Without the lock the column is `--`: an invented seed is worse than a missing one.

**The violation bits are decoded, not re-derived.** `StepEvent::events` is
`es_safety::EventSet::bits()`. `EventSet` has no `from_bits` and `es-safety` is not this
packet's to change, so `decode_events` re-inserts through `ViolationKind::index()` — the same
table the encoder used — and a test round-trips all 14 kinds.

**Buckets are contiguous, gapless and total**, which is what makes their per-kind counts sum
back to the timeline's totals at any `n` (the oracle checks `n ∈ {1, 7, 64}`). A bucket shows
the most severe source it covers (`Policy < Human < Clamped < Fallback`), so one clamped tick
in a 400-tick episode is still a visible mark rather than a rounding loss.

**A run records two clocks and the summary names both.** An `events.json` record is one
*frame* — one control step — and carries the `PhysTick` that frame ran at; at the demo's rates
that is four physics ticks per frame, so "first at tick 544" is nowhere on a 224-column strip.
`Timeline::kind_rows()` returns `KindRow { kind, frames, first: FirstSeen { frame, tick } }` and
`KindRow::label()` writes *"Velocity: 2 frame(s), first at frame 1 (tick 1)"* — the strip's own
index first, the physics tick after it; `Timeline::heading(cell)` counts the same unit
(*"nominal-00: 224 frame(s)"*). The wording is in the model with a test, not in `app.rs`,
because which clock a person is being shown is a decision (§28.10 rule 3).

**Frames are wherever `--frames` pointed.** `es eval run --frames <dir>` writes a directory that
is usually a *sibling* of the run, not `<run>/frames`, so a real run's table showed `frames 0`
for every cell until the tab was pointed at it. `set_frames_root(dir)` moves the root and
rebuilds the rows — which also means a report that stands alone gains its cells from an
external frames directory — and `app.rs` gets one `Frames` field beside `Scene`, filled with
`frames_root()` on open and applied when it loses focus.

**The table, the acceptance rows, the strip and the filmstrip are one vertical scroll area**
(`auto_shrink([false, false])`, so it fills whatever the replay panel leaves), and the
filmstrip has its own horizontal one: eight 160-px thumbnails are wider than a narrow window,
and a cut-off frame looks like a missing frame.

### What it deliberately does not do

- **No live process.** The tab reads a finished directory. Attaching to a running
  `es eval run` is E4's `--telemetry`, and it is a different transport.
- **No editing of a run.** Nothing is written back: an artifact that the editor could rewrite
  is an artifact the hash chain cannot vouch for (§5.3, §10.5).
- **No caching.** `frame()` decodes what it is asked for; `app.rs` keeps the eight filmstrip
  textures of the selected cell and drops them when the path changes.
- **A partial run still opens.** Only `report.json` is required. What is missing is named in
  the status line, because the run someone opens the editor for is often the one that did not
  finish writing. A malformed `events.json`, though, is an error rather than a guess (§25.1).

The fixture is `tests/fixtures/visible-learning/run/`: two suites × two episodes, four ticks
each with two `Clamped` and one `Fallback` tick carrying real violation bits, one 96×96 frame
per cell. It is written by the `#[ignore]`d `generate_fixture_run`, which builds an
`EvaluationReport` and an `EvaluationLock` and hands them to `es_eval::runner::write_artifacts`
and a `FrameSink` — so the bytes are the producing types' own. Only the frame `.bin` and its
`layout.json` are written directly, because `es_eval`'s writer for them is private; the shape
is that writer's documentation.

---

## 11. The Replay panel: an `.estraj` played in the editor (§23.3, M7/E2)

§23.3 asks for an editor that "locally replicates the scene and receives only the pose/joints".
A recorded trajectory is that stream, offline. `es video showcase` already proves a run can be
re-rendered from the scene file and the `.estraj` alone — this does the same on the CPU, in an
`egui` canvas, at interactive rate: **no physics, no GPU, no `ffmpeg`, no server**.

It lives in the Run tab as a bottom panel (§10's table picks the episode, this plays it). The
whole coupling is `RunView::selected_cell()`; the panel adds one text field, because a run
directory does not carry its scene file — the same `--scene` the showcase takes.

**How tall it is is the model's answer, not a widget's.**
`replay_view::panel_height(replay, available)` returns `None` while nothing is loaded — the
panel is then its control rows and nothing else — and 45% of the tab once
`ReplayView::is_loaded()` holds, with a 320-px floor so the canvas is worth looking at and an
80% ceiling so the table above it does not vanish. The first cut took the 45% unconditionally
and an empty canvas clipped the filmstrip above it. `app.rs` gives the two states their own
panel id (`replay-controls` / `replay-canvas`), because egui remembers a panel's dragged
height per id and the two want their own; a run that changes drops the replay and the panel
shrinks back by itself.

### `model/replay_view.rs`

| Call | Gives |
|---|---|
| `ReplayView::open(scene, traj)` | the `SceneDesc` (MJCF or URDF by extension, as `es backend`'s `load_scene`) and the `Trajectory`; tessellates tick 0 so an unsupported geom is an error here, not a blank canvas later |
| `ticks()` / `qpos(t)` / `scene_at(t)` | the length, the joint state, and the tick's `TriScene` — the very call the showcase renders |
| `project(t, &Camera)` | `Vec<Tri2d>`: three screen points, one flat `[u8; 3]`, a depth key and the source triangle index, **sorted back to front** |
| `Camera::view()` | the `es_render::CameraView` those coordinates are in |
| `Camera::orbit(dyaw, dpitch)` / `zoom(f)` | a new camera on the sphere about `look_at`; pure, `#[must_use]`, no interior state |
| `advance(dt, rate_hz)` / `step(±n)` | playback, clamped at both ends; `playing`, `tick` and `speed` are the model's |

**The projection is `es_render`'s own, inverted.** `ViewParams::new(&camera.view())` gives the
renderer's `f32` camera; a world vertex goes through `es_render::cpu::quat_rotate_inv` and the
same `fx, fy, cx, cy` that `cpu::primary_dir` casts rays with. Nothing here re-derives the
convention (§3.1: `OpenCV` camera frame, image origin top-left).

**The shading is `es_shade_lambert`, copied in four lines**, because `cpu::shade_lambert` is
private: `ambient + max(dot(n, light), 0) * (1 - ambient)`, times albedo, plus emission, then
`srgb_encode` (public) and the renderer's rounding. The multiply and the add are separate, not
a `mul_add`: a fused one rounds once where the renderer rounds twice. `RenderConfig::rs` is
built here purely so the light direction and the ambient floor are the renderer's constants and
not a second copy of them. `flat_shade_matches_the_renderer` puts one triangle in front of one
camera, rasterizes it with `es_render::cpu::rasterize`, and asserts the pixel under the
projected centroid equals the colour — which pins the camera, the projection and the shading in
one assertion.

**Near-plane clipping, whole triangles.** A triangle with any vertex at or behind the near
plane is dropped rather than divided by that `z`, which would project it through the eye onto
the far side of the image. One that straddles the plane disappears instead of being split: a
clipper that splits has to interpolate the vertices, and at the scale of this camera the arm is
never half behind it.

**`ponytail:` the painter's algorithm is the deliberate simplification.** Sorting whole
triangles by centroid depth is exact for convex primitives that do not interpenetrate, and
wrong exactly where they do — a gripper closed on a cube can show the wrong face. The upgrade
path is `es_render::cpu::rasterize` per pixel at a low resolution, which R1's BVH is what makes
affordable; the camera the model builds is already a `CameraView`, so that swap is one
function. The sort is stable on the depth alone, so ties keep triangle order and the emitted
sequence is a pure function of (trajectory, camera) — which is what makes
`tests/golden/editor/replay_tick0_order.json` (2,754 indices at tick 0) a golden worth keeping.

### `look_at` is repeated, not reused

`es_env::render::look_at` is behind `es-env`'s `render` feature, and that feature pulls in
`es-render` **and `es-gpu`**: enabling it in the editor would link Vulkan into a viewer that
renders nothing on a GPU, which E2 forbids. Moving the function out from behind the feature is
not a pure move either — it returns `es_render::CameraView` and calls
`es_render::ImageSpec::pinhole`, both from an optional dependency. So `Camera::view` repeats
the arithmetic (basis from forward × world-up, then Shepperd's quaternion) over the same
`es-render` types, and the test above pins it against the renderer rather than against the
copy's source. If `es-env` ever makes `es-render` non-optional, this becomes a one-line
delegation.

### The fixture trajectory

`tests/fixtures/visible-learning/run/traj/nominal-00.estraj`: 48 ticks of
`tests/fixtures/mjcf/so101_pick_place.xml`, 36 kB, written by the `#[ignore]`d
`generate_fixture_traj`. **No physics backend is involved and none is needed.** `Trajectory` is
shaped by an `es_physics_core::backend::ModelInfo` and filled from a `StateView`, both plain
structs: the generator builds a `ModelInfo` with one body index per scene body and `nq`/`nv`
summed from the joints in `MuJoCo`'s own layout, then per tick composes every body's world pose
from the scene's parent chain with `shoulder_pan` swept from -0.6 to +0.6 rad (an MJCF hinge
turns the child frame about its `axis` through its `anchor`: `body.pose * T(a) * R * T(-a)`),
and pushes a `StateView` holding those poses. That is the same composition
`es_render::scene::world_poses` does — private there, twelve lines here, in test code only.
`es-physics-core` is a **dev-dependency** for exactly this reason: the editor itself never
names those types.

The trajectory is a synthetic home-pose sweep, not a policy rollout: what it has to exercise is
that a recorded pose stream re-poses the scene and projects, and a real V19b `.estraj` (which
the orchestrator opens) is the same bytes at a different arm.

### Not here

- **Per-pixel anything**: no depth buffer, no shadows, no textures. The upgrade path above.
- **The run's own control rate.** The panel plays at 50 Hz, the demo deployment's
  `rate.control`; a run directory carries no Deployment IR to read it from, and the wrong rate
  only changes how fast the arm appears to move. A `--rate` would be a field on the panel the
  moment a run writes its rate down.
- **Camera presets and scene cameras.** `--camera NAME` (the showcase's other mode) needs the
  scene's own camera list; the free camera is what a person dragging a mouse wants.
