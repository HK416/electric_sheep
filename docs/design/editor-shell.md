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
`glow` + `default_fonts` + the two Linux windowing backends `x11` / `wayland`; no accesskit and
no wgpu — which keeps the incremental rebuild of this crate at ~2.4 s. `persistence` is added
by this crate alone, for the recent-files list (section 12).

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

- Multi-select, copy/paste, box-select, minimap — section 12, which is where search and the
  parameter inspector went (M7/E3) and where the four of them are argued out.
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

---

## 12. Editing like a person does: the inspector, search, and opening (§23.4, M7/E3)

§23.4 names three things stage 2 needs that stage 2 did not have: **parameters**, **search**,
and **large-graph performance**. Two of the three are here. `Edit::SetParam` and `NodeSchema`
had been in place and tested since M4 with no widget behind them — a node's parameters could
only be changed by editing TOML — and the only way to open anything was to type its path.

Three model files, one rule each (§28.10 rule 3: nothing is decided in `app.rs`).

| Model | Owns | Tested by |
|---|---|---|
| `model/inspector.rs` | which widget a `ParamType` gets, what a person's text means, whether it becomes an `Edit` | `every_param_type_has_one_widget`, `set_param_round_trips_through_the_inspector`, `a_bad_field_never_emits_an_edit` |
| `model/search.rs` | what a query matches, in what order, which hit is next | `search_is_case_insensitive_and_ordered` |
| `model/recent.rs` | what a path on disk is, and the last ten opened | `classify_tells_bundle_documents_and_run_apart`, `recent_is_capped_deduplicated_and_round_trips` |

### The inspector

`Inspector::for_node(session, node)` reads the node's `NodeSchema` from the session's own
registries and its current parameters from **the same re-serialization `Edit::SetParam` does**
before handing the table back to the factory — so what the panel shows and what a `SetParam`
overwrites are one table, and there is no per-kind code here any more than in `edit.rs`
(`docs/design/node-sdk.md`). A kind added to `es-ir` is inspectable with no change.

**One widget per `ParamType`, and the match has no wildcard arm.** `Bool → checkbox`,
`Int → drag int`, `Float → drag float`, `String → text`, `Enum(vs) → combo`,
`Shape → text parsed as [a, b, …]`, `PortType → text parsed as inline TOML`. Adding a variant
to `ParamType` must break this crate's *build*, because the alternative — a wildcard arm — is a
parameter that silently cannot be edited. The seven `Widget`s are pairwise distinct so the
table test is worth running.

**`Field::parse` is the only place text becomes a value**, and an erroneous field never emits
an edit: `Inspector::edit` records the reason on the field and returns `None`, so the session
never hears about `"abc"` in an `Int` box. `Bool` and `Enum` cannot fail from the UI — a
checkbox writes `true`/`false` and a combo writes a variant it was given — but they are
fallible here rather than panicking on text that arrived some other way.

**The fields are the schema's, restricted to the keys the node actually serialized.** An
`Option` field holding `None` is not in the table, and `params_with` refuses a key the node
does not have (`FACTORY-003`), so drawing it would be drawing a dead control. Two smaller
consequences of `es-ir`'s `infer_param_type` having only one observed value to work from:
`ParamType::String` is its catch-all (an array of floats, a datetime), so a field whose current
value is not a string is edited as the TOML it prints as; and an `Enum`'s variant list is
whatever the *example* instance showed, so a value using a variant the schema never saw has it
added to the combo rather than hidden.

**A plain click selects.** Stage 2 set `selected` only on `drag_started`, which was enough
when selection existed to be deleted and dragged; with an inspector hanging off it, a node
that had to be *dragged* before it could be read was the defect that blocked this packet's
acceptance. A click now runs `CanvasView::hit` — the same hit test `start_drag` uses, so what
can be dragged can be clicked — and a click on the background clears the selection. It is
`clicked()`, the primary button only, so the right-click that opens the add-node menu leaves
the selection alone. This is the one piece of `app.rs` with a test of its own
(`a_click_hits_the_node_under_it_and_nothing_on_the_background`): `CanvasView` is positions,
rectangles and names, so it needs no display, and the assertion that the click and the drag
read one geometry is worth more than the section-2 rule of leaving `app.rs` unjudged.

**An edit is meant to be watched.** The status line carries the IR's `*_hash` (its first four
bytes, which is what a person compares at a glance) and the count of what it complains about,
both taken from the session and both live; the Diagnostics tab shows the session's list rather
than the opened bundle's whenever one is open, because `EditSession::apply` re-validates after
every edit and the bundle's list is a snapshot of the files on disk. Changing a `Normalize`
range moves all three at once.

**The panel is rebuilt when the selection moves or the session does** — keyed on
`(selected, history().len())`. Between those the widgets own their text, so typing survives a
repaint; an undo behind the panel's back does not leave stale text in the boxes.

### Search

`Search::filter(query, &LayeredGraph)` is a case-insensitive substring over each node's kind
tag, label and port names, in layer-then-`NodeId` order; an empty query matches **nothing**,
because a box nobody has typed in has not found the whole graph. `advance()` cycles.

There are two sources and one matcher. Read-only mode searches the four stacked IRs;
`filter_session` searches the one IR being edited, which may already hold nodes the layered
view was built before — searching the stale view instead would be one function fewer and
wrong after the first `AddNode`. It reports the same `(layer, node)` pair, using the label the
canvas paints, so `app.rs` resolves a hit the same way whichever half found it.

Centring is the one thing left to `app.rs`, and it is the one line the packet promised:
`pan = canvas/2 − (position + node/2) × zoom`. The hit is ringed in read-only mode and
*selected* in edit mode — which also opens it in the inspector, so "find the node, change its
range" is two gestures.

### Opening

`recent::classify(path)` is the single answer to what a path is: `Run` if
`RunView::is_run_dir` says so (§10 reuses that one call, so the classification and the reader
cannot drift), `Documents` for any other directory, `Bundle` for anything that is not a
directory. A file that turns out not to be a `.esb` fails with the bundle reader's own error,
which says more than "not a directory" would. The text field, the command line, the File
menu's recent list and a dropped file (`ctx.input(|i| i.raw.dropped_files)`) all go through
`EditorApp::open`, so there is one way in and one set of errors out.

`Recent` keeps ten, most recent first, deduplicated on push, and is persisted through
`eframe::App::save` under the key **`es-editor.recent`** — a JSON array of paths in
`%APPDATA%\Electric Sheep editor\data\app.ron` on Windows,
`~/.local/share/electricsheepeditor/app.ron` on Linux. That needs `eframe`'s `persistence`
feature, which is the one feature this crate adds on top of the workspace's minimal set (§7);
it brings `ron` and `home` behind `eframe` and no new dependency entry. A store written by
another version is rebuilt through `push`, so a hand-edited file cannot smuggle in duplicates
or an eleventh entry — and an unreadable one is an empty list, because a lost recent list is
worth less than the editor starting.

### Not here

- **Multi-select, copy/paste and box-select.** All three are `Edit` *sequences*, not new
  edits: the six edits already express them, and what would have to be designed is a selection
  model and a clipboard the undo stack agrees with. Neither is what §23.4 asks stage 2 for.
- **A minimap.** §23.4's third stage-2 need is large-graph *performance*, and search is the
  half of it a person feels; a minimap is a second renderer of the same layout. When the
  canvas is too slow to pan, the answer is culling in `CanvasView`, not a small copy of it.
- **A file dialog.** `rfd` is a dependency, a native modal and a second way in; a text field,
  a recent list and drag-and-drop cover the three ways a path actually arrives.
- **Editing an Observation IR node's parameters.** No factory owns those kinds (`INV-17`), so
  they have no `NodeSchema` and no inspector — the same boundary `Edit::SetParam` reports as
  `FACTORY-001`.

---

## 13. A run watched while it runs: `--telemetry` and `--attach` (§23.1, §23.3, M7/E4)

§10 opens a run that has finished. This is the same run before it has: `es eval run
--telemetry 127.0.0.1:7777` publishes what it is doing, and `es-editor --attach
127.0.0.1:7777` — or the Telemetry tab's **Attach** field and **Connect** button — reads it.
§23.1: the editor hosts nothing; it is a client of a running process.

### The shared type is E1's

`model/live_run.rs` folds the four streams into `CellRow` and `Timeline`, which are
`run_view.rs`'s own types. That is the whole design decision: **the Run tab has one table, one
strip, one set of headers and one selection**, and it does not know which end its rows came
from. `run_table` picks the rows in one `match` at the top

| | finished (§10) | live (this section) |
|---|---|---|
| rows | `RunView::cells()` | `LiveRun::cells()` |
| headers | `RunView::columns()` | `LiveRun::columns()` |
| strip | `RunView::timeline(cell)` | `LiveRun::timeline(cell)` |
| heading | `report.passed` | `LiveRun::status()` |
| frames | `frames/<cell>/NNNNNN.bin` | the latest stream-4 image |

and everything below it is the same code. The oracle
(`live_run_folds_streams_into_run_rows`) is the equality itself: E1's committed fixture run is
replayed as the messages a live run would have sent, and the resulting rows, headers and
timelines must equal what `RunView::open` makes of the same directory. Two folds of the same
`StepEvent` bits exist — `RunView::timeline` reads `events.json`, `LiveRun::timeline` reads the
wire — because `run_view.rs` is not E4's file to change; the oracle is what stops them
drifting.

Two things a live run cannot have. There is no **acceptance verdict**: `report.json` is
written after the last suite, so the heading is `LiveRun::status()` (*"live: nominal-01
running, 2 of 3 cell(s) finished"*) and the acceptance list is empty. And there is no
**sorting**: a live table is in cell-name order because its rows are still arriving, so
clicking a header does nothing until the run is opened from disk. Selection works on both, and
until someone clicks, the selected cell *is* the running one — so an attached editor draws the
live strip with nobody touching it.

### The four streams

Named in `docs/design/telemetry-protocol.md` §9 ("Producers"), which is where the wire shape
belongs. A stream id is data, not schema: `protocol.rs` is frozen at its version.

| Stream | Payload | When |
|---|---|---|
| 1 | `Event { cell.begin \| cell.end \| suite.end }` | at each episode boundary, and once per suite |
| 2 | `Scalars[frame, tick, source, violation bits]` | every control tick that captured an observation |
| 3 | `Metrics(PerfMetrics)` | at each `cell.end` |
| 4 | `Image { rgb8 }` | every `--telemetry-image-every N` ticks (default `0`, never) |

Stream 2 is the `StepEvent` `events.json` records, as four numbers — the same record, not a
second measurement, which is why the live rows can be *equal* to the finished ones rather than
merely similar. `source` is `es_data::ActionSourceCode`'s numbering (`Policy 0, Clamped 1,
Fallback 2, Human 3`), so the dataset column and the wire agree.

### What the producer refuses to do

- **Block.** Every frame goes out through `Server::publish`, which `try_send`s into each
  client's 16-deep queue and *drops* on a full one (`telemetry-protocol.md` §6). The oracle
  `eval_telemetry_never_blocks_the_run` attaches a client that never reads a byte, floods it
  with images, and requires the run to finish with its report unchanged and the server's
  dropped count above zero.
- **Compute anything extra.** The sink is handed what the run already had: the `StepEvent` the
  plane produced, the plan's own image buffer (borrowed, not copied), the counters, the
  `CellResult`s `record_cell` returned. Without the flag nothing binds and nothing changes —
  `report.json` and `events.json` are byte-identical, which the order oracle asserts by
  comparing two runs.
- **Publish from more than one process.** `--telemetry` needs `--jobs 1`: a `--jobs N` run's
  cells happen in worker processes and only one of them could own the address. Refused by name
  rather than half-published.

`es-eval` gains no dependency on `es-telemetry` for any of this — both are layer 10 and §4.2
forbids a same-layer dependency. The evaluator calls a closure (`es_eval::runner::RunSink`, a
closure and not an eighth extension point, `INV-17`); `crates/es/src/cmd/eval.rs` — the one
crate that links both — turns a `RunEvent` into a wire `Frame`.

### Gate 9: what publishing costs the run (§28.7, §23.4)

`es eval run` on the demo documents (one suite, three episodes of 60 control ticks, 96×96
frames rendered every tick, 190 frames published), `--jobs 1`, three runs each way with one
attached subscriber draining every stream, **interleaved** (plain, telemetry, plain, …) so a
box that gets busier during the measurement moves both arms rather than one. Ubuntu, RTX 4090,
16 cores, load average 2.9–5.1 and GPU 0–17 % throughout (a neighbouring agent's run):

| | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| `es eval run`, release | 4.876 s | 4.281 s | 4.266 s | **4.281 s** |
| `es eval run --telemetry`, release | 4.477 s | 4.288 s | 4.255 s | **4.288 s** |
| `es eval run`, debug | 6.610 s | 6.538 s | 6.530 s | **6.538 s** |
| `es eval run --telemetry`, debug | 6.698 s | 6.583 s | 6.577 s | **6.583 s** |

**Observed overhead: +0.16 % release, +0.69 % debug** against §23.3's *"< 1 %"* — the gate
holds on both, and the debug figure is the conservative one because the cost is `serde_json`
encoding, which an unoptimized build pays several times over. Run 1 of each arm carries the
cold page cache; the later pairs differ by under 50 ms on a 4.3 s run, the same order as the
box's own noise. The script, the drainer and the logs are in `~/artifacts/plan-v/m7-e4/` on
the oracle server.

The number is an observation of this run shape, not a general figure: 190 frames of JSON over
a loopback socket beside a control tick that renders a frame and runs a torch forward pass.
A training loop publishing at a higher rate, or a graph view subscribing to a tensor stream,
is a different measurement — `Target / Status: unverified` for those.

### Not here

- **No producer outside `es eval run`.** `es loop collect` and `es train` publish nothing yet;
  the sink is `Evaluation::run_shard_with_sink`'s argument and nothing else calls it.
- **No reconnect.** A dropped connection is a dead `Source`: `try_recv` returns nothing
  forever and the tab keeps what it has. Attaching again is the Connect button.
- **No replay of a live run.** The Replay panel poses a `.estraj`, which a run writes at each
  episode's end; watching a live one would be a second trajectory transport, not this.
- **The image stream is off by default.** One 96×96 frame is 27 kB of pixels and about 100 kB
  as JSON; publishing one every tick is the flood the backpressure oracle uses on purpose.
