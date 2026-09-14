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
