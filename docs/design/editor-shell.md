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
`glow` + `default_fonts`, no accesskit, no wgpu, no persistence — which keeps the incremental
rebuild of this crate at ~2.4 s. A Linux CI that wants a *runnable* binary (not just a
compiled one) adds eframe's `x11` / `wayland` features; building headless does not need them.

## 8. Not here

- 3D scene view, state streaming, pause/step/rewind, hot-patching (§23.3) — they need a
  running process on the other end of the transport.
- Node-level live values and the reward → exposure back-trace of §23.2. The structure they
  hang on (`LayeredGraph` + `TelemetryModel` in one crate) is in place; wiring a stream to a
  node is the next packet.
- The §28.7 gate-9 number (telemetry + graph view cost < 1% of training throughput):
  `Target / Status: unverified` — it cannot be measured from a viewer with no producer.
