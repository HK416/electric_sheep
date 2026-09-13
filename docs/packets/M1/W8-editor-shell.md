# W8 — editor shell (`es-editor`: tabs, read-only layered graph, telemetry, before/after)

Spec: §23.1 (the editor is a client of a running process), §23.2 (layered graph view), §23.3
(watching during training: pre/post preprocessing images, event log, per-node statistics),
§23.4 (three graph-feature stages — read-only is M1), §14.3 (`.eslayout` sidecar, never in the
IR), §12.4 (the metric set, never a single `step/s`), §1.9 cut 5 (the *editable* visual graph
is cuttable; read-only stays), §4.2 rule 4 (nothing depends on `es-editor`) and rule 7 (no
layout in IR), §28.3 W8. Design note: `docs/design/editor-shell.md`.

## context

```
crates/es-editor/Cargo.toml
crates/es-editor/src/lib.rs
crates/es-editor/src/main.rs
crates/es-editor/src/app.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/model/graph_view.rs
crates/es-editor/src/model/telemetry_view.rs
crates/es-editor/src/model/image_view.rs
crates/es-editor/tests/common/mod.rs
crates/es-editor/tests/graph_view.rs
Cargo.toml                       (workspace deps only: es-editor, eframe, egui)
docs/design/editor-shell.md
docs/packets/M1/W8-editor-shell.md
```

## spec

- **New crate `es-editor`, layer 12.** May depend on anything; nothing may depend on it
  (§4.2 rule 4, enforced by `cargo xtask layering`, which already carries the row).
- **Split: headless view-model (tested) + thin egui layer (compiled).** CI has no display, so
  every decision lives in `src/model/**` and is judged by `cargo test`; `src/app.rs` only turns
  positions into rectangles and is judged by `cargo build`.
- **`model::graph_view`.** `LayeredGraph::from_bundle(task, observation, learning, deployment)`
  → four `LayerView`s (Task, Observation, Learning, Deployment) of `NodeView { id, kind, label,
  ports, layout }` plus the IR's own edges; `cross_edges` mirroring the joins `es_ir::cross`
  checks (Task `ObservationSpec` → Observation source, Observation output → Learning contract
  input, Learning action → deployed action contract — the *rules* stay in `es_ir::cross`, this
  only draws the same pairings); `diagnostics` = each IR's `validate()` + `cross::check`.
  Deployment IR is a record, not a graph (§9.2), so its band is four synthetic nodes:
  `ActionContract → SafetyEnvelope → Watchdogs → Fallback`. `auto_layout` is deterministic
  Sugiyama-lite (longest-path rank → `x`, canonical `NodeId` order within a rank → `y`, one
  band per layer); `apply_layout(&es_ir::serial::Layout)` overrides it from an `.eslayout`
  sidecar. **Layout is read, never written into an IR** (§4.2 rule 7, §14.3).
- **`model::telemetry_view`.** `TelemetryModel` consumes `es_telemetry::protocol::Message` from
  a `Box<dyn FnMut() -> Option<Message>>` source, so `es_telemetry::transport` (another packet,
  in flight) plugs in later without touching this crate. Keeps capped per-`(stream, component)`
  scalar histories, the latest `PerfMetrics`, and an event log; `metric_rows()` is the §12.4
  set in fixed order (latency as p50 + p95), an unmeasured metric staying `None`.
- **`model::image_view`.** `BeforeAfter::run(obs, input)` compiles the Observation IR with
  `es_compile::CpuPlan` — the same CPU reference the lowering is judged against (§11.3), never
  a second preprocessing implementation — runs it on one frame and pairs the raw input with
  each image output as `Rgb8Image`. `BeforeAfter::sample(obs)` synthesizes a deterministic
  gradient for a bundle that carries no frame.
- **`app` + `main`.** `eframe` app, tabs `Graph | Telemetry | Images | Diagnostics`, a
  text-field bundle path (a `.esb` opened through `es_compile::PolicyBundle::open`, or a
  directory of the per-IR TOML files under the names the bundle format fixes). The Graph tab
  paints node rectangles per layer band with cubic-bezier edges from the view-model positions,
  pans and zooms, and is **read-only** — there is no drag, and no code path writes a position
  back into an IR. `es-editor [bundle.esb]` opens one at startup.
- **No `egui-snarl`.** §23.4 names it, but a read-only view needs only the `egui::Painter`;
  snarl exists for interactive wiring. Upgrade path recorded in the design note: `LayeredGraph`
  is unchanged by adopting it, only the painting in `app.rs` is replaced.

Constraints: English only, `BTreeMap` only, no new trait (INV-17), ≤ ~1500 source lines,
minimal egui features to keep compile time down, root `Cargo.toml` touched only to add the
three workspace dependencies.

## oracle

```
cargo fmt -p es-editor --check
cargo clippy -p es-editor --all-targets -- -D warnings
cargo test -p es-editor
cargo build -p es-editor
cargo xtask layering
cargo xtask context-budget
```

## acceptance

- `cargo build -p es-editor` succeeds on a CI machine with no display, and no test opens a
  window.
- From `es-ir`'s own cross-IR fixture (copied into `tests/common/mod.rs`, as the `es` CLI tests
  already do): four layers with the node counts of their IRs and four synthetic Deployment
  nodes; five cross-IR edges (`rgb_front` and `joint_state` Task→Observation and
  Observation→Learning, `actions` Learning→Deployment), every endpoint naming a node that
  exists in its layer.
- `auto_layout` is deterministic (two runs over the same bundle are equal), places every node,
  ranks by dataflow (`ImageInput < Resize < Normalize`, both sources sharing rank 0), never
  gives two nodes of a layer the same position, and keeps each layer in its own band.
- `apply_layout` overrides exactly the nodes the sidecar names and leaves the rest.
- A bundle with `action_dim` disagreeing with the deployed contract surfaces `XIR-020` in
  `diagnostics`; the untouched fixture surfaces none.
- Canned telemetry frames fill the tables: per-component scalar series, the event log with its
  fields, `execution_hash` from the `HelloAck`, the ten §12.4 rows with unmeasured ones `None`;
  history is capped oldest-first and `pump` honours its budget.
- An 8×6 gradient through `ImageInput → Dequantize → Resize(4×3)` yields one `ImagePair`:
  `before` 8×6 and byte-identical to the input frame, `after` 4×3.

## forbidden

- `crates/es-telemetry/src/transport.rs`, `crates/es-core/src/ring.rs`,
  `crates/es/src/cmd/backend.rs`, and doc translations — other packets' scope. This crate must
  not depend on the transport yet; the message source stays a closure.
- Writing layout into any IR type, or adding a position/UI field to `es-ir` (§4.2 rule 7).
- Re-implementing the cross-IR rules of `es_ir::cross` or the preprocessing kernels of
  `es_compile` — mirror the pairings, run the plan.
- Graph editing of any kind (§23.4 stage 2, M3), `egui-snarl`, a native file-dialog crate, any
  other new external dependency, and any change to the root `Cargo.toml` beyond the three
  workspace dependency entries.
- Committing.
