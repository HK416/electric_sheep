# W6-editable-graph — the editable graph editor and the Node SDK

Spec: §23.4 (the three graph-feature stages; stage 2 = editing, M3), §23.2 (the layered view
this is added beside), §14.3 (`*.esgraph` / `*.eslayout` save format — layout does not enter the
IR), §14.5 (LLM generation reads the same `NodeSchema`), §28.5 W6, §1.9 item 5 (the *editable*
graph is cut 5; read-only stays), §6.3 / §8.3 (the node sets), Appendix C (`INV-17`, §4.2
rule 7).

## context

```
crates/es-editor/src/model/edit.rs        (new: EditSession, EditIr, Edit, undo/redo, save/load)
crates/es-editor/src/model/palette.rs     (new: Registries, Palette, NodeSchema -> JSON)
crates/es-editor/src/model/mod.rs         (+ two `pub mod` lines)
crates/es-editor/src/app.rs               (+ Edit mode toggle, edit_canvas, CanvasView, save)
crates/es-editor/Cargo.toml               (+ serde, serde_json, toml)
docs/design/node-sdk.md                   (new)
docs/design/editor-shell.md               (+ section 9)
docs/packets/M3/W6-editable-graph.md
```

## spec

1. `model/edit.rs`, headless. `EditSession { graph: EditIr, layout: Layout, registries,
   diagnostics, undo, redo }` where `EditIr` is `Task(TaskIr) | Observation(ObservationIr) |
   Learning(LearningGraph)`. Six edits: `AddNode { kind, params: toml::Value, pos }`,
   `RemoveNode`, `Connect`, `Disconnect`, `SetParam`, `MoveNode`.
   - `apply` / `undo` / `redo` are deterministic. New node ids are `max + 1`.
   - **`MoveNode` writes `Layout::positions` only** and cannot change any `*_hash` (§4.2
     rule 7). `RemoveNode` also drops every incident edge and the node's layout entries.
   - After every edit: `EditIr::validate()` fills the advisory diagnostics list, and an edit
     that introduces a **new error** in `Graph::validate_declared_ports()` is reverted, returned
     as `Err(Vec<Diagnostic>)`, and not recorded in the history. Pre-existing errors do not
     block editing.
   - Undo/redo are whole-state snapshots, not per-edit inverses (`ponytail:` comment names the
     upgrade path).
   - `save() -> (esgraph, eslayout)` via `es_ir::serial::write_esgraph`; `load(graph, layout?)`
     via `parse_esgraph`. A Deployment or Evaluation file is `EditError::NotAGraph`.
   - Node creation and parameter edits go through `TaskNodeRegistry` / `LearningNodeRegistry`
     `create` + `schema`. **No per-kind code in `es-editor`.** Observation IR has no factory
     (`INV-17` allows only the two), so its `AddNode` / `SetParam` report `FACTORY-001`.

2. `model/palette.rs`. `Registries` wraps the two registries and tracks the registered kind
   names (a registry resolves a kind but cannot enumerate one; `*NodeFactory::kinds` gives the
   list at registration). `Palette::from_registries` lists every kind with its `NodeSchema`,
   grouped by a category derived from the kind (`Task / sources`, `Task / ops`,
   `Task / declarations`, `Learning / encoders|fusion|heads|action`, `* / custom`).
   `PaletteEntry::defaults` is the starting parameter table; `Palette::to_json` exports ports,
   parameter types and defaults for §14.5.

3. `docs/design/node-sdk.md`. How to implement `TaskNodeFactory` / `LearningNodeFactory` (one
   example struct), the stability contract (frozen kind lists + closed node enums ⇒ a
   third-party kind is an authoring shortcut that expands to builtin nodes, so a `.esgraph`
   never depends on a plugin), and what is deliberately absent: no plugin loading, no
   scripting, no third factory trait, no per-node UI.

4. `app.rs`, compiled only. An `Edit` toggle in the top bar starts a session on the opened
   bundle's Task IR; the Graph tab then paints the editable canvas: drag a node →
   one `MoveNode` on release, drag from an output pin to an input pin → `Connect`, right-click →
   "Add node" with the palette by category, `Delete` → `RemoveNode`, `Ctrl+Z` / `Ctrl+Y` →
   undo/redo, `Save` → both files beside the loaded path. Every branch ends in exactly one
   `apply` / `undo` / `redo`.

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

- `save` → `load` round-trips a session with added nodes, an edge and a changed parameter: the
  IR, the layout and the hash all compare equal.
- Undo and redo of three edits restore the exact IR, layout and hash; a new edit clears redo.
- `MoveNode` never changes `task_hash`, and the `.esgraph` half of `save()` is byte-identical
  before and after a move; a sidecar-less load has the same hash as a sidecar-ful one.
- A type-mismatched `Connect` is refused with `TYPE-003` on the consumer node, the edge is not
  kept, the hash is unchanged and the history did not grow. An unknown port is `GRAPH-010`; an
  unknown kind is `FACTORY-001`; an unknown parameter key is `FACTORY-003`.
- The palette lists every kind in `BUILTIN_TASK_KINDS` and `BUILTIN_LEARNING_KINDS` and nothing
  else, every one of them builds from its own `defaults()`, and the JSON export carries each
  kind's ports and parameters.
- A `#[cfg(test)]` third-party factory registers a custom kind, appears in the palette under
  `Task / custom`, instantiates through `AddNode`, and cannot claim a builtin kind
  (`FACTORY-002`).
- `es-editor` stays inside the §1.5 context budget and `cargo xtask layering` is clean.

## forbidden

- Any change outside `crates/es-editor/**` and the three doc files above. `es-ir` in
  particular: `NodeSchema`, the factory traits and the frozen kind lists are P29's, and
  `Graph` / `Layout` / `write_esgraph` are P28's.
- A new trait anywhere (`INV-17`): no `ObservationNodeFactory`, no `Editable`, no
  `NodeRenderer`.
- Layout in an IR type, or an IR hash that can see a position (§4.2 rule 7).
- Hand-written per-kind construction, parameter mapping or port lists in `es-editor`.
- `egui-snarl` (re-decided in `docs/design/editor-shell.md` §9: it would replace ~190 lines of
  painting and bring three duplicate concepts).
- `HashMap` anywhere (§3.4 determinism; `BTreeMap` only), and any decision in `app.rs` that CI
  cannot run.
