# M7 E3 — the parameter inspector, node search, and opening files like a person does

Spec: §23.4 stage 2 (editing: parameters, search, large-graph needs), §14.3 (`.eslayout`
sidecar), §4.2 rule 7 (no UI types in `es-ir`), §28.10 rule 3. Design note to extend:
`docs/design/editor-shell.md` (+ `.ko.md`) — section 9's "Not here" list shrinks, new section 12.
Predecessors: M4 editor stage 2 (`EditSession`, `Edit::SetParam`, `NodeSchema`, `Palette`), E1
(the directory/bundle open branch).

## the question

`Edit::SetParam` and `NodeSchema` exist and are tested, and no widget uses them; a node's
parameters can only be changed by editing TOML. There is no way to find a node in a graph of a
hundred, and the only way to open anything is to type a path. **Can a person edit a parameter,
find a node, and open a file — with every decision in a tested model and zero new dependencies?**

## spec

Headless models in `crates/es-editor/src/model/`:

* `inspector.rs` — `Inspector::for_node(session: &EditSession, node: NodeId) -> Inspector` reads
  the node's `NodeSchema` from the session's registries and its current parameter table (the
  same re-serialisation `SetParam` uses). `Inspector::fields() -> &[Field]` where `Field { name,
  ty: ParamType, required, text: String, error: Option<String> }`. Every `ParamType` maps to
  exactly one `Widget` (`Bool → checkbox`, `Int → drag int`, `Float → drag float`, `String →
  text`, `Enum(vs) → combo`, `Shape → text parsed as "[a, b, …]"`, `PortType → text parsed as
  inline TOML`); `Field::parse(&self) -> Result<toml::Value, String>` is the only place text
  becomes a value, and `Inspector::edit(name, text) -> Option<Edit>` returns the `SetParam` to
  apply or records the parse error on the field. The `Enum` and `Bool` widgets cannot produce an
  error; the others can, and an erroneous field never emits an edit.
* `search.rs` — `Search::filter(query, &LayeredGraph) -> Vec<(layer, NodeId)>`: case-insensitive
  substring over the node's kind tag, label and port names, results in layer-then-id order; empty
  query returns nothing; `Search::next(current) -> NodeId` cycles.
* `recent.rs` — `Recent { paths: Vec<PathBuf> }` capped at 10, most recent first, dedup on push,
  `to_json`/`from_json` (serde is already a dependency) — the app persists it through
  `eframe::App::save` / `Storage`, which is built in.
* Opening: `Opened::classify(path) -> Kind::{Bundle, Documents, Run}` decides from what is on
  disk (E1 introduced the run branch — reuse; if E1 has not landed, add the classification and
  E1 will use it) and is tested for a `.esb`, a directory of five `.toml`, and a `report.json`
  directory. Dropped files (`egui`'s `raw.dropped_files`) go through the same path as the text
  field.

`app.rs`: a right-hand inspector panel in edit mode for the selected node (one widget per
field, an error line under a bad field, Apply on Enter / focus loss → `EditSession::apply`), a
search box in the graph toolbar (Enter cycles, the hit is centred and highlighted — centring is
`pan = f(node position)`, a one-liner), a File menu with the recent list, and drop-to-open.
Nothing else is decided there.

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

```
crates/es-editor/src/model/inspector.rs
crates/es-editor/src/model/search.rs
crates/es-editor/src/model/recent.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/lib.rs
crates/es-editor/src/app.rs
crates/es-editor/src/main.rs
crates/es-editor/Cargo.toml
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E3-inspector-search-open.md
docs/packets/M7/E3-inspector-search-open.ko.md
```

`crates/es-editor/src/model/{inspector.rs,search.rs,recent.rs}` (new), `model/mod.rs`, `lib.rs`,
`app.rs`, `main.rs` (persisting `Recent` needs `eframe::NativeOptions`'s default persistence —
say in the note what is stored where), `crates/es-editor/Cargo.toml` **only** if an already
present workspace crate is needed (no external addition), `docs/design/editor-shell*.md`
section 12 and the section 9 "Not here" list, `docs/packets/M7/E3-inspector-search-open*.md`.

## oracle

1. `cargo test -p es-editor every_param_type_has_one_widget` — a table test over every
   `ParamType` variant (`Enum` with two values) asserts one `Widget` each and that the match is
   exhaustive (adding a `ParamType` breaks compilation, not behaviour).
2. `cargo test -p es-editor set_param_round_trips_through_the_inspector` — for every node kind
   the Task and Learning registries expose (`Palette::from_registries`), build a node with
   `Palette::defaults`, open the inspector, re-enter each field's own text, and assert the
   emitted `SetParam` value equals the current value (no spurious edit), then change one field
   and assert `EditSession::apply` accepts it and `task_hash` moves for a hashed parameter.
3. `cargo test -p es-editor a_bad_field_never_emits_an_edit` — `"abc"` in an `Int` field, `"[1,"`
   in a `Shape` field: `edit()` is `None`, `error` is set, the session is unchanged.
4. `cargo test -p es-editor search_is_case_insensitive_and_ordered` and
   `recent_is_capped_deduplicated_and_round_trips`.
5. `cargo test -p es-editor classify_tells_bundle_documents_and_run_apart`.
6. `cargo build -p es-editor`; `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/E3-inspector-search-open.md`.

## acceptance

Oracles 1–6. The orchestrator edits a `Normalize` range in the demo bundle through the panel,
sees the diagnostic list update and `task_hash` change in the status line, finds a node by
typing part of its name, and reopens the bundle from the recent list after a restart. Design
note section 12 records the models and what stays out (multi-select, copy/paste, minimap).

## forbidden

`crates/es-ir/**` (`NodeSchema`/`ParamType` are read, never changed; a UI type in `es-ir` is rule
7's violation); a new external dependency (`rfd`, `egui_extras`, …); any parse or decision in
`app.rs`; the six existing `Edit`s' semantics; `docs/ARCHITECTURE*.md`; goldens.
