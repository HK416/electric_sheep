# Node SDK — how a third party adds a node kind

Design note for `es_ir::factory` as seen from `crates/es-editor`. Spec: §6.3 (the Task node
set), §8.3 (the Learning node set), §14.5 (LLM generation reads the same schema), §23.4 (the
editable graph), Appendix C (`INV-17`).

## 1. What the SDK is

Two traits and one struct, all of them already in `es-ir`:

| Item | Role |
|---|---|
| `TaskNodeFactory` | `(kind, params) -> TaskNode` (§6.3) |
| `LearningNodeFactory` | `(kind, params) -> LearningNode` (§8.3) |
| `NodeSchema` | what a kind takes: `inputs`, `outputs`, `Vec<ParamSchema>` |

That is the whole SDK. `INV-17` allows exactly seven single-impl extension traits and two of
them are these; **nothing new was added for the editor**. The editor's `Registries` (in
`model/palette.rs`) is a container for the two registries plus the kind names each was
registered with, not a third abstraction.

`params` is a `toml::Value` table of the node's own fields, keyed exactly as the node enum's
serde field names — no `kind` key, the factory supplies the tag.

## 2. Implementing one

```rust
use es_ir::factory::{NodeSchema, ParamSchema, ParamType, TaskNodeFactory};
use es_ir::task::{Aggregation, TaskNode};
use es_ir::{codes, Diagnostic, Port};

const GRASP_SCORE: &str = "GraspScore";

struct GraspScoreNodes;

impl TaskNodeFactory for GraspScoreNodes {
    fn kinds(&self) -> &[&'static str] {
        &[GRASP_SCORE]
    }

    fn create(&self, kind: &str, params: &toml::Value) -> Result<TaskNode, Diagnostic> {
        if kind != GRASP_SCORE {
            return Err(Diagnostic::new(codes::FACTORY_001, format!("unknown kind '{kind}'")));
        }
        Ok(TaskNode::Reward {
            name: params.get("name").and_then(toml::Value::as_str).unwrap_or("grasp").to_owned(),
            weight: params.get("weight").and_then(toml::Value::as_float).unwrap_or(1.0),
            aggregation: Aggregation::Sum,
            ty: scalar(),
        })
    }

    fn schema(&self, kind: &str) -> Option<NodeSchema> {
        (kind == GRASP_SCORE).then(|| NodeSchema {
            kind: GRASP_SCORE,
            inputs: vec![Port::new("value", scalar())],
            outputs: vec![],
            params: vec![ParamSchema {
                name: "weight".to_owned(),
                ty: ParamType::Float,
                required: false,
                default: Some(toml::Value::Float(1.0)),
            }],
        })
    }
}
```

Register it and it is in the palette, the add-node menu and the JSON export:

```rust
session.registries.register_task(Box::new(GraspScoreNodes))?;   // FACTORY-002 if the kind is taken
```

The same factory lives as a `#[cfg(test)]` example in `crates/es-editor/src/model/edit.rs`
(`GraspScoreNodes`), where two tests prove it reaches the palette, instantiates through
`Edit::AddNode`, and cannot claim a builtin kind.

## 3. The stability contract

**A factory composes builtin nodes; it does not invent an IR node type.** `TaskNode` and
`LearningNode` are closed enums, and `BUILTIN_TASK_KINDS` / `BUILTIN_LEARNING_KINDS` are frozen
behind `BUILTIN_TASK_KINDS_HASH` / `BUILTIN_LEARNING_KINDS_HASH` — two tests in
`crates/es-ir/src/factory.rs` fail the build if either list moves, because `IrNode::kind` is
hash input and a rename silently changes `task_hash` / `learning_hash` for every graph that
uses the kind (§28.7 gate 10). So a third-party kind is an **authoring shortcut**: the graph
records the builtin tag its factory produced (`GraspScore` becomes `Reward` in the saved
`.esgraph`), and a bundle authored with a plugin present still loads, hashes and compiles with
the plugin absent.

Consequences, in order of how often they surprise people:

1. A custom kind is not a new node in the IR, so it cannot change lowering, determinism or the
   hash chain — the three things a plugin must never reach.
2. A `.esgraph` never references a custom kind, so it is not a dependency of the file.
3. Changing what a custom kind expands to changes the graphs authored after the change, not
   the ones already saved.
4. Kind names are first-come: `register_*` reports `FACTORY-002` rather than shadowing.

## 4. Schema is one source, three consumers

`NodeSchema` feeds the editor's add-node menu, the parameter editor, and — via
`Palette::to_json` — the §14.5 generator path. An LLM asked to produce a Task IR reads the same
ports, parameter types and starting values the editor draws, so the generator and the editor
cannot disagree about what a node takes. `Palette::defaults(kind)` is the parameter table a
node is created with before the user touches anything, and a test builds **every** builtin kind
from its own defaults, so a schema that lies fails CI.

## 5. Deliberately not provided

- **No plugin loading.** No `dlopen`, no dynamic registry, no plugin manifest. A factory is
  Rust code compiled into the host binary. Loading native code at runtime would put an
  unreviewed third party inside the process that also runs the Safety Plane.
- **No scripting language.** §14.2 already draws this line for Python: if an extension can run
  arbitrary code, the IR's guarantees are gone. A factory produces *nodes*, and the node set is
  what the compiler and the validators know how to reason about.
- **No third factory trait.** Observation IR nodes have no factory, so `Edit::AddNode` and
  `Edit::SetParam` on an Observation graph report `FACTORY-001` (its other edits work).
  Adding `ObservationNodeFactory` would be an eighth extension point; `INV-17` says no, so the
  fix — if it is ever worth it — is a spec change, not an editor change.
- **No per-node UI.** A node draws itself from its schema. A factory that wants a custom
  widget is asking for UI types in the IR, which §4.2 rule 7 forbids.
