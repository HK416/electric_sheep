# Python authoring builder (M2 W6, spec 14.2)

## Shape

Three layers, each thinner than the one below it:

```
python/es/*.py            fluent spec 14.2 surface (Task, Observation, Learning, Deployment,
                           Expr operator overloads, es.math) — pure Python, no node-kind logic
        |  JSON text over the pyo3 boundary
crates/es-py/src/pybind.rs "es_native" pyo3 module — Pythonic method names (`.reward`,
                           `.terminate`, `Learning.act`, `.save`), each forwarding to the core
        |  plain Rust calls, no Python involved
crates/es-py/src/builder.rs generic core — TaskBuilder / ObservationBuilder / LearningBuilder /
                           DeploymentBuilder: `add(kind, params)`, `connect`, `declare_*`,
                           `build`, `save_toml`, over the existing `es-ir` factories/graph
```

The bottom layer is the only one this packet's oracle (`cargo test -p es-py`) exercises without
Python; it has zero per-node-kind code — `TaskBuilder::add`/`LearningBuilder::add` route
through `TaskNodeRegistry`/`LearningNodeRegistry` (the two `INV-17` factory extension points),
and `ObservationBuilder::add` re-tags JSON the same way `factory::deserialize_tagged` re-tags
TOML (`ObservationNode` has no factory — it is not one of the 7 extension points — so this is
the same serde trick applied by hand once, not per kind).

## Why Deployment has no `add`/`connect`

`DeploymentIr` (`es-ir/src/deployment.rs`) is a plain struct — safety envelope, watchdogs,
fallback, rate — not a graph (`serial.rs`: "No `NodeId` map, so no mirror"). `DeploymentBuilder`
is direct setters (`envelope`, `watchdog`, `fallback`, `set_robot`, ...) instead of forcing an
`add(kind, params)` shape onto something that was never node-shaped.

## Why params are JSON, not `toml::Value`, at the pyo3 boundary

`es-ir`'s node factories require `&toml::Value` (`TaskNodeFactory::create`), so
`builder::json_to_toml` converts once at the point of use. The *public* surface of both
`builder.rs` and `pybind.rs` speaks `serde_json::Value` / JSON text instead: Python's `json`
module is the natural bridge (`json.dumps` on a dict), and reusing it means `crates/es-py`
never needs a hand-written `PyObject` walker or an extra dependency beyond `pyo3` itself (see
`docs/api-notes/pyo3.md`).

One asymmetry worth remembering: JSON's `null` has no TOML equivalent. `json_to_toml` drops a
`null`-valued table key instead of erroring — exactly how `Option::None` (e.g.
`PortType::image`) already serializes to TOML — and only rejects a `null` that isn't inside a
table (the top level, or an array element), which is never something a real node's fields
produce.

## Where the type inference lives

Task/Observation node ports are fixed by node *kind* (`TaskNode::outputs`,
`ObservationNode`'s `Io`); Learning node ports are the node's own declared `inputs: Vec<TensorPort>`
field, echoed back verbatim by `LearningNode::input_ports`. Either way, something has to know
each node's resulting `PortType` *before* wiring the next node, the same way a graph editor's
"what can I plug in here" has to. `python/es/builder.py`'s `Expr` class and a couple of small
formulas (`_feature_ty`, `_chunk_ty`) do this by mirroring the relevant `es-ir` private helpers
(`f32v`/`boolish`/`flag` in `task.rs`, `feature`/`chunk`/`policy_ty` in `learning.rs`) — each
one commented with the Rust function it has to stay in sync with. This is ordinary front-end
work, not a workaround: an LLM generator or a graph editor would need the identical inference.

## Known gaps (M2 W6 scope, not M0/M1 regressions)

- `python/es/examples/pick_place.py` reproduces the spec 14.2 snippet's node *shape* faithfully;
  a few call sites adapt to what the schema actually has a field for today (documented inline —
  e.g. `Reward`'s sign moves onto `weight` since this builder has no constant-folding path for
  Task IR arithmetic, and `VisionEncoder.pretrained` is a plain `bool` so `pretrained="imagenet"`
  is read as truthy). None of the adaptations touch `es-py`'s generic core.
- Cross-IR checks (`es-ir::cross`, spec 11.1) are the compiler's job, not this builder's; nothing
  here replays them client-side.
- Evaluation IR has no builder — out of this packet's scope (spec 14.2's own snippet does not
  cover it either).
