# W2 — IR-C: the Task IR control graph

Spec: spec 6.2 (IR-D / IR-C split; `Sequence`, `Branch`, `SubTask`, `Repeat`; multi-stage
assembly is the use case), spec 6.3 (`Parallel` does not exist), spec 6.4 (execution
semantics), spec 6.6 (determinism), spec 23.4 (Control Graph *editing* is a later M4 packet),
spec 28.6 (M4 scope), spec 1.2 / spec 1.4 (packets, oracle first).

Design note: `docs/design/control-graph.md` (type C, reviewed before this packet's code).

## context

```
crates/es-ir/src/control.rs            NEW — the IR-C schema, validation, hashing
crates/es-ir/src/lib.rs                `pub mod control;` + re-exports
crates/es-ir/src/task.rs               `TaskIr.control`, SCHEMA_VERSION 1 -> 2, hash, validate
crates/es-ir/src/factory.rs            BUILTIN_CONTROL_KINDS + its frozen hash
crates/es-ir/src/serial.rs             TOML mirror carries `control`
crates/es-ir/src/norm.rs               control tree in the B.7 generators / edits
crates/es-ir-types/src/codes.rs        CTRL-001..006, TYPE-030 (`es_ir::codes` re-exports it)
crates/es-ir/tests/control_graph.rs    validation, serde/TOML round trip, relabel invariance
crates/es-env/src/control.rs           NEW — ControlExecutor (the CPU reference executor)
crates/es-env/src/lib.rs               `pub mod control;` + re-exports
crates/es-env/src/env.rs               additive: control drives reward/done when present
docs/design/control-graph.md           NEW
docs/packets/M4/W2-control-graph.md    this file
```

Mechanical `control: None,` additions at `TaskIr` literal sites outside these crates are part of
the packet only to keep `cargo check --workspace` green; no other change is made there.

## spec

1. `ControlGraph { root, nodes: BTreeMap<ControlNodeId, ControlNode> }`, with `ControlNode` one
   of `Sequence { children }`, `Branch { condition, then_, else_ }`, `Repeat { body, until }`,
   `SubTask { task: SubTaskRef, timeout_ticks }`. Shapes are fixed in
   `docs/design/control-graph.md` §2.
2. `TaskIr.control: Option<ControlGraph>`, `#[serde(default)]`. `None` behaves exactly as
   before. `task::SCHEMA_VERSION` becomes 2 and the migration note goes into
   `docs/design/ir-types.md`.
3. A `SubTask` names its IR-D slice by `Reward` node name and observation channel name, never by
   `NodeId` — otherwise `canon_task` relabelling would move `task_hash` (Appendix B.7 property
   1).
4. Validation: single existing root, tree shape (no sharing, no cycle, nothing unreachable),
   `Repeat` count > 0, `timeout_ticks` > 0, `SubTask` slice names resolve, stage names unique,
   `Branch` / `Until` conditions are boolean (`Expr::Compare` at the root). Codes `CTRL-001`..
   `CTRL-006` and `TYPE-030`.
5. Hashing reuses `canonical_hash` via an `IrNode` impl on `ControlNode` and
   `ControlGraph::as_graph()` (tree edges, indexed `child{i}` port names so sibling order is
   semantic). `params_canonical` never writes a child id.
6. `BUILTIN_CONTROL_KINDS` + `BUILTIN_CONTROL_KINDS_HASH`, frozen the same way as the task and
   learning kind lists (spec 28.7 gate 10). A separate list: a control node is not a `TaskNode`,
   so `TaskNodeFactory` must not claim these kinds.
7. `es_env::control::ControlExecutor`: `new(&TaskIr)`, per-env `Vec<StageState>`,
   `step(env, &mut ports) -> StageOutcome { reward_weight, done, failed, transitioned }`.
   `Sequence` advances on sub-task completion, `Branch` evaluates its condition once at entry,
   `Repeat` re-enters until the count / `until` predicate, a `SubTask` timeout ends the episode
   with the failure flag. `Env::step` uses it only when `task.control` is `Some`.

## oracle

```
cargo fmt --check
cargo clippy -p es-ir -p es-env -p es-editor --all-targets --features es-ir/testing -- -D warnings
cargo test -p es-ir -p es-env -p es-editor --features es-ir/testing
cargo xtask context-budget
cargo xtask check-spec-refs
cargo check --workspace --all-targets
```

## acceptance

`es-ir`

- A good three-stage tree validates clean; a shared child reports `CTRL-003`; a cycle reports
  `CTRL-004`; a `Branch` condition that is not a `Compare` reports `TYPE-030`; a zero `Repeat`
  count and a zero timeout report `CTRL-005`; an unknown reward name reports `CTRL-006`.
- Serde JSON and the `.esgraph` TOML envelope both round-trip a Task IR with a control graph.
- `task_hash` is unchanged by relabelling the IR-D graph and by relabelling the control tree,
  and *does* change when a stage weight, a timeout, or the sibling order of a `Sequence`
  changes. The five Appendix B.7 properties run over `arbitrary_task_ir`, which now emits an
  optional control tree.
- `BUILTIN_CONTROL_KINDS_HASH` is asserted against the live list.

`es-env`

- A three-stage `pick -> move -> place` task on `FakeBackend` visits the stages in order and
  scores only the active stage's reward term.
- A `Branch` on a port value takes the arm the value selects.
- `Repeat { until: Count(3) }` executes its body three times.
- A `SubTask` whose `timeout_ticks` elapses ends the episode with `Termination::Failure`.
- Two runs of the same task and seed are bitwise identical in stage path, reward and
  termination.

## forbidden

- Any new extension-point trait (`INV-17`): `ControlNode` gets the internal `IrNode` impl and
  nothing else. No `ControlNodeFactory`.
- `NodeId`s of the IR-D graph inside `SubTaskRef`, and child ids inside `params_canonical`.
- Editor work: `crates/es-editor/src/model/graph_view.rs` stays unaware of the IR-C kinds
  (spec 23.4 puts Control Graph editing in a later packet).
- `HashMap` anywhere on this path (spec 6.6 `DET-020`), wall-clock reads, RNG, or any
  scheduling decision that is data-dependent beyond `Branch` and `RepeatUntil::Until`.
- `crates/es-gpu/**`, `crates/es-usd/**`, `crates/es-physics-core/**`, `crates/es-script/**`,
  `crates/es-data/**`, `crates/es-eval/**`, `crates/es-physics-backend/**`,
  `crates/es-policy/**`, `crates/es/**` — other packets own them; only the one-line
  `control: None,` literal fix is allowed.
- Editing `tests/golden/**` (spec 1.4, CI read-only).
