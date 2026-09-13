# Control Graph (IR-C) — design

Spec refs: spec 6.2 (IR-D / IR-C split), spec 6.3 (`Parallel` does not exist; `Wait` / `Repeat` /
`Condition` are IR-C or `Compare`+`Select`), spec 6.4 (execution semantics), spec 6.5
(expressions), spec 6.6 (determinism rules), spec 23.4 (Control Graph *editing* is M4),
spec 28.6 (M4 scope). Packet: `docs/packets/M4/W2-control-graph.md`.

Type C artifact: this note is the human-reviewed design; the code follows it, not the other way
round.

## 1. What IR-C is for

Spec 6.2 is explicit that the standard vision-manipulation tasks — pick&place, reach, push —
are expressible in IR-D alone. IR-C exists for **multi-stage assembly**: `pick` then `move` then
`place`, where each stage has its own reward terms, its own completion predicate, and its own
deadline, and where the episode reward is the composition of the stages that actually ran.

IR-C is therefore deliberately *small*. It adds sequencing, one data-dependent branch, bounded
repetition, and a named slice of the IR-D graph. It adds no data flow: every value it reads is
an IR-D port, evaluated by the evaluator that already exists.

## 2. Schema

```
ControlGraph {
    root:  ControlNodeId,
    nodes: BTreeMap<ControlNodeId, ControlNode>,
}

ControlNode =
    | Sequence { children: Vec<ControlNodeId> }
    | Branch   { condition: Expr, then_: ControlNodeId, else_: ControlNodeId }
    | Repeat   { body: ControlNodeId, until: RepeatUntil }
    | SubTask  { task: SubTaskRef, timeout_ticks: u32 }

RepeatUntil = Count(u32) | Until(Expr)

SubTaskRef {
    name:           String,            // unique in the graph; the stage's identity
    rewards:        BTreeSet<String>,  // TaskNode::Reward names this stage scores
    observation:    BTreeSet<String>,  // subset of TaskIr::observation_spec channels
    success:        Option<Expr>,      // stage completion predicate; None = one step
    reset_on_entry: bool,
    weight:         f64,               // this stage's factor in the episode reward
}
```

`ControlNodeId` is `es_ir::graph::NodeId` — the same id type IR-D uses, in its own namespace.

`TaskIr` gains one field:

```
TaskIr { .., control: Option<ControlGraph> }
```

`None` is the IR-D-only task and is the default (`#[serde(default)]`), so every task authored
before IR-C parses unchanged.

### 2.1 Why a SubTask names things instead of listing node ids

A `SubTask` is a *slice* of the parent IR-D `TaskGraph`, but it must not carry `NodeId`s into
it. `canon_task` (spec 11.2, Appendix B.7 property 1) relabels the IR-D graph; a control node
holding IR-D ids would go stale under that relabelling and `task_hash` would stop being
relabel-invariant. Names are stable under relabelling, so a slice is named by:

* its `Reward` node names (`TaskNode::Reward { name, .. }`), and
* its `ObservationSpec` channel names.

`Terminate` nodes are *not* sliced: a stage's completion is `SubTaskRef::success`, and the
task-level `Terminate` nodes stay what they are — whole-episode predicates.

## 3. Execution semantics (spec 6.4)

The executor is a deterministic state machine, one per env, advanced **once per env step**,
after physics and before reward. There is no wall clock anywhere in it (spec 6.6 `DET-002`) and
no scheduling decision that depends on data other than the two declared ones: `Branch::condition`
and `RepeatUntil::Until`.

Per-env state is a stack of `(node, index)` frames — the path from the root down to the active
`SubTask` — plus a tick counter for the active stage:

```
descend(n):
    Sequence { children }  -> push (n, 0); descend(children[0])   (empty -> advance)
    Branch   { .. }        -> push (n, arm); descend(arm)          arm chosen ONCE, here
    Repeat   { body, .. }  -> push (n, 0); descend(body)
    SubTask  { .. }        -> push (n, 0); stage ticks := 0; stop — this is the active stage

advance():                                       // the active SubTask finished
    pop the SubTask frame, then, top frame first:
    Sequence -> index+1 < len   ? descend(children[index+1]) : pop, advance again
    Repeat   -> more iterations ? descend(body) with index+1  : pop, advance again
    Branch   -> pop, advance again                            // one arm only
    stack empty -> the episode is done
```

**Entry is the first step of the episode, not the reset.** A `Branch` reads IR-D ports, and at
reset there are none bound yet, so `reset_env` only clears the state and the first `step` walks
down from the root with the ports the reward cones just filled. The walk is bounded at 256
descent steps (`MAX_DESCENT`) so a generated tree with a degenerate empty body ends the env
instead of spinning.

`Repeat { until: Count(k) }` runs the body exactly `k` times (`k > 0` is validated).
`Repeat { until: Until(e) }` re-enters the body while `e` evaluates false, and is additionally
bounded by the episode budget — the executor never blocks on a predicate that can never hold.

A `SubTask` ends when, in this order:

1. `success` evaluates to a non-zero value (`Expr::eval`; `None` — a missing port, a division by
   zero — counts as "did not fire", exactly as `episode::evaluate` already treats it), or
2. `stage ticks >= timeout_ticks`, which ends the **episode**, not just the stage, and sets the
   stage's failure flag.

`success: None` completes the stage on its first step; that is the "run one step of this slice"
degenerate case and is what `Sequence` over trivial stages means.

### 3.1 Stage ports

Before evaluating any IR-C `Expr`, the executor writes three ports into the same port map the
reward and termination cones read, so a condition can branch on where the machine is:

```
stage.index    the active SubTask's index in its parent Sequence, as f64
stage.ticks    steps spent in the active stage, as f64
stage.done     1.0 on the step the stage completed, else 0.0
```

Everything else an `Expr` can name is an IR-D binding (`qpos[i]`, `qvel[i]`, `sensor[i]`,
`time`, `time.episode`) produced by `ScalarPlan` — IR-C introduces no new evaluator and no new
`Expr` variant.

**The three stage ports are per env.** `Env` keeps one port map for the whole batch and refills
it per env in `bind_ports`; the IR-D bindings are overwritten there, but the stage ports are
not in that set, so `bind_ports` calls `control::clear_stage_ports` first. Without that, the
two readers that run *before* this env's own `write_ports` — the task-level `Terminate` cone in
`episode::evaluate`, and the `Branch` evaluated once on graph entry — would see the previous
env's stage, and the batch's answer would depend on the order envs happen to be stepped in
(`docs/reviews/M4.md` B-1, packet `P-M4-R1`). The consequence for authors: on the step a graph
is entered, no stage has run yet for that env, so `stage.*` is **unbound** and an entry `Branch`
naming one evaluates to `None` — falsy, the `else_` arm. Branch on IR-D state there, not on
`stage.*`.

### 3.2 Reward composition

With `control = None`, the episode reward is unchanged: `sum(weight * term)` over every
`Reward` node (spec 6.4).

With `control = Some(..)`, the sum runs over **the active stage's** reward terms only, scaled by
the stage weight:

```
reward(env) = stage.weight * sum over t in ScalarPlan.rewards, t.name in stage.rewards
                                 of t.weight * t.expr.eval(ports)
```

A stage with an empty `rewards` set scores nothing. A term that cannot be evaluated contributes
nothing rather than poisoning the sum with a `NaN`, as before. Summing stage by stage over the
episode is what makes the episode return the composition of the stages that actually ran: a
branch not taken contributes exactly zero steps and therefore exactly zero reward.

### 3.3 Termination

`done` comes from the executor when a control graph is present:

| executor state | `Termination` |
|---|---|
| stack empty (every stage finished) | `Success` |
| a `SubTask` hit `timeout_ticks` | `Failure` |
| still running | fall through to the episode budget (`max_episode_steps` -> `Timeout`) |

The task-level `Terminate` nodes still run and still win when they fire: a whole-episode failure
predicate is not something a stage may suppress. The `EpisodeRecorder` needs no new column — the
timeout failure is recorded as `Termination::Failure` on the closing step.

### 3.4 Randomization and reset

Reset and domain randomization stay IR-D nodes (`ResetState`, `Randomization`, spec 6.3) drawn
at episode reset. A stage does **not** redraw them unless it says so: `reset_on_entry = false`
is the default and the common case, because a multi-stage assembly is one continuous physical
episode. `reset_on_entry = true` is carried in the schema and reported by the executor as part
of the transition; wiring it into `Env` needs a "redraw this env's reset distributions without
closing its episode" path that `Env::reset` does not have today, and is deferred (§6).

## 4. Determinism (spec 6.6)

* Every collection is a `BTreeMap` / `BTreeSet` / `Vec`; nothing iterates a `HashMap`
  (`DET-020`).
* Stage time is an integer step count. Seconds are never accumulated (spec 18.1).
* No RNG (`DET-001`) and no wall clock (`DET-002`) enter the machine at all — there is no node
  for either.
* The only data-dependent scheduling is `Branch::condition` and `RepeatUntil::Until`, both
  evaluated by `Expr::eval`, which is already specified as deterministic (exact IEEE
  comparisons, non-finite intermediates become `None`).
* `Branch` evaluates its condition **once, at entry**, and the chosen arm is stored in the
  frame. Re-evaluating every step would make the path depend on the physics trajectory mid-arm
  and would not be reproducible across a partial reset.

Two runs of the same task, seed and backend produce bit-identical stage paths, rewards and
terminations. That is the property the `es-env` test asserts.

## 5. Hashing

IR-C is part of `task_hash`. It reuses the IR-D machinery rather than growing a second one:
`ControlNode` implements `IrNode`, and `ControlGraph::as_graph()` projects the tree onto a
`Graph<ControlNode>` whose edges are the parent -> child links:

```
Sequence  out ports "child0" .. "childN-1"   -- indexed, so child ORDER is semantic
Branch    out ports "then", "else"
Repeat    out port  "body"
SubTask   no out ports
every node has one in port "parent"
```

`canonical_hash` then applies unchanged: node ids never enter the digest, the port names carry
the structure, and a `Sequence` whose children are swapped hashes differently because the port
names differ. `params_canonical` writes the node's own parameters and **never** a child id — the
structure is the edges' job.

`TaskIr::task_hash` mixes in `control.is_some()` and, when present, the control digest. Adding
the field is a Task IR schema change, so `task::SCHEMA_VERSION` goes `1 -> 2`; see the migration
note in `docs/design/ir-types.md`.

`BUILTIN_CONTROL_KINDS` (`Sequence`, `Branch`, `Repeat`, `SubTask`) is frozen the same way as
`BUILTIN_TASK_KINDS` / `BUILTIN_LEARNING_KINDS` (spec 28.7 gate 10), with its own frozen digest
`BUILTIN_CONTROL_KINDS_HASH`. It is a *separate* list: `BUILTIN_TASK_KINDS` is the set of kinds
`TaskNodeFactory` can build into a `TaskNode`, and a control node is not a `TaskNode`, so adding
these four to that list would make the factory claim kinds it cannot construct.

## 6. Validation

| code | rule |
|---|---|
| `CTRL-001` | `root`, a child, a `then_`/`else_` or a `body` names a node that is not in `nodes` |
| `CTRL-002` | a node is unreachable from `root` |
| `CTRL-003` | a node has more than one parent — IR-C is a tree, not a DAG |
| `CTRL-004` | the control graph contains a cycle |
| `CTRL-005` | `Repeat` count is 0, or `SubTask` `timeout_ticks` is 0 |
| `CTRL-006` | a `SubTask` names a `Reward` node or an observation channel the Task IR does not declare, or two stages share a name |
| `TYPE-030` | a `Branch` condition or a `RepeatUntil::Until` expression is not boolean |

"Boolean" is structural, not inferred: `Expr` has no type, but only `Expr::Compare` produces
exactly `0.0` / `1.0`, so a condition whose root is not a `Compare` is rejected rather than
silently thresholded at zero. `Expr::Clamp` over a `Compare` is *not* accepted — the narrow rule
is the one a user can predict.

## 7. Deferred

* **Editor support.** Control Graph editing is M4 per spec 23.4 and belongs to a later packet.
  `es-editor`'s layered graph view is untouched here and does not know the IR-C kinds.
* **`Wait`.** Spec 6.3 lists it as replaceable; a `SubTask` with `success = time.episode > t`
  already expresses it, so there is no node for it.
* **Nested `Repeat` depth cap.** `MAX_DESCENT` bounds one step's walk, but there is no *declared*
  maximum nesting depth in the schema and no `CTRL-` code for exceeding one; add both when a
  generated graph needs them, not before.
* **`reset_on_entry` wiring** in `Env` (§3.4).
* **Lowering IR-C to the compiler** (`es-compile`): the executor here is the CPU reference, the
  same way `ScalarPlan` is for reward cones.
