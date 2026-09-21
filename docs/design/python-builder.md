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

## The rollout binding (M8 S4a, spec 13.4)

`es-py` gained a second, unrelated half: `es_native.Rollout`, so `train_ppo.py` can step our
`Env` and the Safety Plane instead of a simulator of its own (`rl-continuation.md` rule 2). The
layering is the builder's, one layer shorter — there is no fluent Python surface, because a
trainer is not an authoring tool:

```
python/es/train_ppo.py     the trainer (S4b) — torch, GAE, the optimizer
        |  lists of floats over the pyo3 boundary
crates/es-py/src/pybind.rs `Rollout` pyclass — a fixed (n_joints, horizon) dispatch table
        |  plain Rust calls, no Python involved
crates/es-py/src/rollout.rs `Rollout<NJ, H>` — Env + one SafetyPlane and one CpuPlan per env
```

| method | what it is |
|---|---|
| `Rollout(task, observation, deployment, scene_xml, seed, n_envs)` | the four documents as **text**; builds `Env<MuJoCoCpuBackend>` with `BatchDomains::single_env_at`'s control period and the batch set to `n_envs` |
| `reset(envs=None)` | `Env::reset`, then `begin_episode` on those planes and `CpuPlan::reset` on those rings |
| `observe() -> {port: [n_envs * dim]}` | `es_eval::runner::capture` + `CpuPlan::run`, per env, row-major |
| `act([n_envs * nu]) -> (executed, events, rewards, dones)` | per env `observe_state → heartbeat → validate`, then one `Env::step` with the plane's output; `events` is `EventSet::bits()` |
| `model() -> dict` | `nq`, `nv`, `nu`, `n_envs`, actuator and joint names in index order, `ctrlrange` |
| `tick() -> int` | control ticks — the plane's clock, not `Env::tick`'s simulation tick |
| `metrics() -> dict` | the nine spec 12.4 fields plus `chunk_underrun_rate`, `None` where nothing measured them |

Three decisions worth writing down:

- **Lists, not numpy.** The boundary carries Python lists both ways and `es-py` grows no array
  dependency. `train_ppo.py` already imports torch and converts on its side; adding `numpy` to
  a layer-11 crate would put a second array ABI in the runtime so that one caller could skip
  one `torch.tensor(...)` call.
- **No second observation capture.** `es_eval::runner`'s `capture` / `input_sources` /
  `joint_state` became `pub` (no logic change) and the binding calls them. A capture written
  again here is how a trainer and an evaluation start reading two different observations off
  one state. The one thing added is a per-env `StateView` — those helpers read env 0, because
  they were written for the single-env evaluation path — so a batch of `n` reaches them
  unchanged. An image input still needs a renderer and `es-py` links none, so `capture`
  refuses it by name; `crates/es-py/tests/fixtures/observation-state.toml` is the demo's
  observation with the camera branch removed, generated from it, never typed in.
- **A fixed `(n_joints, horizon)` table, not a type-erased plane.** `SafetyPlane<NJ, H>` is
  const-generic and a document only reveals its numbers at run time; the pyclass enumerates the
  pairs this repo uses, exactly as `es eval run`'s `dispatch_nj_h!` does. A trait object would
  be an eighth extension point (`INV-17`). Horizon 1 is the trainer's sense of the word — one
  row per control tick under a fresh `seq`, no chunk buffer, no declared latency
  (`rl-continuation.md` section 3) — while `H` stays the deployment's own, because
  `SafetyPlane::from_ir` refuses an envelope whose width is not its.

The oracle is one trace, driven twice: `cargo test -p es-py rollout_matches_es_eval_loop --
--ignored` runs 100 control steps of a scripted control through `Rollout` and through a
hand-rolled `Env` + `CpuPlan` + `SafetyPlane` loop written the way `es_eval::runner::run_episode`
runs it, and pins both to `tests/golden/rollout/so101_100steps.json`;
`python -m es.selfcheck --env` drives the pyclass over the same 100 steps and compares against
the same golden, so the Rust struct and the Python class cannot drift apart silently.

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
