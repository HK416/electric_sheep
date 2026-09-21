# M8 S4d — the reward cone reaches a body: `GetBodyPose`, `Norm{L2}` and an IEEE `sqrt` in `es-env`'s scalar plan

Spec: §6.3 (`GetBodyPose` and `Norm` are Task IR nodes already), §6.4 (execution semantics),
§6.6 (`DET-010`, and the sentence added 2026-09-21: `sqrt` is an IEEE basic operation, not a
transcendental — `Norm{L2}` lowers to `Expr::Sqrt`), §5.4 (unit algebra), §28.11 wave 3 (S4d).
Design notes: `docs/design/batch-domains.md` section 6 "Reward and termination" (the cone as
built: `GetJointState`, `GetSensor`, `GetTime`, `GetContact` leaves + `Arith`, `Compare`, `Clamp`,
`Normalize`, `Logic`), `docs/design/rl-continuation.md` section 5 (the reach task this makes
executable; the 26-dim observation). Found by S4b's report (2026-09-21): the reward
`−‖cube_pos − gripper_pos‖` is spellable in Task IR and not executable by `es-env`. Korean sibling
in the same commit (the orchestrator's).

## the question

Every cone leaf binds one scalar (`GetJointState` → the joint's first `qpos`, `GetSensor` →
`sensor[start]`), `Source` is `Qpos | Qvel | Sensor | Time`, and `es_ir_types::Expr` has no root.
So the distance between two bodies — the reward of every reach, place and follow task — cannot
run here. **Can the cone carry a body's world position as three lanes through `Arith{Sub}` into
`Norm{L2}` with an IEEE `sqrt`, bitwise across the CPU backends, with the committed demo task's
lowering byte-identical to today's?**

## spec

* **`Expr::Sqrt(Box<Expr>)`** in `crates/es-ir-types/src/expr.rs`: `eval` is `f64::sqrt`; a
  negative or non-finite input is `None` like every other non-finite (never a `NaN`). Every
  place the enum is matched (canonical bytes, hashing, display, the parser if it exposes
  functions) gains the arm; the doc comment says why this is not a `DET-010` transcendental
  (IEEE 754 requires a correctly rounded square root — one hardware instruction, bit-identical
  everywhere; §6.6). No other new variant.
* **`Source::Xpos { row, axis }`** in `crates/es-env/src/plan.rs`: reads
  `state.xpos[env * nbody * 3 + row * 3 + axis]`, `row` from `ModelInfo::body` (a body the model
  does not index → `EnvError::Unsupported` naming the body). `xpos` is already on `StateView`
  (`crates/es-physics-core/src/backend.rs`) and every backend fills it.
* **Lanes.** `lower` returns one `Expr` per lane: a scalar leaf is one lane;
  `GetBodyPose { relative_to: World }` is three lanes (position only — orientation, and any
  other `Frame`, is `Unsupported` naming what was asked); `Arith` between equal lane counts is
  lane-wise, a one-lane operand broadcasts; `Norm { kind: L2 }` collapses `n` lanes to
  `Sqrt(l₀² + l₁² + …)` summed **in lane order** (DET-020) — other `NormKind`s are `Unsupported`
  by name; `Compare`, `Clamp`, `Normalize`, `Logic` and every sink require exactly one lane, and
  a vector reaching them is a named error, not a silent first lane. Unit rules stay the ones
  the file applies today (the demo task's header records them: `Mul`'s rhs dimensionless, etc.).
* **Nothing committed moves.** The demo `task.toml`'s reward and termination cones lower to
  byte-identical `Expr`s — pin them (the lowered plan's canonical/debug form, or the fixture
  backend's reward sequence, before you touch anything) and assert after.
* **The reach documents** (`rl-continuation.md` section 5, revised 2026-09-21), under
  `tests/fixtures/rl/`: `task-reach.toml` (scene `tests/fixtures/mjcf/so101_pick_place.xml`,
  the demo's per-episode cube randomization reused; `ObservationSpec` channels `joint_pos[6]`,
  `joint_vel[6]`, `cube_pose[7]`, `gripper_pose[7]` bound the way the demo binds
  `sim_cube_pose`; reward `−‖cube_pos − gripper_pos‖` per step + `1` on success; `Terminate`
  success at distance `< 0.03` m, timeout 200 control steps); `observation-reach.toml`
  (`StateInput` ×4 → `Concat` → `Normalize{Range}` identity-shaped, 26 wide, `task_ref` = the
  reach task's hash); `deployment-reach.toml` (the demo deployment with `action.horizon = 1`,
  `execute_chunk = 1`, `rate.inference = rate.control`, an envelope wide enough that a random
  policy is clamped, not latched — INV-12: widen, never disable); `evaluation-reach.toml` (16
  held-out seeds, `nominal` + the demo suites that apply to a state policy, `success_rate ≥ 0.8`
  acceptance). Each document's header comment says what it is, as the demo's do.

## context

```
crates/es-ir-types/src/expr.rs
crates/es-ir-types/tests/**
crates/es-env/src/plan.rs
crates/es-env/src/env.rs
crates/es-env/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/task-reach.toml
tests/fixtures/rl/observation-reach.toml
tests/fixtures/rl/deployment-reach.toml
tests/fixtures/rl/evaluation-reach.toml
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/packets/M8/S4d-reward-cone.md
docs/packets/M8/S4d-reward-cone.ko.md
```

`expr.rs` (the variant and its arms), `plan.rs` (the source, the lanes, the two node arms),
`env.rs` **only** if the plan needs the body table passed through, `es-env/tests` (oracles 2–3),
`cli.rs` (oracle 4), the four documents, the note (its section 6 lists the cone), this packet.

## oracle

1. `cargo test -p es-ir-types expr_sqrt_is_ieee_and_refuses_negative` — `Sqrt(Const(2.0))`
   evaluates to `2f64.sqrt()` exactly; `Sqrt(Const(-1.0))` is `None`; the canonical bytes of an
   expression without `Sqrt` are unchanged (pin one).
2. `cargo test -p es-env body_norm_cone_lowers_and_the_demo_task_is_unmoved` — on the fixture
   backend with a two-body toy model whose `xpos` the test sets: reward `−Norm{L2}(GetBodyPose(a)
   − GetBodyPose(b))` equals `-((dx*dx + dy*dy) + dz*dz).sqrt()` **bitwise** (that exact
   association); a vector into `Compare` is refused by name; the committed demo task's lowered
   cones equal the pinned form.
3. `cargo test -p es-env reach_task_executes -- --ignored` (`ES_PYTHON`, MuJoCo): `task-reach.toml`
   on the scene — after `reset`, `Env::step`'s reward equals `−‖xpos[cube] − xpos[gripper]‖`
   computed in the test from the backend's own `StateView` (bitwise); with the state set so the
   gripper body is within 0.03 m of the cube (`supports_state_get_set`), the next step
   terminates with success.
4. `cargo test -p es --test cli reach_documents_validate` — the four documents parse, validate,
   cross-validate (`XIR_*`), `es task compile` accepts the task, and the observation's port is
   26 wide.
5. `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p es-ir-types
   -p es-env -p es`, `cargo xtask check-scope docs/packets/M8/S4d-reward-cone.md`.

## acceptance

Oracles 1–5 (3 on the server). `batch-domains.md` section 6 lists the new leaf, the lane rule
and the `Sqrt` with the §6.6 argument; Korean sibling updated.

## forbidden

Any transcendental through libm or a new `es-math` function (`sqrt` is the only new operation,
and it is `f64::sqrt`); a second `Expr` variant; changing what any existing cone lowers to;
touching `es-eval`, `es-py`, `es-safety`; `docs/ARCHITECTURE*.md` (the sentence is there);
`docs/design/rl-continuation.md` (S4b writes its section 7 concurrently); `tests/golden/**`;
INV-17.
