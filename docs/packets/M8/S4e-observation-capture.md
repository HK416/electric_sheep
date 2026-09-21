# M8 S4e — the observation capture reads joint velocities and a body's pose, and the reach run lands

Spec: §7.4 (`ObservationSpec` declares channels; `ObsSource` says where each comes from), §6.3
(`GetJointState` has a `JointQuantity`), §10.1 (capture resolves against the documents, never
guessed), §3.1 (quaternions xyzw), §28.11 wave 3, packet M7/R5's rule (a new field is absent =
default = today's canonical form). Found by S4d (2026-09-21, `tests/fixtures/rl/task-reach.toml`
header): `ObsSource` has no velocity quantity and `es_eval::runner::input_sources` has no `qvel`
or body-pose reading, so the reach observation's `joint_vel[6]` and `gripper_pose[7]` cannot be
captured — by `es eval run` or by `es_native.Rollout` (S4a), which shares that code. Design
notes: `docs/design/evaluation-execution.md` 2.3 "Observation capture" (+ `.ko.md`),
`docs/design/rl-continuation.md` section 7 (the reach run's rows). Korean sibling of this packet:
the orchestrator's.

## the question

**Can a channel say "these joints' velocities" and "this body's pose", can the one capture path
serve both from the backend's own state — `qvel` rows, `xpos ‖ xquat` (xyzw) — bit-for-bit, with
every committed document's hash and every existing channel's bytes unmoved, and does the reach
task then train to its target through `es train`'s `[rl]` route?**

## spec

* **`ObsSource::JointState { body, dof, quantity: JointQuantity }`** in `crates/es-ir/src/task.rs`,
  `#[serde(default, skip_serializing_if = …)]` with `Position` the default and the canonical
  bytes written only for `Velocity` (pin the committed `task_hash eb6efefa…` and `task-pt`'s
  before touching anything; `committed_task_hash_is_unmoved_by_sensor_render` is the precedent).
  `XIR-002` (one channel per source id) keys by `(id, quantity)` so a position channel and a
  velocity channel may name the same joint block. The `NodeSchema` entry for the channel gains
  the parameter.
* **Capture** in `crates/es-eval/src/runner.rs`: `Capture::Qvel(range)` for a `Velocity` channel
  whose id is a joint of the loaded model (`ModelInfo::dof`), `Capture::JointsVel(dof)` for the
  leading-`dof` form (mirror of `Joints`); `Capture::BodyPose(row)` for `ObsSource::BodyPose(b)`
  when `b` is **not** a free-joint body of the model (a free-joint body keeps today's `Qpos`
  arm, which is what serves the demo's `sim_cube_pose` — bytes unchanged), reading
  `xpos[row*3..][..3] ‖ xquat[row*4..][..4]` from `StateView`, 7 values, quaternion in the order
  `StateView` documents (xyzw). The model-free path (recorded frames, `model == None`) refuses
  both new kinds by name — a dataset row does not carry them and nothing is guessed.
* **`Rollout`** (`crates/es-py/src/rollout.rs`) needs no change if it calls the same `capture`;
  if `env_view` / the per-env `StateView` slice does not carry `xpos`/`xquat`, fix the slice.
* **The reach documents regenerate** (`cargo test -p es --test cli -- --ignored
  regenerate_reach_documents`, S4d's generator): `joint_vel` = `JointState { body = base, dof = 6,
  quantity = Velocity }`, `gripper_pose` = `BodyPose(gripper)`; the header's paragraph on the two
  channels is rewritten to say what is now true; `observation-reach.toml` / `evaluation-reach.toml`
  follow the moved `task_hash`.
* **The reach run** (S4b's deferred oracle 4): `tests/fixtures/rl/training-reach.toml` (`[rl]`
  over a bundle packed from the four reach documents + a `learning-reach.toml` you add beside
  them: `StateEncoder{Mlp hidden = [64, 64]}` → `PolicyHead{Regression, horizon 1}` →
  `Normalizer{Inverse}` over the actuator `ctrlrange`s; `envs = 16`, `horizon = 64`); on the
  server, the budget that reaches `success_rate ≥ 0.8` on `evaluation-reach.toml` through
  `es eval run`, seed 0 — or the number reached if it does not, with the curve. Rows into
  `rl-continuation.md` section 7 with the nine metrics as `Rollout.metrics()` reports them and
  `Target / Status: unverified` for the rest.

## context

```
crates/es-ir/src/task.rs
crates/es-ir/src/schema.rs
crates/es-ir/src/cross.rs
crates/es-ir/src/**
crates/es-ir/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es-py/src/rollout.rs
crates/es-py/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/**
tests/golden/train/**
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M8/S4e-observation-capture.md
docs/packets/M8/S4e-observation-capture.ko.md
```

`task.rs` (the field, canonical, `XIR-002`'s key wherever it lives in `es-ir`), the schema,
`runner.rs` (three `Capture` arms and their reads), `rollout.rs` **only** for the state slice,
tests, the regenerated documents plus `learning-reach.toml` and `training-reach.toml` (and a
`--dry-run` plan golden, an addition), the two notes, this packet.

## oracle

1. `cargo test -p es-ir committed_task_hashes_are_unmoved_by_joint_quantity` — the committed
   `task.toml` and `task-pt.toml` hashes equal the pinned literals; `quantity = "Position"`
   spelled out hashes the same as absent; `Velocity` moves it; two channels on one joint block
   with different quantities pass `XIR-002`, the same quantity twice is still refused.
2. `cargo test -p es-eval capture_reads_qvel_and_body_pose` — a synthetic `StateView` with
   distinct values in `qpos`, `qvel`, `xpos`, `xquat`: the three new arms land the right bytes in
   the right ports, the quaternion order is xyzw, the model-free path refuses both by name, and
   the demo documents' resolution is unchanged (pin the `Capture` map of the committed demo).
3. `cargo test -p es-py rollout_observes_the_reach_documents -- --ignored` (`ES_PYTHON`, MuJoCo,
   server): `Rollout` over the four reach documents; after `reset` the 26-wide port equals, slice
   by slice and **bitwise**, the backend's own `qpos[0..6]`, `qvel[0..6]`, cube `qpos[6..13]`,
   gripper `xpos ‖ xquat` (xyzw) read from `StateView` in the test.
4. `cargo test -p es --test cli reach_documents_validate` still green on the regenerated
   documents; `train_rl_dry_run_plan` gains `training-reach.toml`'s plan golden (an addition).
5. Server: the reach run — `es train --recipe tests/fixtures/rl/training-reach.toml`, then
   `es eval run --config tests/fixtures/rl/evaluation-reach.toml --policy <last checkpoint>`;
   `success_rate`, the budget, wall-clock, the metrics, the curve, all in section 7 with server,
   date and path (`~/artifacts/plan-s/s4e/`).
6. `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p es-ir -p
   es-eval -p es-py -p es`, `cargo xtask verify-goldens` (additions only), `cargo xtask
   check-scope docs/packets/M8/S4e-observation-capture.md`.

## acceptance

Oracles 1–6. `evaluation-execution.md` 2.3 lists the two new captures and the free-joint rule;
`rl-continuation.md` section 7 has the reach rows. If the target is not reached, the rows say
what was, and the packet still ships — the measurement is the deliverable, not the number.

## forbidden

Changing what any existing channel captures or any committed document's hash; guessing a
channel the documents do not declare (INV-14's spirit: named refusal, never a default); a second
capture implementation (S4a's rule); `python/es/train_ppo.py` and `crates/es/src/cmd/train.rs`
(S4b's; a trainer bug found here is reported, not fixed here); `es-safety`; `docs/ARCHITECTURE*.md`;
`tests/golden/**` modifications; INV-17.
