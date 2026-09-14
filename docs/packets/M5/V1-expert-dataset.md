# M5 V1 — scripted expert and the demonstration dataset

Design note: `docs/design/visible-learning.md` section 5; read section 2.8 first — there is no inverse
kinematics anywhere in the repo, and SO-101's chain is why the expert is closed-form rather than
damped-least-squares. Depends on V0 (the scene and the four documents) and V0b (frames).

## context

```
crates/es-env/src/expert.rs
crates/es-env/src/lib.rs
crates/es-env/tests/expert.rs
crates/es-data/src/collect.rs
crates/es-data/tests/loop_learning.rs
crates/es-data/python/lerobot_read_ref.py
crates/es-data/tests/lerobot_oracle.rs
crates/es/src/cmd/loop.rs
crates/es/tests/cli.rs
docs/api-notes/lerobot-dataset.md
docs/api-notes/lerobot-dataset.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V1-expert-dataset.md
docs/packets/M5/V1-expert-dataset.ko.md
```

Notes: `expert.rs` is new and carries its unit tests; `lib.rs` gets the module and re-exports only.
`collect.rs` gains the per-step frame write and drops the "no mp4 was written" warning **only** when a
renderer is present. `loop.rs` gains `--expert`, which makes the Torch runtime unnecessary.
`docs/api-notes/lerobot-dataset.md` records what the new oracle measured — the note currently pins
nothing and says so at `:3-8`.

## spec

- §1.4: the expert is judged by a success rate over a pinned seed set, and the dataset by the real
  `lerobot` package reading it back. Both are commands.
- §2.4, §2.5: the expert is Rust and runs without Python; the dataset oracle is Python and is a test tool,
  never on the runtime path.
- §3.4: `es_math::approx` for every trigonometric call, no global RNG, no wall clock, no `f64` time
  accumulation. The expert's only entropy is the Task IR's reset draw.
- §6.3: per-episode variation comes from V0's `Randomization` node, so `(task_hash, seed, episode)` fixes
  the demonstration.
- §8.6, §9.3: what is recorded as `action` is the **post-Safety-Plane** control, as `collect.rs:534`
  already does. A demonstration the plane clamped is recorded clamped; the expert is not exempt (INV-12).
- §17.2: an out-of-reach waypoint is a named failure, never a clamped approximation.
- §19.2: `dataset_hash = H(content, schema, split)`; adding an image feature moves `schema_hash`, which is
  the documented rule (`crates/es-data/src/identity.rs:142-150`).
- §25.1: the dataset root is written, never read back into a trust decision; the Python oracle reads it
  out of process.
- §1.5: `es-env` is at 2,060 and `es-data` at 3,459 code lines; this packet is budgeted under ~700.

## oracle

```
cargo fmt --check
cargo clippy -p es-env -p es-data -p es --all-targets -- -D warnings
cargo test -p es-env --lib expert
cargo test -p es-env --test expert
cargo test -p es-data --test loop_learning
cargo test -p es --test cli loop_collect
cargo xtask context-budget
cargo xtask check-spec-refs
```

Reference — the two oracles that need an interpreter:

```
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es-env --features render --test expert -- --ignored --nocapture
ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot/bin/python \
  cargo test -p es-data --test lerobot_oracle -- --nocapture
```

1. **Expert success rate.** `expert_solves_the_pinned_seeds` runs `es loop collect --expert` over V0's
   scene for a pinned seed set and asserts the fraction of episodes ending in `Termination::Success` is at
   or above a threshold recorded in the test. No `mujoco` -> `SKIP expert_success: <why>`; ran ->
   `RAN expert_success`. The threshold is a property of the expert, not a tuning knob: lowering it to make
   a change pass is the same as editing a golden.
2. **LeRobot reads what we wrote.** `crates/es-data/python/lerobot_read_ref.py <root>` opens the dataset
   with `lerobot.datasets.lerobot_dataset.LeRobotDataset` and prints, as JSON, the episode count, the
   frame count per episode, every feature name with its dtype and shape, and the first and last
   `observation.state` row. `tests/lerobot_oracle.rs` compares that against the same values read by
   `es-data`'s own Rust reader. **Measured 2026-09-14 on the oracle server: `~/venvs/es-lerobot` has
   `lerobot 0.6.1` but not the `[dataset]` extra — `import lerobot.datasets` raises
   `ImportError: 'datasets' is required but not installed`, and `pyarrow` is absent.** Until a human
   installs `lerobot[dataset]` this oracle prints `SKIP lerobot_oracle: <why>` on every machine. Record
   in `docs/api-notes/lerobot-dataset.md` what it reports on the first machine where it runs — including
   whether 0.6.1 accepts `codebase_version: "v2.1"` and whether the uncompressed parquet
   (`crates/es-data/Cargo.toml:22-24`) is readable. That finding, not this packet, decides the v2.1 / v3
   question (design note open question 2).

`crates/es-env/tests/expert.rs` and the `expert` unit tests:

- `ik_round_trips_through_forward_kinematics` — for a grid of reachable targets, `so101_ik` then MuJoCo's
  own forward kinematics puts the gripper body within a pinned tolerance of the target. This is the
  oracle for the closed form: MuJoCo's FK, not our own algebra restated.
- `ik_refuses_rather_than_clamps` — a target beyond the reach, and one whose solution leaves
  `shoulder_lift`'s range, both return `None`; no returned solution ever has an angle outside its range.
- `ik_uses_derived_link_lengths` — `Links` is built from the parsed `SceneDesc`, and a test asserts no
  length literal appears in `expert.rs` (V0's `link_lengths_are_derived_not_transcribed`, enforced here
  as a source scan).
- `the_state_machine_advances_only_on_its_predicate` — with a frozen state, the stage does not change; on
  the exit predicate it advances exactly one stage.
- `an_unreachable_waypoint_fails_the_episode` — the episode ends `Termination::Failure`, and the collector
  records it as such rather than dropping it.
- `the_same_seed_gives_the_same_demonstration` — two runs of `(seed, episode)` produce byte-identical
  `ctrl` rows.
- `clamped_expert_actions_are_recorded_clamped` — with a deliberately tight envelope, the recorded
  `action` equals the post-plane value and `action_source` is `Clamped`, not `Human`.
- `frames_are_written_once_per_control_step` — with a renderer, the frame count equals the episode's
  frame count and `info.json`'s video feature no longer carries the "no mp4 was written" warning
  (`crates/es-data/src/collect.rs:505-508`).
- `without_a_renderer_collect_behaves_as_before` — the existing warning, the empty video map and the
  dangling `VideoRef` are all unchanged, so `loop_learning.rs`'s current assertions still hold.

`crates/es/tests/cli.rs`: `loop_collect_expert_needs_no_torch` — `es loop collect --expert ... ` with an
interpreter that has `mujoco` but not `torch` exits 0 (today it exits 3, because `loop.rs:194-202` gates
on both). Without `mujoco` it still exits 3 with `SKIPPED`.

## acceptance

```rust
// crates/es-env/src/expert.rs
/// Link lengths and offsets, derived from a parsed scene — never literals (design note section 4.2).
pub struct Links { /* base height, upper/lower arm, wrist offset, gripper offset */ }
impl Links {
    pub fn from_scene(scene: &es_assets::scene::SceneDesc) -> Result<Self, EnvError>;
}

/// Closed-form IK for SO-101's base yaw + planar 3R. `None` when unreachable or out of range;
/// never an approximation (spec 17.2).
pub fn so101_ik(links: &Links, target: es_math::Vec3, approach_pitch: f64) -> Option<[f64; 4]>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage { Approach, Descend, Close, Lift, Transport, Release, Done }

pub struct ExpertCfg {
    pub cube_body: es_core::StableId,
    pub gripper_body: es_core::StableId,
    pub bin_center: es_math::Vec3,
    pub hover_height: f64,
    pub approach_pitch: f64,
    pub grip_open: f64,
    pub grip_closed: f64,
    pub close_ticks: u32,
    pub pos_tol: f64,
}

pub struct ScriptedExpert { /* cfg, links, stage, stage_tick */ }
impl ScriptedExpert {
    pub fn new(scene: &SceneDesc, cfg: ExpertCfg) -> Result<Self, EnvError>;
    pub fn reset(&mut self);
    pub fn stage(&self) -> Stage;
    /// One control step of demonstration, or `None` when the waypoint is unreachable
    /// (the caller ends the episode as a failed demonstration).
    pub fn action(&mut self, model: &ModelInfo, state: &StateView<'_>, env: u32) -> Option<Vec<f64>>;
}
```

- `ScriptedExpert` is reached through the intervener closure `Collector::run` already takes; `es loop
  collect --expert <name>` replaces `|_, _, _| None` (`crates/es/src/cmd/loop.rs:152-153`) with it and
  skips `TorchRuntime` entirely. `--policy` is still required, because the bundle carries the Task,
  Observation and Deployment IR the collector needs; its weights are never loaded under `--expert`.
- Recorded demonstrations carry `action_source = Human` and `intervention = 1`, through the existing
  classification (`crates/es-data/src/collect.rs:461-471`) — no new column, no new `ActionSource` variant.
- With a renderer, one frame per control step is written under the dataset root and `info.json`'s video
  feature stops being a dangling reference.
- No new trait, no `HashMap`, no new external dependency, no float time, ≤ ~700 source lines.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` — no new `TaskNode`, no `ActionSource` variant, no schema change.
- A Jacobian, a damped-least-squares solver, an iterative IK, or `mj_jac` on the Python side — design note
  section 2.8 settles this.
- `crates/es-safety` — a demonstration passes through the same plane as a policy (INV-12); no bypass, no
  "expert mode" in the envelope.
- Changing `crates/es-data/src/lerobot/meta.rs`'s `codebase_version` or adding `arrow` / a compression
  codec to `es-data` — both are decided by the oracle's finding, in a later packet if at all.
- `crates/es-policy`, `crates/es/src/cmd/eval.rs`, `crates/es-eval` — V2 and V3.
- A hand-written Python or Rust re-implementation of LeRobot's reader in place of the real package.
- Lowering the expert's success threshold to make a change pass.
