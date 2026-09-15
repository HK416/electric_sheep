# M5 V1c — demonstrations record the executed action

Design note: `docs/design/visible-learning.md` sections 7.5–7.9 and **open question 12**. Read them
first. Depends on V1 (the collector, the scripted expert), V1b (the v3.0 export), V2/V2b (the
lowering, the training script, the baked set) and V3 (the measurement this packet re-runs). It
fixes a defect V2b measured but was forbidden from touching.

## the defect

Two code paths drive the same Safety Plane and disagree about what the envelope is measured
*against*.

`es_eval::runner` — and `es_ros2::hil`, and `es_runtime_embedded` — call
`SafetyPlane::observe_state(q, qd)` with the measured joint state before every `validate`, which
re-seeds `last_safe`, `prev_safe`, `vel` and `prev_vel`. The velocity stage then reads
`(cmd − qpos) / dt` and the envelope becomes a bound on the **following error**: at 50 Hz with
`velocity_max = 3.0`, a command may lead the joint it commands by `0.0600` rad.

`es_data::collect::Collector::run` never called it. There, `last_safe` is the *previous command*,
so the envelope bounds command-to-command motion, and `ScriptedExpert` paces itself to exactly
that (`step_max = 0.054`) — while its anti-windup deliberately lets the command lead the measured
joint by `4 × step_max = 0.216` rad, because that is what a position servo with an inference
latency needs.

So the demonstrations' `action` column leads its own `qpos` by a median **0.2413** rad, four times
what the envelope allows at inference. ACT imitates that faithfully (V2b measured the chunk's first
action within 0.02–0.16 rad of the recorded action), every step is velocity/acceleration/rate
clamped, `envelope_violation_rate` is `1.0000`, the violation-rate watchdog latches
`hold_position` on about a tenth of the steps, and **not one step in 86,400 was
`ActionSource::Policy`**. The demonstrations and the envelope were never checked against each
other — which is open question 12, stated as a root cause rather than as three options.

The same carried plane state is the second defect: `Collector::run` resets the env, the chunk
buffers and the e-stop latch between episodes but never the plane's hold target, velocity or rate
history, so episode 1 begins with the plane believing the arm is still where episode 0's last
command left it. That is why `es loop collect --episodes N` only solved episode 0 (design note
section 7.6, finding 5) — not the scripted expert, whose `reset` was already correct.

**The principle this packet applies: what is executed is what is recorded.** `es loop collect`
tells the plane where the robot is, exactly as every other actuator path does, and records the
`SafeAction` the plane handed toward the actuator. The raw command is kept beside it in
`action_commanded`. Demonstrations are then consistent with the envelope by construction, and one
reset per episode is one reset of everything.

## context

```
crates/es-env/src/domains.rs
crates/es-env/src/expert.rs
crates/es-data/src/collect.rs
crates/es-data/src/lib.rs
crates/es-data/tests/loop_learning.rs
crates/es-data/tests/lerobot_v3.rs
crates/es/src/cmd/loop.rs
docs/api-notes/lerobot-dataset.md
docs/api-notes/lerobot-dataset.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V1c-executed-action-record.md
docs/packets/M5/V1c-executed-action-record.ko.md
```

Notes: the pre-plane command is captured where it is handed over —
`DomainRunner::emit_actions`, `es-env`, layer 9 — because that is the only place the value exists
outside the plane, and `es-safety` is forbidden here. `es-data` reads it through one accessor and
writes one column. `crates/es-data/src/lerobot/{columns,meta,v3}.rs` are **not edited**: the v2.1
writer and the v3.0 exporter are already generic over `Info::features`, so declaring the feature is
the whole change, which is the evidence that V1b's exporter was written correctly.

## spec

- **§9.3, §9.4, INV-12.** Nothing is bypassed and nothing is disabled. The plane gains one input
  it already has an API for (`observe_state`, and its doc comment says a caller that knows the real
  pose calls it), and `validate` keeps its signature (INV-13). The only value that travels toward
  the actuator is still the `SafeAction`; `action_commanded` is provenance on disk and is never
  read back into a control path.
- **§13.2.** A collected frame carries its provenance. `action_source` already said *how* the
  actuator value was produced; `action_commanded` says *what was asked for*, which is what makes a
  `Clamped` frame legible without re-running the plane.
- **§19.1, §19.2.** The dataset schema is part of the training run's input identity. Adding a
  column moves `dataset_schema_hash` and therefore `content`; the api-note records the move and
  V1b's v3.0 export carries both columns, so no consumer sees a dataset that has one and not the
  other.
- **§17.2.** An episode that the expert cannot finish is still a failed demonstration, written and
  not dropped. Nothing here changes that.
- **§3.5, §5.3.** The collect run stays a pure function of the seed: `observe_state` reads the
  backend's own state, adds no RNG and no float accumulation.
- **INV-17.** No new trait. One field, one accessor, one const, one column.
- **§1.5.** `es-env` and `es-data` are both well inside the cap; this packet is budgeted under
  ~150 source lines across the two.

## oracle

```
cargo fmt --check
cargo clippy -p es-env -p es-data -p es --all-targets -- -D warnings
cargo clippy -p es --features render --all-targets -- -D warnings
cargo test -p es-env
cargo test -p es-data
cargo test -p es --test cli
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
cargo xtask ci
```

- `crates/es-env/src/domains.rs`: **`emit_actions_writes_the_planes_answer_and_records_the_command`**.
  Runs the action phase against a deliberately tightened envelope over 16 envs and 8 control steps
  and asserts, per env and per joint, that `ctrl` is `SafetyPlane::last_safe_action()` **bit for
  bit** (`f64::to_bits`, no tolerance — a tolerance here would mean one of the two had grown a
  conversion), that every recorded value is inside the envelope it passed through, and that the
  command the plane was handed differed from it on at least one step. Without the last assertion
  the test would pass on an envelope that clamps nothing.
- `crates/es-data/tests/loop_learning.rs`:
  **`the_dataset_records_the_executed_action_beside_the_raw_command`** — a collect run with a
  tightened position envelope and an intervener that asks for `0.9` rad on every joint of every
  tick. `action` and `action_commanded` must differ on at least one element; every `action` value
  must be inside the envelope; at least one `action_commanded` value must be outside it. A writer
  that clamped the provenance column, or recorded the command as the action, fails it.
- `crates/es-data/tests/loop_learning.rs`: **`a_second_episode_repeats_the_first_exactly`** — the
  episode-reset regression. The fixture's uniform reset draw is pinned to a constant and the
  velocity envelope tightened, so two episodes of one run are physically identical; `action`,
  `action_commanded`, `observation.state` and `action_source` must then be equal row for row.
  **Measured to fail before this packet** (episode 1 continued from episode 0's last command,
  `0.065` where episode 0 started at `0.005`).
- `crates/es-env/src/expert.rs`: `reset_restarts_the_ramp_from_the_measured_joints` — the expert
  side of the same boundary, which was already correct and is now pinned: after `reset` the first
  chunk is one `step_max` from the measured joints, not a continuation of the previous episode's
  integrator.
- `crates/es-data/tests/lerobot_v3.rs`: `export_layout_is_v3` asserts both action columns survive
  V1b's v3.0 export with `float32` and shape `[NJ]`, and the fixture the whole file shares now
  writes both — so `lerobot_v3_export`, the live `lerobot 0.6.1` read oracle, reads a dataset with
  the new column. If 0.6.1 refused an extra feature, that oracle is what would say so.

Reference — the oracle server (`ES_PYTHON`, `~/venvs/es-lerobot-cuda/bin/python`):

```
cargo test -p es-data --test lerobot_v3 -- --ignored --nocapture      # RAN lerobot_v3_export
cargo test -p es --test cli -- --ignored expert_solves_the_pinned_seeds
es loop collect --expert so101-pick-place --episodes 3 --seed 1       # 3 successes, not 1
```

**The measurement** (the point of plan V, oracle server, not a CI tier): re-collect 50 training
episodes plus 5 held-out in **one** `es loop collect --episodes 50` command with frames,
`es dataset bake --policy`, retrain 20,000 steps at `--batch 8 --lr 1e-4 --seed 0` with checkpoints
at 1k/5k/20k — every training knob identical to V2 and V2b — pack three bundles and re-run V3's
measurement exactly as V3 and V2b ran it: 16 nominal episodes per checkpoint, the six-suite table
on 20k, `visible_learning_demo_run` with `ES_TRAINED_BUNDLE`, `es video mosaic` +
`python/es/encode_video.py` + an `ffmpeg -c:v libx264` copy. `evaluation.toml`'s
`success_rate >= 0.5` acceptance **is not lowered**; the measured number is reported against it,
whatever it is. One variable moves in this experiment, and it is the demonstrations' action
convention.

**The diagnostic that comes first** (design note section 7.10): V2b's existing 20,000-step
checkpoint, re-packed into a bundle whose deployment document is a *scratch* copy with the
velocity, acceleration, end-effector and action-rate limits widened until the recorded lead fits,
run over the same 16 nominal episodes. It is five minutes and it says whether the policy can do the
task at all — which decides how to read every number this packet produces. The committed
`deployment.toml` is not edited, and nothing is disabled (INV-12).

## acceptance

```rust
// crates/es-env/src/domains.rs
impl<const NJ: usize, const H: usize> DomainRunner<NJ, H> {
    /// What the last `emit_actions` handed the Safety Plane, before the plane answered:
    /// `n_envs * NJ`, row-major by env. Provenance, never an actuator value (`INV-12`).
    /// On a tick the chunk buffer had no row for, the plane's own answer is repeated;
    /// `action_source` reads `Fallback` for exactly those ticks.
    pub fn commanded(&self) -> &[f64];
}

// crates/es-data/src/collect.rs
/// The raw pre-plane command, beside `action` (spec 13.2).
pub const ACTION_COMMANDED: &str = "action_commanded";
```

```
info.json features:
  "action":            { "dtype": "float32", "shape": [nu] }   # the SafeAction, unchanged
  "action_commanded":  { "dtype": "float32", "shape": [nu] }   # new
```

- `Collector::run` calls `planes[0].observe_state(&q, &qd)` with the backend's own joint state
  before every `step_with_policy`, at the same point in the cycle `es_eval::runner` calls it.
- `action` keeps its meaning and its bytes: `ep.ctrl`, which is `safe.q` copied by `emit_actions`
  and by `Env::step`. This packet does not change what `action` is; it makes what it is *true by
  construction* and pins it with an oracle.
- `dataset_schema_hash` moves. `task_hash`, `observation_hash`, `learning_hash` and the lowering do
  not, so the baked set, `evaluation.toml` and the packed bundles stay valid documents.
- No new trait (INV-17), no new dependency, ≤ ~150 source lines.

## forbidden

- `crates/es-safety/**` — no envelope change, no new method, no signature change (INV-11, INV-12,
  INV-13). `observe_state` and `last_safe_action` already exist and are already called by three
  other crates; this packet adds a fourth caller.
- `tests/fixtures/visible-learning/deployment.toml` — the demo's envelope does not move. The
  diagnostic uses a scratch copy that is never committed and never read by a test, and the fixture
  is restored before anything else runs. Widening the committed envelope is open question 12's
  option (c), and the point of this packet is option (a).
- `crates/es-policy/**`, `crates/es-ir/**`, `tests/fixtures/visible-learning/learning.toml` — the
  Learning IR and the `act_like` graph stay exactly as V2b left them. No delta action space
  (option (b)), no change to the lowering, `lowering_hash` must not move.
- `crates/es-data/src/lerobot/{columns,meta,v3}.rs` — the writers are generic over the declared
  features; if a new column needed an edit there, that would be the finding, not the fix.
- Lowering `evaluation.toml`'s `success_rate >= 0.5`, or any threshold anywhere. A measurement that
  fails its acceptance is reported as failing it.
- Retraining with anything V2 and V2b did not use: same seed, same batch, same learning rate, same
  step counts, same 50 episodes, same held-out seeds.

### artifacts (oracle server, RTX 4090; nothing below is committed)

Everything lives under `~/artifacts/plan-v/v1c/`; the small ones were copied to the requester's
`target/plan-v/v1c/`. Frames, tiles and checkpoints are not committed (design note section 9: the
frames are the evidence, the mp4 is a view of them).

| Artifact | How |
|---|---|
| `ds-train`, `frames-train` | `es loop collect --expert so101-pick-place --episodes 50 --seed 1 --frames …` — **one command**, which is what the episode-reset fix buys |
| `ds-holdout`, `frames-holdout` | the same, `--episodes 5 --seed 101` |
| `untrained.esb` | `cargo test -p es --test cli dataset_bake_writes`, then its `policy.esb` |
| `baked/` | `es dataset bake --policy untrained.esb --frames frames-train ds-train` |
| `model-{1000,5000,20000}.safetensors` | `train_act.py --baked baked/ --batch 8 --lr 1e-4 --seed 0 --device cuda --checkpoint-at 1000,5000,20000` |
| `trained-{1000,5000,20000}.esb` | `es policy pack --policy untrained.esb --weights model-N.safetensors` |
| `nominal-{1000,5000,20000}/`, `suite-20000/` | `es eval run --config … --frames` |
| `demo-{1000,5000,20000}.mp4`, `-h264.mp4` | `es video mosaic --grid 4x4` + `encode_video.py --fps 50` + `ffmpeg -c:v libx264` |
