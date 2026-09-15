# M5 V1c — demonstrations record the executed action

Design note: `docs/design/visible-learning.md` sections 7.5–7.9 and **open question 12**. Read them
first. Depends on V1 (the collector, the scripted expert), V1b (the v3.0 export), V2/V2b (the
lowering, the training script, the baked set) and V3 (the measurement this packet re-runs). It
fixes a defect V2b measured but was forbidden from touching.

## the defect

**Three findings, one of which this packet is forbidden from fixing.**

**1. `action` is already the executed action, and nothing said so.** `DomainRunner::emit_actions`
copies `SafeAction::q` into `ctrl`, `Env::step` records `ctrl`, and `to_lerobot` writes it as
`action` — so the column has always been post-clamp. Open question 12's option (a) ("record
`action` as the next commanded position rather than the servo target") describes a change that
was already made, which nobody could see because nothing in the repository asserted it and the
raw command was thrown away. This packet pins it with a bit-equality oracle and keeps the
pre-plane command in a second column, `action_commanded`, so that a `Clamped` frame can be read
without re-running the plane.

**2. `es loop collect --episodes N` only solved episode 0, and the cause is the inference phase.**
The scripted expert is reset from the intervener hook on `frame == 0`. That hook is
`PolicyRuntime::infer`, which runs when a submitted observation is *released* —
`expected_latency_ms` ticks after the submit (§12.3). The demo's `learning.toml` declares
`15.0` ms against a 20 ms control period, so **the first call of every episode is frame 1, and
`frame == 0` never fires at all**. It looked correct only because a freshly constructed
`ScriptedExpert` starts reset: episode 0 needs no reset and gets away with it, and every episode
after it runs with the previous one's stage and latched cube pose. Keying on the episode index
instead takes `--episodes 50 --seed 1` from **1/50 to 50/50 successes in one command** (oracle
server, measured). The plane's own carried state is a second, smaller instance of the same thing:
`reset_latch` clears the latch but not the hold target, the velocity or the rate history, so the
collector now seeds them from the measured pose once per episode — which is exactly what
`SafetyPlane`'s own documentation says a caller that knows the real pose does before the first
`validate`.

**3. `es loop collect` and `es eval run` disagree about what the envelope is measured against —
and this packet reports it rather than settling it.** `es_eval::runner`, `es_ros2::hil` and
`es_runtime_embedded` call `observe_state` before **every** `validate`, which re-seeds
`last_safe`, `prev_safe`, `vel` and `prev_vel`; the envelope is then a bound on the **following
error**, and the demo's `acceleration_max` makes that bound `0.008` rad. `Collector::run` seeds it
once per episode, so inside an episode the envelope bounds command-to-command motion — which is
what `ScriptedExpert::chunk` paces itself to, and its doc comment says so. The demonstrations'
`action` therefore leads its own `qpos` by a median `0.2413` rad, four times what the evaluation
path allows, which is why V2b measured `envelope_violation_rate 1.0000` and **not one step in
86,400 classified `ActionSource::Policy`**.

Making the collector re-seed every step *was tried and measured*: it drops the scripted expert to
**2/8 on `expert_solves_the_pinned_seeds`**, whose threshold is `0.875` and is a golden, and to
**0/16 on the evaluation's own seeds 101–116**. The expert plans a 16-row chunk executed over 10
ticks from the pose at chunk start; no constant re-pacing satisfies a `0.008` rad following-error
bound over ten open-loop ticks, and the two knobs that would (`deployment.toml`'s envelope,
`execute_chunk`) are both in this packet's `forbidden` list. So the asymmetry is written down with
numbers in design note section 7.10 and open question 12 stays open, now with the measurement that
says which end has to move.

**The principle this packet applies: what is executed is what is recorded** — and now it is
asserted, labelled, and one reset per episode is one reset of everything.

## context

```
crates/es-env/src/domains.rs
crates/es-env/src/expert.rs
crates/es-data/src/collect.rs
crates/es-data/src/lib.rs
crates/es-data/tests/loop_learning.rs
crates/es-data/tests/lerobot_v3.rs
crates/es/src/cmd/loop.rs
crates/es/tests/cli.rs
docs/api-notes/lerobot-dataset.md
docs/api-notes/lerobot-dataset.ko.md
docs/design/learning-loop.md
docs/design/learning-loop.ko.md
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

- **§9.3, §9.4, INV-12.** Nothing is bypassed and nothing is disabled. The plane gains one call it
  already has an API for — `observe_state`, once per episode, which its own doc comment says a
  caller that knows the real pose makes before the first `validate` — and `validate` keeps its
  signature (INV-13). The only value that travels toward the actuator is still the `SafeAction`;
  `action_commanded` is provenance on disk and is never read back into a control path.
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
  plane half of the episode-reset regression. The fixture's uniform reset draw is pinned to a
  constant and the velocity envelope tightened, so two episodes of one run are physically
  identical; `action`, `action_commanded`, `observation.state` and `action_source` must then be
  equal row for row. **Measured to fail before this packet** (episode 1 continued from episode 0's
  last command, `0.065` where episode 0 started at `0.005`).
- `crates/es-data/tests/loop_learning.rs`: **`frame_zero_is_not_a_hook_an_intervener_may_reset_on`**
  — the expert half, and the one that matters. With a contract declaring one control tick of
  inference latency, **no** call of the intervener has `frame == 0`, in any episode. That is the
  assumption `es loop collect --expert` was built on, and the test states it as a property of the
  collector rather than as a comment. The fixture declared `expected_latency_ms = 0.0`, which is
  why V1's own oracle could not have seen it.
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
cargo test -p es --test cli expert_solves_the_pinned_seeds            # 8/8, unchanged
es loop collect --expert so101-pick-place --episodes 50 --seed 1      # 50 successes, not 1
```

`expert_solves_the_pinned_seeds` is the guard that decides how much of the collect/eval asymmetry
this packet may take on: its `0.875` threshold is a property of the expert and lowering it is
editing a golden. Any change to when the collector seeds the plane must keep it at 8/8.

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

- `Collector::run` calls `planes[0].observe_state(&q, &qd)` with the backend's own joint state on
  the first frame of every episode — "before the first `validate`", which is what the method's
  documentation asks for. **Not** every step: that is `es_eval::runner`'s reading of the envelope,
  it is a different reading, and adopting it here drops the expert to 2/8 on V1's own oracle.
- `es loop collect --expert` resets the scripted expert on a change of **episode index**, never on
  `frame == 0`.
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
