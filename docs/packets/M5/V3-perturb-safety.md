# M5 V3 — randomized evaluation, the Safety Plane event stream, and the demo run

Design note: `docs/design/visible-learning.md` sections 2.7 and 8; read section 2.7 first — Task IR
randomization can move the cube but `PerturbationKind::ObjectPose` cannot, and four visual kinds stay
`Unsupported` on purpose. Depends on V2 (a trained bundle) and V0b (frames); produces the input V4 turns
into the video.

## context

```
crates/es-eval/src/perturb.rs
crates/es-eval/src/runner.rs
crates/es-eval/src/lib.rs
crates/es-eval/tests/evaluation.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/evaluation.toml
crates/es-eval/tests/visible_learning.rs
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/packets/M5/V3-perturb-safety.md
docs/packets/M5/V3-perturb-safety.ko.md
```

Notes: `perturb.rs` gains two kernels and loses two `Unsupported` arms; `runner.rs` gains the per-step
event record and the frame sink; `eval.rs` gains `--frames <dir>`. `docs/design/evaluation-execution.md`
section 3 is the support table and must be updated in the same commit, because it is what says why a kind
is unsupported.

## spec

- §10.1, §10.2: the suite is `perturbation suite x metric x acceptance`. Two kinds move from
  `Unsupported` to implemented; the rest keep their reason, restated for the new state of the world.
- §10.4: fairness — every draw is `EnvRng::new(seed, suite_id, episode_idx, stream)`
  (`crates/es-eval/src/perturb.rs:9-12`). Enabling a lighting kind must not perturb any other stream's
  cursor, or every existing report changes.
- §10.5: `es eval compare A.json B.json` still works on the reports this packet writes.
- §9.3–§9.5, INV-12, INV-13: the red overlay reads the Safety Plane's existing per-step outcome. No code
  path disables the plane; the demo's clamps come from a **tightened envelope in the Deployment IR**, not
  from a test hook. `SafetyPlane::validate` keeps its signature.
- INV-11: `es-safety` gains nothing and depends on nothing new.
- §26.1: an `ImageSpec` mismatch between the Observation IR and the renderer stays an error (V0b), and a
  perturbation this build has no kernel for stays `EvalError::Unsupported` naming it — never a silent
  no-op (§17.2's rule, which `perturb.rs` already follows).
- §3.5: the report is reproducible from `evaluation_hash`; the frames are byte-identical on the CPU render
  path; the physics is `DeterminismTier::PhysicsMeaning`, not bitwise. Design note section 9 is the table.
- §12.4: the demo's headline number is a success rate, not a rate of anything per second.
- §1.5: `es-eval` is at 2,139 code lines; this packet is budgeted under ~500.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo clippy -p es-eval --features render --all-targets -- -D warnings
cargo test -p es-eval
cargo test -p es-eval --features render --test visible_learning
cargo test -p es --test cli eval_run
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

Reference — the demo run itself, on the oracle server:

```
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es-eval --features render --test visible_learning -- --ignored --nocapture
```

`tests/visible_learning.rs` runs V0's scene with V2's trained bundle over
`tests/fixtures/visible-learning/evaluation.toml` and judges the run. Missing `mujoco` or `torch` ->
`SKIP visible_learning: <why>`; ran -> `RAN visible_learning`. A genuine backend or runtime error is a
**failure**, not a SKIP.

**The gate, and it is non-vacuous** (the rule W1d's gate uses, `docs/design/ros2-boundary.md` section 7.5):
the run passes only if it contains at least one `Termination::Success` episode **and** at least one step
classified `Clamped`. A demo whose Safety Plane never fires demonstrates nothing, and a demo with no
success demonstrates nothing either. Neither is produced by a flag: successes come from V2's policy and
clamps come from the suite's tightened envelope.

`tests/visible_learning.rs` and `tests/evaluation.rs`:

- `sixteen_cells_produce_sixteen_frame_dirs` — the grid is **16 independent single-env episodes**, because
  `Evaluation::run` hardcodes `BatchDomains::single_env()` (`crates/es-eval/src/runner.rs:132`) and
  `MuJoCoCpuBackend` declares `max_envs: 1` (`crates/es-physics-backend/src/mujoco.rs:43`). The test
  asserts one `frames/<cell>/` per cell with equal `layout.json`s, which is what V4 needs to mosaic them.
- `events_json_has_one_record_per_frame` — `events.json` length equals the frame count, `frame` is dense
  and ascending, and every `tick` is the `PhysTick` of that step.
- `a_tightened_envelope_clamps_and_is_recorded` — with the demo suite's limits, at least one record says
  `Clamped`, the `ViolationKind` bitset for that step is non-zero, and `SafetyCounters` agrees with the
  record count.
- `a_widened_envelope_never_clamps` — the same run with a wide envelope produces zero `Clamped` records
  and the same trajectory, proving the overlay reads the plane rather than the policy.
- `the_plane_is_never_disabled` — a source scan of `es-eval` finds no path that skips `validate`
  (INV-12), and `SafetyPlane::validate`'s signature is unchanged (INV-13).
- `light_intensity_and_direction_change_the_frame` — with a renderer, the two new kinds produce a frame
  that differs from the nominal cell's; without one they stay `EvalError::Unsupported` with the existing
  `NO_RENDERER` message.
- `the_four_remaining_kinds_are_still_unsupported_by_name` — `ColorTemperature`, `Occluder`,
  `CameraExtrinsic`, `CameraIntrinsic` each return `EvalError::Unsupported` naming the kind and its
  reason; `ObjectPose` keeps its own ("`Env::reset` takes no state override",
  `crates/es-eval/src/perturb.rs:196-200`). Cube-pose variation in the demo comes from V0's Task IR
  `Randomization`, not from this kind.
- `enabling_the_light_kinds_does_not_move_other_streams` — an existing evaluation fixture's report is
  byte-identical to the one committed before this packet. This is the §10.4 fairness check, and it is the
  test most likely to catch a careless stream index.
- `the_report_is_reproducible` — two runs of the same `evaluation_hash` give identical `report.json` and
  `evaluation.lock`.
- `the_frames_are_byte_identical_on_the_cpu_path` — two runs give byte-identical `frames/**/*.bin`
  (design note section 9).

`crates/es/tests/cli.rs`: `eval_run_frames_flag_writes_frames_and_events`,
`eval_run_without_frames_is_unchanged` (the existing report output is byte-identical),
`eval_run_frames_without_a_renderer_is_a_usage_error` — exit 2 naming the missing feature, not a silent
run with no frames.

## acceptance

```rust
// crates/es-eval/src/runner.rs
/// One record per rendered frame, written as `events.json` beside `frames/`.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct StepEvent {
    pub frame: u64,
    pub tick: es_core::PhysTick,
    pub source: ActionSource,     // Policy | Clamped | Fallback | Human
    pub events: u32,              // ViolationKind bitset for this step
}

/// Where a run puts its frames and events; `None` is today's behaviour exactly.
pub struct FrameSink { pub dir: std::path::PathBuf }
```

```
es eval run --config eval.toml --policy trained.esb --scene so101_pick_place.xml \
            --out demo-out [--frames demo-out/frames]
# demo-out/
#   report.json  evaluation.lock  report.html
#   frames/<cell>/000000.bin ... + layout.json
#   events.json
```

- `ActionSource` is the four-way classification `es-data` already derives from `SafetyCounters` deltas
  (`crates/es-data/src/collect.rs:461-471`); V3 reuses that derivation rather than adding a second one.
  If it must move to be shared, it moves to `es-eval` (10) and `es-data` (10) cannot depend on it — so it
  is duplicated with a comment naming the original, or lifted to `es-safety` (8) in a separate packet, not
  here.
- `PerturbationKind::LightIntensity` and `LightDirection` resolve to kernels that scale and rotate the
  scene's light before `TriScene` upload; each draws from its own `stream`, and no existing stream's
  cursor moves.
- `ColorTemperature`, `Occluder`, `CameraExtrinsic`, `CameraIntrinsic` and `ObjectPose` keep returning
  `EvalError::Unsupported` with a reason that is true after this packet, and
  `docs/design/evaluation-execution.md` section 3's table is updated to match.
- The demo Deployment IR tightens velocity and rate limits for one suite cell. No flag, no `cfg`, no test
  hook disables or bypasses the plane (INV-12).
- `--frames` without `es-eval/render` is a usage error (exit 2), never a run that quietly writes nothing.
- No new trait, no new external dependency, ≤ ~500 source lines.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` — no new `PerturbationKind`, no `MetricSpec`, no schema change.
- `crates/es-safety` — the envelope is widened or tightened in the Deployment IR document; the crate is
  untouched, and `validate` keeps its signature (INV-11, INV-12, INV-13).
- Implementing `ObjectPose`: it needs `Env::reset_with`, which is an `es-env` packet, and the demo does
  not need it (Task IR randomization already moves the cube).
- Implementing `CameraExtrinsic` / `CameraIntrinsic`: INV-14 requires the `ImageSpec` intrinsics to move
  with the camera rather than be approximated, which is more than a perturbation kernel.
- Raising `BatchDomains::single_env()` or adding an `--envs` flag: the grid is 16 independent episodes
  (design note section 2.1), and batching the simulation domain is a different packet.
- `crates/es-render`, `crates/es-env/src/render.rs` — V0b owns the renderer surface.
- Mosaicking, overlaying or encoding anything — V4.
- Editing an existing evaluation report golden to accommodate a moved RNG stream.
