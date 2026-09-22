# M10 W0b — `scene_hash` without the host libm, and the record re-measured under the moved hashes

Spec: §3.2 (the paragraph added 2026-09-22: `f64` on the offline asset path goes through
`es_math::approx::{sin_cos_f64, acos_f64}`, the pure-Rust `libm` port; `DET-010` names the host's
functions), §3.4, §5.3 (`scene_hash` → `task_hash` → everything), §28.13 rule 1 (a hash fix moves
hashes and nothing else) and wave 0, §28.9 rule 2 (invalidated numbers are marked, not deleted),
§1.4 (golden files are CI read-only). Review: `docs/reviews/M8.md` S-2 and the human decision
"`scene_hash` across platforms" — the owner chose on 2026-09-22 to **re-collect, retrain and
re-score under the new hashes**. Design notes: `docs/design/transcendental.md` (gains the `f64`
paragraph), `visible-learning.md` (row 7.36), `rl-continuation.md` section 7 (the re-measured A0
row). Type B with a D half.

## the question

`crates/es-assets/src/mjcf/orient.rs:56` (`f64::sin_cos`, every `euler=` and `axisangle=`), `:75`
(`acos`, `zaxis=`) and `crates/es-assets/src/urdf.rs:716-718` (`sin_cos`, `rpy`) call the
platform's libm, so the same `so101_pick_place.xml` hashes to different `scene_hash`es on Windows
and Linux (measured, M8/S4d) and `crates/es/tests/cli.rs:12041-12057` works around it. **With
those three calls routed through the pure-Rust `libm` port, is `scene_hash` one number on every
platform — and, since the fix moves every SO-101 `task_hash`, do the demo and the reach task
re-collected, retrained and re-scored under the moved hashes reproduce every committed number
bit for bit?**

## spec

* `crates/es-math/Cargo.toml` gains `libm` (already in `Cargo.lock`; `no_std` by birth — the
  route `es-safety` took in `embedded-runtime.md`). `crates/es-math/src/approx/mod.rs` gains
  `pub fn sin_cos_f64(x: f64) -> (f64, f64)` and `pub fn acos_f64(x: f64) -> f64`, thin wrappers
  over `libm::sincos` / `libm::acos`, documented as: `f64`, CPU-only, no Slang mirror, offline
  asset path only; bits fixed by the crate, not the platform; not the ≤ 2 ULP polynomial family.
  A unit test pins the bit patterns (`to_bits()` hex) of `sin_cos_f64(1.57 * 0.5)`,
  `sin_cos_f64(-0.55 * 0.5)` and `sin_cos_f64(core::f64::consts::FRAC_PI_4)` — the SO-101 scene's
  two half-angles and a control — so the Linux run asserts the same constants.
* `orient.rs` `axis_angle` and `from_zaxis`, `urdf.rs` `quat_from_rpy` call the wrappers; the
  `orient.rs:7-8` header ("the spec 3.2 restriction does not apply") is rewritten to say why it
  does (the bits enter `scene_hash`). `f64::sqrt` stays (IEEE, correctly rounded).
* **Regeneration, once, with the existing generators** (`crates/es/tests/cli.rs`
  `generate_pt_fixtures`, `generate_rl_learning_document`, `generate_augmented_observation_fixture`,
  `generate_train_goldens` … — run them under their env flags; where a committed document has no
  generator, add one in the same `#[ignore]` style rather than editing bytes by hand):
  the four SO-101 task documents (`tests/fixtures/visible-learning/task.toml`, `task-pt.toml`,
  `tests/fixtures/rl/task-reach.toml`, `task-reach-delta.toml` — `scene_hash` only; every other
  byte unchanged), the seven Observation IRs' `task_ref`, the Evaluation IRs and any Learning IR
  that pins a task or observation hash, the training recipes' header comments that quote a
  hash, and the pinned constants in `crates/es-ir/tests/{joint_quantity,sensor_render}.rs`,
  `crates/es/tests/cli.rs` (`9581` and wherever `eb6efefa`, `d546b808`, `b5d3b813`, `4ced8547`,
  `7caac85d`, `967ea296` … appear). The three `task.scene.scene_hash = scene.scene_hash()`
  overwrites (`cli.rs:3372`, `:11199`, `:12056`) and the `12041` comment go: the tests now
  assert equality — that assertion is the defect's oracle. `tests/fixtures/quadruped/task.toml`
  (`xyaxes` only, `sqrt`) is the control: its hash is pinned unmoved before the change.
* `tests/golden/**` do not move (`cargo xtask verify-goldens`); `tests/golden/rollout/
  so101_100steps.json` was generated on Windows and reproduced on Linux across the old
  discrepancy, so it must survive this fix — if it does not, that is a finding, not a
  regeneration.
* **Server (the D half)** — fresh `git archive` of the branch into `~/Projects/es-w0b`,
  artifacts `~/artifacts/plan-w/w0b/`, both queue locks (`~/artifacts/plan-w/queue/{gpu,cpu}.lock`)
  held for the duration, after W0a's measurement has released the CPU lock; every stage under
  `nohup` with `<stage>.start/.end/.done/.log`:
  1. `scene_hash` of the five committed scenes on Linux == the regenerated documents' values
     (the decisive oracle), and the three pinned bit patterns.
  2. V15 re-collected: `es loop collect` 200 episodes, seed 1, the expert, `--frames`, under the
     regenerated documents → `~/artifacts/plan-w/w0b/v15/ds-train`; the dataset's `content`
     digest and every frame beside `~/artifacts/plan-v/v15/ds-train` (expected identical: the
     physics did not change on Linux; if it did, the ULP by which the quaternions moved is in
     the note).
  3. U3 retrained: `training-u3.toml`'s recipe on the new dataset with the re-packed untrained
     bundle (`observation-augmented.toml` regenerated, `learning-pretrained.toml`) → the
     checkpoint's tensors beside `~/artifacts/plan-v/m7-u/U3/train/checkpoints/20000.esb`
     (expected bit-identical; same data, same seed).
  4. U3′ scored: the regenerated `evaluation-augmented` (16 held-out seeds × 6 suites) and the
     train-seed document, `--jobs 4`, plus the expert gate → `report.json` beside
     `~/artifacts/plan-v/m7-u/U3/holdout/report.json` (expected identical except the hashes:
     0.5625 / 0.6668 / 1092.1).
  5. Reach A0 re-measured: `training-reach.toml` (`learning-reach.toml`, 64×64) three seeds ×
     4,000 iterations, `evaluation-reach.toml` → beside 0.4167 (0.5625 / 0.3125 / 0.3750), CPU
     queue, three concurrent.
  Rows: `visible-learning.md` 7.36 ("re-measured under the platform-stable `scene_hash`":
  the old and new hashes side by side, each number "bit for bit" or the deviation named),
  `rl-continuation.md` section 7 (the A0 row, same rule). Wall-clocks recorded; the nine
  metrics `Target / Status: unverified` where not measured.

## context

```
crates/es-math/Cargo.toml
crates/es-math/src/approx/mod.rs
crates/es-assets/src/mjcf/orient.rs
crates/es-assets/src/urdf.rs
crates/es-assets/tests/**
crates/es-ir/tests/**
crates/es-eval/tests/**
crates/es/tests/cli.rs
tests/fixtures/visible-learning/**
tests/fixtures/rl/**
tests/fixtures/quadruped/**
Cargo.lock
docs/design/transcendental.md
docs/design/transcendental.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M10/W0b-scene-hash-libm.md
docs/packets/M10/W0b-scene-hash-libm.ko.md
```

## oracle

1. `cargo test -p es-math approx_f64_bit_patterns_are_pinned` — the three constants.
2. `cargo test -p es-assets` — `orientations.xml` and the SO-101 scene hash equal a pinned hex
   on this platform (Windows) **and** the same test green on the server (Linux); the quadruped
   hash unmoved.
3. `cargo test --workspace` — every regenerated pin green; the `cli.rs` scene-hash equality
   assertions (no overwrite) green on both platforms.
4. `cargo xtask verify-goldens` — no golden moved.
5. Server stages 1–5 above, the two rows written, artifacts under `~/artifacts/plan-w/w0b/`.
6. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W0b-scene-hash-libm.md`.

## acceptance

Oracles 1–6; the `transcendental.md` paragraph; rows 7.36 and the A0 row with their Korean
siblings; a one-table summary of old → new hashes in this packet's note section.

## forbidden

Touching the `f32` `approx` family or `crates/es-math/slang/**`; any `tests/golden/**` edit;
`docs/ARCHITECTURE*.md` (the paragraph is in); editing a committed document by hand where a
generator exists; changing physics, the emitter, or any number's meaning; `tests/fixtures/
quadruped/task.toml`'s hash moving; `es-render`, `es-env`, `es-physics-*` (W2a owns the next
`es-assets` change and starts after this merges).
