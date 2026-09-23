# M11 X2 — adapter v2: an Isaac Lab or Playground policy's I/O conventions, declared and compiled

Spec: §28.14 rule 3 (the IR owns an external policy's I/O conventions; they are declared in the
importer's adapter and compiled into the bundle; nothing framework-specific runs at inference; a term
the IR cannot compute is refused by name) and rule 1 and wave 1, §2.4 / §8.7 (the IR owns
pre/post-processing), §6.2 (`ObservationSpec`), §7 (`TemporalWindow`), §14.4, INV-16 (no pickle at
inference). Design notes: `rl-continuation.md` section 8 "The importer and the adapter" (+ko);
api-notes `isaac-lab.md` (M11 W0: the ObservationManager / ActionManager formulas),
`brax-ppo-so101.md`. Type B.

## the question

`adapter.toml` v1 (`crates/es-import/src/rl_import.rs:116-212`, IMP-001..005) expresses an action
joint permutation, radians, `ctrl = offset + scale·a`, contiguous observation slices mapped onto Task
IR channels, and a mean/std normalizer folded into the first layer. Every Isaac Lab manager-based
policy also uses `joint_pos − default_joint_pos`, per-term `scale`, `last_action`, optional
`history_length` and `clip`, joints resolved **by name** in the articulation's order, and a
`decimation × sim.dt` policy period; Playground's MJX envs do the analogous things. **Can each of
those be declared once in the adapter and compiled into the bundle — exactly, by folding where the
algebra is exact — so that the bundle reproduces the source framework's observation → action map
on our state, with every v1 adapter's output unmoved?**

## spec

* Adapter v2 fields, each optional, `deny_unknown_fields` kept, absent = v1 behaviour byte for byte:
  * `[joints] source_names = [...]` — the source's joint names in its articulation order, resolved
    against our actuator/joint names (an explicit `[joints.rename]` table when names differ);
    replaces `source_order` when present (both present = IMP-008). Observation slices whose channel
    is a per-joint channel are permuted into our order **by permuting the first Dense layer's input
    columns**; the action by permuting the head rows (exact).
  * `[joints] default_pos = [...]` (radians, source order).
  * Per `[[observation.channels]]`: `scale`, `offset` (`"default_pos"` or a vector), `clip`,
    `history` (N frames, newest-last or newest-first as declared). `(x − offset)·scale` folds into
    the MeanStd normalizer (`mean = offset`, `std = 1/scale`); `clip` becomes an Observation IR
    clamp node (existing node type if one exists; else IMP-009 refusal — no new node type in this
    packet); `history` becomes a `TemporalWindow` of N.
  * `[action] use_default_offset = true` → `offset = default_pos` (Isaac's `JointPositionActionCfg`);
    `clip = [lo, hi]` folds into the head's clamp.
  * `[timing] policy_dt` — checked against the Deployment IR control period; a mismatch is IMP-006.
  * `[actuators]` `stiffness`, `damping`, `armature`, `effort_limit` (optional) — compared to the
    scene's actuators and written into the MappingReport as rows; never converted.
  * Channels the IR cannot compute (`projected_gravity`, `base_lin_vel`, `base_ang_vel`,
    `velocity_commands`, `generated_commands` other than a Task IR goal channel) → IMP-007 naming
    the term.
* `prev_action`: a new Task IR `ObsSource::PreviousAction { initial: Option<Vec<f64>> }` —
  the action row the policy emitted for the previous control tick (before the Safety Plane, in
  actuator units), `initial` (absent = zeros) at the episode's first tick. The adapter writes
  `initial = offset` so the folded value is 0 at reset, which is Isaac's `last_action`. It is the
  **last** variant of `ObsSource` so no committed hash moves. Its value is produced wherever
  observations are captured today — `es_eval::runner::capture` (evaluation and collection) and
  `Rollout::observe` — from the previous policy output the loop already holds.
* `python/es/import_rl.py`: reads rsl_rl checkpoints with empirical normalization (the
  `obs_normalizer` running mean/var → `obs_mean`/`obs_std` in the manifest) and Playground brax
  params; records `default_joint_pos`, `action_scale`, `decimation`, `sim_dt` into the manifest
  when the source config is given (`--isaac-env-cfg <yaml/json>` / `--playground-config`).

## context

```
crates/es-import/src/rl_import.rs
crates/es-import/src/lib.rs
crates/es-import/tests/**
crates/es-ir/src/task.rs
crates/es-ir/src/serial.rs
crates/es-ir/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es-py/src/rollout.rs
crates/es/src/cmd/policy*.rs
crates/es/tests/cli.rs
python/es/import_rl.py
python/es/rl_source/**
tests/fixtures/rl/**
tests/golden/**/import*
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/X2-adapter-v2.md
docs/packets/M11/X2-adapter-v2.ko.md
```

## oracle

1. `cargo test -p es-import adapter_v2_` — each new field parses; each refusal (IMP-006..009) is by
   name; every committed v1 adapter's converted Observation IR / Learning IR / weights are
   byte-identical to before (hashes pinned).
2. `ES_PYTHON=… cargo test -p es-import --test isaac_reference` — two synthetic sources written by a
   generator: an Isaac-Lab-style rsl_rl actor (ELU MLP, empirical normalizer, `default_joint_pos`,
   joint_pos_rel ×1, joint_vel_rel ×0.05, target pose, last_action, `action_scale 0.5`,
   `use_default_offset`, source joint order ≠ ours, `clip_observations 100`) and a Playground-style
   brax actor. `python/es/rl_source/isaac_reference.py` implements the frameworks' formulas **from
   the api-note, in NumPy**; on 256 random states (with a previous action) the imported bundle run
   through `es_policy` equals the reference within **1e-6** (max abs, actuator units).
3. `cargo test -p es-ir previous_action_` — committed task hashes unmoved; a `PreviousAction` channel
   moves it; `initial` round-trips.
4. `cargo test -p es-eval previous_action_is_the_last_policy_row` — tick 0 = `initial`, tick k = the
   policy's row at k−1 (before the plane), for a buffered (chunk) and a horizon-1 policy.
5. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

Oracles 1–5; `rl-continuation.md` section 8 gains the v2 table (source convention → where it lands)
with its Korean sibling; the refusal list printed by `es policy import-rl --help`.

## forbidden

Framework code at inference; pickle outside `import_rl.py` (INV-16); a new Observation IR node type;
guessing a default pose, scale or order that neither the adapter nor the manifest states; moving any
committed hash; `es-safety`.
