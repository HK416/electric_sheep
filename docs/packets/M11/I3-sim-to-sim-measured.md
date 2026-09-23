# M11 I3 — sim-to-sim, measured: an Isaac Lab policy and our own, on three engines

Spec: §28.14 rules 2, 3, 6, 7 and wave 3, §3.5 (tier 2 numbers are reported, never claimed
bitwise), §10.4, §12.4. Depends on X1 (+ follow-up), X2, I1, P-M11-R1 (all merged). API notes:
`isaac-lab.md` (the reach env cfg, rsl_rl, exporter), `isaac-sim.md` §7–8. Design note:
`rl-continuation.md` (+ko) gains section 7's I3 subsection. Type D.

## the question

Every earlier continuation number ended on "the source policy was trained in a scene we do not
have" (M8 S-5, M9 S-8). With a PhysX backend and adapter v2, **does an SO-101 reach policy trained
in Isaac Lab on our own scene, imported through adapter v2, score in our runtime what it scores in
Isaac Lab — and how do it and our own PPO policy (A0) move across `physx`, `mujoco-cpu` and
`mjwarp`?**

## spec

* **The Isaac Lab task** (`python/es/rl_source/isaac_so101_reach/`, a manager-based env cfg, not a
  fork of Isaac Lab):
  * The scene is our `so101_pick_place.xml` as `scene_to_mjcf` emits it. It is imported with
    I1's fixups (the same `physx_ref.py` import function, reused, not copied), so the PhysX model
    the policy trains on is the one `--backend physx` evaluates on.
  * Observation terms mirror `tests/fixtures/rl/task-reach-last-action.toml`: `joint_pos_rel`,
    `joint_vel_rel` (declared scale), cube pose and gripper pose in the robot base frame,
    `last_action`, in that order.
  * The action is `JointPositionActionCfg(scale, use_default_offset=True)`.
  * The control period is 20 ms (50 Hz) and must equal `deployment-reach.toml`'s: decimation × dt.
  * The reward is `-‖cube − gripper‖` + 1 on success (< 0.03 m), the episode ending there or at
    200 control steps.
  * Resets draw from the same ranges as the Task IR's `ResetState` nodes.
  Whatever cannot be mirrored exactly is listed in the report beside the adapter's mapping report.
* **Training:** rsl_rl PPO in Isaac Lab, 3 seeds, headless, GPU pipeline, until the Isaac-side
  success plateaus. Record the budget, the wall clock and the Isaac-side success on a held-out set
  of reset seeds.
* **Import:** `import_rl.py --framework rsl-rl` (the checkpoint shape Isaac Lab 2.3.2 writes —
  record which) → `adapter-isaac-so101.toml` (v2: `source_names`, `default_pos`, per-term scale,
  `use_default_offset`, `policy_dt`) → `es policy import-rl`.
* **Evaluate** on `evaluation-reach-last-action.toml` (16 held-out seeds, the reach suites),
  generated like the other reach documents. Run it for each of the 3 Isaac seeds × {`physx` CPU
  pipeline, `physx` GPU, `mujoco-cpu`, `mjwarp`}. A0 (3 seeds, W0b's checkpoints) gets the same
  four columns.
* **Attribution run:** the Isaac policy and A0 on `mujoco-cpu` with the MappingReport's
  approximated rows toggled one at a time where the adapter can toggle them (joint damping as
  explicit effort vs none, friction combine mode). This names which row explains most of the
  engine gap. Report numbers, not verdicts.

## context

```
python/es/rl_source/isaac_so101_reach/**
python/es/import_rl.py
crates/es-physics-backend/python/physx_ref.py
tests/fixtures/rl/adapter-isaac-so101.toml
tests/fixtures/rl/evaluation-reach-last-action.toml
tests/fixtures/rl/**
crates/es/tests/cli.rs
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/api-notes/isaac-lab.md
docs/api-notes/isaac-lab.ko.md
docs/packets/M11/I3-sim-to-sim-measured.md
docs/packets/M11/I3-sim-to-sim-measured.ko.md
```

## oracle

1. `cargo test -p es --test cli isaac_so101_documents_check` — the new documents validate,
   Cross-IR with `deployment-reach.toml`, and the adapter imports a *synthetic* checkpoint with the
   env cfg's shapes (no Isaac needed).
2. Server: the Isaac Lab training logs + checkpoints for 3 seeds (`~/artifacts/plan-x/i3/`); the
   Isaac-side held-out success per seed.
3. Server: the evaluation table — for each of Isaac seeds 0–2 and A0 seeds 0–2, the columns
   physx-CPU / physx-GPU / mujoco-cpu / mjwarp: `success_rate`, `envelope_violation_rate`,
   `episode_length`, `execution_hash`. MJWarp is run twice (not reproducible run to run — X1), and
   so is physx (it is).
4. The attribution rows.
5. fmt, clippy, check-scope, verify-goldens.

## acceptance

Oracles 1–5. `rl-continuation.md` (+ko) gains the I3 table and one paragraph answering the
question in one sentence, with the numbers.

## forbidden

Tuning the Isaac task until it matches our numbers (mirror, then measure); moving any committed
hash; a new trait; framework code at inference; claiming a tolerance.
