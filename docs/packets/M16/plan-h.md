# M16 plan H — Shadow Hand cube reorientation from three path-traced cameras

> The owner's request (2026-09-30): "`server-side-rendering/PhysicalAI/ShadowHand` 를 기반으로 여러 대의
> 카메라를 사용하여 비전 기반 로봇 학습을 수행 … PT 기반 시뮬레이션 환경 … 학습 진행을 es-editor 로 관찰".
> Asked, the owner chose **the Isaac task (cube reorientation)** and **imitation learning (ACT)**.
> Each task is a §1.2 packet: its **Files** block is its `context` for `cargo xtask check-scope`.

**Goal:** the Shadow Hand of the SSR bundle (`shadow_hand_physics_configured.xml`, Isaac Lab's
repose-cube task ported to MuJoCo) runs in this runtime; a vision policy that sees three `Pt`
cameras reorients the cube to a goal orientation; every run is watched in `es-editor`.

**Why a teacher.** Reorienting a cube in hand cannot be scripted as move/grip blocks, so the
demonstrator is itself a policy: a state-based PPO teacher that reads the cube's pose (privileged),
trained with `[rl]` on `mjwarp`; the teacher then drives `es loop collect --frames` under the three
cameras, and the student — ACT over three views plus joint state and the goal, no cube pose — is
trained on those demonstrations (teacher–student distillation; Chen et al., "Visual Dexterity",
arXiv 2211.11744). Collection with a policy instead of `--expert` is `es loop collect`'s own path.

**Measured before planning** (this PC, RTX 3060 12 GB): the bundle's MJCF steps at 39k physics
steps/s on CPU MuJoCo and 22k / 71k / 100k on MJWarp at 256 / 1,024 / 4,096 worlds (`mujoco-warp`
3.13.0 + `warp-lang` 1.16.0 installed into `.venv`; 1.17 fails in MJWarp's CCD kernel). The CPU
backend's JSON pipe gave X7 ≈ 640 control steps/s, too slow for a teacher that needs ~10⁸ steps.

## Global Constraints

- Everything in `docs/packets/M15/plan-n.md`'s Global Constraints holds (worktree + `--ff-only`,
  one `CARGO_TARGET_DIR` per agent, goldens by generator only, never push, no remote server, no
  learning run inside an agent packet, spec first for IR changes, Safety Plane untouched).
- **Every committed document, golden and number stays as it is.** The hand is new files.
- The SSR bundle is read-only input; what this repo needs from it is copied and derived here with
  a provenance note (source path, the bundle's `FILE_MANIFEST.sha256` line, every edit).

## Waves

| Wave | Tasks | Needs |
|---|---|---|
| 1 | H1 the scene in this runtime | — |
| 2 | H2 the task and teacher documents, a throughput smoke | H1 |
| 3 | E1 the teacher run and its evaluation (orchestrator) | H2 |
| 4 | H3 the student documents · E2 collect under three `Pt` cameras, train, evaluate (orchestrator) | E1 |

### Task H1: the Shadow Hand scene in this runtime

**Files:** `tests/fixtures/mjcf/shadow_hand/**` (new), `crates/es-physics-backend/src/{mjcf_out,mapping,mujoco,mjwarp}.rs`
and their tests, `crates/es-assets/src/mjcf/**` only if a parsed element must be carried further,
`crates/es-render/src/scene.rs` (alpha), `docs/design/renderer.md` (+ko) if the alpha rule is
written there.

- **The derived scene** `tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.xml` with
  `meshes/*.stl` (the 12 upstream STLs; the ten with `scale="0.001"` pre-scaled, since this
  parser refuses `<mesh scale>`) and `PROVENANCE.json` + the gymnasium-robotics licence. Edits,
  each listed in the provenance: named skybox texture; single-value site sizes as three values;
  `<general>` servos as `<position kp kv>` (identical: `gainprm = kp`, `biasprm = 0 −kp −kv`); the
  five `<touch>` sensors dropped; **three cameras** framing the palm and cube (`top`, `front`,
  `side`; 96×96 at 50–60 Hz in the Task IR later); a `_light` emitter; a **face-coloured cube**
  (six visual-only slabs, a distinct colour each) because the renderer draws `rgba`, not
  textures — without it a cube's orientation is ambiguous to a camera modulo 90°; the goal cube
  (`target`) coloured the same, placed beside the hand where the cameras see it.
- **Physics**: the MJCF emitter writes **fixed tendons** (the four J1/J0 couplings), `<contact>
  <pair>` / `<exclude>` and `gravcomp` if `SceneDesc` carries them (carry them if the parser drops
  them), and the mapping report marks them native on `mujoco-cpu` and `mjwarp`. What still is not
  mapped is refused by name, as today.
- **Render**: a geom whose `rgba` alpha is 0 is not drawn (the hand's collision capsules and the
  invisible floor are alpha 0 in the bundle's materials); scenes without alpha-0 geoms render
  bit-identically (the render goldens do not move).
- **Oracle:** (1) `es backend compare --scene <derived> --backends mujoco-cpu,mjwarp` runs; (2) a
  test steps the derived scene through `MuJoCoCpuBackend` and through MuJoCo loading the same
  file directly, same controls, 240 ticks: qpos equal to 1e-9 (skips with a reason without
  `ES_PYTHON`); (3) each camera renders the reset pose on `Pt`, the cube's segmentation pixels
  > 0 in all three, and a raster-vs-CPU-reference check like the existing ones; (4) render and
  dataset goldens unchanged (`cargo xtask verify-goldens`).

### Task H2: the reorientation task and the teacher's documents

**Files:** new `tests/fixtures/shadow-hand/*.toml`, their generator (follow `crates/es/tests/views.rs`),
its tests; code only where a document cannot be expressed (then say so first).

- **Task IR** `task-repose.toml` (Isaac's `ShadowHandEnv` semantics, simplified where noted):
  control 60 Hz (physics 120 Hz), `JointPosition` over the 20 actuators; channels `joint_pos`,
  `joint_vel` (24), `cube_pose` (privileged), `goal_quat`, `last_action`, and the three images
  `rgb_top` / `rgb_front` / `rgb_side` (96×96, `render = { path = "pt", … }` at the X7 rerun's
  4 spp, `seed = "tick"`). With `d = |q_cube · q_goal|`: success `d ≥ cos 0.05` (0.1 rad);
  rewards `−10 · ‖p_cube − p_ref‖`, `1 / (√(8(1 − d)) + 0.1)` (the small-angle form of Isaac's
  `1 / (rot_dist + 0.1)`; there is no `acos` node), `−0.0002 · ‖a‖²`, a success bonus;
  failure when the cube is 0.24 m from `p_ref`; timeout 8 s. Reset: hand at the bundle's default
  pose with small joint noise, cube at `p_ref` with random **yaw**, goal a random **yaw** (the
  goal as `normalize(1, 0, 0, u)`, `u ~ U(−1, 1)`): a reduced goal set — rotations about the
  vertical — because full SO(3) goals are out of this PC's budget; the full set is a later row.
- **Teacher**: Observation IR (state only, normalized), Learning IR (MLP actor as the reach
  task's, wider), Deployment IR (envelope = joint and control ranges, velocity bound generous —
  written for the owner's review), `[rl]` recipe on `mjwarp` with as many envs as fit.
- **Oracle:** documents validate and cross-validate; the Task IR evaluated on scripted states
  gives the expected success / failure / rewards; `es train --dry-run`; a 5-iteration smoke
  (orchestrator) with the nine metrics — the throughput that sizes E1.

### E1 (orchestrator): the teacher

Train on `mjwarp`, watched in the editor; evaluate on `mujoco-cpu` (held-out seeds). The stop
rule is written before the run: the budget is set from H2's throughput.

### Task H3 + E2: the student

Observation IR per camera (`observation-views.toml`'s chain), joint state and `goal_quat`, no
`cube_pose`; Learning IR as `learning-views.toml` (or `learning-mad.toml`); Evaluation IR;
training recipe; cycle with `[collect] policy = teacher`. Collect the teacher's successful
episodes under the three `Pt` cameras, train, evaluate, all watched in the editor.
