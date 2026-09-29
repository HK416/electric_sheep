# M15 plan N — multi-camera policies (train with several views, deploy with fewer)

> Packets for `docs/design/multi-camera.md` (the owner's decisions are its section 6). The owner
> started the plan on 2026-09-29 ("시작"). Each task is a §1.2 packet: its **Files** block is its
> `context` for `cargo xtask check-scope`.

**Goal:** the SO-101 cube task seen by three cameras — wrist, overhead, side — through every path
(render, collect, bake, train, evaluate); then MAD's shared encoder with `Sum` fusion and
single-view training, and a policy trained on three views deployed on one. Each step measured.

## Global Constraints

- Everything in `docs/packets/M14/plan-q.md`'s Global Constraints still holds (MSRV 1.85, egui
  0.32.3, goldens by generator only, no trailers, worktree + `--ff-only`, never push, no remote
  server, no learning run inside an agent packet, decisions in `es-editor-model`).
- **One `CARGO_TARGET_DIR` per concurrent agent** (`target/wt-<packet>`), deleted after merge; the
  orchestrator runs the wave's `cargo xtask ci` only when no agent builds.
- **Every committed document and number stays as it is.** Several cameras live in new documents
  with new hashes; a single-camera Task IR renders, collects and bakes byte-identically to today
  (the flat frame layout included) — existing goldens and dataset content hashes must not move.
- **Spec first for IR changes** (N5): `docs/ARCHITECTURE.ko.md` (canonical) and `ARCHITECTURE.md`
  in the same commit; a new optional IR field is absent from the canonical encoding when unset, so
  no committed document's hash moves (tested against the committed documents' hashes).
- **Safety Plane untouched**; a missing camera at runtime stays a `SensorDropout` watchdog.
- **Oracles (§1.4):** renders are bitwise against single-camera renders; Learning IR lowerings
  against hand-written PyTorch modules; a subset bundle against the full graph with the dropped
  terms removed.

## Waves

| Wave | Tasks | Needs |
|---|---|---|
| 1 | N1 the three-view scene and documents · N2 several cameras per env (render, frames, eval) | — |
| 2 | N3 bake and export per camera | N2 |
| 3 | E1 experiment 1: one view against three (orchestrator) | N1–N3 |
| 4 | N5 `Sum` and `share` in the IR and the spec · N6 their lowering · N7 single-view training | — (N6 after N5) |
| 5 | N8 `es policy subset` | N5, N6 |
| 6 | E2 experiment 2: MAD, deployed on fewer views (orchestrator) | N5–N8 |
| 7 | N9 the editor's camera option | E1 |

---

### Task N1: the three-view scene and its documents

**Files:** `tests/fixtures/mjcf/so101_pick_place_views.xml` (+ its `.PROVENANCE.json` if the
existing scene has one), new `tests/fixtures/visible-learning/*-views*.toml` and
`*-cam*.toml` documents, the generator that writes them (find how `task-pt.toml` etc. were made and
follow it), their tests.

- The scene: today's `so101_pick_place.xml` plus a **wrist camera** inside `camera_mount` (looking
  down the gripper at the jaws) and a **side camera** (world-fixed, the workspace and the bin in
  view). The overhead camera, every body, geom and joint unchanged.
- Documents, each a new file: a Task IR with three image channels (`rgb_overhead`, `rgb_wrist`,
  `rgb_side`, 96×96 RGB like today) pinning the new scene; for experiment 1 a camera-only pair on
  the IR route — one view (overhead) and three views — each an Observation IR (the
  `observation-augmented.toml` chain per camera, `joint_state`, **no** `sim_cube_pose`), a Learning
  IR (U3's `learning-pretrained.toml` graph: one ResNet18 per view, `Concat` fusion), an
  Evaluation IR (the suites, seeds, metrics and acceptance of `evaluation-augmented.toml`, only
  the references differ), a deployment (today's), a training recipe (U3's settings) and a cycle.
- **Oracle:** the documents validate and cross-validate (XIR); a render of each camera at the reset
  pose shows the cube (a count of cube-coloured pixels above a floor, per view, over a few pinned
  cube positions) and the wrist view moves with the gripper (two arm poses, two different
  images); committed documents' hashes unchanged.

### Task N2: several cameras per env

**Files:** `crates/es-env/src/render.rs`, `crates/es-tools/src/lib.rs`, `crates/es/src/cmd/{loop,eval}.rs`,
`crates/es-eval/src/runner.rs`, `crates/es-data/src/collect.rs` (frame sink only), their tests,
`tests/golden/**` (generator only).

- `EnvRenderer` renders every image channel of the Task IR per step (one atlas tile per camera,
  §15.2); a single-channel Task IR renders exactly as today.
- `--frames` with several channels writes `<frames>/<channel>/<NNNNNN>.bin` (+ `.json`), the layout
  the LeRobot v3 export already expects; one channel keeps today's flat layout byte for byte.
- The evaluation runner and `es loop collect` feed each image port its own camera's frame.
- **Oracle:** each camera's tile is bitwise its single render (as M11/X3b pinned for envs);
  a two-camera test scene collects and evaluates with both ports fed, and two different frames
  (not one reused); every existing frames golden and dataset content unchanged.

### Task N3: bake and export per camera

**Files:** `crates/es/src/cmd/{dataset,train}.rs`, `crates/es-data/src/{training,lerobot/**}.rs`,
their tests.

`es dataset bake` reads each image port from its own camera directory; `mirror_frames` (which
would hard-link one directory into every camera's) is replaced by the real per-camera layout.
**Oracle:** a three-camera fixture bakes three different image tensors; single-camera bakes
unchanged.

### Task E1: experiment 1 (orchestrator, a quiet PC, the editor open)

200 expert demonstrations rendered by three cameras (one collection serves both arms); the
camera-only IR route trained on the overhead view alone and on all three, two training seeds each,
evaluated by the same suites. Recorded in `docs/packets/M15/NV-verification.md`.

### Task N5: `Sum` and `share` in the IR and the spec

**Files:** `crates/es-ir/src/learning.rs`, `docs/ARCHITECTURE.ko.md`, `docs/ARCHITECTURE.md` (§8.3),
their tests.

`FusionKind::Sum` (inputs of equal width; the output their sum). `VisionEncoder.share:
Option<NodeId>` — the encoder whose weights this one uses; refused by name when it names a node
that is not a `VisionEncoder`, one of another backbone, width, token count or pretrained source,
itself, or one that itself shares (no chains). Absent fields are absent from the canonical
encoding: every committed Learning IR hash stays.

### Task N6: their lowering

**Files:** `crates/es-policy/src/lower/torch.rs`, `python/es/builder.py` (the `shared` argument now
writes `share`), tests.

`Sum` → `torch.stack(inputs).sum(0)`; encoders that share become one module applied per view.
**Oracle:** the lowered module equals a hand-written PyTorch module on fixed inputs (Python-gated);
the weight file holds one copy of a shared encoder.

### Task N7: single-view training

**Files:** `python/es/train_act.py`, `crates/es-data/src/training.rs` (`[run] single_view`), tests.

`[run] single_view = { weight = <alpha> }`: for a graph whose views meet in a `Sum` fusion, each
batch also computes the loss with each view's term alone, weighted by `alpha` (MAD's
"disentangle"). Refused for a graph with no `Sum` over image features.

### Task N8: `es policy subset`

**Files:** `crates/es/src/cmd/policy.rs`, `crates/es-policy/**` (the bundle rewrite), tests.

`es policy subset --policy <bundle> --views <channel,...> --out <bundle>`: the same weights; the
Observation IR and Learning IR without the dropped ports, encoders and `Sum` terms (a kept encoder
that shared a dropped one takes its weights under its own name); new hashes. **Oracle:** the
subset bundle's output equals the full graph with the dropped terms removed, in PyTorch; refused
for a graph whose views do not meet in a `Sum`.

### Task E2: experiment 2 (orchestrator)

The three-view MAD policy (shared encoder, `Sum`, single-view training) evaluated with all three
cameras, with each alone and with the wrist alone as the real-robot case, against E1's policies.

### Task N9: the editor's camera option

A template option "cameras: overhead / overhead + wrist + side", and ⑤'s player showing the views
side by side. Planned after E1 says whether the option is worth offering.
