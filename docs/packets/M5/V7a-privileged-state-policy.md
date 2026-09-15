# M5 V7a — a sim-privileged state policy

Design note: `docs/design/visible-learning.md` **section 7.14**, and sections 7.5–7.13 plus open
questions 11–13 for how it got here. Spec: §5.1, §6.2, §7.2, §7.4, §8.2, §19.2. Read those
before this file. Depends on V1 (the expert and its dataset), V2/V2b (training and the bake), V3
(the suite), V5 (`--resident-gpu`), V6 and V6b (the harness that now passes the expert and draws
the seed's own scene).

## the question this packet exists to answer

**Can the demo's policy do the task at all, given an observation that contains the answer?**

The demo's vision ACT — 50 demonstrations, one 96×96 overhead camera, chunk 10, ResNet18 from
scratch, 20,000 steps — has never produced a working policy, and every evaluation number taken
before V6/V6b was measured on a harness that was failing the *scripted expert* (section 7.12) or
executing row 0 of every chunk (section 7.13). Those numbers are gone; nothing carries forward
from sections 7.8–7.11.

So the next measurement has to separate two questions that the vision demo asks at once:

1. can this graph, this envelope, this expert and this training loop produce a policy that puts
   the cube in the bin **when it is told where the cube is**, and
2. can a ResNet18 trained from scratch on 50 episodes **find** a 25 mm cube in a 96×96 frame.

V7a is question 1, and it is stage 1 of the video ("state policy"). Question 2 is a separate
packet — higher resolution, more demonstrations — and it is not this one.

**Stop rule, and it is part of the packet.** If the state policy does **not** reach
`success_rate ≥ 0.5` nominal on seeds 101–116, the next step is **not** a bigger model, a longer
schedule or more demonstrations. It is to suspect the physics, the contact model or the expert's
trajectories: a policy given the cube's exact pose, the arm's exact joint angles and 50
demonstrations of a 7-second scripted motion has no information left to lack. Report the failure
and open the scene, not the optimizer.

## the decisions, and what each rests on

### 1. Where the cube's pose comes from

**A second `ObservationSpec` channel, `sim_cube_pose`, `ObsSource::JointState { body: <the
`cube_free` joint's `StableId`>, dof: 7 }`, typed `f32[7]`.** The Task IR *declares* it and
implements nothing (§5.1, rule 6); the Observation IR normalizes it; the Learning IR encodes it.
No `es-ir` change, no new node variant, no new field.

Three findings decided the shape, and each is a constraint the repo already had:

* **`ObservationSpec` can carry a seven-wide free joint, and the resolution path already
  exists.** `es_eval::runner::input_sources` resolves a plan input's source `StableId` against
  `ModelInfo.qpos` **before** it falls back to the Task IR channel's `dof` (`crates/es-eval/
  src/runner.rs`). `ModelInfo.qpos` is keyed by **joint** id, so a channel whose source is the
  `cube_free` joint is served `Capture::Qpos(6..13)` — the joint's own seven `qpos` values,
  exactly. A channel whose source is a *body* (which is what `joint_state` is, the robot's
  `base`) falls through to `Capture::Joints(dof)`, "the leading `dof` of the row". Both are
  pre-existing arms; V7a adds neither.
* **Task IR-D cannot emit a free joint's `qpos` from `GetJointState`.** That node's output type
  is `f32[joints.len()]` — one scalar per joint *name* — which is the ceiling section 5.4 of the
  design note already recorded, and it is why the success predicate sees the cube's `x` alone.
  The graph therefore *shows* the value as `GetBodyPose { body: cube, relative_to: World }` →
  `Concat{axis 0}` of its `pos[3]` and `quat[4]` → `ObservationSpec`, which is expressible in
  the existing node set and type-checks edge for edge. The **channel** names the free joint
  rather than the body, because the channel is what decides the *reading*: through the joint it
  is `qpos`, exact; through the body it would be `xpos ‖ xquat`, which the recorded dataset does
  not carry.
* **`Normalize{Range}` is one `(lo, hi)` per port**, so the privileged values need a port of
  their own — which is the second reason not to widen `joint_state` from 6 to 13. At the state
  branch's `±1` the cube's 0.06 m draw spans 0.03 of the output range; at `±0.3` — the arm's
  reach, which contains both the draw (`x ∈ [0.21, 0.27]`) and the bin's interior
  (`x ∈ [0.09, 0.19]`, `y ∈ [-0.15, -0.05]`) — it spans 0.10. The quaternion's four values leave
  `[0, 1]` under that range (`w = 1` maps to 2.17); a `Normalize` is affine and not a clamp, and
  a box resting flat carries no signal there, so this is recorded rather than worked around.

### 2. Dataset — nothing to collect, and one thing to prove

**`observation.state` already carries the cube's pose.** `es_data::collect::to_lerobot` writes
the row as env 0's whole `qpos` followed by its whole `qvel` — `nq + nv = 13 + 12 = 25` for this
scene, with the cube's free joint at `qpos[6..13]`. No column is added, the v2.1 writer and the
v3.0 export are untouched, and **`dataset_schema_hash` does not move**. What moved is which
slices of the row the Observation IR reads. `docs/api-notes/lerobot-dataset.md` now records the
layout and why it is load-bearing.

That layout is also what makes the bake exact: a `qpos` `IndexRange` indexes the recorded row
with the same two bounds `capture` indexes `StateView::qpos_of(0)` with. The row is not a
re-encoding of the state — it is the state with `qvel` appended. `es_eval::ObservationBake`
therefore gains one arm (`Capture::Qpos(r) => row[r]`) and `ObservationBake::new` gains the
`Option<&ModelInfo>` it was denied in V2b, because a `qpos` range has to come from the model
that ran.

**One refusal comes with it, and it is the important half.** Model-free, `Capture::Joints(dof)`
means "the leading `dof` of the row", and only *one* channel can be leading. With a second
`JointState` channel declared and no model, which one that is is not in the documents — it is in
the model — so `input_sources` refuses by name instead of serving one channel the other's
values. Without that refusal the demo would have baked `row[..7]` (six arm angles and the cube's
`x`) into the privileged port and trained on it silently.

### 3. Marking privilege

**`sim_cube_pose` is simulator-privileged: a real SO-101 has no sensor that reports where the
cube is.** `es_ir::task::ObsChannel` carries `source` and `ty` and nothing else — §7.4 is
explicit that the rest is Observation IR's — so **there is no field to tag and none was
invented**. The mark is the `sim_` prefix, and it is carried unchanged by the Task IR channel,
the Observation IR output port, the Learning IR input and the policy contract, so it is visible
at every hop of the chain and in `contract.json`. The convention is documented here, in the
fixture headers and in design note section 7.14.

`sim_cube_pose`, not `sim.cube_pose`: the port name reaches Python as a key of `forward(**inputs)`
in the lowered module, and as a LeRobot feature name a dot is a namespace separator.

Keeping it a *second* port rather than a wider `joint_state` is what lets the vision packet drop
it again without moving `joint_state`'s hash — the arm's six angles come off real encoders and
stay exactly as they were.

### 4. Training knobs for the server phase

Unchanged from V2b, on purpose: the observation is the one variable that moves.

* 50 demonstrations, the same dataset and tiles — a state policy needs no more, and re-collecting
  would move `dataset_schema_hash` and confound the comparison.
* `execute_chunk = 10`, `horizon = 16`, `replan_hz = 5`, `TemporalEnsemble { decay = 0.01 }` —
  all as declared. V6b made evaluation honour them; changing them now would make V7a a
  measurement of two things.
* 20,000 optimizer steps, `--batch 8 --lr 1e-4 --seed 0`, checkpoints at 1k / 5k / 20k, with
  `--resident-gpu` (V5).
* Nominal evaluation on seeds 101–116, the same held-out sixteen; the full suite on the final
  checkpoint only; videos as V3 and V1c produced them.

## context

Allowed file scope:

```
tests/fixtures/visible-learning/{task,observation,learning,evaluation}.toml
crates/es/tests/cli.rs                 (the fixture generator, and the bake CLI tests)
crates/es/src/cmd/dataset.rs           (`es dataset bake` resolves the model when it must)
crates/es-eval/src/bake.rs             (the `qpos` arm, and the `ModelInfo` parameter)
crates/es-eval/src/runner.rs           (`input_sources`: the ambiguity refusal)
crates/es-eval/tests/evaluation.rs     (the bit-identity oracle, two state channels)
docs/api-notes/lerobot-dataset{,.ko}.md
docs/design/visible-learning{,.ko}.md  (section 7.14, open questions)
docs/packets/M5/V7a-privileged-state-policy{,.ko}.md
```

`python/es/train_act.py` is **not** in scope and needs no change: it reads the baked tensors and
`contract.json`'s `inputs`, so a new port arrives on its own.

## spec

* **§5.1, rule 6 — Task IR declares, Observation IR implements.** `sim_cube_pose` is one
  `ObsChannel { source, ty }` and one `TaskNode::ObservationSpec`. Every conversion — the
  `Normalize`, the range, the layout — is in `observation.toml`. No neural net enters the Task
  IR (`TaskNode` has no such variant).
* **§7.4 / `XIR-002`.** The Observation IR's `StateInput { source }` must key-match a declared
  channel and its output type must be compatible with the declared one. Both are checked by
  `es ir check` on the committed fixtures.
* **§7.2 / `INV-14`.** No `Resize` and no `Crop` is added, so no intrinsics transform is owed;
  the image branch is untouched, byte for byte.
* **§8.2.** The privileged port gets its own `StateEncoder { Mlp[256], out_dim 512 }` and a third
  input on `Fusion { Concat, out_dim 512 }`, which the lowering already supports for `n` inputs.
* **`INV-16`.** Weights stay `safetensors`; nothing about the weight path moves.
* **`INV-17`.** No new trait. `Capture` is an `es-eval` enum, not an extension point.
* **§19.2.** The dataset format does not move. `dataset_schema_hash` is unchanged.

## oracle

### local, and these are the gate

```
cargo run -p es -- ir check tests/fixtures/visible-learning/task.toml \
    tests/fixtures/visible-learning/observation.toml \
    tests/fixtures/visible-learning/learning.toml \
    tests/fixtures/visible-learning/deployment.toml \
    tests/fixtures/visible-learning/evaluation.toml
cargo test -p es-eval --test evaluation -- --nocapture bake state_channels
cargo test -p es --test cli -- --nocapture policy_lower dataset_bake
cargo test -p es -p es-eval -p es-data -p es-env -p es-policy
cargo xtask ci
```

* `a_baked_frame_is_bit_identical_to_what_capture_serves` (`es-eval`) now runs **two** state
  ports: `j0` through the Task IR channel's leading-`dof` reading, and `j1` — whose `qpos` range
  starts at **1** — through `Capture::Qpos`. Every output tensor of every frame must be
  byte-equal to what the policy was served, with no tolerance. The `j1` branch is the demo's
  cube at the smallest size that has the property that matters: a bake reading "the leading
  value" would serve `q[0]` there, and the test asserts `q[0] ≠ q[1]` so it cannot pass by
  coincidence.
* `a_bake_refuses_two_state_channels_without_a_model` (`es-eval`) pins the refusal in §2 above,
  and that the same pair resolves once the model is handed over.
* `policy_lower_writes_the_module_and_contract` (`es`) lowers the widened graph and prints the
  `lowering_hash` and the port list, because no golden pins a lowering.
* The two `es dataset bake` CLI tests now need `MuJoCoCpuBackend` — the demo's second channel is
  resolved against the scene's `qpos` ranges — and print a skip reason without it. Their fixture
  row also becomes the real `qpos ‖ qvel` width (25) instead of the six values no run of
  `es loop collect` has ever written.
* `dataset_bake_names_both_refusals_when_the_scene_cannot_be_loaded` (`es`) is the coverage that
  replaces them where the backend is missing, which is PR CI: it drives the CLI's whole fallback
  — model-free resolution refused, scene load attempted, both halves named in one message — and
  steps aside where the backend is present, because the two tests above cover the success there.
  The one step it cannot reach on a machine without `mujoco` is the successful model load, and
  the first command of the server run is exactly that.

### server, phase 2 — run in this order, nothing else moves

```sh
cd ~/electric-sheep && git fetch && git checkout <branch>
. ~/venvs/es/bin/activate.sh 2>/dev/null || true

# 0. the documents, then the bundle
cargo run --release -p es -- ir check tests/fixtures/visible-learning/*.toml

# 1. bake the existing 50-episode dataset through the widened Observation IR.
#    This is the first command that exercises `es dataset bake`'s model load.
cargo run --release -p es -- dataset bake \
    --policy ~/artifacts/plan-v/demo.esb \
    --frames ~/artifacts/plan-v/tiles \
    --out ~/artifacts/plan-v/baked-v7a \
    ~/artifacts/plan-v/dataset

# 2. lower and train, V2b's knobs exactly, one variable moved
cargo run --release -p es -- policy lower \
    --policy ~/artifacts/plan-v/demo.esb --out ~/artifacts/plan-v/module-v7a
~/venvs/es-lerobot-cuda/bin/python python/es/train_act.py \
    --module ~/artifacts/plan-v/module-v7a \
    --baked ~/artifacts/plan-v/baked-v7a \
    --steps 20000 --batch 8 --lr 1e-4 --seed 0 --resident-gpu \
    --checkpoint-at 1000 --checkpoint-at 5000 --checkpoint-at 20000 \
    --loss-curve ~/artifacts/plan-v/loss-v7a.json \
    --out ~/artifacts/plan-v/model-v7a.safetensors

# 3. pack, then the nominal sweep on the three checkpoints
cargo run --release -p es -- policy pack \
    --policy ~/artifacts/plan-v/demo.esb \
    --weights ~/artifacts/plan-v/model-v7a-20000.safetensors \
    --out ~/artifacts/plan-v/trained-v7a-20000.esb
cargo run --release -p es -- eval run \
    --policy ~/artifacts/plan-v/trained-v7a-20000.esb \
    --evaluation tests/fixtures/visible-learning/evaluation.toml \
    --suite nominal --jobs 6 --out ~/artifacts/plan-v/eval-v7a-20000

# 4. the full suite, on the final checkpoint only
cargo run --release -p es -- eval run \
    --policy ~/artifacts/plan-v/trained-v7a-20000.esb \
    --evaluation tests/fixtures/visible-learning/evaluation.toml \
    --jobs 6 --frames ~/artifacts/plan-v/frames-v7a \
    --out ~/artifacts/plan-v/suite-v7a

# 5. the video, as V3 and V1c produced it
cargo run --release -p es -- video mosaic --grid 4x4 \
    --frames ~/artifacts/plan-v/frames-v7a --out ~/artifacts/plan-v/mosaic-v7a
~/venvs/es/bin/python python/es/encode_video.py \
    --frames ~/artifacts/plan-v/mosaic-v7a --fps 50 --out ~/artifacts/plan-v/demo-v7a.mp4
```

Every checkpoint's nominal cell and the final suite go into design note section 7.14, beside the
loss curve. **No number is lowered to make a threshold pass** (`evaluation.toml` asks for
`success_rate ≥ 0.5` and stays asking).

## acceptance

1. All five fixtures validate and cross-check through `es ir check`, and every hash that moved is
   recorded here and in section 7.14:

   | slot | before (V6b) | after (V7a) |
   |---|---|---|
   | `task_hash` | `aec2aea9e04cb0f4b224a49da120a1a39cd04499a6b929ea0bf0c2ccfaf51bf1` | `6cf826c167fef258414ce409ff51e310a1240fdb4b42b510351d5bae3e3f6b7b` |
   | `observation_hash` | `f4a50730ac95b91734c9678e75d9e6bc1845578bc2985e45b409e80f3355f6e0` | `6c18f4552064d24b16cffa770cfc71941566f2b0443fe88569d439f747c72daf` |
   | `learning_hash` | `82faf8c70c804b6fc104f436ad3327e90dd684280f7a7b7c3620797086971b54` | `5dac0a46f56198b1a6d04ead8ee43b18913356c56ee99b01a35decc66dc446f0` |
   | `policy_hash` | `01583940a350cd11a7ae0f304d5e8710db0494c210d904ebb62f1f0d2e4b88cd` | `c94c2732e215e4d8ac39cc0bd280c2d5495e5e36dbbc2da39f41ac952bd84a07` |
   | `evaluation_hash` | `5d70c21c0baa68f3aa9ccb64b2208823ba60e9ba47b16ffb7bc50f32313b9ff7` | `0259fd44ecc87c0e947afb98872984c0607efaf328c775d7f52f8ed00dcf041e` |
   | `lowering_hash` | `956abb67…ec6d` (V2b) | `fdd68ec43a7487dc2eefb373af669a2b6da2b692ff6729c9801242bced840719` |
   | `deployment_hash` | `3b2ad56806899d9d7f381bcfba03e65fdfd1fad0067b2a646f0f371bdf6dbb21` | **unchanged** |
   | `compiler_hash` | `f2a02e84dbc5d186b9c809d9938ab70df70bcfb7e2e95fbc2344ddf1137270d6` | **unchanged** |
   | `dataset_schema_hash` | — | **unchanged**: no column was added |

   Every one of them is derived — `task.toml`, `observation.toml` and `evaluation.toml` come out
   of `cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`, the
   declared generator, and no hash in them is typed in.
2. The bit-identity oracle passes with three ports and both non-vacuity assertions fire.
3. `cargo fmt --check`, `clippy -D warnings` on the touched crates under both feature sets,
   `cargo xtask check-spec-refs`, `cargo xtask context-budget`, `cargo xtask ci`.
4. Phase 2: the tables above are filled from the server run, and the stop rule is applied as
   written — including when it says to stop.

## forbidden

* **`es-safety`.** Not one line. The envelope is V6's and no number in `deployment.toml` moves;
  `deployment_hash` is in the table above precisely so that it can be seen not to have moved.
  `INV-11`, `INV-12`, `INV-13` all stand.
* **`es-ir`.** No new `ObservationNode`, no new `TaskNode`, no new `ObsSource` variant, no new
  field on `ObsChannel`. The crate is at its §1.5 target; if the existing nodes could not express
  the channel the packet was to stop and report, and they could.
* **`es-policy` lowering.** `lower_to_torch` is not touched. The widened graph is lowered by the
  `Fusion{Concat}` arm that already handles `n` inputs.
* **The image branch.** `observation.toml`'s `ImageInput → Dequantize → Normalize` chain and its
  `ImageSpec` are unchanged, so the vision packet inherits them intact.
* **The dataset format.** No new column, no writer change, no exporter change.
* **`python/es/train_act.py`.** It reads ports; it does not know their names.

## as measured (phase 2, 2026-09-15)

Design note `docs/design/visible-learning.md` section 7.15 is the record; this is the packet's
verdict. Server tree of `481e4d4`, `~/artifacts/plan-v/v7a/`.

| bundle | nominal `success_rate` | `envelope_violation_rate` | suite (96 episodes) |
|---|---|---|---|
| 1,000 steps | 0/16 | 0.9853 | — |
| 5,000 steps | 1/16 | 0.2523 | — |
| 20,000 steps | 0/16 (1/16 sharded) | 0.1460 (0.1628) | 4/96 |

- Every hash in the acceptance table came back as predicted; `deployment_hash` did not move.
- The training loss with the cube's exact pose in the input is within half a percent of the
  vision-only run at every checkpoint (`0.0175` vs `0.0176` at 20,000): the L1 objective barely
  distinguishes a policy that knows where the cube is from one that does not.
- **Stop rule: fired.** The acceptance of `success_rate >= 0.5` is not met and was not lowered.
  The next suspects are the physics, the contact model and the expert's trajectories (section
  7.15, finding 8), not model size or schedule length. Stage 2 (vision at a higher resolution) is
  not the next packet; `docs/ARCHITECTURE.ko.md` section 28.9 puts the external ACT (V8) before
  that investigation as the control experiment.
- Two findings outside the verdict: `--jobs N` is not byte-identical to `--jobs 1` with a torch
  runtime (open question 16), and `cv2` lives in `~/venvs/es-lerobot`, not `~/venvs/es`, so the
  `encode_video.py` line of the server phase above names the wrong interpreter.
- Regression found after the run and fixed on main (`9087785`): with two `JointState` channels
  `es dataset bake` opens the Task IR's repository-relative `scene.path`, which only resolves
  from the repository root, so the two model-backed bake oracles in `crates/es/tests/cli.rs`
  failed wherever `mujoco` is present. `es dataset bake --scene <file.xml>` now names the scene
  the way `es eval run --scene` does, and the oracles pass it.
