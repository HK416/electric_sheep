# M5 V19 — LeRobot's own ACT on the current demonstrations, under the repaired harness

Design note: `docs/design/visible-learning.md` **section 7.27**, and 7.16 (V8, the same pipeline
on 50 demonstrations and the old harness), 7.22 (V14, the 200 demonstrations), 7.25 (V17, the
re-plan rate), open questions 22–25. Spec: §1.4, §5.3, §7, §8.1, §8.4, §8.7, §8.9, §9.2, §19.1,
§25.1. API notes: `docs/api-notes/lerobot-act.md`, `docs/api-notes/lerobot-dataset.md`.
Depends on V8 (the export, the import, the equivalence gate), V14 (the 200 demonstrations),
V17 (the cadence and the Deployment IR the evaluation runs under).

## the question

V8 asked whether a policy designed and trained **outside** this project runs here with identical
semantics, and answered yes for the semantics (bitwise) and 0/16, 0/16, 1/16 for the task. Every
one of those numbers was taken on a harness that has since been repaired three times:

* **V11** — the demo ran its control loop at 200 Hz against a 50 Hz Deployment IR;
* **V12** — the observation a policy acted on was paired with the *post*-step state;
* **V17** — the runtime re-planned every control tick instead of the declared 5 Hz, so every
  chunk row past row 0 was dead.

and V8's checkpoint was trained on V1c's 50 demonstrations at the old cadence, where V14 has 200.
So V8's verdict — "a real LeRobot ACT does no better, therefore the model is not the problem" —
rests on a measurement nothing in the current tree reproduces.

**V19 changes one thing against V17: the learner.** Same demonstrations (V14's 200), same Task
IR, same Deployment IR and Safety Plane, same seeds, same budget, same acceptance. The policy is
LeRobot 0.6.1's own ACT — CVAE, DETR encoder/decoder, ImageNet-pretrained ResNet18 — trained by
`lerobot-train` and imported through `es policy import-lerobot`.

## the decisions, and what each rests on

### 1. The state feature is V17's two state ports, concatenated

V17's best policy reads **two** state ports: the arm's six joint angles and V7a's
simulator-privileged `sim_cube_pose` (the cube free joint's `qpos[6..13]`). V8's ACT read only the
first, because `lerobot.utils.feature_utils.dataset_to_policy_features` gives a policy exactly one
`observation.state` feature and V8 was the *vision* question.

Keeping V8's six would have moved two variables at once — the learner **and** what it is allowed
to see. So the two ports are concatenated into the one feature LeRobot allows, in the order the
recorded `observation.state` row (`qpos || qvel`) already holds them:

* the export is `--state-dim 13`, which keeps `qpos[0..13]` — six joint angles then the cube's
  seven-value free-joint pose. The flag is V8's and its value is the only thing that changed;
* the Observation IR is observation-v8's graph plus observation.toml's cube branch, joined by an
  `ObservationNode::Concat` on axis 0, written by `python/es/make_v19_observation.py` from the two
  committed fixtures. Both branches' `Normalize` is the identity `Range{0..1}`, for V8's reason:
  ACT carries its own `MEAN_STD` statistics in the checkpoint and applies them as the first
  operation of its forward pass, and a second affine map here would have to be applied to the
  exported parquet as well. The `Concat` ports are `Frame::Policy` — an arm-frame `[6]` and a
  world-frame `[7]` have no common frame, and `Frame::Policy` is the one the IR already defines
  as compatible with any other, which is what a feature vector handed to a network is.

No `es-ir` change, no new node kind, no fixture edit: `Concat` is an `ObservationNode` variant
since M1 and `es-compile` lowers it to `Op::Join`.

### 2. Everything else is V17's, unchanged

`tests/fixtures/visible-learning/{task,deployment}.toml` are the files in the tree —
`max_episode_steps = 1800`, control 50 Hz, inference 5 Hz, `horizon = 16`, `execute_chunk = 10`,
`TemporalEnsemble { decay = 0.01 }`, the 240 ms inference budget V17 stated for the rate the same
document declares, and `success_rate >= 0.5` as the acceptance. The evaluation is
`evaluation-v8.toml` with its `observation` field pointed at the merged document and the seed list
trimmed to one nominal suite, by V14's own `mkcfg.py` — the same trim V11 – V17 used.

`--policy.chunk_size=16 --policy.n_action_steps=16` because the Deployment IR buffers a horizon of
16 and `es policy import-lerobot` refuses a checkpoint that disagrees (V8 §4). LeRobot's ACT
defaults for everything else, which is the point of the packet.

### 3. What V19 had to fix, and why it was invisible until now

`es policy import-lerobot` wrote the Deployment IR's `deadlines.inference_budget` into **both**
`RuntimeHints::deadline_ms` and `RuntimeHints::expected_latency_ms`. The second is documented in
`es_ir::learning::RuntimeHints` as "a measurement on the reference device of spec 0.3, not a
promise", and `LRN-052` checks `expected_latency_ms <= min(deadline_ms, 1000 / replanning_hz)`.
While the budget was V8's 40 ms against a 200 ms re-plan period the two readings coincided; V17
stated a **240 ms** budget for the same 5 Hz re-plan, and `LRN-052` then refused every imported
checkpoint for claiming a latency the deployment's own rate forbids.

The fix is one line plus the function that names it: the import declares the *bound* — the tighter
of the inference budget and the re-plan period — because a `config.json` carries no measurement.
V8's value is unchanged (`min(40, 200) = 40`), V17's deployment gives 200 ms, and the honest
consequence, stated in the code, is that `LRN-052` has nothing left to catch on this path: a rule
cannot check a number the same command made up. A measured latency would come from `es bench`.

## context

Allowed file scope:

```
crates/es/src/cmd/policy.rs               (declared_latency_ms + its unit test)
python/es/make_v19_observation.py         (the merged Observation IR)
docs/design/visible-learning{,.ko}.md     (section 7.27, open questions 22-25)
docs/packets/M5/V19-external-act-current-data{,.ko}.md
target/plan-v/v19/                        (untracked mirror of the server's results)
```

## spec

* **§8.1.** The network is opaque, the interface is typed. The bundle's `LearningGraph` is one
  `LearningNode::PolicyBundle` carrying the contract's ports, as V8 established.
* **§8.4 / `XIR-010`, `XIR-022`–`XIR-024`, `LRN-052`.** The contract is projected from
  `config.json` and completed from the Deployment IR; the one number the import *invents* is now
  declared as a bound and said to be one.
* **§8.7.** Pre/post-processing stays in the IR: the `Concat` is an Observation IR node, the
  chunker, the ensemble and the unnormalizer stay in the Deployment IR. Nothing moved into
  `PolicyRuntime`.
* **§8.9.** The equivalence gate is V8's, on a frame of this project's own dataset, at tier 4
  (<= 1e-5) — and measured at 0.
* **§5.3.** `task_hash`, `deployment_hash` and the dataset's content hash are V17's and V14's;
  `observation_hash`, `learning_hash` and `policy_hash` are this packet's and are recorded.
* **§7.** One Task IR carries several Observation IRs: this is the third
  (`observation.toml`, `observation-v8.toml`, the merged one).
* **`INV-16`.** safetensors only. **`INV-17`.** No new trait. **`INV-11`/`INV-12`/`INV-13`.**
  `es-safety` is not touched and no envelope number moved.

## oracle

### local, and these are the gate

```sh
cargo test -p es --bin es declared_latency
cargo xtask ci
```

### server — `~/Projects/es-v19`, artifacts under `~/artifacts/plan-v/v19/`

```sh
# 0. build
cargo build --release -p es --features render

# 1. the export: V14's 200 demonstrations, the two selections V8 explains, --state-dim 13
mkdir -p $A/frames-in && ln -s ~/artifacts/plan-v/v14/frames-train $A/frames-in/rgb_overhead
./target/release/es dataset export --lerobot-v3 ~/artifacts/plan-v/v14/ds-train \
    --out $A/ds-v3-s13 --frames $A/frames-in \
    --drop action_commanded,action_source,intervention --state-dim 13
# and the V8-comparable control, six joint angles and no cube pose
./target/release/es dataset export --lerobot-v3 ~/artifacts/plan-v/v14/ds-train \
    --out $A/ds-v3-s6 --frames $A/frames-in \
    --drop action_commanded,action_source,intervention --state-dim 6

# 2. the merged Observation IR, and the evaluation that names it
~/venvs/es-lerobot-cuda/bin/python python/es/make_v19_observation.py \
    tests/fixtures/visible-learning/observation.toml \
    tests/fixtures/visible-learning/observation-v8.toml $A/observation-v19.toml
./target/release/es ir validate $A/observation-v19.toml     # -> observation_hash
sed 's/^observation = ".*"/observation = "<that hash>"/' \
    tests/fixtures/visible-learning/evaluation-v8.toml > $A/evaluation-s13.toml
python mkcfg.py $A/evaluation-s13.toml $A/eval-s13-train.toml    1,...,16
python mkcfg.py $A/evaluation-s13.toml $A/eval-s13-holdout.toml  101,...,116

# 3. LeRobot's own trainer, LeRobot's own ACT defaults
~/venvs/es-lerobot-cuda/bin/lerobot-train \
    --dataset.repo_id=es/v19-so101-cube-s13 --dataset.root=$A/ds-v3-s13 \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=16 --policy.n_action_steps=16 --wandb.enable=false \
    --steps=100000 --batch_size=8 --seed=0 --save_freq=10000 --num_workers=8 \
    --output_dir=$A/train-s13 --job_name=v19-external-act-s13

# 4. the checkpoint becomes a bundle
./target/release/es policy import-lerobot \
    --checkpoint $A/train-s13/checkpoints/100000/pretrained_model \
    --task tests/fixtures/visible-learning/task.toml \
    --observation $A/observation-v19.toml \
    --deployment tests/fixtures/visible-learning/deployment.toml --out $A/s13-100000.esb

# 5. the equivalence gate, on a recorded frame of our own dataset
~/venvs/es-lerobot-cuda/bin/python $A/frame.py $A/ds-v3-s13 20 $A/frame-s13.json
ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python \
ES_ACT_CHECKPOINT=$A/train-s13/checkpoints/100000/pretrained_model \
ES_ACT_OBSERVATION=$A/frame-s13.json \
    cargo test --release -p es-policy --test act_checkpoint -- --nocapture

# 6. the evaluation the demo's acceptance judges
ES_PYTHON=~/venvs/es/bin/python ./target/release/es eval run \
    --config $A/eval-s13-holdout.toml --policy $A/s13-100000.esb \
    --scene tests/fixtures/mjcf/so101_pick_place.xml \
    --jobs 4 --out $A/e-s13-100000-holdout-a --frames $A/f-s13-100000-holdout-a
```

`~/artifacts/plan-v/v19/{pa,pb,pc,pd}.sh` are these steps as they were run;
`probe.py`, `terms.py` and `summary.py` are V17's read-out scripts.

## acceptance

1. The equivalence gate passes on the new checkpoints at spec §8.9 tier 4, with the measured
   `max_abs` / `max_rel` recorded. **Non-negotiable**: this is the thesis.
2. Per checkpoint: `success_rate`, cube lifted / carried / released in V17's per-episode style,
   `envelope_violation_rate`, `episode_length`, on training seeds 1–16 and held-out 101–116.
   The held-out suite is run twice at the best checkpoint and the two `report.json` files agree
   byte for byte.
3. If held-out `success_rate >= 0.5` at any checkpoint: the six-suite sweep, two showcase mp4s
   and a mosaic through the V9 pipeline. Below it the stop rule fires — no second variable.
4. `cargo xtask ci` green; every Rust change carries a test.

## forbidden

* **`es-safety`.** Not one line, not one number in `deployment.toml`. `INV-11`, `INV-12`,
  `INV-13` stand; the external policy runs behind the same envelope as the expert and every plan
  V policy.
* **`es-ir`.** No new node, no new field, no new `ArchKind`.
* **The acceptance threshold.** `success_rate >= 0.5` is not lowered to make a table pass.
* **The committed fixtures.** `task.toml`, `deployment.toml`, `observation.toml`,
  `observation-v8.toml` and `evaluation-v8.toml` are V17's and are not edited; the merged
  Observation IR is generated beside them, never in place of them.
* **Re-collection and re-baking.** V14's 200 demonstrations are the data; `es dataset bake` and
  `python/es/train_act.py` are not on this path at all.
* **`lower_act`, `lower_to_torch`.** The architecture rides in the checkpoint.

## as built

See section 7.27 of the design note. Filled in below after the run.
