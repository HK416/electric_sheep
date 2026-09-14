# M5 V2b — the training set goes through the Observation IR

Design note: `docs/design/visible-learning.md` section 7.6 finding 3, section 7.8 finding "the most
likely cause", and open question 11. Read them first. Depends on V1 (the dataset and its tiles), V2
(the lowering, the training script, the bundle) and V3 (the measurement this packet re-runs). It fixes
a defect, and the fix is the "real fix" finding 3 already named: **an `es` step that bakes the
observation plan over a dataset**.

## the defect

`python/es/train_act.py` feeds the graph's state port the **raw** `observation.state` row and
re-implements exactly one Observation IR node (`Op::Dequantize`) for the image. At inference
`es eval run`'s `capture` runs the Observation IR's compiled plan, and the demo's plan is
`StateInput -> Normalize{Range −1..1}` and `ImageInput -> Dequantize -> Normalize{Range 0..1}`.
The image branch is safe by accident (`normalize_range(x, 0, 1)` is the identity); the state branch
is not — at evaluation the policy receives `(q + 1) / 2` and in training it received `q`. Section
7.8's table is therefore not a measurement of ACT: 2/16 nominal, `envelope_violation_rate` exactly
`1.0`, and **not one step in 85,407** classified `ActionSource::Policy`.

Two implementations of one node is the root cause, not the symptom. The symptom is one missing
`Normalize`; the fix is that there is only ever one implementation, because training reads what the
inference executor wrote.

## context

```
crates/es-eval/src/bake.rs
crates/es-eval/src/runner.rs
crates/es-eval/src/lib.rs
crates/es-eval/tests/evaluation.rs
crates/es/src/cmd/dataset.rs
crates/es/tests/cli.rs
crates/es-policy/src/lower/torch.rs
crates/es-policy/tests/ir_training.rs
python/es/train_act.py
tests/fixtures/visible-learning/learning.toml
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V2b-observation-bake.md
docs/packets/M5/V2b-observation-bake.ko.md
```

Notes: the bake's core lives in `es-eval` **next to `capture`** so that `input_sources`, the
`Capture` enum and the `f64 -> f32` input encoding have exactly one implementation; `es-data` is
layer 10 and so is `es-eval`, so the dataset reading and the safetensors writing stay in `es`
(layer 12), which already depends on both. No new crate, no new dependency.

## spec

- §7.2: the Observation IR **owns** preprocessing. A second implementation of a node — in Python or
  anywhere else — is the thing this packet removes. INV-14 is untouched: the demo has no `Resize`
  and no `Crop`, so no intrinsics transform is owed, and the bake performs no conversion of its own.
- §7.5: the observation stream ends with the episode. The bake resets the plan per episode, the
  same `plan.reset()` `run_episode` does, so a `TemporalWindow` cannot read across an episode
  boundary in training either.
- §2.3: the split stays where it is. The optimizer is Python's; the plan runner is Rust's. This
  packet moves the one node that had leaked across the line back to the Rust side.
- §1.4: "the same IR run in PyTorch is the ground truth" stays literally true — the module is still
  the lowering's, and now the *input* is the executor's too.
- §5.3, §19.2, §19.3: the bake records `observation_hash`, `task_hash`, the compiler hash and the
  dataset's `content` hash in its manifest, so a baked set names the documents it came from. It
  populates no new hash slot and changes no existing one.
- §8.9: the round-trip equivalence check keeps its tier-4 fp32 tolerance; nothing here loosens it.
- INV-16: safetensors out, safetensors in. `train_act.py` gains a header parser (`struct` + `json`,
  15 lines) and no package; it loses `pyarrow`.
- INV-17: no new trait. The bake is a struct with three methods.
- §12.4: the headline this packet re-measures is a success rate.
- §1.5: `es-eval` is at 2,489 code lines and `es` at 4,066; this packet is budgeted under ~300
  across both.

**Open question 3 of V2 ("pretrained backbone") is answered here: rejected.** `lower_to_torch`
ignored `VisionEncoder{pretrained}`, which is a silent divergence between what the IR declares and
what runs. Honouring it is `weights="DEFAULT"` in `_backbone`, one line — and it is the wrong line:
it makes `EsPolicy()` fetch ImageNet weights from the network at **every** construction, including
inside `TorchRuntime::load` at inference, where `load_state_dict(strict=True)` immediately
overwrites every one of them. A lowering that needs the network to instantiate contradicts §2.5.
So `pretrained = true` becomes `LowerError::Unsupported` naming the flag and both ways out, and
the demo's `learning.toml` says `pretrained = false`, which is what V2 measured it doing anyway
(design note 7.6, "open question 6 is answered"). `learning_hash` and `policy_hash` move; `task_hash`
and `observation_hash` do not, so the baked set and `evaluation.toml` are unaffected.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es -p es-policy --all-targets -- -D warnings
cargo clippy -p es --features render --all-targets -- -D warnings
cargo test -p es-eval
cargo test -p es-policy
cargo test -p es --test cli dataset_bake
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
cargo xtask ci
```

**The oracle of this packet is bit identity**, and it runs in the PR tier with no Python, no
physics backend and no GPU:

- `crates/es-eval/tests/evaluation.rs`: **`a_baked_frame_is_bit_identical_to_what_capture_serves`**.
  One evaluation runs through `Evaluation::run_with_frames` over a demo-shaped Observation IR
  (`StateInput -> Normalize{−1..1}` **and** `ImageInput -> Dequantize -> Normalize{0..1}`), with a
  frame source that records the `qpos` row it was called with alongside the tile it served, and a
  `PolicyRuntime` that records every observation map it was handed. The same rows and the same
  tiles then go through `ObservationBake`, and **every** output tensor of **every** frame must be
  byte-equal — `dtype`, `shape` and `data`. Not "close", not "for frame 0": equal, for the whole
  run. This is the test that fails if the bake and `capture` ever stop being the same code, and it
  is the test that would have failed before this packet.
- `crates/es-eval/tests/evaluation.rs`: `a_bake_refuses_an_input_the_dataset_cannot_feed` — a
  `StateInput` naming a joint resolves through `ModelInfo` at inference and has no reading in a
  recorded dataset, so `ObservationBake::new` refuses it by name at construction rather than
  guessing an offset into `observation.state` mid-run.
- `crates/es-policy/tests/ir_training.rs`: `train_act_defines_no_layer` gains `permute(`, `/ 255`
  and `pyarrow` to its forbidden list. The hand copy of `Op::Dequantize` cannot come back without
  this test failing, and it needs no interpreter to say so.
- `crates/es/tests/cli.rs`: `dataset_bake_writes_safetensors_and_a_manifest` — a three-frame
  dataset written with `es-data`'s own writer, baked, and read back: one safetensors per episode
  carrying `[frames, ...]`-shaped tensors under the Observation IR's own output names plus
  `action`, and a `manifest.json` whose `observation_hash` is the bundle's.
  `dataset_bake_without_frames_refuses_an_image_observation` — exit 1 naming the port, never a run
  that bakes zeros into the image channel.
- `crates/es-policy/src/lower/torch.rs`: `a_pretrained_vision_encoder_is_refused_not_ignored`.

Reference — the oracle server (`ES_PYTHON`, `~/venvs/es-lerobot-cuda/bin/python`):

```
cargo test -p es-policy --test ir_training -- --ignored --nocapture
```

`act_training_uses_baked_observations` (the rewiring of V2's
`the_loss_falls_and_the_packed_bundle_round_trips`) runs the whole loop on **baked** input:
`es policy lower` -> `es dataset bake` -> `train_act.py --baked` -> `es policy pack` ->
`TorchRuntime::load` -> `infer`, and compares the chunk against a direct PyTorch forward at §8.9's
tier-4 fp32 tolerance. The loss must still fall to `LOSS_MUST_FALL_TO` of its initial value — the
threshold does not move; lowering it is editing a golden. Missing `torch` -> `SKIP` with its reason;
an error *inside* PyTorch is a failure. Ran -> `RAN act_training_uses_baked_observations`.

**The measurement** (the point of plan V, oracle server, not a CI tier): bake the 50-episode
dataset, retrain 20,000 steps at `--batch 8 --lr 1e-4 --seed 0` with checkpoints at 1k/5k/20k —
every knob identical to V2 — pack three bundles, and re-run V3's measurement exactly as V3 ran it
(`visible_learning_demo_run` with `ES_TRAINED_BUNDLE`, plus the direct `es eval run` invocations of
design note section 7.8): 16 nominal episodes per checkpoint, the full six-suite table on 20k,
`es video mosaic` + `python/es/encode_video.py`. `evaluation.toml`'s `success_rate >= 0.5`
acceptance **is not lowered**; the measured number is reported against it, whatever it is.

## acceptance

```rust
// crates/es-eval/src/bake.rs
/// Runs the Observation IR over *recorded* frames through the same `CpuPlan`, the same
/// `input_sources` resolution and the same input encoding `capture` uses at inference (§7.2).
pub struct ObservationBake { /* plan + resolved sources */ }

impl ObservationBake {
    /// No `ModelInfo`: a recorded dataset carries `observation.state` and tiles and nothing
    /// else, so an input that needs a loaded model is refused here, by name, before any frame.
    pub fn new(obs: &ObservationIr, task: &TaskIr) -> Result<Self, EvalError>;
    /// The episode boundary (§7.5), exactly `run_episode`'s `plan.reset()`.
    pub fn reset(&mut self);
    pub fn frame(
        &mut self,
        state: &[f64],
        image: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
    ) -> Result<BTreeMap<String, Tensor>, EvalError>;
    pub fn outputs(&self) -> impl Iterator<Item = (&String, ElemType, &[u64])>;
}
```

```
es dataset bake --policy <bundle.esb> --out <dir> [--frames <tiles>] <dataset-root>
# <dir>/episode-000000.safetensors   { "<obs output>": [frames, ...], "action": [frames, dim] }
# <dir>/manifest.json                observation_hash, task_hash, compiler_hash,
#                                    dataset_content_hash, episodes[], tensors{}, frames
```

- `--policy <bundle.esb>`, not `--observation <obs.toml>`: the bundle is the one document that
  carries the Task IR **and** the Observation IR that inference will use, it is already the input
  to `es policy lower` one step earlier in the same flow, and `observation_hash` in the manifest is
  then the bundle's own rather than a re-derivation.
- The bake performs no conversion of its own. Every byte between `observation.state` / a raw tile
  and a baked tensor is produced by `CpuPlan::run`.
- A dataset row narrower than a `JointState { dof }` channel, a tile of the wrong size, a plan
  output that is not `F32`, a frame count that disagrees with the tile count: each is a named
  refusal. Nothing is padded, resampled, zero-filled or reshaped to make a bake complete.
- `train_act.py` reads `--baked <dir>` and **only** that: `read_dataset`, `read_frames`,
  `dequantize` and `plan_inputs` are deleted, not kept behind a flag. A contract input with no
  baked tensor is a refusal, which retires the "silently zero" failure mode V2 had to warn about.
- No new trait (INV-17), no new external dependency in Rust or Python, ≤ ~300 source lines.

## forbidden

- `crates/es-data/src/lerobot/**` — the v2.1 writer is not touched, not upgraded and not replaced.
  The bake **reads**; V1b's `es dataset export --lerobot-v3` is the converter and stays as it is.
- `crates/es-env`, `crates/es-safety` — no envelope change, no reset override, no renderer change.
  The measurement re-runs V3's documents unmodified (INV-11, INV-12, INV-13).
- `crates/es-ir` — no schema change. `VisionEncoder{pretrained}` is *rejected by the lowering*, not
  removed from the node, and `es_ir::learning::testing::act_like` keeps declaring `true`; the two
  `es-policy` tests that lower it clear the flag locally.
- `tests/fixtures/visible-learning/{task,observation,deployment,evaluation}.toml` — moving
  `observation_hash` would invalidate the baked set and V3's `evaluation_hash` in the same stroke,
  and the cheap alternative open question 11 rejects (dropping the state `Normalize`) is exactly
  that. Only `learning.toml` moves, and only its `pretrained` flag.
- Lowering `evaluation.toml`'s `success_rate >= 0.5` acceptance, or any threshold in
  `ir_training.rs`. A measurement that fails its acceptance is reported as failing it.
- Retraining with anything V2 did not use: same seed, same batch, same learning rate, same step
  counts. One variable moves in this experiment, and it is the observation.
