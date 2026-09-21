# M7 T6 — training augmentation: the Observation IR's `training_only` nodes, applied by the trainer

Spec: §7.3 (`Augment` nodes: `RandomCrop / ColorJitter / RandomErasing / GaussianNoise`,
`training_only = true`, auto-disabled under an Evaluation IR — INV-15), §19.3 (`augmentation.json`
and `seed.json`'s augmentation seed are identity slots), §5.1 (Observation IR owns preprocessing;
the trainer applies it, it does not invent it), §3.4 (no global RNG), §28.10 rule 1 and (T6).
Design notes to extend: `docs/design/training-recipe.md` (+ `.ko.md`) — a section "augmentation";
`docs/design/observation-lowering.md` (+ `.ko.md`) — what `training_only` means on each path.
Depends on **T1** (`training/`), **T3** (the batch axis). Predecessors: `crates/es-compile/src/plan.rs`
(the `Augment` arm: an identity pass-through, `COMPILE-002` outside `training_only`; `Pad` is not
lowered there yet), `crates/es-ir/src/cross.rs` (XIR-051, INV-15),
`crates/es-eval/src/runner.rs::refuse_augmentation`.

## the question

The IR has augmentation nodes and every path *ignores* them: the bake runs the Release plan, so
`Augment` is an identity; `train_act.py` never sees the Observation IR; `augmentation.json` says
`{"kind":"none"}` for every run. **Can the `training_only` subgraph the author declared be applied
by the trainer — random per sample, reproducible from one seed, recorded as identity — while the
evaluation path keeps producing the bytes it produces today?**

## spec

* **The boundary.** On the IR route `es train` reads the bundle's Observation IR and finds, per
  image input, the **contiguous chain of `training_only` `Augment` nodes** ending at the network's
  input port (a non-augment node inside the chain, or a `training_only` node with a consumer that
  is not the chain's next node, is refused by name). The tensor **entering** the chain is the
  training tensor; the tensor leaving it is what the Learning IR consumes.
* **The bake writes the boundary.** `es dataset bake` (`crates/es-eval/src/bake.rs`) gains
  `--for-training`: for a port whose chain is non-empty it writes the chain's *input* buffer (the
  plan already holds it — `produced[id] = src` — expose it as `CpuPlan::buffer_of(node)`), and its
  `manifest.json` records the chain (`"augmentation": [ {id, kind, params}, ... ]`) and the boundary
  shape. Without the flag, and for a document with no `training_only` node, the bake is byte-identical
  to today (existing bake tests unchanged).
* **The evaluation path.** `CpuPlan` in Release mode keeps `Augment` structurally off (INV-15) with
  one refinement: a `training_only` **`RandomCrop { width, height }` whose input is larger than
  `width x height` lowers to the deterministic centre crop** (offset `((W - width) / 2, (H - height) / 2)`,
  reusing `Crop`'s op *and its intrinsics transform* — INV-14). `Pad` (replicate mode) is lowered
  so that `Pad(4) -> RandomCrop{96, 96}(training_only)` — DrQ's random shift — hands the network
  the same `96x96` in evaluation and in training; with a symmetric pad its evaluation pixels are
  bit-identical to the un-augmented document's, and the test says so. The other three kinds stay
  identities. No committed Observation IR fixture has an `Augment` node (the one grep hit,
  `tests/fixtures/quadruped/task.toml`, is a Task IR — verify and say so in the note), so no golden
  moves. The GPU observation lowering (`crates/es-compile/src/gpu/`) mirrors the same rule or
  refuses the node by name (`COMPILE-0xx`) — never a silent identity that a CPU plan would not be.
* **The trainer applies the chain.** `train_act.py --augmentation <training/augmentation.json>`
  applies, per sample and per optimizer step, in chain order: `RandomCrop{w,h}` — a uniformly drawn
  integer offset in `[0, W-w] x [0, H-h]`; `ColorJitter{brightness, contrast, saturation, hue}` —
  brightness and contrast only (`x * (1 + u*b)` then `(x - mean) * (1 + u*c) + mean`, `u` in
  `[-1, 1]`), saturation/hue refused by name when non-zero; `GaussianNoise{sigma}` — additive;
  `RandomErasing` refused by name (not implemented here; the note lists it). The draws come from a
  **counter-based RNG the trainer and the Rust oracle both implement** — the `mix32` of
  `es_render::rng::{key, uniform}` (ten lines of integer arithmetic), keyed
  `(augmentation_seed, sample_index, step, node_index, draw)`; never `torch.Generator`, whose stream
  is a torch-version detail. Gaussian draws are Box-Muller over two uniforms, `f64` on both sides,
  compared at `f32` after the cast — the last-bit libm question is measured, not assumed (T4
  section 10 did the same for `cos`).
* **Identity.** `augmentation.json` becomes `{"kind":"observation-ir", "observation_hash": ...,
  "chains": {"<port>": [ {id, kind, ...params} ]}, "seed": <s>}`; `seed.json.augmentation` becomes
  the real `[run] augmentation_seed` (default = `[run] seed`). Both move `identity_hash`. A recipe
  whose bundle has no chain writes today's `{"kind":"none"}` and the unset seed — the measured
  runs' `training_hash` is unmoved.
* **The document for the U-measurement.** `tests/fixtures/visible-learning/observation-augmented.toml`:
  the committed `observation.toml` plus `Pad(4, replicate) -> RandomCrop{96,96}(training_only)` and
  `ColorJitter{brightness 0.2, contrast 0.2}(training_only)` on the image path. Its `observation_hash`
  is recorded in the design note beside the committed one; the Learning IR is unchanged
  (`learning_hash` unmoved). Because the Evaluation IR names `observation`, an evaluation of a policy
  trained on it is a **new** `evaluation_hash` (spec 13.3) — say so where the U-measurement will read it.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
python/es/train_act.py
python/es/augment.py
crates/es-compile/src/plan.rs
crates/es-compile/src/gpu/**
crates/es-compile/src/lib.rs
crates/es-compile/tests/**
crates/es-eval/src/bake.rs
crates/es-eval/tests/**
crates/es-policy/tests/ir_training.rs
crates/es-data/src/training.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/dataset.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/observation-augmented.toml
tests/golden/train/augment_seed0.json
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/design/observation-lowering.md
docs/design/observation-lowering.ko.md
docs/packets/M7/T6-augmentation.md
docs/packets/M7/T6-augmentation.ko.md
```

`python/es/augment.py` (new: the RNG and the kinds, importable by the trainer and by the oracle),
`train_act.py` (`--augmentation`), `es-compile` (`buffer_of`, `Pad`, the centre-crop rule, the GPU
mirror/refusal), `es-eval/src/bake.rs` (`--for-training`), `es-data/src/training.rs`
(`augmentation_seed`, the two slots), `es/src/cmd/train.rs` (reads the bundle's Observation IR,
writes `augmentation.json`, passes the flag), `es/src/cmd/dataset.rs` (the bake flag), tests, the
fixture, the golden, the two design notes, this packet.

## oracle

1. `cargo test -p es-policy --test ir_training augmentation_matches_the_golden` — a synthetic
   `3x12x12` image through `RandomCrop{8,8} -> ColorJitter{0.2, 0.2} -> GaussianNoise{0.05}` for
   samples 0..4 at seed 0: the Rust re-implementation equals `tests/golden/train/augment_seed0.json`
   bitwise at `f32`; with `ES_PYTHON`, `python/es/augment.py` equals it too (else `SKIP` with reason).
   The golden is generated once by an `#[ignore]`d test from the Python side.
2. `cargo test -p es-compile a_training_only_random_crop_is_a_centre_crop_in_release` — the plan
   for `Pad(4) -> RandomCrop{96,96}(training_only)` on a `96x96` input produces bytes identical to the
   plan without the two nodes, and the output `ImageSpec` intrinsics are the input's (INV-14); a
   `RandomCrop` whose input already is `width x height` is the identity it was.
3. `cargo test -p es-eval bake_for_training_writes_the_boundary` — on the augmented fixture the
   `--for-training` bake writes `104x104` for the image port and the chain in its manifest; without
   the flag, `96x96` and the bytes the un-augmented document's bake writes.
4. `cargo test -p es --test cli train_identity_moves_with_augmentation` — the augmented bundle's
   run writes `augmentation.json` with two chain entries and a real `seed.json.augmentation`;
   `identity_hash` differs from the un-augmented bundle's and moves with `augmentation_seed`; the
   un-augmented recipe still writes `{"kind":"none"}` (dry-run level, no Python).
5. `cargo test -p es-policy --test ir_training -- --ignored augmented_training_runs` — with
   `ES_PYTHON`: 40 steps on the augmented fixture's bake; the loss is finite; two runs at one
   `augmentation_seed` give byte-identical loss curves on CPU, two seeds do not.
6. `cargo xtask ci` (goldens: 0 changed); `cargo xtask check-scope docs/packets/M7/T6-augmentation.md`.

## acceptance

Oracles 1–6 (1's Python half and 5 on the oracle server). On the server: the augmented document's
20,000-step run at T4's row-D settings (batch 64, `warmup_cosine`), its `training.lock` and
checkpoint under `~/artifacts/plan-v/m7-t6/` for the U-measurement, wall-clock beside T4's table
(augmentation cost per step is an observation). **Do not evaluate** it here. The design note
records the boundary rule, the RNG key table, the four kinds' status (two implemented, one partial,
one refused) and the `observation_hash` pair.

## forbidden

`crates/es-ir/**` (the nodes exist; INV-15's structure is theirs); a `torch.Generator` or any
global RNG in the augmentation; changing the Release-plan bytes of any node other than the
`training_only` `RandomCrop` refinement above; the committed `observation.toml`, `learning.toml`,
`training.toml`; `docs/ARCHITECTURE*.md`; goldens other than the new one. INV-14: no crop without
its intrinsics transform. INV-16: safetensors only. INV-17: no new trait.
