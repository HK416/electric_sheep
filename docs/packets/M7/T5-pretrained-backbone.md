# M7 T5 — `pretrained = true`: ImageNet ResNet18 weights with a `base_model.lock`

Spec: §8.3 (`VisionEncoder { pretrained, frozen }`), §19.3 (`base_model.lock` — "the provenance
of the pretrained backbone directly affects the results, it must be included in the provenance;
it is also the basis for license tracking"), §5.3 (nothing outside the hash chain), §2.5 (no
network at instantiation), §29 licence row (owner decision 2026-09-15: torchvision's ImageNet
ResNet18 weights, BSD-3, are allowed), §28.9 L13, §28.10 (T5). Design note to edit:
`docs/design/learning-lowering.md` (+ `.ko.md`) section 5.1 → 5.3 "pretrained", and
`docs/design/training-recipe.md` (T1) for the lock file. Depends on **T1** (`training/`) and
**T3** (the batch axis). Predecessors: V2b (refusal of `pretrained = true` — read the comment in
`lower/torch.rs` for *why* it was refused: network at construction, no hash slot), V8/V19
(`lerobot.rs` lowers LeRobot's backbone with `FrozenBatchNorm2d` — the precedent), V13
(GroupNorm for from-scratch).

## the question

The from-scratch ResNet18 found a 25 mm cube in 96×96 after 200 demonstrations and 40k steps and
still trails LeRobot's ACT, whose one architectural difference that matters is an ImageNet-
pretrained backbone (design note 7.27). `pretrained = true` is refused because instantiating it
would fetch weights over the network and nothing would hash them. **Can pretrained weights enter
the chain as a named, hashed, licensed artifact so that `pretrained = true` lowers — and does the
lowered function still have no training mode?**

## spec

* **The artifact.** `python/es/fetch_backbone.py --arch resnet18 --out <dir>` (learning path,
  Python is allowed) instantiates `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
  once, writes `resnet18-imagenet1k-v1.safetensors` (every `state_dict` tensor, torchvision's own
  key names, **including** the BatchNorm running statistics — they become frozen constants) and
  `resnet18-imagenet1k-v1.lock.json`: `{ "source": "torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
  "torchvision": "<version>", "url": "<the weights URL torchvision reports>", "sha256_upstream":
  "<torchvision's own file hash>", "blake3": "<blake3 of the safetensors written>", "license":
  "BSD-3-Clause", "license_url": "https://github.com/pytorch/vision/blob/main/LICENSE" }`. The
  pickle read happens inside torchvision on the learning path only — `es` never opens the
  `.pth`; INV-16 is about what *this project's* loaders accept, and they accept safetensors.
  The expected `blake3` is **pinned in the repository** as a constant beside the provenance test
  (the way `go1_primitives.PROVENANCE.json` pins upstream hashes) — the script refuses to write a
  file whose hash differs from the pin unless `--repin` is passed, and says so.
* **The IR side.** `VisionEncoder { pretrained: true }` is no longer refused. It lowers to
  torchvision's `resnet18(weights=None, norm_layer=FrozenBatchNorm2d)` with `fc` replaced as
  today; the prefix claim `nodes.<k>.*` now covers the frozen buffers too. **No network at
  construction** (`weights=None`); the ImageNet tensors arrive through the checkpoint, exactly
  as a trained checkpoint's do — `es policy pack --weights` accepts them, and the *initial*
  checkpoint for training is produced by `es train` (T1): when the recipe's bundle has a
  pretrained encoder, `es train` reads `base_model = "<dir>/resnet18-imagenet1k-v1.safetensors"`
  from the recipe, verifies its blake3 against the lock file **and** the pinned constant, writes
  `training/base_model.lock` from the lock file (§19.3, real values now), and hands
  `train_act.py --init-backbone <file>` the tensors to load under `nodes.<k>.` before the first
  step. `frozen: true` → `train_act.py` excludes `nodes.<k>.*` from the optimizer (the
  parameters' `requires_grad = False`); `frozen: false` fine-tunes. The BatchNorm statistics are
  frozen either way (`FrozenBatchNorm2d` has no training mode) — that is the V13 rule kept:
  `train() == eval()` bit-identical.
* **The hash consequence.** `lowering_hash` moves for graphs with `pretrained: true` only (the
  source differs); `learning_hash` does not; `training_hash` gains a non-zero `base_model` slot.
  The demo's committed `learning.toml` keeps `pretrained = false` — this packet changes no fixture;
  a second document `learning-pretrained.toml` is added under `tests/fixtures/visible-learning/`
  for the U-measurement, with its own `learning_hash` recorded in the design note.
* **Refusals by name.** A `base_model` whose blake3 disagrees with the lock file or the pin
  (`TRAIN-0xx`); `pretrained: true` in a bundle whose recipe names no `base_model`; a lock file
  whose `license` field is empty.

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

```
python/es/fetch_backbone.py
python/es/train_act.py
crates/es-policy/src/lower/torch.rs
crates/es-policy/tests/ir_training.rs
crates/es-policy/tests/backbone_provenance.rs
crates/es-data/src/training.rs
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/learning-pretrained.toml
docs/api-notes/torchvision.md
docs/api-notes/torchvision.ko.md
docs/design/learning-lowering.md
docs/design/learning-lowering.ko.md
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M7/T5-pretrained-backbone.md
docs/packets/M7/T5-pretrained-backbone.ko.md
```

`python/es/fetch_backbone.py` (new), `python/es/train_act.py` (`--init-backbone`, `frozen`
handling), `crates/es-policy/src/lower/torch.rs` (the `pretrained` arm), `crates/es-policy/tests/ir_training.rs`
(new tests), `crates/es-policy/tests/backbone_provenance.rs` (new, `#[ignore]`d: downloads via the
script on a machine with torchvision and checks the pinned blake3), `crates/es-data/src/training.rs`
(the `base_model` recipe field and lock writer), `crates/es/src/cmd/train.rs`, `crates/es/tests/cli.rs`
(`train_*` tests only), `tests/fixtures/visible-learning/learning-pretrained.toml` (new),
`docs/api-notes/torchvision*.md` (a "pretrained weights" subsection: URL, upstream sha256,
version), `docs/design/learning-lowering*.md`, `docs/design/training-recipe*.md`,
`docs/packets/M7/T5-pretrained-backbone*.md`.

## oracle

1. `cargo test -p es-policy --lib lower::torch::tests::a_pretrained_backbone_lowers_frozen` —
   the generated source for `pretrained: true` contains `weights=None` and `FrozenBatchNorm2d`,
   contains no `weights="DEFAULT"`/`IMAGENET1K`, and the declared claim covers `nodes.<k>.*`;
   `pretrained: false` still lowers to GroupNorm (V13's test unchanged). No Python.
2. `cargo test -p es-policy --test ir_training -- --ignored the_pretrained_backbone_has_no_training_mode`
   — with torch: `torch.equal(backbone(x).train(), backbone(x).eval())` on the pretrained module
   after `--init-backbone`; the loaded tensors equal the safetensors file's bitwise.
3. `cargo test -p es-policy --test backbone_provenance -- --ignored` — on a machine with
   torchvision (the server): the script writes the file, its blake3 equals the pinned constant,
   the lock file's `license` is `BSD-3-Clause`; a `--repin`-less run against a tampered pin is
   refused by name.
4. `cargo test -p es --test cli train_refuses_a_mismatched_base_model` and
   `train_writes_base_model_lock_from_the_lock_file` (dry-run level; no Python).
5. `cargo test -p es-policy --test ir_training -- --ignored frozen_excludes_the_backbone_from_the_optimizer`
   — 20 steps with `frozen: true`: `nodes.<k>.*` tensors unchanged bitwise, the head moved.
6. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T5-pretrained-backbone.md`.

## acceptance

Oracles 1–6 (2, 3, 5 on the oracle server). The artifact and lock live at
`~/artifacts/plan-v/m7-t5/` on the server (the repository carries the pin and the lock's expected
fields, not the 45 MB file). Design note section 5.3 records the rule: *a pretrained backbone is
a checkpoint like any other — it enters through `es policy pack`/`es train`, never through the
network at construction — and its provenance is `base_model.lock`.*

## forbidden

Any network access at module construction (`weights=` other than `None`); a pickle reader in
Rust or in `es`'s Python helpers other than torchvision's own inside `fetch_backbone.py`
(INV-16); `crates/es-policy/src/lerobot.rs`; changing the committed `learning.toml`; `es-ir`
schema (`pretrained`/`frozen` exist); `docs/ARCHITECTURE*.md`; goldens. INV-17: no new trait.
