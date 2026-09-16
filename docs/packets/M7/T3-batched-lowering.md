# M7 T3 — the batch axis: `lower_to_torch` emits `[N, …]`, `train_act.py` trains real batches

Spec: §8.7 (lowering), §5.2 (the inference domain has its own batch size — the IR declares
none), §8.9 (tier-4 equivalence), §1.4, §28.9 ladder rung 9 ("remove the per-sample loop in the
batch lowering — the root cause of the training time"), §28.10 (T3). Design note to edit:
`docs/design/learning-lowering.md` (+ `.ko.md`) — sections 2, 3, 5.1 and a new 5.2 "the batch
axis". Predecessors: M5 V2 (`train_act.py`), V5 (the three speed flags moved nothing: at batch 8
the module runs one sample at a time), V13 (GroupNorm; `train()==eval()` is an oracle).

## the question

`--batch 8` is eight single-sample forwards accumulated (`train_act.py` docstring, "the lowered
module is single-sample"): `VisionEncoder` lowers to `self.n{k}(x.unsqueeze(0)).squeeze(0)`, the
regression head to `.reshape(horizon, action_dim)`, the chunker to `[:execute_chunk]`. 20,000
steps take ~11 min on an RTX 4090 and no flag moves it (design note 7.11). **If the lowered
module takes `[N, …]` and the trainer feeds `N` at once, where does the time go — and is the
function unchanged?**

## spec

* `lower_to_torch` emits a module whose `forward(**inputs)` takes every input with a **leading
  batch axis** and returns every output with one: image ports `[N, C, H, W]`, state `[N, D]`,
  noise `[N, …]`, chunk `[N, K, A]`. Node by node: `VisionEncoder` → `self.n{k}(x)` (no
  unsqueeze); `TemporalEncoder{Transformer}` with `token_count == 0` → `x.unsqueeze(1)` … `.squeeze(1)`
  (a one-token *sequence*, batch first — the axis that used to be faked is now real); `PolicyHead
  {Regression}` → `.reshape(-1, horizon, action_dim)`; `ActionChunker` → `[:, :execute_chunk]`;
  `Normalizer` broadcasts as it does; `Diffusion` / `FlowMatching` samplers loop over steps with a
  batched state (`noise.reshape(N, -1)`), the per-step arithmetic otherwise **unchanged
  expression for expression** (the tier-4 tolerance on the sampler heads is `1e-5`, and a moved
  expression is a moved ULP). `Fusion{Concat}` keeps `dim=-1`, `TokenConcat` `dim=-2`.
* `contract.json` keeps the IR's per-sample shapes under `inputs` and adds `"batch_axis": true`
  (a reader that does not know the key treats the module as single-sample and fails loudly on
  shape, which is the right failure).
* `crates/es-policy/python/torch_ref.py` (the `TorchRuntime` side) feeds `[1, …]` and squeezes
  the batch axis off every output before returning — inference stays one sample, per §5.2 the
  inference domain's batch is the runtime's business, not the module's. The LeRobot passthrough
  (`lerobot.rs`, `PolicyBundle`) is untouched: it never went through this lowering.
* `python/es/train_act.py`: `--batch N` becomes a **real batch** — `N` samples stacked along dim 0,
  one forward, `L1` mean over `[N, K, A]` — same optimizer, same seed handling, same sample
  order (the permutation is drawn once per epoch exactly as today, so batch `i` holds the same
  eight samples the accumulation loop would have visited). `--resident-gpu`, `--amp`,
  `--compile`, `--channel-weight`, `--checkpoint-at`, `--loss-curve` keep their meaning; update
  the docstring's "single-sample" paragraphs. The JSON summary gains `"batch_axis": true`.
* `crates/es-policy/tests/ir_training.rs` and `crates/es/tests/cli.rs` tests that build
  `probe` tensors or index outputs are updated for the axis (those edits are in scope); the
  `act_checkpoint` oracle (LeRobot path) is not touched and must still read `max_abs 0e0`.
* Server measurement (acceptance): 20,000 steps, batch 8, seed 0, `--resident-gpu`, on V15's
  baked set (`~/artifacts/plan-v/v15/baked` or re-bake from `ds-train`), before (main) and after
  (this branch), wall-clock and `final_loss` side by side; then batch 64 at the same lr as an
  observation only (T4 owns the schedule — do not tune).

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

```
crates/es-policy/src/lower/torch.rs
crates/es-policy/src/lower/mod.rs
crates/es-policy/src/torch_runtime.rs
crates/es-policy/python/torch_ref.py
crates/es-policy/tests/*.rs
python/es/train_act.py
crates/es/tests/cli.rs
docs/design/learning-lowering.md
docs/design/learning-lowering.ko.md
docs/packets/M7/T3-batched-lowering.md
docs/packets/M7/T3-batched-lowering.ko.md
```

`crates/es-policy/src/lower/torch.rs`, `crates/es-policy/src/lower/mod.rs`,
`crates/es-policy/src/torch_runtime.rs` (only if the contract read needs the flag),
`crates/es-policy/python/torch_ref.py`, `crates/es-policy/tests/ir_training.rs`,
`crates/es-policy/tests/*.rs` that assert on lowered source text, `python/es/train_act.py`,
`crates/es/tests/cli.rs` (existing tests that shape-probe the module; no new files),
`docs/design/learning-lowering*.md`, `docs/packets/M7/T3-batched-lowering*.md`. `lowering_hash`
moves (the source moved); `learning_hash` does not; record both in the design note.

## oracle

1. `cargo test -p es-policy --lib lower::torch::tests::the_module_has_a_batch_axis` — for the
   ACT-shaped fixture and for the Diffusion and FlowMatching fixtures, the generated source
   contains no `.unsqueeze(0)` / `.squeeze(0)` batch idiom, the head reshapes with a leading
   `-1`, the chunker slices `[:, :K]`, and `contract.json` says `batch_axis: true`. No Python.
2. `cargo test -p es-policy --test ir_training -- --ignored the_batch_is_invariant` — with torch:
   eight distinct inputs through the demo bundle's lowered module as one `[8, …]` batch and as
   eight `[1, …]` calls give outputs equal within `Tolerance::TIER4_FP32` (`1e-5`) on CPU, and
   the maximum difference is printed (GroupNorm and Linear are per-sample, so it should be
   small; report what it is).
3. `cargo test -p es-policy --test ir_training -- --ignored the_backbone_computes_the_same_function_in_train_and_eval`
   — V13's oracle, unchanged, still passes on the batched module.
4. `cargo test -p es-policy --test ir_training -- --ignored act_training_uses_baked_observations
   resident_gpu_does_not_move_the_loss channel_weight_of_one_is_the_unweighted_loss` — the
   existing training oracles pass with real batches (their loss numbers may move because the
   sum order moved: that is expected and the design note says so; the *properties* they assert
   must hold).
5. `ES_ACT_CHECKPOINT=… cargo test -p es-policy --test act_checkpoint` on the server —
   `max_abs 0e0`, the LeRobot path untouched.
6. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T3-batched-lowering.md`.

## acceptance

Oracles 1–6 pass (2–5 on the oracle server with `ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`).
The before/after table (wall-clock, `final_loss`, `steps/s` is **not** reported — §12.4 has no
such metric; report seconds per 1,000 optimizer steps and samples per second) is in design note
section 5.2 with the batch-64 observation beneath it. The 20k checkpoint from the *after* run is
kept at `~/artifacts/plan-v/m7-t3/model-20000.safetensors` for the U-measurement (do not
evaluate it here — that is wave 5's job).

## forbidden

`crates/es-ir/**` (the IR keeps no batch axis — §5.2); `crates/es-policy/src/lerobot.rs` and the
LeRobot `act_checkpoint` path; `crates/es-safety/**`, `crates/es-eval/**`, `crates/es-env/**`;
any change to a sampler's per-step arithmetic beyond adding the axis; the learning-rate default,
the optimizer, a schedule (T4); `pretrained` handling (T5); every fixture and golden;
`docs/ARCHITECTURE*.md`. INV-16: safetensors only. INV-17: no new trait.
