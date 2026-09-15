# M5 V2 — training the Learning IR lowering in PyTorch, and packing it back

Design note: `docs/design/visible-learning.md` section 6; read section 2.5 first — there is no training
loop anywhere in the repo, and a bundle produced through `lower_act` **cannot be run by `es eval run`**,
which is why this packet trains the IR-owned graph instead. Depends on V1 (the dataset).

> **Superseded in part by `V2b-observation-bake.md`.** This packet's training read the LeRobot
> parquet directly and re-implemented one Observation IR node in Python, which made every success
> rate V3 measured a measurement of ACT fed an observation it was not trained on (design note
> section 7.9). `train_act.py` now takes `--baked <dir>` from `es dataset bake`, not `--dataset`
> and `--frames`, and V2's `the_loss_falls_and_the_packed_bundle_round_trips` is V2b's
> `act_training_uses_baked_observations`. Everything below is the record of what V2 did.

## context

```
crates/es/src/cmd/policy.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/mod.rs
crates/es/Cargo.toml
xtask/src/main.rs
crates/es/src/main.rs
crates/es/tests/cli.rs
crates/es-policy/src/lower/mod.rs
crates/es-policy/src/lower/torch.rs
python/es/train_act.py
python/es/README.md
python/es/README.ko.md
crates/es-policy/tests/ir_training.rs
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V2-act-training.md
docs/packets/M5/V2-act-training.ko.md
```

Notes: `crates/es/src/cmd/policy.rs` is new — `es policy lower` and `es policy pack`; `mod.rs` and
`main.rs` get the module and one dispatch arm (`main.rs:48-61` is the table). `lower/mod.rs` re-exports
the contract type only. `python/es/train_act.py` is the **only** new Python; it is not counted by
`cargo xtask context-budget` (which reads `*.rs` under `src/`, `xtask/src/context_budget.rs:148-152`).

## spec

- §1.4: training cannot be judged by "matches a reference", so it is judged by four executable facts
  (design note section 6.3): the module is the IR's, the weights fit the contract, the loss falls, the
  bundle round-trips. None of them is a human reading a curve.
- §2.3, §2.4: the optimizer is on the Python side of the split; `es policy lower` and `es policy pack` are
  Rust and run without Python. `es-policy` gains no Python dependency it does not already have.
- §8.1: the optimizer is not in the IR — `train_act.py` reads hyperparameters from its own CLI, and none
  of them enters any hash slot except through `TrainingIdentity` (§19.2), which V2 does not populate.
- §8.7, INV-16: weights are `safetensors`. `WeightsSource` (`crates/es-policy/src/runtime.rs:27-32`) gains
  no variant, and no pickle-loadable file is written or read. The IR owns pre/post-processing: `pack`
  never moves a normalizer into the runtime.
- §5.3: `pack` recomputes the bundle manifest rather than editing it
  (`crates/es-compile/src/bundle.rs:467`, `:513`); `policy_hash` changes because the weights changed.
- §25.1: `model.safetensors` comes from outside the Rust core. `pack` validates every key and shape
  against the contract **before** writing, and refuses an unknown key rather than ignoring it.
- §12.4: no training time, no throughput, no `step/s` anywhere in this packet's output.
- §1.5: `es` is at 2,946 code lines; this packet is budgeted under ~400 in `src/`.

## oracle

```
cargo fmt --check
cargo clippy -p es -p es-policy --all-targets -- -D warnings
cargo test -p es --test cli policy_
cargo test -p es-policy --test ir_training
cargo xtask context-budget
cargo xtask check-spec-refs
```

Reference — the training leg, which needs `torch`:

```
ES_PYTHON=$HOME/venvs/es-lerobot/bin/python \
  cargo test -p es-policy --test ir_training -- --ignored --nocapture
```

`tests/ir_training.rs` drives the whole three-step loop on a tiny fixed dataset in a temp dir and checks
the four facts. No `torch` -> `SKIP ir_training: <why>` per fact; all four ran -> `RAN ir_training`. A
real PyTorch error must **not** be folded into the same SKIP as "torch is not installed" — that is
`docs/reviews/M4.md:67` (S-7) about `act_checkpoint.rs:113`, and this packet must not repeat it.

**Measured 2026-09-14 on the oracle server.** Both venvs are CPU-only builds: `~/venvs/es` has
`torch 2.14.0+cpu` and `~/venvs/es-lerobot` has `torch 2.11.0+cpu`, and `torch.cuda.is_available()` is
`False` in both on an RTX 4090. V0's image is 96x96 and `H = 16` for this reason. This packet states no
training time and no hardware claim (§12.4); the GPU question is design note open question 8.

`tests/ir_training.rs`:

- `the_trained_module_is_the_lowering` — the `es_policy.py` `train_act.py` loaded is byte-equal to
  `lower_to_torch(&bundle.learning)?.source` (`crates/es-policy/src/lower/torch.rs:334`). This is the
  packet's whole reason for existing: it makes §1.4's "the same IR run in PyTorch is the ground truth"
  literally true, instead of "a similar model".
- `train_act_defines_no_layer` — a source scan of `python/es/train_act.py` finds no `nn.Linear`,
  `nn.Conv`, `nn.Transformer`, `nn.Module` subclass or `torchvision.models` call. The architecture comes
  from the lowering or it does not come.
- `pack_refuses_a_missing_key`, `pack_refuses_an_extra_key`, `pack_refuses_a_wrong_shape` — three
  refusals, each naming the offending key, exit code 1.
- `the_loss_falls` — a fixed dataset and `--seed 0`; final loss below a pinned fraction of the initial
  loss, both printed by `train_act.py` as JSON. A threshold, not a golden: CPU PyTorch is not bitwise
  portable across versions (design note section 9).
- `the_packed_bundle_round_trips` — `PolicyBundle::open` on the output succeeds, `TorchRuntime::load`
  accepts it (i.e. it goes through `lower_to_torch`, not `load_lowered`), and `infer` on a fixed
  observation returns a finite chunk of shape `[H, NJ]`.
- `eval_run_accepts_the_trained_bundle` — `es eval run --policy trained.esb` gets past
  `TorchRuntime::load` (`crates/es/src/cmd/eval.rs:387-393`). This is the assertion `lower_act` would
  fail, and it is why plan V does not use it.

`crates/es/tests/cli.rs`: `policy_lower_writes_the_module_and_contract`,
`policy_pack_rejects_a_non_safetensors_file` (no pickle path is reachable, INV-16),
`policy_usage_errors_exit_2`.

## acceptance

```rust
// crates/es/src/cmd/policy.rs
// es policy lower --policy <in.esb> --out <dir>
//   writes <dir>/es_policy.py           = TorchModule::source, verbatim
//          <dir>/contract.json          = { lowering_hash, weight_keys: [String],
//                                           weight_shapes: { key: [usize] },
//                                           action_dim, horizon, inputs: { port: [usize] } }
// es policy pack --policy <in.esb> --weights <model.safetensors> --out <out.esb>
//   validates every key and shape against the contract, then rewrites the bundle with the new
//   weights.safetensors and a recomputed manifest.
pub fn dispatch(args: &[String]) -> i32;   // 0 ok, 1 runtime, 2 usage, 3 SKIPPED
```

```
python/es/train_act.py --module <dir> --dataset <root> --out model.safetensors \
    [--epochs N] [--batch N] [--lr F] [--seed N] [--device cpu]
```

- `train_act.py` `exec`s `<dir>/es_policy.py`, instantiates `EsPolicy()`, reads the LeRobot v2.1 dataset
  V1 wrote, optimizes, and writes `safetensors` whose key set equals `contract.json`'s exactly. It prints
  one JSON line `{"initial_loss": f, "final_loss": f, "steps": n}` and nothing else on stdout.
- It defines no layer and imports no model zoo. Every parameter it optimizes came from the lowering.
- `pack` refuses a missing key, an extra key or a mismatched shape, naming it, with exit 1; it never
  drops, pads or reshapes a tensor.
- The output bundle has the same `task`, `observation`, `learning` and `deployment` entries as the input
  and a new `weights.safetensors`; the manifest is recomputed by `PolicyBundle::build`, not patched.
- `es eval run --policy <out.esb>` loads it with **no change to `crates/es/src/cmd/eval.rs`'s policy path**.
- No new trait, no new external crate, ≤ ~400 source lines in `src/`.

## forbidden

- `crates/es-policy/src/lerobot.rs` — `lower_act` and the M4 bypass are explicitly **not** fixed here
  (design note open question 1). Do not call it, do not change it, do not widen
  `TemporalEncoder { Transformer }`.
- `crates/es-ir`, `crates/es-ir-types` — a spec change to §8.3 is what fixing the bypass would need, and
  `es-ir` has 53 lines of §1.5 budget left.
- `crates/es-policy/src/torch_runtime.rs` — in particular `load_lowered`, whose only caller is
  `lower_act` (`torch_runtime.rs:278-284`).
- `crates/es-policy/src/runtime.rs` — no `WeightsSource` variant, no pickle, no `torch.load` (INV-16).
- Adding an `es learn` or `es train` subcommand, or a Rust-side optimizer:
  `crates/es/src/cmd/loop.rs:5-7` says why.
- Populating `TrainingIdentity`'s non-`dataset` slots with anything but zeros
  (`crates/es-data/src/collect.rs:650-665`) — a fabricated digest makes `training_hash` a lie.
- `crates/es-env`, `crates/es-eval`, `crates/es-render`, `crates/es-data` — V0b, V1, V3.
- Reporting any wall-clock training figure as fact.

## as built

Design note: `docs/design/visible-learning.md` section 7.6 carries the findings; this section
carries the deltas to the packet above and the artifacts.

**The `## context` gained one file.** `crates/es-policy/src/lower/torch.rs`, for one line.
`VisionEncoder{ResNet18}` lowered to `self.n0(inputs["rgb_overhead"])` over a `[3, 96, 96]` IR
port, and every torchvision backbone is `nn.BatchNorm2d`, which refuses a 3-D input: *"expected
4D input (got 3D input)"*. The lowering therefore emitted a module that had never run and could
not run, for every graph with a vision encoder — `torch_equivalence.rs` only ever puts a
state-only MLP through PyTorch. The fix is `self.n0(x.unsqueeze(0)).squeeze(0)`, the same trade
the token-less `TemporalEncoder` arm already makes, in the lowering rather than in this packet,
because every caller routes through it. No golden pins a `lower_to_torch` lowering hash.

**Three flags `train_act.py` gained.** `--checkpoint-at 1000,5000,20000` (which also caps the run,
so a step count is exact rather than an epoch's rounding), `--loss-curve curve.json`, and
`--frames <dir>` for the tiles the collector now writes. Its one JSON line carries `samples`,
`batch`, `chunk`, `zero_filled_inputs` and `image_inputs` beyond the three keys the acceptance
pinned; the oracle reads only the three.

**Test names.** `the_loss_falls` and `the_packed_bundle_round_trips` are one test,
`the_loss_falls_and_the_packed_bundle_round_trips`, because fact 4 needs fact 3's checkpoint.
`eval_run_accepts_the_trained_bundle` is `policy_pack_output_is_accepted_by_eval_run` in
`crates/es/tests/cli.rs`, where the Evaluation IR fixture lives —
`tests/fixtures/visible-learning/` has no `evaluation.toml`, which is V3's document. The
round-trip is stronger than asked: it compares `TorchRuntime::infer` against a *direct* PyTorch
forward on a held-out observation at spec 8.9's tier-4 fp32 tolerance.

**One thing the packet did not scope, added on the orchestrator's decision.**
`es loop collect` had a `FrameSink` and no caller (`crates/es/src/cmd/loop.rs` passed `None`), so
the dataset carried no pixels and the image port would have trained on zeros. `--frames <dir>`
now builds the `Gpu` and the `EnvRenderer` in the collect call, configures them from the Task
IR's own image channel and `ImageSpec`, and writes one raw tile per control step. It is behind a
new `render` feature on `es`, off by default so the ordinary CLI links no Vulkan (spec 4.2);
without it the flag is refused rather than silently ignored. `xtask` gates `es/render` in the PR
tier alongside `es-env/render`, because a branch CI never compiles is a branch that rots. That
adds `crates/es/src/cmd/loop.rs`, `crates/es/Cargo.toml` and `xtask/src/main.rs` to the context.
Design note 7.6 items 2-3 carry it, including the one debt it creates: `train_act.py` now
re-implements `Op::Dequantize`, the single Observation IR node between the tile and the Learning
IR input.

**Two things this packet found that it does not own** (design note 7.6, items 4-5): `es eval run`
still cannot feed an image input (`es-eval/src/runner.rs:475-481`), so **V2 states no success
rate** — its claim is the packet's own, that the bundle gets past `TorchRuntime::load`, and the
demo's `evaluation.toml` does not exist yet either; and `es loop collect --episodes N` solves
only episode 0 (`--episodes 50 --seed 1` ends `success 1, timeout 49`; the same seeds one
episode at a time end 49 successes). Both are V1/V3 code.

### artifacts (oracle server, RTX 4090, `~/venvs/es-lerobot-cuda`, torch 2.11.0+cu129)

Nothing below is committed. Everything lives under `~/artifacts/plan-v/` on the oracle server;
the generator for each is the command in the table.

| Artifact | How | Size / value |
|---|---|---|
| `ds-train` + `frames-train` | `es loop collect --expert so101-pick-place --episodes 1 --seed s --frames ...`, `s = 1..50`, merged by `es loop distill` | 50 episodes, 17,697 frames, 3.4 MB parquet + 485 MB of tiles, **50/50 `Success`** |
| `ds-holdout` + `frames-holdout` | the same, seeds 101-105 | 5 episodes, 2,313 frames, 464 KB + 64 MB, 4/5 `Success` |
| `build/` | `es policy lower --policy untrained.esb --out build/` | `lowering_hash 956abb67775d4db61177fae4051e3883628e07183e4925483bca1e0249aeec6d`, 10 weight keys (8 exact, 2 prefix claims) |
| `model-1000.safetensors` | `train_act.py --frames frames-train --batch 8 --lr 1e-4 --seed 0 --device cuda --checkpoint-at 1000,5000,20000` | 61 MB, 142 tensors, blake3 `57c7e537fbfde3711e76582edc796fca09d0cbc3a2411cbe2e242c6b5b2bf1e8` |
| `model-5000.safetensors` | the same run | blake3 `ae1c38c5662226887eecff6473d44a9f61a51fd2d1e7782c403eb146e3bba2b0` |
| `model-20000.safetensors` | the same run | blake3 `0f0ad5c80faeebd698ae7fecd579b724502d278bd5c0af8e8479d73e064f476a` |
| `trained-20000.esb` | `es policy pack --policy untrained.esb --weights model-20000.safetensors` | `policy_hash 840aef948493bc59e74e510bfe1e87cdce3291cbdb1ed40f4364db1e34aa3662` |
| `loss-curve.json` | `--loss-curve` | 20,000 per-step L1 losses |

Training loss, L1 over the action chunk, mean of the 100 steps ending at each mark:

| step | 1 | 100 | 1,000 | 5,000 | 20,000 |
|---|---|---|---|---|---|
| loss | 0.7267 | 0.1825 | 0.0507 | 0.0309 | 0.0171 |

Measured on the oracle server with `ES_PYTHON=~/venvs/es` (the one venv with both `mujoco` and
`torch`): `es eval run --config <fixture eval.toml> --policy trained-{1000,5000,20000}.esb` gets
**past `TorchRuntime::load`** for all three and fails afterwards, inside `Evaluation::run`, on
the fixture suite's `light_intensity` perturbation. Getting past the load is the packet's claim
and it holds. **There is no success rate here**, and not for want of trying: the demo has no
`evaluation.toml` yet, and `Evaluation::run` still passes `frames: None`, so `capture` would
refuse the image input by name. Both are V3's.

The oracle's own end-to-end equality number, on the bundle it trains and packs itself:
`TorchRuntime::infer` against a direct PyTorch forward on a held-out observation gives
`max_abs = 0` at spec 8.9's tier-4 fp32 tolerance of 1e-5 — exactly equal, not merely within
tolerance. It covers the safetensors writer, `pack`'s validation, the `nodes.N` -> `nN` rename
and the wire protocol in one number.

Collecting the 55 demonstrations took under two minutes and the 20,000-step run about eleven,
both with rendering on. Those are **observations**, not performance claims, and no throughput
figure is derived from them (spec 12.4).
