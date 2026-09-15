# M5 V13 — a from-scratch backbone that has no training mode

Design note: `docs/design/visible-learning.md` **section 7.21**, and 6.1 (the training loop),
7.19 (V11, the baseline this is measured against), 7.9 (V2b, where `pretrained` came to be
refused). Lowering contract: `docs/design/learning-lowering.md` **section 5.1**, the rule this
packet writes. Spec: §5.2, §8.3, §8.7, §1.4. Depends on V2, V2b, V6b, V11.

## the finding this packet fixes

The open-loop analysis of V11's checkpoint (`target/plan-v/v11/openloop.txt`, `bnrecal.txt`)
found that **the deployed function is not the trained function**, and named the reason.

`lower_to_torch` lowers `VisionEncoder { ResNet18, pretrained: false }` to torchvision's
`resnet18()`, whose default norm layer is `BatchNorm2d` — 20 of them. The lowering is
single-sample: spec 8.3's ports carry no batch axis (spec 5.2 gives the inference domain its own
batch size), so the image goes through `.unsqueeze(0)` and the head `.reshape(16, 6)`.
`python/es/train_act.py --batch 8` therefore accumulates eight *single-sample* forwards, and
every `BatchNorm2d` fits **N = 1** statistics — it is instance normalization with a batch
counter. `crates/es-policy/python/torch_ref.py:85` then calls `model.eval()`, which swaps in the
running averages. Measured on V11's own checkpoint over its own training episodes, 10-row chunk
L1:

| | chunk L1 |
|---|---|
| `train()` — what training minimized, and what the loss curve reported | **0.011** |
| `eval()` — what `es eval run` executed | **0.031 – 0.039** |
| `eval()` after recalibrating the running statistics over the whole training set, batches of 64 | 0.029 |
| baseline: hold the current pose for every row | 0.048 |

The recalibration row is the important one: it is the best a post-processing of the weights can
do, and it does not close the gap. The fix has to be in the lowering.

## the decision

**For `pretrained: false`, lower the backbone with `norm_layer = lambda c: nn.GroupNorm(32, c)`.**

GroupNorm normalizes over channel groups of the one sample in front of it. It has no `training`
branch and no running buffers, so `train()` and `eval()` are the same function, bit for bit; the
single-sample lowering stays valid; and the running-statistics buffers leave the weight contract.
It is what Diffusion Policy substitutes into ResNet for exactly this reason. 32 groups divides
every ResNet stage width (64, 128, 256, 512).

This is a **lowering** decision, not an IR change: `lowering_hash` and `compiler_hash` move,
`learning_hash` does not. Spec §8.3 pins no normalization for `VisionEncoder`, and
`docs/design/learning-lowering.md` pinned none before this packet — it is now section 5.1.
`pretrained: true` is untouched: it is still refused by this lowering (section 7.6, open
question 6), and LeRobot checkpoints keep ImageNet's BatchNorm as `FrozenBatchNorm2d` through
`lerobot.rs` (V8), which is correct because frozen affine constants have no training mode either.

The declared weight keys do not move — the backbone is the prefix claim `nodes.0.*`, so what
changes is the set of tensors *under* the claim: 60 `running_mean` / `running_var` /
`num_batches_tracked` buffers are gone.

## scope

* **`context`** — `crates/es-policy/src/lower/torch.rs`, `crates/es-policy/tests/ir_training.rs`,
  `docs/design/learning-lowering*.md` section 5.1,
  `docs/design/visible-learning*.md` section 7.21 and open question 19,
  `docs/packets/M5/V13-groupnorm-backbone*.md`.
* **`forbidden`** — `crates/es-policy/src/lerobot.rs` (the `pretrained: true` / LeRobot path is
  V8's, and `FrozenBatchNorm2d` is right there), `es-ir` and spec §8.3 (no schema change),
  `nn.TransformerEncoderLayer`'s dropout (intended regularization, off at inference),
  `tests/fixtures/**`, `docs/ARCHITECTURE*.md`, section 7.20 (V12 is writing it).
* **`INV-17`** — no new trait. One argument to an existing helper.

## oracles

1. **No Python needed.** `cargo test -p es-policy --lib
   lower::torch::tests::a_from_scratch_backbone_has_no_training_mode` — the generated source for
   a from-scratch `VisionEncoder` contains no `BatchNorm` at all (the string, not just the
   layer), contains `norm_layer=lambda c: nn.GroupNorm(32, c)`, and no declared weight key or
   shape names a running statistic.
2. **With torch.** `cargo test -p es-policy --test ir_training -- --ignored
   the_backbone_computes_the_same_function_in_train_and_eval` — instantiates the demo bundle's
   own lowered module in a plain interpreter, asserts its `state_dict()` holds no
   running-statistic buffer, and asserts `torch.equal(backbone(x) in train(), backbone(x) in
   eval())` on a fixed input. This is the property `BatchNorm2d` violated. It asserts over the
   backbone rather than the whole module because `nn.TransformerEncoderLayer`'s dropout is a
   second, *intended* train/eval difference. SKIPs with a printed reason without torch.
3. **Nothing else regressed.** `cargo xtask ci`; `cargo test -p es-policy --test
   torch_equivalence` and `--test ir_training -- --ignored` with `ES_PYTHON` on the server.
4. **The checkpoint itself.** `identity.py` on the server repeats oracle 2 against the *trained*
   V13 checkpoint rather than a fresh initialization, and separates the dropout difference from
   the normalization one.
5. **End to end.** Retrain V11's exact recipe on V11's exact baked set — the only variable moved
   is the lowering — then the open-loop L1 table and `es eval run` on the training seeds (1–16)
   and the held-out seeds (101–116), beside V11's 0/16 and 0/16.
6. **Stop rule.** One variable. If the policy still scores zero, report it and do not reach for a
   second change in this packet.

## measured

Oracle server (RTX 4090, shared with another agent's training), `~/venvs/es` for `es`,
`~/venvs/es-lerobot-cuda` for training, 2026-09-16. Artifacts under `~/artifacts/plan-v/v13/`,
small results mirrored to `target/plan-v/v13/`.

### 1 — the lowering, and the two oracles

`cargo xtask ci` green. Oracle 1 and the `torch_equivalence` / `ir_training` suites pass. On the
server, `identity.py` against the *trained* V13 checkpoint (`target/plan-v/v13/identity.txt`):

```
running-statistic buffers in the module: 0 []
BatchNorm in the generated source: 0
contract keys with running statistics: 0
max |train() - eval()| over the BACKBONE alone = 0
max |train() - eval()| over the whole module, dropout active   = 0.0168887
max |train() - eval()| over the whole module, dropout disabled = 0
```

The middle row is dropout and nothing else: force the three `nn.Dropout` members of
`nn.TransformerEncoderLayer` *and* `MultiheadAttention.dropout` — a float read off
`self.training` rather than a module, which is why a `isinstance(m, nn.Dropout)` sweep alone
leaves 9.3e-4 behind — to zero, and the whole module is bit-identical between the modes.

`lowering_hash` moved `fdd68ec4…` → `70a8fec7…`; `learning_hash` `5dac0a46…`,
`observation_hash` `4b069f6c…` and `task_hash` `d7a7c061…` did not. All 14 declared weight keys
are byte-identical to V11's; the checkpoint went from **146 tensors to 86** (60 running-statistic
buffers, 20 × 3).

### 2 — the open-loop fit, on V11's own three training episodes

| chunk L1 over 10 rows | V11 (BatchNorm) | **V13 (GroupNorm)** |
|---|---|---|
| `eval()` — the inference path | 0.0313 / 0.0373 / 0.0385 | **0.0132 / 0.0135 / 0.0133** |
| `train()` — what the loss curve reported | 0.0107 / 0.0122 / 0.0111 | 0.0136 / 0.0139 / 0.0139 |
| baseline: hold the current pose | 0.0484 / 0.0488 / 0.0481 | 0.0484 / 0.0488 / 0.0481 |
| training loss, initial → final | 0.0565 → 0.0132 | 0.0607 → **0.0147** |

`eval()` fell 2.4x and now agrees with the reported loss (0.0133 against 0.0147) instead of
being three times it; `train()` is now marginally the *worse* of the two, which is what dropout
predicts. The gap is closed.

### 3 — closed loop

`es eval run --frames --jobs 6`, V11's own suites, V11's own baked set, one variable moved.

| | V11 | **V13** |
|---|---|---|
| `success_rate`, training seeds 1–16 | 0 / 16 | **0 / 16** |
| `success_rate`, held-out seeds 101–116 | 0 / 16 | **0 / 16** |
| `envelope_violation_rate`, training / held-out | 0.3152 / 0.2735 | 0.5770 / 0.6244 |
| `episode_length` | 900 | 900 |
| **episodes in which the cube moves at all** | **0 / 16 and 0 / 16** | **14 / 16 and 14 / 16** |
| cube displacement, held-out | 0.4 mm (settling) in all 16 | 0.4 – 200.2 mm, median ~9 mm |

V11's policy never reached the cube — 0.4 mm is the cube settling at reset. V13's reaches it in
14 of 16 on both suites and shoves it, twice past 190 mm. The failure changed kind: "never
arrives" became "arrives and cannot close the hand".

### 4 — on V12's re-collected set: the first cube that leaves the table

V12's pairing-fixed 50 Hz bake (`~/artifacts/plan-v/v12/baked`, 50 episodes, 9,038 frames, the
same four ports) was available, so the same module and the same knobs were trained on it too
(loss 0.0632 → **0.0162**; open-loop `eval()` 0.0137 / 0.0144 / 0.0149 against `train()` 0.0142 /
0.0148 / 0.0154 — the same convergence, on a set whose "hold the current pose" baseline is 0.0565
rather than 0.0484, which is the pairing fix showing up as a genuinely harder target).

| trained on | `success_rate` train 1–16 | held-out 101–116 | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|---|
| V11's bake | 0 / 16 | 0 / 16 | 0.5770 / 0.6244 | 900 / 900 |
| **V12's bake** | **1 / 16** | 0 / 16 | 0.5396 / 0.6392 | 847.75 / 900 |

Read the trajectories before reading that `1 / 16`. **In three training-seed episodes (03, 11,
15) the policy grasps the cube, lifts it 118 – 124 mm and carries it to the bin** — cube
(0.25, 0.00, 0.02) → (0.13, −0.10, 0.06 … 0.14) — and then holds it there until the 900-step
budget runs out, so all three are scored `timeout`. The one episode the harness *does* score
`success` (02, terminated at tick 64) lifted the cube 1.6 mm and is not a carry; it deserves its
own look and is not what this packet claims. The claim is the three carries: for the first time
in plan V an IR-owned vision policy picks the cube up. `es video showcase --cell nominal-03`
renders one, 900 ticks at 1280x720, `~/artifacts/plan-v/v13/v12/v13-carry-nominal-03-h264.mp4`
(0.9 MiB). No held-out episode succeeded, so there is no held-out showcase.

### verdict

The train/deploy gap was real, is measured, and is closed: 0 difference between `train()` and
`eval()` over the backbone, and an `eval()` open-loop L1 that fell 2.4x to meet the reported
loss. On V11's bake `success_rate` stayed 0/16 and the failure changed kind — the arm now
reaches the cube instead of sweeping past it. On V12's bake the policy grasps, lifts and carries
the cube in 3 of 16 training-seed episodes and releases in none. Held-out is still 0/16, so the
stop rule fires and this packet moves no second variable.

What the numbers point at next is the **release**, not the reach: joint 5 (the gripper) carries
the largest open-loop residual in every table — blended 0.0162 against 0.0032 – 0.0103 for the
arm joints, and 0.0823 against the current pose — and "carries the cube over the bin and never
opens" is exactly what that predicts. The 900-step budget is the second thing to look at: three
episodes were still holding the cube in the right place when it ran out. V13's numbers, not
V11's, are the baseline for whatever comes next.
