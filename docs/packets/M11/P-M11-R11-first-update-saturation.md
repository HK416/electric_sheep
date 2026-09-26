# M11 R11 — the pixel actor's first update, and a learning rate that does not saturate it

Spec: §13.4, §19.3 (`scheduler.json`, `optimizer.json`), §28.14 rule 7. Design note
`rl-continuation.md` section 7 (X7, R10) and S4c (the saturated `tanh`). Found by R10's diagnostic
(2026-09-26). On `rs-pix` the return falls from −8.9 to −16.1 between iteration 0 and iteration 1,
in X7's run and in R10's alike. By iteration 1,000 every channel of the head's pre-`tanh` value `z`
sits beyond |z| = 3 (mean about 20), where the untrained actor has |z| ≈ 0.24. The action is
pinned at a corner, and the gradient through `tanh` is under 1 %. The critic cannot move an actor
in that state, so R10's stage-1 rule stopped on a question that comes before it.

The recipe is the state task's: Adam at `lr = 3e-4`, 4 epochs × 4 minibatches, no gradient
clipping. On the state MLP it works. On an 11 M-parameter from-scratch ResNet18, Adam's early
steps move every parameter by about `lr` in the sign of its gradient, so the output moves in
proportion to the size of the network. This is the hypothesis. The packet measures it and does
not assume it. Type D (measurement; recipes only, no code).

## the question

**Which of a lower learning rate, a warmup and gradient clipping keeps the pixel actor's `tanh`
out of saturation through its first 20 iterations? With that setting and R10's privileged critic,
does pixel-only PPO learn `rs-pix`?**

## spec

Everything runs on the oracle server, with R10's code (`f775681` or later), its `diag.py` hook on
the head's `Linear`, and the GPU lock taken per GPU stage. Artifacts go under
`~/artifacts/plan-x/r11/`. Every arm is `rs-pix` with `critic = "privileged"`, seed 0. The recipes
are R10's `training-reach-vision-rs-pix-critic.toml` with only the named fields changed. They are
committed under `tests/fixtures/rl/`, and only the ones that reach stage 2 get plan goldens.

* **Stage A, probe** (20 iterations, `checkpoint_at = [1, 5, 20]`):

  | arm | change from R10's recipe |
  |---|---|
  | A0 | none (`lr = 3e-4`), the control that should reproduce the collapse |
  | A1 | `lr = 1e-4` |
  | A2 | `lr = 3e-5` |
  | A3 | `lr = 1e-5` |
  | AW | `lr = 3e-4`, `schedule = { kind = "warmup_cosine", warmup = <100 iterations of optimizer steps>, lr_min = 0 }` |
  | AG | `lr = 3e-4`, `grad_clip = 0.5` |

  For each checkpoint, record the mean |z| and the share of |z| > 3, both measured by `diag.py`
  on 16 envs × 64 steps from seed 0, the segment return and the training return. A0 must
  reproduce R10's collapse, or the stage is void and the report says why.
* **Rule A (fixed now):** an arm is **stable** if the share of |z| > 3 at iteration 20 is below
  0.05. Among the stable arms, the one with the largest learning rate at iteration 20 goes to
  stage B, and ties go to the simplest (constant `lr` first). If no arm is stable, stop.
* **Stage B, screen:** the chosen arm, 1,000 iterations. R10's rule applies against X7's
  `rs-pix-s0` over iterations 900–999: the arm passes if its mean return exceeds the baseline's by
  more than the larger of the two standard deviations. Report |z| at iterations 100 and 1,000 as
  well. If it does not pass, stop.
* **Stage C, rows:** the chosen arm × seeds {0, 1, 2} × `rs-pix` and `rs-dr-pix`, 4,000
  iterations, then held-out evaluation on 16 seeds against `evaluation-reach-vision-rs-pix.toml`
  and `evaluation-reach-vision-rs-dr-pix.toml`, suite by suite. Report the value loss, entropy,
  return, |z| at the final checkpoint, wall clock and the nine metrics.
* **Budget:** 25 GPU-hours. A projection above that stops the run before the next stage.

## context

```
tests/fixtures/rl/**
tests/golden/train/**
crates/es/tests/cli.rs
docs/design/rl-continuation.md
docs/packets/M11/P-M11-R11-first-update-saturation.md
docs/packets/M11/P-M11-R11-first-update-saturation.ko.md
```

## oracle

1. Stage A table, and rule A's decision with its numbers.
2. Stage B table (or the stop), and rule B's decision with its numbers.
3. Stage C table (or the stop).
4. If recipes reach stage C: `cargo test -p es --test cli` dry-run plan goldens, as additions only.
   verify-goldens with 0 modifications. `cargo xtask check-scope
   docs/packets/M11/P-M11-R11-first-update-saturation.md --base main`.

## acceptance

Oracles 1–4. `rl-continuation.md` section 7 gains R11 (English; the orchestrator does the `.ko.md`),
with one sentence answering each question.

## forbidden

Code changes (a bug found is reported, not fixed here); any envelope or `crates/es-safety`; `Pt`
rows; any PPO field beyond `lr`, `schedule` and `grad_clip`; single-seed claims from stage C;
`git clone`/`pull` on the server; pushing.
