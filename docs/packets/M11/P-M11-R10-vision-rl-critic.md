# M11 R10 — pixel PPO with a privileged critic, and an ImageNet encoder arm

Spec: §13.4, §19.3 (`training/`, `base_model.lock`), §28.14 rules 1 and 7, INV-12. Design note
`rl-continuation.md` rules 1–3 and section 7 (X7). Found by the orchestrator on 2026-09-26, reading
X7's loss curves against the state row's:

| run (seed 0) | value loss, iter 0 → 1,000 → end | entropy, iter 0 → end | return, iter 0 → end |
|---|---|---|---|
| state reach (S4e, 4,000 it) | 8.4 → 0.57 → 0.40 | 5.52 → 4.87 | −15.8 → −4.7 |
| `rs-pix` (X7, 4,000 it) | 0.42 → 2.3 → 29.7 | 5.52 → 12.64 | −8.9 → −17.6 |

The state row's value function fits and its entropy falls. The pixel row's does not fit, and its
entropy climbs from the first iteration. With advantages that carry no signal, the entropy bonus is
the only consistent gradient on `log_std`. The cause is in the recipe, not in the envelope. Under
X7, `train_ppo.py`'s `Value` is a 64-64 tanh MLP over the flattened observation, so on the `-pix`
rows its first layer reads 27,648 raw pixels (its own docstring says so, and leaves a `ponytail:`
note for exactly this case). The actor is a from-scratch ResNet18 learning from those advantages.
The same envelope clamps the state row on every step (`executed_ne_sampled_rate` 1.0), and that
row learns to 0.42. So S-1 does not explain the pixel row.

The literature fix is the asymmetric actor-critic (Pinto et al., RSS 2018, arXiv:1710.06542). The
actor sees pixels, and the critic sees simulator state that only training has. MuJoCo Playground's
vision `PandaPickCubeCartesian` (arXiv:2502.08844) uses it with the same randomization groups as X5,
and Isaac Lab's rsl_rl does the same through a `critic` observation group. The critic is not
deployed, so this stays inside `rl-continuation.md` rule 1: the value network is training-only
state, it moves `training_hash` and never `learning_hash`. Type C (code + measurement).

## the question

**With a critic that reads privileged simulator state, does pixel-only PPO learn the reach task on
the rasterizer? Does a frozen ImageNet ResNet18 actor encoder learn faster than the from-scratch
one?**

## spec

### code

* **`[rl] critic`** in the training recipe (`crates/es-data/src/training.rs`), with
  `[rl] estimator` as the precedent in every respect: an enum `observation | privileged`.
  Absent means `observation`, which is today's behaviour, and it is not serialized, so no
  committed `training_hash` or plan golden moves. It reaches `train_ppo.py` as `--critic
  privileged`, and `optimizer.json` (or whichever `training/` file already carries `estimator`)
  records it.
* **`train_ppo.py --critic privileged`**: the `Value` MLP (same two hidden layers of 64) reads the
  concatenation of every **non-image** observation port and `Rollout.qpos(env)` for each env. That
  includes the cube's free joint, which is the privileged part. `--critic observation` is today's
  code path, byte for byte. Image ports never reach a privileged critic. The value network is still
  written to `--value-out` and never into the bundle. If `Rollout` lacks something the critic needs
  (for example `qvel`), do not add it. `qpos` plus the state port is the design.
* **`--init-backbone` on the RL path**: if `es train` with `[rl]` refuses `[policy] base_model`,
  or `train_ppo.py` does not load it, wire it the way `train_act.py` does (reuse its function,
  never a second copy), with `base_model.lock` written as for `train_act.py`. If this already
  works, say so and change nothing.
* **Documents:** `tests/fixtures/rl/learning-reach-vision-pix-imagenet.toml`, which is
  `learning-reach-vision-pix.toml` with `pretrained = true, frozen = true`, generated the way the
  other documents are generated, not hand-edited. If the ImageNet input normalization is not
  already applied between the `Normalized{0,1}` port and a pretrained backbone, report that and
  stop the arm. Do not add a normalization to the IR in this packet. Also a training recipe per
  arm under `tests/fixtures/rl/`, which is X7's `rs-pix` recipe plus `critic = "privileged"` (and
  `base_model` for the ImageNet arm).

### measurement (oracle server, GPU lock per GPU stage, `nohup` with marker files, artifacts under `~/artifacts/plan-x/r10/`)

The rule is written here, before anything runs.

* **Stage 1, screen:** `rs-pix`, seed 0, 1,000 iterations, `--device cuda`.
  * Arm P is the privileged critic with the from-scratch encoder.
  * Arm PI is the privileged critic with the frozen ImageNet encoder.
  * The baseline is X7's `rs-pix-s0` curve over its first 1,000 iterations, which already exists
    and is not rerun.
  * An arm **passes** if its mean return over iterations 900–999 exceeds the baseline's mean over
    the same iterations by more than the larger of the two standard deviations over that window.
  * The passing arm with the higher mean goes to stage 2. If both are within one standard deviation
    of each other, the cheaper arm per iteration goes. If no arm passes, stop: that is the result,
    and the next packet is state-to-pixel DAgger (arXiv:2412.13662).
* **Stage 2, rows:** the chosen arm × seeds {0, 1, 2} × `rs-pix` and `rs-dr-pix`, 4,000
  iterations, X7's recipe otherwise.
  * Held-out: 16 seeds on `evaluation-reach-vision-rs-pix.toml` and
    `evaluation-reach-vision-rs-dr-pix.toml` with their perturbation suites.
  * Report per run: success rate per suite, final value loss, final entropy, return, wall clock,
    and the nine metrics.
* **Budget:** 30 GPU-hours total. If a projection goes above that, the run stops before the next
  stage. The `Pt` rows are not part of this packet: if `rs` learns, running `Pt` is the owner's call.

## context

```
crates/es-data/src/training.rs
crates/es-data/tests/**
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
python/es/train_ppo.py
python/es/train_act.py
python/es/selfcheck.py
tests/fixtures/rl/**
tests/golden/train/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/P-M11-R10-vision-rl-critic.md
docs/packets/M11/P-M11-R10-vision-rl-critic.ko.md
```

## oracle

1. `cargo test -p es-data`: a recipe without `critic` serializes byte-identically to today's, and
   `critic = "privileged"` moves `training_hash`. Existing plan goldens are unchanged; new ones are
   additions only.
2. `cargo test -p es --test cli` gains a CPU test: `es train` on a tiny `-pix` recipe with
   `critic = "privileged"` for 2 iterations, run twice, gives bitwise-equal checkpoints and value
   files. The value's first layer is `non-image state width + nq` wide, which the test reads from
   the value safetensors.
3. `train_rl_two_runs_are_bitwise` and every other existing RL test pass unchanged (the
   `observation` path is today's bytes).
4. Stage 1 table, then the stage-2 table (or the stop, with the rule's numbers).
5. `cargo xtask ci` with `ES_PYTHON` set, verify-goldens with 0 modifications, and
   `cargo xtask check-scope docs/packets/M11/P-M11-R10-vision-rl-critic.md --base main` (once
   P-M11-R9 has landed; before that, the diff is checked by hand).

## acceptance

Oracles 1–5. `rl-continuation.md` (+ko) section 7 gains R10 with the two tables and one sentence
answering each question.

## forbidden

Anything in `crates/es-safety` or any deployment document's envelope (S-1 is the owner's decision).
The deployed actor's architecture beyond the one `-imagenet` document. Retuning PPO hyperparameters
per arm. Single-seed claims in stage 2. Running `Pt`. `git clone`/`pull` on the server (scp only).
Pushing.
