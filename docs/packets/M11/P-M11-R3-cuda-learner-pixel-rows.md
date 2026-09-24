# M11 R3 — the PPO learner on CUDA, and vision rows that see the cube only through pixels

Spec: §28.14 rules 5, 7, §3.5 (a CUDA learner is not tier 1; the CPU stays the bitwise one),
§12.4. Found by X7 (2026-09-24):

- `python/es/train_ppo.py --device cuda` dies at the first forward ("mat1 is on cpu"). The
  actor is moved to the device, but the observations and the rollout buffers are not. On the CPU,
  the ResNet18 update is 21.8 of the 23.4 s an iteration takes, which puts X7 stage 2 at about
  305 GPU-hours.
- X7's vision rows keep the 26-wide `state` port with `cube_pose`, so they cannot tell whether
  PPO learns from pixels.

Type B.

## the question

**Does `train_ppo.py` train on `--device cuda` with the same recipe, the CPU path's bytes
unmoved, and how many seconds does an X7 `pt-dr` iteration take then? And can each X7 row exist
without the cube's pose in its state, so the camera is the only way to see the cube?**

## spec

* `train_ppo.py`: every tensor that meets the actor or the value net lives on `device`. That
  covers observations (the state and image ports), actions, log-probs, advantages, returns and
  the minibatch index. The noise and order generators stay on the CPU (their draws do not
  change); a draw is moved to the device after it is made. Anything that is written to disk
  (checkpoint, metrics) is moved back to the CPU first. `--device cpu` produces the same bytes
  as today.
* X7 documents: each of the four rows gets a `-pix` sibling. Its observation and learning
  documents use a state port without `cube_pose` (the joints and the gripper pose, 19 wide),
  and the image port is unchanged. They are generated with X7's `regenerate_x7_documents`
  (extended), and plan goldens are added.

## context

```
python/es/train_ppo.py
crates/es/tests/cli.rs
tests/fixtures/rl/**
tests/golden/train/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/P-M11-R3-cuda-learner-pixel-rows.md
docs/packets/M11/P-M11-R3-cuda-learner-pixel-rows.ko.md
```

## oracle

1. The existing `train_rl_*` cli tests and the PPO bitwise tests pass unchanged (the CPU bytes
   are unmoved).
2. On the server (GPU lock): 3 iterations of the X7 `pt-dr` and `pt-dr-pix` rows with
   `--device cuda` complete. Record s/iteration split into render, rollout and learner. Then
   20 iterations on cuda vs cpu from one seed: the return curves are within PPO noise (report
   both, since this is not a bitwise claim).
3. `cargo test -p es --test cli vision_reach_dr_documents_check` covers the `-pix` rows. Their
   dry-run plan goldens are additions only.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

Oracles 1–4. `rl-continuation.md` (+ko) X7 section gains the measured cuda s/iteration and the
new stage-2 estimate.

## forbidden

Changing the PPO recipe or its hyperparameters; changing any committed training golden; es-py or
Rust runtime changes (a need for one is a new packet).
