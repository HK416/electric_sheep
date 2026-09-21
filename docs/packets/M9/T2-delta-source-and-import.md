# M9 T2 — a delta-action source policy, and the adapter that imports it

Spec: §14.4 (the adapter declares `[action] kind`), §8.5 (increment units), §28.12 wave 1,
INV-16. Depends on **T1**. Precedent: S2c (`python/es/rl_source/`), S2b (`import_rl.py`,
`es policy import-rl`, `adapter-so101.toml`). Api-note to extend: `docs/api-notes/brax-ppo-so101.md`
(+ `.ko.md`), a new section for the delta env.

## the question

**Can the same brax stack train a reach policy whose action is a per-tick joint increment, and
can the importer bring it in as a `JointDelta` bundle that reproduces the source on 1,000
observations — with the adapter, not the code, saying that the action is an increment?**

## spec

* `python/es/rl_source/so101_reach_env.py` gains `--action delta`: `target_t = target_{t−1} +
  a · delta_scale` (`delta_scale` recorded in `meta.json`, e.g. 0.05 rad), target₀ = the reset
  pose; observation, reward, success, timeout unchanged (`rl-continuation.md` section 5); the
  derived MJX scene unchanged. `train_brax_so101.py --action delta` writes the same export
  (`source.npz`, `meta.json` with `action = { kind = "joint_delta", scale = … }`,
  `oracle-1000.npz`).
* `import_rl.py` passes `action.kind` through `import.json`; `adapter-so101-delta.toml` declares
  `[action] kind = "joint_delta"` and the increment unit; `es policy import-rl` refuses a manifest
  whose kind disagrees with the adapter (`IMP-004` already names this) and emits a Deployment IR
  reference with `action.space = JointDelta` and a `Normalizer{Inverse}` in increment units.
* Measured in the source framework: success rate over 64 episodes, and the per-tick command
  change distribution (the number T3 compares the clamp rate against).

## context

```
python/es/rl_source/**
python/es/import_rl.py
crates/es-data/src/rl_import.rs
crates/es-data/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/adapter-so101-delta.toml
tests/fixtures/rl/import/playground-delta/**
docs/api-notes/brax-ppo-so101.md
docs/api-notes/brax-ppo-so101.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M9/T2-delta-source-and-import.md
docs/packets/M9/T2-delta-source-and-import.ko.md
```

## oracle

1. Server: `train_brax_so101.py --action delta --seed 0` twice → `source.npz` bitwise (record);
   success rate over 64 episodes recorded (target 0.8; ship what is reached).
2. Server (`--ignored`): the import → (a) reconstruction vs our runtime bitwise on 1,000 rows,
   (b) our runtime vs JAX ≤ 1e-5 on `obs_scaled`; the bundle's Deployment IR says `JointDelta`.
3. `cargo test -p es --test cli import_rl_delta_fixture` — a synthetic `playground-delta` fixture
   → documents + bundle; the kind mismatch refused by name.
4. `cargo xtask ci`; `check-scope`.

## acceptance

Oracles 1–4; the api-note section and the measured rows (with server, date, path).

## forbidden

Changing the observation, reward or scene of section 5; pickle in Rust; guessing the action
kind (adapter or refusal); `docs/ARCHITECTURE*.md`; `tests/golden/**`.
