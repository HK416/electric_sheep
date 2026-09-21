# M8 S2b — `es policy import-rl`: a PPO actor becomes a bundle through an adapter document

Spec: §14.4 (the paragraph "RL policy import": Python half → `safetensors` + `import.json`, `es`
half → documents + bundle; the Semantic Mapping Report), §13.4 ("the import never guesses"),
§8.9 (equivalence tiers), §28.11 wave 2, INV-16. Depends on **S2a** (activation / squash) and
**S2c** (the real checkpoint and `oracle-1000.npz`). Precedent: `es import lerobot-config`
(`crates/es/src/cmd/import.rs`, `es_data::lerobot_config::convert`) and `es policy import-lerobot`
(`crates/es/src/cmd/policy.rs`, `es_policy::lerobot`). Oracle tiers: `docs/design/rl-continuation.md`
section 4. Design note to extend: `rl-continuation.md` section 7 (measured) and a new section
"The importer and the adapter" (+ `.ko.md`).

## the question

**Can an actor trained by brax, rsl_rl or rl_games be turned into a Learning IR + Observation IR
+ bundle whose runtime output reproduces the source's deterministic action on 1,000 random
observations — with every joint, unit and channel mapping declared in a document and every
mismatch refused by name?**

## spec

* **Python half** `python/es/import_rl.py --from mujoco-playground|rsl-rl|rl-games --checkpoint
  <path> [--activation elu|swish|relu|tanh] --out <dir>` → `weights.safetensors` (neutral keys
  `mlp.<i>.weight|bias` `[out, in]`, `head.weight|bias`, `log_std`) and `import.json`
  (`framework`, `versions`, `obs_dim`, `action_dim`, `hidden`, `activation`, `activate_output`,
  `squash`, `obs_mean`, `obs_std`, `log_std`, `action_scale`, `action_offset` — `null` when the
  framework does not carry it, which the adapter must then supply).
  - `mujoco-playground`: S2c's `source.npz` / `meta.json`, or an orbax checkpoint via
    `brax.training.checkpoint.load`; the mean half of the output Dense; `squash = tanh`,
    `activate_output = true`, `activation = swish`.
  - `rsl-rl`: `torch.load(model_*.pt)` (both the old `model_state_dict` with top-level `std` and
    the ≥ 5.0 `actor_state_dict` layouts); `actor.<i>.weight`; optional `obs_normalizer`
    (`EmpiricalNormalization` mean / var → std); activation from `--activation` (the file does not
    carry it — refuse without the flag).
  - `rl-games`: `.pth` `model` dict, `a2c_network.actor_mlp.<i>`, `a2c_network.mu`,
    `running_mean_std.running_mean|running_var`; activation from `--activation`.
  - pickle / orbax open **only here** (INV-16).
* **Rust half** `es policy import-rl --manifest <import.json> --weights <w.safetensors> --adapter
  <adapter.toml> --task <task.toml> --deployment <d.toml> --out <dir>` writes `observation.toml`
  (`StateInput` per adapter channel → `Concat` → `Normalize{MeanStd}` from the manifest, a `Range`
  identity if the manifest has none), `learning.toml` (`StateEncoder{Mlp hidden, activation,
  activate_output}` → `PolicyHead{Regression, horizon 1, squash}` → `Normalizer{Inverse, MeanStd:
  mean = offset, std = scale}`), `policy.esb` (the pack path, weights remapped to the lowered
  module's `weight_keys`), and `mapping-report.json` — the §14.4 Semantic Mapping Report: one row
  per joint and per observation channel (source index, our name, unit, severity).
* **`adapter.toml`** (`deny_unknown_fields`): `[robot] name`; `[joints] source_order = [...]`
  (the framework's), `units = "rad" | "deg"`; `[action] kind = "position_target" | "torque"`,
  optional `scale` / `offset` overriding the manifest; `[observation] channels = [{ source = "qpos",
  slice = [0, 6], channel = "joint_pos" }, …]` naming the Task IR `ObservationSpec` channel each
  slice feeds. The committed one: `tests/fixtures/rl/adapter-so101.toml`.
* **Refusals, named** (`IMP-001` … in `es-data` or wherever `convert` lives): joint count ≠
  Task IR `ActionSpec.dim`; a joint name not in the scene's actuators; `units = deg`; `kind` not
  matching `ActionSpec.space`; channel slices that do not tile `obs_dim` exactly or name an
  undeclared channel.
* Synthetic fixtures for CI (no Python at test time): `tests/fixtures/rl/import/{playground,
  rsl-rl,rl-games}/{import.json,weights.safetensors}` — 15 → [8, 8] → 6, generated once by
  `import_rl.py --synth <framework>` and committed (a few KB each).

## context

```
python/es/import_rl.py
crates/es-data/src/rl_import.rs
crates/es-data/src/lib.rs
crates/es-data/tests/**
crates/es/src/cmd/policy.rs
crates/es/tests/cli.rs
tests/fixtures/rl/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M8/S2b-import-rl.md
docs/packets/M8/S2b-import-rl.ko.md
```

## oracle

1. `cargo test -p es --test cli import_rl_synthetic_three_frameworks` — each fixture → the two
   documents validate, the bundle opens in `TorchRuntime` (SKIP the open without `ES_PYTHON`),
   `mapping-report.json` has one row per joint and channel.
2. `cargo test -p es-data import_rl_refusals` — the five `IMP-0xx` by name.
3. Server (`--ignored`, `ES_PYTHON`): S2c's checkpoint → bundle; on `oracle-1000.npz`: (a) the
   importer's numpy→torch reconstruction vs our runtime **bitwise** f32; (b) JAX's actions vs our
   runtime max abs error ≤ 1e-5, the value recorded. For rsl_rl / rl_games: a tiny actor trained
   for a few seconds in the source framework in the venv (or a random-weights actor saved by the
   framework's own `save`) → our runtime **bitwise**.
4. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M8/S2b-import-rl.md`.

## acceptance

Oracles 1–4; the design note's new section and the measured rows (hashes of the imported
bundle, the max abs error, dates, paths).

## forbidden

Guessing any mapping the adapter did not declare; pickle in Rust (INV-16); a new node type;
touching `es-policy/src/lerobot.rs`; `docs/ARCHITECTURE*.md`; `tests/golden/**`; INV-17.
