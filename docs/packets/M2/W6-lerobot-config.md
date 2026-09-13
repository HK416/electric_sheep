# W6 — LeRobot policy config -> IR conversion (Rust half)

Spec: §14.4 (외부 변환: LeRobot 변환이 v1.0의 최우선 변환 타깃), §14.2 (Python 빌더가 만드는
Observation IR / Learning IR의 모양 — 변환 결과가 이것과 어긋나면 안 된다), §7.4/§7.5
(`ObservationSpec`과의 연결, 시간 모델 3계층), §8.4 (`PolicyContract`), §5.1 (IR 경계),
§28.5 W6 (`docs/ARCHITECTURE.ko.md` 표: "Python 빌더 완성, LeRobot config 변환").

**This packet is the Rust half only.** The Python builder (PyO3, `es-py`) that spec §14.2
describes is the *other* half of W6 and is **not implemented here** — it is a separate
packet against `crates/es-py`, out of `crates/es-data`'s scope.

## context

```
crates/es-data/src/lerobot_config.rs
crates/es-data/src/lib.rs                  (+ `pub mod lerobot_config;`)
crates/es-data/Cargo.toml                  (+ `es-math` dependency, for `ImageSpec::extrinsics`)
crates/es-data/tests/lerobot_config.rs
tests/fixtures/lerobot_config/act_config.json
tests/fixtures/lerobot_config/diffusion_config.json
tests/fixtures/lerobot_config/stats.json
tests/fixtures/lerobot_config/unknown_type.json
docs/api-notes/lerobot-config.md
docs/packets/M2/W6-lerobot-config.md
```

## spec

No `lerobot` version is pinned in this workspace (§1.7): `docs/api-notes/lerobot-config.md`
is written first and marks every `config.json` / `stats.json` field `unverified`, same
standard as `docs/api-notes/lerobot-dataset.md`.

1. `LeRobotPolicyConfig` (serde, `#[serde(tag = "type")]`, `act` / `diffusion`) with
   `LeRobotPolicyConfig::parse(json) -> Result<Self, ConfigError>`: an unrecognized `"type"`
   is `ConfigError::Unsupported(type)`, never a guessed mapping (§14.4: a Semantic Mapping
   Report's `severity=error` blocks execution). Fields this crate does not model land in a
   `#[serde(flatten)] extra: BTreeMap<String, Value>` bag per variant and are surfaced as
   [`Converted::warnings`] — never silently dropped, never a hard error.

2. `Stats` (`BTreeMap<String, FeatureStats>`) models `meta/stats.json`'s per-feature
   mean/std/min/max, the first consumer of a file `docs/api-notes/lerobot-dataset.md`
   documents as otherwise *ignored*.

3. `convert(cfg, stats, dataset_info) -> Result<Converted, ConfigError>` builds one consistent
   `ObservationIr` + `LearningGraph`:
   - Each `VISUAL` input feature: `ImageInput -> [Crop (Diffusion `crop_shape`, `CropMode::
     Center`, `rescale_intrinsics = true`, INV-14)] -> Resize -> Normalize (`MeanStd` from
     `stats`, or the `ImageNet`/identity fallback named in the api-note) -> [`TemporalWindow`
     node + `History` entry when `n_obs_steps > 1`]`. `crop_is_random` additionally inserts an
     unwired, `training_only` `Augment::RandomCrop` node (§7.3/INV-15: `Evaluation IR`
     structurally disables it, leaving the deterministic centred crop as what eval/sim/real
     run).
   - The one `STATE` input feature: `StateInput -> Normalize -> [TemporalWindow]`, same
     windowing rule.
   - Learning IR mirrors `crates/es-policy`'s expected lowering (§8.3): one `VisionEncoder`
     per camera + one `StateEncoder` -> `Fusion(Concat)` -> `TemporalEncoder` (`None` at
     `n_obs_steps = 1`, `Transformer` above it) -> `PolicyHead` (`Regression` for ACT,
     `Diffusion { n_steps, scheduler }` for Diffusion Policy) -> `ActionChunker` ->
     `Normalizer(Inverse)`. `PolicyContract` is filled per §8.4: `observation_window =
     n_obs_steps`, `horizon = chunk_size`/`horizon`, `execute_chunk = n_action_steps`,
     `action_dim` from the `ACTION` output feature, `replanning_hz` from `dataset_info.fps`
     (a warning when absent, not a guess presented as fact).
   - The converted pair passes `ObservationIr::validate`, `LearningGraph::validate`, and (the
     test file builds a minimal synthetic Task IR / Deployment IR around it, since
     `es_ir::cross::check` takes a full `IrBundle`) the Observation<->Learning boundary rules
     of `es_ir::cross::check`.

## oracle

```
cargo fmt -p es-data --check
cargo clippy -p es-data --all-targets -- -D warnings
cargo test -p es-data
cargo xtask check-spec-refs
```

## acceptance

- `act_config.json` and `diffusion_config.json` (hand-written, < 10 KB, representative of
  `lerobot`'s ACT / Diffusion Policy `config.json` shape — every field `unverified`) both
  convert, validate clean per-IR, and cross-check clean for the Observation<->Learning
  boundary against a generic synthetic Task/Deployment IR pair built from the result.
- `n_obs_steps > 1` registers a `History` depth per stream and sets
  `ObservationIr::temporal.window`.
- Diffusion's `crop_shape` produces a `Crop` node with `rescale_intrinsics = true` and a
  principal point that has moved (INV-14); `crop_is_random` additionally produces a
  `training_only` `Augment` node.
- `unknown_type.json` (`"type": "smolvla"`) is `ConfigError::Unsupported`, not a panic and not
  a best-effort ACT/Diffusion guess.
- An unrecognized `config.json` field (e.g. `optimizer_lr`) is a warning string in
  `Converted::warnings`, not an error and not silently dropped.
- `convert()` run twice on the same input produces identical `observation_hash` /
  `learning_hash` (spec §5.3 determinism).

## forbidden

- The Python builder (PyO3, `es-py`) — a separate packet, not this one.
- `crates/es-env`, `crates/es-eval`, `crates/es-policy`, `crates/es-compile`, `crates/es`
  (owned by concurrent M2 packets) and the root `Cargo.toml`.
- `HashMap`/`HashSet` (§3.4 determinism) — `BTreeMap` only.
- A new extension-point trait (INV-17); this packet adds none.
- Committing — the oracle is run and reported, not landed, by this packet.
