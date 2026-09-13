# LeRobot policy `config.json` — shape and normalization stats

**Pinned version: NONE.** Nothing in this workspace pins `lerobot`; no Python package was
installed and no real `config.json` was read while writing this file. Every field below is
reconstructed from memory of LeRobot around the `v0.1`/`v0.2` `lerobot-train` era (roughly
the same generation as `docs/api-notes/lerobot-dataset.md`'s `codebase_version: "v2.1"`
datasets) and is **`unverified`** unless a line says otherwise. Spec §1.7 names exactly this
failure mode (환각 API): a human must pin a `lerobot` version and correct this file — and
`crates/es-data/src/lerobot_config.rs` if the shape actually differs — before anything here
is treated as ground truth.

Only the fields `crates/es-data/src/lerobot_config.rs` reads are modeled as struct fields;
everything else lands in a `#[serde(flatten)] extra: BTreeMap<String, Value>` bag and is
reported back as a conversion warning rather than silently dropped or rejected.

## Common shape — `unverified`

Every policy config is a JSON object with a `"type"` discriminator:

```json
{ "type": "act", ... }
{ "type": "diffusion", ... }
```

`crates/es-data/src/lerobot_config.rs` recognizes exactly `"act"` and `"diffusion"`; any
other value (`"smolvla"`, `"pi0"`, `"vqbet"`, …) is `ConfigError::Unsupported(type)` rather
than a guess (spec §14.4: severity=error blocks execution rather than fabricating a mapping).

`input_features` / `output_features` — `unverified`: `name -> { type, shape }`, where `type`
is one of `"VISUAL"`, `"STATE"`, `"ACTION"` and `shape` is the per-frame tensor shape
(channels-first for `VISUAL`, e.g. `[3, 224, 224]`). A real config's exact key set for ACT
(`observation.images.<camera>`, `observation.state`, `action`) is taken from
`docs/api-notes/lerobot-dataset.md`'s feature-name convention, not from a verified source.

`normalization_mapping` — `unverified`: `{ "VISUAL": "MEAN_STD", "STATE": "MEAN_STD",
"ACTION": "MEAN_STD" }`. Values seen in LeRobot source at various points: `MEAN_STD`,
`MIN_MAX`, `IDENTITY`. This crate models the three as `NormMode::{MeanStd,MinMax,Identity}`
but only ever *produces* `MeanStd`-shaped `Normalize`/`Normalizer` nodes today — an
`IDENTITY`/`MIN_MAX` config value is accepted (no error) and folded into a `Range`
normalization node with a warning, since the Observation IR's `NormalizeStats` has no
identity variant.

## ACT (`type: "act"`) — `unverified`

| field | type | note |
|---|---|---|
| `chunk_size` | int | prediction horizon `H` (spec §8.4) |
| `n_action_steps` | int | execution length `K` |
| `n_obs_steps` | int | almost always `1` for ACT |
| `vision_backbone` | string | e.g. `"resnet18"` |
| `pretrained_backbone_weights` | string \| null | e.g. `"ResNet18_Weights.IMAGENET1K_V1"` |
| `dim_model` | int | transformer width |
| `n_heads` | int | attention heads |
| `dim_feedforward` | int | |
| `n_encoder_layers` | int | |
| `n_decoder_layers` | int | |
| `use_vae` | bool | CVAE encoder toggle |
| `latent_dim` | int | |
| `temporal_ensemble_coeff` | float \| null | present => temporal ensembling at inference |
| `dropout` | float | |
| `kl_weight` | float | |
| `optimizer_lr`, `optimizer_weight_decay`, … | — | training-only; not modeled, carried in `extra`, reported as warnings |

## Diffusion Policy (`type: "diffusion"`) — `unverified`

| field | type | note |
|---|---|---|
| `horizon` | int | prediction horizon `H` |
| `n_action_steps` | int | execution length `K` |
| `n_obs_steps` | int | typically `2` |
| `crop_shape` | `[height, width]` \| null | center-crop applied before resize |
| `crop_is_random` | bool | random offset at train time, center at eval |
| `use_group_norm` | bool | |
| `down_dims` | `[int, ...]` | U-Net channel widths, e.g. `[512, 1024, 2048]` |
| `kernel_size` | int | |
| `noise_scheduler_type` | string | e.g. `"DDPM"`, `"DDIM"` |
| `num_train_timesteps` | int | default `100`; the grid the beta schedule is built over |
| `beta_schedule` | string | default `"squaredcos_cap_v2"`; `"linear"` is the other one we lower |
| `prediction_type` | string | default `"epsilon"`; `"sample"` / `"v_prediction"` are carried through and refused by the lowering |
| `num_inference_steps` | int \| null | defaults to `num_train_timesteps` when absent |
| `clip_sample` | bool | default `true` — `diffusers` clamps `pred_original_sample` |
| `clip_sample_range` | float | default `1.0` |

These six are `diffusers`' `DDPMScheduler` / `DDIMScheduler` configuration and all of them
reach `HeadKind::Diffusion` unchanged: a checkpoint trained under one schedule does not
reproduce under another. `variance_type` is **not** a LeRobot field, so the conversion pins
`diffusers`' own default, `fixed_small`. The defaults above are `unverified` — recalled from
`lerobot/common/policies/diffusion/configuration_diffusion.py`, not fetched; they are the
`serde` defaults in `es_ir::learning::HeadKind::Diffusion` and in `DiffusionConfig`, so a
config that omits a field still converts.

`noise_scheduler_type` maps to `es_ir::learning::DiffusionScheduler` as: `"DDPM" ->
Ddpm`, `"DDIM" -> Ddim`, anything else -> `DpmSolver` with a warning naming the unrecognized
value (this three-way split is this crate's own choice, not a LeRobot enum — `unverified`
whether LeRobot has other scheduler families in the version a human eventually pins).

## `crop_shape` / `crop_is_random` -> Observation IR — design note, not a LeRobot field

`crop_shape` becomes a `Crop { mode: CropMode::Center, rescale_intrinsics: true, .. }` node
(INV-14: intrinsics are rescaled, never left stale). `crop_is_random` additionally inserts an
**unwired** `Augment { kind: RandomCrop, training_only: true }` node into the same
`ObservationIr` graph, purely to record "training samples a random offset here" in a form
`Evaluation IR` structurally disables (spec §7.3, `INV-15`) — it does not feed the dataflow,
since a `CropMode::Random` sample offset does not change the nominal geometry the type system
tracks (see the doc comment on `es_ir::observation::CropMode::Random`) and the eval-time
graph is defined to run the plain center crop. A cross-IR check that a `cross_fixture`-style
Augment node is already known to trip is `XIR-050`.

## Dataset `stats` (`meta/stats.json`) — `unverified`

Same uncertainty as `docs/api-notes/lerobot-dataset.md`, which documents this file as
*ignored* by `es-data`'s dataset reader (normalization is Observation IR's business, not
dataset identity). This packet is the first consumer:

```json
{
  "observation.state": { "mean": [...], "std": [...], "min": [...], "max": [...] },
  "action":            { "mean": [...], "std": [...], "min": [...], "max": [...] },
  "observation.images.top": { "mean": [[[0.42]], [[0.41]], [[0.39]]], "std": [[[0.19]], ...] }
}
```

`crates/es-data::lerobot_config::Stats` models this as `BTreeMap<String, FeatureStats>` with
`FeatureStats { mean: Vec<f64>, std: Vec<f64>, min: Vec<f64> (default empty), max: Vec<f64>
(default empty) }` — flat, not the nested per-pixel-position array a video feature's stats
may actually carry (`unverified` whether image stats are per-channel `[3]` or per-pixel
`[3,1,1]`/full resolution; this crate's `Stats` loader expects a flat per-channel list and
will fail to parse a nested-array file, a follow-up fix once a real file is seen).

## What `convert()` does with `stats: None`

When no `Stats` is given, or a feature has no entry in it: image features fall back to
ImageNet constants (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` —
`unverified` whether this specific ResNet18 checkpoint was actually trained with these,
though they are the standard `torchvision` ImageNet values) with a warning; state/action
features fall back to the identity (`mean = 0`, `std = 1`) with a warning, since there is no
standard non-dataset default for an arbitrary joint-state vector.

## Physical units — not in the format, assumed by `convert()`

LeRobot's `config.json` and `stats.json` carry no unit or coordinate-frame metadata for
`observation.state` / `action` (spec §5.4's `Unit`/`Frame` have no LeRobot counterpart).
`crates/es-data/src/lerobot_config.rs` assumes joint-position control (`Unit::Angle` before
normalization, `es_ir::task::ActionSpace::JointPosition`/`es_ir::deployment::ActionSpace::
JointPosition` in the test fixture) and says so in a returned warning — this is the single
most common LeRobot setup (`so100`, ALOHA) but is a guess the compiler cannot check without a
Task IR to cross-reference, which a bare policy `config.json` conversion does not have.
