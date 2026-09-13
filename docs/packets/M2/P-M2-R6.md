# P-M2-R6 — the DDPM schedule against `diffusers` (`es-policy`)

Spec: spec 8.3 (`HeadKind::Diffusion` parameters), spec 8.7 (lowering), spec 8.9 (tier 4),
spec 1.4 (the oracle comes before the implementation, and it must be *independent*), spec 3.4
(determinism, no global RNG), spec 5.3 (hash chain), spec 14.4 (LeRobot config).
Design note: `docs/design/learning-lowering.md` section 8 (review class C).
API notes: `docs/api-notes/torch.md`, `docs/api-notes/lerobot-config.md`.

M2 review follow-up. Two findings, one root cause: `HeadKind::Diffusion` carried only
`n_steps` and a scheduler kind, so W3 had to invent the rest of the schedule as lowering
constants — a linear `1e-4 .. 0.02` beta schedule spread across the *inference* steps and
`sigma_t = sqrt(beta_t)` (diffusers' `fixed_large`). `diffusers` and LeRobot build the betas
over `num_train_timesteps` and subsample, and default to `fixed_small`. A real Diffusion Policy
checkpoint would not have reproduced. The second finding is why nobody noticed:
`reference.rs:26` imports `diffusion_schedule` from the lowering it is meant to check, so the
tier-4 test proved Rust<->PyTorch agreement and nothing about the coefficients.

## context

```
crates/es-ir/src/learning.rs                  (extended: HeadKind::Diffusion fields, additive, serde defaults)
crates/es-policy/src/lower/torch.rs           (rewritten: diffusion_schedule, _DdpmHead)
crates/es-policy/src/reference.rs             (extended: the diffusers oracle and its two tests)
crates/es-policy/src/torch_runtime.rs         (python_candidates is pub(crate))
crates/es-policy/python/ddpm_ref_check.py     (new: the independent oracle)
crates/es-data/src/lerobot_config.rs          (extended: the new fields pass through)
crates/es-data/tests/lerobot_config.rs        (extended: one pass-through test)
docs/design/learning-lowering.md              (rewritten: sections 8.3, 8.5)
docs/api-notes/torch.md                       (extended: diffusers surface, known gaps)
docs/api-notes/lerobot-config.md              (extended: the six scheduler fields)
docs/packets/M2/P-M2-R6.md                    (new)
```

## spec

- `HeadKind::Diffusion` gains `num_train_timesteps`, `beta_schedule`, `variance_type`,
  `prediction_type`, `clip_sample`, `clip_sample_range`, each with a `serde` default equal to
  LeRobot's Diffusion Policy default (`100`, `squaredcos_cap_v2`, `fixed_small`, `epsilon`,
  `true`, `1.0` — `Status: unverified`, recalled not fetched). Additive: an IR document written
  before this packet still deserializes. `HeadKind` loses `Eq` (it now carries an `f32`); it was
  only ever compared through `LearningNode`, which is `PartialEq` alone.
- `diffusion_schedule` builds `betas` / `alphas_cumprod` over `num_train_timesteps` and
  subsamples the `n_steps` inference timesteps the way `set_timesteps` does under diffusers'
  default `timestep_spacing = "leading"`: `(arange(0, n_steps) * (num_train // n_steps))[::-1]`.
  `beta_schedule` is `linear` (`torch.linspace`) or `squaredcos_cap_v2` (`betas_for_alpha_bar`,
  cosine, capped at `0.999`).
- Variance is `DDPMScheduler._get_variance` clamped below at `1e-20`: `fixed_small`
  `beta~_t = (1 - abar_prev)/(1 - abar_t) * beta_cur` by default, `fixed_large` `beta_cur`
  selectable. `DDIMScheduler` has no `variance_type`, so it is ignored under `Ddim`.
- `clip_sample` clamps `pred_original_sample` to `+-clip_sample_range`, which forces the loop
  into diffusers' decomposition (`x0` first, then a recombination) instead of the affine
  `c1*x + c3*eps`. The DDIM eta = 0 path is unchanged in form and now runs on the subsampled
  timesteps. `noise_<t>` buffers are keyed by the real timestep.
- `prediction_type != Epsilon` is `LowerError::Unsupported`; `num_train_timesteps < n_steps`
  and an odd conditioning width are `LowerError::Shape`.
- `es-data`'s LeRobot conversion passes all six through; `variance_type` is not a LeRobot field
  and is pinned to diffusers' `fixed_small`.

## oracle

```
cargo test -p es-policy -p es-ir -p es-data --features es-ir/testing
ES_PYTHON=<venv-with-torch-and-diffusers>/Scripts/python cargo test -p es-policy -- --nocapture
```

- `python/ddpm_ref_check.py` is the independent reference: given a scheduler configuration it
  builds a real `DDPMScheduler` / `DDIMScheduler` and prints `alphas_cumprod`, the subsampled
  `timesteps` and `_get_variance` per step as JSON; given weights and inputs as well, it runs
  the whole sampler through `scheduler.step(model_output, t, sample)` and prints the chunk.
- `reference::tests::diffusion_schedule_matches_diffusers` compares `diffusion_schedule` to it
  over all eight scheduler x `beta_schedule` x `variance_type` combinations at
  `Tolerance::TIER4_FP32`, with `timesteps` required to be **exactly** equal.
- `reference::tests::torch_{ddpm,ddim}_matches_diffusers_step_loop` runs the lowered head
  through `TorchRuntime` and compares it to that direct step loop at `Tolerance::TIER4_FP32`.
  DDPM's ancestral draw comes from the checkpoint's `noise_<t>` buffer via a `randn_tensor`
  monkeypatch, so both sides consume the same numbers.
- Both print `RAN ... max_abs = ...` or `SKIPPED ...: <why>` on one line; no interpreter, no
  `torch` or no `diffusers` is a skip, never a pass.

## acceptance

- The five torch-tier tests RAN here (torch 2.14.0+cpu, diffusers 0.40.0): `torch_ddpm` 2.76e-7,
  `torch_ddim` 1.79e-7, `torch_flow_matching` 1.04e-7, `ddpm_matches_diffusers_step_loop`
  6.86e-7, `ddim_matches_diffusers_step_loop` 1.34e-6 `max_abs`, all under the spec 8.9 tier-4
  limit of 1e-5. `diffusion_schedule_matches_diffusers`: `alphas_cumprod` <= 2.39e-7,
  `_get_variance` <= 4.18e-7, `timesteps` exact, over all eight combinations.
- The tier-4 configuration is 8 inference steps out of 100 training timesteps, so it crosses
  the subsampling path: `[84, 72, 60, 48, 36, 24, 12, 0]`, asserted literally in the emitted
  source.
- `cargo fmt --check`, `clippy -D warnings` and the full `es-policy` / `es-ir` / `es-data` test
  runs are clean; `es-ir` stays inside the spec 1.5 context budget.

## forbidden

- No new trait (`INV-17`: the seven extension points are closed). No `PolicyRuntime` change.
- No new lowering of `sample` / `v_prediction`, of `DpmSolver`, of `eta > 0`, of
  `timestep_spacing` other than `"leading"`, or of a U-Net denoiser — those are later packets,
  and each is an explicit `Unsupported` here rather than an approximation.
- No edit outside the `## context` list. `crates/es-data/src/lerobot_config.rs` is shared with
  the in-flight `Augment` wiring fix (P-M2-R2): only the diffusion-head construction and the
  `DiffusionConfig` fields are this packet's.
- No `pickle`, no `torch.load` (`INV-16`): the oracle's weights cross as JSON numbers.
- Golden files are read-only (spec 1.4).
