# P-M2-R2 — a converted LeRobot config plans and evaluates

Spec: §7.3 (augmentation nodes), §10.4 (evaluation auto-disables augmentation), §11.3
(`CpuPlan`), §14.4 (LeRobot config conversion).
Design notes: `docs/design/observation-lowering.md` §3 and §9.1,
`docs/design/evaluation-execution.md` §2.2.
Invariants: INV-14 (intrinsics on crop/resize), INV-15 (augmentation off in evaluation).
Review finding: `docs/reviews/M2.md` — Blocker, `crates/es-data/src/lerobot_config.rs:426`.

W6's converter inserted an *unwired* `Augment { training_only: true }` node for
`crop_is_random`. `topo_order` visits it anyway and the planner had no `Augment` arm, so every
LeRobot config carrying `crop_shape` — the checked-in `diffusion_config.json` fixture
included — produced an `ObservationIr` that could not compile. W1 and W6 did not compose and
no test crossed them.

## context

```
crates/es-compile/src/plan.rs
crates/es-compile/tests/observation_cpu.rs
crates/es-data/Cargo.toml
crates/es-data/src/lerobot_config.rs
crates/es-data/tests/lerobot_config.rs
crates/es-eval/src/lib.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
docs/design/observation-lowering.md
docs/design/evaluation-execution.md
docs/packets/M2/P-M2-R2.md
```

## spec

- **`es-data`** — the `Augment` node is wired into the chain: `ImageInput → Crop(centre) →
  Augment{RandomCrop, training_only} → Resize → Normalize → …`. `Augment` is
  geometry-preserving in this IR (`image_out` propagates the incoming `ImageSpec` unchanged),
  so LeRobot's random crop is the centred `Crop` — which carries the intrinsics transform,
  INV-14 — plus an offset jitter *of that window*, which is what the node names. Its `io` is
  therefore `unary(cropped_ty, cropped_ty)`, not `unary(raw_ty, cropped_ty)`.
- **`es-compile`** — `CpuPlan::lower` gains the `Augment` arm. **No new API:** this plan is
  the evaluation / deployment path, there are no augmentation kernels on it, and there is no
  training mode from which to reach them, so a `PlanOptions { training }` flag would be a
  switch with one position. `training_only: true` lowers to an identity pass-through — no
  step, no buffer, consumers read the producer's buffer — which *is* §10.4's auto-disable.
  `training_only: false` is `COMPILE-002`. The node is never removed from the graph:
  stripping it would make `observation_hash` describe a graph the author never declared. The
  flag belongs in the packet that brings the kernels.
- **`es-eval`** — `refuse_augmentation` skips `training_only` nodes (the plan has already
  disabled them) and still refuses any other `Augment` node outside the allow-list.
- **`es-data`** (nit from the same review) — `config.json` `shape` entries are bounded:
  `checked_dim` rejects a STATE or ACTION dim outside `1..=65_536` with
  `ConfigError::OutOfRange`, so `vec![0.0; dim]` cannot be sized from untrusted input.

## oracle

```
cargo test -p es-data lerobot_config
cargo test -p es-compile a_training_only_augment_is_an_identity_pass_through
cargo test -p es-compile an_augment_that_is_not_training_only_is_a_diagnostic
cargo test -p es-eval a_training_only_augment_node_is_disabled_not_refused
cargo test -p es-eval an_allow_listed_augment_node_gets_past_inv_15_and_dies_in_the_compiler
```

- `es-data`: `every_converted_observation_compiles_to_a_plan` runs `CpuPlan::compile` on
  `Converted::observation` for both fixtures (`act_config.json` and the `crop_shape`-carrying
  `diffusion_config.json`) in both plan modes and asserts no diagnostics — errors *or*
  warnings. `es-compile` is a dev-dependency only; es-data is layer 10 and es-compile layer 7,
  so this is a legal strictly-lower dependency (`cargo xtask layering` covers it).
- `es-eval`: the end-to-end run uses a hand-built fixture rather than a converted config,
  because es-data and es-eval are both layer 10 and may not depend on each other (rule 1), and
  because `capture` refuses image inputs until `es-render` exists. It asserts a `training_only`
  `Augment` node runs and changes nothing: the report equals the one from the same graph
  without the node. The crossing point is the es-data compile oracle above.

## acceptance

- Both checked-in LeRobot fixtures convert **and** plan, in `Debug` and `Release`.
- The converted `Augment` node has an inbound and an outbound edge (asserted), so no
  unreachable node reaches a planner.
- An allow-listed non-`training_only` node gets past INV-15 and is refused by the compiler —
  asserted as that exact outcome, not `Ok(_) | Err(Plan(_))`.
- `cargo test -p es-compile -p es-eval -p es-data` green; no golden changed.

## forbidden

- Augmentation *kernels*: a `training_only` node is identity here, never a sampled crop
  offset. That needs the RNG-stream contract and is a later packet.
- Rewriting or pruning the observation graph to make it plan (INV-15's reasoning:
  `observation_hash` must keep describing what was declared).
- `rescale_intrinsics: false` anywhere on the crop/resize chain (INV-14).
- `crates/es-compile/src/budget.rs`, `crates/es-env`, `crates/es-policy`,
  `crates/es-telemetry`, `crates/es`.
