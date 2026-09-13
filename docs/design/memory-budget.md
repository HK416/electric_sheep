# Memory and bandwidth budget model

Design note for `crates/es-compile/src/budget.rs` (layer 7). Spec: spec 20.1, spec 20.2,
spec 20.3, spec 15.2 (tile atlas sizing), spec 12.4 (nine-metric set), spec 5.2 (batch
domains), spec 28.4 / spec 28.7 gate 13 (±10% accuracy gate). Work packet:
`docs/packets/M2/W5-memory-budget.md`.

## 1. What this is and is not

Spec 20.3: a budget is computed **before** a scene/task/policy loads and checked against real
memory; exceeding it shrinks `N_obs`/`N_inf` or fails explicitly rather than letting the
process OOM. `MemoryBudget::estimate` is the static half of that: every item spec 20.2 lists,
computed from IR shapes and config alone, with the formula spelled out per item
(`BudgetItem::formula`) so a human can check the number by hand.

**Not in this note or this packet:** a running measurement (there is no GPU in this
environment to measure against), the automatic `N_obs`/`N_inf` shrink loop, and the editor's
live display. Those need a device and are later work. Every number this packet produces is
`Target / Status: unverified` against spec 20.1's reference measurements until a GPU run
exists (spec 28.4's ±10% gate, spec 28.7 gate 13).

## 2. The spec 20.1 reference measurements (calibration targets, unverified)

| scenario | VRAM | Status |
|---|---|---|
| Isaac Lab, RTX 4090, Cartpole physics only, 4,096 env | 3.3 GB | `unverified` |
| Isaac Lab, RTX 4090, Cartpole + RGB camera, 1,024 env | 16.7 GB | `unverified` |
| Isaac Lab, RTX 4090, G1 locomotion, 4,096 env | 6.1 GB | `unverified` |

The vision task has a quarter of the envs and five times the VRAM — this is why spec 12.2's
`round_robin` exists, and it is the sanity check a future GPU-in-the-loop run should reproduce
within ±10% (spec 28.4) once this model is calibrated against one.

The spec 20.2 worked example (512 obs env × 2 view × 224×224 RGB, ACT, `N_inf = 256`) is the
second calibration target:

```
atlas RGB8 double-buffered      512x2x224x224x3x2       =  308 MB
normalized FP32 intermediates   512x2x224x224x3x4       =  616 MB
policy weights (ACT 52M fp32)                           =  208 MB
inference activations (batch 256)   needs measurement    ~ 2.0 GB
chunk buffers, 4096 env          4096x50x8x4x2          =   13 MB
--------------------------------------------------------------------
vision + policy subtotal                                 ≈ 3.1 GB
+ physics backend (MJWarp 4096 env)                       ≈ 2-4 GB
+ PyTorch, if training shares the GPU                     ≈ 8-16 GB
```

`MemoryBudget::estimate` reproduces the first two lines exactly from an
`ObservationIr`/`LearningGraph` shaped like that example (render tile atlas, observation
intermediates); inference activations and policy weights are two of the items this model
states as `unavailable` rather than guessing (§4). The chunk-buffer line it deliberately does
**not** reproduce: 4096×8×50×8×8 = 105 MB rather than 13 MB, because the runtime holds eight
overlapping f64 chunks per env — see §3 below.

## 3. The items and their formulas

Every item is a `BudgetItem { name, bytes, formula }`. `bytes == 0` with a formula starting
`"unavailable: ..."` means "no data to size this from", never "this costs nothing" — the
`Display` table and the CLI print the formula for exactly this reason.

| item | formula | source |
|---|---|---|
| `physics_state` | `(nq+nv+nu+nsensordata) x n_sim_envs x 8B` (f64, spec 3.1's state representation) | `ModelSizes` (caller-supplied, mirrors `es_physics_core::backend::ModelInfo`) |
| `render_tile_atlas` | no explicit layout: `n_obs_envs x n_views x H x W x ch x dtype_bytes x 2` (double buffer). With a `TileAtlasCfg`: `rows x tiles_per_row x tile_w x tile_h x ch x dtype_bytes x 2`, `rows = ceil(tiles / tiles_per_row)` — spec 15.2's row padding when the tile count does not divide evenly | the first `ImageInput` node's raw sensor `ImageSpec` |
| `observation_intermediates` | `sum(CPU plan arena buffers) x n_obs_envs` | `es_compile::plan::CpuPlan::compile` in `PlanMode::Release` — this *is* the spec 11.1 liveness analysis, reused rather than re-derived |
| `history_buffers` | `sum over History entries: depth x per_sensor_bytes x n_obs_envs` | `ObservationIr::temporal.history` (spec 7.5 layer 1) joined against the `ImageInput`/`StateInput` node that owns each sensor id |
| `policy_weights` | always `unavailable` | `WeightsRef` (spec 8.3) carries a path and a blake3 hash only — no byte size (`INV-16` keeps loading to `safetensors`, it does not add a size field) |
| `inference_activations` | `sum(LearningNode output shapes) x inference_batch x precision_bytes` | `LearningGraph::nodes` via `IrNode::outputs` (every node's declared output ports, spec 8.3) |
| `chunk_buffers` | `n_sim_envs x CHUNK_SLOTS x horizon x action_dim x 8B` (f64) — `es_core::sizing::chunk_buffer_bytes`, **deviating from spec 20.2's `x 4B x 2`**, see below | `LearningGraph::policy.contract` (`horizon`, `action_dim`) |

### `chunk_buffers`: one model, and it is not spec 20.2's (P-M2-R5)

Spec 20.2 budgets `n_sim_envs x H x NJ x 4B x 2` — one f32 chunk, double-buffered. The
runtime keeps `CHUNK_SLOTS = 8` *overlapping* chunks of f64 per env, because ACT temporal
ensembling averages every live chunk (spec 8.6) and the control path is f64 throughout. That
is 4x the spec's figure, and it is the implementation that is right: the spec's line predates
the ensembling buffer.

Two in-tree models of the same quantity is how the spec 28.7 gate 13 ±10% check ends up
unreachable, so there is exactly one — `es_core::sizing::chunk_buffer_bytes` at layer 1, which
both `es_env::DomainSizing` (layer 9) and this budget (layer 7) call. The two crates cannot
see each other (spec 4.2), which is why the formula sits below them both. At the spec 12.4
gate configuration (4,096 envs, `H = 20`, `NJ = 7`) it is 36.7 MB, and
`cargo test -p es-compile budget` asserts the budget item equals that function.

`bandwidth_per_tick` is `render_tile_atlas + observation_intermediates`: the bytes that move
once per simulation tick, independent of any Hz (spec 12.4's bandwidth column needs a rate to
turn this into a per-second figure; that is the measurement-loop packet, not this one).

`per_domain` buckets each item under the spec 5.2 domain it belongs to (`simulation`,
`observation`, `inference`) plus `control` for chunk buffers, which sit between the inference
and actuation domains and belong to neither cleanly.

## 4. Why some items are `unavailable`, not zero or a guess

- **`policy_weights`**: `WeightsRef` is `{ path, hash }` for every variant (`Safetensors`,
  `Onnx`, `SpirV`). Adding a byte-size field is a schema change outside this packet's
  declared scope (`crates/es-compile/src/budget.rs` and its `lib.rs` re-export line only); a
  future packet that wants this number should widen `WeightsRef` instead of estimating from
  the checkpoint name.
- **`physics_state` / `render_tile_atlas`**: `None` when the caller passes no `ModelSizes` /
  the Observation IR has no `ImageInput` node — a task with no camera legitimately has no
  render budget, and this model does not know how to size a physics backend it was never told
  about.
- **`observation_intermediates`**: `None` when the Observation IR does not compile to a CPU
  plan (`CpuPlan::compile` returns diagnostics) — the budget model does not duplicate the
  compiler's own validation, it reuses its output.

## 5. Rules checked (spec 20.3)

`MemoryReport::violations(&BudgetInputs) -> Vec<BudgetViolation>`:

- `obs_batch_le_sim_batch` — `n_obs_envs > n_sim_envs` (spec 5.2: the observation batch is
  drawn from the simulation batch, never larger than it).
- `total_le_device_minus_reserve` — only when `BudgetInputs::device_bytes` is given; checks
  `total_bytes <= device_bytes - reserve`. The reserve is a fixed 512 MiB placeholder for
  driver/OS overhead (`ponytail` comment at the constant): it needs calibrating against a real
  device once one is available for CI, which is exactly the ±10% gate this whole model is
  waiting on.
- `tile_atlas_within_max_image_dimension_2d` — only when a `TileAtlasCfg` is given; checks the
  packed atlas fits spec 15.2's `maxImageDimension2D` (16384 on the desktop GPUs targeted).

Spec 20.3 also calls for an automatic `N_obs`/`N_inf` shrink loop and a compile-time error in
the editor (its "Memory Plan", spec 11.1). Neither is implemented here: both need this model
wired into `es-env`/`es-editor` (layers 9 and 12), which is other packets' scope.

## 6. What the ±10% gate (spec 28.4, spec 28.7 gate 13) still needs

1. A GPU (or a captured `nvidia-smi`/`vkQueryMemoryHeap` trace) to run the spec 20.1 scenarios
   on and compare against this model's output for the same `BudgetInputs`.
2. Calibrating `DEFAULT_RESERVE_BYTES` and the render tile atlas's implicit assumptions
   (driver padding/alignment on the real texture format) against that trace.
3. Wiring `es bench --memory-report` (or a successor) to run *during* a real backend/session
   so `model: Some(ModelSizes)` and a real `device_bytes` are available, instead of the
   `--scene`-only best-effort path this CLI packet has today.

Until then, every number this model prints is `Target / Status: unverified` per the project
convention (`CLAUDE.md`): a formula that can be audited, not a measurement that has been.
