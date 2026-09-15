# M5 V5 — the fast cycle

Design note: `docs/design/visible-learning.md` section 7.11. Depends on V2b (the bake and the
training entry point) and V3 (the perturbation suite, the frame dump and `events.json`). It changes
no result: every number section 7.9 and section 7.10 recorded must come back byte-identical.

## the problem

One collect -> train -> evaluate cycle of the demo is measured, on the oracle server, at roughly an
hour, sequential:

| stage | measured (V2b/V3) |
| --- | --- |
| `es eval run` nominal, 16 episodes | ~5 min |
| `es eval run` the 6-suite, 96-episode run | ~28 min |
| `train_act.py`, 20,000 steps at batch 8 | ~11 min |

Nothing in that is compute the machine cannot do faster; it is compute the machine is not being
asked to do at all. The physics backends are Python subprocesses with one env per process
(design note section 7.6), `Evaluation::run` hardcodes `BatchDomains::single_env()`, and the
training loop copies one sample at a time from host memory even when the whole baked set would sit
on the device.

## what the sequential runner threads across episodes — read this before widening the split

The natural-looking split is "one episode per worker". It is wrong here, and the reason is in
`crates/es-eval/src/runner.rs`. Inside one **cell** (one suite), `run_with_frames`:

1. builds **one** `Env` and reuses it for every episode of the suite. `Env::reset` bumps a per-env
   episode counter (`crates/es-env/src/env.rs:224`) and `RandomizationPlan::apply` is keyed by it,
   so episode 5's initial state is not reproducible without having run episodes 0..4;
2. builds **one** `SafetyPlane` per cell and reports `safety.counters()` for the whole cell —
   `envelope_violation_rate` and `chunk_underrun_rate` are cell-level sums;
3. accumulates `env.metrics()` over the whole cell;
4. threads a monotonic `seq` across the cell's episodes (spec 8.6: the plane treats a chunk as new
   iff `seq` grew).

Episodes inside a suite are therefore **not** independent, and `--jobs` must not pretend they are.
A **cell** is independent: it opens its own `Env`, its own `SafetyPlane`, starts `seq` at 0 and
calls `plan.reset()` on every episode including its first. The only things the cells share are the
compiled `CpuPlan` (reset per episode, so a freshly compiled plan and a reset one are the same
plan) and the `PolicyRuntime` (feed-forward: `TorchRuntime::infer` carries no state between calls,
and the temporal window lives in the plan, not in the policy).

**So the partition is by cell, round-robin: cell `c` belongs to shard `c % N`.** The demo's
evaluation has six suites, which is where its 28 minutes are; the nominal 16-episode run is one
suite and `--jobs` does not shorten it. That is the honest ceiling of this packet and it is stated
rather than worked around: making it finer means giving `Env` a way to seek its episode counter,
and `es-env` is forbidden here.

## context

```
crates/es-eval/src/runner.rs
crates/es-eval/src/lib.rs
crates/es-eval/tests/evaluation.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
crates/es-policy/tests/ir_training.rs
python/es/train_act.py
python/es/README.md
python/es/README.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V5-fast-cycle.md
docs/packets/M5/V5-fast-cycle.ko.md
```

No new crate and no new dependency. The worker is the same `es` binary
(`std::env::current_exe`), the transport is a JSON file, and both are already linked.

## spec

- **§10.4, §10.5.** `--jobs N` is a scheduling choice, not a different evaluation. The parent
  partitions the cells, the children run them, and the parent alone computes the report, the
  `evaluation_hash`, the `execution_hash` and `evaluation.lock` — from the same `judge` over the
  same merged `measured`/`samples` maps the sequential path builds. A merged report whose cells are
  not exactly `0..suites.len()`, each once, is refused: a report missing a suite would otherwise
  wear a correct `evaluation_hash`.
- **§10.4 (cell order).** Each worker's cells are tagged with their index into
  `EvaluationIr::suites` and the merge is a stable sort on that index, so the merged `cells` array
  is in the order the sequential run would have written it whatever order the workers finished in.
  No `HashMap` iteration, no wall-clock, no dependence on which child answered first (§3.4).
- **§3.5 tier 1.** The sequential path is literally the sharded path with `N = 1`: `run_with_frames`
  calls `run_shard((0, 1))` and `merge`. Byte-identity at `N = 1` is by construction; the oracle
  pins it at `N = 4`.
- **§12.4.** Two of the nine metrics — `physics_steps_per_sec` and `actions_per_sec` — are derived
  from `EnvMetrics::simulation_wall`, and a sharded run measures each worker's own wall clock.
  They are already not reproducible run-to-run in the sequential path; the demo's
  `tests/fixtures/visible-learning/evaluation.toml` declares neither, and the oracle's fixture
  declares neither. This is recorded, not fixed: fixing it is `es-env`'s, and `es-env` is forbidden
  here. Nothing in this packet reports a single `step/s` figure.
- **§7.2, INV-14.** Untouched. No shard resamples, converts or re-runs an observation; each cell's
  frames are written by the same `capture` from the same `CpuPlan`, into its own
  `<frames>/<suite>-<NN>/` directory. Cell names are globally unique and shards own disjoint cells,
  so the children write into one frame directory without a merge step and without a collision.
- **§9.** The Safety Plane is untouched: one plane per cell, as before, validating every step. No
  code path added here can widen, skip or disable it (INV-12), and `SafetyPlane::validate` keeps its
  signature (INV-13).
- **§2.3, §8.1.** `train_act.py` keeps owning the optimizer and nothing above it. `--resident-gpu`
  moves the baked tensors, not the maths: the batch order, the dtype and the loss are identical.
  `--amp bf16` and `--compile` are opt-in and **do** change the bits; they are for the sweep, not
  for a run whose numbers are quoted. Nothing any of the three reads enters a hash slot.
- **§8.9.** The fp32 inference-equivalence check (`compare_actions` at `Tolerance::TIER4_FP32`) is
  the gate that stays, and it stays on a checkpoint trained at the defaults.
- **INV-16.** safetensors in, safetensors out. `--resident-gpu` changes where a tensor lives, not
  what format it was read from.
- **INV-17.** No new trait. `Shard`/`ShardCell` are plain serde structs and `Evaluation::run_shard`
  / `Evaluation::merge` are two more inherent methods on the existing namespace struct.
- **§1.5.** `es-eval` is at 2,595 code lines and `es` at 4,261; this packet is budgeted under ~350
  across both.

## oracle

PR tier, no Python, no GPU, no network:

```
cargo test -p es-eval --test evaluation -- sharding
cargo test -p es --test cli -- eval_run_jobs
cargo test -p es --bin es -- cmd::eval
```

1. `sharding_the_cells_produces_a_byte_identical_report` — one four-suite image evaluation driven
   by the in-test `FakeBackend`/`FakePolicy`, run three ways: sequentially through
   `Evaluation::run_with_frames`, as one shard (`N = 1`), and as four shards (`N = 4`) with a
   **fresh `FakePolicy` per shard**, which is what a separate process would give. `report.json`,
   `evaluation.lock` and `events.json` are written with `write_artifacts`/`write_events` and
   compared as **bytes**; every cell's frame directory is compared file-by-file as bytes.
2. `a_merge_missing_a_cell_is_refused` — dropping one shard's cells is `EvalError::Shard` naming
   the cells that were covered, never a report.
3. `eval_run_jobs_zero_is_a_usage_error` — `es eval run --jobs 0` exits 2 before loading anything.
4. The `cmd::eval` unit tests — `--shard` without `--shard-out` refused, `--shard` with `--jobs > 1`
   refused, `--shard 4/4` refused, and `shard_failed` producing one named error carrying the
   shard index, the exit code and the child's last stderr line.

Oracle tier (needs `ES_PYTHON` with `torch`; skips with a printed reason otherwise):

```
cargo test -p es-policy --test ir_training -- --ignored --nocapture resident_gpu_does_not_move_the_loss
cargo test -p es-policy --test ir_training -- --ignored --nocapture act_training_uses_baked_observations
```

5. `resident_gpu_does_not_move_the_loss` — 40 optimizer steps on the same baked fixture and the
   same `--seed`, once without `--resident-gpu` and once with, comparing the two `--loss-curve`
   JSON files **byte for byte**. `--amp bf16` and `--compile` are deliberately **not** in this
   comparison: they change the bits and are opt-in for that reason.
6. `act_training_uses_baked_observations` — unchanged, at the defaults, including the tier-4 fp32
   round-trip through `TorchRuntime`. This packet must not move it.

Server tier (phase 2, the oracle server; a measurement, not a gate):

```
# baseline, already measured (V3, design note section 7.8/7.10)
es eval run --config evaluation.toml --policy policy.esb --scene scene.xml \
    --out eval-seq --frames frames-seq
# the same evaluation, six workers
es eval run --config evaluation.toml --policy policy.esb --scene scene.xml \
    --out eval-par --frames frames-par --jobs 6
cmp eval-seq/report.json eval-par/report.json
cmp eval-seq/evaluation.lock eval-par/evaluation.lock   # `created` differs; compare the rest
cmp eval-seq/events.json eval-par/events.json
# training, at the defaults and resident
python/es/train_act.py --module build --baked baked --out a.safetensors \
    --batch 8 --seed 0 --checkpoint-at 20000 --device cuda
python/es/train_act.py --module build --baked baked --out b.safetensors \
    --batch 8 --seed 0 --checkpoint-at 20000 --device cuda --resident-gpu
```

| what | Target | Status |
| --- | --- | --- |
| 6-suite 96-episode run, `--jobs 6` | ~5-6 min (from 28 min) | **unverified** |
| `report.json` / `events.json` identical to the sequential run | byte-identical | **unverified** |
| 20,000 steps at batch 8, `--resident-gpu` | faster than 11 min | **unverified** |
| 20,000 steps, `--resident-gpu --amp bf16` | faster still, different bits | **unverified** |

Every row above stays `Status: unverified` until it is measured on the server and the number is
written into design note section 7.11. Nothing in phase 1 may quote one of them as measured.

## acceptance

- `es eval run --jobs N` (default 1) produces a `report.json`, an `evaluation.lock` (modulo
  `created`) and an `events.json` byte-identical to `--jobs 1`, and frames byte-identical per cell.
- `--jobs 0` is refused. `--shard i/N` requires `--shard-out` and refuses `--jobs > 1`.
- A worker that fails surfaces as exactly one error naming the shard, its exit code and its last
  stderr line. No partial report is ever written.
- `train_act.py --resident-gpu` produces a bit-identical loss curve to the default path at the same
  seed on the CPU. `--amp bf16` and `--compile` are opt-in and documented as bit-changing.
- `act_training_uses_baked_observations` passes unchanged.
- `cargo xtask ci` green.

## forbidden

- `es-safety` — no file in it, for any reason. The plane is per cell and stays per cell.
- `es-env` — including the `Env` episode counter that would make an episode-granular split
  possible. That is a different packet with its own oracle.
- The physics backends (`es-physics-backend`, `es-physics-core`) and the backend subprocess
  protocol.
- **Any change to a measured number.** Not the success rates in design note sections 7.8/7.9, not
  the loss thresholds in `ir_training.rs`, not the goldens, not `--batch`'s default of 8. If a
  number moves, the packet has a defect; the number is not the thing to edit.
- `Evaluation::run`'s and `run_with_frames`' existing signatures, and `BatchDomains::single_env()`.
- Adding a runtime, a thread pool or a scheduler crate. The workers are processes because the
  backends are processes and the GIL is real.
