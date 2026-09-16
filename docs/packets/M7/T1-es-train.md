# M7 T1 — `es train`: one document, one command, a real `training_hash`

Spec: §13.1 (each stage of the loop is a CLI command *and* an artifact), §19.3 (training
identity — the `training/` bundle and `training_hash`), §8.8, §2.3 (Python on the learning path
only), §1.4, §28.10 rule 2. Design note to write: `docs/design/training-recipe.md` (+ `.ko.md`).
Predecessors: M5 V2/V2b (`train_act.py`, `es dataset bake`), V8/V19 (`import-lerobot`,
`lerobot-train`), M3 W7 (`es loop distill` writes `training_identity.json` with all-zero slots).

## the question

Today one training run is nine hand-written scripts (`docs/packets/M5/V19` "server" section):
`export → lerobot-train → import-lerobot → eval run`, or `bake → lower → train_act.py → pack`,
with hashes copied by hand between them and §19.3's `training_hash` all zeros except `dataset`.
**Can one document drive either path with one command, and can that command write §19.3's
`training/` from real values?**

## spec

`es train --recipe training.toml [--out <dir>] [--dry-run]`.

The recipe (`kind = "training"`, parsed and hashed in `es-data`, not in the CLI):

```toml
kind = "training"
[dataset]
root   = "runs/collect-001/ds"        # LeRobot v2.1 root (es loop collect)
frames = "runs/collect-001/frames"    # optional; required when the Observation IR has an image input
[policy]                              # exactly one of the two
bundle  = "untrained.esb"             # IR path: Learning IR + Observation IR from the bundle
# lerobot = { type = "act", chunk_size = 16, n_action_steps = 16, extra = [] }   # external path
# task = "task.toml"; observation = "observation.toml"; deployment = "deployment.toml"
[run]
steps = 20000
batch = 8
lr = 1e-4
seed = 0
checkpoint_at = [1000, 5000, 20000]
device = "cuda"
interpreter = "python"                # overridden by ES_PYTHON when set
```

* **IR path**: `es dataset bake --policy … --out <out>/baked [--frames …] <root>` →
  `es policy lower --policy … --out <out>/module` → `<interpreter> python/es/train_act.py
  --module <out>/module --baked <out>/baked --out <out>/weights/model.safetensors --checkpoint-at …
  --seed … --batch … --lr … --device … --loss-curve <out>/metrics/loss.json` → for every
  checkpoint `es policy pack --policy … --weights … --out <out>/checkpoints/<step>.esb`. The
  existing subcommands are called **in-process** (they are functions in `crates/es/src/cmd`), not
  re-spawned; only the trainer is a subprocess.
* **External path**: `es dataset export --lerobot-v3 <root> --out <out>/ds-v3 [--frames …] --drop
  action_commanded,action_source,intervention --state-dim <from the Observation IR>` →
  `<interpreter> -m lerobot.scripts.lerobot_train …` (or the `lerobot-train` entry point next to
  the interpreter) with `--policy.type`, `--policy.chunk_size`, `--steps`, `--batch_size`,
  `--seed`, `--save_freq`, `--output_dir <out>/lerobot`, `--policy.device`, `--wandb.enable=false`,
  `--policy.push_to_hub=false`, plus `extra` verbatim → for every saved checkpoint
  `es policy import-lerobot --checkpoint … --task … --observation … --deployment … --out
  <out>/checkpoints/<step>.esb`.
* **`<out>/training/`** per §19.3, every file written from what the run actually used:
  `config.json` (the recipe, canonical JSON, plus the resolved interpreter path and the trainer's
  reported version), `optimizer.json` (`AdamW`, lr, betas/weight decay as the trainer reports
  them), `scheduler.json` (`{"kind":"constant"}` until T4), `seed.json`, `dataset.lock` (the
  dataset's content/schema/split hashes from `es-data::identity`), `base_model.lock`
  (`{"source":"none"}` for the IR path; for the external path the LeRobot config's
  `vision_backbone` + `pretrained_backbone_weights` string — a **declared** provenance, hash
  unset until T5), `augmentation.json` (`{"kind":"none"}` until T6), `precision.json` (fp32 /
  the `--amp` flag), `topology.json` (`{"world_size":1}`), `checkpoint.manifest` (step → bundle
  path, `policy_hash`, blake3 of the weights), `metrics.json` (the loss curve; write
  `metrics.parquet` instead only if `es-data`'s existing parquet writer takes a scalar table
  in a few lines — say which in the design note), `hardware.json` (device name from
  `torch.cuda.get_device_name` or `cpu`, torch version, driver string, the `es` version and
  git describe). Slots the run genuinely does not know are written as `{"unset": true}` and
  hashed as such — **never a fabricated digest** (§28.10 rule 2).
* **`<out>/training.lock`**: `training_hash` = `TrainingIdentity::training_hash` over the blake3
  of each file above; plus `policy_hash` per checkpoint. Two runs of the same recipe on the same
  inputs with the same interpreter must produce one `training_hash` **before** the trainer runs
  (everything but `checkpoint.manifest`/`metrics`/`hardware` is known up front) — the design
  note states which slots are pre-run and which post-run, and `training.lock` carries both
  digests (`identity_hash` pre-run, `training_hash` post-run).
* **`--dry-run`** prints the command plan — one line per step, paths relative to `<out>` — and
  writes nothing but `<out>/training/plan.txt`. This is what CI judges without Python.
* Refusals by name: both `bundle` and `lerobot` set (or neither); an image input without
  `frames`; a dataset whose recorded `task_hash` differs from the policy's Task IR unless
  `--allow-retired-task` names it (M5 review S-3/R4 — the check is the export/import path's
  refusal, surfaced here); an interpreter that cannot `import torch` (external path: `import
  lerobot`) — print the interpreter's own error.

## context

`cargo xtask check-scope` reads the fence; the prose below it is the same list with the
reason for each entry.

```
crates/es-data/src/training.rs
crates/es-data/src/lib.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/mod.rs
crates/es/src/main.rs
crates/es/src/cmd/dataset.rs
crates/es/src/cmd/policy.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/training*.toml
tests/golden/train/*.txt
docs/design/training-recipe*.md
docs/packets/M7/T1-es-train*.md
python/es/train_act.py
```

`crates/es/src/cmd/{dataset,policy}.rs` are in the fence for the one edit the `forbidden`
section allows them: making a step's function `pub(crate)` so `es train` can call it
in-process.

`crates/es-data/src/training.rs` (recipe schema, plan builder, identity assembly; new),
`crates/es-data/src/lib.rs` (module + re-exports only), `crates/es/src/cmd/train.rs` (new),
`crates/es/src/cmd/mod.rs`, `crates/es/src/main.rs` (dispatch + `TOP_HELP` line),
`crates/es/tests/cli.rs` (new `train_*` tests only; touch nothing else in the file),
`tests/fixtures/visible-learning/training.toml` and `training-lerobot.toml` (new fixtures),
`tests/golden/train/plan-ir.txt`, `plan-lerobot.txt` (new goldens, generated by the sanctioned
test generator you add, `#[ignore]`d like `generate_goldens`), `docs/design/training-recipe.md`
+ `.ko.md` (new), `docs/packets/M7/T1-es-train*.md`, `python/es/train_act.py` **only** to print
`torch.__version__` and the optimizer's hyperparameters in its JSON summary (no training-loop
change — that is T3/T4).

## oracle

1. `cargo test -p es --test cli train_dry_run_plan_is_the_golden` — `--dry-run` on both fixture
   recipes; stdout and `plan.txt` byte-identical to the goldens (paths relative to `<out>`, so the
   golden is machine-independent). No Python needed.
2. `cargo test -p es --test cli train_identity_is_a_function_of_the_recipe` — the pre-run
   `identity_hash` of the same recipe in two scratch dirs is equal; changing `seed` or `lr` moves
   it; the `training/` files exist and each is valid canonical JSON; no slot is an all-zero digest
   (`{"unset": true}` is allowed and counted). No Python needed.
3. `cargo test -p es --test cli train_ir_path_packs_a_bundle_torch_opens -- --ignored` — with
   `ES_PYTHON` set to an interpreter that has torch: 40 steps on the demo bundle and the bake
   fixture (`write_bake_fixture`), then `TorchRuntime` opens `checkpoints/40.esb` and infers a
   finite chunk; `checkpoint.manifest` names it with the right blake3; `training_hash` is set and
   differs from `identity_hash`. Prints `SKIP: <reason>` without `ES_PYTHON`.
4. `cargo test -p es --test cli train_refuses_by_name` — the refusals above each name the field.
5. `cargo xtask ci` green; `cargo xtask check-scope docs/packets/M7/T1-es-train.md` clean.

## acceptance

Oracles 1, 2, 4 pass locally; 3 passes on the oracle server (`~/venvs/es-lerobot-cuda`), and the
external path's `--dry-run` plan is *run once by hand* on the server against
`~/artifacts/plan-v/v15/ds-train` with `steps = 200` to prove the command line is one
`lerobot-train` accepts (record the wall-clock and the `training.lock` in the design note; this
is an observation, not a gate). The design note records: the recipe schema, the pre-run/post-run
split of §19.3 slots, what `hardware.json` holds, and what is `unset` and why.

## forbidden

`crates/es-ir/**` (no IR change; the recipe is a `es-data` document, not a sixth IR);
`crates/es-policy/**` (T3/T5 own the lowering); `crates/es-safety/**`; the training loop of
`train_act.py`; `crates/es/src/cmd/{dataset,policy,loop}.rs` beyond making a function `pub(crate)`
if one is needed; `docs/ARCHITECTURE*.md`; every existing golden and fixture; a new dependency in
`es` or `es-data`. INV-16: nothing pickle-based is read or written. INV-17: no new trait.
