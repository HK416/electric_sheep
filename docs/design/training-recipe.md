# The training recipe — `es train`

Spec: §13.1 (each stage of the loop is a CLI command *and* an artifact), §19.3 (Training
Identity — the `training/` bundle and `training_hash`), §8.8, §2.3 (Python on the learning
path only), §1.4, §28.10 rule 2 (`es train` fills every §19.3 slot with a real value or marks
it unset).

Packet: `docs/packets/M7/T1-es-train.md`. Predecessors: M5 V2/V2b (`train_act.py`,
`es dataset bake`), V8/V19 (`import-lerobot`, `lerobot-train`), M3 W7
(`docs/design/learning-loop.md` section 1, where `training_identity.json` is all zeros except
the dataset slot). This note is the human-review artifact; `crates/es-data/src/training.rs`
and `crates/es/src/cmd/train.rs` are downstream of it.

---

## 1. What one command replaces

Before this packet, one training cycle was nine hand-written scripts. §28.10's "as built"
table counts them and `docs/packets/M5/V19-external-act-current-data.md`'s server section is
the evidence: `mkdir frames-in && ln -s …`, then `es dataset export` with six flags, then
`lerobot-train` with eleven, then `es policy import-lerobot` with five, with `observation_hash`
and `task_hash` copied by hand between them. The IR route was the same shape with `bake`,
`lower`, `train_act.py` and `pack`. Nothing wrote §19.3's `training/`.

`es train --recipe training.toml [--out <dir>] [--dry-run] [--allow-retired-task <hex>]` is
one document and one command for both. The `es` steps of the plan run **in-process** — they
are functions in `crates/es/src/cmd/{dataset,policy}.rs`, and this packet's only edit to
those files is making five of them `pub(crate)` — and the Python trainer is the one
subprocess, which is exactly where §2.3 draws the line.

**Run it from the repository root.** The IR route's trainer is `python/es/train_act.py` and a
Task IR's `scene.path` is repository-relative; both resolve against the working directory,
like every other `es` path flag. Relative paths inside the recipe resolve the same way, and
that is also what makes the `--dry-run` plan reproducible (section 3).

## 2. The recipe

```toml
kind = "training"

[dataset]
root   = "runs/collect-001/ds"        # a LeRobot v2.1 root, as `es loop collect` writes it
frames = "runs/collect-001/frames"    # the flat <NNNNNN>.bin tiles; required for an image input

[policy]                              # exactly one of `bundle` and `lerobot`
bundle = "runs/collect-001/untrained.esb"
# lerobot     = { type = "act", chunk_size = 16, n_action_steps = 16, extra = [] }
# task        = "…/task.toml"         # the three documents the import carries into the bundle
# observation = "…/observation-v8.toml"
# deployment  = "…/deployment.toml"

[run]
steps         = 20000
batch         = 8
lr            = 1e-4
seed          = 0
checkpoint_at = [1000, 5000, 20000]
device        = "cuda"
interpreter   = "python"              # ES_PYTHON wins when it is set
# optional, and absent means absent (section 10):
# schedule     = { kind = "warmup_cosine", warmup = 250, lr_min = 1e-6 }
# weight_decay = 0.01                 # AdamW's; absent is torch's own 1e-2
# grad_clip    = 1.0                  # gradient-norm clip; absent is off
```

Every table is `deny_unknown_fields`: a typo is a refusal, not a silently ignored knob. The
schema, the plan builder and the identity assembly are in `es-data` and are unit-tested
without a process, a file or a GPU (`crates/es-data/src/training.rs`); the CLI is the shell
around them. TOML is read through `es_ir::serial::parse_toml`, which is where this repository's
TOML reader already lives — a recipe is a document like every other one, and no dependency
was added to `es-data` for it.

Two schema decisions worth naming:

* **`steps` is the authority on the length of the run**, so it is always the last checkpoint
  mark. `train_act.py --checkpoint-at` caps its run at the largest mark, so any other reading
  would let `checkpoint_at` silently shorten the run.
* **there is no `amp` field.** `train_act.py`'s own docstring calls `--amp` a sweep flag —
  "use them for a sweep, not for a run whose numbers are quoted" — and a recipe exists to
  describe a run whose numbers are quoted. `precision.json` therefore reads `fp32 / amp off`
  today. T4 adds the field if a recorded sweep ever needs it.

The two committed recipes are `tests/fixtures/visible-learning/training.toml` and
`training-lerobot.toml`. The external one names **`observation-v8.toml`**, not
`observation.toml`: an external checkpoint's input names are its dataset's feature names with
the dots replaced by underscores (§8.4, `XIR-010`), and `observation-v8.toml` is the committed
Observation IR whose ports carry exactly those names. Pointing it at `observation.toml`
produces a refusal that names the port — measured, section 7.

## 3. The plan, and why it is a golden

```
# route: ir
es dataset bake --policy … --out baked --frames … <root>
es policy lower --policy … --out module
python python/es/train_act.py --module module --baked baked --out weights/model.safetensors
    --checkpoint-at 1000,5000,20000 --seed 0 --batch 8 --lr 0.0001 --device cuda
    --loss-curve metrics/loss.json
es policy pack --policy … --weights weights/model-1000.safetensors --out checkpoints/1000.esb
… one per mark
```

```
# route: external
es dataset export --lerobot-v3 <root> --out ds-v3 --frames frames-in
    --drop action_commanded,action_source,intervention --state-dim 6
python -m lerobot.scripts.lerobot_train --dataset.repo_id=es/train --dataset.root=ds-v3
    --policy.type=act --policy.chunk_size=16 --policy.n_action_steps=16 --policy.device=cuda
    --policy.push_to_hub=false --policy.optimizer_lr=0.0001 --steps=20000 --batch_size=8
    --seed=0 --save_freq=10000 --output_dir=lerobot --job_name=es-train --wandb.enable=false
es policy import-lerobot --checkpoint lerobot/checkpoints/010000/pretrained_model
    --task … --observation … --deployment … --out checkpoints/10000.esb
… one per mark
```

`tests/golden/train/plan-ir.txt` and `plan-lerobot.txt` are those two, verbatim, one line per
step. Three things make them a property of the recipe alone rather than of the machine:

1. **Every path under `<out>` is printed relative to `<out>`**, wherever it appears in the
   word — `--output_dir=lerobot`, not only a bare argument — and every separator is `/`. So
   Windows and Linux, and any two scratch directories, render one plan.
2. **Every other path is the recipe's own word, verbatim.** The recipe is the input; the plan
   does not rewrite it.
3. **The oracle removes `ES_PYTHON`.** The interpreter is the one machine-dependent word, and
   `es train` resolves it the same way for the plan and for the run — so the honest way to pin
   a golden is to take it with the override off, not to pretend the plan does not use it.

`--dry-run` builds the plan from the recipe and, on the external route, from the Observation
IR the recipe names; **it opens no dataset and no bundle**, which is what lets CI judge it with
neither on disk and without Python. It writes `<out>/training/plan.txt` and nothing else: a run
that did not happen claims no identity.

Two derived numbers in those lines:

* **`--state-dim`** is the sum of the Observation IR's `StateInput` output widths. For the
  demo's merged observation that is `6 + 7 = 13`, the number M5/V19 exported by hand; for
  `observation-v8.toml` it is 6, V8's. It is a sum and not a resolved index range, because
  resolving a source id against `ModelInfo` needs a scene and `es-data` is layer 10 beside
  `es-eval` (§4.2) and cannot ask. Two channels that read overlapping `qpos` would over-count
  and the export would carry columns nothing reads — wasteful, never wrong.
* **`--save_freq`** is the smallest checkpoint mark, because `lerobot-train` saves at one
  frequency and not at a list. A mark that is not a multiple of it is refused by name rather
  than rounded: a checkpoint that is not on disk cannot be imported, and moving a mark quietly
  would put the wrong step number in `checkpoint.manifest`.

**`frames-in`.** `es dataset export --frames` wants one directory per camera and
`es loop collect --frames` writes a flat one; V19 bridged the two by hand with `mkdir` and
`ln -s`. On the external route `es train` hard-links the tiles into `<out>/frames-in/<camera>`
(copying only if the link fails, skipping what is already there), because a recipe that still
needed a symlink built by hand would not be one command.

## 4. §19.3's twelve slots: what is known before the trainer runs

`<out>/training/` always holds twelve files and `<out>/training.lock` always carries two
digests:

| | slot | written from | when |
|---|---|---|---|
| 1 | `config.json` | the recipe as canonical JSON, the resolved interpreter, the `<out>`-relative plan | pre-run |
| 2 | `optimizer.json` | `AdamW` + `lr`; on the IR route also the betas/eps and the weight decay the trainer is told to use, plus `grad_clip` when `[run]` sets one (section 10) | pre-run |
| 3 | `scheduler.json` | `{"kind":"constant","lr":…}`, or the `warmup_cosine` block `[run] schedule` names (section 10) | pre-run |
| 4 | `seed.json` | `global` and `dataloader` from `[run] seed`; `augmentation` unset | pre-run |
| 5 | `dataset.lock` | `es-data::identity`'s content/schema/split, the episode and frame counts, the recorded `es:task:` name | pre-run |
| 6 | `base_model.lock` | `{"source":"none"}` (IR) or the declared `vision_backbone` + `pretrained_backbone_weights` (external) | pre-run |
| 7 | `augmentation.json` | `{"kind":"none"}` until T6 | pre-run |
| 8 | `precision.json` | `fp32`, `amp off`, `gradient_accumulation` = `[run] batch` on the IR route | pre-run |
| 9 | `topology.json` | `{"world_size":1}` | pre-run |
| 10 | `checkpoint.manifest` | step, bundle path, the bundle's own §5.3 `policy_hash`, blake3 of the bundle bytes | post-run |
| 11 | `metrics.json` | `train_act.py`'s loss curve and its JSON summary | post-run |
| 12 | `hardware.json` | the interpreter probe's reply, plus the `es` version | post-run |

`identity_hash = TrainingIdentity::training_hash` over the twelve with 10–12 still `{"unset":
true}`; `training_hash` is the same function over the twelve once they are filled. Both are in
`training.lock`, beside the blake3 of each file and, per checkpoint, §19.3's
`policy_hash = H(training_hash, checkpoint_hash)`.

**The identity is written before any step runs**, right after the refusals and before the
interpreter is even probed. That is the point of the split: the name of a run exists before a
single GPU-second is spent on it, and two runs of one recipe on one machine are the same run
before either finishes. `crates/es/tests/cli.rs::train_identity_is_a_function_of_the_recipe`
is that property end to end, and section 7 measures it on the oracle server.

**Every slot is the blake3 of a file that exists and parses.** A slot the run genuinely does
not know is the file `{"unset":true}` and is hashed as such. `es loop distill` writes zeros
because it is on the other side of §2.3's boundary and knows nothing
(`docs/design/learning-loop.md` section 1); `es train` is on this side and knows almost
everything, so "unknown" has to be a value it can defend. There is no all-zero digest anywhere
in a `training.lock` this command writes, and the oracle asserts it.

**What is `unset` today, and why.**

| unset | why | owner |
|---|---|---|
| `seed.json.augmentation` | there is no augmentation | T6 |
| `base_model.lock.weights_hash` and `.license` | a *declared* provenance: the backbone is LeRobot's ACT default unless `extra` overrides it, and nothing here has downloaded or verified those weights | T5 |
| `optimizer.json.betas` / `.weight_decay` on the external route | `lerobot`'s optimizer block is not this side's to declare, and T4 therefore refuses `[run] schedule`, `weight_decay` and `grad_clip` on that route rather than declaring a schedule the run never applied (section 10) | T2 |
| `metrics.json.loss` on the external route | `lerobot-train` reports its curve to its own logs, not to a file this command reads | T2 |
| `hardware.json.driver` | `torch` reports the CUDA toolkit, not the driver | — |
| `hardware.json.git_describe` | there is no build script and `Cargo.lock` is gitignored (M5 review S-8 / R7), so a build cannot be named yet; claiming a revision would be the fabrication §28.10 rule 2 forbids | M5 R7 |

**`metrics.json`, not `metrics.parquet`.** §19.3 names `metrics.parquet`. `es-data`'s parquet
writer (`crates/es-data/src/lerobot/{v3,columns}.rs`) is shaped around LeRobot's `Info` and
`Episode` — features, chunks, episode indices, a path template — and is `pub(crate)`. Writing a
two-column scalar table through it is not "a few lines", it is a second writer. The packet's own
condition therefore fails and the slot is JSON; if a curve ever needs to be queried rather than
read, that is a packet.

**`hardware.json` comes from one probe.** Before the bake or the export, `es train` runs the
interpreter once with a four-line script that imports `torch` (and `lerobot` on the external
route) and prints `torch.__version__`, `torch.version.cuda` and
`torch.cuda.get_device_name(0)`. That probe is *both* the packet's "an interpreter that cannot
import torch" refusal — it prints the interpreter's own error — and the source of the slot. It
runs early on purpose: a broken environment is a two-second answer, and finding it out after a
ten-minute bake is the kind of thing this command exists to stop.

**`optimizer.json` is declared, then checked.** On the IR route `train_act.py` now reports the
optimizer it built and `torch.__version__` in its JSON summary (the only change this packet
makes to that file; the training loop is T3/T4's). `es train` compares the report against the
`optimizer.json` it wrote before the run and prints a warning naming both if they differ —
which would mean torch moved its `AdamW` defaults and the declaration is stale. It does not
refuse: the checkpoints are on disk by then, and a stale declaration is a thing to fix in T4,
not a reason to throw a run away.

**`checkpoint.manifest` carries the *bundle's* `policy_hash`, not §19.3's.** §19.3 defines
`policy_hash = H(training_hash, checkpoint_hash)`, and `training_hash` covers
`checkpoint.manifest` — so that number cannot live inside the file it is computed from. The
manifest names the bundle's own §5.3 `policy_hash` (the one `es policy pack` prints) under
`bundle_policy_hash`, and `training.lock` carries §19.3's, per checkpoint. Both exist; neither
is circular.

## 5. The refusals, each by name

| refusal | message names |
|---|---|
| `[policy]` sets both `bundle` and `lerobot`, or neither | which, and what each route means |
| the external route without `task` / `observation` / `deployment` | the missing key |
| an Observation IR with an image input and no `[dataset] frames` | `frames`, and that a zero-filled image channel trains a policy that looks fine |
| `checkpoint_at` that `lerobot-train`'s single `--save_freq` cannot produce | the marks and the frequency |
| a dataset whose recorded `task_hash` is not the policy's | both hashes, and `--allow-retired-task <hex>` |
| an interpreter that cannot import what the route needs | the interpreter, and its own stderr |

The task-provenance one is M5 review **S-3/R4**, surfaced here: the demonstrations collected
against `gripper > 0.6` and those against `> 0.85` sit side by side on the oracle server, and
V19's packet named the wrong one and measured 0/16 by construction. `es loop collect` records
`es:task:<hex>` in `meta/tasks.jsonl`; `es train` compares it with the Task IR the recipe
names and refuses unless `--allow-retired-task` **names the retired hash** — so accepting one
is a deliberate, recorded act and not a flag you can leave on. A dataset with no `es:task:`
name (an imported one) carries no claim and is not checked.

## 6. The oracles

| # | command | needs |
|---|---|---|
| 1 | `cargo test -p es --test cli train_dry_run_plan_is_the_golden` | nothing |
| 2 | `cargo test -p es --test cli train_identity_is_a_function_of_the_recipe` | nothing |
| 3 | `cargo test -p es --test cli train_ir_path_packs_a_bundle_torch_opens -- --ignored` | `ES_PYTHON` with torch, MuJoCo |
| 4 | `cargo test -p es --test cli train_refuses_by_name` | nothing |
| — | `cargo test -p es-data --lib training` | nothing (the headless half: schema, plan, identity) |
| 5 | `cargo test -p es-policy --test ir_training lr_schedule_matches_the_golden` | nothing; the interpreter half runs when `ES_PYTHON` is set and prints `SKIP` otherwise (section 10) |
| 6 | `cargo test -p es --test cli train_identity_moves_with_the_schedule` | nothing |
| 7 | `cargo test -p es-policy --test ir_training -- --ignored the_default_schedule_is_the_old_run` | `ES_PYTHON` with torch |

Oracle 2 runs a real `es train` whose recipe names an interpreter that cannot exist, so the
run always stops — and the identity it asserts is the one written before it stopped. That is
the pre-run/post-run split as an executable statement, and it needs no Python.

The goldens are regenerated only by `cargo test -p es --test cli -- --ignored
generate_train_goldens`, never by hand (§1.4).

## 7. Measured — oracle server, RTX 4090, 2026-09-16

Oracle 3, `ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`: **passes**. 40 steps on the demo
bundle and the bake fixture (4 episodes × 16 frames), `TorchRuntime::load` accepts
`checkpoints/40.esb` and `infer` returns a finite chunk, `checkpoint.manifest` names it with
the blake3 of its bytes, `training_hash 50f17bc2…` ≠ `identity_hash`.

The acceptance's external run — the plan of `training-lerobot.toml` against V15's 200
demonstrations at `steps = 200`, run once to prove the command line is one `lerobot-train`
accepts. This is an **observation, not a gate**.

```sh
cd ~/Projects/es-t1
ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python \
  ./target/release/es train --recipe ~/artifacts/plan-v/m7-t1/training-v15-v8.toml \
                            --out ~/artifacts/plan-v/m7-t1/run2
```

`lerobot-train` accepted every flag the plan builds, `--policy.optimizer_lr=0.0001` included
(its own log line reads `lr:1.0e-04`). Export, 200 steps, import and the lock, in one command:
**19.22 s wall-clock** (the training itself 6.4 s at ~57 step/s; 36,960 frames, 200 episodes,
52M parameters, `loss 4.458`, `l1_loss 0.323`).

```json
{ "schema_version": 1,
  "identity_hash": "476ed25a414a95c4dcd44748127a304eccc393ad5caa3ff7eb1e6e932e293e1f",
  "training_hash": "bd1b3c863877ed0ea923f0995019648c3203f58097e488686d9fe17b4162578a",
  "files": {
    "augmentation.json":   "7d75c7e1f7056cd846c4c047af66b8a6ff019e0b8be156c9e7e757814ac7bf8f",
    "base_model.lock":     "00ea2e622bdbbdd477695b9c317337fda29ec400edfaa76410c75345f13f002e",
    "checkpoint.manifest": "18fdf2cb808378d1a0df31f3506b1e9ccb340ed1199dbd614cd698d7781392d6",
    "config.json":         "e0da61a6b4b6a151d1b7e0080f3b5c2706ef1241ac835f893a2994faa674cc64",
    "dataset.lock":        "a7aad6ab4e04c2b9330cf9dcf7cbd3d5b24554389e492b1b9cbf2322316cb2cf",
    "hardware.json":       "a983e8821036497cdb0017a0fff27f81c8383e8458f69a27385ed64267265033",
    "metrics.json":        "4f62d486b5a031896112a1d11b9254b130bf7997c5ef66527ca7362b338960e9",
    "optimizer.json":      "35a78de1a1aa9a7fd8f09a63897fdfbd469739499697057c782699ac3ac2a9ca",
    "precision.json":      "033d85b1122382be956cefc988393ccafa22260cca72ec3fa53d20a41b612cd7",
    "scheduler.json":      "8847154d020d177c6f68fddf7bbd89ffee17591758a4b6719e0d10fd51afe7aa",
    "seed.json":           "2f6d6194c4517524d5cad5522335f2ceb612722721babe40bf3ea83125dace10",
    "topology.json":       "79cf76183503dd7106c635e79a4d57c6042d06059bf6a8ccdba805d1ee777341" },
  "checkpoints": [ { "step": 200, "bundle": "checkpoints/200.esb",
                     "policy_hash": "10a54412b5188c08e27fd4aae09f8edac8c1f25be642331662480d8b550be0b2" } ] }
```

`hardware.json` from the probe: `torch 2.11.0+cu129`, `cuda 12.9`,
`device_name "NVIDIA GeForce RTX 4090"`, `lerobot 0.6.1`, `trainer "lerobot-train"`,
`es_version 0.1.0`, `driver` and `git_describe` unset. `dataset.lock`:
`content 55a958e1…`, `schema 1f5ddafc…`, `split 872fe162…` (all-train), 200 episodes,
36,960 frames, `recorded_task es:task:eb6efefa…` — which is the committed `task.toml`'s own
`task_hash`, so the S-3 check passed on the real data rather than only on a fixture. The
imported bundle: `observation_hash c69a8e11…`, `learning_hash cc4587b9…`,
`policy_hash a7e06b32…`, `deployment_hash f2f9a510…`.

**The same recipe a second time**, into a different `<out>` (`run3`, 21.04 s): one
`identity_hash 476ed25a…`, a different `training_hash 452ae7d7…`. Exactly one of the twelve
files differs — `checkpoint.manifest` — because `lerobot-train` at the same seed does not
produce bit-identical weights. That is the split doing its job: `identity_hash` names the
recipe, `training_hash` names the run. Whether an external trainer *ought* to be
bit-reproducible is open question 29 below.

The first attempt (`run1`) used the committed `observation.toml` and was refused at the import
with the port-name message quoted in section 2 — the refusal working, and the reason
`training-lerobot.toml` names `observation-v8.toml`.

## 8. Deviations from the packet

1. **The trainer's reported version is in `hardware.json`, not `config.json`.** The packet
   lists it in both. `config.json` has to be a pre-run slot for `identity_hash` to exist before
   the trainer runs, and a version reported by the trainer is not knowable then.
2. **`checkpoint.manifest` names `bundle_policy_hash`** and `training.lock` names §19.3's
   `policy_hash` — the packet's wording is circular, see section 4.
3. **`--policy.optimizer_lr` is on the external command line**, which the packet's flag list
   does not mention. Without it `[run] lr` would enter `optimizer.json` and the hash while
   having no effect on the run, which would be a lie. Verified accepted by `lerobot-train`
   0.6.1, section 7.
4. **`--dataset.repo_id=es/train` and `--job_name=es-train` are constants.** Both are names
   `lerobot-train` requires and neither identifies anything here; deriving them from a path
   would put the scratch directory in the golden.
5. **The `## context` section of the packet gained a fenced glob list** so that
   `cargo xtask check-scope` can parse it; the prose list is unchanged beside it. The parser
   wants a literal `## context` heading and a fence or a bullet list
   (`xtask/src/scope.rs::parse_context_globs`), and `xtask/` is outside this packet's scope.

**Packet M7/T4** (section 10) deviates in three further places.

6. **The four measured rows were run through `train_act.py` directly, not through `es train`,
   so what is kept per row is its JSON summary and its `--loss-curve`, not a `training.lock`.**
   T4's packet asks for a lock per row. `--resident-gpu` is a trainer flag and deliberately not
   a recipe field (it changes where a tensor lives, not what a run is, and section 2's rule is
   that a recipe describes a run whose numbers are quoted), so a row measured through
   `es train` would be a different run from T3's baseline it has to be comparable with. What is
   kept instead: `~/artifacts/plan-v/m7-t4/{A,B,C,D}.json` and `*-curve.json`, plus one real
   `es train` lock from oracle 3 re-run after this packet (`train-ir-training.lock`), whose
   `scheduler.json` digest is the pre-T4 one.
7. **`the_default_schedule_is_the_old_run` cannot compare against the previous script from
   inside the repository**, because that file is not in the tree and a torch loss curve is not
   a legitimate golden (this note's own section 9, and `ir_training.rs`'s header: two runs of
   one optimizer do not agree across torch versions). The in-tree test compares the default
   against every new flag at its default and pins the optimizer block; the comparison against
   `main`'s own script is the server measurement in section 10.
8. **The fixture recipe stays without a schedule**, which the packet left to this note to
   decide — section 10's last subsection says why.

## 9. Open questions

* **29.** `lerobot-train` at one seed does not produce bit-identical weights (section 7), so
  `training_hash` differs between two runs of one recipe on one machine. §3.5 tier 1 is about
  *our* kernels, and an external trainer is not one of them — but the evidence bundle should
  say which tier a checkpoint's provenance belongs to rather than leaving the reader to infer
  it from two hashes that disagree.
* **30.** `--state-dim` is a sum of `StateInput` widths (section 3). The exact answer is the
  union of the resolved `qpos` ranges, which needs a scene and a layer `es-data` may not reach.
  Either the Observation IR should declare the width it reads from the recorded row, or the
  derivation belongs in `es-eval` and `es train` should ask the CLI for it.
* **31.** `es train` must be run from the repository root because `python/es/train_act.py` and
  a Task IR's `scene.path` are repository-relative. An installed `es` has no repository. The
  trainer's location probably wants to be a document value, or `train_act.py` wants to be data
  the binary carries — T3 touches that file anyway.
* **33.** (32 is `visible-learning.md`'s.) `lr_curve_hash` is `null` when the interpreter cannot import `blake3` (section 10).
  Every other digest in this repository is blake3 and computed in Rust; this one is computed
  in Python because the trainer is the only thing that knows which rates it applied. Either
  the trainer should write the applied rates to a file beside `--loss-curve` and let `es train`
  hash them, or `blake3` should join `torch` as a declared dependency of the learning path
  (`python/es/pyproject.toml` already declares it for the USD bake).
* **34.** A schedule is refused on the lerobot route rather than translated (section 10). ACT's
  own config carries `optimizer_lr` and `optimizer_weight_decay`
  (`docs/api-notes/lerobot-config.md`, the "training-only" row), so a mapping from `[run]`
  onto whatever `lerobot-train` 0.6.1 calls its schedule is possible and would make one
  document describe both routes — which is the point of a recipe. It needs that side's flag
  names pinned in the api-note first; until then `[policy] lerobot.extra` is the escape hatch.

---

## 10. The schedule (packet M7/T4)

§28.9's "where the wall-clock goes" ends with one line about large batches: *linear scaling was
measured to diverge, so what is needed is a schedule, not a convention.* This section is that
line turned into a document field, a function, a golden and four measured runs.

**The three flags, and what is a document value.** `train_act.py` gains `--schedule
constant|warmup_cosine`, `--warmup-steps N`, `--lr-min F`, `--weight-decay F` (torch's own
`1e-2`, **made explicit** so `optimizer.json` names a number the script was told rather than
one it assumes) and `--grad-clip F` (`0` = off). The recipe names them in `[run]`:

```toml
[run]
steps        = 20000
batch        = 64
lr           = 4e-4
schedule     = { kind = "warmup_cosine", warmup = 250, lr_min = 1e-6 }
weight_decay = 0.01                   # optional; absent is torch's 1e-2
grad_clip    = 1.0                    # optional; absent is off
```

`es train` puts them on the trainer's command line, writes `scheduler.json` as
`{"kind":"warmup_cosine","lr":…,"lr_min":…,"warmup":…,"total_steps":…}` and `optimizer.json`
with the weight decay and the clip, so `identity_hash` moves with every one of them. **The
`total_steps` is part of the schedule and not decoration**: the cosine's period is the length
of the run, so the same `warmup`/`lr_min` pair at 2,500 and at 20,000 steps are two different
schedules — which is exactly what rows C and D below are.

**The schedule is a plain function, not `torch.optim.lr_scheduler`.**

```
lr(step) = lr * step / warmup                                          step < warmup
         = lr_min + (lr - lr_min) * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))
```

`python/es/train_act.py::lr_at` is those three lines and is called once per step. A torch
scheduler's float sequence is an implementation detail of a torch version, and a run has to be
reproducible from its `scheduler.json` across them — so the sequence is pinned here instead.
`tests/golden/train/lr_warmup_cosine.json` holds the first 1,000 values for the row-D schedule
(`total 20000, lr 4e-4, lr_min 1e-6, warmup 250`), generated once by the `#[ignore]`d
`generate_lr_golden` from the script itself on the oracle server;
`ir_training::lr_schedule_matches_the_golden` compares **a Rust re-implementation of the same
three lines** to it bit for bit, and, when `ES_PYTHON` is set, the script's own `lr_at` too.
The Rust half needs no interpreter, which is what lets CI judge the schedule at all.

That the two agree bitwise across libms was the open risk — the cosine is the one transcendental
here, and a golden produced by glibc's `cos` on the server is read by MSVC's `cos` on the
Windows host. Measured: **1,000 of 1,000 values identical in `f64`**, warmup segment and cosine
segment alike. The argument of the cosine stays in `[0, 0.119]` over the golden's range, where
both libms are correctly rounded; a golden that covered the whole decay might not be so lucky,
and that is the reason the test prints how many values differ rather than only asserting.

**`constant` is the default, and the default is the old run.** Three separate statements, each
with its own check:

1. *nothing on the command line.* A recipe that names no schedule renders the plan it always
   did, so `tests/golden/train/plan-ir.txt` is unchanged, byte for byte
   (`cli::train_dry_run_plan_is_the_golden`).
2. *nothing in the identity.* The new keys enter `scheduler.json` and `optimizer.json` only
   when a recipe asks for them, so a pre-T4 recipe keeps its `identity_hash` — the run recorded
   in section 7 is still the same run (`es-data::a_recipe_without_a_schedule_is_the_run_of_before`,
   `cli::train_identity_moves_with_the_schedule`). Measured rather than argued: oracle 3 of
   section 7 re-run on the server after this packet writes
   `scheduler.json 8847154d…51afe7aa`, which is the digest that table already carried.
3. *nothing in the arithmetic.* `lr_at` is not consulted under `constant`; the optimizer keeps
   the step it was built with. `ir_training::the_default_schedule_is_the_old_run` runs 40 steps
   on CPU with no new flag and 40 with every new flag at its documented default and compares
   the two `--loss-curve` files as **bytes**, and asserts the reported optimizer block is the
   one every measured run in these notes was taken under.

The third is the one a test inside this repository cannot state completely, because the
*previous* script is not in the tree. So it was measured against it once, on the server:
`git show main:python/es/train_act.py` (main at `389ef88`, whose T-track files are identical to
`374d42c`'s) run over the same module, the same baked set, the same seed and the same 40 steps
as the oracle's own run — **the two loss curves are byte-identical**.

**Refused on the lerobot route.** `[run] schedule`, `weight_decay` and `grad_clip` are
`train_act.py`'s flags. `lerobot-train` carries its own optimizer and scheduler configuration,
so accepting them there would write a `scheduler.json` — and therefore a `training_hash` — for
a schedule the run never applied. The recipe refuses by name and points at
`[policy] lerobot.extra` (open question 34).

**`lr_curve_hash`.** The rates actually applied are hashed (blake3 over the `f64` values,
little-endian) and reported in the run summary, so two runs of one schedule can be told apart
by one word and a `constant` run's hash is checkably 40 copies of `--lr`. `blake3` is an
optional import: the script's contract is "no package beyond torch", so an interpreter without
it reports `null` rather than stopping a training run (open question 33). The oracle server's
`es-lerobot-cuda` venv has no `pip`, so the runs below got it through `PYTHONPATH` pointing at
a copy of the `blake3` package from the `es` venv beside it.

### Measured — oracle server, RTX 4090, 2026-09-16

The module is the demo's untrained bundle lowered by this tree's `es policy lower`
(`lowering_hash 3d06811c…d8a2d394`, T3's), the data is V15's 200-demonstration baked set
(36,960 samples, `observation_hash 899c16a9…`), and every run is `--seed 0 --device cuda
--resident-gpu`, torch 2.11.0+cu129. The GPU was idle before the runs (`nvidia-smi`: 56 MiB,
0 %), so no wait was needed.

| run | batch | lr | schedule | steps | samples seen | wall clock | s / 1,000 steps | samples/s | `final_loss` | non-finite step |
|---|---|---|---|---|---|---|---|---|---|---|
| A (baseline) | 8 | 1e-4 | constant | 20,000 | 160,000 | **1:47** (107 s) | 5.4 | 1,495 | 0.018901 | none |
| B | 64 | 1e-4 | constant | 2,500 | 160,000 | **0:26** | 10.4 | 6,154 | 0.020714 | none |
| C | 64 | 4e-4 | warmup_cosine, warmup 250, `lr_min` 1e-6 | 2,500 | 160,000 | **0:26** | 10.4 | 6,154 | **0.012060** | none |
| D | 64 | 4e-4 | warmup_cosine, warmup 250, `lr_min` 1e-6 | 20,000 | 1,280,000 | **2:51** (171 s) | 8.6 | 7,485 | 0.004610 | none |

Per §12.4 no `step/s` figure is quoted; the two rate columns are the run's own units. The short
runs carry the same ~5 s of start-up (module build and the 3.9 GiB resident copy) over one
eighth the steps, which is the whole of the difference between B/C's 10.4 and D's 8.6.

**The packet's question, answered.** *At equal samples seen, does batch 64 with warmup and a
cosine reach a loss at or below batch 8's?* **Yes, and by a third**: C is 0.012060 against A's
0.018901 — 36 % lower — in **26 s against 107 s**. And the schedule is what did it, not the
batch: B is the same 160,000 samples at the same 64 and the same lr as A, and it lands
*above* A at 0.020714. "Not a convention but a schedule" is that pair of rows.

**Nothing diverged, and what that is and is not evidence for.** No row has a non-finite step —
the summary now reports `first_nonfinite_step`, so this is a number and not the absence of a
crash. But C and D differ from `visible-learning.md` 7.11's NaN run in *two* ways: that one
scaled the rate linearly to 8e-4 with no warmup, and these use 4e-4 with 250 steps of warmup
and a decay. T3 already showed batch 64 at the unscaled 1e-4 is finite. So what is measured
here is that **this combination is stable**, not that warmup alone rescues 8e-4; isolating that
would be a fifth run and nothing downstream needs it.

**D is not comparable to A as a fit** — it is eight times the data — and it is not evaluated
here (that is wave 5's U-measurement). Its checkpoint is
`~/artifacts/plan-v/m7-t4/model-D.safetensors` and its `lr_curve_hash` is
`c01d5185b03a5bad9fbe1708b5582f504aabf90c745910713ca0ccaefc15365c`. `final_loss 0.004610` is
the lowest an IR-graph run has recorded on these documents; whether a fit that low is a policy
that succeeds is exactly the question §28.10's stop rule reserves for the re-measurement.

**One number to read with care.** Row A is the same flags T3 measured at
`learning-lowering.md` 5.2 and it reports `final_loss 0.018901` where that run reported
0.019026, with the wall clock 107 s against 137 s. CUDA training does not reproduce run to run
(§28.9 L10), so a 0.7 % difference between two identical invocations is the noise floor of
every number in this table — which is why the comparisons above are 36 % and 76 % differences
and not 1 % ones, and why the bitwise oracle for "the default is unmoved" is on CPU.

### The fixture recipe is unchanged, deliberately

`tests/fixtures/visible-learning/training.toml` still names no schedule. It is the committed
recipe for the measured 20,000-step run at batch 8, and the plan golden beside it is the one
`--dry-run` has to keep producing; showing the new field there would move both. The field is
documented above and exercised by `cli::train_identity_moves_with_the_schedule`, which builds
its own recipe.
