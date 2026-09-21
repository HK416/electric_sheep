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
| 6 | `base_model.lock` | the *verified* provenance of `[policy] base_model` (IR, section 11), `{"source":"none"}` when there is none, or the declared `vision_backbone` + `pretrained_backbone_weights` (external) | pre-run |
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
| `base_model.lock.weights_hash` and `.license`, **on the external route only** | a *declared* provenance: the backbone is LeRobot's ACT default unless `extra` overrides it, and nothing here has downloaded or verified those weights. The IR route's is verified (section 11) | — |
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

**Five flags on the trainer, three fields in the document.** `train_act.py` gains `--schedule
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

## 11. The pretrained backbone and `base_model.lock` (packet M7/T5)

Spec 19.3 says the provenance of a pretrained backbone "directly affects the results, it must
be included in the provenance; it is also the basis for license tracking". Until this packet
the IR route's slot was `{"source":"none"}` and the external route's was a *declaration* with
two `unset` fields. It is now a **verified** record on the IR route, and this is the paragraph
that says what verified means.

The recipe gains one field:

```toml
[policy]
bundle     = "runs/collect-001/untrained.esb"
base_model = "~/artifacts/plan-v/m7-t5/resnet18-imagenet1k-v1.safetensors"
```

It is required by a bundle whose Learning IR declares `VisionEncoder { pretrained = true }`,
refused by one that does not, and refused outright on the `lerobot` route — there the backbone
is `lerobot`'s own and reached through `[policy] lerobot.extra`. Both directions of the first
pair are refusals rather than defaults, and for the same reason: a bundle that wants ImageNet
and gets none would train from scratch under a document saying otherwise, and a recipe naming
weights nothing reads would put a provenance into `base_model.lock` that the run does not have
— the fabrication §28.10 rule 2 forbids.

**Three claims have to agree before a single GPU-second is spent.** `Backbone::verify` reads
the file and `<stem>.lock.json` beside it, then checks, in this order:

| checked | refused when | why this order |
|---|---|---|
| the lock describes these bytes | `lock.blake3 != blake3(file)` | is this lock file about this artifact at all |
| the source | `lock.source` is not `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` | the licence decision is about a *named* source (§29 row, owner 2026-09-15) |
| the licence | `lock.license` is empty | §19.3 makes this file the basis for licence tracking; an empty slot tracks nothing |
| the pin | `blake3(file) != RESNET18_IMAGENET1K_V1_BLAKE3` | the first three ask whether the lock is a well-formed record of these bytes; this one asks whether these bytes are the artifact the repository has measured |

The pin is `8511928e…9e801899`, one `pub const` in `crates/es-data/src/training.rs`. The 45 MB
file is never committed — it lives at `~/artifacts/plan-v/m7-t5/` on the oracle server — so the
constant is what the repository knows about it, the way `tests/fixtures/mjcf/*.PROVENANCE.json`
pins the upstream MJCF this project's scene is derived from. There is exactly one copy of the
string: `python/es/fetch_backbone.py` is *given* it with `--expect` rather than holding its
own, and `crates/es-policy/tests/backbone_provenance.rs` reads it out of `training.rs` as text
because `es-data` is layer 10 and `es-policy` is layer 8 (§4.2) and the constant cannot be
imported upward. `fetch_backbone.py --repin` is the deliberate way to move it, and it says on
stderr that the constant and the two design notes have to move in the same commit.

**What goes into the slot, and what deliberately does not.** `base_model.lock` carries
`source`, `url`, `sha256_upstream`, `blake3`, `dropped`, `license` and `license_url` — the
fields that identify *the weights*. The lock file beside the artifact also records the `torch`
and `torchvision` that fetched it, and those are **not** copied in: two machines fetching the
same upstream file write byte-identical tensors and different version strings (measured:
torchvision 0.26.0+cu129 and 0.29.0+cpu produce the same blake3), so carrying them would put
the fetching machine into `identity_hash` and one recipe would have two identities. The
fetching environment belongs in the artifact's own lock file and in the run's `hardware.json`.

`TrainingIdentity.base_model` therefore ends up with a real `source` and a real `license` on
this route, which is what §19.3 asked for: a run that changed nothing but the licence of its
base model is a different run.

**The trainer's line gains `--init-backbone <path>` and nothing else.** `frozen` is a field of
the Learning IR, so the lowered module carries it as `requires_grad_(False)` and `train_act.py`
builds `AdamW` over the parameters that still require a gradient — see
`learning-lowering.md` section 5.3. A `--frozen` flag would be a second copy of an IR decision
on a command line, and the plan golden would then depend on a bundle that `--dry-run` does not
open.

### The refusals, and why they are named rather than numbered

The packet writes `TRAIN-0xx`. Neither `es_data::training` nor `es train` has ever carried
numeric codes — every refusal there names the field and the file, and §17.2's requirement is
that a refusal be identifiable, not that it be enumerated. A numbering scheme invented for
five messages is a scheme with one user. The five are: the lock file that does not describe
its weights (names both digests and the lock's path), the artifact that is not the pinned one
(names both digests and the constant to move), the empty licence (names the file and §19.3),
`pretrained = true` with no `base_model` (names the flag and the fetch command), and
`base_model` with no `pretrained = true` (names the flag and what would be claimed). All five
are `cli::train_refuses_a_mismatched_base_model`.

### The fixture recipe is unchanged here too

`tests/fixtures/visible-learning/training.toml` names no `base_model`, because its bundle is
the from-scratch one and the plan golden beside it must keep rendering. The pretrained arm of
the experiment is `tests/fixtures/visible-learning/learning-pretrained.toml`, and the recipe
that points at it is the U-measurement's, not a committed fixture.

---

## 12. The cycle (packet M7/T2)

§13.1 draws the loop — collect, train, evaluate, watch — and T1 made one rung of it a
command. The other three stayed four commands whose paths and hashes a person threaded by
hand, and `loop.jsonl` stopped at `distill`: it never recorded that a dataset trained a policy
or that a policy was judged. `es loop cycle` is the whole rung under one document and one
ledger.

```
es loop cycle --recipe <cycle.toml> [--out <dir>] [--dry-run] [--from <stage>]
              [--allow-new-evaluation] [--skip-expert-gate]
```

### 12.1 The document names the stages; it does not re-describe them

```toml
kind  = "cycle"
scene = "tests/fixtures/mjcf/so101_pick_place.xml"

[collect]                        # optional; `dataset = "<root>"` instead, to reuse one
policy   = "runs/collect-001/untrained.esb"
expert   = "so101-pick-place"    # omit for a trained policy's own rollouts
episodes = 200
seed     = 1
frames   = true

[train]
recipe = "tests/fixtures/visible-learning/training.toml"   # T1's, by path or inline

[eval]
config     = "tests/fixtures/visible-learning/evaluation.toml"
checkpoint = "last"              # or a mark the recipe's `checkpoint_at` writes
jobs       = 6
frames     = true

[showcase]                       # optional; needs the `render` feature
cell    = "nominal-00"
eye     = [0.66, -0.46, 0.52]
look_at = [0.14, -0.04, 0.04]
fov     = 36
width   = 1280
height  = 720
```

The rule section 2 states for the recipe holds here too: **a cycle carries no parameter a
stage already owns**. `[train]` is T1's recipe by path — not a copy of its fields — and
`[eval]` is an Evaluation IR by path, so `evaluation_hash` is the document's own and not a
rendering of it. The one thing the cycle *does* override is the training recipe's
`[dataset] root`/`frames`: they become this cycle's collect output, because a cycle whose
training read some other directory chains nothing (§13.3). That override is visible in the
nested plan rather than in a rewritten file.

**Every stage is the function the command already is** — `cmd::r#loop::collect`,
`cmd::eval::run`, `cmd::train::run`, `cmd::showcase::run` — called in-process with the same
words the plan prints, so a printed line and an executed stage cannot drift. Only what those
already spawn (the physics subprocess, the trainer, `--jobs` workers) is a process.
`crates/es/src/cmd/cycle.rs` is thin for T1's reason: the document, the plan and the ledger
steps are `es_data::training` and `es_data::collect`, which are headless and unit-tested.

### 12.2 The stage plan is a golden

`--dry-run` prints one line per stage, every path under `<out>` written relative to it, every
separator a `/`, and T1's plan indented under the `train` line — the same three rules that
make the training plan a property of the recipe alone (section 3):

```
# cycle: collect -> expert-gate -> train -> eval -> showcase
es loop collect --policy runs/collect-001/untrained.esb --scene .../so101_pick_place.xml --episodes 200 --seed 1 --out collect/ds --frames collect/frames --expert so101-pick-place
es eval run --config .../evaluation.toml --policy runs/collect-001/untrained.esb --scene .../so101_pick_place.xml --out eval-expert --jobs 6 --frames eval-expert/frames --expert so101-pick-place
es train --recipe .../training.toml --out train
  # route: ir
  es dataset bake --policy runs/collect-001/untrained.esb --out train/baked --frames collect/frames collect/ds
  ...
es eval run --config .../evaluation.toml --policy train/checkpoints/20000.esb --scene .../so101_pick_place.xml --out eval --jobs 6 --frames eval/frames
es video showcase --run eval --scene .../so101_pick_place.xml --out showcase --cell nominal-00 --eye 0.66,-0.46,0.52 --look-at 0.14,-0.04,0.04 --fov 36 --width 1280 --height 720
```

`tests/golden/train/plan-cycle.txt` pins it byte for byte. Nothing on disk is read to build
it — not the dataset, not the bundle, not Python — which is what makes it judgeable in CI on a
machine that has none of them. A run that did not happen writes nothing, not even a directory.

### 12.3 Two refusals are the point of the command

**The harness passes the expert first** (§28.9 rule 1, M5-R1). With `[collect] expert` set,
the cycle runs the expert through `es eval run` on the *same* `[eval] config` **before**
anything trains, and a failed acceptance stops the cycle. A harness the expert cannot pass is
a harness no policy can pass, and a GPU-hour of training is an expensive way to discover that.
The gate's report is kept beside the policy's, under `<out>/eval-expert/`, because the
evidence has to survive the run that comes after it. `--skip-expert-gate` runs anyway and
records the deviation in the ledger (`expert_gate = "skipped (--skip-expert-gate)"`), so a
cycle that trained without the gate does not look like one that passed it.

**A moved `evaluation_hash` is refused by name** (§13.3). `<out>` is reused for iteration 2,
so `<out>/loop.jsonl` may already hold an `evaluate` step. If the new one's `evaluation_hash`
differs, the comparison the ledger invites is not one — §13.3's "keeping the evaluation
conditions fixed while changing only data·policy is the discipline" — and the cycle refuses
with **both hashes printed**, before a GPU is touched (the check runs under `--dry-run` too,
because the evaluation conditions are a property of the document). `--allow-new-evaluation` is
the deliberate act that starts a new comparison. From iteration 2 on, `es eval compare` runs
on the two reports at the end: the previous `report.json` is moved to `report-prev.json`
before the new one overwrites it.

### 12.4 `--from <stage>` resumes, and checks before it does

`--from collect|train|eval|showcase` skips the earlier stages and reads their outputs from
under `<out>`. It refuses if they are missing or disagree with the ledger: the dataset's
recomputed `content` against the ledger's `collect` step, the checkpoint bundle on disk and
its `policy_hash` in `training.lock`, and `eval/report.json` for `--from showcase`. A resumed
cycle whose stages are not one cycle is the failure mode this exists to prevent.

### 12.5 The ledger reaches the end of the loop

`loop.jsonl` gains `train` and `evaluate` steps and `es_data::check_chain`; both are described
in `docs/design/learning-loop.md` section 4.1, which is their home. The cycle appends each to
the dataset root's ledger and to `<out>/loop.jsonl`, and calls `check_chain` once the run is
over. One real cycle's ledger, in order:

```
collect  ->  evaluate (the expert gate)  ->  train  ->  evaluate (the policy)
```

### 12.6 Deviation from the packet — `es eval run --expert`

The packet allows touching `eval.rs` **only** to expose its entry as `pub(crate)`. The
implementation also added a flag: `es eval run --expert <name>`, which drives the scripted
demonstrator instead of the bundle's weights (`ExpertPolicy` and `SeenState` in
`crates/es/src/cmd/loop.rs`, wired through `crates/es/src/cmd/eval.rs`). This is recorded as a
deviation rather than argued away, but the gate cannot exist without it: the packet's own
spec says the gate is "the expert through `es eval run` on the same `[eval].config`", and
before T2 there was no way to run `es eval run` on anything but a policy's weights. The
scaffold itself is not new — `expert_passes_the_evaluation_harness` has driven the harness
with the expert since packet M5/V6; T2 promoted it out of the test file into the command that
needs it. `ExpertPolicy` is an impl of `PolicyRuntime`, one of INV-17's seven, not an eighth
extension point.

What it changes for a run *without* the flag: nothing. Every existing eval test is unchanged,
`--expert` is `None` on every path that does not pass it, and the `TorchRuntime` load is
skipped only when it is `Some` (the expert loads no weights, the same trade `es loop collect
--expert` already makes). `--expert` without `--frames` is a usage error, because the expert
reads the cube's pose out of the state the frame source is handed and nothing else on this
path gives a policy privileged state.

**The one thing to review.** The episode boundary the expert's `reset` keys off is detected by
an **exact-zero-velocity heuristic**: `Env::reset` zeroes `qpos`/`qvel` before the Task IR's
`Randomization` node writes the cube's pose, and an episode's first tick always reaches the
frame source (the observation ring is empty there, so `observation_delay` cannot drop it), so
a state whose every velocity is exactly `0.0` is that tick and no other. The `ponytail:`
comment on `Seen::episodes` names the ceiling. A mid-episode state with every velocity exactly
zero would restart the expert's stage machine, which fails that episode loudly rather than
passing the gate quietly — the safe direction for a gate to be wrong in — but the runner
having no episode hook at all is the real gap. **This is an M7 review item**: either
`es_eval::Evaluation` grows an episode-boundary callback (the intervener hook `es loop
collect` already has), or the gate accepts the heuristic on the record.

### 12.7 The oracles

| # | Command | Needs |
|---|---|---|
| 1 | `cargo test -p es --test cli cycle_dry_run_plan_is_the_golden` | nothing |
| 2 | `cargo test -p es-data loop_train_and_evaluate_steps_chain` | nothing |
| 3 | `cargo test -p es --test cli cycle_refuses_a_moved_evaluation_hash` | nothing |
| 4 | `cargo test -p es --test cli cycle_runs_the_expert_through_the_harness_first -- --ignored` | `ES_PYTHON` (torch, mujoco), `--features render` |
| 5 | `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T2-loop-cycle.md` | nothing |

Oracle 4 on the oracle server (RTX 4090, `ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`,
`--features render`; 33.27 s):

```
RAN cycle_runs_the_expert_through_the_harness_first: expert gate success_rate 1, policy success_rate 0 at policy_hash b58893a9a48a9da2020a4414baa87db3517bc7f3fc7ac93130d31041510fd95f
```

The gate passes on the harness cut to two seeds; the 40-step policy does not, which is what a
40-step policy is. That asymmetry is the oracle's subject: the gate ran, it ran **first**, and
the ledger chained `collect -> evaluate -> train -> evaluate`.

### 12.8 Measured — one whole cycle, oracle server, RTX 4090, 2026-09-21

`es loop cycle` on the demo's own documents: 200 expert episodes with frames, the gate on the
committed `evaluation.toml`, the IR route at 20,000 steps (batch 8, lr 1e-4, seed 0,
`extra = ["--resident-gpu"]`, `device = "cuda"`), the same `evaluation.toml` at `--jobs 6`
with frames, and the `nominal-00` showcase with V19b's camera. One command, one document, one
ledger. `~/artifacts/plan-v/m7-t2/run.sh` is the invocation; `cycle.log` timestamps every line
of its stdout.

| Stage | Wall-clock | What ran |
|---|---|---|
| `collect` | **5:00** | 200 episodes, 103,881 frames, `success 200 / failure 0 / timeout 0`; safety 4,245 clamped, 1,991 fallback |
| `expert-gate` (`es eval run --expert`) | **2:21** | 6 suites × 16 episodes, `--jobs 6`, frames; 69,380 frames |
| `train` (T1's IR route, nested) | **2:26** | of which `dataset bake` 0:24, `policy lower` < 1 s, **`train_act.py` 2:00** (20,000 steps), `policy pack` ×3 ≈ 1 s |
| `eval` | **20:24** | 6 suites × 16 episodes, `--jobs 6`, frames; 171,116 frames |
| `showcase` | **0:08** | 1,800 ticks of `nominal-00` at 1280×720 lambert, 4.7 ms/frame |
| **total** | **30:19** | §28.9's stop rule is one cycle under 30 minutes: **missed by 19 seconds** |

`training_hash 6c81756c…`, `identity_hash 8914b1d6…`, `lowering_hash 3d06811c…`,
`final_loss 0.014186` from `initial_loss 0.052309`. The judged checkpoint is
`checkpoint.20000 = policy_hash ec8379a9…`, which is the `evaluate` step's `policy_hash` — the
chain property, on real data. `es_data::check_chain` passed: `ledger: …/loop.jsonl (chained)`.

**The gate passed; the policy did not.** Both are in the ledger, side by side:

| Suite | expert gate `success_rate` | policy `success_rate` | policy `envelope_violation_rate` |
|---|---|---|---|
| nominal | **1.000** | **0.000** | 0.949 |
| light_intensity | 1.000 | 0.000 | 0.945 |
| light_direction | 1.000 | 0.000 | 0.947 |
| observation_delay | 1.000 | 0.000 | 0.904 |
| torque_noise | 0.000 | 0.062 | 0.956 |
| backlash | 1.000 | 0.000 | 0.937 |

The acceptance criterion is `nominal success_rate >= 0.5`, so the cycle exits 1 with
`FAILED suite=Some("nominal") metric=success_rate observed=0`. **That is a measurement, not a
defect of this packet** — T2's subject is whether one document can run the cycle and chain it,
and it did. The policy's number is the one §28.10's U-measurement is for, and the
`envelope_violation_rate` of 0.95 against the expert's 0.04 says where to look: the trained
chunk is outside the Deployment IR's envelope on nineteen of every twenty ticks, so the Safety
Plane is what the arm is actually following. `final_loss 0.014186` and a policy that never
succeeds is the same disagreement section 10 flagged for row D — a fit that low is not yet a
policy that works. This run used no `base_model` (section 11) and T6's augmentation was not in
the tree, so it is the *floor* the U-measurement improves on, not a verdict on the IR route.

**Three things this measurement says about §28.9's "where the wall-clock goes".**

1. **Training is no longer the dominant term.** §28.9 records 10:53 for 20,000 steps at batch
   8 and calls the per-sample loop in the batch lowering the root cause. T3 removed it
   (`lowering_hash 3d06811c…`): the same 20,000 steps on eight times the data (200 episodes,
   not 50) now take **2:00**. Rung 9 of §28.9's ladder is done.
2. **Evaluation is, and its wall-clock is a function of how good the policy is.** The *same*
   harness — six suites, 96 episodes, `--jobs 6`, frames — took 2:21 on the expert and
   **20:24** on the policy, an 8.7× difference with no code between them. A successful episode
   terminates when the cube lands in the bin; a failing one runs all 1,800 ticks. §28.9's
   5:49 for this suite was measured on a policy that sometimes succeeds, so it is a
   *best*-case number, and the worst case is what a fresh cycle pays. Episode-level sharding
   (§28.9 rung 10 / T8) is now the only lever that matters on this path.
3. **Collect pays the same tax.** The first attempt at this run used
   `~/artifacts/plan-v/v15/untrained.esb`, whose `deployment_hash 3b2ad568…` is the pre-V18
   envelope (`acceleration_max 20`) rather than the committed `f2f9a510…` (80). Every one of
   the 200 episodes timed out (`success 0 / timeout 200`, 284,163 of 360,000 steps in
   fallback) and collection took **17 minutes instead of 5**. The run was discarded and
   re-run against `~/artifacts/plan-v/v18/untrained-L80.esb`, which carries all four committed
   documents. It is worth recording because **the expert gate is exactly the thing that would
   have caught it**: the gate was already running when the mismatch was spotted, and it would
   have refused to train on a harness the expert could not pass — which is the whole argument
   of §28.9 rule 1, arriving unprompted on its first real run.

Kept under `~/artifacts/plan-v/m7-t2/`: `loop.jsonl`, `training.lock`, `report-policy.json`,
`report-expert-gate.json`, `cycle.log`, `cycle.toml`, `training.toml`, `run.sh`, and the whole
`run/` tree (27 GB: frames, trajectories and the three checkpoint bundles).

## 13. The augmentation (packet M7/T6)

Spec 7.3 has had an `Augment` node family since M1, and until this packet **every path ignored
it**: the bake runs the Release plan, where a `training_only` node is an identity;
`train_act.py` never saw the Observation IR; `augmentation.json` said `{"kind": "none"}` for
every run that ever ran. This section is the boundary that makes the declared subgraph
actually happen — in training only, from one seed, recorded as identity — while the evaluation
path keeps producing the bytes it produced yesterday.

### 13.1 The boundary, in one picture

```
ImageInput -> Dequantize -> Normalize -> Pad(4) -|- Crop{Random 96x96} -> Augment{ColorJitter} -> rgb_overhead
                                                 |
                                         the chain boundary
```

Everything left of the boundary is the Observation IR as it has always been, run by
`CpuPlan` — one implementation, the same one inference runs (packet M5/V2b). Everything right
of it is *per-sample*, so it cannot be baked once:

* `es dataset bake --for-training` writes the port at the boundary — `[frames, 3, 104, 104]`
  for the demo, not `[frames, 3, 96, 96]` — and records the chain in `manifest.json`;
* `es train` writes the same chain, plus its seed, into spec 19.3's
  `training/augmentation.json`, and passes it to the trainer as `--augmentation`;
* `python/es/augment.py` applies it per sample and per optimizer step;
* **evaluation applies none of it** (INV-15). The Release plan lowers the pad and the crop to
  `Pad(4)` then the *centre* crop, whose composition on a symmetric pad is the un-augmented
  image bit for bit — `crates/es/tests/cli.rs::dataset_bake_for_training_writes_the_chain`
  compares the two bakes as whole files.

`docs/design/observation-lowering.md` section 3.1 is the compiler half: which nodes are on a
chain, what the Release plan does with them, and why the GPU path refuses `Pad` by name.

### 13.2 The document and the recipe

The recipe gains one optional field:

```toml
[run]
seed              = 0
augmentation_seed = 7    # optional; absent is `seed`
```

It is separate because it is separable — re-drawing the augmentation of an otherwise identical
run is a different run, and spec 19.3 gives it its own `seed.json` slot. Both slots are real
only when the policy's Observation IR declares a chain:

| slot | no chain | with a chain |
|---|---|---|
| `seed.json.augmentation` | `{"unset": true}` | the number |
| `augmentation.json` | `{"kind": "none"}` | `{"kind": "observation-ir", "observation_hash", "seed", "chains"}` |
| bake step | as before | `... --for-training <root>` |
| trainer step | as before | `... --augmentation <out>/training/augmentation.json` |

So a recipe written before this packet has the same nine pre-run slots, the same
`identity_hash` and the same rendered plan — `tests/golden/train/plan-ir.txt` is unmoved, and
`train_identity_moves_with_augmentation` asserts the unaugmented half explicitly.

One deliberate hole: `--dry-run` on the IR route does not open the bundle (that is what lets a
plan be printed on a machine that has neither bundle nor dataset, packet M7/T1 oracle 1), so a
dry run of an augmented recipe prints the plan *without* the two flags. The real run opens the
bundle first and `config.json` records the line that actually ran.

### 13.3 The RNG: addressed, not stepped

Every draw is a pure function of where it is used (spec 3.4 forbids a global RNG). The mixer is
Murmur3's `fmix32` — the ten lines of integer arithmetic `crates/es-render/src/rng.rs` already
uses for the path tracer — and the key is five coordinates:

| coordinate | type | where it comes from | what it separates |
|---|---|---|---|
| `augmentation_seed` | `u64`, folded as two `u32` | `[run] augmentation_seed`, else `[run] seed` | two runs of one recipe |
| `sample_index` | `u32` | the trainer's own global sample index (`order[cursor]`, not the position in the batch) | two samples in one batch |
| `step` | `u32` | the optimizer step | the same sample seen twice |
| `node_index` | `u32` | the Observation IR **node id** | two nodes, and two ports' chains |
| `draw` | `u32` | `0, 1, …` within one node | the crop's x from its y |

`key(seed, sample, step, node)` is five `mix32` rounds; `uniform(key, i)` is one more, keeping
the top 24 bits so the value is exact in `f32` and never reaches 1.0. **`torch.Generator` is
not used anywhere in the augmentation**, and that is the point: its stream is an implementation
detail of a torch version, so a run reproduced on another torch would silently see different
augmentation and `augmentation.json` would describe something that did not happen.

Two implementations exist and they agree bitwise at `f32`:
`crates/es-policy/tests/ir_training.rs` re-derives the whole chain in Rust with no interpreter
and compares against `tests/golden/train/augment_seed0.json`, which was generated once from
`python/es/augment.py` itself.

**Why bitwise is a reasonable thing to ask for here.** Every scalar is computed in `f64` and
rounded to `f32` *before* it touches a tensor, so each per-element operation is a single IEEE
`f32` op with both operands exactly representable — a kernel that widens an intermediate and
rounds once gives the same bits as one that does not. The two places that could have gone
wrong were measured, not assumed (the same discipline section 10 applied to `cos`):

* **Box-Muller's `ln` and `cos`**, evaluated by torch's vectorised `f64` kernels on one side
  and Rust's `std` on the other. A last-bit disagreement in `f64` is ~2⁻⁵³ relative, far below
  an `f32` ulp, so the cast collapses it — and the golden says it collapses for all 960 values.
* **The contrast mean**, the one reduction in the file: `x.double().mean()` in torch against a
  sequential `f64` sum in Rust. Same argument, same measurement. It is the one place where a
  *much* larger image could in principle drift — the ceiling is that the two summation orders
  differ by ~2⁻⁵³ relative, which needs the f32 rounding to land exactly on a tie to show — and
  if it ever does, the fix is to define the mean as an ordered `f32` sum in both.

### 13.4 The four kinds, and what each one is

| kind | status | what the trainer does |
|---|---|---|
| `RandomCrop { width, height }` | **implemented** | uniform integer offset in `[0, W-w] × [0, H-h]`, two draws; the Release plan's centre crop is the same rectangle with the offset fixed |
| `ColorJitter { brightness, contrast }` | **implemented** | `x * (1 + u·b)`, then `(x - mean) * (1 + u·c) + mean`, `u ∈ [-1, 1]`, two draws |
| `ColorJitter { saturation, hue }` | **refused by name** | both need a colour model this file does not have (`hue` is an HSV rotation). A non-zero value stops the run with the node named; a silently ignored parameter would be worse |
| `GaussianNoise { sigma }` | **implemented** | additive, Box-Muller over two uniforms **per element**, `f64`, cast to `f32` once |
| `RandomErasing { probability }` | **refused by name** | not implemented by this packet |

`GaussianNoise` is the expensive one: two draws per element means a `2 × C × H × W` index
tensor per sample per step. The demo's chain does not use it, so the measured run below pays
nothing for it; a document that does should expect the augmentation to become a visible
fraction of the step.

### 13.5 The document for the U-measurement

`tests/fixtures/visible-learning/observation-augmented.toml` is the committed
`observation.toml` plus three nodes on the image path, regenerated by
`cargo test -p es --test cli -- --ignored generate_augmented_observation_fixture`:

| document | `observation_hash` |
|---|---|
| `observation.toml` (committed) | `899c16a90033eeb406f328060bbd54ef0632f943a2059db7b3f7c369ee218d81` |
| `observation-augmented.toml` | `cc437a2418c36ac003c68258cc017a3783d9b2d876029463e87a9d38d194fb3e` |

The output port carries exactly the `PortType` the committed document's does — `3×96×96`,
`Normalized{0,1}`, the same `ImageSpec` — so `learning.toml` is untouched and `learning_hash`
does not move. **But `evaluation.toml` names `observation`**, so an Evaluation IR pointed at a
policy trained on this document is a *different* `evaluation_hash` (spec 13.3): the
U-measurement has to write the augmented hash into its evaluation document and accept the new
identity deliberately (`es loop cycle` refuses a moved `evaluation_hash` without
`--allow-new-evaluation`, section 12.3). Nothing about the *pixels* an evaluation sees changes
— that is what 13.1 guarantees — but the document that names them does.

### 13.6 Deviations from the packet

1. **`Crop { CropMode::Random }` is on the chain, and the demo document uses it instead of
   `Augment { RandomCrop }`.** Forced, and the finding is worth the review's time:
   `es-ir`'s `image_out` keeps the incoming `ImageSpec` on an `Augment` port, so
   `Pad(4) -> Augment{RandomCrop 96x96}` advertises `96×96` where propagation says `104×104`,
   and `es_ir::cross`'s `TYPE-020` refuses the bundle before it can be built. `es-ir` is
   forbidden ground for this packet. `CropMode::Random` — "a per-sample offset whose nominal
   geometry is the centred one", in the IR's own words — propagates through
   `ImageSpec::cropped` and validates, so it is what the fixture uses, and the
   `Augment{RandomCrop}` lowering is implemented and tested beside it for the day a document
   can carry it. **Open question for the M7 review:** should `image_out` give an
   `Augment { RandomCrop }` the cropped geometry the way `Crop { Random }` gets it?
2. **`crates/es-compile/src/exec.rs` was added to the packet's `## context` globs.** `Op::Pad`
   has to be executed where `CpuPlan::run`'s match is; the alternative, a `kernels.rs` entry,
   appends to `KERNEL_IDS` and moves `compiler_hash` for every plan in the repository — a
   golden move this packet forbids itself. The consequence is that `Pad` has no kernel id and
   the GPU path refuses it by name (observation-lowering.md 3.1).
3. **Oracle 3 is two tests, not one.** `es-eval` owns the bake and `es` owns the manifest —
   `es-eval` is layer 10 and writes no files — so
   `es-eval::bake_for_training_writes_the_boundary` asserts the boundary shapes and the
   byte-identical evaluation frame, and `es::dataset_bake_for_training_writes_the_chain`
   asserts `manifest.json` and compares the two bakes as whole safetensors files.
4. **The trainer's `augmentation.json` writer lives in `es-data` and its reader in
   `augment.py`.** `crates/es-policy/tests/ir_training.rs` writes its own copy of the file
   (layer 8 cannot depend on layer 10), so the *shape* of the real one is pinned in
   `crates/es/tests/cli.rs` instead. Both ends are asserted; they are just not asserted in one
   place.

### 13.7 The oracles

| # | command | what it judges |
|---|---|---|
| 1 | `cargo test -p es-policy --test ir_training augmentation_matches_the_golden` | the Rust re-implementation equals `tests/golden/train/augment_seed0.json` bitwise at `f32`; with `ES_PYTHON`, `augment.py` equals it too |
| 2 | `cargo test -p es-compile a_training_only_random_crop_is_a_centre_crop_in_release` | the padded, centre-cropped Release plan reproduces the un-augmented pixels, and the crop carries `Crop`'s intrinsics transform |
| 3 | `cargo test -p es-eval bake_for_training_writes_the_boundary` / `cargo test -p es --test cli dataset_bake_for_training_writes_the_chain` | `104×104` at the boundary and the chain in the manifest; `96×96` and the un-augmented document's bytes without the flag |
| 4 | `cargo test -p es --test cli train_identity_moves_with_augmentation` | both identity slots are real, both move, and the unaugmented recipe's are where they were |
| 5 | `cargo test -p es-policy --test ir_training -- --ignored augmented_training_runs` | 40 steps on the augmented document; finite loss; two runs at one seed are byte-identical curves, two seeds are not |
| 6 | `cargo xtask ci`, `cargo xtask check-scope docs/packets/M7/T6-augmentation.md` | goldens 0 changed, scope clean |

The golden is generated by `generate_augment_golden`, which is `#[ignore]`d **and** refuses to
run unless `ES_GENERATE_GOLDENS=1` is set — `cargo test -- --include-ignored` runs every
ignored test in the workspace, and an M7 review found exactly that rewriting a golden nobody
meant to touch.

### 13.8 Measured — oracle server, RTX 4090, 2026-09-21

The augmented document's 20,000-step run at section 10's row-D settings, through `es train`:
batch 64, lr 4e-4, `warmup_cosine` warmup 250 `lr_min` 1e-6, seed 0, `--resident-gpu`,
`device = "cuda"`, torch 2.11.0+cu129, on V15's 200 demonstrations
(`~/artifacts/plan-v/v15/ds-train`, 36,960 samples). The GPU was idle before it (56 MiB, 0 %).
One command; the bake is a step of it, and it is the `--for-training` bake.

| step | wall-clock | what ran |
|---|---|---|
| `dataset bake --for-training` | **0:10** | 200 episodes, 36,960 frames, `rgb_overhead [3, 104, 104]`, 4.5 GB |
| `policy lower` | < 1 s | `lowering_hash 3d06811c…`, T3's and T4's — the Learning IR did not move |
| `train_act.py` | **4:21** (261 s) | 20,000 steps, 1,280,000 samples seen, resident copy 4,577.6 MiB |
| `policy pack` | ~1 s | `checkpoints/20000.esb` |
| **total** | **4:33** | |

`identity_hash 2d8f6837…`, `training_hash 61e6c93a…`, checkpoint
`policy_hash df985459…` (spec 19.3's `H(training_hash, checkpoint)`),
`observation_hash cc437a24…`, `lr_curve_hash c01d5185…` — the same schedule as row D, which is
what makes the two rows comparable. `initial_loss 0.049270`, **`final_loss 0.006664`**, no
non-finite step. **Not evaluated here**: that is the U-measurement's, and it needs a new
`evaluation_hash` (13.5).

**Against row D, which is the same run without augmentation:**

| | row D (section 10) | this run | |
|---|---|---|---|
| trainer wall-clock | 2:51 (171 s) | **4:21 (261 s)** | +53 % |
| s / 1,000 steps | 8.6 | **13.0** | |
| samples/s | 7,485 | **4,904** | |
| `final_loss` | 0.004610 | **0.006664** | +45 % |
| resident baked set | 3.9 GiB | **4.5 GiB** | `104²/96²` = 1.17× |

Two numbers, and both are expected rather than disappointing.

**The 90 seconds.** 4.5 ms per step, for a batch of 64 through a two-node chain — 64 Python
slice-assignments for the crop and 64 × (a multiply, an `f64` mean and a fused affine) for the
jitter. That is the cost of a per-sample Python loop, not of the arithmetic; the obvious
lever, if it ever matters, is drawing the whole batch's offsets in one indexed gather rather
than in a loop. It does not matter yet: 4:21 is well inside §28.9's budget and the evaluation
step of a cycle is twenty minutes.

**The higher loss is the point.** Augmentation makes the training distribution wider, so a fit
on it *should* be worse at equal steps — the question augmentation exists to answer is what
happens on the **evaluation** suites, where the un-augmented row D scored `final_loss 0.004610`
and (in section 12.8, at batch 8) a `nominal success_rate` of 0. A lower training loss with a
policy that never succeeds is exactly the disagreement §28.10's U-measurement is for, and this
run is the arm of it that has seen more than one crop of each frame.

Kept under `~/artifacts/plan-v/m7-t6/`: `untrained-augmented.esb` (the four committed
documents with `observation-augmented.toml` in place of `observation.toml`), `training.toml`,
`train.log`, and `run/` (4.5 GB baked set, `module/`, `metrics/loss.json`, `training/`'s twelve
slots, `training.lock`, `weights/model-20000.safetensors`, `checkpoints/20000.esb`).

---

## 14. Starting from a policy (packet M8/S1)

Every fine-tune, and every RL continuation after it, starts from weights that already exist.
Until this packet a recipe could say where its *backbone* came from (section 11) and nothing
about where its **policy** came from: the only way to continue a run was to hand
`train_act.py` a checkpoint by hand, which leaves no record anywhere that it happened. This
section is the table that fixes that, and the lock that makes it provenance rather than a
convention.

### 14.1 The recipe gains one table

```toml
[init]
policy = "runs/train-001/checkpoints/20000.esb"
```

A bundle, not a bare safetensors, and that is the whole design: a `.esb` carries its own Task,
Observation and Learning IR and its own §5.3 `policy_hash`, so `init.lock` can record *which
policy* this run continued and not merely which 60 MB of floats it read. `[init]` is
`deny_unknown_fields` like every other table, and it is the IR route's — on the `lerobot`
route it is refused by name, because `lerobot-train` starts from its own `--policy.path` and a
recipe naming both would write a lock describing tensors the run never loaded.

`tests/fixtures/visible-learning/training-init.toml` is the committed example and
`tests/golden/train/plan-init.txt` its plan. Both are **additions**: `training.toml`,
`training-lerobot.toml` and their two goldens are untouched, and
`crates/es-data/tests/training_init.rs` pins the first one's digests —
`identity_hash a7531245…`, `training_hash 577a3f4a…` — measured before this packet changed a
line. A slot that is absent has to leave every recipe written before it exactly where it was.

### 14.2 Three buckets, and the fourth that does not exist

After the bundle is opened, its safetensors header is compared to the lowered module's
contract — `weight_keys` and `weight_shapes`, the same two lists `es policy lower` writes into
`contract.json`:

| bucket | condition | what happens |
|---|---|---|
| **copied** | the module declares this name, at this shape | the tensor is written into `<out>/weights/init.safetensors` |
| **initialised** | the module declares this name and the bundle has no tensor for it | the lowering's own draw stands |
| **shape_mismatch** | both know the name, at two shapes | recorded, **not** copied |

A tensor the module does not declare at all is in none of the three: there is no slot to put it
in, and a fourth list for "we ignored this" would be a list nobody acts on.

**A shape that disagrees is never reshaped.** Reshaping a trained tensor is guessing, and the
failure mode of a guess here is the worst one this repository has: a policy that still trains,
still converges and is quietly wrong. It goes into the lock instead, so a reader sees it and
fixes the document.

**A `nodes.<k>.*` prefix claim is copied by name alone.** Those keys belong to torchvision or
`torch.nn` — `es_policy::weights::validate_keys` has always treated them that way — so there is
no declared shape on the Rust side to check them against. `train_act.py --init-weights` checks
every key against the real module before loading it, which is where the shape actually lives; a
key that is not a member, or is a member at another shape, is a refusal there rather than a
`strict=False` load that silently keeps a random tensor.

The copied tensors are written to a file rather than the bundle being handed over whole,
because what the trainer loads has to be exactly what the lock says was copied. A file holding
the intersection cannot disagree with the list; a whole checkpoint plus a list can.

### 14.3 `training/init.lock`

```json
{"schema_version":1,
 "source":"runs/train-001/checkpoints/20000.esb",
 "policy_hash":"c109d783…","learning_hash":"fdb5178a…",
 "copied":["nodes.1.0.bias","nodes.1.0.weight", "…"],
 "initialised":["nodes.0.*"],
 "shape_mismatch":[{"name":"nodes.1.2.weight","expected":[512,256],"found":[512,128]}]}
```

`source` is the path **as the recipe wrote it**, and `policy_hash` / `learning_hash` come out
of the bundle's own manifest — so the lock identifies the policy, not the file path, and a
bundle moved to another directory is still the same base. Both name lists are sorted: the
lock's digest is a hash slot, and a digest that depends on iteration order is not one.

**It enters `identity_hash` and `training_hash`, and it is written before the run.** That is
forced rather than chosen: what a run starts from is known before it starts, and §19.3's split
puts everything known before the trainer into `identity_hash`. `es train` therefore does the
whole comparison among the refusals that need the documents — before the interpreter is
probed, before the bake, before a single GPU-second — even though the packet phrases it as
"after `es policy lower`". The module it compares against is the same module: `lower_to_torch`
is a pure function of the Learning IR, so the `weight_keys` and `weight_shapes` it yields here
are the ones the lower step writes into `contract.json` seconds later.

**Deviation: the digest lives in `config.json`.** §19.3 names twelve files and
`es_data::identity::TrainingIdentity` has twelve fields; a thirteenth field is a change to
`es-data`'s identity module, which this packet does not own. So `Training::set_init` writes
`training/init.lock` as a real file *and* puts its blake3 into `config.json` under `"init"` —
exactly the relationship `TrainingIdentity.base_model.hash` already has with `base_model.lock`,
which is the digest of the file and not its contents. Everything downstream follows for free:
`identity_hash`, `training_hash`, and §19.3's `policy_hash = H(training_hash, checkpoint_hash)`
per checkpoint. A recipe without `[init]` never calls it, so its `config.json` is byte-for-byte
the one it always was — which is what the pinned digests above assert. If a later packet moves
`TrainingIdentity` to thirteen fields, this is the one line that changes.

`training.lock`'s `files` map gains an `init.lock` entry when there is one and does not when
there is not, so the lock can always be recomputed from the files on disk.

### 14.4 `[run] steps = 0` — checkpoint immediately

The marks of a run are `checkpoint_at` plus `steps`, and `steps` is always the last of them
(section 2). `steps = 0` is the one run whose single mark is **0**: the module's initial state,
written without an optimizer step. `train_act.py` writes mark 0 before the loop rather than
inside it — the loop's `(step + 1) in marks` can never produce 0 — and the shape probe above it
runs under `model.eval()` and `no_grad`, so nothing, not even a BatchNorm running mean, moves
between `--init-weights` filling the module and `checkpoint_tensors` reading it back.

That is what makes a zero-step run from a full-coverage bundle a **bitwise re-pack**, which is
this packet's oracle. It is also a real thing to want: re-packing a checkpoint against a
document that has moved on is then one command instead of a pipeline. `checkpoint_at` beside it
is refused — a run that takes no optimizer step has no step to also stop at — and so is
`steps = 0` on the `lerobot` route, where `lerobot-train` saves at a single `--save_freq` and
has no step 0 to save at.

### 14.5 The refusals, each by name

Named and not numbered, for the reason section 11 gives: neither `es_data::training` nor
`es train` has ever carried numeric codes, and a numbering scheme invented for a handful of
messages is a scheme with one user. The packet writes `TRN-…`; the file's series is names.

| refusal | the message names |
|---|---|
| `[init]` on the `lerobot` route | the table, the route, and `--policy.path` as where lerobot's own base is reached |
| a bundle sharing no tensor with the module | how many names are absent and how many disagree about a shape, and that this would be `steps` steps from scratch under a document saying otherwise |
| `[run] steps = 0` with `checkpoint_at` | the marks, and that the one mark of such a run is 0 |
| `steps = 0` on the `lerobot` route | `--save_freq`, and that "checkpoint immediately" is the IR route's |
| a non-`F32` tensor in the init bundle | the tensor, its dtype and §8.4 |
| (in the trainer) a key that is not a member of the module, or is at another shape | the key, and that the file holds exactly what `init.lock` lists as copied |

### 14.6 The oracles

| # | command | needs |
|---|---|---|
| 1 | `cargo test -p es-data --test training_init` | nothing — the pin, the slot, and the parse refusals |
| 2 | `cargo test -p es --test cli train_init_partial_and_refused` | nothing |
| 3 | `cargo test -p es --test cli train_init_from_bundle_zero_steps -- --ignored` | `ES_PYTHON` with torch; prints `SKIP` and stops without it |
| 4 | `cargo test -p es --test cli train_dry_run_plan_is_the_golden` | nothing — now three recipes, three goldens |

Oracle 2 needs no Python for the reason section 11's refusal oracle does not: the comparison is
a property of two documents, and `es train` does it before the interpreter is probed. Its
partial case starts from a bundle built on `learning-pretrained.toml`'s graph with one
`StateEncoder`'s hidden width moved from 256 to 128, whose own checkpoint carries the twelve
exact keys and neither prefix claim — so all three buckets are non-empty at once, which a full
copy never exercises. Its "shares nothing" case shifts every node id by ten, which is what a
genuinely differently-laid-out graph produces.

### 14.7 Measured — oracle server, RTX 4090, 2026-09-21

**U3's 20,000-step checkpoint, re-packed from zero steps: bitwise.** The recipe is
`training-u3.toml` with `steps = 0`, `[init] policy` pointing at U3's own
`checkpoints/20000.esb`, and no `schedule` (a warmup of 250 over a run of 0 is refused by
name). Same bundle, same dataset, same verified backbone, so the module the trainer builds is
the module U3 trained. `ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`, torch 2.11.0+cu129,
`device = "cuda"`.

| step | wall-clock | what ran |
|---|---|---|
| the init comparison | < 1 s | 126 copied, 0 initialised, 0 shape mismatches |
| `dataset bake --for-training` | ~8 s | 200 episodes, 36,960 frames, `rgb_overhead [3, 104, 104]` |
| `policy lower` | < 1 s | `lowering_hash 41d11a06…`, 14 keys (12 exact, 2 prefix claims) |
| `train_act.py` | ~21 s | 0 optimizer steps; `--init-backbone` then `--init-weights`, then mark 0 |
| `policy pack` | ~1 s | `checkpoints/0.esb` |
| **total** | **~0:40** | |

`weights/model-0.safetensors` is **byte-identical** to U3's `weights/model-20000.safetensors`
(`sha256 2e0b2f05…`, `cmp` reports no difference), and so is the packed bundle's checkpoint:
both `checkpoint.manifest`s carry `weights_blake3 9375650a…`.

**The two `policy_hash`es, and why they differ.** This is the chain the packet asks for:

| | U3 (20,000 steps) | S1 (0 steps, from U3) |
|---|---|---|
| §19.3 `identity_hash` | `624674252273bb34…` | `4baf0a0ee41c8b36…` |
| §19.3 `training_hash` | `a1bda64afbfd3e1d…` | `1c1c4c4d9a0c57cc…` |
| §19.3 `policy_hash` = `H(training_hash, checkpoint)` | `30340aec73c6d1b5…` | `ba5d68788b1ca0e5…` |
| checkpoint blake3 | `9375650a9e4e3752…` | **`9375650a9e4e3752…`** |
| bundle's §5.3 `policy_hash` | `c109d783bbfd6c5b…` | **`c109d783bbfd6c5b…`** |

The bottom two rows are equal and the top three are not, and that is exactly right. §5.3's
`policy_hash` names *the policy* — the four IR documents and these weights — and the artifact
S1 produced is the same policy, bit for bit. §19.3's names *the run that produced it*, and
these are two different runs: one spent 20,000 optimizer steps and the other spent none, one
has no `init.lock` and the other's is `2bbecb1f…`. A scheme where they came out equal would be
one that cannot tell training from copying.

`training/init.lock` records `source` as the U3 path, `policy_hash c109d783…` and
`learning_hash fdb5178a…` from U3's own manifest, 126 copied names, and two empty lists. The
trainer's summary agrees from its side: `"init_weights"` names the file and
`"initialised_from"` lists the same 126 keys it loaded.

Kept under `~/artifacts/plan-s/s1/`: `training-s1.toml`, `train.log`, and `run/` (the baked
set, `module/`, `training/`'s twelve slots plus `init.lock`, `training.lock`,
`weights/init.safetensors`, `weights/model-0.safetensors`, `checkpoints/0.esb`).
