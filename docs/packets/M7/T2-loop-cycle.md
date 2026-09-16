# M7 T2 — `es loop cycle`: collect → train → eval (→ showcase) under one document and one ledger

Spec: §13.1 (the loop is a first-class workflow; each stage a command and an artifact), §13.3
(every iteration linked by the hash chain; "keeping the evaluation conditions fixed while
changing only data·policy is the discipline" — a changed `evaluation_hash` invalidates the
comparison and the tool warns), §19.2/§19.3, §10.5, §28.9 rule 1 (the harness passes the
expert before anything trains), §28.10 (T2). Design note to extend: `docs/design/training-recipe.md`
(+ `.ko.md`) — a new section "the cycle", and `docs/design/learning-loop.md` section 4 (the
ledger gains `train` and `evaluate` steps). Depends on **T1** (`es train`, the recipe) and on
`es loop collect`, `es eval run`, `es video showcase` as they are.

## the question

T1 made training one command. The cycle §13.1 draws — collect, train, evaluate, watch — is
still four commands whose paths and hashes a person threads by hand, and `loop.jsonl` stops
at `distill`: it never records that a dataset trained a policy or that a policy was judged.
**Can one document run the whole cycle, append every stage to the ledger with the hashes
§13.3 wants, and refuse to compare across a moved `evaluation_hash`?**

## spec

`es loop cycle --recipe cycle.toml [--out <dir>] [--dry-run] [--from collect|train|eval|showcase]`.

```toml
kind = "cycle"
scene = "tests/fixtures/mjcf/so101_pick_place.xml"

[collect]                        # optional; absent = start from an existing dataset
policy   = "untrained.esb"       # the bundle whose Deployment IR is the plane (es loop collect --policy)
expert   = "so101-pick-place"    # or omit for a trained policy's own rollouts
episodes = 200
seed     = 1
frames   = true                  # es loop collect --frames <out>/collect/frames
# dataset = "runs/collect-001/ds"    # instead of [collect]: reuse

[train]                          # T1's recipe, inline or by path
recipe = "training.toml"         # its [dataset] root/frames are overridden by this cycle's collect output

[eval]
config    = "tests/fixtures/visible-learning/evaluation.toml"
checkpoint = "last"              # or a step from the recipe's checkpoint_at
jobs      = 6
frames    = true

[showcase]                       # optional
cell   = "nominal-00"
eye    = [0.66, -0.46, 0.52]
look_at = [0.14, -0.04, 0.04]
fov    = 36
width  = 1280
height = 720
```

* **Stages call the existing functions in-process** (`cmd::r#loop::collect`, `cmd::train::run`,
  `cmd::eval::run`, `cmd::showcase::run`) — nothing is re-implemented, no second CLI parser;
  only what those already spawn (the trainer, the physics subprocess, `--jobs` workers) is a
  subprocess. Outputs go under `<out>/{collect,train,eval,showcase}/`.
* **`loop.jsonl` gains two step kinds** (`es_data::collect::LoopKind`): `Train { inputs:
  dataset content/schema/split, recipe identity_hash; outputs: training_hash, checkpoints
  (step → policy_hash) }` and `Evaluate { inputs: policy_hash, evaluation_hash, deployment,
  observation; outputs: report blake3, passed, success_rate of the acceptance's suite }`. Both
  are appended to the dataset root's ledger (the thing that trained / was judged) and to
  `<out>/loop.jsonl`. The chain property extends: `train.inputs.content == collect.outputs.content`,
  `evaluate.inputs.policy_hash ∈ train.outputs.checkpoints`.
* **The §13.3 warning becomes a refusal by name.** `<out>/loop.jsonl` may hold earlier cycles
  (`--out` reused for iteration 2). If the new `evaluate` step's `evaluation_hash` differs from
  the previous `evaluate` step's in the same ledger, `es loop cycle` refuses unless
  `--allow-new-evaluation` is passed — that is §13.3's "if `evaluation_hash` changes, the
  comparison is invalid" made executable. `es eval compare` on the two reports is printed at
  the end of iteration ≥ 2.
* **Harness-first (§28.9 rule 1, M5-R1).** When `[collect].expert` is set, the cycle runs the
  expert through `es eval run` on the *same* `[eval].config` **before** training and refuses to
  train if the expert's acceptance fails — the collection and the evaluation must share one
  harness, and a harness the expert fails is a harness no policy can pass. `--skip-expert-gate`
  exists and is recorded in the ledger as a deviation.
* **`--dry-run`** prints the stage plan (one line per command, paths relative to `<out>`) and the
  T1 plan nested under `train`; byte-identical golden.
* **`--from <stage>`** resumes an interrupted cycle from a stage, reading the earlier stages'
  outputs under `<out>` and refusing if they are missing or their hashes disagree with the
  ledger.

## context

`cargo xtask check-scope` reads the fence; the prose below is the same list with reasons.

```
crates/es-data/src/collect.rs
crates/es-data/src/training.rs
crates/es-data/src/lib.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/cycle.rs
crates/es/src/cmd/mod.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/showcase.rs
crates/es/src/main.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/cycle.toml
tests/golden/train/plan-cycle.txt
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/design/learning-loop.md
docs/design/learning-loop.ko.md
docs/packets/M7/T2-loop-cycle.md
docs/packets/M7/T2-loop-cycle.ko.md
```

`collect.rs` (the two `LoopKind`s and their append), `training.rs` (the `Cycle` recipe schema
and plan, headless), `cycle.rs` (new, thin), `loop.rs` / `mod.rs` / `main.rs` (dispatch + help),
`train.rs` / `eval.rs` / `showcase.rs` **only** to expose their entry as `pub(crate)` with a
struct of options instead of an argv slice if one is needed, `cli.rs` (new `cycle_*` tests),
the fixture recipe and its plan golden (sanctioned generator), the two design notes, this packet.

## oracle

1. `cargo test -p es --test cli cycle_dry_run_plan_is_the_golden` — the fixture cycle's
   `--dry-run` output is byte-identical to `tests/golden/train/plan-cycle.txt`. No Python.
2. `cargo test -p es-data loop_train_and_evaluate_steps_chain` — a synthetic ledger with
   `collect → train → evaluate` passes the chain check; a `train` whose input `content` is not
   the `collect`'s output fails it by name.
3. `cargo test -p es --test cli cycle_refuses_a_moved_evaluation_hash` — a ledger holding an
   `evaluate` step with hash A and a recipe whose `[eval].config` hashes to B is refused with
   both hashes in the message; `--allow-new-evaluation` proceeds (dry-run level).
4. `cargo test -p es --test cli cycle_runs_the_expert_through_the_harness_first -- --ignored`
   — with `ES_PYTHON` (mujoco): a 2-episode expert collect + a 40-step IR-path train + a
   2-seed eval on the demo fixtures; `loop.jsonl` holds `collect`, `evaluate` (expert gate),
   `train`, `evaluate` in that order with chained hashes; `report.json` exists. `SKIP` with
   reason without `ES_PYTHON`.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T2-loop-cycle.md`.

## acceptance

Oracles 1–5 (4 on the oracle server). One real cycle on the server, `[collect] episodes = 200`
with the expert, `[train]` = the IR route at 20,000 steps (T3's batched module if it has
landed, else the current one), `[eval]` = the committed `evaluation.toml` with `--jobs 6`,
`[showcase]` of `nominal-00`: the wall-clock of each stage and the total, recorded in the design
note beside §28.9's "where the wall-clock goes" table (target: one cycle under 30 minutes —
§28.9's stop rule for speed work; `Target / Status: unverified` until measured). The expert
gate's own report is kept beside the policy's.

## forbidden

Re-implementing any stage; a second recipe parser; changing what `collect`, `train`, `eval` or
`showcase` compute; `crates/es-safety/**`; `crates/es-ir/**`; every fixture and golden other
than the new ones; `docs/ARCHITECTURE*.md`. INV-16: safetensors only. INV-17: no new trait.
