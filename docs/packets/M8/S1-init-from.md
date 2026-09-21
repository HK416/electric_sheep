# M8 S1 — `es train` starts from a policy: `[init] policy`, and `init.lock` says what was copied

Spec: §13.4 ("'start from this policy' is written in a lock"), §19.3 (`base_model.lock`'s shape:
provenance in `training/`), §13.3 (`policy_hash = H(base = previous checkpoint, training.lock)`),
§28.10 rule 2 (slots are real or `unset`), §28.11 wave 2. Precedent: packet M7/T5's
`base_model.lock` (`crates/es-data/src/training.rs` `Backbone`, `es/src/cmd/train.rs`
`--init-backbone`, `train_act.py init_backbone`). Design note to extend:
`docs/design/training-recipe.md` (+ `.ko.md`), new section "Starting from a policy".

## the question

Continuation (S4c) and every fine-tune after it start from weights that already exist. **Can a
recipe name a bundle to start from, copy exactly the tensors whose name and shape match the
lowered module, initialise the rest, and record all of it in a lock that enters
`training_hash` — such that zero steps from a checkpoint reproduce that checkpoint bitwise?**

## spec

* `training.toml` gains an optional table `[init] policy = "<bundle.esb>"` (`deny_unknown_fields`).
  IR route only; on the external (`lerobot`) route it is refused by name (`lerobot-train` has its
  own `--policy.path`, and mixing the two would fabricate provenance).
* After `es policy lower` and before the trainer: open the init bundle, read its `safetensors`
  (INV-16 — never anything else) and compare every tensor to `contract.json`'s `weight_keys` /
  `weight_shapes`: **copied** = name and shape match; **initialised** = in the contract, not in
  the bundle; **shape_mismatch** = name matches, shape does not (recorded, not copied). Zero
  copied tensors is a named refusal (`TRN-…`, follow the file's series): starting from a policy
  that shares nothing is a mistake, not a warm start.
* The copied tensors reach the trainer as `--init-weights <safetensors>` (a new `train_act.py`
  flag beside `--init-backbone`; loads exactly the listed names, refuses any other key).
* `training/init.lock`: `{ source: <path as written>, policy_hash, learning_hash, copied: [...],
  initialised: [...], shape_mismatch: [{name, expected, found}] }`, sorted names. It is a §19.3
  slot: enters `training_hash`; absent when `[init]` is absent so every existing recipe's
  `identity_hash` / `training_hash` is unmoved (pin the fixture recipe's before touching anything).
* `[run] steps = 0` is legal and means "checkpoint immediately": the trainer writes `0.esb`'s
  weights without an optimizer step. This is the oracle's lever and a real use (re-packing).
* `--dry-run` prints the init step; the plan golden for the new fixture recipe is an addition.

## context

```
crates/es-data/src/training.rs
crates/es-data/tests/**
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
python/es/train_act.py
tests/fixtures/visible-learning/training-init.toml
tests/golden/train/**
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M8/S1-init-from.md
docs/packets/M8/S1-init-from.ko.md
```

`training.rs` (the table, the lock, the hash slot, the refusals), `train.rs` (open bundle,
compare, write lock, pass the flag), `train_act.py` (`--init-weights`, `steps = 0`), the new
fixture recipe, goldens (additions only), the note, this packet.

## oracle

1. `cargo test -p es-data init_lock_enters_training_hash_and_is_absent_without_init` — the
   fixture recipe's identity/training hashes pinned and unmoved; a recipe with `[init]` differs.
2. `cargo test -p es --test cli train_init_from_bundle_zero_steps` (`ES_PYTHON`; SKIP with reason):
   T1's 40-step run produces bundle `A`; a recipe `[init] policy = A`, `steps = 0` produces `0.esb`
   whose every tensor is bitwise `A`'s; `init.lock` lists all names as `copied`, none initialised.
3. `cargo test -p es --test cli train_init_partial_and_refused` — init from a bundle whose Learning
   IR differs (`learning-pretrained.toml`'s graph) copies the intersection, lists the rest as
   `initialised`, runs; init from a bundle sharing no tensor is refused by name; `[init]` on the
   lerobot route is refused by name.
4. Server: `[init] policy = <U3's 20000.esb>` (`~/artifacts/plan-v/m7-u/U3/train/checkpoints/`),
   `steps = 0` → bitwise U3; `policy_hash` chain recorded in the note.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M8/S1-init-from.md`.

## acceptance

Oracles 1–5. The note's new section with the lock's schema and the three refusals by name.

## forbidden

Pickle or any loader but `safetensors` (INV-16); copying a tensor whose shape differs
(reshaping is guessing); moving any existing recipe's hash; the external route; a new trait.
`docs/ARCHITECTURE*.md`; `tests/golden/**` modifications (additions only).
