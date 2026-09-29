# M12 R8 — the training stage fetches a pinned backbone it does not have

Found by P-M12-R7 (commit `e536c72`): the hint card's recipe, `training-hint-u3.toml`, names
`target/backbone/resnet18-imagenet1k-v1.safetensors`, which a fresh checkout lacks. Nothing
fetched it, and `needs` could not say it (a name `es --check-deps` does not report counts as
missing and disables the card), so the cycle collected for 26 minutes and passed the expert gate
before the train stage stopped at `Backbone::verify`. A beginner must not hit that.

## spec

- `[policy] base_model_fetch = "resnet18"` — optional, the smallest shape: `fetch_backbone.py`'s
  `--arch` and nothing more. `--out` is `base_model`'s directory and `--expect` is
  `RESNET18_IMAGENET1K_V1_BLAKE3`, so the recipe holds no second copy of the pin. Refused by
  name: with no `base_model`, for an arch other than `resnet18` (one pin, one arch), and when
  `base_model`'s file name is not the `<arch>-imagenet1k-v1.safetensors` the script writes
  (`Recipe::fetch_args`, called by `Recipe::parse`, `Cycle::training` and `Plan::build`).
- `Plan.fetch`: the whole command line — the run's interpreter (`ES_PYTHON` first),
  `python/es/fetch_backbone.py`, the flags. Rendered as a `# fetch:` comment line (it runs only
  when the file is missing): after `# route:` in `es train`'s plan, above every stage in a
  cycle's (hoisted out of the nested training plan, printed once). Absent, every plan, golden
  and `identity_hash` is unchanged; present, the field is part of the recipe in `config.json`
  like every other field.
- `es train`: when `base_model` is missing, runs the fetch before the backbone check, then
  `Backbone::verify` judges the file as before. `es loop cycle`: before the first stage (unless
  `--from eval|showcase`), runs the same fetch when the file is missing and then
  `Backbone::verify`s any named `base_model` — a failed fetch or a wrong file stops the cycle
  before collect. A failed fetch is one plain sentence naming the file, the exit code, the
  command, and what the interpreter needs.
- No new telemetry stage, so the editor is unchanged. The fixture and the template headers say
  what now happens instead of "`needs` cannot express it".

## context

```
crates/es-data/src/training.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/cycle.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/training-hint-u3.toml
templates/cube-into-bin-hint.toml
tests/golden/train/plan-cycle-hint-u3.txt
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M12/P-M12-R8-backbone-fetch.md
docs/packets/M12/P-M12-R8-backbone-fetch.ko.md
```

## oracle

1. `cargo test -p es-data training::tests::base_model_fetch_is_one_line_above_collect`: the
   field is one `# fetch:` line and nothing else in the plan moves; a cycle prints it once,
   above collect; the three refusals.
2. `cargo test -p es --test cli cycle_fetches_the_backbone_before_collect`: with a stand-in
   interpreter (a `.cmd` / `sh` script, no Python) that copies a self-consistent but unpinned
   artifact into `--out`, the cycle fetches and then refuses by the pin with no stage started;
   a present file is not re-fetched; a failing stand-in gives the plain message; `es train`
   fetches before its backbone check.
3. `plan-cycle-hint-u3.txt` regenerated through `generate_cycle_hint_u3_golden`: one line
   added, the `# fetch:` line under the `# cycle:` header. Every other plan golden unchanged.
4. fmt, `cargo clippy -p es -p es-data --all-targets -- -D warnings`, `cargo test -p es-data`,
   `cargo test -p es --test cli -- cycle_ train_`, `cargo xtask verify-goldens`,
   `cargo xtask check-spec-refs`, `cargo xtask check-scope <this file> --base main`.

## forbidden

Moving the pin or `fetch_backbone.py`'s behaviour; a new telemetry stage; any other recipe,
plan golden or `identity_hash`; a training or evaluation run; connecting to any remote server.
