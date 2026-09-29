# M12 R7 — the hint template trains the combination that passed (U3)

Owner decision, 2026-09-29: the practice card should give a beginner a policy that can succeed.
Its recipe today is `cycle.toml` → `training.toml` on `learning.toml` + `observation.toml`, never
measured as is; U0 (same documents) scored 0.25, and the first real run here (short preset)
scored nominal 0/16 with the plane changing 99.6 % of steps (`docs/packets/M12/YV-verification.md`).
M7/U3 — `learning-pretrained.toml` (ImageNet ResNet18 backbone) + `observation-augmented.toml`,
`training-u3.toml`'s run settings — passed the demo's acceptance (held-out 0.5625,
`docs/reviews/M7.md`, `docs/design/visible-learning.md` section 7.31).

## spec

- A new committed cycle recipe (e.g. `tests/fixtures/visible-learning/cycle-hint-u3.toml`) whose
  `[train]` is U3's run settings inline or by a new committed training recipe with repository-
  relative paths (the committed `training-u3.toml` holds server paths and stays as it is), and
  whose `[eval] config` is an Evaluation IR the U3 bundle passes `XIR-040` against — find how U3
  was evaluated (the augmentation nodes are training-only; does the evaluated bundle carry
  `observation-augmented.toml` or `observation.toml`?) and follow exactly that. If a new committed
  Evaluation IR is needed, it changes **no** suite, seed, metric or threshold of `evaluation.toml`.
- The pretrained backbone: how `base_model` is obtained on a fresh machine (`python/es/fetch_backbone.py`
  or the documented step) — the editor's run must work from a fresh checkout on this PC; if a
  download is needed, the recipe/CLI path does it or the template's `needs` says what is missing.
- `templates/cube-into-bin-hint.toml` points at the new cycle; `[bundle]` names the U3 documents
  (so `es policy init` builds the U3 untrained bundle); presets keep "5,000 / 20,000 / 60,000
  steps" and obey the IR route's mark rule; `medium` equals U3's 20,000.
- `--resident-gpu` (U3's `extra`): keep it only if it fits a 12 GB GPU with 200 demonstrations
  (measure); otherwise drop it and say why in the recipe header.
- Dry-run golden for the new cycle through its generator, like `plan-cycle-vision.txt`.

## context

```
tests/fixtures/visible-learning/**
tests/golden/train/**
templates/cube-into-bin-hint.toml
crates/es/tests/cli.rs
crates/es-editor/src/model/template.rs
crates/es-editor-model/src/model/template.rs
python/es/fetch_backbone.py
docs/packets/M12/P-M12-R7-hint-u3.md
docs/packets/M12/P-M12-R7-hint-u3.ko.md
```

## oracle

1. `es loop cycle --recipe <new cycle> --dry-run` matches its new golden (generated through the
   generator, reviewed).
2. `es policy init` from the template's `[bundle]` → a bundle that passes the gate's checks against
   the new cycle's `[eval] config` (the check Y5b's test runs), without Python.
3. The template test (`the_committed_templates_parse_and_name_real_files` or its successor) passes.
4. No learning run inside this packet: other packets compile on the same PC meanwhile (review
   S-6). The orchestrator runs the short and medium end-to-end runs after the merge, on a quiet PC.

## forbidden

Changing `evaluation.toml`, any acceptance threshold, the Safety Plane envelope (S-7 is the
owner's), `training-u3.toml` or any other committed document in place; connecting to any remote
server.
