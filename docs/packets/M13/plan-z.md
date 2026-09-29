# M13 plan Z — S2: watching it learn, and learning from failures

> Packets for sub-project S2 of `docs/design/editor-redesign.md` (sections 3, 5). The owner
> granted autonomy on 2026-09-29 ("남은 구현을 진행"): design follows the agreed direction; only
> spec, Safety Plane and hash decisions go back to the owner. Each task is a §1.2 packet: its
> **Files** block is its `context` for `cargo xtask check-scope`.

**Goal:** a beginner sees the policy get better *while* it trains (a short test after every
checkpoint, as video), reads *why* each failed attempt failed in the task's own words, and can
press one button to train again on more of what it failed at — without training on the
evaluation's seeds.

**What the first real runs changed in S2's direction (2026-09-29).** The cube tasks declare no
failure predicate: every failed episode is a *timeout* (`YV-verification.md`), so "which
`Terminate` fired" (the design note's S2 line) names nothing. S2b classifies failures from the
object's trajectory instead. `es loop collect` has no perturbations, so S2c adds them from the
Evaluation IR's own `PerturbationKind`s. The design note is updated in Z7.

## Global Constraints

- Everything in `plan-y.md`'s Global Constraints still holds (MSRV 1.85, egui 0.32.3, i18n and
  label totality, decisions in `es-editor-model`, goldens by generator only, no trailers,
  worktree + `--ff-only`, never push, `CARGO_TARGET_DIR=target/alt` in the main tree).
- **No remote server.** Local PC only; `ES_PYTHON=F:/Projects/electric_sheep/.venv/Scripts/python.exe`
  for real runs (LeRobot oracles read `ES_LEROBOT_PYTHON`).
- **No learning run inside an agent packet** while other packets compile (review S-6); real runs
  are the orchestrator's, on a quiet PC.
- **§13.3 discipline:** a preview and a re-training never touch the Evaluation IR a run is judged
  by; preview IRs are derived, written beside their results, and never compared with the
  acceptance. Re-training collects with seeds disjoint from every evaluation seed.
- **Safety Plane:** untouched; perturbed collection runs under the same plane (INV-12).
- **Hashes:** no existing hash definition moves. New provenance (preview IR, perturbation spec,
  merged datasets) is recorded in files and the ledger, not in existing hash inputs.

## Review Focus

1. A checkpoint preview must not slow training past what a person would notice: measured in
   Y-V; the preview is one worker, one suite, 4 episodes.
2. A LeRobot checkpoint read while `lerobot-train` is still writing it: import only a complete
   directory (Z1 test).
3. A failed episode with no trajectory recorded (an old run, `--traj` off): no cause invented,
   the generic termination cause shown (Z4 test).
4. "Train again" on a run whose evaluation had no failures: the button says so and does nothing
   (Z4 test).
5. Seeds: a perturbed re-collection whose seeds would overlap the evaluation's is refused (Z2
   test).

---

## Waves

| Wave | Tasks | Needs |
|---|---|---|
| 1 | Z1 checkpoint previews (es) · Z2 perturbed collection (es) | M12 R4–R7 merged |
| 2 | Z3 the "again" cycle (es, es-data) · Z4 S2 models (es-editor-model) | Z1, Z2 |
| 3 | Z5 S2 screens (es-editor) · Z6 template outcome fields | Z3, Z4 |
| 4 | Z7 real runs (orchestrator), docs, review | wave 3 |

---

### Task Z1: a short test after every checkpoint

**Files:** `crates/es/src/cmd/train.rs`, `crates/es/src/cmd/cycle.rs`,
`crates/es/src/cmd/telemetry.rs`, `crates/es-data/src/training.rs`, `crates/es/tests/cli.rs`,
`tests/golden/train/**` (generator only), `docs/packets/M13/Z1-*.md` + `.ko.md`.

**Behaviour.**
- `Cycle` gains `[eval.preview]` (optional; default on in the templates' cycles through Z6):
  `episodes` (default 4), `suite` (default the Evaluation IR's first suite, the nominal one),
  `frames` (default true). Absent table = today's behaviour, byte-identical dry-run plans for the
  committed cycles (their goldens do not move).
- After each checkpoint bundle is written (IR route: after `es policy pack` per mark; LeRobot
  route: **as soon as** `lerobot-train` has finished writing `checkpoints/<NNNNNN>/pretrained_model`
  — `es train` watches the directory while the trainer runs and imports each complete mark then,
  instead of all at the end; a directory is complete when `model.safetensors` and
  `config.json` exist and the trainer's next progress line has been read after they appeared),
  `es loop cycle` starts `es eval run` as a **child process** (the running `es` binary) on that
  bundle with a derived Evaluation IR: the chosen suite only, its first `episodes` seeds, same
  metrics, acceptance removed. It writes `<run>/preview/<mark>/` (the derived IR as
  `evaluation.toml`, then the usual `report.json`, `episodes.json`, `frames/`, `traj/`) with
  `--jobs 1`, and never blocks the trainer; previews are queued one at a time.
- Telemetry (stream 1): `preview.begin{step, dir}` and `preview.end{step, dir, successes,
  episodes, code}`. The cycle's ledger gets one `preview` row per mark (a new `LoopKind` only if
  the ledger schema allows it without moving any hash; otherwise the rows go to
  `<run>/preview/index.jsonl` — decide and say which).
- The cycle waits for the last queued preview before its own eval stage.

**Oracle.** Unit tests: the derived IR (one suite, N seeds, no acceptance) for the committed
`evaluation.toml`; the LeRobot completeness rule on a temp directory written in stages; the
dry-run plan for a cycle with `[eval.preview]` pinned by a new golden; committed cycles' goldens
unchanged. Python-gated: none required (Z7 runs it for real).

### Task Z2: collection under the Evaluation IR's perturbations, with fresh seeds

**Files:** `crates/es/src/cmd/loop.rs`, `crates/es-data/src/collect.rs`,
`crates/es-eval/src/perturb.rs` (visibility only), `crates/es/tests/cli.rs`,
`docs/packets/M13/Z2-*.md` + `.ko.md`.

**Behaviour.**
- `es loop collect … --perturb <evaluation.toml> --suites <a,b> [--seed <S>]`: each episode `i`
  runs under suite `suites[i % len]`'s perturbations through `es_eval::PerturbationPlan`, exactly
  as `es eval run` applies them (same streams, same keying), under the same Safety Plane.
- **Seeds:** the collect seed range `[S, S+N)` must not intersect the Evaluation IR's resolved
  seeds; an overlap is refused by name (`--seed` overlaps evaluation seeds 101–116).
- Provenance: the dataset's per-episode meta records the suite name; the ledger's collect row
  records `perturb.config` (path + `evaluation_hash`) and `perturb.suites`.
- The expert still demonstrates (`--expert`), so demonstrations stay successful under the
  perturbation where the expert can; the collect summary prints successes per suite.

**Oracle.** Unit tests: suite assignment by index; the seed-overlap refusal; the ledger row.
Python-gated CLI test (skips here unless `ES_PYTHON`): 4 episodes over two suites, each episode's
recorded suite and a light-intensity episode's rendered frame differ from the nominal one.

### Task Z3: the "again" cycle

**Files:** `crates/es-data/src/training.rs`, `crates/es/src/cmd/cycle.rs`,
`crates/es/tests/cli.rs`, `tests/golden/train/**` (generator only), `docs/packets/M13/Z3-*.md` +
`.ko.md`.

**Behaviour.** `Cycle` gains, all optional:
- `[collect] perturb = { config = "...", suites = ["..."] }` → Z2's flags;
- `[collect] merge = ["<earlier dataset root>", ...]` → after collect, `es loop distill` merges
  the new dataset with those roots into `collect/merged`, and training uses the merged root;
- `[train] init = "<bundle or lerobot checkpoint dir>"` → the IR route's `[init] policy`; the
  LeRobot route's `--policy.path` (its documented fine-tuning path).
Dry-run plans show the new steps; the committed cycles' goldens do not move.

**Oracle.** Dry-run golden for an "again" cycle (fixture paths); schema tests refusing
`merge` without `collect`, `init` pointing at a missing file.

### Task Z4: S2 models (`es-editor-model`)

**Files:** `crates/es-editor-model/src/model/{watch.rs, results.rs, project.rs, template.rs}`, a new
`model/preview.rs` and `model/outcome.rs`, `crates/es-editor-model/i18n/*`,
`docs/packets/M13/Z4-*.md` + `.ko.md`.

**Behaviour.**
- `preview.rs`: previews of a run from `preview.begin/end` events and from `<run>/preview/*/` on
  disk (a re-opened run shows them too): step, successes/episodes, the cell to play first.
- `outcome.rs`: from a trajectory (`.estraj`) and the template's `[outcome]` (Z6): the object
  body's position over time → `NeverLifted` (max height gain < `lift_m`), `LeftOutside` (lifted,
  final position outside the target region), `InsideTooLate` (final position inside the target
  region but the episode timed out). The target region is the AABB of the scene geoms whose names
  start with the template's `target` stem (the rule Y15 already uses for "bin"). No trajectory →
  `None` (the generic cause stays).
- `results.rs`: a failed tile's cause prefers the outcome class; the "why it failed" list counts
  outcome classes when present.
- `project.rs`: `write_run_again(template, repo, project, previous: &RunFolder, results, settings)`
  → a Z3 cycle: `perturb.suites` = the suites with the lowest success (ties → all tied; none failed
  → refused with a reason the screen shows), fresh seed range after every earlier run's collect
  seeds and outside the evaluation's, `merge` = earlier runs' dataset roots, `init` = the previous
  run's evaluated checkpoint.

**Oracle.** Headless tests for each: preview list from a fixture event stream and from a fixture
directory; outcome classes on synthetic trajectories (lifted-and-dropped, never lifted,
inside-late) and on the committed `.estraj` fixture; `write_run_again`'s recipe parses as a Z3
`Cycle`, its seeds avoid 101–116 and earlier ranges, and a run with no failures is refused.

### Task Z5: S2 screens (`es-editor`)

**Files:** `crates/es-editor/src/ui/{train.rs, results.rs}`, `crates/es-editor/src/app.rs`
(hooks only), `docs/packets/M13/Z5-*.md` + `.ko.md`.

**Behaviour.** ③'s centre plays the newest preview (the existing replay canvas) with "4번 중 1번
성공 — 1,000걸음 때"; the curve marks each preview step, a click selects it. ⑤ shows the outcome
classes and the **"실패 위주로 다시 학습"** button (disabled with the model's reason when refused),
which writes the next run through `write_run_again` and starts it like Start.

**Oracle.** Build, clippy, tests; screenshots of ③ with a preview and ⑤ with the button (in-app
hook, never committed).

### Task Z6: the templates' outcome and preview fields

**Files:** `templates/*.toml`, `tests/fixtures/visible-learning/cycle-*.toml` only through new
files, `crates/es-editor-model/src/model/template.rs` (the `[outcome]` table), i18n.

`[outcome] object = "cube"`, `target = "bin"`, `lift_m = 0.02`, with plain-word names for the
object and target in both languages; the templates' cycles gain `[eval.preview]`.

### Task Z7: real runs and documents (orchestrator)

On a quiet PC: the hint card (U3, R7) at medium with previews; one "train again" run after it;
the numbers, the preview cost on training time, and the outcome classes recorded in
`YV-verification.md`; the design note's S2 section rewritten to what was built; the review.
