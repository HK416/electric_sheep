# M7 E1 — the run browser: a finished `es eval run` / `es loop collect`, opened in the editor

Spec: §23.3 (what is visible while learning: Safety Plane event log, per-term decomposition,
chunk/latency histograms), §10.5 (artifacts of a run), §23.1 (the editor is a client),
§28.10 rule 3 (nothing is decided in `app.rs`). Design note to extend: `docs/design/editor-shell.md`
(+ `.ko.md`) with a new section 10. Predecessors: M5 V3/V4 (`report.json`, `events.json`,
`es video mosaic`), V9 (`traj/<cell>.estraj`), M4 editor stage 2.

## the question

Every run writes `report.json`, `events.json`, `traj/*.estraj` and (with `--frames`)
`frames/<cell>/NNNNNN.bin`. The only reader of those is `es video mosaic` and a person with
`jq`. **Can the editor open a run directory and show what happened — per cell, per tick — with no
process on the other end?**

## spec

A **Run** tab. `es-editor <run-dir>` or the File field opens a directory that holds `report.json`
(and optionally `events.json`, `traj/`, `frames/`); a bundle path still opens the graph as today,
the two are told apart by what is on disk.

Headless view-model `crates/es-editor/src/model/run_view.rs`:

* `RunView::open(dir) -> Result<RunView, RunError>` reads `EvaluationReport` (`es_ir::evaluation`),
  the events map (`BTreeMap<String, Vec<es_eval::runner::StepEvent>>` — `es-editor` may depend on
  `es-eval`, layer 12), lists `traj/<cell>.estraj` and `frames/<cell>/` without loading them.
* `RunView::cells() -> Vec<CellRow>`: cell name, suite, seed if the report carries it, outcome
  (from the report's per-cell metrics: `success_rate`, `episode_length`,
  `envelope_violation_rate` and whatever else is present — the model lists metric names, it does
  not hard-code them), whether a trajectory and frames exist.
* `RunView::timeline(cell) -> Timeline`: per tick the `EventSource` (Policy / Clamped / Fallback /
  Human) and the decoded `ViolationKind` set (`es_safety::EventSet::from_bits` or the equivalent —
  decode, do not re-derive), plus per-kind totals and the first tick of each kind. A bucketed
  form `Timeline::buckets(n)` for drawing at any width.
* `RunView::acceptance()` mirrors `report.acceptance` (rule, value, threshold, passed).
* `RunView::frame(cell, index) -> Option<Rgb8Image>` loads one `.bin` through the existing
  `Rgb8Image` type using the sibling `layout.json`; nothing is cached beyond what is asked for.
* Errors name the missing file; a directory with `report.json` alone still opens (timeline and
  filmstrip empty, said so in the status line).

`app.rs` draws: the cell table (sortable by clicking a header — sorting is a `RunView` method,
tested), the acceptance rows with pass/fail colour, and for the selected cell the timeline strip
(one colour per `EventSource`, violation kinds as ticks beneath it) and a filmstrip of at most 8
frames sampled evenly. Selecting a cell also selects it for E2's replay if that packet has landed
(a `selected_cell()` accessor is the whole coupling).

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

```
crates/es-editor/src/model/run_view.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/lib.rs
crates/es-editor/src/app.rs
crates/es-editor/src/main.rs
crates/es-editor/Cargo.toml
tests/fixtures/visible-learning/run/**
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E1-run-browser.md
docs/packets/M7/E1-run-browser.ko.md
```

`crates/es-editor/src/model/run_view.rs` (new), `crates/es-editor/src/model/mod.rs`,
`crates/es-editor/src/lib.rs` (re-exports), `crates/es-editor/src/app.rs` (the Run tab and the
open-by-directory branch), `crates/es-editor/src/main.rs`, `crates/es-editor/Cargo.toml`
(`es-eval`, `es-safety` as dependencies — both workspace crates, no external one),
`tests/fixtures/visible-learning/run/` (a new small fixture run: `report.json` for 4 cells,
`events.json` with a few `Clamped`/`Fallback` ticks carrying violation bits, one 96×96 frame
per cell with `layout.json`; generate it from real types in a `#[ignore]`d generator test so
its bytes are the types' own), `docs/design/editor-shell*.md` section 10,
`docs/packets/M7/E1-run-browser*.md`.

## oracle

1. `cargo test -p es-editor run_view_reproduces_the_report` — `RunView::open` on the fixture run:
   the cell rows carry the same metric values the report holds (compared as the report's own
   `MetricValue`, not re-parsed floats), `acceptance()` equals `report.acceptance`, `passed`
   agrees.
2. `cargo test -p es-editor timeline_buckets_sum_to_the_events` — for every cell, the per-kind
   totals of `timeline(cell)` equal the counts a direct pass over `events.json` gives, and
   `buckets(n)` sums to the same totals for `n` in `{1, 7, 64}`.
3. `cargo test -p es-editor a_report_alone_still_opens` — a directory with only `report.json`
   opens; `timeline` is empty and `frame` is `None`, no panic, the status names what is missing.
4. `cargo test -p es-editor sorting_is_stable_and_total` — sorting by each column is a
   permutation of the rows and is stable on ties.
5. `cargo build -p es-editor` compiles the tab; `cargo xtask ci` green; layering unchanged.

## acceptance

Oracles 1–5 pass. Opened by the orchestrator on the local machine against a real run copied
from the server (`~/artifacts/plan-v/v19b/`): the table, the acceptance rows, one timeline with
visible `Clamped` ticks and a filmstrip appear. The design note section 10 records the tab, the
view-model API and what it deliberately does not do (no live process, no editing of a run).

## forbidden

`crates/es-eval/**`, `crates/es-safety/**`, `crates/es-ir/**` (read their types, change nothing);
the existing four tabs' behaviour; any decision in `app.rs` (sorting, bucketing, decoding all
live in the model); a new external dependency; `docs/ARCHITECTURE*.md`; every existing golden.
