# M7 E4 — live telemetry: `es eval run --telemetry` publishes, the editor attaches

Spec: §23.1 (the editor is a client of a running process; the protocol is backend-neutral),
§23.3 (what is visible during a run; budgeted sampling; "learning degradation from telemetry +
graph view < 1 %" — §28.7 gate 9), §12.4 (the nine metrics, never `step/s`), §25.1 (token on the
socket). Design notes: `docs/design/telemetry-protocol.md` (the wire shape and the TCP loopback
transport — read all of it), `docs/design/editor-shell.md` section 5 (the Telemetry tab's model)
— extend the latter with section 13. Predecessors: M1 W8 (`es_telemetry::transport::{Server,
Client}`), M4 editor (`TelemetryModel`, `Source` closure), E1 (the Run tab; live and finished
runs should look the same).

## the question

`es_telemetry` has a server, a client, a schema and a test suite, and **no producer**: nothing
in `es` publishes a frame, and the editor's `Source` is a canned replay. **Can an evaluation
publish what it knows — the per-tick `StepEvent`, the cell's progress, the nine metrics — with
no change to what it computes, and can the editor show it while it runs?**

## spec

* **Producer.** `es eval run --telemetry <addr> [--telemetry-token <t>]` binds
  `es_telemetry::transport::Server` before the first cell and publishes:
  - stream `1` `Event { kind: "cell.begin" | "cell.end", fields: { cell, suite, seed, outcome … } }`;
  - stream `2` `Scalars([tick, source as f64, violation bits as f64])` per control tick — one
    frame per tick is the §23.3 budget question: publish every tick but through the server's
    existing non-blocking `publish` (a slow client drops frames, the run never waits — assert
    that in the loopback test);
  - stream `3` `Metrics(PerfMetrics)` at each `cell.end` with the fields the run can honestly
    fill (`actions_per_sec`, `policy_inferences_per_sec`, `chunk_underrun_rate` from the plane's
    counters, `p50/p95_end_to_end_latency` only if measured — else `None`, never zero);
  - stream `4` `Image` of the **current observation frame** at most every `--telemetry-image-every N`
    ticks (default `0` = never), `Rgb8` bytes as the renderer wrote them — §23.3's "actual
    observation images", rate-limited separately as the spec says.
  The hook is one `Option<&dyn FnMut(Frame)>`-shaped sink in `es_eval::RunConfig` (or a channel
  the CLI drains) — `es-eval` must not depend on `es-telemetry`'s transport (layer 10 both; no
  same-layer dependency: put the *frame construction* in `crates/es/src/cmd/eval.rs`, and give
  `es-eval` only a generic per-tick callback carrying the `StepEvent` and cell context). Zero
  behaviour change when the flag is absent: `report.json` and `events.json` byte-identical.
* **Consumer.** `es-editor --attach <addr> [--token <t>]` (and a field + Connect button in the
  Telemetry tab): `Client::connect` + `subscribe` become the `Source` closure through
  `try_recv`; `TelemetryModel` gains a `LiveRun` view-model (`live_run.rs`) that folds streams
  1–4 into the same `CellRow`/`Timeline` shapes E1's `RunView` uses, so the Run tab draws a live
  run and a finished one with the same code — the design note states the shared type.
* **Gate 9 measurement.** `es eval run` on the nominal fixture (`--jobs 1`) with and without
  `--telemetry` (one attached client that drains): wall-clock of the two, three runs each,
  reported as an observation beside the `< 1 %` target (`Target / Status: unverified` until the
  server number exists).

## context

The globs `cargo xtask check-scope` reads (its parser wants a `## context` heading and a
fenced block or a bullet list), then the same scope in prose:

```
crates/es-eval/src/runner.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
crates/es-telemetry/src/transport.rs
crates/es-editor/src/model/live_run.rs
crates/es-editor/src/model/telemetry_view.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/lib.rs
crates/es-editor/src/app.rs
crates/es-editor/src/main.rs
crates/es-editor/Cargo.toml
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/design/telemetry-protocol.md
docs/design/telemetry-protocol.ko.md
docs/packets/M7/E4-live-telemetry.md
docs/packets/M7/E4-live-telemetry.ko.md
```

`crates/es-eval/src/runner.rs` (the per-tick callback in `RunConfig`; nothing else), `crates/es/src/cmd/eval.rs`
(flags, frame construction, server lifetime), `crates/es/tests/cli.rs` (new `eval_telemetry_*`
tests), `crates/es-telemetry/src/transport.rs` **only** if a non-blocking guarantee needs a
test-visible counter (dropped frames) — no protocol change, `crates/es-editor/src/model/live_run.rs`
(new), `model/telemetry_view.rs`, `app.rs`, `main.rs`, `Cargo.toml` (no external addition),
`docs/design/editor-shell*.md` section 13, `docs/design/telemetry-protocol*.md` (a "producers"
subsection naming the four streams), `docs/packets/M7/E4-live-telemetry*.md`.

## oracle

1. `cargo test -p es --test cli eval_telemetry_publishes_every_tick_in_order` — loopback: an
   evaluation of the small fixture with `--telemetry 127.0.0.1:0` (print the bound port), a test
   client subscribed to streams 1–3 receives `cell.begin`/`cell.end` for every cell and one
   stream-2 frame per tick with strictly increasing ticks inside each cell; the run's
   `report.json` and `events.json` are byte-identical to a run without the flag.
2. `cargo test -p es --test cli eval_telemetry_never_blocks_the_run` — a client that connects
   and never reads: the run finishes, its report is unchanged, and the server's `stats()` shows
   dropped frames > 0.
3. `cargo test -p es-editor live_run_folds_streams_into_run_rows` — a scripted `Vec<Message>`
   through the `Source` produces `CellRow`s and a `Timeline` equal to what `RunView::open` gives
   for the same run written to disk (E1's fixture, replayed as messages).
4. `cargo test -p es-telemetry` unchanged and green (S-4's flake is out of scope: if it fires,
   rerun and say so).
5. `cargo build -p es-editor`; `cargo xtask ci` (layering: `es-eval` gains no dependency;
   `es-editor` may depend on anything); `cargo xtask check-scope docs/packets/M7/E4-live-telemetry.md`.

## acceptance

Oracles 1–5. The orchestrator runs `es eval run --telemetry 127.0.0.1:7777` on the fixture and
`es-editor --attach 127.0.0.1:7777` locally and watches cells appear. Gate-9 table on the server
in design note section 13.

## forbidden

Changing what an evaluation computes or writes; a same-layer dependency `es-eval → es-telemetry`;
a blocking send anywhere on the run path; a protocol/schema change (`protocol.rs` is frozen at
its version; a new stream *id* is data, not schema); `crates/es-safety/**`; `docs/ARCHITECTURE*.md`;
goldens. INV-17: no new trait (the sink is a closure type).
