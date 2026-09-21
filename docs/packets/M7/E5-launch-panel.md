# M7 E5 — the launch panel: the editor starts `es eval run` / `es train` and attaches

Spec: §23.1 (**the editor does not host training; it is a client of a running process**), §23.3
("what is visible during a run"; controls: pause/step/reset are *not* this packet — they need a
protocol the run does not speak), §13.1 (each stage is a command and an artifact), §28.10 rule 3
(nothing decided in `app.rs`) and (E5). Design note: `docs/design/editor-shell.md` (+ `.ko.md`)
section 14. Depends on **E4** (`es eval run --telemetry`, `es-editor --attach`, `LiveRun`) and on
**T1**/**T2** (`es train --recipe`, `es loop cycle --recipe`) as they are.

## the question

After E4 a person can watch a run — if they started it in a terminal and typed the address twice.
**Can the editor build the command line from what it already knows (the open bundle or run, a
recipe path, the Evaluation IR path), start it as a child process it does not own the semantics
of, show its exit code and its last lines, and attach itself to the telemetry it asked for — while
staying a client (§23.1) and deciding nothing in `app.rs`?**

## spec

* **`model/launch.rs`** — `LaunchModel { kind: Eval | Train | Cycle, fields }` where the fields are
  exactly the flags of the three commands the model renders (`es eval run --config --policy --scene
  --out [--frames] [--jobs] --telemetry <addr>`, `es train --recipe --out`, `es loop cycle --recipe
  --out`), pre-filled from the session when it can be (`--policy` from the open bundle's path,
  `--out` from the run directory's parent, the `--telemetry` address from the Telemetry tab's
  attach field, default `127.0.0.1:7777`). `argv() -> Vec<String>` is a pure function of the
  fields; `es_binary()` resolves the `es` executable **by one rule** — `ES_BIN` if set, else `es`
  beside the editor's own executable, else `es` on `PATH` — and says which it found.
* **`spawn()`** starts `std::process::Command` with `argv()`, stdout/stderr piped to a reader
  thread that feeds a bounded ring of the last 200 lines through an `mpsc` channel; `poll()` drains
  the channel and `try_wait()`s the child; `state()` is `Idle | Running { pid, since } | Exited
  { code, lines } | Failed(String)`. `kill()` exists for the person who started it — it is the only
  control, because the run speaks no control protocol (§23.3's pause/step are listed as not here).
  Never blocking on the UI thread; no `tokio`, no new dependency.
* **Attach follows launch.** When the launched command carries `--telemetry <addr>`, the model
  reports `attach: Some(addr)` and the app hands it to E4's attach path *after* the child is
  running — the editor connects as a client, exactly as if the person had typed the address. Exit
  codes are shown by the meaning the CLI documents (0 pass, 1 fail/runtime, 2 usage, 3 skipped);
  the mapping table lives in the model, not in the panel.
* **The panel** (`app.rs`, wiring only): a `Launch` section in the Run tab — the three kinds as a
  combo, the fields as text inputs, the rendered command line read-only, Start / Kill, the state
  line, the last lines in a scroll area. Nothing else changes in the tabs.
* **Not here.** Pause/step/reset/hot-patch (§23.3) — no protocol; a job queue; remote hosts;
  environment editing beyond `ES_BIN`. Listed in the note under "not here".

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
crates/es-editor/src/model/launch.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/app.rs
crates/es-editor/src/lib.rs
crates/es-editor/tests/**
tests/golden/editor/launch-*.txt
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E5-launch-panel.md
docs/packets/M7/E5-launch-panel.ko.md
```

`model/launch.rs` (new; the whole model and its tests), `model/mod.rs` (export), `app.rs` (the
section), `lib.rs` (if a re-export is needed), `tests/golden/editor/launch-{eval,train,cycle}.txt`
(the rendered argv of the fixture inputs, one per line, generated once), the design note, this
packet. **No `Cargo.toml` change**: `std::process` and `std::sync::mpsc` are enough.

## oracle

1. `cargo test -p es-editor launch_argv_is_the_golden` — the three kinds from fixed fields render
   argv byte-identical to the three goldens; `--frames`/`--jobs` absent when unset; every path
   passed through verbatim (no normalisation the CLI does not do).
2. `cargo test -p es-editor launch_reports_the_exit_code` — `spawn()` of the platform shell
   (`cmd /C exit 3` on Windows, `sh -c "exit 3"` elsewhere) through the same `spawn` path (the
   program is a parameter) reaches `Exited { code: 3 }` within a bounded number of `poll()`s;
   stdout lines (`echo`) arrive in order; `kill()` on a sleeping child reaches `Exited` with a
   non-zero code.
3. `cargo test -p es-editor launch_attaches_only_after_running` — `attach()` is `None` while `Idle`,
   `Some(addr)` once `Running` for a command that carries `--telemetry`, `None` for one that does not.
4. `cargo test -p es-editor es_binary_resolution_is_one_rule` — `ES_BIN` wins, then the sibling,
   then `PATH`; the reported reason names which.
5. `cargo build -p es-editor`; `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/E5-launch-panel.md`.

## acceptance

Oracles 1–5. The orchestrator opens the fixture bundle, picks `Eval`, fills the Evaluation IR
path, presses Start, sees cells arrive on the Run tab through E4's attach and the exit code at the
end; `Kill` mid-run ends the child. The note's section 14 states the client rule and the exit-code
table.

## forbidden

Hosting anything: no in-process evaluation or training; a control protocol; a new dependency;
`crates/es/**`, `crates/es-telemetry/**`, `crates/es-eval/**`; deciding in `app.rs` what the
command line is; `docs/ARCHITECTURE*.md`; goldens other than the three new ones. INV-17: no new
trait.
