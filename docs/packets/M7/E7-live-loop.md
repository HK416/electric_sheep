# M7 E7 — the whole loop, live: collection and training publish, the editor draws the loss curve and what the network sees

Spec: §23.1 (the editor is a client of a running process), §23.3 ("what is visible during a
run": the real observation images, per-node statistics, the learning curve; budgeted sampling;
images rate-limited separately; gate 9 < 1 %), §13.1 (collect → train → evaluate → watch is one
workflow), §12.4 (the nine metrics, never `step/s`), §19.3 (`metrics.parquet` is the learning
curve's identity slot), §28.10 rule 3. Owner note 2026-09-21 after the local cycle demo: *show
loss-curve-like data in the editor* and *the training process has no rendered view — that is a
pity.* Design notes: `docs/design/telemetry-protocol.md` (E4's "Producers" section gains two
producers and one stream), `docs/design/editor-shell.md` section 16. Depends on **E4** (the
`RunSink`/`Publisher` pattern, streams 1–4, `--attach`), **E5** (the launch panel's `--telemetry`
field), **T1/T2** (`es train`, `es loop cycle`), **T6** (the augmented training tensor is what
the network sees).

## the question

After E4 only `es eval run` publishes. A person who presses Start on *the whole loop* watches
log lines scroll for ten minutes and sees no picture and no curve. **Can collection publish
its episodes the way evaluation does, can training publish its loss, learning rate,
throughput and the very tensor it is fitting — every `N` steps, at < 1 % cost — and can one
address carry all four stages of a cycle so the editor shows the loop as one live run?**

## spec

* **One `Publisher`, shared.** E4's `Publisher` (frame construction, non-blocking `Server::publish`)
  moves from `crates/es/src/cmd/eval.rs` to `crates/es/src/cmd/telemetry.rs` and gains a
  `stage: &'static str` field it puts into every stream-1 event (`"collect"`, `"expert-gate"`,
  `"train"`, `"eval"`, `"showcase"`), so a consumer can tell the cycle's stages apart on one socket.
  `es loop cycle --telemetry <addr> [--telemetry-token] [--telemetry-image-every N]` binds
  **once** before the first stage and hands the same publisher to every in-process stage; the
  standalone commands bind their own as `es eval run` does. Stream-1 `stage.begin` / `stage.end`
  events bracket each stage with its wall-clock.
* **Collection publishes.** `es loop collect --telemetry <addr>` publishes, through the collector's
  existing per-tick hooks (`FrameSink`, the intervener) and **no change to what it writes**:
  stream 1 `episode.begin` / `episode.end { episode, seed, outcome, steps }`, stream 2 one
  `[frame, tick, source, violation bits]` per control tick exactly as `es eval run` does (the
  plane's own verdict), stream 4 the observation image every `--telemetry-image-every` ticks. A
  `--frames`-less collect publishes no image and says so once. Oracle: the dataset under `--out`
  is byte-identical with and without the flag.
* **Training publishes.** `python/es/train_act.py --progress-every N` prints, every `N` optimizer
  steps, one JSON line `{"progress": {"step", "loss", "lr", "samples_per_s", "elapsed_s"}}` on
  stdout **before** the final summary line (which stays the last line and stays byte-identical);
  `--sample-every N` writes `metrics/sample-<step>.bin` + `.json` (`{"shape":[h,w,3],"port":…}`)
  holding **one image input of the current batch after augmentation, as `Rgb8` in the sensor's
  colour space** — the tensor the network is fitting, de-normalised only for display — and prints
  `{"sample": "<path>"}`. `es train --telemetry <addr>` switches the trainer's stdout to a streamed
  read (`Stdio::piped` + `BufReader::lines`, the summary still the last line) and publishes:
  stream **5** `Scalars([step, loss, lr, samples_per_s])` per progress line (the new stream id is
  data, not schema — the note's table gains a row), stream 1 `checkpoint { step, policy_hash }`
  when a mark is packed, stream 3 `Metrics` at the end with the §12.4 fields the run can fill
  honestly (`training_samples_per_sec`; everything else `None`), stream 4 the sample image. With
  no flag `train_act.py` gets neither `--progress-every` nor `--sample-every`, so every measured
  run's plan and `training.lock` are unmoved (`plan-ir.txt` golden unchanged).
* **The editor draws it.** `model/train_view.rs`: `TrainView` folds stream 5 into
  `Curve { step: Vec<u32>, loss: Vec<f32>, lr: Vec<f32> }`, keeps `checkpoints: Vec<(u32, String)>`,
  the last sample image and `throughput`, and answers `eta(total_steps)`. `LiveRun` (E4) gains the
  stage strip and folds `episode.*` from collect the way it folds `cell.*` from eval, so a cycle's
  collect episodes are rows on the Run tab too. The **Live** tab (관찰) gets a *Training* section:
  the loss curve (log-scale toggle) and the lr curve as polylines through `ui.painter()` — **no
  plotting dependency** — with the checkpoint marks, the current step / total / ETA, and the sample
  image beside the run's camera image; the Run tab shows the stage strip above the table. `app.rs`
  wires; every number and label comes from the models and the i18n tables (E6).
* **Launch panel.** `es train` and `es loop cycle` kinds gain the `--telemetry`, token and
  `--telemetry-image-every` fields (E5's `LaunchField`s are shared), so Start on the whole loop
  attaches by itself as evaluation does today; the three `launch-*.txt` goldens are regenerated once
  (the argv of the fixture inputs changes only by the new default `--telemetry` on the two kinds —
  say so in the note; goldens otherwise read-only).
* **Gate 9.** `es train` 2,000 steps on the fixture bake with and without `--telemetry
  --progress-every 10 --sample-every 100` (one draining client), three runs each, on the server;
  `es loop collect` 8 episodes likewise. Table in the note beside E4's; `Target / Status: unverified`
  until measured.
* **Not here.** Per-node activation statistics (§23.3) — the lowered module has no hook for them;
  a rollout during training; pausing/steering the trainer; a protocol change.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
crates/es/src/cmd/telemetry.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/cycle.rs
crates/es/src/cmd/mod.rs
crates/es/tests/cli.rs
crates/es-data/src/collect.rs
crates/es-data/src/training.rs
python/es/train_act.py
crates/es-editor/src/model/train_view.rs
crates/es-editor/src/model/live_run.rs
crates/es-editor/src/model/telemetry_view.rs
crates/es-editor/src/model/launch.rs
crates/es-editor/src/model/labels.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/app.rs
crates/es-editor/i18n/en.toml
crates/es-editor/i18n/ko.toml
tests/golden/editor/launch-*.txt
docs/design/telemetry-protocol.md
docs/design/telemetry-protocol.ko.md
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E7-live-loop.md
docs/packets/M7/E7-live-loop.ko.md
```

`cmd/telemetry.rs` (new: the shared `Publisher` with `stage`), `eval.rs` (uses it), `loop.rs`
(collect publishes), `train.rs` (streamed trainer stdout, stream 5, checkpoints, sample), `cycle.rs`
(one publisher for all stages), `collect.rs` **only** if the collector's hook needs the episode
outcome passed through, `training.rs` **only** for the two new plan flags behind `--telemetry`,
`train_act.py` (`--progress-every`, `--sample-every`), the editor's `train_view.rs` (new) and the
folds, `launch.rs`/`labels.rs`/i18n (the fields), `app.rs` (wiring), the launch goldens, the two
design notes, this packet.

## oracle

1. `cargo test -p es --test cli train_telemetry_streams_the_curve` — `es train --telemetry
   127.0.0.1:<port>` on the fixture bake, 40 steps, `--progress-every 10`: a subscribed client
   receives 4 stream-5 frames with strictly increasing steps and one `checkpoint` event at 40;
   `training.lock`, the checkpoint and `metrics/loss.json` are byte-identical to a run without the
   flag. `SKIP` by name without `ES_PYTHON`.
2. `cargo test -p es --test cli collect_telemetry_publishes_every_episode` — 2 expert episodes
   with `--frames --telemetry`: `episode.begin`/`end` per episode, one stream-2 frame per tick, at
   least one stream-4 image; the dataset directory byte-identical to a run without the flag.
3. `cargo test -p es --test cli cycle_telemetry_is_one_address` — `es loop cycle --dry-run` with
   `--telemetry` shows the address on no stage line (it is the cycle's, not a stage's); the live
   run (oracle 4 of T2's shape, `#[ignore]`) delivers `stage.begin` for `collect`, `expert-gate`,
   `train`, `eval` in that order on one socket.
4. `cargo test -p es-editor train_view_folds_the_curve` — a scripted stream 5 + checkpoint events
   give a `Curve` equal to the trainer's `loss.json` values (bitwise `f32`) with the marks at the
   right steps, `eta` monotone; `live_run_folds_collect_episodes_like_cells`.
5. `cargo test -p es-editor launch_argv_is_the_golden` against the regenerated goldens; the
   i18n completeness test.
6. `cargo xtask ci` (layering: the editor gains no dependency; `es-eval` unchanged);
   `cargo xtask check-scope docs/packets/M7/E7-live-loop.md`.

## acceptance

Oracles 1–6 (1, 3's live half and gate 9 on the oracle server). The orchestrator presses Start on
the whole loop locally (`target/plan-u/demo/cycle.toml`) and watches: collect episodes appear as
rows with images, the Training section draws the loss falling with the sample image beside it,
then the evaluation rows. Screenshots of each stage under the worktree's `target/plan-u/e7/`.
Section 16 records the stream table (1–5), the `stage` field, the trainer's two flags and the
gate-9 table.

## forbidden

Changing what any stage computes or writes (byte-identity is oracle 1–2's assertion); a blocking
send on any run path; a protocol/schema change (`protocol.rs` frozen); a plotting crate or any new
dependency; `crates/es-eval/**`, `crates/es-telemetry/**`, `crates/es-safety/**`;
`docs/ARCHITECTURE*.md`. INV-17: sinks are closures. Rule 3: nothing decided in `app.rs`.
