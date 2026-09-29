# M12 R2 — the LeRobot route publishes training progress

Found by Y-V item 1 on 2026-09-29 (the camera-only cube template, resumed `--from train`). Spec
§23.3 (watching a run); `docs/design/editor-shell.md` section 16 (the training stream);
`docs/design/editor-redesign.md` section 6.4 (the traffic light).

## the defect

`es train` publishes stream 5 (`[step, loss, lr, samples_per_s]`) only from `{"progress": …}`
JSON lines (`crates/es/src/cmd/train.rs`, the line reader near "`{\"progress\": ...}` line reaches a
viewer"). `train_act.py` and `train_ppo.py` print those; **`lerobot-train` does not**. On the
LeRobot route a viewer therefore hears nothing for the whole training stage (minutes to hours):
no curve, no ETA, and the editor's traffic light turns *not responding* after
`THRESHOLDS.silence_s` while training is healthy.

What `lerobot-train` (0.6.1) does print, measured in this run's log:
- a tqdm bar on stderr, `\r`-separated:
  `Training:  70%|███████   | 3510/5000 [03:51<01:39, 14.94step/s]` — exact step, total, rate;
- every `log_freq` steps (200 by default) an INFO line containing
  `step:3K smpl:26K ep:49 epch:0.25 loss:0.131 grdn:… lr:…` — the loss (the step there is
  abbreviated and must not be used as the step).

## spec

- On the external (LeRobot) route only, the trainer's stdout **and** stderr are also read for
  those two forms (split on `\r` as well as `\n`). A progress row is published when a metric line
  gives a new loss: `step` = the latest exact tqdm step, `loss` = that line's loss, `lr` = that
  line's `lr` when present, `samples_per_s` = the latest tqdm rate × `[run] batch`.
- Nothing is published before the first loss is known — never a NaN or a zero standing in for an
  unknown loss (the light reads a NaN as *broken*).
- The IR and RL routes are unchanged; `{"progress": …}` lines keep their meaning. Nothing is
  written to disk that was not written before; `training.lock` and every hash are unmoved.
- The parser is a pure function over one line with its own unit tests; the samples are copied
  from a real `lerobot-train` log (the lines above), not invented.

## context

```
crates/es/src/cmd/train.rs
crates/es/src/cmd/telemetry.rs
crates/es/tests/cli.rs
docs/packets/M12/P-M12-R2-lerobot-progress.md
docs/packets/M12/P-M12-R2-lerobot-progress.ko.md
```

## oracle

1. Unit tests for the line parser: the tqdm form (step, total, rate), the metric form (loss,
   lr), a line with neither, a `\r`-joined run of several tqdm updates (the last one wins), and a
   metric line before any tqdm line (no row yet).
2. The existing train and telemetry tests pass unchanged; `cargo test -p es --test cli train`.
3. fmt, clippy `-D warnings`, `check-scope`, `verify-goldens`.

## forbidden

Changing what `train_act.py` / `train_ppo.py` print or how their lines are read; publishing an
invented loss; any change to `es-editor`; connecting to any remote server.
