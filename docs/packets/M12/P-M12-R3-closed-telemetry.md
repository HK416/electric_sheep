# M12 R3 — the editor notices a closed telemetry connection

Found by Y-V item 1 on 2026-09-29. `docs/design/editor-redesign.md` section 6.4 (re-opened runs,
the traffic light); `docs/design/editor-shell.md` section 13 (the attach); packet Y12's
documented limit ("an attached run whose process dies reads *running*, then *not responding*,
until the project is re-opened").

## the defect

`crates/es-editor/src/model/telemetry_view.rs` `source_of` turns a closed or broken connection
into "nothing right now" forever (a deliberate M7/E4 choice, so a tab is not cleared over a
dropped socket). `model/watch.rs` counts an attached run as alive (`Child::NotOurs`), so after the
run's `es` exits the ③/④ light reads *not responding* indefinitely, and a run resumed from
outside the editor (a new `--telemetry` address in `telemetry.txt`) is never picked up. Seen on
this PC: the editor attached to the camera-only run's first attempt (port 7812); the attempt
failed and exited; the resumed attempt published on 7813; the editor kept showing *not
responding* for a run that no longer existed while the new one trained.

## spec

- A `Source` can say it is closed: the telemetry client's `Err(_)` other than `WouldBlock` sets a
  flag the model can read (keep what the tab already shows — the E4 intent stands).
- `Watch`: an attachment whose source is closed is no longer an attachment. The phases fall back
  to disk (`RunFacts`), exactly as for a run the editor never attached to — *interrupted* with its
  resume point, *done*, or *failed*.
- While the latest run is unfinished on disk and nothing is attached, `Watch` re-reads
  `telemetry.txt` and dials it again, at most once per `REDIAL_S` (10 s), off the UI thread like
  today's dial — so a run resumed elsewhere is picked up without re-opening the project.
- No change to what a run the editor started itself shows while its child lives.

## context

```
crates/es-editor/src/model/telemetry_view.rs
crates/es-editor/src/model/watch.rs
crates/es-editor/src/ui/train.rs
crates/es-editor/src/app.rs
docs/packets/M12/P-M12-R3-closed-telemetry.md
docs/packets/M12/P-M12-R3-closed-telemetry.ko.md
```

## oracle

1. Headless tests in `watch.rs`: an attached source that reports closed → the light is never
   *not responding*; the phases equal `phases(Some(&facts), None)`; a later `telemetry.txt` with a
   new address is dialled again after `REDIAL_S` and not before (inject the clock, as the existing
   tests do).
2. A test in `telemetry_view.rs` that a client whose peer closes reports closed.
3. `cargo test -p es-editor`, clippy `-D warnings`, fmt, `check-scope`, and
   `cargo xtask context-budget` — `es-editor` must stay under 10,000 (it is at 9,715).

## forbidden

Clearing what the tab shows when a connection drops; polling `telemetry.txt` on the UI thread;
any change to `es`; connecting to any remote server.
