# M10 R1 — two pre-existing failures under `ES_PYTHON`: a stale `Env::tick()` after a whole-batch reset, and a CRLF golden generator

Spec: §10.5 (`events.json` and the telemetry stream carry the episode-relative tick — packet
M7/R1), §23.1 (telemetry is a client of the run, never a second clock), §1.4 (golden files are
CI read-only; a generator that cannot reproduce them on one platform is the generator's bug),
§28.13 rule 1. Found during plan W's wave-0 CI (both fail at `c7e736e`, before plan W; neither ran
in earlier CI because both need `ES_PYTHON`). Diagnosis (read-only, 2026-09-22): see "the
question". Type B, small.

## the question

**(A)** `crates/es-compile/tests/gen_goldens.rs` `the_goldens_are_what_the_torch_oracle_produces`
fails on Windows: `crates/es-compile/python/gen_observation_goldens.py:76` writes each `.json`
sidecar with `Path.write_text(...)` and no `newline=`, so Python's text mode emits `\r\n` on
Windows while the committed goldens are LF (`.gitattributes` `eol=lf`); every `.bin` is
byte-identical, every sidecar differs by exactly one byte per line (439 vs 454 for
`crop_8x6_at_2_1_4x4.json`). torch 2.14.0 / torchvision 0.29.0 match the api-note pin. **Does
`newline="\n"` make the generator reproduce the committed goldens byte for byte on Windows too?**

**(B)** `crates/es/tests/cli.rs` `collect_telemetry_publishes_every_episode` (with
`--features render`): the `es loop collect --telemetry` child panics with `attempt to subtract
with overflow` at `crates/es/src/cmd/telemetry.rs:326`
(`record.tick.0 -= *self.episode_start.get_or_insert(tick.0)`). Cause: `Env::tick()`
(`crates/es-env/src/env.rs:187-189`) returns a field refreshed only in `Env::step()`
(`:287`); `Env::reset()` (`:204-261`) does not refresh it, and the MuJoCo backend's whole-batch
`reset(None)` (`crates/es-physics-backend/src/mujoco.rs:316-318`) zeroes its clock — the branch
`crates/es-data/src/collect.rs:674` takes when an episode ends by `--max-steps` rather than by a
termination. The next episode's first `Tick` therefore reports the previous episode's stale
cumulative tick, the publisher latches it as `episode_start`, and the second `Tick` (now small)
underflows. In release builds the subtraction wraps and a tick of ~1.8e19 is published on
stream 2 — a wrong value, not a flake. **If `Env::reset` resyncs the cached tick from the
backend, is `Env::tick()` correct for every caller, does the telemetry test pass, and does no
committed byte move?**

## spec

* (A) `gen_observation_goldens.py`: `write_text(..., encoding="utf-8", newline="\n")` (and any
  other text write in that generator). No golden changes.
* (B) `crates/es-env/src/env.rs` `Env::reset`: after `self.backend.reset(...)`, set
  `self.tick = self.backend.state().tick` (or the equivalent accessor the backend exposes), so
  the cache is never stale — the fix is in the shared function, not at the telemetry call site.
  `telemetry.rs:326` stays as it is (the subtraction is correct once the input is). A unit test
  in `es-env`: after `reset(None)` following steps, `env.tick()` equals the backend's clock (on
  the fixture backend, which must mirror the MuJoCo backend's whole-batch reset semantics — if
  the fixture backend does not zero its clock on `reset(None)`, say so and test through the
  fixture's own semantics plus the CLI test).
* Docs: one sentence in `docs/design/telemetry-protocol.md` (+ `.ko.md`) where the
  episode-relative rebase is described, naming the stale-cache cause and this packet; one
  sentence in `docs/design/observation-lowering.md` (+ `.ko.md`) or wherever the golden
  generator is described, naming the `newline` rule.

## context

```
crates/es-compile/python/gen_observation_goldens.py
crates/es-compile/tests/gen_goldens.rs
crates/es-env/src/env.rs
crates/es-env/tests/**
crates/es/tests/cli.rs
docs/design/telemetry-protocol.md
docs/design/telemetry-protocol.ko.md
docs/design/observation-lowering.md
docs/design/observation-lowering.ko.md
docs/packets/M10/P-M10-R1-stale-tick-and-crlf.md
docs/packets/M10/P-M10-R1-stale-tick-and-crlf.ko.md
```

## oracle

1. `ES_PYTHON=… cargo test -p es-compile the_goldens_are_what_the_torch_oracle_produces` green on
   Windows; `cargo xtask verify-goldens` — no golden moved.
2. `cargo test -p es-env <the new tick test>`; `ES_PYTHON=… cargo test -p es --features render
   --test cli collect_telemetry_publishes_every_episode` green (run it alone; it is
   load-sensitive when other builds run).
3. `cargo test --workspace` (with `ES_PYTHON`; the known `quadruped_eval_run_names_the_observation_gap`
   failure excepted), `cargo xtask verify-goldens`, `cargo xtask ci`,
   `cargo xtask check-scope docs/packets/M10/P-M10-R1-stale-tick-and-crlf.md`.
4. The demo trajectories and rollout golden unmoved (`tests/golden/rollout/so101_100steps.json`,
   the `.estraj` pins in `crates/es-eval/tests`): a reset-time resync cannot change a stepped
   value, and the test suite says so.

## acceptance

Oracles 1–4; the two doc sentences with their Korean siblings.

## forbidden

Editing any golden; touching `telemetry.rs`'s subtraction; changing `Env::step`, the backends,
`collect.rs`, or the dataset schema; `docs/ARCHITECTURE*.md`.
