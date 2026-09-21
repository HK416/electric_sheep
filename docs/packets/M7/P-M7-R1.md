# M7 R1 — episode boundaries, and the `(cell, episode)` partition (T8b)

Spec: §9.4 (`begin_episode` empties the `ViolationRate` window — the sentence added 2026-09-21),
§10.5 (`events.json`'s `tick` is episode-relative — same commit), §10.4 (`--jobs N` cannot produce a
report `--jobs 1` would not), §13.1 (an episode is where a stream ends), §28.11 wave 0, §28.9 rule 2
(an invalidated measurement is marked, never deleted). Review: `docs/reviews/M7.md` S-1, S-2,
follow-up R1; the owner's decision is recorded in §28.11. Packet it completes: `T8-episode-seek.md`.
Design notes to extend: `docs/design/safety-plane.md` ("Counters"), `docs/design/evaluation-execution.md`
2.7 (rewritten: the unit *is* the `(cell, episode)`; the wall-clock table after), `docs/design/visible-learning.md`
7.33 (U3 re-measured under the new semantics). Korean siblings in the same commit.

## the question

T8 proved `Env::seek_episode` is the replay (bitwise for `k ∈ {1, 3, 7}`) and measured the
`(cell, episode)` partition at 2.0× / 2.2× (jobs 4 / 8) on the 16-episode nominal suite, and did not
ship it: `SafetyCounters::window` carries episode `k−1`'s tail into episode `k` (first difference
`nominal-01` tick 24), and `StepEvent::tick` is the cell's cumulative physics clock (`events.json`
`nominal-01` record 0: tick 7200). The owner decided both on 2026-09-21 (§28.11): the window empties
at `begin_episode`, and `tick` counts from the episode. **With both in, is the `(cell, episode)`
partition bitwise the sequential run on the committed documents, what does the nominal suite cost
now, and what does U3's held-out number become?**

## spec

* **`es-safety`, one line.** `SafetyPlane::begin_episode` also empties the `ViolationRate` ring
  (`self.counters.window.clear()`). Nothing else in `SafetyCounters` moves: `violations`, `steps`,
  `clamped_steps`, `dirty_steps`, `fallback_activations` keep summing across the cell, and
  `reset_counters` stays the only way to zero them. The doc comment says what §9.4 now says: the
  latch, the seed and the ring — the envelope, the watchdogs and the sums are untouched (INV-12), and
  `validate`'s signature is untouched (INV-13). The design note's "Counters" section records that
  the ring is per-episode and why (§10.3: a full window or nothing).
* **`es-eval`, the clock.** `StepEvent::tick` and `RunEvent::Observation { tick }` carry
  `env.tick() − start`, where `start` is `env.tick()` read when `run_episode` is entered (the env is
  freshly reset there, packet M5/V6b). `frame` is unchanged. `events.json`'s schema is unchanged;
  only the numbers after the first episode of a cell move.
* **The collector's stream-2 tick.** `es loop collect --telemetry` publishes `[frame, tick, source,
  bits]` too (packet M7/E7). If its `tick` is `env.tick()` (cumulative), make it episode-relative at the
  publish site, so a live viewer sees one clock on both stages; if it already is, say so in the note.
* **The partition (T8b), exactly as T8 specified it.** `Evaluation::run_shard_with_sink`'s unit is
  the `(cell, episode)` pair: units are `cell * n_episodes + episode` over the flattened
  `suites × seeds` list, unit `u` to shard `u % count`. Each unit builds its own `Env` (seeked to
  `episode`, then `reset(Some(&[0]))`), `SafetyPlane`, `ChunkBuffer` + `PlaneFeed`, `AsyncInference`.
  `ShardCell` gains per-episode records (`Episode`, the plane's `SafetyCounters` snapshot, `EnvMetrics`)
  and `record_cell` moves to `merge`, which sums them **in episode order** — so `report.json`,
  `events.json` and every `.estraj` are computed from the same per-episode records whichever worker
  produced them. **One code path**: `--jobs 1` is the same partition with `count = 1` (T8 measured the
  per-episode `--jobs 1` at +1 s, inside the noise), so the sequential run is not a second
  implementation. `merge` refuses a unit set that is not exactly every `(cell, episode)` once.
* **`es eval run`.** `--jobs` clamps to `suites × n_episodes` instead of `suites`; the
  `--shard i/N --shard-out` worker protocol stays; the help text's paragraph on "a one-suite
  evaluation gets no speedup" is replaced by what is now true. `--telemetry` still needs `--jobs 1`.
* **Fixtures, not goldens.** `tests/fixtures/visible-learning/run/events.json` and
  `tests/fixtures/visible-learning/events.json` are read by the editor and video tests. If their
  ticks move under the new clock, regenerate them through the command that made them and say so in
  the note; if they do not (their cells may be one episode long), say that instead. `tests/golden/**`
  is not touched.
* **Re-dating the committed numbers (§28.9 rule 2).** Every evaluation number in
  `visible-learning.md` up to 7.32 was measured under window carry-over. A new section 7.33 says so
  in one paragraph, and re-measures **U3** (`~/artifacts/plan-v/m7-u/U3`, the committed documents
  with `observation-augmented.toml`, held-out 16 seeds × 6 suites) with the new build: the row goes
  beside the old 0.5625, which stays and is marked "pre-R1 semantics". The expert gate's
  `success_rate` and `envelope_violation_rate` on the nominal suite are re-measured beside it, so
  the harness's own number under the new semantics is on record before any policy's.
* **The wall-clock.** The nominal suite (16 episodes, `--frames`, the bundle T8 used —
  `~/artifacts/plan-v/v14/trained-20000.esb`, or U3's if that path is gone; say which) at `--jobs 1`,
  `4`, `8`, two passes, the 1-minute load beside each row, in T8's table format; parity across the
  rows asserted per cell (`report.json`, `events.json`, `.estraj`; `evaluation.lock`'s `created`
  excepted). Reported with §12.4's nine metrics where the run measures them and
  `Target / Status: unverified` where it does not — never a `step/s`.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
crates/es-safety/src/plane.rs
crates/es-safety/src/counters.rs
crates/es-safety/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/src/metrics.rs
crates/es-eval/tests/**
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/telemetry.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/run/events.json
tests/fixtures/visible-learning/events.json
docs/design/safety-plane.md
docs/design/safety-plane.ko.md
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/P-M7-R1.md
docs/packets/M7/P-M7-R1.ko.md
```

`plane.rs` (`begin_episode`, one line and its comment), `counters.rs` (only if `window.clear()` needs
a `pub(crate)` path it does not have), `es-safety/tests` (the scenario suite gains the boundary
case), `runner.rs` (the clock, the unit, `ShardCell`'s per-episode records, `record_cell` at merge),
`metrics.rs` **only** for a per-episode sum helper, `es-eval/tests` (oracles 2–3), `eval.rs` (the
clamp, the help), `loop.rs` / `telemetry.rs` (the collector's tick, only if cumulative), `cli.rs`
(`eval_jobs_*`), the two fixtures (only if they move), the notes, this packet.

## oracle

1. `cargo test -p es-safety window_is_cleared_at_begin_episode` — fill the ring until
   `envelope_violation_rate()` reads above `0`, call `begin_episode`, assert `0.0`; assert every
   summed counter is unchanged and the latch is clear. A second test in the scenario suite: a
   plane that tripped `ViolationRate` in episode `k−1` does not trip on tick 0 of episode `k`.
2. `cargo test -p es-eval tick_is_episode_relative` — a two-episode cell on the fixture backend:
   episode 1's first `StepEvent.tick` is `0` and its tick sequence equals episode 0's; `frame`
   sequences unchanged.
3. `cargo test -p es-eval episode_shards_reproduce_the_sequential_run -- --ignored` — T8's oracle
   3, now an **assertion**: on the committed demo documents, the nominal suite's 4 episodes,
   `count = 1` versus the partition over 2 and 4 workers — `report.json`, `events.json`, every
   `.estraj` byte-identical; prints `RAN … identical`. `SKIP` without `ES_PYTHON`.
4. `cargo test -p es --test cli eval_jobs_splits_episodes` — `--jobs 2` on the one-cell fixture
   spawns two workers each holding one episode (read back from the worker command lines);
   `--jobs 0` and the `--shard` usage refusals unchanged.
5. Server: the nominal suite, jobs 1 / 4 / 8, two passes — per-cell artifacts bitwise across the
   rows (`created` excepted), and **jobs 4 wall-clock ≤ 55 % of jobs 1**. The table in
   `evaluation-execution.md` 2.7.
6. Server: U3 held-out and the nominal expert gate re-measured; `visible-learning.md` 7.33 with
   both rows and the "pre-R1 semantics" mark on the old ones.
7. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/P-M7-R1.md`.

## acceptance

Oracles 1–7 (3, 5, 6 on the server: `~/Projects/es-r1-episodes` from a tarball of the tree,
`ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`, artifacts under `~/artifacts/plan-v/m7-r1-episodes/`;
the box is shared — wait for a 1-minute load below 4 before a timed row, as T8 did). The partition
ships only on identity: if oracle 3 finds a difference, the finding (cell / episode / tick and the
watchdog event that differs) goes in the design note and the packet stops there. The three notes and
their Korean siblings updated; the packet's `.ko.md` sibling present.

## forbidden

`SafetyPlane::validate`'s signature (INV-13); skipping any plane state or zeroing anything beyond the
ring at `begin_episode` (INV-12 — the sums are the cell's); `es-safety` learning about episodes from
anywhere but its own `begin_episode` (INV-11, rule 8); `Env::reset`'s draw order or what `Env::new`
does; `record_cell`'s arithmetic (the same sum in the same order); any metric's definition;
`docs/ARCHITECTURE*.md` (the sentences are already there); `tests/golden/**`; the telemetry protocol
(`crates/es-telemetry/**`); the editor; a flag that keeps the cell-level partition alive beside the
new one (one code path). INV-17: no new trait.
