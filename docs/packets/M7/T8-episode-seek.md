# M7 T8 — `Env::seek_episode`, and `--jobs` that splits episodes, not only cells

Spec: §10.4 (`--jobs N` cannot produce a report `--jobs 1` would not), §6.3 (randomization keyed
by `(seed, env, episode, stream)`), §3.5 tier 1 (bitwise), §28.9 ladder 10 ("`Env` episode seek —
the state after a seek is bit-identical to replaying 0..n"), §28.9 "where the wall-clock goes"
(episode-level sharding: "the episode counter blocks it; with a seek even a one-cell nominal run
parallelises"), §28.10 (T8). Design note to extend: `docs/design/evaluation-execution.md` (+ `.ko.md`)
— the sharding section; `docs/design/visible-learning.md` 7.11's "a shard is a partition of the
cells" paragraph gets a pointer. Depends on **T7** (the evaluator's latency model is the collector's).

## the question

`es eval run --jobs 6` splits the *cells* round-robin, so the six-suite sweep runs six-wide and the
one-cell nominal suite (16 episodes, ~5 min) runs one-wide. `Env::reset` draws episode `k`'s
randomization from `(seed, env, episode_counter, stream)` and nothing can set the counter but `k`
resets. **Can an `Env` be told to be at episode `k` and produce, from there, the bytes the
sequential run produces — and if so, which of the evaluator's per-cell state still stops a
`(cell, episode)` partition from being one?**

## spec

* **`Env::seek_episode(env: u32, episode: u64)`** sets the env's episode counter so that the *next*
  `reset` draws episode `episode`. It touches nothing else — no reset, no backend call — and it is
  refused (`EnvError`) while an episode is open with steps on it (`recorder.open(env).steps() > 0`),
  so a seek is never a silent discard of a recorded episode. `Env::new` draws episode 0 at
  construction as it does today, so the caller of a fresh `Env` that wants episode `k` calls
  `seek_episode(0, k)` then `reset(Some(&[0]))`.
* **The oracle at the `Env` level, first.** With the demo scene and the committed Task IR: a fresh
  `Env` reset `k` times versus a fresh `Env` seeked to `k` and reset once — for `k ∈ {1, 3, 7}`
  the `StateView` after reset (`qpos`, `qvel`), the episode's `ParamScales`, and the whole
  `.estraj` of an episode driven by the scripted expert are **byte-identical**. The MuJoCo backend's
  `reset` calls `mj_resetData` before setting the state (`python/mujoco_ref.py`), which is what
  makes this possible; the oracle is what proves it.
* **The evaluator's per-cell state, named.** `Evaluation::run_shard` keeps, per cell and across its
  episodes: the `Env`, the `SafetyPlane` (`begin_episode` clears the latch, **not** the
  `ViolationRate` window — `SafetyCounters::window` carries the tail of episode `k-1` into episode
  `k`), the `ChunkBuffer` + `PlaneFeed` (cleared per episode), `AsyncInference` (cleared per
  episode), the `seq` (monotonic per cell; only ordering is judged), and `env.metrics()` /
  `safety.counters()` (summed over the cell's episodes by `record_cell`). Of these, the **window** is
  the one whose carry-over is a behaviour, not a sum. This packet does not change it: whether
  `begin_episode` should clear the window is an `es-safety`/spec decision for the M7 review, and
  clearing it would move results on committed documents.
* **`(cell, episode)` partition, gated by parity.** `run_shard`'s unit becomes `(cell, episode)`
  round-robin over the flattened `suites × seeds` list; each unit builds its own `Env` (seeked),
  `SafetyPlane`, buffer, feed and inference; `ShardCell` gains per-episode `Episode`, `SafetyCounters`
  and `EnvMetrics` records and `record_cell` moves to `merge`, which sums them in episode order —
  so `report.json`, `events.json` and every `.estraj` are computed from the same per-episode records
  whichever worker produced them. **The partition is enabled only if oracle 3 holds on the committed
  documents**; if the window carry-over breaks it, the partition stays cell-level, the finding goes
  in the design note with the first differing cell/episode/tick, and the flag that would have
  enabled it does not exist (YAGNI — the M7 review decides).
* **Wall-clock.** The nominal suite (16 episodes) at `--jobs 1`, `--jobs 4`, `--jobs 8` on the
  server, before and after: an observation, `Target / Status: unverified` until measured.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
crates/es-env/src/env.rs
crates/es-env/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/src/metrics.rs
crates/es-eval/tests/**
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/T8-episode-seek.md
docs/packets/M7/T8-episode-seek.ko.md
```

`es-env/src/env.rs` (`seek_episode` and its refusal; nothing else in `es-env`), `es-env/tests`
(oracle 1), `es-eval/src/runner.rs` (the unit, `ShardCell`'s per-episode records, `record_cell` at
merge), `es-eval/src/metrics.rs` **only** if a per-episode sum needs a helper, `es-eval/tests`
(oracles 2–3), `es/src/cmd/eval.rs` (worker spawning: units instead of cells — the `--shard i/N`
protocol stays), `es/tests/cli.rs` (`eval_jobs_*` tests), the design notes, this packet.

## oracle

1. `cargo test -p es-env seek_is_the_replay` — the `Env`-level statement above, byte-identical,
   for `k ∈ {1, 3, 7}`, `.estraj` included (driven by `ScriptedExpert`; `SKIP` with reason without
   the MuJoCo backend).
2. `cargo test -p es-env seek_refuses_an_open_episode` — a seek after a step and before a reset is
   an `EnvError` naming the env and its step count.
3. `cargo test -p es-eval episode_shards_reproduce_the_sequential_run -- --ignored` — on the
   committed demo documents, the nominal suite's 4 episodes: `--jobs 1` versus the `(cell, episode)`
   partition over 2 and 4 workers, `report.json`, `events.json` and every `.estraj` byte-identical.
   Printed either way: `RAN … identical` or `RAN … first difference at <cell>/<episode>/<tick>`. This
   is the gate of the spec's fourth bullet. `SKIP` without `ES_PYTHON`.
4. `cargo test -p es --test cli eval_jobs_splits_episodes` — with the partition enabled: the
   fixture run with `--jobs 2` spawns two workers each holding one episode of the one-cell fixture
   (from the worker command lines the test reads back); with it disabled, the test asserts the
   cell-level behaviour and the design note's finding paragraph exists (grep).
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T8-episode-seek.md`.

## acceptance

Oracles 1–5 (3 on the server). The wall-clock table (nominal, 16 episodes, jobs 1/4/8) in the
design note. If the partition is gated off, the finding names the mechanism with evidence
(the tick and the watchdog event that differs), not a guess.

## forbidden

Changing `Env::reset`'s draw order or what `Env::new` does; `crates/es-safety/**` (the window
decision is the review's); `crates/es-physics-backend/**`; changing any metric's definition or
`record_cell`'s arithmetic (the sum must be the same sum, in the same order); a flag that enables
a partition the oracle did not prove; `docs/ARCHITECTURE*.md`; goldens. INV-12: no plane state is
skipped to make parity hold. INV-17: no new trait.
