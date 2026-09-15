# M5 V17 — the evaluator re-plans at the rate the Deployment IR declares

Design note: `docs/design/visible-learning.md` **section 7.25**, and 7.24 (V16, the release is
not a function of the observation), 7.13 (V6b, evaluation executes chunks the way collection
does), 7.19 (V11, one control step is one control period).
Spec: §9.2, §9.4, §8.5, §8.6, §12.1, App. B.5, §1.4. Depends on V6b, V11, V15, V16.

## the question

`tests/fixtures/visible-learning/deployment.toml` declares `rate.control = 50 Hz` and
`rate.inference = 5 Hz` — re-plan every ten control steps — and `action.execute_chunk = 10`
against a lowered chunk of `[10, 6]`. Neither loop read the second rate. `es_eval::runner`
called `infer_chunk` on every control tick and `es_env::DomainRunner` submitted an inference on
every control step (`BatchDomains::single_env_at` fires the inference domain once per control
period), so rows 1..9 of every chunk never executed: the temporal ensemble averaged row 0..7 of
the eight newest chunks and `HardSwitch` would have served row 0 of the newest.

V16 measured what that costs at the hold. Every input is frozen for about six control ticks
while the demonstrations' target ramps, so the per-tick re-plan keeps executing the conditional
median of a frozen input (+0.007 rad) and the jaw never leaves the cube — while the *same*
chunk's rows 7, 8 and 9 are 0.105, 0.150 and 0.203 rad and cross the 0.093 rad stall angle in
8 of the 8 carrying holds. Open question 22's default is (iii) and open question 23's decision
is this packet: honour the declared re-plan period, on both paths, and re-measure the existing
best checkpoint without retraining.

## deliverable

1. **One rule, both paths.** `es_env::replan_interval(rate)` is `rate.control /
   rate.inference` in whole control ticks, and refuses a rate that does not divide **by name**
   rather than rounding it. `es_eval::runner` infers on `step % replan == 0` and executes the
   chunk's successive rows in between through the same `es_env::plane_chunk` / `ChunkBuffer`
   path V6b routed it through; `DomainRunner::infer_window` gates submission on the same
   number, so `es loop collect` and `es eval run` execute chunks identically (V6b's contract).
   The buffer is kept, not special-cased: with a 10-row chunk at 5 Hz exactly one chunk is live
   and the ensemble degenerates to it, which is arithmetic rather than a branch. Inference
   latency modelling is untouched. `ExpertCfg::pace_to` takes the same number on both paths.
2. **The clock the plane's own period is in.** `SafetyPlane` converts a tick difference to
   microseconds with the Deployment IR's *control* period, but both loops handed it
   `Env::tick`, which counts **simulation** ticks — four per control step on the demo scene
   since V11. It could not show while a chunk arrived every control tick, because `accept`
   stamps `last_chunk_tick = now` before the deadline is measured and the gap was always zero.
   Honouring `rate.inference` makes the gap real, and the first measured run came back with
   `violation.inference_deadline` on 25,920 of 28,800 control ticks and the fallback latched
   for whole episodes. `DomainRunner::emit_actions` now uses its own `control_tick` and
   `es_eval::runner` the episode step. The Safety Plane itself is unchanged.
3. **The document says what it means.** A deployment that declares a 5 Hz re-plan has up to
   180 ms between chunk arrivals and cannot meet a 40 ms `inference_deadline`. The demo's
   `inference_budget` and its watchdog become 240 ms — the 200 ms the document asks for plus
   the 40 ms it already allowed for the inference itself — and `deadlines.observation_age`
   follows only because `DEP_021` refuses one below `inference_budget`. The enforced staleness
   bound (`stale_observation`, 80 ms), every safety limit and the 0.5 acceptance threshold are
   unchanged. `deployment_hash` moves, so V15's weights file is repacked into a bundle built
   from the current documents; the weights are the same file, byte for byte.
4. **Re-measure, no retraining.** V15's 40,000-step checkpoint at the 1,800-step budget on
   training seeds 1–16 and held-out 101–116 (twice), per episode lifted / carried / released /
   harness success, beside V15's and V16's rows.

## scope

* **`context`** — `crates/es-env/src/{domains,expert,env,lib}.rs`,
  `crates/es-eval/src/runner.rs`, `crates/es-data/src/collect.rs`, `crates/es/src/cmd/loop.rs`,
  `crates/es-eval/tests/evaluation.rs`, `crates/es/tests/cli.rs`,
  `tests/fixtures/visible-learning/deployment.toml`,
  `docs/design/visible-learning*.md` section 7.25 and open questions 22–23,
  `docs/packets/M5/V17-inference-rate*.md`.
* **`forbidden`** — every IR schema; `crates/es-safety`; the Deployment IR's safety limits and
  the acceptance threshold (0.5, unchanged); retraining; sections 7.21 – 7.24;
  `docs/ARCHITECTURE*.md`.
* **`INV-17`** — no new trait and no new extension point. One free function and one field.
* **`INV-12`** — nothing is disabled. Every tick still goes buffer → `plane_chunk` →
  `SafetyPlane::validate` → `ctrl`, and the watchdog that could not be met under the declared
  rate has its number stated for that rate rather than being removed.

## oracles

1. **`the_runner_infers_once_per_declared_replan_period`** (`crates/es-eval/tests/evaluation.rs`,
   every CI tier): a counting `PolicyRuntime` over a 100-control-step evaluation infers exactly
   **10** times at a 10:1 rate ratio and exactly **100** times at 1:1.
2. **`an_inference_rate_that_does_not_divide_the_control_rate_is_refused`** (same file): a
   non-dividing rate is an `EvalError::Env` naming `rate.control`, `rate.inference` and the
   fractional period.
3. **`collection_and_evaluation_ask_the_policy_at_the_same_cadence`** (`crates/es/tests/cli.rs`,
   needs `mujoco`): the same scripted expert, the same seed, 120 control ticks, and the policy
   invoked exactly `ceil(120 / replan)` times on **each** path. It also prints the first tick at
   which the two per-tick `qpos ‖ qvel` traces diverge as raw `f64` bits, and does not assert it
   away: `es loop collect` applies `RuntimeHints::expected_latency_ms` (15 ms, one control tick)
   through `AsyncInference` and `es_eval::runner` has no latency model, so the collector's first
   chunk lands at tick 1 and the evaluator's at tick 0.
4. **`expert_passes_the_evaluation_harness`** still ≥ 0.875 on the eight pinned seeds, with its
   `envelope_violation_rate` reported; `expert_solves_the_pinned_seeds`,
   `collection_and_evaluation_draw_the_same_scene_for_a_seed`, V9's
   `a_showcase_replay_reproduces_the_frames_the_policy_saw` and V10's
   `recorded_actions_replay_to_the_same_outcome` unchanged.
5. **`cargo xtask ci`** green — fmt, clippy `-D warnings`, context budget, layering, spec-refs,
   goldens.

## acceptance

The cadence oracles pass, the two paths ask the policy at the same cadence, the expert still
clears the harness, and V15's checkpoint is re-measured at the 1,800 budget on both suites and
reported beside V15 and V16 whether it improves or not. Held-out `success_rate` ≥ 0.5 continues
to the six-suite sweep and the showcase videos; below it the stop rule fires and the packet
reports the per-episode table, and — if releases now happen but success still fails — which
predicate term fails. `cargo xtask ci` is green.
