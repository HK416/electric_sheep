# Domain-gap diagnostics (`es-eval::domain_gap`) — design

Spec refs: §24.3 (domain-gap diagnostics: what is compared, why policy-response gap is the
most useful signal), §18.3 (sensor realism knobs a diagnosed gap should point at), §10.3
(the `domain_gap` metric slot in the §10 metric set), §4.2 (crate layering), §1.2 (work
packets), §28.5 M3 W4, §28.7 gate 15 ("domain gap report generated").

Review class: C — read this before the code.

## 1. What this is, and the one deviation from the obvious signature

§24.3 describes `es domain-gap --real <rosbag|dataset> --sim <scene+task+obs>`: replay a
real-robot log into the sim and compare observation distributions. This crate has no
rosbag replay yet (that is HIL infrastructure, §24.2, not this packet), so W4 diagnoses the
gap between two already-captured `LeRobot` datasets — one from sim rollouts, one from a real
robot — which is the same comparison one level downstream: distributions of observation
channels, action channels, and episode outcomes, plus latency if the real side logged it.

**Deviation:** the obvious signature is `DomainGap::compute(sim: &LeRobotDataset, real:
&LeRobotDataset, opts) -> GapReport`. This crate cannot take that signature: `es-data` and
`es-eval` are both layer 10 (§4.2), and rule 1 forbids same-layer dependencies —
`cargo xtask layering` enforces this on every dependency edge, dev-dependencies included, so
even a test-only `es-data` import from `es-eval` fails CI. `es-eval::domain_gap` therefore
takes a small dataset-agnostic input type ([`GapInput`](#3-types)) built from plain
`Vec<f64>` samples, not a `LeRobotDataset`. The `es` binary (unconstrained by the layer
table — it is not an `es-*` crate) is the only thing that imports both `es-data` and
`es-eval::domain_gap`, and `crates/es/src/cmd/gap.rs` is where `LeRobotDataset` gets turned
into a `GapInput`. This keeps the same division of labour as everywhere else in the repo:
the IR/runtime crate is the hashable, dependency-light contract; the CLI is the glue.

## 2. Metric set (§24.3 comparison table, §10.3 `domain_gap`)

Per numeric channel (an observation feature dim, an action dim, or a `latency.*` series —
see §4):

- `mean`, `std` on each side, and their difference (`real - sim`).
- **KS statistic `D`**: the two-sample Kolmogorov-Smirnov statistic, `sup_x |F_sim(x) -
  F_real(x)|`, computed by a sorted merge — no stats crate (§1 toolchain minimalism). The
  merge advances **both** cursors past a value's whole run of repeats before it evaluates the
  gap (standard `ks_2samp` semantics): an empirical CDF is a right-continuous step function,
  so a point in the middle of a tie run is on neither CDF and the difference measured there is
  not a `D`. Robot data makes that the common case, not a corner one — binary flags, quantised
  encoder counts, and the per-dataset stride in §5 leaves the two sample counts unequal — and
  measuring mid-tie reports e.g. `0.333` for two *identical* binary channels at 300 vs 100
  samples, over the default `0.3` threshold. Unit tests pin `D = 0` for a repeated single
  value, a binary channel and a 10-level quantised channel, all at unequal `n`.
- **Wasserstein-1** (earth mover's distance for 1-D samples): the integral of `|F_sim(x) -
  F_real(x)|` over the support, computed from the sorted samples — also no stats crate.
- A small quantile table (p10/p50/p90) on each side, nearest-rank (no interpolation, same
  convention as `es_eval::metrics::aggregate`'s `P95`), so the table is exactly reproducible.

Episode-level (§10.3 `success_rate`, `episode_length`, `envelope_violation_rate`):

- `success_rate`, mean episode length, mean envelope-violation rate, each side, each an
  `es_ir::evaluation::MetricValue` — `Unavailable` with a reason when the dataset carries no
  column for it, never a fabricated `0.0` (the same rule `es-eval::metrics` already follows).

Latency (§10.3 `p50/p95_latency`, §24.2 deadline/jitter): if the caller supplies
`latency.obs_age_ms` / `latency.inference_ms` / `latency.actuation_ms` channels (read from
`es_telemetry::protocol::PerfMetrics` samples logged alongside the real run), they go through
the same per-channel path as any other numeric channel — a latency gap is a distribution gap
like any other, and p50/p95 fall out of the quantile table.

## 3. Types

```
GapOptions { max_samples_per_feature: usize, threshold: f64 }   // KS D threshold, default 0.3
FeatureSamples { dims: usize, values: Vec<f64>, nonfinite_dropped: usize }  // row-major, n*dims
EpisodeSummary { success: Option<bool>, length: u64, envelope_violation: Option<f64> }
GapInput { channels: BTreeMap<String, FeatureSamples>, episodes: Vec<EpisodeSummary> }

ChannelGap { name, sim_n, real_n, sim_mean, real_mean, sim_std, real_std, mean_diff,
             ks_d, wasserstein1, sim_quantiles: [f64; 3], real_quantiles: [f64; 3],
             sim_nonfinite_dropped, real_nonfinite_dropped, flagged }
DimsMismatch { channel: String, sim_dims: usize, real_dims: usize }
EpisodeGap { sim_success_rate, real_success_rate, sim_length_mean, real_length_mean,
             sim_envelope_violation_rate, real_envelope_violation_rate: MetricValue }
Suspect { channel: String, ks_d: f64, knob: &'static str }
GapReport { threshold, channels: Vec<ChannelGap>, unmatched_sim: Vec<String>,
            unmatched_real: Vec<String>, dims_mismatch: Vec<DimsMismatch>,
            episodes: EpisodeGap, suspects: Vec<Suspect> }
```

`BTreeMap` only (never `HashMap`, §18.4 determinism convention extended here for the same
reason: iteration order must not depend on hasher state); everything is walked in sorted
key order so two runs over the same inputs byte-for-byte agree.

## 4. Pairing rule

`GapInput.channels` keys are already namespaced by the caller: an observation feature
`observation.state` with 6 dims becomes report rows `observation.state[0]` ..
`observation.state[5]`; an action feature likewise; a latency series is `latency.obs_age_ms`
etc. Pairing is by exact key match between `sim.channels` and `real.channels` (a
`BTreeMap` intersection in sorted order); a name present on only one side is not compared —
it is listed in `unmatched_sim` / `unmatched_real` instead, since a feature that does not
exist on both sides is a schema difference, not a distribution gap, and inventing a distance
against nothing would be exactly the "fabricated `0.0`" this codebase forbids.

A name that *is* on both sides but with a different `dims` is the same kind of schema
difference: it goes in `dims_mismatch` and is not scored. Comparing dim `d` of a 6-wide sim
feature against dim `d` of a 7-wide real one (or, worse, clamping the sim index to its last
column) pairs two different physical quantities and reports a `D` for them.

## 5. Subsampling (bounded, deterministic)

`GapInput` construction (in `crates/es/src/cmd/gap.rs`) reads every episode's frames but
keeps at most `max_samples_per_feature` samples per channel: given `n` total frames and cap
`m`, stride = `ceil(n / m)`, keep frame indices `0, stride, 2*stride, ...` in dataset order
(episodes sorted by index, frames in episode order). No RNG, no reservoir sampling with
random replacement — a fixed stride by position is deterministic and, since frames within an
episode are already temporally correlated, no less representative than a random draw for the
per-channel marginal statistics this report computes.

Two things a dataset written by something other than `LeRobotWriter` can be, both handled
here rather than in `es-data`:

- **A non-finite value.** A kept frame with a `NaN`/`Inf` in any of its dims is dropped and
  counted in `FeatureSamples::nonfinite_dropped`, surfaced per channel and side in
  `ChannelGap`. Neither KS nor Wasserstein-1 has a meaning over `NaN`, and `serde_json` has no
  encoding for one — so an unfiltered sample used to take `gap_report.json` down *after* the
  table had been printed. `GapReport::to_json` returns `Result` for the same reason: the last
  step of a command that already produced output must not be a panic.
- **A column shorter than `n * dims`.** `dims` comes from the feature's declared
  `elem_count`; a column that does not hold that many values per frame is an inconsistent
  dataset, so the row slice is a `get(..)` and the miss is a `DataError::Inconsistent`
  (`CliError::Runtime`, exit 1), not an index panic.

## 6. Suspects: mapping a flagged channel to a §18.3 knob

A channel is "flagged" when `ks_d > threshold` (default `0.3`, `--threshold` on the CLI).
Each flagged channel is mapped to a §18.3 sensor-realism knob by a name-prefix heuristic —
deliberately a lookup table, not inference over the actual pixel/signal content:

| channel name contains | §18.3 knob to check |
|---|---|
| `image`, `cam`, `rgb`, `depth` | exposure, white balance, lens distortion, rolling shutter |
| `vel`, `velocity` | IMU/joint-encoder noise model, latency |
| `force`, `torque`, `.ft`, `wrench` | F/T sensor Gaussian noise, temperature drift |
| `state`, `joint`, `position`, `qpos` | joint encoder quantization, delay, offset |
| `action` | actuator/controller latency, action chunk delay |
| `latency.` | deadline/jitter budget (§24.2), not a sensor knob |
| (none of the above) | generic sensor noise model (no specific §18.3 knob) |

This is intentionally coarse (ponytail: a keyword table beats a classifier here — the report
is a pointer for a human to go look, not a diagnosis) and lives as a `const` slice in
`domain_gap.rs`, not a second file.

## 7. Report schema

`GapReport::to_json(&self) -> Result<String, serde_json::Error>` is
`serde_json::to_string_pretty` (the struct already derives `Serialize`); the file is written
by the CLI as `gap_report.json`. `GapReport` implements `Display` for the plain-text table the
CLI also prints to stdout: one row per channel (`name`, `n` each side, `mean`/`std` each side,
`KS D`, `W1`, flagged marker), followed by the unmatched-feature lists, the `dims_mismatch`
list, the non-finite drop count when there is one, the episode-level block, and the suspects
list.

## 8. CLI

```
es gap --sim <root> --real <root> [--out gap_report.json] [--max-samples N] [--threshold D]
```

Opens both `LeRobotDataset`s, builds a `GapInput` per side (§5), calls
`domain_gap::DomainGap::compute`, prints the `Display` table, writes `--out` (default
`gap_report.json`) via `to_json`. Exit code 1 if any channel is flagged (`ks_d >
threshold`), 0 otherwise; usage errors are 2 (the convention every other `es` subcommand
uses, `crates/es/src/error.rs`).
