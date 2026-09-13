# W4-domain-gap — domain-gap diagnostics: metrics + report

Spec: §24.3 (domain-gap diagnostics: observation/action distribution, latency/jitter,
contact/force, success-rate gap; report format), §18.3 (sensor realism knobs a diagnosed gap
should point at), §10.3 (`domain_gap` in the §10 metric set), §4.2 (crate layering — the
reason this crate takes a `GapInput`, not a `LeRobotDataset`, see the design doc), §1.2 (work
packets), §28.5 M3 W4, §28.7 gate 15 ("domain gap report generated").

Design note: `docs/design/domain-gap.md` (read it first — it carries the metric set, the
pairing rule, the subsampling rule, and the suspects-to-knob table).

## context

```
crates/es-eval/src/domain_gap.rs   (new)
crates/es-eval/src/lib.rs          (+ `pub mod domain_gap;`)
crates/es-eval/tests/domain_gap.rs (new)
crates/es/src/cmd/gap.rs           (new)
crates/es/src/cmd/mod.rs           (+ `pub mod gap;`)
crates/es/src/main.rs              (+ one dispatch arm and the usage line)
crates/es/tests/cli.rs             (append tests)
docs/design/domain-gap.md          (new)
docs/packets/M3/W4-domain-gap.md   (this file)
```

## spec

1. `es_eval::domain_gap` carries `GapOptions`, `FeatureSamples`, `EpisodeSummary`, `GapInput`,
   `ChannelGap`, `EpisodeGap`, `Suspect`, `GapReport`, `GapError`, and the single-item
   `DomainGap::compute(sim: &GapInput, real: &GapInput, opts: &GapOptions) ->
   Result<GapReport, GapError>` (no new extension-point trait, INV-17; `BTreeMap` only). The
   deviation from `DomainGap::compute(sim: &LeRobotDataset, ...)` is deliberate: `es-data` and
   `es-eval` are both spec-4.2 layer 10, so a same-layer dependency (`cargo xtask layering`,
   which checks dev-dependencies too) is not available even for the test fixtures — the
   design doc explains this in §1.
2. Per numeric channel: mean/std on each side and their difference, the two-sample KS
   statistic `D` (sorted-merge, no stats crate), Wasserstein-1 (sorted-merge, no stats crate),
   and a p10/p50/p90 quantile table (nearest-rank, same convention as
   `es_eval::metrics::aggregate`'s `P95`). Channels are paired by exact name match in sorted
   order (`BTreeMap` merge-join); a name present on only one side is listed in
   `unmatched_sim` / `unmatched_real`, never scored against nothing.
3. Episode-level: `success_rate`, mean `episode_length`, mean `envelope_violation_rate`, each
   side, each an `es_ir::evaluation::MetricValue` — `Unavailable` with a reason when
   `EpisodeSummary` carries `None` for every episode (no fabricated `0.0`, same rule as
   `es_eval::metrics`).
4. A channel is "flagged" when `ks_d > opts.threshold`; every flagged channel appears in
   `GapReport.suspects` mapped to a §18.3 knob by the name-prefix table in the design doc §6.
5. `GapReport::to_json` (`serde_json::to_string_pretty`) and `impl Display for GapReport` (a
   plain-text table) per the design doc §7.
6. `es gap --sim <root> --real <root> [--out gap_report.json] [--max-samples N] [--threshold
   D]` (`crates/es/src/cmd/gap.rs`): opens both `LeRobotDataset`s, builds one `GapInput` per
   side by reading every episode and keeping every numeric non-reserved feature (`success` and
   `action_source` columns feed the episode block instead — see the design doc's episode-level
   convention), deterministically subsampled to `--max-samples` (default 4096) per channel by
   fixed stride (design doc §5, no RNG). Prints the `Display` table, writes `--out` (default
   `./gap_report.json`). Exit 1 when any channel is flagged, 0 otherwise; usage errors are 2.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-eval -p es
cargo xtask check-spec-refs
```

## acceptance

- `DomainGap::compute` on two identical `GapInput`s: every channel's `ks_d` is ~0, none
  flagged.
- One feature given a large mean shift on one side only: that channel (and only that one) is
  flagged and appears in `suspects`; unrelated channels are not.
- A feature present on only one side is listed in `unmatched_sim`/`unmatched_real`, not
  scored.
- KS `D` and Wasserstein-1 match hand-computed values on small fixed samples (unit tests in
  `domain_gap.rs`).
- `es gap` on two on-disk `LeRobot` fixture datasets (`LeRobotWriter`, `crates/es/tests/cli.rs`)
  prints a table, writes `gap_report.json` that round-trips through `GapReport`'s `Deserialize`,
  and exits 1 exactly when a shifted feature crosses `--threshold`.
- `cargo xtask check-spec-refs` resolves every `§`/`spec N.M` reference this packet's files add.

## forbidden

- Editing `crates/es-eval/src/evidence.rs`, `crates/es-data/**`, `crates/es-safety/**`,
  `crates/es-runtime-embedded/**`, `crates/es-compile/**`, `crates/es-splat/**`,
  `crates/es-editor/**`, `crates/es/src/cmd/evidence.rs`, `crates/es/src/cmd/loop.rs` — other
  in-flight packets own these.
- Adding `es-data` or `es-telemetry` as a dependency (of any kind, including
  dev-dependencies) of `es-eval`: same-layer, rejected by `cargo xtask layering` (spec 4.2
  rule 1).
- A new extension-point trait (INV-17), a `HashMap` anywhere in the report path (spec 18.4),
  or a fabricated episode-level number where the dataset carries no column for it.
- Wiring `es_telemetry::protocol::PerfMetrics` sample ingestion into the CLI: no `--latency`
  flag was requested for this packet, so `latency.*` channels are a capability of `GapInput`
  a future caller may populate, not something `es gap` reads yet.
