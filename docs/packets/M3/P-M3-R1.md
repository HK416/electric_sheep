# P-M3-R1 — the two-sample KS statistic advances past ties

Follow-up to the M3 review blocker (`docs/reviews/M3.md`, `[BLOCK] domain_gap.rs:444`).

Spec: §24.3 (domain-gap diagnostics), §10.3 (`domain_gap` in the §10 metric set), §28.7 gate
15 ("domain gap report generated"), §1.2 (work packets), §1.4 (oracle first).

Design note: `docs/design/domain-gap.md` §2.

## context

```
crates/es-eval/src/domain_gap.rs   (ks_statistic + its unit tests)
docs/design/domain-gap.md          (§2)
docs/packets/M3/P-M3-R1.md         (this file)
```

## spec

1. `ks_statistic` computes `D = sup_x |F_sim(x) - F_real(x)|` over the two *empirical* CDFs,
   which are right-continuous step functions: each step of the sorted merge picks
   `x = min(sim[i], real[j])` and advances **both** cursors past every sample equal to `x`
   before evaluating `|F_sim - F_real|` (standard `ks_2samp` semantics). The previous loop
   advanced one repeat at a time, so it measured the gap at points on neither CDF.
2. The comparison uses `f64::total_cmp` — the same order the two sorts used — so the merge
   cannot disagree with the sort on `-0.0`/`NaN` and always makes progress.
3. No signature change, no new type, no new dependency. `wasserstein1` is already tie-correct
   (it dedups merged breakpoints) and is not touched.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-eval -p es
cargo xtask check-spec-refs
```

## acceptance

- `ks_statistic(&[1.0, 1.0, 1.0], &[1.0]) == 0.0`, and the same with the arguments swapped
  (was `0.667`).
- An identical binary channel — 40% zeros on both sides — at 300 vs 100 samples scores
  exactly `0.0` (was `0.333`, over the default `0.3` threshold).
- An identical 10-level quantised channel at 500 vs 100 samples scores exactly `0.0`.
- Ties do not suppress a real gap: `{0,0,0,0}` vs `{1,1}` is `1.0`, `{0,0,1,1}` vs `{1}` is
  `0.5`.
- The hand-computed cases already in the file are unchanged: `[0,1]` vs `[2,3]` is `1.0`,
  `v` vs `v` is `0.0`, `{0,1,2,3}` vs `{2,3,4,5}` is `0.5`.

## forbidden

- Adding a stats crate (§1 toolchain minimalism) or changing `ks_statistic`'s signature.
- Touching `crates/es-splat/**`, `crates/es-data/**`, `xtask/**`, `.github/**` — other
  in-flight packets own these.
- A new extension-point trait (INV-17) or a `HashMap` in the report path (§18.4).
