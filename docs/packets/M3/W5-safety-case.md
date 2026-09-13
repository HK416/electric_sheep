# W5-safety-case — traceability graph, `evidence.esb`, `es evidence verify`

Spec: §27.1 (Provenance Bundle + Safety Case, `evidence.esb` layout, `traceability.json`,
`revalidation_trigger`), §5.3 (hash chain, `HashChain::diff`), §10.5 (`report.json`,
`evaluation.lock`), §25.1 (trust boundaries, the unimplemented signature), §25.2 (the legal
checklist this bundle feeds), §28.5 M3 W5, §28.7 gate 16 (Safety Case traceability
completeness) and gate 17 (evidence bundle round-trip, M4).

Design note: `docs/design/safety-case.md` (read it first — it carries the completeness rule,
the `revalidation_trigger` table and the list of what verification does *not* prove).

## context

```
crates/es-compile/src/bundle.rs     (+ the Evidence-kind entry name constants only)
crates/es-eval/src/evidence.rs      (new)
crates/es-eval/src/lib.rs           (+ `pub mod evidence;`)
crates/es/src/cmd/evidence.rs       (new)
crates/es/src/cmd/mod.rs            (+ `pub mod evidence;`)
crates/es/src/main.rs               (+ one dispatch arm and the usage line)
crates/es/tests/cli.rs              (append tests; reuses the existing cross-IR `Fixture`)
docs/design/safety-case.md          (new)
docs/packets/M3/W5-safety-case.md   (this file)
```

## spec

1. `es_eval::evidence` carries the graph types (`Requirement`, `Claim`, `Evidence`,
   `EvidenceKind`, `SafetyCase` with `traceability: BTreeMap<String, Vec<String>>`), all
   `serde`, no new traits (INV-17), `BTreeMap` only (§3.4: no `HashMap` iteration order).
2. `SafetyCase::validate() -> Vec<Diagnostic>` rejects duplicate ids, dangling requirement or
   evidence references, and a requirement with no evidence.
3. `EvidenceBundle::build(policy_bundle, chain, reports, case, extra)` writes an `.esb` of
   `BundleKind::Evidence` holding `chain.json`, `safety_case/case.json`,
   `reports/<i>/{report.json,evaluation.lock}` and the policy bundle embedded whole at
   `policy/policy.esb`; the manifest carries the policy bundle's hash slots plus `runtime` and
   `dataset` from the chain. An invalid case is refused.
4. `EvidenceBundle::verify(bytes, against)` returns a `VerifyReport`: every entry hash (the
   container checks its own, `verify` checks each `Evidence.hash` against the entry it names),
   `chain.execution_hash()` against every report's `execution_hash`, chain against the embedded
   policy bundle's manifest slot by slot (`learning`/`policy` mismatches are warnings — see the
   design note §3), per-requirement coverage (gate 16), and, when `against` is given, the
   `HashChain::diff` components paired with the evidence they invalidate.
5. `es evidence verify <bundle.esb> [--against <other.esb>] [--json]` prints the table and exits
   0 only when every requirement is covered and no error diagnostic was produced, 1 otherwise.
   `es evidence build --policy <policy.esb> --chain <chain.json> --report <dir>... --case
   <case.json> --out <evidence.esb>`.
6. `--against` is advisory and never changes the exit code. Nothing claims the bundle is signed:
   `verify` prints `signature: unverified` (§25.1).

## oracle

```
cargo fmt --check
cargo clippy -p es-compile -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-compile -p es-eval -p es
cargo xtask layering
cargo xtask check-spec-refs
```

The pass/fail content is in `crates/es/tests/cli.rs`:

- `evidence_bundle_round_trips` — build then verify is green, every requirement covered.
- `evidence_verify_catches_a_tampered_report` — a report entry rewritten (container rebuilt, so
  the container's own hash check passes) is caught by the `Evidence.hash` check.
- `evidence_verify_fails_an_uncovered_requirement` — one traceability link removed, exit 1.
- `evidence_verify_against_lists_revalidation` — a second bundle built from a deployment with a
  tighter `contact_force_max`: `Deployment` appears in the diff with the evidence that needs
  re-running.
- `evidence_verify_cli_round_trip` — `es evidence build` then `es evidence verify` on the
  bundle built in-test from the cross-IR fixture; `--json` parses.

Plus unit tests in `crates/es-eval/src/evidence.rs` for `SafetyCase::validate` and the
`revalidation_trigger` table.

## acceptance

- Gate 16: a requirement without matching-`execution_hash` evidence fails verification, and a
  bundle whose graph is complete passes.
- No new trait, no new dependency, no `HashMap`.
- `docs/design/safety-case.md` states the completeness rule, the trigger table and the
  limitations (no signature, no replay, no conformity claim) in §27.1's own terms.

## forbidden

- `crates/es-safety`, `crates/es-runtime-embedded`, `crates/es-data`, `crates/es-splat`,
  `crates/es-editor`, `crates/es-eval/src/domain_gap.rs`, `crates/es/src/cmd/loop.rs`,
  `crates/es/src/cmd/gap.rs` — neighbouring M3 packets own these.
- The root `Cargo.toml`, the container format itself (`write`/`read` in `bundle.rs`), and
  `es_eval::runner` (the reports are consumed as they are written today).
- Replay re-execution and signature verification: M4 (§28.6), gate 17.

## follow-ups

- The `EVID-0xx` diagnostic codes are not in the `es-ir-types::codes` dictionary (that crate is
  outside this packet's scope), so they render with an empty title. Register them in the M4
  dictionary pass.
- `es evidence build` does not fill in the evidence hashes for the author; a `case.json` must
  state them and `build` refuses the ones that do not match. An `es evidence link` helper that
  proposes them would make authoring bearable — deliberately not built here, because a tool that
  writes the hashes it later checks verifies nothing.
