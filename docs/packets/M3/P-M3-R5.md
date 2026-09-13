# P-M3-R5 — a stale lock is not coverage, and an empty case is not complete

Follow-up to the M3 review should-fixes (`docs/reviews/M3.md`: `evidence.rs:456`,
`evidence.rs:612`).

Spec: §27.1 (Provenance Bundle + Safety Case, `traceability`, `revalidation_trigger`),
§10.5 (`report.json` / `evaluation.lock`), §5.3 (hash chain), §28.7 gate 16,
§1.2 (work packets), §1.4 (oracle first).

Design note: `docs/design/safety-case.md` §4.

## context

```
crates/es-eval/src/evidence.rs   (SafetyCase::validate, EvidenceBundle::verify)
crates/es/tests/cli.rs           (append tests)
docs/design/safety-case.md       (§4)
docs/packets/M3/P-M3-R5.md       (this file)
```

## spec

1. `EvidenceBundle::verify` walks every `reports/<i>/` entry, not only `report.json`:
   `evaluation.lock` is parsed as an `EvaluationLock` and its `execution_hash` (hex) is
   compared with `chain.execution_hash()`, its `evaluation_hash` with `chain.evaluation` when
   the chain fills that slot. A mismatch on either is an `EVID-006` diagnostic.
2. An entry whose own contents name a different run does not count as coverage. `verify`
   collects those entry names and the per-requirement coverage pass puts evidence pointing at
   one into `stale`. This closes the hole for both spec 10.5 artifacts at once: previously a
   report of another run raised a diagnostic but still marked its requirement `covered`, and a
   *lock* of another run raised nothing at all, because the only `execution_hash` consulted
   was the one the case wrote down about itself.
3. `SafetyCase::validate` reports `EVID-008` for a case with no requirements. `VerifyReport::
   ok()` is "every requirement is covered", which is vacuously true over an empty table, so
   the empty table itself has to be the finding. `EvidenceBundle::build` refuses it for the
   same reason it refuses any invalid case.
4. No new type, no signature change on `verify`, no new dependency; `BTreeSet` for the
   foreign-entry set (§18.4, no `HashMap`).

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es --all-targets -- -D warnings
cargo test -p es-eval -p es
cargo xtask check-spec-refs
```

## acceptance

- `evidence_verify_treats_a_foreign_lock_as_stale`: a bundle whose `reports/0/evaluation.lock`
  is the lock of a *different* run, with `safety_case/case.json` rewritten to record that
  file's blake3 so the entry-hash check passes — the lock's requirement lands in `stale`, is
  not `covered`, `verify` is not ok, and an `EVID-006` diagnostic names the entry.
- `evidence_verify_fails_a_case_with_no_requirements` / `a_case_with_no_requirements_is_a_
  diagnostic`: an empty `requirements` list produces exactly `EVID-008`, `VerifyReport::ok()`
  is false over an empty coverage table, and `es evidence verify` exits 1.
- The existing evidence tests are unchanged: round trip, tampered report (still one covered
  requirement), uncovered requirement, `--against` revalidation.

## forbidden

- Editing `crates/es/src/cmd/evidence.rs`, `crates/es-compile/**`, `crates/es-data/**`,
  `crates/es-splat/**`, `xtask/**`, `.github/**` — other in-flight packets own these.
- Making `--against` affect the exit code (`docs/design/safety-case.md` §5: it is advisory).
- Claiming more than the bundle proves: there is still no signature (§25.1) and no replay
  (§28.6, gate 17), and §6 of the design note says so.
- A new extension-point trait (INV-17) or a `HashMap` anywhere in `evidence.rs` (§18.4).
