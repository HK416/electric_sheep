# M4 W7 — `es evidence verify` completion: signing, replay plan, schema v2

Design note: `docs/design/safety-case.md` (§6 "What verification does not prove" is updated by
this packet — read it first, it states exactly what `Valid`/`replayable` do and do not mean).

## context

- `crates/es-eval/src/evidence.rs` (+ tests)
- `crates/es-eval/Cargo.toml` (add `ed25519-dalek`)
- `crates/es-compile/src/bundle.rs` (`BundleManifest.signer_public_key` companion field to the
  existing `signature` slot; `BUNDLE_SCHEMA_VERSION` 1 -> 2)
- `crates/es/src/cmd/evidence.rs` (+ `keygen`/`sign`/`replay` subcommands, `verify
  --trust/--require-signature`)
- `crates/es/Cargo.toml` (same `ed25519-dalek` pin, needed by the CLI's key handling)
- `crates/es/tests/cli.rs` (append only)
- `.gitignore` (`*.eskey`)
- `docs/design/safety-case.md` (§6 update)
- `docs/packets/M4/W7-evidence-verify.md` (this file)

## spec

- §27.1 (evidence bundle, `traceability.json`, `revalidation_trigger`) and §25.1 (security:
  weights and graph files cross a trust boundary; keys never live in the repo) together define
  what M3 W5 left as a reserved slot: a real signature over the container.
- §25.3 (versioning: bundle format is versioned, old versions stay readable) is the rule that
  makes `schema_version` 1 -> 2 non-breaking: a v1 manifest simply has neither signature field,
  which verifies identically to an unsigned v2 one (`signature: Absent`).
- §28.6 (M4 scope line: "`es evidence verify` 완성") and §28.7 gate 17 ("evidence bundle
  verification round-trip") are the packet's charter; gate 17 stays open after this packet —
  replay re-execution needs a `PhysicsBackend`/`PolicyRuntime` pair `es` does not link, so
  `es evidence replay` prints `SKIPPED` (same convention as `es eval run`) and only
  `--dry-run` prints the plan.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es-compile -p es --all-targets -- -D warnings
cargo test -p es-eval -p es-compile -p es
cargo xtask check-spec-refs
```

Pass/fail content, `crates/es-eval/src/evidence.rs` unit tests plus `crates/es/tests/cli.rs`:

- `evidence_sign_then_verify_is_valid_against_the_trusted_key` — sign, verify against the
  signer's key (`Valid`) and against no trusted keys (`UntrustedKey`); either way `ok()`.
- `evidence_verify_flags_a_tampered_signed_bundle_invalid` — an entry rewritten after signing
  (container re-sealed, so the container's own hash check passes) verifies `Invalid`.
- `evidence_verify_reports_wrong_trusted_key_as_untrusted` — a real signature, a `--trust` key
  that is not the signer's, reports `UntrustedKey`, not `Invalid`.
- `evidence_verify_a_schema_v1_bundle_reports_absent_signature` — a manifest forced back to
  `schema_version: 1` (what every pre-W7 bundle looks like) still opens and verifies `Absent`.
- `evidence_verify_catches_a_reindented_report` — a `report.json` re-serialized compact
  instead of `build`'s canonical sorted-key pretty form trips `EVID_NOT_CANONICAL`, even
  though it parses to the identical value.
- `evidence_keygen_sign_and_verify_trust_cli_round_trip` — `es evidence keygen` writes a
  32-byte `.eskey` and prints a public key; `sign` re-emits a signed bundle and prints the
  same key; `verify` without `--trust` reports `untrusted key`, with the matching `--trust
  pub.hex` reports `valid`; `--require-signature` exits 1 without a trusted key and 0 with one.
- `evidence_replay_dry_run_prints_the_plan_and_is_skipped_otherwise` — `es evidence replay`
  without `--dry-run` exits 3 and prints `SKIPPED`; with it, prints the `ReplayPlan` table and
  exits 0.

Plus the pre-existing M3 W5 tests (`evidence_bundle_round_trips`,
`evidence_verify_catches_a_tampered_report`, `evidence_verify_fails_an_uncovered_requirement`,
`evidence_verify_against_lists_revalidation`, `evidence_build_and_verify_cli_round_trip`),
updated for `verify`'s new `trusted_keys` parameter and the canonical-JSON writer.

## acceptance

- `EvidenceBundle::sign(bytes, &SigningKey) -> Vec<u8>` re-emits the bundle with
  `manifest.signature` = ed25519 over blake3 of every non-manifest entry's `(name, hash)` pair
  in sorted order, and `manifest.signer_public_key` set; no `rand_core`, no in-process key
  generation — `SigningKey::from_bytes` from a caller-supplied 32-byte seed only.
- `EvidenceBundle::verify(bytes, against, trusted_keys) -> VerifyReport` adds `signature:
  SignatureStatus` (`Valid([u8;32]) | Invalid | Absent | UntrustedKey([u8;32])`) and
  `replayable: Vec<ReplayPlan>`; neither affects `VerifyReport::ok()` (gate 16 stays "every
  requirement covered, no error diagnostic" — whether a signature or a replay is *required* is
  a caller policy, enforced at the CLI, not a property of the bundle).
- `report.json` canonical-form check: `verify` flags any `reports/<i>/report.json` whose bytes
  are not exactly `serde_json::to_value` -> `to_string_pretty` of its own parsed form
  (`EVID_NOT_CANONICAL`). `build`'s writer (`to_json`) now produces exactly that form, so
  nothing `build` writes ever trips its own check.
- `es evidence keygen --out key.eskey` (warns to keep it out of the repo; `*.eskey` gitignored),
  `es evidence sign --key key.eskey --in evidence.esb --out signed.esb`, `es evidence verify
  --trust pub.hex [--trust ...] [--require-signature]` (exit 1 on anything but `Valid` when
  `--require-signature` is given), `es evidence replay --dry-run` (prints the plan; without
  `--dry-run`, `SKIPPED`, exit 3).
- `BundleManifest.schema_version` 2; a 1 still opens and verifies (`Absent`), `build` always
  writes 2.
- No new trait (INV-17), `BTreeMap` only, English-only, ~600 new lines or fewer.

## forbidden

- `crates/es-script`, `crates/es-data`, `crates/es-usd`, `crates/es-ir`, `crates/es-env`,
  `crates/es-gpu`, `crates/es-physics-*` — owned by other in-flight packets.
- The container format itself (`bundle::read`/`write`, the `.esb` byte layout) and
  `es_eval::runner` (reports are consumed as `es eval run`/`write_artifacts` already produce
  them) — only the manifest's signature slots and the evidence-bundle JSON writer's
  canonicalization change.
- Replay re-execution: still M4 work after this packet closes gate 17's "verification"
  half; the "rerun and compare" half needs a `PhysicsBackend`/`PolicyRuntime` pair this
  crate does not link.
- A PKI, key rotation, or revocation list: `--trust` is a flat list of keys the caller already
  decided to trust, by design (see `docs/design/safety-case.md` §6).
