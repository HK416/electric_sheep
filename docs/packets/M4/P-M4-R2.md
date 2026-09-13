# P-M4-R2 — sign the manifest, not just the entries

Closes review blocker **B-2** (`docs/reviews/M4.md`, "Follow-up packets" R2). Design note:
`docs/design/safety-case.md` §6 (updated by this packet).

## context

- `crates/es-eval/src/evidence.rs` (+ its unit tests)
- `crates/es/tests/cli.rs` (append only)
- `docs/design/safety-case.md` (§6 bullet on what a signature covers)
- `docs/packets/M4/P-M4-R2.md` (this file)

## spec

- §25.1 — weights and graph files cross a trust boundary; the signature is what makes a
  bundle's own claims about itself authenticated rather than merely self-consistent.
- §5.3 — `manifest.hashes` *is* the declared execution-hash chain. A signature that leaves it
  out authenticates the payload but not the statement about the payload, which is backwards
  for §28.7 gate 17.
- §25.3 — the container format is versioned. This packet changes no bytes of the format, so
  there is no `schema_version` bump: the scheme id is a domain separator inside the signed
  message.
- Appendix B.6 — `CanonWriter` is the project's canonical encoder: little-endian integers,
  length-prefixed strings and blobs. It is the encoder for the signed message.

## oracle

```
cargo fmt --check
cargo clippy -p es-eval -p es-compile -p es-data -p es --all-targets -- -D warnings
cargo test -p es-eval -p es-compile -p es
cargo xtask check-spec-refs
```

Pass/fail content:

- `cargo test -p es-eval evidence` —
  `the_signature_covers_every_manifest_field_but_the_signature`: sign a `(manifest, entries)`
  pair, then rewrite `hashes`, `kind`, `schema_version`, `created_utc` and
  `signer_public_key` in turn, each with the same entries and the same signature bytes; every
  one must report `SignatureStatus::Invalid`, and the untouched one `Valid`.
- `a_name_payload_split_shift_changes_the_digest` — entry `("ab", "c")` and entry
  `("a", "bc")` give different digests, and dropping an entry does too (the entry count is in
  the message).
- `a_v1_signature_is_invalid_under_v2` — a signature over the old `blake3(name || hash)`
  message reports `Invalid`, not `Valid`.
- `cargo test -p es --test cli evidence` —
  `evidence_verify_flags_a_rewritten_manifest_on_a_signed_bundle`: `manifest.hashes.task`
  rewritten on a real signed `evidence.esb`, container re-sealed so its own per-entry blake3
  check passes, reports `Invalid`, raises `EVID-007`, and `ok()` is false.
- The pre-existing W7 signing tests (`evidence_sign_then_verify_is_valid_against_the_trusted_key`,
  `evidence_verify_flags_a_tampered_signed_bundle_invalid`,
  `evidence_verify_reports_wrong_trusted_key_as_untrusted`,
  `evidence_verify_a_schema_v1_bundle_reports_absent_signature`,
  `evidence_keygen_sign_and_verify_trust_cli_round_trip`) pass unchanged.

## acceptance

- `signing_digest(manifest, entries)` encodes, with `CanonWriter`: the scheme id
  `"ed25519-esb-v2"`, the manifest with `signature` cleared as canonical JSON (one
  length-prefixed blob), the entry count, then each entry's length-prefixed name and its
  32-byte payload digest. Every string and blob is length-prefixed, so no two different
  `(manifest, entry set)` pairs share a message by re-splitting bytes.
- The manifest goes in as canonical JSON rather than field by field so that a field added to
  `BundleManifest` later is signed automatically. `signature` is the only excluded field;
  `hashes`, `kind`, `schema_version`, `created_utc` and `signer_public_key` are all covered.
- The scheme id lives *inside* the digest, not in a new manifest field: a v1-signed bundle
  reports `Invalid` (documented, `docs/design/safety-case.md` §6), no new `SignatureStatus`
  variant, and the `.esb` layout is untouched.
- `EvidenceBundle::sign` signs the manifest it is about to write (`schema_version` already
  bumped, `signer_public_key` already set), so `verify` re-derives the same message from what
  it reads.
- `verify` also cross-checks the bundle's *own* `manifest.hashes` against its `chain.json`
  (`EVID-007`), not only the embedded policy bundle's; `chain_vs_manifest` gained a `source`
  label plus the `runtime` and `dataset` slots that only an evidence manifest fills. That is
  the unsigned half of B-2: a rewritten manifest is caught with or without a signature.
- No new trait (INV-17), `BTreeMap` only, English-only, no new dependency.

## forbidden

- The container format (`bundle::read`/`write`, the `.esb` byte layout) and
  `BUNDLE_SCHEMA_VERSION`: nothing about the bytes on disk changes.
- A new manifest field, a new `SignatureStatus` variant, a PKI, key rotation or revocation.
- `crates/es-env`, `crates/es-ir`, `crates/es-usd`, `crates/es-script`, `crates/es-gpu`,
  `crates/es-render`, `crates/es-core`, `python/` — owned by other in-flight review packets.
- Replay re-execution (gate 17's other half) — unchanged by this packet.
