# P-M1-R4 — bundle container hardening (`es-compile::bundle`)

M1 review follow-up (`docs/reviews/M1.md`, Should-fix): `read` silently ignored trailing bytes
after the last payload, falsifying the "the layout for a given set of entries is unique" claim,
and `PolicyBundle::open` let a manifest that omits its hash slots open with no spec 5.3 check on
that slot at all. Spec: spec 5.3 (hash chain), spec 9.6 (deployment bundle), spec 25.1
(untrusted input crosses a trust boundary), spec 25.3 (container versioning). Design note:
`docs/design/policy-bundle.md`.

## context

```
crates/es-compile/src/bundle.rs
docs/design/policy-bundle.md
docs/packets/M1/P-M1-R4.md
```

Tests live inside `bundle.rs`'s own `#[cfg(test)] mod tests` — no new test file.

## spec

- **Trailing bytes are rejected.** `read` tracks its cursor position through every payload; if
  it does not land exactly on `bytes.len()` once the last payload is consumed, `read` returns
  `BundleError::TrailingBytes { extra }` instead of silently accepting (and discarding) the
  extra bytes.
- **Required hash slots.** `PolicyBundle::open` checks the manifest's `task`, `observation`,
  `learning`, `deployment` and `compiler` hash slots are all `Some` *before* trusting anything
  else in the container; any one of them being `None` is `BundleError::MissingHash { slot }`.
  `policy`, `runtime` and `dataset` stay optional exactly as `docs/design/policy-bundle.md`
  already documents (a `Policy` bundle never fills `runtime`/`dataset`, and `policy` is
  redundant with the direct `weights` blake3 check `open` already does).
- **Explicit checked arithmetic, plus a cap.** Every length/offset in `read` already went
  through `Cursor::take`'s `checked_add` (spec 25.1's hostile-input-safe reader); this packet
  adds `MAX_ENTRIES` / `MAX_NAME_LEN` ceilings (4096 each — a real bundle has a handful of
  entries with short names) checked *before* the count or a name length is used for anything,
  so an adversarial header claiming a huge entry count or name length fails fast with a
  `BundleError` rather than looping or allocating on the strength of untrusted input. `write`'s
  two `u32::try_from(..).unwrap_or(u32::MAX)` calls (silently wrong output on overflow, flagged
  as a Nit in the review) become `?` against the same two error variants, so an oversized
  entry count/name is an error on the write side too, not a corrupted count.

## oracle

```
cargo fmt -p es-compile --check
cargo clippy -p es-compile --all-targets -- -D warnings
cargo test -p es-compile bundle
```

## acceptance

- A valid container with one extra trailing byte appended fails to `read` with
  `BundleError::TrailingBytes { extra: 1 }`; the byte-identical round trip test
  (`container_round_trips_and_is_deterministic`) is unaffected.
- A manifest built with `BundleHashes::default()` (every slot `None`) fails `PolicyBundle::open`
  with `BundleError::MissingHash { slot: "task" }` — the first required slot checked — without
  ever needing valid IR TOML in the other entries, since the check runs before they are parsed.
- A header claiming a payload `len` of `u32::MAX` with no payload bytes behind it, a `name_len`
  past the end of the buffer, and a declared entry count of one billion each return a
  `BundleError` (`Truncated`, `Truncated`, `TooManyEntries` respectively); each call is wrapped
  in `std::panic::catch_unwind` in its test to prove no panic occurs.
- Every existing `bundle::tests` case still passes unchanged.

## forbidden

- `crates/es-compile/src/budget.rs`, `crates/es-compile/src/lib.rs`,
  `crates/es-compile/src/kernels.rs`, `crates/es-compile/tests/` — other packets' scope.
- `docs/design/telemetry-protocol.md`, `crates/es-telemetry/*` — P-M1-R5's scope.
- Any new external dependency, any new trait (INV-17), and adding `policy`/`runtime`/`dataset`
  to the required-slot list — the review's fix names five slots, not the full `BundleHashes`.
- Committing.
