# W9 — `policy.esb` and `es-runtime-embedded`

Spec: spec 9.5, spec 9.6, spec 10.5, spec 5.3, spec 8.5, spec 8.6, spec 25.1, spec 25.3,
spec 27.1, spec 4.2. Design note: `docs/design/policy-bundle.md` (written first; the container
layout and the manifest contract are the artifact, the code is downstream of them).

The deployment half of M1: one artifact that carries preprocessing, weights and safety
constraints together (spec 9.6), and the minimal runtime that executes it with the same code
simulation runs (spec 9.5).

## context

```
docs/design/policy-bundle.md                      (new, written first)
docs/packets/M1/W9-deployment.md                  (new)
crates/es-compile/src/bundle.rs                   (new)
crates/es-compile/src/lib.rs                      (one `pub mod`, one re-export block)
crates/es-compile/Cargo.toml                      (blake3)
crates/es-runtime-embedded/Cargo.toml             (new crate)
crates/es-runtime-embedded/src/lib.rs             (new)
crates/es-runtime-embedded/src/runtime.rs         (new)
crates/es-runtime-embedded/src/ring.rs            (new)
crates/es-runtime-embedded/src/hardware.rs        (new)
crates/es-runtime-embedded/tests/embedded.rs      (new)
Cargo.toml                                        (one workspace-dependency line)
xtask/src/layering.rs                             (one LAYERS row)
```

## spec

- **`.esb` container** — `ESB1` magic, `u32` entry count, per-entry
  `{ name_len: u32, name, len: u64, blake3: [u8; 32] }` sorted by name, then the payloads in the
  same order, all little-endian. `write(&BundleManifest, &BTreeMap<String, Vec<u8>>)
  -> Vec<u8>` and `read(&[u8]) -> Result<Bundle, BundleError>`, the latter verifying every
  payload hash and refusing a duplicated or out-of-order name. No archive crate and no
  compression: see the design note section 2.
- **`BundleManifest`** — `schema_version`, `kind: Policy | Evidence`, `hashes` (the spec 5.3
  chain as `Option<[u8; 32]>` per slot, hex in TOML, plus `Option<DatasetHash>`),
  `created_utc: Option<String>`, `signature: Option<Vec<u8>>` (reserved, spec 25.1 — nothing
  signs and nothing verifies). Stored as the `manifest.toml` entry. Absent slot means "not
  claimed", never "claimed as zero".
- **`PolicyBundle::build(task, observation, learning, deployment, weights) -> Vec<u8>`** —
  validates each IR, runs `es_ir::cross::check`, checks `blake3(weights)` against
  `WeightsRef::hash`, compiles the observation plan for the `compiler` slot, fills
  `task/observation/learning/policy/deployment/compiler`. `runtime` and `dataset` stay empty.
- **`PolicyBundle::open(&[u8]) -> Result<PolicyBundle, BundleError>`** — re-parses and
  re-validates the four IRs, re-runs the cross-IR pass, recomputes every hash and fails with
  `HashMismatch { slot }` naming the slot. The manifest is a claim, not evidence.
  `compile_plan()` rebuilds the `CpuPlan` (which is not `Serialize`); the `compiler` hash is
  what makes that sound. Deployment bundles compile in `PlanMode::Release`.
- **`EmbeddedRuntime<NJ, H>`** — `from_bundle(bytes, Box<dyn PolicyRuntime>)` is the only
  constructor: opens the bundle, compiles the plan, builds `SafetyPlane::from_ir` (which is
  what enforces `NJ` and `H` against the Deployment IR), loads the weights as
  `WeightsSource::InMemory`. `tick(&sensors, now, obs_age) -> SafeAction<NJ>` runs plan ->
  `infer` -> `ActionChunk` -> `SafetyPlane::validate` and pushes a `TickRecord`. Replan cadence
  is `rate.control / rate.inference` control ticks capped by `execute_chunk` (spec 8.6); between
  replans the buffered chunk is re-submitted. Any failure becomes an empty chunk, which the
  plane turns into a chunk underrun and the configured fallback — `tick` has no error return
  (`INV-13`).
- **`execution_hash()`** — the spec 5.3 chain from the manifest plus the loaded backend's
  `runtime_hash()` and a `HardwareCapability` computed here as
  `blake3(tag || arch || os || family || pointer width || [feature, present]*)`, with the
  feature probe list fixed and documented in `hardware.rs`.
- **Layering** — `es-runtime-embedded` sits at layer 9 with `es-env`: it may use `es-safety` /
  `es-policy` (8) and `es-compile` (7); only `es-ros2` / `es-py` (11) may depend on it. It may
  **not** use `es-telemetry` (10), so the ~50-line `TelemetryRing` duplicates
  `es_telemetry::ring::RingBuffer` with a comment naming the source and the upgrade path (move
  `RingBuffer` into `es-core`).

## oracle

```
cargo fmt --check
cargo clippy -p es-compile -p es-runtime-embedded --all-targets -- -D warnings
cargo test -p es-compile -p es-runtime-embedded
cargo xtask layering
cargo xtask check-spec-refs
cargo xtask context-budget
```

## acceptance

- Container: round-trip; **byte-identical output for identical inputs**; a flipped payload byte
  is `EntryHash { name }`; a foreign magic and a truncated file are named errors; the manifest
  round-trips through TOML with hex hashes and omits `None` fields.
- `PolicyBundle`: the `es-ir` cross-IR fixture builds and re-opens with every hash matching;
  a swapped manifest hash fails as `HashMismatch { slot: "learning" }`; weights that do not
  match `WeightsRef::hash` are refused at build.
- `EmbeddedRuntime` over the fixture with a `#[cfg(test)]` `FakeRuntime`: a benign chunk emits
  `ActionSource::Policy` and a clean action; a NaN chunk emits
  `Fallback(HoldPosition)` with a finite action; a bundle for 8 joints refuses to load as
  `EmbeddedRuntime<3, H>`; the replan cadence is 10 control ticks (100 Hz / 10 Hz) and 35 ticks
  produce exactly 4 inferences; a chunk-reuse tick passes
  `es_core::alloc_count::assert_no_alloc`; `execution_hash` is stable within a process and equal
  across two loads of the same bundle.
- The fixture's inference budget is widened rather than a watchdog disabled (`INV-12`).
- The `PolicyBundle`-level tests live in `es-runtime-embedded/tests/embedded.rs` because that is
  where the cross-IR fixture copy is; `es-compile` keeps the container-level unit tests, which
  need no fixture.

## forbidden

- Any file outside `context`. In particular `crates/es-physics-backend/src/mjwarp*`,
  `crates/es-sensor`, `crates/es-actuator`, `docs/reviews/M0.md` and `docs/ARCHITECTURE*.md`,
  which belong to other agents in this wave; and `crates/es-ir`, `crates/es-safety`,
  `crates/es-policy`, whose APIs are consumed and never edited.
- An archive dependency (`zip`, `tar`, `flate2`, `zstd`). The container is ~40 lines and
  compression is attack surface on a trust boundary (spec 25.1).
- A new trait. `PolicyRuntime` is consumed as `Box<dyn PolicyRuntime>`; the seven extension
  points of `INV-17` are the whole list.
- Any path that constructs an `EmbeddedRuntime` without a `SafetyPlane`, or that returns an
  action the plane has not seen (`INV-12`, `INV-13`). A test that needs room widens the
  envelope.
- A pickle path or any weight format that executes code on load (`INV-16`).
- `HashMap`/`HashSet` (spec 3.4), threads, global RNG, filesystem access in the runtime,
  Python, physics, rendering (spec 9.6's exclusion list).
- Real cryptography in the `signature` slot. It is a reserved field this milestone, and a
  half-verified signature is worse than none.
- Evidence-bundle content (`safety_case/`, `validation/`, `training/`) and `es evidence verify`
  — spec 27.1 is a later packet. This one only owes it a container that extends without a format
  change.
