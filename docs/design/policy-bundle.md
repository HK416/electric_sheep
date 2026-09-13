# `policy.esb` — the deployment bundle format

Design note for `es-compile::bundle` and `es-runtime-embedded`. Spec: spec 9.5, spec 9.6,
spec 10.5, spec 5.3, spec 25.1, spec 25.3, spec 27.1.

## 1. Why a bundle at all

Spec 9.6: **"the policy weights are not deployed alone — preprocessing and the safety
constraints ship with them."** A checkpoint on its own is not a deployable artifact: the same
weights fed differently-resized pixels are a different policy, and the same policy under a
different envelope is a different machine as far as spec 27.1 is concerned. So the deployment
artifact is one file carrying the four IRs that decide what the robot does, the checkpoint, and
every hash of spec 5.3 needed to recompute `execution_hash`.

The claim spec 9.5 wants to be able to make is: *the same `deployment_hash` means the same safe
behaviour*. That only holds if the thing which enforces safety and the thing which preprocesses
observations are shipped together with the weights and are verified together at load.

## 2. Container

`.esb` is a tiny tar-like container. No archive crate and no compression:

- the format is a few dozen lines of little-endian integers, so a dependency buys nothing;
- safetensors payloads do not compress;
- an inflate path is attack surface on an artifact that crosses a trust boundary (spec 25.1),
  and decompression bombs are a class of bug this file simply does not have.

```
offset  size          field
0       4             magic, ASCII "ESB1"
4       4             entry_count : u32

then entry_count headers, sorted by name, ascending, byte-wise:
        4             name_len : u32
        name_len      name     : UTF-8, no interior NUL, no path normalization
        8             len      : u64, payload length
        32            hash     : blake3 of the payload

then the payloads, concatenated, in the same order.
```

Every integer is little-endian. There is no per-entry offset field: the payloads are in header
order and `len` determines each one, so the layout for a given set of entries is unique.

**Determinism.** Entries come from a `BTreeMap`, so the order is the name order; nothing else in
the writer varies. Two builds of the same inputs are byte-identical, which the test
`policy_bundle_round_trips_and_is_byte_identical` asserts. This is why no build timestamp is
written by default: `created_utc` exists in the manifest but `PolicyBundle::build` leaves it
empty.

**Reading is hostile-input safe.** Every length comes from the file, so the reader is a cursor
that returns `Truncated` rather than slicing out of bounds; names must be UTF-8; a duplicate or
out-of-order name is `Unsorted`; and every payload's `blake3` is checked before it is handed
back. A single flipped byte fails to open. `entry_count` and each `name_len` are checked against
`MAX_ENTRIES` / `MAX_NAME_LEN` (4096 each) before either is used for anything — a header
claiming a billion entries or a multi-gigabyte name is `TooManyEntries` / `NameTooLong`, not a
long loop or a large allocation on the strength of untrusted input — and every arithmetic step
on a length or offset (`Cursor::take`'s `checked_add`, the `u32`/`u64` conversions in both `read`
and `write`) is a checked op that returns a `BundleError` on failure rather than panicking.
Bytes left over after the last payload are `TrailingBytes { extra }`: the "the layout for a
given set of entries is unique" claim above only holds if extra bytes are rejected, not quietly
ignored.

The `1` in `ESB1` is the *container generation*, not the manifest's `schema_version`. A future
incompatible layout gets a new magic so that an old reader fails loudly instead of
misinterpreting (spec 25.3: the bundle format is versioned and old versions stay readable).

## 3. Entries

| name | required | content |
|---|---|---|
| `manifest.toml` | yes | `BundleManifest`, below |
| `task.toml` | policy | Task IR (spec 6), `es_ir::serial` envelope |
| `observation.toml` | policy | Observation IR (spec 7) |
| `learning.toml` | policy | Learning IR (spec 8) |
| `deployment.toml` | policy | Deployment IR + Safety Plane config (spec 9) |
| `weights.safetensors` | policy | checkpoint bytes, verbatim |
| `evaluation.toml` | no | Evaluation IR (spec 10), when the bundle also carries a suite |

The reader knows none of these names; `read` returns every entry and `PolicyBundle::open`
asks for the ones it needs. That is what lets the **evidence bundle of spec 27.1 be the same
container**: `kind = Evidence` plus `safety_case/*.json`, `validation/*.json`, `training/`,
`scene/` and a filled `dataset` hash. No format change, no second reader.

## 4. Manifest

```toml
schema_version = 1
kind = "policy"

[hashes]
task        = "<64 hex>"
observation = "<64 hex>"
learning    = "<64 hex>"
policy      = "<64 hex>"
deployment  = "<64 hex>"
compiler    = "<64 hex>"
# runtime, dataset: absent in a deployment bundle
```

Hashes are hex strings rather than TOML integer arrays because the manifest is the one part of a
bundle a human reads.

Every slot is `Option`, and an absent slot means **"not claimed"**, never "claimed as zero". The
two kinds fill different subsets:

- `runtime` is absent in a `policy.esb`: the inference backend is chosen where the bundle is
  opened, so the artifact cannot know it. `es-runtime-embedded` supplies it from the loaded
  `PolicyRuntime::runtime_hash()`.
- `dataset` is absent in a `policy.esb`: training is not on the deployment path. An evidence
  bundle fills it, and only then does `execution_hash` match the training-time one.
- `hardware_capability` is not a manifest field at all. It describes the machine that runs, not
  the artifact.

`signature` is a reserved `Option<Vec<u8>>` slot (spec 25.1 lists optional signature
verification). **Nothing signs and nothing verifies today**, so a reader must not treat `Some`
as trust. When signing arrives it covers the container bytes with the signature field empty.

## 5. `PolicyBundle::build` / `open`

`build` validates each IR, runs `es_ir::cross::check`, compiles the observation plan for the
`compiler` slot, checks the checkpoint against `WeightsRef::hash`, and fills the six hashes it
can compute.

`open` **does the same work again and trusts nothing in the manifest**. The manifest is the
artifact's claim; the IRs are the evidence. A slot whose recomputed value disagrees is
`BundleError::HashMismatch { slot }` naming the slot, which is the unit spec 27.1's
`revalidation_trigger` is written in.

Before any of that, `open` checks that `task`, `observation`, `learning`, `deployment` and
`compiler` are all present (`Some`) in the manifest's `hashes` — `BundleError::MissingHash
{ slot }` naming the first absent one otherwise. A manifest that leaves one of these `None`
opened with no spec 5.3 check on that slot at all before this check existed, which is a bigger
gap than a wrong hash: `want.is_some() && want != got` treats "not claimed" and "claimed
correctly" the same way. `policy`, `runtime` and `dataset` stay optional — `policy` is redundant
with the direct `weights` blake3 check `open` already does, and `runtime`/`dataset` are absent
from every `Policy` bundle by design (section 4).

### Why the compiled plan is not in the bundle

`CpuPlan` is not `Serialize` — it holds resolved buffer homes, an arena layout and live
`TemporalWindow` ring state — and serializing it would freeze the compiler's internal
representation into a deployment artifact, which spec 25.3 does not want. So the bundle stores
the Observation IR and `PolicyBundle::compile_plan` rebuilds the plan at load. The soundness
argument is the `compiler` hash: `CpuPlan::compiler_hash` covers the crate version, the plan
mode and the kernel id table, so a rebuild that would produce different numerics fails to open
before it can run. Deployment bundles are always compiled in `PlanMode::Release`
(`BUNDLE_PLAN_MODE`), and the mode is inside `compiler_hash`, so it cannot be a silent
difference.

## 6. `es-runtime-embedded`

Spec 9.6 lists the contents; the crate composes them and adds nothing:

```
compiled observation plan   es-compile   (spec 7, spec 11.3)
policy runtime              es-policy    (spec 2.4, one Box<dyn PolicyRuntime>)
Safety Plane                es-safety    (spec 9, whole)
telemetry ring              here         (see below)
```

`EmbeddedRuntime::from_bundle` is the **only** constructor and it always builds a
`SafetyPlane`; `tick` returns `SafeAction`, which only the plane can produce (`INV-12`,
`INV-13`). `SafetyPlane::from_ir` is also what rejects a bundle whose joint count is not `NJ` or
whose action horizon is not `H`.

### Replan cadence (spec 8.6)

One inference per `rate.control / rate.inference` control ticks, capped by
`action.execute_chunk` — rows past K are predictions, not commands (spec 8.5). The ratio is
computed from the two rational `TickRate`s, exactly; `XIR-023` has already checked that it
agrees with `PolicyContract::replanning_hz`, so there is no float period to accumulate
(spec 3.4). Between replans the buffered chunk is re-submitted and the plane advances its own
cursor.

### Failure is a chunk, not an error

A missing sensor tensor, a plan error, an inference error or an action tensor of the wrong shape
all produce an **empty chunk**. The plane turns that into a chunk underrun and the configured
fallback. There is no error return from `tick` because there is no control tick without an
action.

### Allocation

Everything is sized in `from_bundle`. A **reuse** tick allocates nothing — asserted with
`es_core::alloc_count::assert_no_alloc`. A **replan** tick allocates in exactly two places this
crate does not own:

1. `CpuPlan::run` takes a fresh f32 arena per call and `tick` builds a borrowed input map for
   it;
2. `PolicyRuntime::infer` is a trait boundary; implementations own their buffers.

Spec 9.6's "zero heap allocation" is therefore met by the plane, the chunk buffer and the
telemetry ring, and those two boundaries are the M2 work.

### Telemetry ring

`RingBuffer` lives in `es_core::ring` (layer 1); `es_telemetry::ring` re-exports it and
this crate uses it directly, so no ring is duplicated here (see
`docs/packets/M1/W8-telemetry-transport.md`).

 Known ceilings

- No signature, no encryption. The `signature` slot exists; nothing fills it (spec 25.1).
- No compression, deliberately (section 2).
- No streaming: `read` and `write` work on whole `Vec<u8>`. A bundle is a policy and four
  graphs; it fits in memory on the machine that runs the policy by construction.
- `dataset` is absent from a deployment bundle, so `EmbeddedRuntime::execution_hash` zeroes that
  slot and does not equal the training-time `execution_hash` until an evidence bundle supplies
  it.
- `hardware_capability` covers the target triple and the CPU feature probe only. GPU limits and
  driver versions are not visible from layer 9 (spec 9.6 excludes the renderer); a Vulkan
  `PolicyRuntime` must fold the device identity into its own `runtime_hash`.
- The Safety Plane keeps its chunk cursor when a new chunk is byte-identical to the stored one
  (`docs/design/safety-plane.md`). A policy that emits a literally constant chunk therefore runs
  its cursor out and falls back — fail-safe, but surprising; the test `FakeRuntime` carries a
  drift term for exactly this reason.
